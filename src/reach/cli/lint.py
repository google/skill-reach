# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Execute static pre-flight validation on skill files, frontmatter schemas, and catalogs."""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal

from cyclopts import Parameter

from reach.config import RuntimeSettings, load_config, resolve_path
from reach.discovery import resolve_corpus
from reach.lint import (
    LintIssue,
    LintReport,
    LintSettings,
    explain_rule,
    lint_file,
    lint_tree,
)
from reach.runtime import build_runtime
from reach.views import (
    Console,
    build_console,
    print_discovery,
    print_lint,
    print_rule_explanation,
    render_lint_concise,
    render_lint_github,
)

from .app import LOOP, app
from .discovery import _no_skills
from .flags import (
    LIST,
    RULES_GROUP,
    SWITCH,
    AgentName,
    ConfigFlag,
    Global,
    RuleOverrideFlags,
    agent_help_text,
    complete_lint_rules,
    complete_skill_names,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

#: Supported output formats for lint diagnostics.
type LintFormat = Literal["concise", "csv", "github", "json", "jsonl", "text"]


def _format_csv(report: LintReport) -> str:
    """Format lint issues into CSV tabular document."""
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["severity", "rule", "skill", "path", "message", "remedy"])
    for issue in report.issues:
        writer.writerow(
            [
                issue.severity.value,
                issue.rule,
                issue.skill,
                str(issue.path or ""),
                issue.message,
                issue.remedy,
            ],
        )
    return buffer.getvalue()


def _lint_roots(
    roots: Sequence[Path],
    lint_config: LintSettings,
) -> LintReport:
    """Aggregate lint diagnostic reports across multiple discovered directory roots."""
    aggregated_issues: list[LintIssue] = []
    skills_checked = 0
    for root in roots:
        if root.is_dir():
            sub_report = lint_tree(root, config=lint_config)
            aggregated_issues.extend(sub_report.issues)
            skills_checked += sub_report.skills_checked
    return LintReport(issues=tuple(aggregated_issues), skills_checked=skills_checked)


def _resolve_lint_report(
    skills: Path | None,
    agent: AgentName | None,
    lint_config: LintSettings,
    console: Console,
    config_path: Path | str | None = None,
    global_scope: bool = False,
) -> LintReport:
    """Execute lint inspection across direct paths or discovered runtime skill roots."""
    target_path = resolve_path(skills) if skills is not None else None
    if target_path is not None and target_path.is_file():
        return lint_file(target_path, config=lint_config)
    if target_path is not None and target_path.is_dir():
        return lint_tree(target_path, config=lint_config)

    driver = build_runtime(RuntimeSettings(agent=agent) if agent is not None else RuntimeSettings())

    _found, roots, discovered = resolve_corpus(
        driver,
        Path.cwd(),
        skills,
        config_path=config_path,
        agent=agent,
        global_scope=global_scope,
    )
    if discovered is not None:
        print_discovery(console, discovered)
    if not roots:
        raise _no_skills(skills, global_scope=global_scope)

    return _lint_roots(roots, lint_config)


def _render_lint_output(
    console: Console,
    report: LintReport,
    format: LintFormat,
) -> None:
    """Render diagnostic report in the requested serialization format."""
    match format:
        case "json":
            print(report.model_dump_json(indent=2))
        case "jsonl":
            for issue in report.issues:
                print(issue.model_dump_json())
        case "csv":
            print(_format_csv(report), end="")
        case "concise":
            if concise := render_lint_concise(report):
                print(concise)
        case "github":
            if github := render_lint_github(report):
                print(github)
        case _:
            print_lint(console, report)


@app.command(name="lint", group=LOOP)
def _lint(
    skills: Annotated[
        Path | None,
        Parameter(
            help=(
                "Path to a skill directory, SKILL.md file, or catalog tree "
                "(discovered from precedence if omitted)"
            ),
        ),
    ] = None,
    *,
    skill: Annotated[
        tuple[str, ...],
        LIST,
        Parameter(
            completer=complete_skill_names,
            help="Filter lint diagnostics to these specific skill names (repeatable)",
        ),
    ] = (),
    strict: Annotated[
        bool,
        SWITCH,
        Parameter(
            help="Fail with exit code 1 if any warnings are detected",
        ),
    ] = False,
    explain: Annotated[
        str | None,
        Parameter(
            metavar="RULE",
            completer=complete_lint_rules,
            help="Display detailed explanation and remedy for a specific lint rule and exit",
        ),
    ] = None,
    rules: Annotated[RuleOverrideFlags | None, Parameter(group=RULES_GROUP)] = None,
    agent: Annotated[
        AgentName | None,
        Parameter(
            help=agent_help_text("Agent runtime to query for installed skill locations"),
        ),
    ] = None,
    global_: Global = False,
    format: Annotated[
        LintFormat,
        Parameter(help="Output format for diagnostic reporting"),
    ] = "text",
    config: ConfigFlag = None,
) -> int:
    """Validate skill manifests, frontmatter schemas, naming, and runtime limits."""
    console = build_console()

    if explain is not None:
        rule_def = explain_rule(explain)
        if rule_def is None:
            console.print(f"[bold red]error:[/] unknown lint rule {explain!r}")
            return 2
        print_rule_explanation(console, rule_def)
        return 0

    overrides = rules.to_overrides() if rules is not None else None
    settings = load_config(config)
    lint_config = LintSettings.from_settings(settings, overrides)

    report = _resolve_lint_report(
        skills,
        agent,
        lint_config,
        console,
        config_path=config,
        global_scope=global_,
    )

    if skill:
        allowed = frozenset(skill)
        filtered_issues = tuple(i for i in report.issues if i.skill in allowed)
        report = LintReport(issues=filtered_issues, skills_checked=report.skills_checked)

    _render_lint_output(console, report, format)

    if report.has_errors or (strict and report.warnings):
        return 1
    return 0
