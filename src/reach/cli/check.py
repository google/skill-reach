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

"""Verify skills with two-stage CI/CD quality gate and GitHub Actions reporting."""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal

if TYPE_CHECKING:
    from reach.check import CheckOutcome


from cyclopts import Parameter

from reach.check import CheckStage, run_check
from reach.config import CheckSettings, RegistrySettings, RunConfig, resolve_path
from reach.views import (
    Console,
    build_console,
    emit_check_github_annotations,
    is_github_actions,
    print_check,
    print_registry_audit,
    render_check_github_summary,
    write_github_step_summary,
)

from .app import LOOP, app
from .flags import (
    LIST,
    NON_NEGATIVE,
    POSITIVE_INT,
    RATE,
    REGISTRY_GROUP,
    RULES_GROUP,
    SWITCH,
    THRESHOLDS_GROUP,
    AgentName,
    ConfigFlag,
    Global,
    Quiet,
    RegistryFlags,
    RuleOverrideFlags,
    YesFlag,
    agent_help_text,
)

type CheckFormat = Literal["auto", "concise", "github", "json", "text"]


def _render_check_output(
    console: Console,
    outcome: CheckOutcome,
    format: CheckFormat,
    step_summary: bool,
) -> None:
    """Render check outcomes according to active format and write GitHub step summaries."""
    active_format = format
    if active_format == "auto":
        active_format = "github" if is_github_actions() else "text"

    if step_summary and is_github_actions():
        summary_md = render_check_github_summary(outcome)
        write_github_step_summary(summary_md)

    match active_format:
        case "json":
            print(outcome.model_dump_json(indent=2))
        case "concise":
            render_check_concise(console, outcome)
        case "github":
            if annotations := emit_check_github_annotations(outcome):
                print(annotations)
            render_check_concise(console, outcome)
        case "text":
            print_check(console, outcome)


def render_check_concise(console: Console, outcome: CheckOutcome) -> None:
    """Render concise check summary suitable for CI step headers."""
    if outcome.stage_failed is CheckStage.STATIC or outcome.lint_report.has_errors:
        lint_status = (
            f"[red]FAILED ({len(outcome.lint_report.errors)} errors)[/]"
            if outcome.lint_report.has_errors
            else f"[red]FAILED (strict: {len(outcome.lint_report.warnings)} warnings)[/]"
        )
    elif outcome.lint_report.warnings:
        lint_status = f"[yellow]PASSED with {len(outcome.lint_report.warnings)} warnings[/]"
    else:
        lint_status = "[green]PASSED[/]"
    console.print(f"Static lint: {lint_status}")
    if outcome.assertions:
        failed = [a for a in outcome.assertions if not a.passed]
        if failed:
            failed_str = ", ".join(f"{a.name} ({a.observed})" for a in failed)
            console.print(f"Empirical assertions: [red]FAILED ({failed_str})[/]")
        else:
            console.print("Empirical assertions: [green]PASSED[/]")


@app.command(name="check", group=LOOP)
def _check(
    skills: Annotated[
        Path | None,
        Parameter(
            name=["skills", "--skills"],
            help="Directory path or manifest file of skills to inspect",
        ),
    ] = None,
    *,
    queries: Annotated[
        Path | None,
        Parameter(
            name="--queries",
            help="Path to labeled evaluation queries JSON file for empirical validation",
        ),
    ] = None,
    changed: Annotated[
        bool,
        SWITCH,
        Parameter(
            help="Inspect only skills modified since reference git commit",
        ),
    ] = False,
    since: Annotated[
        str | None,
        Parameter(
            help="Git reference specification for changed-skill discovery (default: HEAD~1)",
        ),
    ] = None,
    strict: Annotated[
        bool | None,
        Parameter(
            negative="--no-strict",
            show_default=False,
            help="Fail with exit code 1 if static lint warnings are detected",
        ),
    ] = None,
    min_recall: Annotated[
        float | None,
        RATE,
        Parameter(
            name="--min-recall",
            group=THRESHOLDS_GROUP,
            help="Minimum acceptable target recall threshold (default: 0.80)",
        ),
    ] = None,
    min_accuracy: Annotated[
        float | None,
        RATE,
        Parameter(
            name="--min-accuracy",
            group=THRESHOLDS_GROUP,
            help="Minimum acceptable classification accuracy threshold (default: 0.80)",
        ),
    ] = None,
    max_misroute: Annotated[
        float | None,
        RATE,
        Parameter(
            name="--max-misroute",
            group=THRESHOLDS_GROUP,
            help="Maximum acceptable misroute rate threshold (default: 0.10)",
        ),
    ] = None,
    min_entrypoint: Annotated[
        float | None,
        RATE,
        Parameter(
            name="--min-entrypoint",
            group=THRESHOLDS_GROUP,
            help="Minimum acceptable entrypoint accuracy threshold (0.0 - 1.0)",
        ),
    ] = None,
    min_reachability: Annotated[
        float | None,
        RATE,
        Parameter(
            name="--min-reachability",
            group=THRESHOLDS_GROUP,
            help="Minimum acceptable trajectory reachability threshold (0.0 - 1.0)",
        ),
    ] = None,
    min_efficiency: Annotated[
        float | None,
        RATE,
        Parameter(
            name="--min-efficiency",
            group=THRESHOLDS_GROUP,
            help="Minimum acceptable step efficiency MRR threshold (0.0 - 1.0)",
        ),
    ] = None,
    min_f1: Annotated[
        float | None,
        RATE,
        Parameter(
            name="--min-f1",
            group=THRESHOLDS_GROUP,
            help="Minimum acceptable skill selection F1 threshold (0.0 - 1.0)",
        ),
    ] = None,
    max_redundancy: Annotated[
        float | None,
        NON_NEGATIVE,
        Parameter(
            name="--max-redundancy",
            group=THRESHOLDS_GROUP,
            help="Maximum acceptable skill redundancy threshold (excess invocations)",
        ),
    ] = None,
    budget: Annotated[
        int | None,
        POSITIVE_INT,
        Parameter(
            name="--budget",
            help="Maximum empirical probes permitted (default: 50)",
        ),
    ] = None,
    agent: Annotated[
        AgentName | None,
        Parameter(
            name="--agent",
            show_choices=False,
            help=agent_help_text("Agent runtime for empirical probing (default: from reach.toml)"),
        ),
    ] = None,
    format: Annotated[
        CheckFormat,
        Parameter(
            name="--format",
            help="Output format: auto, github, text, json, concise (default: auto)",
        ),
    ] = "auto",
    step_summary: Annotated[
        bool,
        Parameter(
            show_default=False,
            help="Write GFM Markdown scorecard to $GITHUB_STEP_SUMMARY when available",
        ),
    ] = True,
    global_: Global = False,
    filter_skill: Annotated[
        tuple[str, ...] | None,
        LIST,
        Parameter(
            name=["--filter-skill"],
            help="Filter check queries to those expecting specified skills; repeatable",
        ),
    ] = None,
    filter_id: Annotated[
        tuple[str, ...] | None,
        LIST,
        Parameter(
            name=["--filter-id"],
            help="Filter check queries to specific query identifiers; repeatable",
        ),
    ] = None,
    rules: Annotated[RuleOverrideFlags | None, Parameter(group=RULES_GROUP)] = None,
    registry: Annotated[RegistryFlags | None, Parameter(group=REGISTRY_GROUP)] = None,
    yes: YesFlag = False,
    config: ConfigFlag = None,
    quiet: Quiet = False,
) -> int:
    """Execute two-stage CI/CD quality gate combining linting and empirical assertions."""
    console = build_console(quiet=quiet)

    run_config = None
    if config is not None:
        run_config = RunConfig.from_toml(config)

    eff_registry = RunConfig.resolve(
        RegistrySettings,
        run_config,
        **(registry.overrides() if registry is not None else {}),
    )

    if eff_registry.project and (
        (registry is not None and (registry.registry or registry.project is not None))
        or (run_config is not None and run_config.registry.project is not None)
    ):
        from reach.catalog import load_registry_skills, load_skills

        remote_skills = load_registry_skills(
            project=eff_registry.project,
            location=eff_registry.location,
            publisher=eff_registry.publisher,
            fresh=eff_registry.fresh,
            no_cache=eff_registry.no_cache,
            cache_ttl_seconds=eff_registry.cache_ttl_seconds,
        )
        local_skills = []
        target_root = resolve_path(skills) if skills else Path.cwd()
        if target_root.is_dir():
            with contextlib.suppress(Exception):
                local_skills = load_skills(target_root)

        print_registry_audit(
            console,
            local_skills,
            remote_skills,
            eff_registry.project,
            eff_registry.location,
        )
        if queries is None:
            return 0

    eff_settings = RunConfig.resolve(
        CheckSettings,
        config=run_config,
        strict=strict,
        min_recall=min_recall,
        min_accuracy=min_accuracy,
        max_misroute=max_misroute,
        min_entrypoint=min_entrypoint,
        min_reachability=min_reachability,
        min_efficiency=min_efficiency,
        min_f1=min_f1,
        max_redundancy=max_redundancy,
        budget=budget,
        since=since,
    )
    if run_config is not None:
        run_config = run_config.model_copy(update={"check": eff_settings})
    rule_overrides = rules.to_overrides() if rules is not None else None

    from .safety import confirm_skill_execution

    trusted = run_config.study.trusted if run_config is not None else False

    outcome = run_check(
        skills_paths=[skills] if skills is not None else None,
        queries_path=queries,
        changed=changed,
        since=eff_settings.since,
        strict=eff_settings.strict,
        min_recall=eff_settings.min_recall,
        min_accuracy=eff_settings.min_accuracy,
        max_misroute=eff_settings.max_misroute,
        min_entrypoint=eff_settings.min_entrypoint,
        min_reachability=eff_settings.min_reachability,
        min_efficiency=eff_settings.min_efficiency,
        min_f1=eff_settings.min_f1,
        max_redundancy=eff_settings.max_redundancy,
        budget=eff_settings.budget,
        settings=eff_settings,
        agent=agent,
        config=run_config,
        rule_overrides=rule_overrides or None,
        global_scope=global_,
        filter_skill=filter_skill,
        filter_id=filter_id,
        confirm_callback=lambda rt_name, loaded, roots: confirm_skill_execution(
            console,
            runtime_name=rt_name,
            skills=loaded,
            roots=roots,
            action="check empirical probes",
            yes=yes,
            trusted=trusted,
        ),
    )

    _render_check_output(console, outcome, format, step_summary)
    return outcome.exit_code
