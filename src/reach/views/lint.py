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

"""Render static lint diagnostics, concise line formats, and rule explanation panels."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich import box
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from reach.rendering import format_github_annotation

if TYPE_CHECKING:
    from rich.console import Console

    from reach.lint import LintReport, RuleDefinition, Severity


def _print_lint_clean(console: Console, report: LintReport) -> None:
    """Print success notification and guidance when no lint issues are detected."""
    suffix = "s" if report.skills_checked != 1 else ""
    console.print(
        Text.assemble(
            ("✓ ", "reach.hit"),
            ("0 errors, 0 warnings", "bold"),
            (f" across {report.skills_checked} skill{suffix}.", "default"),
        ),
    )
    console.print(
        Text.assemble(
            ("Tip: ", "reach.label"),
            ("To inspect vocabulary competition between skills, run: ", "dim"),
            ("reach overlap", "reach.catalog"),
        ),
    )


def _severity_style(severity: Severity) -> str:
    """Return canonical Rich style string for a lint diagnostic severity."""
    from reach.lint import Severity

    if severity == Severity.ERROR:
        return "reach.error"
    if severity == Severity.INFO:
        return "reach.label"
    return "reach.misroute"


def _render_lint_table(report: LintReport) -> Table:
    """Construct Rich Table displaying lint issues with styled messages and remedies."""
    table = Table(
        box=box.SIMPLE,
        pad_edge=False,
        show_header=True,
        header_style="reach.label",
    )
    table.add_column("Severity", justify="left", no_wrap=True)
    table.add_column("Rule", style="bold", no_wrap=True)
    table.add_column("Skill / Location", style="reach.catalog", overflow="fold")
    table.add_column("Message", overflow="fold")

    for issue in report.issues:
        sev_style = _severity_style(issue.severity)
        msg_text = Text(issue.message)
        if issue.remedy:
            msg_text.append(f"\nFix: {issue.remedy}", style="dim")
        table.add_row(
            Text(issue.severity.upper(), style=sev_style),
            Text(issue.rule),
            Text(issue.skill),
            msg_text,
        )
    return table


def _render_lint_summary(report: LintReport) -> Text:
    """Assemble colored summary counts of errors, warnings, and info diagnostics."""
    err_count = len(report.errors)
    warn_count = len(report.warnings)
    info_count = len(report.infos)
    skill_suffix = "s" if report.skills_checked != 1 else ""
    parts: list[tuple[str, str]] = [
        (
            f"Found {err_count} error{'s' if err_count != 1 else ''}",
            "reach.error" if err_count else "default",
        ),
        (", " if info_count else " and ", "default"),
        (
            f"{warn_count} warning{'s' if warn_count != 1 else ''}",
            "reach.misroute" if warn_count else "default",
        ),
    ]
    if info_count:
        parts.extend(
            [
                (", and ", "default"),
                (
                    f"{info_count} info",
                    "reach.label",
                ),
            ]
        )
    parts.append((f" across {report.skills_checked} skill{skill_suffix}.", "default"))
    return Text.assemble(*parts)


def print_lint(console: Console, report: LintReport) -> None:
    """Render static lint diagnostics and summary counts to the console."""
    if report.clean:
        _print_lint_clean(console, report)
        return

    console.print(_render_lint_table(report))
    console.print(_render_lint_summary(report))


def render_lint_concise(report: LintReport) -> str:
    """Render lint issues in single-line format suitable for unix piping and parsing."""
    lines = []
    for issue in report.issues:
        loc = str(issue.path) if issue.path else issue.skill
        lines.append(f"{loc}: [{issue.rule}] ({issue.severity}) {issue.message}")
    return "\n".join(lines)


def render_lint_github(report: LintReport) -> str:
    """Render lint issues as GitHub Actions workflow command annotations."""
    return "\n".join(
        format_github_annotation(
            severity=issue.severity.value,
            message=issue.message,
            title=issue.rule,
            file=str(issue.path) if issue.path else None,
            line=issue.line,
        )
        for issue in report.issues
    )


def print_rule_explanation(console: Console, rule: RuleDefinition) -> None:
    """Render detailed documentation and remedy advice for a single lint rule."""
    sev_style = _severity_style(rule.default_severity)
    content = Text.assemble(
        ("Default Severity: ", "reach.label"),
        (rule.default_severity.upper(), sev_style),
        ("\n\nSummary:\n", "reach.label"),
        (f"  {rule.summary}\n\n", "default"),
        ("Why is this bad?\n", "reach.label"),
        (f"  {rule.explanation}\n\n", "default"),
        ("Remedy:\n", "reach.label"),
        (f"  {rule.remedy}", "reach.catalog"),
    )
    panel = Panel(
        content,
        title=f"[bold cyan]reach lint --explain {rule.rule}[/]",
        border_style="reach.help.border",
        box=box.ROUNDED,
    )
    console.print(panel)
