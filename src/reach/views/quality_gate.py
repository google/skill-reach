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

"""Render quality gate assertions, pre-flight checks, and CI step summaries."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Final

from rich import box
from rich.table import Table

from reach.rendering import format_github_annotation
from reach.views.lint import print_lint

if TYPE_CHECKING:
    from collections.abc import Sequence

    from rich.console import Console

    from reach.check import CheckAssertion, CheckOutcome
    from reach.models import Skill


def is_github_actions() -> bool:
    """Return True if executing within a GitHub Actions CI environment."""
    return os.environ.get("GITHUB_ACTIONS") == "true"


def write_github_step_summary(markdown_content: str) -> bool:
    """Append Markdown report to $GITHUB_STEP_SUMMARY file if in CI."""
    if summary_file := os.environ.get("GITHUB_STEP_SUMMARY"):
        try:
            with Path(summary_file).open("a", encoding="utf-8") as f:
                f.write(f"\n{markdown_content}\n")
        except OSError:
            return False
        else:
            return True
    return False


PERCENTAGE_ASSERTION_NAMES: Final[frozenset[str]] = frozenset(
    {
        "recall",
        "entrypoint",
        "reachability",
        "skill_f1",
    }
)


def _format_assertion_strings(a: CheckAssertion) -> tuple[str, str]:
    """Format observed and target values for metric assertions."""
    is_pct = "rate" in a.name or "accuracy" in a.name or a.name in PERCENTAGE_ASSERTION_NAMES
    obs_str = "n/a" if a.observed is None else f"{a.observed:.1%}" if is_pct else str(a.observed)
    tgt_str = f"{a.comparison} {a.threshold:.1%}" if is_pct else f"{a.comparison} {a.threshold}"
    return obs_str, tgt_str


def _render_check_stage2_summary_lines(outcome: CheckOutcome) -> list[str]:
    """Format Stage 2 Markdown summary rows for GitHub Step Summary."""
    lines = [
        "### Stage 2: Empirical Quality Gate",
        f"- **Queries Probed**: {outcome.queries_probed}",
        f"- **Probes Executed**: {outcome.probes_executed} / {outcome.budget} budget",
        "",
        "| Metric | Observed | Target | Status |",
        "| :--- | :--- | :--- | :--- |",
    ]
    for a in outcome.assertions:
        status = "✅ Passed" if a.passed else "❌ Failed"
        obs_str, tgt_str = _format_assertion_strings(a)
        lines.append(f"| {a.name} | {obs_str} | {tgt_str} | {status} |")
    lines.append("")
    return lines


def render_check_github_summary(outcome: CheckOutcome) -> str:
    """Render check outcome as a GitHub Step Summary Markdown document."""
    from reach.check import CheckStage

    status_emoji = "✅ PASSED" if outcome.passed else "❌ FAILED"
    s1_passed = outcome.stage_failed is not CheckStage.STATIC
    lines = [
        f"# Reach Quality Gate: {status_emoji}\n",
        "### Stage 1: Static Pre-flight",
        f"- **Skills Checked**: {outcome.skills_checked}",
        f"- **Errors**: {len(outcome.lint_report.errors)}",
        f"- **Warnings**: {len(outcome.lint_report.warnings)}",
        f"- **Status**: {'Passed' if s1_passed else 'Failed'}\n",
    ]

    if outcome.assertions or outcome.queries_probed > 0:
        lines.extend(_render_check_stage2_summary_lines(outcome))

    return "\n".join(lines)


def emit_check_github_annotations(outcome: CheckOutcome) -> str:
    """Format static lint defects and empirical regressions as GitHub workflow commands."""
    lines = [
        format_github_annotation(
            severity=issue.severity.value,
            message=issue.message,
            title=issue.rule,
            file=str(issue.path) if issue.path else None,
            line=issue.line,
        )
        for issue in outcome.lint_report.issues
    ]
    lines.extend(
        format_github_annotation(
            severity="error",
            message=a.message,
            title=f"regression-{a.name}",
        )
        for a in outcome.assertions
        if not a.passed
    )
    return "\n".join(lines)


def _print_check_stage1(console: Console, outcome: CheckOutcome) -> None:
    """Render Stage 1 static pre-flight check output."""
    from reach.check import CheckStage

    s1_passed = outcome.stage_failed is not CheckStage.STATIC
    s1_status = "[green]✓ PASS[/]" if s1_passed else "[red]✗ FAIL[/]"
    console.print(f"\n[bold]Stage 1 (Static Pre-flight):[/] {s1_status}")
    console.print(
        f"  • {outcome.skills_checked} skill(s) inspected, "
        f"{len(outcome.lint_report.errors)} error(s), "
        f"{len(outcome.lint_report.warnings)} warning(s).",
    )
    if outcome.lint_report.issues:
        print_lint(console, outcome.lint_report)


def _print_check_stage2(console: Console, outcome: CheckOutcome) -> None:
    """Render Stage 2 empirical evaluation assertion table."""
    s2_status = "[green]✓ PASS[/]" if outcome.passed else "[red]✗ FAIL[/]"
    console.print(f"\n[bold]Stage 2 (Empirical Quality Gate):[/] {s2_status}")
    console.print(
        f"  • {outcome.queries_probed} query(ies) evaluated across "
        f"{outcome.probes_executed} probe(s) (budget: {outcome.budget}).",
    )

    table = Table(box=box.ROUNDED, show_header=True, header_style="bold")
    table.add_column("Metric", style="cyan")
    table.add_column("Observed", justify="right")
    table.add_column("Target", justify="right")
    table.add_column("Status", justify="center")

    for a in outcome.assertions:
        obs_str, tgt_str = _format_assertion_strings(a)
        st = "[green]PASS[/]" if a.passed else "[red]FAIL[/]"
        table.add_row(a.name, obs_str, tgt_str, st)

    console.print(table)


def print_check(console: Console, outcome: CheckOutcome) -> None:
    """Render check outcome with Stage 1 and Stage 2 summary tables."""
    from reach.check import CheckStage

    title_style = "bold green" if outcome.passed else "bold red"
    status_text = "PASSED" if outcome.passed else "FAILED"
    console.print()
    console.rule(f"[{title_style}]Reach Quality Gate: {status_text}[/]")

    _print_check_stage1(console, outcome)

    if outcome.stage_failed is not CheckStage.STATIC and (
        outcome.assertions or outcome.queries_probed > 0
    ):
        _print_check_stage2(console, outcome)

    console.print()


def print_registry_audit(
    console: Console,
    local_skills: Sequence[Skill],
    remote_skills: Sequence[Skill],
    project: str,
    location: str,
) -> None:
    """Render static comparison between local workspace skills and Agent Registry skills."""
    console.print()
    console.rule(f"[bold cyan]Agent Registry Audit: {project} ({location})[/]")

    local_map = {s.name: s for s in local_skills}
    remote_map = {s.name: s for s in remote_skills}
    all_names = sorted(set(local_map.keys()) | set(remote_map.keys()))

    table = Table(box=box.ROUNDED, show_header=True, header_style="bold")
    table.add_column("Skill", style="bold")
    table.add_column("Local Status", justify="center")
    table.add_column("Registry Status", justify="center")
    table.add_column("Audit Finding")

    matched = 0
    modified = 0
    local_only = 0
    remote_only = 0

    for name in all_names:
        in_local = name in local_map
        in_remote = name in remote_map

        if in_local and in_remote:
            local_s = local_map[name]
            remote_s = remote_map[name]
            if local_s.description.strip() == remote_s.description.strip():
                matched += 1
                table.add_row(
                    name,
                    "[green]Resident[/]",
                    f"[green]{remote_s.metadata.get('state', 'ACTIVE')}[/]",
                    "[green]✓ In sync[/]",
                )
            else:
                modified += 1
                table.add_row(
                    name,
                    "[green]Resident[/]",
                    f"[green]{remote_s.metadata.get('state', 'ACTIVE')}[/]",
                    "[yellow]⚠ Description modified locally[/]",
                )
        elif in_local:
            local_only += 1
            table.add_row(
                name,
                "[green]Resident[/]",
                "[dim]Missing[/]",
                "[cyan]+ New local skill (unregistered)[/]",
            )
        else:
            remote_only += 1
            remote_s = remote_map[name]
            table.add_row(
                name,
                "[dim]Absent[/]",
                f"[green]{remote_s.metadata.get('state', 'ACTIVE')}[/]",
                "[dim]• Remote registry skill[/]",
            )

    console.print(table)
    console.print(
        f"\n[bold]Summary:[/] {matched} in sync, [yellow]{modified} modified[/], "
        f"[cyan]{local_only} new local[/], [dim]{remote_only} remote-only[/] "
        f"([bold]{len(all_names)} total[/]).\n",
    )
