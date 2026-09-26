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

"""Render closed-loop description optimization reports, candidate tables, and diffs."""

from __future__ import annotations

import difflib
from typing import TYPE_CHECKING, Final, Literal, overload

from rich import box
from rich.table import Table

if TYPE_CHECKING:
    from rich.console import Console

    from reach.optimize import OptimizationCandidate, OptimizationReport


def _should_show_trajectory_recall(report: OptimizationReport) -> bool:
    """Determine whether trajectory recall columns/metrics should be displayed."""
    return report.handoff is not None or any(
        c.trajectory_recall > c.recall for c in report.candidates
    )


def render_optimization_diff(
    report: OptimizationReport,
    candidate_index: int = 1,
) -> str:
    """Generate a unified diff showing changes between baseline and a candidate description."""
    if not report.candidates:
        return ""
    idx = candidate_index - 1
    if idx < 0 or idx >= len(report.candidates):
        return ""
    cand = report.candidates[idx]
    baseline_lines = ["description: >-\n", f"  {report.baseline_description}\n"]
    candidate_lines = ["description: >-\n", f"  {cand.description}\n"]

    if report.handoff is not None:
        baseline_lines.extend(report.handoff.target_body_before.splitlines(keepends=True))
        candidate_lines.extend(report.handoff.target_body_after.splitlines(keepends=True))

    target_diff = "".join(
        difflib.unified_diff(
            baseline_lines,
            candidate_lines,
            fromfile=f"a/{report.skill_name}/SKILL.md",
            tofile=f"b/{report.skill_name}/SKILL.md (candidate #{candidate_index})",
        )
    )
    if report.handoff is None:
        return target_diff

    rival_diff = "".join(
        difflib.unified_diff(
            report.handoff.rival_body_before.splitlines(keepends=True),
            report.handoff.rival_body_after.splitlines(keepends=True),
            fromfile=f"a/{report.handoff.rival_skill}/SKILL.md",
            tofile=f"b/{report.handoff.rival_skill}/SKILL.md (reciprocal handoff)",
        )
    )
    return f"{target_diff}{rival_diff}"


MAX_DESCRIPTION_PREVIEW: Final[int] = 55


def _print_optimization_baseline(console: Console, report: OptimizationReport) -> None:
    """Render baseline description, rivals, terms, and initial evaluation metrics."""
    console.print(f"[bold]Current Description:[/] {report.baseline_description}")
    if report.rival_name:
        console.print(f"[bold]Primary Rival:[/] [yellow]{report.rival_name}[/]")
    if report.ceded_terms:
        console.print(f"[bold]Ceded Terms (Rival Pull):[/] [red]{', '.join(report.ceded_terms)}[/]")
    if report.unclaimed_terms:
        console.print(
            f"[bold]Unclaimed Distinctive Terms:[/] [green]{', '.join(report.unclaimed_terms)}[/]",
        )
    if report.handoff is not None:
        console.print(
            f"[bold]Layer-2 Reciprocal Handoff:[/] "
            f"[cyan]{report.handoff.target_skill} ↔ {report.handoff.rival_skill}[/]"
        )

    if report.rounds:
        console.print(f"[bold]Optimization Rounds:[/] [cyan]{len(report.rounds)}[/]")
        for round_rec in report.rounds:
            best_cand = round_rec.best_candidate
            if best_cand:
                score_str = (
                    f"Holdout Recall: {best_cand.test_recall:.1%}"
                    if best_cand.test_recall is not None
                    else f"Δ Recall: {best_cand.delta_recall:+.1%}, Recall: {best_cand.recall:.1%}"
                )
                desc_snippet = (
                    f"{best_cand.description[:MAX_DESCRIPTION_PREVIEW]}..."
                    if len(best_cand.description) > MAX_DESCRIPTION_PREVIEW
                    else best_cand.description
                )
                console.print(
                    f"  [cyan]Round {round_rec.iteration}:[/] {score_str} — {desc_snippet}",
                )
        console.print()

    show_traj = _should_show_trajectory_recall(report)
    if report.has_probes:
        traj_segment = (
            f"Trajectory Recall: {report.baseline_trajectory_recall:.1%} | " if show_traj else ""
        )
        console.print(
            f"[bold]Baseline Metrics:[/] "
            f"Recall: {report.baseline_recall:.1%} | "
            f"{traj_segment}"
            f"Accuracy: {report.baseline_accuracy:.1%} | "
            f"Misroutes: {report.baseline_misroute:.1%}\n",
        )
    else:
        console.print(
            "[bold]Baseline Metrics:[/] "
            "[dim]Not evaluated (provide --queries to run empirical probes)[/]\n",
        )


def _format_candidate_delta(delta: float) -> str:
    """Format change in recall with directional indicator and color styling."""
    if delta > 0:
        return f"[green]+{delta:.1%} ▲[/]"
    if delta < 0:
        return f"[red]{delta:.1%} ▼[/]"
    return "0.0%"


@overload
def _format_candidate_row(
    cand: OptimizationCandidate,
    has_probes: bool,
    idx: int,
    *,
    has_test: Literal[True],
    show_traj: bool = False,
) -> tuple[str, ...]: ...


@overload
def _format_candidate_row(
    cand: OptimizationCandidate,
    has_probes: bool,
    idx: int,
    *,
    has_test: Literal[False] = False,
    show_traj: bool = False,
) -> tuple[str, ...]: ...


@overload
def _format_candidate_row(
    cand: OptimizationCandidate,
    has_probes: bool,
    idx: int,
    *,
    has_test: bool,
    show_traj: bool = False,
) -> tuple[str, ...]: ...


def _format_candidate_row(
    cand: OptimizationCandidate,
    has_probes: bool,
    idx: int,
    *,
    has_test: bool = False,
    show_traj: bool = False,
) -> tuple[str, ...]:
    """Format single candidate row values for optimization comparison table."""
    if has_probes:
        delta_str = _format_candidate_delta(cand.delta_recall)
        rec_str = f"{cand.recall:.1%}"
        traj_str = f"{cand.trajectory_recall:.1%}"
        mis_str = f"{cand.misroute_rate:.1%}"
    else:
        delta_str = "[dim]—[/]"
        rec_str = "[dim]—[/]"
        traj_str = "[dim]—[/]"
        mis_str = "[dim]—[/]"

    lint_str = "[green]✓ CLEAN[/]" if cand.lint_clean else "[yellow]✗ WARN[/]"
    row: list[str] = [f"#{idx}", cand.description, delta_str, rec_str]
    if show_traj:
        row.append(traj_str)
    if has_test:
        test_str = f"{cand.test_recall:.1%}" if cand.test_recall is not None else "[dim]—[/]"
        row.append(test_str)
    row.extend([mis_str, lint_str])
    return tuple(row)


def _print_optimization_footer(console: Console, report: OptimizationReport) -> None:
    """Print next-step application recommendation or confirmation notice."""
    if report.applied:
        handoff_suffix = (
            f" and {report.handoff.rival_skill}/SKILL.md" if report.handoff is not None else ""
        )
        console.print(
            f"[bold green]✓ Successfully updated {report.skill_name}/SKILL.md{handoff_suffix} "
            "with candidate #1![/]",
        )
    elif report.best_candidate:
        has_improvement = report.candidate_has_improvement(report.best_candidate)
        if report.has_probes and not has_improvement:
            if report.baseline_recall >= 1.0 and report.baseline_misroute <= 0.0:
                console.print(
                    "[yellow]Notice:[/] Baseline already achieves 100.0% recall with no misroutes."
                )
                console.print(
                    "[dim]No candidate improved upon baseline (all Δ Recall ≤ 0.0%). "
                    "Current description remains optimal.[/]"
                )
            else:
                console.print(
                    "[yellow]Notice:[/] No candidate improved upon baseline recall "
                    "(all Δ Recall ≤ 0.0%)."
                )
                console.print("[dim]Current description is retained.[/]")
        else:
            console.print(
                "[bold cyan]Recommendation:[/] To apply candidate #1 to disk, re-run with "
                "[bold]--auto-apply[/].",
            )
            console.print(
                "[dim]Run with [bold]--format diff[/bold] to inspect the unified YAML diff.[/]",
            )


def print_optimization(console: Console, report: OptimizationReport) -> None:
    """Render closed-loop optimization report and candidate comparison table to console."""
    console.print()
    console.rule(f"[bold cyan]Reach Closed-Loop Optimizer: {report.skill_name}[/]")
    console.print()

    _print_optimization_baseline(console, report)

    if not report.candidates:
        console.print("[yellow]No candidates were generated.[/]\n")
        return

    has_test = any(c.test_recall is not None for c in report.candidates)
    show_traj = _should_show_trajectory_recall(report)

    table = Table(box=box.ROUNDED, show_header=True, header_style="bold", expand=True)
    table.add_column("Rank", justify="center", style="bold", no_wrap=True)
    table.add_column("Candidate Description", style="cyan", ratio=4, min_width=18)
    table.add_column("Δ Recall", justify="right", no_wrap=True)
    table.add_column("Recall", justify="right", no_wrap=True)
    if show_traj:
        traj_header = "Traj" if has_test else "Traj Recall"
        table.add_column(traj_header, justify="right", no_wrap=True)
    if has_test:
        holdout_header = "Holdout" if show_traj else "Holdout Recall"
        table.add_column(holdout_header, justify="right", no_wrap=True)
    table.add_column("Misroutes", justify="right", no_wrap=True)
    table.add_column("Linter", justify="center", no_wrap=True)

    for idx, cand in enumerate(report.candidates, start=1):
        table.add_row(
            *_format_candidate_row(
                cand,
                report.has_probes,
                idx,
                has_test=has_test,
                show_traj=show_traj,
            )
        )

    console.print(table)
    console.print()
    _print_optimization_footer(console, report)
    console.print()
