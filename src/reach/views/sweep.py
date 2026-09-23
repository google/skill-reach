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

"""Render multi-scale catalog scaling sweeps and loss decomposition tables."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich import box
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from reach.rendering import csv_document, dispatch_render

if TYPE_CHECKING:
    from collections.abc import Sequence

    from rich.console import Console

    from reach.sweep import ScalingPoint, ScalingStudy

__all__ = [
    "SWEEP_RENDERERS",
    "print_sweep",
    "render_ascii_curve",
    "render_sweep",
    "render_sweep_csv",
    "render_sweep_json",
]


_PASS_RATE_HIGH: float = 0.8
_PASS_RATE_MID: float = 0.5
_MIN_CURVE_POINTS: int = 2
_MIN_SCALES_FOR_LOSS_DECOMPOSITION: int = 2
_LEVEL_TOLERANCE: float = 0.125


def _print_corpus_capacity_sweep(console: Console, study: ScalingStudy) -> None:  # noqa: PLR0912, PLR0915
    """Render a Rich table and capacity summary for whole-corpus scaling sweep."""
    header: list[tuple[str, str]] = [
        ("Corpus Capacity Scaling: ", "bold"),
        (f"{study.total_corpus_skills} skills", "cyan bold"),
        (" across library sizes", "dim"),
    ]
    if study.anchor_skills:
        header.extend(
            [
                (" (", "dim"),
                (f"{len(study.anchor_skills)} anchor skills", "green bold"),
                (")", "dim"),
            ]
        )
    console.print(Text.assemble(*header))

    decision_lines: list[tuple[str, str]] = []
    if study.sla_90_scale is not None:
        decision_lines.append(
            (f"  • Safe Operating Capacity (SLA ≥ 90% F1): K ≤ {study.sla_90_scale}", "bold green")
        )
        if study.sla_90_interpolated is not None:
            decision_lines.append((f" (continuous: {study.sla_90_interpolated:.1f})", "dim"))
        decision_lines.append(("\n", ""))

    if study.sla_85_scale is not None:
        decision_lines.append(
            (f"  • Degraded Capacity Limit (SLA ≥ 85% F1): K ≤ {study.sla_85_scale}", "bold yellow")
        )
        if study.sla_85_interpolated is not None:
            decision_lines.append((f" (continuous: {study.sla_85_interpolated:.1f})", "dim"))
        decision_lines.append(("\n", ""))

    if study.knee_scale is not None:
        decision_lines.append(
            (f"  • Capacity Knee Inflection (Kneedle k*): K = {study.knee_scale}", "bold cyan")
        )
        decision_lines.append(("\n", ""))

    if len(study.scales) >= _MIN_SCALES_FOR_LOSS_DECOMPOSITION and (
        study.total_delta != 0 or study.total_shadowing_loss != 0 or study.total_context_loss != 0
    ):
        loss_text = (
            f"  • Loss Decomposition (K={study.scales[0]}→{study.scales[-1]}): "
            f"Δ Shadowing {study.total_shadowing_loss * 100:+.1f}% | "
            f"Δ Context {study.total_context_loss * 100:+.1f}%"
        )
        decision_lines.append((loss_text, "dim"))
        decision_lines.append(("\n", ""))

    if decision_lines:
        panel = Panel(
            Text.assemble(*decision_lines[:-1]),  # strip trailing newline
            title="[bold]Optimal Catalog Capacity[/]",
            title_align="left",
            box=box.ROUNDED,
            border_style="cyan",
            expand=False,
        )
        console.print(panel)

    has_negatives = any(pt.negative_probes > 0 for pt in study.points)
    has_truncation = any(pt.disclosure_states.get("name_only_elided", 0) > 0 for pt in study.points)

    compact_cols = has_truncation
    table = Table(
        box=box.SIMPLE,
        show_header=True,
        pad_edge=False,
        padding=(0, 0),
        title="Corpus Scaling: Multi-Class Retrieval & Capacity Degradation",
    )
    table.add_column("Scale", justify="right", no_wrap=True)
    table.add_column("Recall", justify="right", no_wrap=True)
    table.add_column("Prec" if compact_cols else "Precision", justify="right", no_wrap=True)
    if has_negatives:
        table.add_column("Abstain", justify="right", no_wrap=True)
    table.add_column("F1", justify="right", no_wrap=True)
    table.add_column("95% CI", justify="center", style="dim", no_wrap=True)
    table.add_column("Δ Shadow", justify="right", style="red", no_wrap=True)
    if has_truncation:
        table.add_column("Trunc", justify="right", style="yellow", no_wrap=True)
    table.add_column("Tokens", justify="right", style="dim", no_wrap=True)
    table.add_column("Probes", justify="right", no_wrap=True)
    table.add_column(
        "Time" if compact_cols else "Duration", justify="right", style="dim", no_wrap=True
    )

    for pt in study.points:
        f1_pct = f"{pt.f1_score * 100:.1f}%"
        rate_style = (
            "bold green"
            if pt.f1_score >= _PASS_RATE_HIGH
            else ("yellow" if pt.f1_score >= _PASS_RATE_MID else "bold red")
        )
        rec_pct = f"{pt.recall * 100:.1f}%"
        prec_pct = f"{pt.precision * 100:.1f}%"
        f1_ci_str = f"[{pt.f1_interval[0] * 100:.1f}%-{pt.f1_interval[1] * 100:.1f}%]"
        shd_str = f"{pt.delta_shadowing * 100:+.1f}%" if pt.delta_shadowing != 0 else "0.0%"
        tok_str = f"{pt.prompt_tokens_mean:,.0f}" if pt.prompt_tokens_mean is not None else "—"
        probes_str = (
            f"{pt.in_scope_probes}/{pt.negative_probes}"
            if has_negatives
            else str(pt.in_scope_probes)
        )
        if pt.duration_ms_mean <= 0:
            dur_str = "—"
        elif compact_cols:
            dur_str = f"{pt.duration_ms_mean / 1000:.1f}s"
        else:
            dur_str = f"{pt.duration_ms_mean:.0f}ms"

        row: list[str | Text] = [
            str(pt.scale),
            rec_pct,
            prec_pct,
        ]
        if has_negatives:
            abs_pct = f"{pt.abstention_rate * 100:.1f}%" if pt.abstention_rate is not None else "—"
            row.append(abs_pct)
        row.extend(
            [
                Text(f1_pct, style=rate_style),
                f1_ci_str,
                shd_str,
            ]
        )
        if has_truncation:
            elided = pt.disclosure_states.get("name_only_elided", 0)
            total_p = max(1, pt.probes_executed)
            row.append(f"{round(elided * 100 / total_p)}% ({elided})")
        row.extend([tok_str, probes_str, dur_str])
        table.add_row(*row)

    console.print(table)
    sparkline_parts = [f"K={pt.scale}: {pt.f1_score * 100:.0f}%" for pt in study.points]
    console.print(Text("F1 Trajectory: " + " ──> ".join(sparkline_parts), style="dim"))

    if len(study.points) >= _MIN_CURVE_POINTS:
        console.print()
        for line in render_ascii_curve(study.points, metric="f1"):
            console.print(Text(line, style="dim"))


def _print_single_skill_sweep(console: Console, study: ScalingStudy) -> None:
    """Render a Rich table and loss decomposition for single-skill scaling sweep."""
    target_display = study.target_skill or "all"
    console.print(
        Text.assemble(
            ("Scaling Sweep: ", "bold"),
            (target_display, "cyan bold"),
            (" across library sizes", "dim"),
        )
    )

    if study.knee_scale is not None:
        console.print(
            Text.assemble(
                ("Capacity Knee: ", "bold yellow"),
                (f"k* = {study.knee_scale}", "bold yellow"),
                (" (inflection point where distractor shadowing accelerates)", "dim"),
            )
        )

    table = Table(
        box=box.SIMPLE,
        show_header=True,
        title="Reachability Decay Across Catalog Scales",
    )
    table.add_column("Scale (N)", justify="right", no_wrap=True)
    table.add_column("Pass Rate", justify="right", no_wrap=True)
    table.add_column("95% CI", justify="center", style="dim", no_wrap=True)
    table.add_column("Δ Total", justify="right", no_wrap=True)
    table.add_column("Δ Context", justify="right", style="cyan", no_wrap=True)
    table.add_column("Δ Shadowing", justify="right", style="red", no_wrap=True)
    table.add_column("Probes", justify="right", no_wrap=True)
    table.add_column("Duration", justify="right", style="dim", no_wrap=True)

    for pt in study.points:
        pct = f"{pt.pass_rate * 100:.1f}%"
        rate_style = (
            "bold green"
            if pt.pass_rate >= _PASS_RATE_HIGH
            else ("yellow" if pt.pass_rate >= _PASS_RATE_MID else "bold red")
        )
        ci = f"[{pt.pass_rate_interval[0] * 100:.1f}% - {pt.pass_rate_interval[1] * 100:.1f}%]"

        tot_str = f"{pt.delta_vs_baseline * 100:+.1f}%" if pt.delta_vs_baseline != 0 else "0.0%"
        ctx_str = f"{pt.delta_context * 100:+.1f}%" if pt.delta_context != 0 else "0.0%"
        shd_str = f"{pt.delta_shadowing * 100:+.1f}%" if pt.delta_shadowing != 0 else "0.0%"
        probes_str = f"{pt.probes_executed - pt.probes_failed}/{pt.probes_executed}"
        dur_str = f"{pt.duration_ms_mean:.0f}ms" if pt.duration_ms_mean > 0 else "—"

        table.add_row(
            str(pt.scale),
            Text(pct, style=rate_style),
            ci,
            tot_str,
            ctx_str,
            shd_str,
            probes_str,
            dur_str,
        )

    console.print(table)
    sparkline_parts = [f"N={pt.scale}: {pt.pass_rate * 100:.0f}%" for pt in study.points]
    console.print(Text("Trajectory: " + " ──> ".join(sparkline_parts), style="dim"))

    if len(study.points) >= _MIN_CURVE_POINTS:
        console.print()
        for line in render_ascii_curve(study.points, metric="pass_rate"):
            console.print(Text(line, style="dim"))


def print_sweep(console: Console, study: ScalingStudy) -> None:
    """Render a Rich table and summary of the scaling sweep study."""
    if study.is_corpus_sweep:
        _print_corpus_capacity_sweep(console, study)
    else:
        _print_single_skill_sweep(console, study)


_ZOOM_HIGH_THRESHOLD: float = 0.75
_ZOOM_MID_THRESHOLD: float = 0.55
_TOP_TIER_BIAS: float = 0.95


def _select_curve_levels(values: Sequence[float]) -> list[float]:
    """Select adaptive Y-axis tick levels so high-accuracy curves show fine slope resolution."""
    if not values:
        return [1.0, 0.75, 0.5, 0.25, 0.0]
    min_v = min(values)
    max_v = max(values)
    if min_v >= _ZOOM_HIGH_THRESHOLD and max_v > min_v:
        return [1.0, 0.95, 0.90, 0.85, 0.80]
    if min_v >= _ZOOM_MID_THRESHOLD and max_v > min_v:
        return [1.0, 0.90, 0.80, 0.70, 0.60]
    return [1.0, 0.75, 0.5, 0.25, 0.0]


def render_ascii_curve(
    points: Sequence[ScalingPoint],
    metric: str = "pass_rate",
) -> list[str]:
    """Render an ASCII scaling curve depicting metric progression across catalog scales."""
    if not points:
        return []

    prefix = "K" if metric == "f1" else "N"
    title = "Scaling Curve (F1):" if metric == "f1" else "Scaling Curve:"
    values = [p.f1_score if metric == "f1" else p.pass_rate for p in points]
    levels = _select_curve_levels(values)
    scale_cols = [f"{prefix}={p.scale}" for p in points]
    col_width = max(max(len(c) for c in scale_cols) + 2, 6)

    nearest_level_by_point = [
        min(
            range(len(levels)),
            key=lambda idx: (
                round(abs(val - levels[idx]), 6),
                idx if val >= _TOP_TIER_BIAS else -idx,
            ),
        )
        for val in values
    ]

    lines: list[str] = [title]
    for lvl_idx, lvl in enumerate(levels):
        row_cells: list[str] = [f"{round(lvl * 100):3d}% |"]
        for pt_idx in range(len(points)):
            symbol = "●" if nearest_level_by_point[pt_idx] == lvl_idx else " "
            row_cells.append(symbol.center(col_width))
        lines.append("".join(row_cells))

    axis_sep = "     +" + "-" * (col_width * len(points))
    lines.append(axis_sep)
    axis_labels = "      " + "".join(c.center(col_width) for c in scale_cols)
    lines.append(axis_labels)
    return lines


def render_sweep_json(study: ScalingStudy) -> str:
    """Serialize ScalingStudy to JSON."""
    return study.model_dump_json(indent=2)


def render_sweep_csv(study: ScalingStudy) -> str:
    """Export ScalingStudy metrics to formatted CSV."""
    if study.is_corpus_sweep:
        headers = [
            "scale",
            "recall",
            "recall_ci_low",
            "recall_ci_high",
            "precision",
            "precision_ci_low",
            "precision_ci_high",
            "abstention_rate",
            "f1_score",
            "f1_ci_low",
            "f1_ci_high",
            "in_scope_probes",
            "negative_probes",
            "duration_ms",
        ]
        rows = [
            [
                pt.scale,
                f"{pt.recall:.4f}",
                f"{pt.recall_interval[0]:.4f}",
                f"{pt.recall_interval[1]:.4f}",
                f"{pt.precision:.4f}",
                f"{pt.precision_interval[0]:.4f}",
                f"{pt.precision_interval[1]:.4f}",
                f"{pt.abstention_rate:.4f}" if pt.abstention_rate is not None else "",
                f"{pt.f1_score:.4f}",
                f"{pt.f1_interval[0]:.4f}",
                f"{pt.f1_interval[1]:.4f}",
                pt.in_scope_probes,
                pt.negative_probes,
                f"{pt.duration_ms_mean:.2f}",
            ]
            for pt in study.points
        ]
        return csv_document(headers, rows)

    rows_single: list[list[object]] = [
        [
            study.target_skill or "all",
            pt.scale,
            f"{pt.pass_rate:.4f}",
            f"{pt.pass_rate_interval[0]:.4f}",
            f"{pt.pass_rate_interval[1]:.4f}",
            f"{pt.delta_vs_baseline:.4f}",
            f"{pt.delta_context:.4f}",
            f"{pt.delta_shadowing:.4f}",
            pt.probes_executed,
            pt.probes_failed,
            f"{pt.duration_ms_mean:.2f}",
        ]
        for pt in study.points
    ]

    headers_single = [
        "skill",
        "scale",
        "pass_rate",
        "wilson_low",
        "wilson_high",
        "delta_total",
        "delta_context",
        "delta_shadowing",
        "probes_executed",
        "probes_failed",
        "duration_ms",
    ]
    return csv_document(headers_single, rows_single)


SWEEP_RENDERERS = {
    "csv": render_sweep_csv,
    "json": render_sweep_json,
}


def render_sweep(study: ScalingStudy, fmt: str) -> str:
    """Render ScalingStudy into the requested format."""
    return dispatch_render(SWEEP_RENDERERS, fmt, study)
