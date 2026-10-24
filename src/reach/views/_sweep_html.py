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

"""Render standalone interactive HTML reports for scaling sweep studies."""

from __future__ import annotations

import html
import math
from typing import TYPE_CHECKING

from reach.static import get_sweep_css, get_sweep_template

if TYPE_CHECKING:
    from reach.sweep import ScalingStudy

__all__ = ["render_sweep_html"]

_PASS_RATE_HIGH: float = 0.8
_PASS_RATE_MID: float = 0.5
_MIN_DROP_PROB_FOR_GRADUAL_LABEL: float = 0.50
_MAX_DISPLAYED_SUSPECTS: int = 3


def _format_duration_s(duration_ms: float) -> str:
    """Format duration in milliseconds as seconds with one decimal place or dash when unset."""
    return f"{duration_ms / 1000:.1f}s" if duration_ms > 0 else "—"


def _format_knee_interval(study: ScalingStudy) -> str:
    """Format bootstrap knee confidence interval with right-censoring indicator when present."""
    if study.knee_scale_interval is None:
        return ""
    low, high = study.knee_scale_interval
    hi_str = f">{high}" if study.knee_upper_censored else str(high)
    return f"[{low}, {hi_str}]"


def _study_label(study: ScalingStudy, default: str) -> str:
    """Return a concise human-readable label for a scaling study."""
    if study.target_skill:
        return study.target_skill
    if study.agent and study.model and study.agent != study.model:
        return f"{study.agent} ({study.model})"
    return study.agent or study.model or default


def _resolve_entrypoint_knee(study: ScalingStudy) -> int | None:
    """Return study entrypoint_knee_scale or compute fallback from entrypoint metrics."""
    if study.entrypoint_knee_scale is not None:
        return study.entrypoint_knee_scale
    has_entry = any(
        (pt.entrypoint_f1_score if study.is_corpus_sweep else pt.entrypoint_pass_rate) is not None
        for pt in study.points
    )
    if not has_entry:
        return None
    from reach.sweep import find_kneedle_knee

    entry_curve = [
        (pt.entrypoint_f1_score if pt.entrypoint_f1_score is not None else pt.f1_score)
        if study.is_corpus_sweep
        else (pt.entrypoint_pass_rate if pt.entrypoint_pass_rate is not None else pt.pass_rate)
        for pt in study.points
    ]
    return find_kneedle_knee(
        study.scales[: len(study.points)],
        entry_curve,
        noise_floor=study.noise_floor,
        auto_smooth=True,
    )


def _render_sweep_html_svg(  # noqa: PLR0912, PLR0915
    study: ScalingStudy,
    compare: ScalingStudy | None,
) -> tuple[str, str]:
    """Generate SVG chart and legend for scaling curve."""
    x_min, x_max = 65.0, 700.0
    y_min, y_max = 42.0, 270.0
    scales = study.scales
    min_s, max_s = min(scales), max(scales)
    effective_entry_knee = _resolve_entrypoint_knee(study)

    def to_x(s: int) -> float:
        if len(scales) <= 1 or min_s == max_s:
            return (x_min + x_max) / 2
        return x_min + (math.log(s) - math.log(min_s)) / (math.log(max_s) - math.log(min_s)) * (
            x_max - x_min
        )

    def to_y(val: float) -> float:
        clamped = max(0.0, min(1.0, val))
        return y_max - clamped * (y_max - y_min)

    grid_lines = []
    y_labels = []
    ticks = [
        (1.0, "100%"),
        (0.75, "75%"),
        (0.50, "50%"),
        (0.25, "25%"),
        (0.0, "0%"),
    ]
    for pct_val, label in ticks:
        gy = to_y(pct_val)
        stroke = "#334155" if pct_val == 0.0 else "#26354a"
        dash = "" if pct_val == 0.0 else ' stroke-dasharray="4"'
        grid_lines.append(
            f'<line x1="{x_min}" y1="{gy:.1f}" x2="{x_max}" y2="{gy:.1f}" '
            f'stroke="{stroke}" stroke-width="1"{dash}/>'
        )
        y_labels.append(
            f'<text x="{x_min - 12}" y="{gy + 4:.1f}" text-anchor="end" '
            f'fill="#94a3b8" font-size="11">{label}</text>'
        )

    x_ticks = []
    x_lines = []
    for s in scales:
        gx = to_x(s)
        x_lines.append(
            f'<line x1="{gx:.1f}" y1="{y_min}" x2="{gx:.1f}" y2="{y_max}" '
            f'stroke="#1e293b" stroke-width="1"/>'
        )
        x_ticks.append(
            f'<text x="{gx:.1f}" y="{y_max + 20}" text-anchor="middle" '
            f'fill="#94a3b8" font-size="11">K={s}</text>'
        )

    ci_pts_high = []
    ci_pts_low = []
    for pt in study.points:
        interval = pt.f1_interval if study.is_corpus_sweep else pt.pass_rate_interval
        ci_pts_high.append(f"{to_x(pt.scale):.1f},{to_y(interval.high):.1f}")
        ci_pts_low.append(f"{to_x(pt.scale):.1f},{to_y(interval.low):.1f}")
    ci_poly_str = " ".join(ci_pts_high + list(reversed(ci_pts_low)))
    ci_polygon = f'<polygon points="{ci_poly_str}" fill="rgba(56, 189, 248, 0.15)"/>'

    traj_pts = [
        (
            pt.scale,
            to_x(pt.scale),
            to_y(pt.f1_score if study.is_corpus_sweep else pt.pass_rate),
            pt.f1_score if study.is_corpus_sweep else pt.pass_rate,
        )
        for pt in study.points
    ]
    traj_path = "M " + " L ".join(f"{x:.1f},{y:.1f}" for _, x, y, _ in traj_pts)
    traj_line = f'<path d="{traj_path}" fill="none" stroke="#38bdf8" stroke-width="3"/>'
    traj_nodes = []
    for s, x, y, val in traj_pts:
        traj_nodes.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="#38bdf8" '
            f'stroke="#0f172a" stroke-width="2"><title>K={s}: {val * 100:.1f}%</title></circle>'
        )
        traj_nodes.append(
            f'<text x="{x:.1f}" y="{y - 10:.1f}" text-anchor="middle" '
            f'fill="#38bdf8" font-size="11" font-weight="600">{val * 100:.1f}%</text>'
        )

    has_entry = any(
        (pt.entrypoint_f1_score if study.is_corpus_sweep else pt.entrypoint_pass_rate) is not None
        for pt in study.points
    )
    entry_elements = []
    if has_entry:
        entry_pts = []
        for pt in study.points:
            e_val = pt.entrypoint_f1_score if study.is_corpus_sweep else pt.entrypoint_pass_rate
            if e_val is not None:
                entry_pts.append((pt.scale, to_x(pt.scale), to_y(e_val), e_val))
        if entry_pts:
            e_path = "M " + " L ".join(f"{x:.1f},{y:.1f}" for _, x, y, _ in entry_pts)
            entry_elements.append(
                f'<path d="{e_path}" fill="none" stroke="#34d399" stroke-width="2" '
                f'stroke-dasharray="5,4"/>'
            )
            for s, x, y, e_val in entry_pts:
                entry_elements.append(
                    f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="#34d399">'
                    f"<title>Entry K={s}: {e_val * 100:.1f}%</title></circle>"
                )

    knee_elements = []
    if study.knee_scale is not None and study.knee_scale in scales:
        kx = to_x(study.knee_scale)
        knee_elements.append(
            f'<line x1="{kx:.1f}" y1="{y_min}" x2="{kx:.1f}" y2="{y_max}" '
            f'stroke="#fb7185" stroke-width="2" stroke-dasharray="4,4"/>'
        )
        knee_elements.append(
            f'<text x="{kx:.1f}" y="{y_min - 14}" text-anchor="middle" '
            f'fill="#fb7185" font-size="11" font-weight="700">K* = {study.knee_scale}</text>'
        )
    if (
        effective_entry_knee is not None
        and effective_entry_knee != study.knee_scale
        and effective_entry_knee in scales
    ):
        ekx = to_x(effective_entry_knee)
        knee_elements.append(
            f'<line x1="{ekx:.1f}" y1="{y_min}" x2="{ekx:.1f}" y2="{y_max}" '
            f'stroke="#34d399" stroke-width="1.5" stroke-dasharray="3,3"/>'
        )
        knee_elements.append(
            f'<text x="{ekx:.1f}" y="{y_min - 14}" text-anchor="middle" '
            f'fill="#34d399" font-size="11" font-weight="600">'
            f"K*(entry) = {effective_entry_knee}</text>"
        )

    compare_elements = []
    if compare is not None:
        comp_pts = [
            (
                cpt.scale,
                to_x(cpt.scale),
                to_y(cpt.f1_score if compare.is_corpus_sweep else cpt.pass_rate),
                cpt.f1_score if compare.is_corpus_sweep else cpt.pass_rate,
            )
            for cpt in compare.points
            if min_s <= cpt.scale <= max_s
        ]
        if comp_pts:
            c_path = "M " + " L ".join(f"{x:.1f},{y:.1f}" for _, x, y, _ in comp_pts)
            compare_elements.append(
                f'<path d="{c_path}" fill="none" stroke="#fbbf24" stroke-width="2.5" '
                f'stroke-dasharray="6,4"/>'
            )
            for s, x, y, val in comp_pts:
                compare_elements.append(
                    f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="#fbbf24" '
                    f'stroke="#0f172a" stroke-width="1.5">'
                    f"<title>Compare K={s}: {val * 100:.1f}%</title></circle>"
                )
                compare_elements.append(
                    f'<text x="{x:.1f}" y="{y + 16:.1f}" text-anchor="middle" '
                    f'fill="#fbbf24" font-size="10" font-weight="600">{val * 100:.1f}%</text>'
                )

    metric_name = "F1 Score" if study.is_corpus_sweep else "Pass Rate"
    legend_items = [
        (
            f'<div class="legend-item"><div class="legend-dot" style="background: #38bdf8;"></div>'
            f"<span>Trajectory ({metric_name})</span></div>"
        ),
        (
            '<div class="legend-item"><div class="legend-dot" '
            'style="background: rgba(56, 189, 248, 0.35);"></div><span>95% CI Band</span></div>'
        ),
    ]
    if has_entry:
        entry_label = (
            f"First-Turn Entrypoint (K*={effective_entry_knee})"
            if effective_entry_knee is not None
            else "First-Turn Entrypoint"
        )
        legend_items.append(
            f'<div class="legend-item"><div class="legend-dot" style="background: #34d399;"></div>'
            f"<span>{entry_label}</span></div>"
        )
    if study.knee_scale is not None:
        legend_items.append(
            f'<div class="legend-item"><div class="legend-dot" style="background: #fb7185;"></div>'
            f"<span>Capacity Knee (K*={study.knee_scale})</span></div>"
        )
    if compare is not None:
        comp_name = html.escape(_study_label(compare, "Comparison Baseline"))
        legend_items.append(
            f'<div class="legend-item"><div class="legend-dot" style="background: #fbbf24;"></div>'
            f"<span>Compare: {comp_name}</span></div>"
        )

    svg_content = f"""
    <svg viewBox="0 0 750 320" style="width: 100%; height: auto; display: block;">
      {" ".join(grid_lines)}
      {" ".join(y_labels)}
      {" ".join(x_lines)}
      {" ".join(x_ticks)}
      {ci_polygon}
      {traj_line}
      {" ".join(traj_nodes)}
      {" ".join(entry_elements)}
      {" ".join(knee_elements)}
      {" ".join(compare_elements)}
    </svg>
    """
    return svg_content, "\n".join(legend_items)


def _render_sweep_html_table(  # noqa: PLR0912, PLR0915
    study: ScalingStudy,
    compare: ScalingStudy | None,
) -> str:
    """Render the per-scale evaluation results table."""
    has_negatives = any(pt.negative_probes > 0 for pt in study.points)
    has_trunc = any(
        pt.delta_truncated != 0 or pt.disclosure_states.get("name_only_elided", 0) > 0
        for pt in study.points
    )
    compare_by_scale = {cpt.scale: cpt for cpt in compare.points} if compare is not None else {}

    entry_title = "Entry F1" if study.is_corpus_sweep else "Entry Pass"
    metric_title = "F1 Score" if study.is_corpus_sweep else "Pass Rate"
    headers = [
        "<th>Scale (K)</th>",
        f'<th class="text-right">{metric_title}</th>',
        '<th class="text-center">95% CI</th>',
        f'<th class="text-right">{entry_title}</th>',
        '<th class="text-right">Step Eff</th>',
    ]
    if study.is_corpus_sweep:
        headers.extend(
            [
                '<th class="text-right">Recall</th>',
                '<th class="text-right">Precision</th>',
            ]
        )
        if has_negatives:
            headers.append('<th class="text-right">Abstain</th>')
    else:
        headers.extend(
            [
                '<th class="text-right">Δ Total</th>',
                '<th class="text-right">Δ Abstain</th>',
            ]
        )
    headers.append('<th class="text-right">Δ Collide</th>')
    if has_trunc:
        table_headers_trunc = '<th class="text-right">Trunc</th>'
        headers.append(table_headers_trunc)
    headers.extend(
        [
            '<th class="text-center">Probes</th>',
            '<th class="text-right">Duration</th>',
        ]
    )
    if compare is not None:
        headers.append('<th class="text-right">Compare</th>')

    rows = []
    for pt in study.points:
        val = pt.f1_score if study.is_corpus_sweep else pt.pass_rate
        cls = (
            "text-green font-bold"
            if val >= _PASS_RATE_HIGH
            else ("text-amber" if val >= _PASS_RATE_MID else "text-rose")
        )
        interval = pt.f1_interval if study.is_corpus_sweep else pt.pass_rate_interval
        ci_str = interval.format_percent(separator=" - ")

        e_val = pt.entrypoint_f1_score if study.is_corpus_sweep else pt.entrypoint_pass_rate
        entry_str = f"{e_val * 100:.1f}%" if e_val is not None else "—"
        eff_str = f"{pt.step_efficiency_mean:.2f}" if pt.step_efficiency_mean > 0 else "—"
        dur_str = _format_duration_s(pt.duration_ms_mean)

        cells = [
            f'<td class="font-mono text-blue"><strong>K={pt.scale}</strong></td>',
            f'<td class="text-right font-mono {cls}">{val * 100:.1f}%</td>',
            f'<td class="text-center font-mono text-muted">{ci_str}</td>',
            f'<td class="text-right font-mono">{entry_str}</td>',
            f'<td class="text-right font-mono">{eff_str}</td>',
        ]
        if study.is_corpus_sweep:
            cells.extend(
                [
                    f'<td class="text-right font-mono">{pt.recall * 100:.1f}%</td>',
                    f'<td class="text-right font-mono">{pt.precision * 100:.1f}%</td>',
                ]
            )
            if has_negatives:
                abs_str = (
                    f"{pt.abstention_rate * 100:.1f}%" if pt.abstention_rate is not None else "—"
                )
                cells.append(f'<td class="text-right font-mono">{abs_str}</td>')
        else:
            tot_str = f"{pt.delta_vs_baseline * 100:+.1f}%" if pt.delta_vs_baseline != 0 else "0.0%"
            abs_str = f"{pt.delta_abstention * 100:+.1f}%" if pt.delta_abstention != 0 else "0.0%"
            cells.extend(
                [
                    f'<td class="text-right font-mono">{tot_str}</td>',
                    f'<td class="text-right font-mono">{abs_str}</td>',
                ]
            )
        col_str = f"{pt.delta_collision * 100:+.1f}%" if pt.delta_collision != 0 else "0.0%"
        cells.append(f'<td class="text-right font-mono text-rose">{col_str}</td>')
        if has_trunc:
            elided = pt.disclosure_states.get("name_only_elided", 0)
            cells.append(f'<td class="text-right font-mono text-amber">{elided}</td>')
        p_str = (
            f"{pt.in_scope_probes}/{pt.negative_probes}"
            if has_negatives
            else f"{pt.probes_executed}"
        )
        if pt.probes_errored > 0:
            p_str += f" ({pt.probes_errored} err)"
        cells.extend(
            [
                f'<td class="text-center font-mono">{p_str}</td>',
                f'<td class="text-right font-mono">{dur_str}</td>',
            ]
        )
        if compare is not None:
            cpt = compare_by_scale.get(pt.scale)
            if cpt is not None:
                c_val = cpt.f1_score if compare.is_corpus_sweep else cpt.pass_rate
                diff = val - c_val
                diff_cls = "text-green" if diff >= 0 else "text-rose"
                cells.append(
                    f'<td class="text-right font-mono">{c_val * 100:.1f}% '
                    f'<span class="{diff_cls}">({diff * 100:+.1f}%)</span></td>'
                )
            else:
                cells.append('<td class="text-right font-mono text-muted">—</td>')

        rows.append(f"<tr>{''.join(cells)}</tr>")

    return f"""
    <div class="card" style="margin-bottom: 2rem;">
      <h2>Per-Scale Evaluation Performance Scorecard</h2>
      <div class="table-wrapper">
        <table>
          <thead>
            <tr>{"".join(headers)}</tr>
          </thead>
          <tbody>
            {"".join(rows)}
          </tbody>
        </table>
      </div>
    </div>
    """


def _render_sweep_html_collisions(study: ScalingStudy) -> str:
    """Render collision diagnostics table if collisions are present."""
    if not study.replicate_collisions:
        return ""
    col_rows = []
    for c in study.replicate_collisions:
        if c.expected_skill and c.expected_skill != c.query_id:
            target_str = (
                f"{html.escape(c.expected_skill)} "
                f'<span class="text-muted">({html.escape(c.query_id)})</span>'
            )
        else:
            target_str = html.escape(c.expected_skill or c.query_id)
        inv_str = html.escape(", ".join(c.invoked_distractors)) if c.invoked_distractors else "—"
        if c.suspect_distractors:
            shown = c.suspect_distractors[:_MAX_DISPLAYED_SUSPECTS]
            extra = len(c.suspect_distractors) - len(shown)
            susp_str = html.escape(", ".join(shown))
            if extra > 0:
                susp_str += f" (+{extra} more)"
        else:
            susp_str = "—"
        reps_str = f"{len(c.failed_replicates)}F / {len(c.passed_replicates)}P"
        col_rows.append(
            f"<tr>"
            f'<td class="font-mono text-blue">{target_str}</td>'
            f'<td class="font-mono text-center">K={c.scale}</td>'
            f'<td class="font-mono text-rose">{inv_str}</td>'
            f'<td class="font-mono text-muted">{susp_str}</td>'
            f'<td class="font-mono text-center">{reps_str}</td>'
            f"</tr>"
        )
    return f"""
    <div class="card" style="margin-bottom: 2rem;">
      <h2>Replicate Collision Diagnostics ({len(study.replicate_collisions)} queries)</h2>
      <div class="table-wrapper">
        <table>
          <thead>
            <tr>
              <th>Target Skill / Query</th>
              <th class="text-center">Scale</th>
              <th>Invoked Distractor(s)</th>
              <th>Suspect Distractor(s) (BM25)</th>
              <th class="text-center">Replicates</th>
            </tr>
          </thead>
          <tbody>
            {"".join(col_rows)}
          </tbody>
        </table>
      </div>
    </div>
    """


def render_sweep_html(
    study: ScalingStudy,
    *,
    compare: ScalingStudy | None = None,
) -> str:
    """Render a self-contained interactive HTML report for a scaling sweep study."""
    catalog_title = (
        "Corpus Capacity Scaling Study"
        if study.is_corpus_sweep
        else f"Scaling Sweep: {study.target_skill}"
    )
    subtitle = (
        "Multi-Class Skill Retrieval & Capacity Degradation under Catalog Expansion"
        if study.is_corpus_sweep
        else f"Reachability Decay across Catalog Scales for {study.target_skill or 'skill'}"
    )

    badges: list[str] = []
    if study.agent or study.model:
        agent_label = (
            f"{study.agent} ({study.model})"
            if study.agent and study.model and study.agent != study.model
            else (study.agent or study.model or "")
        )
        badges.append(f"Agent: <strong>{html.escape(agent_label)}</strong>")
    badges.extend(
        [
            f"Corpus: <strong>{study.total_corpus_skills} skills</strong>",
            (
                f"Scales: <strong>{len(study.scales)} steps "
                f"(K={study.scales[0]}..{study.scales[-1]})</strong>"
            ),
            f"Replicates: <strong>{study.catalog_replicates}</strong>",
        ]
    )
    if study.anchor_skills:
        badges.append(f"Anchors: <strong>{len(study.anchor_skills)}</strong>")
    total_probes = sum(pt.probes_executed for pt in study.points)
    badges.append(f"Probes: <strong>{total_probes}</strong>")
    if compare is not None:
        comp_target = html.escape(_study_label(compare, "Comparison Baseline"))
        badges.append(f"Compare: <strong>{comp_target} ({len(compare.scales)} scales)</strong>")
    badges_html = "\n".join(f'<div class="badge">{b}</div>' for b in badges)

    knee_val = (
        str(study.knee_scale)
        if study.knee_scale is not None
        else (
            "Gradual"
            if (study.drop_probability or 0) >= _MIN_DROP_PROB_FOR_GRADUAL_LABEL
            else "N/A"
        )
    )
    knee_ci_str = _format_knee_interval(study)
    knee_sub = f"95% CI: {knee_ci_str}" if knee_ci_str else "Optimal capacity inflection point"

    p0 = study.points[0]
    p_last = study.points[-1]
    effective_entry_knee = _resolve_entrypoint_knee(study)
    entry_b = p0.entrypoint_f1_score if study.is_corpus_sweep else p0.entrypoint_pass_rate
    entry_f = p_last.entrypoint_f1_score if study.is_corpus_sweep else p_last.entrypoint_pass_rate
    if effective_entry_knee is not None:
        entry_knee_val = str(effective_entry_knee)
        entry_knee_sub = (
            f"Single-turn routing knee ({entry_b * 100:.1f}% → {entry_f * 100:.1f}%)"
            if entry_b is not None and entry_f is not None
            else "Single-turn entrypoint routing knee"
        )
    elif entry_b is not None and entry_f is not None and (entry_b - entry_f) > study.noise_floor:
        entry_knee_val = "Gradual"
        entry_knee_sub = (
            f"Log-linear decay ({entry_b * 100:.1f}% → {entry_f * 100:.1f}%, "
            f"Δ {(entry_f - entry_b) * 100:+.1f}%)"
        )
    else:
        entry_knee_val = "N/A"
        entry_knee_sub = "No distinct knee detected"

    term_metric = p_last.f1_score if study.is_corpus_sweep else p_last.pass_rate
    term_val = f"{term_metric * 100:.1f}%"
    term_label = "Terminal F1" if study.is_corpus_sweep else "Terminal Pass Rate"
    base_metric = p0.f1_score if study.is_corpus_sweep else p0.pass_rate
    net_change = term_metric - base_metric
    term_sub = f"Baseline: {base_metric * 100:.1f}% @ K={p0.scale} (Δ {net_change * 100:+.1f}%)"

    col_val = f"{study.delta_collision * 100:+.1f}%"
    col_sub = (
        f"Abstention Loss: {study.delta_abstention * 100:+.1f}% | "
        f"Truncation Loss: {study.delta_truncated * 100:+.1f}%"
    )

    svg_content, legend_html = _render_sweep_html_svg(study, compare)
    table_section = _render_sweep_html_table(study, compare)
    collision_section = _render_sweep_html_collisions(study)

    return get_sweep_template().format(
        catalog_title=html.escape(catalog_title),
        style=get_sweep_css(),
        subtitle=html.escape(subtitle),
        badges_html=badges_html,
        knee_val=html.escape(knee_val),
        knee_sub=html.escape(knee_sub),
        entry_knee_val=html.escape(entry_knee_val),
        entry_knee_sub=html.escape(entry_knee_sub),
        term_label=html.escape(term_label),
        term_val=html.escape(term_val),
        term_sub=html.escape(term_sub),
        col_val=html.escape(col_val),
        col_sub=html.escape(col_sub),
        svg_content=svg_content,
        legend_html=legend_html,
        table_section=table_section,
        collision_section=collision_section,
    )
