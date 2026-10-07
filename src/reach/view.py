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

"""Render a recorded run artifact as an interactive self-contained HTML diagnostic workbench.

For Rich-based terminal and console formatting views, see `reach.views`.
"""

from __future__ import annotations

import html
from collections import defaultdict
from typing import TYPE_CHECKING

from reach.artifact import NO_SKILL
from reach.rendering import dispatch_render
from reach.static import get_view_css, get_view_js, get_view_template

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from reach.artifact import Artifact, QueryRecord, SkillScore
    from reach.leak import Leak
    from reach.uncertainty import Interval


def _esc(value: object) -> str:
    """Escape a value for safe inclusion in HTML text or attributes."""
    return html.escape(str(value), quote=True)


def _pct(value: float | None) -> str:
    """Format a floating-point ratio as a whole-percentage string, or empty if None."""
    return "" if value is None else f"{value * 100:.0f}%"


def _bounds(interval: Interval | None) -> str:
    """Format an Interval as a hyphen-separated percentage range."""
    if interval is None:
        return ""
    return f"{interval.low * 100:.0f}-{interval.high * 100:.0f}%"


def _header_html(artifact: Artifact) -> str:
    """Generate HTML header with run provenance, metrics, and digests."""
    provenance = artifact.provenance
    digests = artifact.digests
    scores = artifact.scores
    spread = artifact.spread
    error = f" ± {spread.standard_error * 100:.1f}pp" if spread.standard_error else ""
    if spread.repeated_queries == 0:
        consistency_dd = (
            '<span class="dim">—</span> <span class="dim">(1 attempt, no repeated queries)</span>'
        )
    else:
        ci = _bounds(scores.consistency_interval)
        ci_html = f' <span class="dim">({ci})</span>' if ci else ""
        consistency_dd = f"{_pct(scores.consistency)}{ci_html}"
    mode_str = f"mode: {_esc(artifact.catalog_mode.value)}"
    return f"""
<div class="workbench-header">
  <div class="header-main">
    <div class="title-row">
      <h1>{_esc(artifact.catalog_id)}</h1>
      <div class="header-badge">Diagnostic Workbench</div>
    </div>
    <p class="subject">
      {mode_str} &middot; {artifact.catalog_size} skills
      &middot; {_esc(provenance.runtime)}/{_esc(provenance.model)} x{provenance.attempts}
      &middot; {artifact.probes} probes
    </p>
  </div>
  <dl class="figures">
    <div class="figure-cell">
      <dt>consistency</dt>
      <dd>{consistency_dd}</dd>
    </div>
    <div class="figure-cell">
      <dt>top-1</dt><dd>{scores.top1_accuracy * 100:.1f}%{error}</dd>
    </div>
    <div class="figure-cell">
      <dt>abstention</dt>
      <dd>{_pct(scores.abstention.rate)} ({_pct(scores.abstention.false_rate)} false)</dd>
    </div>
    <div class="figure-cell">
      <dt>macro-F1</dt><dd>{scores.not_headline.macro_f1 * 100:.1f}%</dd>
    </div>
  </dl>
  <p class="digests">
    arm {_esc(provenance.arm)} &middot; corpus {_esc(digests.corpus_digest)}
    &middot; truth {_esc(digests.queries_digest)}
  </p>
</div>
"""


def _confusion_cell(
    row: str,
    col: str,
    counts: Mapping[tuple[str, str], int],
    query_samples: Mapping[tuple[str, str], str],
) -> str:
    """Format an individual table data cell in the confusion matrix."""
    n = counts.get((row, col), 0)
    css = "hit" if row == col else ("miss" if n else "")
    title = query_samples.get((row, col))
    attr = f' title="{_esc(title)}"' if title else ""
    return f'<td class="{css}"{attr}>{n or ""}</td>'


def _confusion_html(artifact: Artifact) -> str:
    """Generate the HTML table for the expected vs invoked confusion matrix."""
    pairs = artifact.confusion
    if not pairs:
        return '<p class="empty">No probes were recorded.</p>'
    counts: dict[tuple[str, str], int] = defaultdict(int)
    query_samples: dict[tuple[str, str], str] = {}
    expected_labels: set[str] = set()
    invoked_labels: set[str] = set()
    for pair in pairs:
        counts[(pair.expected, pair.invoked)] += pair.probes
        if pair.queries:
            sample = pair.queries[0]
            if sample.reasoning:
                traces = " ".join(sample.reasoning)
                query_samples[(pair.expected, pair.invoked)] = f"{sample.text}\n\nthought: {traces}"
            else:
                query_samples[(pair.expected, pair.invoked)] = sample.text
        expected_labels.add(pair.expected)
        invoked_labels.add(pair.invoked)
    rows = sorted(expected_labels)
    cols = sorted(invoked_labels)
    head = "".join(f"<th>{_esc(col)}</th>" for col in cols)
    body_rows = [
        (
            f"<tr><th>{_esc(row)}</th>"
            f"{''.join(_confusion_cell(row, col, counts, query_samples) for col in cols)}</tr>"
        )
        for row in rows
    ]
    return (
        '<div class="matrix-container">'
        '<table class="confusion" id="confusion-table">'
        f"<thead><tr><th>expected \\ invoked</th>{head}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        "</table>"
        "</div>"
    )


def _collisions_html(artifact: Artifact) -> str:
    """Generate HTML table detailing misrouting collisions and reasoning traces."""
    collisions = [
        p for p in artifact.confusion if p.invoked not in (p.expected, NO_SKILL) and p.probes > 0
    ]
    if not collisions:
        return ""
    rows = []
    for pair in collisions:
        for q in pair.queries:
            thought_html = (
                f'<div class="thought">thought: {_esc(" ".join(q.reasoning))}</div>'
                if q.reasoning
                else ""
            )
            rows.append(
                "<tr>"
                f"<td>{_esc(pair.expected)}</td>"
                f"<td>{_esc(pair.invoked)}</td>"
                f"<td>{q.probes}</td>"
                f"<td>{_esc(q.text)}{thought_html}</td>"
                "</tr>"
            )
    if not rows:
        return ""
    return f"""
<h2>Collisions</h2>
<table class="collisions" id="collisions-table">
  <thead>
    <tr><th>expected</th><th>invoked</th><th>n</th><th>query</th></tr>
  </thead>
  <tbody>{"".join(rows)}</tbody>
</table>
"""


def _skills_table_html(skills: Sequence[SkillScore]) -> str:
    """Generate a filterable HTML table of resident skills with recall and precision."""
    ordered = sorted(
        skills,
        key=lambda s: (s.recall is None, s.recall or 0.0, -s.absorbed, s.skill),
    )
    body_rows = []
    for skill in ordered:
        reached = (
            f"{skill.reached}/{skill.probes}" if skill.probes else '<span class="dim">—</span>'
        )
        recall_str = (
            _pct(skill.recall) if skill.recall is not None else '<span class="dim">—</span>'
        )
        ci = _bounds(skill.recall_interval)
        ci_str = _esc(ci) if ci else '<span class="dim">—</span>'
        prec_str = (
            _pct(skill.precision) if skill.precision is not None else '<span class="dim">—</span>'
        )
        f1_str = _pct(skill.f1) if skill.f1 is not None else '<span class="dim">—</span>'
        body_rows.append(
            "<tr>"
            f"<td>{_esc(skill.skill)}</td>"
            f"<td>{recall_str}</td>"
            f"<td>{ci_str}</td>"
            f"<td>{reached}</td>"
            f"<td>{skill.absorbed}</td>"
            f"<td>{prec_str}</td>"
            f"<td>{f1_str}</td>"
            "</tr>",
        )
    return f"""
<input
  type="text"
  class="filter"
  placeholder="Filter skills by name..."
  oninput="reachFilterRows(this, 'skills-table')"
  aria-controls="skills-table"
/>
<table class="skills" id="skills-table">
  <thead>
    <tr>
      <th>skill</th><th>recall</th><th>CI</th><th>reached</th>
      <th>absorbed</th><th>precision</th><th>F1</th>
    </tr>
  </thead>
  <tbody>{"".join(body_rows)}</tbody>
</table>
"""


def _leak_description(leak: Leak | None) -> str:
    """Format leak evaluation outcome or routes for query detail display."""
    if leak is None:
        return "not evaluated"
    if not leak.leaked:
        return "clean"
    return "; ".join(_esc(route) for route in leak.routes)


def _query_detail_html(record: QueryRecord, catalog_size: int) -> str:
    """Generate expanded detail HTML for an individual QueryRecord."""
    rank = (
        f"{record.difficulty_rank}/{catalog_size}"
        if record.difficulty_rank is not None
        else "unranked"
    )
    selections = ", ".join(_esc(s) for s in record.selections) or "none"
    leak = _leak_description(record.leak)
    return f"""
<dl>
  <dt>text</dt><dd>{_esc(record.text)}</dd>
  <dt>kind</dt><dd>{_esc(record.kind.value) if record.kind else "unset"}</dd>
  <dt>expected</dt><dd>{_esc(record.expected_skill)}</dd>
  <dt>difficulty rank</dt><dd>{_esc(rank)}</dd>
  <dt>leak</dt><dd>{leak}</dd>
  <dt>selections</dt><dd>{selections}</dd>
</dl>
"""


def _queries_html(artifact: Artifact) -> str:
    """Generate collapsible HTML detail elements for each query in the artifact."""
    if not artifact.queries:
        return '<p class="empty">No queries were recorded.</p>'
    entries = []
    for record in artifact.queries:
        rate = f"{record.hits}/{record.probes}" if record.probes else "unprobed"
        css = (
            "unprobed"
            if record.probes == 0
            else ("hit" if record.hits == record.probes else "miss" if record.hits else "error")
        )
        expected_attr = _esc(record.expected_skill)
        selections_attr = _esc(",".join(record.selections))
        entries.append(
            f'<details class="query {css}" data-expected="{expected_attr}" '
            f'data-selections="{selections_attr}">'
            "<summary>"
            f'<span class="query-id">{_esc(record.query_id)}</span>'
            f'<span class="expected">{_esc(record.expected_skill)}</span>'
            f'<span class="query-text">{_esc(record.text)}</span>'
            f'<span class="rate">{_esc(rate)}</span>'
            "</summary>"
            f"{_query_detail_html(record, artifact.catalog_size)}"
            "</details>",
        )
    return f"""
<div class="filter-row">
  <input
    type="text"
    class="filter"
    placeholder="Filter queries by ID, expected skill, or text..."
    oninput="reachFilterQueries(this, 'queries-container')"
    aria-controls="queries-container"
  />
  <div class="expand-controls">
    <button type="button" class="btn-link" onclick="reachToggleAllQueries(true)">
      Expand all
    </button>
    &middot;
    <button type="button" class="btn-link" onclick="reachToggleAllQueries(false)">
      Collapse all
    </button>
  </div>
</div>
<div id="queries-container">{"".join(entries)}</div>
"""


#: Inlined CSS styling for the self-contained HTML diagnostic workbench report.
_STYLE = get_view_css()

#: Inlined client-side JavaScript for dynamic table row, matrix cross-filtering, and queries.
_SCRIPT = get_view_js()


def render_view_html(artifact: Artifact) -> str:
    """Render a complete Artifact model as an interactive diagnostic workbench HTML document."""
    return get_view_template().format(
        title=_esc(artifact.catalog_id),
        style=_STYLE,
        header=_header_html(artifact),
        confusion_matrix=_confusion_html(artifact),
        collisions=_collisions_html(artifact),
        skills_table=_skills_table_html(artifact.skills),
        queries=_queries_html(artifact),
        script=_SCRIPT,
    )


#: Supported format handlers for artifact visualization.
VIEW_RENDERERS = {
    "html": render_view_html,
}


def render_view(artifact: Artifact, fmt: str) -> str:
    """Render an artifact into the requested visualization format."""
    return dispatch_render(VIEW_RENDERERS, fmt, artifact)
