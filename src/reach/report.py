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

"""Format and render evaluation outcomes directly from canonical Artifact models."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from reach.artifact import Artifact, QueryRecord
from reach.models import ProbeResult
from reach.rendering import csv_document, dispatch_render
from reach.run import Composition
from reach.uncertainty import Interval

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


def _pct(value: float) -> str:
    """Format a floating-point ratio as a percentage string with 1 decimal place."""
    return f"{value * 100:5.1f}%"


def _ci(interval: Interval | None) -> str:
    """Format an Interval as a bracketed percentage range string."""
    if interval is None:
        return ""
    return f"  [{interval.low * 100:.1f}-{interval.high * 100:.1f}]"


def _rank(value: int | None) -> str:
    """Format a lexical difficulty rank as fixed-width L<position> string."""
    return f"L{value:<6}" if value is not None else "       "


def format_pairs(
    pairs: Sequence[tuple[str, str, int]],
    title: str,
) -> str:
    """Format pair transition frequency counts into plain text."""
    if not pairs:
        return f"{title}\n  (none)"
    rows = [f"  {expected} -> {invoked}  x{count}" for expected, invoked, count in pairs]
    return "\n".join([title, *rows])


def per_query_lines(artifact: Artifact) -> list[str]:
    """Format per-query accuracy and lexical difficulty into aligned text rows."""
    width = max((len(q.query_id) for q in artifact.queries), default=8)
    return [
        f"{' ' if q.clean else '!'} {q.query_id:<{width}} {q.hits}/{q.probes}"
        f"{_ci(q.interval):<16}  {_rank(q.difficulty_rank)}  {(q.kind or ''):<18} "
        f"{', '.join(q.selections)}"
        for q in artifact.queries
    ]


def _render_provenance_lines(artifact: Artifact) -> list[str]:
    """Format provenance metadata lines."""
    lines = [
        f"  runtime           {artifact.provenance.runtime or '(unknown)'}",
        f"  model             {artifact.provenance.model or '(unknown)'}"
        + (
            f"  -> {artifact.provenance.resolved_model}"
            if artifact.provenance.resolved_model
            else ""
        ),
        (
            f"  catalog           {artifact.catalog_id or '(unknown)'} "
            f"({artifact.catalog_size} skills)"
        ),
        f"  config            {artifact.digests.config_fingerprint or '(unrecorded)'}",
        f"  corpus            {artifact.digests.corpus_digest or '(unrecorded)'}",
        f"  ground truth      {artifact.digests.queries_digest or '(unrecorded)'}",
    ]
    if artifact.reused:
        lines.append(f"  reused            {artifact.reused} prior probes")
    return lines


def _render_classification_header(artifact: Artifact) -> list[str]:
    """Format classification overview metric lines."""
    scores = artifact.scores
    header = [
        "Classification report",
        f"  probes            {artifact.probes}",
        f"  errors            {artifact.errors}",
        f"  scored            {scores.scored}",
        f"  top-1 accuracy    {_pct(scores.top1_accuracy)}{_ci(scores.top1_interval)}",
    ]
    if scores.scored > 0:
        header.extend(
            [
                (
                    f"  entrypoint acc.   {_pct(scores.entrypoint_accuracy)}"
                    f"{_ci(scores.entrypoint_interval)}"
                ),
                (
                    f"  reachability      {_pct(scores.trajectory_reachability)}"
                    f"{_ci(scores.trajectory_interval)}"
                ),
                (
                    f"  step efficiency   {scores.step_efficiency:6.3f}"
                    if scores.step_efficiency is not None
                    else "  step efficiency      n/a"
                ),
                (
                    f"  skill F1          {_pct(scores.skill_f1)}"
                    if scores.skill_f1 is not None
                    else "  skill F1             n/a"
                ),
                f"  redundancy        +{scores.redundancy:.2f}",
            ]
        )
    header.extend(
        [
            f"  macro-F1          {_pct(scores.not_headline.macro_f1)}",
            f"  macro precision   {_pct(scores.not_headline.macro_precision)}",
            f"  macro recall      {_pct(scores.not_headline.macro_recall)}",
            f"  abstention        {_pct(scores.abstention.rate)}{_ci(scores.abstention.interval)}",
            (
                f"  false abstention  {_pct(scores.abstention.false_rate)}"
                f"{_ci(scores.abstention.false_interval)}"
            ),
        ]
    )
    if scores.abstention.out_of_scope:
        oos = (
            _pct(scores.abstention.out_of_scope_detection)
            if scores.abstention.out_of_scope_detection is not None
            else "    n/a"
        )
        header.append(f"  out-of-scope det. {oos}{_ci(scores.abstention.out_of_scope_interval)}")
    return header


def _render_classification_section(artifact: Artifact) -> str:
    """Format full classification report table and metrics."""
    header = _render_classification_header(artifact)
    width = max((len(s.skill) for s in artifact.skills), default=5)
    columns = (
        f"  {'label':<{width}} {'prec':>6} {'rec':>6} {'F1':>6} {'n':>4}  {'95% CI on rec':>15}"
    )
    skill_rows = [
        f"  {s.skill:<{width}} "
        f"{(f'{s.precision:6.2f}' if s.precision is not None else '   n/a')} "
        f"{(f'{s.recall:6.2f}' if s.recall is not None else '   n/a')} "
        f"{(f'{s.f1:6.2f}' if s.f1 is not None else '   n/a')} "
        f"{s.probes:4d} {_ci(s.recall_interval):>17}"
        for s in artifact.skills
    ]
    return "\n".join([*header, "", columns, *skill_rows])


def render_text(artifact: Artifact) -> str:
    """Render an Artifact into a human-readable text document for console output."""
    provenance = _render_provenance_lines(artifact)
    classification_section = _render_classification_section(artifact)
    scores = artifact.scores

    stability = [
        (
            f"  consistency       {_pct(scores.consistency)}"
            f"  ({scores.unanimous_queries}/{scores.observed_queries} queries)"
            f"{_ci(scores.consistency_interval)}"
        ),
    ]

    collisions = [(p.expected, p.invoked, p.collisions) for p in artifact.confusion if p.collisions]
    title = f"{artifact.catalog_id} on {artifact.provenance.model or '(unknown)'}"
    sections = [
        title,
        "\n".join(["Provenance", *provenance]),
        classification_section,
        "\n".join(["Stability", *stability]),
        "\n".join(
            [
                "Per query (flagged lines are not clean)",
                *per_query_lines(artifact),
            ]
        ),
        format_pairs(
            collisions,
            "Collisions (misroutes to other skills)",
        ),
    ]
    if artifact.spend_usd:
        sections.append(f"Spend  ${artifact.spend_usd:.2f}")

    return "\n\n".join(sections)


def _query_row(record: QueryRecord) -> list[object]:
    """Extract CSV row values from a single QueryRecord object."""
    interval = record.interval
    return [
        record.query_id,
        record.kind or "",
        record.expected_skill,
        record.hits,
        record.probes,
        "yes" if record.clean else "no",
        f"{interval.low:.4f}" if interval else "",
        f"{interval.high:.4f}" if interval else "",
        record.difficulty_rank if record.difficulty_rank is not None else "",
        "",
        " ".join(record.selections),
    ]


def render_csv(artifact: Artifact) -> str:
    """Export per-query outcomes from an Artifact as formatted CSV."""
    return csv_document(
        [
            "query_id",
            "kind",
            "expected",
            "hits",
            "scored",
            "clean",
            "ci_low",
            "ci_high",
            "lexical_rank",
            "lexical_field",
            "selections",
        ],
        (_query_row(record) for record in artifact.queries),
    )


def render_json(artifact: Artifact) -> str:
    """Serialize an Artifact to formatted JSON string."""
    return artifact.model_dump_json(indent=2)


def render_jsonl(artifact: Artifact) -> str:
    """Render each per-query record in Artifact as JSONL line."""
    return "".join(record.model_dump_json() + "\n" for record in artifact.queries)


#: Supported report export format handlers.
RENDERERS = {
    "text": render_text,
    "json": render_json,
    "jsonl": render_jsonl,
    "csv": render_csv,
}


def render(artifact: Artifact, fmt: str = "text") -> str:
    """Render an Artifact into the requested format (text, json, jsonl, csv)."""
    return dispatch_render(RENDERERS, fmt, artifact)


def build_report(
    composition: Composition,
    results: Sequence[ProbeResult],
    *,
    spend_usd: float | None = None,
    reused: int = 0,
    difficulty: Mapping[str, Any] | None = None,
    cross_check: bool = True,
) -> Artifact:
    """Assemble an evaluation Artifact from a Composition and recorded results."""
    effective_spend = (
        spend_usd if spend_usd is not None else sum(r.cost_usd or 0.0 for r in results)
    )
    return Artifact.assemble(
        composition,
        results,
        spend_usd=effective_spend,
        reused=reused,
        difficulty=difficulty or {},
        cross_check=cross_check,
    )


def format_run(
    composition: Composition,
    results: Sequence[ProbeResult],
) -> str:
    """Build and render an evaluation run directly as text."""
    return render_text(build_report(composition, results))
