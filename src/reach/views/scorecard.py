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

"""Render evaluation scorecard, skill rankings, confusion pairs, and query records."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich import box
from rich.table import Table
from rich.text import Text

from reach.uncertainty import DEFAULT_CONFIDENCE, Interval
from reach.views.base import _cell, _digest, _TruncatedName

if TYPE_CHECKING:
    from collections.abc import Sequence

    from rich.console import Console

    from reach.artifact import Artifact, ConfusionPair, QueryRecord, SkillScore


def print_scorecard(
    console: Console,
    artifact: Artifact,
    *,
    verbose: bool = False,
) -> None:
    """Print complete evaluation scorecard, skill breakdown, and confusion."""
    console.print(_scorecard_heading(artifact))
    if _worst_first(artifact.skills):
        console.print(_skill_table(artifact))
    else:
        unnamed = sum(1 for s in artifact.skills if not s.probes and not s.absorbed)
        if unnamed:
            console.print(
                Text(
                    f"  {unnamed} resident skills had no query and took no traffic",
                    style="reach.digest",
                ),
                soft_wrap=True,
            )
    console.print(_run_figures(artifact), soft_wrap=True)
    collisions = _collisions(artifact)
    if collisions:
        console.print(_collision_table(collisions, show_reasoning=verbose))
    console.print(_scorecard_provenance(artifact, verbose=verbose), soft_wrap=True)


def _scorecard_heading(artifact: Artifact) -> Text:
    """Assemble the top-level scorecard summary banner."""
    mode = artifact.catalog_mode
    return Text.assemble(
        (artifact.catalog_id, "reach.catalog"),
        f": {artifact.catalog_size} skills, ",
        *([(f"{mode}, ", "reach.label")] if mode != artifact.catalog_id else []),
        f"{artifact.probes} probes",
        *([(f", {artifact.errors} errored", "reach.error")] if artifact.errors else []),
    )


def _skill_table(artifact: Artifact) -> Table:
    """Construct Table of per-skill reached counts, recall, confidence intervals, and precision."""
    shown = _worst_first(artifact.skills)
    confidence = next(
        (s.recall_interval.confidence for s in shown if s.recall_interval),
        DEFAULT_CONFIDENCE,
    )
    table = Table(box=box.SIMPLE, pad_edge=False, header_style="reach.label")
    table.add_column("skill")
    table.add_column("reached", justify="right")
    table.add_column("recall", justify="right")
    table.add_column(f"{confidence:.0%} CI", justify="right")
    table.add_column("absorbed", justify="right")
    table.add_column("precision", justify="right")
    for skill in shown:
        reached = f"{skill.reached}/{skill.probes}" if skill.probes else ""
        table.add_row(
            _TruncatedName(skill.skill, style=_skill_style(skill)),
            reached,
            _rate(skill.recall),
            Text(_bounds(skill.recall_interval), style="reach.digest"),
            str(skill.absorbed) if skill.absorbed else "",
            _rate(skill.precision),
        )
    unnamed = sum(1 for s in artifact.skills if not s.probes and not s.absorbed)
    if unnamed:
        table.caption = f"{unnamed} resident skills had no query and took no traffic"
        table.caption_style = "reach.digest"
    return table


def _worst_first(skills: Sequence[SkillScore]) -> list[SkillScore]:
    """Sort evaluated skills prioritizing low recall, high absorption, and name."""
    shown = [s for s in skills if s.probes or s.absorbed]
    return sorted(
        shown,
        key=lambda s: (
            s.recall if s.recall is not None else 0.0,
            -s.absorbed,
            s.skill,
        ),
    )


def _skill_style(skill: SkillScore) -> str:
    """Return styling class for a skill score row based on its performance."""
    if skill.probes and skill.recall == 0.0:
        return "reach.error"
    if not skill.probes:
        return "reach.misroute"
    return ""


def _run_figures(artifact: Artifact) -> Text:
    """Format run-level summary metrics including consistency, top-1, and abstention."""
    scores = artifact.scores
    spread = artifact.spread
    error = f" ± {spread.standard_error * 100:.1f}pp" if spread.standard_error else ""
    if spread.repeated_queries == 0:
        consistency_parts: list[str | tuple[str, str]] = [
            ("consistency ", "reach.label"),
            ("—", "reach.digest"),
        ]
    else:
        consistency_parts = [
            ("consistency ", "reach.label"),
            (_rate(scores.consistency).strip(), "reach.count"),
            (_beside(scores.consistency_interval), "reach.digest"),
        ]
    parts: list[str | tuple[str, str]] = [
        *consistency_parts,
        ("   top-1 ", "reach.label"),
        f"{scores.top1_accuracy * 100:.1f}%{error}",
    ]
    parts.extend(
        [
            ("   abstention ", "reach.label"),
            _rate(scores.abstention.rate).strip(),
            (_beside(scores.abstention.interval), "reach.digest"),
            (f" ({_rate(scores.abstention.false_rate).strip()} false)", "reach.digest"),
        ]
    )

    attempts_label = "attempt" if spread.replicates == 1 else "attempts"
    parts.append(
        (
            (
                f"\nmacro-F1 {scores.not_headline.macro_f1 * 100:.1f}%"
                f"  over {len(scores.not_headline.labels)} observed labels"
                f"  [{spread.replicates} {attempts_label},"
                f" {spread.repeated_queries} repeated queries]"
            ),
            "reach.digest",
        )
    )
    return Text.assemble(*parts)


def _collisions(artifact: Artifact) -> list[ConfusionPair]:
    """Extract confusion matrix pairs where misrouting collisions occurred."""
    return sorted(
        (pair for pair in artifact.confusion if pair.collisions),
        key=lambda pair: (-pair.collisions, pair.expected, pair.invoked),
    )


def _collision_table(
    pairs: Sequence[ConfusionPair],
    *,
    show_reasoning: bool = False,
) -> Table:
    """Construct a Table detailing confusion pairs and query snippets."""
    table = Table(box=box.SIMPLE, pad_edge=False, header_style="reach.label")
    table.add_column("expected")
    table.add_column("invoked")
    table.add_column("n", justify="right")
    table.add_column("query")
    for pair in pairs:
        sample = pair.queries[0] if pair.queries else None
        quoted = sample.text if sample else ""
        table.add_row(
            _cell(pair.expected),
            _cell(pair.invoked, style="reach.misroute"),
            _cell(str(pair.collisions)),
            _cell(quoted, style="reach.digest"),
        )
        if show_reasoning and sample and sample.reasoning:
            for trace in sample.reasoning:
                table.add_row(
                    "",
                    "",
                    "",
                    _cell(f"thought: {trace}", style="italic dim"),
                )
    return table


def _scorecard_provenance(artifact: Artifact, *, verbose: bool = False) -> Text:
    """Format run provenance metadata and configuration digests."""
    digests = artifact.digests
    roots = ", ".join(f"{r.path} ({r.skills})" for r in artifact.resolved_roots)
    tag = f"{digests.tag} | " if digests.tag else ""
    return Text.assemble(
        (
            (
                f"{artifact.provenance.runtime}/{artifact.provenance.model}"
                f" x{artifact.provenance.attempts}"
                f"  [{tag}arm {_digest(artifact.provenance.arm, verbose=verbose)}"
                f" corpus {_digest(digests.corpus_digest, verbose=verbose)}"
                f" truth {_digest(digests.queries_digest, verbose=verbose)}]"
            ),
            "reach.digest",
        ),
        *([("\n" + roots, "reach.digest")] if roots else []),
    )


def _beside(interval: Interval | None) -> str:
    """Format confidence interval bounds inside square brackets."""
    bounds = _bounds(interval)
    return f" [{bounds}]" if bounds else ""


def _bounds(interval: Interval | None) -> str:
    """Format interval low and high percentages as a range string."""
    if interval is None:
        return ""
    return f"{interval.low * 100:.0f}-{interval.high * 100:.0f}%"


def _rate(value: float | None) -> str:
    """Format a floating-point rate as a percentage string or empty text if None."""
    return "" if value is None else f"{value * 100:.0f}%"


def print_query_records(console: Console, artifact: Artifact) -> None:
    """Render query evaluations grouped by target skill."""
    if not artifact.queries:
        return

    groups: dict[str, list[QueryRecord]] = {}
    for record in artifact.queries:
        groups.setdefault(record.expected_skill, []).append(record)

    skill_order = {s.skill: i for i, s in enumerate(_worst_first(artifact.skills))}
    sorted_expected = sorted(groups.keys(), key=lambda name: (skill_order.get(name, 999), name))

    first = True
    for expected in sorted_expected:
        records = groups[expected]
        if not first:
            console.print()
        first = False

        hits = sum(r.hits for r in records)
        probes = sum(r.probes for r in records)
        pct = f"{hits / probes * 100:.0f}%" if probes else "-"
        heading = Text.assemble(
            ("● ", "reach.label"),
            (expected, "bold reach.catalog"),
            (f"  {hits}/{probes} reached ({pct})", "reach.digest"),
        )
        console.print(heading)

        table = Table(box=box.SIMPLE, pad_edge=False, header_style="reach.label")
        table.add_column("query")
        table.add_column("reached", justify="right")
        table.add_column("rank", justify="right")
        table.add_column("leak")
        table.add_column("selected")
        table.add_column("text", style="reach.digest")

        for record in records:
            selected_cell: Text | _TruncatedName
            if record.clean:
                selected_cell = Text("✓ match", style="reach.hit")
            elif record.probes == 0:
                selected_cell = Text("-", style="reach.digest")
            else:
                rivals = [s for s in record.selections if s != record.expected_skill]
                disp = ", ".join(rivals) if rivals else ", ".join(record.selections)
                selected_cell = _TruncatedName(disp, style="reach.error")

            table.add_row(
                _cell(record.query_id),
                Text.assemble(
                    (f"{record.hits}/{record.probes}", _record_style(record)),
                    (_beside(record.interval), "reach.digest"),
                ),
                _cell(_ranked(record.difficulty_rank, artifact.catalog_size)),
                _cell("clean", style="reach.hit")
                if record.leak and not record.leak.leaked
                else _cell("; ".join(record.leak.routes), style="reach.misroute")
                if record.leak
                else _cell(""),
                selected_cell,
                Text(record.text, style="reach.digest"),
            )
        console.print(table)


def _record_style(record: QueryRecord) -> str:
    """Return styling class based on hits and probes in a query record."""
    if record.probes == 0:
        return "reach.label"
    if record.hits == record.probes:
        return "reach.hit"
    return "reach.misroute" if record.hits else "reach.error"


def _ranked(position: int | None, field: int) -> str:
    """Format a lexical rank as a position/field string."""
    return "" if position is None else f"{position}/{field}"
