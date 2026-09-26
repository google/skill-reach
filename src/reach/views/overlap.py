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

"""Render lexical overlap tables, competition rankings, rewrites, and query attributions."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Annotated, Final

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field
from rich import box
from rich.table import Table
from rich.text import Text

from reach.config import OverlapSettings
from reach.overlap import OVERLAP_CAVEAT, Standing
from reach.rendering import csv_document, dispatch_render
from reach.retrieval import OverlapQuadrant, classify_overlap_quadrant, tokenize
from reach.rewrite import REWRITE_CAVEAT, CededTerm, Verdict, suggest_all
from reach.views.base import _cell, _TruncatedName

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from rich.console import Console, ConsoleOptions, RenderResult

    from reach.attribution import QueryAttribution
    from reach.models import Skill
    from reach.overlap import Competition, CorpusOverlap
    from reach.rewrite import Rewrite

#: Default maximum rows shown in text mode for large corpora unless --all or --top is set.
DEFAULT_OVERLAP_TOP: Final = 30


def _parse_quadrant_item(raw: str | OverlapQuadrant) -> OverlapQuadrant:
    """Coerce a single quadrant string or enum into an OverlapQuadrant member."""
    try:
        return OverlapQuadrant(raw)
    except ValueError as exc:
        valid = ", ".join("-".join(tokenize(m.name.replace("_", "-"))) for m in OverlapQuadrant)
        msg = f"unknown quadrant {raw!r}; expected one of: {valid}"
        raise ValueError(msg) from exc


def _parse_quadrants_field(raw: object) -> frozenset[OverlapQuadrant]:
    """Normalize CLI strings or sequences into a validated frozenset of OverlapQuadrant."""
    if raw is None:
        return frozenset()
    if isinstance(raw, str):
        parts = [p.strip() for p in raw.split(",") if p.strip()]
    elif isinstance(raw, (list, tuple, set, frozenset)):
        parts = [p.strip() for entry in raw for p in str(entry).split(",") if p.strip()]
    else:
        msg = f"invalid quadrant filter type: {type(raw).__name__}"
        raise TypeError(msg)
    return frozenset(_parse_quadrant_item(p) for p in parts)


def _lookup_similarity(
    skill: str,
    rival: str,
    semantic_similarities: Mapping[tuple[str, str], float] | None,
) -> float | None:
    """Look up bidirectional pairwise semantic similarity with finite validation."""
    if semantic_similarities is None:
        return None
    sim = semantic_similarities.get((skill, rival))
    if sim is None:
        sim = semantic_similarities.get((rival, skill))
    if sim is None or not math.isfinite(sim):
        return None
    return sim


class OverlapFilter(BaseModel):
    """Specify row filtering and slicing criteria for overlap views and reports."""

    model_config = ConfigDict(frozen=True)

    top: Annotated[int | None, Field(ge=1)] = None
    all_skills: bool = False
    quadrants: Annotated[
        frozenset[OverlapQuadrant],
        BeforeValidator(_parse_quadrants_field),
    ] = frozenset()


def filter_competitions(
    competitions: Sequence[Competition],
    *,
    semantic_similarities: Mapping[tuple[str, str], float] | None = None,
    quadrants: frozenset[OverlapQuadrant] = frozenset(),
    top: int | None = None,
) -> tuple[list[Competition], int]:
    """Filter competitions by quadrant and slice to top N, returning (sliced, total_matching)."""
    if quadrants:
        matched: list[Competition] = []
        for comp in competitions:
            nearest = comp.scoring_rival
            if nearest is None or semantic_similarities is None:
                continue
            sim = _lookup_similarity(comp.skill, nearest.name, semantic_similarities)
            if sim is None:
                continue
            quad = classify_overlap_quadrant(comp.rival_ratio, sim)
            if quad in quadrants:
                matched.append(comp)
    else:
        matched = list(competitions)

    total_matching = len(matched)
    shown = matched[:top] if top is not None else matched
    return shown, total_matching


def filter_rewrites(
    overlap: CorpusOverlap,
    skills: Sequence[Skill],
    names: Sequence[str] = (),
    *,
    overlap_filter: OverlapFilter | None = None,
    semantic_similarities: Mapping[tuple[str, str], float] | None = None,
    default_top: int | None = None,
    overlap_settings: OverlapSettings | None = None,
) -> tuple[tuple[Rewrite, ...], int]:
    """Filter and slice rewrite proposals, returning (shown_rewrites, total_matching)."""
    filt = overlap_filter or OverlapFilter()
    cfg = overlap_settings or OverlapSettings()
    if names:
        target_names: Sequence[str] = names
        only_actionable = False
    else:
        filtered_comps, _ = filter_competitions(
            overlap.competitions,
            semantic_similarities=semantic_similarities,
            quadrants=filt.quadrants,
            top=None,
        )
        target_names = [c.skill for c in filtered_comps]
        only_actionable = not filt.all_skills

    rewrites = suggest_all(
        overlap,
        skills,
        target_names,
        only_actionable=only_actionable,
        settings=cfg,
    )
    total_matching = len(rewrites)
    effective_top = filt.top if filt.top is not None else (None if filt.all_skills else default_top)
    shown = rewrites[:effective_top] if effective_top is not None else rewrites
    return shown, total_matching


def print_no_actionable_rewrites(console: Console, corpus_size: int) -> None:
    """Print notice when corpus-wide --suggest finds no actionable rewrites."""
    console.print(
        Text(
            f"No actionable rewrites across {corpus_size} skills "
            f"(pass --all to view all {corpus_size} skills).",
            style="reach.label",
        ),
        soft_wrap=True,
    )


_QUADRANT_STYLES: Final[dict[OverlapQuadrant, str]] = {
    OverlapQuadrant.LATENT_COLLISION: "bold red",
    OverlapQuadrant.NEAR_DUPLICATE: "bold yellow",
    OverlapQuadrant.BOILERPLATE: "dim",
    OverlapQuadrant.DISTINCT: "green",
}


def _format_quadrant_badge(quad: OverlapQuadrant) -> Text:
    """Format an overlap quadrant enum with semantic color styling."""
    return Text(quad.value, style=_QUADRANT_STYLES.get(quad, "green"))


def _format_overlap_cells(
    skill: str,
    rival: str,
    ratio: float,
    semantic_similarities: Mapping[tuple[str, str], float] | None,
) -> tuple[list[Text], list[Text]]:
    """Format similarity and quadrant cells for overlap table rows."""
    if semantic_similarities is None:
        return [], []

    sim = _lookup_similarity(skill, rival, semantic_similarities)
    if sim is None:
        return [_cell("—")], [_cell("—")]
    quad = classify_overlap_quadrant(ratio, sim)
    return [_cell(f"{sim:.1%}")], [_format_quadrant_badge(quad)]


def print_omitted_footer(
    console: Console,
    omitted_count: int,
    total_matching: int,
) -> None:
    """Print standardized footer when rows are omitted due to --top or default cap."""
    if omitted_count <= 0:
        return
    console.print(
        Text(
            f"… and {omitted_count} more skill{'' if omitted_count == 1 else 's'} omitted "
            f"(pass --all or --top N to view all {total_matching})",
            style="reach.label",
        ),
        soft_wrap=True,
    )


class _UnconstrainedTable:
    """Render a Rich Table at its natural width without terminal-width column compression."""

    def __init__(self, table: Table) -> None:
        self.table = table

    def __rich_console__(
        self,
        console: Console,
        options: ConsoleOptions,
    ) -> RenderResult:
        natural = console.measure(
            self.table,
            options=options.update_width(10_000),
        ).maximum
        yield from console.render(
            self.table,
            options.update_width(max(options.max_width, natural)),
        )


def _print_overlap_table(console: Console, table: Table, *, truncate: bool) -> None:
    """Print an overlap table, expanding to natural column widths when truncate is False."""
    if truncate:
        console.print(table)
    else:
        console.print(_UnconstrainedTable(table), crop=False, soft_wrap=True)


def print_overlap(
    console: Console,
    overlap: CorpusOverlap,
    semantic_similarities: Mapping[tuple[str, str], float] | None = None,
    *,
    competitions: Sequence[Competition] | None = None,
    omitted_count: int = 0,
    total_matching: int | None = None,
    truncate: bool = True,
    caveat: bool = True,
) -> None:
    """Render Table ranking skills by lexical overlap and optional semantic similarity."""
    corpus_total = len(overlap.competitions)
    rows = tuple(competitions) if competitions is not None else overlap.competitions
    matching_total = (
        total_matching
        if total_matching is not None
        else (len(rows) + omitted_count if omitted_count > 0 else corpus_total)
    )
    effective_omitted = omitted_count if omitted_count > 0 else max(0, matching_total - len(rows))
    all_names = tuple(c.skill for c in overlap.competitions)

    suffix_label = (
        f" skill{'' if corpus_total == 1 else 's'}, ranked by how hard each is competed for"
    )
    if len(rows) < corpus_total:
        console.print(
            Text.assemble(
                "Showing ",
                (str(len(rows)), "reach.count"),
                " of ",
                (str(corpus_total), "reach.count"),
                suffix_label,
            ),
            soft_wrap=True,
        )
    else:
        console.print(
            Text.assemble(
                (str(corpus_total), "reach.count"),
                suffix_label,
            ),
            soft_wrap=True,
        )
    beaten = any(c.outranked_by for c in rows)
    table = Table(box=box.SIMPLE, pad_edge=False, header_style="reach.label")

    table.add_column("skill")
    table.add_column("overlap", justify="right")
    if semantic_similarities is not None:
        table.add_column("semantic sim", justify="right")
    table.add_column("nearest rival")
    if semantic_similarities is not None:
        table.add_column("quadrant", no_wrap=True)
    if beaten:
        table.add_column("")
    for competition in rows:
        nearest = competition.scoring_rival
        rival = nearest.name if nearest is not None else ""
        sem_cells, quad_cells = _format_overlap_cells(
            competition.skill,
            rival,
            competition.rival_ratio,
            semantic_similarities,
        )
        marker = [_outranked(competition.outranked_by)] if beaten else []
        table.add_row(
            _TruncatedName(competition.skill, peers=all_names, truncate=truncate),
            _cell(f"{competition.rival_ratio:.2f}"),
            *sem_cells,
            _TruncatedName(rival, peers=all_names, truncate=truncate),
            *quad_cells,
            *marker,
        )
    _print_overlap_table(console, table, truncate=truncate)
    print_omitted_footer(console, effective_omitted, matching_total)
    if caveat:
        print_overlap_caveat(console)


def _outranked(count: int) -> Text:
    """Format outranked indicator string for skills beaten on their own vocabulary."""
    if not count:
        return _cell("")
    return _cell(f"outranked by {count}", style="reach.count")


def print_skill_overlap(
    console: Console,
    competition: Competition,
    semantic_similarities: Mapping[tuple[str, str], float] | None = None,
    *,
    caveat: bool = True,
    truncate: bool = True,
    peers: Sequence[str] = (),
) -> None:
    """Render full competition standings table for a single target skill."""
    rivals = len(competition.rivals)
    peer_names = tuple(peers) if peers else tuple(s.name for s in competition.standings)
    console.print(
        Text.assemble(
            (competition.skill, "reach.catalog"),
            ": outranked by ",
            (str(competition.outranked_by), "reach.count"),
            f" of {rivals} rival{'' if rivals == 1 else 's'} for its own vocabulary",
        ),
        soft_wrap=True,
    )
    table = Table(box=box.SIMPLE, pad_edge=False, header_style="reach.label")
    table.add_column("rank", justify="right")
    table.add_column("skill")
    table.add_column("score", justify="right")
    if semantic_similarities is not None:
        table.add_column("semantic sim", justify="right")
        table.add_column("quadrant", no_wrap=True)
    table.add_column("")

    for standing in competition.standings:
        sem_cells: list[Text] = []
        if semantic_similarities is not None:
            if standing.is_target:
                sem_cells = [_cell("—"), _cell("—")]
            else:
                ratio = (
                    standing.score / competition.self_score if competition.self_score > 0 else 0.0
                )
                sem, quad = _format_overlap_cells(
                    competition.skill,
                    standing.name,
                    ratio,
                    semantic_similarities,
                )
                sem_cells = [*sem, *quad]

        table.add_row(
            _cell(str(standing.rank)),
            _TruncatedName(
                standing.name,
                style="reach.catalog" if standing.is_target else "",
                peers=peer_names,
                truncate=truncate,
            ),
            _cell(f"{standing.score:.3f}", style="reach.digest"),
            *sem_cells,
            _cell("this skill" if standing.is_target else "", style="reach.label"),
        )
    _print_overlap_table(console, table, truncate=truncate)
    if caveat:
        print_overlap_caveat(console)


_REWRITE_LABELS = {
    "cede": "drop or qualify",
    "disclaimed": "already disclaimed",
    "unclaimed": "yours alone",
    "contenders": "also within reach",
    "mutual_handoff": "missing mutual handoff",
}


#: Maximum number of extra contender skill names listed inline before "+N more".
_MAX_EXTRA_CONTENDERS: Final = 3


def print_rewrite(
    console: Console,
    rewrite: Rewrite,
    *,
    caveat: bool = True,
) -> None:
    """Render rewrite suggestions and contested vocabulary analysis for a skill."""
    console.print(_rewrite_heading(rewrite), soft_wrap=True)
    extra = [c for c in rewrite.contenders if c != rewrite.rival]
    more_rivals = len(extra)
    extra_names = ", ".join(extra[:_MAX_EXTRA_CONTENDERS]) + (
        f", +{len(extra) - _MAX_EXTRA_CONTENDERS} more"
        if len(extra) > _MAX_EXTRA_CONTENDERS
        else ""
    )
    contender_desc = (
        f"{more_rivals} more rival{'' if more_rivals == 1 else 's'} "
        f"score{'s' if more_rivals == 1 else ''} within {1.0 - rewrite.band:.0%} ({extra_names})"
        if rewrite.crowded
        else ""
    )
    rows = (
        ("cede", ", ".join(f"{c.term} {c.share:.0%}" for c in rewrite.reword)),
        ("disclaimed", ", ".join(c.term for c in rewrite.disclaimed)),
        ("unclaimed", ", ".join(rewrite.unclaimed)),
        ("contenders", contender_desc),
        ("mutual_handoff", ", ".join(rewrite.missing_mutual_handoffs)),
    )
    table = Table(box=box.SIMPLE, pad_edge=False, show_header=False)
    table.add_column("", style="reach.label")
    table.add_column("")
    for field, value in rows:
        if value:
            table.add_row(_cell(_REWRITE_LABELS[field]), Text(value))
    if table.row_count:
        console.print(table)
    if caveat:
        print_overlap_caveat(console)


def _rewrite_heading(rewrite: Rewrite) -> Text:
    """Assemble heading line summarizing rewrite verdict and primary rival."""
    if rewrite.verdict is Verdict.UNRIVALED:
        return Text.assemble(
            (rewrite.skill, "reach.catalog"),
            ": nothing competes for its vocabulary; no rewording needed",
        )
    if rewrite.verdict is Verdict.CONTESTED:
        return Text.assemble(
            (rewrite.skill, "reach.catalog"),
            " and ",
            (rewrite.rival, "reach.catalog"),
            ": no wording change is indicated",
        )
    return Text.assemble(
        (rewrite.skill, "reach.catalog"),
        " cedes ",
        (str(len(rewrite.reword)), "reach.count"),
        f" term{'' if len(rewrite.reword) == 1 else 's'} to ",
        (rewrite.rival, "reach.catalog"),
    )


def print_overlap_caveat(console: Console) -> None:
    """Print the standardized lexical overlap caveat disclaimer."""
    for line in OVERLAP_CAVEAT:
        console.print(Text(line), soft_wrap=True)


def format_annotated_query(attribution: QueryAttribution) -> Text:
    """Format the query text with highlighted driver and anchor tokens."""
    driver_set = {d.token for d in attribution.drivers}
    anchor_set = {a.token for a in attribution.anchors}

    result = Text()
    words = attribution.query_text.split()
    for i, word in enumerate(words):
        cleaned_tokens = tokenize(word)
        if any(t in driver_set for t in cleaned_tokens):
            result.append(word, style="bold red")
        elif any(t in anchor_set for t in cleaned_tokens):
            result.append(word, style="bold green")
        else:
            result.append(word)
        if i < len(words) - 1:
            result.append(" ")
    return result


def render_attribution_table(attribution: QueryAttribution) -> Table:
    """Construct a Rich Table summarizing token driver and anchor scores."""
    if attribution.rival_skill == "(no selection)":
        table = Table(box=box.SIMPLE, pad_edge=False, header_style="reach.label")
        table.add_column("token")
        table.add_column(f"target ({attribution.target_skill})", justify="right")
        table.add_column("role")
        for item in attribution.anchors:
            table.add_row(
                item.token,
                f"{item.target_score:.3f}",
                Text("target match", style="green"),
            )
        for token in attribution.unscored_tokens:
            table.add_row(
                token,
                "0.000",
                Text("target gap (missing)", style="bold red"),
            )
        return table

    table = Table(box=box.SIMPLE, pad_edge=False, header_style="reach.label")
    table.add_column("token")
    table.add_column(f"target ({attribution.target_skill})", justify="right")
    table.add_column(f"rival ({attribution.rival_skill})", justify="right")
    table.add_column("delta", justify="right")
    table.add_column("role")

    for item in attribution.drivers:
        role = "target gap" if item.is_target_gap else "misroute driver"
        role_style = "bold red" if item.is_target_gap else "red"
        table.add_row(
            item.token,
            f"{item.target_score:.3f}",
            f"{item.rival_score:.3f}",
            f"+{item.delta:.3f}",
            Text(role, style=role_style),
        )

    for item in attribution.anchors:
        table.add_row(
            item.token,
            f"{item.target_score:.3f}",
            f"{item.rival_score:.3f}",
            f"{item.delta:.3f}",
            Text("target anchor", style="green"),
        )

    return table


def print_attribution(console: Console, attribution: QueryAttribution) -> None:
    """Render query attribution analysis with annotated prompt and token table."""
    console.print(format_annotated_query(attribution))
    if attribution.rival_skill == "(no selection)":
        console.print(
            f"[dim]target:[/] {attribution.target_skill} "
            f"([dim]affinity: {attribution.target_total:.3f}[/])  "
            f"[dim]rival:[/] [italic](no selection - abstention)[/]",
        )
    else:
        bias_str = (
            f"+{attribution.net_bias:.3f}"
            if attribution.net_bias > 0
            else f"{attribution.net_bias:.3f}"
        )
        console.print(
            f"[dim]target:[/] {attribution.target_skill} ([dim]{attribution.target_total:.3f}[/])  "
            f"[dim]rival:[/] {attribution.rival_skill} ([dim]{attribution.rival_total:.3f}[/])  "
            f"[dim]net bias:[/] [bold]{bias_str}[/]",
        )

    if attribution.target_semantic is not None and attribution.rival_semantic is not None:
        sem_bias = (
            f"+{attribution.semantic_bias:.3f}"
            if attribution.semantic_bias and attribution.semantic_bias > 0
            else f"{attribution.semantic_bias or 0.0:.3f}"
        )
        console.print(
            f"[dim]dense semantic:[/] target [bold]{attribution.target_semantic:.3f}[/]  "
            f"rival [bold]{attribution.rival_semantic:.3f}[/]  "
            f"[dim]semantic bias:[/] [bold]{sem_bias}[/]",
        )
    elif attribution.target_semantic is not None:
        console.print(
            f"[dim]dense semantic:[/] target affinity [bold]{attribution.target_semantic:.3f}[/]",
        )

    if (
        attribution.drivers
        or attribution.anchors
        or (attribution.rival_skill == "(no selection)" and attribution.unscored_tokens)
    ):
        console.print(render_attribution_table(attribution))


class SkillOverlapView(BaseModel):
    """Summarize a single skill's overlap metrics for reporting views."""

    model_config = ConfigDict(frozen=True)

    nearest_rival: str | None = None
    outranked_by: int = 0
    quadrant: OverlapQuadrant | None = None
    rival_ratio: float | None = None
    self_score: float
    semantic_similarity: float | None = None
    skill: str
    standings: tuple[Standing, ...] = ()


class OverlapView(BaseModel):
    """Hold a structured view model of corpus overlap for CLI and export renderers."""

    model_config = ConfigDict(frozen=True)

    caveat: tuple[str, ...] = OVERLAP_CAVEAT
    corpus_size: int
    skills: tuple[SkillOverlapView, ...] = ()


def _viewed_skill_overlap(
    competition: Competition,
    *,
    seated: bool,
    semantic_similarities: Mapping[tuple[str, str], float] | None = None,
) -> SkillOverlapView:
    """Convert a Competition model into a SkillOverlapView."""
    rival = competition.scoring_rival
    ratio = competition.rival_ratio
    sem_sim: float | None = None
    quad: OverlapQuadrant | None = None
    if rival is not None and semantic_similarities:
        sim = _lookup_similarity(competition.skill, rival.name, semantic_similarities)
        if sim is not None:
            sem_sim = round(sim, 4)
            quad = classify_overlap_quadrant(ratio, sim)

    return SkillOverlapView(
        skill=competition.skill,
        self_score=competition.self_score,
        rival_ratio=ratio if math.isfinite(ratio) else None,
        nearest_rival=rival.name if rival is not None else None,
        semantic_similarity=sem_sim,
        quadrant=quad,
        outranked_by=competition.outranked_by,
        standings=competition.standings if seated else (),
    )


def overlap_view(
    overlap: CorpusOverlap,
    skills: Sequence[str] = (),
    *,
    semantic_similarities: Mapping[tuple[str, str], float] | None = None,
    overlap_filter: OverlapFilter | None = None,
) -> OverlapView:
    """Build an OverlapView for all corpus skills or a selected subset."""
    chosen = [overlap.find(name) for name in skills] if skills else None
    if chosen is not None:
        source_competitions: Sequence[Competition] = chosen
    elif overlap_filter is not None and (
        overlap_filter.quadrants or overlap_filter.top is not None
    ):
        source_competitions, _ = filter_competitions(
            overlap.competitions,
            semantic_similarities=semantic_similarities,
            quadrants=overlap_filter.quadrants,
            top=overlap_filter.top,
        )
    else:
        source_competitions = overlap.competitions

    return OverlapView(
        corpus_size=len(overlap.competitions),
        skills=tuple(
            _viewed_skill_overlap(
                competition,
                seated=chosen is not None,
                semantic_similarities=semantic_similarities,
            )
            for competition in source_competitions
        ),
    )


def render_overlap_json(view: OverlapView) -> str:
    """Render an OverlapView as formatted JSON string."""
    return view.model_dump_json(indent=2)


def render_overlap_jsonl(view: OverlapView) -> str:
    """Render an OverlapView as JSONL lines for each skill overlap."""
    return "".join(skill.model_dump_json() + "\n" for skill in view.skills)


def _render_standings_csv(skills: Sequence[SkillOverlapView]) -> str:
    """Render per-skill rank standings as formatted CSV table."""
    return csv_document(
        ["target", "rank", "skill", "score", "is_target"],
        (
            [
                skill.skill,
                standing.rank,
                standing.name,
                f"{standing.score:.4f}",
                "yes" if standing.is_target else "no",
            ]
            for skill in skills
            for standing in skill.standings
        ),
    )


def _render_summary_row(skill: SkillOverlapView, has_semantic: bool) -> list[object]:
    """Format single skill summary metrics row for CSV export."""
    row: list[object] = [
        skill.skill,
        f"{skill.self_score:.4f}",
        f"{skill.rival_ratio:.4f}" if skill.rival_ratio is not None else "",
        skill.nearest_rival or "",
    ]
    if has_semantic:
        sem_str = (
            f"{skill.semantic_similarity:.4f}" if skill.semantic_similarity is not None else ""
        )
        row.extend([sem_str, skill.quadrant or ""])
    row.append(skill.outranked_by)
    return row


def _render_summary_csv(skills: Sequence[SkillOverlapView]) -> str:
    """Render catalog skill overlap summary table as formatted CSV."""
    has_semantic = any(skill.semantic_similarity is not None for skill in skills)
    headers = (
        [
            "skill",
            "self_score",
            "rival_ratio",
            "nearest_rival",
            "semantic_sim",
            "quadrant",
            "outranked_by",
        ]
        if has_semantic
        else ["skill", "self_score", "rival_ratio", "nearest_rival", "outranked_by"]
    )
    return csv_document(headers, (_render_summary_row(s, has_semantic) for s in skills))


def render_overlap_csv(view: OverlapView) -> str:
    """Render an OverlapView as formatted CSV table string."""
    if any(skill.standings for skill in view.skills):
        return _render_standings_csv(view.skills)
    return _render_summary_csv(view.skills)


#: Supported export formats for overlap reports.
OVERLAP_RENDERERS = {
    "csv": render_overlap_csv,
    "json": render_overlap_json,
    "jsonl": render_overlap_jsonl,
}


def render_overlap(view: OverlapView, fmt: str) -> str:
    """Render an OverlapView into the requested format (json, csv)."""
    return dispatch_render(OVERLAP_RENDERERS, fmt, view)


class RewriteView(BaseModel):
    """Summarize a skill rewrite proposal for CLI or export presentation."""

    model_config = ConfigDict(frozen=True)

    skill: str
    verdict: Verdict
    rival: str | None = None
    ceded: tuple[CededTerm, ...] = ()
    unclaimed: tuple[str, ...] = ()
    contenders: tuple[str, ...] = ()
    crowded: bool = False
    missing_mutual_handoffs: tuple[str, ...] = ()


class SuggestView(BaseModel):
    """Hold a collection of rewrite suggestions across target skills."""

    model_config = ConfigDict(frozen=True)

    corpus_size: int
    caveat: tuple[str, ...] = REWRITE_CAVEAT
    skills: tuple[RewriteView, ...] = ()


def _viewed_rewrite(rewrite: Rewrite) -> RewriteView:
    """Convert a Rewrite model into a presentation RewriteView."""
    return RewriteView(
        skill=rewrite.skill,
        verdict=rewrite.verdict,
        rival=rewrite.rival or None,
        ceded=rewrite.ceded,
        unclaimed=rewrite.unclaimed,
        contenders=rewrite.contenders,
        crowded=rewrite.crowded,
        missing_mutual_handoffs=rewrite.missing_mutual_handoffs,
    )


def suggest_view(
    overlap: CorpusOverlap,
    skills: Sequence[Skill],
    names: Sequence[str] = (),
    *,
    overlap_filter: OverlapFilter | None = None,
    semantic_similarities: Mapping[tuple[str, str], float] | None = None,
    overlap_settings: OverlapSettings | None = None,
) -> SuggestView:
    """Build a SuggestView containing proposals for the specified skills."""
    # Structured outputs (JSON/JSONL/CSV) remain uncapped unless --top is explicitly set.
    rewrites, _ = filter_rewrites(
        overlap,
        skills,
        names,
        overlap_filter=overlap_filter,
        semantic_similarities=semantic_similarities,
        default_top=None,
        overlap_settings=overlap_settings,
    )
    return SuggestView(
        corpus_size=len(overlap.competitions),
        skills=tuple(_viewed_rewrite(r) for r in rewrites),
    )


def render_rewrite_json(view: SuggestView) -> str:
    """Serialize a SuggestView to formatted JSON string."""
    return view.model_dump_json(indent=2)


def render_rewrite_jsonl(view: SuggestView) -> str:
    """Serialize a SuggestView to JSONL lines for each skill."""
    return "".join(skill.model_dump_json() + "\n" for skill in view.skills)


def render_rewrite_csv(view: SuggestView) -> str:
    """Export rewrite suggestions and term breakdowns as formatted CSV."""

    def rows_for(skill: RewriteView) -> list[list[object]]:
        """Construct CSV row tuples for a single RewriteView entry."""
        leading: list[object] = [
            skill.skill,
            skill.verdict.value,
            skill.rival or "",
            len(skill.contenders),
        ]
        rows = [
            [
                *leading,
                "disclaimed" if term.disclaimed else "ceded",
                term.term,
                f"{term.share:.4f}",
            ]
            for term in skill.ceded
        ]
        rows += [[*leading, "unclaimed", term, ""] for term in skill.unclaimed]
        return rows or [[*leading, "", "", ""]]

    return csv_document(
        ["skill", "verdict", "rival", "contenders", "role", "term", "share"],
        (row for skill in view.skills for row in rows_for(skill)),
    )


#: Supported export format handlers for rewrite suggestions.
REWRITE_RENDERERS = {
    "csv": render_rewrite_csv,
    "json": render_rewrite_json,
    "jsonl": render_rewrite_jsonl,
}


def render_rewrite(view: SuggestView, fmt: str) -> str:
    """Render a SuggestView into the requested format (json, csv)."""
    return dispatch_render(REWRITE_RENDERERS, fmt, view)
