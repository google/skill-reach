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

"""Compute and report lexical overlap between skill descriptions in a corpus."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Final

from cyclopts import App, Parameter
from pydantic import ValidationError

from reach.attribution import attribute_query
from reach.config import (
    OverlapSettings,
    RegistrySettings,
    RunConfig,
    RuntimeSettings,
    _discover_config_path,
    resolve_sub_settings,
)
from reach.discovery import resolve_corpus
from reach.overlap import CorpusOverlap, did_you_mean_hint, rank_corpus
from reach.retrieval import Bm25Scorer
from reach.rewrite import suggest_all
from reach.runtime import build_runtime
from reach.views import (
    DEFAULT_OVERLAP_TOP,
    Console,
    OverlapFilter,
    build_console,
    filter_competitions,
    filter_rewrites,
    help_formatter,
    overlap_view,
    print_attribution,
    print_discovery,
    print_no_actionable_rewrites,
    print_omitted_footer,
    print_overlap,
    print_overlap_caveat,
    print_rewrite,
    print_skill_overlap,
    render_overlap,
    render_rewrite,
    suggest_view,
)

from .app import LOOP, OPTIONS, app
from .discovery import _corpus, _no_skills
from .flags import (
    LIST,
    REGISTRY_GROUP,
    SWITCH,
    AgentName,
    ConfigFlag,
    Format,
    Global,
    RegistryFlags,
    agent_help_text,
    complete_skill_names,
)

#: Minimum number of skills required to compute pairwise similarity.
MIN_PAIRWISE_SKILLS: Final = 2

#: Default non-TTY console width so long skill families do not collide when piped.
NON_TTY_OVERLAP_WIDTH: Final = 140

if TYPE_CHECKING:
    from collections.abc import Sequence

    from reach.models import Skill


def _compute_semantic_sims(
    found: Sequence[Skill],
    enabled: bool,
) -> dict[tuple[str, str], float] | None:
    """Compute pairwise semantic similarity dictionary if semantic flag enabled."""
    if not enabled or len(found) < MIN_PAIRWISE_SKILLS:
        return None
    try:
        from reach.retrieval import DenseScorer

        scorer = DenseScorer.from_skills(found)
        pairs = scorer.pairwise_similarity(found)
        return {(s1, s2): sim for s1, s2, sim in pairs}
    except (RuntimeError, ValueError, OSError):
        return None


def _render_text_overlap(
    console: Console,
    overlap: CorpusOverlap,
    found: Sequence[Skill],
    skill: Sequence[str],
    *,
    suggest: bool,
    semantic_sims: dict[tuple[str, str], float] | None,
    overlap_filter: OverlapFilter | None = None,
    overlap_settings: OverlapSettings | None = None,
    truncate: bool = True,
) -> None:
    """Render human-readable overlap tables, rewrites, or competitor scorecards."""
    filt = overlap_filter or OverlapFilter()
    cfg = overlap_settings or OverlapSettings()
    all_names = tuple(c.skill for c in overlap.competitions)

    if suggest:
        if skill:
            for rewrite in suggest_all(
                overlap,
                found,
                skill,
                settings=cfg,
            ):
                print_rewrite(console, rewrite, caveat=False)
            print_overlap_caveat(console)
            return

        shown_rewrites, total_matching = filter_rewrites(
            overlap,
            found,
            skill,
            overlap_filter=filt,
            semantic_similarities=semantic_sims,
            default_top=DEFAULT_OVERLAP_TOP,
            overlap_settings=cfg,
        )
        if not shown_rewrites:
            print_no_actionable_rewrites(console, len(found))
            print_overlap_caveat(console)
            return

        for rewrite in shown_rewrites:
            print_rewrite(console, rewrite, caveat=False)
        if len(shown_rewrites) < total_matching:
            print_omitted_footer(
                console,
                total_matching - len(shown_rewrites),
                total_matching,
            )
        print_overlap_caveat(console)
        return

    if not skill:
        effective_top = (
            filt.top if filt.top is not None else (None if filt.all_skills else DEFAULT_OVERLAP_TOP)
        )
        shown_comps, total_matching = filter_competitions(
            overlap.competitions,
            semantic_similarities=semantic_sims,
            quadrants=filt.quadrants,
            top=effective_top,
        )
        print_overlap(
            console,
            overlap,
            semantic_similarities=semantic_sims,
            competitions=shown_comps,
            omitted_count=max(0, total_matching - len(shown_comps)),
            truncate=truncate,
        )
        return

    for name in skill:
        print_skill_overlap(
            console,
            overlap.find(name),
            semantic_similarities=semantic_sims,
            caveat=False,
            truncate=truncate,
            peers=all_names,
        )
    print_overlap_caveat(console)


def _render_overlap_output(
    console: Console,
    overlap: CorpusOverlap,
    found: Sequence[Skill],
    skill: Sequence[str],
    *,
    suggest: bool,
    semantic: bool = False,
    format: Format,
    overlap_filter: OverlapFilter | None = None,
    overlap_settings: OverlapSettings | None = None,
    truncate: bool = True,
) -> None:
    effective_semantic = semantic or bool(overlap_filter and overlap_filter.quadrants)
    semantic_sims = _compute_semantic_sims(found, effective_semantic)

    if format != "text":
        rendered = (
            render_rewrite(
                suggest_view(
                    overlap,
                    found,
                    skill,
                    overlap_filter=overlap_filter,
                    semantic_similarities=semantic_sims,
                    overlap_settings=overlap_settings,
                ),
                format,
            )
            if suggest
            else render_overlap(
                overlap_view(
                    overlap,
                    skill,
                    semantic_similarities=semantic_sims,
                    overlap_filter=overlap_filter,
                ),
                format,
            )
        )
        print(rendered)
        return

    _render_text_overlap(
        console,
        overlap,
        found,
        skill,
        suggest=suggest,
        semantic_sims=semantic_sims,
        overlap_filter=overlap_filter,
        overlap_settings=overlap_settings,
        truncate=truncate,
    )


def _handle_explain(
    console: Console,
    query_text: str,
    target_skill: str,
    rival_skill: str | None,
    found: Sequence[Skill],
    format: Format,
    *,
    semantic: bool = False,
) -> int:
    """Diagnose token-level BM25 contributions driving a query toward a rival skill."""
    scorer = Bm25Scorer.from_skills(found)
    available_names = [s.name for s in found]
    target_obj = next((s for s in found if s.name == target_skill), None)
    if target_obj is None:
        hint = did_you_mean_hint(target_skill, available_names)
        msg = f"target skill {target_skill!r} not found in resident skills{hint}"
        raise ValueError(msg)

    is_abstention = rival_skill is not None and rival_skill.lower() in (
        "none",
        "(no selection)",
        "(no skill)",
        "",
    )
    rival_obj: Skill | None = None

    if is_abstention:
        rival_skill = "(no selection)"
    elif rival_skill is None:
        overlap = rank_corpus(found)
        competition = overlap.find(target_skill)
        scoring = competition.scoring_rival
        if scoring is None:
            msg = f"no competing rival found for skill {target_skill!r}"
            raise ValueError(msg)
        rival_skill = scoring.name
        rival_obj = next((s for s in found if s.name == rival_skill), None)
    else:
        rival_obj = next((s for s in found if s.name == rival_skill), None)
        if rival_obj is None:
            hint = did_you_mean_hint(rival_skill, available_names)
            msg = f"rival skill {rival_skill!r} not found in resident skills{hint}"
            raise ValueError(msg)

    if not is_abstention and target_skill == rival_skill:
        msg = f"--skill and --rival cannot be the same skill ({target_skill!r})"
        raise ValueError(msg)

    target_semantic: float | None = None
    rival_semantic: float | None = None
    if semantic:
        try:
            from reach.retrieval import DenseScorer

            dense_scorer = DenseScorer.from_skills(found)
            target_semantic = dense_scorer.score_query(query_text, target_obj)
            if rival_obj is not None:
                rival_semantic = dense_scorer.score_query(query_text, rival_obj)
        except (RuntimeError, ValueError, OSError):
            pass

    attribution = attribute_query(
        query_text,
        target_skill,
        rival_skill,
        scorer,
        target_semantic=target_semantic,
        rival_semantic=rival_semantic,
    )
    if format == "json":
        print(attribution.model_dump_json(indent=2))
        return 0
    if format == "jsonl":
        print(attribution.model_dump_json())
        return 0

    print_attribution(console, attribution)
    return 0


overlap_app = App(
    name="overlap",
    help="Find which of your installed skills compete to answer the same requests.",
    help_formatter=help_formatter(),
    group_parameters=OPTIONS,
    result_action="return_value",
    group=LOOP,
    default_parameter=Parameter(allow_repeating=True),
)
app.command(overlap_app)


@overlap_app.default
def _overlap(
    skills: Annotated[
        Path | None,
        Parameter(
            help=(
                "Path to the skill directory or corpus to analyze (discovered "
                "from precedence if omitted)"
            ),
        ),
    ] = None,
    *,
    skill: Annotated[
        tuple[str, ...],
        LIST,
        Parameter(
            completer=complete_skill_names,
            help="Analyze overlap specifically for this skill against all "
            "competitors in the corpus (repeatable)",
        ),
    ] = (),
    suggest: Annotated[
        bool,
        SWITCH,
        Parameter(
            help="Generate suggested description rewrites to reduce lexical overlap",
        ),
    ] = False,
    semantic: Annotated[
        bool,
        SWITCH,
        Parameter(
            help="Include dense semantic similarity and dual-axis diagnostic quadrant matrix",
        ),
    ] = False,
    top: Annotated[
        int | None,
        Parameter(
            help="Show only the top N ranked skills",
        ),
    ] = None,
    all_: Annotated[
        bool,
        SWITCH,
        Parameter(
            name="--all",
            help="Show all skills without the default 30-row cap or actionable-only filter",
        ),
    ] = False,
    quadrant: Annotated[
        tuple[str, ...],
        LIST,
        Parameter(
            help=(
                "Filter by diagnostic quadrant: near-duplicate, latent-collision, "
                "boilerplate, distinct (implies --semantic)"
            ),
        ),
    ] = (),
    no_truncate: Annotated[
        bool,
        SWITCH,
        Parameter(
            help="Render full skill names without middle truncation",
        ),
    ] = False,
    agent: Annotated[
        AgentName | None,
        Parameter(
            help=agent_help_text("Agent runtime to query for installed skill locations"),
        ),
    ] = None,
    global_: Global = False,
    registry: Annotated[RegistryFlags | None, Parameter(group=REGISTRY_GROUP)] = None,
    format: Annotated[Format, Parameter(help="How to render the ranking")] = "text",
    config: ConfigFlag = None,
) -> int:
    """Find which of your installed skills compete to answer the same requests."""
    use_wide_console = not sys.stderr.isatty() and "COLUMNS" not in os.environ
    console = build_console(width=NON_TTY_OVERLAP_WIDTH if use_wide_console else None)
    try:
        overlap_filter = OverlapFilter(
            top=top,
            all_skills=all_,
            quadrants=quadrant,
        )
    except ValidationError as exc:
        first_msg = exc.errors()[0].get("msg", str(exc)) if exc.errors() else str(exc)
        raise ValueError(first_msg) from exc

    if skill:
        from reach.catalog import resolve_skill_target

        base_catalog = skills
        resolved_skills_list: list[str] = []
        for s in skill:
            try:
                res = resolve_skill_target(s, explicit_catalog=base_catalog, command_name="overlap")
                resolved_skills_list.append(res.skill_name if res else s)
                if skills is None and res and res.catalog_path:
                    skills = res.catalog_path
            except (FileNotFoundError, ValueError) as err:
                console.print(f"[red]Error:[/] {err}")
                return 2
        skill = tuple(resolved_skills_list)

    driver = build_runtime(RuntimeSettings(agent=agent) if agent is not None else RuntimeSettings())

    resolved_config, _ = _discover_config_path(config)
    run_config: RunConfig | None = None
    if resolved_config is not None:
        run_config = RunConfig.from_toml(resolved_config)

    effective_config = run_config or RunConfig()
    effective_registry = resolve_sub_settings(
        RegistrySettings,
        effective_config.registry,
        **(registry.overrides() if registry is not None else {}),
    )
    has_cli_registry = registry is not None and (registry.registry or registry.project is not None)
    if skills is not None and not has_cli_registry:
        effective_registry = effective_registry.model_copy(
            update={"registry": False, "project": None}
        )
    eff_skills = (
        skills
        if skills is not None
        else (None if has_cli_registry else effective_config.study.skills)
    )
    run_config = effective_config.model_copy(
        update={
            "study": effective_config.study.model_copy(update={"skills": eff_skills}),
            "registry": effective_registry,
        }
    )
    found, _roots, _discovered = _corpus(
        console,
        driver,
        settings=run_config,
        global_scope=global_,
        agent=agent,
    )

    overlap = rank_corpus(found)
    _render_overlap_output(
        console,
        overlap,
        found,
        skill,
        suggest=suggest,
        semantic=semantic,
        format=format,
        overlap_filter=overlap_filter,
        overlap_settings=run_config.overlap,
        truncate=not no_truncate,
    )
    return 0


@overlap_app.command(name="explain")
def _explain_cmd(
    query: Annotated[str, Parameter(help="The prompt or query text to diagnose")],
    *,
    skill: Annotated[
        str,
        Parameter(
            completer=complete_skill_names,
            help="The expected ground-truth skill name",
        ),
    ],
    rival: Annotated[
        str | None,
        Parameter(
            alias="-r",
            completer=complete_skill_names,
            help=(
                "The rival or misrouted skill name (auto-selects nearest rival if "
                "omitted, or 'none' for abstention)"
            ),
        ),
    ] = None,
    semantic: Annotated[
        bool,
        Parameter(
            help="Include dense semantic similarity alongside lexical BM25 scores",
        ),
    ] = False,
    skills: Annotated[
        Path | None,
        Parameter(
            help="Path to the skill directory or corpus to analyze",
        ),
    ] = None,
    agent: Annotated[
        AgentName | None,
        Parameter(
            help=agent_help_text("Agent runtime to query for installed skill locations"),
        ),
    ] = None,
    global_: Global = False,
    format: Annotated[Format, Parameter(help="How to render the diagnosis")] = "text",
) -> int:
    """Explain token-level BM25 contributions driving a query toward a rival skill."""
    console = build_console()
    from reach.catalog import resolve_skill_target

    try:
        resolved_skill = resolve_skill_target(
            skill, explicit_catalog=skills, command_name="overlap explain"
        )
        if resolved_skill is not None:
            skill = resolved_skill.skill_name
            if skills is None and resolved_skill.catalog_path:
                skills = resolved_skill.catalog_path
        if rival is not None and rival.lower() not in (
            "none",
            "(no selection)",
            "(no skill)",
            "",
        ):
            resolved_rival = resolve_skill_target(
                rival, explicit_catalog=skills, command_name="overlap explain"
            )
            if resolved_rival is not None:
                rival = resolved_rival.skill_name
    except (FileNotFoundError, ValueError) as err:
        console.print(f"[red]Error:[/] {err}")
        return 2

    driver = build_runtime(RuntimeSettings(agent=agent) if agent is not None else RuntimeSettings())

    found, _roots, discovered = resolve_corpus(
        driver,
        Path.cwd(),
        skills,
        agent=agent,
        global_scope=global_,
    )
    if discovered is not None:
        print_discovery(console, discovered)
    if not found:
        raise _no_skills(skills, global_scope=global_)
    return _handle_explain(console, query, skill, rival, found, format, semantic=semantic)
