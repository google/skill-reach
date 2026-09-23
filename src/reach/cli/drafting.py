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

"""Coordinate catalog selection and synthetic query drafting for evaluations."""

from __future__ import annotations

from importlib import metadata
from typing import TYPE_CHECKING

from reach.catalog import build_catalogs, resident_skills, resolve_catalog
from reach.config import DEFAULT_GEMINI_MODEL, agent_default_model
from reach.difficulty import lexical_ranks
from reach.generate import (
    Citation,
    CitationTrail,
    DraftCheckpoint,
    GeneratedQuery,
    assert_prompts_fit,
    assert_resumable,
    bodies_digest,
    checkpoint_path,
    citations_path,
    generate_query_set,
    read_checkpoint,
    read_citations,
    text_generator,
    write_checkpoint,
    write_citations,
)
from reach.leak import leaks
from reach.queries import Origin, QuerySet, QuerySetProvenance, save_query_set
from reach.views import (
    Console,
    print_draft_preview,
    print_drafted,
    print_generation,
    print_generation_spend,
    print_query_set,
    print_resuming,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from reach.config import RunConfig
    from reach.models import Catalog, Query, Skill
    from reach.runtime import TextGenerator

    from .flags import GenerateFlags

#: Standard provenance notes for generated query sets.
GENERATED_NOTES = (
    "Synthesized by `reach eval` from SKILL.md bodies with skill metadata masked "
    "and citations verified against target bodies. Review before probing: authored "
    "queries override synthetic benchmarks."
)

#: Next-step instructions presented after synthetic drafting completes.
DRAFTED_THEN = (
    "review queries with `reach query {path} -o queries.csv`, then run "
    "`reach eval` to probe catalog routing"
)


def _sole_catalog(settings: RunConfig, skills: Sequence[Skill]) -> Catalog:
    """Resolve target catalog from configuration or verify unique composition."""
    from reach.models import CatalogMode

    scorer = None
    if settings.catalog.mode == CatalogMode.NEIGHBORHOOD and skills:
        from reach.retrieval import build_scorer

        scorer = build_scorer(settings.catalog.scorer, skills, settings)

    catalogs = build_catalogs(
        skills,
        settings.catalog.mode,
        size=settings.catalog.size,
        rivals=settings.catalog.rivals,
        seed=settings.catalog.seed,
        scorer=scorer,
    )

    if settings.study.catalog is not None and settings.study.catalog != "auto":
        return resolve_catalog(catalogs, settings.study.catalog)
    if len(catalogs) != 1:
        msg = (
            f"{settings.catalog.mode} composes {len(catalogs)} catalogs; name "
            "the one to evaluate with --catalog"
        )
        raise ValueError(
            msg,
        )
    return catalogs[0]


def _resolve_draft_targets(
    catalog: Catalog,
    generate: GenerateFlags,
) -> tuple[str, ...]:
    """Validate and return requested target skill names from generation flags."""
    targets = generate.targets or None
    if targets is not None and (stray := sorted(set(targets) - set(catalog.skills))):
        msg = f"--skill names skills that are not in {catalog.id}: {stray}"
        raise ValueError(msg)
    if targets is not None:
        return tuple(targets)
    if catalog.target:
        return (catalog.target,)
    return tuple(catalog.skills)


def _init_or_recover_checkpoint(
    in_progress: Path,
    terms: DraftCheckpoint,
    requested: tuple[str, ...],
    console: Console,
) -> tuple[DraftCheckpoint | None, tuple[str, ...]]:
    """Recover an existing checkpoint or return the requested drafting targets."""
    if not in_progress.exists():
        return None, requested
    recovered = read_checkpoint(in_progress)
    assert_resumable(recovered, terms)
    print_resuming(console, len(recovered.covered), len(recovered.owed), in_progress)
    return recovered, recovered.owed


def _effective_generator_model(settings: RunConfig, generate: GenerateFlags) -> str:
    """Resolve the effective model for query drafting."""
    generator_agent = generate.generator_agent or settings.runtime.agent
    model = generate.generator_model
    if (
        generator_agent
        and model == DEFAULT_GEMINI_MODEL
        and (agent_default := agent_default_model(generator_agent))
    ):
        return agent_default
    return model


def _build_drafter_runtime(
    settings: RunConfig,
    generate: GenerateFlags,
    model: str | None = None,
) -> TextGenerator:
    """Instantiate the drafter TextGenerator for query generation."""
    generator_agent = generate.generator_agent or settings.runtime.agent
    generator_options = (
        settings.runtime.options if generator_agent == settings.runtime.agent else None
    )
    resolved_model = model or _effective_generator_model(settings, generate)
    return text_generator(
        model=resolved_model,
        agent=generator_agent,
        options=generator_options,
    )


def _execute_draft_generation(
    console: Console,
    settings: RunConfig,
    catalog: Catalog,
    skills: Sequence[Skill],
    generate: GenerateFlags,
    drafter: TextGenerator,
    terms: DraftCheckpoint,
    drafting: tuple[str, ...],
    recovered: DraftCheckpoint | None,
    destination: Path,
    in_progress: Path,
    *,
    keep: bool,
    same_invocation_probe: bool,
    then: str,
    review: bool = False,
    existing_query_set: QuerySet | None = None,
) -> int:
    """Execute generation, update checkpoints, persist queries, and print summary."""
    trail: list[Citation] = list(recovered.citations if recovered else ())
    kept = recovered.drafted.queries if recovered is not None else ()
    covered: list[str] = list(recovered.covered if recovered else ())

    def landed(target: str, drafts: tuple[GeneratedQuery, ...]) -> None:
        print_drafted(console, target, len(drafts))
        trail.extend(
            Citation(skill=target, text=d.text, citation=d.citation, reason=d.reason)
            for d in drafts
        )
        covered.append(target)

    def whole(partial: QuerySet) -> QuerySet:
        existing_ids = {q.id for q in kept}
        remapped: list[Query] = []
        for q in partial.queries:
            new_id = q.id
            if new_id in existing_ids:
                prefix, sep, num_str = new_id.rpartition("-")
                if sep and num_str.isdigit():
                    num = int(num_str)
                    base = prefix
                else:
                    num = 0
                    base = new_id
                while new_id in existing_ids:
                    num += 1
                    new_id = f"{base}-{num}"
            existing_ids.add(new_id)
            remapped.append(q if new_id == q.id else q.model_copy(update={"id": new_id}))
        return partial.model_copy(update={"queries": kept + tuple(remapped)})

    def persist(partial: QuerySet) -> None:
        write_checkpoint(
            terms.model_copy(
                update={
                    "covered": tuple(covered),
                    "drafted": whole(partial),
                    "citations": tuple(trail),
                },
            ),
            in_progress,
        )

    drafted = generate_query_set(
        catalog,
        skills,
        count=generate.count,
        runtime=drafter,
        targets=drafting,
        notes=GENERATED_NOTES,
        progress=landed,
        checkpoint=persist,
        arm=generate.generator_arm,
        top_rivals=generate.top_rivals,
        concurrency=generate.draft_concurrency,
        adversarial=generate.adversarial,
        adversarial_count=generate.adversarial_count,
    )
    query_set = whole(drafted)
    if existing_query_set is None:
        query_set = query_set.model_copy(
            update={
                "provenance": _drafted_by(
                    settings,
                    catalog,
                    generate,
                    terms.bodies,
                    generator_model=terms.generator_model,
                    reviewed=False if same_invocation_probe else None,
                ),
            },
        )
    elif existing_query_set.provenance is not None:
        query_set = query_set.model_copy(
            update={
                "provenance": existing_query_set.provenance.model_copy(
                    update={
                        "config_fingerprint": settings.fingerprint,
                        "tool_version": metadata.version("skill-reach"),
                    }
                ),
            },
        )
    if review and query_set.queries:
        from reach.review import launch_query_review

        residents = resident_skills(catalog, skills)
        target_skill = residents[0] if residents else skills[0]
        rivals = [s for s in skills if s.name != target_skill.name]
        reviewed_qs = launch_query_review(query_set, target_skill, rivals)
        if reviewed_qs.provenance is not None:
            query_set = reviewed_qs.model_copy(
                update={"provenance": reviewed_qs.provenance.model_copy(update={"reviewed": True})},
            )
        else:
            query_set = reviewed_qs

    save_query_set(query_set, destination)
    write_citations(CitationTrail(tuple(trail)), citations_path(destination))
    in_progress.unlink(missing_ok=True)
    print_generation_spend(console, drafter.completion_cost_usd, drafter.completions)

    residents = resident_skills(catalog, skills)
    print_query_set(
        console,
        query_set,
        path=destination if keep else None,
        catalog_id=catalog.id,
        residents=len(residents),
        ranks=lexical_ranks(query_set.queries, residents),
        flags=leaks(query_set.queries, residents, background=skills),
        then=then.format(path=destination),
    )
    return 0


def _draft_query_set(
    console: Console,
    settings: RunConfig,
    skills: Sequence[Skill],
    generate: GenerateFlags,
    *,
    dry_run: bool,
    then: str = DRAFTED_THEN,
    keep: bool = True,
    same_invocation_probe: bool = False,
    review: bool = False,
    existing_query_set: QuerySet | None = None,
) -> int:
    """Draft synthetic query sets and persist checkpoint files."""
    catalog = _sole_catalog(settings, skills)
    requested = _resolve_draft_targets(catalog, generate)
    destination = settings.require_queries()
    in_progress = checkpoint_path(destination)
    generator_model = _effective_generator_model(settings, generate)
    drafter = _build_drafter_runtime(settings, generate, generator_model)
    covered_existing: tuple[str, ...] = ()
    existing_citations: tuple[Citation, ...] = ()
    if existing_query_set is not None:
        covered_existing = tuple(sorted(existing_query_set.covered_skills()))
        c_path = citations_path(destination)
        if c_path.exists():
            existing_citations = read_citations(c_path).root

    all_targets = (
        tuple(dict.fromkeys(covered_existing + requested))
        if existing_query_set is not None
        else requested
    )
    terms = DraftCheckpoint(
        fingerprint=settings.fingerprint,
        bodies=bodies_digest(skills),
        catalog_id=catalog.id,
        arm=generate.generator_arm,
        count=generate.count,
        generator_model=generator_model,
        top_rivals=generate.top_rivals,
        targets=all_targets,
        covered=covered_existing,
        drafted=existing_query_set
        if existing_query_set is not None
        else QuerySet(
            catalog_id=catalog.id,
            queries=(),
            provenance=_drafted_by(
                settings,
                catalog,
                generate,
                bodies_digest(skills),
                generator_model=generator_model,
            ),
        ),
        citations=existing_citations,
        adversarial=generate.adversarial,
        adversarial_count=generate.adversarial_count,
    )
    recovered, drafting = _init_or_recover_checkpoint(in_progress, terms, requested, console)
    if recovered is None and existing_query_set is not None:
        recovered = terms

    print_generation(
        console,
        catalog_id=catalog.id,
        residents=len(catalog.skills),
        targets=len(drafting),
        count=generate.count,
        agent=drafter.name,
        model=generator_model,
        rivals=_rivals_in_view(catalog, generate),
        adversarial=generate.adversarial,
        adversarial_count=generate.adversarial_count,
    )
    if dry_run:
        print_draft_preview(
            console,
            destination=destination if keep else None,
            longest_prompt=assert_prompts_fit(
                drafter,
                catalog,
                skills,
                drafting,
                count=generate.count,
                arm=generate.generator_arm,
                top_rivals=generate.top_rivals,
                adversarial=generate.adversarial,
            ),
            budget=drafter.prompt_budget_chars(),
        )
        return 0

    return _execute_draft_generation(
        console,
        settings,
        catalog,
        skills,
        generate,
        drafter,
        terms,
        drafting,
        recovered,
        destination,
        in_progress,
        keep=keep,
        same_invocation_probe=same_invocation_probe,
        then=then,
        review=review,
        existing_query_set=existing_query_set,
    )


def _rivals_in_view(catalog: Catalog, generate: GenerateFlags) -> int:
    """Calculate the number of rival skills included in generation prompts."""
    field = max(len(catalog.skills) - 1, 0)
    if generate.top_rivals is None:
        return field
    return min(field, generate.top_rivals)


def _drafted_by(
    settings: RunConfig,
    catalog: Catalog,
    generate: GenerateFlags,
    bodies: str,
    *,
    generator_model: str | None = None,
    reviewed: bool | None = None,
) -> QuerySetProvenance:
    """Construct QuerySetProvenance detailing synthetic generation parameters."""
    return QuerySetProvenance(
        origin=Origin.GENERATED,
        tool_version=metadata.version("skill-reach"),
        generator_model=generator_model or generate.generator_model,
        generator_arm=generate.generator_arm.value,
        queries_per_target=generate.count,
        rivals_in_view=_rivals_in_view(catalog, generate),
        bodies_digest=bodies,
        config_fingerprint=settings.fingerprint,
        reviewed=reviewed,
        adversarial=generate.adversarial,
        adversarial_per_target=generate.adversarial_count if generate.adversarial else None,
    )
