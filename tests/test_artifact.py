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

"""Verify Artifact structure, schema validation, metrics aggregation, and serialization."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import pytest
from pydantic import ValidationError as PydanticValidationError

from reach.artifact import (
    SCHEMA_VERSION,
    Artifact,
    ConfusionPair,
    SampleQuery,
    SkillScore,
    Spread,
    artifact_path,
    read_artifact,
    write_artifact,
)
from reach.catalog import corpus_digest
from reach.leak import Leak
from reach.metrics import classification_report, consistency
from reach.models import (
    NO_SKILL,
    Catalog,
    CatalogMode,
    ProbeResult,
    Query,
    QueryKind,
    Skill,
)
from reach.queries import Origin, QuerySet, QuerySetProvenance, query_set_digest
from reach.run import Composition
from reach.runtime import CatalogFit

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from reach.config import RunConfig

LIFECYCLE = "gcs-lifecycle-rules"
RETENTION = "gcs-retention-policy"
BASICS = "gke-basics"


def assemble(
    results: Sequence[ProbeResult],
    query_set: QuerySet,
    catalog: Catalog,
    skills: Sequence[Skill],
    config: RunConfig,
    **kwargs: Any,
) -> Artifact:
    """Assemble an Artifact from components wrapped in a Composition."""
    return Artifact.assemble(
        Composition(
            config=config,
            query_set=query_set,
            catalog=catalog,
            skills=tuple(skills),
        ),
        results,
        **kwargs,
    )


@pytest.fixture
def config(make_config: Callable[..., RunConfig]) -> RunConfig:
    """Return RunConfig configured for a 3-attempt evaluation over the whole synthetic catalog."""
    return make_config(catalog={"mode": CatalogMode.ALL}, plan={"attempts": 3})


def test_artifact_schema_digests_provenance_and_roots(
    artifact: Artifact,
    corpus: list[Skill],
    whole_catalog_queries: QuerySet,
    config: RunConfig,
    skill_repo: Path,
) -> None:
    """Verify artifact schema version, digests, provenance, catalog fields, and resolved roots."""
    assert artifact.schema_version == SCHEMA_VERSION
    assert artifact.digests.config_fingerprint == config.fingerprint
    assert artifact.digests.corpus_digest == corpus_digest(corpus)
    assert artifact.digests.queries_digest == query_set_digest(whole_catalog_queries)

    assert artifact.provenance.runtime == "fake"
    assert artifact.provenance.model == "sonnet"
    assert artifact.provenance.attempts == 3
    assert artifact.provenance.arm == config.arm
    assert artifact.provenance.catalog_fit is None

    assert artifact.catalog_id == "all"
    assert artifact.catalog_mode is CatalogMode.ALL
    assert artifact.catalog_size == 3
    assert artifact.catalog_target is None

    root = skill_repo.resolve()
    assert [(r.path, r.skills) for r in artifact.resolved_roots] == [(root, 3)]
    assert {s.skill: s.root for s in artifact.skills} == dict.fromkeys(
        (LIFECYCLE, RETENTION, BASICS),
        root,
    )


def test_the_artifact_records_how_much_of_the_catalog_was_shown(
    whole_catalog_results: list[ProbeResult],
    whole_catalog_queries: QuerySet,
    whole_catalog: Catalog,
    corpus: list[Skill],
    config: RunConfig,
) -> None:
    """Verify catalog fit metrics are preserved across serialization boundaries."""
    fit = CatalogFit(
        allowed=30,
        asked=91,
        unit="display columns",
        truncated=2,
        remedy="widen it",
    )
    built = assemble(
        whole_catalog_results,
        whole_catalog_queries,
        whole_catalog,
        corpus,
        config,
        fit=fit,
    )
    assert built.catalog_size == 3
    assert built.provenance.catalog_fit == fit
    reloaded = Artifact.model_validate_json(built.model_dump_json())
    assert reloaded.provenance.catalog_fit == fit


def test_roots_attribute_each_skill_to_the_most_specific_tree(
    whole_catalog_results: list[ProbeResult],
    whole_catalog_queries: QuerySet,
    whole_catalog: Catalog,
    corpus: list[Skill],
    config: RunConfig,
    skill_repo: Path,
    tmp_path: Path,
) -> None:
    """Verify skills are attributed to the most specific nested root directory."""
    elsewhere = tmp_path / "elsewhere"
    artifact = assemble(
        whole_catalog_results,
        whole_catalog_queries,
        whole_catalog,
        corpus,
        config,
        roots=[skill_repo, skill_repo / "storage", elsewhere],
    )
    assert {s.skill: s.root for s in artifact.skills} == {
        LIFECYCLE: (skill_repo / "storage").resolve(),
        RETENTION: (skill_repo / "storage").resolve(),
        BASICS: skill_repo.resolve(),
    }
    assert [(r.path, r.skills) for r in artifact.resolved_roots] == [
        (elsewhere.resolve(), 0),
        (skill_repo.resolve(), 1),
        ((skill_repo / "storage").resolve(), 2),
    ]


@pytest.mark.parametrize(
    ("skill", "probes", "reached", "recall", "absorbed", "precision"),
    [
        pytest.param(LIFECYCLE, 3, 3, 1.0, 0, 1.0, id="reached-every-time"),
        pytest.param(RETENTION, 3, 1, 1 / 3, 0, 1.0, id="reached-one-time-in-three"),
        pytest.param(BASICS, 0, 0, None, 1, 0.0, id="asked-for-by-nobody"),
    ],
)
def test_per_skill_recall_is_per_class_recall(
    artifact: Artifact,
    skill: str,
    probes: int,
    reached: int,
    recall: float | None,
    absorbed: int,
    precision: float | None,
) -> None:
    """Verify probe counts, reached counts, precision, and recall per skill."""
    entry = next(s for s in artifact.skills if s.skill == skill)
    assert (entry.probes, entry.reached, entry.recall) == (probes, reached, recall)
    assert (entry.absorbed, entry.precision) == (absorbed, precision)


def test_the_metric_is_not_named_after_the_instrument(artifact: Artifact) -> None:
    """Verify skill metric fields use standard recall terminology."""
    entry = next(s for s in artifact.skills if s.skill == LIFECYCLE)
    payload = entry.model_dump()

    assert "recall" in payload
    assert "recall_interval" in payload
    assert "reach" not in payload
    assert "reach_interval" not in payload


@pytest.mark.parametrize(
    ("skill", "f1"),
    [
        pytest.param(LIFECYCLE, 1.0, id="found-every-time-and-took-nothing-else"),
        pytest.param(RETENTION, 0.5, id="half-of-a-perfect-precision"),
        pytest.param(BASICS, None, id="no-query-named-it-so-there-is-no-f1"),
    ],
)
def test_f1_combines_the_two_halves_of_a_skills_score(
    artifact: Artifact,
    skill: str,
    f1: float | None,
) -> None:
    """Verify F1 score computation across per-skill precision and recall figures."""
    entry = next(s for s in artifact.skills if s.skill == skill)
    if f1 is None:
        assert entry.f1 is None
    else:
        assert entry.f1 == pytest.approx(f1)


def test_f1_is_zero_rather_than_absent_for_a_skill_that_never_fired() -> None:
    """Verify F1 is 0.0 when precision is undefined due to zero invocations."""
    entry = SkillScore(skill=LIFECYCLE, probes=5, reached=0, recall=0.0)

    assert entry.precision is None
    assert entry.f1 == 0.0


def test_f1_is_offered_without_an_interval(artifact: Artifact) -> None:
    """Verify F1 does not define an interval field."""
    payload = next(s for s in artifact.skills if s.skill == LIFECYCLE).model_dump()

    assert payload["f1"] is not None
    assert "f1_interval" not in payload


def test_headline_and_secondary_figures_are_kept_apart(
    artifact: Artifact,
    whole_catalog_results: list[ProbeResult],
    whole_catalog_queries: QuerySet,
) -> None:
    """Verify separation of headline accuracy/consistency from secondary statistical metrics."""
    queries = whole_catalog_queries.queries
    assert artifact.scores.consistency == 0.5
    assert artifact.scores.consistency == consistency(whole_catalog_results, queries)
    assert artifact.scores.top1_accuracy == pytest.approx(4 / 6)
    assert artifact.scores.not_headline.macro_f1 == pytest.approx(0.75)


def test_the_standard_metrics_are_not_recomputed_here(
    artifact: Artifact,
    whole_catalog_results: list[ProbeResult],
    whole_catalog_queries: QuerySet,
) -> None:
    """Verify artifact metrics match calculation from reach.metrics."""
    report = classification_report(whole_catalog_results, whole_catalog_queries.queries)
    assert artifact.scores.top1_accuracy == report.top1_accuracy
    assert artifact.scores.not_headline.macro_f1 == report.macro_f1


def test_confusion_pairs_record_which_skill_went_where(artifact: Artifact) -> None:
    """Verify confusion pairs record expected vs invoked skills and collision counts."""
    assert [(p.expected, p.invoked, p.probes, p.collisions) for p in artifact.confusion] == [
        (LIFECYCLE, LIFECYCLE, 3, 0),
        (RETENTION, NO_SKILL, 1, 0),
        (RETENTION, RETENTION, 1, 0),
        (RETENTION, BASICS, 1, 1),
    ]


def test_sample_query_probe_text_is_embedded_rather_than_referenced(
    artifact: Artifact,
    queries: list[Query],
) -> None:
    """Verify query text is embedded in confusion sample queries."""
    texts = {q.id: q.text for q in queries}
    misroute = next(p for p in artifact.confusion if p.collisions)
    assert misroute.queries == (
        SampleQuery(query_id="q-retention", text=texts["q-retention"], probes=1),
    )
    assert all(e.text for pair in artifact.confusion for e in pair.queries)


@pytest.mark.parametrize("limit", [1, 2, 5])
def test_sample_queries_are_bounded_and_ordered_by_weight(
    whole_catalog: Catalog,
    corpus: list[Skill],
    config: RunConfig,
    make_result,
    limit: int,
) -> None:
    """Verify sample query lists are sorted by descending probe volume and bounded by limit."""
    queries = tuple(
        Query(
            id=f"q{i}",
            text=f"Query number {i}.",
            kind=QueryKind.IMPLICIT,
            expected_skill=RETENTION,
        )
        for i in range(3)
    )
    query_set = QuerySet(
        catalog_id="all",
        queries=queries,
        provenance=QuerySetProvenance(origin=Origin.AUTHORED),
    )
    results = [
        make_result(q.id, BASICS, attempt=attempt)
        for i, q in enumerate(queries, start=1)
        for attempt in range(1, i + 1)
    ]
    artifact = assemble(
        results,
        query_set,
        whole_catalog,
        corpus,
        config,
        sample_queries=limit,
    )
    pair = next(p for p in artifact.confusion if p.invoked == BASICS)
    assert [(e.query_id, e.probes) for e in pair.queries] == [
        ("q2", 3),
        ("q1", 2),
        ("q0", 1),
    ][:limit]


def test_a_negative_sample_queries_limit_is_refused(
    whole_catalog_results: list[ProbeResult],
    whole_catalog_queries: QuerySet,
    whole_catalog: Catalog,
    corpus: list[Skill],
    config: RunConfig,
) -> None:
    """Verify negative sample queries count limits raise ValueError."""
    with pytest.raises(ValueError, match="sample queries"):
        assemble(
            whole_catalog_results,
            whole_catalog_queries,
            whole_catalog,
            corpus,
            config,
            sample_queries=-1,
        )


def test_every_query_is_recorded_with_its_text_verbatim(
    artifact: Artifact,
    queries: list[Query],
) -> None:
    """Verify all queries are embedded with verbatim text in the artifact."""
    assert [(r.query_id, r.text) for r in artifact.queries] == [
        (q.id, q.text) for q in sorted(queries, key=lambda q: q.id)
    ]


def test_per_query_outcomes_summarize_the_attempts(artifact: Artifact) -> None:
    """Verify per-query attempt summaries, hits, and selection sequences."""
    by_id = {r.query_id: r for r in artifact.queries}
    assert (by_id["q-lifecycle"].hits, by_id["q-lifecycle"].probes) == (3, 3)
    assert by_id["q-lifecycle"].selections == (LIFECYCLE,)
    assert (by_id["q-retention"].hits, by_id["q-retention"].probes) == (1, 3)
    assert by_id["q-retention"].selections == (NO_SKILL, RETENTION, BASICS)


def test_every_query_carries_its_difficulty_rank(artifact: Artifact) -> None:
    """Verify artifact records BM25 difficulty rank for each query."""
    assert {r.query_id: r.difficulty_rank for r in artifact.queries} == {
        "q-lifecycle": 3,
        "q-retention": 3,
    }


def test_a_query_written_in_the_target_vocabulary_ranks_first(
    whole_catalog: Catalog,
    corpus: list[Skill],
    config: RunConfig,
    make_result,
) -> None:
    """Verify difficulty rank is 1 when query vocabulary matches target skill description."""
    query = Query(
        id="q-easy",
        text="Configures object lifecycle rules.",
        kind=QueryKind.IMPLICIT,
        expected_skill=LIFECYCLE,
    )
    artifact = assemble(
        [make_result("q-easy", LIFECYCLE)],
        QuerySet(
            catalog_id="all",
            queries=(query,),
            provenance=QuerySetProvenance(origin=Origin.AUTHORED),
        ),
        whole_catalog,
        corpus,
        config,
    )
    assert artifact.queries[0].difficulty_rank == 1


def test_a_query_naming_its_own_target_is_flagged(
    whole_catalog: Catalog,
    corpus: list[Skill],
    config: RunConfig,
    make_result,
) -> None:
    """Verify queries containing verbatim target skill names are flagged as leaky."""
    query = Query(
        id="q-leaky",
        text="Use gcs-lifecycle-rules to tier my objects.",
        kind=QueryKind.IMPLICIT,
        expected_skill=LIFECYCLE,
    )
    artifact = assemble(
        [make_result("q-leaky", LIFECYCLE)],
        QuerySet(
            catalog_id="all",
            queries=(query,),
            provenance=QuerySetProvenance(origin=Origin.AUTHORED),
        ),
        whole_catalog,
        corpus,
        config,
    )
    assert artifact.queries[0].leak == Leak(
        names_target=True,
        distinctive_tokens=("lifecycle", "rules"),
    )


def test_a_clean_query_is_not_flagged(artifact: Artifact) -> None:
    """Verify clean queries have unflagged Leak objects."""
    assert [r.leak for r in artifact.queries] == [Leak(), Leak()]


def test_an_out_of_scope_query_has_no_leak_to_check(
    whole_catalog: Catalog,
    corpus: list[Skill],
    config: RunConfig,
    make_result,
) -> None:
    """Verify out-of-scope queries carry None for leak and difficulty rank."""
    query = Query(
        id="q-oos",
        text="What is the capital of France?",
        kind=QueryKind.OUT_OF_SCOPE,
    )
    artifact = assemble(
        [make_result("q-oos", None)],
        QuerySet(
            catalog_id="all",
            queries=(query,),
            provenance=QuerySetProvenance(origin=Origin.AUTHORED),
        ),
        whole_catalog,
        corpus,
        config,
    )
    assert artifact.queries[0].leak is None
    assert artifact.queries[0].difficulty_rank is None


def test_spread_gives_compare_a_standard_error_on_top_one(
    artifact: Artifact,
) -> None:
    """Verify spread calculation produces expected pooled standard error."""
    assert artifact.spread.repeated_queries == 2
    assert artifact.spread.standard_error == pytest.approx((1.03125**0.5) / 6)
    assert artifact.scores.top1_accuracy == pytest.approx(4 / 6)


def test_the_attempt_slices_are_reported_but_are_not_the_floor(
    artifact: Artifact,
) -> None:
    """Verify spread captures per-attempt accuracies and mean accuracy."""
    assert artifact.spread.replicates == 3
    assert artifact.spread.top1_by_attempt == (1.0, 0.5, 0.5)
    assert artifact.spread.mean == pytest.approx(2 / 3)


def test_the_standard_error_shrinks_as_attempts_are_added(
    whole_catalog_queries: QuerySet,
    whole_catalog: Catalog,
    corpus: list[Skill],
    make_config,
    make_result,
) -> None:
    """Verify standard error decreases monotonically as attempt count increases."""

    def measure(attempts: int) -> Spread:
        results = [
            make_result("q-lifecycle", LIFECYCLE, attempt=i) for i in range(1, attempts + 1)
        ] + [
            make_result("q-retention", RETENTION if i % 2 else BASICS, attempt=i)
            for i in range(1, attempts + 1)
        ]
        return assemble(
            results,
            whole_catalog_queries,
            whole_catalog,
            corpus,
            make_config(catalog={"mode": CatalogMode.ALL}, plan={"attempts": attempts}),
        ).spread

    spreads = [measure(attempts) for attempts in (2, 4, 8, 16)]
    tightening = [s.standard_error for s in spreads if s.standard_error is not None]
    assert len(set(tightening)) == len(spreads)
    assert tightening == sorted(tightening, reverse=True)
    assert tightening[-1] < tightening[0] / 3
    assert all(max(s.top1_by_attempt) - min(s.top1_by_attempt) == 0.5 for s in spreads)


def test_the_standard_error_ignores_which_attempt_a_miss_landed_on(
    whole_catalog_queries: QuerySet,
    whole_catalog: Catalog,
    corpus: list[Skill],
    config: RunConfig,
    make_result,
) -> None:
    """Verify standard error calculation is permutation invariant to attempt ordering."""

    def floor(stagger: bool) -> Spread:
        return assemble(
            [make_result("q-lifecycle", LIFECYCLE if i == 1 else BASICS, attempt=i) for i in (1, 2)]
            + [
                make_result(
                    "q-retention",
                    RETENTION if i == (2 if stagger else 1) else BASICS,
                    attempt=i,
                )
                for i in (1, 2)
            ],
            whole_catalog_queries,
            whole_catalog,
            corpus,
            config,
        ).spread

    lockstep, staggered = floor(stagger=False), floor(stagger=True)
    assert lockstep.top1_by_attempt == (1.0, 0.0)
    assert staggered.top1_by_attempt == (0.5, 0.5)
    assert lockstep.standard_error == pytest.approx(0.25)
    assert staggered.standard_error == lockstep.standard_error


def test_a_query_asked_once_is_uncertain_rather_than_certain(
    whole_catalog_queries: QuerySet,
    whole_catalog: Catalog,
    corpus: list[Skill],
    make_config,
    make_result,
) -> None:
    """Verify single-attempt eval applies Laplace smoothing rather than claiming zero error."""
    artifact = assemble(
        [make_result(q.id, q.expected_skill) for q in whole_catalog_queries.queries],
        whole_catalog_queries,
        whole_catalog,
        corpus,
        make_config(catalog={"mode": CatalogMode.ALL}),
    )
    assert artifact.spread.replicates == 1
    assert artifact.spread.repeated_queries == 0
    assert artifact.scores.top1_accuracy == 1.0
    assert artifact.spread.standard_error == pytest.approx((0.375**0.5) / 2)


def test_a_query_that_never_hit_still_carries_uncertainty(
    whole_catalog_queries: QuerySet,
    whole_catalog: Catalog,
    corpus: list[Skill],
    config: RunConfig,
    make_result,
) -> None:
    """Verify queries with zero hits carry non-zero smoothed standard error."""
    swept = assemble(
        [
            make_result(q.id, q.expected_skill if hit else BASICS, attempt=i)
            for q, hit in zip(whole_catalog_queries.queries, (True, False), strict=True)
            for i in (1, 2, 3)
        ],
        whole_catalog_queries,
        whole_catalog,
        corpus,
        config,
    )
    assert [(q.hits, q.probes) for q in swept.queries] == [(3, 3), (0, 3)]
    assert swept.spread.standard_error == pytest.approx((0.65625**0.5) / 6)


def test_an_errored_attempt_is_not_a_replicate(
    whole_catalog_queries: QuerySet,
    whole_catalog: Catalog,
    corpus: list[Skill],
    config: RunConfig,
    make_result,
) -> None:
    """Verify errored attempts are excluded from replicate counts and top1 accuracies."""
    results = [
        make_result(q.id, q.expected_skill, attempt=1) for q in whole_catalog_queries.queries
    ] + [make_result(q.id, None, attempt=2, error="timeout") for q in whole_catalog_queries.queries]
    artifact = assemble(
        results,
        whole_catalog_queries,
        whole_catalog,
        corpus,
        config,
    )
    assert artifact.probes == 4
    assert artifact.errors == 2
    assert artifact.spread.replicates == 1
    assert artifact.spread.top1_by_attempt == (1.0,)


def test_a_run_with_no_usable_probes_reports_nothing_rather_than_zero(
    whole_catalog_queries: QuerySet,
    whole_catalog: Catalog,
    corpus: list[Skill],
    config: RunConfig,
) -> None:
    """Verify empty run produces artifact with empty metrics and zero probe counts."""
    artifact = assemble([], whole_catalog_queries, whole_catalog, corpus, config)
    assert (artifact.probes, artifact.errors) == (0, 0)
    assert artifact.confusion == ()
    assert artifact.spread == Spread(replicates=0)
    assert all(s.recall is None for s in artifact.skills)
    assert all(r.probes == 0 for r in artifact.queries)


def test_the_artifact_names_which_digests_the_rows_corroborated(
    whole_catalog_queries: QuerySet,
    whole_catalog: Catalog,
    corpus: list[Skill],
    config: RunConfig,
    make_result,
) -> None:
    """Verify artifact records which provenance digests were present and corroborated in results."""
    stamped = assemble(
        [
            make_result(q.id, q.expected_skill, fingerprint=config.fingerprint)
            for q in whole_catalog_queries.queries
        ],
        whole_catalog_queries,
        whole_catalog,
        corpus,
        config,
    )
    assert stamped.verified_digests == ("config_fingerprint",)
    assert stamped.digests.corpus_digest
    assert "corpus_digest" not in stamped.verified_digests


def test_an_artifact_nothing_corroborated_does_not_look_checked(
    whole_catalog_results: list[ProbeResult],
    whole_catalog_queries: QuerySet,
    whole_catalog: Catalog,
    corpus: list[Skill],
    config: RunConfig,
) -> None:
    """Verify verified_digests is empty when rows carry no provenance digest stamps."""
    artifact = assemble(
        whole_catalog_results,
        whole_catalog_queries,
        whole_catalog,
        corpus,
        config,
    )
    assert artifact.verified_digests == ()
    assert all(getattr(artifact.digests, f) for f in ("corpus_digest", "queries_digest"))


def test_a_skill_that_takes_traffic_it_was_never_asked_for_is_an_attractor(
    whole_catalog_queries: QuerySet,
    whole_catalog: Catalog,
    corpus: list[Skill],
    config: RunConfig,
    make_result,
) -> None:
    """Verify unasked skills that absorb invocations are classified as attractors."""
    artifact = assemble(
        [
            make_result(q.id, BASICS, attempt=i)
            for q in whole_catalog_queries.queries
            for i in (1, 2, 3)
        ],
        whole_catalog_queries,
        whole_catalog,
        corpus,
        config,
    )
    basics = next(s for s in artifact.skills if s.skill == BASICS)
    assert (basics.probes, basics.recall, basics.absorbed) == (0, None, 6)
    assert basics not in artifact.unreached
    assert artifact.attractors == (basics,)


def test_the_unclean_queries_are_the_ones_that_did_not_agree_every_time(
    artifact: Artifact,
) -> None:
    """Verify unclean property filters for queries with inconsistent selection across attempts."""
    assert [q.query_id for q in artifact.queries] == ["q-lifecycle", "q-retention"]
    assert [q.query_id for q in artifact.unclean] == ["q-retention"]


def test_a_skill_that_earns_its_selections_is_not_an_attractor(
    artifact: Artifact,
) -> None:
    """Verify skills matching expected queries are not classified as attractors."""
    reached, partial, attractor = (
        next(s for s in artifact.skills if s.skill == name)
        for name in (LIFECYCLE, RETENTION, BASICS)
    )
    assert (reached.probes, reached.recall, reached.absorbed) == (3, 1.0, 0)
    assert (partial.probes, partial.recall, partial.absorbed) == (3, 1 / 3, 0)
    assert (attractor.probes, attractor.recall, attractor.absorbed) == (0, None, 1)
    assert artifact.unreached == ()
    assert artifact.attractors == (attractor,)


def test_declining_is_reported_apart_from_misrouting(
    whole_catalog_queries: QuerySet,
    whole_catalog: Catalog,
    corpus: list[Skill],
    config: RunConfig,
    make_result,
) -> None:
    """Verify abstentions are tracked in abstention metrics rather than misroutes."""
    artifact = assemble(
        [make_result(q.id, None) for q in whole_catalog_queries.queries],
        whole_catalog_queries,
        whole_catalog,
        corpus,
        config,
    )
    assert artifact.scores.abstention.rate == 1.0
    assert artifact.scores.abstention.false_rate == 1.0
    assert artifact.scores.abstention.out_of_scope_detection is None
    assert artifact.scores.top1_accuracy == 0.0


def test_the_secondary_figures_name_the_labels_they_averaged_over(
    artifact: Artifact,
    whole_catalog_queries: QuerySet,
    whole_catalog: Catalog,
    corpus: list[Skill],
    config: RunConfig,
    make_result,
) -> None:
    """Verify secondary metrics state the exact subset of labels evaluated."""
    assert artifact.scores.not_headline.labels == (LIFECYCLE, RETENTION)

    single = assemble(
        [make_result("q-lifecycle", LIFECYCLE)],
        whole_catalog_queries,
        whole_catalog,
        corpus,
        config,
    )
    assert single.scores.not_headline.labels == (LIFECYCLE,)
    assert [s.skill for s in single.skills] == [LIFECYCLE, RETENTION, BASICS]
    assert single.scores.not_headline.macro_f1 == 1.0
    assert sum(s.recall or 0.0 for s in single.skills) / len(single.skills) < 1.0


def test_a_result_for_an_unlabeled_query_is_fatal(
    whole_catalog_queries: QuerySet,
    whole_catalog: Catalog,
    corpus: list[Skill],
    config: RunConfig,
    make_result,
) -> None:
    """Verify KeyError is raised when probe result references unknown query ID."""
    with pytest.raises(KeyError, match="unlabeled"):
        assemble(
            [make_result("q-nowhere", LIFECYCLE)],
            whole_catalog_queries,
            whole_catalog,
            corpus,
            config,
        )


def test_a_catalog_naming_a_skill_that_is_not_loaded_is_fatal(
    whole_catalog_results: list[ProbeResult],
    whole_catalog_queries: QuerySet,
    corpus: list[Skill],
    config: RunConfig,
) -> None:
    """Verify KeyError is raised when catalog contains skills missing from loaded corpus."""
    catalog = Catalog(id="all", mode=CatalogMode.ALL, skills=(LIFECYCLE, "ghost"))
    with pytest.raises(KeyError, match="ghost"):
        assemble(
            whole_catalog_results,
            whole_catalog_queries,
            catalog,
            corpus,
            config,
        )


@pytest.mark.parametrize(
    "field",
    ["config_fingerprint", "corpus_digest", "queries_digest"],
)
def test_rows_that_disagree_with_the_material_in_hand_are_refused(
    whole_catalog_results: list[ProbeResult],
    whole_catalog_queries: QuerySet,
    whole_catalog: Catalog,
    corpus: list[Skill],
    config: RunConfig,
    field: str,
) -> None:
    """Verify ValueError is raised when result rows have conflicting provenance digest values."""
    stamped = [r.model_copy(update={field: "deadbeef1234"}) for r in whole_catalog_results]
    with pytest.raises(ValueError, match=field):
        assemble(stamped, whole_catalog_queries, whole_catalog, corpus, config)


def test_rows_from_the_shallower_plan_that_preceded_this_one_are_kept(
    whole_catalog_results: list[ProbeResult],
    whole_catalog_queries: QuerySet,
    whole_catalog: Catalog,
    corpus: list[Skill],
    config: RunConfig,
) -> None:
    """Verify results from prior incremental runs under the same condition are retained."""
    shallow = config.with_overrides(plan={"attempts": 1})
    assert shallow.fingerprint != config.fingerprint
    rows = [
        r.model_copy(
            update={
                "config_fingerprint": (
                    shallow.fingerprint if r.attempt == 1 else config.fingerprint
                ),
                "condition_digest": config.condition,
            },
        )
        for r in whole_catalog_results
    ]
    artifact = assemble(
        rows,
        whole_catalog_queries,
        whole_catalog,
        corpus,
        config,
    )
    assert artifact.probes == len(whole_catalog_results)
    assert artifact.provenance.condition == config.condition
    assert "condition_digest" in artifact.verified_digests
    assert "config_fingerprint" in artifact.verified_digests


def test_a_stray_fingerprint_is_still_refused_when_the_condition_differs(
    whole_catalog_results: list[ProbeResult],
    whole_catalog_queries: QuerySet,
    whole_catalog: Catalog,
    corpus: list[Skill],
    config: RunConfig,
) -> None:
    """Verify mismatched condition digests reject result rows even if fingerprint differs."""
    rows = [
        r.model_copy(
            update={
                "config_fingerprint": "deadbeef1234",
                "condition_digest": "0ther0ther0t",
            },
        )
        for r in whole_catalog_results
    ]
    with pytest.raises(ValueError, match="measured something else"):
        assemble(rows, whole_catalog_queries, whole_catalog, corpus, config)


def test_a_condition_that_disagrees_is_refused_on_its_own_account(
    whole_catalog_results: list[ProbeResult],
    whole_catalog_queries: QuerySet,
    whole_catalog: Catalog,
    corpus: list[Skill],
    config: RunConfig,
) -> None:
    """Verify mismatched condition digest raises ValueError when fingerprint matches."""
    rows = [
        r.model_copy(
            update={
                "config_fingerprint": config.fingerprint,
                "condition_digest": "0ther0ther0t",
            },
        )
        for r in whole_catalog_results
    ]
    with pytest.raises(ValueError, match="condition_digest"):
        assemble(rows, whole_catalog_queries, whole_catalog, corpus, config)


def test_artifact_resolved_model_provenance_and_conflict_refusal(
    artifact: Artifact,
    whole_catalog_results: list[ProbeResult],
    whole_catalog_queries: QuerySet,
    whole_catalog: Catalog,
    corpus: list[Skill],
    config: RunConfig,
) -> None:
    """Verify resolved_model defaults empty, records single model, and refuses conflicts."""
    assert artifact.provenance.model
    assert artifact.provenance.resolved_model == ""

    rows = [
        r.model_copy(update={"resolved_model": "claude-sonnet-5"}) for r in whole_catalog_results
    ]
    built = assemble(
        rows,
        whole_catalog_queries,
        whole_catalog,
        corpus,
        config,
    )
    assert built.provenance.model == "sonnet"
    assert built.provenance.resolved_model == "claude-sonnet-5"

    mixed = [
        *rows,
        whole_catalog_results[0].model_copy(
            update={"resolved_model": "claude-sonnet-4-5"},
        ),
    ]
    with pytest.raises(ValueError, match="resolved_model"):
        assemble(mixed, whole_catalog_queries, whole_catalog, corpus, config)


def test_rows_from_two_runtimes_are_refused(
    whole_catalog_results: list[ProbeResult],
    whole_catalog_queries: QuerySet,
    whole_catalog: Catalog,
    corpus: list[Skill],
    config: RunConfig,
) -> None:
    """Verify ValueError is raised when results mix different runtime agents."""
    mixed = [
        *whole_catalog_results,
        whole_catalog_results[0].model_copy(update={"runtime": "codex"}),
    ]
    with pytest.raises(ValueError, match="runtime"):
        assemble(mixed, whole_catalog_queries, whole_catalog, corpus, config)


def test_the_artifact_survives_a_round_trip_through_json(
    artifact: Artifact,
) -> None:
    """Verify Artifact round-trips losslessly through JSON serialization."""
    assert Artifact.model_validate_json(artifact.model_dump_json()) == artifact


def test_an_artifact_with_an_unstated_digest_is_refused(artifact: Artifact) -> None:
    """Verify Artifact validation requires all provenance digests to be non-empty."""
    payload = json.loads(artifact.model_dump_json())
    payload["digests"]["corpus_digest"] = ""
    with pytest.raises(ValueError, match="corpus_digest"):
        Artifact.model_validate(payload)


def test_an_artifact_from_a_later_major_version_is_refused(
    artifact: Artifact,
) -> None:
    """Verify major version mismatch raises ValueError during deserialization."""
    payload = json.loads(artifact.model_dump_json())
    payload["schema_version"] = f"{int(SCHEMA_VERSION.split('.')[0]) + 1}.0"
    with pytest.raises(ValueError, match="schema_version"):
        Artifact.model_validate(payload)
    payload["schema_version"] = f"{SCHEMA_VERSION.split('.')[0]}.99"
    assert Artifact.model_validate(payload).schema_version.endswith(".99")


@pytest.mark.parametrize(
    ("probes", "recall"),
    [
        pytest.param(0, 0.0, id="unmeasured-but-scored"),
        pytest.param(3, None, id="measured-but-unscored"),
    ],
)
def test_recall_is_present_exactly_when_a_skill_was_asked_for(
    probes: int,
    recall: float | None,
) -> None:
    """Verify recall is present if and only if probes > 0."""
    with pytest.raises(ValueError, match="recall"):
        SkillScore(skill=LIFECYCLE, probes=probes, reached=0, recall=recall)


@pytest.mark.parametrize(
    ("reached", "absorbed", "precision"),
    [
        pytest.param(0, 0, 0.0, id="never-selected-but-scored"),
        pytest.param(2, 0, None, id="reached-but-unscored"),
        pytest.param(0, 2, None, id="absorbed-but-unscored"),
    ],
)
def test_precision_is_present_exactly_when_a_skill_was_selected(
    reached: int,
    absorbed: int,
    precision: float | None,
) -> None:
    """Verify precision is defined if and only if reached + absorbed > 0."""
    with pytest.raises(ValueError, match="precision"):
        SkillScore(
            skill=LIFECYCLE,
            probes=2,
            recall=float(reached) / 2,
            reached=reached,
            absorbed=absorbed,
            precision=precision,
        )


@pytest.mark.parametrize(
    ("fields", "match"),
    [
        pytest.param(
            {
                "replicates": 2,
                "top1_by_attempt": (1.0,),
                "mean": 1.0,
                "standard_error": 0.0,
            },
            "2 replicates but 1 accuracies",
            id="fewer-accuracies-than-replicates",
        ),
        pytest.param(
            {"replicates": 1, "top1_by_attempt": (1.0,), "standard_error": 0.0},
            "cannot mean to None",
            id="replicates-with-no-mean",
        ),
        pytest.param(
            {"replicates": 0, "top1_by_attempt": (), "mean": 0.5},
            "cannot mean to 0.5",
            id="mean-with-no-replicates",
        ),
        pytest.param(
            {"replicates": 1, "top1_by_attempt": (1.0,), "mean": 1.0},
            "error on its estimate",
            id="replicates-with-no-error",
        ),
        pytest.param(
            {"replicates": 0, "top1_by_attempt": (), "standard_error": 0.0},
            "error on its estimate",
            id="error-with-no-replicates",
        ),
    ],
)
def test_a_spread_cannot_summarize_observations_nobody_made(
    fields: dict[str, Any],
    match: str,
) -> None:
    """Verify Spread validator enforces consistency between replicates, accuracies, and error."""
    with pytest.raises(ValueError, match=match):
        Spread(**fields)


def test_a_pair_cannot_collide_more_often_than_it_occurred() -> None:
    """Verify ConfusionPair validates that collision count does not exceed probe count."""
    with pytest.raises(ValueError, match="collisions"):
        ConfusionPair(expected=LIFECYCLE, invoked=BASICS, probes=1, collisions=2)


def test_an_artifact_written_is_an_artifact_read_back(
    artifact: Artifact,
    tmp_path: Path,
) -> None:
    """Verify write_artifact and read_artifact disk round trip preserves model equality."""
    destination = tmp_path / "runs" / "results.jsonl.artifact.json"

    written = write_artifact(artifact, destination)

    assert written == destination
    assert read_artifact(destination) == artifact


def test_the_artifact_is_written_where_a_reader_would_look_for_it(
    tmp_path: Path,
) -> None:
    """Verify artifact_path generates expected .artifact.json sidecar naming."""
    assert artifact_path(tmp_path / "run.jsonl") == tmp_path / "run.jsonl.artifact.json"


def test_the_written_artifact_is_readable_by_something_that_is_not_this_tool(
    artifact: Artifact,
    tmp_path: Path,
) -> None:
    """Verify write_artifact emits pretty-formatted JSON with trailing newline."""
    path = write_artifact(artifact, tmp_path / "a.artifact.json")
    text = path.read_text(encoding="utf-8")

    assert text.endswith("\n")
    assert "\n  " in text
    assert json.loads(text)["schema_version"] == SCHEMA_VERSION


def test_a_file_that_is_not_an_artifact_is_refused_at_the_boundary(
    tmp_path: Path,
) -> None:
    """Verify read_artifact raises validation error when reading invalid JSON payload."""
    path = tmp_path / "broken.artifact.json"
    path.write_text('{"schema_version": "0.1"}', encoding="utf-8")

    with pytest.raises(PydanticValidationError):
        read_artifact(path)


def test_an_artifact_will_not_be_written_over_the_rows_it_summarizes(
    artifact: Artifact,
    tmp_path: Path,
) -> None:
    """Verify write_artifact raises error when destination is raw results file."""
    rows = tmp_path / "run.jsonl"
    rows.write_text('{"query_id": "q1"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="not an artifact"):
        write_artifact(artifact, rows)

    assert rows.read_text(encoding="utf-8") == '{"query_id": "q1"}\n'


def test_writing_over_an_artifact_leaves_no_trace_of_the_old_one(
    artifact: Artifact,
    tmp_path: Path,
) -> None:
    """Verify overwriting an artifact completely replaces previous file content."""
    path = tmp_path / "a.artifact.json"
    write_artifact(artifact, path)
    write_artifact(artifact.model_copy(update={"catalog_id": "second"}), path)

    assert read_artifact(path).catalog_id == "second"


def test_run_artifact_deserializes_minimal_schema_roundtrip(
    artifact: Artifact,
    tmp_path: Path,
) -> None:
    """Verify Artifact tolerates JSON payloads that omit optional fields."""
    raw = artifact.model_dump(mode="json")
    raw.pop("verified_digests", None)
    raw.pop("contested_skills", None)
    raw.pop("resolved_roots", None)

    path = tmp_path / "minimal.artifact.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    loaded = read_artifact(path)
    assert loaded.schema_version == artifact.schema_version
    assert loaded.catalog_id == artifact.catalog_id
    assert loaded.verified_digests == ()
    assert loaded.contested_skills == ()
    assert loaded.resolved_roots == ()


def test_artifact_assemble_accepts_optional_metadata(
    whole_catalog_results: list[ProbeResult],
    whole_catalog_queries: QuerySet,
    whole_catalog: Catalog,
    corpus: list[Skill],
    config: RunConfig,
    skill_repo: Path,
) -> None:
    """Verify Artifact.assemble respects optional keyword arguments."""
    composition = Composition(
        config=config,
        query_set=whole_catalog_queries,
        catalog=whole_catalog,
        skills=tuple(corpus),
    )
    root = skill_repo.resolve()
    fit = CatalogFit(
        allowed=50,
        asked=10,
        unit="display columns",
        truncated=0,
        remedy="widen it",
    )
    artifact = Artifact.assemble(
        composition,
        whole_catalog_results,
        roots=[root],
        fit=fit,
        spend_usd=1.25,
        reused=3,
    )
    assert artifact.spend_usd == 1.25
    assert artifact.reused == 3
    assert artifact.provenance.catalog_fit == fit
    assert [(r.path, r.skills) for r in artifact.resolved_roots] == [(root, 3)]


@pytest.mark.parametrize(
    (
        "invoked_skills",
        "expected_top1_reached",
        "expected_top1_recall",
        "expected_traj_reached",
        "expected_traj_recall",
        "expected_in_unreached",
    ),
    [
        pytest.param(
            (LIFECYCLE,),
            1,
            1.0,
            1,
            1.0,
            False,
            id="turn-1-direct-hit",
        ),
        pytest.param(
            (BASICS, LIFECYCLE),
            0,
            0.0,
            1,
            1.0,
            False,
            id="turn-2-trajectory-hit-not-unreached",
        ),
        pytest.param(
            (BASICS,),
            0,
            0.0,
            0,
            0.0,
            True,
            id="never-reached-in-any-turn",
        ),
    ],
)
def test_skill_score_and_unreached_respect_multi_turn_trajectory(
    whole_catalog: Catalog,
    corpus: list[Skill],
    make_config: Any,
    invoked_skills: tuple[str, ...],
    expected_top1_reached: int,
    expected_top1_recall: float,
    expected_traj_reached: int,
    expected_traj_recall: float,
    expected_in_unreached: bool,
) -> None:
    """Verify SkillScore tracks trajectory_reached/recall and unreached excludes turn 2+ hits."""
    cfg = make_config(catalog={"mode": CatalogMode.ALL}, plan={"attempts": 1})
    qs = QuerySet(
        catalog_id=whole_catalog.id,
        queries=(Query(id="q-life", text="configure lifecycle", expected_skill=LIFECYCLE),),
        provenance=QuerySetProvenance(origin=Origin.AUTHORED),
    )
    res = [
        ProbeResult(
            query_id="q-life",
            catalog_id=whole_catalog.id,
            catalog_mode=CatalogMode.ALL,
            catalog_size=len(whole_catalog.skills),
            invoked_skills=invoked_skills,
            runtime=cfg.runtime.agent,
            model="sonnet",
            config_fingerprint=cfg.fingerprint,
            corpus_digest=corpus_digest(corpus),
            queries_digest=query_set_digest(qs),
        )
    ]
    built = assemble(res, qs, whole_catalog, corpus, cfg)
    score = next(s for s in built.skills if s.skill == LIFECYCLE)

    assert score.reached == expected_top1_reached
    assert score.recall == expected_top1_recall
    assert score.trajectory_reached == expected_traj_reached
    assert score.trajectory_recall == expected_traj_recall
    assert (score in built.unreached) is expected_in_unreached


def test_confusion_pairs_retains_turn1_collision_on_multi_turn_trajectory_hit(
    whole_catalog: Catalog,
    corpus: list[Skill],
    make_config: Any,
) -> None:
    """Verify assemble confusion pairs record turn-1 misroute even when turn-2 recovers."""
    cfg = make_config(catalog={"mode": CatalogMode.ALL}, plan={"attempts": 1})
    qs = QuerySet(
        catalog_id=whole_catalog.id,
        queries=(Query(id="q-life", text="configure lifecycle", expected_skill=LIFECYCLE),),
        provenance=QuerySetProvenance(origin=Origin.AUTHORED),
    )
    res = [
        ProbeResult(
            query_id="q-life",
            catalog_id=whole_catalog.id,
            catalog_mode=CatalogMode.ALL,
            catalog_size=len(whole_catalog.skills),
            invoked_skills=(BASICS, LIFECYCLE),
            runtime=cfg.runtime.agent,
            model="sonnet",
            config_fingerprint=cfg.fingerprint,
            corpus_digest=corpus_digest(corpus),
            queries_digest=query_set_digest(qs),
        )
    ]
    built = assemble(res, qs, whole_catalog, corpus, cfg)
    assert len(built.confusion) == 1
    assert built.confusion[0].expected == LIFECYCLE
    assert built.confusion[0].invoked == BASICS
    assert built.confusion[0].collisions == 1
    assert built.scores.trajectory_reachability == 1.0


def test_symlinked_skill_is_attributed_to_enclosing_root(
    whole_catalog_results: list[ProbeResult],
    whole_catalog_queries: QuerySet,
    skill_repo: Path,
    config: RunConfig,
    tmp_path: Path,
    write_skill: Any,
) -> None:
    """Verify skills symlinked inside a corpus root are attributed in resolved_roots."""
    from reach.catalog import build_catalogs, load_skills

    external_dir = tmp_path / "external_store" / "symlinked-skill"
    write_skill(
        name="symlinked-skill",
        description="A skill stored outside the repo and linked via symlink.",
        path=external_dir,
    )
    link_path = skill_repo / "symlinked-skill"
    link_path.symlink_to(external_dir, target_is_directory=True)

    loaded = load_skills(skill_repo)
    assert any(s.name == "symlinked-skill" for s in loaded)
    catalog = build_catalogs(loaded, CatalogMode.ALL)[0]

    built = assemble(
        whole_catalog_results,
        whole_catalog_queries,
        catalog,
        loaded,
        config,
        roots=[skill_repo],
    )
    assert [(r.path, r.skills) for r in built.resolved_roots] == [(skill_repo.resolve(), 4)]
    symlinked_score = next(s for s in built.skills if s.skill == "symlinked-skill")
    assert symlinked_score.root == skill_repo.resolve()
