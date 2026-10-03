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

"""Verify property-based invariants for statistical intervals, BM25 scoring, and serialization."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import NamedTuple, Protocol

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from reach.artifact import (
    Abstention,
    Artifact,
    NotHeadline,
    QueryRecord,
    RunProvenance,
    RunScores,
    SkillScore,
    Spread,
)
from reach.diff import Arm, VaryFactor, diff_arms, noise_floor
from reach.exchange import Exchange, export_query_set, import_query_set
from reach.metrics import (
    classification_report,
    classify_invocation_pattern,
    decompose_pass_rate_drop,
    score_trajectory,
)
from reach.models import (
    CatalogMode,
    InvocationPattern,
    ProbeResult,
    Provenance,
    Query,
    QueryKind,
    Skill,
)
from reach.queries import Origin, QuerySet, QuerySetProvenance
from reach.retrieval import Bm25Scorer
from reach.uncertainty import (
    cluster_wilson_interval,
    detectable_delta,
    effective_sample_size,
    required_probes,
    wilson_interval,
)


@st.composite
def hits_and_probes(draw: st.DrawFn) -> tuple[int, int]:
    """Generate valid (hits, probes) counts where probes >= 1 and hits <= probes."""
    probes = draw(st.integers(min_value=1, max_value=2000))
    hits = draw(st.integers(min_value=0, max_value=probes))
    return hits, probes


@given(hits_and_probes(), st.floats(min_value=0.5, max_value=0.99))
def test_the_interval_always_bounds_a_rate(
    counts: tuple[int, int],
    confidence: float,
) -> None:
    """Verify Wilson intervals fall strictly within [0, 1] with lower <= upper."""
    hits, probes = counts
    interval = wilson_interval(hits, probes, confidence)

    assert interval is not None
    assert 0.0 <= interval.low <= interval.high <= 1.0
    assert not interval.excludes(hits / probes)
    assert interval.width >= 0.0


@given(hits_and_probes(), st.floats(min_value=0.5, max_value=0.99))
def test_doubling_the_evidence_at_a_fixed_rate_does_not_widen_the_interval(
    counts: tuple[int, int],
    confidence: float,
) -> None:
    """Verify doubling sample size at constant hit rate does not increase interval width."""
    hits, probes = counts
    shallow = wilson_interval(hits, probes, confidence)
    deep = wilson_interval(2 * hits, 2 * probes, confidence)

    assert shallow is not None
    assert deep is not None
    assert deep.width <= shallow.width + 1e-9


@given(
    delta=st.floats(min_value=0.02, max_value=1.0),
    confidence=st.floats(min_value=0.5, max_value=0.99),
    power=st.floats(min_value=0.5, max_value=0.99),
)
def test_required_probes_and_detectable_delta_are_exact_inverses(
    delta: float,
    confidence: float,
    power: float,
) -> None:
    """Verify required_probes is minimal sample size achieving detectable_delta <= delta."""
    n = required_probes(delta, confidence=confidence, power=power)
    achieved = detectable_delta(n, confidence=confidence, power=power)

    assert n >= 1
    assert achieved is not None
    assert achieved <= delta + 1e-9
    if n > 1:
        underpowered = detectable_delta(n - 1, confidence=confidence, power=power)
        assert underpowered is not None
        assert underpowered == 1.0 or underpowered > delta - 1e-9


@given(
    counts=hits_and_probes(),
    attempts=st.integers(min_value=1, max_value=20),
    icc_pair=st.tuples(
        st.floats(min_value=0.0, max_value=1.0),
        st.floats(min_value=0.0, max_value=1.0),
    ).map(sorted),
)
def test_clustering_reduces_effective_sample_size_and_widens_intervals(
    counts: tuple[int, int],
    attempts: int,
    icc_pair: list[float],
) -> None:
    """Verify cluster design effect shrinks effective sample size and widens intervals."""
    hits, probes = counts
    icc_low, icc_high = icc_pair[0], icc_pair[1]

    neff_1 = effective_sample_size(probes, attempts=1, intra_cluster_correlation=icc_low)
    neff_k = effective_sample_size(probes, attempts=attempts, intra_cluster_correlation=icc_low)
    neff_next = effective_sample_size(
        probes,
        attempts=attempts + 1,
        intra_cluster_correlation=icc_low,
    )
    neff_high_icc = effective_sample_size(
        probes,
        attempts=attempts,
        intra_cluster_correlation=icc_high,
    )

    assert 1 <= neff_next <= neff_k <= neff_1 == probes
    assert neff_high_icc <= neff_k

    unclustered = cluster_wilson_interval(
        hits,
        probes,
        attempts=1,
        intra_cluster_correlation=icc_low,
    )
    clustered = cluster_wilson_interval(
        hits,
        probes,
        attempts=attempts,
        intra_cluster_correlation=icc_low,
    )
    more_attempts = cluster_wilson_interval(
        hits,
        probes,
        attempts=attempts + 1,
        intra_cluster_correlation=icc_low,
    )
    more_correlated = cluster_wilson_interval(
        hits,
        probes,
        attempts=attempts,
        intra_cluster_correlation=icc_high,
    )

    assert unclustered is not None
    assert clustered is not None
    assert more_attempts is not None
    assert more_correlated is not None
    assert unclustered.width <= clustered.width + 1e-9
    assert clustered.width <= more_attempts.width + 1e-9
    assert clustered.width <= more_correlated.width + 1e-9


#: Restricted alphabet to ensure generated skills share overlapping terms.
_TERM = st.text(alphabet="abcdefgh", min_size=1, max_size=4)


def _skill(name: str, description: str) -> Skill:
    """Build in-memory Skill instance without reading filesystem."""
    return Skill(name=name, description=description, path=Path(name))


@st.composite
def skill_corpora(draw: st.DrawFn) -> list[Skill]:
    """Generate synthetic skill corpus with unique skill names."""
    names = draw(
        st.lists(
            st.text(alphabet="abcdefgh", min_size=1, max_size=6),
            min_size=1,
            max_size=8,
            unique=True,
        ),
    )
    descriptions = draw(
        st.lists(
            st.lists(_TERM, min_size=1, max_size=6).map(" ".join),
            min_size=len(names),
            max_size=len(names),
        ),
    )
    return [
        _skill(name, description) for name, description in zip(names, descriptions, strict=True)
    ]


@settings(max_examples=35)
@given(skill_corpora(), st.lists(_TERM, max_size=6))
def test_bm25_scores_are_never_negative(skills: list[Skill], query: list[str]) -> None:
    """Verify BM25 scores are non-negative across all skills and queries."""
    scorer = Bm25Scorer.from_skills(skills)

    for skill in skills:
        assert scorer.score(query, skill.name) >= 0.0


@settings(max_examples=35)
@given(skill_corpora(), st.data())
def test_bm25_ranking_does_not_depend_on_input_order(
    skills: list[Skill],
    data: st.DataObject,
) -> None:
    """Verify BM25 ranking is invariant under arbitrary skill corpus permutations."""
    target = skills[0]
    shuffled = data.draw(st.permutations(skills))

    ranked = Bm25Scorer.from_skills(skills).rank(target, skills)
    reranked = Bm25Scorer.from_skills(shuffled).rank(target, shuffled)

    assert [name for name, _ in ranked] == [name for name, _ in reranked]
    for (_, score), (_, reordered_score) in zip(ranked, reranked, strict=True):
        assert score == reordered_score


#: Punctuation and whitespace characters for fuzzing query text serialization.
_AWKWARD = "\n\r\t,\"'\u200b"
_TEXT_CHAR = st.one_of(
    st.characters(
        min_codepoint=0x20,
        max_codepoint=0x10FFFF,
        exclude_categories=("Cs", "Co"),
    ),
    st.sampled_from(_AWKWARD),
)
_QUERY_TEXT = st.text(alphabet=_TEXT_CHAR, min_size=1, max_size=80).filter(
    lambda text: text.strip(),
)


@given(fmt=st.sampled_from(list(Exchange)), text=_QUERY_TEXT)
def test_export_then_import_reproduces_the_query_text_exactly(
    fmt: Exchange,
    text: str,
) -> None:
    """Verify query text round-trips exactly through export and import across all formats."""
    original = QuerySet(
        catalog_id="all",
        queries=(Query(id="q-1", text=text, expected_skill="some-skill"),),
        provenance=QuerySetProvenance(origin=Origin.AUTHORED),
    )

    document = export_query_set(original, fmt)
    imported = import_query_set(document, fmt, catalog_id="all")

    assert imported.queries[0].text == text


_SKILL_POOL = ("s-alpha", "s-beta", "s-gamma", "s-delta")


@st.composite
def generated_query(draw: st.DrawFn, query_id: str = "q-1") -> Query:
    """Generate an in-scope or out-of-scope Query over a bounded skill pool."""
    is_oos = draw(st.booleans())
    if is_oos:
        return Query(id=query_id, text="unhandled request", kind=QueryKind.OUT_OF_SCOPE)
    expected = draw(st.sampled_from(_SKILL_POOL))
    remaining = [s for s in _SKILL_POOL if s != expected]
    acceptable = tuple(draw(st.lists(st.sampled_from(remaining), max_size=1, unique=True)))
    return Query(
        id=query_id,
        text="handled request",
        expected_skill=expected,
        acceptable_skills=acceptable,
    )


@given(
    query=generated_query(),
    invoked=st.lists(st.sampled_from(_SKILL_POOL), max_size=6),
)
def test_score_trajectory_and_invocation_pattern_invariants(
    query: Query,
    invoked: list[str],
) -> None:
    """Verify trajectory score bounds, implication ordering, and pattern agreement."""
    score = score_trajectory(query, invoked)
    pattern = classify_invocation_pattern(query, invoked)

    if score.entrypoint_hit:
        assert score.trajectory_hit
    assert 0.0 <= score.step_efficiency <= 1.0
    assert 0.0 <= score.skill_f1 <= 1.0
    assert (score.step_efficiency > 0.0) == score.trajectory_hit
    assert (score.skill_f1 > 0.0) == score.trajectory_hit
    assert score.redundancy >= 0

    # Neutral acceptable skills do not alter trajectory scores once stripped
    assert score == score_trajectory(query, query.scored_invocations(invoked))

    passing_patterns = {
        InvocationPattern.ORACLE_ONLY,
        InvocationPattern.MIXED_ORACLE,
        InvocationPattern.CORRECT_ABSTENTION,
    }
    assert (pattern in passing_patterns) == score.trajectory_hit


class _EvaluatedRun(NamedTuple):
    """Hold generated query definitions alongside baseline and scaled probe runs."""

    queries: list[Query]
    baseline: list[ProbeResult]
    scaled: list[ProbeResult]


@st.composite
def evaluated_run(draw: st.DrawFn) -> _EvaluatedRun:
    """Generate a query set alongside baseline and scaled probe result runs."""
    n_queries = draw(st.integers(min_value=1, max_value=5))
    queries = [draw(generated_query(query_id=f"q-{i}")) for i in range(n_queries)]

    def _draw_results() -> list[ProbeResult]:
        rows: list[ProbeResult] = []
        for q in queries:
            attempts = draw(st.integers(min_value=1, max_value=3))
            for attempt in range(1, attempts + 1):
                errored = draw(st.booleans())
                invoked = (
                    ()
                    if errored
                    else tuple(draw(st.lists(st.sampled_from(_SKILL_POOL), max_size=4)))
                )
                rows.append(
                    ProbeResult(
                        query_id=q.id,
                        attempt=attempt,
                        catalog_id="all",
                        catalog_mode=CatalogMode.ALL,
                        catalog_size=len(_SKILL_POOL),
                        model="test-model",
                        runtime="fake",
                        invoked_skills=invoked,
                        error="timeout" if errored else None,
                    ),
                )
        return rows

    return _EvaluatedRun(queries=queries, baseline=_draw_results(), scaled=_draw_results())


@given(evaluated_run())
def test_classification_report_and_decomposition_conservation(
    run_data: _EvaluatedRun,
) -> None:
    """Verify count conservation in classification_report and additive loss in decomposition."""
    report = classification_report(run_data.baseline, run_data.queries)

    assert report.probes == len(run_data.baseline)
    assert report.scored == report.probes - report.errors
    assert report.in_scope + report.out_of_scope == report.scored
    assert sum(c.support for c in report.per_class) == report.scored
    assert sum(c.predicted for c in report.per_class) == report.scored
    assert sum(c.true_positives for c in report.per_class) == report.top1_hits
    assert 0 <= report.entrypoint_hits <= report.trajectory_hits <= report.scored
    assert 0 <= report.false_abstentions <= report.abstentions <= report.scored

    for cls in report.per_class:
        assert cls.true_positives + cls.false_negatives == cls.support
        assert cls.true_positives + cls.false_positives == cls.predicted
        assert cls.recall <= cls.trajectory_recall + 1e-9

    decomp = decompose_pass_rate_drop(
        run_data.baseline,
        run_data.scaled,
        queries=run_data.queries,
        iterations=10,
    )
    assert 0.0 <= decomp.baseline_pass_rate <= 1.0
    assert 0.0 <= decomp.scaled_pass_rate <= 1.0
    assert decomp.baseline_pass_rate - decomp.scaled_pass_rate == pytest.approx(
        decomp.delta_total,
        abs=1e-9,
    )
    assert decomp.delta_context + decomp.delta_shadowing == pytest.approx(
        decomp.delta_total,
        abs=1e-9,
    )


@given(
    se_a=st.floats(min_value=0.001, max_value=0.5),
    se_b=st.floats(min_value=0.001, max_value=0.5),
    confidence=st.floats(min_value=0.5, max_value=0.98),
    inflation=st.floats(min_value=0.5, max_value=3.0),
)
def test_noise_floor_symmetry_and_monotonicity(
    se_a: float,
    se_b: float,
    confidence: float,
    inflation: float,
) -> None:
    """Verify noise_floor is symmetric in arm standard errors and strictly monotonic."""
    base = noise_floor(se_a, se_b, confidence=confidence, noise_inflation=inflation)
    swapped = noise_floor(se_b, se_a, confidence=confidence, noise_inflation=inflation)
    stricter = noise_floor(se_a, se_b, confidence=confidence + 0.01, noise_inflation=inflation)
    inflated = noise_floor(se_a, se_b, confidence=confidence, noise_inflation=inflation + 0.1)

    assert base == pytest.approx(swapped, abs=1e-12)
    assert 0.0 < base < stricter
    assert base < inflated


def _synthetic_arm(
    label: str,
    corpus_digest: str,
    query_counts: list[tuple[int, int]],
    se: float,
) -> Arm:
    """Construct an in-memory Arm with consistent counts and spread for diff testing."""
    total_probes = sum(p for _, p in query_counts)
    total_hits = sum(h for h, _ in query_counts)
    acc = total_hits / total_probes
    queries = tuple(
        QueryRecord(
            query_id=f"q-{idx}",
            text=f"query {idx}",
            expected="s-alpha",
            probes=p,
            hits=h,
        )
        for idx, (h, p) in enumerate(query_counts)
    )
    skill = SkillScore(
        skill="s-alpha",
        probes=total_probes,
        reached=total_hits,
        recall=acc,
        trajectory_reached=total_hits,
        trajectory_recall=acc,
        absorbed=0,
        precision=1.0 if total_hits > 0 else None,
    )
    artifact = Artifact(
        digests=Provenance(
            config_fingerprint="fp-1",
            condition_digest="cond-1",
            corpus_digest=corpus_digest,
            queries_digest="q-digest-1",
        ),
        provenance=RunProvenance(
            runtime="fake",
            model="test-model",
            attempts=2,
            arm="arm-1",
        ),
        catalog_id="all",
        catalog_mode=CatalogMode.ALL,
        catalog_size=1,
        skills=(skill,),
        scores=RunScores(
            consistency=1.0,
            top1_accuracy=acc,
            abstention=Abstention(
                rate=0.0,
                false_rate=0.0,
                scored=total_probes,
                abstentions=0,
                in_scope=total_probes,
                false_abstentions=0,
            ),
            not_headline=NotHeadline(macro_f1=acc),
            unanimous_queries=len(query_counts),
            observed_queries=len(query_counts),
            top1_hits=total_hits,
            entrypoint_hits=total_hits,
            entrypoint_accuracy=acc,
            trajectory_hits=total_hits,
            trajectory_reachability=acc,
            scored=total_probes,
        ),
        spread=Spread(
            replicates=2,
            top1_by_attempt=(acc, acc),
            mean=acc,
            repeated_queries=len(query_counts),
            standard_error=se,
        ),
        queries=queries,
        probes=total_probes,
    )
    return Arm(label=label, artifact=artifact, source_queries_digest="q-digest-1")


class _DeltaRecord(Protocol):
    """Represent a skill or query delta record exposing delta and real significance."""

    @property
    def delta(self) -> float: ...

    @property
    def real(self) -> bool: ...


def _assert_deltas_antisymmetric[T: _DeltaRecord](
    forward_items: Sequence[T],
    reverse_items: Sequence[T],
    key: Callable[[T], str],
) -> None:
    """Assert that reversing arms negates deltas and preserves significance per item."""
    rev_by_key = {key(item): item for item in reverse_items}
    for fwd in forward_items:
        rev = rev_by_key[key(fwd)]
        assert fwd.delta == pytest.approx(-rev.delta, abs=1e-12)
        assert fwd.real == rev.real


@given(
    paired_counts=st.lists(
        st.tuples(hits_and_probes(), hits_and_probes()),
        min_size=1,
        max_size=4,
    ),
    se_control=st.floats(min_value=0.01, max_value=0.25),
    se_treatment=st.floats(min_value=0.01, max_value=0.25),
    confidence=st.floats(min_value=0.80, max_value=0.99),
)
def test_diff_arms_antisymmetry_under_arm_swap(
    paired_counts: list[tuple[tuple[int, int], tuple[int, int]]],
    se_control: float,
    se_treatment: float,
    confidence: float,
) -> None:
    """Verify swapping control and treatment negates deltas while preserving significance."""
    control_counts = [c for c, _ in paired_counts]
    treatment_counts = [t for _, t in paired_counts]
    control = _synthetic_arm("control", "corpus-a", control_counts, se_control)
    treatment = _synthetic_arm("treatment", "corpus-b", treatment_counts, se_treatment)

    forward = diff_arms(control, treatment, VaryFactor.DESCRIPTION, confidence=confidence)
    reverse = diff_arms(treatment, control, VaryFactor.DESCRIPTION, confidence=confidence)

    assert forward.headline.delta == pytest.approx(-reverse.headline.delta, abs=1e-12)
    assert forward.headline.floor == pytest.approx(reverse.headline.floor, abs=1e-12)
    assert forward.headline.real == reverse.headline.real
    assert forward.headline.needed_probes == reverse.headline.needed_probes
    assert forward.corroboration.corroborated is True
    assert reverse.corroboration.corroborated is True
    assert {q.query_id for q in forward.separated} == {q.query_id for q in reverse.separated}

    _assert_deltas_antisymmetric(forward.skills, reverse.skills, lambda s: s.skill)
    _assert_deltas_antisymmetric(forward.queries, reverse.queries, lambda q: q.query_id)
