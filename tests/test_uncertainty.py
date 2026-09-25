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

"""Verify Wilson confidence intervals and statistical power calculations."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError
from scipy.stats import binomtest

from reach.artifact import SCHEMA_VERSION, Abstention, Artifact, NotHeadline, RunScores
from reach.metrics import classification_report
from reach.models import Catalog, CatalogMode, ProbeResult, Skill
from reach.queries import Origin, QuerySet, QuerySetProvenance
from reach.report import build_report, render_csv, render_text
from reach.run import Composition
from reach.uncertainty import (
    DEFAULT_CONFIDENCE,
    DEFAULT_POWER,
    Interval,
    cluster_wilson_interval,
    critical_value,
    detectable_delta,
    effective_sample_size,
    required_probes,
    wilson_interval,
)

#: Grid of (hits, probes) tuples for boundary and sample test cases.
COUNTS = [
    (0, 1),
    (1, 1),
    (0, 5),
    (2, 5),
    (4, 5),
    (5, 5),
    (0, 20),
    (4, 20),
    (5, 20),
    (20, 20),
    (39, 50),
    (130, 150),
]

#: Mapping of test finding names to ((shallow_hits, shallow_probes), (deep_hits, deep_probes)).
DEPTH_PAIRS = {
    "WAF-05": ((0, 5), (5, 20)),
    "AGENT-01": ((0, 3), (0, 20)),
    "WAF-06": ((2, 5), (0, 20)),
    "WAF-03": ((3, 5), (4, 20)),
    "WAF-04": ((4, 5), (20, 20)),
}

#: Set of findings where deep rate falls outside the shallow published interval.
DEEP_RATE_OUTSIDE_PUBLISHED_INTERVAL = {"WAF-06", "WAF-03", "WAF-04"}


@pytest.mark.parametrize(("hits", "probes"), COUNTS)
def test_the_bounds_match_the_reference_implementation(hits: int, probes: int) -> None:
    """Verify Wilson interval bounds match scipy proportion_ci reference."""
    interval = wilson_interval(hits, probes)
    expected = binomtest(hits, probes).proportion_ci(method="wilson")
    assert interval is not None
    assert interval.low == pytest.approx(expected.low, abs=1e-12)
    assert interval.high == pytest.approx(expected.high, abs=1e-12)
    assert interval.confidence == DEFAULT_CONFIDENCE


@pytest.mark.parametrize(("hits", "probes"), COUNTS)
def test_an_interval_stays_inside_the_unit_and_covers_its_own_rate(
    hits: int,
    probes: int,
) -> None:
    """Verify interval bounds stay within [0.0, 1.0] and contain the observed sample rate."""
    interval = wilson_interval(hits, probes)
    assert interval is not None
    assert 0.0 <= interval.low <= interval.high <= 1.0
    assert not interval.excludes(hits / probes)


@pytest.mark.parametrize("probes", [1, 3, 5, 20, 150])
def test_a_unanimous_run_is_not_reported_as_certain(probes: int) -> None:
    """Verify boundary counts (0 or all hits) maintain a positive non-zero interval width."""
    for hits in (0, probes):
        interval = wilson_interval(hits, probes)
        assert interval is not None
        assert interval.width > 0.0


def test_depth_narrows_the_interval_at_a_fixed_rate() -> None:
    """Verify increasing probe depth strictly decreases interval width at constant rate."""
    widths = [wilson_interval(n // 2, n) for n in (4, 8, 20, 100, 400)]
    assert all(w is not None for w in widths)
    measured = [w.width for w in widths if w is not None]
    assert measured == sorted(measured, reverse=True)


def test_a_stricter_confidence_costs_width() -> None:
    """Verify higher confidence levels produce wider intervals."""
    loose = wilson_interval(5, 20, confidence=0.80)
    default = wilson_interval(5, 20)
    strict = wilson_interval(5, 20, confidence=0.99)
    assert loose is not None
    assert default is not None
    assert strict is not None
    assert loose.width < default.width < strict.width


@pytest.mark.parametrize(
    ("confidence", "expected"),
    [(0.80, 1.281552), (0.90, 1.644854), (0.95, 1.959964), (0.99, 2.575829)],
)
def test_the_deviate_a_confidence_is_read_off_is_the_published_one(
    confidence: float,
    expected: float,
) -> None:
    """Verify critical_value returns standard normal two-sided critical z values."""
    assert critical_value(confidence) == pytest.approx(expected, abs=1e-6)


def test_the_deviate_is_the_one_the_interval_was_built_from() -> None:
    """Verify critical_value quantile matches interval width at p=0.5."""
    probes, z = 400, critical_value()
    interval = wilson_interval(probes // 2, probes)
    assert interval is not None
    assert interval.width == pytest.approx(z / (probes * (1.0 + z**2 / probes)) ** 0.5)


def test_the_deviate_refuses_a_confidence_no_interval_would_accept() -> None:
    """Verify critical_value raises ValueError for confidence outside (0.0, 1.0)."""
    for confidence in (0.0, 1.0, -0.5, 1.5):
        with pytest.raises(ValueError, match="strictly between"):
            critical_value(confidence)


def test_nothing_probed_is_absent_rather_than_the_whole_unit() -> None:
    """Verify 0 probes returns None for wilson_interval and detectable_delta."""
    assert wilson_interval(0, 0) is None
    assert detectable_delta(0) is None


@pytest.mark.parametrize(
    ("hits", "probes", "confidence", "message"),
    [
        (3, 2, DEFAULT_CONFIDENCE, "not a possible count"),
        (-1, 5, DEFAULT_CONFIDENCE, "not a possible count"),
        (0, -1, DEFAULT_CONFIDENCE, "probes cannot be negative"),
        (1, 5, 0.0, "strictly between"),
        (1, 5, 1.0, "strictly between"),
        (1, 5, 1.5, "strictly between"),
    ],
)
def test_an_impossible_request_is_refused(
    hits: int,
    probes: int,
    confidence: float,
    message: str,
) -> None:
    """Verify invalid hits, negative probes, and out-of-range confidence raise ValueError."""
    with pytest.raises(ValueError, match=message):
        wilson_interval(hits, probes, confidence=confidence)


def test_inverted_bounds_are_refused() -> None:
    """Verify Interval raises ValueError when low bound exceeds high bound."""
    with pytest.raises(ValueError, match="inverted"):
        Interval(low=0.6, high=0.4)


def test_excludes_answers_the_question_a_finding_rests_on() -> None:
    """Verify excludes returns True for values outside interval and False for values inside."""
    interval = wilson_interval(0, 5)
    assert interval is not None
    assert not interval.excludes(0.25)
    assert interval.excludes(0.75)


def test_overlap_is_symmetric_and_counts_a_single_shared_point() -> None:
    """Verify Interval.overlaps is symmetric and detects boundary-touching intervals."""
    low = Interval(low=0.0, high=0.4)
    high = Interval(low=0.4, high=1.0)
    clear = Interval(low=0.5, high=0.9)
    assert low.overlaps(high)
    assert high.overlaps(low)
    assert not low.overlaps(clear)
    assert not clear.overlaps(low)


def test_two_runs_can_overlap_while_neither_contains_the_other_rate() -> None:
    """Verify intervals can overlap even when point estimates fall outside each other's interval."""
    shallow, deep = wilson_interval(2, 5), wilson_interval(0, 20)
    assert shallow is not None
    assert deep is not None
    assert shallow.excludes(0 / 20)
    assert deep.excludes(2 / 5)
    assert shallow.overlaps(deep)


def test_interval_contains_and_membership() -> None:
    """Verify Interval implements the __contains__ container protocol and contains method."""
    interval = Interval(low=0.2, high=0.6)
    assert 0.2 in interval
    assert 0.4 in interval
    assert 0.6 in interval
    assert 0.19 not in interval
    assert 0.61 not in interval
    assert -0.5 not in interval
    assert 1.5 not in interval
    assert "invalid" not in interval  # Non-numeric type returns False

    assert interval.contains(0.4)
    assert not interval.contains(0.7)

    # Invariant: contains and excludes are exact logical complements
    for rate in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        assert interval.contains(rate) == (not interval.excludes(rate))
        assert (rate in interval) == (not interval.excludes(rate))


def test_interval_as_tuple_and_dict_semantics() -> None:
    """Verify Interval supports tuple conversion and preserves BaseModel dict semantics."""
    interval = Interval(low=0.25, high=0.75)

    assert interval.as_tuple() == (0.25, 0.75)
    assert dict(interval) == {"low": 0.25, "high": 0.75, "confidence": DEFAULT_CONFIDENCE}


def test_interval_center() -> None:
    """Verify center property returns the midpoint of the confidence interval."""
    assert Interval(low=0.2, high=0.8).center == pytest.approx(0.5)
    assert Interval(low=0.0, high=0.0).center == 0.0
    assert Interval(low=1.0, high=1.0).center == 1.0
    assert Interval(low=0.1, high=0.4).center == pytest.approx(0.25)


def test_interval_intersection() -> None:
    """Verify intersection calculates overlapping sub-intervals and handles disjoint cases."""
    a = Interval(low=0.1, high=0.5, confidence=0.95)
    b = Interval(low=0.3, high=0.7, confidence=0.90)

    overlap = a.intersection(b)
    assert overlap is not None
    assert overlap.low == pytest.approx(0.3)
    assert overlap.high == pytest.approx(0.5)
    assert overlap.confidence == pytest.approx(0.90)  # Conservative lower confidence

    # Disjoint intervals
    c = Interval(low=0.8, high=0.9)
    assert a.intersection(c) is None
    assert c.intersection(a) is None

    # Boundary touch (single shared point)
    touching = Interval(low=0.5, high=0.9)
    touch_overlap = a.intersection(touching)
    assert touch_overlap is not None
    assert touch_overlap.low == pytest.approx(0.5)
    assert touch_overlap.high == pytest.approx(0.5)


def test_interval_format_percent() -> None:
    """Verify format_percent produces standard bracketed percentage ranges."""
    interval = Interval(low=0.1234, high=0.5678)
    assert interval.format_percent() == "[12.3% - 56.8%]"
    assert interval.format_percent(digits=0) == "[12% - 57%]"
    assert interval.format_percent(digits=2) == "[12.34% - 56.78%]"


def test_interval_alternative_constructors() -> None:
    """Verify Interval classmethod constructor from tuple."""
    iv_tup = Interval.from_tuple((0.2, 0.8))
    assert iv_tup.low == 0.2
    assert iv_tup.high == 0.8
    assert iv_tup.confidence == DEFAULT_CONFIDENCE

    with pytest.raises(ValueError, match="expected 2 elements"):
        Interval.from_tuple((0.2,))  # type: ignore[arg-type]


def test_interval_strictness() -> None:
    """Verify extra attributes are forbidden during Interval model validation."""
    with pytest.raises(ValidationError, match="extra_forbidden"):
        Interval.model_validate({"low": 0.2, "high": 0.8, "unexpected_field": "bogus"})


@pytest.mark.parametrize(("finding", "counts"), sorted(DEPTH_PAIRS.items()))
def test_no_published_ratio_was_contradicted_by_the_depth_that_replaced_it(
    finding: str,
    counts: tuple[tuple[int, int], tuple[int, int]],
) -> None:
    """Verify shallow and deep interval confidence bounds overlap across sample findings."""
    (shallow_hits, shallow_probes), (deep_hits, deep_probes) = counts
    shallow = wilson_interval(shallow_hits, shallow_probes)
    deep = wilson_interval(deep_hits, deep_probes)
    assert shallow is not None
    assert deep is not None
    assert shallow.overlaps(deep), finding


@pytest.mark.parametrize(("finding", "counts"), sorted(DEPTH_PAIRS.items()))
def test_containment_is_the_stricter_test_the_study_first_reported(
    finding: str,
    counts: tuple[tuple[int, int], tuple[int, int]],
) -> None:
    """Verify deep point estimates fall outside shallow intervals only for flagged findings."""
    (shallow_hits, shallow_probes), (deep_hits, deep_probes) = counts
    shallow = wilson_interval(shallow_hits, shallow_probes)
    assert shallow is not None
    outside = shallow.excludes(deep_hits / deep_probes)
    assert outside == (finding in DEEP_RATE_OUTSIDE_PUBLISHED_INTERVAL), finding


def test_five_attempts_on_one_query_resolve_almost_nothing() -> None:
    """Verify width and detectable delta values for small sample size (n=5)."""
    interval = wilson_interval(0, 5)
    assert interval is not None
    assert interval.width == pytest.approx(0.4345, abs=5e-5)
    assert detectable_delta(5) == pytest.approx(0.886, abs=5e-4)


@pytest.mark.parametrize(
    ("delta", "probes"),
    [
        (0.5, 16),
        (0.2, 99),
        (0.1, 393),
        (0.05, 1570),
        (1.0, 4),
    ],
)
def test_the_power_arithmetic_is_the_one_the_documents_quote(
    delta: float,
    probes: int,
) -> None:
    """Verify required_probes calculates sample size needed to detect delta."""
    assert required_probes(delta) == probes


def test_the_two_directions_agree() -> None:
    """Verify detectable_delta of required_probes recovers target delta."""
    for delta in (0.5, 0.3, 0.2, 0.1, 0.05, 0.02):
        affordable = detectable_delta(required_probes(delta))
        assert affordable is not None
        assert affordable <= delta + 1e-12


def test_the_detectable_delta_never_exceeds_the_whole_unit() -> None:
    """Verify detectable_delta is capped at 1.0."""
    assert detectable_delta(1) == 1.0
    assert detectable_delta(3) == 1.0


@pytest.mark.parametrize(
    ("delta", "confidence", "power", "message"),
    [
        (0.0, 0.95, 0.8, "delta must lie in"),
        (-0.1, 0.95, 0.8, "delta must lie in"),
        (1.5, 0.95, 0.8, "delta must lie in"),
        (0.1, 1.0, 0.8, "confidence must lie strictly between"),
        (0.1, 0.95, 1.0, "power must lie strictly between"),
    ],
)
def test_an_unanswerable_power_question_is_refused(
    delta: float,
    confidence: float,
    power: float,
    message: str,
) -> None:
    """Verify required_probes raises ValueError for invalid delta, confidence, or power."""
    with pytest.raises(ValueError, match=message):
        required_probes(delta, confidence=confidence, power=power)


@pytest.mark.parametrize(
    ("probes", "power", "message"),
    [
        (-1, DEFAULT_POWER, "cannot be negative"),
        (30, 0.0, "strictly between"),
        (30, 1.0, "strictly between"),
    ],
    ids=["negative-probes", "no-power", "certain-power"],
)
def test_asking_what_an_impossible_run_could_resolve_is_refused(
    probes: int,
    power: float,
    message: str,
) -> None:
    """Verify detectable_delta raises ValueError for negative probes or invalid power."""
    with pytest.raises(ValueError, match=message):
        detectable_delta(probes, power=power)


def test_the_classification_report_bounds_each_of_its_counted_rates(
    whole_catalog_results: list[ProbeResult],
    whole_catalog_queries: QuerySet,
) -> None:
    """Verify headline rates in ClassificationReport contain valid confidence intervals."""
    report = classification_report(whole_catalog_results, whole_catalog_queries.queries)
    for rate, interval in (
        (report.top1_accuracy, report.top1_interval),
        (report.entrypoint_accuracy, report.entrypoint_interval),
        (report.trajectory_reachability, report.trajectory_interval),
        (report.abstention_rate, report.abstention_interval),
    ):
        assert interval is not None
        assert not interval.excludes(rate)


def test_the_classification_report_bounds_every_rate_with_a_denominator(
    whole_catalog_results: list[ProbeResult],
    whole_catalog_queries: QuerySet,
) -> None:
    """Verify top-1 and abstention rates in ClassificationReport contain valid intervals."""
    report = classification_report(whole_catalog_results, whole_catalog_queries.queries)
    assert report.top1_interval is not None
    assert not report.top1_interval.excludes(report.top1_accuracy)
    assert report.abstention_interval is not None
    assert not report.abstention_interval.excludes(report.abstention_rate)
    assert report.false_abstention_interval is not None
    assert not report.false_abstention_interval.excludes(report.false_abstention_rate)
    assert report.top1_hits + report.scored - report.top1_hits == report.scored
    assert report.in_scope <= report.scored
    assert report.abstentions >= report.false_abstentions


def test_out_of_scope_detection_stays_absent_until_a_query_has_no_answer(
    whole_catalog_results: list[ProbeResult],
    whole_catalog_queries: QuerySet,
) -> None:
    """Verify out-of-scope detection rate and interval are None when out_of_scope count is 0."""
    report = classification_report(whole_catalog_results, whole_catalog_queries.queries)
    assert (report.out_of_scope == 0) == (report.out_of_scope_detection is None)
    assert (report.out_of_scope == 0) == (report.out_of_scope_interval is None)


def test_a_labels_recall_interval_is_taken_over_that_labels_probes_alone(
    whole_catalog_results: list[ProbeResult],
    whole_catalog_queries: QuerySet,
) -> None:
    """Verify per-class recall interval is computed using class support count as denominator."""
    report = classification_report(whole_catalog_results, whole_catalog_queries.queries)
    for entry in report.per_class:
        support = entry.true_positives + entry.false_negatives
        assert (support == 0) == (entry.recall_interval is None)
        if entry.recall_interval is not None:
            assert support == entry.support
            assert not entry.recall_interval.excludes(entry.recall)
            assert entry.recall_interval.width > 0.0


def test_every_skill_in_the_artifact_carries_a_bound_on_its_recall(
    artifact: Artifact,
) -> None:
    """Verify each skill entry in Artifact carries valid recall and precision intervals."""
    assert artifact.skills
    for skill in artifact.skills:
        assert (skill.recall is None) == (skill.recall_interval is None)
        assert (skill.precision is None) == (skill.precision_interval is None)
        if skill.recall is not None and skill.recall_interval is not None:
            assert not skill.recall_interval.excludes(skill.recall)


def test_f1_is_the_one_per_skill_figure_left_unbounded(
    artifact: Artifact,
) -> None:
    """Verify f1 metric has no binomial interval attribute."""
    assert artifact.skills
    for skill in artifact.skills:
        assert (skill.f1 is None) == (skill.recall is None)
        assert not hasattr(skill, "f1_interval")


def test_every_probed_query_carries_a_bound_on_its_hit_rate(
    artifact: Artifact,
) -> None:
    """Verify all probed query records in Artifact carry valid hit rate intervals."""
    assert artifact.queries
    for record in artifact.queries:
        assert (record.probes == 0) == (record.interval is None)
        if record.interval is None:
            continue
        assert not record.interval.excludes(record.hits / record.probes)
        if record.clean:
            assert record.interval.low < 1.0


def test_the_run_scores_bound_both_headlines_over_their_own_denominators(
    artifact: Artifact,
) -> None:
    """Verify RunScores headline metrics contain intervals over their respective denominators."""
    scores = artifact.scores
    assert scores.observed_queries == len([q for q in artifact.queries if q.probes])
    assert scores.scored == artifact.probes - artifact.errors
    assert scores.consistency_interval is not None
    assert not scores.consistency_interval.excludes(scores.consistency)
    assert scores.top1_interval is not None
    assert not scores.top1_interval.excludes(scores.top1_accuracy)


def test_the_abstention_block_bounds_each_of_its_three_rates(
    artifact: Artifact,
) -> None:
    """Verify Abstention object in RunScores includes intervals for rate and false_rate."""
    abstention = artifact.scores.abstention
    assert abstention.interval is not None
    assert not abstention.interval.excludes(abstention.rate)
    assert abstention.false_interval is not None
    assert not abstention.false_interval.excludes(abstention.false_rate)
    assert (abstention.out_of_scope == 0) == (abstention.out_of_scope_interval is None)


def test_the_intervals_survive_serialization(artifact: Artifact) -> None:
    """Verify confidence interval fields are preserved across JSON serialization."""
    payload = json.loads(artifact.model_dump_json())
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["scores"]["top1_interval"]["confidence"] == 0.95
    assert payload["scores"]["consistency_interval"] is not None
    assert payload["scores"]["abstention"]["interval"] is not None
    assert all("interval" in q for q in payload["queries"])
    assert all("recall_interval" in s for s in payload["skills"])


def test_a_document_that_states_a_rate_without_its_denominator_is_refused() -> None:
    """Verify RunScores validation fails when denominator count fields are omitted."""
    with pytest.raises(ValidationError, match="scored"):
        RunScores.model_validate(
            {
                "consistency": 0.5,
                "top1_accuracy": 0.75,
                "abstention": {
                    "rate": 0.1,
                    "false_rate": 0.1,
                    "scored": 10,
                    "abstentions": 1,
                    "in_scope": 10,
                    "false_abstentions": 1,
                },
                "not_headline": {"macro_f1": 0.5},
                "unanimous_queries": 1,
                "observed_queries": 2,
                "top1_hits": 3,
            },
        )


def test_a_rate_measured_over_nothing_is_unbounded_rather_than_wide() -> None:
    """Verify RunScores intervals evaluate to None when denominator counts are 0."""
    scores = RunScores(
        consistency=0.0,
        top1_accuracy=0.0,
        abstention=Abstention(
            rate=0.0,
            false_rate=0.0,
            scored=0,
            abstentions=0,
            in_scope=0,
            false_abstentions=0,
        ),
        not_headline=NotHeadline(macro_f1=0.0),
        unanimous_queries=0,
        observed_queries=0,
        top1_hits=0,
        scored=0,
    )
    assert scores.consistency_interval is None
    assert scores.top1_interval is None
    assert scores.abstention.interval is None
    assert scores.abstention.out_of_scope_interval is None


def test_a_rate_that_disagrees_with_its_counts_is_refused() -> None:
    """Verify RunScores validation fails when stored rate disagrees with ratio of counts."""
    with pytest.raises(ValueError, match=r"top1_accuracy is 0\.9 but its counts"):
        RunScores(
            consistency=0.5,
            top1_accuracy=0.9,
            top1_hits=3,
            scored=4,
            abstention=Abstention(
                rate=0.0,
                false_rate=0.0,
                scored=4,
                abstentions=0,
                in_scope=4,
                false_abstentions=0,
            ),
            not_headline=NotHeadline(macro_f1=0.5),
            unanimous_queries=1,
            observed_queries=2,
        )


def test_the_terminal_view_prints_an_interval_beside_every_bounded_rate(
    artifact: Artifact,
) -> None:
    """Verify render_text output prints bracketed intervals beside all bounded rates."""
    lines = {
        line.strip().split("  ")[0]: line
        for line in render_text(artifact).splitlines()
        if line.startswith("  ")
    }
    for label in (
        "top-1 accuracy",
        "abstention",
        "false abstention",
        "consistency",
    ):
        assert "[" in lines[label], f"{label} was printed without its interval"


def test_the_terminal_view_prints_no_interval_it_cannot_justify(
    artifact: Artifact,
) -> None:
    """Verify unbounded metrics like macro averages omit interval brackets."""
    for line in render_text(artifact).splitlines():
        head = line.strip()
        if head.startswith(("macro", "inherent")):
            assert "[" not in line, line


def test_the_per_query_table_and_the_csv_both_carry_the_bounds(
    artifact: Artifact,
) -> None:
    """Verify query results in text and CSV reports include confidence interval bounds."""
    rendered = render_text(artifact)
    for query in artifact.queries:
        assert query.interval is not None
        assert f"{query.hits}/{query.probes}" in rendered

    header, *rows = render_csv(artifact).strip().splitlines()
    assert header.split(",")[6:8] == ["ci_low", "ci_high"]
    for row in rows:
        low, high = (float(v) for v in row.split(",")[6:8])
        assert 0.0 <= low <= high <= 1.0


def test_an_unprobed_run_reports_no_bound_rather_than_a_full_one(
    whole_catalog: Catalog,
    corpus: list[Skill],
    make_config,
) -> None:
    """Verify empty report produces no query outcomes and None for consistency interval."""
    empty_queries = QuerySet(
        catalog_id="all",
        queries=(),
        provenance=QuerySetProvenance(origin=Origin.AUTHORED),
    )
    composition = Composition(
        config=make_config(catalog={"mode": CatalogMode.ALL}),
        query_set=empty_queries,
        catalog=whole_catalog,
        skills=tuple(corpus),
    )
    report = build_report(composition, [])
    assert report.queries == ()
    assert report.scores.consistency_interval is None
    assert report.scores.observed_queries == 0


def test_a_query_probed_once_is_bounded_rather_than_certain(
    whole_catalog_queries: QuerySet,
    whole_catalog_results: list[ProbeResult],
    whole_catalog: Catalog,
    corpus: list[Skill],
    make_config,
) -> None:
    """Verify single-probe query outcomes produce wide confidence intervals."""
    composition = Composition(
        config=make_config(catalog={"mode": CatalogMode.ALL}),
        query_set=whole_catalog_queries,
        catalog=whole_catalog,
        skills=tuple(corpus),
    )
    single = [r for r in whole_catalog_results if r.attempt == 1]
    report = build_report(composition, single)
    for query in report.queries:
        assert query.probes == 1
        assert query.interval is not None
        assert query.interval.width > 0.7


def test_effective_sample_size_calculation() -> None:
    """Verify effective_sample_size adjusts sample size based on repeated attempts and ICC."""
    assert effective_sample_size(100, attempts=1) == 100
    assert effective_sample_size(100, attempts=5, intra_cluster_correlation=0.0) == 100
    assert effective_sample_size(100, attempts=5, intra_cluster_correlation=1.0) == 20
    # DEFF = 1 + (5 - 1) * 0.6 = 3.4; 100 / 3.4 = 29.41 -> 29
    assert effective_sample_size(100, attempts=5, intra_cluster_correlation=0.6) == 29
    assert effective_sample_size(0, attempts=5) == 0
    assert effective_sample_size(-10, attempts=5) == 0


def test_cluster_wilson_interval_widens_with_attempts() -> None:
    """Verify cluster_wilson_interval is wider than standard wilson_interval when attempts > 1."""
    std_int = wilson_interval(80, 100)
    assert std_int is not None

    single_int = cluster_wilson_interval(80, 100, attempts=1)
    assert single_int is not None
    assert single_int.low == std_int.low
    assert single_int.high == std_int.high

    clustered_int = cluster_wilson_interval(80, 100, attempts=5, intra_cluster_correlation=0.6)
    assert clustered_int is not None
    assert clustered_int.width > std_int.width
    assert clustered_int.low < std_int.low
    assert clustered_int.high > std_int.high
    assert cluster_wilson_interval(0, 0) is None
