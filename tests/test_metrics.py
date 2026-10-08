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

"""Verify classification, routing, and trajectory scoring over probe results."""

from __future__ import annotations

import pytest
from scipy import stats

from reach.metrics import (
    ClassMetrics,
    classification_report,
    collisions,
    compute_f1,
    confusion,
    consistency,
    score_trajectory,
)
from reach.models import (
    NO_SKILL,
    CatalogMode,
    ProbeResult,
    Query,
    QueryKind,
)


def _result(query_id: str, invoked: str | None, error: str | None = None) -> ProbeResult:
    """Build a probe result with fixed catalog metadata for scoring tests."""
    return ProbeResult(
        query_id=query_id,
        catalog_id="test-cat",
        catalog_mode=CatalogMode.ALL,
        catalog_size=2,
        model="opus",
        runtime="fake",
        invoked_skills=(invoked,) if invoked else (),
        error=error,
    )


def test_compute_f1_is_the_harmonic_mean_of_precision_and_recall() -> None:
    """Verify compute_f1 calculates harmonic mean of precision and recall."""
    assert compute_f1(0.5, 0.5) == pytest.approx(0.5)
    assert compute_f1(1.0, 1.0) == pytest.approx(1.0)
    assert compute_f1(1.0, 0.0) == 0.0


def test_compute_f1_is_zero_rather_than_a_division_by_zero() -> None:
    """Verify compute_f1 returns 0.0 when precision and recall are both zero."""
    assert compute_f1(0.0, 0.0) == 0.0


def test_exact_match_counts_top1(queries) -> None:
    """Verify exact match increments top1 accuracy."""
    report = classification_report([_result("q-lifecycle", "gcs-lifecycle-rules")], queries)
    assert report.top1_accuracy == 1.0


def test_misroute_counts_as_zero_accuracy(queries) -> None:
    """Verify misrouted invocation earns zero accuracy."""
    report = classification_report([_result("q-lifecycle", "gke-basics")], queries)
    assert report.top1_accuracy == 0.0


def test_non_selection_is_tracked_separately(queries) -> None:
    """Verify non-selection is tracked in non_selections count rather than errors."""
    report = classification_report([_result("q-lifecycle", None)], queries)
    assert report.abstentions == 1
    assert report.abstention_rate == 1.0
    assert report.top1_accuracy == 0.0


def test_errored_probes_leave_the_denominator(queries) -> None:
    """Verify errored probes are excluded from scored denominator."""
    results = [
        _result("q-lifecycle", "gcs-lifecycle-rules"),
        _result("q-retention", None, error="timeout"),
    ]
    report = classification_report(results, queries)
    assert report.probes == 2
    assert report.errors == 1
    assert report.scored == 1
    assert report.top1_accuracy == 1.0


def test_all_errors_do_not_divide_by_zero(queries) -> None:
    """Verify score returns 0.0 accuracy metrics when all probes encounter errors."""
    report = classification_report([_result("q-lifecycle", None, error="timeout")], queries)
    assert report.scored == 0
    assert report.top1_accuracy == 0.0


def test_unlabeled_result_raises(queries) -> None:
    """Verify score raises KeyError when encountering unknown query_id."""
    with pytest.raises(KeyError, match="q-unknown"):
        classification_report([_result("q-unknown", "gke-basics")], queries)


def test_confusion_records_non_selection_as_none(queries) -> None:
    """Verify confusion matrix tallies uninvoked queries under None key."""
    matrix = confusion([_result("q-lifecycle", None)], queries)
    assert matrix[("gcs-lifecycle-rules", None)] == 1


def test_collisions_count_all_misroutes(queries) -> None:
    """Verify collisions counts all misroutes between distinct skills."""
    results = [
        _result("q-retention", "gcs-lifecycle-rules"),
        _result("q-lifecycle", "gke-basics"),
    ]
    pairs = collisions(results, queries)
    assert pairs == {
        ("gcs-retention-policy", "gcs-lifecycle-rules"): 1,
        ("gcs-lifecycle-rules", "gke-basics"): 1,
    }


def test_precision_is_bounded_only_when_the_skill_was_ever_selected(queries) -> None:
    """Verify precision interval is calculated only for skills selected at least once."""
    report = classification_report(
        [
            _result("q-lifecycle", "gcs-lifecycle-rules"),
            _result("q-retention", "gcs-lifecycle-rules"),
        ],
        queries,
    )
    selected = report.by_label("gcs-lifecycle-rules").precision_interval
    assert selected is not None
    assert selected.low <= 0.5 <= selected.high
    assert report.by_label("gcs-retention-policy").precision_interval is None


def test_asking_for_a_label_the_run_never_saw_names_the_label(queries) -> None:
    """Verify by_label raises KeyError for labels not present in the run."""
    report = classification_report([_result("q-lifecycle", None)], queries)
    with pytest.raises(KeyError, match="gke-basics"):
        report.by_label("gke-basics")


@pytest.mark.parametrize(
    ("tally", "expected"),
    [(confusion, {}), (collisions, {})],
    ids=["confusion", "collisions"],
)
def test_a_tally_over_a_narrowed_query_set_skips_what_it_cannot_label(
    queries,
    tally,
    expected,
) -> None:
    """Verify confusion and collisions filter out unindexed queries without error."""
    subset = [q for q in queries if q.query_id == "q-lifecycle"]
    results = [_result("q-retention", "gke-basics")]
    assert tally(results, subset) == expected
    with pytest.raises(KeyError, match="q-retention"):
        classification_report(results, subset)


#: label -> (support, predicted, tp, fp, fn) expected counts for worked example.
WORKED_COUNTS = {
    NO_SKILL: (2, 2, 1, 1, 1),
    "waf-cost": (2, 3, 2, 1, 0),
    "waf-reliability": (2, 1, 1, 0, 1),
    "waf-security": (2, 4, 1, 3, 1),
    "waf-sustainability": (2, 0, 0, 0, 2),
}

#: metric -> (expected value, description).
WORKED_HEADLINE = {
    "top1_accuracy": (0.5, "5 of 10 predictions match"),
    "macro_precision": ((2 / 3 + 0.25 + 1.0 + 0.0 + 0.5) / 5, "mean of 5 precisions"),
    "macro_recall": ((1.0 + 0.5 + 0.5 + 0.0 + 0.5) / 5, "mean of 5 recalls"),
    "macro_f1": ((0.8 + 1 / 3 + 2 / 3 + 0.0 + 0.5) / 5, "mean of 5 F1s"),
    "abstention_rate": (0.2, "2 of 10 predictions abstained"),
    "false_abstention_rate": (1 / 8, "1 of the 8 in-scope probes abstained"),
    "out_of_scope_detection": (0.5, "1 of the 2 out-of-scope probes abstained"),
}


@pytest.mark.parametrize(
    ("metric", "expected", "reason"),
    [(k, v[0], v[1]) for k, v in WORKED_HEADLINE.items()],
)
def test_headline_metric_matches_hand_arithmetic(
    worked_results,
    worked_queries,
    metric,
    expected,
    reason,
) -> None:
    """Verify headline metrics match hand-calculated expectations on worked dataset."""
    report = classification_report(worked_results, worked_queries)
    assert getattr(report, metric) == pytest.approx(expected), reason


@pytest.mark.parametrize(("label", "counts"), sorted(WORKED_COUNTS.items()))
def test_per_class_counts_match_hand_arithmetic(
    worked_results,
    worked_queries,
    label,
    counts,
) -> None:
    """Verify support and confusion counts for each label match hand calculations."""
    entry = classification_report(worked_results, worked_queries).by_label(label)
    support, predicted, tp, fp, fn = counts
    actual = (
        entry.support,
        entry.predicted,
        entry.true_positives,
        entry.false_positives,
        entry.false_negatives,
    )
    assert actual == (support, predicted, tp, fp, fn)


def test_per_class_rates_derive_from_the_counts(worked_results, worked_queries) -> None:
    """Verify per-class precision, recall, and F1 correctly compute from raw counts."""
    report = classification_report(worked_results, worked_queries)
    security = report.by_label("waf-security")
    assert security.precision == pytest.approx(0.25)
    assert security.recall == pytest.approx(0.5)
    assert security.f1 == pytest.approx(1 / 3)

    unreachable = report.by_label("waf-sustainability")
    assert (unreachable.precision, unreachable.recall, unreachable.f1) == (
        0.0,
        0.0,
        0.0,
    )


def test_abstention_is_scored_as_a_class_not_a_missing_prediction(
    worked_results,
    worked_queries,
) -> None:
    """Verify NO_SKILL abstention is treated as a first-class evaluated label."""
    report = classification_report(worked_results, worked_queries)
    abstain = report.by_label(NO_SKILL)
    assert (abstain.true_positives, abstain.false_positives) == (1, 1)
    assert len(report.per_class) == 5


def test_classification_report_tracks_top1_and_abstention(
    worked_results,
    worked_queries,
) -> None:
    """Verify classification report distinguishes top-1 hits and abstentions."""
    report = classification_report(worked_results, worked_queries)
    assert (report.probes, report.scored, report.top1_hits) == (10, 10, 5)
    assert report.abstentions == 2
    assert report.false_abstentions == 1
    assert report.top1_accuracy == pytest.approx(0.5)


def test_consistency_counts_queries_whose_attempts_agreed(
    worked_results,
    worked_queries,
) -> None:
    """Verify consistency calculates fraction of queries with identical selections."""
    assert consistency(worked_results, worked_queries) == pytest.approx(0.4)


def test_collisions_count_a_fire_on_an_out_of_scope_query(
    worked_results,
    worked_queries,
) -> None:
    """Verify collisions records unauthorized skill invocations on out-of-scope queries."""
    pairs = collisions(worked_results, worked_queries)
    assert pairs[(NO_SKILL, "waf-security")] == 1
    assert pairs[("waf-sustainability", "waf-security")] == 2
    assert pairs[("waf-security", "waf-cost")] == 1


def test_worst_recall_and_top_attractors_rank_the_actionable_labels(
    worked_results,
    worked_queries,
) -> None:
    """Verify worst_recall and top_attractors sort and rank problem labels."""
    report = classification_report(worked_results, worked_queries)
    assert report.worst_recall(1)[0].label == "waf-sustainability"
    assert report.top_attractors(1)[0].label == "waf-security"


def test_worked_example_agrees_with_sklearn(
    worked_results,
    worked_queries,
    matches_sklearn,
) -> None:
    """Verify reach metrics match scikit-learn reference implementation."""
    matches_sklearn(worked_results, worked_queries)


def test_errored_probes_are_excluded_from_every_metric(
    worked_results,
    worked_queries,
    make_result,
) -> None:
    """Verify errored probes are excluded from classification report calculations."""
    broken = [
        *worked_results,
        make_result("wq-cost", None, attempt=3, error="tool leak"),
    ]
    clean = classification_report(worked_results, worked_queries)
    with_error = classification_report(broken, worked_queries)
    assert with_error.errors == 1
    assert with_error.scored == clean.scored == 10
    assert with_error.top1_accuracy == pytest.approx(clean.top1_accuracy)
    assert with_error.abstention_rate == pytest.approx(clean.abstention_rate)


def test_results_referencing_an_unlabeled_query_are_rejected(
    worked_results,
    worked_queries,
    make_result,
) -> None:
    """Verify classification_report raises KeyError for unindexed query IDs."""
    stray = [*worked_results, make_result("wq-ghost", "waf-cost")]
    with pytest.raises(KeyError, match="wq-ghost"):
        classification_report(stray, worked_queries)


def test_empty_input_scores_zero_rather_than_dividing_by_zero(worked_queries) -> None:
    """Verify metrics return 0.0 or None without ZeroDivisionError on empty datasets."""
    report = classification_report([], worked_queries)
    assert (report.scored, report.top1_accuracy, report.macro_f1) == (0, 0.0, 0.0)
    assert report.out_of_scope_detection is None
    assert consistency([], worked_queries) == 0.0


def test_out_of_scope_detection_reaches_both_extremes(worked_queries, make_result) -> None:
    """Verify out_of_scope_detection correctly evaluates 1.0 and 0.0 boundary conditions."""
    always = [make_result("wq-oos", None, attempt=i) for i in (1, 2)]
    never = [make_result("wq-oos", "waf-cost", attempt=i) for i in (1, 2)]
    oos = [q for q in worked_queries if q.is_out_of_scope]

    assert classification_report(always, oos).out_of_scope_detection == 1.0
    assert classification_report(never, oos).out_of_scope_detection == 0.0
    assert classification_report(always, oos).false_abstention_rate == 0.0


def test_both_label_surfaces_use_the_same_abstain_token(worked_queries, make_result) -> None:
    """Verify query and result models use identical NO_SKILL constant for abstentions."""
    assert make_result("wq-oos", None).predicted_label == NO_SKILL
    assert next(q for q in worked_queries if q.is_out_of_scope).truth_label == NO_SKILL
    assert make_result("wq-cost", "waf-cost").predicted_label == "waf-cost"


@pytest.mark.parametrize(
    (
        "query",
        "invoked",
        "expected_entry",
        "expected_reach",
        "expected_mrr",
        "expected_f1",
        "expected_redundancy",
    ),
    [
        # Direct Primary Hit: [{deploy}], invoked: [deploy]
        (
            Query(query_id="q1", text="deploy", expected_skill="deploy"),
            ("deploy",),
            True,
            True,
            1.0,
            1.0,
            0,
        ),
        # Precursor Setup: [{deploy}], invoked: [gcloud, deploy]
        (
            Query(query_id="q3", text="deploy", expected_skill="deploy"),
            ("gcloud", "deploy"),
            False,
            True,
            0.5,
            0.6667,
            1,
        ),
        # Skill Stuffing (Spam): [{deploy}], invoked: [deploy, s2, s3, s4, s5]
        (
            Query(query_id="q5", text="deploy", expected_skill="deploy"),
            ("deploy", "s2", "s3", "s4", "s5"),
            True,
            True,
            1.0,
            0.3333,
            4,
        ),
        # Looping Retry Bloat: [{deploy}], invoked: [deploy, deploy, deploy]
        (
            Query(query_id="q6", text="deploy", expected_skill="deploy"),
            ("deploy", "deploy", "deploy"),
            True,
            True,
            1.0,
            1.0,
            2,
        ),
        # Out-of-Scope (Abstain): [], invoked: []
        (
            Query(query_id="q7", text="hello", kind=QueryKind.OUT_OF_SCOPE),
            (),
            True,
            True,
            None,
            None,
            0,
        ),
        # Out-of-Scope (Misroute): [], invoked: [deploy]
        (
            Query(query_id="q8", text="hello", kind=QueryKind.OUT_OF_SCOPE),
            ("deploy",),
            False,
            False,
            None,
            None,
            1,
        ),
        # Fatal Misroute: [{deploy}], invoked: [wrong]
        (
            Query(query_id="q9", text="deploy", expected_skill="deploy"),
            ("wrong",),
            False,
            False,
            0.0,
            0.0,
            0,
        ),
    ],
)
def test_score_trajectory_behavior_matrix(
    query: Query,
    invoked: tuple[str, ...],
    expected_entry: bool,
    expected_reach: bool,
    expected_mrr: float | None,
    expected_f1: float | None,
    expected_redundancy: int,
) -> None:
    """Verify score_trajectory conforms to the comprehensive behavior matrix."""
    score = score_trajectory(query, invoked)
    assert score.entrypoint_hit is expected_entry
    assert score.trajectory_hit is expected_reach
    if expected_mrr is None:
        assert score.step_efficiency is None
    else:
        assert score.step_efficiency == pytest.approx(expected_mrr, abs=1e-3)
    if expected_f1 is None:
        assert score.skill_f1 is None
    else:
        assert score.skill_f1 == pytest.approx(expected_f1, abs=1e-3)
    assert score.redundancy == expected_redundancy


def test_classification_report_trajectory_aggregates_and_scipy_cross_check() -> None:
    """Verify aggregated trajectory metrics and validate Wilson intervals against scipy."""
    queries = (
        Query(query_id="q1", text="deploy", expected_skill="deploy"),
        Query(query_id="q2", text="scale", expected_skill="scale"),
        Query(query_id="q3", text="auth", expected_skill="auth"),
        Query(query_id="q4", text="oos", kind=QueryKind.OUT_OF_SCOPE),
    )
    # q1: precursor setup (entrypoint False, trajectory True, mrr 0.5, f1 0.6667, red 1)
    # q2: direct primary hit (entrypoint True, trajectory True, mrr 1.0, f1 1.0, red 0)
    # q3: fatal misroute (entrypoint False, trajectory False, mrr 0.0, f1 0.0, red 0)
    # q4: clean abstention (entrypoint True, trajectory True, mrr None, f1 None, red 0)
    results = [
        ProbeResult(
            query_id="q1",
            catalog_id="cat",
            catalog_mode=CatalogMode.ALL,
            catalog_size=3,
            model="m",
            runtime="fake",
            invoked_skills=("auth", "deploy"),
        ),
        ProbeResult(
            query_id="q2",
            catalog_id="cat",
            catalog_mode=CatalogMode.ALL,
            catalog_size=3,
            model="m",
            runtime="fake",
            invoked_skills=("scale",),
        ),
        ProbeResult(
            query_id="q3",
            catalog_id="cat",
            catalog_mode=CatalogMode.ALL,
            catalog_size=3,
            model="m",
            runtime="fake",
            invoked_skills=("wrong",),
        ),
        ProbeResult(
            query_id="q4",
            catalog_id="cat",
            catalog_mode=CatalogMode.ALL,
            catalog_size=3,
            model="m",
            runtime="fake",
            invoked_skills=(),
        ),
    ]

    report = classification_report(results, queries)
    assert report.scored == 4
    assert report.entrypoint_hits == 2  # q2 and q4
    assert report.entrypoint_accuracy == pytest.approx(2 / 4)
    assert report.trajectory_hits == 3  # q1, q2, q4
    assert report.trajectory_reachability == pytest.approx(3 / 4)
    # q1, q2, q3 are in-scope; q4 is out-of-scope and excluded from step_efficiency and skill_f1
    assert report.step_efficiency == pytest.approx((0.5 + 1.0 + 0.0) / 3)
    assert report.skill_f1 == pytest.approx((0.6667 + 1.0 + 0.0) / 3, abs=1e-3)
    assert report.redundancy == pytest.approx((1 + 0 + 0 + 0) / 4)

    # Cross-check Wilson intervals directly against scipy.stats.binomtest
    scipy_entry = stats.binomtest(2, 4).proportion_ci(confidence_level=0.95, method="wilson")
    assert report.entrypoint_interval is not None
    assert report.entrypoint_interval.low == pytest.approx(scipy_entry.low, abs=1e-5)
    assert report.entrypoint_interval.high == pytest.approx(scipy_entry.high, abs=1e-5)

    scipy_traj = stats.binomtest(3, 4).proportion_ci(confidence_level=0.95, method="wilson")
    assert report.trajectory_interval is not None
    assert report.trajectory_interval.low == pytest.approx(scipy_traj.low, abs=1e-5)
    assert report.trajectory_interval.high == pytest.approx(scipy_traj.high, abs=1e-5)


def test_classification_report_zero_in_scope_returns_none_efficiency() -> None:
    """Verify classification_report returns None for efficiency metrics when in-scope is 0."""
    queries = (
        Query(query_id="q1", text="oos1", kind=QueryKind.OUT_OF_SCOPE),
        Query(query_id="q2", text="oos2", kind=QueryKind.OUT_OF_SCOPE),
    )
    results = [
        ProbeResult(
            query_id="q1",
            catalog_id="cat",
            catalog_mode=CatalogMode.ALL,
            catalog_size=3,
            model="m",
            runtime="fake",
            invoked_skills=(),
        ),
        ProbeResult(
            query_id="q2",
            catalog_id="cat",
            catalog_mode=CatalogMode.ALL,
            catalog_size=3,
            model="m",
            runtime="fake",
            invoked_skills=("wrong",),
        ),
    ]
    report = classification_report(results, queries)
    assert report.in_scope == 0
    assert report.out_of_scope == 2
    assert report.step_efficiency is None
    assert report.skill_f1 is None


def test_classification_report_cluster_wilson_interval_adjusts_for_attempts() -> None:
    """Verify cluster_wilson_interval is computed with design effect when attempts > 1."""
    from reach.uncertainty import cluster_wilson_interval, wilson_interval

    queries = (
        Query(query_id="q1", text="deploy", expected_skill="deploy"),
        Query(query_id="q2", text="scale", expected_skill="scale"),
    )
    results = [
        ProbeResult(
            query_id="q1",
            catalog_id="cat",
            catalog_mode=CatalogMode.ALL,
            catalog_size=2,
            model="m",
            runtime="fake",
            invoked_skills=("deploy",),
            attempt=att,
        )
        for att in (1, 2, 3)
    ] + [
        ProbeResult(
            query_id="q2",
            catalog_id="cat",
            catalog_mode=CatalogMode.ALL,
            catalog_size=2,
            model="m",
            runtime="fake",
            invoked_skills=(),
            attempt=att,
        )
        for att in (1, 2, 3)
    ]
    report = classification_report(results, queries, attempts=3)
    assert report.attempts == 3
    assert report.top1_hits == 3
    assert report.scored == 6

    expected_cluster_ci = cluster_wilson_interval(3, 6, attempts=3)
    naive_ci = wilson_interval(3, 6)
    assert expected_cluster_ci is not None
    assert naive_ci is not None
    assert report.top1_interval is not None
    cluster_span = report.top1_interval.high - report.top1_interval.low
    naive_span = naive_ci.high - naive_ci.low
    assert cluster_span > naive_span


@pytest.mark.parametrize(
    (
        "invoked_skills",
        "expected_hit",
        "expected_f1",
        "expected_top1_hits",
        "expected_false_abs",
        "expected_fn",
        "expected_recall",
        "expected_macro_prec",
        "expected_confusion_invoked",
    ),
    [
        pytest.param(
            ("gke-basics",),
            False,
            0.0,
            0,
            1,
            1,
            0.0,
            0.0,
            None,
            id="solo_neutral",
        ),
        pytest.param(
            ("gke-basics", "cloud-run-basics"),
            True,
            1.0,
            1,
            0,
            0,
            1.0,
            1.0,
            "cloud-run-basics",
            id="assisted_hit",
        ),
    ],
)
def test_acceptable_skills_are_neutral_in_classification_trajectory_and_collisions(
    invoked_skills: tuple[str, ...],
    expected_hit: bool,
    expected_f1: float,
    expected_top1_hits: int,
    expected_false_abs: int,
    expected_fn: int,
    expected_recall: float,
    expected_macro_prec: float,
    expected_confusion_invoked: str | None,
) -> None:
    """Verify acceptable_skills act as neutral steps (neither TP alone nor FP when followed)."""
    query = Query(
        query_id="q-accept",
        text="deploy container to cloud",
        expected_skill="cloud-run-basics",
        acceptable_skills=("gke-basics",),
    )
    probe = ProbeResult(
        query_id="q-accept",
        catalog_id="c",
        catalog_mode=CatalogMode.ALL,
        catalog_size=2,
        model="m",
        runtime="fake",
        invoked_skills=invoked_skills,
    )

    traj = score_trajectory(query, probe.invoked_skills)
    assert traj.entrypoint_hit is expected_hit
    assert traj.trajectory_hit is expected_hit
    assert traj.skill_f1 == expected_f1
    assert traj.redundancy == 0

    report = classification_report(
        [probe],
        [query],
        labels=["cloud-run-basics", "gke-basics"],
    )
    assert report.top1_hits == expected_top1_hits
    assert report.entrypoint_hits == expected_top1_hits
    assert report.false_abstentions == expected_false_abs
    assert report.by_label("cloud-run-basics").false_negatives == expected_fn
    assert report.by_label("cloud-run-basics").recall == expected_recall
    assert report.by_label("gke-basics").false_positives == 0
    assert report.macro_precision == expected_macro_prec
    assert collisions([probe], [query]) == {}
    assert confusion([probe], [query])[("cloud-run-basics", expected_confusion_invoked)] == 1


def test_class_metrics_rejects_trajectory_tp_below_top1_tp() -> None:
    """Verify ClassMetrics raises ValueError when trajectory_true_positives < true_positives."""
    with pytest.raises(ValueError, match="cannot be less than true_positives"):
        ClassMetrics(
            label="deploy",
            support=2,
            predicted=2,
            true_positives=2,
            trajectory_true_positives=1,
            false_positives=0,
            false_negatives=0,
        )


def test_multi_turn_trajectory_hit_preserves_turn1_conservation() -> None:
    """Verify turn-1 metrics preserve FP/FN conservation while trajectory_recall credits turn 2."""
    query = Query(query_id="q-deploy", text="deploy my service", expected_skill="deploy-service")
    result = ProbeResult(
        query_id="q-deploy",
        catalog_id="c",
        catalog_mode=CatalogMode.ALL,
        catalog_size=2,
        model="m",
        runtime="fake",
        invoked_skills=("gcloud-auth", "deploy-service"),
    )
    report = classification_report([result], [query], labels=["gcloud-auth", "deploy-service"])
    deploy_cls = report.by_label("deploy-service")
    auth_cls = report.by_label("gcloud-auth")

    # Entrypoint recall is 0.0 (turn 1 was gcloud-auth), while trajectory recall is 1.0
    assert deploy_cls.true_positives == 0
    assert deploy_cls.false_negatives == 1
    assert deploy_cls.recall == 0.0
    assert deploy_cls.trajectory_true_positives == 1
    assert deploy_cls.trajectory_recall == 1.0

    # Turn-1 prediction conservation: 1 FN on deploy-service pairs with 1 FP on gcloud-auth
    assert auth_cls.false_positives == 1
    assert auth_cls.predicted == 1
    assert sum(c.predicted for c in report.per_class) == report.scored
    assert collisions([result], [query]) == {("deploy-service", "gcloud-auth"): 1}
    conf = confusion([result], [query])
    assert conf[("deploy-service", "gcloud-auth")] == 1
    assert sum(conf.values()) == report.scored


@pytest.mark.parametrize(
    "skill_assignments",
    [
        ("s1", "s2", "s3", "s4"),
        ("s1", "s1", "s2", "s2"),
    ],
    ids=["singleton-strata-pooled", "multi-skill-strata-rao-wu"],
)
def test_decompose_pass_rate_drop_stratum_rescaling(
    skill_assignments: tuple[str, ...],
) -> None:
    """Verify singleton and multi-skill strata produce non-zero, Rao-Wu rescaled CIs."""
    from reach.metrics import decompose_pass_rate_drop

    queries = [
        Query(query_id=f"q{idx + 1}", text=f"t{idx + 1}", expected_skill=skill)
        for idx, skill in enumerate(skill_assignments)
    ]
    baseline = [
        ProbeResult(
            query_id=q.query_id,
            catalog_id="c0",
            catalog_mode=CatalogMode.ALL,
            catalog_size=4,
            model="m",
            runtime="fake",
            invoked_skills=(q.expected_skill or "",),
        )
        for q in queries
    ]
    scaled = [
        ProbeResult(
            query_id=q.query_id,
            catalog_id="c1",
            catalog_mode=CatalogMode.ALL,
            catalog_size=20,
            model="m",
            runtime="fake",
            invoked_skills=((q.expected_skill or "",) if idx % 2 == 0 else ()),
        )
        for idx, q in enumerate(queries)
    ]
    decomp = decompose_pass_rate_drop(baseline, scaled, queries=queries, iterations=200, seed=42)
    assert decomp.delta_total == 0.5
    assert decomp.delta_total_ci[0] < decomp.delta_total_ci[1]
    assert decomp.delta_total_ci[0] <= 0.5 <= decomp.delta_total_ci[1]
