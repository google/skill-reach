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

"""Verify statistical diff computation, noise floor bounds, and arm comparison."""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from reach.diff import (
    CONTROL,
    NOISE_INFLATION,
    PAIRING,
    TREATMENT,
    VaryFactor,
    diff_runs,
    load_arm,
    noise_floor,
    probes_to_resolve,
    survey_runs,
)
from reach.models import CatalogMode, Query, QueryKind
from reach.queries import Origin, QuerySet, QuerySetProvenance, save_query_set
from reach.uncertainty import DEFAULT_CONFIDENCE, critical_value
from reach.views.diff import (
    DIFF_RENDERERS,
    SURVEY_WIDTH,
    WHERE_COLUMN,
    render_diff,
    render_survey,
)

WIDE = {"size": 3, "rivals": 2}
NARROW = {"size": 2, "rivals": 1}

MISSES = {"q-lifecycle": (None,) * 5, "q-retention": (None,) * 5}
HALF = {"q-lifecycle": ("gcs-lifecycle-rules",) * 5, "q-retention": (None,) * 5}
BARELY_MORE = {
    "q-lifecycle": ("gcs-lifecycle-rules",) * 5,
    "q-retention": ("gcs-retention-policy", None, None, None, None),
}
HITS = {
    "q-lifecycle": ("gcs-lifecycle-rules",) * 5,
    "q-retention": ("gcs-retention-policy",) * 5,
}

DILUTED_CONTROL = {
    "q-lifecycle": ("gcs-lifecycle-rules",) * 10,
    "q-retention": (None,) * 10,
}
DILUTED_TREATMENT = {
    "q-lifecycle": ("gcs-lifecycle-rules",) * 10,
    "q-retention": ("gcs-retention-policy",) * 6 + (None,) * 4,
}


@pytest.fixture
def arm(make_config, record_arm: Callable[..., Path]) -> Callable[..., Path]:
    """Provide helper for creating test arm directories and prediction records."""

    def _arm(name: str, predictions: dict[str, Any], **overrides: Any) -> Path:
        catalog = {**WIDE, **overrides.pop("catalog", {})}
        attempts = max((len(picks) for picks in predictions.values()), default=1)
        plan = {"attempts": attempts, **overrides.pop("plan", {})}
        return record_arm(
            name,
            make_config(catalog=catalog, plan=plan, **overrides),
            predictions,
        )

    return _arm


@pytest.fixture
def edited_corpus(skill_repo: Path, tmp_path: Path) -> Path:
    """Provide modified copy of fixture corpus with updated retention policy description."""
    edited = tmp_path / "skills-edited"
    shutil.copytree(skill_repo, edited)
    card = edited / "storage" / "gcs-retention-policy" / "SKILL.md"
    card.write_text(
        card.read_text(encoding="utf-8").replace(
            "Configures retention and bucket lock.",
            "Holds audit logs for a fixed period under bucket lock.",
        ),
        encoding="utf-8",
    )
    return edited


def _unsidecar(path: Path) -> None:
    """Delete sidecar configuration file for given arm path."""
    Path(f"{path}.config.json").unlink()


def _tamper(path: Path) -> None:
    """Modify sidecar configuration to simulate arm digest mismatch."""
    sidecar = Path(f"{path}.config.json")
    document = json.loads(sidecar.read_text(encoding="utf-8"))
    document["arm"] = "0484b93363ba"
    sidecar.write_text(json.dumps(document), encoding="utf-8")


@pytest.fixture
def disjoint_query_file(tmp_path: Path) -> Path:
    """Provide a query set file containing disjoint queries from standard test sets."""
    return save_query_set(
        QuerySet(
            catalog_id="neighborhood:gcs-lifecycle-rules",
            queries=(
                Query(
                    id="q-elsewhere",
                    text="Tier cold objects after a month.",
                    kind=QueryKind.IMPLICIT,
                    expected_skill="gcs-lifecycle-rules",
                ),
            ),
            provenance=QuerySetProvenance(origin=Origin.AUTHORED),
        ),
        tmp_path / "other-queries.json",
    )


def test_the_floor_is_the_deviate_the_inflation_and_both_errors() -> None:
    """Verify noise_floor formula matches critical value, inflation, and standard errors."""
    floor = noise_floor(0.04, 0.06, DEFAULT_CONFIDENCE)
    assert floor == pytest.approx(critical_value(0.95) * NOISE_INFLATION * 0.10)


def test_the_floor_agrees_with_the_power_arithmetic_already_in_the_tool() -> None:
    """Verify noise floor computation aligns with required probe power calculations."""
    probes = 400
    error = 0.5 / probes**0.5
    floor = noise_floor(error, error)
    assert (critical_value(0.95) * NOISE_INFLATION / floor) ** 2 == pytest.approx(
        probes,
    )
    assert probes_to_resolve(floor) == pytest.approx(probes, rel=0.03)


@pytest.mark.parametrize(
    ("delta", "expected"),
    [(0.50, 26), (0.25, 101), (0.20, 157), (0.10, 628), (0.05, 2512)],
)
def test_the_depth_a_delta_needs_matches_the_published_table(delta, expected) -> None:
    """Verify probes_to_resolve yields expected sample sizes for standard deltas."""
    assert probes_to_resolve(delta) == expected


def test_a_delta_of_nothing_needs_no_depth_because_it_is_not_a_delta() -> None:
    """Verify probes_to_resolve raises ValueError when given delta of 0.0."""
    with pytest.raises(ValueError, match="delta must lie"):
        probes_to_resolve(0.0)


def test_noise_floor_default_and_override_inflation() -> None:
    """Verify noise_floor defaults to NOISE_INFLATION and accepts an override factor."""
    default_floor = noise_floor(0.04, 0.06, DEFAULT_CONFIDENCE)
    assert default_floor == noise_floor(
        0.04,
        0.06,
        DEFAULT_CONFIDENCE,
        noise_inflation=NOISE_INFLATION,
    )
    override_floor = noise_floor(0.04, 0.06, DEFAULT_CONFIDENCE, noise_inflation=1.0)
    assert override_floor == pytest.approx(critical_value(0.95) * 1.0 * 0.10)
    assert override_floor < default_floor


def test_probes_to_resolve_accepts_an_override_inflation_factor() -> None:
    """Verify probes_to_resolve respects custom noise_inflation override."""
    from reach.uncertainty import required_probes

    assert probes_to_resolve(0.10, noise_inflation=1.0) == required_probes(0.10)
    assert probes_to_resolve(0.10, noise_inflation=1.0) != probes_to_resolve(0.10)


def test_a_delta_that_clears_the_floor_is_called_real(arm) -> None:
    """Verify statistically significant deltas are marked real with positive verdict."""
    comparison = diff_runs(
        arm("shipped", MISSES),
        arm("patched", HITS),
        VaryFactor.DESCRIPTION,
    )
    assert comparison.headline.delta == pytest.approx(1.0)
    assert comparison.headline.real
    assert "clearing" in comparison.verdict
    assert comparison.shared_queries == 2


def test_a_delta_inside_the_floor_is_refused_and_priced(arm) -> None:
    """Verify sub-threshold deltas are flagged not real and report required sample sizes."""
    comparison = diff_runs(
        arm("shipped", HALF),
        arm("patched", BARELY_MORE),
        "description",
    )
    assert comparison.headline.delta == pytest.approx(0.1)
    assert not comparison.headline.real
    assert comparison.headline.floor is not None
    assert comparison.headline.floor > 0.1
    assert comparison.headline.needed_probes == probes_to_resolve(0.1) == 628
    assert "not an improvement" in comparison.verdict
    assert "628 probes per arm" in comparison.verdict


def test_tightening_the_confidence_widens_the_floor_until_a_delta_falls_inside(
    arm,
) -> None:
    """Verify increasing confidence level widens noise floor threshold."""
    control, treatment = arm("shipped", MISSES), arm("patched", BARELY_MORE)
    assert diff_runs(control, treatment, "description").headline.real
    strict = diff_runs(control, treatment, "description", confidence=0.999)
    assert not strict.headline.real
    assert strict.headline.floor is not None
    assert strict.headline.confidence == 0.999


def test_diff_runs_records_default_and_overridden_inflation_on_the_headline(arm) -> None:
    """Verify diff_runs records default NOISE_INFLATION and propagates overridden inflation."""
    control, treatment = arm("shipped", HALF), arm("patched", BARELY_MORE)
    calibrated = diff_runs(control, treatment, "description")
    assert calibrated.headline.noise_inflation == NOISE_INFLATION
    uninflated = diff_runs(control, treatment, "description", noise_inflation=1.0)
    assert uninflated.headline.noise_inflation == 1.0
    assert uninflated.headline.floor is not None
    assert calibrated.headline.floor is not None
    assert uninflated.headline.floor < calibrated.headline.floor


def test_the_text_view_shows_the_inflation_actually_applied(arm) -> None:
    """Verify rendered diff text reflects applied noise inflation factor."""
    comparison = diff_runs(
        arm("shipped", HALF),
        arm("patched", BARELY_MORE),
        "description",
        noise_inflation=1.0,
    )
    rendered = render_diff(comparison, "text")
    assert "1.00x over-dispersion" in rendered


@pytest.mark.parametrize(
    ("confidence_arg", "expected_confidence"),
    [
        (None, DEFAULT_CONFIDENCE),
        (0.80, 0.80),
        (0.99, 0.99),
    ],
    ids=["default-confidence", "loose-confidence-80", "strict-confidence-99"],
)
def test_the_confidence_moves_every_interval_and_not_only_the_floor(
    arm,
    confidence_arg: float | None,
    expected_confidence: float,
) -> None:
    """Verify confidence parameter scales headline, skill, and query confidence intervals."""
    control, treatment = arm("one", DILUTED_CONTROL), arm("two", DILUTED_TREATMENT)
    comparison = (
        diff_runs(control, treatment, "description")
        if confidence_arg is None
        else diff_runs(control, treatment, "description", confidence=confidence_arg)
    )
    quoted = {delta.control_interval.confidence for delta in comparison.skills} | {
        delta.treatment_interval.confidence for delta in comparison.queries
    }
    assert quoted == {expected_confidence}


def test_loosening_confidence_narrows_intervals_and_clears_real_threshold(arm) -> None:
    """Verify loosening confidence narrows query intervals and flips real verdict."""
    control, treatment = arm("one", DILUTED_CONTROL), arm("two", DILUTED_TREATMENT)
    default = diff_runs(control, treatment, "description")
    loose = diff_runs(control, treatment, "description", confidence=0.80)
    assert loose.queries[0].control_interval.width < default.queries[0].control_interval.width
    assert loose.headline.real
    assert not default.headline.real


def test_an_arm_compared_with_its_own_twin_finds_nothing(arm) -> None:
    """Verify comparing identical arms yields zero delta and no needed probes."""
    comparison = diff_runs(arm("left", HITS), arm("right", HITS), "description")
    assert comparison.headline.delta == 0.0
    assert not comparison.headline.real
    assert comparison.headline.needed_probes is None
    assert "+0.0 points" in comparison.verdict


def test_an_arm_that_probed_nothing_leaves_no_floor_and_no_verdict(arm) -> None:
    """Verify comparison with unprobed arm produces no noise floor and no verdict."""
    empty = arm("empty", {})
    comparison = diff_runs(empty, arm("probed", HITS), "description")
    assert comparison.control.standard_error is None
    assert comparison.headline.floor is None
    assert not comparison.headline.real
    assert "no verdict" in comparison.verdict


def test_a_description_edit_corroborates_when_both_snapshots_are_on_disk(
    arm,
    edited_corpus: Path,
) -> None:
    """Verify description factor corroboration detects corpus digest changes."""
    comparison = diff_runs(
        arm("shipped", HALF),
        arm("patched", BARELY_MORE, study={"skills": edited_corpus}),
        VaryFactor.DESCRIPTION,
        treatment_corpus=edited_corpus,
    )
    check = comparison.corroboration
    assert check.corpus_moved
    assert not check.arm_moved
    assert check.corroborated
    assert comparison.control.corpus_digest != comparison.treatment.corpus_digest


def test_a_description_edit_read_from_one_snapshot_is_reported_uncorroborated(
    arm,
) -> None:
    """Verify description edit without separate corpus snapshot reports uncorroborated."""
    comparison = diff_runs(
        arm("shipped", HALF),
        arm("patched", BARELY_MORE),
        "description",
    )
    check = comparison.corroboration
    assert not check.corpus_moved
    assert not check.corroborated
    assert "corpus digest should have moved" in check.reason
    assert comparison.headline.floor is not None


def test_dropping_a_rival_corroborates_and_names_the_rival(arm) -> None:
    """Verify rival factor corroboration detects removed rival skills."""
    comparison = diff_runs(
        arm("wide", HALF),
        arm("narrow", BARELY_MORE, catalog=NARROW),
        VaryFactor.RIVAL,
    )
    check = comparison.corroboration
    assert check.removed == ("gke-basics",)
    assert check.added == ()
    assert check.arm_moved
    assert not check.corpus_moved
    assert check.corroborated


def test_rescoping_corroborates_on_the_size_of_the_catalog(arm) -> None:
    """Verify scope factor corroboration confirms differing catalog sizes."""
    comparison = diff_runs(
        arm("wide", HALF),
        arm("narrow", BARELY_MORE, catalog=NARROW),
        "scope",
    )
    assert comparison.control.catalog_size == 3
    assert comparison.treatment.catalog_size == 2
    assert comparison.corroboration.corroborated


def test_a_scope_claim_over_two_identically_scoped_arms_is_not_corroborated(
    arm,
) -> None:
    """Verify scope factor fails corroboration when catalog sizes match."""
    comparison = diff_runs(arm("one", HALF), arm("two", BARELY_MORE), "scope")
    assert not comparison.corroboration.corroborated
    assert "differ in size" in comparison.corroboration.reason


def test_only_the_three_factors_are_accepted(arm) -> None:
    """Verify diff_runs rejects invalid vary factor names."""
    with pytest.raises(ValueError, match="not a valid VaryFactor"):
        diff_runs(arm("one", HALF), arm("two", HITS), "attempts")


@pytest.mark.parametrize(
    ("factor", "control_rt", "treatment_rt", "treatment_catalog", "expected_corroborated"),
    [
        (
            VaryFactor.AGENT,
            {"agent": "fake", "options": {"model": "scripted"}},
            {"agent": "keyword", "options": {"model": "scripted"}},
            WIDE,
            True,
        ),
        (
            VaryFactor.AGENT,
            {"agent": "fake", "options": {"model": "scripted"}},
            {"agent": "fake", "options": {"model": "other-model"}},
            WIDE,
            False,
        ),
        (
            VaryFactor.AGENT,
            {"agent": "fake", "options": {"model": "scripted"}},
            {"agent": "keyword", "options": {"model": "scripted"}},
            NARROW,
            False,
        ),
        (
            VaryFactor.MODEL,
            {"agent": "fake", "options": {"model": "scripted"}},
            {"agent": "fake", "options": {"model": "other-model"}},
            WIDE,
            True,
        ),
        (
            VaryFactor.MODEL,
            {"agent": "fake", "options": {"model": "scripted"}},
            {"agent": "keyword", "options": {"model": "scripted"}},
            WIDE,
            False,
        ),
        (
            VaryFactor.MODEL,
            {"agent": "fake", "options": {"model": "scripted"}},
            {"agent": "fake", "options": {"model": "other-model"}},
            NARROW,
            False,
        ),
        (
            VaryFactor.MODEL,
            {"agent": "fake", "options": {"model": "scripted"}},
            {"agent": "keyword", "options": {"model": "other-model"}},
            WIDE,
            False,
        ),
    ],
)
def test_agent_and_model_vary_factors_corroborate_when_arm_moves_and_corpus_holds(
    make_config,
    record_arm,
    factor: VaryFactor,
    control_rt: dict[str, object],
    treatment_rt: dict[str, object],
    treatment_catalog: dict[str, object],
    expected_corroborated: bool,
) -> None:
    """Verify agent/model factors corroborate only when their dimension moves and residents hold."""
    control = record_arm(
        "control-agent",
        make_config(
            catalog=WIDE,
            plan={"attempts": 5},
            runtime=control_rt,
        ),
        HALF,
    )
    treatment = record_arm(
        "treatment-agent",
        make_config(
            catalog=treatment_catalog,
            plan={"attempts": 5},
            runtime=treatment_rt,
        ),
        HITS,
    )
    comparison = diff_runs(control, treatment, factor)
    assert comparison.corroboration.corroborated is expected_corroborated
    assert comparison.corroboration.arm_moved
    assert not comparison.corroboration.corpus_moved


@pytest.mark.parametrize("factor", list(VaryFactor))
def test_vary_factor_accepts_enum_and_string_identically(arm, factor: VaryFactor) -> None:
    """Verify diff_runs produces identical comparisons for enum and string factor inputs."""
    control, treatment = arm("c", HALF), arm("t", BARELY_MORE)
    from_enum = diff_runs(control, treatment, factor)
    from_str = diff_runs(control, treatment, factor.value)
    assert from_enum.factor == from_str.factor == factor
    assert from_enum.headline.delta == from_str.headline.delta
    assert from_enum.model_dump() == from_str.model_dump()


def test_per_skill_recall_is_judged_by_overlap_and_not_by_the_gap(arm) -> None:
    """Verify per-skill recall differences are evaluated by confidence interval overlap."""
    comparison = diff_runs(arm("one", HALF), arm("two", BARELY_MORE), "description")
    moved = {delta.skill: delta for delta in comparison.skills}
    assert moved["gcs-retention-policy"].delta == pytest.approx(0.2)
    assert not moved["gcs-retention-policy"].real
    assert moved["gcs-retention-policy"].control_interval.overlaps(
        moved["gcs-retention-policy"].treatment_interval,
    )
    assert "gke-basics" not in moved


def test_per_skill_recall_becomes_real_once_the_depth_arrives(arm) -> None:
    """Verify per-skill recall delta becomes significant with sufficient probe depth."""
    misses = {"q-lifecycle": ("gcs-lifecycle-rules",) * 20, "q-retention": (None,) * 20}
    hits = {
        "q-lifecycle": ("gcs-lifecycle-rules",) * 20,
        "q-retention": ("gcs-retention-policy",) * 20,
    }
    comparison = diff_runs(arm("shallow", misses), arm("deep", hits), "description")
    moved = {delta.skill: delta for delta in comparison.skills}
    assert moved["gcs-retention-policy"].real
    assert not moved["gcs-lifecycle-rules"].real
    assert comparison.skills[0].skill == "gcs-retention-policy"


def test_a_query_that_separates_survives_a_pooled_delta_that_does_not(arm) -> None:
    """Verify separated queries are surfaced even when pooled delta is sub-threshold."""
    comparison = diff_runs(
        arm("shipped", DILUTED_CONTROL),
        arm("patched", DILUTED_TREATMENT),
        "description",
    )
    assert not comparison.headline.real
    moved = {delta.query_id: delta for delta in comparison.queries}
    assert moved["q-retention"].delta == pytest.approx(0.6)
    assert moved["q-retention"].real
    assert not moved["q-lifecycle"].real
    assert comparison.separated == (moved["q-retention"],)
    assert "1 of 2 queries separated on their own: q-retention" in comparison.verdict
    assert "probes per arm" in comparison.verdict


def test_a_query_neither_arm_probed_is_left_out_of_the_table(arm) -> None:
    """Verify unprobed queries are excluded from the comparison table."""
    lifecycle_only = {"q-lifecycle": ("gcs-lifecycle-rules",) * 5}
    comparison = diff_runs(
        arm("one", lifecycle_only),
        arm("two", {"q-lifecycle": (None,) * 5}),
        "description",
    )
    assert [delta.query_id for delta in comparison.queries] == ["q-lifecycle"]
    assert comparison.shared_queries == 2


def test_a_query_with_no_kind_still_gets_a_row(
    make_config,
    record_arm,
    tmp_path: Path,
    queries,
) -> None:
    """Verify queries with kind=None are preserved and compared cleanly."""
    unlabeled = save_query_set(
        QuerySet(
            catalog_id="neighborhood:gcs-lifecycle-rules",
            queries=tuple(query.model_copy(update={"kind": None}) for query in queries),
            provenance=QuerySetProvenance(origin=Origin.AUTHORED),
        ),
        tmp_path / "unlabeled.json",
    )
    both = {"catalog": WIDE, "plan": {"attempts": 5}, "study": {"queries": unlabeled}}
    comparison = diff_runs(
        record_arm("before", make_config(**both), HALF),
        record_arm("after", make_config(**both), HITS),
        "description",
    )
    assert [delta.kind for delta in comparison.queries] == [None, None]


def test_arms_scored_on_different_queries_are_refused(
    arm,
    make_config,
    record_arm,
    disjoint_query_file: Path,
) -> None:
    """Verify comparing arms evaluated on disjoint query sets raises ValueError."""
    elsewhere = record_arm(
        "elsewhere",
        make_config(catalog=WIDE, plan={"attempts": 5}, study={"queries": disjoint_query_file}),
        {"q-elsewhere": ("gcs-lifecycle-rules",) * 5},
    )
    with pytest.raises(ValueError, match="scored on different queries"):
        diff_runs(arm("here", HITS), elsewhere, "description")


def test_arms_whose_ground_truth_moved_are_refused(
    arm,
    make_config,
    record_arm,
    tmp_path: Path,
    queries,
) -> None:
    """Verify comparing arms with differing ground truth digests raises ValueError."""
    relabeled = save_query_set(
        QuerySet(
            catalog_id="neighborhood:gcs-lifecycle-rules",
            queries=tuple(
                query.model_copy(
                    update={"text": f"{query.text} (edited)"},
                )
                for query in queries
            ),
            provenance=QuerySetProvenance(origin=Origin.AUTHORED),
        ),
        tmp_path / "relabeled.json",
    )
    moved = record_arm(
        "relabeled",
        make_config(catalog=WIDE, plan={"attempts": 5}, study={"queries": relabeled}),
        HITS,
    )
    with pytest.raises(ValueError, match="ground truth digests differ"):
        diff_runs(arm("original", HITS), moved, "description")


def test_an_arm_cannot_be_compared_with_itself(arm) -> None:
    """Verify diff_runs raises ValueError when given identical control and treatment arms."""
    recorded = arm("only", HITS)
    with pytest.raises(ValueError, match="with itself"):
        diff_runs(recorded, recorded, "description")


def test_a_results_file_with_no_sidecar_cannot_be_read_back(arm) -> None:
    """Verify load_arm raises FileNotFoundError when sidecar config file is missing."""
    recorded = arm("kept", HITS)
    _unsidecar(recorded)
    with pytest.raises(FileNotFoundError, match="no sidecar"):
        load_arm(recorded)


def test_a_tampered_sidecar_is_refused_rather_than_misattributed(arm) -> None:
    """Verify load_arm raises ValueError for sidecars whose arm does not match configuration."""
    recorded = arm("tampered", HITS)
    _tamper(recorded)
    with pytest.raises(ValueError, match="does not match"):
        load_arm(recorded)


def test_an_arm_whose_query_set_has_moved_is_found_under_a_given_root(
    arm,
    query_file: Path,
    tmp_path: Path,
) -> None:
    """Verify relocated query sets are resolved under queries_root without digest mismatch."""
    recorded = arm("moved", HITS)
    relocated = tmp_path / "archive" / query_file.name
    relocated.parent.mkdir()
    query_file.rename(relocated)

    loaded = load_arm(recorded, queries_root=relocated.parent)
    assert loaded.artifact.probes == sum(len(picks) for picks in HITS.values())
    assert loaded.artifact.digests.queries_digest


def test_an_arm_whose_query_set_is_nowhere_names_the_path_it_looked_for(
    arm,
    query_file: Path,
) -> None:
    """Verify load_arm raises FileNotFoundError naming missing query file path."""
    recorded = arm("lost", HITS)
    query_file.unlink()
    with pytest.raises(FileNotFoundError, match=query_file.name):
        load_arm(recorded)


def test_an_arm_whose_corpus_has_moved_says_so(arm, tmp_path: Path) -> None:
    """Verify load_arm raises FileNotFoundError when recorded corpus path cannot be found."""
    recorded = arm("shipped", HITS)
    gone = tmp_path / "nowhere"
    with pytest.raises(FileNotFoundError, match="which is not here"):
        load_arm(recorded, corpus=gone)


def test_an_arm_recorded_by_discovery_names_no_corpus_and_is_refused(arm) -> None:
    """Verify load_arm raises ValueError when loading arm recorded without corpus path."""
    recorded = arm("discovered", HITS)
    sidecar = recorded.with_suffix(".jsonl.config.json")
    document = json.loads(sidecar.read_text(encoding="utf-8"))
    document["config"]["study"]["skills"] = None
    sidecar.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="recorded without a named corpus"):
        load_arm(recorded)


def test_rows_probed_against_another_corpus_are_refused(arm, edited_corpus: Path) -> None:
    """Verify load_arm rejects arms whose row corpus digest disagrees with target corpus."""
    recorded = arm("shipped", HITS)
    with pytest.raises(ValueError, match="measured something else"):
        load_arm(recorded, corpus=edited_corpus)


def test_two_arms_that_were_scored_on_nothing_are_not_compared(arm, query_file: Path) -> None:
    """Verify diff rejects comparisons where both arms contain zero queries."""
    query_file.write_text(
        QuerySet(
            catalog_id="neighborhood:gcs-lifecycle-rules",
            queries=(),
            provenance=QuerySetProvenance(origin=Origin.AUTHORED),
        ).model_dump_json(),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="neither arm was scored on any query"):
        diff_runs(arm("nothing", {}), arm("nothing-either", {}), "description")


def test_a_pairing_with_nothing_standing_surveys_clear_and_crosses_where_it_stands(
    arm,
) -> None:
    """Verify survey_runs reports comparable=True and cross() matches direct diff_runs result."""
    control, treatment = arm("control", HALF), arm("treatment", HITS)
    surveyed = survey_runs(control, treatment, "description")

    assert surveyed.comparable
    assert surveyed.walls == ()
    assert surveyed.cross() == diff_runs(control, treatment, "description")


def test_cross_threads_an_overridden_inflation_the_same_way_diff_runs_does(arm) -> None:
    """Verify surveyed.cross respects noise_inflation override parameter."""
    control, treatment = arm("control", HALF), arm("treatment", HITS)
    surveyed = survey_runs(control, treatment, "description")
    assert surveyed.cross(noise_inflation=1.0) == diff_runs(
        control,
        treatment,
        "description",
        noise_inflation=1.0,
    )


def test_both_unreadable_arms_are_named_rather_than_only_the_first(arm) -> None:
    """Verify survey_runs records separate obstruction walls for both missing sidecars."""
    control, treatment = arm("control", HALF), arm("treatment", HITS)
    _unsidecar(control)
    _unsidecar(treatment)

    surveyed = survey_runs(control, treatment, "description")

    assert not surveyed.comparable
    assert [wall.where for wall in surveyed.walls] == [CONTROL, TREATMENT, PAIRING]
    assert [wall.path for wall in surveyed.walls[:2]] == [
        control.resolve(),
        treatment.resolve(),
    ]
    assert all("no sidecar" in wall.reason for wall in surveyed.walls[:2])


@pytest.mark.parametrize(
    ("broken", "standing"),
    [
        pytest.param((0,), "one of the two arms", id="one arm down"),
        pytest.param((0, 1), "neither arm", id="both arms down"),
    ],
)
def test_a_pairing_nobody_reached_says_so_rather_than_reading_as_fine(
    arm,
    broken: tuple[int, ...],
    standing: str,
) -> None:
    """Verify unreached arm pairing records unreached pairing wall in survey."""
    arms = [arm("control", HALF), arm("treatment", HITS)]
    for index in broken:
        _unsidecar(arms[index])

    surveyed = survey_runs(arms[0], arms[1], "description")

    assert surveyed.walls[-1].where == PAIRING
    assert surveyed.walls[-1].path is None
    assert f"not reached: {standing}" in surveyed.walls[-1].reason
    with pytest.raises(ValueError, match="cannot be crossed"):
        surveyed.cross()


def test_a_pairing_that_fails_twice_says_both_of_its_reasons(
    arm,
    make_config,
    record_arm,
    disjoint_query_file: Path,
) -> None:
    """Verify survey captures multiple distinct pairing obstruction walls."""
    elsewhere = record_arm(
        "elsewhere",
        make_config(catalog=WIDE, plan={"attempts": 5}, study={"queries": disjoint_query_file}),
        {"q-elsewhere": ("gcs-lifecycle-rules",) * 5},
    )

    surveyed = survey_runs(arm("here", HITS), elsewhere, "description")

    assert [wall.where for wall in surveyed.walls] == [PAIRING, PAIRING]
    assert "scored on different queries" in surveyed.walls[0].reason
    assert "ground truth digests differ" in surveyed.walls[1].reason


@pytest.mark.parametrize(
    "break_",
    [
        pytest.param(_unsidecar, id="no sidecar"),
        pytest.param(_tamper, id="tampered sidecar"),
    ],
)
def test_the_strict_path_raises_exactly_the_wall_the_survey_puts_first(arm, break_) -> None:
    """Verify diff_runs exception message matches first wall reason in survey."""
    control, treatment = arm("control", HALF), arm("treatment", HITS)
    break_(control)

    surveyed = survey_runs(control, treatment, "description")
    with pytest.raises((ValueError, OSError)) as raised:
        diff_runs(control, treatment, "description")

    assert str(raised.value) == surveyed.walls[0].reason


def test_an_arm_surveyed_against_itself_is_the_one_wall_and_neither_is_read(
    arm,
) -> None:
    """Verify surveying arm against itself flags self-comparison wall without loading arms."""
    recorded = arm("only", HITS)
    _unsidecar(recorded)

    surveyed = survey_runs(recorded, recorded, "description")

    assert [wall.where for wall in surveyed.walls] == [PAIRING]
    assert "with itself" in surveyed.walls[0].reason
    assert surveyed.control is None
    assert surveyed.treatment is None


def test_a_survey_serializes_without_dragging_both_artifacts_behind_it(arm) -> None:
    """Verify SurveyResult serialization omits full Artifact bodies."""
    surveyed = survey_runs(arm("control", HALF), arm("treatment", HITS), "description")

    document = surveyed.model_dump()

    assert surveyed.control is not None
    assert "control" not in document
    assert "treatment" not in document
    assert document["control_path"] == surveyed.control_path


def test_the_text_view_says_which_tables_are_empty_rather_than_omitting_them(
    arm,
) -> None:
    """Verify text diff explicitly notes when skill or query delta tables are empty."""
    rendered = render_diff(
        diff_runs(arm("planned", {}), arm("also-planned", {}), "description"),
        "text",
    )
    assert "(no skill was named by a query in both arms)" in rendered
    assert "(no query was probed by both arms)" in rendered


def test_the_text_view_handles_empty_shown_queries_without_value_error(arm) -> None:
    """Verify text diff handles comparisons with non-empty queries but empty shown sequence."""
    comparison = diff_runs(
        arm("shipped", DILUTED_CONTROL),
        arm("patched", DILUTED_TREATMENT),
        "description",
    )
    template = comparison.queries[-1]
    empty_shown_comparison = comparison.model_copy(
        update={
            "queries": (template.model_copy(update={"real": True}),),
            "separated": (),
        },
    )
    rendered = render_diff(empty_shown_comparison, "text")
    assert "query" in rendered


def test_the_text_view_shows_the_floor_beside_the_delta(arm) -> None:
    """Verify text diff output includes noise floor and over-dispersion context."""
    comparison = diff_runs(arm("one", HALF), arm("two", BARELY_MORE), "description")
    rendered = render_diff(comparison, "text")
    assert "noise floor" in rendered
    assert "over-dispersion" in rendered
    assert comparison.verdict in rendered
    assert "--vary description" in rendered


def test_a_wide_rescope_counts_the_residents_it_stops_naming(arm) -> None:
    """Verify large catalog additions truncate in text diff while preserved in JSON."""
    comparison = diff_runs(
        arm("wide", HALF),
        arm("narrow", BARELY_MORE, catalog=NARROW),
        "scope",
    )
    crowded = comparison.model_copy(
        update={
            "corroboration": comparison.corroboration.model_copy(
                update={"added": tuple(f"rival-{i:02d}" for i in range(20))},
            ),
        },
    )
    rendered = render_diff(crowded, "text")
    assert "rival-00" in rendered
    assert "and 14 more" in rendered
    assert "rival-19" not in rendered
    assert len(json.loads(render_diff(crowded, "json"))["corroboration"]["added"]) == 20


def test_the_text_view_lifts_the_queries_that_separated_above_the_ones_that_held(
    arm,
) -> None:
    """Verify separated queries are rendered above non-separated queries in text diff."""
    comparison = diff_runs(
        arm("shipped", DILUTED_CONTROL),
        arm("patched", DILUTED_TREATMENT),
        "description",
    )
    template = comparison.queries[-1]
    crowded = comparison.model_copy(
        update={
            "queries": (
                *(template.model_copy(update={"query_id": f"q-filler-{i:02d}"}) for i in range(10)),
                *comparison.queries,
            ),
        },
    )
    rendered = render_diff(crowded, "text")
    lines = rendered.splitlines()
    separated = next(i for i, line in enumerate(lines) if "q-retention" in line)
    assert separated < next(i for i, line in enumerate(lines) if "q-filler-00" in line)
    assert "Per-query hit rate (12)" in rendered
    assert "(4 more, none of them separated)" in rendered


def test_the_json_view_round_trips_every_figure(arm) -> None:
    """Verify JSON diff serialization outputs structured numeric and boolean fields."""
    comparison = diff_runs(arm("one", MISSES), arm("two", HITS), "rival")
    document = json.loads(render_diff(comparison, "json"))
    assert document["factor"] == "rival"
    assert document["headline"]["real"] is True
    assert document["headline"]["delta"] == pytest.approx(1.0)
    assert document["corroboration"]["corroborated"] is False
    assert [row["query_id"] for row in document["queries"]] == [
        "q-lifecycle",
        "q-retention",
    ]


def test_the_csv_view_puts_the_headline_first_and_names_each_test(arm) -> None:
    """Verify CSV diff format prefixes rows with metric type and evaluation test name."""
    comparison = diff_runs(arm("one", HALF), arm("two", BARELY_MORE), "description")
    rows = render_diff(comparison, "csv").splitlines()
    assert rows[0].startswith("figure,")
    assert rows[1].startswith("top1_accuracy,")
    assert rows[1].endswith("noise floor")
    assert all(row.endswith("interval overlap") for row in rows[2:])
    assert any(row.startswith("recall:gcs-retention-policy,") for row in rows)
    assert any(row.startswith("query:q-retention,") for row in rows)


def test_an_unknown_format_names_the_ones_that_exist(arm) -> None:
    """Verify render_diff raises ValueError for unsupported output format."""
    comparison = diff_runs(arm("one", HALF), arm("two", HITS), "description")
    with pytest.raises(ValueError, match="unknown format"):
        render_diff(comparison, "yaml")
    assert set(DIFF_RENDERERS) == {"csv", "json", "jsonl", "text"}


@pytest.mark.parametrize("fmt", sorted(DIFF_RENDERERS))
def test_all_diff_renderers_produce_valid_non_empty_output(arm, fmt: str) -> None:
    """Verify all registered diff renderers produce non-empty formatted output."""
    comparison = diff_runs(arm("one", HALF), arm("two", BARELY_MORE), "description")
    rendered = render_diff(comparison, fmt)
    assert rendered
    if fmt == "json":
        assert json.loads(rendered)
    elif fmt == "jsonl":
        lines = rendered.strip().splitlines()
        assert lines
        for line in lines:
            assert json.loads(line)
    elif fmt == "csv":
        assert rendered.startswith("figure,")
    elif fmt == "text":
        assert "diff --vary description" in rendered


def test_a_rendered_survey_puts_each_reason_under_the_side_it_stands_on(arm) -> None:
    """Verify survey view groups failure reasons by control, treatment, and pairing."""
    control, treatment = arm("control", HALF), arm("treatment", HITS)
    _unsidecar(control)
    _tamper(treatment)

    shown = render_survey(survey_runs(control, treatment, "description"))

    assert "3 walls" in shown.splitlines()[0]
    assert "--vary description" in shown.splitlines()[0]
    assert str(control) in shown
    assert str(treatment) in shown
    for side in (CONTROL, TREATMENT, PAIRING):
        assert any(line.startswith(f"  {side}  ") for line in shown.splitlines())


def test_a_rendered_survey_says_each_path_once_rather_than_twice_in_two_lines(
    arm,
) -> None:
    """Verify rendered survey displays arm file path once above failure reason."""
    control, treatment = arm("control", HALF), arm("treatment", HITS)
    _unsidecar(control)

    shown = render_survey(survey_runs(control, treatment, "description"))

    assert shown.count(str(control)) == 1
    assert f"{control} has no sidecar" not in shown
    assert "has no sidecar" in shown


def test_a_rendered_survey_wraps_its_reasons_without_losing_a_word_of_one(arm) -> None:
    """Verify survey view formats multiline reasons with hanging indentation within width limits."""
    control, treatment = arm("control", HALF), arm("treatment", HITS)
    _unsidecar(control)

    surveyed = survey_runs(control, treatment, "description")
    shown = render_survey(surveyed)

    hanging = [line for line in shown.splitlines() if line.startswith(" " * WHERE_COLUMN)]
    assert hanging
    assert all(len(line) <= SURVEY_WIDTH for line in hanging)
    assert all(not line[WHERE_COLUMN].isspace() for line in hanging)
    flattened = " ".join(shown.split())
    for wall in surveyed.walls:
        said = wall.reason.removeprefix(f"{wall.path} ")
        assert " ".join(said.split()) in flattened


def test_a_rendered_survey_says_a_path_under_the_working_directory_the_short_way(
    arm,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify survey view formats paths relative to current working directory."""
    control, treatment = arm("control", HALF), arm("treatment", HITS)
    _unsidecar(control)
    monkeypatch.chdir(tmp_path)

    shown = render_survey(survey_runs(control, treatment, "description"))

    assert str(control.relative_to(tmp_path)) in shown
    assert str(control) not in shown


def test_a_rendered_survey_of_one_wall_does_not_say_walls(arm) -> None:
    """Verify single-wall survey output uses singular form '1 wall'."""
    recorded = arm("only", HITS)

    shown = render_survey(survey_runs(recorded, recorded, "description"))

    assert "1 wall:" in shown
    assert "1 walls" not in shown


def test_a_rendered_clear_survey_says_the_two_compare(arm) -> None:
    """Verify clear survey produces confirmation sentence."""
    control, treatment = arm("control", HALF), arm("treatment", HITS)

    shown = render_survey(survey_runs(control, treatment, "description"))

    assert shown.startswith("nothing stands between")
    assert "--vary description" in shown


def test_an_arm_is_rebuilt_from_the_rows_rather_than_read_off_disk(arm) -> None:
    """Verify load_arm reconstitutes Artifact from row records and sidecar metadata."""
    loaded = load_arm(arm("shipped", HITS))
    assert loaded.label == "shipped"
    assert loaded.artifact.catalog_mode is CatalogMode.NEIGHBORHOOD
    assert loaded.artifact.probes == 10
    assert set(loaded.artifact.verified_digests) == {
        "config_fingerprint",
        "condition_digest",
        "corpus_digest",
        "queries_digest",
    }


def test_load_arm_reads_artifact_json_directly(arm, tmp_path: Path) -> None:
    """Verify load_arm reads an .artifact.json file directly without requiring a sidecar."""
    from reach.artifact import write_artifact

    recorded = arm("base", HITS)
    original_arm = load_arm(recorded)
    artifact_file = tmp_path / "baseline.artifact.json"
    write_artifact(original_arm.artifact, artifact_file)

    loaded_arm = load_arm(artifact_file)
    assert loaded_arm.label == "baseline"
    assert loaded_arm.artifact.probes == original_arm.artifact.probes
    assert loaded_arm.artifact.catalog_id == original_arm.artifact.catalog_id

    custom_labeled = load_arm(artifact_file, label="custom-arm")
    assert custom_labeled.label == "custom-arm"


def test_load_arm_raises_file_not_found_for_missing_artifact(tmp_path: Path) -> None:
    """Verify load_arm raises FileNotFoundError when an artifact file does not exist."""
    missing = tmp_path / "nonexistent.artifact.json"
    with pytest.raises(FileNotFoundError):
        load_arm(missing)


def test_diff_runs_accepts_artifact_files_directly(arm, tmp_path: Path) -> None:
    """Verify diff_runs compares two .artifact.json files directly across a varied factor."""
    from reach.artifact import write_artifact

    control_path = arm("control", HITS, catalog=NARROW)
    treatment_path = arm("treatment", HITS, catalog=WIDE)

    control_art = write_artifact(
        load_arm(control_path).artifact,
        tmp_path / "control.artifact.json",
    )
    treatment_art = write_artifact(
        load_arm(treatment_path).artifact,
        tmp_path / "treatment.artifact.json",
    )

    comparison = diff_runs(control_art, treatment_art, VaryFactor.SCOPE)
    assert comparison.factor == VaryFactor.SCOPE
    assert comparison.control.label == "control"
    assert comparison.treatment.label == "treatment"
    assert comparison.corroboration.corroborated


@pytest.mark.parametrize(
    ("num_separated", "num_held", "expected_line"),
    [
        pytest.param(10, 0, "(2 more: 2 separated, 0 held)", id="all-hidden-separated"),
        pytest.param(9, 2, "(3 more: 1 separated, 2 held)", id="mixed-hidden-separated-and-held"),
        pytest.param(0, 10, "(2 more, none of them separated)", id="no-hidden-separated"),
    ],
)
def test_query_lines_reports_hidden_separated_when_exceeding_queries_shown(
    arm,
    num_separated: int,
    num_held: int,
    expected_line: str,
) -> None:
    """Verify _query_lines reports both hidden separated and hidden held counts."""
    comparison = diff_runs(
        arm("shipped", DILUTED_CONTROL),
        arm("patched", DILUTED_TREATMENT),
        "description",
    )
    sep_template = comparison.queries[0].model_copy(update={"real": True})
    held_template = comparison.queries[0].model_copy(update={"real": False})
    separated = tuple(
        sep_template.model_copy(update={"query_id": f"q-sep-{i}"}) for i in range(num_separated)
    )
    held = tuple(
        held_template.model_copy(update={"query_id": f"q-held-{i}"}) for i in range(num_held)
    )
    over_comparison = comparison.model_copy(
        update={
            "queries": (*separated, *held),
            "separated": separated,
        },
    )
    rendered = render_diff(over_comparison, "text")
    assert expected_line in rendered


def test_sub_slicing_combined_run_via_queries_subset_and_filters(
    arm,
    edited_corpus: Path,
    tmp_path: Path,
) -> None:
    """Verify survey_runs and load_arm slice combined runs without rewriting jsonl digests."""
    control = arm("control", HALF)
    treatment = arm("treatment", HITS, study={"skills": edited_corpus})

    subset_file = save_query_set(
        QuerySet(
            catalog_id="neighborhood:gcs-lifecycle-rules",
            queries=(
                Query(
                    id="q-retention",
                    text="Hold audit logs for seven years under bucket lock.",
                    kind=QueryKind.IMPLICIT,
                    expected_skill="gcs-retention-policy",
                ),
            ),
            provenance=QuerySetProvenance(origin=Origin.AUTHORED),
        ),
        tmp_path / "subset_retention.json",
    )

    # 1. Slice via subset QuerySet file on unfiltered .jsonl runs
    surveyed = survey_runs(control, treatment, "description", queries=subset_file)
    assert surveyed.comparable
    comparison = surveyed.cross()
    assert comparison.shared_queries == 1
    assert comparison.headline.control == pytest.approx(0.0)
    assert comparison.headline.treatment == pytest.approx(1.0)

    # 2. Slice via filter_skill and filter_id globs
    by_skill = survey_runs(
        control,
        treatment,
        "description",
        filter_skill=("gcs-retention-*",),
    ).cross()
    assert by_skill.shared_queries == 1
    assert by_skill.headline.control == pytest.approx(0.0)
    assert by_skill.headline.treatment == pytest.approx(1.0)

    by_id = survey_runs(
        control,
        treatment,
        "description",
        filter_id=("*-lifecycle",),
    ).cross()
    assert by_id.shared_queries == 1
    assert by_id.headline.control == pytest.approx(1.0)
    assert by_id.headline.treatment == pytest.approx(1.0)

    # 3. Slice a pre-filtered .jsonl whose rows still carry the full-run queries_digest
    filtered_control = tmp_path / "filtered_control.jsonl"
    filtered_treatment = tmp_path / "filtered_treatment.jsonl"
    for src, dst in ((control, filtered_control), (treatment, filtered_treatment)):
        rows = [
            line for line in src.read_text(encoding="utf-8").splitlines() if '"q-retention"' in line
        ]
        dst.write_text("\n".join(rows) + "\n", encoding="utf-8")
        shutil.copy2(f"{src}.config.json", f"{dst}.config.json")

    pre_filtered_cmp = survey_runs(
        filtered_control,
        filtered_treatment,
        "description",
        queries=subset_file,
    ).cross()
    assert pre_filtered_cmp.shared_queries == 1
    assert pre_filtered_cmp.headline.control == pytest.approx(0.0)
    assert pre_filtered_cmp.headline.treatment == pytest.approx(1.0)

    # 4. Slicing via .artifact.json path automatically resolves sibling .jsonl + sidecar
    from reach.artifact import write_artifact

    control_artifact = Path(f"{control}.artifact.json")
    treatment_artifact = Path(f"{treatment}.artifact.json")
    write_artifact(load_arm(control).artifact, control_artifact)
    write_artifact(load_arm(treatment).artifact, treatment_artifact)
    artifact_cmp = survey_runs(
        control_artifact,
        treatment_artifact,
        "description",
        queries=subset_file,
    ).cross()
    assert artifact_cmp.shared_queries == 1
    assert artifact_cmp.headline.control == pytest.approx(0.0)
    assert artifact_cmp.headline.treatment == pytest.approx(1.0)

    # 5. Sad path: subset with altered ground-truth label fails validation
    bad_subset = save_query_set(
        QuerySet(
            catalog_id="neighborhood:gcs-lifecycle-rules",
            queries=(
                Query(
                    id="q-retention",
                    text="Hold audit logs for seven years under bucket lock.",
                    kind=QueryKind.IMPLICIT,
                    expected_skill="gcs-lifecycle-rules",
                ),
            ),
            provenance=QuerySetProvenance(origin=Origin.AUTHORED),
        ),
        tmp_path / "bad_subset.json",
    )
    bad_survey = survey_runs(control, treatment, "description", queries=bad_subset)
    assert not bad_survey.comparable
    assert any("ground truth mismatch" in w.reason for w in bad_survey.walls)

    # 6. Sad path: .jsonl rows with a corrupted queries_digest fail _cross_check
    corrupted_control = tmp_path / "corrupted_control.jsonl"
    corrupted_control.write_text(
        control.read_text(encoding="utf-8").replace(
            load_arm(control).artifact.digests.queries_digest,
            "000000000000",
        ),
        encoding="utf-8",
    )
    shutil.copy2(f"{control}.config.json", f"{corrupted_control}.config.json")
    corrupted_survey = survey_runs(
        corrupted_control,
        treatment,
        "description",
        queries=subset_file,
    )
    assert not corrupted_survey.comparable
    assert any("queries_digest" in w.reason for w in corrupted_survey.walls)
