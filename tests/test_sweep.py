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

"""Validate multi-scale catalog scaling sweep, knee detection, and decomposition telemetry."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from reach.catalog import resolve_sweep_scales
from reach.config import CatalogSettings, PlanSettings, RunConfig, StudySettings
from reach.models import CatalogMode, Query, QueryKind, Skill
from reach.queries import Origin, QuerySet, QuerySetProvenance, save_query_set
from reach.runtime import SelectionOutcome
from reach.runtime.fake import FakeRuntime
from reach.sweep import (
    ScalingPoint,
    ScalingStudy,
    compute_scaling_noise_floor,
    find_kneedle_knee,
    run_scaling_sweep,
)

if TYPE_CHECKING:
    from pathlib import Path


def _create_mock_skills(root: Path, count: int) -> list[Skill]:
    """Create synthetic skill directories and Skill models."""
    skills = []
    for i in range(count):
        name = f"skill-{i:02d}"
        s_dir = root / name
        s_dir.mkdir(parents=True, exist_ok=True)
        (s_dir / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: Description for {name}\n---\n# Body\n",
            encoding="utf-8",
        )
        skills.append(Skill(name=name, description=f"Description for {name}", path=s_dir))
    return skills


def test_find_kneedle_knee_sharp_drop() -> None:
    """Verify log-scale knee curvature identifies inflection point k*."""
    scales = (1, 5, 10, 20, 50, 100)
    pass_rates = (1.0, 0.95, 0.60, 0.55, 0.52, 0.50)
    knee = find_kneedle_knee(scales, pass_rates, noise_floor=0.05)
    assert knee == 10


def test_find_kneedle_knee_flat_returns_none() -> None:
    """Verify flat or within-noise curves return None."""
    scales = (1, 5, 10, 20)
    pass_rates = (0.95, 0.94, 0.95, 0.93)
    knee = find_kneedle_knee(scales, pass_rates, noise_floor=0.05)
    assert knee is None


def test_find_kneedle_knee_too_few_points() -> None:
    """Verify fewer than 3 points returns None."""
    assert find_kneedle_knee((1, 5), (1.0, 0.5)) is None


def test_compute_scaling_noise_floor_reuses_diff() -> None:
    """Verify compute_scaling_noise_floor calculates threshold using diff noise floor."""
    floor = compute_scaling_noise_floor(1.0, 0.5, sample_size=20)
    assert floor > 0.0


def test_paired_trial_outcomes_validation() -> None:
    """Verify PairedTrialOutcomes enforces valid counts and invariants."""
    from pydantic import ValidationError

    from reach.sweep import PairedTrialOutcomes

    p = PairedTrialOutcomes(n10=4, n01=2, total_paired=10)
    assert p.n10 == 4
    assert p.n01 == 2
    assert p.total_paired == 10

    with pytest.raises(ValidationError, match=r"total_paired .* cannot be less than"):
        PairedTrialOutcomes(n10=6, n01=5, total_paired=10)

    with pytest.raises(ValidationError):
        PairedTrialOutcomes(n10=0, n01=0, total_paired=-1)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("base_invocations", "scaled_invocations", "expected_counts"),
    [
        (
            [("q1", "s1"), ("q2", "other"), ("q3", "s3")],
            [("q1", "other"), ("q2", "s2"), ("q3", "s3", "timeout")],
            (1, 1, 2),
        ),
        (
            [("q1", "s1"), ("q2", "s2")],
            [("q1", "s1"), ("q2", "s2")],
            (0, 0, 2),
        ),
        (
            [("q1", "other"), ("q2", "other")],
            [("q1", "other"), ("q2", "other")],
            (0, 0, 2),
        ),
    ],
    ids=["discordant-drop-and-gain-with-error", "all-concordant-pass", "all-concordant-fail"],
)
def test_calculate_paired_outcomes(
    make_result: Any,
    base_invocations: list[tuple[Any, ...]],
    scaled_invocations: list[tuple[Any, ...]],
    expected_counts: tuple[int, int, int],
) -> None:
    """Verify _calculate_paired_outcomes computes discordant pairs and ignores errored probes."""
    from reach.models import Query
    from reach.sweep import _calculate_paired_outcomes

    queries = {
        "q1": Query(id="q1", text="text 1", expected_skill="s1"),
        "q2": Query(id="q2", text="text 2", expected_skill="s2"),
        "q3": Query(id="q3", text="text 3", expected_skill="s3"),
    }

    def _build_results(specs: list[tuple[Any, ...]]) -> list[Any]:
        results = []
        for spec in specs:
            qid, invoked = spec[0], spec[1]
            err = spec[2] if len(spec) > 2 else None
            results.append(make_result(query_id=qid, invoked=invoked, error=err))
        return results

    base_res = _build_results(base_invocations)
    scaled_res = _build_results(scaled_invocations)

    outcomes = _calculate_paired_outcomes(base_res, scaled_res, queries)
    assert outcomes is not None
    assert (outcomes.n10, outcomes.n01, outcomes.total_paired) == expected_counts


def test_compute_scaling_noise_floor_paired_mcnemar_variance() -> None:
    """Verify paired McNemar variance yields a tighter noise floor than independent samples."""
    from reach.sweep import PairedTrialOutcomes

    floor_indep = compute_scaling_noise_floor(0.96, 0.80, sample_size=50)
    paired = PairedTrialOutcomes(n10=8, n01=0, total_paired=50)
    floor_paired = compute_scaling_noise_floor(0.96, 0.80, sample_size=50, paired_outcomes=paired)

    assert 0.01 <= floor_paired < floor_indep
    assert floor_paired < floor_indep * 0.75


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([], []),
        ([0.9], [0.9]),
        ([1.0, 0.9, 0.8], [1.0, 0.9, 0.8]),
        ([0.96, 0.98, 0.90], [0.97, 0.97, 0.90]),
        ([0.9, 0.8, 0.85], [0.9, 0.825, 0.825]),
        ([0.5, 0.6, 0.7], [0.6, 0.6, 0.6]),
    ],
)
def test_isotonic_regression_pava(values: list[float], expected: list[float]) -> None:
    """Verify PAVA projects sequences onto monotone non-increasing cone."""
    from reach.sweep import _isotonic_regression_pava

    assert _isotonic_regression_pava(values) == expected


def test_pava_block_invariants() -> None:
    """Verify _PavaBlock merges weighted averages and maintains total sample weights."""
    from reach.sweep import _merge_pava_blocks, _PavaBlock

    b1 = _PavaBlock(mean=0.80, weight=2.0, size=2)
    b2 = _PavaBlock(mean=0.90, weight=3.0, size=3)
    merged = _merge_pava_blocks(b1, b2)

    assert merged.size == 5
    assert merged.weight == 5.0
    assert merged.mean == round((0.80 * 2.0 + 0.90 * 3.0) / 5.0, 4)


@pytest.mark.parametrize(
    ("scales", "rates", "auto_smooth", "expected_knee"),
    [
        ((10, 25, 50, 100, 147), (0.96, 0.98, 0.96, 0.70, 0.50), True, 50),
        ((10, 25, 50, 100, 147), (1.0, 0.95, 0.90, 0.50, 0.20), False, 50),
    ],
    ids=["upward-bump-auto-smooth", "monotone-raw"],
)
def test_find_kneedle_knee_auto_smooth(
    scales: tuple[int, ...],
    rates: tuple[float, ...],
    auto_smooth: bool,
    expected_knee: int | None,
) -> None:
    """Verify find_kneedle_knee supports opt-in auto_smooth behavior."""
    assert (
        find_kneedle_knee(scales, rates, noise_floor=0.05, auto_smooth=auto_smooth) == expected_knee
    )


def test_compute_sla_crossings_auto_pava() -> None:
    """Verify compute_sla_crossings automatically applies PAVA when smoothed_rates is None."""
    from reach.sweep import compute_sla_crossings

    points = (
        ScalingPoint(
            scale=10,
            catalog_id="cat10",
            pass_rate=0.96,
            pass_rate_interval=(0.90, 0.99),
            f1_score=0.96,
            delta_vs_baseline=0.0,
            delta_context=0.0,
            delta_shadowing=0.0,
            probes_executed=50,
        ),
        ScalingPoint(
            scale=25,
            catalog_id="cat25",
            pass_rate=0.98,
            pass_rate_interval=(0.92, 1.0),
            f1_score=0.98,
            delta_vs_baseline=0.02,
            delta_context=0.0,
            delta_shadowing=0.0,
            probes_executed=50,
        ),
        ScalingPoint(
            scale=50,
            catalog_id="cat50",
            pass_rate=0.88,
            pass_rate_interval=(0.80, 0.94),
            f1_score=0.88,
            delta_vs_baseline=-0.08,
            delta_context=0.0,
            delta_shadowing=-0.08,
            probes_executed=50,
        ),
    )
    discrete_k, interp_k = compute_sla_crossings(points, threshold=0.90, auto_smooth=True)
    assert discrete_k == 25
    assert interp_k is not None
    assert 25.0 < interp_k < 50.0


def test_run_scaling_sweep_insufficient_corpus(tmp_path: Path) -> None:
    """Verify sweeping 0 or 1 skill corpus raises descriptive ValueError."""
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    _create_mock_skills(skills_dir, 1)

    q_file = tmp_path / "queries.json"
    qs = QuerySet(
        catalog_id="all",
        queries=(Query(id="q1", text="text", kind=QueryKind.IMPLICIT, expected_skill="skill-00"),),
        provenance=QuerySetProvenance(origin=Origin.AUTHORED),
    )
    save_query_set(qs, q_file)

    config = RunConfig(
        study=StudySettings(
            skills=skills_dir,
            queries=q_file,
            workdir=tmp_path / "ws",
        ),
    )
    with pytest.raises(ValueError, match="Scaling sweep requires at least 2 skills in corpus"):
        run_scaling_sweep(config, target_skill="skill-00")


def test_sweep_scales_clamping() -> None:
    """Verify corpus smaller than requested scales clamps cleanly and gold standard default."""
    scales = resolve_sweep_scales(total_skills=14, requested=(1, 5, 10, 20, 50, 100))
    assert scales == (1, 5, 10, 14)

    # Test default scales with gold standard progression
    scales_default_large = resolve_sweep_scales(total_skills=300)
    assert scales_default_large == (10, 25, 50, 100, 200, 300)

    scales_default_clamped = resolve_sweep_scales(total_skills=150)
    assert scales_default_clamped == (10, 25, 50, 100, 150)


def test_scaling_sweep_happy_path(tmp_path: Path) -> None:
    """Verify scaling sweep runs end-to-end with FakeRuntime and outputs ScalingStudy."""
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    _create_mock_skills(skills_dir, 20)

    q_file = tmp_path / "queries.yaml"
    qs = QuerySet(
        catalog_id="all",
        queries=(
            Query(id="q0", text="run skill 0", kind=QueryKind.IMPLICIT, expected_skill="skill-00"),
            Query(
                id="q1", text="deploy skill 0", kind=QueryKind.IMPLICIT, expected_skill="skill-00"
            ),
        ),
        provenance=QuerySetProvenance(origin=Origin.AUTHORED),
    )
    save_query_set(qs, q_file)

    study_settings = StudySettings(
        skills=skills_dir,
        queries=q_file,
        workdir=tmp_path / "ws",
        rescope=True,
        partial=True,
    )
    config = RunConfig(
        study=study_settings,
        plan=PlanSettings(attempts=1),
        catalog=CatalogSettings(mode=CatalogMode.SWEEP),
    )

    # FakeRuntime selects target skill for scale 1 and 5, but for scale 10 and 20 selects a rival
    selections = {
        "run skill 0": "skill-00",
        "deploy skill 0": "skill-00",
    }
    runtime = FakeRuntime(selections, model="mock-model")

    study = run_scaling_sweep(
        config=config,
        target_skill="skill-00",
        scales=(1, 5, 10, 20),
        runtime=runtime,
    )

    assert isinstance(study, ScalingStudy)
    assert study.target_skill == "skill-00"
    assert study.scales == (1, 5, 10, 20)
    assert len(study.points) == 4
    for pt in study.points:
        assert isinstance(pt, ScalingPoint)
        assert pt.probes_executed == 2
        assert pt.scale in (1, 5, 10, 20)


def test_run_scaling_sweep_with_duplicate_skills_in_corpus(tmp_path: Path) -> None:
    """Verify run_scaling_sweep executes cleanly when corpus contains duplicate skill names."""
    skills_dir = tmp_path / "skills"
    _create_mock_skills(skills_dir, 5)
    dup_dir = skills_dir / "nested" / "skill-01"
    dup_dir.mkdir(parents=True, exist_ok=True)
    (dup_dir / "SKILL.md").write_text(
        "---\nname: skill-01\ndescription: Duplicate copy\n---\n# Body\n",
        encoding="utf-8",
    )

    q_file = tmp_path / "queries.json"
    qs = QuerySet(
        catalog_id="all",
        queries=(
            Query(id="q0", text="run skill 0", kind=QueryKind.IMPLICIT, expected_skill="skill-00"),
        ),
        provenance=QuerySetProvenance(origin=Origin.AUTHORED),
    )
    save_query_set(qs, q_file)

    config = RunConfig(
        study=StudySettings(
            skills=skills_dir,
            queries=q_file,
            workdir=tmp_path / "ws",
            rescope=True,
            partial=True,
        ),
        plan=PlanSettings(attempts=1),
        catalog=CatalogSettings(mode=CatalogMode.SWEEP),
    )

    runtime = FakeRuntime({"run skill 0": "skill-00"}, model="mock-model", materialize=True)
    study = run_scaling_sweep(
        config=config,
        target_skill="skill-00",
        scales=(1, 3, 5),
        runtime=runtime,
    )
    assert len(study.points) == 3
    for pt in study.points:
        assert pt.probes_failed == 0


def test_run_scaling_sweep_in_memory(tmp_path: Path) -> None:
    """Verify run_scaling_sweep executes using in-memory skills and query_set without disk files."""
    skills = [
        Skill(name=f"skill-{i:02d}", description=f"Skill {i} description", path=tmp_path / f"s{i}")
        for i in range(5)
    ]
    qs = QuerySet(
        catalog_id="in-memory",
        queries=(
            Query(id="q0", text="run skill 0", kind=QueryKind.IMPLICIT, expected_skill="skill-00"),
        ),
        provenance=QuerySetProvenance(origin=Origin.AUTHORED),
    )
    runtime = FakeRuntime({"run skill 0": "skill-00"}, model="mock-model", materialize=False)
    study = run_scaling_sweep(
        skills=skills,
        query_set=qs,
        target_skill="skill-00",
        scales=(1, 3, 5),
        runtime=runtime,
    )
    assert study.target_skill == "skill-00"
    assert study.is_corpus_sweep is False
    assert len(study.points) == 3
    assert study.points[0].pass_rate == 1.0


def test_corpus_scaling_sweep_happy_path(tmp_path: Path) -> None:
    """Verify whole-corpus capacity scaling sweep produces multi-class F1 and SLA thresholds."""
    skills_dir = tmp_path / "skills"
    _create_mock_skills(skills_dir, 6)

    queries = [
        Query(
            id=f"q-{i}",
            text=f"Requesting task number {i:02d}",
            expected_skill=f"skill-{i:02d}",
            kind=QueryKind.IMPLICIT,
        )
        for i in range(6)
    ]
    q_file = tmp_path / "queries.json"
    qs = QuerySet(
        catalog_id="synthetic",
        queries=tuple(queries),
        provenance=QuerySetProvenance(origin=Origin.AUTHORED),
    )
    save_query_set(qs, q_file)

    config = RunConfig(
        study=StudySettings(
            skills=skills_dir,
            queries=q_file,
            workdir=tmp_path / "ws",
            early_stop=False,
        ),
        plan=PlanSettings(attempts=1),
        catalog=CatalogSettings(mode=CatalogMode.SWEEP),
    )

    # Runtime selects the correct skill for all queries
    selections = {f"Requesting task number {i:02d}": f"skill-{i:02d}" for i in range(6)}
    runtime = FakeRuntime(selections, model="mock-model")

    study = run_scaling_sweep(
        config=config,
        target_skill=None,
        scales=(1, 3, 6),
        runtime=runtime,
    )

    assert study.is_corpus_sweep is True
    assert study.target_skill is None
    assert study.total_corpus_skills == 6
    assert len(study.points) == 3
    assert study.sla_90_scale == 6
    assert study.sla_85_scale == 6

    for pt in study.points:
        assert pt.recall == 1.0
        assert pt.precision == 1.0
        assert pt.f1_score == 1.0
        assert pt.negative_probes == 0


def test_corpus_scaling_sweep_early_stopping(tmp_path: Path) -> None:
    """Verify corpus scaling sweep early stops when F1 drops below threshold."""
    skills_dir = tmp_path / "skills"
    _create_mock_skills(skills_dir, 10)

    queries = [
        Query(
            id=f"q-{i}",
            text=f"Requesting task number {i:02d}",
            expected_skill=f"skill-{i:02d}",
            kind=QueryKind.IMPLICIT,
        )
        for i in range(10)
    ]
    q_file = tmp_path / "queries.json"
    qs = QuerySet(
        catalog_id="synthetic",
        queries=tuple(queries),
        provenance=QuerySetProvenance(origin=Origin.AUTHORED),
    )
    save_query_set(qs, q_file)

    config = RunConfig(
        study=StudySettings(
            skills=skills_dir,
            queries=q_file,
            workdir=tmp_path / "ws",
            early_stop=True,
        ),
        plan=PlanSettings(attempts=1),
        catalog=CatalogSettings(mode=CatalogMode.SWEEP),
    )

    # Runtime returns wrong skill for everything, dropping F1 to 0.0
    runtime = FakeRuntime({}, model="mock-model")

    study = run_scaling_sweep(
        config=config,
        target_skill=None,
        scales=(1, 3, 5, 8, 10),
        runtime=runtime,
    )

    assert study.is_corpus_sweep is True
    # Should evaluate scale 1 and scale 3, then early-stop before scale 5
    assert len(study.points) == 2
    assert study.points[-1].f1_score == 0.0


def test_corpus_scaling_sweep_anchor_default(tmp_path: Path) -> None:
    """Verify default anchor cohort uses cluster medoids from the first scale step."""
    skills_dir = tmp_path / "skills"
    _create_mock_skills(skills_dir, 8)

    queries = [
        Query(
            id=f"q-{i}",
            text=f"Requesting task number {i:02d}",
            expected_skill=f"skill-{i:02d}",
            kind=QueryKind.IMPLICIT,
        )
        for i in range(8)
    ]
    q_file = tmp_path / "queries.json"
    qs = QuerySet(
        catalog_id="synthetic",
        queries=tuple(queries),
        provenance=QuerySetProvenance(origin=Origin.AUTHORED),
    )
    save_query_set(qs, q_file)

    config = RunConfig(
        study=StudySettings(
            skills=skills_dir,
            queries=q_file,
            workdir=tmp_path / "ws",
            early_stop=False,
        ),
        plan=PlanSettings(attempts=1),
        catalog=CatalogSettings(mode=CatalogMode.SWEEP),
    )
    selections = {f"Requesting task number {i:02d}": f"skill-{i:02d}" for i in range(8)}
    runtime = FakeRuntime(selections, model="mock-model")

    study = run_scaling_sweep(
        config=config,
        target_skill=None,
        scales=(2, 4, 8),
        runtime=runtime,
    )

    assert study.is_corpus_sweep is True
    assert study.anchor_skills is not None
    assert len(study.anchor_skills) == 2
    # In an anchor sweep, only anchor skill queries are probed at every scale
    for pt in study.points:
        assert pt.in_scope_probes == 2
        assert pt.recall == 1.0


def test_corpus_scaling_sweep_anchor_explicit_and_all(tmp_path: Path) -> None:
    """Verify explicit anchor cohort and anchor='all' full-corpus expansion."""
    skills_dir = tmp_path / "skills"
    _create_mock_skills(skills_dir, 8)

    queries = [
        Query(
            id=f"q-{i}",
            text=f"Requesting task number {i:02d}",
            expected_skill=f"skill-{i:02d}",
            kind=QueryKind.IMPLICIT,
        )
        for i in range(8)
    ]
    q_file = tmp_path / "queries.json"
    qs = QuerySet(
        catalog_id="synthetic",
        queries=tuple(queries),
        provenance=QuerySetProvenance(origin=Origin.AUTHORED),
    )
    save_query_set(qs, q_file)

    config = RunConfig(
        study=StudySettings(
            skills=skills_dir,
            queries=q_file,
            workdir=tmp_path / "ws",
            early_stop=False,
        ),
        plan=PlanSettings(attempts=1),
        catalog=CatalogSettings(mode=CatalogMode.SWEEP),
    )
    selections = {f"Requesting task number {i:02d}": f"skill-{i:02d}" for i in range(8)}
    runtime = FakeRuntime(selections, model="mock-model")

    # Explicit anchor cohort by name
    study_explicit = run_scaling_sweep(
        config=config,
        target_skill=None,
        scales=(3, 6, 8),
        anchor=("skill-01", "skill-03", "skill-05"),
        runtime=runtime,
    )
    assert study_explicit.anchor_skills == ("skill-01", "skill-03", "skill-05")
    for pt in study_explicit.points:
        assert pt.in_scope_probes == 3

    # anchor="all" dynamic expansion
    study_all = run_scaling_sweep(
        config=config,
        target_skill=None,
        scales=(2, 4, 8),
        anchor="all",
        runtime=runtime,
    )
    assert study_all.anchor_skills is None
    # In 'all' mode, in-scope probe count expands with catalog scale
    assert [pt.in_scope_probes for pt in study_all.points] == [2, 4, 8]

    # Nonexistent anchor raises ValueError
    with pytest.raises(ValueError, match=r"anchor skill.*not found in corpus"):
        run_scaling_sweep(
            config=config,
            target_skill=None,
            scales=(2, 4),
            anchor=("nonexistent-skill",),
            runtime=runtime,
        )


def test_compute_sla_crossings() -> None:
    """Verify compute_sla_crossings calculates conservative scale and log-linear interpolation."""
    from reach.sweep import compute_sla_crossings

    points = [
        ScalingPoint(
            scale=5,
            catalog_id="sweep:corpus:5",
            pass_rate=0.95,
            pass_rate_interval=(0.90, 1.0),
            recall=0.95,
            recall_interval=(0.90, 1.0),
            precision=0.95,
            precision_interval=(0.90, 1.0),
            internal_precision=0.95,
            abstention_rate=1.0,
            abstention_interval=(1.0, 1.0),
            f1_score=0.95,
            f1_interval=(0.90, 1.0),
            in_scope_probes=10,
            negative_probes=0,
            delta_vs_baseline=0.0,
            delta_context=0.0,
            delta_shadowing=0.0,
            probes_executed=10,
            probes_failed=0,
        ),
        ScalingPoint(
            scale=10,
            catalog_id="sweep:corpus:10",
            pass_rate=0.85,
            pass_rate_interval=(0.80, 0.90),
            recall=0.85,
            recall_interval=(0.80, 0.90),
            precision=0.85,
            precision_interval=(0.80, 0.90),
            internal_precision=0.85,
            abstention_rate=1.0,
            abstention_interval=(1.0, 1.0),
            f1_score=0.85,
            f1_interval=(0.80, 0.90),
            in_scope_probes=20,
            negative_probes=0,
            delta_vs_baseline=-0.10,
            delta_context=-0.05,
            delta_shadowing=-0.05,
            probes_executed=20,
            probes_failed=0,
        ),
    ]

    sla_scale, sla_interp = compute_sla_crossings(points, threshold=0.90)
    assert sla_scale == 5
    assert sla_interp is not None
    assert 5.0 < sla_interp < 10.0


def test_bootstrap_f1_ci() -> None:
    """Verify bootstrap_f1_ci computes bounded empirical confidence intervals."""
    from reach.models import CatalogMode, ProbeResult
    from reach.sweep import bootstrap_f1_ci

    results = [
        ProbeResult(
            query_id="q1",
            catalog_id="c",
            invoked_skills=("s1",),
            catalog_mode=CatalogMode.SWEEP,
            catalog_size=4,
            model="mock-model",
            runtime="mock-runtime",
        ),
        ProbeResult(
            query_id="q2",
            catalog_id="c",
            invoked_skills=("s2",),
            catalog_mode=CatalogMode.SWEEP,
            catalog_size=4,
            model="mock-model",
            runtime="mock-runtime",
        ),
        ProbeResult(
            query_id="q3",
            catalog_id="c",
            invoked_skills=("s1",),
            catalog_mode=CatalogMode.SWEEP,
            catalog_size=4,
            model="mock-model",
            runtime="mock-runtime",
        ),  # FP
        ProbeResult(
            query_id="q4",
            catalog_id="c",
            invoked_skills=(),
            catalog_mode=CatalogMode.SWEEP,
            catalog_size=4,
            model="mock-model",
            runtime="mock-runtime",
        ),  # abstention / FN
    ]

    truth: dict[str, str | None] = {"q1": "s1", "q2": "s2", "q3": "s3", "q4": "s4"}
    installed = {"s1", "s2", "s3", "s4"}

    ci_low, ci_high = bootstrap_f1_ci(results, truth, installed, iterations=200, seed=42)
    assert 0.0 <= ci_low <= ci_high <= 1.0


def test_build_scaling_point_abstention_none_when_no_negatives() -> None:
    """Verify abstention_rate is None when there are no negative probes in evaluation."""
    from reach.models import CatalogMode, ProbeResult, Query, QueryKind
    from reach.queries import Origin, QuerySet, QuerySetProvenance
    from reach.sweep import _build_scaling_point

    queries = [
        Query(id="q1", text="q1", expected_skill="s1", kind=QueryKind.IMPLICIT),
        Query(id="q2", text="q2", expected_skill="s2", kind=QueryKind.IMPLICIT),
    ]
    query_set = QuerySet(
        catalog_id="c",
        queries=tuple(queries),
        provenance=QuerySetProvenance(origin=Origin.AUTHORED),
    )
    results = [
        ProbeResult(
            query_id="q1",
            catalog_id="c",
            invoked_skills=("s1",),
            catalog_mode=CatalogMode.SWEEP,
            catalog_size=2,
            model="mock",
            runtime="mock",
        ),
        ProbeResult(
            query_id="q2",
            catalog_id="c",
            invoked_skills=("s2",),
            catalog_mode=CatalogMode.SWEEP,
            catalog_size=2,
            model="mock",
            runtime="mock",
        ),
    ]
    pt, _decomp = _build_scaling_point(
        scale=2,
        catalog_id="c",
        results=tuple(results),
        resolved_query_set=query_set,
        baseline_results=(),
        installed_skills={"s1", "s2"},
    )
    assert pt.in_scope_probes == 2
    assert pt.negative_probes == 0
    assert pt.abstention_rate is None
    assert pt.abstention_interval is None
    assert pt.prompt_tokens_mean is None


def test_build_scaling_point_abstention_calculated_when_negatives_present() -> None:
    """Verify abstention_rate is calculated when negative probes are present in evaluation."""
    from reach.models import CatalogMode, ProbeResult, Query, QueryKind
    from reach.queries import Origin, QuerySet, QuerySetProvenance
    from reach.sweep import _build_scaling_point

    queries = [
        Query(id="q1", text="in-scope", expected_skill="s1", kind=QueryKind.IMPLICIT),
        Query(id="q2", text="out-of-scope", expected_skill=None, kind=QueryKind.OUT_OF_SCOPE),
    ]
    query_set = QuerySet(
        catalog_id="c",
        queries=tuple(queries),
        provenance=QuerySetProvenance(origin=Origin.AUTHORED),
    )
    results = [
        ProbeResult(
            query_id="q1",
            catalog_id="c",
            invoked_skills=("s1",),
            catalog_mode=CatalogMode.SWEEP,
            catalog_size=2,
            model="mock",
            runtime="mock",
        ),
        ProbeResult(
            query_id="q2",
            catalog_id="c",
            invoked_skills=(),
            catalog_mode=CatalogMode.SWEEP,
            catalog_size=2,
            model="mock",
            runtime="mock",
        ),
    ]
    pt, _decomp = _build_scaling_point(
        scale=2,
        catalog_id="c",
        results=tuple(results),
        resolved_query_set=query_set,
        baseline_results=(),
        installed_skills={"s1", "s2"},
    )
    assert pt.in_scope_probes == 1
    assert pt.negative_probes == 1
    assert pt.abstention_rate == 1.0
    assert pt.abstention_interval is not None
    assert pt.abstention_interval[0] > 0.0


def test_scaling_sweep_does_not_skip_probes_when_out_path_configured(tmp_path: Path) -> None:
    """Verify scaling sweep evaluates all probes across scales even when study.out is configured."""
    skills_dir = tmp_path / "skills"
    _create_mock_skills(skills_dir, 4)
    queries = [
        Query(
            id=f"q-{i}",
            text=f"Requesting task number {i:02d}",
            expected_skill=f"skill-{i:02d}",
            kind=QueryKind.IMPLICIT,
        )
        for i in range(4)
    ]
    q_file = tmp_path / "queries.json"
    save_query_set(
        QuerySet(
            catalog_id="synthetic",
            queries=tuple(queries),
            provenance=QuerySetProvenance(origin=Origin.AUTHORED),
        ),
        q_file,
    )
    out_file = tmp_path / "sweep_results.jsonl"
    config = RunConfig(
        study=StudySettings(
            skills=skills_dir,
            queries=q_file,
            workdir=tmp_path / "ws",
            out=out_file,
        ),
        plan=PlanSettings(attempts=1),
        catalog=CatalogSettings(mode=CatalogMode.SWEEP),
    )
    selections = {f"Requesting task number {i:02d}": f"skill-{i:02d}" for i in range(4)}
    runtime = FakeRuntime(selections, model="mock-model", materialize=False)

    study = run_scaling_sweep(config=config, scales=(2, 4), runtime=runtime)

    assert len(runtime.queries) == 4
    assert len(study.points) == 2
    assert study.points[0].probes_executed == 2
    assert study.points[1].probes_executed == 2


@pytest.mark.parametrize(
    ("scales", "rates", "expected_knee"),
    [
        pytest.param((1, 10, 100), (0.50, 0.80, 0.95), None, id="rising-curve"),
        pytest.param((1, 10, 100), (1.0, 0.749, 0.50), None, id="straight-log-linear-wiggle"),
        pytest.param((1, 10, 100), (0.95, 0.92, 0.88), None, id="drop-below-0.10-noise-floor"),
        pytest.param((1, 10, 100), (1.0, 0.96, 0.60), 10, id="genuine-knee-above-0.10-drop"),
    ],
)
def test_find_kneedle_knee_behavior(
    scales: tuple[int, ...],
    rates: tuple[float, ...],
    expected_knee: int | None,
) -> None:
    """Verify find_kneedle_knee enforces MIN_KNEE_DROP=0.10 and rejects non-falling curves."""
    assert find_kneedle_knee(scales, rates) == expected_knee


@pytest.mark.parametrize(
    ("f1_values", "expected_discrete", "expected_interp_bounds"),
    [
        pytest.param(
            [(10, 0.95), (25, 0.82), (50, 0.91)],
            10,
            (10.0, 25.0),
            id="downward-crossing-with-late-noise",
        ),
        pytest.param(
            [(10, 0.75), (25, 0.92), (50, 0.95)],
            50,
            (10.0, 25.0),
            id="upward-recovery-crossing",
        ),
    ],
)
def test_compute_sla_crossings_interpolation(
    f1_values: list[tuple[int, float]],
    expected_discrete: int | None,
    expected_interp_bounds: tuple[float, float],
) -> None:
    """Verify compute_sla_crossings interpolates crossings via _log_interpolate_scale."""
    from reach.sweep import _log_interpolate_scale, compute_sla_crossings

    def _pt(scale: int, f1: float) -> ScalingPoint:
        return ScalingPoint(
            scale=scale,
            catalog_id=f"sweep:corpus:{scale}",
            pass_rate=f1,
            pass_rate_interval=(0.0, 1.0),
            f1_score=f1,
            delta_vs_baseline=0.0,
            delta_context=0.0,
            delta_shadowing=0.0,
            probes_executed=10,
        )

    points = [_pt(s, f1) for s, f1 in f1_values]
    discrete_k, interp_k = compute_sla_crossings(points, threshold=0.90)
    assert discrete_k == expected_discrete
    assert interp_k is not None
    assert expected_interp_bounds[0] <= interp_k < expected_interp_bounds[1]
    assert _log_interpolate_scale(points[0], points[1], 0.90) == interp_k


def test_run_scaling_sweep_shares_probe_harness_cache_across_identical_scales(
    tmp_path: Path,
) -> None:
    """Verify ProbeHarness uses Pydantic _ProbeOutcomeCacheKey and shares cache across scales."""
    from pydantic import BaseModel

    from reach.run import _ProbeOutcomeCacheKey
    from reach.runtime.keyword import KeywordRuntime

    assert issubclass(_ProbeOutcomeCacheKey, BaseModel)

    from reach.catalog import load_skills

    skills_dir = tmp_path / "skills"
    _create_mock_skills(skills_dir, 3)
    skills = list(load_skills(skills_dir))
    qs = QuerySet(
        catalog_id="in-memory",
        queries=(
            Query(
                id="q0",
                text="please run skill-00",
                kind=QueryKind.IMPLICIT,
                expected_skill="skill-00",
            ),
        ),
        provenance=QuerySetProvenance(origin=Origin.AUTHORED),
    )

    select_calls = 0

    class CountingKeywordRuntime(KeywordRuntime):
        def select(
            self, query_text: str, workdir: Path, target_skill: str | None = None
        ) -> SelectionOutcome:
            nonlocal select_calls
            select_calls += 1
            return super().select(query_text, workdir, target_skill=target_skill)

    runtime = CountingKeywordRuntime()
    from reach.models import Catalog
    from reach.run import ProbeHarness

    shared_cache: dict = {}
    cat_a = Catalog(
        id="scale-3a",
        mode=CatalogMode.SWEEP,
        target="skill-00",
        skills=tuple(s.name for s in skills),
    )
    cat_b = Catalog(
        id="scale-3b",
        mode=CatalogMode.SWEEP,
        target="skill-00",
        skills=tuple(s.name for s in skills),
    )
    workdir = tmp_path / "ws"
    workdir.mkdir()
    runtime.install(cat_a, skills, workdir)
    h1 = ProbeHarness(runtime, outcome_cache=shared_cache)
    res1 = list(h1.run_probes(qs.queries, cat_a, workdir, attempts=1))
    h2 = ProbeHarness(runtime, outcome_cache=shared_cache)
    res2 = list(h2.run_probes(qs.queries, cat_b, workdir, attempts=1))
    assert select_calls == 1
    assert res1[0].invoked_skills == ("skill-00",)
    assert res2[0].invoked_skills == ("skill-00",)
    assert res2[0].catalog_id == "scale-3b"


@pytest.mark.parametrize(
    (
        "positive_invocations",
        "negative_specs",
        "expected_recall",
        "expected_internal_prec",
        "expected_ext_prec",
        "expected_overall_prec",
        "expected_abstention",
    ),
    [
        pytest.param(
            ["my-skill"] * 5 + ["rival-skill"] * 5,
            [],
            0.5,
            1.0,
            None,
            1.0,
            None,
            id="target-fn-misroute-not-penalized-as-fp",
        ),
        pytest.param(
            ["my-skill"] * 5,
            [((), None), ((), None), (("rival-skill",), None), (("my-skill",), None)],
            1.0,
            1.0,
            round(5 / 6, 4),
            5 / 6,
            0.5,
            id="negative-rival-hijack-excluded-from-target-fp-and-tn",
        ),
        pytest.param(
            ["my-skill"] * 5,
            [((), None), ((), "timeout")],
            1.0,
            1.0,
            1.0,
            1.0,
            0.5,
            id="negative-errored-probe-not-counted-as-tn",
        ),
    ],
)
def test_single_skill_sweep_classification_metrics(
    positive_invocations: list[str],
    negative_specs: list[tuple[tuple[str, ...], str | None]],
    expected_recall: float,
    expected_internal_prec: float,
    expected_ext_prec: float | None,
    expected_overall_prec: float,
    expected_abstention: float | None,
) -> None:
    """Verify targeted sweep metrics handle FNs, rival hijacks, and errored probes."""
    from reach.models import CatalogMode, ProbeResult, Query, QueryKind
    from reach.queries import Origin, QuerySet, QuerySetProvenance
    from reach.sweep import _build_scaling_point

    pos_queries = [
        Query(id=f"q{i}", text=f"query {i}", expected_skill="my-skill", kind=QueryKind.IMPLICIT)
        for i in range(len(positive_invocations))
    ]
    neg_queries = [
        Query(id=f"neg{j}", text=f"neg {j}", expected_skill=None, kind=QueryKind.OUT_OF_SCOPE)
        for j in range(len(negative_specs))
    ]
    query_set = QuerySet(
        catalog_id="c",
        queries=tuple(pos_queries + neg_queries),
        provenance=QuerySetProvenance(origin=Origin.AUTHORED),
    )

    pos_results = [
        ProbeResult(
            query_id=f"q{i}",
            catalog_id="c",
            invoked_skills=(inv,),
            catalog_mode=CatalogMode.SWEEP,
            catalog_size=10,
            model="mock-model",
            runtime="mock-runtime",
        )
        for i, inv in enumerate(positive_invocations)
    ]
    neg_results = [
        ProbeResult(
            query_id=f"neg{j}",
            catalog_id="c",
            invoked_skills=invoked,
            error=err,
            catalog_mode=CatalogMode.SWEEP,
            catalog_size=10,
            model="mock-model",
            runtime="mock-runtime",
        )
        for j, (invoked, err) in enumerate(negative_specs)
    ]

    point, _ = _build_scaling_point(
        scale=10,
        catalog_id="sweep:my-skill:10",
        results=tuple(pos_results + neg_results),
        resolved_query_set=query_set,
        baseline_results=(),
        installed_skills={"my-skill", "rival-skill"},
        target_skill="my-skill",
    )

    assert point.recall == pytest.approx(expected_recall, abs=1e-4)
    assert point.internal_precision == pytest.approx(expected_internal_prec, abs=1e-4)
    assert point.external_distractor_precision == expected_ext_prec
    assert point.precision == pytest.approx(expected_overall_prec, abs=1e-4)
    assert point.abstention_rate == expected_abstention


def test_prompt_tokens_telemetry_propagates_from_outcome_to_scaling_point() -> None:
    """Verify prompt_tokens flows from SessionSummary through ProbeResult to ScalingPoint."""
    from reach.models import Catalog, CatalogMode, ProbeResult, Query, QueryKind
    from reach.queries import Origin, QuerySet, QuerySetProvenance
    from reach.runtime import SessionSummary
    from reach.sweep import _build_scaling_point

    summary = SessionSummary(
        invoked_skills=("my-skill",),
        prompt_tokens=1250,
        duration_ms=420,
    )
    outcome = summary.to_outcome(observed_catalog=("my-skill",))
    assert outcome.prompt_tokens == 1250

    query = Query(id="q1", text="use my-skill", expected_skill="my-skill", kind=QueryKind.IMPLICIT)
    cat = Catalog(
        id="sweep:my-skill:5",
        mode=CatalogMode.SWEEP,
        skills=("my-skill",),
        target="my-skill",
    )
    result = ProbeResult.from_outcome(outcome, query, cat, runtime_name="mock", model="mock")
    assert result.prompt_tokens == 1250

    query_set = QuerySet(
        catalog_id=cat.id,
        queries=(query,),
        provenance=QuerySetProvenance(origin=Origin.AUTHORED),
    )
    point, _ = _build_scaling_point(
        scale=5,
        catalog_id=cat.id,
        results=(result,),
        resolved_query_set=query_set,
        baseline_results=(),
        installed_skills={"my-skill"},
        target_skill="my-skill",
    )
    assert point.prompt_tokens_mean == 1250.0


def test_single_skill_sweep_retains_neighbor_negative_queries_and_tracks_internal_fp(
    tmp_path: Path,
) -> None:
    """Verify _resolve_sweep_target_and_queries retains NEIGHBOR_NEGATIVE queries."""
    from reach.models import CatalogMode, ProbeResult, Query, QueryKind, Skill
    from reach.queries import Origin, QuerySet, QuerySetProvenance
    from reach.sweep import _build_scaling_point, _resolve_sweep_target_and_queries

    skills = [
        Skill(name="my-skill", description="Target skill", path=tmp_path / "my-skill"),
        Skill(name="rival-skill", description="Rival skill", path=tmp_path / "rival-skill"),
    ]
    raw_qs = QuerySet(
        catalog_id="c",
        queries=(
            Query(
                id="pos-1", text="use target", expected_skill="my-skill", kind=QueryKind.IMPLICIT
            ),
            Query(
                id="adv-1",
                text="use rival near miss",
                expected_skill="rival-skill",
                kind=QueryKind.NEIGHBOR_NEGATIVE,
            ),
            Query(
                id="other-pos",
                text="unrelated rival positive",
                expected_skill="rival-skill",
                kind=QueryKind.IMPLICIT,
            ),
        ),
        provenance=QuerySetProvenance(origin=Origin.AUTHORED),
    )
    target, filtered_qs = _resolve_sweep_target_and_queries(skills, raw_qs, "my-skill")
    assert target == "my-skill"
    assert [q.id for q in filtered_qs.queries] == ["pos-1", "adv-1"]

    # At k=1 (only my-skill installed), abstaining on adv-1 is a pass (pass_rate = 1.0)
    baseline_results = (
        ProbeResult(
            query_id="pos-1",
            catalog_id="c1",
            invoked_skills=("my-skill",),
            catalog_mode=CatalogMode.SWEEP,
            catalog_size=1,
            model="mock",
            runtime="mock",
        ),
        ProbeResult(
            query_id="adv-1",
            catalog_id="c1",
            invoked_skills=(),
            catalog_mode=CatalogMode.SWEEP,
            catalog_size=1,
            model="mock",
            runtime="mock",
        ),
    )
    point_k1, _ = _build_scaling_point(
        scale=1,
        catalog_id="c1",
        results=baseline_results,
        resolved_query_set=filtered_qs,
        baseline_results=(),
        installed_skills={"my-skill"},
        target_skill="my-skill",
    )
    assert point_k1.pass_rate == 1.0
    assert point_k1.internal_precision == 1.0

    # If my-skill hijacks adv-1 at k=2, pass_rate = 0.5 and internal_precision = 0.5
    results = (
        ProbeResult(
            query_id="pos-1",
            catalog_id="c",
            invoked_skills=("my-skill",),
            catalog_mode=CatalogMode.SWEEP,
            catalog_size=2,
            model="mock",
            runtime="mock",
        ),
        ProbeResult(
            query_id="adv-1",
            catalog_id="c",
            invoked_skills=("my-skill",),
            catalog_mode=CatalogMode.SWEEP,
            catalog_size=2,
            model="mock",
            runtime="mock",
        ),
    )
    point, _ = _build_scaling_point(
        scale=2,
        catalog_id="c",
        results=results,
        resolved_query_set=filtered_qs,
        baseline_results=baseline_results,
        installed_skills={"my-skill", "rival-skill"},
        target_skill="my-skill",
    )
    assert point.pass_rate == 0.5
    assert point.recall == 1.0
    assert point.internal_precision == 0.5
    assert point.delta_vs_baseline == 0.0


def test_render_ascii_curve_single_bullet_per_column_on_midpoint_boundaries() -> None:
    """Verify render_ascii_curve maps midpoint boundary values to exactly one row bullet."""
    from reach.views.sweep import render_ascii_curve

    points = [
        ScalingPoint(
            scale=s,
            catalog_id=f"c:{s}",
            pass_rate=val,
            pass_rate_interval=(0.0, 1.0),
            f1_score=val,
            delta_vs_baseline=0.0,
            delta_context=0.0,
            delta_shadowing=0.0,
            probes_executed=10,
        )
        for s, val in [(10, 0.875), (20, 0.625), (30, 0.375), (40, 0.125)]
    ]
    lines = render_ascii_curve(points, metric="f1")
    level_rows = [line.split("|", 1)[1] for line in lines if "|" in line]
    total_bullets = sum(row.count("●") for row in level_rows)
    assert total_bullets == len(points)


def test_resolve_anchor_and_target_skills_fuzzy_suggestions(tmp_path: Path) -> None:
    """Verify missing --anchor and --target skill names include fuzzy close-match hints."""
    skills = [
        Skill(
            name="cloud-logging-configuration-basics",
            description="Configure logging sinks and buckets",
            path=tmp_path / "s1",
        ),
        Skill(
            name="cloud-logging-cross-project-configuration",
            description="Route cross-project logs",
            path=tmp_path / "s2",
        ),
        Skill(
            name="gke-cluster-autoscaling",
            description="Autoscale GKE node pools",
            path=tmp_path / "s3",
        ),
    ]
    qs = QuerySet(
        catalog_id="all",
        queries=(
            Query(
                id="q1",
                text="configure sink",
                kind=QueryKind.IMPLICIT,
                expected_skill="cloud-logging-configuration-basics",
            ),
        ),
        provenance=QuerySetProvenance(origin=Origin.AUTHORED),
    )

    with pytest.raises(
        ValueError,
        match=(
            r"anchor skill\(s\) not found in corpus: cloud-logging-sinks\. "
            r"Did you mean: .*cloud-logging-configuration-basics"
        ),
    ):
        run_scaling_sweep(
            skills=skills,
            query_set=qs,
            scales=(2, 3),
            anchor="cloud-logging-sinks",
        )

    with pytest.raises(
        (ValueError, KeyError),
        match=(
            r"target skill 'cloud-logging-sinks' not found in loaded skills\. "
            r"Did you mean: .*cloud-logging-configuration-basics"
        ),
    ):
        run_scaling_sweep(
            skills=skills,
            query_set=qs,
            scales=(2, 3),
            target_skill="cloud-logging-sinks",
        )


@pytest.mark.parametrize(
    ("base_workers", "scale", "ref_scale", "expected"),
    [
        (1, 128, 25, 1),
        (16, 12, 25, 16),
        (16, 25, 25, 16),
        (16, 50, 25, 8),
        (16, 90, 25, 4),
        (16, 128, 25, 3),
    ],
)
def test_scale_adaptive_workers_tapers_concurrency(
    base_workers: int,
    scale: int,
    ref_scale: int,
    expected: int,
) -> None:
    """Verify _scale_adaptive_workers tapers worker concurrency linearly with catalog size K."""
    from reach.sweep import _scale_adaptive_workers

    assert _scale_adaptive_workers(base_workers, scale, reference_scale=ref_scale) == expected


def test_run_scaling_sweep_invokes_on_scale_complete_and_tapers_workers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify run_scaling_sweep invokes on_scale_complete after each step and tapers workers."""
    import reach.sweep as sweep_mod

    skills = [
        Skill(name=f"skill-{i:02d}", description=f"Skill {i} description", path=tmp_path / f"s{i}")
        for i in range(60)
    ]
    qs = QuerySet(
        catalog_id="in-memory",
        queries=(
            Query(id="q0", text="run skill 0", kind=QueryKind.IMPLICIT, expected_skill="skill-00"),
        ),
        provenance=QuerySetProvenance(origin=Origin.AUTHORED),
    )
    runtime = FakeRuntime({"run skill 0": "skill-00"}, model="mock-model", materialize=False)

    observed_workers: list[tuple[int, int]] = []
    orig_conduct = sweep_mod.conduct

    def spy_conduct(*args, **kwargs):
        composed = kwargs["composed"]
        observed_workers.append((len(composed.catalog.skills), kwargs["workers"]))
        return orig_conduct(*args, **kwargs)

    monkeypatch.setattr(sweep_mod, "conduct", spy_conduct)

    callbacks: list[tuple[int, int, int, int]] = []

    def on_step(step: int, total: int, point: ScalingPoint, partial: ScalingStudy) -> None:
        callbacks.append((step, total, point.scale, len(partial.points)))

    out_file = tmp_path / ".reach" / "sweep.json"
    cfg = RunConfig(study=StudySettings(out=out_file))

    study = run_scaling_sweep(
        config=cfg,
        skills=skills,
        query_set=qs,
        target_skill="skill-00",
        scales=(12, 25, 50, 70),
        workers=16,
        runtime=runtime,
        on_scale_complete=on_step,
    )

    assert len(study.points) == 4  # 12, 25, 50, 60 (clamped corpus max)
    assert callbacks == [
        (1, 4, 12, 1),
        (2, 4, 25, 2),
        (3, 4, 50, 3),
        (4, 4, 60, 4),
    ]
    assert observed_workers == [
        (12, 16),
        (25, 16),
        (50, 8),
        (60, 7),
    ]

    # Verify interrupted sweep preserves intermediate .jsonl files in temp workdir
    interrupted_workdirs: list[Path] = []

    def fail_on_second_step(
        step: int, total: int, point: ScalingPoint, partial: ScalingStudy
    ) -> None:
        if step == 2:
            err_msg = "Simulated mid-sweep interruption"
            raise RuntimeError(err_msg)

    def capture_workdir(*args, **kwargs):
        cfg_arg = kwargs["config"]
        interrupted_workdirs.append(cfg_arg.study.workdir)
        return orig_conduct(*args, **kwargs)

    monkeypatch.setattr(sweep_mod, "conduct", capture_workdir)
    with pytest.raises(RuntimeError, match="Simulated mid-sweep interruption"):
        run_scaling_sweep(
            skills=skills,
            query_set=qs,
            target_skill="skill-00",
            scales=(12, 25, 50),
            runtime=runtime,
            on_scale_complete=fail_on_second_step,
        )
    assert interrupted_workdirs
    assert interrupted_workdirs[0].is_dir()
    assert list(interrupted_workdirs[0].glob("sweep_*.jsonl"))
    import shutil

    shutil.rmtree(interrupted_workdirs[0], ignore_errors=True)


def test_sweep_scores_two_turn_mutual_handoff_as_true_positive_and_records_entrypoint() -> None:
    """Verify 2-turn mutual handoff (mixed_oracle) and acceptable_skills score as TP in sweep."""
    from reach.models import CatalogMode, InvocationPattern, ProbeResult
    from reach.sweep import _build_scaling_point

    queries = (
        Query(
            id="q-handoff",
            text="harden GKE cluster security posture",
            expected_skill="gke-platform-security",
            kind=QueryKind.IMPLICIT,
        ),
        Query(
            id="q-acceptable",
            text="deploy agent endpoint on Vertex",
            expected_skill="agent-platform-deploy",
            acceptable_skills=("gcloud",),
            kind=QueryKind.IMPLICIT,
        ),
    )
    qs = QuerySet(
        catalog_id="sweep:corpus:128",
        queries=queries,
        provenance=QuerySetProvenance(origin=Origin.AUTHORED),
    )
    baseline_results = (
        ProbeResult(
            query_id="q-handoff",
            catalog_id="sweep:corpus:12",
            invoked_skills=("gke-platform-security",),
            invocation_pattern=InvocationPattern.ORACLE_ONLY,
            turns_taken=1,
            catalog_mode=CatalogMode.SWEEP,
            catalog_size=12,
            model="gemini-3.8-flash",
            runtime="antigravity-sdk",
        ),
        ProbeResult(
            query_id="q-acceptable",
            catalog_id="sweep:corpus:12",
            invoked_skills=("agent-platform-deploy",),
            invocation_pattern=InvocationPattern.ORACLE_ONLY,
            turns_taken=1,
            catalog_mode=CatalogMode.SWEEP,
            catalog_size=12,
            model="gemini-3.8-flash",
            runtime="antigravity-sdk",
        ),
    )
    scaled_results = (
        ProbeResult(
            query_id="q-handoff",
            catalog_id="sweep:corpus:128",
            invoked_skills=("gke-basics", "gke-platform-security"),
            invocation_pattern=InvocationPattern.MIXED_ORACLE,
            turns_taken=2,
            catalog_mode=CatalogMode.SWEEP,
            catalog_size=128,
            model="gemini-3.8-flash",
            runtime="antigravity-sdk",
        ),
        ProbeResult(
            query_id="q-acceptable",
            catalog_id="sweep:corpus:128",
            invoked_skills=("gcloud", "agent-platform-deploy"),
            invocation_pattern=InvocationPattern.ORACLE_ONLY,
            turns_taken=2,
            catalog_mode=CatalogMode.SWEEP,
            catalog_size=128,
            model="gemini-3.8-flash",
            runtime="antigravity-sdk",
        ),
    )

    point, decomp = _build_scaling_point(
        scale=128,
        catalog_id="sweep:corpus:128",
        results=scaled_results,
        resolved_query_set=qs,
        baseline_results=baseline_results,
        installed_skills={
            "gke-basics",
            "gke-platform-security",
            "gcloud",
            "agent-platform-deploy",
        },
        target_skill=None,
    )

    assert point.pass_rate == 1.0
    assert point.recall == 1.0
    assert point.precision == 1.0
    assert point.f1_score == 1.0
    assert point.delta_vs_baseline == 0.0
    assert decomp is not None
    assert decomp.delta_total == 0.0
    # Turn-1 entrypoint metrics reflect that q-handoff needed 2 turns
    # while q-acceptable used neutral gcloud first
    assert point.entrypoint_pass_rate == 0.5
    assert point.entrypoint_f1_score == 0.5


def test_run_scaling_sweep_invalidates_workdir_cache_on_anchor_or_skill_edit(
    tmp_path: Path,
) -> None:
    """Verify persistent workdir sweep_*.jsonl cache invalidates on anchor or SKILL.md change."""
    from reach.catalog import load_skills
    from reach.config import RunConfig, StudySettings

    skills_dir = tmp_path / "corpus"
    _create_mock_skills(skills_dir, 4)
    skills_v1 = load_skills(skills_dir)
    qs = QuerySet(
        catalog_id="corpus",
        queries=tuple(
            Query(
                id=f"q-{idx}",
                text=f"query for skill-{idx:02d}",
                kind=QueryKind.IMPLICIT,
                expected_skill=f"skill-{idx:02d}",
            )
            for idx in range(4)
        ),
        provenance=QuerySetProvenance(origin=Origin.AUTHORED),
    )

    workdir = tmp_path / "persistent_wd"
    cfg = RunConfig(study=StudySettings(workdir=workdir))
    answers = {f"query for skill-{idx:02d}": f"skill-{idx:02d}" for idx in range(4)}

    def _probe_count(anchor: str, corpus: list[Skill]) -> int:
        rt = FakeRuntime(answers, model="mock-model", materialize=False)
        study = run_scaling_sweep(
            skills=corpus,
            query_set=qs,
            scales=(2, 4),
            anchor=anchor,
            attempts=1,
            runtime=rt,
            config=cfg,
        )
        assert study.points[0].scale == 2
        return len(rt.queries)

    # Initial run for anchor="skill-00" executes 2 probes (K=2 and K=4)
    assert _probe_count("skill-00", skills_v1) == 2

    # 1. Change --anchor to "skill-00,skill-03": K=2 catalog changed, so both q-0 and q-3 run at K=2
    #    (2 probes) while at K=4 q-0 reuses K=4 and q-3 runs (1 probe) -> 3 probes total
    assert _probe_count("skill-00,skill-03", skills_v1) == 3

    # 2. Switching back to anchor="skill-00" reuses preserved K=2 and K=4 rows on disk (0 probes)
    assert _probe_count("skill-00", skills_v1) == 0

    # 3. Editing SKILL.md changes corpus_digest and invalidates all cached rows (4 probes)
    (skills_dir / "skill-03" / "SKILL.md").write_text(
        "---\nname: skill-03\ndescription: Updated Desc 3\n---\nUpdated Body 3\n",
        encoding="utf-8",
    )
    skills_v2 = load_skills(skills_dir)
    assert _probe_count("skill-00,skill-03", skills_v2) == 4


def test_run_scaling_sweep_listing_budget_guard(tmp_path: Path) -> None:
    """Verify run_scaling_sweep guards against catalog overflow on rationing runtimes."""
    from reach.catalog import load_skills
    from reach.config import RunConfig, StudySettings
    from reach.runtime import CatalogFit

    class RationingFakeRuntime(FakeRuntime):
        @property
        def rations_catalog(self) -> bool:
            return True

        def fit(self, catalog, skills) -> CatalogFit:
            return CatalogFit(
                truncated=1,
                allowed=50,
                asked=200,
                unit="characters",
                remedy="Reduce catalog size",
            )

    skills_dir = tmp_path / "corpus"
    _create_mock_skills(skills_dir, 4)
    skills = load_skills(skills_dir)
    qs = QuerySet(
        catalog_id="corpus",
        queries=tuple(
            Query(
                id=f"q-{idx}",
                text=f"query for skill-{idx:02d}",
                kind=QueryKind.IMPLICIT,
                expected_skill=f"skill-{idx:02d}",
            )
            for idx in range(4)
        ),
        provenance=QuerySetProvenance(origin=Origin.AUTHORED),
    )

    cfg = RunConfig(study=StudySettings(workdir=tmp_path / "work"))
    answers = {f"query for skill-{idx:02d}": f"skill-{idx:02d}" for idx in range(4)}
    rt = RationingFakeRuntime(answers, model="mock-model", materialize=False)

    # By default (allow_truncation=True), sweep proceeds and measures real runtime capacity
    study = run_scaling_sweep(
        skills=skills,
        query_set=qs,
        scales=(2, 4),
        attempts=1,
        runtime=rt,
        config=cfg,
    )
    assert len(study.points) == 2

    # When allow_truncation=False is explicitly passed, sweep raises ValueError
    # pointing to --allow-truncation
    with pytest.raises(ValueError, match=r"bare name.*--allow-truncation"):
        run_scaling_sweep(
            skills=skills,
            query_set=qs,
            scales=(2, 4),
            attempts=1,
            runtime=rt,
            config=cfg,
            allow_truncation=False,
        )


def test_resolve_anchor_skills_filters_to_queried_skills_and_warns_missing_corpus(
    tmp_path: Path,
) -> None:
    """Verify auto-medoids prefer queried skills and _print_anchor_coverage warns."""
    from io import StringIO

    from reach.cli.sweep import _print_anchor_coverage
    from reach.queries import save_query_set
    from reach.sweep import _resolve_anchor_skills
    from reach.views import Console

    skills = [
        Skill(
            name=f"skill-{idx:02d}",
            description=f"Unique topic-{idx:02d} workflow handler",
            path=tmp_path / f"skill-{idx:02d}" / "SKILL.md",
        )
        for idx in range(5)
    ]
    # Only skill-00 and skill-01 have queries; skill-02, skill-03, skill-04 have 0 queries
    partial_qs = QuerySet(
        catalog_id="all",
        queries=(
            Query(
                id="q0",
                text="query for skill-00",
                kind=QueryKind.IMPLICIT,
                expected_skill="skill-00",
            ),
            Query(
                id="q1",
                text="query for skill-01",
                kind=QueryKind.IMPLICIT,
                expected_skill="skill-01",
            ),
        ),
        provenance=QuerySetProvenance(origin=Origin.AUTHORED),
    )

    anchors = _resolve_anchor_skills(None, skills, (2, 5), query_set=partial_qs)
    assert anchors is not None
    assert set(anchors) == {"skill-00", "skill-01"}

    q_path = tmp_path / "queries.json"
    save_query_set(partial_qs, q_path)
    buf = StringIO()
    console = Console(file=buf, force_terminal=False, width=120)
    _print_anchor_coverage(
        console=console,
        queries_path=q_path,
        anchor=None,
        configured_anchor=None,
        skills=skills,
        scales=(2, 5),
        target=None,
    )
    out = buf.getvalue()
    assert "3 of 5 corpus skill(s) have 0 queries" in out
    assert "reach query draft --sync" in out
    assert "2/2 anchor skills" in out


def test_resolve_anchor_skills_clamps_to_queried_skills_when_fewer_than_scale(
    tmp_path: Path,
) -> None:
    """Verify _resolve_anchor_skills clamps to queried skills when fewer than initial scale."""
    from reach.models import Query, QueryKind, Skill
    from reach.queries import Origin, QuerySet, QuerySetProvenance
    from reach.sweep import _resolve_anchor_skills

    skills = [
        Skill(name=f"skill-{i:02d}", description=f"Skill {i}", path=tmp_path / f"s{i}")
        for i in range(6)
    ]
    partial_qs = QuerySet(
        catalog_id="partial",
        queries=(
            Query(id="q0", text="use s0", kind=QueryKind.IMPLICIT, expected_skill="skill-00"),
            Query(id="q1", text="use s1", kind=QueryKind.IMPLICIT, expected_skill="skill-01"),
        ),
        provenance=QuerySetProvenance(origin=Origin.AUTHORED),
    )
    # Requested medoid count is 4 (actual_scales[0] == 4), but only 2 skills have queries
    anchors = _resolve_anchor_skills(None, skills, (4, 6), query_set=partial_qs)
    assert anchors is not None
    assert len(anchors) == 2
    assert set(anchors) == {"skill-00", "skill-01"}


def test_render_ascii_curve_auto_scales_high_accuracy_band() -> None:
    """Verify render_ascii_curve zooms into [80%..100%] when all points are >= 75%."""
    from reach.sweep import ScalingPoint
    from reach.views.sweep import render_ascii_curve

    def _pt(scale: int, f1: float) -> ScalingPoint:
        return ScalingPoint(
            scale=scale,
            catalog_id=f"c{scale}",
            pass_rate=f1,
            pass_rate_interval=(f1 - 0.05, min(1.0, f1 + 0.05)),
            f1_score=f1,
            delta_vs_baseline=0.975 - f1,
            delta_context=0.0,
            delta_shadowing=0.975 - f1,
            probes_executed=41,
        )

    pts = (_pt(10, 0.975), _pt(50, 0.937), _pt(100, 0.914), _pt(147, 0.875))
    lines = render_ascii_curve(pts, metric="f1")
    joined = "\n".join(lines)
    assert " 95% |" in joined
    assert " 90% |" in joined
    assert " 85% |" in joined
    assert " 80% |" in joined
    marker_rows = [line for line in lines if "●" in line]
    assert len(marker_rows) >= 3


def test_print_sweep_surfaces_shadowing_and_truncation_without_ellipsis() -> None:
    """Verify print_sweep renders Δ Shadow, Truncated, and Loss Decomposition within 80 columns."""
    from io import StringIO

    from rich.console import Console

    from reach.sweep import ScalingPoint, ScalingStudy
    from reach.views.sweep import print_sweep

    study = ScalingStudy(
        target_skill=None,
        is_corpus_sweep=True,
        anchor_skills=("a1", "a2"),
        scales=(10, 100, 147),
        knee_scale=25,
        sla_90_scale=100,
        sla_90_interpolated=114.5,
        sla_85_scale=147,
        sla_85_interpolated=147.0,
        baseline_pass_rate=0.951,
        final_pass_rate=0.854,
        total_delta=0.097,
        total_context_loss=0.024,
        total_shadowing_loss=0.098,
        total_corpus_skills=147,
        points=(
            ScalingPoint(
                scale=10,
                catalog_id="s10",
                pass_rate=0.951,
                pass_rate_interval=(0.88, 0.99),
                recall=0.951,
                precision=1.0,
                f1_score=0.975,
                f1_interval=(0.935, 1.0),
                delta_vs_baseline=0.0,
                delta_context=0.0,
                delta_shadowing=0.0,
                in_scope_probes=41,
                probes_executed=41,
                duration_ms_mean=3393.0,
                disclosure_states={"full": 41},
            ),
            ScalingPoint(
                scale=100,
                catalog_id="s100",
                pass_rate=0.902,
                pass_rate_interval=(0.80, 0.96),
                recall=0.902,
                precision=0.925,
                f1_score=0.914,
                f1_interval=(0.825, 0.988),
                delta_vs_baseline=0.049,
                delta_context=0.0,
                delta_shadowing=0.073,
                in_scope_probes=41,
                probes_executed=41,
                duration_ms_mean=4472.0,
                disclosure_states={"full": 20, "name_only_elided": 21},
            ),
            ScalingPoint(
                scale=147,
                catalog_id="s147",
                pass_rate=0.854,
                pass_rate_interval=(0.74, 0.93),
                recall=0.854,
                precision=0.897,
                f1_score=0.875,
                f1_interval=(0.769, 0.975),
                delta_vs_baseline=0.097,
                delta_shadowing=0.098,
                delta_context=0.024,
                in_scope_probes=41,
                probes_executed=41,
                duration_ms_mean=4394.0,
                disclosure_states={"full": 16, "name_only_elided": 25},
            ),
        ),
    )
    buf = StringIO()
    console = Console(file=buf, force_terminal=False, width=80)
    print_sweep(console, study)
    out = buf.getvalue()
    assert "Loss Decomposition (K=10→147): Δ Shadowing +9.8% | Δ Context +2.4%" in out
    assert "Δ Shadow" in out
    assert "Trunc" in out
    assert "51% (21)" in out
    assert "61% (25)" in out
    assert "…" not in out


def test_resolve_anchor_skills_warns_when_query_set_has_no_resident_queries(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Verify _resolve_anchor_skills warns when query_set has 0 queries for resident skills."""
    import logging

    from reach.models import Skill
    from reach.queries import Origin, QuerySet, QuerySetProvenance
    from reach.sweep import _resolve_anchor_skills

    skills = [
        Skill(name="skill-a", description="A", path=tmp_path / "skill-a"),
        Skill(name="skill-b", description="B", path=tmp_path / "skill-b"),
    ]
    qs = QuerySet(
        catalog_id="cat",
        queries=(Query(id="q-1", text="Other query", expected_skill="other-skill"),),
        provenance=QuerySetProvenance(origin=Origin.AUTHORED),
    )
    with caplog.at_level(logging.WARNING):
        anchors = _resolve_anchor_skills(
            requested_anchor=1,
            resolved_skills=skills,
            actual_scales=(1, 2),
            query_set=qs,
        )
    assert anchors is not None
    assert len(anchors) == 1
    assert "Provided query set contains 0 benchmark queries for resident skills" in caplog.text


def test_resolve_anchor_skills_unclamped_computes_full_corpus_medoids(
    tmp_path: Path,
) -> None:
    """Verify _resolve_anchor_skills selects full-corpus medoids when clamp_to_queried=False."""
    from reach.models import Skill
    from reach.queries import Origin, QuerySet, QuerySetProvenance
    from reach.sweep import _resolve_anchor_skills

    skills = [
        Skill(name=f"skill-{i:02d}", description=f"Action {i:02d}", path=tmp_path / f"s-{i:02d}")
        for i in range(5)
    ]
    partial_qs = QuerySet(
        catalog_id="cat",
        queries=(Query(id="q-0", text="Query 0", expected_skill="skill-00"),),
        provenance=QuerySetProvenance(origin=Origin.AUTHORED),
    )
    clamped = _resolve_anchor_skills(
        requested_anchor=3,
        resolved_skills=skills,
        actual_scales=(3, 5),
        query_set=partial_qs,
        clamp_to_queried=True,
    )
    assert clamped == ("skill-00",)

    unclamped = _resolve_anchor_skills(
        requested_anchor=3,
        resolved_skills=skills,
        actual_scales=(3, 5),
        query_set=partial_qs,
        clamp_to_queried=False,
    )
    assert unclamped is not None
    assert len(unclamped) == 3


@pytest.mark.parametrize(
    ("base_config", "cli_attempts", "expected"),
    [
        pytest.param(RunConfig(), None, 1, id="unset-defaults-to-1"),
        pytest.param(
            RunConfig(plan=PlanSettings(attempts=3)),
            None,
            3,
            id="explicit-config-respected",
        ),
        pytest.param(RunConfig(plan=PlanSettings(attempts=3)), 2, 2, id="cli-override-precedence"),
    ],
)
def test_prepare_sweep_config_defaults_attempts_to_1_unless_explicit(
    base_config: RunConfig,
    cli_attempts: int | None,
    expected: int,
) -> None:
    """Verify _prepare_sweep_config defaults attempts=1 unless explicitly set via CLI or config."""
    from reach.sweep import _prepare_sweep_config

    resolved_cfg = _prepare_sweep_config(base_config, attempts=cli_attempts, early_stop=None)
    assert resolved_cfg.plan.attempts == expected


def test_run_scaling_sweep_early_aborts_on_100_percent_runtime_errors_at_first_scale(
    tmp_path: Path,
) -> None:
    """Verify run_scaling_sweep aborts after scale 1 when all probes error out."""
    from reach.runtime import SelectionOutcome

    skills_dir = tmp_path / "skills"
    _create_mock_skills(skills_dir, 6)

    queries = [
        Query(
            id=f"q-{i}",
            text=f"Requesting task number {i:02d}",
            expected_skill=f"skill-{i:02d}",
            kind=QueryKind.IMPLICIT,
        )
        for i in range(6)
    ]
    q_file = tmp_path / "queries.json"
    save_query_set(
        QuerySet(
            catalog_id="synthetic",
            queries=tuple(queries),
            provenance=QuerySetProvenance(origin=Origin.AUTHORED),
        ),
        q_file,
    )

    config = RunConfig(
        study=StudySettings(
            skills=skills_dir,
            queries=q_file,
            workdir=tmp_path / "ws",
            early_stop=True,
        ),
        plan=PlanSettings(retries=0),
        catalog=CatalogSettings(mode=CatalogMode.SWEEP),
    )

    crashing_runtime = FakeRuntime(
        default=SelectionOutcome(error="Provider gemini is not supported"),
        model="mock-model",
    )

    study = run_scaling_sweep(
        config=config,
        target_skill=None,
        scales=(2, 4, 6),
        runtime=crashing_runtime,
    )

    assert len(study.points) == 1
    assert study.points[0].scale == 2
    assert study.points[0].probes_executed == 2
    assert study.points[0].probes_errored == 2
