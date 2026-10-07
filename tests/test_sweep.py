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

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, override

import pytest

from reach.catalog import resolve_sweep_scales
from reach.config import CatalogSettings, PlanSettings, RunConfig, StudySettings
from reach.models import CatalogMode, Query, QueryKind, Skill
from reach.queries import Origin, QuerySet, QuerySetProvenance, save_query_set
from reach.run import RunOutcome
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


@pytest.mark.parametrize(
    ("scales", "rates", "noise_floor", "auto_smooth", "weights", "expected_knee"),
    [
        pytest.param(
            (1, 5, 10, 20, 50, 100),
            (1.0, 0.95, 0.60, 0.55, 0.52, 0.50),
            0.05,
            False,
            None,
            10,
            id="sharp-drop",
        ),
        pytest.param(
            (1, 5, 10, 20),
            (0.95, 0.94, 0.95, 0.93),
            0.05,
            False,
            None,
            None,
            id="flat-within-noise",
        ),
        pytest.param(
            (1, 5),
            (1.0, 0.5),
            0.05,
            False,
            None,
            None,
            id="too-few-points",
        ),
        pytest.param(
            (10, 25, 50, 100, 147),
            (0.96, 0.98, 0.96, 0.70, 0.50),
            0.05,
            True,
            None,
            50,
            id="upward-bump-auto-smooth",
        ),
        pytest.param(
            (10, 25, 50, 100, 147),
            (1.0, 0.95, 0.90, 0.50, 0.20),
            0.05,
            False,
            None,
            50,
            id="monotone-raw",
        ),
        pytest.param(
            (1, 10, 100),
            (0.50, 0.80, 0.95),
            0.10,
            False,
            None,
            None,
            id="rising-curve",
        ),
        pytest.param(
            (1, 10, 100),
            (1.0, 0.749, 0.50),
            0.10,
            False,
            None,
            None,
            id="straight-log-linear-wiggle",
        ),
        pytest.param(
            (1, 10, 100),
            (0.95, 0.92, 0.88),
            0.10,
            False,
            None,
            None,
            id="drop-below-0.10-noise-floor",
        ),
        pytest.param(
            (1, 10, 100),
            (1.0, 0.96, 0.60),
            0.10,
            False,
            None,
            10,
            id="genuine-knee-above-0.10-drop",
        ),
        pytest.param(
            (10, 25, 50, 100),
            (0.95, 0.70, 0.85, 0.50),
            0.05,
            True,
            (1.0, 100.0, 1.0, 1.0),
            25,
            id="weighted-pava-smooth",
        ),
    ],
)
def test_find_kneedle_knee(
    scales: tuple[int, ...],
    rates: tuple[float, ...],
    noise_floor: float,
    auto_smooth: bool,
    weights: tuple[float, ...] | None,
    expected_knee: int | None,
) -> None:
    """Verify log-scale knee curvature detection across raw, smoothed, and weighted curves."""
    assert (
        find_kneedle_knee(
            scales,
            rates,
            noise_floor=noise_floor,
            auto_smooth=auto_smooth,
            weights=weights,
        )
        == expected_knee
    )


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
        PairedTrialOutcomes(n10=0, n01=0, total_paired=-1)


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


def test_compute_scaling_noise_floor_deff_scaling_no_underflow() -> None:
    """Verify McNemar variance scaling by DEFF avoids negative underflow with disparity."""
    from reach.sweep import PairedTrialOutcomes

    # With total_paired = 100, effective_paired = 20, n10 = 25, n01 = 0:
    # (n10 - n01)^2 / n_raw = 625 / 100 = 6.25 < 25 (valid non-negative numerator)
    # If unscaled n_eff were used: 625 / 20 = 31.25 > 25 (collapsed to 0).
    paired = PairedTrialOutcomes(n10=25, n01=0, total_paired=100, effective_paired=20)
    floor = compute_scaling_noise_floor(0.95, 0.70, sample_size=100, paired_outcomes=paired)
    assert floor > 0.01


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
    assert merged.mean == pytest.approx((0.80 * 2.0 + 0.90 * 3.0) / 5.0)


def test_pava_exact_float_merge_associativity() -> None:
    """Verify cascade PAVA merges maintain exact unrounded floating-point arithmetic."""
    from reach.sweep import _merge_pava_blocks, _PavaBlock

    b1 = _PavaBlock(mean=1.0 / 3.0, weight=3.0, size=3)
    b2 = _PavaBlock(mean=2.0 / 3.0, weight=3.0, size=3)
    b3 = _PavaBlock(mean=1.0, weight=6.0, size=6)
    merged_12 = _merge_pava_blocks(b1, b2)
    assert merged_12.mean == 0.5
    assert isinstance(merged_12.mean, float)

    merged_123 = _merge_pava_blocks(merged_12, b3)
    assert merged_123.mean == 0.75
    assert merged_123.weight == 12.0
    assert merged_123.size == 12


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

    for pt in study.points:
        assert pt.recall == 1.0
        assert pt.precision == 1.0
        assert pt.f1_score == 1.0
        assert pt.negative_probes == 0


def test_corpus_scaling_sweep_completes_all_scales_without_early_stopping(tmp_path: Path) -> None:
    """Verify corpus scaling sweep evaluates all scale steps even when F1 drops to 0.0."""
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
    # Evaluates all requested scales through the study without threshold early-stopping
    assert len(study.points) == 5
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


def test_bootstrap_f1_ci(make_probe_result: Callable[..., Any]) -> None:
    """Verify bootstrap_f1_ci computes bounded empirical confidence intervals."""
    from reach.sweep import bootstrap_f1_ci

    results = [
        make_probe_result(query_id="q1", invoked="s1"),
        make_probe_result(query_id="q2", invoked="s2"),
        make_probe_result(query_id="q3", invoked="s1"),  # FP
        make_probe_result(query_id="q4", invoked_skills=()),  # abstention / FN
    ]

    truth: dict[str, str | None] = {"q1": "s1", "q2": "s2", "q3": "s3", "q4": "s4"}
    installed = {"s1", "s2", "s3", "s4"}

    ci_low, ci_high = bootstrap_f1_ci(results, truth, installed, iterations=200, seed=42)
    assert 0.0 <= ci_low <= ci_high <= 1.0


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
        @override
        def select(
            self, query_text: str, workdir: Path, target_skill: str | None = None
        ) -> SelectionOutcome:
            nonlocal select_calls
            select_calls += 1
            return super().select(query_text, workdir, target_skill=target_skill)

    runtime = CountingKeywordRuntime()
    from reach.models import Catalog
    from reach.run import ProbeHarness

    shared_cache: dict[Any, Any] = {}
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
            1.0,
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
    make_probe_result: Callable[..., Any],
) -> None:
    """Verify targeted sweep metrics handle FNs, rival hijacks, and errored probes."""
    from reach.models import Query, QueryKind
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
        make_probe_result(query_id=f"q{i}", invoked=inv)
        for i, inv in enumerate(positive_invocations)
    ]
    neg_results = [
        make_probe_result(query_id=f"neg{j}", invoked_skills=invoked, error=err)
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
    make_probe_result: Callable[..., Any],
) -> None:
    """Verify _resolve_sweep_target_and_queries retains NEIGHBOR_NEGATIVE queries."""
    from reach.models import Query, QueryKind, Skill
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
        make_probe_result(query_id="pos-1", catalog_id="c1", catalog_size=1, invoked="my-skill"),
        make_probe_result(query_id="adv-1", catalog_id="c1", catalog_size=1, invoked_skills=()),
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
        make_probe_result(query_id="pos-1", catalog_id="c", catalog_size=2, invoked="my-skill"),
        make_probe_result(query_id="adv-1", catalog_id="c", catalog_size=2, invoked="my-skill"),
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
    assert point.pass_rate == 1.0
    assert point.recall == 1.0
    assert point.internal_precision == 0.5
    assert point.delta_vs_baseline == 0.0


def test_single_skill_sweep_filters_out_unrelated_neighbor_negatives_from_corpus(
    tmp_path: Path,
) -> None:
    """Verify _resolve_sweep_target_and_queries filters out neighbor negatives for other skills."""
    from reach.models import Query, QueryKind, Skill
    from reach.queries import Origin, QuerySet, QuerySetProvenance
    from reach.sweep import _resolve_sweep_target_and_queries

    skills = [
        Skill(name="cloud-run", description="Cloud Run", path=tmp_path / "cloud-run"),
        Skill(name="bigquery", description="BigQuery", path=tmp_path / "bigquery"),
        Skill(name="spanner", description="Spanner", path=tmp_path / "spanner"),
    ]
    raw_qs = QuerySet(
        catalog_id="c",
        queries=(
            Query(
                id="cr-1",
                text="Deploy container",
                expected_skill="cloud-run",
                kind=QueryKind.IMPLICIT,
            ),
            Query(
                id="adv-cloud-run-1",
                text="Cloud run near miss",
                expected_skill="bigquery",
                kind=QueryKind.NEIGHBOR_NEGATIVE,
            ),
            Query(
                id="adv-bigquery-1",
                text="Bigquery near miss",
                expected_skill="spanner",
                kind=QueryKind.NEIGHBOR_NEGATIVE,
            ),
            Query(
                id="adv-spanner-1",
                text="Spanner near miss",
                expected_skill="bigquery",
                kind=QueryKind.NEIGHBOR_NEGATIVE,
            ),
            Query(
                id="bq-pos",
                text="Run SQL query",
                expected_skill="bigquery",
                kind=QueryKind.IMPLICIT,
            ),
            Query(
                id="span-pos",
                text="Spanner transaction",
                expected_skill="spanner",
                kind=QueryKind.IMPLICIT,
            ),
        ),
        provenance=QuerySetProvenance(origin=Origin.AUTHORED),
    )
    target, filtered_qs = _resolve_sweep_target_and_queries(skills, raw_qs, "cloud-run")
    assert target == "cloud-run"
    assert [q.id for q in filtered_qs.queries] == ["cr-1", "adv-cloud-run-1"]


def test_single_skill_sweep_preserves_target_adversarial_on_prefix_collision(
    tmp_path: Path,
) -> None:
    """Verify target adversarial query is preserved when a rival is a prefix of target."""
    from reach.models import Query, QueryKind, Skill
    from reach.queries import Origin, QuerySet, QuerySetProvenance
    from reach.sweep import _resolve_sweep_target_and_queries

    skills = [
        Skill(name="cloud-run", description="Cloud Run", path=tmp_path / "cloud-run"),
        Skill(
            name="cloud-run-basics",
            description="Cloud Run Basics",
            path=tmp_path / "cloud-run-basics",
        ),
    ]
    raw_qs = QuerySet(
        catalog_id="c",
        queries=(
            Query(
                id="cr-basics-pos",
                text="Deploy container basics",
                expected_skill="cloud-run-basics",
                kind=QueryKind.IMPLICIT,
            ),
            Query(
                id="adv-cloud-run-basics-1",
                text="Cloud run basics near miss",
                expected_skill="cloud-run",
                kind=QueryKind.NEIGHBOR_NEGATIVE,
            ),
            Query(
                id="adv-cloud-run-1",
                text="Cloud run near miss",
                expected_skill="cloud-run-basics",
                kind=QueryKind.NEIGHBOR_NEGATIVE,
            ),
        ),
        provenance=QuerySetProvenance(origin=Origin.AUTHORED),
    )
    target, filtered_qs = _resolve_sweep_target_and_queries(skills, raw_qs, "cloud-run-basics")
    assert target == "cloud-run-basics"
    assert [q.id for q in filtered_qs.queries] == ["cr-basics-pos", "adv-cloud-run-basics-1"]


def test_render_ascii_curve_single_bullet_per_column_on_midpoint_boundaries(
    make_scaling_point: Callable[..., Any],
) -> None:
    """Verify render_ascii_curve maps midpoint boundary values to exactly one row bullet."""
    from reach.views.sweep import render_ascii_curve

    points = [
        make_scaling_point(
            scale=s,
            catalog_id=f"c:{s}",
            pass_rate=val,
            pass_rate_interval=(0.0, 1.0),
            f1_score=val,
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

    def spy_conduct(*args: Any, **kwargs: Any) -> RunOutcome:
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
        step: int, _total: int, _point: ScalingPoint, _partial: ScalingStudy
    ) -> None:
        if step == 2:
            err_msg = "Simulated mid-sweep interruption"
            raise RuntimeError(err_msg)

    def capture_workdir(*args: Any, **kwargs: Any) -> RunOutcome:
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


def test_sweep_scores_two_turn_mutual_handoff_as_true_positive_and_records_entrypoint(
    make_probe_result: Callable[..., Any],
) -> None:
    """Verify 2-turn mutual handoff (mixed_oracle) and acceptable_skills score as TP in sweep."""
    from reach.models import InvocationPattern
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
        make_probe_result(
            query_id="q-handoff",
            catalog_id="sweep:corpus:12",
            invoked="gke-platform-security",
            invocation_pattern=InvocationPattern.ORACLE_ONLY,
            turns_taken=1,
            catalog_size=12,
            model="gemini-3.8-flash",
            runtime="antigravity-sdk",
        ),
        make_probe_result(
            query_id="q-acceptable",
            catalog_id="sweep:corpus:12",
            invoked="agent-platform-deploy",
            invocation_pattern=InvocationPattern.ORACLE_ONLY,
            turns_taken=1,
            catalog_size=12,
            model="gemini-3.8-flash",
            runtime="antigravity-sdk",
        ),
    )
    scaled_results = (
        make_probe_result(
            query_id="q-handoff",
            catalog_id="sweep:corpus:128",
            invoked_skills=("gke-basics", "gke-platform-security"),
            invocation_pattern=InvocationPattern.MIXED_ORACLE,
            turns_taken=2,
            catalog_size=128,
            model="gemini-3.8-flash",
            runtime="antigravity-sdk",
        ),
        make_probe_result(
            query_id="q-acceptable",
            catalog_id="sweep:corpus:128",
            invoked_skills=("gcloud", "agent-platform-deploy"),
            invocation_pattern=InvocationPattern.ORACLE_ONLY,
            turns_taken=2,
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
        @override
        def rations_catalog(self) -> bool:
            return True

        @override
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

    # When initial scale (4) exceeds queried skill count (2), avoids silent demotion:
    # selects full corpus medoids (4) and logs warning so missing anchors can be auto-drafted.
    full_anchors = _resolve_anchor_skills(None, skills, (4, 5), query_set=partial_qs)
    assert full_anchors is not None
    assert len(full_anchors) == 4

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


def test_render_ascii_curve_auto_scales_high_accuracy_band(
    make_scaling_point: Callable[..., Any],
) -> None:
    """Verify render_ascii_curve zooms into [80%..100%] when all points are >= 75%."""
    from reach.views.sweep import render_ascii_curve

    def _pt(scale: int, f1: float) -> Any:
        return make_scaling_point(
            scale=scale,
            catalog_id=f"c{scale}",
            pass_rate=f1,
            pass_rate_interval=(f1 - 0.05, min(1.0, f1 + 0.05)),
            f1_score=f1,
            delta_vs_baseline=0.975 - f1,
            delta_collision=0.975 - f1,
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


def test_print_sweep_surfaces_collision_and_truncation_without_ellipsis(
    make_scaling_point: Callable[..., Any],
    make_scaling_study: Callable[..., Any],
) -> None:
    """Verify print_sweep renders Δ Collide, Truncated, and Loss Decomposition within 80 columns."""
    from io import StringIO

    from rich.console import Console

    from reach.views.sweep import print_sweep

    study = make_scaling_study(
        anchor_skills=("a1", "a2"),
        scales=(10, 100, 147),
        knee_scale=25,
        baseline_pass_rate=0.951,
        final_pass_rate=0.854,
        total_delta=0.097,
        total_abstention_loss=0.024,
        total_collision_loss=0.098,
        total_corpus_skills=147,
        points=(
            make_scaling_point(
                scale=10,
                catalog_id="s10",
                pass_rate=0.951,
                pass_rate_interval=(0.88, 0.99),
                recall=0.951,
                precision=1.0,
                f1_score=0.975,
                f1_interval=(0.935, 1.0),
                in_scope_probes=41,
                probes_executed=41,
                duration_ms_mean=3393.0,
                disclosure_states={"full": 41},
            ),
            make_scaling_point(
                scale=100,
                catalog_id="s100",
                pass_rate=0.902,
                pass_rate_interval=(0.80, 0.96),
                recall=0.902,
                precision=0.925,
                f1_score=0.914,
                f1_interval=(0.825, 0.988),
                delta_vs_baseline=0.049,
                delta_collision=0.073,
                in_scope_probes=41,
                probes_executed=41,
                duration_ms_mean=4472.0,
                disclosure_states={"full": 20, "name_only_elided": 21},
            ),
            make_scaling_point(
                scale=147,
                catalog_id="s147",
                pass_rate=0.854,
                pass_rate_interval=(0.74, 0.93),
                recall=0.854,
                precision=0.897,
                f1_score=0.875,
                f1_interval=(0.769, 0.975),
                delta_vs_baseline=0.097,
                delta_collision=0.098,
                delta_abstention=0.024,
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
    assert "Loss Decomposition (K=10→147, +9.7% pass-rate drop): " in out
    assert "Δ Collision +9.8%" in out
    assert "Δ Abstention +2.4%" in out
    assert "Δ Collide" in out
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
    """Verify _resolve_anchor_skills selects full medoids when queries are insufficient."""
    from reach.models import Skill
    from reach.queries import Origin, QuerySet, QuerySetProvenance
    from reach.sweep import _resolve_anchor_skills

    skills = [
        Skill(name=f"skill-{i:02d}", description=f"Action {i:02d}", path=tmp_path / f"s-{i:02d}")
        for i in range(5)
    ]
    # 4 queried skills out of 5: enough to select 3 anchors when clamped
    sufficient_qs = QuerySet(
        catalog_id="cat",
        queries=tuple(
            Query(id=f"q-{i}", text=f"Query {i}", expected_skill=f"skill-{i:02d}") for i in range(4)
        ),
        provenance=QuerySetProvenance(origin=Origin.AUTHORED),
    )
    clamped = _resolve_anchor_skills(
        requested_anchor=3,
        resolved_skills=skills,
        actual_scales=(3, 5),
        query_set=sufficient_qs,
        clamp_to_queried=True,
    )
    assert clamped is not None
    assert len(clamped) == 3
    assert set(clamped).issubset({f"skill-{i:02d}" for i in range(4)})

    unclamped = _resolve_anchor_skills(
        requested_anchor=3,
        resolved_skills=skills,
        actual_scales=(3, 5),
        query_set=sufficient_qs,
        clamp_to_queried=False,
    )
    assert unclamped is not None
    assert len(unclamped) == 3

    # When queried skills (1) < requested anchors (3), avoids demotion even when clamped
    sparse_qs = QuerySet(
        catalog_id="cat",
        queries=(Query(id="q-0", text="Query 0", expected_skill="skill-00"),),
        provenance=QuerySetProvenance(origin=Origin.AUTHORED),
    )
    fallback_anchors = _resolve_anchor_skills(
        requested_anchor=3,
        resolved_skills=skills,
        actual_scales=(3, 5),
        query_set=sparse_qs,
        clamp_to_queried=True,
    )
    assert fallback_anchors is not None
    assert len(fallback_anchors) == 3


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

    resolved_cfg = _prepare_sweep_config(base_config, attempts=cli_attempts)
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


def test_bootstrap_f1_ci_clusters_by_query_maintaining_correlated_attempts(
    make_probe_result: Callable[..., Any],
) -> None:
    """Verify cluster bootstrap preserves query attempt correlation without false shrinkage."""
    from reach.sweep import bootstrap_f1_ci

    # 4 distinct queries, 2 correct and 2 incorrect
    truth = {"q1": "s1", "q2": "s2", "q3": "s3", "q4": "s4"}
    installed = {"s1", "s2", "s3", "s4"}

    single_results = [
        make_probe_result(query_id="q1", catalog_id="c", invoked="s1", catalog_size=4, attempt=1),
        make_probe_result(query_id="q2", catalog_id="c", invoked="s2", catalog_size=4, attempt=1),
        make_probe_result(
            query_id="q3", catalog_id="c", invoked="s1", catalog_size=4, attempt=1
        ),  # FP
        make_probe_result(
            query_id="q4", catalog_id="c", invoked_skills=(), catalog_size=4, attempt=1
        ),  # FN
    ]
    ci_single_low, ci_single_high = bootstrap_f1_ci(
        single_results, truth, installed, iterations=1000, seed=42
    )
    single_width = ci_single_high - ci_single_low

    # Repeat each query across 5 deterministic attempts (identical outcomes)
    repeated_results = [
        make_probe_result(
            query_id=r.query_id,
            catalog_id="c",
            invoked_skills=r.invoked_skills,
            catalog_size=4,
            attempt=att,
        )
        for att in range(1, 6)
        for r in single_results
    ]

    ci_rep_low, ci_rep_high = bootstrap_f1_ci(
        repeated_results, truth, installed, iterations=1000, seed=42
    )
    rep_width = ci_rep_high - ci_rep_low

    # Cluster bootstrap avoids artificially shrinking interval width on duplicate attempts
    assert abs(rep_width - single_width) < 0.05
    assert ci_rep_low == pytest.approx(ci_single_low, abs=0.05)
    assert ci_rep_high == pytest.approx(ci_single_high, abs=0.05)


def test_scaling_point_records_step_efficiency_and_skill_f1(
    make_probe_result: Callable[..., Any],
) -> None:
    """Verify _build_scaling_point calculates step_efficiency_mean and skill_f1_mean."""
    from reach.models import Query, QueryKind
    from reach.queries import Origin, QuerySet, QuerySetProvenance
    from reach.sweep import _build_scaling_point

    queries = [
        Query(id="q1", text="Do thing 1", expected_skill="s1", kind=QueryKind.IMPLICIT),
        Query(id="q2", text="Do thing 2", expected_skill="s2", kind=QueryKind.IMPLICIT),
    ]
    query_set = QuerySet(
        catalog_id="c",
        queries=tuple(queries),
        provenance=QuerySetProvenance(origin=Origin.AUTHORED),
    )

    results = (
        # First-rank hit: step_efficiency = 1.0, skill_f1 = 1.0
        make_probe_result(query_id="q1", catalog_id="c", invoked="s1", catalog_size=2),
        # Second-rank hit: step_efficiency = 0.5, skill_f1 = 0.6667
        make_probe_result(
            query_id="q2", catalog_id="c", invoked_skills=("distractor", "s2"), catalog_size=2
        ),
    )

    point, _ = _build_scaling_point(
        scale=2,
        catalog_id="c",
        results=results,
        resolved_query_set=query_set,
        baseline_results=(),
        installed_skills={"s1", "s2", "distractor"},
    )

    # Mean step efficiency = (1.0 + 0.5) / 2 = 0.75
    assert point.step_efficiency_mean == pytest.approx(0.75, abs=0.01)
    assert 0.0 < point.skill_f1_mean <= 1.0


@pytest.fixture
def sample_four_scale_points(make_scaling_point: Callable[..., Any]) -> list[Any]:
    """Provide a 4-scale ScalingPoint sequence for knee interval tests."""
    return [
        make_scaling_point(
            scale=scale,
            catalog_id="c",
            pass_rate=rate,
            pass_rate_interval=ci,
            f1_score=rate,
            f1_interval=ci,
            delta_vs_baseline=round(0.96 - rate, 2),
            delta_collision=round(0.96 - rate, 2),
            probes_executed=20,
        )
        for scale, rate, ci in (
            (10, 0.96, (0.92, 0.99)),
            (25, 0.91, (0.85, 0.96)),
            (50, 0.82, (0.74, 0.89)),
            (100, 0.60, (0.50, 0.70)),
        )
    ]


def test_scaling_study_records_knee_interval(sample_four_scale_points: list[Any]) -> None:
    """Verify _assemble_scaling_study computes uncertainty interval for knee scale."""
    from reach.sweep import _assemble_scaling_study, _QueryOutcome

    scales = (10, 25, 50, 100)
    scale_query_sums = {
        s: {
            f"q{i}": (
                _QueryOutcome(1, 0, 0, 1, 1)
                if (s <= 25 or i % (s // 25) == 0)
                else _QueryOutcome(0, 1, 1, 0, 1)
            )
            for i in range(20)
        }
        for s in scales
    }
    study = _assemble_scaling_study(
        target=None,
        is_corpus=True,
        actual_scales=scales,
        points=sample_four_scale_points,
        noise_floor=0.05,
        baseline_count=20,
        total_skills=100,
        decomp=None,
        anchor_skills=("s1", "s2"),
        scale_query_sums=scale_query_sums,
    )

    assert study.knee_scale is not None
    assert study.knee_scale_interval is not None
    assert study.knee_scale_interval[0] <= study.knee_scale_interval[1]


def test_bootstrap_knee_interval_cluster_resampling(sample_four_scale_points: list[Any]) -> None:
    """Verify _bootstrap_knee_summary calculates interval via non-parametric cluster bootstrap."""
    from reach.sweep import _bootstrap_knee_summary, _QueryOutcome

    scales = [10, 25, 50, 100]
    scale_query_sums = {
        s: {
            f"q{i}": (
                _QueryOutcome(1, 0, 0, 1, 1)
                if (s <= 25 or i % (s // 25) == 0)
                else _QueryOutcome(0, 1, 1, 0, 1)
            )
            for i in range(20)
        }
        for s in scales
    }

    interval = _bootstrap_knee_summary(
        scales=scales,
        points=sample_four_scale_points,
        noise_floor=0.05,
        iterations=50,
        seed=42,
        scale_query_sums=scale_query_sums,
        raw_noise_floor=0.05,
    ).interval
    assert interval is not None
    assert interval[0] <= interval[1]


def test_render_sweep_csv_includes_knee_and_efficiency_metrics(
    make_scaling_point: Callable[..., Any],
    make_scaling_study: Callable[..., Any],
) -> None:
    """Verify render_sweep_csv includes knee uncertainty, right-censoring, and step efficiency."""
    from reach.views.sweep import render_sweep_csv

    study = make_scaling_study(
        knee_scale=25,
        knee_scale_interval=(20, 30),
        points=(
            make_scaling_point(
                scale=10,
                catalog_id="c1",
                pass_rate=0.95,
                pass_rate_interval=(0.90, 0.98),
                recall=0.95,
                precision=0.95,
                f1_score=0.95,
                step_efficiency_mean=0.90,
                skill_f1_mean=0.92,
            ),
        ),
    )
    csv_text = render_sweep_csv(study)
    assert "knee_scale" in csv_text
    assert "knee_ci_low" in csv_text
    assert "knee_ci_high" in csv_text
    assert "step_efficiency" in csv_text
    assert "skill_f1" in csv_text
    assert "25" in csv_text
    assert "20" in csv_text
    assert "30" in csv_text

    censored_study = make_scaling_study(
        scales=(10, 25, 50),
        knee_scale=25,
        knee_scale_interval=(10, 50),
        knee_upper_censored=True,
        points=(
            make_scaling_point(
                scale=10,
                catalog_id="c1",
                pass_rate=0.95,
                pass_rate_interval=(0.90, 0.98),
            ),
        ),
    )
    censored_csv = render_sweep_csv(censored_study)
    assert ",10,>50" in censored_csv


def test_paired_trial_outcomes_calculates_effective_paired(
    make_probe_result: Callable[..., Any],
) -> None:
    """Verify _calculate_paired_outcomes adjusts effective_paired for repeated attempts."""
    from reach.models import Query, QueryKind
    from reach.sweep import _calculate_paired_outcomes

    queries = {
        "q1": Query(id="q1", text="t1", expected_skill="s1", kind=QueryKind.IMPLICIT),
        "q2": Query(id="q2", text="t2", expected_skill="s2", kind=QueryKind.IMPLICIT),
    }

    # 2 queries, 5 attempts each (10 total probes)
    base_results = [
        make_probe_result(query_id="q1", catalog_id="c1", invoked="s1", catalog_size=2, attempt=att)
        for att in range(1, 6)
    ] + [
        make_probe_result(query_id="q2", catalog_id="c1", invoked="s2", catalog_size=2, attempt=att)
        for att in range(1, 6)
    ]

    final_results = [
        make_probe_result(
            query_id="q1", catalog_id="c2", invoked="s1", catalog_size=10, attempt=att
        )
        for att in range(1, 6)
    ] + [
        make_probe_result(
            query_id="q2", catalog_id="c2", invoked_skills=(), catalog_size=10, attempt=att
        )  # fails in final
        for att in range(1, 6)
    ]

    paired = _calculate_paired_outcomes(base_results, final_results, queries)
    assert paired is not None
    assert paired.total_paired == 10
    # Effective sample size adjusts 10 probes across 5 attempts to ~3
    assert paired.effective_paired is not None
    assert paired.effective_paired < paired.total_paired
    assert paired.effective_paired >= 2


@pytest.mark.parametrize(
    ("target_skill", "is_corpus_sweep"),
    [
        (None, True),
        ("skill-a", False),
    ],
    ids=["corpus-sweep", "single-skill-sweep"],
)
def test_sweep_hints_when_fewer_than_three_scales(
    sample_scaling_study: Any,
    target_skill: str | None,
    is_corpus_sweep: bool,
) -> None:
    """Verify sweep displays knee detection hint when evaluated with fewer than 3 scales."""
    from rich.console import Console

    from reach.views.sweep import print_sweep

    study = sample_scaling_study.model_copy(
        update={"target_skill": target_skill, "is_corpus_sweep": is_corpus_sweep}
    )
    console = Console(record=True)
    print_sweep(console, study)
    text = console.export_text()
    assert "requires ≥ 3 scale steps to detect" in text
    assert "evaluated" in text


def test_sweep_steepest_drop_and_truncation_loss_rendering(
    make_scaling_point: Callable[..., Any],
    make_scaling_study: Callable[..., Any],
) -> None:
    """Verify ScalingStudy steepest drop interval and truncation loss render in sweep views."""
    from rich.console import Console

    from reach.sweep import _find_steepest_drop
    from reach.views.sweep import print_sweep

    assert _find_steepest_drop((10, 25, 50), (0.95, 0.70, 0.68)) == ((10, 25), 0.25)
    assert _find_steepest_drop((10, 25), (0.80, 0.85)) == (None, None)

    pt1 = make_scaling_point(
        scale=10,
        catalog_id="c10",
        pass_rate=0.95,
        pass_rate_interval=(0.85, 0.99),
        f1_score=0.95,
        probes_executed=20,
    )
    pt2 = make_scaling_point(
        scale=25,
        catalog_id="c25",
        pass_rate=0.70,
        pass_rate_interval=(0.55, 0.82),
        f1_score=0.70,
        delta_vs_baseline=0.25,
        delta_abstention=0.10,
        delta_collision=0.15,
        delta_truncated=0.10,
        probes_executed=20,
    )
    pt3 = make_scaling_point(
        scale=50,
        catalog_id="c50",
        pass_rate=0.68,
        pass_rate_interval=(0.52, 0.80),
        f1_score=0.68,
        delta_vs_baseline=0.27,
        delta_abstention=0.12,
        delta_collision=0.15,
        delta_truncated=0.10,
        probes_executed=20,
    )

    corpus_study = make_scaling_study(
        points=(pt1, pt2, pt3),
        knee_scale=25,
        steepest_drop_scales=(10, 25),
        steepest_drop_delta=0.25,
        total_abstention_loss=0.12,
        total_collision_loss=0.15,
        total_truncated_loss=0.10,
    )
    console = Console(record=True, width=100)
    print_sweep(console, corpus_study)
    corpus_text = console.export_text()
    assert "Steepest Drop Interval: K=10→25 (-25.0% F1)" in corpus_text
    assert "Loss Decomposition (K=10→50, +27.0% pass-rate drop): " in corpus_text
    assert "Δ Truncation +10.0%" in corpus_text
    assert "Budget Truncation Loss" not in corpus_text

    from reach.views.sweep import render_sweep_csv

    single_study = corpus_study.model_copy(
        update={"is_corpus_sweep": False, "target_skill": "skill-a"}
    )
    console_single = Console(record=True, width=100)
    print_sweep(console_single, single_study)
    single_text = console_single.export_text()
    assert "elbow threshold: stabilizes after N=10→25 drop of 25.0%" in single_text
    assert "Budget Truncation Loss: +10.0%" in single_text
    assert "Δ Trunc" in single_text

    cliff_study = single_study.model_copy(
        update={"knee_scale": 10, "steepest_drop_scales": (10, 25)}
    )
    console_cliff = Console(record=True, width=100)
    print_sweep(console_cliff, cliff_study)
    cliff_text = console_cliff.export_text()
    assert "capacity cliff where skill collisions accelerate" in cliff_text

    csv_single = render_sweep_csv(single_study)
    assert "delta_truncated" in csv_single
    assert "delta_collision" in csv_single
    assert "delta_abstention" in csv_single
    assert "0.1000" in csv_single


def test_scaling_point_pydantic_validation_invariants(
    make_scaling_point: Callable[..., Any],
) -> None:
    """Verify ScalingPoint enforces NonNegativeInt and probe accounting invariants."""
    from pydantic import ValidationError

    # Happy path: valid probe accounting
    pt = make_scaling_point(probes_executed=10, probes_errored=2, probes_failed=3)
    assert pt.probes_executed == 10
    assert pt.probes_errored == 2
    assert pt.probes_failed == 3
    assert not pt.all_probes_errored

    # Happy path: all errored triggers all_probes_errored
    all_err_pt = make_scaling_point(probes_executed=5, probes_errored=5, probes_failed=0)
    assert all_err_pt.all_probes_errored

    # Property: all_probes_errored is excluded from serialized dict
    assert "all_probes_errored" not in pt.model_dump()

    # Sad path: negative probe counts rejected by NonNegativeInt
    with pytest.raises(ValidationError):
        make_scaling_point(probes_executed=-1)
    with pytest.raises(ValidationError):
        make_scaling_point(probes_failed=-1)
    with pytest.raises(ValidationError):
        make_scaling_point(probes_errored=-1)

    # Sad path: probes_errored > probes_executed
    with pytest.raises(ValidationError, match="cannot exceed probes_executed"):
        make_scaling_point(probes_executed=5, probes_errored=6)

    # Sad path: probes_failed > valid non-errored probes
    with pytest.raises(ValidationError, match="cannot exceed valid non-errored probes"):
        make_scaling_point(probes_executed=10, probes_errored=4, probes_failed=7)


def test_calculate_scale_pass_rate_excludes_runtime_errors(
    make_probe_result: Callable[..., Any],
) -> None:
    """Verify runtime errors are excluded from pass_rate denominator and fail counts."""
    from reach.models import Query
    from reach.sweep import _calculate_scale_pass_rate

    queries = {
        "q1": Query(id="q1", text="run 1", expected_skill="s1"),
        "q2": Query(id="q2", text="run 2", expected_skill="s1"),
        "q3": Query(id="q3", text="run 3", expected_skill="s1"),
        "q4": Query(id="q4", text="run 4", expected_skill="s1"),
    }
    results = [
        make_probe_result(query_id="q1", catalog_id="c", invoked="s1"),
        make_probe_result(query_id="q2", catalog_id="c", invoked="s1"),
        make_probe_result(query_id="q3", catalog_id="c", invoked_skills=()),
        make_probe_result(query_id="q4", catalog_id="c", invoked_skills=(), error="HTTP 429"),
    ]

    # Valid results: q1 (hit), q2 (hit), q3 (fail). q4 is errored.
    stats = _calculate_scale_pass_rate(results, queries)
    assert stats.executed == 3
    assert stats.fails == 1
    assert stats.pass_rate == pytest.approx(2 / 3)


def test_calculate_scale_classification_excludes_runtime_errors(
    make_probe_result: Callable[..., Any],
) -> None:
    """Verify classification metrics ignore runtime errors in precision, recall, and abstention."""
    from reach.models import Query, QueryKind
    from reach.sweep import _calculate_scale_classification

    queries = {
        "pos1": Query(id="pos1", text="run pos", expected_skill="s1"),
        "pos_err": Query(id="pos_err", text="run pos err", expected_skill="s1"),
        "neg1": Query(id="neg1", text="run neg", expected_skill=None, kind=QueryKind.OUT_OF_SCOPE),
        "neg_err": Query(
            id="neg_err", text="run neg err", expected_skill=None, kind=QueryKind.OUT_OF_SCOPE
        ),
    }
    results = [
        make_probe_result(query_id="pos1", catalog_id="c", invoked="s1"),
        make_probe_result(query_id="pos_err", catalog_id="c", invoked_skills=(), error="timeout"),
        make_probe_result(query_id="neg1", catalog_id="c", invoked_skills=()),
        make_probe_result(query_id="neg_err", catalog_id="c", invoked_skills=(), error="timeout"),
    ]

    metrics = _calculate_scale_classification(results, queries, installed_skills={"s1"}, seed=42)
    assert metrics.in_scope_probes == 1
    assert metrics.negative_probes == 1
    assert metrics.recall == 1.0
    assert metrics.abstention_rate == 1.0


def test_single_skill_trajectory_metrics_not_inverted_by_adversarial_queries(
    make_probe_result: Callable[..., Any],
) -> None:
    """Verify single-skill trajectory metrics are evaluated only on target positive queries."""
    from reach.models import Query, QueryKind
    from reach.sweep import _calculate_scale_classification

    queries = {
        "target_q": Query(id="target_q", text="target query", expected_skill="cloud-deploy"),
        "rival_adv_q": Query(
            id="rival_adv_q",
            text="rival near miss",
            expected_skill="container-build",
            kind=QueryKind.NEIGHBOR_NEGATIVE,
        ),
    }
    # At K=10, container-build is installed. The agent reaches container-build for rival_adv_q.
    # For target_q, the agent reached cloud-deploy with 1 step.
    results = [
        make_probe_result(query_id="target_q", catalog_id="c", invoked="cloud-deploy"),
        make_probe_result(query_id="rival_adv_q", catalog_id="c", invoked="container-build"),
    ]

    metrics = _calculate_scale_classification(
        results,
        queries,
        installed_skills={"cloud-deploy", "container-build"},
        seed=42,
        target_skill="cloud-deploy",
    )
    # Target efficiency should be 1.0 (from target_q), not diluted by rival query
    assert metrics.step_efficiency_mean == 1.0
    assert metrics.skill_f1_mean == 1.0


def test_pava_weights_zero_variance_receives_maximum_weight(
    make_scaling_point: Callable[..., Any],
) -> None:
    """Verify points with zero variance receive weight 10,000 rather than falling back to 1.0."""
    from reach.diff import DEFAULT_CONFIDENCE
    from reach.sweep import _compute_pava_weights
    from reach.uncertainty import ci_span_sigmas

    pt_unanimous = make_scaling_point(scale=10, f1_score=1.0, f1_interval=(1.0, 1.0))
    pt_noisy = make_scaling_point(scale=25, f1_score=0.95, f1_interval=(0.90, 1.00))

    ci_span = ci_span_sigmas(DEFAULT_CONFIDENCE)
    active_intervals = [pt_unanimous.f1_interval, pt_noisy.f1_interval]
    weights = _compute_pava_weights(active_intervals, ci_span)

    # Zero-variance interval receives 1 / 1e-4 = 10,000
    assert weights[0] == 10000.0
    # Noisy interval (span 0.10) receives ~1,538
    assert weights[1] < weights[0]
    assert 1400.0 < weights[1] < 1600.0

    # Defensive check: ci_span <= 0 raises ValueError
    with pytest.raises(ValueError, match="ci_span must be strictly positive"):
        _compute_pava_weights([(0.9, 1.0)], 0.0)

    # Defensive check: inverted interval clamps negative width to 0.0
    inverted_weights = _compute_pava_weights([(1.0, 0.9)], ci_span)
    assert inverted_weights[0] == 10000.0

    # Defensive check: min_variance=0 clamps to 1e-12 without division by zero
    zero_min_var_weights = _compute_pava_weights([(1.0, 1.0)], ci_span, min_variance=0.0)
    assert zero_min_var_weights[0] == 1e12


def test_compute_effective_noise_floor_uses_rate_curve_in_corpus_mode(
    make_scaling_point: Callable[..., Any],
) -> None:
    """Verify _compute_effective_noise_floor uses rate_curve values when provided."""
    from reach.sweep import _compute_effective_noise_floor

    # Points with identical pass rate (1.0) but dropping F1 (1.0 -> 0.5)
    p1 = make_scaling_point(scale=10, pass_rate=1.0, f1_score=1.0)
    p2 = make_scaling_point(scale=50, pass_rate=1.0, f1_score=0.5)

    floor_without_curve = _compute_effective_noise_floor(None, [p1, p2], baseline_count=100)
    floor_with_f1_curve = _compute_effective_noise_floor(
        None, [p1, p2], baseline_count=100, rate_curve=[1.0, 0.5]
    )

    # Identical pass rates yield minimum noise floor (0.01)
    assert floor_without_curve == 0.01
    # F1 drop from 1.0 to 0.5 yields a higher noise floor reflecting binomial SE of the drop
    assert floor_with_f1_curve > floor_without_curve


def test_bootstrap_knee_interval_guards_against_conditioning_bias(
    make_scaling_point: Callable[..., Any],
) -> None:
    """Verify bootstrap interval returns None if knee is detected in <50% of iterations."""
    from reach.sweep import _bootstrap_knee_summary, _QueryOutcome

    # Flat line: 1.0, 1.0, 1.0 (no true knee)
    p1 = make_scaling_point(scale=10, pass_rate=1.0, pass_rate_interval=(0.95, 1.0))
    p2 = make_scaling_point(scale=25, pass_rate=1.0, pass_rate_interval=(0.95, 1.0))
    p3 = make_scaling_point(scale=50, pass_rate=1.0, pass_rate_interval=(0.95, 1.0))
    scale_query_sums = {
        s: {f"q{i}": _QueryOutcome(1, 0, 0, 1, 1) for i in range(20)} for s in (10, 25, 50)
    }

    knee_ci = _bootstrap_knee_summary(
        scales=(10, 25, 50),
        points=[p1, p2, p3],
        noise_floor=0.05,
        scale_query_sums=scale_query_sums,
        iterations=50,
        seed=42,
        is_corpus=False,
        raw_noise_floor=0.05,
    ).interval
    assert knee_ci is None


def test_single_skill_bootstrap_knee_ci_populated(
    make_scaling_point: Callable[..., Any],
) -> None:
    """Verify single-skill sweeps calculate a valid bootstrap knee interval for clear drops."""
    from reach.sweep import _bootstrap_knee_summary, _QueryOutcome

    # Clear knee at scale 25: 1.0 -> 0.95 -> 0.40
    p1 = make_scaling_point(scale=10, pass_rate=1.0, pass_rate_interval=(0.98, 1.0))
    p2 = make_scaling_point(scale=25, pass_rate=0.95, pass_rate_interval=(0.90, 0.98))
    p3 = make_scaling_point(scale=50, pass_rate=0.40, pass_rate_interval=(0.35, 0.45))
    scale_query_sums = {
        10: {f"q{i}": _QueryOutcome(1, 0, 0, 1, 1) for i in range(20)},
        25: {
            f"q{i}": (_QueryOutcome(1, 0, 0, 1, 1) if i < 19 else _QueryOutcome(0, 0, 1, 0, 1))
            for i in range(20)
        },
        50: {
            f"q{i}": (_QueryOutcome(1, 0, 0, 1, 1) if i < 8 else _QueryOutcome(0, 0, 1, 0, 1))
            for i in range(20)
        },
    }

    knee_ci = _bootstrap_knee_summary(
        scales=(10, 25, 50),
        points=[p1, p2, p3],
        noise_floor=0.05,
        scale_query_sums=scale_query_sums,
        iterations=100,
        seed=42,
        is_corpus=False,
        raw_noise_floor=0.05,
    ).interval
    assert knee_ci is not None
    assert isinstance(knee_ci, tuple)
    assert len(knee_ci) == 2
    assert knee_ci[0] <= knee_ci[1]


def test_resolve_medoid_anchors_avoids_demotion_on_partial_queries(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Verify partial query sets do not demote medoid cohort to a single skill."""
    from reach.models import Skill
    from reach.queries import Origin, QuerySet, QuerySetProvenance
    from reach.sweep import _resolve_anchor_skills

    skills = [
        Skill(
            name=f"skill-{i:02d}",
            description=f"Skill {i:02d} workflow",
            path=tmp_path / f"s-{i:02d}",
        )
        for i in range(8)
    ]
    # Only 1 skill has queries in pre-existing query set
    partial_qs = QuerySet(
        catalog_id="partial",
        queries=(Query(id="q-0", text="Task 0", expected_skill="skill-00"),),
        provenance=QuerySetProvenance(origin=Origin.AUTHORED),
    )
    with caplog.at_level("WARNING"):
        anchors = _resolve_anchor_skills(
            requested_anchor=4,
            resolved_skills=skills,
            actual_scales=(4, 8),
            query_set=partial_qs,
            clamp_to_queried=True,
        )
    assert anchors is not None
    # Must NOT be demoted to len 1!
    assert len(anchors) == 4
    assert (
        "Existing query set covers 1 skill(s), which is less than requested anchor count K0=4"
        in caplog.text
    )


def test_resolve_medoid_anchors_errors_on_no_auto_queries_insufficient(
    tmp_path: Path,
) -> None:
    """Verify ValueError is raised when auto_queries=False and existing queries are insufficient."""
    from reach.config import RunConfig, StudySettings
    from reach.models import Skill
    from reach.queries import Origin, QuerySet, QuerySetProvenance
    from reach.sweep import _setup_sweep_execution

    skills = [
        Skill(
            name=f"skill-{i:02d}",
            description=f"Skill {i:02d} workflow",
            path=tmp_path / f"s-{i:02d}",
        )
        for i in range(8)
    ]
    partial_qs = QuerySet(
        catalog_id="partial",
        queries=(Query(id="q-0", text="Task 0", expected_skill="skill-00"),),
        provenance=QuerySetProvenance(origin=Origin.AUTHORED),
    )
    config = RunConfig(study=StudySettings(auto_queries=False))

    with pytest.raises(
        ValueError,
        match=r"--no-auto-queries was specified, but existing query set only covers 1 skill\(s\)",
    ):
        _setup_sweep_execution(
            is_corpus=True,
            target_skill=None,
            anchor=None,
            resolved_skills=skills,
            actual_scales=(4, 8),
            raw_query_set=partial_qs,
            effective_config=config,
            rivals_share=0.5,
        )


def test_all_probes_errored_property_and_sweep_abort(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_probe_result: Callable[..., Any],
) -> None:
    """Verify sweep aborts after scale 1 if all probes errored."""
    from reach.config import RunConfig, StudySettings
    from reach.models import Skill
    from reach.queries import Origin, Query, QueryKind, QuerySet, QuerySetProvenance
    from reach.sweep import run_scaling_sweep

    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    _create_mock_skills(skills_dir, 4)
    skills = [
        Skill(
            name=f"skill-{i:02d}",
            description=f"Skill {i:02d}",
            path=skills_dir / f"skill-{i:02d}",
        )
        for i in range(4)
    ]
    qs = QuerySet(
        catalog_id="all",
        queries=(
            Query(id="q1", text="q1 text", kind=QueryKind.IMPLICIT, expected_skill="skill-00"),
            Query(id="q2", text="q2 text", kind=QueryKind.IMPLICIT, expected_skill="skill-01"),
        ),
        provenance=QuerySetProvenance(origin=Origin.AUTHORED),
    )

    scale_calls: list[int] = []

    def mock_conduct(**kwargs: Any) -> Any:
        scale_calls.append(len(kwargs["composed"].catalog.skills))
        return type(
            "MockOutcome",
            (),
            {
                "results": [
                    make_probe_result(
                        query_id="q1",
                        error="Execution timed out",
                    ),
                    make_probe_result(
                        query_id="q2",
                        error="Runtime error",
                    ),
                ],
            },
        )()

    monkeypatch.setattr("reach.sweep.conduct", mock_conduct)

    study = run_scaling_sweep(
        skills=skills,
        scales=(2, 4),
        query_set=qs,
        config=RunConfig(study=StudySettings(auto_queries=False, workdir=tmp_path / "work")),
    )

    # Sweep must abort early at scale 1 (first scale)
    assert len(scale_calls) == 1
    assert len(study.points) == 1
    assert study.points[0].all_probes_errored
    assert study.points[0].probes_errored == 2
    assert study.points[0].probes_executed == 2


def test_single_skill_pass_rate_and_decomposition_aligned(
    make_probe_result: Callable[..., Any],
) -> None:
    """Verify single-skill pass rate and decomposition both score target and out-of-scope probes."""
    from reach.models import Query, QueryKind
    from reach.sweep import (
        _calculate_scale_pass_rate,
        _evaluate_baseline_decomposition,
    )

    queries = [
        Query(
            id="q_target",
            text="target query",
            kind=QueryKind.IMPLICIT,
            expected_skill="target-skill",
        ),
        Query(
            id="q_distractor",
            text="distractor query",
            kind=QueryKind.IMPLICIT,
            expected_skill="other-skill",
        ),
        Query(
            id="q_oos",
            text="out of scope query",
            kind=QueryKind.OUT_OF_SCOPE,
            expected_skill=None,
        ),
    ]
    queries_by_id = {q.id: q for q in queries}

    baseline_results = [
        make_probe_result(query_id="q_target", invoked="target-skill"),
        make_probe_result(query_id="q_distractor", invoked="other-skill"),
        make_probe_result(query_id="q_oos", invoked=()),
    ]

    scaled_results = [
        make_probe_result(query_id="q_target", invoked="other-skill"),
        make_probe_result(query_id="q_distractor", invoked="target-skill"),
        make_probe_result(query_id="q_oos", invoked="target-skill"),
    ]

    # 1. _calculate_scale_pass_rate:
    base_rate = _calculate_scale_pass_rate(
        baseline_results,
        queries_by_id,
        target_skill="target-skill",
    )
    assert base_rate.executed == 2
    assert base_rate.pass_rate == 1.0

    scale_rate = _calculate_scale_pass_rate(
        scaled_results,
        queries_by_id,
        target_skill="target-skill",
    )
    assert scale_rate.executed == 2
    assert scale_rate.pass_rate == 0.0

    # 2. _evaluate_baseline_decomposition:
    decomp_result = _evaluate_baseline_decomposition(
        scaled_results,
        baseline_results,
        queries,
        seed=42,
        target_skill="target-skill",
    )
    assert decomp_result.decomposition is not None
    assert decomp_result.delta_vs_baseline == 1.0
    assert decomp_result.decomposition.baseline_pass_rate == 1.0
    assert decomp_result.decomposition.scaled_pass_rate == 0.0


def test_extract_query_outcomes_and_single_skill_bootstrap_resampling(
    make_probe_result: Callable[..., Any],
) -> None:
    """Verify query outcomes and resample curve include out-of-scope probes in single skill."""
    from reach.models import Query, QueryKind
    from reach.sweep import _extract_query_outcomes, _resample_cluster_curve

    queries = [
        Query(id="q1", text="target query", kind=QueryKind.IMPLICIT, expected_skill="skill-a"),
        Query(id="q2", text="distractor query", kind=QueryKind.IMPLICIT, expected_skill="skill-b"),
        Query(id="q3", text="oos query", kind=QueryKind.OUT_OF_SCOPE, expected_skill=None),
    ]
    queries_by_id = {q.id: q for q in queries}
    truth = {q.id: q.expected_skill for q in queries}

    results = [
        make_probe_result(query_id="q1", invoked="skill-a"),
        make_probe_result(query_id="q2", invoked="skill-b"),
        make_probe_result(query_id="q3", invoked=()),
        make_probe_result(query_id="q1", error="Transient timeout"),
    ]

    outcomes = _extract_query_outcomes(
        results,
        truth,
        installed_skills={"skill-a", "skill-b"},
        target_skill="skill-a",
        queries_by_id=queries_by_id,
    )

    # Distractor q2 is skipped in single-skill mode
    assert "q2" not in outcomes

    # Errored probe on q1 is excluded from denominator
    assert outcomes["q1"].total == 1
    assert outcomes["q1"].hits == 1
    assert outcomes["q1"].tp == 1

    # Out-of-scope q3 is included with hit=1, total=1
    assert outcomes["q3"].total == 1
    assert outcomes["q3"].hits == 1
    assert outcomes["q3"].fp == 0

    # Resampling across scales evaluates hits / total in single-skill mode
    scale_query_sums = {10: outcomes}
    resampled = _resample_cluster_curve(
        scales=[10],
        scale_query_sums=scale_query_sums,
        sample_qids=["q1", "q3"],
        is_corpus=False,
    )
    assert len(resampled) == 1
    assert resampled[0] == 1.0


def test_build_scaling_point_scopes_probe_accounting_with_rivals(
    make_probe_result: Callable[..., Any],
) -> None:
    """Verify _build_scaling_point scopes probes_executed and errored in single skill."""
    from reach.models import Query, QueryKind
    from reach.queries import Origin, QuerySet, QuerySetProvenance
    from reach.sweep import _build_scaling_point

    queries = [
        Query(
            id="q_target",
            text="target task",
            kind=QueryKind.IMPLICIT,
            expected_skill="target-skill",
        ),
        Query(
            id="q_oos",
            text="out of scope task",
            kind=QueryKind.OUT_OF_SCOPE,
            expected_skill=None,
        ),
        Query(
            id="q_rival",
            text="rival negative task",
            kind=QueryKind.NEIGHBOR_NEGATIVE,
            expected_skill="rival-skill",
        ),
    ]
    qs = QuerySet(
        catalog_id="test",
        queries=tuple(queries),
        provenance=QuerySetProvenance(origin=Origin.AUTHORED),
    )

    results = (
        make_probe_result(query_id="q_target", invoked="target-skill"),
        make_probe_result(query_id="q_oos", invoked="target-skill"),
        make_probe_result(query_id="q_rival", invoked="rival-skill"),
        make_probe_result(query_id="q_target", error="Connection failed"),
        make_probe_result(query_id="q_rival", error="Timeout"),
    )

    point, _ = _build_scaling_point(
        scale=10,
        catalog_id="cat-10",
        results=results,
        resolved_query_set=qs,
        baseline_results=(),
        installed_skills={"target-skill", "rival-skill"},
        target_skill="target-skill",
    )

    # Scoped probes are q_target (1 hit + 1 error = 2) and q_oos (1 fail = 1) -> 3 total
    assert point.probes_executed == 3
    assert point.probes_errored == 1
    assert point.probes_failed == 1
    # Valid non-errored probes = 2; pass rate = 1 hit / 2 valid = 0.50
    assert point.pass_rate == 0.5
    # Strict invariant validation holds
    assert point.probes_failed <= point.probes_executed - point.probes_errored


def test_stratified_bootstrap_preserves_skill_stratum_sizes() -> None:
    """Verify _build_query_strata and _draw_stratified_qids preserve per-skill query counts."""
    import random
    from collections import Counter

    from reach.models import Query, QueryKind
    from reach.sweep import _build_query_strata, _draw_stratified_qids

    queries_by_id = {
        "q1": Query(id="q1", text="t1", expected_skill="skill-a"),
        "q2": Query(id="q2", text="t2", expected_skill="skill-a"),
        "q3": Query(id="q3", text="t3", expected_skill="skill-b"),
        "q4": Query(id="q4", text="t4", expected_skill=None, kind=QueryKind.OUT_OF_SCOPE),
    }
    strata = _build_query_strata(["q1", "q2", "q3", "q4"], queries_by_id=queries_by_id)
    assert len(strata) == 3

    rng = random.Random(42)  # noqa: S311
    for _ in range(25):
        drawn = _draw_stratified_qids(strata, rng)
        counts = Counter(queries_by_id[qid].expected_skill or "__oos__" for qid in drawn)
        assert counts["skill-a"] == 2
        assert counts["skill-b"] == 1
        assert counts["__oos__"] == 1


def test_summarize_bootstrap_knees_right_censoring_and_pmf() -> None:
    """Verify _summarize_bootstrap_knees separates gradual drops from right-censored replicates."""
    from reach.sweep import (
        _classify_kneedle_replicate,
        _ReplicateRegime,
        _summarize_bootstrap_knees,
    )

    scales = (10, 25, 50, 100)
    # 40 replicates at K=10 (cliff), 30 at K=25, 20 gradual drops, 10 flat/no-drop (>K_max)
    regimes = (
        [_ReplicateRegime(knee=10, is_gradual_drop=False, is_cliff=True)] * 40
        + [_ReplicateRegime(knee=25, is_gradual_drop=False, is_cliff=False)] * 30
        + [_ReplicateRegime(knee=None, is_gradual_drop=True, is_cliff=False)] * 20
        + [_ReplicateRegime(knee=None, is_gradual_drop=False, is_cliff=False)] * 10
    )
    summary = _summarize_bootstrap_knees(regimes, scales, iterations=100, q_low=0.025, q_high=0.975)
    assert summary.interval == (10, 100)
    assert summary.upper_censored is True
    assert summary.pmf == {10: 0.4, 25: 0.3}
    assert summary.cliff_probability == 0.4
    assert summary.drop_probability == 0.9

    # Gradual-drop majority (26% localized knees, 74% gradual log-linear decay, 0% flat):
    # Must NOT fabricate a right-censored [25, >100] interval when drop_probability == 1.0
    gradual_majority_regimes = (
        [_ReplicateRegime(knee=25, is_gradual_drop=False, is_cliff=False)] * 12
        + [_ReplicateRegime(knee=50, is_gradual_drop=False, is_cliff=False)] * 14
        + [_ReplicateRegime(knee=None, is_gradual_drop=True, is_cliff=False)] * 74
    )
    gradual_summary = _summarize_bootstrap_knees(
        gradual_majority_regimes, scales, iterations=100, q_low=0.025, q_high=0.975
    )
    assert gradual_summary.interval is None
    assert gradual_summary.upper_censored is False
    assert gradual_summary.pmf == {25: 0.12, 50: 0.14}
    assert gradual_summary.drop_probability == 1.0

    # First-interval cliff detection in _classify_kneedle_replicate
    cliff_regime = _classify_kneedle_replicate(
        scales=(10, 25, 50, 100),
        values=(1.0, 0.20, 0.19, 0.18),
        noise_floor=0.05,
        weights=None,
    )
    assert cliff_regime.knee == 25
    assert cliff_regime.is_cliff is True


def test_scaling_study_pydantic_statistical_validators(
    make_scaling_point: Callable[..., Any],
    make_scaling_study: Callable[..., Any],
) -> None:
    """Verify ScalingStudy cross-field validators, extra=forbid, and find_kneedle_knee default."""
    from pydantic import ValidationError

    from reach.sweep import ReplicateCollisionDiagnostic

    # find_kneedle_knee defaults to auto_smooth=True (weighted PAVA smooths bounce to K=25)
    assert (
        find_kneedle_knee(
            (10, 25, 50, 100),
            (0.95, 0.70, 0.85, 0.50),
            noise_floor=0.05,
            weights=(1.0, 100.0, 1.0, 1.0),
        )
        == 25
    )
    assert (
        find_kneedle_knee(
            (10, 25, 50, 100),
            (0.95, 0.70, 0.85, 0.50),
            noise_floor=0.05,
            weights=(1.0, 100.0, 1.0, 1.0),
            auto_smooth=False,
        )
        == 50
    )

    # Valid censored study with catalog_replicates
    valid_study = make_scaling_study(
        scales=(10, 25, 50),
        catalog_replicates=2,
        knee_scale=25,
        knee_scale_interval=(10, 50),
        knee_upper_censored=True,
        knee_scale_pmf={10: 0.2, 25: 0.5},
        cliff_probability=0.2,
        drop_probability=0.8,
        skill_icc=0.35,
    )
    assert valid_study.knee_upper_censored is True
    assert valid_study.catalog_replicates == 2

    # extra="forbid" rejects unknown fields on ScalingPoint and ScalingStudy
    with pytest.raises(ValidationError):
        make_scaling_point(scale=10, unknown_field="bad")
    with pytest.raises(ValidationError):
        make_scaling_study(scales=(10, 25), unknown_field="bad")

    # knee_upper_censored without interval is rejected
    with pytest.raises(ValidationError, match="knee_upper_censored requires knee_scale_interval"):
        make_scaling_study(
            scales=(10, 25, 50),
            knee_scale_interval=None,
            knee_upper_censored=True,
        )

    # knee_upper_censored with upper bound != scales[-1] is rejected
    with pytest.raises(ValidationError, match="upper bound must equal max scale"):
        make_scaling_study(
            scales=(10, 25, 50),
            knee_scale_interval=(10, 25),
            knee_upper_censored=True,
        )

    # knee_scale_pmf key outside scales is rejected
    with pytest.raises(ValidationError, match="keys outside evaluated scales"):
        make_scaling_study(
            scales=(10, 25, 50),
            knee_scale_pmf={99: 0.5},
        )

    # knee_scale_pmf sum > 1.0 is rejected
    with pytest.raises(ValidationError, match=r"probabilities sum to .* > 1\.0"):
        make_scaling_study(
            scales=(10, 25, 50),
            knee_scale_pmf={10: 0.7, 25: 0.5},
        )

    # ReplicateCollisionDiagnostic requires disjoint non-empty failed/passed replicate tuples
    with pytest.raises(ValidationError, match="at least one failed and one passed"):
        ReplicateCollisionDiagnostic(
            query_id="q1", scale=25, failed_replicates=(), passed_replicates=(0,)
        )
    with pytest.raises(ValidationError, match="disjoint"):
        ReplicateCollisionDiagnostic(
            query_id="q1", scale=25, failed_replicates=(0,), passed_replicates=(0, 1)
        )


def test_detect_replicate_collisions_identifies_flipped_queries_and_suspect_distractors(
    make_probe_result: Callable[..., Any],
) -> None:
    """Verify _detect_replicate_collisions flags queries that flip across catalog replicates."""
    from reach.models import Catalog, CatalogMode, Query
    from reach.sweep import _detect_replicate_collisions, _QueryOutcome, _ScaleReplicateRecord

    cat_r0 = Catalog(
        id="sweep:corpus:4", mode=CatalogMode.SWEEP, skills=("s1", "s2", "rival-x", "d1")
    )
    cat_r1 = Catalog(
        id="sweep:corpus:4-r1", mode=CatalogMode.SWEEP, skills=("s1", "s2", "d2", "d3")
    )
    queries_by_id = {
        "q1": Query(id="q1", text="use s1", expected_skill="s1"),
        "q2": Query(id="q2", text="use s2", expected_skill="s2"),
    }
    replicates_by_scale = {
        4: (
            _ScaleReplicateRecord(
                catalog=cat_r0,
                results=(
                    make_probe_result(query_id="q1", invoked="rival-x"),
                    make_probe_result(query_id="q2", invoked="s2"),
                ),
                outcomes={
                    "q1": _QueryOutcome(tp=0, fp=1, fn=1, hits=0, total=1),
                    "q2": _QueryOutcome(tp=1, fp=0, fn=0, hits=1, total=1),
                },
            ),
            _ScaleReplicateRecord(
                catalog=cat_r1,
                results=(
                    make_probe_result(query_id="q1", invoked="s1"),
                    make_probe_result(query_id="q2", invoked="s2"),
                ),
                outcomes={
                    "q1": _QueryOutcome(tp=1, fp=0, fn=0, hits=1, total=1),
                    "q2": _QueryOutcome(tp=1, fp=0, fn=0, hits=1, total=1),
                },
            ),
        )
    }

    collisions = _detect_replicate_collisions(
        scales=(4,),
        replicates_by_scale=replicates_by_scale,
        queries_by_id=queries_by_id,
    )
    assert len(collisions) == 1
    assert collisions[0].query_id == "q1"
    assert collisions[0].expected_skill == "s1"
    assert collisions[0].scale == 4
    assert collisions[0].failed_replicates == (0,)
    assert collisions[0].passed_replicates == (1,)
    assert collisions[0].suspect_distractors == ("rival-x",)


def test_print_corpus_capacity_sweep_renders_all_summary_lines(
    make_scaling_point: Callable[..., ScalingPoint],
    make_scaling_study: Callable[..., ScalingStudy],
) -> None:
    """Verify _print_corpus_capacity_sweep groups replicate collisions and shows gradual share."""
    from io import StringIO

    from rich.console import Console

    from reach.sweep import ReplicateCollisionDiagnostic
    from reach.views.sweep import _print_corpus_capacity_sweep, _print_single_skill_sweep

    study = make_scaling_study(
        target_skill=None,
        catalog_replicates=2,
        scales=(10, 25, 50, 100),
        points=(
            make_scaling_point(scale=10, pass_rate=0.80, f1_score=0.80),
            make_scaling_point(scale=25, pass_rate=0.525, f1_score=0.525, delta_collision=0.275),
            make_scaling_point(scale=50, pass_rate=0.338, f1_score=0.338, delta_collision=0.462),
            make_scaling_point(scale=100, pass_rate=0.20, f1_score=0.20, delta_collision=0.60),
        ),
        knee_scale=None,
        knee_scale_interval=None,
        knee_upper_censored=False,
        knee_scale_pmf={25: 0.12, 50: 0.14},
        cliff_probability=0.0,
        drop_probability=1.0,
        skill_icc=0.43,
        replicate_collisions=(
            ReplicateCollisionDiagnostic(
                query_id="gke-storage-1",
                expected_skill="gke-storage",
                scale=10,
                failed_replicates=(0,),
                passed_replicates=(1,),
                suspect_distractors=("gke-storage-troubleshooting",),
            ),
            ReplicateCollisionDiagnostic(
                query_id="gke-storage-2",
                expected_skill="gke-storage",
                scale=25,
                failed_replicates=(0,),
                passed_replicates=(1,),
                suspect_distractors=("gke-storage-troubleshooting",),
            ),
            ReplicateCollisionDiagnostic(
                query_id="secops-1",
                expected_skill="secops-detection-engineering",
                scale=10,
                failed_replicates=(1,),
                passed_replicates=(0,),
                suspect_distractors=("secops-hunt",),
            ),
        ),
    )
    buf = StringIO()
    console = Console(file=buf, width=200, force_terminal=False)
    _print_corpus_capacity_sweep(console, study)
    output = buf.getvalue()

    assert "2 replicates" in output
    assert "Capacity Knee Inflection: gradual decay" in output
    assert "Knee Bootstrap Distribution: K=25: 12%, K=50: 14% (gradual=74%, drop=100%)" in output
    assert "Baseline Intra-Skill Correlation (ICC): rho=0.43" in output
    assert (
        "Replicate Collision Sensitivity: gke-storage ← gke-storage-troubleshooting "
        "(2 flips @ K=10,25), secops-detection-engineering ← secops-hunt (1 flip @ K=10)"
    ) in output

    # Single-skill sweep view parity
    single_study = make_scaling_study(
        target_skill="gcloud",
        is_corpus_sweep=False,
        catalog_replicates=3,
        scales=(5, 15, 30, 60),
        points=(
            make_scaling_point(scale=5, pass_rate=0.95),
            make_scaling_point(scale=15, pass_rate=0.90, delta_collision=0.05),
            make_scaling_point(scale=30, pass_rate=0.50, delta_collision=0.45),
            make_scaling_point(scale=60, pass_rate=0.45, delta_collision=0.50),
        ),
        knee_scale=15,
        knee_scale_interval=(15, 60),
        knee_upper_censored=True,
        knee_scale_pmf={15: 0.70, 30: 0.15},
        drop_probability=0.85,
        replicate_collisions=(
            ReplicateCollisionDiagnostic(
                query_id="gcloud-1",
                expected_skill="gcloud",
                scale=30,
                failed_replicates=(1,),
                passed_replicates=(0, 2),
                suspect_distractors=("cloud-run-basics",),
            ),
        ),
    )
    single_buf = StringIO()
    single_console = Console(file=single_buf, width=120, force_terminal=False)
    _print_single_skill_sweep(single_console, single_study)
    single_out = single_buf.getvalue()

    assert "3 replicates" in single_out
    assert "k* = 15" in single_out
    assert "[15, >60]" in single_out
    assert "K=15: 70%, K=30: 15%" in single_out
    assert "gcloud ← cloud-run-basics (1 flip @ K=30)" in single_out


def test_legacy_decomposition_field_aliases_and_collision_bounds() -> None:
    """Verify legacy field aliases deserialize and out-of-bounds replicates are safe."""
    from reach.metrics import DecompositionResult
    from reach.sweep import ScalingPoint, ScalingStudy, _resolve_collision_suspects

    decomp = DecompositionResult.model_validate(
        {
            "baseline_pass_rate": 0.9,
            "scaled_pass_rate": 0.7,
            "delta_total": 0.2,
            "delta_context": 0.08,
            "delta_shadowing": 0.12,
            "delta_context_ci": (0.02, 0.14),
            "delta_shadowing_ci": (0.05, 0.19),
        }
    )
    assert decomp.delta_abstention == pytest.approx(0.08)
    assert decomp.delta_collision == pytest.approx(0.12)
    assert decomp.delta_abstention_ci == (0.02, 0.14)
    assert decomp.delta_collision_ci == (0.05, 0.19)

    point = ScalingPoint.model_validate(
        {
            "scale": 10,
            "catalog_id": "cat-10",
            "pass_rate": 0.8,
            "pass_rate_interval": (0.6, 0.9),
            "delta_vs_baseline": 0.1,
            "delta_context": 0.04,
            "delta_shadowing": 0.06,
            "probes_executed": 10,
        }
    )
    assert point.delta_abstention == pytest.approx(0.04)
    assert point.delta_collision == pytest.approx(0.06)

    study = ScalingStudy.model_validate(
        {
            "is_corpus_sweep": True,
            "scales": (10,),
            "points": (point,),
            "baseline_pass_rate": 0.8,
            "final_pass_rate": 0.8,
            "total_delta": 0.1,
            "total_context_loss": 0.04,
            "total_shadowing_loss": 0.06,
        }
    )
    assert study.total_abstention_loss == pytest.approx(0.04)
    assert study.total_collision_loss == pytest.approx(0.06)

    assert _resolve_collision_suspects("q1", "s1", [5, 6], [0], ()) == ()


def test_scaling_study_load_and_save(
    tmp_path: Path,
    make_scaling_point: Callable[..., ScalingPoint],
    make_scaling_study: Callable[..., ScalingStudy],
) -> None:
    """Verify ScalingStudy load and save methods serialize and restore data faithfully."""
    pt = make_scaling_point(scale=10, pass_rate=0.9, f1_score=0.9, probes_executed=10)
    study = make_scaling_study(
        points=(pt,),
        knee_scale=10,
        drop_probability=0.8,
    )
    save_path = tmp_path / "subdir" / "study.json"
    saved = study.save(save_path)
    assert saved == save_path.resolve()
    assert save_path.exists()
    assert not any(save_path.parent.glob("*.tmp.*"))

    loaded = ScalingStudy.load(save_path)
    assert loaded == study
    assert loaded.knee_scale == 10
    assert loaded.drop_probability == 0.8


def test_scaling_study_load_raises_for_missing_or_invalid_file(tmp_path: Path) -> None:
    """Verify ScalingStudy.load raises appropriate errors for invalid or missing files."""
    missing = tmp_path / "does_not_exist.json"
    with pytest.raises(FileNotFoundError, match="Scaling study file not found"):
        ScalingStudy.load(missing)

    directory = tmp_path / "somedir"
    directory.mkdir()
    with pytest.raises(IsADirectoryError, match="Scaling study path is a directory"):
        ScalingStudy.load(directory)

    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("not json content", encoding="utf-8")
    with pytest.raises(ValueError, match="Failed to parse scaling study JSON"):
        ScalingStudy.load(corrupt)


def test_select_curve_levels_deduplicates_overlapping_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify _select_curve_levels deduplicates if high threshold overlaps with standard ticks."""
    import reach.views.sweep as sweep_views

    monkeypatch.setattr(sweep_views, "_ZOOM_HIGH_THRESHOLD", 0.80)
    levels = sweep_views._select_curve_levels([0.85, 0.90])
    assert levels == (1.0, 0.95, 0.90, 0.85, 0.80)


def test_select_curve_levels_high_accuracy_includes_75_percent(
    make_scaling_point: Callable[..., ScalingPoint],
) -> None:
    """Verify _select_curve_levels includes 75% tick and ASCII curve renders it."""
    from reach.views.sweep import _select_curve_levels, render_ascii_curve

    claude_vals = [0.9362, 0.8571, 0.8211, 0.7629, 0.7789]
    levels = _select_curve_levels(claude_vals)
    assert levels == (1.0, 0.95, 0.90, 0.85, 0.80, 0.75)

    scales = [10, 25, 50, 100, 150]
    points = [
        make_scaling_point(scale=s, f1_score=v, pass_rate=v)
        for s, v in zip(scales, claude_vals, strict=True)
    ]
    lines = render_ascii_curve(points, metric="f1")
    full_curve = "\n".join(lines)
    assert " 75% |" in full_curve
    assert " 80% |" in full_curve
    row_80 = next(line for line in lines if " 80% |" in line)
    row_75 = next(line for line in lines if " 75% |" in line)
    assert row_80.count("●") == 2
    assert row_75.count("●") == 1


def test_format_knee_bootstrap_distribution_first_step_cliff(
    make_scaling_study: Callable[..., ScalingStudy],
) -> None:
    """Verify _format_knee_bootstrap_distribution renders cliff after drop probability."""
    from reach.views.sweep import _format_knee_bootstrap_distribution

    study = make_scaling_study(
        scales=(10, 25, 50),
        knee_scale_pmf={25: 0.08, 50: 0.02},
        cliff_probability=0.08,
        drop_probability=0.86,
    )
    formatted = _format_knee_bootstrap_distribution(study)
    assert formatted == "K=25: 8%, K=50: 2% (gradual=76%, drop=86%; first-step cliff=8%)"


def test_print_sweep_table_standardizes_prec_and_time_headers(
    make_scaling_point: Callable[..., ScalingPoint],
    make_scaling_study: Callable[..., ScalingStudy],
) -> None:
    """Verify corpus and single-skill sweeps render Prec and Time across all truncation cases."""
    from io import StringIO

    from rich.console import Console

    from reach.views.sweep import print_sweep

    pt1 = make_scaling_point(
        scale=10, pass_rate=0.9, f1_score=0.9, duration_ms_mean=3400.0, probes_executed=10
    )
    pt2 = make_scaling_point(
        scale=25, pass_rate=0.8, f1_score=0.8, duration_ms_mean=4200.0, probes_executed=10
    )
    pt_trunc = make_scaling_point(
        scale=50,
        pass_rate=0.7,
        f1_score=0.7,
        duration_ms_mean=5100.0,
        delta_truncated=0.05,
        probes_executed=10,
    )

    # Case 1: Corpus sweep without truncation
    corpus_study = make_scaling_study(points=(pt1, pt2))
    buf1 = StringIO()
    print_sweep(Console(file=buf1, force_terminal=False, width=100), corpus_study)
    corpus_out = buf1.getvalue()
    assert "Prec" in corpus_out
    assert "Time" in corpus_out
    assert "3.4s" in corpus_out
    assert "4.2s" in corpus_out

    # Case 2: Corpus sweep with truncation
    corpus_trunc_study = make_scaling_study(
        points=(pt1, pt_trunc),
        total_truncated_loss=0.05,
    )
    buf2 = StringIO()
    print_sweep(Console(file=buf2, force_terminal=False, width=100), corpus_trunc_study)
    corpus_trunc_out = buf2.getvalue()
    assert "Prec" in corpus_trunc_out
    assert "Time" in corpus_trunc_out
    assert "Trunc" in corpus_trunc_out
    assert "5.1s" in corpus_trunc_out

    # Case 3: Single-skill sweep without truncation
    single_study_notrunc = make_scaling_study(
        is_corpus_sweep=False,
        target_skill="skill-demo",
        points=(pt1, pt2),
    )
    buf3 = StringIO()
    print_sweep(Console(file=buf3, force_terminal=False, width=100), single_study_notrunc)
    single_notrunc_out = buf3.getvalue()
    assert "Pass Rate" in single_notrunc_out
    assert "Time" in single_notrunc_out
    assert "3.4s" in single_notrunc_out
    assert "4.2s" in single_notrunc_out
    assert "ms" not in single_notrunc_out.split("Time")[1]

    # Case 4: Single-skill sweep with truncation
    single_study_trunc = make_scaling_study(
        is_corpus_sweep=False,
        target_skill="skill-demo",
        points=(pt1, pt_trunc),
        total_truncated_loss=0.05,
    )
    buf4 = StringIO()
    print_sweep(Console(file=buf4, force_terminal=False, width=100), single_study_trunc)
    single_out = buf4.getvalue()
    assert "Pass Rate" in single_out
    assert "Time" in single_out
    assert "3.4s" in single_out
    assert "5.1s" in single_out
    assert "ms" not in single_out.split("Time")[1]
