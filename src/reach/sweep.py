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

"""Execute catalog scaling sweeps and detect capacity knees."""

from __future__ import annotations

import difflib
import logging
import math
import os
import random
import statistics
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, NamedTuple, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
    PositiveInt,
    StringConstraints,
    ValidationError,
    model_validator,
)

from reach.catalog import (
    CorpusScalingPlan,
    build_scaling_catalogs,
    find_cluster_medoids,
    load_skills,
    resolve_sweep_scales,
)
from reach.config import RunConfig, StudySettings
from reach.diff import DEFAULT_CONFIDENCE, NOISE_INFLATION
from reach.diff import noise_floor as diff_noise_floor
from reach.metrics import (
    DecompositionResult,
    _build_query_strata,
    _draw_stratified_qids,
    _rao_wu_rescale,
    compute_f1,
    decompose_pass_rate_drop,
    score_trajectory,
)
from reach.models import Catalog, CatalogMode, ProbeResult, Query, QueryKind, Skill
from reach.queries import QuerySet, load_query_set
from reach.run import Composition, conduct, validate_catalog_fit
from reach.runtime import AgentRuntime, build_runtime
from reach.uncertainty import (
    Interval,
    bootstrap_quantiles,
    ci_span_sigmas,
    cluster_wilson_interval,
    effective_sample_size,
    estimate_skill_icc,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

__all__ = [
    "PairedTrialOutcomes",
    "ReplicateCollisionDiagnostic",
    "ScalingPoint",
    "ScalingStudy",
    "bootstrap_f1_ci",
    "compute_scaling_noise_floor",
    "find_kneedle_knee",
    "run_scaling_sweep",
]


logger = logging.getLogger(__name__)

type UnitInterval = Annotated[float, Field(ge=0.0, le=1.0)]
type KneePmf = dict[PositiveInt, UnitInterval]


class ReplicateCollisionDiagnostic(BaseModel):
    """Identify a query whose routing outcome flips across catalog replicates at scale K."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    query_id: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
    expected_skill: str | None = None
    scale: PositiveInt
    failed_replicates: tuple[NonNegativeInt, ...]
    passed_replicates: tuple[NonNegativeInt, ...]
    suspect_distractors: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _validate_replicate_partition(self) -> Self:
        """Ensure failed and passed replicate sets are non-empty and disjoint."""
        if not self.failed_replicates or not self.passed_replicates:
            msg = (
                "ReplicateCollisionDiagnostic requires at least one failed and one passed replicate"
            )
            raise ValueError(msg)
        if set(self.failed_replicates) & set(self.passed_replicates):
            msg = "failed_replicates and passed_replicates must be disjoint"
            raise ValueError(msg)
        return self


class PairedTrialOutcomes(BaseModel):
    """Represent paired McNemar discordant trial outcomes across sweep scales."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    n10: Annotated[
        NonNegativeInt,
        Field(description="Probes successful at baseline but failed at scaled catalog"),
    ]
    n01: Annotated[
        NonNegativeInt,
        Field(description="Probes failed at baseline but successful at scaled catalog"),
    ]
    total_paired: Annotated[
        NonNegativeInt,
        Field(description="Total mutually executed probes in both baseline and scaled arms"),
    ]
    effective_paired: Annotated[
        NonNegativeInt | None,
        Field(description="Effective sample size adjusting for repeated attempts"),
    ] = None

    @model_validator(mode="after")
    def _validate_paired_totals(self) -> Self:
        """Enforce that total paired trials is at least the sum of discordant pairs."""
        if self.total_paired < (self.n10 + self.n01):
            msg = (
                f"total_paired ({self.total_paired}) cannot be less than the sum of "
                f"discordant pairs ({self.n10 + self.n01})"
            )
            raise ValueError(msg)
        return self


class ScalingPoint(BaseModel):
    """Represent evaluation outcomes and decomposition for a single catalog scale point."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    # 1. Identifiers
    scale: PositiveInt
    catalog_id: str

    # 2. Primary rates & intervals (alphabetical pairs)
    f1_score: UnitInterval = 0.0
    f1_interval: Interval = Field(default_factory=Interval.unit)
    pass_rate: UnitInterval
    pass_rate_interval: Interval
    precision: UnitInterval = 0.0
    precision_interval: Interval = Field(default_factory=Interval.unit)
    recall: UnitInterval = 0.0
    recall_interval: Interval = Field(default_factory=Interval.unit)

    # 3. Secondary rates
    abstention_rate: UnitInterval | None = None
    abstention_interval: Interval | None = None
    entrypoint_f1_score: UnitInterval | None = None
    entrypoint_pass_rate: UnitInterval | None = None
    external_distractor_precision: UnitInterval | None = None
    internal_precision: UnitInterval = 0.0

    # 4. Probe accounting counts
    probes_executed: NonNegativeInt
    in_scope_probes: NonNegativeInt = 0
    negative_probes: NonNegativeInt = 0
    probes_failed: NonNegativeInt = 0
    probes_errored: NonNegativeInt = 0

    # 5. Delta attribution
    delta_vs_baseline: float
    delta_abstention: float = 0.0
    delta_collision: float = 0.0
    delta_truncated: float = 0.0

    # 6. Disclosure states & telemetry
    disclosure_states: dict[str, int] = Field(default_factory=dict)
    prompt_tokens_mean: float | None = None
    duration_ms_mean: float = 0.0
    step_efficiency_mean: float = 0.0
    skill_f1_mean: float = 0.0

    @model_validator(mode="after")
    def _validate_probe_accounting(self) -> Self:
        """Ensure runtime errors and routing misses do not exceed executed probe bounds."""
        if self.probes_errored > self.probes_executed:
            msg = (
                f"probes_errored ({self.probes_errored}) cannot exceed "
                f"probes_executed ({self.probes_executed})"
            )
            raise ValueError(msg)
        valid_probes = self.probes_executed - self.probes_errored
        if self.probes_failed > valid_probes:
            msg = (
                f"probes_failed ({self.probes_failed}) cannot exceed "
                f"valid non-errored probes ({valid_probes})"
            )
            raise ValueError(msg)
        return self

    @property
    def all_probes_errored(self) -> bool:
        """Return True when at least one probe ran and every probe failed with a runtime error."""
        return self.probes_executed > 0 and self.probes_errored == self.probes_executed


class ScalingStudy(BaseModel):
    """Represent multi-scale catalog scaling study and knee curvature analysis."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    # 1. Corpus metadata
    target_skill: str | None = None
    is_corpus_sweep: bool = False
    catalog_replicates: PositiveInt = 1
    total_corpus_skills: int = 0
    scales: tuple[int, ...]

    # 2. Executive knee & drop findings
    knee_scale: int | None = None
    knee_scale_interval: tuple[int, int] | None = None
    knee_upper_censored: bool = False
    cliff_probability: UnitInterval | None = None
    drop_probability: UnitInterval | None = None
    steepest_drop_scales: tuple[int, int] | None = None
    steepest_drop_delta: float | None = None

    # 3. Headline pass rates and loss deltas (flattened)
    baseline_pass_rate: UnitInterval = 0.0
    final_pass_rate: UnitInterval = 0.0
    delta_total: float = 0.0
    delta_abstention: float = 0.0
    delta_collision: float = 0.0
    delta_truncated: float = 0.0
    delta_total_interval: Interval = Field(default_factory=Interval.zero)
    delta_abstention_interval: Interval = Field(default_factory=Interval.zero)
    delta_collision_interval: Interval = Field(default_factory=Interval.zero)
    delta_truncated_interval: Interval = Field(default_factory=Interval.zero)
    noise_floor: float = 0.05
    sample_size: NonNegativeInt = 0
    baseline_interval: Interval = Field(default_factory=Interval.zero)
    scaled_interval: Interval = Field(default_factory=Interval.zero)

    # 4. Diagnostics & maps
    anchor_skills: tuple[str, ...] | None = None
    knee_scale_pmf: KneePmf | None = None
    paired_outcomes: PairedTrialOutcomes | None = None
    replicate_collisions: tuple[ReplicateCollisionDiagnostic, ...] = ()
    skill_icc: UnitInterval | None = None

    # 5. Points leaf collection at the bottom
    points: tuple[ScalingPoint, ...]

    @classmethod
    def load(cls, path: Path | str) -> Self:
        """Load and deserialize a ScalingStudy from a JSON file.

        Raises:
            FileNotFoundError: If the target file does not exist.
            IsADirectoryError: If the target path is a directory.
            ValueError: If the target path contains invalid study data.
        """
        p = Path(path).expanduser().resolve()
        if not p.exists():
            msg = f"Scaling study file not found: {p}"
            raise FileNotFoundError(msg)
        if p.is_dir():
            msg = f"Scaling study path is a directory, not a file: {p}"
            raise IsADirectoryError(msg)
        try:
            content = p.read_text(encoding="utf-8")
        except OSError as err:
            msg = f"Failed to read scaling study file at {p}: {err}"
            raise OSError(msg) from err

        try:
            return cls.model_validate_json(content)
        except (ValueError, ValidationError) as err:
            msg = f"Failed to parse scaling study JSON from {p}: {err}"
            raise ValueError(msg) from err

    def save(self, path: Path | str) -> Path:
        """Serialize the scaling study to a formatted JSON file atomically.

        Args:
            path: Target file path for the serialized JSON output.

        Returns:
            The resolved Path where the study was written.

        Raises:
            OSError: If directory creation or file writing fails.
        """
        target = Path(path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp_target = target.with_suffix(f"{target.suffix}.tmp.{os.getpid()}")
        try:
            tmp_target.write_text(self.model_dump_json(indent=2) + "\n", encoding="utf-8")
            tmp_target.replace(target)
        finally:
            if tmp_target.exists():
                tmp_target.unlink(missing_ok=True)
        return target

    @model_validator(mode="after")
    def _validate_target_skill_for_mode(self) -> Self:
        """Ensure targeted sweeps specify a target skill and statistical bounds are valid."""
        if not self.is_corpus_sweep and self.target_skill is None:
            msg = "Targeted scaling sweep requires target_skill to be specified."
            raise ValueError(msg)
        if (
            self.knee_scale_interval is not None
            and self.knee_scale_interval[0] > self.knee_scale_interval[1]
        ):
            msg = f"knee_scale_interval bounds are inverted: {self.knee_scale_interval}"
            raise ValueError(msg)
        if self.knee_upper_censored:
            if self.knee_scale_interval is None:
                msg = "knee_upper_censored requires knee_scale_interval to be set"
                raise ValueError(msg)
            if self.scales and self.knee_scale_interval[1] != self.scales[-1]:
                msg = (
                    f"knee_upper_censored upper bound must equal max scale "
                    f"{self.scales[-1]}, got {self.knee_scale_interval[1]}"
                )
                raise ValueError(msg)
        if (
            self.cliff_probability is not None
            and self.drop_probability is not None
            and self.cliff_probability > self.drop_probability + 1e-6
        ):
            msg = (
                f"cliff_probability ({self.cliff_probability}) cannot exceed "
                f"drop_probability ({self.drop_probability})"
            )
            raise ValueError(msg)
        if self.knee_scale_pmf:
            if self.scales and any(k not in self.scales for k in self.knee_scale_pmf):
                msg = f"knee_scale_pmf contains keys outside evaluated scales {self.scales}"
                raise ValueError(msg)
            pmf_sum = sum(self.knee_scale_pmf.values())
            if pmf_sum > _PMF_SUM_MAX:
                msg = f"knee_scale_pmf probabilities sum to {pmf_sum} > 1.0"
                raise ValueError(msg)
        return self


_PMF_SUM_MAX: float = 1.001
_MIN_KNEE_POINTS: int = 3
_MIN_DIFF_POINTS: int = 2
_MIN_NOISE_FLOOR: float = 0.01
_PAVA_DECIMAL_PRECISION: int = 4


class _PavaBlock(NamedTuple):
    """Represent an aggregated block in the Pool Adjacent Violators Algorithm."""

    mean: float
    weight: float
    size: int


def _merge_pava_blocks(left: _PavaBlock, right: _PavaBlock) -> _PavaBlock:
    """Merge two adjacent PAVA blocks into a combined weighted mean block."""
    w_total = left.weight + right.weight
    mean = (left.mean * left.weight + right.mean * right.weight) / w_total
    return _PavaBlock(
        mean=mean,
        weight=w_total,
        size=left.size + right.size,
    )


def _isotonic_regression_pava(
    values: Sequence[float],
    weights: Sequence[float] | None = None,
) -> list[float]:
    """Compute maximum likelihood monotone non-increasing regression via PAVA."""
    if not values:
        return []
    w = [float(x) for x in weights] if weights is not None else [1.0] * len(values)
    blocks: list[_PavaBlock] = [
        _PavaBlock(mean=float(v), weight=float(wt), size=1) for v, wt in zip(values, w, strict=True)
    ]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i].mean < blocks[i + 1].mean:
            blocks[i] = _merge_pava_blocks(blocks[i], blocks[i + 1])
            del blocks[i + 1]
            while i > 0 and blocks[i - 1].mean < blocks[i].mean:
                i -= 1
                blocks[i] = _merge_pava_blocks(blocks[i], blocks[i + 1])
                del blocks[i + 1]
        else:
            i += 1
    result = []
    for b in blocks:
        result.extend([round(b.mean, _PAVA_DECIMAL_PRECISION)] * b.size)
    return result


_MIN_PAVA_VARIANCE: float = 1e-4


def _compute_pava_weights(
    intervals: Sequence[Interval | tuple[float, float]],
    ci_span: float,
    *,
    min_variance: float = _MIN_PAVA_VARIANCE,
) -> list[float]:
    """Compute inverse-variance PAVA weights from confidence intervals."""
    if ci_span <= 0:
        msg = f"ci_span must be strictly positive, got {ci_span}"
        raise ValueError(msg)
    clamped_min_var = max(1e-12, min_variance)
    return [
        1.0
        / max(
            clamped_min_var,
            (
                max(
                    0.0,
                    (iv.high - iv.low) if isinstance(iv, Interval) else (iv[1] - iv[0]),
                )
                / ci_span
            )
            ** 2,
        )
        for iv in intervals
    ]


def _compute_mcnemar_noise_floor(
    n10: int,
    n01: int,
    total_paired: int,
    effective_paired: float | None = None,
    confidence: float = DEFAULT_CONFIDENCE,
    noise_inflation: float = NOISE_INFLATION,
) -> float:
    """Compute cluster-adjusted McNemar scaling noise floor from discordant pair counts."""
    n_raw = max(1, total_paired)
    n_eff = max(1.0, effective_paired if effective_paired is not None else float(n_raw))
    # Asymptotic McNemar variance scaled by cluster survey design effect (DEFF = n_raw / n_eff)
    # Var_cluster = Var_raw * DEFF = (n10 + n01 - (n10 - n01)^2 / n_raw) / (n_raw * n_eff)
    var_num = max(0.0, float(n10 + n01) - ((float(n10 - n01) ** 2) / n_raw))
    var_paired = var_num / (float(n_raw) * n_eff)
    se_paired = math.sqrt(var_paired)
    eff_inflation = (
        1.0 if effective_paired is not None and effective_paired < n_raw else noise_inflation
    )
    floor = diff_noise_floor(se_paired / 2.0, se_paired / 2.0, confidence, eff_inflation)
    return max(_MIN_NOISE_FLOOR, floor)


def compute_scaling_noise_floor(
    baseline_pass_rate: float,
    scaled_pass_rate: float,
    sample_size: int,
    confidence: float = DEFAULT_CONFIDENCE,
    noise_inflation: float = NOISE_INFLATION,
    *,
    paired_outcomes: PairedTrialOutcomes | None = None,
) -> float:
    """Calculate minimum scaling pass-rate drop distinguishable from noise using diff."""
    if paired_outcomes is not None:
        return _compute_mcnemar_noise_floor(
            paired_outcomes.n10,
            paired_outcomes.n01,
            paired_outcomes.total_paired,
            paired_outcomes.effective_paired,
            confidence=confidence,
            noise_inflation=noise_inflation,
        )

    n = max(1, sample_size)
    control_se = math.sqrt(max(0.0, baseline_pass_rate * (1.0 - baseline_pass_rate)) / n)
    treatment_se = math.sqrt(max(0.0, scaled_pass_rate * (1.0 - scaled_pass_rate)) / n)
    floor = diff_noise_floor(control_se, treatment_se, confidence, noise_inflation)
    return max(_MIN_NOISE_FLOOR, floor)


def _compute_effective_noise_floor(
    noise_floor: float | None,
    points: Sequence[ScalingPoint],
    baseline_count: int,
    *,
    paired_outcomes: PairedTrialOutcomes | None = None,
    rate_curve: Sequence[float] | None = None,
) -> float:
    """Resolve user-configured or diff-derived scaling noise floor."""
    if noise_floor is not None:
        return noise_floor
    if len(points) >= _MIN_DIFF_POINTS:
        baseline_rate = rate_curve[0] if rate_curve else points[0].pass_rate
        scaled_rate = rate_curve[-1] if rate_curve else points[-1].pass_rate
        return round(
            compute_scaling_noise_floor(
                baseline_rate,
                scaled_rate,
                sample_size=max(1, baseline_count),
                paired_outcomes=paired_outcomes,
            ),
            4,
        )
    return 0.05


def find_kneedle_knee(
    scales: Sequence[int],
    pass_rates: Sequence[float],
    noise_floor: float = 0.10,
    *,
    weights: Sequence[float] | None = None,
    auto_smooth: bool = True,
) -> int | None:
    """Identify the inflection knee scale k* using normalized log-scale Kneedle curvature."""
    if (
        len(scales) < _MIN_KNEE_POINTS
        or len(scales) != len(pass_rates)
        or (max(pass_rates) - min(pass_rates)) <= noise_floor
    ):
        return None

    if weights is not None and len(weights) == len(scales):
        paired = sorted(zip(scales, pass_rates, weights, strict=True), key=lambda p: p[0])
        k_vals = [p[0] for p in paired]
        y_raw = [p[1] for p in paired]
        w_sorted = [p[2] for p in paired]
        y_vals = _isotonic_regression_pava(y_raw, weights=w_sorted) if auto_smooth else y_raw
    else:
        points = sorted(zip(scales, pass_rates, strict=True), key=lambda p: p[0])
        k_vals = [p[0] for p in points]
        y_raw = [p[1] for p in points]
        y_vals = _isotonic_regression_pava(y_raw) if auto_smooth else y_raw

    if (y_vals[0] - y_vals[-1]) <= noise_floor:
        return None

    log_k = [math.log(k) for k in k_vals]
    min_log = log_k[0]
    max_log = log_k[-1]
    log_range = max_log - min_log
    min_y = min(y_vals)
    max_y = max(y_vals)
    y_range = max_y - min_y

    if log_range <= 0.0 or y_range <= 0.0:
        return None

    norm_x = [(lk - min_log) / log_range for lk in log_k]
    norm_y = [(y - min_y) / y_range for y in y_vals]

    y0 = norm_y[0]
    y_end = norm_y[-1]
    steepest_idx = max(range(1, len(y_vals)), key=lambda i: y_vals[i - 1] - y_vals[i])
    diffs_below = [(y0 + x * (y_end - y0)) - y for x, y in zip(norm_x, norm_y, strict=True)]
    diffs_above = [y - (y0 + x * (y_end - y0)) for x, y in zip(norm_x, norm_y, strict=True)]

    below_window = diffs_below[1 : min(len(diffs_below) - 1, steepest_idx + 2)]
    max_below = max(below_window) if below_window else 0.0
    max_above = max(diffs_above[1:-1])
    min_prominence = max(0.05, (noise_floor * 0.5) / y_range)

    if max_above >= max_below and max_above > min_prominence:
        knee_idx = 1 + diffs_above[1:-1].index(max_above)
        return k_vals[knee_idx]
    if max_below > min_prominence:
        knee_idx = 1 + diffs_below[1:-1].index(max_below)
        return k_vals[knee_idx]
    return None


def _invoked_in_scope(
    scored_seq: Sequence[str],
    targets: set[str] | frozenset[str],
    *,
    trajectory: bool,
) -> bool:
    """Check whether scored_seq invokes any skill in targets under the trajectory mode."""
    return (
        any(s in targets for s in scored_seq)
        if trajectory
        else (bool(scored_seq) and scored_seq[0] in targets)
    )


def _evaluate_probe_trajectory(
    result: ProbeResult,
    expected: str | None,
    query: Query | None = None,
    *,
    trajectory: bool = True,
) -> tuple[bool, tuple[str, ...]]:
    """Evaluate probe trajectory via score_trajectory and return (hit, scored_invocations)."""
    raw_seq = (
        tuple(result.invoked_skills)
        if result.invoked_skills
        else ((result.invoked_skill,) if result.invoked_skill is not None else ())
    )
    if query is not None:
        t_score = score_trajectory(query, raw_seq)
        scored_seq = query.scored_invocations(raw_seq)
        hit = t_score.trajectory_hit if trajectory else t_score.entrypoint_hit
        return (not result.error and hit), scored_seq

    scored_seq = raw_seq
    if expected is None:
        return (not result.error and len(scored_seq) == 0), scored_seq
    hit = _invoked_in_scope(scored_seq, {expected}, trajectory=trajectory)
    return (not result.error and hit), scored_seq


def _classify_probe_outcome(
    result: ProbeResult,
    expected: str | None,
    installed_skills: set[str],
    target_skill: str | None = None,
    query: Query | None = None,
    *,
    trajectory: bool = True,
) -> tuple[bool, bool, bool]:
    """Classify a single probe result into (is_tp, is_fp, is_fn) confusion indicators."""
    hit, scored_seq = _evaluate_probe_trajectory(result, expected, query, trajectory=trajectory)
    if target_skill is not None:
        is_tp = expected == target_skill and hit
        is_fn = expected == target_skill and not is_tp
        invoked_target = _invoked_in_scope(scored_seq, {target_skill}, trajectory=trajectory)
        is_fp = not result.error and invoked_target and expected != target_skill
        return is_tp, is_fp, is_fn

    is_tp = expected is not None and hit
    is_fn = expected is not None and not is_tp
    invoked_installed = _invoked_in_scope(scored_seq, installed_skills, trajectory=trajectory)
    is_fp = not result.error and not hit and invoked_installed
    return is_tp, is_fp, is_fn


class _QueryOutcome(NamedTuple):
    """Confusion counts and pass/hit totals for a query at a given scale."""

    tp: int = 0
    fp: int = 0
    fn: int = 0
    hits: int = 0
    total: int = 0


_EMPTY_QUERY_OUTCOME = _QueryOutcome()


def _extract_query_outcomes(
    results: Sequence[ProbeResult],
    truth: Mapping[str, str | None],
    installed_skills: set[str],
    target_skill: str | None = None,
    queries_by_id: Mapping[str, Query] | None = None,
    *,
    trajectory: bool = True,
    include_other_skills: bool = False,
) -> dict[str, _QueryOutcome]:
    """Aggregate (tp, fp, fn, hits, total) counts grouped by query_id for a single scale."""
    valid_results = [r for r in results if not r.error]
    outcomes: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0, 0, 0])
    for r in valid_results:
        q = queries_by_id.get(r.query_id) if queries_by_id is not None else None
        exp = truth.get(r.query_id) if q is None else q.expected_skill

        if (
            target_skill is not None
            and not include_other_skills
            and exp != target_skill
            and exp is not None
        ):
            continue

        is_tp, is_fp, is_fn = _classify_probe_outcome(
            r,
            exp,
            installed_skills,
            target_skill=target_skill,
            query=q,
            trajectory=trajectory,
        )
        counts = outcomes[r.query_id]
        counts[0] += int(is_tp)
        counts[1] += int(is_fp)
        counts[2] += int(is_fn)

        if target_skill is not None:
            if exp == target_skill or exp is None:
                hit = (exp == target_skill and is_tp) or (exp is None and not is_fp)
                counts[3] += int(hit)
                counts[4] += 1
        else:
            hit, _ = _evaluate_probe_trajectory(r, exp, q, trajectory=trajectory)
            counts[3] += int(hit)
            counts[4] += 1

    return {
        qid: _QueryOutcome(counts[0], counts[1], counts[2], counts[3], counts[4])
        for qid, counts in outcomes.items()
    }


def bootstrap_f1_ci(
    results: Sequence[ProbeResult],
    truth: Mapping[str, str | None],
    installed_skills: set[str],
    iterations: int = 1000,
    seed: int = 42,
    target_skill: str | None = None,
    queries_by_id: Mapping[str, Query] | None = None,
    *,
    trajectory: bool = True,
) -> tuple[float, float]:
    """Compute stratified cluster-bootstrap confidence interval for micro F1 score."""
    if not results or iterations <= 0:
        return (0.0, 1.0)

    outcomes = _extract_query_outcomes(
        results,
        truth,
        installed_skills,
        target_skill=target_skill,
        queries_by_id=queries_by_id,
        trajectory=trajectory,
        include_other_skills=True,
    )
    qids = list(outcomes.keys())
    if not qids:
        return (0.0, 1.0)

    strata = _build_query_strata(qids, truth=truth, queries_by_id=queries_by_id)
    rng = random.Random(seed)  # noqa: S311
    f1_boots: list[float] = []

    for _ in range(iterations):
        sample_qids = _draw_stratified_qids(strata, rng)
        tp_s = sum(outcomes[q].tp for q in sample_qids)
        fp_s = sum(outcomes[q].fp for q in sample_qids)
        fn_s = sum(outcomes[q].fn for q in sample_qids)
        denom = 2 * tp_s + fp_s + fn_s
        f1_boots.append(2.0 * tp_s / denom if denom > 0 else 0.0)

    q_low, q_high = bootstrap_quantiles()
    low_idx = max(0, int(iterations * q_low))
    high_idx = min(int(iterations * q_high), iterations - 1)
    return _rao_wu_rescale(f1_boots, strata, low_idx, high_idx, min_val=0.0, max_val=1.0)


def _estimate_baseline_skill_icc(
    baseline_outcomes: Mapping[str, _QueryOutcome] | None,
    queries_by_id: Mapping[str, Query] | None,
) -> float | None:
    """Estimate intra-skill correlation at baseline scale K0 across evaluated skills."""
    if not baseline_outcomes or not queries_by_id:
        return None
    outcomes_by_skill: dict[str, list[float]] = defaultdict(list)
    for qid, entry in baseline_outcomes.items():
        q = queries_by_id.get(qid)
        if q is None or q.expected_skill is None:
            continue
        if entry.total > 0:
            outcomes_by_skill[q.expected_skill].append(entry.hits / entry.total)
    return estimate_skill_icc(outcomes_by_skill)


_FUZZY_MATCH_CUTOFF = 0.5
_MAX_FUZZY_SUGGESTIONS = 3


class _SkillNotFoundError(KeyError, ValueError):
    """Raise when a requested target or anchor skill is absent from the loaded corpus."""

    def __str__(self) -> str:
        """Return unquoted exception message string."""
        return str(self.args[0]) if self.args else ""


def _format_missing_skill_hint(
    missing_names: Sequence[str],
    corpus_names: Sequence[str],
) -> str:
    """Build a fuzzy close-match suggestion suffix for missing skill names."""
    sorted_names = sorted(corpus_names)
    suggestions: list[str] = []
    for name in missing_names:
        scored = [
            (difflib.SequenceMatcher(None, name, candidate).ratio(), candidate)
            for candidate in sorted_names
        ]
        scored.sort(key=lambda item: (-item[0], item[1]))
        matches = [candidate for ratio, candidate in scored if ratio >= _FUZZY_MATCH_CUTOFF][
            :_MAX_FUZZY_SUGGESTIONS
        ]
        if len(matches) < _MAX_FUZZY_SUGGESTIONS and "-" in name:
            prefix = "-".join(name.split("-")[:-1]) + "-"
            for candidate in sorted_names:
                if candidate.startswith(prefix) and candidate not in matches:
                    matches.append(candidate)
                    if len(matches) >= _MAX_FUZZY_SUGGESTIONS:
                        break
        for match in matches:
            if match not in suggestions:
                suggestions.append(match)
    if not suggestions:
        return ""
    return f". Did you mean: {', '.join(suggestions[:_MAX_FUZZY_SUGGESTIONS])}?"


def _resolve_sweep_target_and_queries(
    skills: Sequence[Skill],
    raw_query_set: QuerySet,
    target_skill: str | None,
) -> tuple[str, QuerySet]:
    """Resolve target skill and filter query set to relevant target queries."""
    target = target_skill
    if not target:
        target = next(
            (q.expected_skill for q in raw_query_set.queries if q.expected_skill),
            skills[0].name,
        )

    by_name = {s.name: s for s in skills}
    if target not in by_name:
        hint = _format_missing_skill_hint((target,), tuple(by_name))
        msg = f"target skill {target!r} not found in loaded skills{hint}"
        raise _SkillNotFoundError(msg)

    other_skills = sorted(
        (set(by_name) | raw_query_set.covered_skills()) - {target},
        key=len,
        reverse=True,
    )
    target_queries = tuple(
        q
        for q in raw_query_set.queries
        if (q.expected_skill == target and q.kind != QueryKind.NEIGHBOR_NEGATIVE)
        or q.is_out_of_scope
        or (
            q.kind == QueryKind.NEIGHBOR_NEGATIVE
            and not (
                q.query_id.startswith("adv-")
                and not q.query_id.removeprefix("adv-").startswith(f"{target}-")
                and any(
                    q.query_id.removeprefix("adv-").startswith(f"{other}-")
                    for other in other_skills
                )
            )
        )
    )
    if not target_queries:
        target_queries = raw_query_set.queries

    query_set = QuerySet(
        catalog_id="all",
        queries=target_queries,
        provenance=raw_query_set.provenance,
    )
    return target, query_set


class _ScalePassRate(NamedTuple):
    """Represent point pass rate and Wilson confidence interval."""

    executed: int
    fails: int
    pass_rate: float
    pass_interval: tuple[float, float]


class _ScaleClassificationMetrics(NamedTuple):
    """Represent classification and confusion matrix outcomes for a scaling point."""

    in_scope_probes: int
    negative_probes: int
    recall: float
    recall_interval: tuple[float, float]
    precision: float
    precision_interval: tuple[float, float]
    internal_precision: float
    external_distractor_precision: float | None
    abstention_rate: float | None
    abstention_interval: tuple[float, float] | None
    f1_score: float
    f1_interval: tuple[float, float]
    step_efficiency_mean: float = 0.0
    skill_f1_mean: float = 0.0


class _ScaleTelemetry(NamedTuple):
    """Represent aggregate execution telemetry and resource utilization."""

    disclosure_states: dict[str, int]
    prompt_tokens_mean: float | None
    duration_ms_mean: float


class _ScaleDecomposition(NamedTuple):
    """Represent comparative degradation components against baseline."""

    delta_vs_baseline: float
    delta_abstention: float
    delta_collision: float
    decomposition: DecompositionResult | None
    delta_truncated: float = 0.0


def _estimate_query_attempts(results: Sequence[ProbeResult]) -> int:
    """Estimate average attempts per query across probe results."""
    if not results:
        return 1
    attempts_counter = Counter(r.query_id for r in results)
    return max(1, round(statistics.fmean(attempts_counter.values())))


def _calculate_scale_pass_rate(
    results: Sequence[ProbeResult],
    queries_by_id: Mapping[str, Query],
    target_skill: str | None = None,
    *,
    trajectory: bool = True,
) -> _ScalePassRate:
    """Calculate overall pass rate, failure count, and Wilson confidence interval."""
    valid_results = [r for r in results if not r.error]
    outcomes = _extract_query_outcomes(
        valid_results,
        {},
        set(),
        target_skill=target_skill,
        queries_by_id=queries_by_id,
        trajectory=trajectory,
    )
    hits = sum(o.hits for o in outcomes.values())
    scored_count = sum(o.total for o in outcomes.values())
    fails = scored_count - hits
    pass_rate = hits / scored_count if scored_count else 0.0
    attempts = _estimate_query_attempts(valid_results)
    s_icc = _estimate_baseline_skill_icc(outcomes, queries_by_id) if target_skill is None else None
    skills_n = len(
        {
            queries_by_id[q].expected_skill
            for q in outcomes
            if q in queries_by_id and queries_by_id[q].expected_skill is not None
        }
    )
    q_per_skill = len(outcomes) / skills_n if s_icc and skills_n > 0 else 1.0
    interval_obj = cluster_wilson_interval(
        hits,
        scored_count,
        attempts=attempts,
        queries_per_skill=q_per_skill,
        skill_icc=s_icc or 0.0,
    )
    pass_interval = (interval_obj.low, interval_obj.high) if interval_obj else (0.0, 1.0)
    return _ScalePassRate(
        executed=scored_count,
        fails=fails,
        pass_rate=pass_rate,
        pass_interval=pass_interval,
    )


def _compute_scope_counts(
    in_scope: Sequence[ProbeResult],
    queries_by_id: Mapping[str, Query],
    installed: set[str],
    target_skill: str | None = None,
    *,
    trajectory: bool = True,
) -> tuple[int, int, float, tuple[float, float]]:
    """Compute true positives, routing false positives, recall, and Wilson interval."""
    relevant = [
        r
        for r in in_scope
        if target_skill is None
        or (
            queries_by_id[r.query_id].expected_skill == target_skill
            if r.query_id in queries_by_id
            else False
        )
    ]
    outcomes = [
        _classify_probe_outcome(
            r,
            queries_by_id[r.query_id].expected_skill if r.query_id in queries_by_id else None,
            installed,
            target_skill,
            query=queries_by_id.get(r.query_id),
            trajectory=trajectory,
        )
        for r in in_scope
    ]
    tp = sum(1 for is_tp, _, _ in outcomes if is_tp)
    internal_fp = sum(1 for _, is_fp, _ in outcomes if is_fp)
    recall = tp / len(relevant) if relevant else 0.0
    attempts = _estimate_query_attempts(relevant)
    rec_int = cluster_wilson_interval(tp, len(relevant), attempts=attempts)
    recall_interval = (rec_int.low, rec_int.high) if rec_int else (0.0, 1.0)
    return tp, internal_fp, recall, recall_interval


def _compute_negative_counts(
    negative: Sequence[ProbeResult],
    queries_by_id: Mapping[str, Query],
    installed: set[str],
    target_skill: str | None = None,
    *,
    trajectory: bool = True,
) -> tuple[int, int, float | None, tuple[float, float] | None]:
    """Compute true negatives, distractor false positives, and abstention intervals."""
    tn = sum(
        1
        for r in negative
        if _evaluate_probe_trajectory(
            r,
            None,
            queries_by_id.get(r.query_id),
            trajectory=trajectory,
        )[0]
    )
    fp_distractor = sum(
        1
        for r in negative
        if _classify_probe_outcome(
            r,
            None,
            installed,
            target_skill,
            query=queries_by_id.get(r.query_id),
            trajectory=trajectory,
        )[1]
    )
    if not negative:
        return tn, fp_distractor, None, None
    abstention_rate = round(tn / len(negative), 4)
    attempts = _estimate_query_attempts(negative)
    abst_int = cluster_wilson_interval(tn, len(negative), attempts=attempts)
    abstention_interval = (round(abst_int.low, 4), round(abst_int.high, 4)) if abst_int else None
    return tn, fp_distractor, abstention_rate, abstention_interval


def _compute_precision_metrics(
    tp: int,
    internal_fp: int,
    fp_distractor: int,
    has_negatives: bool,
    attempts: int = 1,
) -> tuple[float, float | None, float, tuple[float, float]]:
    """Compute internal precision, external distractor precision, and overall precision."""
    internal_precision = tp / (tp + internal_fp) if (tp + internal_fp) else 1.0
    ext_prec = (
        round(tp / (tp + fp_distractor), 4)
        if (has_negatives and (tp + fp_distractor) > 0)
        else None
    )
    overall_fp = internal_fp + fp_distractor
    precision = tp / (tp + overall_fp) if (tp + overall_fp) else 1.0
    prec_int = (
        cluster_wilson_interval(tp, tp + overall_fp, attempts=attempts)
        if (tp + overall_fp)
        else None
    )
    precision_interval = (prec_int.low, prec_int.high) if prec_int else (1.0, 1.0)
    return internal_precision, ext_prec, precision, precision_interval


def _calculate_scale_classification(
    results: Sequence[ProbeResult],
    queries_by_id: Mapping[str, Query],
    installed_skills: set[str] | None,
    seed: int,
    target_skill: str | None = None,
    *,
    trajectory: bool = True,
    compute_ci: bool = True,
) -> _ScaleClassificationMetrics:
    """Calculate precision, recall, abstention rate, and F1 confidence intervals."""
    installed = installed_skills if installed_skills is not None else set()
    valid_results = [r for r in results if not r.error]
    in_scope = [
        r
        for r in valid_results
        if (queries_by_id[r.query_id].expected_skill if r.query_id in queries_by_id else None)
        is not None
    ]
    negative = [
        r
        for r in valid_results
        if (queries_by_id[r.query_id].expected_skill if r.query_id in queries_by_id else None)
        is None
    ]

    tp, internal_fp, recall, recall_interval = _compute_scope_counts(
        in_scope,
        queries_by_id,
        installed,
        target_skill=target_skill,
        trajectory=trajectory,
    )
    _tn, fp_distractor, abstention_rate, abstention_interval = _compute_negative_counts(
        negative,
        queries_by_id,
        installed,
        target_skill=target_skill,
        trajectory=trajectory,
    )
    attempts = _estimate_query_attempts(valid_results)
    internal_prec, ext_prec, precision, precision_interval = _compute_precision_metrics(
        tp, internal_fp, fp_distractor, bool(negative), attempts=attempts
    )

    f1 = compute_f1(precision, recall)
    if compute_ci:
        truth_expected = {qid: q.expected_skill for qid, q in queries_by_id.items()}
        f1_ci = bootstrap_f1_ci(
            valid_results,
            truth_expected,
            installed,
            iterations=1000,
            seed=seed,
            target_skill=target_skill,
            queries_by_id=queries_by_id,
            trajectory=trajectory,
        )
    else:
        rounded_f1 = round(f1, 4)
        f1_ci = (rounded_f1, rounded_f1)

    step_effs: list[float] = []
    skill_f1s: list[float] = []
    for r in valid_results:
        q = queries_by_id.get(r.query_id)
        if q is not None:
            if target_skill is not None and q.expected_skill != target_skill:
                continue
            raw_seq = (
                tuple(r.invoked_skills)
                if r.invoked_skills
                else ((r.invoked_skill,) if r.invoked_skill is not None else ())
            )
            t_score = score_trajectory(q, raw_seq)
            if t_score.step_efficiency is not None:
                step_effs.append(t_score.step_efficiency)
            if t_score.skill_f1 is not None:
                skill_f1s.append(t_score.skill_f1)

    step_eff_mean = round(statistics.fmean(step_effs), 4) if step_effs else 0.0
    sk_f1_mean = round(statistics.fmean(skill_f1s), 4) if skill_f1s else 0.0

    return _ScaleClassificationMetrics(
        in_scope_probes=len(in_scope),
        negative_probes=len(negative),
        recall=recall,
        recall_interval=recall_interval,
        precision=precision,
        precision_interval=precision_interval,
        internal_precision=internal_prec,
        external_distractor_precision=ext_prec,
        abstention_rate=abstention_rate,
        abstention_interval=abstention_interval,
        f1_score=f1,
        f1_interval=(f1_ci[0], f1_ci[1]),
        step_efficiency_mean=step_eff_mean,
        skill_f1_mean=sk_f1_mean,
    )


def _aggregate_scale_telemetry(results: Sequence[ProbeResult]) -> _ScaleTelemetry:
    """Aggregate disclosure states, prompt tokens, and duration across probes."""
    disclosure_counts = dict(
        Counter(
            r.disclosure_state.value
            for r in results
            if getattr(r, "disclosure_state", None) is not None
        )
    )
    tokens: list[float] = [float(r.prompt_tokens) for r in results if r.prompt_tokens is not None]
    avg_tokens = statistics.fmean(tokens) if tokens else None
    prompt_tokens_mean = round(avg_tokens, 2) if avg_tokens is not None else None

    durations = [r.duration_ms for r in results if r.duration_ms is not None]
    avg_duration = round(statistics.fmean(durations), 2) if durations else 0.0

    return _ScaleTelemetry(
        disclosure_states=disclosure_counts,
        prompt_tokens_mean=prompt_tokens_mean,
        duration_ms_mean=avg_duration,
    )


def _evaluate_baseline_decomposition(
    results: Sequence[ProbeResult],
    baseline_results: Sequence[ProbeResult],
    queries: Sequence[Query],
    seed: int,
    target_skill: str | None = None,
) -> _ScaleDecomposition:
    """Evaluate loss decomposition against baseline results if baseline is provided."""
    if not baseline_results:
        return _ScaleDecomposition(
            delta_vs_baseline=0.0,
            delta_abstention=0.0,
            delta_collision=0.0,
            decomposition=None,
            delta_truncated=0.0,
        )
    scoped_queries = (
        [q for q in queries if q.expected_skill == target_skill or q.is_out_of_scope]
        if target_skill is not None
        else list(queries)
    )
    scoped_ids = {q.query_id for q in scoped_queries}
    scoped_base = [r for r in baseline_results if r.query_id in scoped_ids]
    scoped_res = [r for r in results if r.query_id in scoped_ids]
    decomp = decompose_pass_rate_drop(
        baseline_results=scoped_base,
        scaled_results=scoped_res,
        queries=scoped_queries,
        seed=seed,
    )
    return _ScaleDecomposition(
        delta_vs_baseline=decomp.delta_total,
        delta_abstention=decomp.delta_abstention,
        delta_collision=decomp.delta_collision,
        decomposition=decomp,
        delta_truncated=decomp.delta_truncated,
    )


def _build_scaling_point(
    scale: int,
    catalog_id: str,
    results: tuple[ProbeResult, ...],
    resolved_query_set: QuerySet,
    baseline_results: tuple[ProbeResult, ...],
    installed_skills: set[str] | None = None,
    target_skill: str | None = None,
    seed: int = 42,
) -> tuple[ScalingPoint, DecompositionResult | None]:
    """Calculate point metrics, Wilson confidence intervals, and pass-rate decomposition."""
    queries = resolved_query_set.queries
    queries_by_id = {q.query_id: q for q in queries}

    pass_stats = _calculate_scale_pass_rate(
        results,
        queries_by_id,
        target_skill=target_skill,
        trajectory=True,
    )
    entry_pass_stats = _calculate_scale_pass_rate(
        results,
        queries_by_id,
        target_skill=target_skill,
        trajectory=False,
    )
    class_stats = _calculate_scale_classification(
        results,
        queries_by_id,
        installed_skills,
        seed,
        target_skill=target_skill,
        trajectory=True,
        compute_ci=True,
    )
    entry_class_stats = _calculate_scale_classification(
        results,
        queries_by_id,
        installed_skills,
        seed,
        target_skill=target_skill,
        trajectory=False,
        compute_ci=False,
    )
    telemetry = _aggregate_scale_telemetry(results)
    decomp_stats = _evaluate_baseline_decomposition(
        results, baseline_results, queries, seed, target_skill=target_skill
    )

    scoped_results = [
        r
        for r in results
        if target_skill is None
        or (
            (q := queries_by_id.get(r.query_id)) is not None
            and (q.expected_skill == target_skill or q.is_out_of_scope)
        )
    ]

    point = ScalingPoint(
        scale=scale,
        catalog_id=catalog_id,
        pass_rate=round(pass_stats.pass_rate, 4),
        pass_rate_interval=Interval(
            low=round(pass_stats.pass_interval[0], 4),
            high=round(pass_stats.pass_interval[1], 4),
        ),
        recall=round(class_stats.recall, 4),
        recall_interval=Interval(
            low=round(class_stats.recall_interval[0], 4),
            high=round(class_stats.recall_interval[1], 4),
        ),
        precision=round(class_stats.precision, 4),
        precision_interval=Interval(
            low=round(class_stats.precision_interval[0], 4),
            high=round(class_stats.precision_interval[1], 4),
        ),
        internal_precision=round(class_stats.internal_precision, 4),
        external_distractor_precision=class_stats.external_distractor_precision,
        abstention_rate=class_stats.abstention_rate,
        abstention_interval=(
            Interval.from_tuple(class_stats.abstention_interval)
            if class_stats.abstention_interval is not None
            else None
        ),
        f1_score=round(class_stats.f1_score, 4),
        f1_interval=Interval(
            low=round(class_stats.f1_interval[0], 4),
            high=round(class_stats.f1_interval[1], 4),
        ),
        entrypoint_pass_rate=round(entry_pass_stats.pass_rate, 4),
        entrypoint_f1_score=round(entry_class_stats.f1_score, 4),
        in_scope_probes=class_stats.in_scope_probes,
        negative_probes=class_stats.negative_probes,
        disclosure_states=telemetry.disclosure_states,
        delta_vs_baseline=round(decomp_stats.delta_vs_baseline, 4),
        delta_abstention=round(decomp_stats.delta_abstention, 4),
        delta_collision=round(decomp_stats.delta_collision, 4),
        delta_truncated=round(decomp_stats.delta_truncated, 4),
        probes_executed=len(scoped_results),
        probes_failed=pass_stats.fails,
        probes_errored=sum(1 for r in scoped_results if r.error),
        prompt_tokens_mean=telemetry.prompt_tokens_mean,
        duration_ms_mean=telemetry.duration_ms_mean,
        step_efficiency_mean=class_stats.step_efficiency_mean,
        skill_f1_mean=class_stats.skill_f1_mean,
    )
    return point, decomp_stats.decomposition


def _prepare_sweep_config(
    config: RunConfig | None,
    attempts: int | None,
    bootstrap_iterations: int | None = None,
    seed: int | None = None,
    catalog_replicates: int | None = None,
) -> RunConfig:
    """Apply CLI overrides to execution configuration."""
    cfg = config or RunConfig()
    plan_update: dict[str, object] = {"attempts": cfg.plan.resolve_sweep_attempts(attempts)}
    study_update: dict[str, object] = {}
    if bootstrap_iterations is not None:
        study_update["bootstrap_iterations"] = bootstrap_iterations
    if seed is not None:
        study_update["bootstrap_seed"] = seed
    if catalog_replicates is not None:
        study_update["catalog_replicates"] = catalog_replicates

    cfg = cfg.model_copy(update={"plan": cfg.plan.model_copy(update=plan_update)})
    if study_update:
        cfg = cfg.model_copy(update={"study": cfg.study.model_copy(update=study_update)})
    return cfg


def _find_steepest_drop(
    scales: Sequence[int],
    values: Sequence[float],
) -> tuple[tuple[int, int] | None, float | None]:
    """Identify the adjacent scale interval with the largest positive metric drop."""
    if len(scales) < _MIN_DIFF_POINTS or len(scales) != len(values):
        return None, None
    best_drop = 0.0
    best_pair: tuple[int, int] | None = None
    for idx in range(len(scales) - 1):
        drop = values[idx] - values[idx + 1]
        if drop > best_drop:
            best_drop = drop
            best_pair = (scales[idx], scales[idx + 1])
    return best_pair, (round(best_drop, 4) if best_pair is not None else None)


def _build_study_result(
    target: str | None,
    is_corpus: bool,
    evaluated_scales: Sequence[int],
    points: Sequence[ScalingPoint],
    knee: int | None,
    noise_floor: float,
    total_skills: int,
    decomp: DecompositionResult | None,
    *,
    catalog_replicates: int = 1,
    anchor_skills: Sequence[str] | None = None,
    paired_outcomes: PairedTrialOutcomes | None = None,
    knee_interval: tuple[int, int] | None = None,
    knee_scale_pmf: Mapping[int, float] | None = None,
    knee_upper_censored: bool = False,
    cliff_probability: float | None = None,
    drop_probability: float | None = None,
    steepest_drop_scales: tuple[int, int] | None = None,
    steepest_drop_delta: float | None = None,
    skill_icc: float | None = None,
    replicate_collisions: Sequence[ReplicateCollisionDiagnostic] = (),
) -> ScalingStudy:
    """Construct finished ScalingStudy data model."""
    b_rate = points[0].pass_rate if points else 0.0
    f_rate = points[-1].pass_rate if points else 0.0
    t_delta = points[-1].delta_vs_baseline if len(points) > 1 else 0.0
    t_abs = points[-1].delta_abstention if len(points) > 1 else 0.0
    t_col = points[-1].delta_collision if len(points) > 1 else 0.0
    t_trunc = points[-1].delta_truncated if len(points) > 1 else 0.0

    d_total = decomp.delta_total if decomp else t_delta
    d_abs = decomp.delta_abstention if decomp else t_abs
    d_col = decomp.delta_collision if decomp else t_col
    d_trunc = decomp.delta_truncated if decomp else t_trunc
    d_tot_iv = Interval.from_tuple(decomp.delta_total_ci) if decomp else Interval.zero()
    d_abs_iv = Interval.from_tuple(decomp.delta_abstention_ci) if decomp else Interval.zero()
    d_col_iv = Interval.from_tuple(decomp.delta_collision_ci) if decomp else Interval.zero()
    d_trunc_iv = Interval.from_tuple(decomp.delta_truncated_ci) if decomp else Interval.zero()
    s_size = decomp.sample_size if decomp else 0
    b_iv = Interval.from_tuple(decomp.baseline_ci) if decomp else Interval.zero()
    s_iv = Interval.from_tuple(decomp.scaled_ci) if decomp else Interval.zero()

    return ScalingStudy(
        target_skill=target,
        is_corpus_sweep=is_corpus,
        catalog_replicates=max(1, catalog_replicates),
        total_corpus_skills=total_skills,
        scales=tuple(evaluated_scales),
        knee_scale=knee,
        knee_scale_interval=knee_interval,
        knee_upper_censored=knee_upper_censored,
        cliff_probability=cliff_probability,
        drop_probability=drop_probability,
        steepest_drop_scales=steepest_drop_scales,
        steepest_drop_delta=steepest_drop_delta,
        baseline_pass_rate=b_rate,
        final_pass_rate=f_rate,
        delta_total=d_total,
        delta_abstention=d_abs,
        delta_collision=d_col,
        delta_truncated=d_trunc,
        delta_total_interval=d_tot_iv,
        delta_abstention_interval=d_abs_iv,
        delta_collision_interval=d_col_iv,
        delta_truncated_interval=d_trunc_iv,
        noise_floor=noise_floor,
        sample_size=s_size,
        baseline_interval=b_iv,
        scaled_interval=s_iv,
        anchor_skills=tuple(anchor_skills) if anchor_skills is not None else None,
        knee_scale_pmf=dict(knee_scale_pmf) if knee_scale_pmf is not None else None,
        paired_outcomes=paired_outcomes,
        replicate_collisions=tuple(replicate_collisions),
        skill_icc=skill_icc,
        points=tuple(points),
    )


def _resolve_medoid_anchors(
    medoid_count: int,
    resolved_skills: Sequence[Skill],
    query_set: QuerySet | None,
    *,
    clamp_to_queried: bool,
) -> tuple[str, ...]:
    """Select TF-IDF cluster medoid anchors across the full or queried corpus."""
    full_medoids = find_cluster_medoids(resolved_skills, medoid_count)
    if not clamp_to_queried or query_set is None:
        return full_medoids
    queried_names = query_set.covered_skills()
    if set(full_medoids).issubset(queried_names):
        return full_medoids
    queried_skills = [s for s in resolved_skills if s.name in queried_names]
    if len(queried_skills) >= medoid_count:
        return find_cluster_medoids(queried_skills, medoid_count)
    if not queried_skills:
        logger.warning(
            "Provided query set contains 0 benchmark queries for resident skills; "
            "falling back to unqueried corpus medoids."
        )
    else:
        logger.warning(
            "Existing query set covers %d skill(s), which is less than requested anchor "
            "count K0=%d; selecting full corpus medoids for auto-drafting.",
            len(queried_skills),
            medoid_count,
        )
    return full_medoids


def _resolve_anchor_count(
    requested_anchor: int | Sequence[str] | str | None,
    default_count: int,
) -> int | None:
    """Extract numeric anchor count from requested anchor specification, if applicable."""
    match requested_anchor:
        case int():
            return requested_anchor
        case str() if requested_anchor.isdigit():
            return int(requested_anchor)
        case None:
            return default_count
        case _:
            return None


def _resolve_anchor_skills(
    requested_anchor: int | Sequence[str] | str | None,
    resolved_skills: Sequence[Skill],
    actual_scales: Sequence[int],
    query_set: QuerySet | None = None,
    *,
    clamp_to_queried: bool = True,
) -> tuple[str, ...] | None:
    """Resolve anchor skills cohort from configuration or initial scale medoids."""
    if isinstance(requested_anchor, str) and requested_anchor.lower() == "all":
        return None

    medoid_count = _resolve_anchor_count(requested_anchor, actual_scales[0])
    if medoid_count is not None:
        return _resolve_medoid_anchors(
            medoid_count,
            resolved_skills,
            query_set,
            clamp_to_queried=clamp_to_queried,
        )

    if isinstance(requested_anchor, str):
        raw_names = tuple(p.strip() for p in requested_anchor.split(",") if p.strip())
    elif isinstance(requested_anchor, (list, tuple)):
        raw_names = tuple(str(s).strip() for s in requested_anchor if str(s).strip())
    else:
        return find_cluster_medoids(resolved_skills, actual_scales[0])

    corpus_names = {s.name for s in resolved_skills}
    missing = [a for a in raw_names if a not in corpus_names]
    if missing:
        hint = _format_missing_skill_hint(missing, tuple(corpus_names))
        msg = f"anchor skill(s) not found in corpus: {', '.join(missing)}{hint}"
        raise ValueError(msg)
    return raw_names


class _ReplicateSetup(NamedTuple):
    """Store resolved target, catalogs, query set, and corpus plan for a catalog replicate."""

    target: str | None
    catalogs: list[Catalog]
    query_set: QuerySet
    corpus_plan: CorpusScalingPlan | None
    resolved_anchors: tuple[str, ...] | None


class _ScaleReplicateRecord(NamedTuple):
    """Store catalog, probe outcomes, and query outcome totals for a single replicate at scale K."""

    catalog: Catalog
    results: tuple[ProbeResult, ...]
    outcomes: dict[str, _QueryOutcome]


def _setup_sweep_execution(
    is_corpus: bool,
    target_skill: str | None,
    anchor: int | Sequence[str] | str | None,
    resolved_skills: Sequence[Skill],
    actual_scales: Sequence[int],
    raw_query_set: QuerySet,
    effective_config: RunConfig,
    rivals_share: float,
    seed_override: int | None = None,
) -> _ReplicateSetup:
    """Configure catalogs, query sets, and scaling plan for sweep execution."""
    catalog_seed = seed_override if seed_override is not None else effective_config.catalog.seed
    if is_corpus:
        requested_anchor = anchor if anchor is not None else effective_config.study.anchor
        resolved_anchors = _resolve_anchor_skills(
            requested_anchor,
            resolved_skills,
            actual_scales,
            query_set=raw_query_set,
            clamp_to_queried=not effective_config.study.auto_queries or bool(raw_query_set.queries),
        )
        min_required = (
            actual_scales[0]
            if isinstance(requested_anchor, str) and requested_anchor.lower() == "all"
            else _resolve_anchor_count(requested_anchor, actual_scales[0])
        )

        if (
            not effective_config.study.auto_queries
            and raw_query_set.queries
            and min_required is not None
            and len(raw_query_set.covered_skills()) < min_required
        ):
            msg = (
                f"--no-auto-queries was specified, but existing query set only covers "
                f"{len(raw_query_set.covered_skills())} skill(s), which is less than "
                f"baseline scale K0={min_required}. Draft queries for at least "
                f"{min_required} skills or remove --no-auto-queries."
            )
            raise ValueError(msg)

        plan = CorpusScalingPlan.create(
            skills=resolved_skills,
            scales=actual_scales,
            anchor_skills=resolved_anchors,
            rivals_share=rivals_share,
            seed=catalog_seed,
        )
        return _ReplicateSetup(None, list(plan.catalogs), raw_query_set, plan, resolved_anchors)

    target, query_set = _resolve_sweep_target_and_queries(
        resolved_skills, raw_query_set, target_skill
    )
    catalogs = build_scaling_catalogs(
        skills=resolved_skills,
        target_skill=target,
        scales=actual_scales,
        rivals_share=rivals_share,
        seed=catalog_seed,
    )
    return _ReplicateSetup(target, catalogs, query_set, None, None)


class _PairedQueryOutcome(NamedTuple):
    """Represent query-level discordant pair counts between baseline and final scale."""

    n10: int = 0
    n01: int = 0
    total_paired: int = 0


def _extract_paired_query_outcomes(
    baseline_results: Sequence[ProbeResult],
    final_results: Sequence[ProbeResult],
    queries_by_id: Mapping[str, Query],
    target_skill: str | None = None,
) -> dict[str, _PairedQueryOutcome]:
    """Extract discordant pair counts grouped by query_id between baseline and final scales."""
    if not baseline_results or not final_results:
        return {}
    base_out = _extract_query_outcomes(
        baseline_results, {}, set(), target_skill=target_skill, queries_by_id=queries_by_id
    )
    final_out = _extract_query_outcomes(
        final_results, {}, set(), target_skill=target_skill, queries_by_id=queries_by_id
    )
    paired: dict[str, _PairedQueryOutcome] = {}
    for qid in sorted(base_out.keys() & final_out.keys()):
        b, f = base_out[qid], final_out[qid]
        n_q = min(b.total, f.total)
        if n_q <= 0:
            continue
        p_b = b.hits / b.total
        p_f = f.hits / f.total
        n10 = min(n_q, int(n_q * p_b * (1.0 - p_f) + 0.5))
        n01 = min(n_q - n10, int(n_q * (1.0 - p_b) * p_f + 0.5))
        paired[qid] = _PairedQueryOutcome(
            n10=n10,
            n01=n01,
            total_paired=n_q,
        )
    return paired


def _aggregate_paired_counts(
    outcomes: Sequence[_PairedQueryOutcome],
) -> tuple[int, int, int, int]:
    """Aggregate discordant pair counts and cluster-adjusted effective sample size."""
    n10 = sum(v.n10 for v in outcomes)
    n01 = sum(v.n01 for v in outcomes)
    total_paired = sum(v.total_paired for v in outcomes)
    attempts = max(1, round(total_paired / max(1, len(outcomes))))
    neff = effective_sample_size(total_paired, attempts=attempts)
    return n10, n01, total_paired, neff


def _build_paired_outcomes(
    per_query: Mapping[str, _PairedQueryOutcome],
) -> PairedTrialOutcomes | None:
    """Construct PairedTrialOutcomes from per-query discordant pair counts."""
    if not per_query:
        return None
    n10, n01, total_paired, neff = _aggregate_paired_counts(tuple(per_query.values()))
    if total_paired <= 0:
        return None
    return PairedTrialOutcomes(
        n10=n10,
        n01=n01,
        total_paired=total_paired,
        effective_paired=neff,
    )


def _calculate_paired_outcomes(
    baseline_results: Sequence[ProbeResult],
    final_results: Sequence[ProbeResult],
    queries_by_id: Mapping[str, Query],
    target_skill: str | None = None,
) -> PairedTrialOutcomes | None:
    """Calculate discordant pair counts (n10, n01) between baseline and final scale runs."""
    per_query = _extract_paired_query_outcomes(
        baseline_results, final_results, queries_by_id, target_skill=target_skill
    )
    return _build_paired_outcomes(per_query)


class _ReplicateRegime(NamedTuple):
    """Classify a single bootstrap replicate curve's knee behavior."""

    knee: int | None
    is_gradual_drop: bool
    is_cliff: bool


class _BootstrapKneeSummary(NamedTuple):
    """Summarize bootstrap knee distribution, censoring, and cliff/drop probabilities."""

    interval: tuple[int, int] | None = None
    pmf: dict[int, float] | None = None
    upper_censored: bool = False
    cliff_probability: float | None = None
    drop_probability: float | None = None


_INITIAL_CLIFF_DROP_SHARE: float = 0.60


def _classify_kneedle_replicate(
    scales: Sequence[int],
    values: Sequence[float],
    noise_floor: float,
    weights: Sequence[float] | None,
) -> _ReplicateRegime:
    """Classify a resampled curve into flat, gradual drop, or detected knee/cliff."""
    k = find_kneedle_knee(
        scales,
        values,
        noise_floor=noise_floor,
        weights=weights,
        auto_smooth=True,
    )
    smoothed = _isotonic_regression_pava(values, weights=weights)
    total_drop = smoothed[0] - smoothed[-1] if smoothed else 0.0
    if k is not None:
        first_step_drop = (smoothed[0] - smoothed[1]) if len(smoothed) > 1 else 0.0
        is_first_step_cliff = k == scales[0] or (
            len(scales) > 1
            and k == scales[1]
            and total_drop > 0.0
            and first_step_drop >= _INITIAL_CLIFF_DROP_SHARE * total_drop
        )
        return _ReplicateRegime(
            knee=k,
            is_gradual_drop=False,
            is_cliff=is_first_step_cliff,
        )
    return _ReplicateRegime(
        knee=None,
        is_gradual_drop=(total_drop > noise_floor),
        is_cliff=False,
    )


def _summarize_bootstrap_knees(
    regimes: Sequence[_ReplicateRegime],
    scales: Sequence[int],
    iterations: int,
    q_low: float,
    q_high: float,
) -> _BootstrapKneeSummary:
    """Summarize bootstrap knee replicates with right-censoring and PMF over scales."""
    if not regimes or iterations <= 0 or not scales:
        return _BootstrapKneeSummary()

    detected = sorted(r.knee for r in regimes if r.knee is not None)
    gradual_count = sum(1 for r in regimes if r.is_gradual_drop)
    cliff_count = sum(1 for r in regimes if r.is_cliff)
    b_drop = len(detected) + gradual_count
    no_drop_count = max(0, len(regimes) - b_drop)

    counts = Counter(detected)
    pmf = {s: round(counts[s] / iterations, 4) for s in scales if counts[s] > 0}
    cliff_prob = round(cliff_count / iterations, 4)
    drop_prob = round(b_drop / iterations, 4)

    min_detections = max(1, int(iterations * 0.50))
    n_censor_pool = len(detected) + no_drop_count
    raw_low_idx = int(n_censor_pool * q_low)
    if len(detected) < min_detections or raw_low_idx >= len(detected):
        return _BootstrapKneeSummary(
            interval=None,
            pmf=pmf or None,
            upper_censored=False,
            cliff_probability=cliff_prob,
            drop_probability=drop_prob,
        )

    low_idx = max(0, raw_low_idx)
    high_idx = min(int(n_censor_pool * q_high), n_censor_pool - 1)
    low_val = detected[low_idx]
    if high_idx >= len(detected):
        return _BootstrapKneeSummary(
            interval=(low_val, scales[-1]),
            pmf=pmf or None,
            upper_censored=True,
            cliff_probability=cliff_prob,
            drop_probability=drop_prob,
        )
    return _BootstrapKneeSummary(
        interval=(low_val, detected[high_idx]),
        pmf=pmf or None,
        upper_censored=False,
        cliff_probability=cliff_prob,
        drop_probability=drop_prob,
    )


def _resample_cluster_curve(
    scales: Sequence[int],
    scale_query_sums: Mapping[int, Mapping[str, _QueryOutcome]],
    sample_qids: Sequence[str],
    *,
    is_corpus: bool,
) -> list[float]:
    """Calculate resampled metric curve from sampled query clusters across scales."""
    resampled_curve: list[float] = []
    for s in scales:
        scale_sums = scale_query_sums[s]
        if is_corpus:
            tp = sum(scale_sums.get(q, _EMPTY_QUERY_OUTCOME).tp for q in sample_qids)
            fp = sum(scale_sums.get(q, _EMPTY_QUERY_OUTCOME).fp for q in sample_qids)
            fn = sum(scale_sums.get(q, _EMPTY_QUERY_OUTCOME).fn for q in sample_qids)
            denom = 2 * tp + fp + fn
            val = (2.0 * tp / denom) if denom > 0 else 0.0
        else:
            total_hits = sum(scale_sums.get(q, _EMPTY_QUERY_OUTCOME).hits for q in sample_qids)
            total_count = sum(scale_sums.get(q, _EMPTY_QUERY_OUTCOME).total for q in sample_qids)
            val = (total_hits / total_count) if total_count > 0 else 0.0
        resampled_curve.append(val)
    return resampled_curve


def _resample_paired_noise_floor(
    raw_noise_floor: float | None,
    fallback_noise_floor: float,
    points: Sequence[ScalingPoint],
    sample_qids: Sequence[str],
    paired_query_outcomes: Mapping[str, _PairedQueryOutcome] | None,
) -> float:
    """Compute replicate-specific dynamic noise floor when raw_noise_floor is unset."""
    if raw_noise_floor is not None:
        return raw_noise_floor
    if not paired_query_outcomes:
        return fallback_noise_floor
    sampled = [paired_query_outcomes.get(q, _PairedQueryOutcome()) for q in sample_qids]
    rep_n10, rep_n01, rep_total, rep_neff = _aggregate_paired_counts(sampled)
    if rep_total <= 0:
        return fallback_noise_floor
    if len(points) >= _MIN_DIFF_POINTS:
        return round(_compute_mcnemar_noise_floor(rep_n10, rep_n01, rep_total, rep_neff), 4)
    return 0.05


def _bootstrap_knee_summary(
    scales: Sequence[int],
    points: Sequence[ScalingPoint],
    noise_floor: float,
    iterations: int = 200,
    seed: int = 42,
    scale_query_sums: Mapping[int, Mapping[str, _QueryOutcome]] | None = None,
    confidence: float = DEFAULT_CONFIDENCE,
    *,
    is_corpus: bool = True,
    queries_by_id: Mapping[str, Query] | None = None,
    paired_query_outcomes: Mapping[str, _PairedQueryOutcome] | None = None,
    raw_noise_floor: float | None = None,
) -> _BootstrapKneeSummary:
    """Calculate stratified bootstrap knee summary, PMF, right-censoring, and cliff probability."""
    if (
        len(points) < _MIN_DIFF_POINTS
        or iterations <= 0
        or not scale_query_sums
        or not all(s in scale_query_sums for s in scales)
    ):
        return _BootstrapKneeSummary()

    common_qids = list(scale_query_sums[scales[0]].keys())
    if not common_qids:
        return _BootstrapKneeSummary()

    rng = random.Random(seed)  # noqa: S311
    regimes: list[_ReplicateRegime] = []

    ci_span = ci_span_sigmas(confidence)
    q_low, q_high = bootstrap_quantiles(confidence)

    active_intervals = (
        [p.f1_interval for p in points] if is_corpus else [p.pass_rate_interval for p in points]
    )
    weights = _compute_pava_weights(active_intervals, ci_span)
    strata = _build_query_strata(common_qids, queries_by_id=queries_by_id)

    for _ in range(iterations):
        sample_qids = _draw_stratified_qids(strata, rng)
        resampled_curve = _resample_cluster_curve(
            scales, scale_query_sums, sample_qids, is_corpus=is_corpus
        )
        rep_floor = _resample_paired_noise_floor(
            raw_noise_floor=raw_noise_floor,
            fallback_noise_floor=noise_floor,
            points=points,
            sample_qids=sample_qids,
            paired_query_outcomes=paired_query_outcomes,
        )
        regimes.append(_classify_kneedle_replicate(scales, resampled_curve, rep_floor, weights))

    return _summarize_bootstrap_knees(regimes, scales, iterations, q_low, q_high)


_MIN_REPLICATES_FOR_COLLISION: int = 2


def _resolve_collision_suspects(
    qid: str,
    expected_skill: str | None,
    failed_reps: Sequence[int],
    passed_reps: Sequence[int],
    replicates: Sequence[_ScaleReplicateRecord],
) -> tuple[str, ...]:
    """Identify suspect distractors present in failed replicates only."""
    valid_failed = [rep_idx for rep_idx in failed_reps if 0 <= rep_idx < len(replicates)]
    if not valid_failed:
        return ()

    passed_skills: set[str] = {
        sk
        for rep_idx in passed_reps
        if 0 <= rep_idx < len(replicates)
        for sk in replicates[rep_idx].catalog.skills
    }
    invoked_suspects: set[str] = set()
    for rep_idx in valid_failed:
        for r in replicates[rep_idx].results:
            if r.query_id != qid or r.error:
                continue
            invoked_seq = (
                tuple(r.invoked_skills)
                if r.invoked_skills
                else ((r.invoked_skill,) if r.invoked_skill is not None else ())
            )
            invoked_suspects.update(
                sk for sk in invoked_seq if sk != expected_skill and sk not in passed_skills
            )

    if not invoked_suspects:
        failed_common = set(replicates[valid_failed[0]].catalog.skills)
        for rep_idx in valid_failed[1:]:
            failed_common &= set(replicates[rep_idx].catalog.skills)
        excluded = passed_skills | ({expected_skill} if expected_skill else set())
        invoked_suspects = failed_common - excluded

    return tuple(sorted(invoked_suspects))


def _detect_replicate_collisions(
    scales: Sequence[int],
    replicates_by_scale: Mapping[int, Sequence[_ScaleReplicateRecord]],
    queries_by_id: Mapping[str, Query],
) -> tuple[ReplicateCollisionDiagnostic, ...]:
    """Identify queries whose pass/fail outcome flips across catalog replicates at scale K."""
    diagnostics: list[ReplicateCollisionDiagnostic] = []
    for scale in scales:
        replicates = replicates_by_scale.get(scale, ())
        if len(replicates) < _MIN_REPLICATES_FOR_COLLISION:
            continue

        all_qids = sorted({qid for rep in replicates for qid in rep.outcomes})
        for qid in all_qids:
            passed_reps: list[int] = []
            failed_reps: list[int] = []
            for rep_idx, rep in enumerate(replicates):
                entry = rep.outcomes.get(qid)
                if entry is None or entry.total == 0:
                    continue
                if entry.hits * 2 >= entry.total:
                    passed_reps.append(rep_idx)
                else:
                    failed_reps.append(rep_idx)

            if not failed_reps or not passed_reps:
                continue

            q_obj = queries_by_id.get(qid)
            expected_skill = q_obj.expected_skill if q_obj is not None else None
            suspects = _resolve_collision_suspects(
                qid, expected_skill, failed_reps, passed_reps, replicates
            )
            diagnostics.append(
                ReplicateCollisionDiagnostic(
                    query_id=qid,
                    expected_skill=expected_skill,
                    scale=scale,
                    failed_replicates=tuple(failed_reps),
                    passed_replicates=tuple(passed_reps),
                    suspect_distractors=suspects,
                )
            )
    return tuple(diagnostics)


def _assemble_scaling_study(
    *,
    target: str | None,
    is_corpus: bool,
    actual_scales: Sequence[int],
    points: Sequence[ScalingPoint],
    noise_floor: float | None,
    baseline_count: int,
    total_skills: int,
    decomp: DecompositionResult | None,
    anchor_skills: Sequence[str] | None,
    paired_outcomes: PairedTrialOutcomes | None = None,
    study_config: StudySettings | None = None,
    scale_query_sums: Mapping[int, Mapping[str, _QueryOutcome]] | None = None,
    queries_by_id: Mapping[str, Query] | None = None,
    paired_query_outcomes: Mapping[str, _PairedQueryOutcome] | None = None,
    replicate_collisions: Sequence[ReplicateCollisionDiagnostic] = (),
) -> ScalingStudy:
    """Compute effective noise floor and knee and construct a ScalingStudy."""
    rate_curve = [p.f1_score for p in points] if is_corpus else [p.pass_rate for p in points]
    effective_noise_floor = _compute_effective_noise_floor(
        noise_floor,
        points,
        baseline_count,
        paired_outcomes=paired_outcomes,
        rate_curve=rate_curve,
    )
    evaluated_scales = actual_scales[: len(points)]
    steepest_scales, steepest_delta = _find_steepest_drop(evaluated_scales, rate_curve)
    ci_span = ci_span_sigmas(DEFAULT_CONFIDENCE)
    active_intervals = (
        [p.f1_interval for p in points] if is_corpus else [p.pass_rate_interval for p in points]
    )
    weights = _compute_pava_weights(active_intervals, ci_span)
    knee = find_kneedle_knee(
        evaluated_scales,
        rate_curve,
        noise_floor=effective_noise_floor,
        weights=weights,
        auto_smooth=True,
    )
    bootstrap_iterations = study_config.bootstrap_iterations if study_config is not None else 200
    bootstrap_seed = study_config.bootstrap_seed if study_config is not None else 42
    effective_seed = bootstrap_seed if bootstrap_seed is not None else 42
    boot_summary = _bootstrap_knee_summary(
        evaluated_scales,
        points,
        effective_noise_floor,
        iterations=bootstrap_iterations,
        seed=effective_seed,
        scale_query_sums=scale_query_sums,
        is_corpus=is_corpus,
        queries_by_id=queries_by_id,
        paired_query_outcomes=paired_query_outcomes,
        raw_noise_floor=noise_floor,
    )
    baseline_outcomes = (
        scale_query_sums.get(evaluated_scales[0]) if scale_query_sums and evaluated_scales else None
    )
    skill_icc = _estimate_baseline_skill_icc(baseline_outcomes, queries_by_id)
    catalog_replicates = study_config.catalog_replicates if study_config is not None else 1

    return _build_study_result(
        target=target,
        is_corpus=is_corpus,
        catalog_replicates=catalog_replicates,
        evaluated_scales=evaluated_scales,
        points=points,
        knee=knee,
        noise_floor=effective_noise_floor,
        total_skills=total_skills,
        decomp=decomp,
        anchor_skills=anchor_skills,
        paired_outcomes=paired_outcomes,
        knee_interval=boot_summary.interval,
        knee_scale_pmf=boot_summary.pmf,
        knee_upper_censored=boot_summary.upper_censored,
        cliff_probability=boot_summary.cliff_probability,
        drop_probability=boot_summary.drop_probability,
        steepest_drop_scales=steepest_scales,
        steepest_drop_delta=steepest_delta,
        skill_icc=skill_icc,
        replicate_collisions=replicate_collisions,
    )


@dataclass(frozen=True)
class _SweepContext:
    """Encapsulate sweep execution invariants across scale iterations."""

    effective_config: RunConfig
    work_dir: Path
    corpus_plan: CorpusScalingPlan | None
    raw_query_set: QuerySet
    resolved_query_set: QuerySet
    resolved_skills: Sequence[Skill]
    raw_query_map: Mapping[str, Query]
    is_corpus: bool


def _prepare_scale_iteration(
    ctx: _SweepContext,
    catalog: Catalog,
) -> tuple[RunConfig, QuerySet, Composition]:
    """Prepare configuration, query set, and composition for a scale iteration."""
    safe_cat_id = catalog.id.replace(":", "_").replace("/", "_")
    scale_out = ctx.work_dir / f"sweep_{safe_cat_id}.jsonl"
    scale_config = ctx.effective_config.model_copy(
        update={
            "study": ctx.effective_config.study.model_copy(
                update={
                    "catalog": catalog.id,
                    "rescope": True,
                    "partial": True,
                    "workdir": ctx.work_dir,
                    "out": scale_out,
                }
            ),
            "catalog": ctx.effective_config.catalog.model_copy(update={"mode": CatalogMode.SWEEP}),
        }
    )
    scale_query_set = (
        ctx.corpus_plan.queries_for_scale(
            catalog=catalog,
            raw_query_set=ctx.raw_query_set,
        )
        if ctx.is_corpus and ctx.corpus_plan is not None
        else ctx.resolved_query_set
    )
    composed = Composition(
        config=scale_config,
        query_set=scale_query_set,
        catalog=catalog,
        skills=tuple(ctx.resolved_skills),
    )
    return scale_config, scale_query_set, composed


def _resolve_sweep_work_dir(
    study_workdir: Path | None,
) -> tuple[Path, tempfile.TemporaryDirectory[str] | None]:
    """Resolve active sweep workspace directory, initializing temporary directory if needed."""
    if study_workdir is not None:
        return study_workdir, None
    temp_dir_obj = tempfile.TemporaryDirectory(prefix="reach_sweep_", delete=False)
    return Path(temp_dir_obj.name), temp_dir_obj


def _execute_scale_replicates(
    *,
    step_idx: int,
    scale_workers: int,
    replicate_setups: Sequence[_ReplicateSetup],
    ctx: _SweepContext,
    resolved_runtime: AgentRuntime,
    allow_truncation: bool,
    shared_outcome_cache: dict[Any, Any],
    target: str | None,
) -> tuple[tuple[ProbeResult, ...], set[str], QuerySet, tuple[_ScaleReplicateRecord, ...]]:
    """Execute all catalog replicates for a single scale step and pool their probe outcomes."""
    pooled_results: list[ProbeResult] = []
    installed_union: set[str] = set()
    primary_query_set = ctx.resolved_query_set
    attempts_per_rep = ctx.effective_config.plan.attempts
    records: list[_ScaleReplicateRecord] = []

    for rep_idx, setup in enumerate(replicate_setups):
        rep_catalog = setup.catalogs[step_idx - 1]
        if rep_idx > 0:
            rep_catalog = rep_catalog.model_copy(update={"id": f"{rep_catalog.id}-r{rep_idx}"})
        rep_ctx = _SweepContext(
            effective_config=ctx.effective_config,
            work_dir=ctx.work_dir,
            corpus_plan=setup.corpus_plan,
            raw_query_set=ctx.raw_query_set,
            resolved_query_set=setup.query_set,
            resolved_skills=ctx.resolved_skills,
            raw_query_map=ctx.raw_query_map,
            is_corpus=ctx.is_corpus,
        )
        scale_config, scale_query_set, composed = _prepare_scale_iteration(rep_ctx, rep_catalog)
        if rep_idx == 0:
            primary_query_set = scale_query_set

        outcome = conduct(
            config=scale_config,
            runtime=resolved_runtime,
            composed=composed,
            allow_truncation=allow_truncation,
            append_across_arms=True,
            workers=scale_workers,
            outcome_cache=shared_outcome_cache,
        )
        rep_results = (
            tuple(
                r.model_copy(update={"attempt": rep_idx * attempts_per_rep + r.attempt})
                for r in outcome.results
            )
            if rep_idx > 0
            else outcome.results
        )
        pooled_results.extend(rep_results)
        installed_union.update(rep_catalog.skills)

        rep_qmap = {q.query_id: q for q in scale_query_set.queries}
        rep_truth = {q.query_id: q.expected_skill for q in scale_query_set.queries}
        rep_q_outcomes = _extract_query_outcomes(
            outcome.results,
            rep_truth,
            set(rep_catalog.skills),
            target_skill=target if not ctx.is_corpus else None,
            queries_by_id=rep_qmap,
        )
        records.append(
            _ScaleReplicateRecord(
                catalog=rep_catalog,
                results=outcome.results,
                outcomes=rep_q_outcomes,
            )
        )

    return tuple(pooled_results), installed_union, primary_query_set, tuple(records)


def _snapshot_scaling_study(
    *,
    target: str | None,
    is_corpus: bool,
    actual_scales: Sequence[int],
    points: Sequence[ScalingPoint],
    noise_floor: float | None,
    baseline_results: Sequence[ProbeResult],
    latest_results: Sequence[ProbeResult],
    total_skills: int,
    decomp: DecompositionResult | None,
    anchor_skills: Sequence[str] | None,
    study_config: StudySettings,
    scale_query_sums: Mapping[int, Mapping[str, _QueryOutcome]],
    queries_by_id: Mapping[str, Query],
    replicates_by_scale: Mapping[int, Sequence[_ScaleReplicateRecord]],
    scoped_target: str | None,
) -> ScalingStudy:
    """Build a partial or final ScalingStudy from accumulated sweep state."""
    paired_by_q = _extract_paired_query_outcomes(
        baseline_results, latest_results, queries_by_id, target_skill=scoped_target
    )
    return _assemble_scaling_study(
        target=target,
        is_corpus=is_corpus,
        actual_scales=actual_scales,
        points=points,
        noise_floor=noise_floor,
        baseline_count=len(baseline_results),
        total_skills=total_skills,
        decomp=decomp,
        anchor_skills=anchor_skills,
        paired_outcomes=_build_paired_outcomes(paired_by_q),
        study_config=study_config,
        scale_query_sums=scale_query_sums,
        queries_by_id=queries_by_id,
        paired_query_outcomes=paired_by_q,
        replicate_collisions=_detect_replicate_collisions(
            actual_scales[: len(points)],
            replicates_by_scale,
            queries_by_id,
        ),
    )


def run_scaling_sweep(
    config: RunConfig | None = None,
    target_skill: str | None = None,
    scales: Sequence[int] | None = None,
    anchor: int | Sequence[str] | str | None = None,
    runtime: AgentRuntime | None = None,
    rivals_share: float = 0.5,
    noise_floor: float | None = None,
    workers: int = 1,
    skills: Sequence[Skill] | None = None,
    query_set: QuerySet | None = None,
    attempts: int | None = None,
    allow_truncation: bool = True,
    *,
    bootstrap_iterations: int | None = None,
    seed: int | None = None,
    catalog_replicates: int | None = None,
    on_scale_complete: Callable[[int, int, ScalingPoint, ScalingStudy], None] | None = None,
) -> ScalingStudy:
    """Execute multi-scale catalog evaluation sweep and return scaling analysis."""
    effective_config = _prepare_sweep_config(
        config,
        attempts=attempts,
        bootstrap_iterations=bootstrap_iterations,
        seed=seed,
        catalog_replicates=catalog_replicates,
    )
    resolved_skills = (
        list(skills) if skills is not None else load_skills(effective_config.require_skills())
    )
    if len(resolved_skills) <= 1:
        msg = f"Scaling sweep requires at least 2 skills in corpus, got {len(resolved_skills)}"
        raise ValueError(msg)

    raw_query_set = (
        query_set if query_set is not None else load_query_set(effective_config.require_queries())
    )

    is_corpus = target_skill is None
    requested_scales = scales if scales is not None else effective_config.study.scales
    actual_scales = resolve_sweep_scales(len(resolved_skills), requested_scales)

    replicate_setups = [
        _setup_sweep_execution(
            is_corpus=is_corpus,
            target_skill=target_skill,
            anchor=anchor,
            resolved_skills=resolved_skills,
            actual_scales=actual_scales,
            raw_query_set=raw_query_set,
            effective_config=effective_config,
            rivals_share=rivals_share,
            seed_override=(
                effective_config.catalog.seed
                if rep_idx == 0
                else (effective_config.catalog.seed or effective_config.study.bootstrap_seed or 42)
                + rep_idx
            ),
        )
        for rep_idx in range(effective_config.study.catalog_replicates)
    ]
    primary_setup = replicate_setups[0]

    resolved_runtime = runtime or build_runtime(effective_config.runtime)
    if not allow_truncation and resolved_runtime.rations_catalog:
        for setup in replicate_setups:
            for cat in setup.catalogs:
                validate_catalog_fit(
                    resolved_runtime,
                    cat,
                    resolved_skills,
                    allow_truncation=False,
                )
    baseline_results: tuple[ProbeResult, ...] = ()
    latest_results: tuple[ProbeResult, ...] = ()
    points: list[ScalingPoint] = []
    final_decomp: DecompositionResult | None = None

    work_dir, temp_dir_obj = _resolve_sweep_work_dir(effective_config.study.workdir)

    ctx = _SweepContext(
        effective_config=effective_config,
        work_dir=work_dir,
        corpus_plan=primary_setup.corpus_plan,
        raw_query_set=raw_query_set,
        resolved_query_set=primary_setup.query_set,
        resolved_skills=resolved_skills,
        raw_query_map={q.query_id: q for q in raw_query_set.queries},
        is_corpus=is_corpus,
    )

    total_scales = len(actual_scales)
    shared_outcome_cache: dict[Any, Any] = {}
    scale_query_sums: dict[int, dict[str, _QueryOutcome]] = {}
    replicates_by_scale: dict[int, tuple[_ScaleReplicateRecord, ...]] = {}
    scoped_target = primary_setup.target if not is_corpus else None
    scale_workers = max(1, workers)

    for step_idx, scale in enumerate(actual_scales, start=1):
        (
            combined_results,
            installed_union,
            primary_query_set,
            scale_replicates,
        ) = _execute_scale_replicates(
            step_idx=step_idx,
            scale_workers=scale_workers,
            replicate_setups=replicate_setups,
            ctx=ctx,
            resolved_runtime=resolved_runtime,
            allow_truncation=allow_truncation,
            shared_outcome_cache=shared_outcome_cache,
            target=primary_setup.target,
        )
        replicates_by_scale[scale] = scale_replicates
        point, decomp = _build_scaling_point(
            scale=scale,
            catalog_id=primary_setup.catalogs[step_idx - 1].id,
            results=combined_results,
            resolved_query_set=primary_query_set,
            baseline_results=baseline_results,
            installed_skills=installed_union,
            target_skill=scoped_target,
            seed=effective_config.catalog.seed,
        )
        points.append(point)
        latest_results = combined_results
        scale_query_map = {q.query_id: q for q in primary_query_set.queries}
        truth_expected = {q.query_id: q.expected_skill for q in primary_query_set.queries}
        scale_query_sums[scale] = _extract_query_outcomes(
            combined_results,
            truth_expected,
            installed_union,
            target_skill=scoped_target,
            queries_by_id=scale_query_map,
        )

        if scale == actual_scales[0]:
            baseline_results = combined_results
        elif decomp is not None:
            final_decomp = decomp

        if on_scale_complete is not None:
            partial_study = _snapshot_scaling_study(
                target=primary_setup.target,
                is_corpus=is_corpus,
                actual_scales=actual_scales,
                points=points,
                noise_floor=noise_floor,
                baseline_results=baseline_results,
                latest_results=latest_results,
                total_skills=len(resolved_skills),
                decomp=final_decomp,
                anchor_skills=primary_setup.resolved_anchors,
                study_config=effective_config.study,
                scale_query_sums=scale_query_sums,
                queries_by_id=ctx.raw_query_map,
                replicates_by_scale=replicates_by_scale,
                scoped_target=scoped_target,
            )
            on_scale_complete(step_idx, total_scales, point, partial_study)

        if step_idx == 1 and point.all_probes_errored:
            break

    result = _snapshot_scaling_study(
        target=primary_setup.target,
        is_corpus=is_corpus,
        actual_scales=actual_scales,
        points=points,
        noise_floor=noise_floor,
        baseline_results=baseline_results,
        latest_results=latest_results,
        total_skills=len(resolved_skills),
        decomp=final_decomp,
        anchor_skills=primary_setup.resolved_anchors,
        study_config=effective_config.study,
        scale_query_sums=scale_query_sums,
        queries_by_id=ctx.raw_query_map,
        replicates_by_scale=replicates_by_scale,
        scoped_target=scoped_target,
    )
    if temp_dir_obj is not None:
        temp_dir_obj.cleanup()
    return result
