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
import random
import statistics
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, NamedTuple, Self

from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, model_validator

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
from reach.metrics import DecompositionResult, decompose_pass_rate_drop, score_trajectory
from reach.models import NO_SKILL, Catalog, CatalogMode, ProbeResult, Query, QueryKind, Skill
from reach.queries import QuerySet, load_query_set
from reach.run import Composition, conduct, validate_catalog_fit
from reach.runtime import AgentRuntime, build_runtime
from reach.uncertainty import cluster_wilson_interval, effective_sample_size

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

__all__ = [
    "PairedTrialOutcomes",
    "ScalingPoint",
    "ScalingStudy",
    "bootstrap_f1_ci",
    "compute_scaling_noise_floor",
    "find_kneedle_knee",
    "run_scaling_sweep",
]


logger = logging.getLogger(__name__)


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
        Field(
            default=None,
            description="Effective sample size adjusting for repeated attempts",
        ),
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

    model_config = ConfigDict(frozen=True)

    scale: int
    catalog_id: str
    pass_rate: float
    pass_rate_interval: tuple[float, float]
    recall: float = 0.0
    recall_interval: tuple[float, float] = (0.0, 1.0)
    precision: float = 0.0
    precision_interval: tuple[float, float] = (0.0, 1.0)
    internal_precision: float = 0.0
    external_distractor_precision: float | None = None
    abstention_rate: float | None = None
    abstention_interval: tuple[float, float] | None = None
    f1_score: float = 0.0
    f1_interval: tuple[float, float] = (0.0, 1.0)
    entrypoint_pass_rate: float | None = None
    entrypoint_f1_score: float | None = None
    in_scope_probes: int = 0
    negative_probes: int = 0
    disclosure_states: dict[str, int] = Field(default_factory=dict)
    delta_vs_baseline: float
    delta_context: float
    delta_shadowing: float
    probes_executed: int
    probes_failed: int = 0
    probes_errored: NonNegativeInt = 0
    prompt_tokens_mean: float | None = None
    duration_ms_mean: float = 0.0
    step_efficiency_mean: float = 0.0
    skill_f1_mean: float = 0.0

    @property
    def all_probes_errored(self) -> bool:
        """Return True when at least one probe ran and every probe failed with a runtime error."""
        return self.probes_executed > 0 and self.probes_errored == self.probes_executed


class ScalingStudy(BaseModel):
    """Represent multi-scale catalog scaling study and knee curvature analysis."""

    model_config = ConfigDict(frozen=True)

    target_skill: str | None = None
    is_corpus_sweep: bool = False
    scales: tuple[int, ...]
    points: tuple[ScalingPoint, ...]
    knee_scale: int | None = None
    knee_scale_interval: tuple[int, int] | None = None
    baseline_pass_rate: float
    final_pass_rate: float
    total_delta: float
    total_context_loss: float
    total_shadowing_loss: float
    noise_floor: float = 0.05
    total_corpus_skills: int = 0
    decomposition: DecompositionResult | None = None
    anchor_skills: tuple[str, ...] | None = None
    paired_outcomes: PairedTrialOutcomes | None = None

    @model_validator(mode="after")
    def _validate_target_skill_for_mode(self) -> Self:
        """Ensure targeted sweeps specify a target skill."""
        if not self.is_corpus_sweep and self.target_skill is None:
            msg = "Targeted scaling sweep requires target_skill to be specified."
            raise ValueError(msg)
        return self


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
        mean=round(mean, _PAVA_DECIMAL_PRECISION),
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
        n_raw = max(1, paired_outcomes.total_paired)
        n_eff = max(1, paired_outcomes.effective_paired or n_raw)
        n10, n01 = paired_outcomes.n10, paired_outcomes.n01
        # Asymptotic McNemar variance scaled by cluster survey design effect (DEFF = n_raw / n_eff)
        # Var_cluster = Var_raw * DEFF = (n10 + n01 - (n10 - n01)^2 / n_raw) / (n_raw * n_eff)
        var_num = max(0.0, float(n10 + n01) - ((float(n10 - n01) ** 2) / n_raw))
        var_paired = var_num / (float(n_raw) * float(n_eff))
        se_paired = math.sqrt(var_paired)
        floor = diff_noise_floor(se_paired / 2.0, se_paired / 2.0, confidence, noise_inflation)
        return max(_MIN_NOISE_FLOOR, floor)

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
) -> float:
    """Resolve user-configured or diff-derived scaling noise floor."""
    if noise_floor is not None:
        return noise_floor
    if len(points) >= _MIN_DIFF_POINTS:
        return round(
            compute_scaling_noise_floor(
                points[0].pass_rate,
                points[-1].pass_rate,
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
    auto_smooth: bool = False,
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
    diffs_below = [(y0 + x * (y_end - y0)) - y for x, y in zip(norm_x, norm_y, strict=True)]
    diffs_above = [y - (y0 + x * (y_end - y0)) for x, y in zip(norm_x, norm_y, strict=True)]

    max_below = max(diffs_below[1:-1])
    max_above = max(diffs_above[1:-1])
    min_prominence = max(0.05, (noise_floor * 0.5) / y_range)

    if max_above >= max_below and max_above > min_prominence:
        knee_idx = 1 + diffs_above[1:-1].index(max_above)
        return k_vals[knee_idx]
    if max_below > min_prominence:
        knee_idx = 1 + diffs_below[1:-1].index(max_below)
        return k_vals[knee_idx]
    return None


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
    hit = (
        (expected in scored_seq) if trajectory else (bool(scored_seq) and scored_seq[0] == expected)
    )
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
        invoked_target = (
            (target_skill in scored_seq)
            if trajectory
            else (bool(scored_seq) and scored_seq[0] == target_skill)
        )
        is_fp = not result.error and invoked_target and expected != target_skill
        return is_tp, is_fp, is_fn

    is_tp = expected is not None and hit
    is_fn = expected is not None and not is_tp
    first_invoked = scored_seq[0] if scored_seq else NO_SKILL
    is_fp = not result.error and not hit and first_invoked in installed_skills
    return is_tp, is_fp, is_fn


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
    """Compute cluster-bootstrap confidence interval for micro F1 score clustered by query."""
    if not results or iterations <= 0:
        return (0.0, 1.0)

    outcomes_by_query: dict[str, list[tuple[int, int, int]]] = defaultdict(list)
    for r in results:
        is_tp, is_fp, is_fn = _classify_probe_outcome(
            r,
            truth.get(r.query_id),
            installed_skills,
            target_skill=target_skill,
            query=queries_by_id.get(r.query_id) if queries_by_id is not None else None,
            trajectory=trajectory,
        )
        outcomes_by_query[r.query_id].append((int(is_tp), int(is_fp), int(is_fn)))

    qids = list(outcomes_by_query.keys())
    m_queries = len(qids)
    if m_queries == 0:
        return (0.0, 1.0)

    query_sums = [
        (
            sum(tp for tp, _, _ in outcomes_by_query[qid]),
            sum(fp for _, fp, _ in outcomes_by_query[qid]),
            sum(fn for _, _, fn in outcomes_by_query[qid]),
        )
        for qid in qids
    ]

    rng = random.Random(seed)  # noqa: S311
    f1_boots: list[float] = []

    for _ in range(iterations):
        sample_sums = [rng.choice(query_sums) for _ in range(m_queries)]
        tp_s = sum(s[0] for s in sample_sums)
        fp_s = sum(s[1] for s in sample_sums)
        fn_s = sum(s[2] for s in sample_sums)
        denom = 2 * tp_s + fp_s + fn_s
        f1_boots.append(2.0 * tp_s / denom if denom > 0 else 0.0)

    f1_boots.sort()
    low_idx = max(0, int(iterations * 0.025))
    high_idx = min(int(iterations * 0.975), iterations - 1)
    return (round(f1_boots[low_idx], 4), round(f1_boots[high_idx], 4))


def _extract_query_outcomes(
    results: Sequence[ProbeResult],
    truth: Mapping[str, str | None],
    installed_skills: set[str],
    target_skill: str | None = None,
    queries_by_id: Mapping[str, Query] | None = None,
    *,
    trajectory: bool = True,
) -> dict[str, tuple[int, int, int]]:
    """Aggregate (tp, fp, fn) counts grouped by query_id for a single scale."""
    outcomes: dict[str, tuple[int, int, int]] = defaultdict(lambda: (0, 0, 0))
    for r in results:
        is_tp, is_fp, is_fn = _classify_probe_outcome(
            r,
            truth.get(r.query_id),
            installed_skills,
            target_skill=target_skill,
            query=queries_by_id.get(r.query_id) if queries_by_id is not None else None,
            trajectory=trajectory,
        )
        cur_tp, cur_fp, cur_fn = outcomes[r.query_id]
        outcomes[r.query_id] = (cur_tp + int(is_tp), cur_fp + int(is_fp), cur_fn + int(is_fn))
    return dict(outcomes)


_FUZZY_MATCH_CUTOFF = 0.5
_MAX_FUZZY_SUGGESTIONS = 3
_ADAPTIVE_WORKER_REFERENCE_SCALE = 25


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

    target_queries = tuple(
        q
        for q in raw_query_set.queries
        if q.expected_skill == target or q.is_out_of_scope or q.kind == QueryKind.NEIGHBOR_NEGATIVE
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
    delta_context: float
    delta_shadowing: float
    decomposition: DecompositionResult | None


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
    executed = len(results)
    hits = 0
    for r in results:
        if r.error:
            continue
        q = queries_by_id.get(r.query_id)
        exp = q.expected_skill if q is not None else None
        if target_skill is not None:
            is_tp, is_fp, _ = _classify_probe_outcome(
                r, exp, set(), target_skill, query=q, trajectory=trajectory
            )
            if (exp == target_skill and is_tp) or (exp != target_skill and not is_fp):
                hits += 1
        else:
            hit, _ = _evaluate_probe_trajectory(r, exp, q, trajectory=trajectory)
            if hit:
                hits += 1
    fails = executed - hits
    pass_rate = hits / executed if executed else 0.0
    attempts = _estimate_query_attempts(results)
    interval_obj = cluster_wilson_interval(hits, executed, attempts=attempts)
    pass_interval = (interval_obj.low, interval_obj.high) if interval_obj else (0.0, 1.0)
    return _ScalePassRate(
        executed=executed,
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


def _compute_f1_score(precision: float, recall: float) -> float:
    """Compute standard harmonic mean F1 score from precision and recall."""
    denom = precision + recall
    return 2.0 * precision * recall / denom if denom > 0 else 0.0


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
    in_scope = [
        r
        for r in results
        if (queries_by_id[r.query_id].expected_skill if r.query_id in queries_by_id else None)
        is not None
    ]
    negative = [
        r
        for r in results
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
    attempts = _estimate_query_attempts(results)
    internal_prec, ext_prec, precision, precision_interval = _compute_precision_metrics(
        tp, internal_fp, fp_distractor, bool(negative), attempts=attempts
    )

    f1 = _compute_f1_score(precision, recall)
    if compute_ci:
        truth_expected = {qid: q.expected_skill for qid, q in queries_by_id.items()}
        f1_ci = bootstrap_f1_ci(
            results,
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
    for r in results:
        if r.error:
            continue
        q = queries_by_id.get(r.query_id)
        if q is not None:
            raw_seq = (
                tuple(r.invoked_skills)
                if r.invoked_skills
                else ((r.invoked_skill,) if r.invoked_skill is not None else ())
            )
            t_score = score_trajectory(q, raw_seq)
            step_effs.append(t_score.step_efficiency)
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
            delta_context=0.0,
            delta_shadowing=0.0,
            decomposition=None,
        )
    scoped_queries = (
        [q for q in queries if q.expected_skill == target_skill or q.is_out_of_scope]
        if target_skill is not None
        else list(queries)
    )
    scoped_ids = {q.id for q in scoped_queries}
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
        delta_context=decomp.delta_context,
        delta_shadowing=decomp.delta_shadowing,
        decomposition=decomp,
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
    queries_by_id = {q.id: q for q in queries}

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

    point = ScalingPoint(
        scale=scale,
        catalog_id=catalog_id,
        pass_rate=round(pass_stats.pass_rate, 4),
        pass_rate_interval=(
            round(pass_stats.pass_interval[0], 4),
            round(pass_stats.pass_interval[1], 4),
        ),
        recall=round(class_stats.recall, 4),
        recall_interval=(
            round(class_stats.recall_interval[0], 4),
            round(class_stats.recall_interval[1], 4),
        ),
        precision=round(class_stats.precision, 4),
        precision_interval=(
            round(class_stats.precision_interval[0], 4),
            round(class_stats.precision_interval[1], 4),
        ),
        internal_precision=round(class_stats.internal_precision, 4),
        external_distractor_precision=class_stats.external_distractor_precision,
        abstention_rate=class_stats.abstention_rate,
        abstention_interval=class_stats.abstention_interval,
        f1_score=round(class_stats.f1_score, 4),
        f1_interval=(
            round(class_stats.f1_interval[0], 4),
            round(class_stats.f1_interval[1], 4),
        ),
        entrypoint_pass_rate=round(entry_pass_stats.pass_rate, 4),
        entrypoint_f1_score=round(entry_class_stats.f1_score, 4),
        in_scope_probes=class_stats.in_scope_probes,
        negative_probes=class_stats.negative_probes,
        disclosure_states=telemetry.disclosure_states,
        delta_vs_baseline=round(decomp_stats.delta_vs_baseline, 4),
        delta_context=round(decomp_stats.delta_context, 4),
        delta_shadowing=round(decomp_stats.delta_shadowing, 4),
        probes_executed=pass_stats.executed,
        probes_failed=pass_stats.fails,
        probes_errored=sum(1 for r in results if r.error),
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
) -> RunConfig:
    """Apply CLI overrides to execution configuration."""
    cfg = config or RunConfig()
    plan_update = {"attempts": cfg.plan.resolve_sweep_attempts(attempts)}
    study_update: dict[str, object] = {}
    if bootstrap_iterations is not None:
        study_update["bootstrap_iterations"] = bootstrap_iterations
    if seed is not None:
        study_update["bootstrap_seed"] = seed

    cfg = cfg.model_copy(update={"plan": cfg.plan.model_copy(update=plan_update)})
    if study_update:
        cfg = cfg.model_copy(update={"study": cfg.study.model_copy(update=study_update)})
    return cfg


def _build_study_result(
    target: str | None,
    is_corpus: bool,
    evaluated_scales: Sequence[int],
    points: Sequence[ScalingPoint],
    knee: int | None,
    noise_floor: float,
    total_skills: int,
    decomp: DecompositionResult | None,
    anchor_skills: Sequence[str] | None = None,
    paired_outcomes: PairedTrialOutcomes | None = None,
    knee_interval: tuple[int, int] | None = None,
) -> ScalingStudy:
    """Construct finished ScalingStudy data model."""
    b_rate = points[0].pass_rate if points else 0.0
    f_rate = points[-1].pass_rate if points else 0.0
    t_delta = points[-1].delta_vs_baseline if len(points) > 1 else 0.0
    t_ctx = points[-1].delta_context if len(points) > 1 else 0.0
    t_shd = points[-1].delta_shadowing if len(points) > 1 else 0.0

    return ScalingStudy(
        target_skill=target,
        is_corpus_sweep=is_corpus,
        scales=tuple(evaluated_scales),
        points=tuple(points),
        knee_scale=knee,
        knee_scale_interval=knee_interval,
        baseline_pass_rate=b_rate,
        final_pass_rate=f_rate,
        total_delta=t_delta,
        total_context_loss=t_ctx,
        total_shadowing_loss=t_shd,
        noise_floor=noise_floor,
        total_corpus_skills=total_skills,
        decomposition=decomp,
        anchor_skills=tuple(anchor_skills) if anchor_skills is not None else None,
        paired_outcomes=paired_outcomes,
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
    if queried_skills:
        return tuple(s.name for s in queried_skills)
    logger.warning(
        "Provided query set contains 0 benchmark queries for resident skills; "
        "falling back to unqueried corpus medoids."
    )
    return full_medoids


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

    medoid_count: int | None = None
    if isinstance(requested_anchor, int):
        medoid_count = requested_anchor
    elif isinstance(requested_anchor, str) and requested_anchor.isdigit():
        medoid_count = int(requested_anchor)
    elif requested_anchor is None:
        medoid_count = actual_scales[0]

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


def _scale_adaptive_workers(
    base_workers: int,
    scale: int,
    reference_scale: int = _ADAPTIVE_WORKER_REFERENCE_SCALE,
) -> int:
    """Taper concurrent worker count inversely with catalog size past reference scale."""
    if base_workers <= 1 or scale <= reference_scale:
        return max(1, base_workers)
    return max(1, round(base_workers * (reference_scale / scale)))


def _setup_sweep_execution(
    is_corpus: bool,
    target_skill: str | None,
    anchor: int | Sequence[str] | str | None,
    resolved_skills: Sequence[Skill],
    actual_scales: Sequence[int],
    raw_query_set: QuerySet,
    effective_config: RunConfig,
    rivals_share: float,
) -> tuple[str | None, list[Catalog], QuerySet, CorpusScalingPlan | None, tuple[str, ...] | None]:
    """Configure catalogs, query sets, and scaling plan for sweep execution."""
    if is_corpus:
        requested_anchor = anchor if anchor is not None else effective_config.study.anchor
        resolved_anchors = _resolve_anchor_skills(
            requested_anchor,
            resolved_skills,
            actual_scales,
            query_set=raw_query_set,
            clamp_to_queried=not effective_config.study.auto_queries,
        )
        plan = CorpusScalingPlan.create(
            skills=resolved_skills,
            scales=actual_scales,
            anchor_skills=resolved_anchors,
        )
        return None, list(plan.catalogs), raw_query_set, plan, resolved_anchors

    target, query_set = _resolve_sweep_target_and_queries(
        resolved_skills, raw_query_set, target_skill
    )
    catalogs = build_scaling_catalogs(
        skills=resolved_skills,
        target_skill=target,
        scales=actual_scales,
        rivals_share=rivals_share,
        seed=effective_config.catalog.seed,
    )
    return target, catalogs, query_set, None, None


def _calculate_paired_outcomes(
    baseline_results: Sequence[ProbeResult],
    final_results: Sequence[ProbeResult],
    queries_by_id: Mapping[str, Query],
    target_skill: str | None = None,
) -> PairedTrialOutcomes | None:
    """Calculate discordant pair counts (n10, n01) between baseline and final scale runs."""
    if not baseline_results or not final_results:
        return None

    def probe_hit(r: ProbeResult) -> bool:
        if r.error:
            return False
        q = queries_by_id.get(r.query_id)
        exp = q.expected_skill if q is not None else None
        if target_skill is not None:
            is_tp, is_fp, _ = _classify_probe_outcome(
                r, exp, set(), target_skill, query=q, trajectory=True
            )
            return (exp == target_skill and is_tp) or (exp != target_skill and not is_fp)
        hit, _ = _evaluate_probe_trajectory(r, exp, q, trajectory=True)
        return hit

    base_map = {(r.query_id, r.attempt): probe_hit(r) for r in baseline_results if not r.error}
    final_map = {(r.query_id, r.attempt): probe_hit(r) for r in final_results if not r.error}
    common_keys = set(base_map.keys()) & set(final_map.keys())
    if not common_keys:
        return None

    n10 = sum(1 for k in common_keys if base_map[k] and not final_map[k])
    n01 = sum(1 for k in common_keys if not base_map[k] and final_map[k])
    unique_qids = {k[0] for k in common_keys}
    attempts = max(1, round(len(common_keys) / max(1, len(unique_qids))))
    neff = effective_sample_size(len(common_keys), attempts=attempts)
    return PairedTrialOutcomes(
        n10=n10, n01=n01, total_paired=len(common_keys), effective_paired=neff
    )


def _bootstrap_knee_interval(
    scales: Sequence[int],
    points: Sequence[ScalingPoint],
    noise_floor: float,
    iterations: int = 200,
    seed: int = 42,
    scale_query_sums: Mapping[int, Mapping[str, tuple[int, int, int]]] | None = None,
) -> tuple[int, int] | None:
    """Calculate bootstrap confidence interval for knee scale k* using weighted PAVA."""
    if len(points) < _MIN_DIFF_POINTS or iterations <= 0:
        return None

    rng = random.Random(seed)  # noqa: S311
    knees: list[int] = []

    weights = [
        1.0 / max(1e-4, ((p.f1_interval[1] - p.f1_interval[0]) / 3.92) ** 2)
        if (p.f1_interval[1] > p.f1_interval[0])
        else 1.0
        for p in points
    ]

    # Non-parametric cluster bootstrap when scale query outcomes are available
    if scale_query_sums and all(s in scale_query_sums for s in scales):
        common_qids = list(scale_query_sums[scales[0]].keys())
        if common_qids:
            for _ in range(iterations):
                sample_qids = [rng.choice(common_qids) for _ in range(len(common_qids))]
                resampled_curve: list[float] = []
                for s in scales:
                    tp = sum(scale_query_sums[s].get(q, (0, 0, 0))[0] for q in sample_qids)
                    fp = sum(scale_query_sums[s].get(q, (0, 0, 0))[1] for q in sample_qids)
                    fn = sum(scale_query_sums[s].get(q, (0, 0, 0))[2] for q in sample_qids)
                    denom = 2 * tp + fp + fn
                    f1 = (2.0 * tp / denom) if denom > 0 else 0.0
                    resampled_curve.append(f1)

                k = find_kneedle_knee(
                    scales,
                    resampled_curve,
                    noise_floor=noise_floor,
                    weights=weights,
                    auto_smooth=True,
                )
                if k is not None:
                    knees.append(k)

            if knees:
                knees.sort()
                low_idx = int(len(knees) * 0.025)
                high_idx = min(int(len(knees) * 0.975), len(knees) - 1)
                return (knees[low_idx], knees[high_idx])
            return None

    # Parametric perturbation fallback
    for _ in range(iterations):
        perturbed_rates: list[float] = []
        sampled_weights: list[float] = []
        for p in points:
            ci_width = max(0.001, p.f1_interval[1] - p.f1_interval[0])
            se = max(0.005, ci_width / 3.92)
            sampled_f1 = max(0.0, min(1.0, rng.gauss(p.f1_score, se)))
            perturbed_rates.append(sampled_f1)
            sampled_weights.append(1.0 / (se * se))

        k = find_kneedle_knee(
            scales,
            perturbed_rates,
            noise_floor=noise_floor,
            weights=sampled_weights,
            auto_smooth=True,
        )
        if k is not None:
            knees.append(k)

    if not knees:
        return None

    knees.sort()
    low_idx = int(len(knees) * 0.025)
    high_idx = min(int(len(knees) * 0.975), len(knees) - 1)
    return (knees[low_idx], knees[high_idx])


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
    scale_query_sums: Mapping[int, Mapping[str, tuple[int, int, int]]] | None = None,
) -> ScalingStudy:
    """Compute effective noise floor and knee and construct a ScalingStudy."""
    effective_noise_floor = _compute_effective_noise_floor(
        noise_floor, points, baseline_count, paired_outcomes=paired_outcomes
    )
    evaluated_scales = actual_scales[: len(points)]
    rate_curve = [p.f1_score for p in points] if is_corpus else [p.pass_rate for p in points]
    weights = [
        1.0 / max(1e-4, ((p.f1_interval[1] - p.f1_interval[0]) / 3.92) ** 2)
        if (p.f1_interval[1] > p.f1_interval[0])
        else 1.0
        for p in points
    ]
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
    knee_int = (
        _bootstrap_knee_interval(
            evaluated_scales,
            points,
            effective_noise_floor,
            iterations=bootstrap_iterations,
            seed=effective_seed,
            scale_query_sums=scale_query_sums,
        )
        if is_corpus
        else None
    )
    return _build_study_result(
        target=target,
        is_corpus=is_corpus,
        evaluated_scales=evaluated_scales,
        points=points,
        knee=knee,
        noise_floor=effective_noise_floor,
        total_skills=total_skills,
        decomp=decomp,
        anchor_skills=anchor_skills,
        paired_outcomes=paired_outcomes,
        knee_interval=knee_int,
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
    bootstrap_iterations: int | None = None,
    seed: int | None = None,
    on_scale_complete: Callable[[int, int, ScalingPoint, ScalingStudy], None] | None = None,
) -> ScalingStudy:
    """Execute multi-scale catalog evaluation sweep and return scaling analysis."""
    effective_config = _prepare_sweep_config(
        config,
        attempts=attempts,
        bootstrap_iterations=bootstrap_iterations,
        seed=seed,
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

    target, catalogs, resolved_query_set, corpus_plan, resolved_anchors = _setup_sweep_execution(
        is_corpus=is_corpus,
        target_skill=target_skill,
        anchor=anchor,
        resolved_skills=resolved_skills,
        actual_scales=actual_scales,
        raw_query_set=raw_query_set,
        effective_config=effective_config,
        rivals_share=rivals_share,
    )

    resolved_runtime = runtime or build_runtime(effective_config.runtime)
    if not allow_truncation and resolved_runtime.rations_catalog:
        for cat in catalogs:
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

    raw_query_map = {q.id: q for q in raw_query_set.queries}
    ctx = _SweepContext(
        effective_config=effective_config,
        work_dir=work_dir,
        corpus_plan=corpus_plan,
        raw_query_set=raw_query_set,
        resolved_query_set=resolved_query_set,
        resolved_skills=resolved_skills,
        raw_query_map=raw_query_map,
        is_corpus=is_corpus,
    )

    reference_scale = max(actual_scales[0], _ADAPTIVE_WORKER_REFERENCE_SCALE)
    total_scales = len(actual_scales)
    shared_outcome_cache: dict[Any, Any] = {}
    scale_query_sums: dict[int, dict[str, tuple[int, int, int]]] = {}
    for step_idx, (scale, catalog) in enumerate(zip(actual_scales, catalogs, strict=True), start=1):
        scale_config, scale_query_set, composed = _prepare_scale_iteration(ctx, catalog)

        scale_workers = _scale_adaptive_workers(workers, scale, reference_scale=reference_scale)
        outcome = conduct(
            config=scale_config,
            runtime=resolved_runtime,
            composed=composed,
            allow_truncation=allow_truncation,
            append_across_arms=True,
            workers=scale_workers,
            outcome_cache=shared_outcome_cache,
        )

        point, decomp = _build_scaling_point(
            scale=scale,
            catalog_id=catalog.id,
            results=outcome.results,
            resolved_query_set=scale_query_set,
            baseline_results=baseline_results,
            installed_skills=set(catalog.skills),
            target_skill=target if not is_corpus else None,
            seed=effective_config.catalog.seed,
        )
        points.append(point)
        latest_results = outcome.results
        scale_query_map = {q.id: q for q in scale_query_set.queries}
        truth_expected = {q.id: q.expected_skill for q in scale_query_set.queries}
        scale_query_sums[scale] = _extract_query_outcomes(
            outcome.results,
            truth_expected,
            set(catalog.skills),
            target_skill=target if not is_corpus else None,
            queries_by_id=scale_query_map,
        )

        if scale == actual_scales[0]:
            baseline_results = outcome.results
        elif decomp is not None:
            final_decomp = decomp

        if on_scale_complete is not None:
            partial_paired = _calculate_paired_outcomes(
                baseline_results,
                latest_results,
                ctx.raw_query_map,
                target_skill=target if not is_corpus else None,
            )
            partial_study = _assemble_scaling_study(
                target=target,
                is_corpus=is_corpus,
                actual_scales=actual_scales,
                points=points,
                noise_floor=noise_floor,
                baseline_count=len(baseline_results),
                total_skills=len(resolved_skills),
                decomp=final_decomp,
                anchor_skills=resolved_anchors,
                paired_outcomes=partial_paired,
                study_config=effective_config.study,
                scale_query_sums=scale_query_sums,
            )
            on_scale_complete(step_idx, total_scales, point, partial_study)

        if step_idx == 1 and point.all_probes_errored:
            break

    final_paired = _calculate_paired_outcomes(
        baseline_results,
        latest_results,
        ctx.raw_query_map,
        target_skill=target if not is_corpus else None,
    )
    result = _assemble_scaling_study(
        target=target,
        is_corpus=is_corpus,
        actual_scales=actual_scales,
        points=points,
        noise_floor=noise_floor,
        baseline_count=len(baseline_results),
        total_skills=len(resolved_skills),
        decomp=final_decomp,
        anchor_skills=resolved_anchors,
        paired_outcomes=final_paired,
        study_config=effective_config.study,
        scale_query_sums=scale_query_sums,
    )
    if temp_dir_obj is not None:
        temp_dir_obj.cleanup()
    return result
