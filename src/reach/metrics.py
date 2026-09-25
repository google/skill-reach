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

"""Compute standard evaluation and routing metrics for skill selection probes."""

from __future__ import annotations

import random
import statistics
from collections import Counter, defaultdict
from typing import TYPE_CHECKING, Final, NamedTuple, Self

from pydantic import BaseModel, ConfigDict, computed_field, model_validator

from reach.models import (
    NO_SKILL,
    InvocationPattern,
    ProbeResult,
    Query,
    Skill,
)
from reach.uncertainty import (
    DEFAULT_CONFIDENCE,
    Interval,
    bootstrap_quantiles,
    wilson_interval,
)

#: Minimum number of skill invocations required to trace transition precursors.
MIN_TRANSITION_STEPS: Final = 2

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

type PrecursorTransitions = dict[tuple[str, str], list[int]]
type RequirementSynonyms = frozenset[str]
type ConfusionMatrix = Counter[tuple[str, str | None]]

__all__ = [
    "ClassMetrics",
    "ClassificationReport",
    "DecompositionResult",
    "PrecursorEdge",
    "TrajectoryScore",
    "classification_report",
    "classify_invocation_pattern",
    "collisions",
    "compute_f1",
    "compute_precursor_graph",
    "confusion",
    "consistency",
    "consistency_counts",
    "decompose_pass_rate_drop",
    "labeled_pairs",
    "score_trajectory",
    "trajectory_scores",
]


def classify_invocation_pattern(
    query: Query,
    invoked_skills: Sequence[str],
) -> InvocationPattern:
    """Classify trajectory invocation behavior relative to query ground truth."""
    invoked = query.scored_invocations(invoked_skills)
    if query.is_out_of_scope:
        return (
            InvocationPattern.CORRECT_ABSTENTION
            if not invoked
            else InvocationPattern.UNWANTED_TRIGGER
        )
    if not invoked:
        return InvocationPattern.ABANDONED
    valid_targets = query.valid_skills
    if set(invoked) <= valid_targets:
        return InvocationPattern.ORACLE_ONLY
    if set(invoked) & valid_targets:
        return InvocationPattern.MIXED_ORACLE
    return InvocationPattern.DISTRACTOR_HIJACK


def compute_f1(precision: float, recall: float) -> float:
    """Calculate the harmonic mean of precision and recall."""
    total = precision + recall
    return 2 * precision * recall / total if total else 0.0


class TrajectoryScore(BaseModel):
    """Evaluation outcomes for a single query across its invocation trajectory."""

    model_config = ConfigDict(frozen=True)

    entrypoint_hit: bool
    trajectory_hit: bool
    step_efficiency: float
    skill_f1: float
    redundancy: int


def score_trajectory(
    query: Query,
    invoked_skills: Sequence[str],
) -> TrajectoryScore:
    """Evaluate an observed skill trajectory against query target skill and acceptable skills."""
    invoked_seq = query.scored_invocations(invoked_skills)

    if query.is_out_of_scope:
        abstained = len(invoked_seq) == 0
        return TrajectoryScore(
            entrypoint_hit=abstained,
            trajectory_hit=abstained,
            step_efficiency=1.0 if abstained else 0.0,
            skill_f1=1.0 if abstained else 0.0,
            redundancy=len(invoked_seq),
        )

    valid_targets = query.valid_skills
    invoked_set = set(invoked_seq)
    entry_hit = bool(invoked_seq and invoked_seq[0] in valid_targets)
    traj_hit = bool(invoked_set & valid_targets)

    first_rank = next(
        (i + 1 for i, s in enumerate(invoked_seq) if s in valid_targets),
        None,
    )
    mrr = (1.0 / first_rank) if first_rank else 0.0

    prec = (
        (len(invoked_set & valid_targets) / len(invoked_set)) if (traj_hit and invoked_set) else 0.0
    )
    rec = 1.0 if traj_hit else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) else 0.0

    excess = max(0, len(invoked_seq) - 1)

    return TrajectoryScore(
        entrypoint_hit=entry_hit,
        trajectory_hit=traj_hit,
        step_efficiency=round(mrr, 4),
        skill_f1=round(f1, 4),
        redundancy=excess,
    )


class ClassMetrics(BaseModel):
    """Report precision, recall, support, and F1 metrics for a single skill class."""

    model_config = ConfigDict(frozen=True)

    false_negatives: int
    false_positives: int
    label: str
    predicted: int
    support: int
    true_positives: int
    trajectory_true_positives: int = 0

    @model_validator(mode="after")
    def _ensure_trajectory_at_least_top1(self) -> Self:
        """Ensure trajectory true positives are at least top-1 true positives."""
        if self.trajectory_true_positives < self.true_positives:
            object.__setattr__(self, "trajectory_true_positives", self.true_positives)
        return self

    @property
    def precision(self) -> float:
        """Calculate precision (true positives / predicted positives)."""
        denominator = self.true_positives + self.false_positives
        return self.true_positives / denominator if denominator else 0.0

    @property
    def recall(self) -> float:
        """Calculate recall (true positives / ground truth support)."""
        denominator = self.true_positives + self.false_negatives
        return self.true_positives / denominator if denominator else 0.0

    @property
    def trajectory_recall(self) -> float:
        """Calculate trajectory recall (trajectory true positives / ground truth support)."""
        return self.trajectory_true_positives / self.support if self.support else 0.0

    @property
    def f1(self) -> float:
        """Calculate F1 score (harmonic mean of precision and recall)."""
        return compute_f1(self.precision, self.recall)

    @property
    def recall_interval(self) -> Interval | None:
        """Calculate the Wilson confidence interval for recall."""
        return wilson_interval(
            self.true_positives,
            self.true_positives + self.false_negatives,
        )

    @property
    def precision_interval(self) -> Interval | None:
        """Calculate the Wilson confidence interval for precision."""
        return wilson_interval(
            self.true_positives,
            self.true_positives + self.false_positives,
        )


class ClassificationReport(BaseModel):
    """Hold aggregated classification and routing metrics for an evaluation run."""

    model_config = ConfigDict(frozen=True)

    errors: int
    macro_f1: float
    macro_precision: float
    macro_recall: float
    per_class: tuple[ClassMetrics, ...]
    probes: int
    scored: int
    abstentions: int = 0
    false_abstentions: int = 0
    in_scope: int = 0
    out_of_scope: int = 0
    out_of_scope_detected: int = 0
    top1_hits: int = 0
    entrypoint_hits: int = 0
    trajectory_hits: int = 0
    step_efficiency: float = 0.0
    skill_f1: float = 0.0
    redundancy: float = 0.0

    @computed_field
    @property
    def entrypoint_accuracy(self) -> float:
        """Calculate entrypoint accuracy across scored probes."""
        return self.entrypoint_hits / self.scored if self.scored else 0.0

    @computed_field
    @property
    def entrypoint_interval(self) -> Interval | None:
        """Calculate the Wilson confidence interval for entrypoint accuracy."""
        return wilson_interval(self.entrypoint_hits, self.scored)

    @computed_field
    @property
    def trajectory_reachability(self) -> float:
        """Calculate trajectory reachability across scored probes."""
        return self.trajectory_hits / self.scored if self.scored else 0.0

    @computed_field
    @property
    def trajectory_interval(self) -> Interval | None:
        """Calculate the Wilson confidence interval for trajectory reachability."""
        return wilson_interval(self.trajectory_hits, self.scored)

    @computed_field
    @property
    def top1_accuracy(self) -> float:
        """Calculate top-1 accuracy across scored probes."""
        return self.top1_hits / self.scored if self.scored else 0.0

    @computed_field
    @property
    def top1_interval(self) -> Interval | None:
        """Calculate the Wilson confidence interval for top-1 accuracy."""
        return wilson_interval(self.top1_hits, self.scored)

    @computed_field
    @property
    def abstention_rate(self) -> float:
        """Calculate the overall abstention rate across scored probes."""
        return self.abstentions / self.scored if self.scored else 0.0

    @computed_field
    @property
    def abstention_interval(self) -> Interval | None:
        """Calculate the Wilson confidence interval for overall abstention rate."""
        return wilson_interval(self.abstentions, self.scored)

    @computed_field
    @property
    def false_abstention_rate(self) -> float:
        """Calculate the false abstention rate over in-scope queries."""
        return self.false_abstentions / self.in_scope if self.in_scope else 0.0

    @computed_field
    @property
    def false_abstention_interval(self) -> Interval | None:
        """Calculate the Wilson confidence interval for false abstention rate."""
        return wilson_interval(self.false_abstentions, self.in_scope)

    @computed_field
    @property
    def out_of_scope_detection(self) -> float | None:
        """Calculate the out-of-scope detection accuracy."""
        if not self.out_of_scope:
            return None
        return self.out_of_scope_detected / self.out_of_scope

    @computed_field
    @property
    def out_of_scope_interval(self) -> Interval | None:
        """Calculate the Wilson confidence interval for out-of-scope detection."""
        return wilson_interval(self.out_of_scope_detected, self.out_of_scope)

    def by_label(self, label: str) -> ClassMetrics:
        """Return per-class metrics for a specified label."""
        for entry in self.per_class:
            if entry.label == label:
                return entry
        msg = f"no metrics for label {label!r}"
        raise KeyError(msg)

    def worst_recall(self, limit: int = 5) -> tuple[ClassMetrics, ...]:
        """Return classes with the lowest recall scores."""
        real = [c for c in self.per_class if c.label != NO_SKILL and c.support]
        return tuple(sorted(real, key=lambda c: (c.recall, c.label))[:limit])

    def top_attractors(self, limit: int = 5) -> tuple[ClassMetrics, ...]:
        """Return classes receiving the highest false-positive traffic."""
        real = [c for c in self.per_class if c.label != NO_SKILL and c.false_positives]
        return tuple(sorted(real, key=lambda c: (-c.false_positives, c.label))[:limit])


def _paired(
    results: Sequence[ProbeResult],
    queries: Sequence[Query],
) -> list[tuple[Query, ProbeResult]]:
    """Pair each usable result with its corresponding query."""
    truth = {q.id: q for q in queries}
    unknown = {r.query_id for r in results} - truth.keys()
    if unknown:
        msg = f"results reference unlabeled queries: {sorted(unknown)}"
        raise KeyError(msg)
    return [(truth[r.query_id], r) for r in results if not r.error]


def labeled_pairs(
    results: Sequence[ProbeResult],
    queries: Sequence[Query],
) -> tuple[list[str], list[str]]:
    """Extract aligned ground truth and predicted label sequences."""
    pairs = _paired(results, queries)
    return (
        [query.truth_label for query, _ in pairs],
        [query.effective_predicted_label(result) for query, result in pairs],
    )


def _label_universe(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str] | None = None,
) -> list[str]:
    """Determine the universe of classification labels."""
    if labels is not None:
        return list(labels)
    return sorted(set(y_true) | set(y_pred))


def _class_metrics(
    label: str,
    y_true: Sequence[str],
    y_pred: Sequence[str],
    trajectory_tp: int | None = None,
) -> ClassMetrics:
    """Compute precision, recall, and support metrics for a single label."""
    tp = sum(t == label and p == label for t, p in zip(y_true, y_pred, strict=True))
    fp = sum(t != label and p == label for t, p in zip(y_true, y_pred, strict=True))
    fn = sum(t == label and p != label for t, p in zip(y_true, y_pred, strict=True))
    return ClassMetrics(
        label=label,
        support=sum(t == label for t in y_true),
        predicted=sum(p == label for p in y_pred),
        true_positives=tp,
        trajectory_true_positives=trajectory_tp if trajectory_tp is not None else tp,
        false_positives=fp,
        false_negatives=fn,
    )


def _build_per_class(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    universe: Sequence[str],
    trajectory_hits_by_label: Mapping[str, int] | None = None,
) -> tuple[ClassMetrics, ...]:
    """Compute per-class metrics across all labels in the universe."""
    return tuple(
        _class_metrics(
            label,
            y_true,
            y_pred,
            trajectory_tp=trajectory_hits_by_label.get(label, 0)
            if trajectory_hits_by_label is not None
            else None,
        )
        for label in universe
    )


def _mean(values: Sequence[float]) -> float:
    """Calculate average of a sequence, treating empty as zero."""
    return statistics.fmean(values) if values else 0.0


def _macro_averages(
    per_class: Sequence[ClassMetrics],
) -> tuple[float, float, float]:
    """Calculate macro precision over active classes and recall/f1 over classes with support."""
    active = [c for c in per_class if c.support or c.predicted]
    present = [c for c in per_class if c.support]
    return (
        _mean([c.precision for c in active]),
        _mean([c.recall for c in present]),
        _mean([c.f1 for c in present]),
    )


def _scope_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
) -> tuple[int, int, int, int]:
    """Compute in-scope and out-of-scope counts along with abstention detections."""
    in_scope = [(t, p) for t, p in zip(y_true, y_pred, strict=True) if t != NO_SKILL]
    out_of_scope = [(t, p) for t, p in zip(y_true, y_pred, strict=True) if t == NO_SKILL]
    false_abstentions = sum(p == NO_SKILL for _, p in in_scope)
    out_of_scope_detected = sum(p == NO_SKILL for _, p in out_of_scope)
    return len(in_scope), false_abstentions, len(out_of_scope), out_of_scope_detected


def classification_report(
    results: Sequence[ProbeResult],
    queries: Sequence[Query],
    labels: Sequence[str] | None = None,
) -> ClassificationReport:
    """Generate a comprehensive classification report across all probe results."""
    pairs = _paired(results, queries)
    y_true = [q.truth_label for q, _ in pairs]
    y_pred = [q.effective_predicted_label(r) for q, r in pairs]
    universe = _label_universe(y_true, y_pred, labels)
    traj_scores = [score_trajectory(q, r.invoked_skills) for q, r in pairs]
    traj_hits_by_label = Counter(
        t for t, s in zip(y_true, traj_scores, strict=True) if s.trajectory_hit
    )
    per_class = _build_per_class(
        y_true,
        y_pred,
        universe,
        trajectory_hits_by_label=traj_hits_by_label,
    )
    precision, recall, f1 = _macro_averages(per_class)
    in_count, false_abs, out_count, out_detected = _scope_metrics(y_true, y_pred)

    entrypoint_hits = sum(1 for s in traj_scores if s.entrypoint_hit)
    trajectory_hits = sum(1 for s in traj_scores if s.trajectory_hit)
    step_eff = _mean([s.step_efficiency for s in traj_scores])
    s_f1 = _mean([s.skill_f1 for s in traj_scores])
    redundancy = _mean([float(s.redundancy) for s in traj_scores])

    return ClassificationReport(
        probes=len(results),
        errors=sum(1 for r in results if r.error),
        scored=len(y_true),
        per_class=per_class,
        top1_hits=sum(t == p for t, p in zip(y_true, y_pred, strict=True)),
        entrypoint_hits=entrypoint_hits,
        trajectory_hits=trajectory_hits,
        step_efficiency=round(step_eff, 4),
        skill_f1=round(s_f1, 4),
        redundancy=round(redundancy, 4),
        macro_precision=precision,
        macro_recall=recall,
        macro_f1=f1,
        abstentions=sum(p == NO_SKILL for p in y_pred),
        in_scope=in_count,
        false_abstentions=false_abs,
        out_of_scope=out_count,
        out_of_scope_detected=out_detected,
    )


def trajectory_scores(
    results: Sequence[ProbeResult],
    queries: Sequence[Query],
) -> dict[str, TrajectoryScore]:
    """Return mapping of query_id to its aggregated TrajectoryScore across replicates."""
    pairs = _paired(results, queries)
    by_query: dict[str, list[TrajectoryScore]] = defaultdict(list)
    for q, r in pairs:
        by_query[q.id].append(score_trajectory(q, r.invoked_skills))

    aggregated: dict[str, TrajectoryScore] = {}
    for q_id, scores in by_query.items():
        if len(scores) == 1:
            aggregated[q_id] = scores[0]
        else:
            aggregated[q_id] = TrajectoryScore(
                entrypoint_hit=sum(1 for s in scores if s.entrypoint_hit) * 2 >= len(scores),
                trajectory_hit=sum(1 for s in scores if s.trajectory_hit) * 2 >= len(scores),
                step_efficiency=round(_mean([s.step_efficiency for s in scores]), 4),
                skill_f1=round(_mean([s.skill_f1 for s in scores]), 4),
                redundancy=round(_mean([float(s.redundancy) for s in scores])),
            )
    return aggregated


def consistency_counts(
    results: Sequence[ProbeResult],
    queries: Sequence[Query],
) -> tuple[int, int]:
    """Count unanimous and total observed queries across attempt replicates."""
    _paired(results, queries)
    grouped: dict[str, set[str]] = {}
    for result in results:
        if not result.error:
            grouped.setdefault(result.query_id, set()).add(result.predicted_label)
    return sum(len(picks) == 1 for picks in grouped.values()), len(grouped)


def consistency(results: Sequence[ProbeResult], queries: Sequence[Query]) -> float:
    """Calculate the fraction of queries with unanimous selection outcomes."""
    unanimous, observed = consistency_counts(results, queries)
    return unanimous / observed if observed else 0.0


def confusion(
    results: Sequence[ProbeResult],
    queries: Sequence[Query],
) -> Counter[tuple[str, str | None]]:
    """Count occurrences of expected-to-invoked skill selection pairs."""
    truth = {q.id: q for q in queries}
    pairs: Counter[tuple[str, str | None]] = Counter()
    for result in results:
        if result.error:
            continue
        query = truth.get(result.query_id)
        if query is None:
            continue
        effective_invoked = query.effective_invoked_skill(result)
        pairs[(query.truth_label, effective_invoked)] += 1
    return pairs


def collisions(
    results: Sequence[ProbeResult],
    queries: Sequence[Query],
) -> Counter[tuple[str, str]]:
    """Count misroutes between skill pairs."""
    truth = {q.id: q for q in queries}
    pairs: Counter[tuple[str, str]] = Counter()
    for result in results:
        if result.error or not result.selected:
            continue
        query = truth.get(result.query_id)
        if query is None:
            continue
        effective_invoked = query.effective_invoked_skill(result)
        if effective_invoked is None or query.matches_skill(effective_invoked):
            continue
        pairs[(query.truth_label, effective_invoked)] += 1
    return pairs


class PrecursorEdge(BaseModel):
    """Represent an empirical directed transition between two skills in a trajectory."""

    model_config = ConfigDict(frozen=True)

    precursor: str
    target: str
    attempts: int
    handoffs: int
    handoff_rate: float
    avg_step_latency: float
    is_declared_dependency: bool = False


def compute_precursor_graph(
    results: Sequence[ProbeResult],
    queries: Sequence[Query],
    skills: Sequence[Skill] = (),
    min_observations: int = 1,
) -> tuple[PrecursorEdge, ...]:
    """Compute empirical precursor transition matrix T_i,j from observed trajectories."""
    truth = {q.id: q.truth_label for q in queries if not q.is_out_of_scope}
    declared_map = {s.name: set(s.declared_dependencies) for s in skills}
    transitions: dict[tuple[str, str], list[int]] = defaultdict(list)
    attempts: Counter[tuple[str, str]] = Counter()

    for r in results:
        if r.error:
            continue
        target = truth.get(r.query_id)
        if not target or len(r.invoked_skills) < MIN_TRANSITION_STEPS:
            continue

        for idx, skill in enumerate(r.invoked_skills):
            if skill == target:
                break
            attempts[(skill, target)] += 1
            remaining = r.invoked_skills[idx + 1 :]
            if target in remaining:
                step_gap = remaining.index(target) + 1
                transitions[(skill, target)].append(step_gap)

    edges = []
    for (src, dst), total_attempts in attempts.items():
        if total_attempts >= min_observations:
            gaps = transitions.get((src, dst), [])
            is_declared = src in declared_map.get(dst, set()) or dst in declared_map.get(src, set())
            edges.append(
                PrecursorEdge(
                    precursor=src,
                    target=dst,
                    attempts=total_attempts,
                    handoffs=len(gaps),
                    handoff_rate=round(len(gaps) / total_attempts, 3),
                    avg_step_latency=round(statistics.fmean(gaps), 2) if gaps else 0.0,
                    is_declared_dependency=is_declared,
                )
            )
    return tuple(sorted(edges, key=lambda e: (-e.handoff_rate, -e.attempts, e.precursor, e.target)))


class DecompositionResult(BaseModel):
    """Represent decomposition of pass-rate drop between baseline and scaled catalogs."""

    model_config = ConfigDict(frozen=True)

    baseline_pass_rate: float
    scaled_pass_rate: float
    delta_total: float
    delta_context: float
    delta_shadowing: float
    delta_total_ci: tuple[float, float] = (0.0, 0.0)
    delta_context_ci: tuple[float, float] = (0.0, 0.0)
    delta_shadowing_ci: tuple[float, float] = (0.0, 0.0)
    sample_size: int = 0
    baseline_ci: tuple[float, float] = (0.0, 0.0)
    scaled_ci: tuple[float, float] = (0.0, 0.0)


def _probe_outcome_is_pass(result: ProbeResult, query: Query | None = None) -> bool:
    """Determine whether a single probe execution succeeded."""
    if result.error:
        return False
    if query is not None:
        return score_trajectory(query, result.invoked_skills).trajectory_hit
    if result.invocation_pattern is not None:
        return result.invocation_pattern in (
            InvocationPattern.ORACLE_ONLY,
            InvocationPattern.MIXED_ORACLE,
            InvocationPattern.CORRECT_ABSTENTION,
        )
    return result.selected


def _probe_failure_is_context(result: ProbeResult) -> bool:
    """Determine whether a failed probe is attributed to context dilution or omission."""
    if result.invocation_pattern is not None:
        return result.invocation_pattern is InvocationPattern.ABANDONED
    return not result.selected


def _bootstrap_decomposition_ci(
    q_deltas: Sequence[float],
    q_ctx_deltas: Sequence[float],
    q_shd_deltas: Sequence[float],
    iterations: int,
    seed: int,
    confidence: float = DEFAULT_CONFIDENCE,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    """Compute empirical bootstrap confidence intervals for loss components."""
    m = len(q_deltas)
    if m <= 0 or iterations <= 0:
        return (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)

    # Standard pseudo-random generator is appropriate for Monte Carlo bootstrap
    rng = random.Random(seed)  # noqa: S311
    boot_deltas: list[float] = []
    boot_ctx: list[float] = []
    boot_shd: list[float] = []
    indices = list(range(m))

    for _ in range(iterations):
        sample_idx = [rng.choice(indices) for _ in range(m)]
        boot_deltas.append(statistics.fmean([q_deltas[i] for i in sample_idx]))
        boot_ctx.append(statistics.fmean([q_ctx_deltas[i] for i in sample_idx]))
        boot_shd.append(statistics.fmean([q_shd_deltas[i] for i in sample_idx]))

    boot_deltas.sort()
    boot_ctx.sort()
    boot_shd.sort()

    q_low, q_high = bootstrap_quantiles(confidence)
    low_idx = max(0, int(iterations * q_low))
    high_idx = min(int(iterations * q_high), iterations - 1)
    delta_ci = (round(boot_deltas[low_idx], 4), round(boot_deltas[high_idx], 4))
    ctx_ci = (round(boot_ctx[low_idx], 4), round(boot_ctx[high_idx], 4))
    shd_ci = (round(boot_shd[low_idx], 4), round(boot_shd[high_idx], 4))
    return delta_ci, ctx_ci, shd_ci


class _QueryDrop(NamedTuple):
    """Represent single-query pass-rates and loss decomposition."""

    base_pass: float
    scaled_pass: float
    delta: float
    delta_context: float
    delta_shadowing: float


def _group_valid_results_by_query(
    results: Sequence[ProbeResult],
) -> dict[str, list[ProbeResult]]:
    """Group non-error probe results by their associated query identifier."""
    grouped: dict[str, list[ProbeResult]] = defaultdict(list)
    for r in results:
        if not r.error:
            grouped[r.query_id].append(r)
    return grouped


def _query_pass_rate(results: Sequence[ProbeResult], query: Query | None) -> float:
    """Calculate the empirical pass rate for a list of probe results."""
    if not results:
        return 0.0
    return sum(1.0 for r in results if _probe_outcome_is_pass(r, query)) / len(results)


def _decompose_query_drop(
    b_res: Sequence[ProbeResult],
    s_res: Sequence[ProbeResult],
    query: Query | None,
) -> _QueryDrop:
    """Calculate pass rates and decompose performance drop for a single query."""
    p_base = _query_pass_rate(b_res, query)
    p_scaled = _query_pass_rate(s_res, query)
    delta = p_base - p_scaled

    s_fails = [r for r in s_res if not _probe_outcome_is_pass(r, query)]
    if not s_fails or delta == 0.0:
        return _QueryDrop(p_base, p_scaled, delta, 0.0, 0.0)

    n_ctx = sum(1.0 for r in s_fails if _probe_failure_is_context(r))
    n_shd = len(s_fails) - n_ctx
    return _QueryDrop(
        p_base,
        p_scaled,
        delta,
        delta * (n_ctx / len(s_fails)),
        delta * (n_shd / len(s_fails)),
    )


def _resolve_evaluation_query_ids(
    base_by_query: dict[str, list[ProbeResult]],
    scaled_by_query: dict[str, list[ProbeResult]],
) -> list[str]:
    """Identify query identifiers present across runs, falling back to set union."""
    common_qids = sorted(set(base_by_query.keys()) & set(scaled_by_query.keys()))
    if common_qids:
        return common_qids
    return sorted(set(base_by_query.keys()) | set(scaled_by_query.keys()))


def _compute_pass_rate_interval(pass_rate: float, sample_size: int) -> tuple[float, float]:
    """Compute rounded 95% Wilson confidence interval for an aggregate pass rate."""
    if sample_size <= 0:
        return (0.0, 0.0)
    w = wilson_interval(round(pass_rate * sample_size), sample_size)
    return (round(w.low, 4), round(w.high, 4)) if w else (0.0, 0.0)


def decompose_pass_rate_drop(
    baseline_results: Sequence[ProbeResult],
    scaled_results: Sequence[ProbeResult],
    queries: Sequence[Query] | None = None,
    iterations: int = 2000,
    seed: int = 42,
) -> DecompositionResult:
    """Decompose overall pass-rate drop into context dilution versus skill shadowing components."""
    truth_map: dict[str, Query] = {q.id: q for q in queries} if queries else {}
    base_by_query = _group_valid_results_by_query(baseline_results)
    scaled_by_query = _group_valid_results_by_query(scaled_results)

    common_qids = _resolve_evaluation_query_ids(base_by_query, scaled_by_query)
    if not common_qids:
        return DecompositionResult(
            baseline_pass_rate=0.0,
            scaled_pass_rate=0.0,
            delta_total=0.0,
            delta_context=0.0,
            delta_shadowing=0.0,
            delta_total_ci=(0.0, 0.0),
            delta_context_ci=(0.0, 0.0),
            delta_shadowing_ci=(0.0, 0.0),
            sample_size=0,
        )

    drops = [
        _decompose_query_drop(
            base_by_query.get(qid, []),
            scaled_by_query.get(qid, []),
            truth_map.get(qid),
        )
        for qid in common_qids
    ]

    q_deltas = [d.delta for d in drops]
    q_ctx_deltas = [d.delta_context for d in drops]
    q_shd_deltas = [d.delta_shadowing for d in drops]

    base_pass_rate = statistics.fmean(d.base_pass for d in drops)
    scaled_pass_rate = statistics.fmean(d.scaled_pass for d in drops)
    delta_total = statistics.fmean(q_deltas)
    delta_ctx = statistics.fmean(q_ctx_deltas)
    delta_shd = statistics.fmean(q_shd_deltas)

    delta_ci, ctx_ci, shd_ci = _bootstrap_decomposition_ci(
        q_deltas, q_ctx_deltas, q_shd_deltas, iterations=iterations, seed=seed
    )

    m = len(common_qids)
    return DecompositionResult(
        baseline_pass_rate=base_pass_rate,
        scaled_pass_rate=scaled_pass_rate,
        delta_total=delta_total,
        delta_context=delta_ctx,
        delta_shadowing=delta_shd,
        delta_total_ci=delta_ci,
        delta_context_ci=ctx_ci,
        delta_shadowing_ci=shd_ci,
        sample_size=m,
        baseline_ci=_compute_pass_rate_interval(base_pass_rate, m),
        scaled_ci=_compute_pass_rate_interval(scaled_pass_rate, m),
    )
