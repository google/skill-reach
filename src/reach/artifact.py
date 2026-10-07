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

"""Assemble, validate, and serialize schema-versioned evaluation artifacts."""

from __future__ import annotations

import math
import statistics
from collections import Counter
from fnmatch import fnmatchcase
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_validator,
    model_validator,
)

from reach._io import read_model, write_model
from reach.catalog import corpus_digest, resident_skills
from reach.config import RunConfig, resolve_path
from reach.difficulty import LexicalRank, lexical_ranks
from reach.leak import Leak, leaks
from reach.metrics import (
    ClassificationReport,
    ClassMetrics,
    classification_report,
    collisions,
    compute_f1,
    confusion,
    consistency_counts,
)
from reach.models import (
    NO_SKILL,
    Catalog,
    CatalogMode,
    NonBlankStr,
    ProbeResult,
    Provenance,
    Query,
    QueryKind,
    Skill,
)
from reach.queries import QuerySet, query_set_digest
from reach.runtime import CatalogFit
from reach.uncertainty import (
    Interval,
    cluster_wilson_interval,
    estimate_skill_icc,
    wilson_interval,
)

#: Numerical tolerance for rate vs count equality checks.
RATE_TOLERANCE: float = 1e-9

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from reach.run import Composition

__all__ = [
    "Abstention",
    "Artifact",
    "ConfusionPair",
    "ContestedSkill",
    "NotHeadline",
    "QueryRecord",
    "ResolvedRoot",
    "RunProvenance",
    "RunScores",
    "SampleQuery",
    "SkillScore",
    "Spread",
    "artifact_path",
    "filter_query_set",
    "read_artifact",
    "write_artifact",
]

#: Schema contract version for artifact serialization.
SCHEMA_VERSION = "0.1"

#: Default number of sample queries quoted per confusion pair.
DEFAULT_SAMPLE_QUERIES = 3

#: Standard filename suffix for evaluation artifacts.
ARTIFACT_SUFFIX = ".artifact.json"


class ResolvedRoot(BaseModel):
    """Record a source tree root and the count of skills loaded from it."""

    model_config = ConfigDict(frozen=True)

    path: Path
    skills: Annotated[int, Field(ge=0)]


class ContestedSkill(BaseModel):
    """Record a skill name provided by multiple roots and the resolved path used."""

    model_config = ConfigDict(frozen=True)

    name: str
    kept: Path
    dropped: tuple[Path, ...]
    ranked: bool = True


class SkillScore(BaseModel):
    """Represent precision, recall, and uncertainty for an individual skill."""

    model_config = ConfigDict(frozen=True)

    skill: str
    root: Path | None = None
    attempts: Annotated[int, Field(ge=1)] = 1
    probes: int = Field(ge=0)
    reached: int = Field(ge=0)
    recall: float | None = None
    trajectory_reached: int = Field(default=0, ge=0)
    trajectory_recall: float | None = None
    absorbed: int = Field(default=0, ge=0)
    precision: float | None = None

    @model_validator(mode="after")
    def _rates_agree_with_their_counts(self) -> Self:
        """Tie each rate to the count it was taken over, in both directions."""
        if self.trajectory_reached < self.reached:
            msg = (
                f"{self.skill}: trajectory_reached ({self.trajectory_reached}) cannot "
                f"be less than reached ({self.reached})"
            )
            raise ValueError(msg)
        if self.reached > self.probes:
            msg = f"{self.skill}: reached ({self.reached}) cannot exceed probes ({self.probes})"
            raise ValueError(msg)
        if self.trajectory_reached > self.probes:
            msg = (
                f"{self.skill}: trajectory_reached ({self.trajectory_reached}) cannot "
                f"exceed probes ({self.probes})"
            )
            raise ValueError(msg)
        if (self.probes > 0) != (self.recall is not None):
            msg = (
                f"{self.skill}: recall is defined exactly when a query named the "
                f"skill; got recall={self.recall} over {self.probes} probes"
            )
            raise ValueError(msg)
        if (self.probes > 0) != (self.trajectory_recall is not None):
            msg = (
                f"{self.skill}: trajectory_recall is defined exactly when a query "
                f"named the skill; got trajectory_recall={self.trajectory_recall} "
                f"over {self.probes} probes"
            )
            raise ValueError(msg)
        if (self.reached + self.absorbed > 0) != (self.precision is not None):
            msg = (
                f"{self.skill}: precision is defined exactly when the skill was "
                f"selected; got precision={self.precision} over "
                f"{self.reached + self.absorbed} selections"
            )
            raise ValueError(msg)
        return self

    @computed_field
    @property
    def f1(self) -> float | None:
        """Return harmonic mean of precision and recall, or None if undefined."""
        if self.recall is None:
            return None
        return compute_f1(self.precision or 0.0, self.recall)

    @computed_field
    @property
    def recall_interval(self) -> Interval | None:
        """Calculate the cluster-adjusted Wilson confidence interval for this skill's recall."""
        return cluster_wilson_interval(self.reached, self.probes, attempts=self.attempts)

    @computed_field
    @property
    def precision_interval(self) -> Interval | None:
        """Calculate the cluster-adjusted Wilson confidence interval for this skill's precision."""
        return cluster_wilson_interval(
            self.reached,
            self.reached + self.absorbed,
            attempts=self.attempts,
        )

    @classmethod
    def from_class_metrics(
        cls,
        metrics: ClassMetrics,
        root: Path | None = None,
        attempts: int | None = None,
    ) -> SkillScore:
        """Construct a SkillScore model from ClassMetrics and optional root path."""
        selected = metrics.true_positives + metrics.false_positives
        traj_reached = max(metrics.trajectory_true_positives, metrics.true_positives)
        traj_recall = (traj_reached / metrics.support) if metrics.support else None
        return cls(
            skill=metrics.label,
            root=root,
            probes=metrics.support,
            reached=metrics.true_positives,
            recall=metrics.recall if metrics.support else None,
            trajectory_reached=traj_reached,
            trajectory_recall=traj_recall,
            absorbed=metrics.false_positives,
            precision=metrics.precision if selected else None,
            attempts=attempts if attempts is not None else metrics.attempts,
        )


class SampleQuery(BaseModel):
    """Quote one query verbatim with probe count and model reasoning traces."""

    model_config = ConfigDict(frozen=True)

    query_id: str
    text: str
    probes: int = Field(
        ge=1,
        description="Number of attempts that resulted in this confusion pairing.",
    )
    reasoning: tuple[str, ...] = ()


class ConfusionPair(BaseModel):
    """Record misroute pairings, probe counts, and sample queries."""

    model_config = ConfigDict(frozen=True)

    expected: str
    invoked: str
    probes: int = Field(ge=1)
    collisions: int = Field(default=0, ge=0)
    queries: tuple[SampleQuery, ...] = ()

    @model_validator(mode="after")
    def _collisions_are_a_subset(self) -> Self:
        """Validate that collisions do not exceed total probe count."""
        if self.collisions > self.probes:
            msg = (
                f"{self.expected} -> {self.invoked}: {self.collisions} collisions "
                f"out of {self.probes} probes"
            )
            raise ValueError(
                msg,
            )
        return self


class QueryRecord(BaseModel):
    """Stage 3 telemetry: summarize probe outcomes, confidence interval, and leak status."""

    model_config = ConfigDict(frozen=True)

    query_id: NonBlankStr
    text: NonBlankStr
    kind: QueryKind | None = None
    expected_skill: str
    probes: int = Field(default=0, ge=0)
    hits: int = Field(default=0, ge=0)
    selections: tuple[str, ...] = ()
    difficulty_rank: int | None = None
    leak: Leak | None = None

    @property
    def clean(self) -> bool:
        """Return True if all probe attempts matched ground truth."""
        return self.probes > 0 and self.hits == self.probes

    @computed_field
    @property
    def interval(self) -> Interval | None:
        """Calculate the Wilson score confidence interval for query hit rate."""
        return wilson_interval(self.hits, self.probes)


class Spread(BaseModel):
    """Summarize variance and standard error across probe replicates."""

    model_config = ConfigDict(frozen=True)

    replicates: int = Field(default=0, ge=0)
    top1_by_attempt: tuple[float, ...] = ()
    mean: float | None = None
    repeated_queries: int = Field(default=0, ge=0)
    standard_error: float | None = None
    skill_icc: Annotated[float, Field(ge=0.0, le=1.0)] | None = None

    @model_validator(mode="after")
    def _figures_match_what_was_observed(self) -> Self:
        """Validate replicate counts against observed statistics."""
        if len(self.top1_by_attempt) != self.replicates:
            msg = f"{self.replicates} replicates but {len(self.top1_by_attempt)} accuracies"
            raise ValueError(
                msg,
            )
        if (self.replicates > 0) != (self.mean is not None):
            msg = f"{self.replicates} replicates cannot mean to {self.mean}"
            raise ValueError(msg)
        if (self.replicates > 0) != (self.standard_error is not None):
            msg = (
                f"a probed run has an error on its estimate; got {self.replicates} "
                f"replicates and standard_error={self.standard_error}"
            )
            raise ValueError(
                msg,
            )
        return self


class NotHeadline(BaseModel):
    """Hold secondary evaluation metrics (macro precision, recall, F1)."""

    model_config = ConfigDict(frozen=True)

    macro_f1: float
    macro_precision: float = 0.0
    macro_recall: float = 0.0
    labels: tuple[str, ...] = ()


class Abstention(BaseModel):
    """Report overall, false, and out-of-scope abstention rates with bounds."""

    model_config = ConfigDict(frozen=True)

    rate: float
    false_rate: float
    out_of_scope_detection: float | None = None
    scored: int = Field(ge=0)
    abstentions: int = Field(ge=0)
    in_scope: int = Field(ge=0)
    false_abstentions: int = Field(ge=0)
    out_of_scope: int = Field(default=0, ge=0)
    out_of_scope_detected: int = Field(default=0, ge=0)
    attempts: Annotated[int, Field(ge=1)] = 1

    @computed_field
    @property
    def interval(self) -> Interval | None:
        """Calculate the cluster-adjusted Wilson confidence interval for overall abstention rate."""
        return cluster_wilson_interval(self.abstentions, self.scored, attempts=self.attempts)

    @computed_field
    @property
    def false_interval(self) -> Interval | None:
        """Calculate the cluster-adjusted Wilson confidence interval for false abstention rate."""
        return cluster_wilson_interval(
            self.false_abstentions,
            self.in_scope,
            attempts=self.attempts,
        )

    @computed_field
    @property
    def out_of_scope_interval(self) -> Interval | None:
        """Calculate the cluster Wilson confidence interval for out-of-scope detection rate."""
        return cluster_wilson_interval(
            self.out_of_scope_detected,
            self.out_of_scope,
            attempts=self.attempts,
        )


class RunScores(BaseModel):
    """Hold primary run-level evaluation scores and confidence intervals."""

    model_config = ConfigDict(frozen=True)

    consistency: float
    top1_accuracy: float
    abstention: Abstention
    not_headline: NotHeadline
    unanimous_queries: int = Field(ge=0)
    observed_queries: int = Field(ge=0)
    top1_hits: int = Field(ge=0)
    entrypoint_hits: int = Field(default=0, ge=0)
    entrypoint_accuracy: float = 0.0
    trajectory_hits: int = Field(default=0, ge=0)
    trajectory_reachability: float = 0.0
    step_efficiency: Annotated[float | None, Field(ge=0.0, le=1.0)] = None
    skill_f1: Annotated[float | None, Field(ge=0.0, le=1.0)] = None
    redundancy: float = 0.0
    scored: int = Field(ge=0)
    attempts: Annotated[int, Field(ge=1)] = 1

    @computed_field
    @property
    def consistency_interval(self) -> Interval | None:
        """Calculate the Wilson confidence interval for query consistency."""
        return wilson_interval(self.unanimous_queries, self.observed_queries)

    @computed_field
    @property
    def top1_interval(self) -> Interval | None:
        """Calculate the cluster-adjusted Wilson confidence interval for top-1 accuracy."""
        return cluster_wilson_interval(self.top1_hits, self.scored, attempts=self.attempts)

    @computed_field
    @property
    def entrypoint_interval(self) -> Interval | None:
        """Calculate the cluster-adjusted Wilson confidence interval for entrypoint accuracy."""
        return cluster_wilson_interval(self.entrypoint_hits, self.scored, attempts=self.attempts)

    @computed_field
    @property
    def trajectory_interval(self) -> Interval | None:
        """Calculate the cluster-adjusted Wilson confidence interval for trajectory reachability."""
        return cluster_wilson_interval(self.trajectory_hits, self.scored, attempts=self.attempts)

    @model_validator(mode="after")
    def _rates_match_the_counts_they_came_from(self) -> Self:
        """Validate that stored rates match underlying hit and total counts."""
        for name, rate, hits, total in (
            (
                "consistency",
                self.consistency,
                self.unanimous_queries,
                self.observed_queries,
            ),
            ("top1_accuracy", self.top1_accuracy, self.top1_hits, self.scored),
            (
                "entrypoint_accuracy",
                self.entrypoint_accuracy,
                self.entrypoint_hits,
                self.scored,
            ),
            (
                "trajectory_reachability",
                self.trajectory_reachability,
                self.trajectory_hits,
                self.scored,
            ),
        ):
            if not total:
                continue
            if abs(rate - hits / total) > RATE_TOLERANCE:
                msg = f"{name} is {rate} but its counts give {hits}/{total}"
                raise ValueError(msg)
        return self


class RunProvenance(BaseModel):
    """Record runtime, model, attempt count, and experimental arm for a run."""

    model_config = ConfigDict(frozen=True)

    runtime: str
    model: str
    resolved_model: str = ""
    attempts: int = Field(ge=1)
    arm: str
    condition: str = ""
    catalog_fit: CatalogFit | None = None


class Artifact(BaseModel):
    """Represent complete evaluation results, summary scores, and provenance."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    # 1. Schema & Metadata
    schema_version: str = SCHEMA_VERSION
    catalog_id: str
    catalog_mode: CatalogMode
    catalog_size: int = Field(ge=0)
    catalog_target: str | None = None
    provenance: RunProvenance
    digests: Provenance = Field(
        description="Configuration fingerprint, corpus digest, and query set digest.",
    )
    verified_digests: tuple[str, ...] = Field(
        default=(),
        description="Provenance digests corroborated by individual probe results.",
    )
    resolved_roots: tuple[ResolvedRoot, ...] = ()
    contested_skills: tuple[ContestedSkill, ...] = Field(
        default=(),
        description="Skills provided by multiple roots and resolved by precedence.",
    )

    # 2. Executive scores
    scores: RunScores
    spread: Spread

    # 3. Run accounting counters
    probes: int = Field(default=0, ge=0)
    errors: int = Field(default=0, ge=0)
    spend_usd: float = Field(default=0.0, ge=0.0)
    reused: int = Field(default=0, ge=0)

    # 4. Diagnostics
    confusion: tuple[ConfusionPair, ...] = ()

    # 5. Leaf collections at the bottom
    skills: tuple[SkillScore, ...] = Field(
        default=(),
        description="Per-skill evaluation scores in catalog order.",
    )
    queries: tuple[QueryRecord, ...] = ()

    @field_validator("schema_version")
    @classmethod
    def _readable_version(cls, value: str) -> str:
        """Refuse a document written to a contract this code does not implement."""
        if value.split(".", maxsplit=1)[0] != SCHEMA_VERSION.split(".", maxsplit=1)[0]:
            msg = (
                f"artifact schema_version {value!r} is not readable by "
                f"{SCHEMA_VERSION!r}: the major version differs"
            )
            raise ValueError(
                msg,
            )
        return value

    @model_validator(mode="after")
    def _every_digest_is_stated(self) -> Self:
        """Validate that all required provenance digests are populated."""
        missing = [
            field
            for field in ("config_fingerprint", "corpus_digest", "queries_digest")
            if not getattr(self.digests, field)
        ]
        if missing:
            msg = f"an artifact must state every digest; missing: {missing}"
            raise ValueError(msg)
        return self

    @property
    def unreached(self) -> tuple[SkillScore, ...]:
        """Return the skills that were asked for and never arrived at across any turn."""
        return tuple(
            s
            for s in self.skills
            if s.recall is not None
            and (s.trajectory_recall if s.trajectory_recall is not None else s.recall) == 0.0
        )

    @property
    def attractors(self) -> tuple[SkillScore, ...]:
        """Return skills selected more often than requested in ground truth."""
        taking = (s for s in self.skills if s.reached + s.absorbed > s.probes)
        return tuple(sorted(taking, key=lambda s: (-s.absorbed, s.skill)))

    @property
    def unclean(self) -> tuple[QueryRecord, ...]:
        """Return the queries that did not agree with ground truth on every attempt."""
        return tuple(q for q in self.queries if q.probes > 0 and not q.clean)

    @classmethod
    def assemble(
        cls,
        composition: Composition,
        results: Sequence[ProbeResult],
        *,
        roots: Sequence[Path] | None = None,
        contested: Sequence[ContestedSkill] = (),
        sample_queries: int = DEFAULT_SAMPLE_QUERIES,
        fit: CatalogFit | None = None,
        spend_usd: float = 0.0,
        reused: int = 0,
        difficulty: Mapping[str, LexicalRank] | None = None,
        digests: Provenance | None = None,
        cross_check: bool = True,
    ) -> Artifact:
        """Assemble an Artifact from a validated Composition and probe results.

        Derive query set, catalog, skills, and configuration directly from the
        Composition instance, reducing the orchestration seam from fifteen
        parameters to two essential inputs.
        """
        return _ArtifactAssembler(
            results=results,
            query_set=composition.query_set,
            catalog=composition.catalog,
            skills=composition.skills,
            config=composition.config,
            roots=roots,
            contested=contested,
            sample_queries=sample_queries,
            fit=fit,
            spend_usd=spend_usd,
            reused=reused,
            difficulty=difficulty,
            digests=digests,
            cross_check=cross_check,
        ).assemble()


def _query_matches_skill(query: Query, pattern: str) -> bool:
    """Return True if query truth label or expected skill matches glob pattern."""
    return fnmatchcase(query.truth_label, pattern) or (
        query.expected_skill is not None and fnmatchcase(query.expected_skill, pattern)
    )


def _resolve_subset_queries(query_set: QuerySet, subset: QuerySet | None) -> list[Query]:
    """Validate and extract subset queries against the parent QuerySet."""
    if subset is None:
        return list(query_set.queries)
    by_id = {q.query_id: q for q in query_set.queries}
    missing = sorted(q.query_id for q in subset.queries if q.query_id not in by_id)
    if missing:
        msg = f"subset queries not present in query set: {missing}"
        raise ValueError(msg)
    for sq in subset.queries:
        full_q = by_id[sq.query_id]
        if sq.truth_label != full_q.truth_label or frozenset(sq.acceptable_skills) != frozenset(
            full_q.acceptable_skills
        ):
            msg = (
                f"ground truth mismatch for query {sq.query_id!r}: query set expected "
                f"{full_q.truth_label!r}, subset expected {sq.truth_label!r}"
            )
            raise ValueError(msg)
    return [by_id[sq.query_id] for sq in subset.queries]


def filter_query_set(
    query_set: QuerySet,
    *,
    subset: QuerySet | None = None,
    filter_skill: Sequence[str] = (),
    filter_id: Sequence[str] = (),
) -> QuerySet:
    """Filter a QuerySet to a subset QuerySet and/or skill and query ID globs."""
    if subset is None and not filter_skill and not filter_id:
        return query_set

    queries = _resolve_subset_queries(query_set, subset)

    for pat in filter_skill:
        if not any(_query_matches_skill(q, pat) for q in queries):
            msg = f"query slice matched 0 queries for filter_skill pattern {pat!r}"
            raise ValueError(msg)
    if filter_skill:
        queries = [q for q in queries if any(_query_matches_skill(q, pat) for pat in filter_skill)]

    for pat in filter_id:
        if not any(fnmatchcase(q.query_id, pat) for q in queries):
            msg = f"query slice matched 0 queries for filter_id pattern {pat!r}"
            raise ValueError(msg)
    if filter_id:
        queries = [q for q in queries if any(fnmatchcase(q.query_id, pat) for pat in filter_id)]
    if not queries:
        msg = "query slice matched 0 queries"
        raise ValueError(msg)

    return query_set.model_copy(update={"queries": tuple(queries)})


def artifact_path(results_path: Path) -> Path:
    """Return the default artifact path corresponding to a results file."""
    return Path(f"{results_path}{ARTIFACT_SUFFIX}")


def write_artifact(artifact: Artifact, path: Path) -> Path:
    """Serialize and write a Artifact model to disk as formatted JSON."""
    destination = Path(path)
    if destination.exists() and not _is_artifact(destination):
        msg = (
            f"refusing to overwrite {destination}: it is not an artifact, and "
            "writing one here would destroy it. Artifacts belong at the path "
            "artifact_path() names."
        )
        raise ValueError(
            msg,
        )
    return write_model(artifact, destination)


def _is_artifact(path: Path) -> bool:
    """Return True if a file exists and parses as a valid Artifact."""
    try:
        read_artifact(path)
    except (OSError, ValueError):
        return False
    return True


def read_artifact(path: Path | str) -> Artifact:
    """Read and validate an Artifact from a JSON file."""
    return read_model(Artifact, path)


def _named_root(corpus: Path | None) -> tuple[Path, ...]:
    """Convert an optional corpus path to a tuple of roots."""
    return () if corpus is None else (corpus,)


def _attribute(
    skills: Sequence[Skill],
    roots: Sequence[Path],
) -> dict[str, Path | None]:
    """Map each skill to the most specific resolved root containing it."""
    ordered = sorted(
        {(root.expanduser().absolute(), resolve_path(root)) for root in roots},
        key=lambda pair: (len(pair[1].parts), len(pair[0].parts)),
        reverse=True,
    )
    attributed: dict[str, Path | None] = {}
    for skill in skills:
        lexical = skill.path.expanduser().absolute()
        resolved = resolve_path(skill.path)
        attributed[skill.name] = next(
            (
                resolved_root
                for lexical_root, resolved_root in ordered
                if lexical.is_relative_to(lexical_root)
                or lexical.is_relative_to(resolved_root)
                or resolved.is_relative_to(resolved_root)
            ),
            None,
        )
    return attributed


def _single[T](values: Iterable[T], field: str, fallback: T) -> T:
    """Extract a single uniform value from results or return a fallback."""
    stated = {value for value in values if value}
    if len(stated) > 1:
        msg = (
            f"rows disagree on {field}: {sorted(str(v) for v in stated)}; an artifact describes "
            "one run, so these belong in two"
        )
        raise ValueError(
            msg,
        )
    return stated.pop() if stated else fallback


def _cross_check(
    results: Sequence[ProbeResult],
    digests: Provenance,
) -> tuple[str, ...]:
    """Verify results against expected provenance digests and return verified fields."""

    def _same_condition(row: ProbeResult) -> bool:
        """Return True if a row matches the expected condition digest."""
        return bool(digests.condition_digest) and (row.condition_digest == digests.condition_digest)

    verified = []
    for field in (
        "config_fingerprint",
        "condition_digest",
        "corpus_digest",
        "queries_digest",
    ):
        derived = getattr(digests, field)
        rows = (
            [
                row
                for row in results
                if row.config_fingerprint == derived or not _same_condition(row)
            ]
            if field == "config_fingerprint"
            else results
        )
        recorded = {getattr(row, field) for row in rows if getattr(row, field)}
        if stray := sorted(recorded - {derived}):
            msg = (
                f"rows carry {field} {stray} but the material in hand digests to "
                f"{derived!r}: these rows measured something else"
            )
            raise ValueError(
                msg,
            )
        if recorded:
            verified.append(field)
    return tuple(verified)


def _skill_score(
    metrics: ClassMetrics,
    root: Path | None,
    attempts: int | None = None,
) -> SkillScore:
    """Construct a SkillScore model from ClassMetrics and root path."""
    eff_attempts = attempts if attempts is not None else metrics.attempts
    return SkillScore.from_class_metrics(metrics, root=root, attempts=eff_attempts)


def _sample_queries(
    attempts: Mapping[str, int],
    texts: Mapping[str, str],
    reasonings: Mapping[str, tuple[str, ...]],
    limit: int,
) -> tuple[SampleQuery, ...]:
    """Extract top sample queries for a confusion pair."""
    ranked = sorted(attempts.items(), key=lambda item: (-item[1], item[0]))
    return tuple(
        SampleQuery(
            query_id=query_id,
            text=texts[query_id],
            probes=count,
            reasoning=reasonings.get(query_id, ()),
        )
        for query_id, count in ranked[:limit]
    )


def _confusion_pairs(
    results: Sequence[ProbeResult],
    queries: Sequence[Query],
    limit: int,
) -> tuple[ConfusionPair, ...]:
    """Compute confusion pairs and attach sample queries."""
    texts = {query.query_id: query.text for query in queries}
    truth = {query.query_id: query for query in queries}
    counts = confusion(results, queries)
    collided = collisions(results, queries)

    quoted: dict[tuple[str, str | None], Counter[str]] = {}
    reasonings: dict[tuple[tuple[str, str | None], str], tuple[str, ...]] = {}
    for row in results:
        query = truth.get(row.query_id)
        if row.error or query is None:
            continue
        effective_invoked = query.effective_invoked_skill(row)
        pair_key = (query.truth_label, effective_invoked)
        quoted.setdefault(pair_key, Counter())[row.query_id] += 1
        if row.reasoning and (pair_key, row.query_id) not in reasonings:
            reasonings[(pair_key, row.query_id)] = row.reasoning

    def order(item: tuple[tuple[str, str | None], int]) -> tuple[int, str, str]:
        """Sort confusion entries by descending frequency and skill name."""
        (expected, invoked), count = item
        return (-count, expected, invoked or NO_SKILL)

    return tuple(
        ConfusionPair(
            expected=expected,
            invoked=invoked if invoked is not None else NO_SKILL,
            probes=count,
            collisions=collided[(expected, invoked)] if invoked is not None else 0,
            queries=_sample_queries(
                quoted.get((expected, invoked), Counter()),
                texts,
                {qid: r for (pk, qid), r in reasonings.items() if pk == (expected, invoked)},
                limit,
            ),
        )
        for (expected, invoked), count in sorted(counts.items(), key=order)
    )


def _query_variance(record: QueryRecord) -> float:
    """Compute the variance contribution for a single query record."""
    rate = (record.hits + 0.5) / (record.probes + 1)
    return record.probes * rate * (1 - rate)


def _standard_error(records: Sequence[QueryRecord]) -> float | None:
    """Estimate standard error of top-1 accuracy pooled across queries."""
    probes = sum(record.probes for record in records)
    if not probes:
        return None
    return math.sqrt(sum(_query_variance(record) for record in records)) / probes


def _spread(
    results: Sequence[ProbeResult],
    queries: Sequence[Query],
    records: Sequence[QueryRecord],
) -> Spread:
    """Compute replicate variance and error estimates across probe attempts."""
    by_attempt: dict[int, list[ProbeResult]] = {}
    for row in results:
        by_attempt.setdefault(row.attempt, []).append(row)

    scores = tuple(
        classification_report(rows, queries).top1_accuracy
        for _, rows in sorted(by_attempt.items())
        if any(not row.error for row in rows)
    )
    outcomes_by_skill: dict[str, list[float]] = {}
    for record in records:
        if record.probes > 0 and record.expected_skill != NO_SKILL:
            outcomes_by_skill.setdefault(record.expected_skill, []).append(
                record.hits / record.probes
            )
    return Spread(
        replicates=len(scores),
        top1_by_attempt=scores,
        mean=statistics.fmean(scores) if scores else None,
        repeated_queries=sum(1 for record in records if record.probes > 1),
        standard_error=_standard_error(records),
        skill_icc=estimate_skill_icc(outcomes_by_skill),
    )


def _query_records(
    results: Sequence[ProbeResult],
    queries: Sequence[Query],
    ranks: Mapping[str, LexicalRank],
    flags: Mapping[str, Leak],
) -> tuple[QueryRecord, ...]:
    """Construct QueryRecord models for all queries in the query set."""
    grouped: dict[str, list[ProbeResult]] = {}
    for row in results:
        if not row.error:
            grouped.setdefault(row.query_id, []).append(row)

    records = []
    for query in sorted(queries, key=lambda q: q.query_id):
        usable = grouped.get(query.query_id, [])
        records.append(
            QueryRecord(
                query_id=query.query_id,
                text=query.text,
                kind=query.kind,
                expected_skill=query.truth_label,
                probes=len(usable),
                hits=sum(
                    1 for row in usable if query.matches_skill(query.effective_invoked_skill(row))
                ),
                selections=tuple(sorted({row.predicted_label for row in usable})),
                difficulty_rank=(rank.position if (rank := ranks.get(query.query_id)) else None),
                leak=flags.get(query.query_id),
            ),
        )
    return tuple(records)


def _build_provenance(
    results: Sequence[ProbeResult],
    config: RunConfig,
    fit: CatalogFit | None,
) -> RunProvenance:
    """Extract runtime, model, attempts, and arm metadata into RunProvenance."""
    return RunProvenance(
        runtime=_single(
            (row.runtime for row in results),
            "runtime",
            config.runtime.agent,
        ),
        model=_single(
            (row.model for row in results),
            "model",
            config.runtime.resolved_options().get("model", ""),
        ),
        resolved_model=_single(
            (row.resolved_model for row in results),
            "resolved_model",
            "",
        ),
        attempts=config.plan.attempts,
        arm=config.arm,
        condition=config.condition,
        catalog_fit=fit,
    )


def _build_resolved_roots(
    resolved: Sequence[Path],
    root_of: Mapping[str, Path | None],
) -> tuple[ResolvedRoot, ...]:
    """Map attributed skill counts to each resolved root path."""
    return tuple(
        ResolvedRoot(
            path=root,
            skills=sum(1 for attributed in root_of.values() if attributed == root),
        )
        for root in resolved
    )


def _build_run_scores(
    standard: ClassificationReport,
    unanimous: int,
    observed: int,
    attempts: int | None = None,
) -> RunScores:
    """Assemble consistency, accuracy, and abstention metrics into RunScores."""
    eff_attempts = attempts if attempts is not None else standard.attempts
    return RunScores(
        consistency=unanimous / observed if observed else 0.0,
        top1_accuracy=standard.top1_accuracy,
        unanimous_queries=unanimous,
        observed_queries=observed,
        top1_hits=standard.top1_hits,
        entrypoint_hits=standard.entrypoint_hits,
        entrypoint_accuracy=standard.entrypoint_accuracy,
        trajectory_hits=standard.trajectory_hits,
        trajectory_reachability=standard.trajectory_reachability,
        step_efficiency=standard.step_efficiency,
        skill_f1=standard.skill_f1,
        redundancy=standard.redundancy,
        scored=standard.scored,
        attempts=eff_attempts,
        abstention=Abstention(
            rate=standard.abstention_rate,
            false_rate=standard.false_abstention_rate,
            out_of_scope_detection=standard.out_of_scope_detection,
            scored=standard.scored,
            abstentions=standard.abstentions,
            in_scope=standard.in_scope,
            false_abstentions=standard.false_abstentions,
            out_of_scope=standard.out_of_scope,
            out_of_scope_detected=standard.out_of_scope_detected,
            attempts=eff_attempts,
        ),
        not_headline=NotHeadline(
            macro_f1=standard.macro_f1,
            macro_precision=standard.macro_precision,
            macro_recall=standard.macro_recall,
            labels=tuple(entry.label for entry in standard.per_class if entry.support),
        ),
    )


class _ArtifactAssembler:
    """Internal method object coordinating the construction and validation of an Artifact."""

    def __init__(
        self,
        results: Sequence[ProbeResult],
        query_set: QuerySet,
        catalog: Catalog,
        skills: Sequence[Skill],
        config: RunConfig,
        *,
        roots: Sequence[Path] | None = None,
        contested: Sequence[ContestedSkill] = (),
        sample_queries: int = DEFAULT_SAMPLE_QUERIES,
        fit: CatalogFit | None = None,
        spend_usd: float = 0.0,
        reused: int = 0,
        difficulty: Mapping[str, LexicalRank] | None = None,
        digests: Provenance | None = None,
        cross_check: bool = True,
    ) -> None:
        if sample_queries < 0:
            msg = f"sample queries per pair cannot be negative, got {sample_queries}"
            raise ValueError(msg)
        self.results = results
        self.query_set = query_set
        self.queries = query_set.queries
        self.catalog = catalog
        self.skills = skills
        self.config = config
        self.roots = roots
        self.contested = contested
        self.sample_queries = sample_queries
        self.fit = fit
        self.spend_usd = spend_usd
        self.reused = reused
        self.difficulty = difficulty
        self.digests = digests
        self.cross_check = cross_check

    def assemble(self) -> Artifact:
        """Execute assembly pipeline and return an immutable Artifact."""
        resident = resident_skills(self.catalog, self.skills)
        derived_digests = (
            self.digests
            if self.digests is not None
            else Provenance(
                config_fingerprint=self.config.fingerprint,
                condition_digest=self.config.condition,
                corpus_digest=corpus_digest(self.skills),
                queries_digest=query_set_digest(self.query_set),
                tag=self.config.study.tag,
            )
        )
        verified = _cross_check(self.results, derived_digests) if self.cross_check else ()

        attempts = self.config.plan.attempts
        catalog_wide = classification_report(
            self.results,
            self.queries,
            labels=[*self.catalog.skills, NO_SKILL],
            attempts=attempts,
        )
        standard = classification_report(
            self.results,
            self.queries,
            attempts=attempts,
        )
        named = self.roots if self.roots is not None else _named_root(self.config.study.skills)
        resolved = sorted({resolve_path(root) for root in named})
        root_of = _attribute(self.skills, resolved)
        ranks = (
            self.difficulty
            if self.difficulty is not None
            else lexical_ranks(self.queries, resident)
        )
        records = _query_records(
            self.results,
            self.queries,
            ranks,
            leaks(self.queries, resident, background=self.skills),
        )
        unanimous, observed = consistency_counts(self.results, self.queries)

        return Artifact(
            digests=derived_digests,
            verified_digests=verified,
            provenance=_build_provenance(self.results, self.config, self.fit),
            catalog_id=self.catalog.id,
            catalog_mode=self.catalog.mode,
            catalog_size=self.catalog.size,
            catalog_target=self.catalog.target,
            resolved_roots=_build_resolved_roots(resolved, root_of),
            contested_skills=tuple(self.contested),
            skills=tuple(
                _skill_score(catalog_wide.by_label(name), root_of.get(name), attempts=attempts)
                for name in self.catalog.skills
            ),
            scores=_build_run_scores(standard, unanimous, observed, attempts=attempts),
            spread=_spread(self.results, self.queries, records),
            confusion=_confusion_pairs(self.results, self.queries, self.sample_queries),
            queries=records,
            probes=len(self.results),
            errors=standard.errors,
            spend_usd=self.spend_usd,
            reused=self.reused,
        )
