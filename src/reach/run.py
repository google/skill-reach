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

"""Orchestrate evaluation runs, catalog composition, probe execution, and persistence."""

from __future__ import annotations

import functools
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from reach._io import atomic_write_text
from reach.artifact import (
    DEFAULT_SAMPLE_QUERIES,
    Artifact,
    ContestedSkill,
)
from reach.catalog import (
    build_catalogs,
    corpus_digest,
    load_skills,
    resolve_catalog,
)
from reach.config import DEFAULT_ATTEMPTS, DEFAULT_GEMINI_MODEL, RunConfig, agent_default_model
from reach.models import Catalog, CatalogMode, ProbeResult, Provenance, Query, Skill
from reach.queries import QuerySet, load_query_set, query_set_digest
from reach.runtime import AgentRuntime, CatalogFit, build_runtime
from reach.uncertainty import detectable_delta

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Sequence

__all__ = [
    "Composition",
    "ConfigSidecar",
    "Plan",
    "ProbeHarness",
    "RunOutcome",
    "append_result",
    "completed_attempts",
    "compose",
    "conduct",
    "evaluate",
    "load_corpus",
    "load_results",
    "plan_only",
    "plan_probes",
    "read_sidecar",
    "recorded_fingerprints",
    "sidecar_path",
    "validate_appendable",
    "validate_catalog_fit",
    "validate_query_coverage",
    "validate_residency",
    "write_results",
    "write_sidecar",
]


def validate_query_coverage(
    query_set: QuerySet,
    skills: Sequence[str],
    *,
    partial: bool = False,
) -> None:
    """Validate query set expected skills align with resident catalog skills.

    Args:
        query_set: Labeled evaluation queries with assigned target skills.
        skills: Names of skills resident in the catalog.
        partial: If True, allow query sets that target only a subset of resident skills.

    Raises:
        ValueError: If resident skills have no queries (unless partial is True) or if queries
            target skills not resident in the catalog.
    """
    expected = {q.expected_skill for q in query_set.queries if q.expected_skill}
    missing = sorted(set(skills) - expected)
    stray = sorted(expected - set(skills))
    problems = []
    if missing and not partial:
        problems.append(f"skills with no query: {missing}")
    if stray:
        problems.append(f"queries naming skills outside the catalog: {stray}")
    if problems:
        raise ValueError("; ".join(problems))


def validate_catalog_fit(
    runtime: AgentRuntime,
    catalog: Catalog,
    skills: Sequence[Skill],
    *,
    allow_truncation: bool = False,
) -> CatalogFit:
    """Verify that resident catalog fits within runtime listing budgets.

    Args:
        runtime: Target agent runtime adapter.
        catalog: Catalog definition specifying resident skill names.
        skills: Resident Skill objects loaded from disk.
        allow_truncation: If True, allow execution even if runtime truncates descriptions.

    Returns:
        A CatalogFit assessment recording whether the catalog fits and truncation counts.

    Raises:
        ValueError: If the catalog exceeds runtime prompt budget and allow_truncation is False.
    """
    fit = runtime.fit(catalog, skills)
    if fit.whole or allow_truncation:
        return fit
    advice = f"{fit.remedy}, or pass" if fit.remedy else "Pass"
    msg = (
        f"{fit.truncated} of {catalog.size} skills would reach the model as a bare "
        f"name: {runtime.name} allows {fit.allowed:,} {fit.unit} for the listing "
        f"it shows and this catalog asks {fit.asked:,}. A skill with no "
        "description is not competing on one, and which ones keep theirs is "
        "decided by state local to this machine that nothing here records. "
        f"{advice} --allow-truncation to measure the catalog as the runtime will "
        "really show it."
    )
    raise ValueError(msg)


def validate_residency(
    catalog: Catalog,
    observed: tuple[str, ...],
    *,
    dynamic: bool = False,
) -> str | None:
    """Return error string if catalog skill is missing or unrecognized."""
    if not observed:
        return None
    if dynamic:
        unrecognized = sorted(set(observed) - set(catalog.skills))
        if unrecognized:
            return f"catalog contains unknown skills: {', '.join(unrecognized)}"
        return None
    missing = sorted(set(catalog.skills) - set(observed))
    if missing:
        return f"catalog not resident: {', '.join(missing)}"
    return None


def plan_probes(
    queries: Sequence[Query],
    attempts: int = DEFAULT_ATTEMPTS,
) -> list[tuple[Query, int]]:
    """Generate an interleaved list of (query, attempt) pairs for probing."""
    return [(query, attempt) for attempt in range(1, attempts + 1) for query in queries]


def _with_retries(
    execute: Callable[[], ProbeResult],
    retries: int,
    backoff_s: float,
    sleep: Callable[[float], None],
) -> ProbeResult:
    """Execute a probe operation with exponential backoff on error."""
    result = execute()
    for retry in range(1, retries + 1):
        if not result.error:
            return result
        sleep(backoff_s * (2 ** (retry - 1)))
        result = execute()
    return result


def append_result(path: Path | str, result: ProbeResult) -> None:
    """Append a ProbeResult model as a JSON line to the target file."""
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(result.model_dump_json() + "\n")


def write_results(path: Path | str, results: Iterable[ProbeResult]) -> Path:
    """Persist a collection of ProbeResult models as JSON lines to the target file."""
    resolved = Path(path).expanduser().resolve()
    content = "".join(f"{r.model_dump_json()}\n" for r in results)
    atomic_write_text(resolved, content)
    return resolved


def load_results(path: Path | str) -> list[ProbeResult]:
    """Load previously recorded ProbeResult records from a JSONL file."""
    resolved = Path(path).expanduser().resolve()
    return [
        ProbeResult.model_validate_json(line)
        for line in resolved.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _is_resumable_row(
    row: ProbeResult,
    *,
    active_query_ids: set[str] | None = None,
    corpus_digest: str = "",
    catalog_skills: set[str] | None = None,
    config_fingerprint: str = "",
    condition_digest: str = "",
) -> bool:
    """Return whether a recorded ProbeResult satisfies resume criteria for the active run."""
    if row.error:
        return False
    if active_query_ids is not None and row.query_id not in active_query_ids:
        return False
    if corpus_digest and row.corpus_digest != corpus_digest:
        return False
    if (
        config_fingerprint
        and row.config_fingerprint
        and row.config_fingerprint != config_fingerprint
        and not (condition_digest and row.condition_digest == condition_digest)
    ):
        return False
    return not (catalog_skills is not None and set(row.observed_catalog) != catalog_skills)


def _partition_resumed_results(
    raw_previous: Sequence[ProbeResult],
    *,
    active_query_ids: set[str],
    corpus_digest: str,
    catalog_skills: set[str],
    config_fingerprint: str = "",
    condition_digest: str = "",
) -> tuple[list[ProbeResult], set[tuple[str, int]], list[ProbeResult], bool]:
    """Partition JSONL rows into active resumed rows and preserved other-anchor/arm rows."""
    by_attempt: dict[tuple[str, int], ProbeResult] = {}
    other_by_key: dict[tuple[tuple[str, ...], str, str, int], ProbeResult] = {}
    for row in raw_previous:
        if _is_resumable_row(
            row,
            active_query_ids=active_query_ids,
            corpus_digest=corpus_digest,
            catalog_skills=catalog_skills,
            config_fingerprint=config_fingerprint,
            condition_digest=condition_digest,
        ):
            by_attempt[(row.query_id, row.attempt)] = row
        elif (
            not row.error
            and (not corpus_digest or row.corpus_digest == corpus_digest)
            and (
                row.query_id not in active_query_ids
                or set(row.observed_catalog) != catalog_skills
                or (
                    bool(config_fingerprint)
                    and bool(row.config_fingerprint)
                    and row.config_fingerprint != config_fingerprint
                    and not (condition_digest and row.condition_digest == condition_digest)
                )
            )
        ):
            key = (
                tuple(sorted(row.observed_catalog)),
                row.config_fingerprint,
                row.query_id,
                row.attempt,
            )
            other_by_key[key] = row

    resumed = list(by_attempt.values())
    retained_other = list(other_by_key.values())
    needs_compaction = (len(resumed) + len(retained_other)) < len(raw_previous)
    return resumed, set(by_attempt.keys()), retained_other, needs_compaction


def completed_attempts(
    path: Path | str,
    *,
    active_query_ids: set[str] | None = None,
    corpus_digest: str = "",
    catalog_skills: set[str] | None = None,
    config_fingerprint: str = "",
    condition_digest: str = "",
) -> set[tuple[str, int]]:
    """Return completed, successful (query_id, attempt) pairs from existing results."""
    resolved = Path(path).expanduser()
    if not resolved.exists():
        return set()
    return {
        (r.query_id, r.attempt)
        for r in load_results(resolved)
        if _is_resumable_row(
            r,
            active_query_ids=active_query_ids,
            corpus_digest=corpus_digest,
            catalog_skills=catalog_skills,
            config_fingerprint=config_fingerprint,
            condition_digest=condition_digest,
        )
    }


def recorded_fingerprints(path: Path | str) -> set[str]:
    """Return all unique config fingerprints recorded in a results file."""
    resolved = Path(path).expanduser()
    if not resolved.exists():
        return set()
    return {result.config_fingerprint for result in load_results(resolved)}


def _digest_triples(path: Path | str) -> set[tuple[str, str, str]]:
    """Extract (fingerprint, condition_digest, queries_digest) tuples."""
    resolved = Path(path).expanduser()
    if not resolved.exists():
        return set()
    return {
        (result.config_fingerprint, result.condition_digest, result.queries_digest)
        for result in load_results(resolved)
    }


def _check_stale_fingerprints(
    path: Path | str,
    recorded_digests: set[tuple[str, str, str]],
    fingerprint: str,
    condition: str,
) -> None:
    """Validate that existing rows match the target configuration fingerprint."""
    stale = {
        recorded
        for recorded, taken_under, _ in recorded_digests
        if recorded != fingerprint and not (condition and taken_under == condition)
    }
    if stale:
        msg = (
            f"refusing to append to {path}: its rows were recorded under "
            f"{', '.join(sorted(f or '(unrecorded)' for f in stale))}, not {fingerprint}. "
            "These are separate measurements. Write to a new file, or pass "
            "append_across_arms to mix them anyway."
        )
        raise ValueError(msg)


def _check_stale_queries(
    path: Path | str,
    recorded_digests: set[tuple[str, str, str]],
    queries: str,
) -> None:
    """Validate that existing rows match the target query set digest."""
    asked = {recorded for _, _, recorded in recorded_digests if recorded != queries}
    if queries and asked:
        msg = (
            f"refusing to append to {path}: its rows were scored on ground truth "
            f"{', '.join(sorted(asked))}, not {queries}. The configuration matches "
            "but the questions do not, so the rows would pool as one measurement of "
            "two different query sets. Write to a new file, or pass append_across_arms "
            "to mix them anyway."
        )
        raise ValueError(msg)


def validate_appendable(
    path: Path | str,
    fingerprint: str,
    condition: str = "",
    queries: str = "",
) -> None:
    """Validate new probes match existing experimental conditions in results."""
    recorded_digests = _digest_triples(path)
    _check_stale_fingerprints(path, recorded_digests, fingerprint, condition)
    _check_stale_queries(path, recorded_digests, queries)


def load_corpus(config: RunConfig) -> list[Skill]:
    """Load skill corpus from configured path."""
    if config.study.skills is None:
        msg = (
            "no skill corpus: pass --skills, or use a verb that discovers them "
            "by asking the runtime"
        )
        raise ValueError(
            msg,
        )
    return load_skills(config.study.skills)


class Plan(BaseModel):
    """Summarize execution parameters, target sizes, and workloads."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    catalog_id: str
    catalog_size: int
    probes: int
    attempts: int = Field(default=0, ge=0)
    corpus_digest: str
    queries_digest: str = ""

    @property
    def per_query_resolution(self) -> float | None:
        """Calculate detectable effect size threshold for individual queries."""
        return detectable_delta(self.attempts)


class Composition(BaseModel):
    """Hold validated components including config, query set, catalog, and corpus."""

    model_config = ConfigDict(frozen=True)

    config: RunConfig
    query_set: QuerySet
    catalog: Catalog
    skills: tuple[Skill, ...]

    @property
    def provenance(self) -> Provenance:
        """Derive Provenance metadata from configuration, corpus, and query set."""
        return Provenance(
            config_fingerprint=self.config.fingerprint,
            condition_digest=self.config.condition,
            corpus_digest=corpus_digest(self.skills),
            queries_digest=query_set_digest(self.query_set),
        )

    @property
    def truth(self) -> dict[str, str]:
        """Return a mapping of query ID to ground truth skill label."""
        return {q.query_id: q.truth_label for q in self.query_set.queries}

    @property
    def plan(self) -> Plan:
        """Derive an execution Plan from the composed components."""
        return Plan(
            catalog_id=self.catalog.id,
            catalog_size=self.catalog.size,
            probes=len(self.query_set.queries) * self.config.plan.attempts,
            attempts=self.config.plan.attempts,
            corpus_digest=corpus_digest(self.skills),
            queries_digest=query_set_digest(self.query_set),
        )


def compose(config: RunConfig, skills: Sequence[Skill] | None = None) -> Composition:
    """Resolve and validate the target catalog and query set for an evaluation run."""
    query_set = load_query_set(config.require_queries())
    resolved = list(skills) if skills is not None else load_corpus(config)
    scorer = None
    if config.catalog.mode == CatalogMode.NEIGHBORHOOD and resolved:
        from reach.retrieval import build_scorer

        scorer = build_scorer(config.catalog.scorer, resolved, config)

    catalogs = build_catalogs(
        resolved,
        config.catalog.mode,
        size=config.catalog.size,
        rivals=config.catalog.rivals,
        seed=config.catalog.seed,
        scorer=scorer,
    )

    if not config.study.catalog or config.study.catalog.strip().lower() == "auto":
        wanted = query_set.catalog_id or "all"
    else:
        wanted = config.study.catalog

    if query_set.catalog_id and wanted != query_set.catalog_id and not config.study.rescope:
        msg = (
            f"query set was labeled in {query_set.catalog_id!r} but would be "
            f"probed against {wanted!r}: ground truth derived in one catalog is "
            "not valid in another. Pass --rescope to accept the labels anyway."
        )
        raise ValueError(
            msg,
        )
    catalog = resolve_catalog(catalogs, wanted)
    validate_query_coverage(query_set, catalog.skills, partial=config.study.partial)
    return Composition(
        config=config,
        query_set=query_set,
        catalog=catalog,
        skills=tuple(resolved),
    )


class RunOutcome(BaseModel):
    """Hold results, artifact, and full context models from a completed evaluation run."""

    model_config = ConfigDict(frozen=True)

    results: tuple[ProbeResult, ...]
    query_set: QuerySet
    catalog: Catalog
    skills: tuple[Skill, ...]
    config: RunConfig
    fit: CatalogFit | None = None
    spend_usd: float = 0.0
    reused: int = 0

    @property
    def composition(self) -> Composition:
        """Construct the Composition corresponding to this run outcome."""
        return Composition(
            config=self.config,
            query_set=self.query_set,
            catalog=self.catalog,
            skills=self.skills,
        )

    @property
    def report(self) -> Artifact:
        """Return the evaluation artifact."""
        return self.artifact()

    def artifact(
        self,
        *,
        roots: Sequence[Path] | None = None,
        contested: Sequence[ContestedSkill] = (),
        sample_queries: int = DEFAULT_SAMPLE_QUERIES,
    ) -> Artifact:
        """Construct the canonical Artifact document for this evaluation outcome."""
        return Artifact.assemble(
            self.composition,
            self.results,
            roots=roots,
            contested=contested,
            sample_queries=sample_queries,
            fit=self.fit,
            spend_usd=self.spend_usd,
            reused=self.reused,
        )


class _ProbeOutcomeCacheKey(BaseModel):
    """Identify a content-addressed probe outcome by resident skills and query."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    runtime_name: str
    runtime_model: str
    resident_skills: tuple[str, ...]
    query_id: str
    query_text: str
    expected_skill: str | None = None
    attempt: int = 1


class ProbeHarness:
    """Coordinate concurrent probe attempts, retries, and result streaming."""

    def __init__(
        self,
        runtime: AgentRuntime,
        *,
        workers: int = 1,
        retries: int = 2,
        backoff_s: float = 5.0,
        pause_s: float = 0.0,
        sleep: Callable[[float], None] = time.sleep,
        cache_outcomes: bool = True,
        outcome_cache: dict[_ProbeOutcomeCacheKey, Any] | None = None,
    ) -> None:
        """Initialize probe harness with runtime driver and execution options."""
        self.runtime = runtime
        self.workers = max(1, workers)
        self.retries = retries
        self.backoff_s = backoff_s
        self.pause_s = pause_s
        self.sleep = sleep
        self.cache_outcomes = cache_outcomes and runtime.name != "fake"
        self._cache_lock = threading.Lock()
        self._outcome_cache: dict[_ProbeOutcomeCacheKey, Any] = (
            outcome_cache if outcome_cache is not None else {}
        )

    def probe(
        self,
        query: Query,
        catalog: Catalog,
        workdir: Path,
        *,
        attempt: int = 1,
        provenance: Provenance | None = None,
        fit: CatalogFit | None = None,
    ) -> ProbeResult:
        """Execute a single query probe attempt with catalog residency validation."""
        elided = frozenset(fit.elided_skills) if fit is not None else frozenset()
        resident = tuple(s for s in catalog.skills if s not in elided)
        cache_key = _ProbeOutcomeCacheKey(
            runtime_name=self.runtime.name,
            runtime_model=self.runtime.model,
            resident_skills=resident,
            query_id=query.query_id,
            query_text=query.text,
            expected_skill=query.expected_skill,
            attempt=attempt,
        )
        outcome = None
        if self.cache_outcomes:
            with self._cache_lock:
                outcome = self._outcome_cache.get(cache_key)

        if outcome is None:
            try:
                try:
                    outcome = self.runtime.select(
                        query.text, workdir, target_skill=query.expected_skill
                    )
                except TypeError:
                    outcome = self.runtime.select(query.text, workdir)
            except Exception as err:
                err.add_note(
                    f"Reach probe execution context: query_id={query.query_id!r}, "
                    f"attempt={attempt}, catalog_id={catalog.id!r}, runtime={self.runtime.name!r}"
                )
                raise
            if self.cache_outcomes and not getattr(outcome, "error", None):
                with self._cache_lock:
                    self._outcome_cache[cache_key] = outcome

        is_dyn = getattr(self.runtime, "is_dynamic", False)
        error = outcome.error or validate_residency(
            catalog, outcome.observed_catalog, dynamic=is_dyn
        )
        return ProbeResult.from_outcome(
            outcome=outcome,
            query=query,
            catalog=catalog,
            runtime_name=self.runtime.name,
            model=self.runtime.model,
            attempt=attempt,
            provenance=provenance,
            error=error,
            fit=fit,
            is_dynamic=is_dyn,
        )

    def run_probes(
        self,
        queries: Sequence[Query],
        catalog: Catalog,
        workdir: Path,
        *,
        attempts: int = DEFAULT_ATTEMPTS,
        out_path: Path | None = None,
        skip: set[tuple[str, int]] | None = None,
        provenance: Provenance | None = None,
        fit: CatalogFit | None = None,
    ) -> Iterator[ProbeResult]:
        """Execute probe batches across queries with thread pooling, retries, and persistence."""
        planned = [
            (query, attempt)
            for query, attempt in plan_probes(queries, attempts)
            if (query.query_id, attempt) not in (skip or set())
        ]

        if self.workers <= 1:
            for query, attempt in planned:
                result = _with_retries(
                    functools.partial(
                        self.probe,
                        query,
                        catalog,
                        workdir,
                        attempt=attempt,
                        provenance=provenance,
                        fit=fit,
                    ),
                    self.retries,
                    self.backoff_s,
                    self.sleep,
                )
                if out_path is not None:
                    append_result(out_path, result)
                yield result
                if self.pause_s:
                    self.sleep(self.pause_s)
            return

        write_lock = threading.Lock()

        def _run_one(query: Query, attempt: int) -> ProbeResult:
            result = _with_retries(
                functools.partial(
                    self.probe,
                    query,
                    catalog,
                    workdir,
                    attempt=attempt,
                    provenance=provenance,
                    fit=fit,
                ),
                self.retries,
                self.backoff_s,
                self.sleep,
            )
            if out_path is not None:
                with write_lock:
                    append_result(out_path, result)
            if self.pause_s:
                self.sleep(self.pause_s)
            return result

        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = [executor.submit(_run_one, query, attempt) for query, attempt in planned]
            for future in as_completed(futures):
                yield future.result()

    def run(
        self,
        composition: Composition,
        *,
        out_path: Path | None = None,
        resume: bool = True,
        append_across_arms: bool = False,
        allow_truncation: bool = False,
        progress: Callable[[int, int, ProbeResult], None] | None = None,
    ) -> RunOutcome:
        """Execute an evaluation run end-to-end and persist results and sidecar configuration."""
        query_set, catalog, corpus = composition.query_set, composition.catalog, composition.skills
        provenance = composition.provenance
        config = composition.config

        resolved_out: Path | None = None
        if out_path is not None:
            resolved_out = Path(out_path).resolve()
            if not append_across_arms:
                validate_appendable(
                    resolved_out,
                    config.fingerprint,
                    config.condition,
                    provenance.queries_digest,
                )

        fit = validate_catalog_fit(self.runtime, catalog, corpus, allow_truncation=allow_truncation)
        workdir = config.require_workdir()
        workdir.mkdir(parents=True, exist_ok=True)
        self.runtime.install(catalog, corpus, workdir)

        previous: list[ProbeResult] = []
        retained_other: list[ProbeResult] = []
        skip: set[tuple[str, int]] = set()
        needs_compaction = False
        if resolved_out is not None:
            resolved_out.parent.mkdir(parents=True, exist_ok=True)
            if resume and resolved_out.exists():
                raw_previous = load_results(resolved_out)
                previous, skip, retained_other, needs_compaction = _partition_resumed_results(
                    raw_previous,
                    active_query_ids={q.query_id for q in query_set.queries},
                    corpus_digest=provenance.corpus_digest,
                    catalog_skills=set(catalog.skills),
                    config_fingerprint=provenance.config_fingerprint,
                    condition_digest=provenance.condition_digest,
                )

        total = len(query_set.queries) * config.plan.attempts
        results = list(previous)
        for index, result in enumerate(
            self.run_probes(
                query_set.queries,
                catalog,
                workdir,
                attempts=config.plan.attempts,
                out_path=resolved_out,
                skip=skip,
                provenance=provenance,
                fit=fit,
            ),
            start=len(skip) + 1,
        ):
            results.append(result)
            if progress is not None:
                progress(index, total, result)

        if resolved_out is not None:
            if resume and needs_compaction:
                write_results(resolved_out, [*retained_other, *results])
            write_sidecar(config, resolved_out)

        return RunOutcome(
            results=tuple(results),
            query_set=query_set,
            catalog=catalog,
            skills=tuple(corpus),
            config=config,
            fit=fit,
            spend_usd=sum(r.cost_usd or 0.0 for r in results),
            reused=len(skip),
        )


def conduct(
    config: RunConfig,
    runtime: AgentRuntime | None = None,
    *,
    progress: Callable[[int, int, ProbeResult], None] | None = None,
    resume: bool = True,
    append_across_arms: bool = False,
    allow_truncation: bool = False,
    composed: Composition | None = None,
    workers: int | None = None,
    outcome_cache: dict[_ProbeOutcomeCacheKey, Any] | None = None,
) -> RunOutcome:
    """Execute an evaluation run end-to-end and return the full RunOutcome."""
    resolved_runtime = runtime or build_runtime(config.runtime)
    composition = composed if composed is not None else compose(config)
    resolved_workers = workers if workers is not None else config.plan.workers
    harness = ProbeHarness(
        resolved_runtime,
        workers=resolved_workers,
        retries=config.plan.retries,
        backoff_s=config.plan.backoff_s,
        pause_s=config.plan.pause_s,
        outcome_cache=outcome_cache,
    )
    return harness.run(
        composition,
        out_path=config.study.out,
        resume=resume,
        append_across_arms=append_across_arms,
        allow_truncation=allow_truncation,
        progress=progress,
    )


def evaluate(
    config: RunConfig,
    runtime: AgentRuntime | None = None,
    *,
    skills: Sequence[Skill] | None = None,
    auto_draft: bool = False,
    progress: Callable[[int, int, ProbeResult], None] | None = None,
    resume: bool = True,
    append_across_arms: bool = False,
    allow_truncation: bool = False,
    composed: Composition | None = None,
    workers: int | None = None,
) -> RunOutcome:
    """Execute an evaluation run, resolving or drafting queries, and probing skills.

    Args:
        config: Run configuration specifying study parameters and options.
        runtime: Optional AgentRuntime instance to use; if None, built from config.
        skills: Optional sequence of pre-loaded Skill instances; if None, loaded from corpus.
        auto_draft: Whether to automatically draft queries if not already present on disk.
        progress: Optional callback invoked as each probe completes.
        resume: Whether to resume from existing results on disk.
        append_across_arms: Whether to append results across different study arms.
        allow_truncation: Whether to proceed if the catalog exceeds runtime prompt limits.
        composed: Optional pre-constructed Composition instance.
        workers: Number of concurrent workers executing probes.

    Returns:
        A RunOutcome instance containing results, catalog, query set, and artifact.
    """
    if composed is not None:
        target_composition = composed
    else:
        resolved_skills = list(skills) if skills is not None else load_corpus(config)

        queries_path = config.study.queries
        has_queries = queries_path is not None and queries_path.exists()

        if not has_queries and auto_draft:
            from reach.catalog import build_catalogs, resolve_catalog
            from reach.generate import generate_query_set, text_generator
            from reach.queries import save_query_set

            scorer = None
            if config.catalog.mode == CatalogMode.NEIGHBORHOOD and resolved_skills:
                from reach.retrieval import build_scorer

                scorer = build_scorer(config.catalog.scorer, resolved_skills, config)

            catalogs = build_catalogs(
                resolved_skills,
                config.catalog.mode,
                size=config.catalog.size,
                rivals=config.catalog.rivals,
                seed=config.catalog.seed,
                scorer=scorer,
            )
            wanted = (
                config.study.catalog
                if config.study.catalog and config.study.catalog.strip().lower() != "auto"
                else (catalogs[0].id if catalogs else "all")
            )
            target_catalog = resolve_catalog(catalogs, wanted)
            model_opt = config.runtime.resolved_options().get("model")
            model_str = (
                str(model_opt)
                if model_opt
                else (agent_default_model(config.runtime.agent) or DEFAULT_GEMINI_MODEL)
            )
            generator = text_generator(
                agent=config.runtime.agent,
                model=model_str,
            )
            query_set = generate_query_set(
                target_catalog,
                resolved_skills,
                runtime=generator,
            )
            if queries_path is not None:
                save_query_set(query_set, queries_path)
            target_composition = Composition(
                config=config,
                query_set=query_set,
                catalog=target_catalog,
                skills=tuple(resolved_skills),
            )
        else:
            target_composition = compose(config, resolved_skills)

    return conduct(
        config,
        runtime=runtime,
        progress=progress,
        resume=resume,
        append_across_arms=append_across_arms,
        allow_truncation=allow_truncation,
        composed=target_composition,
        workers=workers,
    )


def plan_only(config: RunConfig, skills: Sequence[Skill] | None = None) -> Plan:
    """Plan and validate an evaluation run without executing probes."""
    return compose(config, skills).plan


class ConfigSidecar(BaseModel):
    """Record provenance digests and configuration beside a results file."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    arm: str
    fingerprint: str
    condition: str
    config: RunConfig


#: Required top-level keys for valid sidecar metadata documents.
SIDECAR_KEYS = ("arm", "fingerprint", "condition", "config")

#: Filesystem suffix appended to results files for sidecar configuration storage.
SIDECAR_SUFFIX = ".config.json"


def sidecar_path(results_path: Path) -> Path:
    """Return corresponding sidecar configuration path for a results file."""
    return Path(f"{results_path}{SIDECAR_SUFFIX}")


def write_sidecar(config: RunConfig, out_path: Path) -> Path:
    """Write serialized sidecar configuration beside the output results file."""
    sidecar = sidecar_path(out_path)
    text = json.dumps(_sidecar_document(config), indent=2) + "\n"
    sidecar.write_text(text, encoding="utf-8")
    return sidecar


def _sidecar_document(config: RunConfig) -> dict[str, Any]:
    """Generate the dictionary representation of a RunConfig sidecar document."""
    return {
        "arm": config.arm,
        "fingerprint": config.fingerprint,
        "condition": config.condition,
        "config": json.loads(config.model_dump_json()),
    }


def read_sidecar(path: Path) -> ConfigSidecar:
    """Load and validate sidecar configuration metadata from disk."""
    document = json.loads(path.read_text(encoding="utf-8"))
    if missing := [key for key in SIDECAR_KEYS if key not in document]:
        msg = (
            f"{path} predates recorded digests (no {', '.join(missing)}); "
            "its arm cannot be established from the file alone"
        )
        raise ValueError(msg)
    sidecar = ConfigSidecar.model_validate(document)
    keys = ("arm", "fingerprint", "condition")
    for key in keys:
        recorded, derived = getattr(sidecar, key), getattr(sidecar.config, key)
        if recorded != derived:
            msg = (
                f"{path}: recorded {key} {recorded} does not match {derived}, "
                "derived from the configuration beside it"
            )
            raise ValueError(msg)
    return sidecar
