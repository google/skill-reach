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

"""Orchestrate the two-stage CI/CD regression check and quality gate."""

from __future__ import annotations

import fnmatch
import re
import subprocess
import tempfile
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict

from reach.catalog import deduplicate_skills, load_skills
from reach.config import (
    CheckSettings,
    RunConfig,
    RuntimeSettings,
    resolve_path,
)
from reach.lint import LintReport, LintSettings, Severity, lint_skills, lint_tree
from reach.metrics import ClassificationReport, classification_report
from reach.models import NO_SKILL, Catalog, CatalogMode, ProbeResult, Query, Skill
from reach.queries import load_query_set
from reach.runtime import FAKE_AGENT, AgentRuntime, build_runtime
from reach.runtime.keyword import KeywordRuntime

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Mapping, Sequence

#: Pattern matching skill directories or files in git diff output.
_SKILL_PATH_PATTERN = re.compile(
    r"(?:^|/)(?:skills|\.agents/skills|\.claude/skills|\.cursor/skills|\.github/skills|\.copilot/skills|\.pi/skills|\.pi/agent/skills|\.codex/skills)/([^/]+)/"
)


__all__ = [
    "CheckAssertion",
    "CheckOutcome",
    "CheckStage",
    "EmpiricalMetrics",
    "changed_skills",
    "run_check",
]


class CheckStage(StrEnum):
    """Enumerate execution stages in the CI quality gate."""

    EMPIRICAL = "empirical"
    STATIC = "static"


class EmpiricalMetrics(BaseModel):
    """Aggregated empirical metrics observed during quality gate execution."""

    model_config = ConfigDict(frozen=True)

    recall: float
    accuracy: float
    misroute_rate: float
    entrypoint_accuracy: float = 0.0
    trajectory_reachability: float = 0.0
    step_efficiency: float = 0.0
    skill_f1: float = 0.0
    redundancy: float = 0.0


class CheckAssertion(BaseModel):
    """Single threshold assertion evaluated during empirical quality gate."""

    model_config = ConfigDict(frozen=True)

    comparison: str
    message: str
    name: str
    observed: float
    passed: bool
    threshold: float


class CheckOutcome(BaseModel):
    """Comprehensive outcome of a two-stage quality gate check."""

    model_config = ConfigDict(frozen=True)

    assertions: tuple[CheckAssertion, ...] = ()
    budget: int = 50
    budget_exhausted: bool = False
    exit_code: int = 0
    lint_report: LintReport
    probes_executed: int = 0
    queries_probed: int = 0
    skills_checked: int = 0
    stage_failed: CheckStage | None = None
    classification: ClassificationReport | None = None

    @property
    def passed(self) -> bool:
        """Return True if all stages and assertions passed."""
        return self.exit_code == 0


def changed_skills(
    since: str = "HEAD~1",
    root: Path | str | None = None,
    *,
    timeout: float = 30.0,
) -> tuple[str, ...]:
    """Identify skill names modified in git repository relative to a reference."""
    work_dir = Path(root).resolve() if root is not None else Path.cwd().resolve()
    clean_since = since.strip()
    if clean_since.startswith("-"):
        msg = f"git reference must not begin with a dash: {since!r}"
        raise ValueError(msg)
    try:
        completed = subprocess.run(
            ["git", "diff", "--name-only", clean_since, "--"],
            capture_output=True,
            text=True,
            cwd=work_dir,
            check=False,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        msg = f"git diff timed out after {timeout}s against ref '{since}'"
        raise ValueError(msg) from exc
    except OSError:
        return ()

    if completed.returncode != 0:
        err = completed.stderr.strip() or "git command failed"
        msg = f"git diff failed against ref '{since}': {err}"
        err_lower = err.lower()
        if any(
            pattern in err_lower
            for pattern in ("bad revision", "unknown revision", "shallow", "ambiguous argument")
        ):
            msg += (
                f"\nHint: Git reference '{since}' may not exist in this clone. "
                "If running in CI (e.g. GitHub Actions), ensure full git history is fetched "
                "(e.g. 'fetch-depth: 0')."
            )
        raise ValueError(msg)

    discovered: set[str] = set()
    for line in completed.stdout.splitlines():
        trimmed = line.strip()
        if not trimmed:
            continue
        if match := _SKILL_PATH_PATTERN.search(trimmed):
            discovered.add(match.group(1))
        elif trimmed.endswith("SKILL.md"):
            skill_name = Path(trimmed).parent.name or Path(work_dir).resolve().name
            if skill_name:
                discovered.add(skill_name)

    return tuple(sorted(discovered))


def _resolve_candidate_skills(
    skills_paths: Sequence[Path | str] | None,
    lint_config: LintSettings | None = None,
    agent: str | None = None,
    config_path: Path | str | None = None,
    global_scope: bool = False,
) -> tuple[list[Path], LintReport]:
    """Resolve skill files or directories and perform Stage 1 static linting."""
    if skills_paths:
        resolved = [resolve_path(p) for p in skills_paths]
        for p in resolved:
            if not p.exists():
                msg = f"skill path does not exist: {p}"
                raise ValueError(msg)
        # If single directory containing multiple sub-skills
        if len(resolved) == 1 and resolved[0].is_dir() and not (resolved[0] / "SKILL.md").exists():
            report = lint_tree(resolved[0], config=lint_config)
            return [resolved[0]], report
        report = lint_skills(resolved, config=lint_config)
        return resolved, report

    # Auto-discover from project or global skill locations in precedence order
    from reach.config import resolve_discovery_candidates

    workdir = Path.home() if global_scope else Path.cwd().resolve()
    for candidate in resolve_discovery_candidates(
        workdir,
        agent=agent,
        config_path=config_path,
        global_scope=global_scope,
    ):
        if candidate == workdir:
            if (candidate / "SKILL.md").is_file():
                report = lint_skills([candidate], config=lint_config)
                return [candidate], report
        elif candidate.is_dir():
            report = lint_tree(candidate, config=lint_config)
            if report.skills_checked > 0:
                return [candidate.resolve()], report

    empty_report = LintReport(issues=(), skills_checked=0)
    return [], empty_report


def _setup_runtime(
    agent: str | None,
    skills: Sequence[Skill],  # noqa: ARG001
    runtime_options: dict[str, Any] | None = None,
    config: RunConfig | None = None,
) -> AgentRuntime:
    """Build or mock agent runtime for Stage 2 empirical evaluation."""
    from reach.config import default_agent

    resolved_agent = agent or (config.runtime.agent if config is not None else default_agent())
    if resolved_agent in ("keyword", FAKE_AGENT):
        return KeywordRuntime()

    settings = RuntimeSettings(agent=resolved_agent, options=runtime_options or {})
    return build_runtime(settings)


def _load_catalog_skills(resolved_paths: Sequence[Path]) -> list[Skill]:
    """Load resident skills from resolved candidate paths, deduplicated by name."""
    loaded_skills: list[Skill] = []
    for path in resolved_paths:
        if path.is_dir() and (path / "SKILL.md").exists():
            loaded_skills.extend(load_skills(path.parent))
        elif path.is_dir():
            loaded_skills.extend(load_skills(path))
        elif path.is_file() and path.name == "SKILL.md":
            loaded_skills.extend(load_skills(path.parent.parent))
    return deduplicate_skills(loaded_skills)


def _competing_neighbors_for_modified(
    modified: set[str],
    skills: Sequence[Skill],
) -> set[str]:
    """Identify competing neighbor skills that could be hijacked by modified skills."""
    from reach.lint import find_competing_neighbors

    return find_competing_neighbors(modified, skills)


def _filter_check_queries(
    queries_path: Path | str,
    modified: set[str],
    changed: bool,
    budget: int,
    skills: Sequence[Skill] = (),
    filter_skill: str | Sequence[str] | None = None,
    filter_id: str | Sequence[str] | None = None,
) -> tuple[list[Query], bool]:
    """Load, filter by modified skills, specific skills or IDs, and slice by budget."""
    query_set = load_query_set(queries_path)
    all_queries = list(query_set.queries)
    if filter_skill is not None:
        target_skills = [filter_skill] if isinstance(filter_skill, str) else list(filter_skill)
        all_queries = [
            q
            for q in all_queries
            if q.expected_skill is not None
            and any(fnmatch.fnmatchcase(q.expected_skill, pat) for pat in target_skills)
        ]
    if filter_id is not None:
        target_ids = [filter_id] if isinstance(filter_id, str) else list(filter_id)
        all_queries = [
            q for q in all_queries if any(fnmatch.fnmatchcase(q.id, pat) for pat in target_ids)
        ]
    if changed:
        primary = [q for q in all_queries if q.expected_skill in modified]
        neighbors = _competing_neighbors_for_modified(modified, skills)
        guardrails = [q for q in all_queries if q.expected_skill in neighbors]
        all_queries = primary + guardrails
    total_queries = len(all_queries)
    budget_exhausted = total_queries > budget
    return all_queries[:budget], budget_exhausted


class _CheckCacheKey(BaseModel):
    """Represent a deterministic, hashable in-memory cache key for a check probe."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    paths_key: tuple[str, ...]
    runtime_name: str
    runtime_model: str | None
    opts_key: str
    corpus_digest: str
    query_id: str
    query_text: str
    expected_skill: str | None
    acceptable_skills: tuple[str, ...]

    @classmethod
    def for_query(
        cls,
        *,
        paths_key: tuple[str, ...],
        runtime_name: str,
        runtime_model: str | None,
        opts_key: str,
        corpus_digest: str,
        query: Query,
    ) -> _CheckCacheKey:
        """Construct a frozen cache key for a single Query."""
        return cls(
            paths_key=paths_key,
            runtime_name=runtime_name,
            runtime_model=runtime_model,
            opts_key=opts_key,
            corpus_digest=corpus_digest,
            query_id=query.id,
            query_text=query.text,
            expected_skill=query.expected_skill,
            acceptable_skills=tuple(sorted(query.acceptable_skills)),
        )


_CHECK_PROBE_CACHE: dict[_CheckCacheKey, ProbeResult] = {}


def _execute_empirical_probes(
    queries_to_run: Sequence[Query],
    resolved_paths: Sequence[Path],
    agent: str | None,
    runtime_options: dict[str, Any] | None,
    config: RunConfig | None,
) -> tuple[ClassificationReport, EmpiricalMetrics, int]:
    """Execute probes in an isolated workspace and compute classification metrics."""
    import json

    from reach.catalog import corpus_digest
    from reach.run import ProbeHarness

    loaded_skills = _load_catalog_skills(resolved_paths)
    c_digest = corpus_digest(loaded_skills)
    paths_key = tuple(str(p.resolve()) for p in resolved_paths)
    opts_key = json.dumps(runtime_options or {}, sort_keys=True)
    catalog = Catalog(
        id="check-catalog",
        skills=tuple(s.name for s in loaded_skills),
        mode=CatalogMode.ALL,
    )
    runtime = _setup_runtime(agent, loaded_skills, runtime_options, config=config)
    use_cache = runtime.name != FAKE_AGENT

    def _key_for(query: Query) -> _CheckCacheKey:
        return _CheckCacheKey.for_query(
            paths_key=paths_key,
            runtime_name=runtime.name,
            runtime_model=runtime.model,
            opts_key=opts_key,
            corpus_digest=c_digest,
            query=query,
        )

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            workdir = Path(temp_dir)
            fit = runtime.fit(catalog, loaded_skills)
            cached_by_id: dict[str, ProbeResult] = {}
            missing_queries: list[Query] = []
            for q in queries_to_run:
                key = _key_for(q)
                if use_cache and key in _CHECK_PROBE_CACHE:
                    cached_by_id[q.id] = _CHECK_PROBE_CACHE[key]
                else:
                    missing_queries.append(q)

            if missing_queries:
                runtime.install(catalog, loaded_skills, workdir)
                harness = ProbeHarness(runtime)
                fresh_results = list(
                    harness.run_probes(
                        missing_queries,
                        catalog,
                        workdir,
                        attempts=1,
                        out_path=None,
                        fit=fit,
                    )
                )
                for q, res in zip(missing_queries, fresh_results, strict=True):
                    cached_by_id[q.id] = res
                    if use_cache and not res.error:
                        _CHECK_PROBE_CACHE[_key_for(q)] = res

            results = [cached_by_id[q.id] for q in queries_to_run]
    finally:
        runtime.cleanup()

    report = classification_report(results, queries_to_run)
    total_triggers = sum(c.true_positives for c in report.per_class if c.label != NO_SKILL)
    obs_recall = total_triggers / report.in_scope if report.in_scope else 1.0
    obs_accuracy = report.top1_accuracy
    misroutes = sum(
        1
        for q, r in zip(queries_to_run, results, strict=True)
        if (eff := q.effective_invoked_skill(r)) is not None and not q.matches_skill(eff)
    )
    obs_misroute = misroutes / report.scored if report.scored else 0.0

    metrics = EmpiricalMetrics(
        recall=obs_recall,
        accuracy=obs_accuracy,
        misroute_rate=obs_misroute,
        entrypoint_accuracy=report.entrypoint_accuracy,
        trajectory_reachability=report.trajectory_reachability,
        step_efficiency=report.step_efficiency,
        skill_f1=report.skill_f1,
        redundancy=report.redundancy,
    )
    return report, metrics, len(results)


def _build_check_assertions(
    metrics: EmpiricalMetrics,
    settings: CheckSettings | None = None,
) -> tuple[CheckAssertion, ...]:
    """Evaluate empirical thresholds and construct diagnostic assertions."""
    settings = settings if settings is not None else CheckSettings()

    specs: tuple[tuple[str, str, float, float | None, bool, str], ...] = (
        ("recall", "Recall", metrics.recall, settings.min_recall, False, ".1%"),
        ("accuracy", "Accuracy", metrics.accuracy, settings.min_accuracy, False, ".1%"),
        (
            "misroute_rate",
            "Misroute rate",
            metrics.misroute_rate,
            settings.max_misroute,
            True,
            ".1%",
        ),
        (
            "entrypoint",
            "Entrypoint accuracy",
            metrics.entrypoint_accuracy,
            settings.min_entrypoint,
            False,
            ".1%",
        ),
        (
            "reachability",
            "Reachability",
            metrics.trajectory_reachability,
            settings.min_reachability,
            False,
            ".1%",
        ),
        (
            "step_efficiency",
            "Step efficiency",
            metrics.step_efficiency,
            settings.min_efficiency,
            False,
            ".3f",
        ),
        ("skill_f1", "Skill F1", metrics.skill_f1, settings.min_f1, False, ".1%"),
        (
            "redundancy",
            "Redundancy",
            metrics.redundancy,
            settings.max_redundancy,
            True,
            "+.2f",
        ),
    )

    assertions: list[CheckAssertion] = []
    for name, label, observed, threshold, is_upper_bound, fmt in specs:
        if threshold is None:
            continue
        passed = observed <= threshold if is_upper_bound else observed >= threshold
        obs_str = f"{observed:{fmt}}" if not fmt.startswith("+") else f"+{observed:.2f}"
        thresh_str = f"{threshold:{fmt}}" if not fmt.startswith("+") else f"+{threshold:.2f}"
        if is_upper_bound:
            msg = (
                f"{label} ({obs_str}) within threshold ({thresh_str})"
                if passed
                else (
                    f"{label} regression: observed {obs_str} exceeds maximum allowed {thresh_str}"
                )
            )
        else:
            msg = (
                f"{label} ({obs_str}) meets threshold ({thresh_str})"
                if passed
                else (f"{label} regression: observed {obs_str} is below required {thresh_str}")
            )
        assertions.append(
            CheckAssertion(
                name=name,
                passed=passed,
                observed=round(observed, 4),
                threshold=threshold,
                comparison="<=" if is_upper_bound else ">=",
                message=msg,
            )
        )
    return tuple(assertions)


def _apply_changed_scope(
    lint_report: LintReport,
    changed: bool,
    since: str,
    budget: int,
    available: Collection[str] = (),
) -> tuple[LintReport, set[str], CheckOutcome | None]:
    """Filter lint report to changed skills or return early clean outcome if none changed."""
    if not changed:
        return lint_report, set(), None
    # A diff reports deleted and renamed skills too, and those can never be probed
    # against the current corpus, so scope the gate to skills still on disk.
    modified = set(changed_skills(since=since)) & set(available)
    if not modified:
        return (
            LintReport(issues=(), skills_checked=0),
            set(),
            CheckOutcome(
                lint_report=LintReport(issues=(), skills_checked=0),
                skills_checked=0,
                queries_probed=0,
                probes_executed=0,
                budget=budget,
                exit_code=0,
            ),
        )
    scoped_report = LintReport(
        issues=tuple(i for i in lint_report.issues if i.skill in modified),
        skills_checked=len(modified),
    )
    return scoped_report, modified, None


def _check_static_gate_or_early_exit(
    lint_report: LintReport,
    strict: bool,
    budget: int,
    queries_path: Path | str | None,
) -> CheckOutcome | None:
    """Validate Stage 1 static gate and check for query-free early completion."""
    if lint_report.has_errors or (strict and bool(lint_report.warnings)):
        return CheckOutcome(
            lint_report=lint_report,
            stage_failed=CheckStage.STATIC,
            skills_checked=lint_report.skills_checked,
            budget=budget,
            exit_code=1,
        )

    if queries_path is None:
        return CheckOutcome(
            lint_report=lint_report,
            skills_checked=lint_report.skills_checked,
            budget=budget,
            exit_code=0,
        )
    return None


def _check_empty_queries_exit(
    lint_report: LintReport,
    queries_to_run: Sequence[Query],
    budget: int,
) -> CheckOutcome | None:
    """Return early pass outcome when filtered empirical query set is empty."""
    if not queries_to_run:
        return CheckOutcome(
            lint_report=lint_report,
            skills_checked=lint_report.skills_checked,
            queries_probed=0,
            probes_executed=0,
            budget=budget,
            exit_code=0,
        )
    return None


def run_check(  # noqa: PLR0913
    *,
    skills_paths: Sequence[Path | str] | None = None,
    queries_path: Path | str | None = None,
    changed: bool = False,
    since: str = "HEAD~1",
    strict: bool = True,
    min_recall: float = 0.80,
    min_accuracy: float = 0.80,
    max_misroute: float = 0.10,
    min_entrypoint: float | None = None,
    min_reachability: float | None = None,
    min_efficiency: float | None = None,
    min_f1: float | None = None,
    max_redundancy: float | None = None,
    budget: int = 50,
    config: RunConfig | None = None,
    settings: CheckSettings | None = None,
    agent: str | None = None,
    rule_overrides: Mapping[str, Severity] | None = None,
    runtime_options: dict[str, Any] | None = None,
    global_scope: bool = False,
    confirm_callback: Callable[[str, list[Skill], Sequence[Path]], int] | None = None,
    filter_skill: str | Sequence[str] | None = None,
    filter_id: str | Sequence[str] | None = None,
) -> CheckOutcome:
    """Execute two-stage quality gate: static lint pre-flight then empirical assertions."""
    if settings is not None:
        check_settings = settings
    elif config is not None:
        check_settings = config.check
    else:
        check_settings = CheckSettings(
            min_recall=min_recall,
            min_accuracy=min_accuracy,
            max_misroute=max_misroute,
            min_entrypoint=min_entrypoint,
            min_reachability=min_reachability,
            min_efficiency=min_efficiency,
            min_f1=min_f1,
            max_redundancy=max_redundancy,
            budget=budget,
            strict=strict,
            since=since,
        )

    if check_settings.budget < 1:
        msg = "Probe budget must be at least 1"
        raise ValueError(msg)

    lint_cfg = LintSettings.from_settings(overrides=rule_overrides)
    resolved_paths, lint_report = _resolve_candidate_skills(
        skills_paths,
        lint_config=lint_cfg,
        agent=agent,
        global_scope=global_scope,
    )
    if not resolved_paths:
        scope_msg = (
            "in user global configuration (~/.agents/skills, ~/.claude/skills, etc.)"
            if global_scope
            else "in .agents/skills or skills"
        )
        msg = f"No skill paths specified and no skills found {scope_msg}."
        raise ValueError(msg)

    catalog_skills = _load_catalog_skills(resolved_paths) if changed else []
    available = {s.name for s in catalog_skills} if changed else set()
    lint_report, modified, early_outcome = _apply_changed_scope(
        lint_report, changed, check_settings.since, check_settings.budget, available
    )
    if early_outcome is not None:
        return early_outcome

    static_outcome = _check_static_gate_or_early_exit(
        lint_report, check_settings.strict, check_settings.budget, queries_path
    )
    if static_outcome is not None:
        return static_outcome
    if queries_path is None:
        msg = "queries_path must be provided for empirical assertions"
        raise ValueError(msg)

    queries_to_run, budget_exhausted = _filter_check_queries(
        queries_path,
        modified,
        changed,
        check_settings.budget,
        skills=catalog_skills,
        filter_skill=filter_skill,
        filter_id=filter_id,
    )

    empty_outcome = _check_empty_queries_exit(lint_report, queries_to_run, check_settings.budget)
    if empty_outcome is not None:
        return empty_outcome

    from reach.config import default_agent

    resolved_agent = agent or (config.runtime.agent if config is not None else default_agent())
    if confirm_callback is not None and resolved_agent not in ("keyword", FAKE_AGENT):
        loaded = _load_catalog_skills(resolved_paths)
        if code := confirm_callback(resolved_agent, loaded, resolved_paths):
            return CheckOutcome(
                lint_report=lint_report,
                stage_failed=CheckStage.EMPIRICAL,
                assertions=(),
                skills_checked=lint_report.skills_checked,
                queries_probed=0,
                probes_executed=0,
                budget=check_settings.budget,
                exit_code=code,
            )

    report, metrics, probes_executed = _execute_empirical_probes(
        queries_to_run,
        resolved_paths,
        agent,
        runtime_options,
        config=config,
    )

    assertions = _build_check_assertions(metrics, check_settings)

    all_passed = all(a.passed for a in assertions)

    return CheckOutcome(
        lint_report=lint_report,
        stage_failed=None if all_passed else CheckStage.EMPIRICAL,
        assertions=assertions,
        skills_checked=lint_report.skills_checked,
        queries_probed=len(queries_to_run),
        probes_executed=probes_executed,
        budget=check_settings.budget,
        budget_exhausted=budget_exhausted,
        exit_code=0 if all_passed else 2,
        classification=report,
    )
