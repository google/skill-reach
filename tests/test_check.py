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

"""Verify the two-stage CI/CD quality gate engine (`reach check`)."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from reach.check import (
    CheckStage,
    EmpiricalMetrics,
    _build_check_assertions,
    changed_skills,
    run_check,
)
from reach.config import CheckSettings, RunConfig
from reach.models import Query
from reach.runtime import SelectionOutcome

if TYPE_CHECKING:
    from collections.abc import Callable


def test_stage1_fail_fast_on_error(
    write_skill: Callable[..., Path],
    write_queries: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """Verify that Stage 1 static errors abort execution instantly with exit code 1."""
    # Create a skill with invalid naming
    write_skill(
        name="BadName",
        description="A sufficiently detailed description that satisfies standard rules.",
    )
    queries_file = write_queries(target="BadName", count=5)

    outcome = run_check(
        skills_paths=[tmp_path / "BadName"],
        queries_path=queries_file,
    )

    assert outcome.exit_code == 1
    assert outcome.stage_failed is CheckStage.STATIC
    assert outcome.lint_report.has_errors
    assert outcome.probes_executed == 0  # Abort before issuing probes or API calls


def test_stage1_fail_fast_on_strict_warning(
    write_skill: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """Verify that Stage 1 warnings fail under strict=True with exit code 1."""
    # Verify short descriptions trigger warnings under default configuration
    write_skill(
        name="short-skill",
        description="Short desc",
    )

    outcome = run_check(
        skills_paths=[tmp_path / "short-skill"],
        strict=True,
    )

    assert outcome.exit_code == 1
    assert outcome.stage_failed is CheckStage.STATIC
    assert outcome.lint_report.warnings
    assert outcome.probes_executed == 0


def test_stage1_pass_on_warning_when_not_strict(
    write_skill: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """Verify that Stage 1 warnings pass under strict=False."""
    write_skill(
        name="short-skill",
        description="Short desc",
    )

    outcome = run_check(
        skills_paths=[tmp_path / "short-skill"],
        strict=False,
    )

    assert outcome.exit_code == 0
    assert outcome.stage_failed is None


def test_stage1_clean_without_queries_exits_0(
    write_skill: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """Verify that clean skill manifests without queries pass Stage 1 and exit 0."""
    write_skill(
        name="valid-skill",
        description="A sufficiently detailed description that satisfies standard rules.",
    )

    outcome = run_check(skills_paths=[tmp_path / "valid-skill"])

    assert outcome.exit_code == 0
    assert outcome.stage_failed is None
    assert outcome.skills_checked == 1
    assert outcome.probes_executed == 0


def test_stage2_empirical_pass(
    write_skill: Callable[..., Path],
    write_queries: Callable[..., Path],
) -> None:
    """Verify Stage 2 passes when empirical assertions meet configured thresholds."""
    skill_dir = write_skill(
        name="calc-skill",
        description="Perform calculations and math conversions accurately and quickly.",
    )
    queries_file = write_queries(target="calc-skill", count=4)

    outcome = run_check(
        skills_paths=[skill_dir],
        queries_path=queries_file,
        agent="keyword",
        min_recall=0.75,
        min_accuracy=0.75,
        max_misroute=0.20,
    )

    assert outcome.exit_code == 0
    assert outcome.stage_failed is None
    assert outcome.probes_executed > 0
    assert len(outcome.assertions) > 0
    assert all(a.passed for a in outcome.assertions)


def test_stage2_empirical_regression_recall(
    write_skill: Callable[..., Path],
    write_queries: Callable[..., Path],
) -> None:
    """Verify Stage 2 fails with exit code 2 when observed recall drops below threshold."""
    skill_dir = write_skill(
        name="deploy-skill",
        description="Deploy containerized applications to cloud platforms seamlessly.",
    )
    queries_file = write_queries(target="deploy-skill", count=4)

    outcome = run_check(
        skills_paths=[skill_dir],
        queries_path=queries_file,
        agent="keyword",
        min_recall=1.0,  # Enforce 100% recall requirement
    )

    if not all(a.passed for a in outcome.assertions):
        assert outcome.exit_code == 2
        assert outcome.stage_failed is CheckStage.EMPIRICAL
        failed = [a for a in outcome.assertions if not a.passed]
        assert len(failed) > 0


def test_stage2_probe_budget_limit(
    write_skill: Callable[..., Path],
    write_queries: Callable[..., Path],
) -> None:
    """Verify check respects probe budget upper bound."""
    skill_dir = write_skill(
        name="math-skill",
        description="A sufficiently detailed description that satisfies standard rules.",
    )
    queries_file = write_queries(target="math-skill", count=10)

    outcome = run_check(
        skills_paths=[skill_dir],
        queries_path=queries_file,
        agent="keyword",
        budget=2,
    )

    assert outcome.probes_executed <= 2


def test_changed_skills_parsing() -> None:
    """Verify changed_skills extracts modified skill names from git diff output."""
    mock_diff = (
        ".agents/skills/deploy-service/SKILL.md\n"
        "src/reach/main.py\n"
        "skills/code-review/SKILL.md\n"
        ".claude/skills/lint-code/scripts/run.py\n"
        ".cursor/skills/editor-skill/SKILL.md\n"
        ".pi/skills/pi-task/helper.py\n"
        "skills/code-review/tests/test_aux.py\n"
        ".agents/skills/README.md\n"
        "README.md\n"
    )
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = mock_diff

        changed = changed_skills(since="origin/main")
        assert "deploy-service" in changed
        assert "code-review" in changed
        assert "lint-code" in changed
        assert "editor-skill" in changed
        assert "pi-task" in changed
        assert "main.py" not in changed
        assert len(changed) == 5


def test_changed_skills_bare_skill_md_in_root_uses_work_dir_name(tmp_path: Path) -> None:
    """Verify changed_skills infers skill name from work_dir when modified file is root SKILL.md."""
    skill_root = tmp_path / "standalone-my-skill"
    skill_root.mkdir()
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "SKILL.md\n"

        changed = changed_skills(root=skill_root, since="HEAD~1")
        assert changed == ("standalone-my-skill",)


def test_changed_skills_git_error_raises_value_error() -> None:
    """Verify changed_skills raises ValueError when git diff fails."""
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 128
        mock_run.return_value.stderr = "fatal: bad revision 'bad-ref'"

        with pytest.raises(ValueError, match="git diff failed"):
            changed_skills(since="bad-ref")


def test_changed_skills_rejects_flag_injection() -> None:
    """Verify changed_skills raises ValueError when since starts with a dash."""
    with pytest.raises(ValueError, match="git reference must not begin with a dash"):
        changed_skills(since="--output=/tmp/pwned")


def test_changed_skills_passes_double_dash_delimiter() -> None:
    """Verify changed_skills passes double dash to git diff to delimit revisions."""
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = ""
        changed_skills(since="HEAD~1")
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert args == ["git", "diff", "--name-only", "HEAD~1", "--"]


def test_changed_skills_default_since_is_head_minus_one() -> None:
    """Verify changed_skills defaults to HEAD~1 when since is omitted."""
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = ""
        changed_skills()
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert args == ["git", "diff", "--name-only", "HEAD~1", "--"]


@pytest.mark.parametrize(
    "stderr_msg",
    [
        "fatal: bad revision 'HEAD~1'",
        "fatal: ambiguous argument 'HEAD~1': unknown revision or path not in the working tree.",
        "fatal: shallow clone has insufficient history",
    ],
)
def test_changed_skills_shallow_or_bad_revision_includes_hint(stderr_msg: str) -> None:
    """Verify changed_skills raises ValueError with CI shallow clone guidance on revision errors."""
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 128
        mock_run.return_value.stderr = stderr_msg

        with pytest.raises(ValueError, match="fetch-depth: 0"):
            changed_skills(since="HEAD~1")


def test_changed_skills_missing_git_returns_empty_tuple() -> None:
    """Verify changed_skills returns () safely when git executable raises OSError."""
    with patch("subprocess.run", side_effect=FileNotFoundError("git not found")):
        assert changed_skills(since="HEAD~1") == ()


def test_changed_skills_timeout_raises_value_error() -> None:
    """Verify changed_skills raises ValueError with timeout hint when git diff hangs."""
    with (
        patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd=["git", "diff"], timeout=30.0),
        ),
        pytest.raises(ValueError, match=r"git diff timed out after 30\.0s"),
    ):
        changed_skills(since="HEAD~1")


def test_run_check_non_existent_path_raises_value_error() -> None:
    """Verify run_check raises ValueError on missing skill paths."""
    with pytest.raises(ValueError, match="skill path does not exist"):
        run_check(skills_paths=[Path("/non/existent/path")])


def test_run_check_budget_less_than_one_raises_value_error(
    write_skill: Callable[..., Path],
) -> None:
    """Verify run_check rejects budget < 1."""
    skill_dir = write_skill(
        name="valid-skill",
        description="Valid description with sufficient length.",
    )
    with pytest.raises((ValueError, ValidationError)):
        run_check(skills_paths=[skill_dir], budget=0)


def test_run_check_changed_no_modified_skills_passes_instantly(
    write_skill: Callable[..., Path],
) -> None:
    """Verify run_check passes with 0 probes when no skills changed."""
    skill_dir = write_skill(
        name="valid-skill",
        description="Valid description with sufficient length.",
    )
    with patch("reach.check.changed_skills", return_value=()):
        outcome = run_check(
            skills_paths=[skill_dir],
            changed=True,
            since="origin/main",
        )
        assert outcome.passed
        assert outcome.skills_checked == 0
        assert outcome.probes_executed == 0
        assert outcome.exit_code == 0


def test_run_check_changed_scope_drops_skills_deleted_from_disk(
    write_skill: Callable[..., Path],
    write_queries: Callable[..., Path],
) -> None:
    """Verify a skill removed by the diff is excluded from the empirical scope."""
    skill_dir = write_skill(
        name="valid-skill",
        description="Valid description with sufficient length.",
    )
    queries_file = write_queries(
        queries=[
            Query(id="q-live", text="Sample query for valid-skill", expected_skill="valid-skill"),
            Query(
                id="q-gone",
                text="Sample query for deleted tool",
                expected_skill="deleted-skill",
            ),
        ],
    )

    with patch("reach.check.changed_skills", return_value=("deleted-skill", "valid-skill")):
        outcome = run_check(
            skills_paths=[skill_dir],
            queries_path=queries_file,
            changed=True,
            agent="keyword",
            strict=False,
        )

    assert outcome.skills_checked == 1
    assert outcome.queries_probed == 1


def test_run_check_changed_scope_with_only_deletions_passes_instantly(
    write_skill: Callable[..., Path],
    write_queries: Callable[..., Path],
) -> None:
    """Verify a deletion-only diff passes rather than gating on unreachable queries."""
    skill_dir = write_skill(
        name="valid-skill",
        description="Valid description with sufficient length.",
    )
    queries_file = write_queries(target="deleted-skill", count=2)

    with patch("reach.check.changed_skills", return_value=("deleted-skill",)):
        outcome = run_check(
            skills_paths=[skill_dir],
            queries_path=queries_file,
            changed=True,
            agent="keyword",
            strict=False,
        )

    assert outcome.passed
    assert outcome.skills_checked == 0
    assert outcome.probes_executed == 0
    assert outcome.exit_code == 0


def test_build_check_assertions_trajectory_metrics_pass() -> None:
    """Verify _build_check_assertions generates passing assertions for all 5 trajectory metrics."""
    metrics = EmpiricalMetrics(
        recall=0.90,
        accuracy=0.90,
        misroute_rate=0.05,
        entrypoint_accuracy=0.88,
        trajectory_reachability=0.92,
        step_efficiency=0.85,
        skill_f1=0.90,
        redundancy=0.20,
    )
    settings = CheckSettings(
        min_recall=0.80,
        min_accuracy=0.80,
        max_misroute=0.10,
        min_entrypoint=0.80,
        min_reachability=0.85,
        min_efficiency=0.80,
        min_f1=0.80,
        max_redundancy=0.50,
    )
    assertions = _build_check_assertions(metrics, settings)
    assert len(assertions) == 8  # 3 standard + 5 trajectory
    by_name = {a.name: a for a in assertions}
    assert by_name["entrypoint"].passed is True
    assert by_name["reachability"].passed is True
    assert by_name["step_efficiency"].passed is True
    assert by_name["skill_f1"].passed is True
    assert by_name["redundancy"].passed is True


@pytest.mark.parametrize(
    ("metric_kwarg", "observed_kwarg", "metric_name", "expected_msg_fragment"),
    [
        (
            {"min_entrypoint": 0.85},
            {"entrypoint_accuracy": 0.70},
            "entrypoint",
            "Entrypoint accuracy regression",
        ),
        (
            {"min_reachability": 0.90},
            {"trajectory_reachability": 0.75},
            "reachability",
            "Reachability regression",
        ),
        (
            {"min_efficiency": 0.80},
            {"step_efficiency": 0.60},
            "step_efficiency",
            "Step efficiency regression",
        ),
        ({"min_f1": 0.80}, {"skill_f1": 0.50}, "skill_f1", "Skill F1 regression"),
        ({"max_redundancy": 0.50}, {"redundancy": 1.20}, "redundancy", "Redundancy regression"),
    ],
)
def test_build_check_assertions_trajectory_metrics_fail(
    metric_kwarg: dict[str, float],
    observed_kwarg: dict[str, float],
    metric_name: str,
    expected_msg_fragment: str,
) -> None:
    """Verify _build_check_assertions detects regressions on each trajectory metric."""
    metrics = EmpiricalMetrics(
        recall=0.90,
        accuracy=0.90,
        misroute_rate=0.05,
        **observed_kwarg,
    )
    settings = CheckSettings.model_validate(
        {
            "min_recall": 0.80,
            "min_accuracy": 0.80,
            "max_misroute": 0.10,
            **metric_kwarg,
        },
    )
    assertions = _build_check_assertions(metrics, settings)
    matching = [a for a in assertions if a.name == metric_name]
    assert len(matching) == 1
    assert matching[0].passed is False
    assert expected_msg_fragment in matching[0].message


def test_run_check_with_trajectory_thresholds_pass_and_fail(
    write_skill: Callable[..., Path],
    write_queries: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """Verify run_check evaluates trajectory thresholds end-to-end."""
    skill_dir = write_skill(
        name="valid-skill",
        description="Valid description with sufficient length for deploy skill.",
    )
    query_file = write_queries(target="valid-skill", count=3)

    # Happy path: realistic thresholds met
    pass_outcome = run_check(
        skills_paths=[skill_dir],
        queries_path=query_file,
        agent="keyword",
        min_entrypoint=0.0,
        min_reachability=0.0,
        min_efficiency=0.0,
        min_f1=0.0,
        max_redundancy=10.0,
    )
    assert pass_outcome.passed is True
    assert pass_outcome.exit_code == 0

    # Sad path: unattainable entrypoint threshold fails empirical stage
    query_file_fail = write_queries(target="other-skill", count=3, root=tmp_path / "fail")
    fail_outcome = run_check(
        skills_paths=[skill_dir],
        queries_path=query_file_fail,
        agent="keyword",
        min_entrypoint=0.8,
    )
    assert fail_outcome.passed is False
    assert fail_outcome.stage_failed == CheckStage.EMPIRICAL
    assert fail_outcome.exit_code == 2


def test_run_config_resolve_check_settings() -> None:
    """Verify RunConfig.resolve overrides baseline config cleanly with Pydantic."""
    base = CheckSettings(min_recall=0.75, min_accuracy=0.80, budget=10)
    resolved = RunConfig.resolve(CheckSettings, explicit_settings=base, min_recall=0.90, budget=25)
    assert resolved.min_recall == 0.90
    assert resolved.min_accuracy == 0.80
    assert resolved.budget == 25


def test_load_catalog_skills_deduplicates_overlapping_paths(
    write_skill: Callable[..., Path],
) -> None:
    """Verify _load_catalog_skills deduplicates duplicate skills across candidate paths."""
    from reach.check import _load_catalog_skills
    from reach.models import Catalog, CatalogMode

    skill_path = write_skill(
        name="duplicate-tool",
        description="A distinct description for testing duplicate skill loading.",
    )
    # Provide both the skills parent directory and the skill path itself (overlapping)
    skills_root = skill_path.parent
    loaded = _load_catalog_skills([skills_root, skill_path])

    names = [s.name for s in loaded]
    assert names == ["duplicate-tool"]
    catalog = Catalog(
        id="check-catalog",
        skills=tuple(names),
        mode=CatalogMode.ALL,
    )
    assert catalog.skills == ("duplicate-tool",)


@pytest.mark.parametrize(
    "target_is_file", [False, True], ids=["directory-path", "skill-md-file-path"]
)
def test_check_empirical_probes_cache_and_invalidation(
    tmp_path: Path,
    write_skill: Callable[..., Path],
    target_is_file: bool,
) -> None:
    """Verify in-memory Pydantic cache reuses probes, works on SKILL.md, and skips disk writes."""
    import inspect
    from unittest.mock import patch

    from pydantic import BaseModel

    from reach.check import _CheckCacheKey, _execute_empirical_probes
    from reach.models import QueryKind
    from reach.runtime.keyword import KeywordRuntime

    assert issubclass(_CheckCacheKey, BaseModel)
    assert "yes" not in inspect.signature(run_check).parameters

    skill_dir = write_skill(
        name="cache-tool",
        description="Execute cache-tool diagnostics and workflows.",
    )
    target_path = skill_dir if target_is_file else skill_dir.parent
    queries = [
        Query(
            id="q-cache-1",
            text="please run cache-tool now",
            expected_skill="cache-tool",
            kind=QueryKind.IMPLICIT,
        )
    ]

    select_calls = 0

    class CountingKeywordRuntime(KeywordRuntime):
        def select(
            self, query_text: str, workdir: Path, target_skill: str | None = None
        ) -> SelectionOutcome:
            nonlocal select_calls
            select_calls += 1
            return super().select(query_text, workdir, target_skill=target_skill)

    with patch(
        "reach.check._setup_runtime", side_effect=lambda *_a, **_k: CountingKeywordRuntime()
    ):
        # 1st check run executes probe
        _, m1, count1 = _execute_empirical_probes(queries, [target_path], "keyword", None, None)
        assert count1 == 1
        assert m1.accuracy == 1.0
        assert select_calls == 1
        assert not (skill_dir.parent / ".reach" / "cache").exists()

        # 2nd check run with unchanged SKILL.md reuses cache (0 extra select calls)
        _, m2, count2 = _execute_empirical_probes(queries, [target_path], "keyword", None, None)
        assert count2 == 1
        assert m2.accuracy == 1.0
        assert select_calls == 1

        # Editing SKILL.md changes corpus_digest -> invalidates cache and re-probes
        write_skill(
            name="cache-tool",
            description="Updated description for cache-tool diagnostics.",
        )
        _, m3, count3 = _execute_empirical_probes(queries, [target_path], "keyword", None, None)
        assert count3 == 1
        assert m3.accuracy == 1.0
        assert select_calls == 2


def test_filter_check_queries_includes_competing_neighbor_guardrails(
    write_skill: Callable[..., Path],
    write_queries: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """Verify _filter_check_queries includes competing neighbor queries when --changed is active."""
    from reach.catalog import load_skills
    from reach.check import _filter_check_queries

    write_skill(
        name="gke-storage-troubleshooting",
        description=(
            "Diagnoses GKE storage issues including PVC Pending and Cloud Storage FUSE OOM. "
            "Don't use for initial storage provisioning (use `gke-storage`)."
        ),
    )
    write_skill(
        name="google-cloud-storage-fuse",
        description="Configures, mounts, and tunes Cloud Storage FUSE on GKE clusters.",
    )
    write_skill(
        name="unrelated-billing-export",
        description="Exports billing invoices to CSV reports for finance auditing.",
    )
    skills = load_skills(tmp_path)

    queries_file = write_queries(
        queries=[
            Query(
                id="q-gke-1",
                text="debug PVC pending state in GKE",
                expected_skill="gke-storage-troubleshooting",
            ),
            Query(
                id="oos-storage-fuse-1",
                text="tune Cloud Storage FUSE read cache for GKE training",
                expected_skill="google-cloud-storage-fuse",
            ),
            Query(
                id="q-billing-1",
                text="export monthly finance invoice to CSV",
                expected_skill="unrelated-billing-export",
            ),
        ],
    )

    selected, exhausted = _filter_check_queries(
        queries_file,
        modified={"gke-storage-troubleshooting"},
        changed=True,
        budget=10,
        skills=skills,
    )
    assert not exhausted
    selected_ids = [q.id for q in selected]
    assert "q-gke-1" in selected_ids
    assert "oos-storage-fuse-1" in selected_ids
    assert "q-billing-1" not in selected_ids


def test_filter_check_queries_by_skill_and_id(tmp_path: Path) -> None:
    """Verify _filter_check_queries filters by skill name and query id."""
    from reach.check import _filter_check_queries
    from reach.models import Query, QueryKind
    from reach.queries import Origin, QuerySet, QuerySetProvenance, save_query_set

    queries_file = tmp_path / "test_queries.json"
    save_query_set(
        QuerySet(
            catalog_id="catalog",
            queries=(
                Query(id="q-1", text="text 1", expected_skill="skill-a", kind=QueryKind.IMPLICIT),
                Query(id="q-2", text="text 2", expected_skill="skill-b", kind=QueryKind.IMPLICIT),
                Query(id="q-3", text="text 3", expected_skill="skill-a", kind=QueryKind.IMPLICIT),
            ),
            provenance=QuerySetProvenance(origin=Origin.AUTHORED),
        ),
        queries_file,
    )

    # Filter by skill
    filtered_skill, _ = _filter_check_queries(
        queries_file,
        modified=set(),
        changed=False,
        budget=10,
        filter_skill="skill-a",
    )
    assert [q.id for q in filtered_skill] == ["q-1", "q-3"]

    # Filter by query id
    filtered_id, _ = _filter_check_queries(
        queries_file,
        modified=set(),
        changed=False,
        budget=10,
        filter_id=("q-2",),
    )
    assert [q.id for q in filtered_id] == ["q-2"]

    # Filter by skill glob pattern
    filtered_skill_glob, _ = _filter_check_queries(
        queries_file,
        modified=set(),
        changed=False,
        budget=10,
        filter_skill="skill-*",
    )
    assert [q.id for q in filtered_skill_glob] == ["q-1", "q-2", "q-3"]

    # Filter by query id glob pattern
    filtered_id_glob, _ = _filter_check_queries(
        queries_file,
        modified=set(),
        changed=False,
        budget=10,
        filter_id="q-[13]",
    )
    assert [q.id for q in filtered_id_glob] == ["q-1", "q-3"]


def test_find_competing_neighbors_includes_dense_semantic_rivals(tmp_path: Path) -> None:
    """Verify find_competing_neighbors retains neighbors with semantic similarity >= 0.75."""
    from reach.lint import find_competing_neighbors
    from reach.models import Skill

    skills = [
        Skill(
            name="bigquery-observability",
            description="Monitor active slot utilization and job timeline metrics.",
            path=tmp_path / "bigquery-observability" / "SKILL.md",
        ),
        Skill(
            name="bigquery-slot-cost-optimizer",
            description="Analyze warehouse reservation sizing and query billing tiers.",
            path=tmp_path / "bigquery-slot-cost-optimizer" / "SKILL.md",
        ),
        Skill(
            name="cloud-dns-routing",
            description="Configure DNS forwarding zones and health check policies.",
            path=tmp_path / "cloud-dns-routing" / "SKILL.md",
        ),
    ]
    dense_sims = {("bigquery-observability", "bigquery-slot-cost-optimizer"): 0.81}
    neighbors = find_competing_neighbors(
        {"bigquery-observability"},
        skills,
        dense_similarities=dense_sims,
    )
    assert neighbors == {"bigquery-slot-cost-optimizer"}
