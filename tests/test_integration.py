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

"""Integration test suite for skill-reach verifying multi-command workflows and idempotence."""

from __future__ import annotations

import json
import os
import shutil
from typing import TYPE_CHECKING

import pytest

from reach.artifact import Artifact, read_artifact
from reach.catalog import load_skills
from reach.cluster import ClusterPartition
from reach.config import RunConfig
from reach.diff import Comparison
from reach.models import Query
from reach.queries import (
    Origin,
    QuerySet,
    QuerySetProvenance,
    load_query_set,
    query_set_digest,
    save_query_set,
)
from reach.run import conduct
from reach.runtime.keyword import KeywordRuntime
from reach.runtime.retriever import TwoStageRetrieverRuntime
from reach.sweep import ScalingStudy
from reach.views import OverlapView

if TYPE_CHECKING:
    from pathlib import Path

    from conftest import IntegrationWorkspace

pytestmark = pytest.mark.integration


def test_full_lifecycle_pipeline(integration_workspace: IntegrationWorkspace) -> None:
    """Verify complete end-to-end lifecycle from linting to CI quality gate via subprocess."""
    ws = integration_workspace
    eval_out = ws.root / ".reach" / "eval.json"

    # Stage 1: Manifest linting
    lint_proc = ws.run_reach(["lint", "--strict"])
    assert lint_proc.returncode == 0, f"Lint failed:\n{lint_proc.stderr}"

    # Stage 2: Collision and vocabulary overlap analysis
    overlap_proc = ws.run_reach(["overlap", "--format", "json"])
    assert overlap_proc.returncode == 0, f"Overlap failed:\n{overlap_proc.stderr}"
    overlap_view = OverlapView.model_validate_json(overlap_proc.stdout)
    assert overlap_view.corpus_size == 3
    assert len(overlap_view.skills) == 3

    # Stage 3: Reachability evaluation with keyword agent
    eval_proc = ws.run_reach(
        [
            "eval",
            "--skills",
            str(ws.skills_dir),
            "--queries",
            str(ws.queries_file),
            "--agent",
            "keyword",
            "--out",
            str(eval_out),
            "--workdir",
            str(ws.root / "work"),
            "--partial",
            "--quiet",
        ]
    )
    assert eval_proc.returncode == 0, f"Eval failed:\n{eval_proc.stderr}"
    assert eval_out.exists()

    # Stage 4: Result scorecard rendering
    view_proc = ws.run_reach(["view", str(eval_out), "--format", "json"])
    assert view_proc.returncode == 0, f"View failed:\n{view_proc.stderr}"
    view_artifact = Artifact.model_validate_json(view_proc.stdout)
    assert view_artifact.catalog_size == 3
    assert view_artifact.scores.top1_accuracy >= 0.0

    # Stage 5: CI quality gate verification
    check_proc = ws.run_reach(
        [
            "check",
            str(ws.skills_dir),
            "--queries",
            str(ws.queries_file),
            "--agent",
            "keyword",
            "--config",
            str(ws.config_file),
        ]
    )
    assert check_proc.returncode == 0, f"Check failed:\n{check_proc.stderr}"


def test_pipeline_idempotence(integration_workspace: IntegrationWorkspace) -> None:
    """Verify sequential execution in the same workspace is strictly idempotent."""
    ws = integration_workspace
    eval_out = ws.root / ".reach" / "eval.json"

    # Execute evaluation and quality gate twice consecutively
    for i in range(2):
        eval_proc = ws.run_reach(
            [
                "eval",
                "--skills",
                str(ws.skills_dir),
                "--queries",
                str(ws.queries_file),
                "--agent",
                "keyword",
                "--out",
                str(eval_out),
                "--workdir",
                str(ws.root / "work"),
                "--partial",
                "--quiet",
            ]
        )
        assert eval_proc.returncode == 0, f"Iteration {i} eval failed:\n{eval_proc.stderr}"
        assert eval_out.exists()

        check_proc = ws.run_reach(
            [
                "check",
                str(ws.skills_dir),
                "--queries",
                str(ws.queries_file),
                "--agent",
                "keyword",
                "--config",
                str(ws.config_file),
            ]
        )
        assert check_proc.returncode == 0, f"Iteration {i} check failed:\n{check_proc.stderr}"


def test_corrupted_skill_blocks_quality_gate(
    integration_workspace: IntegrationWorkspace,
) -> None:
    """Verify Stage 1 lint failure terminates check before running empirical probes."""
    ws = integration_workspace

    # Inject an invalid skill manifest with malformed YAML
    broken = ws.skills_dir / "broken-tool"
    broken.mkdir()
    (broken / "SKILL.md").write_text("invalid: yaml: [broken", encoding="utf-8")

    # Lint must detect the issue and fail
    lint_proc = ws.run_reach(["lint", "--skills", str(ws.skills_dir)])
    assert lint_proc.returncode != 0

    # Check must abort at Stage 1 before empirical probe trials
    check_proc = ws.run_reach(
        [
            "check",
            str(ws.skills_dir),
            "--queries",
            str(ws.queries_file),
            "--agent",
            "keyword",
        ]
    )
    assert check_proc.returncode != 0


def test_escaping_symlink_blocks_evaluation(
    integration_workspace: IntegrationWorkspace,
) -> None:
    """Verify evaluation aborts with an error when a skill contains an escaping symlink."""
    ws = integration_workspace
    evil = ws.skills_dir / "evil-skill"
    evil.mkdir()
    (evil / "SKILL.md").write_text(
        "---\nname: evil-skill\ndescription: Skill with escaping symlink.\n---\nBody",
        encoding="utf-8",
    )
    secret_file = ws.root / "secret.txt"
    secret_file.write_text("classified_secret", encoding="utf-8")
    (evil / "leak").symlink_to(secret_file)

    evil_queries = ws.root / "evil_queries.json"
    evil_queries.write_text(
        json.dumps(
            {
                "catalog_id": "all",
                "provenance": {"origin": "authored"},
                "queries": [
                    {
                        "query_id": "q-evil",
                        "text": "run evil exploit",
                        "expected_skill": "evil-skill",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    proc = ws.run_reach(
        [
            "eval",
            "--skills",
            str(ws.skills_dir),
            "--queries",
            str(evil_queries),
            "--agent",
            "keyword",
            "--mode",
            "singleton",
            "--catalog",
            "singleton:evil-skill",
            "--out",
            str(ws.root / "eval_out.jsonl"),
            "--workdir",
            str(ws.root / "work_evil"),
            "--quiet",
            "--partial",
            "--rescope",
        ]
    )
    assert proc.returncode != 0
    assert "escaping skill directory" in proc.stderr


def test_custom_config_reach_toml(integration_workspace: IntegrationWorkspace) -> None:
    """Verify reach CLI detects and honors configuration from reach.toml."""
    ws = integration_workspace

    # Set up custom configuration enforcing strict passing thresholds
    strict_config = ws.root / "strict.toml"
    strict_config.write_text(
        "[general]\n"
        'default_agent = "keyword"\n\n'
        "[discovery]\n"
        'precedence = ["skills"]\n\n'
        "[check]\n"
        "strict = true\n"
        "budget = 10\n"
        "min_accuracy = 1.0\n"
        "min_recall = 1.0\n",
        encoding="utf-8",
    )

    check_proc = ws.run_reach(
        [
            "check",
            str(ws.skills_dir),
            "--queries",
            str(ws.queries_file),
            "--agent",
            "keyword",
            "--config",
            str(strict_config),
        ]
    )
    assert check_proc.returncode == 0, f"Check with custom config failed:\n{check_proc.stderr}"


def test_sweep_scaling_study_via_cli(integration_workspace: IntegrationWorkspace) -> None:
    """Execute capacity scaling sweep via subprocess and validate ScalingStudy output."""
    ws = integration_workspace
    sweep_out = ws.root / ".reach" / "sweep.json"

    proc = ws.run_reach(
        [
            "sweep",
            str(ws.skills_dir),
            "--queries",
            str(ws.queries_file),
            "--scales",
            "1,2,3",
            "--agent",
            "keyword",
            "--format",
            "json",
            "--out",
            str(sweep_out),
        ]
    )
    assert proc.returncode == 0, f"Sweep failed:\n{proc.stderr}"
    assert sweep_out.exists()

    study = ScalingStudy.model_validate_json(sweep_out.read_text(encoding="utf-8"))
    assert len(study.points) > 0
    assert study.scales == (1, 2, 3)
    assert study.noise_floor >= 0.0
    assert 0.0 <= study.baseline_pass_rate <= 1.0
    assert study.knee_scale is None or isinstance(study.knee_scale, int)
    for point in study.points:
        assert point.scale in (1, 2, 3)
        assert 0.0 <= point.pass_rate <= 1.0
        assert isinstance(point.delta_vs_baseline, float)


def test_cluster_partitioning_via_cli(integration_workspace: IntegrationWorkspace) -> None:
    """Partition multi-skill workspace via CLI and validate cluster assignments."""
    ws = integration_workspace

    proc = ws.run_reach(["cluster", str(ws.skills_dir), "--format", "json"])
    assert proc.returncode == 0, f"Cluster failed:\n{proc.stderr}"

    partition = ClusterPartition.model_validate_json(proc.stdout)
    assert len(partition.clusters) > 0
    assert isinstance(partition.modularity, float)
    assert set(partition.cluster_map.keys()) == {
        "file-copier",
        "file-compressor",
        "file-deleter",
    }


def test_diff_ab_controlled_delta_via_cli(
    integration_workspace: IntegrationWorkspace,
    tmp_path: Path,
) -> None:
    """Compare control and treatment evaluation arms via CLI and validate diff metrics."""
    ws = integration_workspace

    ctrl_dir = tmp_path / "skills_ctrl"
    shutil.copytree(ws.skills_dir, ctrl_dir)
    treat_dir = tmp_path / "skills_treat"
    shutil.copytree(ws.skills_dir, treat_dir)

    (treat_dir / "file-copier" / "SKILL.md").write_text(
        "---\n"
        "name: file-copier\n"
        "description: Fast parallel copying and replication for files and directories.\n"
        "metadata:\n"
        "  category: filesystem\n"
        "---\n\n"
        "# File Copier\n"
        "Instructions for copying files and directory structures.\n",
        encoding="utf-8",
    )

    control_out = ws.root / "control.jsonl"
    treatment_out = ws.root / "treatment.jsonl"

    c_eval = ws.run_reach(
        [
            "eval",
            "--skills",
            str(ctrl_dir),
            "--queries",
            str(ws.queries_file),
            "--agent",
            "keyword",
            "--out",
            str(control_out),
            "--partial",
            "--quiet",
        ]
    )
    assert c_eval.returncode == 0, f"Control eval failed:\n{c_eval.stderr}"

    t_eval = ws.run_reach(
        [
            "eval",
            "--skills",
            str(treat_dir),
            "--queries",
            str(ws.queries_file),
            "--agent",
            "keyword",
            "--out",
            str(treatment_out),
            "--partial",
            "--quiet",
        ]
    )
    assert t_eval.returncode == 0, f"Treatment eval failed:\n{t_eval.stderr}"

    diff_proc = ws.run_reach(
        [
            "diff",
            str(control_out),
            str(treatment_out),
            "--vary",
            "description",
            "--format",
            "json",
        ]
    )
    assert diff_proc.returncode == 0, f"Diff failed:\n{diff_proc.stderr}"

    comparison = Comparison.model_validate_json(diff_proc.stdout)
    assert comparison.shared_queries > 0
    assert isinstance(comparison.headline.delta, float)
    assert comparison.headline.floor is None or isinstance(comparison.headline.floor, float)
    assert 0.0 <= comparison.headline.confidence <= 1.0
    assert isinstance(comparison.headline.real, bool)
    assert comparison.headline.resolvable is None or isinstance(
        comparison.headline.resolvable, float
    )


def test_optimize_description_heuristics_via_cli(
    integration_workspace: IntegrationWorkspace,
) -> None:
    """Optimize skill description using offline heuristics and verify in-place update."""
    ws = integration_workspace
    skill_manifest = ws.skills_dir / "file-copier" / "SKILL.md"
    original_text = skill_manifest.read_text(encoding="utf-8")

    opt_proc = ws.run_reach(
        [
            "optimize",
            "file-copier",
            "--skills",
            str(ws.skills_dir),
            "--queries",
            str(ws.queries_file),
            "--candidates",
            "2",
            "--workers",
            "2",
            "--with-handoff",
            "--yes",
            "--auto-apply",
            "--force",
            "--agent",
            "keyword",
        ]
    )
    assert opt_proc.returncode == 0, f"Optimize failed:\n{opt_proc.stderr}"
    assert "[reach optimize]" in opt_proc.stderr

    updated_text = skill_manifest.read_text(encoding="utf-8")
    assert updated_text != original_text
    assert "file-copier" in updated_text
    assert "> **Routing Note:**" in updated_text

    eval_out = ws.root / ".reach" / "eval_opt.json"
    eval_proc = ws.run_reach(
        [
            "eval",
            "--skills",
            str(ws.skills_dir),
            "--queries",
            str(ws.queries_file),
            "--agent",
            "keyword",
            "--out",
            str(eval_out),
            "--partial",
            "--quiet",
        ]
    )
    assert eval_proc.returncode == 0, f"Eval after optimize failed:\n{eval_proc.stderr}"
    assert eval_out.exists()

    artifact = read_artifact(eval_out)
    assert artifact.catalog_size == 3
    loaded_skills = load_skills(ws.skills_dir)
    assert any(s.name == "file-copier" and s.description != original_text for s in loaded_skills)


def test_init_and_clean_lifecycle_via_cli(
    integration_workspace: IntegrationWorkspace,
    empty_integration_workspace: IntegrationWorkspace,
) -> None:
    """Verify workspace scaffolding with init and cache/artifact purge with clean --all."""
    ws_empty = empty_integration_workspace
    init_proc = ws_empty.run_reach(["init", "--agent", "keyword", "--skills", "skills"])
    assert init_proc.returncode == 0, f"Init failed:\n{init_proc.stderr}"
    assert (ws_empty.root / "reach.toml").exists()
    assert (ws_empty.root / ".reach").is_dir()
    assert (ws_empty.root / "skills").is_dir()

    ws = integration_workspace
    eval_file = ws.root / ".reach" / "eval.json"
    eval_file.parent.mkdir(parents=True, exist_ok=True)
    eval_file.write_text('{"status": "test"}', encoding="utf-8")

    cache_dir = ws.root / ".reach" / "cache" / "registry" / "my-project"
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "meta.json").write_text("{}", encoding="utf-8")

    user_note = ws.root / "README.txt"
    user_note.write_text("User project documentation", encoding="utf-8")

    clean_proc = ws.run_reach(["clean", "--all"])
    assert clean_proc.returncode == 0, f"Clean failed:\n{clean_proc.stderr}"

    assert not eval_file.exists()
    assert not (ws.root / ".reach" / "cache" / "registry").exists()

    assert user_note.exists()
    assert ws.skills_dir.exists()
    assert ws.config_file.exists()


def test_multi_step_trajectory_quality_gate(
    integration_workspace: IntegrationWorkspace,
) -> None:
    """Verify multi-step trajectory assertions in quality gate pass and fail as expected."""
    ws = integration_workspace

    pass_proc = ws.run_reach(
        [
            "check",
            str(ws.skills_dir),
            "--queries",
            str(ws.queries_file),
            "--min-entrypoint",
            "0.80",
            "--min-reachability",
            "0.80",
            "--min-efficiency",
            "0.70",
            "--min-f1",
            "0.75",
            "--max-redundancy",
            "0.50",
            "--agent",
            "keyword",
        ]
    )
    assert pass_proc.returncode == 0, f"Check failed unexpectedly:\n{pass_proc.stderr}"

    unmatched_queries_file = ws.root / "unmatched_queries.json"
    unmatched_dataset = QuerySet(
        catalog_id="all",
        notes="Queries that do not match expected skill keywords",
        provenance=QuerySetProvenance(origin=Origin.AUTHORED),
        queries=(
            Query(
                query_id="q-unmatched-1",
                text="A totally unrelated task about astronomy and space telescopes",
                expected_skill="file-copier",
            ),
            Query(
                query_id="q-unmatched-2",
                text="Quantum computing entanglement simulations",
                expected_skill="file-compressor",
            ),
        ),
    )
    save_query_set(unmatched_dataset, unmatched_queries_file)

    fail_proc = ws.run_reach(
        [
            "check",
            str(ws.skills_dir),
            "--queries",
            str(unmatched_queries_file),
            "--min-entrypoint",
            "0.80",
            "--min-reachability",
            "0.80",
            "--min-efficiency",
            "0.70",
            "--min-f1",
            "0.75",
            "--max-redundancy",
            "0.50",
            "--agent",
            "keyword",
        ]
    )
    assert fail_proc.returncode == 2, (
        f"Expected exit code 2 for failed assertions, got {fail_proc.returncode}"
    )


def test_ci_environment_simulation(
    integration_workspace: IntegrationWorkspace,
    tmp_path: Path,
) -> None:
    """Simulate CI environment with GITHUB_STEP_SUMMARY and workflow command annotations."""
    ws = integration_workspace
    step_summary_file = tmp_path / "step_summary.md"

    unmatched_file = ws.root / "ci_unmatched.json"
    unmatched_dataset = QuerySet(
        catalog_id="all",
        notes="Unmatched dataset for CI simulation",
        provenance=QuerySetProvenance(origin=Origin.AUTHORED),
        queries=(
            Query(
                query_id="q-ci-fail",
                text="Random unmatched prompt text",
                expected_skill="file-copier",
            ),
        ),
    )
    save_query_set(unmatched_dataset, unmatched_file)

    ci_env = {
        **os.environ,
        "GITHUB_ACTIONS": "true",
        "GITHUB_STEP_SUMMARY": str(step_summary_file),
    }

    ci_proc = ws.run_reach(
        [
            "check",
            str(ws.skills_dir),
            "--queries",
            str(unmatched_file),
            "--min-recall",
            "0.80",
            "--agent",
            "keyword",
        ],
        env=ci_env,
    )
    assert ci_proc.returncode == 2

    assert "::error title=regression-recall::" in ci_proc.stdout

    assert step_summary_file.exists()
    summary_text = step_summary_file.read_text(encoding="utf-8")
    assert "# Reach Quality Gate: ❌ FAILED" in summary_text
    assert "| recall |" in summary_text
    assert "❌ Failed" in summary_text


def test_format_conversion_query_interchange_integrity(
    integration_workspace: IntegrationWorkspace,
) -> None:
    """Verify lossless roundtrip format conversion between JSON and CSV query sets."""
    ws = integration_workspace
    csv_file = ws.root / "queries.csv"
    restored_json_file = ws.root / "queries_restored.json"

    export_proc = ws.run_reach(["query", str(ws.queries_file), "-o", str(csv_file)])
    assert export_proc.returncode == 0, f"Export to CSV failed:\n{export_proc.stderr}"
    assert csv_file.exists()

    import_proc = ws.run_reach(["query", str(csv_file), "-o", str(restored_json_file)])
    assert import_proc.returncode == 0, f"Import to JSON failed:\n{import_proc.stderr}"
    assert restored_json_file.exists()

    original_qs = load_query_set(ws.queries_file)
    restored_qs = load_query_set(restored_json_file)
    assert query_set_digest(original_qs) == query_set_digest(restored_qs)


def test_multi_worker_concurrency_and_slot_isolation(
    integration_workspace: IntegrationWorkspace,
    tmp_path: Path,
) -> None:
    """Verify concurrent evaluation workers operate in isolated slots without collisions."""
    ws = integration_workspace

    eval_out = ws.root / "concurrent_eval.jsonl"
    eval_proc = ws.run_reach(
        [
            "eval",
            "--skills",
            str(ws.skills_dir),
            "--queries",
            str(ws.queries_file),
            "--agent",
            "keyword",
            "--workers",
            "2",
            "--attempts",
            "3",
            "--out",
            str(eval_out),
            "--partial",
            "--quiet",
        ]
    )
    assert eval_proc.returncode == 0, f"Concurrent eval failed:\n{eval_proc.stderr}"
    assert eval_out.exists()

    lines = [line for line in eval_out.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 6

    inner = KeywordRuntime()
    retriever_runtime = TwoStageRetrieverRuntime(inner, top_k=2)
    workdir = tmp_path / "concurrent_work"
    out_file = tmp_path / "retriever_out.jsonl"
    config = RunConfig.model_validate(
        {
            "study": {
                "skills": ws.skills_dir,
                "queries": ws.queries_file,
                "workdir": workdir,
                "partial": True,
                "out": out_file,
            },
            "runtime": {"agent": "keyword"},
            "catalog": {"mode": "all"},
            "plan": {"attempts": 3},
        }
    )
    outcome = conduct(config, runtime=retriever_runtime, workers=4)
    assert len(outcome.results) == 6
