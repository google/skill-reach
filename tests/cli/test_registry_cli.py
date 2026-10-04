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

"""Verify CLI commands with Google Cloud Agent Registry integration."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

from reach.cli import main
from reach.models import Skill

if TYPE_CHECKING:
    import pytest


def test_doctor_includes_agent_registry(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify reach doctor inspects Google Cloud Agent Registry status."""
    with patch("reach.registry.get_access_token", return_value="mock-adc-token"):
        assert main(["doctor"]) == 0
    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert "Agent Registry" in combined


def test_check_registry_drift_and_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify reach check --project performs static audit and prints drift table."""
    monkeypatch.chdir(tmp_path)

    # Create local skill
    local_skill_dir = tmp_path / "skill-a"
    local_skill_dir.mkdir(parents=True)
    (local_skill_dir / "SKILL.md").write_text(
        "---\nname: skill-a\ndescription: Local modified description.\n---\nBody",
    )

    mock_remote_skills = [
        Skill(
            name="skill-a",
            description="Remote original description.",
            path=tmp_path / "cache" / "skill-a",
            metadata={"state": "ACTIVE"},
        ),
        Skill(
            name="skill-b",
            description="Remote only skill.",
            path=tmp_path / "cache" / "skill-b",
            metadata={"state": "ACTIVE"},
        ),
    ]

    with patch("reach.catalog.load_registry_skills", return_value=mock_remote_skills):
        assert main(["check", "--project", "test-proj", "--location", "global"]) == 0

    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert "Agent Registry Audit: test-proj (global)" in combined
    assert "skill-a" in combined
    assert "Description modified locally" in combined
    assert "skill-b" in combined
    assert "Remote registry skill" in combined


def test_overlap_with_registry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify reach overlap --project ranks lexical competition across registry skills."""
    monkeypatch.chdir(tmp_path)

    mock_remote_skills = [
        Skill(
            name="cloud-monitoring",
            description="Monitor Google Cloud resources, metrics, and dashboards.",
            path=tmp_path / "cache" / "monitoring",
        ),
        Skill(
            name="cloud-logging",
            description="Query and analyze Google Cloud logs and log metrics.",
            path=tmp_path / "cache" / "logging",
        ),
    ]

    with patch("reach.catalog.load_registry_skills", return_value=mock_remote_skills):
        assert main(["overlap", "--project", "test-proj"]) == 0

    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert "cloud-monitoring" in combined
    assert "cloud-logging" in combined


def test_eval_dry_run_with_registry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify reach eval --project resolves skills from Agent Registry in dry-run mode."""
    monkeypatch.chdir(tmp_path)

    dir1 = tmp_path / "cache" / "monitoring"
    dir2 = tmp_path / "cache" / "logging"
    dir1.mkdir(parents=True)
    dir2.mkdir(parents=True)
    (dir1 / "SKILL.md").write_text("---\nname: cloud-monitoring\ndescription: Monitor.\n---\nBody")
    (dir2 / "SKILL.md").write_text("---\nname: cloud-logging\ndescription: Logging.\n---\nBody")

    mock_remote_skills = [
        Skill(
            name="cloud-monitoring",
            description="Monitor Google Cloud resources, metrics, and dashboards.",
            path=dir1,
        ),
        Skill(
            name="cloud-logging",
            description="Query and analyze Google Cloud logs and log metrics.",
            path=dir2,
        ),
    ]

    with patch("reach.catalog.load_registry_skills", return_value=mock_remote_skills):
        assert (
            main(
                [
                    "eval",
                    "--project",
                    "test-proj",
                    "--auto",
                    "--dry-run",
                ],
            )
            == 0
        )

    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert "2 skills resident" in combined


def test_query_draft_with_registry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify reach query draft --project resolves skills from registry."""
    monkeypatch.chdir(tmp_path)

    mock_remote_skills = [
        Skill(
            name="cloud-storage",
            description="Manage Google Cloud Storage buckets and objects.",
            path=tmp_path / "cache" / "storage",
        ),
    ]

    with (
        patch("reach.catalog.load_registry_skills", return_value=mock_remote_skills),
        patch("reach.cli.query._draft_query_set", return_value=0) as mock_draft,
    ):
        assert (
            main(
                [
                    "query",
                    "draft",
                    "--project",
                    "test-proj",
                    "--out",
                    "queries.json",
                ],
            )
            == 0
        )
        assert mock_draft.called


def test_explicit_skills_flag_overrides_toml_registry_project(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    skill_repo: Path,
) -> None:
    """Verify explicit --skills overrides [registry].project unless --registry is passed."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "reach.toml").write_text(
        '[registry]\nproject = "toml-registry-proj"\n',
        encoding="utf-8",
    )

    with patch("reach.catalog.load_registry_skills") as mock_reg:
        # 1. overlap with explicit skills path does not call Agent Registry
        assert main(["overlap", str(skill_repo)]) == 0
        assert not mock_reg.called

        # 2. cluster with explicit skills path does not call Agent Registry
        assert main(["cluster", str(skill_repo)]) == 0
        assert not mock_reg.called

        # 3. eval with explicit --skills does not call Agent Registry
        assert main(["eval", "--skills", str(skill_repo), "--auto", "--dry-run"]) == 0
        assert not mock_reg.called


def test_registry_flag_without_project_fails_fast(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify --registry without a configured or discoverable project fails with a clear error."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.delenv("GCP_PROJECT_ID", raising=False)
    monkeypatch.delenv("GCLOUD_PROJECT", raising=False)

    with patch("reach.config.resolve_registry_project", return_value=None):
        rc = main(["overlap", "--registry"])

    assert rc != 0
    captured = capsys.readouterr()
    assert "Agent Registry requires a Google Cloud project ID" in (captured.out + captured.err)
