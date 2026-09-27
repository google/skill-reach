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

"""Verify reach clean CLI command, dry-run previews, and project scoping."""

from __future__ import annotations

from pathlib import Path

import pytest

from reach.cli import main


def test_clean_empty_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify reach clean exits cleanly when cache is already empty."""
    monkeypatch.chdir(tmp_path)
    assert main(["clean"]) == 0
    captured = capsys.readouterr()
    combined = (captured.out + captured.err).lower()
    assert "already clean" in combined


def test_clean_dry_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify reach clean --dry-run reports space without deleting files."""
    monkeypatch.chdir(tmp_path)
    cache_dir = tmp_path / ".reach" / "cache" / "registry" / "proj-a" / "global"
    cache_dir.mkdir(parents=True, exist_ok=True)
    sample_file = cache_dir / "test.txt"
    sample_file.write_text("dummy cache content")

    assert main(["clean", "--dry-run"]) == 0
    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert "Would remove" in combined
    assert "space to reclaim" in combined
    assert sample_file.exists()


def test_clean_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify reach clean removes cache files and reports reclaimed space."""
    monkeypatch.chdir(tmp_path)
    cache_dir = tmp_path / ".reach" / "cache" / "registry" / "proj-a" / "global"
    cache_dir.mkdir(parents=True, exist_ok=True)
    sample_file = cache_dir / "test.txt"
    sample_file.write_text("dummy cache content")

    assert main(["clean"]) == 0
    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert "Cleaned" in combined
    assert not sample_file.exists()


def test_clean_project_scoping(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify reach clean --project only purges the specified project cache."""
    monkeypatch.chdir(tmp_path)
    proj_a = tmp_path / ".reach" / "cache" / "registry" / "proj-a"
    proj_b = tmp_path / ".reach" / "cache" / "registry" / "proj-b"
    proj_a.mkdir(parents=True, exist_ok=True)
    proj_b.mkdir(parents=True, exist_ok=True)
    file_a = proj_a / "a.txt"
    file_b = proj_b / "b.txt"
    file_a.write_text("a")
    file_b.write_text("b")

    assert main(["clean", "--project", "proj-a"]) == 0
    assert not file_a.exists()
    assert file_b.exists()


def test_clean_all_flag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify reach clean --all deletes evaluation artifacts and query sets in addition to cache."""
    monkeypatch.chdir(tmp_path)
    reach_dir = tmp_path / ".reach"
    reach_dir.mkdir(parents=True, exist_ok=True)
    eval_file = reach_dir / "eval.json"
    queries_file = reach_dir / "queries.json"
    report_file = reach_dir / "report.html"
    eval_file.write_text("{}")
    queries_file.write_text("{}")
    report_file.write_text("<html></html>")

    assert main(["clean", "--all"]) == 0
    assert not eval_file.exists()
    assert not queries_file.exists()
    assert not report_file.exists()


def test_clean_all_removes_sweep_results_and_artifact_sidecars(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify reach clean --all purges sweep output and artifact sidecars, sparing other files."""
    monkeypatch.chdir(tmp_path)
    reach_dir = tmp_path / ".reach"
    reach_dir.mkdir(parents=True, exist_ok=True)
    purged = [
        reach_dir / name
        for name in (
            "eval.json",
            "eval.json.artifact.json",
            "sweep.json",
            "sweep.json.artifact.json",
            "custom-run.json.artifact.json",
            "custom-eval.config.json",
        )
    ]
    for path in purged:
        path.write_text("{}")
    preserved = reach_dir / "notes.md"
    preserved.write_text("hand-written notes")

    assert main(["clean", "--all"]) == 0

    assert [p.name for p in purged if p.exists()] == []
    assert preserved.exists()


def test_clean_rejects_malicious_project_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify reach clean --project exits with 1 on path traversal."""
    monkeypatch.chdir(tmp_path)
    assert main(["clean", "--project", "../../escape"]) == 1
    captured = capsys.readouterr()
    output = captured.out + captured.err
    assert "Error:" in output
    assert "escapes cache directory" in output


@pytest.mark.parametrize(
    ("num_bytes", "expected"),
    [
        (512, "512 B"),
        (2048, "2.0 KB"),
        (5 * 1024 * 1024, "5.0 MB"),
        (3 * 1024 * 1024 * 1024, "3.0 GB"),
    ],
)
def test_format_size_tiers(num_bytes: int, expected: str) -> None:
    """Verify _format_size formats byte counts across B, KB, MB, and GB tiers."""
    from reach.cli.clean import _format_size

    assert _format_size(num_bytes) == expected


def test_clean_quiet_empty_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify reach clean --quiet exits 0 silently when cache is already empty."""
    monkeypatch.chdir(tmp_path)
    assert main(["clean", "--quiet"]) == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
