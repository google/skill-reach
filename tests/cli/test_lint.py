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

"""Verify `reach lint` CLI arguments, exit codes, formatting, and strict mode."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from reach.cli import main

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    import pytest


def test_lint_help(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify that reach lint --help outputs command documentation and exits 0."""
    assert main(["lint", "--help"]) == 0
    captured = capsys.readouterr()
    assert "Validate skill manifests" in captured.out or "Validate skill manifests" in captured.err


def test_lint_explain_known_rule(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify that --explain outputs rule details and exits 0."""
    assert main(["lint", "--explain", "description-too-short"]) == 0
    captured = capsys.readouterr()
    assert "description-too-short" in captured.out or "description-too-short" in captured.err
    assert "Remedy:" in captured.out or "Remedy:" in captured.err


def test_lint_explain_unknown_rule(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify that --explain on unknown rule outputs error and exits 2."""
    assert main(["lint", "--explain", "non-existent-rule"]) == 2
    captured = capsys.readouterr()
    assert "unknown lint rule" in captured.out or "unknown lint rule" in captured.err


def test_lint_clean_corpus(
    write_skill: Callable[..., Path],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify that a clean skill catalog exits 0 and prints success message."""
    write_skill(
        name="good-skill",
        description="A completely valid skill description that provides sufficient context.",
    )
    assert main(["lint", "--skills", str(tmp_path)]) == 0
    captured = capsys.readouterr()
    assert "0 errors, 0 warnings" in captured.out or "0 errors, 0 warnings" in captured.err


def test_lint_errors_exit_one(
    write_skill: Callable[..., Path],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify that errors cause reach lint to exit with code 1."""
    write_skill(
        name="BadName_Invalid",
        description="A description for an invalid skill name.",
    )
    assert main(["lint", "--skills", str(tmp_path)]) == 1
    captured = capsys.readouterr()
    assert "invalid-name-format" in captured.out or "invalid-name-format" in captured.err


def test_lint_warnings_exit_zero_without_strict(
    write_skill: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """Verify that warnings alone do not cause failure when --strict is omitted."""
    write_skill(
        name="short-desc",
        description="Too brief.",
    )
    assert main(["lint", "--skills", str(tmp_path)]) == 0


def test_lint_warnings_exit_one_with_strict(
    write_skill: Callable[..., Path],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify that --strict elevates warnings to exit code 1."""
    write_skill(
        name="short-desc",
        description="Too brief.",
    )
    assert main(["lint", "--skills", str(tmp_path), "--strict"]) == 1
    captured = capsys.readouterr()
    assert "description-too-short" in captured.out or "description-too-short" in captured.err


def test_lint_format_json(
    write_skill: Callable[..., Path],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify that --format json outputs machine-readable JSON structure."""
    write_skill(
        name="short-desc",
        description="Too brief.",
    )
    main(["lint", "--skills", str(tmp_path), "--format", "json"])
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert "issues" in data
    assert "skills_checked" in data
    assert any(i["rule"] == "description-too-short" for i in data["issues"])


def test_lint_format_concise(
    write_skill: Callable[..., Path],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify that --format concise outputs single line per diagnostic."""
    write_skill(
        name="short-desc",
        description="Too brief.",
    )
    main(["lint", "--skills", str(tmp_path), "--format", "concise"])
    captured = capsys.readouterr()
    assert "[description-too-short]" in captured.out


def test_lint_format_csv(
    write_skill: Callable[..., Path],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify that --format csv outputs CSV header and diagnostic rows."""
    write_skill(
        name="short-desc",
        description="Too brief.",
    )
    main(["lint", "--skills", str(tmp_path), "--format", "csv"])
    captured = capsys.readouterr()
    assert "severity,rule,skill,path,message,remedy" in captured.out
    assert "description-too-short" in captured.out


def test_lint_ignore_flag(write_skill: Callable[..., Path], tmp_path: Path) -> None:
    """Verify that --ignore silences specified rule."""
    write_skill(
        name="short-desc",
        description="Too brief.",
    )
    assert (
        main(
            [
                "lint",
                "--skills",
                str(tmp_path),
                "--strict",
                "--ignore",
                "description-too-short",
            ],
        )
        == 0
    )


def test_lint_skill_filter(
    write_skill: Callable[..., Path],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify that --skill filters output to the named skill."""
    write_skill(
        name="skill-one",
        description="Too brief.",
    )
    write_skill(
        name="skill-two",
        description="A completely valid skill description that provides sufficient context.",
    )
    # Linting only skill-two should be completely clean
    assert main(["lint", "--skills", str(tmp_path), "--skill", "skill-two", "--strict"]) == 0


def test_lint_custom_config_thresholds(
    write_skill: Callable[..., Path],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify that reach lint --config respects custom thresholds in TOML."""
    skill_dir = write_skill(
        name="moderate-length-name",
        description="A sufficiently detailed description that satisfies standard rules.",
        root=tmp_path / "skill-under-test",
    )
    # Default allows moderate-length-name (20 chars)
    assert main(["lint", "--skills", str(skill_dir)]) == 0

    # Custom config with max_name_length = 15 should reject moderate-length-name (20 chars)
    custom_toml = tmp_path / "custom.toml"
    custom_toml.write_text(
        """
[lint]
max_name_length = 15
""",
        encoding="utf-8",
    )
    assert (
        main(
            [
                "lint",
                "--skills",
                str(skill_dir),
                "--config",
                str(custom_toml),
                "--format",
                "json",
            ],
        )
        == 1
    )
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    messages = [issue["message"] for issue in data["issues"]]
    assert any("max 15 characters" in m for m in messages)


def test_lint_format_github(
    write_skill: Callable[..., Path],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify that reach lint --format github outputs GitHub Actions workflow commands."""
    write_skill(
        name="bad_naming",
        description="A sufficiently detailed description that satisfies standard rules.",
    )
    assert main(["lint", "--skills", str(tmp_path), "--format", "github"]) == 1
    captured = capsys.readouterr()
    assert "::error" in captured.out
    assert "title=invalid-name-format" in captured.out
    assert "bad_naming" in captured.out


def test_lint_positional_path(write_skill: Callable[..., Path], tmp_path: Path) -> None:
    """Verify that reach lint accepts target skill directory as positional argument."""
    write_skill(
        name="valid-skill",
        description="A sufficiently detailed description that satisfies standard rules.",
    )
    assert main(["lint", str(tmp_path)]) == 0


def test_lint_info_flag_and_summary(
    write_skill: Callable[..., Path],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify --info downgrades a rule to info severity and does not fail --strict."""
    write_skill(
        name="short-desc",
        description="Too brief.",
    )
    assert (
        main(
            [
                "lint",
                "--skills",
                str(tmp_path),
                "--strict",
                "--info",
                "description-too-short",
            ]
        )
        == 0
    )
    captured = capsys.readouterr()
    out = captured.out + captured.err
    assert "INFO" in out
    assert "1 info" in out
