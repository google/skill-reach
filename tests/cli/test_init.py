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

"""Test suite for reach init project scaffolding CLI command."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

from reach.cli import main
from reach.cli.init import _detect_default_agent, _detect_skills_directory

if TYPE_CHECKING:
    import pytest


def test_init_help(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify reach init --help displays usage and options."""
    assert main(["init", "--help"]) == 0
    out = capsys.readouterr().out
    assert "init" in out
    assert "--agent" in out
    assert "--skills" in out
    assert "--force" in out


def test_init_generates_config_and_directories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify reach init creates reach.toml, .reach/ cache, and .agents/skills/."""
    monkeypatch.chdir(tmp_path)
    assert main(["init"]) == 0

    toml_path = tmp_path / "reach.toml"
    assert toml_path.is_file()
    content = toml_path.read_text(encoding="utf-8")
    assert "[runtime]" in content
    assert "[study]" in content
    assert (tmp_path / ".agents" / "skills").is_dir() or (tmp_path / ".claude" / "skills").is_dir()
    assert (tmp_path / ".reach").is_dir()


def test_init_with_custom_agent_and_skills(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify reach init respects --agent and --skills flag inputs."""
    monkeypatch.chdir(tmp_path)
    custom_skills = tmp_path / "my-custom-skills"
    assert (
        main(
            [
                "init",
                "--agent",
                "antigravity-cli",
                "--skills",
                str(custom_skills),
            ]
        )
        == 0
    )

    toml_path = tmp_path / "reach.toml"
    content = toml_path.read_text(encoding="utf-8")
    assert 'agent = "antigravity-cli"' in content
    assert f'skills = "{custom_skills}"' in content or 'skills = "my-custom-skills"' in content
    assert custom_skills.is_dir()


def test_init_refuses_to_overwrite_without_force(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify reach init fails when reach.toml already exists and --force is not passed."""
    monkeypatch.chdir(tmp_path)
    existing_toml = tmp_path / "reach.toml"
    existing_toml.write_text("# existing", encoding="utf-8")

    assert main(["init"]) == 1
    captured = capsys.readouterr()
    out = captured.out + captured.err
    assert "already exists" in out
    assert existing_toml.read_text(encoding="utf-8") == "# existing"


def test_init_overwrites_with_force(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify reach init --force successfully overwrites existing reach.toml."""
    monkeypatch.chdir(tmp_path)
    existing_toml = tmp_path / "reach.toml"
    existing_toml.write_text("# existing", encoding="utf-8")

    assert main(["init", "--force", "--agent", "keyword"]) == 0
    content = existing_toml.read_text(encoding="utf-8")
    assert 'agent = "keyword"' in content


def test_detect_skills_directory_finds_existing(tmp_path: Path) -> None:
    """Verify skill directory detection picks existing .claude/skills if present."""
    claude_skills = tmp_path / ".claude" / "skills"
    claude_skills.mkdir(parents=True)
    detected = _detect_skills_directory(tmp_path)
    assert detected == claude_skills


def test_detect_default_agent() -> None:
    """Verify default agent detection prioritizes available binaries."""
    with patch("shutil.which", return_value="/usr/local/bin/claude"):
        assert _detect_default_agent() == "claude-code"

    with patch("shutil.which", side_effect=lambda x: "/usr/local/bin/agy" if x == "agy" else None):
        assert _detect_default_agent() == "antigravity-cli"


def test_detect_skills_directory_agent_aware(tmp_path: Path) -> None:
    """Verify skill directory detection respects agent native directory when none exist."""
    assert (
        _detect_skills_directory(tmp_path, agent="claude-code") == tmp_path / ".claude" / "skills"
    )
    assert _detect_skills_directory(tmp_path, agent="pi") == tmp_path / ".pi" / "skills"
    assert (
        _detect_skills_directory(tmp_path, agent="antigravity-cli")
        == tmp_path / ".agents" / "skills"
    )
    assert _detect_skills_directory(tmp_path, agent="unknown") == tmp_path / ".agents" / "skills"

    # If an existing directory exists, it still takes precedence
    existing = tmp_path / "skills"
    existing.mkdir()
    assert _detect_skills_directory(tmp_path, agent="claude-code") == existing


def test_init_with_agent_claude_code_creates_claude_skills(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify reach init --agent claude-code creates .claude/skills when no skills dir exists."""
    monkeypatch.chdir(tmp_path)
    assert main(["init", "--agent", "claude-code"]) == 0
    assert (tmp_path / ".claude" / "skills").is_dir()
    toml_path = tmp_path / "reach.toml"
    content = toml_path.read_text(encoding="utf-8")
    assert 'skills = ".claude/skills"' in content


def test_init_with_path_flag(tmp_path: Path) -> None:
    """Verify reach init --path initializes reach project in specified directory."""
    target_dir = tmp_path / "subproject"
    assert main(["init", "--path", str(target_dir), "--agent", "claude-code"]) == 0
    assert (target_dir / "reach.toml").is_file()
    assert (target_dir / ".claude" / "skills").is_dir()
    toml_path = target_dir / "reach.toml"
    content = toml_path.read_text(encoding="utf-8")
    assert 'skills = ".claude/skills"' in content
