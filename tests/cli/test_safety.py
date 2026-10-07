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

"""Verify CLI safety confirmation gates, bypass mechanisms, and catalog formatting."""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import patch

import pytest

from reach.cli.app import app
from reach.cli.safety import (
    catalog_summary,
    confirm_skill_execution,
    runtime_summary,
)
from reach.models import Skill
from reach.views import Console


def test_yes_flag_registered_on_target_commands() -> None:
    """Verify --yes and -y flags are registered on eval, sweep, check, and optimize."""
    for cmd_name in ("eval", "sweep", "check", "optimize"):
        cmd = app[cmd_name]
        arg_names = {
            name
            for arg in cmd.assemble_argument_collection()
            if arg.parameter.name
            for name in arg.parameter.name
        }
        assert "--yes" in arg_names, f"--yes missing from 'reach {cmd_name}'"
        assert "-y" in arg_names, f"-y missing from 'reach {cmd_name}'"


@pytest.fixture
def test_console() -> Console:
    """Provide a capture-enabled Console instance for testing."""
    return Console(file=io.StringIO(), force_terminal=True, width=100)


@pytest.fixture
def sample_skills(tmp_path: Path) -> list[Skill]:
    """Provide sample skills across multiple directories for testing."""
    proj_skills = tmp_path / "workspace" / "project" / ".agents" / "skills"
    home_skills = tmp_path / "home" / "user" / ".claude" / "skills"
    return [
        Skill(
            name="skill-a",
            description="First test skill.",
            path=proj_skills / "skill-a",
        ),
        Skill(
            name="skill-b",
            description="Second test skill.",
            path=proj_skills / "skill-b",
        ),
        Skill(
            name="skill-c",
            description="Third test skill.",
            path=home_skills / "skill-c",
        ),
    ]


def test_catalog_summary_multi_root(sample_skills: list[Skill], tmp_path: Path) -> None:
    """Verify catalog summary formats counts across distinct directory roots."""
    roots = [
        tmp_path / "workspace" / "project" / ".agents" / "skills",
        tmp_path / "home" / "user" / ".claude" / "skills",
    ]
    headline, bullets = catalog_summary(sample_skills, roots=roots)
    assert headline == "3 skills resolved from 2 locations:"
    assert len(bullets) == 2
    assert "2 skills" in bullets[0]
    assert "1 skill" in bullets[1]


def test_catalog_summary_single_root() -> None:
    """Verify catalog summary formats a single directory root cleanly."""
    root = Path("/workspace/project/skills")
    skills = [
        Skill(name="s1", description="One", path=root / "s1"),
        Skill(name="s2", description="Two", path=root / "s2"),
    ]
    headline, bullets = catalog_summary(skills, roots=[root])
    assert "2 skills in" in headline
    assert not bullets


def test_catalog_summary_count_only() -> None:
    """Verify catalog summary handles raw integer counts without roots."""
    headline, bullets = catalog_summary(5)
    assert headline == "5 skills"
    assert not bullets


@pytest.mark.parametrize(
    ("runtime_name", "expected_fragment"),
    [
        ("pi", "pi (No built-in sandbox)"),
        ("antigravity-cli", "antigravity-cli (--dangerously-skip-permissions)"),
        ("claude-code", "claude-code (tool-restricted session)"),
        ("goose", "goose"),
    ],
)
def test_runtime_summary_descriptions(runtime_name: str, expected_fragment: str) -> None:
    """Verify runtime summary describes driver execution models accurately."""
    summary = runtime_summary(runtime_name)
    assert expected_fragment in summary


@pytest.mark.parametrize(
    ("runtime_name", "dry_run", "yes", "trusted"),
    [
        ("pi", True, False, False),
        ("fake", False, False, False),
        ("keyword", False, False, False),
        ("pi", False, True, False),
        ("pi", False, False, True),
    ],
)
def test_confirm_skill_execution_bypass_conditions(
    test_console: Console,
    runtime_name: str,
    dry_run: bool,
    yes: bool,
    trusted: bool,
) -> None:
    """Verify dry_run, simulated runtimes, yes, and trusted flags bypass confirmation."""
    code = confirm_skill_execution(
        test_console,
        runtime_name=runtime_name,
        skills=5,
        dry_run=dry_run,
        yes=yes,
        trusted=trusted,
    )
    assert code == 0


@pytest.mark.parametrize("val", ["1", "true", "yes"])
def test_confirm_bypass_via_environment_variables(
    test_console: Console,
    monkeypatch: pytest.MonkeyPatch,
    val: str,
) -> None:
    """Verify REACH_YES bypasses confirmation prompts across truthy values."""
    monkeypatch.setenv("REACH_YES", val)
    code = confirm_skill_execution(
        test_console,
        runtime_name="pi",
        skills=5,
    )
    assert code == 0


def test_confirm_fail_closed_in_non_tty(
    test_console: Console,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify non-interactive TTY fails closed with exit code 2 and prints remedy."""
    with (
        patch("sys.stdin.isatty", return_value=False),
        patch("sys.stdout.isatty", return_value=False),
    ):
        code = confirm_skill_execution(
            test_console,
            runtime_name="pi",
            skills=5,
        )
    assert code == 2
    captured = capsys.readouterr()
    assert "Confirmation required to probe skills with runtime 'pi'" in captured.err
    assert "pass '--yes' / '-y'" in captured.err


def test_confirm_interactive_accept(test_console: Console) -> None:
    """Verify user typing 'y' in interactive terminal confirms execution with code 0."""
    with (
        patch("sys.stdin.isatty", return_value=True),
        patch("sys.stdout.isatty", return_value=True),
        patch("builtins.input", return_value="y"),
    ):
        code = confirm_skill_execution(
            test_console,
            runtime_name="pi",
            skills=5,
        )
    assert code == 0


def test_confirm_interactive_decline(test_console: Console) -> None:
    """Verify user typing 'n' in interactive terminal aborts with code 1."""
    with (
        patch("sys.stdin.isatty", return_value=True),
        patch("sys.stdout.isatty", return_value=True),
        patch("builtins.input", return_value="n"),
    ):
        code = confirm_skill_execution(
            test_console,
            runtime_name="pi",
            skills=5,
        )
    assert code == 1


def test_confirm_interactive_eof_aborts(test_console: Console) -> None:
    """Verify EOF / Ctrl+D in interactive terminal aborts cleanly with code 1."""
    with (
        patch("sys.stdin.isatty", return_value=True),
        patch("sys.stdout.isatty", return_value=True),
        patch("builtins.input", side_effect=EOFError),
    ):
        code = confirm_skill_execution(
            test_console,
            runtime_name="pi",
            skills=5,
        )
    assert code == 1


def test_confirm_interactive_when_stdout_redirected(
    test_console: Console,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify confirmation works when stdout is redirected if stdin and stderr are TTYs."""
    with (
        patch("sys.stdin.isatty", return_value=True),
        patch("sys.stdout.isatty", return_value=False),
        patch("sys.stderr.isatty", return_value=True),
        patch("builtins.input", return_value="y"),
    ):
        code = confirm_skill_execution(
            test_console,
            runtime_name="pi",
            skills=5,
        )
    assert code == 0
    captured = capsys.readouterr()
    assert captured.out == ""
