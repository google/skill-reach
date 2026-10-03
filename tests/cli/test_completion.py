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

"""Test suite for reach completion shell tab autocomplete generation CLI command."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from cyclopts.completion._base import extract_completion_data
from cyclopts.completion._engine import compute_completions

from reach.cli import _reorder_argv, main
from reach.cli.app import app
from reach.cli.completion import _detect_shell
from reach.lint import RULES
from reach.runtime import FAKE_AGENT


def test_completion_generates_shell_script_with_unique_flags(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify reach completion prints valid shell scripts and has no duplicate flags."""
    for shell, needle in (
        ("zsh", "_cyclopts_reach"),
        ("bash", "complete -F _reach reach"),
        ("fish", "complete -c reach"),
    ):
        assert main(["completion", shell]) == 0
        out = capsys.readouterr().out
        assert needle in out

    data = extract_completion_data(app)
    for cmd_path, entry in data.items():
        seen_names: set[str] = set()
        for arg in entry.arguments:
            for name in arg.names:
                if not name.startswith("-"):
                    continue
                assert name not in seen_names, (
                    f"Duplicate completion option {name!r} in command {cmd_path}"
                )
                seen_names.add(name)


def test_reorder_argv_preserves_complete_sentinel() -> None:
    """Verify _reorder_argv does not reorder tokens when invoked via __complete."""
    raw = ["__complete", "lint", "--skills", "./skills", "--skill", ""]
    assert _reorder_argv(raw) == raw


def test_dynamic_completers_lint_rules_and_agents(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify dynamic completers return lint rules with descriptions and public agents."""
    rule_completions = compute_completions(app, ["lint", "--explain", ""])
    assert {c.value for c in rule_completions} == set(RULES.keys())
    assert all(c.help for c in rule_completions)

    assert main(["__complete", "lint", "--explain", "desc"]) == 0
    rule_out = capsys.readouterr().out
    assert "description-too-short\t" in rule_out

    agent_completions = compute_completions(app, ["lint", "--agent", ""])
    agent_values = {c.value for c in agent_completions}
    assert FAKE_AGENT not in agent_values
    assert "claude-code" in agent_values

    assert main(["__complete", "eval", "--agent", ""]) == 0
    agent_out = capsys.readouterr().out
    assert "claude-code" in agent_out
    assert FAKE_AGENT not in agent_out


def test_dynamic_completer_skill_names_respects_skills_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify complete_skill_names resolves skill names from --skills or CWD discovery."""
    skills_root = tmp_path / ".agents" / "skills"
    skill_dir = skills_root / "alpha-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: alpha-skill\ndescription: Alpha skill description.\n---\nBody\n",
        encoding="utf-8",
    )

    completions = compute_completions(
        app,
        ["lint", "--skills", str(skills_root), "--skill", ""],
    )
    assert [(c.value, c.help) for c in completions] == [
        ("alpha-skill", "Alpha skill description."),
    ]

    assert main(["__complete", "lint", "--skills", str(skills_root), "--skill", ""]) == 0
    skill_out = capsys.readouterr().out
    assert "alpha-skill\tAlpha skill description." in skill_out

    broken_root = tmp_path / "broken_skills" / "bad-skill"
    broken_root.mkdir(parents=True)
    (broken_root / "SKILL.md").write_text("---\n: invalid_yaml: [\n---\n", encoding="utf-8")

    for bad_dir in (tmp_path / "nonexistent", broken_root.parent):
        assert (
            compute_completions(
                app,
                ["lint", "--skills", str(bad_dir), "--skill", ""],
            )
            == []
        )

    monkeypatch.chdir(tmp_path)
    cwd_completions = compute_completions(app, ["lint", "--skill", ""])
    assert [(c.value, c.help) for c in cwd_completions] == [
        ("alpha-skill", "Alpha skill description."),
    ]

    empty_dir = tmp_path / "empty_workspace"
    empty_dir.mkdir()
    monkeypatch.chdir(empty_dir)
    assert compute_completions(app, ["lint", "--skill", ""]) == []


def test_completion_install_mode(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify reach completion --install delegates to app.install_completion."""
    with patch("cyclopts.App.install_completion", return_value=Path("~/.zshrc")) as mock_inst:
        assert main(["completion", "zsh", "--install"]) == 0
        assert mock_inst.call_count == 1
        assert mock_inst.call_args.kwargs.get("shell") == "zsh"
        captured = capsys.readouterr()
        out = captured.out + captured.err
        assert "installed" in out


@pytest.mark.parametrize(
    ("shell_env", "expected"),
    [
        ("/bin/zsh", "zsh"),
        ("/bin/bash", "bash"),
        ("/usr/local/bin/fish", "fish"),
    ],
)
def test_detect_shell(
    monkeypatch: pytest.MonkeyPatch,
    shell_env: str,
    expected: str,
) -> None:
    """Verify shell detection identifies active SHELL environment variable."""
    monkeypatch.setenv("SHELL", shell_env)
    assert _detect_shell() == expected
