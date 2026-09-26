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

from reach.cli import main
from reach.cli.completion import _detect_shell


def test_completion_help(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify reach completion --help displays shell choices and options."""
    assert main(["completion", "--help"]) == 0
    out = capsys.readouterr().out
    assert "completion" in out
    assert "--install" in out


@pytest.mark.parametrize(
    ("shell", "needle"),
    [
        ("zsh", "_cyclopts_reach"),
        ("bash", "complete -F _reach reach"),
        ("fish", "complete -c reach"),
    ],
)
def test_completion_generates_shell_script(
    shell: str,
    needle: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify reach completion prints valid shell completion definitions."""
    assert main(["completion", shell]) == 0
    out = capsys.readouterr().out
    assert needle in out


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


def test_completion_deduplicates_dual_positional_and_flag_options() -> None:
    """Verify _cached_extract_completion_data dedupes flags and skips redundant evolve."""
    from reach.cli.app import app
    from reach.cli.completion import (
        _cached_extract_completion_data,
        _completion_data_cache,
        _dedupe_argument_collection,
    )

    _completion_data_cache.clear()
    data = _cached_extract_completion_data(app)
    for cmd_path, entry in data.items():
        seen_names: set[str] = set()
        for arg in entry.arguments:
            for name in arg.parameter.name or ():
                assert name not in seen_names, (
                    f"Duplicate completion option {name!r} in command {cmd_path}"
                )
                seen_names.add(name)
        second_pass = _dedupe_argument_collection(entry.arguments)
        for orig_arg, deduped_arg in zip(entry.arguments, second_pass, strict=True):
            assert orig_arg is deduped_arg

    import attrs
    from cyclopts.argument import ArgumentCollection

    sample_collection = next(iter(data.values())).arguments
    unnamed_arg = attrs.evolve(
        sample_collection[0],
        parameter=attrs.evolve(sample_collection[0].parameter, name=()),
    )
    deduped_with_unnamed = _dedupe_argument_collection(ArgumentCollection([unnamed_arg]))
    assert len(deduped_with_unnamed) == 1
    assert deduped_with_unnamed[0] is unnamed_arg
