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

"""Test suite for reach doctor environment diagnostics CLI command."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from reach.cli import main
from reach.cli.doctor import (
    _check_cli_binary,
    _check_config,
    _check_env_var,
    _check_google_adc,
    _check_python,
    _check_sdk,
    _check_skills,
    run_doctor_checks,
)
from reach.views import build_console, render_doctor_table


def test_doctor_runs_cleanly(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    """Verify reach doctor runs all diagnostics and outputs results."""
    assert main(["doctor", "--path", str(tmp_path)]) == 0
    captured = capsys.readouterr()
    out = captured.out + captured.err
    assert "Reach Environment & Runtime Diagnostics" in out
    assert "Python Version" in out
    assert "Claude Code" in out
    assert "Keyword" in out


def test_doctor_verbose_includes_remedy_panel(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """Verify reach doctor --verbose renders Recommended Actions panel."""
    assert main(["doctor", "--verbose", "--path", str(tmp_path)]) == 0
    captured = capsys.readouterr()
    out = captured.out + captured.err
    assert "Recommended Actions" in out


def test_check_python_unsupported() -> None:
    """Verify Python check fails on unsupported major.minor versions."""
    res = _check_python((3, 11, 0))
    assert res.status == "fail"
    assert "unsupported" in res.detail


@pytest.mark.parametrize(
    ("label", "binary", "agent_name", "alternates", "found_map", "expected_status", "expected_sub"),
    [
        (
            "Claude Code CLI",
            "claude",
            "claude-code",
            (),
            {"claude": "/usr/local/bin/claude"},
            "ok",
            "/usr/local/bin/claude",
        ),
        (
            "Antigravity CLI",
            "agy",
            "antigravity-cli",
            ("antigravity",),
            {"antigravity": "/usr/local/bin/antigravity"},
            "ok",
            "/usr/local/bin/antigravity",
        ),
        (
            "Goose CLI",
            "goose",
            "goose",
            (),
            {},
            "warn",
            "not found in PATH",
        ),
    ],
)
def test_check_cli_binary_found_and_missing(
    label: str,
    binary: str,
    agent_name: str,
    alternates: tuple[str, ...],
    found_map: dict[str, str],
    expected_status: str,
    expected_sub: str,
) -> None:
    """Verify CLI binary checks detect primary, alternate, and missing executables."""
    with patch("shutil.which", side_effect=found_map.get):
        res = _check_cli_binary(label, binary, agent_name, alternates=alternates)
        assert res.status == expected_status
        assert expected_sub in res.detail


def test_check_sdk_installed_and_missing() -> None:
    """Verify SDK module checks reflect importability."""
    with patch("importlib.util.find_spec", return_value=object()):
        res = _check_sdk("Antigravity SDK", "google.antigravity", "antigravity-sdk")
        assert res.status == "ok"

    with patch("importlib.util.find_spec", return_value=None):
        res = _check_sdk("Antigravity SDK", "google.antigravity", "antigravity-sdk")
        assert res.status == "warn"
        assert "not installed" in res.detail


def test_check_env_var_reports_configured_without_displaying_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify environment check confirms configuration without leaking API keys."""
    monkeypatch.setenv("TEST_API_KEY", "super-secret-key-12345")
    res = _check_env_var("TEST_API_KEY", "testing purpose")
    assert res.status == "ok"
    assert res.detail == "configured"
    assert "super-secret-key" not in res.detail


def test_check_google_adc(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify Google ADC check identifies custom and standard credential files."""
    custom_cred = tmp_path / "creds.json"
    custom_cred.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(custom_cred))
    res = _check_google_adc()
    assert res.status == "ok"
    assert "GOOGLE_APPLICATION_CREDENTIALS" in res.detail


def test_check_google_adc_detects_windows_appdata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify _check_google_adc locates credentials in Windows APPDATA directory."""
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    monkeypatch.delenv("CLOUDSDK_CONFIG", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "fake_home")
    appdata = tmp_path / "AppData" / "Roaming"
    gcloud = appdata / "gcloud"
    gcloud.mkdir(parents=True)
    adc = gcloud / "application_default_credentials.json"
    adc.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("APPDATA", str(appdata))

    res = _check_google_adc()
    assert res.status == "ok"
    assert "standard location" in res.detail


def test_check_google_adc_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify _check_google_adc reports warning when no credentials exist."""
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    monkeypatch.delenv("CLOUDSDK_CONFIG", raising=False)
    monkeypatch.delenv("APPDATA", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "fake_home")

    res = _check_google_adc()
    assert res.status == "warn"
    assert res.detail == "not found"


@pytest.mark.parametrize(
    ("rel_path", "expected_label"),
    [
        (".agents/skills", ".agents/skills/"),
        (".claude/skills", ".claude/skills/"),
        (".cursor/skills", ".cursor/skills/"),
        (".github/skills", ".github/skills/"),
        (".pi/skills", ".pi/skills/"),
        ("skills", "skills/"),
    ],
)
def test_check_skills_discovers_skills_across_standard_dirs(
    tmp_path: Path, rel_path: str, expected_label: str
) -> None:
    """Verify skill detection finds skills in all standard subdirectories."""
    skill_dir = tmp_path / Path(rel_path) / "test-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: test-skill\ndescription: A test skill\n---\n",
        encoding="utf-8",
    )
    res = _check_skills(tmp_path)
    assert res.status == "ok"
    assert f"1 in {expected_label}" in res.detail


def test_check_skills_warns_when_empty(tmp_path: Path) -> None:
    """Verify skill detection warns with all standard paths when no skills exist."""
    res = _check_skills(tmp_path)
    assert res.status == "warn"
    assert ".github/skills" in res.detail
    assert ".pi/skills" in res.detail


def test_check_config_valid_and_invalid(tmp_path: Path) -> None:
    """Verify reach.toml validation identifies valid and malformed configs."""
    cfg = tmp_path / "reach.toml"
    cfg.write_text('[runtime]\nagent = "keyword"\n', encoding="utf-8")
    res = _check_config(tmp_path)
    assert res.status == "ok"
    assert "default agent: 'keyword'" in res.detail

    cfg.write_text("invalid = toml [ broken", encoding="utf-8")
    res_err = _check_config(tmp_path)
    assert res_err.status == "fail"


def test_check_config_warns_with_target_directory(tmp_path: Path) -> None:
    """Verify _check_config includes the target directory path when not in cwd."""
    empty_dir = tmp_path / "custom_workdir"
    empty_dir.mkdir()
    res = _check_config(empty_dir)
    assert res.status == "warn"
    assert f"target directory '{empty_dir}'" in res.detail


def test_render_doctor_returns_1_on_failure() -> None:
    """Verify render_doctor exits 1 when any diagnostic failure is recorded."""
    console = build_console(quiet=True)
    fail_res = [
        ("Env", "Python", "fail", "Version 3.9 unsupported", "Upgrade Python"),
    ]
    assert render_doctor_table(console, fail_res) == 1


def test_run_doctor_checks_includes_google_adc(tmp_path: Path) -> None:
    """Verify run_doctor_checks includes Google Cloud ADC diagnostic check."""
    results = run_doctor_checks(tmp_path)
    adc_checks = [r for r in results if r.name == "Google Cloud ADC"]
    assert len(adc_checks) == 1
    assert adc_checks[0].category == "Credentials & Environment"


def test_check_agent_registry_uses_custom_workdir_cache(tmp_path: Path) -> None:
    """Verify _check_agent_registry checks cache scoped to the specified workdir."""
    from reach.cli.doctor import _check_agent_registry

    custom_workdir = tmp_path / "custom_workdir"
    custom_cache = custom_workdir / ".reach" / "cache" / "registry" / "my-project" / "global"
    custom_cache.mkdir(parents=True)
    manifest = custom_cache / ".manifest.json"
    manifest.write_text('{"skills": []}', encoding="utf-8")

    with (
        patch("reach.config.resolve_registry_project", return_value="my-project"),
        patch("reach.registry.is_adc_available", return_value=True),
    ):
        res = _check_agent_registry(custom_workdir)
        assert res.status == "ok"
        assert "bytes cached" in res.detail


def test_doctor_positional_path(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """Verify reach doctor accepts target directory as a positional argument."""
    assert main(["doctor", str(tmp_path)]) == 0
    captured = capsys.readouterr()
    out = captured.out + captured.err
    assert "Reach Environment & Runtime Diagnostics" in out
