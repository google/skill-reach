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

import csv
import io
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from pydantic import ValidationError

from reach.cli import main
from reach.cli.doctor import (
    CheckCategory,
    CheckResult,
    CheckStatus,
    DoctorReport,
    _check_agent_registry,
    _check_cli_binary,
    _check_config,
    _check_env_var,
    _check_google_adc,
    _check_python,
    _check_sdk,
    _check_skills,
    run_doctor_checks,
)
from reach.views import (
    render_doctor,
    render_doctor_csv,
    render_doctor_json,
    render_doctor_jsonl,
    render_doctor_table,
)

if TYPE_CHECKING:
    from collections.abc import Callable


_DOCTOR_ENV_VARS: tuple[str, ...] = (
    "AGY_ADC_AUTH",
    "APPDATA",
    "CLOUDSDK_CONFIG",
    "CLOUDSDK_CORE_PROJECT",
    "GCLOUD_PROJECT",
    "GEMINI_API_KEY",
    "GEMINI_API_KEY_FILE",
    "GOOGLE_API_KEY",
    "GOOGLE_API_KEY_FILE",
    "GOOGLE_APPLICATION_CREDENTIALS",
    "GOOGLE_CLOUD_PROJECT",
    "GOOGLE_CLOUD_QUOTA_PROJECT",
    "GOOGLE_GENAI_USE_ENTERPRISE",
    "GOOGLE_GENAI_USE_VERTEXAI",
)


@pytest.fixture
def doctor_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Provide a hermetic environment with credential variables scrubbed and isolated HOME."""
    for var in _DOCTOR_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    fake_home = tmp_path / "fake_home"
    fake_home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(Path, "home", lambda: fake_home)
    monkeypatch.setenv("HOME", str(fake_home))
    return fake_home


# ==============================================================================
# 1. Pydantic Boundary Models (CheckResult & DoctorReport)
# ==============================================================================


def test_check_result_model_validates_and_freezes_fields() -> None:
    """Verify CheckResult enforces frozen immutability, non-empty strings, and extra=forbid."""
    res = CheckResult(
        category=CheckCategory.ENVIRONMENT,
        name="  Python Version  ",
        status="ok",
        detail=" 3.13.0 ",
        remedy="  Upgrade Python  ",
    )
    assert res.category == "Environment"
    assert res.name == "Python Version"
    assert res.status is CheckStatus.OK
    assert res.detail == "3.13.0"
    assert res.remedy == "Upgrade Python"

    with pytest.raises(ValidationError):
        setattr(res, "status", CheckStatus.FAIL)  # noqa: B010


@pytest.mark.parametrize(
    "invalid_kwargs",
    [
        {"category": "   ", "name": "Python", "status": "ok", "detail": "3.13"},
        {"category": "Environment", "name": "", "status": "ok", "detail": "3.13"},
        {"category": "Environment", "name": "Python", "status": "unknown", "detail": "3.13"},
        {"category": "Environment", "name": "Python", "status": "ok", "detail": ""},
        {
            "category": "Environment",
            "name": "Python",
            "status": "ok",
            "detail": "3.13",
            "extra_field": "bad",
        },
    ],
    ids=["blank_category", "empty_name", "invalid_status", "empty_detail", "extra_forbid"],
)
def test_check_result_rejects_malformed_inputs(invalid_kwargs: dict[str, Any]) -> None:
    """Verify CheckResult fails fast on empty strings, unknown statuses, or extra fields."""
    with pytest.raises(ValidationError):
        CheckResult(**invalid_kwargs)


def test_doctor_report_coerce_supports_models_and_legacy_tuples() -> None:
    """Verify DoctorReport.coerce normalizes DoctorReport, CheckResult, and 5-tuples."""
    model_item = CheckResult(
        category=CheckCategory.ENVIRONMENT,
        name="Python Version",
        status=CheckStatus.OK,
        detail="3.13.0",
    )
    tuple_item = ("Configuration", "reach.toml", "warn", "No reach.toml", "Run reach init")
    report = DoctorReport.coerce([model_item, tuple_item])
    assert len(report.checks) == 2
    assert report.has_warnings is True
    assert report.has_failures is False
    assert DoctorReport.coerce(report) is report


# ==============================================================================
# 2. Individual Check Helpers (Happy, Sad, Edge & Seam)
# ==============================================================================


@pytest.mark.parametrize(
    ("version_tuple", "expected_status", "expected_sub"),
    [
        ((3, 12, 4), CheckStatus.OK, "3.12.4"),
        ((3, 13), CheckStatus.OK, "3.13 ("),
        ((3, 11, 0), CheckStatus.FAIL, "3.11.0 (unsupported, requires >= 3.12)"),
    ],
    ids=["supported_3_tuple", "supported_2_tuple_edge", "unsupported_3_11"],
)
def test_check_python_versions(
    version_tuple: tuple[int, ...],
    expected_status: CheckStatus,
    expected_sub: str,
) -> None:
    """Verify _check_python handles 3-tuple, 2-tuple, and unsupported Python versions."""
    res = _check_python(version_tuple)
    assert res.category == CheckCategory.ENVIRONMENT
    assert res.status is expected_status
    assert expected_sub in res.detail


@pytest.mark.parametrize(
    ("label", "binary", "agent_name", "alternates", "found_map", "expected_status", "expected_sub"),
    [
        (
            "Claude Code CLI",
            "claude",
            "claude-code",
            (),
            {"claude": "/usr/local/bin/claude"},
            CheckStatus.OK,
            "/usr/local/bin/claude",
        ),
        (
            "Antigravity CLI",
            "agy",
            "antigravity-cli",
            ("antigravity",),
            {"antigravity": "/usr/local/bin/antigravity"},
            CheckStatus.OK,
            "/usr/local/bin/antigravity",
        ),
        (
            "Antigravity CLI",
            "agy",
            "antigravity-cli",
            ("antigravity",),
            {"agy": "/opt/bin/agy", "antigravity": "/usr/local/bin/antigravity"},
            CheckStatus.OK,
            "/opt/bin/agy",
        ),
        (
            "Goose CLI",
            "goose",
            "goose",
            (),
            {},
            CheckStatus.WARN,
            "not found in PATH",
        ),
    ],
    ids=["primary_found", "alternate_found", "primary_wins_over_alternate", "missing"],
)
def test_check_cli_binary_found_and_missing(
    monkeypatch: pytest.MonkeyPatch,
    label: str,
    binary: str,
    agent_name: str,
    alternates: tuple[str, ...],
    found_map: dict[str, str],
    expected_status: CheckStatus,
    expected_sub: str,
) -> None:
    """Verify CLI binary checks detect primary, alternate, and missing executables."""
    monkeypatch.setattr("shutil.which", found_map.get)
    res = _check_cli_binary(label, binary, agent_name, alternates=alternates)
    assert res.category == CheckCategory.RUNTIMES
    assert res.status is expected_status
    assert expected_sub in res.detail


@pytest.mark.parametrize(
    ("side_effect", "return_value", "category", "expected_status", "expected_detail"),
    [
        (None, object(), CheckCategory.RUNTIMES, CheckStatus.OK, "installed and importable"),
        (None, None, CheckCategory.ENVIRONMENT, CheckStatus.WARN, "not installed"),
        (
            ModuleNotFoundError("No module named 'google'"),
            None,
            CheckCategory.RUNTIMES,
            CheckStatus.WARN,
            "not installed",
        ),
        (
            ImportError("Broken parent import"),
            None,
            CheckCategory.RUNTIMES,
            CheckStatus.WARN,
            "not installed",
        ),
        (
            AttributeError("Parent is not a package"),
            None,
            CheckCategory.RUNTIMES,
            CheckStatus.WARN,
            "not installed",
        ),
        (
            ValueError("Empty module name"),
            None,
            CheckCategory.RUNTIMES,
            CheckStatus.WARN,
            "not installed",
        ),
    ],
    ids=[
        "installed",
        "missing_semantic",
        "module_not_found",
        "import_error",
        "attribute_error",
        "value_error",
    ],
)
def test_check_sdk_importability(
    monkeypatch: pytest.MonkeyPatch,
    side_effect: Exception | None,
    return_value: object | None,
    category: CheckCategory,
    expected_status: CheckStatus,
    expected_detail: str,
) -> None:
    """Verify SDK module checks handle found, missing, and broken parent packages."""

    def _fake_find_spec(_name: str) -> object | None:
        if side_effect is not None:
            raise side_effect
        return return_value

    monkeypatch.setattr("importlib.util.find_spec", _fake_find_spec)
    res = _check_sdk(
        "Antigravity SDK",
        "google.antigravity",
        "antigravity-sdk",
        category=category,
    )
    assert res.category == category
    assert res.status is expected_status
    assert expected_detail in res.detail
    if expected_status is CheckStatus.WARN:
        assert "uv add 'skill-reach[antigravity-sdk]'" in res.remedy


@pytest.mark.parametrize(
    ("env_map", "expected_status", "expected_detail"),
    [
        (
            {"GEMINI_API_KEY": "super-secret-key-12345"},
            CheckStatus.OK,
            "configured",
        ),
        (
            {"GOOGLE_API_KEY": "fallback-secret-key-67890"},
            CheckStatus.OK,
            "configured via GOOGLE_API_KEY",
        ),
        (
            {},
            CheckStatus.WARN,
            "not set",
        ),
    ],
    ids=["primary_set", "alternate_google_api_key_set", "unset"],
)
def test_check_env_var_states_and_redaction(
    doctor_env: Path,
    monkeypatch: pytest.MonkeyPatch,
    env_map: dict[str, str],
    expected_status: CheckStatus,
    expected_detail: str,
) -> None:
    """Verify _check_env_var handles primary, alternate, and unset states safely."""
    assert doctor_env.is_dir()
    for k, v in env_map.items():
        monkeypatch.setenv(k, v)
    res = _check_env_var(
        "GEMINI_API_KEY",
        "Google Gemini model completions",
        alternates=("GOOGLE_API_KEY",),
    )
    assert res.category == CheckCategory.CREDENTIALS
    assert res.status is expected_status
    assert res.detail == expected_detail
    assert "secret-key" not in res.detail


@pytest.mark.parametrize("use_file", [False, True], ids=["env_var", "file_pointer"])
@pytest.mark.parametrize("is_fallback", [False, True], ids=["primary", "fallback"])
def test_check_env_var_supports_file_pointer_and_redacts_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    doctor_env: Path,
    use_file: bool,
    is_fallback: bool,
) -> None:
    """Verify _check_env_var confirms direct and *_FILE config for primary and fallback vars."""
    assert doctor_env.is_dir()
    dummy_payload = "dummy-value-12345"
    key_prefix = "GOOGLE_API_KEY" if is_fallback else "GEMINI_API_KEY"
    expected_detail = "configured via GOOGLE_API_KEY" if is_fallback else "configured"

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY_FILE", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY_FILE", raising=False)

    if use_file:
        secret_file = tmp_path / "secret.txt"
        secret_file.write_text(f"{dummy_payload}\n", encoding="utf-8")
        monkeypatch.setenv(f"{key_prefix}_FILE", str(secret_file))
    else:
        monkeypatch.setenv(key_prefix, dummy_payload)

    res = _check_env_var(
        "GEMINI_API_KEY",
        "Google Gemini model completions",
        alternates=("GOOGLE_API_KEY",),
    )
    assert res.category == CheckCategory.CREDENTIALS
    assert res.status is CheckStatus.OK
    assert res.detail == expected_detail
    assert dummy_payload not in res.detail


def test_check_google_adc_valid_custom_and_tilde_expansion(
    doctor_env: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify _check_google_adc expands ~ in GOOGLE_APPLICATION_CREDENTIALS."""
    custom_cred = doctor_env / "creds.json"
    custom_cred.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "~/creds.json")
    res = _check_google_adc()
    assert res.category == CheckCategory.CREDENTIALS
    assert res.status is CheckStatus.OK
    assert "configured via GOOGLE_APPLICATION_CREDENTIALS" in res.detail
    assert str(custom_cred) in res.detail


def test_check_google_adc_missing_custom_file_warns_even_when_standard_adc_exists(
    doctor_env: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify broken GOOGLE_APPLICATION_CREDENTIALS warns instead of using standard ADC."""
    gcloud_dir = doctor_env / ".config" / "gcloud"
    gcloud_dir.mkdir(parents=True)
    (gcloud_dir / "application_default_credentials.json").write_text("{}", encoding="utf-8")

    missing_cred = doctor_env / "nonexistent.json"
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(missing_cred))

    res = _check_google_adc()
    assert res.status is CheckStatus.WARN
    assert "GOOGLE_APPLICATION_CREDENTIALS points to missing file" in res.detail
    assert "GOOGLE_APPLICATION_CREDENTIALS" in res.remedy


def test_check_google_adc_detects_windows_appdata(
    doctor_env: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify _check_google_adc locates credentials in Windows APPDATA directory."""
    appdata = doctor_env / "AppData" / "Roaming"
    gcloud = appdata / "gcloud"
    gcloud.mkdir(parents=True)
    adc = gcloud / "application_default_credentials.json"
    adc.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("APPDATA", str(appdata))

    res = _check_google_adc()
    assert res.status is CheckStatus.OK
    assert "standard location" in res.detail


def test_check_google_adc_missing(doctor_env: Path) -> None:
    """Verify _check_google_adc reports warning when no credentials exist."""
    assert doctor_env.is_dir()
    res = _check_google_adc()
    assert res.status is CheckStatus.WARN
    assert res.detail == "not found"


@pytest.mark.parametrize(
    "toggle_var",
    [
        "AGY_ADC_AUTH",
        "GOOGLE_GENAI_USE_VERTEXAI",
        "GOOGLE_GENAI_USE_ENTERPRISE",
    ],
)
@pytest.mark.parametrize("auth_val", ["true", "1", "yes"])
@pytest.mark.parametrize("has_creds", [True, False], ids=["with_creds", "without_creds"])
def test_check_google_adc_reports_vertex_toggles(
    doctor_env: Path,
    monkeypatch: pytest.MonkeyPatch,
    toggle_var: str,
    auth_val: str,
    has_creds: bool,
) -> None:
    """Verify _check_google_adc reports active Vertex/ADC toggles across ok and warn states."""
    monkeypatch.setenv(toggle_var, auth_val)
    if has_creds:
        custom_cred = doctor_env / "creds.json"
        custom_cred.write_text("{}", encoding="utf-8")
        monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(custom_cred))

    res = _check_google_adc()
    expected_status = CheckStatus.OK if has_creds else CheckStatus.WARN
    assert res.status is expected_status
    assert f"{toggle_var} enabled" in res.detail


@pytest.mark.parametrize(
    ("project", "has_adc", "cached_bytes", "expected_status", "expected_detail_sub"),
    [
        (
            "my-project",
            True,
            256,
            CheckStatus.OK,
            "Project: my-project, ADC available (256 bytes cached)",
        ),
        (
            "my-project",
            True,
            2048,
            CheckStatus.OK,
            "Project: my-project, ADC available (2.0 KB cached)",
        ),
        (
            "my-project",
            False,
            0,
            CheckStatus.WARN,
            "Project: my-project, but ADC credentials not found",
        ),
        (
            None,
            True,
            0,
            CheckStatus.OK,
            "ADC available (no default project configured)",
        ),
        (
            None,
            False,
            0,
            CheckStatus.WARN,
            "Project not set and ADC credentials not found",
        ),
    ],
    ids=[
        "project_and_adc_with_byte_cache",
        "project_and_adc_with_kb_cache",
        "project_without_adc",
        "adc_without_project",
        "neither_project_nor_adc",
    ],
)
def test_check_agent_registry_state_matrix(
    tmp_path: Path,
    doctor_env: Path,
    monkeypatch: pytest.MonkeyPatch,
    project: str | None,
    has_adc: bool,
    cached_bytes: int,
    expected_status: CheckStatus,
    expected_detail_sub: str,
) -> None:
    """Verify _check_agent_registry handles all 4 (project, has_adc) states and cache reporting."""
    assert doctor_env.is_dir()
    custom_workdir = tmp_path / "custom_workdir"
    custom_workdir.mkdir(parents=True)
    if cached_bytes > 0:
        custom_cache = custom_workdir / ".reach" / "cache" / "registry" / "my-project" / "global"
        custom_cache.mkdir(parents=True)
        (custom_cache / ".manifest.json").write_bytes(b"x" * cached_bytes)

    monkeypatch.setattr("reach.config.resolve_registry_project", lambda **_kw: project)
    monkeypatch.setattr("reach.registry.is_adc_available", lambda: has_adc)

    res = _check_agent_registry(custom_workdir)
    assert res.category == CheckCategory.CREDENTIALS
    assert res.status is expected_status
    assert expected_detail_sub in res.detail


def test_check_agent_registry_isolates_from_cwd_reach_toml(
    tmp_path: Path,
    doctor_env: Path,
    monkeypatch: pytest.MonkeyPatch,
    write_reach_toml: Callable[..., Path],
) -> None:
    """Verify _check_agent_registry does not leak CWD reach.toml when inspecting another path."""
    assert doctor_env.is_dir()
    cwd_dir = tmp_path / "cwd_with_project"
    cwd_dir.mkdir()
    write_reach_toml('[registry]\nproject = "leaked-cwd-project"\n', directory=cwd_dir)
    monkeypatch.chdir(cwd_dir)
    monkeypatch.setattr("reach.registry.is_adc_available", lambda: True)

    other_dir = tmp_path / "other_target"
    other_dir.mkdir()
    res = _check_agent_registry(other_dir)
    assert "leaked-cwd-project" not in res.detail
    assert "no default project configured" in res.detail


def test_check_agent_registry_survives_corrupt_target_toml(
    tmp_path: Path,
    doctor_env: Path,
    monkeypatch: pytest.MonkeyPatch,
    write_reach_toml: Callable[..., Path],
) -> None:
    """Verify _check_agent_registry falls back cleanly when target reach.toml is corrupt."""
    assert doctor_env.is_dir()
    monkeypatch.setattr("reach.registry.is_adc_available", lambda: True)
    write_reach_toml("invalid = toml [ broken", directory=tmp_path)

    res = _check_agent_registry(tmp_path)
    assert res.status is CheckStatus.OK
    assert "no default project configured" in res.detail


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
    tmp_path: Path,
    write_skill: Callable[..., Path],
    rel_path: str,
    expected_label: str,
) -> None:
    """Verify skill detection finds skills in all standard subdirectories."""
    write_skill("test-skill", root=tmp_path / rel_path)
    res = _check_skills(tmp_path)
    assert res.category == CheckCategory.SKILLS
    assert res.status is CheckStatus.OK
    assert f"1 in {expected_label}" in res.detail


@pytest.mark.parametrize("symlink_mode", ["directory_symlink", "per_skill_symlink"])
def test_check_skills_deduplicates_directory_and_per_skill_symlinks(
    tmp_path: Path,
    write_skill: Callable[..., Path],
    write_reach_toml: Callable[..., Path],
    symlink_mode: str,
) -> None:
    """Verify _check_skills counts symlinked directories and per-skill symlinks only once."""
    write_skill("alpha-skill", root=tmp_path / ".agents" / "skills")
    write_reach_toml('[study]\nskills = ".agents/skills"\n', directory=tmp_path)

    if symlink_mode == "directory_symlink":
        claude_parent = tmp_path / ".claude"
        claude_parent.mkdir(parents=True)
        (claude_parent / "skills").symlink_to(tmp_path / ".agents" / "skills")
    else:
        claude_skills = tmp_path / ".claude" / "skills"
        claude_skills.mkdir(parents=True)
        (claude_skills / "alpha-skill").symlink_to(
            tmp_path / ".agents" / "skills" / "alpha-skill",
        )

    res = _check_skills(tmp_path)
    assert res.status is CheckStatus.OK
    assert res.detail == "Found 1 skill(s): 1 in .agents/skills/"


@pytest.mark.parametrize(
    ("is_valid_frontmatter", "expected_status", "expected_sub"),
    [
        (True, CheckStatus.OK, "Found 1 skill(s): 1 in ./"),
        (False, CheckStatus.WARN, "No skills detected"),
    ],
    ids=["valid_root_skill", "invalid_root_frontmatter"],
)
def test_check_skills_supports_root_skill_md(
    tmp_path: Path,
    write_skill: Callable[..., Path],
    is_valid_frontmatter: bool,
    expected_status: CheckStatus,
    expected_sub: str,
) -> None:
    """Verify _check_skills detects valid root SKILL.md and ignores invalid frontmatter."""
    if is_valid_frontmatter:
        write_skill("root-skill", path=tmp_path)
    else:
        (tmp_path / "SKILL.md").write_text("not frontmatter", encoding="utf-8")

    res = _check_skills(tmp_path)
    assert res.status is expected_status
    assert expected_sub in res.detail


@pytest.mark.parametrize("use_external_abs", [False, True], ids=["relative_study", "external_abs"])
def test_check_skills_supports_study_skills_config(
    tmp_path: Path,
    write_skill: Callable[..., Path],
    write_reach_toml: Callable[..., Path],
    use_external_abs: bool,
) -> None:
    """Verify _check_skills discovers custom [study].skills relative and external directories."""
    ws = tmp_path / "workspace"
    ws.mkdir()
    if use_external_abs:
        external_dir = tmp_path / "external_catalog"
        write_skill("ext-skill", root=external_dir)
        write_reach_toml(f'[study]\nskills = "{external_dir}"\n', directory=ws)
        expected_label = f"1 in {external_dir}/"
    else:
        write_skill("custom-skill", root=ws / "custom_catalog")
        write_reach_toml('[study]\nskills = "custom_catalog"\n', directory=ws)
        expected_label = "1 in custom_catalog/"

    res = _check_skills(ws)
    assert res.status is CheckStatus.OK
    assert expected_label in res.detail


def test_check_skills_resolves_relative_workdir_labels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    write_skill: Callable[..., Path],
) -> None:
    """Verify _check_skills formats relative labels when passed a relative workdir Path."""
    write_skill("rel-skill", root=tmp_path / ".agents" / "skills")
    sub = tmp_path / "nested"
    sub.mkdir()
    monkeypatch.chdir(sub)

    res = _check_skills(Path(".."))
    assert res.status is CheckStatus.OK
    assert res.detail == "Found 1 skill(s): 1 in .agents/skills/"


def test_check_skills_warns_on_empty_existing_dir(tmp_path: Path) -> None:
    """Verify _check_skills warns when standard skill directories exist but contain no skills."""
    (tmp_path / ".agents" / "skills").mkdir(parents=True)
    res = _check_skills(tmp_path)
    assert res.status is CheckStatus.WARN
    assert ".github/skills" in res.detail


@pytest.mark.parametrize(
    ("has_global_skill", "expected_status", "expected_sub"),
    [
        (True, CheckStatus.OK, "1 in ~/.agents/skills/"),
        (False, CheckStatus.WARN, "No global skills detected"),
    ],
    ids=["populated_global", "empty_global"],
)
def test_check_skills_global_scope(
    tmp_path: Path,
    doctor_env: Path,
    write_skill: Callable[..., Path],
    has_global_skill: bool,
    expected_status: CheckStatus,
    expected_sub: str,
) -> None:
    """Verify _check_skills inspects user home directories when global_scope=True."""
    if has_global_skill:
        write_skill("global-helper", root=doctor_env / ".agents" / "skills")
    res = _check_skills(tmp_path, global_scope=True)
    assert res.name == "Global Skill Directories"
    assert res.status is expected_status
    assert expected_sub in res.detail


@pytest.mark.parametrize(
    ("toml_content", "expected_status", "expected_detail_sub"),
    [
        (
            '[runtime]\nagent = "keyword"\n',
            CheckStatus.OK,
            "default agent: 'keyword'",
        ),
        (
            "invalid = toml [ broken",
            CheckStatus.FAIL,
            "Configuration error:",
        ),
        (
            "[catalog]\nsize = 0\n",
            CheckStatus.FAIL,
            "catalog.size:",
        ),
    ],
    ids=["valid_toml", "malformed_toml_syntax", "pydantic_schema_validation_error"],
)
def test_check_config_states(
    tmp_path: Path,
    write_reach_toml: Callable[..., Path],
    toml_content: str,
    expected_status: CheckStatus,
    expected_detail_sub: str,
) -> None:
    """Verify _check_config handles valid TOML, syntax errors, and concise Pydantic errors."""
    write_reach_toml(toml_content, directory=tmp_path)
    res = _check_config(tmp_path)
    assert res.category == CheckCategory.CONFIGURATION
    assert res.status is expected_status
    assert expected_detail_sub in res.detail
    assert "https://errors.pydantic.dev" not in res.detail


@pytest.mark.parametrize(
    ("scenario", "expected_status", "expected_sub"),
    [
        ("target_dir", CheckStatus.WARN, "target directory"),
        ("cwd", CheckStatus.WARN, "current directory"),
        ("explicit_config", CheckStatus.FAIL, "Configuration file not found"),
    ],
)
def test_check_config_missing_file_scenarios(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scenario: str,
    expected_status: CheckStatus,
    expected_sub: str,
) -> None:
    """Verify _check_config distinguishes CWD, target dir, and missing explicit --config."""
    empty_dir = tmp_path / "custom_workdir"
    empty_dir.mkdir()
    if scenario == "cwd":
        monkeypatch.chdir(empty_dir)
    cfg_arg = (empty_dir / "missing.toml") if scenario == "explicit_config" else None

    res = _check_config(empty_dir, config_path=cfg_arg)
    assert res.status is expected_status
    assert expected_sub in res.detail


# ==============================================================================
# 3. Views, Rich Escaping & Output Renderers (text, json, jsonl, csv)
# ==============================================================================


def test_render_doctor_table_escapes_brackets_and_filters_failure_remedies(
    make_console: Callable[..., tuple[Any, io.StringIO]],
    rendered: Callable[[io.StringIO], str],
) -> None:
    """Verify render_doctor_table preserves [registry] and filters non-verbose failure panels."""
    checks = [
        CheckResult(
            category=CheckCategory.CREDENTIALS,
            name="Agent Registry",
            status=CheckStatus.WARN,
            detail="ADC available (no default project configured)",
            remedy="Set $GOOGLE_CLOUD_PROJECT or [registry] project in reach.toml",
        ),
        CheckResult(
            category=CheckCategory.CONFIGURATION,
            name="reach.toml",
            status=CheckStatus.FAIL,
            detail="Invalid field [catalog.size]",
            remedy="Fix [catalog] in reach.toml",
        ),
    ]

    # Non-verbose with failure: shows FAIL remedy only + tip for additional recommendations
    console, buf = make_console(width=120)
    code = render_doctor_table(console, checks, verbose=False)
    out = rendered(buf)
    assert code == 1
    assert "Reach Diagnostics" in out
    assert "Invalid field [catalog.size]" in out
    assert "Fix [catalog] in reach.toml" in out
    assert "[registry] project in reach.toml" not in out
    assert "reach doctor --verbose" in out

    # Verbose: includes both WARN and FAIL remedies with literal [registry] preserved
    console_v, buf_v = make_console(width=120)
    code_v = render_doctor_table(console_v, checks, verbose=True)
    out_v = rendered(buf_v)
    assert code_v == 1
    assert "Set $GOOGLE_CLOUD_PROJECT or [registry] project in reach.toml" in out_v


def test_render_doctor_table_shows_verbose_tip_on_warnings_only(
    make_console: Callable[..., tuple[Any, io.StringIO]],
    rendered: Callable[[io.StringIO], str],
) -> None:
    """Verify render_doctor_table prints a dim --verbose tip when only warnings exist."""
    console, buf = make_console(width=120)
    legacy_rows = [
        ("Credentials", "GEMINI_API_KEY", "warn", "not set", "Set GEMINI_API_KEY"),
    ]
    assert render_doctor_table(console, legacy_rows, verbose=False) == 0
    out = rendered(buf)
    assert "Recommended Actions" not in out
    assert "Tip: Run 'reach doctor --verbose' to view recommended actions." in out


def test_render_doctor_serialization_formats() -> None:
    """Verify json, jsonl, and csv doctor format renderers produce valid structured output."""
    report = DoctorReport(
        checks=(
            CheckResult(
                category=CheckCategory.ENVIRONMENT,
                name="Python Version",
                status=CheckStatus.OK,
                detail="3.13.0",
            ),
            CheckResult(
                category=CheckCategory.CREDENTIALS,
                name="GEMINI_API_KEY",
                status=CheckStatus.WARN,
                detail="not set",
                remedy="Set GEMINI_API_KEY",
            ),
        ),
    )

    assert report.has_failures is False
    assert report.has_warnings is True

    # JSON
    json_data = json.loads(render_doctor_json(report))
    assert len(json_data["checks"]) == 2
    assert render_doctor(report, "json") == render_doctor_json(report)

    # JSONL
    jsonl_lines = [json.loads(line) for line in render_doctor_jsonl(report).strip().splitlines()]
    assert len(jsonl_lines) == 2
    assert jsonl_lines[0]["category"] == "Environment"
    assert render_doctor(report, "jsonl") == render_doctor_jsonl(report)

    # CSV
    csv_rows = list(csv.DictReader(io.StringIO(render_doctor_csv(report))))
    assert len(csv_rows) == 2
    assert csv_rows[1]["name"] == "GEMINI_API_KEY"
    assert csv_rows[1]["status"] == "warn"
    assert render_doctor(report, "csv") == render_doctor_csv(report)


# ==============================================================================
# 4. End-to-End CLI & Orchestration Tests
# ==============================================================================


def test_run_doctor_checks_includes_semantic_and_google_adc(
    tmp_path: Path,
    doctor_env: Path,
) -> None:
    """Verify run_doctor_checks includes Semantic Scoring, ADC, and Global Skills."""
    assert doctor_env.is_dir()
    results = run_doctor_checks(tmp_path, global_scope=True)
    names = [r.name for r in results]
    assert "Semantic Scoring (model2vec)" in names
    assert "Google Cloud ADC" in names
    assert "Skill Directories" in names
    assert "Global Skill Directories" in names


def test_doctor_cli_runs_cleanly_and_supports_positional_path(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    doctor_env: Path,
    wide: None,
) -> None:
    """Verify reach doctor runs diagnostics with --path, -p, and positional path."""
    _ = wide
    assert doctor_env.is_dir()
    assert main(["doctor", "--path", str(tmp_path)]) == 0
    captured = capsys.readouterr()
    out = captured.out + captured.err
    assert "Reach Diagnostics" in out
    assert "Python Version" in out
    assert "Semantic Scoring (model2vec)" in out
    assert "Claude Code" in out
    assert "Keyword" in out

    assert main(["doctor", str(tmp_path)]) == 0
    captured_pos = capsys.readouterr()
    assert "Reach Diagnostics" in (captured_pos.out + captured_pos.err)


def test_doctor_cli_quiet_suppresses_table_output(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    doctor_env: Path,
) -> None:
    """Verify reach doctor --quiet suppresses table output while returning status code."""
    assert doctor_env.is_dir()
    assert main(["doctor", "--quiet", "-p", str(tmp_path)]) == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_doctor_cli_verbose_includes_remedy_panel_with_literal_brackets(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    doctor_env: Path,
    monkeypatch: pytest.MonkeyPatch,
    wide: None,
) -> None:
    """Verify reach doctor --verbose renders Recommended Actions without stripping [registry]."""
    _ = wide
    assert doctor_env.is_dir()
    monkeypatch.setattr("reach.registry.is_adc_available", lambda: True)
    assert main(["doctor", "--verbose", "-p", str(tmp_path)]) == 0
    captured = capsys.readouterr()
    out = captured.out + captured.err
    assert "Recommended Actions" in out
    assert "[registry] project in reach.toml" in out


def test_doctor_cli_handles_corrupt_reach_toml_end_to_end_without_crashing(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    doctor_env: Path,
    write_reach_toml: Callable[..., Path],
) -> None:
    """Verify reach doctor exits 1 and renders table when reach.toml is malformed."""
    assert doctor_env.is_dir()
    write_reach_toml("invalid = toml [ broken", directory=tmp_path)
    assert main(["doctor", "--path", str(tmp_path)]) == 1
    captured = capsys.readouterr()
    out = captured.out + captured.err
    assert "Reach Diagnostics" in out
    assert "Configuration error:" in out
    assert "Recommended Actions" in out


@pytest.mark.parametrize("fmt", ["json", "jsonl", "csv"])
def test_doctor_cli_structured_formats_and_config_flag(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    doctor_env: Path,
    write_reach_toml: Callable[..., Path],
    fmt: str,
) -> None:
    """Verify reach doctor --format and --config flags emit structured output."""
    assert doctor_env.is_dir()
    custom_cfg = write_reach_toml(
        '[runtime]\nagent = "keyword"\n',
        filename="custom.toml",
        directory=tmp_path,
    )
    assert main(["doctor", "-p", str(tmp_path), "-c", str(custom_cfg), "--format", fmt]) == 0
    out = capsys.readouterr().out.strip()
    assert out
    if fmt == "json":
        parsed = json.loads(out)
        assert any(c["name"] == "custom.toml" and c["status"] == "ok" for c in parsed["checks"])
    elif fmt == "jsonl":
        rows = [json.loads(line) for line in out.splitlines()]
        assert any(r["name"] == "custom.toml" for r in rows)
    else:
        rows = list(csv.DictReader(io.StringIO(out)))
        assert any(r["name"] == "custom.toml" for r in rows)
