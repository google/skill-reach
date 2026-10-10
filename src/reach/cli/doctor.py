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

"""Diagnose development environment, runtime agent binaries, credentials, and skill catalogs."""

from __future__ import annotations

import importlib.util
import os
import platform
import shutil
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Final

from cyclopts import Parameter

from reach.catalog import load_skills
from reach.config import RunConfig
from reach.registry import find_adc_path
from reach.runtime._env import is_truthy_env
from reach.views import build_console, render_doctor_table

from .app import SETUP, app
from .flags import SWITCH

#: Number of version tuple components (major, minor, micro) in standard semver.
MIN_VERSION_COMPONENTS: Final = 3


@dataclass(frozen=True, slots=True)
class CheckResult:
    """Represent the diagnostic result of a single environment check."""

    category: str
    name: str
    status: str  # "ok", "warn", "fail"
    detail: str
    remedy: str = ""


def _check_python(version_info: tuple[int, ...] | None = None) -> CheckResult:
    """Verify Python runtime version compatibility."""
    v = version_info or sys.version_info
    version_str = f"{v[0]}.{v[1]}.{v[2]}" if len(v) >= MIN_VERSION_COMPONENTS else f"{v[0]}.{v[1]}"
    if (v[0], v[1]) >= (3, 12):
        return CheckResult(
            category="Python Environment",
            name="Python Version",
            status="ok",
            detail=f"{version_str} ({platform.python_implementation()} on {platform.system()})",
        )
    return CheckResult(
        category="Python Environment",
        name="Python Version",
        status="fail",
        detail=f"{version_str} (unsupported, requires >= 3.12)",
        remedy="Upgrade to Python 3.12 or newer via mise or pyenv",
    )


def _check_cli_binary(
    name: str,
    executable: str,
    required_by: str,
    *,
    alternates: Sequence[str] = (),
) -> CheckResult:
    """Check whether an external CLI agent executable exists in PATH."""
    path = next(
        (found for candidate in (executable, *alternates) if (found := shutil.which(candidate))),
        None,
    )
    if path is None:
        return CheckResult(
            category="Agent Runtime Drivers",
            name=name,
            status="warn",
            detail=f"executable '{executable}' not found in PATH",
            remedy=f"Install {name} CLI to enable --agent {required_by}",
        )

    return CheckResult(
        category="Agent Runtime Drivers",
        name=name,
        status="ok",
        detail=path,
    )


def _check_sdk(name: str, module_name: str, required_by: str) -> CheckResult:
    """Check whether a Python SDK dependency is importable."""
    try:
        spec = importlib.util.find_spec(module_name)
    except (ModuleNotFoundError, ValueError):
        spec = None

    if spec is None:
        return CheckResult(
            category="Agent Runtime Drivers",
            name=name,
            status="warn",
            detail=f"Python package '{module_name}' not installed",
            remedy=f"Install with: uv add skill-reach[{required_by}]",
        )
    return CheckResult(
        category="Agent Runtime Drivers",
        name=name,
        status="ok",
        detail="installed and importable",
    )


def _check_env_var(var_name: str, purpose: str) -> CheckResult:
    """Check if an environment variable is configured in the current shell."""
    val = os.environ.get(var_name)
    if val:
        return CheckResult(
            category="Credentials & Environment",
            name=var_name,
            status="ok",
            detail="configured",
        )
    return CheckResult(
        category="Credentials & Environment",
        name=var_name,
        status="warn",
        detail="not set",
        remedy=f"Set {var_name} to enable {purpose}",
    )


def _check_google_adc() -> CheckResult:
    """Check whether Google Cloud Application Default Credentials exist."""
    agy_adc = is_truthy_env(os.environ, "AGY_ADC_AUTH")
    agy_suffix = " (AGY_ADC_AUTH enabled)" if agy_adc else ""
    custom = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if custom and Path(custom).is_file():
        return CheckResult(
            category="Credentials & Environment",
            name="Google Cloud ADC",
            status="ok",
            detail=f"configured via GOOGLE_APPLICATION_CREDENTIALS ({custom}){agy_suffix}",
        )
    adc_standard = find_adc_path()
    if adc_standard is not None and adc_standard.is_file():
        return CheckResult(
            category="Credentials & Environment",
            name="Google Cloud ADC",
            status="ok",
            detail=f"found at standard location ({adc_standard}){agy_suffix}",
        )
    return CheckResult(
        category="Credentials & Environment",
        name="Google Cloud ADC",
        status="warn",
        detail=f"not found{agy_suffix}",
        remedy=(
            "Run 'gcloud auth application-default login --project <PROJECT_ID>' if using "
            "Google Cloud Model Garden on Agent Platform"
        ),
    )


def _check_skills(workdir: Path) -> CheckResult:
    """Discover and count skills in standard workspace directories."""
    standard_dirs = [
        ".agents/skills",
        ".claude/skills",
        ".cursor/skills",
        ".github/skills",
        ".pi/skills",
        "skills",
    ]
    found_locations: list[str] = []
    total_skills = 0

    for rel_dir in standard_dirs:
        dir_path = workdir / rel_dir
        if dir_path.is_dir():
            skills = load_skills(dir_path)
            if skills:
                total_skills += len(skills)
                found_locations.append(f"{len(skills)} in {rel_dir}/")

    if total_skills > 0:
        return CheckResult(
            category="Skill Catalogs & Workspaces",
            name="Skill Directories",
            status="ok",
            detail=f"Found {total_skills} skill(s): {', '.join(found_locations)}",
        )
    return CheckResult(
        category="Skill Catalogs & Workspaces",
        name="Skill Directories",
        status="warn",
        detail=(
            "No skills detected in standard locations (.agents/skills, .claude/skills, "
            ".cursor/skills, .github/skills, .pi/skills, skills)"
        ),
        remedy="Run 'reach init' or place skill definitions with SKILL.md under .agents/skills/",
    )


def _check_config(workdir: Path) -> CheckResult:
    """Validate reach.toml configuration file syntax and schema."""
    config_path = workdir / "reach.toml"
    if not config_path.is_file():
        is_cwd = workdir.resolve() == Path.cwd().resolve()
        location_desc = "current directory" if is_cwd else f"target directory '{workdir}'"
        return CheckResult(
            category="Project Configuration",
            name="reach.toml",
            status="warn",
            detail=f"No reach.toml found in {location_desc} (using defaults)",
            remedy="Run 'reach init' to generate a tailored reach.toml",
        )

    try:
        config = RunConfig.from_toml(config_path)
        agent = config.runtime.agent
        return CheckResult(
            category="Project Configuration",
            name="reach.toml",
            status="ok",
            detail=f"Valid (default agent: '{agent}', catalog mode: '{config.catalog.mode}')",
        )
    except Exception as err:  # noqa: BLE001
        return CheckResult(
            category="Project Configuration",
            name="reach.toml",
            status="fail",
            detail=f"Configuration error: {err}",
            remedy="Check reach.toml syntax or regenerate with 'reach init --force'",
        )


def _check_agent_registry(workdir: Path) -> CheckResult:
    """Check Google Cloud Agent Registry configuration, ADC credentials, and cache."""
    from reach.config import resolve_registry_project
    from reach.registry import RegistryCacheManager, is_adc_available

    config_path = workdir / "reach.toml"
    project = resolve_registry_project(config_path=config_path if config_path.is_file() else None)
    has_adc = is_adc_available()

    cache_mgr = RegistryCacheManager.for_workdir(workdir)
    total_bytes, _ = cache_mgr.clean(dry_run=True)

    cache_info = f" ({total_bytes} bytes cached)" if total_bytes > 0 else ""

    if project and has_adc:
        return CheckResult(
            category="Credentials & Environment",
            name="Agent Registry",
            status="ok",
            detail=f"Project: {project}, ADC available{cache_info}",
        )
    if project and not has_adc:
        return CheckResult(
            category="Credentials & Environment",
            name="Agent Registry",
            status="warn",
            detail=f"Project: {project}, but ADC credentials not found",
            remedy="Run 'gcloud auth application-default login' to authorize Agent Registry access",
        )
    if has_adc and not project:
        return CheckResult(
            category="Credentials & Environment",
            name="Agent Registry",
            status="ok",
            detail=f"ADC available (no default project configured){cache_info}",
            remedy="Set $GOOGLE_CLOUD_PROJECT or [registry] project in reach.toml",
        )
    return CheckResult(
        category="Credentials & Environment",
        name="Agent Registry",
        status="warn",
        detail="Project not set and ADC credentials not found",
        remedy="Set $GOOGLE_CLOUD_PROJECT and run 'gcloud auth application-default login'",
    )


def run_doctor_checks(workdir: Path | None = None) -> list[CheckResult]:
    """Run all system, runtime, credential, and skill diagnostics."""
    root = workdir or Path.cwd()
    return [
        _check_python(),
        _check_cli_binary("Claude Code CLI", "claude", "claude-code"),
        _check_cli_binary(
            "Antigravity CLI",
            "agy",
            "antigravity-cli",
            alternates=("antigravity",),
        ),
        _check_sdk("Antigravity SDK", "google.antigravity", "antigravity-sdk"),
        _check_cli_binary("Goose CLI", "goose", "goose"),
        _check_cli_binary("Pi CLI", "pi", "pi"),
        CheckResult(
            category="Agent Runtime Drivers",
            name="Keyword Runtime (BM25)",
            status="ok",
            detail="built-in Python driver (always available)",
        ),
        _check_env_var("GEMINI_API_KEY", "Google Gemini model completions"),
        _check_google_adc(),
        _check_agent_registry(root),
        _check_skills(root),
        _check_config(root),
    ]


@app.command(name="doctor", group=SETUP)
def _doctor(
    path: Annotated[
        Path | None,
        Parameter(
            help="Target project directory to inspect (defaults to current working directory)",
        ),
    ] = None,
    *,
    verbose: Annotated[
        bool,
        SWITCH,
        Parameter(
            alias="-v",
            help="Display detailed diagnostics and recommended remediation steps",
        ),
    ] = False,
) -> int:
    """Inspect local development environment, runtime agent binaries, keys, and skill catalogs."""
    console = build_console()
    results = run_doctor_checks(workdir=path)
    table_items = [(res.category, res.name, res.status, res.detail, res.remedy) for res in results]
    return render_doctor_table(console, table_items, verbose=verbose)
