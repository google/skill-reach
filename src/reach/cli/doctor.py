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

import contextlib
import importlib.util
import os
import platform
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Final

from cyclopts import Parameter
from pydantic import ValidationError as PydanticValidationError

from reach.catalog import load_skills, parse_frontmatter
from reach.config import (
    BUNDLED_CONFIG_PATH,
    KNOWN_CLIENT_GLOBAL_SKILLS_DIRS,
    KNOWN_CLIENT_SKILLS_DIRS,
    RunConfig,
    _env_google_project,
    resolve_discovery_candidates,
    resolve_path,
)
from reach.registry import find_adc_path
from reach.runtime._env import _AGY_VERTEX_ENV_VARS, is_truthy_env
from reach.views import (
    CheckCategory,
    CheckResult,
    CheckStatus,
    DoctorReport,
    build_console,
    render_doctor,
    render_doctor_table,
)

from .app import SETUP, app
from .clean import _BYTES_PER_KB, _format_size
from .flags import SWITCH, Format, Global, Quiet

__all__ = [
    "CheckCategory",
    "CheckResult",
    "CheckStatus",
    "DoctorReport",
    "run_doctor_checks",
]

#: Number of version tuple components (major, minor, micro) in standard semver.
MIN_VERSION_COMPONENTS: Final = 3


def _resolve_effective_config(workdir: Path, config_path: Path | None) -> Path:
    """Return the resolved reach.toml path for workdir or explicit config_path."""
    if config_path is not None:
        return resolve_path(config_path)
    return workdir.expanduser() / "reach.toml"


def _check_python(version_info: tuple[int, ...] | None = None) -> CheckResult:
    """Verify Python runtime version compatibility."""
    v = version_info or sys.version_info
    version_str = f"{v[0]}.{v[1]}.{v[2]}" if len(v) >= MIN_VERSION_COMPONENTS else f"{v[0]}.{v[1]}"
    if (v[0], v[1]) >= (3, 12):
        return CheckResult(
            category=CheckCategory.ENVIRONMENT,
            name="Python Version",
            status=CheckStatus.OK,
            detail=f"{version_str} ({platform.python_implementation()} on {platform.system()})",
        )
    return CheckResult(
        category=CheckCategory.ENVIRONMENT,
        name="Python Version",
        status=CheckStatus.FAIL,
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
            category=CheckCategory.RUNTIMES,
            name=name,
            status=CheckStatus.WARN,
            detail=f"executable '{executable}' not found in PATH",
            remedy=f"Install {name} CLI to enable --agent {required_by}",
        )

    return CheckResult(
        category=CheckCategory.RUNTIMES,
        name=name,
        status=CheckStatus.OK,
        detail=path,
    )


def _check_sdk(
    name: str,
    module_name: str,
    required_by: str,
    *,
    category: CheckCategory = CheckCategory.RUNTIMES,
) -> CheckResult:
    """Check whether a Python SDK or optional dependency is importable."""
    try:
        spec = importlib.util.find_spec(module_name)
    except (ImportError, AttributeError, ValueError):
        spec = None

    if spec is None:
        return CheckResult(
            category=category,
            name=name,
            status=CheckStatus.WARN,
            detail=f"Python package '{module_name}' not installed",
            remedy=(
                f"Install with: uv add 'skill-reach[{required_by}]' "
                f"(or pip install 'skill-reach[{required_by}]')"
            ),
        )
    return CheckResult(
        category=category,
        name=name,
        status=CheckStatus.OK,
        detail="installed and importable",
    )


def _check_env_var(
    var_name: str,
    purpose: str,
    *,
    alternates: Sequence[str] = (),
) -> CheckResult:
    """Check if an environment variable or any of its fallback aliases is configured."""
    if os.environ.get(var_name):
        return CheckResult(
            category=CheckCategory.CREDENTIALS,
            name=var_name,
            status=CheckStatus.OK,
            detail="configured",
        )
    for alt in alternates:
        if os.environ.get(alt):
            return CheckResult(
                category=CheckCategory.CREDENTIALS,
                name=var_name,
                status=CheckStatus.OK,
                detail=f"configured via {alt}",
            )
    return CheckResult(
        category=CheckCategory.CREDENTIALS,
        name=var_name,
        status=CheckStatus.WARN,
        detail="not set",
        remedy=f"Set {var_name} to enable {purpose}",
    )


def _check_google_adc() -> CheckResult:
    """Check whether Google Cloud Application Default Credentials exist."""
    active_toggles = [key for key in _AGY_VERTEX_ENV_VARS if is_truthy_env(os.environ, key)]
    toggle_suffix = (
        f" ({', '.join(f'{k} enabled' for k in active_toggles)})" if active_toggles else ""
    )
    default_remedy = (
        "Run 'gcloud auth application-default login --project <PROJECT_ID>' if using "
        "Google Cloud Model Garden on Agent Platform"
    )

    custom = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if custom:
        custom_path = Path(custom).expanduser()
        if custom_path.is_file():
            return CheckResult(
                category=CheckCategory.CREDENTIALS,
                name="Google Cloud ADC",
                status=CheckStatus.OK,
                detail=(
                    f"configured via GOOGLE_APPLICATION_CREDENTIALS ({custom_path}){toggle_suffix}"
                ),
            )
        return CheckResult(
            category=CheckCategory.CREDENTIALS,
            name="Google Cloud ADC",
            status=CheckStatus.WARN,
            detail=(
                f"GOOGLE_APPLICATION_CREDENTIALS points to missing file ({custom}){toggle_suffix}"
            ),
            remedy=(
                "Verify the GOOGLE_APPLICATION_CREDENTIALS file path or unset it to use "
                "standard ADC"
            ),
        )

    adc_standard = find_adc_path()
    if adc_standard is not None and adc_standard.is_file():
        return CheckResult(
            category=CheckCategory.CREDENTIALS,
            name="Google Cloud ADC",
            status=CheckStatus.OK,
            detail=f"found at standard location ({adc_standard}){toggle_suffix}",
        )
    return CheckResult(
        category=CheckCategory.CREDENTIALS,
        name="Google Cloud ADC",
        status=CheckStatus.WARN,
        detail=f"not found{toggle_suffix}",
        remedy=default_remedy,
    )


def _format_skill_location_label(cand: Path, workdir: Path, *, global_scope: bool) -> str:
    """Format a human-readable relative path label for a discovered skill directory."""
    base = Path.home().expanduser() if global_scope else workdir.expanduser()
    prefix = "~/" if global_scope else ""
    for cand_p, base_p in (
        (cand.expanduser(), base),
        (cand.expanduser().resolve(), base.resolve()),
    ):
        with contextlib.suppress(ValueError):
            rel = cand_p.relative_to(base_p)
            return f"{prefix}{rel}/"
    return f"{cand}/"


def _collect_skill_candidates(
    base_workdir: Path,
    effective_config: Path,
    *,
    global_scope: bool,
) -> list[Path]:
    """Assemble candidate skill directories from reach.toml and default precedence."""
    discovery_config = effective_config if effective_config.is_file() else BUNDLED_CONFIG_PATH
    candidates: list[Path] = []
    if not global_scope and effective_config.is_file():
        with contextlib.suppress(ValueError, OSError):
            cfg = RunConfig.from_toml(effective_config)
            if cfg.study.skills is not None:
                study_dir = (
                    cfg.study.skills
                    if cfg.study.skills.is_absolute()
                    else (base_workdir / cfg.study.skills)
                )
                candidates.append(study_dir)

    try:
        candidates.extend(
            resolve_discovery_candidates(
                workdir=base_workdir,
                config_path=discovery_config,
                global_scope=global_scope,
            ),
        )
    except (ValueError, OSError):
        candidates.extend(
            resolve_discovery_candidates(
                workdir=base_workdir,
                config_path=BUNDLED_CONFIG_PATH,
                global_scope=global_scope,
            ),
        )
    return candidates


def _check_skills(
    workdir: Path,
    *,
    config_path: Path | None = None,
    global_scope: bool = False,
) -> CheckResult:
    """Discover and count skills across configured workspace or global directories."""
    base_workdir = workdir.expanduser()
    effective_config = _resolve_effective_config(base_workdir, config_path)
    candidates = _collect_skill_candidates(
        base_workdir,
        effective_config,
        global_scope=global_scope,
    )

    resolved_workdir = base_workdir.resolve()
    seen_resolved: set[Path] = set()
    seen_skill_paths: set[Path] = set()
    found_locations: list[str] = []
    total_skills = 0

    for cand in candidates:
        if not cand.is_dir():
            continue
        canonical = cand.resolve()
        if canonical in seen_resolved:
            continue

        if not global_scope and canonical == resolved_workdir:
            manifest = cand / "SKILL.md"
            if manifest.is_file():
                with contextlib.suppress(OSError, ValueError):
                    skill = parse_frontmatter(manifest.read_text(encoding="utf-8"), manifest)
                    if skill is not None:
                        seen_resolved.add(canonical)
                        seen_skill_paths.add(manifest.resolve())
                        total_skills += 1
                        found_locations.append("1 in ./")
            continue

        seen_resolved.add(canonical)
        skills = load_skills(cand)
        unique_count = 0
        for s in skills:
            resolved_skill = s.path.resolve() if s.path is not None else (canonical / s.name)
            if resolved_skill in seen_skill_paths:
                continue
            seen_skill_paths.add(resolved_skill)
            unique_count += 1
        if unique_count > 0:
            total_skills += unique_count
            label = _format_skill_location_label(cand, base_workdir, global_scope=global_scope)
            found_locations.append(f"{unique_count} in {label}")

    check_name = "Global Skill Directories" if global_scope else "Skill Directories"
    if total_skills > 0:
        return CheckResult(
            category=CheckCategory.SKILLS,
            name=check_name,
            status=CheckStatus.OK,
            detail=f"Found {total_skills} skill(s): {', '.join(found_locations)}",
        )

    if global_scope:
        global_dirs = ", ".join(
            sorted({f"~/{d}" for dirs in KNOWN_CLIENT_GLOBAL_SKILLS_DIRS.values() for d in dirs}),
        )
        return CheckResult(
            category=CheckCategory.SKILLS,
            name=check_name,
            status=CheckStatus.WARN,
            detail=f"No global skills detected in user home directory ({global_dirs})",
            remedy="Place global skill definitions with SKILL.md under ~/.agents/skills/",
        )

    local_dirs = ", ".join(sorted({*KNOWN_CLIENT_SKILLS_DIRS.values(), "skills"}))
    return CheckResult(
        category=CheckCategory.SKILLS,
        name=check_name,
        status=CheckStatus.WARN,
        detail=f"No skills detected in standard locations ({local_dirs})",
        remedy="Run 'reach init' or place skill definitions with SKILL.md under .agents/skills/",
    )


def _check_config(workdir: Path, config_path: Path | None = None) -> CheckResult:
    """Validate reach.toml configuration file syntax and schema."""
    base_workdir = workdir.expanduser()
    target_path = _resolve_effective_config(base_workdir, config_path)
    config_label = config_path.name if config_path is not None else "reach.toml"

    if not target_path.is_file():
        if config_path is not None:
            return CheckResult(
                category=CheckCategory.CONFIGURATION,
                name=config_label,
                status=CheckStatus.FAIL,
                detail=f"Configuration file not found: '{target_path}'",
                remedy="Pass a valid path to --config or run 'reach init'",
            )
        is_cwd = base_workdir.resolve() == Path.cwd().resolve()
        location_desc = "current directory" if is_cwd else f"target directory '{workdir}'"
        return CheckResult(
            category=CheckCategory.CONFIGURATION,
            name="reach.toml",
            status=CheckStatus.WARN,
            detail=f"No reach.toml found in {location_desc} (using defaults)",
            remedy="Run 'reach init' to generate a tailored reach.toml",
        )

    try:
        config = RunConfig.from_toml(target_path)
        agent = config.runtime.agent
        return CheckResult(
            category=CheckCategory.CONFIGURATION,
            name=config_label,
            status=CheckStatus.OK,
            detail=f"Valid (default agent: '{agent}', catalog mode: '{config.catalog.mode}')",
        )
    except PydanticValidationError as err:
        issues = "; ".join(
            f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in err.errors()
        )
        return CheckResult(
            category=CheckCategory.CONFIGURATION,
            name=config_label,
            status=CheckStatus.FAIL,
            detail=f"Configuration error: {issues}",
            remedy="Check reach.toml schema or regenerate with 'reach init --force'",
        )
    except (ValueError, OSError) as err:
        return CheckResult(
            category=CheckCategory.CONFIGURATION,
            name=config_label,
            status=CheckStatus.FAIL,
            detail=f"Configuration error: {err}",
            remedy="Check reach.toml syntax or regenerate with 'reach init --force'",
        )


def _check_agent_registry(workdir: Path, config_path: Path | None = None) -> CheckResult:
    """Check Google Cloud Agent Registry configuration, ADC credentials, and cache."""
    from reach.config import resolve_registry_project
    from reach.registry import RegistryCacheManager, is_adc_available

    base_workdir = workdir.expanduser()
    effective_config = _resolve_effective_config(base_workdir, config_path)
    discovery_config = effective_config if effective_config.is_file() else BUNDLED_CONFIG_PATH
    try:
        project = resolve_registry_project(config_path=discovery_config)
    except (ValueError, OSError):
        project = _env_google_project()

    has_adc = is_adc_available()

    cache_mgr = RegistryCacheManager.for_workdir(base_workdir)
    total_bytes, _ = cache_mgr.clean(dry_run=True)

    if total_bytes <= 0:
        cache_info = ""
    elif total_bytes < _BYTES_PER_KB:
        cache_info = f" ({total_bytes} bytes cached)"
    else:
        cache_info = f" ({_format_size(total_bytes)} cached)"

    if project and has_adc:
        return CheckResult(
            category=CheckCategory.CREDENTIALS,
            name="Agent Registry",
            status=CheckStatus.OK,
            detail=f"Project: {project}, ADC available{cache_info}",
        )
    if project and not has_adc:
        return CheckResult(
            category=CheckCategory.CREDENTIALS,
            name="Agent Registry",
            status=CheckStatus.WARN,
            detail=f"Project: {project}, but ADC credentials not found",
            remedy="Run 'gcloud auth application-default login' to authorize Agent Registry access",
        )
    if has_adc and not project:
        return CheckResult(
            category=CheckCategory.CREDENTIALS,
            name="Agent Registry",
            status=CheckStatus.OK,
            detail=f"ADC available (no default project configured){cache_info}",
            remedy="Set $GOOGLE_CLOUD_PROJECT or [registry] project in reach.toml",
        )
    return CheckResult(
        category=CheckCategory.CREDENTIALS,
        name="Agent Registry",
        status=CheckStatus.WARN,
        detail="Project not set and ADC credentials not found",
        remedy="Set $GOOGLE_CLOUD_PROJECT and run 'gcloud auth application-default login'",
    )


def run_doctor_checks(
    workdir: Path | None = None,
    *,
    config_path: Path | None = None,
    global_scope: bool = False,
) -> list[CheckResult]:
    """Run all system, runtime, credential, and skill diagnostics."""
    root = workdir.expanduser() if workdir is not None else Path.cwd()
    checks: list[CheckResult] = [
        _check_python(),
        _check_sdk(
            "Semantic Scoring (model2vec)",
            "model2vec",
            "semantic",
            category=CheckCategory.ENVIRONMENT,
        ),
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
            category=CheckCategory.RUNTIMES,
            name="Keyword Runtime (BM25)",
            status=CheckStatus.OK,
            detail="built-in Python driver (always available)",
        ),
        _check_env_var(
            "GEMINI_API_KEY",
            "Google Gemini model completions",
            alternates=("GOOGLE_API_KEY",),
        ),
        _check_google_adc(),
        _check_agent_registry(root, config_path=config_path),
        _check_skills(root, config_path=config_path, global_scope=False),
    ]
    if global_scope:
        checks.append(_check_skills(root, config_path=config_path, global_scope=True))
    checks.append(_check_config(root, config_path=config_path))
    return checks


@app.command(name="doctor", group=SETUP)
def _doctor(
    path: Annotated[
        Path | None,
        Parameter(
            alias="-p",
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
    quiet: Quiet = False,
    global_: Global = False,
    format: Annotated[
        Format,
        Parameter(
            help="Output format: text, json, jsonl, csv",
        ),
    ] = "text",
    config: Annotated[
        Path | None,
        Parameter(
            alias="-c",
            help="Path to reach.toml configuration file",
        ),
    ] = None,
) -> int:
    """Inspect local development environment, runtime agent binaries, keys, and skill catalogs."""
    results = run_doctor_checks(workdir=path, config_path=config, global_scope=global_)
    report = DoctorReport(checks=tuple(results))
    if format != "text":
        sys.stdout.write(f"{render_doctor(report, format).rstrip()}\n")
        return 1 if report.has_failures else 0
    console = build_console(quiet=quiet)
    return render_doctor_table(console, report, verbose=verbose)
