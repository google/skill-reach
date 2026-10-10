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

"""Encapsulate provider-specific API key mapping and environment synchronization."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import NoReturn

_PROVIDER_KEY_ENV_VARS: dict[str, tuple[str, ...]] = {
    "anthropic": ("ANTHROPIC_API_KEY",),
    "openai": ("OPENAI_API_KEY",),
    "google": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
}

#: Default ambient environment variables stripped from child agent subprocesses.
DEFAULT_BLOCKED_ENV_VARS: tuple[str, ...] = (
    "AWS_ACCESS_KEY_ID",
    "AWS_PROFILE",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SECURITY_TOKEN",
    "AWS_SESSION_TOKEN",
    "DISCORD_TOKEN",
    "GH_TOKEN",
    "GITHUB_PAT",
    "GITHUB_TOKEN",
    "GOOGLE_APPLICATION_CREDENTIALS",
    "KUBECONFIG",
    "NPM_TOKEN",
    "PYPI_API_TOKEN",
    "SLACK_BOT_TOKEN",
    "SSH_AGENT_PID",
    "SSH_AUTH_SOCK",
    "SSH_KEY",
    "STRIPE_API_KEY",
)
_DEFAULT_BLOCKED_SET = frozenset(DEFAULT_BLOCKED_ENV_VARS)


_AGY_VERTEX_ENV_VARS: tuple[str, ...] = (
    "GOOGLE_GENAI_USE_ENTERPRISE",
    "GOOGLE_GENAI_USE_VERTEXAI",
    "AGY_ADC_AUTH",
)


_ALL_PROVIDER_KEY_ENV_VARS: tuple[str, ...] = tuple(
    dict.fromkeys(var for env_vars in _PROVIDER_KEY_ENV_VARS.values() for var in env_vars)
)


def resolve_env_secret(
    env: Mapping[str, str],
    *var_names: str,
    max_bytes: int = 16384,
) -> str | None:
    """Resolve a secret value from direct environment variables or ``*_FILE`` paths."""
    for name in var_names:
        if val := env.get(name):
            return val
    for name in var_names:
        if file_path := env.get(f"{name}_FILE"):
            try:
                path = Path(file_path).expanduser().resolve()
                if not path.is_file():
                    continue
                with path.open("rb") as fh:
                    content = fh.read(max_bytes).decode("utf-8").strip()
                if content:
                    return content
            except (OSError, UnicodeDecodeError):
                continue
    return None


def is_truthy_env(env: Mapping[str, str], key: str) -> bool:
    """Return True if the specified environment variable is set to 'true', '1', or 'yes'."""
    return env.get(key, "").strip().lower() in ("true", "1", "yes")


def has_agy_vertex_env(env: Mapping[str, str]) -> bool:
    """Return True if any Antigravity Vertex/Enterprise/ADC environment variable is enabled."""
    if any(is_truthy_env(env, key) for key in _AGY_VERTEX_ENV_VARS):
        return True
    has_api_key = bool(resolve_env_secret(env, "GEMINI_API_KEY", "GOOGLE_API_KEY"))
    return bool(env.get("GOOGLE_CLOUD_PROJECT")) and not has_api_key


def sanitize_subprocess_env(
    env: dict[str, str],
    *,
    keep: Iterable[str] = (),
    blocked_env_vars: Iterable[str] | None = None,
) -> dict[str, str]:
    """Resolve ``*_FILE`` provider secrets and strip blocked ambient credentials."""
    keep_set = set(keep)
    vertex_env_active = env.get("CLAUDE_CODE_USE_VERTEX") == "1" or has_agy_vertex_env(env)
    if blocked_env_vars is None and vertex_env_active:
        keep_set.add("GOOGLE_APPLICATION_CREDENTIALS")

    effective_blocked = (
        set(blocked_env_vars) if blocked_env_vars is not None else _DEFAULT_BLOCKED_SET
    )
    for key in _ALL_PROVIDER_KEY_ENV_VARS:
        if (
            (key not in effective_blocked or key in keep_set)
            and not env.get(key)
            and (resolved := resolve_env_secret(env, key))
        ):
            env[key] = resolved
        env.pop(f"{key}_FILE", None)

    blocked_with_files = effective_blocked | {f"{var}_FILE" for var in effective_blocked}
    for key in list(env.keys()):
        if key in keep_set:
            continue
        if key in blocked_with_files:
            env.pop(key, None)
    return env


def sync_google_and_gemini_keys(env: dict[str, str]) -> dict[str, str]:
    """Synchronize GEMINI_API_KEY and GOOGLE_API_KEY bidirectionally in environment dict."""
    if (
        not env.get("GEMINI_API_KEY")
        and not env.get("GOOGLE_API_KEY")
        and (resolved := resolve_env_secret(env, "GEMINI_API_KEY", "GOOGLE_API_KEY"))
    ):
        env["GEMINI_API_KEY"] = resolved
        env["GOOGLE_API_KEY"] = resolved
    elif env.get("GEMINI_API_KEY") and not env.get("GOOGLE_API_KEY"):
        env["GOOGLE_API_KEY"] = env["GEMINI_API_KEY"]
    elif env.get("GOOGLE_API_KEY") and not env.get("GEMINI_API_KEY"):
        env["GEMINI_API_KEY"] = env["GOOGLE_API_KEY"]
    if env.get("GEMINI_API_KEY") or env.get("GOOGLE_API_KEY"):
        env.pop("GEMINI_API_KEY_FILE", None)
        env.pop("GOOGLE_API_KEY_FILE", None)
    return env


def _read_claude_settings_env(claude_home: Path | None = None) -> dict[str, object]:
    """Read the env mapping from Claude settings.json if present and valid."""
    settings_file = (claude_home or (Path.home() / ".claude")) / "settings.json"
    if not settings_file.is_file():
        return {}
    try:
        data = json.loads(settings_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    if isinstance(data, dict) and isinstance(data.get("env"), dict):
        return data["env"]
    return {}


def sync_claude_settings_env(
    env: dict[str, str],
    claude_home: Path | None = None,
    *,
    blocked_env_vars: Iterable[str] | None = None,
    vertex_override: bool | None = None,
) -> dict[str, str]:
    """Synchronize user Claude Code settings and Vertex defaults into environment dict.

    Args:
        env: Active environment variables dictionary to update in-place.
        claude_home: Optional path to Claude home directory (defaults to ~/.claude).
        blocked_env_vars: Sensitive environment variables blocked from being injected.
        vertex_override: Explicit boolean override for Vertex AI mode (True, False, or None).

    Returns:
        Updated environment dictionary.
    """
    effective_blocked = (
        set(blocked_env_vars) if blocked_env_vars is not None else _DEFAULT_BLOCKED_SET
    )
    if vertex_override is True:
        env["CLAUDE_CODE_USE_VERTEX"] = "1"
    elif vertex_override is False:
        env.pop("CLAUDE_CODE_USE_VERTEX", None)
        if "GOOGLE_APPLICATION_CREDENTIALS" in effective_blocked:
            env.pop("GOOGLE_APPLICATION_CREDENTIALS", None)

    settings_env = _read_claude_settings_env(claude_home)
    vertex_enabled = vertex_override is not False and (
        env.get("CLAUDE_CODE_USE_VERTEX") == "1"
        or str(settings_env.get("CLAUDE_CODE_USE_VERTEX", "")) == "1"
    )
    allow_vertex_creds = blocked_env_vars is None and vertex_enabled

    for key, val in settings_env.items():
        if vertex_override is False and key == "CLAUDE_CODE_USE_VERTEX":
            continue
        if key in effective_blocked and not (
            key == "GOOGLE_APPLICATION_CREDENTIALS" and allow_vertex_creds
        ):
            continue
        if isinstance(val, (str, int, float)):
            env[key] = str(val)

    if vertex_override is not False and env.get("CLAUDE_CODE_USE_VERTEX") == "1":
        if not env.get("CLOUD_ML_REGION"):
            env["CLOUD_ML_REGION"] = "global"
        creds_blocked = (
            "GOOGLE_APPLICATION_CREDENTIALS" in effective_blocked and not allow_vertex_creds
        )
        if (
            not creds_blocked
            and "GOOGLE_APPLICATION_CREDENTIALS" in os.environ
            and "GOOGLE_APPLICATION_CREDENTIALS" not in env
        ):
            env["GOOGLE_APPLICATION_CREDENTIALS"] = os.environ["GOOGLE_APPLICATION_CREDENTIALS"]

    return env


def apply_provider_api_key(
    env: dict[str, str],
    provider: str | None,
    api_key: str | None,
    default_provider: str = "google",
) -> dict[str, str]:
    """Map provider-specific API key into environment variables dictionary.

    Supported providers:
        - google / gemini: GEMINI_API_KEY, GOOGLE_API_KEY
        - anthropic: ANTHROPIC_API_KEY
        - openai: OPENAI_API_KEY

    When provider is None, the key is mapped according to default_provider.
    When provider is explicitly specified but unrecognized, raises ValueError
    to prevent unintended key leakage to other provider endpoints.
    """
    if not api_key:
        return env

    if provider is not None:
        prov = provider.strip().lower()
        for name, env_vars in _PROVIDER_KEY_ENV_VARS.items():
            if name in prov:
                for var in env_vars:
                    env[var] = str(api_key)
                return env

        supported = ", ".join(sorted(_PROVIDER_KEY_ENV_VARS))
        msg = (
            f"Unrecognized provider {provider!r} for automatic API key mapping. "
            f"Supported providers: {supported}. "
            "For custom or self-hosted providers, set required environment variables directly."
        )
        raise ValueError(msg)

    # Provider omitted: fall back to default_provider mapping
    default_prov = default_provider.strip().lower()
    fallback_vars = _PROVIDER_KEY_ENV_VARS.get(default_prov, ("OPENAI_API_KEY",))
    for var in fallback_vars:
        env[var] = str(api_key)
    return env


def raise_missing_agent_dependency(
    agent: str,
    err: ImportError,
    role: str = "runtime",
) -> NoReturn:
    """Raise actionable RuntimeError when an optional agent dependency is missing."""
    if agent == "antigravity-sdk":
        msg = (
            f"google-antigravity is required for the antigravity-sdk {role}. "
            "Install it with: pip install 'skill-reach[antigravity-sdk]'"
        )
        raise RuntimeError(msg) from err
    raise err


def detect_model_provider(
    model: str,
    explicit_provider: str | None = None,
    *,
    api_key: str | None = None,
    env: Mapping[str, str] | None = None,
) -> str | None:
    """Infer model provider from explicit setting, environment variables, or model prefix.

    Args:
        model: Identifier of the model.
        explicit_provider: Explicitly configured provider override, if any.
        api_key: Explicitly configured API key, if any.
        env: Optional environment mapping; defaults to os.environ.

    Returns:
        Canonical provider name ("gemini"), or None if unmapped.
    """
    if explicit_provider is not None:
        return explicit_provider
    active_env = os.environ if env is None else env
    has_gemini_key = bool(
        api_key or resolve_env_secret(active_env, "GEMINI_API_KEY", "GOOGLE_API_KEY")
    )
    if model.lower().startswith("gemini") and has_gemini_key:
        return "gemini"
    return None


def resolve_blocked_env_vars(
    options: object | None = None,
    settings: object | None = None,
) -> tuple[str, ...] | None:
    """Return blocked environment variables prioritizing options over settings."""
    if (blocked := getattr(options, "blocked_env_vars", None)) is not None:
        return tuple(blocked)
    if (blocked := getattr(settings, "blocked_env_vars", None)) is not None:
        return tuple(blocked)
    return None
