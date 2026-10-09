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

"""Define configuration models, path expansion, and deterministic digest generation."""

from __future__ import annotations

import hashlib
import os
import tomllib
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from enum import StrEnum
from functools import cached_property
from pathlib import Path
from typing import Annotated, Any, ClassVar, NamedTuple, Self, cast

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    field_validator,
    model_validator,
)

from reach.models import CatalogMode
from reach.uncertainty import DEFAULT_CONFIDENCE, DEFAULT_POWER

__all__ = [
    "DEFAULT_CATALOG_BUDGET_CHARS",
    "QUICK_ATTEMPTS",
    "AgentProfile",
    "CatalogSettings",
    "CheckSettings",
    "DiffSettings",
    "Digests",
    "DiscoverySettings",
    "GeneralSettings",
    "LintSettings",
    "OptimizeSettings",
    "OverlapSettings",
    "PlanSettings",
    "QuerySettings",
    "RegistrySettings",
    "RetrievalSettings",
    "RunConfig",
    "RuntimeSettings",
    "StudySettings",
    "agent_default_model",
    "agent_profiles",
    "default_agent",
    "digest_material",
    "load_config",
    "resolve_registry_location",
    "resolve_registry_project",
    "resolve_sub_settings",
]

#: Default probe attempts per query.
DEFAULT_ATTEMPTS = 5

#: Default probe attempts per query in quick evaluation mode.
QUICK_ATTEMPTS = 3

#: Canonical identifier for the deterministic offline keyword/BM25 runtime agent.
KEYWORD_AGENT = "keyword"

#: Default resident listing budget in characters before truncation occurs in rationing runtimes.
DEFAULT_CATALOG_BUDGET_CHARS = 30_000

#: Default Gemini model family identifier.
DEFAULT_GEMINI_MODEL: str = "gemini-3.8-flash"

#: Default Claude model family identifier.
DEFAULT_CLAUDE_MODEL: str = "claude-sonnet-5"

#: Default model mappings for builtin agent runtimes.
BUILTIN_AGENT_DEFAULT_MODELS: dict[str, str] = {
    "antigravity-cli": DEFAULT_GEMINI_MODEL,
    "antigravity-sdk": DEFAULT_GEMINI_MODEL,
    "claude-code": DEFAULT_CLAUDE_MODEL,
    "goose": DEFAULT_GEMINI_MODEL,
    "pi": DEFAULT_GEMINI_MODEL,
}

_ANTIGRAVITY_AGENTS: frozenset[str] = frozenset({"antigravity-cli", "antigravity-sdk"})


def expand_path(value: object, base: Path | None = None) -> Path:
    """Expand env vars and user home shortcuts in path string, anchoring to base."""
    expanded = os.path.expandvars(str(value))
    if "$" in expanded:
        msg = f"unset environment variable in path: {value}"
        raise ValueError(msg)
    path = Path(expanded).expanduser()
    if base is not None and not path.is_absolute():
        path = base / path
    return path


def resolve_path(path: Path | str) -> Path:
    """Expand user home directory and resolve to a canonical absolute filesystem path."""
    return Path(path).expanduser().resolve()


def reanchor(recorded: Path, root: Path) -> Path:
    """Resolve a recorded relative path against an alternative root directory."""
    parts = recorded.parts
    for start in range(1, len(parts)):
        candidate = root.joinpath(*parts[start:])
        if candidate.exists():
            return candidate
    return recorded


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge two dictionaries."""
    merged = dict(base)
    for k, v in overlay.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = _deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged


BUNDLED_CONFIG_PATH = Path(__file__).resolve().parent / "reach.toml"


def _discover_config_path(
    config: Path | str | None = None,
) -> tuple[Path | None, bool]:
    """Resolve explicit config path or auto-discover project reach.toml in CWD.

    Returns:
        Tuple of (resolved_path, is_explicit).
    """
    if config is not None:
        target = Path(config).expanduser().resolve()
        if not target.is_file():
            msg = f"Configuration file not found: {target}"
            raise FileNotFoundError(msg)
        return target, True

    project_candidate = Path.cwd() / "reach.toml"
    if project_candidate.is_file() and project_candidate.resolve() != BUNDLED_CONFIG_PATH:
        return project_candidate.resolve(), False
    return None, False


def load_config(config_path: Path | str | None = None) -> dict[str, Any]:
    """Load configuration from bundled reach.toml overlaid with optional project config."""
    base: dict[str, Any] = {}
    if BUNDLED_CONFIG_PATH.is_file():
        base = tomllib.loads(BUNDLED_CONFIG_PATH.read_text(encoding="utf-8"))

    target, _ = _discover_config_path(config_path)
    if target is not None:
        override = tomllib.loads(target.read_text(encoding="utf-8"))
        base = _deep_merge(base, override)

    return base


class AgentProfile(BaseModel):
    """Configuration profile for a named agent runtime in reach.toml."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    default_model: str | None = None
    default_provider: str | None = None
    skills_dir: str | None = None
    user_skills_dir: str | None = None
    models: tuple[str, ...] = ()


class GeneralSettings(BaseModel):
    """Global default settings from reach.toml."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    default_agent: str = "antigravity-cli"


DEFAULT_DISCOVERY_PRECEDENCE: tuple[str, ...] = (
    ".",
    "skills",
    ".agents/skills",
    "claude-code",
    "cursor",
    "github",
    "pi",
    "goose",
)


class DiscoverySettings(BaseModel):
    """Configuration settings for skill corpus discovery precedence."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    precedence: tuple[str, ...] = DEFAULT_DISCOVERY_PRECEDENCE


def default_agent(config_path: Path | str | None = None) -> str:
    """Return default agent runtime name from reach.toml configuration."""
    config = load_config(config_path)
    general = config.get("general", {})
    if isinstance(general, dict) and general:
        try:
            return GeneralSettings.model_validate(general).default_agent
        except ValidationError:
            pass
    return GeneralSettings().default_agent


def _load_section[T: BaseModel](
    section: str,
    model_cls: type[T],
    config_path: Path | str | None = None,
) -> T:
    """Load and validate a configuration section model from reach.toml."""
    raw = load_config(config_path).get(section, {})
    return model_cls.model_validate(raw) if isinstance(raw, dict) else model_cls()


def _env_google_project() -> str | None:
    """Return Google Cloud project ID from standard environment variables."""
    if env_proj := (
        os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GOOGLE_CLOUD_QUOTA_PROJECT")
    ):
        return env_proj.strip() or None
    return None


def _env_google_location() -> str:
    """Return Google Cloud location from standard environment variable or 'global'."""
    if env_loc := os.environ.get("GOOGLE_CLOUD_LOCATION"):
        return env_loc.strip() or "global"
    return "global"


def resolve_registry_project(
    cli_project: str | None = None,
    config_path: Path | str | None = None,
) -> str | None:
    """Resolve active Google Cloud project ID using 3-tier precedence.

    Precedence:
        1. Explicit CLI argument (--project)
        2. reach.toml [registry] project
        3. Environment variable ($GOOGLE_CLOUD_PROJECT or $GOOGLE_CLOUD_QUOTA_PROJECT)
    """
    if cli_project:
        return cli_project.strip()
    if from_toml := _load_section("registry", RegistrySettings, config_path).project:
        return from_toml.strip()
    return _env_google_project()


def resolve_registry_location(
    cli_location: str | None = None,
    config_path: Path | str | None = None,
) -> str:
    """Resolve active Agent Registry location using 3-tier precedence."""
    if cli_location:
        return cli_location.strip()
    target, _ = _discover_config_path(config_path)
    if target is not None and RunConfig.declared(target, "registry", "location"):
        reg_sec = _load_section("registry", RegistrySettings, target)
        if reg_sec.location:
            return reg_sec.location.strip()
    return _env_google_location()


def agent_profiles(config_path: Path | str | None = None) -> dict[str, AgentProfile]:
    """Return validated mapping of agent names to their configured AgentProfiles."""
    raw = load_config(config_path).get("agents", {})
    if not isinstance(raw, dict):
        return {}
    return {
        str(name): AgentProfile.model_validate(spec)
        for name, spec in raw.items()
        if isinstance(spec, dict)
    }


def agent_default_model(agent: str, config_path: Path | str | None = None) -> str | None:
    """Return default model identifier for the specified agent runtime."""
    profiles = agent_profiles(config_path)
    profile = profiles.get(agent)
    if profile is not None and profile.default_model:
        return profile.default_model
    return BUILTIN_AGENT_DEFAULT_MODELS.get(agent)


KNOWN_CLIENT_SKILLS_DIRS: dict[str, str] = {
    "agents": ".agents/skills",
    "antigravity-cli": ".agents/skills",
    "antigravity-sdk": ".agents/skills",
    "claude-code": ".claude/skills",
    "codex": ".agents/skills",
    "copilot": ".github/skills",
    "cursor": ".cursor/skills",
    "github": ".github/skills",
    "goose": ".agents/skills",
    "pi": ".pi/skills",
}

KNOWN_CLIENT_GLOBAL_SKILLS_DIRS: dict[str, tuple[str, ...]] = {
    "agents": (".agents/skills",),
    "antigravity-cli": (".agents/skills",),
    "antigravity-sdk": (".agents/skills",),
    "codex": (".agents/skills", ".codex/skills"),
    "claude-code": (".claude/skills",),
    "cursor": (".cursor/skills", ".agents/skills"),
    "github": (".copilot/skills", ".agents/skills"),
    "copilot": (".copilot/skills", ".agents/skills"),
    "goose": (".agents/skills",),
    "pi": (".pi/agent/skills", ".agents/skills"),
}


def _resolve_global_discovery_candidates(
    profiles: Mapping[str, AgentProfile],
    precedence: Sequence[str],
    agent: str | None,
    add_candidate: Callable[[Path], None],
) -> None:
    """Collect global candidate paths under user home directory."""
    home = Path.home()
    if agent:
        agent_prof = profiles.get(agent)
        if agent_prof is not None and (user_dir := agent_prof.user_skills_dir):
            add_candidate(home / user_dir)
        elif agent in KNOWN_CLIENT_GLOBAL_SKILLS_DIRS:
            for target in KNOWN_CLIENT_GLOBAL_SKILLS_DIRS[agent]:
                add_candidate(home / target)

    for item in precedence:
        if item in (".", "skills"):
            continue
        item_prof = profiles.get(item)
        if item_prof is not None and (user_dir := item_prof.user_skills_dir):
            add_candidate(home / user_dir)
        elif item in KNOWN_CLIENT_GLOBAL_SKILLS_DIRS:
            for target in KNOWN_CLIENT_GLOBAL_SKILLS_DIRS[item]:
                add_candidate(home / target)
        elif item in KNOWN_CLIENT_SKILLS_DIRS:
            add_candidate(home / KNOWN_CLIENT_SKILLS_DIRS[item])
        else:
            add_candidate(expand_path(item, base=home))


def _resolve_workspace_discovery_candidates(
    workdir: Path,
    profiles: Mapping[str, AgentProfile],
    precedence: Sequence[str],
    agent: str | None,
    add_candidate: Callable[[Path], None],
) -> None:
    """Collect candidate paths within a workspace directory."""
    agent_dir: str | None = None
    if agent and agent in profiles and profiles[agent].skills_dir:
        agent_dir = profiles[agent].skills_dir
    elif agent and agent in KNOWN_CLIENT_SKILLS_DIRS:
        agent_dir = KNOWN_CLIENT_SKILLS_DIRS[agent]

    for item in precedence:
        if item == ".":
            add_candidate(workdir)
        elif item == "skills":
            add_candidate(workdir / "skills")
            if agent_dir:
                add_candidate(workdir / agent_dir)
        elif item in profiles and (s_dir := profiles[item].skills_dir):
            add_candidate(workdir / s_dir)
        elif item in KNOWN_CLIENT_SKILLS_DIRS:
            add_candidate(workdir / KNOWN_CLIENT_SKILLS_DIRS[item])
        else:
            add_candidate(expand_path(item, base=workdir))


def resolve_discovery_candidates(
    workdir: Path | str,
    agent: str | None = None,
    config_path: Path | str | None = None,
    global_scope: bool = False,
) -> list[Path]:
    """Resolve ordered candidate paths for skill discovery within a workspace or globally.

    Args:
        workdir: The workspace directory being inspected.
        agent: Optional target agent runtime name to prioritize.
        config_path: Optional path to custom reach.toml configuration file.
        global_scope: If True, resolves candidate paths under user's home directory.

    Returns:
        Ordered list of candidate paths to probe for skills.
    """
    cfg = _load_section("discovery", DiscoverySettings, config_path)
    profiles = agent_profiles(config_path)

    candidates: list[Path] = []
    seen: set[Path] = set()

    def add_candidate(path: Path) -> None:
        canonical = path.resolve() if path.exists() else path
        if canonical not in seen:
            seen.add(canonical)
            candidates.append(path)

    if global_scope:
        _resolve_global_discovery_candidates(profiles, cfg.precedence, agent, add_candidate)
    else:
        resolved_workdir = resolve_path(workdir)
        _resolve_workspace_discovery_candidates(
            resolved_workdir, profiles, cfg.precedence, agent, add_candidate
        )

    return candidates


class CatalogSettings(BaseModel):
    """Configuration settings for catalog composition."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    mode: CatalogMode = CatalogMode.NEIGHBORHOOD
    size: int = Field(default=20, ge=2)
    rivals: int = Field(default=10, ge=1)
    seed: int = 0
    scorer: str = "hybrid"


class RuntimeSettings(BaseModel):
    """Configuration settings for agent runtime execution."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    agent: str = Field(default_factory=default_agent)
    allowed_tools: tuple[str, ...] | None = None
    blocked_env_vars: tuple[str, ...] | None = None
    options: dict[str, object] = Field(default_factory=dict)
    timeout_s: int = Field(default=200, gt=0)
    max_turns: int = Field(default=3, ge=1)
    early_exit: bool = True

    @model_validator(mode="after")
    def _options_match_the_agent(self) -> Self:
        """Reject options the chosen agent does not accept, at load time."""
        self.resolved_options()
        return self

    def resolved_options(self) -> dict[str, Any]:
        """Return this agent's options with its own defaults filled in."""
        from reach.runtime import resolve_options

        resolved = resolve_options(self)
        return resolved.model_dump(mode="json") if resolved is not None else {}

    def with_overrides(self, **overrides: object) -> RuntimeSettings:
        """Return a copy with overrides applied, discarding stale options when the agent changes."""
        options_specified = "options" in overrides
        clean_overrides = {k: v for k, v in overrides.items() if v is not None}
        agent_specified = "agent" in clean_overrides
        target_agent = clean_overrides.get("agent", self.agent)
        same_agent = target_agent == self.agent
        model = clean_overrides.pop("model", None)

        if options_specified and overrides["options"] is None:
            base_opts: Any = {}
        else:
            base_opts = dict(self.options) if same_agent else {}
        options_override = clean_overrides.pop("options", None)
        if isinstance(options_override, Mapping):
            base_opts.update(options_override)
        elif options_override is not None:
            base_opts = options_override

        if model is not None and not isinstance(base_opts, dict):
            opts_type = type(base_opts).__name__
            msg = f"Cannot set 'model' override when options is of type {opts_type}"
            raise ValueError(msg)
        if model is not None:
            base_opts["model"] = model

        payload: dict[str, Any] = {
            **self.model_dump(exclude_unset=True),
            **clean_overrides,
            "options": base_opts,
        }
        if agent_specified or "agent" in self.model_fields_set:
            payload["agent"] = target_agent
        else:
            payload.pop("agent", None)
        return type(self).model_validate(payload)

    @classmethod
    def resolve_for_optimize(
        cls,
        config: Path | str | None = None,
        *,
        agent: str | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> RuntimeSettings:
        """Resolve RuntimeSettings merging reach.toml [runtime.options] with overrides."""
        raw_cfg = load_config(config)
        raw_runtime = raw_cfg.get("runtime", {})
        cfg_agent = (
            str(raw_runtime.get("agent"))
            if isinstance(raw_runtime, Mapping) and raw_runtime.get("agent")
            else default_agent(config)
        )
        if isinstance(raw_runtime, Mapping) and isinstance(raw_runtime.get("options"), Mapping):
            raw_opts = dict(raw_runtime["options"])
            try:
                base_runtime = cls(agent=cfg_agent, options=raw_opts)
            except ValueError:
                base_runtime = cls(agent=cfg_agent)
        else:
            base_runtime = cls(agent=cfg_agent)
        overrides: dict[str, Any] = {}
        if agent is not None:
            overrides["agent"] = agent
        if options is not None:
            overrides["options"] = dict(options)
        return base_runtime.with_overrides(**overrides)


class Severity(StrEnum):
    """Specify the severity level for a lint diagnostic."""

    ERROR = "error"
    IGNORE = "ignore"
    INFO = "info"
    WARN = "warn"


def _normalize_severity(value: object) -> object:
    """Normalize string severity values to lowercase before enum validation."""
    return value.strip().lower() if isinstance(value, str) else value


type RuleSeverity = Annotated[Severity, BeforeValidator(_normalize_severity)]


class LintSettings(BaseModel):
    """Configuration settings for static skill linting and validation thresholds."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_description_length: Annotated[int, Field(ge=1)] = 1024
    max_name_length: Annotated[int, Field(ge=1)] = 64
    min_description_length: Annotated[int, Field(ge=1)] = 20
    catalog_budget_chars: Annotated[int, Field(ge=1)] | None = DEFAULT_CATALOG_BUDGET_CHARS
    similarity_threshold: Annotated[float, Field(ge=0.0, le=1.0)] = 0.92
    mutual_handoff_similarity_threshold: Annotated[float, Field(ge=0.0, le=1.0)] = 0.75
    mutual_handoff_lexical_threshold: Annotated[float, Field(ge=0.0, le=1.0)] = 0.35
    rules: dict[str, RuleSeverity] = Field(default_factory=dict)

    def with_rule_overrides(
        self,
        overrides: Mapping[str, Any] | None = None,
    ) -> LintSettings:
        """Return a validated copy of LintSettings with rule severity overrides applied."""
        if not overrides:
            return self
        merged_rules = {**self.rules, **overrides}
        validated_rules = (
            type(self).model_validate({**self.model_dump(), "rules": merged_rules}).rules
        )
        return self.model_copy(update={"rules": validated_rules})

    @classmethod
    def from_settings(
        cls,
        settings: Mapping[str, object] | None = None,
        overrides: Mapping[str, Any] | None = None,
    ) -> LintSettings:
        """Construct a LintSettings from loaded reach.toml settings and CLI overrides."""
        if settings is None:
            settings = load_config()

        lint_section = settings.get("lint", {}) if isinstance(settings, Mapping) else {}
        retrieval_section = settings.get("retrieval", {}) if isinstance(settings, Mapping) else {}
        catalog_section = settings.get("catalog", {}) if isinstance(settings, Mapping) else {}

        payload: dict[str, Any] = dict(lint_section) if isinstance(lint_section, Mapping) else {}
        raw_rules = payload.get("rules")
        rules: dict[str, Any] = dict(raw_rules) if isinstance(raw_rules, Mapping) else {}
        if overrides:
            rules.update(overrides)
        if rules or "rules" in payload:
            payload["rules"] = rules

        lint_base = cls.model_validate(payload)
        catalog_cfg = CatalogSettings.model_validate(
            catalog_section if isinstance(catalog_section, Mapping) else {}
        )
        retrieval_cfg = RetrievalSettings.model_validate(
            retrieval_section if isinstance(retrieval_section, Mapping) else {}
        )
        return _apply_lint_inheritance(lint_base, catalog_cfg, retrieval_cfg)


class CheckSettings(BaseModel):
    """Configuration settings for CI/CD regression checks and quality gates."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    min_recall: float = Field(default=0.80, ge=0.0, le=1.0)
    min_accuracy: float = Field(default=0.80, ge=0.0, le=1.0)
    max_misroute: float = Field(default=0.10, ge=0.0, le=1.0)
    min_entrypoint: float | None = Field(default=None, ge=0.0, le=1.0)
    min_reachability: float | None = Field(default=None, ge=0.0, le=1.0)
    min_efficiency: float | None = Field(default=None, ge=0.0, le=1.0)
    min_f1: float | None = Field(default=None, ge=0.0, le=1.0)
    max_redundancy: float | None = Field(default=None, ge=0.0)
    budget: int = Field(default=50, ge=1)
    strict: bool = True
    since: str = "HEAD~1"


class RetrievalSettings(BaseModel):
    """Configuration settings for dense, lexical, and hybrid retrieval."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    scorer: str = Field(default="hybrid")
    model: str = Field(default="minishlab/potion-retrieval-32M")
    rrf_k: int = Field(default=60, gt=0)
    similarity_threshold: float = Field(default=0.92, ge=0.0, le=1.0)
    bm25_k1: float = Field(default=1.5, gt=0.0)
    bm25_b: float = Field(default=0.75, ge=0.0, le=1.0)


def _apply_lint_inheritance(
    lint: LintSettings,
    catalog: CatalogSettings,
    retrieval: RetrievalSettings,
) -> LintSettings:
    """Propagate retrieval similarity threshold and scoped catalog mode into LintSettings."""
    original_fields_set = set(lint.model_fields_set)
    updates: dict[str, Any] = {}
    if (
        "similarity_threshold" not in original_fields_set
        and "similarity_threshold" in retrieval.model_fields_set
    ):
        updates["similarity_threshold"] = retrieval.similarity_threshold
    if (
        "catalog_budget_chars" not in original_fields_set
        and "mode" in catalog.model_fields_set
        and catalog.mode in {CatalogMode.NEIGHBORHOOD, CatalogMode.SINGLETON}
    ):
        updates["catalog_budget_chars"] = None
    if not updates:
        return lint
    return lint.model_copy(update=updates)


class OverlapSettings(BaseModel):
    """Configuration settings for lexical overlap and vocabulary rewrite heuristics."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    contender_band: Annotated[float, Field(ge=0.0, le=1.0)] = 0.90
    material_share: Annotated[float, Field(ge=0.0, le=1.0)] = 0.01
    claim_limit: Annotated[int, Field(ge=1)] = 8
    min_claim_length: Annotated[int, Field(ge=1)] = 3
    min_claim_uses: Annotated[int, Field(ge=1)] = 2


class DiffSettings(BaseModel):
    """Configuration settings for A/B evaluation diffing and noise floor calibration."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    confidence: float = Field(default=DEFAULT_CONFIDENCE, gt=0.0, lt=1.0)
    power: float = Field(default=DEFAULT_POWER, gt=0.0, lt=1.0)
    noise_inflation: float = Field(default=1.265, gt=0.0)

    @property
    def over_dispersion(self) -> float:
        """Derive the variance over-dispersion ratio from noise_inflation squared."""
        return self.noise_inflation**2


class QuerySettings(BaseModel):
    """Configuration settings for query synthesis, difficulty, and leakage detection."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    count: int = Field(default=3, ge=1)
    adversarial_count: int = Field(default=1, ge=0)
    top_rivals: int = Field(default=3, ge=0)
    distinctive_idf_floor: float = Field(default=0.693147, ge=0.0)


class OptimizeSettings(BaseModel):
    """Configuration settings for automated skill description optimization."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    budget: int = Field(default=30, ge=1)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    iterations: int = Field(default=1, ge=1, le=10)
    holdout: float = Field(default=0.2, ge=0.0, le=0.9)
    review: bool = Field(default=False)
    auto_queries: bool = Field(default=True)
    adversarial_count: int = Field(default=5, ge=0)
    positive_count: int = Field(default=5, ge=1)
    seed: int = Field(default=42)
    review_timeout: float = Field(default=600.0, gt=0.0)
    workers: int = Field(default=4, ge=1)
    with_handoff: bool = Field(default=False)


class RegistrySettings(BaseModel):
    """Configuration settings for Google Cloud Agent Registry."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    project: str | None = None
    location: str = "global"
    publisher: str | None = None
    cache_ttl_seconds: int = 300
    registry: bool = False
    fresh: bool = False
    no_cache: bool = False


class PlanSettings(BaseModel):
    """Configuration settings for probe planning and retry policies."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    attempts: int = Field(default=DEFAULT_ATTEMPTS, ge=1)
    retries: int = Field(default=2, ge=0)
    backoff_s: float = Field(default=5.0, ge=0)
    pause_s: float = Field(default=0.0, ge=0)
    workers: int = Field(default=1, ge=1)

    def resolve_sweep_attempts(self, cli_attempts: int | None = None) -> int:
        """Return effective sweep attempts, defaulting to 1 unless explicitly configured."""
        if cli_attempts is not None:
            if cli_attempts < 1:
                msg = f"Sweep attempts must be at least 1, got {cli_attempts}"
                raise ValueError(msg)
            return cli_attempts
        return self.attempts if "attempts" in self.model_fields_set else 1

    def resolve_eval_attempts(
        self,
        agent: str,
        cli_attempts: int | None = None,
        *,
        quick: bool = False,
    ) -> int:
        """Return effective eval attempts, defaulting to 1 for deterministic keyword runs."""
        if cli_attempts is not None:
            if cli_attempts < 1:
                msg = f"Eval attempts must be at least 1, got {cli_attempts}"
                raise ValueError(msg)
            return cli_attempts
        if "attempts" in self.model_fields_set:
            return self.attempts
        if agent == KEYWORD_AGENT:
            return 1
        return QUICK_ATTEMPTS if quick else self.attempts


class StudySettings(BaseModel):
    """Configuration settings for experiment inputs, workspaces, and outputs."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    auto_queries: bool = True
    skills: Path | None = None
    queries: Path | None = None
    workdir: Path | None = None
    out: Path | None = None
    tag: str = ""
    partial: bool = False
    catalog: str | None = "auto"
    rescope: bool = False
    scales: tuple[int, ...] | None = None
    anchor: int | tuple[str, ...] | str | None = None
    trusted: bool = False
    bootstrap_iterations: int = Field(default=200, ge=10)
    bootstrap_seed: int | None = Field(default=42)
    catalog_replicates: Annotated[int, Field(ge=1, le=20)] = 1

    @field_validator("anchor", mode="before")
    @classmethod
    def _coerce_anchor(cls, value: object) -> int | tuple[str, ...] | str | None:
        """Coerce anchor specification into integer count, skill names tuple, or 'all'."""
        if value is None:
            return None
        if isinstance(value, int):
            if value <= 0:
                msg = f"anchor size must be a positive integer, got {value}"
                raise ValueError(msg)
            return value
        raw_items: Sequence[object]
        if isinstance(value, str):
            val = value.strip()
            if not val:
                return None
            if val.lower() == "all":
                return "all"
            if val.isdigit():
                iv = int(val)
                if iv <= 0:
                    msg = f"anchor size must be a positive integer, got {iv}"
                    raise ValueError(msg)
                return iv
            raw_items = val.split(",")
        elif isinstance(value, (list, tuple)):
            raw_items = value
        else:
            msg = f"invalid anchor specification: {value}"
            raise ValueError(msg)
        return tuple(str(p).strip() for p in raw_items if str(p).strip())

    @field_validator("scales", mode="before")
    @classmethod
    def _coerce_scales(cls, value: object) -> tuple[int, ...] | None:
        """Coerce comma-separated string or sequence into tuple of positive integers."""
        if value is None:
            return None
        if isinstance(value, str):
            parts = [int(p.strip()) for p in value.split(",") if p.strip()]
            return tuple(parts)
        if isinstance(value, (list, tuple)):
            return tuple(int(v) for v in value)
        msg = f"invalid scales specification: {value}"
        raise ValueError(msg)

    @field_validator("skills", "queries", "workdir", "out", mode="before")
    @classmethod
    def _expand(cls, value: object) -> object:
        """Expand path variables and user directories for filesystem paths."""
        if value is None:
            return None
        return str(expand_path(value))

    def require_queries(self, hint: str = "") -> Path:
        """Return the configured queries path or raise a ValueError."""
        if self.queries is None:
            msg = f"queries path is required{f': {hint}' if hint else ''}"
            raise ValueError(msg)
        return self.queries

    def require_skills(self, hint: str = "") -> Path:
        """Return the configured skills path or raise a ValueError."""
        if self.skills is None:
            msg = f"skills path is required{f': {hint}' if hint else ''}"
            raise ValueError(msg)
        return self.skills

    def require_workdir(self, hint: str = "") -> Path:
        """Return the resolved configured workdir path or raise a ValueError."""
        if self.workdir is None:
            msg = f"workdir path is required{f': {hint}' if hint else ''}"
            raise ValueError(msg)
        return self.workdir.resolve()


_AGENT_REGISTRY_ADAPTER: TypeAdapter[dict[str, AgentProfile]] = TypeAdapter(dict[str, AgentProfile])


class RunConfig(BaseModel):
    """Encapsulate all parameters required to drive an evaluation run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    general: GeneralSettings = Field(default_factory=GeneralSettings)
    catalog: CatalogSettings = Field(default_factory=CatalogSettings)
    runtime: RuntimeSettings = Field(default_factory=RuntimeSettings)
    plan: PlanSettings = Field(default_factory=PlanSettings)
    study: StudySettings = Field(default_factory=StudySettings)
    lint: LintSettings = Field(default_factory=LintSettings)
    check: CheckSettings = Field(default_factory=CheckSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    discovery: DiscoverySettings = Field(default_factory=DiscoverySettings)
    overlap: OverlapSettings = Field(default_factory=OverlapSettings)
    diff: DiffSettings = Field(default_factory=DiffSettings)
    query: QuerySettings = Field(default_factory=QuerySettings)
    optimize: OptimizeSettings = Field(default_factory=OptimizeSettings)
    registry: RegistrySettings = Field(default_factory=RegistrySettings)

    @model_validator(mode="before")
    @classmethod
    def _strip_registry_tables(cls, data: object) -> object:
        """Validate and strip top-level agent and model registry tables merged by load_config()."""
        if isinstance(data, Mapping) and ("agents" in data or "models" in data):
            if "agents" in data:
                _AGENT_REGISTRY_ADAPTER.validate_python(data["agents"])
            if "models" in data:
                from reach.runtime.profiles import ModelProfile

                TypeAdapter(dict[str, ModelProfile]).validate_python(data["models"])
            return {k: v for k, v in data.items() if k not in {"agents", "models"}}
        return data

    @model_validator(mode="after")
    def _inherit_optimize_workers_from_plan(self) -> Self:
        """Inherit plan.workers into optimize.workers when optimize.workers is unset."""
        if (
            "workers" not in self.optimize.model_fields_set
            and "workers" in self.plan.model_fields_set
        ):
            object.__setattr__(
                self,
                "optimize",
                self.optimize.model_copy(update={"workers": self.plan.workers}),
            )
        return self

    @model_validator(mode="after")
    def _inherit_runtime_agent_from_general(self) -> Self:
        """Inherit general.default_agent into runtime.agent when runtime.agent is unset."""
        if "agent" not in self.runtime.model_fields_set and self.general.default_agent:
            object.__setattr__(
                self,
                "runtime",
                self.runtime.model_copy(update={"agent": self.general.default_agent}),
            )
        return self

    @model_validator(mode="after")
    def _inherit_lint_settings(self) -> Self:
        """Inherit retrieval threshold and scoped catalog mode into lint settings."""
        inherited = _apply_lint_inheritance(self.lint, self.catalog, self.retrieval)
        if inherited is not self.lint:
            object.__setattr__(self, "lint", inherited)
        return self

    def require_queries(self, hint: str = "") -> Path:
        """Forward queries path requirement to study settings."""
        return self.study.require_queries(hint)

    def require_skills(self, hint: str = "") -> Path:
        """Forward skills path requirement to study settings."""
        return self.study.require_skills(hint)

    def require_workdir(self, hint: str = "") -> Path:
        """Forward workdir path requirement to study settings."""
        return self.study.require_workdir(hint)

    _SECTION_MAP: ClassVar[dict[type[BaseModel], str]] = {
        GeneralSettings: "general",
        CatalogSettings: "catalog",
        RuntimeSettings: "runtime",
        PlanSettings: "plan",
        StudySettings: "study",
        LintSettings: "lint",
        CheckSettings: "check",
        RetrievalSettings: "retrieval",
        DiscoverySettings: "discovery",
        OverlapSettings: "overlap",
        DiffSettings: "diff",
        QuerySettings: "query",
        OptimizeSettings: "optimize",
        RegistrySettings: "registry",
    }

    def get_section[T: BaseModel](self, section_cls: type[T]) -> T:
        """Retrieve the configuration section corresponding to section_cls."""
        attr_name = self._SECTION_MAP.get(section_cls)
        if attr_name is not None and hasattr(self, attr_name):
            val = getattr(self, attr_name)
            if isinstance(val, section_cls):
                return val
        for attr in self.__dict__:
            val = getattr(self, attr)
            if isinstance(val, section_cls):
                return val
        return section_cls()

    def resolve_section[T: BaseModel](
        self,
        section_cls: type[T],
        *,
        explicit_settings: T | None = None,
        **overrides: object,
    ) -> T:
        """Resolve effective settings section against this RunConfig instance."""
        return self.resolve(
            section_cls,
            config=self,
            explicit_settings=explicit_settings,
            **overrides,
        )

    @classmethod
    def resolve[T: BaseModel](
        cls,
        section_cls: type[T],
        config: RunConfig | None = None,
        *,
        explicit_settings: T | None = None,
        **overrides: object,
    ) -> T:
        """Resolve effective settings section from an optional RunConfig or defaults."""
        section = config.get_section(section_cls) if config is not None else None
        active_overrides = dict(overrides)

        if issubclass(section_cls, RuntimeSettings):
            configured_runtime = section if isinstance(section, RuntimeSettings) else None
            base_runtime = (
                explicit_settings
                if isinstance(explicit_settings, RuntimeSettings)
                else (configured_runtime if configured_runtime is not None else RuntimeSettings())
            )
            resolved_runtime = base_runtime.with_overrides(**active_overrides)
            if config is not None:
                resolved_runtime = _propagate_registry_to_runtime(resolved_runtime, config.registry)
            return cast(T, resolved_runtime)

        resolved = resolve_sub_settings(
            section_cls,
            config_section=section,
            explicit_settings=explicit_settings,
            **active_overrides,
        )

        if isinstance(resolved, RegistrySettings):
            return cast(T, cls._resolve_registry_section(resolved, config))
        return cast(T, resolved)

    @staticmethod
    def _resolve_registry_section(
        resolved: RegistrySettings,
        config: RunConfig | None,
    ) -> RegistrySettings:
        """Resolve environment and TOML fallbacks for a RegistrySettings section."""
        if resolved.project:
            project: str | None = resolved.project.strip()
        elif config is not None:
            project = _env_google_project()
        else:
            project = resolve_registry_project(None)

        explicit_loc = resolved.location if "location" in resolved.model_fields_set else None
        if explicit_loc:
            location = explicit_loc.strip()
        elif config is not None:
            location = _env_google_location()
        else:
            location = resolve_registry_location(None)
        return resolved.model_copy(update={"project": project, "location": location})

    @classmethod
    def from_toml(
        cls,
        path: Path | str,
        *,
        skills: Path | None = None,
    ) -> RunConfig:
        """Load and validate a RunConfig from a TOML configuration file."""
        resolved = Path(path).expanduser().resolve()
        payload = tomllib.loads(resolved.read_text(encoding="utf-8"))
        study = payload.get("study", {})
        if skills is not None:
            study["skills"] = skills
        for key in ("skills", "queries", "workdir", "out"):
            raw = study.get(key)
            if raw is not None:
                study[key] = expand_path(raw, base=resolved.parent)
        payload["study"] = study
        return cls.model_validate(payload)

    @staticmethod
    def declared(path: Path | str, section: str, key: str) -> bool:
        """Check whether a setting key is explicitly declared in a TOML file."""
        payload = tomllib.loads(
            Path(path).expanduser().resolve().read_text(encoding="utf-8"),
        )
        return key in payload.get(section, {})

    def with_overrides(self, **sections: dict[str, Any]) -> RunConfig:
        """Return copy of configuration with section field overrides applied."""
        payload: dict[str, Any] = {}
        for section_name in type(self).model_fields:
            section_model = getattr(self, section_name)
            if isinstance(section_model, BaseModel):
                sec_dump = {
                    **section_model.model_dump(exclude_defaults=True),
                    **section_model.model_dump(exclude_unset=True),
                }
                if section_name == "runtime" and "agent" not in section_model.model_fields_set:
                    sec_dump.pop("agent", None)
                if section_name == "optimize" and "workers" not in section_model.model_fields_set:
                    sec_dump.pop("workers", None)
                if isinstance(section_model, LintSettings):
                    if (
                        self.catalog.mode in {CatalogMode.NEIGHBORHOOD, CatalogMode.SINGLETON}
                        and section_model.catalog_budget_chars is None
                    ):
                        sec_dump.pop("catalog_budget_chars", None)
                    if (
                        "similarity_threshold" in self.retrieval.model_fields_set
                        and section_model.similarity_threshold
                        == self.retrieval.similarity_threshold
                    ):
                        sec_dump.pop("similarity_threshold", None)
                payload[section_name] = sec_dump
            else:
                payload[section_name] = section_model

        for section, fields in sections.items():
            live = {k: v for k, v in (fields or {}).items() if v is not None}
            if section == "runtime":
                new_runtime = self.runtime.with_overrides(**live)
                sec_dump = {
                    **new_runtime.model_dump(exclude_defaults=True),
                    **new_runtime.model_dump(exclude_unset=True),
                }
                if "agent" not in new_runtime.model_fields_set:
                    sec_dump.pop("agent", None)
                payload["runtime"] = sec_dump
            else:
                payload[section] = {**payload.get(section, {}), **live}
        return type(self).model_validate(payload)

    @cached_property
    def _digests(self) -> Digests:
        """Compute and memoize fingerprint, arm, and condition digests."""
        return digest_material(self.material())

    @property
    def fingerprint(self) -> str:
        """Return deterministic configuration fingerprint."""
        return self._digests.fingerprint

    def material(self) -> dict[str, Any]:
        """Dump configuration dictionary with agent options resolved to defaults."""
        material = self.model_dump(mode="json")
        material["runtime"]["options"] = self.runtime.resolved_options()
        return material

    @property
    def arm(self) -> str:
        """Return digest representing experimental condition without study paths."""
        return self._digests.arm

    @property
    def condition(self) -> str:
        """Return arm digest excluding plan attempt count."""
        return self._digests.condition


class Digests(NamedTuple):
    """Container holding fingerprint, arm, and condition digests."""

    fingerprint: str
    arm: str
    condition: str


def digest_material(material: dict[str, Any]) -> Digests:
    """Compute fingerprint, arm, and condition digests from raw config material."""
    base = deepcopy(material)
    for key in ("lint", "check", "discovery"):
        base.pop(key, None)
    if "plan" in base:
        base["plan"].pop("workers", None)
    if "optimize" in base:
        base["optimize"].pop("workers", None)
        base["optimize"].pop("with_handoff", None)

    fingerprint = deepcopy(base)
    fingerprint["study"] = {
        k: v
        for k, v in fingerprint.get("study", {}).items()
        if k not in {"out", "workdir", "skills", "queries", "tag", "trusted", "auto_queries"}
    }

    arm = deepcopy(base)
    arm.pop("study", None)

    condition = deepcopy(arm)
    if "plan" in condition:
        condition["plan"].pop("attempts", None)

    return Digests(*(_digest(reduced) for reduced in (fingerprint, arm, condition)))


def _digest(material: dict[str, Any]) -> str:
    """Hash flattened configuration dictionary into 12-char SHA-256 digest."""
    canonical = repr(sorted(_flatten(material).items()))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def _flatten(payload: dict[str, Any], prefix: str = "") -> dict[str, str]:
    """Flatten a nested config payload into dotted keys with stringified values."""
    flat: dict[str, str] = {}
    for key, value in payload.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(_flatten(value, f"{path}."))
        else:
            flat[path] = repr(value)
    return flat


def resolve_sub_settings[T: BaseModel](
    default_factory: type[T],
    config_section: T | None = None,
    *,
    explicit_settings: T | None = None,
    **overrides: object,
) -> T:
    """Resolve effective configuration model by layering CLI overrides over config sections."""
    base = (
        explicit_settings
        if explicit_settings is not None
        else (config_section if config_section is not None else default_factory())
    )
    if isinstance(base, RuntimeSettings):
        return cast(T, base.with_overrides(**overrides))
    active = {k: v for k, v in overrides.items() if v is not None}
    base_dump = {
        **base.model_dump(exclude_defaults=True),
        **base.model_dump(exclude_unset=True),
    }
    payload = {**base_dump, **active}
    return default_factory.model_validate(payload)


def _propagate_registry_to_runtime(
    runtime: RuntimeSettings,
    registry: RegistrySettings,
) -> RuntimeSettings:
    """Propagate registry project and location into Antigravity runtime options when unset."""
    if (
        runtime.agent in _ANTIGRAVITY_AGENTS
        and registry.project
        and not runtime.options.get("project")
    ):
        opts = dict(runtime.options)
        opts["project"] = registry.project
        if "location" in registry.model_fields_set and not opts.get("location"):
            opts["location"] = registry.location
        return runtime.model_copy(update={"options": opts})
    return runtime
