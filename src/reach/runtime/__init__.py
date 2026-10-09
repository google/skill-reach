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

"""Define the core AgentRuntime interface and shared runtime agent helpers."""

from __future__ import annotations

import copy
import importlib
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping, Sequence
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Self, cast, override

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from reach.catalog import load_skills
from reach.config import (
    KEYWORD_AGENT,
    RuntimeSettings,
    agent_default_model,
    agent_profiles,
    resolve_path,
    resolve_registry_location,
    resolve_registry_project,
)
from reach.runtime._env import (
    detect_model_provider,
    has_agy_vertex_env,
    raise_missing_agent_dependency,
    resolve_blocked_env_vars,
    sanitize_subprocess_env,
)
from reach.runtime._fs import (
    install_skills,
    resolve_catalog_skills,
    safe_cleanup_isolated_dir,
)
from reach.runtime._subprocess import (
    check_tool_leak,
    process_failure_reason,
    run_subprocess_probe,
)
from reach.runtime.generator import BaseTextGenerator, TextGenerator, build_text_generator
from reach.runtime.profiles import model_profile

if TYPE_CHECKING:
    from reach.models import Catalog, Skill
    from reach.runtime.retriever import TwoStageRetrieverRuntime

logger = logging.getLogger(__name__)

#: Canonical identifier for the internal fake test runtime agent.
FAKE_AGENT = "fake"

__all__ = [
    "FAKE_AGENT",
    "KEYWORD_AGENT",
    "AgentOptions",
    "AgentRuntime",
    "AntigravityOptions",
    "AntigravityRuntime",
    "BaseTextGenerator",
    "CatalogFit",
    "CliAgentRuntime",
    "CliOptions",
    "SelectionOutcome",
    "SessionStatus",
    "SessionSummary",
    "SkillRoot",
    "SkillSelectionBase",
    "TextGenerator",
    "ToolCallInfo",
    "TrajectoryTracker",
    "TwoStageRetrieverRuntime",
    "VertexOptions",
    "agent_default_model",
    "antigravity_agents",
    "build_runtime",
    "build_text_generator",
    "builtin_tool_names",
    "cli_agents",
    "find_agent_for_model",
    "known_agents",
    "options_model",
    "register_agent",
    "resolve_options",
    "runtime_class",
]


class AgentOptions(BaseModel):
    """Base configuration common to all agent drivers."""

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)

    model: str = ""
    effort: str | None = None
    provider: str | None = None
    max_turns: int = Field(default=3, ge=1)
    early_exit: bool = True
    allowed_tools: tuple[str, ...] | None = None
    blocked_env_vars: tuple[str, ...] | None = None
    api_key: str | None = None
    use_symlinks: bool = True
    isolate_config_dir: bool = True
    auto_clean: bool = False
    json_schema: str | None = None

    #: Name of the field configuring an explicit isolation directory, if supported.
    isolation_dir_field: ClassVar[str | None] = None

    @property
    def custom_isolation_dir(self) -> Path | None:
        """Return custom isolation directory path configured on this options instance, if any."""
        if self.isolation_dir_field:
            val = getattr(self, self.isolation_dir_field, None)
            return Path(val) if val is not None else None
        return None

    @model_validator(mode="after")
    def _validate_isolation_dir_boundary(self) -> Self:
        """Validate that custom isolation directory does not point to active home or root."""
        if self.isolation_dir_field and self.custom_isolation_dir is not None:
            from reach.runtime._fs import validate_isolated_directory

            validate_isolated_directory(self.custom_isolation_dir, self.isolation_dir_field)
        return self


class VertexOptions(AgentOptions):
    """Configure Google Cloud Vertex AI and Application Default Credentials options."""

    vertex: bool | None = None
    project: str | None = None
    location: str | None = None


class AntigravityOptions(VertexOptions):
    """Configure common options for Antigravity-ecosystem agent runtimes."""

    use_symlinks: bool = False

    @property
    def effective_vertex(self) -> bool:
        """Return True if Vertex AI / Enterprise ADC mode is active.

        When vertex is not explicitly set and no Vertex toggle env var is active,
        ambient GEMINI_API_KEY / GOOGLE_API_KEY takes precedence over implicit
        GOOGLE_CLOUD_PROJECT detection.
        """
        if self.vertex is not None:
            return self.vertex
        if has_agy_vertex_env(os.environ):
            return True
        has_api_key = bool(
            self.api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        )
        return not has_api_key and bool(self.project or resolve_registry_project(self.project))

    @property
    def effective_project(self) -> str | None:
        """Return resolved Google Cloud project ID when Vertex/ADC is active.

        When api_key is explicitly set alongside Vertex (Express Mode), omit
        ambient/registry project fallback because VertexEndpoint forbids combining
        api_key with regional project/location parameters.
        """
        if not self.effective_vertex:
            return None
        if self.api_key:
            return self.project
        return resolve_registry_project(self.project)

    @property
    def effective_location(self) -> str | None:
        """Return resolved Google Cloud location when Vertex/ADC is active.

        When api_key is explicitly set alongside Vertex (Express Mode), omit
        ambient/registry location fallback because VertexEndpoint forbids combining
        api_key with regional project/location parameters.
        """
        if not self.effective_vertex:
            return None
        if self.api_key:
            return self.location
        return resolve_registry_location(self.location)

    @property
    def effective_api_key(self) -> str | None:
        """Return configured API key or fallback to environment variables."""
        if self.api_key:
            return self.api_key
        if self.effective_vertex:
            return None
        return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

    def resolve_model_provider(self, model: str = "") -> str | None:
        """Resolve model provider unless Vertex/ADC mode is active without explicit provider."""
        if self.effective_vertex and not self.provider:
            return None
        return detect_model_provider(
            model or self.model,
            self.provider,
            api_key=self.api_key,
        )

    def resolve_schema_dict(
        self,
        schema: str | Mapping[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Parse and validate explicit or configured JSON schema into a dictionary."""
        if isinstance(schema, Mapping):
            return dict(schema)
        raw_schema = schema if isinstance(schema, str) else self.json_schema
        if not raw_schema:
            return None
        try:
            parsed = json.loads(raw_schema)
        except json.JSONDecodeError as exc:
            msg = f"Invalid JSON schema: {exc}"
            raise ValueError(msg) from exc
        if not isinstance(parsed, dict):
            msg = f"Expected JSON schema object, got {type(parsed).__name__}"
            raise ValueError(msg)
        return parsed


class CliOptions(AgentOptions):
    """Hold common configuration options for CLI subprocess-driven agent runtimes."""

    executable: str = ""
    extra_args: tuple[str, ...] = ()

    def effort_args(self, flag: str = "--effort") -> list[str]:
        """Format CLI argument pair for non-empty effort setting."""
        return [flag, self.effort] if self.effort else []

    def provider_args(self, flag: str = "--provider") -> list[str]:
        """Format CLI argument pair for non-empty provider setting."""
        return [flag, self.provider] if self.provider else []

    def max_turns_args(self, flag: str = "--max-turns") -> list[str]:
        """Format CLI argument pair for non-None max_turns setting."""
        return [flag, str(self.max_turns)] if self.max_turns is not None else []

    def api_key_args(self, flag: str = "--api-key") -> list[str]:
        """Format CLI argument pair for non-empty api_key setting."""
        return [flag, self.api_key] if self.api_key else []


class ToolCallInfo(BaseModel):
    """Represent an observed tool invocation and its parameters."""

    model_config = ConfigDict(frozen=True)

    name: str = ""
    parameters: dict[str, Any] = Field(default_factory=dict)

    def __init__(
        self,
        name: str = "",
        parameters: dict[str, Any] | None = None,
        path: str | None = None,
        **data: Any,  # noqa: ANN401
    ) -> None:
        """Initialize tool invocation with optional direct path argument."""
        params = dict(parameters or data.pop("parameters", None) or {})
        if path is not None and "path" not in params and "AbsolutePath" not in params:
            params["path"] = str(path)
        super().__init__(name=name, parameters=params, **data)

    @model_validator(mode="before")
    @classmethod
    def _coerce_path(cls, data: Any) -> Any:  # noqa: ANN401
        """Populate parameters with path keyword argument when provided directly."""
        if isinstance(data, dict) and "path" in data:
            data = dict(data)
            params = dict(data.get("parameters") or {})
            val = data.pop("path")
            if val is not None and "path" not in params and "AbsolutePath" not in params:
                params["path"] = str(val)
            data["parameters"] = params
        return data

    @property
    def path(self) -> str | None:
        """Extract first non-empty filesystem path from recognized tool parameters."""
        from reach.runtime._fs import extract_tool_path

        return extract_tool_path(self.parameters)


class SessionStatus(StrEnum):
    """Enumerate canonical terminal execution statuses of an agent session."""

    CANCELLED = "CANCELLED"
    ERROR = "ERROR"
    SUCCESS = "SUCCESS"
    TIMEOUT = "TIMEOUT"


class SessionSummary(BaseModel):
    """Hold parsed session outcomes and telemetry across CLI runtime logs."""

    model_config = ConfigDict(frozen=True)

    cost_usd: float | None = None
    duration_ms: int | None = None
    prompt_tokens: int | None = None
    error: str | None = None
    invoked_skills: tuple[str, ...] = ()
    early_exit: bool = False
    turns_taken: int = Field(default=1, ge=1)
    tool_calls: tuple[ToolCallInfo, ...] = ()
    observed_tools: tuple[str, ...] = ()
    reasoning: tuple[str, ...] = ()
    observed_catalog: tuple[str, ...] = ()
    resolved_model: str = ""
    status: SessionStatus | str | None = None
    retries: int = 0

    @property
    def invoked_skill(self) -> str | None:
        """Return the first invoked skill name, or None if none was invoked."""
        return self.invoked_skills[0] if self.invoked_skills else None

    @field_validator("status", mode="before")
    @classmethod
    def _normalize_status(cls, value: object) -> SessionStatus | str | None:
        """Coerce known status variants to canonical SessionStatus enum members."""
        if value is None or isinstance(value, SessionStatus):
            return value
        if isinstance(value, str):
            normalized = value.strip()
            upper = normalized.upper()
            try:
                return SessionStatus(upper)
            except ValueError:
                return normalized
        return str(value)

    @property
    def saw_result(self) -> bool:
        """Return True if stream contained a terminal result event or early exit."""
        return self.status is not None or self.early_exit

    @model_validator(mode="before")
    @classmethod
    def _derive_observed_tools(cls, data: Any) -> Any:  # noqa: ANN401 (Pydantic before validator)
        """Populate observed tools from tool calls if not explicitly provided."""
        if isinstance(data, dict):
            data = dict(data)
            calls = data.get("tool_calls")
            if calls and not data.get("observed_tools"):
                names = {getattr(c, "name", None) or c.get("name") for c in calls if c}
                data["observed_tools"] = tuple(sorted(n for n in names if n))
        return data

    def to_outcome(
        self,
        observed_catalog: tuple[str, ...] | list[str] = (),
        fallback_model: str = "",
        early_exit: bool = False,
        turns_taken: int | None = None,
    ) -> SelectionOutcome:
        """Convert parsed session summary into a canonical SelectionOutcome."""
        effective_early_exit = early_exit or self.early_exit
        turns = turns_taken if turns_taken is not None else self.turns_taken

        catalog = tuple(observed_catalog) or self.observed_catalog
        return SelectionOutcome(
            cost_usd=self.cost_usd,
            duration_ms=self.duration_ms,
            prompt_tokens=self.prompt_tokens,
            error=self.error,
            invoked_skills=self.invoked_skills,
            early_exit=effective_early_exit,
            turns_taken=turns,
            tool_calls=self.tool_calls,
            observed_tools=self.observed_tools,
            reasoning=self.reasoning,
            resolved_model=self.resolved_model or fallback_model,
            status=self.status,
            observed_catalog=catalog,
        )


class SelectionOutcome(SessionSummary):
    """Represent the observable result of probing an agent runtime with a query."""

    @model_validator(mode="before")
    @classmethod
    def _enforce_early_exit_invariants(cls, data: Any) -> Any:  # noqa: ANN401 (Pydantic before validator)
        """Ensure process cancellation artifacts are never reported as probe errors."""
        if isinstance(data, dict) and data.get("early_exit"):
            data = dict(data)
            err = str(data.get("error") or "")
            if err and not err.startswith(("tool leak", "residency leak")):
                data["error"] = None
        return data


class TrajectoryTracker:
    """Track multi-turn skill invocations and evaluate early exit conditions."""

    def __init__(
        self,
        target_skill: str | None = None,
        max_turns: int = 3,
        early_exit: bool = True,
    ) -> None:
        """Initialize tracker with optional target skill and turn boundaries."""
        self.target_skill = target_skill
        self.max_turns = max(max_turns, 1)
        self.early_exit = early_exit
        self.invoked_skills: list[str] = []
        self.early_exit_hit: bool = False

    def observe(self, skill: str | Sequence[str] | None) -> bool:
        """Record skill invocation(s) and return True if early-exit stop condition is met."""
        if self.early_exit and self.early_exit_hit:
            return True
        if not skill:
            return False
        items = (skill,) if isinstance(skill, str) else tuple(skill)
        for s in items:
            if not s:
                continue
            if not self.invoked_skills or self.invoked_skills[-1] != s:
                self.invoked_skills.append(s)
                if self.early_exit:
                    if self.target_skill is not None and s == self.target_skill:
                        self.early_exit_hit = True
                        return True
                    if len(self.invoked_skills) >= self.max_turns:
                        self.early_exit_hit = True
                        return True
        return False

    @property
    def turns_taken(self) -> int:
        """Return count of turns taken based on distinct recorded invocations."""
        return len(self.invoked_skills) if self.invoked_skills else 1

    def apply_to_outcome(self, outcome: SelectionOutcome) -> SelectionOutcome:
        """Normalize a SelectionOutcome through tracker early-exit and turn invariants."""
        if self.early_exit and self.early_exit_hit:
            skills = tuple(self.invoked_skills)
            early = True
        else:
            replay = TrajectoryTracker(
                target_skill=self.target_skill,
                max_turns=self.max_turns,
                early_exit=self.early_exit,
            )
            replay.observe(outcome.invoked_skills)
            skills = tuple(replay.invoked_skills[: self.max_turns])
            early = bool(self.early_exit and (replay.early_exit_hit or outcome.early_exit))
        was_truncated = len(outcome.invoked_skills) > len(skills)
        turns = min(outcome.turns_taken, len(skills) or 1) if was_truncated else outcome.turns_taken
        new_invoked_skill = skills[0] if skills else None
        return outcome.model_copy(
            update={
                "invoked_skill": new_invoked_skill,
                "invoked_skills": skills,
                "early_exit": early,
                "turns_taken": turns,
            },
        )


class SkillRoot(BaseModel):
    """Represent a skill discovery directory location and its resolution precedence."""

    model_config = ConfigDict(frozen=True)

    path: Path
    scope: str
    precedence: int = Field(default=0, ge=0)


class CatalogFit(BaseModel):
    """Report catalog residency limits and potential description truncation metrics."""

    model_config = ConfigDict(frozen=True)

    allowed: int = Field(default=0, ge=0)
    asked: int = Field(default=0, ge=0)
    unit: str = ""
    truncated: int = Field(default=0, ge=0)
    remedy: str = ""
    elided_skills: tuple[str, ...] = ()

    @property
    def rations(self) -> bool:
        """Return True if the runtime enforces space limitations on catalog listings."""
        return self.allowed > 0

    @property
    def whole(self) -> bool:
        """Return True if all skills are presented without description truncation."""
        return not self.truncated


_AGENT_FACTORIES: dict[
    str,
    tuple[Callable[[RuntimeSettings], AgentRuntime], type[BaseModel] | None],
] = {}


def register_agent(
    name: str,
    factory: Callable[[RuntimeSettings], AgentRuntime],
    options: type[BaseModel] | None = None,
) -> None:
    """Register a runtime factory and optional options schema for an agent name."""
    _AGENT_FACTORIES[name] = (factory, options)


_BUILTIN_AGENTS: dict[str, tuple[str, str, str]] = {
    "antigravity-cli": (
        "reach.runtime.antigravity_cli",
        "AntigravityCliOptions",
        "AntigravityCliRuntime",
    ),
    "antigravity-sdk": (
        "reach.runtime.antigravity_sdk",
        "AntigravitySdkOptions",
        "AntigravitySdkRuntime",
    ),
    "claude-code": (
        "reach.runtime.claude_code",
        "ClaudeCodeOptions",
        "ClaudeCodeRuntime",
    ),
    "goose": (
        "reach.runtime.goose",
        "GooseOptions",
        "GooseRuntime",
    ),
    "keyword": (
        "reach.runtime.keyword",
        "KeywordOptions",
        "KeywordRuntime",
    ),
    "pi": (
        "reach.runtime.pi",
        "PiOptions",
        "PiRuntime",
    ),
}


def _load_builtin_entry(
    agent: str,
) -> tuple[Callable[[RuntimeSettings], AgentRuntime], type[BaseModel] | None] | None:
    """Dynamically import and return factory and options model for a builtin agent."""
    spec = _BUILTIN_AGENTS.get(agent)
    if spec is None:
        return None
    mod_name, opt_name, rt_name = spec
    try:
        mod = importlib.import_module(mod_name)
    except ImportError as err:
        raise_missing_agent_dependency(agent, err, role="runtime")
    opt_cls = getattr(mod, opt_name, None)
    rt_cls = getattr(mod, rt_name)
    return (rt_cls), opt_cls


def known_agents(config_path: Path | str | None = None) -> tuple[str, ...]:
    """Return tuple of supported agent runtime names from configuration."""
    profiles = agent_profiles(config_path)
    builtins = set(_BUILTIN_AGENTS.keys())
    if profiles:
        return tuple(sorted(set(profiles.keys()) | builtins | set(_AGENT_FACTORIES.keys())))
    return tuple(sorted(builtins | set(_AGENT_FACTORIES.keys())))


def cli_agents(config_path: Path | str | None = None) -> tuple[str, ...]:
    """Return tuple of supported agent runtime names that execute via CLI subprocesses."""
    return tuple(
        agent
        for agent in known_agents(config_path)
        if (opt := options_model(agent)) is not None and issubclass(opt, CliOptions)
    )


def antigravity_agents(config_path: Path | str | None = None) -> tuple[str, ...]:
    """Return tuple of supported agent runtime names that inherit from AntigravityRuntime."""
    return tuple(
        agent
        for agent in known_agents(config_path)
        if (cls := runtime_class(agent)) is not None and issubclass(cls, AntigravityRuntime)
    )


def _agent_supported_models(
    agents: dict[object, object],
) -> list[tuple[str, list[str]]]:
    """Extract list of (agent_name, supported_models) pairs with lowered model names."""
    pairs: list[tuple[str, list[str]]] = []
    for agent_name, agent_info in agents.items():
        if isinstance(agent_info, dict):
            models = [str(m).lower() for m in agent_info.get("models", [])]
            pairs.append((str(agent_name), models))
    return pairs


def _match_agent_model(
    agent_models: list[tuple[str, list[str]]],
    model_lower: str,
) -> str | None:
    """Find agent matching exact model name, falling back to substring match."""
    for agent_name, supported in agent_models:
        if model_lower in supported:
            return agent_name
    for agent_name, supported in agent_models:
        for candidate in supported:
            if candidate in model_lower or model_lower in candidate:
                return agent_name
    return None


def find_agent_for_model(model: str, config_path: Path | str | None = None) -> str | None:
    """Dynamically determine which agent runtime supports the given model."""
    if not model or not model.strip():
        return None
    profiles = agent_profiles(config_path)
    if not profiles:
        return None
    pairs = [(name, [m.lower() for m in prof.models]) for name, prof in profiles.items()]
    return _match_agent_model(pairs, model.strip().lower())


class AgentRuntime[OptionsT: AgentOptions](ABC):
    """Define standard abstract base class for agent runtime implementations."""

    name: str
    settings: RuntimeSettings | None = None
    options: OptionsT = cast("Any", AgentOptions())
    _resident: tuple[str, ...] = ()
    is_dynamic: bool = False

    def __init__(
        self,
        settings: RuntimeSettings | None = None,
        options: OptionsT | None = None,
    ) -> None:
        """Initialize agent runtime settings and options."""
        agent_name = getattr(self, "name", "agent")
        self.settings = settings
        if options is not None:
            self.options = options
        elif settings is not None:
            resolved = resolve_options(settings)
            opt_cls = options_model(agent_name)
            if opt_cls is not None and issubclass(opt_cls, AgentOptions):
                self.options = cast(
                    "OptionsT",
                    resolved
                    if isinstance(resolved, opt_cls)
                    else opt_cls.model_validate(dict(settings.options or {})),
                )
            else:
                self.options = cast(
                    "OptionsT",
                    resolved
                    if isinstance(resolved, AgentOptions)
                    else AgentOptions.model_validate(dict(settings.options or {})),
                )
        else:
            opt_cls = options_model(agent_name)
            if opt_cls is not None and issubclass(opt_cls, AgentOptions):
                self.options = cast("OptionsT", opt_cls())
            else:
                self.options = cast("OptionsT", AgentOptions())
        self._resident = ()

    @property
    def timeout_s(self) -> int | None:
        """Return per-probe execution timeout in seconds from settings."""
        return self.settings.timeout_s if self.settings is not None else None

    @property
    def skills_subpath(self) -> str:
        """Return relative skill directory subpath from profile or runtime default."""
        profile = agent_profiles().get(self.name)

        if profile is not None and profile.skills_dir:
            return profile.skills_dir
        if subpath := getattr(self, "_skills_subpath", None):
            return str(subpath)
        return ""

    @property
    def model(self) -> str:
        """Return the identifier of the model being evaluated."""
        return self.options.model

    @model.setter
    def model(self, value: str) -> None:
        """Update the configured model identifier in options."""
        self.options = self.options.model_copy(update={"model": value})

    @property
    def effort(self) -> str | None:
        """Return the reasoning effort tier being evaluated."""
        return self.options.effort

    @property
    def provider(self) -> str | None:
        """Return the model provider identifier being evaluated."""
        return self.options.provider

    @property
    def max_turns(self) -> int:
        """Return the maximum turns configured on the runtime options."""
        return self.options.max_turns

    @property
    def early_exit(self) -> bool:
        """Return whether early exit is enabled on the runtime options."""
        return self.options.early_exit

    @property
    def api_key(self) -> str | None:
        """Return the API key string configured on the runtime options."""
        return self.options.api_key

    @property
    def allowed_tools(self) -> tuple[str, ...] | None:
        """Return explicit allowed tools override from options or settings."""
        return (
            self.options.allowed_tools
            if self.options.allowed_tools is not None
            else getattr(self.settings, "allowed_tools", None)
        )

    @property
    def blocked_env_vars(self) -> tuple[str, ...] | None:
        """Return explicit blocked environment variables override from options or settings."""
        return resolve_blocked_env_vars(self.options, self.settings)

    @property
    def use_symlinks(self) -> bool:
        """Return whether symlink installation is enabled on the runtime options."""
        return self.options.use_symlinks

    @property
    def isolate_config_dir(self) -> bool:
        """Return whether isolated runtime config is enabled on the runtime options."""
        return self.options.isolate_config_dir

    @property
    def auto_clean(self) -> bool:
        """Return whether automatic post-probe cleanup is enabled on the runtime options."""
        return self.options.auto_clean

    @property
    def rations_catalog(self) -> bool:
        """Return True if this runtime actively rations skill listing budgets."""
        return False

    @property
    def is_cli(self) -> bool:
        """Return True if this runtime executes via a CLI subprocess."""
        return False

    @property
    def isolation_dir_field(self) -> str | None:
        """Return the options field name used for custom directory isolation, if supported."""
        return getattr(self.options, "isolation_dir_field", None)

    @property
    def custom_isolation_dir(self) -> Path | None:
        """Return the configured custom isolation directory path, or None."""
        return getattr(self.options, "custom_isolation_dir", None)

    def skills_dir(self, workdir: Path) -> Path:
        """Return standard skill directory path in the workspace."""
        return Path(workdir) / self.skills_subpath

    def skill_roots(self, workdir: Path) -> tuple[SkillRoot, ...]:
        """Return discovery directories where runtime searches for skills."""
        here = self.skills_dir(resolve_path(workdir))
        return (SkillRoot(path=here, scope="project", precedence=0),) if here.is_dir() else ()

    def fit(self, catalog: Catalog, skills: Iterable[Skill]) -> CatalogFit:  # noqa: ARG002
        """Evaluate whether a catalog fits listing budgets without materializing."""
        return CatalogFit()

    def install(self, catalog: Catalog, skills: Iterable[Skill], workdir: Path) -> Path:
        """Materialize resident skills in workspace and return workspace path."""
        target = resolve_path(workdir)
        self._validate_install(target)
        by_name = resolve_catalog_skills(catalog, skills)
        self._resident = install_skills(
            catalog,
            by_name,
            self.skills_dir(target),
            use_symlinks=self.use_symlinks,
        )
        self._post_install(target)
        return target

    def _validate_install(self, workdir: Path) -> None:
        """Validate workspace preconditions before installation."""
        del workdir

    def _post_install(self, workdir: Path) -> None:
        """Configure permissions or settings after installation."""
        del workdir

    def post_probe(self, workdir: Path) -> None:
        """Execute post-probe cleanup actions."""
        del workdir

    def clone_isolated(self) -> Self:
        """Create a thread-local isolated clone of this runtime."""
        clone = copy.copy(self)
        clone._resident = ()  # noqa: SLF001
        return clone

    def cleanup(self) -> None:  # noqa: B027
        """Release any isolated temporary resources owned by this runtime."""

    def build_env(self, workdir: Path | None = None) -> dict[str, str]:
        """Assemble process environment for agent execution."""
        del workdir
        return sanitize_subprocess_env(dict(os.environ), blocked_env_vars=self.blocked_env_vars)

    def make_tracker(self, target_skill: str | None = None) -> TrajectoryTracker:
        """Create a TrajectoryTracker configured with this runtime's turn and early-exit options."""
        return TrajectoryTracker(
            target_skill=target_skill,
            max_turns=self.options.max_turns,
            early_exit=self.options.early_exit,
        )

    @abstractmethod
    def select(
        self,
        query_text: str,
        workdir: Path,
        target_skill: str | None = None,
    ) -> SelectionOutcome:
        """Execute a single query probe and return the observed skill selection."""
        ...

    @property
    def effective_effort(self) -> str | None:
        """Return configured reasoning effort or default from model profile."""
        options = getattr(self, "options", None)
        if options is not None and (effort := getattr(options, "effort", None)):
            return None if effort.lower() in ("none", "off") else effort
        try:
            return model_profile(self.model).effort
        except (KeyError, ValueError):
            return None

    def resident_skill_paths(self, workdir: Path) -> frozenset[Path]:
        """Return set of valid filesystem directory paths for resident skills."""
        skills_dir = self.skills_dir(resolve_path(workdir))
        return frozenset(resolve_path(skills_dir / name) for name in self._resident)

    def parse_stream(
        self,
        lines: Iterable[str],
        resident: Sequence[str] = (),
        early_exit: bool = False,
    ) -> SessionSummary:
        """Parse transcript or log lines into a standardized SessionSummary.

        Default implementation records early_exit and assigns SUCCESS status
        if lines were produced. Subclasses override this to extract tool
        invocations, reasoning, and telemetry.
        """
        del resident
        line_list = list(lines)
        return SessionSummary(
            early_exit=early_exit,
            status=SessionStatus.SUCCESS if line_list else None,
        )


class CliAgentRuntime[CliOptionsT: CliOptions](AgentRuntime[CliOptionsT], ABC):
    """Abstract base runtime for command-line interface agent drivers."""

    options: CliOptionsT
    api_key_env_var: str | None = None

    def __init__(
        self,
        settings: RuntimeSettings | None = None,
        options: CliOptionsT | None = None,
    ) -> None:
        """Initialize CLI agent runtime settings and options."""
        super().__init__(settings=settings, options=options)
        agent_name = getattr(self, "name", "agent")
        if self.settings is None and options_model(agent_name) is not None:
            self.settings = RuntimeSettings(
                agent=agent_name,
                options=self.options.model_dump(mode="json") if options is not None else {},
            )
        self.completion_cost_usd = 0.0
        self.completions = 0

    @property
    @override
    def is_cli(self) -> bool:
        """Return True if this runtime executes via a CLI subprocess."""
        return True

    @abstractmethod
    def build_command(self, query_text: str) -> list[str]:
        """Assemble command-line arguments for executing a probe."""
        ...

    @abstractmethod
    @override
    def parse_stream(
        self,
        lines: Iterable[str],
        resident: Sequence[str] = (),
        early_exit: bool = False,
    ) -> SessionSummary:
        """Parse CLI stdout lines into a standardized session summary."""
        ...

    def extract_skills_from_line(self, line: str) -> Sequence[str]:
        """Extract invoked skill names from an event line for early-exit detection."""
        single = self.extract_skill_from_line(line)
        return (single,) if single else ()

    def extract_skill_from_line(self, line: str) -> str | None:
        """Extract invoked skill name from an event line for early-exit detection."""
        del line
        return None

    @override
    def build_env(self, workdir: Path | None = None) -> dict[str, str]:
        """Assemble process environment with API keys and workspace overrides."""
        del workdir
        env = dict(os.environ)
        if (home_dir := getattr(self.options, "home_dir", None)) is not None:
            env["HOME"] = str(home_dir)
        if (api_key := getattr(self.options, "api_key", None)) is not None and self.api_key_env_var:
            env[self.api_key_env_var] = str(api_key)
        return sanitize_subprocess_env(env, blocked_env_vars=self.blocked_env_vars)

    def validate_outcome(
        self,
        summary: SessionSummary,
        workdir: Path,
    ) -> str | None:
        """Validate status and security isolation boundaries for a parsed session."""
        del workdir
        if summary.status is not None and summary.status != SessionStatus.SUCCESS:
            return summary.error or f"runtime error: {summary.status}"
        return check_tool_leak(summary.observed_tools, self.allowed_tools)

    isolation_dir_name: ClassVar[str | None] = None

    def effective_isolation_dir(self, workdir: Path | None = None) -> Path | None:
        """Return the effective isolated configuration directory for the given workspace."""
        if not self.options.isolate_config_dir:
            return None
        if self.custom_isolation_dir:
            return self.custom_isolation_dir
        if workdir is not None and self.isolation_dir_name:
            return Path(workdir) / self.isolation_dir_name
        return None

    @override
    def post_probe(self, workdir: Path) -> None:
        """Clean isolated configuration directory after probe if auto_clean is enabled."""
        if not self.options.auto_clean:
            return
        iso_dir = self.effective_isolation_dir(workdir)
        if iso_dir is not None:
            safe_cleanup_isolated_dir(workdir, iso_dir)

    @override
    def select(
        self,
        query_text: str,
        workdir: Path,
        target_skill: str | None = None,
    ) -> SelectionOutcome:
        """Execute a query probe via the unified CLI subprocess template pipeline."""
        tracker = self.make_tracker(target_skill)

        def _on_line(line: str) -> bool:
            skills = self.extract_skills_from_line(line)
            return tracker.observe(skills)

        start_time = time.monotonic()
        try:
            completed, err = run_subprocess_probe(
                self.build_command(query_text),
                workdir,
                self.timeout_s,
                env=self.build_env(workdir),
                on_line=_on_line,
            )
            elapsed_ms = int((time.monotonic() - start_time) * 1000)
            if err is not None or completed is None:
                return SelectionOutcome(
                    error=err or "subprocess failed",
                    observed_catalog=self._resident,
                )

            summary = self.parse_stream(
                completed.stdout.splitlines(),
                resident=self._resident,
                early_exit=tracker.early_exit_hit,
            )
            if not summary.saw_result:
                reason = process_failure_reason(completed)
                if getattr(summary, "retries", 0) > 0:
                    reason = f"rate limit (429): {summary.retries} retries exceeded"
                return SelectionOutcome(
                    error=f"no result event: {reason}",
                    observed_catalog=self._resident,
                )

            catalog = getattr(summary, "observed_catalog", ()) or self._resident
            outcome = tracker.apply_to_outcome(
                summary.to_outcome(
                    observed_catalog=catalog,
                    fallback_model=self.model,
                    early_exit=tracker.early_exit_hit,
                ),
            )

            # Security tool leak and outcome validation
            if validation_error := self.validate_outcome(summary, workdir):
                return outcome.model_copy(update={"error": validation_error})

            duration_ms = outcome.duration_ms or elapsed_ms
            return outcome.model_copy(update={"duration_ms": duration_ms})
        except Exception as exc:  # noqa: BLE001
            return SelectionOutcome(
                error=f"unexpected runtime error: {exc}",
                observed_catalog=self._resident,
            )
        finally:
            self.post_probe(workdir)


class SkillSelectionBase(BaseModel):
    """Base model declaring structured skill selection response interface."""

    model_config = ConfigDict(extra="forbid")

    selected_skill: str | None = Field(
        default=None,
        description="The skill to invoke, or null if no skill applies",
    )
    reasoning: str = Field(
        default="",
        description="Brief explanation of why this skill was selected or why null was returned",
    )


class AntigravityRuntime[AntigravityOptionsT: AntigravityOptions](
    AgentRuntime[AntigravityOptionsT], ABC
):
    """Shared base runtime for Antigravity-ecosystem drivers (CLI and SDK)."""

    ANTIGRAVITY_SELECTION_TOOLS: frozenset[str] = frozenset(
        {
            "view_file",
            "list_dir",
            "grep_search",
            "find_by_name",
        }
    )
    ANTIGRAVITY_BUILTIN_TOOLS: frozenset[str] = ANTIGRAVITY_SELECTION_TOOLS | frozenset(
        {
            "ask_question",
            "finish",
            "multi_replace_file_content",
            "read_url_content",
            "replace_file_content",
            "run_command",
            "search_web",
            "write_to_file",
        }
    )

    @property
    def selection_tools(self) -> frozenset[str]:
        """Return standard inspection tool identifiers permitted during skill selection."""
        return self.ANTIGRAVITY_SELECTION_TOOLS

    @classmethod
    def selection_schema(cls, resident: Sequence[str]) -> type[SkillSelectionBase]:
        """Generate dynamic Pydantic model constraining selection to resident skills."""
        import typing

        from pydantic import create_model

        literal_type: Any = typing.cast("Any", typing.Literal)[tuple(resident)] if resident else str

        return create_model(
            "SkillSelection",
            __base__=SkillSelectionBase,
            selected_skill=(
                literal_type | None,
                Field(default=None, description="The skill to invoke, or null if no skill applies"),
            ),
            reasoning=(
                str,
                Field(
                    default="",
                    description=(
                        "Brief explanation of why this skill was selected or why null was returned"
                    ),
                ),
            ),
        )

    @classmethod
    def selection_json_schema(cls, resident: Sequence[str]) -> str:
        """Derive JSON Schema string directly from the canonical Pydantic model."""
        schema_dict = cls.selection_schema(resident).model_json_schema()
        schema_dict["required"] = ["selected_skill", "reasoning"]
        return json.dumps(schema_dict)

    def format_available_skills_prompt(self, workdir: Path) -> str | None:
        """Format the progressive-disclosure prompt describing available skills in the workspace."""
        skills_dir = self.skills_dir(workdir)
        if not skills_dir.is_dir():
            return None
        try:
            installed_skills = load_skills(skills_dir)
        except (OSError, ValueError, KeyError):
            logger.warning(
                "Failed to load skills from %s for prompt formatting",
                skills_dir,
                exc_info=True,
            )
            return None
        if not installed_skills:
            return None
        header = (
            "If a skill seems relevant to your current task, you MUST read its "
            "`SKILL.md` instructions using `view_file` before proceeding.\n\n"
            "Available skills:"
        )
        skills_lines = [header]
        for s in installed_skills:
            skill_md_path = skills_dir / s.name / "SKILL.md"
            skills_lines.append(f"- {s.name} ({skill_md_path}): {s.description}")
        return "\n".join(skills_lines)


def builtin_tool_names() -> frozenset[str]:
    """Return canonical built-in tool primitives across all supported agent harnesses."""
    from reach.runtime.claude_code import DEFAULT_DENIED_TOOLS, SKILL_TOOL_NAME

    return (
        frozenset(DEFAULT_DENIED_TOOLS)
        | {SKILL_TOOL_NAME}
        | AntigravityRuntime.ANTIGRAVITY_BUILTIN_TOOLS
        | {"load_skill"}
    )


def options_model(agent: str) -> type[BaseModel] | None:
    """Return the options schema class corresponding to the named agent."""
    if agent in _AGENT_FACTORIES:
        return _AGENT_FACTORIES[agent][1]
    builtin = _load_builtin_entry(agent)
    if builtin is not None:
        return builtin[1]
    return None


def runtime_class(agent: str) -> type[AgentRuntime] | None:
    """Return the runtime implementation class corresponding to the named agent."""
    if agent in _AGENT_FACTORIES:
        factory = _AGENT_FACTORIES[agent][0]
        return factory if isinstance(factory, type) and issubclass(factory, AgentRuntime) else None
    builtin = _load_builtin_entry(agent)
    if builtin is not None:
        rt = builtin[0]
        return rt if isinstance(rt, type) and issubclass(rt, AgentRuntime) else None
    return None


def resolve_options(settings: RuntimeSettings) -> BaseModel | None:
    """Parse and validate agent-specific options dictionary against its schema."""
    model = options_model(settings.agent)
    raw = dict(settings.options or {})
    if model is not None:
        if "max_turns" not in raw and getattr(settings, "max_turns", None) is not None:
            raw["max_turns"] = settings.max_turns
        if "early_exit" not in raw and getattr(settings, "early_exit", None) is not None:
            raw["early_exit"] = settings.early_exit
        if "allowed_tools" not in raw and getattr(settings, "allowed_tools", None) is not None:
            raw["allowed_tools"] = settings.allowed_tools
        if (
            "blocked_env_vars" not in raw
            and getattr(settings, "blocked_env_vars", None) is not None
        ):
            raw["blocked_env_vars"] = settings.blocked_env_vars
        return model.model_validate(raw)
    if raw:
        msg = f"{_no_options_reason(settings.agent)}; got {sorted(raw)}"
        raise ValueError(
            msg,
        )
    return None


def _no_options_reason(agent: str) -> str:
    """Generate error message when options are passed to an invalid agent."""
    if agent in known_agents():
        return f"runtime agent {agent!r} takes no options"
    return f"unknown runtime agent {agent!r}; expected one of {', '.join(known_agents())}"


def build_runtime(settings: RuntimeSettings) -> AgentRuntime:
    """Instantiate and configure an AgentRuntime from settings."""
    if settings.agent in _AGENT_FACTORIES:
        rt = _AGENT_FACTORIES[settings.agent][0](settings)
    else:
        builtin = _load_builtin_entry(settings.agent)
        if builtin is not None:
            rt = builtin[0](settings)
        else:
            agents = ", ".join(known_agents())
            msg = f"unknown runtime agent {settings.agent!r}; expected one of {agents}"
            raise ValueError(msg)
    if getattr(rt, "settings", None) is None:
        rt.settings = settings
    return rt


def __getattr__(name: str) -> object:
    """Provide lazy dynamic exports for reach.runtime members."""
    if name == "TwoStageRetrieverRuntime":
        from .retriever import TwoStageRetrieverRuntime

        return TwoStageRetrieverRuntime
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
