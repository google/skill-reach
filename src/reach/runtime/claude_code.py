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

"""Drive Claude Code as an agent evaluation runtime."""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Iterable, Mapping, Sequence
from math import floor
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, override

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_serializer
from pydantic import ValidationError as PydanticValidationError

from reach.catalog import resident_skills
from reach.config import DEFAULT_CLAUDE_MODEL, RuntimeSettings
from reach.runtime import (
    CatalogFit,
    CliAgentRuntime,
    CliOptions,
    SessionStatus,
    SessionSummary,
    SkillRoot,
    agent_default_model,
)
from reach.runtime._env import sync_claude_settings_env
from reach.runtime._fs import ensure_private_directory
from reach.runtime._subprocess import (
    iter_json_lines,
    process_failure_reason,
)
from reach.runtime.claude_listing import (
    BUDGET_FRACTION_PLACES,
    DEFAULT_CHARS_PER_TOKEN,
    DEFAULT_COMPLETION_WINDOW,
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_LISTING_BUDGET_CHARS,
    WIDE,
    ZERO_WIDTH,
    ListingEntry,
    SkillListing,
    _listing_profile,
    budget_fraction_for,
    display_width,
    fit_skill_listing,
    listing_budget_chars,
    listing_chars,
    parse_fraction,
)
from reach.runtime.generator import BaseTextGenerator

if TYPE_CHECKING:
    from pydantic_core.core_schema import SerializerFunctionWrapHandler

    from reach.models import Catalog, Skill

__all__ = [
    "BUDGET_FRACTION_PLACES",
    "DEFAULT_CHARS_PER_TOKEN",
    "DEFAULT_COMPLETION_WINDOW",
    "DEFAULT_CONTEXT_WINDOW",
    "DEFAULT_DENIED_TOOLS",
    "DEFAULT_LISTING_BUDGET_CHARS",
    "DEFAULT_SKILL_OVERRIDES",
    "EXPECTED_TERMINAL_SUBTYPES",
    "POSIX_ENTERPRISE_SKILL_DIRS",
    "REQUIRED_TOOLS",
    "SKILL_TOOL_NAME",
    "WIDE",
    "ZERO_WIDTH",
    "ClaudeCodeOptions",
    "ClaudeCodeRuntime",
    "ClaudeGenerator",
    "ListingEntry",
    "SkillListing",
    "StreamSummary",
    "budget_fraction_for",
    "display_width",
    "enterprise_skill_dirs",
    "fit_skill_listing",
    "inherited_skill_dirs",
    "listing_budget_chars",
    "listing_chars",
    "parse_fraction",
    "parse_stream",
    "user_skills_dir",
]

#: Tool-use block name that indicates a skill selection.
SKILL_TOOL_NAME = "Skill"

#: Subtype emitted when the turn cap terminates an evaluation probe.
EXPECTED_TERMINAL_SUBTYPES = frozenset({"success", "error_max_turns", "early_exit"})

#: Allowed tool identifier for skill selection.
REQUIRED_TOOLS = frozenset({SKILL_TOOL_NAME})

#: Tools explicitly denied to restrict model execution to skill selection.
DEFAULT_DENIED_TOOLS: tuple[str, ...] = (
    "Artifact",
    "Bash",
    "BashOutput",
    "CronCreate",
    "CronDelete",
    "CronList",
    "Edit",
    "EnterPlanMode",
    "EnterWorktree",
    "ExitPlanMode",
    "ExitWorktree",
    "Glob",
    "Grep",
    "KillShell",
    "MultiEdit",
    "NotebookEdit",
    "Read",
    "ReportFindings",
    "ScheduleWakeup",
    "SendMessage",
    "Task",
    "TaskCreate",
    "TaskGet",
    "TaskList",
    "TaskOutput",
    "TaskStop",
    "TaskUpdate",
    "TodoWrite",
    "ToolSearch",
    "WebFetch",
    "WebSearch",
    "Workflow",
    "Write",
)

#: Bundled skills explicitly disabled via skillOverrides settings.
DEFAULT_SKILL_OVERRIDES: Mapping[str, str] = {"doctor": "off"}


class ClaudeCodeOptions(CliOptions):
    """Specify runtime configuration options for the Claude Code CLI agent."""

    executable: str = "claude"
    model: str = Field(
        default_factory=lambda: agent_default_model("claude-code") or DEFAULT_CLAUDE_MODEL,
    )
    setting_sources: str = "project"
    denied_tools: tuple[str, ...] = DEFAULT_DENIED_TOOLS
    exclude_dynamic_prompt: bool = True
    disable_bundled_skills: bool = True
    skill_overrides: Mapping[str, str] = Field(
        default_factory=lambda: dict(DEFAULT_SKILL_OVERRIDES),
    )
    skill_listing_budget_fraction: float | None = Field(default=None, gt=0.0, le=1.0)
    skill_listing_max_desc_chars: int | None = Field(default=None, gt=0)
    tools: str | None = "Skill"
    strict_mcp_config: bool = True
    no_session_persistence: bool = True
    config_dir: Path | None = None
    cloud_ml_region: str | None = None
    vertex_project_id: str | None = None
    isolation_dir_field: ClassVar[str | None] = "config_dir"

    @model_serializer(mode="wrap")
    def _omit_unset_listing_controls(
        self,
        handler: SerializerFunctionWrapHandler,
    ) -> dict[str, Any]:
        """Exclude unset listing budget fields from serialized dictionary."""
        payload: dict[str, Any] = handler(self)
        for key in ("skill_listing_budget_fraction", "skill_listing_max_desc_chars"):
            if payload.get(key) is None:
                payload.pop(key, None)
        return payload

    def settings_json(self) -> str | None:
        """Render inline JSON configuration string for the --settings argument."""
        payload: dict[str, object] = {}
        if self.disable_bundled_skills:
            payload["disableBundledSkills"] = True
        if self.skill_overrides:
            payload["skillOverrides"] = dict(self.skill_overrides)
        if self.skill_listing_budget_fraction is not None:
            payload["skillListingBudgetFraction"] = self.skill_listing_budget_fraction
        if self.skill_listing_max_desc_chars is not None:
            payload["skillListingMaxDescChars"] = self.skill_listing_max_desc_chars
        return json.dumps(payload, sort_keys=True) if payload else None


class StreamSummary(SessionSummary):
    """Aggregate invocations, telemetry, and observed catalog from stream logs."""

    result_subtype: str | None = None
    retries: int = 0

    @property
    def leaked_tools(self) -> tuple[str, ...]:
        """Return tuple of tools executed that were not in the required tool set."""
        return tuple(sorted(set(self.observed_tools) - REQUIRED_TOOLS))


POSIX_ENTERPRISE_SKILL_DIRS: tuple[Path, ...] = (
    Path("/Library/Application Support/ClaudeCode/skills"),
    Path("/etc/claude-code/skills"),
)


def enterprise_skill_dirs() -> tuple[Path, ...]:
    """Return configured enterprise managed skill directories."""
    managed = os.environ.get("PROGRAMDATA")
    if not managed:
        return POSIX_ENTERPRISE_SKILL_DIRS
    return (*POSIX_ENTERPRISE_SKILL_DIRS, Path(managed) / "ClaudeCode" / "skills")


def user_skills_dir() -> Path:
    """Return user-scoped personal skills directory path under HOME."""
    return Path.home() / ".claude" / "skills"


def inherited_skill_dirs(workdir: Path) -> tuple[Path, ...]:
    """Return ancestor dirs with .claude/skills that could leak into workdir."""
    user_scope = user_skills_dir()
    return tuple(
        found
        for parent in Path(workdir).parents
        if (found := parent / ".claude" / "skills").is_dir() and found != user_scope
    )


def _parse_assistant_event(
    event: dict[str, Any],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Extract skill invocations and reasoning/thinking blocks from an assistant event."""
    msg = event.get("message")
    if not isinstance(msg, dict):
        return (), ()
    content = msg.get("content")
    if not isinstance(content, list):
        return (), ()
    invocations: list[str] = []
    thoughts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "thinking" and (thought := block.get("thinking")):
            thoughts.append(str(thought).strip())
        elif block_type == "text" and (text := block.get("text")):
            thoughts.append(str(text).strip())
        elif block_type == "tool_use" and block.get("name") == SKILL_TOOL_NAME:
            tool_input = block.get("input")
            if isinstance(tool_input, dict) and (skill := tool_input.get("skill")):
                invocations.append(str(skill))
    return tuple(invocations), tuple(thoughts)


def _parse_init_event(
    event: dict[str, Any],
) -> tuple[tuple[str, ...], tuple[str, ...], str]:
    """Extract observed skills, tools, and model name from system init event."""
    skills = event.get("skills")
    observed = tuple(str(s) for s in skills) if isinstance(skills, list) else ()
    available = event.get("tools")
    tools = tuple(str(t) for t in available) if isinstance(available, list) else ()
    named = event.get("model")
    resolved_model = named if isinstance(named, str) and named else ""
    return observed, tools, resolved_model


class ClaudeUsage(BaseModel):
    """Represent usage metrics from Claude Code stream events."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    input_tokens: int = Field(default=0, ge=0)
    cache_creation_input_tokens: int = Field(default=0, ge=0)
    cache_read_input_tokens: int = Field(default=0, ge=0)
    prompt_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int = Field(default=0, ge=0)

    @computed_field
    @property
    def total_prompt_tokens(self) -> int | None:
        """Calculate total input/prompt tokens avoiding double-counting."""
        anthropic_input = (
            self.input_tokens + self.cache_creation_input_tokens + self.cache_read_input_tokens
        )
        if anthropic_input > 0:
            return anthropic_input
        return self.prompt_tokens


def _extract_usage_prompt_tokens(usage_obj: object) -> int | None:
    """Extract total input/prompt tokens via typed ClaudeUsage model."""
    if not isinstance(usage_obj, dict):
        return None
    try:
        usage = ClaudeUsage.model_validate(usage_obj)
        return usage.total_prompt_tokens
    except PydanticValidationError:
        return None


def _parse_result_event(
    event: dict[str, Any],
) -> tuple[str, float | None, int | None]:
    """Extract subtype, cost, and duration from result event."""
    subtype = str(event.get("subtype") or "unknown")
    raw_cost = event.get("total_cost_usd")
    cost: float | None = None
    if raw_cost is not None:
        try:
            cost = float(raw_cost)
        except (ValueError, TypeError):
            cost = None
    raw_duration = event.get("duration_ms")
    duration: int | None = None
    if raw_duration is not None:
        try:
            duration = int(raw_duration)
        except (ValueError, TypeError):
            duration = None
    return subtype, cost, duration


def parse_stream(lines: Iterable[str], early_exit: bool = False) -> StreamSummary:
    """Extract observed catalog, invoked skills, reasoning, and telemetry from events."""
    observed: tuple[str, ...] = ()
    tools: tuple[str, ...] = ()
    invocations: tuple[str, ...] = ()
    reasoning: tuple[str, ...] = ()
    resolved_model = ""
    cost: float | None = None
    duration: int | None = None
    prompt_tokens: int | None = None
    subtype: str | None = None
    retries = 0
    assistant_turns = 0

    for event in iter_json_lines(lines):
        match event.get("type"):
            case "system" if event.get("subtype") == "init":
                observed, tools, model = _parse_init_event(event)
                if model:
                    resolved_model = model
            case "system" if event.get("subtype") == "api_retry":
                retries += 1
            case "assistant":
                assistant_turns += 1
                invs, ths = _parse_assistant_event(event)
                invocations += invs
                reasoning += ths
                msg = event.get("message")
                usage = msg.get("usage") if isinstance(msg, dict) else event.get("usage")
                if (toks := _extract_usage_prompt_tokens(usage)) is not None:
                    prompt_tokens = toks
            case "result":
                subtype, cost, duration = _parse_result_event(event)
                if (toks := _extract_usage_prompt_tokens(event.get("usage"))) is not None:
                    prompt_tokens = toks

    status: SessionStatus | str | None = None
    if early_exit:
        subtype = "early_exit"
        status = SessionStatus.SUCCESS
    elif subtype:
        status = SessionStatus.SUCCESS if subtype in EXPECTED_TERMINAL_SUBTYPES else subtype

    return StreamSummary(
        observed_catalog=observed,
        observed_tools=tools,
        invoked_skills=invocations,
        early_exit=early_exit,
        turns_taken=max(1, assistant_turns),
        reasoning=reasoning,
        resolved_model=resolved_model,
        cost_usd=cost,
        duration_ms=duration,
        prompt_tokens=prompt_tokens,
        result_subtype=subtype,
        status=status,
        retries=retries,
    )


class ClaudeCodeRuntime(CliAgentRuntime[ClaudeCodeOptions]):
    """Execute agent skill selection probes via the Claude Code CLI."""

    name = "claude-code"
    options: ClaudeCodeOptions
    api_key_env_var: str | None = "ANTHROPIC_API_KEY"
    _skills_subpath = ".claude/skills"
    isolation_dir_name: ClassVar[str | None] = ".reach_claude_config"

    def __init__(
        self,
        settings: RuntimeSettings | None = None,
        options: ClaudeCodeOptions | None = None,
    ) -> None:
        """Initialize Claude runtime options and settings."""
        if options is not None:
            opts_dict = options.model_dump()
            self.settings = settings or RuntimeSettings(agent=self.name, options=opts_dict)
        else:
            self.settings = settings or RuntimeSettings(agent=self.name)
            options = ClaudeCodeOptions.model_validate(dict(self.settings.options))
        self.options = options
        self._resident: tuple[str, ...] = ()

    @override
    def skill_roots(self, workdir: Path) -> tuple[SkillRoot, ...]:
        """Return ordered tuple of skill discovery locations in precedence order."""
        here = Path(workdir).expanduser().resolve()
        candidates: list[tuple[Path, str]] = [
            *((path, "enterprise") for path in enterprise_skill_dirs()),
            (user_skills_dir(), "user"),
            (self.skills_dir(here), "project"),
            *((path, "project") for path in inherited_skill_dirs(here)),
        ]
        found: dict[Path, str] = {}
        for path, scope in candidates:
            resolved = path.expanduser()
            resolved = resolved.resolve() if resolved.is_dir() else resolved
            if resolved.is_dir():
                found.setdefault(resolved, scope)
        return tuple(
            SkillRoot(path=path, scope=scope, precedence=rank)
            for rank, (path, scope) in enumerate(found.items())
        )

    @property
    @override
    def rations_catalog(self) -> bool:
        """Return True indicating Claude Code enforces skill listing character budgets."""
        return True

    @override
    def fit(self, catalog: Catalog, skills: Iterable[Skill]) -> CatalogFit:
        """Evaluate catalog fit against CLI listing budget limits."""
        resident = resident_skills(catalog, list(skills))
        listing = fit_skill_listing(
            resident,
            budget_chars=listing_budget_chars(
                self.options.skill_listing_budget_fraction,
                model=self.options.model,
            ),
            max_desc_chars=self.options.skill_listing_max_desc_chars,
        )
        return CatalogFit(
            allowed=listing.budget_chars,
            asked=listing.full_chars,
            unit="display columns",
            truncated=len(listing.name_only),
            remedy=self._remedy(listing, self.options.model),
            elided_skills=listing.name_only,
        )

    @staticmethod
    def _remedy(listing: SkillListing, model: str) -> str:
        """Generate human-readable guidance on adjusting budget fractions."""
        if not listing.over_budget:
            return ""
        fraction = budget_fraction_for(listing.full_chars, model)
        if fraction is None:
            profile = _listing_profile(model)
            window = floor(profile.context_window * profile.chars_per_token)

            return (
                f"listing exceeds the maximum window of {window:,} columns; "
                "hold fewer skills resident, or shorten their descriptions"
            )
        return (
            f"set skill_listing_budget_fraction = {fraction} under "
            f"[runtime.options], which allows "
            f"{listing_budget_chars(fraction, model):,}"
        )

    @override
    def _validate_install(self, workdir: Path) -> None:
        """Verify workspace does not inherit skills from ancestor directories."""
        inherited = inherited_skill_dirs(workdir)
        if inherited:
            msg = (
                f"workspace {workdir} inherits skills from "
                f"{', '.join(str(p) for p in inherited)}; evaluations require an "
                "isolated catalog. Choose a workspace directory with no parent .claude/skills."
            )
            raise ValueError(msg)

    def build_command(self, query_text: str) -> list[str]:
        """Assemble command-line arguments for running a single-turn probe."""
        options = self.options
        command = [
            options.executable,
            "-p",
            query_text,
            "--model",
            options.model,
            "--output-format",
            "stream-json",
            "--verbose",
            "--max-turns",
            str(options.max_turns),
            "--setting-sources",
            options.setting_sources,
        ]
        if options.tools:
            command += ["--tools", options.tools]
        if options.strict_mcp_config:
            command.append("--strict-mcp-config")
        if options.denied_tools:
            command += ["--disallowedTools", *options.denied_tools]
        if options.exclude_dynamic_prompt:
            command.append("--exclude-dynamic-system-prompt-sections")
        if options.no_session_persistence:
            command.append("--no-session-persistence")
        inline = options.settings_json()
        if inline is not None:
            command += ["--settings", inline]
        if effort := self.effective_effort:
            command += ["--effort", effort]
        return [*command, *options.extra_args]

    def _extra_residents(self, observed: tuple[str, ...]) -> tuple[str, ...]:
        """Identify undeclared resident skills present in the observed catalog."""
        if not observed or not self._resident:
            return ()
        if not self.options.disable_bundled_skills:
            return ()
        return tuple(sorted(set(observed) - set(self._resident)))

    @override
    def parse_stream(
        self,
        lines: Iterable[str],
        resident: Sequence[str] = (),
        early_exit: bool = False,
    ) -> SessionSummary:
        """Parse CLI stdout lines into a standardized session summary."""
        del resident
        return parse_stream(lines, early_exit=early_exit)

    @override
    def extract_skills_from_line(self, line: str) -> Sequence[str]:
        """Extract invoked skill names from assistant tool calls in event line."""
        try:
            event = json.loads(line.strip())
        except (json.JSONDecodeError, UnicodeDecodeError):
            return ()
        if isinstance(event, dict) and event.get("type") == "assistant":
            invs, _ = _parse_assistant_event(event)
            return invs
        return ()

    @override
    def build_env(self, workdir: Path | None = None) -> dict[str, str]:
        """Assemble process environment with API keys and workspace directory."""
        env = super().build_env(workdir)
        sync_claude_settings_env(env, blocked_env_vars=self.blocked_env_vars)
        if self.options.cloud_ml_region:
            env["CLOUD_ML_REGION"] = self.options.cloud_ml_region
        if self.options.vertex_project_id:
            env["ANTHROPIC_VERTEX_PROJECT_ID"] = self.options.vertex_project_id
        if self.options.disable_bundled_skills:
            env["CLAUDE_CODE_DISABLE_BUNDLED_SKILLS"] = "1"
        if workdir is not None and (iso_dir := self.effective_isolation_dir(workdir)) is not None:
            config_dir = ensure_private_directory(iso_dir)
            env["CLAUDE_CONFIG_DIR"] = str(config_dir)
        return env

    @override
    def validate_outcome(
        self,
        summary: SessionSummary,
        workdir: Path,
    ) -> str | None:
        """Validate terminal subtype, tool leakage, and candidate catalog residency."""
        if err := super().validate_outcome(summary, workdir):
            return err
        subtype = getattr(summary, "result_subtype", "success")
        if subtype not in EXPECTED_TERMINAL_SUBTYPES:
            return f"runtime error: {subtype}"
        observed_catalog = getattr(summary, "observed_catalog", ())
        if extra := self._extra_residents(observed_catalog):
            return f"residency leak: {', '.join(extra)}"
        return None


class _ClaudeCompletion(BaseModel):
    """Internal model for Claude Code CLI JSON completion envelope."""

    model_config = ConfigDict(frozen=True)

    result: str
    total_cost_usd: float | None = None


class ClaudeGenerator(BaseTextGenerator[ClaudeCodeOptions]):
    """Generate text completions using the Claude Code CLI."""

    name: str = "claude-code"
    options: ClaudeCodeOptions

    def __init__(
        self,
        model: str = "",
        *,
        timeout_s: int = 300,
        options: ClaudeCodeOptions | None = None,
    ) -> None:
        """Initialize Claude generator with model, timeout, and options."""
        if isinstance(options, ClaudeCodeOptions):
            opts = options
        elif model:
            opts = ClaudeCodeOptions(model=model)
        else:
            opts = ClaudeCodeOptions()
        super().__init__(model=opts.model or model, timeout_s=timeout_s, options=opts)

    def build_completion_command(self, prompt: str = "") -> list[str]:
        """Assemble command-line arguments for text completion."""
        del prompt
        command = [
            self.options.executable,
            "-p",
            "--model",
            self.options.model or self.model,
            "--output-format",
            "json",
        ]
        if self.options.no_session_persistence:
            command.append("--no-session-persistence")
        if effort := getattr(self.options, "effort", None):
            command += ["--effort", effort]
        return [*command, *self.options.extra_args]

    def _failure_reason(self, completed: subprocess.CompletedProcess[str]) -> str:
        """Extract detailed failure reason from completion response."""
        reason = ""
        try:
            reason = _ClaudeCompletion.model_validate_json(completed.stdout).result.strip()
        except PydanticValidationError:
            reason = ""
        if not reason:
            reason = process_failure_reason(completed)
        if "too long" in reason.lower():
            reason += (
                "; the masked prompt carries every resident body, so cap what "
                "the generator is shown with --top-rivals rather than shrinking "
                "the catalog, which would change what the queries are scored "
                "against"
            )
        return reason

    @override
    def build_env(self) -> dict[str, str]:
        """Assemble process environment with API keys for Claude Code completion."""
        env = super().build_env()
        sync_claude_settings_env(env, blocked_env_vars=self.blocked_env_vars)
        if (cloud_ml_region := getattr(self.options, "cloud_ml_region", None)) is not None:
            env["CLOUD_ML_REGION"] = str(cloud_ml_region)
        if (vertex_project_id := getattr(self.options, "vertex_project_id", None)) is not None:
            env["ANTHROPIC_VERTEX_PROJECT_ID"] = str(vertex_project_id)
        if (api_key := getattr(self.options, "api_key", None)) is not None:
            env["ANTHROPIC_API_KEY"] = str(api_key)
        return env

    @override
    def complete(self, prompt: str, *, schema: str | Mapping[str, Any] | None = None) -> str:
        """Execute text completion subprocess and return response string."""
        effective_prompt = self.format_prompt_with_schema(prompt, schema)
        completed = subprocess.run(
            self.build_completion_command(effective_prompt),
            input=effective_prompt,
            capture_output=True,
            text=True,
            timeout=self.timeout_s,
            check=False,
            env=self.build_env(),
        )
        if completed.returncode != 0:
            reason = self._failure_reason(completed)
            msg = f"generation failed: {reason}"
            raise RuntimeError(msg)
        try:
            envelope = _ClaudeCompletion.model_validate_json(completed.stdout)
            self.completion_cost_usd += envelope.total_cost_usd or 0.0
            self.completions += 1
            return envelope.result
        except PydanticValidationError:
            self.completions += 1
            return completed.stdout.strip()
