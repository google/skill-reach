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

"""Drive the Google Antigravity CLI (agy) as an agent evaluation runtime."""

from __future__ import annotations

import atexit
import contextlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Self, override

from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, StringConstraints
from pydantic import ValidationError as PydanticValidationError

from reach.config import DEFAULT_GEMINI_MODEL, RuntimeSettings, resolve_path
from reach.runtime import (
    AntigravityOptions,
    AntigravityRuntime,
    CliAgentRuntime,
    CliOptions,
    SessionStatus,
    SessionSummary,
    ToolCallInfo,
    agent_default_model,
)
from reach.runtime._env import (
    detect_model_provider,
    sync_google_and_gemini_keys,
)
from reach.runtime._fs import (
    ensure_private_directory,
    resolve_skill_from_path,
)
from reach.runtime._subprocess import (
    iter_json_lines,
)
from reach.runtime.generator import BaseTextGenerator
from reach.runtime.profiles import model_profile

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence


#: Tool identifier used by JSON schema structured outputs.
FINISH_TOOL = "finish"

#: Built-in tool identifier for viewing local files.
VIEW_TOOL = "view_file"

#: Built-in tools permitted for workspace exploration across resident skills.
INSPECTION_TOOLS: frozenset[str] = frozenset({"find_by_name", "grep_search", "list_dir"})

#: Permission action patterns explicitly denied in the isolated settings file.
DENIED_PERMISSION_ACTIONS: tuple[str, ...] = (
    "command(*)",
    "write_file(*)",
    "execute_url(*)",
    "unsandboxed(*)",
    "mcp(*)",
)

#: Expected terminal execution status for successful CLI probes.
EXPECTED_STATUSES = frozenset({SessionStatus.SUCCESS})


def normalize_agy_model(model: str) -> str:
    """Normalize model slug into standard naming expected by agy CLI."""
    match = re.match(
        r"^gemini-flash-([0-9.]+)(?:-(low|medium|high))?$",
        model.strip(),
        re.IGNORECASE,
    )
    if match:
        version, tier = match.groups()
        if tier:
            return f"gemini-{version}-flash-{tier.lower()}"
        return f"gemini-{version}-flash"
    return model


def _isolated_settings_path(home_dir: Path) -> Path:
    """Return the filesystem location of the settings.json file under home_dir."""
    return home_dir / ".gemini" / "antigravity-cli" / "settings.json"


def _conversation_state_paths(home_dir: Path) -> tuple[Path, Path]:
    """Return paths to conversation history directories and databases under home_dir."""
    base = resolve_path(home_dir) / ".gemini" / "antigravity-cli"
    return (base / "conversations", base / "conversation_summaries.db")


def _ensure_isolated_settings(
    home_dir: Path,
    *,
    trust: Path | None = None,
    allow_read: Path | None = None,
    model_provider: str | None = None,
) -> None:
    """Write or update isolated permissions and workspace trusts in settings.json."""
    resolved_home = resolve_path(home_dir)
    ensure_private_directory(resolved_home)
    path = _isolated_settings_path(resolved_home)
    ensure_private_directory(path.parent)
    settings: dict[str, Any] = {}
    if path.exists():
        settings = json.loads(path.read_text())
    permissions: dict[str, Any] = {"deny": list(DENIED_PERMISSION_ACTIONS)}
    if allow_read is not None:
        permissions["allow"] = [f"read_file({resolve_path(allow_read)})"]
    settings["permissions"] = permissions
    if model_provider is not None:
        settings["modelProvider"] = model_provider
    else:
        settings.pop("modelProvider", None)
    if trust is not None:
        settings["trustedWorkspaces"] = [str(resolve_path(trust))]
    path.write_text(json.dumps(settings, sort_keys=True, indent=2))


class AntigravityCliOptions(AntigravityOptions, CliOptions):
    """Specify runtime configuration options for the Antigravity CLI agent."""

    executable: str = "agy"
    model: str = Field(
        default_factory=lambda: agent_default_model("antigravity-cli") or DEFAULT_GEMINI_MODEL,
    )
    home_dir: Path | None = None
    isolation_dir_field: ClassVar[str | None] = "home_dir"
    model_provider: str | None = None
    disable_slash_commands: bool = True
    dangerously_skip_permissions: bool = True
    print_timeout: str | None = None
    go_max_procs: int = 4


class AntigravityUsage(BaseModel):
    """Represent token usage payload emitted by Antigravity CLI (agy)."""

    model_config = ConfigDict(extra="ignore")

    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)

    @property
    def prompt_tokens(self) -> int | None:
        """Return input prompt tokens."""
        return self.input_tokens


def _extract_usage_prompt_tokens(usage_obj: object) -> int | None:
    """Extract input prompt tokens from usage object via AntigravityUsage schema."""
    if not isinstance(usage_obj, dict):
        return None
    try:
        usage = AntigravityUsage.model_validate(usage_obj)
        return usage.prompt_tokens
    except PydanticValidationError:
        return None


class ToolAttempt(BaseModel):
    """Record an observed tool invocation attempt during query execution."""

    model_config = ConfigDict(frozen=True)

    name: str
    path: str | None = None


class StreamSummary(SessionSummary):
    """Aggregate tool attempts and outcome metrics from streaming event logs."""

    @property
    def tool_attempts(self) -> tuple[ToolAttempt, ...]:
        """Return tool attempts derived from structured tool calls."""
        return tuple(ToolAttempt(name=c.name, path=c.target_path) for c in self.tool_calls)


def _is_inspection_tool_allowed(attempt: ToolAttempt | ToolCallInfo, workdir: Path | None) -> bool:
    """Determine whether an inspection tool path complies with workspace sandbox."""
    if workdir is None:
        return False
    if attempt.path is None:
        return True
    resolved = resolve_path(attempt.path)
    return resolved == workdir or resolved.is_relative_to(workdir)


def _is_tool_allowed(
    attempt: ToolAttempt | ToolCallInfo,
    resident_bases: Sequence[Path],
    workdir: Path | None,
    allowed_tools: frozenset[str],
) -> bool:
    """Determine whether a single tool attempt complies with sandbox policies."""
    if attempt.name not in allowed_tools:
        return False

    if attempt.name == VIEW_TOOL:
        if attempt.path is None:
            return False
        resolved = resolve_path(attempt.path)
        return any(resolved.is_relative_to(base) for base in resident_bases)

    if attempt.name in INSPECTION_TOOLS:
        return _is_inspection_tool_allowed(attempt, workdir)

    return True


def _leaked_tools(
    attempts: Iterable[ToolAttempt | ToolCallInfo],
    resident_dirs: Iterable[Path | str],
    workdir: Path | str | None = None,
    allowed_tools: Iterable[str] | None = None,
) -> tuple[str, ...]:
    """Identify tool calls that violate sandbox isolation boundaries."""
    if allowed_tools is None:
        return ()

    resolved_workdir = resolve_path(workdir) if workdir is not None else None
    resolved_bases = tuple(
        p.parent if p.name == "SKILL.md" else p for p in (resolve_path(r) for r in resident_dirs)
    )
    allowed_set = frozenset(allowed_tools)
    leaked = {
        attempt.name
        for attempt in attempts
        if not _is_tool_allowed(attempt, resolved_bases, resolved_workdir, allowed_set)
    }
    return tuple(sorted(leaked))


def _extract_init_model(event: dict[str, Any]) -> str | None:
    """Extract model identifier from an init event if present."""
    init = event.get("init") or {}
    named = init.get("model")
    return named if isinstance(named, str) and named else None


def _extract_tool_attempt(event: dict[str, Any]) -> ToolCallInfo | None:
    """Extract tool attempt and target path from a step_update event if present."""
    if not isinstance(event, dict):
        return None
    step = event.get("step_update")
    if not isinstance(step, dict) or step.get("step_type") != "tool":
        return None
    name = step.get("tool_name")
    if not isinstance(name, str) or not name:
        return None
    info = step.get("tool_info")
    params = info.get("parameters") if isinstance(info, dict) else {}
    return ToolCallInfo(name=name, parameters=params if isinstance(params, dict) else {})


def _extract_step_thought(event: dict[str, Any]) -> str | None:
    """Extract reasoning or thought text from a step_update event if present."""
    if not isinstance(event, dict):
        return None
    step = event.get("step_update")
    if not isinstance(step, dict):
        return None
    if thought := step.get("thought"):
        return str(thought).strip()
    if step.get("step_type") in ("thought", "thinking") and (
        text := step.get("content") or step.get("text")
    ):
        return str(text).strip()
    if (
        step.get("step_type") == "agent_response"
        and step.get("state") == "DONE"
        and (text := step.get("text_delta"))
    ):
        s = str(text).strip()
        if s and not (s.startswith("{") and s.endswith("}")):
            return s
    return None


StrippedStr = Annotated[str, StringConstraints(strip_whitespace=True)]


class _AgyResultEvent(BaseModel):
    """Represent parsed outcome metrics and status from a CLI result event."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    status: str = "unknown"
    duration_ms: NonNegativeInt | None = None
    selected_skill: str | None = None
    reasoning: StrippedStr | None = None
    error: StrippedStr | None = None
    prompt_tokens: NonNegativeInt | None = None


def _extract_result_event(event: dict[str, Any]) -> _AgyResultEvent:
    """Parse status, duration, skill, reasoning, error, and tokens from a result event."""
    result = event.get("result") or {}
    status = str(result.get("status") or "unknown")
    raw_duration = result.get("duration_seconds")
    duration_ms = (
        int(raw_duration * 1000)
        if isinstance(raw_duration, int | float) and raw_duration >= 0
        else None
    )
    structured = result.get("structured_output")
    invoked = structured.get("selected_skill") if isinstance(structured, dict) else None
    reasoning = structured.get("reasoning") if isinstance(structured, dict) else None
    error = result.get("error")
    error_str = str(error).strip() if error else None
    reasoning_str = str(reasoning).strip() if reasoning else None
    prompt_tokens = _extract_usage_prompt_tokens(result.get("usage"))
    return _AgyResultEvent(
        status=status,
        duration_ms=duration_ms,
        selected_skill=str(invoked) if invoked else None,
        reasoning=reasoning_str or None,
        error=error_str or None,
        prompt_tokens=prompt_tokens,
    )


def _handle_step_update(
    event: dict[str, Any],
    attempts: dict[tuple[str, str | None], ToolCallInfo],
    invoked_skills: list[str],
    reasoning: list[str],
    resident: Iterable[str],
) -> int | None:
    """Process a step_update event and return extracted prompt tokens if present."""
    prompt_tokens: int | None = None
    step_update = event.get("step_update")
    if isinstance(step_update, dict):
        prompt_tokens = _extract_usage_prompt_tokens(step_update.get("usage"))
    if attempt := _extract_tool_attempt(event):
        attempts[(attempt.name, attempt.path)] = attempt
        if attempt.path:
            detected = resolve_skill_from_path(attempt.path, resident)
            if detected and (not invoked_skills or invoked_skills[-1] != detected):
                invoked_skills.append(detected)
    if thought := _extract_step_thought(event):
        reasoning.append(thought)
    return prompt_tokens


def parse_stream(
    lines: Iterable[str],
    resident: Iterable[str] = (),
    early_exit: bool = False,
) -> StreamSummary:
    """Parse streaming event lines into a StreamSummary model."""
    attempts: dict[tuple[str, str | None], ToolCallInfo] = {}
    invoked_skills: list[str] = []
    reasoning: list[str] = []
    resolved_model = ""
    duration_ms: int | None = None
    prompt_tokens: int | None = None
    status: str | None = None
    result_error: str | None = None
    step_turns = 0

    for event in iter_json_lines(lines):
        match event.get("event"):
            case "init":
                if model_name := _extract_init_model(event):
                    resolved_model = model_name
            case "step_update":
                step_turns += 1
                if (
                    toks := _handle_step_update(
                        event, attempts, invoked_skills, reasoning, resident
                    )
                ) is not None:
                    prompt_tokens = toks
            case "result":
                res_event = _extract_result_event(event)
                status = res_event.status
                duration_ms = res_event.duration_ms
                result_error = res_event.error
                if res_event.prompt_tokens is not None:
                    prompt_tokens = res_event.prompt_tokens
                if res_event.selected_skill and (
                    not invoked_skills or invoked_skills[-1] != res_event.selected_skill
                ):
                    invoked_skills.append(res_event.selected_skill)
                if res_event.reasoning and res_event.reasoning not in reasoning:
                    reasoning.append(res_event.reasoning)

    if early_exit:
        status = SessionStatus.SUCCESS
        result_error = None

    tool_calls = tuple(attempts.values())
    return StreamSummary(
        tool_calls=tool_calls,
        invoked_skills=tuple(invoked_skills),
        early_exit=early_exit,
        turns_taken=max(1, step_turns),
        reasoning=tuple(reasoning),
        resolved_model=resolved_model,
        duration_ms=duration_ms,
        prompt_tokens=prompt_tokens,
        status=status,
        error=result_error,
    )


class AntigravityCliRuntime(CliAgentRuntime[AntigravityCliOptions], AntigravityRuntime):
    """Drive Antigravity CLI (agy) subprocess for skill selection probes."""

    name = "antigravity-cli"
    options: AntigravityCliOptions
    api_key_env_var: str | None = "GEMINI_API_KEY"
    _skills_subpath = ".agents/skills"

    def __init__(
        self,
        settings: RuntimeSettings | None = None,
        options: AntigravityCliOptions | None = None,
    ) -> None:
        """Initialize agent settings and isolated configuration directory."""
        if options is not None:
            opts_dict = options.model_dump()
            self.settings = settings or RuntimeSettings(agent=self.name, options=opts_dict)
        else:
            self.settings = settings or RuntimeSettings(agent=self.name)
            options = AntigravityCliOptions.model_validate(dict(self.settings.options))
        self._temp_home = False
        if options.home_dir is None:
            home_dir = Path(tempfile.mkdtemp(prefix="reach-agy-home-"))
            options = options.model_copy(update={"home_dir": home_dir})
            self._temp_home = True
            atexit.register(self.cleanup)
        self.options = options
        self._resident: tuple[str, ...] = ()
        _ensure_isolated_settings(
            self.home_dir,
            model_provider=self.effective_model_provider,
        )

    @override
    def clone_isolated(self) -> Self:
        """Create a thread-local isolated clone with a dedicated temporary home directory."""
        cloned_options = self.options.model_copy(update={"home_dir": None})
        return type(self)(settings=self.settings, options=cloned_options)

    def cleanup(self) -> None:
        """Remove isolated temporary home directory if automatically created."""
        if getattr(self, "_temp_home", False) and self.options.home_dir is not None:
            if self.options.home_dir.is_dir():
                shutil.rmtree(self.options.home_dir, ignore_errors=True)
            self._temp_home = False
            with contextlib.suppress(Exception):
                atexit.unregister(self.cleanup)

    def __del__(self) -> None:
        """Clean up resources when garbage collected."""
        self.cleanup()

    def __enter__(self) -> Self:
        """Enter runtime context."""
        return self

    def __exit__(self, *args: object) -> None:
        """Exit runtime context and clean up resources."""
        self.cleanup()

    @property
    def home_dir(self) -> Path:
        """Return the guaranteed isolated home directory path."""
        if self.options.home_dir is None:
            msg = "Isolated home directory has not been configured"
            raise ValueError(msg)
        return self.options.home_dir

    @override
    def _post_install(self, workdir: Path) -> None:
        """Grant required filesystem permissions and workspace trusts."""
        _ensure_isolated_settings(
            self.home_dir,
            trust=workdir,
            allow_read=self.skills_dir(workdir),
            model_provider=self.effective_model_provider,
        )

    def clean_conversation_state(self) -> None:
        """Purge persisted conversation records and SQLite summaries."""
        conversations, summaries_db = _conversation_state_paths(self.home_dir)
        if conversations.is_dir():
            shutil.rmtree(conversations)
        if summaries_db.exists():
            summaries_db.unlink()

    @property
    def normalized_model(self) -> str:
        """Return model identifier normalized for agy CLI compatibility."""
        return normalize_agy_model(self.options.model)

    def build_command(self, query_text: str) -> list[str]:
        """Assemble command-line arguments for executing a probe."""
        options = self.options
        command = [
            options.executable,
            "-p",
            query_text,
            "--model",
            self.normalized_model,
            "--output-format",
            "stream-json",
            "--new-project",
        ]
        if options.dangerously_skip_permissions:
            command.append("--dangerously-skip-permissions")
        if options.disable_slash_commands:
            command.append("--disable-slash-commands")
        timeout_s = self.timeout_s or 200
        timeout_val = options.print_timeout or f"{round(timeout_s)}s"
        command += ["--print-timeout", timeout_val]
        if options.json_schema:
            command += [
                "--json-schema",
                options.json_schema,
            ]
        if self.effective_effort:
            command += ["--effort", self.effective_effort]
        return [*command, *options.extra_args]

    @override
    def parse_stream(
        self,
        lines: Iterable[str],
        resident: Sequence[str] = (),
        early_exit: bool = False,
    ) -> SessionSummary:
        """Parse CLI stdout lines into a standardized session summary."""
        return parse_stream(lines, resident=resident or self._resident, early_exit=early_exit)

    @override
    def extract_skill_from_line(self, line: str) -> str | None:
        """Extract invoked skill name from an event line for early-exit detection."""
        try:
            event = json.loads(line.strip())
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None
        if not isinstance(event, dict):
            return None
        if (attempt := _extract_tool_attempt(event)) and attempt.path:
            return resolve_skill_from_path(attempt.path, self._resident)
        if event.get("event") == "result":
            res_event = _extract_result_event(event)
            invoked = res_event.selected_skill
            if invoked and (not self._resident or invoked in self._resident):
                return invoked
        return None

    @override
    def build_env(self, workdir: Path | None = None) -> dict[str, str]:
        """Assemble process environment with API keys and isolated home directory."""
        env = super().build_env(workdir)
        key = self.effective_api_key
        if key:
            env["GEMINI_API_KEY"] = key
            env["GOOGLE_API_KEY"] = key
        sync_google_and_gemini_keys(env)
        env["GOMAXPROCS"] = str(self.options.go_max_procs)
        return env

    @override
    def validate_outcome(
        self,
        summary: SessionSummary,
        workdir: Path,
    ) -> str | None:
        """Validate status and tool path isolation against workspace sandbox."""
        if err := super().validate_outcome(summary, workdir):
            return err

        allowed = self.allowed_tools
        leaked = _leaked_tools(
            summary.tool_calls,
            self.resident_skill_paths(workdir),
            workdir=resolve_path(workdir),
            allowed_tools=allowed,
        )
        if leaked:
            return f"tool leak: {', '.join(leaked)}"
        return None

    @override
    def post_probe(self, workdir: Path) -> None:
        """Clean conversation state after probe execution if auto_clean is enabled."""
        del workdir
        if self.options.auto_clean:
            self.clean_conversation_state()


class _AntigravityCompletion(BaseModel):
    """Internal model for Antigravity CLI JSON completion envelope."""

    model_config = ConfigDict(frozen=True)

    response: str


class AntigravityCliGenerator(BaseTextGenerator[AntigravityCliOptions]):
    """Generate text completions using the Antigravity CLI."""

    name: str = "antigravity-cli"
    options: AntigravityCliOptions

    def __init__(
        self,
        model: str = "",
        *,
        timeout_s: int = 300,
        options: AntigravityCliOptions | None = None,
    ) -> None:
        """Initialize Antigravity CLI generator with model, timeout, and options."""
        if isinstance(options, AntigravityCliOptions):
            opts = options
        elif model:
            opts = AntigravityCliOptions(model=model)
        else:
            opts = AntigravityCliOptions()
        super().__init__(model=opts.model or model, timeout_s=timeout_s, options=opts)
        self.home_dir = opts.home_dir or Path(tempfile.mkdtemp(prefix="reach-agy-draft-"))
        self._owns_home_dir = opts.home_dir is None
        if self._owns_home_dir:
            atexit.register(self.cleanup)
        _ensure_isolated_settings(
            self.home_dir,
            model_provider=self.effective_model_provider,
        )

    def cleanup(self) -> None:
        """Clean temporary home directory if created by this generator instance."""
        if getattr(self, "_owns_home_dir", False) and self.home_dir.exists():
            shutil.rmtree(self.home_dir, ignore_errors=True)
            self._owns_home_dir = False
            with contextlib.suppress(Exception):
                atexit.unregister(self.cleanup)

    def __del__(self) -> None:
        """Clean temporary home directory if created by this generator instance."""
        self.cleanup()

    @property
    def effective_model_provider(self) -> str | None:
        """Return configured model_provider or auto-detect 'gemini' when API keys are present."""
        return detect_model_provider(
            self.model,
            getattr(self.options, "model_provider", None),
            api_key=self.options.api_key,
        )

    @property
    def effective_api_key(self) -> str | None:
        """Return configured API key or probe environment for fallback."""
        return (
            str(self.options.api_key)
            if self.options.api_key
            else os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        )

    @override
    def build_env(self) -> dict[str, str]:
        """Assemble process environment with API keys and isolated home directory."""
        env = super().build_env()
        env["HOME"] = str(self.home_dir)
        key = self.effective_api_key
        if key:
            env["GEMINI_API_KEY"] = key
            env["GOOGLE_API_KEY"] = key
        sync_google_and_gemini_keys(env)
        env["GOMAXPROCS"] = str(self.options.go_max_procs)
        return env

    @property
    def normalized_model(self) -> str:
        """Map canonical models to Antigravity CLI naming conventions."""
        return normalize_agy_model(self.model)

    @property
    def effective_effort(self) -> str | None:
        """Return configured reasoning effort or default from model profile."""
        if self.options.effort:
            return None if self.options.effort.lower() in ("none", "off") else self.options.effort
        try:
            return model_profile(self.model).effort
        except (KeyError, ValueError):
            return None

    def build_completion_command(
        self,
        prompt: str = "",
        *,
        schema: str | None = None,
    ) -> list[str]:
        """Assemble command-line arguments for raw text completion."""
        del prompt  # Prompt is passed via stdin to avoid Linux MAX_ARG_STRLEN limits
        options = self.options
        cmd = [
            options.executable,
            "--model",
            self.normalized_model,
            "--output-format",
            "json",
            "--new-project",
        ]
        if options.dangerously_skip_permissions:
            cmd.append("--dangerously-skip-permissions")
        if options.disable_slash_commands:
            cmd.append("--disable-slash-commands")
        timeout_s = self.timeout_s or 200
        timeout_val = options.print_timeout or f"{round(timeout_s)}s"
        cmd += ["--print-timeout", timeout_val]
        effective_schema = schema if schema is not None else options.json_schema
        if effective_schema:
            cmd += ["--json-schema", effective_schema]
        if self.effective_effort:
            cmd += ["--effort", self.effective_effort]
        return [*cmd, *options.extra_args]

    @override
    def complete(
        self,
        prompt: str,
        *,
        schema: str | Mapping[str, Any] | None = None,
    ) -> str:
        """Execute text completion subprocess and return response string."""
        schema_str = self.resolve_schema(schema)
        completed = subprocess.run(
            self.build_completion_command(prompt, schema=schema_str),
            input=prompt,
            capture_output=True,
            text=True,
            timeout=self.timeout_s,
            check=False,
            env=self.build_env(),
        )
        if completed.returncode != 0:
            reason = completed.stderr.strip()
            if not reason and completed.stdout:
                try:
                    data = json.loads(completed.stdout)
                    reason = data.get("error", "") if isinstance(data, dict) else ""
                except (json.JSONDecodeError, UnicodeDecodeError):
                    reason = ""
            reason = reason or f"exit code {completed.returncode}"
            msg = f"generation failed: {reason}"
            raise RuntimeError(msg)
        try:
            envelope = _AntigravityCompletion.model_validate_json(completed.stdout)
            self.completions += 1
            return envelope.response
        except PydanticValidationError:
            self.completions += 1
            return completed.stdout.strip()
