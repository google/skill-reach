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
import shutil
import subprocess
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Self, override

from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, PositiveInt
from pydantic import ValidationError as PydanticValidationError

from reach.config import (
    DEFAULT_GEMINI_MODEL,
    RuntimeSettings,
    StrippedStr,
    resolve_path,
)
from reach.registry import find_adc_path
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
    DEFAULT_BLOCKED_ENV_VARS,
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


def _isolated_settings_path(home_dir: Path) -> Path:
    """Return the filesystem location of the settings.json file under home_dir."""
    return home_dir / ".gemini" / "antigravity-cli" / "settings.json"


def _isolated_adc_path(home_dir: Path) -> Path:
    """Return the filesystem location of the ADC credentials file under home_dir."""
    return resolve_path(home_dir) / ".config" / "gcloud" / "application_default_credentials.json"


def _provision_isolated_adc(
    home_dir: Path,
    *,
    project_override: str | None = None,
) -> Path | None:
    """Copy host ADC file into isolated home_dir with 0600 permissions, if available."""
    src = find_adc_path()
    if src is None or not src.is_file():
        return None
    dst = _isolated_adc_path(home_dir)
    ensure_private_directory(dst.parent)
    if resolve_path(src) != dst or project_override:
        raw_bytes = src.read_bytes()
        if project_override:
            try:
                payload = json.loads(raw_bytes.decode("utf-8"))
                if isinstance(payload, dict):
                    payload["quota_project_id"] = project_override
                    raw_bytes = json.dumps(payload, indent=2).encode("utf-8")
            except (UnicodeDecodeError, json.JSONDecodeError):
                pass
        fd = os.open(dst, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "wb") as handle:
            handle.write(raw_bytes)
    dst.chmod(0o600)
    return dst


def _unlink_isolated_adc(home_dir: Path | None) -> None:
    """Remove isolated ADC file from home_dir if present."""
    if home_dir is None:
        return
    adc_file = _isolated_adc_path(home_dir)
    with contextlib.suppress(OSError):
        if adc_file.is_file() or adc_file.is_symlink():
            adc_file.unlink()


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
    permissions: dict[str, Any] = {
        "deny": [*DENIED_PERMISSION_ACTIONS, f"read_file({resolved_home / '.config'})"]
    }
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
    disable_slash_commands: bool = True
    dangerously_skip_permissions: bool = True
    print_timeout: StrippedStr | None = None
    go_max_procs: PositiveInt = 4

    def common_cli_args(
        self,
        *,
        timeout_s: int | None,
        effort: str | None,
        schema: str | None = None,
    ) -> list[str]:
        """Assemble shared CLI flags for Antigravity CLI runtime and generator."""
        args: list[str] = []
        if self.dangerously_skip_permissions:
            args.append("--dangerously-skip-permissions")
        if self.disable_slash_commands:
            args.append("--disable-slash-commands")
        effective_timeout = timeout_s or 200
        timeout_val = self.print_timeout or f"{round(effective_timeout)}s"
        args += ["--print-timeout", timeout_val]
        effective_schema = schema if schema is not None else self.json_schema
        if effective_schema:
            args += ["--json-schema", effective_schema]
        if effort:
            args += ["--effort", effort]
        return [*args, *self.extra_args]


def _apply_agy_cli_env(
    env: dict[str, str],
    options: AntigravityCliOptions,
    home_dir: Path,
    blocked_env_vars: Iterable[str] | None = None,
) -> dict[str, str]:
    """Apply Antigravity CLI environment variables, ADC credentials, and API keys."""
    env["HOME"] = str(home_dir)
    explicit_blocked = set(blocked_env_vars or ())
    if options.effective_vertex:
        env["AGY_ADC_AUTH"] = "true"
        if proj := options.effective_project:
            env["GOOGLE_CLOUD_PROJECT"] = proj
            env["GOOGLE_CLOUD_QUOTA_PROJECT"] = proj
        if loc := options.effective_location:
            env["GOOGLE_CLOUD_LOCATION"] = loc
        if "GOOGLE_APPLICATION_CREDENTIALS" not in explicit_blocked:
            isolated_adc = _provision_isolated_adc(
                home_dir,
                project_override=options.effective_project,
            )
            if isolated_adc is not None:
                env["GOOGLE_APPLICATION_CREDENTIALS"] = str(isolated_adc)
        else:
            env.pop("GOOGLE_APPLICATION_CREDENTIALS", None)
        if not options.api_key:
            env.pop("GEMINI_API_KEY", None)
            env.pop("GOOGLE_API_KEY", None)
    else:
        env.pop("AGY_ADC_AUTH", None)
        effective_blocked = (
            explicit_blocked if blocked_env_vars is not None else DEFAULT_BLOCKED_ENV_VARS
        )
        if "GOOGLE_APPLICATION_CREDENTIALS" in effective_blocked:
            env.pop("GOOGLE_APPLICATION_CREDENTIALS", None)
    if key := options.effective_api_key:
        env["GEMINI_API_KEY"] = key
        env["GOOGLE_API_KEY"] = key
    sync_google_and_gemini_keys(env)
    env["GOMAXPROCS"] = str(options.go_max_procs)
    return env


class AntigravityUsage(BaseModel):
    """Represent token usage payload emitted by Antigravity CLI (agy)."""

    model_config = ConfigDict(extra="ignore")

    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)


def _extract_usage_prompt_tokens(usage_obj: object) -> int | None:
    """Extract input prompt tokens from usage object via AntigravityUsage schema."""
    if not isinstance(usage_obj, dict):
        return None
    try:
        usage = AntigravityUsage.model_validate(usage_obj)
        return usage.input_tokens
    except PydanticValidationError:
        return None


_SENSITIVE_CREDENTIAL_PATH_PARTS: frozenset[str] = frozenset(
    {
        "application_default_credentials.json",
        ".gcloud",
        ".config",
    }
)


def _is_sensitive_credential_path(path_str: str | None) -> bool:
    """Return True if a raw or resolved path references sensitive credential locations."""
    if not path_str:
        return False
    raw_parts = set(Path(path_str).parts)
    resolved_parts = set(resolve_path(path_str).parts)
    return bool((raw_parts | resolved_parts) & _SENSITIVE_CREDENTIAL_PATH_PARTS)


def _is_inspection_tool_allowed(attempt: ToolCallInfo, workdir: Path | None) -> bool:
    """Determine whether an inspection tool path complies with workspace sandbox."""
    if workdir is None:
        return False
    if attempt.path is None:
        return True
    resolved = resolve_path(attempt.path)
    return resolved == workdir or resolved.is_relative_to(workdir)


def _is_tool_allowed(
    attempt: ToolCallInfo,
    resident_bases: Sequence[Path],
    workdir: Path | None,
    allowed_tools: frozenset[str],
) -> bool:
    """Determine whether a single tool attempt complies with sandbox policies."""
    if attempt.name not in allowed_tools:
        return False
    if _is_sensitive_credential_path(attempt.path):
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
    attempts: Iterable[ToolCallInfo],
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
) -> SessionSummary:
    """Parse streaming event lines into a SessionSummary model."""
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
    return SessionSummary(
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


class _IsolatedHomeMixin:
    """Manage an isolated Antigravity CLI home directory and its lifecycle."""

    options: AntigravityCliOptions
    model: str
    _owns_home_dir: bool = False

    def _init_isolated_home(self, prefix: str) -> None:
        """Allocate an isolated home directory if unset and write initial settings.json."""
        self._owns_home_dir = self.options.home_dir is None
        if self._owns_home_dir:
            home_dir = Path(tempfile.mkdtemp(prefix=prefix))
            self.options = self.options.model_copy(update={"home_dir": home_dir})
            atexit.register(self.cleanup)
        _ensure_isolated_settings(
            self.home_dir,
            model_provider=self.options.resolve_model_provider(self.model),
        )

    @property
    def home_dir(self) -> Path:
        """Return the guaranteed isolated home directory path."""
        if self.options.home_dir is None:
            msg = "Isolated home directory has not been configured"
            raise ValueError(msg)
        return self.options.home_dir

    def cleanup(self) -> None:
        """Remove isolated ADC credentials and temporary home directory if owned."""
        opts = getattr(self, "options", None)
        home_dir = opts.home_dir if opts is not None else None
        _unlink_isolated_adc(home_dir)
        if getattr(self, "_owns_home_dir", False) and home_dir is not None:
            if home_dir.is_dir():
                shutil.rmtree(home_dir, ignore_errors=True)
            self._owns_home_dir = False
            with contextlib.suppress(Exception):
                atexit.unregister(self.cleanup)

    def __del__(self) -> None:
        """Clean up resources when garbage collected."""
        self.cleanup()


class AntigravityCliRuntime(
    _IsolatedHomeMixin,
    CliAgentRuntime[AntigravityCliOptions],
    AntigravityRuntime[AntigravityCliOptions],
):
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
        super().__init__(settings=settings, options=options)
        self._init_isolated_home("reach-agy-home-")

    @override
    def clone_isolated(self) -> Self:
        """Create a thread-local isolated clone with a dedicated temporary home directory."""
        cloned_options = self.options.model_copy(update={"home_dir": None})
        return type(self)(settings=self.settings, options=cloned_options)

    def __enter__(self) -> Self:
        """Enter runtime context."""
        return self

    def __exit__(self, *args: object) -> None:
        """Exit runtime context and clean up resources."""
        self.cleanup()

    @override
    def _post_install(self, workdir: Path) -> None:
        """Grant required filesystem permissions and workspace trusts."""
        skills_root = resolve_path(self.skills_dir(workdir))
        resolved_home = resolve_path(self.home_dir)
        if resolved_home == skills_root or resolved_home.is_relative_to(skills_root):
            msg = "home_dir must not reside inside workspace skills_dir"
            raise ValueError(msg)
        _ensure_isolated_settings(
            self.home_dir,
            trust=workdir,
            allow_read=skills_root,
            model_provider=self.options.resolve_model_provider(self.model),
        )

    def clean_conversation_state(self) -> None:
        """Purge persisted conversation records, SQLite summaries, and isolated ADC file."""
        _unlink_isolated_adc(self.home_dir)
        conversations, summaries_db = _conversation_state_paths(self.home_dir)
        if conversations.is_dir():
            shutil.rmtree(conversations)
        if summaries_db.exists():
            summaries_db.unlink()

    def build_command(self, query_text: str) -> list[str]:
        """Assemble command-line arguments for executing a probe."""
        options = self.options
        return [
            options.executable,
            "-p",
            query_text,
            "--model",
            self.model,
            "--output-format",
            "stream-json",
            "--new-project",
            *options.common_cli_args(
                timeout_s=self.timeout_s,
                effort=self.effective_effort,
            ),
        ]

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
        """Assemble process environment with API keys, ADC credentials, and isolated home."""
        return _apply_agy_cli_env(
            super().build_env(workdir),
            self.options,
            self.home_dir,
            self.blocked_env_vars,
        )

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


class AntigravityCliGenerator(_IsolatedHomeMixin, BaseTextGenerator[AntigravityCliOptions]):
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
        self._init_isolated_home("reach-agy-draft-")

    @override
    def build_env(self) -> dict[str, str]:
        """Assemble process environment with API keys, ADC credentials, and isolated home."""
        return _apply_agy_cli_env(
            super().build_env(),
            self.options,
            self.home_dir,
            self.blocked_env_vars,
        )

    def build_completion_command(
        self,
        prompt: str = "",
        *,
        schema: str | None = None,
    ) -> list[str]:
        """Assemble command-line arguments for raw text completion."""
        del prompt  # Prompt is passed via stdin to avoid Linux MAX_ARG_STRLEN limits
        options = self.options
        return [
            options.executable,
            "--model",
            self.model,
            "--output-format",
            "json",
            "--new-project",
            *options.common_cli_args(
                timeout_s=self.timeout_s,
                effort=self.effective_effort,
                schema=schema,
            ),
        ]

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
