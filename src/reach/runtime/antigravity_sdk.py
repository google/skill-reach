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

"""Drive the Google Antigravity Python SDK as an agent evaluation runtime."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import re
import threading
from collections.abc import Collection, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, cast, override

if TYPE_CHECKING:
    from collections.abc import Coroutine, Mapping

    from google.antigravity import Agent, LocalAgentConfig
    from google.antigravity import hooks as ag_hooks
    from google.antigravity import types as ag_types
    from google.antigravity.types import AntigravityValidationError

    _HAS_ANTIGRAVITY = True
else:
    try:
        from google.antigravity import Agent, LocalAgentConfig
        from google.antigravity import hooks as ag_hooks
        from google.antigravity import types as ag_types
        from google.antigravity.types import AntigravityValidationError

        _HAS_ANTIGRAVITY = True
    except ImportError:
        Agent = None
        LocalAgentConfig = None
        ag_hooks = None
        ag_types = None

        class AntigravityValidationError(Exception):
            """Stub exception when google-antigravity is not installed."""

        _HAS_ANTIGRAVITY = False

from pydantic import BaseModel, Field

from reach.config import DEFAULT_GEMINI_MODEL, RuntimeSettings
from reach.runtime import (
    AntigravityOptions,
    AntigravityRuntime,
    SelectionOutcome,
    TrajectoryTracker,
    agent_default_model,
)
from reach.runtime._env import (
    raise_missing_agent_dependency,
    sync_google_and_gemini_keys,
)
from reach.runtime._fs import (
    ensure_private_directory,
    extract_tool_path,
    normalize_skill_tool_args,
    probe_slot_dir,
    resolve_skill_from_path,
    safe_cleanup_isolated_dir,
)
from reach.runtime._subprocess import check_tool_leak
from reach.runtime.generator import BaseTextGenerator
from reach.runtime.profiles import model_profile

#: Selection tool set configured for single-turn probe evaluations.
SELECTION_TOOLS: tuple[Any, ...] = (
    (ag_types.BuiltinTools.FINISH,) if ag_types is not None else ("finish",)
)


def _build_multi_turn_selection_tools() -> tuple[Any, ...]:
    """Assemble available read-only workspace inspection tools."""
    if ag_types is None or not hasattr(ag_types, "BuiltinTools"):
        return ("view_file", "list_dir", "grep_search", "find_by_name")
    tools: list[Any] = []
    for attr in ("VIEW_FILE", "LIST_DIR", "GREP_SEARCH", "FIND_BY_NAME"):
        member = getattr(ag_types.BuiltinTools, attr, None)
        if member is not None:
            tools.append(member)
    if not tools:
        tools.append("view_file")
    return tuple(tools)


#: Selection tool set configured for multi-turn trajectory probe evaluations.
MULTI_TURN_SELECTION_TOOLS: tuple[Any, ...] = _build_multi_turn_selection_tools()


def _build_expected_stop_reasons() -> frozenset[Any]:
    """Construct expected terminal stop reasons across SDK versions."""
    reasons: set[Any] = {"UNSPECIFIED", "MAX_MODEL_CALLS_EXCEEDED", "END_TURN"}
    stop_reason_cls = getattr(ag_types, "StopReason", None) if ag_types is not None else None
    if stop_reason_cls is not None:
        for attr_name in ("UNSPECIFIED", "MAX_MODEL_CALLS_EXCEEDED", "END_TURN"):
            if (member := getattr(stop_reason_cls, attr_name, None)) is not None:
                reasons.add(member)
    return frozenset(reasons)


#: Terminal stop reasons considered normal for single-turn and multi-turn evaluations.
EXPECTED_STOP_REASONS: frozenset[Any] = _build_expected_stop_reasons()


def _run_sync[T](coro: Coroutine[Any, Any, T]) -> T:
    """Run an async coroutine synchronously, handling existing active event loops."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        with asyncio.Runner() as runner:
            return runner.run(coro)
    with ThreadPoolExecutor(max_workers=1) as executor:
        # pyrefly: ignore[redundant-cast]
        return cast("T", executor.submit(asyncio.run, coro).result())


def _tool_name(name: ag_types.BuiltinTools | str) -> str:
    """Extract the string name of a tool from an enum or raw string."""
    return (
        name.value
        if ag_types is not None and isinstance(name, ag_types.BuiltinTools)
        else str(name)
    )


def _model_supports_thinking(model: str) -> bool:
    """Return whether a model supports reasoning effort or thinking level."""
    lower = model.lower()
    return not any(p in lower for p in ("gemini-2.5", "gemini-2.0", "gemini-1.5", "gemini-1.0"))


def _build_model_spec(
    model: str,
    effort: str | None = None,
    *,
    vertex: bool = False,
    project: str | None = None,
    location: str | None = None,
    api_key: str | None = None,
) -> str | ag_types.ModelTarget:
    """Construct model target with reasoning effort endpoint options when configured."""
    valid_effort = effort if (effort and _model_supports_thinking(model)) else None
    if ag_types is not None and (valid_effort or vertex):
        options = ag_types.GeminiModelOptions(thinking_level=valid_effort) if valid_effort else None
        endpoint = (
            ag_types.VertexEndpoint(
                project=project,
                location=location,
                api_key=api_key,
                options=options,
            )
            if vertex
            else ag_types.GeminiAPIEndpoint(
                api_key=api_key,
                options=options,
            )
        )
        return ag_types.ModelTarget(name=model, endpoint=endpoint)
    return model


def _extract_selection_and_reasoning(
    data: object,
) -> tuple[str | None, tuple[str, ...]]:
    """Extract selected skill and reasoning tuple from structured output payload."""
    invoked = None
    reasoning: tuple[str, ...] = ()
    if isinstance(data, BaseModel):
        invoked = getattr(data, "selected_skill", None)
        if r := getattr(data, "reasoning", None):
            r_str = str(r).strip()
            if r_str:
                reasoning = (r_str,)
    elif isinstance(data, dict):
        invoked = data.get("selected_skill")
        if r := data.get("reasoning"):
            r_str = str(r).strip()
            if r_str:
                reasoning = (r_str,)
    return invoked, reasoning


_HTTP_TOO_MANY_REQUESTS = 429


_HTTP_ERROR_THRESHOLD = 400


class _SuppressRetryableStepErrorFilter(logging.Filter):
    """Filter transient retryable SDK step warnings unless root logger is at DEBUG level."""

    @override
    def filter(self, record: logging.LogRecord) -> bool:
        """Return False for transient retryable step warnings unless DEBUG logging is enabled."""
        if record.levelno != logging.WARNING:
            return True
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            return True
        msg = record.getMessage()
        if "System step error" not in msg:
            return True
        lower_msg = msg.lower()
        return not any(token in lower_msg for token in ("retryable error", "http 503", "http 429"))


_RETRYABLE_STEP_FILTER = _SuppressRetryableStepErrorFilter()
_FILTER_LOCK = threading.Lock()
_FILTER_REFCOUNT = 0


@contextlib.contextmanager
def _suppress_retryable_step_warnings() -> Iterator[None]:
    """Attach _SuppressRetryableStepErrorFilter to the root logger while SDK calls are active."""
    global _FILTER_REFCOUNT  # noqa: PLW0603
    root_logger = logging.getLogger()
    with _FILTER_LOCK:
        if _FILTER_REFCOUNT == 0 and _RETRYABLE_STEP_FILTER not in root_logger.filters:
            root_logger.addFilter(_RETRYABLE_STEP_FILTER)
        _FILTER_REFCOUNT += 1
    try:
        yield
    finally:
        with _FILTER_LOCK:
            _FILTER_REFCOUNT = max(0, _FILTER_REFCOUNT - 1)
            if _FILTER_REFCOUNT == 0 and _RETRYABLE_STEP_FILTER in root_logger.filters:
                root_logger.removeFilter(_RETRYABLE_STEP_FILTER)


_DIR_READ_ERR_PATTERN = re.compile(r"read\s+([^\r\n]+?):\s*is a directory", flags=re.IGNORECASE)


def _extract_step_dir_error_skill(step: object, resident: Collection[str]) -> str | None:
    """Resolve a resident skill name when a step failed attempting to read a skill directory."""
    raw_err = str(getattr(step, "error", "") or "").strip()
    if not raw_err or "is a directory" not in raw_err.lower():
        return None
    for match in _DIR_READ_ERR_PATTERN.finditer(raw_err):
        captured_path = match.group(1).strip().strip("'\"`")
        if skill := resolve_skill_from_path(captured_path, resident):
            return skill
    return None


def _iter_conversation_history(agent: object) -> Sequence[object]:
    """Return conversation history steps from an Agent instance when present."""
    conv = getattr(agent, "conversation", None)
    history = getattr(conv, "history", None)
    return history if isinstance(history, Sequence) else ()


def _extract_step_skills(step: object, resident: Collection[str]) -> list[str]:
    """Extract resident skill names from a step's directory error or tool calls."""
    skills: list[str] = []
    if dir_skill := _extract_step_dir_error_skill(step, resident):
        skills.append(dir_skill)
    for tc in getattr(step, "tool_calls", None) or ():
        args = getattr(tc, "args", None) or getattr(tc, "arguments", {}) or {}
        if (
            isinstance(args, Mapping)
            and (path := extract_tool_path(args))
            and (tc_skill := resolve_skill_from_path(path, resident))
            and tc_skill not in skills
        ):
            skills.append(tc_skill)
    return skills


def _format_step_error(
    step: object,
    error_status: object,
    resident: Collection[str] = (),
    *,
    is_dir_skill_step: bool = False,
) -> str | None:
    """Format a single SDK conversation step error when present."""
    status = getattr(step, "status", None)
    raw_err = str(getattr(step, "error", "") or "").strip()
    try:
        http_code = int(getattr(step, "http_code", 0) or 0)
    except (ValueError, TypeError):
        http_code = 0

    is_error_status = status in (error_status, "STATE_ERROR", "ERROR")
    if (
        not is_error_status
        and http_code < _HTTP_ERROR_THRESHOLD
        and (status is not None or not raw_err)
    ):
        return None

    if is_dir_skill_step or (
        resident and _extract_step_dir_error_skill(step, resident) is not None
    ):
        return None

    err_msg = raw_err or "unknown system error"
    lower_err = err_msg.lower()
    if http_code == _HTTP_TOO_MANY_REQUESTS or any(
        k in lower_err for k in ("429", "resource exhausted", "quota")
    ):
        return f"rate limit (429): {err_msg}"
    if http_code > 0:
        return f"sdk step error (HTTP {http_code}): {err_msg}"
    return f"sdk step error: {err_msg}"


def _extract_history_error(
    agent: object,
    resident: Collection[str] = (),
) -> str | None:
    """Extract formatted rate-limit or system error from SDK conversation history."""
    history = _iter_conversation_history(agent)
    if not history:
        return None
    step_status_cls = getattr(ag_types, "StepStatus", None) if ag_types is not None else None
    error_status = (
        getattr(step_status_cls, "ERROR", "STATE_ERROR")
        if step_status_cls is not None
        else "STATE_ERROR"
    )
    for step in reversed(history):
        if formatted := _format_step_error(step, error_status, resident=resident):
            return formatted
    return None


def _inspect_conversation_history(
    agent: object,
    resident: Collection[str],
    tracker: TrajectoryTracker,
    base_tools: Iterable[str],
    *,
    post_step_ran: bool = False,
) -> tuple[tuple[str, ...], str | None]:
    """Inspect conversation history in a single pass to recover chronological skills and errors."""
    history = _iter_conversation_history(agent)
    tools = list(base_tools)
    if not history:
        return tuple(tools), None

    step_status_cls = getattr(ag_types, "StepStatus", None) if ag_types is not None else None
    error_status = (
        getattr(step_status_cls, "ERROR", "STATE_ERROR")
        if step_status_cls is not None
        else "STATE_ERROR"
    )

    chronological_skills: list[str] = []
    saw_dir_skill = False
    latest_error: str | None = None

    for step in history:
        dir_skill = _extract_step_dir_error_skill(step, resident)
        is_dir_step = dir_skill is not None
        if is_dir_step:
            saw_dir_skill = True
        for s in _extract_step_skills(step, resident):
            if s not in chronological_skills:
                chronological_skills.append(s)
        if formatted := _format_step_error(
            step,
            error_status,
            resident=resident,
            is_dir_skill_step=is_dir_step,
        ):
            latest_error = formatted

    if saw_dir_skill and "view_file" not in tools:
        tools.append("view_file")

    if saw_dir_skill and not post_step_ran:
        # Preserve forward chronological order across Turn-1 directory errors and subsequent hooks
        merged_order = list(chronological_skills)
        for existing in tracker.invoked_skills:
            if existing not in merged_order:
                merged_order.append(existing)
        tracker.invoked_skills.clear()
        tracker.early_exit_hit = False
        for s in merged_order:
            tracker.observe(s)

    return tuple(tools), latest_error


def _resolve_empty_selection_error(
    history_error: str | None,
    observed_tools: tuple[str, ...],
    *,
    has_text: bool = False,
) -> str | None:
    """Resolve error string when an agent turn produces no skill selection or structured output."""
    if history_error:
        return history_error
    if has_text or observed_tools:
        return None
    return "empty selection (likely rate-limited)"


class AntigravitySdkOptions(AntigravityOptions):
    """Specify runtime configuration options for the Antigravity SDK driver."""

    model: str = Field(
        default_factory=lambda: agent_default_model("antigravity-sdk") or DEFAULT_GEMINI_MODEL,
    )
    app_data_dir: Path | None = None
    isolation_dir_field: ClassVar[str | None] = "app_data_dir"
    vertex: bool | None = None
    project: str | None = None
    location: str | None = None
    api_max_retries: int | None = Field(
        default=None,
        ge=0,
        description="Maximum retry attempts configured for the Antigravity SDK API client.",
    )
    api_retry_jitter: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Jitter fraction applied to exponential backoff retries in the SDK client.",
    )


class _AntigravitySdkConfigMixin:
    """Consolidate shared Vertex AI and ADC resolution for SDK runtime and generator."""

    options: AntigravitySdkOptions

    @property
    def effective_vertex(self) -> bool:
        """Determine whether Vertex AI backend is active."""
        if self.options.vertex is not None:
            return self.options.vertex
        if getattr(self.options, "provider", None) == "vertex":
            return True
        return os.environ.get("GOOGLE_GENAI_USE_ENTERPRISE", "").lower() in (
            "true",
            "1",
        ) or os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").lower() in ("true", "1")

    @property
    def effective_project(self) -> str | None:
        """Resolve GCP project ID for Vertex AI execution."""
        if not self.effective_vertex:
            return None
        if self.options.api_key:
            return self.options.project
        return self.options.project or os.environ.get("GOOGLE_CLOUD_PROJECT")

    @property
    def effective_location(self) -> str | None:
        """Resolve GCP region/location for Vertex AI execution."""
        if not self.effective_vertex:
            return None
        if self.options.api_key:
            return self.options.location
        return self.options.location or os.environ.get("GOOGLE_CLOUD_LOCATION") or "global"

    @property
    def effective_api_key(self) -> str | None:
        """Return configured API key or fallback to environment variables."""
        if self.options.api_key:
            return self.options.api_key
        if self.effective_vertex:
            return None
        return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

    def _sync_sdk_env(self, env: dict[str, str]) -> dict[str, str]:
        """Apply API key synchronization and Vertex credential preservation to environment."""
        if (api_key := self.effective_api_key) is not None:
            env["GEMINI_API_KEY"] = api_key
            env["GOOGLE_API_KEY"] = api_key
        elif self.effective_vertex and not self.options.api_key:
            env.pop("GEMINI_API_KEY", None)
            env.pop("GOOGLE_API_KEY", None)

        blocked = getattr(self, "blocked_env_vars", None) or ()
        if (
            self.effective_vertex
            and "GOOGLE_APPLICATION_CREDENTIALS" not in blocked
            and "GOOGLE_APPLICATION_CREDENTIALS" in os.environ
            and "GOOGLE_APPLICATION_CREDENTIALS" not in env
        ):
            env["GOOGLE_APPLICATION_CREDENTIALS"] = os.environ["GOOGLE_APPLICATION_CREDENTIALS"]

        return sync_google_and_gemini_keys(env)

    def _target_model_spec(self, model: str, effort: str | None) -> str | ag_types.ModelTarget:
        """Construct model target with reasoning effort and endpoint options when configured."""
        return _build_model_spec(
            model,
            effort,
            vertex=self.effective_vertex,
            project=self.effective_project,
            location=self.effective_location,
            api_key=self.effective_api_key,
        )

    def _build_retry_config(self) -> ag_types.RetryConfig | None:
        """Construct RetryConfig when api_max_retries or api_retry_jitter is configured."""
        if (
            ag_types is None
            or not hasattr(ag_types, "RetryConfig")
            or not hasattr(ag_types, "ModelAPIRetryConfig")
            or (self.options.api_max_retries is None and self.options.api_retry_jitter is None)
        ):
            return None
        api_kwargs: dict[str, Any] = {}
        if self.options.api_max_retries is not None:
            api_kwargs["max_retries"] = self.options.api_max_retries
        if self.options.api_retry_jitter is not None:
            api_kwargs["jitter_range"] = self.options.api_retry_jitter
        return ag_types.RetryConfig(api_retry=ag_types.ModelAPIRetryConfig(**api_kwargs))

    def _base_config_kwargs(
        self,
        model: str | ag_types.ModelTarget,
        env: dict[str, str],
    ) -> dict[str, Any]:
        """Assemble shared LocalAgentConfig keyword arguments across runtime and generator."""
        kwargs: dict[str, Any] = {
            "model": model,
            "api_key": self.effective_api_key,
            "vertex": self.effective_vertex,
            "project": self.effective_project,
            "location": self.effective_location,
            "env": env,
        }
        if (retry_cfg := self._build_retry_config()) is not None:
            kwargs["retry_config"] = retry_cfg
        return kwargs

    def _resolve_schema_dict(
        self,
        schema: str | Mapping[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Parse optional JSON schema argument or options fallback into a dictionary."""
        if isinstance(schema, Mapping):
            return dict(schema)
        raw_schema = schema if isinstance(schema, str) else self.options.json_schema
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


class AntigravitySdkRuntime(_AntigravitySdkConfigMixin, AntigravityRuntime):
    """Execute evaluation queries using the Google Antigravity Python SDK."""

    name = "antigravity-sdk"
    options: AntigravitySdkOptions
    api_key_env_var: str | None = "GEMINI_API_KEY"
    _skills_subpath = ".agents/skills"

    def __init__(
        self,
        settings: RuntimeSettings | None = None,
        options: AntigravitySdkOptions | None = None,
    ) -> None:
        """Initialize Antigravity SDK driver with runtime settings and options."""
        if not _HAS_ANTIGRAVITY:
            raise_missing_agent_dependency(
                "antigravity-sdk",
                ImportError("google-antigravity is not installed"),
                role="runtime",
            )
        super().__init__(settings=settings, options=options)

    @override
    def build_env(self, workdir: Path | None = None) -> dict[str, str]:
        """Assemble environment variables with API key synchronization."""
        env = super().build_env(workdir)
        return self._sync_sdk_env(env)

    def _model_spec(self) -> str | ag_types.ModelTarget:
        """Construct model target with reasoning effort endpoint options when configured."""
        return self._target_model_spec(self.options.model, self.effective_effort)

    @property
    def _default_selection_tools(self) -> tuple[Any, ...]:
        """Return default selection tool tuple, including finish only when json_schema is set."""
        if self.options.json_schema:
            return (*SELECTION_TOOLS, *MULTI_TURN_SELECTION_TOOLS)
        return MULTI_TURN_SELECTION_TOOLS

    @staticmethod
    def _resolve_sdk_slot(workdir: Path) -> tuple[Path, Path]:
        """Resolve the root SDK directory and thread-isolated slot directory."""
        sdk_root = (Path(workdir) / ".reach_antigravity_sdk").resolve()
        return sdk_root, probe_slot_dir(sdk_root)

    def _select_config(
        self,
        workdir: Path,
        hooks: list[Any] | None = None,
    ) -> LocalAgentConfig:
        """Assemble LocalAgentConfig with turn budget, inspection tools, and hooks."""
        app_data_dir = None
        if self.options.isolate_config_dir or self.options.app_data_dir:
            sdk_target = (
                self.options.app_data_dir
                if self.options.app_data_dir is not None
                else self._resolve_sdk_slot(workdir)[1]
            )
            sdk_dir = ensure_private_directory(sdk_target)
            app_data_dir = str(sdk_dir)

        enabled_tools = list(self._default_selection_tools)
        if self.allowed_tools:
            builtin_by_name = (
                {_tool_name(member): member for member in ag_types.BuiltinTools}
                if ag_types is not None and hasattr(ag_types, "BuiltinTools")
                else {}
            )
            enabled_tools = [builtin_by_name.get(t, t) for t in self.allowed_tools]

        skills_paths = [str(self.skills_dir(workdir))]
        if self.options.use_symlinks and (skills_dir := self.skills_dir(workdir)).is_dir():
            resolved_targets = {
                str(resolved)
                for child in skills_dir.iterdir()
                if child.is_symlink() and (resolved := child.resolve()).is_dir()
            }
            skills_paths.extend(p for p in sorted(resolved_targets) if p not in skills_paths)

        kwargs = self._base_config_kwargs(self._model_spec(), self.build_env(workdir))
        kwargs.update(
            {
                "skills_paths": skills_paths,
                "capabilities": ag_types.CapabilitiesConfig(
                    enabled_tools=enabled_tools,
                    enable_subagents=False,
                ),
                "budget_config": ag_types.BudgetConfig(max_model_calls=self.options.max_turns),
                "app_data_dir": app_data_dir,
                "hooks": hooks,
            }
        )
        if (schema_dict := self._resolve_schema_dict()) is not None:
            kwargs["response_schema"] = schema_dict
        return LocalAgentConfig(**kwargs)

    def _build_selection_hooks(
        self,
        tracker: TrajectoryTracker,
        hook_observed_tools: list[str],
        agent_holder: list[Any],
        post_step_seen: list[bool],
    ) -> list[Any]:
        """Build pre-tool and post-step hooks for real-time skill tracking and early exit."""
        if ag_hooks is None:
            return []
        hooks_list: list[Any] = []

        @ag_hooks.pre_tool_call_decide
        async def _on_tool_call(call: ag_types.ToolCall) -> ag_types.HookResult:
            if call_name := getattr(call, "name", None):
                hook_observed_tools.append(_tool_name(call_name))
            if tracker.early_exit and tracker.early_exit_hit:
                return ag_types.HookResult(allow=False)
            args = getattr(call, "args", None) or getattr(call, "arguments", {}) or {}
            skill: str | None = None
            if (
                isinstance(args, Mapping)
                and (path := extract_tool_path(args))
                and (skill := resolve_skill_from_path(path, self._resident))
            ):
                should_stop = tracker.observe(skill)
                if self.options.early_exit and should_stop:
                    return ag_types.HookResult(allow=False)
            if isinstance(args, Mapping) and (
                modified_args := normalize_skill_tool_args(args, skill)
            ):
                return ag_types.HookResult(allow=True, modified_args=modified_args)
            return ag_types.HookResult(allow=True)

        hooks_list.append(_on_tool_call)
        post_step_dec = getattr(ag_hooks, "_post_step", None) or getattr(
            getattr(ag_hooks, "hooks", None),
            "_post_step",
            None,
        )
        if callable(post_step_dec):

            @post_step_dec
            async def _on_post_step(step: object) -> None:
                if (skill := _extract_step_dir_error_skill(step, self._resident)) is None:
                    return
                post_step_seen.append(True)
                hook_observed_tools.append("view_file")
                should_stop = tracker.observe(skill)
                if self.options.early_exit and should_stop and agent_holder:
                    conn = getattr(
                        getattr(agent_holder[0], "conversation", None),
                        "connection",
                        None,
                    )
                    cancel_fn = getattr(conn, "cancel", None)
                    if callable(cancel_fn):
                        with contextlib.suppress(Exception):
                            res = cancel_fn()
                            if asyncio.iscoroutine(res):
                                await res

            hooks_list.append(_on_post_step)
        return hooks_list

    async def _select_async(
        self,
        query_text: str,
        workdir: Path,
        target_skill: str | None = None,
    ) -> SelectionOutcome:
        """Execute chat evaluation asynchronously and return observed outcome."""
        tracker = self.make_tracker(target_skill)
        hook_observed_tools: list[str] = []
        agent_holder: list[Any] = []
        post_step_seen: list[bool] = []
        hooks_list = self._build_selection_hooks(
            tracker,
            hook_observed_tools,
            agent_holder,
            post_step_seen,
        )
        config = self._select_config(workdir, hooks=hooks_list)

        data: Any = None
        text_out = ""
        stream_tools: list[str] = []
        stop_reason: Any = "END_TURN"

        try:
            with _suppress_retryable_step_warnings():
                async with Agent(config) as agent:
                    agent_holder.append(agent)
                    try:
                        response = await agent.chat(query_text)
                        data = await response.structured_output()
                        text_fn: Any = getattr(response, "text", None)
                        raw_text = await text_fn() if text_fn is not None else None
                        text_out = str(raw_text).strip() if isinstance(raw_text, str) else ""
                        stream_tools = [_tool_name(call.name) async for call in response.tool_calls]
                        stop_reason = getattr(response, "stop_reason", "END_TURN")
                    except (Exception, asyncio.CancelledError):
                        if not tracker.early_exit_hit:
                            raise
                    observed_tools, history_error = _inspect_conversation_history(
                        agent,
                        self._resident,
                        tracker,
                        stream_tools or hook_observed_tools,
                        post_step_ran=bool(post_step_seen),
                    )
        except (AntigravityValidationError, Exception) as err:
            if isinstance(err, RuntimeError):
                raise
            msg = f"Antigravity SDK execution error: {err}"
            raise RuntimeError(msg) from err

        allowed_tool_names = (
            set(self.allowed_tools)
            if self.allowed_tools
            else (
                set(self.ANTIGRAVITY_SELECTION_TOOLS)
                | {_tool_name(t) for t in self._default_selection_tools}
            )
        )
        error = None
        if stop_reason not in EXPECTED_STOP_REASONS:
            reason_str = getattr(stop_reason, "value", str(stop_reason))
            error = f"runtime error: {reason_str}"
        elif tool_leak := check_tool_leak(observed_tools, allowed_tool_names):
            error = tool_leak

        invoked, reasoning = _extract_selection_and_reasoning(data)
        if not reasoning and text_out:
            reasoning = (text_out,)

        invoked_skills = tuple(tracker.invoked_skills) or ((invoked,) if invoked else ())

        if not error and not tracker.early_exit_hit and not invoked_skills and data is None:
            error = _resolve_empty_selection_error(
                history_error,
                observed_tools,
                has_text=bool(text_out),
            )

        return tracker.apply_to_outcome(
            SelectionOutcome(
                invoked_skills=invoked_skills,
                early_exit=tracker.early_exit_hit,
                turns_taken=tracker.turns_taken,
                reasoning=reasoning,
                observed_catalog=self._resident,
                observed_tools=observed_tools,
                cost_usd=None,
                error=error,
            ),
        )

    @override
    def post_probe(self, workdir: Path) -> None:
        """Clean ephemeral per-thread SDK slot or isolated app_data_dir after probe execution."""
        if not self.options.isolate_config_dir:
            return
        if self.options.app_data_dir is None:
            sdk_root, slot_dir = self._resolve_sdk_slot(workdir)
            had_slot = slot_dir.exists()
            safe_cleanup_isolated_dir(workdir, slot_dir)
            if self.options.auto_clean or had_slot:
                # Prune the root only once the last concurrent worker has released its slot.
                with contextlib.suppress(OSError):
                    sdk_root.rmdir()
        elif self.options.auto_clean:
            safe_cleanup_isolated_dir(workdir, self.options.app_data_dir)

    @override
    def select(
        self,
        query_text: str,
        workdir: Path,
        target_skill: str | None = None,
    ) -> SelectionOutcome:
        """Execute query evaluation probe and return SelectionOutcome."""
        import time

        t0 = time.monotonic()
        try:
            try:
                outcome = _run_sync(
                    asyncio.wait_for(
                        self._select_async(query_text, workdir, target_skill=target_skill),
                        timeout=self.timeout_s,
                    ),
                )
            except TimeoutError:
                outcome = SelectionOutcome(
                    error="timeout",
                    observed_catalog=self._resident,
                )
            except Exception as exc:  # noqa: BLE001
                outcome = SelectionOutcome(
                    error=str(exc),
                    observed_catalog=self._resident,
                )
            elapsed_ms = max(1, int((time.monotonic() - t0) * 1000))
            return outcome.model_copy(update={"duration_ms": outcome.duration_ms or elapsed_ms})
        finally:
            self.post_probe(workdir)


class AntigravitySdkGenerator(_AntigravitySdkConfigMixin, BaseTextGenerator[AntigravitySdkOptions]):
    """Generate text completions using the Antigravity SDK."""

    name: str = "antigravity-sdk"
    options: AntigravitySdkOptions

    def __init__(
        self,
        model: str = "",
        *,
        timeout_s: int = 300,
        options: AntigravitySdkOptions | None = None,
        settings: RuntimeSettings | None = None,
    ) -> None:
        """Initialize Antigravity SDK generator with model, timeout, and options."""
        if not _HAS_ANTIGRAVITY:
            raise_missing_agent_dependency(
                "antigravity-sdk",
                ImportError("google-antigravity is not installed"),
                role="generator",
            )
        if isinstance(options, AntigravitySdkOptions):
            opts = options.model_copy(update={"model": model}) if model else options
        elif settings is not None and settings.options:
            base_opts = dict(settings.options)
            if model:
                base_opts["model"] = model
            opts = AntigravitySdkOptions.model_validate(base_opts)
        elif model:
            opts = AntigravitySdkOptions(model=model)
        else:
            opts = AntigravitySdkOptions()
        super().__init__(model=opts.model or model, timeout_s=timeout_s, options=opts)
        self.settings = settings

    @override
    def build_env(self) -> dict[str, str]:
        """Assemble environment variables with API key synchronization."""
        env = super().build_env()
        return self._sync_sdk_env(env)

    @property
    def effective_effort(self) -> str | None:
        """Return configured reasoning effort or default from model profile."""
        if self.options.effort:
            effort = self.options.effort
            return None if effort.lower() in ("none", "off") else effort
        if not _model_supports_thinking(self.model):
            return None
        try:
            return model_profile(self.model).effort
        except (KeyError, ValueError):
            return None

    def _model_spec(self) -> str | ag_types.ModelTarget:
        """Construct model target with reasoning effort endpoint options when configured."""
        return self._target_model_spec(self.options.model or self.model, self.effective_effort)

    @override
    def complete(self, prompt: str, *, schema: str | Mapping[str, Any] | None = None) -> str:
        """Execute text completion using the Antigravity SDK."""
        schema_dict = self._resolve_schema_dict(schema)
        caps = (
            ag_types.CapabilitiesConfig(enabled_tools=[], enable_subagents=False)
            if ag_types is not None and hasattr(ag_types, "CapabilitiesConfig")
            else None
        )

        async def _complete_async() -> str:
            kwargs = self._base_config_kwargs(self._model_spec(), self.build_env())
            if schema_dict is not None:
                kwargs["response_schema"] = schema_dict
            if caps is not None:
                kwargs["capabilities"] = caps

            config = LocalAgentConfig(**kwargs)
            with _suppress_retryable_step_warnings():
                async with Agent(config) as agent:
                    response = await agent.chat(prompt)
                    if schema_dict is not None:
                        attr = getattr(response, "structured_output", None)
                        structured = attr() if callable(attr) else attr
                        if asyncio.iscoroutine(structured):
                            structured = await structured
                        if structured is not None:
                            if hasattr(structured, "model_dump_json"):
                                return structured.model_dump_json()
                            if hasattr(structured, "model_dump"):
                                return json.dumps(structured.model_dump())
                            return json.dumps(structured)
                        err_msg = _extract_history_error(agent) or (
                            "empty structured output (likely rate-limited)"
                        )
                        raise ValueError(err_msg)
                    text_out = await response.text()
                    if not (text_out or "").strip():
                        err_msg = _extract_history_error(agent) or (
                            "empty response (likely rate-limited)"
                        )
                        raise ValueError(err_msg)
                    return text_out

        try:
            text = _run_sync(asyncio.wait_for(_complete_async(), timeout=self.timeout_s))
        except TimeoutError as err:
            msg = f"generation failed: timed out after {self.timeout_s}s"
            raise RuntimeError(msg) from err
        except (AntigravityValidationError, Exception) as err:
            msg = f"generation failed: {err}"
            raise RuntimeError(msg) from err
        self.completions += 1
        return text
