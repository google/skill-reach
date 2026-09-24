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

"""Drive the Goose agent CLI (aaif-goose/goose) as an evaluation runtime."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, override

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    ValidationError,
)

from reach.config import DEFAULT_GEMINI_MODEL, RuntimeSettings
from reach.runtime import (
    CliAgentRuntime,
    CliOptions,
    SessionStatus,
    SessionSummary,
    agent_default_model,
)
from reach.runtime._env import (
    apply_provider_api_key,
    detect_model_provider,
    sync_google_and_gemini_keys,
)
from reach.runtime._fs import (
    ensure_private_directory,
    extract_tool_path,
    resolve_skill_from_path,
)
from reach.runtime._subprocess import (
    extract_content_reasoning,
)
from reach.runtime.generator import BaseTextGenerator

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence


def _normalize_goose_provider(provider: str | None) -> str | None:
    """Normalize 'gemini' provider alias to Goose's canonical 'google' provider."""
    if provider is None:
        return None
    cleaned = provider.strip()
    if cleaned.lower() == "gemini":
        return "google"
    return cleaned or None


class GooseOptions(CliOptions):
    """Hold configuration options for driving the Goose agent CLI."""

    executable: str = "goose"
    model: str = Field(
        default_factory=lambda: agent_default_model("goose") or DEFAULT_GEMINI_MODEL,
        description="The model identifier to evaluate.",
    )
    provider: Annotated[str | None, AfterValidator(_normalize_goose_provider)] = None
    home_dir: Path | None = None
    isolation_dir_field: ClassVar[str | None] = "home_dir"
    no_profile: bool = True
    with_builtin: str = "skills"

    @property
    def effective_provider(self) -> str | None:
        """Return canonical Goose provider, inferring from model when provider is unset."""
        if self.provider:
            return self.provider
        if self.model.strip().lower().startswith("gemini"):
            return "google"
        return _normalize_goose_provider(detect_model_provider(self.model, api_key=self.api_key))


def resolve_skill_from_tool_call(
    tool_name: str,
    arguments: Mapping[str, Any] | None,
    resident: Iterable[str],
) -> str | None:
    """Extract skill name from Goose tool call arguments if it matches a resident skill."""
    if not tool_name:
        return None

    resident_lookup = {r.lower(): r for r in resident}
    args = arguments or {}

    # Check native load_skill tool call
    if "load_skill" in tool_name.lower():
        name_val = args.get("name")
        if not isinstance(name_val, str) or not name_val.strip():
            return None
        # Handle "skill-name/subfile.md" formats
        primary_name = name_val.strip().split("/")[0].strip()
        if primary_name.lower() in resident_lookup:
            return resident_lookup[primary_name.lower()]

    # Fallback: check file path arguments (e.g. read or developer tools)
    return resolve_skill_from_path(extract_tool_path(args), resident)


def _parse_jsonl_messages(text: str) -> dict[str, Any] | None:
    """Parse line-delimited JSON messages into a combined Goose payload dict."""
    messages: list[dict[str, Any]] = []
    metadata: dict[str, Any] = {}
    for line in text.splitlines():
        sline = line.strip()
        if not sline:
            continue
        try:
            obj = json.loads(sline)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        if "messages" in obj and isinstance(obj["messages"], list):
            messages.extend(obj["messages"])
        elif obj.get("role") in ("user", "assistant", "system"):
            messages.append(obj)
        if "metadata" in obj and isinstance(obj["metadata"], dict):
            metadata.update(obj["metadata"])
    return {"messages": messages, "metadata": metadata} if messages else None


def _parse_goose_payload(
    data: dict[str, Any] | str,
) -> tuple[dict[str, Any] | None, str | None]:
    """Extract dict payload from json string or dict, returning (payload, error)."""
    if not isinstance(data, str):
        if not isinstance(data, dict):
            return None, "unexpected goose output format"
        return data, None

    text = data.strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        if (jsonl_payload := _parse_jsonl_messages(text)) is not None:
            return jsonl_payload, None
        start = text.find("{")
        end = text.rfind("}")
        json_str = text[start : end + 1] if 0 <= start < end else text
        try:
            payload = json.loads(json_str)
        except json.JSONDecodeError as exc:
            return None, f"failed to parse goose output json: {exc}"

    if not isinstance(payload, dict):
        return None, "unexpected goose output format"
    return payload, None


def _extract_tool_request(
    item: dict[str, Any],
    resident: Iterable[str],
) -> tuple[str, str | None]:
    """Extract tool name and resolved skill invocation from a toolRequest content block."""
    tool_call = item.get("toolCall", {})
    if not isinstance(tool_call, dict):
        return "", None

    value = tool_call.get("value", {})
    if isinstance(value, dict):
        t_name = str(value.get("name", item.get("name", "")))
        t_args = value.get("arguments", item.get("arguments", {}))
    else:
        t_name = str(item.get("name", ""))
        t_args = item.get("arguments", {})

    skill = resolve_skill_from_tool_call(t_name, t_args, resident) if t_name else None
    return t_name, skill


def _extract_message_details(
    msg: dict[str, Any],
    resident: Iterable[str],
) -> tuple[list[str], list[str], list[str], str]:
    """Extract model, reasoning, observed tools, and skill invocations from an assistant message."""
    invoked: list[str] = []
    reasoning: list[str] = []
    observed_tools: list[str] = []
    resolved_model = ""

    metadata = msg.get("metadata")
    if isinstance(metadata, dict):
        inference = metadata.get("inference")
        if isinstance(inference, dict):
            req_model = inference.get("requestedModel")
            if isinstance(req_model, str) and req_model:
                resolved_model = req_model

    content = msg.get("content", [])
    if isinstance(content, list):
        reasoning.extend(extract_content_reasoning(content))
        for item in content:
            if isinstance(item, dict) and item.get("type") == "toolRequest":
                t_name, skill = _extract_tool_request(item, resident)
                if t_name:
                    observed_tools.append(t_name)
                if skill:
                    invoked.append(skill)

    return invoked, reasoning, observed_tools, resolved_model


class GooseMetadata(BaseModel):
    """Represent session metadata and token counts emitted by Goose JSON output."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    input_tokens: NonNegativeInt = 0
    cache_read_input_tokens: NonNegativeInt = 0
    cache_write_input_tokens: NonNegativeInt = 0
    prompt_tokens: NonNegativeInt | None = None
    cost_usd: NonNegativeFloat | None = None

    @property
    def total_prompt_tokens(self) -> int | None:
        """Calculate total input/prompt tokens across direct and cached prompt segments."""
        goose_input = (
            self.input_tokens + self.cache_read_input_tokens + self.cache_write_input_tokens
        )
        if goose_input > 0:
            return goose_input
        return self.prompt_tokens


def _extract_goose_metadata(meta_obj: object) -> tuple[float | None, int | None]:
    """Extract cost_usd and total prompt tokens via GooseMetadata."""
    if not isinstance(meta_obj, dict):
        return None, None
    try:
        meta = GooseMetadata.model_validate(meta_obj)
    except ValidationError:
        return None, None
    return meta.cost_usd, meta.total_prompt_tokens


def parse_goose_output(
    data: dict[str, Any] | str,
    resident: Iterable[str],
) -> SessionSummary:
    """Parse Goose output payload to extract skill invocations, tools, cost, and tokens."""
    payload, err = _parse_goose_payload(data)
    if err is not None or payload is None:
        return SessionSummary(error=err or "unexpected goose output format")

    invoked: list[str] = []
    reasoning: list[str] = []
    observed_tools: list[str] = []
    resolved_model = ""
    assistant_turns = 0

    messages = payload.get("messages", [])
    if isinstance(messages, list):
        for msg in messages:
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                assistant_turns += 1
                m_invoked, m_reasoning, m_tools, m_model = _extract_message_details(msg, resident)
                invoked.extend(m_invoked)
                reasoning.extend(m_reasoning)
                observed_tools.extend(m_tools)
                if m_model:
                    resolved_model = m_model

    cost_usd, prompt_tokens = _extract_goose_metadata(payload.get("metadata"))

    return SessionSummary(
        invoked_skills=tuple(invoked),
        turns_taken=max(1, assistant_turns),
        reasoning=tuple(reasoning),
        observed_tools=tuple(observed_tools),
        cost_usd=cost_usd,
        prompt_tokens=prompt_tokens,
        resolved_model=resolved_model,
        status=SessionStatus.SUCCESS,
        error=None,
    )


class GooseRuntime(CliAgentRuntime[GooseOptions]):
    """Drive the Goose agent CLI as an evaluation runtime."""

    name: str = "goose"
    options: GooseOptions
    api_key_env_var: str | None = None
    _skills_subpath: str = ".agents/skills"
    isolation_dir_name: ClassVar[str | None] = ".reach_goose"

    def __init__(
        self,
        settings: RuntimeSettings | None = None,
        options: GooseOptions | None = None,
    ) -> None:
        """Initialize the GooseRuntime with settings or defaults."""
        if options is None:
            effective_settings = settings or RuntimeSettings(agent=self.name)
            options = GooseOptions.model_validate(dict(effective_settings.options or {}))
        else:
            opts_dict = options.model_dump(mode="json")
            effective_settings = settings or RuntimeSettings(agent=self.name, options=opts_dict)
        super().__init__(settings=effective_settings, options=options)

    def build_command(self, query_text: str) -> list[str]:
        """Assemble command-line arguments for running a Goose evaluation probe."""
        opts = self.options
        cmd = [
            opts.executable,
            "run",
            "-q",
            "--text",
            query_text,
            "--output-format",
            "json",
            "--no-session",
            "--max-turns",
            str(opts.max_turns),
        ]
        if opts.no_profile:
            cmd.append("--no-profile")
        if opts.with_builtin:
            cmd += ["--with-builtin", opts.with_builtin]
        if opts.model:
            cmd += ["--model", opts.model]
        if opts.effective_provider:
            cmd += ["--provider", opts.effective_provider]
        if opts.extra_args:
            cmd += list(opts.extra_args)
        return cmd

    @override
    def parse_stream(
        self,
        lines: Iterable[str],
        resident: Sequence[str] = (),
        early_exit: bool = False,
    ) -> SessionSummary:
        """Parse CLI stdout lines into a standardized session summary."""
        line_list = list(lines)
        if not line_list or not "".join(line_list).strip():
            return SessionSummary(
                early_exit=early_exit,
                status=SessionStatus.SUCCESS if early_exit else None,
            )
        text = "\n".join(line_list)
        summary = parse_goose_output(text, resident or self._resident)
        if early_exit:
            summary = summary.model_copy(
                update={"early_exit": True, "status": summary.status or SessionStatus.SUCCESS},
            )
        return summary

    @override
    def extract_skills_from_line(self, line: str) -> Sequence[str]:
        """Extract invoked skill names from an event line for early-exit detection."""
        detected = []
        try:
            data = json.loads(line.strip())
            if isinstance(data, dict):
                msgs = data.get("messages", [data])
                for msg in msgs if isinstance(msgs, list) else [msgs]:
                    if isinstance(msg, dict):
                        content = msg.get("content", [msg])
                        for item in content if isinstance(content, list) else [content]:
                            if isinstance(item, dict):
                                _, skill = _extract_tool_request(item, self._resident)
                                if skill:
                                    detected.append(skill)
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
            pass
        return tuple(detected)

    @override
    def build_env(self, workdir: Path | None = None) -> dict[str, str]:
        """Assemble process environment with isolation, telemetry suppression, and API keys."""
        env = super().build_env(workdir)
        env["OTEL_SDK_DISABLED"] = "true"

        apply_provider_api_key(
            env,
            provider=self.options.provider,
            api_key=self.options.api_key,
            default_provider="openai",
        )
        sync_google_and_gemini_keys(env)

        if workdir is not None and (iso_dir := self.effective_isolation_dir(workdir)) is not None:
            isolated_dir = ensure_private_directory(iso_dir)
            env["HOME"] = str(isolated_dir)
            env["XDG_CONFIG_HOME"] = str(isolated_dir / ".config")
            env["XDG_DATA_HOME"] = str(isolated_dir / ".local" / "share")
            env["XDG_STATE_HOME"] = str(isolated_dir / ".local" / "state")

        return env


class GooseGenerator(BaseTextGenerator[GooseOptions]):
    """Generate text completions using the Goose CLI."""

    name: str = "goose"
    options: GooseOptions

    def __init__(
        self,
        model: str = "",
        *,
        timeout_s: int = 300,
        options: GooseOptions | None = None,
    ) -> None:
        """Initialize Goose generator with model, timeout, and options."""
        if isinstance(options, GooseOptions):
            opts = options
        elif model:
            opts = GooseOptions(model=model)
        else:
            opts = GooseOptions()
        super().__init__(model=opts.model or model, timeout_s=timeout_s, options=opts)

    @override
    def build_completion_command(self, prompt: str = "") -> list[str]:
        """Assemble command-line arguments for raw text completion."""
        del prompt  # Prompt is passed via stdin to avoid Linux MAX_ARG_STRLEN limits
        opts = self.options
        cmd = [
            opts.executable,
            "run",
            "-q",
            "-i",
            "-",
            "--no-session",
        ]
        if opts.no_profile:
            cmd.append("--no-profile")
        if self.model:
            cmd += ["--model", self.model]
        if opts.effective_provider:
            cmd += ["--provider", opts.effective_provider]
        return [*cmd, *opts.extra_args]

    @override
    def build_env(self) -> dict[str, str]:
        """Assemble process environment with API keys and telemetry suppression."""
        env = super().build_env()
        env["OTEL_SDK_DISABLED"] = "true"
        apply_provider_api_key(
            env,
            provider=self.options.provider,
            api_key=self.options.api_key,
            default_provider="openai",
        )
        return sync_google_and_gemini_keys(env)
