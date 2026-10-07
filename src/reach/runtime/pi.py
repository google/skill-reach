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

"""Drive the Pi agent harness CLI (earendil-works/pi) as an evaluation runtime."""

from __future__ import annotations

import contextlib
import shutil
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, override

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    ValidationError,
    field_validator,
)

from reach.config import DEFAULT_GEMINI_MODEL, resolve_path
from reach.runtime import (
    CliAgentRuntime,
    CliOptions,
    SelectionOutcome,
    SessionStatus,
    SessionSummary,
    agent_default_model,
)
from reach.runtime._env import (
    apply_provider_api_key,
    sync_google_and_gemini_keys,
)
from reach.runtime._fs import (
    ensure_private_directory,
    extract_tool_path,
    probe_slot_dir,
    resolve_skill_from_path,
)
from reach.runtime._subprocess import (
    extract_content_reasoning,
    format_subprocess_error,
    iter_json_lines,
    process_failure_reason,
    run_subprocess_probe,
)
from reach.runtime.generator import BaseTextGenerator

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

#: Workdir-relative root holding one session slot per concurrent probe worker.
SESSION_DIRNAME = ".reach_pi_sessions"


def _newest_transcript(session_dir: Path, exclude: set[Path] | None = None) -> Path | None:
    """Return the most recently modified session transcript, ignoring excluded files."""
    skip = exclude or set()
    candidates = [f for f in session_dir.glob("*.jsonl") if f not in skip]
    if not candidates:
        return None
    return max(candidates, key=lambda f: f.stat().st_mtime)


#: Valid reasoning effort levels accepted by Pi CLI.
PI_THINKING_LEVELS: frozenset[str] = frozenset(
    {"off", "minimal", "low", "medium", "high", "xhigh", "max"}
)


class PiOptions(CliOptions):
    """Hold configuration options for driving the Pi coding agent CLI."""

    executable: str = "pi"
    model: str = Field(
        default_factory=lambda: agent_default_model("pi") or DEFAULT_GEMINI_MODEL,
        description="The model identifier to evaluate.",
    )
    provider: str | None = "google"
    tools: str = "read"
    no_themes: bool = True
    no_extensions: bool = True
    mcp_config: Path | None = Field(
        default=None,
        description="Path to an MCP JSON configuration file to pass via --mcp-config.",
    )
    agent_dir: Path | None = None
    isolation_dir_field: ClassVar[str | None] = "agent_dir"

    @field_validator("mcp_config", mode="after")
    @classmethod
    def _resolve_mcp_config_path(cls, value: Path | None) -> Path | None:
        """Resolve mcp_config to a canonical absolute path."""
        return resolve_path(value) if value is not None else None

    def resolve_thinking(self, profile_effort: str | None = None) -> str | None:
        """Resolve effective thinking level for Pi CLI.

        Returns:
            Validated thinking level string ('off', 'minimal', ..., 'max') or None if thinking
            should be omitted (e.g. effort='none' or unset).

        Raises:
            ValueError: If candidate effort is unrecognized.
        """
        candidate = self.effort or profile_effort
        if candidate is None:
            return None
        candidate_lower = candidate.lower()
        if candidate_lower == "none":
            return None
        if candidate_lower in PI_THINKING_LEVELS:
            return candidate_lower
        msg = (
            f"Invalid thinking level for Pi: {candidate!r}. "
            f"Must be one of: {', '.join(sorted(PI_THINKING_LEVELS))} or 'none'."
        )
        raise ValueError(msg)

    def common_cli_args(
        self,
        model: str = "",
        profile_effort: str | None = None,
        thinking: str | None = None,
    ) -> list[str]:
        """Assemble shared CLI arguments for Pi runtime and generator."""
        cmd: list[str] = []
        if self.no_extensions:
            cmd.append("--no-extensions")
        if self.mcp_config is not None:
            cmd += ["--mcp-config", str(self.mcp_config)]
        if self.no_themes:
            cmd.append("--no-themes")
        target_model = model or self.model
        if target_model:
            cmd += ["--model", target_model]
        cmd += self.provider_args("--provider")
        cmd += self.api_key_args("--api-key")
        effective_thinking = (
            thinking if thinking is not None else self.resolve_thinking(profile_effort)
        )
        if effective_thinking:
            cmd += ["--thinking", effective_thinking]
        return cmd


def _extract_pi_tool_call(
    item: dict[str, Any],
    resident: Iterable[str],
) -> tuple[str, str | None]:
    """Extract tool name and resolved skill from a Pi toolCall content block."""
    t_name = str(item.get("name", ""))
    skill = None
    if t_name == "read":
        args = item.get("arguments")
        if isinstance(args, dict):
            skill = resolve_skill_from_path(extract_tool_path(args), resident)
    return t_name, skill


def _extract_pi_message_content(
    msg: dict[str, Any],
    resident: Iterable[str],
) -> tuple[list[str], list[str], list[str]]:
    """Extract reasoning, tools, and skill invocations from message content."""
    reasoning: list[str] = []
    observed_tools: list[str] = []
    invoked: list[str] = []

    content = msg.get("content")
    if isinstance(content, list):
        reasoning.extend(extract_content_reasoning(content))
        for item in content:
            if isinstance(item, dict) and item.get("type") == "toolCall":
                t_name, skill = _extract_pi_tool_call(item, resident)
                if t_name:
                    observed_tools.append(t_name)
                if skill:
                    invoked.append(skill)
    return reasoning, observed_tools, invoked


class PiCost(BaseModel):
    """Represent cost breakdown from Pi message usage metadata."""

    model_config = ConfigDict(extra="ignore", frozen=True, populate_by_name=True)

    input: NonNegativeFloat | None = None
    output: NonNegativeFloat | None = None
    cache_read: NonNegativeFloat | None = Field(default=None, alias="cacheRead")
    cache_write: NonNegativeFloat | None = Field(default=None, alias="cacheWrite")
    total: NonNegativeFloat | None = None


class PiUsage(BaseModel):
    """Represent token and cost usage metrics from Pi assistant messages."""

    model_config = ConfigDict(extra="ignore", frozen=True, populate_by_name=True)

    input: NonNegativeInt = 0
    output: NonNegativeInt = 0
    cache_read: NonNegativeInt = Field(default=0, alias="cacheRead")
    cache_write: NonNegativeInt = Field(default=0, alias="cacheWrite")
    reasoning: NonNegativeInt = 0
    total_tokens: NonNegativeInt = Field(default=0, alias="totalTokens")
    prompt_tokens: NonNegativeInt | None = Field(default=None, alias="promptTokens")
    cost: PiCost | None = None

    @property
    def total_prompt_tokens(self) -> int | None:
        """Calculate total input/prompt tokens across direct and cached prompt segments."""
        pi_input = self.input + self.cache_read + self.cache_write
        if pi_input > 0:
            return pi_input
        return self.prompt_tokens


def _extract_pi_message_usage(msg: dict[str, Any]) -> tuple[float | None, int | None]:
    """Extract total cost and prompt tokens from message usage metadata via PiUsage."""
    usage_raw = msg.get("usage")
    if not isinstance(usage_raw, dict):
        return None, None
    try:
        usage = PiUsage.model_validate(usage_raw)
    except ValidationError:
        return None, None
    cost_val = usage.cost.total if usage.cost is not None else None
    return cost_val, usage.total_prompt_tokens


def parse_session_entries(
    entries: Iterable[dict[str, Any]],
    resident: Iterable[str],
) -> SessionSummary:
    """Parse session entries from Pi JSONL to extract skill selection, cost, and tokens."""
    invoked: list[str] = []
    reasoning: list[str] = []
    observed_tools: list[str] = []
    cost_usd: float | None = None
    prompt_tokens: int | None = None
    resolved_model = ""
    assistant_turns = 0

    has_entries = False
    for entry in entries:
        has_entries = True
        msg = entry.get("message")
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue

        assistant_turns += 1
        if "model" in msg and isinstance(msg["model"], str):
            resolved_model = msg["model"]

        cost, toks = _extract_pi_message_usage(msg)
        if cost is not None:
            current_cost = cost_usd if cost_usd is not None else 0.0
            cost_usd = round(current_cost + cost, 6)
        if toks is not None:
            current_tokens = prompt_tokens if prompt_tokens is not None else 0
            prompt_tokens = current_tokens + toks

        m_reasoning, m_tools, m_invoked = _extract_pi_message_content(msg, resident)
        reasoning.extend(m_reasoning)
        observed_tools.extend(m_tools)
        invoked.extend(m_invoked)

    return SessionSummary(
        invoked_skills=tuple(invoked),
        turns_taken=max(1, assistant_turns),
        reasoning=tuple(reasoning),
        observed_tools=tuple(observed_tools),
        cost_usd=cost_usd,
        prompt_tokens=prompt_tokens,
        resolved_model=resolved_model,
        status=SessionStatus.SUCCESS if has_entries else None,
        error=None,
    )


def _apply_pi_env(
    env: dict[str, str],
    options: PiOptions,
    isolation_dir: Path | None = None,
) -> dict[str, str]:
    """Apply Pi telemetry suppression, API keys, and agent directory isolation to environment."""
    env["PI_TELEMETRY"] = "0"
    env["PI_SKIP_VERSION_CHECK"] = "1"
    apply_provider_api_key(
        env,
        provider=options.provider,
        api_key=options.api_key,
        default_provider="google",
    )
    sync_google_and_gemini_keys(env)
    if isolation_dir is not None:
        agent_dir = ensure_private_directory(isolation_dir)
        env["PI_CODING_AGENT_DIR"] = str(agent_dir)
    return env


class PiRuntime(CliAgentRuntime[PiOptions]):
    """Drive the Pi agent harness CLI as an evaluation runtime."""

    name = "pi"
    options: PiOptions
    api_key_env_var: str | None = None
    _skills_subpath: str = ".pi/skills"
    isolation_dir_name: ClassVar[str | None] = ".reach_pi_agent"

    @override
    def parse_stream(
        self,
        lines: Iterable[str],
        resident: Sequence[str] = (),
        early_exit: bool = False,
    ) -> SessionSummary:
        """Parse session entries or stdout lines into a standardized SessionSummary."""
        entries = list(iter_json_lines(lines))
        summary = parse_session_entries(entries, resident or self._resident)
        if early_exit:
            summary = summary.model_copy(
                update={"early_exit": True, "status": summary.status or SessionStatus.SUCCESS},
            )
        return summary

    @property
    def effective_effort(self) -> str | None:
        """Return configured reasoning effort or default from model profile."""
        return self.options.resolve_thinking(super().effective_effort)

    def build_command(self, query_text: str, session_dir: Path | None = None) -> list[str]:
        """Assemble command-line arguments for running a single-turn Pi probe."""
        options = self.options
        effective_session_dir = session_dir or Path(SESSION_DIRNAME)
        cmd = [
            options.executable,
            "-p",
            query_text,
            "--session-dir",
            str(effective_session_dir),
            "--tools",
            options.tools,
            "--no-context-files",
            "--no-prompt-templates",
            "--approve",
        ]
        cmd += options.common_cli_args(
            model=options.model,
            thinking=self.effective_effort,
        )
        if options.extra_args:
            cmd += list(options.extra_args)
        return cmd

    @override
    def build_env(self, workdir: Path | None = None) -> dict[str, str]:
        """Assemble process environment with telemetry suppression, isolation, and API keys."""
        return _apply_pi_env(
            super().build_env(workdir),
            self.options,
            self.effective_isolation_dir(workdir),
        )

    @override
    def post_probe(self, workdir: Path) -> None:
        """Clean session and agent artifacts after probe execution if auto_clean is enabled."""
        if not self.options.auto_clean:
            return
        session_root = (Path(workdir) / SESSION_DIRNAME).resolve()
        slot_dir = probe_slot_dir(session_root)
        if slot_dir.exists():
            shutil.rmtree(slot_dir, ignore_errors=True)
        # Prune the root only once the last concurrent worker has released its slot.
        with contextlib.suppress(OSError):
            session_root.rmdir()
        super().post_probe(workdir)

    def _select_unmeasured(
        self,
        query_text: str,
        workdir: Path,
        target_skill: str | None = None,
    ) -> SelectionOutcome:
        """Execute query evaluation probe without attaching wall-clock duration."""
        session_dir = ensure_private_directory(probe_slot_dir(Path(workdir) / SESSION_DIRNAME))
        existing_files = set(session_dir.glob("*.jsonl"))

        cmd = self.build_command(query_text, session_dir)
        env = self.build_env(workdir)
        completed, err = run_subprocess_probe(cmd, workdir, self.timeout_s, env=env)
        if err is not None or completed is None:
            return SelectionOutcome(
                error=format_subprocess_error("pi", err, self.timeout_s),
                observed_catalog=self._resident,
            )

        if completed.returncode != 0:
            return SelectionOutcome(
                error=process_failure_reason(completed),
                observed_catalog=self._resident,
            )

        session_file = _newest_transcript(session_dir, exclude=existing_files)
        if session_file is None:
            session_file = _newest_transcript(session_dir)

        if session_file is None:
            return SelectionOutcome(
                error="pi produced no session transcript file",
                observed_catalog=self._resident,
            )

        try:
            lines = session_file.read_text(encoding="utf-8").splitlines()
            entries = list(iter_json_lines(lines))
            summary = parse_session_entries(entries, self._resident)
            outcome = self.make_tracker(target_skill).apply_to_outcome(
                summary.to_outcome(
                    self._resident,
                    fallback_model=self.model,
                ),
            )
            if validation_error := self.validate_outcome(summary, workdir):
                return outcome.model_copy(update={"error": validation_error})
            return outcome
        except (OSError, ValueError) as exc:
            return SelectionOutcome(
                error=f"failed to parse pi session log: {exc}",
                observed_catalog=self._resident,
            )

    @override
    def select(
        self,
        query_text: str,
        workdir: Path,
        target_skill: str | None = None,
    ) -> SelectionOutcome:
        """Execute query evaluation probe and return SelectionOutcome."""
        t0 = time.monotonic()
        try:
            outcome = self._select_unmeasured(query_text, workdir, target_skill=target_skill)
            elapsed_ms = max(1, int((time.monotonic() - t0) * 1000))
            return outcome.model_copy(update={"duration_ms": outcome.duration_ms or elapsed_ms})
        finally:
            self.post_probe(workdir)


class PiGenerator(BaseTextGenerator[PiOptions]):
    """Generate text completions using the Pi CLI."""

    name: str = "pi"
    options: PiOptions

    def __init__(
        self,
        model: str = "",
        *,
        timeout_s: int = 300,
        options: PiOptions | None = None,
    ) -> None:
        """Initialize Pi generator with model, timeout, and options."""
        if isinstance(options, PiOptions):
            opts = options
        elif model:
            opts = PiOptions(model=model)
        else:
            opts = PiOptions()
        super().__init__(model=opts.model or model, timeout_s=timeout_s, options=opts)

    @property
    @override
    def effective_effort(self) -> str | None:
        """Return configured reasoning effort or default from model profile."""
        return self.options.resolve_thinking(super().effective_effort)

    @override
    def build_completion_command(self, prompt: str = "") -> list[str]:
        """Assemble command-line arguments for raw text completion."""
        del prompt  # Prompt is passed via stdin to avoid Linux MAX_ARG_STRLEN limits
        cmd = [
            self.options.executable,
            "-p",  # Boolean flag (--print); pi reads stdin when no positional message is given
            "--no-session",
            "--no-skills",
            "--no-context-files",
            "--no-prompt-templates",
        ]
        cmd += self.options.common_cli_args(
            model=self.model,
            thinking=self.effective_effort,
        )
        return [*cmd, *self.options.extra_args]

    @override
    def build_env(self) -> dict[str, str]:
        """Assemble process environment with API keys, isolation, and telemetry suppression."""
        return _apply_pi_env(
            super().build_env(),
            self.options,
            self.options.custom_isolation_dir,
        )
