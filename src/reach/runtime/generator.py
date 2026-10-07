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

"""Provide TextGenerator protocol, base generator class, and generator factory."""

from __future__ import annotations

import importlib
import json
import os
import subprocess
from abc import ABC
from collections.abc import Mapping
from math import floor
from typing import TYPE_CHECKING, Any, Protocol, cast, runtime_checkable

from reach.config import DEFAULT_GEMINI_MODEL
from reach.runtime._env import (
    raise_missing_agent_dependency,
    resolve_blocked_env_vars,
    sanitize_subprocess_env,
)
from reach.runtime.profiles import model_profile

if TYPE_CHECKING:
    from reach.runtime import AgentOptions

__all__ = [
    "BaseTextGenerator",
    "TextGenerator",
    "build_text_generator",
]


@runtime_checkable
class TextGenerator(Protocol):
    """Protocol for models or drivers capable of generating raw text completions."""

    options: Any

    @property
    def name(self) -> str:
        """Return generator name or agent identifier."""
        ...

    @property
    def model(self) -> str:
        """Return the model identifier used for generation."""
        ...

    @model.setter
    def model(self, value: str) -> None:
        """Update configured model identifier."""
        ...

    @property
    def completions(self) -> int:
        """Return total number of completions executed."""
        ...

    @property
    def completion_cost_usd(self) -> float:
        """Return total cost accumulated across completions in USD."""
        ...

    def complete(self, prompt: str, *, schema: str | Mapping[str, Any] | None = None) -> str:
        """Generate a raw text completion for an arbitrary prompt."""
        ...

    def prompt_budget_chars(self) -> int | None:
        """Return maximum character length for prompts, or None if unbounded."""
        ...


class BaseTextGenerator[OptionsT: AgentOptions](ABC):
    """Abstract base class providing character budgeting and completion accounting."""

    name: str = "generator"
    options: OptionsT

    def __init__(
        self,
        model: str,
        *,
        timeout_s: int = 300,
        options: OptionsT | None = None,
    ) -> None:
        """Initialize generator with model ID, timeout, and options."""
        self._model = model
        self.timeout_s = timeout_s
        self.options = cast("OptionsT", options)
        self.completions = 0
        self.completion_cost_usd = 0.0

    @property
    def model(self) -> str:
        """Return configured model identifier."""
        return self._model

    @model.setter
    def model(self, value: str) -> None:
        """Update configured model identifier."""
        self._model = value
        copy_fn = getattr(self.options, "model_copy", None)
        if callable(copy_fn):
            self.options = cast("OptionsT", copy_fn(update={"model": value}))

    @property
    def effective_effort(self) -> str | None:
        """Return configured reasoning effort or default from model profile."""
        if self.options is not None and self.options.effort:
            return None if self.options.effort.lower() in ("none", "off") else self.options.effort
        try:
            return model_profile(self.model).effort
        except (KeyError, ValueError):
            return None

    def prompt_budget_chars(self) -> int | None:
        """Return maximum character length for prompts, or None if unbounded."""
        try:
            profile = model_profile(self.model)
            window = profile.completion_window or profile.context_window
            return floor(window * profile.chars_per_token)
        except (KeyError, ValueError):
            return None

    @property
    def blocked_env_vars(self) -> tuple[str, ...] | None:
        """Return blocked environment variables configured on options or settings."""
        return resolve_blocked_env_vars(self.options, getattr(self, "settings", None))

    def build_env(self) -> dict[str, str]:
        """Assemble sanitized process environment for generator execution."""
        return sanitize_subprocess_env(dict(os.environ), blocked_env_vars=self.blocked_env_vars)

    def resolve_schema(
        self,
        schema: str | Mapping[str, Any] | None = None,
    ) -> str | None:
        """Resolve explicit schema argument or configured options schema to a JSON string."""
        effective = schema if schema is not None else getattr(self.options, "json_schema", None)
        if isinstance(effective, Mapping):
            return json.dumps(effective)
        return effective

    def format_prompt_with_schema(
        self,
        prompt: str,
        schema: str | Mapping[str, Any] | None = None,
    ) -> str:
        """Append schema instruction to prompt when schema is provided and not already present."""
        schema_str = self.resolve_schema(schema)
        if not schema_str or schema_str in prompt:
            return prompt
        return f"{prompt}\n\nRespond with valid JSON adhering to this JSON schema:\n{schema_str}"

    def build_completion_command(self, prompt: str = "") -> list[str]:
        """Assemble command-line arguments for raw text completion."""
        del prompt
        msg = f"{type(self).__name__} does not implement build_completion_command()"
        raise NotImplementedError(msg)

    def complete(self, prompt: str, *, schema: str | Mapping[str, Any] | None = None) -> str:
        """Execute text completion subprocess piping prompt via stdin and return response string."""
        effective_prompt = self.format_prompt_with_schema(prompt, schema)
        completed: subprocess.CompletedProcess[str] = subprocess.run(
            self.build_completion_command(effective_prompt),
            input=effective_prompt,
            capture_output=True,
            text=True,
            timeout=self.timeout_s,
            check=False,
            env=self.build_env(),
        )
        if completed.returncode != 0:
            reason = completed.stderr.strip() or f"exit code {completed.returncode}"
            msg = f"generation failed: {reason}"
            raise RuntimeError(msg)
        self.completions += 1
        return completed.stdout.strip()


def _build_agent_generator(
    target_agent: str,
    model: str,
    timeout_s: int,
    opts: dict[str, Any],
) -> BaseTextGenerator[Any] | None:
    """Instantiate a concrete TextGenerator matching the requested agent name."""
    if target_agent == "fake":
        from reach.runtime.fake import FakeGenerator, FakeOptions, resolve_fake_options

        fake_opts = resolve_fake_options(opts) if opts else FakeOptions(model=model)
        budget = opts.get("prompt_budget_chars")
        return FakeGenerator(
            model=fake_opts.model or model,
            prompt_budget_chars=budget if isinstance(budget, int) else None,
            cost_usd=float(opts.get("cost_usd", 0.0) or 0.0),
            timeout_s=timeout_s,
        )
    if target_agent == "keyword":
        from reach.runtime.keyword import KeywordGenerator, KeywordOptions

        kw_opts = KeywordOptions.model_validate({"model": model, **opts})
        return KeywordGenerator(model=model, timeout_s=timeout_s, options=kw_opts)

    cli_map = {
        "claude-code": (
            "reach.runtime.claude_code",
            "ClaudeCodeOptions",
            "ClaudeGenerator",
        ),
        "antigravity-cli": (
            "reach.runtime.antigravity_cli",
            "AntigravityCliOptions",
            "AntigravityCliGenerator",
        ),
        "goose": ("reach.runtime.goose", "GooseOptions", "GooseGenerator"),
        "pi": ("reach.runtime.pi", "PiOptions", "PiGenerator"),
        "antigravity-sdk": (
            "reach.runtime.antigravity_sdk",
            "AntigravitySdkOptions",
            "AntigravitySdkGenerator",
        ),
    }
    spec = cli_map.get(target_agent)
    if spec is not None:
        mod_name, opt_cls_name, gen_cls_name = spec
        try:
            mod = importlib.import_module(mod_name)
        except ImportError as err:
            raise_missing_agent_dependency(target_agent, err, role="generator")
        opt_cls = getattr(mod, opt_cls_name)
        gen_cls = getattr(mod, gen_cls_name)
        validated_opts = opt_cls.model_validate({"model": model, **opts})
        return gen_cls(model=model, timeout_s=timeout_s, options=validated_opts)
    return None


def build_text_generator(
    model: str | None = None,
    agent: str | None = None,
    timeout_s: float | None = None,
    options: Mapping[str, Any] | None = None,
) -> BaseTextGenerator[Any]:
    """Construct a TextGenerator instance configured for query drafting or optimization."""
    from reach.config import agent_default_model, default_agent
    from reach.runtime import known_agents

    target_agent = agent or default_agent()
    opts = dict(options or {})
    target_model = (
        model
        or (opts.get("model") if isinstance(opts.get("model"), str) else None)
        or agent_default_model(target_agent)
        or DEFAULT_GEMINI_MODEL
    )
    opts["model"] = target_model

    timeout_int = int(timeout_s) if timeout_s is not None else 300
    gen = _build_agent_generator(target_agent, str(target_model), timeout_int, opts)
    if gen is not None:
        return gen

    if target_agent not in known_agents():
        agents = ", ".join(known_agents())
        msg = f"unknown runtime agent {target_agent!r}; expected one of {agents}"
        raise ValueError(msg)

    msg = f"runtime agent {target_agent!r} does not support text generation"
    raise ValueError(msg)
