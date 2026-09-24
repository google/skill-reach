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

"""Verify the Goose agent runtime: transcript parsing, CLI arguments, isolation, and live probes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable

from reach.config import RuntimeSettings, agent_default_model
from reach.runtime.goose import (
    GooseGenerator,
    GooseOptions,
    GooseRuntime,
    parse_goose_output,
    resolve_skill_from_tool_call,
)

from .conftest import goose_payload


@pytest.fixture
def runtime() -> GooseRuntime:
    """Provide a GooseRuntime instance with default options."""
    return GooseRuntime()


def test_goose_options_defaults() -> None:
    """Verify default GooseOptions values."""
    opts = GooseOptions()
    assert opts.executable == "goose"
    assert opts.model == agent_default_model("goose")
    assert opts.provider is None
    assert opts.home_dir is None
    assert opts.max_turns == 3
    assert opts.early_exit is True
    assert opts.no_profile is True
    assert opts.with_builtin == "skills"
    assert opts.effort is None
    assert opts.extra_args == ()
    assert opts.use_symlinks is True
    assert opts.isolate_config_dir is True
    assert opts.auto_clean is False

    opts_with_effort = GooseOptions(effort="high")
    assert opts_with_effort.effort == "high"


def test_resolve_skill_from_tool_call_load_skill(resident_names: tuple[str, ...]) -> None:
    """Verify skill name extraction from load_skill tool call arguments."""
    assert (
        resolve_skill_from_tool_call(
            "load_skill",
            {"name": "pizza-calculator"},
            resident_names,
        )
        == "pizza-calculator"
    )
    assert (
        resolve_skill_from_tool_call(
            "skills__load_skill",
            {"name": "cloud-deploy"},
            resident_names,
        )
        == "cloud-deploy"
    )
    assert (
        resolve_skill_from_tool_call(
            "load_skill",
            {"name": "unknown-skill"},
            resident_names,
        )
        is None
    )


def test_resolve_skill_from_tool_call_path_argument(resident_names: tuple[str, ...]) -> None:
    """Verify skill name extraction from file path arguments (e.g. read / developer tools)."""
    assert (
        resolve_skill_from_tool_call(
            "developer__text_editor",
            {"path": "/workspace/.agents/skills/cloud-deploy/SKILL.md"},
            resident_names,
        )
        == "cloud-deploy"
    )
    assert (
        resolve_skill_from_tool_call(
            "read",
            {"path": "relative/pizza-calculator/skill.md"},
            resident_names,
        )
        == "pizza-calculator"
    )
    assert (
        resolve_skill_from_tool_call(
            "read",
            {"path": "/workspace/README.md"},
            resident_names,
        )
        is None
    )


def test_resolve_skill_from_tool_call_non_string_arguments(
    resident_names: tuple[str, ...],
) -> None:
    """Verify resolve_skill_from_tool_call safely handles non-string, None, and empty arguments."""
    assert resolve_skill_from_tool_call("load_skill", {"name": None}, resident_names) is None
    assert resolve_skill_from_tool_call("load_skill", {"name": 123}, resident_names) is None
    assert resolve_skill_from_tool_call("load_skill", {"name": ""}, resident_names) is None
    assert resolve_skill_from_tool_call("load_skill", {}, resident_names) is None
    assert resolve_skill_from_tool_call("read", {"path": None}, resident_names) is None
    assert resolve_skill_from_tool_call("read", {"path": 123}, resident_names) is None
    assert resolve_skill_from_tool_call("read", {"path": ["/skill.md"]}, resident_names) is None
    assert resolve_skill_from_tool_call("", {"path": "/workspace"}, resident_names) is None
    assert resolve_skill_from_tool_call("read", None, resident_names) is None


def test_parse_goose_output_extracts_load_skill() -> None:
    """Verify parsing of Goose JSON output containing load_skill toolRequest."""
    resident = ("cloud-deploy", "pizza-calculator")
    sample_payload = goose_payload(invoked="pizza-calculator")

    summary = parse_goose_output(sample_payload, resident)
    assert summary.invoked_skill == "pizza-calculator"
    assert summary.invoked_skills == ("pizza-calculator",)
    assert summary.observed_tools == ("load_skill",)
    assert summary.cost_usd == pytest.approx(0.0042)
    assert summary.error is None


def test_parse_goose_output_preserves_repeated_invocations_sequence() -> None:
    """Verify Goose parser preserves full trajectory sequence without deduplication."""
    resident = ("cloud-deploy", "pizza-calculator")
    payload = goose_payload(
        invoked_skills=("cloud-deploy", "pizza-calculator", "cloud-deploy"),
    )
    summary = parse_goose_output(payload, resident)
    assert summary.invoked_skills == ("cloud-deploy", "pizza-calculator", "cloud-deploy")
    assert summary.invoked_skill == "cloud-deploy"


def test_parse_goose_output_extracts_reasoning() -> None:
    """Verify assistant reasoning text is extracted into reasoning tuple."""
    resident = ("pizza-calculator",)
    payload = {
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Analyzing the calculation request."},
                    {
                        "type": "toolRequest",
                        "toolCall": {
                            "value": {
                                "name": "load_skill",
                                "arguments": {"name": "pizza-calculator"},
                            },
                        },
                    },
                ],
            },
        ],
    }
    summary = parse_goose_output(payload, resident)
    assert summary.invoked_skill == "pizza-calculator"
    assert summary.reasoning == ("Analyzing the calculation request.",)


def test_parse_goose_output_with_startup_banner() -> None:
    """Verify parsing when stdout contains Goose ASCII banner before JSON output."""
    resident = ("pizza-calculator",)
    raw_with_banner = """
    __( O)>  ● new session · gemini_oauth gemini-3.7-flash
   \\____)    20260829_1 · /workspace
     L L     goose is ready
{
  "messages": [
    {
      "role": "assistant",
      "content": [
        {
          "type": "toolRequest",
          "toolCall": {
            "value": {
              "name": "load_skill",
              "arguments": {"name": "pizza-calculator"}
            }
          }
        }
      ]
    }
  ]
}
"""
    summary = parse_goose_output(raw_with_banner, resident)
    assert summary.invoked_skill == "pizza-calculator"
    assert summary.error is None


def test_parse_goose_output_no_tool_call() -> None:
    """Verify parsing when model answers directly without invoking any tools."""
    resident = ("cloud-deploy",)
    sample_payload = {
        "messages": [
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "I can help directly."}],
            },
        ],
        "metadata": {"total_tokens": 100, "status": "completed"},
    }

    summary = parse_goose_output(sample_payload, resident)
    assert summary.invoked_skill is None
    assert summary.invoked_skills == ()
    assert summary.observed_tools == ()
    assert summary.error is None


def test_parse_goose_output_non_resident_skill() -> None:
    """Verify tool call to non-resident skill does not count as invoked."""
    resident = ("cloud-deploy",)
    sample_payload = {
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "toolRequest",
                        "id": "call_999",
                        "toolCall": {
                            "status": "success",
                            "value": {
                                "name": "load_skill",
                                "arguments": {"name": "other-skill"},
                            },
                        },
                    },
                ],
            },
        ],
    }

    summary = parse_goose_output(sample_payload, resident)
    assert summary.invoked_skill is None
    assert summary.invoked_skills == ()
    assert summary.observed_tools == ("load_skill",)


def test_build_command_arguments() -> None:
    """Verify CLI arguments assembled by GooseRuntime."""
    settings = RuntimeSettings(
        agent="goose",
        options={
            "model": "gemini-3.8-flash",
            "provider": "gemini_oauth",
            "max_turns": 2,
            "extra_args": ("--debug",),
        },
    )
    rt = GooseRuntime(settings)
    cmd = rt.build_command("how to calculate pizza")

    assert cmd[0] == "goose"
    assert "run" in cmd
    assert "-q" in cmd
    assert "--text" in cmd
    assert "how to calculate pizza" in cmd
    assert "--output-format" in cmd
    assert "json" in cmd
    assert "--no-session" in cmd
    assert "--max-turns" in cmd
    assert "2" in cmd
    assert "--no-profile" in cmd
    assert "--with-builtin" in cmd
    assert "skills" in cmd
    assert "--model" in cmd
    assert "gemini-3.8-flash" in cmd
    assert "--provider" in cmd
    assert "gemini_oauth" in cmd
    assert "--debug" in cmd


def test_select_successful_probe(
    mock_subprocess: Callable[..., Any],
    tmp_path: Path,
) -> None:
    """Verify select executes subprocess, parses JSON output, and returns SelectionOutcome."""
    workdir = tmp_path / "work"
    workdir.mkdir()

    rt = GooseRuntime()
    rt._resident = ("pizza-calculator",)

    output = goose_payload(invoked="pizza-calculator", cost_usd=0.01)
    mock_subprocess(stdout=json.dumps(output))

    outcome = rt.select("how to make pizza", workdir)
    assert outcome.invoked_skill == "pizza-calculator"
    assert outcome.invoked_skills == ("pizza-calculator",)
    assert outcome.cost_usd == pytest.approx(0.01)
    assert outcome.error is None


def test_goose_multi_turn_early_exit(
    mock_subprocess: Callable[..., Any],
    tmp_path: Path,
) -> None:
    """Verify GooseRuntime stops and records early exit when target skill is hit."""
    workdir = tmp_path / "work"
    workdir.mkdir()
    rt = GooseRuntime()
    rt._resident = ("s1", "target-skill", "s3")

    output = goose_payload(invoked_skills=("s1", "target-skill", "s3"))
    lines = [json.dumps({"messages": [m]}) for m in output["messages"]]
    mock_subprocess(lines=lines)
    outcome = rt.select("how to make pizza", workdir, target_skill="target-skill")
    assert outcome.early_exit is True
    assert outcome.invoked_skills == ("s1", "target-skill")
    assert outcome.turns_taken == 2
    assert outcome.error is None


def test_goose_parse_stream_whitespace_and_early_exit(runtime: GooseRuntime) -> None:
    """Verify parse_stream handles whitespace input and early exit status."""
    ws_summary = runtime.parse_stream(["   \n", " \t \n"])
    assert ws_summary.invoked_skill is None
    assert ws_summary.error is None

    early_exit_summary = runtime.parse_stream([], early_exit=True)
    assert early_exit_summary.early_exit is True
    assert early_exit_summary.status == "SUCCESS"


def test_goose_build_completion_command() -> None:
    """Verify build_completion_command constructs arguments correctly without CLI prompt."""
    gen = GooseGenerator()
    cmd = gen.build_completion_command("test prompt")
    assert cmd[:6] == ["goose", "run", "-q", "-i", "-", "--no-session"]
    assert "-t" not in cmd
    assert "test prompt" not in cmd
    assert "--no-profile" in cmd


def test_goose_build_env_defaults_and_isolation(tmp_path: Path) -> None:
    """Verify build_env sets telemetry suppression, isolated home, and XDG directories."""
    rt = GooseRuntime()
    workdir = tmp_path / "work"
    workdir.mkdir()

    env = rt.build_env(workdir)
    assert env["OTEL_SDK_DISABLED"] == "true"
    assert env["HOME"] == str(workdir / ".reach_goose")
    assert env["XDG_CONFIG_HOME"] == str(workdir / ".reach_goose" / ".config")
    assert env["XDG_DATA_HOME"] == str(workdir / ".reach_goose" / ".local" / "share")
    assert env["XDG_STATE_HOME"] == str(workdir / ".reach_goose" / ".local" / "state")
    assert (workdir / ".reach_goose").is_dir()


def test_goose_build_env_provider_api_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify build_env maps api_key to appropriate provider environment variables."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    # Test default provider (openai)
    rt_default = GooseRuntime(RuntimeSettings(agent="goose", options={"api_key": "openai-key"}))
    env_default = rt_default.build_env()
    assert env_default["OPENAI_API_KEY"] == "openai-key"
    assert "ANTHROPIC_API_KEY" not in env_default
    assert "GEMINI_API_KEY" not in env_default

    # Test explicit api_key option for Google
    rt_custom = GooseRuntime(
        RuntimeSettings(
            agent="goose",
            options={"provider": "google", "api_key": "custom-key"},
        )
    )
    env_custom = rt_custom.build_env()
    assert env_custom["GEMINI_API_KEY"] == "custom-key"
    assert env_custom["GOOGLE_API_KEY"] == "custom-key"
    assert "OPENAI_API_KEY" not in env_custom

    # Test explicit api_key option for Anthropic
    rt_anthropic = GooseRuntime(
        RuntimeSettings(
            agent="goose",
            options={"provider": "anthropic", "api_key": "anthropic-key"},
        )
    )
    env_anthropic = rt_anthropic.build_env()
    assert env_anthropic["ANTHROPIC_API_KEY"] == "anthropic-key"
    assert "GEMINI_API_KEY" not in env_anthropic


def test_select_passes_env_and_cleans_when_auto_clean(
    mock_subprocess: Callable[..., Any],
    tmp_path: Path,
) -> None:
    """Verify select passes isolated env to subprocess and cleans state when auto_clean is True."""
    workdir = tmp_path / "work"
    workdir.mkdir()

    rt = GooseRuntime(RuntimeSettings(agent="goose", options={"auto_clean": True}))
    rt._resident = ("test-skill",)

    captured_env: dict[str, str] = {}

    def mock_run(cmd: list[str], **kwargs: Any) -> Any:
        nonlocal captured_env
        captured_env = dict(kwargs.get("env", {}))
        output = {
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "toolRequest",
                            "toolCall": {
                                "status": "success",
                                "value": {
                                    "name": "load_skill",
                                    "arguments": {"name": "test-skill"},
                                },
                            },
                        },
                    ],
                },
            ],
            "metadata": {"cost_usd": 0.01},
        }
        import subprocess

        return subprocess.CompletedProcess(cmd, returncode=0, stdout=json.dumps(output), stderr="")

    mock_subprocess(handler=mock_run)

    outcome = rt.select("test query", workdir)
    assert outcome.invoked_skill == "test-skill"
    assert captured_env.get("OTEL_SDK_DISABLED") == "true"
    assert captured_env.get("HOME") == str(workdir / ".reach_goose")
    assert captured_env.get("XDG_CONFIG_HOME") == str(workdir / ".reach_goose" / ".config")

    # Because auto_clean=True, isolated directory should be removed in post_probe
    assert not (workdir / ".reach_goose").exists()


def test_goose_runtime_initializes_base_attributes() -> None:
    """Verify GooseRuntime properly calls base class __init__ and sets tracking attributes."""
    opts = GooseOptions(auto_clean=False)
    rt = GooseRuntime(options=opts)
    assert rt.options == opts
    assert rt.completions == 0
    assert rt.completion_cost_usd == 0.0
    assert rt.is_cli is True


def test_goose_generator_command_and_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify GooseGenerator command line assembly, environment, and schema formatting."""
    opts = GooseOptions(
        model="gemini-3.8-flash",
        provider="google",
        api_key="test-secret-key",
        no_profile=True,
    )
    generator = GooseGenerator(model="gemini-3.8-flash", options=opts)

    cmd = generator.build_completion_command("test prompt")
    assert cmd[:6] == ["goose", "run", "-q", "-i", "-", "--no-session"]
    assert "-t" not in cmd
    assert "--no-profile" in cmd
    assert "--model" in cmd
    assert cmd[cmd.index("--model") + 1] == "gemini-3.8-flash"
    assert "--provider" in cmd
    assert cmd[cmd.index("--provider") + 1] == "google"

    env = generator.build_env()
    assert env["OTEL_SDK_DISABLED"] == "true"
    assert env.get("GEMINI_API_KEY") == "test-secret-key"

    captured: dict[str, Any] = {}

    def mock_run(command: list[str], **kwargs: Any) -> Any:
        captured["command"] = command
        captured["env"] = kwargs.get("env")
        captured["input"] = kwargs.get("input")
        import subprocess

        return subprocess.CompletedProcess(
            command, returncode=0, stdout='{"queries": []}', stderr=""
        )

    monkeypatch.setattr("subprocess.run", mock_run)

    schema = {"type": "object", "properties": {"queries": {"type": "array"}}}
    result = generator.complete("draft queries", schema=schema)
    assert result == '{"queries": []}'
    assert captured["env"]["OTEL_SDK_DISABLED"] == "true"
    assert "Respond with valid JSON adhering to this JSON schema:" in captured["input"]
    assert "-t" not in captured["command"]


def test_parse_goose_output_prompt_tokens_and_gemini_provider_normalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify parse_goose_output extracts prompt_tokens and maps gemini-* to google."""
    from reach.runtime.goose import (
        GooseGenerator,
        GooseOptions,
        GooseRuntime,
        _extract_goose_metadata,
        parse_goose_output,
    )

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    assert GooseOptions(provider="gemini").provider == "google"
    assert GooseOptions(provider="  anthropic  ").provider == "anthropic"

    rt = GooseRuntime(RuntimeSettings(agent="goose", options={"model": "gemini-3.8-flash"}))
    cmd = rt.build_command("test query")
    assert "--provider" in cmd
    assert cmd[cmd.index("--provider") + 1] == "google"

    rt_explicit = GooseRuntime(
        RuntimeSettings(agent="goose", options={"model": "gemini-3.8-flash", "provider": "gemini"})
    )
    assert rt_explicit.options.provider == "google"
    cmd_explicit = rt_explicit.build_command("test query")
    assert cmd_explicit[cmd_explicit.index("--provider") + 1] == "google"

    gen = GooseGenerator(model="gemini-3.8-flash")
    gen_cmd = gen.build_completion_command("draft queries")
    assert "--provider" in gen_cmd
    assert gen_cmd[gen_cmd.index("--provider") + 1] == "google"

    assert _extract_goose_metadata({"input_tokens": -5, "cost_usd": 0.01}) == (None, None)

    payload = {
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "toolRequest",
                        "toolCall": {
                            "value": {
                                "name": "load_skill",
                                "arguments": {"name": "alpha"},
                            }
                        },
                    }
                ],
            }
        ],
        "metadata": {
            "total_tokens": 755,
            "input_tokens": 700,
            "output_tokens": 30,
            "cache_read_input_tokens": 20,
            "cache_write_input_tokens": 5,
            "cost_usd": 0.00064,
        },
    }
    summary = parse_goose_output(payload, resident=("alpha",))
    assert summary.invoked_skills == ("alpha",)
    assert summary.prompt_tokens == 725
    assert summary.cost_usd == pytest.approx(0.00064)
