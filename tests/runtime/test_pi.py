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

"""Verify the Pi agent runtime: transcript parsing, CLI arguments, and isolation."""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable

from reach.config import RuntimeSettings, agent_default_model
from reach.runtime.pi import (
    PiGenerator,
    PiOptions,
    PiRuntime,
    parse_session_entries,
)

from .conftest import pi_entries


@pytest.fixture
def runtime() -> PiRuntime:
    """Provide a PiRuntime instance with default options."""
    return PiRuntime()


def test_parse_session_entries_extracts_skill_invocation() -> None:
    """Verify parsing of Pi session entries containing toolCall to read."""
    resident = ("cloud-deploy", "pizza-calculator")
    entries = pi_entries(invoked="cloud-deploy")

    summary = parse_session_entries(entries, resident)
    assert summary.invoked_skill == "cloud-deploy"
    assert summary.invoked_skills == ("cloud-deploy",)
    assert summary.observed_tools == ("read",)
    assert summary.cost_usd == pytest.approx(0.0125)
    assert summary.resolved_model == "gemini-3.5-flash"
    assert summary.error is None


def test_parse_session_entries_preserves_repeated_invocations_sequence() -> None:
    """Verify Pi parser preserves full trajectory sequence without deduplication."""
    resident = ("cloud-deploy", "pizza-calculator")
    entries = pi_entries(
        invoked_skills=("cloud-deploy", "pizza-calculator", "cloud-deploy"),
        include_session=False,
    )
    summary = parse_session_entries(entries, resident)
    assert summary.invoked_skills == ("cloud-deploy", "pizza-calculator", "cloud-deploy")
    assert summary.invoked_skill == "cloud-deploy"


def test_parse_session_entries_extracts_reasoning() -> None:
    """Verify assistant text and thinking content are extracted into reasoning tuple."""
    resident = ("cloud-deploy",)
    entries = [
        {
            "type": "message",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "text": "Determining best deployment skill."},
                    {
                        "type": "toolCall",
                        "id": "call_01",
                        "name": "read",
                        "arguments": {"path": "/workspace/.pi/skills/cloud-deploy/SKILL.md"},
                    },
                ],
            },
        },
    ]
    summary = parse_session_entries(entries, resident)
    assert summary.invoked_skill == "cloud-deploy"
    assert summary.reasoning == ("Determining best deployment skill.",)


def test_parse_session_entries_no_tool_call() -> None:
    """Verify parsing when model answers directly without invoking read tool."""
    resident = ("cloud-deploy",)
    entries = [
        {
            "type": "message",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "I can help with that directly."}],
                "model": "gemini-3.5-flash",
            },
        },
    ]

    summary = parse_session_entries(entries, resident)
    assert summary.invoked_skill is None
    assert summary.invoked_skills == ()
    assert summary.observed_tools == ()
    assert summary.error is None


def test_parse_session_entries_non_skill_tool_call() -> None:
    """Verify tool calls to non-skill files do not count as skill invocation."""
    resident = ("cloud-deploy",)
    entries = [
        {
            "type": "message",
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "toolCall",
                        "id": "call_02",
                        "name": "read",
                        "arguments": {"path": "/workspace/package.json"},
                    },
                ],
            },
        },
    ]

    summary = parse_session_entries(entries, resident)
    assert summary.invoked_skill is None
    assert summary.invoked_skills == ()
    assert summary.observed_tools == ("read",)


def test_pi_options_defaults() -> None:
    """Verify default PiOptions values."""
    opts = PiOptions()
    assert opts.executable == "pi"
    assert opts.tools == "read"
    assert opts.model == agent_default_model("pi")
    assert opts.provider == "google"
    assert opts.thinking is None
    assert opts.use_symlinks is True
    assert opts.no_themes is True
    assert opts.isolate_config_dir is True
    assert opts.auto_clean is False


def test_build_command_arguments(tmp_path: Path) -> None:
    """Verify command list assembled by PiRuntime."""
    settings = RuntimeSettings(
        agent="pi",
        options={
            "model": "gemini-3.5-flash",
            "provider": "google",
            "thinking": "high",
        },
    )
    rt = PiRuntime(settings)
    session_dir = tmp_path / "sessions"
    cmd = rt.build_command("test query", session_dir=session_dir)

    assert cmd[0] == "pi"
    assert "-p" in cmd
    assert "test query" in cmd
    assert "--session-dir" in cmd
    assert str(session_dir) in cmd
    assert "--tools" in cmd
    assert "read" in cmd
    assert "--no-context-files" in cmd
    assert "--no-prompt-templates" in cmd
    assert "--no-extensions" in cmd
    assert "--no-themes" in cmd
    assert "--approve" in cmd
    assert "--model" in cmd
    assert "gemini-3.5-flash" in cmd
    assert "--provider" in cmd
    assert "--thinking" in cmd
    assert "high" in cmd


def test_pi_build_command_effort_option(tmp_path: Path) -> None:
    """Verify effort in options maps to --thinking flag in Pi build_command."""
    settings = RuntimeSettings(
        agent="pi",
        options={
            "model": "gemini-3.7-flash",
            "effort": "low",
        },
    )
    rt = PiRuntime(settings)
    cmd = rt.build_command("test query", session_dir=tmp_path / "sessions")
    idx = cmd.index("--thinking")
    assert cmd[idx + 1] == "low"


def test_pi_build_command_api_key_option(tmp_path: Path) -> None:
    """Verify api_key in options maps to --api-key flag in Pi build_command."""
    settings = RuntimeSettings(
        agent="pi",
        options={
            "model": "gemini-3.7-flash",
            "api_key": "pi-secret-key",
        },
    )
    rt = PiRuntime(settings)
    cmd = rt.build_command("test query", session_dir=tmp_path / "sessions")
    idx = cmd.index("--api-key")
    assert cmd[idx + 1] == "pi-secret-key"


def test_select_successful_probe(
    mock_subprocess: Callable[..., Any],
    tmp_path: Path,
) -> None:
    """Verify select executes subprocess, finds session file, and returns outcome."""
    workdir = tmp_path / "work"
    workdir.mkdir()

    rt = PiRuntime()
    rt._resident = ("test-skill",)

    def mock_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        # Emulate pi creating session file in session_dir
        idx = cmd.index("--session-dir")
        s_dir = Path(cmd[idx + 1])
        s_dir.mkdir(parents=True, exist_ok=True)
        session_file = s_dir / "2026-08-29T20-00-00_uuid.jsonl"
        session_file.write_text(
            json.dumps(
                {
                    "type": "message",
                    "message": {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "toolCall",
                                "name": "read",
                                "arguments": {
                                    "path": str(
                                        workdir / ".pi" / "skills" / "test-skill" / "SKILL.md",
                                    ),
                                },
                            },
                        ],
                        "model": "gemini-3.5-flash",
                        "usage": {"cost": {"total": 0.005}},
                    },
                },
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="OK", stderr="")

    mock_subprocess(handler=mock_run)

    outcome = rt.select("how to test", workdir)
    assert outcome.invoked_skill == "test-skill"
    assert outcome.invoked_skills == ("test-skill",)
    assert outcome.cost_usd == pytest.approx(0.005)
    assert outcome.resolved_model == "gemini-3.5-flash"
    assert outcome.error is None


def test_select_no_session_file_returns_error(
    mock_subprocess: Callable[..., Any],
    tmp_path: Path,
) -> None:
    """Verify select returns error outcome when pi produces no session transcript."""
    workdir = tmp_path / "work"
    workdir.mkdir()

    rt = PiRuntime()
    rt._resident = ("test-skill",)

    def mock_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        # Emulate pi running without creating any session file
        idx = cmd.index("--session-dir")
        s_dir = Path(cmd[idx + 1])
        s_dir.mkdir(parents=True, exist_ok=True)
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="OK", stderr="")

    mock_subprocess(handler=mock_run)

    outcome = rt.select("how to test", workdir)
    assert outcome.invoked_skill is None
    assert outcome.invoked_skills == ()
    assert outcome.error == "pi produced no session transcript file"
    assert outcome.observed_catalog == ("test-skill",)


def test_pi_build_completion_command() -> None:
    """Verify build_completion_command constructs arguments correctly without CLI prompt."""
    gen = PiGenerator()
    cmd = gen.build_completion_command("test prompt")
    assert cmd[:3] == ["pi", "-p", "--no-session"]
    assert "test prompt" not in cmd
    assert "--no-skills" in cmd
    assert "--no-themes" in cmd


def test_pi_build_command_profile_effort_fallback(tmp_path: Path) -> None:
    """Verify build_command falls back to profile effort when thinking/effort are unset."""
    settings = RuntimeSettings(
        agent="pi",
        options={"model": "gemini-3.7-flash", "provider": "google"},
    )
    rt = PiRuntime(settings)
    cmd = rt.build_command("test query", session_dir=tmp_path / "sessions")
    assert "--thinking" in cmd
    assert "low" in cmd


def test_pi_build_env_defaults_and_isolation(tmp_path: Path) -> None:
    """Verify build_env sets telemetry suppression, skip version check, and isolated agent dir."""
    rt = PiRuntime()
    workdir = tmp_path / "work"
    workdir.mkdir()

    env = rt.build_env(workdir)
    assert env["PI_TELEMETRY"] == "0"
    assert env["PI_SKIP_VERSION_CHECK"] == "1"
    assert env["PI_CODING_AGENT_DIR"] == str(workdir / ".reach_pi_agent")
    assert (workdir / ".reach_pi_agent").is_dir()


def test_pi_build_env_provider_api_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify build_env maps api_key to appropriate provider environment variables."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    # Test default provider (google)
    rt_default = PiRuntime(RuntimeSettings(agent="pi", options={"api_key": "google-key"}))
    env_default = rt_default.build_env()
    assert env_default["GEMINI_API_KEY"] == "google-key"
    assert env_default["GOOGLE_API_KEY"] == "google-key"
    assert "ANTHROPIC_API_KEY" not in env_default
    assert "OPENAI_API_KEY" not in env_default

    # Test explicit api_key option for Anthropic
    rt_anthropic = PiRuntime(
        RuntimeSettings(
            agent="pi",
            options={"provider": "anthropic", "api_key": "anthropic-key"},
        )
    )
    env_anthropic = rt_anthropic.build_env()
    assert env_anthropic["ANTHROPIC_API_KEY"] == "anthropic-key"
    assert "GEMINI_API_KEY" not in env_anthropic

    # Test explicit api_key option for OpenAI
    rt_openai = PiRuntime(
        RuntimeSettings(
            agent="pi",
            options={"provider": "openai", "api_key": "openai-key"},
        )
    )
    env_openai = rt_openai.build_env()
    assert env_openai["OPENAI_API_KEY"] == "openai-key"
    assert "ANTHROPIC_API_KEY" not in env_openai
    assert "GEMINI_API_KEY" not in env_openai


def test_select_passes_env_and_cleans_when_auto_clean(
    mock_subprocess: Callable[..., Any],
    tmp_path: Path,
) -> None:
    """Verify select passes isolated env to subprocess and cleans state when auto_clean is True."""
    workdir = tmp_path / "work"
    workdir.mkdir()

    rt = PiRuntime(RuntimeSettings(agent="pi", options={"auto_clean": True}))
    rt._resident = ("test-skill",)

    captured_env: dict[str, str] = {}

    def mock_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        nonlocal captured_env
        captured_env = dict(kwargs.get("env", {}))
        idx = cmd.index("--session-dir")
        s_dir = Path(cmd[idx + 1])
        s_dir.mkdir(parents=True, exist_ok=True)
        session_file = s_dir / "test_session.jsonl"
        session_file.write_text(
            json.dumps(
                {
                    "type": "message",
                    "message": {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "toolCall",
                                "name": "read",
                                "arguments": {
                                    "path": str(
                                        workdir / ".pi" / "skills" / "test-skill" / "SKILL.md",
                                    ),
                                },
                            },
                        ],
                        "model": "gemini-3.7-flash",
                    },
                },
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="OK", stderr="")

    mock_subprocess(handler=mock_run)

    outcome = rt.select("test query", workdir)
    assert outcome.invoked_skill == "test-skill"
    assert captured_env.get("PI_TELEMETRY") == "0"
    assert captured_env.get("PI_SKIP_VERSION_CHECK") == "1"
    assert captured_env.get("PI_CODING_AGENT_DIR") == str(workdir / ".reach_pi_agent")

    # Because auto_clean=True, session dir should have been removed in post_probe
    assert not (workdir / ".reach_pi_sessions").exists()
    assert not (workdir / ".reach_pi_agent").exists()


#: Resident skills used by the concurrent probe isolation tests.
SLOT_SKILLS = ("alpha", "beta")


def write_pi_session(path: Path, skill: str) -> None:
    """Write a Pi JSONL transcript recording a single skill invocation."""
    path.write_text(
        "\n".join(json.dumps(entry) for entry in pi_entries(invoked=skill)),
        encoding="utf-8",
    )


def session_writer(
    captured: list[Path] | None = None,
    barrier: threading.Barrier | None = None,
) -> Callable[..., subprocess.CompletedProcess[str]]:
    """Return a subprocess handler writing a transcript that names the queried skill."""

    def _run(cmd: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        skill = cmd[cmd.index("-p") + 1].rsplit(" ", 1)[-1]
        session_dir = Path(cmd[cmd.index("--session-dir") + 1])
        session_dir.mkdir(parents=True, exist_ok=True)
        if captured is not None:
            captured.append(session_dir)
        write_pi_session(session_dir / f"session_{skill}.jsonl", skill)
        if barrier is not None:
            barrier.wait()
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="OK", stderr="")

    return _run


def probe_concurrently(runtime: PiRuntime, workdir: Path) -> dict[str, Any]:
    """Run one probe per resident skill on separate threads and collect outcomes."""
    outcomes: dict[str, Any] = {}

    def _probe(skill: str) -> None:
        outcomes[skill] = runtime.select(f"please use {skill}", workdir)

    threads = [threading.Thread(target=_probe, args=(skill,)) for skill in SLOT_SKILLS]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
    return outcomes


def test_concurrent_selects_do_not_cross_attribute_sessions(
    mock_subprocess: Callable[..., Any],
    tmp_path: Path,
) -> None:
    """Verify parallel probes read their own transcript rather than a sibling worker's."""
    workdir = tmp_path / "work"
    workdir.mkdir()

    rt = PiRuntime()
    rt._resident = SLOT_SKILLS

    # Both transcripts exist before either probe reads, as under real parallel execution.
    mock_subprocess(handler=session_writer(barrier=threading.Barrier(2, timeout=10)))

    outcomes = probe_concurrently(rt, workdir)

    assert set(outcomes) == set(SLOT_SKILLS)
    for skill in SLOT_SKILLS:
        assert outcomes[skill].error is None
        assert outcomes[skill].invoked_skill == skill


def test_concurrent_selects_use_distinct_session_directories(
    mock_subprocess: Callable[..., Any],
    tmp_path: Path,
) -> None:
    """Verify each worker thread receives its own session directory under the session root."""
    workdir = tmp_path / "work"
    workdir.mkdir()

    rt = PiRuntime()
    rt._resident = SLOT_SKILLS

    captured: list[Path] = []
    mock_subprocess(
        handler=session_writer(captured=captured, barrier=threading.Barrier(2, timeout=10)),
    )

    probe_concurrently(rt, workdir)

    root = workdir / ".reach_pi_sessions"
    assert len(captured) == 2
    assert len({str(path) for path in captured}) == 2
    assert all(path.is_relative_to(root) for path in captured)


def test_select_prefers_the_newest_session_transcript(
    mock_subprocess: Callable[..., Any],
    tmp_path: Path,
) -> None:
    """Verify select reads the most recently written transcript, not an arbitrary glob entry."""
    workdir = tmp_path / "work"
    workdir.mkdir()

    rt = PiRuntime()
    rt._resident = SLOT_SKILLS

    def mock_run(cmd: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        session_dir = Path(cmd[cmd.index("--session-dir") + 1])
        session_dir.mkdir(parents=True, exist_ok=True)
        newest = session_dir / "aaa_current.jsonl"
        stale = session_dir / "zzz_resumed.jsonl"
        write_pi_session(newest, "alpha")
        write_pi_session(stale, "beta")
        stale_time = time.time() - 3600
        os.utime(stale, (stale_time, stale_time))
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="OK", stderr="")

    mock_subprocess(handler=mock_run)

    outcome = rt.select("please use alpha", workdir)
    assert outcome.invoked_skill == "alpha"


def test_auto_clean_spares_transcripts_of_other_workers(
    mock_subprocess: Callable[..., Any],
    tmp_path: Path,
) -> None:
    """Verify auto_clean removes only the calling thread's session slot."""
    workdir = tmp_path / "work"
    workdir.mkdir()

    foreign_slot = workdir / ".reach_pi_sessions" / "slot_other_worker"
    foreign_slot.mkdir(parents=True)
    foreign_session = foreign_slot / "in_flight.jsonl"
    write_pi_session(foreign_session, "beta")

    rt = PiRuntime(RuntimeSettings(agent="pi", options={"auto_clean": True}))
    rt._resident = SLOT_SKILLS

    captured: list[Path] = []
    mock_subprocess(handler=session_writer(captured=captured))

    outcome = rt.select("please use alpha", workdir)

    assert outcome.invoked_skill == "alpha"
    assert not captured[0].exists()
    assert foreign_session.exists()


def test_pi_generator_command_and_env() -> None:
    """Verify PiGenerator includes provider and api-key args in command and syncs env."""
    from reach.runtime.pi import PiGenerator, PiOptions

    opts = PiOptions(model="gemini-3.8-flash", provider="google", api_key="secret-key")
    gen = PiGenerator(options=opts)
    cmd = gen.build_completion_command("test prompt")
    assert cmd[:3] == ["pi", "-p", "--no-session"]
    assert "--provider" in cmd
    assert cmd[cmd.index("--provider") + 1] == "google"
    assert "--api-key" in cmd
    assert cmd[cmd.index("--api-key") + 1] == "secret-key"

    env = gen.build_env()
    assert env["GEMINI_API_KEY"] == "secret-key"
    assert env["PI_TELEMETRY"] == "0"
    assert env["PI_SKIP_VERSION_CHECK"] == "1"


@pytest.mark.parametrize("effort", ["none", "off", "None", "OFF"])
def test_pi_generator_suppresses_disabled_thinking(effort: str) -> None:
    """Verify PiGenerator omits --thinking when reasoning effort is disabled."""
    gen = PiGenerator(options=PiOptions(effort=effort))
    cmd = gen.build_completion_command("test")
    assert "--thinking" not in cmd


def test_pi_generator_includes_valid_thinking() -> None:
    """Verify PiGenerator includes --thinking when valid effort or thinking is set."""
    gen = PiGenerator(options=PiOptions(effort="low"))
    cmd = gen.build_completion_command("test")
    assert "--thinking" in cmd
    assert cmd[cmd.index("--thinking") + 1] == "low"


def test_parse_session_entries_extracts_prompt_tokens() -> None:
    """Verify parse_session_entries extracts prompt_tokens and cost_usd via PiUsage."""
    from reach.runtime.pi import parse_session_entries

    entries = [
        {
            "type": "message",
            "message": {
                "role": "assistant",
                "model": "gemini-3.8-flash",
                "usage": {
                    "input": 8500,
                    "output": 12,
                    "cacheRead": 70,
                    "cacheWrite": 4,
                    "cost": {"total": 0.0065},
                },
                "content": [
                    {
                        "type": "toolCall",
                        "name": "read",
                        "arguments": {"path": ".pi/skills/alpha/SKILL.md"},
                    }
                ],
            },
        }
    ]
    summary = parse_session_entries(entries, resident=("alpha",))
    assert summary.invoked_skills == ("alpha",)
    assert summary.prompt_tokens == 8574
    assert summary.cost_usd == pytest.approx(0.0065)

    from reach.runtime.pi import PiUsage

    assert PiUsage(input=10, cache_read=5, cache_write=2).total_prompt_tokens == 17


def test_pi_select_records_duration_ms(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify PiRuntime.select populates wall-clock duration_ms on outcomes."""
    import subprocess

    from reach.runtime.pi import PiRuntime

    def _fake_probe(
        *args: object, **kwargs: object
    ) -> tuple[subprocess.CompletedProcess[str], None]:
        return subprocess.CompletedProcess(["pi"], returncode=1, stdout="", stderr="boom"), None

    monkeypatch.setattr("reach.runtime.pi.run_subprocess_probe", _fake_probe)
    rt = PiRuntime(RuntimeSettings(agent="pi"))
    outcome = rt.select("test query", workdir=tmp_path)
    assert outcome.duration_ms is not None
    assert outcome.duration_ms >= 1
