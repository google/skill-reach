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

"""Provide shared test fixtures, mock objects, and event stream generators for runtime tests."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self, cast

import pytest

from reach.catalog import build_catalogs, load_skills
from reach.config import DEFAULT_GEMINI_MODEL, RuntimeSettings
from reach.models import Catalog, CatalogMode, Skill
from reach.runtime import AgentRuntime, build_runtime

if TYPE_CHECKING:
    from collections.abc import Sequence

    from google.antigravity import types as ag_types
else:
    try:
        from google.antigravity import types as ag_types
    except ImportError:
        ag_types = None


#: True when google.antigravity SDK is installed and available.
HAS_ANTIGRAVITY: bool = ag_types is not None


def agent_dependencies_met(agent: str) -> bool:
    """Check whether optional dependencies for a given agent runtime are available."""
    if agent == "antigravity-sdk":
        return HAS_ANTIGRAVITY
    return True


@pytest.fixture(autouse=True)
def require_agent_dependencies(request: pytest.FixtureRequest) -> None:
    """Automatically skip tests when an agent's optional dependencies are missing."""
    callspec = getattr(request.node, "callspec", None)
    if callspec is not None:
        agent = callspec.params.get("agent")
        if isinstance(agent, str) and not agent_dependencies_met(agent):
            pytest.skip(f"Optional dependencies for agent {agent!r} are not installed")


#: Minimal required options for initializing each agent type in test suites.
MINIMAL_OPTIONS: dict[str, dict[str, object]] = {
    "antigravity-sdk": {"model": "test-model"},
    "antigravity-cli": {
        "model": "test-model",
        "home_dir": str(Path.home() / ".reach" / "test-antigravity-cli-home"),
    },
    "fake": {"materialize": True},
}


def build_agent(agent: str, tmp_path: Path, **extra_options: object) -> AgentRuntime:
    """Instantiate a runtime instance configured for test execution."""
    if not agent_dependencies_met(agent):
        pytest.skip(f"Optional dependencies for agent {agent!r} are not installed")
    opts: dict[str, object] = dict(MINIMAL_OPTIONS.get(agent, {}))
    if agent == "antigravity-cli":
        opts["home_dir"] = tmp_path / f"home_{agent}"
    opts.update(extra_options)
    return build_runtime(RuntimeSettings(agent=agent, options=opts))


@pytest.fixture
def catalog() -> Catalog:
    """Provide a sample two-skill catalog."""
    return Catalog(id="c", mode=CatalogMode.ALL, skills=("a", "b"))


@pytest.fixture
def skills(tmp_path: Path) -> list[Skill]:
    """Provide sample Skill objects corresponding to catalog members."""
    return [Skill(name=n, description=f"does {n}", path=tmp_path / n) for n in ("a", "b")]


@pytest.fixture
def home_dir(tmp_path: Path) -> Path:
    """Provide an isolated, agent-owned home directory."""
    return tmp_path / "agy-home"


@pytest.fixture
def clean_api_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear ambient model provider API keys from test environment."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)


def read_isolated_settings(home_dir: Path) -> dict[str, Any]:
    """Read and parse the JSON settings file from an isolated home directory."""
    path = home_dir / ".gemini" / "antigravity-cli" / "settings.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def canned(
    mock: Callable[..., Any] | pytest.MonkeyPatch,
    lines: list[str],
    returncode: int = 0,
) -> None:
    """Mock subprocess.run with canned transcript output lines."""
    if callable(mock) and not hasattr(mock, "setattr"):
        mock(lines=lines, returncode=returncode)
    else:
        cast("pytest.MonkeyPatch", mock).setattr(
            subprocess,
            "run",
            lambda *a, **_kw: subprocess.CompletedProcess(
                args=a,
                returncode=returncode,
                stdout="\n".join(lines),
                stderr="",
            ),
        )


def install_one(runtime: AgentRuntime, skill_repo: Path, workdir: Path) -> str:
    """Install single skill singleton catalog and return resident skill name."""
    all_skills = load_skills(skill_repo)
    cat = build_catalogs(all_skills, CatalogMode.SINGLETON)[0]
    runtime.install(cat, all_skills, workdir)
    return cat.skills[0]


def agy_stream(
    *,
    invoked: str | None = None,
    reasoning: str | None = None,
    include_structured: bool = True,
    tools: Sequence[tuple[str, str | None]] = (),
    status: str | None = "SUCCESS",
    duration_seconds: float | None = 1.234,
    model: str | None = DEFAULT_GEMINI_MODEL,
    error: str | None = None,
    prompt_tokens: int | None = None,
    usage: dict[str, Any] | None = None,
) -> list[str]:
    """Render stream-json transcript lines matching antigravity CLI output format."""
    events: list[dict[str, Any]] = []
    init: dict[str, Any] = {}
    if model is not None:
        init["model"] = model
    events.append({"event": "init", "init": init})
    for name, path in tools:
        info: dict[str, Any] = {"parameters": {}}
        if path is not None:
            info["parameters"]["AbsolutePath"] = path
        events.append(
            {
                "event": "step_update",
                "step_update": {
                    "step_type": "tool",
                    "tool_name": name,
                    "tool_info": info,
                },
            },
        )
    if status is not None:
        result: dict[str, Any] = {"status": status}
        if error is not None:
            result["error"] = error
        if duration_seconds is not None:
            result["duration_seconds"] = duration_seconds
        if prompt_tokens is not None:
            result["usage"] = {"input_tokens": prompt_tokens}
        elif usage is not None:
            result["usage"] = usage
        if include_structured:
            structured: dict[str, Any] = {"selected_skill": invoked}
            if reasoning is not None:
                structured["reasoning"] = reasoning
            result["structured_output"] = structured
        events.append({"event": "result", "result": result})
    return [json.dumps(e) for e in events]


def _reasoning_blocks(items: Sequence[str | tuple[str, str]]) -> list[dict[str, str]]:
    """Convert string or tuple reasoning items into structured block dicts."""
    return [
        {"type": item[0], item[0]: item[1]}
        if isinstance(item, tuple)
        else {"type": "thinking", "thinking": item}
        for item in items
    ]


def _build_assistant_events(
    *,
    assistant_turns: Sequence[Sequence[dict[str, Any]] | dict[str, Any]] | None,
    parallel_skills: Sequence[str],
    invoked_skills: Sequence[str],
    turn_texts: Sequence[str],
    invoked: str | None,
    reasoning: Sequence[str | tuple[str, str]],
    turn_thoughts: Sequence[str],
    include_init: bool,
    include_result: bool,
) -> list[dict[str, Any]]:
    """Generate assistant turn events for Claude Code transcripts."""
    if assistant_turns is not None:
        return [
            turn
            if isinstance(turn, dict) and turn.get("type") == "assistant"
            else {"type": "assistant", "message": {"content": list(turn)}}
            for turn in assistant_turns
        ]

    if turn_texts:
        return [
            {"type": "assistant", "message": {"content": [{"type": "text", "text": text}]}}
            for text in turn_texts
        ]

    if invoked_skills:
        return [
            {
                "type": "assistant",
                "message": {
                    "content": [
                        *(
                            [{"type": "thinking", "thinking": turn_thoughts[i]}]
                            if turn_thoughts and i < len(turn_thoughts)
                            else _reasoning_blocks(reasoning)
                            if i == 0
                            else []
                        ),
                        {
                            "type": "tool_use",
                            "name": "Skill",
                            "input": {"skill": skill, "args": "..."},
                        },
                    ],
                },
            }
            for i, skill in enumerate(invoked_skills)
        ]

    if parallel_skills:
        content = [
            *_reasoning_blocks(reasoning),
            *[
                {"type": "tool_use", "name": "Skill", "input": {"skill": s, "args": "..."}}
                for s in parallel_skills
            ],
        ]
    elif invoked is not None:
        content = [
            *_reasoning_blocks(reasoning),
            {"type": "tool_use", "name": "Skill", "input": {"skill": invoked, "args": "..."}},
        ]
    elif not include_init and not include_result and not reasoning:
        return []
    else:
        content = [*_reasoning_blocks(reasoning), {"type": "text", "text": "No skill needed."}]

    return [{"type": "assistant", "message": {"content": content}}]


def claude_stream(
    *,
    catalog: Sequence[str] | None = None,
    invoked: str | None = None,
    invoked_skills: Sequence[str] = (),
    parallel_skills: Sequence[str] = (),
    reasoning: Sequence[str | tuple[str, str]] = (),
    turn_thoughts: Sequence[str] = (),
    turn_texts: Sequence[str] = (),
    tools: Sequence[str] | None = None,
    model: str | None = "claude-opus-5",
    cost: float | None = 0.21,
    duration_ms: int | None = 1911,
    retries: int = 0,
    subtype: str = "success",
    include_init: bool = True,
    include_result: bool = True,
    assistant_turns: Sequence[Sequence[dict[str, Any]] | dict[str, Any]] | None = None,
) -> list[str]:
    """Format stream-json transcript lines matching Claude Code runtime output format."""
    events: list[dict[str, Any]] = []

    if include_init:
        init: dict[str, Any] = {
            "type": "system",
            "subtype": "init",
            "skills": list(catalog) if catalog is not None else [],
            "tools": ["Skill"] if tools is None else list(tools),
        }
        if model is not None:
            init["model"] = model
        events.append(init)

    if retries > 0:
        events.extend(
            {
                "type": "system",
                "subtype": "api_retry",
                "attempt": attempt,
                "error_status": 429,
            }
            for attempt in range(1, retries + 1)
        )

    events.extend(
        _build_assistant_events(
            assistant_turns=assistant_turns,
            parallel_skills=parallel_skills,
            invoked_skills=invoked_skills,
            turn_texts=turn_texts,
            invoked=invoked,
            reasoning=reasoning,
            turn_thoughts=turn_thoughts,
            include_init=include_init,
            include_result=include_result,
        ),
    )

    if include_result:
        events.append(
            {
                "type": "result",
                "subtype": subtype,
                "total_cost_usd": cost,
                "duration_ms": duration_ms,
            },
        )

    return [json.dumps(e) for e in events]


@pytest.fixture
def make_stream() -> Callable[..., list[str]]:
    """Return factory function generating mock Claude Code stream lines."""
    return claude_stream


def goose_payload(
    *,
    invoked: str | None = None,
    invoked_skills: Sequence[str] = (),
    tools: Sequence[tuple[str, dict[str, Any]]] = (),
    cost_usd: float | None = 0.0042,
    total_tokens: int = 1500,
    status: str = "completed",
    user_query: str = "Calculate pizza for 10",
) -> dict[str, Any]:
    """Construct a Goose agent JSON response dictionary."""
    effective_skills = list(invoked_skills)
    if invoked is not None and not effective_skills:
        effective_skills.append(invoked)

    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": [{"type": "text", "text": user_query}],
        },
    ]
    if effective_skills:
        for i, skill in enumerate(effective_skills):
            messages.append(
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "toolRequest",
                            "id": f"call_{i:03d}",
                            "toolCall": {
                                "status": "success",
                                "value": {
                                    "name": "load_skill",
                                    "arguments": {"name": skill},
                                },
                            },
                        },
                    ],
                },
            )
    if tools:
        tool_content = [
            {
                "type": "toolRequest",
                "id": f"tool_call_{i:03d}",
                "toolCall": {
                    "status": "success",
                    "value": {
                        "name": tool_name,
                        "arguments": args,
                    },
                },
            }
            for i, (tool_name, args) in enumerate(tools)
        ]
        messages.append({"role": "assistant", "content": tool_content})
    elif not effective_skills:
        messages.append({"role": "assistant", "content": []})

    return {
        "messages": messages,
        "metadata": {
            "total_tokens": total_tokens,
            "cost_usd": cost_usd,
            "status": status,
        },
    }


def pi_entries(
    *,
    invoked: str | None = None,
    invoked_skills: Sequence[str] = (),
    tools: Sequence[tuple[str, str]] = (),
    cost_usd: float | None = 0.0125,
    model: str = "gemini-3.5-flash",
    total_tokens: int = 450,
    user_query: str = "Deploy to cloud",
    session_id: str = "test-session-1",
    include_session: bool = True,
) -> list[dict[str, Any]]:
    """Construct Pi session event entries matching agent transcript format."""
    entries: list[dict[str, Any]] = []
    if include_session:
        entries.append(
            {
                "type": "session",
                "version": 3,
                "id": session_id,
                "timestamp": "2026-08-29T20:00:00.000Z",
                "cwd": "/workspace",
            },
        )
    if user_query:
        entries.append(
            {
                "type": "message",
                "message": {
                    "role": "user",
                    "content": [{"type": "text", "text": user_query}],
                },
            },
        )

    assistant_content: list[dict[str, Any]] = []
    effective_skills = list(invoked_skills)
    if invoked is not None and not effective_skills:
        effective_skills.append(invoked)

    for i, skill in enumerate(effective_skills):
        assistant_content.append(
            {
                "type": "toolCall",
                "id": f"call_{i:02d}",
                "name": "read",
                "arguments": {"path": f"/workspace/.pi/skills/{skill}/SKILL.md"},
            },
        )

    for i, (tool_name, tool_path) in enumerate(tools):
        assistant_content.append(
            {
                "type": "toolCall",
                "id": f"tool_call_{i:02d}",
                "name": tool_name,
                "arguments": {"path": tool_path},
            },
        )

    usage: dict[str, Any] = {"totalTokens": total_tokens}
    if cost_usd is not None:
        usage["cost"] = {"total": cost_usd}

    entries.append(
        {
            "type": "message",
            "message": {
                "role": "assistant",
                "content": assistant_content,
                "model": model,
                "usage": usage,
            },
        },
    )
    return entries


class FakeSdkToolCallStream:
    """Async iterator over canned tool call objects."""

    def __init__(self, calls: Sequence[object]) -> None:
        """Initialize the stream with canned tool calls."""
        self._calls = list(calls)

    def __aiter__(self) -> Self:
        """Return self as the async iterator."""
        return self

    async def __anext__(self) -> object:
        """Yield the next canned tool call."""
        if not self._calls:
            raise StopAsyncIteration
        return self._calls.pop(0)


class FakeSdkResponse:
    """Mock agent turn response with configurable structured output and tool calls."""

    def __init__(
        self,
        *,
        structured: object = None,
        tool_calls: Sequence[object] = (),
        stop_reason: Any = None,
        text: str = "",
    ) -> None:
        """Initialize mock response with structured output and tool calls."""
        if stop_reason is None:
            stop_reason = ag_types.StopReason.UNSPECIFIED if ag_types is not None else "UNSPECIFIED"
        self._structured = structured
        self._tool_calls = tool_calls
        self.stop_reason = stop_reason
        self._text = text

    async def structured_output(self) -> object:
        """Return scripted structured output payload."""
        return self._structured

    @property
    def tool_calls(self) -> FakeSdkToolCallStream:
        """Return async iterator stream over canned tool calls."""
        return FakeSdkToolCallStream(self._tool_calls)

    async def text(self) -> str:
        """Return canned textual response string."""
        return self._text


class FakeSdkStep:
    """Simulate a single SDK conversation step with HTTP status and error details."""

    def __init__(
        self,
        *,
        status: Any = "STATE_ERROR",
        http_code: int | str = 0,
        error: str = "",
    ) -> None:
        """Initialize mock conversation step with status, HTTP code, and error message."""
        self.status = status
        self.http_code = http_code
        self.error = error


class FakeSdkConversation:
    """Mock SDK Conversation exposing step history for error inspection tests."""

    def __init__(self, history: Sequence[object] = ()) -> None:
        """Initialize mock conversation with canned step history."""
        self.history: list[object] = list(history)


class FakeSdkAgent:
    """Mock SDK Agent context manager tracking configurations and queries."""

    response: Any = None

    def __init__(self, config: Any, *, history: Sequence[object] = ()) -> None:
        """Initialize mock agent with configuration and optional step history."""
        self.config = config
        self.sent: str | None = None
        self.conversation = FakeSdkConversation(history=history)

    async def __aenter__(self) -> Self:
        """Enter the asynchronous agent context manager."""
        return self

    async def __aexit__(self, *exc_info: object) -> bool:
        """Exit the asynchronous agent context manager."""
        return False

    async def chat(self, query_text: str) -> Any:
        """Record the query text and return canned response."""
        self.sent = query_text
        return self.response


def patch_sdk_agent(
    monkeypatch: pytest.MonkeyPatch,
    response: Any,
    *,
    history: Sequence[object] = (),
) -> list[FakeSdkAgent]:
    """Patch SDK Agent class with mock agent answering scripted response."""
    instances: list[FakeSdkAgent] = []

    def factory(config: Any) -> FakeSdkAgent:
        agent = FakeSdkAgent(config, history=history)
        agent.response = response
        instances.append(agent)
        return agent

    monkeypatch.setattr("reach.runtime.antigravity_sdk.Agent", factory)
    return instances
