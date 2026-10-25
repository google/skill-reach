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

import asyncio
import json
import subprocess
import threading
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Never, Self

import pytest

from reach.catalog import build_catalogs, load_skills
from reach.config import DEFAULT_GEMINI_MODEL, RuntimeSettings
from reach.models import Catalog, CatalogMode, Skill
from reach.runtime import AgentRuntime, build_runtime, known_agents, runtime_class
from reach.runtime._fs import WorkspaceSlot

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

#: Default join and synchronization timeout in seconds for concurrent worker tests.
WORKER_TIMEOUT: float = 10.0


def slot_agents() -> tuple[str, ...]:
    """Return known agent names whose runtime class defines an isolation_dir_name."""
    return tuple(
        agent
        for agent in known_agents()
        if getattr(runtime_class(agent), "isolation_dir_name", None)
    )


def expected_slot(runtime: AgentRuntime[Any], workdir: Path) -> Path:
    """Return the calling thread's expected WorkspaceSlot directory for runtime and workdir."""
    iso_name = runtime.isolation_dir_name
    assert iso_name is not None
    return WorkspaceSlot.for_subdir(workdir, iso_name).slot


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


def build_agent(agent: str, tmp_path: Path, **extra_options: object) -> AgentRuntime[Any]:
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
def workdir(tmp_path: Path) -> Path:
    """Provide an initialized workspace directory inside tmp_path."""
    path = tmp_path / "work"
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture
def clean_api_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear ambient model provider API keys and GCP flags from test environment."""
    for var in (
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "GOOGLE_CLOUD_PROJECT",
        "GOOGLE_CLOUD_QUOTA_PROJECT",
        "AGY_ADC_AUTH",
        "GOOGLE_GENAI_USE_ENTERPRISE",
        "GOOGLE_GENAI_USE_VERTEXAI",
    ):
        monkeypatch.delenv(var, raising=False)


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
    if isinstance(mock, pytest.MonkeyPatch):
        mock.setattr(
            subprocess,
            "run",
            lambda *a, **_kw: subprocess.CompletedProcess(
                args=a,
                returncode=returncode,
                stdout="\n".join(lines),
                stderr="",
            ),
        )
    else:
        mock(lines=lines, returncode=returncode)


def install_one(runtime: AgentRuntime[Any], skill_repo: Path, workdir: Path) -> str:
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
            stop_reason = ag_types.StopReason.UNSPECIFIED if HAS_ANTIGRAVITY else "UNSPECIFIED"
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
    """Simulate a single SDK conversation step with HTTP status, error, and usage details."""

    def __init__(
        self,
        *,
        status: Any = "STATE_ERROR",
        http_code: int | str = 0,
        error: str = "",
        step_index: int | None = None,
        usage_metadata: object = None,
        tool_calls: Sequence[object] = (),
    ) -> None:
        """Initialize mock conversation step with status, HTTP code, error, and usage metadata."""
        self.status = status
        self.http_code = http_code
        self.error = error
        self.step_index = step_index
        self.usage_metadata = usage_metadata
        self.tool_calls = list(tool_calls)


class FakeSdkConversation:
    """Mock SDK Conversation exposing step history and usage for inspection tests."""

    def __init__(
        self,
        history: Sequence[object] = (),
        *,
        total_usage: object = None,
        connection: object = None,
    ) -> None:
        """Initialize mock conversation with canned step history, total usage, and connection."""
        self.history: list[object] = list(history)
        self.total_usage = total_usage
        self.connection = connection


class FakeSdkAgent:
    """Mock SDK Agent context manager tracking configurations and queries."""

    response: Any = None

    def __init__(
        self,
        config: Any,
        *,
        history: Sequence[object] = (),
        total_usage: object = None,
        connection: object = None,
    ) -> None:
        """Initialize mock agent with configuration, step history, and usage metadata."""
        self.config = config
        self.sent: str | None = None
        self.conversation = FakeSdkConversation(
            history=history,
            total_usage=total_usage,
            connection=connection,
        )

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
    total_usage: object = None,
    connection: object = None,
) -> list[FakeSdkAgent]:
    """Patch SDK Agent class with mock agent answering scripted response."""
    instances: list[FakeSdkAgent] = []

    def factory(config: Any) -> FakeSdkAgent:
        agent = FakeSdkAgent(
            config,
            history=history,
            total_usage=total_usage,
            connection=connection,
        )
        agent.response = response
        instances.append(agent)
        return agent

    monkeypatch.setattr("reach.runtime.antigravity_sdk.Agent", factory)
    return instances


class FakeSdkPopen:
    """Mock subprocess.Popen tracking termination and wait calls during SDK startup."""

    def __init__(self) -> None:
        """Initialize mock process in running state."""
        self.killed = False
        self.terminated = False
        self.waited = False

    def poll(self) -> int | None:
        """Return 0 once terminated or killed, otherwise None."""
        return 0 if (self.killed or self.terminated) else None

    def terminate(self) -> None:
        """Mark the mock process as terminated."""
        self.terminated = True

    def kill(self) -> None:
        """Mark the mock process as killed."""
        self.killed = True

    def wait(self, timeout: float | None = None) -> int:
        """Record wait invocation and return exit code 0."""
        del timeout
        self.waited = True
        return 0


def patch_cancelled_sdk_startup(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[list[FakeSdkPopen], list[bool]]:
    """Patch Antigravity SDK Agent and LocalAgentConfig to raise CancelledError during startup."""
    from reach.runtime.antigravity_sdk import AntigravitySdkOptions

    spawned_processes: list[FakeSdkPopen] = []
    exit_stacks_closed: list[bool] = []

    class _FakeStrategy:
        async def _connect_websocket(
            self,
            port: int,
            api_key: str,
            process: FakeSdkPopen,
        ) -> Never:
            del port, api_key, process
            raise asyncio.CancelledError

        async def __aenter__(self) -> Never:
            proc = FakeSdkPopen()
            spawned_processes.append(proc)
            await self._connect_websocket(12345, "key", proc)

        async def __aexit__(self, *args: object) -> None:
            del args

    class _FakeExitStack:
        async def enter_async_context(self, cm: _FakeStrategy) -> Never:
            return await cm.__aenter__()

        async def aclose(self) -> None:
            exit_stacks_closed.append(True)

    class _CancelledStartupAgent:
        def __init__(self, config: Any) -> None:
            self.config = config
            self._config = config
            self._exit_stack = _FakeExitStack()
            self._strategy: Any = None

        async def __aenter__(self) -> Never:
            create_fn = getattr(self._config, "create_strategy", None)
            strategy = create_fn() if callable(create_fn) else _FakeStrategy()
            self._strategy = strategy
            await self._exit_stack.enter_async_context(strategy)

        async def __aexit__(self, *args: object) -> None:
            del args
            await self._exit_stack.aclose()

    class _FakeLocalAgentConfig:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs
            self.app_data_dir = kwargs.get("app_data_dir")
            self.save_dir = kwargs.get("save_dir")

        def create_strategy(self) -> _FakeStrategy:
            return _FakeStrategy()

    monkeypatch.setattr("reach.runtime.antigravity_sdk.Agent", _CancelledStartupAgent)
    monkeypatch.setattr("reach.runtime.antigravity_sdk.LocalAgentConfig", _FakeLocalAgentConfig)
    monkeypatch.setattr(
        AntigravitySdkOptions,
        "base_config_kwargs",
        lambda _self, _model_spec, _env_map: {},
    )
    return spawned_processes, exit_stacks_closed


@pytest.fixture
def cancelled_sdk_startup(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[list[FakeSdkPopen], list[bool]]:
    """Provide patched SDK startup classes that raise CancelledError on handshake."""
    return patch_cancelled_sdk_startup(monkeypatch)


def run_concurrent_workers[T](
    count: int,
    fn: Callable[[int], T],
    *,
    timeout: float = WORKER_TIMEOUT,
    on_error: Callable[[], None] | None = None,
) -> list[T]:
    """Run ``count`` worker threads, join with ``timeout``, and re-raise any worker exception."""
    results: dict[int, T] = {}
    errors: list[BaseException] = []

    def _target(idx: int) -> None:
        try:
            results[idx] = fn(idx)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)
            if on_error is not None:
                on_error()

    threads = [threading.Thread(target=_target, args=(idx,)) for idx in range(count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=timeout)
        assert not thread.is_alive()

    if errors:
        raise errors[0]
    return [results[idx] for idx in range(count)]


def run_staggered_two_worker_probe(
    *,
    setup_slot: Callable[[], Path],
    cleanup_slot: Callable[[Path], None],
    is_intact: Callable[[Path], bool],
    is_cleaned: Callable[[Path], bool] = lambda slot: not slot.exists(),
    timeout: float = WORKER_TIMEOUT,
) -> tuple[Path, Path]:
    """Run two concurrent worker threads where worker B cleans up while worker A is still active."""
    ready_barrier = threading.Barrier(2, timeout=timeout)
    b_cleaned_event = threading.Event()
    checks: dict[str, bool] = {}

    barrier_passed = False

    def _abort_sync() -> None:
        if not barrier_passed:
            ready_barrier.abort()
        b_cleaned_event.set()

    def _worker(idx: int) -> Path:
        nonlocal barrier_passed
        clean_first = idx == 1
        slot = setup_slot()
        ready_barrier.wait()
        barrier_passed = True
        if clean_first:
            try:
                cleanup_slot(slot)
            finally:
                checks["b_cleaned"] = is_cleaned(slot)
                b_cleaned_event.set()
        else:
            b_cleaned_event.wait(timeout=timeout)
            checks["a_survived_b_cleanup"] = is_intact(slot)
            cleanup_slot(slot)
            checks["a_cleaned"] = is_cleaned(slot)
        return slot

    slot_a, slot_b = run_concurrent_workers(
        2,
        _worker,
        timeout=timeout,
        on_error=_abort_sync,
    )
    assert slot_a != slot_b
    assert checks.get("b_cleaned") is True
    assert checks.get("a_survived_b_cleanup") is True
    assert checks.get("a_cleaned") is True
    return slot_a, slot_b
