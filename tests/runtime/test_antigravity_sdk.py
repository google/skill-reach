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

"""Verify antigravity-sdk agent isolation, selection schema, and execution."""

from __future__ import annotations

import asyncio
import importlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Never

import pytest

from reach.config import RuntimeSettings, agent_default_model
from reach.runtime import AntigravityRuntime
from reach.runtime.antigravity_sdk import (
    _HAS_ANTIGRAVITY,
    AntigravitySdkGenerator,
    AntigravitySdkOptions,
    AntigravitySdkRuntime,
    _build_model_spec,
    _tool_name,
)

from .conftest import (
    FakeSdkAgent as _FakeAgent,
)
from .conftest import (
    FakeSdkResponse as _FakeResponse,
)
from .conftest import (
    FakeSdkStep as _FakeStep,
)
from .conftest import (
    patch_sdk_agent as _fake_agent,
)

if TYPE_CHECKING:
    from pathlib import Path

    from google.antigravity import types as ag_types
else:
    try:
        from google.antigravity import types as ag_types
    except ImportError:
        ag_types = None


@pytest.fixture(autouse=True)
def _require_antigravity(request: pytest.FixtureRequest) -> None:
    """Skip test if google.antigravity is not installed and test requires it."""
    exempt = (
        "test_model_has_default",
        "test_antigravity_sdk_options_effort",
        "test_antigravity_sdk_options_defaults",
        "test_tool_name_passes_a_custom_tool_name_through",
        "missing_dependency",
        "uninstalled",
    )
    if not _HAS_ANTIGRAVITY and not any(ex in request.node.name for ex in exempt):
        pytest.skip("google-antigravity is not installed")


@pytest.fixture
def runtime() -> AntigravitySdkRuntime:
    """Provide an AntigravitySdkRuntime instance configured with test-model."""
    return AntigravitySdkRuntime(options=AntigravitySdkOptions(model="test-model"))


@pytest.fixture
def generator() -> AntigravitySdkGenerator:
    """Provide an AntigravitySdkGenerator instance configured with test-model."""
    return AntigravitySdkGenerator(options=AntigravitySdkOptions(model="test-model"))


def test_model_has_default() -> None:
    """Verify model parameter defaults to configured agent default model."""
    default = agent_default_model("antigravity-sdk")
    assert default is not None
    assert AntigravitySdkOptions().model == default


def test_the_agent_reports_the_configured_model(runtime: AntigravitySdkRuntime) -> None:
    """Verify runtime.model returns the configured model identifier."""
    assert runtime.model == "test-model"


def test_tool_name_reads_the_plain_value_not_the_enum_repr() -> None:
    """Verify _tool_name extracts string value from BuiltinTools enum members."""
    assert _tool_name(ag_types.BuiltinTools.FINISH) == "finish"


def test_tool_name_passes_a_custom_tool_name_through() -> None:
    """Verify _tool_name returns plain string tool names unmodified."""
    assert _tool_name("my_mcp_tool") == "my_mcp_tool"


def test_select_config_omits_response_schema_by_default_and_respects_explicit_json_schema(
    runtime: AntigravitySdkRuntime,
    tmp_path: Path,
) -> None:
    """Verify _select_config omits response_schema by default and applies explicit json_schema."""
    runtime._resident = ("gke-basics",)
    config = runtime._select_config(tmp_path / "work")
    assert config.response_schema is None

    explicit_schema = runtime.selection_json_schema(runtime._resident)
    rt_explicit = AntigravitySdkRuntime(
        options=AntigravitySdkOptions(model="test-model", json_schema=explicit_schema),
    )
    rt_explicit._resident = ("gke-basics",)
    cfg_explicit = rt_explicit._select_config(tmp_path / "work")
    assert isinstance(cfg_explicit.response_schema, str)
    schema = json.loads(cfg_explicit.response_schema)
    selected = schema["properties"]["selected_skill"]
    assert {"const": "gke-basics", "type": "string"} in selected["anyOf"]


def test_select_config_enables_multi_turn_tools_across_all_turn_budgets(
    runtime: AntigravitySdkRuntime,
    tmp_path: Path,
) -> None:
    """Verify _select_config enables MULTI_TURN_SELECTION_TOOLS for max_turns=3 and max_turns=1."""
    from reach.runtime.antigravity_sdk import MULTI_TURN_SELECTION_TOOLS

    runtime._resident = ("a", "b")
    config = runtime._select_config(tmp_path / "work")
    assert config.capabilities.enabled_tools == list(MULTI_TURN_SELECTION_TOOLS)
    assert config.capabilities.enable_subagents is False
    assert config.budget_config is not None
    assert config.budget_config.max_model_calls == 3

    single_turn_rt = AntigravitySdkRuntime(
        options=AntigravitySdkOptions(model="test-model", max_turns=1),
    )
    single_turn_rt._resident = ("a", "b")
    single_cfg = single_turn_rt._select_config(tmp_path / "work")
    assert single_cfg.capabilities.enabled_tools == list(MULTI_TURN_SELECTION_TOOLS)
    assert single_cfg.budget_config is not None
    assert single_cfg.budget_config.max_model_calls == 1


def test_selection_schema_names_the_resident_catalog(
    runtime: AntigravitySdkRuntime,
) -> None:
    """Verify selection_json_schema helper contains enum of resident skill names."""
    runtime._resident = ("gke-basics", "gcs-lifecycle-rules")
    schema = json.loads(runtime.selection_json_schema(runtime._resident))
    selected = schema["properties"]["selected_skill"]
    enum_values = next(branch["enum"] for branch in selected["anyOf"] if "enum" in branch)
    assert sorted(enum_values) == ["gcs-lifecycle-rules", "gke-basics"]


def test_antigravity_sdk_options_effort() -> None:
    """Verify AntigravitySdkOptions accepts model and optional effort."""
    opts_with_effort = AntigravitySdkOptions(model="gemini-3.8-flash", effort="medium")
    assert opts_with_effort.model == "gemini-3.8-flash"
    assert opts_with_effort.effort == "medium"

    opts_no_effort = AntigravitySdkOptions(model="gemini-3.8-flash")
    assert opts_no_effort.model == "gemini-3.8-flash"
    assert opts_no_effort.effort is None


def test_select_config_sets_thinking_config(tmp_path: Path) -> None:
    """Verify _select_config sets thinking_config when effort is specified."""
    runtime = AntigravitySdkRuntime(
        options=AntigravitySdkOptions(model="gemini-3.8-flash", effort="high"),
    )
    runtime._resident = ("skill-a",)
    config = runtime._select_config(tmp_path)
    assert isinstance(config.model, ag_types.ModelTarget)
    assert config.model.name == "gemini-3.8-flash"
    assert isinstance(config.model.endpoint, ag_types.GeminiAPIEndpoint)
    assert config.model.endpoint.options is not None
    assert config.model.endpoint.options.thinking_level == ag_types.ThinkingLevel.HIGH

    runtime_default = AntigravitySdkRuntime(
        options=AntigravitySdkOptions(model="gemini-3.8-flash"),
    )
    runtime_default._resident = ("skill-a",)
    config_default = runtime_default._select_config(tmp_path)
    assert isinstance(config_default.model, ag_types.ModelTarget)
    assert isinstance(config_default.model.endpoint, ag_types.GeminiAPIEndpoint)
    assert isinstance(config_default.model.endpoint.options, ag_types.GeminiModelOptions)
    assert config_default.model.endpoint.options.thinking_level == ag_types.ThinkingLevel.LOW

    runtime_no_effort = AntigravitySdkRuntime(
        options=AntigravitySdkOptions(model="custom-model"),
    )
    runtime_no_effort._resident = ("skill-a",)
    config_no_effort = runtime_no_effort._select_config(tmp_path)
    assert config_no_effort.model == "custom-model"


def test_select_config_points_at_the_installed_skills_directory(
    runtime: AntigravitySdkRuntime,
    tmp_path: Path,
) -> None:
    """Verify skills_paths in select config points to workspace skills directory."""
    workdir = tmp_path / "work"
    runtime._resident = ("gke-basics",)
    config = runtime._select_config(workdir)
    assert config.skills_paths == [str(runtime.skills_dir(workdir))]


def test_select_config_includes_symlink_targets_when_use_symlinks_true(
    tmp_path: Path,
) -> None:
    """Verify skills_paths includes resolved symlink target directories when enabled."""
    runtime = AntigravitySdkRuntime(options=AntigravitySdkOptions(use_symlinks=True))
    workdir = tmp_path / "work"
    skills_dir = runtime.skills_dir(workdir)
    skills_dir.mkdir(parents=True, exist_ok=True)

    external_source = tmp_path / "external_skills" / "custom-skill"
    external_source.mkdir(parents=True, exist_ok=True)
    symlink_dst = skills_dir / "custom-skill"
    symlink_dst.symlink_to(external_source, target_is_directory=True)

    config = runtime._select_config(workdir)
    assert str(skills_dir) in config.skills_paths
    assert str(external_source) in config.skills_paths


def test_select_reports_the_structured_selection(
    monkeypatch: pytest.MonkeyPatch,
    runtime: AntigravitySdkRuntime,
    tmp_path: Path,
) -> None:
    """Verify select parses selected skill correctly from structured response."""
    runtime._resident = ("gke-basics",)
    _fake_agent(monkeypatch, _FakeResponse(structured={"selected_skill": "gke-basics"}))
    outcome = runtime.select("how do I set up a cluster?", tmp_path / "work")
    assert outcome.invoked_skill == "gke-basics"
    assert outcome.invoked_skills == ("gke-basics",)
    assert outcome.observed_catalog == ("gke-basics",)
    assert outcome.error is None
    assert outcome.cost_usd is None


def test_select_reports_the_structured_selection_from_pydantic_model(
    monkeypatch: pytest.MonkeyPatch,
    runtime: AntigravitySdkRuntime,
    tmp_path: Path,
) -> None:
    """Verify select parses selected skill from a Pydantic model instance."""
    runtime._resident = ("gke-basics",)
    schema_cls = AntigravityRuntime.selection_schema(("gke-basics",))
    model_obj = schema_cls.model_validate(
        {"selected_skill": "gke-basics", "reasoning": "Fits target"},
    )
    _fake_agent(monkeypatch, _FakeResponse(structured=model_obj))
    outcome = runtime.select("how do I set up a cluster?", tmp_path / "work")
    assert outcome.invoked_skill == "gke-basics"
    assert outcome.invoked_skills == ("gke-basics",)
    assert outcome.reasoning == ("Fits target",)
    assert outcome.observed_catalog == ("gke-basics",)
    assert outcome.error is None


def test_select_populates_reasoning_in_outcome(
    monkeypatch: pytest.MonkeyPatch,
    runtime: AntigravitySdkRuntime,
    tmp_path: Path,
) -> None:
    """Verify select populates reasoning in SelectionOutcome."""
    runtime._resident = ("gke-basics",)
    _fake_agent(
        monkeypatch,
        _FakeResponse(
            structured={"selected_skill": "gke-basics", "reasoning": "Target skill matches"},
        ),
    )
    outcome = runtime.select("how do I set up a cluster?", tmp_path / "work")
    assert outcome.reasoning == ("Target skill matches",)


def test_select_reports_abstention_when_nothing_was_selected(
    monkeypatch: pytest.MonkeyPatch,
    runtime: AntigravitySdkRuntime,
    tmp_path: Path,
) -> None:
    """Verify select reports None when structured output specifies null selected_skill."""
    runtime._resident = ("gke-basics",)
    _fake_agent(monkeypatch, _FakeResponse(structured={"selected_skill": None}))
    outcome = runtime.select("what's the weather", tmp_path / "work")
    assert outcome.invoked_skill is None
    assert outcome.invoked_skills == ()
    assert outcome.error is None


def test_select_reports_no_structured_output_as_rate_limited_error(
    monkeypatch: pytest.MonkeyPatch,
    runtime: AntigravitySdkRuntime,
    tmp_path: Path,
) -> None:
    """Verify missing structured output with no invoked skills flags empty selection error."""
    runtime._resident = ("gke-basics",)
    _fake_agent(monkeypatch, _FakeResponse(structured=None))
    outcome = runtime.select("q", tmp_path / "work")
    assert outcome.invoked_skill is None
    assert outcome.invoked_skills == ()
    assert outcome.error == "empty selection (likely rate-limited)"


def test_select_flags_a_tool_surviving_denial_as_a_leak(
    monkeypatch: pytest.MonkeyPatch,
    runtime: AntigravitySdkRuntime,
    tmp_path: Path,
) -> None:
    """Verify unexpected non-finish tool calls in response are flagged as tool leaks."""
    runtime._resident = ("gke-basics",)
    leaked_call = ag_types.ToolCall(name=ag_types.BuiltinTools.RUN_COMMAND, args={})
    _fake_agent(
        monkeypatch,
        _FakeResponse(
            structured={"selected_skill": "gke-basics"},
            tool_calls=[leaked_call],
        ),
    )
    outcome = runtime.select("q", tmp_path / "work")
    assert outcome.error is not None
    assert "run_command" in outcome.error


def test_select_accepts_max_model_calls_exceeded_as_a_clean_completion(
    monkeypatch: pytest.MonkeyPatch,
    runtime: AntigravitySdkRuntime,
    tmp_path: Path,
) -> None:
    """Verify MAX_MODEL_CALLS_EXCEEDED stop reason is treated as valid successful turn."""
    runtime._resident = ("gke-basics",)
    _fake_agent(
        monkeypatch,
        _FakeResponse(
            structured={"selected_skill": "gke-basics"},
            stop_reason=ag_types.StopReason.MAX_MODEL_CALLS_EXCEEDED,
        ),
    )
    outcome = runtime.select("q", tmp_path / "work")
    assert outcome.error is None
    assert outcome.invoked_skill == "gke-basics"


def test_select_reports_an_unexpected_stop_reason_as_an_error(
    monkeypatch: pytest.MonkeyPatch,
    runtime: AntigravitySdkRuntime,
    tmp_path: Path,
) -> None:
    """Verify unexpected stop reasons like QUOTA_EXHAUSTED record errors."""
    runtime._resident = ("gke-basics",)
    _fake_agent(
        monkeypatch,
        _FakeResponse(stop_reason=ag_types.StopReason.QUOTA_EXHAUSTED),
    )
    outcome = runtime.select("q", tmp_path / "work")
    assert outcome.error is not None
    assert "QUOTA_EXHAUSTED" in outcome.error


def test_select_reports_a_backend_failure_not_raises(
    monkeypatch: pytest.MonkeyPatch,
    runtime: AntigravitySdkRuntime,
    tmp_path: Path,
) -> None:
    """Verify backend agent exceptions are captured into outcome.error string."""
    runtime._resident = ("gke-basics",)

    class _RaisingAgent(_FakeAgent):
        async def __aenter__(self):
            msg = "no credentials found"
            raise RuntimeError(msg)

    monkeypatch.setattr("reach.runtime.antigravity_sdk.Agent", _RaisingAgent)
    outcome = runtime.select("q", tmp_path / "work")
    assert outcome.error == "no credentials found"


def test_select_falls_back_to_a_thread_when_a_loop_is_already_running(
    monkeypatch: pytest.MonkeyPatch,
    runtime: AntigravitySdkRuntime,
    tmp_path: Path,
) -> None:
    """Verify select executes cleanly when called from within an existing event loop."""
    runtime._resident = ("gke-basics",)
    _fake_agent(monkeypatch, _FakeResponse(structured={"selected_skill": "gke-basics"}))

    async def call_from_inside_a_running_loop():
        return runtime.select("how do I set up a cluster?", tmp_path / "work")

    outcome = asyncio.run(call_from_inside_a_running_loop())
    assert outcome.invoked_skill == "gke-basics"
    assert outcome.error is None


def test_complete_falls_back_to_a_thread_when_a_loop_is_already_running(
    monkeypatch: pytest.MonkeyPatch,
    generator: AntigravitySdkGenerator,
) -> None:
    """Verify complete executes cleanly when invoked from within a running event loop."""
    _fake_agent(monkeypatch, _FakeResponse(text="drafted query set"))

    async def call_from_inside_a_running_loop():
        return generator.complete("draft some queries")

    result = asyncio.run(call_from_inside_a_running_loop())
    assert result == "drafted query set"


def test_select_reports_a_timeout(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Verify select returns timeout error when asyncio.wait_for times out."""
    runtime = AntigravitySdkRuntime(
        options=AntigravitySdkOptions(model="test-model"),
    )
    runtime._resident = ("gke-basics",)

    async def timeout(fut, *_args, **_kwargs) -> Never:
        if asyncio.iscoroutine(fut):
            fut.close()
        raise TimeoutError

    monkeypatch.setattr(asyncio, "wait_for", timeout)
    outcome = runtime.select("q", tmp_path / "work")
    assert outcome.error == "timeout"


def test_complete_returns_the_scripted_text(
    monkeypatch: pytest.MonkeyPatch,
    generator: AntigravitySdkGenerator,
) -> None:
    """Verify complete returns plain response text from agent."""
    instances = _fake_agent(monkeypatch, _FakeResponse(text="drafted query set"))
    result = generator.complete("draft some queries")
    assert result == "drafted query set"
    assert instances[0].sent == "draft some queries"
    assert generator.completions == 1


def test_complete_uses_no_isolation(
    monkeypatch: pytest.MonkeyPatch,
    generator: AntigravitySdkGenerator,
) -> None:
    """Verify complete creates agent config without budget or skill path constraints."""
    instances = _fake_agent(monkeypatch, _FakeResponse(text="ok"))
    generator.complete("q")
    config = instances[0].config
    assert config.budget_config is None
    assert config.skills_paths == []


def test_complete_passes_response_schema(
    monkeypatch: pytest.MonkeyPatch,
    generator: AntigravitySdkGenerator,
) -> None:
    """Verify complete populates response_schema on agent config when schema is provided."""
    instances = _fake_agent(monkeypatch, _FakeResponse(structured={"queries": []}, text="{}"))
    schema = {"type": "object", "properties": {"queries": {"type": "array"}}}
    generator.complete("q", schema=schema)
    config = instances[0].config
    assert config.response_schema == json.dumps(schema)


def test_complete_uses_thinking_config_when_effort_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify complete creates agent config with thinking options when effort is configured."""
    gen = AntigravitySdkGenerator(
        options=AntigravitySdkOptions(model="gemini-3.8-flash", effort="high"),
    )
    instances = _fake_agent(monkeypatch, _FakeResponse(text="response text"))
    gen.complete("prompt")
    config = instances[0].config
    assert isinstance(config.model, ag_types.ModelTarget)
    assert isinstance(config.model.endpoint, ag_types.GeminiAPIEndpoint)
    assert isinstance(config.model.endpoint.options, ag_types.GeminiModelOptions)
    assert config.model.endpoint.options.thinking_level == ag_types.ThinkingLevel.HIGH


def test_antigravity_sdk_missing_dependency_raises_helpful_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify build_runtime raises actionable RuntimeError when google.antigravity is missing."""
    from reach.config import RuntimeSettings
    from reach.runtime import build_runtime

    orig_import_module = importlib.import_module

    def mock_import_module(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "reach.runtime.antigravity_sdk":
            msg = "No module named 'google.antigravity'"
            raise ImportError(msg)
        return orig_import_module(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", mock_import_module)

    with pytest.raises(RuntimeError, match=r"pip install 'skill-reach\[antigravity-sdk\]'"):
        build_runtime(RuntimeSettings(agent="antigravity-sdk"))


def test_antigravity_sdk_generator_missing_dependency_raises_helpful_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify build_text_generator raises RuntimeError when google.antigravity is missing."""
    from reach.runtime.generator import build_text_generator

    orig_import_module = importlib.import_module

    def mock_import_module(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "reach.runtime.antigravity_sdk":
            msg = "No module named 'google.antigravity'"
            raise ImportError(msg)
        return orig_import_module(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", mock_import_module)

    with pytest.raises(RuntimeError, match=r"pip install 'skill-reach\[antigravity-sdk\]'"):
        build_text_generator(agent="antigravity-sdk")


def test_antigravity_sdk_uninstalled_raises_helpful_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify instantiating AntigravitySdkRuntime raises RuntimeError when uninstalled."""
    monkeypatch.setattr("reach.runtime.antigravity_sdk._HAS_ANTIGRAVITY", False)
    with pytest.raises(RuntimeError, match=r"pip install 'skill-reach\[antigravity-sdk\]'"):
        AntigravitySdkRuntime()


def test_antigravity_sdk_generator_uninstalled_raises_helpful_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify instantiating AntigravitySdkGenerator raises RuntimeError when uninstalled."""
    monkeypatch.setattr("reach.runtime.antigravity_sdk._HAS_ANTIGRAVITY", False)
    with pytest.raises(RuntimeError, match=r"pip install 'skill-reach\[antigravity-sdk\]'"):
        AntigravitySdkGenerator()


def test_antigravity_sdk_options_defaults() -> None:
    """Verify AntigravitySdkOptions default parameters for performance and isolation."""
    opts = AntigravitySdkOptions()
    assert opts.use_symlinks is False
    assert opts.isolate_config_dir is True
    assert opts.auto_clean is False
    assert opts.app_data_dir is None


def test_antigravity_sdk_build_env_synchronizes_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify build_env synchronizes GEMINI_API_KEY and GOOGLE_API_KEY bidirectionally."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    # Option key takes priority and sets both
    rt_opt = AntigravitySdkRuntime(
        options=AntigravitySdkOptions(model="test-model", api_key="my-api-key"),
    )
    env_opt = rt_opt.build_env()
    assert env_opt["GEMINI_API_KEY"] == "my-api-key"
    assert env_opt["GOOGLE_API_KEY"] == "my-api-key"

    # Ambient GEMINI_API_KEY synchronizes to GOOGLE_API_KEY
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-only")
    rt_gemini = AntigravitySdkRuntime(
        options=AntigravitySdkOptions(model="test-model"),
    )
    env_gemini = rt_gemini.build_env()
    assert env_gemini["GEMINI_API_KEY"] == "gemini-only"
    assert env_gemini["GOOGLE_API_KEY"] == "gemini-only"

    # Ambient GOOGLE_API_KEY synchronizes to GEMINI_API_KEY
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "google-only")
    rt_google = AntigravitySdkRuntime(
        options=AntigravitySdkOptions(model="test-model"),
    )
    env_google = rt_google.build_env()
    assert env_google["GEMINI_API_KEY"] == "google-only"
    assert env_google["GOOGLE_API_KEY"] == "google-only"


def test_select_config_sets_isolated_app_data_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify _select_config sets app_data_dir and creates directory when isolation is enabled."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-api-key")
    workdir = tmp_path / "work"
    workdir.mkdir()

    # Default isolation sets per-worker slot under workdir / .reach_antigravity_sdk
    from reach.runtime._fs import probe_slot_dir

    rt = AntigravitySdkRuntime(options=AntigravitySdkOptions(model="test-model"))
    config = rt._select_config(workdir)
    expected_dir = probe_slot_dir((workdir / ".reach_antigravity_sdk").resolve())
    assert config.app_data_dir == str(expected_dir)
    assert expected_dir.is_dir()
    assert "GEMINI_API_KEY" in (config.env or {}) or "GOOGLE_API_KEY" in (config.env or {})

    # Custom app_data_dir is respected
    custom_dir = tmp_path / "custom_app_data"
    rt_custom = AntigravitySdkRuntime(
        options=AntigravitySdkOptions(model="test-model", app_data_dir=custom_dir),
    )
    config_custom = rt_custom._select_config(workdir)
    assert config_custom.app_data_dir == str(custom_dir.resolve())
    assert custom_dir.is_dir()

    # Disabling isolation leaves app_data_dir None
    rt_no_iso = AntigravitySdkRuntime(
        options=AntigravitySdkOptions(model="test-model", isolate_config_dir=False),
    )
    config_no_iso = rt_no_iso._select_config(workdir)
    assert config_no_iso.app_data_dir is None


def test_select_invokes_post_probe(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify select cleans isolated directory when auto_clean is True."""
    workdir = tmp_path / "work"
    workdir.mkdir()

    rt = AntigravitySdkRuntime(
        options=AntigravitySdkOptions(model="test-model", auto_clean=True),
    )
    rt._resident = ("gke-basics",)
    _fake_agent(monkeypatch, _FakeResponse(structured={"selected_skill": "gke-basics"}))

    outcome = rt.select("how do I set up a cluster?", workdir)
    assert outcome.invoked_skill == "gke-basics"
    assert not (workdir / ".reach_antigravity_sdk").exists()


@pytest.mark.parametrize(
    ("use_custom_dir", "expect_exists"),
    [
        (False, False),
        (True, True),
    ],
)
def test_post_probe_always_cleans_ephemeral_slot_even_when_auto_clean_false(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    use_custom_dir: bool,
    expect_exists: bool,
) -> None:
    """Verify ephemeral slot_dir is cleaned when auto_clean=False while custom dir stays."""
    workdir = tmp_path / "work_default_clean"
    workdir.mkdir()
    target_dir = (workdir / "user_app_data") if use_custom_dir else None

    rt = AntigravitySdkRuntime(
        options=AntigravitySdkOptions(
            model="test-model",
            auto_clean=False,
            app_data_dir=target_dir,
        ),
    )
    rt._resident = ("gke-basics",)
    _fake_agent(monkeypatch, _FakeResponse(structured={"selected_skill": "gke-basics"}))

    outcome = rt.select("how do I set up a cluster?", workdir)
    assert outcome.invoked_skill == "gke-basics"
    checked_path = target_dir if target_dir is not None else (workdir / ".reach_antigravity_sdk")
    assert checked_path.exists() is expect_exists


def test_select_recovers_skill_from_view_file_directory_step_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify select recovers resident skill when cortex rejects view_file on a skill directory."""
    workdir = tmp_path / "work_dir_err"
    skill_dir = workdir / ".agents" / "skills" / "bigquery-slot-cost-optimizer"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: bigquery-slot-cost-optimizer\n---\n",
        encoding="utf-8",
    )

    rt = AntigravitySdkRuntime(
        options=AntigravitySdkOptions(model="test-model"),
    )
    rt._resident = ("bigquery-slot-cost-optimizer", "other-skill")

    step_err = (
        "The model produced an invalid tool call. "
        '("model output error: invalid tool call error (invalid_args) failed to read file: '
        f"read '{skill_dir}': is a directory\")"
    )
    history_step = type(
        "_Step",
        (),
        {"status": "ERROR", "error": step_err, "http_code": 0},
    )()
    _fake_agent(
        monkeypatch,
        _FakeResponse(structured=None, text=""),
        history=[history_step],
    )

    outcome = rt.select(
        "optimize bigquery slots",
        workdir,
        target_skill="bigquery-slot-cost-optimizer",
    )
    assert outcome.error is None
    assert outcome.invoked_skills == ("bigquery-slot-cost-optimizer",)


def test_select_preserves_turn1_directory_skill_order_and_cancels_on_early_exit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify Turn-1 directory view_file precedes Turn-2 skill and triggers early-exit cancel."""
    workdir = tmp_path / "work_traj_order"
    for name in ("skill-a", "skill-b"):
        s_dir = workdir / ".agents" / "skills" / name
        s_dir.mkdir(parents=True)
        (s_dir / "SKILL.md").write_text(f"---\nname: {name}\n---\n", encoding="utf-8")

    skill_a_dir = workdir / ".agents" / "skills" / "skill-a"
    skill_b_file = workdir / ".agents" / "skills" / "skill-b" / "SKILL.md"
    step_err_a = (
        "The model produced an invalid tool call. "
        '("model output error: invalid tool call error (invalid_args) failed to read file: '
        f'read {skill_a_dir}: is a directory")'
    )
    step1 = _FakeStep(status="ERROR", error=step_err_a, http_code=0)
    tc_b = ag_types.ToolCall(name="view_file", args={"AbsolutePath": str(skill_b_file)})
    step2 = type("_Step", (), {"status": "DONE", "error": "", "tool_calls": [tc_b]})()

    # 1. Chronological trajectory order when Turn 2 fires _on_tool_call after Turn 1 dir error
    rt_multi = AntigravitySdkRuntime(
        options=AntigravitySdkOptions(model="test-model", early_exit=False),
    )
    rt_multi._resident = ("skill-a", "skill-b")
    _fake_agent(
        monkeypatch,
        _FakeResponse(structured=None, text="done", tool_calls=[tc_b]),
        history=[step1, step2],
    )
    outcome_multi = rt_multi.select("use skills", workdir)
    assert outcome_multi.invoked_skills == ("skill-a", "skill-b")

    # 2. Real-time _on_post_step hook triggers early_exit and cancels connection on Turn 1,
    # and _select_async suppresses asyncio.CancelledError when early_exit_hit is True
    rt_early = AntigravitySdkRuntime(
        options=AntigravitySdkOptions(model="test-model", early_exit=True),
    )
    rt_early._resident = ("skill-a", "skill-b")
    cancelled: list[bool] = []

    from typing import Self

    class _CancellingAgent:
        def __init__(self, config: Any) -> None:
            self.config = config
            conn = type("_Conn", (), {"cancel": staticmethod(lambda: cancelled.append(True))})()
            self.conversation = type("_Conv", (), {"connection": conn, "history": [step1]})()

        async def __aenter__(self) -> Self:
            return self

        async def __aexit__(self, *_: object) -> None:
            return None

        async def chat(self, _query: str) -> Any:
            await self.config.hooks[1](step1)
            raise asyncio.CancelledError

    monkeypatch.setattr("reach.runtime.antigravity_sdk.Agent", _CancellingAgent)
    outcome_early = rt_early.select("use skill a", workdir, target_skill="skill-a")
    assert outcome_early.error is None
    assert outcome_early.early_exit is True
    assert outcome_early.invoked_skills == ("skill-a",)
    assert cancelled == [True]


def test_post_probe_concurrent_workers_do_not_delete_active_sibling_slots(
    tmp_path: Path,
) -> None:
    """Verify post_probe cleans only its own slot and preserves active sibling slots."""
    import threading

    workdir = tmp_path / "concurrent_work"
    workdir.mkdir()
    rt = AntigravitySdkRuntime(
        options=AntigravitySdkOptions(model="test-model", auto_clean=True),
    )

    config_main = rt._select_config(workdir)
    assert config_main.app_data_dir is not None
    main_slot = Path(config_main.app_data_dir)
    sentinel = main_slot / "active_session.json"
    sentinel.write_text("{}", encoding="utf-8")

    worker_slot_holder: list[Path] = []

    def _worker_probe() -> None:
        cfg_w = rt._select_config(workdir)
        assert cfg_w.app_data_dir is not None
        w_slot = Path(cfg_w.app_data_dir)
        worker_slot_holder.append(w_slot)
        (w_slot / "worker_session.json").write_text("{}", encoding="utf-8")
        rt.post_probe(workdir)

    t = threading.Thread(target=_worker_probe)
    t.start()
    t.join()

    assert len(worker_slot_holder) == 1
    assert worker_slot_holder[0] != main_slot
    assert not worker_slot_holder[0].exists()
    # Main thread's slot and sentinel file must remain intact while active!
    assert main_slot.is_dir()
    assert sentinel.is_file()

    # When main thread finishes and runs post_probe, both slot and parent are removed
    rt.post_probe(workdir)
    assert not main_slot.exists()
    assert not (workdir / ".reach_antigravity_sdk").exists()


def test_complete_passes_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify complete passes synchronized environment to LocalAgentConfig."""
    gen = AntigravitySdkGenerator(
        options=AntigravitySdkOptions(model="test-model", api_key="complete-key"),
    )
    instances = _fake_agent(monkeypatch, _FakeResponse(text="completion text"))
    gen.complete("test prompt")
    config = instances[0].config
    assert config.env is not None
    assert config.env.get("GEMINI_API_KEY") == "complete-key"
    assert config.env.get("GOOGLE_API_KEY") == "complete-key"


def test_generator_build_env_sanitizes_blocked_env_vars(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify generator build_env sanitizes ambient secrets and respects blocked_env_vars."""
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "leaked-secret")
    monkeypatch.setenv("CUSTOM_SECRET", "custom-value")
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")

    # Default settings strip default blocked vars
    default_gen = AntigravitySdkGenerator(
        options=AntigravitySdkOptions(model="test-model"),
    )
    default_env = default_gen.build_env()
    assert "AWS_SECRET_ACCESS_KEY" not in default_env
    assert default_env.get("CUSTOM_SECRET") == "custom-value"
    assert default_env.get("GEMINI_API_KEY") == "test-gemini-key"

    # Custom blocked settings strip specified vars
    settings = RuntimeSettings(blocked_env_vars=["CUSTOM_SECRET"])
    custom_gen = AntigravitySdkGenerator(
        options=AntigravitySdkOptions(model="test-model"),
        settings=settings,
    )
    custom_env = custom_gen.build_env()
    assert "CUSTOM_SECRET" not in custom_env
    assert custom_env.get("GEMINI_API_KEY") == "test-gemini-key"


@pytest.mark.parametrize(
    ("max_turns", "early_exit"),
    [
        (1, False),
        (3, True),
    ],
)
def test_antigravity_sdk_omits_forced_schema_in_both_single_and_multi_turn(
    max_turns: int,
    early_exit: bool,
    tmp_path: Path,
) -> None:
    """Verify AntigravitySdkRuntime omits forced response_schema in both single and multi-turn."""
    rt = AntigravitySdkRuntime(
        options=AntigravitySdkOptions(
            model="test-model",
            app_data_dir=tmp_path / "app_data",
            max_turns=max_turns,
            early_exit=early_exit,
        ),
    )
    rt._resident = ("skill-a", "skill-b")
    config = rt._select_config(tmp_path)
    assert config.response_schema is None


def test_select_organic_text_abstention_vs_empty_rate_limit(
    monkeypatch: pytest.MonkeyPatch,
    runtime: AntigravitySdkRuntime,
    tmp_path: Path,
) -> None:
    """Verify non-empty text response without view_file is an abstention, not an error."""
    runtime._resident = ("gke-basics",)
    _fake_agent(
        monkeypatch,
        _FakeResponse(
            structured=None,
            text="You can create a regional bucket with gcloud storage buckets create.",
            tool_calls=[],
        ),
    )
    abstention = runtime.select("how do I create a bucket?", tmp_path / "work")
    assert abstention.error is None
    assert abstention.invoked_skills == ()
    assert abstention.reasoning == (
        "You can create a regional bucket with gcloud storage buckets create.",
    )


def test_build_model_spec_plain_and_effort() -> None:
    """Verify _build_model_spec returns plain string or ModelTarget based on effort."""
    assert _build_model_spec("plain-model") == "plain-model"
    assert _build_model_spec("plain-model", None) == "plain-model"

    target = _build_model_spec("gemini-3.8-flash", "high")
    assert isinstance(target, ag_types.ModelTarget)
    assert target.name == "gemini-3.8-flash"
    assert isinstance(target.endpoint, ag_types.GeminiAPIEndpoint)
    assert target.endpoint.options is not None
    assert target.endpoint.options.thinking_level == ag_types.ThinkingLevel.HIGH


def test_select_async_registers_hooks_in_config(
    monkeypatch: pytest.MonkeyPatch,
    runtime: AntigravitySdkRuntime,
    tmp_path: Path,
) -> None:
    """Verify _select_async registers decide and post-step hooks in LocalAgentConfig hooks."""
    runtime._resident = ("gke-basics",)
    instances = _fake_agent(monkeypatch, _FakeResponse(structured={"selected_skill": "gke-basics"}))
    runtime.select("how to setup", tmp_path / "work")
    assert len(instances) == 1
    assert instances[0].config.hooks is not None
    assert len(instances[0].config.hooks) == 2


def test_select_async_hook_intercepts_target_skill_early_exit(
    monkeypatch: pytest.MonkeyPatch,
    runtime: AntigravitySdkRuntime,
    tmp_path: Path,
) -> None:
    """Verify pre-tool decide hook detects skill and denies tool call on target match."""
    runtime._resident = ("gke-basics", "cloud-run-basics")
    instances = _fake_agent(monkeypatch, _FakeResponse(structured={"selected_skill": "gke-basics"}))

    # Run select to trigger config assembly with target_skill
    outcome = runtime.select("how to setup", tmp_path / "work", target_skill="gke-basics")
    assert outcome.invoked_skill == "gke-basics"
    assert len(instances) == 1
    hook_fn = instances[0].config.hooks[0]

    async def invoke_hook() -> None:
        # Test non-matching skill on Turn 1 (allowed to continue)
        tool_call_other = ag_types.ToolCall(
            name="view_file",
            args={"path": "/workspace/.agents/skills/cloud-run-basics/SKILL.md"},
        )
        result_other = await hook_fn(tool_call_other)
        assert result_other.allow is True

        # Test invoking the registered hook with matching target skill on Turn 2 (early-exit deny)
        tool_call_match = ag_types.ToolCall(
            name="view_file",
            args={"path": "/workspace/.agents/skills/gke-basics/SKILL.md"},
        )
        result_match = await hook_fn(tool_call_match)
        assert result_match.allow is False

        # Verify post-exit lock: any subsequent tool call after early_exit_hit remains denied
        result_after_exit = await hook_fn(tool_call_other)
        assert result_after_exit.allow is False

    asyncio.run(invoke_hook())


def test_generator_model_precedence() -> None:
    """Verify explicit model argument overrides default options.model in generator."""
    opts = AntigravitySdkOptions(model="default-model")
    gen = AntigravitySdkGenerator(model="explicit-model", options=opts)
    assert gen.model == "explicit-model"
    assert gen.options.model == "explicit-model"


def test_generator_settings_options_fallback() -> None:
    """Verify generator resolves options from settings.options when options is None."""
    settings = RuntimeSettings(options={"model": "settings-model", "effort": "low"})
    gen = AntigravitySdkGenerator(settings=settings)
    assert gen.model == "settings-model"
    assert gen.options.model == "settings-model"
    assert gen.options.effort == "low"


def test_generator_effective_effort_robustness() -> None:
    """Verify generator effective_effort handles off/none strings and unknown models."""
    gen_off = AntigravitySdkGenerator(
        options=AntigravitySdkOptions(model="gemini-3.8-flash", effort="off"),
    )
    assert gen_off.effective_effort is None

    gen_none = AntigravitySdkGenerator(
        options=AntigravitySdkOptions(model="gemini-3.8-flash", effort="None"),
    )
    assert gen_none.effective_effort is None

    gen_unknown = AntigravitySdkGenerator(
        options=AntigravitySdkOptions(model="custom-unknown-model-xyz"),
    )
    assert gen_unknown.effective_effort is None


def test_select_handles_string_stop_reason_without_error(
    monkeypatch: pytest.MonkeyPatch,
    runtime: AntigravitySdkRuntime,
    tmp_path: Path,
) -> None:
    """Verify non-enum string stop_reason is cleanly converted to error message."""
    runtime._resident = ("gke-basics",)
    _fake_agent(monkeypatch, _FakeResponse(stop_reason="BACKEND_DISCONNECT"))
    outcome = runtime.select("q", tmp_path / "work")
    assert outcome.error == "runtime error: BACKEND_DISCONNECT"


def test_select_ignores_empty_or_whitespace_reasoning(
    monkeypatch: pytest.MonkeyPatch,
    runtime: AntigravitySdkRuntime,
    tmp_path: Path,
) -> None:
    """Verify whitespace-only reasoning string results in empty tuple reasoning."""
    runtime._resident = ("gke-basics",)
    _fake_agent(
        monkeypatch,
        _FakeResponse(structured={"selected_skill": "gke-basics", "reasoning": "   \n\t  "}),
    )
    outcome = runtime.select("q", tmp_path / "work")
    assert outcome.reasoning == ()


def test_build_model_spec_vertex_endpoint() -> None:
    """Verify _build_model_spec returns VertexEndpoint when vertex is True."""
    target = _build_model_spec(
        "gemini-3.8-flash",
        "low",
        vertex=True,
        project="my-project",
        location="global",
    )
    assert isinstance(target, ag_types.ModelTarget)
    assert target.name == "gemini-3.8-flash"
    assert isinstance(target.endpoint, ag_types.VertexEndpoint)
    assert target.endpoint.project == "my-project"
    assert target.endpoint.location == "global"
    assert target.endpoint.options is not None
    assert target.endpoint.options.thinking_level == ag_types.ThinkingLevel.LOW


def test_effective_vertex_and_project_location_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify effective_vertex, effective_project, and effective_location resolution."""
    monkeypatch.delenv("GOOGLE_GENAI_USE_ENTERPRISE", raising=False)
    monkeypatch.delenv("GOOGLE_GENAI_USE_VERTEXAI", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_LOCATION", raising=False)

    # Explicit options
    rt = AntigravitySdkRuntime(
        options=AntigravitySdkOptions(vertex=True, project="p1", location="loc1"),
    )
    assert rt.effective_vertex is True
    assert rt.effective_project == "p1"
    assert rt.effective_location == "loc1"

    # Environment fallback
    monkeypatch.setenv("GOOGLE_GENAI_USE_ENTERPRISE", "true")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "env-project")
    rt_env = AntigravitySdkRuntime(options=AntigravitySdkOptions())
    assert rt_env.effective_vertex is True
    assert rt_env.effective_project == "env-project"
    assert rt_env.effective_location == "global"


def test_express_vs_standard_mode_adc_key_isolation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify ADC key isolation and Express Mode precedence."""
    monkeypatch.setenv("GOOGLE_GENAI_USE_ENTERPRISE", "true")
    monkeypatch.setenv("GEMINI_API_KEY", "ambient-key")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "ambient-project")

    # Standard Mode (ADC): options.api_key is None
    rt = AntigravitySdkRuntime(options=AntigravitySdkOptions())
    assert rt.effective_vertex is True
    assert rt.effective_api_key is None
    env = rt.build_env()
    assert "GEMINI_API_KEY" not in env
    assert "GOOGLE_API_KEY" not in env

    # Express Mode: options.api_key is explicitly provided
    rt_express = AntigravitySdkRuntime(options=AntigravitySdkOptions(api_key="express-key"))
    assert rt_express.effective_api_key == "express-key"
    assert rt_express.effective_project is None
    assert rt_express.effective_location is None


def test_select_config_passes_vertex_and_project_location(tmp_path: Path) -> None:
    """Verify _select_config passes vertex, project, and location to LocalAgentConfig."""
    rt = AntigravitySdkRuntime(
        options=AntigravitySdkOptions(vertex=True, project="my-p", location="my-loc"),
    )
    cfg = rt._select_config(tmp_path)
    assert cfg.vertex is True
    assert cfg.project == "my-p"
    assert cfg.location == "my-loc"


def test_complete_converts_antigravity_validation_error_to_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify complete converts AntigravityValidationError to RuntimeError."""
    from google.antigravity.types import AntigravityValidationError

    class _ValidatingAgent(_FakeAgent):
        async def __aenter__(self):
            msg = "A Gemini API key is required."
            raise AntigravityValidationError(msg)

    monkeypatch.setattr("reach.runtime.antigravity_sdk.Agent", _ValidatingAgent)
    gen = AntigravitySdkGenerator()
    with pytest.raises(RuntimeError, match=r"generation failed: A Gemini API key is required\."):
        gen.complete("test prompt")


def test_build_model_spec_vertex_endpoint_without_effort() -> None:
    """Verify _build_model_spec instantiates VertexEndpoint when vertex=True and effort=None."""
    target = _build_model_spec(
        "gemini-3.8-flash",
        effort=None,
        vertex=True,
        project="p1",
        location="us-central1",
    )
    assert isinstance(target, ag_types.ModelTarget)
    assert target.name == "gemini-3.8-flash"
    assert isinstance(target.endpoint, ag_types.VertexEndpoint)
    assert target.endpoint.project == "p1"
    assert target.endpoint.location == "us-central1"
    assert target.endpoint.options is None


def test_effective_project_and_location_none_when_vertex_is_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify effective_project and effective_location return None when vertex is disabled."""
    monkeypatch.delenv("GOOGLE_GENAI_USE_ENTERPRISE", raising=False)
    monkeypatch.delenv("GOOGLE_GENAI_USE_VERTEXAI", raising=False)
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "ambient-project")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "us-east1")

    rt = AntigravitySdkRuntime(options=AntigravitySdkOptions(vertex=False))
    assert rt.effective_vertex is False
    assert rt.effective_project is None
    assert rt.effective_location is None


def test_blocked_env_vars_strips_google_application_credentials_in_vertex(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify build_env does not restore GOOGLE_APPLICATION_CREDENTIALS when explicitly blocked."""
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/path/to/creds.json")
    rt = AntigravitySdkRuntime(
        settings=RuntimeSettings(blocked_env_vars=("GOOGLE_APPLICATION_CREDENTIALS",)),
        options=AntigravitySdkOptions(vertex=True),
    )
    env = rt.build_env()
    assert "GOOGLE_APPLICATION_CREDENTIALS" not in env


def test_antigravity_validation_error_stub_is_exception_subclass() -> None:
    """Verify AntigravityValidationError is a valid Exception subclass."""
    from reach.runtime.antigravity_sdk import AntigravityValidationError

    assert issubclass(AntigravityValidationError, Exception)


def test_select_async_hook_records_skill_when_early_exit_false_and_allows_view_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify _on_tool_call records trajectory when early_exit=False and allows view_file."""
    rt = AntigravitySdkRuntime(
        options=AntigravitySdkOptions(
            model="test-model",
            early_exit=False,
            allowed_tools=("finish", "view_file"),
        ),
    )
    rt._resident = ("gke-basics", "cloud-run-basics")
    config = rt._select_config(tmp_path / "work")
    assert config.capabilities is not None
    assert config.capabilities.enabled_tools is not None
    assert len(config.capabilities.enabled_tools) == 2

    instances = _fake_agent(
        monkeypatch,
        _FakeResponse(
            structured={"selected_skill": "gke-basics"},
            tool_calls=[
                ag_types.ToolCall(
                    name="view_file",
                    args={"path": "/workspace/.agents/skills/gke-basics/SKILL.md"},
                )
            ],
        ),
    )
    outcome = rt.select("how to setup", tmp_path / "work", target_skill="gke-basics")
    assert outcome.error is None
    assert outcome.invoked_skill == "gke-basics"
    hook_fn = instances[0].config.hooks[0]
    res = asyncio.run(
        hook_fn(
            ag_types.ToolCall(
                name="view_file",
                args={"path": "/workspace/.agents/skills/gke-basics/SKILL.md"},
            )
        )
    )
    assert res.allow is True


@pytest.mark.parametrize(
    ("max_turns", "early_exit", "allowed_tools", "expect_multi_turn"),
    [
        pytest.param(3, True, (), True, id="default-early-exit-uses-multi-turn-selection-tools"),
        pytest.param(
            1, True, (), True, id="single-turn-early-exit-uses-multi-turn-selection-tools"
        ),
        pytest.param(1, False, (), True, id="single-turn-uses-multi-turn-selection-tools"),
        pytest.param(
            3, False, (), True, id="multi-turn-trajectory-uses-multi-turn-selection-tools"
        ),
        pytest.param(
            3, False, ("finish",), False, id="explicit-allowed-tools-overrides-multi-turn"
        ),
    ],
)
def test_select_config_multi_turn_selection_tools(
    tmp_path: Path,
    max_turns: int,
    early_exit: bool,
    allowed_tools: tuple[str, ...],
    expect_multi_turn: bool,
) -> None:
    """Verify _select_config enables MULTI_TURN_SELECTION_TOOLS when max_turns > 1."""
    from reach.runtime.antigravity_sdk import MULTI_TURN_SELECTION_TOOLS, SELECTION_TOOLS

    assert len(MULTI_TURN_SELECTION_TOOLS) > len(SELECTION_TOOLS)
    rt = AntigravitySdkRuntime(
        options=AntigravitySdkOptions(
            model="test-model",
            max_turns=max_turns,
            early_exit=early_exit,
            allowed_tools=allowed_tools,
        ),
    )
    rt._resident = ("gke-basics",)
    config = rt._select_config(tmp_path / "work")
    assert config.capabilities is not None
    assert config.capabilities.enabled_tools is not None
    if expect_multi_turn:
        assert tuple(config.capabilities.enabled_tools) == MULTI_TURN_SELECTION_TOOLS
    else:
        assert tuple(config.capabilities.enabled_tools) == SELECTION_TOOLS


@pytest.mark.parametrize(
    ("model_name", "expected_thinking"),
    [
        ("gemini-2.5-flash", False),
        ("gemini-2.5-pro", False),
        ("gemini-2.0-flash", False),
        ("gemini-1.5-pro", False),
        ("gemini-3.8-flash", True),
        ("gemini-3.7-flash", True),
        ("gemini-3-flash-preview", True),
        ("claude-sonnet-5", True),
    ],
)
def test_model_supports_thinking(model_name: str, expected_thinking: bool) -> None:
    """Verify _model_supports_thinking flags models that do not support thinking levels."""
    from reach.runtime.antigravity_sdk import _model_supports_thinking

    assert _model_supports_thinking(model_name) is expected_thinking


def test_build_model_spec_drops_effort_for_non_thinking_models() -> None:
    """Verify _build_model_spec does not set thinking_level on models that do not support it."""
    spec25 = _build_model_spec("gemini-2.5-flash", effort="low")
    assert spec25 == "gemini-2.5-flash"

    spec38 = _build_model_spec("gemini-3.8-flash", effort="low")
    if ag_types is not None:
        assert isinstance(spec38, ag_types.ModelTarget)
        assert isinstance(spec38.endpoint, ag_types.GeminiAPIEndpoint)
        assert spec38.endpoint.options is not None
        assert spec38.endpoint.options.thinking_level == "low"


def test_generator_effective_effort_guards_against_unsupported_models() -> None:
    """Verify AntigravitySdkGenerator.effective_effort avoids fallback effort on 2.5 models."""
    gen25 = AntigravitySdkGenerator(model="gemini-2.5-flash")
    assert gen25.effective_effort is None

    gen38 = AntigravitySdkGenerator(model="gemini-3.8-flash")
    assert gen38.effective_effort == "low"


def test_generator_complete_uses_structured_output_when_available(
    monkeypatch: pytest.MonkeyPatch,
    generator: AntigravitySdkGenerator,
) -> None:
    """Verify AntigravitySdkGenerator.complete extracts structured output as JSON."""
    canned = {"queries": [{"text": "deploy a job", "citation": "cloud-run docs"}]}
    agents = _fake_agent(monkeypatch, _FakeResponse(structured=canned, text="Finished"))
    out = generator.complete("generate", schema={"type": "object"})
    assert json.loads(out) == canned
    assert json.loads(agents[0].config.response_schema) == {"type": "object"}
    assert agents[0].config.capabilities is not None
    assert agents[0].config.capabilities.enabled_tools == []
    assert agents[0].config.capabilities.enable_subagents is False


def test_generator_complete_returns_text_when_no_schema(
    monkeypatch: pytest.MonkeyPatch,
    generator: AntigravitySdkGenerator,
) -> None:
    """Verify AntigravitySdkGenerator.complete falls back to text when no schema provided."""
    agents = _fake_agent(monkeypatch, _FakeResponse(text="plain completion"))
    out = generator.complete("hello")
    assert out == "plain completion"
    assert getattr(agents[0].config, "response_schema", None) is None


def test_generator_complete_handles_non_callable_structured_output(
    monkeypatch: pytest.MonkeyPatch,
    generator: AntigravitySdkGenerator,
) -> None:
    """Verify AntigravitySdkGenerator.complete handles non-callable structured_output."""
    canned = {"queries": [{"text": "deploy a job", "citation": "cloud-run docs"}]}

    class PropertyResponse:
        structured_output = canned
        stop_reason = "STOP"

        async def text(self) -> str:
            return ""

    _fake_agent(monkeypatch, PropertyResponse())
    out = generator.complete("generate", schema={"type": "object"})
    assert json.loads(out) == canned


def test_generator_complete_handles_pydantic_structured_output(
    monkeypatch: pytest.MonkeyPatch,
    generator: AntigravitySdkGenerator,
) -> None:
    """Verify AntigravitySdkGenerator.complete serializes Pydantic model structured output."""
    from pydantic import BaseModel

    class QueryItem(BaseModel):
        text: str
        citation: str

    class QuerySet(BaseModel):
        queries: list[QueryItem]

    model_obj = QuerySet(queries=[QueryItem(text="deploy a job", citation="cloud-run docs")])
    _fake_agent(monkeypatch, _FakeResponse(structured=model_obj, text="Finished"))
    out = generator.complete("generate", schema={"type": "object"})
    assert json.loads(out) == {"queries": [{"text": "deploy a job", "citation": "cloud-run docs"}]}


@pytest.mark.parametrize(
    ("steps", "stop_reason", "expected_error"),
    [
        pytest.param(
            [_FakeStep(http_code=429, error="Resource exhausted")],
            "UNSPECIFIED",
            "rate limit (429): Resource exhausted",
            id="http-429-resource-exhausted",
        ),
        pytest.param(
            [_FakeStep(http_code=429, error="Resource exhausted")],
            "END_TURN",
            "rate limit (429): Resource exhausted",
            id="end-turn-stop-reason-surfaces-429-instead-of-runtime-error",
        ),
        pytest.param(
            [_FakeStep(http_code=0, error="HTTP 429 Too Many Requests: quota exceeded")],
            "UNSPECIFIED",
            "rate limit (429): HTTP 429 Too Many Requests: quota exceeded",
            id="implicit-429-in-error-string",
        ),
        pytest.param(
            [_FakeStep(http_code=503, error="Service Unavailable")],
            "UNSPECIFIED",
            "sdk step error (HTTP 503): Service Unavailable",
            id="http-503-service-unavailable",
        ),
        pytest.param(
            [_FakeStep(http_code="invalid", error="Internal stream disconnect")],
            "UNSPECIFIED",
            "sdk step error: Internal stream disconnect",
            id="non-numeric-http-code-system-step-error",
        ),
        pytest.param(
            [_FakeStep(status="COMPLETED", http_code=200, error="non-fatal info notice")],
            "UNSPECIFIED",
            "empty selection (likely rate-limited)",
            id="completed-step-with-info-string-ignored",
        ),
    ],
)
def test_select_surfaces_history_error_when_output_empty(
    monkeypatch: pytest.MonkeyPatch,
    runtime: AntigravitySdkRuntime,
    tmp_path: Path,
    steps: list[_FakeStep],
    stop_reason: str,
    expected_error: str,
) -> None:
    """Verify select extracts HTTP/system error from conversation history on empty output."""
    runtime._resident = ("gke-basics",)
    _fake_agent(
        monkeypatch,
        _FakeResponse(structured=None, stop_reason=stop_reason),
        history=steps,
    )
    outcome = runtime.select("how do I set up a cluster?", tmp_path / "work")
    assert outcome.invoked_skills == ()
    assert outcome.error == expected_error
    assert outcome.observed_catalog == ("gke-basics",)


def test_select_ignores_transient_history_error_when_turn_succeeded(
    monkeypatch: pytest.MonkeyPatch,
    runtime: AntigravitySdkRuntime,
    tmp_path: Path,
) -> None:
    """Verify transient 429 step is ignored when internal retry succeeds."""
    runtime._resident = ("gke-basics",)
    transient_429 = _FakeStep(http_code=429, error="Resource exhausted")
    _fake_agent(
        monkeypatch,
        _FakeResponse(structured={"selected_skill": "gke-basics", "reasoning": "Recovered"}),
        history=[transient_429],
    )
    outcome = runtime.select("how do I set up a cluster?", tmp_path / "work")
    assert outcome.invoked_skill == "gke-basics"
    assert outcome.error is None


@pytest.mark.parametrize(
    ("tool_calls", "history_steps", "expected_error"),
    [
        pytest.param(
            [ag_types.ToolCall(name=ag_types.BuiltinTools.LIST_DIR, args={})]
            if ag_types is not None
            else [],
            [],
            None,
            id="max-model-calls-with-observed-tools-is-valid-abstention",
        ),
        pytest.param(
            [],
            [],
            "empty selection (likely rate-limited)",
            id="max-model-calls-with-zero-tools-is-flagged-as-error",
        ),
        pytest.param(
            [ag_types.ToolCall(name=ag_types.BuiltinTools.LIST_DIR, args={})]
            if ag_types is not None
            else [],
            [_FakeStep(http_code=429, error="Resource exhausted")],
            "rate limit (429): Resource exhausted",
            id="max-model-calls-with-history-429-surfaces-rate-limit",
        ),
    ],
)
def test_select_multi_turn_max_model_calls_exceeded_abstention_vs_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    tool_calls: list[Any],
    history_steps: list[Any],
    expected_error: str | None,
) -> None:
    """Verify multi-turn MAX_MODEL_CALLS_EXCEEDED separates exploration from silent drops."""
    rt = AntigravitySdkRuntime(
        options=AntigravitySdkOptions(
            model="test-model",
            max_turns=3,
            early_exit=False,
        ),
    )
    rt._resident = ("gke-basics",)
    _fake_agent(
        monkeypatch,
        _FakeResponse(
            structured=None,
            tool_calls=tool_calls,
            stop_reason=ag_types.StopReason.MAX_MODEL_CALLS_EXCEEDED,
        ),
        history=history_steps,
    )
    outcome = rt.select("explore workspace", tmp_path / "work")
    assert outcome.invoked_skills == ()
    assert outcome.error == expected_error


def test_select_single_turn_flags_multi_turn_only_tool_as_leak(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify single-turn (max_turns=1) select flags MULTI_TURN_SELECTION_TOOLS as tool leak."""
    rt = AntigravitySdkRuntime(
        options=AntigravitySdkOptions(model="test-model", max_turns=1, allowed_tools=("finish",)),
    )
    rt._resident = ("gke-basics",)
    _fake_agent(
        monkeypatch,
        _FakeResponse(
            structured={"selected_skill": "gke-basics", "reasoning": "ok"},
            tool_calls=[ag_types.ToolCall(name=ag_types.BuiltinTools.LIST_DIR, args={})],
        ),
    )
    outcome = rt.select("how do I set up a cluster?", tmp_path / "work")
    assert outcome.error == "tool leak: list_directory"


def test_select_triggers_probe_harness_retry_on_429_and_empty_selection(
    monkeypatch: pytest.MonkeyPatch,
    runtime: AntigravitySdkRuntime,
    tmp_path: Path,
) -> None:
    """Verify ProbeHarness._with_retries automatically retries 429/empty selection until success."""
    from reach.models import Catalog, CatalogMode, Query
    from reach.run import ProbeHarness

    runtime._resident = ("gke-basics",)
    attempts = 0

    def stateful_factory(config: Any) -> _FakeAgent:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            agent = _FakeAgent(
                config,
                history=[_FakeStep(http_code=429, error="Resource exhausted")],
            )
            agent.response = _FakeResponse(structured=None)
            return agent
        if attempts == 2:
            agent = _FakeAgent(config)
            agent.response = _FakeResponse(structured=None)
            return agent
        agent = _FakeAgent(config)
        agent.response = _FakeResponse(
            structured={"selected_skill": "gke-basics", "reasoning": "Succeeded on retry 2"},
        )
        return agent

    monkeypatch.setattr("reach.runtime.antigravity_sdk.Agent", stateful_factory)
    sleeps: list[float] = []
    harness = ProbeHarness(
        runtime,
        retries=2,
        backoff_s=1.5,
        sleep=sleeps.append,
        cache_outcomes=False,
    )
    catalog = Catalog(id="cat-1", mode=CatalogMode.ALL, skills=("gke-basics",))
    query = Query(id="q-1", text="create a gke cluster", expected_skill="gke-basics")

    results = list(harness.run_probes([query], catalog, tmp_path / "work", attempts=1))
    assert len(results) == 1
    assert attempts == 3
    assert sleeps == [1.5, 3.0]
    assert results[0].error is None
    assert results[0].invoked_skill == "gke-basics"


@pytest.mark.parametrize(
    ("schema", "history_steps", "expected_match"),
    [
        pytest.param(
            {"type": "object"},
            [_FakeStep(http_code=429, error="Resource exhausted")],
            r"generation failed: rate limit \(429\): Resource exhausted",
            id="schema-with-429-history-step",
        ),
        pytest.param(
            {"type": "object"},
            [],
            r"generation failed: empty structured output \(likely rate-limited\)",
            id="schema-with-missing-structured-output",
        ),
        pytest.param(
            None,
            [_FakeStep(http_code=429, error="Quota exceeded")],
            r"generation failed: rate limit \(429\): Quota exceeded",
            id="plain-text-with-429-history-step",
        ),
        pytest.param(
            None,
            [],
            r"generation failed: empty response \(likely rate-limited\)",
            id="plain-text-with-empty-response",
        ),
    ],
)
def test_generator_complete_surfaces_history_and_empty_errors(
    monkeypatch: pytest.MonkeyPatch,
    generator: AntigravitySdkGenerator,
    schema: dict[str, Any] | None,
    history_steps: list[Any],
    expected_match: str,
) -> None:
    """Verify AntigravitySdkGenerator.complete raises RuntimeError on 429 and empty responses."""
    _fake_agent(
        monkeypatch,
        _FakeResponse(structured=None, text=""),
        history=history_steps,
    )
    with pytest.raises(RuntimeError, match=expected_match):
        generator.complete("generate queries", schema=schema)
    assert generator.completions == 0


def test_antigravity_sdk_options_retry_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify api_max_retries and api_retry_jitter wire RetryConfig into LocalAgentConfig."""
    opts = AntigravitySdkOptions(
        model="test-model",
        api_max_retries=5,
        api_retry_jitter=0.25,
    )
    rt = AntigravitySdkRuntime(options=opts)
    rt._resident = ("gke-basics",)
    select_cfg = rt._select_config(tmp_path / "work")
    assert select_cfg.retry_config is not None
    assert select_cfg.retry_config.api_retry is not None
    assert select_cfg.retry_config.api_retry.max_retries == 5
    assert select_cfg.retry_config.api_retry.jitter_range == 0.25

    gen = AntigravitySdkGenerator(options=opts)
    instances = _fake_agent(monkeypatch, _FakeResponse(text="ok"))
    gen.complete("hello")
    assert instances[0].config.retry_config is not None
    assert instances[0].config.retry_config.api_retry is not None
    assert instances[0].config.retry_config.api_retry.max_retries == 5
    assert instances[0].config.retry_config.api_retry.jitter_range == 0.25


def test_select_preserves_observed_catalog_on_timeout_and_exception(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify select preserves observed_catalog when TimeoutError or Exception occurs."""
    rt = AntigravitySdkRuntime(options=AntigravitySdkOptions(model="test-model"))
    rt._resident = ("gke-basics", "cloud-run-basics")

    async def timeout(fut: Any, *_args: Any, **_kwargs: Any) -> Never:
        if asyncio.iscoroutine(fut):
            fut.close()
        raise TimeoutError

    monkeypatch.setattr(asyncio, "wait_for", timeout)
    timeout_outcome = rt.select("q", tmp_path / "work")
    assert timeout_outcome.error == "timeout"
    assert timeout_outcome.observed_catalog == ("gke-basics", "cloud-run-basics")

    async def boom(fut: Any, *_args: Any, **_kwargs: Any) -> Never:
        if asyncio.iscoroutine(fut):
            fut.close()
        msg = "connection reset"
        raise RuntimeError(msg)

    monkeypatch.setattr(asyncio, "wait_for", boom)
    exc_outcome = rt.select("q", tmp_path / "work")
    assert exc_outcome.error == "connection reset"
    assert exc_outcome.observed_catalog == ("gke-basics", "cloud-run-basics")


def test_trajectory_tracker_enforces_max_turns_when_early_exit_disabled() -> None:
    """Verify TrajectoryTracker.apply_to_outcome truncates at max_turns when early_exit=False."""
    from reach.runtime import SelectionOutcome, TrajectoryTracker

    tracker = TrajectoryTracker(target_skill="s1", max_turns=2, early_exit=False)
    assert tracker.observe("s1") is False
    assert tracker.observe("s2") is False
    assert tracker.observe("s3") is False
    assert tracker.early_exit_hit is False

    normalized = tracker.apply_to_outcome(
        SelectionOutcome(invoked_skills=("s1", "s2", "s3"), turns_taken=3),
    )
    assert normalized.invoked_skills == ("s1", "s2")
    assert normalized.turns_taken == 2
    assert normalized.early_exit is False


def test_select_async_hook_rewrites_skill_directory_to_skill_md_for_multi_turn_recovery(
    monkeypatch: pytest.MonkeyPatch,
    runtime: AntigravitySdkRuntime,
    tmp_path: Path,
) -> None:
    """Verify pre-tool hook rewrites skill directory paths to SKILL.md."""
    skills_root = tmp_path / "work" / ".agents" / "skills"
    distractor_dir = skills_root / "agent-platform-endpoint-management"
    target_dir = skills_root / "agent-platform-deploy"
    distractor_dir.mkdir(parents=True)
    target_dir.mkdir(parents=True)
    (distractor_dir / "SKILL.md").write_text("---\nname: agent-platform-endpoint-management\n---\n")
    (target_dir / "SKILL.md").write_text("---\nname: agent-platform-deploy\n---\n")

    runtime._resident = ("agent-platform-endpoint-management", "agent-platform-deploy")

    def agent_with_directory_tool_calls(config: Any) -> _FakeAgent:
        agent = _FakeAgent(config)
        hook_fn = config.hooks[0]

        class _HookRunnerResponse(_FakeResponse):
            async def structured_output(self) -> object:
                # Turn 1: Model calls view_file on distractor directory (without /SKILL.md)
                call_turn1 = ag_types.ToolCall(
                    name="view_file",
                    args={"AbsolutePath": str(distractor_dir)},
                )
                res_turn1 = await hook_fn(call_turn1)
                assert res_turn1.allow is True
                expected_md = str(distractor_dir / "SKILL.md")
                assert res_turn1.modified_args == {"AbsolutePath": expected_md}

                # Turn 2: Model recovers and calls view_file on target skill directory
                call_turn2 = ag_types.ToolCall(
                    name="view_file",
                    args={"AbsolutePath": str(target_dir)},
                )
                res_turn2 = await hook_fn(call_turn2)
                assert res_turn2.allow is False
                return {"selected_skill": "agent-platform-deploy"}

        agent.response = _HookRunnerResponse(
            tool_calls=[
                ag_types.ToolCall(name="view_file", args={"AbsolutePath": str(distractor_dir)}),
                ag_types.ToolCall(name="view_file", args={"AbsolutePath": str(target_dir)}),
            ],
        )
        return agent

    monkeypatch.setattr("reach.runtime.antigravity_sdk.Agent", agent_with_directory_tool_calls)
    outcome = runtime.select(
        "deploy my model to an endpoint",
        tmp_path / "work",
        target_skill="agent-platform-deploy",
    )
    assert outcome.invoked_skills == (
        "agent-platform-endpoint-management",
        "agent-platform-deploy",
    )
    assert outcome.turns_taken == 2
    assert outcome.early_exit is True

    # Verify list_dir(DirectoryPath=...) also attributes the skill without rewriting DirectoryPath
    def agent_with_list_dir_call(config: Any) -> _FakeAgent:
        agent = _FakeAgent(config)
        hook_fn = config.hooks[0]

        class _DirRunnerResponse(_FakeResponse):
            async def structured_output(self) -> object:
                call_dir = ag_types.ToolCall(
                    name="list_dir",
                    args={"DirectoryPath": str(target_dir)},
                )
                res_dir = await hook_fn(call_dir)
                assert res_dir.allow is False
                assert res_dir.modified_args is None
                assert call_dir.args == {"DirectoryPath": str(target_dir)}
                return {"selected_skill": "agent-platform-deploy"}

        agent.response = _DirRunnerResponse(
            tool_calls=[
                ag_types.ToolCall(name="list_dir", args={"DirectoryPath": str(target_dir)}),
            ],
        )
        return agent

    monkeypatch.setattr("reach.runtime.antigravity_sdk.Agent", agent_with_list_dir_call)
    dir_outcome = runtime.select(
        "Deploy to GKE",
        tmp_path / "work",
        target_skill="agent-platform-deploy",
    )
    assert dir_outcome.invoked_skills == ("agent-platform-deploy",)
    assert dir_outcome.early_exit is True

    from reach.metrics import classification_report
    from reach.models import Catalog, CatalogMode, ProbeResult, Query

    query = Query(
        id="q-1",
        text="deploy my model to an endpoint",
        expected_skill="agent-platform-deploy",
    )
    result = ProbeResult.from_outcome(
        outcome=outcome,
        query=query,
        catalog=Catalog(
            id="all",
            mode=CatalogMode.ALL,
            skills=("agent-platform-endpoint-management", "agent-platform-deploy"),
        ),
        runtime_name=runtime.name,
        model=runtime.model,
    )
    report = classification_report([result], [query])
    assert report.entrypoint_hits == 0
    assert report.trajectory_hits == 1


def test_suppress_retryable_step_warnings_filters_503_unless_debug() -> None:
    """Verify _suppress_retryable_step_warnings filters 503 and 429 root warnings unless DEBUG."""
    import logging

    from reach.runtime.antigravity_sdk import _suppress_retryable_step_warnings

    root_logger = logging.getLogger()
    orig_level = root_logger.level
    records: list[logging.LogRecord] = []

    class _ListHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = _ListHandler()
    root_logger.addHandler(handler)
    try:
        root_logger.setLevel(logging.WARNING)
        with _suppress_retryable_step_warnings():
            root_logger.warning(
                "System step error (HTTP 503): Encountered retryable error from model provider."
            )
            root_logger.warning("System step error (HTTP 503): Service Unavailable")
            root_logger.warning("System step error (HTTP 429): Too Many Requests")
            root_logger.warning("Unrelated warning message")
        assert [r.getMessage() for r in records] == ["Unrelated warning message"]

        records.clear()
        root_logger.setLevel(logging.DEBUG)
        with _suppress_retryable_step_warnings():
            root_logger.warning(
                "System step error (HTTP 503): Encountered retryable error from model provider."
            )
        assert len(records) == 1
        assert "HTTP 503" in records[0].getMessage()
    finally:
        root_logger.removeHandler(handler)
        root_logger.setLevel(orig_level)
