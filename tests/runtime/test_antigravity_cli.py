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

"""Verify antigravity-cli agent behavior, stream parsing, isolation, and CLI arguments."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from pydantic import ValidationError

if TYPE_CHECKING:
    pass

from reach.catalog import build_catalogs, load_skills
from reach.config import DEFAULT_GEMINI_MODEL, agent_default_model
from reach.models import CatalogMode
from reach.runtime.antigravity_cli import (
    DENIED_PERMISSION_ACTIONS,
    AntigravityCliGenerator,
    AntigravityCliOptions,
    AntigravityCliRuntime,
    ToolAttempt,
    _AgyResultEvent,
    _conversation_state_paths,
    _ensure_isolated_settings,
    _extract_result_event,
    _isolated_settings_path,
    _leaked_tools,
    parse_stream,
)

from .conftest import (
    agy_stream,
    canned,
    install_one,
    read_isolated_settings,
)


@pytest.fixture
def runtime(home_dir: Path) -> AntigravityCliRuntime:
    """Provide an AntigravityCliRuntime configured with isolated home directory."""
    return AntigravityCliRuntime(
        options=AntigravityCliOptions(model="test-model", home_dir=home_dir),
    )


@pytest.fixture
def generator(home_dir: Path) -> AntigravityCliGenerator:
    """Provide an AntigravityCliGenerator configured with isolated home directory."""
    return AntigravityCliGenerator(
        options=AntigravityCliOptions(model="test-model", home_dir=home_dir),
    )


def test_antigravity_cli_options_defaults() -> None:
    """Verify AntigravityCliOptions defaults use_symlinks to False and supports overriding."""
    opts = AntigravityCliOptions()
    assert opts.use_symlinks is False
    assert opts.executable == "agy"
    assert opts.dangerously_skip_permissions is True

    override = AntigravityCliOptions(use_symlinks=True)
    assert override.use_symlinks is True


def test_parses_the_resolved_model_off_the_init_event() -> None:
    """Verify parse_stream extracts resolved model name from init event payload."""
    summary = parse_stream(agy_stream(model="gemini-3.7-flash"))
    assert summary.resolved_model == "gemini-3.7-flash"


@pytest.mark.parametrize(
    "model",
    [None, "", 5],
    ids=["absent", "empty", "not-a-string"],
)
def test_a_stream_naming_no_model_leaves_it_unrecorded(model) -> None:
    """Verify empty or missing model fields produce empty string in parse_stream summary."""
    assert parse_stream(agy_stream(model=model)).resolved_model == ""


def test_invoked_skill_is_read_from_the_structured_output() -> None:
    """Verify selected_skill from structured_output is parsed as invoked_skill."""
    summary = parse_stream(agy_stream(invoked="a"))
    assert summary.invoked_skill == "a"


def test_abstention_is_distinct_from_no_structured_output_at_all() -> None:
    """Verify explicit null in selected_skill sets invoked_skill to None."""
    assert parse_stream(agy_stream(invoked=None)).invoked_skill is None


def test_a_result_with_no_structured_output_reports_no_selection() -> None:
    """Verify missing structured_output field defaults invoked_skill to None."""
    summary = parse_stream(agy_stream(include_structured=False))
    assert summary.invoked_skill is None
    assert summary.saw_result is True


def test_duration_is_converted_from_seconds_to_milliseconds() -> None:
    """Verify duration_seconds is converted to milliseconds in summary output."""
    assert parse_stream(agy_stream(duration_seconds=2.5)).duration_ms == 2500


def test_prompt_tokens_parsed_from_result_usage() -> None:
    """Verify parse_stream extracts prompt tokens from result usage payload."""
    summary = parse_stream(agy_stream(prompt_tokens=15420))
    assert summary.prompt_tokens == 15420


def test_prompt_tokens_parsed_from_step_update_usage() -> None:
    """Verify parse_stream extracts prompt tokens from step_update usage when result is missing."""
    lines = [
        json.dumps({"event": "init", "init": {"model": "gemini-3.8-flash"}}),
        json.dumps(
            {
                "event": "step_update",
                "step_update": {
                    "step_type": "agent_response",
                    "usage": {"input_tokens": 8400, "output_tokens": 12},
                },
            }
        ),
    ]
    summary = parse_stream(lines)
    assert summary.prompt_tokens == 8400


@pytest.mark.parametrize(
    "usage",
    [None, {}, {"input_tokens": None}, {"input_tokens": -5}, {"input_tokens": "many"}],
    ids=["none", "empty", "null-tokens", "negative-tokens", "non-integer"],
)
def test_prompt_tokens_invalid_or_missing_defaults_to_none(usage: dict[str, Any] | None) -> None:
    """Verify missing, non-integer, or negative token values gracefully evaluate to None."""
    summary = parse_stream(agy_stream(usage=usage))
    assert summary.prompt_tokens is None


@pytest.mark.parametrize(
    ("event", "expected"),
    [
        (
            {
                "event": "result",
                "result": {
                    "status": "success",
                    "duration_seconds": 1.25,
                    "structured_output": {
                        "selected_skill": "skill-a",
                        "reasoning": "  selected based on task  ",
                    },
                    "error": "  warning logged  ",
                    "usage": {"input_tokens": 1500},
                    "extra_unmodeled_field": 42,
                },
            },
            {
                "status": "success",
                "duration_ms": 1250,
                "selected_skill": "skill-a",
                "reasoning": "selected based on task",
                "error": "warning logged",
                "prompt_tokens": 1500,
            },
        ),
        (
            {"event": "result", "result": {}},
            {
                "status": "unknown",
                "duration_ms": None,
                "selected_skill": None,
                "reasoning": None,
                "error": None,
                "prompt_tokens": None,
            },
        ),
        (
            {
                "event": "result",
                "result": {
                    "status": "error",
                    "duration_seconds": -1.0,
                    "structured_output": {"selected_skill": "", "reasoning": "   "},
                    "error": "   ",
                    "usage": {"input_tokens": -10},
                },
            },
            {
                "status": "error",
                "duration_ms": None,
                "selected_skill": None,
                "reasoning": None,
                "error": None,
                "prompt_tokens": None,
            },
        ),
    ],
    ids=["full-event", "empty-event", "whitespace-and-negative-fallback"],
)
def test_extract_result_event_produces_validated_model(
    event: dict[str, Any],
    expected: dict[str, Any],
) -> None:
    """Verify _extract_result_event constructs a validated _AgyResultEvent model."""
    res = _extract_result_event(event)
    assert isinstance(res, _AgyResultEvent)
    assert res.status == expected["status"]
    assert res.duration_ms == expected["duration_ms"]
    assert res.selected_skill == expected["selected_skill"]
    assert res.reasoning == expected["reasoning"]
    assert res.error == expected["error"]
    assert res.prompt_tokens == expected["prompt_tokens"]


def test_agy_result_event_model_invariants() -> None:
    """Verify _AgyResultEvent enforces immutability, extra='ignore', and field constraints."""
    event = _AgyResultEvent.model_validate(
        {
            "status": "done",
            "duration_ms": 500,
            "prompt_tokens": 100,
            "unmodeled_key": "ignored",
        }
    )
    assert event.status == "done"
    assert not hasattr(event, "unmodeled_key")

    with pytest.raises(ValidationError):
        _AgyResultEvent(duration_ms=-1)  # type: ignore[arg-type]

    with pytest.raises(ValidationError):
        _AgyResultEvent(prompt_tokens=-10)  # type: ignore[arg-type]

    with pytest.raises(ValidationError):
        setattr(event, "status", "mutated")  # noqa: B010


def test_a_view_file_call_records_its_target_path() -> None:
    """Verify view_file tool calls record absolute target file paths."""
    summary = parse_stream(
        agy_stream(tools=[("view_file", "/work/.agents/skills/a/SKILL.md")]),
    )
    assert summary.tool_attempts == (
        ToolAttempt(name="view_file", path="/work/.agents/skills/a/SKILL.md"),
    )
    assert summary.observed_tools == ("view_file",)


def test_a_non_view_file_tool_carries_no_path() -> None:
    """Verify non-file tools record None for path in tool attempt objects."""
    summary = parse_stream(agy_stream(tools=[("search_web", None)]))
    assert summary.tool_attempts == (ToolAttempt(name="search_web", path=None),)


def test_repeated_identical_attempts_are_recorded_once() -> None:
    """Verify consecutive duplicate tool attempts are deduplicated."""
    summary = parse_stream(
        agy_stream(tools=[("view_file", "/a/SKILL.md"), ("view_file", "/a/SKILL.md")]),
    )
    assert summary.tool_attempts == (ToolAttempt(name="view_file", path="/a/SKILL.md"),)


@pytest.mark.parametrize(
    "line",
    ["", "   ", "not json at all", "[1, 2, 3]", '{"event":"step_update"}'],
    ids=["empty", "whitespace", "garbage", "json-array", "missing-step_update"],
)
def test_malformed_lines_are_skipped(line) -> None:
    """Verify malformed JSON or invalid stream lines are ignored during parsing."""
    lines = agy_stream(invoked="a")
    lines.insert(1, line)
    assert parse_stream(lines).invoked_skill == "a"


def test_status_is_reported_verbatim() -> None:
    """Verify result event status string is preserved in parse summary."""
    assert parse_stream(agy_stream(status="ERROR")).status == "ERROR"


def test_step_update_reasoning_and_thoughts_are_collected() -> None:
    """Verify thought and reasoning fields from step_update events are extracted."""
    lines = [
        json.dumps(
            {
                "event": "step_update",
                "step_update": {
                    "step_type": "thought",
                    "thought": "Considering which skill best matches the user's intent.",
                },
            },
        ),
        json.dumps(
            {
                "event": "result",
                "result": {
                    "status": "COMPLETED",
                    "structured_output": {"selected_skill": "skill-a"},
                },
            },
        ),
    ]
    summary = parse_stream(lines)
    assert summary.invoked_skill == "skill-a"
    assert summary.reasoning == ("Considering which skill best matches the user's intent.",)


def test_step_update_multi_turn_thoughts_are_collected_in_order() -> None:
    """Verify multiple step_update thoughts are preserved sequentially."""
    lines = [
        json.dumps(
            {
                "event": "step_update",
                "step_update": {"thought": "First step thinking."},
            },
        ),
        json.dumps(
            {
                "event": "step_update",
                "step_update": {"thought": "Second step thinking."},
            },
        ),
        json.dumps(
            {
                "event": "result",
                "result": {
                    "status": "COMPLETED",
                    "structured_output": {"selected_skill": "skill-a"},
                },
            },
        ),
    ]
    summary = parse_stream(lines)
    assert summary.reasoning == ("First step thinking.", "Second step thinking.")


def test_parse_stream_extracts_reasoning_from_result_structured_output() -> None:
    """Verify reasoning string from result.structured_output is collected."""
    lines = agy_stream(invoked="skill-a", reasoning="Target directly addresses the query.")
    summary = parse_stream(lines)
    assert summary.invoked_skill == "skill-a"
    assert summary.reasoning == ("Target directly addresses the query.",)


def test_parse_stream_extracts_agent_response_text_delta() -> None:
    """Verify non-JSON intermediate text deltas in agent_response steps are collected."""
    lines = [
        json.dumps(
            {
                "event": "step_update",
                "step_update": {
                    "step_type": "agent_response",
                    "state": "DONE",
                    "text_delta": "Checking available skills in directory...",
                },
            },
        ),
        json.dumps(
            {
                "event": "result",
                "result": {
                    "status": "SUCCESS",
                    "structured_output": {
                        "selected_skill": "skill-a",
                        "reasoning": "Matched based on description.",
                    },
                },
            },
        ),
    ]
    summary = parse_stream(lines)
    assert summary.reasoning == (
        "Checking available skills in directory...",
        "Matched based on description.",
    )


def test_finish_always_survives() -> None:
    """Verify finish tool invocations are not flagged as tool leaks."""
    attempts = [ToolAttempt(name="finish", path=None)]
    assert _leaked_tools(attempts, frozenset()) == ()


def test_a_view_file_on_a_resident_skill_survives() -> None:
    """Verify view_file on resident skill files is not flagged as a tool leak."""
    resident = frozenset({"/work/.agents/skills/a/SKILL.md"})
    attempts = [ToolAttempt(name="view_file", path="/work/.agents/skills/a/SKILL.md")]
    assert _leaked_tools(attempts, resident) == ()


def test_a_view_file_through_a_symlinked_workdir_survives(tmp_path) -> None:
    """Verify view_file through symlinked paths resolves target correctly against resident set."""
    real = tmp_path / "real"
    real.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(real)
    resident = frozenset({str((real / "SKILL.md").resolve())})
    attempts = [ToolAttempt(name="view_file", path=str(linked / "SKILL.md"))]
    assert _leaked_tools(attempts, resident) == ()


def test_a_view_file_elsewhere_is_a_leak() -> None:
    """Verify view_file targeting files outside resident skill paths is flagged as leak."""
    resident = frozenset({"/work/.agents/skills/a/SKILL.md"})
    attempts = [ToolAttempt(name="view_file", path="/etc/passwd")]
    assert _leaked_tools(attempts, resident, allowed_tools=("view_file",)) == ("view_file",)


def test_any_other_tool_is_a_leak() -> None:
    """Verify unapproved tool invocations are flagged when allowed_tools is specified."""
    attempts = [ToolAttempt(name="unapproved_custom_tool", path=None)]
    assert _leaked_tools(
        attempts,
        frozenset(),
        allowed_tools=("finish",),
    ) == ("unapproved_custom_tool",)


def test_tools_bypass_leak_check_when_allowed_tools_is_none() -> None:
    """Verify all tool invocations are permitted when allowed_tools is omitted."""
    attempts = [
        ToolAttempt(name="search_web", path=None),
        ToolAttempt(name="run_command", path=None),
        ToolAttempt(name="unapproved_custom_tool", path=None),
    ]
    assert _leaked_tools(attempts, frozenset(), allowed_tools=None) == ()


def test_a_clean_probe_reports_no_leak() -> None:
    """Verify clean tool execution sequences report empty leak tuple."""
    resident = frozenset({"/work/.agents/skills/a/SKILL.md"})
    attempts = [
        ToolAttempt(name="view_file", path="/work/.agents/skills/a/SKILL.md"),
        ToolAttempt(name="finish", path=None),
    ]
    assert _leaked_tools(attempts, resident) == ()


def test_a_view_file_on_a_resident_skill_subpath_survives() -> None:
    """Verify view_file on bundled reference files within resident skill survives."""
    resident = frozenset({"/work/.agents/skills/a/SKILL.md"})
    attempts = [
        ToolAttempt(
            name="view_file",
            path="/work/.agents/skills/a/references/guide.md",
        ),
    ]
    assert _leaked_tools(attempts, resident) == ()


def test_inspection_tools_within_workdir_survive() -> None:
    """Verify inspection tools targeting paths within workdir are permitted."""
    attempts = [
        ToolAttempt(name="list_dir", path="/work/.agents/skills"),
        ToolAttempt(name="find_by_name", path="/work"),
        ToolAttempt(name="grep_search", path="/work/.agents/skills/a"),
    ]
    assert _leaked_tools(attempts, (), workdir="/work") == ()


def test_inspection_tools_targeting_outside_workdir_are_leaks() -> None:
    """Verify inspection tools targeting directories outside workdir are flagged as leaks."""
    attempts = [
        ToolAttempt(name="list_dir", path="/etc"),
        ToolAttempt(name="grep_search", path="/var/other"),
    ]
    assert _leaked_tools(
        attempts,
        (),
        workdir="/work",
        allowed_tools=("list_dir", "grep_search"),
    ) == ("grep_search", "list_dir")


def test_inspection_tools_without_workdir_are_leaks() -> None:
    """Verify inspection tools without active workdir context are flagged as leaks."""
    attempts = [ToolAttempt(name="list_dir", path="/work/.agents/skills")]
    assert _leaked_tools(attempts, (), workdir=None, allowed_tools=("list_dir",)) == ("list_dir",)


def test_documentation_tools_survive() -> None:
    """Verify documentation reading tools are permitted."""
    attempts = [
        ToolAttempt(name="read_url_content", path="https://docs.cloud.google.com/foo"),
        ToolAttempt(name="read_url", path="https://cloud.google.com/bar"),
    ]
    assert _leaked_tools(attempts, ()) == ()


def test_execution_tools_survive() -> None:
    """Verify execution and artifact tools used during skill workflows are permitted."""
    attempts = [
        ToolAttempt(name="run_command", path=None),
        ToolAttempt(name="write_to_file", path=None),
    ]
    assert _leaked_tools(attempts, ()) == ()


def test_search_web_survives() -> None:
    """Verify search_web is permitted by default allowed tools."""
    attempts = [ToolAttempt(name="search_web", path=None)]
    assert _leaked_tools(attempts, ()) == ()


def test_custom_allowed_tools_filters_tools() -> None:
    """Verify custom allowed_tools setting filters out tools not included in the set."""
    attempts = [ToolAttempt(name="search_web", path=None)]
    assert _leaked_tools(
        attempts,
        (),
        allowed_tools=("finish", "view_file"),
    ) == ("search_web",)


def test_model_has_default() -> None:
    """Verify model parameter defaults to configured agent default model."""
    default = agent_default_model("antigravity-cli")
    assert default is not None
    assert AntigravityCliOptions().model == default


def test_home_dir_defaults_to_none() -> None:
    """Verify home_dir parameter defaults to None before runtime initialization."""
    assert AntigravityCliOptions().home_dir is None


def test_home_dir_must_not_be_the_operators_real_home(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify ValidationError is raised if home_dir resolves to user's real home directory."""
    monkeypatch.setattr(Path, "home", classmethod(lambda _cls: tmp_path))
    with pytest.raises(ValidationError, match="home_dir"):
        AntigravityCliOptions.model_validate(
            {"model": "test-model", "home_dir": str(tmp_path)},
        )


def test_home_dir_must_not_be_root() -> None:
    """Verify ValidationError is raised if home_dir resolves to root directory."""
    with pytest.raises(ValidationError, match="home_dir"):
        AntigravityCliOptions.model_validate(
            {"model": "test-model", "home_dir": "/"},
        )


def test_model_provider_accepts_configured_provider(home_dir: Path) -> None:
    """Verify model_provider and provider fields validate successfully."""
    opts = AntigravityCliOptions.model_validate(
        {
            "model": "claude-sonnet-4.6",
            "home_dir": str(home_dir),
            "model_provider": "anthropic",
        },
    )
    assert opts.model == "claude-sonnet-4.6"
    assert opts.model_provider == "anthropic"


def test_model_provider_gemini_accepts_a_gemini_model(home_dir: Path) -> None:
    """Verify valid gemini model validation succeeds under gemini model provider."""
    options = AntigravityCliOptions.model_validate(
        {
            "model": DEFAULT_GEMINI_MODEL,
            "home_dir": str(home_dir),
            "model_provider": "gemini",
        },
    )
    assert options.model == DEFAULT_GEMINI_MODEL


def test_the_agent_reports_the_configured_model(runtime: AntigravityCliRuntime) -> None:
    """Verify runtime.model returns the configured model identifier."""
    assert runtime.model == "test-model"


def test_constructing_the_agent_writes_the_closed_permission_policy(
    home_dir: Path,
) -> None:
    """Verify agent initialization creates isolated settings with default deny permissions."""
    AntigravityCliRuntime(options=AntigravityCliOptions(model="m", home_dir=home_dir))
    settings = read_isolated_settings(home_dir)
    assert settings["permissions"]["deny"] == list(DENIED_PERMISSION_ACTIONS)
    assert "experimental" not in settings


def test_configuring_model_provider_writes_it_at_construction(home_dir: Path) -> None:
    """Verify modelProvider configuration is written to isolated settings at init."""
    AntigravityCliRuntime(
        options=AntigravityCliOptions(
            model=DEFAULT_GEMINI_MODEL,
            home_dir=home_dir,
            model_provider="gemini",
        ),
    )
    settings = read_isolated_settings(home_dir)
    assert settings["modelProvider"] == "gemini"


def test_model_provider_defaults_to_unset(home_dir: Path) -> None:
    """Verify modelProvider key is omitted from settings when not explicitly configured."""
    AntigravityCliRuntime(options=AntigravityCliOptions(model="m", home_dir=home_dir))
    settings = read_isolated_settings(home_dir)
    assert "modelProvider" not in settings


@pytest.mark.parametrize("target_cls", [AntigravityCliRuntime, AntigravityCliGenerator])
@pytest.mark.parametrize(
    ("env_vars", "model", "expected_provider"),
    [
        ({"GEMINI_API_KEY": "test-key-123"}, DEFAULT_GEMINI_MODEL, "gemini"),
        ({"GOOGLE_API_KEY": "test-key-456"}, DEFAULT_GEMINI_MODEL, "gemini"),
        ({"GEMINI_API_KEY": "test-key-123"}, "claude-3-opus", None),
        ({}, DEFAULT_GEMINI_MODEL, None),
    ],
    ids=["gemini-key", "google-key", "non-gemini-model", "no-keys"],
)
def test_effective_model_provider_configures_settings(
    home_dir: Path,
    clean_api_keys: None,
    monkeypatch: pytest.MonkeyPatch,
    target_cls: type[AntigravityCliRuntime | AntigravityCliGenerator],
    env_vars: dict[str, str],
    model: str,
    expected_provider: str | None,
) -> None:
    """Verify effective_model_provider detects keys and writes modelProvider to settings."""
    for key, val in env_vars.items():
        monkeypatch.setenv(key, val)
    instance = target_cls(
        options=AntigravityCliOptions(model=model, home_dir=home_dir),
    )
    assert instance.effective_model_provider == expected_provider
    settings = read_isolated_settings(home_dir)
    if expected_provider is not None:
        assert settings.get("modelProvider") == expected_provider
    else:
        assert "modelProvider" not in settings


def test_read_file_is_never_in_the_deny_list() -> None:
    """Verify read_file actions are not included in wholesale deny list."""
    assert "read_file(*)" not in DENIED_PERMISSION_ACTIONS
    assert not any(action.startswith("read_file") for action in DENIED_PERMISSION_ACTIONS)


def test_isolated_settings_preserve_unrelated_keys(home_dir: Path) -> None:
    """Verify _ensure_isolated_settings preserves pre-existing settings keys."""
    path = _isolated_settings_path(home_dir)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"security": {"auth": {"selectedType": "vertex-ai"}}}))
    _ensure_isolated_settings(home_dir)
    settings = read_isolated_settings(home_dir)
    assert settings["security"] == {"auth": {"selectedType": "vertex-ai"}}
    assert settings["permissions"]["deny"] == list(DENIED_PERMISSION_ACTIONS)


def test_a_stray_allow_rule_is_replaced_not_merged(home_dir: Path) -> None:
    """Verify pre-existing allow rules are replaced rather than merged during settings isolation."""
    path = _isolated_settings_path(home_dir)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"permissions": {"allow": ["command(rm)"]}}))
    _ensure_isolated_settings(home_dir)
    settings = read_isolated_settings(home_dir)
    assert "allow" not in settings["permissions"]


def test_allow_read_scopes_a_read_file_grant_to_one_directory(
    home_dir: Path,
    tmp_path: Path,
) -> None:
    """Verify allow_read grants read_file permission for specified directory path."""
    skills_dir = tmp_path / "work" / ".agents" / "skills"
    _ensure_isolated_settings(home_dir, allow_read=skills_dir)
    settings = read_isolated_settings(home_dir)
    assert settings["permissions"]["allow"] == [f"read_file({skills_dir.resolve()})"]


def test_model_provider_is_written_when_given(home_dir: Path) -> None:
    """Verify modelProvider is written to settings when provided."""
    _ensure_isolated_settings(home_dir, model_provider="gemini")
    settings = read_isolated_settings(home_dir)
    assert settings["modelProvider"] == "gemini"


def test_model_provider_is_cleared_when_not_given(home_dir: Path) -> None:
    """Verify modelProvider is removed from settings when omitted in update."""
    _ensure_isolated_settings(home_dir, model_provider="gemini")
    _ensure_isolated_settings(home_dir)
    settings = read_isolated_settings(home_dir)
    assert "modelProvider" not in settings


def test_trust_replaces_rather_than_accumulates(home_dir: Path, tmp_path: Path) -> None:
    """Verify trustedWorkspaces replaces previous workspace paths rather than accumulating."""
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    _ensure_isolated_settings(home_dir, trust=first)
    _ensure_isolated_settings(home_dir, trust=second)
    settings = read_isolated_settings(home_dir)
    assert settings["trustedWorkspaces"] == [str(second.resolve())]


# --- Catalog installation and permission scoping -----------------------------


def test_install_trusts_and_scopes_read_access_to_the_workdir(
    runtime: AntigravityCliRuntime,
    skill_repo: Path,
    tmp_path: Path,
    home_dir: Path,
) -> None:
    """Verify install grants read_file permission for skills and trusts workdir."""
    skills = load_skills(skill_repo)
    catalog = build_catalogs(skills, CatalogMode.SINGLETON)[0]
    workdir = tmp_path / "work"
    target = runtime.install(catalog, skills, workdir)
    settings = read_isolated_settings(home_dir)
    assert settings["permissions"]["allow"] == [
        f"read_file({runtime.skills_dir(target)})",
    ]
    assert str(target) in settings["trustedWorkspaces"]


def test_install_does_not_clear_a_configured_model_provider(
    skill_repo: Path,
    tmp_path: Path,
    home_dir: Path,
) -> None:
    """Verify install retains configured modelProvider in isolated settings."""
    runtime = AntigravityCliRuntime(
        options=AntigravityCliOptions(
            model=DEFAULT_GEMINI_MODEL,
            home_dir=home_dir,
            model_provider="gemini",
        ),
    )
    skills = load_skills(skill_repo)
    catalog = build_catalogs(skills, CatalogMode.SINGLETON)[0]
    runtime.install(catalog, skills, tmp_path / "work")
    settings = read_isolated_settings(home_dir)
    assert settings["modelProvider"] == "gemini"


def test_command_names_the_query_the_model_and_the_stream_format(
    runtime: AntigravityCliRuntime,
) -> None:
    """Verify build_command generates expected command flags and arguments."""
    command = runtime.build_command("do a thing")
    assert command[:3] == ["agy", "-p", "do a thing"]
    assert command[command.index("--model") + 1] == "test-model"
    assert command[command.index("--output-format") + 1] == "stream-json"


def test_command_always_asks_for_a_new_project(runtime: AntigravityCliRuntime) -> None:
    """Verify build_command always includes --new-project flag."""
    assert "--new-project" in runtime.build_command("q")


def test_command_disables_slash_command_expansion_by_default(
    runtime: AntigravityCliRuntime,
) -> None:
    """Verify build_command includes --disable-slash-commands by default."""
    assert "--disable-slash-commands" in runtime.build_command("q")


def test_command_allows_slash_commands_when_disabled_in_options(home_dir: Path) -> None:
    """Verify build_command omits --disable-slash-commands when disabled in options."""
    runtime = AntigravityCliRuntime(
        options=AntigravityCliOptions(home_dir=home_dir, disable_slash_commands=False),
    )
    assert "--disable-slash-commands" not in runtime.build_command("q")


def test_command_includes_dangerously_skip_permissions_by_default(
    runtime: AntigravityCliRuntime,
) -> None:
    """Verify build_command includes --dangerously-skip-permissions by default."""
    assert "--dangerously-skip-permissions" in runtime.build_command("q")


def test_command_omits_dangerously_skip_permissions_when_false(home_dir: Path) -> None:
    """Verify build_command omits --dangerously-skip-permissions when disabled."""
    runtime = AntigravityCliRuntime(
        options=AntigravityCliOptions(home_dir=home_dir, dangerously_skip_permissions=False),
    )
    assert "--dangerously-skip-permissions" not in runtime.build_command("q")


def test_command_includes_print_timeout(runtime: AntigravityCliRuntime) -> None:
    """Verify build_command includes --print-timeout matching settings timeout."""
    assert runtime.settings is not None
    cmd = runtime.build_command("q")
    assert cmd[cmd.index("--print-timeout") + 1] == f"{int(runtime.settings.timeout_s)}s"


def test_command_uses_explicit_print_timeout(home_dir: Path) -> None:
    """Verify build_command uses explicit print_timeout when configured."""
    runtime = AntigravityCliRuntime(
        options=AntigravityCliOptions(home_dir=home_dir, print_timeout="45s"),
    )
    cmd = runtime.build_command("q")
    assert cmd[cmd.index("--print-timeout") + 1] == "45s"


def test_command_omits_schema_by_default_even_with_resident_skills(
    runtime: AntigravityCliRuntime,
) -> None:
    """Verify build_command omits --json-schema by default to preserve organic selection."""
    runtime._resident = ("a", "b")
    command = runtime.build_command("q")
    assert "--json-schema" not in command


def test_command_includes_explicit_json_schema_when_configured(
    home_dir: Path,
) -> None:
    """Verify build_command passes --json-schema when explicitly configured in options."""
    explicit_schema = AntigravityCliRuntime.selection_json_schema(("a", "b"))
    runtime = AntigravityCliRuntime(
        options=AntigravityCliOptions(home_dir=home_dir, json_schema=explicit_schema),
    )
    runtime._resident = ("a", "b")
    command = runtime.build_command("q")
    schema = json.loads(command[command.index("--json-schema") + 1])
    branches = schema["properties"]["selected_skill"]["anyOf"]
    enum = next(b["enum"] for b in branches if "enum" in b)
    assert sorted(enum) == ["a", "b"]


def test_effort_is_omitted_when_unset(runtime: AntigravityCliRuntime) -> None:
    """Verify --effort flag is omitted when effort option is not configured."""
    assert "--effort" not in runtime.build_command("q")


def test_effort_reaches_the_command_line_when_set(home_dir: Path) -> None:
    """Verify --effort flag is included in build_command when set."""
    runtime = AntigravityCliRuntime(
        options=AntigravityCliOptions(model="m", home_dir=home_dir, effort="high"),
    )
    command = runtime.build_command("q")
    assert command[command.index("--effort") + 1] == "high"


def test_effort_allows_custom_model_and_effort(home_dir: Path) -> None:
    """Verify custom model and effort tiers pass through without validation restrictions."""
    options = AntigravityCliOptions.model_validate(
        {
            "model": "gemini-3.8-flash",
            "home_dir": str(home_dir),
            "effort": "custom_tier",
        },
    )
    assert options.model == "gemini-3.8-flash"
    assert options.effort == "custom_tier"


def test_effort_allows_a_bare_model_family_name(home_dir: Path) -> None:
    """Verify effort configuration validates successfully on model slugs without embedded tiers."""
    options = AntigravityCliOptions.model_validate(
        {
            "model": "gemini-3.7-flash",
            "home_dir": str(home_dir),
            "effort": "high",
        },
    )
    assert options.effort == "high"


def test_select_accepts_a_clean_probe(
    monkeypatch,
    runtime: AntigravityCliRuntime,
    skill_repo: Path,
    tmp_path: Path,
) -> None:
    """Verify successful probe execution with valid resident tool calls and top1 selection."""
    workdir = tmp_path / "work"
    resident = install_one(runtime, skill_repo, workdir)
    skill_path = str(runtime.skills_dir(workdir) / resident / "SKILL.md")
    canned(monkeypatch, agy_stream(invoked=resident, tools=[("view_file", skill_path)]))
    outcome = runtime.select("do a thing", workdir)
    assert outcome.error is None
    assert outcome.invoked_skill == resident
    assert outcome.invoked_skills == (resident,)
    assert outcome.observed_catalog == (resident,)


def test_select_populates_reasoning_in_outcome(
    monkeypatch,
    runtime: AntigravityCliRuntime,
    skill_repo: Path,
    tmp_path: Path,
) -> None:
    """Verify runtime.select populates reasoning in SelectionOutcome."""
    workdir = tmp_path / "work"
    resident = install_one(runtime, skill_repo, workdir)
    canned(
        monkeypatch,
        agy_stream(invoked=resident, reasoning="Selected skill matches intent"),
    )
    outcome = runtime.select("do a thing", workdir)
    assert outcome.reasoning == ("Selected skill matches intent",)


def test_select_reports_no_invocations_on_abstention(
    monkeypatch,
    runtime: AntigravityCliRuntime,
    tmp_path: Path,
) -> None:
    """Verify empty tuple and None invoked_skill on explicit abstention."""
    canned(monkeypatch, agy_stream(invoked=None))
    outcome = runtime.select("q", tmp_path)
    assert outcome.invoked_skill is None
    assert outcome.invoked_skills == ()


def test_select_discards_a_probe_that_used_an_ungated_tool(
    monkeypatch,
    runtime: AntigravityCliRuntime,
    skill_repo: Path,
    tmp_path: Path,
) -> None:
    """Verify unapproved tool is marked as a leak when allowed_tools is set."""
    runtime.options = runtime.options.model_copy(update={"allowed_tools": ("finish",)})
    workdir = tmp_path / "work"
    resident = install_one(runtime, skill_repo, workdir)
    canned(monkeypatch, agy_stream(invoked=resident, tools=[("unauthorized_tool", None)]))
    outcome = runtime.select("do a thing", workdir)
    assert outcome.error == "tool leak: unauthorized_tool"


def test_select_allows_all_tools_by_default_when_allowed_tools_omitted(
    monkeypatch,
    runtime: AntigravityCliRuntime,
    skill_repo: Path,
    tmp_path: Path,
) -> None:
    """Verify all tools are permitted by default when allowed_tools is not configured."""
    workdir = tmp_path / "work"
    resident = install_one(runtime, skill_repo, workdir)
    canned(
        monkeypatch,
        agy_stream(
            invoked=resident,
            tools=[("run_command", None), ("search_web", None), ("custom_tool", None)],
        ),
    )
    outcome = runtime.select("do a thing", workdir)
    assert outcome.invoked_skill == resident
    assert outcome.error is None


def test_select_discards_a_view_file_outside_the_resident_catalog(
    monkeypatch,
    runtime: AntigravityCliRuntime,
    skill_repo: Path,
    tmp_path: Path,
) -> None:
    """Verify probe attempting view_file outside resident catalog is marked as leak."""
    runtime.options = runtime.options.model_copy(update={"allowed_tools": ("finish", "view_file")})
    workdir = tmp_path / "work"
    resident = install_one(runtime, skill_repo, workdir)
    canned(
        monkeypatch,
        agy_stream(invoked=resident, tools=[("view_file", "/etc/passwd")]),
    )
    outcome = runtime.select("do a thing", workdir)
    assert outcome.error == "tool leak: view_file"


def test_select_discards_a_leaked_hit_not_just_a_miss(
    monkeypatch,
    runtime: AntigravityCliRuntime,
    skill_repo: Path,
    tmp_path: Path,
) -> None:
    """Verify tool leak error is reported even if target skill was selected."""
    runtime.options = runtime.options.model_copy(update={"allowed_tools": ("finish",)})
    workdir = tmp_path / "work"
    resident = install_one(runtime, skill_repo, workdir)
    canned(monkeypatch, agy_stream(invoked=resident, tools=[("unauthorized_tool", None)]))
    outcome = runtime.select("do a thing", workdir)
    assert outcome.invoked_skill == resident
    assert outcome.error == "tool leak: unauthorized_tool"


def test_select_reports_a_non_success_status_as_an_error(
    monkeypatch,
    runtime: AntigravityCliRuntime,
    tmp_path: Path,
) -> None:
    """Verify non-success event status is translated to runtime error outcome."""
    canned(monkeypatch, agy_stream(status="ERROR"))
    outcome = runtime.select("q", tmp_path)
    assert outcome.error == "runtime error: ERROR"


def test_select_runs_in_the_workspace_it_was_given_with_the_isolated_home(
    monkeypatch,
    runtime: AntigravityCliRuntime,
    tmp_path: Path,
    home_dir: Path,
) -> None:
    """Verify select runs process in workdir cwd with HOME pointing to isolated directory."""
    seen: dict[str, object] = {}

    def record(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        seen.update(kwargs)
        return subprocess.CompletedProcess(
            args=args,
            returncode=0,
            stdout="\n".join(agy_stream(invoked=None)),
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", record)
    runtime.select("q", tmp_path)
    assert seen["cwd"] == tmp_path
    env = seen["env"]
    assert isinstance(env, dict)
    assert env["HOME"] == str(home_dir.resolve())


def test_residency_is_reported_from_install_not_from_the_stream(
    monkeypatch,
    runtime: AntigravityCliRuntime,
    skill_repo: Path,
    tmp_path: Path,
) -> None:
    """Verify observed_catalog reflects skills installed by runtime rather than stream event."""
    workdir = tmp_path / "work"
    resident = install_one(runtime, skill_repo, workdir)
    canned(monkeypatch, agy_stream(invoked=resident))
    assert runtime.select("q", workdir).observed_catalog == (resident,)


def test_cost_is_always_none(
    monkeypatch,
    runtime: AntigravityCliRuntime,
    tmp_path: Path,
) -> None:
    """Verify cost_usd is None for antigravity CLI probes."""
    canned(monkeypatch, agy_stream(invoked=None))
    assert runtime.select("q", tmp_path).cost_usd is None


def _write_conversation_state(home_dir: Path) -> None:
    """Create dummy conversation files in isolated home directory."""
    conversations, summaries_db = _conversation_state_paths(home_dir)
    conversations.mkdir(parents=True)
    (conversations / "one.json").write_text("{}", encoding="utf-8")
    summaries_db.parent.mkdir(parents=True, exist_ok=True)
    summaries_db.write_text("db", encoding="utf-8")


def test_auto_clean_defaults_to_false(home_dir: Path) -> None:
    """Verify auto_clean option defaults to False."""
    options = AntigravityCliOptions(model="m", home_dir=home_dir)
    assert options.auto_clean is False


def test_select_leaves_conversation_state_alone_by_default(
    monkeypatch,
    runtime: AntigravityCliRuntime,
    tmp_path: Path,
) -> None:
    """Verify conversation state files remain intact when auto_clean is False."""
    _write_conversation_state(runtime.home_dir)
    canned(monkeypatch, agy_stream(invoked=None))
    runtime.select("q", tmp_path)
    conversations, summaries_db = _conversation_state_paths(runtime.home_dir)

    assert conversations.is_dir()
    assert summaries_db.exists()


def test_select_prunes_conversation_state_when_auto_clean_is_set(
    monkeypatch,
    home_dir: Path,
    tmp_path: Path,
) -> None:
    """Verify conversation state files are cleaned up after probe when auto_clean is True."""
    runtime = AntigravityCliRuntime(
        options=AntigravityCliOptions(model="m", home_dir=home_dir, auto_clean=True),
    )
    _write_conversation_state(home_dir)
    canned(monkeypatch, agy_stream(invoked=None))
    runtime.select("q", tmp_path)
    conversations, summaries_db = _conversation_state_paths(home_dir)
    assert not conversations.exists()
    assert not summaries_db.exists()


def test_auto_clean_prunes_even_when_the_probe_is_discarded(
    monkeypatch,
    home_dir: Path,
    tmp_path: Path,
) -> None:
    """Verify conversation state is pruned when auto_clean is True even on error outcomes."""
    runtime = AntigravityCliRuntime(
        options=AntigravityCliOptions(model="m", home_dir=home_dir, auto_clean=True),
    )
    _write_conversation_state(home_dir)
    canned(monkeypatch, agy_stream(status="ERROR"))
    outcome = runtime.select("q", tmp_path)
    assert outcome.error is not None
    conversations, summaries_db = _conversation_state_paths(home_dir)
    assert not conversations.exists()
    assert not summaries_db.exists()


def test_clean_conversation_state_is_idempotent(runtime: AntigravityCliRuntime) -> None:
    """Verify clean_conversation_state runs safely when directories do not exist."""
    runtime.clean_conversation_state()
    runtime.clean_conversation_state()


def test_complete_returns_the_response_field(
    monkeypatch,
    generator: AntigravityCliGenerator,
) -> None:
    """Verify complete parses response text field from JSON output."""
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **_kw: subprocess.CompletedProcess(
            args=a,
            returncode=0,
            stdout=json.dumps({"status": "SUCCESS", "response": "a query"}),
            stderr="",
        ),
    )
    assert generator.complete("draft me a query") == "a query"
    assert generator.completions == 1


def test_the_completion_command_still_asks_for_a_new_project(
    generator: AntigravityCliGenerator,
) -> None:
    """Verify build_completion_command includes --new-project flag."""
    assert "--new-project" in generator.build_completion_command()


def test_build_completion_command_omits_prompt_and_passes_effort(
    home_dir: Path,
) -> None:
    """Verify build_completion_command omits prompt from argv and includes effort."""
    generator = AntigravityCliGenerator(
        options=AntigravityCliOptions(
            model=DEFAULT_GEMINI_MODEL,
            effort="low",
            home_dir=home_dir,
        ),
    )
    cmd = generator.build_completion_command("test prompt")
    assert "-p" not in cmd
    assert "test prompt" not in cmd
    assert "--effort" in cmd
    assert cmd[cmd.index("--effort") + 1] == "low"


def test_build_completion_command_resolves_profile_effort(
    home_dir: Path,
) -> None:
    """Verify build_completion_command falls back to model profile effort."""
    generator = AntigravityCliGenerator(
        options=AntigravityCliOptions(
            model=DEFAULT_GEMINI_MODEL,
            home_dir=home_dir,
        ),
    )
    cmd = generator.build_completion_command("test prompt")
    assert "-p" not in cmd
    assert "test prompt" not in cmd
    assert "--effort" in cmd
    assert cmd[cmd.index("--effort") + 1] == "low"


def test_build_completion_command_passes_json_schema_from_parameter(
    generator: AntigravityCliGenerator,
) -> None:
    """Verify build_completion_command includes --json-schema when explicitly passed."""
    cmd = generator.build_completion_command(schema='{"type": "object"}')
    assert "--json-schema" in cmd
    assert cmd[cmd.index("--json-schema") + 1] == '{"type": "object"}'


def test_build_completion_command_passes_json_schema_from_options(
    home_dir: Path,
) -> None:
    """Verify build_completion_command uses json_schema from configured options."""
    generator = AntigravityCliGenerator(
        options=AntigravityCliOptions(
            model=DEFAULT_GEMINI_MODEL,
            home_dir=home_dir,
            json_schema='{"type": "array"}',
        ),
    )
    cmd = generator.build_completion_command()
    assert "--json-schema" in cmd
    assert cmd[cmd.index("--json-schema") + 1] == '{"type": "array"}'


def test_complete_serializes_mapping_schema(
    monkeypatch: pytest.MonkeyPatch,
    generator: AntigravityCliGenerator,
) -> None:
    """Verify complete serializes Mapping schema to JSON string for --json-schema."""
    captured_cmd: list[str] | None = None

    def fake_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        nonlocal captured_cmd
        del kwargs
        captured_cmd = list(args[0])
        return subprocess.CompletedProcess(
            args=args,
            returncode=0,
            stdout=json.dumps({"response": "result"}),
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    generator.complete("test", schema={"type": "object", "properties": {}})
    assert captured_cmd is not None
    assert "--json-schema" in captured_cmd
    idx = captured_cmd.index("--json-schema")
    assert json.loads(captured_cmd[idx + 1]) == {"type": "object", "properties": {}}


def test_resolve_schema_allows_empty_mapping_and_string_override() -> None:
    """Verify explicit empty mapping or string overrides options schema rather than falling back."""
    opts = AntigravityCliOptions(json_schema='{"type": "object"}')
    gen = AntigravityCliGenerator(options=opts)
    assert gen.resolve_schema({}) == "{}"
    assert gen.resolve_schema("") == ""
    assert gen.resolve_schema(None) == '{"type": "object"}'


def test_build_completion_command_allows_empty_schema_override() -> None:
    """Verify build_completion_command respects empty schema override without falling back."""
    opts = AntigravityCliOptions(json_schema='{"type": "object"}')
    gen = AntigravityCliGenerator(options=opts)
    cmd_override = gen.build_completion_command(schema="")
    assert "--json-schema" not in cmd_override
    cmd_default = gen.build_completion_command()
    assert "--json-schema" in cmd_default


def test_complete_pipes_prompt_via_stdin_and_handles_large_payload(
    monkeypatch: pytest.MonkeyPatch,
    generator: AntigravityCliGenerator,
) -> None:
    """Verify complete pipes prompt via stdin and succeeds with payloads exceeding 128 KB."""
    captured_input: str | None = None

    def fake_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        nonlocal captured_input
        captured_input = kwargs.get("input")
        return subprocess.CompletedProcess(
            args=args,
            returncode=0,
            stdout=json.dumps({"response": "completion result"}),
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    large_prompt = "x" * 150_000  # Exceeds Linux 128 KB MAX_ARG_STRLEN
    res = generator.complete(large_prompt)
    assert res == "completion result"
    assert captured_input == large_prompt


def test_complete_raises_when_the_runtime_fails(
    monkeypatch,
    generator: AntigravityCliGenerator,
) -> None:
    """Verify complete raises RuntimeError when subprocess exit code is non-zero."""
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **_kw: subprocess.CompletedProcess(
            args=a,
            returncode=1,
            stdout="",
            stderr="authentication failed",
        ),
    )
    with pytest.raises(RuntimeError, match="authentication failed"):
        generator.complete("draft me a query")


def test_complete_extracts_structured_error_from_stdout_when_stderr_empty(
    monkeypatch: pytest.MonkeyPatch,
    generator: AntigravityCliGenerator,
) -> None:
    """Verify complete parses error field from stdout JSON when stderr is empty."""
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **_kw: subprocess.CompletedProcess(
            args=a,
            returncode=1,
            stdout=json.dumps({"error": "model quota exceeded"}),
            stderr="",
        ),
    )
    with pytest.raises(RuntimeError, match="model quota exceeded"):
        generator.complete("draft me a query")


@pytest.mark.parametrize(
    ("input_model", "expected_model"),
    [
        ("gemini-flash-3.7-medium", "gemini-3.7-flash-medium"),
        ("gemini-flash-3.7-low", "gemini-3.7-flash-low"),
        ("gemini-flash-3.7-high", "gemini-3.7-flash-high"),
        ("gemini-flash-3.7", "gemini-3.7-flash"),
        ("gemini-3.7-flash-medium", "gemini-3.7-flash-medium"),
        ("gemini-3.7-flash", "gemini-3.7-flash"),
    ],
)
def test_normalize_agy_model(input_model: str, expected_model: str) -> None:
    """Verify normalize_agy_model standardizes flash version ordering for agy CLI."""
    from reach.runtime.antigravity_cli import normalize_agy_model

    assert normalize_agy_model(input_model) == expected_model


def test_build_command_emits_effort_flag(home_dir: Path) -> None:
    """Verify build_command passes --effort flag when configured."""
    runtime = AntigravityCliRuntime(
        options=AntigravityCliOptions(
            model=DEFAULT_GEMINI_MODEL,
            effort="medium",
            home_dir=home_dir,
        ),
    )
    cmd = runtime.build_command("test query")
    idx = cmd.index("--effort")
    assert cmd[idx + 1] == "medium"
    assert cmd[cmd.index("--model") + 1] == DEFAULT_GEMINI_MODEL


def test_build_command_omits_effort_flag_when_none(home_dir: Path) -> None:
    """Verify build_command omits --effort flag when effort is None."""
    runtime = AntigravityCliRuntime(
        options=AntigravityCliOptions(
            model="custom-unprofiled-model",
            home_dir=home_dir,
        ),
    )
    cmd = runtime.build_command("test query")
    assert "--effort" not in cmd
    assert cmd[cmd.index("--model") + 1] == "custom-unprofiled-model"


def test_build_command_defaults_to_profile_effort(home_dir: Path) -> None:
    """Verify build_command falls back to profile effort for gemini-3.8-flash."""
    runtime = AntigravityCliRuntime(
        options=AntigravityCliOptions(
            model="gemini-3.8-flash",
            home_dir=home_dir,
        ),
    )
    cmd = runtime.build_command("test query")
    assert "--effort" in cmd
    assert cmd[cmd.index("--effort") + 1] == "low"
    assert cmd[cmd.index("--model") + 1] == "gemini-3.8-flash"


def test_parse_stream_extracts_result_error() -> None:
    """Verify parse_stream extracts error message from error result event."""
    lines = [
        json.dumps(
            {
                "event": "result",
                "result": {
                    "status": "ERROR",
                    "error": "invalid model selection (--model 'unknown-model')",
                },
            },
        ),
    ]
    summary = parse_stream(lines)
    assert summary.status == "ERROR"
    assert summary.error == "invalid model selection (--model 'unknown-model')"


def test_select_reports_stream_error_message(
    monkeypatch,
    runtime: AntigravityCliRuntime,
    tmp_path: Path,
) -> None:
    """Verify select reports native CLI error string from result stream."""
    err_msg = "invalid model selection (--model 'gemini-3.7-flash' --effort 'minimal')"
    lines = [
        json.dumps(
            {
                "event": "result",
                "result": {
                    "status": "ERROR",
                    "error": err_msg,
                },
            },
        ),
    ]
    canned(monkeypatch, lines)
    outcome = runtime.select("do something", tmp_path)
    assert outcome.error == err_msg


def test_antigravity_cli_multi_turn_early_exit_stops_at_target(
    monkeypatch,
    runtime: AntigravityCliRuntime,
    tmp_path: Path,
) -> None:
    """Verify early exit triggers as soon as target skill is invoked."""
    runtime._resident = ("s1", "target-skill", "s3")
    lines = agy_stream(
        tools=[
            ("view_file", ".agents/skills/s1/SKILL.md"),
            ("view_file", ".agents/skills/target-skill/SKILL.md"),
            ("view_file", ".agents/skills/s3/SKILL.md"),
        ],
        status=None,
    )
    canned(monkeypatch, lines)
    outcome = runtime.select("query", tmp_path, target_skill="target-skill")
    assert outcome.early_exit is True
    assert outcome.invoked_skill == "s1"
    assert outcome.invoked_skills == ("s1", "target-skill")
    assert outcome.turns_taken == 2
    assert outcome.error is None


def test_antigravity_cli_multi_turn_budget_exhaustion(
    monkeypatch,
    runtime: AntigravityCliRuntime,
    tmp_path: Path,
) -> None:
    """Verify early exit triggers when maximum turn budget is reached."""
    runtime._resident = ("s1", "s2", "s3", "s4")
    lines = agy_stream(
        tools=[
            ("view_file", ".agents/skills/s1/SKILL.md"),
            ("view_file", ".agents/skills/s2/SKILL.md"),
            ("view_file", ".agents/skills/s3/SKILL.md"),
            ("view_file", ".agents/skills/s4/SKILL.md"),
        ],
        status=None,
    )
    canned(monkeypatch, lines)
    outcome = runtime.select("query", tmp_path, target_skill="other-target")
    assert outcome.early_exit is True
    assert outcome.invoked_skills == ("s1", "s2", "s3")
    assert outcome.turns_taken == 3
    assert outcome.error is None


def test_antigravity_cli_multi_turn_early_exit_suppresses_sigterm_timeout_error(
    monkeypatch,
    runtime: AntigravityCliRuntime,
    tmp_path: Path,
) -> None:
    """Verify early exit suppresses termination error emitted when process is killed."""
    runtime._resident = ("s1", "target-skill", "s3")
    lines = agy_stream(
        tools=[
            ("view_file", ".agents/skills/s1/SKILL.md"),
            ("view_file", ".agents/skills/target-skill/SKILL.md"),
        ],
        status="ERROR",
        error="timeout waiting for response",
    )
    canned(monkeypatch, lines)
    outcome = runtime.select("query", tmp_path, target_skill="target-skill")
    assert outcome.early_exit is True
    assert outcome.invoked_skills == ("s1", "target-skill")
    assert outcome.turns_taken == 2
    assert outcome.error is None


def test_antigravity_cli_abstention_counts_exploratory_turns(
    monkeypatch: pytest.MonkeyPatch,
    runtime: AntigravityCliRuntime,
    tmp_path: Path,
) -> None:
    """Verify non-skill tool exploratory steps report turns_taken accurately on abstention."""
    runtime._resident = ("target-skill",)
    lines = agy_stream(
        tools=[
            ("list_dir", "/workspace"),
            ("view_file", "/workspace/README.md"),
            ("run_command", None),
        ],
        status="SUCCESS",
    )
    canned(monkeypatch, lines)
    outcome = runtime.select("query", tmp_path, target_skill="target-skill")
    assert outcome.invoked_skills == ()
    assert outcome.turns_taken == 3
    assert outcome.early_exit is False


def test_parse_stream_early_exit_suppresses_error_result() -> None:
    """Verify parse_stream clears error and forces SUCCESS status when early_exit is True."""
    lines = agy_stream(
        tools=[
            ("view_file", ".agents/skills/s1/SKILL.md"),
            ("view_file", ".agents/skills/target-skill/SKILL.md"),
        ],
        status="ERROR",
        error="timeout waiting for response",
    )
    summary = parse_stream(lines, resident=["s1", "target-skill"], early_exit=True)
    assert summary.status == "SUCCESS"
    assert summary.error is None


def test_temporary_home_dir_cleanup_on_cleanup_call(clean_api_keys: None) -> None:
    """Verify runtime.cleanup() removes auto-generated temporary home directory."""
    runtime = AntigravityCliRuntime(options=AntigravityCliOptions(model=DEFAULT_GEMINI_MODEL))
    home = runtime.home_dir
    assert home.is_dir()
    runtime.cleanup()
    assert not home.exists()


def test_temporary_home_dir_cleanup_via_context_manager(clean_api_keys: None) -> None:
    """Verify context manager automatically cleans up temporary home directory on exit."""
    opts = AntigravityCliOptions(model=DEFAULT_GEMINI_MODEL)
    with AntigravityCliRuntime(options=opts) as runtime:
        home = runtime.home_dir
        assert home.is_dir()
    assert not home.exists()


def test_explicit_home_dir_is_not_removed_by_cleanup(home_dir: Path) -> None:
    """Verify explicitly provided home directory is preserved during cleanup."""
    runtime = AntigravityCliRuntime(options=AntigravityCliOptions(home_dir=home_dir))
    assert home_dir.is_dir()
    runtime.cleanup()
    assert home_dir.is_dir()


def test_build_env_sets_both_gemini_and_google_api_keys(
    home_dir: Path,
    clean_api_keys: None,
) -> None:
    """Verify build_env synchronizes GEMINI_API_KEY and GOOGLE_API_KEY from effective_api_key."""
    runtime = AntigravityCliRuntime(
        options=AntigravityCliOptions(
            home_dir=home_dir,
            api_key="custom-secret-key",
        ),
    )
    env = runtime.build_env()
    assert env["GEMINI_API_KEY"] == "custom-secret-key"
    assert env["GOOGLE_API_KEY"] == "custom-secret-key"
    assert env["GOMAXPROCS"] == "4"


def test_build_env_custom_gomaxprocs(home_dir: Path) -> None:
    """Verify build_env sets custom GOMAXPROCS when configured in options."""
    runtime = AntigravityCliRuntime(
        options=AntigravityCliOptions(home_dir=home_dir, go_max_procs=2),
    )
    env = runtime.build_env()
    assert env["GOMAXPROCS"] == "2"


def test_extract_skill_from_line_extracts_result_structured_output(home_dir: Path) -> None:
    """Verify extract_skill_from_line extracts selected_skill from result event."""
    runtime = AntigravityCliRuntime(options=AntigravityCliOptions(home_dir=home_dir))
    runtime._resident = ("my-skill", "other-skill")
    result_line = json.dumps(
        {
            "event": "result",
            "result": {
                "status": "success",
                "structured_output": {"selected_skill": "my-skill", "reasoning": "best match"},
            },
        }
    )
    assert runtime.extract_skill_from_line(result_line) == "my-skill"


def test_extract_skill_from_line_ignores_non_resident_result(home_dir: Path) -> None:
    """Verify extract_skill_from_line returns None when result skill is not resident."""
    runtime = AntigravityCliRuntime(options=AntigravityCliOptions(home_dir=home_dir))
    runtime._resident = ("allowed-skill",)
    result_line = json.dumps(
        {
            "event": "result",
            "result": {
                "status": "success",
                "structured_output": {"selected_skill": "hallucinated-skill"},
            },
        }
    )
    assert runtime.extract_skill_from_line(result_line) is None


def test_antigravity_cli_generator_configures_isolated_settings(home_dir: Path) -> None:
    """Verify AntigravityCliGenerator initializes isolated settings.json with denied permissions."""
    AntigravityCliGenerator(options=AntigravityCliOptions(home_dir=home_dir))
    settings = read_isolated_settings(home_dir)
    assert "permissions" in settings
    assert set(settings["permissions"]["deny"]) == set(DENIED_PERMISSION_ACTIONS)


def test_antigravity_cli_generator_build_env_syncs_keys_and_procs(home_dir: Path) -> None:
    """Verify AntigravityCliGenerator build_env sets home, synced api keys, and GOMAXPROCS."""
    generator = AntigravityCliGenerator(
        options=AntigravityCliOptions(
            home_dir=home_dir,
            api_key="custom-api-key",
            go_max_procs=6,
        ),
    )
    env = generator.build_env()
    assert env["HOME"] == str(home_dir)
    assert env["GEMINI_API_KEY"] == "custom-api-key"
    assert env["GOOGLE_API_KEY"] == "custom-api-key"
    assert env["GOMAXPROCS"] == "6"


def test_antigravity_cli_generator_cleans_owned_home_dir() -> None:
    """Verify AntigravityCliGenerator cleans up temporary home directory upon teardown."""
    gen = AntigravityCliGenerator()
    created_home = gen.home_dir
    assert created_home.is_dir()
    gen.cleanup()
    assert not created_home.exists()


@pytest.mark.parametrize("target_cls", [AntigravityCliRuntime, AntigravityCliGenerator])
def test_antigravity_cli_registers_atexit_for_temp_home(
    monkeypatch: pytest.MonkeyPatch,
    target_cls: type[AntigravityCliRuntime | AntigravityCliGenerator],
) -> None:
    """Verify temporary home directory registers atexit cleanup upon creation."""
    registered: list[object] = []
    monkeypatch.setattr("atexit.register", registered.append)
    instance = target_cls()
    try:
        assert instance.cleanup in registered
    finally:
        instance.cleanup()


@pytest.mark.parametrize(
    ("max_turns", "early_exit"),
    [
        (1, False),
        (3, True),
    ],
)
def test_antigravity_cli_omits_forced_schema_in_both_single_and_multi_turn(
    max_turns: int,
    early_exit: bool,
    home_dir: Path,
) -> None:
    """Verify AntigravityCliRuntime omits forced --json-schema in both single and multi-turn."""
    rt = AntigravityCliRuntime(
        options=AntigravityCliOptions(
            home_dir=home_dir,
            max_turns=max_turns,
            early_exit=early_exit,
        )
    )
    rt._resident = ("skill-a", "skill-b")
    cmd = rt.build_command("query")
    assert "--json-schema" not in cmd
    assert "--disable-slash-commands" in cmd


def test_antigravity_cli_generator_normalized_model_delegates_to_normalize_agy_model() -> None:
    """Verify AntigravityCliGenerator.normalized_model does not hardcode gemini-3.8-flash (5.D)."""
    gen = AntigravityCliGenerator(model="gemini-2.5-flash")
    try:
        assert gen.normalized_model == "gemini-2.5-flash"
    finally:
        gen.cleanup()


def test_antigravity_cli_runtime_clone_isolated_creates_distinct_home_dir(home_dir: Path) -> None:
    """Verify clone_isolated allocates a separate temporary home directory for worker isolation."""
    rt = AntigravityCliRuntime(options=AntigravityCliOptions(home_dir=home_dir))
    clone = rt.clone_isolated()
    try:
        assert clone.home_dir != rt.home_dir
        assert clone.home_dir.is_dir()
    finally:
        clone.cleanup()
        assert not clone.home_dir.exists()
        assert rt.home_dir.is_dir()
