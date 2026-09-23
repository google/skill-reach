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

"""Verify the Claude Code agent: transcripts, CLI arguments, and isolation."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from reach.catalog import build_catalogs, load_skills
from reach.config import agent_default_model
from reach.models import CatalogMode
from reach.runtime.claude_code import (
    POSIX_ENTERPRISE_SKILL_DIRS,
    ClaudeCodeOptions,
    ClaudeCodeRuntime,
    ClaudeGenerator,
    enterprise_skill_dirs,
    inherited_skill_dirs,
    parse_stream,
)

from .conftest import canned, claude_stream, install_one


@pytest.fixture
def runtime() -> ClaudeCodeRuntime:
    """Provide a ClaudeCodeRuntime instance with default options."""
    return ClaudeCodeRuntime()


@pytest.fixture
def generator() -> ClaudeGenerator:
    """Provide a ClaudeGenerator instance with default options."""
    return ClaudeGenerator()


def test_extracts_selection_from_real_transcript(real_stream) -> None:
    """Verify parse_stream extracts selection metadata from real Claude Code transcript."""
    summary = parse_stream(real_stream)
    assert summary.invoked_skill == "gcs-lifecycle-rules"
    assert summary.invoked_skills == ("gcs-lifecycle-rules",)
    assert summary.observed_catalog == ("gcs-lifecycle-rules", "gcs-retention-policy")
    assert summary.cost_usd == pytest.approx(0.1949)
    assert summary.duration_ms == 1911
    assert summary.resolved_model == "claude-opus-5"


def test_the_init_event_names_the_model_that_actually_answered(make_stream) -> None:
    """Verify parse_stream extracts resolved model identifier from init event."""
    summary = parse_stream(
        make_stream(catalog=["a"], invoked="a", model="claude-sonnet-5"),
    )
    assert summary.resolved_model == "claude-sonnet-5"


@pytest.mark.parametrize(
    "model",
    [None, "", 5],
    ids=["absent", "empty", "not-a-string"],
)
def test_a_stream_that_names_no_model_leaves_it_unrecorded(make_stream, model) -> None:
    """Verify missing or empty model fields leave resolved_model empty."""
    lines = make_stream(catalog=["a"], invoked="a", model=model)
    assert parse_stream(lines).resolved_model == ""


def test_an_empty_stream_names_no_model(make_stream) -> None:
    """Verify empty stream input reports empty string for resolved_model."""
    assert parse_stream([]).resolved_model == ""


def test_non_selection_yields_none(make_stream) -> None:
    """Verify invoked_skill is None when transcript contains no tool use event."""
    summary = parse_stream(make_stream(catalog=["a", "b"], invoked=None))
    assert summary.invoked_skill is None
    assert summary.observed_catalog == ("a", "b")


def test_first_invocation_wins() -> None:
    """Verify invoked_skill retains the first tool invocation when multiple occur."""
    lines = claude_stream(catalog=["a", "b"], invoked_skills=["a", "b"])
    assert parse_stream(lines).invoked_skill == "a"


def test_every_invocation_is_collected_in_order() -> None:
    """Verify invoked_skills collects all invoked skills in sequential order."""
    lines = claude_stream(catalog=["a", "b"], invoked_skills=["a", "b"])
    summary = parse_stream(lines)
    assert summary.invoked_skill == "a"
    assert summary.invoked_skills == ("a", "b")


def test_a_single_invocation_is_still_collected(make_stream) -> None:
    """Verify single skill invocation populates invoked_skills tuple."""
    lines = make_stream(catalog=["a"], invoked="a")
    assert parse_stream(lines).invoked_skills == ("a",)


def test_no_invocation_collects_nothing(make_stream) -> None:
    """Verify invoked_skills is empty tuple when no skills are invoked."""
    lines = make_stream(catalog=["a"], invoked=None)
    assert parse_stream(lines).invoked_skills == ()


def test_assistant_reasoning_and_thinking_blocks_are_collected() -> None:
    """Verify thinking and preamble text blocks are extracted into reasoning tuple."""
    lines = claude_stream(
        invoked="skill-a",
        reasoning=[
            ("thinking", "Let's analyze whether to use skill-a or skill-b."),
            ("text", "I'll invoke skill-a because it matches the query."),
        ],
        include_init=False,
        include_result=False,
    )
    summary = parse_stream(lines)
    assert summary.invoked_skill == "skill-a"
    assert summary.reasoning == (
        "Let's analyze whether to use skill-a or skill-b.",
        "I'll invoke skill-a because it matches the query.",
    )


def test_assistant_multi_turn_reasoning_traces_are_collected_in_order() -> None:
    """Verify thinking blocks across multiple assistant turns are collected sequentially."""
    lines = claude_stream(
        assistant_turns=[
            [
                {"type": "thinking", "thinking": "First turn thought."},
                {"type": "tool_use", "name": "Bash", "input": {"command": "ls"}},
            ],
            [
                {"type": "thinking", "thinking": "Second turn thought."},
                {"type": "tool_use", "name": "Skill", "input": {"skill": "skill-a"}},
            ],
        ],
        include_init=False,
        include_result=False,
    )
    summary = parse_stream(lines)
    assert summary.invoked_skill == "skill-a"
    assert summary.reasoning == ("First turn thought.", "Second turn thought.")


@pytest.mark.parametrize(
    "line",
    ["", "   ", "not json at all", "[1, 2, 3]", '{"type":"assistant"}'],
    ids=["empty", "whitespace", "garbage", "json-array", "missing-message"],
)
def test_malformed_lines_are_skipped(line, make_stream) -> None:
    """Verify unparseable or unexpected transcript lines are ignored during parsing."""
    lines = make_stream(catalog=["a"], invoked="a")
    lines.insert(1, line)
    assert parse_stream(lines).invoked_skill == "a"


def test_the_selection_is_read_past_the_blocks_around_it(make_stream) -> None:
    """Verify skill selection is extracted even when message contains extra content blocks."""
    lines = make_stream(catalog=["a"], invoked="a")
    crowded = json.loads(lines[1])
    crowded["message"]["content"] = [
        "a bare string where a block should be",
        {"type": "text", "text": "Let me use a skill."},
        {"type": "tool_use", "name": "Skill", "input": {}},
        *crowded["message"]["content"],
    ]
    lines[1] = json.dumps(crowded)
    assert parse_stream(lines).invoked_skill == "a"


def test_a_catalog_announced_in_the_wrong_shape_is_read_as_unannounced(
    make_stream,
) -> None:
    """Verify invalid catalog formats are skipped while tool leak scanning continues."""
    lines = make_stream(catalog=["a"], invoked="a", tools=["Skill", "Glob"])
    init = json.loads(lines[0])
    init["skills"] = "a"
    lines[0] = json.dumps(init)

    summary = parse_stream(lines)
    assert summary.observed_catalog == ()
    assert summary.leaked_tools == ("Glob",)
    assert summary.invoked_skill == "a"


def test_result_without_cost_is_tolerated(make_stream) -> None:
    """Verify result events omitting cost fields set cost_usd to None."""
    summary = parse_stream(make_stream(catalog=["a"], invoked="a", cost=None))
    assert summary.invoked_skill == "a"
    assert summary.cost_usd is None


def test_max_turns_termination_is_not_an_error() -> None:
    """Verify error_max_turns subtype is recognized as valid saw_result completion."""
    lines = claude_stream(catalog=["a"], invoked="a", subtype="error_max_turns")
    summary = parse_stream(lines)
    assert summary.result_subtype == "error_max_turns"
    assert summary.saw_result is True
    assert summary.invoked_skill == "a"


def test_absent_result_event_is_detectable(make_stream) -> None:
    """Verify saw_result is False when result event is missing from stream."""
    lines = make_stream(catalog=["a"], invoked="a")[:-1]
    assert parse_stream(lines).saw_result is False


def test_clean_probe_reports_no_leak(make_stream) -> None:
    """Verify transcripts with only Skill tool report empty leaked_tools."""
    summary = parse_stream(make_stream(catalog=["a"], invoked="a"))
    assert summary.observed_tools == ("Skill",)
    assert summary.leaked_tools == ()


def test_surviving_tool_is_reported_as_a_leak(make_stream) -> None:
    """Verify non-Skill tools in init event are flagged in leaked_tools."""
    lines = make_stream(catalog=["a"], invoked=None, tools=["Skill", "Glob", "Grep"])
    assert parse_stream(lines).leaked_tools == ("Glob", "Grep")


def test_stream_without_tools_key_reports_no_leak(make_stream) -> None:
    """Verify init events without tools list report empty leaked_tools."""
    lines = make_stream(catalog=["a"], invoked="a", tools=[])
    assert parse_stream(lines).leaked_tools == ()


def test_command_pins_selection_only_execution() -> None:
    """Verify build_command sets --max-turns 3, project setting sources, and model."""
    command = ClaudeCodeRuntime(
        options=ClaudeCodeOptions(model="claude-sonnet-5"),
    ).build_command("do a thing")
    assert command[command.index("--max-turns") + 1] == "3"
    assert command[command.index("--setting-sources") + 1] == "project"
    assert command[command.index("--model") + 1] == "claude-sonnet-5"


def test_command_includes_effort_flag_when_configured() -> None:
    """Verify build_command appends --effort flag when effort is configured."""
    command = ClaudeCodeRuntime(
        options=ClaudeCodeOptions(model="claude-sonnet-5", effort="high"),
    ).build_command("do a thing")
    idx = command.index("--effort")
    assert command[idx + 1] == "high"


def test_command_omits_effort_flag_when_none() -> None:
    """Verify build_command omits --effort flag when effort is None."""
    command = ClaudeCodeRuntime(
        options=ClaudeCodeOptions(model="custom-unprofiled"),
    ).build_command("do a thing")
    assert "--effort" not in command


def test_command_defaults_to_model_profile_effort() -> None:
    """Verify build_command sets default effort from model profile for claude-sonnet-5."""
    command = ClaudeCodeRuntime(
        options=ClaudeCodeOptions(model="claude-sonnet-5"),
    ).build_command("do a thing")
    assert "--effort" in command
    assert command[command.index("--effort") + 1] == "low"


def test_command_omits_effort_when_explicitly_disabled() -> None:
    """Verify build_command omits --effort flag when effort is explicitly off."""
    command = ClaudeCodeRuntime(
        options=ClaudeCodeOptions(model="claude-sonnet-5", effort="off"),
    ).build_command("do a thing")
    assert "--effort" not in command


def test_model_defaults_to_the_one_every_recorded_result_used() -> None:
    """Verify ClaudeCodeOptions and Runtime default model to configured default."""
    default = agent_default_model("claude-code")
    assert default is not None
    assert ClaudeCodeOptions().model == default
    assert ClaudeCodeRuntime().model == default


def test_command_denies_tools_that_would_consume_the_turn(
    runtime: ClaudeCodeRuntime,
) -> None:
    """Verify build_command denies Bash tool and excludes dynamic prompt sections."""
    command = runtime.build_command("do a thing")
    denied = command[command.index("--disallowedTools") + 1 :]
    assert "Bash" in denied
    assert "Skill" not in denied
    assert "--exclude-dynamic-system-prompt-sections" in command


def test_command_omits_denial_when_unconfigured() -> None:
    """Verify disallowedTools is omitted when denied_tools is empty."""
    options = ClaudeCodeOptions(denied_tools=(), exclude_dynamic_prompt=False)
    command = ClaudeCodeRuntime(options=options).build_command("do a thing")
    assert "--disallowedTools" not in command
    assert "--exclude-dynamic-system-prompt-sections" not in command


def test_command_configures_tools_and_strict_mcp(runtime: ClaudeCodeRuntime) -> None:
    """Verify tools and strict_mcp_config flags are included by default."""
    command = runtime.build_command("do a thing")
    assert "--tools" in command
    assert command[command.index("--tools") + 1] == "Skill"
    assert "--strict-mcp-config" in command


def test_command_configures_no_session_persistence() -> None:
    """Verify --no-session-persistence is passed by default and omitted when disabled."""
    runtime = ClaudeCodeRuntime()
    command = runtime.build_command("do a thing")
    assert "--no-session-persistence" in command

    generator = ClaudeGenerator()
    completion_cmd = generator.build_completion_command()
    assert "--no-session-persistence" in completion_cmd

    disabled_runtime = ClaudeCodeRuntime(
        options=ClaudeCodeOptions(no_session_persistence=False),
    )
    disabled_generator = ClaudeGenerator(
        options=ClaudeCodeOptions(no_session_persistence=False),
    )
    assert "--no-session-persistence" not in disabled_runtime.build_command("do a thing")
    assert "--no-session-persistence" not in disabled_generator.build_completion_command()


def test_env_disables_bundled_skills(
    runtime: ClaudeCodeRuntime,
    tmp_path: Path,
) -> None:
    """Verify build_env sets CLAUDE_CODE_DISABLE_BUNDLED_SKILLS env variable."""
    env = runtime.build_env(tmp_path)
    assert env.get("CLAUDE_CODE_DISABLE_BUNDLED_SKILLS") == "1"

    kept_runtime = ClaudeCodeRuntime(options=ClaudeCodeOptions(disable_bundled_skills=False))
    kept_env = kept_runtime.build_env(tmp_path)
    assert "CLAUDE_CODE_DISABLE_BUNDLED_SKILLS" not in kept_env


def test_env_configures_isolated_claude_config_dir(
    runtime: ClaudeCodeRuntime,
    tmp_path: Path,
) -> None:
    """Verify build_env configures CLAUDE_CONFIG_DIR pointing inside workdir."""
    env = runtime.build_env(tmp_path)
    assert env["CLAUDE_CONFIG_DIR"] == str(tmp_path / ".reach_claude_config")
    assert (tmp_path / ".reach_claude_config").is_dir()


def test_env_syncs_claude_settings_and_defaults_region(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify build_env syncs settings and defaults CLOUD_ML_REGION to global for Vertex."""
    claude_home = tmp_path / ".claude"
    claude_home.mkdir()
    settings_file = claude_home / "settings.json"
    settings_file.write_text(
        json.dumps(
            {
                "env": {
                    "CLAUDE_CODE_USE_VERTEX": "1",
                    "ANTHROPIC_VERTEX_PROJECT_ID": "auto-proj",
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("CLOUD_ML_REGION", raising=False)

    workdir = tmp_path / "work"
    workdir.mkdir()
    rt = ClaudeCodeRuntime()
    env = rt.build_env(workdir)

    assert env["CLAUDE_CODE_USE_VERTEX"] == "1"
    assert env["ANTHROPIC_VERTEX_PROJECT_ID"] == "auto-proj"
    assert env["CLOUD_ML_REGION"] == "global"


def test_claude_code_build_env_strips_blocked_vars_from_settings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify build_env filters blocked environment variables from settings.json."""
    claude_home = tmp_path / ".claude"
    claude_home.mkdir()
    settings_file = claude_home / "settings.json"
    settings_file.write_text(
        json.dumps(
            {
                "env": {
                    "ALLOWED_VAR": "allowed",
                    "AWS_SECRET_ACCESS_KEY": "should-be-blocked",
                    "CUSTOM_SECRET": "should-also-be-blocked",
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    workdir = tmp_path / "work"
    workdir.mkdir()

    # Custom blocked var configured on options
    options = ClaudeCodeOptions(blocked_env_vars=("CUSTOM_SECRET",))
    rt = ClaudeCodeRuntime(options=options)
    env = rt.build_env(workdir)
    assert env.get("ALLOWED_VAR") == "allowed"
    assert "CUSTOM_SECRET" not in env

    # Default blocked vars stripped when options have no explicit blocked_env_vars
    default_rt = ClaudeCodeRuntime()
    default_env = default_rt.build_env(workdir)
    assert default_env.get("ALLOWED_VAR") == "allowed"
    assert "AWS_SECRET_ACCESS_KEY" not in default_env


def test_denial_covers_the_replacements_the_runtime_substitutes() -> None:
    """Verify default denied_tools includes Glob and Grep."""
    denied = set(ClaudeCodeOptions().denied_tools)
    assert {"Glob", "Grep"} <= denied
    assert "Skill" not in denied


def test_command_removes_the_skills_the_cli_ships_with(
    runtime: ClaudeCodeRuntime,
) -> None:
    """Verify build_command sets disableBundledSkills and disables doctor override."""
    command = runtime.build_command("do a thing")
    payload = json.loads(command[command.index("--settings") + 1])
    assert payload["disableBundledSkills"] is True
    assert payload["skillOverrides"] == {"doctor": "off"}


def test_residency_controls_can_be_turned_off() -> None:
    """Verify --settings is omitted when bundled skill disabling is turned off."""
    options = ClaudeCodeOptions(disable_bundled_skills=False, skill_overrides={})
    assert options.settings_json() is None
    assert "--settings" not in ClaudeCodeRuntime(options=options).build_command("q")


def test_settings_payload_is_ordered_so_a_command_line_is_reproducible() -> None:
    """Verify settings JSON serialization produces sorted keys."""
    options = ClaudeCodeOptions(skill_overrides={"z": "off", "a": "off"})
    assert options.settings_json() == (
        '{"disableBundledSkills": true, "skillOverrides": {"a": "off", "z": "off"}}'
    )


def test_the_roots_are_the_directories_this_runtime_actually_reads(
    runtime: ClaudeCodeRuntime,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify skill_roots returns user and project directories in precedence order."""
    home = tmp_path / "home"
    (home / ".claude" / "skills" / "personal").mkdir(parents=True)
    monkeypatch.setattr(Path, "home", classmethod(lambda _cls: home))
    project = tmp_path / "project"
    (project / ".claude" / "skills" / "local").mkdir(parents=True)

    roots = runtime.skill_roots(project)

    assert [(r.scope, r.path) for r in roots] == [
        ("user", home / ".claude" / "skills"),
        ("project", project / ".claude" / "skills"),
    ]
    assert [r.precedence for r in roots] == [0, 1]


def test_a_directory_that_is_not_there_is_not_offered_as_a_root(
    runtime: ClaudeCodeRuntime,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify non-existent paths are omitted from skill_roots output."""
    monkeypatch.setattr(Path, "home", classmethod(lambda _cls: tmp_path / "nohome"))
    workdir = tmp_path / "bare"
    workdir.mkdir()

    assert runtime.skill_roots(workdir) == ()


def test_a_catalog_installed_above_the_workspace_is_a_root_too(
    runtime: ClaudeCodeRuntime,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify skill directories in parent directory tree are included in skill_roots."""
    monkeypatch.setattr(Path, "home", classmethod(lambda _cls: tmp_path / "nohome"))
    (tmp_path / ".claude" / "skills" / "inherited").mkdir(parents=True)
    workdir = tmp_path / "mid" / "sub"
    workdir.mkdir(parents=True)

    assert tmp_path / ".claude" / "skills" in [r.path for r in runtime.skill_roots(workdir)]


def test_an_enterprise_directory_outranks_everything_else(
    runtime: ClaudeCodeRuntime,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify enterprise skill directories receive precedence index 0."""
    managed = tmp_path / "managed"
    managed.mkdir()
    monkeypatch.setattr(
        "reach.runtime.claude_code.enterprise_skill_dirs",
        lambda: (managed,),
    )
    home = tmp_path / "home"
    (home / ".claude" / "skills").mkdir(parents=True)
    monkeypatch.setattr(Path, "home", classmethod(lambda _cls: home))

    first, *rest = runtime.skill_roots(tmp_path / "anywhere")

    assert (first.scope, first.path, first.precedence) == ("enterprise", managed, 0)
    assert all(r.precedence > 0 for r in rest)


def test_every_managed_location_is_absolute_wherever_this_runs() -> None:
    """Verify all paths returned by enterprise_skill_dirs are absolute paths."""
    assert POSIX_ENTERPRISE_SKILL_DIRS
    assert all(path.is_absolute() for path in enterprise_skill_dirs())


def test_the_windows_location_is_read_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify PROGRAMDATA environment variable is used for Windows enterprise path."""
    monkeypatch.setenv("PROGRAMDATA", str(tmp_path / "PD"))
    assert enterprise_skill_dirs()[-1] == tmp_path / "PD" / "ClaudeCode" / "skills"


def test_no_windows_location_is_offered_where_there_is_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify enterprise_skill_dirs returns POSIX list when PROGRAMDATA is unset."""
    monkeypatch.delenv("PROGRAMDATA", raising=False)
    assert enterprise_skill_dirs() == POSIX_ENTERPRISE_SKILL_DIRS


# --- Workspace catalog inheritance isolation ---------------------------------


def test_a_stale_ancestor_directory_is_detected(tmp_path: Path) -> None:
    """Verify discovery finds skill directory in parent directories above workspace."""
    (tmp_path / ".claude" / "skills" / "ghost").mkdir(parents=True)
    nested = tmp_path / "mid" / "sub"
    nested.mkdir(parents=True)
    assert inherited_skill_dirs(nested) == (tmp_path / ".claude" / "skills",)


def test_a_clean_workspace_inherits_nothing(tmp_path: Path) -> None:
    """Verify clean workspace without parent skill directories inherits nothing."""
    workdir = tmp_path / "work"
    workdir.mkdir()
    assert inherited_skill_dirs(workdir) == ()


def test_the_workspaces_own_directory_is_not_inheritance(tmp_path: Path) -> None:
    """Verify skills within workspace are not treated as inherited parent catalogs."""
    (tmp_path / ".claude" / "skills").mkdir(parents=True)
    assert inherited_skill_dirs(tmp_path) == ()


def test_the_user_scope_directory_is_exempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify user-scope home directory skills are exempt from ancestor inheritance checks."""
    (tmp_path / ".claude" / "skills" / "personal").mkdir(parents=True)
    monkeypatch.setattr(Path, "home", classmethod(lambda _cls: tmp_path))
    workdir = tmp_path / "projects" / "work"
    workdir.mkdir(parents=True)
    assert inherited_skill_dirs(workdir) == ()


def test_install_refuses_a_workspace_that_inherits_skills(
    runtime: ClaudeCodeRuntime,
    skill_repo: Path,
    tmp_path: Path,
) -> None:
    """Verify install raises ValueError when ancestor directories contain skill definitions."""
    skills = load_skills(skill_repo)
    catalog = build_catalogs(skills, CatalogMode.SINGLETON)[0]
    (tmp_path / ".claude" / "skills" / "stale").mkdir(parents=True)
    with pytest.raises(ValueError, match="inherits skills"):
        runtime.install(catalog, skills, tmp_path / "nested" / "work")


def test_select_discards_a_probe_with_an_undeclared_resident(
    monkeypatch,
    make_stream,
    runtime,
    skill_repo,
    tmp_path,
) -> None:
    """Verify select reports residency leak error when extra skills appear in transcript."""
    workdir = tmp_path / "work"
    resident = install_one(runtime, skill_repo, workdir)
    canned(monkeypatch, make_stream(catalog=[resident, "dataviz"], invoked=resident))
    assert runtime.select("do a thing", workdir).error == "residency leak: dataviz"


def test_select_accepts_a_probe_whose_residency_is_exact(
    monkeypatch,
    make_stream,
    runtime,
    skill_repo,
    tmp_path,
) -> None:
    """Verify select succeeds when transcript catalog matches installed skills exactly."""
    workdir = tmp_path / "work"
    resident = install_one(runtime, skill_repo, workdir)
    canned(monkeypatch, make_stream(catalog=[resident], invoked=resident))
    assert runtime.select("do a thing", workdir).error is None


def test_a_run_that_keeps_the_bundled_skills_expects_extras(
    monkeypatch,
    make_stream,
    skill_repo,
    tmp_path,
) -> None:
    """Verify extra skills are allowed when disable_bundled_skills is False."""
    runtime = ClaudeCodeRuntime(options=ClaudeCodeOptions(disable_bundled_skills=False))
    workdir = tmp_path / "work"
    resident = install_one(runtime, skill_repo, workdir)
    canned(monkeypatch, make_stream(catalog=[resident, "dataviz"], invoked=resident))
    assert runtime.select("do a thing", workdir).error is None


def test_residency_is_unjudged_before_a_catalog_is_installed(
    monkeypatch,
    make_stream,
    runtime,
    tmp_path,
) -> None:
    """Verify select does not validate residency when no catalog is installed."""
    canned(monkeypatch, make_stream(catalog=["a", "b"], invoked="a"))
    assert runtime.select("do a thing", tmp_path).error is None


@pytest.mark.parametrize(
    ("tools", "expected_error"),
    [(["Skill"], None), (["Skill", "Glob"], "tool leak: Glob")],
    ids=["isolated", "leaked"],
)
def test_select_discards_a_leaked_probe(
    monkeypatch,
    make_stream,
    runtime,
    tmp_path,
    tools,
    expected_error,
) -> None:
    """Verify select reports tool leak error when unauthorized tools appear in transcript."""
    runtime.options = runtime.options.model_copy(update={"allowed_tools": ("Skill",)})
    canned(monkeypatch, make_stream(catalog=["a"], invoked="a", tools=tools))
    outcome = runtime.select("do a thing", tmp_path)
    assert outcome.error == expected_error
    assert outcome.observed_tools == tuple(tools)


def test_select_allows_all_tools_by_default_when_allowed_tools_omitted(
    monkeypatch,
    make_stream,
    runtime,
    tmp_path,
) -> None:
    """Verify all tools are permitted by default when allowed_tools is not configured."""
    canned(
        monkeypatch,
        make_stream(catalog=["a"], invoked="a", tools=["Skill", "Bash", "Read"]),
    )
    outcome = runtime.select("do a thing", tmp_path)
    assert outcome.invoked_skill == "a"
    assert outcome.error is None


def test_select_discards_a_leaked_hit_not_just_a_miss(
    monkeypatch,
    make_stream,
    runtime,
    tmp_path,
) -> None:
    """Verify tool leak error is reported even on correct skill selection."""
    runtime.options = runtime.options.model_copy(update={"allowed_tools": ("Skill",)})
    canned(
        monkeypatch,
        make_stream(catalog=["a"], invoked="a", tools=["Skill", "Bash"]),
    )
    outcome = runtime.select("do a thing", tmp_path)
    assert outcome.invoked_skill == "a"
    assert outcome.error == "tool leak: Bash"


def test_select_reports_every_invocation_alongside_the_first(
    monkeypatch,
    runtime,
    tmp_path,
) -> None:
    """Verify select records both primary selection and full invocation sequence."""
    canned(monkeypatch, claude_stream(catalog=["a", "b"], invoked_skills=["a", "b"]))
    outcome = runtime.select("do a thing", tmp_path)
    assert outcome.invoked_skill == "a"
    assert outcome.invoked_skills == ("a", "b")


def test_select_reports_a_runtime_error_subtype(
    monkeypatch,
    runtime,
    tmp_path,
) -> None:
    """Verify non-success result subtype is returned as runtime error."""
    canned(
        monkeypatch,
        claude_stream(catalog=["a"], invoked=None, subtype="error_during_execution"),
    )
    assert runtime.select("q", tmp_path).error == "runtime error: error_during_execution"


def test_select_reports_rate_limiting_when_retries_occur(
    mock_subprocess,
    runtime,
    tmp_path,
) -> None:
    """Verify select reports rate limiting retries when result event is missing."""
    mock_subprocess(
        returncode=1,
        lines=claude_stream(retries=2, include_result=False, model="claude-sonnet-5"),
    )
    outcome = runtime.select("q", tmp_path)
    assert outcome.error is not None
    assert "rate limit (429): 2 retries exceeded" in outcome.error


def test_complete_returns_raw_text(mock_subprocess, generator) -> None:
    """Verify complete returns stdout text from subprocess execution."""
    mock_subprocess(stdout="{}")
    assert generator.complete("draft me a query") == "{}"


def test_a_generation_prompt_travels_on_stdin(mock_subprocess, generator) -> None:
    """Verify prompt text is passed via stdin input rather than CLI arguments."""
    seen: dict[str, object] = {}
    command: list[str] = []

    def record(args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        command.extend(args)
        seen.update(kwargs)
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")

    mock_subprocess(handler=record)
    generator.complete("draft me a query")
    assert seen["input"] == "draft me a query"
    assert "draft me a query" not in command


def test_complete_appends_schema_to_prompt(mock_subprocess, generator) -> None:
    """Verify prompt receives schema instructions when schema is provided."""
    seen: dict[str, object] = {}

    def record(args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        del args
        seen.update(kwargs)
        return subprocess.CompletedProcess(args=[], returncode=0, stdout="{}", stderr="")

    mock_subprocess(handler=record)
    generator.complete("draft", schema={"type": "object"})
    assert "draft" in str(seen["input"])
    assert '{"type": "object"}' in str(seen["input"])


def test_generation_banks_what_it_cost(mock_subprocess, generator) -> None:
    """Verify completion_cost_usd accumulates cost across multiple completions."""
    mock_subprocess(stdout=json.dumps({"result": "a query", "total_cost_usd": 0.25}))
    assert generator.complete("draft") == "a query"
    assert generator.complete("draft again") == "a query"
    assert generator.completion_cost_usd == pytest.approx(0.5)
    assert generator.completions == 2


def test_an_envelope_without_a_cost_still_yields_its_reply(
    mock_subprocess,
    generator,
) -> None:
    """Verify complete parses response text when JSON envelope omits total_cost_usd."""
    mock_subprocess(stdout=json.dumps({"result": "a query"}))
    assert generator.complete("draft") == "a query"
    assert generator.completion_cost_usd == 0.0
    assert generator.completions == 1


def test_complete_raises_when_the_runtime_fails(mock_subprocess, generator) -> None:
    """Verify complete raises RuntimeError when subprocess fails."""
    mock_subprocess(returncode=1, stderr="credit balance too low")
    with pytest.raises(RuntimeError, match="credit balance too low"):
        generator.complete("draft me a query")


def test_a_refusal_the_cli_explained_is_repeated_not_summarized(
    mock_subprocess,
    generator,
) -> None:
    """Verify complete extracts error message from JSON envelope stdout on nonzero exit."""
    mock_subprocess(
        returncode=1,
        stdout=json.dumps(
            {"is_error": True, "result": "Prompt is too long", "total_cost_usd": 0},
        ),
    )
    with pytest.raises(RuntimeError, match="Prompt is too long"):
        generator.complete("a prompt carrying every resident body")


def test_a_prompt_refused_for_its_length_names_the_flag_that_shortens_it(
    mock_subprocess,
    generator,
) -> None:
    """Verify prompt length error message suggests --top-rivals option."""
    mock_subprocess(
        returncode=1,
        stdout=json.dumps({"is_error": True, "result": "Prompt is too long"}),
    )
    with pytest.raises(RuntimeError, match="top-rivals"):
        generator.complete("a prompt carrying every resident body")


def test_a_failure_that_explained_itself_nowhere_still_names_the_exit(
    mock_subprocess,
    generator,
) -> None:
    """Verify complete error message includes exit code when stderr/stdout are empty."""
    mock_subprocess(returncode=137)
    with pytest.raises(RuntimeError, match="exit 137"):
        generator.complete("draft me a query")


def test_stderr_still_wins_over_an_envelope_that_says_nothing(
    mock_subprocess,
    generator,
) -> None:
    """Verify complete prefers stderr message when JSON stdout is invalid."""
    mock_subprocess(
        returncode=1,
        stdout="not the envelope at all",
        stderr="connection reset by peer",
    )
    with pytest.raises(RuntimeError, match="connection reset by peer"):
        generator.complete("draft me a query")


def test_residency_is_reported_not_assumed(monkeypatch, make_stream, runtime, tmp_path) -> None:
    """Verify observed_catalog contains skill list reported in transcript."""
    canned(monkeypatch, make_stream(catalog=["a", "b"], invoked="a"))
    outcome = runtime.select("q", tmp_path)
    assert outcome.observed_catalog == ("a", "b")


def test_claude_code_multi_turn_early_exit_stops_at_target(monkeypatch, tmp_path) -> None:
    """Verify early exit stops reading stream when target skill is observed."""
    events = claude_stream(
        catalog=["data-prep", "target-skill", "extra-skill"],
        invoked_skills=["data-prep", "target-skill", "extra-skill"],
        model="claude-sonnet-5",
    )
    canned(monkeypatch, events)
    rt = ClaudeCodeRuntime(options=ClaudeCodeOptions(early_exit=True, max_turns=3))
    outcome = rt.select("test query", tmp_path, target_skill="target-skill")

    assert outcome.early_exit is True
    assert outcome.invoked_skill == "data-prep"
    assert outcome.invoked_skills == ("data-prep", "target-skill")
    assert outcome.turns_taken == 2
    assert outcome.error is None


def test_claude_code_multi_turn_budget_exhaustion(monkeypatch, tmp_path) -> None:
    """Verify early exit triggers when maximum turn budget is exhausted."""
    events = claude_stream(
        catalog=["s1", "s2", "s3", "s4"],
        invoked_skills=["s1", "s2", "s3", "s4"],
        model="claude-sonnet-5",
    )
    canned(monkeypatch, events)
    rt = ClaudeCodeRuntime(options=ClaudeCodeOptions(early_exit=True, max_turns=3))
    outcome = rt.select("test query", tmp_path, target_skill="different-target")

    assert outcome.early_exit is True
    assert outcome.invoked_skills == ("s1", "s2", "s3")
    assert outcome.turns_taken == 3
    assert outcome.error is None


def test_claude_code_extract_skills_from_line(runtime: ClaudeCodeRuntime) -> None:
    """Verify extract_skills_from_line extracts multiple invocations and handles invalid JSON."""
    event = json.dumps(
        {
            "type": "assistant",
            "message": {
                "content": [
                    {"type": "tool_use", "name": "Skill", "input": {"skill": "s1"}},
                    {"type": "tool_use", "name": "Skill", "input": {"skill": "s2"}},
                ],
            },
        },
    )
    assert runtime.extract_skills_from_line(event) == ("s1", "s2")
    assert runtime.extract_skills_from_line("not-json") == ()
    assert runtime.extract_skills_from_line(json.dumps({"type": "system"})) == ()


def test_claude_code_parallel_tool_calls_counts_single_turn(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify multiple skill tool calls in a single assistant turn reports 1 turn taken."""
    events = claude_stream(
        catalog=["s1", "s2"],
        parallel_skills=["s1", "s2"],
        model="claude-sonnet-5",
        cost=0.01,
        duration_ms=500,
    )
    canned(monkeypatch, events)
    rt = ClaudeCodeRuntime(options=ClaudeCodeOptions(early_exit=False))
    outcome = rt.select("test query", tmp_path)

    assert outcome.invoked_skills == ("s1", "s2")
    assert outcome.turns_taken == 1


def test_claude_code_multi_turn_abstention_counts_all_turns(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify multiple assistant turns without skill invocations reports all turns taken."""
    events = claude_stream(
        catalog=["s1"],
        tools=["Bash"],
        turn_texts=[
            "Exploring workspace files.",
            "Still checking directory contents.",
            "No skill is needed for this query.",
        ],
        model="claude-sonnet-5",
        cost=0.02,
        duration_ms=1200,
    )
    canned(monkeypatch, events)
    rt = ClaudeCodeRuntime(options=ClaudeCodeOptions(early_exit=False))
    outcome = rt.select("test query", tmp_path)

    assert outcome.invoked_skills == ()
    assert outcome.turns_taken == 3


def test_claude_code_build_completion_command(generator: ClaudeGenerator) -> None:
    """Verify build_completion_command constructs arguments correctly."""
    cmd = generator.build_completion_command("test prompt")
    assert cmd[:4] == ["claude", "-p", "--model", "claude-sonnet-5"]
    assert "--output-format" in cmd


def test_select_accepts_error_max_turns_subtype(
    monkeypatch: pytest.MonkeyPatch,
    runtime: ClaudeCodeRuntime,
    tmp_path: Path,
) -> None:
    """Verify error_max_turns terminal subtype is treated as valid completion without error."""
    canned(
        monkeypatch,
        claude_stream(catalog=["a"], invoked="a", subtype="error_max_turns"),
    )
    outcome = runtime.select("q", tmp_path)
    assert outcome.error is None
    assert outcome.invoked_skill == "a"


def test_assistant_event_with_non_dict_message_or_input() -> None:
    """Verify assistant events with non-dict message or input payloads do not crash."""
    lines = [
        json.dumps({"type": "assistant", "message": "not-a-dict"}),
        json.dumps(
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {"type": "tool_use", "name": "Skill", "input": "not-a-dict"},
                        {"type": "tool_use", "name": "Skill", "input": None},
                    ],
                },
            },
        ),
    ]
    summary = parse_stream(lines)
    assert summary.invoked_skill is None
    assert summary.invoked_skills == ()


def test_result_event_with_non_numeric_telemetry() -> None:
    """Verify result event parses non-numeric telemetry values gracefully as None."""
    lines = [
        json.dumps(
            {
                "type": "result",
                "subtype": "success",
                "total_cost_usd": "non-numeric",
                "duration_ms": "invalid-int",
            },
        ),
    ]
    summary = parse_stream(lines)
    assert summary.cost_usd is None
    assert summary.duration_ms is None
    assert summary.result_subtype == "success"


def test_claude_generator_build_env_forwards_api_key() -> None:
    """Verify ClaudeGenerator build_env passes ANTHROPIC_API_KEY when configured in options."""
    gen = ClaudeGenerator(
        options=ClaudeCodeOptions(
            api_key="claude-secret-key",
        ),
    )
    env = gen.build_env()
    assert env["ANTHROPIC_API_KEY"] == "claude-secret-key"


def test_claude_code_build_env_applies_region_and_project_options(
    tmp_path: Path,
) -> None:
    """Verify ClaudeCodeRuntime build_env respects cloud_ml_region and vertex_project_id options."""
    rt = ClaudeCodeRuntime(
        options=ClaudeCodeOptions(
            cloud_ml_region="us-east5",
            vertex_project_id="custom-proj",
        )
    )
    env = rt.build_env(tmp_path)
    assert env["CLOUD_ML_REGION"] == "us-east5"
    assert env["ANTHROPIC_VERTEX_PROJECT_ID"] == "custom-proj"


def test_claude_generator_build_env_applies_region_and_project_options() -> None:
    """Verify ClaudeGenerator build_env respects cloud_ml_region and vertex_project_id options."""
    gen = ClaudeGenerator(
        options=ClaudeCodeOptions(
            cloud_ml_region="europe-west1",
            vertex_project_id="gen-proj",
        )
    )
    env = gen.build_env()
    assert env["CLOUD_ML_REGION"] == "europe-west1"
    assert env["ANTHROPIC_VERTEX_PROJECT_ID"] == "gen-proj"


def test_claude_code_parse_stream_extracts_prompt_tokens(
    runtime: ClaudeCodeRuntime,
) -> None:
    """Verify parse_stream extracts prompt_tokens from assistant and result usage fields."""
    stream = "\n".join(
        [
            json.dumps(
                {
                    "type": "assistant",
                    "message": {
                        "content": [
                            {
                                "type": "tool_use",
                                "name": "Skill",
                                "input": {"skill": "bigquery-basics"},
                            }
                        ],
                        "usage": {
                            "input_tokens": 1200,
                            "cache_creation_input_tokens": 300,
                            "cache_read_input_tokens": 4500,
                            "output_tokens": 45,
                        },
                    },
                }
            ),
            json.dumps(
                {
                    "type": "result",
                    "subtype": "success",
                    "duration_ms": 2150,
                    "usage": {
                        "input_tokens": 1250,
                        "cache_creation_input_tokens": 300,
                        "cache_read_input_tokens": 4500,
                        "output_tokens": 80,
                    },
                }
            ),
        ]
    )
    summary = runtime.parse_stream(stream.splitlines())
    assert summary.prompt_tokens == 1250 + 300 + 4500
    assert summary.invoked_skills == ("bigquery-basics",)


def test_claude_usage_schema_validation_and_token_resolution() -> None:
    """Verify ClaudeUsage validates constraints, computes totals, and avoids double-counting."""
    from pydantic import ValidationError

    from reach.runtime.claude_code import ClaudeUsage, _extract_usage_prompt_tokens

    # Anthropic input + cache read + cache creation
    u1 = ClaudeUsage.model_validate(
        {"input_tokens": 1000, "cache_creation_input_tokens": 200, "cache_read_input_tokens": 300}
    )
    assert u1.total_prompt_tokens == 1500

    # Avoid double-counting when both input_tokens and prompt_tokens are present
    u2 = ClaudeUsage.model_validate({"input_tokens": 1000, "prompt_tokens": 1000})
    assert u2.total_prompt_tokens == 1000

    # Fallback to prompt_tokens when input_tokens is absent/0
    u3 = ClaudeUsage.model_validate({"prompt_tokens": 750})
    assert u3.total_prompt_tokens == 750

    # Non-negative constraint enforcement
    with pytest.raises(ValidationError):
        ClaudeUsage.model_validate({"input_tokens": -5})

    # _extract_usage_prompt_tokens handles malformed or non-dict input gracefully
    assert _extract_usage_prompt_tokens(None) is None
    assert _extract_usage_prompt_tokens("invalid") is None
    assert _extract_usage_prompt_tokens({"input_tokens": -5}) is None
