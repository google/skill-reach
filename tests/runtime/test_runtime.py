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

"""Verify AgentRuntime interface contracts, factory registration, and model profiles."""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import tomllib
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any, override

import pytest
from pydantic import ValidationError as PydanticValidationError

from reach.catalog import build_catalogs, load_skills
from reach.config import RuntimeSettings, agent_profiles, default_agent, load_config
from reach.models import Catalog, CatalogMode, Skill
from reach.runtime import (
    AgentOptions,
    AgentRuntime,
    AntigravityRuntime,
    CatalogFit,
    CliAgentRuntime,
    CliOptions,
    SelectionOutcome,
    SessionSummary,
    SkillRoot,
    SkillSelectionBase,
    TextGenerator,
    ToolCallInfo,
    agent_default_model,
    antigravity_agents,
    build_runtime,
    build_text_generator,
    cli_agents,
    find_agent_for_model,
    known_agents,
    options_model,
    resolve_options,
    runtime_class,
)
from reach.runtime._env import (
    apply_provider_api_key,
    sanitize_subprocess_env,
    sync_claude_settings_env,
    sync_google_and_gemini_keys,
)
from reach.runtime._fs import (
    install_skills,
    probe_slot_dir,
    probe_slot_id,
    resolve_skill_from_path,
)
from reach.runtime._subprocess import (
    check_tool_leak,
    extract_content_reasoning,
    format_subprocess_error,
    process_failure_reason,
)
from reach.runtime.fake import FakeGenerator, FakeOptions, FakeRuntime
from reach.runtime.profiles import model_profile

from .conftest import MINIMAL_OPTIONS
from .conftest import build_agent as _build_agent

_KEY_SYNC_AGENTS = ("antigravity-cli", "antigravity-sdk", "goose", "pi")


def _assert_runtime_attributes(runtime: AgentRuntime[Any], agent: str) -> None:
    """Verify runtime instance meets required attribute and counter types."""
    assert isinstance(runtime.name, str)
    assert runtime.name == agent
    assert isinstance(runtime.model, str)
    assert len(runtime.model) > 0
    assert isinstance(runtime.skills_subpath, str)
    profile = agent_profiles().get(agent)
    if profile is not None and profile.skills_dir:
        assert runtime.skills_subpath == profile.skills_dir
    assert isinstance(runtime.options, AgentOptions)


def _assert_runtime_interface(runtime: AgentRuntime[Any], tmp_path: Path) -> None:
    """Verify runtime instance satisfies core interface method contracts."""
    workdir = tmp_path / runtime.name
    sdir = runtime.skills_dir(workdir)
    assert isinstance(sdir, Path)
    assert sdir == workdir / runtime.skills_subpath

    roots = runtime.skill_roots(workdir)
    assert isinstance(roots, tuple)
    assert all(isinstance(root, SkillRoot) for root in roots)

    empty_catalog = Catalog(id="test", mode=CatalogMode.ALL, skills=())
    fit = runtime.fit(empty_catalog, ())
    assert isinstance(fit, CatalogFit)

    assert callable(runtime.skills_dir)
    assert callable(runtime.skill_roots)
    assert callable(runtime.fit)
    assert callable(runtime.install)
    assert callable(runtime.select)


def _assert_generator_interface(gen: TextGenerator, agent: str) -> None:
    """Verify text generator instance satisfies generator contract."""
    assert isinstance(gen.name, str)
    assert gen.name == agent
    assert isinstance(gen.model, str)
    assert isinstance(gen.completion_cost_usd, float)
    assert gen.completion_cost_usd >= 0.0
    assert isinstance(gen.completions, int)
    assert gen.completions >= 0
    budget = gen.prompt_budget_chars()
    assert budget is None or (isinstance(budget, int) and budget > 0)
    assert callable(gen.complete)
    assert callable(gen.prompt_budget_chars)


@pytest.mark.parametrize("agent", known_agents())
def test_every_agent_runtime_conforms_to_agent_runtime_contract(
    agent: str,
    tmp_path: Path,
) -> None:
    """Verify every registered agent implementation satisfies the AgentRuntime ABC contract."""
    runtime = build_runtime(
        RuntimeSettings(agent=agent, options=MINIMAL_OPTIONS.get(agent, {})),
    )
    assert isinstance(runtime, AgentRuntime)
    assert issubclass(type(runtime), AgentRuntime)
    _assert_runtime_attributes(runtime, agent)
    _assert_runtime_interface(runtime, tmp_path)


def test_an_unknown_agent_names_the_alternatives() -> None:
    """Verify ValueError lists available agents when unknown agent is requested."""
    with pytest.raises(ValueError, match="unknown runtime agent") as exc_info:
        build_runtime(RuntimeSettings(agent="codex"))
    err_msg = str(exc_info.value)
    for agent in known_agents():
        assert agent in err_msg


@pytest.mark.parametrize("agent", known_agents())
def test_the_agent_carries_the_configured_model(agent: str) -> None:
    """Verify configured model string is stored on constructed instance across all agents."""
    rt = build_runtime(RuntimeSettings(agent=agent, options={"model": "haiku"}))
    assert rt.model == "haiku"
    assert rt.options.model == "haiku"


def test_two_agents_make_the_seam_real() -> None:
    """Verify known_agents registers multiple distinct implementations."""
    agents = known_agents()
    assert len(agents) >= 2
    assert "fake" in agents


def test_a_root_ranks_itself_and_says_under_which_scope(tmp_path: Path) -> None:
    """Verify SkillRoot model validates path, scope, and non-negative precedence."""
    root = SkillRoot(path=tmp_path, scope="user")
    assert root.precedence == 0
    with pytest.raises(PydanticValidationError):
        SkillRoot.model_validate({"path": tmp_path, "scope": "user", "precedence": -1})


@pytest.mark.parametrize("agent", known_agents())
def test_install_rejects_a_catalog_naming_an_unloaded_skill(
    agent: str,
    catalog: Catalog,
    skills: list[Skill],
    tmp_path: Path,
) -> None:
    """Verify install raises KeyError when catalog references unindexed skill."""
    ghost = catalog.model_copy(update={"skills": ("ghost",)})
    runtime = _build_agent(agent, tmp_path)
    with pytest.raises(KeyError, match="ghost"):
        runtime.install(ghost, skills, tmp_path / agent)


def test_the_fake_reports_the_catalog_it_was_given(catalog, skills, tmp_path) -> None:
    """Verify FakeRuntime observed_catalog and invoked_skills match installed catalog skills."""
    runtime = FakeRuntime({"q": "a"})
    runtime.install(catalog, skills, tmp_path)
    outcome = runtime.select("q", tmp_path)
    assert outcome.observed_catalog == ("a", "b")
    assert outcome.invoked_skills == ("a",)


def test_the_fake_abstains_on_anything_unscripted(catalog, skills, tmp_path) -> None:
    """Verify FakeRuntime returns None and empty invoked_skills for unscripted queries."""
    runtime = FakeRuntime({"scripted": "a"})
    runtime.install(catalog, skills, tmp_path)
    outcome = runtime.select("unscripted", tmp_path)
    assert outcome.invoked_skill is None
    assert outcome.invoked_skills == ()


def test_a_scripted_outcome_can_override_residency(catalog, skills, tmp_path) -> None:
    """Verify FakeRuntime allows explicit SelectionOutcome return overrides."""
    runtime = FakeRuntime({"q": SelectionOutcome(observed_catalog=("wrong",))})
    runtime.install(catalog, skills, tmp_path)
    assert runtime.select("q", tmp_path).observed_catalog == ("wrong",)


def test_a_callable_script_sees_the_query(catalog, skills, tmp_path) -> None:
    """Verify FakeRuntime accepts callable dispatch function for query selection."""
    runtime = FakeRuntime(lambda text: "a" if "storage" in text else None)
    runtime.install(catalog, skills, tmp_path)
    assert runtime.select("about storage", tmp_path).invoked_skill == "a"
    assert runtime.select("about networks", tmp_path).invoked_skill is None


def test_the_fake_records_what_it_was_asked(catalog, skills, tmp_path) -> None:
    """Verify FakeRuntime tracks historical installs and queries."""
    runtime = FakeRuntime()
    runtime.install(catalog, skills, tmp_path)
    runtime.select("q1", tmp_path)
    assert runtime.installs == [(catalog, tmp_path)]
    assert runtime.queries == ["q1"]


@pytest.mark.parametrize("agent", known_agents())
def test_every_agent_says_how_much_of_a_catalog_it_would_show(
    agent: str,
    catalog: Catalog,
    skills: list[Skill],
    tmp_path: Path,
) -> None:
    """Verify every agent implements fit returning a CatalogFit instance."""
    runtime = _build_agent(agent, tmp_path)
    fit = runtime.fit(catalog, skills)
    assert isinstance(fit, CatalogFit)
    assert fit.whole
    assert fit.rations is runtime.rations_catalog


def test_a_runtime_that_rations_nothing_says_so_rather_than_saying_it_fits() -> None:
    """Verify CatalogFit.rations distinguishes unconstrained fits from measured limits."""
    silent = CatalogFit()
    assert silent.whole
    assert not silent.rations
    assert CatalogFit(allowed=30_000, asked=51_910, truncated=47).rations


def test_a_fit_is_whole_exactly_when_nothing_lost_its_description() -> None:
    """Verify CatalogFit.whole is True only when truncated is 0."""
    assert CatalogFit(allowed=100, asked=90).whole
    assert not CatalogFit(allowed=100, asked=900, truncated=1).whole


def test_a_fit_cannot_report_a_negative_measurement() -> None:
    """Verify CatalogFit raises ValidationError on negative count values."""
    with pytest.raises(PydanticValidationError):
        CatalogFit.model_validate({"allowed": 30_000, "asked": 51_910, "truncated": -1})


def test_the_fake_returns_the_fit_it_was_scripted_with(catalog, skills) -> None:
    """Verify FakeRuntime returns scripted CatalogFit and records fitting calls."""
    scripted = CatalogFit(allowed=10, asked=99, unit="columns", truncated=2, remedy="ask")
    runtime = FakeRuntime(fit=scripted)
    assert runtime.fit(catalog, skills) == scripted
    assert runtime.fittings == [catalog]


def test_model_profile_matches_by_name_prefix() -> None:
    """Verify model_profile resolves registry entries matching model prefix."""
    profile = model_profile("gemini-3.7-flash")
    assert profile is not None
    assert profile.chars_per_token > 0
    assert profile.context_window > 0


def test_model_profile_returns_defaults_for_an_unregistered_model() -> None:
    """Verify model_profile returns standard default ModelProfile for unregistered models."""
    profile = model_profile("some-future-model")
    assert profile.chars_per_token == 4.0
    assert profile.context_window == 1_048_576


def test_model_profile_is_overridden_by_project_config(tmp_path, monkeypatch) -> None:
    """Verify project reach.toml overrides existing model profile registry entries."""
    override = tmp_path / "reach.toml"
    override.write_text("[models.claude-opus-5]\nchars_per_token = 9.9\ncontext_window = 42\n")
    monkeypatch.chdir(tmp_path)
    profile = model_profile("claude-opus-5")
    assert profile is not None
    assert (profile.chars_per_token, profile.context_window) == (9.9, 42)


def test_model_profile_override_matches_regardless_of_key_case(tmp_path, monkeypatch) -> None:
    """Verify model profile TOML section names match case-insensitively."""
    override = tmp_path / "reach.toml"
    override.write_text("[models.Claude-Opus-5]\nchars_per_token = 9.9\ncontext_window = 42\n")
    monkeypatch.chdir(tmp_path)
    profile = model_profile("claude-opus-5")
    assert profile is not None
    assert (profile.chars_per_token, profile.context_window) == (9.9, 42)


def test_model_profile_defaults_are_sound() -> None:
    """Verify ModelProfile defaults chars_per_token to 4.0 and context_window to 1_048_576."""
    from reach.runtime.profiles import ModelProfile

    profile = ModelProfile()
    assert profile.chars_per_token == 4.0
    assert profile.context_window == 1_048_576


@pytest.mark.parametrize("agent", known_agents())
def test_every_agent_supports_prompt_budget_query(agent: str) -> None:
    """Verify prompt_budget_chars method returns None or positive integer across all generators."""
    gen = build_text_generator(agent=agent, options=MINIMAL_OPTIONS.get(agent, {}))
    budget = gen.prompt_budget_chars()
    assert budget is None or budget > 0


def test_fake_generator_custom_completion() -> None:
    """Verify FakeGenerator supports completions, budget override, and tracks counts."""
    generator = FakeGenerator(completion="custom completion response", prompt_budget_chars=500)
    assert generator.prompt_budget_chars() == 500
    assert generator.complete("prompt 1") == "custom completion response"
    assert generator.complete("prompt 2") == "custom completion response"
    assert generator.completions == 2
    assert generator.prompts == ["prompt 1", "prompt 2"]


def test_full_model_ids_have_profiles_in_reach_toml() -> None:
    """Verify full model IDs for Claude and Gemini resolve directly without aliasing."""
    for model_id in (
        "claude-sonnet-5",
        "claude-opus-5",
        "claude-haiku-4-5",
        "gemini-3.8-flash",
        "gemini-3.7-flash",
        "gemini-3.5-flash-lite",
        "gemini-3.1-pro-preview",
    ):
        profile = model_profile(model_id)
        assert profile is not None, f"Full model ID {model_id!r} lacks profile in reach.toml"
        assert profile.context_window > 0
        assert profile.chars_per_token > 0


def _assert_agent_default_model(agent_name: str) -> None:
    """Verify that an agent's default model has an accompanying profile."""
    default = agent_default_model(agent_name)
    if default:
        assert isinstance(default, str)
        profile = model_profile(default)
        assert profile is not None, f"default_model {default!r} for {agent_name!r} lacks a profile"


def test_production_defaults_are_internally_consistent() -> None:
    """Verify known agents have valid default models."""
    for agent_name in known_agents():
        _assert_agent_default_model(agent_name)


def test_fake_constants_consistency() -> None:
    """Verify FAKE_AGENT and FAKE_MODEL constants match fake runtime attributes."""
    from reach.runtime import FAKE_AGENT
    from reach.runtime.fake import FAKE_MODEL, FakeOptions, FakeRuntime

    assert FAKE_AGENT == "fake"
    assert FAKE_MODEL == "fake-model"
    assert FakeRuntime.name == FAKE_AGENT
    assert FakeOptions().model == FAKE_MODEL


def test_default_agent_reads_general_section_from_reach_toml() -> None:
    """Verify default_agent reads default_agent from bundled reach.toml."""
    assert default_agent() == "antigravity-cli"
    assert RuntimeSettings().agent == "antigravity-cli"


def test_default_agent_respects_project_config_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify default_agent and RuntimeSettings dynamically track reach.toml override."""
    override = tmp_path / "reach.toml"
    override.write_text("[general]\ndefault_agent = 'fake'\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    assert default_agent() == "fake"
    assert RuntimeSettings().agent == "fake"


@pytest.mark.usefixtures("fake_registry")
@pytest.mark.parametrize(
    ("model_name", "expected_agent"),
    [
        ("mock-model", "mock-agent"),
        ("MOCK-MODEL", "mock-agent"),
        ("mock-model-variant", "mock-agent"),
        ("mock-opus", "mock-agent"),
        ("mock-flash", "mock-antigravity"),
        ("mock-pro", "mock-antigravity"),
    ],
)
def test_find_agent_for_model_happy_paths(model_name: str, expected_agent: str) -> None:
    """Verify find_agent_for_model resolves models across case variations with fake registry."""
    assert find_agent_for_model(model_name) == expected_agent


@pytest.mark.usefixtures("fake_registry")
@pytest.mark.parametrize(
    "unknown_model",
    [
        "unregistered-model",
        "future-model",
        "",
        "   ",
    ],
)
def test_find_agent_for_model_sad_paths(unknown_model: str) -> None:
    """Verify find_agent_for_model returns None for unknown or empty model strings."""
    assert find_agent_for_model(unknown_model) is None


@pytest.mark.usefixtures("fake_registry")
def test_agent_default_model_with_fake_registry() -> None:
    """Verify agent_default_model returns correct default or None using fake registry."""
    assert agent_default_model("mock-agent") == "mock-model"
    assert agent_default_model("mock-antigravity") == "mock-flash"
    assert agent_default_model("mock-empty") is None
    assert agent_default_model("unregistered-agent") is None


def test_find_agent_for_model_edge_case_empty_or_corrupted_agents(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify find_agent_for_model handles empty or missing agents configuration gracefully."""
    override = tmp_path / "reach.toml"
    override.write_text("[agents]\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    assert find_agent_for_model("some-model") is None
    monkeypatch.setattr("reach.runtime.agent_profiles", lambda _=None: {})
    assert "claude-code" in known_agents()


def test_load_config_rejects_corrupted_config(
    tmp_path: Path,
) -> None:
    """Verify load_config fails fast with TOMLDecodeError on malformed syntax."""
    corrupted = tmp_path / "reach.toml"
    corrupted.write_text("this is not valid toml = [[[", encoding="utf-8")
    with pytest.raises(tomllib.TOMLDecodeError):
        load_config(corrupted)


def test_load_config_deep_merge_preserves_sibling_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify deep merge updates targeted field without overwriting sibling keys."""
    override = tmp_path / "reach.toml"
    override.write_text(
        "[agents.claude-code]\ndefault_model = 'custom-model'\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    assert agent_default_model("claude-code") == "custom-model"
    assert "antigravity-cli" in known_agents()


def test_process_failure_reason_extracts_trailing_stderr_or_exit_code() -> None:
    """Verify process_failure_reason pulls last line of stderr or formats exit code."""
    with_stderr = subprocess.CompletedProcess(
        args=["fake"],
        returncode=1,
        stdout="",
        stderr="line 1\nline 2: out of memory\n",
    )
    assert process_failure_reason(with_stderr) == "line 2: out of memory"

    empty_stderr = subprocess.CompletedProcess(
        args=["fake"],
        returncode=137,
        stdout="",
        stderr="",
    )
    assert process_failure_reason(empty_stderr) == "exit 137"


def test_antigravity_runtime_resolves_skills_dir_and_roots(tmp_path: Path) -> None:
    """Verify AntigravityRuntime provides .agents/skills layout and discovers existing roots."""
    runtime = build_runtime(
        RuntimeSettings(
            agent="antigravity-cli",
            options=MINIMAL_OPTIONS["antigravity-cli"],
        ),
    )
    assert isinstance(runtime, AntigravityRuntime)
    assert runtime.skills_dir(tmp_path) == tmp_path / ".agents" / "skills"
    assert runtime.skill_roots(tmp_path) == ()

    skills_dir = tmp_path / ".agents" / "skills"
    skills_dir.mkdir(parents=True)
    roots = runtime.skill_roots(tmp_path)
    assert len(roots) == 1
    assert roots[0].path == skills_dir.resolve()
    assert roots[0].scope == "project"
    assert roots[0].precedence == 0


def test_agent_runtime_default_model_reads_options_or_empty() -> None:
    """Verify AgentRuntime.model extracts model name from options or falls back to empty string."""
    fake = FakeRuntime(model="custom-gemini")
    assert fake.model == "custom-gemini"

    class BareRuntime(AgentRuntime[Any]):
        name = "bare"

        @override
        def skill_roots(self, workdir: Path) -> tuple[SkillRoot, ...]:
            del workdir
            return ()

        @override
        def install(self, catalog: Catalog, skills: Any, workdir: Path) -> Path:
            del catalog, skills
            return workdir

        @override
        def select(
            self,
            query_text: str,
            workdir: Path,
            target_skill: str | None = None,
        ) -> SelectionOutcome:
            del query_text, workdir, target_skill
            return SelectionOutcome()

        def complete(self, prompt: str) -> str:
            del prompt
            return ""

    bare = BareRuntime()
    assert bare.model == ""


@pytest.mark.parametrize("agent", known_agents())
def test_options_model_matches_registered_schema(agent: str) -> None:
    """Verify options_model returns a valid AgentOptions schema model across all agents."""
    from reach.runtime import AgentOptions

    model = options_model(agent)
    assert model is not None
    assert issubclass(model, AgentOptions)


@pytest.mark.parametrize("agent", known_agents())
def test_install_places_only_catalog_members(
    agent: str,
    skill_repo: Path,
    tmp_path: Path,
) -> None:
    """Verify install copies only catalog member skill directories into workspace."""
    runtime = _build_agent(agent, tmp_path)
    skills = load_skills(skill_repo)
    catalog = Catalog(
        id="test-subset",
        mode=CatalogMode.ALL,
        skills=("gcs-lifecycle-rules", "gcs-retention-policy"),
    )
    workdir = runtime.install(catalog, skills, tmp_path / "work")
    skills_dir = runtime.skills_dir(workdir)
    installed = sorted(p.name for p in skills_dir.iterdir())
    assert installed == ["gcs-lifecycle-rules", "gcs-retention-policy"]
    assert runtime._resident == catalog.skills


@pytest.mark.parametrize("agent", known_agents())
def test_install_is_idempotent(
    agent: str,
    skill_repo: Path,
    tmp_path: Path,
) -> None:
    """Verify consecutive install calls replace previously installed skills."""
    runtime = _build_agent(agent, tmp_path)
    skills = load_skills(skill_repo)
    catalogs = build_catalogs(skills, CatalogMode.SINGLETON)
    workdir = tmp_path / "work"
    runtime.install(catalogs[0], skills, workdir)
    runtime.install(catalogs[1], skills, workdir)
    skills_dir = runtime.skills_dir(workdir)
    installed = sorted(p.name for p in skills_dir.iterdir())
    assert installed == [catalogs[1].skills[0]]


@pytest.mark.parametrize("agent", known_agents())
def test_skill_roots_empty_before_install_and_present_after(
    agent: str,
    skill_repo: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify skill_roots is empty on clean workspace and discovered once catalog is installed."""
    monkeypatch.setattr(Path, "home", classmethod(lambda _cls: tmp_path / "nohome"))
    runtime = _build_agent(agent, tmp_path)
    workdir = tmp_path / "work"
    assert runtime.skill_roots(workdir) == ()

    skills = load_skills(skill_repo)
    catalog = build_catalogs(skills, CatalogMode.SINGLETON)[0]
    runtime.install(catalog, skills, workdir)
    roots = runtime.skill_roots(workdir)
    assert len(roots) == 1
    assert roots[0].path == runtime.skills_dir(workdir).resolve()
    assert roots[0].scope in ("project", "user")
    assert roots[0].precedence == 0


def test_install_skills_uses_symlinks_when_requested(
    skill_repo: Path,
    tmp_path: Path,
) -> None:
    """Verify install_skills creates fast symlinks when use_symlinks is enabled."""
    skills = load_skills(skill_repo)
    by_name = {s.name: s for s in skills}
    catalog = Catalog(id="c", mode=CatalogMode.SINGLETON, skills=(skills[0].name,))
    dest = tmp_path / "installed_skills"
    installed = install_skills(catalog, by_name, dest, use_symlinks=True)
    assert installed == (skills[0].name,)
    assert (dest / skills[0].name).is_symlink()


def test_install_skills_copies_when_use_symlinks_is_false(
    skill_repo: Path,
    tmp_path: Path,
) -> None:
    """Verify install_skills copies directories when use_symlinks is disabled."""
    skills = load_skills(skill_repo)
    by_name = {s.name: s for s in skills}
    catalog = Catalog(id="c", mode=CatalogMode.SINGLETON, skills=(skills[0].name,))
    dest = tmp_path / "installed_skills"
    installed = install_skills(catalog, by_name, dest, use_symlinks=False)
    assert installed == (skills[0].name,)
    assert not (dest / skills[0].name).is_symlink()
    assert (dest / skills[0].name / "SKILL.md").is_file()


def test_install_skills_falls_back_to_copy_on_symlink_error(
    skill_repo: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify install_skills falls back to copytree when symlink raises OSError."""

    def _failing_symlink(self: Path, _target: Path, target_is_directory: bool = False) -> None:
        msg = "symlink operation not permitted"
        raise OSError(msg)

    monkeypatch.setattr(Path, "symlink_to", _failing_symlink)
    skills = load_skills(skill_repo)
    by_name = {s.name: s for s in skills}
    catalog = Catalog(id="c", mode=CatalogMode.SINGLETON, skills=(skills[0].name,))
    dest = tmp_path / "installed_skills"
    installed = install_skills(catalog, by_name, dest, use_symlinks=True)
    assert installed == (skills[0].name,)
    assert not (dest / skills[0].name).is_symlink()
    assert (dest / skills[0].name / "SKILL.md").is_file()


def test_install_skills_rejects_path_traversal(
    skill_repo: Path,
    tmp_path: Path,
) -> None:
    """Verify install_skills raises ValueError if a skill name attempts directory escape."""
    skills = load_skills(skill_repo)
    by_name = {s.name: s for s in skills}
    catalog = Catalog.model_construct(id="c", mode=CatalogMode.SINGLETON, skills=("../escape",))
    by_name["../escape"] = skills[0]
    dest = tmp_path / "installed_skills"
    with pytest.raises(ValueError, match="escapes destination directory"):
        install_skills(catalog, by_name, dest)


@pytest.mark.parametrize(
    ("scenario", "expected_match"),
    [
        ("external_escape", "escaping skill directory"),
        ("relative_escape", "escaping skill directory"),
        ("broken_target", "broken or cyclical symlink"),
        ("self_root_loop", "escaping skill directory"),
        ("ancestor_cycle", "escaping skill directory"),
    ],
)
def test_install_skills_rejects_invalid_symlinks(
    scenario: str,
    expected_match: str,
    tmp_path: Path,
) -> None:
    """Verify install_skills rejects escaping, broken, self-referential, and cyclical symlinks."""
    skill_dir = tmp_path / "repo" / "skills" / "subdir" / "test-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: test-skill\ndescription: Test.\n---\nBody",
        encoding="utf-8",
    )
    secret_file = tmp_path / "repo" / "secret.txt"
    secret_file.write_text("super_secret", encoding="utf-8")

    if scenario == "external_escape":
        (skill_dir / "leak").symlink_to(secret_file)
    elif scenario == "relative_escape":
        (skill_dir / "escape_link").symlink_to(Path("../../../secret.txt"))
    elif scenario == "broken_target":
        (skill_dir / "missing_target").symlink_to(skill_dir / "non_existent.txt")
    elif scenario == "self_root_loop":
        (skill_dir / "loop").symlink_to(Path())
    elif scenario == "ancestor_cycle":
        sub_dir = skill_dir / "nested" / "deep"
        sub_dir.mkdir(parents=True)
        (sub_dir / "back_to_nested").symlink_to(Path(".."))

    skill = Skill(name="test-skill", description="Test", path=skill_dir)
    catalog = Catalog(id="c", mode=CatalogMode.SINGLETON, skills=("test-skill",))
    with pytest.raises(ValueError, match=expected_match):
        install_skills(catalog, {"test-skill": skill}, tmp_path / "installed_skills")


def test_install_skills_allows_valid_internal_symlinks(
    tmp_path: Path,
) -> None:
    """Verify install_skills cleanly installs skills with valid intra-directory symlinks."""
    skill_dir = tmp_path / "valid_symlink_skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nname: symlink-skill\ndescription: Symlink skill.\n---\nBody",
        encoding="utf-8",
    )
    docs_dir = skill_dir / "docs"
    docs_dir.mkdir()
    (docs_dir / "guide.md").write_text("Valid guide documentation.", encoding="utf-8")
    internal_link = skill_dir / "README.md"
    internal_link.symlink_to(docs_dir / "guide.md")

    skill = Skill(name="symlink-skill", description="Symlink skill", path=skill_dir)
    catalog = Catalog(id="c", mode=CatalogMode.SINGLETON, skills=("symlink-skill",))
    by_name = {"symlink-skill": skill}
    dest = tmp_path / "installed_skills"

    installed = install_skills(catalog, by_name, dest)
    assert "symlink-skill" in installed
    installed_skill = dest / "symlink-skill"
    assert (installed_skill / "SKILL.md").exists()
    assert (installed_skill / "README.md").exists()


@pytest.mark.parametrize("agent", known_agents())
def test_install_uses_symlinks_by_default(
    agent: str,
    skill_repo: Path,
    tmp_path: Path,
) -> None:
    """Verify install creates fast directory symlinks for resident skills by default."""
    runtime = _build_agent(agent, tmp_path)
    skills = load_skills(skill_repo)
    catalog = build_catalogs(skills, CatalogMode.SINGLETON)[0]
    workdir = runtime.install(catalog, skills, tmp_path / f"work_{agent}")
    skill_entry = runtime.skills_dir(workdir) / catalog.skills[0]
    if isinstance(runtime, AntigravityRuntime):
        assert not skill_entry.is_symlink()
    else:
        assert skill_entry.is_symlink()


@pytest.mark.parametrize("agent", known_agents())
def test_install_copies_when_use_symlinks_is_false(
    agent: str,
    skill_repo: Path,
    tmp_path: Path,
) -> None:
    """Verify install copies skill files directly when use_symlinks is disabled."""
    runtime = _build_agent(agent, tmp_path, use_symlinks=False)
    skills = load_skills(skill_repo)
    catalog = build_catalogs(skills, CatalogMode.SINGLETON)[0]
    workdir = runtime.install(catalog, skills, tmp_path / f"work_{agent}")
    skill_entry = runtime.skills_dir(workdir) / catalog.skills[0]
    assert not skill_entry.is_symlink()


@pytest.mark.parametrize("agent", antigravity_agents())
def test_antigravity_agents_can_enable_use_symlinks(
    agent: str,
    skill_repo: Path,
    tmp_path: Path,
) -> None:
    """Verify Antigravity runtimes create symlinks when use_symlinks is explicitly set to True."""
    runtime = _build_agent(agent, tmp_path, use_symlinks=True)
    assert runtime.use_symlinks is True
    skills = load_skills(skill_repo)
    catalog = build_catalogs(skills, CatalogMode.SINGLETON)[0]
    workdir = runtime.install(catalog, skills, tmp_path / f"work_{agent}_symlinks")
    skill_entry = runtime.skills_dir(workdir) / catalog.skills[0]
    assert skill_entry.is_symlink()


@pytest.mark.parametrize("agent", known_agents())
def test_install_falls_back_to_copy_on_symlink_error(
    agent: str,
    skill_repo: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify install falls back to shutil.copytree when symlink creation raises OSError."""

    def _failing_symlink(self: Path, _target: Path, target_is_directory: bool = False) -> None:
        msg = "symlink operation not permitted"
        raise OSError(msg)

    monkeypatch.setattr(Path, "symlink_to", _failing_symlink)
    runtime = _build_agent(agent, tmp_path)
    skills = load_skills(skill_repo)
    catalog = build_catalogs(skills, CatalogMode.SINGLETON)[0]
    workdir = runtime.install(catalog, skills, tmp_path / f"work_{agent}")
    skill_entry = runtime.skills_dir(workdir) / catalog.skills[0]
    assert not skill_entry.is_symlink()
    assert (skill_entry / "SKILL.md").is_file()


@pytest.mark.parametrize("agent", known_agents())
def test_prompt_budget_uses_default_for_unregistered_model(agent: str) -> None:
    """Verify prompt_budget_chars returns standard default budget on unregistered model."""
    gen = build_text_generator(agent=agent, model="unmeasured-future-model-999")
    assert gen.prompt_budget_chars() == 4_194_304


@pytest.mark.parametrize("agent", [a for a in known_agents() if agent_default_model(a)])
def test_prompt_budget_is_measured_for_recognized_models(agent: str) -> None:
    """Verify prompt_budget_chars returns positive integer for registered model profiles."""
    gen = build_text_generator(agent=agent)
    budget = gen.prompt_budget_chars()
    assert budget is not None
    assert budget > 0


@pytest.mark.parametrize("agent", cli_agents())
def test_cli_executable_is_configurable(agent: str, tmp_path: Path) -> None:
    """Verify custom executable option is placed as the first command token."""
    runtime = _build_agent(agent, tmp_path, executable="/custom/binary")
    assert isinstance(runtime, CliAgentRuntime)
    cmd = runtime.build_command("test query")
    assert cmd[0] == "/custom/binary"


@pytest.mark.parametrize("agent", cli_agents())
def test_cli_extra_args_reach_command_line_last(agent: str, tmp_path: Path) -> None:
    """Verify extra_args tokens are placed at the end of the constructed command line."""
    runtime = _build_agent(agent, tmp_path, extra_args=("--custom-flag", "value"))
    assert isinstance(runtime, CliAgentRuntime)
    cmd = runtime.build_command("test query")
    assert cmd[-2:] == ["--custom-flag", "value"]


@pytest.mark.parametrize("agent", cli_agents())
def test_cli_runtime_select_executes_in_workdir(
    agent: str,
    mock_subprocess: Callable[..., Any],
    tmp_path: Path,
) -> None:
    """Verify select executes subprocess with cwd pointing to the provided workspace."""
    runtime = _build_agent(agent, tmp_path)
    workdir = tmp_path / "work"
    workdir.mkdir(parents=True, exist_ok=True)
    seen: dict[str, Any] = {}

    def mock_run(args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        seen.update(kwargs)
        return subprocess.CompletedProcess(args=args, returncode=1, stdout="", stderr="abort")

    mock_subprocess(handler=mock_run)
    runtime.select("test query", workdir)
    assert seen.get("cwd") == workdir


@pytest.mark.parametrize("agent", cli_agents())
def test_cli_runtime_select_handles_timeout(
    agent: str,
    mock_subprocess: Callable[..., Any],
    tmp_path: Path,
) -> None:
    """Verify select handles subprocess timeout by returning error outcome without raising."""
    runtime = _build_agent(agent, tmp_path)
    mock_subprocess(side_effect=subprocess.TimeoutExpired(cmd="fake", timeout=1))
    workdir = tmp_path / "work"
    workdir.mkdir(parents=True, exist_ok=True)
    outcome = runtime.select("query", workdir)
    assert outcome.invoked_skill is None
    assert outcome.error is not None
    assert "time" in outcome.error.lower()


@pytest.mark.parametrize("agent", cli_agents())
def test_cli_runtime_select_handles_spawn_oserror(
    agent: str,
    mock_subprocess: Callable[..., Any],
    tmp_path: Path,
) -> None:
    """Verify select handles subprocess spawn failure (e.g. missing binary) gracefully."""
    runtime = _build_agent(agent, tmp_path)
    mock_subprocess(
        side_effect=FileNotFoundError(2, "No such file or directory: 'fake-executable'"),
    )
    workdir = tmp_path / "work"
    workdir.mkdir(parents=True, exist_ok=True)
    outcome = runtime.select("query", workdir)
    assert outcome.invoked_skill is None
    assert outcome.error is not None
    assert "failed to spawn" in outcome.error or "No such file" in outcome.error


def test_skills_subpath_fallback_and_override() -> None:
    """Verify custom runtime subclasses can override _skills_subpath or fallback to empty."""

    class CustomRuntime(AgentRuntime[Any]):
        name = "unregistered-agent"

        @override
        def select(
            self,
            query_text: str,
            workdir: Path,
            target_skill: str | None = None,
        ) -> SelectionOutcome:
            del query_text, workdir, target_skill
            return SelectionOutcome()

        def complete(self, prompt: str) -> str:
            del prompt
            return ""

    class ExplicitRuntime(AgentRuntime[Any]):
        name = "explicit-agent"
        _skills_subpath = "custom/path"

        @override
        def select(
            self,
            query_text: str,
            workdir: Path,
            target_skill: str | None = None,
        ) -> SelectionOutcome:
            del query_text, workdir, target_skill
            return SelectionOutcome()

        def complete(self, prompt: str) -> str:
            del prompt
            return ""

    assert CustomRuntime().skills_subpath == ""
    assert ExplicitRuntime().skills_subpath == "custom/path"


def test_check_tool_leak() -> None:
    """Verify check_tool_leak detects unauthorized tool attempts."""
    assert check_tool_leak(["read", "bash"], None) is None
    assert check_tool_leak(["read", "write"], ["read", "write", "glob"]) is None
    leak_err = check_tool_leak(["read", "bash", "curl"], ["read"])
    assert leak_err == "tool leak: bash, curl"


def test_session_summary_to_outcome() -> None:
    """Verify SessionSummary converts cleanly to SelectionOutcome with sync and fallback."""
    assert issubclass(SelectionOutcome, SessionSummary)
    summary = SessionSummary(
        invoked_skills=("pizza-calculator",),
        reasoning=("thought 1",),
        observed_tools=("load_skill",),
        cost_usd=0.005,
        resolved_model="",
    )
    assert summary.invoked_skills == ("pizza-calculator",)
    outcome = summary.to_outcome(
        observed_catalog=("pizza-calculator", "cloud-deploy"),
        fallback_model="default-model",
    )
    assert outcome.invoked_skill == "pizza-calculator"
    assert outcome.invoked_skills == ("pizza-calculator",)
    assert outcome.reasoning == ("thought 1",)
    assert outcome.observed_catalog == ("pizza-calculator", "cloud-deploy")
    assert outcome.observed_tools == ("load_skill",)
    assert outcome.resolved_model == "default-model"
    assert outcome.cost_usd == pytest.approx(0.005)


def test_agent_runtime_timeout_s() -> None:
    """Verify AgentRuntime exposes timeout_s from settings."""
    settings_with = RuntimeSettings(agent="fake", timeout_s=42)
    rt_with = build_runtime(settings_with)
    assert rt_with.timeout_s == 42

    rt_none = FakeRuntime()
    assert rt_none.timeout_s is None


@pytest.mark.parametrize("agent", cli_agents())
def test_cli_agent_reports_subprocess_failure(
    agent: str,
    mock_subprocess: Callable[..., Any],
    tmp_path: Path,
) -> None:
    """Verify CLI runtimes capture non-zero subprocess returncode and report failure reason."""
    runtime = _build_agent(agent, tmp_path)
    mock_subprocess(returncode=1, stderr="Error: connection refused")
    outcome = runtime.select("q", tmp_path / "work")
    assert outcome.error is not None
    assert "connection refused" in outcome.error


@pytest.mark.parametrize(
    ("path_arg", "expected"),
    [
        ("/path/to/cloud-deploy/SKILL.md", "cloud-deploy"),
        ("cloud-deploy/skill.md", "cloud-deploy"),
        ("/path/to/.agents/skills/cloud-deploy", "cloud-deploy"),
        ("/path/to/.agents/skills/CLOUD-DEPLOY/", "cloud-deploy"),
        ("/path/to/.agents/skills/cloud-deploy/references/guide.md", "cloud-deploy"),
        ("/path/to/.agents/skills/cloud-deploy/scripts/run.sh", "cloud-deploy"),
        ("/path/to/.agents/skills/cloud-deploy/references", "cloud-deploy"),
        ("PIZZA-CALCULATOR.MD", "pizza-calculator"),
        (Path("/skills/pizza-calculator.md"), "pizza-calculator"),
        ("/other/README.md", None),
        ("/other/non-resident-dir", None),
        ("/path/to/cloud-deploy/scripts/run.py", None),
        ("", None),
        (None, None),
        (123, None),
    ],
)
def test_resolve_skill_from_path_variations(path_arg: Any, expected: str | None) -> None:
    """Verify resolve_skill_from_path handles paths, bare skill dirs, casing, and edge cases."""
    residents = ["cloud-deploy", "pizza-calculator"]
    assert resolve_skill_from_path(path_arg, residents) == expected


def test_extract_content_reasoning() -> None:
    """Verify extract_content_reasoning extracts non-empty thoughts and text."""
    content = [
        {"type": "thought", "thought": "Thinking step 1"},
        {"type": "thinking", "text": "Thinking step 2"},
        {"type": "thinking", "thinking": "Thinking step 3 (Pi format)"},
        {"type": "thinking", "thinking": "   "},
        {"type": "thought", "thought": ""},
        {"type": "text", "text": "Direct response text"},
        {"type": "text", "text": "   "},
        {"type": "toolCall", "name": "read"},
        "invalid_item",
    ]
    extracted = extract_content_reasoning(content)
    assert extracted == [
        "Thinking step 1",
        "Thinking step 2",
        "Thinking step 3 (Pi format)",
        "Direct response text",
    ]


def test_format_subprocess_error() -> None:
    """Verify format_subprocess_error formats timeout, spawn error, and generic failures."""
    assert format_subprocess_error("pi", "timeout", 30) == "pi process timed out after 30s"
    assert format_subprocess_error("goose", "timeout", None) == "goose process timed out"
    assert (
        format_subprocess_error("pi", "executable not found", 10)
        == "failed to spawn pi: executable not found"
    )
    assert format_subprocess_error("pi", None, 10) == "pi subprocess failed"


def test_cli_options_helpers() -> None:
    """Verify CliOptions effort_args, provider_args, max_turns_args, and api_key_args helpers."""
    opts = CliOptions(
        executable="test-cli",
        model="test-model",
        effort="high",
        provider="google",
        max_turns=3,
        api_key="secret-123",
    )
    assert opts.effort == "high"
    assert opts.provider == "google"
    assert opts.max_turns == 3
    assert opts.api_key == "secret-123"
    assert opts.effort_args("--effort") == ["--effort", "high"]
    assert opts.provider_args("--provider") == ["--provider", "google"]
    assert opts.max_turns_args("--max-turns") == ["--max-turns", "3"]
    assert opts.api_key_args("--api-key") == ["--api-key", "secret-123"]

    opts_empty = CliOptions(executable="test-cli", model="test-model")
    assert opts_empty.effort_args("--effort") == []
    assert opts_empty.provider_args("--provider") == []
    assert opts_empty.max_turns_args("--max-turns") == ["--max-turns", "3"]
    assert opts_empty.api_key_args("--api-key") == []


def test_agent_options_defaults() -> None:
    """Verify AgentOptions default values for turn budget, isolation, and performance."""
    from reach.runtime import AntigravityOptions

    opts = AgentOptions()
    assert opts.max_turns == 3
    assert opts.early_exit is True
    assert opts.use_symlinks is True
    assert opts.isolate_config_dir is True
    assert opts.auto_clean is False
    assert AntigravityOptions().use_symlinks is False


@pytest.mark.parametrize("agent", known_agents())
def test_all_agents_shared_options_conformance(agent: str, tmp_path: Path) -> None:
    """Verify all agent drivers expose common options and properties uniformly."""
    default_rt = _build_agent(agent, tmp_path)
    expected_symlinks = not isinstance(default_rt, AntigravityRuntime)
    assert default_rt.max_turns == 3
    assert default_rt.early_exit is True
    assert default_rt.use_symlinks is expected_symlinks
    assert default_rt.isolate_config_dir is True
    assert default_rt.auto_clean is False

    rt = _build_agent(
        agent,
        tmp_path,
        model="custom-model",
        effort="medium",
        provider="custom-provider",
        max_turns=2,
        early_exit=False,
        api_key="secret-key",
        allowed_tools=("custom-tool",),
        use_symlinks=False,
        isolate_config_dir=False,
        auto_clean=True,
    )
    assert rt.model == "custom-model"
    assert rt.effort == "medium"
    assert rt.provider == "custom-provider"
    assert rt.max_turns == 2
    assert rt.early_exit is False
    assert rt.api_key == "secret-key"
    assert rt.allowed_tools == ("custom-tool",)
    assert rt.use_symlinks is False
    assert rt.isolate_config_dir is False
    assert rt.auto_clean is True


@pytest.mark.parametrize("agent", known_agents())
def test_all_agents_select_invokes_post_probe(
    agent: str,
    tmp_path: Path,
    mock_subprocess: Callable[..., Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify select invokes post_probe lifecycle hook upon completion across all agents."""
    mock_subprocess(stdout="")
    rt_cls = runtime_class(agent)
    if rt_cls is not None and hasattr(rt_cls, "_select_async"):

        async def _mock_select_async(*_args: Any, **_kwargs: Any) -> SelectionOutcome:
            return SelectionOutcome()

        monkeypatch.setattr(rt_cls, "_select_async", _mock_select_async)

    runtime = _build_agent(agent, tmp_path)
    called_workdirs: list[Path] = []

    def mock_post_probe(workdir: Path) -> None:
        called_workdirs.append(workdir)

    monkeypatch.setattr(runtime, "post_probe", mock_post_probe)
    workdir = tmp_path / f"work_{agent}"
    workdir.mkdir(parents=True, exist_ok=True)

    with contextlib.suppress(Exception):
        runtime.select("test query", workdir)

    assert workdir in called_workdirs


@pytest.mark.parametrize(
    "agent",
    _KEY_SYNC_AGENTS,
)
def test_agent_build_env_synchronizes_google_and_gemini_keys(
    agent: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify build_env synchronizes GEMINI_API_KEY and GOOGLE_API_KEY bidirectionally."""
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-secret")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    rt = _build_agent(agent, tmp_path)
    env = rt.build_env()
    assert env["GEMINI_API_KEY"] == "gemini-secret"
    assert env["GOOGLE_API_KEY"] == "gemini-secret"

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "google-secret")
    rt2 = _build_agent(agent, tmp_path)
    env2 = rt2.build_env()
    assert env2["GEMINI_API_KEY"] == "google-secret"
    assert env2["GOOGLE_API_KEY"] == "google-secret"


@pytest.mark.parametrize(
    "agent",
    known_agents(),
)
def test_agent_build_env_sanitizes_ambient_sensitive_variables(
    agent: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify build_env strips ambient sensitive credentials from child process environment."""
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "unwanted-aws-secret")
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_unwanted_token")
    monkeypatch.setenv("SSH_AUTH_SOCK", str(tmp_path / "ssh.sock"))
    monkeypatch.setenv("GEMINI_API_KEY", "valid-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "valid-key")
    monkeypatch.setenv("OPENAI_API_KEY", "valid-key")
    rt = _build_agent(agent, tmp_path)
    env = rt.build_env()
    assert "AWS_SECRET_ACCESS_KEY" not in env
    assert "GITHUB_TOKEN" not in env
    assert "SSH_AUTH_SOCK" not in env
    assert env.get("ANTHROPIC_API_KEY") == "valid-key"
    assert env.get("OPENAI_API_KEY") == "valid-key"


def test_sanitize_subprocess_env_custom_override() -> None:
    """Verify sanitize_subprocess_env with explicit blocked_env_vars overrides defaults."""
    k_aws = "AWS_SECRET_ACCESS_KEY"
    k_gh = "GITHUB_TOKEN"
    env = {
        k_aws: "val-1",
        k_gh: "val-2",
        "CUSTOM_VAR": "val-3",
    }
    # Only CUSTOM_VAR should be stripped
    result = sanitize_subprocess_env(dict(env), blocked_env_vars=("CUSTOM_VAR",))
    assert "CUSTOM_VAR" not in result
    assert result[k_aws] == env[k_aws]
    assert result[k_gh] == env[k_gh]

    # Empty tuple means nothing is stripped
    result_empty = sanitize_subprocess_env(dict(env), blocked_env_vars=())
    assert result_empty == env


@pytest.mark.parametrize(
    "agent",
    [
        "antigravity-cli",
        "claude-code",
        "antigravity-sdk",
    ],
)
def test_agent_build_env_honors_custom_blocked_env_vars(
    agent: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify build_env honors explicit blocked_env_vars configured in settings."""
    monkeypatch.setenv("GITHUB_TOKEN", "keep-me")
    monkeypatch.setenv("CUSTOM_BLOCKED_VAR", "strip-me")
    settings = RuntimeSettings(
        agent=agent,
        blocked_env_vars=("CUSTOM_BLOCKED_VAR",),
    )
    rt = build_runtime(settings)
    env = rt.build_env(tmp_path)
    assert "CUSTOM_BLOCKED_VAR" not in env
    assert env.get("GITHUB_TOKEN") == "keep-me"


@pytest.mark.parametrize(
    ("flag_key", "flag_val", "expected_retained"),
    [
        ("CLAUDE_CODE_USE_VERTEX", "1", True),
        ("GOOGLE_GENAI_USE_ENTERPRISE", "true", True),
        ("CLAUDE_CODE_USE_VERTEX", "0", False),
        ("OTHER_ENV_VAR", "1", False),
    ],
)
def test_sanitize_subprocess_env_google_application_credentials_exemption(
    flag_key: str,
    flag_val: str,
    *,
    expected_retained: bool,
) -> None:
    """Verify GOOGLE_APPLICATION_CREDENTIALS is preserved under Vertex or Enterprise mode."""
    env = {
        "GOOGLE_APPLICATION_CREDENTIALS": "/path/to/sa.json",
        flag_key: flag_val,
    }
    result = sanitize_subprocess_env(env)
    if expected_retained:
        assert result.get("GOOGLE_APPLICATION_CREDENTIALS") == "/path/to/sa.json"
    else:
        assert "GOOGLE_APPLICATION_CREDENTIALS" not in result


def test_sync_claude_settings_env_loads_settings_file(tmp_path: Path) -> None:
    """Verify sync_claude_settings_env populates environment variables from settings.json."""
    settings_file = tmp_path / "settings.json"
    settings_file.write_text(
        json.dumps(
            {
                "env": {
                    "CLAUDE_CODE_USE_VERTEX": "1",
                    "ANTHROPIC_VERTEX_PROJECT_ID": "my-vertex-project",
                }
            }
        ),
        encoding="utf-8",
    )
    env: dict[str, str] = {}
    result = sync_claude_settings_env(env, claude_home=tmp_path)
    assert result["CLAUDE_CODE_USE_VERTEX"] == "1"
    assert result["ANTHROPIC_VERTEX_PROJECT_ID"] == "my-vertex-project"
    assert result["CLOUD_ML_REGION"] == "global"


def test_sync_claude_settings_env_preserves_google_application_credentials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify sync_claude_settings_env restores credentials when Vertex is enabled."""
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/path/to/creds.json")
    settings_file = tmp_path / "settings.json"
    settings_file.write_text(
        json.dumps({"env": {"CLAUDE_CODE_USE_VERTEX": "1"}}),
        encoding="utf-8",
    )
    env: dict[str, str] = {}
    result = sync_claude_settings_env(env, claude_home=tmp_path)
    assert result["GOOGLE_APPLICATION_CREDENTIALS"] == "/path/to/creds.json"
    assert result["CLOUD_ML_REGION"] == "global"


def test_sync_claude_settings_env_respects_blocked_env_vars(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify sync_claude_settings_env blocks specified credentials even when Vertex is active."""
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/path/to/creds.json")
    settings_file = tmp_path / "settings.json"
    settings_file.write_text(
        json.dumps(
            {
                "env": {
                    "CLAUDE_CODE_USE_VERTEX": "1",
                    "CUSTOM_SECRET": "blocked-value",
                    "ALLOWED_VAR": "kept-value",
                }
            }
        ),
        encoding="utf-8",
    )
    env: dict[str, str] = {}
    result = sync_claude_settings_env(
        env,
        claude_home=tmp_path,
        blocked_env_vars=["GOOGLE_APPLICATION_CREDENTIALS", "CUSTOM_SECRET"],
    )
    assert result["ALLOWED_VAR"] == "kept-value"
    assert "CUSTOM_SECRET" not in result
    assert "GOOGLE_APPLICATION_CREDENTIALS" not in result
    assert result["CLOUD_ML_REGION"] == "global"


def test_sync_claude_settings_env_handles_missing_or_corrupt_file(tmp_path: Path) -> None:
    """Verify sync_claude_settings_env handles missing or malformed settings.json gracefully."""
    env = {"EXISTING": "1"}
    assert sync_claude_settings_env(env, claude_home=tmp_path) == {"EXISTING": "1"}

    bad_file = tmp_path / "settings.json"
    bad_file.write_text("{not valid json", encoding="utf-8")
    assert sync_claude_settings_env(env, claude_home=tmp_path) == {"EXISTING": "1"}


def test_subprocess_probe_preserves_caller_configured_env(
    mock_subprocess: Callable[..., Any],
    tmp_path: Path,
) -> None:
    """Verify run_subprocess_probe preserves explicitly allowed variables in caller env."""
    from reach.runtime._subprocess import run_subprocess_probe

    seen: dict[str, Any] = {}

    def record(args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        seen.update(kwargs)
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="hello\n", stderr="")

    mock_subprocess(handler=record)
    caller_env = {
        "PATH": os.environ.get("PATH", ""),
        "GITHUB_TOKEN": "custom-permitted-token",
    }
    completed, err = run_subprocess_probe(
        ["echo", "hello"],
        workdir=tmp_path,
        env=caller_env,
    )
    assert err is None
    assert completed is not None
    assert completed.returncode == 0
    assert seen["env"] == caller_env


@pytest.mark.parametrize(
    ("agent", "dir_names"),
    [
        ("claude-code", [".reach_claude_config"]),
        ("goose", [".reach_goose"]),
        ("pi", [".reach_pi_sessions", ".reach_pi_agent"]),
        ("antigravity-sdk", [".reach_antigravity_sdk"]),
    ],
)
def test_isolated_config_dir_cleanup_on_auto_clean(
    agent: str,
    dir_names: list[str],
    tmp_path: Path,
) -> None:
    """Verify post_probe removes isolated runtime directories when auto_clean is enabled."""
    workdir = tmp_path / f"work_{agent}"
    workdir.mkdir(parents=True, exist_ok=True)
    dirs = [workdir / name for name in dir_names]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

    # auto_clean=False preserves directories
    rt_no_clean = _build_agent(agent, tmp_path, auto_clean=False)
    rt_no_clean.post_probe(workdir)
    for d in dirs:
        assert d.exists()

    # auto_clean=True removes directories
    rt_clean = _build_agent(agent, tmp_path, auto_clean=True)
    rt_clean.post_probe(workdir)
    for d in dirs:
        assert not d.exists()


@pytest.mark.parametrize(
    ("agent", "expected_prefix"),
    [
        pytest.param("goose", ["goose", "run", "-q", "-i", "-", "--no-session"], id="goose-stdin"),
        pytest.param("pi", ["pi", "-p", "--no-session"], id="pi-stdin"),
    ],
)
def test_stdin_cli_generators_pipe_large_prompts_with_non_interactive_flags(
    agent: str,
    expected_prefix: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify stdin CLI generators pass non-interactive flags and pipe >128KB prompts via stdin."""
    gen = build_text_generator(agent=agent)
    large_prompt = "x" * 150_000
    captured: dict[str, Any] = {}

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        captured["command"] = command
        captured["input"] = kwargs.get("input")
        return subprocess.CompletedProcess(
            args=command, returncode=0, stdout='{"queries": []}', stderr=""
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    res = gen.complete(large_prompt)
    assert res == '{"queries": []}'
    assert captured["input"] == large_prompt
    assert captured["command"][: len(expected_prefix)] == expected_prefix
    assert large_prompt not in captured["command"]


@pytest.mark.parametrize("agent", [a for a in known_agents() if agent_default_model(a)])
def test_build_text_generator_resolves_agent_default_model(agent: str) -> None:
    """Verify build_text_generator resolves agent-specific default model when none is passed."""
    expected_model = agent_default_model(agent)
    assert expected_model is not None
    gen = build_text_generator(agent=agent)
    assert gen.model == expected_model
    assert gen.options.model == expected_model


def test_build_text_generator_explicit_model_overrides_agent_default() -> None:
    """Verify explicit model parameter overrides the agent runtime default model."""
    gen = build_text_generator(agent="claude-code", model="claude-opus-5")
    assert gen.model == "claude-opus-5"
    assert gen.options.model == "claude-opus-5"


def test_build_text_generator_options_model_is_respected() -> None:
    """Verify options dictionary model is respected when model argument is omitted."""
    gen = build_text_generator(agent="claude-code", options={"model": "claude-haiku-4-5"})
    assert gen.model == "claude-haiku-4-5"
    assert gen.options.model == "claude-haiku-4-5"


def test_agent_options_inheritance_hierarchy() -> None:
    """Verify AgentOptions is the root model and all registered agent options subclass it."""
    from reach.runtime import AgentOptions, AntigravityOptions, CliOptions

    assert issubclass(CliOptions, AgentOptions)
    assert issubclass(AntigravityOptions, AgentOptions)

    for agent in known_agents():
        opt_cls = options_model(agent)
        assert opt_cls is not None
        assert issubclass(opt_cls, AgentOptions)

    # AgentOptions has core routing, isolation, and performance flags
    base = AgentOptions(
        model="gpt-4o",
        effort="low",
        provider="openai",
        max_turns=5,
        early_exit=False,
        allowed_tools=("search",),
        api_key="key-123",
    )
    assert base.model == "gpt-4o"
    assert base.effort == "low"
    assert base.max_turns == 5
    assert base.early_exit is False
    assert base.use_symlinks is True
    assert base.isolate_config_dir is True
    assert base.auto_clean is False

    # All options classes inherit performance and isolation defaults
    for agent in known_agents():
        opt_cls = options_model(agent)
        assert opt_cls is not None
        assert issubclass(opt_cls, AgentOptions)
        instance = opt_cls()
        expected_symlinks = not issubclass(opt_cls, AntigravityOptions)
        assert instance.use_symlinks is expected_symlinks
        assert instance.isolate_config_dir is True
        assert instance.auto_clean is False

    # Non-CLI options do not expose executable or extra_args
    for agent in set(known_agents()) - set(cli_agents()):
        opt_cls = options_model(agent)
        assert opt_cls is not None
        assert issubclass(opt_cls, AgentOptions)
        instance = opt_cls()
        assert not hasattr(instance, "executable")
        assert not hasattr(instance, "extra_args")


@pytest.mark.parametrize(
    ("early_exit", "turns_taken"),
    [
        (True, 1),
        (True, 2),
        (False, 3),
    ],
)
def test_selection_outcome_and_probe_result_early_exit_propagation(
    catalog: Catalog,
    early_exit: bool,
    turns_taken: int,
) -> None:
    """Verify early_exit and turns_taken propagate faithfully to ProbeResult."""
    from reach.models import ProbeResult, Query

    outcome = SelectionOutcome(
        invoked_skills=("target-skill", "precursor")[:turns_taken],
        early_exit=early_exit,
        turns_taken=turns_taken,
        observed_catalog=catalog.skills,
        observed_tools=("Skill",),
    )
    query = Query(id="q1", text="test query", expected_skill="target-skill")
    result = ProbeResult.from_outcome(
        outcome=outcome,
        query=query,
        catalog=catalog,
        runtime_name="test-runtime",
        model="test-model",
    )
    assert result.early_exit is early_exit
    assert result.turns_taken == turns_taken
    assert result.invoked_skill == "target-skill"


# --- Trajectory tracking and session telemetry -------------------------------


def test_trajectory_tracker_direct_hit() -> None:
    """Verify TrajectoryTracker signals immediate early exit upon target detection."""
    from reach.runtime import TrajectoryTracker

    tracker = TrajectoryTracker(target_skill="cloud-sql", max_turns=3, early_exit=True)
    stop = tracker.observe("cloud-sql")
    assert stop
    assert tracker.early_exit_hit
    assert tracker.turns_taken == 1
    assert tracker.invoked_skills == ["cloud-sql"]


def test_trajectory_tracker_multi_turn_recovery() -> None:
    """Verify TrajectoryTracker records intermediate precursor skills and stops on target."""
    from reach.runtime import TrajectoryTracker

    tracker = TrajectoryTracker(target_skill="cloud-sql", max_turns=3, early_exit=True)
    assert not tracker.observe("gcloud")
    assert not tracker.early_exit_hit
    assert tracker.turns_taken == 1

    assert tracker.observe("cloud-sql")
    assert tracker.early_exit_hit
    assert tracker.turns_taken == 2
    assert tracker.invoked_skills == ["gcloud", "cloud-sql"]


def test_trajectory_tracker_deduplicates_consecutive_tool_calls() -> None:
    """Verify multiple consecutive calls to the same skill do not advance turn budget."""
    from reach.runtime import TrajectoryTracker

    tracker = TrajectoryTracker(target_skill="cloud-sql", max_turns=3, early_exit=True)
    assert not tracker.observe("gcloud")
    assert not tracker.observe("gcloud")  # consecutive call to same skill
    assert tracker.turns_taken == 1
    assert tracker.invoked_skills == ["gcloud"]

    assert tracker.observe("cloud-sql")
    assert tracker.turns_taken == 2
    assert tracker.invoked_skills == ["gcloud", "cloud-sql"]


def test_trajectory_tracker_exhausts_max_turns() -> None:
    """Verify TrajectoryTracker stops when turn budget is exhausted without target hit."""
    from reach.runtime import TrajectoryTracker

    tracker = TrajectoryTracker(target_skill="cloud-sql", max_turns=3, early_exit=True)
    assert not tracker.observe("skill-1")
    assert not tracker.observe("skill-2")
    assert tracker.observe("skill-3")  # 3rd turn reached
    assert tracker.early_exit_hit
    assert tracker.turns_taken == 3
    assert tracker.invoked_skills == ["skill-1", "skill-2", "skill-3"]


def test_trajectory_tracker_disabled_early_exit() -> None:
    """Verify TrajectoryTracker records all skills and truncates outcome when early_exit=False."""
    from reach.runtime import TrajectoryTracker

    tracker = TrajectoryTracker(target_skill="cloud-sql", max_turns=2, early_exit=False)
    assert not tracker.observe("cloud-sql")
    assert not tracker.observe("another-skill")
    assert not tracker.observe("third-skill")
    assert not tracker.early_exit_hit
    assert tracker.turns_taken == 3

    normalized = tracker.apply_to_outcome(
        SelectionOutcome(
            invoked_skills=("cloud-sql", "another-skill", "third-skill"),
            turns_taken=3,
        ),
    )
    assert normalized.invoked_skills == ("cloud-sql", "another-skill")
    assert normalized.turns_taken == 2
    assert normalized.early_exit is False


def test_trajectory_tracker_ignores_none_or_empty() -> None:
    """Verify TrajectoryTracker safely ignores None or empty string invocations."""
    from reach.runtime import TrajectoryTracker

    tracker = TrajectoryTracker(target_skill="cloud-sql", max_turns=3, early_exit=True)
    assert not tracker.observe(None)
    assert not tracker.observe("")
    assert tracker.turns_taken == 1
    assert tracker.invoked_skills == []


@pytest.mark.parametrize(
    ("invoked_skills", "summary_turns", "expected_turns"),
    [
        ((), 3, 3),  # Abstention: 0 skills, 3 turns taken
        (("skill-a", "skill-b"), 1, 1),  # Parallel tool calls: 2 skills, 1 turn
        (("skill-a",), 4, 4),  # Delayed execution: 1 skill, 4 turns taken
    ],
)
def test_to_outcome_preserves_telemetry_independent_of_invoked_skills(
    invoked_skills: tuple[str, ...],
    summary_turns: int,
    expected_turns: int,
) -> None:
    """Verify to_outcome does not guess or overwrite turns_taken with len(invoked_skills)."""
    from reach.runtime import SessionSummary

    summary = SessionSummary(
        invoked_skills=invoked_skills,
        turns_taken=summary_turns,
        early_exit=bool(invoked_skills),
    )
    outcome = summary.to_outcome(observed_catalog=("skill-a", "skill-b"))
    assert outcome.invoked_skills == invoked_skills
    assert outcome.turns_taken == expected_turns


def test_to_outcome_honors_explicit_turns_taken_override() -> None:
    """Verify to_outcome accepts explicit turns_taken parameter override."""
    from reach.runtime import SessionSummary

    summary = SessionSummary(
        invoked_skills=("skill-a",),
        turns_taken=1,
    )
    outcome = summary.to_outcome(turns_taken=5)
    assert outcome.turns_taken == 5


def test_selection_outcome_suppresses_cancellation_error_on_early_exit() -> None:
    """Verify SelectionOutcome automatically clears process cancellation errors on early exit."""
    from reach.runtime import SelectionOutcome

    # When early_exit=True, synthetic cancellation/timeout errors must be cleared
    outcome = SelectionOutcome(
        early_exit=True,
        invoked_skills=("skill-a",),
        error="timeout waiting for response",
    )
    assert outcome.error is None

    outcome2 = SelectionOutcome(
        early_exit=True,
        invoked_skills=("skill-a",),
        error="subprocess failed",
    )
    assert outcome2.error is None


def test_selection_outcome_preserves_security_tool_and_residency_leaks() -> None:
    """Verify SelectionOutcome strictly retains tool and residency leak errors on early exit."""
    from reach.runtime import SelectionOutcome

    # Genuine security leaks must never be masked
    outcome_tool = SelectionOutcome(
        early_exit=True,
        invoked_skills=("skill-a",),
        error="tool leak: Bash, write_file",
    )
    assert outcome_tool.error == "tool leak: Bash, write_file"

    outcome_res = SelectionOutcome(
        early_exit=True,
        invoked_skills=("skill-a",),
        error="residency leak: rogue-skill",
    )
    assert outcome_res.error == "residency leak: rogue-skill"


def test_session_summary_syncs_tool_calls_to_observed_tools() -> None:
    """Verify SessionSummary synthesizes observed_tools from tool_calls if not provided."""
    from reach.runtime import SessionSummary, ToolCallInfo

    summary = SessionSummary(
        tool_calls=(
            ToolCallInfo(name="view_file", parameters={"AbsolutePath": "/path/to/skill"}),
            ToolCallInfo(name="list_dir", parameters={}),
        ),
    )
    assert summary.observed_tools == ("list_dir", "view_file")

    # If observed_tools provided directly (e.g. Claude Code), it is preserved
    summary_direct = SessionSummary(observed_tools=("View", "Skill"))
    assert summary_direct.observed_tools == ("View", "Skill")


def test_session_summary_saw_result_property() -> None:
    """Verify saw_result returns True if status is present or early_exit is True."""
    from reach.runtime import SessionStatus, SessionSummary

    assert SessionSummary(status=SessionStatus.SUCCESS).saw_result
    assert SessionSummary(status="SUCCESS").saw_result
    assert SessionSummary(status="success").saw_result
    assert SessionSummary(early_exit=True).saw_result
    assert not SessionSummary(status=None, early_exit=False).saw_result


def test_tool_call_info_path_property_and_coercion() -> None:
    """Verify ToolCallInfo provides path property and parameter coercion."""
    from reach.runtime import ToolCallInfo

    call1 = ToolCallInfo(name="view_file", path="/work/skill/SKILL.md")
    assert call1.path == "/work/skill/SKILL.md"
    assert call1.target_path == "/work/skill/SKILL.md"
    assert call1.parameters == {"path": "/work/skill/SKILL.md"}

    call2 = ToolCallInfo(name="view_file", parameters={"AbsolutePath": "/work/abs.md"})
    assert call2.path == "/work/abs.md"
    assert call2.target_path == "/work/abs.md"


def test_agent_runtime_effective_effort_and_resident_paths(tmp_path: Path) -> None:
    """Verify AgentRuntime effective_effort and resident_skill_paths properties."""
    from reach.runtime.fake import FakeOptions, FakeRuntime

    # Default model profile effort resolution
    runtime = FakeRuntime(options=FakeOptions(model="gemini-3.8-flash"))
    assert runtime.effective_effort == "low"

    # Explicit effort override on options takes precedence
    runtime_override = FakeRuntime(options=FakeOptions(model="gemini-3.8-flash", effort="high"))
    assert runtime_override.effective_effort == "high"

    # resident_skill_paths resolves resident skills under skills_dir
    runtime._resident = ("skill-a", "skill-b")
    paths = runtime.resident_skill_paths(tmp_path)
    expected = {
        (tmp_path / ".agents/skills/skill-a").resolve(),
        (tmp_path / ".agents/skills/skill-b").resolve(),
    }
    assert paths == expected


def test_session_status_normalization_and_idiosyncrasy_handling() -> None:
    """Verify SessionStatus coerces known variants and preserves vendor states gracefully."""
    from reach.runtime import SessionStatus, SessionSummary

    # Canonical enum members
    assert SessionSummary(status=SessionStatus.SUCCESS).status == SessionStatus.SUCCESS
    assert SessionSummary(status=SessionStatus.ERROR).status == SessionStatus.ERROR
    assert SessionSummary(status=SessionStatus.TIMEOUT).status == SessionStatus.TIMEOUT
    assert SessionSummary(status=SessionStatus.CANCELLED).status == SessionStatus.CANCELLED

    # Case-insensitive string normalization to enum
    assert SessionSummary(status="SUCCESS").status == SessionStatus.SUCCESS
    assert SessionSummary(status="success").status == SessionStatus.SUCCESS
    assert SessionSummary(status="error").status == SessionStatus.ERROR
    assert SessionSummary(status="timeout").status == SessionStatus.TIMEOUT
    assert SessionSummary(status="cancelled").status == SessionStatus.CANCELLED

    # Unrecognized / vendor-specific idiosyncratic status preserved without validation crash
    vendor_summary = SessionSummary(status="rate_limited_tier_2")
    assert vendor_summary.status == "rate_limited_tier_2"
    assert vendor_summary.saw_result


# --- Antigravity domain schema and tool contract -----------------------------


def test_antigravity_runtime_selection_schema_and_json_schema() -> None:
    """Verify AntigravityRuntime generates both Pydantic model and derived JSON schema."""
    import json

    from reach.runtime import AntigravityRuntime

    schema_cls = AntigravityRuntime.selection_schema(["skill-1", "skill-2"])
    assert issubclass(schema_cls, SkillSelectionBase)
    valid_instance = schema_cls(selected_skill="skill-1", reasoning="matched intent")
    assert valid_instance.selected_skill == "skill-1"
    assert valid_instance.reasoning == "matched intent"

    json_schema_str = AntigravityRuntime.selection_json_schema(["skill-1", "skill-2"])
    schema_dict = json.loads(json_schema_str)
    assert schema_dict["type"] == "object"
    assert "selected_skill" in schema_dict["properties"]
    assert "reasoning" in schema_dict["properties"]


def test_antigravity_runtime_selection_schema_empty_catalog() -> None:
    """Verify AntigravityRuntime handles empty resident catalog gracefully."""
    from reach.runtime import AntigravityRuntime

    schema_cls = AntigravityRuntime.selection_schema([])
    instance = schema_cls(selected_skill="arbitrary", reasoning="fallback")
    assert instance.selected_skill == "arbitrary"


def test_antigravity_runtime_selection_tools(tmp_path: Path) -> None:
    """Verify AntigravityRuntime defines canonical selection tools set."""
    from reach.runtime import AntigravityRuntime

    expected = frozenset({"view_file", "list_dir", "grep_search", "find_by_name"})
    assert expected == AntigravityRuntime.ANTIGRAVITY_SELECTION_TOOLS
    runtime = _build_agent("antigravity-cli", tmp_path)
    assert isinstance(runtime, AntigravityRuntime)
    assert runtime.selection_tools == expected


# --- CLI agent runtime execution template ------------------------------------


class DummyCliRuntime(CliAgentRuntime[CliOptions]):
    """Concrete dummy CLI runtime for testing template method behavior."""

    name = "dummy-cli"
    options: CliOptions

    @override
    def build_command(self, query_text: str) -> list[str]:
        """Assemble probe command."""
        return ["dummy", "-q", query_text]

    @override
    def parse_stream(
        self,
        lines: Iterable[str],
        resident: Sequence[str] = (),
        early_exit: bool = False,
    ) -> SessionSummary:
        """Parse stream lines into SessionSummary."""
        del resident
        invoked = [line.split(":", 1)[1].strip() for line in lines if line.startswith("SKILL:")]
        return SessionSummary(
            invoked_skills=tuple(invoked),
            early_exit=early_exit,
            status="SUCCESS" if invoked else None,
        )

    @override
    def extract_skill_from_line(self, line: str) -> str | None:
        """Extract skill from single line."""
        if line.startswith("SKILL:"):
            return line.split(":", 1)[1].strip()
        return None


def test_cli_template_method_clean_execution(
    mock_subprocess: Callable[..., Any],
    tmp_path: Path,
) -> None:
    """Verify CliAgentRuntime template method runs subprocess and returns outcome."""
    mock_subprocess(stdout="SKILL:target-skill\n")
    rt = DummyCliRuntime()
    rt._resident = ("target-skill", "other-skill")

    outcome = rt.select("test query", tmp_path, target_skill="target-skill")
    assert outcome.error is None
    assert outcome.invoked_skill == "target-skill"
    assert outcome.early_exit is True
    assert outcome.turns_taken == 1
    assert outcome.observed_catalog == ("target-skill", "other-skill")


def test_cli_agent_runtime_validate_outcome_status_check(tmp_path: Path) -> None:
    """Verify CliAgentRuntime base validate_outcome detects non-success status."""
    from reach.runtime import SessionStatus, SessionSummary

    rt = DummyCliRuntime()
    clean_summary = SessionSummary(status=SessionStatus.SUCCESS)
    assert rt.validate_outcome(clean_summary, tmp_path) is None

    error_summary = SessionSummary(status=SessionStatus.ERROR, error="bad state")
    assert rt.validate_outcome(error_summary, tmp_path) == "bad state"

    timeout_summary = SessionSummary(status=SessionStatus.TIMEOUT)
    assert rt.validate_outcome(timeout_summary, tmp_path) == "runtime error: TIMEOUT"


def test_cli_select_handles_spawn_failure_without_raising(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify CliAgentRuntime handles subprocess spawn failures without raising."""

    def _mock_run(*args: Any, **kwargs: Any) -> tuple[None, str]:
        return None, "executable not found: dummy"

    monkeypatch.setattr("reach.runtime.run_subprocess_probe", _mock_run)
    rt = DummyCliRuntime()

    outcome = rt.select("query", tmp_path)
    assert outcome.error == "executable not found: dummy"
    assert outcome.invoked_skill is None


def test_cli_select_handles_missing_result_event(
    mock_subprocess: Callable[..., Any],
    tmp_path: Path,
) -> None:
    """Verify CliAgentRuntime flags missing result event when process produces no status."""
    mock_subprocess(stdout="plain log without skill", returncode=1, stderr="crashed")
    rt = DummyCliRuntime()

    outcome = rt.select("query", tmp_path)
    assert outcome.error is not None
    assert "no result event" in outcome.error


def test_cli_generator_complete_success(
    mock_subprocess: Callable[..., Any],
) -> None:
    """Verify CLI generator complete executes subprocess and returns text."""
    mock_subprocess(stdout="generated answer\n")
    gen = build_text_generator(agent="pi")
    assert gen.complete("test prompt") == "generated answer"
    assert gen.completions == 1


def test_cli_generator_complete_failure(
    mock_subprocess: Callable[..., Any],
) -> None:
    """Verify CLI generator complete raises on subprocess failure."""
    mock_subprocess(returncode=1, stderr="fatal error")
    gen = build_text_generator(agent="pi")
    with pytest.raises(RuntimeError, match=r"generation failed: fatal error"):
        gen.complete("test prompt")


@pytest.mark.parametrize("agent", cli_agents())
def test_all_cli_generators_build_completion_command(agent: str, tmp_path: Path) -> None:
    """Verify all CLI text generators construct valid completion commands starting with binary."""
    gen = build_text_generator(agent=agent)
    cmd_fn = getattr(gen, "build_completion_command", None)
    assert callable(cmd_fn)
    cmd = cmd_fn("hello world")
    assert isinstance(cmd, list)
    assert len(cmd) > 0
    runtime = _build_agent(agent, tmp_path)
    assert isinstance(runtime, CliAgentRuntime)
    assert cmd[0] == runtime.options.executable


# --- Cross-client runtime conformance ----------------------------------------


@pytest.mark.parametrize("agent", known_agents())
def test_all_agents_cli_hierarchy_consistency(agent: str, tmp_path: Path) -> None:
    """Verify runtime.is_cli matches CliAgentRuntime and CliOptions hierarchy across agents."""
    runtime = _build_agent(agent, tmp_path)
    opt = options_model(agent)
    is_cli_runtime = isinstance(runtime, CliAgentRuntime)
    assert runtime.is_cli is is_cli_runtime
    assert is_cli_runtime == (opt is not None and issubclass(opt, CliOptions))


@pytest.mark.parametrize("agent", cli_agents())
def test_all_cli_agents_implement_build_command_contract(agent: str, tmp_path: Path) -> None:
    """Verify all CLI drivers implement the build_command contract."""
    runtime = _build_agent(agent, tmp_path)
    assert isinstance(runtime, CliAgentRuntime)
    cmd = runtime.build_command("test query")
    assert isinstance(cmd, list)
    assert len(cmd) > 0
    assert isinstance(cmd[0], str)


@pytest.mark.parametrize("agent", known_agents())
def test_all_agents_parse_stream_empty_input(agent: str, tmp_path: Path) -> None:
    """Verify all drivers return a valid SessionSummary without skills for empty stream."""
    runtime = _build_agent(agent, tmp_path)
    summary = runtime.parse_stream([])
    assert isinstance(summary, SessionSummary)
    assert summary.invoked_skill is None
    assert summary.invoked_skills == ()
    assert not summary.saw_result


def test_fake_runtime_parse_stream_uses_configured_default() -> None:
    """Verify FakeRuntime extracts default response during stream parsing."""
    runtime = FakeRuntime(default="alpha")
    summary = runtime.parse_stream(["some log line"])
    assert summary.invoked_skill == "alpha"
    assert summary.invoked_skills == ("alpha",)
    assert summary.saw_result


@pytest.mark.parametrize("agent", cli_agents())
@pytest.mark.parametrize(
    "raw_line",
    ["", "   ", "not json", "[1, 2, 3]", '"just-a-string"', "12345", "true", "null", "{}"],
    ids=[
        "empty",
        "whitespace",
        "garbage",
        "json-array",
        "json-string",
        "json-int",
        "json-bool",
        "json-null",
        "empty-dict",
    ],
)
def test_all_cli_agents_extract_skills_from_line_gracefully_handles_malformed_inputs(
    agent: str,
    raw_line: str,
    tmp_path: Path,
) -> None:
    """Verify extract_skills_from_line safely returns empty sequence for malformed inputs."""
    runtime = _build_agent(agent, tmp_path)
    assert isinstance(runtime, CliAgentRuntime)
    assert runtime.extract_skills_from_line(raw_line) == ()


@pytest.mark.parametrize("agent", list(known_agents()))
def test_all_agents_parse_stream_malformed_input(agent: str, tmp_path: Path) -> None:
    """Verify all drivers return a valid SessionSummary without raising on malformed stream."""
    runtime = _build_agent(agent, tmp_path)

    summary = runtime.parse_stream(["not json", "[1, 2, 3]", "12345", "true", "null", "{}"])
    assert isinstance(summary, SessionSummary)


def test_sync_google_and_gemini_keys_bidirectional() -> None:
    """Verify sync_google_and_gemini_keys synchronizes keys in both directions."""
    # GEMINI -> GOOGLE
    env1 = {"GEMINI_API_KEY": "secret-1"}
    sync_google_and_gemini_keys(env1)
    assert env1 == {"GEMINI_API_KEY": "secret-1", "GOOGLE_API_KEY": "secret-1"}

    # GOOGLE -> GEMINI
    env2 = {"GOOGLE_API_KEY": "secret-2"}
    sync_google_and_gemini_keys(env2)
    assert env2 == {"GEMINI_API_KEY": "secret-2", "GOOGLE_API_KEY": "secret-2"}

    # Both present -> preserve existing
    env3 = {"GEMINI_API_KEY": "gemini-orig", "GOOGLE_API_KEY": "google-orig"}
    sync_google_and_gemini_keys(env3)
    assert env3 == {"GEMINI_API_KEY": "gemini-orig", "GOOGLE_API_KEY": "google-orig"}

    # Neither present -> no changes
    env4 = {"OTHER_KEY": "other"}
    sync_google_and_gemini_keys(env4)
    assert env4 == {"OTHER_KEY": "other"}


@pytest.mark.parametrize(
    ("provider", "expected_vars"),
    [
        ("anthropic", ("ANTHROPIC_API_KEY",)),
        ("openai", ("OPENAI_API_KEY",)),
        ("google", ("GEMINI_API_KEY", "GOOGLE_API_KEY")),
        ("gemini", ("GEMINI_API_KEY", "GOOGLE_API_KEY")),
    ],
)
def test_apply_provider_api_key_known_providers(
    provider: str,
    expected_vars: tuple[str, ...],
) -> None:
    """Verify apply_provider_api_key maps keys for all known providers."""
    env: dict[str, str] = {}
    apply_provider_api_key(env, provider=provider, api_key="test-key")
    for var in expected_vars:
        assert env[var] == "test-key"


def test_apply_provider_api_key_fallback_and_empty() -> None:
    """Verify provider mapping handles missing key, default provider, and unknown fallbacks."""
    # Empty / None key does not modify env
    env1 = {"EXISTING": "val"}
    apply_provider_api_key(env1, provider="google", api_key=None)
    assert env1 == {"EXISTING": "val"}
    apply_provider_api_key(env1, provider="google", api_key="")
    assert env1 == {"EXISTING": "val"}

    # None provider uses default_provider
    env2: dict[str, str] = {}
    apply_provider_api_key(env2, provider=None, api_key="k1", default_provider="openai")
    assert env2 == {"OPENAI_API_KEY": "k1"}

    env3: dict[str, str] = {}
    apply_provider_api_key(env3, provider=None, api_key="k2", default_provider="google")
    assert env3 == {"GEMINI_API_KEY": "k2", "GOOGLE_API_KEY": "k2"}

    # Explicit unknown provider raises ValueError to avoid secret leakage
    env4: dict[str, str] = {}
    with pytest.raises(ValueError, match="Unrecognized provider 'unknown-custom'"):
        apply_provider_api_key(
            env4,
            provider="unknown-custom",
            api_key="k3",
            default_provider="openai",
        )


def test_probe_slot_dir_includes_pid_and_thread(tmp_path: Path) -> None:
    """Verify probe_slot_dir isolates directories by process ID and thread ID."""
    import os

    slot = probe_slot_dir(tmp_path, prefix="test_slot")
    expected_name = f"test_slot_{os.getpid()}_{probe_slot_id()}"
    assert slot.name == expected_name
    assert slot.parent == tmp_path


def test_cli_agents_discovery() -> None:
    """Verify cli_agents returns registered CLI driver names and filters non-CLI runtimes."""
    cli = set(cli_agents())
    assert cli <= set(known_agents())
    for agent in cli:
        opt = options_model(agent)
        assert opt is not None
        assert issubclass(opt, CliOptions)
    for agent in set(known_agents()) - cli:
        opt = options_model(agent)
        assert opt is None or not issubclass(opt, CliOptions)


def test_model_validators_preserve_input_dict_immutability() -> None:
    """Verify before-validators in runtime models do not mutate caller dictionaries."""
    tool_dict = {"name": "read", "path": "/path/to/file"}
    coerced = ToolCallInfo.model_validate(tool_dict)
    assert coerced.parameters["path"] == "/path/to/file"
    assert "path" in tool_dict
    assert "parameters" not in tool_dict

    summary_dict = {"tool_calls": [{"name": "read"}]}
    summary = SessionSummary.model_validate(summary_dict)
    assert summary.observed_tools == ("read",)
    assert "observed_tools" not in summary_dict

    outcome_dict = {"early_exit": True, "error": "process killed"}
    outcome = SelectionOutcome.model_validate(outcome_dict)
    assert outcome.error is None
    assert outcome_dict["error"] == "process killed"


def test_ensure_private_directory_creates_and_secures(tmp_path: Path) -> None:
    """Verify ensure_private_directory creates directory with 0o700 permissions on POSIX."""
    from reach.runtime._fs import ensure_private_directory

    target = tmp_path / "deep" / "nested" / "private_dir"
    res = ensure_private_directory(target)
    assert res == target.resolve()
    assert res.is_dir()

    if os.name == "posix":
        mode = res.stat().st_mode & 0o777
        assert mode == 0o700


def test_ensure_private_directory_tightens_existing_permissions(tmp_path: Path) -> None:
    """Verify ensure_private_directory tightens pre-existing loose permissions to 0o700."""
    from reach.runtime._fs import ensure_private_directory

    target = tmp_path / "preexisting_dir"
    target.mkdir(mode=0o777)
    if os.name == "posix":
        target.chmod(0o755)

    res = ensure_private_directory(target)
    assert res.is_dir()

    if os.name == "posix":
        mode = res.stat().st_mode & 0o777
        assert mode == 0o700


def test_ensure_private_directory_accepts_str(tmp_path: Path) -> None:
    """Verify ensure_private_directory works seamlessly when passed a string path."""
    from reach.runtime._fs import ensure_private_directory

    target_str = str(tmp_path / "str_dir")
    res = ensure_private_directory(target_str)
    assert res.is_dir()
    assert res == Path(target_str).resolve()


def test_ensure_private_directory_tolerates_chmod_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify ensure_private_directory tolerates OSError during chmod without raising."""
    from reach.runtime._fs import ensure_private_directory

    target = tmp_path / "chmod_fail_dir"

    def _failing_chmod(path: Any, mode: int, **kwargs: Any) -> None:
        msg = "Operation not permitted"
        raise OSError(msg)

    monkeypatch.setattr(os, "chmod", _failing_chmod)
    res = ensure_private_directory(target)
    assert res.is_dir()


def test_runtimes_create_private_isolation_directories(tmp_path: Path) -> None:
    """Verify runtimes establish private 0o700 isolation directories on POSIX."""
    from reach.runtime.claude_code import ClaudeCodeRuntime
    from reach.runtime.goose import GooseRuntime
    from reach.runtime.pi import PiRuntime

    workdir = tmp_path / "workspace"
    workdir.mkdir()

    claude = ClaudeCodeRuntime()
    claude_env = claude.build_env(workdir)
    claude_cfg = Path(claude_env["CLAUDE_CONFIG_DIR"])
    assert claude_cfg.is_dir()

    goose = GooseRuntime()
    goose_env = goose.build_env(workdir)
    goose_home = Path(goose_env["HOME"])
    assert goose_home.is_dir()

    pi = PiRuntime()
    pi_env = pi.build_env(workdir)
    pi_agent = Path(pi_env["PI_CODING_AGENT_DIR"])
    assert pi_agent.is_dir()

    if os.name == "posix":
        assert (claude_cfg.stat().st_mode & 0o777) == 0o700
        assert (goose_home.stat().st_mode & 0o777) == 0o700
        assert (pi_agent.stat().st_mode & 0o777) == 0o700


def test_registry_cache_manager_creates_private_directories(tmp_path: Path) -> None:
    """Verify RegistryCacheManager creates private directories when writing manifest or skill."""
    from datetime import UTC, datetime

    from reach.registry import RegistryCacheManager, RegistryManifest, RegistrySkillData

    cache_dir = tmp_path / "cache"
    mgr = RegistryCacheManager(cache_root=cache_dir)
    manifest = RegistryManifest(
        project="my-proj",
        location="us-central1",
        fetched_at=datetime.now(UTC),
        skills=(),
    )
    mgr.save_manifest(manifest)

    manifest_file = mgr.manifest_path("my-proj", "us-central1")
    assert manifest_file.is_file()
    if os.name == "posix":
        assert (manifest_file.parent.stat().st_mode & 0o777) == 0o700

    skill_data = RegistrySkillData(name="projects/my-proj/locations/us-central1/skills/demo")
    skill_dir = mgr.hydrate_skill_file("my-proj", "us-central1", skill_data)
    assert skill_dir.is_dir()
    if os.name == "posix":
        assert (skill_dir.stat().st_mode & 0o777) == 0o700


def test_validate_isolated_directory(tmp_path: Path) -> None:
    """Verify validate_isolated_directory rejects active home and root."""
    from reach.runtime._fs import validate_isolated_directory

    assert validate_isolated_directory(None) is None
    custom = tmp_path / "reach_isolated"
    assert validate_isolated_directory(custom) == custom

    with pytest.raises(ValueError, match="must not be the user's active home or root"):
        validate_isolated_directory(Path.home(), "home_dir")

    with pytest.raises(ValueError, match="must not be the user's active home or root"):
        validate_isolated_directory(Path("/"), "home_dir")


@pytest.mark.parametrize(
    "agent",
    [a for a in known_agents() if getattr(options_model(a), "isolation_dir_field", None)],
)
def test_agent_options_reject_home_and_root(agent: str, tmp_path: Path) -> None:
    """Verify all agent options models reject active home and root paths."""
    opt_cls = options_model(agent)
    assert opt_cls is not None
    assert issubclass(opt_cls, AgentOptions)
    field_name = opt_cls.isolation_dir_field
    assert field_name is not None

    with pytest.raises(PydanticValidationError, match="must not be the user's active home or root"):
        opt_cls.model_validate({field_name: str(Path.home())})

    with pytest.raises(PydanticValidationError, match="must not be the user's active home or root"):
        opt_cls.model_validate({field_name: "/"})

    valid_path = tmp_path / "isolated_test"
    valid = opt_cls.model_validate({field_name: str(valid_path)})
    assert getattr(valid, field_name) == valid_path


@pytest.mark.parametrize("agent", known_agents())
def test_runtime_post_probe_never_cleans_outside_workdir(agent: str, tmp_path: Path) -> None:
    """Verify post_probe on all runtimes safely ignores external directories without raising."""
    workdir = tmp_path / "workspace"
    workdir.mkdir()
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    canary = outside_dir / "canary.txt"
    canary.write_text("safe")

    opts: dict[str, Any] = dict(MINIMAL_OPTIONS.get(agent, {}))
    opts["auto_clean"] = True
    field = getattr(options_model(agent), "isolation_dir_field", None)
    if field:
        opts[field] = outside_dir

    runtime = build_runtime(RuntimeSettings(agent=agent, options=opts))
    runtime.post_probe(workdir)

    assert canary.exists()


@pytest.mark.parametrize(
    ("target_rel", "should_delete"),
    [
        (".isolated_dir", True),
        ("nested/sub_isolated", True),
    ],
)
def test_safe_cleanup_isolated_dir_removes_contained_dir(
    target_rel: str,
    should_delete: bool,
    tmp_path: Path,
) -> None:
    """Verify safe_cleanup_isolated_dir removes directories strictly inside workdir."""
    from reach.runtime._fs import safe_cleanup_isolated_dir

    workdir = tmp_path / "work"
    workdir.mkdir()
    target = workdir / target_rel
    target.mkdir(parents=True)
    (target / "dummy.txt").write_text("content")

    deleted = safe_cleanup_isolated_dir(workdir, target)
    assert deleted is should_delete
    assert not target.exists()


def test_safe_cleanup_isolated_dir_refuses_outside_or_root(tmp_path: Path) -> None:
    """Verify safe_cleanup_isolated_dir refuses outside dirs, None, or workdir itself."""
    from reach.runtime._fs import safe_cleanup_isolated_dir

    workdir = tmp_path / "work"
    workdir.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "file.txt").write_text("content")

    # Outside dir
    assert safe_cleanup_isolated_dir(workdir, outside) is False
    assert outside.exists()

    # Workdir itself
    assert safe_cleanup_isolated_dir(workdir, workdir) is False
    assert workdir.exists()

    # None
    assert safe_cleanup_isolated_dir(workdir, None) is False

    # Non-existent
    assert safe_cleanup_isolated_dir(workdir, workdir / "non_existent") is False


@pytest.mark.parametrize("agent", known_agents())
def test_runtime_isolation_dir_properties(agent: str, tmp_path: Path) -> None:
    """Verify isolation_dir_field and custom_isolation_dir properties reflect configuration."""
    opts: dict[str, Any] = dict(MINIMAL_OPTIONS.get(agent, {}))
    field = getattr(options_model(agent), "isolation_dir_field", None)
    custom_dir = tmp_path / f"custom_{agent}"
    if field:
        opts[field] = custom_dir

    runtime = build_runtime(RuntimeSettings(agent=agent, options=opts))
    if field:
        assert runtime.isolation_dir_field == field
        assert runtime.custom_isolation_dir == custom_dir
        assert runtime.options.custom_isolation_dir == custom_dir
    else:
        assert runtime.isolation_dir_field is None
        assert runtime.custom_isolation_dir is None


def test_antigravity_agents_discovery() -> None:
    """Verify antigravity_agents and runtime_class discover Antigravity runtimes."""
    agents = antigravity_agents()
    assert "antigravity-cli" in agents
    assert "antigravity-sdk" in agents
    for agent in agents:
        cls = runtime_class(agent)
        assert cls is not None
        assert issubclass(cls, AntigravityRuntime)


@pytest.mark.parametrize("agent", antigravity_agents())
def test_antigravity_runtime_effective_model_provider(
    agent: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify effective_model_provider auto-detects gemini across all Antigravity runtimes."""
    # 1. Auto-detect when GEMINI_API_KEY is present
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-123")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    rt = build_runtime(RuntimeSettings(agent=agent, options={"model": "gemini-3.7-flash"}))
    assert isinstance(rt, AntigravityRuntime)
    assert hasattr(rt, "effective_model_provider")
    assert rt.effective_model_provider == "gemini"

    # 2. Auto-detect when GOOGLE_API_KEY is present
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key-456")
    rt = build_runtime(RuntimeSettings(agent=agent, options={"model": "gemini-3.7-flash"}))
    assert isinstance(rt, AntigravityRuntime)
    assert rt.effective_model_provider == "gemini"

    # 3. Explicit provider overrides auto-detect when supported by options model
    opt_cls = options_model(agent)
    if opt_cls is not None and "model_provider" in opt_cls.model_fields:
        rt = build_runtime(
            RuntimeSettings(
                agent=agent,
                options={"model": "gemini-3.7-flash", "model_provider": "custom-prov"},
            ),
        )
        assert isinstance(rt, AntigravityRuntime)
        assert rt.effective_model_provider == "custom-prov"

    # 4. Non-gemini model does not auto-detect
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-123")
    rt = build_runtime(RuntimeSettings(agent=agent, options={"model": "claude-3-opus"}))
    assert isinstance(rt, AntigravityRuntime)
    assert rt.effective_model_provider is None

    # 5. No keys present results in None
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    rt = build_runtime(RuntimeSettings(agent=agent, options={"model": "gemini-3.7-flash"}))
    assert isinstance(rt, AntigravityRuntime)
    assert rt.effective_model_provider is None


@pytest.mark.parametrize("agent", antigravity_agents())
def test_antigravity_runtime_effective_api_key_resolution(
    agent: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify effective_api_key resolution hierarchy across Antigravity runtimes."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    # Fallback to GEMINI_API_KEY
    monkeypatch.setenv("GEMINI_API_KEY", "env-gemini-key")
    rt = build_runtime(RuntimeSettings(agent=agent, options={"model": "gemini-3.7-flash"}))
    assert isinstance(rt, AntigravityRuntime)
    assert rt.effective_api_key == "env-gemini-key"

    # Fallback to GOOGLE_API_KEY if GEMINI_API_KEY unset
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "env-google-key")
    rt = build_runtime(RuntimeSettings(agent=agent, options={"model": "gemini-3.7-flash"}))
    assert isinstance(rt, AntigravityRuntime)
    assert rt.effective_api_key == "env-google-key"

    # Options api_key takes precedence
    rt = build_runtime(
        RuntimeSettings(
            agent=agent,
            options={"model": "gemini-3.7-flash", "api_key": "explicit-key"},
        ),
    )
    assert isinstance(rt, AntigravityRuntime)
    assert rt.effective_api_key == "explicit-key"


def test_agent_options_accepts_and_validates_blocked_env_vars() -> None:
    """Verify AgentOptions accepts strongly-typed blocked_env_vars tuple."""
    opts = AgentOptions(blocked_env_vars=["CUSTOM_VAR", "ANOTHER_VAR"])
    assert opts.blocked_env_vars == ("CUSTOM_VAR", "ANOTHER_VAR")


def test_resolve_options_propagates_blocked_env_vars() -> None:
    """Verify resolve_options propagates blocked_env_vars into driver options."""
    settings = RuntimeSettings(
        agent="fake",
        blocked_env_vars=["CUSTOM_SECRET"],
        options={"model": "fake-model"},
    )
    resolved = resolve_options(settings)
    assert isinstance(resolved, AgentOptions)
    assert resolved.blocked_env_vars == ("CUSTOM_SECRET",)


def test_base_text_generator_build_env_sanitizes_via_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify BaseTextGenerator.build_env strips blocked env vars specified in options."""
    monkeypatch.setenv("SECRET_TOKEN", "sensitive")
    monkeypatch.setenv("PUBLIC_VAR", "harmless")

    opts = FakeOptions(model="test-model", blocked_env_vars=("SECRET_TOKEN",))
    gen = FakeGenerator(model="test-model", options=opts)
    env = gen.build_env()
    assert "SECRET_TOKEN" not in env
    assert env.get("PUBLIC_VAR") == "harmless"


def test_agent_runtime_build_env_prioritizes_options_over_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify AgentRuntime.build_env prioritizes options over settings for blocked vars."""
    monkeypatch.setenv("VAR_A", "a")
    monkeypatch.setenv("VAR_B", "b")

    settings = RuntimeSettings(agent="fake", blocked_env_vars=["VAR_A"])
    opts = FakeOptions(model="test-model", blocked_env_vars=["VAR_B"])
    rt = FakeRuntime(settings=settings, options=opts)
    env = rt.build_env()
    assert "VAR_B" not in env
    assert env.get("VAR_A") == "a"


def test_fake_runtime_materialize_delegates_to_super_install_and_skill_roots(
    skill_repo: Path,
    tmp_path: Path,
) -> None:
    """Verify FakeRuntime with materialize=True delegates install and skill_roots to base."""
    skills = load_skills(skill_repo)
    catalog = Catalog(
        id="test-subset",
        mode=CatalogMode.ALL,
        skills=("gcs-lifecycle-rules", "gcs-retention-policy"),
    )
    runtime = FakeRuntime(materialize=True)
    target = runtime.install(catalog, skills, tmp_path / "work")

    # Verify installation materialized on disk via super().install()
    installed_dir = runtime.skills_dir(target)
    assert installed_dir.is_dir()
    for name in catalog.skills:
        assert (installed_dir / name).exists()

    # Verify skill_roots returns materialized root via super().skill_roots()
    roots = runtime.skill_roots(target)
    assert len(roots) == 1
    assert roots[0].path == installed_dir
    assert roots[0].scope == "project"


def test_runtime_and_generator_model_setter_synchronizes_options() -> None:
    """Verify AgentRuntime and BaseTextGenerator model setters update options.model."""
    runtime = FakeRuntime(model="initial-model")
    assert runtime.model == "initial-model"
    assert runtime.options.model == "initial-model"
    runtime.model = "updated-model"
    assert runtime.model == "updated-model"
    assert runtime.options.model == "updated-model"

    gen = FakeGenerator(model="initial-model")
    assert gen.model == "initial-model"
    assert gen.options.model == "initial-model"
    gen.model = "updated-model"
    assert gen.model == "updated-model"
    assert gen.options.model == "updated-model"


def test_fake_runtime_multi_turn_sequence_trajectory_tracker(tmp_path: Path) -> None:
    """Verify FakeRuntime select uses TrajectoryTracker for multi-turn sequences."""
    # Case 1: Early exit triggered by reaching target_skill
    runtime_hit = FakeRuntime(
        {"q": ("s1", "target", "s3")},
        options=FakeOptions(max_turns=3, early_exit=True),
    )
    outcome_hit = runtime_hit.select("q", tmp_path, target_skill="target")
    assert outcome_hit.invoked_skills == ("s1", "target")
    assert outcome_hit.early_exit is True
    assert outcome_hit.turns_taken == 2

    # Case 2: Max turns boundary reached before target_skill
    runtime_max = FakeRuntime(
        {"q": ("s1", "s2", "s3", "target")},
        options=FakeOptions(max_turns=2, early_exit=True),
    )
    outcome_max = runtime_max.select("q", tmp_path, target_skill="target")
    assert outcome_max.invoked_skills == ("s1", "s2")
    assert outcome_max.early_exit is True
    assert outcome_max.turns_taken == 2
