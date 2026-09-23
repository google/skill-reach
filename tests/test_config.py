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

"""Verify that a run configuration validates, loads, overrides, and fingerprints."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from inspect import signature
from pathlib import Path

import pytest
from pydantic import ValidationError

from reach.cli import StudyFlags, build_config
from reach.config import (
    DEFAULT_ATTEMPTS,
    QuerySettings,
    RunConfig,
    RuntimeSettings,
    StudySettings,
    expand_path,
)
from reach.models import CatalogMode
from reach.run import ProbeHarness, load_corpus, plan_probes
from reach.runtime import FAKE_AGENT, known_agents

_EXECUTION_AGENTS = tuple(a for a in known_agents() if a not in ("keyword", FAKE_AGENT))


@pytest.fixture
def minimal(tmp_path: Path) -> RunConfig:
    """Provide a configuration with only the study paths supplied."""
    return RunConfig(
        study=StudySettings(
            skills=tmp_path / "corpus",
            queries=tmp_path / "q.json",
            workdir=tmp_path / "work",
        ),
    )


def write_toml(tmp_path: Path, body: str) -> Path:
    """Write a TOML run configuration to disk and return its path."""
    path = tmp_path / "reach.toml"
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return path


def test_neighborhood_is_the_default_scoping(minimal: RunConfig) -> None:
    """Verify NEIGHBORHOOD mode is default catalog scoping with 20 items and 10 rivals."""
    assert minimal.catalog.mode is CatalogMode.NEIGHBORHOOD
    assert (minimal.catalog.size, minimal.catalog.rivals) == (20, 10)


def test_default_attempts_across_the_whole_tool(minimal: RunConfig) -> None:
    """Verify DEFAULT_ATTEMPTS constant is 5 and matches default parameter across runner modules."""
    assert minimal.plan.attempts == DEFAULT_ATTEMPTS == 5
    assert signature(plan_probes).parameters["attempts"].default == DEFAULT_ATTEMPTS
    assert signature(ProbeHarness.run_probes).parameters["attempts"].default == DEFAULT_ATTEMPTS


def test_isolation_is_configured_not_assumed(minimal: RunConfig) -> None:
    """Verify runtime isolation options restrict turns to default 3 and deny Bash execution."""
    claude_cfg = minimal.with_overrides(runtime={"agent": "claude-code"})
    options = claude_cfg.runtime.resolved_options()
    assert options["max_turns"] == 3
    assert "Bash" in options["denied_tools"]
    assert "Skill" not in options["denied_tools"]


def test_any_module_may_be_imported_first() -> None:
    """Verify modules can be imported independently without circular errors."""
    modules = [
        "reach.config",
        "reach.runtime",
        "reach.runtime.profiles",
        "reach.runtime.claude_code",
        "reach.runtime.claude_listing",
        "reach.cli",
        "reach.artifact",
        "reach.views",
        "reach.views.base",
        "reach.views.diff",
        "reach.views.lint",
        "reach.views.optimize",
        "reach.views.overlap",
        "reach.views.quality_gate",
        "reach.views.scorecard",
    ]
    script = "; ".join(f"import {mod}" for mod in modules)
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, f"Module imports failed:\n{completed.stderr}"


def test_shared_config_names_no_runtime_specific_setting() -> None:
    """Verify RuntimeSettings schema defines only common fields."""
    shared = set(RuntimeSettings.model_fields)
    assert shared == {
        "agent",
        "allowed_tools",
        "blocked_env_vars",
        "early_exit",
        "max_turns",
        "options",
        "timeout_s",
    }


def test_runtime_settings_parses_blocked_env_vars_from_toml(tmp_path: Path) -> None:
    """Verify blocked_env_vars parses from [runtime] TOML table."""
    path = write_toml(
        tmp_path,
        """
        [study]
        skills = "corpus"
        queries = "q.json"
        workdir = "work"

        [runtime]
        agent = "antigravity-cli"
        blocked_env_vars = ["AWS_SECRET_ACCESS_KEY", "CUSTOM_SECRET"]
        """,
    )
    config = RunConfig.from_toml(path)
    assert config.runtime.blocked_env_vars == ("AWS_SECRET_ACCESS_KEY", "CUSTOM_SECRET")


def test_an_option_the_agent_does_not_know_is_rejected_at_load() -> None:
    """Verify validation error is raised for unknown agent options."""
    with pytest.raises(ValidationError, match="setting_sauces"):
        RuntimeSettings(agent="claude-code", options={"setting_sauces": "project"})


def test_options_are_refused_by_an_agent_that_takes_none() -> None:
    """Verify validation error is raised when options are passed to agents without options."""
    with pytest.raises(ValidationError, match="setting_sources"):
        RuntimeSettings(agent="keyword", options={"setting_sources": "project"})


def test_a_misspelled_agent_is_named_as_the_mistake() -> None:
    """Verify validation error explicitly cites unknown agent name."""
    with pytest.raises(ValidationError, match="unknown runtime agent 'nonexistent-agent'"):
        RuntimeSettings(agent="nonexistent-agent", options={"setting_sources": "project"})


def test_options_load_from_their_own_toml_table(tmp_path: Path) -> None:
    """Verify agent options parse properly from nested [runtime.options] TOML table."""
    path = write_toml(
        tmp_path,
        """
        [study]
        skills = "corpus"
        queries = "q.json"
        workdir = "work"

        [runtime]
        agent = "claude-code"

        [runtime.options]
        setting_sources = "user,project"
        disable_bundled_skills = false
        """,
    )
    resolved = RunConfig.from_toml(path).runtime.resolved_options()
    assert resolved["setting_sources"] == "user,project"
    assert resolved["disable_bundled_skills"] is False


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("catalog", "size", 1),
        ("catalog", "rivals", 0),
        ("plan", "attempts", 0),
        ("plan", "retries", -1),
        ("plan", "backoff_s", -1.0),
        ("runtime", "timeout_s", 0),
    ],
    ids=["size", "rivals", "attempts", "retries", "backoff", "timeout"],
)
def test_impossible_settings_are_rejected(minimal, section, field, value) -> None:
    """Verify schema boundary validation rejects invalid out-of-range parameters."""
    with pytest.raises(ValidationError):
        minimal.with_overrides(**{section: {field: value}})


def test_paths_expand(tmp_path: Path) -> None:
    """Verify tilde paths expand to absolute user home paths."""
    config = RunConfig(
        study=StudySettings.model_validate(
            {"skills": "~/corpus", "queries": "~/q.json", "workdir": "~/work"},
        ),
    )
    assert config.study.skills is not None
    assert config.study.skills.is_absolute()
    assert "~" not in str(config.study.skills)


def test_toml_resolves_relative_paths_against_the_file(tmp_path: Path) -> None:
    """Verify relative paths in TOML configuration resolve relative to TOML file location."""
    path = write_toml(
        tmp_path,
        """
        [study]
        skills = "corpus"
        queries = "queries/waf.json"
        workdir = "work"
        """,
    )
    config = RunConfig.from_toml(path)
    assert config.study.skills == tmp_path / "corpus"
    assert config.study.queries == tmp_path / "queries" / "waf.json"


def test_toml_absolute_paths_are_left_alone(tmp_path: Path) -> None:
    """Verify absolute paths in TOML configuration remain unmodified."""
    path = write_toml(
        tmp_path,
        f"""
        [study]
        skills = "/opt/corpus"
        queries = "{tmp_path}/q.json"
        workdir = "work"
        """,
    )
    assert RunConfig.from_toml(path).study.skills == Path("/opt/corpus")


def test_a_variable_names_a_corpus_a_study_cannot_commit(tmp_path: Path, monkeypatch) -> None:
    """Verify environment variable substitutions expand in TOML path fields."""
    monkeypatch.setenv("REACH_SKILL_ROOT", str(tmp_path / "gskills"))
    path = write_toml(
        tmp_path,
        """
        [study]
        skills = "${REACH_SKILL_ROOT}"
        queries = "queries/waf.json"
        workdir = "work"
        """,
    )
    assert RunConfig.from_toml(path).study.skills == tmp_path / "gskills"


def test_an_unset_variable_is_named_where_it_was_written(tmp_path: Path, monkeypatch) -> None:
    """Verify ValueError is raised when referenced environment variable is unset."""
    monkeypatch.delenv("REACH_SKILL_ROOT", raising=False)
    path = write_toml(
        tmp_path,
        """
        [study]
        skills = "${REACH_SKILL_ROOT}"
        queries = "queries/waf.json"
        workdir = "work"
        """,
    )
    with pytest.raises(ValueError, match="REACH_SKILL_ROOT"):
        RunConfig.from_toml(path)


def test_toml_carries_every_section(tmp_path: Path) -> None:
    """Verify RunConfig.from_toml correctly parses study, runtime, catalog, and plan sections."""
    path = write_toml(
        tmp_path,
        """
        [study]
        skills = "corpus"
        queries = "q.json"
        workdir = "work"

        [runtime]
        agent = "claude-code"

        [runtime.options]
        executable = "/opt/claude/bin/claude"
        model = "opus"

        [catalog]
        mode = "all"

        [plan]
        attempts = 5
        """,
    )
    config = RunConfig.from_toml(path)
    assert config.runtime.options["executable"] == "/opt/claude/bin/claude"
    assert config.runtime.options["model"] == "opus"
    assert config.catalog.mode is CatalogMode.ALL
    assert config.plan.attempts == 5


def test_unset_overrides_cannot_clobber_a_file(minimal: RunConfig) -> None:
    """Verify None values in overrides dictionary preserve existing configured values."""
    configured = minimal.with_overrides(runtime={"timeout_s": 5})
    assert configured.with_overrides(runtime={"timeout_s": None}).runtime.timeout_s == 5


def test_overrides_apply_per_section(minimal: RunConfig) -> None:
    """Verify partial dictionary overrides retain other fields in the target section."""
    changed = minimal.with_overrides(plan={"attempts": 7})
    assert changed.plan.attempts == 7
    assert changed.plan.retries == minimal.plan.retries


def test_the_fingerprint_changes_with_anything_that_moves_the_numbers(minimal) -> None:
    """Verify fingerprint updates when runtime options, catalog seed, or plan depth change."""
    baseline = minimal.fingerprint
    assert minimal.with_overrides(runtime={"options": {"model": "opus"}}).fingerprint != baseline
    assert minimal.with_overrides(catalog={"seed": 1}).fingerprint != baseline
    assert minimal.with_overrides(plan={"attempts": 9}).fingerprint != baseline


def test_the_fingerprint_covers_agent_options(minimal: RunConfig) -> None:
    """Verify changes to runtime agent options alter the configuration fingerprint."""
    claude_cfg = minimal.with_overrides(runtime={"agent": "claude-code"})
    loosened = claude_cfg.with_overrides(
        runtime={"options": {"disable_bundled_skills": False}},
    )
    assert loosened.fingerprint != claude_cfg.fingerprint


def test_spelling_out_an_agent_default_is_the_same_run(minimal: RunConfig) -> None:
    """Verify explicit default agent options yield identical fingerprint and arm."""
    claude_cfg = minimal.with_overrides(runtime={"agent": "claude-code"})
    explicit = claude_cfg.with_overrides(
        runtime={"options": {"setting_sources": "project"}},
    )
    assert explicit.runtime.options == {"setting_sources": "project"}
    assert explicit.fingerprint == claude_cfg.fingerprint
    assert explicit.arm == claude_cfg.arm


def test_the_fingerprint_ignores_where_a_run_writes(minimal: RunConfig, tmp_path: Path) -> None:
    """Verify output path and workdir do not affect the fingerprint."""
    elsewhere = minimal.with_overrides(
        study={"out": tmp_path / "other.jsonl", "workdir": tmp_path / "other"},
    )
    assert elsewhere.fingerprint == minimal.fingerprint


def test_the_fingerprint_ignores_where_the_corpus_is_checked_out(
    minimal: RunConfig,
    tmp_path: Path,
) -> None:
    """Verify filesystem path of the corpus directory does not affect the fingerprint."""
    elsewhere = minimal.with_overrides(study={"skills": tmp_path / "somewhere-else"})
    assert elsewhere.study.skills != minimal.study.skills
    assert elsewhere.fingerprint == minimal.fingerprint


def test_the_fingerprint_ignores_where_the_query_set_sits(
    minimal: RunConfig,
    tmp_path: Path,
) -> None:
    """Verify query set file path does not affect configuration fingerprint."""
    elsewhere = minimal.with_overrides(
        study={"queries": tmp_path / "moved" / "queries.json"},
    )
    assert elsewhere.study.queries != minimal.study.queries
    assert elsewhere.fingerprint == minimal.fingerprint


def test_the_arm_holds_across_the_slices_of_one_sweep(minimal: RunConfig, tmp_path) -> None:
    """Verify arm digest remains consistent across query set slices in a sweep."""
    slice_two = minimal.with_overrides(
        study={"queries": tmp_path / "gke.json", "catalog": "neighborhood:gke"},
    )
    assert slice_two.fingerprint != minimal.fingerprint
    assert slice_two.arm == minimal.arm


def test_the_arm_changes_with_anything_that_moves_the_numbers(
    minimal: RunConfig,
) -> None:
    """Verify arm digest updates when runtime options, catalog seed, or plan depth change."""
    baseline = minimal.arm
    assert minimal.with_overrides(runtime={"options": {"model": "opus"}}).arm != baseline
    assert minimal.with_overrides(catalog={"seed": 1}).arm != baseline
    assert minimal.with_overrides(plan={"attempts": 9}).arm != baseline


def test_the_condition_survives_the_depth_the_arm_does_not(minimal: RunConfig) -> None:
    """Verify condition digest remains identical when only attempt count varies."""
    deeper = minimal.with_overrides(plan={"attempts": 20})
    assert deeper.arm != minimal.arm
    assert deeper.fingerprint != minimal.fingerprint
    assert deeper.condition == minimal.condition


def test_the_condition_changes_with_everything_else_the_arm_does(
    minimal: RunConfig,
) -> None:
    """Verify condition digest changes when model, seed, retries, or agent change."""
    baseline = minimal.condition
    assert minimal.with_overrides(runtime={"options": {"model": "opus"}}).condition != baseline
    assert minimal.with_overrides(catalog={"seed": 1}).condition != baseline
    assert minimal.with_overrides(plan={"retries": 9}).condition != baseline
    assert minimal.with_overrides(runtime={"agent": "fake"}).condition != baseline


def test_the_condition_is_not_the_arm_of_some_other_depth(minimal: RunConfig) -> None:
    """Verify condition digest does not collide with arm digests of various attempt depths."""
    digests = {minimal.condition} | {
        minimal.with_overrides(plan={"attempts": n}).arm for n in range(1, 25)
    }
    assert len(digests) == 25


def test_the_condition_is_as_quotable_as_the_others(minimal: RunConfig) -> None:
    """Verify condition digest is a 12-character string."""
    assert len(minimal.condition) == 12


def test_the_fingerprint_is_stable_across_construction(minimal: RunConfig) -> None:
    """Verify fingerprint is deterministic across dump/validation cycles."""
    twin = RunConfig.model_validate(minimal.model_dump())
    assert twin.fingerprint == minimal.fingerprint


def test_the_fingerprint_is_short_enough_to_quote(minimal: RunConfig) -> None:
    """Verify fingerprint is a 12-character string."""
    assert len(minimal.fingerprint) == 12


def test_the_digests_are_cached_on_frozen_config(
    minimal: RunConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify RunConfig memoizes digests so digest_material() is called only once."""
    import reach.config

    calls = 0
    original_digest = reach.config.digest_material

    def counting_digest(material: dict[str, object]) -> reach.config.Digests:
        nonlocal calls
        calls += 1
        return original_digest(material)

    monkeypatch.setattr("reach.config.digest_material", counting_digest)
    _ = minimal.fingerprint
    _ = minimal.arm
    _ = minimal.condition
    _ = minimal.fingerprint
    assert calls == 1


def test_an_unknown_runtime_setting_is_refused() -> None:
    """Verify ValidationError is raised when unknown fields are passed to RuntimeSettings."""
    with pytest.raises(ValidationError):
        RuntimeSettings.model_validate(
            {"agent": "claude-code", "denied_tools": ["Bash"], "max_turns": 1},
        )


def test_an_unknown_configuration_table_is_refused(tmp_path: Path) -> None:
    """Verify ValidationError is raised when unknown section names are passed to RunConfig."""
    with pytest.raises(ValidationError):
        RunConfig.model_validate(
            {
                "study": {
                    "skills": tmp_path / "corpus",
                    "queries": tmp_path / "q.json",
                    "workdir": tmp_path / "work",
                },
                "plann": {"attempts": 3},
            },
        )


def test_an_unknown_study_setting_is_refused(tmp_path: Path) -> None:
    """Verify ValidationError is raised when unknown study fields are passed to RunConfig."""
    with pytest.raises(ValidationError):
        RunConfig.model_validate(
            {
                "study": {
                    "skills": tmp_path / "corpus",
                    "queries": tmp_path / "q.json",
                    "workdir": tmp_path / "work",
                    "rescoped": True,
                },
            },
        )


def test_study_settings_require_paths_success(minimal: RunConfig, tmp_path: Path) -> None:
    """Verify require_* extractors return expected paths when populated."""
    assert minimal.study.require_queries() == tmp_path / "q.json"
    assert minimal.study.require_skills() == tmp_path / "corpus"
    assert minimal.study.require_workdir() == (tmp_path / "work").resolve()
    assert minimal.require_queries() == tmp_path / "q.json"
    assert minimal.require_skills() == tmp_path / "corpus"
    assert minimal.require_workdir() == (tmp_path / "work").resolve()


def test_study_settings_require_paths_missing() -> None:
    """Verify require_* extractors raise descriptive ValueError when paths are None."""
    empty_config = RunConfig()
    with pytest.raises(ValueError, match="queries path is required"):
        empty_config.study.require_queries()
    with pytest.raises(ValueError, match="queries path is required: custom context"):
        empty_config.require_queries("custom context")
    with pytest.raises(ValueError, match="skills path is required"):
        empty_config.study.require_skills()
    with pytest.raises(ValueError, match="skills path is required: for indexing"):
        empty_config.require_skills("for indexing")
    with pytest.raises(ValueError, match="workdir path is required"):
        empty_config.study.require_workdir()
    with pytest.raises(ValueError, match="workdir path is required: for probe isolation"):
        empty_config.require_workdir("for probe isolation")


#: Test environment variable name for corpus root path expansion tests.
CORPUS_VAR = "REACH_TEST_CORPUS_ROOT"


@pytest.fixture
def unset_corpus_var(monkeypatch: pytest.MonkeyPatch) -> str:
    """Ensure the corpus environment variable is unset and return variable name."""
    monkeypatch.delenv(CORPUS_VAR, raising=False)
    return CORPUS_VAR


@pytest.fixture
def variable_config(tmp_path: Path, query_file: Path) -> Path:
    """Write temporary TOML study configuration with corpus specified via environment variable."""
    path = tmp_path / "study.toml"
    path.write_text(
        "\n".join(
            (
                "[study]",
                f'skills = "${{{CORPUS_VAR}}}"',
                f'queries = "{query_file}"',
                f'workdir = "{tmp_path / "work"}"',
                "",
                "[catalog]",
                'mode = "all"',
                "",
            ),
        ),
        encoding="utf-8",
    )
    return path


def test_a_supplied_corpus_loads_a_config_whose_variable_is_unset(
    variable_config: Path,
    skill_repo: Path,
    unset_corpus_var: str,
) -> None:
    """Verify supplied corpus overrides unset environment variables during TOML load."""
    config = RunConfig.from_toml(variable_config, skills=skill_repo)
    assert config.study.skills == skill_repo
    assert load_corpus(config)


def test_a_supplied_corpus_also_wins_over_one_the_file_could_resolve(
    variable_config: Path,
    skill_repo: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify supplied corpus takes precedence over resolved environment variable."""
    other = tmp_path / "elsewhere"
    other.mkdir()
    monkeypatch.setenv(CORPUS_VAR, str(other))
    config = RunConfig.from_toml(variable_config, skills=skill_repo)
    assert config.study.skills == skill_repo


def test_a_resolvable_variable_is_still_expanded_when_nothing_is_supplied(
    variable_config: Path,
    skill_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify environment variables expand correctly when no explicit corpus override is given."""
    monkeypatch.setenv(CORPUS_VAR, str(skill_repo))
    config = RunConfig.from_toml(variable_config)
    assert config.study.skills == skill_repo


def test_an_unset_variable_still_fails_the_load_when_no_corpus_is_supplied(
    variable_config: Path,
    unset_corpus_var: str,
) -> None:
    """Verify ValueError is raised when environment variable is unset and no override is given."""
    with pytest.raises(ValueError, match=unset_corpus_var) as raised:
        RunConfig.from_toml(variable_config)
    assert "unset environment variable" in str(raised.value)


def test_a_config_that_names_no_corpus_at_all_is_unchanged(
    tmp_path: Path,
    query_file: Path,
) -> None:
    """Verify configuration loading succeeds when corpus is omitted, but load_corpus fails."""
    path = tmp_path / "discover.toml"
    path.write_text(
        "\n".join(
            (
                "[study]",
                f'queries = "{query_file}"',
                f'workdir = "{tmp_path / "work"}"',
                "",
            ),
        ),
        encoding="utf-8",
    )
    config = RunConfig.from_toml(path)
    assert config.study.skills is None
    with pytest.raises(ValueError, match="no skill corpus"):
        load_corpus(config)


def test_a_supplied_corpus_rescues_a_config_that_named_none(
    tmp_path: Path,
    query_file: Path,
    skill_repo: Path,
) -> None:
    """Verify explicit corpus argument populates corpus when TOML config omits corpus."""
    path = tmp_path / "discover.toml"
    path.write_text(
        "\n".join(
            (
                "[study]",
                f'queries = "{query_file}"',
                f'workdir = "{tmp_path / "work"}"',
                "",
            ),
        ),
        encoding="utf-8",
    )
    config = RunConfig.from_toml(path, skills=skill_repo)
    assert config.study.skills == skill_repo


def test_an_unresolvable_queries_path_still_fails_at_load(
    tmp_path: Path,
    unset_corpus_var: str,
) -> None:
    """Verify unresolvable environment variable in queries path raises ValueError during load."""
    path = tmp_path / "bad-queries.toml"
    path.write_text(
        "\n".join(
            (
                "[study]",
                f'queries = "${{{unset_corpus_var}}}/q.json"',
                f'workdir = "{tmp_path / "work"}"',
                "",
            ),
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unset environment variable"):
        RunConfig.from_toml(path)


def test_a_supplied_corpus_does_not_rescue_an_unresolvable_queries_path(
    tmp_path: Path,
    skill_repo: Path,
    unset_corpus_var: str,
) -> None:
    """Verify corpus parameter override does not bypass errors in unset queries path variables."""
    path = tmp_path / "bad-queries.toml"
    path.write_text(
        "\n".join(
            (
                "[study]",
                f'skills = "${{{unset_corpus_var}}}"',
                f'queries = "${{{unset_corpus_var}}}/q.json"',
                f'workdir = "{tmp_path / "work"}"',
                "",
            ),
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unset environment variable"):
        RunConfig.from_toml(path, skills=skill_repo)


def test_expand_path_still_rejects_what_it_always_rejected(
    unset_corpus_var: str,
) -> None:
    """Verify expand_path raises ValueError on unset environment variable tokens."""
    with pytest.raises(ValueError, match="unset environment variable"):
        expand_path(f"${{{unset_corpus_var}}}/skills")


def test_a_relative_corpus_still_anchors_to_the_config_file(
    tmp_path: Path,
    query_file: Path,
) -> None:
    """Verify relative corpus path in TOML config resolves relative to TOML file path."""
    (tmp_path / "corpus").mkdir()
    path = tmp_path / "relative.toml"
    path.write_text(
        "\n".join(
            (
                "[study]",
                'skills = "corpus"',
                f'queries = "{query_file}"',
                f'workdir = "{tmp_path / "work"}"',
                "",
            ),
        ),
        encoding="utf-8",
    )
    config = RunConfig.from_toml(path)
    assert config.study.skills == tmp_path / "corpus"


def test_a_supplied_relative_corpus_anchors_to_the_config_file_too(
    tmp_path: Path,
    query_file: Path,
) -> None:
    """Verify relative corpus path passed to from_toml resolves relative to TOML file path."""
    (tmp_path / "corpus").mkdir()
    path = tmp_path / "supplied-relative.toml"
    path.write_text(
        "\n".join(
            (
                "[study]",
                f'queries = "{query_file}"',
                f'workdir = "{tmp_path / "work"}"',
                "",
            ),
        ),
        encoding="utf-8",
    )
    config = RunConfig.from_toml(path, skills=Path("corpus"))
    assert config.study.skills == tmp_path / "corpus"


def test_supplying_a_corpus_moves_no_fingerprint(
    variable_config: Path,
    skill_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify configuration fingerprint and arm remain unchanged when corpus is overridden."""
    monkeypatch.delenv(CORPUS_VAR, raising=False)
    supplied = RunConfig.from_toml(variable_config, skills=skill_repo)
    monkeypatch.setenv(CORPUS_VAR, str(skill_repo))
    from_file = RunConfig.from_toml(variable_config)
    assert supplied.fingerprint == from_file.fingerprint
    assert supplied.arm == from_file.arm
    assert supplied.condition == from_file.condition


def test_the_loader_gained_no_field(variable_config: Path, skill_repo: Path) -> None:
    """Verify study section fields match expected schema keys without extraneous properties."""
    config = RunConfig.from_toml(variable_config, skills=skill_repo)
    dumped = config.model_dump(mode="json")
    assert set(dumped["study"]) == {
        "skills",
        "queries",
        "workdir",
        "out",
        "catalog",
        "partial",
        "rescope",
        "tag",
        "early_stop",
        "scales",
        "anchor",
        "trusted",
        "auto_queries",
    }
    assert RunConfig.model_validate(dumped).fingerprint == config.fingerprint


def test_skills_overrides_a_corpus_the_file_could_not_resolve(
    variable_config: Path,
    skill_repo: Path,
    unset_corpus_var: str,
) -> None:
    """Verify CLI --skills flag resolves corpus path when config file contains unset variable."""
    config = build_config(
        config=variable_config,
        study=StudyFlags(skills=skill_repo),
        required=(),
    )
    assert config.study.skills == skill_repo


def test_skills_still_overrides_a_corpus_the_file_could_resolve(
    variable_config: Path,
    skill_repo: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify CLI --skills flag takes precedence when config file environment variable is set."""
    other = tmp_path / "elsewhere"
    other.mkdir()
    monkeypatch.setenv(CORPUS_VAR, str(other))
    config = build_config(
        config=variable_config,
        study=StudyFlags(skills=skill_repo),
        required=(),
    )
    assert config.study.skills == skill_repo


def test_no_skills_flag_leaves_the_file_to_answer(
    variable_config: Path,
    skill_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify configuration uses environment variable when CLI flag is omitted."""
    monkeypatch.setenv(CORPUS_VAR, str(skill_repo))
    config = build_config(config=variable_config, required=())
    assert config.study.skills == skill_repo


def test_no_skills_flag_and_an_unset_variable_still_fails_at_the_command_line(
    variable_config: Path,
    unset_corpus_var: str,
) -> None:
    """Verify build_config raises ValueError when CLI flag omitted and env var is unset."""
    with pytest.raises(ValueError, match=unset_corpus_var):
        build_config(config=variable_config, required=())


def test_check_settings_defaults_and_validation() -> None:
    """Verify CheckSettings defaults and threshold validation boundaries."""
    from reach.config import CheckSettings

    settings = CheckSettings()
    assert settings.min_recall == 0.80
    assert settings.min_accuracy == 0.80
    assert settings.max_misroute == 0.10
    assert settings.budget == 50
    assert settings.strict is True
    assert settings.since == "HEAD~1"

    with pytest.raises(ValidationError):
        CheckSettings.model_validate({"min_recall": 1.5})

    with pytest.raises(ValidationError):
        CheckSettings.model_validate({"budget": 0})


def test_run_config_loads_check_and_lint_sections(tmp_path: Path) -> None:
    """Verify RunConfig loads [lint] and [check] tables from toml."""
    toml_path = tmp_path / "reach.toml"
    toml_path.write_text(
        textwrap.dedent(
            """
            [check]
            min_recall = 0.90
            max_misroute = 0.05
            budget = 40
            strict = false
            since = "origin/main"

            [lint]
            max_name_length = 32
            """,
        ),
        encoding="utf-8",
    )
    config = RunConfig.from_toml(toml_path)
    assert config.check.min_recall == 0.90
    assert config.check.max_misroute == 0.05
    assert config.check.budget == 40
    assert config.check.strict is False
    assert config.check.since == "origin/main"
    assert config.lint.max_name_length == 32


def test_all_known_agents_have_skills_dir() -> None:
    """Verify every execution agent runtime declares a native skills directory."""
    from reach.config import KNOWN_CLIENT_SKILLS_DIRS, agent_profiles
    from reach.runtime import known_agents
    from reach.runtime.fake import FAKE_AGENT

    profiles = agent_profiles()
    execution_agents = [a for a in known_agents() if a not in ("keyword", FAKE_AGENT)]

    for agent in execution_agents:
        has_dir = (
            agent in profiles and bool(profiles[agent].skills_dir)
        ) or agent in KNOWN_CLIENT_SKILLS_DIRS
        assert has_dir, (
            f"Agent runtime {agent!r} has no declared skills directory in "
            f"agent profiles or KNOWN_CLIENT_SKILLS_DIRS"
        )


def test_client_skills_directory_symmetry() -> None:
    """Verify workspace and global client directory registries cover identical client profiles."""
    from reach.config import KNOWN_CLIENT_GLOBAL_SKILLS_DIRS, KNOWN_CLIENT_SKILLS_DIRS

    workspace_clients = set(KNOWN_CLIENT_SKILLS_DIRS.keys())
    global_clients = set(KNOWN_CLIENT_GLOBAL_SKILLS_DIRS.keys())
    assert workspace_clients == global_clients, (
        f"Mismatch between workspace and global client directories: "
        f"only in workspace={workspace_clients - global_clients}, "
        f"only in global={global_clients - workspace_clients}"
    )


@pytest.mark.parametrize("agent", _EXECUTION_AGENTS)
def test_known_agent_elevates_workspace_skills_dir(tmp_path: Path, agent: str) -> None:
    """Verify that specifying an execution agent elevates its native skills directory."""
    from reach.config import KNOWN_CLIENT_SKILLS_DIRS, agent_profiles, resolve_discovery_candidates

    profiles = agent_profiles()
    expected_rel = (
        profiles[agent].skills_dir
        if (agent in profiles and profiles[agent].skills_dir)
        else KNOWN_CLIENT_SKILLS_DIRS.get(agent)
    )
    assert expected_rel is not None

    candidates = resolve_discovery_candidates(tmp_path, agent=agent)
    assert len(candidates) >= 3
    assert candidates[0] == tmp_path
    assert candidates[1] == tmp_path / "skills"
    assert candidates[2] == tmp_path / expected_rel


@pytest.mark.parametrize("agent", _EXECUTION_AGENTS)
def test_known_agent_elevates_global_skills_dir(agent: str, tmp_path: Path) -> None:
    """Verify specifying an execution agent elevates its native user global skills directory."""
    from reach.config import (
        KNOWN_CLIENT_GLOBAL_SKILLS_DIRS,
        KNOWN_CLIENT_SKILLS_DIRS,
        agent_profiles,
        resolve_discovery_candidates,
    )

    profiles = agent_profiles()
    home = Path.home()
    if agent in profiles and (user_dir := profiles[agent].user_skills_dir):
        expected_path = home / user_dir
    elif agent in KNOWN_CLIENT_GLOBAL_SKILLS_DIRS:
        expected_path = home / KNOWN_CLIENT_GLOBAL_SKILLS_DIRS[agent][0]
    else:
        expected_path = home / KNOWN_CLIENT_SKILLS_DIRS[agent]

    candidates = resolve_discovery_candidates(tmp_path, agent=agent, global_scope=True)
    assert len(candidates) > 0
    assert candidates[0] == expected_path


def test_custom_agent_profile_elevates_custom_skills_dir(tmp_path: Path) -> None:
    """Verify custom agent configured in reach.toml elevates its declared skills_dir."""
    from reach.config import resolve_discovery_candidates

    custom_toml = tmp_path / "reach.toml"
    custom_toml.write_text(
        textwrap.dedent(
            """
            [agents.custom-bot]
            skills_dir = ".custom/skills"
            """,
        ),
        encoding="utf-8",
    )
    candidates = resolve_discovery_candidates(
        tmp_path,
        agent="custom-bot",
        config_path=custom_toml,
    )
    assert candidates[2] == tmp_path / ".custom" / "skills"


def test_discovery_settings_default_precedence() -> None:
    """Verify default discovery precedence order."""
    from reach.config import DiscoverySettings

    settings = DiscoverySettings()
    assert settings.precedence == (
        ".",
        "skills",
        ".agents/skills",
        "claude-code",
        "cursor",
        "github",
        "pi",
        "goose",
    )


def test_discovery_settings_rejects_extra_fields() -> None:
    """Verify DiscoverySettings forbids unknown extra keys."""
    from reach.config import DiscoverySettings

    with pytest.raises(ValidationError):
        DiscoverySettings.model_validate({"unknown_key": "val"})


def test_resolve_discovery_candidates_order(tmp_path: Path) -> None:
    """Verify discovery candidate paths are resolved in precedence order."""
    from reach.config import resolve_discovery_candidates

    candidates = resolve_discovery_candidates(tmp_path)
    expected = [
        tmp_path,
        tmp_path / "skills",
        tmp_path / ".agents" / "skills",
        tmp_path / ".claude" / "skills",
        tmp_path / ".cursor" / "skills",
        tmp_path / ".github" / "skills",
        tmp_path / ".pi" / "skills",
    ]
    assert candidates == expected


def test_resolve_discovery_candidates_elevates_specified_agent(tmp_path: Path) -> None:
    """Verify specified agent elevates its native skills directory."""
    from reach.config import resolve_discovery_candidates

    candidates = resolve_discovery_candidates(tmp_path, agent="claude-code")
    expected = [
        tmp_path,
        tmp_path / "skills",
        tmp_path / ".claude" / "skills",
        tmp_path / ".agents" / "skills",
        tmp_path / ".cursor" / "skills",
        tmp_path / ".github" / "skills",
        tmp_path / ".pi" / "skills",
    ]
    assert candidates == expected


def test_resolve_discovery_candidates_elevates_cursor(tmp_path: Path) -> None:
    """Verify cursor agent elevates .cursor/skills directory."""
    from reach.config import resolve_discovery_candidates

    candidates = resolve_discovery_candidates(tmp_path, agent="cursor")
    expected = [
        tmp_path,
        tmp_path / "skills",
        tmp_path / ".cursor" / "skills",
        tmp_path / ".agents" / "skills",
        tmp_path / ".claude" / "skills",
        tmp_path / ".github" / "skills",
        tmp_path / ".pi" / "skills",
    ]
    assert candidates == expected


def test_resolve_discovery_candidates_elevates_copilot_or_github(tmp_path: Path) -> None:
    """Verify copilot / github agent elevates .github/skills directory."""
    from reach.config import resolve_discovery_candidates

    for alias in ("github", "copilot"):
        candidates = resolve_discovery_candidates(tmp_path, agent=alias)
        expected = [
            tmp_path,
            tmp_path / "skills",
            tmp_path / ".github" / "skills",
            tmp_path / ".agents" / "skills",
            tmp_path / ".claude" / "skills",
            tmp_path / ".cursor" / "skills",
            tmp_path / ".pi" / "skills",
        ]
        assert candidates == expected


def test_resolve_discovery_candidates_elevates_codex(tmp_path: Path) -> None:
    """Verify codex agent prioritizes .agents/skills directory."""
    from reach.config import resolve_discovery_candidates

    candidates = resolve_discovery_candidates(tmp_path, agent="codex")
    expected = [
        tmp_path,
        tmp_path / "skills",
        tmp_path / ".agents" / "skills",
        tmp_path / ".claude" / "skills",
        tmp_path / ".cursor" / "skills",
        tmp_path / ".github" / "skills",
        tmp_path / ".pi" / "skills",
    ]
    assert candidates == expected


def test_resolve_discovery_candidates_deduplicates_duplicate_paths(tmp_path: Path) -> None:
    """Verify discovery candidate paths are deduplicated properly."""
    from reach.config import resolve_discovery_candidates

    # Custom configuration containing duplicate precedence paths
    custom_toml = tmp_path / "custom.toml"
    custom_toml.write_text(
        "[discovery]\n"
        'precedence = [".", "skills", ".agents/skills", "codex", "agents", "cursor"]\n',
        encoding="utf-8",
    )
    candidates = resolve_discovery_candidates(tmp_path, config_path=custom_toml)
    expected = [
        tmp_path,
        tmp_path / "skills",
        tmp_path / ".agents" / "skills",
        tmp_path / ".cursor" / "skills",
    ]
    assert candidates == expected


def test_resolve_discovery_candidates_global_scope(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify discovery candidate paths in global scope resolve under home directory."""
    from reach.config import resolve_discovery_candidates

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    candidates = resolve_discovery_candidates(tmp_path, global_scope=True)
    expected = [
        tmp_path / ".agents" / "skills",
        tmp_path / ".claude" / "skills",
        tmp_path / ".cursor" / "skills",
        tmp_path / ".copilot" / "skills",
        tmp_path / ".pi" / "agent" / "skills",
    ]
    assert candidates == expected


@pytest.mark.parametrize(
    ("agent", "expected_first"),
    [
        ("claude-code", ".claude/skills"),
        ("antigravity-cli", ".agents/skills"),
        ("cursor", ".cursor/skills"),
        ("copilot", ".copilot/skills"),
        ("github", ".copilot/skills"),
        ("pi", ".pi/agent/skills"),
        ("codex", ".agents/skills"),
    ],
)
def test_resolve_discovery_candidates_global_scope_elevates_agent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    agent: str,
    expected_first: str,
) -> None:
    """Verify that specifying an agent in global scope elevates its global directory."""
    from reach.config import resolve_discovery_candidates

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    candidates = resolve_discovery_candidates(tmp_path, agent=agent, global_scope=True)
    assert candidates[0] == tmp_path / expected_first


def test_retrieval_settings_defaults_and_validation() -> None:
    """Verify RetrievalSettings defaults for BM25 and range constraints."""
    from reach.config import RetrievalSettings

    settings = RetrievalSettings()
    assert settings.bm25_k1 == 1.5
    assert settings.bm25_b == 0.75

    with pytest.raises(ValidationError):
        RetrievalSettings(bm25_k1=0.0)

    with pytest.raises(ValidationError):
        RetrievalSettings(bm25_b=-0.1)

    with pytest.raises(ValidationError):
        RetrievalSettings(bm25_b=1.1)


def test_overlap_settings_defaults_and_validation() -> None:
    """Verify OverlapSettings defaults and range constraints."""
    from reach.config import OverlapSettings

    settings = OverlapSettings()
    assert settings.contender_band == 0.90
    assert settings.material_share == 0.01
    assert settings.claim_limit == 8
    assert settings.min_claim_length == 3
    assert settings.min_claim_uses == 2

    with pytest.raises(ValidationError):
        OverlapSettings.model_validate({"contender_band": 1.5})

    with pytest.raises(ValidationError):
        OverlapSettings.model_validate({"claim_limit": 0})


def test_diff_settings_defaults_and_validation() -> None:
    """Verify DiffSettings defaults and confidence interval bounds."""
    from reach.config import DiffSettings

    settings = DiffSettings()
    assert settings.confidence == 0.95
    assert settings.power == 0.80
    assert settings.noise_inflation == 1.265
    assert settings.over_dispersion == pytest.approx(1.60, abs=0.01)

    with pytest.raises(ValidationError):
        DiffSettings.model_validate({"confidence": 0.0})

    with pytest.raises(ValidationError):
        DiffSettings.model_validate({"confidence": 1.0})

    with pytest.raises(ValidationError):
        DiffSettings.model_validate({"over_dispersion": 1.60})


def test_query_settings_defaults_and_validation() -> None:
    """Verify QuerySettings defaults and count constraints."""
    from reach.config import QuerySettings

    settings = QuerySettings()
    assert settings.count == 3
    assert settings.adversarial_count == 1
    assert settings.top_rivals == 3
    assert pytest.approx(settings.distinctive_idf_floor, rel=1e-3) == 0.693147

    with pytest.raises(ValidationError):
        QuerySettings.model_validate({"count": 0})


def test_optimize_settings_defaults_and_validation() -> None:
    """Verify OptimizeSettings defaults and budget constraints."""
    from reach.config import OptimizeSettings

    settings = OptimizeSettings()
    assert settings.budget == 30
    assert settings.temperature == 0.7

    with pytest.raises(ValidationError):
        OptimizeSettings.model_validate({"budget": 0})


def test_run_config_loads_all_custom_sections(tmp_path: Path) -> None:
    """Verify RunConfig loads custom sections from TOML correctly."""
    toml_file = write_toml(
        tmp_path,
        """
        [retrieval]
        bm25_k1 = 1.2
        bm25_b = 0.60
        scorer = "bm25"

        [overlap]
        contender_band = 0.85
        material_share = 0.02
        claim_limit = 12

        [diff]
        confidence = 0.90
        power = 0.85
        noise_inflation = 1.40

        [query]
        count = 5
        adversarial_count = 2
        top_rivals = 4

        [optimize]
        budget = 40
        temperature = 0.8
        """,
    )
    cfg = RunConfig.from_toml(toml_file)
    assert cfg.retrieval.bm25_k1 == 1.2
    assert cfg.retrieval.bm25_b == 0.60
    assert cfg.retrieval.scorer == "bm25"
    assert cfg.overlap.contender_band == 0.85
    assert cfg.overlap.material_share == 0.02
    assert cfg.overlap.claim_limit == 12
    assert cfg.diff.confidence == 0.90
    assert cfg.diff.power == 0.85
    assert cfg.diff.noise_inflation == 1.40
    assert cfg.query.count == 5
    assert cfg.query.adversarial_count == 2
    assert cfg.query.top_rivals == 4
    assert cfg.optimize.budget == 40
    assert cfg.optimize.temperature == 0.8


def test_retrieval_build_scorer_respects_custom_bm25_settings(tmp_path: Path) -> None:
    """Verify build_scorer passes custom BM25 settings from RunConfig."""
    from reach.models import Skill
    from reach.retrieval import Bm25Scorer, build_scorer

    skills = [
        Skill(name="alpha", description="alpha description", path=Path("alpha/SKILL.md")),
        Skill(name="beta", description="beta description", path=Path("beta/SKILL.md")),
    ]
    toml_file = write_toml(
        tmp_path,
        """
        [retrieval]
        bm25_k1 = 2.0
        bm25_b = 0.50
        """,
    )
    cfg = RunConfig.from_toml(toml_file)
    scorer = build_scorer("bm25", skills, config=cfg)
    assert isinstance(scorer, Bm25Scorer)
    assert scorer.k1 == 2.0
    assert scorer.b == 0.50


def test_resolve_settings_across_sections() -> None:
    """Verify section resolvers merge overrides with settings models and RunConfig."""
    from reach.config import (
        DiffSettings,
        DiscoverySettings,
        OptimizeSettings,
        OverlapSettings,
        QuerySettings,
        RetrievalSettings,
        RunConfig,
    )

    diff_res = RunConfig.resolve(
        DiffSettings,
        explicit_settings=DiffSettings(confidence=0.99),
        noise_inflation=1.5,
    )
    assert diff_res.confidence == 0.99
    assert diff_res.noise_inflation == 1.5

    opt_res = RunConfig.resolve(
        OptimizeSettings,
        explicit_settings=OptimizeSettings(budget=50),
        temperature=1.2,
    )
    assert opt_res.budget == 50
    assert opt_res.temperature == 1.2

    over_res = RunConfig.resolve(
        OverlapSettings,
        explicit_settings=OverlapSettings(claim_limit=12),
        contender_band=0.85,
    )
    assert over_res.claim_limit == 12
    assert over_res.contender_band == 0.85

    disc_res = RunConfig.resolve(
        DiscoverySettings,
        explicit_settings=DiscoverySettings(precedence=("a", "b")),
        precedence=("custom",),
    )
    assert disc_res.precedence == ("custom",)

    query_res = RunConfig.resolve(
        QuerySettings,
        explicit_settings=QuerySettings(distinctive_idf_floor=2.5),
        distinctive_idf_floor=3.0,
    )
    assert query_res.distinctive_idf_floor == 3.0

    ret_res = RunConfig.resolve(
        RetrievalSettings,
        explicit_settings=RetrievalSettings(bm25_k1=1.8),
        bm25_k1=2.0,
    )
    assert ret_res.bm25_k1 == 2.0

    # With RunConfig
    cfg = RunConfig(
        diff=DiffSettings(confidence=0.90),
        discovery=DiscoverySettings(precedence=("cfg_prec",)),
        optimize=OptimizeSettings(budget=20),
        overlap=OverlapSettings(claim_limit=5),
        query=QuerySettings(distinctive_idf_floor=1.8),
        retrieval=RetrievalSettings(bm25_k1=1.4),
    )
    assert RunConfig.resolve(DiffSettings, config=cfg).confidence == 0.90
    assert RunConfig.resolve(DiscoverySettings, config=cfg).precedence == ("cfg_prec",)
    assert RunConfig.resolve(OptimizeSettings, config=cfg).budget == 20
    assert RunConfig.resolve(OverlapSettings, config=cfg).claim_limit == 5
    assert RunConfig.resolve(QuerySettings, config=cfg).distinctive_idf_floor == 1.8
    assert RunConfig.resolve(RetrievalSettings, config=cfg).bm25_k1 == 1.4


def test_resolve_sub_settings_precedence_hierarchy() -> None:
    """Verify multi-tier precedence: overrides > explicit_settings > config_section > defaults."""
    from reach.config import (
        DiffSettings,
        RunConfig,
        resolve_sub_settings,
    )

    # 1. Defaults only
    default_res = resolve_sub_settings(DiffSettings)
    assert default_res.confidence == 0.95
    assert default_res.noise_inflation == 1.265

    # 2. Config section overrides default
    cfg = RunConfig(diff=DiffSettings(confidence=0.80, noise_inflation=1.0))
    from_cfg = RunConfig.resolve(DiffSettings, config=cfg)
    assert from_cfg.confidence == 0.80
    assert from_cfg.noise_inflation == 1.0

    # 3. Explicit settings override config section
    explicit = DiffSettings(confidence=0.85, noise_inflation=1.1)
    from_explicit = RunConfig.resolve(DiffSettings, config=cfg, explicit_settings=explicit)
    assert from_explicit.confidence == 0.85
    assert from_explicit.noise_inflation == 1.1

    # 4. Keyword overrides override explicit settings (and thus config and defaults)
    from_overrides = RunConfig.resolve(
        DiffSettings,
        config=cfg,
        explicit_settings=explicit,
        confidence=0.99,
    )
    assert from_overrides.confidence == 0.99
    assert from_overrides.noise_inflation == 1.1  # Preserved from explicit

    # 5. None in overrides does NOT clobber underlying settings
    from_none_override = RunConfig.resolve(
        DiffSettings,
        config=cfg,
        explicit_settings=explicit,
        confidence=None,
    )
    assert from_none_override.confidence == 0.85


def test_lint_rules_declared_in_example_config() -> None:
    """Verify every lint rule in RULES is documented in reach.example.toml."""
    import tomllib

    from reach.lint import RULES

    root = Path(__file__).resolve().parent.parent
    example_config_path = root / "reach.example.toml"
    config = tomllib.loads(example_config_path.read_text(encoding="utf-8"))
    lint_rules = config.get("lint", {}).get("rules", {})

    for rule_id in RULES:
        assert rule_id in lint_rules, (
            f"Rule {rule_id!r} is missing from [lint.rules] in reach.example.toml"
        )


def test_reach_example_toml_matches_base_structure() -> None:
    """Verify reach.example.toml has valid TOML structure and documents public agents."""
    import tomllib

    from reach.runtime import FAKE_AGENT, known_agents

    root = Path(__file__).resolve().parent.parent
    example_path = root / "reach.example.toml"
    assert example_path.is_file(), "reach.example.toml missing"

    active_content = example_path.read_text(encoding="utf-8")
    parsed_active = tomllib.loads(active_content)
    assert "general" in parsed_active
    assert "discovery" in parsed_active
    assert "check" in parsed_active
    assert "lint" in parsed_active

    public_agents = [a for a in known_agents() if a != FAKE_AGENT]
    for agent in public_agents:
        assert agent in active_content, f"Agent {agent!r} missing from reach.example.toml"


def test_reach_example_toml_documents_sample_models() -> None:
    """Verify reach.example.toml documents sample model profile overrides for default models."""
    from reach.config import DEFAULT_CLAUDE_MODEL, DEFAULT_GEMINI_MODEL

    root = Path(__file__).resolve().parent.parent
    example_path = root / "reach.example.toml"
    content = example_path.read_text(encoding="utf-8")
    assert "models." in content, "reach.example.toml should document sample [models.<id>] overrides"

    gemini_key = DEFAULT_GEMINI_MODEL.replace(".", "-")
    claude_key = DEFAULT_CLAUDE_MODEL.replace(".", "-")
    assert gemini_key in content, f"Expected {gemini_key!r} in reach.example.toml"
    assert claude_key in content, f"Expected {claude_key!r} in reach.example.toml"


def test_reach_example_toml_documents_optimize_settings() -> None:
    """Verify reach.example.toml documents all OptimizeSettings fields."""
    import tomllib

    from reach.config import OptimizeSettings

    root = Path(__file__).resolve().parent.parent
    example_path = root / "reach.example.toml"
    config = tomllib.loads(example_path.read_text(encoding="utf-8"))
    optimize_sec = config.get("optimize", {})

    for field_name in OptimizeSettings.model_fields:
        assert field_name in optimize_sec, (
            f"Field {field_name!r} missing from [optimize] in reach.example.toml"
        )


def test_reach_example_toml_sections_and_keys_are_alphabetized() -> None:
    """Verify active sections and keys in reach.example.toml are alphabetized."""
    import re
    import tomllib

    root = Path(__file__).resolve().parent.parent
    text = (root / "reach.example.toml").read_text(encoding="utf-8")
    sections = [
        m.group(1)
        for line in text.splitlines()
        if (m := re.match(r"^\[([a-zA-Z0-9_.-]+)\]$", line.strip()))
    ]
    assert sections[0] == "general"
    assert sections[1:] == sorted(sections[1:]), (
        f"Sections after [general] in reach.example.toml are not alphabetized: {sections[1:]}"
    )

    config = tomllib.loads(text)
    for sec_name, sec_val in config.items():
        if isinstance(sec_val, dict):
            sub_keys = [k for k, v in sec_val.items() if not isinstance(v, dict)]
            assert sub_keys == sorted(sub_keys), (
                f"Keys in [{sec_name}] of reach.example.toml are not alphabetized: {sub_keys}"
            )
            for nested_name, nested_val in sec_val.items():
                if isinstance(nested_val, dict):
                    nested_keys = list(nested_val.keys())
                    assert nested_keys == sorted(nested_keys), (
                        f"Keys in [{sec_name}.{nested_name}] of reach.example.toml "
                        f"are not alphabetized: {nested_keys}"
                    )


def test_resolve_sub_settings_revalidates_overrides() -> None:
    """Verify resolve_sub_settings validates overrides and rejects out-of-range values."""
    from reach.config import CheckSettings

    resolved = RunConfig.resolve(CheckSettings, min_recall=0.8)
    assert resolved.min_recall == 0.8

    with pytest.raises(ValidationError):
        RunConfig.resolve(CheckSettings, min_recall=-0.5)


def test_scaling_study_requires_target_skill_when_targeted() -> None:
    """Verify ScalingStudy raises ValueError when target_skill is None for targeted sweep."""
    from reach.sweep import ScalingStudy

    with pytest.raises(ValidationError, match="Targeted scaling sweep requires target_skill"):
        ScalingStudy(
            target_skill=None,
            is_corpus_sweep=False,
            scales=(1, 5),
            points=(),
            baseline_pass_rate=1.0,
            final_pass_rate=1.0,
            total_delta=0.0,
            total_context_loss=0.0,
            total_shadowing_loss=0.0,
        )


def test_default_model_constants_have_valid_profiles() -> None:
    """Verify default Gemini and Claude model constants are defined and have model profiles."""
    from reach.config import (
        BUILTIN_AGENT_DEFAULT_MODELS,
        DEFAULT_CLAUDE_MODEL,
        DEFAULT_GEMINI_MODEL,
    )
    from reach.runtime.profiles import model_profile

    assert isinstance(DEFAULT_GEMINI_MODEL, str)
    assert DEFAULT_GEMINI_MODEL
    assert isinstance(DEFAULT_CLAUDE_MODEL, str)
    assert DEFAULT_CLAUDE_MODEL

    for const_model in (DEFAULT_GEMINI_MODEL, DEFAULT_CLAUDE_MODEL):
        profile = model_profile(const_model)
        assert profile.context_window > 0
        assert profile.chars_per_token > 0

    for model_id in BUILTIN_AGENT_DEFAULT_MODELS.values():
        assert isinstance(model_id, str)
        assert model_id
        profile = model_profile(model_id)
        assert profile.context_window > 0


def test_agent_default_model_builtin_resolution() -> None:
    """Verify agent_default_model resolves builtin agent defaults and returns None for unknown."""
    from reach.config import BUILTIN_AGENT_DEFAULT_MODELS, agent_default_model

    for agent, expected_model in BUILTIN_AGENT_DEFAULT_MODELS.items():
        assert agent_default_model(agent) == expected_model

    assert agent_default_model("unregistered-agent-xyz") is None


def test_agent_default_model_custom_override(tmp_path: Path) -> None:
    """Verify agent_default_model respects custom configuration file overrides."""
    from reach.config import agent_default_model

    custom_toml = tmp_path / "custom_reach.toml"
    custom_toml.write_text(
        textwrap.dedent(
            """
            [agents.antigravity-cli]
            default_model = "gemini-2.5-pro"

            [agents.custom-agent]
            default_model = "custom-llm-v1"
            """,
        ),
        encoding="utf-8",
    )
    assert agent_default_model("antigravity-cli", config_path=custom_toml) == "gemini-2.5-pro"
    assert agent_default_model("custom-agent", config_path=custom_toml) == "custom-llm-v1"
    assert agent_default_model("claude-code", config_path=custom_toml) == "claude-sonnet-5"


def test_study_settings_scales_configuration(tmp_path: Path) -> None:
    """Verify StudySettings accepts scales as sequence, comma-delimited string, or TOML."""
    # Direct model construction with list of ints
    s1 = StudySettings(scales=[10, 25, 50])
    assert s1.scales == (10, 25, 50)

    # Coercion from comma-separated string
    s2 = StudySettings(scales="10, 25, 50")
    assert s2.scales == (10, 25, 50)

    # Empty / None default
    s3 = StudySettings()
    assert s3.scales is None

    # Load from reach.toml
    toml_path = write_toml(
        tmp_path,
        """
        [study]
        scales = [10, 25, 50, 100]
        """,
    )
    cfg = RunConfig.from_toml(toml_path)
    assert cfg.study.scales == (10, 25, 50, 100)

    # Invalid scale value raises ValidationError
    with pytest.raises(ValidationError):
        StudySettings(scales="invalid-scale")


def test_study_settings_anchor_configuration(tmp_path: Path) -> None:
    """Verify StudySettings accepts anchor as int, sequence, comma-delimited string, or 'all'."""
    # Integer cohort size
    s1 = StudySettings(anchor=10)
    assert s1.anchor == 10

    # String integer coercion
    s2 = StudySettings(anchor="15")
    assert s2.anchor == 15

    # "all" mode
    s3 = StudySettings(anchor="all")
    assert s3.anchor == "all"
    assert StudySettings(anchor="ALL").anchor == "all"

    # Explicit skill list
    s4 = StudySettings(anchor=["skill-a", "skill-b"])
    assert s4.anchor == ("skill-a", "skill-b")

    # Comma-separated string of skill names
    s5 = StudySettings(anchor="skill-a, skill-b")
    assert s5.anchor == ("skill-a", "skill-b")

    # Default None
    assert StudySettings().anchor is None

    # Load from reach.toml
    toml_path = write_toml(
        tmp_path,
        """
        [study]
        anchor = 12
        """,
    )
    cfg = RunConfig.from_toml(toml_path)
    assert cfg.study.anchor == 12

    # Invalid anchor values raise ValidationError
    with pytest.raises(ValidationError):
        StudySettings(anchor=0)
    with pytest.raises(ValidationError):
        StudySettings(anchor=-5)


def test_run_config_resolve_section_instance_and_class_method() -> None:
    """Verify RunConfig.resolve and resolve_section layer overrides over configuration."""
    config = RunConfig(study=StudySettings(anchor=42))

    # Instance method with overrides
    study_resolved = config.resolve_section(StudySettings, anchor=99)
    assert study_resolved.anchor == 99

    # Instance method without overrides retains config values
    study_retained = config.resolve_section(StudySettings)
    assert study_retained.anchor == 42

    # Class method with config
    class_resolved = RunConfig.resolve(StudySettings, config=config, anchor=100)
    assert class_resolved.anchor == 100

    # Class method without config uses defaults with overrides
    default_resolved = RunConfig.resolve(StudySettings, anchor=7)
    assert default_resolved.anchor == 7


def test_run_config_resolve_registry_and_runtime_specializations() -> None:
    """Verify RunConfig.resolve handles RegistrySettings env and RuntimeSettings model."""
    from reach.config import RegistrySettings

    # RuntimeSettings model override updates options and preserves other options
    rt = RunConfig.resolve(
        RuntimeSettings,
        agent="claude-code",
        options={"executable": "/bin/claude"},
        model="claude-3-5-sonnet",
    )
    assert rt.options.get("model") == "claude-3-5-sonnet"
    assert rt.options.get("executable") == "/bin/claude"

    # RegistrySettings env resolution
    reg = RunConfig.resolve(RegistrySettings, project="my-project")
    assert reg.project == "my-project"
    assert reg.location == "global"


def test_registry_flags_overrides_do_not_clobber_config() -> None:
    """Verify RegistryFlags overrides do not clobber configuration settings when unspecified."""
    from reach.cli.flags import RegistryFlags
    from reach.config import RegistrySettings, resolve_sub_settings

    flags = RegistryFlags()
    assert flags.overrides() == {}

    flags_with_project = RegistryFlags(project="my-project")
    assert flags_with_project.overrides() == {"project": "my-project"}

    base_config = RunConfig(registry=RegistrySettings(project="initial", fresh=True, no_cache=True))
    resolved = resolve_sub_settings(
        RegistrySettings,
        base_config.registry,
        **flags_with_project.overrides(),
    )
    assert resolved.project == "my-project"
    assert resolved.fresh is True
    assert resolved.no_cache is True

    flags_with_fresh = RegistryFlags(fresh=True)
    assert flags_with_fresh.overrides() == {"fresh": True}
    base_no_fresh = RunConfig(registry=RegistrySettings(fresh=False))
    resolved_fresh = resolve_sub_settings(
        RegistrySettings,
        base_no_fresh.registry,
        **flags_with_fresh.overrides(),
    )
    assert resolved_fresh.fresh is True


def test_resolve_discovery_candidates_elevates_antigravity(tmp_path: Path) -> None:
    """Verify resolve_discovery_candidates elevates .agents/skills for antigravity-cli."""
    from reach.config import resolve_discovery_candidates

    candidates = resolve_discovery_candidates(tmp_path, agent="antigravity-cli")
    assert tmp_path / ".agents" / "skills" in candidates


def test_plan_settings_workers_validation_and_digest_invariance(tmp_path: Path) -> None:
    """Verify PlanSettings.workers validates >= 1 and does not change config digests."""
    from reach.config import PlanSettings

    assert PlanSettings().workers == 1
    assert PlanSettings(workers=8).workers == 8
    with pytest.raises(ValidationError):
        PlanSettings(workers=int("0"))

    toml_path = write_toml(
        tmp_path,
        """
        [plan]
        attempts = 3
        workers = 8
        """,
    )
    cfg_8 = RunConfig.from_toml(toml_path)
    assert cfg_8.plan.workers == 8

    cfg_1 = cfg_8.with_overrides(plan={"workers": 1})
    assert cfg_1.plan.workers == 1
    assert cfg_8.fingerprint == cfg_1.fingerprint
    assert cfg_8.arm == cfg_1.arm
    assert cfg_8.condition == cfg_1.condition


def test_run_config_validates_load_config_output_while_forbidding_unknown_keys(
    tmp_path: Path,
) -> None:
    """Verify RunConfig.model_validate accepts load_config() dicts containing agents/models."""
    from reach.config import load_config

    loaded_default = load_config()
    assert "agents" in loaded_default
    assert "models" in loaded_default
    cfg_default = RunConfig.model_validate(loaded_default)
    assert isinstance(cfg_default, RunConfig)

    custom_toml = write_toml(
        tmp_path,
        """
        [plan]
        attempts = 2
        workers = 4
        """,
    )
    cfg_custom = RunConfig.model_validate(load_config(custom_toml))
    assert cfg_custom.plan.attempts == 2
    assert cfg_custom.plan.workers == 4

    with pytest.raises(ValidationError):
        RunConfig.model_validate({**loaded_default, "unknown_section": {}})


def test_run_config_from_toml_and_resolve_presence_based_optimize_workers_inheritance(
    tmp_path: Path,
) -> None:
    """Verify RunConfig inherits plan.workers into optimize.workers via model_fields_set."""
    from reach.config import OptimizeSettings, RuntimeSettings

    # Case 1: [plan] workers = 1 explicitly set, [optimize] workers unset -> inherits 1
    d1 = tmp_path / "c1"
    d1.mkdir()
    toml_inherit_1 = write_toml(
        d1,
        """
        [plan]
        workers = 1
        """,
    )
    cfg_1 = RunConfig.from_toml(toml_inherit_1)
    assert cfg_1.optimize.workers == 1
    assert RunConfig.resolve(OptimizeSettings, cfg_1).workers == 1

    # Case 2: [plan] workers = 8 and [optimize] workers = 4 explicitly set -> preserves 4
    d2 = tmp_path / "c2"
    d2.mkdir()
    toml_explicit_4 = write_toml(
        d2,
        """
        [plan]
        workers = 8
        [optimize]
        workers = 4
        """,
    )
    cfg_4 = RunConfig.from_toml(toml_explicit_4)
    assert cfg_4.optimize.workers == 4
    assert RunConfig.resolve(OptimizeSettings, cfg_4).workers == 4

    # Case 3: RuntimeSettings.resolve_for_optimize merges [runtime.options] with overrides
    d3 = tmp_path / "c3"
    d3.mkdir()
    toml_rt = write_toml(
        d3,
        """
        [runtime]
        agent = "antigravity-sdk"
        [runtime.options]
        vertex = true
        project = "test-cloud-project-123"
        """,
    )
    rt = RuntimeSettings.resolve_for_optimize(
        toml_rt,
        agent="antigravity-sdk",
        options={"location": "us-central1"},
    )
    assert rt.agent == "antigravity-sdk"
    assert rt.options["vertex"] is True
    assert rt.options["project"] == "test-cloud-project-123"
    assert rt.options["location"] == "us-central1"


def test_run_config_inherits_runtime_agent_from_general(tmp_path: Path) -> None:
    """Verify RunConfig inherits general.default_agent into runtime.agent when unset."""
    # Case 1: [general] default_agent without [runtime] section
    d1 = tmp_path / "c1"
    d1.mkdir()
    toml_general = write_toml(
        d1,
        """
        [general]
        default_agent = "antigravity-sdk"
        """,
    )
    cfg1 = RunConfig.from_toml(toml_general)
    assert cfg1.general.default_agent == "antigravity-sdk"
    assert cfg1.runtime.agent == "antigravity-sdk"

    # Case 2: [general] default_agent with [runtime.options] but no explicit agent
    d2 = tmp_path / "c2"
    d2.mkdir()
    toml_opts = write_toml(
        d2,
        """
        [general]
        default_agent = "antigravity-sdk"
        [runtime.options]
        use_symlinks = false
        """,
    )
    cfg2 = RunConfig.from_toml(toml_opts)
    assert cfg2.general.default_agent == "antigravity-sdk"
    assert cfg2.runtime.agent == "antigravity-sdk"

    # Case 3: [general] default_agent with explicit [runtime] agent preserves explicit agent
    d3 = tmp_path / "c3"
    d3.mkdir()
    toml_explicit = write_toml(
        d3,
        """
        [general]
        default_agent = "antigravity-sdk"
        [runtime]
        agent = "claude-code"
        """,
    )
    cfg3 = RunConfig.from_toml(toml_explicit)
    assert cfg3.general.default_agent == "antigravity-sdk"
    assert cfg3.runtime.agent == "claude-code"


def test_study_settings_auto_queries_fingerprint_and_resolution(tmp_path: Path) -> None:
    """Verify StudySettings.auto_queries layers via RunConfig.resolve and keeps fingerprints."""
    cfg_true = RunConfig(study=StudySettings(auto_queries=True))
    cfg_false = RunConfig(study=StudySettings(auto_queries=False))

    assert cfg_true.study.auto_queries is True
    assert cfg_false.study.auto_queries is False
    assert cfg_true.fingerprint == cfg_false.fingerprint
    assert cfg_true.arm == cfg_false.arm
    assert cfg_true.condition == cfg_false.condition

    resolved_inherit = RunConfig.resolve(StudySettings, cfg_false, auto_queries=None)
    assert resolved_inherit.auto_queries is False

    resolved_override = RunConfig.resolve(StudySettings, cfg_false, auto_queries=True)
    assert resolved_override.auto_queries is True


@pytest.mark.parametrize(
    (
        "query_settings",
        "targets",
        "expected_count",
        "expected_adversarial",
        "expected_adversarial_count",
        "expected_top_rivals",
    ),
    [
        (
            QuerySettings(count=4, adversarial_count=0, top_rivals=0),
            ("skill-a", "skill-b"),
            4,
            False,
            1,
            None,
        ),
        (
            QuerySettings(count=2, adversarial_count=3, top_rivals=5),
            ("skill-c",),
            2,
            True,
            3,
            5,
        ),
    ],
    ids=["zero_constraints_fallback", "active_constraints_preserved"],
)
def test_generate_flags_from_query_settings_handles_zero_constraints(
    query_settings: QuerySettings,
    targets: tuple[str, ...],
    expected_count: int,
    expected_adversarial: bool,
    expected_adversarial_count: int,
    expected_top_rivals: int | None,
) -> None:
    """Verify GenerateFlags.from_query_settings safely maps zero values in QuerySettings."""
    from reach.cli.flags import GenerateFlags

    flags = GenerateFlags.from_query_settings(query_settings, targets=targets)
    assert flags.targets == targets
    assert flags.count == expected_count
    assert flags.adversarial is expected_adversarial
    assert flags.adversarial_count == expected_adversarial_count
    assert flags.top_rivals == expected_top_rivals
