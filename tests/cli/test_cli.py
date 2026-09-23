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

"""Verify CLI argument parsing, configuration building, and command routing."""

from __future__ import annotations

import json
from collections.abc import Callable
from enum import Enum
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, Never, get_args
from unittest.mock import patch

import pytest
from cyclopts.exceptions import CycloptsError

from reach import run as reach_run
from reach.cli import (
    EVAL_REQUIRED,
    AgentName,
    Format,
    Vary,
    _reorder_argv,
    app,
    build_config,
    drafting,
    main,
    parse_agent_options,
)
from reach.cli import _verbs as registered_verbs
from reach.diff import VaryFactor
from reach.models import Catalog, Query
from reach.queries import (
    Origin,
    QuerySet,
    QuerySetProvenance,
    load_query_set,
    query_set_digest,
    save_query_set,
)
from reach.report import RENDERERS
from reach.runtime import known_agents
from reach.runtime.fake import FakeGenerator
from reach.views import OVERLAP_RENDERERS, REWRITE_RENDERERS, badge
from reach.views.diff import DIFF_RENDERERS

if TYPE_CHECKING:
    from reach.config import RunConfig

#: Configuration sections folded into RunConfig.
FOLDED = ("config", "catalog", "runtime", "plan", "study", "record")


def bind(*argv: str):
    """Parse CLI arguments without executing command handler."""
    _, bound, _ = app.parse_args(argv, exit_on_error=False)
    return bound.arguments


def configure(*argv: str) -> RunConfig:
    """Build RunConfig from CLI arguments without executing run."""
    passed = bind(*argv)
    return build_config(
        **{k: v for k, v in passed.items() if k in FOLDED},
        required=EVAL_REQUIRED,
    )


@pytest.fixture
def base_argv(skill_repo: Path, query_file: Path, tmp_path: Path) -> list[str]:
    """Provide standard eval command-line arguments for test corpus."""
    return [
        "eval",
        "--skills",
        str(skill_repo),
        "--queries",
        str(query_file),
        "--workdir",
        str(tmp_path / "work"),
        "--agent",
        "fake",
        "--mode",
        "neighborhood",
        "--catalog-size",
        "3",
        "--rivals",
        "2",
        "--attempts",
        "1",
        "--partial",
    ]


def test_flags_become_a_configuration(base_argv: list[str]) -> None:
    """Verify CLI flags are mapped into RunConfig attributes."""
    config = configure(*base_argv)
    assert config.runtime.agent == "fake"
    assert config.catalog.size == 3
    assert config.plan.attempts == 1
    assert config.study.partial is True


def test_missing_required_flags_are_named(tmp_path: Path) -> None:
    """Verify missing required flags raise ValueError naming the missing flag."""
    with pytest.raises(ValueError, match="--queries"):
        configure("eval")


def test_a_config_file_supplies_everything(
    write_reach_toml: Callable[..., Path],
    tmp_path: Path,
    skill_repo: Path,
    query_file: Path,
) -> None:
    """Verify --config TOML file populates RunConfig options and study paths."""
    path = write_reach_toml(
        f"""
        [study]
        skills = "{skill_repo}"
        queries = "{query_file}"
        workdir = "{tmp_path / "work"}"

        [runtime]
        agent = "fake"

        [runtime.options]
        model = "opus"
        """,
    )
    config = configure("eval", "--config", str(path))
    assert config.runtime.options["model"] == "opus"
    assert config.study.skills == skill_repo


def test_a_flag_overrides_the_file(
    write_reach_toml: Callable[..., Path],
    tmp_path: Path,
    skill_repo: Path,
    query_file: Path,
) -> None:
    """Verify explicit CLI flags override values from --config TOML."""
    path = write_reach_toml(
        f"""
        [study]
        skills = "{skill_repo}"
        queries = "{query_file}"
        workdir = "{tmp_path / "work"}"

        [plan]
        attempts = 9
        """,
    )
    assert configure("eval", "--config", str(path)).plan.attempts == 9
    assert configure("eval", "--config", str(path), "--attempts", "2").plan.attempts == 2


def test_an_unpassed_flag_does_not_clobber_the_file(
    write_reach_toml: Callable[..., Path],
    tmp_path: Path,
    skill_repo: Path,
    query_file: Path,
) -> None:
    """Verify unpassed CLI flags do not overwrite values defined in TOML config."""
    path = write_reach_toml(
        f"""
        [study]
        skills = "{skill_repo}"
        queries = "{query_file}"
        workdir = "{tmp_path / "work"}"

        [runtime]
        timeout_s = 500

        [runtime.options]
        executable = "/opt/claude/bin/claude"
        """,
    )
    config = configure("eval", "--config", str(path))
    assert config.runtime.timeout_s == 500
    assert config.runtime.options["executable"] == "/opt/claude/bin/claude"


def test_a_config_file_keeps_the_settings_no_flag_can_reach(
    write_reach_toml: Callable[..., Path],
    tmp_path: Path,
    skill_repo: Path,
    query_file: Path,
) -> None:
    """Verify TOML configuration resolves relative paths against config directory."""
    path = write_reach_toml(
        f"""
        [study]
        skills = "{skill_repo}"
        queries = "{query_file}"
        workdir = "work"

        [runtime]
        agent = "claude-code"

        [runtime.options]
        executable = "/opt/claude/bin/claude"
        """,
        directory=tmp_path / "study",
    )
    config = configure("eval", "--config", str(path))
    assert config.study.workdir == tmp_path / "study" / "work"
    assert config.runtime.options["executable"] == "/opt/claude/bin/claude"


def test_only_known_agents_are_accepted(base_argv: list[str], capsys) -> None:
    """Verify unrecognized agent string exits with code 2 and error message."""
    assert main([*base_argv, "--agent", "codex"]) == 2
    assert "codex" in capsys.readouterr().err


def test_agent_help_text_excludes_fake() -> None:
    """Verify dynamic agent help text excludes internal fake runtime test double."""
    from reach.cli.flags import agent_help_text
    from reach.runtime.fake import FAKE_AGENT

    text = agent_help_text()
    assert FAKE_AGENT not in text
    assert "claude-code" in text
    assert "antigravity-cli" in text
    assert "goose" in text
    assert "pi" in text


def test_a_setting_the_config_model_refuses_is_refused_while_parsing(
    base_argv: list[str],
    capsys,
) -> None:
    """Verify invalid catalog mode string exits with code 2 during parsing."""
    assert main([*base_argv, "--mode", "handwritten"]) == 2
    assert capsys.readouterr().out == ""


def test_a_refused_value_is_named_by_the_flag_that_carried_it(
    base_argv: list[str],
    capsys,
) -> None:
    """Verify validation error names the command-line flag rather than internal field name."""
    assert main([*base_argv, "--catalog-size", "roomy"]) == 2
    reported = capsys.readouterr().err.replace("\n", " ")
    assert "--catalog-size" in reported
    assert "roomy" in reported


LEAK_INDICATORS = (
    "nullable[",
    "CatalogFlags",
    "type=enum",
    "pydantic.dev",
    "input_value",
)


@pytest.mark.parametrize("leak", LEAK_INDICATORS)
def test_a_refused_value_suppresses_internal_schema_dumps(
    base_argv: list[str],
    capsys,
    leak: str,
) -> None:
    """Verify schema error formatting suppresses raw Pydantic validation dumps."""
    assert main([*base_argv, "--mode", "handwritten"]) == 2
    reported = capsys.readouterr().err.replace("\n", " ")
    assert "--mode" in reported
    assert "neighborhood" in reported
    assert leak not in reported


def test_explain_handles_empty_loc_model_level_validation_error() -> None:
    """Verify _explain formats model-level validation errors with empty loc without IndexError."""
    from cyclopts.exceptions import ValidationError as CycloptsValidationError
    from pydantic import BaseModel, model_validator
    from pydantic import ValidationError as PydanticValidationError

    from reach.cli.app import _explain

    class ModelWithRootValidator(BaseModel):
        val: int

        @model_validator(mode="after")
        def check_root(self) -> ModelWithRootValidator:
            msg = "Whole model is invalid"
            raise ValueError(msg)

    pydantic_error = None
    try:
        ModelWithRootValidator(val=1)
    except PydanticValidationError as exc:
        pydantic_error = exc

    assert pydantic_error is not None
    cyclopts_err = CycloptsValidationError(value="mock", msg="Validation failed")
    cyclopts_err.__cause__ = pydantic_error

    explanations = _explain(cyclopts_err)
    assert len(explanations) == 1
    assert "Invalid value" in explanations[0]
    assert "--input" in explanations[0]
    assert "Whole model is invalid" in explanations[0]


def test_opt_lands_in_runtime_options_for_the_fake_agent(base_argv: list[str]) -> None:
    """Verify -O key=value pairs populate runtime.options."""
    config = configure(*base_argv, "--agent", "fake", "-O", "model=custom-model")
    assert config.runtime.options == {"model": "custom-model"}


def test_repeated_opt_flags_accumulate_into_one_dict(base_argv: list[str]) -> None:
    """Verify multiple -O flags accumulate into options dictionary."""
    config = configure(
        *base_argv,
        "--agent",
        "claude-code",
        "-O",
        "model=opus",
        "-O",
        "max_turns=3",
    )
    assert config.runtime.options == {"model": "opus", "max_turns": 3}


@pytest.mark.parametrize(
    ("flags", "expected_key", "expected_val"),
    [
        (("--max-turns", "5"), "max_turns", 5),
        (("-T", "4"), "max_turns", 4),
        (("--early-exit",), "early_exit", True),
        (("--no-early-exit",), "early_exit", False),
        (("--model", "claude-opus-5"), "model", "claude-opus-5"),
        (("-m", "claude-haiku-4-5"), "model", "claude-haiku-4-5"),
        (("--effort", "high"), "effort", "high"),
        (("-e", "low"), "effort", "low"),
    ],
)
def test_cli_option_flag_aliases(
    base_argv: list[str],
    flags: tuple[str, ...],
    expected_key: str,
    expected_val: object,
) -> None:
    """Verify CLI option flags and short aliases populate corresponding runtime options."""
    config = configure(*base_argv, "--agent", "claude-code", *flags)
    assert config.runtime.options[expected_key] == expected_val


def test_model_and_effort_flags_combined(base_argv: list[str]) -> None:
    """Verify combining --model and --effort sets both runtime options."""
    config = configure(
        *base_argv,
        "--agent",
        "claude-code",
        "-m",
        "claude-sonnet-5",
        "-e",
        "medium",
    )
    assert config.runtime.options == {"model": "claude-sonnet-5", "effort": "medium"}


def test_opt_is_agent_agnostic_across_two_different_options_models(
    base_argv: list[str],
) -> None:
    """Verify -O parses options according to agent-specific schemas."""
    fake = configure(*base_argv, "--agent", "fake", "-O", "model=scripted")
    claude = configure(
        *base_argv,
        "--agent",
        "claude-code",
        "-O",
        "setting_sources=user",
    )
    assert fake.runtime.options == {"model": "scripted"}
    assert claude.runtime.options == {"setting_sources": "user"}


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("true", True),
        ("false", False),
        ("3", 3),
        ("1.5", 1.5),
        ("/opt/claude/bin/claude", "/opt/claude/bin/claude"),
        ("opus", "opus"),
    ],
)
def test_opt_values_are_coerced_through_json_with_a_string_fallback(
    raw: str,
    expected: object,
) -> None:
    """Verify parse_agent_options coerces scalar types and preserves strings."""
    assert parse_agent_options((f"key={raw}",)) == {"key": expected}


def test_opt_without_an_equals_sign_is_refused(base_argv: list[str], capsys) -> None:
    """Verify malformed -O flag missing '=' character exits with code 2."""
    assert main([*base_argv, "--agent", "fake", "-O", "justaword"]) == 2
    assert "justaword" in capsys.readouterr().err


def test_an_invalid_opt_key_names_the_flag_and_the_agents_model(
    base_argv: list[str],
    capsys,
) -> None:
    """Verify invalid -O key raises error referencing agent options model."""
    assert main([*base_argv, "--agent", "fake", "-O", "not_a_real_option=1"]) == 2

    complaint = capsys.readouterr().err
    assert "-O" in complaint
    assert "not_a_real_option" in complaint
    assert "fake" in complaint
    assert "FakeOptions" in complaint
    assert "pydantic.dev" not in complaint


def test_an_invalid_opt_value_type_is_refused_with_the_same_clarity(
    base_argv: list[str],
    capsys,
) -> None:
    """Verify invalid -O value type raises error referencing agent options model."""
    assert (
        main(
            [
                *base_argv,
                "--agent",
                "claude-code",
                "-O",
                "max_turns=not-a-number",
            ],
        )
        == 2
    )

    complaint = capsys.readouterr().err
    assert "-O" in complaint
    assert "max_turns" in complaint
    assert "ClaudeCodeOptions" in complaint


def test_main_catches_oserror_and_renders_error_panel(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify unhandled OSError is caught and rendered in error_panel with exit code 2."""

    def _bomb(*_a: object, **_kw: object) -> Never:
        msg = "Argument list too long"
        raise OSError(7, msg, "agy")

    monkeypatch.setattr("reach.cli.app", _bomb)
    exit_code = main(["check", "."])
    assert exit_code == 2
    err = capsys.readouterr().err
    assert "Argument list too long" in err


def test_main_catches_pydantic_validation_error_and_renders_error_panel(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify unhandled PydanticValidationError is caught and rendered in error_panel."""
    from pydantic import BaseModel, Field

    class _SampleModel(BaseModel):
        val: int = Field(...)

    def _bomb(*_a: object, **_kw: object) -> Never:
        _SampleModel.model_validate({"val": "not-an-int"})
        msg = "unreachable"
        raise AssertionError(msg)

    monkeypatch.setattr("reach.cli.app", _bomb)
    exit_code = main(["check", "."])
    assert exit_code == 2
    err = capsys.readouterr().err
    assert "validation error" in err.lower()


def test_main_catches_broken_pipe_error_and_exits_zero(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify BrokenPipeError exits cleanly with 0 and without printing an error panel."""

    def _broken_pipe(*_a: object, **_kw: object) -> Never:
        raise BrokenPipeError(32, "Broken pipe")

    monkeypatch.setattr("reach.cli.app", _broken_pipe)
    exit_code = main(["check", "."])
    assert exit_code == 0
    err = capsys.readouterr().err
    assert err == ""


def test_opt_replaces_rather_than_merges_into_a_config_files_options_table(
    write_reach_toml: Callable[..., Path],
    skill_repo: Path,
    query_file: Path,
    tmp_path: Path,
) -> None:
    """Verify -O replaces options dictionary from config file completely."""
    config_path = write_reach_toml(
        f"""
        [study]
        skills = "{skill_repo}"
        queries = "{query_file}"
        workdir = "{tmp_path / "work"}"

        [runtime]
        agent = "fake"

        [runtime.options]
        model = "from-toml"
        """,
    )
    config = configure("eval", "--config", str(config_path), "-O", "model=from-opt")
    assert config.runtime.options == {"model": "from-opt"}


@pytest.mark.parametrize(
    ("declared", "registry"),
    [
        (Format, tuple(sorted(RENDERERS))),
        (Format, tuple(sorted(DIFF_RENDERERS))),
        (Vary, tuple(factor.value for factor in VaryFactor)),
        (AgentName, known_agents()),
    ],
    ids=["renderers", "comparison-renderers", "vary", "agents"],
)
def test_a_declared_choice_matches_the_registry_behind_it(declared, registry) -> None:
    """Verify declared CLI enum types match backend registry keys."""
    target = getattr(declared, "__value__", declared)
    if isinstance(target, type) and issubclass(target, Enum):
        args = tuple(member.value for member in target)
    else:
        args = get_args(target)
    assert tuple(sorted(args)) == tuple(sorted(registry))


@pytest.mark.parametrize(
    "registry",
    [OVERLAP_RENDERERS, REWRITE_RENDERERS],
    ids=["listing", "suggestion"],
)
def test_the_overlap_renderers_are_spellings_the_parser_already_accepts(
    registry,
) -> None:
    """Verify overlap and rewrite renderers are a subset of declared Format options."""
    args = get_args(getattr(Format, "__value__", Format))
    assert set(registry) < set(args)
    assert "text" not in registry


def test_dry_run_issues_no_probes(base_argv: list[str], capsys) -> None:
    """Verify --dry-run prints probe plan without executing probes."""
    assert main([*base_argv, "--dry-run"]) == 0
    assert "2 probes" in capsys.readouterr().err


def test_dry_run_states_the_configuration_fingerprint(base_argv: list[str], capsys) -> None:
    """Verify --dry-run outputs configuration fingerprint."""
    assert main([*base_argv, "--dry-run"]) == 0
    assert "[config " in capsys.readouterr().err


def test_verbose_puts_the_hex_back_on_the_line_the_badge_replaced(
    base_argv: list[str],
    capsys,
) -> None:
    """Verify --verbose flag prints both human badge and raw hex fingerprint."""
    assert main([*base_argv, "--dry-run", "--verbose"]) == 0
    stamp = capsys.readouterr().err.partition("[config ")[2]
    said, hexed = stamp.split()[:2]
    assert badge(hexed) == said
    assert said != hexed


def test_dry_run_says_what_the_depth_can_resolve_before_anything_is_spent(
    base_argv: list[str],
    capsys,
) -> None:
    """Verify --dry-run prints statistical resolution and probe requirement estimate."""
    assert main([*base_argv, "--dry-run"]) == 0
    err = capsys.readouterr().err
    assert "resolves a per-query rate to +-1.00 at 1 attempts" in err
    assert "+-0.20 would take 99 per query" in err


@pytest.fixture
def cramped_argv(
    write_reach_toml: Callable[..., Path],
    tmp_path: Path,
    skill_repo: Path,
    query_file: Path,
) -> list[str]:
    """Generate CLI arguments configuring an overly constrained skill listing budget."""
    path = write_reach_toml(
        f"""
        [study]
        skills = "{skill_repo}"
        queries = "{query_file}"
        workdir = "{tmp_path / "work"}"
        partial = true
        trusted = true

        [runtime]
        agent = "claude-code"

        [runtime.options]
        skill_listing_budget_fraction = 0.00001

        [catalog]
        mode = "neighborhood"
        size = 3
        rivals = 2

        [plan]
        attempts = 1
        """,
    )
    return ["eval", "--config", str(path)]


def test_a_catalog_too_wide_for_the_listing_is_refused_by_name(
    cramped_argv: list[str],
    capsys,
    wide,
) -> None:
    """Verify evaluation fails when catalog exceeds listing budget and suggests fix."""
    assert main(cramped_argv) == 2
    shown = capsys.readouterr().err
    assert "bare name" in shown
    assert "skill_listing_budget_fraction" in shown


def test_a_dry_run_refuses_what_the_real_run_would_refuse(
    cramped_argv: list[str],
    capsys,
    wide,
) -> None:
    """Verify dry run validation triggers the same listing budget refusal as full evaluation."""
    assert main([*cramped_argv, "--dry-run"]) == 2
    assert "bare name" in capsys.readouterr().err


def test_allowing_truncation_lets_the_run_through(cramped_argv: list[str], capsys) -> None:
    """Verify --allow-truncation permits evaluation when catalog exceeds listing budget."""
    assert main([*cramped_argv, "--dry-run", "--allow-truncation"]) == 0
    assert "2 probes" in capsys.readouterr().err


def test_a_run_shows_what_it_measured(base_argv: list[str], capsys) -> None:
    """Verify eval command outputs consistency metrics and neighborhood catalog name to stderr."""
    assert main(base_argv) == 0
    shown = capsys.readouterr().err
    assert "consistency" in shown
    assert "neighborhood:gcs-lifecycle-rules" in shown


def test_the_plan_a_run_prints_is_the_run_it_then_conducts(
    base_argv: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify catalog resolution occurs exactly once between planning and probe execution."""
    real, builds = reach_run.build_catalogs, []

    def counted(*args, **kwargs) -> list[Catalog]:
        builds.append(args)
        return real(*args, **kwargs)

    monkeypatch.setattr(reach_run, "build_catalogs", counted)
    assert main(base_argv) == 0
    assert len(builds) == 1, "the catalog was resolved more than once"


def test_a_redirected_run_reports_one_line_per_probe(base_argv: list[str], capsys) -> None:
    """Verify redirected output writes progress messages per probe without ANSI rewrite codes."""
    assert main(base_argv) == 0
    err = capsys.readouterr().err.splitlines()
    assert err[0].startswith("neighborhood:gcs-lifecycle-rules: 3 skills, 2 probes on")
    assert err[1:4] == [
        ("  resolves a per-query rate to +-1.00 at 1 attempts; +-0.20 would take 99 per query"),
        "[1/2] q-lifecycle -> (no selection)",
        "[2/2] q-retention -> (no selection)",
    ]


def test_quiet_mutes_the_view_and_leaves_the_report(base_argv: list[str], capsys) -> None:
    """Verify --quiet suppresses stderr progress while preserving stdout report."""
    assert main([*base_argv, "--quiet", "--format", "json"]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert json.loads(captured.out)["catalog_id"] == "neighborhood:gcs-lifecycle-rules"


def test_quiet_still_lets_a_setup_failure_be_heard(base_argv: list[str], capsys) -> None:
    """Verify errors remain audible on stderr when --quiet is set."""
    assert main([*base_argv, "--quiet", "--catalog", "neighborhood:gke-basics"]) == 2
    assert "rescope" in capsys.readouterr().err


def test_json_output_is_machine_readable(base_argv: list[str], capsys) -> None:
    """Verify --format json produces valid JSON containing provenance."""
    assert main([*base_argv, "--format", "json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["provenance"]["runtime"] == "fake"


def test_workers_runs_probes_without_changing_the_recorded_configuration(
    base_argv: list[str],
    tmp_path: Path,
) -> None:
    """Verify --workers alters worker count in plan settings without modifying fingerprints."""
    sequential_out = tmp_path / "sequential.jsonl"
    concurrent_out = tmp_path / "concurrent.jsonl"
    assert main([*base_argv, "--out", str(sequential_out)]) == 0
    assert main([*base_argv, "--out", str(concurrent_out), "--workers", "3"]) == 0
    sequential_sidecar = json.loads(Path(f"{sequential_out}.config.json").read_text())
    concurrent_sidecar = json.loads(Path(f"{concurrent_out}.config.json").read_text())
    assert concurrent_sidecar["config"]["plan"]["workers"] == 3
    assert sequential_sidecar["fingerprint"] == concurrent_sidecar["fingerprint"]
    assert sequential_sidecar["arm"] == concurrent_sidecar["arm"]
    assert sequential_sidecar["condition"] == concurrent_sidecar["condition"]


def test_the_short_workers_flag_is_accepted(base_argv: list[str]) -> None:
    """Verify -j flag is parsed as workers alias."""
    assert main([*base_argv, "-j", "2"]) == 0


def test_results_and_their_configuration_land_together(
    base_argv: list[str],
    tmp_path: Path,
) -> None:
    """Verify --out creates both JSONL results and companion .config.json sidecar file."""
    out = tmp_path / "results.jsonl"
    assert main([*base_argv, "--out", str(out)]) == 0
    assert out.exists()
    assert Path(f"{out}.config.json").exists()


def test_a_setup_failure_exits_two_with_a_message(
    skill_repo: Path,
    query_file: Path,
    tmp_path: Path,
    capsys,
) -> None:
    """Verify referencing nonexistent catalog exits with code 2 and descriptive error."""
    code = main(
        [
            "eval",
            "--skills",
            str(skill_repo),
            "--queries",
            str(query_file),
            "--workdir",
            str(tmp_path / "work"),
            "--agent",
            "fake",
            "--mode",
            "neighborhood",
            "--catalog",
            "neighborhood:ghost",
            "--rescope",
        ],
    )
    assert code == 2
    reported = capsys.readouterr().err
    assert "no catalog named" in reported
    assert not reported.lstrip().startswith('"')
    assert '"no catalog' not in reported


def test_a_refusal_looks_the_same_whenever_it_arrives(base_argv: list[str], capsys) -> None:
    """Verify setup errors format with consistent panel styling."""
    assert main([*base_argv, "--catalog", "neighborhood:ghost", "--rescope"]) == 2
    framed = capsys.readouterr().err
    assert "Error" in framed
    assert "╭" in framed


def test_reprobing_elsewhere_needs_saying_so(base_argv: list[str], capsys) -> None:
    """Verify mismatched catalog for labeled query set requires explicit --rescope."""
    assert main([*base_argv, "--catalog", "neighborhood:gke-basics"]) == 2
    assert "rescope" in capsys.readouterr().err


def test_a_missing_corpus_exits_two(query_file: Path, tmp_path: Path, capsys) -> None:
    """Verify missing skill directory path exits with code 2 and error message."""
    code = main(
        [
            "eval",
            "--skills",
            str(tmp_path / "absent"),
            "--queries",
            str(query_file),
            "--workdir",
            str(tmp_path / "work"),
            "--agent",
            "fake",
        ],
    )
    assert code == 2
    assert "skill root does not exist" in capsys.readouterr().err


def test_a_runtime_that_refused_the_work_is_reported_not_raised(
    skill_repo: Path,
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    """Verify runtime exception during drafting is reported cleanly without traceback."""

    def refuse(*_args, **_kwargs) -> Never:
        msg = "generation failed: Prompt is too long"
        raise RuntimeError(msg)

    monkeypatch.setattr(drafting, "generate_query_set", refuse)
    code = main(
        [
            "eval",
            "--skills",
            str(skill_repo),
            "--queries",
            str(tmp_path / "drafted.json"),
            "--workdir",
            str(tmp_path / "work"),
            "--catalog",
            "neighborhood:gke-basics",
            "--mode",
            "neighborhood",
            "--catalog-size",
            "3",
            "--rivals",
            "2",
            "--agent",
            "fake",
            "--partial",
        ],
    )
    assert code == 2
    assert "Prompt is too long" in capsys.readouterr().err


@pytest.fixture
def two_arms(skill_repo: Path, query_file: Path, tmp_path: Path, capsys) -> list[Path]:
    """Record two evaluation runs under different catalog sizes to provide comparison arms."""
    paths = []
    for label, size, rivals in (("wide", "3", "2"), ("narrow", "2", "1")):
        out = tmp_path / "arms" / f"{label}.jsonl"
        assert (
            main(
                [
                    "eval",
                    "--skills",
                    str(skill_repo),
                    "--queries",
                    str(query_file),
                    "--workdir",
                    str(tmp_path / "work"),
                    "--agent",
                    "fake",
                    "--mode",
                    "neighborhood",
                    "--catalog-size",
                    size,
                    "--rivals",
                    rivals,
                    "--attempts",
                    "1",
                    "--partial",
                    "--out",
                    str(out),
                ],
            )
            == 0
        )
        paths.append(out)
    capsys.readouterr()
    return paths


def test_a_comparison_prices_the_delta_against_the_floor(two_arms, capsys) -> None:
    """Verify diff command prints delta verdict against noise floor."""
    control, treatment = two_arms
    assert main(["diff", str(control), str(treatment), "--vary", "scope"]) == 0
    out = capsys.readouterr().out
    assert "noise floor" in out
    assert "Verdict" in out


def test_diff_accepts_artifact_json_files(two_arms, tmp_path: Path, capsys) -> None:
    """Verify diff command accepts .artifact.json files directly without sidecars."""
    from reach.artifact import write_artifact
    from reach.diff import load_arm

    control, treatment = two_arms
    control_art = write_artifact(
        load_arm(control).artifact,
        tmp_path / "control.artifact.json",
    )
    treatment_art = write_artifact(
        load_arm(treatment).artifact,
        tmp_path / "treatment.artifact.json",
    )

    assert (
        main(
            [
                "diff",
                str(control_art),
                str(treatment_art),
                "--vary",
                "scope",
            ],
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "noise floor" in out
    assert "Verdict" in out


def test_the_noise_inflation_flag_overrides_the_calibrated_default(two_arms, capsys) -> None:
    """Verify --noise-inflation flag sets over-dispersion multiplier in diff output."""
    control, treatment = two_arms
    assert (
        main(
            [
                "diff",
                str(control),
                str(treatment),
                "--vary",
                "scope",
                "--noise-inflation",
                "1.0",
            ],
        )
        == 0
    )
    assert "1.00x over-dispersion" in capsys.readouterr().out


def test_a_refused_delta_still_exits_zero(two_arms, capsys) -> None:
    """Verify non-improving diff outcome exits with code 0."""
    control, treatment = two_arms
    assert main(["diff", str(control), str(treatment), "--vary", "scope"]) == 0
    assert "not an improvement" in capsys.readouterr().out


def test_diff_cli_respects_config_file(
    write_reach_toml: Callable[..., Path],
    two_arms,
    tmp_path: Path,
    capsys,
) -> None:
    """Verify diff command reads [diff] settings from reach.toml configuration."""
    control, treatment = two_arms
    cfg_file = write_reach_toml(
        """
        [diff]
        noise_inflation = 1.75
        confidence = 0.90
        """,
    )
    assert (
        main(
            [
                "diff",
                str(control),
                str(treatment),
                "--vary",
                "scope",
                "--config",
                str(cfg_file),
            ],
        )
        == 0
    )
    assert "1.75x over-dispersion" in capsys.readouterr().out


def test_a_comparison_without_a_factor_is_refused(two_arms, capsys) -> None:
    """Verify diff command without --vary flag exits with code 2."""
    control, treatment = two_arms
    assert main(["diff", str(control), str(treatment)]) == 2
    assert "--vary" in capsys.readouterr().err


def test_a_comparison_of_a_file_with_itself_exits_two(two_arms, capsys) -> None:
    """Verify diffing a file against itself exits with code 2 and descriptive error."""
    control, _ = two_arms
    assert main(["diff", str(control), str(control), "--vary", "scope"]) == 2
    assert "with itself" in capsys.readouterr().err


@pytest.fixture
def two_broken_arms(two_arms) -> list[Path]:
    """Remove sidecar config files from comparison arms."""
    for path in two_arms:
        path.with_suffix(".jsonl.config.json").unlink()
    return two_arms


def test_a_refused_comparison_names_every_wall_by_default(
    two_broken_arms,
    wide: None,
    capsys,
) -> None:
    """Verify diff command reports all comparability failures when runs cannot be compared."""
    control, treatment = two_broken_arms
    assert main(["diff", str(control), str(treatment), "--vary", "scope"]) == 2

    complaint = capsys.readouterr().err
    assert "3 walls" in complaint
    assert "no sidecar" in complaint
    assert control.name in complaint
    assert treatment.name in complaint
    assert "not reached" in complaint


def test_a_pairing_that_compares_prints_the_comparison(two_arms, capsys) -> None:
    """Verify diff prints comparison table when arms are comparable."""
    control, treatment = two_arms
    assert main(["diff", str(control), str(treatment), "--vary", "scope"]) == 0
    assert "Verdict" in capsys.readouterr().out


def test_control_and_treatment_labels_replace_the_filenames(two_arms, capsys) -> None:
    """Verify --control-label and --treatment-label replace file paths in diff table header."""
    control, treatment = two_arms
    assert (
        main(
            [
                "diff",
                str(control),
                str(treatment),
                "--vary",
                "scope",
                "--control-label",
                "v1",
                "--treatment-label",
                "v2",
            ],
        )
        == 0
    )

    shown = capsys.readouterr().out
    assert "v1" in shown
    assert "v2" in shown
    assert control.stem not in shown
    assert treatment.stem not in shown


def test_labels_are_not_offered_without_asking_for_them(two_arms, capsys) -> None:
    """Verify default diff output uses filenames when labels are not specified."""
    control, treatment = two_arms
    assert main(["diff", str(control), str(treatment), "--vary", "scope"]) == 0

    shown = capsys.readouterr().out
    assert control.stem in shown
    assert treatment.stem in shown


FOREIGN_ROWS = (
    "prompt,answer,neutral,rationale\n"
    "rotate our keys,kms-rotation,skill-finder|kms-router,reviewed manually\n"
)
FOREIGN_FLAGS = [
    "--text-column",
    "prompt",
    "--expected-skill-column",
    "answer",
    "--acceptable-skills-column",
    "neutral",
    "--notes-column",
    "rationale",
    "--separator",
    "|",
]


@pytest.fixture
def exported(tmp_path: Path, exchange_set: QuerySet) -> Path:
    """Write exchange query set to JSON file on disk."""
    return save_query_set(exchange_set, tmp_path / "set.json")


@pytest.fixture
def foreign_file(tmp_path: Path) -> Path:
    """Write synthetic foreign CSV dataset with custom headers."""
    path = tmp_path / "theirs.csv"
    path.write_text(FOREIGN_ROWS, encoding="utf-8")
    return path


def test_a_set_exports_to_stdout_when_no_file_is_named(exported: Path, capsys) -> None:
    """Verify query export writes CSV to stdout when --out is omitted."""
    assert main(["query", str(exported), "--format", "csv"]) == 0
    header = capsys.readouterr().out.splitlines()[0]
    assert header == "id,text,kind,expected_skill,acceptable_skills,notes"


def test_a_custom_separator_is_used_for_stdout_exports(exported: Path, capsys) -> None:
    """Verify --separator controls acceptable skill joining on stdout export."""
    assert main(["query", str(exported), "--format", "csv", "--separator", "|"]) == 0
    assert "finding-google-skills|gcs-router" in capsys.readouterr().out


def test_a_custom_separator_is_used_for_file_exports(exported: Path, tmp_path: Path) -> None:
    """Verify --separator controls acceptable skill joining on file export."""
    out = tmp_path / "rows.csv"
    assert main(["query", str(exported), "--out", str(out), "--separator", "|"]) == 0
    assert "finding-google-skills|gcs-router" in out.read_text(encoding="utf-8")


def test_the_row_format_is_read_off_the_name_it_writes(exported: Path, tmp_path: Path) -> None:
    """Verify query export infers output format from --out file extension."""
    out = tmp_path / "rows.jsonl"
    assert main(["query", str(exported), "--out", str(out)]) == 0
    first = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
    assert first["id"] == "x-lifecycle"


def test_a_name_that_says_nothing_about_its_format_asks_for_one(
    exported: Path,
    tmp_path: Path,
    capsys,
) -> None:
    """Verify query export requires explicit --format when --out extension is unrecognized."""
    out = tmp_path / "rows.txt"
    assert main(["query", str(exported), "--out", str(out)]) == 2
    assert "--format" in capsys.readouterr().err
    assert not out.exists()


def test_the_command_line_round_trip_does_not_fork_the_evidence(
    exported: Path,
    exchange_set: QuerySet,
    tmp_path: Path,
) -> None:
    """Verify query export followed by query import preserves ground truth digest."""
    rows = tmp_path / "rows.csv"
    back = tmp_path / "back.json"
    assert main(["query", str(exported), "--out", str(rows)]) == 0
    assert (
        main(
            [
                "query",
                str(rows),
                "--out",
                str(back),
                "--catalog",
                exchange_set.catalog_id,
            ],
        )
        == 0
    )
    assert query_set_digest(load_query_set(back)) == query_set_digest(exchange_set)


def test_an_export_names_the_catalog_its_labels_are_valid_in(
    exported: Path,
    exchange_set: QuerySet,
    tmp_path: Path,
    capsys,
) -> None:
    """Verify query export prints the valid catalog identifier to stderr."""
    out = tmp_path / "rows.csv"
    assert main(["query", str(exported), "--out", str(out)]) == 0
    assert exchange_set.catalog_id in capsys.readouterr().err


def test_a_spreadsheet_saved_export_comes_back_as_the_same_evidence(
    exported: Path,
    exchange_set: QuerySet,
    tmp_path: Path,
) -> None:
    """Verify exported and re-imported BOM-prefixed CSV produces identical ground truth digest."""
    rows = tmp_path / "rows.csv"
    back = tmp_path / "back.json"
    assert main(["query", str(exported), "--out", str(rows)]) == 0
    # Simulate spreadsheet export saved as UTF-8 with BOM prefix.
    rows.write_text(rows.read_text(encoding="utf-8"), encoding="utf-8-sig")
    assert (
        main(
            [
                "query",
                str(rows),
                "--out",
                str(back),
                "--catalog",
                exchange_set.catalog_id,
            ],
        )
        == 0
    )
    assert query_set_digest(load_query_set(back)) == query_set_digest(exchange_set)


def test_an_imported_set_says_where_it_came_from(foreign_file: Path, tmp_path: Path) -> None:
    """Verify imported query set records Origin.IMPORTED and source path in provenance."""
    destination = tmp_path / "set.json"
    assert (
        main(
            [
                "query",
                str(foreign_file),
                "--out",
                str(destination),
                "--catalog",
                "all",
                *FOREIGN_FLAGS,
            ],
        )
        == 0
    )
    provenance = load_query_set(destination).provenance
    assert provenance is not None
    assert provenance.origin is Origin.IMPORTED
    assert provenance.source == str(foreign_file)


def test_a_foreign_file_is_mapped_by_flags_rather_than_by_code(
    foreign_file: Path,
    tmp_path: Path,
) -> None:
    """Verify column mapping flags map foreign CSV columns to Query fields."""
    destination = tmp_path / "set.json"
    assert (
        main(
            [
                "query",
                str(foreign_file),
                "--out",
                str(destination),
                "--catalog",
                "all",
                *FOREIGN_FLAGS,
            ],
        )
        == 0
    )
    imported = load_query_set(destination).queries[0]
    assert imported.id == "q-1"
    assert imported.expected_skill == "kms-rotation"
    assert imported.acceptable_skills == ("skill-finder", "kms-router")
    assert imported.notes == "reviewed manually"


def test_a_column_the_file_does_not_have_is_named_in_the_refusal(
    foreign_file: Path,
    tmp_path: Path,
    capsys,
) -> None:
    """Verify missing mapped column name in import file exits with code 2."""
    destination = tmp_path / "set.json"
    assert (
        main(
            [
                "query",
                str(foreign_file),
                "--out",
                str(destination),
                "--catalog",
                "all",
                "--text-column",
                "question",
            ],
        )
        == 2
    )

    reported = capsys.readouterr().err
    assert "question" in reported
    assert "prompt" in reported
    assert not destination.exists()


@pytest.mark.parametrize(
    ("source_format", "target_format"),
    [
        ("json", "jsonl"),
        ("jsonl", "csv"),
        ("csv", "json"),
    ],
    ids=["json-to-jsonl", "jsonl-to-csv", "csv-to-json"],
)
def test_query_command_converts_between_formats_directly(
    source_format: str,
    target_format: str,
    exported: Path,
    tmp_path: Path,
) -> None:
    """Verify reach query directly converts query sets between JSON, JSONL, and CSV."""
    if source_format == "json":
        src = exported
    else:
        src = tmp_path / f"intermediate.{source_format}"
        assert main(["query", str(exported), "--out", str(src)]) == 0

    dest = tmp_path / f"converted.{target_format}"
    assert main(["query", str(src), "--out", str(dest)]) == 0
    assert dest.exists()

    loaded = load_query_set(dest)
    assert len(loaded.queries) == 4
    assert any(q.id == "x-lifecycle" for q in loaded.queries)


def test_query_view_renders_table_without_skills_catalog(
    exported: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify reach query <file.json> renders query table even when no skills exist in cwd."""
    empty_cwd = tmp_path / "empty_workspace"
    empty_cwd.mkdir()
    monkeypatch.chdir(empty_cwd)
    assert main(["query", str(exported)]) == 0
    err = capsys.readouterr().err
    assert "query" in err
    assert "expects" in err
    assert "text" in err


def test_query_import_refuses_to_write_over_an_existing_set(
    tmp_path: Path,
    query_file: Path,
    capsys,
) -> None:
    """Verify query import refuses to overwrite an existing destination file."""
    theirs = tmp_path / "theirs.csv"
    theirs.write_text("text\nfoo\n", encoding="utf-8")
    argv = [
        "query",
        str(theirs),
        "--out",
        str(query_file),
        "--catalog",
        "all",
    ]
    assert main(argv) == 2
    assert "already exists" in capsys.readouterr().err


def test_query_draft_refuses_to_write_over_an_existing_set(
    query_file: Path,
    skill_repo: Path,
    capsys,
) -> None:
    """Verify query draft refuses to overwrite an existing destination file."""
    argv = [
        "query",
        "draft",
        "--queries",
        str(query_file),
        "--skills",
        str(skill_repo),
        "--agent",
        "fake",
    ]
    assert main(argv) == 2
    assert "already exists" in capsys.readouterr().err


def test_a_dry_draft_says_what_it_would_buy_and_writes_nothing(
    skill_repo: Path,
    tmp_path: Path,
    capsys,
) -> None:
    """Verify query draft --dry-run prints target count and avoids writing output."""
    destination = tmp_path / "drafted.json"
    assert (
        main(
            [
                "query",
                "draft",
                "--skills",
                str(skill_repo),
                "--queries",
                str(destination),
                "--agent",
                "fake",
                "--dry-run",
            ],
        )
        == 0
    )
    assert not destination.exists()
    assert "drafting 9 queries for 3 targets" in capsys.readouterr().err


def test_query_draft_with_adversarial_flag_dry_run(
    skill_repo: Path,
    tmp_path: Path,
    capsys,
) -> None:
    """Verify query draft --adversarial reports adversarial queries in progress output."""
    destination = tmp_path / "drafted.json"
    assert (
        main(
            [
                "query",
                "draft",
                "--skills",
                str(skill_repo),
                "--queries",
                str(destination),
                "--agent",
                "fake",
                "--adversarial",
                "--adversarial-count",
                "2",
                "--dry-run",
            ],
        )
        == 0
    )
    assert not destination.exists()
    assert "drafting 9 (+6 adversarial) queries for 3 targets" in capsys.readouterr().err


def test_a_draft_that_names_a_mode_drafts_into_the_catalog_it_was_given(
    skill_repo: Path,
    tmp_path: Path,
    capsys,
) -> None:
    """Verify query draft honors explicit --mode and --catalog selections."""
    assert (
        main(
            [
                "query",
                "draft",
                "--skills",
                str(skill_repo),
                "--queries",
                str(tmp_path / "drafted.json"),
                "--agent",
                "fake",
                "--mode",
                "singleton",
                "--catalog",
                "singleton:gke-basics",
                "--dry-run",
            ],
        )
        == 0
    )
    shown = capsys.readouterr().err
    assert "singleton:gke-basics: 1 skills resident" in shown
    assert "for 1 targets" in shown


@pytest.mark.parametrize("source", ["flag", "config"], ids=["a-flag", "a-config"])
def test_a_draft_stands_in_for_a_workspace_only_when_nobody_supplied_one(
    source: str,
    write_reach_toml: Callable[..., Path],
    skill_repo: Path,
    tmp_path: Path,
    capsys,
) -> None:
    """Verify query draft supplies workspace defaults when omitted from flags or config."""
    destination = tmp_path / "drafted.json"
    argv = ["query", "draft", "--agent", "fake", "--dry-run"]
    if source == "flag":
        argv += [
            "--skills",
            str(skill_repo),
            "--queries",
            str(destination),
            "--workdir",
            str(tmp_path / "shared-work"),
        ]
    else:
        config = write_reach_toml(
            f"""
            [study]
            skills = "{skill_repo}"
            queries = "{destination}"
            workdir = "{tmp_path / "shared-work"}"
            """,
        )
        argv += ["--config", str(config)]

    assert main(argv) == 0
    assert "drafting 9 queries for 3 targets" in capsys.readouterr().err
    assert not destination.exists()


def test_a_run_dir_nests_the_query_set_and_the_workspace(
    skill_repo: Path,
    tmp_path: Path,
    capsys,
) -> None:
    """Verify --run-dir derives default locations for queries and workspace."""
    run_dir = tmp_path / "run"
    assert (
        main(
            [
                "query",
                "draft",
                "--skills",
                str(skill_repo),
                "--run-dir",
                str(run_dir),
                "--agent",
                "fake",
                "--dry-run",
            ],
        )
        == 0
    )
    assert "drafting 9 queries for 3 targets" in capsys.readouterr().err
    assert not run_dir.exists(), "a dry run wrote where a real draft would land"


def test_a_run_dir_still_yields_to_an_explicit_queries_flag(
    skill_repo: Path,
    tmp_path: Path,
    capsys,
) -> None:
    """Verify explicit --queries flag overrides --run-dir query location."""
    elsewhere = tmp_path / "elsewhere.json"
    assert (
        main(
            [
                "query",
                "draft",
                "--skills",
                str(skill_repo),
                "--run-dir",
                str(tmp_path / "run"),
                "--queries",
                str(elsewhere),
                "--agent",
                "fake",
                "--dry-run",
            ],
        )
        == 0
    )
    assert str(elsewhere) in capsys.readouterr().err
    assert not (tmp_path / "run").exists()


def _assert_draft_provenance(destination: Path) -> None:
    """Verify generated query set records generator provenance and configuration digests."""
    provenance = load_query_set(destination).provenance
    assert provenance is not None
    expected = {
        "origin": Origin.GENERATED,
        "generator_model": "opus",
        "generator_arm": "content",
        "queries_per_target": 3,
        "rivals_in_view": 2,
        "tool_version": metadata.version("skill-reach"),
    }
    actual = {
        "origin": provenance.origin,
        "generator_model": provenance.generator_model,
        "generator_arm": provenance.generator_arm,
        "queries_per_target": provenance.queries_per_target,
        "rivals_in_view": provenance.rivals_in_view,
        "tool_version": provenance.tool_version,
    }
    assert actual == expected
    assert bool(provenance.bodies_digest)
    assert bool(provenance.config_fingerprint)


def test_a_drafted_set_records_the_terms_it_was_drafted_under(
    skill_repo: Path,
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify generated query set records generator provenance and configuration digests."""
    drafted = QuerySet(
        catalog_id="all",
        queries=(Query(id="d-1", text="Tier old objects.", expected_skill="gke-basics"),),
        provenance=QuerySetProvenance(origin=Origin.AUTHORED),
    )
    monkeypatch.setattr(
        "reach.cli.drafting.text_generator",
        lambda **_: FakeGenerator(),
    )
    monkeypatch.setattr(
        "reach.cli.drafting.generate_query_set",
        lambda *_, **__: drafted,
    )

    destination = tmp_path / "drafted.json"
    assert (
        main(
            [
                "query",
                "draft",
                "--skills",
                str(skill_repo),
                "--queries",
                str(destination),
                "--agent",
                "fake",
                "--generator-model",
                "opus",
                "--count",
                "3",
            ],
        )
        == 0
    )
    _assert_draft_provenance(destination)


def test_draft_with_review_flag_invokes_review_curator(
    skill_repo: Path,
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify reach query draft with --review triggers launch_query_review before saving."""
    from reach.queries import load_query_set

    drafted = QuerySet(
        catalog_id="all",
        queries=(Query(id="d-1", text="Tier old objects.", expected_skill="gke-basics"),),
        provenance=QuerySetProvenance(origin=Origin.AUTHORED),
    )
    monkeypatch.setattr(
        "reach.cli.drafting.text_generator",
        lambda **_: FakeGenerator(),
    )
    monkeypatch.setattr(
        "reach.cli.drafting.generate_query_set",
        lambda *_, **__: drafted,
    )

    destination = tmp_path / "reviewed.json"
    with patch(
        "reach.review.launch_query_review",
        side_effect=lambda qs, *_: qs,
    ) as mock_review:
        assert (
            main(
                [
                    "query",
                    "draft",
                    "--skills",
                    str(skill_repo),
                    "--queries",
                    str(destination),
                    "--agent",
                    "fake",
                    "--count",
                    "1",
                    "--review",
                ],
            )
            == 0
        )
        assert mock_review.called
        assert destination.exists()
        loaded = load_query_set(destination)
        assert loaded.provenance is not None
        assert loaded.provenance.reviewed is True


def test_draft_concurrency_flag_reaches_generate_query_set(
    skill_repo: Path,
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify --draft-concurrency flag is passed to generate_query_set."""
    captured: dict[str, object] = {}

    def fake_generate(*args, **kwargs) -> QuerySet:
        captured.update(kwargs)
        return QuerySet(
            catalog_id="all",
            queries=(),
            provenance=QuerySetProvenance(origin=Origin.AUTHORED),
        )

    monkeypatch.setattr(
        "reach.cli.drafting.text_generator",
        lambda **_: FakeGenerator(),
    )
    monkeypatch.setattr("reach.cli.drafting.generate_query_set", fake_generate)

    destination = tmp_path / "drafted.json"
    assert (
        main(
            [
                "query",
                "draft",
                "--skills",
                str(skill_repo),
                "--queries",
                str(destination),
                "--agent",
                "fake",
                "--draft-concurrency",
                "3",
            ],
        )
        == 0
    )
    assert captured["concurrency"] == 3


def test_recording_how_a_set_was_made_does_not_move_its_digest(
    skill_repo: Path,
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify query set ground truth digest remains invariant under provenance attachment."""
    drafted = QuerySet(
        catalog_id="all",
        queries=(Query(id="d-1", text="Tier old objects.", expected_skill="gke-basics"),),
        provenance=QuerySetProvenance(origin=Origin.AUTHORED),
    )
    monkeypatch.setattr(
        "reach.cli.drafting.text_generator",
        lambda **_: FakeGenerator(),
    )
    monkeypatch.setattr(
        "reach.cli.drafting.generate_query_set",
        lambda *_, **__: drafted,
    )

    destination = tmp_path / "drafted.json"
    assert (
        main(
            [
                "query",
                "draft",
                "--skills",
                str(skill_repo),
                "--queries",
                str(destination),
                "--agent",
                "fake",
            ],
        )
        == 0
    )
    assert query_set_digest(load_query_set(destination)) == query_set_digest(drafted)


@pytest.fixture
def viewable_set(skill_repo: Path, tmp_path: Path) -> Path:
    """Write sample query set for CLI table view testing."""
    query_set = QuerySet(
        catalog_id="all",
        queries=(
            Query(
                id="v-1",
                text="Tier old objects to Coldline after 30 days.",
                expected_skill="gcs-lifecycle-rules",
            ),
            Query(
                id="v-2",
                text="Keep audit logs for seven years for compliance.",
                expected_skill="gcs-retention-policy",
            ),
        ),
        provenance=QuerySetProvenance(origin=Origin.AUTHORED),
    )
    return save_query_set(query_set, tmp_path / "viewable.json")


def test_query_view_renders_a_table_without_writing_anything(
    viewable_set: Path,
    skill_repo: Path,
    tmp_path: Path,
    capsys,
) -> None:
    """Verify query view prints table to stderr without modifying filesystem."""
    before = sorted(tmp_path.rglob("*"))
    assert main(["query", "view", str(viewable_set), "--skills", str(skill_repo)]) == 0

    shown = capsys.readouterr().err
    assert "v-1" in shown
    assert "gcs-lifecycle-rules" in shown
    assert "Tier old objects" in shown
    assert sorted(tmp_path.rglob("*")) == before


def test_query_view_leaks_column_is_off_by_default(
    viewable_set: Path,
    skill_repo: Path,
    capsys,
) -> None:
    """Verify query view omits leak column by default."""
    assert main(["query", "view", str(viewable_set), "--skills", str(skill_repo)]) == 0
    assert "leak" not in capsys.readouterr().err


def test_query_view_leaks_flag_adds_a_column(skill_repo: Path, tmp_path: Path, capsys) -> None:
    """Verify --leaks flag adds leak detection column in query view table."""
    query_set = QuerySet(
        catalog_id="all",
        queries=(
            Query(
                id="leaky",
                text="I need the gcs-lifecycle-rules skill.",
                expected_skill="gcs-lifecycle-rules",
            ),
        ),
        provenance=QuerySetProvenance(origin=Origin.AUTHORED),
    )
    path = save_query_set(query_set, tmp_path / "leaky.json")
    argv = ["query", "view", str(path), "--skills", str(skill_repo), "--leaks"]

    assert main(argv) == 0

    shown = capsys.readouterr().err
    assert "leak" in shown
    assert "names target" in shown


def test_query_view_citations_flag_reads_the_trail_beside_the_set(
    skill_repo: Path,
    tmp_path: Path,
    wide: None,
    capsys,
) -> None:
    """Verify --citations flag loads and displays passages from companion citation trail."""
    query_set = QuerySet(
        catalog_id="all",
        queries=(
            Query(
                id="grounded",
                text="Tier old objects to Coldline after 30 days.",
                expected_skill="gcs-lifecycle-rules",
            ),
        ),
        provenance=QuerySetProvenance(origin=Origin.AUTHORED),
    )
    path = save_query_set(query_set, tmp_path / "grounded.json")
    trail_path = tmp_path / "grounded-citations.json"
    trail_path.write_text(
        json.dumps(
            [
                {
                    "skill": "gcs-lifecycle-rules",
                    "text": "Tier old objects to Coldline after 30 days.",
                    "citation": "Objects move to Coldline after thirty days idle.",
                },
            ],
        ),
        encoding="utf-8",
    )

    assert main(["query", "view", str(path), "--skills", str(skill_repo), "--citations"]) == 0

    shown = capsys.readouterr().err
    assert "citation" in shown
    assert "Objects move to Coldline" in shown


def test_query_view_citations_flag_refuses_a_set_with_no_trail(
    viewable_set: Path,
    skill_repo: Path,
    capsys,
) -> None:
    """Verify --citations exits with code 2 when citation trail file is missing."""
    argv = [
        "query",
        "view",
        str(viewable_set),
        "--skills",
        str(skill_repo),
        "--citations",
    ]
    assert main(argv) == 2
    assert "no citation trail" in capsys.readouterr().err


def test_a_subcommand_is_required(capsys) -> None:
    """Verify bare CLI invocation exits with code 2 and usage instructions."""
    assert main([]) == 2
    captured = capsys.readouterr()
    assert captured.err.startswith("Usage:")
    assert captured.out == ""


def test_an_unknown_verb_names_the_verbs_that_exist(capsys) -> None:
    """Verify unrecognized subcommand lists available registered verbs."""
    assert main(["frobnicate"]) == 2
    reported = capsys.readouterr().err
    assert "frobnicate" in reported
    assert "eval" in reported


REORDER_CASES: list[tuple[list[str], list[str]]] = [
    ([], []),
    (["eval", "--dry-run"], ["eval", "--dry-run"]),
    (["--help"], ["--help"]),
    (["--verbose", "eval"], ["eval", "--verbose"]),
    (["--config", "reach.toml", "eval"], ["eval", "--config", "reach.toml"]),
    (["--skills", "./corpus", "eval", "--dry-run"], ["eval", "--dry-run", "--skills", "./corpus"]),
    (["--skill", "diff", "lint"], ["lint", "--skill", "diff"]),
    (["--skill=diff", "lint"], ["lint", "--skill=diff"]),
]
REORDER_IDS = [" ".join(raw) or "empty" for raw, _ in REORDER_CASES]


@pytest.mark.parametrize(("raw", "expected"), REORDER_CASES, ids=REORDER_IDS)
def test_the_verb_leads_however_it_was_typed(raw: list[str], expected: list[str]) -> None:
    """Verify argv normalization hoists the subcommand ahead of leading options."""
    assert _reorder_argv(raw) == expected


@pytest.mark.parametrize("verb", registered_verbs())
def test_an_option_value_is_never_mistaken_for_the_verb(verb: str) -> None:
    """Verify a verb-shaped option value does not hijack the executed subcommand."""
    assert _reorder_argv(["--skills", verb, "lint"]) == ["lint", "--skills", verb]


ALL_HELP_COMMANDS = [
    [],
    *[[verb] for verb in sorted(registered_verbs())],
    ["query", "draft"],
]
ALL_HELP_IDS = ["reach" if not argv else "-".join(argv) for argv in ALL_HELP_COMMANDS]

FORMATTED_ARGV = [["overlap"], ["eval"], ["diff"]]
ALL_VERB_COMMANDS = [argv for argv in ALL_HELP_COMMANDS if argv and argv != ["query"]]


@pytest.mark.parametrize("argv", ALL_HELP_COMMANDS, ids=ALL_HELP_IDS)
def test_every_verb_gets_the_styled_help(argv: list[str], capsys) -> None:
    """Verify --help renders with themed console panel styling across all verbs."""
    assert main([*argv, "--help"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("Usage:")
    assert "╭─" in out


@pytest.mark.parametrize("argv", FORMATTED_ARGV, ids=" ".join)
def test_every_format_flag_says_what_it_renders(argv: list[str], capsys, snapshot) -> None:
    """Verify --format flag help line explains rendered output format."""
    assert main([*argv, "--help"]) == 0
    assert _flag_line(capsys.readouterr().out, "--format") == snapshot


def test_a_runtime_that_probes_nothing_is_not_offered_as_though_it_did(
    capsys,
    snapshot,
) -> None:
    """Verify public agents are dynamically listed in help without internal fake runtime."""
    assert main(["eval", "--help"]) == 0
    assert _flag_line(capsys.readouterr().out, "--agent") == snapshot


@pytest.mark.parametrize("argv", ALL_VERB_COMMANDS, ids=" ".join)
def test_a_switch_never_states_its_default(argv: list[str], capsys) -> None:
    """Verify boolean switch flags omit redundant default false annotations in help output."""
    assert main([*argv, "--help"]) == 0
    assert "[default: False]" not in capsys.readouterr().out


def _flag_line(rendered: str, flag: str) -> str:
    """Extract and normalize help table row for specified flag."""
    lines = rendered.splitlines()
    start = next(
        i
        for i, line in enumerate(lines)
        if line.startswith("│") and line.strip("│ ").startswith(flag)
    )
    row = [lines[start]]
    for line in lines[start + 1 :]:
        if "--" in line or "╰" in line:
            break
        row.append(line)
    return " ".join(part.strip("│ ") for part in row)


@pytest.mark.parametrize("argv", ALL_HELP_COMMANDS, ids=ALL_HELP_IDS)
def test_help_answers_on_stdout(argv: list[str], capsys) -> None:
    """Verify --help writes formatted usage text to stdout rather than stderr."""
    assert main([*argv, "--help"]) == 0
    captured = capsys.readouterr()
    assert "Usage:" in captured.out
    assert captured.err == ""


LOOP_VERBS = {
    "check",
    "cluster",
    "diff",
    "eval",
    "lint",
    "optimize",
    "overlap",
    "query",
    "sweep",
    "view",
}
SETUP_VERBS = {"clean", "completion", "doctor", "init"}
ALL_VERBS = LOOP_VERBS | SETUP_VERBS


def test_the_verbs_are_listed_under_one_heading(capsys, snapshot) -> None:
    """Verify main help groups commands under standard headings."""
    assert main(["--help"]) == 0
    assert capsys.readouterr().out == snapshot


def test_no_removed_verb_is_still_listed() -> None:
    """Verify deprecated verbs are removed from the command registry."""
    assert not set(registered_verbs()) & {
        "run",
        "findings",
        "difficulty",
        "taxonomy",
    }


def test_every_registered_verb_is_one_the_loop_heading_claims() -> None:
    """Verify registered verbs match expected command sets."""
    assert set(registered_verbs()) == ALL_VERBS


def _verbs(section: str) -> set[str]:
    """Extract recognized loop verbs present within help section text."""
    return {word for word in section.split() if word in ALL_VERBS}


def test_the_listing_asks_for_a_verb_and_nothing_else(capsys) -> None:
    """Verify root help usage line specifies COMMAND placeholder."""
    assert main(["--help"]) == 0
    out = capsys.readouterr().out
    assert out.splitlines()[0] == "Usage: reach COMMAND"
    assert "TOKENS" not in out


def test_every_verb_is_filed_under_what_it_is_for(capsys, snapshot) -> None:
    """Verify help command categories match snapshot layout."""
    assert main(["--help"]) == 0
    _, _, about = capsys.readouterr().out.partition("About")
    assert about == snapshot


def test_the_version_flag_reports_the_version_that_is_installed(capsys) -> None:
    """Verify --version returns package metadata version."""
    assert main(["--version"]) == 0
    assert capsys.readouterr().out.strip() == metadata.version("skill-reach")


def test_the_run_flags_are_grouped_by_what_they_configure(capsys) -> None:
    """Verify eval --help organizes configuration options into distinct panels."""
    assert main(["eval", "--help"]) == 0
    out = capsys.readouterr().out
    for section in ("Catalog", "Runtime", "Plan", "Study"):
        assert section in out
    assert out.index("Study") < out.index("Catalog")


def test_the_unbuilt_verbs_are_not_advertised(capsys) -> None:
    """Verify unbuilt or legacy command names are absent from help output."""
    assert main(["--help"]) == 0
    out = capsys.readouterr().out
    assert not {"generate"} & set(out.split())
    assert not {"ablate", "compare"} & set(out.split())


VERB_MINIMUM = {
    ("overlap",): [],
    ("eval",): [],
    ("diff",): ["--vary", "description", "control.jsonl", "treatment.jsonl"],
    ("query", "draft"): [],
}

VIEWED_VERBS = {("eval",), ("query", "draft")}


@pytest.mark.parametrize(("verb", "required"), sorted(VERB_MINIMUM.items()))
@pytest.mark.parametrize("spelling", ["--quiet", "-q"], ids=["long", "short"])
def test_the_mute_is_offered_where_there_is_a_view(
    verb: tuple[str, ...],
    required: list[str],
    spelling: str,
) -> None:
    """Verify --quiet is available exclusively on commands with progress views."""
    if verb in VIEWED_VERBS:
        assert bind(*verb, spelling, *required)["quiet"] is True
        return
    with pytest.raises(CycloptsError):
        bind(*verb, spelling, *required)


def test_a_run_that_would_mix_arms_stops_before_touching_the_sidecar(
    base_argv: list[str],
    tmp_path: Path,
    capsys,
) -> None:
    """Verify appending with mismatched arm fails without altering existing sidecar file."""
    out = tmp_path / "results.jsonl"
    assert main([*base_argv, "--out", str(out)]) == 0
    sidecar = out.with_suffix(".jsonl.config.json")
    before = sidecar.read_text(encoding="utf-8")

    assert main([*base_argv, "--out", str(out), "--timeout", "999"]) == 2
    assert "refusing to append" in capsys.readouterr().err
    assert sidecar.read_text(encoding="utf-8") == before
    assert len(out.read_text(encoding="utf-8").splitlines()) == 2


def test_raising_the_attempt_count_deepens_the_file_it_already_wrote(
    base_argv: list[str],
    tmp_path: Path,
) -> None:
    """Verify increasing --attempts appends additional probes to existing result file."""
    out = tmp_path / "results.jsonl"
    assert main([*base_argv, "--out", str(out)]) == 0
    assert main([*base_argv, "--out", str(out), "--attempts", "3"]) == 0
    assert len(out.read_text(encoding="utf-8").splitlines()) == 6

    sidecar = json.loads(
        out.with_suffix(".jsonl.config.json").read_text(encoding="utf-8"),
    )
    assert sidecar["config"]["plan"]["attempts"] == 3
    assert sidecar["condition"]


def test_a_corrected_query_set_is_refused_before_the_sidecar_is_restated(
    base_argv: list[str],
    query_file: Path,
    tmp_path: Path,
    capsys,
) -> None:
    """Verify appending results from modified query set fails ground truth check."""
    out = tmp_path / "results.jsonl"
    assert main([*base_argv, "--out", str(out)]) == 0
    sidecar = out.with_suffix(".jsonl.config.json")
    before = sidecar.read_text(encoding="utf-8")

    original = json.loads(query_file.read_text(encoding="utf-8"))
    original["queries"][0]["expected_skill"] = "gcs-retention-policy"
    corrected = tmp_path / "corrected.json"
    corrected.write_text(json.dumps(original), encoding="utf-8")

    argv = [a if a != str(query_file) else str(corrected) for a in base_argv]
    assert main([*argv, "--out", str(out)]) == 2
    assert "ground truth" in capsys.readouterr().err
    assert sidecar.read_text(encoding="utf-8") == before
    assert len(out.read_text(encoding="utf-8").splitlines()) == 2


def test_mixing_arms_can_be_asked_for_on_the_command_line(
    base_argv: list[str],
    tmp_path: Path,
) -> None:
    """Verify --append-across-arms allows appending runs with different arm configurations."""
    out = tmp_path / "results.jsonl"
    assert main([*base_argv, "--out", str(out)]) == 0
    assert main([*base_argv, "--out", str(out), "--attempts", "3", "--append-across-arms"]) == 0
    assert len(out.read_text(encoding="utf-8").splitlines()) == 6


@pytest.fixture
def global_skills_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Provide a mock user home directory populated with valid global skills."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    skills = [
        ("k8s-cluster", "Deploys cloud compute instances and manages cluster lifecycles."),
        ("db-analyzer", "Analyzes relational database query performance and indexes."),
        ("net-dns", "Configures DNS records and manages network load balancers."),
    ]
    for name, desc in skills:
        s_dir = tmp_path / ".agents" / "skills" / name
        s_dir.mkdir(parents=True, exist_ok=True)
        (s_dir / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: {desc}\n---\nBody\n",
            encoding="utf-8",
        )
    return tmp_path


@pytest.mark.parametrize("verb", ["lint", "overlap", "check"])
@pytest.mark.parametrize("flag", ["--global", "-g"])
def test_global_flag_prefix_and_postfix(
    verb: str,
    flag: str,
    global_skills_home: Path,
) -> None:
    """Verify reach verbs support global flag in both prefix and postfix positions."""
    assert main([verb, flag]) == 0
    assert main([flag, verb]) == 0


def test_global_flag_combined_with_top_level_agent(global_skills_home: Path) -> None:
    """Verify global flag works when combined with top-level --agent flag."""
    assert main(["--agent", "antigravity-cli", "--global", "lint"]) == 0
    assert main(["--global", "--agent", "antigravity-cli", "lint"]) == 0


def test_global_flag_sad_path_empty_home(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify helpful error message when no skills exist in user global locations."""
    empty_home = tmp_path / "empty_home"
    empty_home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: empty_home)

    outcome = main(["lint", "--global"])
    assert outcome == 2
    captured = capsys.readouterr()
    assert "user global" in captured.err or "user global" in captured.out


@pytest.fixture(scope="module")
def cli_init_content() -> str:
    """Read reach/cli/__init__.py content once per test module."""
    root = Path(__file__).resolve().parent.parent.parent
    return (root / "src" / "reach" / "cli" / "__init__.py").read_text(encoding="utf-8")


@pytest.mark.parametrize("verb", registered_verbs())
def test_cli_subcommand_is_imported_in_cli_init(verb: str, cli_init_content: str) -> None:
    """Verify registered CLI verb is explicitly imported in reach.cli.__init__."""
    import re

    assert re.search(rf"\bfrom\s+\.\s+import\s+.*\b{verb}\b", cli_init_content, re.DOTALL), (
        f"CLI verb {verb!r} is not explicitly imported in reach/cli/__init__.py"
    )


def test_agent_cli_literal_matches_known_agents() -> None:
    """Verify AgentName literal type in reach.cli.flags matches known_agents registry."""
    import typing

    target_type = AgentName.__value__ if hasattr(AgentName, "__value__") else AgentName
    cli_agents = set(typing.get_args(target_type))
    registered_agents = set(known_agents())
    assert cli_agents == registered_agents, (
        f"AgentName choices {cli_agents} do not match known_agents {registered_agents}"
    )


def test_query_draft_destination_collision_and_force(
    skill_repo: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify query drafting rejects existing destination unless --force is supplied."""
    out = tmp_path / "existing_queries.json"
    out.write_text("{}", encoding="utf-8")

    # Without --force: exits with error code 2 and informs user
    outcome = main(["query", str(skill_repo), "--out", str(out), "--agent", "fake"])
    assert outcome == 2
    err = capsys.readouterr().err
    assert "already exists" in err
    assert "--force" in err

    # With --force: overwrites destination file
    outcome = main(["query", str(skill_repo), "--out", str(out), "--force", "--agent", "fake"])
    assert outcome == 0
    assert out.exists()

    # Verify -f format works alongside --force without short-flag collision
    out_jsonl = tmp_path / "existing_queries.jsonl"
    out_jsonl.write_text("{}", encoding="utf-8")
    outcome = main(
        [
            "query",
            str(skill_repo),
            "--out",
            str(out_jsonl),
            "-f",
            "jsonl",
            "--force",
            "--agent",
            "fake",
        ]
    )
    assert outcome == 0


def test_bare_query_command_auto_discovers_skills(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify bare reach query without target auto-discovers skills from workspace."""
    workspace = tmp_path / "my_project"
    skill_dir = workspace / ".agents" / "skills" / "test-skill"
    skill_dir.mkdir(parents=True)
    manifest = skill_dir / "SKILL.md"
    manifest.write_text(
        "---\nname: test-skill\ndescription: A test skill for discovery.\n---\n# Test\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(workspace)
    out = workspace / "discovered_queries.json"
    outcome = main(["query", "--out", str(out), "--generator-agent", "fake"])
    assert outcome == 0
    assert out.is_file()


def test_build_drafter_runtime_resolution() -> None:
    """Verify _build_drafter_runtime respects generator_agent override and defaults."""
    from reach.cli.drafting import _build_drafter_runtime
    from reach.cli.flags import GenerateFlags
    from reach.config import RunConfig, RuntimeSettings

    # Inherits settings.runtime.agent by default
    cfg = RunConfig(runtime=RuntimeSettings(agent="fake"))
    drafter = _build_drafter_runtime(cfg, GenerateFlags())
    assert drafter.name == "fake"

    # Explicit generator_agent overrides runtime agent
    flags = GenerateFlags(generator_agent="fake")
    cfg_cli = RunConfig(runtime=RuntimeSettings(agent="antigravity-cli"))
    drafter_override = _build_drafter_runtime(cfg_cli, flags)
    assert drafter_override.name == "fake"


def test_build_drafter_runtime_selects_agent_default_model_for_non_gemini() -> None:
    """Verify _build_drafter_runtime switches default model when non-Gemini agent is chosen."""
    from reach.cli.drafting import _build_drafter_runtime
    from reach.cli.flags import GenerateFlags
    from reach.config import RunConfig, RuntimeSettings

    # Explicit generator_agent switch
    flags = GenerateFlags(generator_agent="claude-code")
    cfg = RunConfig(runtime=RuntimeSettings(agent="antigravity-cli"))
    drafter = _build_drafter_runtime(cfg, flags)
    assert drafter.name == "claude-code"
    assert "claude" in drafter.model

    # Inherited from runtime settings without explicit generator_agent flag
    cfg_inherited = RunConfig(runtime=RuntimeSettings(agent="claude-code"))
    drafter_inherited = _build_drafter_runtime(cfg_inherited, GenerateFlags())
    assert drafter_inherited.name == "claude-code"
    assert "claude" in drafter_inherited.model


def test_cli_diff_and_view_with_slicing_flags(
    make_config,
    record_arm,
    skill_repo: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify reach diff and reach view support --queries, --filter-skill, and --filter-id."""
    import shutil

    edited = tmp_path / "skills-edited"
    shutil.copytree(skill_repo, edited)
    card = edited / "storage" / "gcs-retention-policy" / "SKILL.md"
    card.write_text(
        card.read_text(encoding="utf-8").replace(
            "Configures retention and bucket lock.",
            "Holds audit logs for a fixed period under bucket lock.",
        ),
        encoding="utf-8",
    )

    control = record_arm(
        "control",
        make_config(catalog={"size": 3, "rivals": 2}, plan={"attempts": 5}),
        {"q-lifecycle": ("gcs-lifecycle-rules",) * 5, "q-retention": (None,) * 5},
    )
    treatment = record_arm(
        "treatment",
        make_config(
            catalog={"size": 3, "rivals": 2},
            plan={"attempts": 5},
            study={"skills": edited},
        ),
        {
            "q-lifecycle": ("gcs-lifecycle-rules",) * 5,
            "q-retention": ("gcs-retention-policy",) * 5,
        },
    )

    # 1. reach diff with --filter-skill
    assert (
        main(
            [
                "diff",
                str(control),
                str(treatment),
                "--vary",
                "description",
                "--filter-skill",
                "gcs-retention-*",
                "--format",
                "json",
            ]
        )
        == 0
    )
    diff_json = capsys.readouterr().out
    assert '"shared_queries": 1' in diff_json
    assert '"q-retention"' in diff_json
    assert '"q-lifecycle"' not in diff_json

    # 2. reach view with --filter-id on a .jsonl file (with sidecar)
    assert (
        main(
            [
                "view",
                str(treatment),
                "--filter-id",
                "*-retention",
                "--format",
                "json",
            ]
        )
        == 0
    )
    view_json = capsys.readouterr().out
    assert '"q-retention"' in view_json
    assert '"q-lifecycle"' not in view_json
