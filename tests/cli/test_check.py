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

"""Verify `reach check` CLI arguments, exit codes, and GitHub Actions integration."""

from __future__ import annotations

from typing import TYPE_CHECKING

from reach.cli import main
from reach.models import Query

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    import pytest


def test_check_help(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify reach check --help prints command options and exits 0."""
    assert main(["check", "--help"]) == 0
    captured = capsys.readouterr()
    out = captured.out.lower()
    assert any(term in out for term in ("quality gate", "regression", "check"))


def test_check_stage1_clean_exits_0(
    write_skill: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """Verify reach check passes on clean skill manifests when queries omitted."""
    write_skill(
        name="valid-tool",
        description="A completely valid skill description providing sufficient context.",
    )
    assert main(["check", "--skills", str(tmp_path / "valid-tool")]) == 0


def test_check_stage1_static_error_exits_1(
    write_skill: Callable[..., Path],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify reach check exits 1 on static lint errors."""
    write_skill(
        name="InvalidName",
        description="A sufficiently detailed description that satisfies standard rules.",
    )
    assert main(["check", "--skills", str(tmp_path / "InvalidName")]) == 1


def test_check_empirical_pass(
    write_skill: Callable[..., Path],
    write_queries: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """Verify reach check passes Stage 1 and Stage 2 with clean skills and queries."""
    write_skill(
        name="math-helper",
        description="Perform arithmetic operations and math calculations reliably.",
    )
    query_file = write_queries(target="math-helper", count=4)
    assert (
        main(
            [
                "check",
                "--skills",
                str(tmp_path / "math-helper"),
                "--queries",
                str(query_file),
                "--agent",
                "fake",
                "--min-recall",
                "0.75",
                "--min-accuracy",
                "0.75",
            ],
        )
        == 0
    )


def test_check_empirical_regression_exits_2(
    write_skill: Callable[..., Path],
    write_queries: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """Verify reach check exits 2 when empirical assertions are violated."""
    write_skill(
        name="crypto-tool",
        description="Encrypt and decrypt sensitive payload data with AES keys.",
    )
    # Queries targeting an unknown skill will cause recall to be 0%
    queries = [
        Query(
            id="q-0",
            text="Query without mentioning the target skill name",
            expected_skill="crypto-tool",
        ),
    ]
    query_file = write_queries(queries=queries)

    assert (
        main(
            [
                "check",
                "--skills",
                str(tmp_path / "crypto-tool"),
                "--queries",
                str(query_file),
                "--agent",
                "fake",
                "--min-recall",
                "0.90",
            ],
        )
        == 2
    )


def test_check_format_github_emits_annotations(
    write_skill: Callable[..., Path],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify reach check --format github emits workflow command annotations."""
    write_skill(
        name="bad_naming",
        description="A sufficiently detailed description that satisfies standard rules.",
    )
    assert main(["check", "--skills", str(tmp_path / "bad_naming"), "--format", "github"]) == 1
    captured = capsys.readouterr()
    assert "::error" in captured.out
    assert "title=invalid-name-format" in captured.out


def test_check_auto_detects_github_actions_and_writes_summary(
    write_skill: Callable[..., Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify reach check auto-detects GITHUB_ACTIONS and writes step summary."""
    summary_file = tmp_path / "step_summary.md"
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary_file))

    write_skill(
        name="ci-skill",
        description="A sufficiently detailed description that satisfies standard rules.",
    )

    assert main(["check", "--skills", str(tmp_path / "ci-skill")]) == 0
    assert summary_file.exists()
    content = summary_file.read_text(encoding="utf-8")
    assert "# Reach Quality Gate: ✅ PASSED" in content
    assert "Stage 1: Static Pre-flight" in content


def test_check_no_strict_flag(
    write_skill: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """Verify reach check --no-strict permits warnings and exits 0."""
    write_skill(
        name="bash",
        description="Run shell commands reliably in terminal environment.",
    )
    # Default strict mode exits 1 due to reserved name collision warning
    assert main(["check", "--skills", str(tmp_path / "bash")]) == 1
    # --no-strict permits warnings and exits 0
    assert main(["check", "--skills", str(tmp_path / "bash"), "--no-strict"]) == 0


def test_check_non_existent_path_fails() -> None:
    """Verify reach check on a non-existent path fails with exit code 2."""
    assert main(["check", "--skills", "/non/existent/path"]) == 2


def test_check_ignore_rule_flag(
    write_skill: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """Verify reach check --ignore bypasses named rule failures."""
    write_skill(
        name="different-name",
        description="A sufficiently detailed description that satisfies standard rules.",
        dir_name="actual-dir",
    )
    assert main(["check", "--skills", str(tmp_path)]) == 1
    assert main(["check", "--skills", str(tmp_path), "--ignore", "name-mismatch"]) == 0


def test_check_positional_path(
    write_skill: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """Verify reach check accepts skills directory as positional argument."""
    write_skill(
        name="valid-skill",
        description="A sufficiently detailed description that satisfies standard rules.",
    )
    assert main(["check", str(tmp_path)]) == 0


def test_check_cli_since_flag_overrides_config(
    write_skill: Callable[..., Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify explicit --since HEAD~1 overrides config since = origin/main."""
    from unittest.mock import patch

    from reach.check import CheckOutcome
    from reach.lint import LintReport

    cfg_file = tmp_path / "reach.toml"
    cfg_file.write_text(
        '[check]\nsince = "origin/main"\n',
        encoding="utf-8",
    )
    skill_dir = write_skill(
        name="valid-skill",
        description="A sufficiently detailed description that satisfies standard rules.",
    )

    mock_outcome = CheckOutcome(lint_report=LintReport(), skills_checked=1)

    with patch("reach.cli.check.run_check", return_value=mock_outcome) as mock_run:
        # Case 1: Flag omitted -> uses config value "origin/main"
        main(["check", str(skill_dir), "--config", str(cfg_file)])
        assert mock_run.call_args.kwargs["since"] == "origin/main"

        # Case 2: Explicit --since HEAD~1 -> overrides config value "origin/main"
        main(["check", str(skill_dir), "--config", str(cfg_file), "--since", "HEAD~1"])
        assert mock_run.call_args.kwargs["since"] == "HEAD~1"


def test_check_cli_trajectory_flags_passed_to_run_check(
    write_skill: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """Verify trajectory threshold CLI flags are passed correctly to run_check."""
    from unittest.mock import patch

    from reach.check import CheckOutcome
    from reach.lint import LintReport

    skill_dir = write_skill(
        name="valid-skill",
        description="A sufficiently detailed description that satisfies standard rules.",
    )
    mock_outcome = CheckOutcome(lint_report=LintReport(), skills_checked=1)

    with patch("reach.cli.check.run_check", return_value=mock_outcome) as mock_run:
        main(
            [
                "check",
                str(skill_dir),
                "--min-entrypoint",
                "0.85",
                "--min-reachability",
                "0.90",
                "--min-efficiency",
                "0.80",
                "--min-f1",
                "0.82",
                "--max-redundancy",
                "0.5",
            ]
        )
        kwargs = mock_run.call_args.kwargs
        assert kwargs["min_entrypoint"] == 0.85
        assert kwargs["min_reachability"] == 0.90
        assert kwargs["min_efficiency"] == 0.80
        assert kwargs["min_f1"] == 0.82
        assert kwargs["max_redundancy"] == 0.5


def test_check_cli_range_validators_reject_invalid_values(capsys) -> None:
    """Verify cyclopts range validators reject out-of-bounds numeric CLI flags."""
    assert main(["check", "--min-recall", "1.5"]) == 2
    assert "Must be <= 1.0" in capsys.readouterr().err

    assert main(["check", "--min-recall", "-0.1"]) == 2
    assert "Must be >= 0.0" in capsys.readouterr().err

    assert main(["check", "--budget", "0"]) == 2
    assert "Must be >= 1" in capsys.readouterr().err

    assert main(["check", "--max-redundancy", "-0.5"]) == 2
    assert "Must be >= 0.0" in capsys.readouterr().err


def test_check_quiet_short_flag(
    write_skill: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """Verify -q maps to quiet mode and executes cleanly without flag collisions."""
    write_skill(
        name="valid-tool",
        description="A completely valid skill description providing sufficient context.",
    )
    assert main(["check", "-q", "--skills", str(tmp_path / "valid-tool")]) == 0


def test_render_check_concise_reports_failed_on_strict_warnings(capsys) -> None:
    """Verify render_check_concise prints FAILED when strict mode fails on warnings."""
    from pathlib import Path

    from reach.check import CheckOutcome, CheckStage
    from reach.cli.check import render_check_concise
    from reach.lint import LintIssue, LintReport, Severity
    from reach.views import build_console

    warning = LintIssue(
        rule="description-too-short",
        severity=Severity.WARN,
        skill="my-skill",
        path=Path("SKILL.md"),
        message="Too short",
    )
    outcome = CheckOutcome(
        lint_report=LintReport(issues=(warning,), skills_checked=1),
        exit_code=1,
        stage_failed=CheckStage.STATIC,
    )
    render_check_concise(build_console(), outcome)
    err = capsys.readouterr().err
    assert "FAILED (strict: 1 warnings)" in err


def test_check_cli_filter_flags(
    write_skill: Callable[..., Path],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify reach check passes filter_skill and filter_id to run_check."""
    import reach.cli.check as cli_check

    passed_kwargs = {}

    def _mock_run_check(*args, **kwargs):
        passed_kwargs.update(kwargs)
        from reach.check import CheckOutcome
        from reach.lint import LintReport

        return CheckOutcome(
            lint_report=LintReport(skills_checked=1),
            exit_code=0,
            queries_probed=1,
            probes_executed=1,
            budget=50,
        )

    monkeypatch.setattr(cli_check, "run_check", _mock_run_check)

    skill_path = write_skill(name="tool-a", description="Valid description.")
    code = main(
        [
            "check",
            "--skills",
            str(skill_path),
            "--queries",
            str(tmp_path / "queries.json"),
            "--filter-skill",
            "tool-a",
            "--filter-id",
            "q-1",
        ]
    )
    assert code == 0
    assert passed_kwargs.get("filter_skill") == ("tool-a",)
    assert passed_kwargs.get("filter_id") == ("q-1",)


def test_check_cli_filter_glob_flags(
    write_skill: Callable[..., Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that reach check passes glob filter patterns to run_check."""
    import reach.cli.check as cli_check

    passed_kwargs = {}

    def _mock_run_check(*args, **kwargs):
        passed_kwargs.update(kwargs)
        from reach.check import CheckOutcome
        from reach.lint import LintReport

        return CheckOutcome(
            lint_report=LintReport(skills_checked=1),
            exit_code=0,
            queries_probed=1,
            probes_executed=1,
            budget=50,
        )

    monkeypatch.setattr(cli_check, "run_check", _mock_run_check)

    skill_path = write_skill(name="tool-a", description="Valid description.")
    code = main(
        [
            "check",
            "--skills",
            str(skill_path),
            "--queries",
            str(tmp_path / "queries.json"),
            "--filter-skill",
            "tool-*",
            "--filter-id",
            "q-*",
        ]
    )
    assert code == 0
    assert passed_kwargs.get("filter_skill") == ("tool-*",)
    assert passed_kwargs.get("filter_id") == ("q-*",)
