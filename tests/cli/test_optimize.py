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

"""Verify reach optimize CLI command operations."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from reach.cli import main
from reach.optimize import OptimizationCandidate

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


def test_optimize_missing_skill_flag_fails() -> None:
    """Verify reach optimize without --skill fails with exit code 2."""
    assert main(["optimize"]) == 2


def test_optimize_budget_less_than_one_fails(
    write_skill: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """Verify reach optimize with --budget 0 fails with exit code 2."""
    write_skill(name="my-tool", description="Valid description.")
    assert main(["optimize", "--skill", "my-tool", "--skills", str(tmp_path), "--budget", "0"]) == 2


def test_optimize_positional_unknown_skill_fails(
    write_skill: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """Verify reach optimize with positional missing skill fails with exit code 2."""
    write_skill(name="other-tool", description="Other description.")
    assert main(["optimize", "missing-tool", "--skills", str(tmp_path)]) == 2


def test_optimize_text_output(
    write_skill: Callable[..., Path],
    write_queries: Callable[..., Path],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify reach optimize runs with default text format and emits progress and scorecard."""
    write_skill(
        name="opt-tool",
        description="Old basic description.",
        body="# Opt Tool\nProvides tokenization and string formatting.",
    )
    query_file = write_queries(target="opt-tool", count=3)

    with patch("sys.stderr.isatty", return_value=False):
        ret = main(
            [
                "optimize",
                "--skill",
                "opt-tool",
                "--skills",
                str(tmp_path),
                "--queries",
                str(query_file),
                "--agent",
                "fake",
                "--budget",
                "6",
            ],
        )
    assert ret == 0
    captured = capsys.readouterr()
    assert "[reach optimize]" in captured.err
    output = captured.err + captured.out
    assert "Reach Closed-Loop Optimizer: opt-tool" in output
    assert "Candidate Description" in output
    assert "#1" in output


def test_optimize_json_output(
    write_skill: Callable[..., Path],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify reach optimize --format json emits valid machine-readable JSON."""
    write_skill(name="json-tool", description="Initial description.")
    ret = main(
        [
            "optimize",
            "--skill",
            "json-tool",
            "--skills",
            str(tmp_path),
            "--agent",
            "fake",
            "--iterations",
            "1",
            "--candidates",
            "1",
            "--budget",
            "1",
            "--format",
            "json",
        ],
    )
    assert ret == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["skill_name"] == "json-tool"
    assert "candidates" in data
    assert len(data["candidates"]) >= 1


def test_optimize_diff_output(
    write_skill: Callable[..., Path],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify reach optimize --format diff emits unified diff against best candidate."""
    write_skill(name="diff-tool", description="Initial description.")
    ret = main(
        [
            "optimize",
            "--skill",
            "diff-tool",
            "--skills",
            str(tmp_path),
            "--agent",
            "fake",
            "--no-auto-queries",
            "--iterations",
            "1",
            "--format",
            "diff",
        ],
    )
    assert ret == 0

    captured = capsys.readouterr()
    assert "--- a/diff-tool/SKILL.md" in captured.out
    assert "+++ b/diff-tool/SKILL.md" in captured.out
    assert "-  Initial description." in captured.out


def _mock_eval_candidate(
    candidate: OptimizationCandidate,
    is_test: bool = False,
    **_kwargs: object,
) -> OptimizationCandidate:
    """Mock evaluate_candidate preserving candidate recall and accuracy."""
    if is_test:
        return candidate.model_copy(
            update={"test_recall": candidate.recall, "test_accuracy": candidate.accuracy}
        )
    return candidate


def test_optimize_auto_apply_writes_to_disk(
    write_skill: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """Verify reach optimize --auto-apply modifies the SKILL.md file on disk."""
    skill_dir = write_skill(
        name="apply-tool",
        description="Description before optimization.",
        body="# Body\nDistinct content to preserve.",
    )
    manifest = skill_dir / "SKILL.md"

    mock_candidates = [
        OptimizationCandidate(
            description="Optimized candidate description written to disk.",
            rationale="Auto-apply test.",
            recall=1.0,
            delta_recall=0.5,
        ),
    ]

    with (
        patch("reach.optimize.synthesize_candidates", return_value=mock_candidates),
        patch("reach.optimize.evaluate_candidate", side_effect=_mock_eval_candidate),
    ):
        ret = main(
            [
                "optimize",
                "--skill",
                "apply-tool",
                "--skills",
                str(tmp_path),
                "--no-auto-queries",
                "--auto-apply",
                "--agent",
                "fake",
            ],
        )
        assert ret == 0

    content = manifest.read_text(encoding="utf-8")
    assert "Optimized candidate description written to disk." in content
    assert "Distinct content to preserve." in content


def test_optimize_interactive_prompt_yes_applies_candidate(
    write_skill: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """Verify interactive TTY prompt applies candidate when user inputs 'y'."""
    skill_dir = write_skill(
        name="prompt-tool",
        description="Original description.",
    )
    manifest = skill_dir / "SKILL.md"

    mock_candidates = [
        OptimizationCandidate(
            description="Interactively applied description.",
            recall=0.9,
            delta_recall=0.3,
        ),
    ]

    with (
        patch("reach.optimize.synthesize_candidates", return_value=mock_candidates),
        patch("reach.optimize.evaluate_candidate", side_effect=_mock_eval_candidate),
        patch("sys.stdin.isatty", return_value=True),
        patch("sys.stdout.isatty", return_value=True),
        patch("builtins.input", return_value="y"),
    ):
        ret = main(
            [
                "optimize",
                "prompt-tool",
                "--skills",
                str(tmp_path),
                "--no-auto-queries",
                "--agent",
                "fake",
            ]
        )
        assert ret == 0

    assert "Interactively applied description." in manifest.read_text(encoding="utf-8")


def test_optimize_interactive_prompt_no_skips_candidate(
    write_skill: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """Verify interactive TTY prompt leaves file unchanged when user inputs 'n'."""
    skill_dir = write_skill(
        name="prompt-tool",
        description="Original description.",
    )
    manifest = skill_dir / "SKILL.md"

    mock_candidates = [
        OptimizationCandidate(
            description="Declined description.",
            recall=0.9,
            delta_recall=0.3,
        ),
    ]

    with (
        patch("reach.optimize.synthesize_candidates", return_value=mock_candidates),
        patch("reach.optimize.evaluate_candidate", side_effect=_mock_eval_candidate),
        patch("sys.stdin.isatty", return_value=True),
        patch("sys.stdout.isatty", return_value=True),
        patch("builtins.input", return_value="n"),
    ):
        ret = main(
            [
                "optimize",
                "prompt-tool",
                "--skills",
                str(tmp_path),
                "--no-auto-queries",
                "--agent",
                "fake",
            ]
        )
        assert ret == 0

    assert "Original description." in manifest.read_text(encoding="utf-8")


def test_optimize_interactive_prompt_diff_then_yes(
    write_skill: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """Verify user can request diff before applying candidate in interactive prompt."""
    skill_dir = write_skill(
        name="prompt-tool",
        description="Original description.",
    )
    manifest = skill_dir / "SKILL.md"

    mock_candidates = [
        OptimizationCandidate(
            description="Diff inspected then applied description.",
            recall=0.9,
            delta_recall=0.3,
        ),
    ]

    inputs = iter(["d", "y"])
    with (
        patch("reach.optimize.synthesize_candidates", return_value=mock_candidates),
        patch("reach.optimize.evaluate_candidate", side_effect=_mock_eval_candidate),
        patch("sys.stdin.isatty", return_value=True),
        patch("sys.stdout.isatty", return_value=True),
        patch("builtins.input", side_effect=lambda _: next(inputs)),
    ):
        ret = main(
            [
                "optimize",
                "prompt-tool",
                "--skills",
                str(tmp_path),
                "--no-auto-queries",
                "--agent",
                "fake",
            ]
        )
        assert ret == 0

    assert "Diff inspected then applied description." in manifest.read_text(encoding="utf-8")


def test_prompt_interactive_apply_noop_when_best_candidate_is_none(tmp_path: Path) -> None:
    """Verify _prompt_interactive_apply returns cleanly when best_candidate is None."""
    from reach.cli.optimize import _prompt_interactive_apply
    from reach.optimize import OptimizationReport
    from reach.views import build_console

    console = build_console()
    report = OptimizationReport(
        skill_name="test-tool",
        baseline_description="test",
        manifest_path=tmp_path / "SKILL.md",
        candidates=(),
    )
    # Calling this should return immediately without raising AttributeError or prompt
    _prompt_interactive_apply(console, report)


def test_optimize_cli_forwards_new_parameters(
    write_skill: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """Verify reach optimize passes iterations, holdout, workers, candidate, and no-auto-queries."""
    write_skill(name="cli-tool", description="A CLI test tool.")

    with patch("reach.optimize.optimize_skill") as mock_opt:
        from reach.optimize import OptimizationReport

        mock_opt.return_value = OptimizationReport(
            skill_name="cli-tool",
            baseline_description="A CLI test tool.",
            manifest_path=tmp_path / "SKILL.md",
            candidates=(),
        )

        ret = main(
            [
                "optimize",
                "--skill",
                "cli-tool",
                "--skills",
                str(tmp_path),
                "--iterations",
                "3",
                "--holdout",
                "0.25",
                "--no-auto-queries",
                "--review",
                "--candidate",
                "2",
                "-j",
                "3",
                "-y",
            ],
        )
        assert ret == 0
        mock_opt.assert_called_once()
        _, kwargs = mock_opt.call_args
        settings = kwargs["settings"]
        assert settings.iterations == 3
        assert settings.holdout == 0.25
        assert settings.auto_queries is False
        assert settings.review is True
        assert settings.workers == 3
        assert kwargs["candidate_index"] == 2


def test_optimize_cli_defaults_holdout_and_yes_skips_tty_prompt(
    write_skill: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """Verify reach optimize defaults holdout to 0.2 and --yes skips TTY interactive prompt."""
    write_skill(name="def-tool", description="Default holdout tool.")

    with (
        patch("reach.optimize.optimize_skill") as mock_opt,
        patch("sys.stdin.isatty", return_value=True),
        patch("sys.stdout.isatty", return_value=True),
        patch("reach.cli.optimize._prompt_interactive_apply") as mock_prompt,
    ):
        from reach.optimize import OptimizationReport

        mock_opt.return_value = OptimizationReport(
            skill_name="def-tool",
            baseline_description="Default holdout tool.",
            manifest_path=tmp_path / "SKILL.md",
            candidates=(OptimizationCandidate(description="Cand 1", recall=0.9, delta_recall=0.2),),
        )

        ret = main(["optimize", "--skill", "def-tool", "--skills", str(tmp_path), "-y"])
        assert ret == 0
        mock_opt.assert_called_once()
        _, kwargs = mock_opt.call_args
        assert kwargs["settings"].holdout == 0.2
        mock_prompt.assert_not_called()


def test_optimize_diff_with_candidate_flag(
    write_skill: Callable[..., Path],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify reach optimize --format diff --candidate 2 displays candidate #2 diff."""
    write_skill(name="diff-tool", description="Original description.")
    mock_candidates = [
        OptimizationCandidate(description="Candidate 1 chosen rank 1.", recall=0.95),
        OptimizationCandidate(description="Candidate 2 chosen rank 2.", recall=0.75),
    ]
    with patch("reach.optimize.synthesize_candidates", return_value=mock_candidates):
        ret = main(
            [
                "optimize",
                "diff-tool",
                "--skills",
                str(tmp_path),
                "--agent",
                "fake",
                "--no-auto-queries",
                "--format",
                "diff",
                "--candidate",
                "2",
            ]
        )
        assert ret == 0
    captured = capsys.readouterr()
    assert "candidate #2" in captured.out
    assert "+  Candidate 2 chosen rank 2." in captured.out


def test_optimize_diff_with_out_of_range_candidate_fails(
    write_skill: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """Verify reach optimize --format diff --candidate 99 exits with error code 2."""
    write_skill(name="range-tool", description="Original description.")
    mock_candidates = [
        OptimizationCandidate(description="Candidate 1.", recall=0.8),
    ]
    with patch("reach.optimize.synthesize_candidates", return_value=mock_candidates):
        ret = main(
            [
                "optimize",
                "range-tool",
                "--skills",
                str(tmp_path),
                "--agent",
                "fake",
                "--no-auto-queries",
                "--format",
                "diff",
                "--candidate",
                "99",
            ]
        )
        assert ret == 2


def test_prompt_interactive_apply_selects_specific_candidate(
    write_skill: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """Verify user can choose candidate #2 in interactive prompt."""
    skill_dir = write_skill(
        name="prompt-multi",
        description="Original description.",
    )
    manifest = skill_dir / "SKILL.md"

    mock_candidates = [
        OptimizationCandidate(description="Candidate 1 description.", recall=0.9, delta_recall=0.2),
        OptimizationCandidate(description="Candidate 2 description.", recall=0.8, delta_recall=0.1),
    ]

    inputs = iter(["2"])
    with (
        patch("reach.optimize.synthesize_candidates", return_value=mock_candidates),
        patch("reach.optimize.evaluate_candidate", side_effect=_mock_eval_candidate),
        patch("sys.stdin.isatty", return_value=True),
        patch("sys.stdout.isatty", return_value=True),
        patch("builtins.input", side_effect=lambda _: next(inputs)),
    ):
        ret = main(
            [
                "optimize",
                "prompt-multi",
                "--skills",
                str(tmp_path),
                "--no-auto-queries",
                "--agent",
                "fake",
            ]
        )
        assert ret == 0

    assert "Candidate 2 description." in manifest.read_text(encoding="utf-8")


def test_optimize_interactive_prompt_skipped_when_no_improvement(
    write_skill: Callable[..., Path],
    write_queries: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """Verify interactive prompt is skipped when candidates show zero improvement over baseline."""
    skill_dir = write_skill(
        name="no-imp-tool",
        description="Original baseline description.",
    )
    manifest = skill_dir / "SKILL.md"
    qfile = write_queries(target="no-imp-tool", count=1)

    mock_candidates = [
        OptimizationCandidate(
            description="Candidate with zero improvement.",
            recall=1.0,
            delta_recall=0.0,
        ),
    ]

    with (
        patch("reach.optimize.synthesize_candidates", return_value=mock_candidates),
        patch("reach.optimize.filter_candidates", return_value=mock_candidates),
        patch(
            "reach.optimize.evaluate_candidate",
            return_value=mock_candidates[0].model_copy(update={"recall": 1.0, "delta_recall": 0.0}),
        ),
        patch("sys.stdin.isatty", return_value=True),
        patch("sys.stdout.isatty", return_value=True),
        patch("builtins.input", side_effect=AssertionError("input() should not be called")),
    ):
        ret = main(
            [
                "optimize",
                "no-imp-tool",
                "--skills",
                str(tmp_path),
                "--queries",
                str(qfile),
                "--iterations",
                "1",
                "--agent",
                "fake",
            ]
        )
        assert ret == 0

    # Manifest should not be modified
    assert "Original baseline description." in manifest.read_text(encoding="utf-8")


def test_optimize_cli_auto_apply_skips_when_no_improvement_unless_forced(
    write_skill: Callable[..., Path],
    write_queries: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """Verify reach optimize --auto-apply skips writing without improvement unless --force."""
    skill_dir = write_skill(
        name="no-imp-apply",
        description="Original baseline description that must stay.",
    )
    manifest = skill_dir / "SKILL.md"
    qfile = write_queries(target="no-imp-apply", count=1)

    mock_candidates = [
        OptimizationCandidate(
            description="Candidate with zero improvement.",
            recall=1.0,
            delta_recall=0.0,
        ),
    ]

    with (
        patch("reach.optimize.synthesize_candidates", return_value=mock_candidates),
        patch("reach.optimize.filter_candidates", return_value=mock_candidates),
        patch(
            "reach.optimize.evaluate_candidate",
            return_value=mock_candidates[0].model_copy(update={"recall": 1.0, "delta_recall": 0.0}),
        ),
    ):
        # 1. Without --force: file is NOT overwritten
        ret1 = main(
            [
                "optimize",
                "no-imp-apply",
                "--skills",
                str(tmp_path),
                "--queries",
                str(qfile),
                "--iterations",
                "1",
                "--agent",
                "fake",
                "--auto-apply",
            ]
        )
        assert ret1 == 0
        content = manifest.read_text(encoding="utf-8")
        assert "Original baseline description that must stay." in content

        # 2. With --force: file IS overwritten
        ret2 = main(
            [
                "optimize",
                "no-imp-apply",
                "--skills",
                str(tmp_path),
                "--queries",
                str(qfile),
                "--iterations",
                "1",
                "--agent",
                "fake",
                "--auto-apply",
                "--force",
            ]
        )
        assert ret2 == 0
        assert "Candidate with zero improvement." in manifest.read_text(encoding="utf-8")


def test_optimize_cli_explicit_candidate_prompts_even_without_improvement(
    write_skill: Callable[..., Path],
    write_queries: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """Verify specifying explicit --candidate prompts even if candidate has no improvement."""
    skill_dir = write_skill(
        name="explicit-cand-tool",
        description="Original baseline description.",
    )
    manifest = skill_dir / "SKILL.md"
    qfile = write_queries(target="explicit-cand-tool", count=1)

    mock_candidates = [
        OptimizationCandidate(
            description="Candidate 1 with zero improvement.",
            recall=1.0,
            delta_recall=0.0,
        ),
        OptimizationCandidate(
            description="Candidate 2 chosen by user.",
            recall=1.0,
            delta_recall=0.0,
        ),
    ]

    with (
        patch("reach.optimize.synthesize_candidates", return_value=mock_candidates),
        patch("reach.optimize.filter_candidates", return_value=mock_candidates),
        patch(
            "reach.optimize.evaluate_candidate",
            side_effect=lambda candidate, **_kw: candidate.model_copy(
                update={"recall": 1.0, "delta_recall": 0.0}
            ),
        ),
        patch("sys.stdin.isatty", return_value=True),
        patch("sys.stdout.isatty", return_value=True),
        patch("builtins.input", return_value="y"),
    ):
        ret = main(
            [
                "optimize",
                "explicit-cand-tool",
                "--skills",
                str(tmp_path),
                "--queries",
                str(qfile),
                "--iterations",
                "1",
                "--agent",
                "fake",
                "--candidate",
                "2",
            ]
        )
        assert ret == 0

    assert "Candidate 2 chosen by user." in manifest.read_text(encoding="utf-8")


@pytest.mark.parametrize("use_manifest", [False, True])
def test_optimize_positional_skill_directory_and_manifest_paths(
    write_skill: Callable[..., Path],
    capsys: pytest.CaptureFixture[str],
    use_manifest: bool,
) -> None:
    """Verify reach optimize accepts a skill directory or SKILL.md path positionally."""
    skill_dir = write_skill(name="path-tool", description="Old description.")
    target = skill_dir / "SKILL.md" if use_manifest else skill_dir
    ret = main(
        [
            "optimize",
            str(target),
            "--no-auto-queries",
            "--iterations",
            "1",
            "--candidates",
            "1",
            "--agent",
            "fake",
            "--format",
            "json",
        ]
    )
    assert ret == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["skill_name"] == "path-tool"


def test_optimize_multi_skill_directory_fails_with_guidance(
    write_skill: Callable[..., Path],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify reach optimize on a catalog directory fails fast with single-skill guidance."""
    write_skill(name="skill-a", description="A.")
    write_skill(name="skill-b", description="B.")
    ret = main(["optimize", str(tmp_path), "--agent", "fake"])
    assert ret == 2
    captured = capsys.readouterr()
    clean = " ".join((captured.err + captured.out).split())
    assert "is a directory containing" in clean
    assert "reach optimize" in clean


def test_optimize_typo_path_fails_cleanly(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify reach optimize on a non-existent path exits 2 with a clear error."""
    ret = main(["optimize", "./nonexistent/path/to/skill", "--agent", "fake"])
    assert ret == 2
    captured = capsys.readouterr()
    output = captured.err + captured.out
    assert "skill path does not exist" in output


def test_optimize_safety_notice_displays_inferred_catalog_count(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify safety notice reflects inferred catalog root count rather than 0 skills in ./.."""
    skills_root = tmp_path / "skills"
    skills_root.mkdir()
    skill_a = skills_root / "skill-a"
    skill_a.mkdir()
    (skill_a / "SKILL.md").write_text(
        "---\nname: skill-a\ndescription: A.\n---\n", encoding="utf-8"
    )
    skill_b = skills_root / "skill-b"
    skill_b.mkdir()
    (skill_b / "SKILL.md").write_text(
        "---\nname: skill-b\ndescription: B.\n---\n", encoding="utf-8"
    )

    with (
        patch("sys.stdin.isatty", return_value=True),
        patch("sys.stdout.isatty", return_value=True),
        patch("builtins.input", return_value="n"),
    ):
        ret = main(["optimize", str(skill_a), "--agent", "antigravity-cli"])
        assert ret == 1

    captured = capsys.readouterr()
    output = captured.err + captured.out
    assert "Target Catalog: 2 skills" in output


def test_optimize_with_handoff_cli_diff_and_auto_apply(
    write_skill: Callable[..., Path],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify --with-handoff renders target+rival diffs and patches both SKILL.md files."""
    target_dir = write_skill(
        name="metrics-collector",
        description="Provides query bottlenecks and telemetry collection.",
        body="# Metrics Collector\n\nUse region qualifier lookups in metric schemas.\n",
    )
    rival_dir = write_skill(
        name="metrics-analyzer",
        description="Analyzes query bottlenecks and cost optimization.",
        body="# Metrics Analyzer\n\nFix query plan bottlenecks and contention.\n",
    )
    ret_diff = main(
        [
            "optimize",
            "metrics-collector",
            "--skills",
            str(tmp_path),
            "--agent",
            "fake",
            "--no-auto-queries",
            "--iterations",
            "1",
            "--with-handoff",
            "--format",
            "diff",
        ]
    )
    assert ret_diff == 0
    diff_out = capsys.readouterr().out
    assert "a/metrics-collector/SKILL.md" in diff_out
    assert "a/metrics-analyzer/SKILL.md" in diff_out
    assert "> **Routing Note:**" in diff_out

    ret_apply = main(
        [
            "optimize",
            "metrics-collector",
            "--skills",
            str(tmp_path),
            "--agent",
            "fake",
            "--no-auto-queries",
            "--iterations",
            "1",
            "--with-handoff",
            "--auto-apply",
            "--force",
            "--yes",
        ]
    )
    assert ret_apply == 0
    assert "> **Routing Note:**" in (target_dir / "SKILL.md").read_text(encoding="utf-8")
    assert "> **Routing Note:**" in (rival_dir / "SKILL.md").read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("raw_choice", "input_reply", "expected"),
    [
        ("d 1", "y", (True, 1)),
        ("d", "n", (False, 1)),
        ("99", "n", None),
    ],
)
def test_parse_apply_choice_and_interactive_diff_helpers(
    write_skill: Callable[..., Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    raw_choice: str,
    input_reply: str,
    expected: tuple[bool, int] | None,
) -> None:
    """Verify _parse_apply_choice, _inspect_interactive_diff, and _prompt_interactive_apply."""
    from reach.cli.optimize import (
        _inspect_interactive_diff,
        _parse_apply_choice,
        _prompt_interactive_apply,
    )
    from reach.optimize import OptimizationReport
    from reach.views import build_console

    console = build_console()
    skill_dir = write_skill(name="helper-tool", description="Original helper description.")
    manifest = skill_dir / "SKILL.md"
    cand = OptimizationCandidate(
        description="Improved helper description.",
        recall=0.9,
        delta_recall=0.2,
    )
    report = OptimizationReport(
        skill_name="helper-tool",
        baseline_description="Original helper description.",
        manifest_path=manifest,
        baseline_recall=0.7,
        candidates=(cand,),
    )

    assert not _inspect_interactive_diff(console, report, 99, 1)
    monkeypatch.setattr("builtins.input", lambda _p: input_reply)
    assert _parse_apply_choice(raw_choice, 1, 1, console, report) == expected

    if raw_choice == "d 1":
        with patch("reach.optimize.apply_optimization_candidate", return_value=False):
            _prompt_interactive_apply(console, report)

        def _raise_eof(_p: str) -> str:
            raise EOFError

        monkeypatch.setattr("builtins.input", _raise_eof)
        _prompt_interactive_apply(console, report)
    capsys.readouterr()


def test_optimize_safety_config_toml_and_error_exits(
    write_skill: Callable[..., Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify _confirm_optimize_safety, --config TOML loading, and ValueError/RuntimeError exits."""
    from reach.cli.optimize import _confirm_optimize_safety
    from reach.views import build_console

    console = build_console()
    monkeypatch.setattr(
        "reach.config.resolve_discovery_candidates",
        lambda *_a, **_kw: [tmp_path / "empty_dir"],
    )
    assert (
        _confirm_optimize_safety(
            console,
            None,
            "fake",
            yes=True,
            run_config=None,
        )
        == 0
    )

    write_skill(name="helper-tool", description="Original helper description.")
    cfg_file = tmp_path / "reach.toml"
    cfg_file.write_text('[runtime]\nagent = "fake"\n', encoding="utf-8")

    with patch("reach.optimize.optimize_skill", side_effect=ValueError("bad opt")):
        assert (
            main(
                [
                    "optimize",
                    "helper-tool",
                    "--skills",
                    str(tmp_path),
                    "--config",
                    str(cfg_file),
                    "--yes",
                ]
            )
            == 2
        )
    with patch("reach.optimize.optimize_skill", side_effect=RuntimeError("boom")):
        assert main(["optimize", "helper-tool", "--skills", str(tmp_path), "--yes"]) == 3
    capsys.readouterr()
