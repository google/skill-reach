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

"""Verify `reach sweep` command line interface, argument validation, and output formatting."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from reach.cli import main
from reach.models import Query, QueryKind
from reach.queries import Origin, QuerySet, QuerySetProvenance, save_query_set
from reach.runtime.fake import FakeGenerator
from reach.sweep import ScalingPoint, ScalingStudy
from reach.uncertainty import Interval


@pytest.fixture
def sweep_corpus(corpus_builder, tmp_path: Path) -> tuple[Path, Path]:
    """Provide a corpus and query set for scaling sweep CLI tests."""
    corpus_dir = tmp_path / "corpus"
    builder = corpus_builder()
    for i in range(6):
        builder.add(
            f"skill-{i:02d}",
            f"Perform specialized task number {i:02d}",
            body=f"Instructions and documentation for specialized task number {i:02d}.",
        )
    builder.build_disk(corpus_dir)

    queries_file = tmp_path / "queries.json"
    queries = [
        Query(
            query_id=f"q-{i}",
            text=f"Requesting task number {i:02d}",
            expected_skill=f"skill-{i:02d}",
            kind=QueryKind.IMPLICIT,
        )
        for i in range(6)
    ]
    query_set = QuerySet(
        catalog_id="synthetic",
        queries=tuple(queries),
        provenance=QuerySetProvenance(origin=Origin.AUTHORED),
    )
    save_query_set(query_set, queries_file)
    return corpus_dir, queries_file


@pytest.fixture
def generator(monkeypatch: pytest.MonkeyPatch) -> FakeGenerator:
    """Provide a mock generator runtime producing valid positive and adversarial queries."""

    def _responder(prompt: str) -> str:
        if "near-miss" in prompt or "OUT OF SCOPE" in prompt:
            return json.dumps(
                {
                    "queries": [
                        {
                            "text": "Adversarial out-of-scope query",
                            "citation": "specialized task",
                            "rival_index": None,
                            "reason": "Not supported",
                        }
                    ]
                }
            )
        return json.dumps(
            {
                "queries": [
                    {
                        "text": "Perform specialized task for skill",
                        "citation": "specialized task",
                    }
                ]
            }
        )

    runtime = FakeGenerator(responses=_responder)
    monkeypatch.setattr("reach.cli.drafting.text_generator", lambda **_: runtime)
    return runtime


def _make_stub_study(
    *,
    scales: tuple[int, ...] = (2,),
    probes_errored: int = 0,
) -> ScalingStudy:
    """Construct a minimal ScalingStudy fixture for CLI plumbing tests."""
    point_scale = scales[0] if scales else 2
    point = ScalingPoint(
        scale=point_scale,
        catalog_id=f"sweep:corpus:{point_scale}",
        pass_rate=0.0 if probes_errored else 1.0,
        pass_rate_interval=(
            Interval(low=0.0, high=0.27) if probes_errored else Interval(low=1.0, high=1.0)
        ),
        recall=0.0 if probes_errored else 1.0,
        precision=0.0 if probes_errored else 1.0,
        f1_score=0.0 if probes_errored else 1.0,
        delta_vs_baseline=0.0,
        delta_abstention=0.0,
        delta_collision=0.0,
        probes_executed=max(1, probes_errored),
        probes_errored=probes_errored,
    )
    return ScalingStudy(
        target_skill=None,
        is_corpus_sweep=True,
        scales=scales,
        points=(point,) if scales else (),
        baseline_pass_rate=point.pass_rate,
        final_pass_rate=point.pass_rate,
        delta_total=0.0,
        delta_abstention=0.0,
        delta_collision=0.0,
        noise_floor=0.05,
        total_corpus_skills=max(scales) if scales else 2,
    )


@pytest.fixture
def stub_scaling_sweep(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Patch run_scaling_sweep in reach.cli.sweep and record all invocation kwargs."""
    calls: list[dict[str, Any]] = []

    def _fake_sweep(*_args: object, **kwargs: Any) -> ScalingStudy:
        calls.append(kwargs)
        return _make_stub_study()

    monkeypatch.setattr("reach.cli.sweep.run_scaling_sweep", _fake_sweep)
    return calls


def test_sweep_invalid_scales_exits_2(sweep_corpus: tuple[Path, Path]) -> None:
    """Verify invalid scales argument returns error code 2."""
    corpus_dir, queries_file = sweep_corpus
    exit_code = main(
        [
            "sweep",
            str(corpus_dir),
            "--queries",
            str(queries_file),
            "--scales",
            "invalid,not_numbers",
        ]
    )
    assert exit_code == 2


def test_sweep_text_output_corpus(sweep_corpus: tuple[Path, Path], capsys) -> None:
    """Verify reach sweep text output renders corpus capacity scaling table."""
    corpus_dir, queries_file = sweep_corpus
    exit_code = main(
        [
            "sweep",
            str(corpus_dir),
            "--queries",
            str(queries_file),
            "--scales",
            "1,3,5",
            "--agent",
            "fake",
        ]
    )
    assert exit_code == 0
    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert "Corpus Capacity Scaling:" in combined
    assert "Corpus Scaling: Multi-Class Retrieval & Capacity Degradation" in combined
    assert "F1 Trajectory:" in combined
    assert "—" in combined


def test_sweep_text_output_single_target(sweep_corpus: tuple[Path, Path], capsys) -> None:
    """Verify reach sweep with --target renders single-skill reachability decay table."""
    corpus_dir, queries_file = sweep_corpus
    exit_code = main(
        [
            "sweep",
            str(corpus_dir),
            "--target",
            "skill-00",
            "--queries",
            str(queries_file),
            "--scales",
            "1,3,5",
            "--agent",
            "fake",
        ]
    )
    assert exit_code == 0
    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert "Scaling Sweep: " in combined
    assert "Reachability Decay Across Catalog Scales" in combined
    assert "Trajectory:" in combined


def test_sweep_json_output(sweep_corpus: tuple[Path, Path], capsys) -> None:
    """Verify reach sweep --format json outputs ScalingStudy JSON and accepts bootstrap flags."""
    corpus_dir, queries_file = sweep_corpus
    exit_code = main(
        [
            "sweep",
            str(corpus_dir),
            "--queries",
            str(queries_file),
            "--scales",
            "1,3",
            "--agent",
            "fake",
            "--bootstrap-iterations",
            "50",
            "--seed",
            "123",
            "--format",
            "json",
        ]
    )
    assert exit_code == 0
    out = capsys.readouterr().out
    data = json.loads(out)
    assert data["is_corpus_sweep"] is True
    assert "points" in data
    # resolve_sweep_scales preserves explicit scales (1, 3) without appending total skills (6)
    assert len(data["points"]) == 2
    assert data["points"][0]["scale"] == 1
    assert data["points"][-1]["scale"] == 3
    assert data["points"][0]["abstention_rate"] is None
    assert data["points"][0]["prompt_tokens_mean"] is None
    assert data["points"][0]["negative_probes"] == 0


def test_sweep_csv_output(sweep_corpus: tuple[Path, Path], capsys) -> None:
    """Verify reach sweep --format csv outputs CSV document with headers."""
    corpus_dir, queries_file = sweep_corpus
    exit_code = main(
        [
            "sweep",
            str(corpus_dir),
            "--queries",
            str(queries_file),
            "--scales",
            "1,3",
            "--agent",
            "fake",
            "--format",
            "csv",
        ]
    )
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "scale,recall,recall_ci_low" in out
    assert "f1_score" in out


def test_sweep_empty_corpus_fails(tmp_path: Path, capsys) -> None:
    """Verify reach sweep on empty corpus exits with error code 2."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    exit_code = main(["sweep", str(empty_dir)])
    assert exit_code == 2
    err = capsys.readouterr().err
    assert "no skills found" in err


def test_sweep_missing_queries_fails_descriptively(
    sweep_corpus: tuple[Path, Path],
    monkeypatch,
    capsys,
) -> None:
    """Verify reach sweep with --no-auto-queries and without query file fails descriptively."""
    corpus_dir, _ = sweep_corpus
    empty_workspace = corpus_dir.parent / "nowhere"
    empty_workspace.mkdir()
    monkeypatch.chdir(empty_workspace)
    exit_code = main(["sweep", str(corpus_dir), "--no-auto-queries"])
    assert exit_code == 2
    err = capsys.readouterr().err
    assert "--queries" in err or ".reach/queries.json" in err


def test_sweep_discovers_default_queries_and_writes_default_out(
    sweep_corpus: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    """Verify reach sweep discovers .reach/queries.json and defaults --out to .reach/sweep.json."""
    corpus_dir, queries_file = sweep_corpus
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    reach_dir = workspace / ".reach"
    reach_dir.mkdir()
    (reach_dir / "queries.json").write_text(queries_file.read_text())
    monkeypatch.chdir(workspace)

    exit_code = main(["sweep", str(corpus_dir), "--scales", "1,3", "--agent", "fake"])
    assert exit_code == 0
    expected_out = reach_dir / "sweep.json"
    assert expected_out.exists()
    saved = json.loads(expected_out.read_text(encoding="utf-8"))
    assert "points" in saved

    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert "Corpus Capacity Scaling:" in combined or "Corpus Scaling" in combined
    assert "wrote" in combined
    assert ".reach/sweep.json" in combined


def test_sweep_discovers_queries_beside_skills_dir(
    corpus_builder,
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    """Verify reach sweep discovers .reach/queries.json relative to skills corpus path."""
    project_dir = tmp_path / "project"
    skills_dir = project_dir / "skills"
    builder = corpus_builder()
    for i in range(4):
        builder.add(f"skill-{i:02d}", f"Perform task {i:02d}")
    builder.build_disk(skills_dir)

    reach_dir = project_dir / ".reach"
    reach_dir.mkdir(parents=True)
    queries = [
        Query(
            query_id=f"q-{i}",
            text=f"Requesting task {i:02d}",
            expected_skill=f"skill-{i:02d}",
            kind=QueryKind.IMPLICIT,
        )
        for i in range(4)
    ]
    query_set = QuerySet(
        catalog_id="synthetic",
        queries=tuple(queries),
        provenance=QuerySetProvenance(origin=Origin.AUTHORED),
    )
    save_query_set(query_set, reach_dir / "queries.json")

    isolated_cwd = tmp_path / "isolated_cwd"
    isolated_cwd.mkdir()
    monkeypatch.chdir(isolated_cwd)

    exit_code = main(["sweep", str(skills_dir), "--scales", "1,2", "--agent", "fake"])
    assert exit_code == 0
    captured = capsys.readouterr()
    combined = captured.err + captured.out
    assert "Using benchmark queries from:" in combined


def test_sweep_custom_attempts_executes_expected_probe_count(
    sweep_corpus: tuple[Path, Path],
    capsys,
) -> None:
    """Verify --attempts flag controls probe executions per query at each scale."""
    corpus_dir, queries_file = sweep_corpus
    exit_code = main(
        [
            "sweep",
            str(corpus_dir),
            "--target",
            "skill-00",
            "--queries",
            str(queries_file),
            "--scales",
            "1",
            "--attempts",
            "3",
            "--agent",
            "fake",
            "--format",
            "json",
        ]
    )
    assert exit_code == 0
    out = capsys.readouterr().out
    data = json.loads(out)
    assert len(data["points"]) == 1  # scale 1 (exact requested scale)
    # Target skill has 1 query, with attempts=3, 3 probes should be executed
    assert data["points"][0]["probes_executed"] == 3


@pytest.mark.parametrize(
    ("scales_arg", "anchor_args", "expected_anchors"),
    [
        pytest.param("2,4", [], 2, id="default-medoids"),
        pytest.param("3,6", ["--anchor", "3"], 3, id="integer-cohort-count"),
        pytest.param(
            "2,4",
            ["--anchor", "skill-00,skill-02"],
            ["skill-00", "skill-02"],
            id="explicit-skill-names",
        ),
        pytest.param("2,4", ["--anchor", "all"], None, id="all-dynamic-expansion"),
    ],
)
def test_sweep_anchor_cli_modes(
    sweep_corpus: tuple[Path, Path],
    capsys: pytest.CaptureFixture[str],
    scales_arg: str,
    anchor_args: list[str],
    expected_anchors: int | list[str] | None,
) -> None:
    """Verify --anchor CLI option supports default medoids, integer count, names, and 'all'."""
    corpus_dir, queries_file = sweep_corpus
    code = main(
        [
            "sweep",
            str(corpus_dir),
            "--queries",
            str(queries_file),
            "--scales",
            scales_arg,
            *anchor_args,
            "--agent",
            "fake",
            "--format",
            "json",
        ]
    )
    assert code == 0
    data = json.loads(capsys.readouterr().out)
    if isinstance(expected_anchors, int):
        assert data["anchor_skills"] is not None
        assert len(data["anchor_skills"]) == expected_anchors
    else:
        assert data["anchor_skills"] == expected_anchors


def test_sweep_cli_flags_override_config_file(
    sweep_corpus: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stub_scaling_sweep: list[dict[str, Any]],
) -> None:
    """Verify CLI runtime and registry flags override settings defined in reach.toml."""
    corpus_dir, queries_file = sweep_corpus
    config_file = tmp_path / "reach.toml"
    config_file.write_text(
        """
[runtime]
agent = "keyword"
options = { model = "base-model" }

[registry]
location = "us-central1"
publisher = "base-publisher"
""",
        encoding="utf-8",
    )

    captured_settings = None
    from reach.runtime import build_runtime as orig_build

    def spy_build_runtime(settings: Any) -> Any:
        nonlocal captured_settings
        captured_settings = settings
        return orig_build(settings)

    monkeypatch.setattr("reach.cli.sweep.build_runtime", spy_build_runtime)

    code = main(
        [
            "sweep",
            str(corpus_dir),
            "--config",
            str(config_file),
            "--queries",
            str(queries_file),
            "--scales",
            "2",
            "--agent",
            "fake",
            "--model",
            "override-model",
            "--location",
            "europe-west1",
            "--publisher",
            "cli-publisher",
        ]
    )
    assert code == 0
    assert captured_settings is not None
    assert captured_settings.agent == "fake"
    assert captured_settings.options.get("model") == "override-model"
    assert len(stub_scaling_sweep) == 1
    captured_config = stub_scaling_sweep[0]["config"]
    assert captured_config.registry.location == "europe-west1"
    assert captured_config.registry.publisher == "cli-publisher"


def test_sweep_cli_inherits_agent_from_config_file(
    sweep_corpus: tuple[Path, Path],
    tmp_path: Path,
    stub_scaling_sweep: list[dict[str, Any]],
) -> None:
    """Verify reach sweep inherits agent from reach.toml [general] and [runtime]."""
    corpus_dir, queries_file = sweep_corpus
    config_file = tmp_path / "reach.toml"
    config_file.write_text(
        """
[general]
default_agent = "keyword"

[study]
trusted = true
""",
        encoding="utf-8",
    )

    code = main(
        [
            "sweep",
            str(corpus_dir),
            "--config",
            str(config_file),
            "--queries",
            str(queries_file),
            "--scales",
            "2",
            "--yes",
        ]
    )
    assert code == 0
    assert len(stub_scaling_sweep) == 1
    assert stub_scaling_sweep[0]["config"].runtime.agent == "keyword"
    assert stub_scaling_sweep[0]["runtime"].name == "keyword"


def test_sweep_with_target_skill_path(
    sweep_corpus: tuple[Path, Path],
    capsys,
) -> None:
    """Verify reach sweep accepts a path to a skill directory as --target."""
    corpus_dir, queries_file = sweep_corpus
    skill_dir = corpus_dir / "skill-00"

    code = main(
        [
            "sweep",
            "--target",
            str(skill_dir),
            "--queries",
            str(queries_file),
            "--scales",
            "2",
            "--agent",
            "fake",
            "--format",
            "json",
        ]
    )
    assert code == 0
    data = json.loads(capsys.readouterr().out)
    assert data["target_skill"] == "skill-00"


def test_sweep_auto_discovers_skills_when_reach_toml_omits_skills_path(
    sweep_corpus: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify reach sweep auto-discovers skills when reach.toml exists without study.skills."""
    import shutil

    corpus_dir, queries_file = sweep_corpus
    workspace = tmp_path / "auto_ws"
    skills_dest = workspace / ".agents" / "skills"
    skills_dest.parent.mkdir(parents=True)
    shutil.copytree(corpus_dir, skills_dest)
    (workspace / "reach.toml").write_text('[runtime]\nagent = "keyword"\n', encoding="utf-8")
    monkeypatch.chdir(workspace)

    code = main(
        [
            "sweep",
            "--config",
            str(workspace / "reach.toml"),
            "--queries",
            str(queries_file),
            "--scales",
            "2",
        ]
    )
    assert code == 0


def test_sweep_passes_loaded_skills_once_and_checkpoints_each_scale(
    sweep_corpus: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify _sweep passes skills=found, prints per-scale lines, and checkpoints --out."""
    import reach.cli.sweep as cli_sweep_mod

    corpus_dir, queries_file = sweep_corpus
    out_file = tmp_path / "checkpoints" / "sweep.json"

    captured_kwargs: dict[str, object] = {}
    checkpoints_seen: list[int] = []
    orig_run_scaling_sweep = cli_sweep_mod.run_scaling_sweep

    def spy_run_scaling_sweep(*args: Any, **kwargs: Any) -> ScalingStudy:
        captured_kwargs.update(kwargs)
        orig_cb = kwargs.get("on_scale_complete")

        def wrapped_cb(step, total, point, partial) -> None:
            if orig_cb is not None:
                orig_cb(step, total, point, partial)
            if out_file.exists():
                data = json.loads(out_file.read_text(encoding="utf-8"))
                checkpoints_seen.append(len(data["points"]))

        kwargs["on_scale_complete"] = wrapped_cb
        return orig_run_scaling_sweep(*args, **kwargs)

    monkeypatch.setattr(cli_sweep_mod, "run_scaling_sweep", spy_run_scaling_sweep)

    code = main(
        [
            "sweep",
            str(corpus_dir),
            "--queries",
            str(queries_file),
            "--scales",
            "1,3,6",
            "--agent",
            "fake",
            "--out",
            str(out_file),
        ]
    )
    assert code == 0
    passed_skills = captured_kwargs.get("skills")
    assert isinstance(passed_skills, (list, tuple))
    assert len(passed_skills) == 6
    assert checkpoints_seen == [1, 2, 3]
    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert "[1/3] Scale K=1:" in combined
    assert "[2/3] Scale K=3:" in combined
    assert "[3/3] Scale K=6:" in combined


def test_sweep_cli_fuzzy_suggestions_for_anchor_and_target(
    sweep_corpus: tuple[Path, Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify CLI exits with code 2 and fuzzy suggestions for unknown --anchor or --target."""
    corpus_dir, queries_file = sweep_corpus

    code_anchor = main(
        [
            "sweep",
            str(corpus_dir),
            "--queries",
            str(queries_file),
            "--scales",
            "2,4",
            "--anchor",
            "skill-0",
            "--agent",
            "fake",
        ]
    )
    assert code_anchor == 2
    err_anchor = capsys.readouterr().err
    assert "Did you mean:" in err_anchor
    assert "skill-00" in err_anchor

    code_target = main(
        [
            "sweep",
            str(corpus_dir),
            "--queries",
            str(queries_file),
            "--scales",
            "2,4",
            "--target",
            "skill-0",
            "--agent",
            "fake",
        ]
    )
    assert code_target == 2
    err_target = capsys.readouterr().err
    assert "Did you mean:" in err_target
    assert "skill-00" in err_target


def test_sweep_warns_when_anchor_has_zero_matching_queries(
    sweep_corpus: tuple[Path, Path],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify reach sweep logs anchor coverage and warns when an anchor skill has 0 queries."""
    corpus_dir, _ = sweep_corpus
    partial_queries_file = tmp_path / "partial_queries.json"
    save_query_set(
        QuerySet(
            catalog_id="synthetic",
            queries=(
                Query(
                    query_id="q-0",
                    text="Requesting task number 00",
                    expected_skill="skill-00",
                    kind=QueryKind.IMPLICIT,
                ),
            ),
            provenance=QuerySetProvenance(origin=Origin.AUTHORED),
        ),
        partial_queries_file,
    )

    code = main(
        [
            "sweep",
            str(corpus_dir),
            "--queries",
            str(partial_queries_file),
            "--scales",
            "2,4",
            "--anchor",
            "skill-00,skill-05",
            "--agent",
            "fake",
            "--no-auto-queries",
        ]
    )
    assert code == 0
    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert "Anchor coverage:" in combined
    assert "skill-05" in combined


def test_sweep_cli_allow_truncation_flag(
    sweep_corpus: tuple[Path, Path],
    stub_scaling_sweep: list[dict[str, Any]],
) -> None:
    """Verify reach sweep accepts --allow-truncation and passes it to run_scaling_sweep."""
    corpus_dir, queries_file = sweep_corpus

    # By default, allow_truncation is True
    code = main(
        [
            "sweep",
            str(corpus_dir),
            "--queries",
            str(queries_file),
            "--scales",
            "2",
            "--agent",
            "fake",
        ]
    )
    assert code == 0
    assert stub_scaling_sweep[-1].get("allow_truncation") is True

    # When --no-allow-truncation is passed, allow_truncation is False
    code = main(
        [
            "sweep",
            str(corpus_dir),
            "--queries",
            str(queries_file),
            "--scales",
            "2",
            "--agent",
            "fake",
            "--no-allow-truncation",
        ]
    )
    assert code == 0
    assert not stub_scaling_sweep[-1].get("allow_truncation")


def test_sweep_auto_queries_cold_start_colocated_with_corpus(
    sweep_corpus: tuple[Path, Path],
    generator: FakeGenerator,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify reach sweep drafts anchor queries into <skills>/.reach/queries.json on cold start."""
    from reach.queries import load_query_set

    corpus_dir, _ = sweep_corpus
    elsewhere = corpus_dir.parent / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    colocated_queries = corpus_dir / ".reach" / "queries.json"
    assert not colocated_queries.exists()

    code = main(
        [
            "sweep",
            str(corpus_dir),
            "--scales",
            "2,4",
            "--agent",
            "fake",
        ]
    )
    assert code == 0
    assert colocated_queries.exists()
    saved_qs = load_query_set(colocated_queries)
    # Only the 2 anchor medoid skills for initial scale K=2 should be synthesized
    assert len(saved_qs.covered_skills()) == 2


def test_sweep_auto_queries_backfills_missing_anchor_skills(
    sweep_corpus: tuple[Path, Path],
    generator: FakeGenerator,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify reach sweep backfills missing anchor skills into a partial query set."""
    from reach.queries import load_query_set

    corpus_dir, _ = sweep_corpus
    partial_queries_file = tmp_path / "partial_queries.json"
    save_query_set(
        QuerySet(
            catalog_id="synthetic",
            queries=(
                Query(
                    query_id="q-0",
                    text="Requesting task number 00",
                    expected_skill="skill-00",
                    kind=QueryKind.IMPLICIT,
                ),
                Query(
                    query_id="adv-skill-00-1",
                    text="Near-miss task number 05",
                    expected_skill="skill-05",
                    kind=QueryKind.NEIGHBOR_NEGATIVE,
                ),
            ),
            provenance=QuerySetProvenance(origin=Origin.AUTHORED),
        ),
        partial_queries_file,
    )

    code = main(
        [
            "sweep",
            str(corpus_dir),
            "--queries",
            str(partial_queries_file),
            "--scales",
            "2,4",
            "--anchor",
            "skill-00,skill-05",
            "--agent",
            "fake",
        ]
    )
    assert code == 0
    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert "2/2 anchor skills" in combined

    updated_qs = load_query_set(partial_queries_file)
    assert {"skill-00", "skill-05"}.issubset(updated_qs.covered_skills())


def test_sweep_auto_queries_single_target_cold_start(
    sweep_corpus: tuple[Path, Path],
    generator: FakeGenerator,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify reach sweep --target drafts queries only for the target skill when missing."""
    from reach.queries import load_query_set

    corpus_dir, _ = sweep_corpus
    elsewhere = corpus_dir.parent / "elsewhere_target"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    colocated_queries = corpus_dir / ".reach" / "queries.json"
    assert not colocated_queries.exists()

    code = main(
        [
            "sweep",
            str(corpus_dir),
            "--target",
            "skill-03",
            "--scales",
            "2,4",
            "--agent",
            "fake",
        ]
    )
    assert code == 0
    assert colocated_queries.exists()
    saved_qs = load_query_set(colocated_queries)
    assert saved_qs.covered_skills() == {"skill-03"}


def test_sweep_auto_queries_format_json_suppresses_draft_logs(
    sweep_corpus: tuple[Path, Path],
    generator: FakeGenerator,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify reach sweep --format json suppresses draft logs and produces valid JSON."""
    corpus_dir, _ = sweep_corpus
    out_file = tmp_path / "sweep.json"

    code = main(
        [
            "sweep",
            str(corpus_dir),
            "--scales",
            "2,4",
            "--agent",
            "fake",
            "--format",
            "json",
            "--out",
            str(out_file),
        ]
    )
    assert code == 0
    captured = capsys.readouterr()
    assert "Drafting queries for" not in captured.err
    data = json.loads(captured.out)
    assert "points" in data


def test_sweep_auto_queries_refreshes_stale_anchor_skills(
    sweep_corpus: tuple[Path, Path],
    generator: FakeGenerator,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify reach sweep refreshes anchor skills whose SKILL.md body SHA is stale."""
    from reach.generate import skill_body_digest
    from reach.models import Skill
    from reach.queries import load_query_set

    corpus_dir, _ = sweep_corpus
    skill_00 = Skill(name="skill-00", description="Desc 0", path=corpus_dir / "skill-00")
    fresh_digest_00 = skill_body_digest(skill_00)

    stale_queries_file = tmp_path / "stale_queries.json"
    save_query_set(
        QuerySet(
            catalog_id="synthetic",
            queries=(
                Query(
                    query_id="skill-00-1",
                    text="Keep unchanged query for skill 00",
                    expected_skill="skill-00",
                    kind=QueryKind.IMPLICIT,
                ),
                Query(
                    query_id="skill-05-1",
                    text="Outdated query for skill 05 before body edit",
                    expected_skill="skill-05",
                    kind=QueryKind.IMPLICIT,
                ),
            ),
            provenance=QuerySetProvenance(
                origin=Origin.GENERATED,
                skill_digests={
                    "skill-00": fresh_digest_00,
                    "skill-05": "000000stale0",
                },
            ),
        ),
        stale_queries_file,
    )

    # Edge case: --no-auto-queries warns about the updated body and runs offline
    # without calling the LLM drafter or mutating stale_queries.json on disk.
    code_no_auto = main(
        [
            "sweep",
            str(corpus_dir),
            "--queries",
            str(stale_queries_file),
            "--scales",
            "2,4",
            "--anchor",
            "skill-00,skill-05",
            "--agent",
            "fake",
            "--no-auto-queries",
        ]
    )
    assert code_no_auto == 0
    err_no_auto = capsys.readouterr().err
    assert "have updated bodies since queries were drafted" in err_no_auto
    assert "skill-05" in err_no_auto
    assert "reach query draft --sync" in err_no_auto
    assert load_query_set(stale_queries_file).provenance.skill_digests["skill-05"] == "000000stale0"

    code = main(
        [
            "sweep",
            str(corpus_dir),
            "--queries",
            str(stale_queries_file),
            "--scales",
            "2,4",
            "--anchor",
            "skill-00,skill-05",
            "--agent",
            "fake",
        ]
    )
    assert code == 0
    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert "Refreshing queries for 1 updated anchor skill(s) (skill-05)" in combined

    updated_qs = load_query_set(stale_queries_file)
    # skill-00's original query is preserved, while skill-05's stale query was replaced
    texts = [q.text for q in updated_qs.queries]
    assert "Keep unchanged query for skill 00" in texts
    assert "Outdated query for skill 05 before body edit" not in texts
    skill_05 = Skill(name="skill-05", description="Desc 5", path=corpus_dir / "skill-05")
    assert updated_qs.provenance.skill_digests["skill-05"] == skill_body_digest(skill_05)


@pytest.mark.parametrize(
    ("is_corpus", "target_skill", "expected_cell"),
    [
        pytest.param(True, None, "10 (10 err)", id="corpus-sweep"),
        pytest.param(False, "skill-00", "0/10 (10 err)", id="single-skill-sweep"),
    ],
)
def test_print_sweep_displays_errored_probes_and_warning_banner(
    is_corpus: bool,
    target_skill: str | None,
    expected_cell: str,
    make_scaling_point: Callable[..., Any],
    make_scaling_study: Callable[..., Any],
) -> None:
    """Verify print_sweep renders (N err) in Probes column and prints warning banner."""
    import io

    from rich.console import Console

    from reach.views.sweep import print_sweep

    buf = io.StringIO()
    console = Console(file=buf, width=120, force_terminal=False)
    study = make_scaling_study(
        target_skill=target_skill,
        is_corpus_sweep=is_corpus,
        scales=(10,),
        points=(
            make_scaling_point(
                scale=10,
                catalog_id="sweep:corpus:10",
                pass_rate=0.0,
                pass_rate_interval=(0.0, 0.27),
                recall=0.0,
                precision=1.0,
                f1_score=0.0,
                f1_interval=(0.0, 0.0),
                in_scope_probes=10,
                negative_probes=0,
                probes_executed=10,
                probes_failed=0,
                probes_errored=10,
                duration_ms_mean=150.0,
            ),
        ),
        noise_floor=0.05,
        total_corpus_skills=147,
    )
    print_sweep(console, study)
    rendered = buf.getvalue()
    assert expected_cell in rendered
    assert "Warning: 10 of 10 probe(s) failed due to runtime or agent errors" in rendered


@pytest.mark.parametrize(
    ("explicit_out", "configured_out", "queries_rel", "expected_suffix"),
    [
        ("custom.json", None, None, "custom.json"),
        (None, "configured.json", None, "configured.json"),
        (None, "eval_results.jsonl", ".reach/queries.json", ".reach/sweep.json"),
        (None, None, ".reach/queries.json", ".reach/sweep.json"),
        (None, None, "queries.json", ".reach/sweep.json"),
    ],
)
def test_resolve_sweep_out_and_queries_paths(
    tmp_path: Path,
    explicit_out: str | None,
    configured_out: str | None,
    queries_rel: str | None,
    expected_suffix: str,
) -> None:
    """Verify _resolve_sweep_out_path, _resolve_sweep_queries, and _write_sweep_file."""
    from reach.cli.sweep import (
        _resolve_sweep_out_path,
        _resolve_sweep_queries,
        _write_sweep_file,
    )

    out_arg = tmp_path / explicit_out if explicit_out else None
    cfg_arg = tmp_path / configured_out if configured_out else None
    q_arg = tmp_path / queries_rel if queries_rel else None
    resolved = _resolve_sweep_out_path(out_arg, cfg_arg, q_arg)
    assert str(resolved).endswith(expected_suffix)

    assert _resolve_sweep_queries(
        None,
        None,
        skills_path=None,
        auto_queries=True,
        global_scope=True,
    ) == Path(".reach/queries.json")

    if explicit_out == "custom.json":
        study = _make_stub_study(scales=())
        csv_out = tmp_path / "out.csv"
        _write_sweep_file(study, format="text", out=csv_out)
        assert csv_out.is_file()


def test_print_anchor_coverage_guards_and_truncation_notice(
    sweep_corpus: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify _print_anchor_coverage guard clauses and CatalogFit truncation preflight notice."""
    import io

    from rich.console import Console

    from reach.cli.sweep import _prepare_and_confirm_sweep, _print_anchor_coverage
    from reach.config import RunConfig
    from reach.models import Catalog, Skill
    from reach.runtime import CatalogFit
    from reach.runtime.fake import FakeRuntime

    buf = io.StringIO()
    console = Console(file=buf, width=120, force_terminal=False)
    corpus_dir, queries_file = sweep_corpus
    skills = [Skill(name="skill-00", description="Desc 0", path=corpus_dir / "skill-00")]

    # Guard 1: missing queries file
    _print_anchor_coverage(
        console=console,
        queries_path=tmp_path / "missing.json",
        anchor=None,
        configured_anchor=None,
        skills=skills,
        scales=(1,),
        target=None,
    )
    # Guard 2: corrupt JSON queries file
    bad_q = tmp_path / "bad.json"
    bad_q.write_text("{not-json", encoding="utf-8")
    _print_anchor_coverage(
        console=console,
        queries_path=bad_q,
        anchor=None,
        configured_anchor=None,
        skills=skills,
        scales=(1,),
        target=None,
    )
    # Guard 3: empty resolved anchors (line 216)
    monkeypatch.setattr("reach.cli.sweep._resolve_anchor_skills", lambda *_a, **_kw: ())
    _print_anchor_coverage(
        console=console,
        queries_path=queries_file,
        anchor=None,
        configured_anchor=None,
        skills=skills,
        scales=(1,),
        target=None,
    )

    from typing import override

    # Typed FakeRuntime subclass returning a truncated CatalogFit
    class _RationedFakeRuntime(FakeRuntime):
        @property
        @override
        def rations_catalog(self) -> bool:
            return True

        @override
        def fit(self, catalog: Catalog, skills: object) -> CatalogFit:
            _ = (catalog, skills)
            return CatalogFit(asked=20000, allowed=15000, truncated=2)

    assert (
        _prepare_and_confirm_sweep(
            console=console,
            driver=_RationedFakeRuntime(),
            effective_config=RunConfig(),
            found=skills,
            resolved_queries=queries_file,
            anchor=None,
            scales=(1,),
            target=None,
            allow_truncation=True,
            format="text",
            yes=True,
        )
        == 0
    )
    assert "descriptions will be truncated to bare names" in " ".join(buf.getvalue().split())


def test_draft_missing_sweep_queries_error_unlinks_checkpoint_and_abort_notice(
    sweep_corpus: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify _draft_missing_sweep_queries unlinks checkpoint on failure and sweep abort notice."""
    import io

    from rich.console import Console

    from reach.cli.sweep import _draft_missing_sweep_queries
    from reach.config import RunConfig
    from reach.generate import checkpoint_path
    from reach.models import Skill

    console = Console(file=io.StringIO(), width=120, force_terminal=False)
    corpus_dir, queries_file = sweep_corpus
    skills = [Skill(name="skill-00", description="Desc 0", path=corpus_dir / "skill-00")]

    q_target = tmp_path / "draft_q.json"
    ckpt = checkpoint_path(q_target)
    ckpt.write_text("{}", encoding="utf-8")
    monkeypatch.setattr("reach.cli.sweep._draft_query_set", lambda *_a, **_kw: 2)
    assert (
        _draft_missing_sweep_queries(
            console=console,
            effective_config=RunConfig(),
            found=skills,
            resolved_queries=q_target,
            missing_targets=("skill-00",),
            stale_targets=("skill-01",),
            format="text",
        )
        == 2
    )
    assert not ckpt.exists()

    ckpt.write_text("{}", encoding="utf-8")

    def _raise_draft(*_a: object, **_kw: object) -> int:
        msg = "draft failure"
        raise RuntimeError(msg)

    monkeypatch.setattr("reach.cli.sweep._draft_query_set", _raise_draft)
    assert (
        _draft_missing_sweep_queries(
            console=console,
            effective_config=RunConfig(),
            found=skills,
            resolved_queries=q_target,
            missing_targets=("skill-00",),
            format="text",
        )
        == 2
    )
    assert not ckpt.exists()

    # Exercise _on_scale_complete abort message when all probes fail at step 1 of multi-scale sweep
    def _fake_sweep_abort(**kwargs: object) -> ScalingStudy:
        cb = kwargs.get("on_scale_complete")
        study = _make_stub_study(scales=(1, 2), probes_errored=2)
        if callable(cb):
            cb(1, 2, study.points[0], study)
        return study

    monkeypatch.setattr("reach.cli.sweep.run_scaling_sweep", _fake_sweep_abort)
    assert (
        main(
            [
                "sweep",
                str(corpus_dir),
                "--queries",
                str(queries_file),
                "--scales",
                "1,2",
                "--out",
                str(tmp_path / "abort_sweep.json"),
                "--agent",
                "fake",
                "--yes",
            ]
        )
        == 0
    )
    out = capsys.readouterr()
    assert "Aborting sweep:" in (out.out + out.err)

    captured_kwargs: dict[str, object] = {}

    def _capture_sweep(**kwargs: object) -> ScalingStudy:
        captured_kwargs.update(kwargs)
        return _fake_sweep_abort(**kwargs)

    monkeypatch.setattr("reach.cli.sweep.run_scaling_sweep", _capture_sweep)
    assert (
        main(
            [
                "sweep",
                str(corpus_dir),
                "--queries",
                str(queries_file),
                "--scales",
                "1,2",
                "--agent",
                "fake",
                "--yes",
            ]
        )
        == 0
    )
    assert captured_kwargs.get("noise_floor") is None
    capsys.readouterr()

    monkeypatch.setattr(
        "reach.cli.sweep.run_scaling_sweep",
        lambda **_kw: (_ for _ in ()).throw(RuntimeError("sweep runtime failure")),
    )
    assert (
        main(
            [
                "sweep",
                str(corpus_dir),
                "--queries",
                str(queries_file),
                "--agent",
                "fake",
                "--yes",
            ]
        )
        == 3
    )
    capsys.readouterr()


@pytest.mark.parametrize(
    "target_mode",
    [
        pytest.param("skill_dir", id="positional-skill-dir"),
        pytest.param("skill_md", id="positional-skill-md"),
        pytest.param("bare_name", id="positional-bare-skill-name"),
    ],
)
def test_sweep_positional_target_promotion(
    sweep_corpus: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    stub_scaling_sweep: list[dict[str, Any]],
    target_mode: str,
) -> None:
    """Verify reach sweep promotes positional skill directory, SKILL.md, or bare name to target."""
    corpus_dir, queries_file = sweep_corpus
    monkeypatch.chdir(corpus_dir)

    if target_mode == "skill_dir":
        pos_arg = str(corpus_dir / "skill-00")
    elif target_mode == "skill_md":
        pos_arg = str(corpus_dir / "skill-00" / "SKILL.md")
    else:
        pos_arg = "skill-00"

    args = [
        "sweep",
        pos_arg,
        "--queries",
        str(queries_file),
        "--scales",
        "2",
        "--agent",
        "fake",
    ]

    assert main(args) == 0
    assert len(stub_scaling_sweep) == 1
    assert stub_scaling_sweep[0]["target_skill"] == "skill-00"
    assert len(stub_scaling_sweep[0]["skills"]) == 6


def test_sweep_runtime_and_plan_flags_happy_and_sad_path(
    sweep_corpus: tuple[Path, Path],
    stub_scaling_sweep: list[dict[str, Any]],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify reach sweep accepts RuntimeFlags and PlanFlags and rejects unknown -O options."""
    corpus_dir, queries_file = sweep_corpus

    code = main(
        [
            "sweep",
            str(corpus_dir),
            "--queries",
            str(queries_file),
            "--scales",
            "2",
            "--agent",
            "fake",
            "-O",
            "model=opt-model",
            "--timeout",
            "95",
            "--max-turns",
            "2",
            "--no-early-exit",
            "-n",
            "4",
            "--retries",
            "1",
            "--backoff",
            "2.5",
            "--pause",
            "0.5",
            "-j",
            "2",
        ]
    )
    assert code == 0
    call = stub_scaling_sweep[-1]
    cfg = call["config"]
    assert cfg.runtime.timeout_s == 95
    assert cfg.runtime.max_turns == 2
    assert cfg.runtime.early_exit is False
    assert cfg.runtime.options.get("model") == "opt-model"
    assert call["attempts"] == 4
    assert call["workers"] == 2
    assert cfg.plan.retries == 1
    assert cfg.plan.backoff_s == 2.5
    assert cfg.plan.pause_s == 0.5

    # Sad path: unrecognized -O option for the selected agent exits with code 2
    sad_code = main(
        [
            "sweep",
            str(corpus_dir),
            "--queries",
            str(queries_file),
            "--scales",
            "2",
            "--agent",
            "keyword",
            "-O",
            "nonexistent_option_xyz=1",
        ]
    )
    assert sad_code == 2
    err = capsys.readouterr().err
    assert "nonexistent_option_xyz" in err


def test_sweep_existing_partial_query_set_does_not_draft_or_mutate_on_corpus_sweep(
    sweep_corpus: tuple[Path, Path],
    generator: FakeGenerator,
    tmp_path: Path,
) -> None:
    """Verify corpus sweep with an existing non-empty query set reuses it without LLM drafting."""
    corpus_dir, _ = sweep_corpus
    partial_queries_file = tmp_path / "partial_existing.json"
    # Cover only 4 of the 6 skills so at least 1 medoid would have been unqueried if unclamped
    queries = tuple(
        Query(
            query_id=f"q-{i}",
            text=f"Requesting task number {i:02d}",
            expected_skill=f"skill-{i:02d}",
            kind=QueryKind.IMPLICIT,
        )
        for i in range(4)
    )
    save_query_set(
        QuerySet(
            catalog_id="synthetic",
            queries=queries,
            provenance=QuerySetProvenance(origin=Origin.AUTHORED),
        ),
        partial_queries_file,
    )
    original_bytes = partial_queries_file.read_bytes()

    out_file = tmp_path / "sweep_out.json"
    code = main(
        [
            "sweep",
            str(corpus_dir),
            "--queries",
            str(partial_queries_file),
            "--scales",
            "2,6",
            "--agent",
            "fake",
            "--out",
            str(out_file),
        ]
    )
    assert code == 0
    assert generator.completions == 0
    assert partial_queries_file.read_bytes() == original_bytes
    study_data = json.loads(out_file.read_text(encoding="utf-8"))
    assert study_data["total_corpus_skills"] == 6
    covered = {"skill-00", "skill-01", "skill-02", "skill-03"}
    assert set(study_data["anchor_skills"]).issubset(covered)


def test_sweep_auto_discovers_local_reach_toml_and_guards_jsonl_out(
    sweep_corpus: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stub_scaling_sweep: list[dict[str, Any]],
) -> None:
    """Verify reach sweep auto-discovers ./reach.toml, keeps attempts=1, and ignores .jsonl out."""
    corpus_dir, queries_file = sweep_corpus
    workspace = tmp_path / "ws_toml"
    workspace.mkdir()
    eval_jsonl = workspace / ".reach" / "eval_results.jsonl"
    eval_jsonl.parent.mkdir(parents=True)
    eval_jsonl.write_text('{"probe": 1}\n', encoding="utf-8")

    (workspace / "reach.toml").write_text(
        f"""
[runtime]
agent = "fake"
timeout_s = 175

[study]
skills = "{corpus_dir.as_posix()}"
queries = "{queries_file.as_posix()}"
out = "{eval_jsonl.as_posix()}"
trusted = true

[registry]
project = "ignored-when-local-skills-configured"
""",
        encoding="utf-8",
    )
    monkeypatch.chdir(workspace)

    assert main(["sweep", "--scales", "2"]) == 0
    assert len(stub_scaling_sweep) == 1
    call = stub_scaling_sweep[0]
    assert call["config"].runtime.timeout_s == 175
    # Unset [plan].attempts in reach.toml must still default to 1 for sweep
    assert call["attempts"] is None
    assert call["config"].plan.resolve_sweep_attempts(call["attempts"]) == 1
    # .jsonl configured out must not be overwritten; .reach/sweep.json is written instead
    assert eval_jsonl.read_text(encoding="utf-8") == '{"probe": 1}\n'
    assert (workspace / ".reach" / "sweep.json").is_file()
