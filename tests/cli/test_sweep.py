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
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from reach.cli import main
from reach.models import Query, QueryKind
from reach.queries import Origin, QuerySet, QuerySetProvenance, save_query_set
from reach.runtime.fake import FakeGenerator
from reach.sweep import ScalingStudy

if TYPE_CHECKING:
    pass


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
            id=f"q-{i}",
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
            "--no-early-stop",
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
    """Verify reach sweep --format json outputs valid ScalingStudy JSON."""
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
            "--no-early-stop",
            "--format",
            "json",
        ]
    )
    assert exit_code == 0
    out = capsys.readouterr().out
    data = json.loads(out)
    assert data["is_corpus_sweep"] is True
    assert "points" in data
    # resolve_sweep_scales appends total skills (6) if not already included
    assert len(data["points"]) == 3
    assert data["points"][0]["scale"] == 1
    assert data["points"][-1]["scale"] == 6
    assert data["points"][0]["abstention_rate"] is None
    assert data["points"][0]["prompt_tokens_mean"] is None


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
            "--no-early-stop",
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


def test_sweep_discovers_default_queries(
    sweep_corpus: tuple[Path, Path],
    monkeypatch,
    capsys,
) -> None:
    """Verify reach sweep automatically discovers .reach/queries.json when --queries is omitted."""
    corpus_dir, queries_file = sweep_corpus
    workspace = corpus_dir.parent
    reach_dir = workspace / ".reach"
    reach_dir.mkdir(exist_ok=True)
    (reach_dir / "queries.json").write_text(queries_file.read_text())
    monkeypatch.chdir(workspace)
    exit_code = main(["sweep", str(corpus_dir), "--scales", "1,3", "--agent", "fake"])
    assert exit_code == 0
    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert "Corpus Capacity Scaling:" in combined or "Corpus Scaling" in combined


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
            id=f"q-{i}",
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
    assert len(data["points"]) == 2  # scale 1 and scale 6 (total skills)
    # Target skill has 1 query, with attempts=3, 3 probes should be executed
    assert data["points"][0]["probes_executed"] == 3


def test_sweep_early_stop_flag(
    sweep_corpus: tuple[Path, Path],
    capsys,
) -> None:
    """Verify --no-early-stop flag parsing in CLI."""
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
            "--no-early-stop",
            "--format",
            "json",
        ]
    )
    assert exit_code == 0
    out = capsys.readouterr().out
    data = json.loads(out)
    assert data["points"][0]["negative_probes"] == 0


def test_sweep_out_flag_writes_file(
    sweep_corpus: tuple[Path, Path],
    tmp_path: Path,
    capsys,
) -> None:
    """Verify --out flag writes formatted sweep results to disk and prints notification."""
    corpus_dir, queries_file = sweep_corpus
    out_json = tmp_path / "results" / "sweep_out.json"

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
            "--no-early-stop",
            "--out",
            str(out_json),
        ]
    )
    assert exit_code == 0
    assert out_json.exists()
    saved = json.loads(out_json.read_text(encoding="utf-8"))
    assert "points" in saved
    assert len(saved["points"]) == 3

    # Verify console notification
    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert "wrote" in combined
    assert str(out_json) in combined


def test_sweep_default_out_writes_to_reach_dir(
    sweep_corpus: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    """Verify omitting --out flag defaults to writing sweep.json beside discovered queries."""
    corpus_dir, queries_file = sweep_corpus
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    reach_dir = workspace / ".reach"
    reach_dir.mkdir()
    (reach_dir / "queries.json").write_text(queries_file.read_text())
    monkeypatch.chdir(workspace)

    exit_code = main(
        [
            "sweep",
            str(corpus_dir),
            "--scales",
            "1,3",
            "--agent",
            "fake",
            "--no-early-stop",
        ]
    )
    assert exit_code == 0
    expected_out = reach_dir / "sweep.json"
    assert expected_out.exists()
    saved = json.loads(expected_out.read_text(encoding="utf-8"))
    assert "points" in saved

    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert "wrote" in combined
    assert ".reach/sweep.json" in combined


def test_sweep_anchor_cli_modes(
    sweep_corpus: tuple[Path, Path],
    capsys,
) -> None:
    """Verify --anchor CLI option supports default medoids, integer count, names, and 'all'."""
    corpus_dir, queries_file = sweep_corpus

    # 1. Default anchor (medoids of scale[0] = 2)
    code = main(
        [
            "sweep",
            str(corpus_dir),
            "--queries",
            str(queries_file),
            "--scales",
            "2,4",
            "--agent",
            "fake",
            "--no-early-stop",
            "--format",
            "json",
        ]
    )
    assert code == 0
    data_default = json.loads(capsys.readouterr().out)
    assert data_default["anchor_skills"] is not None
    assert len(data_default["anchor_skills"]) == 2

    # 2. Integer cohort count (--anchor 3)
    code = main(
        [
            "sweep",
            str(corpus_dir),
            "--queries",
            str(queries_file),
            "--scales",
            "3,6",
            "--anchor",
            "3",
            "--agent",
            "fake",
            "--no-early-stop",
            "--format",
            "json",
        ]
    )
    assert code == 0
    data_int = json.loads(capsys.readouterr().out)
    assert len(data_int["anchor_skills"]) == 3

    # 3. Explicit skill names
    code = main(
        [
            "sweep",
            str(corpus_dir),
            "--queries",
            str(queries_file),
            "--scales",
            "2,4",
            "--anchor",
            "skill-00,skill-02",
            "--agent",
            "fake",
            "--no-early-stop",
            "--format",
            "json",
        ]
    )
    assert code == 0
    data_names = json.loads(capsys.readouterr().out)
    assert data_names["anchor_skills"] == ["skill-00", "skill-02"]

    # 4. --anchor all (full dynamic expansion)
    code = main(
        [
            "sweep",
            str(corpus_dir),
            "--queries",
            str(queries_file),
            "--scales",
            "2,4",
            "--anchor",
            "all",
            "--agent",
            "fake",
            "--no-early-stop",
            "--format",
            "json",
        ]
    )
    assert code == 0
    data_all = json.loads(capsys.readouterr().out)
    assert data_all["anchor_skills"] is None


def test_sweep_accepts_model_flag(
    sweep_corpus: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify reach sweep accepts --model option and applies it to scaling study."""
    corpus_dir, queries_file = sweep_corpus
    captured_settings = None
    from reach.runtime import build_runtime as orig_build

    def spy_build_runtime(settings):
        nonlocal captured_settings
        captured_settings = settings
        return orig_build(settings)

    monkeypatch.setattr("reach.cli.sweep.build_runtime", spy_build_runtime)

    code = main(
        [
            "sweep",
            str(corpus_dir),
            "--queries",
            str(queries_file),
            "--scales",
            "2,4",
            "--agent",
            "fake",
            "--model",
            "custom-sweep-model",
            "--no-early-stop",
        ]
    )
    assert code == 0
    assert captured_settings is not None
    assert captured_settings.options.get("model") == "custom-sweep-model"


def test_sweep_cli_flags_override_config_file(
    sweep_corpus: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify CLI runtime flags override settings defined in reach.toml."""
    corpus_dir, queries_file = sweep_corpus
    config_file = tmp_path / "reach.toml"
    config_file.write_text(
        """
[runtime]
agent = "keyword"
options = { model = "base-model" }
""",
        encoding="utf-8",
    )

    captured_settings = None
    from reach.runtime import build_runtime as orig_build

    def spy_build_runtime(settings):
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
            "2,4",
            "--agent",
            "fake",
            "--model",
            "override-model",
            "--no-early-stop",
        ]
    )
    assert code == 0
    assert captured_settings is not None
    assert captured_settings.agent == "fake"
    assert captured_settings.options.get("model") == "override-model"


def test_sweep_cli_inherits_agent_from_config_file(
    sweep_corpus: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
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

    captured_config = None
    captured_runtime = None

    def fake_sweep(config=None, runtime=None, **kwargs) -> ScalingStudy:
        nonlocal captured_config, captured_runtime
        captured_config = config
        captured_runtime = runtime
        from reach.sweep import ScalingPoint, ScalingStudy

        return ScalingStudy(
            scales=(2,),
            points=(
                ScalingPoint(
                    scale=2,
                    catalog_id="test",
                    pass_rate=1.0,
                    pass_rate_interval=(1.0, 1.0),
                    delta_vs_baseline=0.0,
                    delta_context=0.0,
                    delta_shadowing=0.0,
                    probes_executed=1,
                ),
            ),
            baseline_pass_rate=1.0,
            final_pass_rate=1.0,
            total_delta=0.0,
            total_context_loss=0.0,
            total_shadowing_loss=0.0,
            is_corpus_sweep=True,
        )

    monkeypatch.setattr("reach.cli.sweep.run_scaling_sweep", fake_sweep)

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
            "--no-early-stop",
        ]
    )
    assert code == 0
    assert captured_config is not None
    assert captured_config.runtime.agent == "keyword"
    assert captured_runtime is not None
    assert captured_runtime.name == "keyword"


def test_sweep_cli_registry_flags_override_config_file(
    sweep_corpus: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify CLI registry flags override settings defined in reach.toml."""
    corpus_dir, queries_file = sweep_corpus
    config_file = tmp_path / "reach.toml"
    config_file.write_text(
        """
[registry]
location = "us-central1"
publisher = "base-publisher"
""",
        encoding="utf-8",
    )

    captured_config = None

    def fake_sweep(config=None, **kwargs) -> ScalingStudy:
        nonlocal captured_config
        captured_config = config
        from reach.sweep import ScalingPoint, ScalingStudy

        return ScalingStudy(
            scales=(2,),
            points=(
                ScalingPoint(
                    scale=2,
                    catalog_id="test",
                    pass_rate=1.0,
                    pass_rate_interval=(1.0, 1.0),
                    delta_vs_baseline=0.0,
                    delta_context=0.0,
                    delta_shadowing=0.0,
                    probes_executed=1,
                ),
            ),
            baseline_pass_rate=1.0,
            final_pass_rate=1.0,
            total_delta=0.0,
            total_context_loss=0.0,
            total_shadowing_loss=0.0,
            is_corpus_sweep=True,
        )

    monkeypatch.setattr("reach.cli.sweep.run_scaling_sweep", fake_sweep)

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
            "--location",
            "europe-west1",
            "--publisher",
            "cli-publisher",
            "--no-early-stop",
        ]
    )
    assert code == 0
    assert captured_config is not None
    assert captured_config.registry.location == "europe-west1"
    assert captured_config.registry.publisher == "cli-publisher"


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
            "--no-early-stop",
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
            "--no-early-stop",
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

    def spy_run_scaling_sweep(*args, **kwargs):
        captured_kwargs.update(kwargs)
        orig_cb = kwargs.get("on_scale_complete")

        def wrapped_cb(step, total, point, partial):
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
            "--no-early-stop",
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
                    id="q-0",
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
            "--no-early-stop",
        ]
    )
    assert code == 0
    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert "Anchor coverage:" in combined
    assert "skill-05" in combined


def test_sweep_cli_allow_truncation_flag(
    sweep_corpus: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify reach sweep accepts --allow-truncation and passes it to run_scaling_sweep."""
    import reach.cli.sweep as cli_sweep

    passed_kwargs: dict[str, object] = {}

    def _mock_run_scaling_sweep(*args, **kwargs):
        passed_kwargs.update(kwargs)
        from reach.sweep import ScalingPoint, ScalingStudy

        p = ScalingPoint(
            scale=2,
            catalog_id="cat",
            pass_rate=1.0,
            pass_rate_interval=(0.8, 1.0),
            delta_vs_baseline=0.0,
            delta_context=0.0,
            delta_shadowing=0.0,
            probes_executed=2,
        )
        return ScalingStudy(
            is_corpus_sweep=True,
            scales=(2,),
            points=(p,),
            baseline_pass_rate=1.0,
            final_pass_rate=1.0,
            total_delta=0.0,
            total_context_loss=0.0,
            total_shadowing_loss=0.0,
        )

    monkeypatch.setattr(cli_sweep, "run_scaling_sweep", _mock_run_scaling_sweep)
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
            "--no-early-stop",
        ]
    )
    assert code == 0
    assert passed_kwargs.get("allow_truncation") is True

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
            "--no-early-stop",
        ]
    )
    assert code == 0
    assert passed_kwargs.get("allow_truncation") is False


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
            "--no-early-stop",
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
                    id="q-0",
                    text="Requesting task number 00",
                    expected_skill="skill-00",
                    kind=QueryKind.IMPLICIT,
                ),
                Query(
                    id="adv-skill-00-1",
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
            "--no-early-stop",
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
            "--no-early-stop",
        ]
    )
    assert code == 0
    assert colocated_queries.exists()
    saved_qs = load_query_set(colocated_queries)
    assert saved_qs.covered_skills() == {"skill-03"}
