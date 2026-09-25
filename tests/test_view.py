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

"""Verify CLI view command for inspection of saved Artifact files."""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Never

import pytest
from selectolax.parser import HTMLParser

from reach.artifact import Artifact, write_artifact
from reach.cli import main
from reach.models import CatalogMode, ProbeResult, Query, QueryKind
from reach.queries import Origin, QuerySet, QuerySetProvenance
from reach.run import Composition
from reach.view import VIEW_RENDERERS, render_view, render_view_html
from reach.views import badge

if TYPE_CHECKING:
    from pathlib import Path

    from reach.sweep import ScalingStudy


@pytest.fixture
def recorded(artifact: Artifact, tmp_path: Path) -> Path:
    """Write test Artifact file to disk."""
    return write_artifact(artifact, tmp_path / "run.artifact.json")


def test_a_recorded_run_reads_back_as_the_scorecard_it_printed(recorded: Path, capsys) -> None:
    """Verify view displays scorecard summary containing skill names and metrics."""
    assert main(["view", str(recorded)]) == 0

    shown = capsys.readouterr().err
    assert "recall" in shown.split()
    assert "gcs-lifecycle-rules" in shown
    assert "gke-basics" in shown


def test_the_queries_are_not_listed_until_they_are_asked_for(
    recorded: Path,
    wide: None,
    capsys,
) -> None:
    """Verify individual queries are displayed only when --show-queries flag is passed."""
    assert main(["view", str(recorded)]) == 0
    assert "q-lifecycle" not in capsys.readouterr().err

    assert main(["view", str(recorded), "--show-queries"]) == 0
    assert "q-lifecycle" in capsys.readouterr().err


def test_the_listing_says_which_question_missed_and_what_it_reached_instead(
    recorded: Path,
    wide: None,
    capsys,
) -> None:
    """Verify query listing displays ground truth skill, hit ratio, and misrouted selection."""
    assert main(["view", str(recorded), "--show-queries"]) == 0

    err = capsys.readouterr().err
    assert "gcs-retention-policy" in err
    row = next(line for line in err.splitlines() if "q-retention" in line)
    assert "1/3" in row
    assert "gke-basics" in row


def test_the_rank_on_a_row_is_shown_over_the_field_it_was_taken_in(
    artifact: Artifact,
    tmp_path: Path,
    wide: None,
    capsys,
) -> None:
    """Verify query listing renders difficulty rank over catalog size."""
    widened = artifact.model_copy(update={"catalog_size": 111})
    banked = write_artifact(widened, tmp_path / "wide.artifact.json")

    assert main(["view", str(banked), "--show-queries"]) == 0

    assert "3/111" in capsys.readouterr().err


def test_reading_a_run_back_asks_nothing_of_a_runtime(
    recorded: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    """Verify view reads and renders artifacts without invoking runtime constructors."""

    def refuse(*args: object, **kwargs: object) -> Never:
        msg = "view built a runtime to read a file"
        raise AssertionError(msg)

    monkeypatch.setattr("reach.runtime.build_runtime", refuse)
    assert main(["view", str(recorded)]) == 0
    assert "recall" in capsys.readouterr().err.split()


def test_verbose_puts_the_hex_back_beside_the_badges_it_replaced(
    recorded: Path,
    capsys,
) -> None:
    """Verify --verbose outputs both proquint badge and raw hexadecimal digest."""
    assert main(["view", str(recorded), "--verbose"]) == 0

    stamp = capsys.readouterr().err.rpartition("[arm ")[2]
    said, hexed = stamp.split()[:2]
    assert badge(hexed) == said
    assert said != hexed


def test_format_html_prints_a_self_contained_report_instead_of_the_scorecard(
    recorded: Path,
    capsys,
) -> None:
    """Verify view --format html prints standalone HTML document to stdout."""
    assert main(["view", str(recorded), "--format", "html"]) == 0
    out = capsys.readouterr().out
    assert out.lstrip().startswith("<!DOCTYPE html>")
    tree = HTMLParser(out)
    assert tree.css_first("html") is not None
    skill_cells = [td.text().strip() for td in tree.css("#skills-table tbody td:first-child")]
    assert "gcs-lifecycle-rules" in skill_cells


def test_view_writes_to_out_file(
    recorded: Path,
    tmp_path: Path,
    capsys,
) -> None:
    """Verify view --out writes report directly to target file."""
    report_file = tmp_path / "report.html"
    assert main(["view", str(recorded), "--format", "html", "--out", str(report_file)]) == 0
    assert report_file.is_file()
    assert report_file.read_text(encoding="utf-8").lstrip().startswith("<!DOCTYPE html>")
    assert "wrote" in capsys.readouterr().err


def test_view_format_jsonl(
    recorded: Path,
    capsys,
) -> None:
    """Verify view --format jsonl prints query records as JSON lines."""
    assert main(["view", str(recorded), "--format", "jsonl"]) == 0
    lines = [line for line in capsys.readouterr().out.strip().splitlines() if line]
    assert len(lines) > 0
    parsed = json.loads(lines[0])
    assert "query_id" in parsed


def test_a_file_that_is_not_an_artifact_is_named_back_rather_than_dumped(
    tmp_path: Path,
    capsys,
) -> None:
    """Verify view rejects non-artifact JSON files with informative error."""
    rows = tmp_path / "results.jsonl"
    rows.write_text('{"query_id": "q-retention", "invoked": null}\n', encoding="utf-8")

    assert main(["view", str(rows)]) == 2

    complaint = capsys.readouterr().err
    unwrapped = re.sub(r"\s*│\s*\n\s*│\s*", "", complaint)
    assert "results.jsonl" in unwrapped
    assert ".artifact.json" in unwrapped


def test_an_artifact_that_is_not_there_is_refused_rather_than_traced(
    tmp_path: Path,
    capsys,
) -> None:
    """Verify view exits cleanly with error when target file does not exist."""
    assert main(["view", str(tmp_path / "absent.artifact.json")]) == 2
    err = capsys.readouterr().err
    assert "No evaluation artifact found" in err
    assert "reach eval" in err


def test_view_defaults_to_reach_eval_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact: Artifact,
    capsys,
) -> None:
    """Verify reach view reads .reach/eval.json by default when no path is given."""
    monkeypatch.chdir(tmp_path)
    reach_dir = tmp_path / ".reach"
    reach_dir.mkdir()
    (reach_dir / "eval.json").write_text(artifact.model_dump_json(), encoding="utf-8")

    assert main(["view"]) == 0
    assert "recall" in capsys.readouterr().err.lower()


def test_view_without_artifact_and_no_default_gives_clear_guidance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    """Verify reach view explains how to generate an artifact when .reach/eval.json is absent."""
    monkeypatch.chdir(tmp_path)
    assert main(["view"]) == 2
    clean = " ".join(capsys.readouterr().err.split())
    assert ".reach/eval.json" in clean
    assert "reach eval" in clean


def test_the_document_is_one_self_contained_file(
    artifact: Artifact, external_reference_re: re.Pattern[str]
) -> None:
    """Verify render_view_html generates self-contained HTML without external assets."""
    out = render_view_html(artifact)
    assert out.lstrip().startswith("<!DOCTYPE html>")
    tree = HTMLParser(out)
    assert tree.css_first("html") is not None
    assert tree.css_first("style") is not None
    assert tree.css_first("script") is not None
    assert not external_reference_re.search(out)


def test_render_view_dispatches_by_format_name(artifact: Artifact) -> None:
    """Verify render_view correctly routes 'html' format to render_view_html."""
    assert {"html": render_view_html} == VIEW_RENDERERS
    assert render_view(artifact, "html") == render_view_html(artifact)


def test_an_unknown_format_names_the_ones_that_exist(artifact: Artifact) -> None:
    """Verify render_view raises ValueError on unsupported output format."""
    with pytest.raises(ValueError, match="unknown format 'pdf'; expected one of html"):
        render_view(artifact, "pdf")


def test_the_confusion_matrix_shows_the_split_querys_misroute(
    artifact: Artifact,
) -> None:
    """Verify confusion matrix renders misrouted cell with query text hover title."""
    tree = HTMLParser(render_view_html(artifact))
    miss_cell = tree.css_first("#confusion-table td.miss")
    assert miss_cell is not None
    assert miss_cell.text().strip() == "1"
    assert "Keep audit logs for seven years for compliance." in (
        miss_cell.attributes.get("title") or ""
    )


def test_confusion_matrix_and_collisions_show_reasoning_when_present(
    artifact: Artifact,
) -> None:
    """Verify HTML view renders thought traces in tooltip and collisions table."""
    from reach.artifact import NO_SKILL, SampleQuery

    collision_pair = next(p for p in artifact.confusion if p.invoked not in (p.expected, NO_SKILL))
    updated_pair = collision_pair.model_copy(
        update={
            "queries": (
                SampleQuery(
                    query_id="q-1",
                    text="Keep audit logs for seven years for compliance.",
                    probes=1,
                    reasoning=("Thinking about bucket retention rules.",),
                ),
            ),
        },
    )
    test_artifact = artifact.model_copy(update={"confusion": (updated_pair,)})
    tree = HTMLParser(render_view_html(test_artifact))
    assert "thought: Thinking about bucket retention rules." in tree.text()
    assert tree.css_first("#collisions-table") is not None


def test_abstention_is_a_column_of_its_own(artifact: Artifact) -> None:
    """Verify confusion matrix includes explicit '(no skill)' column for abstentions."""
    tree = HTMLParser(render_view_html(artifact))
    headers = [th.text().strip() for th in tree.css("#confusion-table thead th")]
    assert "(no skill)" in headers


def test_an_unprobed_artifact_says_so_rather_than_rendering_an_empty_table(
    artifact: Artifact,
) -> None:
    """Verify HTML rendering outputs fallback message when artifact has no recorded probes."""
    empty = artifact.model_copy(update={"confusion": (), "queries": ()})
    tree = HTMLParser(render_view_html(empty))
    text = tree.text()
    assert "No probes were recorded." in text
    assert "No queries were recorded." in text


def test_the_skills_table_is_filterable(artifact: Artifact) -> None:
    """Verify HTML output includes interactive JavaScript filter function and skills table."""
    tree = HTMLParser(render_view_html(artifact))
    assert tree.css_first("input[oninput*=\"reachFilterRows(this, 'skills-table')\"]") is not None
    table = tree.css_first("#skills-table")
    assert table is not None
    rendered_skills = [td.text().strip() for td in table.css("tbody td:first-child")]
    for skill in artifact.skills:
        assert skill.skill in rendered_skills


def test_queries_are_expandable_and_carry_their_full_text(
    artifact: Artifact,
) -> None:
    """Verify queries are rendered as expandable HTML details elements containing full text."""
    tree = HTMLParser(render_view_html(artifact))
    details = tree.css("details.query")
    assert len(details) == len(artifact.queries)
    preview_texts = [node.text().strip() for node in tree.css("details.query summary .query-text")]
    assert "Keep audit logs for seven years for compliance." in preview_texts
    assert "Tier old objects to Coldline after 30 days." in preview_texts


def test_a_clean_query_and_a_split_one_are_styled_apart(artifact: Artifact) -> None:
    """Verify HTML styling distinguishes fully hit queries from misrouted queries."""
    tree = HTMLParser(render_view_html(artifact))
    assert tree.css_first("details.query.hit") is not None
    assert tree.css_first("details.query.miss, details.query.error") is not None


def test_query_text_is_escaped_not_executed(corpus, whole_catalog, make_config) -> None:
    """Verify query text containing HTML/JS tags is properly escaped in rendered HTML."""
    hostile = Query(
        id="q-hostile",
        text='<script>alert(1)</script> & "quoted"',
        kind=QueryKind.IMPLICIT,
        expected_skill="gcs-lifecycle-rules",
    )
    query_set = QuerySet(
        catalog_id=whole_catalog.id,
        queries=(hostile,),
        provenance=QuerySetProvenance(origin=Origin.AUTHORED),
    )
    results = [
        ProbeResult(
            query_id="q-hostile",
            catalog_id=whole_catalog.id,
            catalog_mode=whole_catalog.mode,
            catalog_size=whole_catalog.size,
            model="fake-model",
            runtime="fake",
            attempt=1,
            invoked_skills=("gcs-lifecycle-rules",),
        ),
    ]
    built = Artifact.assemble(
        Composition(
            config=make_config(catalog={"mode": CatalogMode.ALL}),
            query_set=query_set,
            catalog=whole_catalog,
            skills=tuple(corpus),
        ),
        results,
    )
    out = render_view_html(built)
    tree = HTMLParser(out)
    # Ensure no executable script tag was injected into DOM
    assert all("alert(1)" not in (s.text() or "") for s in tree.css("script"))
    # Ensure safely rendered text in DOM
    preview = tree.css_first("details.query summary .query-text")
    assert preview is not None
    assert '<script>alert(1)</script> & "quoted"' in preview.text()
    # Ensure raw output escaped dangerous characters
    assert "<script>alert(1)</script>" not in out
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in out


def test_the_queries_container_is_filterable(artifact: Artifact) -> None:
    """Verify HTML output includes interactive query filter function and container."""
    tree = HTMLParser(render_view_html(artifact))
    filter_selector = "input[oninput*=\"reachFilterQueries(this, 'queries-container')\"]"
    assert tree.css_first(filter_selector) is not None
    assert tree.css_first("#queries-container") is not None


def test_view_open_flag_launches_browser(recorded: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify reach view --open renders HTML and triggers webbrowser.open."""
    opened_urls: list[str] = []
    monkeypatch.setattr("webbrowser.open", opened_urls.append)

    assert main(["view", str(recorded), "--open"]) == 0
    assert len(opened_urls) == 1
    assert opened_urls[0].startswith("file://")
    assert opened_urls[0].endswith(".html")


def test_header_consistency_interval_formatting_includes_parentheses(
    artifact: Artifact,
) -> None:
    """Verify consistency interval is rendered with spacing and dim parentheses."""
    tree = HTMLParser(render_view_html(artifact))
    dim_spans = [span.text().strip() for span in tree.css(".figures dd span.dim")]
    assert any(s.startswith("(") and s.endswith(")") for s in dim_spans)


def test_skills_table_renders_dash_for_missing_and_explicit_absorbed_zero(
    artifact: Artifact,
) -> None:
    """Verify skills table formats missing metrics as dim dashes and absorbed count as 0."""
    from reach.artifact import SkillScore

    mock_skill = SkillScore(
        skill="untested-skill",
        recall=None,
        precision=None,
        probes=0,
        reached=0,
        absorbed=0,
    )
    art = artifact.model_copy(update={"skills": (mock_skill,)})
    tree = HTMLParser(render_view_html(art))
    row = tree.css_first("#skills-table tbody tr")
    assert row is not None
    dash = row.css_first("td span.dim")
    assert dash is not None
    assert dash.text().strip() == "—"
    cell_texts = [td.text().strip() for td in row.css("td")]
    assert "0" in cell_texts


def test_query_summary_contains_query_text_preview(artifact: Artifact) -> None:
    """Verify query summary accordion contains query text preview element."""
    tree = HTMLParser(render_view_html(artifact))
    query_texts = tree.css("details.query summary span.query-text")
    assert len(query_texts) > 0


def test_queries_support_expand_and_collapse_all(artifact: Artifact) -> None:
    """Verify view HTML contains expand all and collapse all toggles and script function."""
    tree = HTMLParser(render_view_html(artifact))
    assert tree.css_first('button[onclick*="reachToggleAllQueries(true)"]') is not None
    assert tree.css_first('button[onclick*="reachToggleAllQueries(false)"]') is not None
    script = tree.css_first("script")
    assert script is not None
    assert "function reachToggleAllQueries" in script.text()


def test_queries_render_exact_data_attributes_and_exact_js_matching(artifact: Artifact) -> None:
    """Verify details.query elements expose data attributes and JS uses exact array matching."""
    tree = HTMLParser(render_view_html(artifact))
    details = tree.css("details.query")
    assert len(details) > 0
    for item in details:
        assert "data-expected" in item.attributes
        assert "data-selections" in item.attributes
    script = tree.css_first("script")
    assert script is not None
    js_text = script.text()
    assert "getSelectedSkills(q).includes(" in js_text
    assert "q.textContent.includes(" not in js_text
    assert "td.textContent.trim() === ''" in js_text


def test_collisions_html_renders_individual_query_probe_count(artifact: Artifact) -> None:
    """Verify _collisions_html renders q.probes for each sample query row instead of pair.probes."""
    from reach.artifact import NO_SKILL, SampleQuery

    collision_pair = next(p for p in artifact.confusion if p.invoked not in (p.expected, NO_SKILL))
    updated_pair = collision_pair.model_copy(
        update={
            "probes": 5,
            "collisions": 5,
            "queries": (
                SampleQuery(query_id="q-1", text="First query", probes=2),
                SampleQuery(query_id="q-2", text="Second query", probes=3),
            ),
        },
    )
    test_artifact = artifact.model_copy(update={"confusion": (updated_pair,)})
    tree = HTMLParser(render_view_html(test_artifact))
    rows = tree.css("#collisions-table tbody tr")
    assert len(rows) == 2
    assert rows[0].css("td")[2].text().strip() == "2"
    assert rows[1].css("td")[2].text().strip() == "3"


def test_view_scaling_study_text_output(
    tmp_path: Path, sample_scaling_study: ScalingStudy, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify reach view renders terminal scorecard from a sweep ScalingStudy JSON artifact."""
    sweep_path = tmp_path / "sweep.json"
    sweep_path.write_text(sample_scaling_study.model_dump_json(), encoding="utf-8")

    assert main(["view", str(sweep_path)]) == 0
    captured = capsys.readouterr()
    assert "Corpus Capacity Scaling" in (captured.out + captured.err)


def test_view_scaling_study_json_format(
    tmp_path: Path, sample_scaling_study: ScalingStudy, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify reach view --format json prints valid JSON for a sweep artifact."""
    sweep_path = tmp_path / "sweep.json"
    sweep_path.write_text(sample_scaling_study.model_dump_json(), encoding="utf-8")

    assert main(["view", str(sweep_path), "--format", "json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["is_corpus_sweep"] is True
    assert data["scales"] == [10, 25]


def test_view_scaling_study_csv_format(
    tmp_path: Path, sample_scaling_study: ScalingStudy, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify reach view --format csv prints formatted CSV for a sweep artifact."""
    sweep_path = tmp_path / "sweep.json"
    sweep_path.write_text(sample_scaling_study.model_dump_json(), encoding="utf-8")

    assert main(["view", str(sweep_path), "--format", "csv"]) == 0
    csv_out = capsys.readouterr().out
    assert "scale" in csv_out
    assert "recall" in csv_out


def test_view_scaling_study_out_file(tmp_path: Path, sample_scaling_study: ScalingStudy) -> None:
    """Verify reach view --out writes sweep study output to target file."""
    sweep_path = tmp_path / "sweep.json"
    sweep_path.write_text(sample_scaling_study.model_dump_json(), encoding="utf-8")
    out_file = tmp_path / "exported.csv"

    assert main(["view", str(sweep_path), "--out", str(out_file)]) == 0
    assert out_file.is_file()
    assert "scale" in out_file.read_text(encoding="utf-8")


def test_view_scaling_study_rejects_html_and_slice(
    tmp_path: Path, sample_scaling_study: ScalingStudy, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify reach view raises ValueError when HTML or slicing flags are passed with sweep."""
    sweep_path = tmp_path / "sweep.json"
    sweep_path.write_text(sample_scaling_study.model_dump_json(), encoding="utf-8")

    assert main(["view", str(sweep_path), "--format", "html"]) == 2
    assert "HTML format is not supported for sweep artifacts" in capsys.readouterr().err

    assert main(["view", str(sweep_path), "--filter-skill", "skill-a"]) == 2
    assert "Slicing flags" in capsys.readouterr().err


def test_view_defaults_to_sweep_json_when_eval_absent(
    tmp_path: Path,
    sample_scaling_study: ScalingStudy,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify reach view falls back to .reach/sweep.json when .reach/eval.json is absent."""
    monkeypatch.chdir(tmp_path)
    reach_dir = tmp_path / ".reach"
    reach_dir.mkdir()
    (reach_dir / "sweep.json").write_text(sample_scaling_study.model_dump_json(), encoding="utf-8")

    assert main(["view"]) == 0
    captured = capsys.readouterr()
    assert "Corpus Capacity Scaling" in (captured.out + captured.err)


def test_view_eval_artifact_rejects_csv_format(
    recorded: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify reach view rejects --format csv when viewing an evaluation artifact."""
    assert main(["view", str(recorded), "--format", "csv"]) == 2
    assert "CSV format is only supported for sweep artifacts" in capsys.readouterr().err


def test_view_rejects_unparseable_artifact_with_dual_schema_guidance(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify reach view emits descriptive error referencing both eval and sweep schemas."""
    bad_file = tmp_path / "corrupt.json"
    bad_file.write_text('{"unrecognized_key": 42}', encoding="utf-8")

    assert main(["view", str(bad_file)]) == 2
    err = capsys.readouterr().err
    assert "Cannot read artifact" in err
    assert "neither a valid evaluation artifact" in err
    assert "nor a valid scaling sweep study" in err
