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

"""Verify run report compilation, metrics aggregation, provenance, and format renderers."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import pytest

from reach.artifact import Artifact
from reach.config import CatalogSettings, PlanSettings, RunConfig, RuntimeSettings
from reach.difficulty import LexicalRank
from reach.models import Catalog, CatalogMode, ProbeResult, Query, QueryKind, Skill
from reach.queries import Origin, QuerySet, QuerySetProvenance
from reach.rendering import csv_document
from reach.report import (
    build_report,
    format_pairs,
    format_run,
    per_query_lines,
    render,
    render_csv,
    render_json,
    render_text,
)
from reach.run import Composition


def probe(
    query_id: str,
    invoked: str | None,
    *,
    error: str | None = None,
    attempt: int = 1,
    resolved_model: str = "",
) -> ProbeResult:
    """Build a scored probe result for report rendering."""
    return ProbeResult(
        query_id=query_id,
        catalog_id="test-cat",
        catalog_mode=CatalogMode.ALL,
        catalog_size=2,
        model="sonnet",
        resolved_model=resolved_model,
        runtime="fake",
        attempt=attempt,
        invoked_skills=(invoked,) if invoked else (),
        error=error,
        cost_usd=0.10,
    )


@pytest.fixture
def composition(queries: list[Query]) -> Composition:
    """Build test Composition fixture."""
    catalog = Catalog(
        id="test-cat",
        mode=CatalogMode.ALL,
        skills=("gcs-lifecycle-rules", "gcs-retention-policy"),
    )
    skills = [
        Skill(name=name, description=f"Skill {name}", path=Path(f"/skills/{name}"))
        for name in catalog.skills
    ]
    query_set = QuerySet(
        catalog_id="test-cat",
        queries=tuple(queries),
        provenance=QuerySetProvenance(origin=Origin.AUTHORED),
    )
    config = RunConfig(
        catalog=CatalogSettings(mode=CatalogMode.ALL),
        runtime=RuntimeSettings(agent="fake", options={"model": "sonnet"}),
        plan=PlanSettings(attempts=1),
    )
    return Composition(
        config=config,
        query_set=query_set,
        catalog=catalog,
        skills=tuple(skills),
    )


@pytest.fixture
def report(composition: Composition) -> Artifact:
    """Build test Artifact fixture with sample queries and probe results."""
    results = [
        probe("q-lifecycle", "gcs-lifecycle-rules"),
        probe("q-retention", "gcs-lifecycle-rules"),
    ]
    return build_report(composition, results)


def test_provenance_is_recovered_from_the_results(
    report: Artifact,
    composition: Composition,
) -> None:
    """Verify build_report derives runtime metadata directly from ProbeResult entries."""
    assert report.provenance.runtime == "fake"
    assert report.provenance.model == "sonnet"
    assert report.catalog_id == "test-cat"
    assert report.digests.config_fingerprint == composition.config.fingerprint


def test_explicit_provenance_wins_over_the_results(composition: Composition) -> None:
    """Verify Composition config provides provenance when ProbeResult lacks runtime."""
    cfg = composition.config.model_copy(
        update={
            "runtime": RuntimeSettings(agent="claude-code", options={"model": "sonnet"}),
        },
    )
    cat = composition.catalog.model_copy(update={"id": "neighborhood:gcs-lifecycle-rules"})
    custom = Composition(
        config=cfg,
        catalog=cat,
        query_set=composition.query_set,
        skills=composition.skills,
    )
    row = probe("q-lifecycle", "gcs-lifecycle-rules").model_copy(update={"runtime": ""})
    built = build_report(
        custom,
        [row],
    )
    assert built.catalog_id == "neighborhood:gcs-lifecycle-rules"
    assert built.provenance.runtime == "claude-code"


def test_spend_counts_errored_probes(composition: Composition) -> None:
    """Verify total cost computation sums spend across errored probe attempts."""
    results = [
        probe("q-lifecycle", "gcs-lifecycle-rules"),
        probe("q-lifecycle", None, error="timeout", attempt=2),
    ]
    assert build_report(composition, results).spend_usd == pytest.approx(0.20)


def test_errored_probes_leave_the_accuracy_denominator(composition: Composition) -> None:
    """Verify errored probes are omitted from accuracy metric calculations."""
    results = [
        probe("q-lifecycle", "gcs-lifecycle-rules"),
        probe("q-lifecycle", None, error="timeout", attempt=2),
    ]
    built = build_report(composition, results)
    assert built.errors == 1
    assert built.scores.top1_accuracy == pytest.approx(1.0)


def test_a_result_with_no_ground_truth_is_fatal(composition: Composition) -> None:
    """Verify build_report raises KeyError when results reference unknown query IDs."""
    results = [probe("q-lifecycle", "gcs-lifecycle-rules"), probe("q-ghost", "x")]
    with pytest.raises(KeyError, match="q-ghost"):
        build_report(composition, results)


def test_unclean_queries_are_identifiable(composition: Composition) -> None:
    """Verify unclean list captures queries with non-matching attempts."""
    results = [
        probe("q-lifecycle", "gcs-lifecycle-rules"),
        probe("q-lifecycle", None, attempt=2),
    ]
    assert [o.query_id for o in build_report(composition, results).unclean] == [
        "q-lifecycle",
    ]


def test_an_empty_run_reports_rather_than_raises(composition: Composition) -> None:
    """Verify empty result lists produce empty reports."""
    built = build_report(composition, [])
    assert built.probes == 0
    assert all(q.probes == 0 for q in built.queries)


def test_text_shows_the_accuracy(composition: Composition) -> None:
    """Verify text rendering displays top-1 accuracy rate and matches format_run."""
    results = [
        probe("q-lifecycle", "gcs-lifecycle-rules"),
        probe("q-retention", "gcs-lifecycle-rules"),
    ]
    text = render_text(build_report(composition, results))
    assert "top-1 accuracy" in text
    assert "50.0%" in text
    assert format_run(composition, results) == text


def test_collisions_name_the_offending_pair(composition: Composition) -> None:
    """Verify collisions list captures expected and misrouted skill pairs with counts."""
    built = build_report(composition, [probe("q-lifecycle", "gcs-retention-policy")])
    collisions = [(p.expected, p.invoked, p.collisions) for p in built.confusion if p.collisions]
    assert collisions == [("gcs-lifecycle-rules", "gcs-retention-policy", 1)]
    text = format_pairs(collisions, "Collisions")
    assert "gcs-lifecycle-rules -> gcs-retention-policy  x1" in text


@pytest.mark.parametrize(
    ("invoked", "expected_marks"),
    [
        (["gcs-lifecycle-rules"] * 2, ("  q-lifecycle", "2/2")),
        (["gcs-lifecycle-rules", None], ("! q-lifecycle", "1/2")),
    ],
    ids=["clean", "non-selection"],
)
def test_per_query_flags_unclean_rows(
    composition: Composition,
    invoked: list[str | None],
    expected_marks: tuple[str, str],
) -> None:
    """Verify per_query_lines marks unclean query attempts with exclamation points."""
    results = [probe("q-lifecycle", pick, attempt=i) for i, pick in enumerate(invoked, start=1)]
    line = per_query_lines(build_report(composition, results))[0]
    assert all(mark in line for mark in expected_marks)


def test_text_states_the_provenance(report: Artifact, composition: Composition) -> None:
    """Verify render_text formats runtime and configuration fingerprint header fields."""
    text = render_text(report)
    assert "runtime           fake" in text
    assert f"config            {composition.config.fingerprint}" in text


def test_text_names_the_resolved_model_beside_the_alias(composition: Composition) -> None:
    """Verify render_text outputs model alias alongside resolved concrete model identifier."""
    built = build_report(
        composition,
        [probe("q-lifecycle", "gcs-lifecycle-rules", resolved_model="claude-opus-5")],
    )
    assert "model             sonnet  -> claude-opus-5" in render_text(built)


def test_text_says_only_the_alias_when_nothing_resolved_it(report: Artifact) -> None:
    """Verify render_text displays alias without mapping arrow when resolved_model is empty."""
    line = next(line for line in render_text(report).splitlines() if "model  " in line)
    assert line.strip() == "model             sonnet"


def test_a_resumed_run_finds_the_resolved_model_in_whichever_row_names_it(
    composition: Composition,
) -> None:
    """Verify build_report retrieves resolved_model from later rows if initial row lacks it."""
    built = build_report(
        composition,
        [
            probe("q-lifecycle", "gcs-lifecycle-rules"),
            probe("q-retention", "gcs-lifecycle-rules", resolved_model="claude-opus-5"),
        ],
    )
    assert built.provenance.resolved_model == "claude-opus-5"


def test_text_reports_reused_probes(composition: Composition) -> None:
    """Verify render_text displays reused prior probe counts in header output."""
    built = build_report(
        composition,
        [probe("q-lifecycle", "gcs-lifecycle-rules")],
        reused=5,
    )
    assert "reused            5 prior probes" in render_text(built)


def test_class_column_widens_for_a_long_label(composition: Composition) -> None:
    """Verify text rendering dynamically formats width to fit long skill names."""
    long_name = "gcs-" + "x" * 60
    results = [probe("q-lifecycle", long_name)]
    assert long_name in render_text(build_report(composition, results))


def test_json_round_trips(report: Artifact) -> None:
    """Verify render_json serializes Artifact validatable by Pydantic."""
    payload = json.loads(render_json(report))
    assert payload["probes"] == 2
    assert Artifact.model_validate(payload).catalog_id == report.catalog_id


def test_csv_has_one_row_per_query(report: Artifact) -> None:
    """Verify render_csv outputs structured CSV row entries per query outcome."""
    rows = list(csv.DictReader(io.StringIO(render_csv(report))))
    assert [r["query_id"] for r in rows] == ["q-lifecycle", "q-retention"]
    assert rows[0]["clean"] == "yes"


def test_difficulty_is_absent_when_no_corpus_supplied(report: Artifact) -> None:
    """Verify lexical rank columns are empty when difficulty mapping is omitted."""
    assert [q.difficulty_rank for q in report.queries] == [None, None]
    rows = list(csv.DictReader(io.StringIO(render_csv(report))))
    assert [r["lexical_rank"] for r in rows] == ["", ""]
    assert [r["lexical_field"] for r in rows] == ["", ""]


def test_difficulty_reaches_the_per_query_rows(composition: Composition) -> None:
    """Verify lexical rank is rendered into text and CSV outputs."""
    built = build_report(
        composition,
        [probe("q-lifecycle", "gcs-lifecycle-rules")],
        difficulty={"q-lifecycle": LexicalRank(position=7, field_size=20)},
    )
    assert built.queries[0].difficulty_rank == 7
    assert "L7" in per_query_lines(built)[0]
    rows = list(csv.DictReader(io.StringIO(render_csv(built))))
    assert rows[0]["lexical_rank"] == "7"


def test_unknown_format_names_the_alternatives(report: Artifact) -> None:
    """Verify render raises ValueError listing supported formats when given unrecognized format."""
    with pytest.raises(ValueError, match="csv, json, jsonl, text"):
        render(report, "xml")


def test_csv_document_formatting() -> None:
    """Verify csv_document serializes headers, rows, LF newlines, empty bodies, and quotes."""
    document = csv_document(["a", "b"], [[1, 2], [3, 4]])
    assert "\r" not in document
    assert document == "a,b\n1,2\n3,4\n"
    assert csv_document(["a", "b"], []) == "a,b\n"
    assert csv_document(["note"], [["a, b"]]) == 'note\n"a, b"\n'


def test_trajectory_section_renders_when_reachability_is_zero(composition: Composition) -> None:
    """Verify trajectory metrics render in text report when scored > 0 even at 0% reachability."""
    built = build_report(composition, [probe("q-lifecycle", None)])
    text = render_text(built)
    assert "entrypoint acc." in text
    assert "reachability" in text


def test_trajectory_section_renders_na_when_metrics_are_none(
    whole_catalog,
    corpus,
    make_config,
    make_result,
) -> None:
    """Verify step efficiency and skill F1 render as n/a when None (no in-scope queries)."""
    oos_query = Query(
        id="q-oos",
        text="Out of scope query",
        kind=QueryKind.OUT_OF_SCOPE,
        expected_skill=None,
    )
    query_set = QuerySet(
        queries=(oos_query,),
        provenance=QuerySetProvenance(origin=Origin.GENERATED, generator="test"),
    )
    results = [make_result("q-oos", None)]
    config = make_config(catalog={"mode": CatalogMode.ALL})
    built = Artifact.assemble(
        Composition(
            config=config,
            query_set=query_set,
            catalog=whole_catalog,
            skills=tuple(corpus),
        ),
        results,
    )
    text = render_text(built)
    assert "step efficiency      n/a" in text
    assert "skill F1             n/a" in text
