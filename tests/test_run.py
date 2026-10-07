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

"""Verify study composition, execution invariants, coverage validation, and listing budgets."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest
from pydantic import ValidationError

from reach.catalog import corpus_digest, load_skills
from reach.models import Catalog, CatalogMode, ProbeResult
from reach.queries import Origin, QuerySet, QuerySetProvenance, load_query_set
from reach.run import (
    Plan,
    ProbeHarness,
    compose,
    conduct,
    evaluate,
    load_results,
    plan_only,
    read_sidecar,
    sidecar_path,
    validate_catalog_fit,
    validate_query_coverage,
    write_results,
    write_sidecar,
)
from reach.runtime import CatalogFit, SelectionOutcome
from reach.runtime.fake import FakeRuntime

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


def test_coverage_reports_unqueried_skills(
    write_queries: Callable[..., Path],
) -> None:
    """Verify validate_query_coverage fails if catalog contains skills with no queries."""
    rows = [{"id": "q", "text": "a", "kind": "implicit", "expected_skill": "s1"}]
    query_set = load_query_set(write_queries(queries=rows, catalog_id="c"))
    with pytest.raises(ValueError, match="skills with no query"):
        validate_query_coverage(query_set, ["s1", "s2"])


def test_partial_allows_a_set_targeting_one_boundary(
    write_queries: Callable[..., Path],
) -> None:
    """Verify partial=True permits query sets covering only a subset of catalog skills."""
    rows = [{"id": "q", "text": "a", "kind": "implicit", "expected_skill": "s1"}]
    query_set = load_query_set(write_queries(queries=rows, catalog_id="c"))
    validate_query_coverage(query_set, ["s1", "s2"], partial=True)


@pytest.mark.parametrize("partial", [False, True])
def test_coverage_reports_stray_ground_truth(
    write_queries: Callable[..., Path],
    partial: bool,
) -> None:
    """Verify validate_query_coverage raises ValueError if query expects a non-resident skill."""
    rows = [{"id": "q", "text": "a", "kind": "implicit", "expected_skill": "ghost"}]
    query_set = load_query_set(write_queries(queries=rows, catalog_id="c"))
    with pytest.raises(ValueError, match="outside the catalog"):
        validate_query_coverage(query_set, ["s1"], partial=partial)


def test_out_of_scope_queries_cover_nothing(
    write_queries: Callable[..., Path],
) -> None:
    """Verify OUT_OF_SCOPE queries do not satisfy skill coverage requirements."""
    rows = [
        {"id": "q1", "text": "a", "kind": "implicit", "expected_skill": "s1"},
        {"id": "q2", "text": "b", "kind": "out_of_scope"},
    ]
    query_set = load_query_set(write_queries(queries=rows, catalog_id="c"))
    validate_query_coverage(query_set, ["s1"])


@pytest.fixture
def cramped() -> CatalogFit:
    """Provide a CatalogFit fixture indicating skill listing budget overflow."""
    return CatalogFit(
        allowed=30,
        asked=91,
        unit="display columns",
        truncated=2,
        remedy="set skill_listing_budget_fraction = 0.001",
    )


def test_a_catalog_the_model_would_half_see_is_refused(cramped, whole_catalog, corpus) -> None:
    """Verify validate_catalog_fit raises ValueError when skills exceed listing budget."""
    runtime = FakeRuntime(fit=cramped)
    with pytest.raises(ValueError, match="2 of 3 skills") as raised:
        validate_catalog_fit(runtime, whole_catalog, corpus)
    message = str(raised.value)
    assert "allows 30 display columns" in message
    assert "asks 91" in message
    assert "skill_listing_budget_fraction = 0.001, or pass --allow-truncation" in message


def test_a_refusal_the_agent_has_no_fix_for_is_still_a_sentence(
    whole_catalog,
    corpus,
) -> None:
    """Verify validate_catalog_fit message handles agents with no remedy configuration."""
    runtime = FakeRuntime(fit=CatalogFit(allowed=10, asked=99, truncated=1))
    with pytest.raises(ValueError, match="Pass --allow-truncation"):
        validate_catalog_fit(runtime, whole_catalog, corpus)


def test_a_catalog_the_runtime_will_show_whole_passes_and_reports(
    whole_catalog,
    corpus,
) -> None:
    """Verify validate_catalog_fit succeeds and returns fit object when catalog fits."""
    fit = validate_catalog_fit(FakeRuntime(), whole_catalog, corpus)
    assert fit.whole
    assert not fit.rations


def test_allowing_truncation_reports_instead_of_refusing(cramped, whole_catalog, corpus) -> None:
    """Verify allow_truncation=True bypasses refusal and returns cramped fit."""
    fit = validate_catalog_fit(
        FakeRuntime(fit=cramped),
        whole_catalog,
        corpus,
        allow_truncation=True,
    )
    assert fit == cramped


def test_a_run_is_refused_before_it_installs_anything(make_config, queries, cramped) -> None:
    """Verify listing budget overflow halts execution before workspace installation."""
    runtime = FakeRuntime({q.text: q.expected_skill for q in queries}, fit=cramped)
    with pytest.raises(ValueError, match="bare name"):
        conduct(make_config(), runtime)
    assert runtime.installs == []
    assert runtime.queries == []


def test_a_run_may_be_told_to_measure_the_catalog_as_shown(
    make_config,
    queries,
    cramped,
) -> None:
    """Verify allow_truncation=True proceeds with run execution."""
    runtime = FakeRuntime({q.text: q.expected_skill for q in queries}, fit=cramped)
    report = conduct(make_config(), runtime, allow_truncation=True).report
    assert report.probes == 2
    assert len(runtime.installs) == 1


def test_a_run_records_what_the_runtime_said_it_would_show(
    make_config,
    answering_runtime,
) -> None:
    """Verify conduct captures runtime CatalogFit outcome."""
    outcome = conduct(make_config(), answering_runtime)
    assert outcome.fit is not None
    assert outcome.fit.whole
    assert answering_runtime.fittings == [outcome.catalog]


def test_compose_resolves_the_catalog_the_query_set_names(make_config) -> None:
    """Verify compose infers catalog identifier from query set when not specified."""
    composed = compose(make_config())
    assert composed.catalog.id == "neighborhood:gcs-lifecycle-rules"
    assert len(composed.skills) == 3


def test_compose_explicit_auto_catalog_resolves_to_query_set_catalog(make_config) -> None:
    """Verify explicit catalog = 'auto' adopts query set catalog without requiring rescope."""
    composed = compose(make_config(study={"catalog": "auto"}))
    assert composed.catalog.id == "neighborhood:gcs-lifecycle-rules"
    assert len(composed.skills) == 3


def test_an_explicit_catalog_id_wins(make_config) -> None:
    """Verify explicit catalog in study configuration overrides query set catalog."""
    config = make_config(
        study={"catalog": "neighborhood:gke-basics", "rescope": True},
    )
    assert compose(config).catalog.id == "neighborhood:gke-basics"


def test_reprobing_a_set_in_another_catalog_is_refused(make_config) -> None:
    """Verify compose raises ValueError if evaluated in mismatched catalog without rescope."""
    config = make_config(study={"catalog": "neighborhood:gke-basics"})
    with pytest.raises(ValueError, match="not valid in another"):
        compose(config)


def test_a_missing_catalog_names_the_alternatives(make_config) -> None:
    """Verify compose KeyError lists available catalog IDs when target catalog is absent."""
    config = make_config(study={"catalog": "neighborhood:ghost", "rescope": True})
    with pytest.raises(KeyError, match="neighborhood:gke-basics"):
        compose(config)


def test_a_configuration_naming_no_corpus_says_which_verb_can_find_one(
    make_config,
) -> None:
    """Verify compose raises ValueError if corpus is omitted without preloaded skills."""
    with pytest.raises(ValueError, match="no skill corpus"):
        compose(make_config(study={"skills": None}))


def test_a_corpus_the_caller_already_loaded_is_composed_without_a_path(
    make_config,
    corpus,
) -> None:
    """Verify preloaded skills can be passed to compose directly."""
    composed = compose(make_config(study={"skills": None}), corpus)
    assert composed.catalog.id == "neighborhood:gcs-lifecycle-rules"
    assert len(composed.skills) == 3


def test_plan_only_is_a_real_rehearsal(make_config) -> None:
    """Verify plan_only validates catalog, probe counts, and corpus digest."""
    config = make_config(plan={"attempts": 3})
    plan = plan_only(config)
    assert plan.catalog_id == "neighborhood:gcs-lifecycle-rules"
    assert (plan.catalog_size, plan.probes) == (3, 6)
    assert plan.corpus_digest == corpus_digest(load_skills(config.study.skills))


def test_the_plan_prices_the_resolution_the_depth_buys(make_config) -> None:
    """Verify per_query_resolution is computed from configured attempts."""
    assert plan_only(make_config(plan={"attempts": 5})).per_query_resolution == (
        pytest.approx(0.886, abs=0.001)
    )
    assert plan_only(make_config(plan={"attempts": 20})).per_query_resolution == (
        pytest.approx(0.443, abs=0.001)
    )


def test_a_plan_that_predates_the_recorded_depth_prices_nothing(make_config) -> None:
    """Verify per_query_resolution is None when attempts is 0."""
    plan = plan_only(make_config(plan={"attempts": 3}))
    assert plan.model_copy(update={"attempts": 0}).per_query_resolution is None


def test_plan_only_catches_a_mismatched_query_set(make_config) -> None:
    """Verify plan_only raises ValueError if ground truth expects skill outside catalog."""
    config = make_config(
        study={
            "partial": False,
            "catalog": "neighborhood:gke-basics",
            "rescope": True,
        },
    )
    config = config.with_overrides(catalog={"size": 2, "rivals": 1})
    with pytest.raises(ValueError, match="outside the catalog"):
        plan_only(config)


def test_a_run_probes_every_query_and_reports(make_config, answering_runtime) -> None:
    """Verify conduct executes all queries and generates scorecards."""
    report = conduct(make_config(plan={"attempts": 2}), answering_runtime).report
    assert report.probes == 4
    assert report.scores.top1_accuracy == pytest.approx(1.0)
    assert report.provenance.runtime == "fake"
    assert len(answering_runtime.queries) == 4


def test_workers_reaches_conduct_without_changing_the_fingerprint(
    make_config,
    answering_runtime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify workers in PlanSettings reaches ProbeHarness without altering fingerprint."""
    import reach.run as run_mod

    observed_workers: list[int] = []
    orig_harness = run_mod.ProbeHarness

    def spy_harness(*args: Any, **kwargs: Any) -> ProbeHarness:
        observed_workers.append(kwargs.get("workers", 1))
        return orig_harness(*args, **kwargs)

    monkeypatch.setattr(run_mod, "ProbeHarness", spy_harness)

    config_default = make_config(plan={"attempts": 2})
    config_parallel = make_config(plan={"attempts": 2, "workers": 4})
    assert config_parallel.fingerprint == config_default.fingerprint
    assert config_parallel.arm == config_default.arm
    assert config_parallel.condition == config_default.condition

    report = conduct(config_parallel, answering_runtime).report
    assert observed_workers[-1] == 4
    assert report.probes == 4
    assert report.scores.top1_accuracy == pytest.approx(1.0)

    conduct(config_parallel, answering_runtime, workers=2)
    assert observed_workers[-1] == 2


def test_a_run_labels_every_query_with_its_lexical_difficulty(
    make_config,
    answering_runtime,
) -> None:
    """Verify conduct assigns BM25 lexical difficulty ranks to all queries."""
    report = conduct(make_config(), answering_runtime).report
    ranks = {q.query_id: q.difficulty_rank for q in report.queries}
    assert all(r is not None and 1 <= r <= 3 for r in ranks.values())


def test_the_report_carries_the_configuration_that_produced_it(
    make_config,
    answering_runtime,
) -> None:
    """Verify report preserves configuration fingerprint and corpus digest."""
    config = make_config()
    report = conduct(config, answering_runtime).report
    expected = corpus_digest(load_skills(config.study.skills))
    assert report.digests.config_fingerprint == config.fingerprint
    assert report.digests.corpus_digest == expected
    assert report.catalog_id == "neighborhood:gcs-lifecycle-rules"


def test_editing_a_description_makes_a_run_a_different_subject(
    make_config,
    answering_runtime,
    skill_repo,
) -> None:
    """Verify corpus_digest changes when a skill description is modified."""
    config = make_config()
    before = conduct(config, answering_runtime).report
    target = skill_repo / "storage" / "gcs-lifecycle-rules" / "SKILL.md"
    target.write_text(
        target.read_text(encoding="utf-8").replace(
            "Configures object lifecycle rules.",
            "Tiers and expires objects.",
        ),
        encoding="utf-8",
    )
    after = conduct(config, answering_runtime, resume=False).report
    assert after.digests.config_fingerprint == before.digests.config_fingerprint
    assert after.digests.corpus_digest != before.digests.corpus_digest


def test_editing_ground_truth_makes_a_run_a_different_measurement(
    make_config,
    answering_runtime,
    query_file,
) -> None:
    """Verify queries_digest changes when query ground truth is modified."""
    config = make_config()
    before = conduct(config, answering_runtime).report
    loaded = load_query_set(query_file)
    query_file.write_text(
        loaded.model_copy(
            update={
                "queries": (
                    loaded.queries[0].model_copy(
                        update={"expected_skill": "gke-basics"},
                    ),
                    *loaded.queries[1:],
                ),
            },
        ).model_dump_json(),
        encoding="utf-8",
    )
    after = conduct(config, answering_runtime, resume=False).report
    assert after.digests.config_fingerprint == before.digests.config_fingerprint
    assert after.digests.queries_digest != before.digests.queries_digest


def test_the_catalog_is_installed_before_any_probe(make_config, answering_runtime) -> None:
    """Verify runtime.install is called before queries are evaluated."""
    config = make_config()
    conduct(config, answering_runtime)
    (catalog, workdir), *rest = answering_runtime.installs
    assert rest == []
    assert catalog.id == "neighborhood:gcs-lifecycle-rules"
    assert workdir == config.study.workdir.resolve()


def test_a_drifted_install_is_caught_not_scored(make_config, queries) -> None:
    """Verify installation discrepancies result in probe errors instead of scores."""
    runtime = FakeRuntime(
        {
            q.text: SelectionOutcome(
                invoked_skills=(q.expected_skill,) if q.expected_skill else (),
                observed_catalog=("gke-basics",),
            )
            for q in queries
        },
    )
    report = conduct(make_config(plan={"retries": 0}), runtime).report
    assert report.errors == report.probes
    assert report.scores.scored == 0


def test_a_run_without_an_out_path_still_reports(make_config, answering_runtime) -> None:
    """Verify conduct completes and reports metrics when out path is None."""
    assert conduct(make_config(), answering_runtime).report.probes == 2


def test_results_are_written_where_the_study_says(
    make_config,
    answering_runtime,
    tmp_path,
) -> None:
    """Verify probe rows are appended to configured results file."""
    out = tmp_path / "nested" / "results.jsonl"
    conduct(make_config(study={"out": out}), answering_runtime)
    assert len(out.read_text(encoding="utf-8").strip().splitlines()) == 2


def test_progress_is_reported_per_probe(make_config, answering_runtime) -> None:
    """Verify progress callback is invoked for each probe."""
    seen: list[tuple[int, int, str]] = []
    conduct(
        make_config(plan={"attempts": 2}),
        answering_runtime,
        progress=lambda i, total, r: seen.append((i, total, r.query_id)),
    )
    assert [(i, total) for i, total, _ in seen] == [(1, 4), (2, 4), (3, 4), (4, 4)]


# --- Resuming prior evaluation runs ------------------------------------------


def test_a_resumed_run_does_not_pay_for_its_own_history(
    make_config,
    answering_runtime,
    tmp_path,
) -> None:
    """Verify resuming a study reuses existing probe rows without re-issuing queries."""
    out = tmp_path / "results.jsonl"
    config = make_config(study={"out": out})
    conduct(config, answering_runtime)

    second = FakeRuntime(
        dict.fromkeys(answering_runtime.queries, "gcs-lifecycle-rules"),
    )
    report = conduct(config, second).report
    assert second.queries == []
    assert report.reused == 2
    assert report.probes == 2


def test_resume_can_be_refused(make_config, answering_runtime, tmp_path) -> None:
    """Verify resume=False re-executes all probes regardless of existing output file."""
    out = tmp_path / "results.jsonl"
    config = make_config(study={"out": out})
    conduct(config, answering_runtime)
    conduct(config, answering_runtime, resume=False)
    assert len(answering_runtime.queries) == 4


def test_a_resumed_run_reports_on_the_whole_set(
    make_config,
    answering_runtime,
    queries,
    tmp_path,
) -> None:
    """Verify resumed run combines prior probe results with newly executed probes."""
    out = tmp_path / "results.jsonl"
    config = make_config(study={"out": out}, plan={"attempts": 1, "retries": 0})
    half = FakeRuntime(
        {
            queries[0].text: queries[0].expected_skill,
            queries[1].text: SelectionOutcome(error="timeout"),
        },
    )
    conduct(config, half)

    report = conduct(config, answering_runtime).report
    assert answering_runtime.queries == [queries[1].text]
    assert {o.query_id for o in report.queries} == {"q-lifecycle", "q-retention"}
    assert report.scores.scored == 2


def test_resumed_run_deduplicates_errored_probes_in_outcome_and_disk(
    make_config,
    answering_runtime,
    queries,
    tmp_path,
) -> None:
    """Verify resuming replaces prior errored attempts without duplicates in memory or disk."""
    out = tmp_path / "results.jsonl"
    config = make_config(study={"out": out}, plan={"attempts": 1, "retries": 0})
    half = FakeRuntime(
        {
            queries[0].text: queries[0].expected_skill,
            queries[1].text: SelectionOutcome(error="timeout"),
        },
    )
    # First run: 1 success, 1 failure
    conduct(config, half)
    assert len(load_results(out)) == 2

    # Second run: retry query 1 with answering runtime
    outcome = conduct(config, answering_runtime)
    assert outcome.reused == 1
    assert len(outcome.results) == 2
    assert len({(r.query_id, r.attempt) for r in outcome.results}) == 2

    q1_result = next(r for r in outcome.results if r.query_id == queries[1].query_id)
    assert q1_result.error is None
    assert q1_result.invoked_skill == queries[1].expected_skill

    # Disk file must be compacted without duplicate/stale error records
    disk_rows = load_results(out)
    assert len(disk_rows) == 2
    assert all(r.error is None for r in disk_rows)

    report = outcome.report
    assert report.probes == 2
    assert report.errors == 0


def test_topping_up_a_recorded_run_to_a_deeper_plan_pays_only_for_the_tail(
    make_config,
    answering_runtime,
    tmp_path,
) -> None:
    """Verify deepening attempt count executes only additional attempts and appends rows."""
    out = tmp_path / "results.jsonl"
    conduct(
        make_config(study={"out": out}, plan={"attempts": 1}),
        answering_runtime,
    )
    answering_runtime.queries.clear()

    report = conduct(
        make_config(study={"out": out}, plan={"attempts": 3}),
        answering_runtime,
    ).report
    assert len(answering_runtime.queries) == 4
    assert len(load_results(out)) == 6
    assert report.probes == 6


def test_the_shallow_rows_a_deepened_file_keeps_still_name_their_own_depth(
    make_config,
    answering_runtime,
    tmp_path,
) -> None:
    """Verify earlier rows in deepened results file preserve original configuration fingerprints."""
    out = tmp_path / "results.jsonl"
    shallow = make_config(study={"out": out}, plan={"attempts": 1})
    deep = make_config(study={"out": out}, plan={"attempts": 3})
    conduct(shallow, answering_runtime)
    conduct(deep, answering_runtime)

    rows = load_results(out)
    assert shallow.fingerprint != deep.fingerprint
    assert shallow.condition == deep.condition
    assert {r.config_fingerprint for r in rows} == {
        shallow.fingerprint,
        deep.fingerprint,
    }
    assert {r.condition_digest for r in rows} == {shallow.condition}


def test_appending_still_refuses_a_difference_that_is_not_depth(
    make_config,
    answering_runtime,
    tmp_path,
) -> None:
    """Verify appending to existing results file with mismatched model raises ValueError."""
    out = tmp_path / "results.jsonl"
    conduct(make_config(study={"out": out}), answering_runtime)
    with pytest.raises(ValueError, match="refusing to append"):
        conduct(
            make_config(study={"out": out}, runtime={"options": {"model": "opus"}}),
            answering_runtime,
        )


def test_appending_refuses_rows_that_never_recorded_a_condition(
    make_config,
    answering_runtime,
    tmp_path,
) -> None:
    """Verify appending raises ValueError if existing rows lack condition digests."""
    out = tmp_path / "results.jsonl"
    config = make_config(study={"out": out}, plan={"attempts": 1})
    conduct(config, answering_runtime)
    stripped = [
        r.model_copy(update={"condition_digest": "", "config_fingerprint": "old"})
        for r in load_results(out)
    ]
    out.write_text(
        "".join(r.model_dump_json() + "\n" for r in stripped),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="refusing to append"):
        conduct(
            make_config(study={"out": out}, plan={"attempts": 3}),
            answering_runtime,
        )


def test_a_corrected_query_set_cannot_be_probed_onto_the_draft_it_replaced(
    make_config,
    answering_runtime,
    tmp_path,
    queries,
) -> None:
    """Verify appending to existing file raises ValueError if query set contents have changed."""
    out = tmp_path / "results.jsonl"
    conduct(make_config(study={"out": out}), answering_runtime)

    corrected = tmp_path / "corrected.json"
    edited = [q.model_copy(update={"text": q.text + " (corrected)"}) for q in queries]
    corrected.write_text(
        QuerySet(
            catalog_id="neighborhood:gcs-lifecycle-rules",
            queries=tuple(edited),
            provenance=QuerySetProvenance(origin=Origin.AUTHORED),
        ).model_dump_json(),
        encoding="utf-8",
    )
    second = make_config(study={"out": out, "queries": corrected})

    assert second.fingerprint == make_config(study={"out": out}).fingerprint
    with pytest.raises(ValueError, match="ground truth"):
        conduct(second, answering_runtime)


def test_probing_the_same_query_set_again_still_resumes_its_own_file(
    make_config,
    answering_runtime,
    tmp_path,
) -> None:
    """Verify re-running identical study configuration resumes cleanly."""
    out = tmp_path / "results.jsonl"
    first = conduct(make_config(study={"out": out}), answering_runtime).report
    again = conduct(make_config(study={"out": out}), answering_runtime).report
    assert first.reused == 0
    assert again.reused == again.probes
    assert {r.queries_digest for r in load_results(out)} != {""}


def test_a_fresh_file_takes_the_deeper_plan_without_complaint(
    make_config,
    answering_runtime,
    tmp_path,
) -> None:
    """Verify deeper plan runs without errors when targeting a new output file."""
    conduct(
        make_config(study={"out": tmp_path / "shallow.jsonl"}),
        answering_runtime,
    )
    report = conduct(
        make_config(study={"out": tmp_path / "deep.jsonl"}, plan={"attempts": 3}),
        answering_runtime,
    ).report
    assert report.probes == 6


def test_mixing_arms_in_one_file_can_be_asked_for_outright(
    make_config,
    answering_runtime,
    tmp_path,
) -> None:
    """Verify append_across_arms=True allows appending results from differing configurations."""
    out = tmp_path / "results.jsonl"
    conduct(
        make_config(study={"out": out}, plan={"attempts": 1}),
        answering_runtime,
    )
    conduct(
        make_config(study={"out": out}, runtime={"options": {"model": "opus"}}),
        answering_runtime,
        resume=False,
        append_across_arms=True,
    )
    rows = load_results(out)
    assert len(rows) == 4
    assert len({row.config_fingerprint for row in rows}) == 2
    assert len({row.condition_digest for row in rows}) == 2


def test_a_sidecar_states_what_the_run_matched_on(make_config, tmp_path: Path) -> None:
    """Verify write_sidecar creates a JSON sidecar recording complete configuration."""
    config = make_config()
    sidecar = write_sidecar(config, tmp_path / "results.jsonl")
    assert sidecar.name == "results.jsonl.config.json"
    reloaded = json.loads(sidecar.read_text(encoding="utf-8"))
    assert reloaded["config"]["runtime"]["agent"] == "fake"
    assert reloaded["config"]["catalog"]["rivals"] == config.catalog.rivals


def test_a_sidecar_is_readable_by_the_class_that_wrote_it(make_config, tmp_path: Path) -> None:
    """Verify read_sidecar reconstructs configuration and fingerprint."""
    config = make_config()
    reloaded = read_sidecar(write_sidecar(config, tmp_path / "results.jsonl"))
    assert reloaded.config.study.queries == config.study.queries
    assert reloaded.fingerprint == config.fingerprint


def test_a_sidecar_records_the_digests_of_the_run_it_annotates(
    make_config,
    tmp_path: Path,
) -> None:
    """Verify sidecar payload contains explicit fingerprint, arm, and condition strings."""
    config = make_config()
    recorded = json.loads(
        write_sidecar(config, tmp_path / "results.jsonl").read_text(encoding="utf-8"),
    )
    assert recorded["fingerprint"] == config.fingerprint
    assert recorded["arm"] == config.arm
    assert recorded["condition"] == config.condition


def test_a_sidecar_from_before_the_condition_is_refused(make_config, tmp_path: Path) -> None:
    """Verify legacy sidecars lacking condition fields fail validation."""
    sidecar = write_sidecar(make_config(), tmp_path / "results.jsonl")
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    del payload["condition"]
    sidecar.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="predates recorded digests"):
        read_sidecar(sidecar)


def test_a_sidecar_whose_condition_disagrees_is_refused(make_config, tmp_path: Path) -> None:
    """Verify read_sidecar raises ValueError if stored condition disagrees with config."""
    sidecar = write_sidecar(make_config(), tmp_path / "results.jsonl")
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    payload["condition"] = "000000000000"
    sidecar.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="condition 000000000000 does not match"):
        read_sidecar(sidecar)


def test_a_deepened_run_rewrites_its_sidecar_to_the_depth_it_reached(
    make_config,
    answering_runtime,
    tmp_path: Path,
) -> None:
    """Verify deepening a run updates sidecar to reflect new attempt counts."""
    out = tmp_path / "results.jsonl"
    shallow = make_config(study={"out": out}, plan={"attempts": 1})
    deep = make_config(study={"out": out}, plan={"attempts": 3})
    write_sidecar(shallow, out)
    conduct(shallow, answering_runtime)
    conduct(deep, answering_runtime)
    write_sidecar(deep, out)

    recorded = read_sidecar(sidecar_path(out))
    assert recorded.config.plan.attempts == 3
    assert recorded.fingerprint == deep.fingerprint
    assert recorded.condition == shallow.condition


def test_a_sidecar_whose_digest_disagrees_with_its_configuration_is_refused(
    make_config,
    tmp_path: Path,
) -> None:
    """Verify read_sidecar raises ValueError if stored fingerprint disagrees with config."""
    sidecar = write_sidecar(make_config(), tmp_path / "results.jsonl")
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    payload["fingerprint"] = "000000000000"
    sidecar.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match"):
        read_sidecar(sidecar)


def test_a_sidecar_written_before_digests_were_recorded_is_refused(
    make_config,
    tmp_path: Path,
) -> None:
    """Verify sidecars lacking required digest keys raise ValueError."""
    sidecar = tmp_path / "results.jsonl.config.json"
    sidecar.write_text(make_config().model_dump_json(), encoding="utf-8")
    with pytest.raises(ValueError, match="predates"):
        read_sidecar(sidecar)


def test_a_flat_runtime_table_cannot_re_derive_a_wrong_arm(
    make_config,
    tmp_path: Path,
) -> None:
    """Verify sidecars with flat runtime options fail schema validation."""
    payload = {
        "arm": "0" * 12,
        "fingerprint": "0" * 12,
        "condition": "0" * 12,
        "config": json.loads(make_config().model_dump_json()),
    }
    payload["config"]["runtime"]["denied_tools"] = ["Bash", "Read"]
    sidecar = tmp_path / "results.jsonl.config.json"
    sidecar.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises((ValueError, ValidationError)):
        read_sidecar(sidecar)


def test_a_sidecar_forbids_legacy_or_extra_keys(
    make_config,
    tmp_path: Path,
) -> None:
    """Verify sidecars with legacy or unrecognized extra keys fail schema validation."""
    payload = {
        "arm": "a" * 12,
        "fingerprint": "f" * 12,
        "condition": "c" * 12,
        "legacy": True,
        "config": json.loads(make_config().model_dump_json()),
    }
    sidecar = tmp_path / "results.jsonl.config.json"
    sidecar.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValidationError, match="extra_forbidden"):
        read_sidecar(sidecar)


@pytest.fixture
def catalog() -> Catalog:
    """Provide a test catalog with skills s1 and s2."""
    return Catalog(id="c", mode=CatalogMode.ALL, skills=("s1", "s2"))


def test_probe_harness_probe_and_residency(catalog: Catalog, tmp_path: Path) -> None:
    """Verify ProbeHarness executes a single probe and validates residency."""
    from reach.models import Query, QueryKind

    q = Query(query_id="q1", text="run a task", kind=QueryKind.IMPLICIT, expected_skill="s1")
    runtime = FakeRuntime({q.text: "s1"}, model="fake-model")
    harness = ProbeHarness(runtime)
    result = harness.probe(q, catalog, tmp_path)
    assert result.predicted_label == "s1"
    assert result.error is None


def test_probe_harness_run_probes_and_retries(catalog, tmp_path: Path) -> None:
    """Verify ProbeHarness.run_probes executes batches with retries and worker pool."""
    from reach.models import Query, QueryKind

    q = Query(query_id="q1", text="flaky query", kind=QueryKind.IMPLICIT, expected_skill="s1")
    attempts = 0

    def _flaky(text: str) -> SelectionOutcome:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return SelectionOutcome(error="transient network drop")
        return SelectionOutcome(invoked_skills=("s1",))

    runtime = FakeRuntime(_flaky, model="fake-model")
    harness = ProbeHarness(runtime, workers=2, retries=2, backoff_s=0.01)
    results = list(harness.run_probes([q], catalog, tmp_path, attempts=1))
    assert len(results) == 1
    assert results[0].error is None
    assert results[0].predicted_label == "s1"


def test_probe_harness_run_full_cycle(make_config, answering_runtime, tmp_path: Path) -> None:
    """Verify ProbeHarness.run executes end-to-end, writing results and sidecar."""
    out_file = tmp_path / "harness_run.jsonl"
    config = make_config(study={"out": out_file})
    harness = ProbeHarness(answering_runtime)
    outcome = harness.run(compose(config), out_path=out_file)
    assert len(outcome.results) == 2
    assert out_file.exists()
    assert (tmp_path / "harness_run.jsonl.config.json").exists()
    loaded = load_results(out_file)
    assert len(loaded) == 2


def test_evaluate_runs_end_to_end(make_config, answering_runtime, tmp_path: Path) -> None:
    """Verify evaluate executes end-to-end and returns a valid RunOutcome."""
    out_file = tmp_path / "eval_run.jsonl"
    config = make_config(study={"out": out_file})
    outcome = evaluate(config, answering_runtime)
    assert len(outcome.results) == 2
    assert outcome.report.probes == 2
    assert out_file.exists()


def test_evaluate_missing_queries_without_auto_draft_raises(
    make_config,
    answering_runtime,
) -> None:
    """Verify evaluate raises ValueError when queries are missing and auto_draft is False."""
    config = make_config(study={"queries": None})
    with pytest.raises(ValueError, match="queries path is required"):
        evaluate(config, answering_runtime, auto_draft=False)


def test_evaluate_with_auto_draft(
    make_config,
    answering_runtime,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify evaluate auto-drafts queries when missing and auto_draft is True."""
    queries_dest = tmp_path / "drafted_queries.yaml"
    config = make_config(study={"queries": queries_dest})

    from reach.models import Query, QueryKind
    from reach.queries import QuerySet

    fake_qs = QuerySet(
        catalog_id="fixture",
        queries=(
            Query(
                query_id="auto-1",
                text="How do I configure lifecycle?",
                kind=QueryKind.IMPLICIT,
                expected_skill="gcs-lifecycle-rules",
            ),
        ),
        provenance=QuerySetProvenance(origin=Origin.AUTHORED),
    )

    import reach.generate

    def _fake_generate(*_args: object, **_kwargs: object) -> QuerySet:
        return fake_qs

    monkeypatch.setattr(reach.generate, "generate_query_set", _fake_generate)

    outcome = evaluate(config, answering_runtime, auto_draft=True)
    assert len(outcome.results) == 1
    assert queries_dest.exists()


def test_evaluate_with_auto_draft_neighborhood(
    make_config,
    answering_runtime,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify evaluate builds a neighborhood scorer when catalog mode is NEIGHBORHOOD."""
    queries_dest = tmp_path / "drafted_queries.yaml"
    config = make_config(
        catalog={"mode": CatalogMode.NEIGHBORHOOD},
        study={"queries": queries_dest},
    )

    from reach.models import Query, QueryKind
    from reach.queries import QuerySet

    fake_qs = QuerySet(
        catalog_id="fixture",
        queries=(
            Query(
                query_id="auto-1",
                text="How do I configure lifecycle?",
                kind=QueryKind.IMPLICIT,
                expected_skill="gcs-lifecycle-rules",
            ),
        ),
        provenance=QuerySetProvenance(origin=Origin.AUTHORED),
    )

    import reach.generate
    import reach.retrieval

    scorer_called = []

    class _DummyScorer:
        def rank(self, _target: object, _candidates: object) -> list[tuple[str, float]]:
            return []

    def _fake_build_scorer(scorer_name: object, skills: object, cfg: object) -> object:
        scorer_called.append((scorer_name, skills, cfg))
        return _DummyScorer()

    def _fake_generate(*_args: object, **_kwargs: object) -> QuerySet:
        return fake_qs

    monkeypatch.setattr(reach.retrieval, "build_scorer", _fake_build_scorer)
    monkeypatch.setattr(reach.generate, "generate_query_set", _fake_generate)

    outcome = evaluate(config, answering_runtime, auto_draft=True)
    assert len(outcome.results) == 1
    assert len(scorer_called) == 1


def test_plan_forbids_extra_fields() -> None:
    """Verify Plan model rejects unexpected extra attributes."""
    with pytest.raises(ValidationError, match="extra"):
        Plan.model_validate(
            {
                "catalog_id": "cat1",
                "catalog_size": 1,
                "probes": 1,
                "corpus_digest": "d1",
                "extra_field": "disallowed",
            }
        )


def test_write_results_atomic_roundtrip(tmp_path: Path) -> None:
    """Verify write_results atomically writes ProbeResult items and roundtrips with load_results."""
    results = [
        ProbeResult(
            query_id="q1",
            catalog_id="cat1",
            catalog_mode=CatalogMode.ALL,
            catalog_size=2,
            model="test-model",
            runtime="fake",
            attempt=1,
            invoked_skills=("s1",),
        ),
        ProbeResult(
            query_id="q2",
            catalog_id="cat1",
            catalog_mode=CatalogMode.ALL,
            catalog_size=2,
            model="test-model",
            runtime="fake",
            attempt=1,
            error="timeout",
        ),
    ]
    target = tmp_path / "subdir" / "out.jsonl"
    written = write_results(target, results)
    assert written == target.resolve()
    assert target.exists()

    loaded = load_results(target)
    assert len(loaded) == 2
    assert loaded[0].query_id == "q1"
    assert loaded[0].invoked_skills == ("s1",)
    assert loaded[1].query_id == "q2"
    assert loaded[1].error == "timeout"


def test_resume_filters_stale_corpus_digest_catalog_and_out_of_scope_queries(
    make_config,
    answering_runtime,
    queries,
    tmp_path: Path,
) -> None:
    """Verify resume ignores rows with stale corpus_digest, catalog, or out-of-scope query_id."""
    out = tmp_path / "results.jsonl"
    config = make_config(study={"out": out}, plan={"attempts": 1})
    first_outcome = conduct(config, answering_runtime)
    assert len(first_outcome.results) == 2

    # Tamper with stored rows to simulate an out-of-scope query, a changed SKILL.md digest,
    # and a changed catalog membership (e.g. different --anchor at the same K).
    valid_row = first_outcome.results[0]
    stale_digest_row = first_outcome.results[1].model_copy(
        update={"corpus_digest": "sha256:stale-digest"},
    )
    stale_catalog_row = first_outcome.results[0].model_copy(
        update={
            "query_id": queries[1].query_id,
            "observed_catalog": ("some-other-skill",),
        },
    )
    extra_query_row = first_outcome.results[0].model_copy(
        update={"query_id": "q-unrelated-anchor"},
    )
    other_arm_row = first_outcome.results[1].model_copy(
        update={
            "config_fingerprint": "sha256:other-arm-fp",
            "condition_digest": "sha256:other-arm-cond",
        },
    )
    write_results(
        out,
        [valid_row, stale_digest_row, stale_catalog_row, extra_query_row, other_arm_row],
    )

    second_runtime = FakeRuntime(
        {q.text: q.expected_skill for q in queries},
    )
    resumed_outcome = conduct(config, second_runtime, append_across_arms=True)

    # Only valid_row (queries[0]) should be reused for the active run; queries[1] must be re-probed,
    # and q-unrelated-anchor / other_arm_row must not leak into outcome.results, while valid rows
    # from other anchors/catalogs/arms are preserved on disk.
    assert resumed_outcome.reused == 1
    assert second_runtime.queries == [queries[1].text]
    assert [r.query_id for r in resumed_outcome.results] == [
        queries[0].query_id,
        queries[1].query_id,
    ]
    disk_rows = load_results(out)
    assert len(disk_rows) == 5
    assert {r.corpus_digest for r in disk_rows} == {valid_row.corpus_digest}
