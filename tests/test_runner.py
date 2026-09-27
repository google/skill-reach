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

"""Verify probe scheduling, retry-with-backoff, concurrency, resume, and persistence."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

import pytest

from reach.models import Catalog, CatalogMode, ProbeResult, Query
from reach.run import (
    ProbeHarness,
    append_result,
    completed_attempts,
    load_results,
    plan_probes,
    recorded_fingerprints,
    validate_appendable,
)
from reach.runtime import SelectionOutcome
from reach.runtime.fake import FakeRuntime

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def catalog() -> Catalog:
    """Provide a two-skill catalog for probe testing."""
    return Catalog(
        id="test:Storage",
        mode=CatalogMode.ALL,
        skills=("gcs-lifecycle-rules", "gcs-retention-policy"),
    )


def drive(
    queries: list[Query],
    catalog: Catalog,
    tmp_path: Path,
    runtime: FakeRuntime,
    sleep=None,
    **kwargs: Any,
) -> list[ProbeResult]:
    """Run probe set synchronously with optional sleep callback."""
    workers = kwargs.pop("workers", 1)
    retries = kwargs.pop("retries", 2)
    backoff_s = kwargs.pop("backoff_s", 5.0)
    pause_s = kwargs.pop("pause_s", 0.0)
    harness = ProbeHarness(
        runtime,
        workers=workers,
        retries=retries,
        backoff_s=backoff_s,
        pause_s=pause_s,
        sleep=sleep if sleep is not None else (lambda _: None),
    )
    return list(
        harness.run_probes(
            queries,
            catalog,
            tmp_path,
            **kwargs,
        ),
    )


def test_attempts_are_interleaved_not_grouped(queries: list[Query]) -> None:
    """Verify planned probes interleave all queries per attempt index."""
    pairs = plan_probes(queries, attempts=2)
    assert [(q.id, a) for q, a in pairs] == [
        ("q-lifecycle", 1),
        ("q-retention", 1),
        ("q-lifecycle", 2),
        ("q-retention", 2),
    ]


def test_runs_each_query_the_planned_number_of_times(
    queries: list[Query],
    catalog: Catalog,
    tmp_path: Path,
    answering_runtime,
) -> None:
    """Verify probe execution runs each query for each configured attempt."""
    results = drive(queries, catalog, tmp_path, answering_runtime, attempts=3)
    assert len(results) == 6
    assert sorted(r.attempt for r in results) == [1, 1, 2, 2, 3, 3]


def test_failed_probe_is_retried_with_exponential_backoff(
    queries: list[Query],
    catalog: Catalog,
    tmp_path: Path,
) -> None:
    """Verify probe failure retries with exponential backoff delay."""
    calls = {"n": 0}

    def flaky(_text: str) -> SelectionOutcome:
        calls["n"] += 1
        if calls["n"] == 1:
            return SelectionOutcome(error="timeout")
        return SelectionOutcome(invoked_skills=("gcs-lifecycle-rules",))

    slept: list[float] = []
    results = drive(
        queries[:1],
        catalog,
        tmp_path,
        FakeRuntime(flaky),
        sleep=slept.append,
        attempts=1,
        retries=2,
        backoff_s=5.0,
    )
    assert calls["n"] == 2
    assert slept == [5.0]
    assert results[0].error is None


def test_persistent_failure_is_recorded_not_raised(
    queries: list[Query],
    catalog: Catalog,
    tmp_path: Path,
) -> None:
    """Verify repeated probe failures exhaust retries and record error without raising."""
    slept: list[float] = []
    runtime = FakeRuntime(lambda _t: SelectionOutcome(error="timeout"))
    results = drive(
        queries[:1],
        catalog,
        tmp_path,
        runtime,
        sleep=slept.append,
        attempts=1,
        retries=2,
        backoff_s=5.0,
    )
    assert slept == [5.0, 10.0]
    assert results[0].error == "timeout"


def test_results_are_persisted_as_they_land(
    queries: list[Query],
    catalog: Catalog,
    tmp_path: Path,
    answering_runtime,
) -> None:
    """Verify probe results are appended to JSONL file as stream yields."""
    out = tmp_path / "nested" / "results.jsonl"
    out.parent.mkdir()
    harness = ProbeHarness(answering_runtime, sleep=lambda _: None)
    stream = harness.run_probes(
        queries,
        catalog,
        tmp_path,
        attempts=1,
        out_path=out,
    )
    next(stream)
    assert len(load_results(out)) == 1

    list(stream)
    reloaded = load_results(out)
    assert [r.query_id for r in reloaded] == [q.id for q in queries]
    assert reloaded[0].recorded_at.tzinfo is not None


def test_pause_between_probes_is_honored(
    queries: list[Query],
    catalog: Catalog,
    tmp_path: Path,
    answering_runtime,
) -> None:
    """Verify pause_s delay is applied between successive probes."""
    slept: list[float] = []
    drive(
        queries,
        catalog,
        tmp_path,
        answering_runtime,
        sleep=slept.append,
        attempts=1,
        pause_s=2.5,
    )
    assert slept == [2.5, 2.5]


def test_workers_default_to_the_existing_sequential_behavior(
    queries: list[Query],
    catalog: Catalog,
    tmp_path: Path,
    answering_runtime,
) -> None:
    """Verify default workers behavior matches workers=1 execution."""
    with_default = drive(queries, catalog, tmp_path, answering_runtime, attempts=2)
    with_explicit_one = drive(
        queries,
        catalog,
        tmp_path,
        answering_runtime,
        attempts=2,
        workers=1,
    )
    assert [(r.query_id, r.attempt, r.invoked_skill) for r in with_default] == [
        (r.query_id, r.attempt, r.invoked_skill) for r in with_explicit_one
    ]


def test_concurrent_workers_run_every_planned_probe_exactly_once(
    queries: list[Query],
    catalog: Catalog,
    tmp_path: Path,
    answering_runtime,
) -> None:
    """Verify concurrent workers execute every planned probe exactly once."""
    results = drive(
        queries,
        catalog,
        tmp_path,
        answering_runtime,
        attempts=3,
        workers=4,
    )
    assert len(results) == 6
    assert {(r.query_id, r.attempt) for r in results} == {
        (q.id, a) for q, a in plan_probes(queries, attempts=3)
    }


def test_worker_count_bounds_how_many_probes_run_at_once(
    queries: list[Query],
    catalog: Catalog,
    tmp_path: Path,
) -> None:
    """Verify active concurrent probe executions never exceed worker count."""
    lock = threading.Lock()
    active = 0
    peak = 0

    def script(_text: str) -> None:
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        threading.Event().wait(0.005)
        with lock:
            active -= 1

    runtime = FakeRuntime(script)
    drive(queries, catalog, tmp_path, runtime, attempts=2, workers=2)
    assert peak <= 2


def test_concurrency_actually_overlaps_work_rather_than_serializing_it(
    queries: list[Query],
    catalog: Catalog,
    tmp_path: Path,
) -> None:
    """Verify workers execute concurrently in parallel."""
    lock = threading.Lock()
    active = 0
    peak = 0
    gate = threading.Event()

    def script(_text: str) -> None:
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
            if peak >= 2:
                gate.set()
        if not gate.is_set():
            gate.wait(timeout=1.0)
        with lock:
            active -= 1

    runtime = FakeRuntime(script)
    drive(queries, catalog, tmp_path, runtime, attempts=3, workers=2)
    assert peak == 2


def test_concurrent_appends_are_all_individually_valid_json_lines(
    queries: list[Query],
    catalog: Catalog,
    tmp_path: Path,
    answering_runtime,
) -> None:
    """Verify thread-safe JSONL appends produce uncorrupted lines under concurrency."""
    out = tmp_path / "results.jsonl"
    drive(
        queries,
        catalog,
        tmp_path,
        answering_runtime,
        attempts=5,
        out_path=out,
        workers=4,
    )
    loaded = load_results(out)
    assert len(loaded) == len(queries) * 5
    assert {(r.query_id, r.attempt) for r in loaded} == {
        (q.id, a) for q, a in plan_probes(queries, attempts=5)
    }


def test_retries_still_happen_per_probe_under_concurrency(
    queries: list[Query],
    catalog: Catalog,
    tmp_path: Path,
) -> None:
    """Verify per-probe retries function correctly with concurrent workers."""
    calls: dict[str, int] = {}
    lock = threading.Lock()

    def flaky(text: str) -> SelectionOutcome:
        with lock:
            calls[text] = calls.get(text, 0) + 1
            n = calls[text]
        if n == 1:
            return SelectionOutcome(error="timeout")
        return SelectionOutcome(invoked_skills=("gcs-lifecycle-rules",))

    runtime = FakeRuntime(flaky)
    results = drive(
        queries,
        catalog,
        tmp_path,
        runtime,
        attempts=1,
        retries=2,
        backoff_s=0.0,
        workers=2,
    )
    assert all(r.error is None for r in results)
    assert all(calls[q.text] == 2 for q in queries)


def test_pause_is_applied_per_worker_not_once_for_the_whole_run(
    queries: list[Query],
    catalog: Catalog,
    tmp_path: Path,
    answering_runtime,
) -> None:
    """Verify pause_s delay applies per worker thread across probe executions."""
    slept: list[float] = []
    drive(
        queries,
        catalog,
        tmp_path,
        answering_runtime,
        sleep=slept.append,
        attempts=2,
        pause_s=2.5,
        workers=2,
    )
    assert slept.count(2.5) == len(queries) * 2


def test_completed_attempts_are_read_back(
    queries: list[Query],
    catalog: Catalog,
    tmp_path: Path,
    answering_runtime,
) -> None:
    """Verify completed_attempts extracts (query_id, attempt) pairs from output file."""
    out = tmp_path / "results.jsonl"
    drive(queries, catalog, tmp_path, answering_runtime, attempts=1, out_path=out)
    assert completed_attempts(out) == {("q-lifecycle", 1), ("q-retention", 1)}


def test_an_absent_results_file_has_nothing_to_resume(tmp_path: Path) -> None:
    """Verify completed_attempts returns empty set for non-existent file."""
    assert completed_attempts(tmp_path / "never-written.jsonl") == set()


def test_errored_rows_are_work_that_still_needs_doing(
    queries: list[Query],
    catalog: Catalog,
    tmp_path: Path,
) -> None:
    """Verify completed_attempts excludes errored probe results."""
    out = tmp_path / "results.jsonl"
    runtime = FakeRuntime(lambda _t: SelectionOutcome(error="timeout"))
    drive(queries[:1], catalog, tmp_path, runtime, attempts=1, retries=0, out_path=out)
    assert completed_attempts(out) == set()


def test_skipping_completed_work_costs_nothing(
    queries: list[Query],
    catalog: Catalog,
    tmp_path: Path,
    answering_runtime,
) -> None:
    """Verify skip set bypasses execution for specified probes."""
    results = drive(
        queries,
        catalog,
        tmp_path,
        answering_runtime,
        attempts=1,
        skip={("q-lifecycle", 1)},
    )
    assert [r.query_id for r in results] == ["q-retention"]
    assert answering_runtime.queries == [queries[1].text]


def test_a_resumed_run_appends_rather_than_truncating(
    queries: list[Query],
    catalog: Catalog,
    tmp_path: Path,
    answering_runtime,
) -> None:
    """Verify resuming execution appends missing probe results to existing file."""
    out = tmp_path / "results.jsonl"
    drive(
        queries,
        catalog,
        tmp_path,
        answering_runtime,
        attempts=1,
        out_path=out,
        skip={("q-retention", 1)},
    )
    drive(
        queries,
        catalog,
        tmp_path,
        answering_runtime,
        attempts=1,
        out_path=out,
        skip=completed_attempts(out),
    )
    assert sorted(r.query_id for r in load_results(out)) == [
        "q-lifecycle",
        "q-retention",
    ]


def test_append_is_line_delimited(
    tmp_path: Path,
    make_result,
) -> None:
    """Verify append_result writes one JSON object per newline-delimited row."""
    out = tmp_path / "results.jsonl"
    append_result(out, make_result("q-lifecycle", "gcs-lifecycle-rules"))
    append_result(out, make_result("q-retention", None))
    assert len(out.read_text(encoding="utf-8").strip().splitlines()) == 2


def record(path: Path, make_result, *rows: tuple[str, str]) -> None:
    """Append mock results with specified fingerprints."""
    for query_id, fingerprint in rows:
        append_result(
            path,
            make_result(query_id, "gcs-lifecycle-rules", fingerprint=fingerprint),
        )


# --- Inspecting recorded arms and fingerprints -----------------------------


def test_an_absent_results_file_names_no_arm(tmp_path: Path) -> None:
    """Verify recorded_fingerprints returns empty set for non-existent file."""
    assert recorded_fingerprints(tmp_path / "never-written.jsonl") == set()


@pytest.mark.parametrize(
    ("recorded", "expected"),
    [
        ((("q-lifecycle", "aaaaaaaaaaaa"),), {"aaaaaaaaaaaa"}),
        (
            (("q-lifecycle", "aaaaaaaaaaaa"), ("q-retention", "aaaaaaaaaaaa")),
            {"aaaaaaaaaaaa"},
        ),
        (
            (("q-lifecycle", "aaaaaaaaaaaa"), ("q-retention", "bbbbbbbbbbbb")),
            {"aaaaaaaaaaaa", "bbbbbbbbbbbb"},
        ),
        ((("q-lifecycle", ""), ("q-retention", "aaaaaaaaaaaa")), {"", "aaaaaaaaaaaa"}),
    ],
    ids=["one-row", "one-arm", "two-arms", "one-arm-and-one-undated"],
)
def test_the_arms_a_results_file_holds_are_read_off_every_row(
    tmp_path: Path,
    make_result,
    recorded,
    expected,
) -> None:
    """Verify recorded_fingerprints returns all distinct fingerprints from results file."""
    out = tmp_path / "results.jsonl"
    record(out, make_result, *recorded)
    assert recorded_fingerprints(out) == expected


def test_appending_under_the_same_arm_is_allowed(tmp_path: Path, make_result) -> None:
    """Verify validate_appendable passes when fingerprint matches existing rows."""
    out = tmp_path / "results.jsonl"
    record(out, make_result, ("q-lifecycle", "aaaaaaaaaaaa"))
    validate_appendable(out, "aaaaaaaaaaaa")


def test_an_absent_results_file_is_appendable(tmp_path: Path) -> None:
    """Verify validate_appendable passes for non-existent destination file."""
    validate_appendable(tmp_path / "nothing-here.jsonl", "aaaaaaaaaaaa")


def test_appending_across_arms_is_refused(tmp_path: Path, make_result) -> None:
    """Verify validate_appendable raises ValueError on fingerprint mismatch."""
    out = tmp_path / "results.jsonl"
    record(out, make_result, ("q-lifecycle", "aaaaaaaaaaaa"))
    with pytest.raises(ValueError, match="refusing to append"):
        validate_appendable(out, "bbbbbbbbbbbb")


def test_rows_recorded_before_configurations_were_citable_are_refused(
    tmp_path: Path,
    make_result,
) -> None:
    """Verify validate_appendable refuses appending to files containing unrecorded fingerprints."""
    out = tmp_path / "results.jsonl"
    record(out, make_result, ("q-lifecycle", ""))
    with pytest.raises(ValueError, match=r"\(unrecorded\)"):
        validate_appendable(out, "bbbbbbbbbbbb")


def test_a_file_already_holding_two_arms_names_both(tmp_path: Path, make_result) -> None:
    """Verify validate_appendable error message enumerates all existing fingerprints."""
    out = tmp_path / "results.jsonl"
    record(
        out,
        make_result,
        ("q-lifecycle", "aaaaaaaaaaaa"),
        ("q-retention", "bbbbbbbbbbbb"),
    )
    with pytest.raises(ValueError, match="aaaaaaaaaaaa, bbbbbbbbbbbb"):
        validate_appendable(out, "cccccccccccc")


def test_appending_at_a_greater_depth_under_one_condition_is_allowed(
    tmp_path: Path,
    make_result,
) -> None:
    """Verify validate_appendable allows differing fingerprints when condition digest matches."""
    out = tmp_path / "results.jsonl"
    append_result(
        out,
        make_result(
            "q-lifecycle",
            "gcs-lifecycle-rules",
            fingerprint="aaaaaaaaaaaa",
            condition="cccccccccccc",
        ),
    )
    validate_appendable(out, "bbbbbbbbbbbb", "cccccccccccc")


def test_a_matching_condition_does_not_excuse_a_caller_that_names_none(
    tmp_path: Path,
    make_result,
) -> None:
    """Verify validate_appendable requires explicit condition argument to allow depth changes."""
    out = tmp_path / "results.jsonl"
    append_result(
        out,
        make_result(
            "q-lifecycle",
            "gcs-lifecycle-rules",
            fingerprint="aaaaaaaaaaaa",
            condition="cccccccccccc",
        ),
    )
    with pytest.raises(ValueError, match="refusing to append"):
        validate_appendable(out, "bbbbbbbbbbbb")


def test_rows_carrying_no_condition_are_refused_by_a_deepening_run(
    tmp_path: Path,
    make_result,
) -> None:
    """Verify validate_appendable refuses deepening against rows lacking condition digests."""
    out = tmp_path / "results.jsonl"
    record(out, make_result, ("q-lifecycle", "aaaaaaaaaaaa"))
    with pytest.raises(ValueError, match="refusing to append"):
        validate_appendable(out, "bbbbbbbbbbbb", "cccccccccccc")


def test_appending_the_same_query_set_is_allowed(tmp_path: Path, make_result) -> None:
    """Verify validate_appendable passes when query set digest matches existing rows."""
    out = tmp_path / "results.jsonl"
    append_result(
        out,
        make_result(
            "q-lifecycle",
            "gcs-lifecycle-rules",
            fingerprint="aaaaaaaaaaaa",
            queries="111111111111",
        ),
    )
    validate_appendable(out, "aaaaaaaaaaaa", queries="111111111111")


def test_appending_a_different_query_set_under_one_arm_is_refused(
    tmp_path: Path,
    make_result,
) -> None:
    """Verify validate_appendable raises ValueError when query set digest disagrees with file."""
    out = tmp_path / "results.jsonl"
    append_result(
        out,
        make_result(
            "q-lifecycle",
            "gcs-lifecycle-rules",
            fingerprint="aaaaaaaaaaaa",
            queries="111111111111",
        ),
    )
    with pytest.raises(ValueError, match="111111111111"):
        validate_appendable(out, "aaaaaaaaaaaa", queries="222222222222")


def test_deepening_does_not_forgive_a_changed_query_set(tmp_path: Path, make_result) -> None:
    """Verify matching condition digest does not permit appending mismatched query set digests."""
    out = tmp_path / "results.jsonl"
    append_result(
        out,
        make_result(
            "q-lifecycle",
            "gcs-lifecycle-rules",
            fingerprint="aaaaaaaaaaaa",
            condition="cccccccccccc",
            queries="111111111111",
        ),
    )
    with pytest.raises(ValueError, match="111111111111"):
        validate_appendable(out, "bbbbbbbbbbbb", "cccccccccccc", queries="222222222222")


def test_rows_recorded_without_queries_digest_refuse_appending_with_queries(
    tmp_path: Path,
    make_result,
) -> None:
    """Verify legacy rows with empty query digests refuse appending with explicit query digests."""
    out = tmp_path / "results.jsonl"
    record(out, make_result, ("q-lifecycle", "aaaaaaaaaaaa"))
    with pytest.raises(ValueError, match="refusing to append"):
        validate_appendable(out, "aaaaaaaaaaaa", queries="111111111111")


def test_a_caller_naming_no_query_set_does_not_refuse_rows_that_carry_one(
    tmp_path: Path,
    make_result,
) -> None:
    """Verify omitted query digest argument does not fail validation against populated rows."""
    out = tmp_path / "results.jsonl"
    append_result(
        out,
        make_result(
            "q-lifecycle",
            "gcs-lifecycle-rules",
            fingerprint="aaaaaaaaaaaa",
            queries="111111111111",
        ),
    )
    validate_appendable(out, "aaaaaaaaaaaa")


def test_a_file_already_holding_two_query_sets_names_both(tmp_path: Path, make_result) -> None:
    """Verify validate_appendable error message enumerates all conflicting query digests in file."""
    out = tmp_path / "results.jsonl"
    for query_id, digest in (
        ("q-lifecycle", "111111111111"),
        ("q-retention", "222222222222"),
    ):
        append_result(
            out,
            make_result(
                query_id,
                "gcs-lifecycle-rules",
                fingerprint="aaaaaaaaaaaa",
                queries=digest,
            ),
        )
    with pytest.raises(ValueError, match="111111111111, 222222222222"):
        validate_appendable(out, "aaaaaaaaaaaa", queries="333333333333")


def test_a_changed_arm_is_reported_before_a_changed_query_set(
    tmp_path: Path,
    make_result,
) -> None:
    """Verify arm fingerprint mismatch is reported prior to query digest mismatch."""
    out = tmp_path / "results.jsonl"
    append_result(
        out,
        make_result(
            "q-lifecycle",
            "gcs-lifecycle-rules",
            fingerprint="aaaaaaaaaaaa",
            queries="111111111111",
        ),
    )
    with pytest.raises(ValueError, match="recorded under"):
        validate_appendable(out, "bbbbbbbbbbbb", queries="222222222222")


def test_a_condition_forgives_depth_and_not_the_row_beside_it(
    tmp_path: Path,
    make_result,
) -> None:
    """Verify condition digest match does not forgive unrelated arms in the same file."""
    out = tmp_path / "results.jsonl"
    append_result(
        out,
        make_result(
            "q-lifecycle",
            "gcs-lifecycle-rules",
            fingerprint="aaaaaaaaaaaa",
            condition="cccccccccccc",
        ),
    )
    append_result(
        out,
        make_result(
            "q-retention",
            "gcs-lifecycle-rules",
            fingerprint="dddddddddddd",
            condition="eeeeeeeeeeee",
        ),
    )
    with pytest.raises(ValueError, match="dddddddddddd"):
        validate_appendable(out, "bbbbbbbbbbbb", "cccccccccccc")
