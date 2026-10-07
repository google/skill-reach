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

"""Verify runtime skill selection converts into provenanced probe results."""

from __future__ import annotations

from typing import TYPE_CHECKING, override

import pytest

from reach.models import Catalog, CatalogMode, ProbeResult, Provenance, Query, QueryKind, Skill
from reach.run import ProbeHarness, validate_residency
from reach.runtime import SelectionOutcome
from reach.runtime.fake import FakeRuntime

if TYPE_CHECKING:
    from pathlib import Path


def _skills(names: tuple[str, ...], root: Path) -> list[Skill]:
    """Build mock Skill instances for catalog installation testing."""
    return [Skill(name=n, description=f"does {n}", path=root / n) for n in names]


@pytest.fixture
def catalog() -> Catalog:
    """Provide a two-skill catalog for probe testing."""
    return Catalog(id="c", mode=CatalogMode.ALL, skills=("a", "b"))


@pytest.fixture
def query() -> Query:
    """Provide a labeled query fixture."""
    return Query(
        query_id="q1",
        text="do a thing",
        kind=QueryKind.IMPLICIT,
        expected_skill="a",
    )


def test_missing_skill_invalidates_the_probe(catalog: Catalog) -> None:
    """Verify validate_residency reports uninstalled catalog skills."""
    assert validate_residency(catalog, ("a",)) == "catalog not resident: b"


def test_extra_skills_are_not_a_fault(catalog: Catalog) -> None:
    """Verify extra installed skills (such as runtime built-ins) do not trigger residency error."""
    builtins = tuple(f"builtin-{i}" for i in range(14))
    assert validate_residency(catalog, ("a", "b", *builtins)) is None


def test_an_unreported_catalog_is_not_evidence_of_absence(catalog: Catalog) -> None:
    """Verify empty observed catalog from runtime produces no residency error."""
    assert validate_residency(catalog, ()) is None


def test_every_missing_skill_is_named(catalog: Catalog) -> None:
    """Verify all missing skills are enumerated in residency error message."""
    assert validate_residency(catalog, ("z",)) == "catalog not resident: a, b"


def test_validate_residency_dynamic_allows_subset(catalog: Catalog) -> None:
    """Verify dynamic retrieval allows a subset of catalog skills to be observed."""
    assert validate_residency(catalog, ("a",), dynamic=True) is None


def test_validate_residency_dynamic_reports_unknown_skills(catalog: Catalog) -> None:
    """Verify dynamic retrieval reports skills that are not part of the catalog."""
    assert (
        validate_residency(catalog, ("a", "mystery"), dynamic=True)
        == "catalog contains unknown skills: mystery"
    )


def _probe(
    query: Query,
    catalog: Catalog,
    workdir: Path,
    runtime: FakeRuntime,
    attempt: int = 1,
    provenance: Provenance | None = None,
) -> ProbeResult:
    """Execute single probe via ProbeHarness."""
    return ProbeHarness(runtime).probe(
        query,
        catalog,
        workdir,
        attempt=attempt,
        provenance=provenance,
    )


def test_records_the_selection_and_its_provenance(
    query: Query,
    catalog: Catalog,
    tmp_path: Path,
) -> None:
    """Verify probe result captures skill selection, runtime identity, and provenance digests."""
    runtime = FakeRuntime({query.text: "a"}, model="sonnet")
    runtime.install(catalog, _skills(("a", "b"), tmp_path), tmp_path)
    result = _probe(
        query,
        catalog,
        tmp_path,
        runtime,
        provenance=Provenance(config_fingerprint="abc123", corpus_digest="def456"),
    )
    assert result.invoked_skill == "a"
    assert (result.runtime, result.model) == ("fake", "sonnet")
    assert (result.config_fingerprint, result.corpus_digest) == ("abc123", "def456")
    assert (result.catalog_id, result.catalog_size) == ("c", 2)
    assert result.error is None


@pytest.mark.parametrize("resolved", ["claude-sonnet-5", ""])
def test_the_row_records_what_answered_beside_what_was_asked_for(
    query: Query,
    catalog: Catalog,
    tmp_path: Path,
    resolved: str,
) -> None:
    """Verify probe result preserves configured model alias alongside resolved runtime model."""

    class Named(FakeRuntime):
        """Mock runtime returning a specific resolved_model string."""

        @override
        def select(
            self,
            query_text: str,
            workdir: Path,
            target_skill: str | None = None,
        ) -> SelectionOutcome:
            outcome = super().select(query_text, workdir, target_skill=target_skill)
            return outcome.model_copy(update={"resolved_model": resolved})

    runtime = Named({query.text: "a"}, model="sonnet")
    runtime.install(catalog, _skills(("a", "b"), tmp_path), tmp_path)
    result = _probe(query, catalog, tmp_path, runtime)
    assert result.model == "sonnet"
    assert result.resolved_model == resolved


def test_abstention_is_recorded_not_treated_as_failure(
    query: Query,
    catalog: Catalog,
    tmp_path: Path,
) -> None:
    """Verify non-selection results in invoked_skill=None and predicted_label='(no skill)'."""
    result = _probe(query, catalog, tmp_path, FakeRuntime())
    assert result.invoked_skill is None
    assert result.error is None
    assert result.predicted_label == "(no skill)"


def test_runtime_error_is_carried_through(
    query: Query,
    catalog: Catalog,
    tmp_path: Path,
) -> None:
    """Verify runtime error strings are preserved on probe result."""
    runtime = FakeRuntime({query.text: SelectionOutcome(error="timeout")})
    assert _probe(query, catalog, tmp_path, runtime).error == "timeout"


def test_a_runtime_error_outranks_a_residency_check(
    query: Query,
    catalog: Catalog,
    tmp_path: Path,
) -> None:
    """Verify runtime error takes precedence over residency check failures."""
    runtime = FakeRuntime(
        {query.text: SelectionOutcome(error="timeout", observed_catalog=("a",))},
    )
    assert _probe(query, catalog, tmp_path, runtime).error == "timeout"


def test_a_drifted_catalog_invalidates_the_probe(
    query: Query,
    catalog: Catalog,
    tmp_path: Path,
) -> None:
    """Verify probe result records residency error if observed catalog is missing skills."""
    runtime = FakeRuntime(
        {query.text: SelectionOutcome(invoked_skills=("a",), observed_catalog=("a",))},
    )
    result = _probe(query, catalog, tmp_path, runtime)
    assert result.error == "catalog not resident: b"


@pytest.mark.parametrize(
    ("scripted", "attempt", "expected_first", "expected_tuple"),
    [
        (SelectionOutcome(invoked_skills=("a", "b")), 3, "a", ("a", "b")),
        ("a", 1, "a", ("a",)),
        (SelectionOutcome(invoked_skills=()), 1, None, ()),
    ],
)
def test_invocations_and_attempt_are_carried_onto_the_row(
    query: Query,
    catalog: Catalog,
    tmp_path: Path,
    scripted: SelectionOutcome | str,
    attempt: int,
    expected_first: str | None,
    expected_tuple: tuple[str, ...],
) -> None:
    """Verify invoked_skills tuple and attempt number are preserved on probe result."""
    runtime = FakeRuntime({query.text: scripted})
    result = _probe(query, catalog, tmp_path, runtime, attempt=attempt)
    assert result.attempt == attempt
    assert result.invoked_skill == expected_first
    assert result.invoked_skills == expected_tuple
