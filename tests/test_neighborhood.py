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

"""Verify neighborhood catalog construction and rival selection."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from pydantic import ValidationError

from reach.catalog import build_catalogs, build_neighborhood_catalogs
from reach.models import Catalog, CatalogMode, Skill
from reach.retrieval import Bm25Scorer

if TYPE_CHECKING:
    from conftest import SyntheticCorpusBuilder


@pytest.fixture
def small_corpus(corpus_builder: type[SyntheticCorpusBuilder]) -> list[Skill]:
    """Provide a corpus with two distinct clusters and one outlier skill."""
    return (
        corpus_builder()
        .add("gke-networking", "Configure GKE cluster networking and services.")
        .add("gke-service-mesh", "Configure GKE service mesh networking.")
        .add("gke-storage", "Configure GKE persistent volumes and storage.")
        .add("bigquery-sql", "Write and optimize BigQuery SQL queries.")
        .add("bigquery-ml", "Train BigQuery ML models with SQL.")
        .add("ads-reporting", "Pull Google Ads campaign reports.")
        .build_skills()
    )


def test_every_skill_is_the_target_of_exactly_one_catalog(small_corpus) -> None:
    """Verify each corpus skill appears as target in exactly one neighborhood catalog."""
    catalogs = build_neighborhood_catalogs(small_corpus, size=4, rivals=2)
    targets = [c.target for c in catalogs if c.target]
    assert sorted(targets) == sorted(s.name for s in small_corpus)
    assert len(targets) == len(set(targets))


def test_the_target_is_always_resident(small_corpus) -> None:
    """Verify target skill is always included in its own catalog's skills list."""
    for catalog in build_neighborhood_catalogs(small_corpus, size=4, rivals=2):
        assert catalog.target in catalog.skills


def test_catalog_size_is_constant(small_corpus) -> None:
    """Verify all constructed neighborhood catalogs have equal size."""
    for catalog in build_neighborhood_catalogs(small_corpus, size=4, rivals=2):
        assert catalog.size == 4


def test_top_rivals_are_included_ahead_of_filler(small_corpus) -> None:
    """Verify highest scoring BM25 rivals are prioritized before random filler."""
    scorer = Bm25Scorer.from_skills(small_corpus)
    catalogs = {c.target: c for c in build_neighborhood_catalogs(small_corpus, size=3, rivals=2)}
    for skill in small_corpus:
        expected = {n for n, _ in scorer.rank(skill, small_corpus)[:2]}
        assert expected <= set(catalogs[skill.name].skills), skill.name


def test_clustered_skills_find_each_other(small_corpus) -> None:
    """Verify semantically related skills are paired together as top rivals."""
    catalogs = {c.target: c for c in build_neighborhood_catalogs(small_corpus, size=3, rivals=2)}
    assert "gke-service-mesh" in catalogs["gke-networking"].skills
    assert "bigquery-ml" in catalogs["bigquery-sql"].skills


def test_composition_is_reproducible_across_runs(small_corpus) -> None:
    """Verify identical seeds produce identical catalog composition."""
    first = build_neighborhood_catalogs(small_corpus, size=5, rivals=2, seed=7)
    second = build_neighborhood_catalogs(small_corpus, size=5, rivals=2, seed=7)
    assert [c.model_dump() for c in first] == [c.model_dump() for c in second]


def test_a_different_seed_changes_only_the_filler(small_corpus) -> None:
    """Verify altering random seed changes filler skills while preserving top rivals."""
    a = {
        c.target: set(c.skills)
        for c in build_neighborhood_catalogs(small_corpus, size=5, rivals=2, seed=1)
    }
    b = {
        c.target: set(c.skills)
        for c in build_neighborhood_catalogs(small_corpus, size=5, rivals=2, seed=2)
    }
    scorer = Bm25Scorer.from_skills(small_corpus)
    for skill in small_corpus:
        rivals = {n for n, _ in scorer.rank(skill, small_corpus)[:2]} | {skill.name}
        assert rivals <= a[skill.name]
        assert rivals <= b[skill.name]
    assert a != b, "expected the seed to move the filler"


def test_one_target_composition_does_not_depend_on_the_others(small_corpus) -> None:
    """Verify neighborhood composition per target is independent of input list order."""
    full = {c.target: c.skills for c in build_neighborhood_catalogs(small_corpus, size=3, rivals=2)}
    reordered = list(reversed(small_corpus))
    shuffled = {
        c.target: c.skills for c in build_neighborhood_catalogs(reordered, size=3, rivals=2)
    }
    assert full == shuffled


def test_a_corpus_smaller_than_the_requested_size_is_not_padded(small_corpus) -> None:
    """Verify catalog size matches available corpus size when corpus is small."""
    catalogs = build_neighborhood_catalogs(small_corpus[:3], size=20, rivals=10)
    for catalog in catalogs:
        assert catalog.size == 3


def test_build_catalogs_dispatches_to_neighborhood_mode(small_corpus) -> None:
    """Verify build_catalogs with CatalogMode.NEIGHBORHOOD produces neighborhood catalogs."""
    catalogs = build_catalogs(small_corpus, CatalogMode.NEIGHBORHOOD)
    assert len(catalogs) == len(small_corpus)
    assert all(c.mode is CatalogMode.NEIGHBORHOOD for c in catalogs)


def test_empty_corpus_yields_no_catalogs() -> None:
    """Verify empty corpus produces empty list of catalogs."""
    assert build_neighborhood_catalogs([]) == []


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [({"size": 1}, "at least 2 skills"), ({"rivals": 0}, "at least 1 rival")],
)
def test_degenerate_parameters_are_rejected(small_corpus, kwargs, match) -> None:
    """Verify size < 2 or rivals < 1 raises ValueError."""
    with pytest.raises(ValueError, match=match):
        build_neighborhood_catalogs(small_corpus, **kwargs)


def test_a_neighborhood_must_name_its_target() -> None:
    """Verify Catalog validation fails when target is None in NEIGHBORHOOD mode."""
    with pytest.raises(ValidationError, match="must name its target"):
        Catalog(id="n:x", mode=CatalogMode.NEIGHBORHOOD, skills=("a", "b"))


def test_a_target_outside_its_own_catalog_is_rejected() -> None:
    """Verify Catalog validation fails when target is not in skills tuple."""
    with pytest.raises(ValidationError, match="is not resident"):
        Catalog(id="n:x", mode=CatalogMode.NEIGHBORHOOD, skills=("a", "b"), target="c")
