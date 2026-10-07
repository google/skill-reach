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

"""Verify lexical difficulty ranking and BM25 baseline computations."""

from __future__ import annotations

import pytest

from reach.difficulty import LexicalRank, lexical_rank, lexical_ranks
from reach.models import Query, QueryKind, Skill
from reach.retrieval import Bm25Scorer


@pytest.fixture
def skills(make_skill) -> list[Skill]:
    """Provide three skills with disjoint domain vocabulary."""
    return [
        make_skill("bucket-lifecycle", "Tier cold objects between storage classes."),
        make_skill("cluster-upgrade", "Roll nodes onto a newer control plane."),
        make_skill("billing-export", "Send invoices into a warehouse dataset."),
    ]


def test_query_in_the_target_vocabulary_ranks_first(skills: list[Skill]) -> None:
    """Verify exact target vocabulary query ranks in position 1."""
    rank = lexical_rank(
        "Tier my cold objects into cheaper storage classes.",
        "bucket-lifecycle",
        skills,
    )
    assert rank == LexicalRank(position=1, field_size=3)


def test_query_in_a_rival_vocabulary_ranks_behind_the_rival(
    skills: list[Skill],
) -> None:
    """Verify rival vocabulary query ranks target behind rival skill."""
    rank = lexical_rank(
        "Roll my nodes onto a newer control plane.",
        "bucket-lifecycle",
        skills,
    )
    assert rank == LexicalRank(position=3, field_size=3)


def test_query_sharing_nothing_takes_the_worst_rank(skills: list[Skill]) -> None:
    """Verify non-matching query receives worst possible position tie-break."""
    worst = LexicalRank(position=3, field_size=3)
    assert lexical_rank("Quel temps fait-il demain?", "billing-export", skills) == worst
    assert lexical_rank("Quel temps fait-il demain?", "cluster-upgrade", skills) == worst


def test_sole_resident_always_ranks_first(make_skill) -> None:
    """Verify singleton skill list always returns rank position 1."""
    only = [make_skill("bucket-lifecycle", "Tier cold objects.")]
    assert lexical_rank("Anything at all.", "bucket-lifecycle", only) == LexicalRank(
        position=1,
        field_size=1,
    )


def test_rank_is_undefined_for_a_skill_that_is_not_resident(
    skills: list[Skill],
) -> None:
    """Verify lexical_rank returns None for skills not present in catalog."""
    assert lexical_rank("Tier cold objects.", "not-here", skills) is None


def test_ranks_skip_queries_with_no_rank(skills: list[Skill]) -> None:
    """Verify lexical_ranks skips out-of-scope and unindexed skills."""
    queries = [
        Query(
            query_id="scoreable",
            text="Tier my cold objects into cheaper storage classes.",
            kind=QueryKind.NEIGHBOR_NEGATIVE,
            expected_skill="bucket-lifecycle",
        ),
        Query(query_id="abstain", text="Book me a flight.", kind=QueryKind.OUT_OF_SCOPE),
        Query(
            query_id="stray",
            text="Something else entirely.",
            kind=QueryKind.NEIGHBOR_NEGATIVE,
            expected_skill="not-here",
        ),
    ]
    assert lexical_ranks(queries, skills) == {
        "scoreable": LexicalRank(position=1, field_size=3),
    }


def test_batched_ranks_match_the_one_at_a_time_computation(
    skills: list[Skill],
) -> None:
    """Verify lexical_ranks batch computation matches individual lexical_rank calls."""
    queries = [
        Query(
            query_id=f"q{i}",
            text=text,
            kind=QueryKind.NEIGHBOR_NEGATIVE,
            expected_skill="bucket-lifecycle",
        )
        for i, text in enumerate(
            [
                "Tier my cold objects into cheaper storage classes.",
                "Roll my nodes onto a newer control plane.",
                "Send the invoices to a warehouse dataset.",
                "Quel temps fait-il demain?",
            ],
        )
    ]
    batched = lexical_ranks(queries, skills)
    assert batched == {
        q.query_id: lexical_rank(q.text, "bucket-lifecycle", skills) for q in queries
    }


def test_a_rank_carries_the_field_it_was_taken_in(
    skills: list[Skill],
) -> None:
    """Verify LexicalRank includes total field size and formatting."""
    narrow = lexical_rank("Tier my cold objects.", "bucket-lifecycle", skills[:2])
    wide = lexical_rank("Tier my cold objects.", "bucket-lifecycle", skills)
    assert narrow is not None
    assert wide is not None
    assert narrow.position == wide.position == 1
    assert (narrow.field_size, wide.field_size) == (2, 3)
    assert narrow != wide, "the same position in a different field is a different rank"
    assert narrow.is_top
    assert wide.is_top
    assert str(wide) == "1/3"


def test_ranks_from_different_catalogs_do_not_compare_equal(make_skill) -> None:
    """Verify LexicalRank instances with different field sizes evaluate unequal."""
    small = [make_skill(f"s{i}", f"Skill about topic {i}.") for i in range(3)]
    large = [*small, *(make_skill(f"x{i}", f"Other topic {i}.") for i in range(9))]
    here = lexical_rank("Skill about topic 0.", "s0", small)
    there = lexical_rank("Skill about topic 0.", "s0", large)
    assert here is not None
    assert there is not None
    assert here.position == there.position == 1
    assert here != there, "same position in different fields is not the same rank"
    assert len({here, there}) == 2, "pooling these two must not silently collapse"


def test_rank_text_reproduces_rank_when_the_target_is_excluded(
    skills: list[Skill],
) -> None:
    """Verify Bm25Scorer.rank matches rank_text on concatenated skill text."""
    scorer = Bm25Scorer.from_skills(skills)
    target = skills[0]
    assert scorer.rank(target, skills) == scorer.rank_text(
        f"{target.name} {target.description}",
        skills[1:],
    )


def test_lexical_rank_with_precomputed_scorer_and_resident(skills: list[Skill]) -> None:
    """Verify lexical_rank functions correctly when passing precomputed scorer and resident set."""
    scorer = Bm25Scorer.from_skills(skills)
    resident = {s.name for s in skills}

    rank = lexical_rank(
        "Tier my cold objects into cheaper storage classes.",
        "bucket-lifecycle",
        skills,
        scorer=scorer,
        resident=resident,
    )
    assert rank == LexicalRank(position=1, field_size=3)

    absent = lexical_rank(
        "Tier my cold objects.",
        "not-here",
        skills,
        scorer=scorer,
        resident=resident,
    )
    assert absent is None
