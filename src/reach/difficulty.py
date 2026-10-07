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

"""Assess query difficulty based on lexical overlap rankings with skill descriptions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from reach.retrieval import Bm25Scorer

if TYPE_CHECKING:
    from collections.abc import Container, Iterable, Sequence

    from reach.models import Query, Skill


class LexicalRank(BaseModel):
    """Represent the lexical rank of an expected skill among candidate skills."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    position: int
    field_size: int

    @property
    def is_top(self) -> bool:
        """Return True if lexical overlap ranked the target skill first."""
        return self.position == 1

    def __str__(self) -> str:
        """Format the lexical rank as 'position/field_size'."""
        return f"{self.position}/{self.field_size}"

    def __lt__(self, other: LexicalRank) -> bool:
        """Compare ranks by position then field size for stable ordering."""
        return (self.position, self.field_size) < (other.position, other.field_size)


def _position(
    scorer: Bm25Scorer,
    text: str,
    expected: str,
    skills: Sequence[Skill],
) -> int:
    """Calculate the 1-based pessimistic rank of expected skill for query text."""
    ranked = scorer.rank_text(text, skills)
    mine = next(score for name, score in ranked if name == expected)
    return sum(1 for _, score in ranked if score >= mine)


def lexical_rank(
    text: str,
    expected: str,
    skills: Sequence[Skill],
    scorer: Bm25Scorer | None = None,
    resident: Container[str] | None = None,
) -> LexicalRank | None:
    """Calculate LexicalRank for a query against candidate skills, or None if absent."""
    if resident is not None:
        if expected not in resident:
            return None
    elif not any(s.name == expected for s in skills):
        return None
    rank_scorer = scorer if scorer is not None else Bm25Scorer.from_skills(skills)
    return LexicalRank(
        position=_position(rank_scorer, text, expected, skills),
        field_size=len(skills),
    )


def lexical_ranks(
    queries: Iterable[Query],
    skills: Sequence[Skill],
) -> dict[str, LexicalRank]:
    """Map query IDs to their computed LexicalRank across candidate skills."""
    scorer = Bm25Scorer.from_skills(skills)
    resident = {s.name for s in skills}
    results: dict[str, LexicalRank] = {}
    for q in queries:
        if q.expected_skill is None:
            continue
        rank = lexical_rank(q.text, q.expected_skill, skills, scorer=scorer, resident=resident)
        if rank is not None:
            results[q.query_id] = rank
    return results
