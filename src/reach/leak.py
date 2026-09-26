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

"""Detect query leakage containing explicit skill names or distinctive terms."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from reach.config import QuerySettings
from reach.retrieval import Bm25Scorer, skill_text, tokenize

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from reach.models import Query, Skill

_DEFAULT_QUERY = QuerySettings()

#: Minimum IDF threshold for distinctive terms against background corpus.
BACKGROUND_IDF = _DEFAULT_QUERY.distinctive_idf_floor

#: Canonical 33-word English stopword list from Apache Lucene
#: (EnglishAnalyzer.ENGLISH_STOP_WORDS_SET) and bm25s.stopwords.STOPWORDS_EN.
LUCENE_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "but",
        "by",
        "for",
        "if",
        "in",
        "into",
        "is",
        "it",
        "no",
        "not",
        "of",
        "on",
        "or",
        "such",
        "that",
        "the",
        "their",
        "then",
        "there",
        "these",
        "they",
        "this",
        "to",
        "was",
        "will",
        "with",
    },
)

#: Closed class of English personal, possessive, and reflexive pronouns (holding out 'i' and 'us').
PRONOUNS = frozenset(
    {
        "he",
        "her",
        "hers",
        "herself",
        "him",
        "himself",
        "his",
        "it",
        "its",
        "itself",
        "me",
        "mine",
        "my",
        "myself",
        "our",
        "ours",
        "ourselves",
        "she",
        "their",
        "theirs",
        "them",
        "themselves",
        "they",
        "we",
        "you",
        "your",
        "yours",
        "yourself",
        "yourselves",
    },
)

#: Unified set of canonical Lucene stopwords and conversational pronouns.
FUNCTION_WORDS = LUCENE_STOPWORDS | PRONOUNS


class Leak(BaseModel):
    """Record whether and how a query explicitly references its target skill."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    names_target: bool = False
    distinctive_tokens: tuple[str, ...] = ()

    @property
    def leaked(self) -> bool:
        """Return True if query names target skill or has distinctive tokens."""
        return self.names_target or bool(self.distinctive_tokens)

    @property
    def routes(self) -> tuple[str, ...]:
        """Return diagnostic strings explaining why this probe was flagged as leaking."""
        routes: list[str] = []
        if self.names_target:
            routes.append("names target")
        if self.distinctive_tokens:
            routes.append(", ".join(self.distinctive_tokens))
        return tuple(routes)


def contains_run(haystack: Sequence[str], needle: Sequence[str]) -> bool:
    """Return True if needle appears as a contiguous subsequence in haystack."""
    if not needle:
        return False
    needle_tuple = tuple(needle)
    span = len(needle_tuple)
    return any(
        tuple(haystack[start : start + span]) == needle_tuple
        for start in range(len(haystack) - span + 1)
    )


def distinctive_tokens(
    skills: Sequence[Skill],
    background: Sequence[Skill] | None = None,
    floor: float = BACKGROUND_IDF,
    function_words: frozenset[str] = FUNCTION_WORDS,
) -> dict[str, frozenset[str]]:
    """Map each skill to unique description terms absent from rival surfaces."""
    surfaces = {s.name: frozenset(tokenize(skill_text(s))) for s in skills}
    scorer = None if background is None else Bm25Scorer.from_skills(background)

    def informative(term: str) -> bool:
        """Return True if term meets background IDF threshold."""
        return scorer is None or scorer.idf(term) > floor

    return {
        skill.name: frozenset(
            term
            for term in frozenset(tokenize(skill.description)).difference(
                function_words,
                *(tokens for name, tokens in surfaces.items() if name != skill.name),
            )
            if informative(term)
        )
        for skill in skills
    }


def leak_check(
    query: Query,
    skills: Sequence[Skill],
    distinctive: Mapping[str, frozenset[str]] | None = None,
    skills_by_name: Mapping[str, Skill] | None = None,
) -> Leak | None:
    """Evaluate target name and distinctive term leakage for a single query."""
    if query.expected_skill is None:
        return None
    if skills_by_name is not None:
        target = skills_by_name.get(query.expected_skill)
    else:
        target = next((s for s in skills if s.name == query.expected_skill), None)
    if target is None:
        return None
    tokens = tokenize(query.text)
    table = distinctive if distinctive is not None else distinctive_tokens(skills)
    return Leak(
        names_target=contains_run(tokens, tokenize(target.name)),
        distinctive_tokens=tuple(
            sorted(set(tokens) & table.get(target.name, frozenset())),
        ),
    )


def leaks(
    queries: Iterable[Query],
    skills: Sequence[Skill],
    background: Sequence[Skill] | None = None,
    function_words: frozenset[str] = FUNCTION_WORDS,
) -> dict[str, Leak]:
    """Map query IDs to Leak evaluations for all resident target skills."""
    skills_by_name = {s.name: s for s in skills}
    table = distinctive_tokens(skills, background, function_words=function_words)
    results: dict[str, Leak] = {}
    for query in queries:
        leak = leak_check(
            query,
            skills,
            distinctive=table,
            skills_by_name=skills_by_name,
        )
        if leak is not None:
            results[query.id] = leak
    return results
