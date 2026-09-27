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

"""Verify property-based invariants for statistical intervals, BM25 scoring, and serialization."""

from __future__ import annotations

from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

from reach.exchange import Exchange, export_query_set, import_query_set
from reach.models import Query, Skill
from reach.queries import Origin, QuerySet, QuerySetProvenance
from reach.retrieval import Bm25Scorer
from reach.uncertainty import wilson_interval


@st.composite
def hits_and_probes(draw: st.DrawFn) -> tuple[int, int]:
    """Generate valid (hits, probes) counts where probes >= 1 and hits <= probes."""
    probes = draw(st.integers(min_value=1, max_value=2000))
    hits = draw(st.integers(min_value=0, max_value=probes))
    return hits, probes


@given(hits_and_probes(), st.floats(min_value=0.5, max_value=0.99))
def test_the_interval_always_bounds_a_rate(
    counts: tuple[int, int],
    confidence: float,
) -> None:
    """Verify Wilson intervals fall strictly within [0, 1] with lower <= upper."""
    hits, probes = counts
    interval = wilson_interval(hits, probes, confidence)

    assert interval is not None
    assert 0.0 <= interval.low <= interval.high <= 1.0
    assert interval.width >= 0.0


@given(hits_and_probes(), st.floats(min_value=0.5, max_value=0.99))
def test_doubling_the_evidence_at_a_fixed_rate_does_not_widen_the_interval(
    counts: tuple[int, int],
    confidence: float,
) -> None:
    """Verify doubling sample size at constant hit rate does not increase interval width."""
    hits, probes = counts
    shallow = wilson_interval(hits, probes, confidence)
    deep = wilson_interval(2 * hits, 2 * probes, confidence)

    assert shallow is not None
    assert deep is not None
    assert deep.width <= shallow.width + 1e-9


#: Restricted alphabet to ensure generated skills share overlapping terms.
_TERM = st.text(alphabet="abcdefgh", min_size=1, max_size=4)


def _skill(name: str, description: str) -> Skill:
    """Build in-memory Skill instance without reading filesystem."""
    return Skill(name=name, description=description, path=Path(name))


@st.composite
def skill_corpora(draw: st.DrawFn) -> list[Skill]:
    """Generate synthetic skill corpus with unique skill names."""
    names = draw(
        st.lists(
            st.text(alphabet="abcdefgh", min_size=1, max_size=6),
            min_size=1,
            max_size=8,
            unique=True,
        ),
    )
    descriptions = draw(
        st.lists(
            st.lists(_TERM, min_size=1, max_size=6).map(" ".join),
            min_size=len(names),
            max_size=len(names),
        ),
    )
    return [
        _skill(name, description) for name, description in zip(names, descriptions, strict=True)
    ]


@settings(max_examples=35)
@given(skill_corpora(), st.lists(_TERM, max_size=6))
def test_bm25_scores_are_never_negative(skills: list[Skill], query: list[str]) -> None:
    """Verify BM25 scores are non-negative across all skills and queries."""
    scorer = Bm25Scorer.from_skills(skills)

    for skill in skills:
        assert scorer.score(query, skill.name) >= 0.0


@settings(max_examples=35)
@given(skill_corpora(), st.data())
def test_bm25_ranking_does_not_depend_on_input_order(
    skills: list[Skill],
    data: st.DataObject,
) -> None:
    """Verify BM25 ranking is invariant under arbitrary skill corpus permutations."""
    target = skills[0]
    shuffled = data.draw(st.permutations(skills))

    ranked = Bm25Scorer.from_skills(skills).rank(target, skills)
    reranked = Bm25Scorer.from_skills(shuffled).rank(target, shuffled)

    assert [name for name, _ in ranked] == [name for name, _ in reranked]
    for (_, score), (_, reordered_score) in zip(ranked, reranked, strict=True):
        assert score == reordered_score


#: Punctuation and whitespace characters for fuzzing query text serialization.
_AWKWARD = "\n\r\t,\"'\u200b"
_TEXT_CHAR = st.one_of(
    st.characters(
        min_codepoint=0x20,
        max_codepoint=0x10FFFF,
        exclude_categories=("Cs", "Co"),
    ),
    st.sampled_from(_AWKWARD),
)
_QUERY_TEXT = st.text(alphabet=_TEXT_CHAR, min_size=1, max_size=80).filter(
    lambda text: text.strip(),
)


@given(fmt=st.sampled_from(list(Exchange)), text=_QUERY_TEXT)
def test_export_then_import_reproduces_the_query_text_exactly(
    fmt: Exchange,
    text: str,
) -> None:
    """Verify query text round-trips exactly through export and import across all formats."""
    original = QuerySet(
        catalog_id="all",
        queries=(Query(id="q-1", text=text, expected_skill="some-skill"),),
        provenance=QuerySetProvenance(origin=Origin.AUTHORED),
    )

    document = export_query_set(original, fmt)
    imported = import_query_set(document, fmt, catalog_id="all")

    assert imported.queries[0].text == text
