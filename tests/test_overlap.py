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

"""Verify BM25 ranking, IDF calculation, and technical tokenization."""

from __future__ import annotations

from math import log
from typing import TYPE_CHECKING

import pytest

from reach.overlap import Competition, Rival
from reach.retrieval import Bm25Scorer, skill_text, tokenize

if TYPE_CHECKING:
    from conftest import SyntheticCorpusBuilder

    from reach.models import Skill


@pytest.fixture
def tiny_corpus(corpus_builder: type[SyntheticCorpusBuilder]) -> list[Skill]:
    """Provide a small three-skill corpus for term-level overlap verification."""
    return (
        corpus_builder()
        .add("alpha-one", "alpha alpha beta")
        .add("beta-two", "beta gamma")
        .add("gamma-three", "gamma gamma gamma delta")
        .build_skills()
    )


@pytest.mark.parametrize(
    "query",
    [["alpha"], ["beta"], ["gamma"], ["alpha", "beta"], ["absent"]],
)
def test_tiny_corpus_scores_match_bm25s(tiny_corpus, bm25s_reference, query) -> None:
    """Verify BM25 scores match reference implementation on test corpus."""
    scorer = Bm25Scorer.from_skills(tiny_corpus)
    ours = [scorer.score(query, s.name) for s in tiny_corpus]
    assert ours == pytest.approx(bm25s_reference(tiny_corpus)(query), abs=1e-4)


def test_ranking_excludes_the_target_and_is_ordered(tiny_corpus) -> None:
    """Verify BM25 ranking excludes target skill and sorts scores in descending order."""
    scorer = Bm25Scorer.from_skills(tiny_corpus)
    ranked = scorer.rank(tiny_corpus[0], tiny_corpus)
    assert [n for n, _ in ranked] == ["beta-two", "gamma-three"]
    assert [v for _, v in ranked] == sorted((v for _, v in ranked), reverse=True)


def test_ties_break_on_name_so_runs_reproduce(corpus_builder) -> None:
    """Verify tied BM25 scores break alphabetically by skill name."""
    skills = (
        corpus_builder()
        .add("target", "shared vocabulary")
        .add("zulu", "shared vocabulary")
        .add("alpha", "shared vocabulary")
        .build_skills()
    )
    scorer = Bm25Scorer.from_skills(skills)
    ranked = scorer.rank(skills[0], skills)
    assert [n for n, _ in ranked] == ["alpha", "zulu"]
    assert ranked[0][1] == pytest.approx(ranked[1][1])


def test_document_frequency_counts_documents_not_occurrences(tiny_corpus) -> None:
    """Verify document frequency counts distinct documents rather than total term count."""
    scorer = Bm25Scorer.from_skills(tiny_corpus)
    naive = {
        term: sum(1 for tokens in scorer.documents.values() if term in tokens)
        for term in ("alpha", "beta", "gamma", "absent")
    }
    assert naive == {"alpha": 1, "beta": 2, "gamma": 2, "absent": 0}
    for term, df in naive.items():
        expected = log(1 + (len(scorer.documents) - df + 0.5) / (df + 0.5))
        assert scorer.idf(term) == pytest.approx(expected), term


def test_idf_is_available_on_a_directly_constructed_scorer() -> None:
    """Verify IDF values compute correctly on direct Bm25Scorer dictionary instantiation."""
    scorer = Bm25Scorer(documents={"a": ("x", "y"), "b": ("y",)})
    assert scorer.idf("y") < scorer.idf("x"), "the rarer term must score higher"
    assert scorer.idf("y") == pytest.approx(log(1 + 0.5 / 2.5))


def test_scorer_handles_an_empty_and_a_single_skill_corpus(corpus_builder) -> None:
    """Verify Bm25Scorer handles empty and singleton skill lists safely."""
    empty = Bm25Scorer.from_skills([])
    assert empty.average_length == 0.0
    assert empty.idf("anything") == 0.0
    solo = corpus_builder().add("solo", "text").build_skills()[0]
    assert empty.rank(solo, []) == []

    lone = corpus_builder().add("solo", "some text").build_skills()[0]
    assert Bm25Scorer.from_skills([lone]).rank(lone, [lone]) == []


def test_unknown_document_scores_zero(tiny_corpus) -> None:
    """Verify score returns 0.0 for unindexed document names."""
    assert Bm25Scorer.from_skills(tiny_corpus).score(["alpha"], "not-indexed") == 0.0


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("GKE Autopilot", ["gke", "autopilot"]),
        ("bigquery-ai-ml", ["bigquery", "ai", "ml"]),
        ("Cloud Run (v2)!", ["cloud", "run", "v2"]),
        ("", []),
    ],
)
def test_tokenizer_splits_the_product_vocabulary(text, expected) -> None:
    """Verify tokenize correctly splits hyphenated and parenthesized product names."""
    assert tokenize(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("café résumé", ["café", "résumé"]),
        ("naïve Zürich piñata", ["naïve", "zürich", "piñata"]),
        ("C++ and C#", ["c++", "and", "c#"]),
        ("ASP.NET and Node.js", ["asp.net", "and", "node.js"]),
        ("v2.0.1", ["v2.0.1"]),
    ],
)
def test_tokenizer_keeps_accented_letters_and_technical_marks_intact(text, expected) -> None:
    """Verify tokenize preserves accented Unicode characters and technical symbols."""
    assert tokenize(text) == expected


def test_skill_text_includes_the_name(tiny_corpus) -> None:
    """Verify skill_text prefixes description with skill name."""
    assert skill_text(tiny_corpus[0]).startswith("alpha-one")


def test_contributions_sum_matches_score(tiny_corpus) -> None:
    """Verify sum of term contributions equals overall BM25 score."""
    scorer = Bm25Scorer.from_skills(tiny_corpus)
    query = ["alpha", "beta"]
    score = scorer.score(query, "alpha-one")
    contribs = scorer.contributions(query, "alpha-one")
    assert sum(contribs.values()) == pytest.approx(score)


def test_empty_query_scores_zero_and_empty_contributions(tiny_corpus) -> None:
    """Verify empty query returns zero score and empty contributions."""
    scorer = Bm25Scorer.from_skills(tiny_corpus)
    assert scorer.score([], "alpha-one") == 0.0
    assert scorer.contributions([], "alpha-one") == {}


def test_rank_text_matches_dense_computation(corpus_builder) -> None:
    """Verify sparse postings lookup produces exact identical results to dense scoring."""
    cb = corpus_builder()
    for i in range(25):
        cb.add(f"skill-{i}", f"word-{i % 5} common-word domain-{i // 5}")
    corpus = cb.build_skills()
    scorer = Bm25Scorer.from_skills(corpus)

    ranked = scorer.rank_text("word-2 query specific", corpus)
    dense_expected = sorted(
        [(c.name, scorer.score(tokenize("word-2 query specific"), c.name)) for c in corpus],
        key=lambda pair: (-pair[1], pair[0]),
    )
    assert ranked == dense_expected


def test_competition_presorts_and_caches_ranked_rivals() -> None:
    """Verify Competition pre-sorts rivals on initialization and provides O(1) access."""
    rivals = (
        Rival(name="rival-b", score=5.0),
        Rival(name="rival-a", score=10.0),
        Rival(name="rival-c", score=5.0),
    )
    comp = Competition(skill="target", self_score=8.0, rivals=rivals)

    assert [r.name for r in comp.ranked_rivals] == ["rival-a", "rival-b", "rival-c"]
    assert comp.nearest_rival == Rival(name="rival-a", score=10.0)
    assert comp.scoring_rival == Rival(name="rival-a", score=10.0)
    assert comp.rival_ratio == pytest.approx(10.0 / 8.0)
    assert comp.outranked_by == 1


def test_competition_properties_empty_rivals() -> None:
    """Verify Competition handles an empty rivals list safely."""
    comp = Competition(skill="target", self_score=10.0, rivals=())
    assert comp.ranked_rivals == ()
    assert comp.nearest_rival is None
    assert comp.scoring_rival is None
    assert comp.rival_ratio == 0.0
    assert comp.outranked_by == 0
