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

"""Verify query leak detection against skill names, distinctive tokens, and stopwords."""

from __future__ import annotations

import pytest

from reach.generate import GeneratedQuery, verify_citation
from reach.leak import FUNCTION_WORDS, Leak, contains_run, distinctive_tokens, leak_check, leaks
from reach.models import Query, QueryKind, Skill
from reach.retrieval import tokenize


def make_query(text: str, expected: str | None = "bucket-lifecycle") -> Query:
    """Build a labeled query around the text under test."""
    return Query(
        id="q",
        text=text,
        kind=QueryKind.OUT_OF_SCOPE if expected is None else QueryKind.IMPLICIT,
        expected_skill=expected,
    )


@pytest.fixture
def skills(make_skill) -> list[Skill]:
    """Provide three skills with disjoint domain vocabulary."""
    return [
        make_skill("bucket-lifecycle", "Tier cold objects between storage classes."),
        make_skill("cluster-upgrade", "Roll nodes onto a newer control plane."),
        make_skill("billing-export", "Send invoices into a warehouse dataset."),
    ]


@pytest.mark.parametrize(
    ("text", "names_target", "tokens"),
    [
        pytest.param(
            "Use the bucket-lifecycle skill for this.",
            True,
            (),
            id="names-the-skill-outright",
        ),
        pytest.param(
            "Please run bucket lifecycle on my data.",
            True,
            (),
            id="names-the-skill-unpunctuated",
        ),
        pytest.param(
            "Tier my cold objects.",
            False,
            ("cold", "objects", "tier"),
            id="quotes-the-description",
        ),
        pytest.param(
            "Roll nodes onto a newer control plane.",
            False,
            (),
            id="speaks-a-rival-vocabulary",
        ),
        pytest.param("Quel temps fait-il demain?", False, (), id="shares-nothing"),
    ],
)
def test_leak_reports_how_a_query_gives_itself_away(
    skills: list[Skill],
    text: str,
    names_target: bool,
    tokens: tuple[str, ...],
) -> None:
    """Verify leak_check flags direct name references and distinctive vocabulary tokens."""
    leak = leak_check(make_query(text), skills)
    assert leak is not None
    assert leak == Leak(names_target=names_target, distinctive_tokens=tokens)
    assert leak.leaked is (names_target or bool(tokens))


def test_a_shared_word_is_not_distinctive(corpus: list[Skill]) -> None:
    """Verify vocabulary shared across multiple skills is not marked as distinctive."""
    query = make_query("Configures the thing.", "gcs-lifecycle-rules")
    assert leak_check(query, corpus) == Leak()


def test_a_term_in_both_the_name_and_the_description_trips_both_routes(
    make_skill,
) -> None:
    """Verify term occurring in both name and description triggers both leak flags."""
    skills = [
        make_skill("bucket-lifecycle", "Tier objects through the lifecycle."),
        make_skill("cluster-upgrade", "Roll the nodes onto a newer control plane."),
    ]
    assert leak_check(make_query("Use bucket-lifecycle now."), skills) == Leak(
        names_target=True,
        distinctive_tokens=("lifecycle",),
    )
    assert leak_check(make_query("Something about the lifecycle."), skills) == Leak(
        distinctive_tokens=("lifecycle",),
    )


def test_every_word_is_distinctive_to_a_sole_resident(make_skill) -> None:
    """Verify non-stopword tokens are treated as distinctive in singleton catalog."""
    only = [make_skill("bucket-lifecycle", "Tier cold objects.")]
    assert leak_check(make_query("Tier the things."), only) == Leak(
        distinctive_tokens=("tier",),
    )


def test_a_name_with_no_words_in_it_is_named_by_nothing(make_skill) -> None:
    """Verify punctuation-only skill names do not cause false positive leak detections."""
    unnameable = [make_skill("++", "Tier cold objects between storage classes.")]
    leak = leak_check(make_query("Anything at all.", "++"), unnameable)
    assert leak == Leak(names_target=False, distinctive_tokens=())


def test_a_query_with_no_resident_target_has_no_leak(skills: list[Skill]) -> None:
    """Verify leak_check returns None for out-of-scope or unindexed queries."""
    assert leak_check(make_query("Book me a flight.", None), skills) is None
    assert leak_check(make_query("Anything.", "not-here"), skills) is None


def test_leaks_skip_the_queries_with_nothing_to_check(skills: list[Skill]) -> None:
    """Verify leaks batch function filters out queries with no target skill."""
    queries = [
        Query(
            id="leaky",
            text="Tier my cold objects.",
            kind=QueryKind.IMPLICIT,
            expected_skill="bucket-lifecycle",
        ),
        Query(id="abstain", text="Book me a flight.", kind=QueryKind.OUT_OF_SCOPE),
        Query(
            id="stray",
            text="Something else.",
            kind=QueryKind.IMPLICIT,
            expected_skill="not-here",
        ),
    ]
    assert leaks(queries, skills) == {
        "leaky": Leak(distinctive_tokens=("cold", "objects", "tier")),
    }


def test_batched_leaks_match_the_one_at_a_time_computation(
    skills: list[Skill],
) -> None:
    """Verify leaks batch result matches individual leak_check calls."""
    queries = [
        Query(
            id=f"q{i}",
            text=text,
            kind=QueryKind.IMPLICIT,
            expected_skill="bucket-lifecycle",
        )
        for i, text in enumerate(
            [
                "Tier my cold objects.",
                "Use bucket-lifecycle.",
                "Roll nodes onto a newer control plane.",
                "Quel temps fait-il demain?",
            ],
        )
    ]
    assert leaks(queries, skills) == {q.id: leak_check(q, skills) for q in queries}


def test_a_term_the_background_corpus_spends_freely_does_not_leak(
    skills: list[Skill],
    make_skill,
) -> None:
    """Verify frequent background corpus terms are filtered out of distinctive tokens."""
    arena = [
        make_skill("bucket-lifecycle", "Use this when you tier cold objects."),
        *skills[1:],
    ]
    background = [
        *arena,
        *(make_skill(f"filler-{i}", "Use this when you need something else.") for i in range(6)),
    ]
    query = make_query("Tell me when to act.")
    assert leak_check(query, arena) == Leak(distinctive_tokens=("when",))
    assert leak_check(query, arena, distinctive_tokens(arena, background)) == Leak()


def test_a_rare_term_survives_the_background(skills: list[Skill], make_skill) -> None:
    """Verify rare distinctive terms remain flagged when background corpus is provided."""
    background = [
        *skills,
        *(make_skill(f"filler-{i}", "Something wholly unrelated.") for i in range(6)),
    ]
    table = distinctive_tokens(skills, background)
    assert leak_check(make_query("Tier my cold objects."), skills, table) == Leak(
        distinctive_tokens=("cold", "objects", "tier"),
    )


def test_masking_the_description_does_not_prevent_a_leak(make_skill) -> None:
    """Verify generated queries using body citations can still trigger description leaks."""
    body = "Tier cold objects between storage classes once they go untouched."
    skills = [
        make_skill("bucket-lifecycle", "Tier cold objects between storage classes."),
        make_skill("cluster-upgrade", "Roll nodes onto a newer control plane."),
    ]
    drafted = GeneratedQuery(
        text="Our bucket is full of cold objects nobody has touched in a year.",
        citation="Tier cold objects between storage classes",
    )
    assert verify_citation(drafted, body), "the citation gate passes this draft"
    leak = leak_check(make_query(drafted.text), skills)
    assert leak is not None
    assert not leak.names_target
    assert leak.leaked, "and the draft still hands over the description's own terms"
    assert set(leak.distinctive_tokens) >= {"cold", "objects"}


def test_a_first_person_pronoun_never_leaks(make_skill) -> None:
    """Verify first-person pronouns in queries are excluded from distinctive tokens."""
    skills = [
        make_skill("bucket-lifecycle", "Tier your cold objects between classes."),
        make_skill("cluster-upgrade", "Roll nodes onto a newer control plane."),
    ]
    leak = leak_check(make_query("Tier our cold objects, would you."), skills)
    assert leak is not None
    assert set(leak.distinctive_tokens) == {"cold", "objects", "tier"}


#: Retained function words absent from scikit-learn's default stop word list.
UNPUBLISHED = {"theirs"}


@pytest.mark.parametrize("word", sorted(FUNCTION_WORDS - UNPUBLISHED))
def test_no_held_out_function_word_is_domain_vocabulary(word: str) -> None:
    """Verify FUNCTION_WORDS elements are present in scikit-learn English stop words."""
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    assert word in ENGLISH_STOP_WORDS


def test_the_one_pronoun_no_published_list_backs_is_named_not_quietly_kept() -> None:
    """Verify UNPUBLISHED word theirs is in FUNCTION_WORDS but absent in sklearn stop words."""
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    assert UNPUBLISHED <= FUNCTION_WORDS
    assert not UNPUBLISHED & set(ENGLISH_STOP_WORDS)
    assert {"mine", "ours", "yours", "hers"} <= set(ENGLISH_STOP_WORDS)


def test_the_two_words_a_published_list_would_have_cost_us_are_held_out() -> None:
    """Verify 'i' and 'us' are preserved as valid tokens and excluded from FUNCTION_WORDS."""
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    assert {"i", "us"} <= set(ENGLISH_STOP_WORDS)
    assert not {"i", "us"} & FUNCTION_WORDS
    assert tokenize("us-central1") == ["us", "central1"]
    assert tokenize("I/O") == ["i", "o"]


@pytest.mark.parametrize("word", ["them", "they", "their", "its", "it"])
def test_a_third_person_pronoun_is_a_function_word_too(make_skill, word: str) -> None:
    """Verify third-person pronouns are excluded from distinctive token tables."""
    skills = [
        make_skill("bucket-lifecycle", f"Tier cold objects, and expire {word} later."),
        make_skill("cluster-upgrade", "Roll nodes onto a newer control plane."),
    ]
    assert word not in distinctive_tokens(skills)["bucket-lifecycle"]


def test_a_function_word_is_dropped_before_the_background_is_consulted(
    make_skill,
) -> None:
    """Verify function words are filtered out even when no background corpus is given."""
    skills = [
        make_skill("bucket-lifecycle", "Tell us what you need for your objects."),
        make_skill("cluster-upgrade", "Roll nodes onto a newer control plane."),
    ]
    table = distinctive_tokens(skills)
    assert "your" not in table["bucket-lifecycle"]
    assert "you" not in table["bucket-lifecycle"]
    assert "us" in table["bucket-lifecycle"], "held out; see the test above"


def test_a_study_supplies_its_own_function_words_rather_than_editing_the_tool(
    make_skill,
) -> None:
    """Verify caller can provide custom function_words set to override defaults."""
    skills = [
        make_skill("bucket-lifecycle", "Tell us what you need for your objects."),
        make_skill("cluster-upgrade", "Roll nodes onto a newer control plane."),
    ]
    whole_class = FUNCTION_WORDS | {"i", "us"}
    table = distinctive_tokens(skills, function_words=whole_class)
    assert "us" not in table["bucket-lifecycle"]
    assert "objects" in table["bucket-lifecycle"], "only the pronoun moved"

    flagged = leaks(
        [make_query("Us and our objects, please?")],
        skills,
        function_words=whole_class,
    )
    assert flagged["q"].distinctive_tokens == ("objects",), "and leaks passes it on"


def test_distinctive_tokens_exclude_every_rival_surface(
    skills: list[Skill],
    make_skill,
) -> None:
    """Verify terms in rival skill names are excluded from distinctive tokens."""
    rival_named = [
        *skills,
        make_skill("objects-api", "Talk to the REST endpoint."),
    ]
    table = distinctive_tokens(rival_named)
    assert "objects" not in table["bucket-lifecycle"]
    assert "cold" in table["bucket-lifecycle"]


def test_leak_routes_diagnose_target_and_distinctive_tokens() -> None:
    """Verify Leak.routes provides formatted diagnostic strings for clean and leaking probes."""
    clean = Leak()
    assert clean.routes == ()
    assert not clean.leaked

    target_leak = Leak(names_target=True)
    assert target_leak.routes == ("names target",)
    assert target_leak.leaked

    token_leak = Leak(distinctive_tokens=("foo", "bar"))
    assert token_leak.routes == ("foo, bar",)
    assert token_leak.leaked

    both_leak = Leak(names_target=True, distinctive_tokens=("foo", "bar"))
    assert both_leak.routes == ("names target", "foo, bar")
    assert both_leak.leaked


@pytest.mark.parametrize(
    ("haystack", "needle", "expected"),
    [
        (("a", "b", "c", "d"), ("b", "c"), True),
        (("a", "b", "c", "d"), ["b", "c"], True),
        (["a", "b", "c", "d"], ("b", "c"), True),
        (["a", "b", "c", "d"], ["b", "c"], True),
        (("a", "b"), ("a", "b"), True),
        (("a", "b"), ("b", "a"), False),
        (("a", "b"), (), False),
        ((), ("a",), False),
        (("a", "b"), ("a", "b", "c"), False),
        (("token_a", "token_b"), ("token_a", "token_b"), True),
    ],
)
def test_contains_run_tuples_and_lists(
    haystack: tuple[str, ...] | list[str],
    needle: tuple[str, ...] | list[str],
    expected: bool,
) -> None:
    """Verify contains_run operates correctly with tuples and lists across sequences."""
    assert contains_run(haystack, needle) is expected


def test_leak_check_with_skills_by_name_and_none_expected(skills: list[Skill]) -> None:
    """Verify leak_check handles precomputed skills_by_name mapping and None expected_skill."""
    skills_by_name = {s.name: s for s in skills}

    query_with_target = make_query("Tier my cold objects.", "bucket-lifecycle")
    leak = leak_check(query_with_target, skills, skills_by_name=skills_by_name)
    assert leak is not None
    assert "cold" in leak.distinctive_tokens

    query_without_target = make_query("Out of scope query.", None)
    assert leak_check(query_without_target, skills, skills_by_name=skills_by_name) is None


def test_lucene_stopwords_match_bm25s() -> None:
    """Verify LUCENE_STOPWORDS matches canonical Apache Lucene / bm25s.stopwords.STOPWORDS_EN."""
    from bm25s.stopwords import STOPWORDS_EN

    from reach.leak import LUCENE_STOPWORDS, PRONOUNS

    expected_lucene = frozenset(STOPWORDS_EN)
    assert expected_lucene == LUCENE_STOPWORDS
    assert FUNCTION_WORDS == LUCENE_STOPWORDS | PRONOUNS


@pytest.mark.parametrize(
    "stopword",
    [
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
        "no",
        "not",
        "of",
        "on",
        "or",
        "such",
        "that",
        "the",
        "then",
        "there",
        "these",
        "this",
        "to",
        "was",
        "will",
        "with",
    ],
)
def test_lucene_stopwords_never_leak_as_distinctive_tokens(
    make_skill,
    stopword: str,
) -> None:
    """Verify canonical Lucene stopwords are in FUNCTION_WORDS and never leak."""
    assert stopword in FUNCTION_WORDS
    corpus = [
        make_skill("target-skill", f"Configure {stopword} storage bucket."),
        make_skill("rival-skill", "Roll nodes onto control plane."),
    ]
    table = distinctive_tokens(corpus)
    assert stopword not in table["target-skill"]
