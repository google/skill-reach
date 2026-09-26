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

"""Verify rewrite suggestions, ceded term calculations, and refusal logic."""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Callable
from typing import TYPE_CHECKING

import pytest

from reach.catalog import load_skills
from reach.cli import main
from reach.overlap import (
    Competition,
    Rival,
    rank_corpus,
)
from reach.retrieval import (
    Bm25Scorer,
    skill_text,
)
from reach.retrieval import tokenize as split
from reach.rewrite import (
    REWRITE_CAVEAT,
    CededTerm,
    Rewrite,
    Verdict,
    ceded_terms,
    skill_body,
    suggest_all,
    suggest_rewrite,
    unclaimed_terms,
)
from reach.runtime import SkillRoot
from reach.views import print_rewrite, render_rewrite, suggest_view

if TYPE_CHECKING:
    from pathlib import Path

    from reach.models import Skill


def emitted(argv: list[str], capsys) -> str:
    """Execute CLI command and return captured stdout output."""
    assert main(argv) == 0
    return capsys.readouterr().out


#: Sample description lacking rival disambiguation.
BLUNT = "Explain widget fundamentals, including rollout planning."

#: Sample description containing explicit rival disambiguation.
HEDGED = "Explain widget fundamentals. For rollout planning use widget-rollout."

#: Sample skill markdown body with distinct repeated vocabulary.
BODY = """
Guidance on staging. A staging environment mirrors production closely.
Staging catches drift before rollout. Use canary cohorts; canary first.
"""


@pytest.fixture
def bystanders(corpus_builder) -> list[Skill]:
    """Provide five non-competing skills to establish baseline corpus size."""
    return (
        corpus_builder()
        .add("gadget-tuning", "Tune gadget throughput for busy pipelines.")
        .add("ledger-audit", "Reconcile ledger entries against statements.")
        .add("mailer-templates", "Author transactional mail templates.")
        .add("photo-resize", "Crop photographs to a target aspect ratio.")
        .add("query-planner", "Report how the planner picks a strategy.")
        .build_skills()
    )


#: Rival skill description for testing.
RIVAL = "Plan and execute a widget rollout across fleets."


@pytest.fixture
def make_corpus(corpus_builder, bystanders: list[Skill]) -> Callable[..., list[Skill]]:
    """Provide a factory creating a corpus with a competing skill pair and background skills."""

    def build(
        description: str = BLUNT,
        name: str = "widget-basics",
        rival: str = RIVAL,
    ) -> list[Skill]:
        """Build the corpus with the pair described as asked."""
        pair = corpus_builder().add(name, description).add("widget-rollout", rival).build_skills()
        return [*pair, *bystanders]

    return build


@pytest.fixture
def suggest(make_corpus):
    """Return a factory that ranks a corpus and suggests against one skill of it."""

    def build(
        description: str = BLUNT,
        name: str = "widget-basics",
        body: str = BODY,
        rival: str = RIVAL,
        **options,
    ) -> Rewrite:
        """Suggest for the named skill of a corpus described as asked."""
        corpus = make_corpus(description, name, rival)
        overlap = rank_corpus(corpus)
        return suggest_rewrite(overlap.find(name), corpus, body=body, **options)

    return build


@pytest.fixture
def shown(make_console, rendered) -> Callable[..., str]:
    """Return what `print_rewrite` draws for one suggestion."""

    def render(rewrite: Rewrite, width: int = 100) -> str:
        """Draw the suggestion into a buffer of a fixed width and read it back."""
        console, buffer = make_console(terminal=False, width=width)
        print_rewrite(console, rewrite)
        assert isinstance(buffer, io.StringIO)
        return rendered(buffer)

    return render


@pytest.mark.parametrize(
    "document",
    ["widget-basics", "widget-rollout", "gadget-tuning"],
)
def test_the_itemization_sums_to_the_score_it_claims_to_explain(
    document: str,
    make_corpus,
) -> None:
    """Verify term contribution scores sum exactly to total BM25 document score."""
    corpus = make_corpus()
    scorer = Bm25Scorer.from_skills(corpus)
    query = split(skill_text(corpus[0]))

    itemized = scorer.contributions(query, document)

    assert sum(itemized.values()) == pytest.approx(scorer.score(query, document))


def test_a_term_the_query_repeats_is_itemized_as_often_as_it_is_asked(
    make_corpus,
) -> None:
    """Verify repeated query terms contribute proportionally to itemized score."""
    scorer = Bm25Scorer.from_skills(make_corpus())

    once = scorer.contributions(["widget"], "widget-rollout")
    twice = scorer.contributions(["widget", "widget"], "widget-rollout")

    assert twice["widget"] == pytest.approx(2 * once["widget"])


def test_a_term_the_document_does_not_carry_is_absent_rather_than_zero(
    make_corpus,
) -> None:
    """Verify non-occurring query terms are excluded from contributions dict."""
    scorer = Bm25Scorer.from_skills(make_corpus())

    itemized = scorer.contributions(["widget", "ledger"], "widget-rollout")

    assert "ledger" not in itemized
    assert itemized["widget"] > 0


def test_an_unknown_document_itemizes_to_nothing(make_corpus) -> None:
    """Verify contributions returns empty dictionary for unindexed document names."""
    scorer = Bm25Scorer.from_skills(make_corpus())
    assert scorer.contributions(["widget"], "absent") == {}


def test_a_description_spending_its_rival_s_noun_is_told_which_noun(suggest) -> None:
    """Verify suggest_rewrite identifies specific ceded nouns and outputs REWORD verdict."""
    rewrite = suggest()

    assert rewrite.verdict is Verdict.REWORD
    assert rewrite.rival == "widget-rollout"
    assert [term.term for term in rewrite.reword] == ["rollout"]


def test_the_share_is_a_fraction_of_the_rival_s_own_answer(suggest) -> None:
    """Verify ceded term share is normalized to a value between 0 and 1."""
    ceded = suggest().reword[0]

    assert 0 < ceded.share < 1


def test_the_ceded_terms_come_out_strongest_first(corpus_builder, bystanders) -> None:
    """Verify ceded terms are sorted in descending order by share."""
    pair = (
        corpus_builder()
        .add("widget-basics", "Explain widget rollout planning and telemetry.")
        .add(
            "widget-rollout",
            "Plan a widget rollout, watch rollout telemetry, and stage telemetry.",
        )
        .build_skills()
    )
    corpus = [*pair, *bystanders]
    rewrite = suggest_rewrite(
        rank_corpus(corpus).find("widget-basics"),
        corpus,
        body="",
    )
    shares = [term.share for term in rewrite.ceded]

    assert len(shares) > 1
    assert shares == sorted(shares, reverse=True)


def test_a_term_only_the_name_carries_is_never_offered_as_an_edit(
    corpus_builder,
    bystanders,
) -> None:
    """Verify terms appearing only in the skill name are excluded from ceded terms."""
    pair = (
        corpus_builder()
        .add("rollout-notes", "Keep engineering notes about shipped work.")
        .add("widget-rollout", "Plan and execute a widget rollout for fleets.")
        .build_skills()
    )
    corpus = [*pair, *bystanders]
    rewrite = suggest_rewrite(
        rank_corpus(corpus).find("rollout-notes"),
        corpus,
        body="",
    )

    assert "rollout" not in [term.term for term in rewrite.ceded]


def test_a_term_below_the_material_share_is_left_off_the_list(suggest) -> None:
    """Verify terms falling below material share threshold are excluded."""
    rewrite = suggest(share=0.99)

    assert rewrite.verdict is Verdict.CONTESTED
    assert rewrite.ceded == ()


@pytest.mark.parametrize("pronoun", ["your", "you", "my"])
def test_a_pronoun_the_pair_shares_is_never_ceded(pronoun: str, suggest) -> None:
    """Verify shared pronouns are excluded from ceded terms."""
    rewrite = suggest(
        f"Explain {pronoun} widget fundamentals for {pronoun} rollout.",
        rival=f"Plan and execute {pronoun} widget rollout across {pronoun} fleets.",
    )

    assert rewrite.ceded != ()
    assert pronoun not in [term.term for term in rewrite.ceded]


def test_a_boundary_the_author_already_drew_is_not_reported_as_a_defect(
    suggest,
) -> None:
    """Verify disclaimed terms result in CONTESTED verdict without reword recommendations."""
    rewrite = suggest(HEDGED)

    assert rewrite.verdict is Verdict.CONTESTED
    assert [term.term for term in rewrite.disclaimed] == ["rollout"]
    assert rewrite.reword == ()


def test_a_term_disclaimed_once_and_spent_again_is_still_ceded(suggest) -> None:
    """Verify terms used again in non-disclaimed sentences remain flagged as ceded."""
    rewrite = suggest(
        "For rollout planning use widget-rollout. Rollout rehearsals included.",
        rival="Rollout, rollout, rollout: plan the widget rollout.",
    )

    assert rewrite.verdict is Verdict.REWORD
    assert "rollout" in [term.term for term in rewrite.reword]


def test_a_declined_suggestion_offers_no_vocabulary_to_claim(suggest) -> None:
    """Verify unclaimed terms tuple is empty when rewrite is declined."""
    assert suggest(HEDGED).unclaimed == ()


def test_a_skill_nothing_competes_with_is_answered_rather_than_refused(
    corpus_builder,
) -> None:
    """Verify non-competing skill receives UNRIVALED verdict."""
    lone = corpus_builder().add("solo", "The only skill installed.").build_skills()

    rewrite = suggest_rewrite(rank_corpus(lone).find("solo"), lone)

    assert rewrite.verdict is Verdict.UNRIVALED
    assert rewrite.rival == ""
    assert rewrite.ceded == ()


def test_a_rival_that_scored_nothing_is_not_named_as_one(corpus_builder) -> None:
    """Verify zero-scoring rivals do not trigger rival suggestions."""
    strangers = (
        corpus_builder()
        .add("solo", "Quantum entanglement lattice calibration.")
        .add("other", "Reconcile ledger entries against statements.")
        .build_skills()
    )
    rewrite = suggest_rewrite(rank_corpus(strangers).find("solo"), strangers)

    assert rewrite.verdict is Verdict.UNRIVALED


def test_a_corpus_too_small_for_the_floor_cedes_nothing(corpus_builder) -> None:
    """Verify small corpora below background threshold yield CONTESTED with no ceded terms."""
    tiny = (
        corpus_builder()
        .add("widget-basics", "Explain widget fundamentals and rollout planning.")
        .add("widget-rollout", "Plan and execute a widget rollout for fleets.")
        .add("ledger-audit", "Reconcile ledger entries against statements.")
        .build_skills()
    )
    rewrite = suggest_rewrite(rank_corpus(tiny).find("widget-basics"), tiny, body="")

    assert rewrite.verdict is Verdict.CONTESTED


def test_a_skill_the_corpus_does_not_carry_is_refused_by_name() -> None:
    """Verify suggest_rewrite raises ValueError when skill is missing from corpus."""
    contest = Competition(
        skill="absent",
        self_score=1.0,
        rivals=(Rival(name="other", score=0.5),),
    )
    with pytest.raises(ValueError, match="absent"):
        suggest_rewrite(contest, [])


def test_the_body_supplies_the_vocabulary_the_description_never_claimed(
    suggest,
) -> None:
    """Verify unclaimed terms extracts distinctive tokens from markdown body."""
    assert suggest().unclaimed == ("staging", "canary")


@pytest.mark.parametrize(
    ("body", "why"),
    [
        ("A drift report. Another drift report.", "the description already claims it"),
        ("Fleets are fleets.", "a rival's own surface carries it"),
        ("Mentioned once: kubernetes.", "one use is an aside rather than a subject"),
        ("The xy axis and the xy plane.", "too short to be a word a reader claims"),
        ("```python\nquiesce()\nquiesce()\n```", "a fenced block is code"),
        ("Call `quiesce` twice: `quiesce`.", "an inline span is code"),
        (
            "See https://quiesce.example and https://quiesce.example",
            "a URL is an address",
        ),
        (
            "[a](/quiesce/one.md) and [b](/quiesce/two.md)",
            "a link target is an address",
        ),
    ],
)
def test_a_term_that_is_not_this_skill_s_own_is_not_offered(
    body: str,
    why: str,
    suggest,
) -> None:
    """Verify non-distinctive, code, URL, or short tokens are excluded from unclaimed terms."""
    offered = suggest("Explain widget drift, including rollout planning.", body=body)

    assert offered.unclaimed == (), why


def test_the_claim_list_stops_where_the_reader_stops_reading(suggest) -> None:
    """Verify unclaimed terms are truncated to the specified limit."""
    assert suggest(limit=1).unclaimed == ("staging",)


def test_a_negative_limit_is_a_caller_error_rather_than_an_empty_list(
    make_corpus,
) -> None:
    """Verify negative candidate limit raises a ValueError."""
    corpus = make_corpus()
    with pytest.raises(ValueError, match="fewer than 0"):
        unclaimed_terms(
            corpus[0],
            corpus,
            BODY,
            Bm25Scorer.from_skills(corpus),
            limit=-1,
        )


def test_the_body_of_a_skill_on_disk_is_its_prose_and_not_its_frontmatter(
    skill_repo: Path,
) -> None:
    """Verify skill body extracts markdown prose without YAML frontmatter."""
    loaded = load_skills(skill_repo)
    body = skill_body(next(s for s in loaded if s.name == "gke-basics"))

    assert "description:" not in body
    assert "gke-basics" in body


def test_a_skill_with_no_file_behind_it_has_no_body_rather_than_an_error(
    corpus_builder,
) -> None:
    """Verify in-memory skill without backing file returns empty body string."""
    nowhere = corpus_builder().add("nowhere", "Built in memory, never written.")
    assert skill_body(nowhere.build_skills()[0]) == ""


# --- CLI suggestion output formatting ----------------------------------------


def test_the_page_leads_with_the_edit_and_names_both_skills(suggest, shown) -> None:
    """Verify printed suggestion displays skill name, rival name, ceded term, and unclaimed term."""
    page = shown(suggest())

    assert "widget-basics" in page
    assert "widget-rollout" in page
    assert "rollout" in page
    assert "staging" in page


def test_the_page_says_outright_that_no_wording_change_will_help(suggest, shown) -> None:
    """Verify printed page explicitly explains when no wording change is recommended."""
    page = shown(suggest(HEDGED))

    assert "no wording change is indicated" in page
    assert "already disclaimed" in page


def test_the_page_for_a_skill_nothing_competes_with_says_so(corpus_builder, shown) -> None:
    """Verify printed page explains when no rivals compete with target skill."""
    lone = corpus_builder().add("solo", "The only skill installed.").build_skills()
    page = shown(suggest_rewrite(rank_corpus(lone).find("solo"), lone))

    assert "nothing competes" in page


@pytest.mark.parametrize("description", [BLUNT, HEDGED])
def test_every_suggestion_repeats_that_nothing_was_measured(
    description: str,
    suggest,
    shown,
) -> None:
    """Verify suggestion output contains disclaimer stating no probes were issued."""
    page = shown(suggest(description))

    assert "no probe was issued" in page
    assert "reach eval" in page


def test_a_crowded_field_is_reported_beside_the_one_rival_that_was_named(
    corpus_builder,
    bystanders,
    shown,
) -> None:
    """Verify suggestion flags when multiple rival scores are within tenth of top rival."""
    trio = (
        corpus_builder()
        .add(
            "widget-basics",
            "Explain widget fundamentals, rollout and rollback planning.",
        )
        .add("widget-rollout", "Plan and execute a widget rollout across fleets.")
        .add("widget-rollback", "Plan and execute a widget rollback across fleets.")
        .build_skills()
    )
    crowd = [*trio, *bystanders]
    rewrite = suggest_rewrite(
        rank_corpus(crowd).find("widget-basics"),
        crowd,
        body=BODY,
    )

    assert rewrite.crowded
    rendered_rewrite = shown(rewrite)
    assert "1 more rival scores within 10% (widget-" in rendered_rewrite


def test_a_field_of_one_says_nothing_about_how_many_others_are_close(suggest, shown) -> None:
    """Verify single contender does not report crowded field warning."""
    rewrite = suggest()

    assert rewrite.contenders == ("widget-rollout",)
    assert "also within reach" not in shown(rewrite)


def test_the_terms_survive_a_terminal_too_narrow_for_the_line(suggest, shown) -> None:
    """Verify terms render completely when formatted for narrow terminal width."""
    page = shown(suggest(), width=44)

    assert "staging" in page
    assert "canary" in page


def test_the_page_never_writes_a_replacement_description(suggest, shown, snapshot) -> None:
    """Verify suggestion output matches expected snapshot without synthetic prose generation."""
    assert shown(suggest()) == snapshot


def test_the_verb_suggests_for_the_skill_it_was_pointed_at(skill_repo: Path, capsys) -> None:
    """Verify overlap --suggest executes successfully and prints suggestion for named skill."""
    exit_code = main(
        ["overlap", "--skills", str(skill_repo), "--skill", "gke-basics", "--suggest"],
    )
    shown_page = capsys.readouterr().err

    assert exit_code == 0
    assert "gke-basics" in shown_page
    assert "no probe was issued" in shown_page


def test_the_verb_suggests_across_a_whole_corpus_without_skill_flag(
    skill_repo: Path,
    capsys,
) -> None:
    """Verify overlap --suggest runs across the corpus when --skill argument is omitted."""
    assert main(["overlap", "--skills", str(skill_repo), "--suggest"]) == 0
    err = capsys.readouterr().err
    assert "No actionable rewrites across 3 skills" in err
    assert err.count("no probe was issued") == 1

    assert main(["overlap", "--skills", str(skill_repo), "--suggest", "--all"]) == 0
    err_all = capsys.readouterr().err
    assert "gke-basics" in err_all
    assert "gcs-lifecycle-rules" in err_all
    assert err_all.count("no probe was issued") == 1


def test_corpus_wide_suggest_filters_to_actionable_rewrites(
    tmp_path: Path,
    capsys,
) -> None:
    """Verify corpus-wide --suggest emits only REWORD or missing-handoff skills by default."""
    for name, desc in (
        ("widget-basics", BLUNT),
        ("widget-rollout", RIVAL),
        (
            "bq-observability",
            (
                "Monitors BigQuery slot utilization and INFORMATION_SCHEMA telemetry. "
                "Don't use for warehouse cost analysis (use `bq-cost-optimizer`)."
            ),
        ),
        (
            "bq-cost-optimizer",
            "Analyzes BigQuery slot utilization and INFORMATION_SCHEMA telemetry.",
        ),
        ("gadget-tuning", "Tune gadget throughput for busy pipelines."),
        ("ledger-audit", "Reconcile ledger entries against statements."),
        ("mailer-templates", "Author transactional mail templates."),
        ("photo-resize", "Crop photographs to a target aspect ratio."),
        ("query-planner", "Report how the planner picks a strategy."),
    ):
        d = tmp_path / name
        d.mkdir()
        (d / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: {desc}\n---\n{BODY}",
            encoding="utf-8",
        )

    assert main(["overlap", "--skills", str(tmp_path), "--suggest"]) == 0
    err = capsys.readouterr().err
    assert "widget-basics cedes" in err
    assert "ledger-audit" not in err
    assert err.count("no probe was issued") == 1

    payload = json.loads(
        emitted(
            ["overlap", "--skills", str(tmp_path), "--suggest", "--format", "json"],
            capsys,
        ),
    )
    assert payload["corpus_size"] == 9
    by_skill = {s["skill"]: s for s in payload["skills"]}
    assert "ledger-audit" not in by_skill
    assert by_skill["widget-basics"]["verdict"] == "reword"
    assert by_skill["bq-observability"]["missing_mutual_handoffs"] == ["bq-cost-optimizer"]


def test_the_suggestion_replaces_the_standings_rather_than_following_them(
    skill_repo: Path,
    capsys,
    wide,
) -> None:
    """Verify overlap --suggest replaces standings table rather than appending."""
    main(["overlap", "--skills", str(skill_repo), "--skill", "gke-basics", "--suggest"])
    assert "this skill" not in capsys.readouterr().err


def test_suggesting_writes_nothing_at_all(skill_repo: Path, tmp_path: Path) -> None:
    """Verify overlap --suggest creates no files or directories on disk."""
    before = sorted(tmp_path.rglob("*"))
    main(["overlap", "--skills", str(skill_repo), "--skill", "gke-basics", "--suggest"])
    assert sorted(tmp_path.rglob("*")) == before


def test_a_ceded_term_cannot_carry_a_negative_share() -> None:
    """Verify CededTerm validates share is greater than or equal to 0."""
    with pytest.raises(ValueError, match="greater than or equal to 0"):
        CededTerm(term="rollout", share=-0.1)


def test_the_suggestion_renders_as_json(skill_repo: Path, capsys) -> None:
    """Verify overlap --suggest --format json outputs structured suggestion payload."""
    payload = json.loads(
        emitted(
            [
                "overlap",
                "--skills",
                str(skill_repo),
                "--skill",
                "gke-basics",
                "--suggest",
                "--format",
                "json",
            ],
            capsys,
        ),
    )
    row = payload["skills"][0]

    assert payload["corpus_size"] == 3
    assert row["skill"] == "gke-basics"
    assert row["verdict"] in {v.value for v in Verdict}
    assert isinstance(row["crowded"], bool)
    assert isinstance(row["contenders"], list)


def test_the_rendering_inherits_the_measure_s_caveat_and_adds_its_own(
    skill_repo: Path,
    capsys,
) -> None:
    """Verify suggestion JSON caveat includes base overlap caveat plus drafting disclaimer."""
    payload = json.loads(
        emitted(
            [
                "overlap",
                "--skills",
                str(skill_repo),
                "--skill",
                "gke-basics",
                "--suggest",
                "--format",
                "json",
            ],
            capsys,
        ),
    )

    assert payload["caveat"] == list(REWRITE_CAVEAT)
    assert "replacement descriptions" in payload["caveat"][-1]


def test_a_skill_nothing_competes_with_renders_its_rival_as_absent(
    corpus_builder,
) -> None:
    """Verify rival field is null in JSON when verdict is UNRIVALED."""
    corpus = (
        corpus_builder().add("lonely-skill", "Do the one thing nobody else does.").build_skills()
    )
    view = suggest_view(rank_corpus(corpus), corpus, ["lonely-skill"])
    payload = json.loads(render_rewrite(view, "json"))

    assert view.skills[0].verdict is Verdict.UNRIVALED
    assert payload["skills"][0]["rival"] is None


def test_the_rendering_answers_for_every_skill_it_was_asked_about(
    skill_repo: Path,
    capsys,
) -> None:
    """Verify JSON output includes entries for each repeated --skill flag."""
    payload = json.loads(
        emitted(
            [
                "overlap",
                "--skills",
                str(skill_repo),
                "--skill",
                "gke-basics",
                "--skill",
                "gcs-lifecycle-rules",
                "--suggest",
                "--format",
                "json",
            ],
            capsys,
        ),
    )

    assert [r["skill"] for r in payload["skills"]] == [
        "gke-basics",
        "gcs-lifecycle-rules",
    ]


def test_the_csv_draws_one_row_per_term_and_says_which_kind_it_is(make_corpus) -> None:
    """Verify CSV output breaks down suggestion terms by role column."""
    corpus = make_corpus()
    view = suggest_view(rank_corpus(corpus), corpus, ["widget-basics"])
    rows = list(csv.DictReader(io.StringIO(render_rewrite(view, "csv"))))
    roles = {row["role"] for row in rows}
    ceded = next(row for row in rows if row["role"] == "ceded")

    assert list(rows[0]) == [
        "skill",
        "verdict",
        "rival",
        "contenders",
        "role",
        "term",
        "share",
    ]
    assert roles <= {"ceded", "disclaimed", "unclaimed"}
    assert "ceded" in roles
    assert ceded["rival"] == "widget-rollout"
    assert float(ceded["share"]) > 0


def test_a_declined_suggestion_still_draws_a_row(make_corpus) -> None:
    """Verify CSV output contains rows for contested skills with no rewrite recommendation."""
    corpus = make_corpus(HEDGED)
    view = suggest_view(rank_corpus(corpus), corpus, ["widget-basics"])
    rows = list(csv.DictReader(io.StringIO(render_rewrite(view, "csv"))))

    assert view.skills[0].verdict is Verdict.CONTESTED
    assert [row["skill"] for row in rows] == ["widget-basics"] * len(rows)
    assert all(row["verdict"] == "contested" for row in rows)


def test_the_csv_counts_the_contenders_the_json_names(make_corpus) -> None:
    """Verify CSV contenders column contains integer count of contenders list."""
    corpus = make_corpus()
    view = suggest_view(rank_corpus(corpus), corpus, ["widget-basics"])
    rows = list(csv.DictReader(io.StringIO(render_rewrite(view, "csv"))))
    named = json.loads(render_rewrite(view, "json"))["skills"][0]["contenders"]

    assert {row["contenders"] for row in rows} == {str(len(named))}


def test_the_rendering_goes_to_stdout_and_the_notes_beside_it(
    skill_repo: Path,
    make_runtime,
    monkeypatch,
    capsys,
) -> None:
    """Verify JSON rendering is emitted to stdout while runtime discovery notes go to stderr."""
    monkeypatch.setattr(
        "reach.cli.overlap.build_runtime",
        lambda _: make_runtime(roots=(SkillRoot(path=skill_repo, scope="user"),)),
    )
    argv = ["overlap", "--agent", "fake", "--skill", "gke-basics", "--suggest"]
    assert main([*argv, "--format", "json"]) == 0
    streams = capsys.readouterr()

    assert json.loads(streams.out)["skills"][0]["skill"] == "gke-basics"
    assert str(skill_repo) in streams.err


def test_an_unknown_format_is_refused_by_name(make_corpus) -> None:
    """Verify render_rewrite raises ValueError when unsupported format is requested."""
    corpus = make_corpus()
    view = suggest_view(rank_corpus(corpus), corpus, ["widget-basics"])

    with pytest.raises(
        ValueError,
        match=r"unknown format 'yaml'; expected .*csv, json",
    ):
        render_rewrite(view, "yaml")


def test_suggesting_for_many_skills_ranks_the_corpus_once(make_corpus, monkeypatch) -> None:
    """Verify suggest_all reuses single Bm25Scorer instance across all requested skills."""
    corpus = make_corpus()
    overlap = rank_corpus(corpus)
    built = 0
    original = Bm25Scorer.from_skills

    def counted(skills) -> Bm25Scorer:
        """Count each corpus ranking on the way through."""
        nonlocal built
        built += 1
        return original(skills)

    monkeypatch.setattr(Bm25Scorer, "from_skills", counted)
    suggest_all(overlap, corpus, ["widget-basics", "widget-rollout"])

    assert built == 1


def test_unclaimed_terms_with_pretokenized_corpus(make_corpus) -> None:
    """Verify unclaimed_terms correctly uses pre-tokenized corpus mapping."""
    corpus = make_corpus()
    scorer = Bm25Scorer.from_skills(corpus)
    tokenized = {name: frozenset(tokens) for name, tokens in scorer.documents.items()}

    terms = unclaimed_terms(
        corpus[0],
        corpus,
        BODY,
        scorer,
        tokenized_corpus=tokenized,
    )
    assert isinstance(terms, tuple)


def test_suggest_rewrite_tracks_missing_mutual_handoffs(corpus_builder) -> None:
    """Verify suggest_rewrite identifies rivals missing reciprocal handoffs back to target."""
    unreciprocated = (
        corpus_builder()
        .add(
            "warehouse-telemetry",
            "Monitors warehouse worker pool utilization and execution telemetry. "
            "Don't use for query cost tuning (use `warehouse-cost-tuner`).",
        )
        .add(
            "warehouse-cost-tuner",
            "Analyzes warehouse worker pool consumption and query costs using execution telemetry.",
        )
        .build_skills()
    )
    overlap_1 = rank_corpus(unreciprocated)
    rw_1 = suggest_rewrite(overlap_1.find("warehouse-telemetry"), unreciprocated)
    assert rw_1.rival == "warehouse-cost-tuner"
    assert rw_1.rival_disclaims_target is False
    assert "warehouse-cost-tuner" in rw_1.missing_mutual_handoffs

    reciprocated = (
        corpus_builder()
        .add(
            "warehouse-telemetry",
            "Monitors warehouse worker pool utilization and execution telemetry. "
            "Don't use for query cost tuning (use `warehouse-cost-tuner`).",
        )
        .add(
            "warehouse-cost-tuner",
            "Analyzes warehouse worker pool consumption and query costs using execution telemetry. "
            "Don't use for operational telemetry monitoring (use `warehouse-telemetry`).",
        )
        .build_skills()
    )
    overlap_2 = rank_corpus(reciprocated)
    rw_2 = suggest_rewrite(overlap_2.find("warehouse-telemetry"), reciprocated)
    assert rw_2.rival == "warehouse-cost-tuner"
    assert rw_2.rival_disclaims_target is True
    assert rw_2.missing_mutual_handoffs == ()


@pytest.mark.parametrize(
    ("band", "contenders", "expected_text"),
    [
        (
            0.90,
            ("primary-rival", "second-rival"),
            "1 more rival scores within 10% (second-rival)",
        ),
        (
            0.85,
            ("primary-rival", "second-rival", "third-rival"),
            "2 more rivals score within 15% (second-rival, third-rival)",
        ),
    ],
)
def test_print_rewrite_formats_dynamic_contender_band_and_names(
    shown,
    band: float,
    contenders: tuple[str, ...],
    expected_text: str,
) -> None:
    """Verify print_rewrite formats contender percentage from rewrite.band and lists rival names."""
    rw = Rewrite(
        skill="target-skill",
        rival=contenders[0],
        contenders=contenders,
        band=band,
    )
    rendered_text = shown(rw, width=120)
    assert expected_text in rendered_text


def test_overlap_cli_wires_overlap_settings_from_reach_toml(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify reach overlap --suggest respects [overlap] contender_band configured in reach.toml."""
    skills_dir = tmp_path / "skills"
    for name, desc in (
        ("widget-basics", "Explain widget fundamentals, rollout and rollback planning."),
        ("widget-rollout", "Plan and execute a widget rollout across fleets."),
        ("widget-rollback", "Plan and execute a widget rollback across fleets."),
        ("gadget-tuning", "Tune gadget throughput for busy pipelines."),
        ("ledger-audit", "Reconcile ledger entries against statements."),
    ):
        d = skills_dir / name
        d.mkdir(parents=True)
        (d / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: {desc}\n---\n{BODY}",
            encoding="utf-8",
        )
    toml_path = tmp_path / "reach.toml"
    toml_path.write_text("[overlap]\ncontender_band = 0.80\n", encoding="utf-8")

    code = main(
        [
            "overlap",
            "--skills",
            str(skills_dir),
            "--skill",
            "widget-basics",
            "--suggest",
            "--config",
            str(toml_path),
        ]
    )
    assert code == 0
    err = capsys.readouterr().err
    assert "within 20%" in err


def test_ceded_terms_excludes_target_skill_name_tokens(corpus_builder) -> None:
    """Verify ceded_terms never flags tokens from the target skill's own name."""
    corpus = (
        corpus_builder()
        .add(
            "flux-engine",
            "Guide Flux Engine usage on relay clusters with batch pipeline transforms.",
        )
        .add(
            "flux-engine-streaming",
            "Use the Flux Engine streaming endpoint for Flux Engine events and Flux Engine hooks.",
        )
        .add(
            "vault-archives",
            "Manage cold storage vaults and archive retention lock policies.",
        )
        .add(
            "parcel-routing",
            "Route inbound freight parcels across regional sorting hubs.",
        )
        .build_skills()
    )
    scorer = Bm25Scorer.from_skills(corpus)
    overlap = rank_corpus(corpus)
    comp = overlap.find("flux-engine")
    assert comp.nearest_rival is not None
    target = next(s for s in corpus if s.name == "flux-engine")
    rival = next(s for s in corpus if s.name == comp.nearest_rival.name)

    ceded = ceded_terms(target, rival, scorer, comp.nearest_rival.score)
    ceded_names = {c.term for c in ceded}
    assert "flux" not in ceded_names
    assert "engine" not in ceded_names


def test_suggest_all_honors_overlap_settings_min_claim_length_and_uses(
    corpus_builder,
    tmp_path: Path,
) -> None:
    """Verify suggest_all wires all OverlapSettings fields including min_claim_length and uses."""
    from reach.config import OverlapSettings

    body = (
        "Use zod schema validation (zod parser, zod types) or "
        "orchestration pipelines (orchestration workflows) to validate manifests."
    )
    builder = (
        corpus_builder()
        .add(
            "widget-basics",
            "Explain widget fundamentals, rollout and rollback planning.",
            body=body,
        )
        .add("widget-rollout", "Plan and execute a widget rollout across fleets.")
        .add("widget-rollback", "Plan and execute a widget rollback across fleets.")
        .add("gadget-tuning", "Tune gadget throughput for busy pipelines.")
        .add("ledger-audit", "Reconcile ledger entries against statements.")
    )
    builder.build_disk(tmp_path)
    corpus = builder.build_skills(tmp_path)
    overlap = rank_corpus(corpus)

    # Default min_claim_length=3, min_claim_uses=2 includes both 'zod' and 'orchestration'
    default_rw = suggest_all(overlap, corpus, ["widget-basics"], settings=OverlapSettings())[0]
    assert "zod" in default_rw.unclaimed
    assert "orchestration" in default_rw.unclaimed

    # Raising min_claim_length=6 excludes 'zod' while keeping 'orchestration'
    long_only_rw = suggest_all(
        overlap,
        corpus,
        ["widget-basics"],
        settings=OverlapSettings(min_claim_length=6, min_claim_uses=2),
    )[0]
    assert "zod" not in long_only_rw.unclaimed
    assert "orchestration" in long_only_rw.unclaimed

    # Raising min_claim_uses=3 excludes 'orchestration' (2 uses) while keeping 'zod' (3 uses)
    frequent_only_rw = suggest_all(
        overlap,
        corpus,
        ["widget-basics"],
        settings=OverlapSettings(min_claim_length=3, min_claim_uses=3),
    )[0]
    assert "zod" in frequent_only_rw.unclaimed
    assert "orchestration" not in frequent_only_rw.unclaimed
