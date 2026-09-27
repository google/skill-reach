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

"""Verify `reach overlap` command calculations, ranking, formatting, and CLI output."""

from __future__ import annotations

import csv
import io
import json
import math
import random
import re
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from reach.catalog import load_skills
from reach.cli import main
from reach.discovery import resolve_corpus
from reach.models import Skill
from reach.overlap import (
    OVERLAP_CAVEAT,
    Competition,
    CorpusOverlap,
    Rival,
    rank_corpus,
)
from reach.runtime import SkillRoot
from reach.views import (
    middle_truncate,
    overlap_view,
    print_overlap,
    print_skill_overlap,
    render_overlap,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from _pytest.capture import CaptureFixture
    from conftest import SyntheticCorpusBuilder


@pytest.fixture
def rivals(corpus_builder: type[SyntheticCorpusBuilder]) -> list[Skill]:
    """Provide a corpus with one isolated skill and one pair that competes."""
    return (
        corpus_builder()
        .add("alpha-solo", "Quantum entanglement lattice calibration.")
        .add("beta-pair", "Configure storage lifecycle rules for buckets.")
        .add("zeta-pair", "Configure storage lifecycle rules for objects.")
        .build_skills()
    )


def ordering(skills: list[Skill]) -> list[str]:
    """Return the corpus listing's skill names, in the order it puts them."""
    return [c.skill for c in rank_corpus(skills).competitions]


def test_the_target_holds_its_own_rank_position_in_its_field() -> None:
    """Verify standings seats target skill at rank position corresponding to outranked count."""
    contest = Competition(
        skill="target",
        self_score=1.0,
        rivals=(
            Rival(name="over-one", score=3.0),
            Rival(name="over-two", score=2.0),
            Rival(name="under", score=0.5),
        ),
    )
    seated = contest.standings

    assert [s.name for s in seated] == ["over-one", "over-two", "target", "under"]
    assert [s.rank for s in seated] == [1, 2, 3, 4]
    assert [s.is_target for s in seated] == [False, False, True, False]
    assert seated[contest.outranked_by].rank == contest.outranked_by + 1


def test_a_tied_rival_is_seated_below_the_target_it_did_not_beat() -> None:
    """Verify tied rivals are ranked below target skill in standings."""
    contest = Competition(
        skill="target",
        self_score=1.0,
        rivals=(Rival(name="twin", score=1.0),),
    )
    assert [s.name for s in contest.standings] == ["target", "twin"]
    assert contest.outranked_by == 0


def test_an_empty_corpus_ranks_nothing() -> None:
    """Verify rank_corpus returns empty competitions tuple for empty skill list."""
    assert rank_corpus([]).competitions == ()


def test_a_single_skill_has_no_rival_to_be_outranked_by(corpus_builder) -> None:
    """Verify singleton skill corpus produces competition with empty rivals and zero outranked."""
    lone = corpus_builder().add("solo", "The only skill installed.").build_skills()[0]
    only = rank_corpus([lone]).competitions

    assert len(only) == 1
    assert only[0].rivals == ()
    assert only[0].outranked_by == 0
    assert only[0].nearest_rival is None
    assert [s.name for s in only[0].standings] == ["solo"]


def test_identical_descriptions_are_still_separated_by_the_name(corpus_builder) -> None:
    """Verify identical descriptions are differentiated by skill name in self score."""
    twins = (
        corpus_builder()
        .add("alpha-twin", "Configure retention and bucket lock.")
        .add("omega-twin", "Configure retention and bucket lock.")
        .build_skills()
    )
    overlap = rank_corpus(twins)
    alpha = overlap.find("alpha-twin")

    assert alpha.outranked_by == 0
    assert alpha.nearest_rival is not None
    assert alpha.nearest_rival.name == "omega-twin"
    assert alpha.nearest_rival.score < alpha.self_score
    assert overlap.find("omega-twin").outranked_by == 0


def test_a_description_with_no_scoreable_term_still_takes_a_place(
    corpus_builder,
) -> None:
    """Verify punctuation-only descriptions are scored using skill name alone."""
    quiet, loud = (
        corpus_builder()
        .add("quiet", "...")
        .add("loud", "Configure storage lifecycle rules.")
        .build_skills()
    )
    contest = rank_corpus([quiet, loud]).find("quiet")

    assert contest.self_score > 0.0
    assert contest.outranked_by == 0
    assert contest.nearest_rival is not None
    assert contest.nearest_rival.score == 0.0


def test_the_ratio_orders_a_corpus_the_count_says_nothing_about(rivals) -> None:
    """Verify rival_ratio sorts competing skills when outranked_by count is identical."""
    overlap = rank_corpus(rivals)

    assert all(c.outranked_by == 0 for c in overlap.competitions)
    assert [c.skill for c in overlap.competitions] == [
        "beta-pair",
        "zeta-pair",
        "alpha-solo",
    ]
    assert overlap.find("beta-pair").rival_ratio > overlap.find("alpha-solo").rival_ratio
    assert overlap.find("alpha-solo").rival_ratio == 0.0


@pytest.mark.parametrize(
    ("rival_score", "expected"),
    [(0.7, math.inf), (0.0, 0.0)],
    ids=["a-rival-scores", "nobody-scores"],
)
def test_a_target_that_scores_nothing_on_its_own_words_sorts_by_who_beat_it(
    rival_score: float,
    expected: float,
) -> None:
    """Verify rival_ratio evaluates to math.inf when self score is 0 and rival score > 0."""
    competition = Competition(
        skill="house-vocabulary",
        self_score=0.0,
        rivals=(Rival(name="louder", score=rival_score),),
    )
    assert competition.rival_ratio == expected


@pytest.fixture
def beaten(corpus_builder: type[SyntheticCorpusBuilder]) -> list[Skill]:
    """Provide a corpus where a rival skill outranks the target skill."""
    return (
        corpus_builder()
        .add("zzz-beaten", "storage lifecycle rules")
        .add(
            "bbb-louder",
            "zzz beaten storage lifecycle rules zzz beaten storage lifecycle rules",
        )
        .build_skills()
    )


def test_a_rival_saying_the_same_words_more_densely_outranks_the_target(beaten) -> None:
    """Verify denser rival description causes outranked_by count to increment."""
    overlap = rank_corpus(beaten)
    target = overlap.find("zzz-beaten")

    assert target.outranked_by == 1
    assert target.nearest_rival is not None
    assert target.nearest_rival.score > target.self_score
    assert overlap.find("bbb-louder").outranked_by == 0


def test_an_outranked_skill_leads_because_its_ratio_exceeds_one(beaten) -> None:
    """Verify outranked skill sorts first due to rival_ratio greater than 1.0."""
    overlap = rank_corpus(beaten)
    leader = overlap.competitions[0]

    assert leader.skill == "zzz-beaten"
    assert leader.rival_ratio > 1.0
    assert overlap.competitions[1].rival_ratio < 1.0


def test_the_corpus_listing_marks_an_outranked_skill_rather_than_columning_zeros(
    make_console,
    rendered,
    beaten,
    rivals,
) -> None:
    """Verify outranked count column is rendered only when at least one skill is outranked."""
    console, buffer = make_console()
    print_overlap(console, rank_corpus(beaten))
    marked = rendered(buffer)

    console, buffer = make_console()
    print_overlap(console, rank_corpus(rivals))
    clean = rendered(buffer)

    assert "outranked by 1" in marked
    row = next(line for line in marked.splitlines() if line.strip().startswith("zzz-beaten"))
    assert "outranked by 1" in row
    assert "outranked by" not in clean


@pytest.mark.parametrize("seed", [0, 1, 7])
def test_the_whole_ordering_is_determined_by_the_corpus_alone(
    seed: int,
    rivals,
    corpus_builder,
) -> None:
    """Verify rank_corpus competition ordering is invariant under input skill shuffling."""
    corpus = [*rivals, *corpus_builder().add("zzz-beaten", "rules").build_skills()]
    shuffled = random.Random(seed).sample(corpus, len(corpus))  # noqa: S311

    assert ordering(shuffled) == ordering(corpus)
    assert ordering(corpus) == ordering(corpus)


def test_a_skill_the_corpus_does_not_carry_is_refused_by_name(rivals) -> None:
    """Verify find raises ValueError when requested skill is not in corpus."""
    with pytest.raises(ValueError, match="not-installed"):
        rank_corpus(rivals).find("not-installed")


def test_asking_the_runtime_where_skills_live_is_not_a_probe(
    make_runtime,
    skill_repo: Path,
) -> None:
    """Verify resolve_corpus uses filesystem discovery without executing runtime probes."""
    driver = make_runtime(roots=(SkillRoot(path=skill_repo, scope="user"),))
    found, _roots, discovered = resolve_corpus(driver, skill_repo, None)

    assert discovered is not None
    assert [c.skill for c in rank_corpus(found).competitions]
    assert driver.queries == []


@pytest.fixture
def listed(make_console, rendered: Callable[..., str], rivals) -> Callable[..., str]:
    """Render one of the two listings into a buffer at a fixed width."""

    def _listed(skill: str | None = None, **kwargs: object) -> str:
        """Print the corpus listing, or one skill's field, and return the text."""
        console, buffer = make_console(**kwargs)
        overlap = rank_corpus(rivals)
        if skill is None:
            print_overlap(console, overlap)
        else:
            print_skill_overlap(console, overlap.find(skill))
        return rendered(buffer)

    return _listed


@pytest.fixture
def name_family(corpus_builder: type[SyntheticCorpusBuilder]) -> list[Skill]:
    """Provide skills whose names agree on everything but their tails."""
    return (
        corpus_builder()
        .add(
            "google-cloud-solution-agentic-ai-bidirectional-streaming",
            "Stream turns between agents in both directions.",
        )
        .add(
            "google-cloud-solution-agentic-ai-borderless-data-lakehouse",
            "Query a lakehouse spanning several regions.",
        )
        .add(
            "google-cloud-solution-agentic-analytics-spark-knowledge-catalog",
            "Catalog Spark knowledge for analytics agents.",
        )
        .add(
            "google-cloud-solution-rag-enterprise-search-gke-sqldb",
            "Search enterprise documents from a GKE cluster.",
        )
        .build_skills()
    )


def shown_cells(text: str, column: int) -> list[str]:
    """Return one column of the corpus listing's rows, as the terminal drew it."""
    rows = [
        match.groups()
        for line in text.splitlines()
        if (match := re.match(r"\s*(\S+)\s+(\d+\.\d\d)\s+(\S*)", line))
    ]
    return [row[column] for row in rows]


@pytest.mark.parametrize("width", [60, 80, 100, 160])
def test_the_listing_keeps_a_name_family_apart_at_the_width_a_terminal_has(
    make_console,
    rendered,
    name_family,
    width: int,
) -> None:
    """Verify skill names in shared prefix families remain distinct across terminal widths."""
    console, buffer = make_console(width=width)
    print_overlap(console, rank_corpus(name_family))
    cells = shown_cells(rendered(buffer), 0)

    assert len(cells) == len(name_family)
    assert len(set(cells)) == len(name_family), (
        f"rows collapsed onto one another at width {width}: {cells}"
    )


@pytest.mark.parametrize("width", [60, 80, 100, 160])
def test_the_listing_names_a_rival_a_reader_can_tell_from_its_siblings(
    make_console,
    rendered,
    name_family,
    width: int,
) -> None:
    """Verify rival column renders unambiguous truncated names across various terminal widths."""
    console, buffer = make_console(width=width)
    print_overlap(console, rank_corpus(name_family))
    named = [cell for cell in shown_cells(rendered(buffer), 2) if cell]

    assert named, "expected these to compete"
    for cell in named:
        candidates = [skill.name for skill in name_family if _renders_as(skill.name, cell)]
        assert len(candidates) == 1, f"{cell!r} could be any of {candidates} at width {width}"


def _renders_as(name: str, cell: str) -> bool:
    """Determine whether full skill name could have produced the given truncated cell."""
    if cell == name:
        return True
    if "…" not in cell:
        return False
    head, _, tail = cell.partition("…")
    return name.startswith(head) and name.endswith(tail) and len(name) > len(cell) - 1


@pytest.mark.parametrize(
    ("value", "width", "expected"),
    [
        # No truncation needed when width accommodates string.
        ("gke-storage", 20, "gke-storage"),
        ("gke-storage", 11, "gke-storage"),
        # Truncate middle segment and preserve distinctive prefix/suffix.
        ("google-cloud-waf-security", 15, "google-…ecurity"),
        ("google-cloud-waf-sustainability", 15, "google-…ability"),
        # Handle degenerate and boundary widths gracefully.
        ("gke-storage", 1, "…"),
        ("gke-storage", 2, "…e"),
        ("gke-storage", 0, "gke-storage"),
    ],
)
def test_a_name_gives_up_its_middle_rather_than_its_tail(
    value: str,
    width: int,
    expected: str,
) -> None:
    """Verify middle truncation preserves distinctive prefix and suffix."""
    assert middle_truncate(value, width) == expected


def test_a_one_skill_corpus_is_not_reported_as_plural(
    make_console,
    rendered,
    corpus_builder,
) -> None:
    """Verify single-skill corpus summary uses singular phrasing."""
    console, buffer = make_console()
    solo = corpus_builder().add("solo", "The only skill installed.").build_skills()
    print_overlap(console, rank_corpus(solo))

    assert "1 skill, ranked" in rendered(buffer)


def test_the_corpus_listing_carries_a_row_per_skill_and_names_its_rival(
    listed,
    rivals,
) -> None:
    """Verify corpus overview table renders sorted rows, ratios, and nearest rivals."""
    shown = listed()
    header = next(line for line in shown.splitlines() if "overlap" in line)

    assert "nearest rival" in header
    for name in ("alpha-solo", "beta-pair", "zeta-pair"):
        assert name in shown
    row = next(line for line in shown.splitlines() if line.strip().startswith("beta-pair"))
    assert "zeta-pair" in row

    expected = [f"{c.rival_ratio:.2f}" for c in rank_corpus(rivals).competitions]
    assert re.findall(r"\d+\.\d+", shown) == expected

    solo_row = next(line for line in shown.splitlines() if line.strip().startswith("alpha-solo"))
    assert "0.00" in solo_row
    assert "beta-pair" not in solo_row
    assert "zeta-pair" not in solo_row

    ratios = [float(value) for value in re.findall(r"\d+\.\d+", shown)]
    assert ratios == sorted(ratios, reverse=True)
    assert len(set(ratios)) > 1, "a constant column orders nothing"
    assert ratios == [
        pytest.approx(c.rival_ratio, abs=5e-3) for c in rank_corpus(rivals).competitions
    ]


def test_the_skill_listing_seats_the_target_in_its_own_rank_position(listed) -> None:
    """Verify detailed skill view renders target row in its computed rank position and count."""
    shown = listed("beta-pair")
    rows = [line for line in shown.splitlines() if "pair" in line or "solo" in line]
    target = next(line for line in rows if line.lstrip().startswith("1"))

    assert "beta-pair" in target
    assert "this skill" in target
    assert "zeta-pair" in shown
    assert "alpha-solo" in shown
    assert "outranked by 0 of 2 rivals" in shown


@pytest.mark.parametrize("skill", [None, "beta-pair"])
def test_every_listing_says_overlap_does_not_predict_a_misroute(
    skill: str | None,
    listed,
) -> None:
    """Verify overlap report includes disclaimer and notes no probes were executed."""
    shown = listed(skill)

    assert "does not predict" in shown
    assert "misroute" in shown
    assert "reach eval" in shown
    assert "no probe was issued" in shown


@pytest.mark.parametrize("width", [60, 100])
@pytest.mark.parametrize("skill", [None, "beta-pair"])
def test_a_listing_holds_together_at_the_width_a_terminal_has(
    width: int,
    skill: str | None,
    listed,
) -> None:
    """Verify overlap views render without exceptions across different terminal widths."""
    shown = listed(skill, width=width)

    assert "beta-pair" in shown or "beta-pai" in shown
    assert "reach eval" in shown


def test_the_verb_ranks_the_corpus_it_was_pointed_at(
    skill_repo: Path,
    tmp_path: Path,
    capsys,
) -> None:
    """Verify overlap CLI runs on --skills directory without writing filesystem artifacts."""
    before = sorted(p for p in tmp_path.rglob("*"))
    assert main(["overlap", "--skills", str(skill_repo)]) == 0
    assert sorted(p for p in tmp_path.rglob("*")) == before
    shown = capsys.readouterr().err

    assert "gcs-lifecycle-rules" in shown
    assert "overlap" in shown
    assert "does not predict" in shown


def test_the_verb_reports_one_skill_s_whole_field_when_asked(skill_repo: Path, capsys) -> None:
    """Verify overlap CLI command filters to individual skill when --skill is provided."""
    assert main(["overlap", "--skills", str(skill_repo), "--skill", "gke-basics"]) == 0
    shown = capsys.readouterr().err

    assert "gke-basics" in shown
    assert "this skill" in shown
    assert "rivals" in shown


def test_the_verb_answers_for_every_skill_it_was_asked_about(skill_repo: Path, capsys) -> None:
    """Verify overlap CLI command renders standings for each repeated --skill flag."""
    assert (
        main(
            [
                "overlap",
                "--skills",
                str(skill_repo),
                "--skill",
                "gke-basics",
                "--skill",
                "gcs-lifecycle-rules",
            ],
        )
        == 0
    )
    shown = capsys.readouterr().err

    assert "gke-basics: outranked by" in shown
    assert "gcs-lifecycle-rules: outranked by" in shown
    assert shown.count("this skill") == 2
    assert shown.count("no probe was issued") == 1


def test_the_verb_says_where_it_found_the_corpus_nobody_named(
    skill_repo: Path,
    make_runtime,
    monkeypatch,
    capsys,
) -> None:
    """Verify overlap CLI command prints discovered skill root directory path."""
    monkeypatch.setattr(
        "reach.cli.overlap.build_runtime",
        lambda _: make_runtime(roots=(SkillRoot(path=skill_repo, scope="user"),)),
    )
    assert main(["overlap", "--agent", "fake"]) == 0
    shown = capsys.readouterr().err

    assert str(skill_repo) in shown
    assert "gcs-lifecycle-rules" in shown


def test_one_unknown_name_among_several_still_refuses(skill_repo: Path, capsys) -> None:
    """Verify overlap CLI exits with code 2 if any requested --skill name is unknown."""
    argv = ["overlap", "--skills", str(skill_repo)]
    assert main([*argv, "--skill", "gke-basics", "--skill", "absent"]) == 2
    assert "absent" in capsys.readouterr().err


def test_the_verb_says_where_to_look_when_discovery_finds_no_skills(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    """Verify overlap CLI reports error mentioning --skills when no skills are discovered."""
    monkeypatch.chdir(tmp_path)

    assert main(["overlap", "--agent", "fake"]) == 2
    reported = capsys.readouterr().err
    assert "--skills" in reported


def _panel_text(err: str) -> str:
    """Return error message text with borders and excess whitespace stripped."""
    return " ".join(err.replace("│", " ").split())


@pytest.mark.usefixtures("wide")
def test_an_empty_named_corpus_is_refused_by_the_path_that_was_named(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    """Verify overlap CLI reports empty directory path when --skills directory has no skills."""
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.chdir(tmp_path)

    assert main(["overlap", "--skills", str(empty), "--agent", "fake"]) == 2
    reported = _panel_text(capsys.readouterr().err)
    assert "no skills found" in reported
    assert str(empty) in reported
    assert "pass --skills" not in reported


@pytest.fixture
def ranked(skill_repo: Path) -> CorpusOverlap:
    """Provide a CorpusOverlap instance calculated from sample skill repository."""
    return rank_corpus(load_skills(skill_repo))


def emitted(argv: list[str], capsys: CaptureFixture[str]) -> str:
    """Run CLI main with argv and return captured stdout string."""
    assert main(argv) == 0
    return capsys.readouterr().out


def test_the_corpus_listing_renders_as_json(skill_repo: Path, ranked, capsys) -> None:
    """Verify overlap --format json outputs structured CorpusOverlap JSON payload."""
    payload = json.loads(
        emitted(["overlap", "--skills", str(skill_repo), "--format", "json"], capsys),
    )
    row = next(r for r in payload["skills"] if r["skill"] == "gcs-lifecycle-rules")
    expected = ranked.find("gcs-lifecycle-rules")

    assert payload["corpus_size"] == len(payload["skills"]) == 3
    assert payload["caveat"] == list(OVERLAP_CAVEAT)
    assert [r["skill"] for r in payload["skills"]] == [c.skill for c in ranked.competitions]
    assert row["nearest_rival"] == "gcs-retention-policy"
    assert row["rival_ratio"] == pytest.approx(expected.rival_ratio)
    assert row["self_score"] == pytest.approx(expected.self_score)
    assert row["outranked_by"] == 0
    assert all(r["standings"] == [] for r in payload["skills"])


def test_the_json_listing_of_one_skill_seats_it_among_its_rivals(
    skill_repo: Path,
    ranked,
    capsys,
) -> None:
    """Verify JSON output for single --skill includes full standings list."""
    payload = json.loads(
        emitted(
            [
                "overlap",
                "--skills",
                str(skill_repo),
                "--skill",
                "gke-basics",
                "--format",
                "json",
            ],
            capsys,
        ),
    )
    (row,) = payload["skills"]

    assert row["skill"] == "gke-basics"
    assert [s["name"] for s in row["standings"]] == [
        s.name for s in ranked.find("gke-basics").standings
    ]
    assert [s["name"] for s in row["standings"] if s["is_target"]] == ["gke-basics"]


def test_the_rendered_listing_answers_for_every_skill_it_was_asked_about(
    skill_repo: Path,
    capsys,
) -> None:
    """Verify JSON output includes entries for each requested --skill parameter."""
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


def test_the_corpus_listing_renders_as_csv(skill_repo: Path, ranked, capsys) -> None:
    """Verify overlap --format csv outputs tabular overview rows."""
    text = emitted(["overlap", "--skills", str(skill_repo), "--format", "csv"], capsys)
    rows = list(csv.DictReader(io.StringIO(text)))

    assert [r["skill"] for r in rows] == [c.skill for c in ranked.competitions]
    assert list(rows[0]) == [
        "skill",
        "self_score",
        "rival_ratio",
        "nearest_rival",
        "outranked_by",
    ]


def test_the_csv_listing_of_one_skill_is_the_field_the_terminal_draws(
    skill_repo: Path,
    ranked,
    capsys,
) -> None:
    """Verify CSV output for single --skill formats full standings table."""
    text = emitted(
        [
            "overlap",
            "--skills",
            str(skill_repo),
            "--skill",
            "gke-basics",
            "--format",
            "csv",
        ],
        capsys,
    )
    rows = list(csv.DictReader(io.StringIO(text)))

    assert list(rows[0]) == ["target", "rank", "skill", "score", "is_target"]
    assert [r["skill"] for r in rows] == [s.name for s in ranked.find("gke-basics").standings]
    assert [r["is_target"] for r in rows].count("yes") == 1
    assert {r["target"] for r in rows} == {"gke-basics"}


def test_a_rendering_goes_to_stdout_and_the_notes_it_needs_go_beside_it(
    skill_repo: Path,
    make_runtime,
    monkeypatch,
    capsys,
) -> None:
    """Verify machine-readable output goes to stdout while logs go to stderr."""
    monkeypatch.setattr(
        "reach.cli.overlap.build_runtime",
        lambda _: make_runtime(roots=(SkillRoot(path=skill_repo, scope="user"),)),
    )
    assert main(["overlap", "--agent", "fake", "--format", "json"]) == 0
    streams = capsys.readouterr()

    assert json.loads(streams.out)["corpus_size"] == 3
    assert str(skill_repo) in streams.err


def test_an_unknown_format_is_refused_by_name(ranked) -> None:
    """Verify render_overlap raises ValueError when unsupported format is requested."""
    with pytest.raises(ValueError, match="unknown format"):
        render_overlap(overlap_view(ranked), "yaml")


def test_a_target_beaten_from_a_standing_start_renders_no_ratio() -> None:
    """Verify zero self-score targets render rival_ratio as None in JSON and empty in CSV."""
    overlap = CorpusOverlap(
        competitions=(
            Competition(
                skill="house-vocabulary",
                self_score=0.0,
                rivals=(Rival(name="louder", score=0.7),),
            ),
        ),
    )
    view = overlap_view(overlap)
    (row,) = json.loads(render_overlap(view, "json"))["skills"]
    (line,) = list(csv.DictReader(io.StringIO(render_overlap(view, "csv"))))

    assert row["rival_ratio"] is None
    assert row["nearest_rival"] == "louder"
    assert line["rival_ratio"] == ""


def test_a_rival_that_scored_nothing_is_named_in_no_rendering() -> None:
    """Verify rivals with 0.0 score render as None in JSON output."""
    overlap = CorpusOverlap(
        competitions=(
            Competition(
                skill="solo",
                self_score=1.0,
                rivals=(Rival(name="unrelated", score=0.0),),
            ),
        ),
    )
    (row,) = json.loads(render_overlap(overlap_view(overlap), "json"))["skills"]

    assert row["nearest_rival"] is None
    assert row["rival_ratio"] == 0.0


def test_overlap_cli_semantic_and_positional_path_formats(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify reach overlap accepts positional path and --semantic across text, JSON, and CSV."""
    from unittest.mock import patch

    from reach.cli import app
    from reach.retrieval import DenseScorer

    _write_skills(
        tmp_path,
        [
            ("skill-a", "Manage git repositories."),
            ("skill-b", "Handle git branches."),
        ],
    )

    mock_vectors = {
        "skill-a": [1.0, 0.0],
        "skill-b": [0.8, 0.6],
    }
    with patch.object(DenseScorer, "from_skills", return_value=DenseScorer(vectors=mock_vectors)):
        # Test text format with positional path
        assert app(["overlap", str(tmp_path), "--semantic"]) == 0
        captured_text = capsys.readouterr()
        output = captured_text.out + captured_text.err
        assert "semantic sim" in output
        assert "quadrant" in output

        # Test JSON format
        assert app(["overlap", "--skills", str(tmp_path), "--semantic", "--format", "json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        row = payload["skills"][0]
        assert "semantic_similarity" in row
        assert row["semantic_similarity"] == pytest.approx(0.80, abs=0.01)
        assert row["quadrant"] is not None

        # Test CSV format
        assert app(["overlap", "--skills", str(tmp_path), "--semantic", "--format", "csv"]) == 0
        csv_lines = capsys.readouterr().out.strip().splitlines()
        header = csv_lines[0].split(",")
        assert "semantic_sim" in header
        assert "quadrant" in header


def test_overlap_explain_with_skill_path_infers_catalog(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify reach overlap explain resolves skill paths and infers parent catalog."""
    catalog = tmp_path / "skills"
    catalog.mkdir()
    skill_a = catalog / "skill-a"
    skill_a.mkdir()
    (skill_a / "SKILL.md").write_text(
        "---\nname: skill-a\ndescription: Manage git repositories.\n---\n",
        encoding="utf-8",
    )
    skill_b = catalog / "skill-b"
    skill_b.mkdir()
    (skill_b / "SKILL.md").write_text(
        "---\nname: skill-b\ndescription: Handle git branches.\n---\n",
        encoding="utf-8",
    )

    code = main(
        [
            "overlap",
            "explain",
            "commit changes to repository",
            "--skill",
            str(skill_a),
            "--format",
            "json",
        ]
    )
    assert code == 0
    data = json.loads(capsys.readouterr().out)
    assert data["target_skill"] == "skill-a"


def test_overlap_multiple_skills_preserves_independent_catalogs(
    tmp_path: Path,
) -> None:
    """Verify passing multiple --skill paths resolves against base catalog without loop mutation."""
    cat = tmp_path / "skills"
    cat.mkdir()
    s1 = cat / "s1"
    s1.mkdir()
    (s1 / "SKILL.md").write_text("---\nname: s1\ndescription: Skill 1.\n---\n", encoding="utf-8")

    s2 = cat / "s2"
    s2.mkdir()
    (s2 / "SKILL.md").write_text("---\nname: s2\ndescription: Skill 2.\n---\n", encoding="utf-8")

    code = main(["overlap", "--skill", str(s1), "--skill", str(s2)])
    assert code == 0


@pytest.fixture
def prefix_suffix_family(corpus_builder: type[SyntheticCorpusBuilder]) -> list[Skill]:
    """Provide skills sharing both prefix and suffix where only the middle token differs."""
    return (
        corpus_builder()
        .add(
            "agent-platform-tuning-management",
            "Manage model fine-tuning jobs and adapter hyperparameters on Agent Platform.",
        )
        .add(
            "agent-platform-endpoint-management",
            "Manage model deployment endpoints and traffic splits on Agent Platform.",
        )
        .add(
            "agent-platform-prompt-management",
            "Manage versioned prompt templates and variables on Agent Platform.",
        )
        .add(
            "google-cloud-waf-cost-optimization",
            "Optimize Google Cloud architecture for cloud cost and billing efficiency.",
        )
        .add(
            "google-cloud-waf-performance-optimization",
            "Optimize Google Cloud architecture for low latency and throughput performance.",
        )
        .build_skills()
    )


@pytest.mark.parametrize("width", [60, 80, 100, 160])
def test_prefix_and_suffix_family_never_collides_across_widths(
    make_console,
    rendered,
    prefix_suffix_family,
    width: int,
) -> None:
    """Verify skills sharing both prefix and suffix render distinct cells across terminal widths."""
    console, buffer = make_console(width=width)
    print_overlap(console, rank_corpus(prefix_suffix_family))
    cells = shown_cells(rendered(buffer), 0)

    assert len(cells) == len(prefix_suffix_family)
    assert len(set(cells)) == len(prefix_suffix_family), (
        f"prefix-suffix family collided at width {width}: {cells}"
    )


def _write_skills(root: Path, specs: list[tuple[str, str]]) -> None:
    """Create temporary skill subdirectories with SKILL.md files."""
    for name, desc in specs:
        d = root / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: {desc}\n---\n",
            encoding="utf-8",
        )


def test_no_truncate_flag_preserves_full_skill_names(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify --no-truncate renders full skill and rival names without ellipsis."""
    _write_skills(
        tmp_path,
        [
            ("agent-platform-tuning-management", "Manage tuning jobs on Agent Platform."),
            ("agent-platform-endpoint-management", "Manage endpoints on Agent Platform."),
        ],
    )

    assert main(["overlap", "--skills", str(tmp_path), "--no-truncate"]) == 0
    err = capsys.readouterr().err
    assert "agent-platform-tuning-management" in err
    assert "agent-platform-endpoint-management" in err
    assert "…" not in err.splitlines()[2]


def test_large_corpus_auto_caps_text_at_30_rows_and_respects_top_and_all(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify >30 skills auto-cap at 30 in text mode while --all and --top control row count."""
    from unittest.mock import patch

    from reach.retrieval import DenseScorer

    _write_skills(
        tmp_path,
        [
            (f"cloud-skill-{i:02d}", f"Configure cloud resource group {i % 5} rules.")
            for i in range(35)
        ],
    )

    # Default text output caps at 30 with omission footer
    assert main(["overlap", "--skills", str(tmp_path)]) == 0
    err_default = capsys.readouterr().err
    assert "Showing 30 of 35 skills" in err_default
    assert "5 more skills omitted" in err_default

    # --all displays all 35 rows
    assert main(["overlap", "--skills", str(tmp_path), "--all"]) == 0
    err_all = capsys.readouterr().err
    assert "35 skills, ranked by" in err_all
    assert "more skills omitted" not in err_all

    # --quadrant also respects the 30-row cap in text mode unless --all is passed
    mock_vectors = {f"cloud-skill-{i:02d}": [1.0, 0.0] for i in range(35)}
    with patch.object(DenseScorer, "from_skills", return_value=DenseScorer(vectors=mock_vectors)):
        assert main(["overlap", "--skills", str(tmp_path), "--quadrant", "latent-collision"]) == 0
        err_quad = capsys.readouterr().err
        assert "Showing 30 of 35 skills" in err_quad
        assert "5 more skills omitted" in err_quad

    # --top 5 displays 5 rows in both text and JSON
    assert main(["overlap", "--skills", str(tmp_path), "--top", "5"]) == 0
    err_top = capsys.readouterr().err
    assert "Showing 5 of 35 skills" in err_top
    assert "30 more skills omitted" in err_top

    payload_unfiltered = json.loads(
        emitted(["overlap", "--skills", str(tmp_path), "--format", "json"], capsys)
    )
    assert payload_unfiltered["corpus_size"] == 35
    assert len(payload_unfiltered["skills"]) == 35

    payload_top = json.loads(
        emitted(
            ["overlap", "--skills", str(tmp_path), "--top", "5", "--format", "json"],
            capsys,
        )
    )
    assert payload_top["corpus_size"] == 35
    assert len(payload_top["skills"]) == 5


def test_quadrant_filter_auto_enables_semantic_and_filters_rows(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify --quadrant implies --semantic and filters both text and JSON outputs."""
    from unittest.mock import patch

    from reach.retrieval import DenseScorer

    _write_skills(
        tmp_path,
        [
            ("dup-a", "Manage Kubernetes cluster autoscaling and node pools."),
            ("dup-b", "Manage Kubernetes cluster autoscaling and node pools."),
            ("distinct-c", "Reconcile billing invoices and tax ledgers."),
        ],
    )

    mock_vectors = {
        "dup-a": [1.0, 0.0],
        "dup-b": [0.99, 0.01],
        "distinct-c": [0.0, 1.0],
    }
    with patch.object(DenseScorer, "from_skills", return_value=DenseScorer(vectors=mock_vectors)):
        payload = json.loads(
            emitted(
                [
                    "overlap",
                    "--skills",
                    str(tmp_path),
                    "--quadrant",
                    "near-duplicate",
                    "--format",
                    "json",
                ],
                capsys,
            )
        )
        assert payload["corpus_size"] == 3
        assert [s["skill"] for s in payload["skills"]] == ["dup-a", "dup-b"]
        assert all(s["quadrant"] == "Near-Duplicate" for s in payload["skills"])


@pytest.mark.parametrize(
    ("extra_args", "expected_snippet"),
    [
        (["--top", "0"], "greater than or equal to 1"),
        (["--quadrant", "not-a-quadrant"], "unknown quadrant"),
    ],
)
def test_invalid_top_or_quadrant_fails_with_validation_error(
    skill_repo: Path,
    capsys: pytest.CaptureFixture[str],
    extra_args: list[str],
    expected_snippet: str,
) -> None:
    """Verify invalid --top or --quadrant values exit with code 2 and descriptive error."""
    assert main(["overlap", "--skills", str(skill_repo), *extra_args]) == 2
    assert expected_snippet in _panel_text(capsys.readouterr().err)


def test_unknown_skill_in_overlap_and_explain_suggests_fuzzy_matches(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify mistyped or middle-expanded --skill suggests close matches in corpus."""
    _write_skills(
        tmp_path,
        [
            (name, f"Skill for {name}.")
            for name in (
                "agent-platform-tuning",
                "agent-platform-tuning-management",
                "agent-platform-endpoint-management",
            )
        ],
    )

    assert (
        main(
            [
                "overlap",
                "--skills",
                str(tmp_path),
                "--skill",
                "agent-platform-model-tuning",
            ]
        )
        == 2
    )
    err = _panel_text(capsys.readouterr().err)
    assert "did you mean" in err
    assert "agent-platform-tuning" in err

    assert (
        main(
            [
                "overlap",
                "explain",
                "tune model",
                "--skills",
                str(tmp_path),
                "--skill",
                "agent-platform-model-tuning",
            ]
        )
        == 2
    )
    err_explain = _panel_text(capsys.readouterr().err)
    assert "did you mean" in err_explain
    assert "agent-platform-tuning" in err_explain


def test_multi_skill_overlap_suggest_prints_caveat_once(
    skill_repo: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify OVERLAP_CAVEAT is printed only once at the end of multi-skill --suggest output."""
    assert (
        main(
            [
                "overlap",
                "--skills",
                str(skill_repo),
                "--skill",
                "gke-basics",
                "--skill",
                "gcs-lifecycle-rules",
                "--suggest",
            ]
        )
        == 0
    )
    err_suggest = capsys.readouterr().err
    assert err_suggest.count("no probe was issued") == 1
