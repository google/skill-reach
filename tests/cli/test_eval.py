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

"""Verify `eval` CLI command for query drafting, execution, residency, and artifact generation."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Never, override

import pytest

from reach import run as run_module
from reach.artifact import read_artifact
from reach.cli import drafting, main
from reach.generate import read_checkpoint
from reach.models import Catalog
from reach.queries import load_query_set
from reach.runtime import SkillRoot
from reach.runtime.fake import FakeGenerator, FakeRuntime

if TYPE_CHECKING:
    from pathlib import Path

TOML = """\
[study]
skills = "{corpus}"
queries = "{queries}"
workdir = "{workdir}"
partial = true

[runtime]
agent = "fake"

[plan]
attempts = 1
"""

BODY = """\
---
name: {name}
description: >-
  {description}
---

# {name}

## Overview

{body}
"""

#: Sample corpus specifications with descriptions and markdown bodies.
SPECS = {
    "gcs-lifecycle-rules": (
        "Configures object lifecycle rules.",
        "Objects are tiered to Coldline once they stop being read.",
    ),
    "gcs-retention-policy": (
        "Configures retention and bucket lock.",
        "A bucket lock makes a retention period permanent.",
    ),
    "gke-basics": (
        "Explains GKE cluster fundamentals.",
        "A node pool is a group of nodes sharing one configuration.",
    ),
}


@pytest.fixture
def bodied_corpus(tmp_path: Path) -> Path:
    """Write a corpus whose skills have valid markdown bodies for drafting."""
    root = tmp_path / "corpus"
    for name, (description, body) in SPECS.items():
        directory = root / name
        directory.mkdir(parents=True)
        (directory / "SKILL.md").write_text(
            BODY.format(name=name, description=description, body=body),
            encoding="utf-8",
        )
    return root


@pytest.fixture
def generator(monkeypatch: pytest.MonkeyPatch) -> FakeGenerator:
    """Provide a mock generator runtime producing valid drafted query JSON."""
    runtime = FakeGenerator()
    runtime.completion = json.dumps(
        {
            "queries": [
                {"text": "How do I tier cold objects?", "citation": "Overview"},
                {"text": "What locks a retention period?", "citation": "Overview"},
            ],
        },
    )
    monkeypatch.setattr("reach.cli.drafting.text_generator", lambda **_: runtime)
    return runtime


@pytest.fixture
def argv(bodied_corpus: Path, tmp_path: Path) -> list[str]:
    """Provide an `eval` command line pointed at the bodied corpus."""
    return [
        "eval",
        "--skills",
        str(bodied_corpus),
        "--queries",
        str(tmp_path / "queries.json"),
        "--workdir",
        str(tmp_path / "work"),
        "--agent",
        "fake",
        "--attempts",
        "1",
        "--partial",
    ]


@pytest.fixture
def config_argv(bodied_corpus: Path, tmp_path: Path) -> list[str]:
    """Provide an `eval` command line configured via TOML file."""
    config = tmp_path / "reach.toml"
    config.write_text(
        TOML.format(
            corpus=bodied_corpus,
            queries=tmp_path / "queries.json",
            workdir=tmp_path / "work",
        ),
        encoding="utf-8",
    )
    return ["eval", "--config", str(config)]


@pytest.fixture
def discovering(bodied_corpus: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock runtime discovery to return the bodied corpus root without --skills flag."""
    monkeypatch.setattr(
        "reach.cli.eval.build_runtime",
        lambda _: FakeRuntime(roots=[SkillRoot(path=bodied_corpus, scope="user")]),
    )


@pytest.fixture
def discovered_argv(tmp_path: Path) -> list[str]:
    """Provide an `eval` command line that relies on runtime corpus discovery."""
    return [
        "eval",
        "--queries",
        str(tmp_path / "q.json"),
        "--workdir",
        str(tmp_path / "work"),
        "--agent",
        "fake",
    ]


def test_missing_query_set_is_drafted_with_citations_notes_and_cost_without_probing(
    argv: list[str],
    generator: FakeGenerator,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys,
) -> None:
    """Verify cold-start eval drafts queries, citations, notes, and cost without probing."""

    def refuse(*_args: object, **_kwargs: object) -> Never:
        msg = "eval probed a query set it had just written"
        raise AssertionError(msg)

    monkeypatch.setattr("reach.cli.eval.conduct", refuse)
    generator.completion_cost_usd = 0.1234
    assert main(argv) == 0

    written = load_query_set(tmp_path / "queries.json")
    assert written.catalog_id == "all"
    assert len(written.queries) == 6
    assert "content" in written.notes
    assert "Review before probing" in written.notes
    assert written.provenance is not None
    assert written.provenance.reviewed is None

    trail = json.loads((tmp_path / "queries-citations.json").read_text())
    assert len(trail) == 6
    assert {entry["citation"] for entry in trail} == {"Overview"}
    assert {entry["skill"] for entry in trail} == set(SPECS)
    assert not (tmp_path / "queries.json.artifact.json").exists()

    err = capsys.readouterr().err
    assert "$0.1234" in err
    assert "reach query" in err
    assert str(tmp_path / "queries.json") in err
    assert "queries.csv" in err


def test_the_whole_catalog_is_resident_even_when_one_skill_is_asked_about(
    argv: list[str],
    generator: FakeGenerator,
    tmp_path: Path,
) -> None:
    """Verify --skill restricts drafted questions while keeping whole catalog resident."""
    assert main([*argv, "--skill", "gke-basics"]) == 0

    written = load_query_set(tmp_path / "queries.json")
    assert {q.truth_label for q in written.queries} == {"gke-basics"}
    (prompt,) = generator.prompts
    assert prompt.count("Overview") == len(SPECS)


def test_a_rival_cap_narrows_the_prompt_and_leaves_residency_alone(
    argv: list[str],
    generator: FakeGenerator,
    tmp_path: Path,
    capsys,
) -> None:
    """Verify --top-rivals limits prompt rival skills while keeping catalog resident."""
    assert main([*argv, "--skill", "gke-basics", "--top-rivals", "1"]) == 0

    (prompt,) = generator.prompts
    assert prompt.count("=== RIVAL") == 1, "the cap did not reach the prompt"
    assert prompt.count("Overview") == 2, "target plus one rival, and no more"

    written = load_query_set(tmp_path / "queries.json")
    assert written.catalog_id == "all", "a prompt cap rescoped the run"
    assert "Rival cap: 1." in written.notes, written.notes
    provenance = written.provenance
    assert provenance is not None
    assert provenance.rivals_in_view == 1
    assert "1 of them shown as rivals" in capsys.readouterr().err


def test_an_uncapped_draft_says_nothing_about_a_cap(
    argv: list[str],
    generator: FakeGenerator,
    tmp_path: Path,
    capsys,
) -> None:
    """Verify notes and provenance omit rival cap mentions when --top-rivals is omitted."""
    assert main([*argv, "--skill", "gke-basics"]) == 0

    written = load_query_set(tmp_path / "queries.json")
    assert "Rival cap" not in written.notes
    provenance = written.provenance
    assert provenance is not None
    assert provenance.rivals_in_view == len(SPECS) - 1
    assert "shown as rivals" not in capsys.readouterr().err


def test_a_rival_cap_does_not_move_the_configuration_fingerprint(
    bodied_corpus: Path,
    generator: FakeGenerator,
    tmp_path: Path,
) -> None:
    """Verify configuration fingerprint is identical with or without --top-rivals."""

    def draft(name: str, *extra: str) -> str:
        """Draft into a destination of its own and return the fingerprint recorded."""
        destination = tmp_path / f"{name}.json"
        assert (
            main(
                [
                    "eval",
                    "--skills",
                    str(bodied_corpus),
                    "--queries",
                    str(destination),
                    "--workdir",
                    str(tmp_path / "work"),
                    "--agent",
                    "fake",
                    "--attempts",
                    "1",
                    "--partial",
                    "--skill",
                    "gke-basics",
                    *extra,
                ],
            )
            == 0
        )
        provenance = load_query_set(destination).provenance
        assert provenance is not None
        return provenance.config_fingerprint

    assert draft("capped", "--top-rivals", "1") == draft("uncapped")


def test_a_partial_drafted_against_a_different_field_of_rivals_is_refused(
    argv: list[str],
    dying: FakeGenerator,
    generator: FakeGenerator,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    """Verify resume is refused if partial was drafted under a different rival cap."""
    assert main([*argv, "--top-rivals", "1"]) == 2

    monkeypatch.setattr("reach.cli.drafting.text_generator", lambda **_: generator)
    assert main(argv) == 2

    err = capsys.readouterr().err
    assert "different terms" in err
    assert "top_rivals" in err
    assert generator.prompts == [], "it spent before it checked"


def test_whole_catalog_residency_is_the_default_however_the_run_was_configured(
    config_argv: list[str],
    generator: FakeGenerator,
    tmp_path: Path,
) -> None:
    """Verify catalog_id defaults to 'all' when mode is omitted in configuration."""
    assert main(config_argv) == 0
    assert load_query_set(tmp_path / "queries.json").catalog_id == "all"


def test_a_mode_the_config_does_set_is_not_overridden(
    bodied_corpus: Path,
    generator: FakeGenerator,
    tmp_path: Path,
    capsys,
) -> None:
    """Verify catalog mode specified in configuration file is preserved."""
    config = tmp_path / "reach.toml"
    config.write_text(
        TOML.format(
            corpus=bodied_corpus,
            queries=tmp_path / "queries.json",
            workdir=tmp_path / "work",
        )
        + '\n[catalog]\nmode = "singleton"\n',
        encoding="utf-8",
    )
    assert main(["eval", "--config", str(config)]) == 2
    assert "name the one to evaluate" in capsys.readouterr().err


def test_a_mode_named_on_the_command_line_beats_the_default_and_is_chosen_between(
    argv: list[str],
    generator: FakeGenerator,
    tmp_path: Path,
) -> None:
    """Verify --mode and --catalog command-line flags set specific catalog mode and target."""
    assert (
        main(
            [
                *argv,
                "--mode",
                "singleton",
                "--catalog",
                "singleton:gke-basics",
            ],
        )
        == 0
    )
    written = load_query_set(tmp_path / "queries.json")
    assert written.catalog_id == "singleton:gke-basics"
    assert {q.truth_label for q in written.queries} == {"gke-basics"}


def test_a_mode_named_on_the_command_line_still_has_to_be_chosen_between(
    argv: list[str],
    generator: FakeGenerator,
    capsys,
) -> None:
    """Verify --mode singleton prompts user to specify --catalog flag."""
    assert main([*argv, "--mode", "singleton"]) == 2
    assert "name the one to evaluate" in capsys.readouterr().err


def test_the_generator_receives_the_requested_count(
    argv: list[str],
    generator: FakeGenerator,
) -> None:
    """Verify --count flag customizes number of queries requested per target in prompt."""
    assert main([*argv, "--skill", "gke-basics", "--count", "1"]) == 0

    prompt = generator.prompts[0]
    assert "1 " in prompt or "one" in prompt.lower()


def test_a_target_that_is_not_resident_is_refused_before_anything_is_spent(
    argv: list[str],
    generator: FakeGenerator,
    capsys,
) -> None:
    """Verify eval exits with code 2 and sends no prompts if non-resident target requested."""
    assert main([*argv, "--skill", "gke-bascis"]) == 2
    assert "gke-bascis" in capsys.readouterr().err
    assert generator.prompts == []


@pytest.fixture
def dying(generator: FakeGenerator, monkeypatch: pytest.MonkeyPatch) -> FakeGenerator:
    """Provide a generator runtime that succeeds once and then raises RuntimeError."""

    class OneTargetThenGone(FakeGenerator):
        """Mock generator that answers first prompt and then raises exception."""

        @override
        def complete(self, prompt: str, *args: Any, **kwargs: Any) -> str:
            if self.completions:
                msg = "the generator went away"
                raise RuntimeError(msg)
            return super().complete(prompt, *args, **kwargs)

    runtime = OneTargetThenGone()
    runtime.completion = generator.completion
    monkeypatch.setattr("reach.cli.drafting.text_generator", lambda **_: runtime)
    return runtime


def test_what_was_drafted_before_a_failure_is_still_on_disk(
    argv: list[str],
    dying: FakeGenerator,
    tmp_path: Path,
) -> None:
    """Verify partial checkpoint preserves drafted queries without writing final files."""
    assert main(argv) == 2

    banked = read_checkpoint(tmp_path / "queries.json.partial")
    assert len(banked.drafted.queries) == 2, "the paid-for target did not survive"
    assert len(banked.citations) == 2, "the grounding for it did not survive either"
    assert len(banked.covered) == 1
    assert len(banked.owed) == 2, "the run must know what it still owes"
    assert not (tmp_path / "queries.json").exists(), "a partial reached the destination"
    assert not (tmp_path / "queries-citations.json").exists()


def test_a_resumed_draft_does_not_re_buy_the_targets_already_paid_for(
    argv: list[str],
    dying: FakeGenerator,
    generator: FakeGenerator,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    """Verify resuming interrupted draft logs recovery and only requests owed targets."""
    assert main(argv) == 2
    capsys.readouterr()

    monkeypatch.setattr("reach.cli.drafting.text_generator", lambda **_: generator)
    assert main(argv) == 0

    err = capsys.readouterr().err
    assert "resuming" in err
    assert "queries.json.partial" in err
    assert "1 targets already drafted, 2 still owed" in err
    assert generator.completions == 2, "a target already bought was bought again"
    written = load_query_set(tmp_path / "queries.json")
    assert len(written.queries) == 6, "the two halves did not add up to a whole set"
    trail = json.loads((tmp_path / "queries-citations.json").read_text())
    assert len(trail) == 6, "the grounding did not carry across the resume"
    assert not (tmp_path / "queries.json.partial").exists(), (
        "the partial outlived the set it was superseded by"
    )


def test_a_partial_drafted_under_another_arm_is_refused_rather_than_extended(
    argv: list[str],
    dying: FakeGenerator,
    generator: FakeGenerator,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    """Verify resume fails if generator arm differs from partial checkpoint."""
    assert main([*argv, "--generator-arm", "content"]) == 2

    monkeypatch.setattr("reach.cli.drafting.text_generator", lambda **_: generator)
    assert main([*argv, "--generator-arm", "framing"]) == 2

    err = capsys.readouterr().err
    assert "different terms" in err
    assert "arm" in err
    assert generator.prompts == [], "it spent before it checked"


def test_a_partial_is_refused_when_the_corpus_it_was_drafted_from_has_changed(
    argv: list[str],
    dying: FakeGenerator,
    generator: FakeGenerator,
    bodied_corpus: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    """Verify resume fails if corpus markdown bodies were modified after checkpoint was written."""
    assert main(argv) == 2

    body = bodied_corpus / "gke-basics" / "SKILL.md"
    body.write_text(
        body.read_text(encoding="utf-8") + "\nA cluster autoscaler adds nodes.\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("reach.cli.drafting.text_generator", lambda **_: generator)
    assert main(argv) == 2

    assert "bodies" in capsys.readouterr().err
    assert generator.prompts == []


def test_a_partial_that_cannot_be_read_names_itself_rather_than_crashing(
    argv: list[str],
    generator: FakeGenerator,
    tmp_path: Path,
    capsys,
) -> None:
    """Verify corrupt partial checkpoint produces clear error message instructing deletion."""
    (tmp_path / "queries.json.partial").write_text("{", encoding="utf-8")

    assert main(argv) == 2

    err = capsys.readouterr().err
    assert "delete it" in err
    assert "not a readable partial draft" in err or "partial draft" in err
    assert generator.prompts == [], "it drafted over a file it could not read"


def test_a_dry_run_prices_what_is_owed_rather_than_the_whole_draft(
    argv: list[str],
    dying: FakeGenerator,
    generator: FakeGenerator,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    """Verify --dry-run after partial checkpoint quotes only remaining owed targets."""
    assert main(argv) == 2
    capsys.readouterr()

    monkeypatch.setattr("reach.cli.drafting.text_generator", lambda **_: generator)
    assert main([*argv, "--dry-run"]) == 0

    assert "6 queries for 2 targets" in capsys.readouterr().err
    assert generator.prompts == []
    assert (tmp_path / "queries.json.partial").exists(), "a dry run discarded a draft"


def test_a_dry_run_says_what_it_would_draft_and_drafts_nothing(
    argv: list[str],
    generator: FakeGenerator,
    tmp_path: Path,
    capsys,
) -> None:
    """Verify --dry-run prints planned query counts and destination without writing files."""
    assert main([*argv, "--dry-run"]) == 0

    assert not (tmp_path / "queries.json").exists()
    assert generator.prompts == []
    err = capsys.readouterr().err
    assert "9 queries" in err
    assert str(tmp_path / "queries.json") in err
    assert "second invocation" in err


def test_a_dry_run_prices_the_prompt_the_real_draft_goes_on_to_send(
    argv: list[str],
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    """Verify prompt length and budget character counts match in --dry-run."""
    runtime = FakeGenerator(prompt_budget_chars=100_000)
    runtime.completion = json.dumps(
        {"queries": [{"text": "How do I tier cold objects?", "citation": "Overview"}]},
    )
    monkeypatch.setattr("reach.cli.drafting.text_generator", lambda **_: runtime)

    assert main([*argv, "--dry-run"]) == 0
    rehearsed = capsys.readouterr().err
    assert main(argv) == 0

    longest = max(len(prompt) for prompt in runtime.prompts)
    assert f"{longest:,} chars of the 100,000" in rehearsed


def test_a_dry_run_refuses_the_draft_the_real_run_would_be_refused_for(
    argv: list[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys,
) -> None:
    """Verify --dry-run enforces prompt budget limits and exits with code 2 if budget exceeded."""
    runtime = FakeGenerator(prompt_budget_chars=400)
    monkeypatch.setattr("reach.cli.drafting.text_generator", lambda **_: runtime)

    assert main([*argv, "--dry-run"]) == 2

    err = capsys.readouterr().err
    assert "400" in err
    assert runtime.prompts == [], "a rehearsal sent the prompt it was refusing"
    assert not (tmp_path / "queries.json").exists()


def test_auto_eval_measures_in_one_invocation(
    argv: list[str],
    generator: FakeGenerator,
    tmp_path: Path,
    capsys,
) -> None:
    """Verify --auto drafts unreviewed queries and immediately probes in a single run."""
    assert main([*argv, "--auto"]) == 0

    loaded = load_query_set(tmp_path / "queries.json")
    assert loaded.queries
    assert loaded.provenance is not None
    assert loaded.provenance.reviewed is False
    assert read_artifact(tmp_path / "queries.json.artifact.json").catalog_id == "all"
    err = capsys.readouterr().err
    assert "--auto" in err
    assert "recall" in err.split()


def test_auto_eval_without_explicit_queries_drafts_and_probes(
    bodied_corpus: Path,
    generator: FakeGenerator,
    capsys,
) -> None:
    """Verify --auto runs full catalog evaluation without requiring --queries flag."""
    assert (
        main(
            [
                "eval",
                "--skills",
                str(bodied_corpus),
                "--agent",
                "fake",
                "--auto",
                "--attempts",
                "1",
            ],
        )
        == 0
    )
    clean = " ".join(capsys.readouterr().err.split())
    assert "--auto was passed" in clean


def test_a_quick_drafted_set_is_also_stamped_unreviewed(
    quick_argv: list[str],
    generator: FakeGenerator,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify quick mode drafts record reviewed=False in provenance."""
    written = {}
    real_save = drafting.save_query_set

    def spy(query_set, path) -> Path:
        written["provenance"] = query_set.provenance
        return real_save(query_set, path)

    monkeypatch.setattr(drafting, "save_query_set", spy)
    assert main(quick_argv) == 0

    assert written["provenance"].reviewed is False


def test_auto_has_no_effect_on_a_set_already_there(
    argv: list[str],
    generator: FakeGenerator,
    tmp_path: Path,
) -> None:
    """Verify --auto does not overwrite provenance of an existing query set."""
    assert main(argv) == 0
    unstamped = load_query_set(tmp_path / "queries.json").provenance
    assert unstamped is not None
    assert unstamped.reviewed is None

    assert main([*argv, "--auto"]) == 0
    restamped = load_query_set(tmp_path / "queries.json").provenance
    assert restamped is not None
    assert restamped.reviewed is None


# --- Subsequent invocation and caching ---------------------------------------


def test_a_set_that_is_already_there_is_probed_rather_than_redrafted(
    argv: list[str],
    generator: FakeGenerator,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    """Verify existing query set is probed once without redrafting and writes scorecard."""
    real, builds = run_module.build_catalogs, []

    def counted(*args: Any, **kwargs: Any) -> list[Catalog]:
        builds.append(args)
        return real(*args, **kwargs)

    monkeypatch.setattr(run_module, "build_catalogs", counted)
    assert main(argv) == 0
    drafted = (tmp_path / "queries.json").read_text()
    generator.prompts.clear()
    builds.clear()
    capsys.readouterr()

    assert main(argv) == 0

    assert (tmp_path / "queries.json").read_text() == drafted
    assert generator.prompts == [], "the set on disk was redrafted"
    assert len(builds) == 1, "the catalog was resolved more than once"

    artifact = read_artifact(tmp_path / "queries.json.artifact.json")
    assert artifact.catalog_id == "all"
    assert artifact.catalog_size == len(SPECS)

    shown = capsys.readouterr().err
    assert "recall" in shown.split()
    assert "reach" not in shown.split()
    assert "gke-basics" in shown
    assert "| arm" not in shown
    assert "[arm" in shown


def test_the_artifact_follows_the_rows_when_a_run_records_them(
    argv: list[str],
    generator: FakeGenerator,
    tmp_path: Path,
) -> None:
    """Verify artifact is colocated with results JSONL when --records is specified."""
    out = tmp_path / "runs" / "results.jsonl"
    assert main(argv) == 0
    assert main([*argv, "--records", str(out)]) == 0

    assert (tmp_path / "runs" / "results.jsonl.artifact.json").exists()
    assert not (tmp_path / "queries.json.artifact.json").exists()


def test_the_artifact_can_be_sent_somewhere_named(
    argv: list[str],
    generator: FakeGenerator,
    tmp_path: Path,
) -> None:
    """Verify --out writes artifact to explicitly requested path."""
    named = tmp_path / "reports" / "pilot.json"
    assert main(argv) == 0
    assert main([*argv, "--out", str(named)]) == 0

    assert read_artifact(named).catalog_id == "all"


def test_tag_is_shown_in_scorecard_persisted_on_artifact_and_read_by_view(
    argv: list[str],
    generator: FakeGenerator,
    tmp_path: Path,
    capsys,
) -> None:
    """Verify --tag is shown in scorecard, persisted on artifact, and read back by reach view."""
    assert main([*argv, "--tag", "v1-baseline"]) == 0
    capsys.readouterr()
    assert main([*argv, "--tag", "v1-baseline"]) == 0

    assert "v1-baseline | arm" in capsys.readouterr().err
    artifact_path = tmp_path / "queries.json.artifact.json"
    artifact = read_artifact(artifact_path)
    assert artifact.digests.tag == "v1-baseline"

    assert main(["view", str(artifact_path)]) == 0
    assert "v1-baseline | arm" in capsys.readouterr().err


def test_the_tag_does_not_move_the_configuration_fingerprint(
    bodied_corpus: Path,
    generator: FakeGenerator,
    tmp_path: Path,
) -> None:
    """Verify config_fingerprint is invariant under different --tag options."""

    def draft(name: str, tag: str) -> str:
        destination = tmp_path / f"{name}.json"
        assert (
            main(
                [
                    "eval",
                    "--skills",
                    str(bodied_corpus),
                    "--queries",
                    str(destination),
                    "--workdir",
                    str(tmp_path / "work"),
                    "--agent",
                    "fake",
                    "--attempts",
                    "1",
                    "--partial",
                    "--skill",
                    "gke-basics",
                    "--tag",
                    tag,
                ],
            )
            == 0
        )
        provenance = load_query_set(destination).provenance
        assert provenance is not None
        return provenance.config_fingerprint

    assert draft("tagged", "v1-baseline") == draft("untagged", "")


def test_the_artifact_can_be_had_as_json_for_something_downstream(
    argv: list[str],
    generator: FakeGenerator,
    capsys,
) -> None:
    """Verify eval --format json writes artifact payload to stdout."""
    assert main(argv) == 0
    capsys.readouterr()
    assert main([*argv, "--format", "json"]) == 0

    assert json.loads(capsys.readouterr().out)["catalog_id"] == "all"


def test_a_dry_run_over_an_existing_set_plans_without_probing(
    argv: list[str],
    generator: FakeGenerator,
    tmp_path: Path,
) -> None:
    """Verify --dry-run over existing query set displays plan and exits without probing."""
    assert main(argv) == 0
    assert main([*argv, "--dry-run"]) == 0

    assert not (tmp_path / "queries.json.artifact.json").exists()


def test_the_corpus_can_come_from_the_runtime_rather_than_a_flag(
    discovered_argv: list[str],
    discovering: None,
    generator: FakeGenerator,
    tmp_path: Path,
) -> None:
    """Verify eval drafts from discovered runtime roots when --skills is omitted."""
    assert main(discovered_argv) == 0

    assert len(load_query_set(tmp_path / "q.json").queries) == 6


def test_where_the_skills_were_found_is_shown_back(
    discovered_argv: list[str],
    discovering: None,
    generator: FakeGenerator,
    bodied_corpus: Path,
    capsys,
) -> None:
    """Verify discovered corpus root paths are printed to stderr."""
    main(discovered_argv)

    assert str(bodied_corpus) in capsys.readouterr().err


@pytest.fixture
def shadowing(
    bodied_corpus: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    """Provide multiple roots with duplicate skill names and defined precedence."""
    losing = tmp_path / "project"
    (losing / "gke-basics").mkdir(parents=True)
    (losing / "gke-basics" / "SKILL.md").write_text(
        BODY.format(
            name="gke-basics",
            description="An older copy of the GKE skill that lost.",
            body="## Overview\n\nStale.\n",
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "reach.cli.eval.build_runtime",
        lambda _: FakeRuntime(
            roots=[
                SkillRoot(path=bodied_corpus, scope="user", precedence=0),
                SkillRoot(path=losing, scope="project", precedence=1),
            ],
        ),
    )
    return losing


def test_a_skill_that_two_roots_offered_is_recorded_on_the_artifact(
    discovered_argv: list[str],
    shadowing: Path,
    generator: FakeGenerator,
    tmp_path: Path,
) -> None:
    """Verify shadowed skills across multiple roots are recorded in artifact.contested_skills."""
    assert main(discovered_argv) == 0
    assert main(discovered_argv) == 0

    contested = read_artifact(tmp_path / "q.json.artifact.json").contested_skills
    assert [c.name for c in contested] == ["gke-basics"]
    assert contested[0].dropped == (shadowing / "gke-basics",)
    assert contested[0].ranked, "the runtime did rank these roots"


def test_a_tie_the_runtime_refused_to_break_is_recorded_as_such(
    discovered_argv: list[str],
    shadowing: Path,
    bodied_corpus: Path,
    generator: FakeGenerator,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify tie between unranked equal-precedence roots is recorded with ranked=False."""
    monkeypatch.setattr(
        "reach.cli.eval.build_runtime",
        lambda _: FakeRuntime(
            roots=[
                SkillRoot(path=bodied_corpus, scope="user", precedence=0),
                SkillRoot(path=shadowing, scope="project", precedence=0),
            ],
        ),
    )
    assert main(discovered_argv) == 0
    assert main(discovered_argv) == 0

    contested = read_artifact(tmp_path / "q.json.artifact.json").contested_skills
    assert [c.name for c in contested] == ["gke-basics"]
    assert not contested[0].ranked


def test_a_run_that_was_pointed_at_a_corpus_contests_nothing(
    argv: list[str],
    generator: FakeGenerator,
    tmp_path: Path,
) -> None:
    """Verify explicitly specified --skills produces empty contested_skills in artifact."""
    assert main(argv) == 0
    assert main(argv) == 0

    assert read_artifact(tmp_path / "queries.json.artifact.json").contested_skills == ()


def test_a_runtime_that_finds_no_skills_is_told_so_plainly(
    discovered_argv: list[str],
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    """Verify eval exits with code 2 when runtime discovery yields zero skills."""
    monkeypatch.setattr("reach.cli.eval.build_runtime", lambda _: FakeRuntime())
    assert main(discovered_argv) == 2

    assert "no skills found" in capsys.readouterr().err


def test_catalog_eval_without_queries_provides_actionable_guidance(capsys) -> None:
    """Verify catalog eval without --queries guides the user with actionable next steps."""
    assert main(["eval", "--agent", "fake"]) == 2
    clean = " ".join(capsys.readouterr().err.split())
    assert "catalog evaluation requires a labeled query benchmark" in clean
    assert "reach eval <skill-name>" in clean
    assert "reach query draft" in clean
    assert "--queries" in clean


def test_catalog_eval_with_skills_suggests_exact_corpus_path(
    bodied_corpus: Path,
    capsys,
) -> None:
    """Verify error message embeds explicit --skills path in suggested commands."""
    assert main(["eval", "--skills", str(bodied_corpus), "--agent", "fake"]) == 2
    clean = " ".join(capsys.readouterr().err.replace("│", " ").split())
    assert "reach query draft" in clean
    assert "queries.json" in clean


def test_eval_with_directory_positional_suggests_child_skills(
    bodied_corpus: Path,
    capsys,
) -> None:
    """Verify passing a directory of skills positionally guides the user to specific skills."""
    assert main(["eval", str(bodied_corpus), "--agent", "fake"]) == 2
    clean = " ".join(capsys.readouterr().err.replace("│", " ").split())
    assert "is a directory containing" in clean
    assert "not a single skill" in clean
    assert "reach eval" in clean


def test_eval_with_queries_does_not_require_workdir(
    bodied_corpus: Path,
    query_file: Path,
    generator: FakeGenerator,
) -> None:
    """Verify evaluation runs with --queries without requiring explicit --workdir."""
    argv = [
        "eval",
        "--skills",
        str(bodied_corpus),
        "--queries",
        str(query_file),
        "--catalog",
        "all",
        "--rescope",
        "--partial",
        "--agent",
        "fake",
        "--dry-run",
    ]
    assert main(argv) == 0


@pytest.mark.parametrize("corpus_arg_mode", ["skills_flag", "positional"])
def test_zero_config_corpus_eval_with_skills_or_positional_dir(
    bodied_corpus: Path,
    tmp_path: Path,
    corpus_arg_mode: str,
) -> None:
    """Verify zero-config `reach eval` works with --skills or positional corpus path."""
    subset_queries = tmp_path / "subset_5.json"
    subset_queries.write_text(
        json.dumps(
            {
                "queries": [
                    {
                        "id": "q-1",
                        "text": "deploy a container to cloud run",
                        "expected_skill": "gke-basics",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    if corpus_arg_mode == "skills_flag":
        corpus_args = ["--skills", str(bodied_corpus)]
    else:
        corpus_args = [str(bodied_corpus)]

    argv = [
        "eval",
        *corpus_args,
        "--queries",
        str(subset_queries),
        "--workers",
        "4",
        "--agent",
        "fake",
        "--dry-run",
    ]
    assert main(argv) == 0


@pytest.fixture
def run_dir_argv(bodied_corpus: Path, tmp_path: Path) -> list[str]:
    """Provide an `eval` command line naming only --run-dir, no file flags."""
    return [
        "eval",
        "--skills",
        str(bodied_corpus),
        "--run-dir",
        str(tmp_path / "run"),
        "--agent",
        "fake",
        "--attempts",
        "1",
        "--partial",
    ]


def test_run_dir_nests_the_query_set_the_workspace_and_the_results(
    run_dir_argv: list[str],
    generator: FakeGenerator,
    tmp_path: Path,
) -> None:
    """Verify --run-dir configures default locations for queries, workspace, and results."""
    run_dir = tmp_path / "run"
    assert main(run_dir_argv) == 0
    assert load_query_set(run_dir / "queries.json").catalog_id == "all"

    assert main([*run_dir_argv, "--records", str(run_dir / "results.jsonl")]) == 0
    assert (run_dir / "workspace").exists()


def test_run_dir_defaults_out_and_the_artifact_follows_it(
    run_dir_argv: list[str],
    generator: FakeGenerator,
    tmp_path: Path,
) -> None:
    """Verify --run-dir defaults output path to results.jsonl and colocates artifact."""
    run_dir = tmp_path / "run"
    assert main(run_dir_argv) == 0
    assert main(run_dir_argv) == 0

    assert (run_dir / "results.jsonl").exists()
    assert read_artifact(run_dir / "results.jsonl.artifact.json").catalog_id == "all"


def test_an_explicit_flag_beats_the_run_dir_default(
    bodied_corpus: Path,
    generator: FakeGenerator,
    tmp_path: Path,
) -> None:
    """Verify explicitly provided file flag overrides --run-dir default location."""
    elsewhere = tmp_path / "elsewhere.json"
    assert (
        main(
            [
                "eval",
                "--skills",
                str(bodied_corpus),
                "--run-dir",
                str(tmp_path / "run"),
                "--queries",
                str(elsewhere),
                "--agent",
                "fake",
                "--attempts",
                "1",
                "--partial",
            ],
        )
        == 0
    )
    assert elsewhere.exists()
    assert not (tmp_path / "run" / "queries.json").exists()


def test_run_dir_does_not_reach_for_a_quick_run(
    quick_argv: list[str],
    generator: FakeGenerator,
    tmp_path: Path,
) -> None:
    """Verify --run-dir is ignored during quick evaluation runs."""
    assert main([*quick_argv, "--run-dir", str(tmp_path / "run")]) == 0
    assert not (tmp_path / "run").exists()


@pytest.fixture
def scratch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Provide a temporary scratch directory for quick-mode runs."""
    made = tmp_path / "scratch"

    def here(**_: object) -> str:
        made.mkdir()
        return str(made)

    monkeypatch.setattr("reach.cli.eval.tempfile.mkdtemp", here)
    return made


@pytest.fixture
def quick_argv(bodied_corpus: Path) -> list[str]:
    """Provide a quick-mode command line evaluating a single skill."""
    return ["eval", "gke-basics", "--skills", str(bodied_corpus), "--agent", "fake"]


def test_naming_a_skill_drafts_probes_and_summarizes_in_one_invocation(
    quick_argv: list[str],
    generator: FakeGenerator,
    scratch: Path,
    tmp_path: Path,
    capsys,
) -> None:
    """Verify positional skill name drafts, probes, logs scope, and cleans scratch in one run."""
    assert main(quick_argv) == 0

    shown = capsys.readouterr().err
    assert generator.prompts, "nothing was drafted"
    assert "recall" in shown.split(), shown
    assert "gke-basics" in shown
    assert "neighborhood:gke-basics" in shown
    assert f"{len(SPECS)} of {len(SPECS)} skills resident" in shown
    assert "3 attempts per query" in shown
    assert "drafted here and probed unreviewed, against 2 rivals" in shown
    assert not scratch.exists()
    assert list(tmp_path.rglob("*artifact*")) == []
    assert list(tmp_path.rglob("*.json")) == []


def test_a_quick_dry_run_offers_no_path_to_a_file_nobody_could_open(
    quick_argv: list[str],
    scratch: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    """Verify quick mode --dry-run omits references to temporary scratch paths."""
    runtime = FakeGenerator(prompt_budget_chars=100_000)
    monkeypatch.setattr("reach.cli.drafting.text_generator", lambda **_: runtime)

    assert main([*quick_argv, "--dry-run"]) == 0

    err = capsys.readouterr().err
    assert "longest prompt" in err
    assert str(scratch) not in err
    assert "would write" not in err


def test_a_quick_run_holds_the_named_skills_neighborhood_and_not_the_corpus(
    quick_argv: list[str],
    generator: FakeGenerator,
    tmp_path: Path,
) -> None:
    """Verify quick mode constructs a neighborhood catalog centered on target skill."""
    named = tmp_path / "quick.artifact.json"
    assert main([*quick_argv, "--out", str(named)]) == 0

    assert read_artifact(named).catalog_id == "neighborhood:gke-basics"


@pytest.mark.parametrize("source", ["cli", "autodiscovered_config"])
def test_the_depth_a_quick_run_states_is_the_one_it_was_given(
    source: str,
    quick_argv: list[str],
    generator: FakeGenerator,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    """Verify CLI --attempts and auto-discovered reach.toml override quick mode depth."""
    if source == "cli":
        assert main([*quick_argv, "--attempts", "1"]) == 0
        assert "1 attempt per query" in capsys.readouterr().err
        return

    cfg_dir = tmp_path / "cfg_workspace"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "reach.toml").write_text("[plan]\nattempts = 4\n", encoding="utf-8")
    monkeypatch.chdir(cfg_dir)
    assert main(quick_argv) == 0
    assert "4 attempts per query" in capsys.readouterr().err


def test_a_quick_run_records_when_it_is_asked_to(
    quick_argv: list[str],
    generator: FakeGenerator,
    tmp_path: Path,
) -> None:
    """Verify --records flag persists results and artifact from quick mode evaluation."""
    out = tmp_path / "rows.jsonl"
    assert main([*quick_argv, "--records", str(out)]) == 0

    assert out.exists()
    assert read_artifact(tmp_path / "rows.jsonl.artifact.json").catalog_id == (
        "neighborhood:gke-basics"
    )


@pytest.mark.parametrize("subpath", ["gke-basics", "gke-basics/SKILL.md"])
def test_a_directory_names_both_the_skill_and_the_corpus_it_sits_in(
    bodied_corpus: Path,
    generator: FakeGenerator,
    capsys,
    subpath: str,
) -> None:
    """Verify path to skill directory or SKILL.md resolves target skill and parent corpus."""
    assert main(["eval", str(bodied_corpus / subpath), "--agent", "fake"]) == 0

    shown = capsys.readouterr().err
    assert "neighborhood:gke-basics" in shown
    assert f"{len(SPECS)} of {len(SPECS)} skills resident" in shown


def test_a_typed_query_is_probed_as_authored_ground_truth_without_drafting(
    bodied_corpus: Path,
    generator: FakeGenerator,
    tmp_path: Path,
    capsys,
) -> None:
    """Verify --query and --expected probe typed question without invoking generator."""
    saved = tmp_path / "saved_typed"
    assert (
        main(
            [
                "eval",
                "--skills",
                str(bodied_corpus),
                "--agent",
                "fake",
                "--query",
                "How do I resize a node pool?",
                "--expected",
                "gke-basics",
                "--save",
                str(saved),
            ],
        )
        == 0
    )

    assert generator.prompts == [], "a typed query was drafted over"
    assert "ground truth the 1 question you typed" in capsys.readouterr().err
    assert (saved / "queries.json").is_file()


def test_several_typed_queries_all_take_the_one_label_they_were_given(
    quick_argv: list[str],
    generator: FakeGenerator,
    capsys,
) -> None:
    """Verify multiple --query flags share positional skill name as expected label."""
    assert main([*quick_argv, "--query", "What is a node pool?", "--query", "Scale it?"]) == 0

    assert generator.prompts == []
    assert "the 2 questions you typed" in capsys.readouterr().err


def test_a_skill_the_corpus_does_not_hold_is_named_back_with_what_it_does(
    bodied_corpus: Path,
    generator: FakeGenerator,
    capsys,
) -> None:
    """Verify misspelled positional skill name produces suggestion in error output."""
    typo = ["eval", "gke-bascis", "--skills", str(bodied_corpus), "--agent", "fake"]
    assert main(typo) == 2

    complaint = capsys.readouterr().err
    assert "gke-bascis" in complaint
    assert "did you mean gke-basics" in complaint


def test_a_quick_invocation_refuses_a_configuration_file(
    bodied_corpus: Path,
    tmp_path: Path,
    capsys,
) -> None:
    """Verify passing both positional skill and --config is rejected with code 2."""
    config = tmp_path / "reach.toml"
    config.write_text(
        TOML.format(
            corpus=bodied_corpus,
            queries=tmp_path / "queries.json",
            workdir=tmp_path / "work",
        ),
        encoding="utf-8",
    )
    assert main(["eval", "gke-basics", "--config", str(config)]) == 2

    assert "--config" in capsys.readouterr().err


def test_a_quick_invocation_refuses_a_named_query_set(tmp_path: Path, capsys) -> None:
    """Verify passing both positional skill and --queries is rejected with code 2."""
    assert main(["eval", "gke-basics", "--queries", str(tmp_path / "q.json")]) == 2

    assert "--queries" in capsys.readouterr().err


def test_a_typed_query_with_no_ground_truth_is_refused_before_anything_runs(
    bodied_corpus: Path,
    capsys,
) -> None:
    """Verify --query without --expected or positional skill is rejected before execution."""
    assert main(["eval", "--skills", str(bodied_corpus), "--query", "What is this?"]) == 2

    complaint = capsys.readouterr().err
    assert "--expected" in complaint


def test_an_expectation_with_no_question_to_attach_it_to_is_refused(capsys) -> None:
    """Verify --expected without --query is rejected with code 2."""
    assert main(["eval", "gke-basics", "--expected", "gke-basics"]) == 2

    assert "--expected labels a --query" in capsys.readouterr().err


def test_save_promotes_the_query_set_the_citations_and_the_artifact(
    quick_argv: list[str],
    generator: FakeGenerator,
    scratch: Path,
    tmp_path: Path,
) -> None:
    """Verify --save copies queries, citations, and artifact from scratch directory."""
    kept = tmp_path / "kept"
    assert main([*quick_argv, "--save", str(kept)]) == 0

    assert not scratch.exists(), "the scratch directory itself must still go"
    saved_set = load_query_set(kept / "queries.json")
    assert saved_set.queries
    assert saved_set.provenance is not None
    assert saved_set.provenance.reviewed is False
    trail = json.loads((kept / "queries-citations.json").read_text())
    assert trail
    assert read_artifact(kept / "queries.json.artifact.json").catalog_id == (
        "neighborhood:gke-basics"
    )


def test_save_needs_a_quick_invocation(tmp_path: Path, capsys) -> None:
    """Verify --save on formal evaluation path is rejected with code 2."""
    assert main(["eval", "--save", str(tmp_path / "kept"), "--agent", "fake"]) == 2

    complaint = capsys.readouterr().err
    assert "--save" in complaint
    assert "quick" in complaint


def test_save_does_not_promote_an_artifact_that_already_landed_elsewhere(
    quick_argv: list[str],
    generator: FakeGenerator,
    tmp_path: Path,
) -> None:
    """Verify explicit --out destination is preserved when --save is also passed."""
    named = tmp_path / "reports" / "quick.artifact.json"
    kept = tmp_path / "kept"
    assert main([*quick_argv, "--out", str(named), "--save", str(kept)]) == 0

    assert read_artifact(named).catalog_id == "neighborhood:gke-basics"
    assert not (kept / "quick.artifact.json").exists()
    assert load_query_set(kept / "queries.json").queries


def test_save_is_not_offered_a_partial_run(
    quick_argv: list[str],
    generator: FakeGenerator,
    scratch: Path,
    tmp_path: Path,
) -> None:
    """Verify --dry-run combined with --save writes no files to saved destination."""
    kept = tmp_path / "kept"
    assert main([*quick_argv, "--save", str(kept), "--dry-run"]) == 0

    assert not kept.exists()


def test_eval_range_validators_reject_invalid_flags(quick_argv: list[str], capsys) -> None:
    """Verify cyclopts range validators reject invalid parameters in eval."""
    assert main([*quick_argv, "--attempts", "0"]) == 2
    err = capsys.readouterr().err
    assert "Invalid value '0' for --attempts" in err
    assert "greater than or equal" in err

    assert main([*quick_argv, "--retries", "-1"]) == 2
    err = capsys.readouterr().err
    assert "Invalid value '-1' for --retries" in err
    assert "greater than or equal" in err

    assert main([*quick_argv, "--timeout", "0"]) == 2
    err = capsys.readouterr().err
    assert "Invalid value '0' for --timeout" in err
    assert "greater than or equal" in err

    assert main([*quick_argv, "--catalog-size", "0"]) == 2
    err = capsys.readouterr().err
    assert "Invalid value '0' for --catalog-size" in err
    assert "greater than or equal" in err


@pytest.mark.parametrize(
    "argv",
    [
        ["query", "--count", "0"],
        ["query", "draft", "--count", "0"],
    ],
)
def test_query_range_validators_reject_invalid_flags(
    capsys: pytest.CaptureFixture[str],
    argv: list[str],
) -> None:
    """Verify cyclopts range validators reject invalid count in query commands."""
    assert main(argv) == 2
    err = capsys.readouterr().err
    assert "Invalid value '0' for --count" in err
    assert "greater than or equal" in err


def test_eval_nonexistent_path_fails_cleanly() -> None:
    """Verify reach eval exits 2 when given a nonexistent skill path."""
    assert main(["eval", "./nonexistent/path/to/skill", "--agent", "fake"]) == 2


def test_eval_filters_existing_queries_file_by_skill_flag(
    bodied_corpus: Path,
    tmp_path: Path,
) -> None:
    """Verify --skill filters an existing --queries file to only the requested skill(s)."""
    queries_file = tmp_path / "multi.yaml"
    queries_file.write_text(
        "queries:\n"
        "  - text: deploy gke cluster\n"
        "    expected_skill: gke-basics\n"
        "  - text: run bigquery sql\n"
        "    expected_skill: bigquery-basics\n",
        encoding="utf-8",
    )
    out_file = tmp_path / "eval.json"
    assert (
        main(
            [
                "eval",
                "--skills",
                str(bodied_corpus),
                "--queries",
                str(queries_file),
                "--skill",
                "gke-basics",
                "--mode",
                "all",
                "--agent",
                "fake",
                "--out",
                str(out_file),
                "--yes",
            ]
        )
        == 0
    )
    artifact = json.loads(out_file.read_text(encoding="utf-8"))
    evaluated_skills = {q["expected"] for q in artifact["queries"]}
    assert evaluated_skills == {"gke-basics"}


def test_quick_eval_with_keyword_agent_drafts_and_saves_cleanly(
    bodied_corpus: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify quick eval with --agent keyword drafts offline, uses 1 attempt, and saves."""
    saved_dir = tmp_path / "saved_run"
    rc = main(
        [
            "eval",
            "gke-basics",
            "--skills",
            str(bodied_corpus),
            "--agent",
            "keyword",
            "--save",
            str(saved_dir),
            "--yes",
        ]
    )
    assert rc == 0
    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert "1 attempt per query" in combined
    assert "keyword/keyword" in combined
    assert "keyword/gemini-" not in combined
    assert str(saved_dir / "queries.json.artifact.json") in combined
    assert combined.count("wrote  ") == 1

    saved_qs = load_query_set(saved_dir / "queries.json")
    assert len(saved_qs.queries) > 0
    assert saved_qs.provenance is not None
    assert saved_qs.provenance.generator_model == "keyword"


def test_draft_generation_aborts_with_error_when_zero_queries_produced(
    bodied_corpus: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify query drafting exits with code 1 when the generator produces 0 verified queries."""
    from unittest.mock import patch

    from reach.runtime.keyword import KeywordGenerator

    with patch.object(KeywordGenerator, "complete", return_value='{"queries": []}'):
        rc = main(
            [
                "query",
                "draft",
                "--skills",
                str(bodied_corpus),
                "--skill",
                "gke-basics",
                "--out",
                str(tmp_path / "empty_queries.json"),
                "--agent",
                "keyword",
            ]
        )
    assert rc == 1
    err = capsys.readouterr().err
    assert "0 verified queries" in err
