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

"""Verify KeywordRuntime matching performance, specificity, word boundaries, and symlinks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from reach.models import Catalog, CatalogMode, Skill
from reach.runtime.keyword import KeywordGenerator, KeywordOptions, KeywordRuntime

if TYPE_CHECKING:
    from pathlib import Path


def _mock_skills(root: Path, names: tuple[str, ...]) -> list[Skill]:
    skills = []
    for name in names:
        p = root / name
        p.mkdir(parents=True, exist_ok=True)
        (p / "SKILL.md").write_text(f"# {name}\n", encoding="utf-8")
        skills.append(Skill(name=name, description=f"Description for {name}", path=p))
    return skills


def test_keyword_runtime_select_exact_and_space_separated(tmp_path: Path) -> None:
    """Verify matching works for both kebab-case and space-separated skill names."""
    names = ("cloud-sql", "cloud-storage")
    skills = _mock_skills(tmp_path / "src", names)
    catalog = Catalog(id="cat1", mode=CatalogMode.ALL, skills=names)
    runtime = KeywordRuntime()
    runtime.install(catalog, skills, tmp_path / "work")

    # Kebab-case
    outcome1 = runtime.select("Please query database with cloud-sql", tmp_path)
    assert outcome1.invoked_skill == "cloud-sql"

    # Space-separated
    outcome2 = runtime.select("Upload objects to cloud storage bucket", tmp_path)
    assert outcome2.invoked_skill == "cloud-storage"


@pytest.mark.parametrize(
    ("query", "expected_skill"),
    [
        ("Deploy batch task on cloud-run-jobs", "cloud-run-jobs"),
        ("Deploy web service on cloud-run", "cloud-run"),
        (
            "Create and schedule social media dispatches using chronicle-publisher",
            "chronicle-publisher",
        ),
        ("Write a Go worker for cloud-run-jobs", "cloud-run-jobs"),
    ],
)
def test_keyword_runtime_specificity_priority(
    tmp_path: Path,
    query: str,
    expected_skill: str,
) -> None:
    """Verify longer skill names take precedence over shorter prefix or generic matches."""
    names = ("cloud-run", "cloud-run-jobs", "social", "go", "chronicle-publisher")
    skills = _mock_skills(tmp_path / "src", names)
    catalog = Catalog(id="cat1", mode=CatalogMode.ALL, skills=names)
    runtime = KeywordRuntime()
    runtime.install(catalog, skills, tmp_path / "work")

    outcome = runtime.select(query, tmp_path)
    assert outcome.invoked_skill == expected_skill


def test_keyword_runtime_word_boundary_prevents_partial_word_matches(tmp_path: Path) -> None:
    """Verify short skill names do not falsely match substrings inside unrelated words."""
    names = ("log", "ai", "sql", "go")
    skills = _mock_skills(tmp_path / "src", names)
    catalog = Catalog(id="cat1", mode=CatalogMode.ALL, skills=names)
    runtime = KeywordRuntime()
    runtime.install(catalog, skills, tmp_path / "work")

    # "catalog" contains "log", "obtain" contains "ai", "algorithm" contains "go"
    outcome1 = runtime.select("Look at the catalog to obtain an algorithm", tmp_path)
    assert outcome1.invoked_skill is None
    assert outcome1.invoked_skills == ()

    # Distinct tokens should match correctly
    outcome2 = runtime.select("View the system log output", tmp_path)
    assert outcome2.invoked_skill == "log"

    outcome3 = runtime.select("Ask the AI model for help", tmp_path)
    assert outcome3.invoked_skill == "ai"


def test_keyword_runtime_early_exit(tmp_path: Path) -> None:
    """Verify early exit flag is triggered when matched skill matches target_skill."""
    names = ("target-tool", "other-tool")
    skills = _mock_skills(tmp_path / "src", names)
    catalog = Catalog(id="cat1", mode=CatalogMode.ALL, skills=names)
    runtime = KeywordRuntime(options=KeywordOptions(early_exit=True))
    runtime.install(catalog, skills, tmp_path / "work")

    outcome_hit = runtime.select("Execute target-tool now", tmp_path, target_skill="target-tool")
    assert outcome_hit.early_exit is True

    outcome_miss = runtime.select("Execute other-tool now", tmp_path, target_skill="target-tool")
    assert outcome_miss.early_exit is False


def test_keyword_runtime_empty_query_and_catalog(tmp_path: Path) -> None:
    """Verify edge cases with empty queries, empty lines, or uninstalled runtime."""
    runtime = KeywordRuntime()
    outcome = runtime.select("", tmp_path)
    assert outcome.invoked_skill is None
    assert outcome.invoked_skills == ()

    summary = runtime.parse_stream([])
    assert summary.invoked_skill is None
    assert summary.invoked_skills == ()


def test_keyword_runtime_parse_stream_extracts_skill() -> None:
    """Verify KeywordRuntime extracts invoked skill from stream lines."""
    runtime = KeywordRuntime()
    summary = runtime.parse_stream(
        ["Model chose tool cloud-sql to proceed"],
        resident=("cloud-sql", "gcloud"),
    )
    assert summary.invoked_skill == "cloud-sql"
    assert summary.invoked_skills == ("cloud-sql",)
    assert summary.saw_result


def test_keyword_runtime_match_skill_helper() -> None:
    """Verify match_skill correctly matches skills and handles empty or missing inputs."""
    runtime = KeywordRuntime()
    runtime._set_resident(("cloud-run", "cloud-sql"))

    # Matches resident skill
    assert runtime.match_skill("Deploy app to cloud-run") == "cloud-run"
    assert runtime.match_skill("Query database using cloud sql") == "cloud-sql"

    # Dynamic resident override
    assert runtime.match_skill("Use bigquery storage", resident=("bigquery",)) == "bigquery"

    # Non-matches and empty cases
    assert runtime.match_skill("Unrelated query") is None
    assert runtime.match_skill("") is None
    assert runtime.match_skill("cloud-run", resident=()) is not None
    empty_runtime = KeywordRuntime()
    assert empty_runtime.match_skill("cloud-run") is None


def test_keyword_generator_complete() -> None:
    """Verify KeywordGenerator complete returns empty string and increments counter."""
    generator = KeywordGenerator()
    assert generator.complete("hello") == ""
    assert generator.completions == 1


def test_keyword_generator_synthesizes_grounded_queries_from_target_documentation() -> None:
    """Verify KeywordGenerator synthesizes valid cited JSON queries from target documentation."""
    from reach.generate import (
        build_adversarial_prompt,
        build_prompt,
        parse_response,
        verify_citation,
    )

    target_body = (
        "## Telemetry Relay Buffers\n"
        "Use Beacon Relay to buffer and forward high-volume telemetry packets.\n"
        "Configure ring buffer capacity, flush intervals, and batch compression.\n"
    )
    generator = KeywordGenerator()
    prompt = build_prompt(target_body, count=2)
    raw = generator.complete(prompt)

    drafts = parse_response(raw)
    assert len(drafts) == 2
    assert all(verify_citation(d, target_body) for d in drafts)
    assert len({d.text for d in drafts}) == 2

    adv_prompt = build_adversarial_prompt(
        target_body,
        rival_bodies=("Reconcile double-entry ledger journals and immutable tape snapshots.",),
        count=1,
    )
    adv_drafts = parse_response(generator.complete(adv_prompt))
    assert len(adv_drafts) == 1
    assert adv_drafts[0].rival_index == 1
    assert verify_citation(
        adv_drafts[0],
        "Reconcile double-entry ledger journals and immutable tape snapshots.",
    )


def test_keyword_runtime_bm25_description_fallback(tmp_path: Path) -> None:
    """Verify KeywordRuntime falls back to BM25 description scoring when literal name is absent."""
    skills = [
        Skill(
            name="beacon-relay",
            description="Buffer and forward high-volume telemetry packets and ring buffers.",
            path=tmp_path / "src" / "beacon-relay",
        ),
        Skill(
            name="vault-ledger",
            description=(
                "Manage double-entry ledger journals, reconciliations, and audit snapshots."
            ),
            path=tmp_path / "src" / "vault-ledger",
        ),
    ]
    for skill in skills:
        skill.path.mkdir(parents=True, exist_ok=True)
        (skill.path / "SKILL.md").write_text(f"# {skill.name}\n", encoding="utf-8")

    catalog = Catalog(
        id="cat1",
        mode=CatalogMode.ALL,
        skills=tuple(s.name for s in skills),
    )
    runtime = KeywordRuntime()
    runtime.install(catalog, skills, tmp_path / "work")

    # Query does not mention 'beacon-relay' or 'vault-ledger' literally
    outcome_relay = runtime.select(
        "How do I buffer high-volume telemetry packets with ring buffers?",
        tmp_path,
    )
    assert outcome_relay.invoked_skill == "beacon-relay"

    outcome_ledger = runtime.select(
        "Configure audit snapshots for double-entry ledger journals",
        tmp_path,
    )
    assert outcome_ledger.invoked_skill == "vault-ledger"

    # Literal skill name match still takes Priority 1 over description terms
    outcome_literal = runtime.select(
        "Use beacon-relay to export double-entry ledger journals",
        tmp_path,
    )
    assert outcome_literal.invoked_skill == "beacon-relay"


def test_keyword_generator_strips_urls_and_defers_blockquote_meta_instructions() -> None:
    """Verify KeywordGenerator strips URLs/paths and defers blockquote meta-instructions."""
    from reach.generate import build_prompt, parse_response, verify_citation

    target_body = (
        "> **Script paths** below are relative to this skill's directory.\n"
        "> **Freshness check**: If more than 30 days have passed since `last-updated`, warn user.\n"
        "> **Authentication failures**: If the CLI returns HTTP 401, update API_KEY and stop.\n\n"
        "Draft, schedule, and publish dispatch bulletins via [REDACTED] or when the user "
        "drops a bulletin URL such as https://chronicle.example.com/?w=<ws_id>&d=<draft_id>.\n"
        "Run the CLI via `./scripts/chronicle.js` to manage dispatch queues and schedules.\n"
    )
    generator = KeywordGenerator()
    drafts = parse_response(generator.complete(build_prompt(target_body, count=2)))
    assert len(drafts) == 2
    for draft in drafts:
        assert verify_citation(draft, target_body)
        assert "chronicle" not in draft.text
        assert "https://" not in draft.text
        assert "Freshness check" not in draft.text
        assert "Authentication failures" not in draft.text
