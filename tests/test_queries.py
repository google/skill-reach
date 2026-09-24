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

"""Verify QuerySet loading, schema validation, digests, and serialization."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest
from pydantic import ValidationError

from reach.models import NO_SKILL, Query, QueryKind
from reach.queries import (
    Origin,
    QuerySet,
    QuerySetProvenance,
    load_query_set,
    query_set_digest,
    save_query_set,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


def test_duplicate_ids_are_rejected(
    write_queries: Callable[..., Path],
) -> None:
    """Verify loading fails with ValueError if query set contains duplicate IDs."""
    rows = [
        {"id": "dupe", "text": "a", "kind": "implicit", "expected_skill": "s1"},
        {"id": "dupe", "text": "b", "kind": "implicit", "expected_skill": "s2"},
    ]
    with pytest.raises(ValueError, match="duplicate query ids"):
        load_query_set(write_queries(queries=rows, catalog_id="c"))


@pytest.mark.parametrize(
    "bad_kind",
    ["explicit", "", "NEGATIVE"],
    ids=["dropped", "empty", "wrong-case"],
)
def test_unknown_kind_is_rejected(
    write_queries: Callable[..., Path],
    bad_kind: str,
) -> None:
    """Verify unrecognized query kind string raises ValueError."""
    rows = [{"id": "q", "text": "a", "kind": bad_kind, "expected_skill": "s1"}]
    with pytest.raises(ValueError, match="kind"):
        load_query_set(write_queries(queries=rows, catalog_id="c"))


def test_a_query_may_go_unlabeled(
    write_queries: Callable[..., Path],
) -> None:
    """Verify query with omitted kind parses with kind=None."""
    rows = [{"id": "q", "text": "a", "expected_skill": "s1"}]
    assert load_query_set(write_queries(queries=rows, catalog_id="c")).queries[0].kind is None


def test_an_unlabeled_query_still_needs_ground_truth(
    write_queries: Callable[..., Path],
) -> None:
    """Verify query lacking expected_skill and kind raises ValueError."""
    rows = [{"id": "q", "text": "a"}]
    with pytest.raises(ValueError, match="expected_skill"):
        load_query_set(write_queries(queries=rows, catalog_id="c"))


def test_an_absent_kind_does_not_digest_as_a_kind(
    write_queries: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """Verify query_set_digest distinguishes between unlabeled kind and explicit kind."""
    unlabeled = [{"id": "q", "text": "a", "expected_skill": "s1"}]
    labeled = [{**unlabeled[0], "kind": "neighbor_negative"}]
    before = load_query_set(write_queries(queries=unlabeled, root=tmp_path / "a", catalog_id="c"))
    after = load_query_set(write_queries(queries=labeled, root=tmp_path / "b", catalog_id="c"))
    assert query_set_digest(after) != query_set_digest(before)


def test_a_set_names_the_catalog_it_was_written_for(
    write_queries: Callable[..., Path],
) -> None:
    """Verify loaded QuerySet preserves catalog_id from file."""
    rows = [{"id": "q", "text": "a", "kind": "implicit", "expected_skill": "s1"}]
    assert load_query_set(write_queries(queries=rows, catalog_id="c")).catalog_id == "c"


ROWS = [
    {"id": "q1", "text": "a", "kind": "implicit", "expected_skill": "s1"},
    {"id": "q2", "text": "b", "kind": "out_of_scope"},
]


def test_a_digest_ignores_where_the_set_sits(
    write_queries: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """Verify query_set_digest is independent of file path location."""
    here = load_query_set(write_queries(queries=ROWS, root=tmp_path / "a", catalog_id="c"))
    there = load_query_set(write_queries(queries=ROWS, root=tmp_path / "b", catalog_id="c"))
    assert query_set_digest(here) == query_set_digest(there)


def test_a_digest_ignores_the_order_rows_were_written_in(
    write_queries: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """Verify query_set_digest is invariant under row reordering."""
    forwards = load_query_set(write_queries(queries=ROWS, root=tmp_path / "a", catalog_id="c"))
    backwards = load_query_set(
        write_queries(queries=list(reversed(ROWS)), root=tmp_path / "b", catalog_id="c")
    )
    assert query_set_digest(forwards) == query_set_digest(backwards)


def test_a_digest_ignores_notes_nobody_is_scored_against(
    write_queries: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """Verify query_set_digest ignores editorial notes."""
    plain = load_query_set(write_queries(queries=ROWS, root=tmp_path / "a", catalog_id="c"))
    annotated = plain.model_copy(update={"notes": "reviewed 2026-08-21"})
    assert query_set_digest(annotated) == query_set_digest(plain)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("text", "rewritten"),
        ("kind", "contextual"),
        ("expected_skill", "s2"),
        ("id", "q9"),
    ],
)
def test_a_digest_moves_when_a_query_or_its_truth_moves(
    write_queries: Callable[..., Path],
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    """Verify modifying query fields alters query_set_digest."""
    before = load_query_set(write_queries(queries=ROWS, root=tmp_path / "a", catalog_id="c"))
    edited = [{**ROWS[0], field: value}, ROWS[1]]
    after = load_query_set(write_queries(queries=edited, root=tmp_path / "b", catalog_id="c"))
    assert query_set_digest(after) != query_set_digest(before)


def test_a_digest_tracks_acceptable_skills_without_caring_about_order(
    write_queries: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """Verify neutral skill membership affects the digest while ordering does not."""
    first = [
        {**ROWS[0], "acceptable_skills": ["router-a", "router-b"]},
        ROWS[1],
    ]
    reordered = [
        {**ROWS[0], "acceptable_skills": ["router-b", "router-a"]},
        ROWS[1],
    ]
    changed = [
        {**ROWS[0], "acceptable_skills": ["router-a", "router-c"]},
        ROWS[1],
    ]
    before = load_query_set(
        write_queries(queries=first, root=tmp_path / "a", catalog_id="c"),
    )
    same = load_query_set(
        write_queries(queries=reordered, root=tmp_path / "b", catalog_id="c"),
    )
    after = load_query_set(
        write_queries(queries=changed, root=tmp_path / "c", catalog_id="c"),
    )
    assert query_set_digest(same) == query_set_digest(before)
    assert query_set_digest(after) != query_set_digest(before)


def test_a_digest_moves_when_the_labeling_catalog_changes(
    write_queries: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """Verify changing catalog_id alters query_set_digest."""
    before = load_query_set(write_queries(queries=ROWS, root=tmp_path / "a", catalog_id="c"))
    rescoped = before.model_copy(update={"catalog_id": "neighborhood:s1"})
    assert query_set_digest(rescoped) != query_set_digest(before)


def provenance(**overrides) -> QuerySetProvenance:
    """Build a sample QuerySetProvenance instance with optional field overrides."""
    return QuerySetProvenance.model_validate(
        {
            "origin": Origin.GENERATED,
            "recorded_at": datetime(2026, 8, 22, 12, 0, tzinfo=UTC),
            "generator_model": "opus",
            "generator_arm": "content",
            "queries_per_target": 2,
            "rivals_in_view": 110,
            "bodies_digest": "b664ddebc7c9",
            "config_fingerprint": "9aadef3d65c9",
            **overrides,
        },
    )


def test_a_digest_ignores_how_the_set_was_made(
    write_queries: Callable[..., Path],
) -> None:
    """Verify query_set_digest is unaffected by provenance metadata."""
    plain = load_query_set(write_queries(queries=ROWS, catalog_id="c"))
    recorded = plain.model_copy(update={"provenance": provenance()})
    assert query_set_digest(recorded) == query_set_digest(plain)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("origin", Origin.IMPORTED),
        ("generator_model", "sonnet"),
        ("generator_arm", "framing"),
        ("rivals_in_view", 10),
        ("queries_per_target", 5),
    ],
)
def test_no_provenance_field_can_move_a_digest(
    write_queries: Callable[..., Path],
    field: str,
    value,
) -> None:
    """Verify changing individual provenance fields leaves query_set_digest unchanged."""
    plain = load_query_set(write_queries(queries=ROWS, catalog_id="c"))
    recorded = plain.model_copy(update={"provenance": provenance(**{field: value})})
    assert query_set_digest(recorded) == query_set_digest(plain)


def test_a_set_without_catalog_id_or_provenance_uses_defaults(tmp_path: Path) -> None:
    """Verify hand-authored query sets without catalog_id or provenance load with defaults."""
    minimal_model = QuerySet.model_validate(
        {
            "queries": [
                {
                    "id": "q1",
                    "text": "deploy container service",
                    "expected_skill": "container-deploy",
                }
            ]
        }
    )
    assert minimal_model.catalog_id == ""
    assert minimal_model.provenance.origin == Origin.AUTHORED

    custom_json = tmp_path / "queries_dir_test_5.json"
    custom_json.write_text(
        json.dumps(
            {
                "queries": [
                    {
                        "id": "q1",
                        "text": "deploy container service",
                        "expected_skill": "container-deploy",
                    },
                    {"text": "create k8s cluster", "expected_skill": "k8s-basics"},
                ]
            }
        ),
        encoding="utf-8",
    )
    loaded = load_query_set(custom_json)
    assert loaded.catalog_id == "all"
    assert loaded.provenance.origin == Origin.AUTHORED
    assert len(loaded.queries) == 2
    assert loaded.queries[0].id == "q1"
    assert loaded.queries[1].id == "q-002"

    custom_yaml = tmp_path / "queries.yaml"
    custom_yaml.write_text(
        "queries:\n"
        "  - id: q1\n"
        "    text: deploy container service\n"
        "    expected_skill: container-deploy\n",
        encoding="utf-8",
    )
    loaded_yaml = load_query_set(custom_yaml)
    assert loaded_yaml.catalog_id == "all"
    assert loaded_yaml.provenance.origin == Origin.AUTHORED
    assert len(loaded_yaml.queries) == 1


def test_query_accepts_query_alias_and_serializes_to_canonical_text(tmp_path: Path) -> None:
    """Verify Query accepts 'query' as alias for 'text' and serializes back to 'text'."""
    direct = Query.model_validate(
        {"id": "q-alias", "query": "restart database service", "expected_skill": "db-admin"}
    )
    assert direct.text == "restart database service"
    assert "query" not in direct.model_dump()
    assert direct.model_dump()["text"] == "restart database service"

    mixed_file = tmp_path / "mixed.json"
    mixed_file.write_text(
        json.dumps(
            {
                "queries": [
                    {"id": "q1", "text": "deploy container service", "expected_skill": "s1"},
                    {"id": "q2", "query": "restart database service", "expected_skill": "s2"},
                ]
            }
        ),
        encoding="utf-8",
    )
    loaded = load_query_set(mixed_file)
    assert len(loaded.queries) == 2
    assert loaded.queries[0].text == "deploy container service"
    assert loaded.queries[1].text == "restart database service"

    saved_path = save_query_set(loaded, tmp_path / "normalized.json")
    saved_raw = json.loads(saved_path.read_text(encoding="utf-8"))
    for item in saved_raw["queries"]:
        assert "text" in item
        assert "query" not in item


@pytest.mark.parametrize("bad_alias", ["prompt", "question", "utterance"])
def test_query_rejects_unauthorized_aliases(bad_alias: str) -> None:
    """Verify unauthorized field aliases are strictly rejected by Query schema."""
    with pytest.raises(ValidationError, match=r"text|Field required"):
        Query.model_validate(
            {"id": "q1", bad_alias: "deploy container service", "expected_skill": "s1"}
        )


def test_provenance_survives_a_trip_through_disk(tmp_path: Path) -> None:
    """Verify provenance metadata round-trips through disk save and load."""
    written = QuerySet(catalog_id="c", queries=(), provenance=provenance())
    reloaded = load_query_set(save_query_set(written, tmp_path / "out" / "set.json"))
    assert reloaded.provenance == written.provenance


def test_an_unrecognized_origin_is_refused() -> None:
    """Verify invalid origin string raises ValueError."""
    with pytest.raises(ValueError, match="origin"):
        QuerySetProvenance.model_validate({"origin": "invented"})


def test_a_recorded_time_must_carry_its_zone() -> None:
    """Verify naive datetime timestamps raise ValueError on QuerySetProvenance creation."""
    naive = datetime(2026, 8, 22, 12, 0, tzinfo=UTC).replace(tzinfo=None)
    with pytest.raises(ValueError, match=r"timezone.*info|timezone[-_ ]?aware"):
        QuerySetProvenance(origin=Origin.AUTHORED, recorded_at=naive)


def test_saving_a_set_writes_what_the_loader_reads(tmp_path: Path) -> None:
    """Verify QuerySet saved to disk equals loaded QuerySet."""
    written = QuerySet(catalog_id="c", notes="n", queries=(), provenance=provenance())
    assert load_query_set(save_query_set(written, tmp_path / "set.json")) == written


@pytest.mark.parametrize("filename", ["queries.jsonl", "queries.csv"])
def test_save_and_load_query_set_tabular(tmp_path: Path, filename: str) -> None:
    """Verify QuerySet saved to JSONL or CSV round-trips correctly through load_query_set."""
    initial = QuerySet(
        catalog_id="c",
        queries=(
            Query(
                id="q1",
                text="tier cold objects",
                expected_skill="gcs-lifecycle-rules",
            ),
        ),
        provenance=provenance(),
    )
    saved = save_query_set(initial, tmp_path / filename)
    loaded = load_query_set(saved)
    assert len(loaded.queries) == 1
    assert loaded.queries[0].id == "q1"
    assert loaded.queries[0].text == "tier cold objects"
    assert loaded.queries[0].expected_skill == "gcs-lifecycle-rules"


@pytest.mark.parametrize(
    ("target", "expected_ids"),
    [
        ("s1", ("q1",)),
        ("s2", ("q2",)),
        ("s3", ()),
        (NO_SKILL, ("q3",)),
        ("(no skill)", ("q3",)),
    ],
)
def test_query_set_for_skill_lookup(target: str, expected_ids: tuple[str, ...]) -> None:
    """Verify QuerySet.for_skill matches expected skill names and out-of-scope sentinels."""
    q1 = Query(id="q1", text="text 1", expected_skill="s1")
    q2 = Query(id="q2", text="text 2", expected_skill="s2")
    q3 = Query(id="q3", text="out of scope", kind=QueryKind.OUT_OF_SCOPE)
    qs = QuerySet(catalog_id="c", queries=(q1, q2, q3), provenance=provenance())
    assert tuple(q.id for q in qs.for_skill(target)) == expected_ids


def test_query_draft_sync_targets_missing_and_stale_skills(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify --sync skips in-sync skills and targets both missing and updated skills."""
    from io import StringIO

    from reach.cli.query import _handle_draft_query_generation
    from reach.generate import skill_body_digest
    from reach.models import Skill
    from reach.views import Console

    skills_dir = tmp_path / "skills"
    for name in ("skill-a", "skill-b", "skill-c"):
        s_dir = skills_dir / name
        s_dir.mkdir(parents=True)
        (s_dir / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: Desc for {name}\n---\nBody for {name}\n",
            encoding="utf-8",
        )

    skill_a = Skill(name="skill-a", description="Desc for skill-a", path=skills_dir / "skill-a")
    digest_a = skill_body_digest(skill_a)

    # Edit skill-b's body so its recorded digest becomes stale, while editing skill-a's
    # frontmatter description only (which should NOT mark skill-a stale).
    (skills_dir / "skill-a" / "SKILL.md").write_text(
        "---\nname: skill-a\ndescription: Optimized desc\n---\nBody for skill-a\n",
        encoding="utf-8",
    )
    (skills_dir / "skill-b" / "SKILL.md").write_text(
        "---\nname: skill-b\ndescription: Desc for skill-b\n---\nUpdated body for skill-b!\n",
        encoding="utf-8",
    )

    out_file = tmp_path / ".reach" / "queries.json"
    existing = QuerySet(
        catalog_id="all",
        queries=(
            Query(id="skill-a-1", text="query a", expected_skill="skill-a"),
            Query(id="skill-b-1", text="old query b", expected_skill="skill-b"),
        ),
        provenance=QuerySetProvenance(
            origin=Origin.GENERATED,
            skill_digests={
                "skill-a": digest_a,
                "skill-b": "000000000000",
            },
        ),
    )
    save_query_set(existing, out_file)

    captured: dict[str, object] = {}

    def fake_draft_query_set(
        console,
        settings,
        skills,
        generate,
        *,
        dry_run,
        review=False,
        existing_query_set=None,
    ) -> int:
        captured["targets"] = generate.targets
        captured["existing_count"] = len(existing_query_set.queries) if existing_query_set else 0
        return 0

    monkeypatch.setattr("reach.cli.query._draft_query_set", fake_draft_query_set)
    buf = StringIO()
    console = Console(file=buf, force_terminal=False, width=120)

    rc = _handle_draft_query_generation(
        console,
        target=skills_dir,
        count=2,
        out=out_file,
        format_opt=None,
        study=None,
        run_dir=None,
        config=None,
        catalog=None,
        runtime=None,
        generate=None,
        sync=True,
    )
    assert rc == 0
    # skill-c is missing, skill-b is stale, skill-a (frontmatter-only change) is still in sync
    assert captured["targets"] == ("skill-c", "skill-b")
    assert captured["existing_count"] == 2
    out_text = buf.getvalue()
    assert "1 missing" in out_text
    assert "1 updated" in out_text


def test_draft_query_set_prunes_stale_queries_and_updates_skill_digests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify _draft_query_set replaces stale skill queries and updates skill_digests."""
    from io import StringIO

    from reach.cli.drafting import _draft_query_set
    from reach.cli.flags import GenerateFlags
    from reach.config import CatalogSettings, RunConfig, StudySettings
    from reach.generate import (
        Citation,
        CitationTrail,
        citations_path,
        read_citations,
        skill_body_digest,
        write_citations,
    )
    from reach.models import CatalogMode, Skill
    from reach.runtime.fake import FakeGenerator
    from reach.views import build_console

    for name, body in (("skill-a", "Body A v1"), ("skill-b", "Body B v2 updated")):
        s_dir = tmp_path / name
        s_dir.mkdir(parents=True)
        (s_dir / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: Desc {name}\n---\n{body}\n",
            encoding="utf-8",
        )

    skill_a = Skill(name="skill-a", description="Desc skill-a", path=tmp_path / "skill-a")
    skill_b = Skill(name="skill-b", description="Desc skill-b", path=tmp_path / "skill-b")
    skills = (skill_a, skill_b)

    dest = tmp_path / "queries.json"
    existing_qs = QuerySet(
        catalog_id="all",
        queries=(
            Query(id="skill-a-1", text="Keep query A", expected_skill="skill-a"),
            Query(id="skill-b-1", text="Stale query B", expected_skill="skill-b"),
            Query(
                id="adv-skill-b-1",
                text="Stale adv B",
                expected_skill="skill-a",
                kind=QueryKind.NEIGHBOR_NEGATIVE,
            ),
        ),
        provenance=QuerySetProvenance(
            origin=Origin.GENERATED,
            skill_digests={
                "skill-a": skill_body_digest(skill_a),
                "skill-b": "old-sha-1234",
            },
        ),
    )
    save_query_set(existing_qs, dest)
    write_citations(
        CitationTrail(
            (
                Citation(skill="skill-a", text="Keep query A", citation="Body A v1"),
                Citation(skill="skill-b", text="Stale query B", citation="Body B old"),
            )
        ),
        citations_path(dest),
    )

    fake_drafter = FakeGenerator(
        completion=json.dumps(
            {
                "queries": [
                    {
                        "text": "Fresh query B",
                        "citation": "Body B v2 updated",
                        "reason": "Updated body",
                    }
                ]
            }
        )
    )
    monkeypatch.setattr(
        "reach.cli.drafting._build_drafter_runtime",
        lambda *_a, **_k: fake_drafter,
    )

    buf = StringIO()
    console = build_console(file=buf, force_terminal=False, width=120)
    settings = RunConfig(
        catalog=CatalogSettings(mode=CatalogMode.ALL),
        study=StudySettings(queries=dest),
    )
    generate = GenerateFlags(count=1, targets=("skill-b",))

    rc = _draft_query_set(
        console,
        settings,
        skills,
        generate,
        dry_run=False,
        existing_query_set=existing_qs,
    )
    assert rc == 0
    saved = load_query_set(dest)
    assert [(q.id, q.text) for q in saved.queries] == [
        ("skill-a-1", "Keep query A"),
        ("skill-b-1", "Fresh query B"),
    ]
    assert saved.provenance.skill_digests == {
        "skill-a": skill_body_digest(skill_a),
        "skill-b": skill_body_digest(skill_b),
    }
    saved_citations = read_citations(citations_path(dest)).root
    assert [(c.skill, c.text) for c in saved_citations] == [
        ("skill-a", "Keep query A"),
        ("skill-b", "Fresh query B"),
    ]


def test_draft_backfill_resolves_id_collisions_and_preserves_provenance(
    tmp_path: Path,
) -> None:
    """Verify backfilling offsets colliding query IDs and preserves original provenance."""
    from io import StringIO

    from reach.cli.drafting import _execute_draft_generation
    from reach.cli.flags import GenerateFlags
    from reach.config import RunConfig, StudySettings
    from reach.generate import DraftCheckpoint
    from reach.models import Catalog, CatalogMode, Skill
    from reach.runtime.fake import FakeGenerator
    from reach.views import build_console

    skills = [
        Skill(
            name="skill-a",
            description="Perform action A.",
            path=tmp_path / "skill-a",
        ),
    ]
    catalog = Catalog(id="cat", mode=CatalogMode.ALL, skills=("skill-a",))
    dest = tmp_path / "queries.json"
    in_progress = tmp_path / "queries.json.drafting"

    original_recorded = datetime(2025, 1, 1, 12, 0, tzinfo=UTC)
    existing_prov = QuerySetProvenance(
        origin=Origin.IMPORTED,
        source="benchmark-v1.json",
        recorded_at=original_recorded,
        queries_per_target=5,
    )
    existing_qs = QuerySet(
        catalog_id="cat",
        queries=(Query(id="skill-a-1", text="Existing query 1", expected_skill="skill-a"),),
        provenance=existing_prov,
    )

    terms = DraftCheckpoint(
        fingerprint="test-fp",
        bodies="bodies-hash",
        catalog_id="cat",
        arm="content",
        count=1,
        generator_model="fake",
        targets=("skill-a",),
        covered=("skill-a",),
        drafted=existing_qs,
    )

    fake_drafter = FakeGenerator(
        completion=json.dumps(
            {
                "queries": [
                    {
                        "text": "Newly backfilled query",
                        "citation": "Perform action A.",
                        "reason": "Direct citation",
                    }
                ]
            }
        )
    )

    buf = StringIO()
    console = build_console(file=buf, force_terminal=False, width=120)
    settings = RunConfig(study=StudySettings(queries=dest))
    generate = GenerateFlags(count=1)

    s_dir = tmp_path / "skill-a"
    s_dir.mkdir(parents=True)
    (s_dir / "SKILL.md").write_text(
        "---\nname: skill-a\ndescription: Perform action A.\n---\nPerform action A.\n",
        encoding="utf-8",
    )

    rc = _execute_draft_generation(
        console,
        settings,
        catalog,
        skills,
        generate,
        fake_drafter,
        terms,
        drafting=("skill-a",),
        recovered=terms,
        destination=dest,
        in_progress=in_progress,
        keep=True,
        same_invocation_probe=False,
        then="probed {path}",
        existing_query_set=existing_qs,
    )
    assert rc == 0
    saved = load_query_set(dest)
    assert len(saved.queries) == 2
    assert [q.id for q in saved.queries] == ["skill-a-1", "skill-a-2"]
    assert saved.provenance.origin == Origin.IMPORTED
    assert saved.provenance.source == "benchmark-v1.json"
    assert saved.provenance.recorded_at == original_recorded


def test_draft_backfill_resolves_id_collisions_without_numeric_suffix(
    tmp_path: Path,
) -> None:
    """Verify backfill increments from 1 when existing ID has no numeric suffix."""
    from io import StringIO

    from reach.cli.drafting import _execute_draft_generation
    from reach.cli.flags import GenerateFlags
    from reach.config import RunConfig, StudySettings
    from reach.generate import DraftCheckpoint
    from reach.models import Catalog, CatalogMode, Skill
    from reach.runtime.fake import FakeGenerator
    from reach.views import build_console

    skills = [
        Skill(
            name="skill-a",
            description="Perform action A.",
            path=tmp_path / "skill-a",
        ),
    ]
    catalog = Catalog(id="cat", mode=CatalogMode.ALL, skills=("skill-a",))
    dest = tmp_path / "queries.json"
    in_progress = tmp_path / "queries.json.drafting"

    existing_qs = QuerySet(
        catalog_id="cat",
        queries=(
            Query(id="skill-a", text="Existing query without suffix", expected_skill="skill-a"),
        ),
        provenance=QuerySetProvenance(origin=Origin.AUTHORED),
    )

    terms = DraftCheckpoint(
        fingerprint="test-fp",
        bodies="bodies-hash",
        catalog_id="cat",
        arm="content",
        count=1,
        generator_model="fake",
        targets=("skill-a",),
        covered=("skill-a",),
        drafted=existing_qs,
    )

    fake_drafter = FakeGenerator(
        completion=json.dumps(
            {
                "queries": [
                    {
                        "text": "Newly drafted query",
                        "citation": "Perform action A.",
                        "reason": "Direct citation",
                    }
                ]
            }
        )
    )
    buf = StringIO()
    console = build_console(file=buf, force_terminal=False, width=120)
    settings = RunConfig(study=StudySettings(queries=dest))
    generate = GenerateFlags(count=1)

    s_dir = tmp_path / "skill-a"
    s_dir.mkdir(parents=True)
    (s_dir / "SKILL.md").write_text(
        "---\nname: skill-a\ndescription: Perform action A.\n---\nPerform action A.\n",
        encoding="utf-8",
    )

    rc = _execute_draft_generation(
        console,
        settings,
        catalog,
        skills,
        generate,
        fake_drafter,
        terms,
        drafting=("skill-a",),
        recovered=terms,
        destination=dest,
        in_progress=in_progress,
        keep=True,
        same_invocation_probe=False,
        then="probed {path}",
        existing_query_set=existing_qs,
    )
    assert rc == 0
    saved = load_query_set(dest)
    assert len(saved.queries) == 2
    assert [q.id for q in saved.queries] == ["skill-a", "skill-a-1"]


def test_query_set_covered_skills_filters_negatives_and_unassigned() -> None:
    """Verify covered_skills returns positive target skills excluding negatives."""
    from reach.models import Query, QueryKind
    from reach.queries import QuerySet

    qs = QuerySet(
        queries=(
            Query(id="q1", text="Task 1", expected_skill="skill-a", kind=QueryKind.IMPLICIT),
            Query(id="q2", text="Task 2", expected_skill="skill-b", kind=QueryKind.CONTEXTUAL),
            Query(
                id="q3", text="Task 3", expected_skill="skill-c", kind=QueryKind.NEIGHBOR_NEGATIVE
            ),
            Query(id="q4", text="Out of scope", expected_skill=None, kind=QueryKind.OUT_OF_SCOPE),
            Query(id="q5", text="Task 1 repeat", expected_skill="skill-a", kind=QueryKind.IMPLICIT),
        )
    )
    assert qs.covered_skills() == frozenset({"skill-a", "skill-b"})


@pytest.mark.parametrize(
    ("origin", "recorded_digest", "expected_stale"),
    [
        (Origin.AUTHORED, None, frozenset()),
        (Origin.GENERATED, None, frozenset()),
        (Origin.GENERATED, "MATCH", frozenset()),
        (Origin.GENERATED, "000000stale0", frozenset({"skill-a"})),
    ],
)
def test_stale_skills_provenance_states(
    tmp_path: Path,
    origin: Origin,
    recorded_digest: str | None,
    expected_stale: frozenset[str],
) -> None:
    """Verify stale_skills evaluates per-skill SHA digests and handles missing SKILL.md."""
    from reach.generate import skill_body_digest
    from reach.models import Skill

    s_dir = tmp_path / "skill-a"
    s_dir.mkdir(parents=True)
    (s_dir / "SKILL.md").write_text(
        "---\nname: skill-a\ndescription: Desc A\n---\nBody A\n",
        encoding="utf-8",
    )
    skill_a = Skill(name="skill-a", description="Desc A", path=s_dir)
    missing_disk_skill = Skill(
        name="ghost-skill", description="Ghost desc", path=tmp_path / "nonexistent"
    )
    assert len(skill_body_digest(missing_disk_skill)) == 12

    digests: dict[str, str] = {}
    if recorded_digest == "MATCH":
        digests["skill-a"] = skill_body_digest(skill_a)
    elif recorded_digest is not None:
        digests["skill-a"] = recorded_digest

    qs = QuerySet(
        queries=(Query(id="q1", text="t1", expected_skill="skill-a"),),
        provenance=QuerySetProvenance(origin=origin, skill_digests=digests),
    )
    assert qs.stale_skills((skill_a,)) == expected_stale


def test_provenance_skill_digests_string_constraints_and_partial_sync(
    tmp_path: Path,
) -> None:
    """Verify Pydantic StringConstraints on skill_digests and partial-sync digest preservation."""
    from pydantic import ValidationError

    from reach.generate import skill_body_digest
    from reach.models import Skill

    # 1. StringConstraints normalizes whitespace/casing and rejects empty skill keys or digests
    prov = QuerySetProvenance(skill_digests={"  skill-a ": " ABCDEF123456 "})
    assert prov.skill_digests == {"skill-a": "abcdef123456"}
    with pytest.raises(ValidationError):
        QuerySetProvenance(skill_digests={"": "abcdef123456"})
    with pytest.raises(ValidationError):
        QuerySetProvenance(skill_digests={"skill-a": "   "})

    # 2. Partial sync updates only drafted_targets and preserves un-redrafted skills' digests
    for name, body in (("skill-a", "Modified body A on disk"), ("skill-b", "Fresh body B")):
        s_dir = tmp_path / name
        s_dir.mkdir(parents=True)
        (s_dir / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: Desc {name}\n---\n{body}\n",
            encoding="utf-8",
        )
    skill_a = Skill(name="skill-a", description="Desc skill-a", path=tmp_path / "skill-a")
    skill_b = Skill(name="skill-b", description="Desc skill-b", path=tmp_path / "skill-b")

    initial_prov = QuerySetProvenance(
        origin=Origin.GENERATED,
        skill_digests={"skill-a": "old-recorded-sha", "skill-b": "old-b-sha"},
    )
    updated_prov = initial_prov.with_updated_digests(
        (skill_a, skill_b),
        drafted_targets=("skill-b",),
        covered_targets=("skill-a", "skill-b"),
    )
    # skill-a was NOT re-drafted, so its old recorded SHA is preserved (remaining stale)
    assert updated_prov.skill_digests["skill-a"] == "old-recorded-sha"
    assert updated_prov.skill_digests["skill-b"] == skill_body_digest(skill_b)

    # 3. without_skills('cloud') prunes adv-cloud-01 without pruning adv-cloud-run-01
    qs_prefix = QuerySet(
        queries=(
            Query(
                id="adv-cloud-01",
                text="t1",
                expected_skill="other",
                kind=QueryKind.NEIGHBOR_NEGATIVE,
            ),
            Query(
                id="adv-cloud-run-01",
                text="t2",
                expected_skill="other",
                kind=QueryKind.NEIGHBOR_NEGATIVE,
            ),
        ),
    )
    pruned_prefix = qs_prefix.without_skills(("cloud",))
    assert [q.id for q in pruned_prefix.queries] == ["adv-cloud-run-01"]


def _invoke_draft_cli(
    skills_dir: Path,
    out_file: Path,
    *,
    sync: bool = False,
    force: bool = False,
) -> tuple[int, str]:
    """Run _handle_draft_query_generation with a captured test console."""
    from io import StringIO

    from reach.cli.query import _handle_draft_query_generation
    from reach.views import build_console

    buf = StringIO()
    console = build_console(file=buf, force_terminal=False, width=120)
    rc = _handle_draft_query_generation(
        console,
        target=skills_dir,
        count=1,
        out=out_file,
        format_opt=None,
        study=None,
        run_dir=None,
        config=None,
        catalog=None,
        runtime=None,
        generate=None,
        force=force,
        sync=sync,
    )
    return rc, buf.getvalue()


def test_query_draft_sync_sad_paths_and_backfill(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify --sync sad paths (--force conflict) and zero-LLM digest backfill."""
    from io import StringIO

    from reach.cli.query import _render_query_view
    from reach.generate import skill_body_digest
    from reach.models import Skill
    from reach.views import build_console

    skills_dir = tmp_path / "skills"
    s_dir = skills_dir / "skill-a"
    s_dir.mkdir(parents=True)
    (s_dir / "SKILL.md").write_text(
        "---\nname: skill-a\ndescription: Desc A\n---\nBody A\n",
        encoding="utf-8",
    )
    skill_a = Skill(name="skill-a", description="Desc A", path=s_dir)
    out_file = tmp_path / ".reach" / "queries.json"

    # 1. Sad path: --sync combined with --force fails with exit code 2
    rc_conflict, out_conflict = _invoke_draft_cli(skills_dir, out_file, sync=True, force=True)
    assert rc_conflict == 2
    assert "Cannot combine --sync with --force" in out_conflict

    # 2. GENERATED query set without skill_digests backfills digests in-place without LLM calls
    save_query_set(
        QuerySet(
            catalog_id="all",
            queries=(Query(id="skill-a-1", text="query a", expected_skill="skill-a"),),
            provenance=QuerySetProvenance(origin=Origin.GENERATED, skill_digests={}),
        ),
        out_file,
    )

    def fail_if_drafted(*_a, **_k):
        pytest.fail("LLM drafter should not be called when backfilling missing digests")

    monkeypatch.setattr("reach.cli.query._draft_query_set", fail_if_drafted)

    rc_backfill, out_backfill = _invoke_draft_cli(skills_dir, out_file, sync=True)
    assert rc_backfill == 0
    assert "All 1 skill(s) are in sync" in out_backfill
    reloaded = load_query_set(out_file)
    assert reloaded.provenance.skill_digests == {"skill-a": skill_body_digest(skill_a)}

    # 3. Subsequent --sync when already in sync is a clean no-op
    rc_noop, out_noop = _invoke_draft_cli(skills_dir, out_file, sync=True)
    assert rc_noop == 0
    assert "All 1 skill(s) are in sync" in out_noop

    # 4. Modifying skill-a body triggers out-of-sync warning in view mode and draft collision
    (s_dir / "SKILL.md").write_text(
        "---\nname: skill-a\ndescription: Desc A\n---\nModified body A\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"1 updated skill\(s\) out of sync"):
        _invoke_draft_cli(skills_dir, out_file, sync=False, force=False)

    buf_view = StringIO()
    console_view = build_console(file=buf_view, force_terminal=False, width=120)
    rc_view = _render_query_view(
        console_view,
        reloaded,
        out_file,
        skills=skills_dir,
        agent=None,
        global_scope=False,
        show_leaks=False,
        show_citations=False,
    )
    assert rc_view == 0
    assert "query set is out of sync with corpus" in buf_view.getvalue()
    assert "1 updated (skill-a)" in buf_view.getvalue()
