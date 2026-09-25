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

"""Verify skill loading, frontmatter parsing, catalog composition, and corpus digests."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, cast

import pytest
from pydantic import ValidationError

from reach.catalog import (
    _compute_cosine_bm25_distance_matrix,
    _extract_allowed_skills,
    _extract_declared_dependencies,
    build_catalogs,
    build_corpus_scaling_catalogs,
    build_corpus_scaling_queries,
    build_corpus_scaling_sequence,
    build_neighborhood_catalogs,
    build_scaling_catalogs,
    corpus_digest,
    find_cluster_medoids,
    find_skill_manifest,
    generate_log_scales,
    load_skills,
    parse_frontmatter,
    resolve_catalog,
    resolve_sweep_scales,
    split_frontmatter,
)
from reach.models import CatalogMode, Skill

if TYPE_CHECKING:
    from pathlib import Path


def test_loads_every_skill(skill_repo: Path) -> None:
    """Verify load_skills returns all skills in repository root sorted alphabetically."""
    skills = load_skills(skill_repo)
    assert [s.name for s in skills] == [
        "gcs-lifecycle-rules",
        "gcs-retention-policy",
        "gke-basics",
    ]
    assert skills[0].metadata.get("category") == "Storage"


def test_missing_root_raises(tmp_path: Path) -> None:
    """Verify NotADirectoryError is raised when skill directory does not exist."""
    with pytest.raises(NotADirectoryError):
        load_skills(tmp_path / "absent")


@pytest.mark.parametrize(
    "text",
    [
        "no frontmatter here",
        "---\nunterminated",
        "---\njust a string\n---\n",
        "---\nname: my-skill\ndescription: ''\n---\nbody",
        "---\nname: my-skill\ndescription: '   '\n---\nbody",
        "---\nname: my-skill\ndescription: null\n---\nbody",
    ],
    ids=[
        "absent",
        "unterminated",
        "not-a-mapping",
        "empty-description",
        "whitespace-description",
        "null-description",
    ],
)
def test_unparseable_frontmatter_is_skipped(text: str, tmp_path: Path) -> None:
    """Verify parse_frontmatter returns None for missing, unparseable, or invalid frontmatter."""
    assert parse_frontmatter(text, tmp_path / "x" / "SKILL.md") is None


def test_split_frontmatter_separates_the_yaml_from_the_body() -> None:
    """Verify split_frontmatter partitions YAML frontmatter from markdown body."""
    assert split_frontmatter("---\nname: x\n---\nbody text\n") == (
        "\nname: x\n",
        "\nbody text\n",
    )


@pytest.mark.parametrize(
    "text",
    ["no frontmatter here", "---\nunterminated"],
    ids=["absent", "unterminated"],
)
def test_split_frontmatter_is_none_without_a_closed_block(text: str) -> None:
    """Verify split_frontmatter returns None when frontmatter closing delimiter is missing."""
    assert split_frontmatter(text) is None


def test_split_frontmatter_only_consumes_the_first_two_delimiters() -> None:
    """Verify split_frontmatter preserves horizontal rules within markdown body."""
    split = split_frontmatter("---\nname: x\n---\nfirst\n---\nsecond\n")
    assert split is not None
    assert split[1] == "\nfirst\n---\nsecond\n"


def test_split_frontmatter_handles_unicode_bom() -> None:
    """Verify split_frontmatter strips leading Unicode BOM marker before parsing."""
    raw = "\ufeff---\nname: x\n---\nbody text\n"
    assert split_frontmatter(raw) == (
        "\nname: x\n",
        "\nbody text\n",
    )


def test_parse_frontmatter_handles_unicode_bom(tmp_path: Path) -> None:
    """Verify parse_frontmatter loads Skill from markdown starting with a Unicode BOM."""
    raw = "\ufeff---\nname: bom-skill\ndescription: A skill with a BOM\n---\nBody content\n"
    skill = parse_frontmatter(raw, tmp_path / "SKILL.md")
    assert skill is not None
    assert skill.name == "bom-skill"
    assert skill.description == "A skill with a BOM"


def test_split_frontmatter_handles_em_dash_in_description() -> None:
    """Verify split_frontmatter does not split on em-dash substrings in frontmatter values."""
    raw = '---\nname: em-skill\ndescription: "Automate---specifically Cloud Run"\n---\nBody\n'
    split = split_frontmatter(raw)
    assert split is not None
    assert 'description: "Automate---specifically Cloud Run"' in split[0]
    assert split[1] == "\nBody\n"


def test_parse_frontmatter_handles_em_dash_in_description(tmp_path: Path) -> None:
    """Verify parse_frontmatter correctly parses a description containing an em-dash."""
    raw = '---\nname: em-skill\ndescription: "Automate---specifically Cloud Run"\n---\nBody\n'
    skill = parse_frontmatter(raw, tmp_path / "SKILL.md")
    assert skill is not None
    assert skill.name == "em-skill"
    assert skill.description == "Automate---specifically Cloud Run"


def test_parse_frontmatter_handles_malformed_yaml(tmp_path: Path) -> None:
    """Verify parse_frontmatter returns None on malformed YAML instead of raising ScannerError."""
    raw = "---\nname: [unclosed list\n---\nBody\n"
    assert parse_frontmatter(raw, tmp_path / "SKILL.md") is None


def test_load_skills_handles_unicode_bom(tmp_path: Path) -> None:
    """Verify load_skills discovers and loads files encoded with a UTF-8 BOM."""
    skill_dir = tmp_path / "bom-skill"
    skill_dir.mkdir(parents=True)
    manifest = skill_dir / "SKILL.md"
    manifest.write_text(
        "\ufeff---\nname: bom-skill\ndescription: Saved with UTF-8 BOM\n---\nContent\n",
        encoding="utf-8",
    )
    skills = load_skills(tmp_path)
    assert len(skills) == 1
    assert skills[0].name == "bom-skill"


def test_a_file_that_is_not_a_skill_is_walked_past_not_stumbled_over(
    skill_repo: Path,
) -> None:
    """Verify non-skill markdown files without valid frontmatter are skipped during loading."""
    stray = skill_repo / "storage" / "template" / "SKILL.md"
    stray.parent.mkdir(parents=True)
    stray.write_text("# not a skill, just a heading\n", encoding="utf-8")
    assert [s.name for s in load_skills(skill_repo)] == [
        "gcs-lifecycle-rules",
        "gcs-retention-policy",
        "gke-basics",
    ]


def test_a_skill_installed_as_a_symlink_is_loaded(make_skill_root, tmp_path: Path) -> None:
    """Verify skills installed as directory symlinks are resolved and loaded."""
    store = make_skill_root(
        "store",
        {
            "gcs-lifecycle-rules": "Configures object lifecycle rules.",
            "gke-basics": "Explains GKE cluster fundamentals.",
        },
    )
    root = tmp_path / "installed"
    root.mkdir()
    for name in ("gcs-lifecycle-rules", "gke-basics"):
        (root / name).symlink_to(store / name, target_is_directory=True)

    assert [s.name for s in load_skills(root)] == [
        "gcs-lifecycle-rules",
        "gke-basics",
    ]


def test_a_skill_reached_two_ways_is_one_skill(make_skill_root) -> None:
    """Verify deduplication when a skill is reachable via multiple symlink paths."""
    root = make_skill_root("root", {"gke-basics": "Explains GKE cluster fundamentals."})
    (root / "alias").symlink_to(root / "gke-basics", target_is_directory=True)

    assert [s.name for s in load_skills(root)] == ["gke-basics"]


def test_a_symlink_loop_terminates(make_skill_root) -> None:
    """Verify recursive symlink loops terminate without infinite traversal."""
    root = make_skill_root("root", {"gke-basics": "Explains GKE cluster fundamentals."})
    (root / "self").symlink_to(root, target_is_directory=True)

    assert [s.name for s in load_skills(root)] == ["gke-basics"]


def test_a_broken_symlink_is_walked_past(make_skill_root, tmp_path: Path) -> None:
    """Verify broken directory symlinks are ignored without raising errors."""
    root = make_skill_root("root", {"gke-basics": "Explains GKE cluster fundamentals."})
    (root / "departed").symlink_to(tmp_path / "absent", target_is_directory=True)

    assert [s.name for s in load_skills(root)] == ["gke-basics"]


def test_a_symlinked_skill_takes_the_name_it_was_installed_under(
    tmp_path: Path,
) -> None:
    """Verify unnamed skills inherit the symlink directory name."""
    upstream = tmp_path / "checkout" / "upstream-name"
    upstream.mkdir(parents=True)
    (upstream / "SKILL.md").write_text(
        "---\ndescription: Explains GKE cluster fundamentals.\n---\n",
        encoding="utf-8",
    )
    root = tmp_path / "installed"
    root.mkdir()
    (root / "installed-name").symlink_to(upstream, target_is_directory=True)

    assert [s.name for s in load_skills(root)] == ["installed-name"]


def test_empty_description_is_rejected(tmp_path: Path) -> None:
    """Verify parse_frontmatter returns None when description is empty."""
    assert parse_frontmatter("---\nname: x\ndescription: ''\n---\n", tmp_path / "SKILL.md") is None


def test_singleton_mode_isolates_each_skill(skill_repo: Path) -> None:
    """Verify SINGLETON mode produces one single-skill catalog per skill."""
    catalogs = build_catalogs(load_skills(skill_repo), CatalogMode.SINGLETON)
    assert len(catalogs) == 3
    assert all(c.size == 1 for c in catalogs)


def test_all_mode_is_one_catalog(skill_repo: Path) -> None:
    """Verify ALL mode produces a single catalog containing all corpus skills."""
    (catalog,) = build_catalogs(load_skills(skill_repo), CatalogMode.ALL)
    assert catalog.size == 3


def test_empty_input_yields_no_catalogs() -> None:
    """Verify build_catalogs returns empty list when given empty skill list."""
    assert build_catalogs([], CatalogMode.ALL) == []


@pytest.mark.parametrize(
    "mode",
    [
        CatalogMode.ALL,
        CatalogMode.SINGLETON,
        CatalogMode.NEIGHBORHOOD,
    ],
)
def test_every_mode_composes_only_nonempty_catalogs(
    skill_repo: Path,
    mode: CatalogMode,
) -> None:
    """Verify every catalog mode composes non-empty catalogs when given non-empty skill list."""
    catalogs = build_catalogs(load_skills(skill_repo), mode, size=2, rivals=1)
    assert catalogs
    assert all(c.skills for c in catalogs)


def test_neighborhood_mode_carries_its_parameters(skill_repo: Path) -> None:
    """Verify NEIGHBORHOOD mode propagates size and rival parameters."""
    catalogs = build_catalogs(
        load_skills(skill_repo),
        CatalogMode.NEIGHBORHOOD,
        size=2,
        rivals=1,
    )
    assert len(catalogs) == 3
    assert all(c.size == 2 for c in catalogs)


def test_unsupported_mode_is_rejected(skill_repo: Path) -> None:
    """Verify build_catalogs raises ValueError for unhandled catalog modes."""
    unsupported = cast("CatalogMode", "sideways")
    with pytest.raises(ValueError, match="unsupported catalog mode"):
        build_catalogs(load_skills(skill_repo), unsupported)


def test_unread_frontmatter_is_kept_as_metadata(tmp_path: Path) -> None:
    """Verify unrecognized frontmatter fields are stored in the metadata dictionary."""
    text = (
        "---\nname: x\ndescription: does x\n"
        "metadata:\n  category: Storage\n  license: apache-2.0\n---\n"
    )
    skill = parse_frontmatter(text, tmp_path / "x" / "SKILL.md")
    assert skill is not None
    assert skill.metadata == {"category": "Storage", "license": "apache-2.0"}


@pytest.mark.parametrize(
    ("declaration", "invocable"),
    [
        ("", True),
        ("disable-model-invocation: true\n", False),
        ("disable-model-invocation: yes\n", False),
        ('disable-model-invocation: "true"\n', False),
        ("disable-model-invocation: 'On'\n", False),
        ("disable-model-invocation: 1\n", False),
        ("disable-model-invocation: false\n", True),
        ('disable-model-invocation: "false"\n', True),
        ("disable-model-invocation: 'no'\n", True),
        ("disable-model-invocation: 0\n", True),
        ("disable-model-invocation:\n", True),
        ("disable-model-invocation: someday\n", True),
    ],
    ids=[
        "absent",
        "bare-true",
        "bare-yes",
        "quoted-true",
        "quoted-on-mixed-case",
        "numeric-one",
        "bare-false",
        "quoted-false",
        "quoted-no",
        "numeric-zero",
        "empty",
        "unrecognized",
    ],
)
def test_the_hide_flag_is_read_in_the_spellings_yaml_can_hand_over(
    declaration: str,
    invocable: bool,
    tmp_path: Path,
) -> None:
    """Verify disable-model-invocation parses truthy YAML variants to set model_invocable."""
    text = f"---\nname: x\ndescription: does x\n{declaration}---\n"
    skill = parse_frontmatter(text, tmp_path / "x" / "SKILL.md")
    assert skill is not None
    assert skill.model_invocable is invocable


def test_the_hide_flag_is_not_also_filed_as_dialect(tmp_path: Path) -> None:
    """Verify disable-model-invocation is stripped from metadata when parsed."""
    text = (
        "---\nname: x\ndescription: does x\ndisable-model-invocation: true\n"
        "metadata:\n  category: Storage\n---\n"
    )
    skill = parse_frontmatter(text, tmp_path / "x" / "SKILL.md")
    assert skill is not None
    assert skill.metadata == {"category": "Storage"}
    assert skill.model_invocable is False


def test_a_hidden_skill_survives_the_walk_from_disk(make_skill_root) -> None:
    """Verify model_invocable flag is preserved through load_skills traversal."""
    root = make_skill_root("hidden", {"a": "does a"})
    skill_file = root / "a" / "SKILL.md"
    skill_file.write_text(
        skill_file.read_text(encoding="utf-8").replace(
            "name: a\n",
            "name: a\ndisable-model-invocation: true\n",
        ),
        encoding="utf-8",
    )
    (loaded,) = load_skills(root)
    assert loaded.model_invocable is False


def test_a_skill_without_metadata_loads(tmp_path: Path) -> None:
    """Verify skills without frontmatter metadata load with empty metadata dict."""
    skill = parse_frontmatter(
        "---\nname: x\ndescription: does x\n---\n",
        tmp_path / "x" / "SKILL.md",
    )
    assert skill is not None
    assert skill.metadata == {}


def test_the_corpus_digest_follows_the_selection_surface(
    corpus_builder,
    tmp_path: Path,
) -> None:
    """Verify corpus_digest changes when skill descriptions change or skills are added."""
    original = corpus_builder().add("a", "does a").add("b", "does b").build_skills(tmp_path)
    baseline = corpus_digest(original)
    edited_a = corpus_builder().add("a", "does a, and also c").build_skills(tmp_path)[0]
    edited = [edited_a, original[1]]
    added_c = corpus_builder().add("c", "does c").build_skills(tmp_path)[0]
    added = [*original, added_c]
    assert corpus_digest(edited) != baseline
    assert corpus_digest(added) != baseline
    assert corpus_digest(original) == baseline


def test_the_corpus_digest_of_a_visible_corpus_is_the_bytes_it_always_was(
    corpus_builder,
    tmp_path: Path,
) -> None:
    """Verify corpus_digest matches expected 12-character SHA-256 prefix of skill surface."""
    skills = corpus_builder().add("a", "does a").add("b", "does b").build_skills(tmp_path)
    material = "\n".join(f"{s.name}\n{s.description}" for s in skills)
    expected = hashlib.sha256(material.encode("utf-8")).hexdigest()[:12]
    assert corpus_digest(skills) == expected


def test_hiding_a_skill_makes_a_different_subject(corpus_builder, tmp_path: Path) -> None:
    """Verify toggling model_invocable modifies the computed corpus_digest."""
    visible = corpus_builder().add("a", "does a").add("b", "does b").build_skills(tmp_path)
    hidden_b = corpus_builder().add("b", "does b", model_invocable=False).build_skills(tmp_path)[0]
    hidden = [visible[0], hidden_b]
    other_hidden_a = (
        corpus_builder().add("a", "does a", model_invocable=False).build_skills(tmp_path)[0]
    )
    other_hidden = [other_hidden_a, visible[1]]

    assert corpus_digest(hidden) != corpus_digest(visible)
    assert corpus_digest(other_hidden) != corpus_digest(hidden)


def test_the_corpus_digest_ignores_location_and_order(corpus_builder, tmp_path: Path) -> None:
    """Verify corpus_digest is invariant to disk path locations and insertion order."""
    here = (
        corpus_builder()
        .add("a", "does a", path=tmp_path / "here" / "a")
        .add("b", "does b", path=tmp_path / "b")
        .build_skills()
    )
    there = (
        corpus_builder()
        .add("b", "does b", path=tmp_path / "elsewhere" / "deep" / "b")
        .add("a", "does a", path=tmp_path / "elsewhere" / "a")
        .build_skills()
    )
    assert corpus_digest(there) == corpus_digest(here)


def test_the_corpus_digest_is_short_enough_to_quote(skill_repo: Path) -> None:
    """Verify corpus_digest produces a 12-character string."""
    assert len(corpus_digest(load_skills(skill_repo))) == 12


def test_resolve_finds_a_catalog_by_id(skill_repo: Path) -> None:
    """Verify resolve_catalog returns catalog matching requested ID."""
    catalogs = build_catalogs(load_skills(skill_repo), CatalogMode.SINGLETON)
    assert resolve_catalog(catalogs, "singleton:gke-basics").size == 1


def test_resolve_names_the_alternatives(skill_repo: Path) -> None:
    """Verify resolve_catalog raises KeyError naming available catalog IDs on lookup failure."""
    catalogs = build_catalogs(load_skills(skill_repo), CatalogMode.SINGLETON)
    with pytest.raises(KeyError, match="singleton:gke-basics"):
        resolve_catalog(catalogs, "singleton:ghost")


@pytest.mark.parametrize(
    ("raw_text", "expected_name", "expected_desc"),
    [
        (
            "---\nname: unicode-skill\ndescription: ✨ Special chars & emojis 🚀\n---\n# Body\n",
            "unicode-skill",
            "✨ Special chars & emojis 🚀",
        ),
        (
            "---\nname: empty-body\ndescription: No content below\n---\n",
            "empty-body",
            "No content below",
        ),
        (
            "---\nname: quoted-desc\ndescription: 'Quoted: colons & dashes'\n---\nBody\n",
            "quoted-desc",
            "Quoted: colons & dashes",
        ),
    ],
)
def test_parse_frontmatter_handles_unicode_and_formatting_edge_cases(
    raw_text: str,
    expected_name: str,
    expected_desc: str,
    tmp_path: Path,
) -> None:
    """Verify parse_frontmatter handles unicode, empty bodies, and quoted strings cleanly."""
    skill_file = tmp_path / "skill" / "SKILL.md"
    skill_file.parent.mkdir(parents=True)
    skill_file.write_text(raw_text, encoding="utf-8")
    skill = parse_frontmatter(raw_text, skill_file)
    assert skill is not None
    assert skill.name == expected_name
    assert skill.description == expected_desc


def test_find_skill_manifest_finds_skills_lock_json(tmp_path: Path) -> None:
    """Verify find_skill_manifest traverses upward to locate skills-lock.json."""
    workspace = tmp_path / "workspace"
    skill_dir = workspace / ".agents" / "skills" / "my-skill"
    skill_dir.mkdir(parents=True)
    skill_file = skill_dir / "SKILL.md"
    skill_file.write_text("---\nname: my-skill\ndescription: Test\n---\n", encoding="utf-8")

    lockfile = workspace / "skills-lock.json"
    lockfile.write_text('{"version": 1, "skills": {}}', encoding="utf-8")

    found_from_file = find_skill_manifest(skill_file)
    assert found_from_file == lockfile

    found_from_dir = find_skill_manifest(skill_dir)
    assert found_from_dir == lockfile


def test_find_skill_manifest_finds_skills_json(tmp_path: Path) -> None:
    """Verify find_skill_manifest traverses upward to locate skills.json in publisher repo."""
    repo = tmp_path / "bundle-repo"
    skill_dir = repo / "skills" / "deploy-skill"
    skill_dir.mkdir(parents=True)
    skill_file = skill_dir / "SKILL.md"
    skill_file.write_text("---\nname: deploy-skill\ndescription: Test\n---\n", encoding="utf-8")

    manifest = repo / "skills.json"
    manifest.write_text('{"name": "bundle", "version": "1.0.0"}', encoding="utf-8")

    assert find_skill_manifest(skill_file) == manifest


def test_find_skill_manifest_stops_at_git_worktree_file(tmp_path: Path) -> None:
    """Verify find_skill_manifest respects .git pointer files in git worktrees."""
    worktree = tmp_path / "worktree"
    skill_dir = worktree / "skills" / "sub-skill"
    skill_dir.mkdir(parents=True)
    skill_file = skill_dir / "SKILL.md"
    skill_file.write_text("---\nname: sub-skill\ndescription: Test\n---\n", encoding="utf-8")

    # .git is a file in worktrees/submodules
    git_file = worktree / ".git"
    git_file.write_text("gitdir: /path/to/main/.git/worktrees/wt\n", encoding="utf-8")

    # Outside lockfile that should not be reached because .git stops traversal
    outside_lock = tmp_path / "skills-lock.json"
    outside_lock.write_text("{}", encoding="utf-8")

    assert find_skill_manifest(skill_file) is None


def test_extract_allowed_skills_from_string_and_list() -> None:
    """Verify _extract_allowed_skills parses Skill(X) tool definitions from string or sequence."""
    assert _extract_allowed_skills("Skill(gcloud-auth), Bash(cmd), Skill(deploy-helper)") == (
        "deploy-helper",
        "gcloud-auth",
    )
    assert _extract_allowed_skills(["Skill(alpha)", "Read", "Skill(beta)"]) == (
        "alpha",
        "beta",
    )
    assert _extract_allowed_skills(None) == ()


def test_extract_declared_dependencies_from_various_keys() -> None:
    """Verify _extract_declared_dependencies gathers and unions dependencies from metadata keys."""
    raw = {
        "metadata": {
            "requires_skill": "skill-a, skill-b",
            "depends_on": ["skill-c"],
            "requires": {
                "skills": ["skill-d", "skill-a"],
            },
        }
    }
    allowed = ("skill-e",)
    deps = _extract_declared_dependencies(raw, allowed)
    assert deps == ("skill-a", "skill-b", "skill-c", "skill-d", "skill-e")


def test_parse_frontmatter_populates_relationship_fields(tmp_path: Path) -> None:
    """Verify parse_frontmatter extracts tools, dependencies, and manifest_source."""
    workspace = tmp_path / "ws"
    skill_dir = workspace / ".agents" / "skills" / "my-skill"
    skill_dir.mkdir(parents=True)
    skill_file = skill_dir / "SKILL.md"

    lockfile = workspace / "skills-lock.json"
    lockfile.write_text(
        '{"version": 1, "skills": {"my-skill": {"source": "github.com/example/my-skill"}}}',
        encoding="utf-8",
    )

    text = (
        "---\n"
        "name: my-skill\n"
        "description: Deploy cloud infrastructure.\n"
        "allowed-tools: Skill(gcloud-auth), Bash(cmd)\n"
        "metadata:\n"
        "  requires_skill: setup-env\n"
        "---\n"
        "# Body\n"
    )
    skill_file.write_text(text, encoding="utf-8")

    skill = parse_frontmatter(text, skill_file)
    assert skill is not None
    assert skill.allowed_tools == ("gcloud-auth",)
    assert skill.declared_dependencies == ("gcloud-auth", "setup-env")
    assert skill.manifest_source == "github.com/example/my-skill"


def test_generate_log_scales() -> None:
    """Verify logarithmic 1-2-5 scale generation with adaptive floor up to total corpus size."""
    assert generate_log_scales(1) == (1,)
    assert generate_log_scales(2) == (1, 2)
    assert generate_log_scales(5) == (2, 5)
    assert generate_log_scales(8) == (2, 5, 8)
    assert generate_log_scales(12) == (2, 5, 10, 12)
    assert generate_log_scales(14) == (5, 10, 14)
    assert generate_log_scales(25) == (5, 10, 25)
    assert generate_log_scales(50) == (5, 10, 25, 50)
    assert generate_log_scales(60) == (10, 25, 50, 60)
    assert generate_log_scales(100) == (10, 25, 50, 100)
    assert generate_log_scales(132) == (10, 25, 50, 100, 132)
    assert generate_log_scales(350) == (10, 25, 50, 100, 200, 350)
    assert generate_log_scales(800) == (10, 25, 50, 100, 200, 400, 800)
    assert generate_log_scales(2000) == (10, 25, 50, 100, 200, 400, 800, 1600, 2000)

    # Explicit min_scale override
    assert generate_log_scales(50, min_scale=10) == (10, 25, 50)
    assert generate_log_scales(100, min_scale=5) == (5, 10, 25, 50, 100)

    with pytest.raises(ValueError, match="total_skills must be positive"):
        generate_log_scales(0)


def test_resolve_sweep_scales() -> None:
    """Verify scale resolution uses adaptive defaults and clamps requested overrides."""
    # Default without requested argument generates adaptive gold standard
    assert resolve_sweep_scales(14) == (5, 10, 14)
    assert resolve_sweep_scales(1) == (1,)
    assert resolve_sweep_scales(50) == (5, 10, 25, 50)
    assert resolve_sweep_scales(100) == (10, 25, 50, 100)
    assert resolve_sweep_scales(132) == (10, 25, 50, 100, 132)
    assert resolve_sweep_scales(5) == (2, 5)

    # Explicit requested overrides are respected and clamped to total_skills
    assert resolve_sweep_scales(14, requested=(1, 5, 10, 20)) == (1, 5, 10, 14)
    assert resolve_sweep_scales(100, requested=(5, 15, 30)) == (5, 15, 30)
    assert resolve_sweep_scales(100, requested=(100,)) == (100,)
    assert resolve_sweep_scales(50, requested=(10, 20, 150)) == (10, 20, 50)

    with pytest.raises(ValueError, match="total_skills must be positive"):
        resolve_sweep_scales(0)

    with pytest.raises(ValueError, match="must specify integers >= 1"):
        resolve_sweep_scales(10, requested=(0, -5))


def test_build_scaling_catalogs(tmp_path: Path) -> None:
    """Verify build_scaling_catalogs generates deterministically sized catalogs."""
    skills = [
        Skill(name=f"skill-{i:02d}", description=f"Skill {i} description", path=tmp_path / f"s{i}")
        for i in range(14)
    ]
    scales = resolve_sweep_scales(len(skills), requested=(1, 5, 10, 20))
    catalogs = build_scaling_catalogs(skills, target_skill="skill-00", scales=scales)

    assert len(catalogs) == 4
    assert [c.size for c in catalogs] == [1, 5, 10, 14]
    assert [c.id for c in catalogs] == [
        "sweep:skill-00:1",
        "sweep:skill-00:5",
        "sweep:skill-00:10",
        "sweep:skill-00:14",
    ]
    for c in catalogs:
        assert c.target == "skill-00"
        assert "skill-00" in c.skills
        assert c.mode is CatalogMode.SWEEP

    with pytest.raises(KeyError, match="target skill 'nonexistent' not in skills"):
        build_scaling_catalogs(skills, target_skill="nonexistent", scales=scales)


def test_build_catalogs_sweep_mode(tmp_path: Path) -> None:
    """Verify build_catalogs with CatalogMode.SWEEP returns valid scaling catalogs."""
    skills = [
        Skill(name=f"skill-{i:02d}", description=f"Skill {i} description", path=tmp_path / f"s{i}")
        for i in range(10)
    ]
    # Default without target_skill generates whole-corpus capacity scaling catalogs
    catalogs = build_catalogs(skills, mode=CatalogMode.SWEEP)
    assert len(catalogs) > 0
    assert all(c.mode is CatalogMode.SWEEP for c in catalogs)
    assert all(c.id.startswith("sweep:corpus:") for c in catalogs)
    assert all(c.target is None for c in catalogs)

    # Explicit target_skill generates targeted scaling catalogs
    target_catalogs = build_catalogs(skills, mode=CatalogMode.SWEEP, target_skill="skill-00")
    assert len(target_catalogs) > 0
    assert all(c.mode is CatalogMode.SWEEP for c in target_catalogs)
    assert all(c.id.startswith("sweep:skill-00:") for c in target_catalogs)
    assert all(c.target == "skill-00" for c in target_catalogs)


def test_build_scaling_catalogs_with_duplicate_skill_names(tmp_path: Path) -> None:
    """Verify build_scaling_catalogs handles corpora containing duplicate skill names."""
    skills = [
        Skill(name="target", description="Target skill", path=tmp_path / "target"),
        Skill(name="rival-1", description="Rival one primary", path=tmp_path / "r1_a"),
        Skill(name="rival-1", description="Rival one duplicate", path=tmp_path / "r1_b"),
        Skill(name="rival-2", description="Rival two primary", path=tmp_path / "r2"),
        Skill(name="rival-3", description="Rival three primary", path=tmp_path / "r3"),
    ]
    catalogs = build_scaling_catalogs(skills, target_skill="target", scales=(1, 3, 4))
    assert len(catalogs) == 3
    for cat in catalogs:
        assert len(cat.skills) == len(set(cat.skills)), (
            f"Catalog {cat.id} contains duplicate skills: {cat.skills}"
        )


def test_build_neighborhood_catalogs_with_duplicate_skill_names(tmp_path: Path) -> None:
    """Verify build_neighborhood_catalogs handles corpora containing duplicate skill names."""
    skills = [
        Skill(name="target", description="Target skill", path=tmp_path / "target"),
        Skill(name="rival-1", description="Rival one primary", path=tmp_path / "r1_a"),
        Skill(name="rival-1", description="Rival one duplicate", path=tmp_path / "r1_b"),
        Skill(name="rival-2", description="Rival two primary", path=tmp_path / "r2"),
    ]
    catalogs = build_neighborhood_catalogs(skills, size=3, rivals=2)
    assert len(catalogs) == 3
    for cat in catalogs:
        assert len(cat.skills) == len(set(cat.skills)), (
            f"Catalog {cat.id} contains duplicate skills: {cat.skills}"
        )


def test_cosine_bm25_distance_axioms(tmp_path: Path) -> None:
    """Verify Cosine-BM25 distance matrix satisfies metric axioms and bounds."""
    skills = [
        Skill(
            name="alpha",
            description="Deploy container services to Kubernetes engine",
            path=tmp_path / "a",
        ),
        Skill(
            name="beta",
            description="Configure relational database tables with Spanner SQL",
            path=tmp_path / "b",
        ),
        Skill(name="minimal", description="xyz", path=tmp_path / "minimal"),
    ]
    names, dist, sim = _compute_cosine_bm25_distance_matrix(skills)
    assert names == ("alpha", "beta", "minimal")
    n = len(names)
    for i in range(n):
        assert dist[i][i] == 0.0
        for j in range(n):
            assert dist[i][j] == dist[j][i]
            assert 0.0 <= dist[i][j] <= 1.0
            assert 0.0 <= sim[i][j] <= 1.0


def test_build_corpus_scaling_sequence(tmp_path: Path) -> None:
    """Verify Farthest-First Traversal starts at medoid and maximizes diversity."""
    skills = [
        Skill(
            name="s-shared1",
            description="cloud logging monitoring tracing metrics",
            path=tmp_path / "s1",
        ),
        Skill(
            name="s-shared2",
            description="cloud logging alerting dashboard metrics",
            path=tmp_path / "s2",
        ),
        Skill(
            name="s-shared3",
            description="cloud logging log export metrics",
            path=tmp_path / "s3",
        ),
        Skill(
            name="s-isolated",
            description="quantum physics simulation tensor network",
            path=tmp_path / "s4",
        ),
    ]
    sequence = build_corpus_scaling_sequence(skills)
    assert len(sequence) == 4
    # The shared logging skills have high pairwise similarity, so one of them will be the medoid
    assert sequence[0] in ("s-shared1", "s-shared2", "s-shared3")
    # The farthest point from the logging medoid should be the isolated quantum skill
    assert sequence[1] == "s-isolated"
    # Determinism: running twice returns identical sequence
    assert build_corpus_scaling_sequence(skills) == sequence


def test_build_corpus_scaling_catalogs(tmp_path: Path) -> None:
    """Verify build_corpus_scaling_catalogs creates deterministic nested catalogs."""
    skills = [
        Skill(name=f"skill-{i:02d}", description=f"Skill {i} description", path=tmp_path / f"s{i}")
        for i in range(12)
    ]
    scales = (2, 5, 10, 12)
    catalogs = build_corpus_scaling_catalogs(skills, scales=scales)
    assert len(catalogs) == 4
    assert [c.size for c in catalogs] == [2, 5, 10, 12]
    assert [c.id for c in catalogs] == [
        "sweep:corpus:2",
        "sweep:corpus:5",
        "sweep:corpus:10",
        "sweep:corpus:12",
    ]
    for c in catalogs:
        assert c.target is None
        assert c.mode is CatalogMode.SWEEP
        # Skills in catalog must be sorted alphabetically for prompt cache alignment
        assert list(c.skills) == sorted(c.skills)

    # Nested containment
    for i in range(len(catalogs) - 1):
        assert set(catalogs[i].skills).issubset(set(catalogs[i + 1].skills))


def test_build_corpus_scaling_queries() -> None:
    """Verify build_corpus_scaling_queries slices queries for installed skills."""
    from reach.models import Query
    from reach.queries import Origin, QuerySet, QuerySetProvenance

    raw_queries = [
        Query(id="q1", text="deploy alpha", expected_skill="alpha"),
        Query(id="q2", text="scale alpha", expected_skill="alpha"),
        Query(id="q3", text="deploy beta", expected_skill="beta"),
        Query(id="q4", text="scale beta", expected_skill="beta"),
        Query(id="q5", text="query gamma", expected_skill="gamma"),
    ]
    qset = QuerySet(
        catalog_id="test",
        queries=tuple(raw_queries),
        provenance=QuerySetProvenance(origin=Origin.AUTHORED),
    )

    # Scale with alpha installed
    q_alpha = build_corpus_scaling_queries(
        scale_skills=["alpha"],
        raw_query_set=qset,
    )
    assert len(q_alpha.queries) == 2
    assert {q.id for q in q_alpha.queries} == {"q1", "q2"}
    assert all(q.expected_skill == "alpha" for q in q_alpha.queries)

    # Scale with all skills installed
    q_all_installed = build_corpus_scaling_queries(
        scale_skills=["alpha", "beta", "gamma"],
        raw_query_set=qset,
    )
    assert len(q_all_installed.queries) == 5
    assert all(q.expected_skill is not None for q in q_all_installed.queries)


def test_corpus_scaling_plan(tmp_path: Path) -> None:
    """Verify CorpusScalingPlan generates consistent distance geometry, catalogs, and probes."""
    from reach.catalog import CorpusScalingPlan
    from reach.models import Query, QueryKind
    from reach.queries import Origin, QuerySet, QuerySetProvenance

    skills = [
        Skill(name="alpha", description="alpha kubernetes deployment", path=tmp_path / "alpha"),
        Skill(name="beta", description="beta kubernetes service", path=tmp_path / "beta"),
        Skill(name="gamma", description="gamma spanner database", path=tmp_path / "gamma"),
    ]
    raw_queries = [
        Query(id="q1", text="deploy alpha", expected_skill="alpha", kind=QueryKind.IMPLICIT),
        Query(id="q2", text="scale alpha", expected_skill="alpha", kind=QueryKind.IMPLICIT),
        Query(id="q3", text="deploy beta", expected_skill="beta", kind=QueryKind.IMPLICIT),
        Query(id="q4", text="query gamma", expected_skill="gamma", kind=QueryKind.IMPLICIT),
    ]
    qset = QuerySet(
        catalog_id="test",
        queries=tuple(raw_queries),
        provenance=QuerySetProvenance(origin=Origin.AUTHORED),
    )

    plan = CorpusScalingPlan.create(skills=skills, scales=(1, 2, 3))
    assert len(plan.catalogs) == 3
    assert [c.size for c in plan.catalogs] == [1, 2, 3]
    assert len(plan.distance_matrix) == 3
    assert len(plan.similarity_matrix) == 3
    assert set(plan.sequence) == {"alpha", "beta", "gamma"}

    # Test query slicing via plan
    q_scale_1 = plan.queries_for_scale(plan.catalogs[0], qset)
    assert len(q_scale_1.queries) > 0
    # In-scope probes match catalog skills
    for q in q_scale_1.queries:
        assert q.expected_skill is not None
        assert q.expected_skill in plan.catalogs[0].skills


def test_find_cluster_medoids(tmp_path: Path) -> None:
    """Verify find_cluster_medoids selects central archetypes from modularity clusters."""
    skills = [
        # Cluster A: Logging and observability
        Skill(name="log-view", description="cloud logging view entries", path=tmp_path / "s1"),
        Skill(name="log-export", description="cloud logging sink export", path=tmp_path / "s2"),
        Skill(name="log-alert", description="cloud monitoring alerting", path=tmp_path / "s3"),
        # Cluster B: Database management
        Skill(name="db-spanner", description="spanner database sql schema", path=tmp_path / "s4"),
        Skill(name="db-sql", description="cloud sql postgres query", path=tmp_path / "s5"),
        Skill(name="video-render", description="render video frames", path=tmp_path / "s6"),
        Skill(name="video-caption", description="generate video subtitles", path=tmp_path / "s7"),
    ]

    # Extract 3 cluster medoids
    medoids = find_cluster_medoids(skills, k=3)
    assert len(medoids) == 3

    # Medoids should span the 3 distinct domains
    assert any(m.startswith("log-") for m in medoids)
    assert any(m.startswith("db-") for m in medoids)
    assert any(m.startswith("video-") for m in medoids)

    # Determinism
    assert find_cluster_medoids(skills, k=3) == medoids


def test_find_cluster_medoids_edge_cases(tmp_path: Path) -> None:
    """Verify find_cluster_medoids handles edge cases gracefully."""
    skills = [
        Skill(name="s1", description="skill one", path=tmp_path / "s1"),
        Skill(name="s2", description="skill two", path=tmp_path / "s2"),
    ]
    # Non-positive k values
    assert find_cluster_medoids(skills, k=0) == ()
    assert find_cluster_medoids(skills, k=-1) == ()
    # Empty skills
    assert find_cluster_medoids([], k=5) == ()
    # k >= total skills returns all skills
    assert set(find_cluster_medoids(skills, k=5)) == {"s1", "s2"}
    assert len(find_cluster_medoids(skills, k=2)) == 2


def test_corpus_scaling_plan_with_anchors(tmp_path: Path) -> None:
    """Verify CorpusScalingPlan seeds sequence with anchor skills and fixes probe cohort."""
    from reach.catalog import CorpusScalingPlan
    from reach.models import Query, QueryKind
    from reach.queries import Origin, QuerySet, QuerySetProvenance

    skills = [
        Skill(name=f"skill-{i:02d}", description=f"Skill {i} description", path=tmp_path / f"s{i}")
        for i in range(10)
    ]
    raw_queries = [
        Query(
            id=f"q-{i:02d}",
            text=f"query for {i}",
            expected_skill=f"skill-{i:02d}",
            kind=QueryKind.IMPLICIT,
        )
        for i in range(10)
    ]
    qset = QuerySet(
        catalog_id="test",
        queries=tuple(raw_queries),
        provenance=QuerySetProvenance(origin=Origin.AUTHORED),
    )

    anchors = ("skill-03", "skill-07")
    plan = CorpusScalingPlan.create(skills=skills, scales=(2, 5, 10), anchor_skills=anchors)

    # Anchors appear at the start of sequence
    assert plan.sequence[:2] == ("skill-03", "skill-07")
    assert plan.anchor_skills == ("skill-03", "skill-07")

    # Every scale K >= len(anchors) contains all anchors
    for cat in plan.catalogs:
        assert "skill-03" in cat.skills
        assert "skill-07" in cat.skills

    # Query slicing with anchors only evaluates queries for anchor skills
    for cat in plan.catalogs:
        sliced_queries = plan.queries_for_scale(cat, qset)
        # Should evaluate exactly the 2 anchor queries across all scales
        assert len(sliced_queries.queries) == 2
        assert {q.expected_skill for q in sliced_queries.queries} == {"skill-03", "skill-07"}


def test_corpus_scaling_plan_low_discrepancy_striding(tmp_path: Path) -> None:
    """Verify CorpusScalingPlan interleaves rivals proportionally across scales."""
    from reach.catalog import CorpusScalingPlan

    skills = [
        # Anchor 1 and its rivals
        Skill(name="a1", description="cloud storage bucket lifecycle", path=tmp_path / "a1"),
        Skill(name="a1-r1", description="cloud storage bucket policy", path=tmp_path / "a1_r1"),
        Skill(name="a1-r2", description="cloud storage bucket acl", path=tmp_path / "a1_r2"),
        # Anchor 2 and its rivals
        Skill(name="a2", description="kubernetes container deployment", path=tmp_path / "a2"),
        Skill(name="a2-r1", description="kubernetes cluster upgrade", path=tmp_path / "a2_r1"),
        Skill(name="a2-r2", description="kubernetes ingress service", path=tmp_path / "a2_r2"),
        # Orthogonal filler
        Skill(name="f1", description="quantum physics simulation", path=tmp_path / "f1"),
        Skill(name="f2", description="astronomy planetary mechanics", path=tmp_path / "f2"),
        Skill(name="f3", description="biology sequence alignment", path=tmp_path / "f3"),
        Skill(name="f4", description="geology seismic wave propagation", path=tmp_path / "f4"),
    ]
    anchors = ("a1", "a2")
    plan = CorpusScalingPlan.create(skills=skills, scales=(2, 4, 7, 10), anchor_skills=anchors)

    assert plan.sequence[:2] == ("a1", "a2")
    assert len(plan.sequence) == 10
    assert set(plan.sequence) == {s.name for s in skills}

    # At K=4 (adding 2 distractors), rivals should not be locked out until K=10
    cat4_skills = set(plan.catalogs[1].skills)
    assert "a1" in cat4_skills
    assert "a2" in cat4_skills

    # Sibling rivals must arrive across intermediate scales, not all at index >= 8
    rival_indices = [plan.sequence.index(r) for r in ("a1-r1", "a1-r2", "a2-r1", "a2-r2")]
    # At least one rival should arrive before scale 6
    assert any(idx < 6 for idx in rival_indices)


def test_build_scaling_sequence_nested_containment(tmp_path: Path) -> None:
    """Verify scaling catalogs satisfy strict subset containment S_K1 subset S_K2."""
    from reach.catalog import CorpusScalingPlan

    skills = [
        Skill(
            name=f"skill_{i:02d}",
            description=f"cloud system capability {i}",
            path=tmp_path / f"s{i}",
        )
        for i in range(20)
    ]
    anchors = ("skill_00", "skill_05", "skill_10")
    scales = (3, 6, 10, 15, 20)
    plan = CorpusScalingPlan.create(skills=skills, scales=scales, anchor_skills=anchors)

    # 1. Determinism
    plan2 = CorpusScalingPlan.create(skills=skills, scales=scales, anchor_skills=anchors)
    assert plan.sequence == plan2.sequence

    # 2. Strict nested catalog containment
    for i in range(len(plan.catalogs) - 1):
        c_current = set(plan.catalogs[i].skills)
        c_next = set(plan.catalogs[i + 1].skills)
        assert c_current.issubset(c_next)
        assert len(c_current) < len(c_next)

    # 3. Anchors present in every catalog
    for cat in plan.catalogs:
        for a in anchors:
            assert a in cat.skills


def test_find_cluster_medoids_display_quantiles(tmp_path: Path) -> None:
    """Verify cluster medoids balance semantic centrality with alphabetical display quantiles."""
    skills = [
        # Cluster Alpha (late alphabet initial z)
        Skill(
            name="z_cluster_lead",
            description="kubernetes pod deployment service",
            path=tmp_path / "z1",
        ),
        Skill(
            name="z_cluster_sub",
            description="kubernetes container deployment cluster",
            path=tmp_path / "z2",
        ),
        # Cluster Beta (early alphabet initial a)
        Skill(
            name="a_cluster_lead",
            description="database sql postgres schema",
            path=tmp_path / "a1",
        ),
        Skill(
            name="a_cluster_sub",
            description="database relational table query",
            path=tmp_path / "a2",
        ),
        # Cluster Gamma (mid alphabet initial m)
        Skill(
            name="m_cluster_lead",
            description="machine learning neural network train",
            path=tmp_path / "m1",
        ),
        Skill(
            name="m_cluster_sub",
            description="machine learning model inference tensor",
            path=tmp_path / "m2",
        ),
    ]

    medoids = find_cluster_medoids(skills, k=3)
    assert len(medoids) == 3
    # Anchors should be chosen across distinct alphabet groups
    assert any(m.startswith("a_") for m in medoids)
    assert any(m.startswith("m_") for m in medoids)
    assert any(m.startswith("z_") for m in medoids)


def test_low_discrepancy_striding_intra_cluster_threat_ordering() -> None:
    """Verify _low_discrepancy_striding preserves descending similarity within clusters."""
    from reach.catalog import _low_discrepancy_striding

    names = ["a0", "a1", "d_high", "d_mid", "d_low", "e_high", "e_mid", "e_low"]
    sim = [[0.0] * 8 for _ in range(8)]
    sim[2][0] = 0.9
    sim[3][0] = 0.6
    sim[4][0] = 0.3
    sim[5][1] = 0.9
    sim[6][1] = 0.6
    sim[7][1] = 0.3

    order = _low_discrepancy_striding(names, sim, [0, 1])
    ordered_names = [names[i] for i in order]

    assert ordered_names[:2] == ["a0", "a1"]

    d_indices = [ordered_names.index(name) for name in ("d_high", "d_mid", "d_low")]
    assert d_indices == sorted(d_indices)

    e_indices = [ordered_names.index(name) for name in ("e_high", "e_mid", "e_low")]
    assert e_indices == sorted(e_indices)

    assert ordered_names.index("d_high") < ordered_names.index("e_low")
    assert ordered_names.index("e_high") < ordered_names.index("d_low")


def test_find_cluster_medoids_configurable_parameters(tmp_path: Path) -> None:
    """Verify find_cluster_medoids accepts configurable ratios and weights."""
    skills = [
        Skill(name="k1", description="kubernetes cluster pod", path=tmp_path / "k1"),
        Skill(name="k2", description="kubernetes cluster container", path=tmp_path / "k2"),
        Skill(name="d1", description="database postgres table", path=tmp_path / "d1"),
        Skill(name="d2", description="database postgres schema", path=tmp_path / "d2"),
    ]
    medoids = find_cluster_medoids(
        skills,
        k=2,
        near_optimal_ratio=0.85,
        display_quantile_weight=0.25,
    )
    assert len(medoids) == 2


def test_build_corpus_scaling_queries_with_anchors() -> None:
    """Verify build_corpus_scaling_queries restricts evaluation to installed anchor skills."""
    from reach.models import Query
    from reach.queries import Origin, QuerySet, QuerySetProvenance

    queries = [
        Query(id="q1", text="text 1", expected_skill="s1"),
        Query(id="q2", text="text 2", expected_skill="s2"),
        Query(id="q3", text="text 3", expected_skill="s3"),
    ]
    qset = QuerySet(
        catalog_id="test",
        queries=tuple(queries),
        provenance=QuerySetProvenance(origin=Origin.AUTHORED),
    )

    # Catalog contains s1, s2, s3, but anchor is only s1
    res = build_corpus_scaling_queries(
        scale_skills=["s1", "s2", "s3"],
        raw_query_set=qset,
        anchor_skills=["s1"],
    )
    assert len(res.queries) == 1
    assert res.queries[0].expected_skill == "s1"

    # When anchor is None, all installed skills are included
    res_all = build_corpus_scaling_queries(
        scale_skills=["s1", "s2", "s3"],
        raw_query_set=qset,
        anchor_skills=None,
    )
    assert len(res_all.queries) == 3


@pytest.mark.parametrize("empty_name", ["", "   ", "\t\n"])
def test_skill_name_requires_non_empty(empty_name: str, tmp_path: Path) -> None:
    """Verify Skill model raises ValidationError when name is empty or whitespace."""
    with pytest.raises(ValidationError):
        Skill(
            name=empty_name,
            description="A valid description.",
            path=tmp_path,
        )


@pytest.mark.parametrize(
    "valid_name",
    [
        "code-review",
        "pdf-processing",
        "++",
        "my_skill",
        "SkillWithCamelCase",
    ],
)
def test_skill_name_accepts_non_empty(valid_name: str, tmp_path: Path) -> None:
    """Verify Skill model accepts non-empty names across naming conventions."""
    skill = Skill(
        name=valid_name,
        description="A valid description.",
        path=tmp_path,
    )
    assert skill.name == valid_name


def test_resolve_skill_target_none_returns_none() -> None:
    """Verify resolve_skill_target returns None when target is None or empty."""
    from reach.catalog import resolve_skill_target

    assert resolve_skill_target(None) is None
    assert resolve_skill_target("") is None
    assert resolve_skill_target("   ") is None


def test_resolve_skill_target_raw_name() -> None:
    """Verify resolve_skill_target accepts plain skill names without filesystem paths."""
    from reach.catalog import resolve_skill_target

    res = resolve_skill_target("my-skill")
    assert res is not None
    assert res.skill_name == "my-skill"
    assert res.catalog_path is None
    assert res.manifest_path is None


def test_resolve_skill_target_raw_name_with_explicit_catalog(tmp_path: Path) -> None:
    """Verify resolve_skill_target preserves explicit catalog when given raw skill name."""
    from reach.catalog import resolve_skill_target

    res = resolve_skill_target("my-skill", explicit_catalog=tmp_path)
    assert res is not None
    assert res.skill_name == "my-skill"
    assert res.catalog_path == tmp_path.resolve()


def test_resolve_skill_target_typo_path_raises_file_not_found() -> None:
    """Verify resolve_skill_target raises FileNotFoundError when a path does not exist."""
    from reach.catalog import resolve_skill_target

    with pytest.raises(FileNotFoundError, match="skill path does not exist"):
        resolve_skill_target("./nonexistent/path/to/skill")

    with pytest.raises(FileNotFoundError, match="skill path does not exist"):
        resolve_skill_target("~/.agents/skills/missing-skill")


def test_resolve_skill_target_non_skill_file_raises_value_error(tmp_path: Path) -> None:
    """Verify resolve_skill_target raises ValueError when given a non-SKILL.md file."""
    from reach.catalog import resolve_skill_target

    text_file = tmp_path / "notes.txt"
    text_file.write_text("just some notes", encoding="utf-8")

    with pytest.raises(ValueError, match=r"expected a SKILL\.md file or skill directory"):
        resolve_skill_target(text_file)


def test_resolve_skill_target_invalid_frontmatter_raises_value_error(tmp_path: Path) -> None:
    """Verify resolve_skill_target raises ValueError when SKILL.md lacks valid frontmatter."""
    from reach.catalog import resolve_skill_target

    skill_dir = tmp_path / "corrupt-skill"
    skill_dir.mkdir()
    manifest = skill_dir / "SKILL.md"
    manifest.write_text("No frontmatter at all", encoding="utf-8")

    with pytest.raises(ValueError, match="does not contain valid YAML frontmatter"):
        resolve_skill_target(skill_dir)

    with pytest.raises(ValueError, match="does not contain valid YAML frontmatter"):
        resolve_skill_target(manifest)


def test_resolve_skill_target_skill_directory_path(tmp_path: Path) -> None:
    """Verify resolve_skill_target parses name from directory containing SKILL.md."""
    from reach.catalog import resolve_skill_target

    skill_dir = tmp_path / "dir-skill"
    skill_dir.mkdir()
    manifest = skill_dir / "SKILL.md"
    manifest.write_text(
        "---\nname: parsed-dir-skill\ndescription: A test skill.\n---\n# Body\n", encoding="utf-8"
    )

    res = resolve_skill_target(skill_dir)
    assert res is not None
    assert res.skill_name == "parsed-dir-skill"
    assert res.manifest_path == manifest.resolve()


def test_resolve_skill_target_direct_skill_md_path(tmp_path: Path) -> None:
    """Verify resolve_skill_target parses name directly from SKILL.md file path."""
    from reach.catalog import resolve_skill_target

    skill_dir = tmp_path / "direct-skill"
    skill_dir.mkdir()
    manifest = skill_dir / "SKILL.md"
    manifest.write_text(
        "---\nname: parsed-direct-skill\ndescription: Direct file test.\n---\n# Body\n",
        encoding="utf-8",
    )

    res = resolve_skill_target(manifest)
    assert res is not None
    assert res.skill_name == "parsed-direct-skill"
    assert res.manifest_path == manifest.resolve()


def test_resolve_skill_target_smart_parent_catalog(tmp_path: Path) -> None:
    """Verify resolve_skill_target infers parent catalog when parent has peer skills."""
    from reach.catalog import resolve_skill_target

    catalog_dir = tmp_path / "custom-catalog"
    catalog_dir.mkdir()
    skill1 = catalog_dir / "skill1"
    skill1.mkdir()
    (skill1 / "SKILL.md").write_text(
        "---\nname: skill-one\ndescription: First.\n---\n", encoding="utf-8"
    )
    skill2 = catalog_dir / "skill2"
    skill2.mkdir()
    (skill2 / "SKILL.md").write_text(
        "---\nname: skill-two\ndescription: Second.\n---\n", encoding="utf-8"
    )

    # Pass skill directory
    res1 = resolve_skill_target(skill1)
    assert res1 is not None
    assert res1.skill_name == "skill-one"
    assert res1.catalog_path == catalog_dir.resolve()

    # Pass manifest file
    res2 = resolve_skill_target(skill2 / "SKILL.md")
    assert res2 is not None
    assert res2.skill_name == "skill-two"
    assert res2.catalog_path == catalog_dir.resolve()


def test_resolve_skill_target_standalone_repo_bounds(tmp_path: Path) -> None:
    """Verify standalone skill directory does not escape to arbitrary parent directory."""
    from reach.catalog import resolve_skill_target

    # A standalone directory with no peer skills and not named 'skills'
    repo_dir = tmp_path / "my-standalone-project"
    repo_dir.mkdir()
    (repo_dir / "SKILL.md").write_text(
        "---\nname: standalone-tool\ndescription: Alone.\n---\n", encoding="utf-8"
    )

    res = resolve_skill_target(repo_dir)
    assert res is not None
    assert res.skill_name == "standalone-tool"
    # Should be repo_dir, NOT tmp_path
    assert res.catalog_path == repo_dir.resolve()


def test_resolve_skill_target_empty_dir_raises_value_error(tmp_path: Path) -> None:
    """Verify resolve_skill_target raises ValueError when directory has no SKILL.md or children."""
    from reach.catalog import resolve_skill_target

    empty_dir = tmp_path / "empty-dir"
    empty_dir.mkdir()

    with pytest.raises(ValueError, match=r"does not contain a SKILL\.md file"):
        resolve_skill_target(empty_dir)


def test_resolve_skill_target_contained_skills_raises_with_guidance(tmp_path: Path) -> None:
    """Verify resolve_skill_target raises ValueError with guidance when target contains skills."""
    from reach.catalog import resolve_skill_target

    catalog_dir = tmp_path / "skills-box"
    catalog_dir.mkdir()
    child1 = catalog_dir / "child1"
    child1.mkdir()
    (child1 / "SKILL.md").write_text(
        "---\nname: child-one\ndescription: C1.\n---\n", encoding="utf-8"
    )

    # Single child skill
    with pytest.raises(ValueError, match=r"(?s)containing 1 skill.*reach optimize"):
        resolve_skill_target(catalog_dir, command_name="optimize")

    # Multiple child skills
    child2 = catalog_dir / "child2"
    child2.mkdir()
    (child2 / "SKILL.md").write_text(
        "---\nname: child-two\ndescription: C2.\n---\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match=r"(?s)containing 2 skills.*reach eval"):
        resolve_skill_target(catalog_dir, command_name="eval")


def test_resolve_skill_target_raw_name_matches_cwd_dir_without_skill_md(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify raw skill name matching a cwd folder without SKILL.md does not trigger error."""
    from reach.catalog import resolve_skill_target

    monkeypatch.chdir(tmp_path)
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()

    # Plain name without explicit catalog should resolve as raw name, not crash on cwd docs/
    res = resolve_skill_target("docs")
    assert res is not None
    assert res.skill_name == "docs"
    assert res.catalog_path is None

    # Plain name with explicit catalog should resolve against explicit catalog
    other_cat = tmp_path / "my-skills"
    other_cat.mkdir()
    res_cat = resolve_skill_target("docs", explicit_catalog=other_cat)
    assert res_cat is not None
    assert res_cat.skill_name == "docs"
    assert res_cat.catalog_path == other_cat.resolve()

    # Explicit path './docs' should still raise ValueError because user explicitly asked for path
    with pytest.raises(ValueError, match=r"does not contain a SKILL\.md file"):
        resolve_skill_target("./docs")


def test_infer_parent_catalog_relative_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify _infer_parent_catalog handles relative 1-level paths without short-circuiting."""
    from pathlib import Path

    from reach.catalog import _infer_parent_catalog

    catalog = tmp_path / "custom-cat"
    catalog.mkdir()
    s1 = catalog / "s1"
    s1.mkdir()
    (s1 / "SKILL.md").write_text("---\nname: s1\ndescription: S1.\n---\n", encoding="utf-8")
    s2 = catalog / "s2"
    s2.mkdir()
    (s2 / "SKILL.md").write_text("---\nname: s2\ndescription: S2.\n---\n", encoding="utf-8")

    monkeypatch.chdir(catalog)
    # Pass 1-level relative path
    inferred = _infer_parent_catalog(Path("s1"))
    assert inferred == catalog.resolve()


def test_build_scaling_catalogs_strict_nested_subset(tmp_path: Path) -> None:
    """Verify single-skill scaling catalogs satisfy strict subset nestedness C_k1 subset C_k2."""
    skills = [
        Skill(
            name=f"skill-{i:02d}",
            description=f"Skill {i} specialized tool for domain {i % 4}",
            path=tmp_path / f"s{i}",
        )
        for i in range(25)
    ]
    import itertools

    catalogs = build_scaling_catalogs(skills, target_skill="skill-00", scales=(1, 5, 10, 15, 25))
    for earlier, later in itertools.pairwise(catalogs):
        assert set(earlier.skills).issubset(set(later.skills)), (
            f"Catalog {earlier.id} is not a subset of {later.id}"
        )
