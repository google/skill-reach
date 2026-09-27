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

"""Verify runtime skill discovery, precedence resolution, and corpus binding."""

from __future__ import annotations

from pathlib import Path

import pytest

from reach.discovery import discover, resolve_corpus
from reach.runtime import SkillRoot
from reach.runtime.fake import FakeRuntime


def test_discovery_asks_the_runtime_rather_than_walking_the_tree_itself(
    make_skill_root,
) -> None:
    """Verify discovery queries runtime agent roots instead of manual filesystem traversal."""
    root = make_skill_root("personal", {"gcs-lifecycle-rules": "Lifecycle rules."})
    runtime = FakeRuntime(roots=[SkillRoot(path=root, scope="user")])

    found = discover(runtime, Path("/nowhere"))

    assert [s.name for s in found.skills] == ["gcs-lifecycle-rules"]
    assert found.paths == (root,)


def test_discovery_is_asked_about_the_directory_it_was_given(make_skill_root) -> None:
    """Verify discovery passes given working directory to runtime root lookup."""
    root = make_skill_root("personal", {"gke-basics": "Cluster fundamentals."})
    runtime = FakeRuntime(roots=[SkillRoot(path=root, scope="user")])

    discover(runtime, Path("/somewhere/else"))

    assert runtime.root_queries == [Path("/somewhere/else")]


def test_two_roots_make_one_corpus(make_skill_root) -> None:
    """Verify discovery unions skills across multiple distinct root paths."""
    personal = make_skill_root("personal", {"gcs-lifecycle-rules": "Lifecycle."})
    project = make_skill_root("project", {"gke-basics": "Fundamentals."})
    runtime = FakeRuntime(
        roots=[
            SkillRoot(path=personal, scope="user", precedence=0),
            SkillRoot(path=project, scope="project", precedence=1),
        ],
    )

    found = discover(runtime, Path.cwd())

    assert [s.name for s in found.skills] == ["gcs-lifecycle-rules", "gke-basics"]
    assert not found.shadowed
    assert not found.ambiguous


def test_the_better_ranked_root_wins_and_the_loss_is_recorded(make_skill_root) -> None:
    """Verify higher precedence root shadows lower precedence duplicate skill."""
    personal = make_skill_root("personal", {"gke-basics": "The personal copy."})
    project = make_skill_root("project", {"gke-basics": "The project copy."})
    runtime = FakeRuntime(
        roots=[
            SkillRoot(path=personal, scope="user", precedence=0),
            SkillRoot(path=project, scope="project", precedence=1),
        ],
    )

    found = discover(runtime, Path.cwd())

    (skill,) = found.skills
    assert skill.description == "The personal copy."
    (hidden,) = found.shadowed
    assert hidden.name == "gke-basics"
    assert hidden.kept == personal / "gke-basics"
    assert hidden.hidden == (project / "gke-basics",)
    assert any("shadowing" in warning for warning in found.warnings)


def test_roots_the_runtime_did_not_rank_are_reported_as_unresolved(
    make_skill_root,
) -> None:
    """Verify duplicate skills with equal precedence are recorded as ambiguous."""
    first = make_skill_root("one", {"gke-basics": "The first copy."})
    second = make_skill_root("two", {"gke-basics": "The second copy."})
    runtime = FakeRuntime(
        roots=[
            SkillRoot(path=first, scope="project", precedence=0),
            SkillRoot(path=second, scope="project", precedence=0),
        ],
    )

    found = discover(runtime, Path.cwd())

    assert not found.shadowed
    (pair,) = found.ambiguous
    assert pair.name == "gke-basics"
    assert pair.paths == (first / "gke-basics", second / "gke-basics")
    assert any("does not rank" in warning for warning in found.warnings)


def test_one_skill_reached_through_nested_roots_is_not_a_duplicate(
    make_skill_root,
) -> None:
    """Verify identical directory paths across roots do not generate duplicate warnings."""
    outer = make_skill_root("outer", {"gke-basics": "Fundamentals."})
    runtime = FakeRuntime(
        roots=[
            SkillRoot(path=outer, scope="project", precedence=0),
            SkillRoot(path=outer, scope="project", precedence=1),
        ],
    )

    found = discover(runtime, Path.cwd())

    assert len(found.skills) == 1
    assert not found.shadowed
    assert not found.ambiguous


def test_one_skill_linked_into_a_second_root_is_not_a_duplicate(
    make_skill_root,
    tmp_path: Path,
) -> None:
    """Verify symlinked identical directories do not trigger duplication warnings."""
    checkout = make_skill_root("checkout", {"gke-basics": "Fundamentals."})
    linked = tmp_path / "linked"
    linked.mkdir()
    (linked / "gke-basics").symlink_to(
        checkout / "gke-basics",
        target_is_directory=True,
    )
    runtime = FakeRuntime(
        roots=[
            SkillRoot(path=checkout, scope="project", precedence=0),
            SkillRoot(path=linked, scope="user", precedence=1),
        ],
    )

    found = discover(runtime, Path.cwd())

    assert len(found.skills) == 1
    assert not found.shadowed
    assert not found.ambiguous


def test_a_runtime_that_reads_nowhere_discovers_nothing(make_runtime) -> None:
    """Verify runtime with no configured roots returns empty discovery."""
    found = discover(make_runtime(), Path.cwd())

    assert found.skills == ()
    assert found.paths == ()
    assert found.warnings == ()


def test_an_empty_root_is_worth_saying_out_loud(tmp_path: Path) -> None:
    """Verify empty discovered roots are recorded in empty_roots and warnings."""
    empty = tmp_path / "empty"
    empty.mkdir()
    runtime = FakeRuntime(roots=[SkillRoot(path=empty, scope="user")])

    found = discover(runtime, Path.cwd())

    assert found.empty_roots == (empty,)
    assert any("empty" in warning for warning in found.warnings)


def test_a_root_that_is_not_there_is_refused_rather_than_skipped(
    tmp_path: Path,
) -> None:
    """Verify discover raises NotADirectoryError when a declared root path does not exist."""
    runtime = FakeRuntime(roots=[SkillRoot(path=tmp_path / "gone", scope="user")])

    with pytest.raises(NotADirectoryError):
        discover(runtime, Path.cwd())


def test_a_named_corpus_is_taken_as_given_and_the_runtime_is_not_asked(
    skill_repo: Path,
    make_runtime,
) -> None:
    """Verify explicitly named corpus path bypasses runtime discovery."""
    runtime = make_runtime()

    skills, roots, found = resolve_corpus(runtime, Path.cwd(), skill_repo)

    assert len(skills) == 3
    assert roots == (skill_repo,)
    assert found is None
    assert runtime.root_queries == []


def test_resolve_corpus_warns_on_duplicate_skills_in_named_corpus(
    tmp_path: Path,
    make_runtime,
) -> None:
    """Verify resolve_corpus reports duplicate skills in a named corpus as ambiguous warnings."""
    corpus = tmp_path / "corpus"
    s1 = corpus / "dir1" / "skill-a"
    s1.mkdir(parents=True)
    (s1 / "SKILL.md").write_text(
        "---\nname: skill-a\ndescription: Primary copy\n---\n# Body\n", encoding="utf-8"
    )
    s2 = corpus / "dir2" / "skill-a"
    s2.mkdir(parents=True)
    (s2 / "SKILL.md").write_text(
        "---\nname: skill-a\ndescription: Duplicate copy\n---\n# Body\n", encoding="utf-8"
    )

    runtime = make_runtime()
    skills, _roots, found = resolve_corpus(runtime, Path.cwd(), corpus)

    assert len(skills) == 1
    assert skills[0].name == "skill-a"
    assert found is not None
    assert len(found.ambiguous) == 1
    assert found.ambiguous[0].name == "skill-a"
    assert any("skill-a" in w for w in found.warnings)


def test_no_named_corpus_falls_through_to_the_runtime(make_skill_root) -> None:
    """Verify resolve_corpus falls back to runtime discovery when corpus is None."""
    root = make_skill_root("personal", {"gke-basics": "Fundamentals."})
    runtime = FakeRuntime(roots=[SkillRoot(path=root, scope="user")])

    skills, roots, found = resolve_corpus(runtime, Path.cwd(), None)

    assert [s.name for s in skills] == ["gke-basics"]
    assert tuple(roots) == (root,)
    assert found is not None
    assert found.roots[0].scope == "user"


def test_three_tier_precedence_shadowing(make_skill_root) -> None:
    """Verify three-tier root precedence resolves correctly with enterprise winning."""
    enterprise = make_skill_root("ent", {"shared-skill": "Enterprise version."})
    project = make_skill_root("proj", {"shared-skill": "Project version."})
    user = make_skill_root("usr", {"shared-skill": "User version."})
    runtime = FakeRuntime(
        roots=[
            SkillRoot(path=enterprise, scope="enterprise", precedence=0),
            SkillRoot(path=project, scope="project", precedence=1),
            SkillRoot(path=user, scope="user", precedence=2),
        ],
    )

    found = discover(runtime, Path.cwd())

    (skill,) = found.skills
    assert skill.description == "Enterprise version."
    (hidden,) = found.shadowed
    assert hidden.name == "shared-skill"
    assert hidden.kept == enterprise / "shared-skill"
    assert set(hidden.hidden) == {project / "shared-skill", user / "shared-skill"}


def test_resolve_corpus_discovers_single_skill_at_root(tmp_path: Path) -> None:
    """Verify resolve_corpus detects when working directory is a single skill root."""
    from reach.runtime.keyword import KeywordRuntime

    skill_file = tmp_path / "SKILL.md"
    skill_file.write_text(
        "---\nname: root-skill\ndescription: Single skill at project root.\n---\nBody",
        encoding="utf-8",
    )
    runtime = KeywordRuntime()
    skills, roots, found = resolve_corpus(runtime, tmp_path, None)

    assert len(skills) == 1
    assert skills[0].name == "root-skill"
    assert roots == (tmp_path,)
    assert found is not None


def test_resolve_corpus_discovers_skills_directory(tmp_path: Path) -> None:
    """Verify resolve_corpus prefers generic ./skills/ directory over agent-specific folders."""
    from reach.runtime.keyword import KeywordRuntime

    skills_dir = tmp_path / "skills" / "my-skill"
    skills_dir.mkdir(parents=True)
    (skills_dir / "SKILL.md").write_text(
        "---\nname: my-skill\ndescription: A project skill.\n---\nBody",
        encoding="utf-8",
    )
    # Also create .agents/skills to verify skills/ takes precedence
    agent_dir = tmp_path / ".agents" / "skills" / "agent-skill"
    agent_dir.mkdir(parents=True)
    (agent_dir / "SKILL.md").write_text(
        "---\nname: agent-skill\ndescription: An agent skill.\n---\nBody",
        encoding="utf-8",
    )

    runtime = KeywordRuntime()
    skills, roots, _found = resolve_corpus(runtime, tmp_path, None)

    assert len(skills) == 1
    assert skills[0].name == "my-skill"
    assert roots == (tmp_path / "skills",)


def test_resolve_corpus_prefers_agent_directory_when_agent_specified(tmp_path: Path) -> None:
    """Verify resolve_corpus elevates agent's native directory when agent is specified."""
    from reach.runtime.claude_code import ClaudeCodeRuntime

    agents_dir = tmp_path / ".agents" / "skills" / "agent-skill"
    agents_dir.mkdir(parents=True)
    (agents_dir / "SKILL.md").write_text(
        "---\nname: agent-skill\ndescription: An agent skill.\n---\nBody",
        encoding="utf-8",
    )
    claude_dir = tmp_path / ".claude" / "skills" / "claude-skill"
    claude_dir.mkdir(parents=True)
    (claude_dir / "SKILL.md").write_text(
        "---\nname: claude-skill\ndescription: A Claude skill.\n---\nBody",
        encoding="utf-8",
    )

    runtime = ClaudeCodeRuntime()
    skills, roots, _found = resolve_corpus(runtime, tmp_path, None, agent="claude-code")

    assert len(skills) == 1
    assert skills[0].name == "claude-skill"
    assert roots == (tmp_path / ".claude" / "skills",)


def test_resolve_corpus_falls_through_empty_skills_folder(tmp_path: Path) -> None:
    """Verify resolve_corpus skips empty skills/ folder and falls through to next candidate."""
    from reach.runtime.keyword import KeywordRuntime

    (tmp_path / "skills").mkdir(parents=True)
    agents_dir = tmp_path / ".agents" / "skills" / "agent-skill"
    agents_dir.mkdir(parents=True)
    (agents_dir / "SKILL.md").write_text(
        "---\nname: agent-skill\ndescription: An agent skill.\n---\nBody",
        encoding="utf-8",
    )

    runtime = KeywordRuntime()
    skills, roots, _found = resolve_corpus(runtime, tmp_path, None)

    assert len(skills) == 1
    assert skills[0].name == "agent-skill"
    assert roots == (tmp_path / ".agents" / "skills",)


@pytest.mark.parametrize(
    ("client_subdir", "skill_name"),
    [
        (".cursor/skills", "cursor-skill"),
        (".github/skills", "copilot-skill"),
    ],
)
def test_resolve_corpus_discovers_fallback_client_skills_when_earlier_candidates_absent(
    tmp_path: Path,
    client_subdir: str,
    skill_name: str,
) -> None:
    """Verify resolve_corpus discovers fallback client directories when earlier ones are absent."""
    from reach.runtime.keyword import KeywordRuntime

    skill_dir = tmp_path / client_subdir / skill_name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {skill_name}\ndescription: A fallback client skill.\n---\nBody",
        encoding="utf-8",
    )

    runtime = KeywordRuntime()
    skills, roots, _found = resolve_corpus(runtime, tmp_path, None)

    assert len(skills) == 1
    assert skills[0].name == skill_name
    assert roots == (tmp_path / client_subdir,)


def test_resolve_corpus_discovers_global_skills_when_global_scope_set(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify resolve_corpus discovers ~/.agents/skills when global_scope=True."""
    from reach.runtime.keyword import KeywordRuntime

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    global_dir = tmp_path / ".agents" / "skills" / "global-skill"
    global_dir.mkdir(parents=True)
    (global_dir / "SKILL.md").write_text(
        "---\nname: global-skill\ndescription: A user global skill.\n---\nBody",
        encoding="utf-8",
    )

    runtime = KeywordRuntime()
    fake_workdir = tmp_path / "some-empty-project"
    fake_workdir.mkdir()
    skills, roots, found = resolve_corpus(
        runtime,
        fake_workdir,
        None,
        global_scope=True,
    )

    assert len(skills) == 1
    assert skills[0].name == "global-skill"
    assert roots == (tmp_path / ".agents" / "skills",)
    assert found is not None
    assert found.roots[0].scope == "user"


def test_resolve_corpus_global_elevates_agent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify resolve_corpus in global scope elevates specified agent directory."""
    from reach.runtime.keyword import KeywordRuntime

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    agy_dir = tmp_path / ".agents" / "skills" / "agy-skill"
    agy_dir.mkdir(parents=True)
    (agy_dir / "SKILL.md").write_text(
        "---\nname: agy-skill\ndescription: An Antigravity skill.\n---\nBody",
        encoding="utf-8",
    )

    claude_dir = tmp_path / ".claude" / "skills" / "claude-skill"
    claude_dir.mkdir(parents=True)
    (claude_dir / "SKILL.md").write_text(
        "---\nname: claude-skill\ndescription: A Claude skill.\n---\nBody",
        encoding="utf-8",
    )

    runtime = KeywordRuntime()
    fake_workdir = tmp_path / "some-empty-project"
    fake_workdir.mkdir()

    # When agent="claude-code", ~/.claude/skills is elevated over ~/.agents/skills
    skills, roots, _ = resolve_corpus(
        runtime,
        fake_workdir,
        None,
        agent="claude-code",
        global_scope=True,
    )
    assert len(skills) == 1
    assert skills[0].name == "claude-skill"
    assert roots == (tmp_path / ".claude" / "skills",)

    # When agent="antigravity-cli", ~/.agents/skills is elevated
    skills, roots, _ = resolve_corpus(
        runtime,
        fake_workdir,
        None,
        agent="antigravity-cli",
        global_scope=True,
    )
    assert len(skills) == 1
    assert skills[0].name == "agy-skill"
    assert roots == (tmp_path / ".agents" / "skills",)


def test_cli_corpus_registry_discovery_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify _corpus with registry settings resolves single registry path segment."""
    from unittest.mock import patch

    from reach.cli.discovery import _corpus
    from reach.models import Skill

    monkeypatch.chdir(tmp_path)
    mock_skills = [
        Skill(
            name="cloud-logging",
            description="Query logs",
            path=tmp_path / "cache" / "cloud-logging",
        ),
    ]

    from reach.config import RegistrySettings, RunConfig
    from reach.runtime.keyword import KeywordRuntime
    from reach.views import build_console

    with patch("reach.catalog.load_registry_skills", return_value=mock_skills):
        corpus_skills, roots, _ = _corpus(
            console=build_console(quiet=True),
            driver=KeywordRuntime(),
            settings=RunConfig(
                registry=RegistrySettings(project="test-proj", location="global"),
            ),
        )
        assert len(corpus_skills) == 1
        assert len(roots) == 1
        root_str = str(roots[0])
        assert "/registry/registry/" not in root_str
        assert root_str.endswith("test-proj/global")
