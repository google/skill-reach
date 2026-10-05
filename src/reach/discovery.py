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

"""Discover and load skills from configured root paths and resolve duplicates."""

from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from reach.artifact import ContestedSkill
from reach.catalog import load_skills
from reach.models import Skill
from reach.runtime import FAKE_AGENT, AgentRuntime, SkillRoot


class Shadowed(BaseModel):
    """Record a skill shadowed by a higher-precedence copy in another root."""

    model_config = ConfigDict(frozen=True)

    name: str
    kept: Path
    hidden: tuple[Path, ...]


class Ambiguity(BaseModel):
    """Record a duplicate skill name occurring across roots with equal precedence."""

    model_config = ConfigDict(frozen=True)

    name: str
    paths: tuple[Path, ...]


class Discovery(BaseModel):
    """Summarize discovered skills, candidate roots, and conflict resolutions."""

    model_config = ConfigDict(frozen=True)

    roots: tuple[SkillRoot, ...] = ()
    skills: tuple[Skill, ...] = ()
    shadowed: tuple[Shadowed, ...] = ()
    ambiguous: tuple[Ambiguity, ...] = ()
    empty_roots: tuple[Path, ...] = ()

    @property
    def paths(self) -> tuple[Path, ...]:
        """Return the sequence of root filesystem paths."""
        return tuple(root.path for root in self.roots)

    @property
    def contested(self) -> tuple[ContestedSkill, ...]:
        """Return all shadowed and ambiguous skills formatted for artifact storage."""
        return tuple(
            ContestedSkill(
                name=pair.name,
                kept=pair.paths[0],
                dropped=pair.paths[1:],
                ranked=False,
            )
            for pair in self.ambiguous
        ) + tuple(
            ContestedSkill(name=hidden.name, kept=hidden.kept, dropped=hidden.hidden)
            for hidden in self.shadowed
        )

    @property
    def warnings(self) -> tuple[str, ...]:
        """Generate warning messages for ambiguities, shadowing, and empty roots."""
        lines = []
        for pair in self.ambiguous:
            kept = pair.paths[0]
            ignored = ", ".join(str(path) for path in pair.paths[1:])
            lines.append(
                f"duplicate skill '{pair.name}' found with equal precedence; "
                f"using {kept} (ignoring {ignored})",
            )
        lines.extend(
            f"{hidden.name} loaded from {hidden.kept}, shadowing "
            + ", ".join(str(path) for path in hidden.hidden)
            for hidden in self.shadowed
        )
        lines.extend(
            f"{root} is a skill root the runtime reads but it is empty" for root in self.empty_roots
        )
        return tuple(lines)


def discover(runtime: AgentRuntime, workdir: Path) -> Discovery:
    """Discover skill definitions available to a runtime from a directory."""
    roots = runtime.skill_roots(workdir)
    by_name: dict[str, Skill] = {}
    from_root: dict[str, SkillRoot] = {}
    shadowed: dict[str, list[Path]] = {}
    ambiguous: dict[str, list[Path]] = {}
    empty: list[Path] = []

    for root in sorted(roots, key=lambda r: r.precedence):
        loaded = load_skills(root.path)
        if not loaded:
            empty.append(root.path)
        for skill in loaded:
            held = by_name.get(skill.name)
            if held is None:
                by_name[skill.name] = skill
                from_root[skill.name] = root
                continue
            if held.path.resolve() == skill.path.resolve():
                continue
            if from_root[skill.name].precedence == root.precedence:
                ambiguous.setdefault(skill.name, [held.path]).append(skill.path)
            else:
                shadowed.setdefault(skill.name, []).append(skill.path)

    found: list[Skill] = sorted(by_name.values(), key=lambda skill: skill.name)
    return Discovery(
        roots=roots,
        skills=tuple(found),
        shadowed=tuple(
            Shadowed(name=name, kept=by_name[name].path, hidden=tuple(paths))
            for name, paths in sorted(shadowed.items())
        ),
        ambiguous=tuple(
            Ambiguity(name=name, paths=tuple(paths)) for name, paths in sorted(ambiguous.items())
        ),
        empty_roots=tuple(empty),
    )


def _resolve_named_corpus(
    corpus: Path | str,
) -> tuple[list[Skill], Sequence[Path], Discovery | None]:
    """Resolve skills from an explicit directory path and record duplicate warnings."""
    from reach.catalog import _skill_files, parse_frontmatter
    from reach.config import resolve_path

    resolved_corpus = resolve_path(corpus)
    if not resolved_corpus.is_dir():
        msg = f"skill root does not exist: {resolved_corpus}"
        raise NotADirectoryError(msg)

    by_name: dict[str, Skill] = {}
    duplicates: dict[str, list[Path]] = {}
    candidate_files = sorted(
        _skill_files(resolved_corpus),
        key=lambda p: (len(p.parts), str(p)),
    )
    for skill_file in candidate_files:
        skill = parse_frontmatter(skill_file.read_text(encoding="utf-8"), skill_file)
        if skill is not None:
            if skill.name in by_name:
                duplicates.setdefault(skill.name, [by_name[skill.name].path]).append(skill.path)
            else:
                by_name[skill.name] = skill

    unique_skills = sorted(by_name.values(), key=lambda s: s.name)
    if duplicates:
        discovery = Discovery(
            roots=(SkillRoot(path=resolved_corpus, scope="corpus", precedence=0),),
            skills=tuple(unique_skills),
            ambiguous=tuple(
                Ambiguity(name=name, paths=tuple(paths))
                for name, paths in sorted(duplicates.items())
            ),
        )
        return unique_skills, (resolved_corpus,), discovery
    return unique_skills, (resolved_corpus,), None


def resolve_corpus(
    runtime: AgentRuntime,
    workdir: Path,
    corpus: Path | None,
    config_path: Path | str | None = None,
    agent: str | None = None,
    global_scope: bool = False,
) -> tuple[list[Skill], Sequence[Path], Discovery | None]:
    """Resolve skills and roots from explicit corpus path or discovery precedence."""
    if corpus is not None:
        return _resolve_named_corpus(corpus)

    # Test doubles should be respected directly
    if runtime.name == FAKE_AGENT:
        found = discover(runtime, workdir)

        return list(found.skills), found.paths, found

    from reach.catalog import parse_frontmatter
    from reach.config import resolve_discovery_candidates

    resolved_workdir = Path.home() if global_scope else workdir.expanduser().resolve()
    candidates = resolve_discovery_candidates(
        workdir=resolved_workdir,
        agent=agent,
        config_path=config_path,
        global_scope=global_scope,
    )
    scope_name = "user" if global_scope else "project"

    for candidate in candidates:
        if candidate == resolved_workdir:
            manifest = candidate / "SKILL.md"
            if manifest.is_file():
                skill = parse_frontmatter(manifest.read_text(encoding="utf-8"), manifest)
                if skill is not None:
                    discovery = Discovery(
                        roots=(SkillRoot(path=candidate, scope=scope_name, precedence=0),),
                        skills=(skill,),
                    )
                    return [skill], (candidate,), discovery
        elif candidate.is_dir():
            skills = load_skills(candidate)
            if skills:
                discovery = Discovery(
                    roots=(SkillRoot(path=candidate, scope=scope_name, precedence=0),),
                    skills=tuple(skills),
                )
                return skills, (candidate,), discovery

    if global_scope:
        return [], (), Discovery()

    found = discover(runtime, workdir)
    return list(found.skills), found.paths, found
