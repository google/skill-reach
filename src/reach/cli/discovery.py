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

"""Resolve skill corpora, runtime discovery, and study directory layouts."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from reach.config import RunConfig
from reach.discovery import Discovery, resolve_corpus
from reach.views import Console, print_discovery

if TYPE_CHECKING:
    from collections.abc import Sequence

    from reach.models import Skill
    from reach.runtime import AgentRuntime

    from .flags import CatalogFlags, RecordFlags, StudyFlags


def _no_skills(corpus: Path | None, global_scope: bool = False) -> ValueError:
    """Return a descriptive ValueError when no skill definitions could be found."""
    if corpus is None:
        if global_scope:
            return ValueError(
                "no skills found: no skills installed in user global locations "
                "(~/.agents/skills, ~/.claude/skills, etc.)",
            )
        return ValueError(
            "no skills found: pass --skills, or install some where the runtime reads them",
        )
    return ValueError(
        f"no skills found under {corpus}: it holds no SKILL.md at any depth, "
        "so there is nothing here to rank or probe",
    )


def _corpus(
    console: Console,
    driver: AgentRuntime,
    settings: RunConfig,
    global_scope: bool = False,
    agent: str | None = None,
) -> tuple[list[Skill], Sequence[Path], Discovery | None]:
    """Resolve skill corpus from configuration, runtime environment discovery, or Agent Registry."""
    if settings.registry.registry or settings.registry.project is not None:
        from reach.catalog import load_registry_skills
        from reach.config import RegistrySettings
        from reach.registry import RegistryCacheManager

        eff_registry = RunConfig.resolve(RegistrySettings, settings)
        if eff_registry.project:
            cache_mgr = RegistryCacheManager()
            reg_skills = load_registry_skills(
                project=eff_registry.project,
                location=eff_registry.location,
                publisher=eff_registry.publisher,
                fresh=eff_registry.fresh,
                no_cache=eff_registry.no_cache,
                cache_ttl_seconds=eff_registry.cache_ttl_seconds,
            )
            reg_roots: Sequence[Path] = [
                cache_mgr.location_dir(eff_registry.project, eff_registry.location)
            ]
            return reg_skills, reg_roots, None

    skills, roots, discovered = resolve_corpus(
        driver,
        Path.cwd(),
        settings.study.skills,
        global_scope=global_scope,
        agent=agent,
    )
    if discovered is not None:
        print_discovery(console, discovered)
    if not skills:
        raise _no_skills(settings.study.skills, global_scope=global_scope)
    return skills, roots, discovered


def _asks_for_a_mode(config: Path | None, catalog: CatalogFlags | None) -> bool:
    """Return True if a catalog mode was specified via CLI flags or config file."""
    if catalog is not None and catalog.mode is not None:
        return True
    return config is not None and RunConfig.declared(config, "catalog", "mode")


#: Canonical directory name for Reach study artifacts and benchmark queries.
REACH_DIR_NAME = ".reach"

#: Default benchmark query set filename.
QUERIES_FILENAME = "queries.json"

#: Default project-local path to the benchmark query set.
DEFAULT_QUERIES_PATH = Path(REACH_DIR_NAME) / QUERIES_FILENAME

#: Supported benchmark query set filenames checked during auto-discovery.
QUERY_FILENAMES: tuple[str, ...] = (QUERIES_FILENAME, "queries.jsonl", "queries.csv")


def find_existing_queries_path(
    skills_path: Path | None = None,
    *,
    prefer_local: bool = True,
) -> Path | None:
    """Discover an existing .reach/queries.{json,jsonl,csv} file from CWD or skills corpus roots."""
    local_candidates = [Path(REACH_DIR_NAME) / name for name in QUERY_FILENAMES]
    corpus_candidates: list[Path] = []
    if skills_path is not None:
        sp = Path(skills_path).resolve()
        search_roots = [sp] if sp.is_dir() else [sp.parent]
        if sp.parent != sp and sp.parent not in search_roots:
            search_roots.append(sp.parent)
        for root in search_roots:
            corpus_candidates.extend(root / REACH_DIR_NAME / name for name in QUERY_FILENAMES)
    candidates = (
        local_candidates + corpus_candidates
        if prefer_local
        else corpus_candidates + local_candidates
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _run_dir_study(run_dir: Path, study: StudyFlags) -> StudyFlags:
    """Populate default queries and workspace paths relative to a study root."""
    return study.model_copy(
        update={
            "queries": (study.queries if study.queries is not None else run_dir / QUERIES_FILENAME),
            "workdir": (study.workdir if study.workdir is not None else run_dir / "workspace"),
        },
    )


def _run_dir_defaults(
    run_dir: Path,
    *,
    study: StudyFlags,
    record: RecordFlags,
) -> tuple[StudyFlags, RecordFlags]:
    """Configure default directory layout for queries, workspace, and results."""
    return (
        _run_dir_study(run_dir, study),
        record.model_copy(
            update={
                "records": (
                    record.records if record.records is not None else run_dir / "results.jsonl"
                ),
            },
        ),
    )
