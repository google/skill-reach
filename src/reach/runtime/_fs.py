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

"""Encapsulate skill directory installation, symlink management, and path parsing."""

from __future__ import annotations

import contextlib
import os
import shutil
import threading
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from reach.catalog import resident_skills
from reach.config import resolve_path

if TYPE_CHECKING:
    from reach.models import Catalog, Skill


def ensure_private_directory(path: Path | str) -> Path:
    """Create directory with 0o700 permissions if supported on host platform.

    Args:
        path: Filesystem path to create as a private directory.

    Returns:
        Resolved Path object for the created directory.
    """
    target = resolve_path(path)
    target.mkdir(parents=True, exist_ok=True)
    with contextlib.suppress(OSError):
        target.chmod(0o700)
    return target


def validate_isolated_directory(value: Path | None, field_name: str = "directory") -> Path | None:
    """Validate that a directory path does not point to user's home or root directory."""
    if value is not None:
        resolved = resolve_path(value)
        if resolved == Path.home().resolve() or resolved == Path(resolved.root).resolve():
            msg = (
                f"{field_name} must not be the user's active home or root directory; "
                "point it at an isolated directory for agent workspace files"
            )
            raise ValueError(msg)
    return value


def safe_cleanup_isolated_dir(workdir: Path | str, target_dir: Path | str | None) -> bool:
    """Safely remove isolated target directory only if strictly contained within workdir.

    Args:
        workdir: Filesystem workspace root directory.
        target_dir: Subdirectory candidate to clean up, or None.

    Returns:
        True if the directory existed and was removed, False otherwise.
    """
    if target_dir is None:
        return False
    resolved_workdir = Path(workdir).resolve()
    resolved_target = Path(target_dir).resolve()
    if resolved_workdir in resolved_target.parents and resolved_target.exists():
        shutil.rmtree(resolved_target, ignore_errors=True)
        return True
    return False


def probe_slot_id() -> int:
    """Return an identifier unique to the calling OS thread."""
    return getattr(threading, "get_native_id", threading.get_ident)()


def probe_slot_dir(parent: Path, prefix: str = "slot") -> Path:
    """Return the scratch directory reserved for the calling process and thread under parent."""
    return Path(parent) / f"{prefix}_{os.getpid()}_{probe_slot_id()}"


def resolve_catalog_skills(
    catalog: Catalog,
    skills: Iterable[Skill],
) -> dict[str, Skill]:
    """Resolve and validate that all skills referenced by a catalog exist."""
    return {s.name: s for s in resident_skills(catalog, tuple(skills))}


def _validate_skill_symlinks(skill_dir: Path) -> None:
    """Ensure no symlink inside skill_dir resolves outside skill_dir."""
    resolved_root = skill_dir.resolve()
    for root_dir, dirnames, filenames in os.walk(resolved_root, followlinks=False):
        current = Path(root_dir)
        for item in (*dirnames, *filenames):
            p = current / item
            if p.is_symlink():
                try:
                    target = p.resolve(strict=True)
                    if (
                        not target.is_relative_to(resolved_root)
                        or target in p.parents
                        or target == p
                    ):
                        msg = f"Skill contains symlink escaping skill directory: {p} -> {target}"
                        raise ValueError(msg)
                except (OSError, RuntimeError) as exc:
                    msg = f"Skill contains broken or cyclical symlink: {p}"
                    raise ValueError(msg) from exc


def install_skills(
    catalog: Catalog,
    by_name: Mapping[str, Skill],
    destination: Path,
    *,
    use_symlinks: bool = False,
) -> tuple[str, ...]:
    """Copy or symlink skill directories for a catalog into destination workspace."""
    resolved_dest = destination.resolve()
    if resolved_dest.exists():
        shutil.rmtree(resolved_dest)
    resolved_dest.mkdir(parents=True, exist_ok=True)
    for name in catalog.skills:
        src = by_name[name].path
        resolved_src = src.resolve()
        if not resolved_src.is_dir():
            msg = f"Skill source directory not found: {src}"
            raise ValueError(msg)
        _validate_skill_symlinks(resolved_src)
        dst = (resolved_dest / name).resolve()
        if not dst.is_relative_to(resolved_dest) or dst == resolved_dest:
            msg = f"Skill destination path escapes destination directory: {name!r}"
            raise ValueError(msg)
        if use_symlinks:
            try:
                dst.symlink_to(resolved_src, target_is_directory=True)
                continue
            except OSError:
                pass
        shutil.copytree(resolved_src, dst, symlinks=True)
    return tuple(catalog.skills)


def _clean_path(val: object) -> Path | None:
    """Parse a string or Path into a cleaned Path, or return None if invalid."""
    if not val or not isinstance(val, (str, Path)):
        return None
    cleaned = val.strip() if isinstance(val, str) else val
    if not cleaned:
        return None
    try:
        return Path(cleaned)
    except (ValueError, TypeError, OSError):
        return None


TOOL_PATH_KEYS: tuple[str, ...] = (
    "path",
    "AbsolutePath",
    "DirectoryPath",
    "SearchDirectory",
    "SearchPath",
    "file",
    "file_path",
    "filePath",
    "dir_path",
    "directory_path",
    "directoryPath",
    "path_str",
)


def extract_tool_path(args: Mapping[str, Any]) -> str | None:
    """Extract the first non-empty filesystem path string from recognized tool arguments."""
    for key in TOOL_PATH_KEYS:
        val = args.get(key)
        if isinstance(val, (str, Path)):
            cleaned = str(val).strip()
            if cleaned:
                return cleaned
    return None


def resolve_skill_from_path(
    path_str: str | Path | None,
    resident: Iterable[str],
) -> str | None:
    """Extract matching resident skill name from file, skill directory, or nested reference path."""
    p = _clean_path(path_str)
    if p is None:
        return None

    resident_lookup = {r.lower(): r for r in resident}
    candidates: list[str] = []
    if p.name.lower() == "skill.md":
        candidates.append(p.parent.name.lower())

    lower_parts = [part.lower() for part in p.parts]
    for idx, part in enumerate(lower_parts[:-1]):
        if part == "skills":
            candidates.append(lower_parts[idx + 1])

    if p.suffix.lower() == ".md":
        candidates.append(p.stem.lower())
    elif not p.suffix and ((p / "SKILL.md").is_file() or (p / "skill.md").is_file()):
        candidates.append(p.name.lower())

    for candidate in candidates:
        if candidate in resident_lookup:
            return resident_lookup[candidate]
    return None


def normalize_skill_tool_args(
    args: Mapping[str, Any],
    skill: str | None = None,
) -> dict[str, Any] | None:
    """Rewrite skill directory path arguments in tool args to point at SKILL.md."""
    for key in (
        "AbsolutePath",
        "path",
        "file_path",
        "filePath",
        "dir_path",
        "directory_path",
        "directoryPath",
    ):
        p = _clean_path(args.get(key))
        if p is None or p.name.lower() == "skill.md" or p.suffix.lower() == ".md":
            continue
        if p.is_dir():
            for candidate_name in ("SKILL.md", "skill.md"):
                candidate_file = p / candidate_name
                if candidate_file.is_file():
                    return {key: str(candidate_file)}
        if skill is not None and not p.suffix and p.name.lower() == skill.lower():
            return {key: str(p / "SKILL.md")}
    return None
