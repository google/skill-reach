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
from typing import TYPE_CHECKING, Any, NamedTuple

from reach.catalog import resident_skills
from reach.config import resolve_path

if TYPE_CHECKING:
    from reach.models import Catalog, Skill


#: Prefix shared by all default workspace isolation subdirectories (e.g. .reach_claude).
ISOLATION_DIR_PREFIX = ".reach_"


def _reject_symlink(path: Path) -> None:
    """Raise ValueError if path or any of its .reach_* ancestor directories is a symbolic link."""
    if path.is_symlink():
        msg = f"Refusing to use symlinked isolation directory: {path}"
        raise ValueError(msg)
    for ancestor in path.parents:
        if ancestor.name.startswith(ISOLATION_DIR_PREFIX) and ancestor.is_symlink():
            msg = f"Refusing to use symlinked isolation directory: {ancestor}"
            raise ValueError(msg)


def ensure_private_directory(path: Path | str) -> Path:
    """Create directory with 0o700 permissions if supported on host platform.

    Args:
        path: Filesystem path to create as a private directory.

    Returns:
        Resolved Path object for the created directory.
    """
    # Check before resolve_path() because Path.resolve() follows symlinks, which
    # would obscure a symlink at the raw target path or its .reach_* parent.
    _reject_symlink(Path(path).expanduser())
    target = resolve_path(path)
    _reject_symlink(target)
    try:
        target.mkdir(mode=0o700, parents=True, exist_ok=True)
    except FileNotFoundError:
        # Retry once if a sibling worker pruned an empty parent directory concurrently
        # while pathlib was walking upward during mkdir(parents=True).
        target.mkdir(mode=0o700, parents=True, exist_ok=True)
    with contextlib.suppress(OSError):
        target.chmod(0o700)
        if target.parent.name.startswith(ISOLATION_DIR_PREFIX):
            target.parent.chmod(0o700)
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
    resolved_workdir = resolve_path(workdir)
    resolved_target = resolve_path(target_dir)
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


def resolve_thread_slot_dir(base_dir: Path | str, *, owner_slot_id: int | None = None) -> Path:
    """Return resolved base_dir on owner_slot_id or a per-thread slot directory under base_dir."""
    _reject_symlink(Path(base_dir).expanduser())
    resolved_base = resolve_path(base_dir)
    if owner_slot_id is not None and probe_slot_id() == owner_slot_id:
        return resolved_base
    slot = probe_slot_dir(resolved_base)
    _reject_symlink(slot)
    return slot


def cleanup_custom_isolation_dir(
    workdir: Path | str,
    custom_dir: Path | str,
    *,
    owner_slot_id: int | None = None,
    clean_slot: bool = False,
    clean_owner: bool = False,
) -> None:
    """Clean a user-supplied custom isolation directory for the calling thread.

    On a non-owner worker thread, remove the calling thread's ``slot_<pid>_<tid>``
    subdirectory when ``clean_slot`` is True and prune ``custom_dir`` if empty
    and contained within ``workdir``. On the owner thread, remove non-``slot_*``
    entries inside ``custom_dir`` when ``clean_owner`` is True and ``custom_dir``
    is contained within ``workdir``, then prune ``custom_dir`` if empty.
    """
    resolved_workdir = resolve_path(workdir)
    resolved_custom = resolve_path(custom_dir)
    slot_dir = resolve_thread_slot_dir(resolved_custom, owner_slot_id=owner_slot_id)
    if slot_dir != resolved_custom:
        if clean_slot:
            safe_cleanup_isolated_dir(resolved_custom, slot_dir)
        if clean_owner and resolved_workdir in resolved_custom.parents:
            with contextlib.suppress(OSError):
                resolved_custom.rmdir()
        return
    if (
        not clean_owner
        or resolved_workdir not in resolved_custom.parents
        or not resolved_custom.is_dir()
    ):
        return
    for entry in list(resolved_custom.iterdir()):
        if entry.name.startswith("slot_"):
            continue
        with contextlib.suppress(OSError):
            if entry.is_dir() and not entry.is_symlink():
                shutil.rmtree(entry, ignore_errors=True)
            else:
                entry.unlink(missing_ok=True)
    with contextlib.suppress(OSError):
        resolved_custom.rmdir()


class WorkspaceSlot(NamedTuple):
    """Represent a thread-isolated slot directory inside a workspace root directory."""

    workdir: Path
    root: Path
    slot: Path

    @classmethod
    def for_subdir(cls, workdir: Path | str, subdir_name: str) -> WorkspaceSlot:
        """Resolve a workspace slot for the calling thread under workdir/subdir_name."""
        resolved_workdir = resolve_path(workdir)
        raw_root = resolved_workdir / subdir_name
        _reject_symlink(raw_root)
        root = resolve_path(raw_root)
        return cls(workdir=resolved_workdir, root=root, slot=resolve_thread_slot_dir(root))

    def cleanup(self) -> bool:
        """Remove this thread's slot directory and prune the parent root when empty."""
        removed = safe_cleanup_isolated_dir(self.workdir, self.slot)
        with contextlib.suppress(OSError):
            self.root.rmdir()
        return removed


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
    manifest_only: bool = False,
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
        dst = (resolved_dest / name).resolve()
        if not dst.is_relative_to(resolved_dest) or dst == resolved_dest:
            msg = f"Skill destination path escapes destination directory: {name!r}"
            raise ValueError(msg)
        if manifest_only:
            manifest_candidate = resolved_src / "SKILL.md"
            if not manifest_candidate.is_file() and not manifest_candidate.is_symlink():
                manifest_candidate = resolved_src / "skill.md"
            if not manifest_candidate.is_file() and not manifest_candidate.is_symlink():
                msg = f"Skill manifest not found in {resolved_src}"
                raise ValueError(msg)

            canonical_src = resolved_src.resolve(strict=True)
            try:
                target = manifest_candidate.resolve(strict=True)
                if not target.is_relative_to(canonical_src) or (
                    target == manifest_candidate and manifest_candidate.is_symlink()
                ):
                    msg = (
                        f"Skill contains symlink escaping skill directory: "
                        f"{manifest_candidate} -> {target}"
                    )
                    raise ValueError(msg)
            except (OSError, RuntimeError) as exc:
                msg = f"Skill contains broken or cyclical symlink: {manifest_candidate}"
                raise ValueError(msg) from exc

            dst.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, dst / "SKILL.md", follow_symlinks=False)
            continue
        _validate_skill_symlinks(resolved_src)
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
    "Path",
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
        "Path",
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
