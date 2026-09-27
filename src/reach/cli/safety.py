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

"""Provide safety confirmation prompts, sandbox advice, and batch bypass evaluation."""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from reach.views import safety_panel

if TYPE_CHECKING:
    from reach.models import Skill
    from reach.runtime import AgentRuntime
    from reach.views import Console

__all__ = [
    "catalog_summary",
    "confirm_skill_execution",
    "runtime_summary",
]


def _format_path(
    path: Path,
    *,
    resolved_path: Path | None = None,
    resolved_cwd: Path | None = None,
    resolved_home: Path | None = None,
) -> str:
    """Format filesystem path relative to working directory or user home."""
    target = resolved_path or path.resolve()
    cwd = resolved_cwd or Path.cwd().resolve()
    try:
        rel_cwd = target.relative_to(cwd)
        return f"./{rel_cwd}"
    except ValueError:
        pass
    home = resolved_home or Path.home().resolve()
    try:
        rel_home = target.relative_to(home)
        return f"~/{rel_home}"
    except ValueError:
        return str(path)


def catalog_summary(
    skills: Sequence[Skill] | Sequence[Path] | int,
    roots: Sequence[Path] | None = None,
) -> tuple[str, list[str]]:
    """Format total skill count and directory origin breakdown."""
    if isinstance(skills, int):
        total = skills
        skill_paths: list[Path] = []
    else:
        total = len(skills)
        skill_paths = [s if isinstance(s, Path) else s.path for s in skills]

    resolved_roots = list(roots) if roots else []
    if not resolved_roots and not skill_paths:
        label = "skill" if total == 1 else "skills"
        return f"{total} {label}", []

    resolved_cwd = Path.cwd().resolve()
    resolved_home = Path.home().resolve()

    if len(resolved_roots) == 1:
        fmt = _format_path(
            resolved_roots[0],
            resolved_cwd=resolved_cwd,
            resolved_home=resolved_home,
        )
        label = "skill" if total == 1 else "skills"
        return f"{total} {label} in {fmt}", []

    if len(resolved_roots) > 1:
        bullets: list[str] = []
        label = "skill" if total == 1 else "skills"
        loc_label = "location" if len(resolved_roots) == 1 else "locations"
        headline = f"{total} {label} resolved from {len(resolved_roots)} {loc_label}:"
        resolved_skill_parents = [set(p.resolve().parents) for p in skill_paths]
        for root in resolved_roots:
            resolved_root = root.resolve()
            matching = sum(1 for parents in resolved_skill_parents if resolved_root in parents)
            count = matching if skill_paths else max(1, total // len(resolved_roots))
            count_label = "skill" if count == 1 else "skills"
            fmt_root = _format_path(
                root,
                resolved_path=resolved_root,
                resolved_cwd=resolved_cwd,
                resolved_home=resolved_home,
            )
            bullets.append(f"{fmt_root} ({count} {count_label})")
        return headline, bullets

    label = "skill" if total == 1 else "skills"
    return f"{total} {label}", []


def runtime_summary(runtime: AgentRuntime | str) -> str:
    """Format runtime security and sandboxing characteristics."""
    name = runtime if isinstance(runtime, str) else getattr(runtime, "name", str(runtime))
    if name == "pi":
        return "pi (No built-in sandbox)"
    if name == "antigravity-cli":
        return "antigravity-cli (--dangerously-skip-permissions)"
    if name == "claude-code":
        return "claude-code (tool-restricted session)"
    if name == "goose":
        return "goose (tool extensions enabled)"
    return name


def _can_bypass(
    *,
    dry_run: bool,
    runtime_name: str,
    yes: bool,
    trusted: bool,
) -> bool:
    """Evaluate whether confirmation can be bypassed based on runtime or configuration."""
    if dry_run or runtime_name in ("keyword", "fake"):
        return True
    if yes or trusted:
        return True
    return os.environ.get("REACH_YES", "").lower() in ("1", "true", "yes") or os.environ.get(
        "REACH_FORCE", ""
    ).lower() in ("1", "true", "yes")


def confirm_skill_execution(
    console: Console,
    *,
    runtime_name: str,
    skills: Sequence[Skill] | Sequence[Path] | int,
    roots: Sequence[Path] | None = None,
    action: str = "execution",
    yes: bool = False,
    trusted: bool = False,
    dry_run: bool = False,
) -> int:
    """Prompt for user confirmation before executing agent probes on resident skills.

    Returns:
        0 if approved or bypassed, 1 if declined by user, 2 if unconfirmed in non-TTY.
    """
    if _can_bypass(dry_run=dry_run, runtime_name=runtime_name, yes=yes, trusted=trusted):
        return 0

    # Fail-closed in non-interactive (non-TTY) environments
    if not (sys.stdin.isatty() and (sys.stderr.isatty() or sys.stdout.isatty())):
        sys.stderr.write(
            f"Error: Confirmation required to probe skills with runtime '{runtime_name}' "
            "in a non-interactive environment.\n"
            "Remedy: If in a trusted environment, pass '--yes' / '-y', set REACH_YES=1, "
            "or configure 'trusted = true' in reach.toml.\n"
        )
        sys.stderr.flush()
        return 2

    # Interactive prompt
    headline, bullets = catalog_summary(skills, roots)
    content_lines = [f"[bold]Target Catalog:[/] {headline}"]
    content_lines.extend(f"  • {b}" for b in bullets)
    content_lines.append("")
    content_lines.append(f"[bold]Runtime:[/] {runtime_summary(runtime_name)}")
    content_lines.append(
        "Probing skills executes real agent processes that can run shell commands "
        "and file operations. Ensure you trust all resident skills."
    )
    content_lines.append("")
    content_lines.append(
        "[dim]Tip: For unverified skills, consider running Reach inside a container "
        "(e.g. Docker) or using the offline keyword driver (--agent keyword).[/dim]"
    )

    panel = safety_panel(content_lines)
    console.print(panel)

    try:
        console.print(f"Proceed with {action}? [y/N]: ", end="")
        response = input().strip().lower()
        if response in ("y", "yes"):
            return 0
        console.print("[dim]Aborted.[/dim]")
        return 1
    except (EOFError, KeyboardInterrupt):
        console.print("\n[dim]Aborted.[/dim]")
        return 1
