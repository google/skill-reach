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

"""Clean cached Agent Registry payloads and evaluation artifacts."""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Annotated

from cyclopts import Parameter

from reach.artifact import ARTIFACT_SUFFIX
from reach.config import resolve_path
from reach.registry import RegistryCacheManager
from reach.views import build_console

from .app import SETUP, app
from .flags import SWITCH, ProjectFlag, Quiet

_BYTES_PER_KB = 1024
_BYTES_PER_MB = 1024 * 1024
_BYTES_PER_GB = 1024 * 1024 * 1024
CONFIG_SIDECAR_GLOB = "*.config.json"


def _format_size(num_bytes: int) -> str:
    """Format byte count into human-readable string (B, KB, MB, GB)."""
    if num_bytes < _BYTES_PER_KB:
        return f"{num_bytes} B"
    if num_bytes < _BYTES_PER_MB:
        return f"{num_bytes / _BYTES_PER_KB:.1f} KB"
    if num_bytes < _BYTES_PER_GB:
        return f"{num_bytes / _BYTES_PER_MB:.1f} MB"
    return f"{num_bytes / _BYTES_PER_GB:.1f} GB"


@app.command(group=SETUP)
def clean(
    *,
    project: ProjectFlag = None,
    dry_run: Annotated[
        bool,
        SWITCH,
        Parameter(
            alias="-n",
            help="Display paths and space that would be reclaimed without deleting",
        ),
    ] = False,
    purge_all: Annotated[
        bool,
        SWITCH,
        Parameter(
            name=["--all"],
            help=(
                "Purge all caches, run artifacts (.reach/eval.json, .reach/sweep.json, "
                "*.artifact.json), and query sets (.reach/queries.*)"
            ),
        ),
    ] = False,
    quiet: Quiet = False,
) -> int:
    """Clean cached Agent Registry payloads and evaluation artifacts.

    By default, clean safely purges the .reach/cache/registry/ directory containing
    cached Agent Registry metadata and revision bundles without touching
    user-generated evaluation results or query sets.

    Examples:
        reach clean                  # Clean all registry cache files
        reach clean --dry-run        # Preview what would be cleaned
        reach clean --project <ID>   # Clean cache for a specific GCP project
        reach clean --all            # Purge cache, run files, and queries
    """
    console = build_console(quiet=quiet)
    cache_mgr = RegistryCacheManager()

    try:
        total_bytes, paths = cache_mgr.clean(project=project, dry_run=dry_run)
    except ValueError as err:
        console.print(f"[bold red]Error:[/bold red] {err}")
        return 1

    # If --all is passed, also clean .reach run artifacts
    extra_paths: list[Path] = []
    if purge_all:
        reach_dir = resolve_path(".reach")
        if reach_dir.is_dir():
            target_files = (
                "eval.json",
                "report.html",
                "queries.json",
                "queries.csv",
                "queries.jsonl",
                "queries-citations.json",
                "sweep.json",
            )
            # Sidecars carry a reach-owned suffix, so sweep them up wherever they landed.
            targets = {reach_dir / fname for fname in target_files}
            targets.update(reach_dir.glob(f"*{ARTIFACT_SUFFIX}"))
            targets.update(reach_dir.glob(CONFIG_SIDECAR_GLOB))
            for extra in sorted(targets):
                if extra.is_file():
                    with contextlib.suppress(OSError):
                        sz = extra.stat().st_size
                        total_bytes += sz
                        extra_paths.append(extra)
                        if not dry_run:
                            extra.unlink(missing_ok=True)

    all_paths = paths + extra_paths

    if not all_paths or total_bytes == 0:
        if not quiet:
            console.print("[dim]Cache is already clean (0 bytes reclaimed).[/dim]")
        return 0

    if dry_run:
        console.print("[bold]Would remove the following paths:[/bold]")
        for p in all_paths:
            console.print(f"  [dim]•[/dim] {p}")
        formatted_total = _format_size(total_bytes)
        console.print(f"\n[bold green]Total space to reclaim:[/bold green] {formatted_total}")
    else:
        console.print(
            f"[bold green]✓[/bold green] Cleaned {len(all_paths)} target(s) "
            f"([bold]{_format_size(total_bytes)}[/bold] reclaimed).",
        )

    return 0
