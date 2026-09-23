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

"""Initialize new Reach project configuration and skill workspace scaffold."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Annotated

from cyclopts import Parameter

from reach.config import KNOWN_CLIENT_SKILLS_DIRS
from reach.views import build_console

from .app import SETUP, app
from .flags import SWITCH, AgentName, Quiet, agent_help_text


def _detect_skills_directory(root: Path, agent: str | None = None) -> Path:
    """Discover existing skill directory or determine canonical default."""
    candidates = [
        root / ".agents" / "skills",
        root / ".claude" / "skills",
        root / ".cursor" / "skills",
        root / "skills",
    ]
    for cand in candidates:
        if cand.is_dir():
            return cand
    if agent:
        native = KNOWN_CLIENT_SKILLS_DIRS.get(agent)
        if native:
            return root / native
    return root / ".agents" / "skills"


def _detect_default_agent() -> str:
    """Determine the most appropriate default agent runtime based on local environment."""
    if shutil.which("claude") is not None or os.environ.get("ANTHROPIC_API_KEY"):
        return "claude-code"
    if shutil.which("agy") is not None or os.environ.get("GEMINI_API_KEY"):
        return "antigravity-cli"
    return "claude-code"


def generate_reach_toml(
    agent: str,
    skills_dir: str,
    *,
    queries_path: str = ".reach/queries.json",
    out_path: str = ".reach/eval.json",
) -> str:
    """Generate starter reach.toml content with documented defaults."""
    return f"""# reach.toml - Reach configuration for skill evaluation and optimization
# Documentation: https://google.github.io/skill-reach/configuration/

[runtime]
agent = "{agent}"
# timeout_s = 60

[catalog]
mode = "neighborhood"

[study]
skills = "{skills_dir}"
queries = "{queries_path}"
out = "{out_path}"
catalog = "auto"
"""


@app.command(name="init", group=SETUP)
def _init(
    *,
    path: Annotated[
        Path | None,
        Parameter(
            name=["--path", "-p"],
            help="Project root directory to initialize (defaults to current working directory)",
        ),
    ] = None,
    agent: Annotated[
        AgentName | None,
        Parameter(
            name="--agent",
            show_choices=False,
            help=agent_help_text("Default agent runtime to configure in reach.toml"),
        ),
    ] = None,
    skills: Annotated[
        Path | None,
        Parameter(
            name=["--skills", "-s"],
            help="Directory path where skills are stored (defaults to .agents/skills)",
        ),
    ] = None,
    force: Annotated[
        bool,
        SWITCH,
        Parameter(
            name=["--force", "-f"],
            help="Overwrite existing reach.toml configuration if present",
        ),
    ] = False,
    quiet: Quiet = False,
) -> int:
    """Scaffold reach.toml configuration and initialize skill directories."""
    console = build_console(quiet=quiet)
    cwd = (path or Path.cwd()).resolve()
    config_file = cwd / "reach.toml"

    if config_file.exists() and not force:
        console.print(
            f"[yellow]Warning:[/] reach.toml already exists at {config_file}. "
            "Use [bold]--force[/bold] / [bold]-f[/bold] to overwrite.",
        )
        return 1

    resolved_agent = agent or _detect_default_agent()
    resolved_skills = skills or _detect_skills_directory(cwd, agent=resolved_agent)
    resolved_skills_str = str(
        resolved_skills.relative_to(cwd) if resolved_skills.is_relative_to(cwd) else resolved_skills
    )

    # Ensure directories exist
    resolved_skills.mkdir(parents=True, exist_ok=True)
    reach_dir = cwd / ".reach"
    reach_dir.mkdir(parents=True, exist_ok=True)

    toml_content = generate_reach_toml(
        agent=resolved_agent,
        skills_dir=resolved_skills_str,
    )
    config_file.write_text(toml_content, encoding="utf-8")

    console.print(f"[green]✓[/green] Created [bold]{config_file.name}[/bold]")
    console.print(
        f"[green]✓[/green] Configured skill directory at [bold]{resolved_skills_str}[/bold]"
    )
    console.print(
        f"[green]✓[/green] Configured default agent runtime: [bold]{resolved_agent}[/bold]"
    )
    console.print()
    console.print("[bold cyan]Next Steps:[/bold cyan]")
    console.print("  1. Run [bold]reach doctor[/bold] to verify your environment and API keys")
    console.print(
        f"  2. Place your skills ([dim]SKILL.md[/dim]) in [bold]{resolved_skills_str}/[/bold]"
    )
    console.print("  3. Run [bold]reach lint[/bold] to validate skill frontmatter and budgets")
    console.print("  4. Run [bold]reach eval[/bold] to measure reachability against rivals")
    return 0
