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

"""Optimize skill descriptions with closed-loop candidate generation and empirical probes."""

from __future__ import annotations

import contextlib
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Final, Literal

from cyclopts import Parameter

if TYPE_CHECKING:
    from reach.config import RunConfig
    from reach.optimize import OptimizationReport
    from reach.views import Console

from reach.optimize import (
    DEFAULT_BUDGET,
    DEFAULT_HOLDOUT,
    DEFAULT_ITERATIONS,
)

from .app import LOOP, app
from .flags import POSITIVE_INT, RATE, SWITCH, AgentName, Global, YesFlag, agent_help_text

type OptimizeFormat = Literal["text", "json", "diff"]

#: Default number of candidate descriptions to synthesize during optimization.
DEFAULT_CANDIDATES: Final = 3

#: Default probe budget allocated across candidate evaluations.
DEFAULT_OPTIMIZE_BUDGET: Final = DEFAULT_BUDGET


def _confirm_optimize_safety(
    console: Console,
    skills: Path | None,
    resolved_agent: str,
    *,
    yes: bool,
    run_config: RunConfig | None,
    config: Path | None = None,
    global_scope: bool = False,
) -> int:
    """Prompt for safety confirmation before launching live optimization probes."""
    from reach.catalog import load_skills
    from reach.cli.safety import confirm_skill_execution
    from reach.config import resolve_discovery_candidates

    if skills is not None:
        roots = [skills]
    else:
        workdir = Path.home() if global_scope else Path.cwd().resolve()
        candidates = resolve_discovery_candidates(
            workdir,
            agent=resolved_agent,
            config_path=config,
            global_scope=global_scope,
        )
        roots = candidates or [workdir]

    loaded_skills = 0
    for root in roots:
        if root.is_dir():
            with contextlib.suppress(OSError, ValueError):
                loaded_skills += len(load_skills(root))
    if loaded_skills == 0:
        loaded_skills = 1

    trusted = run_config.study.trusted if run_config is not None else False
    return confirm_skill_execution(
        console,
        runtime_name=resolved_agent,
        skills=loaded_skills,
        roots=roots,
        action="optimization",
        yes=yes,
        trusted=trusted,
    )


@app.command(name="optimize", group=LOOP)
def _optimize(
    skill: Annotated[
        str,
        Parameter(
            name=["skill", "--skill"],
            help="Name of the target skill to optimize, or path to skill directory / SKILL.md",
        ),
    ],
    *,
    skills: Annotated[
        Path | None,
        Parameter(
            help=(
                "Path to a skill directory, SKILL.md file, or catalog tree (discovered if omitted)"
            ),
        ),
    ] = None,
    queries: Annotated[
        Path | None,
        Parameter(
            help="Path to labeled evaluation queries JSON file for empirical validation",
        ),
    ] = None,
    candidates: Annotated[
        int,
        POSITIVE_INT,
        Parameter(
            help="Number of candidate descriptions to synthesize (default: 3)",
        ),
    ] = DEFAULT_CANDIDATES,
    budget: Annotated[
        int,
        POSITIVE_INT,
        Parameter(
            help="Maximum empirical probes to execute across candidate evaluations (default: 30)",
        ),
    ] = DEFAULT_OPTIMIZE_BUDGET,
    iterations: Annotated[
        int,
        POSITIVE_INT,
        Parameter(
            name=["--iterations", "-i"],
            help=(
                f"Number of iterative hill-climbing refinement rounds "
                f"(default: {DEFAULT_ITERATIONS})"
            ),
        ),
    ] = DEFAULT_ITERATIONS,
    holdout: Annotated[
        float,
        RATE,
        Parameter(
            name="--holdout",
            help="Fraction of queries held out for evaluation (0.0 - 0.9, default: 0.2)",
        ),
    ] = DEFAULT_HOLDOUT,
    review: Annotated[
        bool,
        SWITCH,
        Parameter(
            name="--review",
            help="Launch interactive browser review for generated queries",
        ),
    ] = False,
    auto_queries: Annotated[
        bool,
        Parameter(
            help="Synthesize adversarial queries if none are provided (default: True)",
        ),
    ] = True,
    agent: Annotated[
        AgentName | None,
        Parameter(
            name="--agent",
            show_choices=False,
            help=agent_help_text(
                "Agent runtime for candidate empirical probing (default: from reach.toml)",
            ),
        ),
    ] = None,
    global_: Global = False,
    auto_apply: Annotated[
        bool,
        SWITCH,
        Parameter(
            name="--auto-apply",
            help="Automatically write the highest-ranking candidate description to SKILL.md",
        ),
    ] = False,
    force: Annotated[
        bool,
        SWITCH,
        Parameter(
            name=["--force", "-f"],
            help="Force apply candidate to SKILL.md even if no empirical improvement is detected",
        ),
    ] = False,
    yes: YesFlag = False,
    candidate: Annotated[
        int,
        POSITIVE_INT,
        Parameter(
            name=["--candidate", "-c"],
            help="1-based candidate rank to inspect diff or apply (default: 1)",
        ),
    ] = 1,
    workers: Annotated[
        int | None,
        POSITIVE_INT,
        Parameter(
            name=["--workers", "-j"],
            help="Number of parallel probe workers (default: from reach.toml or 4)",
        ),
    ] = None,
    with_handoff: Annotated[
        bool,
        SWITCH,
        Parameter(
            name="--with-handoff",
            help="Synthesize and stage reciprocal Layer-2 SKILL.md Routing Notes",
        ),
    ] = False,
    format: Annotated[
        OptimizeFormat,
        Parameter(
            help="Output format: text, json, diff (default: text)",
        ),
    ] = "text",
    config: Annotated[
        Path | None,
        Parameter(
            name="--config",
            help="Path to reach.toml configuration file",
        ),
    ] = None,
) -> int:
    """Optimize a skill's description using candidate synthesis and empirical probes."""
    from reach.optimize import optimize_skill
    from reach.views import Console, build_console

    console = build_console()
    err_console = Console(stderr=True)

    from reach.config import OptimizeSettings, RunConfig, default_agent

    run_config = None
    if config is not None:
        run_config = RunConfig.from_toml(config)

    resolved_agent = agent or (
        run_config.runtime.agent if run_config is not None else default_agent()
    )
    runtime_options = (
        dict(run_config.runtime.options)
        if run_config is not None and run_config.runtime.options
        else None
    )
    eff_settings = RunConfig.resolve(
        OptimizeSettings,
        config=run_config,
        budget=budget if budget != DEFAULT_OPTIMIZE_BUDGET or run_config is None else None,
        iterations=iterations if iterations != DEFAULT_ITERATIONS or run_config is None else None,
        holdout=holdout if holdout != DEFAULT_HOLDOUT or run_config is None else None,
        review=review if review or run_config is None else None,
        auto_queries=auto_queries if not auto_queries or run_config is None else None,
        workers=workers,
        with_handoff=with_handoff if with_handoff or run_config is None else None,
    )

    from reach.catalog import resolve_skill_target

    try:
        resolved = resolve_skill_target(skill, explicit_catalog=skills, command_name="optimize")
    except (FileNotFoundError, ValueError) as err:
        console.print(f"[red]Error:[/] {err}")
        return 2

    effective_skill = resolved.skill_name if resolved else skill
    effective_skills = skills or (resolved.catalog_path if resolved else None)

    if code := _confirm_optimize_safety(
        console,
        effective_skills,
        resolved_agent,
        yes=yes,
        run_config=run_config,
        config=config,
        global_scope=global_,
    ):
        return code

    from contextlib import nullcontext

    is_tty = sys.stderr.isatty()
    status_msg = (
        f"[cyan]Optimizing skill [bold]{effective_skill}[/bold] "
        f"(budget: {eff_settings.budget}, workers: {eff_settings.workers})...[/cyan]"
    )
    status_ctx = console.status(status_msg) if format == "text" and is_tty else nullcontext()

    def _on_progress(msg: str) -> None:
        if format != "text":
            return
        update_fn = getattr(status_ctx, "update", None)
        if is_tty and callable(update_fn):
            update_fn(f"[cyan]{msg}[/cyan]")
        else:
            err_console.print(f"[dim]\\[reach optimize][/dim] {msg}")

    try:
        with status_ctx:
            report = optimize_skill(
                skill_name=effective_skill,
                skills_path=effective_skills,
                queries_path=queries,
                agent=resolved_agent,
                candidates_count=candidates,
                auto_apply=auto_apply,
                runtime_options=runtime_options,
                force=force,
                config=config,
                global_scope=global_,
                settings=eff_settings,
                candidate_index=candidate,
                progress_callback=_on_progress if format == "text" else None,
            )
    except ValueError as err:
        console.print(f"[red]Error:[/] {err}")
        return 2
    except (OSError, RuntimeError) as err:
        console.print(f"[red]Runtime Error:[/] {err}")
        return 3

    return _render_optimization_output(
        console,
        report,
        format=format,
        candidate=candidate,
        auto_apply=auto_apply,
        force=force,
        yes=yes,
    )


def _render_optimization_output(
    console: Console,
    report: OptimizationReport,
    *,
    format: OptimizeFormat,
    candidate: int,
    auto_apply: bool,
    force: bool,
    yes: bool = False,
) -> int:
    """Render optimization report in requested format or launch interactive prompt."""
    from reach.views import print_optimization, render_optimization_diff

    match format:
        case "json":
            print(report.model_dump_json(indent=2))
        case "diff":
            if candidate < 1 or (report.candidates and candidate > len(report.candidates)):
                console.print(
                    f"[red]Error:[/] Candidate index #{candidate} out of range "
                    f"(available: 1..{len(report.candidates)})."
                )
                return 2
            diff_text = render_optimization_diff(report, candidate_index=candidate)
            if diff_text:
                print(diff_text, end="")
            else:
                console.print("[dim]No modifications recommended or diff unavailable.[/]")
        case _:
            print_optimization(console, report)
            if (
                not auto_apply
                and not yes
                and report.candidates
                and sys.stdin.isatty()
                and sys.stdout.isatty()
            ):
                _prompt_interactive_apply(
                    console,
                    report,
                    default_candidate=candidate,
                    force=force,
                )

    return 0


def _inspect_interactive_diff(
    console: Console,
    report: OptimizationReport,
    diff_cand: int,
    n_cands: int,
) -> bool:
    """Render diff for candidate and prompt user whether to apply it."""
    from reach.views import render_optimization_diff

    if not (1 <= diff_cand <= n_cands):
        console.print(f"[red]Error:[/] Candidate index #{diff_cand} out of range (1..{n_cands})")
        return False

    diff_text = render_optimization_diff(report, candidate_index=diff_cand)
    if diff_text:
        console.print(diff_text)

    manifest_name = report.manifest_path.name if report.manifest_path else "SKILL.md"
    prompt = f"Apply candidate #{diff_cand} to {manifest_name}? [y/N]: "
    return input(prompt).strip().lower() in ("y", "yes")


def _parse_apply_choice(
    response: str,
    def_idx: int,
    n_cands: int,
    console: Console,
    report: OptimizationReport,
) -> tuple[bool, int] | None:
    """Parse user interactive candidate selection input."""
    parts = response.split()
    if parts and parts[0].lower() in ("d", "diff"):
        diff_cand = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else def_idx
        if _inspect_interactive_diff(console, report, diff_cand, n_cands):
            return True, diff_cand
        return False, def_idx

    if response.isdigit():
        cand_num = int(response)
        if 1 <= cand_num <= n_cands:
            return True, cand_num
        console.print(f"[red]Error:[/] Candidate index #{cand_num} out of range (1..{n_cands})")
        return None

    return response.lower() in ("y", "yes"), def_idx


def _prompt_interactive_apply(
    console: Console,
    report: OptimizationReport,
    default_candidate: int = 1,
    force: bool = False,
) -> None:
    """Prompt the user interactively to apply one of the proposed candidate descriptions."""
    if not report.candidates or not report.manifest_path:
        return

    from reach.optimize import apply_optimization_candidate

    n_cands = len(report.candidates)
    def_idx = default_candidate if 1 <= default_candidate <= n_cands else 1

    has_improvement = report.has_improvement
    user_chose_candidate = default_candidate != 1
    if not has_improvement and not force and not user_chose_candidate:
        return

    console.print()
    try:
        cand_range = f"1..{n_cands}" if n_cands > 1 else "1"
        manifest_name = report.manifest_path.name
        prompt_msg = (
            f"Apply candidate #{def_idx} to {manifest_name}? "
            f"[{cand_range}, y, N, d(iff), d <num>]: "
        )
        response = input(prompt_msg).strip()
        parsed = _parse_apply_choice(response, def_idx, n_cands, console, report)
        if parsed is None:
            return

        should_apply, selected_idx = parsed
        if should_apply:
            target_cand = report.candidates[selected_idx - 1]
            if apply_optimization_candidate(report, target_cand):
                console.print(
                    f"[green]✓[/green] Applied candidate #{selected_idx} description to "
                    f"[bold]{report.manifest_path}[/bold]"
                )
            else:
                console.print(f"[red]Error:[/] Failed to update {report.manifest_path}")
        else:
            console.print("[dim]Skipped applying candidate.[/dim]")
    except (EOFError, KeyboardInterrupt):
        console.print("\n[dim]Skipped applying candidate.[/dim]")
