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

"""Execute catalog scaling sweeps across geometric scale steps and detect capacity knees."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated, NamedTuple

from cyclopts import Parameter

from reach.config import (
    RegistrySettings,
    RunConfig,
    RuntimeSettings,
    StudySettings,
    default_agent,
    resolve_sub_settings,
)
from reach.generate import checkpoint_path
from reach.models import Catalog, CatalogMode
from reach.queries import QuerySet, load_query_set
from reach.runtime import AgentRuntime, build_runtime
from reach.sweep import _resolve_anchor_skills, resolve_sweep_scales, run_scaling_sweep
from reach.views import Console, build_console, print_sweep, print_wrote, render_sweep

from .app import LOOP, app
from .discovery import REACH_DIR_NAME, _corpus, _no_skills, find_existing_queries_path
from .drafting import _draft_query_set
from .flags import (
    POSITIVE_INT,
    RATE,
    REGISTRY_GROUP,
    AgentName,
    ConfigFlag,
    EarlyStopFlag,
    Format,
    GenerateFlags,
    Global,
    RegistryFlags,
    YesFlag,
    agent_help_text,
)
from .safety import confirm_skill_execution

if TYPE_CHECKING:
    from collections.abc import Sequence

    from reach.models import Skill
    from reach.sweep import ScalingPoint, ScalingStudy


def _resolve_sweep_queries(
    queries: Path | None,
    configured: Path | None,
    skills_path: Path | None = None,
    *,
    auto_queries: bool = True,
    global_scope: bool = False,
) -> Path:
    """Resolve queries file from explicit argument, configuration, or .reach fallback."""
    resolved = queries if queries is not None else configured
    if resolved is not None:
        return resolved
    if found := find_existing_queries_path(skills_path):
        return found
    if auto_queries:
        if skills_path is not None and not global_scope:
            sp = Path(skills_path)
            base = sp if sp.is_dir() else sp.parent
            return base / REACH_DIR_NAME / "queries.json"
        return Path(REACH_DIR_NAME) / "queries.json"
    msg = (
        "scaling sweep requires a labeled query benchmark to test reachability across "
        "catalog scales.\n\n"
        "• Pass an existing queries file:\n"
        "    reach sweep ./skills --queries .reach/queries.json\n\n"
        "• Or allow Reach to automatically draft anchor queries (default):\n"
        "    reach sweep ./skills\n\n"
        "• Or draft benchmark queries first:\n"
        "    reach query draft --skills ./skills --out .reach/queries.json"
    )
    raise ValueError(msg)


def _resolve_sweep_out_path(
    out: Path | None,
    configured: Path | None,
    queries_path: Path | None = None,
) -> Path:
    """Determine the file path where the sweep artifact should be written."""
    if out is not None:
        return out
    if configured is not None:
        return configured
    if queries_path is not None:
        qp = Path(queries_path)
        if qp.parent.name == REACH_DIR_NAME:
            return qp.parent / "sweep.json"
    return Path(REACH_DIR_NAME) / "sweep.json"


def _write_sweep_file(
    study: ScalingStudy,
    *,
    format: Format,
    out: Path,
) -> None:
    """Serialize scaling study results to the destination file."""
    out_format = format
    if format == "text":
        if out.suffix == ".json":
            out_format = "json"
        elif out.suffix == ".csv":
            out_format = "csv"

    out_content = (
        render_sweep(study, out_format)
        if out_format in ("json", "csv")
        else render_sweep(study, "json")
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(out_content, encoding="utf-8")


def _output_sweep(
    console: Console,
    study: ScalingStudy,
    *,
    format: Format,
    out: Path,
) -> None:
    """Render and write scaling study results to disk."""
    _write_sweep_file(study, format=format, out=out)

    if format == "text":
        print_sweep(console, study)
        print_wrote(console, out)
    else:
        rendered = render_sweep(study, format)
        print(rendered)


_MAX_SAMPLE_SKILLS = 3


def _print_anchor_coverage(
    *,
    console: Console,
    queries_path: Path,
    anchor: str | None,
    configured_anchor: int | Sequence[str] | str | None,
    skills: Sequence[Skill],
    scales: Sequence[int] | None,
    target: str | None,
    clamp_to_queried: bool = True,
    query_set: QuerySet | None = None,
) -> None:
    """Print corpus and anchor query coverage summary and warn on 0-query skills."""
    if target is not None:
        return
    raw_qs = query_set
    if raw_qs is None:
        if not queries_path.exists():
            return
        try:
            raw_qs = load_query_set(queries_path)
        except (OSError, ValueError):
            return

    queried_corpus = raw_qs.covered_skills()
    missing_corpus = sorted(s.name for s in skills if s.name not in queried_corpus)
    if missing_corpus:
        sample = ", ".join(missing_corpus[:_MAX_SAMPLE_SKILLS])
        more = (
            f", +{len(missing_corpus) - _MAX_SAMPLE_SKILLS} more"
            if len(missing_corpus) > _MAX_SAMPLE_SKILLS
            else ""
        )
        console.print(
            f"[yellow]Warning:[/] {len(missing_corpus)} of {len(skills)} corpus skill(s) "
            f"have 0 queries in [cyan]{queries_path}[/] ([bold]{sample}{more}[/]). "
            f"Run [bold]reach query draft --missing[/] to backfill."
        )

    try:
        actual_scales = resolve_sweep_scales(len(skills), scales)
        resolved_anchors = _resolve_anchor_skills(
            anchor if anchor is not None else configured_anchor,
            skills,
            actual_scales,
            query_set=raw_qs,
            clamp_to_queried=clamp_to_queried,
        )
    except ValueError:
        return
    if not resolved_anchors:
        return
    counts = {a: sum(1 for q in raw_qs.queries if q.expected_skill == a) for a in resolved_anchors}
    covered = sum(1 for count in counts.values() if count > 0)
    total_matched = sum(counts.values())
    console.print(
        f"[dim]Anchor coverage:[/] {total_matched} queries matched across "
        f"{covered}/{len(resolved_anchors)} anchor skills"
    )
    zero_anchors = [a for a, count in counts.items() if count == 0]
    if zero_anchors:
        console.print(
            "[yellow]Warning:[/] anchor skill(s) with 0 matching queries in --queries: "
            f"[bold]{', '.join(zero_anchors)}[/]"
        )


@app.command(name="sweep", group=LOOP)
def _sweep(
    skills: Annotated[
        Path | None,
        Parameter(
            name=["skills", "--skills"],
            help="Path to the skill directory or corpus to sweep (discovered if omitted)",
        ),
    ] = None,
    *,
    target: Annotated[
        str | None,
        Parameter(
            name=["target", "--target"],
            help=(
                "Target skill to evaluate across scaling steps "
                "(omitted for whole-corpus capacity evaluation)"
            ),
        ),
    ] = None,
    queries: Annotated[
        Path | None,
        Parameter(
            name="--queries",
            help="Path to labeled evaluation queries JSON file",
        ),
    ] = None,
    scales: Annotated[
        str | None,
        Parameter(
            name="--scales",
            help="Comma-separated list of catalog sizes to evaluate",
        ),
    ] = None,
    anchor: Annotated[
        str | None,
        Parameter(
            name="--anchor",
            help=(
                "Anchor skills cohort evaluated across all scales. "
                "Defaults to cluster medoids of the initial scale step. "
                "Accepts integer count (e.g. 10), comma-separated skill names, "
                "or 'all' for full-corpus expansion."
            ),
        ),
    ] = None,
    rivals_share: Annotated[
        float,
        RATE,
        Parameter(
            name="--rivals-share",
            help="Proportion of distractor skills selected as nearest rivals",
        ),
    ] = 0.5,
    workers: Annotated[
        int | None,
        POSITIVE_INT,
        Parameter(
            name=["--workers", "-j"],
            help="Number of concurrent probe execution workers",
        ),
    ] = None,
    attempts: Annotated[
        int | None,
        POSITIVE_INT,
        Parameter(
            name=["--attempts", "-a"],
            help="Number of probe execution attempts per query at each scale step",
        ),
    ] = None,
    early_stop: EarlyStopFlag = True,
    noise_floor: Annotated[
        float,
        RATE,
        Parameter(
            name="--noise-floor",
            help="Minimum pass rate drop to trigger knee detection",
        ),
    ] = 0.05,
    agent: Annotated[
        AgentName | None,
        Parameter(
            name="--agent",
            show_choices=False,
            help=agent_help_text("Agent runtime to execute scaling probes"),
        ),
    ] = None,
    model: Annotated[
        str | None,
        Parameter(
            name=["--model", "-m"],
            help="Target model identifier",
        ),
    ] = None,
    global_: Global = False,
    registry: Annotated[
        RegistryFlags | None,
        Parameter(group=REGISTRY_GROUP),
    ] = None,
    format: Annotated[Format, Parameter(help="Output format: text, json, csv")] = "text",
    out: Annotated[
        Path | None,
        Parameter(
            name=["--out", "-o"],
            help="Where to write sweep output (default: .reach/sweep.json)",
        ),
    ] = None,
    workdir: Annotated[
        Path | None,
        Parameter(
            name="--workdir",
            help="Working directory for probe execution",
        ),
    ] = None,
    allow_truncation: Annotated[
        bool,
        Parameter(
            name="--allow-truncation",
            negative="--no-allow-truncation",
            help=(
                "Probe scaling steps even if catalogs exceed the runtime listing "
                "budget (default: True)"
            ),
        ),
    ] = True,
    auto_queries: Annotated[
        bool | None,
        Parameter(
            name="--auto-queries",
            negative="--no-auto-queries",
            show_default=False,
            help=(
                "Automatically synthesize and backfill queries for unqueried "
                "anchor skills (default: True)"
            ),
        ),
    ] = None,
    yes: YesFlag = False,
    config: ConfigFlag = None,
) -> int:
    """Execute multi-scale catalog evaluation sweeps to measure reachability decay."""
    console = build_console()
    run_config: RunConfig | None = RunConfig.from_toml(config) if config is not None else None

    if target is not None:
        from reach.catalog import resolve_skill_target

        try:
            resolved_target = resolve_skill_target(
                target, explicit_catalog=skills, command_name="sweep"
            )
            if resolved_target is not None:
                target = resolved_target.skill_name
                if skills is None and resolved_target.catalog_path:
                    skills = resolved_target.catalog_path
        except (FileNotFoundError, ValueError) as err:
            console.print(f"[red]Error:[/] {err}")
            return 2

    effective_config, driver = _resolve_sweep_effective_config(
        run_config=run_config,
        agent=agent,
        model=model,
        registry=registry,
        skills=skills,
        queries=queries,
        workdir=workdir,
        out=out,
        auto_queries=auto_queries,
    )

    found = _load_sweep_corpus(
        console=console,
        driver=driver,
        effective_config=effective_config,
        skills=skills,
        global_scope=global_,
    )

    try:
        parsed_scales = _parse_scales_cli(scales, effective_config.study.scales)
    except ValueError as err:
        console.print(f"[red]Error:[/] {err}")
        return 2

    effective_config, resolved_queries = _finalize_sweep_study_config(
        effective_config,
        found,
        skills,
        queries,
        workdir,
        early_stop,
        global_scope=global_,
    )
    destination = _resolve_sweep_out_path(
        out,
        configured=effective_config.study.out,
        queries_path=resolved_queries,
    )
    effective_config = effective_config.model_copy(
        update={
            "study": effective_config.study.model_copy(update={"out": destination}),
        }
    )

    if code := _prepare_and_confirm_sweep(
        console=console,
        driver=driver,
        effective_config=effective_config,
        found=found,
        resolved_queries=resolved_queries,
        anchor=anchor,
        scales=parsed_scales or effective_config.study.scales,
        target=target,
        allow_truncation=allow_truncation,
        format=format,
        yes=yes,
    ):
        return code

    def _on_scale_complete(
        step: int,
        total: int,
        point: ScalingPoint,
        partial_study: ScalingStudy,
    ) -> None:
        _write_sweep_file(partial_study, format=format, out=destination)
        if format == "text":
            secondary = (
                f"F1={point.f1_score:.1%}"
                if partial_study.is_corpus_sweep
                else f"recall={point.recall:.1%}"
            )
            console.print(
                f"  [dim]\\[{step}/{total}][/] Scale [bold]K={point.scale}[/]: "
                f"pass_rate={point.pass_rate:.1%}, {secondary} "
                f"[dim]({point.probes_executed} probes)[/]"
            )

    try:
        study = run_scaling_sweep(
            config=effective_config,
            target_skill=target,
            scales=parsed_scales,
            anchor=anchor,
            runtime=driver,
            rivals_share=rivals_share,
            noise_floor=noise_floor,
            workers=workers if workers is not None else effective_config.plan.workers,
            skills=found,
            attempts=attempts,
            early_stop=early_stop,
            allow_truncation=allow_truncation,
            on_scale_complete=_on_scale_complete,
        )
    except ValueError as err:
        console.print(f"[red]Error:[/] {err}")
        return 2
    except (OSError, RuntimeError) as err:
        console.print(f"[red]Runtime Error:[/] {err}")
        return 3

    _output_sweep(console, study, format=format, out=destination)
    return 0


class _SweepQueryResolution(NamedTuple):
    """Hold missing anchor target skill names and cached query set."""

    missing_targets: tuple[str, ...]
    existing_query_set: QuerySet | None


def _prepare_and_confirm_sweep(
    *,
    console: Console,
    driver: AgentRuntime,
    effective_config: RunConfig,
    found: Sequence[Skill],
    resolved_queries: Path,
    anchor: str | None,
    scales: Sequence[int] | None,
    target: str | None,
    allow_truncation: bool,
    format: Format,
    yes: bool,
) -> int:
    """Confirm skill execution, draft missing anchor queries, and print preflight notices."""
    resolution = (
        _resolve_missing_sweep_targets(
            queries_path=resolved_queries,
            anchor=anchor,
            configured_anchor=effective_config.study.anchor,
            skills=found,
            scales=scales,
            target=target,
        )
        if effective_config.study.auto_queries
        else _SweepQueryResolution((), None)
    )

    if code := confirm_skill_execution(
        console,
        runtime_name=driver.name,
        skills=found,
        action="scaling sweep",
        yes=yes,
        trusted=effective_config.study.trusted,
    ):
        return code

    fresh_qs: QuerySet | None = resolution.existing_query_set
    if resolution.missing_targets:
        if code := _draft_missing_sweep_queries(
            console=console,
            effective_config=effective_config,
            found=found,
            resolved_queries=resolved_queries,
            missing_targets=resolution.missing_targets,
            format=format,
            existing_qs=resolution.existing_query_set,
        ):
            return code
        fresh_qs = None

    if format == "text":
        console.print(f"[dim]Using benchmark queries from:[/] [cyan]{resolved_queries}[/]")
        _print_anchor_coverage(
            console=console,
            queries_path=resolved_queries,
            anchor=anchor,
            configured_anchor=effective_config.study.anchor,
            skills=found,
            scales=scales,
            target=target,
            clamp_to_queried=not effective_config.study.auto_queries,
            query_set=fresh_qs,
        )
        if driver.rations_catalog and allow_truncation and found:
            fit = driver.fit(
                Catalog(
                    id="sweep:corpus:full",
                    mode=CatalogMode.SWEEP,
                    skills=tuple(s.name for s in found),
                ),
                found,
            )
            if not fit.whole:
                console.print(
                    f"[yellow]Notice:[/] full catalog ({fit.asked:,} chars) exceeds "
                    f"[bold]{driver.name}[/] listing budget ({fit.allowed:,} chars); "
                    f"{fit.truncated} of {len(found)} descriptions will be truncated to "
                    "bare names at higher scales."
                )
        console.print()
    return 0


def _resolve_missing_sweep_targets(
    *,
    queries_path: Path,
    anchor: str | None,
    configured_anchor: int | Sequence[str] | str | None,
    skills: Sequence[Skill],
    scales: Sequence[int] | None,
    target: str | None,
) -> _SweepQueryResolution:
    """Identify anchor or target skills that lack positive queries in the query file."""
    existing_query_set = load_query_set(queries_path) if queries_path.is_file() else None
    query_set = existing_query_set or QuerySet(queries=())
    if target is not None:
        anchor_names: tuple[str, ...] = tuple(s.name for s in skills if s.name == target)
    else:
        try:
            actual_scales = resolve_sweep_scales(len(skills), scales)
            resolved_anchors = _resolve_anchor_skills(
                anchor if anchor is not None else configured_anchor,
                skills,
                actual_scales,
                query_set=query_set,
                clamp_to_queried=False,
            )
        except ValueError:
            return _SweepQueryResolution((), existing_query_set)
        anchor_names = (
            resolved_anchors if resolved_anchors is not None else tuple(s.name for s in skills)
        )
    queried_names = query_set.covered_skills()
    missing = tuple(name for name in anchor_names if name not in queried_names)
    return _SweepQueryResolution(missing, existing_query_set)


def _draft_missing_sweep_queries(
    *,
    console: Console,
    effective_config: RunConfig,
    found: Sequence[Skill],
    resolved_queries: Path,
    missing_targets: tuple[str, ...],
    format: Format,
    existing_qs: QuerySet | None = None,
) -> int:
    """Draft or backfill missing anchor queries and save them to resolved_queries."""
    draft_console = console if format == "text" else build_console(quiet=True)
    if format == "text":
        draft_console.print(
            f"[dim]Drafting queries for {len(missing_targets)} unqueried anchor "
            f"skill(s) ({', '.join(missing_targets)}) ->[/] [cyan]{resolved_queries}[/]"
        )

    flags = GenerateFlags.from_query_settings(
        effective_config.query,
        targets=missing_targets,
    )
    draft_settings = effective_config.with_overrides(
        catalog={"mode": CatalogMode.ALL},
        study={"queries": resolved_queries},
    )
    try:
        code = _draft_query_set(
            draft_console,
            draft_settings,
            found,
            flags,
            dry_run=False,
            same_invocation_probe=True,
            existing_query_set=existing_qs,
        )
        if code != 0:
            checkpoint_path(resolved_queries).unlink(missing_ok=True)
            return code
    except (OSError, ValueError, RuntimeError) as err:
        checkpoint_path(resolved_queries).unlink(missing_ok=True)
        console.print(f"[red]Error drafting queries:[/] {err}")
        return 2
    return 0


def _parse_scales_cli(
    scales: str | None,
    configured_scales: tuple[int, ...] | None,
) -> tuple[int, ...] | None:
    """Parse comma-separated scale integers, falling back to configured scales."""
    if not scales:
        return configured_scales
    try:
        return tuple(int(s.strip()) for s in scales.split(",") if s.strip())
    except ValueError as err:
        msg = "Invalid scales format. Use comma-separated integers, e.g. 10,25,50,100"
        raise ValueError(msg) from err


def _resolve_sweep_effective_config(
    run_config: RunConfig | None,
    agent: str | None,
    model: str | None,
    registry: RegistryFlags | None,
    skills: Path | None,
    queries: Path | None,
    workdir: Path | None,
    out: Path | None,
    auto_queries: bool | None = None,
) -> tuple[RunConfig, AgentRuntime]:
    """Resolve layered runtime, registry, and study settings across CLI flags and configs."""
    resolved_agent = (
        agent
        or (
            run_config.runtime.agent
            if run_config is not None and run_config.runtime.agent
            else None
        )
        or (
            run_config.general.default_agent
            if run_config is not None and run_config.general.default_agent
            else None
        )
        or default_agent()
    )
    eff_runtime = RunConfig.resolve(
        RuntimeSettings,
        run_config,
        agent=resolved_agent,
        model=model,
    )
    if not eff_runtime.agent:
        eff_runtime = eff_runtime.model_copy(update={"agent": "keyword"})

    eff_registry = resolve_sub_settings(
        RegistrySettings,
        run_config.registry if run_config is not None else None,
        **(registry.overrides() if registry is not None else {}),
    )

    eff_study = RunConfig.resolve(
        StudySettings,
        run_config,
        skills=skills,
        queries=queries,
        workdir=workdir,
        out=out,
        auto_queries=auto_queries,
    )

    effective_config = (run_config or RunConfig()).model_copy(
        update={
            "runtime": eff_runtime,
            "registry": eff_registry,
            "study": eff_study,
        }
    )
    driver = build_runtime(eff_runtime)
    return effective_config, driver


def _load_sweep_corpus(
    console: Console,
    driver: AgentRuntime,
    effective_config: RunConfig,
    skills: Path | None,
    global_scope: bool,
) -> tuple[Skill, ...]:
    """Load or discover candidate skills for sweep execution."""
    if skills is not None or effective_config.study.skills is None:
        eff_runtime = effective_config.runtime
        found, _roots, _discovered = _corpus(
            console,
            driver,
            settings=effective_config,
            global_scope=global_scope,
            agent=eff_runtime.agent,
        )
    else:
        from reach.catalog import load_skills

        found = load_skills(effective_config.require_skills())

    if not found:
        raise _no_skills(skills, global_scope=global_scope)
    return tuple(found)


def _finalize_sweep_study_config(
    effective_config: RunConfig,
    found: Sequence[Skill],
    skills: Path | None,
    queries: Path | None,
    workdir: Path | None,
    early_stop: bool,
    *,
    global_scope: bool = False,
) -> tuple[RunConfig, Path]:
    """Determine working directory and resolved benchmark queries path."""
    work_dir = workdir or effective_config.study.workdir
    resolved_skills = (
        found[0].path.parent
        if effective_config.study.skills is None and found
        else effective_config.study.skills
    )
    resolved_queries = _resolve_sweep_queries(
        queries,
        effective_config.study.queries,
        skills_path=resolved_skills or skills,
        auto_queries=effective_config.study.auto_queries,
        global_scope=global_scope,
    )
    updated_study = effective_config.study.model_copy(
        update={
            "workdir": work_dir,
            "skills": resolved_skills,
            "queries": resolved_queries,
            "early_stop": early_stop,
        }
    )
    return effective_config.model_copy(update={"study": updated_study}), resolved_queries
