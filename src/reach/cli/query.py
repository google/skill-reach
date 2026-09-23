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

"""Manage query sets: draft synthetic queries, export, import, and inspect sets."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Literal

from cyclopts import App, Group, Parameter
from pydantic import Field

from reach.config import RunConfig, RuntimeSettings
from reach.difficulty import lexical_ranks
from reach.discovery import resolve_corpus
from reach.exchange import (
    Exchange,
    FieldMap,
    export_query_set,
    import_query_set,
    infer_format,
)
from reach.generate import citations_path, read_citations
from reach.leak import leaks
from reach.models import CatalogMode, Skill
from reach.queries import QuerySet, load_query_set, save_query_set
from reach.runtime import build_runtime
from reach.views import (
    Console,
    build_console,
    help_formatter,
    print_discovery,
    print_query_view,
    print_wrote,
)

from .app import LOOP, OPTIONS, app
from .discovery import (
    DEFAULT_QUERIES_PATH,
    _asks_for_a_mode,
    _corpus,
    _no_skills,
    _run_dir_study,
    find_existing_queries_path,
)
from .drafting import _draft_query_set
from .flags import (
    CATALOG_GROUP,
    FLAT,
    GENERATE_GROUP,
    POSITIVE_INT,
    REGISTRY_GROUP,
    RUNTIME_GROUP,
    STUDY_GROUP,
    SWITCH,
    AgentName,
    CatalogFlags,
    ConfigFlag,
    Flags,
    GenerateFlags,
    Global,
    Quiet,
    RegistryFlags,
    RuntimeFlags,
    StudyFlags,
    agent_help_text,
    build_config,
)

#: Help panel group for import field mapping parameters.
MAPPING_GROUP = Group("Field mapping", sort_key=7)

#: Help panel group for query sub-commands.
QUERY_OPERATIONS = Group("Operations", sort_key=0)

#: Required fields when drafting query sets without configuration files.
DRAFT_REQUIRED: tuple[str, ...] = ()


@FLAT
class MapFlags(Flags):
    """CLI parameter flags configuring field mappings for external dataset imports."""

    text_column: Annotated[
        str | None,
        Field(serialization_alias="text"),
        Parameter(help="Column holding the query text"),
    ] = None
    id_column: Annotated[
        str | None,
        Field(serialization_alias="id"),
        Parameter(help="Column holding the query id; numbered when absent"),
    ] = None
    kind_column: Annotated[
        str | None,
        Field(serialization_alias="kind"),
        Parameter(help="Column holding the query kind; left unlabeled when absent"),
    ] = None
    expected_skill_column: Annotated[
        str | None,
        Field(serialization_alias="expected_skill"),
        Parameter(help="Column holding the skill that should be selected"),
    ] = None
    acceptable_skills_column: Annotated[
        str | None,
        Field(serialization_alias="acceptable_skills"),
        Parameter(help="Column holding neutral router or helper skills"),
    ] = None
    notes_column: Annotated[
        str | None,
        Field(serialization_alias="notes"),
        Parameter(help="Column holding per-query notes"),
    ] = None
    separator: Annotated[
        str | None,
        Parameter(help="Delimiter separating multiple skill names in a single cell"),
    ] = None
    id_prefix: Annotated[
        str | None,
        Parameter(
            help="Prefix prepended to generated query IDs to avoid "
            "collisions across imported files",
        ),
    ] = None

    def field_map(self) -> FieldMap:
        """Construct a validated FieldMap instance from non-None flag values."""
        return FieldMap.model_validate(self.overrides())


query_app = App(
    name="query",
    help="Synthesize, convert, and inspect evaluation query sets.",
    help_formatter=help_formatter(),
    group_parameters=OPTIONS,
    result_action="return_value",
    group=LOOP,
)
app.command(query_app)


def _resolve_draft_study_flags(
    study: StudyFlags | None,
    run_dir: Path | None,
    config: Path | None,
    dry_run: bool,
    out: Path | None = None,
) -> StudyFlags:
    """Resolve study flags with appropriate working directory and preview query defaults."""
    resolved = study or StudyFlags()
    if out is not None:
        resolved = resolved.model_copy(update={"queries": out})
    if run_dir is not None:
        resolved = _run_dir_study(run_dir, resolved)
    if config is None and resolved.workdir is None:
        resolved = resolved.model_copy(update={"workdir": Path.cwd()})
    if dry_run and resolved.queries is None and config is None:
        resolved = resolved.model_copy(update={"queries": Path("queries/draft-preview.json")})
    elif resolved.queries is None and config is None:
        found_q = find_existing_queries_path(resolved.skills)
        resolved = resolved.model_copy(update={"queries": found_q or DEFAULT_QUERIES_PATH})
    return resolved


def _build_draft_settings(
    config: Path | None,
    catalog: CatalogFlags | None,
    runtime: RuntimeFlags | None,
    study: StudyFlags,
    registry: RegistryFlags | None = None,
) -> RunConfig:
    """Construct draft run settings defaulting catalog mode to ALL when unspecified."""
    settings = build_config(
        config,
        catalog=catalog,
        runtime=runtime,
        study=study,
        registry=registry,
        required=DRAFT_REQUIRED,
    )
    if not _asks_for_a_mode(config, catalog):
        settings = settings.with_overrides(catalog={"mode": CatalogMode.ALL})
    return settings


def _render_query_view(
    console: Console,
    query_set: QuerySet,
    queries_path: Path,
    *,
    skills: Path | None = None,
    agent: str | None = None,
    global_scope: bool = False,
    show_leaks: bool = False,
    show_citations: bool = False,
) -> int:
    """Render query sets as a table with lexical difficulty ranks and leak detection."""
    settings = RuntimeSettings(agent=agent) if agent is not None else RuntimeSettings()
    driver = build_runtime(settings)
    found, _roots, discovered = resolve_corpus(
        driver,
        Path.cwd(),
        skills,
        agent=agent,
        global_scope=global_scope,
    )
    if discovered is not None:
        print_discovery(console, discovered)
    if not found:
        if skills is not None:
            raise _no_skills(skills, global_scope=global_scope)
        ranks = {}
        flags = None
    else:
        ranks = lexical_ranks(query_set.queries, found)
        flags = leaks(query_set.queries, found, background=found) if show_leaks else None

    trail = None
    if show_citations:
        trail_path = citations_path(queries_path)
        if not trail_path.exists():
            msg = (
                f"no citation trail found at {trail_path}. Citations are only "
                "available for synthetic query sets generated by `reach query` "
                "or `reach eval`."
            )
            raise ValueError(msg)
        trail = {
            (citation.skill, citation.text): citation.citation
            for citation in read_citations(trail_path).root
        }

    print_query_view(
        console,
        query_set,
        ranks=ranks,
        flags=flags,
        citations=trail,
    )
    return 0


def _resolve_query_source_file(
    target: str | Path | None,
    study: StudyFlags | None,
    *,
    show_leaks: bool,
    show_citations: bool,
    out: Path | None,
    draft_only: bool,
) -> Path | None:
    """Determine existing query set source file if target is a file or inspection is requested."""
    if draft_only:
        return None
    if target is not None:
        target_path = Path(target) if not isinstance(target, Path) else target
        if target_path.is_file():
            return target_path
    if (
        study is not None
        and study.queries is not None
        and study.queries.is_file()
        and (
            show_leaks
            or show_citations
            or (out is not None and study.skills is None and target is None)
        )
    ):
        return study.queries
    return None


def _resolve_query_target_format(format_opt: str | None, out: Path | None) -> str | None:
    """Determine serialization format for query dataset output."""
    if format_opt is not None:
        return format_opt
    if out is None:
        return None
    suffix = out.suffix.lower()
    if suffix in {".jsonl", ".csv", ".json"}:
        return suffix.lstrip(".")
    return infer_format(out).value


def _load_or_import_query_set(
    source_file: Path,
    study: StudyFlags | None,
    mapping: MapFlags | None,
    notes: str,
    format_opt: str | None,
) -> QuerySet:
    """Load query set directly or import from tabular CSV/JSONL with custom mapping."""
    catalog_id = (study.catalog if study and study.catalog else None) or "all"
    has_mapping = mapping is not None and any(v is not None for v in mapping.overrides().values())
    is_tabular = source_file.suffix.lower() in {".csv", ".jsonl"}
    has_import_options = has_mapping or bool(notes) or bool(study and study.catalog)
    if is_tabular and has_import_options:
        exchange_fmt = (
            Exchange(format_opt) if format_opt in {"csv", "jsonl"} else infer_format(source_file)
        )
        return import_query_set(
            source_file.read_text(encoding="utf-8"),
            exchange_fmt,
            catalog_id=catalog_id,
            mapping=mapping.field_map() if mapping is not None else None,
            notes=notes,
            source=str(source_file),
        )
    return load_query_set(source_file, catalog_id=catalog_id)


def _handle_existing_query_source(
    console: Console,
    source_file: Path,
    *,
    out: Path | None,
    format_opt: str | None,
    study: StudyFlags | None,
    mapping: MapFlags | None,
    notes: str,
    agent: str | None,
    runtime: RuntimeFlags | None,
    global_scope: bool,
    show_leaks: bool,
    show_citations: bool,
) -> int:
    """Process, convert, export, or inspect an existing query dataset."""
    if out is not None and out.exists():
        msg = (
            f"Query set already exists at {out}. Move it aside, "
            "or point --out somewhere else to avoid overwriting it."
        )
        raise ValueError(msg)

    query_set = _load_or_import_query_set(source_file, study, mapping, notes, format_opt)
    target_fmt = _resolve_query_target_format(format_opt, out)
    field_map = mapping.field_map() if mapping is not None else None

    if out is None:
        if target_fmt in {"csv", "jsonl"}:
            rendered = export_query_set(
                query_set,
                Exchange(target_fmt),
                mapping=field_map,
            )
            print(rendered, end="")
            return 0
        effective_skills = study.skills if study and study.skills else None
        effective_agent = agent or (runtime.agent if runtime and runtime.agent else None)
        return _render_query_view(
            console,
            query_set,
            source_file,
            skills=effective_skills,
            agent=effective_agent,
            global_scope=global_scope,
            show_leaks=show_leaks,
            show_citations=show_citations,
        )

    save_query_set(query_set, out, fmt=target_fmt, mapping=field_map)
    then_msg = (
        f"{len(query_set.queries)} queries for catalog '{query_set.catalog_id}'; "
        "review them, then probe with `reach eval`"
    )
    print_wrote(console, out, then=then_msg)
    return 0


def _resolve_target_and_study(
    console: Console,
    target: str | Path | None,
    study: StudyFlags | None,
) -> tuple[str | None, StudyFlags | None, int | None]:
    """Resolve target skill or corpus directory into study flags and skill target name."""
    if target is None:
        return None, study, None

    raw_target = str(target).strip()
    target_path = Path(raw_target)
    is_corpus_dir = False
    try:
        if (
            target_path.is_dir()
            and not (target_path / "SKILL.md").is_file()
            and next(target_path.glob("**/SKILL.md"), None) is not None
        ):
            is_corpus_dir = True
    except (OSError, PermissionError):
        pass

    if is_corpus_dir:
        return None, (study or StudyFlags()).model_copy(update={"skills": target_path}), None

    from reach.catalog import resolve_skill_target

    try:
        resolved = resolve_skill_target(
            raw_target,
            explicit_catalog=study.skills if study else None,
            command_name="query draft",
        )
        target_skill_name: str | None = None
        if resolved is not None:
            target_skill_name = resolved.skill_name
            if resolved.catalog_path and (study is None or study.skills is None):
                study = (study or StudyFlags()).model_copy(update={"skills": resolved.catalog_path})
        return target_skill_name, study, None
    except (FileNotFoundError, ValueError) as err:
        console.print(f"[red]Error:[/] {err}")
        return None, study, 2


def _resolve_missing_backfill(
    console: Console,
    destination: Path,
    skills_found: Sequence[Skill],
    existing_query_set: QuerySet,
    effective_generate: GenerateFlags,
    count: int | None,
    generate: GenerateFlags | None,
) -> tuple[GenerateFlags | None, int | None]:
    """Filter generation targets to unqueried skills when running in --missing mode."""
    covered_names = {
        q.expected_skill for q in existing_query_set.queries if q.expected_skill is not None
    }
    requested_targets = effective_generate.targets
    candidate_skills = (
        [s for s in skills_found if s.name in requested_targets]
        if requested_targets
        else list(skills_found)
    )
    missing_names = tuple(s.name for s in candidate_skills if s.name not in covered_names)
    if not missing_names:
        console.print(
            f"[green]✓[/] All {len(candidate_skills)} skill(s) already have queries in "
            f"[cyan]{destination}[/] ({len(existing_query_set.queries)} queries)."
        )
        return None, 0
    console.print(
        f"[dim]Backfilling[/] [bold]{len(missing_names)}[/] [dim]missing skill(s) into[/] "
        f"[cyan]{destination}[/] "
        f"[dim]({len(covered_names)}/{len(skills_found)} already covered)[/]"
    )
    updates: dict[str, object] = {"targets": missing_names}
    if existing_query_set.provenance is not None:
        prov = existing_query_set.provenance
        if count is None and prov.queries_per_target:
            updates["count"] = prov.queries_per_target
        if (generate is None or generate.top_rivals is None) and prov.rivals_in_view is not None:
            updates["top_rivals"] = prov.rivals_in_view
    return effective_generate.model_copy(update=updates), None


def _handle_draft_query_generation(
    console: Console,
    *,
    target: str | Path | None,
    count: int | None,
    out: Path | None,
    format_opt: str | None,
    study: StudyFlags | None,
    run_dir: Path | None,
    config: Path | None,
    catalog: CatalogFlags | None,
    runtime: RuntimeFlags | None,
    generate: GenerateFlags | None,
    registry: RegistryFlags | None = None,
    dry_run: bool = False,
    review: bool = False,
    force: bool = False,
    missing: bool = False,
) -> int:
    """Synthesize new synthetic benchmark queries for discovered skills."""
    target_skill_name, study, exit_code = _resolve_target_and_study(console, target, study)
    if exit_code is not None:
        return exit_code

    if target_skill_name is not None:
        existing_targets = generate.targets if generate is not None else ()
        if not existing_targets:
            generate = (generate or GenerateFlags()).model_copy(
                update={"targets": (target_skill_name,)}
            )

    if count is not None:
        generate = (generate or GenerateFlags()).model_copy(update={"count": count})

    effective_out = out
    if effective_out is None and format_opt == "jsonl":
        effective_out = Path(".reach/queries.jsonl")
    elif effective_out is None and format_opt == "csv":
        effective_out = Path(".reach/queries.csv")

    resolved_study = _resolve_draft_study_flags(study, run_dir, config, dry_run, out=effective_out)
    settings = _build_draft_settings(config, catalog, runtime, resolved_study, registry=registry)

    destination = settings.require_queries()
    existing_query_set: QuerySet | None = None
    if destination.exists():
        if missing:
            existing_query_set = load_query_set(destination)
        elif not force:
            msg = (
                f"Query set already exists at {destination}. Move it aside, "
                "use --missing to backfill missing skills, use --force to overwrite, "
                "or point --out / --queries somewhere else."
            )
            raise ValueError(msg)
    skills_found, _roots, _found = _corpus(console, build_runtime(settings.runtime), settings)
    effective_generate = generate or GenerateFlags()
    if missing and existing_query_set is not None:
        updated_generate, exit_code = _resolve_missing_backfill(
            console,
            destination,
            skills_found,
            existing_query_set,
            effective_generate,
            count,
            generate,
        )
        if exit_code is not None:
            return exit_code
        if updated_generate is not None:
            effective_generate = updated_generate

    return _draft_query_set(
        console,
        settings,
        skills_found,
        effective_generate,
        dry_run=dry_run,
        review=review,
        existing_query_set=existing_query_set,
    )


@query_app.default
def _query(
    target: Annotated[
        str | Path | None,
        Parameter(
            help="Skill corpus to draft queries for, or existing query set to convert/inspect",
        ),
    ] = None,
    *,
    out: Annotated[
        Path | None,
        Parameter(
            name=["--out", "-o"],
            help=(
                "Where to write the query set (default: .reach/queries.json or "
                ".reach/queries.jsonl)"
            ),
        ),
    ] = None,
    format: Annotated[
        Literal["json", "jsonl", "csv"] | None,
        Parameter(
            name=["--format", "-f"],
            help=(
                "Output format: 'json', 'jsonl', or 'csv' (inferred from --out "
                "file extension if provided)"
            ),
        ),
    ] = None,
    count: Annotated[
        int | None,
        POSITIVE_INT,
        Parameter(
            name=["--count"],
            group=GENERATE_GROUP,
            help="Number of queries to draft per target skill (default: 3)",
        ),
    ] = None,
    config: ConfigFlag = None,
    catalog: Annotated[CatalogFlags | None, Parameter(group=CATALOG_GROUP)] = None,
    runtime: Annotated[RuntimeFlags | None, Parameter(group=RUNTIME_GROUP)] = None,
    study: Annotated[StudyFlags | None, Parameter(group=STUDY_GROUP)] = None,
    registry: Annotated[RegistryFlags | None, Parameter(group=REGISTRY_GROUP)] = None,
    run_dir: Annotated[
        Path | None,
        Parameter(
            group=STUDY_GROUP,
            help="Directory to contain default --queries and --workdir paths "
            "(explicit flags override this)",
        ),
    ] = None,
    generate: Annotated[GenerateFlags | None, Parameter(group=GENERATE_GROUP)] = None,
    dry_run: Annotated[
        bool,
        SWITCH,
        Parameter(
            help="Dry run without writing outputs or executing models (reports would-be plan)",
        ),
    ] = False,
    mapping: Annotated[MapFlags | None, Parameter(group=MAPPING_GROUP)] = None,
    notes: Annotated[
        str,
        Parameter(
            help="Provenance notes or reviewer comments describing this query set",
        ),
    ] = "",
    show_leaks: Annotated[
        bool,
        SWITCH,
        Parameter(
            name="--leaks",
            help="Add a column flagging potential target skill name leakage in query text",
        ),
    ] = False,
    show_citations: Annotated[
        bool,
        SWITCH,
        Parameter(
            name="--citations",
            help="Add a column quoting each query's grounding passage",
        ),
    ] = False,
    quiet: Quiet = False,
    agent: Annotated[
        AgentName | None,
        Parameter(
            name="--agent",
            show_choices=False,
            help=agent_help_text(
                "Agent runtime used for skill corpus discovery (no probes executed)",
            ),
        ),
    ] = None,
    global_: Global = False,
    review: Annotated[
        bool,
        SWITCH,
        Parameter(
            name="--review",
            help="Launch interactive browser review for drafted queries",
        ),
    ] = False,
    force: Annotated[
        bool,
        SWITCH,
        Parameter(
            name="--force",
            help="Overwrite destination query set file if it already exists",
        ),
    ] = False,
    missing: Annotated[
        bool,
        SWITCH,
        Parameter(
            name="--missing",
            help="Backfill queries only for skills missing from an existing destination query set",
        ),
    ] = False,
    draft_only: Annotated[bool, Parameter(show=False)] = False,
) -> int:
    """Synthesize benchmark queries for skills, convert formats, or inspect query datasets."""
    console = build_console(quiet=quiet)

    source_file = _resolve_query_source_file(
        target,
        study,
        show_leaks=show_leaks,
        show_citations=show_citations,
        out=out,
        draft_only=draft_only or missing,
    )

    if source_file is not None:
        return _handle_existing_query_source(
            console,
            source_file,
            out=out,
            format_opt=format,
            study=study,
            mapping=mapping,
            notes=notes,
            agent=agent,
            runtime=runtime,
            global_scope=global_,
            show_leaks=show_leaks,
            show_citations=show_citations,
        )

    return _handle_draft_query_generation(
        console,
        target=target,
        count=count,
        out=out,
        format_opt=format,
        study=study,
        run_dir=run_dir,
        config=config,
        catalog=catalog,
        runtime=runtime,
        generate=generate,
        registry=registry,
        dry_run=dry_run,
        review=review,
        force=force,
        missing=missing,
    )


@query_app.command(name="draft", group=QUERY_OPERATIONS)
def _query_draft(
    target: Annotated[
        str | Path | None,
        Parameter(
            help="Skill corpus directory or target skill to draft queries for",
        ),
    ] = None,
    *,
    out: Annotated[
        Path | None,
        Parameter(
            name=["--out", "-o"],
            help=(
                "Where to write the drafted query set; defaults to --queries or .reach/queries.json"
            ),
        ),
    ] = None,
    format: Annotated[
        Literal["json", "jsonl", "csv"] | None,
        Parameter(
            name=["--format", "-f"],
            help="Output format: 'json', 'jsonl', or 'csv' (inferred from --out if omitted)",
        ),
    ] = None,
    count: Annotated[
        int | None,
        POSITIVE_INT,
        Parameter(
            name=["--count"],
            group=GENERATE_GROUP,
            help="Number of queries to draft per target skill (default: 3)",
        ),
    ] = None,
    config: ConfigFlag = None,
    catalog: Annotated[CatalogFlags | None, Parameter(group=CATALOG_GROUP)] = None,
    runtime: Annotated[RuntimeFlags | None, Parameter(group=RUNTIME_GROUP)] = None,
    study: Annotated[StudyFlags | None, Parameter(group=STUDY_GROUP)] = None,
    registry: Annotated[RegistryFlags | None, Parameter(group=REGISTRY_GROUP)] = None,
    run_dir: Annotated[
        Path | None,
        Parameter(
            group=STUDY_GROUP,
            help="Directory to contain default --queries and --workdir paths "
            "(explicit flags override this)",
        ),
    ] = None,
    generate: Annotated[GenerateFlags | None, Parameter(group=GENERATE_GROUP)] = None,
    dry_run: Annotated[
        bool,
        SWITCH,
        Parameter(
            help="Preview what queries would be drafted without making model calls",
        ),
    ] = False,
    review: Annotated[
        bool,
        SWITCH,
        Parameter(
            name="--review",
            help="Launch interactive browser review for drafted queries",
        ),
    ] = False,
    force: Annotated[
        bool,
        SWITCH,
        Parameter(
            name="--force",
            help="Overwrite destination query set file if it already exists",
        ),
    ] = False,
    missing: Annotated[
        bool,
        SWITCH,
        Parameter(
            name="--missing",
            help="Backfill queries only for skills missing from an existing destination query set",
        ),
    ] = False,
    quiet: Quiet = False,
) -> int:
    """Generate synthetic query set for target catalog from skill markdown bodies."""
    return _query(
        target=target,
        out=out,
        format=format,
        count=count,
        config=config,
        catalog=catalog,
        runtime=runtime,
        study=study,
        registry=registry,
        run_dir=run_dir,
        generate=generate,
        dry_run=dry_run,
        review=review,
        force=force,
        missing=missing,
        quiet=quiet,
        draft_only=True,
    )


@query_app.command(name="view", group=QUERY_OPERATIONS)
def _query_view(
    queries: Annotated[Path, Parameter(help="Query set to render (JSON)")],
    *,
    skills: Annotated[
        Path | None,
        Parameter(
            help="Skill corpus to rank against; discovered from the runtime if omitted",
        ),
    ] = None,
    agent: Annotated[
        AgentName | None,
        Parameter(
            name="--agent",
            show_choices=False,
            help=agent_help_text(
                "Agent runtime used for skill corpus discovery (no probes executed)",
            ),
        ),
    ] = None,
    global_: Global = False,
    show_leaks: Annotated[
        bool,
        SWITCH,
        Parameter(
            name="--leaks",
            help="Add a column flagging potential target skill name leakage in query text",
        ),
    ] = False,
    show_citations: Annotated[
        bool,
        SWITCH,
        Parameter(
            name="--citations",
            help="Add a column quoting each query's grounding passage",
        ),
    ] = False,
) -> int:
    """Render query sets as a table with lexical difficulty ranks and leak detection."""
    query_set = load_query_set(queries)
    return _render_query_view(
        build_console(),
        query_set,
        queries,
        skills=skills,
        agent=agent,
        global_scope=global_,
        show_leaks=show_leaks,
        show_citations=show_citations,
    )
