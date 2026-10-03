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

"""Render recorded evaluation run artifacts to terminal text or HTML formats."""

from __future__ import annotations

import webbrowser
from pathlib import Path
from typing import Annotated, Literal

from cyclopts import Parameter
from pydantic import TypeAdapter
from pydantic import ValidationError as PydanticValidationError

from reach.artifact import ARTIFACT_SUFFIX, Artifact
from reach.sweep import ScalingStudy
from reach.view import render_view
from reach.views import (
    Console,
    build_console,
    print_query_records,
    print_scorecard,
    print_sweep,
    print_wrote,
    render_sweep_csv,
    render_sweep_json,
)

from .app import LOOP, app
from .flags import SWITCH, SliceFlags, Verbose

_VIEW_PAYLOAD_ADAPTER: TypeAdapter[Artifact | ScalingStudy] = TypeAdapter(Artifact | ScalingStudy)


def _handle_browser_view(
    console: Console,
    recorded: Artifact,
    out: Path | None,
) -> int:
    """Save HTML view and open in default web browser."""
    target_out = out
    if target_out is None:
        reach_dir = Path(".reach")
        reach_dir.mkdir(parents=True, exist_ok=True)
        target_out = reach_dir / "report.html"

    target_out.parent.mkdir(parents=True, exist_ok=True)
    html_content = render_view(recorded, "html")
    target_out.write_text(html_content, encoding="utf-8")
    console.print(f"Opening [bold]{target_out}[/bold] in your default browser...")
    webbrowser.open(target_out.resolve().as_uri())
    return 0


def _serialize_eval_format(recorded: Artifact, format_opt: str) -> str | None:
    """Serialize an evaluation artifact for non-text formats, or return None for text."""
    match format_opt:
        case "html":
            return render_view(recorded, "html")
        case "json":
            return recorded.model_dump_json(indent=2)
        case "jsonl":
            return "".join(query.model_dump_json() + "\n" for query in recorded.queries)
        case "csv":
            msg = (
                "CSV format is only supported for sweep artifacts; "
                "choose 'text', 'html', 'json', or 'jsonl' for evaluation artifacts."
            )
            raise ValueError(msg)
        case _:
            return None


def _handle_file_view(
    console: Console,
    recorded: Artifact,
    out: Path,
    format_opt: str,
    *,
    verbose: bool,
    show_queries: bool,
) -> int:
    """Render and write artifact output to a file."""
    out.parent.mkdir(parents=True, exist_ok=True)
    content = _serialize_eval_format(recorded, format_opt)
    if content is None:
        file_console = build_console(record=True)
        print_scorecard(file_console, recorded, verbose=verbose)
        if show_queries:
            print_query_records(file_console, recorded)
        content = file_console.export_text()
    out.write_text(content, encoding="utf-8")
    print_wrote(console, out)
    return 0


def _handle_stdout_view(
    console: Console,
    recorded: Artifact,
    format_opt: str,
    *,
    verbose: bool,
    show_queries: bool,
) -> int:
    """Render artifact output directly to stdout."""
    content = _serialize_eval_format(recorded, format_opt)
    if content is not None:
        print(content, end="" if format_opt == "jsonl" else "\n")
        return 0
    print_scorecard(console, recorded, verbose=verbose)
    if show_queries:
        print_query_records(console, recorded)
    return 0


def _handle_sweep_view(
    console: Console,
    study: ScalingStudy,
    out: Path | None,
    format_opt: str,
    *,
    open_browser: bool,
) -> int:
    """Render and display or write scaling sweep study results."""
    if open_browser or format_opt == "html":
        msg = "HTML format is not supported for sweep artifacts; choose 'text', 'json', or 'csv'."
        raise ValueError(msg)

    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        if format_opt == "csv" or out.suffix == ".csv":
            content = render_sweep_csv(study)
        elif format_opt in ("json", "jsonl") or out.suffix == ".json":
            content = render_sweep_json(study)
        else:
            file_console = build_console(record=True)
            print_sweep(file_console, study)
            content = file_console.export_text()
        out.write_text(content, encoding="utf-8")
        print_wrote(console, out)
        return 0

    match format_opt:
        case "csv":
            print(render_sweep_csv(study))
        case "json" | "jsonl":
            print(render_sweep_json(study))
        case _:
            print_sweep(console, study)
    return 0


def _resolve_view_artifact(artifact: Path | None) -> Path:
    """Resolve target artifact path from explicit parameter or default locations."""
    target_artifact = artifact or Path(".reach/eval.json")
    if artifact is None and not target_artifact.is_file():
        alt_artifact = Path(".reach/queries.json.artifact.json")
        sweep_artifact = Path(".reach/sweep.json")
        if alt_artifact.is_file():
            target_artifact = alt_artifact
        elif sweep_artifact.is_file():
            target_artifact = sweep_artifact
    if not target_artifact.is_file():
        msg = (
            f"No evaluation artifact found at {target_artifact}.\n\n"
            "Run 'reach eval' first, or pass an artifact path: 'reach view <artifact.json>'"
        )
        raise ValueError(msg)
    return target_artifact


@app.command(name="view", group=LOOP)
def _view(
    artifact: Annotated[
        Path | None,
        Parameter(
            help="Path to the evaluation artifact JSON file (default: .reach/eval.json)",
        ),
    ] = None,
    *,
    out: Annotated[
        Path | None,
        Parameter(
            alias="-o",
            help="Where to write the rendered report; defaults to stdout",
        ),
    ] = None,
    show_queries: Annotated[
        bool,
        SWITCH,
        Parameter(
            help="Display individual scored query records below the scorecard",
        ),
    ] = False,
    format: Annotated[
        Literal["text", "html", "json", "jsonl", "csv"],
        Parameter(
            help="Output format: 'text' (terminal scorecard), "
            "'html' (standalone interactive HTML report), "
            "'json' (raw artifact JSON), "
            "'jsonl' (scored query records as JSON lines), or "
            "'csv' (sweep summary CSV)",
        ),
    ] = "text",
    slice_flags: SliceFlags | None = None,
    open_browser: Annotated[
        bool,
        SWITCH,
        Parameter(
            name="--open",
            alias="-O",
            help="Open the rendered HTML report directly in the default web browser",
        ),
    ] = False,
    verbose: Verbose = False,
) -> int:
    """Read back a recorded run and render what it measured."""
    from reach.diff import load_arm
    from reach.run import sidecar_path

    eff_slice = slice_flags or SliceFlags()
    target_artifact = _resolve_view_artifact(artifact)

    console = build_console()
    try:
        raw_text = target_artifact.read_text(encoding="utf-8")
    except OSError as err:
        msg = f"Cannot read artifact file at {target_artifact}: {err}"
        raise ValueError(msg) from err

    has_sidecar = sidecar_path(target_artifact).exists()
    try:
        if has_sidecar:
            parsed: Artifact | ScalingStudy = load_arm(
                target_artifact,
                queries=eff_slice.queries,
                filter_skill=eff_slice.filter_skill,
                filter_id=eff_slice.filter_id,
            ).artifact
        else:
            parsed = _VIEW_PAYLOAD_ADAPTER.validate_json(raw_text)
    except PydanticValidationError as sweep_error:
        msg = (
            f"Cannot read artifact at {target_artifact}: file is neither a valid "
            f"evaluation artifact (written by `reach eval` as <results>{ARTIFACT_SUFFIX}) "
            f"nor a valid scaling sweep study (written by `reach sweep`)."
        )
        raise ValueError(msg) from sweep_error

    if isinstance(parsed, ScalingStudy):
        if eff_slice.active:
            msg = (
                "Slicing flags (--filter-skill, --filter-id, --queries) "
                "are not supported for sweep artifacts."
            )
            raise ValueError(msg)
        return _handle_sweep_view(console, parsed, out, format, open_browser=open_browser)

    recorded = (
        load_arm(
            target_artifact,
            queries=eff_slice.queries,
            filter_skill=eff_slice.filter_skill,
            filter_id=eff_slice.filter_id,
        ).artifact
        if eff_slice.active and not has_sidecar
        else parsed
    )

    if open_browser:
        return _handle_browser_view(console, recorded, out)

    if out is not None:
        return _handle_file_view(
            console, recorded, out, format, verbose=verbose, show_queries=show_queries
        )

    return _handle_stdout_view(
        console, recorded, format, verbose=verbose, show_queries=show_queries
    )
