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

"""Compare two evaluation arms across a varied factor for significant deltas."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from cyclopts import Parameter

from .app import LOOP, app
from .flags import NON_NEGATIVE, RATE, ConfigFlag, Factor, Format, SliceFlags


@app.command(name="diff", group=LOOP)
def _diff(
    control: Annotated[
        Path,
        Parameter(help="Recorded results for the baseline/control arm"),
    ],
    treatment: Annotated[
        Path,
        Parameter(help="Recorded results for the treatment/experimental arm"),
    ],
    *,
    vary: Annotated[
        Factor,
        Parameter(
            help="The experimental factor varied between arms (description, rival, or scope)",
        ),
    ],
    queries_root: Annotated[
        Path | None,
        Parameter(
            help="Root directory containing query sets if their paths have "
            "changed since evaluation",
        ),
    ] = None,
    control_corpus: Annotated[
        Path | None,
        Parameter(
            help="The corpus the control arm was probed against, if it has moved",
        ),
    ] = None,
    treatment_corpus: Annotated[
        Path | None,
        Parameter(
            help="The corpus the treatment arm was probed against, if it has moved",
        ),
    ] = None,
    control_label: Annotated[
        str | None,
        Parameter(
            help="Custom display label for the control arm (defaults to filename)",
        ),
    ] = None,
    treatment_label: Annotated[
        str | None,
        Parameter(
            help="Custom display label for the treatment arm (defaults to filename)",
        ),
    ] = None,
    slice_flags: SliceFlags | None = None,
    confidence: Annotated[
        float | None,
        RATE,
        Parameter(
            help="Confidence level used to estimate the noise floor (default: from reach.toml)",
        ),
    ] = None,
    noise_inflation: Annotated[
        float | None,
        NON_NEGATIVE,
        Parameter(
            help=(
                "Multiplier to inflate the estimated noise floor for over-dispersion "
                "(default: from reach.toml)"
            ),
        ),
    ] = None,
    format: Annotated[Format, Parameter(help="How to render the comparison")] = "text",
    config: ConfigFlag = None,
) -> int:
    """Change one factor, hold the queries fixed, and report the delta."""
    from reach.config import DiffSettings, RunConfig
    from reach.diff import survey_runs
    from reach.views.diff import render_diff, render_survey

    eff_slice = slice_flags or SliceFlags()
    run_config = RunConfig.from_toml(config) if config is not None else None
    eff_settings = RunConfig.resolve(
        DiffSettings,
        config=run_config,
        confidence=confidence,
        noise_inflation=noise_inflation,
    )

    surveyed = survey_runs(
        control,
        treatment,
        vary,
        queries_root=queries_root,
        control_corpus=control_corpus,
        treatment_corpus=treatment_corpus,
        control_label=control_label,
        treatment_label=treatment_label,
        queries=eff_slice.queries,
        filter_skill=eff_slice.filter_skill,
        filter_id=eff_slice.filter_id,
    )
    if not surveyed.comparable:
        raise ValueError(render_survey(surveyed))
    print(
        render_diff(
            surveyed.cross(settings=eff_settings),
            format,
        )
    )
    return 0
