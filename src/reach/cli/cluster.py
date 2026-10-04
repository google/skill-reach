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

"""Partition skills into modular subagent catalogs using modularity optimization."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated

from cyclopts import Parameter

from reach.cluster import cluster_skills
from reach.config import (
    RegistrySettings,
    RunConfig,
    RuntimeSettings,
    _discover_config_path,
    resolve_sub_settings,
)
from reach.runtime import build_runtime
from reach.views import build_console, print_cluster, render_cluster

from .app import LOOP, app
from .discovery import _corpus, _no_skills
from .flags import (
    NON_NEGATIVE,
    POSITIVE_INT,
    REGISTRY_GROUP,
    AgentName,
    ConfigFlag,
    Format,
    Global,
    RegistryFlags,
    agent_help_text,
)

if TYPE_CHECKING:
    pass


@app.command(name="cluster", group=LOOP)
def _cluster(
    skills: Annotated[
        Path | None,
        Parameter(
            help="Path to the skill directory or corpus to partition (discovered if omitted)",
        ),
    ] = None,
    *,
    resolution: Annotated[
        float,
        NON_NEGATIVE,
        Parameter(
            help="Resolution parameter: higher values yield smaller clusters",
        ),
    ] = 1.0,
    target_size: Annotated[
        int | None,
        POSITIVE_INT,
        Parameter(
            help="Target maximum skills per cluster",
        ),
    ] = None,
    max_clusters: Annotated[
        int | None,
        POSITIVE_INT,
        Parameter(
            help="Maximum number of clusters",
        ),
    ] = None,
    agent: Annotated[
        AgentName | None,
        Parameter(
            help=agent_help_text("Agent runtime to query for installed skill locations"),
        ),
    ] = None,
    global_: Global = False,
    registry: Annotated[RegistryFlags | None, Parameter(group=REGISTRY_GROUP)] = None,
    format: Annotated[Format, Parameter(help="Output format: text, json, csv")] = "text",
    config: ConfigFlag = None,
) -> int:
    """Partition skill catalogs into cohesive subagent scopes to prevent routing decay."""
    console = build_console()
    driver = build_runtime(RuntimeSettings(agent=agent) if agent is not None else RuntimeSettings())
    resolved_config, _ = _discover_config_path(config)
    run_config: RunConfig | None = None
    if resolved_config is not None:
        run_config = RunConfig.from_toml(resolved_config)

    effective_config = run_config or RunConfig()
    effective_registry = resolve_sub_settings(
        RegistrySettings,
        effective_config.registry,
        **(registry.overrides() if registry is not None else {}),
    )
    has_cli_registry = registry is not None and (registry.registry or registry.project is not None)
    if skills is not None and not has_cli_registry:
        effective_registry = effective_registry.model_copy(
            update={"registry": False, "project": None}
        )
    eff_skills = (
        skills
        if skills is not None
        else (None if has_cli_registry else effective_config.study.skills)
    )
    run_config = effective_config.model_copy(
        update={
            "study": effective_config.study.model_copy(update={"skills": eff_skills}),
            "registry": effective_registry,
        }
    )
    found, _roots, _discovered = _corpus(
        console,
        driver,
        settings=run_config,
        global_scope=global_,
        agent=agent,
    )

    if not found:
        raise _no_skills(skills, global_scope=global_)

    partition = cluster_skills(
        found,
        resolution=resolution,
        max_clusters=max_clusters,
        target_size=target_size,
    )

    if format == "text":
        print_cluster(console, partition)
    else:
        rendered = render_cluster(partition, format)
        print(rendered)

    return 0
