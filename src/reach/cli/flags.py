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

"""Define CLI parameter groups, models, and configuration builder mappings."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Any, Literal, Self, override

from cyclopts import Group, Parameter, validators
from pydantic import BaseModel, ConfigDict, Field
from pydantic import ValidationError as PydanticValidationError

from reach.config import DEFAULT_GEMINI_MODEL, QuerySettings, RunConfig, StudySettings
from reach.diff import VaryFactor
from reach.generate import GeneratorArm
from reach.lint import Severity
from reach.models import CatalogMode
from reach.runtime import options_model

#: Supported output formats for CLI display.
type Format = Literal["csv", "json", "jsonl", "text"]

#: Supported comparison factors for diff analysis.
Factor = VaryFactor
type Vary = VaryFactor

#: Supported agent runtime identifiers.
type AgentName = Literal[
    "antigravity-cli",
    "antigravity-sdk",
    "claude-code",
    "fake",
    "goose",
    "keyword",
    "pi",
]

#: Standard boolean switch configuration for CLI parameters.
SWITCH = Parameter(negative=(), show_default=False)

#: Universal confirmation bypass switch for CI and automated scripts.
YesFlag = Annotated[
    bool,
    SWITCH,
    Parameter(
        name=["--yes", "-y"],
        help="Bypass interactive safety confirmation prompts (recommended for CI and scripts)",
    ),
]

#: Standard repeatable list parameter configuration.
LIST = Parameter(negative=(), show_default=False)

#: Rate parameter constraint requiring float in [0.0, 1.0].
RATE = Parameter(validator=validators.Number(gte=0.0, lte=1.0))

#: Non-negative score constraint requiring float >= 0.0.
NON_NEGATIVE = Parameter(validator=validators.Number(gte=0.0))

#: Positive count constraint requiring int >= 1.
POSITIVE_INT = Parameter(validator=validators.Number(gte=1))


def build_rule_overrides(
    ignore: tuple[str, ...],
    warn: tuple[str, ...],
    error: tuple[str, ...],
) -> dict[str, Severity]:
    """Map rule override CLI lists to their configured Severity enum values."""
    overrides: dict[str, Severity] = {}
    for rule_name in ignore:
        overrides[rule_name] = Severity.IGNORE
    for rule_name in warn:
        overrides[rule_name] = Severity.WARN
    for rule_name in error:
        overrides[rule_name] = Severity.ERROR
    return overrides


#: Flag parameter for muting standard console output.
Quiet = Annotated[
    bool,
    SWITCH,
    Parameter(
        name=["--quiet", "-q"],
        help="Mute the terminal view; results and errors still print",
    ),
]

#: Flag parameter for enabling verbose digest output.
Verbose = Annotated[
    bool,
    SWITCH,
    Parameter(
        help="Display full hexadecimal hash digests alongside badges",
    ),
]

#: Flag parameter for inspecting skills from user's global configuration.
Global = Annotated[
    bool,
    SWITCH,
    Parameter(
        name=["--global", "-g"],
        help="Discover and inspect skills from user's global configuration (~/)",
    ),
]

#: Parameter for Google Cloud project ID.
ProjectFlag = Annotated[
    str | None,
    Parameter(
        name=["--project", "-p"],
        help="Google Cloud project ID hosting the Agent Registry",
    ),
]

#: Parameter for Agent Registry location.
LocationFlag = Annotated[
    str | None,
    Parameter(
        name=["--location"],
        help="Agent Registry location (default: 'global')",
    ),
]

#: Parameter for filtering skills by publisher.
PublisherFlag = Annotated[
    str | None,
    Parameter(
        name=["--publisher"],
        help="Filter skills by publisher identifier (e.g. 'cloud.google.com')",
    ),
]

#: Parameter for targeting Agent Registry.
RegistryFlag = Annotated[
    bool,
    SWITCH,
    Parameter(
        name=["--registry"],
        help="Target the Google Cloud Agent Registry instead of local workspace",
    ),
]

#: Parameter for bypassing cache TTL.
FreshFlag = Annotated[
    bool,
    SWITCH,
    Parameter(
        name=["--fresh"],
        help="Bypass cached metadata and fetch latest revision pointers from registry",
    ),
]

#: Parameter for disabling cache.
NoCacheFlag = Annotated[
    bool,
    SWITCH,
    Parameter(
        name=["--no-cache"],
        help="Run without reading or persisting local disk cache",
    ),
]

#: Flag parameter for enabling/disabling early stopping in scaling sweeps.
EarlyStopFlag = Annotated[
    bool,
    Parameter(
        name="--early-stop",
        negative="--no-early-stop",
        help=(
            "Terminate scaling sweep early if F1 95% CI upper bound drops below minimum SLA "
            "(default: True)"
        ),
    ),
]

#: Parameter for reach.toml configuration file path.
ConfigFlag = Annotated[
    Path | None,
    Parameter(
        name=["--config", "-c"],
        help="Path to reach.toml configuration file",
    ),
]


class Flags(BaseModel):
    """Base model for CLI flag sections supporting partial configuration extraction."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    def overrides(self) -> dict[str, Any]:
        """Return non-None model fields formatted as configuration overrides."""
        return self.model_dump(exclude_none=True, by_alias=True)


FLAT = Parameter(name="*")


@FLAT
class CatalogFlags(Flags):
    """CLI parameter flags configuring catalog composition rules."""

    mode: Annotated[
        CatalogMode | None,
        Parameter(
            help="How catalogs are assembled from the corpus (default: 'all' for "
            "whole catalog, 'neighborhood' in quick mode)",
        ),
    ] = None
    catalog_size: Annotated[
        int | None,
        POSITIVE_INT,
        Field(default=None, ge=1, serialization_alias="size"),
        Parameter(
            help="Skills per neighborhood catalog",
        ),
    ] = None
    rivals: Annotated[
        int | None,
        NON_NEGATIVE,
        Field(default=None, ge=0),
        Parameter(
            help="Number of top-ranked rival skills per neighborhood catalog",
        ),
    ] = None
    seed: Annotated[
        int | None,
        Parameter(help="Random seed for catalog filler selection"),
    ] = None


@FLAT
class RegistryFlags(Flags):
    """CLI parameter flags configuring Google Cloud Agent Registry interactions."""

    project: ProjectFlag = None
    location: LocationFlag = None
    publisher: PublisherFlag = None
    registry: RegistryFlag = False
    fresh: FreshFlag = False
    no_cache: NoCacheFlag = False

    @override
    def overrides(self) -> dict[str, Any]:
        """Return non-None and non-False model fields formatted as configuration overrides."""
        return {
            k: v
            for k, v in self.model_dump(exclude_none=True, by_alias=True).items()
            if v is not False
        }


@FLAT
class SliceFlags(Flags):
    """CLI parameter flags configuring post-run query and skill sub-slicing."""

    queries: Annotated[
        Path | None,
        Parameter(
            name=["--queries", "-q"],
            help="Subset query set file used to slice recorded evaluation runs",
        ),
    ] = None
    filter_skill: Annotated[
        tuple[str, ...],
        Parameter(
            name="--filter-skill",
            help="Glob pattern(s) matching target skill names to slice recorded runs",
        ),
    ] = ()
    filter_id: Annotated[
        tuple[str, ...],
        Parameter(
            name="--filter-id",
            help="Glob pattern(s) matching query IDs to slice recorded runs",
        ),
    ] = ()

    @property
    def active(self) -> bool:
        """Return True if any sub-slicing flag was supplied."""
        return bool(self.queries is not None or self.filter_skill or self.filter_id)


@FLAT
class RuleOverrideFlags(Flags):
    """CLI parameter flags configuring static lint rule severity overrides."""

    ignore: Annotated[
        tuple[str, ...],
        LIST,
        Parameter(
            name="--ignore",
            help="Disable specific lint rule(s) (repeatable)",
        ),
    ] = ()
    error: Annotated[
        tuple[str, ...],
        LIST,
        Parameter(
            name="--error",
            help="Treat specific lint rule(s) as error (repeatable)",
        ),
    ] = ()
    warn: Annotated[
        tuple[str, ...],
        LIST,
        Parameter(
            name="--warn",
            help="Treat specific lint rule(s) as warning (repeatable)",
        ),
    ] = ()

    def to_overrides(self) -> dict[str, Severity]:
        """Convert configured rule lists into severity override dictionary."""
        return build_rule_overrides(self.ignore, self.warn, self.error)


def _coerce_opt_value(raw: str) -> object:
    """Attempt JSON decoding of option string, falling back to raw string."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def parse_agent_options(pairs: Sequence[str]) -> dict[str, object]:
    """Parse key=value string pairs into an agent options dictionary."""
    parsed: dict[str, object] = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep:
            msg = f"invalid agent option {pair!r}: expected key=value format"
            raise ValueError(
                msg,
            )
        parsed[key] = _coerce_opt_value(value)
    return parsed


def agent_help_text(prefix: str = "Which agent runtime to drive") -> str:
    """Format dynamic runtime agent choices excluding internal test doubles."""
    from reach.runtime import FAKE_AGENT, known_agents

    public = [a for a in known_agents() if a != FAKE_AGENT]
    return f"{prefix} ({', '.join(public)})"


@FLAT
class RuntimeFlags(Flags):
    """CLI parameter flags configuring the target agent runtime."""

    agent: Annotated[
        AgentName | None,
        Parameter(
            name="--agent",
            show_choices=False,
            help=agent_help_text(),
        ),
    ] = None

    model: Annotated[
        str | None,
        Parameter(
            name=["--model", "-m"],
            help="Target model identifier",
        ),
    ] = None
    effort: Annotated[
        str | None,
        Parameter(
            name=["--effort", "-e"],
            help="Reasoning effort level (e.g. low, medium, high)",
        ),
    ] = None
    timeout: Annotated[
        int | None,
        POSITIVE_INT,
        Field(default=None, ge=1, serialization_alias="timeout_s"),
        Parameter(help="Seconds allowed per probe"),
    ] = None
    max_turns: Annotated[
        int | None,
        POSITIVE_INT,
        Field(default=None, ge=1),
        Parameter(
            name=["--max-turns", "-T"],
            help="Maximum conversation turns to execute and evaluate (default: 3)",
        ),
    ] = None
    early_exit: Annotated[
        bool | None,
        Parameter(
            name=["--early-exit"],
            negative="--no-early-exit",
            show_default=False,
            help="Terminate multi-turn probe immediately when target skill is invoked",
        ),
    ] = None
    opt: Annotated[
        tuple[str, ...],
        LIST,
        Parameter(
            name=["--opt", "-O"],
            help="Agent option as key=value; repeatable",
        ),
    ] = ()

    @override
    def overrides(self) -> dict[str, Any]:
        """Extract runtime flags, parsing agent option pairs into options dict."""
        parsed = super().overrides()
        parsed.pop("opt", None)
        parsed.pop("max_turns", None)
        parsed.pop("early_exit", None)
        parsed.pop("model", None)
        parsed.pop("effort", None)
        options: dict[str, object] = {}
        if self.model is not None:
            options["model"] = self.model
        if self.effort is not None:
            options["effort"] = self.effort
        if self.max_turns is not None:
            options["max_turns"] = self.max_turns
            parsed["max_turns"] = self.max_turns
        if self.early_exit is not None:
            options["early_exit"] = self.early_exit
            parsed["early_exit"] = self.early_exit
        if self.opt:
            options.update(parse_agent_options(self.opt))
        if options:
            parsed["options"] = options
        return parsed


@FLAT
class PlanFlags(Flags):
    """CLI parameter flags configuring probing execution parameters and retry policy."""

    attempts: Annotated[
        int | None,
        POSITIVE_INT,
        Field(default=None, ge=1),
        Parameter(help="Probes per query"),
    ] = None
    retries: Annotated[
        int | None,
        NON_NEGATIVE,
        Field(default=None, ge=0),
        Parameter(help="Number of retries for failed probes"),
    ] = None
    backoff: Annotated[
        float | None,
        NON_NEGATIVE,
        Field(default=None, ge=0.0, serialization_alias="backoff_s"),
        Parameter(help="Seconds before the first retry"),
    ] = None
    pause: Annotated[
        float | None,
        NON_NEGATIVE,
        Field(default=None, ge=0.0, serialization_alias="pause_s"),
        Parameter(help="Seconds between probes"),
    ] = None
    workers: Annotated[
        int | None,
        POSITIVE_INT,
        Field(default=None, ge=1),
        Parameter(
            name=["--workers", "-j"],
            help="Number of concurrent probes to run (defaults to 1 for sequential execution)",
        ),
    ] = None


@FLAT
class StudyFlags(Flags):
    """CLI parameter flags configuring corpus paths, queries, and study workspace."""

    skills: Annotated[
        Path | None,
        Parameter(
            name=["--skills", "-s"],
            help="Root of the skill directory",
        ),
    ] = None
    queries: Annotated[Path | None, Parameter(help="Labeled query set (JSON)")] = None
    workdir: Annotated[
        Path | None,
        Parameter(help="Workspace to install the catalog into"),
    ] = None
    catalog: Annotated[
        str | None,
        Parameter(help="Catalog to evaluate; defaults to the query set's"),
    ] = None
    partial: Annotated[
        bool | None,
        SWITCH,
        Parameter(
            help="Allow a query set that targets only some of the catalog's skills",
        ),
    ] = None
    rescope: Annotated[
        bool | None,
        SWITCH,
        Parameter(
            help="Probe the set against a catalog other than the one it was labeled in",
        ),
    ] = None
    tag: Annotated[
        str | None,
        Parameter(
            help="Short semantic label for this run (e.g. v1-baseline); "
            "excluded from fingerprint and shown in headers",
        ),
    ] = None


@FLAT
class RecordFlags(Flags):
    """CLI parameter flags configuring probe results destination paths."""

    records: Annotated[
        Path | None,
        Field(default=None, serialization_alias="out"),
        Parameter(name=["--records"], help="JSONL file to append raw probe results to"),
    ] = None


@FLAT
class GenerateFlags(BaseModel):
    """CLI parameter flags configuring synthetic query generation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    targets: Annotated[
        tuple[str, ...],
        LIST,
        Parameter(
            name=["--skill"],
            help="Draft queries for these skills only; repeatable. "
            "Narrows what is asked, never what is resident",
        ),
    ] = ()

    count: Annotated[
        int,
        POSITIVE_INT,
        Field(ge=1),
        Parameter(help="Queries to draft per target"),
    ] = 3
    generator_model: Annotated[
        str,
        Parameter(help="Model that drafts the queries, not the one probed"),
    ] = DEFAULT_GEMINI_MODEL
    generator_agent: Annotated[
        str | None,
        Parameter(help="Agent driver that drafts the queries, overriding the probe agent"),
    ] = None

    generator_arm: Annotated[
        GeneratorArm,
        Parameter(help="Which generation prompt to use; recorded in the notes"),
    ] = GeneratorArm.CONTENT
    top_rivals: Annotated[
        int | None,
        POSITIVE_INT,
        Field(ge=1),
        Parameter(
            help="Maximum number of top-ranked rival skills to include in generation prompts",
        ),
    ] = None
    draft_concurrency: Annotated[
        int,
        POSITIVE_INT,
        Field(ge=1),
        Parameter(
            help="Targets to draft concurrently (defaults to 1 for sequential order)",
        ),
    ] = 1
    adversarial: Annotated[
        bool,
        SWITCH,
        Parameter(
            help="Synthesize near-miss adversarial negative queries sharing target vocabulary",
        ),
    ] = False
    adversarial_count: Annotated[
        int,
        POSITIVE_INT,
        Field(ge=1),
        Parameter(
            help="Number of adversarial negative queries to synthesize per target",
        ),
    ] = 1

    @classmethod
    def from_query_settings(
        cls,
        query: QuerySettings,
        *,
        targets: tuple[str, ...] = (),
    ) -> Self:
        """Construct GenerateFlags from QuerySettings while respecting ge=1 field constraints."""
        return cls(
            targets=targets,
            count=query.count,
            top_rivals=query.top_rivals if query.top_rivals > 0 else None,
            adversarial=query.adversarial_count > 0,
            adversarial_count=max(1, query.adversarial_count),
        )


def _load_base_config(
    config: Path | None,
    study: StudyFlags,
    required: Sequence[str],
) -> RunConfig:
    """Load base RunConfig from a TOML file or initialize from study flags."""
    if config is not None:
        return RunConfig.from_toml(config, skills=study.skills)
    missing = [flag for flag in required if getattr(study, flag) is None]
    if missing:
        raise ValueError(
            "without --config these are required: " + ", ".join(f"--{flag}" for flag in missing),
        )
    return RunConfig(study=StudySettings.model_validate(study.overrides()))


def _collect_flag_overrides(
    catalog: CatalogFlags | None,
    runtime: RuntimeFlags,
    plan: PlanFlags | None,
    study: StudyFlags,
    record: RecordFlags | None,
    registry: RegistryFlags | None = None,
) -> dict[str, Any]:
    """Assemble dictionary of flag overrides across config sections."""
    return {
        "catalog": (catalog or CatalogFlags()).overrides(),
        "runtime": runtime.overrides(),
        "plan": (plan or PlanFlags()).overrides(),
        "study": {**study.overrides(), **(record or RecordFlags()).overrides()},
        "registry": (registry or RegistryFlags()).overrides(),
    }


def build_config(
    config: Path | None = None,
    *,
    catalog: CatalogFlags | None = None,
    runtime: RuntimeFlags | None = None,
    plan: PlanFlags | None = None,
    study: StudyFlags | None = None,
    record: RecordFlags | None = None,
    registry: RegistryFlags | None = None,
    required: Sequence[str],
) -> RunConfig:
    """Build RunConfig by layering CLI flags over optional config files."""
    study_flags = study or StudyFlags()
    runtime_flags = runtime or RuntimeFlags()
    loaded = _load_base_config(config, study_flags, required)
    overrides = _collect_flag_overrides(
        catalog,
        runtime_flags,
        plan,
        study_flags,
        record,
        registry,
    )

    try:
        return loaded.with_overrides(**overrides)
    except PydanticValidationError as error:
        reason = _opt_error_reason(error, loaded, runtime_flags)
        if reason is None:
            raise
        raise ValueError(reason) from error


def _opt_error_reason(
    error: PydanticValidationError,
    loaded: RunConfig,
    runtime: RuntimeFlags,
) -> str | None:
    """Translate Pydantic option validation errors into readable error descriptions."""
    failures = [f for f in error.errors() if f["loc"] and f["loc"][0] == "runtime"]
    if not failures:
        return None
    agent = runtime.agent or loaded.runtime.agent
    model = options_model(agent)
    named = f" ({model.__name__})" if model is not None else ""
    typed = {pair.partition("=")[0]: pair for pair in runtime.opt}
    lines = [
        (
            f"-O {typed[field]!r}: {failure['msg']}"
            if (field := str(failure["loc"][-1])) in typed
            else f"{field}: {failure['msg']}"
        )
        for failure in failures
    ]
    return f"invalid option for agent {agent!r}{named}: " + "; ".join(lines)


#: Help panel grouping specifications for Cyclopts help formatting.
STUDY_GROUP = Group("Study", sort_key=1)
CATALOG_GROUP = Group("Catalog", sort_key=2)
RUNTIME_GROUP = Group("Runtime", sort_key=3)
PLAN_GROUP = Group("Plan", sort_key=4)
RECORD_GROUP = Group("Recording", sort_key=5)
GENERATE_GROUP = Group("Generation", sort_key=6)
REGISTRY_GROUP = Group("Agent Registry", sort_key=7)
RULES_GROUP = Group("Lint rules", sort_key=8)
THRESHOLDS_GROUP = Group("Threshold assertions", sort_key=9)
