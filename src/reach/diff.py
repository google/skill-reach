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

"""Cross two recorded evaluation arms and statistically test performance deltas."""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, computed_field

from reach.artifact import (
    ARTIFACT_SUFFIX,
    Artifact,
    _cross_check,
    filter_query_set,
    read_artifact,
)
from reach.catalog import corpus_digest
from reach.config import DiffSettings, RunConfig, reanchor
from reach.models import Provenance, QueryKind
from reach.queries import load_query_set, query_set_digest
from reach.run import (
    Composition,
    ConfigSidecar,
    compose,
    load_results,
    read_sidecar,
    sidecar_path,
)
from reach.uncertainty import (
    DEFAULT_CONFIDENCE,
    Interval,
    critical_value,
    detectable_delta,
    required_probes,
    wilson_interval,
)


class VaryFactor(StrEnum):
    """Specify the single experimental factor varied between compared runs."""

    AGENT = "agent"
    DESCRIPTION = "description"
    MODEL = "model"
    RIVAL = "rival"
    SCOPE = "scope"


_DEFAULT_DIFF = DiffSettings()

#: Noise inflation multiplier applied to standard errors.
NOISE_INFLATION = _DEFAULT_DIFF.noise_inflation

#: Maximum number of roster items shown in summaries before truncation.
ROSTER_SHOWN = 6


def noise_floor(
    control: float,
    treatment: float,
    confidence: float = DEFAULT_CONFIDENCE,
    noise_inflation: float = NOISE_INFLATION,
) -> float:
    """Calculate minimum top-1 delta distinguishable from noise at confidence."""
    return critical_value(confidence) * noise_inflation * (control + treatment)


def probes_to_resolve(delta: float, noise_inflation: float = NOISE_INFLATION) -> int:
    """Calculate required probes per arm to reliably detect a given accuracy delta."""
    if not 0.0 < delta <= 1.0:
        msg = f"delta must lie in (0, 1], got {delta}"
        raise ValueError(msg)
    return required_probes(delta / noise_inflation)


class Arm(BaseModel):
    """Hold a single experimental arm and its associated Artifact."""

    model_config = ConfigDict(frozen=True)

    label: str
    artifact: Artifact
    source_queries_digest: str = ""

    @property
    def resident(self) -> tuple[str, ...]:
        """Return the sequence of skill names resident in this arm."""
        return tuple(entry.skill for entry in self.artifact.skills)

    @property
    def query_ids(self) -> frozenset[str]:
        """Return the set of query IDs evaluated in this arm."""
        return frozenset(record.query_id for record in self.artifact.queries)


class ArmSummary(BaseModel):
    """Summarize provenance digests, catalog sizes, and accuracy metrics for an arm."""

    model_config = ConfigDict(frozen=True)

    label: str
    arm: str
    corpus_digest: str
    queries_digest: str
    catalog_id: str
    catalog_size: Annotated[int, Field(ge=0)]
    resident: tuple[str, ...] = ()
    attempts: Annotated[int, Field(ge=1)]
    probes: Annotated[int, Field(ge=0)]
    scored: Annotated[int, Field(ge=0)]
    top1_hits: Annotated[int, Field(ge=0)]
    top1_accuracy: float
    consistency: float
    standard_error: float | None = None
    entrypoint_accuracy: float = 0.0
    trajectory_reachability: float = 0.0
    step_efficiency: float | None = None
    skill_f1: float | None = None
    redundancy: float = 0.0


class Corroboration(BaseModel):
    """Record verification status of provenance changes between arms."""

    model_config = ConfigDict(frozen=True)

    factor: VaryFactor
    arm_moved: bool
    corpus_moved: bool
    added: tuple[str, ...] = ()
    removed: tuple[str, ...] = ()
    corroborated: bool
    reason: str


class SkillDelta(BaseModel):
    """Record per-skill recall differences and overlap significance between arms."""

    model_config = ConfigDict(frozen=True)

    skill: str
    control_reached: Annotated[int, Field(ge=0)]
    control_probes: Annotated[int, Field(gt=0)]
    control_recall: float
    control_interval: Interval
    treatment_reached: Annotated[int, Field(ge=0)]
    treatment_probes: Annotated[int, Field(gt=0)]
    treatment_recall: float
    treatment_interval: Interval
    real: bool

    @computed_field
    @property
    def delta(self) -> float:
        """Calculate treatment recall minus control recall."""
        return self.treatment_recall - self.control_recall


class QueryDelta(BaseModel):
    """Record per-query hit rate differences and overlap significance between arms."""

    model_config = ConfigDict(frozen=True)

    query_id: str
    kind: QueryKind | None = None
    control_hits: Annotated[int, Field(ge=0)]
    control_probes: Annotated[int, Field(gt=0)]
    control_rate: float
    control_interval: Interval
    treatment_hits: Annotated[int, Field(ge=0)]
    treatment_probes: Annotated[int, Field(gt=0)]
    treatment_rate: float
    treatment_interval: Interval
    real: bool

    @computed_field
    @property
    def delta(self) -> float:
        """Calculate treatment hit rate minus control hit rate."""
        return self.treatment_rate - self.control_rate


class Headline(BaseModel):
    """Summarize run-wide top-1 accuracy change and noise floor significance."""

    model_config = ConfigDict(frozen=True)

    control: float
    treatment: float
    confidence: Annotated[float, Field(gt=0.0, lt=1.0)] = DEFAULT_CONFIDENCE
    noise_inflation: Annotated[float, Field(gt=0.0)] = NOISE_INFLATION
    floor: float | None = None
    real: bool
    resolvable: float | None = None
    needed_probes: int | None = None

    @computed_field
    @property
    def delta(self) -> float:
        """Calculate treatment top-1 accuracy minus control top-1 accuracy."""
        return self.treatment - self.control


class Comparison(BaseModel):
    """Hold diff analysis between two arms across headline, skill, and query levels."""

    model_config = ConfigDict(frozen=True)

    factor: VaryFactor
    control: ArmSummary
    treatment: ArmSummary
    shared_queries: Annotated[int, Field(gt=0)]
    corroboration: Corroboration
    headline: Headline
    skills: tuple[SkillDelta, ...] = ()
    queries: tuple[QueryDelta, ...] = ()

    @property
    def deltas(self) -> tuple[QueryDelta, ...]:
        """Return query delta records."""
        return self.queries

    @property
    def separated(self) -> tuple[QueryDelta, ...]:
        """Return queries whose intervals showed significant divergence."""
        return tuple(delta for delta in self.queries if delta.real)

    @property
    def verdict(self) -> str:
        """Generate a concise headline summary verdict string for the comparison."""
        points = self.headline.delta * 100
        if self.headline.floor is None:
            return (
                f"top-1 moved {points:+.1f} points, and neither arm reported a "
                "spread to price it against: no verdict"
            )
        floor = self.headline.floor * 100
        if self.headline.real:
            direction = "rose" if points > 0 else "fell"
            return (
                f"top-1 {direction} {abs(points):.1f} points, clearing the "
                f"{floor:.1f}-point noise floor"
            )
        needed = self.headline.needed_probes
        owed = (
            f"; separating a delta this size needs about {needed} probes per arm"
            if needed is not None
            else ""
        )
        if separated := self.separated:
            named = ", ".join(delta.query_id for delta in separated[:ROSTER_SHOWN])
            rest = len(separated) - len(separated[:ROSTER_SHOWN])
            more = f" and {rest} more" if rest else ""
            return (
                f"top-1 moved {points:+.1f} points, inside the {floor:.1f}-point "
                f"noise floor, but {len(separated)} of {len(self.queries)} "
                f"queries separated on their own: {named}{more}{owed}"
            )
        return (
            f"top-1 moved {points:+.1f} points, inside the {floor:.1f}-point "
            f"noise floor: not an improvement{owed}"
        )


CONTROL, TREATMENT, PAIRING = "control", "treatment", "pairing"


class Wall(BaseModel):
    """Represent an incompatibility barrier preventing comparison between two runs."""

    model_config = ConfigDict(frozen=True)

    where: str
    path: Path | None = None
    reason: str


class Survey(BaseModel):
    """Survey and validate readiness of two runs for pairwise diff comparison."""

    model_config = ConfigDict(frozen=True)

    factor: VaryFactor
    control_path: Path
    treatment_path: Path
    walls: tuple[Wall, ...] = ()
    control: Annotated[Arm | None, Field(exclude=True)] = None
    treatment: Annotated[Arm | None, Field(exclude=True)] = None

    @property
    def comparable(self) -> bool:
        """Return True if no validation barriers prevent comparison."""
        return not self.walls

    def cross(
        self,
        *,
        confidence: float = DEFAULT_CONFIDENCE,
        noise_inflation: float = NOISE_INFLATION,
        settings: DiffSettings | None = None,
    ) -> Comparison:
        """Perform comparison across validated arms, raising error if barriers exist."""
        if self.control is None or self.treatment is None:
            msg = (
                f"{self.control_path.name} and {self.treatment_path.name} cannot be "
                f"crossed: {len(self.walls)} validation barriers detected"
            )
            raise ValueError(msg)
        return diff_arms(
            self.control,
            self.treatment,
            self.factor,
            confidence=confidence,
            noise_inflation=noise_inflation,
            settings=settings,
        )


def _read_valid_sidecar(path: Path) -> ConfigSidecar:
    """Read sidecar file for a results path, validating existence."""
    sidecar = sidecar_path(path)
    if not sidecar.exists():
        msg = (
            f"{path} has no sidecar at {sidecar.name}: the query set and the "
            "corpus it was scored against cannot be identified"
        )
        raise FileNotFoundError(msg)
    return read_sidecar(sidecar)


def _resolve_arm_config(
    path: Path,
    config: RunConfig,
    queries_root: Path | str | None,
    corpus: Path | str | None,
) -> RunConfig:
    """Re-anchor queries and corpus paths in recorded configuration and validate existence."""
    queries_path = config.require_queries(f"recorded at {path}")
    if not queries_path.exists() and queries_root is not None:
        queries_path = reanchor(queries_path, Path(queries_root).expanduser())
    corpus_path = Path(corpus).expanduser() if corpus is not None else config.study.skills
    reanchored = config.with_overrides(
        study={"queries": queries_path, "skills": corpus_path},
    )

    if reanchored.study.queries is None or not reanchored.study.queries.exists():
        msg = (
            f"{path} was scored against {queries_path}, which is not here; pass "
            "a queries root to specify their current location"
        )
        raise FileNotFoundError(msg)
    if reanchored.study.skills is None:
        msg = (
            f"{path} was recorded without a named corpus; skills were "
            "discovered at runtime without a recorded path. Pass --corpus to "
            "specify the skill directory it was evaluated against."
        )
        raise ValueError(msg)
    if not reanchored.study.skills.exists():
        msg = (
            f"{path} was probed against the corpus at {reanchored.study.skills}, "
            "which is not here; pass --corpus to specify its current location"
        )
        raise FileNotFoundError(msg)
    return reanchored


def _try_read_unsliced_artifact(path: Path) -> Artifact | None:
    """Read an already-assembled Artifact JSON when path is not a sidecar-backed JSONL run."""
    if path.name.endswith(ARTIFACT_SUFFIX):
        return read_artifact(path)
    if path.suffix == ".json" and not (
        path.with_name(f"{path.name}.config.json").exists()
        or path.with_suffix(".config.json").exists()
    ):
        try:
            return read_artifact(path)
        except (OSError, ValueError):
            return None
    return None


def _resolve_sibling_jsonl(path: Path) -> Path | None:
    """Locate sibling raw .jsonl results file (with sidecar) for an .artifact.json path."""
    candidates: list[Path] = []
    if path.name.endswith(ARTIFACT_SUFFIX):
        base = path.with_name(path.name[: -len(ARTIFACT_SUFFIX)])
        candidates.extend((base, base.with_suffix(".jsonl")))
    elif path.suffix == ".json":
        candidates.append(path.with_suffix(".jsonl"))
    for candidate in candidates:
        if candidate.is_file() and sidecar_path(candidate).exists():
            return candidate
    return None


def load_arm(
    results_path: Path | str,
    *,
    queries_root: Path | str | None = None,
    corpus: Path | str | None = None,
    label: str | None = None,
    queries: Path | str | None = None,
    filter_skill: Sequence[str] = (),
    filter_id: Sequence[str] = (),
) -> Arm:
    """Load results and reconstruct the Artifact model for an experimental arm."""
    path = Path(results_path).expanduser()
    if not path.exists():
        msg = f"{path} does not exist"
        raise FileNotFoundError(msg)

    subset_qs = load_query_set(queries) if queries is not None else None
    slicing = bool(subset_qs is not None or filter_skill or filter_id)

    resolved_label = (
        label
        if label is not None
        else (
            path.name[: -len(ARTIFACT_SUFFIX)] if path.name.endswith(ARTIFACT_SUFFIX) else path.stem
        )
    )
    if not slicing and (unsliced := _try_read_unsliced_artifact(path)) is not None:
        return Arm(
            label=resolved_label,
            artifact=unsliced,
            source_queries_digest=unsliced.digests.queries_digest,
        )

    if slicing and not sidecar_path(path).exists():
        sibling = _resolve_sibling_jsonl(path)
        if sibling is None:
            msg = (
                f"{path} has no sibling .jsonl results and .config.json sidecar; "
                "query sub-slicing requires raw probe results to recompute exact scores"
            )
            raise ValueError(msg)
        path = sibling

    recorded = _read_valid_sidecar(path)
    config = _resolve_arm_config(path, recorded.config, queries_root, corpus)
    composed = compose(config)
    raw_results = load_results(path)

    if not slicing:
        return Arm(
            label=resolved_label,
            artifact=Artifact.assemble(
                composed,
                raw_results,
            ),
        )

    expected_q_digest = query_set_digest(composed.query_set)
    _cross_check(
        raw_results,
        Provenance(
            config_fingerprint=config.fingerprint,
            condition_digest=config.condition,
            corpus_digest=corpus_digest(composed.skills),
            queries_digest=expected_q_digest,
            tag=config.study.tag,
        ),
    )

    sliced_qs = filter_query_set(
        composed.query_set,
        subset=subset_qs,
        filter_skill=filter_skill,
        filter_id=filter_id,
    )
    sliced_digest = query_set_digest(sliced_qs)
    kept_ids = {q.id for q in sliced_qs.queries}
    sliced_results = [
        row.model_copy(update={"queries_digest": sliced_digest})
        for row in raw_results
        if row.query_id in kept_ids
    ]
    if not sliced_results:
        msg = f"{path} has no probe results matching the requested query slice"
        raise ValueError(msg)

    sliced_composition = Composition(
        config=config,
        query_set=sliced_qs,
        catalog=composed.catalog,
        skills=composed.skills,
    )
    return Arm(
        label=resolved_label,
        artifact=Artifact.assemble(
            sliced_composition,
            sliced_results,
        ),
        source_queries_digest=expected_q_digest,
    )


_EXPECTED = {
    VaryFactor.AGENT: (
        "the runtime agent driver should have moved while the corpus digest "
        "and resident catalog held: switching agent runtimes changes how the "
        "corpus is probed, not what the skills say or who is resident"
    ),
    VaryFactor.DESCRIPTION: (
        "the corpus digest should have moved and the arm should have held: a "
        "description is not a setting, so rewriting one changes what was "
        "probed without changing how"
    ),
    VaryFactor.MODEL: (
        "the runtime model should have moved while the corpus digest and "
        "resident catalog held: switching models changes how the corpus is "
        "probed, not what the skills say or who is resident"
    ),
    VaryFactor.RIVAL: (
        "the resident catalog should differ by the rival, and the corpus "
        "digest should have held: taking a skill out of the room does not "
        "edit anyone's description"
    ),
    VaryFactor.SCOPE: (
        "the catalog should differ in size, and the corpus digest should have "
        "held: rescoping changes who is resident, not what they say"
    ),
}


def _corroborate(factor: VaryFactor, control: Arm, treatment: Arm) -> Corroboration:
    """Verify observed changes between arms match varied factor expectations."""
    resident_control, resident_treatment = (
        set(control.resident),
        set(treatment.resident),
    )
    added = tuple(sorted(resident_treatment - resident_control))
    removed = tuple(sorted(resident_control - resident_treatment))
    residents_held = not (added or removed)
    arm_moved = control.artifact.provenance.arm != treatment.artifact.provenance.arm
    runtime_moved = control.artifact.provenance.runtime != treatment.artifact.provenance.runtime
    model_moved = control.artifact.provenance.model != treatment.artifact.provenance.model
    corpus_moved = (
        control.artifact.digests.corpus_digest != treatment.artifact.digests.corpus_digest
    )

    match factor:
        case VaryFactor.AGENT:
            held = arm_moved and runtime_moved and not corpus_moved and residents_held
        case VaryFactor.MODEL:
            held = (
                arm_moved
                and model_moved
                and not runtime_moved
                and not corpus_moved
                and residents_held
            )
        case VaryFactor.DESCRIPTION:
            held = corpus_moved and not arm_moved
        case VaryFactor.RIVAL:
            held = bool(added or removed) and not corpus_moved
        case VaryFactor.SCOPE:
            held = (
                control.artifact.catalog_size != treatment.artifact.catalog_size
                and not corpus_moved
            )

    return Corroboration(
        factor=factor,
        arm_moved=arm_moved,
        corpus_moved=corpus_moved,
        added=added,
        removed=removed,
        corroborated=held,
        reason=(
            f"what moved matches --vary {factor}"
            if held
            else f"{_EXPECTED[factor]}; that is not what these two arms show"
        ),
    )


def _summarize(arm: Arm) -> ArmSummary:
    """Extract key metrics and provenance properties from an Arm into an ArmSummary."""
    artifact = arm.artifact
    return ArmSummary(
        label=arm.label,
        arm=artifact.provenance.arm,
        corpus_digest=artifact.digests.corpus_digest,
        queries_digest=artifact.digests.queries_digest,
        catalog_id=artifact.catalog_id,
        catalog_size=artifact.catalog_size,
        resident=arm.resident,
        attempts=artifact.provenance.attempts,
        probes=artifact.probes,
        scored=artifact.scores.scored,
        top1_hits=artifact.scores.top1_hits,
        top1_accuracy=artifact.scores.top1_accuracy,
        consistency=artifact.scores.consistency,
        standard_error=artifact.spread.standard_error,
        entrypoint_accuracy=artifact.scores.entrypoint_accuracy,
        trajectory_reachability=artifact.scores.trajectory_reachability,
        step_efficiency=artifact.scores.step_efficiency,
        skill_f1=artifact.scores.skill_f1,
        redundancy=artifact.scores.redundancy,
    )


def _headline(
    control: Arm,
    treatment: Arm,
    confidence: float,
    noise_inflation: float = NOISE_INFLATION,
) -> Headline:
    """Calculate top-1 delta and test significance against the noise floor."""
    control_error = control.artifact.spread.standard_error
    treatment_error = treatment.artifact.spread.standard_error
    delta = abs(
        treatment.artifact.scores.top1_accuracy - control.artifact.scores.top1_accuracy,
    )
    floor = (
        noise_floor(control_error, treatment_error, confidence, noise_inflation)
        if control_error is not None and treatment_error is not None
        else None
    )
    return Headline(
        control=control.artifact.scores.top1_accuracy,
        treatment=treatment.artifact.scores.top1_accuracy,
        confidence=confidence,
        noise_inflation=noise_inflation,
        floor=floor,
        real=floor is not None and delta > floor,
        resolvable=detectable_delta(
            min(control.artifact.scores.scored, treatment.artifact.scores.scored),
        ),
        needed_probes=(probes_to_resolve(delta, noise_inflation) if delta > 0.0 else None),
    )


def _skill_deltas(
    control: Arm,
    treatment: Arm,
    confidence: float,
) -> tuple[SkillDelta, ...]:
    """Compute per-skill recall deltas and Wilson interval non-overlap significance."""
    theirs = {entry.skill: entry for entry in treatment.artifact.skills}
    deltas = []
    for mine in control.artifact.skills:
        yours = theirs.get(mine.skill)
        if yours is None or mine.recall is None or yours.recall is None:
            continue
        left = wilson_interval(mine.reached, mine.probes, confidence)
        right = wilson_interval(yours.reached, yours.probes, confidence)
        if left is None or right is None:  # pragma: no cover
            continue
        deltas.append(
            SkillDelta(
                skill=mine.skill,
                control_reached=mine.reached,
                control_probes=mine.probes,
                control_recall=mine.recall,
                control_interval=left,
                treatment_reached=yours.reached,
                treatment_probes=yours.probes,
                treatment_recall=yours.recall,
                treatment_interval=right,
                real=not left.overlaps(right),
            ),
        )
    return tuple(sorted(deltas, key=lambda d: (-abs(d.delta), d.skill)))


def _query_deltas(
    control: Arm,
    treatment: Arm,
    confidence: float,
) -> tuple[QueryDelta, ...]:
    """Compute per-query hit rate deltas and Wilson interval divergence."""
    theirs = {record.query_id: record for record in treatment.artifact.queries}
    deltas = []
    for mine in control.artifact.queries:
        yours = theirs.get(mine.query_id)
        if yours is None or not mine.probes or not yours.probes:
            continue
        left = wilson_interval(mine.hits, mine.probes, confidence)
        right = wilson_interval(yours.hits, yours.probes, confidence)
        if left is None or right is None:  # pragma: no cover
            continue
        deltas.append(
            QueryDelta(
                query_id=mine.query_id,
                kind=mine.kind,
                control_hits=mine.hits,
                control_probes=mine.probes,
                control_rate=mine.hits / mine.probes,
                control_interval=left,
                treatment_hits=yours.hits,
                treatment_probes=yours.probes,
                treatment_rate=yours.hits / yours.probes,
                treatment_interval=right,
                real=not left.overlaps(right),
            ),
        )
    return tuple(sorted(deltas, key=lambda d: (-abs(d.delta), d.query_id)))


def pairing_walls(control: Arm, treatment: Arm) -> tuple[Wall, ...]:
    """Identify structural barriers preventing comparison between two loaded arms."""
    walls = []
    shared = control.query_ids & treatment.query_ids
    if control.query_ids != treatment.query_ids:
        only_control = sorted(control.query_ids - treatment.query_ids)
        only_treatment = sorted(treatment.query_ids - control.query_ids)
        walls.append(
            Wall(
                where=PAIRING,
                reason=(
                    f"refusing to compare {control.label} with {treatment.label}: "
                    f"they were scored on different queries ({len(only_control)} "
                    f"only in {control.label}, {len(only_treatment)} only in "
                    f"{treatment.label}). A comparison holds the query set fixed "
                    "and varies one factor; these vary the questions."
                ),
            ),
        )
    elif not shared:
        walls.append(
            Wall(
                where=PAIRING,
                reason=(
                    f"refusing to compare {control.label} with {treatment.label}: "
                    "neither arm was scored on any query"
                ),
            ),
        )

    digests = (
        control.artifact.digests.queries_digest,
        treatment.artifact.digests.queries_digest,
    )
    if not digests[0] or not digests[1]:
        walls.append(
            Wall(
                where=PAIRING,
                reason=(
                    f"refusing to compare {control.label} with {treatment.label}: "
                    "both arms must have recorded queries_digest"
                ),
            ),
        )
    elif digests[0] != digests[1]:
        walls.append(
            Wall(
                where=PAIRING,
                reason=(
                    f"refusing to compare {control.label} with {treatment.label}: "
                    f"their ground truth digests differ ({digests[0]} and "
                    f"{digests[1]}). The queries are the same but their labels are "
                    "not, so the delta would measure the reviewer rather than the "
                    "change."
                ),
            ),
        )
    elif (
        control.source_queries_digest
        and treatment.source_queries_digest
        and control.source_queries_digest != treatment.source_queries_digest
    ):
        walls.append(
            Wall(
                where=PAIRING,
                reason=(
                    f"refusing to compare {control.label} with {treatment.label}: "
                    f"their recorded source query set digests differ "
                    f"({control.source_queries_digest} and {treatment.source_queries_digest})."
                ),
            ),
        )
    return tuple(walls)


def diff_arms(
    control: Arm,
    treatment: Arm,
    factor: VaryFactor | str,
    *,
    confidence: float = DEFAULT_CONFIDENCE,
    noise_inflation: float = NOISE_INFLATION,
    settings: DiffSettings | None = None,
) -> Comparison:
    """Compare two experimental arms across the specified variation factor."""
    if settings is not None:
        confidence = settings.confidence
        noise_inflation = settings.noise_inflation

    factor = VaryFactor(factor)

    if walls := pairing_walls(control, treatment):
        raise ValueError(walls[0].reason)

    shared = control.query_ids & treatment.query_ids
    return Comparison(
        factor=factor,
        control=_summarize(control),
        treatment=_summarize(treatment),
        shared_queries=len(shared),
        corroboration=_corroborate(factor, control, treatment),
        headline=_headline(control, treatment, confidence, noise_inflation),
        skills=_skill_deltas(control, treatment, confidence),
        queries=_query_deltas(control, treatment, confidence),
    )


def _same_file(left: Path, right: Path) -> Wall | None:
    """Return a Wall if both comparison paths resolve to the identical file."""
    if left != right:
        return None
    return Wall(
        where=PAIRING,
        reason=(
            f"refusing to compare {left} with itself: a comparison needs two "
            "arms, and this one would report a delta of zero by construction"
        ),
    )


def _survey_arm(
    where: str,
    path: Path,
    *,
    label: str | None = None,
    queries_root: Path | str | None = None,
    corpus: Path | str | None = None,
    queries: Path | str | None = None,
    filter_skill: Sequence[str] = (),
    filter_id: Sequence[str] = (),
) -> tuple[
    Arm | None,
    Wall | None,
]:
    """Attempt loading an arm, returning either the loaded Arm or a Wall error."""
    try:
        return (
            load_arm(
                path,
                label=label,
                queries_root=queries_root,
                corpus=corpus,
                queries=queries,
                filter_skill=filter_skill,
                filter_id=filter_id,
            ),
            None,
        )
    except (ValueError, LookupError, OSError) as error:
        return None, Wall(where=where, path=path, reason=str(error))


def survey_runs(
    control_path: Path | str,
    treatment_path: Path | str,
    factor: VaryFactor | str,
    *,
    queries_root: Path | str | None = None,
    control_corpus: Path | str | None = None,
    treatment_corpus: Path | str | None = None,
    control_label: str | None = None,
    treatment_label: str | None = None,
    queries: Path | str | None = None,
    filter_skill: Sequence[str] = (),
    filter_id: Sequence[str] = (),
) -> Survey:
    """Survey and validate two run paths against all comparative requirements."""
    factor = VaryFactor(factor)
    left = Path(control_path).expanduser().resolve()
    right = Path(treatment_path).expanduser().resolve()
    if wall := _same_file(left, right):
        return Survey(
            factor=factor,
            control_path=left,
            treatment_path=right,
            walls=(wall,),
        )

    control, control_wall = _survey_arm(
        CONTROL,
        left,
        queries_root=queries_root,
        corpus=control_corpus,
        label=control_label,
        queries=queries,
        filter_skill=filter_skill,
        filter_id=filter_id,
    )
    treatment, treatment_wall = _survey_arm(
        TREATMENT,
        right,
        queries_root=queries_root,
        corpus=treatment_corpus,
        label=treatment_label,
        queries=queries,
        filter_skill=filter_skill,
        filter_id=filter_id,
    )
    walls = [wall for wall in (control_wall, treatment_wall) if wall is not None]

    if control is not None and treatment is not None:
        walls.extend(pairing_walls(control, treatment))
    else:
        standing = "neither arm" if walls[1:] else "one of the two arms"
        walls.append(
            Wall(
                where=PAIRING,
                reason=(
                    f"not reached: {standing} could be rebuilt, so whether these "
                    "two were scored on the same questions is not yet knowable"
                ),
            ),
        )

    return Survey(
        factor=factor,
        control_path=left,
        treatment_path=right,
        walls=tuple(walls),
        control=control,
        treatment=treatment,
    )


def diff_runs(
    control_path: Path | str,
    treatment_path: Path | str,
    factor: VaryFactor | str,
    *,
    queries_root: Path | str | None = None,
    control_corpus: Path | str | None = None,
    treatment_corpus: Path | str | None = None,
    control_label: str | None = None,
    treatment_label: str | None = None,
    queries: Path | str | None = None,
    filter_skill: Sequence[str] = (),
    filter_id: Sequence[str] = (),
    confidence: float = DEFAULT_CONFIDENCE,
    noise_inflation: float = NOISE_INFLATION,
) -> Comparison:
    """Load and execute comparison between two results files across a factor."""
    left = Path(control_path).expanduser().resolve()
    right = Path(treatment_path).expanduser().resolve()
    if wall := _same_file(left, right):
        raise ValueError(wall.reason)
    return diff_arms(
        load_arm(
            left,
            queries_root=queries_root,
            corpus=control_corpus,
            label=control_label,
            queries=queries,
            filter_skill=filter_skill,
            filter_id=filter_id,
        ),
        load_arm(
            right,
            queries_root=queries_root,
            corpus=treatment_corpus,
            label=treatment_label,
            queries=queries,
            filter_skill=filter_skill,
            filter_id=filter_id,
        ),
        factor,
        confidence=confidence,
        noise_inflation=noise_inflation,
    )


__all__ = [
    "Arm",
    "ArmSummary",
    "Comparison",
    "Corroboration",
    "Headline",
    "QueryDelta",
    "SkillDelta",
    "Survey",
    "VaryFactor",
    "Wall",
    "diff_arms",
    "diff_runs",
    "load_arm",
    "noise_floor",
    "pairing_walls",
    "probes_to_resolve",
    "survey_runs",
]
