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

"""Load skills from a repository root and compose them into catalogs for evaluation."""

from __future__ import annotations

import contextlib
import functools
import hashlib
import json
import logging
import math
import re
from pathlib import Path
from random import Random
from typing import TYPE_CHECKING, Any, Final

import yaml
from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
)

from reach.config import resolve_path
from reach.models import Catalog, CatalogMode, Skill
from reach.queries import QuerySet
from reach.retrieval import Bm25Scorer, Scorer, skill_text, tokenize

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

__all__ = [
    "DEFAULT_SWEEP_SCALES",
    "CorpusScalingPlan",
    "ResolvedTarget",
    "build_catalogs",
    "build_corpus_scaling_catalogs",
    "build_corpus_scaling_queries",
    "build_corpus_scaling_sequence",
    "build_neighborhood_catalogs",
    "build_scaling_catalogs",
    "corpus_digest",
    "deduplicate_skills",
    "determine_min_scale",
    "find_cluster_medoids",
    "find_skill_manifest",
    "generate_log_scales",
    "load_registry_skills",
    "load_skills",
    "parse_frontmatter",
    "resident_skills",
    "resolve_catalog",
    "resolve_skill_target",
    "resolve_sweep_scales",
    "split_frontmatter",
]

FRONTMATTER_DELIMITER = "---"
FRONTMATTER_SPLIT_PARTS: Final = 3
MIN_NEIGHBORHOOD_SIZE: Final = 2
BOM: Final = "\ufeff"
_FRONTMATTER_PATTERN = re.compile(r"^---\s*$", re.MULTILINE)

#: Key in SKILL.md frontmatter indicating exclusion from model tool selection.
HIDE_FROM_MODEL_KEY: Final = "disable-model-invocation"

#: Truthy string representations accepted for boolean frontmatter values.
TRUTHY = frozenset({"true", "yes", "on", "1"})

#: Marker appended to corpus digest when a skill is hidden from model invocation.
HIDDEN_MARKER = "\ndisclosure: hidden"


def _is_set(value: object) -> bool:
    """Check if a frontmatter value represents a truthy boolean or string value."""
    if isinstance(value, str):
        return value.strip().lower() in TRUTHY
    return bool(value)


_SKILL_TOOL_PATTERN = re.compile(r"Skill\(\s*([a-z0-9_-]+)\s*\)", re.IGNORECASE)


@functools.lru_cache(maxsize=512)
def _find_manifest_for_dir(directory: Path, _mtime_ns: int) -> Path | None:
    """Recursively search upward from directory for skills.json or skills-lock.json."""
    for candidate in ("skills.json", "skills-lock.json"):
        manifest = directory / candidate
        if manifest.is_file():
            return manifest
    if (directory / ".git").exists() or directory == directory.parent:
        return None
    parent = directory.parent
    try:
        parent_mtime = parent.stat().st_mtime_ns
    except OSError:
        parent_mtime = 0
    return _find_manifest_for_dir(parent, parent_mtime)


def find_skill_manifest(skill_path: Path) -> Path | None:
    """Traverse upward from SKILL.md or directory to locate nearest skills.json or lockfile."""
    resolved = resolve_path(skill_path)
    current = resolved if resolved.is_dir() else resolved.parent
    try:
        mtime_ns = current.stat().st_mtime_ns
    except OSError:
        mtime_ns = 0
    return _find_manifest_for_dir(current, mtime_ns)


def _extract_allowed_skills(raw_allowed: object) -> tuple[str, ...]:
    """Extract scoped skill names from allowed-tools string or sequence."""
    if isinstance(raw_allowed, str):
        return tuple(sorted(set(_SKILL_TOOL_PATTERN.findall(raw_allowed))))
    if isinstance(raw_allowed, (list, tuple)):
        found: set[str] = set()
        for item in raw_allowed:
            found.update(_SKILL_TOOL_PATTERN.findall(str(item)))
        return tuple(sorted(found))
    return ()


def _extract_declared_dependencies(
    raw_frontmatter: dict[str, Any],
    allowed_skills: tuple[str, ...],
) -> tuple[str, ...]:
    """Extract and union declared dependencies from allowed-tools and metadata."""
    deps: set[str] = set(allowed_skills)
    meta = raw_frontmatter.get("metadata", {})

    if isinstance(meta, dict):
        for key in ("requires_skill", "depends_on", "depends-on", "requires_skills", "helpers"):
            val = meta.get(key)
            if isinstance(val, str):
                deps.update(s.strip() for s in val.split(",") if s.strip())
            elif isinstance(val, (list, tuple)):
                deps.update(str(s).strip() for s in val if s)

        req = meta.get("requires")
        if isinstance(req, dict):
            skill_list = req.get("skills", [])
            if isinstance(skill_list, (list, tuple)):
                deps.update(str(s).strip() for s in skill_list if s)

    return tuple(sorted(deps))


@functools.lru_cache(maxsize=256)
def _load_manifest_json(manifest_path: Path, _mtime_ns: int) -> dict[str, Any] | None:
    """Load and cache parsed JSON dictionary from a manifest file."""
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def _resolve_manifest_source(skill_name: str, manifest_path: Path | None) -> str | None:
    """Extract package source or identifier from a resolved manifest or lockfile."""
    if manifest_path is None or not manifest_path.is_file():
        return None
    try:
        mtime_ns = manifest_path.stat().st_mtime_ns
    except OSError:
        return None
    data = _load_manifest_json(manifest_path, mtime_ns)
    if not isinstance(data, dict):
        return None
    if manifest_path.name == "skills-lock.json":
        skills = data.get("skills")
        if isinstance(skills, dict):
            info = skills.get(skill_name)
            if isinstance(info, dict):
                src = info.get("source")
                if isinstance(src, str) and src:
                    return src
    elif manifest_path.name == "skills.json":
        name = data.get("name")
        if isinstance(name, str) and name:
            return name
    return None


class _SkillFrontmatter(BaseModel):
    """Represent the parsed YAML frontmatter metadata block from a SKILL.md file."""

    model_config = ConfigDict(extra="ignore")

    name: str | None = None
    description: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    allowed_tools: Any = Field(
        default=None,
        validation_alias=AliasChoices("allowed-tools", "allowed_tools"),
    )
    disable_model_invocation: Any = Field(
        default=None,
        validation_alias=AliasChoices(HIDE_FROM_MODEL_KEY, "disable_model_invocation"),
    )

    @field_validator("description")
    @classmethod
    def _reject_blank_description(cls, value: str) -> str:
        """Validate that frontmatter description is a non-empty, non-whitespace string."""
        if not value or not value.strip():
            msg = "skill description must not be empty or whitespace"
            raise ValueError(msg)
        return value

    def is_model_invocable(self) -> bool:
        """Check whether the skill allows model invocation based on frontmatter flags."""
        return not _is_set(self.disable_model_invocation)

    def stringified_metadata(self) -> dict[str, str]:
        """Return metadata dictionary with all non-None values cast to strings."""
        return {k: str(v) for k, v in self.metadata.items() if v is not None}


def split_frontmatter(text: str) -> tuple[str, str] | None:
    """Split raw markdown text into frontmatter YAML and markdown body content.

    Args:
        text: Raw content of a markdown skill file.

    Returns:
        A tuple of (frontmatter_yaml, markdown_body) if valid delimiter lines are found,
        or None if the file lacks valid frontmatter delimiters.
    """
    stripped = text.removeprefix(BOM)
    parts = _FRONTMATTER_PATTERN.split(stripped, maxsplit=2)
    if len(parts) < FRONTMATTER_SPLIT_PARTS or parts[0] != "":
        return None
    return parts[1], parts[2]


def parse_frontmatter(text: str, path: Path) -> Skill | None:
    """Parse a SKILL.md file's YAML frontmatter into a validated Skill model.

    Args:
        text: Raw markdown file contents including frontmatter block.
        path: Filesystem path to the SKILL.md file (used for fallback naming).

    Returns:
        A validated Skill model instance, or None if frontmatter cannot be parsed.
    """
    split = split_frontmatter(text)
    if split is None:
        return None
    frontmatter, _body = split
    try:
        loaded = yaml.safe_load(frontmatter)
    except yaml.YAMLError:
        return None
    if not isinstance(loaded, dict):
        return None
    try:
        parsed = _SkillFrontmatter.model_validate(loaded)
        name = parsed.name or path.parent.name
        raw_allowed = parsed.allowed_tools
        allowed = _extract_allowed_skills(raw_allowed)
        deps = _extract_declared_dependencies(loaded, allowed)
        manifest = find_skill_manifest(path)
        manifest_src = _resolve_manifest_source(name, manifest)

        return Skill(
            name=name,
            description=parsed.description,
            metadata=parsed.stringified_metadata(),
            path=path.parent,
            allowed_tools=allowed,
            declared_dependencies=deps,
            manifest_source=manifest_src,
            model_invocable=parsed.is_model_invocable(),
        )
    except (ValueError, ValidationError) as exc:
        logging.getLogger(__name__).warning("Skipping invalid SKILL.md at %s: %s", path, exc)
        return None


def _skill_files(root: Path) -> list[Path]:
    """Find SKILL.md files under a root, following symlinks without cycles."""
    resolved_root = resolve_path(root)
    seen: set[Path] = set()
    found: list[Path] = []
    for dirpath, dirnames, filenames in resolved_root.walk(follow_symlinks=True):
        real = dirpath.resolve()
        if real in seen:
            dirnames.clear()
            continue
        seen.add(real)
        dirnames.sort()
        if "SKILL.md" in filenames:
            found.append(dirpath / "SKILL.md")
    return found


def load_skills(root: Path | str) -> list[Skill]:
    """Load and parse all skills under a directory root, sorted by skill name.

    Deduplicates skills sharing the same name by selecting the shortest path.

    Args:
        root: Directory path containing skill subdirectories or SKILL.md files.

    Returns:
        Sorted list of resident Skill objects found under the directory.

    Raises:
        NotADirectoryError: If the resolved path does not exist or is not a directory.
    """
    resolved = resolve_path(root)
    if not resolved.is_dir():
        msg = f"skill root does not exist: {resolved}"
        raise NotADirectoryError(msg)
    by_name: dict[str, Skill] = {}
    candidate_files = sorted(
        _skill_files(resolved),
        key=lambda p: (len(p.parts), str(p)),
    )
    for skill_file in candidate_files:
        skill = parse_frontmatter(skill_file.read_text(encoding="utf-8"), skill_file)
        if skill is not None and skill.name not in by_name:
            by_name[skill.name] = skill
    return sorted(by_name.values(), key=lambda s: s.name)


def load_registry_skills(
    project: str,
    location: str = "global",
    publisher: str | None = None,
    fresh: bool = False,
    no_cache: bool = False,
    cache_ttl_seconds: int = 300,
    cache_root: Path | str | None = None,
) -> list[Skill]:
    """Fetch and load skills from Google Cloud Agent Registry via local cache mirror.

    Args:
        project: Google Cloud project ID.
        location: Registry location (default: 'global').
        publisher: Optional publisher filter.
        fresh: If True, bypass metadata TTL and query live.
        no_cache: If True, run in ephemeral memory/tempdir.
        cache_ttl_seconds: TTL in seconds for metadata cache validity.
        cache_root: Optional custom cache directory.

    Returns:
        Sorted list of resident Skill objects.
    """
    from reach.registry import RegistryCacheManager

    manager = RegistryCacheManager(cache_root=cache_root)
    return manager.resolve_skills(
        project=project,
        location=location,
        publisher=publisher,
        fresh=fresh,
        no_cache=no_cache,
        cache_ttl_seconds=cache_ttl_seconds,
    )


def corpus_digest(skills: Sequence[Skill]) -> str:
    """Compute deterministic 12-char SHA-256 digest of corpus names/descriptions.

    Args:
        skills: Sequence of resident Skill objects.

    Returns:
        A 12-character hexadecimal SHA-256 digest identifying the corpus selection surface.
    """
    material = "\n".join(
        f"{skill.name}\n{skill.description}" + ("" if skill.model_invocable else HIDDEN_MARKER)
        for skill in sorted(skills, key=lambda s: s.name)
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:12]


def build_catalogs(
    skills: Sequence[Skill],
    mode: CatalogMode,
    size: int = 20,
    rivals: int = 10,
    seed: int = 0,
    scorer: Scorer | None = None,
    target_skill: str | None = None,
) -> list[Catalog]:
    """Assemble skill catalogs from a corpus using the specified cataloging mode."""
    if not skills:
        return []
    match mode:
        case CatalogMode.ALL:
            return [
                Catalog(
                    id="all",
                    mode=mode,
                    skills=tuple(s.name for s in skills),
                ),
            ]
        case CatalogMode.SINGLETON:
            return [Catalog(id=f"singleton:{s.name}", mode=mode, skills=(s.name,)) for s in skills]
        case CatalogMode.NEIGHBORHOOD:
            return build_neighborhood_catalogs(
                skills,
                size=size,
                rivals=rivals,
                seed=seed,
                scorer=scorer,
            )
        case CatalogMode.SWEEP:
            scales = resolve_sweep_scales(len(skills))
            if target_skill is not None:
                return build_scaling_catalogs(
                    skills,
                    target_skill=target_skill,
                    scales=scales,
                    seed=seed,
                    scorer=scorer,
                )
            return build_corpus_scaling_catalogs(
                skills,
                scales=scales,
                scorer=scorer,
            )

        case _:
            msg = f"unsupported catalog mode: {mode}"
            raise ValueError(msg)


_CANONICAL_LOG_STEPS: tuple[int, ...] = (
    1,
    2,
    5,
    10,
    25,
    50,
    100,
    200,
    400,
    800,
    1600,
    3200,
    6400,
    12800,
)

DEFAULT_SWEEP_SCALES: tuple[int, ...] = _CANONICAL_LOG_STEPS[3:10]

LARGE_CORPUS_THRESHOLD: Final = 50
MEDIUM_CORPUS_THRESHOLD: Final = 12
SMALL_CORPUS_THRESHOLD: Final = 2


def determine_min_scale(total_skills: int) -> int:
    """Determine optimal baseline starting scale based on corpus size."""
    if total_skills > LARGE_CORPUS_THRESHOLD:
        return 10
    if total_skills > MEDIUM_CORPUS_THRESHOLD:
        return 5
    if total_skills > SMALL_CORPUS_THRESHOLD:
        return 2
    return 1


def generate_log_scales(
    total_skills: int,
    min_scale: int | None = None,
) -> tuple[int, ...]:
    """Generate human-friendly logarithmic sweep scales up to total_skills."""
    if total_skills <= 0:
        msg = f"total_skills must be positive, got {total_skills}"
        raise ValueError(msg)
    if total_skills == 1:
        return (1,)

    start = min_scale if min_scale is not None else determine_min_scale(total_skills)
    scales = [s for s in _CANONICAL_LOG_STEPS if start <= s < total_skills]
    if (not scales or scales[0] != start) and (start < total_skills and start not in scales):
        scales.insert(0, start)
    if total_skills not in scales:
        scales.append(total_skills)
    return tuple(sorted(set(scales)))


def resolve_sweep_scales(
    total_skills: int,
    requested: Sequence[int] | None = None,
) -> tuple[int, ...]:
    """Resolve and clamp catalog sweep scales against available corpus size."""
    if total_skills <= 0:
        msg = f"total_skills must be positive, got {total_skills}"
        raise ValueError(msg)

    if not requested:
        return generate_log_scales(total_skills)

    scales = sorted({min(s, total_skills) for s in requested if s >= 1})
    if not scales:
        msg = (
            f"No valid sweep scales resolved from requested={requested}; must specify integers >= 1"
        )
        raise ValueError(msg)
    return tuple(scales)


def _extend_unique_up_to(
    target_list: list[str],
    seen_set: set[str],
    candidates: Sequence[str],
    limit: int,
) -> None:
    """Append unseen candidate names to target_list until its length reaches limit."""
    for name in candidates:
        if len(target_list) >= limit:
            break
        if name not in seen_set:
            seen_set.add(name)
            target_list.append(name)


def build_scaling_catalogs(
    skills: Sequence[Skill],
    target_skill: str,
    scales: Sequence[int],
    rivals_share: float = 0.5,
    seed: int = 0,
    scorer: Scorer | None = None,
) -> list[Catalog]:
    """Generate multi-scale catalogs for a target skill across requested scales."""
    if not skills:
        return []
    by_name = {s.name: s for s in skills}
    if target_skill not in by_name:
        msg = f"target skill {target_skill!r} not in skills"
        raise KeyError(msg)

    unique_skills = list(by_name.values())
    target_obj = by_name[target_skill]
    ranker = scorer or Bm25Scorer.from_skills(unique_skills)
    ranked = [name for name, _ in ranker.rank(target_obj, unique_skills) if name != target_skill]

    rng = Random(f"{seed}:{target_skill}")  # noqa: S311 (deterministic benchmark sampling)
    filler_order = list(ranked)
    rng.shuffle(filler_order)

    unique_sorted_scales = sorted({max(1, k) for k in scales})
    chosen_by_scale: dict[int, tuple[str, ...]] = {}
    current_chosen: list[str] = [target_skill]
    current_set: set[str] = {target_skill}

    for k in unique_sorted_scales:
        if k <= 1:
            chosen_by_scale[k] = (target_skill,)
            continue

        r = max(1, round((k - 1) * rivals_share))
        _extend_unique_up_to(current_chosen, current_set, ranked[:r], k)
        _extend_unique_up_to(current_chosen, current_set, filler_order, k)
        chosen_by_scale[k] = tuple(sorted(current_chosen[:k]))

    return [
        Catalog(
            id=f"sweep:{target_skill}:{max(1, raw_k)}",
            mode=CatalogMode.SWEEP,
            skills=chosen_by_scale[max(1, raw_k)],
            target=target_skill,
        )
        for raw_k in scales
    ]


def deduplicate_skills(skills: Sequence[Skill]) -> list[Skill]:
    """Filter duplicate skills by name, preserving first insertion order."""
    unique_skills: list[Skill] = []
    seen: set[str] = set()
    for s in skills:
        if s.name not in seen:
            seen.add(s.name)
            unique_skills.append(s)
    return unique_skills


_deduplicate_skills = deduplicate_skills


def _compute_cosine_bm25_distance_matrix(
    skills: Sequence[Skill],
    scorer: Scorer | None = None,
) -> tuple[tuple[str, ...], list[list[float]], list[list[float]]]:
    """Compute symmetric Cosine-BM25 distance and similarity matrices for a skills corpus."""
    unique_skills = _deduplicate_skills(skills)
    names = tuple(s.name for s in unique_skills)
    n = len(names)
    if n == 0:
        return (), [], []
    if n == 1:
        return names, [[0.0]], [[1.0]]

    bm25 = scorer if isinstance(scorer, Bm25Scorer) else Bm25Scorer.from_skills(unique_skills)
    tokens = [tuple(tokenize(skill_text(s))) for s in unique_skills]
    self_scores = [bm25.score(tokens[i], names[i]) for i in range(n)]

    dist: list[list[float]] = [[0.0] * n for _ in range(n)]
    sim: list[list[float]] = [[0.0] * n for _ in range(n)]
    for i in range(n):
        sim[i][i] = 1.0 if self_scores[i] > 0 else 0.0
        dist[i][i] = 0.0

    for i in range(n):
        for j in range(i + 1, n):
            s_ij = bm25.score(tokens[i], names[j])
            s_ji = bm25.score(tokens[j], names[i])
            denom = 2.0 * math.sqrt(max(0.0, self_scores[i] * self_scores[j]))
            s_norm = 0.0 if denom <= 0.0 else max(0.0, min(1.0, (s_ij + s_ji) / denom))
            d = max(0.0, min(1.0, 1.0 - s_norm))
            dist[i][j] = d
            dist[j][i] = d
            sim[i][j] = s_norm
            sim[j][i] = s_norm

    return names, dist, sim


def _farthest_first_traversal(
    dist: Sequence[Sequence[float]],
    initial_indices: Sequence[int],
    target_count: int,
) -> list[int]:
    """Expand an initial index cohort to target_count via greedy farthest-first selection."""
    n = len(dist)
    target = min(target_count, n)
    if not initial_indices:
        return []
    order = list(dict.fromkeys(initial_indices))
    if len(order) >= target:
        return order[:target]

    chosen_set = set(order)
    min_dist = [min(dist[i][c] for c in order) for i in range(n)]
    while len(order) < target:
        next_idx = max(
            (i for i in range(n) if i not in chosen_set),
            key=lambda i: (min_dist[i], -i),
        )
        chosen_set.add(next_idx)
        order.append(next_idx)
        for i in range(n):
            min_dist[i] = min(min_dist[i], dist[i][next_idx])
    return order


_NEAR_OPTIMAL_SIMILARITY_RATIO: float = 0.90
_DISPLAY_QUANTILE_WEIGHT: float = 0.15


def _compute_skill_display_widths(skills: Sequence[Skill]) -> list[int]:
    """Compute formatted prompt display character width for each skill."""
    return [len(f"- {s.name}: {s.description}") for s in skills]


def _extract_cluster_medoid_indices(
    partition_clusters: Sequence[Any],
    name_to_idx: Mapping[str, int],
    sim: Sequence[Sequence[float]],
    display_quantiles: Mapping[int, float] | None = None,
    near_optimal_ratio: float = _NEAR_OPTIMAL_SIMILARITY_RATIO,
    display_quantile_weight: float = _DISPLAY_QUANTILE_WEIGHT,
) -> list[int]:
    """Select the central medoid skill index from each partition cluster.

    Clusters are ordered by their median alphabetical display quantile so target
    quantiles align with natural cluster positions across prompt space.
    """
    valid_clusters = []
    for c in partition_clusters:
        c_indices = [name_to_idx[name] for name in c.skills if name in name_to_idx]
        if c_indices:
            valid_clusters.append(c_indices)

    if not valid_clusters:
        return []

    if display_quantiles is not None:

        def cluster_display_median(indices: list[int]) -> float:
            q_vals = sorted(display_quantiles.get(i, 0.5) for i in indices)
            return q_vals[len(q_vals) // 2]

        valid_clusters.sort(key=cluster_display_median)

    chosen_indices: list[int] = []
    chosen_set: set[int] = set()
    num_clusters = len(valid_clusters)

    for c_idx, c_indices in enumerate(valid_clusters):
        sim_scores = {i: sum(sim[i][j] for j in c_indices) for i in c_indices}
        max_sim = max(sim_scores.values()) if sim_scores else 0.0

        if display_quantiles is not None and max_sim > 0:
            target_q = (c_idx + 0.5) / max(1, num_clusters)
            min_acceptable_sim = near_optimal_ratio * max_sim
            candidates = [i for i in c_indices if sim_scores[i] >= min_acceptable_sim]
            if not candidates:
                candidates = c_indices

            best_idx = max(
                candidates,
                key=lambda i: (
                    sim_scores[i]
                    - display_quantile_weight * abs(display_quantiles.get(i, 0.5) - target_q),
                    sim_scores[i],
                    -i,
                ),
            )
        else:
            best_idx = max(c_indices, key=lambda i: (sim_scores[i], -i))

        if best_idx not in chosen_set:
            chosen_set.add(best_idx)
            chosen_indices.append(best_idx)
    return chosen_indices


def find_cluster_medoids(
    skills: Sequence[Skill],
    k: int,
    scorer: Scorer | None = None,
    *,
    near_optimal_ratio: float = _NEAR_OPTIMAL_SIMILARITY_RATIO,
    display_quantile_weight: float = _DISPLAY_QUANTILE_WEIGHT,
) -> tuple[str, ...]:
    """Find k representative skill medoids across modularity clusters.

    Partition skills into communities via modularity optimization, then select
    the central medoid skill from each cluster (maximizing intra-cluster BM25 similarity).
    If fewer than k clusters exist, iteratively select the farthest remaining
    skills from the chosen cohort to ensure maximal vocabulary diversity.

    Args:
        skills: The corpus of skills to partition and select from.
        k: The desired number of anchor medoid skills.
        scorer: Optional BM25 scorer for computing skill distances.
        near_optimal_ratio: Relative fraction of max cluster similarity to retain.
        display_quantile_weight: Penalty weight for deviations from prompt display quantiles.

    Returns:
        Tuple of up to k representative skill names.
    """
    if k <= 0 or not skills:
        return ()

    unique_skills = _deduplicate_skills(skills)
    names, dist, sim = _compute_cosine_bm25_distance_matrix(unique_skills, scorer=scorer)
    n = len(names)
    if n <= k:
        return names

    from reach.cluster import cluster_skills

    partition = cluster_skills(unique_skills, resolution=1.5, max_clusters=k)
    name_to_idx = {name: i for i, name in enumerate(names)}

    sorted_indices = sorted(range(n), key=lambda i: names[i])
    sorted_skills = [unique_skills[i] for i in sorted_indices]
    widths = _compute_skill_display_widths(sorted_skills)
    total_w = sum(widths) or 1
    cum_w = 0
    display_quantiles: dict[int, float] = {}
    for rank, idx in enumerate(sorted_indices):
        cum_w += widths[rank]
        display_quantiles[idx] = cum_w / total_w

    chosen_indices = _extract_cluster_medoid_indices(
        partition.clusters,
        name_to_idx,
        sim,
        display_quantiles=display_quantiles,
        near_optimal_ratio=near_optimal_ratio,
        display_quantile_weight=display_quantile_weight,
    )

    order = _farthest_first_traversal(dist, chosen_indices, min(k, n))
    return tuple(names[i] for i in order)


def _van_der_corput(n: int) -> float:
    """Compute base-2 Van der Corput radical inverse for positive integer n."""
    res = 0.0
    denom = 1.0
    while n > 0:
        denom *= 2.0
        res += (n % 2) / denom
        n //= 2
    return res


def _permute_by_van_der_corput(candidates: Sequence[int]) -> list[int]:
    """Deterministically permute candidate indices via Van der Corput radical inverse."""
    m = len(candidates)
    if m <= 1:
        return list(candidates)
    available = list(range(m))
    ordered: list[int] = []
    t = 1
    denom = max(1, m - 1)
    while available:
        target = _van_der_corput(t)
        best_pos = min(
            range(len(available)),
            key=lambda idx: (abs(available[idx] / denom - target), available[idx]),
        )
        ordered.append(candidates[available.pop(best_pos)])
        t += 1
    return ordered


def _low_discrepancy_striding(
    names: Sequence[str],
    sim: Sequence[Sequence[float]],
    anchor_indices: Sequence[int],
    skills: Sequence[Skill] | None = None,
) -> list[int]:
    """Order non-anchor skills via cluster-partitioned 2D low-discrepancy striding."""
    n = len(names)
    anchors = list(dict.fromkeys(anchor_indices))
    if len(anchors) >= n:
        return anchors[:n]

    anchor_set = set(anchors)
    non_anchors = [i for i in range(n) if i not in anchor_set]
    if not non_anchors:
        return anchors

    # Compute formatted display character width for tie-breaking
    if skills is not None and len(skills) == n:
        widths = _compute_skill_display_widths(skills)
    else:
        widths = [len(names[i]) for i in range(n)]

    # Partition non-anchors into nearest anchor clusters
    clusters_by_anchor: dict[int, list[int]] = {a: [] for a in anchors}
    for i in non_anchors:
        best_anchor = max(anchors, key=lambda a: (sim[i][a], -a))
        clusters_by_anchor[best_anchor].append(i)

    # Sort each cluster's candidate pool by similarity descending, then
    # display width descending, then alphabetically
    for a in anchors:
        clusters_by_anchor[a].sort(key=lambda i: (-sim[i][a], -widths[i], names[i]))

    from collections import deque

    # Interleave active clusters using Van der Corput radical inverse sequence
    active_anchors = [a for a in anchors if clusters_by_anchor[a]]
    cluster_order = _permute_by_van_der_corput(active_anchors)
    cluster_queues = {a: deque(clusters_by_anchor[a]) for a in cluster_order}

    interleaved: list[int] = []
    active = list(cluster_order)
    while active:
        next_active = []
        for a in active:
            interleaved.append(cluster_queues[a].popleft())
            if cluster_queues[a]:
                next_active.append(a)
        active = next_active

    return [*anchors, *interleaved]


def _build_scaling_sequence(
    names: tuple[str, ...],
    dist: list[list[float]],
    sim: list[list[float]],
    name_to_idx: Mapping[str, int],
    resolved_anchors: tuple[str, ...] | None,
    skills: Sequence[Skill] | None = None,
) -> tuple[str, ...]:
    """Determine complete scaling order across unique skills."""
    n = len(names)
    if n <= 1:
        return names
    if resolved_anchors:
        initial = [name_to_idx[a] for a in resolved_anchors]
        order = _low_discrepancy_striding(names, sim, initial, skills=skills)
        return tuple(names[i] for i in order)

    medoid_idx = max(range(n), key=lambda i: (sum(sim[i]), -i))
    order = _farthest_first_traversal(dist, [medoid_idx], n)
    return tuple(names[i] for i in order)


def _build_nested_catalogs(seq: tuple[str, ...], scales: Sequence[int]) -> list[Catalog]:
    """Construct nested Catalog instances corresponding to requested sweep scale counts."""
    catalogs: list[Catalog] = []
    for k in scales:
        count = max(1, min(k, len(seq)))
        chosen = seq[:count]
        catalogs.append(
            Catalog(
                id=f"sweep:corpus:{count}",
                mode=CatalogMode.SWEEP,
                skills=tuple(sorted(chosen)),
                target=None,
            )
        )
    return catalogs


class CorpusScalingPlan(BaseModel):
    """Encapsulate precomputed distance geometry and nested catalogs for a scaling sweep."""

    model_config = ConfigDict(frozen=True)

    skills: tuple[Skill, ...]
    skill_names: tuple[str, ...]
    distance_matrix: tuple[tuple[float, ...], ...]
    similarity_matrix: tuple[tuple[float, ...], ...]
    sequence: tuple[str, ...]
    catalogs: tuple[Catalog, ...]
    anchor_skills: tuple[str, ...] | None = None

    @classmethod
    def create(
        cls,
        skills: Sequence[Skill],
        scales: Sequence[int],
        anchor_skills: Sequence[str] | None = None,
        scorer: Scorer | None = None,
    ) -> CorpusScalingPlan:
        """Construct a scaling plan by computing distance geometry and k-Center ordering once."""
        unique_skills = _deduplicate_skills(skills)
        names, dist, sim = _compute_cosine_bm25_distance_matrix(unique_skills, scorer=scorer)
        name_to_idx = {name: i for i, name in enumerate(names)}

        resolved_anchors: tuple[str, ...] | None = None
        if anchor_skills is not None:
            valid_anchors = tuple(dict.fromkeys(a for a in anchor_skills if a in name_to_idx))
            if valid_anchors:
                resolved_anchors = valid_anchors

        seq = _build_scaling_sequence(
            names, dist, sim, name_to_idx, resolved_anchors, skills=unique_skills
        )
        catalogs = _build_nested_catalogs(seq, scales)

        return cls(
            skills=tuple(unique_skills),
            skill_names=names,
            distance_matrix=tuple(tuple(row) for row in dist),
            similarity_matrix=tuple(tuple(row) for row in sim),
            sequence=seq,
            catalogs=tuple(catalogs),
            anchor_skills=resolved_anchors,
        )

    def queries_for_scale(
        self,
        catalog: Catalog,
        raw_query_set: QuerySet,
        anchor_skills: Sequence[str] | None = None,
    ) -> QuerySet:
        """Slice query set into in-scope reachability probes for catalog."""
        effective_anchors = anchor_skills if anchor_skills is not None else self.anchor_skills
        return build_corpus_scaling_queries(
            scale_skills=catalog.skills,
            raw_query_set=raw_query_set,
            anchor_skills=effective_anchors,
        )


def build_corpus_scaling_sequence(
    skills: Sequence[Skill],
    anchor_skills: Sequence[str] | None = None,
    scorer: Scorer | None = None,
) -> tuple[str, ...]:
    """Order skills using Farthest-First Traversal (k-Center) on Cosine-BM25 distance."""
    plan = CorpusScalingPlan.create(
        skills=skills,
        scales=(),
        anchor_skills=anchor_skills,
        scorer=scorer,
    )
    return plan.sequence


def build_corpus_scaling_catalogs(
    skills: Sequence[Skill],
    scales: Sequence[int],
    ordered_names: Sequence[str] | None = None,
    anchor_skills: Sequence[str] | None = None,
    scorer: Scorer | None = None,
) -> list[Catalog]:
    """Generate deterministic nested catalogs for whole-corpus capacity evaluation."""
    if not skills:
        return []

    if ordered_names is not None:
        catalogs: list[Catalog] = []
        for k in scales:
            count = max(1, min(k, len(ordered_names)))
            chosen = ordered_names[:count]
            catalogs.append(
                Catalog(
                    id=f"sweep:corpus:{count}",
                    mode=CatalogMode.SWEEP,
                    skills=tuple(sorted(chosen)),
                    target=None,
                )
            )
        return catalogs

    plan = CorpusScalingPlan.create(
        skills=skills,
        scales=scales,
        anchor_skills=anchor_skills,
        scorer=scorer,
    )
    return list(plan.catalogs)


def build_corpus_scaling_queries(
    scale_skills: Sequence[str],
    raw_query_set: QuerySet,
    anchor_skills: Sequence[str] | None = None,
) -> QuerySet:
    """Slice query set into in-scope reachability probes for installed skills."""
    scale_set = set(scale_skills)
    target_skills = scale_set & set(anchor_skills) if anchor_skills is not None else scale_set
    in_scope_queries = [q for q in raw_query_set.queries if q.expected_skill in target_skills]
    return QuerySet(
        catalog_id=f"sweep:corpus:{len(scale_skills)}",
        queries=tuple(in_scope_queries),
        provenance=raw_query_set.provenance,
    )


def build_neighborhood_catalogs(
    skills: Sequence[Skill],
    size: int = 20,
    rivals: int = 10,
    seed: int = 0,
    scorer: Scorer | None = None,
) -> list[Catalog]:
    """Generate fixed-size catalogs per skill containing target, rivals, and filler."""
    if not skills:
        return []
    if size < MIN_NEIGHBORHOOD_SIZE:
        msg = f"a neighborhood needs at least 2 skills, got {size}"
        raise ValueError(msg)
    if rivals < 1:
        msg = f"a neighborhood needs at least 1 rival, got {rivals}"
        raise ValueError(msg)

    by_name = {s.name: s for s in skills}
    unique_skills = sorted(by_name.values(), key=lambda s: s.name)
    ranker = scorer or Bm25Scorer.from_skills(unique_skills)
    catalogs = []
    for skill in unique_skills:
        ranked: list[str] = []
        seen: set[str] = {skill.name}
        for name, _ in ranker.rank(skill, unique_skills):
            if name not in seen:
                seen.add(name)
                ranked.append(name)

        chosen = [skill.name, *ranked[:rivals]]

        rng = Random(f"{seed}:{skill.name}")  # noqa: S311 (deterministic benchmark sampling)
        chosen_set = set(chosen)
        pool = [s for s in ranked[rivals:] if s not in chosen_set]
        rng.shuffle(pool)
        chosen.extend(pool[: max(0, size - len(chosen))])

        catalogs.append(
            Catalog(
                id=f"neighborhood:{skill.name}",
                mode=CatalogMode.NEIGHBORHOOD,
                skills=tuple(sorted(chosen)),
                target=skill.name,
            ),
        )
    return catalogs


def resolve_catalog(catalogs: Sequence[Catalog], catalog_id: str) -> Catalog:
    """Retrieve a catalog by identifier from a sequence of catalogs."""
    for catalog in catalogs:
        if catalog.id == catalog_id:
            return catalog
    available = ", ".join(sorted(c.id for c in catalogs)[:8]) or "(none)"
    hint = ""
    if any(c.id.startswith("neighborhood:") for c in catalogs):
        hint = "; pass --catalog <name> --rescope to evaluate against an available catalog"
    msg = f"no catalog named {catalog_id!r}; available: {available}{hint}"
    raise KeyError(msg)


def resident_skills(catalog: Catalog, skills: Sequence[Skill]) -> list[Skill]:
    """Retrieve ordered Skill objects resident in the specified catalog."""
    by_name = {s.name: s for s in skills}
    missing = [name for name in catalog.skills if name not in by_name]
    if missing:
        msg = f"catalog {catalog.id!r} names skills not loaded: {missing}"
        raise KeyError(msg)
    return [by_name[name] for name in catalog.skills]


class ResolvedTarget(BaseModel):
    """Structured resolution of a user-specified skill target and optional catalog."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    skill_name: str
    catalog_path: Path | None = None
    manifest_path: Path | None = None


def _infer_parent_catalog(skill_dir: Path) -> Path:
    """Infer catalog root directory containing resident and rival skills."""
    resolved_dir = skill_dir.resolve()
    parent = resolved_dir.parent
    home = Path.home().resolve()
    if parent in (home, parent.parent):
        return resolved_dir

    if parent.name in ("skills", ".skills"):
        return parent

    with contextlib.suppress(OSError):
        peer_skills = [d for d in parent.iterdir() if d.is_dir() and (d / "SKILL.md").is_file()]
        if len(peer_skills) > 1:
            return parent

    return resolved_dir


def _resolve_manifest_file(
    named: Path,
    catalog_path: Path | None,
    *,
    looks_like_path: bool,
    target: str | Path,
    raw_str: str,
) -> ResolvedTarget:
    if named.name != "SKILL.md":
        if not looks_like_path:
            return ResolvedTarget(skill_name=raw_str, catalog_path=catalog_path)
        msg = (
            f"'{target}' is not a SKILL.md file; "
            f"expected a SKILL.md file or skill directory containing one"
        )
        raise ValueError(msg)
    skill = parse_frontmatter(named.read_text(encoding="utf-8"), named)
    if skill is None or not skill.name:
        msg = f"'{named}' does not contain valid YAML frontmatter"
        raise ValueError(msg)
    catalog = catalog_path or _infer_parent_catalog(named.parent)
    return ResolvedTarget(skill_name=skill.name, catalog_path=catalog, manifest_path=named)


def _resolve_skill_directory(
    named: Path,
    catalog_path: Path | None,
    *,
    looks_like_path: bool,
    target: str | Path,
    raw_str: str,
    command_name: str,
) -> ResolvedTarget:
    manifest = named / "SKILL.md"
    if manifest.is_file():
        skill = parse_frontmatter(manifest.read_text(encoding="utf-8"), manifest)
        if skill is None or not skill.name:
            msg = f"'{manifest}' does not contain valid YAML frontmatter"
            raise ValueError(msg)
        catalog = catalog_path or _infer_parent_catalog(named)
        return ResolvedTarget(skill_name=skill.name, catalog_path=catalog, manifest_path=manifest)

    contained_skills = sorted(
        d.name for d in named.iterdir() if d.is_dir() and (d / "SKILL.md").is_file()
    )
    max_preview = 3
    if contained_skills:
        preview = ", ".join(f"'{s}'" for s in contained_skills[:max_preview])
        more = (
            f" (and {len(contained_skills) - max_preview} more)"
            if len(contained_skills) > max_preview
            else ""
        )
        label = "skill" if len(contained_skills) == 1 else "skills"
        hint = (
            f"reach {command_name} --skill {raw_str.rstrip('/')}/{contained_skills[0]}"
            if "explain" in command_name
            else f"reach {command_name} {raw_str.rstrip('/')}/{contained_skills[0]}"
        )
        msg = (
            f"'{target}' is a directory containing {len(contained_skills)} {label} "
            f"({preview}{more}), not a single skill.\n\n"
            f"• To {command_name} a single skill immediately:\n"
            f"    {hint}"
        )
        raise ValueError(msg)

    if looks_like_path:
        msg = f"directory '{target}' does not contain a SKILL.md file"
        raise ValueError(msg)

    # Fall back to raw name if an unrelated cwd directory matched target
    return ResolvedTarget(skill_name=raw_str, catalog_path=catalog_path)


def resolve_skill_target(
    target: str | Path | None,
    explicit_catalog: Path | str | None = None,
    *,
    command_name: str = "eval",
) -> ResolvedTarget | None:
    """Resolve a skill name and catalog path from a name, directory, or SKILL.md file.

    Args:
        target: Skill name, directory path, or SKILL.md file path.
        explicit_catalog: Explicit catalog path if specified by user flag (e.g. --skills).
        command_name: CLI command name for formatting multi-skill error remedies.

    Returns:
        ResolvedTarget with canonical skill_name and inferred catalog_path,
        or None if target is None.

    Raises:
        FileNotFoundError: If target looks like a path but does not exist on disk.
        ValueError: If target is a non-SKILL.md file, a directory containing skills,
            a directory with no SKILL.md, or a SKILL.md with invalid frontmatter.
    """
    if not target or not (raw_str := str(target).strip()):
        return None

    catalog_path = resolve_path(explicit_catalog) if explicit_catalog else None

    looks_like_path = (
        isinstance(target, Path)
        or "/" in raw_str
        or "\\" in raw_str
        or raw_str.startswith(("~", "."))
    )

    # If an explicit catalog was provided and the target is a raw name, don't probe cwd
    if catalog_path is not None and not looks_like_path:
        return ResolvedTarget(skill_name=raw_str, catalog_path=catalog_path)

    named = resolve_path(target)

    # 1. Path does not exist
    if not named.exists():
        if looks_like_path:
            msg = f"skill path does not exist: '{target}'"
            raise FileNotFoundError(msg)
        return ResolvedTarget(skill_name=raw_str, catalog_path=catalog_path)

    # 2. Path is a file
    if named.is_file():
        return _resolve_manifest_file(
            named,
            catalog_path,
            looks_like_path=looks_like_path,
            target=target,
            raw_str=raw_str,
        )

    # 3. Path is a directory
    if named.is_dir():
        return _resolve_skill_directory(
            named,
            catalog_path,
            looks_like_path=looks_like_path,
            target=target,
            raw_str=raw_str,
            command_name=command_name,
        )

    return None
