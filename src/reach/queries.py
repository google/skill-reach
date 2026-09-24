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

"""Load, save, and validate labeled evaluation query sets and calculate digests."""

import hashlib
import json
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Self

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    model_validator,
)

from reach._io import write_model
from reach.models import Query, QueryKind

if TYPE_CHECKING:
    from reach.exchange import FieldMap
    from reach.models import Skill

__all__ = [
    "Origin",
    "QuerySet",
    "QuerySetProvenance",
    "SkillDigestHex",
    "SkillNameKey",
    "format_skill_sample",
    "format_sync_counts",
    "load_query_set",
    "query_set_digest",
    "save_query_set",
]

SkillNameKey = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
SkillDigestHex = Annotated[
    str, StringConstraints(strip_whitespace=True, to_lower=True, min_length=1)
]


def format_sync_counts(missing_count: int, stale_count: int) -> str:
    """Format human-readable missing and updated skill count breakdown."""
    parts: list[str] = []
    if missing_count:
        parts.append(f"{missing_count} missing")
    if stale_count:
        parts.append(f"{stale_count} updated")
    return ", ".join(parts)


def format_skill_sample(names: Sequence[str], *, limit: int = 5) -> str:
    """Format a truncated comma-separated sample of skill names."""
    sample = ", ".join(names[:limit])
    more = f", +{len(names) - limit} more" if len(names) > limit else ""
    return f"{sample}{more}"


def _is_generated_for_skill(query: Query, skill_names: frozenset[str]) -> bool:
    """Return True when a positive or adversarial query belongs to one of `skill_names`."""
    if query.expected_skill in skill_names:
        return True
    adv_prefixes = tuple(f"adv-{name}-" for name in skill_names)
    return bool(adv_prefixes) and query.id.startswith(adv_prefixes)


class Origin(StrEnum):
    """Enumerate origins of query sets."""

    AUTHORED = "authored"
    GENERATED = "generated"
    IMPORTED = "imported"


class QuerySetProvenance(BaseModel):
    """Record creation metadata, generator settings, per-skill SHA digests, and review status.

    `skill_digests` maps each drafted skill name to a 12-character SHA-256 digest of its
    `SKILL.md` body (excluding YAML frontmatter `description` so description tuning does
    not invalidate benchmark queries). `reach query draft --sync` and `reach sweep`
    compare `skill_digests` against the workspace corpus to detect updated skills.
    """

    model_config = ConfigDict(frozen=True)

    origin: Origin = Origin.AUTHORED
    recorded_at: AwareDatetime = Field(default_factory=lambda: datetime.now(UTC))
    tool_version: str = ""
    generator_model: str = ""
    generator_arm: str = ""
    queries_per_target: int | None = None
    rivals_in_view: int | None = None
    bodies_digest: str = ""
    skill_digests: dict[SkillNameKey, SkillDigestHex] = Field(default_factory=dict)
    config_fingerprint: str = ""
    source: str = ""
    reviewed: bool | None = None
    adversarial: bool | None = None
    adversarial_per_target: int | None = None

    def with_updated_digests(
        self,
        skills: Sequence["Skill"],
        *,
        drafted_targets: Iterable[str] = (),
        covered_targets: Iterable[str] = (),
        bodies_hash: str | None = None,
        extra_updates: dict[str, object] | None = None,
    ) -> Self:
        """Return a validated copy with updated `skill_digests` for `drafted_targets`."""
        from reach.generate import bodies_digest, skill_body_digest

        drafted_set = frozenset(drafted_targets)
        covered_set = frozenset(covered_targets) | drafted_set
        merged_digests = dict(self.skill_digests)
        for skill in skills:
            if skill.name in drafted_set or (
                skill.name in covered_set and skill.name not in merged_digests
            ):
                merged_digests[skill.name] = skill_body_digest(skill)

        payload = self.model_dump()
        if extra_updates:
            payload.update(extra_updates)
        payload["skill_digests"] = dict(sorted(merged_digests.items()))
        payload["bodies_digest"] = bodies_hash if bodies_hash is not None else bodies_digest(skills)
        return type(self).model_validate(payload)


class QuerySet(BaseModel):
    """Represent a collection of labeled evaluation queries targeting a catalog."""

    model_config = ConfigDict(frozen=True)

    catalog_id: str = ""
    queries: tuple[Query, ...]
    notes: str = ""
    provenance: QuerySetProvenance = Field(default_factory=QuerySetProvenance)

    @model_validator(mode="before")
    @classmethod
    def _default_missing_query_ids(cls, data: object) -> object:
        """Assign sequential default IDs (`q-001`, ...) to query entries that omit `id`."""
        if isinstance(data, dict) and isinstance(data.get("queries"), (list, tuple)):
            normalized_queries = [
                {**q, "id": f"q-{idx:03d}"}
                if isinstance(q, dict) and not str(q.get("id") or "").strip()
                else q
                for idx, q in enumerate(data["queries"], start=1)
            ]
            return {**data, "queries": normalized_queries}
        return data

    @model_validator(mode="after")
    def _assert_unique_ids(self) -> Self:
        """Validate that all query IDs within the query set are unique."""
        seen: set[str] = set()
        duplicates: set[str] = set()
        for q in self.queries:
            if q.id in seen:
                duplicates.add(q.id)
            seen.add(q.id)
        if duplicates:
            msg = f"duplicate query ids: {sorted(duplicates)}"
            raise ValueError(msg)
        return self

    def for_skill(self, name: str) -> tuple[Query, ...]:
        """Return queries whose expected skill or truth label matches the specified name."""
        return tuple(q for q in self.queries if name in (q.expected_skill, q.truth_label))

    def covered_skills(self) -> frozenset[str]:
        """Return target skill names covered by positive benchmark queries."""
        return frozenset(
            q.expected_skill
            for q in self.queries
            if q.expected_skill is not None and q.kind != QueryKind.NEIGHBOR_NEGATIVE
        )

    def stale_skills(self, skills: Sequence["Skill"]) -> frozenset[str]:
        """Return covered skill names whose markdown body differs from recorded provenance SHA."""
        from reach.generate import skill_body_digest

        covered = self.covered_skills()
        recorded = self.provenance.skill_digests
        if not recorded:
            return frozenset()
        return frozenset(
            s.name
            for s in skills
            if s.name in covered and s.name in recorded and recorded[s.name] != skill_body_digest(s)
        )

    def out_of_sync_skills(
        self,
        skills: Sequence["Skill"],
    ) -> tuple[frozenset[str], frozenset[str]]:
        """Return (missing_skills, stale_skills) relative to the provided skill corpus."""
        covered = self.covered_skills()
        missing = frozenset(s.name for s in skills if s.name not in covered)
        stale = self.stale_skills(skills)
        return missing, stale

    def without_skills(self, skill_names: Iterable[str]) -> Self:
        """Return a copy with positive and adversarial queries for `skill_names` removed."""
        target_set = frozenset(skill_names)
        if not target_set:
            return self
        retained = tuple(q for q in self.queries if not _is_generated_for_skill(q, target_set))
        return self.model_copy(update={"queries": retained})

    def with_updated_digests(
        self,
        skills: Sequence["Skill"],
        *,
        drafted_targets: Iterable[str] = (),
        covered_targets: Iterable[str] = (),
        bodies_hash: str | None = None,
        extra_updates: dict[str, object] | None = None,
    ) -> Self:
        """Return a copy with `provenance.skill_digests` updated for `drafted_targets`."""
        effective_covered = frozenset(covered_targets) or self.covered_skills()
        updated_prov = self.provenance.with_updated_digests(
            skills,
            drafted_targets=drafted_targets,
            covered_targets=effective_covered,
            bodies_hash=bodies_hash,
            extra_updates=extra_updates,
        )
        return self.model_copy(update={"provenance": updated_prov})


def _apply_catalog_id_fallback(parsed: QuerySet, catalog_id: str) -> QuerySet:
    """Populate catalog_id on a parsed QuerySet when omitted in the source payload."""
    if not parsed.catalog_id and catalog_id:
        return parsed.model_copy(update={"catalog_id": catalog_id})
    return parsed


def _parse_json_query_set(
    content: str,
    *,
    catalog_id: str,
) -> QuerySet:
    """Parse and validate a QuerySet JSON payload."""
    return _apply_catalog_id_fallback(QuerySet.model_validate_json(content), catalog_id)


def load_query_set(
    path: Path | str,
    *,
    catalog_id: str = "all",
) -> QuerySet:
    """Load and validate a QuerySet from a JSON, YAML, JSONL, or CSV file."""
    from reach.config import resolve_path

    resolved = resolve_path(path)
    content = resolved.read_text(encoding="utf-8")
    suffix = resolved.suffix.lower()
    if suffix in (".jsonl", ".csv"):
        from reach.exchange import Exchange, import_query_set

        fmt = Exchange.JSONL if suffix == ".jsonl" else Exchange.CSV
        return import_query_set(content, fmt, catalog_id=catalog_id, source=str(resolved))
    if suffix in (".yaml", ".yml"):
        import yaml

        return _apply_catalog_id_fallback(
            QuerySet.model_validate(yaml.safe_load(content)), catalog_id
        )
    if suffix == ".json":
        return _parse_json_query_set(content, catalog_id=catalog_id)
    try:
        return _parse_json_query_set(content, catalog_id=catalog_id)
    except (ValueError, ValidationError):
        from reach.exchange import Exchange, import_query_set

        return import_query_set(
            content, Exchange.JSONL, catalog_id=catalog_id, source=str(resolved)
        )


def save_query_set(
    query_set: QuerySet,
    path: Path | str,
    *,
    fmt: str | None = None,
    mapping: "FieldMap | None" = None,
) -> Path:
    """Serialize a QuerySet instance to disk in JSON, JSONL, or CSV format."""
    resolved = Path(path).expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    target_fmt = fmt or (
        "jsonl"
        if resolved.suffix.lower() == ".jsonl"
        else "csv"
        if resolved.suffix.lower() == ".csv"
        else "json"
    )
    if target_fmt == "jsonl":
        from reach.exchange import Exchange, export_query_set

        resolved.write_text(
            export_query_set(query_set, Exchange.JSONL, mapping=mapping),
            encoding="utf-8",
        )
        return resolved
    if target_fmt == "csv":
        from reach.exchange import Exchange, export_query_set

        resolved.write_text(
            export_query_set(query_set, Exchange.CSV, mapping=mapping),
            encoding="utf-8",
        )
        return resolved
    return write_model(query_set, resolved)


def query_set_digest(query_set: QuerySet) -> str:
    """Compute a deterministic 12-character SHA-256 digest of query set content."""
    return _digest_queries(query_set.catalog_id, _query_rows(query_set))


def _query_rows(query_set: QuerySet) -> list[tuple[str, ...]]:
    """Convert a query set into canonical row tuples for digest calculation."""
    rows: list[tuple[str, ...]] = []
    for query in query_set.queries:
        row: tuple[str, ...] = (
            query.id,
            query.text,
            query.kind or "",
            query.expected_skill or "",
        )
        if query.acceptable_skills:
            acceptable = json.dumps(
                sorted(query.acceptable_skills),
                separators=(",", ":"),
            )
            row += (acceptable,)
        rows.append(row)
    return rows


def _digest_queries(catalog_id: str, rows: Iterable[tuple[str, ...]]) -> str:
    """Compute a 12-character SHA-256 digest over sorted query rows and catalog ID."""
    material = "\n".join("\t".join(row) for row in sorted(rows, key=lambda row: row[0]))
    canonical = f"{catalog_id}\n{material}"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]
