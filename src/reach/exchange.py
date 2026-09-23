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

"""Import and export evaluation query sets to and from CSV and JSONL formats."""

from __future__ import annotations

import csv
import io
import json
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from reach.models import Query, QueryKind
from reach.queries import Origin, QuerySet, QuerySetProvenance


class Exchange(StrEnum):
    """Enumerate supported row-oriented query exchange formats."""

    CSV = "csv"
    JSONL = "jsonl"


#: Standard field names for tabular query export.
FIELDS = ("id", "text", "kind", "expected_skill", "acceptable_skills", "notes")

#: Mapping of file extensions to exchange format types.
SUFFIXES = {".csv": Exchange.CSV, ".jsonl": Exchange.JSONL}

#: UTF-8 Byte Order Mark character.
BOM = "\ufeff"

#: Maximum number of unusable rows listed in validation error messages.
LISTED = 20


class FieldMap(BaseModel):
    """Map source column or property names to standard query set fields."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = "id"
    text: str = "text"
    kind: str = "kind"
    expected_skill: str = "expected_skill"
    acceptable_skills: str = "acceptable_skills"
    notes: str = "notes"
    separator: str = ","
    id_prefix: str = "q"

    @field_validator("separator")
    @classmethod
    def _require_separator(cls, value: str) -> str:
        """Ensure multi-skill cells have a non-empty delimiter."""
        if not value:
            msg = "separator must not be empty"
            raise ValueError(msg)
        return value

    def columns(self) -> tuple[str, ...]:
        """Return the mapped source column names in canonical field order."""
        return tuple(getattr(self, field) for field in FIELDS)

    def named(self) -> tuple[str, ...]:
        """Return all explicitly specified (non-default) column names."""
        return tuple(getattr(self, field) for field in FIELDS if field in self.model_fields_set)


class SourceRow(BaseModel):
    """Validate raw row values from an external query source."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = ""
    text: str
    kind: QueryKind | None = None
    expected_skill: str | None = None
    acceptable_skills: tuple[str, ...] = ()
    notes: str = ""

    @field_validator("text")
    @classmethod
    def _require_a_query(cls, value: str) -> str:
        """Validate that query text is not empty or whitespace."""
        if not value.strip():
            msg = "text is blank"
            raise ValueError(msg)
        return value

    @field_validator("kind", "expected_skill", mode="before")
    @classmethod
    def _empty_is_absent(cls, value: object) -> object:
        """Convert empty strings to None for optional fields."""
        if isinstance(value, str) and not value.strip():
            return None
        return value

    def as_query(self, fallback_id: str) -> Query:
        """Construct a validated Query model from the row data."""
        return Query(
            id=self.id.strip() or fallback_id,
            text=self.text,
            kind=self.kind,
            expected_skill=self.expected_skill,
            acceptable_skills=self.acceptable_skills,
            notes=self.notes,
        )


def infer_format(path: Path | str) -> Exchange:
    """Infer exchange format from file extension, raising ValueError if unrecognized."""
    suffix = Path(path).suffix.lower()
    if suffix not in SUFFIXES:
        msg = (
            f"cannot tell what format {Path(path).name} is from its name; "
            f"pass --format, or use one of {', '.join(sorted(SUFFIXES))}"
        )
        raise ValueError(
            msg,
        )
    return SUFFIXES[suffix]


def export_query_set(
    query_set: QuerySet,
    fmt: Exchange,
    mapping: FieldMap | None = None,
) -> str:
    """Serialize a QuerySet into CSV or JSONL format using the given field mapping."""
    mapping = mapping or FieldMap()
    rows = [_as_row(query, mapping, fmt) for query in query_set.queries]
    if fmt is Exchange.JSONL:
        return "".join(json.dumps(row) + "\n" for row in rows)
    buffer = io.StringIO()
    writer = csv.DictWriter(
        buffer,
        fieldnames=list(mapping.columns()),
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def _as_row(query: Query, mapping: FieldMap, fmt: Exchange) -> dict[str, Any]:
    """Convert a Query model into a mapped dictionary row."""
    return {
        mapping.id: query.id,
        mapping.text: query.text,
        mapping.kind: query.kind.value if query.kind else "",
        mapping.expected_skill: query.expected_skill or "",
        mapping.acceptable_skills: (
            list(query.acceptable_skills)
            if fmt is Exchange.JSONL
            else mapping.separator.join(query.acceptable_skills)
        ),
        mapping.notes: query.notes,
    }


def import_query_set(
    document: str,
    fmt: Exchange,
    *,
    catalog_id: str,
    mapping: FieldMap | None = None,
    notes: str = "",
    source: str = "",
) -> QuerySet:
    """Parse a CSV or JSONL document into a QuerySet model for a specific catalog."""
    mapping = mapping or FieldMap()
    where = source or "input"
    rows = _read_rows(document, fmt, where)
    _assert_columns(rows[0], mapping, where)
    queries = _as_queries(rows, mapping, where)
    return QuerySet(
        catalog_id=catalog_id,
        notes=notes,
        provenance=QuerySetProvenance(origin=Origin.IMPORTED, source=source),
        queries=queries,
    )


def _read_rows(document: str, fmt: Exchange, where: str) -> list[dict[str, Any]]:
    """Parse CSV or JSONL text into row dictionaries, stripping BOM markers."""
    rows = (
        _read_jsonl(document, where)
        if fmt is Exchange.JSONL
        else list(csv.DictReader(io.StringIO(document.removeprefix(BOM))))
    )
    if not rows:
        msg = f"{where} has no rows to import"
        raise ValueError(msg)
    return rows


def _read_jsonl(document: str, where: str) -> list[dict[str, Any]]:
    """Parse JSON Lines formatted text into row dictionaries."""
    rows = []
    for number, line in enumerate(document.removeprefix(BOM).splitlines(), start=1):
        if not line.strip():
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError as error:
            msg = f"{where} row {number}: not JSON: {error}"
            raise ValueError(msg) from error
        if not isinstance(parsed, dict):
            msg = f"{where} row {number}: expected an object, got {type(parsed).__name__}"
            raise ValueError(
                msg,
            )
        rows.append(parsed)
    return rows


def _assert_columns(first: dict[str, Any], mapping: FieldMap, where: str) -> None:
    """Validate that required mapped column names exist in the first row."""
    present = tuple(first)
    wanted = {mapping.text, *mapping.named()}
    missing = sorted(column for column in wanted if column not in present)
    if missing:
        msg = (
            f"{where} has no column named {', '.join(repr(m) for m in missing)}; "
            f"it has {', '.join(repr(p) for p in present)}"
        )
        raise ValueError(
            msg,
        )


def _as_queries(
    rows: list[dict[str, Any]],
    mapping: FieldMap,
    where: str,
) -> tuple[Query, ...]:
    """Validate all rows and convert them into Query objects, collecting any errors."""
    queries: list[Query] = []
    unusable: list[str] = []
    for number, row in enumerate(rows, start=1):
        try:
            queries.append(_as_query(row, mapping, number))
        except (ValidationError, ValueError) as error:
            unusable.append(f"row {number}: {_why(error)}")
    if not unusable:
        return tuple(queries)
    if len(unusable) == 1:
        msg = f"{where} {unusable[0]}"
        raise ValueError(msg)
    listed = [f"  {refusal}" for refusal in unusable[:LISTED]]
    if (over := len(unusable) - LISTED) > 0:
        listed.append(f"  ... and {over} more")
    raise ValueError("\n".join([f"{where}: {len(unusable)} unusable rows", *listed]))


def _as_query(row: dict[str, Any], mapping: FieldMap, number: int) -> Query:
    """Validate an individual row dictionary and construct a Query."""
    return SourceRow.model_validate(
        {
            "id": _text(row.get(mapping.id, "")),
            "text": _text(row.get(mapping.text, "")),
            "kind": row.get(mapping.kind),
            "expected_skill": row.get(mapping.expected_skill),
            "acceptable_skills": _skills(
                row.get(mapping.acceptable_skills),
                mapping.separator,
            ),
            "notes": _text(row.get(mapping.notes, "")),
        },
    ).as_query(f"{mapping.id_prefix}-{number}")


def _why(error: Exception) -> str:
    """Format a validation exception into a concise single-line error message."""
    if not isinstance(error, ValidationError):
        return str(error)
    reasons = []
    for failure in error.errors():
        field = ".".join(str(part) for part in failure["loc"])
        reason = failure["msg"].removeprefix("Value error, ")
        reasons.append(reason if not field or field in reason else f"{field}: {reason}")
    return "; ".join(reasons)


def _text(value: object) -> str:
    """Convert input value to string or empty string if None."""
    return "" if value is None else str(value)


def _skills(value: object, separator: str) -> tuple[str, ...]:
    """Normalize a delimited cell or JSON array into skill names."""
    if value is None:
        return ()
    if isinstance(value, str):
        return tuple(skill for part in value.split(separator) if (skill := part.strip()))
    if isinstance(value, (list, tuple)):
        skills: list[str] = []
        for part in value:
            if not isinstance(part, str):
                msg = (
                    "acceptable_skills array elements must be strings, "
                    f"got {type(part).__name__}"
                )
                raise ValueError(msg)
            if skill := part.strip():
                skills.append(skill)
        return tuple(skills)
    msg = f"acceptable_skills must be a delimited string or array, got {type(value).__name__}"
    raise ValueError(msg)
