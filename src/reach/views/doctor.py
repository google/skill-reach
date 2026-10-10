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

"""Format and render environment and runtime diagnostic check reports."""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from typing import TYPE_CHECKING, Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from reach.models import NonEmptyStr
from reach.rendering import csv_document, dispatch_render

if TYPE_CHECKING:
    from reach.views.base import Console

__all__ = [
    "DOCTOR_RENDERERS",
    "CheckCategory",
    "CheckResult",
    "CheckRowTuple",
    "CheckStatus",
    "DoctorReport",
    "render_doctor",
    "render_doctor_csv",
    "render_doctor_json",
    "render_doctor_jsonl",
    "render_doctor_table",
]

type CheckRowTuple = tuple[str, str, str, str, str]


class CheckStatus(StrEnum):
    """Enumerate diagnostic health check outcomes."""

    OK = "ok"
    WARN = "warn"
    FAIL = "fail"


class CheckCategory(StrEnum):
    """Enumerate canonical diagnostic categories for reach doctor."""

    ENVIRONMENT = "Environment"
    RUNTIMES = "Runtimes"
    CREDENTIALS = "Credentials"
    SKILLS = "Skills"
    CONFIGURATION = "Configuration"


class CheckResult(BaseModel):
    """Represent the validated diagnostic result of a single environment check."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    category: Annotated[
        NonEmptyStr,
        Field(description="Diagnostic domain category."),
    ]
    name: Annotated[
        NonEmptyStr,
        Field(description="Human-readable name of the inspected component."),
    ]
    status: Annotated[
        CheckStatus,
        Field(description="Health status of the check (ok, warn, fail)."),
    ]
    detail: Annotated[
        NonEmptyStr,
        Field(description="Observed diagnostic details."),
    ]
    remedy: Annotated[
        str,
        StringConstraints(strip_whitespace=True),
        Field(description="Recommended remediation action when status is warn or fail."),
    ] = ""


class DoctorReport(BaseModel):
    """Aggregate diagnostic check results for serialization and rendering."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    checks: Annotated[
        tuple[CheckResult, ...],
        Field(description="Ordered sequence of diagnostic check results."),
    ] = ()

    @property
    def has_failures(self) -> bool:
        """Return True if any check recorded a FAIL status."""
        return any(c.status is CheckStatus.FAIL for c in self.checks)

    @property
    def has_warnings(self) -> bool:
        """Return True if any check recorded a WARN status."""
        return any(c.status is CheckStatus.WARN for c in self.checks)

    @classmethod
    def coerce(
        cls,
        items: DoctorReport | Sequence[CheckResult | CheckRowTuple],
    ) -> DoctorReport:
        """Normalize a DoctorReport or sequence of CheckResults/5-tuples into a DoctorReport."""
        if isinstance(items, DoctorReport):
            return items
        normalized: list[CheckResult] = []
        for item in items:
            if isinstance(item, CheckResult):
                normalized.append(item)
            else:
                cat, name, status, detail, remedy = item
                normalized.append(
                    CheckResult.model_validate(
                        {
                            "category": cat,
                            "name": name,
                            "status": status,
                            "detail": detail,
                            "remedy": remedy,
                        },
                    ),
                )
        return cls(checks=tuple(normalized))


def render_doctor_table(
    console: Console,
    results: DoctorReport | Sequence[CheckResult | CheckRowTuple],
    *,
    verbose: bool = False,
) -> int:
    """Render diagnostic check results formatted in Rich tables."""
    report = DoctorReport.coerce(results)

    table = Table(title="Reach Diagnostics", border_style="dim")
    table.add_column("Category", style="reach.catalog", no_wrap=True)
    table.add_column("Item", style="bold")
    table.add_column("Status", justify="center")
    table.add_column("Details")

    status_cells: dict[str, Text] = {
        CheckStatus.OK: Text("✓ OK", style="green"),
        CheckStatus.WARN: Text("! WARN", style="yellow"),
        CheckStatus.FAIL: Text("✗ FAIL", style="red"),
    }

    fail_remedies: list[tuple[str, str]] = []
    warn_remedies: list[tuple[str, str]] = []

    for check in report.checks:
        sym = status_cells.get(check.status, Text(str(check.status)))
        table.add_row(
            Text(check.category, style="reach.catalog"),
            Text(check.name, style="bold"),
            sym,
            Text(check.detail),
        )
        if check.remedy:
            if check.status is CheckStatus.FAIL:
                fail_remedies.append((check.name, check.remedy))
            else:
                warn_remedies.append((check.name, check.remedy))

    console.print(table)

    shown_remedies = (
        [(c.name, c.remedy) for c in report.checks if c.remedy] if verbose else fail_remedies
    )

    if shown_remedies:
        console.print()
        remedy_text = Text("\n").join(
            Text.assemble((f"{name}: ", "bold"), rem) for name, rem in shown_remedies
        )
        console.print(
            Panel(
                remedy_text,
                title="Recommended Actions",
                border_style="red" if report.has_failures else "yellow",
            ),
        )

    if not verbose and warn_remedies:
        console.print()
        tip_msg = (
            "Tip: Run 'reach doctor --verbose' to view additional recommendations."
            if report.has_failures
            else "Tip: Run 'reach doctor --verbose' to view recommended actions."
        )
        console.print(Text(tip_msg, style="dim"))

    return 1 if report.has_failures else 0


def render_doctor_json(report: DoctorReport) -> str:
    """Serialize DoctorReport to formatted JSON."""
    return report.model_dump_json(indent=2)


def render_doctor_jsonl(report: DoctorReport) -> str:
    """Serialize DoctorReport checks to newline-delimited JSON (JSONL)."""
    return "\n".join(c.model_dump_json() for c in report.checks) + ("\n" if report.checks else "")


def render_doctor_csv(report: DoctorReport) -> str:
    """Export DoctorReport checks to CSV."""
    rows: list[list[object]] = [
        [c.category, c.name, str(c.status), c.detail, c.remedy] for c in report.checks
    ]
    return csv_document(["category", "name", "status", "detail", "remedy"], rows)


DOCTOR_RENDERERS = {
    "csv": render_doctor_csv,
    "json": render_doctor_json,
    "jsonl": render_doctor_jsonl,
}


def render_doctor(report: DoctorReport, fmt: str = "json") -> str:
    """Render DoctorReport in the requested structured output format."""
    return dispatch_render(DOCTOR_RENDERERS, fmt, report)
