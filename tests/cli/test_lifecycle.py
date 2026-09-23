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

"""Verify end-to-end workflow transitions between CLI subcommands."""

from __future__ import annotations

from typing import TYPE_CHECKING

from reach.artifact import artifact_path
from reach.cli import main
from reach.models import Query, QueryKind
from reach.queries import Origin, QuerySet, QuerySetProvenance, load_query_set, query_set_digest
from reach.runtime.fake import FakeGenerator

if TYPE_CHECKING:
    from pathlib import Path

#: Neighborhood identifier for test corpus.
NEIGHBORHOOD = "neighborhood:gcs-lifecycle-rules"

#: Scripted QuerySet fixture matching expected drafting output.
DRAFTED = QuerySet(
    catalog_id=NEIGHBORHOOD,
    queries=(
        Query(
            id="q-lifecycle",
            text="Tier old objects to Coldline after 30 days.",
            kind=QueryKind.IMPLICIT,
            expected_skill="gcs-lifecycle-rules",
        ),
        Query(
            id="q-retention",
            text="Keep audit logs for seven years for compliance.",
            kind=QueryKind.NEIGHBOR_NEGATIVE,
            expected_skill="gcs-retention-policy",
        ),
    ),
    provenance=QuerySetProvenance(origin=Origin.AUTHORED),
)


def _lifecycle_overlap(corpus: Path, capsys) -> None:
    """Verify overlap CLI reads corpus alone and outputs skill names."""
    assert main(["overlap", "--skills", str(corpus)]) == 0
    assert "gcs-lifecycle-rules" in capsys.readouterr().err


def _lifecycle_draft_and_export(
    tmp_path: Path,
    corpus: Path,
    monkeypatch,
) -> Path:
    """Run query draft, export, and import round-trip, returning imported query set path."""
    drafted = tmp_path / "drafted.json"
    exported = tmp_path / "exported.csv"
    imported = tmp_path / "imported.json"

    monkeypatch.setattr(
        "reach.cli.drafting.text_generator",
        lambda **_: FakeGenerator(),
    )
    monkeypatch.setattr(
        "reach.cli.drafting.generate_query_set",
        lambda *_, **__: DRAFTED,
    )
    assert (
        main(
            [
                "query",
                "draft",
                "--skills",
                str(corpus),
                "--queries",
                str(drafted),
                "--agent",
                "fake",
            ],
        )
        == 0
    )
    assert drafted.exists()
    assert query_set_digest(load_query_set(drafted)) == query_set_digest(DRAFTED)

    assert main(["query", str(drafted), "--out", str(exported)]) == 0
    header = exported.read_text(encoding="utf-8").splitlines()[0]
    assert header == "id,text,kind,expected_skill,acceptable_skills,notes"

    assert (
        main(
            [
                "query",
                str(exported),
                "--out",
                str(imported),
                "--catalog",
                NEIGHBORHOOD,
            ],
        )
        == 0
    )
    assert query_set_digest(load_query_set(imported)) == query_set_digest(DRAFTED)
    return imported


def _lifecycle_eval_and_diff(
    tmp_path: Path,
    corpus: Path,
    imported: Path,
    capsys,
) -> None:
    """Execute control and treatment evaluation runs, view artifact, and compare via diff."""
    workdir = tmp_path / "work"
    control = tmp_path / "control.jsonl"
    treatment = tmp_path / "treatment.jsonl"

    assert (
        main(
            [
                "eval",
                "--skills",
                str(corpus),
                "--queries",
                str(imported),
                "--workdir",
                str(workdir),
                "--agent",
                "fake",
                "--mode",
                "neighborhood",
                "--catalog-size",
                "3",
                "--rivals",
                "2",
                "--attempts",
                "1",
                "--partial",
                "--out",
                str(control),
            ],
        )
        == 0
    )
    control_artifact = artifact_path(control)
    assert control_artifact.exists()
    capsys.readouterr()

    assert main(["view", str(control_artifact)]) == 0
    shown = capsys.readouterr().err
    assert "recall" in shown.split()
    assert "gcs-lifecycle-rules" in shown

    assert (
        main(
            [
                "eval",
                "--skills",
                str(corpus),
                "--queries",
                str(imported),
                "--workdir",
                str(workdir),
                "--agent",
                "fake",
                "--mode",
                "neighborhood",
                "--catalog-size",
                "2",
                "--rivals",
                "1",
                "--attempts",
                "1",
                "--partial",
                "--out",
                str(treatment),
            ],
        )
        == 0
    )
    assert artifact_path(treatment).exists()
    capsys.readouterr()

    assert main(["diff", str(control), str(treatment), "--vary", "scope"]) == 0
    assert "Verdict" in capsys.readouterr().out


def test_the_full_lifecycle_round_trips_through_every_verb(
    tmp_path: Path,
    corpus_builder,
    monkeypatch,
    capsys,
) -> None:
    """Verify end-to-end pipeline spanning overlap, draft, export, import, eval, view, and diff."""
    corpus = (
        corpus_builder()
        .add("gcs-lifecycle-rules", "Configures object lifecycle rules.")
        .add("gcs-retention-policy", "Configures retention and bucket lock.")
        .add("gke-basics", "Explains GKE cluster fundamentals.")
        .build_disk(tmp_path / "skills")
    )
    _lifecycle_overlap(corpus, capsys)
    imported = _lifecycle_draft_and_export(tmp_path, corpus, monkeypatch)
    _lifecycle_eval_and_diff(tmp_path, corpus, imported, capsys)
