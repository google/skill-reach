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

"""Verify JSON schema semantic tiering, alphabetization, and Pydantic v2 clean-break refactoring."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError

from reach._io import read_model, write_model
from reach.artifact import Artifact, SkillScore
from reach.check import CheckAssertion, CheckOutcome
from reach.lint import LintIssue, LintReport, Severity
from reach.metrics import ClassMetrics
from reach.models import Query
from reach.optimize import OptimizationReport
from reach.queries import Origin, QuerySet, QuerySetProvenance, load_query_set
from reach.sweep import ScalingPoint, ScalingStudy
from reach.uncertainty import Interval


def test_interval_strict_model_and_factories() -> None:
    """Verify Interval requires keyword fields and provides standard factories."""
    iv = Interval(low=0.1, high=0.9)
    assert iv.low == 0.1
    assert iv.high == 0.9

    zero = Interval.zero()
    assert zero.low == 0.0
    assert zero.high == 0.0

    unit = Interval.unit()
    assert unit.low == 0.0
    assert unit.high == 1.0

    from_tup = Interval.from_tuple((0.2, 0.8))
    assert from_tup.low == 0.2
    assert from_tup.high == 0.8

    # Rejection of sequence duck-typing and sequence coercion
    with pytest.raises(ValidationError):
        Interval.model_validate((0.1, 0.9))

    with pytest.raises(ValidationError):
        Interval.model_validate([-0.5, 0.5])

    with pytest.raises(TypeError):
        len(iv)  # type: ignore[arg-type]

    with pytest.raises(TypeError):
        _ = iv[0]  # type: ignore[index]


def test_query_clean_break_schema() -> None:
    """Verify Query strictly requires query_id and non-blank strings, rejecting id alias."""
    q = Query(query_id="qid-1", text="Find files", expected_skill="search")
    assert q.query_id == "qid-1"
    assert not hasattr(q, "id")

    # Legacy 'id' key is rejected under extra='forbid'
    with pytest.raises(ValidationError, match="extra_forbidden"):
        Query.model_validate({"id": "qid-2", "text": "Deploy app", "expected_skill": "deploy"})

    # Whitespace-only query_id or text is rejected by NonBlankStr
    with pytest.raises(ValidationError, match="cannot be empty or whitespace only"):
        Query(query_id="   ", text="Deploy app", expected_skill="deploy")

    with pytest.raises(ValidationError, match="cannot be empty or whitespace only"):
        Query(query_id="qid-3", text="   \t  ", expected_skill="deploy")


def test_queryset_leaf_at_bottom() -> None:
    """Verify QuerySet serializes queries as the final leaf collection."""
    qs = QuerySet(
        catalog_id="test-cat",
        notes="sample notes",
        provenance=QuerySetProvenance(origin=Origin.AUTHORED),
        queries=(Query(query_id="q1", text="do work", expected_skill="worker"),),
    )
    data = json.loads(qs.model_dump_json())
    keys = list(data.keys())
    assert keys == ["catalog_id", "notes", "provenance", "queries"]
    assert keys[-1] == "queries"


def test_scaling_point_semantic_tiering_and_intervals() -> None:
    """Verify ScalingPoint groups metrics, serializes Interval objects, and forbids extra keys."""
    pt = ScalingPoint(
        scale=10,
        catalog_id="cat-1",
        f1_score=0.9,
        f1_interval=Interval(low=0.85, high=0.95),
        pass_rate=0.92,
        pass_rate_interval=Interval(low=0.88, high=0.96),
        precision=0.95,
        precision_interval=Interval(low=0.9, high=1.0),
        recall=0.88,
        recall_interval=Interval(low=0.8, high=0.94),
        delta_vs_baseline=0.0,
        delta_abstention=0.0,
        delta_collision=0.0,
        probes_executed=100,
        probes_failed=8,
        probes_errored=2,
    )
    data = json.loads(pt.model_dump_json())
    assert isinstance(data["pass_rate_interval"], dict)
    assert data["pass_rate_interval"]["low"] == 0.88
    assert data["pass_rate_interval"]["high"] == 0.96

    # Extra legacy fields are rejected
    with pytest.raises(ValidationError, match="extra_forbidden"):
        ScalingPoint(
            scale=10,
            catalog_id="cat-1",
            delta_context=0.04,  # legacy alias rejected
            probes_executed=10,
        )

    # Probe accounting validation
    with pytest.raises(ValidationError, match="probes_errored"):
        ScalingPoint(
            scale=10,
            catalog_id="cat-1",
            pass_rate=1.0,
            pass_rate_interval=Interval(low=1.0, high=1.0),
            delta_vs_baseline=0.0,
            delta_abstention=0.0,
            delta_collision=0.0,
            probes_executed=10,
            probes_errored=15,
        )


def test_scaling_study_flattening_and_clean_break(tmp_path: Path) -> None:
    """Verify ScalingStudy flattens loss metrics and forbids legacy decomposition keys."""
    pt = ScalingPoint(
        scale=10,
        catalog_id="cat-10",
        pass_rate=1.0,
        pass_rate_interval=Interval(low=0.9, high=1.0),
        delta_vs_baseline=0.0,
        delta_abstention=0.0,
        delta_collision=0.0,
        probes_executed=20,
    )
    study = ScalingStudy(
        is_corpus_sweep=True,
        scales=(10,),
        points=(pt,),
        baseline_pass_rate=1.0,
        final_pass_rate=0.85,
        delta_total=0.15,
        delta_abstention=0.05,
        delta_collision=0.10,
    )

    # Canonical fields only, no legacy properties
    assert study.delta_total == 0.15
    assert study.delta_abstention == 0.05
    assert study.delta_collision == 0.10
    assert not hasattr(study, "total_delta")
    assert not hasattr(study, "decomposition")

    # Extra fields are forbidden
    with pytest.raises(ValidationError, match="extra_forbidden"):
        ScalingStudy.model_validate(
            {
                "is_corpus_sweep": True,
                "scales": (10,),
                "baseline_pass_rate": 1.0,
                "final_pass_rate": 0.85,
                "delta_total": 0.15,
                "total_delta": 0.15,
            }
        )

    # JSON serialization places points at the bottom
    json_path = tmp_path / "sweep.json"
    study.save(json_path)
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert "decomposition" not in data
    assert list(data.keys())[-1] == "points"

    # Load roundtrip
    loaded = ScalingStudy.load(json_path)
    assert loaded.delta_total == 0.15
    assert len(loaded.points) == 1


def test_skillscore_from_class_metrics_and_validation() -> None:
    """Verify SkillScore constructs cleanly from ClassMetrics and validates bounds."""
    metrics = ClassMetrics(
        label="gcs-read",
        support=20,
        predicted=15,
        true_positives=15,
        false_positives=0,
        false_negatives=5,
        trajectory_true_positives=18,
    )
    score = SkillScore.from_class_metrics(metrics)
    assert score.reached == 15
    assert score.trajectory_reached == 18
    assert score.trajectory_recall == pytest.approx(0.9)
    assert score.recall == pytest.approx(0.75)
    assert score.f1 is not None

    # Inverted trajectory_reached < reached raises ValidationError
    with pytest.raises(ValidationError, match="trajectory_reached"):
        SkillScore(
            skill="gcs-write",
            probes=20,
            reached=18,
            trajectory_reached=10,
            trajectory_recall=0.5,
            recall=0.9,
            precision=1.0,
        )


def test_artifact_semantic_ordering_and_io(artifact: Artifact, tmp_path: Path) -> None:
    """Verify Artifact serializes fields semantically with skills and queries at the bottom."""
    art_file = tmp_path / "artifact.json"
    write_model(artifact, art_file)
    data = json.loads(art_file.read_text(encoding="utf-8"))
    keys = list(data.keys())

    # Metadata and scores first
    assert keys[0] == "schema_version"
    assert keys[1] == "catalog_id"
    # Leaf collections at bottom
    assert keys[-2] == "skills"
    assert keys[-1] == "queries"

    # read_model roundtrip
    loaded = read_model(Artifact, art_file)
    assert loaded.catalog_id == artifact.catalog_id
    assert len(loaded.skills) == len(artifact.skills)
    assert len(loaded.queries) == len(artifact.queries)


@pytest.mark.parametrize(
    ("model_cls", "leaf_field", "instance"),
    [
        (
            LintReport,
            "issues",
            LintReport(
                skill_name="test-skill",
                skills_checked=1,
                issues=(
                    LintIssue(
                        rule="RULE-1",
                        skill="test-skill",
                        severity=Severity.INFO,
                        message="Sample info",
                    ),
                ),
            ),
        ),
        (
            CheckOutcome,
            "assertions",
            CheckOutcome(
                exit_code=0,
                lint_report=LintReport(skill_name="test-skill", skills_checked=1, issues=()),
                assertions=(
                    CheckAssertion(
                        name="assert-1",
                        passed=True,
                        observed=1.0,
                        threshold=0.8,
                        comparison=">=",
                        message="Passed assertion",
                    ),
                ),
            ),
        ),
        (
            OptimizationReport,
            "candidates",
            OptimizationReport(
                skill_name="opt-skill",
                baseline_description="Old description",
            ),
        ),
    ],
)
def test_cli_reports_leaf_collections_at_bottom(
    model_cls: type[BaseModel],
    leaf_field: str,
    instance: BaseModel,
) -> None:
    """Verify CLI reports place collection fields at the bottom of the schema."""
    field_keys = list(model_cls.model_fields.keys())
    assert leaf_field in field_keys[-2:], (
        f"{leaf_field} should be near the end in {model_cls.__name__}"
    )
    data = json.loads(instance.model_dump_json())
    assert leaf_field in list(data.keys())[-2:], f"{leaf_field} should be serialized near the end"


def test_skillscore_count_bounds_validation() -> None:
    """Verify SkillScore validates that reached and trajectory_reached do not exceed probes."""
    # reached > probes
    with pytest.raises(ValidationError, match=r"reached .* cannot exceed probes"):
        SkillScore(
            skill="test-skill",
            probes=10,
            reached=12,
            trajectory_reached=12,
            trajectory_recall=1.0,
            recall=1.0,
            precision=1.0,
        )

    # trajectory_reached > probes
    with pytest.raises(ValidationError, match=r"trajectory_reached .* cannot exceed probes"):
        SkillScore(
            skill="test-skill",
            probes=10,
            reached=8,
            trajectory_reached=15,
            trajectory_recall=1.0,
            recall=0.8,
            precision=1.0,
        )


def test_read_model_file_validation(tmp_path: Path) -> None:
    """Verify read_model checks for regular file existence and raises FileNotFoundError."""
    missing_file = tmp_path / "nonexistent.json"
    with pytest.raises(FileNotFoundError, match="Model source file not found"):
        read_model(Query, missing_file)

    # Directory instead of file
    with pytest.raises(FileNotFoundError, match="Model source file not found"):
        read_model(Query, tmp_path)


def test_interval_from_tuple_confidence_propagation() -> None:
    """Verify Interval.from_tuple propagates or updates confidence when given an Interval."""
    base = Interval(low=0.2, high=0.8, confidence=0.95)

    # Same confidence returns the identical instance
    same = Interval.from_tuple(base, confidence=0.95)
    assert same is base

    # Different confidence constructs a new instance with the updated confidence
    updated = Interval.from_tuple(base, confidence=0.99)
    assert updated is not base
    assert updated.low == 0.2
    assert updated.high == 0.8
    assert updated.confidence == 0.99


def test_load_query_set_extensionless_json(tmp_path: Path) -> None:
    """Verify load_query_set correctly parses an extensionless JSON file using read_model."""
    qs = QuerySet(
        queries=[
            Query(query_id="q1", text="Deploy app", expected_skill="deploy"),
        ],
        catalog_id="cat-test",
    )
    ext_less = tmp_path / "query_data"
    write_model(qs, ext_less)

    loaded = load_query_set(ext_less)
    assert loaded.catalog_id == "cat-test"
    assert len(loaded.queries) == 1
    assert loaded.queries[0].query_id == "q1"
