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

"""Verify JSON schema semantic tiering, alphabetization, and Pydantic v2 modernization."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from reach._io import read_model, write_model
from reach.artifact import Artifact, SkillScore
from reach.check import CheckAssertion, CheckOutcome
from reach.lint import LintIssue, LintReport, Severity
from reach.models import Query
from reach.optimize import OptimizationReport
from reach.queries import Origin, QuerySet, QuerySetProvenance
from reach.sweep import ScalingPoint, ScalingStudy
from reach.uncertainty import Interval


def test_interval_sequence_coercion_and_protocol() -> None:
    """Verify Interval coerces sequence inputs, supports tuple unpacking and indexing."""
    # Coercion from tuple and list
    iv_tuple = Interval.model_validate((0.1, 0.9))
    assert iv_tuple.low == 0.1
    assert iv_tuple.high == 0.9

    iv_list = Interval.model_validate([-0.5, 0.5])
    assert iv_list.low == -0.5
    assert iv_list.high == 0.5

    # Sequence protocol
    assert len(iv_tuple) == 2
    assert iv_tuple[0] == 0.1
    assert iv_tuple[1] == 0.9
    low, high = iv_tuple
    assert (low, high) == (0.1, 0.9)


def test_query_id_standardization_and_alias() -> None:
    """Verify Query populates query_id from both 'query_id' and 'id', exposing .id property."""
    q1 = Query(query_id="qid-1", text="Find files", expected_skill="search")
    assert q1.query_id == "qid-1"
    assert q1.id == "qid-1"

    q2 = Query.model_validate({"id": "qid-2", "text": "Deploy app", "expected_skill": "deploy"})
    assert q2.query_id == "qid-2"
    assert q2.id == "qid-2"

    with pytest.raises(ValidationError):
        Query(query_id="   ", text="Deploy app", expected_skill="deploy")


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
    """Verify ScalingPoint groups metrics, serializes Interval objects, and validates bounds."""
    pt = ScalingPoint(
        scale=10,
        catalog_id="cat-1",
        f1_score=0.9,
        f1_interval=(0.85, 0.95),
        pass_rate=0.92,
        pass_rate_interval=(0.88, 0.96),
        precision=0.95,
        precision_interval=(0.9, 1.0),
        recall=0.88,
        recall_interval=(0.8, 0.94),
        delta_vs_baseline=0.0,
        delta_abstention=0.0,
        delta_collision=0.0,
        probes_executed=100,
        probes_failed=8,
        probes_errored=2,
    )
    data = json.loads(pt.model_dump_json())
    assert isinstance(data["pass_rate_interval"], dict)
    assert "low" in data["pass_rate_interval"]
    assert "high" in data["pass_rate_interval"]
    assert data["pass_rate_interval"]["low"] == 0.88

    # Ensure probe accounting validation works
    with pytest.raises(ValidationError, match="probes_errored"):
        ScalingPoint(
            scale=10,
            catalog_id="cat-1",
            pass_rate=1.0,
            pass_rate_interval=(1.0, 1.0),
            delta_vs_baseline=0.0,
            delta_abstention=0.0,
            delta_collision=0.0,
            probes_executed=10,
            probes_errored=15,
        )


def test_scaling_study_flattening_and_decomposition_property(tmp_path: Path) -> None:
    """Verify ScalingStudy flattens loss metrics, omits duplicate decomposition in JSON."""
    pt = ScalingPoint(
        scale=10,
        catalog_id="cat-10",
        pass_rate=1.0,
        pass_rate_interval=(0.9, 1.0),
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

    # Backward compatible properties
    assert study.total_delta == 0.15
    assert study.total_abstention_loss == 0.05
    assert study.total_collision_loss == 0.10
    decomp = study.decomposition
    assert decomp.delta_total == 0.15
    assert decomp.delta_abstention == 0.05
    assert decomp.delta_collision == 0.10

    # JSON serialization omits decomposition sub-object and places points at the bottom
    json_path = tmp_path / "sweep.json"
    study.save(json_path)
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert "decomposition" not in data
    assert list(data.keys())[-1] == "points"

    # Load roundtrip
    loaded = ScalingStudy.load(json_path)
    assert loaded.delta_total == 0.15
    assert loaded.total_delta == 0.15
    assert len(loaded.points) == 1


def test_skillscore_no_frozen_mutation() -> None:
    """Verify SkillScore initializes trajectory metrics in before-validator cleanly."""
    score = SkillScore(
        skill="gcs-read",
        probes=20,
        reached=15,
        recall=0.75,
        precision=1.0,
    )
    assert score.trajectory_reached == 15
    assert score.trajectory_recall == pytest.approx(0.75)
    assert score.f1 is not None

    # Clamping when trajectory_reached < reached
    clamped = SkillScore(
        skill="gcs-write",
        probes=20,
        reached=18,
        trajectory_reached=10,  # lower than reached
        recall=0.9,
        precision=1.0,
    )
    assert clamped.trajectory_reached == 18
    assert clamped.trajectory_recall == pytest.approx(0.9)


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


def test_cli_reports_leaf_collections_at_bottom() -> None:
    """Verify CheckOutcome, OptimizationReport, and LintReport place collections at the bottom."""
    lint = LintReport(
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
    )
    lint_keys = list(json.loads(lint.model_dump_json()).keys())
    assert lint_keys[-1] == "issues"

    check = CheckOutcome(
        exit_code=0,
        lint_report=lint,
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
    )
    check_keys = list(json.loads(check.model_dump_json()).keys())
    assert check_keys[0] == "exit_code"
    assert check_keys[-1] == "assertions"

    opt = OptimizationReport(
        skill_name="opt-skill",
        baseline_description="Old description",
    )
    assert list(OptimizationReport.model_fields.keys())[-1] == "candidates"
    opt_keys = list(json.loads(opt.model_dump_json()).keys())
    assert opt_keys[0] == "skill_name"
    assert opt_keys[-2] == "candidates"
    assert opt_keys[-1] == "has_improvement"
