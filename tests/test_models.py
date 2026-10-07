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

"""Verify Query and ProbeResult models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from reach.models import (
    NO_SKILL,
    Catalog,
    CatalogMode,
    ProbeResult,
    Query,
    QueryKind,
)
from reach.runtime import SelectionOutcome


def test_query_attributes_and_out_of_scope() -> None:
    """Verify Query stores expected_skill, computes properties, and handles out-of-scope."""
    q = Query(
        query_id="q1",
        text="Deploy the app",
        expected_skill="cloud-run-deploy",
    )
    assert q.query_id == "q1"
    assert q.text == "Deploy the app"
    assert q.expected_skill == "cloud-run-deploy"
    assert q.truth_label == "cloud-run-deploy"
    assert not q.is_out_of_scope

    q_oos = Query(query_id="q3", text="What is the weather?", kind=QueryKind.OUT_OF_SCOPE)
    assert q_oos.is_out_of_scope
    assert q_oos.truth_label == NO_SKILL


def test_probe_result_invoked_skill_property() -> None:
    """Verify ProbeResult exposes invoked_skill as property over invoked_skills."""
    res1 = ProbeResult(
        query_id="q1",
        catalog_id="cat1",
        catalog_mode=CatalogMode.ALL,
        catalog_size=5,
        model="test-model",
        runtime="test-runtime",
        invoked_skills=("gcloud-auth", "cloud-run-deploy"),
    )
    assert res1.invoked_skills == ("gcloud-auth", "cloud-run-deploy")
    assert res1.invoked_skill == "gcloud-auth"
    assert res1.selected is True
    assert res1.predicted_label == "gcloud-auth"

    res2 = ProbeResult(
        query_id="q2",
        catalog_id="cat1",
        catalog_mode=CatalogMode.ALL,
        catalog_size=5,
        model="test-model",
        runtime="test-runtime",
    )
    assert res2.invoked_skills == ()
    assert res2.invoked_skill is None
    assert res2.selected is False
    assert res2.predicted_label == NO_SKILL


def test_probe_result_from_outcome() -> None:
    """Verify ProbeResult.from_outcome converts SelectionOutcome into a ProbeResult."""
    from reach.models import Catalog, Provenance

    q = Query(query_id="q1", text="Deploy app", expected_skill="deploy")
    cat = Catalog(id="cat1", mode=CatalogMode.ALL, skills=("deploy", "build"))
    outcome = SelectionOutcome(
        invoked_skills=("deploy",),
        resolved_model="gemini-flash",
        observed_catalog=("deploy", "build"),
        cost_usd=0.002,
        duration_ms=450,
        reasoning=("Selected deploy skill",),
    )
    prov = Provenance(config_fingerprint="fp123", corpus_digest="cd456")
    result = ProbeResult.from_outcome(
        outcome=outcome,
        query=q,
        catalog=cat,
        runtime_name="fake",
        model="fake-model",
        attempt=2,
        provenance=prov,
    )
    assert result.query_id == "q1"
    assert result.catalog_id == "cat1"
    assert result.catalog_size == 2
    assert result.invoked_skill == "deploy"
    assert result.invoked_skills == ("deploy",)
    assert result.runtime == "fake"
    assert result.model == "fake-model"
    assert result.resolved_model == "gemini-flash"
    assert result.attempt == 2
    assert result.config_fingerprint == "fp123"
    assert result.corpus_digest == "cd456"
    assert result.cost_usd == 0.002
    assert result.duration_ms == 450
    assert result.error is None
    assert result.selected is True
    assert result.predicted_label == "deploy"


def test_catalog_rejects_duplicate_skills() -> None:
    """Verify Catalog model rejects duplicate skill names."""
    with pytest.raises(ValueError, match="duplicate"):
        Catalog(
            id="cat-dup",
            mode=CatalogMode.ALL,
            skills=("skill-a", "skill-b", "skill-a"),
        )


def test_skill_requires_non_empty_name_and_description() -> None:
    """Verify Skill custom field validators reject whitespace-only name or description."""
    from pathlib import Path

    from reach.models import Skill

    with pytest.raises(ValidationError, match="name must be non-empty"):
        Skill(name="   ", description="Valid description.", path=Path("SKILL.md"))

    with pytest.raises(ValidationError, match="description must be non-empty"):
        Skill(name="s", description="   ", path=Path("SKILL.md"))


def test_query_acceptable_skills() -> None:
    """Verify Query accepts optional acceptable_skills as sequence of skill names."""
    q_default = Query(query_id="q1", text="Deploy app", expected_skill="deploy")
    assert q_default.acceptable_skills == ()

    q_with_list = Query.model_validate(
        {
            "query_id": "q2",
            "text": "Deploy app",
            "expected_skill": "deploy",
            "acceptable_skills": ["cloud-run-deploy", "app-engine-deploy"],
        }
    )
    assert q_with_list.acceptable_skills == ("cloud-run-deploy", "app-engine-deploy")
