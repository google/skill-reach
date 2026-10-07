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

"""Validate 6-state trajectory invocation pattern classification and negative probe handling."""

from __future__ import annotations

import pytest

from reach.metrics import classify_invocation_pattern
from reach.models import (
    Catalog,
    CatalogMode,
    DisclosureState,
    InvocationPattern,
    ProbeResult,
    Query,
    QueryKind,
)
from reach.runtime import SelectionOutcome


@pytest.mark.parametrize(
    ("expected_skill", "kind", "invoked", "expected_pattern"),
    [
        ("gcs-deploy", QueryKind.IMPLICIT, ("gcs-deploy",), InvocationPattern.ORACLE_ONLY),
        (
            "gcs-deploy",
            QueryKind.IMPLICIT,
            ("gcs-deploy", "gcs-deploy"),
            InvocationPattern.ORACLE_ONLY,
        ),
        (
            "gcs-deploy",
            QueryKind.IMPLICIT,
            ("gcs-deploy", "gke-deploy"),
            InvocationPattern.MIXED_ORACLE,
        ),
        (
            "gcs-deploy",
            QueryKind.IMPLICIT,
            ("gke-deploy", "gcs-deploy"),
            InvocationPattern.MIXED_ORACLE,
        ),
        ("gcs-deploy", QueryKind.IMPLICIT, ("gke-deploy",), InvocationPattern.DISTRACTOR_HIJACK),
        (
            "gcs-deploy",
            QueryKind.IMPLICIT,
            ("gke-deploy", "docker-run"),
            InvocationPattern.DISTRACTOR_HIJACK,
        ),
        ("gcs-deploy", QueryKind.IMPLICIT, (), InvocationPattern.ABANDONED),
        (None, QueryKind.OUT_OF_SCOPE, (), InvocationPattern.CORRECT_ABSTENTION),
        (None, QueryKind.OUT_OF_SCOPE, ("any-skill",), InvocationPattern.UNWANTED_TRIGGER),
        (None, QueryKind.OUT_OF_SCOPE, ("s1", "s2"), InvocationPattern.UNWANTED_TRIGGER),
    ],
)
def test_classify_invocation_pattern(
    expected_skill: str | None,
    kind: QueryKind,
    invoked: tuple[str, ...],
    expected_pattern: InvocationPattern,
) -> None:
    """Verify trajectory classification matches expected InvocationPattern across all cases."""
    query = Query(query_id="q-test", text="Test query", kind=kind, expected_skill=expected_skill)
    pattern = classify_invocation_pattern(query, invoked)
    assert pattern == expected_pattern


def test_probe_result_from_outcome_sets_invocation_pattern() -> None:
    """Verify ProbeResult.from_outcome assigns invocation_pattern and disclosure_state."""
    query = Query(
        query_id="q-1", text="Deploy service", kind=QueryKind.IMPLICIT, expected_skill="cloud-run"
    )
    catalog = Catalog(id="cat-1", mode=CatalogMode.SINGLETON, skills=("cloud-run",))
    outcome = SelectionOutcome(invoked_skills=("cloud-run",))

    result = ProbeResult.from_outcome(
        outcome=outcome,
        query=query,
        catalog=catalog,
        runtime_name="fake",
    )
    assert result.invocation_pattern == InvocationPattern.ORACLE_ONLY
    assert result.disclosure_state == DisclosureState.FULL
