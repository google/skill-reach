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

"""Verify DOM contracts between server-rendered HTML and client JavaScript via selectolax.

These tests strike a balance between high safety and low brittleness:
1. High safety: catches broken IDs, mismatched onclick/oninput handlers, and missing components.
2. Low brittleness: does not check layout styling, child element ordering, or hardcoded counts.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from selectolax.parser import HTMLParser

from reach.models import Query, QueryKind, Skill
from reach.queries import Origin, QuerySet, QuerySetProvenance
from reach.review import ReviewSentinel, render_query_review_html
from reach.static import get_review_js, get_view_js
from reach.view import render_view_html

if TYPE_CHECKING:
    from collections.abc import Callable

    from reach.artifact import Artifact

#: Match inline DOM event handler invocations: e.g. "approveQueries()" -> "approveQueries".
_HANDLER_CALL_RE = re.compile(r"^([a-zA-Z0-9_]+)\(")

#: Detect syntax errors in JavaScript event handler arguments, e.g. "addRow(, true)".
_MALFORMED_ARGS_RE = re.compile(r"\(\s*,|,\s*,")

#: Element IDs in view.html that view.js queries or binds to.
VIEW_REQUIRED_ELEMENT_IDS: frozenset[str] = frozenset(
    {
        "active-filter-banner",
        "active-filter-label",
        "confusion-table",
        "queries-container",
        "skills-table",
    }
)

#: Element IDs in review.html that review.js queries or updates.
REVIEW_REQUIRED_ELEMENT_IDS: frozenset[str] = frozenset(
    {
        "balance-guardrails",
        "balance-triggers",
        "empty-state",
        "guardrails-count",
        "guardrails-empty",
        "guardrails-list",
        "status-bar",
        "summary",
        "triggers-count",
        "triggers-empty",
        "triggers-list",
    }
)


@pytest.fixture
def sample_skills() -> tuple[Skill, Skill, Skill]:
    """Provide target skill and two rival skills."""
    target = Skill(
        name="mac-storage-cleanup",
        description="Clean up disk storage safely on macOS.",
        path=Path("./mac-storage-cleanup"),
    )
    rival_docker = Skill(
        name="docker-clean",
        description="Clean up docker containers.",
        path=Path("./docker-clean"),
    )
    rival_git = Skill(
        name="git-workflow",
        description="Clean up git branches.",
        path=Path("./git-workflow"),
    )
    return target, rival_docker, rival_git


@pytest.fixture
def sample_queryset() -> QuerySet:
    """Provide a sample query set with implicit, negative, and out-of-scope queries."""
    queries = (
        Query(
            id="q1",
            text="my mac disk is almost full",
            expected_skill="mac-storage-cleanup",
            kind=QueryKind.IMPLICIT,
        ),
        Query(
            id="q2",
            text="prune stopped docker containers",
            expected_skill="docker-clean",
            kind=QueryKind.NEIGHBOR_NEGATIVE,
        ),
        Query(
            id="q3",
            text="clean git branch history",
            expected_skill="git-workflow",
            kind=QueryKind.NEIGHBOR_NEGATIVE,
        ),
        Query(
            id="q4",
            text="unrelated question",
            expected_skill=None,
            kind=QueryKind.OUT_OF_SCOPE,
        ),
    )
    return QuerySet(
        catalog_id="neighborhood:mac-storage-cleanup",
        queries=queries,
        provenance=QuerySetProvenance(origin=Origin.AUTHORED),
    )


@pytest.fixture
def review_html_tree(
    sample_skills: tuple[Skill, Skill, Skill],
    sample_queryset: QuerySet,
) -> HTMLParser:
    """Render query review HTML and return parsed selectolax tree."""
    target, rival_docker, rival_git = sample_skills
    html = render_query_review_html(sample_queryset, target, [rival_docker, rival_git])
    return HTMLParser(html)


@pytest.fixture
def view_html_tree(artifact: Artifact) -> HTMLParser:
    """Render diagnostic workbench HTML and return parsed selectolax tree."""
    html = render_view_html(artifact)
    return HTMLParser(html)


# --- 1. ID & Hook Point Contracts ---


@pytest.mark.parametrize(
    ("tree_fixture_name", "js_loader", "required_ids", "context_name"),
    [
        ("view_html_tree", get_view_js, VIEW_REQUIRED_ELEMENT_IDS, "view"),
        ("review_html_tree", get_review_js, REVIEW_REQUIRED_ELEMENT_IDS, "review"),
    ],
)
def test_html_contains_required_interactive_element_ids(
    request: pytest.FixtureRequest,
    tree_fixture_name: str,
    js_loader: Callable[[], str],
    required_ids: frozenset[str],
    context_name: str,
) -> None:
    """Verify rendered HTML includes all element IDs actively required by client JS."""
    tree = request.getfixturevalue(tree_fixture_name)
    assert isinstance(tree, HTMLParser)
    js_code = js_loader()

    # 1. Ensure the contract IDs are actively referenced in client JS (prevent stale rules)
    stale_ids = {eid for eid in required_ids if eid not in js_code}
    assert not stale_ids, (
        f"Contract IDs no longer referenced in {context_name}.js: {sorted(stale_ids)}"
    )

    # 2. Ensure all contract IDs exist in rendered HTML (reports all missing IDs at once)
    rendered_ids = {node.attributes["id"] for node in tree.css("[id]") if "id" in node.attributes}
    missing_ids = required_ids - rendered_ids
    assert not missing_ids, f"Element IDs missing from {context_name}.html: {sorted(missing_ids)}"


# --- 2. Bidirectional Event Handler Contracts ---


@pytest.mark.parametrize(
    ("tree_fixture_name", "js_loader", "js_name"),
    [
        ("view_html_tree", get_view_js, "view.js"),
        ("review_html_tree", get_review_js, "review.js"),
    ],
)
def test_html_inline_event_handlers_exist_in_js(
    request: pytest.FixtureRequest,
    tree_fixture_name: str,
    js_loader: Callable[[], str],
    js_name: str,
) -> None:
    """Verify every onclick/oninput handler in HTML maps to a function in companion JS."""
    tree = request.getfixturevalue(tree_fixture_name)
    assert isinstance(tree, HTMLParser)
    js_code = js_loader()

    unresolved_handlers: list[str] = []
    malformed_handlers: list[str] = []
    for node in tree.css("[onclick], [oninput]"):
        for attr in ("onclick", "oninput"):
            attr_val = node.attributes.get(attr)
            if not attr_val:
                continue
            stripped = attr_val.strip()
            if _MALFORMED_ARGS_RE.search(stripped):
                malformed_handlers.append(f"{attr}='{stripped}'")
            match = _HANDLER_CALL_RE.match(stripped)
            if match:
                fn_name = match.group(1)
                defined = f"function {fn_name}" in js_code or f"{fn_name}(" in js_code
                if not defined:
                    unresolved_handlers.append(f"{attr}='{fn_name}'")

    assert not malformed_handlers, (
        f"Malformed arguments in {js_name} event handlers: {sorted(set(malformed_handlers))}"
    )
    assert not unresolved_handlers, (
        f"HTML event handlers not defined in {js_name}: {sorted(set(unresolved_handlers))}"
    )


# --- 3. Workbench View Component Anatomy & Data Flow ---


def test_view_html_table_structures_match_interactive_event_delegations(
    view_html_tree: HTMLParser,
) -> None:
    """Verify confusion, collisions, and skills tables contain expected DOM cells for filtering."""
    matrix = view_html_tree.css_first("#confusion-table")
    assert matrix is not None
    assert matrix.css_first("thead tr") is not None
    rows = matrix.css("tbody tr")
    assert len(rows) > 0
    for row in rows:
        assert row.css_first("th") is not None, "Each confusion matrix row must have a row header"
        assert len(row.css("td")) > 0, "Each confusion matrix row must have score cells"

    collisions = view_html_tree.css_first("#collisions-table")
    assert collisions is not None, "Collisions table must be rendered when collisions exist"
    collision_rows = collisions.css("tbody tr")
    assert len(collision_rows) > 0, "Collisions table must have at least one collision row"
    for row in collision_rows:
        cells = row.css("td")
        assert len(cells) >= 2, "Collision table rows must have expected and invoked cells"

    skills = view_html_tree.css_first("#skills-table")
    assert skills is not None
    skill_rows = skills.css("tbody tr")
    assert len(skill_rows) > 0
    for row in skill_rows:
        first_cell = row.css_first("td:first-child")
        assert first_cell is not None, "Skills table row missing first-child cell"
        assert first_cell.text().strip(), "Skill cell must contain text for click filter"


def test_view_html_queries_container_and_details_items(view_html_tree: HTMLParser) -> None:
    """Verify queries container and child query details match expected filtering structure."""
    container = view_html_tree.css_first("#queries-container")
    assert container is not None

    query_nodes = container.css("details.query")
    assert len(query_nodes) > 0

    for q in query_nodes:
        expected_el = q.css_first(".expected")
        assert expected_el is not None, "Query element missing .expected indicator"
        assert expected_el.text().strip(), "Expected indicator must not be empty"


# --- 4. Review Curation Component Anatomy & Data Flow ---


def test_review_html_body_dataset_contract(
    review_html_tree: HTMLParser,
    sample_skills: tuple[Skill, Skill, Skill],
) -> None:
    """Verify body element exposes target, rivals, and out-of-scope dataset attributes."""
    target, rival_docker, rival_git = sample_skills
    body = review_html_tree.css_first("body")
    assert body is not None

    # 1. data-target-skill consumed by getTargetSkill()
    assert body.attributes.get("data-target-skill") == target.name

    # 2. data-rivals consumed as parsed JSON in addNewQueryCard()
    rivals_raw = str(body.attributes.get("data-rivals") or "[]")
    rivals_data = json.loads(rivals_raw)
    assert rival_docker.name in rivals_data
    assert rival_git.name in rivals_data

    # 3. data-out-of-scope consumed by getOutOfScope()
    assert body.attributes.get("data-out-of-scope") == ReviewSentinel.OUT_OF_SCOPE.value


def test_review_html_query_card_component_contract(
    review_html_tree: HTMLParser,
    sample_queryset: QuerySet,
) -> None:
    """Verify query cards contain all interactive sub-components without asserting exact layout."""
    cards = review_html_tree.css(".query-card")
    assert len(cards) == len(sample_queryset.queries)

    # Verify partitioning into triggers and guardrails matching updateSummary() contract
    trigger_cards = review_html_tree.css("#triggers-list .query-card")
    guardrail_cards = review_html_tree.css("#guardrails-list .query-card")
    assert len(trigger_cards) == 1
    assert len(guardrail_cards) == 3

    for card in cards:
        # Text display & editor controls
        assert card.css_first(".query-text-preview") is not None
        assert card.css_first(".query-card-editor") is not None
        query_input = card.css_first(".query-input")
        assert query_input is not None
        assert query_input.text().strip(), "Query input value should contain query text"

        # Interactive action buttons
        assert card.css_first(".card-btn-edit") is not None
        assert card.css_first(".card-btn-done") is not None
        assert card.css_first(".card-btn-swap") is not None
        assert card.css_first(".card-btn-delete") is not None

        # Badge container
        assert card.css_first(".badge-target-wrapper") is not None


def test_review_html_rival_select_options(
    review_html_tree: HTMLParser,
    sample_skills: tuple[Skill, Skill, Skill],
) -> None:
    """Verify rival select dropdowns contain options matching passed rival skills."""
    _, rival_docker, rival_git = sample_skills
    guardrails_list = review_html_tree.css_first("#guardrails-list")
    assert guardrails_list is not None
    selects = guardrails_list.css(".rival-select")
    assert len(selects) > 0

    select = selects[0]
    options = [opt.attributes.get("value") for opt in select.css("option")]
    assert rival_docker.name in options
    assert rival_git.name in options
    assert ReviewSentinel.OUT_OF_SCOPE.value in options


def test_review_html_renders_empty_state_cleanly(
    sample_skills: tuple[Skill, Skill, Skill],
) -> None:
    """Verify review HTML gracefully renders empty state when zero queries are provided."""
    target, rival_docker, _ = sample_skills
    empty_qs = QuerySet(
        catalog_id=f"neighborhood:{target.name}",
        queries=(),
        provenance=QuerySetProvenance(origin=Origin.AUTHORED),
    )
    html = render_query_review_html(empty_qs, target, [rival_docker])
    tree = HTMLParser(html)

    # Empty containers should be present
    assert tree.css_first("#empty-state") is not None
    assert tree.css_first("#triggers-empty") is not None
    assert tree.css_first("#guardrails-empty") is not None

    # Summary text should reflect 0 queries
    summary = tree.css_first("#summary")
    assert summary is not None
    assert "0 queries total" in summary.text()
