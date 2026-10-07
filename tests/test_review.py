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

"""Verify interactive query review HTML rendering and ephemeral HTTP server lifecycle."""

from __future__ import annotations

import email.message
import json
import urllib.error
import urllib.parse
import urllib.request
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import pytest
from selectolax.parser import HTMLParser

from reach.models import Query, QueryKind, Skill
from reach.queries import QuerySet
from reach.review import (
    ReviewSentinel,
    ReviewServerHandler,
    _convert_saved_queries,
    _ReviewQueryItem,
    _ReviewSession,
    launch_query_review,
    render_query_review_html,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from pathlib import Path


@pytest.fixture
def sample_review_bundle(
    write_skill: Callable[..., Path],
) -> tuple[Skill, Skill, QuerySet]:
    """Provide target skill, rival skill, and sample QuerySet for review tests."""
    target_dir = write_skill(
        name="target-skill",
        description="Target skill description for testing.",
    )
    target = Skill(
        name="target-skill",
        description="Target skill description for testing.",
        path=target_dir,
    )
    rival_dir = write_skill(
        name="rival-skill",
        description="Rival skill description for testing.",
    )
    rival = Skill(
        name="rival-skill",
        description="Rival skill description for testing.",
        path=rival_dir,
    )

    from reach.queries import Origin, QuerySetProvenance

    prov = QuerySetProvenance(origin=Origin.AUTHORED)
    queries = (
        Query(
            query_id="q1", text="Query 1", expected_skill="target-skill", kind=QueryKind.IMPLICIT
        ),
        Query(
            query_id="q2",
            text="Query 2",
            expected_skill="rival-skill",
            kind=QueryKind.NEIGHBOR_NEGATIVE,
        ),
        Query(query_id="q3", text="Query 3", expected_skill=None, kind=QueryKind.OUT_OF_SCOPE),
    )
    qs = QuerySet(catalog_id="test-cat", provenance=prov, queries=queries)
    return target, rival, qs


def test_render_query_review_html_skill_and_territory_cards(
    sample_review_bundle: tuple[Skill, Skill, QuerySet],
) -> None:
    """Verify target skill context, rival selector, territory cards, summary, and controls."""
    target, rival, qs = sample_review_bundle
    html = render_query_review_html(qs, target, [rival])
    tree = HTMLParser(html)

    # 1. Target skill context card displays name and description in designated semantic elements
    target_name = tree.css_first("#skill-name")
    assert target_name is not None
    assert "target-skill" in target_name.text()

    target_desc = tree.css_first("#skill-desc")
    assert target_desc is not None
    assert "Target skill description for testing." in target_desc.text()

    # 2. Rival skill is populated in rival selector options
    rival_opt = tree.css_first('option[value="rival-skill"]')
    assert rival_opt is not None
    assert "rival-skill" in rival_opt.text()

    # 3. Territory section headers are clearly labeled
    triggers_header = tree.css_first(".triggers-header .column-title")
    assert triggers_header is not None
    assert "In-Scope Territory" in triggers_header.text()

    guardrails_header = tree.css_first(".guardrails-header .column-title")
    assert guardrails_header is not None
    assert "Neighbor Guardrails" in guardrails_header.text()

    # 4. Queries are partitioned into triggers and guardrails with exact query texts
    trigger_cards = tree.css("#triggers-list .query-card")
    assert len(trigger_cards) == 1
    trigger_input = trigger_cards[0].css_first(".query-input")
    assert trigger_input is not None
    assert trigger_input.text().strip() == "Query 1"

    guardrail_cards = tree.css("#guardrails-list .query-card")
    assert len(guardrail_cards) == 2
    guardrail_inputs = [
        inp.text().strip()
        for card in guardrail_cards
        if (inp := card.css_first(".query-input")) is not None
    ]
    assert guardrail_inputs == ["Query 2", "Query 3"]

    # 5. Territory count badges and balance summary reflect initial partition
    triggers_count = tree.css_first("#triggers-count")
    assert triggers_count is not None
    assert "1 Triggers" in triggers_count.text()

    guardrails_count = tree.css_first("#guardrails-count")
    assert guardrails_count is not None
    assert "2 Guardrails" in guardrails_count.text()

    summary = tree.css_first("#summary")
    assert summary is not None
    assert "3 queries total: 1 should trigger, 2 should not trigger" in summary.text()


def test_render_query_review_html_empty_queries(
    write_skill: Callable[..., Path],
) -> None:
    """Verify render_query_review_html displays 0 queries and reveals empty state."""
    from reach.queries import Origin, QuerySetProvenance

    target_dir = write_skill(name="t-empty", description="Empty test")
    target = Skill(name="t-empty", description="Empty test", path=target_dir)
    qs = QuerySet(
        catalog_id="test-cat",
        provenance=QuerySetProvenance(origin=Origin.AUTHORED),
        queries=(),
    )
    html = render_query_review_html(qs, target, [])
    tree = HTMLParser(html)
    body = tree.css_first("body")
    assert body is not None
    assert body.attributes.get("data-catalog-id") == "test-cat"
    assert body.attributes.get("data-origin") == "authored"

    # Both territory columns are empty
    assert len(tree.css(".query-card")) == 0
    assert len(tree.css("#triggers-list .query-card")) == 0
    assert len(tree.css("#guardrails-list .query-card")) == 0

    # Empty state banner is visible and unhidden
    empty_state = tree.css_first("#empty-state")
    assert empty_state is not None
    assert "No Queries in Review Set" in empty_state.text()
    style = empty_state.attributes.get("style") or ""
    assert "display: none" not in style

    # Counters reflect zero state
    summary = tree.css_first("#summary")
    assert summary is not None
    assert "0 queries total: 0 should trigger, 0 should not trigger" in summary.text()

    triggers_count = tree.css_first("#triggers-count")
    assert triggers_count is not None
    assert "0 Triggers" in triggers_count.text()

    guardrails_count = tree.css_first("#guardrails-count")
    assert guardrails_count is not None
    assert "0 Guardrails" in guardrails_count.text()


def test_launch_query_review_bypassed_in_headless_or_non_interactive(
    write_skill: Callable[..., Path],
) -> None:
    """Verify launch_query_review returns unmodified query set when non-interactive."""
    from reach.queries import Origin, QuerySetProvenance

    target_dir = write_skill(name="t-skill", description="desc")
    target = Skill(name="t-skill", description="desc", path=target_dir)
    qs = QuerySet(
        catalog_id="test-cat",
        provenance=QuerySetProvenance(origin=Origin.AUTHORED),
        queries=(),
    )

    with patch("sys.stdin.isatty", return_value=False):
        result = launch_query_review(qs, target, [])
        assert result == qs


def test_launch_query_review_bypasses_when_reach_no_browser_set(
    write_skill: Callable[..., Path],
) -> None:
    """Verify launch_query_review immediately bypasses when REACH_NO_BROWSER is set."""
    from reach.queries import Origin, QuerySetProvenance

    target_dir = write_skill(name="t-skill", description="desc")
    target = Skill(name="t-skill", description="desc", path=target_dir)
    qs = QuerySet(
        catalog_id="test-cat",
        provenance=QuerySetProvenance(origin=Origin.AUTHORED),
        queries=(),
    )

    with (
        patch("sys.stdin.isatty", return_value=True),
        patch.dict("os.environ", {"REACH_NO_BROWSER": "1", "CI": ""}),
    ):
        result = launch_query_review(qs, target, [])
        assert result == qs


def _assert_security_headers(resp: Any) -> None:
    """Verify security headers are present on response."""
    assert resp.headers.get("X-Content-Type-Options") == "nosniff"
    assert resp.headers.get("X-Frame-Options") == "DENY"
    assert (
        resp.headers.get("Content-Security-Policy")
        == "default-src 'self' 'unsafe-inline'; connect-src 'self';"
    )


def _assert_review_server_endpoints(url: str) -> None:
    """Exercise GET and POST review endpoints on the running ephemeral HTTP server."""
    parsed = urllib.parse.urlsplit(url)
    token = urllib.parse.parse_qs(parsed.query).get("token", [""])[0]
    base_url = f"{parsed.scheme}://{parsed.netloc}"
    post_headers: dict[str, str] = {
        "Content-Type": "application/json",
        "Origin": base_url,
    }
    if token:
        post_headers["X-Reach-Token"] = token

    # 1. Test GET / serves valid HTML with target skill and interactive buttons
    with urllib.request.urlopen(url) as resp:  # noqa: S310
        assert resp.status == 200
        html_text = resp.read().decode("utf-8")
        resp_tree = HTMLParser(html_text)
        skill_name = resp_tree.css_first("#skill-name")
        assert skill_name is not None
        assert "t-skill" in skill_name.text()
        assert resp_tree.css_first("#approve-btn") is not None
        assert resp_tree.css_first("#approve-btn-bottom") is not None
        if token:
            meta_token = resp_tree.css_first('meta[name="reach-token"]')
            assert meta_token is not None
            assert meta_token.attributes.get("content") == token

    # 2. Test GET /api/status returns running health check
    with urllib.request.urlopen(f"{base_url}/api/status") as resp:  # noqa: S310
        assert resp.status == 200
        status_data = json.loads(resp.read().decode("utf-8"))
        assert status_data == {"status": "running"}

    # 3. Test GET /nonexistent returns 404 Not Found
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(f"{base_url}/nonexistent")  # noqa: S310
    exc_info.value.close()
    assert exc_info.value.code == 404

    # 4. Test POST /nonexistent returns 404 Not Found
    post_404 = urllib.request.Request(  # noqa: S310
        f"{base_url}/nonexistent",
        data=b"{}",
        headers=post_headers,
        method="POST",
    )
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(post_404)  # noqa: S310
    exc_info.value.close()
    assert exc_info.value.code == 404

    # 5. Test POST /api/save with malformed payload (returns 400)
    invalid_req = urllib.request.Request(  # noqa: S310
        f"{base_url}/api/save",
        data=b"not valid json",
        headers=post_headers,
        method="POST",
    )
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(invalid_req)  # noqa: S310
    assert exc_info.value.code == 400
    err_data = json.loads(exc_info.value.read().decode("utf-8"))
    exc_info.value.close()
    assert err_data["status"] == "error"

    # 6. Test POST /api/save with valid updated queries (returns 200)
    post_data = {
        "queries": [
            {"text": "Curated query 1", "expected_skill": "t-skill"},
            {"text": "Curated query 2", "expected_skill": "r-skill"},
            {"text": "Curated query 3", "expected_skill": None},
        ],
    }
    req = urllib.request.Request(  # noqa: S310
        f"{base_url}/api/save",
        data=json.dumps(post_data).encode("utf-8"),
        headers=post_headers,
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:  # noqa: S310
        assert resp.status == 200
        res = json.loads(resp.read().decode("utf-8"))
        assert res.get("status") == "ok"
        assert res.get("saved") == 3
        _assert_security_headers(resp)

    _assert_review_security_endpoints(url, post_data)


def _assert_review_security_endpoints(url: str, post_data: Mapping[str, object]) -> None:
    """Verify security controls on the running review server (DNS rebinding, CSRF, IPv6)."""
    parsed = urllib.parse.urlsplit(url)
    token = urllib.parse.parse_qs(parsed.query).get("token", [""])[0]
    base_url = f"{parsed.scheme}://{parsed.netloc}"

    # 7. Test security: Reject invalid Host header (DNS rebinding protection)
    bad_host_req = urllib.request.Request(  # noqa: S310
        f"{base_url}/api/status",
        headers={"Host": "attacker.com"},
    )
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(bad_host_req)  # noqa: S310
    exc_info.value.close()
    assert exc_info.value.code == 403

    # 8. Test security: Reject cross-origin requests (CSRF protection)
    csrf_req = urllib.request.Request(  # noqa: S310
        f"{base_url}/api/save",
        data=json.dumps(post_data).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Origin": "https://malicious-site.example.com",
            "X-Reach-Token": token,
        },
        method="POST",
    )
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(csrf_req)  # noqa: S310
    exc_info.value.close()
    assert exc_info.value.code == 403

    # 8b. Test security: Reject Origin: null (sandboxed iframe CSRF protection)
    null_origin_req = urllib.request.Request(  # noqa: S310
        f"{base_url}/api/save",
        data=json.dumps(post_data).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Origin": "null",
            "X-Reach-Token": token,
        },
        method="POST",
    )
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(null_origin_req)  # noqa: S310
    exc_info.value.close()
    assert exc_info.value.code == 403

    _assert_review_token_security(base_url, post_data, token)

    port = parsed.port

    # 9. Test security: Accept valid IPv6 Host header
    ipv6_host_req = urllib.request.Request(  # noqa: S310
        f"{base_url}/api/status",
        headers={"Host": f"[::1]:{port}"},
    )
    with urllib.request.urlopen(ipv6_host_req) as resp:  # noqa: S310
        assert resp.status == 200

    # 10. Test security: Accept valid IPv6 Origin header
    ipv6_headers: dict[str, str] = {
        "Content-Type": "application/json",
        "Origin": f"http://[::1]:{port}",
    }
    if token:
        ipv6_headers["X-Reach-Token"] = token
    ipv6_origin_req = urllib.request.Request(  # noqa: S310
        f"{base_url}/api/save",
        data=json.dumps(post_data).encode("utf-8"),
        headers=ipv6_headers,
        method="POST",
    )
    with urllib.request.urlopen(ipv6_origin_req) as resp:  # noqa: S310
        assert resp.status == 200


def _assert_review_token_security(
    base_url: str,
    post_data: Mapping[str, object],
    token: str,
) -> None:
    """Verify session token validation on GET and POST endpoints."""
    bad_token_req = urllib.request.Request(  # noqa: S310
        f"{base_url}/api/save",
        data=json.dumps(post_data).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Origin": base_url,
            "X-Reach-Token": "invalid-token",
        },
        method="POST",
    )
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(bad_token_req)  # noqa: S310
    exc_info.value.close()
    assert exc_info.value.code == 403

    if token:
        query_token_post_req = urllib.request.Request(  # noqa: S310
            f"{base_url}/api/save?token={token}",
            data=json.dumps(post_data).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Origin": base_url,
            },
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(query_token_post_req)  # noqa: S310
        exc_info.value.close()
        assert exc_info.value.code == 403

    unauth_get_req = urllib.request.Request(f"{base_url}/")  # noqa: S310
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(unauth_get_req)  # noqa: S310
    exc_info.value.close()
    assert exc_info.value.code == 403

    unauth_index_req = urllib.request.Request(f"{base_url}/index.html")  # noqa: S310
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(unauth_index_req)  # noqa: S310
    exc_info.value.close()
    assert exc_info.value.code == 403

    bad_token_get_req = urllib.request.Request(f"{base_url}/?token=wrong-token")  # noqa: S310
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(bad_token_get_req)  # noqa: S310
    exc_info.value.close()
    assert exc_info.value.code == 403

    if token:
        valid_index_req = urllib.request.Request(f"{base_url}/index.html?token={token}")  # noqa: S310
        with urllib.request.urlopen(valid_index_req) as resp:  # noqa: S310
            assert resp.status == 200


def test_launch_query_review_http_server_saves_and_shuts_down(
    write_skill: Callable[..., Path],
    clean_browser_env: dict[str, str],
) -> None:
    """Verify ephemeral HTTP server serves HTML, validates payloads, and saves curated queries."""
    from reach.queries import Origin, QuerySetProvenance

    target_dir = write_skill(name="t-skill", description="desc")
    target = Skill(name="t-skill", description="desc", path=target_dir)
    rival_dir = write_skill(name="r-skill", description="desc")
    rival = Skill(name="r-skill", description="desc", path=rival_dir)

    initial_queries = (
        Query(
            query_id="init-1",
            text="Initial query",
            expected_skill="t-skill",
            kind=QueryKind.IMPLICIT,
        ),
    )
    qs = QuerySet(
        catalog_id="test-cat",
        provenance=QuerySetProvenance(origin=Origin.AUTHORED),
        queries=initial_queries,
    )

    server_port: list[int] = []
    bg_error: list[Exception] = []

    def mock_webbrowser_open(url: str) -> bool:
        port = urllib.parse.urlsplit(url).port or int(url.rsplit(":", maxsplit=1)[-1])
        server_port.append(port)
        try:
            _assert_review_server_endpoints(url)
        except (urllib.error.URLError, AssertionError, json.JSONDecodeError) as exc:
            bg_error.append(exc)
        return True

    with (
        patch("sys.stdin.isatty", return_value=True),
        patch.dict("os.environ", clean_browser_env, clear=True),
        patch("webbrowser.open", side_effect=mock_webbrowser_open),
    ):
        curated_qs = launch_query_review(qs, target, [rival], timeout=5)

    if bg_error:
        raise bg_error[0]

    assert len(curated_qs.queries) == 3
    q1, q2, q3 = curated_qs.queries
    assert q1.text == "Curated query 1"
    assert q1.expected_skill == "t-skill"
    assert q1.kind == QueryKind.IMPLICIT

    assert q2.text == "Curated query 2"
    assert q2.expected_skill == "r-skill"
    assert q2.kind == QueryKind.NEIGHBOR_NEGATIVE

    assert q3.text == "Curated query 3"
    assert q3.expected_skill is None
    assert q3.kind == QueryKind.OUT_OF_SCOPE


def test_launch_query_review_terminal_enter_proceeds_with_defaults(
    write_skill: Callable[..., Path],
    clean_browser_env: dict[str, str],
) -> None:
    """Verify pressing [Enter] in terminal unblocks review and retains initial queries."""
    from reach.queries import Origin, QuerySetProvenance

    target_dir = write_skill(name="t-skill", description="desc")
    target = Skill(name="t-skill", description="desc", path=target_dir)

    initial_queries = (
        Query(
            query_id="init-1",
            text="Initial query",
            expected_skill="t-skill",
            kind=QueryKind.IMPLICIT,
        ),
    )
    qs = QuerySet(
        catalog_id="test-cat",
        provenance=QuerySetProvenance(origin=Origin.AUTHORED),
        queries=initial_queries,
    )

    with (
        patch("sys.stdin.isatty", return_value=True),
        patch.dict("os.environ", clean_browser_env, clear=True),
        patch("webbrowser.open", return_value=True),
        patch("reach.review.select.select", return_value=([True], [], [])),
        patch("sys.stdin.readline", return_value="\n"),
    ):
        result = launch_query_review(qs, target, [], timeout=5)

    assert result == qs


def test_convert_saved_queries_maps_kinds() -> None:
    """Verify _convert_saved_queries maps query kinds based on expected_skill."""
    assert ReviewSentinel.OUT_OF_SCOPE == "__OUT_OF_SCOPE__"
    items = [
        _ReviewQueryItem(text="  Trigger target  ", expected_skill="my-skill"),
        _ReviewQueryItem(text="Rival query", expected_skill="other-skill"),
        _ReviewQueryItem(text="Out of scope query", expected_skill=None),
    ]
    converted = _convert_saved_queries(items, "my-skill")
    assert len(converted) == 3

    assert converted[0].text == "Trigger target"
    assert converted[0].kind == QueryKind.IMPLICIT
    assert converted[0].expected_skill == "my-skill"

    assert converted[1].text == "Rival query"
    assert converted[1].kind == QueryKind.NEIGHBOR_NEGATIVE
    assert converted[1].expected_skill == "other-skill"

    assert converted[2].text == "Out of scope query"
    assert converted[2].kind == QueryKind.OUT_OF_SCOPE
    assert converted[2].expected_skill is None


def test_review_session_isolation() -> None:
    """Verify separate review sessions do not cross-contaminate state."""
    import io
    from email.message import Message
    from unittest.mock import MagicMock

    session1 = _ReviewSession(html_content="<h1>Session 1</h1>")
    session2 = _ReviewSession(html_content="<h1>Session 2</h1>")

    mock_server = MagicMock()
    handler1 = ReviewServerHandler.__new__(ReviewServerHandler)
    handler1.session = session1
    h1 = Message()
    h1["Host"] = "127.0.0.1"
    h1["Origin"] = "http://127.0.0.1:8000"
    h1["X-Reach-Token"] = session1.auth_token
    handler1.headers = h1
    handler1.server = mock_server

    handler2 = ReviewServerHandler.__new__(ReviewServerHandler)
    handler2.session = session2
    h2 = Message()
    h2["Host"] = "127.0.0.1"
    h2["Origin"] = "http://127.0.0.1:8000"
    handler2.headers = h2
    handler2.server = mock_server

    payload1 = json.dumps({"queries": [{"text": "Query 1", "expected_skill": "skill-1"}]}).encode(
        "utf-8",
    )
    handler1.rfile = io.BytesIO(payload1)
    handler1.wfile = io.BytesIO()
    handler1.headers["Content-Length"] = str(len(payload1))
    handler1.path = "/api/save"
    handler1.requestline = "POST /api/save HTTP/1.1"
    handler1.request_version = "HTTP/1.1"
    handler1.do_POST()

    assert session1.done_event.is_set()
    assert session1.saved_queries is not None
    assert len(session1.saved_queries) == 1
    assert session1.saved_queries[0].text == "Query 1"

    # Verify session 2 remains completely pristine and untouched
    assert not session2.done_event.is_set()
    assert session2.saved_queries is None


def test_review_query_item_validation_and_normalization() -> None:
    """Verify _ReviewPayload strips whitespace, rejects empty text, and normalizes skills."""
    from pydantic import ValidationError

    from reach.models import QueryKind
    from reach.review import _convert_saved_queries, _ReviewPayload

    with pytest.raises(ValidationError, match="String should have at least 1 character"):
        _ReviewPayload.model_validate({"queries": [{"text": "   "}]})

    payload = _ReviewPayload.model_validate(
        {"queries": [{"text": "  test query text  ", "expected_skill": ""}]},
    )
    assert payload.queries[0].text == "test query text"
    assert payload.queries[0].expected_skill is None
    assert isinstance(payload.queries, tuple)

    converted = _convert_saved_queries(payload.queries, "target-skill")
    assert len(converted) == 1
    assert converted[0].expected_skill is None
    assert converted[0].kind == QueryKind.OUT_OF_SCOPE


def test_review_query_item_unrecognized_expected_skill_rejected_with_context() -> None:
    """Verify unrecognized expected_skill is rejected when allowed_skills context is supplied."""
    from pydantic import ValidationError

    from reach.review import _ReviewPayload

    with pytest.raises(ValidationError, match="not in allowed catalog skills"):
        _ReviewPayload.model_validate(
            {"queries": [{"text": "test query", "expected_skill": "unrecognized-skill"}]},
            context={"allowed_skills": {"target-skill", "rival-skill"}},
        )

    # Allowed skills succeed
    valid_target = _ReviewPayload.model_validate(
        {"queries": [{"text": "test query", "expected_skill": "target-skill"}]},
        context={"allowed_skills": {"target-skill", "rival-skill"}},
    )
    assert valid_target.queries[0].expected_skill == "target-skill"

    # Out of scope (None) succeeds
    valid_oos = _ReviewPayload.model_validate(
        {"queries": [{"text": "test query", "expected_skill": None}]},
        context={"allowed_skills": {"target-skill", "rival-skill"}},
    )
    assert valid_oos.queries[0].expected_skill is None


@pytest.mark.parametrize(
    ("host", "expected"),
    [
        ("", True),
        ("localhost", True),
        ("localhost:8080", True),
        ("127.0.0.1", True),
        ("127.0.0.1:3000", True),
        ("testserver", True),
        ("testserver:80", True),
        ("[::1]", True),
        ("[::1]:8080", True),
        ("attacker.com", False),
        ("attacker.com:8080", False),
        ("127.0.0.1.attacker.com", False),
        ("evil.com:127.0.0.1", False),
    ],
)
def test_review_server_handler_is_valid_host(host: str, *, expected: bool) -> None:
    """Verify Host header validation permits local IPv4/IPv6 and blocks foreign hosts."""
    handler = object.__new__(ReviewServerHandler)
    msg = email.message.EmailMessage()
    if host:
        msg["Host"] = host
    handler.headers = msg
    assert handler._is_valid_host() is expected


@pytest.mark.parametrize(
    ("origin", "expected"),
    [
        (None, False),
        ("", False),
        ("null", False),
        ("http://127.0.0.1:8080", True),
        ("http://localhost:3000", True),
        ("http://[::1]:8080", True),
        ("https://localhost:443", True),
        ("https://attacker.com", False),
        ("http://127.0.0.1.attacker.com", False),
        ("javascript:void(0)", False),
    ],
)
def test_review_server_handler_is_valid_origin(origin: str | None, *, expected: bool) -> None:
    """Verify Origin header validation permits local loopback and blocks cross-origin requests."""
    handler = object.__new__(ReviewServerHandler)
    msg = email.message.EmailMessage()
    if origin is not None:
        msg["Origin"] = origin
    handler.headers = msg
    assert handler._is_valid_origin() is expected


def test_poll_terminal_enter_windows_msvcrt(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify _poll_terminal_enter uses msvcrt on Windows without calling select.select."""
    import sys
    import types

    from reach.review import _poll_terminal_enter

    fake_msvcrt = types.ModuleType("msvcrt")
    monkeypatch.setattr(fake_msvcrt, "kbhit", lambda: True, raising=False)
    monkeypatch.setattr(fake_msvcrt, "getwch", lambda: "\r", raising=False)
    monkeypatch.setattr("os.name", "nt")
    monkeypatch.setitem(sys.modules, "msvcrt", fake_msvcrt)

    assert _poll_terminal_enter() is True
