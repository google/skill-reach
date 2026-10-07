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

"""Provide ephemeral HTTP server and browser UI for interactive query boundary curation."""

from __future__ import annotations

import contextlib
import functools
import html
import http.server
import json
import os
import secrets
import select
import socketserver
import sys
import threading
import time
import urllib.parse
import webbrowser
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Annotated, Self, cast, override

from pydantic import (
    BaseModel,
    ConfigDict,
    StringConstraints,
    ValidationInfo,
    field_validator,
    model_validator,
)

from reach.models import Query, QueryKind, Skill
from reach.queries import QuerySet, QuerySetProvenance
from reach.static import get_review_css, get_review_js, get_review_template
from reach.view import _esc
from reach.views import build_console

if TYPE_CHECKING:
    import socket
    from collections.abc import Sequence

__all__ = [
    "ReviewSentinel",
    "ReviewServerHandler",
    "launch_query_review",
    "render_query_review_html",
]

#: Inlined CSS styling for the split territory boundary curation interface.
_REVIEW_STYLE = get_review_css()

#: Inlined client-side JavaScript for keyboard navigation and interactive curation.
_REVIEW_SCRIPT = get_review_js()

#: Allowed hostnames for loopback / local review server validation.
_ALLOWED_HOSTS: frozenset[str] = frozenset({"127.0.0.1", "localhost", "testserver", "::1"})

#: Poll interval (in seconds) for the background HTTP review server thread.
_SERVER_POLL_INTERVAL: float = 0.05

#: Poll timeout (in seconds) when checking terminal stdin for Enter confirmation.
_TERMINAL_POLL_TIMEOUT: float = 0.02


class ReviewSentinel(StrEnum):
    """Sentinel values used across the interactive review interface."""

    OUT_OF_SCOPE = "__OUT_OF_SCOPE__"


def render_query_review_html(
    query_set: QuerySet,
    skill: Skill,
    rivals: Sequence[Skill],
) -> str:
    """Render an interactive standalone HTML boundary curation interface for a QuerySet."""
    rival_names = [r.name for r in rivals if r.name != skill.name]
    target_name = skill.name

    trigger_cards = []
    guardrail_cards = []

    for q in query_set.queries:
        exp = q.expected_skill
        is_trigger = exp == target_name
        kind_name = q.kind.value if q.kind else "implicit"

        oos_val = ReviewSentinel.OUT_OF_SCOPE.value
        rival_options = [f'<option value="{oos_val}">Out of Scope (General Distractor)</option>']
        for r_name in rival_names:
            sel = ' selected="selected"' if exp == r_name else ""
            rival_options.append(
                f'<option value="{_esc(r_name)}"{sel}>Rival: {_esc(r_name)}</option>'
            )

        rival_select_disp = "none" if is_trigger else "block"
        swap_btn_text = "⇄ Move to Guardrails" if is_trigger else "⇄ Move to Triggers"

        if is_trigger:
            badge_target = f'<span class="badge-target">→ {_esc(target_name)}</span>'
        elif exp and exp in rival_names:
            badge_target = f'<span class="badge-rival">→ {_esc(exp)}</span>'
        else:
            badge_target = '<span class="badge-distractor">→ Out of Scope</span>'

        card_html = (
            f'<div class="query-card">'
            f'  <div class="card-top">'
            f'    <div class="card-badges">'
            f'      <span class="badge-kind">{_esc(kind_name)}</span>'
            f'      <span class="badge-target-wrapper">{badge_target}</span>'
            f"    </div>"
            f'    <button type="button" class="card-btn card-btn-delete"'
            f' onclick="deleteCard(this)" title="Delete query">✕</button>'
            f"  </div>"
            f'  <div class="query-preview query-text-preview"'
            f' onclick="editCard(this)">{_esc(q.text)}</div>'
            f'  <div class="query-card-editor">'
            f'    <textarea class="query-input"'
            f' placeholder="Enter query prompt...">{_esc(q.text)}</textarea>'
            f'    <select class="rival-select" style="display:{rival_select_disp};"'
            f' onchange="updateCardRival(this)">'
            f"      {''.join(rival_options)}"
            f"    </select>"
            f"  </div>"
            f'  <div class="card-actions">'
            f'    <div class="card-left-actions">'
            f'      <button type="button" class="card-btn card-btn-swap"'
            f' onclick="swapCard(this)">{swap_btn_text}</button>'
            f"    </div>"
            f'    <div class="card-right-actions">'
            f'      <button type="button" class="card-btn card-btn-edit"'
            f' onclick="editCard(this)">✎ Edit</button>'
            f'      <button type="button" class="card-btn card-btn-done"'
            f' onclick="closeCard(this)">✓ Done</button>'
            f"    </div>"
            f"  </div>"
            f"</div>"
        )

        if is_trigger:
            trigger_cards.append(card_html)
        else:
            guardrail_cards.append(card_html)

    rivals_json = html.escape(json.dumps(rival_names), quote=True)
    initial_count = len(query_set.queries)
    empty_style = "display: none;" if initial_count > 0 else ""
    footer_text = (
        f"{initial_count} queries total: {len(trigger_cards)} should trigger, "
        f"{len(guardrail_cards)} should not trigger"
    )

    rival_chips = "".join(f'<span class="rival-chip">{_esc(r)}</span>' for r in rival_names)
    no_rivals_msg = (
        '<div class="rivals-bar"><span>No competing skills detected in neighborhood.</span></div>'
    )
    rivals_section = (
        f'<div class="rivals-bar"><span>Competing rivals:</span> {rival_chips}</div>'
        if rival_names
        else no_rivals_msg
    )

    trig_pct = (len(trigger_cards) / (initial_count or 1)) * 100
    guard_pct = (len(guardrail_cards) / (initial_count or 1)) * 100
    trig_empty_style = "display:none;" if trigger_cards else ""
    guard_empty_style = "display:none;" if guardrail_cards else ""

    return get_review_template().format(
        skill_name=_esc(skill.name),
        style=_REVIEW_STYLE,
        target_skill=_esc(target_name),
        catalog_id=_esc(query_set.catalog_id),
        origin=_esc(query_set.provenance.origin.value),
        rivals_json=rivals_json,
        out_of_scope=ReviewSentinel.OUT_OF_SCOPE.value,
        skill_desc=_esc(skill.description),
        rivals_section=rivals_section,
        summary=footer_text,
        trig_pct=f"{trig_pct:.1f}",
        guard_pct=f"{guard_pct:.1f}",
        triggers_count=len(trigger_cards),
        triggers_cards="".join(trigger_cards),
        trig_empty_style=trig_empty_style,
        guardrails_count=len(guardrail_cards),
        guardrails_cards="".join(guardrail_cards),
        guard_empty_style=guard_empty_style,
        empty_style=empty_style,
        script=_REVIEW_SCRIPT,
    )


class _ReviewQueryItem(BaseModel):
    """Represent a single query item received from the curation UI."""

    # UI payload flexibility: UI may submit extraneous metadata or layout attributes
    model_config = ConfigDict(frozen=True, extra="ignore")

    text: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
    expected_skill: str | None = None

    @field_validator("expected_skill", mode="before")
    @classmethod
    def _empty_is_absent(cls, value: object) -> object:
        """Convert empty strings to None for optional skill names."""
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @model_validator(mode="after")
    def _validate_expected_skill(self, info: ValidationInfo) -> Self:
        """Validate expected skill name against catalog scope when context is available."""
        if info.context and "allowed_skills" in info.context:
            allowed = info.context["allowed_skills"]
            if self.expected_skill is not None and self.expected_skill not in allowed:
                msg = f"expected_skill {self.expected_skill!r} is not in allowed catalog skills"
                raise ValueError(msg)
        return self


class _ReviewPayload(BaseModel):
    """Represent the JSON payload submitted by the curation UI upon approval."""

    # UI payload flexibility: browser JSON may include extra client-side state
    model_config = ConfigDict(frozen=True, extra="ignore")

    queries: tuple[_ReviewQueryItem, ...]


@dataclass
class _ReviewSession:
    """Encapsulate session-specific state for an ephemeral review server."""

    html_content: str
    auth_token: str = field(default_factory=lambda: secrets.token_urlsafe(16))
    done_event: threading.Event = field(default_factory=threading.Event)
    saved_queries: tuple[_ReviewQueryItem, ...] | None = None
    allowed_skills: set[str] = field(default_factory=set)


class ReviewServerHandler(http.server.BaseHTTPRequestHandler):
    """Handle HTTP requests for the ephemeral interactive review server."""

    def __init__(
        self,
        request: socket.socket | tuple[bytes, socket.socket],
        client_address: tuple[str, int] | str,
        server: socketserver.BaseServer,
        *,
        session: _ReviewSession | None = None,
    ) -> None:
        """Initialize the request handler with a bound review session."""
        self.session = session if session is not None else _ReviewSession(html_content="")
        super().__init__(request, client_address, server)

    def _is_valid_host(self) -> bool:
        """Validate Host header to prevent DNS rebinding attacks."""
        host = self.headers.get("Host", "")
        if not host:
            return True
        try:
            parsed = urllib.parse.urlsplit(f"//{host}")
            hostname = (parsed.hostname or "").lower()
        except ValueError:
            return False
        return hostname in _ALLOWED_HOSTS

    def _is_valid_origin(self) -> bool:
        """Validate Origin header on mutating requests to protect against CSRF."""
        origin = self.headers.get("Origin")
        if not origin or origin == "null":
            return False
        try:
            parsed = urllib.parse.urlsplit(origin)
        except ValueError:
            return False
        if parsed.scheme not in ("http", "https"):
            return False
        hostname = (parsed.hostname or "").lower()
        return hostname in _ALLOWED_HOSTS

    def _send_security_headers(self) -> None:
        """Add defensive security headers to all responses."""
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self' 'unsafe-inline'; connect-src 'self';",
        )

    def _send_json(self, status: int, payload: object) -> None:
        """Serialize and transmit a JSON response with standard security headers."""
        resp = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(resp)))
        self._send_security_headers()
        self.end_headers()
        self.wfile.write(resp)

    def do_GET(self) -> None:
        """Serve the review HTML page or health status."""
        if not self._is_valid_host():
            self.send_error(403, "Forbidden: Invalid Host header")
            return

        path_clean = urllib.parse.urlsplit(self.path).path
        if path_clean in ("/", "/index.html"):
            if not self._is_valid_token():
                self.send_error(403, "Forbidden: Invalid or missing session token")
                return
            html_source = self.session.html_content
            token = self.session.auth_token
            if token and "</head>" in html_source:
                meta_tag = (
                    f'<meta name="reach-token" content="{html.escape(token, quote=True)}">\n</head>'
                )
                html_source = html_source.replace("</head>", meta_tag, 1)
            encoded = html_source.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self._send_security_headers()
            self.end_headers()
            self.wfile.write(encoded)
        elif path_clean == "/api/status":
            self._send_json(200, {"status": "running"})
        else:
            self.send_error(404, "Not Found")

    def _is_valid_token(self) -> bool:
        """Validate ephemeral session authentication token if configured."""
        expected_token = self.session.auth_token
        if not expected_token:
            return True
        provided_token = self.headers.get("X-Reach-Token")
        if not provided_token and self.command == "GET":
            query_params = urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query)
            provided_token = query_params.get("token", [None])[0]
        return bool(provided_token and secrets.compare_digest(provided_token, expected_token))

    def do_POST(self) -> None:
        """Receive and validate curated queries from the browser."""
        if not self._is_valid_host():
            self.send_error(403, "Forbidden: Invalid Host header")
            return

        if not self._is_valid_origin():
            self.send_error(403, "Forbidden: Cross-origin request rejected")
            return

        path_clean = urllib.parse.urlsplit(self.path).path
        if path_clean != "/api/save":
            self.send_error(404, "Not Found")
            return

        if not self._is_valid_token():
            self.send_error(403, "Forbidden: Invalid or missing session token")
            return

        try:
            content_len = int(self.headers.get("Content-Length", 0))
            if content_len < 0 or content_len > 5 * 1024 * 1024:
                msg = "Content-Length must be between 0 and 5MB"
                raise ValueError(msg)
        except (ValueError, TypeError) as exc:
            self._send_json(400, {"status": "error", "message": str(exc)})
            return

        post_body = self.rfile.read(content_len)
        try:
            data = json.loads(post_body.decode("utf-8"))
            allowed = self.session.allowed_skills
            context = {"allowed_skills": allowed} if allowed else None
            payload = _ReviewPayload.model_validate(data, context=context)
        except (json.JSONDecodeError, ValueError) as exc:
            self._send_json(400, {"status": "error", "message": str(exc)})
            return

        # Save queries and unblock review loop
        self.session.saved_queries = payload.queries
        self.session.done_event.set()

        self._send_json(200, {"status": "ok", "saved": len(payload.queries)})

    @override
    def log_message(self, format: str, *args: object) -> None:
        """Suppress standard HTTP request logging to keep console clean."""


def _convert_saved_queries(
    saved: Sequence[_ReviewQueryItem],
    target_skill_name: str,
) -> list[Query]:
    """Convert curated query items into validated Query instances."""
    updated_queries: list[Query] = []
    for i, item in enumerate(saved):
        text = item.text
        expected_skill = item.expected_skill

        if expected_skill == target_skill_name:
            kind = QueryKind.IMPLICIT
        elif expected_skill:
            kind = QueryKind.NEIGHBOR_NEGATIVE
        else:
            kind = QueryKind.OUT_OF_SCOPE

        updated_queries.append(
            Query(
                query_id=f"curated-{i + 1}",
                text=text,
                expected_skill=expected_skill,
                kind=kind,
            )
        )
    return updated_queries


def _poll_terminal_enter(timeout: float = _TERMINAL_POLL_TIMEOUT) -> bool:
    """Poll terminal stdin for an Enter keypress across Windows and POSIX."""
    with contextlib.suppress(Exception):
        if os.name == "nt":
            import msvcrt  # type: ignore[import-not-found]

            if msvcrt.kbhit():  # type: ignore[attr-defined]
                ch = msvcrt.getwch()  # type: ignore[attr-defined]
                return ch in ("\r", "\n")
            return False
        rlist, _, _ = select.select([sys.stdin], [], [], timeout)
        if rlist:
            line = sys.stdin.readline()
            return bool(line)
    return False


class _LocalThreadingHTTPServer(http.server.ThreadingHTTPServer):
    """ThreadingHTTPServer for 127.0.0.1 that skips blocking reverse DNS in server_bind."""

    @override
    def server_bind(self) -> None:
        """Bind socket without calling socket.getfqdn on loopback."""
        socketserver.TCPServer.server_bind(self)
        host, port = self.server_address[:2]
        self.server_name = str(host)
        self.server_port = int(port)


def launch_query_review(
    query_set: QuerySet,
    skill: Skill,
    rivals: Sequence[Skill],
    *,
    timeout: float = 600.0,
) -> QuerySet:
    """Launch interactive browser review on an ephemeral HTTP server on 127.0.0.1.

    If in a headless or non-interactive environment, the review is gracefully bypassed.
    """
    console = build_console()

    # Bypass review if non-interactive, explicitly in CI, or REACH_NO_BROWSER is set
    no_browser = os.environ.get("REACH_NO_BROWSER", "").lower() in ("true", "1")
    is_ci = os.environ.get("CI", "").lower() in ("true", "1")
    if not sys.stdin.isatty() or is_ci or no_browser:
        n_queries = len(query_set.queries)
        console.print(
            f"[dim]Non-interactive environment; proceeding with {n_queries} queries.[/dim]"
        )
        return query_set

    # Render HTML content with target skill and rival options
    html_content = render_query_review_html(query_set, skill, rivals)

    allowed_skills = {skill.name, *(r.name for r in rivals)}
    session = _ReviewSession(
        html_content=html_content,
        allowed_skills=allowed_skills,
    )

    handler = functools.partial(ReviewServerHandler, session=session)
    server = _LocalThreadingHTTPServer(
        ("127.0.0.1", 0),
        cast(type[http.server.BaseHTTPRequestHandler], handler),
    )
    server_port = server.server_port
    url = f"http://127.0.0.1:{server_port}/?token={session.auth_token}"

    server_thread = threading.Thread(
        target=server.serve_forever,
        kwargs={"poll_interval": _SERVER_POLL_INTERVAL},
        daemon=True,
    )
    server_thread.start()

    console.print(
        f"[bold blue]Launched interactive boundary review on[/bold blue] [link={url}]{url}[/link]"
    )
    console.print(
        "[dim]Press [Enter] in this terminal to proceed immediately with current queries.[/dim]"
    )

    with contextlib.suppress(Exception):
        webbrowser.open(url)

    # Wait for either browser approval, terminal [Enter], or timeout
    start_time = time.monotonic()
    try:
        while not session.done_event.is_set():
            if time.monotonic() - start_time > timeout:
                console.print(
                    f"\n[dim]Review timed out after {int(timeout)}s; "
                    "proceeding with current queries.[/dim]"
                )
                break

            if _poll_terminal_enter(timeout=_TERMINAL_POLL_TIMEOUT):
                console.print("\n[dim]Proceeding from terminal approval...[/dim]")
                break

            session.done_event.wait(timeout=0.02)
    finally:
        server.shutdown()
        server_thread.join(timeout=2.0)
        server.server_close()

    # If queries were saved through the browser, convert them into QuerySet
    saved = session.saved_queries
    if saved is not None:
        updated_queries = _convert_saved_queries(saved, skill.name)
        if updated_queries:
            console.print(
                f"[bold green]✓[/bold green] Curated {len(updated_queries)} "
                f"queries for '{skill.name}'."
            )
            return QuerySet(
                catalog_id=query_set.catalog_id,
                provenance=QuerySetProvenance(origin=query_set.provenance.origin),
                queries=tuple(updated_queries),
            )

    return query_set
