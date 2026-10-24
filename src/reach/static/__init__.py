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

"""Load static design assets and CSS stylesheets for HTML report generation."""

from __future__ import annotations

import re
from functools import cache
from importlib.resources import files

__all__ = [
    "get_review_css",
    "get_review_js",
    "get_review_template",
    "get_sweep_css",
    "get_sweep_template",
    "get_view_css",
    "get_view_js",
    "get_view_template",
    "load_static_asset",
]

_CSS_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_JS_HEADER_COMMENT_RE = re.compile(r"^\s*/\*.*?\*/\s*", re.DOTALL)


@cache
def load_static_asset(filename: str) -> str:
    """Load a static text resource from the reach.static package with in-memory caching."""
    raw = files("reach.static").joinpath(filename).read_text(encoding="utf-8")
    if filename.endswith(".html"):
        return _HTML_COMMENT_RE.sub("", raw).strip()
    if filename.endswith(".css"):
        return _CSS_COMMENT_RE.sub("", raw).strip()
    if filename.endswith(".js"):
        return _JS_HEADER_COMMENT_RE.sub("", raw).strip()
    return raw.strip()


def _bundle_css(sheet_name: str) -> str:
    """Bundle design tokens, shared base surfaces, and a target stylesheet."""
    return "\n\n".join(
        [
            load_static_asset("tokens.css"),
            load_static_asset("base.css"),
            load_static_asset(sheet_name),
        ]
    )


def get_review_css() -> str:
    """Combine tokens, base, and review CSS into a single self-contained stylesheet."""
    return _bundle_css("review.css")


def get_view_css() -> str:
    """Combine tokens, base, and view CSS into a single self-contained stylesheet."""
    return _bundle_css("view.css")


def get_review_js() -> str:
    """Load the interactive client JavaScript for boundary curation."""
    return load_static_asset("review.js")


def get_view_js() -> str:
    """Load the interactive client JavaScript for diagnostic workbench filtering."""
    return load_static_asset("view.js")


def get_review_template() -> str:
    """Load the outer HTML template skeleton for boundary curation."""
    return load_static_asset("review.html")


def get_view_template() -> str:
    """Load the outer HTML template skeleton for diagnostic workbench reports."""
    return load_static_asset("view.html")


def get_sweep_css() -> str:
    """Combine tokens, base, and sweep CSS into a single self-contained stylesheet."""
    return _bundle_css("sweep.css")


def get_sweep_template() -> str:
    """Load the outer HTML template skeleton for scaling sweep reports."""
    return load_static_asset("sweep.html")
