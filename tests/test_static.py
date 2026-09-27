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

"""Verify static CSS and JS asset loading and bundle assembly."""

from __future__ import annotations

import re
from collections.abc import Callable, Generator, Sequence
from importlib.resources import files

import pytest

from reach.static import (
    get_review_css,
    get_review_js,
    get_review_template,
    get_view_css,
    get_view_js,
    get_view_template,
    load_static_asset,
)


@pytest.fixture(autouse=True)
def clean_static_cache() -> Generator[None, None, None]:
    """Ensure in-memory static asset cache is cleared before and after each test."""
    load_static_asset.cache_clear()
    yield
    load_static_asset.cache_clear()


@pytest.mark.parametrize(
    ("filename", "forbidden_markers"),
    [
        ("tokens.css", ("/*", "*/")),
        ("base.css", ("/*", "*/")),
        ("view.html", ("<!--", "-->")),
        ("review.html", ("<!--", "-->")),
    ],
    ids=["tokens.css", "base.css", "view.html", "review.html"],
)
def test_load_static_asset_strips_comments(filename: str, forbidden_markers: Sequence[str]) -> None:
    """Verify load_static_asset strips comment blocks from CSS and HTML assets."""
    content = load_static_asset(filename)
    for marker in forbidden_markers:
        assert marker not in content


def test_load_static_asset_js_strips_header_comment_and_preserves_code() -> None:
    """Verify load_static_asset strips leading license block from JS while preserving code."""
    raw = files("reach.static").joinpath("review.js").read_text(encoding="utf-8")
    assert raw.startswith("/*")

    review_js = load_static_asset("review.js")
    assert not review_js.startswith("/*")
    assert "function getTargetSkill()" in review_js
    assert "function addRow(" in review_js
    assert len(review_js) > 1000


@pytest.mark.parametrize(
    ("scale_name", "expected_tokens"),
    [
        (
            "radius",
            (
                "--reach-radius-sm:",
                "--reach-radius-md:",
                "--reach-radius-lg:",
                "--reach-radius-xl:",
                "--reach-radius-full:",
            ),
        ),
        (
            "spacing",
            (
                "--reach-spacing-xs:",
                "--reach-spacing-sm:",
                "--reach-spacing-md:",
                "--reach-spacing-lg:",
                "--reach-spacing-xl:",
            ),
        ),
        (
            "color",
            (
                "--reach-surface-container-high:",
                "--reach-surface-container-highest:",
                "--reach-on-error:",
                "--reach-on-error-container:",
                "--reach-on-primary:",
            ),
        ),
        (
            "typography",
            ("ui-monospace", "Menlo", "-apple-system"),
        ),
    ],
    ids=["radius", "spacing", "color", "typography"],
)
def test_tokens_css_defines_material3_token_scales(
    scale_name: str, expected_tokens: Sequence[str]
) -> None:
    """Verify tokens.css defines all required radius, spacing, color, and font tokens."""
    tokens = load_static_asset("tokens.css")
    missing = [token for token in expected_tokens if token not in tokens]
    assert not missing, f"Missing {scale_name} tokens in tokens.css: {missing}"


def test_review_css_uses_tokenized_focus_ring_and_overlay() -> None:
    """Verify review.css does not hardcode decimal primary or white RGBA values."""
    review_css = load_static_asset("review.css")
    assert "rgba(11, 87, 208" not in review_css
    assert "rgba(255, 255, 255" not in review_css
    assert "color-mix(in srgb, var(--reach-primary)" in review_css
    assert "color-mix(in srgb, var(--reach-on-primary)" in review_css


def test_load_static_asset_nonexistent_raises_file_not_found() -> None:
    """Verify load_static_asset raises FileNotFoundError for missing assets."""
    with pytest.raises(FileNotFoundError):
        load_static_asset("non_existent_stylesheet.css")


@pytest.mark.parametrize(
    "asset_fn",
    [
        get_review_css,
        get_view_css,
        get_review_js,
        get_view_js,
        get_review_template,
        get_view_template,
    ],
    ids=["review_css", "view_css", "review_js", "view_js", "review_tmpl", "view_tmpl"],
)
def test_bundled_assets_contain_no_external_urls(
    asset_fn: Callable[[], str],
    external_reference_re: re.Pattern[str],
) -> None:
    """Verify bundled static assets and templates are self-contained without external URLs."""
    assert not external_reference_re.search(asset_fn())


@pytest.mark.parametrize(
    ("css_getter", "expected_rules"),
    [
        (
            get_review_css,
            (":root", "body", ".territory-board", ".balance-fill-triggers"),
        ),
        (
            get_view_css,
            (":root", "body", "table.confusion", ".figure-cell"),
        ),
    ],
    ids=["review_css", "view_css"],
)
def test_css_bundles_contain_expected_rules(
    css_getter: Callable[[], str],
    expected_rules: Sequence[str],
) -> None:
    """Verify get_review_css and get_view_css combine tokens, base, and view-specific rules."""
    css = css_getter()
    missing = [rule for rule in expected_rules if rule not in css]
    assert not missing, f"Missing rules in bundled CSS: {missing}"


@pytest.mark.parametrize(
    ("js_getter", "expected_symbols"),
    [
        (
            get_review_js,
            ("getAllCards", "approveQueries", "exportEvalSetJson", "addEventListener"),
        ),
        (
            get_view_js,
            (
                "reachFilterRows",
                "reachFilterQueries",
                "reachToggleAllQueries",
                "reachApplyFilter",
                "reachClearActiveFilter",
            ),
        ),
    ],
    ids=["review_js", "view_js"],
)
def test_js_modules_contain_expected_client_logic(
    js_getter: Callable[[], str],
    expected_symbols: Sequence[str],
) -> None:
    """Verify interactive client JavaScript modules load expected client functions."""
    js = js_getter()
    missing = [symbol for symbol in expected_symbols if symbol not in js]
    assert not missing, f"Missing symbols in client JS: {missing}"
