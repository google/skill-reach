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

"""Verify Claude Code skill listing budget calculations, fitting, and character limits."""

from __future__ import annotations

import json
import math
from typing import Any

import pytest

from reach.config import RuntimeSettings
from reach.models import Catalog, CatalogMode
from reach.runtime.claude_code import (
    DEFAULT_CHARS_PER_TOKEN,
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_ENABLED_PLUGINS,
    DEFAULT_LISTING_BUDGET_CHARS,
    DEFAULT_SKILL_OVERRIDES,
    ClaudeCodeOptions,
    ClaudeCodeRuntime,
    budget_fraction_for,
    display_width,
    fit_skill_listing,
    listing_budget_chars,
    listing_chars,
)


@pytest.fixture
def three(corpus_builder) -> Any:
    """Provide three sample skills with descriptions of lengths 10, 20, and 30."""
    return (
        corpus_builder()
        .add("alpha", "a" * 10)
        .add("bravo", "b" * 20)
        .add("charlie", "c" * 30)
        .build_skills()
    )


def test_the_listing_controls_are_unset_by_default() -> None:
    """Verify listing budget fraction and max desc chars default to None in options."""
    options = ClaudeCodeOptions()
    assert options.skill_listing_budget_fraction is None
    assert options.skill_listing_max_desc_chars is None


def test_an_unset_control_leaves_no_trace_in_the_serialized_options() -> None:
    """Verify unset listing fields are excluded from model_dump serialization."""
    dumped = ClaudeCodeOptions().model_dump(mode="json")
    assert "skill_listing_budget_fraction" not in dumped
    assert "skill_listing_max_desc_chars" not in dumped


def test_a_set_control_does_appear_and_does_move_the_arm() -> None:
    """Verify configured listing parameters appear in resolved_options output."""
    plain = RuntimeSettings(agent="claude-code")
    tuned = RuntimeSettings(
        agent="claude-code",
        options={"skill_listing_max_desc_chars": 239},
    )
    assert "skill_listing_max_desc_chars" in tuned.resolved_options()
    assert plain.resolved_options() != tuned.resolved_options()


@pytest.mark.parametrize(
    ("field", "value", "key"),
    [
        ("skill_listing_budget_fraction", 0.06, "skillListingBudgetFraction"),
        ("skill_listing_max_desc_chars", 239, "skillListingMaxDescChars"),
    ],
)
def test_a_set_control_reaches_the_command_line_under_the_runtimes_own_name(
    field: str,
    value: float,
    key: str,
) -> None:
    """Verify listing options serialize into camelCase keys in settings JSON."""
    options = ClaudeCodeOptions.model_validate({field: value})
    assert json.loads(options.settings_json() or "{}")[key] == value


def test_the_settings_payload_is_unchanged_when_neither_control_is_set() -> None:
    """Verify settings_json maintains default JSON payload when listing options are unset."""
    assert ClaudeCodeOptions().settings_json() == json.dumps(
        {
            "disableBundledSkills": True,
            "enabledPlugins": dict(DEFAULT_ENABLED_PLUGINS),
            "skillOverrides": dict(DEFAULT_SKILL_OVERRIDES),
        },
        sort_keys=True,
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("skill_listing_budget_fraction", 0.0),
        ("skill_listing_budget_fraction", 1.5),
        ("skill_listing_max_desc_chars", 0),
        ("skill_listing_max_desc_chars", -1),
    ],
)
def test_a_control_outside_its_range_is_refused_at_load_time(
    field: str,
    value: float,
) -> None:
    """Verify ValueError is raised when listing settings are outside valid numeric ranges."""
    with pytest.raises(ValueError, match=field):
        ClaudeCodeOptions.model_validate({field: value})


@pytest.mark.parametrize(
    ("text", "columns", "why"),
    [
        ("- alpha: aaa", 12, "plain text is one column each"),
        ("a\nb", 2, "a newline prints as nothing"),
        ("á", 1, "a combining acute rides on its letter"),
        ("中文", 4, "an ideograph takes two columns"),
        ("—", 1, "an em dash is ambiguous, and asked for narrow"),
    ],
)
def test_the_listing_is_measured_in_columns_rather_than_characters(
    text: str,
    columns: int,
    why: str,
) -> None:
    """Verify display_width calculates expected terminal column display width."""
    assert display_width(text) == columns, why


def test_the_listing_costs_its_entries_and_the_newlines_between_them(three) -> None:
    """Verify listing_chars sums entry character lengths and separator newlines."""
    entries = len("- alpha: ") + 10 + len("- bravo: ") + 20 + len("- charlie: ") + 30
    assert listing_chars(three) == entries + 2
    assert listing_chars(three) == 91


def test_a_lone_entry_pays_for_no_separator(three) -> None:
    """Verify single skill listing excludes trailing newline character length."""
    assert listing_chars(three[:1]) == len("- alpha: aaaaaaaaaa")
    assert listing_chars([]) == 0


def test_a_declared_maximum_shortens_the_descriptions_that_exceed_it(three) -> None:
    """Verify max_desc_chars caps description lengths in listing_chars calculations."""
    assert listing_chars(three, max_desc_chars=15) < listing_chars(three)
    assert listing_chars(three, max_desc_chars=100) == listing_chars(three)


def test_a_listing_inside_its_budget_is_left_alone(three) -> None:
    """Verify fit_skill_listing retains all descriptions when within budget."""
    fitted = fit_skill_listing(three, budget_chars=10_000)
    assert not fitted.over_budget
    assert fitted.fits
    assert fitted.name_only == ()
    assert len(fitted.described) == 3


def test_over_budget_a_description_is_dropped_whole_rather_than_shortened(
    three,
) -> None:
    """Verify descriptions are omitted completely rather than truncated when over budget."""
    fitted = fit_skill_listing(three, budget_chars=49)
    assert fitted.over_budget
    assert fitted.fits
    for entry in fitted.entries:
        assert entry.chars in {entry.full_chars, display_width(f"- {entry.skill}")}


def test_an_entry_too_expensive_to_restore_is_skipped_and_the_walk_continues(
    three,
) -> None:
    """Verify fitting continues past oversized entries to include smaller matching skills."""
    fitted = fit_skill_listing(
        three,
        budget_chars=49,
        priority={"charlie": 3.0, "alpha": 2.0, "bravo": 1.0},
    )
    assert fitted.described == ("alpha",)
    assert set(fitted.name_only) == {"bravo", "charlie"}


def test_the_priority_order_decides_who_keeps_a_description(three) -> None:
    """Verify priority dictionary weights determine description inclusion order."""
    by_default = fit_skill_listing(three, budget_chars=49)
    reordered = fit_skill_listing(three, budget_chars=49, priority={"bravo": 1.0})
    assert by_default.described == ("alpha",), "ties leave the given order alone"
    assert reordered.described == ("bravo",)


def test_a_protected_entry_keeps_its_description_and_crowds_out_the_rest(
    three,
) -> None:
    """Verify protected skill descriptions are prioritized before remaining budget allocation."""
    open_field = fit_skill_listing(three, budget_chars=60)
    shielded = fit_skill_listing(three, budget_chars=60, protected=["charlie"])
    assert open_field.described == ("alpha", "bravo"), "charlie is too expensive"
    assert shielded.described == ("charlie",), "and now nothing else fits at all"
    assert shielded.entries[2].protected


def test_a_budget_too_small_for_the_names_alone_is_reported_not_papered_over(
    three,
) -> None:
    """Verify fit_skill_listing sets fits=False when budget cannot accommodate skill names."""
    fitted = fit_skill_listing(three, budget_chars=10)
    assert not fitted.fits
    assert fitted.described == ()
    assert fitted.chars == 25, "the floor, which is what the runtime would emit"


def test_the_default_budget_is_the_one_that_was_measured() -> None:
    """Verify DEFAULT_LISTING_BUDGET_CHARS is 30,000 and used as default."""
    assert DEFAULT_LISTING_BUDGET_CHARS == 30_000
    assert fit_skill_listing([]).budget_chars == DEFAULT_LISTING_BUDGET_CHARS


def test_an_unset_fraction_prices_at_the_budget_that_was_observed() -> None:
    """Verify listing_budget_chars returns DEFAULT_LISTING_BUDGET_CHARS when fraction is None."""
    assert listing_budget_chars() == DEFAULT_LISTING_BUDGET_CHARS
    assert listing_budget_chars(None) == DEFAULT_LISTING_BUDGET_CHARS


def test_a_declared_fraction_prices_at_the_derivation_the_runtime_uses() -> None:
    """Verify listing_budget_chars calculates floor budget based on fraction."""
    assert listing_budget_chars(0.01) == DEFAULT_LISTING_BUDGET_CHARS
    assert listing_budget_chars(0.02) == 60_000
    assert listing_budget_chars(0.0000001) == 0, "floored, not rounded"


@pytest.mark.parametrize(
    "chars",
    [51_910, 54_000, 30_000, 3_000, 1, 2_999_999, 3_000_000],
)
def test_a_suggested_fraction_prices_at_or_above_the_listing_that_asked(
    chars: int,
) -> None:
    """Verify budget_fraction_for produces budget greater than or equal to requested chars."""
    fraction = budget_fraction_for(chars)
    assert fraction is not None
    assert listing_budget_chars(fraction) >= chars
    if chars == 51_910:
        assert fraction == 0.018


def test_a_listing_wider_than_the_window_is_not_a_budget_to_raise() -> None:
    """Verify budget_fraction_for returns None for character counts exceeding window size."""
    window = DEFAULT_CONTEXT_WINDOW * DEFAULT_CHARS_PER_TOKEN
    assert budget_fraction_for(window) == 1.0
    assert budget_fraction_for(window + 1) is None


@pytest.fixture
def three_catalog() -> Catalog:
    """Provide a Catalog model referencing the three sample skills."""
    return Catalog(
        id="three",
        mode=CatalogMode.ALL,
        skills=("alpha", "bravo", "charlie"),
    )


def _tuned(**options: object) -> ClaudeCodeRuntime:
    """Instantiate ClaudeCodeRuntime with specified options."""
    return ClaudeCodeRuntime(RuntimeSettings(agent="claude-code", options=options))


def test_a_catalog_inside_the_budget_is_reported_whole(three, three_catalog) -> None:
    """Verify fit reports whole=True and empty remedy string when catalog fits budget."""
    fit = _tuned().fit(three_catalog, three)
    assert fit.whole
    assert fit.rations
    assert (fit.allowed, fit.asked) == (DEFAULT_LISTING_BUDGET_CHARS, 91)
    assert fit.unit == "display columns"
    assert fit.remedy == ""


def test_a_catalog_over_the_budget_reports_what_it_would_cost_the_descriptions(
    three,
    three_catalog,
) -> None:
    """Verify fit calculates truncated description count and remedy when catalog exceeds budget."""
    fit = _tuned(skill_listing_budget_fraction=0.000013).fit(three_catalog, three)
    assert not fit.whole
    assert (fit.allowed, fit.asked) == (39, 91)
    assert fit.truncated == 2, "alpha's description costs 12 of the 14; the others more"
    assert "skill_listing_budget_fraction = 0.001" in fit.remedy
    assert "[runtime.options]" in fit.remedy


def test_a_catalog_no_fraction_could_hold_says_to_hold_fewer_skills(
    corpus_builder,
) -> None:
    """Verify remedy suggests reducing catalog size when characters exceed maximum window."""
    window = DEFAULT_CONTEXT_WINDOW * DEFAULT_CHARS_PER_TOKEN
    huge = corpus_builder().add("huge", "h" * window).build_skills()
    catalog = Catalog(id="one", mode=CatalogMode.ALL, skills=("huge",))
    fit = _tuned().fit(catalog, huge)
    assert not fit.whole
    assert "hold fewer skills resident" in fit.remedy


def test_the_per_description_ceiling_is_applied_before_the_budget(
    three,
    three_catalog,
) -> None:
    """Verify skill_listing_max_desc_chars limits description size prior to budget check."""
    tight = _tuned(skill_listing_budget_fraction=0.000013)
    capped = _tuned(
        skill_listing_budget_fraction=0.000013,
        skill_listing_max_desc_chars=2,
    )
    assert not tight.fit(three_catalog, three).whole
    assert capped.fit(three_catalog, three).whole


def test_a_fit_is_asked_about_the_catalog_rather_than_the_corpus(
    three,
    three_catalog,
) -> None:
    """Verify fit evaluates character count of catalog subset rather than entire corpus."""
    pair = three_catalog.model_copy(update={"skills": ("alpha", "bravo")})
    assert _tuned().fit(pair, three).asked == listing_chars(three[:2])


def test_a_catalog_naming_an_unloaded_skill_is_refused_here_too(
    three,
    three_catalog,
) -> None:
    """Verify fit raises KeyError when catalog references an unknown skill name."""
    ghost = three_catalog.model_copy(update={"skills": ("ghost",)})
    with pytest.raises(KeyError, match="ghost"):
        _tuned().fit(ghost, three)


def test_unregistered_model_uses_default_listing_budget(three, three_catalog) -> None:
    """Verify unregistered models seamlessly use the default ModelProfile listing budget."""
    default_budget = math.floor(1_048_576 * 4.0 * 0.01)

    assert listing_budget_chars(0.01, model="nonexistent-model") == default_budget
    assert budget_fraction_for(1_000, model="nonexistent-model") is not None
    fit = _tuned(model="nonexistent-model").fit(three_catalog, three)
    assert fit.whole


def test_the_unset_default_tracks_a_correction_to_sonnets_profile(
    tmp_path,
    monkeypatch,
) -> None:
    """Verify listing_budget_chars dynamically tracks updated profile values."""
    override = tmp_path / "reach.toml"
    override.write_text(
        "[models.claude-sonnet-5]\nchars_per_token = 5.0\ncontext_window = 1_000_000\n",
    )
    monkeypatch.chdir(tmp_path)
    assert listing_budget_chars(None, "claude-sonnet-5") == listing_budget_chars(
        0.01,
        "claude-sonnet-5",
    )
    assert listing_budget_chars(None, "claude-sonnet-5") == 50_000


def test_claude_listing_exports_directly() -> None:
    """Verify claude_listing exports all listing budget primitives directly."""
    import reach.runtime.claude_code as code
    import reach.runtime.claude_listing as listing

    assert listing.fit_skill_listing is code.fit_skill_listing
    assert listing.listing_budget_chars is code.listing_budget_chars
    assert listing.budget_fraction_for is code.budget_fraction_for
    assert listing.display_width is code.display_width
    assert listing.ListingEntry is code.ListingEntry
    assert listing.SkillListing is code.SkillListing
    assert listing.BUDGET_FRACTION_PLACES == 3


def test_budget_fraction_for_uses_configured_places_and_fraction(
    tmp_path,
    monkeypatch,
) -> None:
    """Verify budget_fraction_for and listing_budget_chars dynamically track profile config."""
    override = tmp_path / "reach.toml"
    override.write_text(
        "[models.claude-sonnet-5]\n"
        "chars_per_token = 3.0\n"
        "context_window = 1_000_000\n"
        "completion_window = 200_000\n"
        "budget_fraction_places = 4\n"
        "listing_budget_fraction = 0.02\n",
    )

    monkeypatch.chdir(tmp_path)
    # Default fraction should now be 0.02 (60,000) instead of 0.01 (30,000)
    assert listing_budget_chars(None, "claude-sonnet-5") == 60_000
    # Fraction should be resolved to 4 decimal places
    fraction = budget_fraction_for(51_910, "claude-sonnet-5")
    assert fraction is not None
    assert round(fraction, 4) == fraction
