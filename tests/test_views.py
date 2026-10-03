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

"""Verify terminal UI components, themed formatting, and progress reporting."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

import cyclopts
import pytest

if TYPE_CHECKING:
    from collections.abc import Callable

    from rich.console import Console

import reach
import reach.views
from reach.artifact import Artifact, QueryRecord, SkillScore, Spread
from reach.difficulty import LexicalRank
from reach.discovery import Discovery, Shadowed
from reach.leak import Leak
from reach.models import Query, QueryKind
from reach.queries import Origin, QuerySet, QuerySetProvenance
from reach.run import Plan
from reach.runtime import SkillRoot
from reach.views import (
    ERROR,
    HIT,
    MISROUTE,
    NON_SELECTION,
    REACH_THEME,
    badge,
    build_console,
    error_panel,
    help_console,
    help_formatter,
    is_live,
    print_discovery,
    print_generation,
    print_plan,
    print_query_records,
    print_query_set,
    print_quick_scope,
    print_scorecard,
    probe_progress,
)

STYLE_USE = re.compile(r'[\["]((?:reach|cyclopts)\.[a-z][a-z.\-_]*)[\]"]')

BORROWED_STYLES = {
    "progress.download": "reach.count",
    "progress.elapsed": "reach.digest",
    "progress.remaining": "reach.digest",
}

BAR_STYLES = ["reach.bar", "reach.bar.done", "reach.bar.track", "reach.spinner"]


@pytest.fixture
def plan() -> Plan:
    """Return sample Plan for console testing."""
    return Plan(
        catalog_id="neighborhood:gcs-lifecycle-rules",
        catalog_size=3,
        probes=2,
        corpus_digest="a5d0a15480d1",
        queries_digest="c0ffee123456",
    )


@pytest.fixture
def truth() -> dict[str, str]:
    """Return mapping of query IDs to expected skill names."""
    return {
        "q-lifecycle": "gcs-lifecycle-rules",
        "q-retention": "gcs-retention-policy",
    }


def test_the_seam_is_the_only_module_that_imports_rich() -> None:
    """Verify only reach.views imports rich within the codebase."""
    package = Path(reach.__file__).parent
    importers = sorted(
        str(path.relative_to(package))
        for path in package.rglob("*.py")
        if not path.is_relative_to(package / "views")
        and re.search(
            r"^\s*(?:import|from) rich",
            path.read_text(encoding="utf-8"),
            re.MULTILINE,
        )
    )
    assert importers == []


def test_a_console_renders_into_whatever_it_was_given(make_console, rendered) -> None:
    """Verify custom Console instance renders output to provided buffer."""
    console, buffer = make_console()
    console.print("probing")
    assert "probing" in rendered(buffer)


def test_every_style_the_view_names_is_declared_in_the_theme() -> None:
    """Verify all style names used in views are declared in REACH_THEME."""
    package = Path(reach.__file__).parent
    sources = [p.read_text(encoding="utf-8") for p in (package / "views").glob("*.py")]
    used = {match for source in sources for match in STYLE_USE.findall(source)}
    assert used
    assert used <= set(REACH_THEME.styles)


@pytest.mark.parametrize(
    "name",
    ["reach.catalog", "reach.count", "reach.digest", "reach.hit", "reach.misroute"],
)
def test_a_console_resolves_the_theme(name: str, make_console) -> None:
    """Verify Console resolves Reach theme styles."""
    console, _ = make_console()
    assert console.get_style(name) is not None


@pytest.mark.parametrize("outcome", [HIT, MISROUTE, NON_SELECTION, ERROR])
def test_each_way_a_probe_can_land_has_a_style_of_its_own(outcome: str, make_console) -> None:
    """Verify Reach theme defines styles for all probe outcomes."""
    console, _ = make_console()
    assert console.get_style(f"reach.{outcome}") is not None


@pytest.mark.parametrize(("borrowed", "declared"), sorted(BORROWED_STYLES.items()))
def test_a_column_that_styles_itself_is_painted_from_the_theme_too(
    borrowed: str,
    declared: str,
    make_console,
) -> None:
    """Verify default Rich progress styles map to Reach theme styles."""
    console, _ = make_console()
    assert console.get_style(borrowed) == console.get_style(declared)


@pytest.mark.parametrize("name", BAR_STYLES)
def test_the_bar_never_wears_a_color_that_already_means_an_outcome(
    name: str,
    make_console,
) -> None:
    """Verify progress bar styles do not collide with outcome color styles."""
    console, _ = make_console()
    outcomes = {
        console.get_style(f"reach.{outcome}") for outcome in (HIT, MISROUTE, NON_SELECTION, ERROR)
    }
    assert console.get_style(name) not in outcomes


def test_a_style_outside_the_theme_is_refused(make_console) -> None:
    """Verify requesting an undeclared style raises an exception."""
    console, _ = make_console()
    with pytest.raises(Exception, match=r"reach\.invented"):
        console.get_style("reach.invented")


@pytest.mark.parametrize(
    ("terminal", "quiet", "expected"),
    [
        (True, False, True),
        (True, True, False),
        (False, False, False),
        (False, True, False),
    ],
)
def test_the_view_animates_only_for_someone_watching(
    terminal: bool,
    quiet: bool,
    expected: bool,
    make_console,
) -> None:
    """Verify is_live returns True only for interactive non-quiet terminals."""
    console, _ = make_console(terminal=terminal, quiet=quiet)
    assert is_live(console) is expected


def test_a_live_view_counts_probes_completed_of_total(
    make_console,
    rendered,
    make_result,
    truth,
) -> None:
    """Verify live progress displays completed and total probe counts."""
    console, buffer = make_console()
    with probe_progress(
        console,
        catalog_id="neighborhood:gcs-lifecycle-rules",
        total=2,
        truth=truth,
    ) as record:
        record(1, 2, make_result("q-lifecycle", "gcs-lifecycle-rules"))
    assert "1/2" in rendered(buffer)


def test_a_live_view_says_how_much_longer_it_expects_to_take(
    make_console,
    rendered,
    make_result,
    truth,
) -> None:
    """Verify live progress displays estimated remaining time."""
    console, buffer = make_console()
    with probe_progress(
        console,
        catalog_id="neighborhood:gcs-lifecycle-rules",
        total=2,
        truth=truth,
    ) as record:
        record(1, 2, make_result("q-lifecycle", "gcs-lifecycle-rules"))
    assert "eta" in rendered(buffer)


def test_the_clock_turns_from_an_estimate_into_what_the_run_cost(
    make_console,
    rendered,
    make_result,
    truth,
) -> None:
    """Verify live progress converts ETA into total elapsed time upon completion."""
    console, buffer = make_console()
    with probe_progress(
        console,
        catalog_id="neighborhood:gcs-lifecycle-rules",
        total=2,
        truth=truth,
    ) as record:
        record(1, 2, make_result("q-lifecycle", "gcs-lifecycle-rules"))
        record(2, 2, make_result("q-retention", "gcs-retention-policy"))
    assert rendered(buffer).rstrip().endswith("took 00:00")


def test_a_live_view_names_the_resident_catalog(make_console, rendered, truth) -> None:
    """Verify live progress displays resident catalog ID."""
    console, buffer = make_console()
    with probe_progress(
        console,
        catalog_id="neighborhood:gcs-lifecycle-rules",
        total=2,
        truth=truth,
    ):
        pass
    shown = rendered(buffer)
    assert "neighborhood:gcs-lifecycle-rules" in shown
    assert "skills" not in shown


@pytest.mark.parametrize(
    ("invoked", "error", "expected"),
    [
        ("gcs-lifecycle-rules", None, "1 hit"),
        ("gcs-retention-policy", None, "1 misroute"),
        (None, None, "1 non-selection"),
        (None, "catalog was not resident", "1 error"),
    ],
)
def test_progress_counts_hits_misses_none_and_errors_separately(
    invoked: str | None,
    error: str | None,
    expected: str,
    make_console,
    rendered,
    make_result,
    truth,
) -> None:
    """Verify live progress displays distinct tallies for hits, misroutes, and errors."""
    console, buffer = make_console()
    with probe_progress(
        console,
        catalog_id="neighborhood:gcs-lifecycle-rules",
        total=1,
        truth=truth,
    ) as record:
        record(1, 1, make_result("q-lifecycle", invoked, error=error))
    assert expected in rendered(buffer)


def test_a_live_view_stays_quiet_about_errors_it_has_not_seen(
    make_console,
    rendered,
    make_result,
    truth,
) -> None:
    """Verify error tally is omitted when no errors occur."""
    console, buffer = make_console()
    with probe_progress(
        console,
        catalog_id="neighborhood:gcs-lifecycle-rules",
        total=1,
        truth=truth,
    ) as record:
        record(1, 1, make_result("q-lifecycle", "gcs-lifecycle-rules"))
    assert "error" not in rendered(buffer)


def test_the_bar_holds_its_width_when_the_first_error_lands(
    make_console,
    rendered,
    make_result,
    truth,
) -> None:
    """Verify progress bar layout maintains constant width upon error encounter."""

    def bar(error: str | None) -> int:
        console, buffer = make_console(width=80)
        with probe_progress(
            console,
            catalog_id="neighborhood:gcs-lifecycle-rules",
            total=2,
            truth=truth,
        ) as record:
            record(1, 2, make_result("q-lifecycle", None, error=error))
        return len(re.findall(r"[━╸╺]", rendered(buffer)))

    assert bar("catalog was not resident") == bar(None)


def test_a_correct_abstention_counts_as_a_hit(make_console, rendered, make_result) -> None:
    """Verify non-selection on out-of-scope query is counted as a hit."""
    console, buffer = make_console()
    with probe_progress(
        console,
        catalog_id="neighborhood:gcs-lifecycle-rules",
        total=1,
        truth={"q-oos": "(no skill)"},
    ) as record:
        record(1, 1, make_result("q-oos", None))
    assert "1 hit" in rendered(buffer)


def test_a_redirected_run_gets_one_line_per_probe(make_console, make_result, truth) -> None:
    """Verify non-terminal output prints one line per probe outcome."""
    console, buffer = make_console(terminal=False)
    with probe_progress(
        console,
        catalog_id="neighborhood:gcs-lifecycle-rules",
        total=2,
        truth=truth,
    ) as record:
        record(1, 2, make_result("q-lifecycle", "gcs-lifecycle-rules"))
        record(2, 2, make_result("q-retention", None))
    assert buffer.getvalue() == (
        "[1/2] q-lifecycle -> gcs-lifecycle-rules\n[2/2] q-retention -> (no selection)\n"
    )


def test_a_redirected_run_reports_the_error_a_probe_returned(
    make_console,
    make_result,
    truth,
) -> None:
    """Verify non-terminal output includes error message on probe error."""
    console, buffer = make_console(terminal=False)
    with probe_progress(
        console,
        catalog_id="neighborhood:gcs-lifecycle-rules",
        total=1,
        truth=truth,
    ) as record:
        record(1, 1, make_result("q-lifecycle", None, error="catalog was not resident"))
    assert buffer.getvalue() == "[1/1] q-lifecycle -> catalog was not resident\n"


@pytest.mark.parametrize("terminal", [True, False])
def test_quiet_writes_nothing_at_all(
    terminal: bool,
    make_console,
    make_result,
    truth,
    plan,
) -> None:
    """Verify quiet mode mutes all console output."""
    console, buffer = make_console(terminal=terminal, quiet=True)
    print_plan(
        console,
        plan,
        agent="fake",
        model="sonnet",
        fingerprint="b23b1052aa7a",
    )
    with probe_progress(
        console,
        catalog_id="neighborhood:gcs-lifecycle-rules",
        total=1,
        truth=truth,
    ) as record:
        record(1, 1, make_result("q-lifecycle", "gcs-lifecycle-rules"))
    assert buffer.getvalue() == ""


@pytest.mark.parametrize(
    ("hex_digest", "spoken"),
    [
        ("7f000001", "lusab-babad"),
        ("3f54dcc1", "gutih-tugad"),
        ("801e342d", "mabiv-gibot"),
        ("d43afd44", "tibup-zujah"),
    ],
)
def test_a_badge_is_the_published_encoding_and_not_one_of_our_own(
    hex_digest: str,
    spoken: str,
) -> None:
    """Verify badge proquint encoding matches reference examples."""
    assert badge(hex_digest) == spoken


def test_every_character_of_a_digest_reaches_its_badge() -> None:
    """Verify single character changes in hex digest alter the generated badge."""
    base = "b23b1052aa7a"
    seen = {badge(base)}
    for at in range(len(base)):
        moved = base[:at] + ("0" if base[at] != "0" else "1") + base[at + 1 :]
        assert badge(moved) not in seen
        seen.add(badge(moved))


@pytest.mark.parametrize("given", ["", "(unrecorded)", "abc", "b23b1052aa7"])
def test_something_that_is_not_a_digest_is_printed_as_it_arrived(given: str) -> None:
    """Verify non-hex strings are passed through badge unchanged."""
    assert badge(given) == given


def test_a_redirected_plan_line_is_plain_and_unwrapped(make_console, plan) -> None:
    """Verify redirected plan line prints unwrapped text."""
    console, buffer = make_console(terminal=False, width=40)
    print_plan(
        console,
        plan,
        agent="fake",
        model="sonnet",
        fingerprint="b23b1052aa7a",
    )
    assert buffer.getvalue() == (
        "neighborhood:gcs-lifecycle-rules: 3 skills, 2 probes on fake/sonnet "
        "[config ramur-dadif-ponup corpus pilib-pajih-magid]\n"
    )


def test_a_watched_plan_line_says_the_same_thing_with_emphasis(
    make_console,
    rendered,
    plan,
) -> None:
    """Verify terminal plan line includes ANSI styling escape sequences."""
    console, buffer = make_console(terminal=True, width=200)
    print_plan(
        console,
        plan,
        agent="fake",
        model="sonnet",
        fingerprint="b23b1052aa7a",
    )
    assert rendered(buffer).rstrip("\n") == (
        "neighborhood:gcs-lifecycle-rules: 3 skills, 2 probes on fake/sonnet "
        "[config ramur-dadif-ponup corpus pilib-pajih-magid]"
    )
    assert "\x1b[" in buffer.getvalue()


def test_a_verbose_plan_line_says_the_badge_and_the_hex_behind_it(make_console, plan) -> None:
    """Verify verbose plan line prints both badge words and raw hex digests."""
    console, buffer = make_console(terminal=False, width=200)
    print_plan(
        console,
        plan,
        agent="fake",
        model="sonnet",
        fingerprint="b23b1052aa7a",
        verbose=True,
    )
    stamp = "[config ramur-dadif-ponup b23b1052aa7a corpus pilib-pajih-magid a5d0a15480d1]"
    assert buffer.getvalue().rstrip("\n").endswith(stamp)


def opening_escape(console: Console, name: str) -> str:
    """Extract opening ANSI escape sequence for specified style name."""
    return console.get_style(name).render("|").split("|")[0]


@pytest.mark.parametrize(
    "name",
    [
        "cyclopts.name",
        "cyclopts.required_marker",
        "cyclopts.choices",
        "cyclopts.env_var",
        "cyclopts.default",
        "cyclopts.required",
        "cyclopts.border",
        "cyclopts.usage",
    ],
)
def test_a_console_resolves_cyclopts_help_theme_keys(name: str, make_console) -> None:
    """Verify Console resolves all Cyclopts 5 help theme keys."""
    console, _ = make_console()
    assert console.get_style(name) is not None


def test_the_help_formatter_carries_the_theme(make_console) -> None:
    """Verify CLI help formatter applies Reach theme styles via cyclopts.* keys."""
    console, buffer = make_console()
    app = cyclopts.App(name="reach", help_formatter=help_formatter())

    @app.command
    def probe() -> None:
        """Issue one probe."""

    app.help_print([], console=console)
    shown = buffer.getvalue()
    assert f"{opening_escape(console, 'cyclopts.name')}probe" in shown
    assert opening_escape(console, "cyclopts.border") in shown


def test_the_help_formatter_names_no_style_the_theme_does_not_declare(
    make_console,
) -> None:
    """Verify help formatter executes cleanly without undeclared styles."""
    console, _ = make_console()
    app = cyclopts.App(name="reach", help_formatter=help_formatter())

    @app.command
    def probe(*, queries: str, quiet: bool = False) -> None:
        """Issue one probe."""

    app.help_print([], console=console)
    app.help_print(["probe"], console=console)


def test_the_error_panel_is_drawn_from_the_theme(make_console) -> None:
    """Verify error panel renders using Reach theme border styles."""
    console, buffer = make_console()
    console.print(error_panel(["Invalid value 'roomy' for --catalog-size."]))
    shown = buffer.getvalue()
    assert opening_escape(console, "reach.error.border") in shown
    assert "--catalog-size" in shown


def test_the_error_panel_names_no_style_the_theme_does_not_declare(
    make_console,
) -> None:
    """Verify error panel contains only valid declared styles."""
    console, _ = make_console()
    console.print(error_panel(["something went wrong", "and here is a second line"]))


def test_the_console_seam_can_be_asked_for_the_default_sink() -> None:
    """Verify build_console returns a configured Console instance."""
    assert build_console().file is not None


def test_help_goes_where_the_view_does_not() -> None:
    """Verify help_console targets stdout while default build_console targets stderr."""
    assert help_console().stderr is False
    assert build_console().stderr is True


def test_help_is_never_muted() -> None:
    """Verify help_console is never created in quiet mode."""
    assert help_console().quiet is False


@pytest.fixture
def card(make_console, rendered: Callable[..., str], artifact) -> Callable[..., str]:
    """Provide helper rendering Artifact scorecards into test string buffers."""

    def _card(
        shown: Artifact | None = None,
        *,
        verbose: bool = False,
        **kwargs: object,
    ) -> str:
        console, buffer = make_console(**kwargs)
        print_scorecard(
            console,
            shown if shown is not None else artifact,
            verbose=verbose,
        )
        return rendered(buffer)

    return _card


def test_scorecard_skill_table_layout_and_metrics(card, artifact) -> None:
    """Verify scorecard skill table ordering, headers, intervals, and unprobed rows."""
    shown = card()
    assert "reach" in shown
    assert shown.index("gcs-retention-policy") < shown.index("consistency")
    assert shown.index("gcs-retention-policy") < shown.index("gcs-lifecycle-rules")
    assert re.search(r"1/3\s+33%\s+6-79%", shown)

    measured = next(s for s in artifact.skills if s.skill == "gcs-retention-policy")
    assert measured.recall_interval is not None
    low = f"{measured.recall_interval.low * 100:.0f}"
    high = f"{measured.recall_interval.high * 100:.0f}"
    assert f"{low}-{high}%" in shown

    header = next(line for line in shown.splitlines() if "precision" in line)
    columns = header.split()
    assert "recall" in columns
    assert "reach" not in columns
    assert "F1" not in columns
    assert "95% CI" in shown

    # Unprobed skills render without CI or 0% reach, but appear if they absorbed traffic
    row = next(line for line in shown.splitlines() if "gke-basics" in line)
    assert not re.search(r"\d+-\d+%", row)
    assert re.search(r"gke-basics\s+1\s+0%", shown)
    assert not re.search(r"gke-basics\s+0%", shown)


def test_an_attractor_skill_sorts_above_a_well_served_one() -> None:
    """Verify unrequested attractor skills with absorption sort above well-served skills."""
    from reach.views.scorecard import _worst_first

    attractor = SkillScore(
        skill="attractor",
        probes=0,
        reached=0,
        recall=None,
        absorbed=10,
        precision=0.0,
    )
    well_served = SkillScore(
        skill="well-served",
        probes=10,
        reached=10,
        recall=1.0,
        absorbed=0,
        precision=1.0,
    )
    sorted_skills = _worst_first([well_served, attractor])
    assert sorted_skills[0].skill == "attractor"
    assert sorted_skills[1].skill == "well-served"


def _rename_skill(artifact: Artifact, old_name: str, new_name: str) -> Artifact:
    """Return artifact copy with named skill renamed to new_name."""
    skills = tuple(
        skill.model_copy(update={"skill": new_name}) if skill.skill == old_name else skill
        for skill in artifact.skills
    )
    return artifact.model_copy(update={"skills": skills})


def test_a_name_too_long_for_the_row_gives_way_before_a_figure_does(card, artifact) -> None:
    """Verify long skill names truncate cleanly without dropping metric columns."""
    long_name = "google-cloud-solution-agentic-analytics-spark-knowledge-catalog"
    renamed = _rename_skill(artifact, "gcs-retention-policy", long_name)
    measured = next(s for s in renamed.skills if s.skill == long_name)
    row = next(line for line in card(renamed).splitlines() if line.startswith(" google-cloud"))
    assert "…" in row
    assert measured.precision is not None
    assert row.rstrip().endswith(f"{measured.precision:.0%}")
    assert "6-79%" in row
    assert len(row) <= 100


def test_scorecard_headline_and_summary_metrics(card) -> None:
    """Verify scorecard headline stability, pp standard error, intervals, and macro-F1."""
    shown = card()
    assert "consistency" in shown
    assert re.search(r"top-1 66\.7% ± 16\.9pp", shown)
    assert "± 16.9pp" in shown
    assert "± 16.9%" not in shown
    assert re.search(r"consistency 50% \[9-91%\]", shown)
    assert re.search(r"abstention 17% \[3-56%\]", shown)

    figures = next(line for line in shown.splitlines() if "top-1" in line)
    assert "± 16.9pp" in figures
    assert figures.count("[") == 2

    assert "macro-F1" in shown
    assert shown.index("consistency") < shown.index("macro-F1")


def test_scorecard_collision_table_and_provenance_footer(card, artifact) -> None:
    """Verify collision table entries, excluded non-collisions, quoted queries, and footer."""
    shown = card()
    assert re.search(r"gcs-retention-policy\s+gke-basics\s+1", shown)
    assert shown.count("gcs-lifecycle-rules") == 1
    assert "Keep audit logs for seven years" in shown
    assert badge(artifact.digests.corpus_digest) in shown
    assert badge(artifact.provenance.arm) in shown


def test_the_collision_table_displays_reasoning_traces(artifact: Artifact) -> None:
    """Verify collision table renders thought trace subrows when show_reasoning is enabled."""
    from reach.artifact import SampleQuery
    from reach.views import build_console
    from reach.views.scorecard import _collision_table

    pair = artifact.confusion[1]
    updated_pair = pair.model_copy(
        update={
            "queries": (
                SampleQuery(
                    query_id="q-1",
                    text="Keep audit logs for seven years.",
                    probes=1,
                    reasoning=("Thinking about bucket retention rules.",),
                ),
            ),
        },
    )
    test_artifact = artifact.model_copy(update={"confusion": (artifact.confusion[0], updated_pair)})

    console = build_console(width=200)
    table = _collision_table(test_artifact.confusion, show_reasoning=True)
    assert table is not None
    with console.capture() as capture:
        console.print(table)
    rendered = capture.get()
    assert "thought: Thinking about bucket retention rules." in rendered


def test_the_collision_table_suppresses_reasoning_when_disabled(artifact: Artifact) -> None:
    """Verify collision table suppresses thought trace subrows when show_reasoning is False."""
    from reach.artifact import SampleQuery
    from reach.views import build_console
    from reach.views.scorecard import _collision_table

    pair = artifact.confusion[1]
    updated_pair = pair.model_copy(
        update={
            "queries": (
                SampleQuery(
                    query_id="q-1",
                    text="Keep audit logs for seven years.",
                    probes=1,
                    reasoning=("Thinking about bucket retention rules.",),
                ),
            ),
        },
    )
    test_artifact = artifact.model_copy(update={"confusion": (artifact.confusion[0], updated_pair)})

    console = build_console()
    table = _collision_table(test_artifact.confusion, show_reasoning=False)
    assert table is not None
    with console.capture() as capture:
        console.print(table)
    rendered = capture.get()
    assert "thought:" not in rendered


def test_the_scorecard_keeps_the_hex_off_the_stamp_until_it_is_asked_for(
    card,
    artifact,
) -> None:
    """Verify raw hex digests are omitted from footer stamp unless verbose=True."""
    quietly = card(width=200)
    assert artifact.digests.corpus_digest not in quietly

    loudly = card(width=200, verbose=True)
    assert f"corpus {badge(artifact.digests.corpus_digest)} " in loudly
    assert artifact.digests.corpus_digest in loudly
    assert artifact.digests.queries_digest in loudly
    assert artifact.provenance.arm in loudly


def test_the_scorecard_is_muted_along_with_everything_else(card) -> None:
    """Verify print_scorecard emits no output in quiet mode."""
    assert card(quiet=True) == ""


def test_the_scorecard_renders_without_a_terminal_to_render_into(card) -> None:
    """Verify scorecard renders cleanly when redirected to non-terminal output."""
    shown = card(terminal=False)
    assert "gcs-retention-policy" in shown
    assert "consistency" in shown


def test_a_run_that_probed_nothing_still_renders(card, artifact) -> None:
    """Verify scorecard renders without error or floating table header when probe count is zero."""
    empty = artifact.model_copy(
        update={
            "skills": (),
            "confusion": (),
            "queries": (),
            "probes": 0,
            "spread": Spread(),
        },
    )
    assert "consistency" in card(empty)

    unprobed_skills = tuple(
        SkillScore(skill=s.skill, probes=0, reached=0, absorbed=0) for s in artifact.skills
    )
    shown_unprobed = card(empty.model_copy(update={"skills": unprobed_skills}))
    assert "3 resident skills had no query and took no traffic" in shown_unprobed
    assert "95% CI" not in shown_unprobed


def test_the_unshown_residents_are_counted_rather_than_listed(card, artifact) -> None:
    """Verify resident skills without queries are summarized in a footer count."""
    padded = artifact.model_copy(
        update={
            "skills": (
                *artifact.skills,
                SkillScore(skill="unnamed-skill", probes=0, reached=0),
            ),
        },
    )
    shown = card(padded)
    assert "unnamed-skill" not in shown
    assert "1 resident skills had no query" in shown


@pytest.fixture
def listing(make_console, rendered: Callable[..., str], artifact) -> Callable[..., str]:
    """Provide helper rendering query record listings into test string buffers."""

    def _listing(shown: Artifact | None = None, **kwargs: object) -> str:
        console, buffer = make_console(**kwargs)
        print_query_records(console, shown if shown is not None else artifact)
        return rendered(buffer)

    return _listing


def test_every_query_the_run_scored_gets_a_row_that_quotes_it(listing) -> None:
    """Verify query record listing includes query ID and query text."""
    shown = listing(width=200)
    assert "q-lifecycle" in shown
    assert "q-retention" in shown
    assert "Keep audit logs for seven years" in shown


def test_a_row_says_what_it_reached_when_it_did_not_reach_what_it_expected(
    listing,
) -> None:
    """Verify misrouted query rows display expected and invoked skill names."""
    shown = listing(width=200)
    assert "gcs-retention-policy" in shown
    row = next(line for line in shown.splitlines() if "q-retention" in line)
    assert "gke-basics" in row


def test_the_count_on_a_row_carries_the_interval_the_artifact_recorded(
    listing,
    artifact,
) -> None:
    """Verify query row displays hit count and recorded confidence interval."""
    record = next(q for q in artifact.queries if q.query_id == "q-retention")
    assert record.interval is not None
    low = f"{record.interval.low * 100:.0f}"
    high = f"{record.interval.high * 100:.0f}"
    row = next(line for line in listing(width=200).splitlines() if "q-retention" in line)
    assert "1/3" in row
    assert f"{low}-{high}%" in row


def test_a_rank_is_shown_over_the_field_it_was_taken_in(listing, artifact) -> None:
    """Verify lexical rank is rendered relative to catalog size."""
    assert "3/111" in listing(artifact.model_copy(update={"catalog_size": 111}))


def _painted(console: Console, name: str) -> str:
    """Return opening escape sequence for theme style."""
    return console.get_style(name).render("|").partition("|")[0]


@pytest.mark.parametrize(
    ("probes", "hits", "painted"),
    [(0, 0, False), (3, 0, True)],
    ids=["never probed", "probed and missed"],
)
def test_a_query_nobody_probed_is_not_painted_as_one_that_failed(
    make_console,
    artifact,
    probes: int,
    hits: int,
    painted: bool,
) -> None:
    """Verify unprobed queries are not highlighted with error styling."""
    console, buffer = make_console(width=200)
    only = QueryRecord(
        query_id="q-alone",
        text="Does anything reach this?",
        expected="gke-basics",
        probes=probes,
        hits=hits,
    )
    print_query_records(console, artifact.model_copy(update={"queries": (only,)}))

    assert (_painted(console, "reach.error") in buffer.getvalue()) is painted


@pytest.fixture
def drafted(queries) -> QuerySet:
    """Return synthetic QuerySet for view formatting tests."""
    return QuerySet(
        catalog_id="all",
        notes="drafted",
        queries=tuple(queries),
        provenance=QuerySetProvenance(origin=Origin.AUTHORED),
    )


def test_the_discovered_roots_are_shown_back_with_the_scope_each_carries(
    make_console,
    rendered,
    tmp_path: Path,
) -> None:
    """Verify print_discovery displays discovered skill roots and scopes."""
    console, buffer = make_console()
    print_discovery(
        console,
        Discovery(roots=(SkillRoot(path=tmp_path / "personal", scope="user"),)),
    )
    shown = rendered(buffer)

    assert "user" in shown
    assert "personal" in shown


def test_every_warning_discovery_raised_is_printed_rather_than_counted(
    make_console,
    rendered,
    tmp_path: Path,
) -> None:
    """Verify discovery warnings detail shadowed skills and empty root paths."""
    console, buffer = make_console()
    print_discovery(
        console,
        Discovery(
            roots=(SkillRoot(path=tmp_path / "p", scope="user"),),
            shadowed=(
                Shadowed(
                    name="gke-basics",
                    kept=tmp_path / "p" / "gke-basics",
                    hidden=(tmp_path / "q" / "gke-basics",),
                ),
            ),
            empty_roots=(tmp_path / "p",),
        ),
    )
    shown = rendered(buffer)

    assert "shadowing" in shown
    assert "empty" in shown


def test_a_drafted_set_lists_its_ground_truth_and_lexical_prospects(
    make_console,
    rendered,
    drafted: QuerySet,
) -> None:
    """Verify print_query_set outputs lexical rank and leak inspection flags."""
    console, buffer = make_console(width=200)
    print_query_set(
        console,
        drafted,
        path=Path("queries.json"),
        catalog_id="all",
        residents=3,
        ranks={"q-lifecycle": LexicalRank(position=1, field_size=3)},
        flags={"q-lifecycle": Leak(names_target=True, distinctive_tokens=())},
    )
    shown = rendered(buffer)

    assert "1/3" in shown
    assert "names target" in shown
    assert "queries.json" in shown


def test_a_query_that_gave_nothing_away_says_so_rather_than_saying_nothing(
    make_console,
    rendered,
    drafted: QuerySet,
) -> None:
    """Verify clean leak check displays clean indicator in query listing."""
    console, buffer = make_console(width=200)
    print_query_set(
        console,
        drafted,
        path=Path("q.json"),
        catalog_id="all",
        residents=3,
        ranks={},
        flags={"q-lifecycle": Leak(names_target=False, distinctive_tokens=())},
    )
    shown = rendered(buffer)

    assert "clean" in shown


LONG_NAMED = QuerySet(
    catalog_id="all",
    notes="drafted",
    queries=(
        Query(
            id="google-cloud-waf-security-2",
            text=(
                "Someone got into one of our service accounts overnight. What "
                "should we have in place to catch that and work through it?"
            ),
            kind=QueryKind.NEIGHBOR_NEGATIVE,
            expected_skill="google-cloud-waf-security",
        ),
    ),
    provenance=QuerySetProvenance(origin=Origin.AUTHORED),
)


def test_a_drafted_set_keeps_its_labels_at_the_width_a_terminal_actually_has(
    make_console,
    rendered,
) -> None:
    """Verify print_query_set preserves table columns at 100-character terminal width."""
    console, buffer = make_console(width=100)
    print_query_set(
        console,
        LONG_NAMED,
        path=Path("queries.json"),
        catalog_id="all",
        residents=4,
        ranks={
            "google-cloud-waf-security-2": LexicalRank(position=3, field_size=4),
        },
        flags={
            "google-cloud-waf-security-2": Leak(
                names_target=False,
                distinctive_tokens=(),
            ),
        },
    )
    shown = rendered(buffer)

    assert "google-cloud-waf-sec" in shown
    assert "3/4" in shown
    assert "clean" in shown
    assert "Someone got into" in shown
    assert "\n" not in shown[shown.index("google-cloud-waf") :].split("wrote")[0].strip()


def test_a_collision_keeps_the_two_skill_names_when_the_row_will_not_fit(card) -> None:
    """Verify collision table retains both skill names on narrow console widths."""
    shown = card(width=60)

    assert "gcs-retention-poli" in shown
    assert "gke-basics" in shown
    assert "Keep audit logs" in shown


def test_what_is_about_to_be_drafted_is_stated_before_it_is_paid_for(
    make_console,
    rendered,
) -> None:
    """Verify print_generation outputs resident count, total queries, and runtime target."""
    console, buffer = make_console(width=200)
    print_generation(
        console,
        catalog_id="all",
        residents=111,
        targets=6,
        count=2,
        agent="claude-code",
        model="opus",
    )
    shown = rendered(buffer)

    assert "111 skills resident" in shown
    assert "12" in shown
    assert "claude-code/opus" in shown
    assert "rivals" not in shown


def test_a_capped_prompt_says_how_much_of_the_catalog_the_generator_saw(
    make_console,
    rendered,
) -> None:
    """Verify capped prompt output reports rival count in view."""
    console, buffer = make_console(width=200)
    print_generation(
        console,
        catalog_id="all",
        residents=111,
        targets=6,
        count=2,
        agent="claude-code",
        model="opus",
        rivals=10,
    )
    shown = rendered(buffer)

    assert "111 skills resident" in shown
    assert "10 of them shown as rivals" in shown


def test_adversarial_drafting_indicates_adversarial_count(
    make_console,
    rendered,
) -> None:
    """Verify print_generation notes the count of adversarial queries when enabled."""
    console, buffer = make_console(width=200)
    print_generation(
        console,
        catalog_id="all",
        residents=20,
        targets=4,
        count=2,
        agent="claude-code",
        model="opus",
        adversarial=True,
        adversarial_count=2,
    )
    shown = rendered(buffer)

    assert "8 (+8 adversarial) queries" in shown


def test_a_set_that_is_not_kept_is_still_listed_but_names_no_file(
    make_console,
    rendered,
    drafted: QuerySet,
) -> None:
    """Verify ephemeral query sets omit file paths when printed."""
    console, buffer = make_console(width=200)
    print_query_set(
        console,
        drafted,
        path=None,
        catalog_id="neighborhood:gcs-lifecycle-rules",
        residents=3,
        ranks={},
        flags={},
        then="probing it now",
    )
    shown = rendered(buffer)

    assert "q-lifecycle" in shown
    assert "probing it now" in shown
    assert "wrote" not in shown


def test_a_quick_run_states_the_three_defaults_that_produced_its_answer(
    make_console,
    rendered,
) -> None:
    """Verify print_quick_scope outputs residency, attempt count, and rival scope."""
    console, buffer = make_console(width=200)
    print_quick_scope(
        console,
        catalog_id="neighborhood:gke-inference",
        residents=20,
        corpus=111,
        attempts=3,
        rivals=19,
        authored=0,
    )
    shown = rendered(buffer)

    assert "neighborhood:gke-inference" in shown
    assert "20 of 111 skills resident" in shown
    assert "3 attempts per query" in shown
    assert "drafted here and probed unreviewed, against 19 rivals" in shown
    assert "ground truth drafted here" in shown
    assert "ground truth  drafted here" not in shown


def test_a_typed_question_is_not_reported_as_unreviewed_ground_truth(
    make_console,
    rendered,
) -> None:
    """Verify hand-authored queries omit unreviewed draft warnings."""
    console, buffer = make_console(width=200)
    print_quick_scope(
        console,
        catalog_id="neighborhood:gke-inference",
        residents=20,
        corpus=111,
        attempts=1,
        rivals=None,
        authored=1,
    )
    shown = rendered(buffer)

    assert "1 attempt per query" in shown
    assert "the 1 question you typed" in shown
    assert "unreviewed" not in shown


def test_print_check_renders_stage1_and_stage2(make_console, rendered) -> None:
    """Verify print_check outputs Stage 1 and Stage 2 summary status tables."""
    from reach.check import CheckAssertion, CheckOutcome
    from reach.lint import LintReport
    from reach.views import print_check

    outcome = CheckOutcome(
        lint_report=LintReport(issues=(), skills_checked=3),
        assertions=(
            CheckAssertion(
                name="recall",
                passed=True,
                observed=0.95,
                threshold=0.80,
                comparison=">=",
                message="Recall meets threshold",
            ),
        ),
        skills_checked=3,
        queries_probed=5,
        probes_executed=5,
        budget=50,
        exit_code=0,
    )
    console, buffer = make_console(width=160)
    print_check(console, outcome)
    output = rendered(buffer)

    assert "Quality Gate" in output
    assert "Stage 1" in output
    assert "Stage 2" in output
    assert "recall" in output
    assert "PASSED" in output


def test_render_check_github_summary_markdown() -> None:
    """Verify render_check_github_summary formats a GFM Markdown report."""
    from reach.check import CheckAssertion, CheckOutcome
    from reach.lint import LintReport
    from reach.views import render_check_github_summary

    outcome = CheckOutcome(
        lint_report=LintReport(issues=(), skills_checked=4),
        assertions=(
            CheckAssertion(
                name="recall",
                passed=True,
                observed=0.90,
                threshold=0.80,
                comparison=">=",
                message="Recall meets threshold",
            ),
        ),
        skills_checked=4,
        queries_probed=10,
        probes_executed=10,
        budget=50,
        exit_code=0,
    )
    md = render_check_github_summary(outcome)
    assert "# Reach Quality Gate" in md
    assert "Stage 1: Static Pre-flight" in md
    assert "Stage 2: Empirical Quality Gate" in md
    assert "| Metric | Observed | Target | Status |" in md
    assert "90.0%" in md


def test_format_assertion_strings_percentage_and_float_metrics() -> None:
    """Verify _format_assertion_strings formats percentages and float metrics correctly."""
    from reach.check import CheckAssertion
    from reach.views.quality_gate import _format_assertion_strings

    pct_specs = [
        ("recall", 0.90, 0.80, ">="),
        ("accuracy", 0.85, 0.75, ">="),
        ("misroute_rate", 0.05, 0.10, "<="),
        ("entrypoint", 0.92, 0.80, ">="),
        ("reachability", 0.88, 0.70, ">="),
        ("skill_f1", 0.825, 0.75, ">="),
    ]
    pct_assertions = [
        CheckAssertion(
            name=name,
            passed=True,
            observed=obs_val,
            threshold=thresh_val,
            comparison=comp,
            message="",
        )
        for name, obs_val, thresh_val, comp in pct_specs
    ]
    for a in pct_assertions:
        obs, tgt = _format_assertion_strings(a)
        assert "%" in obs, f"Expected percentage in {a.name} observed string: {obs}"
        assert "%" in tgt, f"Expected percentage in {a.name} target string: {tgt}"

    # Verify exact formatted string values for trajectory metrics
    entrypoint_obs, entrypoint_tgt = _format_assertion_strings(pct_assertions[3])
    assert entrypoint_obs == "92.0%"
    assert entrypoint_tgt == ">= 80.0%"

    reach_obs, reach_tgt = _format_assertion_strings(pct_assertions[4])
    assert reach_obs == "88.0%"
    assert reach_tgt == ">= 70.0%"

    f1_obs, f1_tgt = _format_assertion_strings(pct_assertions[5])
    assert f1_obs == "82.5%"
    assert f1_tgt == ">= 75.0%"

    non_pct = [
        CheckAssertion(
            name="step_efficiency",
            passed=True,
            observed=0.95,
            threshold=0.90,
            comparison=">=",
            message="",
        ),
        CheckAssertion(
            name="redundancy",
            passed=True,
            observed=0.10,
            threshold=0.25,
            comparison="<=",
            message="",
        ),
    ]
    for a in non_pct:
        obs, tgt = _format_assertion_strings(a)
        assert "%" not in obs, f"Unexpected percentage in {a.name} observed string: {obs}"
        assert "%" not in tgt, f"Unexpected percentage in {a.name} target string: {tgt}"


def test_emit_check_github_annotations_for_regressions() -> None:
    """Verify emit_check_github_annotations outputs workflow error commands."""
    from reach.check import CheckAssertion, CheckOutcome, CheckStage
    from reach.lint import LintReport
    from reach.views import emit_check_github_annotations

    outcome = CheckOutcome(
        lint_report=LintReport(issues=(), skills_checked=2),
        stage_failed=CheckStage.EMPIRICAL,
        assertions=(
            CheckAssertion(
                name="recall",
                passed=False,
                observed=0.60,
                threshold=0.80,
                comparison=">=",
                message="observed 60.0% is below required 80.0%",
            ),
        ),
        exit_code=2,
    )
    annotations = emit_check_github_annotations(outcome)
    assert "::error" in annotations
    assert "title=regression-recall" in annotations
    assert "below required 80.0%" in annotations


def test_write_github_step_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify write_github_step_summary appends to file pointed by GITHUB_STEP_SUMMARY."""
    from reach.views import write_github_step_summary

    summary_file = tmp_path / "step_summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary_file))

    success = write_github_step_summary("## Test Summary Header")
    assert success is True
    assert "## Test Summary Header" in summary_file.read_text(encoding="utf-8")


@pytest.mark.parametrize("width", [100, 80])
def test_print_optimization_renders_candidates_and_table(
    width: int,
    make_console,
    rendered,
) -> None:
    """Verify print_optimization renders candidate descriptions even at 80 cols with all columns."""
    from reach.optimize import OptimizationCandidate, OptimizationReport
    from reach.views import print_optimization

    report = OptimizationReport(
        skill_name="test-skill",
        baseline_description="Old baseline description.",
        baseline_recall=0.50,
        baseline_trajectory_recall=0.60,
        baseline_accuracy=0.60,
        baseline_misroute=0.10,
        rival_name="rival-skill",
        ceded_terms=("term1",),
        unclaimed_terms=("distinct1", "distinct2"),
        candidates=(
            OptimizationCandidate(
                description="New improved candidate description.",
                rationale="Reason for change.",
                recall=0.80,
                trajectory_recall=0.90,
                test_recall=0.85,
                accuracy=0.85,
                misroute_rate=0.05,
                delta_recall=0.30,
                lint_clean=True,
            ),
        ),
        applied=False,
        has_probes=True,
    )
    console, buffer = make_console(width=width)
    print_optimization(console, report)
    output = rendered(buffer)

    assert "Reach Closed-Loop Optimizer: test-skill" in output
    assert "Old baseline description." in output
    assert "rival-skill" in output
    assert "distinct1" in output
    assert "+30.0%" in output
    assert "improved" in output
    assert "candidate" in output
    assert "││" not in output
    assert "--auto-apply" in output

    # Handle unevaluated optimization report without probe data
    unprobed_report = report.model_copy(update={"has_probes": False})
    c2, b2 = make_console(width=width)
    print_optimization(c2, unprobed_report)
    out2 = rendered(b2)
    assert "Not evaluated" in out2
    assert "—" in out2


def test_print_optimization_renders_zero_improvement_notice(make_console, rendered) -> None:
    """Verify print_optimization notes when no candidate improves on baseline."""
    from reach.optimize import OptimizationCandidate, OptimizationReport
    from reach.views import print_optimization

    report = OptimizationReport(
        skill_name="perfect-skill",
        baseline_description="Already perfect description.",
        baseline_recall=1.0,
        baseline_accuracy=1.0,
        baseline_misroute=0.0,
        candidates=(
            OptimizationCandidate(
                description="Candidate with no improvement.",
                recall=1.0,
                accuracy=1.0,
                misroute_rate=0.0,
                delta_recall=0.0,
            ),
        ),
        has_probes=True,
    )
    console, buffer = make_console()
    print_optimization(console, report)
    output = rendered(buffer)

    assert "No candidate improved" in output
    assert "--auto-apply" not in output


def test_render_optimization_diff_produces_unified_diff() -> None:
    """Verify render_optimization_diff produces valid unified diff text."""
    from reach.optimize import OptimizationCandidate, OptimizationReport
    from reach.views import render_optimization_diff

    report = OptimizationReport(
        skill_name="my-tool",
        baseline_description="Baseline text.",
        candidates=(
            OptimizationCandidate(
                description="Candidate text.",
                rationale="Rationale.",
            ),
        ),
    )
    diff = render_optimization_diff(report)
    assert "--- a/my-tool/SKILL.md" in diff
    assert "+++ b/my-tool/SKILL.md" in diff
    assert "-  Baseline text." in diff
    assert "+  Candidate text." in diff


def test_render_optimization_diff_supports_specific_candidate_index() -> None:
    """Verify render_optimization_diff generates diff against requested candidate index."""
    from reach.optimize import OptimizationCandidate, OptimizationReport
    from reach.views import render_optimization_diff

    report = OptimizationReport(
        skill_name="multi-tool",
        baseline_description="Baseline text.",
        candidates=(
            OptimizationCandidate(description="Cand 1 text.", rationale="R1."),
            OptimizationCandidate(description="Cand 2 text.", rationale="R2."),
        ),
    )
    diff1 = render_optimization_diff(report, candidate_index=1)
    assert "+  Cand 1 text." in diff1
    assert "candidate #1" in diff1

    diff2 = render_optimization_diff(report, candidate_index=2)
    assert "+  Cand 2 text." in diff2
    assert "candidate #2" in diff2


def test_overlap_and_suggest_views_renderers() -> None:
    """Verify overlap_view and suggest_view DTOs and renderers operate through reach.views."""
    from reach.models import Skill
    from reach.overlap import rank_corpus
    from reach.views import (
        OVERLAP_RENDERERS,
        REWRITE_RENDERERS,
        OverlapView,
        RewriteView,
        SkillOverlapView,
        SuggestView,
        overlap_view,
        render_overlap,
        render_rewrite,
        suggest_view,
    )

    s1 = Skill(name="a", description="alpha tool", path=Path("/a"))
    s2 = Skill(name="b", description="beta tool", path=Path("/b"))
    corpus = [s1, s2]
    overlap = rank_corpus(corpus)

    oview = overlap_view(overlap)
    assert isinstance(oview, OverlapView)
    assert len(oview.skills) == 2
    assert isinstance(oview.skills[0], SkillOverlapView)
    json_out = render_overlap(oview, "json")
    assert "skills" in json_out
    assert "csv" in OVERLAP_RENDERERS
    assert "json" in OVERLAP_RENDERERS
    assert "jsonl" in OVERLAP_RENDERERS

    sview = suggest_view(overlap, corpus, ["a"])
    assert isinstance(sview, SuggestView)
    assert len(sview.skills) == 1
    assert isinstance(sview.skills[0], RewriteView)
    json_rewrite = render_rewrite(sview, "json")
    assert "skills" in json_rewrite
    assert "csv" in REWRITE_RENDERERS
    assert "json" in REWRITE_RENDERERS
    assert "jsonl" in REWRITE_RENDERERS


def test_console_isolates_terminal_rendering_from_ambient_environment(
    make_console,
) -> None:
    """Verify test console preserves ANSI styling and width under standard test fixtures."""
    console, buffer = make_console(terminal=True, width=120)
    console.print("[reach.hit]matched[/reach.hit]")
    assert "\x1b[" in buffer.getvalue()
    assert console.width == 120


def test_probe_progress_prints_resident_catalog_header_once(
    make_console,
    rendered,
    make_result,
    truth,
) -> None:
    """Verify live probe_progress prints the resident catalog header once without duplication."""
    console, buffer = make_console()
    with probe_progress(
        console,
        catalog_id="neighborhood:gcs-lifecycle-rules",
        total=2,
        truth=truth,
    ) as record:
        record(1, 2, make_result("q-lifecycle", "gcs-lifecycle-rules"))
        record(2, 2, make_result("q-retention", "gcs-retention-policy"))
    shown = rendered(buffer)
    assert shown.count("probing  neighborhood:gcs-lifecycle-rules") == 1


def test_print_query_set_omits_floating_table_header_when_empty(
    make_console,
    rendered,
) -> None:
    """Verify print_query_set omits the table header row when zero queries were drafted."""
    empty_set = QuerySet(
        catalog_id="all",
        notes="",
        queries=(),
        provenance=QuerySetProvenance(origin=Origin.AUTHORED),
    )
    console, buffer = make_console(width=200)
    print_query_set(
        console,
        empty_set,
        path=None,
        catalog_id="all",
        residents=3,
        ranks={},
        flags={},
    )
    shown = rendered(buffer)
    assert "0 queries drafted" in shown
    assert "expects" not in shown


def test_scorecard_renders_consistency_dash_when_no_repeated_queries(
    card,
    artifact: Artifact,
) -> None:
    """Verify print_scorecard renders 'consistency —' when spread.repeated_queries is zero."""
    single_attempt = artifact.model_copy(
        update={
            "spread": artifact.spread.model_copy(
                update={"replicates": 1, "repeated_queries": 0},
            ),
        },
    )
    shown = card(single_attempt)
    assert "consistency —" in shown
    assert not re.search(r"consistency \d+%", shown)


def test_print_query_view_folds_long_leaks_citations_and_text_without_ellipsis(
    make_console,
    rendered,
) -> None:
    """Verify print_query_view wraps long leak, citation, and text cells instead of truncating."""
    from reach.views.base import print_query_view

    q_text = (
        "Configure vault retention locks and governance policies for archival storage canisters "
        "with compliance auditing enabled."
    )
    citation_text = (
        "## Retention Policies\nUse `vaultctl canisters update --retention-period` "
        "to lock archival retention windows."
    )
    qs = QuerySet(
        catalog_id="all",
        notes="",
        queries=(
            Query(
                id="q-long-1",
                text=q_text,
                expected_skill="vault-retention-policy",
            ),
        ),
        provenance=QuerySetProvenance(origin=Origin.AUTHORED),
    )
    console, buffer = make_console(width=80)
    print_query_view(
        console,
        qs,
        ranks={"q-long-1": LexicalRank(position=1, field_size=10)},
        flags={
            "q-long-1": Leak(
                names_target=True,
                distinctive_tokens=("retention", "canisters", "archival", "governance"),
            )
        },
        citations={("vault-retention-policy", q_text): citation_text},
    )
    shown = rendered(buffer)
    flat = " ".join(shown.split())
    assert "…" not in shown
    assert "governance" in flat
    assert "windows." in flat
    assert "enabled." in flat


@pytest.mark.parametrize(
    ("severity", "expected_style"),
    [
        ("error", "reach.error"),
        ("info", "reach.label"),
        ("warn", "reach.misroute"),
    ],
)
def test_severity_style_maps_severity_enum(severity: str, expected_style: str) -> None:
    """Verify _severity_style maps Severity enum members to their canonical Rich styles."""
    from reach.lint import Severity
    from reach.views.lint import _severity_style

    assert _severity_style(Severity(severity)) == expected_style
