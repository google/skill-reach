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

"""Unit tests for token attribution diagnostics and misrouting analysis."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from reach.attribution import (
    QueryAttribution,
    attribute_query,
)
from reach.models import Skill
from reach.retrieval import Bm25Scorer
from reach.views import (
    format_annotated_query,
    render_attribution_table,
)


@pytest.fixture
def sample_skills() -> list[Skill]:
    """Provide a minimal corpus of skills for attribution testing."""
    return [
        Skill(
            name="gcs-lifecycle-rules",
            description=(
                "Manage Google Cloud Storage objects, lifecycle rules, and delete old files."
            ),
            path=Path("/skills/gcs-lifecycle-rules"),
        ),
        Skill(
            name="gcs-retention-policy",
            description=(
                "Configure bucket retention policy, compliance locks, and legal hold rules."
            ),
            path=Path("/skills/gcs-retention-policy"),
        ),
        Skill(
            name="gcs-monitoring",
            description="Inspect bucket metrics, operation latency, and access logs.",
            path=Path("/skills/gcs-monitoring"),
        ),
    ]


@pytest.fixture
def scorer(sample_skills: list[Skill]) -> Bm25Scorer:
    """Instantiate a Bm25Scorer over the sample skill corpus."""
    return Bm25Scorer.from_skills(sample_skills)


def test_attribute_query_identifies_drivers_and_anchors(scorer: Bm25Scorer) -> None:
    """Verify attribute_query identifies rival drivers, target anchors, and gaps."""
    query = "Set retention policy to delete old objects"
    attribution = attribute_query(
        query_text=query,
        target_skill="gcs-lifecycle-rules",
        rival_skill="gcs-retention-policy",
        scorer=scorer,
    )

    assert isinstance(attribution, QueryAttribution)
    assert attribution.target_skill == "gcs-lifecycle-rules"
    assert attribution.rival_skill == "gcs-retention-policy"

    driver_tokens = {d.token for d in attribution.drivers}
    assert "retention" in driver_tokens or "policy" in driver_tokens

    gap_tokens = {d.token for d in attribution.drivers if d.is_target_gap}
    assert "policy" in gap_tokens or "retention" in gap_tokens

    anchor_tokens = {a.token for a in attribution.anchors}
    assert "delete" in anchor_tokens or "lifecycle" in anchor_tokens or "objects" in anchor_tokens

    assert attribution.target_total > 0
    assert attribution.rival_total > 0
    assert attribution.net_bias == pytest.approx(attribution.rival_total - attribution.target_total)


def test_attribute_query_with_unmatched_tokens(scorer: Bm25Scorer) -> None:
    """Verify completely unmatched tokens are categorized in unscored_tokens."""
    query = "quantum physics entanglement"
    attribution = attribute_query(
        query_text=query,
        target_skill="gcs-lifecycle-rules",
        rival_skill="gcs-retention-policy",
        scorer=scorer,
    )

    assert attribution.drivers == ()
    assert attribution.anchors == ()
    assert attribution.target_total == 0.0
    assert attribution.rival_total == 0.0
    assert attribution.net_bias == 0.0
    assert "quantum" in attribution.unscored_tokens
    assert "physics" in attribution.unscored_tokens


def test_format_annotated_query(scorer: Bm25Scorer) -> None:
    """Verify format_annotated_query returns styled text for driver and anchor words."""
    query = "Set retention policy to delete old objects"
    attribution = attribute_query(
        query_text=query,
        target_skill="gcs-lifecycle-rules",
        rival_skill="gcs-retention-policy",
        scorer=scorer,
    )
    annotated = format_annotated_query(attribution)
    plain = annotated.plain
    assert plain == query
    # Check that styles were applied to the spans
    assert len(annotated.spans) > 0


def test_render_attribution_table(scorer: Bm25Scorer) -> None:
    """Verify render_attribution_table builds a valid Rich Table with columns."""
    query = "Set retention policy to delete old objects"
    attribution = attribute_query(
        query_text=query,
        target_skill="gcs-lifecycle-rules",
        rival_skill="gcs-retention-policy",
        scorer=scorer,
    )
    table = render_attribution_table(attribution)
    column_names = [col.header for col in table.columns]
    assert "token" in column_names
    assert "role" in column_names
    assert len(table.rows) > 0


def test_query_attribution_json_serialization(scorer: Bm25Scorer) -> None:
    """Verify QueryAttribution serializes and deserializes cleanly to JSON."""
    query = "Set retention policy"
    attribution = attribute_query(
        query_text=query,
        target_skill="gcs-lifecycle-rules",
        rival_skill="gcs-retention-policy",
        scorer=scorer,
    )
    raw = attribution.model_dump_json()
    data = json.loads(raw)
    assert data["target_skill"] == "gcs-lifecycle-rules"
    assert data["rival_skill"] == "gcs-retention-policy"
    restored = QueryAttribution.model_validate(data)
    assert restored.net_bias == attribution.net_bias


@pytest.fixture
def attributed_skills_dir(tmp_path: Path) -> Path:
    """Create a temporary skills directory containing skill-a and skill-b."""
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    (skills_dir / "skill-a").mkdir()
    (skills_dir / "skill-a" / "SKILL.md").write_text(
        "---\nname: skill-a\n"
        "description: Manage cloud storage lifecycle rules and delete logs.\n"
        "---\nBody",
        encoding="utf-8",
    )
    (skills_dir / "skill-b").mkdir()
    (skills_dir / "skill-b" / "SKILL.md").write_text(
        "---\nname: skill-b\n"
        "description: Configure bucket retention policy and compliance locks.\n"
        "---\nBody",
        encoding="utf-8",
    )
    return skills_dir


def test_cli_overlap_explain_subcommand(attributed_skills_dir: Path) -> None:
    """Verify reach overlap explain subcommand runs and outputs attribution."""
    from reach.cli.app import app

    code = app(
        [
            "overlap",
            "explain",
            "Set retention policy to delete logs",
            "--skill",
            "skill-a",
            "--rival",
            "skill-b",
            "--skills",
            str(attributed_skills_dir),
        ],
    )
    assert code == 0


def test_cli_overlap_explain_json_format(
    attributed_skills_dir: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify reach overlap explain supports JSON output."""
    from reach.cli.app import app

    code = app(
        [
            "overlap",
            "explain",
            "Set retention policy",
            "--skill",
            "skill-a",
            "--rival",
            "skill-b",
            "--skills",
            str(attributed_skills_dir),
            "--format",
            "json",
        ],
    )
    assert code == 0
    captured = capsys.readouterr().out
    data = json.loads(captured)
    assert data["target_skill"] == "skill-a"
    assert data["rival_skill"] == "skill-b"


def test_cli_overlap_explain_flag(attributed_skills_dir: Path) -> None:
    """Verify reach overlap explain subcommand runs attribution."""
    from reach.cli.app import app

    code = app(
        [
            "overlap",
            "explain",
            "Set retention policy to delete logs",
            "--skill",
            "skill-a",
            "--rival",
            "skill-b",
            "--skills",
            str(attributed_skills_dir),
        ],
    )
    assert code == 0


def test_attribute_query_empty_string(scorer: Bm25Scorer) -> None:
    """Verify empty or whitespace-only query produces empty attribution with 0 net bias."""
    attribution = attribute_query(
        query_text="   ",
        target_skill="gcs-lifecycle-rules",
        rival_skill="gcs-retention-policy",
        scorer=scorer,
    )
    assert attribution.drivers == ()
    assert attribution.anchors == ()
    assert attribution.unscored_tokens == ()
    assert attribution.target_total == 0.0
    assert attribution.rival_total == 0.0
    assert attribution.net_bias == 0.0


def test_attribute_query_neutral_token_equal_scores() -> None:
    """Verify token with identical BM25 contribution in both skills has delta == 0."""
    skills = [
        Skill(
            name="skill-x",
            description="Manage common deployment cluster.",
            path=Path("/skills/skill-x"),
        ),
        Skill(
            name="skill-y",
            description="Inspect common deployment cluster.",
            path=Path("/skills/skill-y"),
        ),
    ]
    scorer = Bm25Scorer.from_skills(skills)
    attribution = attribute_query(
        query_text="common deployment cluster",
        target_skill="skill-x",
        rival_skill="skill-y",
        scorer=scorer,
    )
    # Equal contributions mean delta is 0.0, so neither driver nor anchor
    assert attribution.target_total == pytest.approx(attribution.rival_total)
    assert attribution.net_bias == pytest.approx(0.0)
    assert len(attribution.drivers) == 0
    assert len(attribution.anchors) == 0


def test_cli_overlap_explain_auto_selects_nearest_rival(
    attributed_skills_dir: Path,
) -> None:
    """Verify reach overlap explain auto-selects the nearest rival if --rival is omitted."""
    from reach.cli.app import app

    code = app(
        [
            "overlap",
            "explain",
            "Set retention policy to delete logs",
            "--skill",
            "skill-a",
            "--skills",
            str(attributed_skills_dir),
        ],
    )
    assert code == 0


def test_cli_overlap_explain_jsonl_format(
    attributed_skills_dir: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify reach overlap explain supports JSONL output."""
    from reach.cli.app import app

    code = app(
        [
            "overlap",
            "explain",
            "Set retention policy",
            "--skill",
            "skill-a",
            "--skills",
            str(attributed_skills_dir),
            "--format",
            "jsonl",
        ],
    )
    assert code == 0
    captured = capsys.readouterr().out.strip()
    data = json.loads(captured)
    assert data["target_skill"] == "skill-a"
    assert data["rival_skill"] == "skill-b"


def test_cli_overlap_explain_unknown_target_fails(attributed_skills_dir: Path) -> None:
    """Verify reach overlap explain raises ValueError for unknown target skill."""
    from reach.cli.app import app

    with pytest.raises(ValueError, match="target skill 'nonexistent' not found in resident skills"):
        app(
            [
                "overlap",
                "explain",
                "Set retention policy",
                "--skill",
                "nonexistent",
                "--skills",
                str(attributed_skills_dir),
            ],
        )


def test_cli_overlap_explain_unknown_rival_fails(attributed_skills_dir: Path) -> None:
    """Verify reach overlap explain raises ValueError for unknown rival skill."""
    from reach.cli.app import app

    with pytest.raises(ValueError, match="rival skill 'nonexistent' not found in resident skills"):
        app(
            [
                "overlap",
                "explain",
                "Set retention policy",
                "--skill",
                "skill-a",
                "--rival",
                "nonexistent",
                "--skills",
                str(attributed_skills_dir),
            ],
        )


def test_cli_overlap_explain_no_rival_available_fails(tmp_path: Path) -> None:
    """Verify reach overlap explain raises ValueError when target has no competing rivals."""
    lone_dir = tmp_path / "lone"
    lone_dir.mkdir()
    (lone_dir / "solo").mkdir()
    (lone_dir / "solo" / "SKILL.md").write_text(
        "---\nname: solo\ndescription: A solo skill with no rivals.\n---\nBody",
        encoding="utf-8",
    )

    from reach.cli.app import app

    with pytest.raises(ValueError, match="no competing rival found for skill 'solo'"):
        app(
            [
                "overlap",
                "explain",
                "Solo task",
                "--skill",
                "solo",
                "--skills",
                str(lone_dir),
            ],
        )


def test_cli_overlap_flag_without_skill_fails(attributed_skills_dir: Path) -> None:
    """Verify reach overlap explain without --skill exits 2."""
    from reach.cli import main

    assert (
        main(
            [
                "overlap",
                "explain",
                "Some query",
                "--skills",
                str(attributed_skills_dir),
            ],
        )
        == 2
    )


def test_cli_overlap_explain_self_comparison_fails(attributed_skills_dir: Path) -> None:
    """Verify reach overlap explain rejects comparing a skill to itself."""
    from reach.cli.app import app

    with pytest.raises(ValueError, match="cannot be the same skill"):
        app(
            [
                "overlap",
                "explain",
                "Some query",
                "--skill",
                "skill-a",
                "--rival",
                "skill-a",
                "--skills",
                str(attributed_skills_dir),
            ],
        )


def test_attribute_query_abstention_diagnostics(scorer: Bm25Scorer) -> None:
    """Verify attribute_query diagnoses abstentions when rival is None or 'none'."""
    query = "Set retention policy to archive old files"
    attribution = attribute_query(
        query_text=query,
        target_skill="gcs-retention-policy",
        rival_skill="none",
        scorer=scorer,
    )
    assert attribution.rival_skill == "(no selection)"
    assert attribution.rival_total == 0.0
    assert attribution.target_total > 0.0
    assert attribution.net_bias == -attribution.target_total

    # Target matching tokens are anchors
    anchor_tokens = {a.token for a in attribution.anchors}
    assert "retention" in anchor_tokens or "policy" in anchor_tokens
    # Missing words in target description are in unscored_tokens
    assert "archive" in attribution.unscored_tokens


def test_attribute_query_with_semantic_scores(scorer: Bm25Scorer) -> None:
    """Verify attribute_query incorporates semantic similarity when provided."""
    attribution = attribute_query(
        query_text="delete logs",
        target_skill="gcs-lifecycle-rules",
        rival_skill="gcs-retention-policy",
        scorer=scorer,
        target_semantic=0.85,
        rival_semantic=0.60,
    )
    assert attribution.target_semantic == 0.85
    assert attribution.rival_semantic == 0.60
    assert attribution.semantic_bias == pytest.approx(-0.25)


def test_cli_overlap_explain_abstention_subcommand(attributed_skills_dir: Path) -> None:
    """Verify reach overlap explain --rival none runs abstention diagnosis."""
    from reach.cli.app import app

    code = app(
        [
            "overlap",
            "explain",
            "Set retention policy to archive old files",
            "--skill",
            "skill-b",
            "--rival",
            "none",
            "--skills",
            str(attributed_skills_dir),
        ],
    )
    assert code == 0


def test_cli_overlap_explain_semantic_flag(attributed_skills_dir: Path) -> None:
    """Verify reach overlap explain --semantic incorporates dense embeddings."""
    from reach.cli.app import app

    code = app(
        [
            "overlap",
            "explain",
            "Set retention policy",
            "--skill",
            "skill-a",
            "--rival",
            "skill-b",
            "--skills",
            str(attributed_skills_dir),
            "--semantic",
        ],
    )
    assert code == 0


def test_cli_overlap_explain_abstention_with_semantic_flag(attributed_skills_dir: Path) -> None:
    """Verify reach overlap explain with --rival none and --semantic handles None rival_obj."""
    from reach.cli.app import app

    code = app(
        [
            "overlap",
            "explain",
            "Set retention policy to archive old files",
            "--skill",
            "skill-b",
            "--rival",
            "none",
            "--skills",
            str(attributed_skills_dir),
            "--semantic",
        ],
    )
    assert code == 0


def test_attribute_query_excludes_function_words_from_drivers_and_anchors() -> None:
    """Verify stopwords like 'a', 'and', 'for', 'to', 'on' are excluded from drivers and anchors."""
    skills = [
        Skill(
            name="widget-deployer",
            description=(
                "Deploy a packaged widget service on relay nodes and scale for burst traffic."
            ),
            path=Path("/skills/widget-deployer"),
        ),
        Skill(
            name="gadget-scheduler",
            description="Deploy scheduled batch jobs and recurring workflows.",
            path=Path("/skills/gadget-scheduler"),
        ),
    ]
    scorer = Bm25Scorer.from_skills(skills)
    attribution = attribute_query(
        query_text="Use a skill to deploy and scale on relay nodes for burst traffic",
        target_skill="gadget-scheduler",
        rival_skill="widget-deployer",
        scorer=scorer,
    )
    scored_tokens = {d.token for d in attribution.drivers} | {a.token for a in attribution.anchors}
    for stopword in ("a", "and", "for", "on", "to"):
        assert stopword not in scored_tokens
        assert stopword in attribution.unscored_tokens
    assert "relay" in scored_tokens or "scale" in scored_tokens
