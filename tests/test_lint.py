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

"""Verify static skill linting rules, severity configuration, and report generation."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from reach.lint import (
    LintSettings,
    RuleDefinition,
    Severity,
    explain_rule,
    lint_file,
    lint_tree,
)
from reach.rendering import format_github_annotation

if TYPE_CHECKING:
    from collections.abc import Callable


def test_valid_skill_passes_clean(write_skill: Callable[..., Path]) -> None:
    """Verify that a well-formed skill produces zero lint issues."""
    manifest = (
        write_skill(
            name="valid-skill",
            description="Extract and parse tabular data from PDF invoices accurately.",
        )
        / "SKILL.md"
    )
    report = lint_file(manifest)
    assert report.clean
    assert len(report.issues) == 0


@pytest.mark.parametrize(
    ("name", "raw_yaml", "expected_rule", "expected_severity"),
    [
        (
            "bad-yaml",
            "---\nname: [unclosed list\n---\n",
            "invalid-yaml",
            Severity.ERROR,
        ),
        (
            "no-frontmatter",
            "# Just markdown\nNo frontmatter here.\n",
            "invalid-yaml",
            Severity.ERROR,
        ),
        (
            "no-name",
            "---\ndescription: Has a description but no name.\n---\n",
            "missing-name",
            Severity.ERROR,
        ),
        (
            "no-desc",
            "---\nname: no-desc\ndescription: ''\n---\n",
            "missing-description",
            Severity.ERROR,
        ),
    ],
)
def test_invalid_frontmatter_rules(
    write_skill: Callable[..., Path],
    name: str,
    raw_yaml: str,
    expected_rule: str,
    expected_severity: Severity,
) -> None:
    """Verify malformed frontmatter triggers corresponding error rules."""
    manifest = write_skill(name=name, raw_yaml=raw_yaml) / "SKILL.md"
    report = lint_file(manifest)
    assert report.has_errors
    assert any(i.rule == expected_rule and i.severity == expected_severity for i in report.issues)


def test_lint_file_accepts_unicode_bom(tmp_path: Path) -> None:
    """Verify lint_file successfully parses and lints files with leading Unicode BOM."""
    skill_dir = tmp_path / "bom-skill"
    skill_dir.mkdir(parents=True)
    manifest = skill_dir / "SKILL.md"
    manifest.write_text(
        "\ufeff---\nname: bom-skill\ndescription: Skill saved with Unicode BOM.\n---\nBody\n",
        encoding="utf-8",
    )
    report = lint_file(manifest)
    assert not report.has_errors
    assert report.skills_checked == 1


@pytest.mark.parametrize(
    "invalid_name",
    [
        "CamelCaseName",
        "name_with_underscores",
        "name with spaces",
        "-leading-hyphen",
        "trailing-hyphen-",
        "double--hyphen",
        "special@chars!",
    ],
)
def test_invalid_name_format(write_skill: Callable[..., Path], invalid_name: str) -> None:
    """Verify that names not matching kebab-case produce invalid-name-format error."""
    manifest = (
        write_skill(
            name=invalid_name,
            description="A sufficiently long description for testing invalid names.",
        )
        / "SKILL.md"
    )
    report = lint_file(manifest)
    assert any(
        i.rule == "invalid-name-format" and i.severity == Severity.ERROR for i in report.issues
    )


def test_name_mismatch(write_skill: Callable[..., Path]) -> None:
    """Verify that a name differing from the directory name produces name-mismatch error."""
    manifest = (
        write_skill(
            name="declared-name",
            description="A sufficiently long description for testing directory name mismatch.",
            dir_name="dir-name",
        )
        / "SKILL.md"
    )
    report = lint_file(manifest)
    assert any(i.rule == "name-mismatch" and i.severity == Severity.ERROR for i in report.issues)


def test_description_too_short(write_skill: Callable[..., Path]) -> None:
    """Verify that descriptions under threshold produce description-too-short warning."""
    manifest = (
        write_skill(
            name="terse-skill",
            description="Does stuff.",
        )
        / "SKILL.md"
    )
    report = lint_file(manifest, config=LintSettings(min_description_length=20))
    assert any(
        i.rule == "description-too-short" and i.severity == Severity.WARN for i in report.issues
    )


@pytest.mark.parametrize(
    "placeholder",
    ["TODO: write this", "FIXME later", "Run <FILL_IN> here", "Check [TODO] item"],
)
def test_unresolved_placeholder(write_skill: Callable[..., Path], placeholder: str) -> None:
    """Verify that placeholder strings produce unresolved-placeholder warning."""
    manifest = (
        write_skill(
            name="placeholder-skill",
            description=f"Automate cloud deployments {placeholder} for clusters.",
        )
        / "SKILL.md"
    )
    report = lint_file(manifest)
    assert any(
        i.rule == "unresolved-placeholder" and i.severity == Severity.WARN for i in report.issues
    )


@pytest.mark.parametrize(
    "reserved",
    ["bash", "edit", "grep", "read", "skill", "task", "view_file"],
)
def test_reserved_name_collision(write_skill: Callable[..., Path], reserved: str) -> None:
    """Verify that skill names colliding with built-in primitives produce a warning."""
    manifest = (
        write_skill(
            name=reserved,
            description="Execute shell commands and manage background processes.",
        )
        / "SKILL.md"
    )
    report = lint_file(manifest)
    assert any(
        i.rule == "reserved-name-collision" and i.severity == Severity.WARN for i in report.issues
    )


def test_listing_overflow(write_skill: Callable[..., Path]) -> None:
    """Verify that descriptions exceeding listing threshold produce listing-overflow warning."""
    huge_desc = "A" * 1200
    manifest = (
        write_skill(
            name="huge-skill",
            description=huge_desc,
        )
        / "SKILL.md"
    )
    report = lint_file(manifest, config=LintSettings(max_description_length=1000))
    assert any(i.rule == "listing-overflow" and i.severity == Severity.WARN for i in report.issues)


def test_duplicate_name_across_corpus(write_skill: Callable[..., Path], tmp_path: Path) -> None:
    """Verify that duplicate skill names across the tree produce duplicate-name error."""
    write_skill(
        name="shared-name",
        description="First skill implementation with this specific name.",
        root=tmp_path / "repo1",
    )
    write_skill(
        name="shared-name",
        description="Second skill implementation colliding with the first.",
        root=tmp_path / "repo2",
    )
    report = lint_tree(tmp_path)
    assert report.has_errors
    duplicates = [i for i in report.issues if i.rule == "duplicate-name"]
    assert len(duplicates) >= 1
    assert duplicates[0].severity == Severity.ERROR


def test_rule_severity_override_ignore(write_skill: Callable[..., Path]) -> None:
    """Verify that setting rule severity to ignore silences the issue."""
    manifest = (
        write_skill(
            name="terse-skill",
            description="Does stuff.",
        )
        / "SKILL.md"
    )
    config = LintSettings(rules={"description-too-short": Severity.IGNORE})
    report = lint_file(manifest, config=config)
    assert not any(i.rule == "description-too-short" for i in report.issues)


def test_rule_severity_override_elevate_to_error(write_skill: Callable[..., Path]) -> None:
    """Verify that overriding a warning to error promotes its severity."""
    manifest = (
        write_skill(
            name="terse-skill",
            description="Does stuff.",
        )
        / "SKILL.md"
    )
    config = LintSettings(rules={"description-too-short": Severity.ERROR})
    report = lint_file(manifest, config=config)
    issue = next(i for i in report.issues if i.rule == "description-too-short")
    assert issue.severity == Severity.ERROR


def test_explain_rule() -> None:
    """Verify explain_rule returns complete metadata for known rules and None for unknown."""
    rule = explain_rule("description-too-short")
    assert rule is not None
    assert isinstance(rule, RuleDefinition)
    assert rule.rule == "description-too-short"
    assert rule.default_severity == Severity.WARN
    assert rule.summary
    assert rule.explanation
    assert rule.remedy

    assert explain_rule("non-existent-rule") is None


def test_lint_config_from_settings_loads_thresholds_and_rules() -> None:
    """Verify LintSettings.from_settings parses thresholds, overrides, and validates rules."""
    from pydantic import ValidationError

    settings = {
        "lint": {
            "max_description_length": 500,
            "max_name_length": 32,
            "min_description_length": 40,
            "mutual_handoff_similarity_threshold": 0.82,
            "mutual_handoff_lexical_threshold": 0.45,
            "rules": {
                "description-too-short": "error",
                "invalid-name-format": "warn",
                "lockfile-drift": "INFO",
            },
        },
        "retrieval": {"similarity_threshold": 0.85},
    }
    config = LintSettings.from_settings(
        settings,
        overrides={"kebab-case-name": Severity.WARN},
    )
    assert config.max_description_length == 500
    assert config.max_name_length == 32
    assert config.min_description_length == 40
    assert config.similarity_threshold == pytest.approx(0.85)
    assert config.mutual_handoff_similarity_threshold == pytest.approx(0.82)
    assert config.mutual_handoff_lexical_threshold == pytest.approx(0.45)
    assert config.rules["description-too-short"] == Severity.ERROR
    assert config.rules["invalid-name-format"] == Severity.WARN
    assert config.rules["lockfile-drift"] == Severity.INFO
    assert config.rules["kebab-case-name"] == Severity.WARN

    with pytest.raises(ValidationError):
        LintSettings.from_settings({"lint": {"rules": {"lockfile-drift": "not-a-severity"}}})


def test_custom_max_name_length_threshold(write_skill: Callable[..., Path]) -> None:
    """Verify that skill names exceeding a custom max_name_length are rejected."""
    long_name = "this-is-a-moderately-long-name"
    assert len(long_name) == 30
    manifest = (
        write_skill(
            name=long_name,
            description="A sufficiently long description for testing custom max name length.",
        )
        / "SKILL.md"
    )
    # Default 64 chars allows 30 chars
    report_default = lint_file(manifest)
    assert report_default.clean

    # Custom 25 chars rejects 30 chars
    config_custom = LintSettings(max_name_length=25)
    report_custom = lint_file(manifest, config=config_custom)
    assert not report_custom.clean
    issue = next(i for i in report_custom.issues if i.rule == "invalid-name-format")
    assert "max 25 characters" in issue.message


def test_custom_min_description_length_threshold(write_skill: Callable[..., Path]) -> None:
    """Verify that descriptions below a custom min_description_length are flagged."""
    desc = "Twenty-five char desc...."
    assert len(desc) == 25
    manifest = (
        write_skill(
            name="custom-len-skill",
            description=desc,
        )
        / "SKILL.md"
    )
    # Default 20 chars allows 25 chars
    report_default = lint_file(manifest)
    assert not any(i.rule == "description-too-short" for i in report_default.issues)

    # Custom 30 chars flags 25 chars
    config_custom = LintSettings(min_description_length=30)
    report_custom = lint_file(manifest, config=config_custom)
    issue = next(i for i in report_custom.issues if i.rule == "description-too-short")
    assert "30 chars" in issue.message


def test_format_github_annotation_basic() -> None:
    """Verify format_github_annotation formats severity, properties, and message."""
    from reach.rendering import format_github_annotation

    result = format_github_annotation(
        "error",
        "Invalid schema in manifest",
        title="invalid-yaml",
        file="skills/demo/SKILL.md",
        line=1,
    )
    expected = (
        "::error file=skills/demo/SKILL.md,line=1,title=invalid-yaml::Invalid schema in manifest"
    )
    assert result == expected


def test_format_github_annotation_escapes_newlines_and_percent() -> None:
    """Verify format_github_annotation encodes newlines and percent signs in messages."""
    result = format_github_annotation(
        "warn",
        "First line\nSecond line has 50% rate",
    )
    assert result == "::warning::First line%0ASecond line has 50%25 rate"


def test_format_github_annotation_escapes_parameter_delimiters() -> None:
    """Verify format_github_annotation escapes %, CRLF, colons, and commas in parameter values."""
    result = format_github_annotation(
        "error",
        "Found issue",
        title="bad:rule,v100%\r\n::workflow-cmd",
        file="skills/demo:special,v1%0A/SKILL.md",
    )
    assert "title=bad%3Arule%2Cv100%25%0D%0A%3A%3Aworkflow-cmd" in result
    assert "file=skills/demo%3Aspecial%2Cv1%250A/SKILL.md" in result
    # Ensure raw unescaped delimiters do not split parameters or commands
    assert "\r" not in result
    assert "\n" not in result


def test_format_github_annotation_converts_absolute_workspace_path_to_relative(
    tmp_path: Path,
) -> None:
    """Verify format_github_annotation converts absolute file paths under root to relative paths."""
    workspace = tmp_path / "repo"
    skill_file = workspace / "skills" / "demo" / "SKILL.md"
    result = format_github_annotation(
        "error",
        "Invalid schema",
        file=skill_file,
        root=workspace,
    )
    assert "file=skills/demo/SKILL.md" in result
    assert str(skill_file) not in result


def test_format_github_annotation_converts_cwd_path_to_relative() -> None:
    """Verify format_github_annotation relativizes paths under current working directory."""
    from reach.rendering import format_github_annotation

    abs_path = Path.cwd() / "skills" / "demo" / "SKILL.md"
    result = format_github_annotation(
        "error",
        "Invalid schema",
        file=abs_path,
    )
    assert "file=skills/demo/SKILL.md" in result


def test_render_lint_github_formats_all_issues(write_skill: Callable[..., Path]) -> None:
    """Verify render_lint_github formats all report issues as workflow commands."""
    from reach.lint import lint_file
    from reach.views.lint import render_lint_github

    manifest = (
        write_skill(
            name="bad_naming",
            description="A short description for testing github workflow formatting.",
        )
        / "SKILL.md"
    )
    report = lint_file(manifest)
    rendered = render_lint_github(report)

    assert "::error" in rendered
    assert "title=invalid-name-format" in rendered
    assert str(manifest) in rendered


def test_lint_tree_flags_near_duplicate_capability(
    write_skill: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """Verify lint_tree identifies semantically redundant skill descriptions."""
    from unittest.mock import patch

    from reach.retrieval import DenseScorer

    write_skill(
        name="pdf-parser",
        description="Extract structured tables from PDF files into spreadsheets.",
    )
    write_skill(
        name="pdf-extractor",
        description="Extract tabular information from PDF documents into spreadsheets.",
    )
    write_skill(
        name="image-editor",
        description="Crop and resize bitmap images and photographs.",
    )

    mock_vectors = {
        "pdf-parser": [1.0, 0.95, 0.0],
        "pdf-extractor": [0.99, 0.94, 0.0],
        "image-editor": [0.0, 0.0, 1.0],
    }
    with patch.object(DenseScorer, "from_skills", return_value=DenseScorer(vectors=mock_vectors)):
        report = lint_tree(tmp_path)
        dup_issues = [i for i in report.issues if i.rule == "duplicate-capability"]
        assert len(dup_issues) >= 1
        assert "pdf-parser" in dup_issues[0].message
        assert "pdf-extractor" in dup_issues[0].message


def test_unresolved_declared_dependency_warns_when_missing_from_corpus(
    write_skill: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """Verify that a declared dependency missing from the corpus produces a warning."""
    write_skill(
        name="caller-skill",
        description="A skill that requires an absent dependency.",
        raw_yaml=(
            "---\n"
            "name: caller-skill\n"
            "description: A skill that requires an absent dependency.\n"
            "metadata:\n"
            "  requires_skill: missing-helper\n"
            "---\n"
        ),
    )
    report = lint_tree(tmp_path)
    issue = next((i for i in report.issues if i.rule == "unresolved-declared-dependency"), None)
    assert issue is not None
    assert issue.severity == Severity.WARN
    assert "missing-helper" in issue.message


def test_unresolved_declared_dependency_passes_when_present(
    write_skill: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """Verify that a declared dependency present in the corpus passes cleanly."""
    write_skill(
        name="caller-skill",
        description="A skill that requires a present dependency.",
        raw_yaml=(
            "---\n"
            "name: caller-skill\n"
            "description: A skill that requires a present dependency.\n"
            "metadata:\n"
            "  requires_skill: helper-skill\n"
            "---\n"
        ),
    )
    write_skill(
        name="helper-skill",
        description="The helper skill present in the corpus.",
    )
    report = lint_tree(tmp_path)
    dep_issues = [i for i in report.issues if i.rule == "unresolved-declared-dependency"]
    assert len(dep_issues) == 0


def test_lockfile_drift_warns_on_hash_mismatch(
    write_skill: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """Verify that lockfile-drift emits a warning when SKILL.md hash diverges from lockfile."""
    skill_dir = write_skill(
        name="drift-skill",
        description="A skill whose content diverges from skills-lock.json.",
        root=tmp_path / ".agents" / "skills",
    )
    manifest = skill_dir / "SKILL.md"
    lockfile = tmp_path / "skills-lock.json"
    lockfile.write_text(
        '{"version": 1, "skills": {"drift-skill": {"computedHash": "expected-old-hash"}}}',
        encoding="utf-8",
    )
    report = lint_file(manifest)
    issue = next((i for i in report.issues if i.rule == "lockfile-drift"), None)
    assert issue is not None
    assert issue.severity == Severity.INFO
    assert report.infos == (issue,)
    assert "drift-skill" in issue.message


def test_lockfile_clean_when_hash_matches(
    write_skill: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """Verify that lockfile-drift does not fire when SKILL.md hash matches lockfile."""
    import hashlib

    skill_dir = write_skill(
        name="matching-skill",
        description="A skill whose content matches skills-lock.json perfectly.",
        root=tmp_path / ".agents" / "skills",
    )
    manifest = skill_dir / "SKILL.md"
    content_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    lockfile = tmp_path / "skills-lock.json"
    lockfile.write_text(
        f'{{"version": 1, "skills": {{"matching-skill": {{"computedHash": "{content_hash}"}}}}}}',
        encoding="utf-8",
    )
    report = lint_file(manifest)
    drift_issues = [i for i in report.issues if i.rule == "lockfile-drift"]
    assert len(drift_issues) == 0


@pytest.mark.parametrize(
    "empty_val",
    ['""', "''", "   ", "null"],
    ids=["double-quoted-empty", "single-quoted-empty", "whitespace", "null"],
)
def test_missing_description_in_lint_tree_does_not_crash(
    tmp_path: Path,
    empty_val: str,
) -> None:
    """Verify lint_tree handles empty/null descriptions and records missing-description."""
    skill_dir = tmp_path / "broken-skill"
    skill_dir.mkdir(parents=True)
    manifest = skill_dir / "SKILL.md"
    manifest.write_text(
        f"---\nname: broken-skill\ndescription: {empty_val}\n---\n# Body\n",
        encoding="utf-8",
    )
    report = lint_tree(tmp_path)
    assert report.skills_checked == 1
    assert any(
        i.rule == "missing-description" and i.severity == Severity.ERROR for i in report.issues
    )


@pytest.mark.parametrize(
    ("description", "self_name", "expected"),
    [
        (
            (
                "Monitors database connection pools and query execution history. "
                "Don't use for root-cause diagnosis when cause is unknown "
                "(use `db-troubleshooting` first), or for rewriting slow "
                "application queries (use `db-optimization`)."
            ),
            "db-observability",
            ("db-optimization", "db-troubleshooting"),
        ),
        (
            (
                "Analyzes SQL query execution plans and index costs. "
                "Don't use for generic database administration, vector search, or DataFrames "
                "(use `db-basics`, `db-vector-search`, or `db-dataframes`). "
                "Do not use for streaming ingestion — use kafka-streaming, flink-pipelines and "
                "db-bulk-writer."
            ),
            "db-cost-optimizer",
            (
                "db-basics",
                "db-bulk-writer",
                "db-dataframes",
                "db-vector-search",
                "flink-pipelines",
                "kafka-streaming",
            ),
        ),
        (
            "For query plan tuning and index cost analysis, use db-cost-optimizer instead.",
            "db-observability",
            ("db-cost-optimizer",),
        ),
        (
            (
                "Use on-demand pricing, real-time utf-8 streaming, and command-line flags. "
                "Use `docker compose` over legacy scripts and do not use "
                "db-observability for SQL."
            ),
            "db-observability",
            (),
        ),
        (
            "For database cluster tasks, defer to `service-specific` or prefer `db-related`.",
            "my-skill",
            ("db-related", "service-specific"),
        ),
    ],
)
def test_extract_skill_references_parses_handoffs_and_ignores_non_skills(
    description: str,
    self_name: str,
    expected: tuple[str, ...],
) -> None:
    """Verify extract_skill_references extracts skill handoffs while ignoring prose/CLI terms."""
    from reach.lint import extract_skill_references

    assert extract_skill_references(description, self_name=self_name) == expected


def test_unknown_skill_reference_flags_dangling_boundary_targets(
    write_skill: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """Verify unknown-skill-reference warns when description hands off to missing skills."""
    write_skill(
        name="db-observability",
        description=(
            "Monitors and analyzes database operational telemetry, query execution "
            "history, and connection pool utilization. "
            "Don't use for root-cause diagnosis or symptom troubleshooting when the cause is "
            "unknown (use `db-troubleshooting` first), or for writing or optimizing "
            "business logic SQL (use `db-optimization`)."
        ),
    )
    write_skill(
        name="db-basics",
        description="Manages database schemas, tables, and standard administrative operations.",
    )

    report = lint_tree(tmp_path)
    unknown_issues = [i for i in report.issues if i.rule == "unknown-skill-reference"]
    assert len(unknown_issues) == 2
    assert all(i.severity == Severity.WARN for i in unknown_issues)
    assert all(i.skill == "db-observability" for i in unknown_issues)
    messages = " ".join(i.message for i in unknown_issues)
    assert "db-troubleshooting" in messages
    assert "db-optimization" in messages


def test_unknown_skill_reference_passes_when_targets_exist_in_catalog(
    write_skill: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """Verify unknown-skill-reference does not fire when referenced handoff skills exist."""
    write_skill(
        name="db-observability",
        description=(
            "Monitors database telemetry. Don't use for root-cause troubleshooting "
            "(use `db-troubleshooting` first)."
        ),
    )
    write_skill(
        name="db-troubleshooting",
        description=(
            "Troubleshoots database errors. Don't use for routine telemetry monitoring "
            "(use `db-observability`)."
        ),
    )

    report = lint_tree(tmp_path)
    unknown_issues = [i for i in report.issues if i.rule == "unknown-skill-reference"]
    assert len(unknown_issues) == 0


def test_missing_mutual_handoff_flags_overlapping_database_skills(
    write_skill: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """Verify missing-mutual-handoff catches unguarded overlap between database neighbors."""
    write_skill(
        name="db-cost-optimizer",
        description=(
            "Analyzes PostgreSQL worker utilization, query execution bottlenecks, and "
            "pg_stat_statements telemetry. Use when diagnosing slow SQL queries, "
            "worker starvation, high query costs, or join performance bottlenecks. "
            "Don't use for generic database administration (use `db-basics`)."
        ),
    )
    write_skill(
        name="db-observability",
        description=(
            "Monitors PostgreSQL operational telemetry, worker utilization, and query "
            "execution bottlenecks using pg_stat_statements. Use when investigating worker "
            "usage trends, query concurrency, or capacity planning. "
            "Don't use for generic database administration (use `db-basics`)."
        ),
    )
    write_skill(
        name="db-basics",
        description=(
            "Creates and administers PostgreSQL schemas and tables. "
            "Don't use for query cost optimization (use `db-cost-optimizer`) "
            "or operational telemetry (use `db-observability`)."
        ),
    )

    report = lint_tree(tmp_path)
    mutual_issues = [i for i in report.issues if i.rule == "missing-mutual-handoff"]
    flagged_skills = {i.skill for i in mutual_issues}
    assert "db-cost-optimizer" in flagged_skills
    assert "db-observability" in flagged_skills
    optimizer_msg = next(i.message for i in mutual_issues if i.skill == "db-cost-optimizer")
    assert "db-observability" in optimizer_msg


def test_missing_mutual_handoff_flags_k8s_storage_troubleshooting_and_object_storage_fuse(
    write_skill: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """Verify missing-mutual-handoff catches k8s-storage-troubleshooting vs object-storage-fuse."""
    write_skill(
        name="k8s-storage-troubleshooting",
        description=(
            "Diagnoses and resolves Kubernetes storage issues including "
            "PVC Pending states, PersistentVolume mount failures, CSI driver errors, volume "
            "expansion failures, and Object Storage FUSE OOM. Use when pods fail to mount "
            "volumes or Kubernetes storage workloads crash. Don't use for initial storage "
            "provisioning or choosing storage types (use `k8s-storage`)."
        ),
    )
    write_skill(
        name="object-storage-fuse",
        description=(
            "Configures, mounts, and tunes Object Storage FUSE (s3fs/fuse) on Linux VMs "
            "and Kubernetes clusters for high-throughput training, caching, and file system "
            "performance."
        ),
    )
    write_skill(
        name="k8s-storage",
        description=(
            "Provisions and configures Kubernetes storage classes and volumes. "
            "Don't use for troubleshooting volume mount failures "
            "(use `k8s-storage-troubleshooting`)."
        ),
    )

    report = lint_tree(tmp_path)
    mutual_issues = [i for i in report.issues if i.rule == "missing-mutual-handoff"]
    k8s_issues = [
        i
        for i in mutual_issues
        if i.skill == "k8s-storage-troubleshooting" and "object-storage-fuse" in i.message
    ]
    fuse_issues = [
        i
        for i in mutual_issues
        if i.skill == "object-storage-fuse" and "k8s-storage-troubleshooting" in i.message
    ]
    assert len(k8s_issues) == 1
    assert len(fuse_issues) == 1
    assert k8s_issues[0].severity == Severity.WARN


def test_missing_mutual_handoff_resolves_when_reciprocal_handoffs_added(
    write_skill: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """Verify missing-mutual-handoff passes cleanly once both neighbors hand off to each other."""
    write_skill(
        name="k8s-storage-troubleshooting",
        description=(
            "Diagnoses and resolves Kubernetes storage issues and Object Storage FUSE OOM. "
            "Don't use for Object Storage FUSE performance tuning or mount configuration "
            "(use `object-storage-fuse`)."
        ),
    )
    write_skill(
        name="object-storage-fuse",
        description=(
            "Configures, mounts, and tunes Object Storage FUSE on Kubernetes clusters. "
            "Don't use for diagnosing PVC Pending or CSI crash troubleshooting "
            "(use `k8s-storage-troubleshooting`)."
        ),
    )

    report = lint_tree(tmp_path)
    mutual_issues = [i for i in report.issues if i.rule == "missing-mutual-handoff"]
    assert len(mutual_issues) == 0


def test_missing_mutual_handoff_fires_when_neither_skill_has_existing_boundaries(
    write_skill: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """Verify missing-mutual-handoff fires on unbounded pairs with phrase encroachment."""
    write_skill(
        name="k8s-storage-troubleshooting",
        description="Diagnoses and resolves Kubernetes storage issues and Object Storage FUSE OOM.",
    )
    write_skill(
        name="object-storage-fuse",
        description="Configures, mounts, and tunes Object Storage FUSE on Kubernetes clusters.",
    )

    report = lint_tree(tmp_path)
    mutual_issues = [i for i in report.issues if i.rule == "missing-mutual-handoff"]
    skills_flagged = {i.skill for i in mutual_issues}
    assert skills_flagged == {"k8s-storage-troubleshooting", "object-storage-fuse"}


@pytest.mark.parametrize(
    ("description", "self_name", "expected"),
    [
        (
            (
                "Plans, executes, and validates Kubernetes cluster upgrades "
                "and maintenance operations. Handles node pool upgrade strategies (surge, "
                "blue-green) and workload-specific concerns. Use this skill whenever the user "
                "mentions cluster upgrades or node pool maintenance. Don't use for cluster "
                "creation, general networking/routing setup, or security policy configurations "
                "(use k8s-basics or relevant cluster skills instead)."
            ),
            "k8s-upgrades",
            ("k8s-basics",),
        ),
        (
            (
                "Use this skill to manage compliance evaluations, rules, "
                "scanned resources, and validation results because no service-specific public "
                "CLI or MCP server is available."
            ),
            "compliance-manager-basics",
            (),
        ),
        (
            (
                "Interactively discovers requirements and designs holistic, multi-service system "
                "architectures. Don't use for single-service tasks (use "
                "service-specific skills), initial onboarding or authentication (use "
                "platform-recipe-*), architecture pillar reviews or audits (use "
                "platform-waf-*), or workloads covered by specialized solution skills."
            ),
            "platform-solution-architecture",
            ("platform-recipe", "platform-waf"),
        ),
        (
            (
                "Analyzes the downstream impact (blast radius) when a database table or view is "
                "broken or modified. Don't use for: - General SQL querying or data analysis "
                "(use database-related tools instead)."
            ),
            "lineage-asset-impact-analysis",
            (),
        ),
        (
            "Audits RBAC policies. Do not use for org-level, role-based, or read-only skills.",
            "rbac-audit",
            (),
        ),
        (
            (
                "Guides initial onboarding for the Webhook Ingestion API. Don't use for "
                "writing payload ingestion code (use the webhook-api-audience-ingestion "
                "or webhook-api-event-ingestion skills instead)."
            ),
            "webhook-api-setup",
            (
                "webhook-api-audience-ingestion",
                "webhook-api-event-ingestion",
            ),
        ),
        (
            "Use this skill when writing code or prefer that skill instead.",
            "my-skill",
            (),
        ),
        (
            "For language-specific configuration, prefer the framework-related skill first.",
            "my-skill",
            (),
        ),
    ],
)
def test_extract_skill_references_ignores_determiners_and_category_adjectives(
    description: str,
    self_name: str,
    expected: tuple[str, ...],
) -> None:
    """Verify Pattern A (determiners) and Pattern B (-specific/-related + skills) are ignored."""
    from reach.lint import extract_skill_references

    assert extract_skill_references(description, self_name=self_name) == expected


def test_unknown_skill_reference_allows_valid_wildcard_prefix_families_and_flags_missing(
    write_skill: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """Verify Pattern C wildcard handoffs (foo-*) pass when prefix family exists."""
    from reach.lint import find_unknown_skill_references, hands_off_to_skill

    desc_valid = (
        "Designs multi-service system architectures. Don't use for step-by-step "
        "onboarding recipes (use platform-recipe-*) or architecture pillar reviews "
        "(use `platform-waf-*`)."
    )
    known = (
        "platform-solution-architecture",
        "platform-recipe-auth",
        "platform-waf-security",
    )
    assert (
        find_unknown_skill_references(
            desc_valid,
            known,
            self_name="platform-solution-architecture",
        )
        == ()
    )
    assert hands_off_to_skill(desc_valid, "platform-recipe-auth")
    assert hands_off_to_skill(desc_valid, "platform-waf-security")

    desc_invalid = "Designs architectures. Don't use for missing family (use nonexistent-family-*)."
    assert find_unknown_skill_references(
        desc_invalid,
        known,
        self_name="platform-solution-architecture",
    ) == ("nonexistent-family",)

    write_skill(
        name="platform-solution-architecture",
        description=desc_valid,
    )
    write_skill(
        name="platform-recipe-auth",
        description="Configures service authentication and OAuth credentials.",
    )
    write_skill(
        name="platform-waf-security",
        description="Reviews architecture security pillar.",
    )
    report = lint_tree(tmp_path)
    unknown_issues = [i for i in report.issues if i.rule == "unknown-skill-reference"]
    assert unknown_issues == []


@pytest.mark.parametrize(
    ("description", "target_name", "expected"),
    [
        pytest.param(
            (
                "Manage social media queues. Use when the user asks to "
                "publish across connected social accounts."
            ),
            "social",
            False,
            id="positive-use-sentence-does-not-handoff-to-social",
        ),
        pytest.param(
            (
                "ALWAYS use this skill when asked to draft social media "
                "content for microblogging platforms."
            ),
            "social",
            False,
            id="always-use-sentence-does-not-handoff-to-social",
        ),
        pytest.param(
            (
                "Use when asked to audit a codebase or generate handoff "
                "plans for another agent to implement."
            ),
            "handoff",
            False,
            id="positive-use-sentence-does-not-handoff-to-handoff",
        ),
        pytest.param(
            (
                "Use when asked to audit a codebase or generate handoff "
                "plans for another agent to implement."
            ),
            "implement",
            False,
            id="positive-use-sentence-does-not-handoff-to-implement",
        ),
        pytest.param(
            "Use for notebook and source management, grounded chat and research.",
            "research",
            False,
            id="positive-use-sentence-does-not-handoff-to-research",
        ),
        pytest.param(
            "Do not use for the generic OpenAI API or unrelated content creation.",
            "openai-api",
            True,
            id="negative-clause-disclaims-multi-word-skill",
        ),
        pytest.param(
            "Don't use for quick red-green-refactor loops (use tdd).",
            "tdd",
            True,
            id="explicit-parenthetical-single-word-skill-handoff",
        ),
        pytest.param(
            "For broader social listening, see `social` instead.",
            "social",
            True,
            id="explicit-backtick-single-word-skill-handoff",
        ),
    ],
)
def test_hands_off_to_skill_single_word_and_multi_word_boundaries(
    description: str,
    target_name: str,
    expected: bool,
) -> None:
    """Verify positive 'Use when...' sentences do not falsely hand off to single-word skills."""
    from reach.lint import hands_off_to_skill

    assert hands_off_to_skill(description, target_name) is expected


def test_extract_corpus_semantics_deterministic_pre_filter(tmp_path: Path) -> None:
    """Verify extract_corpus_semantics pre-filters and extracts deterministic semantics."""
    from reach.lint import SkillLintSemantics, extract_corpus_semantics
    from reach.models import Skill

    skills = [
        Skill(
            name="skill-a",
            description="Do not use for specialized tasks; use target-one instead.",
            path=tmp_path / "a" / "SKILL.md",
        ),
        Skill(
            name="skill-b",
            description="Assist with any task. For specialized tasks, defer to target-two.",
            path=tmp_path / "b" / "SKILL.md",
        ),
        Skill(
            name="skill-plain",
            description="Profiles CPU and memory bottlenecks in Python scripts.",
            path=tmp_path / "plain" / "SKILL.md",
        ),
    ]

    semantics = extract_corpus_semantics(skills)
    assert "skill-plain" not in semantics
    assert semantics["skill-a"] == SkillLintSemantics(
        skill="skill-a",
        handoff_targets=("target-one",),
        unbounded_attractor_phrase=None,
    )
    assert semantics["skill-b"] == SkillLintSemantics(
        skill="skill-b",
        handoff_targets=("target-two",),
        unbounded_attractor_phrase="any",
    )


def test_shared_trigger_terms_uses_corpus_idf_without_stopword_list(tmp_path: Path) -> None:
    """Verify _shared_trigger_terms filters ubiquitous terms via BM25 IDF > BACKGROUND_IDF."""
    from reach.lint import _shared_trigger_terms
    from reach.models import Skill
    from reach.retrieval import Bm25Scorer

    # Create a 6-skill corpus where 'workflow', 'guide', 'project' appear across almost all skills
    # (low IDF <= BACKGROUND_IDF), while 'kubernetes' and 'helm' appear only in 2 competing skills.
    skills = [
        Skill(
            name="k8s-deploy",
            description="Guide for project workflow deploying kubernetes helm charts.",
            path=tmp_path / "1",
        ),
        Skill(
            name="k8s-debug",
            description="Guide for project workflow debugging kubernetes helm releases.",
            path=tmp_path / "2",
        ),
        Skill(
            name="doc-1",
            description="Guide for project workflow documentation and release notes.",
            path=tmp_path / "3",
        ),
        Skill(
            name="doc-2",
            description="Guide for project workflow testing and continuous integration.",
            path=tmp_path / "4",
        ),
        Skill(
            name="doc-3",
            description="Guide for project workflow formatting and static analysis.",
            path=tmp_path / "5",
        ),
        Skill(
            name="doc-4",
            description="Guide for project workflow packaging and publishing artifacts.",
            path=tmp_path / "6",
        ),
    ]
    scorer = Bm25Scorer.from_skills(skills)
    shared = _shared_trigger_terms(
        skills[0],
        skills[1],
        scorer=scorer,
    )
    assert "kubernetes" in shared
    assert "helm" in shared
    assert "guide" not in shared
    assert "project" not in shared
    assert "workflow" not in shared


@pytest.mark.parametrize(
    ("source_name", "source_desc", "neighbor_name", "neighbor_desc", "taxonomy", "expected"),
    [
        (
            "tdd",
            "Test-driven development. Use when building features test-first.",
            "test-driven-development",
            "Use when implementing any feature before writing implementation code.",
            frozenset({"use"}),
            True,
        ),
        (
            "numpy-best-practices",
            "Best practices for NumPy array programming and performance optimization in Python.",
            "python-performance-optimization",
            "Profile and optimize Python code using cProfile and performance best practices.",
            frozenset({"code"}),
            False,
        ),
        (
            "beacon-gateway",
            "Official skill for integrating Beacon Gateway (Flux API) into web applications.",
            "flux-api",
            "Guides usage of the Flux API with the core client SDK.",
            frozenset(),
            False,
        ),
        (
            "beacon-gateway",
            (
                "Official skill for integrating Beacon Gateway "
                "(formerly (v1) Flux API) into web applications."
            ),
            "flux-api",
            "Guides usage of the Flux API with the core client SDK.",
            frozenset(),
            False,
        ),
        (
            "beacon-gateway",
            "Official skill for integrating Beacon Gateway and Flux API into web applications.",
            "flux-api",
            "Guides usage of the Flux API with the core client SDK.",
            frozenset(),
            True,
        ),
    ],
    ids=[
        "acronym-expansion-claim",
        "shared-suffix-different-subject",
        "parenthetical-clarification",
        "nested-parenthetical-clarification",
        "unparenthesized-direct-claim",
    ],
)
def test_claims_neighbor_name_phrase(
    tmp_path: Path,
    source_name: str,
    source_desc: str,
    neighbor_name: str,
    neighbor_desc: str,
    taxonomy: frozenset[str],
    expected: bool,
) -> None:
    """Verify neighbor name phrase detection across acronyms, suffixes, and parentheticals."""
    from reach.lint import _claims_neighbor_name_phrase
    from reach.models import Skill

    source = Skill(name=source_name, description=source_desc, path=tmp_path / source_name)
    neighbor = Skill(name=neighbor_name, description=neighbor_desc, path=tmp_path / neighbor_name)
    assert _claims_neighbor_name_phrase(source, neighbor, taxonomy) is expected


def test_missing_mutual_handoff_ignores_multi_skill_template_cliques(tmp_path: Path) -> None:
    """Suppress k >= 3 template sibling cliques while flagging 1-to-1 peer collisions."""
    skills = [
        (
            "sdk-client-go",
            (
                "Use this skill when building service client integrations with streaming handlers "
                "and diagnosing configuration or runtime problems in Go."
            ),
        ),
        (
            "sdk-client-js",
            (
                "Use this skill when building service client integrations with streaming handlers "
                "and diagnosing configuration or runtime problems in JavaScript."
            ),
        ),
        (
            "sdk-client-python",
            (
                "Use this skill when building service client integrations with streaming handlers "
                "and diagnosing configuration or runtime problems in Python."
            ),
        ),
        (
            "workflow-pipeline-authoring",
            (
                "Author and configure distributed task scheduler DAG definitions, "
                "operators, and orchestration pipelines."
            ),
        ),
        (
            "workflow-pipeline-debugging",
            (
                "Debug and troubleshoot distributed task scheduler DAG definitions, "
                "operators, and orchestration pipelines."
            ),
        ),
    ]
    for name, desc in skills:
        d = tmp_path / name
        d.mkdir()
        (d / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: >\n  {desc}\n---\n# {name}\n",
            encoding="utf-8",
        )

    report = lint_tree(tmp_path)
    handoff_skills = {i.skill for i in report.issues if i.rule == "missing-mutual-handoff"}
    assert handoff_skills == {
        "workflow-pipeline-authoring",
        "workflow-pipeline-debugging",
    }


def test_explain_catalog_budget_overflow_rule() -> None:
    """Verify explain_rule returns definition for catalog-budget-overflow."""
    rule = explain_rule("catalog-budget-overflow")
    assert rule is not None
    assert rule.rule == "catalog-budget-overflow"
    assert rule.default_severity == Severity.WARN


def _populate_overflow_corpus(root: Path, count: int = 5) -> Path:
    """Populate directory with test skills that together exceed small listing budgets."""
    for i in range(count):
        d = root / f"skill-{i}"
        d.mkdir()
        (d / "SKILL.md").write_text(
            f"---\nname: skill-{i}\ndescription: Tool for category {i} operations.\n---\n",
            encoding="utf-8",
        )
    return root


def test_catalog_budget_overflow_detected_when_exceeding_budget(tmp_path: Path) -> None:
    """Verify catalog-budget-overflow emits a single consolidated warning when exceeding budget."""
    corpus = _populate_overflow_corpus(tmp_path, count=8)
    # With a small budget (e.g. 100 chars), not all 8 skills can fit
    report = lint_tree(corpus, config=LintSettings(catalog_budget_chars=100))
    overflow_issues = [i for i in report.issues if i.rule == "catalog-budget-overflow"]
    assert len(overflow_issues) == 1
    issue = overflow_issues[0]
    assert issue.severity == Severity.WARN
    assert issue.skill == "<catalog>"
    assert "exceeds listing budget" in issue.message
    assert "skill description(s) truncated to bare name" in issue.message
    assert "+5 more)" in issue.message


@pytest.mark.parametrize("mode", ["neighborhood", "singleton"])
def test_catalog_budget_overflow_skipped_for_scoped_catalog_modes(
    tmp_path: Path,
    mode: str,
) -> None:
    """Verify catalog-budget-overflow is skipped when catalog.mode is neighborhood or singleton."""
    from reach.config import CatalogSettings, RunConfig
    from reach.models import CatalogMode

    corpus = _populate_overflow_corpus(tmp_path)
    cfg = LintSettings.from_settings({"catalog": {"mode": mode}})
    assert cfg.catalog_budget_chars is None
    report = lint_tree(corpus, config=cfg)
    overflow_issues = [i for i in report.issues if i.rule == "catalog-budget-overflow"]
    assert len(overflow_issues) == 0

    # RunConfig with explicit mode inherits None, while unconfigured RunConfig keeps 30,000
    assert RunConfig().lint.catalog_budget_chars == 30_000
    rc = RunConfig(catalog=CatalogSettings(mode=CatalogMode(mode)))
    assert rc.lint.catalog_budget_chars is None
    assert "catalog_budget_chars" not in rc.lint.model_fields_set
    # Overriding catalog.mode back to ALL restores default budget because lint was not explicit
    rc_all = rc.with_overrides(catalog={"mode": CatalogMode.ALL})
    assert rc_all.lint.catalog_budget_chars == 30_000


def test_catalog_budget_overflow_explicit_lint_budget_overrides_catalog_mode(
    tmp_path: Path,
) -> None:
    """Verify explicit lint.catalog_budget_chars is honored in neighborhood mode."""
    corpus = _populate_overflow_corpus(tmp_path)
    cfg = LintSettings.from_settings(
        {"catalog": {"mode": "neighborhood"}, "lint": {"catalog_budget_chars": 100}}
    )
    assert cfg.catalog_budget_chars == 100
    report = lint_tree(corpus, config=cfg)
    overflow_issues = [i for i in report.issues if i.rule == "catalog-budget-overflow"]
    assert len(overflow_issues) == 1


def test_catalog_budget_overflow_suppressed_when_ignored(tmp_path: Path) -> None:
    """Verify catalog-budget-overflow is omitted when configured to ignore."""
    corpus = _populate_overflow_corpus(tmp_path)
    report = lint_tree(
        corpus,
        config=LintSettings(
            catalog_budget_chars=100,
            rules={"catalog-budget-overflow": Severity.IGNORE},
        ),
    )
    overflow_issues = [i for i in report.issues if i.rule == "catalog-budget-overflow"]
    assert len(overflow_issues) == 0


def test_catalog_budget_overflow_skipped_when_budget_is_none(tmp_path: Path) -> None:
    """Verify catalog-budget-overflow is skipped when catalog_budget_chars is None."""
    corpus = _populate_overflow_corpus(tmp_path)
    cfg = LintSettings.from_settings({"lint": {"catalog_budget_chars": None}})
    assert cfg.catalog_budget_chars is None
    report = lint_tree(corpus, config=cfg)
    overflow_issues = [i for i in report.issues if i.rule == "catalog-budget-overflow"]
    assert len(overflow_issues) == 0


def test_lockfile_clean_when_folder_tree_hash_matches(
    write_skill: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """Verify lockfile-drift accepts npx skills folder tree SHA-256 computedHash."""
    import hashlib
    import json

    skill_dir = write_skill(
        name="folder-hashed-skill",
        description="A skill pinned using npx skills computeSkillFolderHash algorithm.",
        root=tmp_path / ".agents" / "skills",
    )
    helper_file = skill_dir / "references" / "guide.md"
    helper_file.parent.mkdir(parents=True)
    helper_file.write_text("# Reference Guide\n", encoding="utf-8")
    # Ignored directories (.git, node_modules) should not affect folder hash
    git_dir = skill_dir / ".git"
    git_dir.mkdir()
    (git_dir / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")

    manifest = skill_dir / "SKILL.md"
    hasher = hashlib.sha256()
    for rel_posix, file_path in [
        ("references/guide.md", helper_file),
        ("SKILL.md", manifest),
    ]:
        hasher.update(rel_posix.encode("utf-8"))
        hasher.update(file_path.read_bytes())
    folder_hash = hasher.hexdigest()

    lockfile = tmp_path / "skills-lock.json"
    lockfile.write_text(
        json.dumps(
            {
                "version": 1,
                "skills": {"folder-hashed-skill": {"computedHash": folder_hash}},
            }
        ),
        encoding="utf-8",
    )
    report = lint_file(manifest)
    drift_issues = [i for i in report.issues if i.rule == "lockfile-drift"]
    assert drift_issues == []


def test_lockfile_drift_skips_workspace_root_walk_and_symlink_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify root SKILL.md skips workspace root walk and symlink dirs return empty."""
    import json

    from reach import lint as lint_mod

    manifest = tmp_path / "SKILL.md"
    manifest.write_text(
        (
            "---\nname: root-skill\n"
            "description: Standalone manifest alongside skills-lock.json.\n---\n"
        ),
        encoding="utf-8",
    )
    lockfile = tmp_path / "skills-lock.json"
    lockfile.write_text(
        json.dumps(
            {
                "version": 1,
                "skills": {"root-skill": {"computedHash": "stale-hash"}},
            }
        ),
        encoding="utf-8",
    )

    walked_dirs: list[Path] = []
    orig_compute = lint_mod._compute_skill_folder_hash

    def spy_compute(skill_dir: Path) -> str:
        walked_dirs.append(skill_dir)
        return orig_compute(skill_dir)

    monkeypatch.setattr(lint_mod, "_compute_skill_folder_hash", spy_compute)
    report = lint_file(manifest)
    assert any(i.rule == "lockfile-drift" for i in report.issues)
    assert walked_dirs == []

    real_dir = tmp_path / "real-dir"
    real_dir.mkdir()
    symlink_dir = tmp_path / "symlink-dir"
    symlink_dir.symlink_to(real_dir, target_is_directory=True)
    assert orig_compute(symlink_dir) == ""


def test_empty_skill_directory_emits_info(
    write_skill: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """Verify lint_tree emits an INFO-level empty-skill-directory issue for empty skill folders."""
    write_skill(
        name="valid-skill",
        description="A valid skill sibling next to an empty skill directory.",
    )
    empty_dir = tmp_path / "empty-skill"
    empty_dir.mkdir()

    report = lint_tree(tmp_path)
    empty_issues = [i for i in report.issues if i.rule == "empty-skill-directory"]
    assert len(empty_issues) == 1
    assert empty_issues[0].skill == "empty-skill"
    assert empty_issues[0].severity == Severity.INFO
    assert report.infos == (empty_issues[0],)


@pytest.mark.parametrize(
    ("rules_override", "expected_severity"),
    [
        ({}, Severity.INFO),
        ({"missing-mutual-handoff": Severity.WARN}, Severity.WARN),
        ({"missing-mutual-handoff": Severity.IGNORE}, None),
    ],
    ids=["default-info", "override-warn", "override-ignore"],
)
def test_missing_mutual_handoff_unacknowledged_overlap_severity(
    write_skill: Callable[..., Path],
    tmp_path: Path,
    rules_override: dict[str, Severity],
    expected_severity: Severity | None,
) -> None:
    """Verify unacknowledged peer overlap defaults to INFO and honors rule overrides."""
    from reach.lint import _create_issue

    cfg = LintSettings(rules=rules_override)
    direct_issue = _create_issue(
        "missing-mutual-handoff",
        "db-cost-optimizer",
        tmp_path / "SKILL.md",
        "Overlap diagnostic",
        cfg,
        default_severity=Severity.INFO,
    )
    if expected_severity is None:
        assert direct_issue is None
    else:
        assert direct_issue is not None
        assert direct_issue.severity == expected_severity

    write_skill(
        name="db-cost-optimizer",
        description=(
            "Analyzes PostgreSQL worker utilization, query execution bottlenecks, and "
            "pg_stat_statements telemetry. Use when diagnosing slow SQL queries, "
            "worker starvation, high query costs, or join performance bottlenecks. "
            "Don't use for generic database administration (use `db-basics`)."
        ),
    )
    write_skill(
        name="db-observability",
        description=(
            "Monitors PostgreSQL operational telemetry, worker utilization, and query "
            "execution bottlenecks using pg_stat_statements. Use when investigating worker "
            "usage trends, query concurrency, or capacity planning. "
            "Don't use for generic database administration (use `db-basics`)."
        ),
    )
    write_skill(
        name="db-basics",
        description=(
            "Creates and administers PostgreSQL schemas and tables. "
            "Don't use for query cost optimization (use `db-cost-optimizer`) "
            "or operational telemetry (use `db-observability`)."
        ),
    )

    report = lint_tree(tmp_path, config=cfg)
    mutual_issues = [i for i in report.issues if i.rule == "missing-mutual-handoff"]
    if expected_severity is None:
        assert mutual_issues == []
    else:
        assert len(mutual_issues) == 2
        assert all(i.severity == expected_severity for i in mutual_issues)


@pytest.mark.parametrize(
    "greedy_desc",
    [
        "Assist with any coding task and debug issues.",
        "A tool to help with any programming problem.",
        "Manage files and run commands in the terminal.",
        "General-purpose developer assistant.",
        "Universal assistant for your workflows.",
        "All-in-one helper for software engineering.",
        "Handle any request given by the user.",
        (
            "Use this general-purpose assistant when implementing any feature "
            "or fixing bugs in the codebase."
        ),
    ],
)
def test_unbounded_attractor_flags_greedy_descriptions(tmp_path: Path, greedy_desc: str) -> None:
    """Verify unbounded-attractor rule flags catch-all phrases in descriptions."""
    skill_dir = tmp_path / "greedy-skill"
    skill_dir.mkdir(parents=True)
    skill_file = skill_dir / "SKILL.md"
    skill_file.write_text(
        f"---\nname: greedy-skill\ndescription: {greedy_desc}\n---\n# Body\n",
        encoding="utf-8",
    )

    report = lint_file(skill_file, LintSettings())
    attractor_issues = [i for i in report.issues if i.rule == "unbounded-attractor"]
    assert len(attractor_issues) == 1
    assert attractor_issues[0].severity == Severity.WARN
    assert "greedy-skill" in attractor_issues[0].skill


@pytest.mark.parametrize(
    "bounded_desc",
    [
        "Deploy containerized microservices to Google Cloud Run with gcloud.",
        "Parse and validate JSON schemas against Draft 7 specifications.",
        "Generate Terraform templates for Google Kubernetes Engine clusters.",
        "Format Python code using the Black code formatter.",
    ],
)
def test_unbounded_attractor_permits_bounded_descriptions(
    tmp_path: Path, bounded_desc: str
) -> None:
    """Verify unbounded-attractor rule passes domain-specific bounded descriptions."""
    skill_dir = tmp_path / "bounded-skill"
    skill_dir.mkdir(parents=True)
    skill_file = skill_dir / "SKILL.md"
    skill_file.write_text(
        f"---\nname: bounded-skill\ndescription: {bounded_desc}\n---\n# Body\n",
        encoding="utf-8",
    )

    report = lint_file(skill_file, LintSettings())
    attractor_issues = [i for i in report.issues if i.rule == "unbounded-attractor"]
    assert len(attractor_issues) == 0
