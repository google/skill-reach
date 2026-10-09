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

"""Verify documentation configuration, CLI subcommand sync, and navigation integrity."""

from __future__ import annotations

import ast
import functools
import importlib
import inspect
import pkgutil
import re
import shlex
import tomllib
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, NamedTuple, get_origin

import markdown
import pytest
from mkdocs.config import load_config
from pydantic import BaseModel

if TYPE_CHECKING:
    from collections.abc import Iterator

    from mkdocs.config.defaults import MkDocsConfig

import reach
import reach.artifact
import reach.cli
import reach.models
from reach.cli.app import _verbs, app
from reach.config import (
    BUILTIN_AGENT_DEFAULT_MODELS,
    DEFAULT_CLAUDE_MODEL,
    DEFAULT_GEMINI_MODEL,
    KNOWN_CLIENT_SKILLS_DIRS,
    AgentProfile,
    CatalogSettings,
    CheckSettings,
    DiffSettings,
    DiscoverySettings,
    GeneralSettings,
    LintSettings,
    OptimizeSettings,
    OverlapSettings,
    PlanSettings,
    QuerySettings,
    RegistrySettings,
    RetrievalSettings,
    RunConfig,
    RuntimeSettings,
    StudySettings,
)
from reach.lint import RULES
from reach.models import Query
from reach.runtime import (
    FAKE_AGENT,
    AgentOptions,
    CliOptions,
    VertexOptions,
    build_runtime,
    known_agents,
    options_model,
)
from reach.runtime.profiles import ModelProfile

_ROOT = Path(__file__).resolve().parent.parent
_DOCS_DIR = _ROOT / "docs"
_MKDOCS_FILE = _ROOT / "mkdocs.yml"


@pytest.fixture(scope="module")
def mkdocs_config() -> MkDocsConfig:
    """Provide the parsed mkdocs.yml configuration for documentation tests."""
    assert _MKDOCS_FILE.exists(), f"mkdocs.yml not found at {_MKDOCS_FILE}"
    return load_config(str(_MKDOCS_FILE))


def _flatten_nav(nav_entries: list[Any]) -> list[str]:
    """Recursively extract all target document paths from the nav tree."""
    paths: list[str] = []
    for entry in nav_entries:
        if isinstance(entry, dict):
            for val in entry.values():
                if isinstance(val, str):
                    paths.append(val)
                elif isinstance(val, list):
                    paths.extend(_flatten_nav(val))
        elif isinstance(entry, str):
            paths.append(entry)
    return paths


@pytest.fixture(scope="module")
def nav_paths(mkdocs_config: MkDocsConfig) -> list[str]:
    """Provide flattened list of document paths referenced in mkdocs.yml nav."""
    return _flatten_nav(mkdocs_config.get("nav", []))


@pytest.fixture(scope="module")
def all_markdown_files() -> tuple[Path, ...]:
    """Provide all documentation and root markdown files."""
    return (*sorted(_DOCS_DIR.rglob("*.md")), _ROOT / "README.md", _ROOT / "CONTRIBUTING.md")


@pytest.fixture(scope="module")
def markdown_contents(all_markdown_files: tuple[Path, ...]) -> dict[Path, str]:
    """Provide cached UTF-8 text content for every markdown file."""
    return {path: path.read_text(encoding="utf-8") for path in all_markdown_files}


@pytest.fixture(scope="module")
def config_doc_text(markdown_contents: dict[Path, str]) -> str:
    """Provide text content of docs/configuration.md."""
    return markdown_contents[_DOCS_DIR / "configuration.md"]


@pytest.fixture(scope="module")
def readme_text(markdown_contents: dict[Path, str]) -> str:
    """Provide text content of README.md."""
    return markdown_contents[_ROOT / "README.md"]


@pytest.fixture(scope="module")
def reach_source_text() -> str:
    """Provide concatenated source text for all reach Python modules."""
    src_files = sorted((_ROOT / "src" / "reach").rglob("*.py"))
    return "\n".join(f.read_text(encoding="utf-8") for f in src_files)


def extract_fenced_blocks(markdown_text: str, *langs: str) -> list[str]:
    """Extract non-empty fenced code blocks matching the given language tags."""
    lang_group = "|".join(re.escape(lang) for lang in langs) if langs else ""
    pattern = re.compile(rf"```(?:{lang_group})\s*\n(.*?)\n```", re.DOTALL)
    return [m.group(1).strip() for m in pattern.finditer(markdown_text) if m.group(1).strip()]


def extract_section(markdown_text: str, heading: str, *, level: int = 2) -> str:
    """Extract markdown section body under a specific heading up to the next peer heading."""
    hashes = "#" * level
    min_level = min(2, level)
    pattern = re.compile(
        rf"^{hashes}\s+{re.escape(heading)}\s*$\n(.*?)(?=^#{{{min_level},{level}}}\s+|\Z)",
        re.MULTILINE | re.DOTALL,
    )
    match = pattern.search(markdown_text)
    return match.group(1) if match else ""


def _iter_markdown_table_rows(markdown_section: str) -> Iterator[list[str]]:
    """Yield split column lists for non-delimiter markdown table rows."""
    for raw_line in markdown_section.splitlines():
        line = raw_line.strip()
        if not line.startswith("|") or line.startswith(("| :---", "| ---")):
            continue
        yield line.split("|")


def extract_table_column_entries(
    markdown_section: str,
    column_index: int = 1,
    pattern: str = r"`([a-zA-Z0-9_.-]+)`",
) -> set[str]:
    """Extract backtick-enclosed identifiers from a column of a markdown table."""
    entries: set[str] = set()
    for cols in _iter_markdown_table_rows(markdown_section):
        if len(cols) > column_index + 1:
            entries.update(re.findall(pattern, cols[column_index]))
    return entries


def extract_table_row_flag_groups(markdown_section: str) -> list[set[str]]:
    """Extract sets of CLI flags grouped per row from the first column of markdown tables."""
    groups: list[set[str]] = []
    for cols in _iter_markdown_table_rows(markdown_section):
        if len(cols) > 2:
            flags = set(re.findall(r"`(-[a-zA-Z]|--[a-zA-Z0-9-]+)`", cols[1]))
            if flags:
                groups.append(flags)
    return groups


@functools.cache
def _command_arg_groups(verb: str, subverb: str | None = None) -> tuple[frozenset[str], ...]:
    """Return tuple of alias sets for each argument on a registered CLI command or subcommand."""
    cmd = app[verb][subverb] if subverb else app[verb]
    groups: list[frozenset[str]] = [frozenset({"--help", "-h"}), frozenset({"--version"})]
    for arg in cmd.assemble_argument_collection():
        names = frozenset(name for name in arg.names if name.startswith("-"))
        if names:
            groups.append(names)
    return tuple(groups)


@functools.cache
def _command_valid_flags(verb: str, subverb: str | None = None) -> frozenset[str]:
    """Assemble all valid option flags for a registered CLI command or subcommand."""
    return frozenset(flag for group in _command_arg_groups(verb, subverb) for flag in group)


# ==============================================================================
# 1. MkDocs & Navigation Parity
# ==============================================================================


def test_mkdocs_config_is_valid(mkdocs_config: MkDocsConfig) -> None:
    """Verify mkdocs.yml can be loaded and contains expected top-level keys."""
    assert mkdocs_config.get("site_name") == "skill-reach"
    assert "nav" in mkdocs_config
    assert "theme" in mkdocs_config
    assert mkdocs_config["theme"].name == "material"


def test_mkdocs_nav_files_match_docs_on_disk(nav_paths: list[str]) -> None:
    """Verify 1-to-1 parity between files in mkdocs.yml nav and markdown files in docs/."""
    nav_set = set(nav_paths)
    missing_on_disk = [rel for rel in nav_paths if not (_DOCS_DIR / rel).is_file()]
    assert not missing_on_disk, (
        f"Files referenced in mkdocs.yml nav do not exist: {missing_on_disk}"
    )

    all_docs_on_disk = {p.relative_to(_DOCS_DIR).as_posix() for p in _DOCS_DIR.rglob("*.md")}
    assert nav_set == all_docs_on_disk


def test_cli_doc_pages_and_nav_match_registered_verbs(nav_paths: list[str]) -> None:
    """Verify docs/cli/*.md pages and mkdocs.yml CLI nav entries match registered CLI verbs."""
    expected_verbs = set(_verbs())
    cli_doc_files = {p.stem for p in (_DOCS_DIR / "cli").glob("*.md") if p.name != "index.md"}
    assert cli_doc_files == expected_verbs

    cli_nav_entries = {p for p in nav_paths if p.startswith("cli/")}
    expected_nav = {f"cli/{verb}.md" for verb in expected_verbs} | {"cli/index.md"}
    assert cli_nav_entries == expected_nav


def test_cli_index_overview_matches_registered_verbs(markdown_contents: dict[Path, str]) -> None:
    """Verify docs/cli/index.md Command Overview table matches registered CLI subcommands."""
    cli_index_text = markdown_contents[_DOCS_DIR / "cli" / "index.md"]
    overview_verbs = set(re.findall(r"\[`reach ([a-z0-9-]+)`\]", cli_index_text))
    assert overview_verbs == set(_verbs())


def test_readme_commands_table_matches_registered_verbs(readme_text: str) -> None:
    """Verify README.md command tables match registered CLI subcommands."""
    commands_section = extract_section(readme_text, "Commands", level=2)
    doc_verbs = extract_table_column_entries(
        commands_section,
        column_index=1,
        pattern=r"`([a-z0-9-]+)`",
    )
    assert doc_verbs == set(_verbs())


# ==============================================================================
# 2. CLI Options, Subcommands & Short-Flag Pairing
# ==============================================================================

#: Deliberately unlisted pass-through flags per command
UNDOCUMENTED_CLI_FLAGS: dict[str, set[str]] = {
    "query": {
        "--catalog-size",
        "--draft-only",
        "--early-exit",
        "--effort",
        "--max-turns",
        "--mode",
        "--model",
        "--opt",
        "--partial",
        "--rescope",
        "--rivals",
        "--run-dir",
        "--seed",
        "--tag",
        "--target",
        "--timeout",
        "--workdir",
    },
}


def _assert_table_flag_groups_valid(
    section_text: str,
    verb: str,
    subverb: str | None = None,
) -> None:
    """Assert every option table row in section_text contains valid, co-aliased flags."""
    target_label = f"reach {verb} {subverb}" if subverb else f"reach {verb}"
    valid_flags = _command_valid_flags(verb, subverb)
    arg_groups = _command_arg_groups(verb, subverb)

    for row_flags in extract_table_row_flag_groups(section_text):
        unknown = row_flags - valid_flags
        assert not unknown, (
            f"Unrecognized CLI flag(s) {sorted(unknown)} documented for {target_label!r}"
        )
        if len(row_flags) > 1:
            assert any(row_flags <= group for group in arg_groups), (
                f"Flags {sorted(row_flags)} in {target_label!r} table row do not alias "
                "the same CLI parameter!"
            )


@pytest.mark.parametrize("verb", _verbs())
def test_documented_cli_options_and_subcommands_valid(
    verb: str,
    markdown_contents: dict[Path, str],
) -> None:
    """Verify documented CLI options, short aliases, and subcommands match Cyclopts definitions."""
    content = markdown_contents[_DOCS_DIR / "cli" / f"{verb}.md"]

    options_section = extract_section(content, "Options", level=2)
    if options_section:
        _assert_table_flag_groups_valid(options_section, verb)

    subcommands_section = extract_section(content, "Subcommands", level=2)
    if subcommands_section:
        for match in re.finditer(
            rf"^###\s+`reach\s+{re.escape(verb)}\s+([a-z0-9-]+)`\s*$\n(.*?)(?=^###\s+|\Z)",
            subcommands_section,
            re.MULTILINE | re.DOTALL,
        ):
            subverb, sub_body = match.group(1), match.group(2)
            _assert_table_flag_groups_valid(sub_body, verb, subverb)


@pytest.mark.parametrize("verb", _verbs())
def test_all_cli_options_are_documented(verb: str, markdown_contents: dict[Path, str]) -> None:
    """Verify that every CLI option registered on a command is documented in docs/cli/{verb}.md."""
    content = markdown_contents[_DOCS_DIR / "cli" / f"{verb}.md"]
    doc_flags = set(re.findall(r"`(--[a-zA-Z0-9-]+)`", content))
    doc_positionals = set(re.findall(r"`\[?([A-Z0-9_-]+)\]?`", content))

    cmd = app[verb]
    cli_flags = {
        name
        for arg in cmd.assemble_argument_collection()
        for name in (set(arg.names) - set(arg.negatives))
        if name.startswith("--") and name not in {"--help", "--version"}
    }

    covered = {
        f
        for f in cli_flags
        if f.removeprefix("--").replace("-", "_").upper() in doc_positionals
        or (f.startswith("--no-") and f.replace("--no-", "--") in doc_flags)
    }

    missing = cli_flags - doc_flags - covered - UNDOCUMENTED_CLI_FLAGS.get(verb, set())
    assert not missing, f"Undocumented CLI options on 'reach {verb}': {sorted(missing)}"


@pytest.mark.parametrize("verb", _verbs())
def test_cli_docs_have_synopsis_and_headings(
    verb: str,
    markdown_contents: dict[Path, str],
) -> None:
    """Verify CLI subcommand document has a proper heading and Synopsis section."""
    content = markdown_contents[_DOCS_DIR / "cli" / f"{verb}.md"]
    first_line = content.splitlines()[0].strip() if content.splitlines() else ""
    assert first_line in (f"# `reach {verb}`", f"# reach {verb}"), (
        f"Expected top-level heading '# `reach {verb}`' in {verb}.md, got {first_line!r}"
    )
    assert "## Synopsis" in content, f"Missing '## Synopsis' section in {verb}.md"


def test_cli_key_defaults_documented(markdown_contents: dict[Path, str]) -> None:
    """Verify critical CLI defaults (generator model, check --strict, clean --all) match code."""
    for rel in ("cli/query.md", "cli/eval.md"):
        assert f"`{DEFAULT_GEMINI_MODEL}`" in markdown_contents[_DOCS_DIR / rel]

    check_doc = markdown_contents[_DOCS_DIR / "cli" / "check.md"]
    assert re.search(r"`--strict` / `--no-strict`\s*\|\s*Flag\s*\|\s*`true`", check_doc)

    clean_doc = markdown_contents[_DOCS_DIR / "cli" / "clean.md"]
    assert ".reach/queries" in clean_doc or "queries.json" in clean_doc

    query_doc = markdown_contents[_DOCS_DIR / "cli" / "query.md"]
    assert "queries.citations.json" not in query_doc


# ==============================================================================
# 3. Configuration, TOML Blocks, Lint Rules & Schema Tables
# ==============================================================================


@pytest.mark.parametrize("agent", [a for a in known_agents() if a != FAKE_AGENT])
def test_all_public_agents_documented(
    agent: str,
    readme_text: str,
    config_doc_text: str,
    markdown_contents: dict[Path, str],
) -> None:
    """Verify supported agent runtime is documented in README, index, and configuration docs."""
    index_text = markdown_contents[_DOCS_DIR / "index.md"]
    assert agent in readme_text, f"Agent {agent!r} missing from README.md"
    assert agent in index_text, f"Agent {agent!r} missing from docs/index.md"
    assert agent in config_doc_text, f"Agent {agent!r} missing from docs/configuration.md"


@pytest.mark.parametrize("client", sorted(KNOWN_CLIENT_SKILLS_DIRS))
def test_discovery_client_profiles_documented(
    client: str,
    readme_text: str,
    config_doc_text: str,
) -> None:
    """Verify client discovery profile is documented in discovery precedence docs."""
    assert client.lower() in readme_text.lower(), f"Client {client!r} missing from README.md"
    assert client.lower() in config_doc_text.lower(), (
        f"Client {client!r} missing from docs/configuration.md"
    )


def test_lint_rules_match_docs(
    markdown_contents: dict[Path, str],
    config_doc_text: str,
) -> None:
    """Verify 1-to-1 parity between RULES registry and rules in lint.md & configuration.md."""
    lint_doc_text = markdown_contents[_DOCS_DIR / "cli" / "lint.md"]
    rules_section = extract_section(lint_doc_text, "Built-in Lint Rules", level=2)
    assert rules_section, "Missing '## Built-in Lint Rules' section in docs/cli/lint.md"

    doc_rules = extract_table_column_entries(
        rules_section, column_index=1, pattern=r"`([a-z0-9-]+)`"
    )
    expected_rules = set(RULES.keys())
    assert doc_rules == expected_rules

    missing_in_config = {rule_id for rule_id in expected_rules if rule_id not in config_doc_text}
    assert not missing_in_config, (
        f"Lint rules missing from docs/configuration.md: {sorted(missing_in_config)}"
    )


def test_explain_examples_reference_valid_rules(markdown_contents: dict[Path, str]) -> None:
    """Verify all --explain CLI examples in documentation reference registered rules."""
    explain_re = re.compile(r"--explain\s+([a-z0-9-]+)")
    for md_file, content in markdown_contents.items():
        for match in explain_re.finditer(content):
            rule_id = match.group(1)
            assert rule_id in RULES, (
                f"Invalid rule {rule_id!r} in --explain example at {md_file.relative_to(_ROOT)}"
            )


@pytest.mark.parametrize(
    ("agent", "expected_model"),
    sorted(BUILTIN_AGENT_DEFAULT_MODELS.items()),
)
def test_builtin_agent_default_models_documented(
    agent: str,
    expected_model: str,
    config_doc_text: str,
) -> None:
    """Verify default model for builtin agent is documented in docs/configuration.md."""
    escaped_agent = re.escape(agent)
    escaped_model = re.escape(expected_model)
    pattern = rf"\[agents\.{escaped_agent}\][\s\S]*?default_model\s*=\s*\"{escaped_model}\""
    assert re.search(pattern, config_doc_text), (
        f"Agent {agent!r} default_model {expected_model!r} not in docs/configuration.md"
    )


SECTION_MODEL_MAP: dict[str, type[BaseModel]] = {
    "agents": AgentProfile,
    "catalog": CatalogSettings,
    "check": CheckSettings,
    "diff": DiffSettings,
    "discovery": DiscoverySettings,
    "general": GeneralSettings,
    "lint": LintSettings,
    "models": ModelProfile,
    "optimize": OptimizeSettings,
    "overlap": OverlapSettings,
    "plan": PlanSettings,
    "query": QuerySettings,
    "registry": RegistrySettings,
    "retrieval": RetrievalSettings,
    "runtime": RuntimeSettings,
    "study": StudySettings,
}

UNDOCUMENTED_CONFIG_FIELDS: dict[str, set[str]] = {
    "runtime": {"agent", "allowed_tools", "options"},
    "lint": {"rules", "similarity_threshold"},
    "registry": {"registry", "fresh", "no_cache"},
}


def _extract_section_block(config_doc_text: str, section_name: str) -> str:
    """Extract markdown text block for a specific section from docs/configuration.md."""
    sections_ref = extract_section(config_doc_text, "Sections & Options Reference", level=2)
    section_blocks = re.split(r"\n(?=### `?\[)", sections_ref)
    for block in section_blocks:
        header_match = re.search(r"### `?\[([a-zA-Z0-9_.<>-]+)\]`?", block)
        if header_match:
            raw_sec = header_match.group(1).split(".")[0]
            if header_match.group(1) == section_name or raw_sec == section_name:
                return str(block)
    return ""


def test_documented_configuration_sections_valid(config_doc_text: str) -> None:
    """Verify section headings in docs/configuration.md correspond to valid models."""
    sections_ref = extract_section(config_doc_text, "Sections & Options Reference", level=2)
    section_matches = re.findall(r"### `?\[([a-zA-Z0-9_.<>-]+)\]`?", sections_ref)
    allowed_sections = set(SECTION_MODEL_MAP.keys()) | {"lint.rules"}

    stale_sections = [
        sec
        for sec in section_matches
        if sec not in allowed_sections and sec.split(".")[0] not in allowed_sections
    ]
    assert not stale_sections, (
        f"Unrecognized or stale configuration sections in docs/configuration.md: {stale_sections}"
    )


@pytest.mark.parametrize(("section_name", "model_cls"), sorted(SECTION_MODEL_MAP.items()))
def test_configuration_section_fields_match_models(
    section_name: str,
    model_cls: type[BaseModel],
    config_doc_text: str,
) -> None:
    """Verify 1-to-1 parity between configuration.md section tables and Pydantic model fields."""
    section_block = _extract_section_block(config_doc_text, section_name)
    assert section_block, f"Missing section block for [{section_name}] in configuration.md"
    doc_keys = {
        k for k in extract_table_column_entries(section_block) if not k.startswith("rules.")
    }
    expected_keys = set(model_cls.model_fields.keys()) - UNDOCUMENTED_CONFIG_FIELDS.get(
        section_name,
        set(),
    )
    assert doc_keys == expected_keys, f"Field mismatch in [{section_name}] table"


def test_driver_option_tables_match_runtime_options_models(config_doc_text: str) -> None:
    """Verify driver option tables in configuration.md match AgentRuntime options models."""
    driver_section = extract_section(
        config_doc_text, "Driver Options (`[runtime.options]`)", level=2
    )
    assert driver_section, (
        "Missing '## Driver Options (`[runtime.options]`)' section in configuration.md"
    )

    common_block = extract_section(
        driver_section,
        "Common Runtime Options (`AgentOptions`, `CliOptions`, `VertexOptions`)",
        level=3,
    )
    assert common_block, "Missing '### Common Runtime Options' subsection in configuration.md"
    expected_common_keys = (
        set(AgentOptions.model_fields.keys())
        | set(CliOptions.model_fields.keys())
        | set(VertexOptions.model_fields.keys())
    )
    assert extract_table_column_entries(common_block) == expected_common_keys

    public_agents = {a for a in known_agents() if a != FAKE_AGENT}
    for agent_name in sorted(public_agents):
        opts_cls = options_model(agent_name)
        assert opts_cls is not None

        # Match subsection heading containing the agent_name
        pattern = re.compile(
            rf"^###\s+[^\n]*`{re.escape(agent_name)}`[^\n]*$\n(.*?)(?=^###\s+|\Z)",
            re.MULTILINE | re.DOTALL,
        )
        match = pattern.search(driver_section)
        assert match is not None, f"Missing driver options subsection for {agent_name!r}"
        sub_block = match.group(1)
        doc_keys = extract_table_column_entries(sub_block)
        assert doc_keys, (
            f"Driver options subsection for {agent_name!r} has no documented option table rows"
        )
        valid_keys = set(opts_cls.model_fields.keys())
        stale_keys = doc_keys - valid_keys
        assert not stale_keys, (
            f"Unrecognized driver options in docs/configuration.md for {agent_name!r}: "
            f"{sorted(stale_keys)}"
        )


def test_query_schema_table_matches_query_model(markdown_contents: dict[Path, str]) -> None:
    """Verify Query Schema table in docs/cli/query.md matches Query model fields."""
    query_doc = markdown_contents[_DOCS_DIR / "cli" / "query.md"]
    schema_section = extract_section(query_doc, "Query Schema (`Query`)", level=2)
    assert schema_section, "Missing '## Query Schema (`Query`)' section in docs/cli/query.md"
    # Slice before ### Provenance subsection
    schema_table_part = schema_section.split("\n### ", maxsplit=1)[0]
    doc_fields = extract_table_column_entries(schema_table_part)
    expected_fields = set(Query.model_fields.keys())
    assert doc_fields == expected_fields


def test_markdown_toml_blocks_and_example_toml_validate(
    markdown_contents: dict[Path, str],
    config_doc_text: str,
) -> None:
    """Verify TOML blocks in docs and reach.example.toml pass strict RunConfig validation."""
    public_agents = set(known_agents()) - {FAKE_AGENT}
    bundled_toml = tomllib.loads(
        (_ROOT / "src" / "reach" / "reach.toml").read_text(encoding="utf-8")
    )
    bundled_models = set(bundled_toml.get("models", {}).keys())

    # 1. Validate all fenced ```toml blocks across all markdown files
    for md_file, content in markdown_contents.items():
        rel_path = md_file.relative_to(_ROOT)
        for i, block in enumerate(extract_fenced_blocks(content, "toml"), 1):
            try:
                parsed = tomllib.loads(block)
            except tomllib.TOMLDecodeError as err:
                pytest.fail(f"Invalid TOML syntax in block #{i} of {rel_path}: {err}")
            try:
                cfg = RunConfig.model_validate(parsed)
                if cfg.runtime.options:
                    build_runtime(cfg.runtime)
            except Exception as err:  # noqa: BLE001
                pytest.fail(f"TOML block #{i} in {rel_path} failed RunConfig validation: {err}")
            if "agents" in parsed and isinstance(parsed["agents"], dict):
                stale_agents = set(parsed["agents"].keys()) - public_agents
                assert not stale_agents, (
                    f"Unrecognized agent profile(s) {sorted(stale_agents)} in {rel_path}"
                )

    # 2. Validate reach.example.toml parses and has zero unintended side-effecting overrides
    example_path = _ROOT / "reach.example.toml"
    example_text = example_path.read_text(encoding="utf-8")
    example_cfg = RunConfig.from_toml(example_path)
    default_cfg = RunConfig()
    normalized_example = example_cfg.model_copy(
        update={"lint": example_cfg.lint.model_copy(update={"rules": {}})}
    )
    assert normalized_example == default_cfg, (
        "reach.example.toml diverges from default RunConfig()! Ensure uncommented keys do not "
        "trigger side-effecting overrides (such as setting lint.catalog_budget_chars=None)."
    )
    assert example_cfg.plan.resolve_sweep_attempts() == default_cfg.plan.resolve_sweep_attempts()
    assert example_cfg.plan.resolve_eval_attempts(
        "antigravity-cli", quick=True
    ) == default_cfg.plan.resolve_eval_attempts("antigravity-cli", quick=True)

    # 3. Verify reach.example.toml documents all public schema keys (active or commented)
    mentioned_keys = set(
        re.findall(r"^\s*(?:#\s*)?([a-zA-Z0-9_-]+)\s*=", example_text, re.MULTILINE)
    )
    for sec_name, model_cls in SECTION_MODEL_MAP.items():
        expected_sec_keys = set(model_cls.model_fields.keys()) - UNDOCUMENTED_CONFIG_FIELDS.get(
            sec_name,
            set(),
        )
        missing_sec_keys = expected_sec_keys - mentioned_keys
        assert not missing_sec_keys, (
            f"reach.example.toml is missing documented [{sec_name}] keys: "
            f"{sorted(missing_sec_keys)}"
        )

    # 4. Verify commented model profiles in reach.example.toml and configuration.md
    example_models = set(re.findall(r"\[models\.([a-zA-Z0-9_-]+)\]", example_text))
    assert example_models <= bundled_models
    for default_model in (DEFAULT_GEMINI_MODEL, DEFAULT_CLAUDE_MODEL):
        model_key = default_model.replace(".", "-")
        assert f"[models.{model_key}]" in config_doc_text


def test_reach_env_vars_match_source(
    config_doc_text: str,
    reach_source_text: str,
) -> None:
    """Verify 1-to-1 parity between REACH_* env variables in configuration.md and source code."""
    env_section = extract_section(config_doc_text, "Environment Variables", level=2)
    assert env_section, "Missing '## Environment Variables' section in docs/configuration.md"

    doc_vars = set(re.findall(r"`(REACH_[A-Z0-9_]+)`", env_section))
    code_vars = set(
        re.findall(
            r'os\.(?:environ(?:\.get)?|getenv)\(\s*["\'](REACH_[A-Z0-9_]+)["\']',
            reach_source_text,
        )
    )
    assert doc_vars == code_vars


# ==============================================================================
# 4. Python API Reference & Signature Cross-Reference Integrity
# ==============================================================================

INTERNAL_MODULES: frozenset[str] = frozenset(
    {
        "reach.attribution",
        "reach.cli",
        "reach.difficulty",
        "reach.discovery",
        "reach.exchange",
        "reach.generate",
        "reach.leak",
        "reach.rendering",
        "reach.report",
        "reach.review",
        "reach.rewrite",
        "reach.static",
        "reach.view",
        "reach.views",
    },
)

PUBLIC_API_MODULES: tuple[str, ...] = tuple(
    sorted(
        {
            f"reach.{info.name}"
            for info in pkgutil.iter_modules(reach.__path__)
            if not info.name.startswith("_")
        }
        - INTERNAL_MODULES,
        key=str.casefold,
    ),
)


class ParsedApiDoc(NamedTuple):
    """Represent parsed mkdocstrings declaration and member list from an API doc page."""

    module_name: str
    members: list[str]


def _parse_api_doc_members(md_file: Path, content: str) -> ParsedApiDoc:
    """Extract the documented module name and declared members list from an API doc page."""
    lines = content.splitlines()
    mod_name: str | None = None
    members: list[str] = []
    in_members = False

    for line in lines:
        if line.startswith("::: "):
            mod_name = line[4:].strip()
        elif "options:" in line:
            assert line.startswith("    options:"), (
                f"Invalid indentation for 'options:' in {md_file.name}: expected 4 spaces"
            )
        elif "members:" in line:
            assert line.startswith("      members:"), (
                f"Invalid indentation for 'members:' in {md_file.name}: expected 6 spaces"
            )
            in_members = True
        elif in_members:
            if line.strip().startswith("- "):
                assert line.startswith("        - "), (
                    f"Invalid indentation for member in {md_file.name}: expected 8 spaces"
                )
                members.append(line.strip()[2:].strip())
            elif not line.startswith(" ") and line.strip():
                in_members = False

    assert mod_name is not None, f"No module declaration found in {md_file}"
    return ParsedApiDoc(module_name=mod_name, members=members)


@pytest.fixture(scope="module")
def parsed_api_docs(markdown_contents: dict[Path, str]) -> dict[str, ParsedApiDoc]:
    """Provide parsed API doc definitions keyed by module short name."""
    api_docs_dir = _DOCS_DIR / "api"
    return {
        md_file.stem: _parse_api_doc_members(md_file, markdown_contents[md_file])
        for md_file in sorted(api_docs_dir.glob("*.md"))
        if md_file.name != "index.md"
    }


def test_all_repo_modules_and_api_docs_parity(
    nav_paths: list[str],
    markdown_contents: dict[Path, str],
) -> None:
    """Verify all modules in src/reach are classified and public modules match docs/api/ & nav."""
    all_discovered = {
        f"reach.{info.name}"
        for info in pkgutil.iter_modules(reach.__path__)
        if not info.name.startswith("_")
    }
    expected_public = set(PUBLIC_API_MODULES)
    assert all_discovered == expected_public | INTERNAL_MODULES

    doc_modules = {
        f"reach.{p.stem}" for p in (_DOCS_DIR / "api").glob("*.md") if p.name != "index.md"
    }
    assert doc_modules == expected_public

    api_index_text = markdown_contents[_DOCS_DIR / "api" / "index.md"]
    index_modules = {
        f"reach.{stem}" for stem in re.findall(r"\[`reach\.([a-z0-9_-]+)`\]", api_index_text)
    }
    assert index_modules == expected_public

    nav_set = set(nav_paths)
    for mod_name in PUBLIC_API_MODULES:
        short = mod_name.rsplit(".", maxsplit=1)[-1]
        assert f"api/{short}.md" in nav_set
        assert f"{short}.md" in api_index_text


@pytest.mark.parametrize("mod_name", PUBLIC_API_MODULES)
def test_documented_api_members_match_module_exports(
    mod_name: str,
    parsed_api_docs: dict[str, ParsedApiDoc],
) -> None:
    """Verify 1-to-1 parity between docs/api/{mod}.md members and module __all__."""
    short_name = mod_name.rsplit(".", maxsplit=1)[-1]
    parsed = parsed_api_docs[short_name]
    mod = importlib.import_module(mod_name)

    for member in parsed.members:
        assert hasattr(mod, member), f"Member {member!r} in {short_name}.md missing from {mod_name}"

    if hasattr(mod, "__all__"):
        private_exports = [s for s in mod.__all__ if s.startswith("_")]
        assert not private_exports, f"Private symbol(s) {private_exports} in {mod_name}.__all__"
        for s in mod.__all__:
            assert hasattr(mod, s), f"{mod_name}.__all__ symbol {s!r} does not exist on module"
        expected_members = set(mod.__all__)
    else:
        expected_members = {
            name
            for name, obj in inspect.getmembers(mod)
            if not name.startswith("_")
            and getattr(obj, "__module__", None) == mod_name
            and (inspect.isclass(obj) or inspect.isfunction(obj))
        }

    assert set(parsed.members) == expected_members


def test_public_api_modules_and_members_are_alphabetized(
    mkdocs_config: MkDocsConfig,
    parsed_api_docs: dict[str, ParsedApiDoc],
    markdown_contents: dict[Path, str],
) -> None:
    """Verify PUBLIC_API_MODULES, mkdocs nav, docs/api/index.md, and members are alphabetized."""
    assert tuple(sorted(PUBLIC_API_MODULES, key=str.casefold)) == PUBLIC_API_MODULES

    nav = mkdocs_config.get("nav", [])
    python_api_entries: list[str] = []
    for item in nav:
        if isinstance(item, dict) and "Python API" in item:
            for subitem in item["Python API"]:
                if isinstance(subitem, dict):
                    python_api_entries.extend(label for label in subitem if label != "Overview")
    assert sorted(python_api_entries, key=str.casefold) == python_api_entries

    api_index_text = markdown_contents[_DOCS_DIR / "api" / "index.md"]
    module_names = [
        cols[1].strip().split("`")[1]
        for cols in _iter_markdown_table_rows(api_index_text)
        if len(cols) > 2 and cols[1].strip().startswith("[`reach.")
    ]
    assert sorted(module_names, key=str.casefold) == module_names

    unsorted = {
        f"{stem}.md": parsed.members
        for stem, parsed in parsed_api_docs.items()
        if parsed.members != sorted(parsed.members, key=str.casefold)
    }
    assert not unsorted, f"API doc members must be sorted alphabetically: {unsorted}"


def _collect_export_callables(mod: Any, exported: set[str]) -> list[tuple[str, Any]]:
    """Collect public functions and public methods on exported classes in a module."""
    targets: list[tuple[str, Any]] = []
    for sym_name in sorted(exported):
        obj = getattr(mod, sym_name, None)
        if inspect.isfunction(obj):
            targets.append((sym_name, obj))
        elif inspect.isclass(obj):
            targets.extend(
                (f"{sym_name}.{attr_name}", attr_val)
                for attr_name, attr_val in inspect.getmembers(obj, predicate=inspect.isfunction)
                if (not attr_name.startswith("_") or attr_name == "__init__")
                and getattr(attr_val, "__qualname__", "").startswith(f"{sym_name}.")
            )
    return targets


def _collect_export_annotations(mod: Any, exported: set[str]) -> list[tuple[str, str]]:
    """Collect (label, annotation_str) pairs for public callables and Pydantic model_fields."""
    annotations: list[tuple[str, str]] = []
    for qualname, func in _collect_export_callables(mod, exported):
        sig = inspect.signature(func)
        annotations.extend(
            (f"{qualname} parameter '{param.name}'", str(param.annotation))
            for param in sig.parameters.values()
            if param.annotation is not inspect.Parameter.empty
        )
        if sig.return_annotation is not inspect.Signature.empty:
            annotations.append((f"{qualname} return annotation", str(sig.return_annotation)))

    for sym_name in sorted(exported):
        obj = getattr(mod, sym_name, None)
        if inspect.isclass(obj) and issubclass(obj, BaseModel):
            raw_annotations = getattr(obj, "__annotations__", {})
            annotations.extend(
                (f"{sym_name}.{fname}", str(raw_annotations.get(fname, finfo.annotation)))
                for fname, finfo in obj.model_fields.items()
                if not fname.startswith("_")
            )
    return annotations


@pytest.mark.parametrize("mod_name", PUBLIC_API_MODULES)
def test_public_api_signatures_reference_no_private_or_unexported_types(mod_name: str) -> None:
    """Verify public signatures and model_fields do not expose private or unexported types."""
    mod = importlib.import_module(mod_name)
    exported: set[str] = set(getattr(mod, "__all__", ()))
    if not exported:
        return

    # Intra-module classes, functions, and non-Annotated domain type aliases must be exported
    intra_domain_defs = {
        name
        for name, val in vars(mod).items()
        if not name.startswith("__")
        and (
            (
                (inspect.isclass(val) or inspect.isfunction(val))
                and getattr(val, "__module__", None) == mod_name
            )
            or (
                type(val).__name__ == "TypeAliasType"
                and get_origin(getattr(val, "__value__", None)) is not Annotated
            )
        )
    }

    private_type_re = re.compile(r"\b(_[A-Za-z][A-Za-z0-9_]*)\b")
    ident_re = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\b")
    leaks: list[str] = []

    for label, ann_str in _collect_export_annotations(mod, exported):
        leaks.extend(
            f"{label} exposes private type {priv!r}"
            for priv in private_type_re.findall(ann_str)
            if priv != "_"
        )
        leaks.extend(
            f"{label} references unexported intra-module type {tok!r}"
            for tok in ident_re.findall(ann_str)
            if tok in intra_domain_defs and tok not in exported
        )

    assert not leaks, f"Private or unexported type(s) exposed in public API of {mod_name}: {leaks}"


# ==============================================================================
# 5. Code Snippets, Mermaid Diagrams & Prettier/MkDocs Formatting Guards
# ==============================================================================


def test_doc_python_snippets_syntax_and_imports(markdown_contents: dict[Path, str]) -> None:
    """Verify Python code blocks parse cleanly and all reach.* imports resolve."""
    for md_file, content in markdown_contents.items():
        for i, snippet in enumerate(extract_fenced_blocks(content, "python"), 1):
            try:
                tree = ast.parse(snippet)
            except SyntaxError as err:
                pytest.fail(
                    f"Syntax error in Python snippet #{i} in {md_file.relative_to(_ROOT)}: {err}"
                )
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name == "reach" or alias.name.startswith("reach."):
                            importlib.import_module(alias.name)
                elif (
                    isinstance(node, ast.ImportFrom)
                    and node.module
                    and (node.module == "reach" or node.module.startswith("reach."))
                ):
                    mod = importlib.import_module(node.module)
                    for alias in node.names:
                        assert hasattr(mod, alias.name), (
                            f"Imported symbol {alias.name!r} not found in {node.module} "
                            f"in snippet #{i} of {md_file.relative_to(_ROOT)}"
                        )


def test_mermaid_class_diagrams_match_models(markdown_contents: dict[Path, str]) -> None:
    """Verify attributes in Mermaid classDiagram blocks exist on actual reach model classes."""
    known_classes: dict[str, type[Any]] = {
        name: obj
        for mod in (reach.models, reach.artifact)
        for name, obj in inspect.getmembers(mod, inspect.isclass)
        if obj.__module__ == mod.__name__
    }

    class_block_re = re.compile(r"class\s+([A-Za-z0-9_]+)\s*\{([^}]*)\}", re.MULTILINE)
    attr_line_re = re.compile(
        r"^\s*[+#~-]\s*(?:[A-Za-z0-9_|.\[\]]+\s+)?([a-zA-Z_][a-zA-Z0-9_]*)\s*$"
    )

    for md_file, content in markdown_contents.items():
        rel_path = md_file.relative_to(_ROOT)
        for block in extract_fenced_blocks(content, "mermaid"):
            if not block.lstrip().startswith("classDiagram"):
                continue
            for match in class_block_re.finditer(block):
                cls_name, body = match.group(1), match.group(2)
                cls = known_classes.get(cls_name)
                if cls is None:
                    continue
                valid_attrs = (
                    set(getattr(cls, "model_fields", {}).keys())
                    | set(getattr(cls, "model_computed_fields", {}).keys())
                    | set(dir(cls))
                )
                for line in body.splitlines():
                    attr_match = attr_line_re.match(line)
                    if attr_match:
                        attr_name = attr_match.group(1)
                        assert attr_name in valid_attrs, (
                            f"Mermaid classDiagram in {rel_path} declares "
                            f"{cls_name}.{attr_name}, missing on {cls.__module__}.{cls_name}"
                        )


def test_mkdocs_admonition_and_block_fences_are_prettier_and_pymdownx_safe(
    markdown_contents: dict[Path, str],
    mkdocs_config: MkDocsConfig,
) -> None:
    """Ensure docs/ uses Prettier-safe '///' blocks and renders cleanly via Python-Markdown."""
    github_alert_re = re.compile(
        r"^\s*>\s*\[!(?:NOTE|TIP|IMPORTANT|WARNING|CAUTION)\]",
        re.MULTILINE,
    )
    legacy_admonition_re = re.compile(r"^\s*(?:!!!|\?\?\?|===)\s+\w+")
    indented_fence_re = re.compile(r"^[ \t]+///(?:\s|$)")
    unrendered_html_re = re.compile(
        r"<p>\s*(?:!!!|\?\?\?|///|===)\s+|"
        r'<div class="tabbed-content">\s*</div>',
    )

    md_renderer = markdown.Markdown(
        extensions=mkdocs_config["markdown_extensions"],
        extension_configs=mkdocs_config["mdx_configs"],
    )

    for md_file, content in markdown_contents.items():
        if not md_file.is_relative_to(_DOCS_DIR):
            continue
        rel = md_file.relative_to(_ROOT)
        alerts = github_alert_re.findall(content)
        assert not alerts, (
            f"Unparsed GitHub alert blockquote(s) {alerts} in {rel}; "
            "use '/// warning | Title' pymdownx.blocks syntax instead."
        )

        in_code_fence = False
        open_block_line: int | None = None
        for lineno, line in enumerate(content.splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("```"):
                in_code_fence = not in_code_fence
                continue
            if in_code_fence:
                continue

            assert not legacy_admonition_re.match(line), (
                f"Prettier-incompatible legacy admonition/tab syntax at {rel}:{lineno}: {line!r}"
            )
            assert not indented_fence_re.match(line), (
                f"Indented '///' fence at {rel}:{lineno} ({line!r}); add a blank line before "
                "closing '///' so Prettier does not indent it as a list continuation."
            )
            if line.startswith("///"):
                if stripped == "///":
                    assert open_block_line is not None, (
                        f"Unmatched closing '///' fence at {rel}:{lineno}"
                    )
                    open_block_line = None
                else:
                    assert open_block_line is None, (
                        f"Nested or unclosed '///' block at {rel}:{lineno} "
                        f"(previous block opened at line {open_block_line})"
                    )
                    open_block_line = lineno

        assert open_block_line is None, f"Unclosed '///' block opened at {rel}:{open_block_line}"

        md_renderer.reset()
        rendered_html = md_renderer.convert(content)
        unrendered_match = unrendered_html_re.search(rendered_html)
        assert unrendered_match is None, (
            f"Unrendered admonition/tab/block syntax in rendered HTML of {rel}: "
            f"{unrendered_match.group(0)!r}"
        )


def test_no_overescaped_latex_in_markdown(markdown_contents: dict[Path, str]) -> None:
    """Detect accidental double-backslash escaping in LaTeX math blocks."""
    bad_pattern = re.compile(r"\$[^$\n]*(?:\\{2}\w+|\\_)[^$\n]*\$")
    for md_file, content in markdown_contents.items():
        matches = bad_pattern.findall(content)
        assert not matches, (
            f"Over-escaped LaTeX math found in {md_file.relative_to(_ROOT)}: {matches}"
        )


def test_markdown_tables_have_no_blank_lines(markdown_contents: dict[Path, str]) -> None:
    """Detect broken markdown tables separated by blank lines within table rows."""
    table_row_re = re.compile(r"^\s*\|.*\|\s*$")
    delim_re = re.compile(r"^\|(\s*:?-+:?\s*\|)+$")

    for md_file, content in markdown_contents.items():
        lines = content.splitlines()
        for i, line in enumerate(lines[:-1]):
            if table_row_re.match(line) and not lines[i + 1].strip():
                next_idx = i + 2
                while next_idx < len(lines) and not lines[next_idx].strip():
                    next_idx += 1
                if next_idx < len(lines) and table_row_re.match(lines[next_idx]):
                    is_new_table = next_idx + 1 < len(lines) and bool(
                        delim_re.match(lines[next_idx + 1].strip())
                    )
                    if not is_new_table:
                        pytest.fail(
                            f"Broken table with blank line inside table rows at "
                            f"{md_file.relative_to(_ROOT)}:{i + 1}",
                        )


def test_readme_repository_layout_paths_exist(readme_text: str) -> None:
    """Verify all files and directories listed in README Repository Layout exist on disk."""
    layout_section = extract_section(readme_text, "Repository Layout", level=2)
    blocks = extract_fenced_blocks(layout_section)
    assert blocks, "Repository Layout code block missing from README.md"

    current_root: Path | None = None
    for raw_line in blocks[0].splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.endswith("/"):
            current_root = _ROOT / line.rstrip("/")
            assert current_root.exists(), f"Root directory {current_root} does not exist"
            continue
        if "├──" in line or "└──" in line:
            parts = re.split(r"[├└]──\s*", line)
            if len(parts) < 2:
                continue
            entry = parts[1].split("#")[0].strip().rstrip("/")
            if entry in ("...", ""):
                continue
            assert current_root is not None, "Layout entry without current root"
            target_path = current_root / entry
            assert target_path.exists(), f"README layout path {target_path} does not exist"


def test_concept_docs_are_cross_referenced(markdown_contents: dict[Path, str]) -> None:
    """Verify every concept document in docs/concepts/ is referenced across the docs."""
    concept_docs_dir = _DOCS_DIR / "concepts"
    all_content = "\n".join(
        text for path, text in markdown_contents.items() if path.parent != concept_docs_dir
    )
    for concept_file in concept_docs_dir.glob("*.md"):
        rel_path = f"concepts/{concept_file.name}"
        assert rel_path in all_content or concept_file.name in all_content, (
            f"Concept doc {concept_file.name} is orphaned and not referenced elsewhere"
        )


def _extract_markdown_cli_commands(
    content: str,
    verbs: set[str],
) -> list[tuple[str, list[str], str]]:
    """Extract concrete CLI command invocations from markdown code blocks."""
    invocations: list[tuple[str, list[str], str]] = []
    for block in extract_fenced_blocks(content, "bash", "sh"):
        cleaned_block = block.replace("\\\n", " ")
        for raw_line in cleaned_block.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "## Synopsis" in line:
                continue
            match = re.search(r"(?:^|[$\s`])(?:uv run )?reach\s+([a-z0-9-]+)\s*(.*)", line)
            if not match:
                continue
            verb, rest = match.group(1), match.group(2)
            if verb not in verbs:
                continue
            cmd_args = re.split(r"[|#>]", rest)[0].strip().rstrip("`")
            if re.search(r"\[[A-Z_]+\]|<[a-z_.-]+>|\{[a-z,]+\}", cmd_args):
                continue
            try:
                words = shlex.split(cmd_args)
            except ValueError:
                continue
            invocations.append((verb, words, line))
    return invocations


def test_cli_command_snippets_in_markdown_use_valid_flags(
    markdown_contents: dict[Path, str],
) -> None:
    """Verify all CLI command options used in markdown code blocks exist on commands."""
    verbs = set(_verbs())

    for md_file, content in markdown_contents.items():
        for verb, words, line in _extract_markdown_cli_commands(content, verbs):
            subverb: str | None = None
            rem_words = words
            if rem_words and not rem_words[0].startswith("-"):
                with_sub = app[verb]
                if rem_words[0] in with_sub:
                    subverb = rem_words[0]
                    rem_words = rem_words[1:]
            valid_flags = _command_valid_flags(verb, subverb)
            for word in rem_words:
                if word.startswith("-") and not word.startswith("---") and word != "-":
                    flag = word.split("=")[0]
                    if re.match(r"^-[0-9]", flag):
                        continue
                    assert flag in valid_flags, (
                        f"Invalid CLI option {flag!r} used with 'reach {verb}' in "
                        f"{md_file.relative_to(_ROOT)}: {line!r}"
                    )
