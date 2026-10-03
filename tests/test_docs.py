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
from typing import TYPE_CHECKING, Any, NamedTuple

import pytest
from mkdocs.config import load_config

if TYPE_CHECKING:
    from mkdocs.config.defaults import MkDocsConfig
    from pydantic import BaseModel

import reach
import reach.cli
from reach.cli.app import _verbs, app
from reach.config import (
    BUILTIN_AGENT_DEFAULT_MODELS,
    DEFAULT_CLAUDE_MODEL,
    DEFAULT_GEMINI_MODEL,
    KNOWN_CLIENT_SKILLS_DIRS,
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
from reach.runtime import FAKE_AGENT, known_agents

_ROOT = Path(__file__).resolve().parent.parent
_DOCS_DIR = _ROOT / "docs"
_MKDOCS_FILE = _ROOT / "mkdocs.yml"


@pytest.fixture(scope="module")
def mkdocs_config() -> MkDocsConfig:
    """Provide the parsed mkdocs.yml configuration for documentation tests."""
    assert _MKDOCS_FILE.exists(), f"mkdocs.yml not found at {_MKDOCS_FILE}"
    return load_config(str(_MKDOCS_FILE))


@pytest.fixture(scope="module")
def config_doc_text() -> str:
    """Provide text content of docs/configuration.md."""
    return (_DOCS_DIR / "configuration.md").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def readme_text() -> str:
    """Provide text content of README.md."""
    return (_ROOT / "README.md").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def all_markdown_files() -> tuple[Path, ...]:
    """Provide all documentation and root markdown files."""
    return (*_DOCS_DIR.rglob("*.md"), _ROOT / "README.md", _ROOT / "CONTRIBUTING.md")


@pytest.fixture(scope="module")
def reach_source_text() -> str:
    """Provide concatenated source text for all reach Python modules."""
    src_files = list((_ROOT / "src" / "reach").rglob("*.py"))
    return "\n".join(f.read_text(encoding="utf-8") for f in src_files)


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


@functools.cache
def _command_valid_flags(verb: str) -> frozenset[str]:
    """Assemble all valid option flags for a registered CLI command."""
    cmd = app[verb]
    flags: set[str] = {"--help", "--version"}
    for arg in cmd.assemble_argument_collection():
        flags.update(name for name in arg.names if name.startswith("-"))
    return frozenset(flags)


def extract_table_column_entries(
    markdown_section: str,
    column_index: int = 1,
    pattern: str = r"`([a-zA-Z0-9_.-]+)`",
) -> set[str]:
    """Extract backtick-enclosed identifiers from a column of a markdown table."""
    entries: set[str] = set()
    for raw_line in markdown_section.splitlines():
        line = raw_line.strip()
        if not line.startswith("|") or line.startswith("| :---"):
            continue
        cols = line.split("|")
        if len(cols) > column_index + 1:
            entries.update(re.findall(pattern, cols[column_index]))
    return entries


def extract_python_snippets(content: str) -> list[str]:
    """Extract non-empty Python code blocks from markdown text."""
    py_block_re = re.compile(r"```python\s*\n(.*?)\n```", re.DOTALL)
    return [
        match.group(1).strip() for match in py_block_re.finditer(content) if match.group(1).strip()
    ]


def test_mkdocs_config_is_valid(mkdocs_config: MkDocsConfig) -> None:
    """Verify mkdocs.yml can be loaded and contains expected top-level keys."""
    assert mkdocs_config.get("site_name") == "skill-reach"
    assert "nav" in mkdocs_config
    assert "theme" in mkdocs_config
    assert mkdocs_config["theme"].name == "material"


@pytest.mark.parametrize("verb", _verbs())
def test_every_cli_verb_has_documentation_page(verb: str) -> None:
    """Verify registered CLI subcommand has a dedicated documentation page in docs/cli/."""
    doc_page = _DOCS_DIR / "cli" / f"{verb}.md"
    assert doc_page.is_file(), f"Missing CLI documentation page for {verb} at {doc_page}"


def test_cli_doc_pages_correspond_to_registered_verbs() -> None:
    """Verify all documentation pages in docs/cli/ correspond to registered CLI subcommands."""
    cli_docs_dir = _DOCS_DIR / "cli"
    assert cli_docs_dir.is_dir(), f"CLI docs directory missing at {cli_docs_dir}"

    verbs = set(_verbs())
    cli_doc_files = {p.stem for p in cli_docs_dir.glob("*.md") if p.name != "index.md"}
    stale_pages = cli_doc_files - verbs
    assert not stale_pages, (
        "Orphaned or stale CLI doc pages in docs/cli/ not in registered verbs: "
        f"{sorted(stale_pages)}"
    )


@pytest.mark.parametrize("verb", _verbs())
def test_every_cli_doc_is_in_mkdocs_nav(verb: str, mkdocs_config: MkDocsConfig) -> None:
    """Verify CLI documentation page is referenced in mkdocs.yml navigation."""
    nav_paths = set(_flatten_nav(mkdocs_config.get("nav", [])))
    expected_path = f"cli/{verb}.md"
    assert expected_path in nav_paths, f"Expected {expected_path} to be in mkdocs.yml nav"


def test_cli_nav_entries_correspond_to_registered_verbs(mkdocs_config: MkDocsConfig) -> None:
    """Verify all CLI documentation entries in mkdocs.yml correspond to registered subcommands."""
    nav_paths = set(_flatten_nav(mkdocs_config.get("nav", [])))
    cli_nav_entries = {p for p in nav_paths if p.startswith("cli/")}
    expected_entries = {f"cli/{verb}.md" for verb in _verbs()} | {"cli/index.md"}
    stale_entries = cli_nav_entries - expected_entries
    assert not stale_entries, (
        f"Stale CLI documentation entries in mkdocs.yml nav: {sorted(stale_entries)}"
    )


def test_referenced_nav_files_exist(mkdocs_config: MkDocsConfig) -> None:
    """Verify every file referenced in the mkdocs.yml nav exists on disk."""
    nav_paths = _flatten_nav(mkdocs_config.get("nav", []))

    missing: list[str] = []
    for rel_path in nav_paths:
        full_path = _DOCS_DIR / rel_path
        if not full_path.exists():
            missing.append(rel_path)

    assert not missing, f"Files referenced in mkdocs.yml nav do not exist: {missing}"


def test_all_doc_files_are_in_mkdocs_nav(mkdocs_config: MkDocsConfig) -> None:
    """Verify every markdown file in docs/ is referenced in mkdocs.yml navigation."""
    all_docs_on_disk = {p.relative_to(_DOCS_DIR).as_posix() for p in _DOCS_DIR.rglob("*.md")}
    nav_paths = set(_flatten_nav(mkdocs_config.get("nav", [])))
    orphaned_docs = all_docs_on_disk - nav_paths
    assert not orphaned_docs, (
        f"Orphaned markdown files in docs/ missing from mkdocs.yml nav: {sorted(orphaned_docs)}"
    )


@pytest.mark.parametrize("verb", _verbs())
def test_documented_cli_options_exist_on_commands(verb: str) -> None:
    """Verify all documented CLI options in docs/cli/{verb}.md exist on the registered command."""
    doc_file = _DOCS_DIR / "cli" / f"{verb}.md"
    assert doc_file.is_file(), f"Missing CLI doc file for {verb}: {doc_file}"
    content = doc_file.read_text(encoding="utf-8")
    if "## Options" not in content:
        return

    options_part = content.split("## Options")[1]
    if "\n## " in options_part:
        options_part = options_part.split("\n## ")[0]

    doc_flags = extract_table_column_entries(
        options_part,
        column_index=1,
        pattern=r"`(--[a-zA-Z0-9-]+)`",
    )
    cli_flags = _command_valid_flags(verb)
    stale_flags = doc_flags - cli_flags
    assert not stale_flags, (
        f"Stale or unrecognized CLI options in docs/cli/{verb}.md: {sorted(stale_flags)}"
    )


#: Deliberately unlisted or internal flags per command
UNDOCUMENTED_CLI_FLAGS: dict[str, set[str]] = {
    "query": {
        "--catalog-size",
        "--config",
        "--draft-concurrency",
        "--draft-only",
        "--early-exit",
        "--effort",
        "--generator-arm",
        "--global",
        "--max-turns",
        "--mode",
        "--model",
        "--opt",
        "--partial",
        "--quiet",
        "--rescope",
        "--rivals",
        "--run-dir",
        "--seed",
        "--tag",
        "--target",
        "--timeout",
        "--top-rivals",
        "--workdir",
    },
}


@pytest.mark.parametrize("verb", _verbs())
def test_all_cli_options_are_documented(verb: str) -> None:
    """Verify that every CLI option registered on a command is documented in docs/cli/{verb}.md."""
    doc_file = _DOCS_DIR / "cli" / f"{verb}.md"
    assert doc_file.is_file(), f"Missing CLI doc file for {verb}: {doc_file}"
    content = doc_file.read_text(encoding="utf-8")
    doc_flags = set(re.findall(r"`(--[a-zA-Z0-9-]+)`", content))
    doc_positionals = set(re.findall(r"`\[?([A-Z0-9_-]+)\]?`", content))

    cmd = app[verb]
    cli_flags = {
        name
        for arg in cmd.assemble_argument_collection()
        for name in (set(arg.names) - set(arg.negatives))
        if name.startswith("--") and name not in {"--help", "--version"}
    }

    # Filter out flags covered by documented positionals and inverse booleans
    covered = {
        f
        for f in cli_flags
        if f.removeprefix("--").replace("-", "_").upper() in doc_positionals
        or (f.startswith("--no-") and f.replace("--no-", "--") in doc_flags)
    }

    missing = cli_flags - doc_flags - covered - UNDOCUMENTED_CLI_FLAGS.get(verb, set())
    assert not missing, f"Undocumented CLI options on 'reach {verb}': {sorted(missing)}"


@pytest.mark.parametrize("agent", [a for a in known_agents() if a != FAKE_AGENT])
def test_all_public_agents_documented(
    agent: str,
    readme_text: str,
    config_doc_text: str,
) -> None:
    """Verify supported agent runtime is documented in README, index, and configuration docs."""
    index_text = (_DOCS_DIR / "index.md").read_text(encoding="utf-8")
    assert agent in readme_text, f"Agent {agent!r} missing from README.md"
    assert agent in index_text, (
        f"Agent {agent!r} missing from Live Empirical Probing in docs/index.md"
    )
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


@pytest.mark.parametrize("rule_id", sorted(RULES))
def test_lint_rules_documented_in_lint_docs(rule_id: str, config_doc_text: str) -> None:
    """Verify static lint rule is documented in docs/cli/lint.md and configuration docs."""
    lint_doc_text = (_DOCS_DIR / "cli" / "lint.md").read_text(encoding="utf-8")
    assert rule_id in lint_doc_text, f"Rule {rule_id!r} missing from docs/cli/lint.md"
    assert rule_id in config_doc_text, f"Rule {rule_id!r} missing from docs/configuration.md"


def test_documented_lint_rules_exist_in_registry() -> None:
    """Verify all lint rules documented in docs/cli/lint.md exist in the RULES registry."""
    lint_doc_text = (_DOCS_DIR / "cli" / "lint.md").read_text(encoding="utf-8")
    assert "## Built-in Lint Rules" in lint_doc_text, "Missing '## Built-in Lint Rules' section"
    rules_part = lint_doc_text.split("## Built-in Lint Rules")[1].split("\n## ")[0]
    doc_rules = extract_table_column_entries(rules_part, column_index=1, pattern=r"`([a-z0-9-]+)`")
    stale_rules = doc_rules - set(RULES.keys())
    assert not stale_rules, (
        f"Stale or unregistered lint rules in docs/cli/lint.md: {sorted(stale_rules)}"
    )


def test_explain_examples_reference_valid_rules(all_markdown_files: tuple[Path, ...]) -> None:
    """Verify all --explain CLI examples in documentation reference registered rules."""
    explain_re = re.compile(r"--explain\s+([a-z0-9-]+)")

    for md_file in all_markdown_files:
        content = md_file.read_text(encoding="utf-8")
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


def test_documented_configuration_sections_valid(config_doc_text: str) -> None:
    """Verify section headings in docs/configuration.md correspond to valid models."""
    sections_ref = config_doc_text.split("## Sections & Options Reference", maxsplit=1)[1].split(
        "\n## ",
        maxsplit=1,
    )[0]
    section_matches = re.findall(r"### `?\[([a-zA-Z0-9_.-]+)\]`?", sections_ref)
    allowed_sections = set(RunConfig.model_fields.keys()) | {"lint.rules", "agents", "models"}

    stale_sections: list[str] = []
    for sec in section_matches:
        base_sec = sec.split(".")[0]
        if sec not in allowed_sections and base_sec not in allowed_sections:
            stale_sections.append(sec)

    assert not stale_sections, (
        f"Unrecognized or stale configuration sections in docs/configuration.md: {stale_sections}"
    )


SECTION_MODEL_MAP: dict[str, type[BaseModel]] = {
    "general": GeneralSettings,
    "discovery": DiscoverySettings,
    "study": StudySettings,
    "catalog": CatalogSettings,
    "plan": PlanSettings,
    "runtime": RuntimeSettings,
    "lint": LintSettings,
    "check": CheckSettings,
    "retrieval": RetrievalSettings,
    "overlap": OverlapSettings,
    "diff": DiffSettings,
    "query": QuerySettings,
    "optimize": OptimizeSettings,
    "registry": RegistrySettings,
}

# Settings fields that are internal, dynamic, or CLI-only, and deliberately omitted from tables
UNDOCUMENTED_CONFIG_FIELDS: dict[str, set[str]] = {
    "runtime": {"agent", "allowed_tools", "options"},
    "lint": {"rules", "similarity_threshold"},
    "registry": {"registry", "fresh", "no_cache"},
}


def _extract_section_block(config_doc_text: str, section_name: str) -> str:
    """Extract markdown text block for a specific section from docs/configuration.md."""
    sections_ref = config_doc_text.split("## Sections & Options Reference", maxsplit=1)[1].split(
        "\n## ",
        maxsplit=1,
    )[0]
    section_blocks = re.split(r"\n(?=### `?\[)", sections_ref)
    for block in section_blocks:
        header_match = re.search(r"### `?\[([a-zA-Z0-9_.-]+)\]`?", block)
        if header_match and header_match.group(1) == section_name:
            return str(block)
    return ""


@pytest.mark.parametrize(("section_name", "model_cls"), sorted(SECTION_MODEL_MAP.items()))
def test_all_model_fields_are_documented(
    section_name: str,
    model_cls: type[BaseModel],
    config_doc_text: str,
) -> None:
    """Verify that every setting on each configuration model is documented in configuration.md."""
    section_block = _extract_section_block(config_doc_text, section_name)
    assert section_block, f"Missing section block for [{section_name}] in configuration.md"
    doc_keys = extract_table_column_entries(section_block)
    expected_keys = set(model_cls.model_fields.keys()) - UNDOCUMENTED_CONFIG_FIELDS.get(
        section_name,
        set(),
    )
    missing = expected_keys - doc_keys
    assert not missing, (
        f"Model fields in [{section_name}] missing from docs/configuration.md: {sorted(missing)}"
    )


@pytest.mark.parametrize(("section_name", "model_cls"), sorted(SECTION_MODEL_MAP.items()))
def test_documented_configuration_keys_exist_in_models(
    section_name: str,
    model_cls: type[BaseModel],
    config_doc_text: str,
) -> None:
    """Verify configuration keys in docs/configuration.md exist on settings models."""
    section_block = _extract_section_block(config_doc_text, section_name)
    assert section_block, f"Missing section block for [{section_name}] in configuration.md"
    doc_keys = {
        k for k in extract_table_column_entries(section_block) if not k.startswith("rules.")
    }
    stale_keys = doc_keys - set(model_cls.model_fields.keys())
    assert not stale_keys, (
        f"Stale or unrecognized configuration keys in docs/configuration.md [{section_name}]: "
        f"{sorted(stale_keys)}"
    )


def test_documented_agent_profiles_exist_in_known_agents(config_doc_text: str) -> None:
    """Verify all agent profiles documented in docs/configuration.md exist in known_agents."""
    agent_matches = re.findall(r"\[agents\.([a-zA-Z0-9_-]+)\]", config_doc_text)
    public_agents = set(known_agents()) - {FAKE_AGENT}

    stale_agents = set(agent_matches) - public_agents
    assert not stale_agents, (
        f"Stale or unregistered agent profiles in docs/configuration.md: {sorted(stale_agents)}"
    )


@pytest.mark.parametrize("doc_rel_path", ["cli/query.md", "cli/eval.md"])
def test_generator_model_default_documented(doc_rel_path: str) -> None:
    """Verify CLI reference docs document the default generator model correctly."""
    doc_text = (_DOCS_DIR / doc_rel_path).read_text(encoding="utf-8")
    assert f"`{DEFAULT_GEMINI_MODEL}`" in doc_text, (
        f"Expected {DEFAULT_GEMINI_MODEL!r} to be documented in docs/{doc_rel_path}"
    )


def test_clean_all_documents_benchmark_query_removal() -> None:
    """Verify docs/cli/clean.md explicitly documents benchmark query set deletion under --all."""
    clean_doc_text = (_DOCS_DIR / "cli" / "clean.md").read_text(encoding="utf-8")
    assert ".reach/queries" in clean_doc_text or "queries.json" in clean_doc_text, (
        "Expected docs/cli/clean.md to document .reach/queries deletion under --all"
    )


def test_check_strict_default_documented() -> None:
    """Verify docs/cli/check.md documents --strict default as true."""
    check_doc_text = (_DOCS_DIR / "cli" / "check.md").read_text()
    assert re.search(r"`--strict` / `--no-strict`\s*\|\s*Flag\s*\|\s*`true`", check_doc_text), (
        "Expected docs/cli/check.md to document --strict default as true"
    )


def test_configuration_doc_documents_default_model_profiles(config_doc_text: str) -> None:
    """Verify docs/configuration.md documents default Gemini and Claude model profiles."""
    gemini_key = DEFAULT_GEMINI_MODEL.replace(".", "-")
    claude_key = DEFAULT_CLAUDE_MODEL.replace(".", "-")
    assert f"[models.{gemini_key}]" in config_doc_text, (
        f"docs/configuration.md should show [models.{gemini_key}] under [models.*]"
    )
    assert f"[models.{claude_key}]" in config_doc_text, (
        f"docs/configuration.md should show [models.{claude_key}] under [models.*]"
    )


#: Modules and packages in reach that serve internal execution, CLI, or presentation logic.
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

#: Discover all top-level public API modules dynamically. Any newly added module
#: is treated as public by default unless explicitly classified in INTERNAL_MODULES.
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


def _parse_api_doc_members(md_file: Path) -> ParsedApiDoc:
    """Extract the documented module name and declared members list from an API doc page."""
    lines = md_file.read_text().splitlines()
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
def parsed_api_docs() -> dict[str, ParsedApiDoc]:
    """Provide parsed API doc definitions keyed by module short name."""
    api_docs_dir = _DOCS_DIR / "api"
    docs: dict[str, ParsedApiDoc] = {}
    for md_file in sorted(api_docs_dir.glob("*.md")):
        if md_file.name == "index.md":
            continue
        docs[md_file.stem] = _parse_api_doc_members(md_file)
    return docs


def test_mkdocstrings_api_members_exist_in_modules(
    parsed_api_docs: dict[str, ParsedApiDoc],
) -> None:
    """Verify all member symbols declared in docs/api/*.md exist in target modules."""
    for stem, parsed in parsed_api_docs.items():
        mod = importlib.import_module(parsed.module_name)
        for member in parsed.members:
            assert hasattr(mod, member), (
                f"Member {member!r} declared in {stem}.md does not exist in {parsed.module_name}"
            )


def test_all_repo_modules_are_accounted_for() -> None:
    """Verify all modules in src/reach are either documented as public or classified as internal."""
    all_discovered = {
        f"reach.{info.name}"
        for info in pkgutil.iter_modules(reach.__path__)
        if not info.name.startswith("_")
    }
    unclassified = all_discovered - (set(PUBLIC_API_MODULES) | INTERNAL_MODULES)
    assert not unclassified, (
        f"Unclassified module(s) {unclassified} found in src/reach! "
        "Every module must either be documented in docs/api/ or explicitly declared in "
        "INTERNAL_MODULES."
    )


@pytest.mark.parametrize("mod_name", PUBLIC_API_MODULES)
def test_every_public_api_module_has_documentation_page(mod_name: str) -> None:
    """Verify each public API module has a dedicated doc page in docs/api/."""
    short_name = mod_name.rsplit(".", maxsplit=1)[-1]
    doc_page = _DOCS_DIR / "api" / f"{short_name}.md"
    assert doc_page.is_file(), f"Missing API documentation page for {mod_name} at {doc_page}"


def test_api_doc_pages_correspond_to_public_modules() -> None:
    """Verify all documentation pages in docs/api/ correspond to discovered public modules."""
    api_docs_dir = _DOCS_DIR / "api"
    doc_modules = {f"reach.{p.stem}" for p in api_docs_dir.glob("*.md") if p.name != "index.md"}
    expected_modules = set(PUBLIC_API_MODULES)
    stale_api_docs = doc_modules - expected_modules
    assert not stale_api_docs, (
        "Orphaned or stale API doc pages in docs/api/ not in PUBLIC_API_MODULES: "
        f"{sorted(stale_api_docs)}"
    )


def test_api_index_modules_correspond_to_public_modules() -> None:
    """Verify all modules listed in docs/api/index.md Module Index match public API modules."""
    api_index_text = (_DOCS_DIR / "api" / "index.md").read_text(encoding="utf-8")
    index_modules = {
        f"reach.{stem}" for stem in re.findall(r"\[`reach\.([a-z0-9_-]+)`\]", api_index_text)
    }
    expected_modules = set(PUBLIC_API_MODULES)
    stale_index_modules = index_modules - expected_modules
    assert not stale_index_modules, (
        "Stale module links in docs/api/index.md not in PUBLIC_API_MODULES: "
        f"{sorted(stale_index_modules)}"
    )


@pytest.mark.parametrize("mod_name", PUBLIC_API_MODULES)
def test_documented_api_members_cover_public_interface(
    mod_name: str,
    parsed_api_docs: dict[str, ParsedApiDoc],
) -> None:
    """Verify all primary public classes, functions, and exports are documented."""
    short_name = mod_name.rsplit(".", maxsplit=1)[-1]
    parsed = parsed_api_docs.get(short_name)
    assert parsed is not None, f"Documentation page missing for module {mod_name}"
    doc_member_set = set(parsed.members)

    mod = importlib.import_module(mod_name)
    if hasattr(mod, "__all__"):
        for s in mod.__all__:
            assert hasattr(mod, s), (
                f"Module {mod_name} lists {s!r} in __all__ but attribute does not exist on module"
            )
        expected_members = [s for s in mod.__all__ if not s.startswith("_")]
    else:
        expected_members = [
            name
            for name, obj in inspect.getmembers(mod)
            if not name.startswith("_")
            and getattr(obj, "__module__", None) == mod_name
            and (inspect.isclass(obj) or inspect.isfunction(obj))
        ]

    missing = [sym for sym in expected_members if sym not in doc_member_set]
    assert not missing, f"Public symbols missing from docs/api/{short_name}.md: {missing}"


@pytest.mark.parametrize("mod_name", PUBLIC_API_MODULES)
def test_documented_api_members_match_module_all(
    mod_name: str,
    parsed_api_docs: dict[str, ParsedApiDoc],
) -> None:
    """Verify all documented API members are officially exported in module __all__ when defined."""
    short_name = mod_name.rsplit(".", maxsplit=1)[-1]
    parsed = parsed_api_docs.get(short_name)
    assert parsed is not None, f"Documentation page missing for module {mod_name}"

    mod = importlib.import_module(mod_name)
    if hasattr(mod, "__all__"):
        stale_exports = set(parsed.members) - set(mod.__all__)
        assert not stale_exports, (
            f"Stale or unexported members documented in docs/api/{short_name}.md not in __all__: "
            f"{sorted(stale_exports)}"
        )


@pytest.mark.parametrize("mod_name", PUBLIC_API_MODULES)
def test_module_all_contains_no_private_symbols(mod_name: str) -> None:
    """Verify that module __all__ does not export private or internal symbols."""
    mod = importlib.import_module(mod_name)
    if hasattr(mod, "__all__"):
        private_exports = [s for s in mod.__all__ if s.startswith("_")]
        assert not private_exports, (
            f"Private symbol(s) {private_exports} exported in {mod_name}.__all__"
        )


def test_api_module_index_and_nav_parity(mkdocs_config: MkDocsConfig) -> None:
    """Verify all docs/api/*.md files are listed in docs/api/index.md and mkdocs.yml."""
    api_index_text = (_DOCS_DIR / "api" / "index.md").read_text()
    nav_paths = set(_flatten_nav(mkdocs_config.get("nav", [])))

    for md_file in sorted((_DOCS_DIR / "api").glob("*.md")):
        if md_file.name == "index.md":
            continue
        module_doc_path = f"api/{md_file.name}"
        assert module_doc_path in nav_paths, (
            f"{module_doc_path} is missing from mkdocs.yml navigation"
        )
        assert md_file.name in api_index_text, (
            f"{md_file.name} is missing from Module Index table in docs/api/index.md"
        )


def test_doc_python_snippets_syntax(all_markdown_files: tuple[Path, ...]) -> None:
    """Verify all Python code snippets in markdown files parse as valid syntax."""
    for md_file in all_markdown_files:
        content = md_file.read_text(encoding="utf-8")
        for i, snippet in enumerate(extract_python_snippets(content), 1):
            try:
                ast.parse(snippet)
            except SyntaxError as err:
                pytest_fail_msg = (
                    f"Syntax error in Python snippet #{i} in {md_file.relative_to(_ROOT)}: {err}"
                )
                raise AssertionError(pytest_fail_msg) from err


def test_citations_companion_path_documentation() -> None:
    """Verify documentation references queries-citations.json companion file."""
    query_doc_text = (_DOCS_DIR / "cli" / "query.md").read_text()
    assert "queries.citations.json" not in query_doc_text, (
        "Found invalid dotted 'queries.citations.json' in docs/cli/query.md; "
        "expected 'queries-citations.json'"
    )


def test_readme_repository_layout_paths_exist(readme_text: str) -> None:
    """Verify all files and directories listed in README Repository Layout exist on disk."""
    layout_match = re.search(r"## Repository Layout\s+```\s*(.*?)\s*```", readme_text, re.DOTALL)
    assert layout_match is not None, "Repository Layout code block missing from README.md"

    current_root: Path | None = None
    for raw_line in str(layout_match.group(1)).splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.endswith("/"):
            current_root = _ROOT / line.rstrip("/")
            assert current_root.exists(), f"Root directory {current_root} does not exist"
            continue
        if "├──" in line or "└──" in line:
            # Extract entry name (strip tree symbols and comments)
            parts = re.split(r"[├└]──\s*", line)
            if len(parts) < 2:
                continue
            entry = parts[1].split("#")[0].strip().rstrip("/")
            if entry in ("...", ""):
                continue
            assert current_root is not None, "Layout entry without current root"
            target_path = current_root / entry
            assert target_path.exists(), f"README layout path {target_path} does not exist"


@pytest.mark.parametrize("verb", _verbs())
def test_every_cli_verb_in_cli_index_overview(verb: str) -> None:
    """Verify registered CLI subcommand is listed in docs/cli/index.md overview table."""
    cli_index_text = (_DOCS_DIR / "cli" / "index.md").read_text(encoding="utf-8")
    expected_entry = f"[`reach {verb}`]"
    assert expected_entry in cli_index_text, (
        f"CLI verb {verb!r} missing from Command Overview table in docs/cli/index.md"
    )


def test_cli_index_overview_only_lists_registered_verbs() -> None:
    """Verify docs/cli/index.md Command Overview table does not list obsolete subcommands."""
    cli_index_text = (_DOCS_DIR / "cli" / "index.md").read_text(encoding="utf-8")
    overview_verbs = set(re.findall(r"\[`reach ([a-z0-9-]+)`\]", cli_index_text))
    registered_verbs = set(_verbs())
    stale_overview_verbs = overview_verbs - registered_verbs
    assert not stale_overview_verbs, (
        "Obsolete commands listed in docs/cli/index.md Command Overview: "
        f"{sorted(stale_overview_verbs)}"
    )


@pytest.mark.parametrize("verb", _verbs())
def test_every_cli_verb_in_readme_commands_table(verb: str, readme_text: str) -> None:
    """Verify registered CLI verb appears in README.md command tables."""
    pattern = rf"\|\s*\[?`{re.escape(verb)}`\]?\s*\|"
    assert re.search(pattern, readme_text), (
        f"CLI verb '{verb}' is registered but missing from README.md command table"
    )


def test_readme_commands_table_only_lists_registered_verbs(readme_text: str) -> None:
    """Verify README.md command tables do not list obsolete or unregistered commands."""
    commands_split = readme_text.split("## Commands")[1]
    command_section = commands_split.split("## Supported Agents", maxsplit=1)[0]
    doc_verbs = extract_table_column_entries(
        command_section,
        column_index=1,
        pattern=r"`([a-z0-9-]+)`",
    )
    registered_verbs = set(_verbs())
    stale_verbs = doc_verbs - registered_verbs
    assert not stale_verbs, (
        f"Obsolete commands listed in README.md command tables: {sorted(stale_verbs)}"
    )


def test_no_overescaped_latex_in_markdown(all_markdown_files: tuple[Path, ...]) -> None:
    """Detect accidental double-backslash escaping in LaTeX math blocks."""
    bad_pattern = re.compile(r"\$[^$\n]*(?:\\{2}\w+|\\_)[^$\n]*\$")

    for md_file in all_markdown_files:
        content = md_file.read_text(encoding="utf-8")
        matches = bad_pattern.findall(content)
        assert not matches, (
            f"Over-escaped LaTeX math found in {md_file.relative_to(_ROOT)}: {matches}"
        )


def test_doc_python_snippets_reach_imports_resolve(
    all_markdown_files: tuple[Path, ...],
) -> None:
    """Verify all symbols imported from reach.* in doc python snippets resolve to real objects."""
    for md_file in all_markdown_files:
        content = md_file.read_text(encoding="utf-8")
        for i, snippet in enumerate(extract_python_snippets(content), 1):
            tree = ast.parse(snippet)
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.ImportFrom)
                    and node.module
                    and node.module.startswith("reach")
                ):
                    try:
                        mod = importlib.import_module(node.module)
                    except ImportError as err:
                        pytest_fail_msg = (
                            f"Cannot import module {node.module!r} in snippet #{i} of "
                            f"{md_file.relative_to(_ROOT)}: {err}"
                        )
                        raise AssertionError(pytest_fail_msg) from err
                    for alias in node.names:
                        assert hasattr(mod, alias.name), (
                            f"Imported symbol {alias.name!r} not found in {node.module} "
                            f"in snippet #{i} of {md_file.relative_to(_ROOT)}"
                        )


@pytest.mark.parametrize("verb", _verbs())
def test_cli_docs_have_synopsis_and_headings(verb: str) -> None:
    """Verify CLI subcommand document has a proper heading and Synopsis section."""
    doc_file = _DOCS_DIR / "cli" / f"{verb}.md"
    assert doc_file.is_file(), f"Missing CLI doc file: {doc_file}"
    content = doc_file.read_text(encoding="utf-8")
    first_line = content.splitlines()[0].strip() if content.splitlines() else ""
    assert first_line in (f"# `reach {verb}`", f"# reach {verb}"), (
        f"Expected top-level heading '# `reach {verb}`' in {doc_file.name}, got {first_line!r}"
    )
    assert "## Synopsis" in content, f"Missing '## Synopsis' section in {doc_file.name}"


def test_concept_docs_are_cross_referenced(all_markdown_files: tuple[Path, ...]) -> None:
    """Verify every concept document in docs/concepts/ is referenced across the docs."""
    concept_docs_dir = _DOCS_DIR / "concepts"
    non_concept_files = [f for f in all_markdown_files if f.parent != concept_docs_dir]
    all_content = "\n".join(f.read_text(encoding="utf-8") for f in non_concept_files)

    for concept_file in concept_docs_dir.glob("*.md"):
        rel_path = f"concepts/{concept_file.name}"
        assert rel_path in all_content or concept_file.name in all_content, (
            f"Concept doc {concept_file.name} is orphaned and not referenced elsewhere"
        )


def test_public_api_modules_and_nav_are_alphabetized(
    mkdocs_config: MkDocsConfig,
) -> None:
    """Verify PUBLIC_API_MODULES, mkdocs nav, and docs/api/index.md are alphabetized."""
    assert tuple(sorted(PUBLIC_API_MODULES, key=str.casefold)) == PUBLIC_API_MODULES, (
        "PUBLIC_API_MODULES must be sorted alphabetically"
    )

    nav = mkdocs_config.get("nav", [])
    python_api_entries: list[str] = []
    for item in nav:
        if isinstance(item, dict) and "Python API" in item:
            for subitem in item["Python API"]:
                if isinstance(subitem, dict):
                    python_api_entries.extend(label for label in subitem if label != "Overview")
    assert sorted(python_api_entries, key=str.casefold) == python_api_entries, (
        f"mkdocs.yml Python API navigation must be sorted alphabetically: {python_api_entries}"
    )

    api_index_text = (_DOCS_DIR / "api" / "index.md").read_text()
    table_lines = [
        line.split("|")[1].strip()
        for line in api_index_text.splitlines()
        if line.startswith("| [`reach.")
    ]
    module_names = [entry.split("`")[1] for entry in table_lines]
    assert sorted(module_names, key=str.casefold) == module_names, (
        f"docs/api/index.md Module Index table must be sorted alphabetically: {module_names}"
    )


def test_api_doc_members_are_alphabetized(
    parsed_api_docs: dict[str, ParsedApiDoc],
) -> None:
    """Verify that all members: lists in docs/api/*.md are sorted alphabetically."""
    unsorted: dict[str, list[str]] = {}

    for stem, parsed in parsed_api_docs.items():
        expected = sorted(parsed.members, key=str.casefold)
        if parsed.members != expected:
            unsorted[f"{stem}.md"] = parsed.members

    assert not unsorted, f"API doc members must be sorted alphabetically: {unsorted}"


def _extract_markdown_cli_commands(
    md_file: Path,
    verbs: set[str],
) -> list[tuple[str, list[str], str]]:
    """Extract concrete CLI command invocations from markdown code blocks."""
    invocations: list[tuple[str, list[str], str]] = []
    text = md_file.read_text(encoding="utf-8")
    for block in re.findall(r"```(?:bash|sh)\s*\n(.*?)\n```", text, re.DOTALL):
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
    all_markdown_files: tuple[Path, ...],
) -> None:
    """Verify all CLI command options used in markdown code blocks exist on commands."""
    verbs = set(_verbs())

    for md_file in all_markdown_files:
        for verb, words, line in _extract_markdown_cli_commands(md_file, verbs):
            valid_flags = _command_valid_flags(verb)
            for word in words:
                if word.startswith("-") and not word.startswith("---") and word != "-":
                    flag = word.split("=")[0]
                    if re.match(r"^-[0-9]", flag):
                        continue
                    assert flag in valid_flags, (
                        f"Invalid CLI option {flag!r} used with 'reach {verb}' in "
                        f"{md_file.relative_to(_ROOT)}: {line!r}"
                    )


def test_example_models_defined_in_bundled_reach_toml() -> None:
    """Verify all model definitions in reach.example.toml exist in bundled reach.toml."""
    bundled_toml = tomllib.loads((_ROOT / "src/reach/reach.toml").read_text(encoding="utf-8"))
    bundled_models = set(bundled_toml.get("models", {}).keys())

    example_text = (_ROOT / "reach.example.toml").read_text(encoding="utf-8")
    example_models = set(re.findall(r"\[models\.([a-zA-Z0-9_-]+)\]", example_text))
    missing = example_models - bundled_models
    assert not missing, (
        f"Models configured in reach.example.toml missing from src/reach/reach.toml: {missing}"
    )


def test_markdown_tables_have_no_blank_lines(all_markdown_files: tuple[Path, ...]) -> None:
    """Detect broken markdown tables separated by blank lines within table rows."""
    table_row_re = re.compile(r"^\s*\|.*\|\s*$")

    for md_file in all_markdown_files:
        lines = md_file.read_text(encoding="utf-8").splitlines()
        for i, line in enumerate(lines[:-1]):
            if table_row_re.match(line) and not lines[i + 1].strip():
                # Look ahead past blank lines
                next_idx = i + 2
                while next_idx < len(lines) and not lines[next_idx].strip():
                    next_idx += 1
                if next_idx < len(lines) and table_row_re.match(lines[next_idx]):
                    # Check that the line after the blank line is not a new table header + delimiter
                    is_new_table = False
                    if next_idx + 1 < len(lines):
                        delimiter_line = lines[next_idx + 1].strip()
                        delim_pattern = r"^\|(\s*:?-+:?\s*\|)+$"
                        if delimiter_line.startswith("|") and re.match(
                            delim_pattern,
                            delimiter_line,
                        ):
                            is_new_table = True
                    if not is_new_table:
                        pytest.fail(
                            f"Broken table with blank line inside table rows at "
                            f"{md_file.relative_to(_ROOT)}:{i + 1}",
                        )


def test_documented_environment_variables_exist_in_source(
    config_doc_text: str,
    reach_source_text: str,
) -> None:
    """Verify all REACH_* environment variables documented in configuration.md exist in code."""
    assert "## Environment Variables" in config_doc_text, (
        "Missing '## Environment Variables' section"
    )
    env_section = config_doc_text.split("## Environment Variables")[1]
    if "\n## " in env_section:
        env_section = env_section.split("\n## ")[0]

    documented_vars = set(re.findall(r"`(REACH_[A-Z0-9_]+)`", env_section))
    assert documented_vars, (
        "Expected at least one REACH_* environment variable in docs/configuration.md"
    )

    unreferenced = {var for var in documented_vars if var not in reach_source_text}
    assert not unreferenced, (
        f"Documented env vars not referenced anywhere in src/reach: {sorted(unreferenced)}"
    )


def test_all_reach_env_vars_are_documented(
    config_doc_text: str,
    reach_source_text: str,
) -> None:
    """Verify all REACH_* environment variables read in source code are documented."""
    doc_vars = set(re.findall(r"`(REACH_[A-Z0-9_]+)`", config_doc_text))
    code_vars = set(
        re.findall(
            r'os\.(?:environ(?:\.get)?|getenv)\(\s*["\'](REACH_[A-Z0-9_]+)["\']',
            reach_source_text,
        )
    )
    undocumented = code_vars - doc_vars
    assert not undocumented, f"Undocumented REACH_* env vars in src/reach: {sorted(undocumented)}"
