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

"""Extract semantic routing handoffs and unbounded scope attractors from skills."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Final

from pydantic import BaseModel, ConfigDict

from reach.runtime import builtin_tool_names

if TYPE_CHECKING:
    from collections.abc import Sequence

    from reach.models import Skill

__all__ = [
    "KEBAB_NAME_RE",
    "RESERVED_TOOL_NAMES",
    "SkillLintSemantics",
    "detect_unbounded_attractor",
    "extract_corpus_semantics",
    "extract_skill_references",
]

#: Unconditional greedy scope phrases that flag unbounded attractors at any length
#: when a description lacks concrete technical/domain specificity.
_STRONG_ATTRACTOR_RE: Final = re.compile(
    r"\b(?:(?P<any_kw>any)\s+(?:[a-z]+\s+)?(?:task|tasks|problem|problems|request|requests|"
    r"feature|features|bugfix)|general[- ]purpose|all[- ]in[- ]one|"
    r"universal\s+(?:assistant|helper|tool|skill)|everything|"
    r"manage\s+files\s+and\s+run\s+commands|run\s+commands\s+in\s+the\s+terminal)\b",
    re.IGNORECASE,
)
_SHORT_BROAD_MARKER_RE: Final = re.compile(
    r"\b(?:any|all|every|universal|assistant|helper)\b",
    re.IGNORECASE,
)
_TECHNICAL_TOKEN_RE: Final = re.compile(r"`[^\n`]+`|\.[a-z0-9]{2,4}\b|[a-z0-9]+/[a-z0-9]+|\b\d+\b")
_MIN_SPECIFIC_DESCRIPTION_CHARS: Final = 85
_MIN_ACRONYM_LENGTH: Final = 2

_POSITIVE_HANDOFF_VERB: Final = (
    r"(?<!don't\s)(?<!do\snot\s)(?<!never\s)\b"
    r"(?:use|see|prefer|refer\s+to|defer\s+to|delegate\s+to|hand\s+off\s+to)"
)
_KEBAB_ID: Final = r"[a-z0-9]+(?:-[a-z0-9]+)+"
_KEBAB_ID_RE: Final = re.compile(_KEBAB_ID, re.IGNORECASE)
_WILDCARD_PATTERN: Final = r"[a-z0-9]+(?:-[a-z0-9]+)*-\*"
_KEBAB_TOKEN: Final = rf"(?:{_KEBAB_ID}|{_WILDCARD_PATTERN})"
_KEBAB_TOKEN_RE: Final = re.compile(_KEBAB_TOKEN, re.IGNORECASE)
_TARGET_TOKEN_RE: Final = re.compile(
    rf"(?P<bt>`?)(?P<token>{_KEBAB_TOKEN})(?P=bt)",
    re.IGNORECASE,
)

#: Matches a kebab-case token when backticked or in terminal noun position
#: (followed by clause punctuation, end-of-string, 'instead', 'first', singular 'skill',
#: or 'or'/'and'), naturally excluding compound adjectives modifying a following noun.
_TERMINAL_KEBAB: Final = (
    rf"(?:`{_KEBAB_TOKEN}`"
    rf"|{_KEBAB_TOKEN}(?=\s*(?:instead\b|first\b|skill\b(?!s)|[).,;:]|$|\s+(?:or|and)\b)))"
)
_TERMINAL_KEBAB_COORDINATE: Final = (
    rf"(?:`{_KEBAB_TOKEN}`"
    rf"|{_KEBAB_TOKEN}(?=\s*(?:instead\b|first\b|skills?\b|[).,;:]|$|\s+(?:or|and)\b)))"
)
_TERMINAL_LIST_SINGULAR: Final = (
    rf"{_TERMINAL_KEBAB}(?:\s+(?:first|instead|skill\b(?!s)))?"
    rf"(?:\s*(?:,\s*(?:or|and)\b|,|\bor\b|\band\b)\s*"
    rf"(?:use\s+|see\s+|prefer\s+|the\s+)?{_TERMINAL_KEBAB}(?:\s+(?:first|instead|skill\b(?!s)))?)*"
)
_TERMINAL_LIST_WITH_THE: Final = (
    rf"the\s+{_TERMINAL_KEBAB_COORDINATE}(?:\s+(?:first|instead|skills?\b))?"
    rf"(?:\s*(?:,\s*(?:or|and)\b|,|\bor\b|\band\b)\s*"
    rf"(?:use\s+|see\s+|prefer\s+|the\s+)?{_TERMINAL_KEBAB_COORDINATE}(?:\s+(?:first|instead|skills?\b))?)*"
)
_TERMINAL_LIST: Final = rf"(?:{_TERMINAL_LIST_WITH_THE}|{_TERMINAL_LIST_SINGULAR})"

_PAREN_HANDOFF_RE: Final = re.compile(
    rf"\([^)]*?{_POSITIVE_HANDOFF_VERB}\s+(?P<targets>{_TERMINAL_LIST})[^)]*\)",
    re.IGNORECASE,
)

_BACKTICK_HANDOFF_RE: Final = re.compile(
    rf"{_POSITIVE_HANDOFF_VERB}\s+(?:the\s+)?`({_KEBAB_TOKEN})`",
    re.IGNORECASE,
)

_NEGATIVE_CLAUSE_MARKER_RE: Final = re.compile(
    rf"\b(?:don't\s+use|do\s+not\s+use|not\s+for\b|never\s+use|avoid\s+using|"
    rf"instead\s+of\b|rather\s+than\b|for\s+[^.!?;]+,\s*(?:use|prefer|defer\s+to|see)\b)"
    rf"|{_POSITIVE_HANDOFF_VERB}\s+(?:the\s+)?`?{_KEBAB_TOKEN}`?\s+(?:instead|first|skill\b(?!s))",
    re.IGNORECASE,
)

_VERB_TARGET_IN_CLAUSE_RE: Final = re.compile(
    rf"{_POSITIVE_HANDOFF_VERB}\s+(?P<targets>{_TERMINAL_LIST})",
    re.IGNORECASE,
)

_HANDOFF_CANDIDATE_PREFILTER_RE: Final = re.compile(
    r"\b(?:don't\s+use|do\s+not\s+use|not\s+for\b|never\s+use|avoid\s+using|"
    r"instead\b|rather\s+than\b|see\b|prefer\b|refer\s+to\b|defer\s+to\b|"
    r"delegate\s+to\b|hand\s+off\s+to\b|use\s+(?:the\s+)?`?[a-z0-9]+-[a-z0-9*-]+|"
    r"any\b|all\b|every\b|universal\b|general[- ]purpose\b|all[- ]in[- ]one\b)\b",
    re.IGNORECASE,
)


def _has_domain_specificity(description: str) -> bool:
    """Return True if description contains concrete technical anchors or domain specification."""
    if _TECHNICAL_TOKEN_RE.search(description):
        return True
    for sentence in re.split(r"[.!?]+", description):
        words = sentence.strip().split()
        for word in words[1:]:
            cleaned = word.strip("(),;:\"'")
            if any(ch.isupper() for ch in cleaned):
                return True
        if words:
            first = words[0].strip("(),;:\"'")
            if len(first) >= _MIN_ACRONYM_LENGTH and (
                first.isupper() or any(ch.isupper() for ch in first[1:])
            ):
                return True
    return False


def detect_unbounded_attractor(description: str) -> str | None:
    """Return the broad-scope marker if a description lacks domain specificity, else None."""
    if _has_domain_specificity(description):
        return None
    if strong := _STRONG_ATTRACTOR_RE.search(description):
        return strong.group("any_kw") or strong.group(0)
    if len(description) < _MIN_SPECIFIC_DESCRIPTION_CHARS and (
        short := _SHORT_BROAD_MARKER_RE.search(description)
    ):
        return short.group(0)
    return None


KEBAB_NAME_RE: Final = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
_NON_SKILL_SUFFIXES: Final[tuple[str, ...]] = ("-specific", "-related")


def _derive_reserved_tool_names() -> frozenset[str]:
    """Derive lowercase, kebab-case, and flat reserved tool names from runtime primitives."""
    names: set[str] = set()
    for raw in builtin_tool_names():
        kebab = re.sub(r"(?<!^)(?=[A-Z])", "-", raw).replace("_", "-").lower()
        flat = raw.lower().replace("_", "").replace("-", "")
        names.add(raw.lower())
        names.add(kebab)
        names.add(flat)
    return frozenset(names)


RESERVED_TOOL_NAMES: Final[frozenset[str]] = _derive_reserved_tool_names()


def _add_if_valid_ref(
    found: set[str],
    token: str,
    self_lower: str | None,
    *,
    is_explicit_backtick: bool = False,
) -> None:
    """Add normalized kebab-case reference token if non-empty and not self."""
    raw_token = token.strip().lower()
    is_wildcard = raw_token.endswith("-*")
    cleaned = raw_token.removesuffix("-*")
    if (
        cleaned
        and ("-" in cleaned or is_wildcard)
        and cleaned != self_lower
        and KEBAB_NAME_RE.match(cleaned)
        and (is_explicit_backtick or not cleaned.endswith(_NON_SKILL_SUFFIXES))
        and cleaned not in RESERVED_TOOL_NAMES
    ):
        found.add(cleaned)


def extract_skill_references(
    description: str,
    *,
    self_name: str | None = None,
) -> tuple[str, ...]:
    """Extract explicit skill references from negative/redirect clauses in a description.

    Uses grammatical noun-position boundaries: a kebab-case token after a positive
    handoff verb (`use`, `see`, `prefer`, `defer to`) must either be enclosed in
    backticks or stand in terminal noun position (followed by clause punctuation,
    `instead`, `first`, singular `skill`, or `or`/`and` to another skill). Compound
    adjectives modifying a following noun (e.g. `use product-specific skills`)
    are excluded structurally without word blocklists.

    Args:
        description: Frontmatter description string to inspect.
        self_name: Optional name of the skill itself to exclude self-references.

    Returns:
        Sorted tuple of unique referenced skill names in kebab-case.
    """
    if not description or not description.strip():
        return ()

    self_lower = self_name.strip().lower() if self_name else None
    found: set[str] = set()

    for match in _PAREN_HANDOFF_RE.finditer(description):
        targets = match.group("targets")
        for m in _TARGET_TOKEN_RE.finditer(targets):
            _add_if_valid_ref(
                found,
                m.group("token"),
                self_lower,
                is_explicit_backtick=bool(m.group("bt")),
            )

    for match in _BACKTICK_HANDOFF_RE.finditer(description):
        _add_if_valid_ref(found, match.group(1), self_lower, is_explicit_backtick=True)

    for sentence in re.split(r"[.!?]+", description):
        if not _NEGATIVE_CLAUSE_MARKER_RE.search(sentence):
            continue
        for match in _VERB_TARGET_IN_CLAUSE_RE.finditer(sentence):
            targets = match.group("targets")
            for m in _TARGET_TOKEN_RE.finditer(targets):
                _add_if_valid_ref(
                    found,
                    m.group("token"),
                    self_lower,
                    is_explicit_backtick=bool(m.group("bt")),
                )

    return tuple(sorted(found))


class SkillLintSemantics(BaseModel):
    """Represent structured routing boundaries and scope attractors for a skill."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    skill: str
    handoff_targets: tuple[str, ...] = ()
    unbounded_attractor_phrase: str | None = None


def extract_corpus_semantics(
    skills: Sequence[Skill],
) -> dict[str, SkillLintSemantics]:
    """Extract routing handoff targets and attractor semantics deterministically across a corpus."""
    results: dict[str, SkillLintSemantics] = {}
    for skill in skills:
        if not _HANDOFF_CANDIDATE_PREFILTER_RE.search(skill.description):
            continue
        results[skill.name] = SkillLintSemantics(
            skill=skill.name,
            handoff_targets=extract_skill_references(skill.description, self_name=skill.name),
            unbounded_attractor_phrase=detect_unbounded_attractor(skill.description),
        )
    return results
