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

"""Provide a deterministic keyword-matching runtime for offline CI evaluation."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Annotated, Any, Self, override

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from reach.config import RuntimeSettings, resolve_path
from reach.leak import FUNCTION_WORDS
from reach.models import Catalog, Skill
from reach.retrieval import Bm25Scorer, tokenize
from reach.runtime import (
    AgentOptions,
    AgentRuntime,
    SelectionOutcome,
    SessionStatus,
    SessionSummary,
    SkillRoot,
)
from reach.runtime.generator import BaseTextGenerator

if TYPE_CHECKING:
    from pathlib import Path

__all__ = [
    "KeywordGenerator",
    "KeywordOptions",
    "KeywordRuntime",
]

_TARGET_DOC_RE = re.compile(
    r"<target_documentation>\s*(.*?)\s*</target_documentation>",
    re.DOTALL,
)
_RIVAL_DOC_RE = re.compile(
    r'<rival_documentation\s+index="(\d+)">\s*(.*?)\s*</rival_documentation>',
    re.DOTALL,
)
_WRITE_COUNT_RE = re.compile(r"Write\s+(\d+)\s+(?:[\w-]+\s+)*queries", re.IGNORECASE)
_MARKDOWN_PREFIX_RE = re.compile(r"^(?:[#>*+-]+|\d+\.)\s*")
_INLINE_MD_RE = re.compile(r"[*`~]")


class _KeywordDraftItem(BaseModel):
    """Hold a single deterministic query draft synthesized from documentation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    text: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
    citation: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
    reason: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
    rival_index: int | None = None


class _KeywordDraftEnvelope(BaseModel):
    """Wrap synthesized keyword query drafts in the standard response schema."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    queries: tuple[_KeywordDraftItem, ...] = ()


class KeywordOptions(AgentOptions):
    """Configure options for keyword heuristic routing."""

    model: str = "keyword"
    scope: str = Field(default="user")


class KeywordRuntime(AgentRuntime[KeywordOptions]):
    """Implement a deterministic keyword matching agent for offline evaluation."""

    name = "keyword"
    _skills_subpath = ".agents/skills"

    def __init__(
        self,
        settings_or_options: RuntimeSettings | KeywordOptions | None = None,
        options: KeywordOptions | None = None,
    ) -> None:
        """Initialize keyword heuristic runtime with settings or options."""
        resolved_settings: RuntimeSettings | None = None
        resolved_options: KeywordOptions | None = options
        if isinstance(settings_or_options, KeywordOptions):
            resolved_options = settings_or_options
        elif isinstance(settings_or_options, RuntimeSettings):
            resolved_settings = settings_or_options

        super().__init__(settings=resolved_settings, options=resolved_options)
        self._pattern: re.Pattern[str] | None = None
        self._term_to_skill: dict[str, str] = {}
        self._bm25: Bm25Scorer | None = None

    def _set_resident(self, resident: Sequence[str]) -> None:
        """Precompute normalized term mapping and compiled regex for fast matching."""
        self._resident = tuple(resident)
        self._pattern, self._term_to_skill = _compile_term_matcher(self._resident)

    @override
    def clone_isolated(self) -> Self:
        """Create a thread-local isolated clone with fresh regex and term map state."""
        clone = super().clone_isolated()
        clone._pattern = None  # noqa: SLF001
        clone._term_to_skill = {}  # noqa: SLF001
        clone._bm25 = None  # noqa: SLF001
        return clone

    @override
    def install(self, catalog: Catalog, skills: Iterable[Skill], workdir: Path) -> Path:
        """Materialize resident skills and index their descriptions for BM25 fallback."""
        skill_list = tuple(skills)
        target = super().install(catalog, skill_list, workdir)
        resident_set = set(self._resident)
        docs = {
            s.name: tuple(t for t in tokenize(s.description) if t not in FUNCTION_WORDS)
            for s in skill_list
            if isinstance(s, Skill) and s.name in resident_set
        }
        self._bm25 = Bm25Scorer(documents=docs) if docs else None
        return target

    def match_skill(self, text: str, resident: Sequence[str] = ()) -> str | None:
        """Find the first matching resident skill mentioned in the given text."""
        active = tuple(resident) if resident else self._resident
        if not active:
            return None
        pattern: re.Pattern[str] | None
        if active == self._resident and self._pattern is not None:
            pattern, term_map = self._pattern, self._term_to_skill
        else:
            pattern, term_map = _compile_term_matcher(active)

        if pattern is None:
            return None
        matches = list(pattern.finditer(text))
        if not matches:
            return None
        best = max(matches, key=lambda m: len(m.group(1)))
        return term_map.get(best.group(1).lower())

    def _match_bm25(self, text: str) -> str | None:
        """Rank resident skills by BM25 description overlap when no literal name matches."""
        if self._bm25 is None or not self._resident:
            return None
        query_tokens = [t for t in tokenize(text) if t not in FUNCTION_WORDS]
        if not query_tokens:
            return None
        best_skill: str | None = None
        best_score = 0.0
        for name in self._resident:
            score = self._bm25.score(query_tokens, name)
            if score > best_score:
                best_score = score
                best_skill = name
        return best_skill

    @override
    def _post_install(self, workdir: Path) -> None:
        """Precompute normalized term mapping and compiled regex for fast matching."""
        del workdir
        self._set_resident(self._resident)

    @override
    def skill_roots(self, workdir: Path) -> tuple[SkillRoot, ...]:
        """Return default user skill directory for workspace."""
        here = self.skills_dir(resolve_path(workdir))
        return (SkillRoot(path=here, scope=self.options.scope),) if here.is_dir() else ()

    @override
    def select(
        self,
        query_text: str,
        workdir: Path,
        target_skill: str | None = None,
    ) -> SelectionOutcome:
        """Route query to literal skill mention first, falling back to BM25 description match."""
        try:
            if self._pattern is None and self._resident:
                self._set_resident(self._resident)

            matched = self.match_skill(query_text)
            if matched is None:
                matched = self._match_bm25(query_text)
            return self.make_tracker(target_skill).apply_to_outcome(
                SelectionOutcome(
                    invoked_skills=(matched,) if matched is not None else (),
                    observed_catalog=self._resident,
                    observed_tools=("keyword",),
                    cost_usd=0.0,
                    duration_ms=1,
                ),
            )
        finally:
            self.post_probe(workdir)

    @override
    def parse_stream(
        self,
        lines: Iterable[str],
        resident: Sequence[str] = (),
        early_exit: bool = False,
    ) -> SessionSummary:
        """Extract invoked skills mentioned in stream lines."""
        line_list = list(lines)
        invoked = self.match_skill("\n".join(line_list), resident)
        invoked_skills = (invoked,) if invoked is not None else ()
        return SessionSummary(
            invoked_skills=invoked_skills,
            early_exit=early_exit,
            status=SessionStatus.SUCCESS if line_list else None,
        )


def _compile_term_matcher(
    resident: Sequence[str],
) -> tuple[re.Pattern[str] | None, dict[str, str]]:
    """Compile a regex pattern and term lookup mapping from a sequence of resident skills."""
    if not resident:
        return None, {}
    term_map: dict[str, str] = {}
    for name in resident:
        term_map[name.lower()] = name
        term_map[name.replace("-", " ").lower()] = name

    sorted_terms = sorted(term_map.keys(), key=len, reverse=True)
    escaped = "|".join(re.escape(t) for t in sorted_terms)
    pattern = re.compile(rf"(?<![\w-])({escaped})(?![\w-])", re.IGNORECASE)
    return pattern, term_map


_MIN_PASSAGE_LEN = 10
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_URL_RE = re.compile(r"https?://\S+")
_FILE_PATH_RE = re.compile(r"(?:\.{1,2}/|~/)\S+")
_TRAILING_CONNECTOR_RE = re.compile(
    r"\b(?:such as|via|at|from|in|on|with|for|to|or|and)\s*$",
    re.IGNORECASE,
)
_BOILERPLATE_PREFIXES = (
    "script paths below are relative",
    "all commands output json",
    "you don't remember",
    "freshness check",
    "authentication failures",
    "routing note",
    "load these only when",
)


def _extract_doc_passages(doc_text: str) -> list[str]:
    """Extract clean, citable lines from a documentation block."""
    passages: list[str] = []
    deferred: list[str] = []
    in_code_block = False
    for raw_line in doc_text.splitlines():
        stripped = raw_line.strip()
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            continue
        if in_code_block or not stripped:
            continue
        if set(stripped) <= {"|", "-", ":", " "}:
            continue
        cleaned_cite = _MARKDOWN_PREFIX_RE.sub("", stripped).strip()
        if (len(cleaned_cite) < _MIN_PASSAGE_LEN or "[REDACTED]" in cleaned_cite) and len(
            cleaned_cite.replace("[REDACTED]", "").strip()
        ) < _MIN_PASSAGE_LEN:
            continue
        if not cleaned_cite or cleaned_cite in passages or cleaned_cite in deferred:
            continue
        plain_lower = _INLINE_MD_RE.sub("", cleaned_cite).strip().lower()
        if stripped.startswith(">") or plain_lower.startswith(_BOILERPLATE_PREFIXES):
            deferred.append(cleaned_cite)
        else:
            passages.append(cleaned_cite)
    passages.extend(deferred)
    if not passages and doc_text.strip():
        fallback = doc_text.strip().splitlines()[0].strip()
        if fallback:
            passages.append(fallback)
    return passages


_QUERY_TEMPLATES: tuple[str, ...] = (
    "How do I handle: {topic}?",
    "Help me with {topic}.",
    "What is the recommended workflow for {topic}?",
    "Can you walk me through {topic}?",
    "I need assistance with {topic}.",
)


def _format_topic(citation: str) -> str:
    """Convert a verbatim citation line into a clean topic phrase for query synthesis."""
    topic = _MD_LINK_RE.sub(r"\1", citation)
    topic = _URL_RE.sub("", topic)
    topic = _INLINE_MD_RE.sub("", topic).replace("[REDACTED]", "this workflow")
    topic = _FILE_PATH_RE.sub("the CLI script", topic)
    topic = " ".join(topic.split()).strip(" .:;,-")
    topic = _TRAILING_CONNECTOR_RE.sub("", topic).strip(" .:;,-")
    return topic or "this task"


def _synthesize_keyword_drafts(prompt: str) -> str:
    """Synthesize deterministic grounded JSON queries from a generation prompt."""
    target_match = _TARGET_DOC_RE.search(prompt)
    if target_match is None:
        return ""
    count_match = _WRITE_COUNT_RE.search(prompt)
    count = max(1, int(count_match.group(1))) if count_match else 1
    is_adversarial = "NEAR-MISS adversarial" in prompt or "OUT OF SCOPE near-miss" in prompt

    items: list[_KeywordDraftItem] = []
    if is_adversarial:
        rival_matches = _RIVAL_DOC_RE.findall(prompt)
        rival_sources: list[tuple[int, str]] = []
        for idx_str, rival_body in rival_matches:
            idx = int(idx_str)
            rival_sources.extend((idx, p) for p in _extract_doc_passages(rival_body))
        if rival_sources:
            for i in range(count):
                r_idx, cite = rival_sources[i % len(rival_sources)]
                template = _QUERY_TEMPLATES[i % len(_QUERY_TEMPLATES)]
                items.append(
                    _KeywordDraftItem(
                        text=template.format(topic=_format_topic(cite)),
                        citation=cite,
                        reason=(
                            f"Deterministic rival-{r_idx} near-miss query from keyword generator."
                        ),
                        rival_index=r_idx,
                    ),
                )
            return _KeywordDraftEnvelope(queries=tuple(items)).model_dump_json()

    passages = _extract_doc_passages(target_match.group(1))
    if not passages:
        return _KeywordDraftEnvelope(queries=()).model_dump_json()

    for i in range(count):
        cite = passages[i % len(passages)]
        template = _QUERY_TEMPLATES[i % len(_QUERY_TEMPLATES)]
        topic = _format_topic(cite)
        text = (
            f"Out of scope hardware request regarding {topic}?"
            if is_adversarial
            else template.format(topic=topic)
        )
        items.append(
            _KeywordDraftItem(
                text=text,
                citation=cite,
                reason="Deterministic offline query synthesized from documentation.",
                rival_index=None,
            ),
        )
    return _KeywordDraftEnvelope(queries=tuple(items)).model_dump_json()


class KeywordGenerator(BaseTextGenerator[KeywordOptions]):
    """Deterministic offline text generator for keyword evaluation and query drafting."""

    name: str = "keyword"

    def __init__(
        self,
        model: str = "keyword",
        *,
        timeout_s: int = 300,
        options: KeywordOptions | None = None,
    ) -> None:
        """Initialize keyword text generator with model, timeout, and options."""
        super().__init__(
            model=model,
            timeout_s=timeout_s,
            options=options if options is not None else KeywordOptions(model=model),
        )

    @override
    def complete(self, prompt: str, *, schema: str | Mapping[str, Any] | None = None) -> str:
        """Synthesize grounded JSON queries when given target documentation, else empty string."""
        del schema
        self.completions += 1
        if "<target_documentation>" in prompt:
            return _synthesize_keyword_drafts(prompt)
        return ""
