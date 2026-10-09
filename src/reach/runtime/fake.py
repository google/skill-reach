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

"""Provide deterministic fake runtime test doubles for evaluation and optimization."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, override

from reach.config import RuntimeSettings, resolve_path
from reach.runtime import (
    FAKE_AGENT,
    AgentOptions,
    AgentRuntime,
    CatalogFit,
    SelectionOutcome,
    SessionSummary,
    SkillRoot,
    resolve_options,
)
from reach.runtime.generator import BaseTextGenerator

if TYPE_CHECKING:
    from reach.models import Catalog, Skill

#: Canonical default model identifier for the internal fake test runtime.
FAKE_MODEL = "fake-model"

type Response = str | SelectionOutcome | Sequence[str] | None

__all__ = [
    "FAKE_AGENT",
    "FAKE_MODEL",
    "FakeGenerator",
    "FakeOptions",
    "FakeRuntime",
    "Response",
    "register_fake_agent",
    "resolve_fake_options",
]


class FakeOptions(AgentOptions):
    """Specify configuration options for the fake runtime agent."""

    model: str = FAKE_MODEL
    materialize: bool = False


def resolve_fake_options(raw: Mapping[str, Any] | None) -> FakeOptions:
    """Resolve FakeOptions instance from raw mapping."""
    return FakeOptions.model_validate(dict(raw or {}))


class FakeGenerator(BaseTextGenerator[FakeOptions]):
    """Scripted text generator test double for query drafting and optimization."""

    name: str = FAKE_AGENT
    _responses: Mapping[str, str] | Callable[[str], str] | str

    def __init__(
        self,
        responses: Mapping[str, str] | Callable[[str], str] | str = "",
        *,
        model: str = FAKE_MODEL,
        prompt_budget_chars: int | None = None,
        cost_usd: float = 0.0,
        timeout_s: int = 300,
        completion: str = "",
        options: FakeOptions | None = None,
    ) -> None:
        """Initialize fake generator with canned responses or string completion."""
        opts = options if options is not None else FakeOptions(model=model)
        super().__init__(model=opts.model or model, timeout_s=timeout_s, options=opts)
        self._responses = responses
        self._prompt_budget_chars = prompt_budget_chars
        self.cost_usd = cost_usd
        self.completion = completion
        self.prompts: list[str] = []
        self.schemas: list[Any] = []

    @override
    def prompt_budget_chars(self) -> int | None:
        """Return configured prompt character budget limit."""
        if self._prompt_budget_chars is not None:
            return self._prompt_budget_chars
        return super().prompt_budget_chars()

    def complete(self, prompt: str, *, schema: str | Mapping[str, Any] | None = None) -> str:
        """Record prompt and return scripted completion string."""
        self.prompts.append(prompt)
        self.schemas.append(schema)
        self.completions += 1
        self.completion_cost_usd += self.cost_usd
        if self.completion:
            return self.completion
        if isinstance(self._responses, Mapping):
            return str(self._responses.get(prompt, ""))
        if isinstance(self._responses, str):
            return self._responses
        return str(self._responses(prompt))


class FakeRuntime(AgentRuntime[FakeOptions]):
    """Implement AgentRuntime using deterministic scripted responses for testing."""

    name = FAKE_AGENT
    _skills_subpath = ".agents/skills"

    def __init__(
        self,
        responses: Mapping[str, Response] | Callable[[str], Response] | None = None,
        *,
        model: str = FAKE_MODEL,
        default: Response = None,
        cost_usd: float | None = 0.01,
        roots: Sequence[SkillRoot] = (),
        fit: CatalogFit | None = None,
        options: FakeOptions | None = None,
        materialize: bool = False,
        settings: RuntimeSettings | None = None,
    ) -> None:
        """Initialize fake runtime with scripted responses, roots, and budgets."""
        opts = options if options is not None else FakeOptions(model=model, materialize=materialize)
        super().__init__(settings=settings, options=opts)
        self._responses = responses
        self._default = default
        self._roots = tuple(roots)
        self._fit = fit if fit is not None else CatalogFit()
        self.cost_usd = cost_usd
        self.installs: list[tuple[Catalog, Path]] = []
        self.fittings: list[Catalog] = []
        self.queries: list[str] = []
        self.root_queries: list[Path] = []
        self.materialize = materialize or self.options.materialize

    @override
    def skill_roots(self, workdir: Path) -> tuple[SkillRoot, ...]:
        """Record the queried workspace path and return configured fake skill roots."""
        target = resolve_path(workdir)
        self.root_queries.append(target)
        if self._roots:
            return self._roots
        if self.materialize:
            return super().skill_roots(target)
        return ()

    @override
    def fit(self, catalog: Catalog, skills: Iterable[Skill]) -> CatalogFit:
        """Record the catalog fitting request and return configured CatalogFit."""
        self.fittings.append(catalog)
        return self._fit

    @override
    def install(self, catalog: Catalog, skills: Iterable[Skill], workdir: Path) -> Path:
        """Validate catalog skills against corpus and record installation request."""
        target = resolve_path(workdir)
        self.installs.append((catalog, target))
        if self.materialize:
            return super().install(catalog, skills, target)
        self._resident = catalog.skills
        return target

    @override
    def parse_stream(
        self,
        lines: Iterable[str],
        resident: Sequence[str] = (),
        early_exit: bool = False,
    ) -> SessionSummary:
        """Parse stream lines into a SessionSummary using fake configuration."""
        summary = super().parse_stream(lines, resident, early_exit)
        invoked = self._default if isinstance(self._default, str) else None
        if invoked:
            return summary.model_copy(
                update={"invoked_skills": (invoked,)},
            )
        return summary

    def _lookup(self, query_text: str) -> Response:
        """Look up the scripted response for a query string."""
        if self._responses is None:
            return self._default
        if isinstance(self._responses, Mapping):
            return self._responses.get(query_text, self._default)
        return self._responses(query_text)

    @override
    def select(
        self,
        query_text: str,
        workdir: Path,
        target_skill: str | None = None,
    ) -> SelectionOutcome:
        """Return scripted SelectionOutcome for the given query."""
        try:
            self.queries.append(query_text)
            answer = self._lookup(query_text)
            if isinstance(answer, SelectionOutcome):
                if answer.observed_catalog:
                    return answer
                return answer.model_copy(update={"observed_catalog": self._resident})

            raw_skills = (
                (answer,)
                if isinstance(answer, str)
                else (tuple(s for s in answer if s) if answer else ())
            )
            return self._make_tracker(target_skill).apply_to_outcome(
                SelectionOutcome(
                    invoked_skills=raw_skills,
                    turns_taken=len(raw_skills) or 1,
                    observed_catalog=self._resident,
                    observed_tools=("Skill",),
                    cost_usd=self.cost_usd,
                    duration_ms=1,
                ),
            )
        finally:
            self.post_probe(workdir)


def register_fake_agent() -> None:
    """Register the fake runtime driver factory in the global runtime registry."""
    from reach.runtime import register_agent

    def _factory(s: RuntimeSettings) -> FakeRuntime:
        resolved = resolve_options(s) if getattr(s, "options", None) else None
        options = (
            resolved
            if isinstance(resolved, FakeOptions)
            else FakeOptions.model_validate(dict(s.options or {}))
        )
        return FakeRuntime(
            model=options.model,
            options=options,
            materialize=options.materialize,
            settings=s,
        )

    register_agent(
        FAKE_AGENT,
        _factory,
        FakeOptions,
    )
