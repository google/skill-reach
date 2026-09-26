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

"""Analyze competing vocabulary and suggest rewrites between rival skills."""

from __future__ import annotations

import re
from collections import Counter
from enum import StrEnum
from typing import TYPE_CHECKING, Annotated

from pydantic import BaseModel, ConfigDict, Field

from reach.catalog import split_frontmatter
from reach.config import OverlapSettings
from reach.leak import BACKGROUND_IDF, FUNCTION_WORDS, contains_run
from reach.overlap import OVERLAP_CAVEAT, Competition, CorpusOverlap
from reach.retrieval import Bm25Scorer, skill_text, tokenize

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from reach.models import Skill

_DEFAULT_OVERLAP = OverlapSettings()

#: Minimum contribution share for a rival term to be considered materially ceded.
MATERIAL_SHARE = _DEFAULT_OVERLAP.material_share

#: Proximity ratio (within 90% of top score) defining close competitor skills.
CONTENDER_BAND = _DEFAULT_OVERLAP.contender_band

#: Maximum number of suggested unclaimed terms to extract from skill body.
CLAIM_LIMIT = _DEFAULT_OVERLAP.claim_limit

#: Minimum token character length for suggested unclaimed terms.
MIN_CLAIM_LENGTH = _DEFAULT_OVERLAP.min_claim_length

#: Minimum frequency count for a term in the skill body to qualify as a claim candidate.
MIN_CLAIM_USES = _DEFAULT_OVERLAP.min_claim_uses

#: Pattern matching sentence delimiters for disclaimer boundary checks.
_SENTENCE = re.compile(r"[.!?]+")

#: Patterns matching non-prose elements (code blocks, links, URLs) in skill manifests.
_NOT_PROSE = (
    re.compile(r"```.*?```", re.DOTALL),
    re.compile(r"`[^`]*`"),
    re.compile(r"\]\([^)]*\)"),
    re.compile(r"https?://\S+"),
)


class Verdict(StrEnum):
    """Categorize the recommended action for a skill's description."""

    CONTESTED = "contested"
    REWORD = "reword"
    UNRIVALED = "unrivaled"


class CededTerm(BaseModel):
    """Represent a description term providing greater BM25 score to a rival skill."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    term: str
    share: Annotated[float, Field(ge=0.0)]
    disclaimed: bool = False


class Rewrite(BaseModel):
    """Hold suggested modifications, ceded terms, and replacement vocabulary."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    skill: str
    rival: str = ""
    ceded: tuple[CededTerm, ...] = ()
    unclaimed: tuple[str, ...] = ()
    contenders: tuple[str, ...] = ()
    band: Annotated[float, Field(ge=0.0, le=1.0)] = CONTENDER_BAND
    rival_disclaims_target: bool = False
    missing_mutual_handoffs: tuple[str, ...] = ()

    @property
    def reword(self) -> tuple[CededTerm, ...]:
        """Return ceded terms that have not been explicitly disclaimed."""
        return tuple(term for term in self.ceded if not term.disclaimed)

    @property
    def disclaimed(self) -> tuple[CededTerm, ...]:
        """Return ceded terms appearing in sentences that explicitly name the rival."""
        return tuple(term for term in self.ceded if term.disclaimed)

    @property
    def verdict(self) -> Verdict:
        """Derive the appropriate recommendation verdict based on rival competition."""
        if not self.rival:
            return Verdict.UNRIVALED
        return Verdict.REWORD if self.reword else Verdict.CONTESTED

    @property
    def crowded(self) -> bool:
        """Return True if multiple competitors fall within the contender band."""
        return len(self.contenders) > 1

    @property
    def actionable(self) -> bool:
        """Return True if the skill requires rewording or is missing a mutual handoff."""
        return self.verdict is Verdict.REWORD or bool(self.missing_mutual_handoffs)


def skill_body(skill: Skill) -> str:
    """Extract the markdown body of a skill, stripping YAML frontmatter."""
    manifest = skill.path / "SKILL.md"
    try:
        text = manifest.read_text(encoding="utf-8")
    except OSError:
        return ""
    split = split_frontmatter(text)
    return split[1] if split is not None else text


def _prose(body: str) -> str:
    """Filter non-prose code blocks and URLs from body text."""
    for pattern in _NOT_PROSE:
        body = pattern.sub(" ", body)
    return body


def _corpus_terms_excluding(
    corpus: Sequence[Skill],
    target_name: str,
    tokenized_corpus: Mapping[str, frozenset[str]] | None = None,
) -> frozenset[str]:
    """Gather all tokens appearing in corpus skills other than the target."""
    if tokenized_corpus is not None:
        return frozenset().union(
            *(tokens for name, tokens in tokenized_corpus.items() if name != target_name)
        )
    return frozenset(
        term
        for skill in corpus
        if skill.name != target_name
        for term in tokenize(skill_text(skill))
    )


def _is_unclaimed_candidate(
    term: str,
    uses: int,
    elsewhere: frozenset[str],
    own: frozenset[str],
    scorer: Bm25Scorer,
    *,
    min_uses: int = MIN_CLAIM_USES,
) -> bool:
    """Determine whether a term qualifies as an unclaimed descriptive term."""
    return (
        uses >= min_uses
        and term not in elsewhere
        and term not in own
        and term not in FUNCTION_WORDS
        and scorer.idf(term) > BACKGROUND_IDF
    )


def unclaimed_terms(
    target: Skill,
    corpus: Sequence[Skill],
    body: str,
    scorer: Bm25Scorer,
    limit: int = CLAIM_LIMIT,
    tokenized_corpus: Mapping[str, frozenset[str]] | None = None,
    *,
    min_length: int = MIN_CLAIM_LENGTH,
    min_uses: int = MIN_CLAIM_USES,
    settings: OverlapSettings | None = None,
) -> tuple[str, ...]:
    """Identify distinctive body terms absent from competitor selection surfaces."""
    effective_limit = (
        settings.claim_limit if settings is not None and limit == CLAIM_LIMIT else limit
    )
    effective_min_len = (
        settings.min_claim_length
        if settings is not None and min_length == MIN_CLAIM_LENGTH
        else min_length
    )
    effective_min_uses = (
        settings.min_claim_uses if settings is not None and min_uses == MIN_CLAIM_USES else min_uses
    )
    if effective_limit < 0:
        msg = f"cannot name fewer than 0 terms, got {effective_limit}"
        raise ValueError(msg)
    tokenized = tokenized_corpus or {
        name: frozenset(tokens) for name, tokens in scorer.documents.items()
    }
    elsewhere = _corpus_terms_excluding(corpus, target.name, tokenized_corpus=tokenized)
    own = tokenized.get(target.name) or frozenset(tokenize(skill_text(target)))
    counted = Counter(
        term for term in tokenize(_prose(body)) if term.isalpha() and len(term) >= effective_min_len
    )
    available = [
        term
        for term, uses in counted.items()
        if _is_unclaimed_candidate(
            term,
            uses,
            elsewhere,
            own,
            scorer,
            min_uses=effective_min_uses,
        )
    ]
    available.sort(key=lambda term: (-counted[term], term))
    return tuple(available[:effective_limit])


def _disclaims(description: str, rival: str, term: str) -> bool:
    """Return True if every sentence containing the term also names the rival."""
    wanted = tokenize(rival)
    carrying = [
        tokens
        for sentence in _SENTENCE.split(description)
        if term in (tokens := tokenize(sentence))
    ]
    return bool(carrying) and all(contains_run(tokens, wanted) for tokens in carrying)


def ceded_terms(
    target: Skill,
    rival: Skill,
    scorer: Bm25Scorer,
    rival_score: float,
    share: float = MATERIAL_SHARE,
) -> tuple[CededTerm, ...]:
    """Identify terms in target description that contribute more to rival BM25 score."""
    if rival_score <= 0:
        return ()
    query = tokenize(skill_text(target))
    mine = scorer.contributions(query, target.name)
    theirs = scorer.contributions(query, rival.name)
    described = frozenset(tokenize(target.description))
    own_name_tokens = frozenset(tokenize(target.name))
    found = [
        CededTerm(
            term=term,
            share=(value - mine.get(term, 0.0)) / rival_score,
            disclaimed=_disclaims(target.description, rival.name, term),
        )
        for term, value in theirs.items()
        if term in described
        and term not in own_name_tokens
        and term not in FUNCTION_WORDS
        and scorer.idf(term) > BACKGROUND_IDF
        and (value - mine.get(term, 0.0)) / rival_score >= share
    ]
    return tuple(sorted(found, key=lambda c: (-c.share, c.term)))


def _resolve_target_and_rival(
    competition: Competition,
    skills: Sequence[Skill],
) -> tuple[Skill, Skill | None]:
    """Look up the target skill and its nearest scoring rival in the corpus."""
    target = next((s for s in skills if s.name == competition.skill), None)
    if target is None:
        msg = f"no skill named {competition.skill!r} in this corpus"
        raise ValueError(msg)
    nearest = competition.nearest_rival
    if nearest is None or nearest.score <= 0:
        return target, None
    rival = next((s for s in skills if s.name == nearest.name), None)
    return target, rival


def _evaluate_unclaimed_terms(
    target: Skill,
    skills: Sequence[Skill],
    body: str | None,
    ranker: Bm25Scorer,
    ceded: tuple[CededTerm, ...],
    limit: int,
    tokenized_corpus: Mapping[str, frozenset[str]] | None = None,
    *,
    settings: OverlapSettings | None = None,
) -> tuple[str, ...]:
    """Compute unclaimed terms if any ceded terms remain un-disclaimed."""
    if not any(not term.disclaimed for term in ceded):
        return ()
    body_text = skill_body(target) if body is None else body
    return unclaimed_terms(
        target,
        skills,
        body_text,
        ranker,
        limit=limit,
        tokenized_corpus=tokenized_corpus,
        settings=settings,
    )


def suggest_rewrite(
    competition: Competition,
    skills: Sequence[Skill],
    scorer: Bm25Scorer | None = None,
    body: str | None = None,
    limit: int = CLAIM_LIMIT,
    band: float = CONTENDER_BAND,
    share: float = MATERIAL_SHARE,
    tokenized_corpus: Mapping[str, frozenset[str]] | None = None,
    *,
    settings: OverlapSettings | None = None,
) -> Rewrite:
    """Generate Rewrite recommendation for a skill based on competitor scores."""
    effective_limit = (
        settings.claim_limit if settings is not None and limit == CLAIM_LIMIT else limit
    )
    effective_band = (
        settings.contender_band if settings is not None and band == CONTENDER_BAND else band
    )
    effective_share = (
        settings.material_share if settings is not None and share == MATERIAL_SHARE else share
    )
    target, rival = _resolve_target_and_rival(competition, skills)
    if rival is None:
        return Rewrite(skill=target.name, band=effective_band)

    nearest = competition.nearest_rival
    if nearest is None:
        return Rewrite(skill=target.name, band=effective_band)
    ranker = scorer or Bm25Scorer.from_skills(skills)
    tokenized = tokenized_corpus or {
        name: frozenset(tokens) for name, tokens in ranker.documents.items()
    }
    ceded = ceded_terms(target, rival, ranker, nearest.score, share=effective_share)
    unclaimed = _evaluate_unclaimed_terms(
        target,
        skills,
        body,
        ranker,
        ceded,
        effective_limit,
        tokenized_corpus=tokenized,
        settings=settings,
    )
    contenders = tuple(
        r.name for r in competition.ranked_rivals if r.score >= effective_band * nearest.score
    )
    from reach.lint import (
        _claims_neighbor_name_phrase,
        extract_skill_references,
        hands_off_to_skill,
    )

    by_name = {s.name: s for s in skills}
    target_refs = frozenset(extract_skill_references(target.description, self_name=target.name))
    rival_refs = frozenset(extract_skill_references(rival.description, self_name=rival.name))
    rival_disclaims = hands_off_to_skill(rival.description, target.name, rival_refs)
    target_has_boundaries = bool(target_refs) or any(t.disclaimed for t in ceded)

    missing_mutual: list[str] = []
    for contender_name in contenders:
        contender_skill = by_name.get(contender_name)
        if contender_skill is None:
            continue
        c_refs = frozenset(
            extract_skill_references(contender_skill.description, self_name=contender_name)
        )
        claims_phrase = _claims_neighbor_name_phrase(
            contender_skill, target
        ) or _claims_neighbor_name_phrase(target, contender_skill)
        if (target_has_boundaries or bool(c_refs) or claims_phrase) and not hands_off_to_skill(
            contender_skill.description, target.name, c_refs
        ):
            missing_mutual.append(contender_name)

    return Rewrite(
        skill=target.name,
        rival=rival.name,
        ceded=ceded,
        unclaimed=unclaimed,
        contenders=contenders,
        band=effective_band,
        rival_disclaims_target=rival_disclaims,
        missing_mutual_handoffs=tuple(missing_mutual),
    )


#: Standard advisory caveats accompanying suggested vocabulary rewrites.
REWRITE_CAVEAT = (
    *OVERLAP_CAVEAT,
    "Suggestions highlight candidate vocabulary terms, not complete replacement descriptions.",
)


def suggest_all(
    overlap: CorpusOverlap,
    skills: Sequence[Skill],
    names: Sequence[str] = (),
    *,
    only_actionable: bool = False,
    limit: int = CLAIM_LIMIT,
    band: float = CONTENDER_BAND,
    share: float = MATERIAL_SHARE,
    settings: OverlapSettings | None = None,
) -> tuple[Rewrite, ...]:
    """Generate rewrite proposals for multiple named skills using a shared scorer."""
    target_names = tuple(names) if names else tuple(c.skill for c in overlap.competitions)
    scorer = Bm25Scorer.from_skills(skills)
    tokenized = {name: frozenset(tokens) for name, tokens in scorer.documents.items()}
    rewrites = (
        suggest_rewrite(
            overlap.find(name),
            skills,
            scorer=scorer,
            limit=limit,
            band=band,
            share=share,
            tokenized_corpus=tokenized,
            settings=settings,
        )
        for name in target_names
    )
    if only_actionable:
        return tuple(r for r in rewrites if r.actionable)
    return tuple(rewrites)


def synthesize_directional_disclaimer(
    current_description: str,
    rival_skill: str,
    ceded_terms: Sequence[str] = (),
) -> str:
    """Synthesize a description with an explicit directional disclaimer to the rival skill."""
    cleaned = current_description.strip()
    if cleaned and cleaned[-1] not in ".!?":
        cleaned += "."

    if ceded_terms:
        terms_str = ", ".join(ceded_terms[:3])
        disclaimer = f"For {terms_str}, use {rival_skill} instead."
    else:
        disclaimer = f"For {rival_skill}-related tasks, use {rival_skill} instead."

    return f"{cleaned} {disclaimer}"


__all__ = [
    "REWRITE_CAVEAT",
    "CededTerm",
    "Rewrite",
    "Verdict",
    "suggest_all",
    "suggest_rewrite",
    "synthesize_directional_disclaimer",
]
