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

"""Provide dense semantic, sparse lexical, and hybrid retrieval scorers."""

from __future__ import annotations

import functools
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from enum import StrEnum
from math import log
from typing import TYPE_CHECKING, Any, Literal, Protocol, override, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from reach.config import RetrievalSettings

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from reach.config import RunConfig
    from reach.models import Skill

__all__ = [
    "Bm25Scorer",
    "DenseScorer",
    "HybridScorer",
    "OverlapQuadrant",
    "Scorer",
    "TextScorer",
    "build_scorer",
    "classify_overlap_quadrant",
    "compute_rrf",
    "cosine_similarity",
    "directional_projection",
    "skill_text",
    "tokenize",
]

_DEFAULT_RETRIEVAL = RetrievalSettings()

#: Lucene BM25 default parameter for term-frequency saturation.
K1 = _DEFAULT_RETRIEVAL.bm25_k1
#: Lucene BM25 default parameter for document length normalization.
B = _DEFAULT_RETRIEVAL.bm25_b

DEFAULT_RETRIEVAL_MODEL = _DEFAULT_RETRIEVAL.model
DEFAULT_RRF_K = _DEFAULT_RETRIEVAL.rrf_k
DEFAULT_SIMILARITY_THRESHOLD = _DEFAULT_RETRIEVAL.similarity_threshold

#: Tokenizer pattern matching words, numbers, and technical symbols (node.js, c++).
_TOKEN = re.compile(r"[^\W_]+(?:\.[^\W_]+)*[+#]*")

type EmbeddingVector = list[float]
type SimilarityMatrix = dict[tuple[str, str], float]
type DocumentPostings = dict[str, tuple[str, ...]]
type TermFrequencyTable = dict[str, Counter[str]]
type ScoredPair = tuple[str, float]
type PairwiseSimilarity = tuple[str, str, float]


def tokenize(text: str) -> list[str]:
    """Split text into lowercase alphanumeric and technical tokens."""
    return _TOKEN.findall(text.lower())


def skill_text(skill: Skill) -> str:
    """Combine a skill's name and description for lexical indexing."""
    return f"{skill.name} {skill.description}"


@runtime_checkable
class Scorer(Protocol):
    """Protocol for scoring candidate skills against a target skill."""

    def rank(
        self,
        target: Skill,
        candidates: Sequence[Skill],
    ) -> list[ScoredPair]:
        """Return candidate names paired with scores, strongest first."""
        ...


@runtime_checkable
class TextScorer(Protocol):
    """Protocol for scoring candidate skills against arbitrary query text."""

    def rank_text(
        self,
        text: str,
        candidates: Sequence[Skill],
    ) -> list[tuple[str, float]]:
        """Return candidate names paired with scores against query text, strongest first."""
        ...


def _lucene_idf(n: int, df: int) -> float:
    """Calculate the Lucene non-negative inverse document frequency."""
    return log(1 + (n - df + 0.5) / (df + 0.5)) if n else 0.0


def _build_postings(
    documents: Mapping[str, tuple[str, ...]],
) -> DocumentPostings:
    """Build an inverted index mapping terms to document names containing them."""
    postings: dict[str, list[str]] = defaultdict(list)
    for name, tokens in documents.items():
        for term in set(tokens):
            postings[term].append(name)
    return {term: tuple(docs) for term, docs in postings.items()}


def _document_frequencies(
    documents: Mapping[str, tuple[str, ...]],
) -> Counter[str]:
    """Count the number of documents in which each term appears."""
    return Counter(term for tokens in documents.values() for term in set(tokens))


@dataclass(frozen=True, slots=True)
class _CorpusTables:
    """Hold derived per-corpus term frequencies, document counts, IDF values, and inverted index."""

    document_frequency: Counter[str]
    term_frequency: TermFrequencyTable
    average_length: float
    idf: dict[str, float]
    postings: DocumentPostings

    @classmethod
    def of(cls, documents: Mapping[str, tuple[str, ...]]) -> _CorpusTables:
        """Construct corpus frequency tables and inverted index from tokenized documents."""
        n = len(documents)
        df_counts = _document_frequencies(documents)
        postings = _build_postings(documents)
        avg_len = sum(len(d) for d in documents.values()) / n if n else 0.0

        return cls(
            document_frequency=df_counts,
            term_frequency={name: Counter(tokens) for name, tokens in documents.items()},
            average_length=avg_len,
            idf={term: _lucene_idf(n, df) for term, df in df_counts.items()},
            postings=postings,
        )


def _matching_candidate_names(
    query: Sequence[str],
    candidate_names: set[str],
    postings: Mapping[str, tuple[str, ...]],
) -> set[str]:
    """Find candidate names that contain at least one query term."""
    matching: set[str] = set()
    for term in query:
        if term_docs := postings.get(term):
            matching.update(d for d in term_docs if d in candidate_names)
    return matching


class Bm25Scorer(BaseModel):
    """Score texts and skill descriptions using Lucene-variant BM25."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    documents: dict[str, tuple[str, ...]]
    k1: float = Field(default=K1, gt=0)
    b: float = Field(default=B, ge=0, le=1)

    _tables: _CorpusTables = PrivateAttr()

    @override
    def model_post_init(self, _context: object) -> None:
        """Derive cached corpus tables after initialization."""
        object.__setattr__(self, "_tables", _CorpusTables.of(self.documents))

    @classmethod
    def from_skills(
        cls,
        skills: Sequence[Skill],
        k1: float = K1,
        b: float = B,
    ) -> Bm25Scorer:
        """Instantiate a Bm25Scorer from a sequence of Skill models."""
        return cls(
            documents={s.name: tuple(tokenize(skill_text(s))) for s in skills},
            k1=k1,
            b=b,
        )

    @property
    def average_length(self) -> float:
        """Return the average document token length in the corpus."""
        return self._tables.average_length

    def idf(self, term: str) -> float:
        """Calculate the non-negative Lucene IDF for a term."""
        cached = self._tables.idf.get(term)
        if cached is not None:
            return cached
        df = self._tables.document_frequency[term]
        return _lucene_idf(len(self.documents), df)

    def score(self, query: Sequence[str], name: str) -> float:
        """Compute the Lucene BM25 score of a document against a tokenized query."""
        tokens = self.documents.get(name)
        if not tokens:
            return 0.0
        tables = self._tables
        counts = tables.term_frequency[name]
        idf = tables.idf
        norm = self.k1 * (1 - self.b + self.b * len(tokens) / tables.average_length)
        total = 0.0
        for term in query:
            tf = counts.get(term, 0)
            if tf:
                total += idf[term] * tf / (tf + norm)
        return total

    def contributions(self, query: Sequence[str], name: str) -> dict[str, float]:
        """Itemize BM25 score contributions for each matching query term."""
        tokens = self.documents.get(name)
        if not tokens:
            return {}
        tables = self._tables
        counts = tables.term_frequency[name]
        idf = tables.idf
        norm = self.k1 * (1 - self.b + self.b * len(tokens) / tables.average_length)
        asked = Counter(query)
        return {
            term: repeats * idf[term] * tf / (tf + norm)
            for term, repeats in asked.items()
            if (tf := counts.get(term, 0))
        }

    def rank_text(
        self,
        text: str,
        candidates: Sequence[Skill],
    ) -> list[tuple[str, float]]:
        """Rank candidate skills against query text, returning (name, score) pairs."""
        query = tokenize(text)
        if not query or not candidates:
            return sorted(((c.name, 0.0) for c in candidates), key=lambda pair: pair[0])

        matching_names = _matching_candidate_names(
            query,
            {c.name for c in candidates},
            self._tables.postings,
        )
        scored = [
            (c.name, self.score(query, c.name) if c.name in matching_names else 0.0)
            for c in candidates
        ]
        return sorted(scored, key=lambda pair: (-pair[1], pair[0]))

    def rank(
        self,
        target: Skill,
        candidates: Sequence[Skill],
    ) -> list[tuple[str, float]]:
        """Rank candidates against target skill vocabulary, excluding target itself."""
        return self.rank_text(
            skill_text(target),
            [c for c in candidates if c.name != target.name],
        )


def _unit_vector(vec: Sequence[float]) -> list[float]:
    """Normalize a vector to unit Euclidean length."""
    norm_sq = sum(x * x for x in vec)
    if norm_sq <= 0.0:
        return [0.0] * len(vec)
    inv_norm = 1.0 / math.sqrt(norm_sq)
    return [x * inv_norm for x in vec]


def cosine_similarity(v1: Sequence[float], v2: Sequence[float]) -> float:
    """Calculate the cosine similarity between two numeric vectors."""
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for a, b in zip(v1, v2, strict=False):
        dot += a * b
        norm_a += a * a
        norm_b += b * b

    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0
    raw = dot / (math.sqrt(norm_a) * math.sqrt(norm_b))
    return max(-1.0, min(1.0, raw))


def directional_projection(target: Sequence[float], candidate: Sequence[float]) -> float:
    """Calculate the directional projection of target onto candidate."""
    dot = 0.0
    norm_target_sq = 0.0
    for t, c in zip(target, candidate, strict=False):
        dot += t * c
        norm_target_sq += t * t

    if norm_target_sq <= 0.0:
        return 0.0
    return dot / norm_target_sq


def compute_rrf(
    rankings: Sequence[Sequence[str]],
    k: int = DEFAULT_RRF_K,
) -> list[tuple[str, float]]:
    """Fuse multiple ranked candidate name lists using Reciprocal Rank Fusion."""
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank_idx, name in enumerate(ranking, start=1):
            scores[name] += 1.0 / (k + rank_idx)

    # Sort descending by fused score; break ties deterministically by alphabetical name
    return sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))


@functools.lru_cache(maxsize=4)
def _load_model2vec_model(model_name: str) -> Any:  # noqa: ANN401 (optional dynamic dependency)
    """Load and cache a Model2Vec StaticModel instance."""
    import warnings

    try:
        from model2vec import (  # type: ignore[import-not-found,import-untyped,unresolved-import]
            StaticModel,
        )
    except ImportError as err:
        msg = (
            "model2vec is required for dense semantic scoring. "
            "Install it with: pip install 'skill-reach[semantic]'"
        )
        raise RuntimeError(msg) from err

    try:
        from huggingface_hub.utils import (  # type: ignore[import-not-found,import-untyped,unresolved-import]
            disable_progress_bars,
        )

        disable_progress_bars()
    except (ImportError, AttributeError):
        pass

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        return StaticModel.from_pretrained(model_name)


class DenseScorer(BaseModel):
    """Score skills using dense semantic embeddings."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    vectors: dict[str, EmbeddingVector] = Field(default_factory=dict)
    model_name: str = DEFAULT_RETRIEVAL_MODEL
    mode: Literal["cosine", "directional"] = "cosine"

    _unit_vectors: dict[str, EmbeddingVector] = PrivateAttr(default_factory=dict)
    _text_vectors: dict[str, list[float]] = PrivateAttr(default_factory=dict)

    @override
    def model_post_init(self, _context: object) -> None:
        """Compute and cache normalized unit vectors after initialization."""
        object.__setattr__(
            self,
            "_unit_vectors",
            {name: _unit_vector(vec) for name, vec in self.vectors.items()},
        )
        object.__setattr__(self, "_text_vectors", {})

    @classmethod
    def from_skills(
        cls,
        skills: Sequence[Skill],
        model_name: str | None = None,
        mode: Literal["cosine", "directional"] = "cosine",
    ) -> DenseScorer:
        """Embed skill texts and instantiate a DenseScorer."""
        chosen_model = model_name or DEFAULT_RETRIEVAL_MODEL
        model = _load_model2vec_model(chosen_model)

        texts = [skill_text(s) for s in skills]
        embeddings = model.encode(texts)

        vectors: dict[str, EmbeddingVector] = {}
        for skill, emb in zip(skills, embeddings, strict=False):
            vectors[skill.name] = (
                emb.tolist() if hasattr(emb, "tolist") else [float(x) for x in emb]
            )

        return cls(vectors=vectors, model_name=chosen_model, mode=mode)

    def _get_or_compute_vector(self, skill: Skill) -> EmbeddingVector:
        """Retrieve pre-computed vector or embed on demand if available."""
        vec = self.vectors.get(skill.name)
        if vec:
            return vec
        try:
            model = _load_model2vec_model(self.model_name)
            emb = model.encode([skill_text(skill)])[0]
            return emb.tolist() if hasattr(emb, "tolist") else [float(x) for x in emb]
        except (RuntimeError, ValueError, TypeError, AttributeError):
            return []

    def _get_or_compute_unit_vector(self, skill: Skill) -> EmbeddingVector:
        """Retrieve pre-computed unit vector or embed on demand."""
        unit = self._unit_vectors.get(skill.name)
        if unit is not None:
            return unit
        vec = self._get_or_compute_vector(skill)
        if not vec:
            return []
        unit_vec = _unit_vector(vec)
        self._unit_vectors[skill.name] = unit_vec
        return unit_vec

    def rank(
        self,
        target: Skill,
        candidates: Sequence[Skill],
    ) -> list[ScoredPair]:
        """Rank candidate skills against target using semantic similarity."""
        if self.mode == "directional":
            target_vec = self._get_or_compute_vector(target)
            if not target_vec:
                return [(c.name, 0.0) for c in candidates if c.name != target.name]

            scored = [
                (c.name, directional_projection(target_vec, self._get_or_compute_vector(c)))
                for c in candidates
                if c.name != target.name
            ]
            return sorted(scored, key=lambda pair: (-pair[1], pair[0]))

        target_u = self._get_or_compute_unit_vector(target)
        if not target_u:
            return [(c.name, 0.0) for c in candidates if c.name != target.name]

        scored = [
            (
                c.name,
                sum(a * b for a, b in zip(target_u, cand_u, strict=False))
                if (cand_u := self._get_or_compute_unit_vector(c))
                else 0.0,
            )
            for c in candidates
            if c.name != target.name
        ]
        return sorted(scored, key=lambda pair: (-pair[1], pair[0]))

    def pairwise_similarity(
        self,
        skills: Sequence[Skill],
    ) -> list[PairwiseSimilarity]:
        """Calculate pairwise cosine similarity for all distinct skill pairs."""
        pairs: list[PairwiseSimilarity] = []

        skill_list = list(skills)
        unit_vecs = {s.name: self._get_or_compute_unit_vector(s) for s in skill_list}

        for i in range(len(skill_list)):
            s1 = skill_list[i]
            u1 = unit_vecs.get(s1.name)
            if not u1:
                continue
            for j in range(i + 1, len(skill_list)):
                s2 = skill_list[j]
                u2 = unit_vecs.get(s2.name)
                if u2:
                    sim = sum(a * b for a, b in zip(u1, u2, strict=False))
                    pairs.append((s1.name, s2.name, sim))

        return sorted(pairs, key=lambda p: (-p[2], p[0], p[1]))

    def _get_or_compute_text_vector(self, text: str) -> list[float]:
        """Retrieve or compute the dense embedding vector for a given text string."""
        if not text:
            return []
        if text in self.vectors:
            return self.vectors[text]
        cached = self._text_vectors.get(text)
        if cached is not None:
            return cached
        try:
            model = _load_model2vec_model(self.model_name)
            query_vec = model.encode([text])[0]
            vec_list = (
                query_vec.tolist()
                if hasattr(query_vec, "tolist")
                else [float(x) for x in query_vec]
            )
            self._text_vectors[text] = vec_list
            return vec_list
        except (RuntimeError, ValueError, TypeError, AttributeError):
            return []

    def rank_text(
        self,
        text: str,
        candidates: Sequence[Skill],
    ) -> list[tuple[str, float]]:
        """Rank candidate skills against query text using semantic similarity."""
        if not text or not candidates:
            return sorted(((c.name, 0.0) for c in candidates), key=lambda pair: pair[0])

        query_vec_list = self._get_or_compute_text_vector(text)
        if not query_vec_list:
            return sorted(((c.name, 0.0) for c in candidates), key=lambda pair: pair[0])

        if self.mode == "directional":
            scored = [
                (
                    c.name,
                    directional_projection(query_vec_list, self._get_or_compute_vector(c)),
                )
                for c in candidates
            ]
            return sorted(scored, key=lambda pair: (-pair[1], pair[0]))

        query_u = _unit_vector(query_vec_list)
        scored = [
            (
                c.name,
                sum(a * b for a, b in zip(query_u, cand_u, strict=False))
                if (cand_u := self._get_or_compute_unit_vector(c))
                else 0.0,
            )
            for c in candidates
        ]
        return sorted(scored, key=lambda pair: (-pair[1], pair[0]))

    def score_query(self, query: str, skill: Skill) -> float:
        """Calculate semantic similarity between a query text and a skill."""
        query_vec_list = self._get_or_compute_text_vector(query)
        skill_u = self._get_or_compute_unit_vector(skill)
        if not skill_u or not query_vec_list:
            return 0.0
        query_u = _unit_vector(query_vec_list)
        return sum(a * b for a, b in zip(query_u, skill_u, strict=False))


class HybridScorer(BaseModel):
    """Fuse lexical BM25 and dense semantic rankings using Reciprocal Rank Fusion."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    lexical: Bm25Scorer
    semantic: DenseScorer | Scorer
    rrf_k: int = Field(default=DEFAULT_RRF_K, gt=0)

    @classmethod
    def from_skills(
        cls,
        skills: Sequence[Skill],
        model_name: str | None = None,
        rrf_k: int = DEFAULT_RRF_K,
        k1: float = K1,
        b: float = B,
    ) -> HybridScorer:
        """Instantiate both lexical and dense semantic scorers over skills."""
        lexical = Bm25Scorer.from_skills(skills, k1=k1, b=b)
        semantic = DenseScorer.from_skills(skills, model_name=model_name)
        return cls(lexical=lexical, semantic=semantic, rrf_k=rrf_k)

    @classmethod
    def from_skills_and_vectors(
        cls,
        skills: Sequence[Skill],
        vectors: dict[str, list[float]],
        rrf_k: int = DEFAULT_RRF_K,
        k1: float = K1,
        b: float = B,
    ) -> HybridScorer:
        """Instantiate a HybridScorer using precomputed vectors."""
        lexical = Bm25Scorer.from_skills(skills, k1=k1, b=b)
        semantic = DenseScorer(vectors=vectors)
        return cls(lexical=lexical, semantic=semantic, rrf_k=rrf_k)

    def _fuse_rankings(
        self,
        bm25_ranked: Sequence[tuple[str, float]],
        dense_ranked: Sequence[tuple[str, float]],
    ) -> list[tuple[str, float]]:
        """Fuse positive-score lexical rankings with dense rankings via Reciprocal Rank Fusion."""
        lex_ranks = [name for name, score in bm25_ranked if score > 0.0]
        sem_ranks = [name for name, _ in dense_ranked]
        return compute_rrf([lex_ranks, sem_ranks], k=self.rrf_k)

    def rank_text(
        self,
        text: str,
        candidates: Sequence[Skill],
    ) -> list[tuple[str, float]]:
        """Rank candidate skills against query text using Reciprocal Rank Fusion."""
        if not candidates:
            return []
        sem_ranked = (
            self.semantic.rank_text(text, candidates)
            if isinstance(self.semantic, DenseScorer | TextScorer)
            else []
        )
        return self._fuse_rankings(
            self.lexical.rank_text(text, candidates),
            sem_ranked,
        )

    def rank(
        self,
        target: Skill,
        candidates: Sequence[Skill],
    ) -> list[tuple[str, float]]:
        """Rank candidate skills using Reciprocal Rank Fusion."""
        pool = [c for c in candidates if c.name != target.name]
        if not pool:
            return []
        return self._fuse_rankings(
            self.lexical.rank(target, pool),
            self.semantic.rank(target, pool),
        )


def build_scorer(
    name: str,
    skills: Sequence[Skill],
    config: RunConfig | None = None,
) -> Scorer:
    """Build a Scorer instance according to the configured retrieval strategy."""
    scorer_type = name.lower().strip()
    k1 = config.retrieval.bm25_k1 if config else K1
    b = config.retrieval.bm25_b if config else B
    match scorer_type:
        case "bm25":
            return Bm25Scorer.from_skills(skills, k1=k1, b=b)
        case "dense":
            model_name = config.retrieval.model if config else DEFAULT_RETRIEVAL_MODEL
            return DenseScorer.from_skills(skills, model_name=model_name)
        case "hybrid":
            model_name = config.retrieval.model if config else DEFAULT_RETRIEVAL_MODEL
            rrf_k = config.retrieval.rrf_k if config else DEFAULT_RRF_K
            try:
                return HybridScorer.from_skills(
                    skills, model_name=model_name, rrf_k=rrf_k, k1=k1, b=b
                )
            except RuntimeError:
                return Bm25Scorer.from_skills(skills, k1=k1, b=b)
        case _:
            msg = f"unknown scorer: {name!r}; valid choices: 'bm25', 'dense', 'hybrid'"
            raise ValueError(msg)


#: Minimum prefix token length for single-word quadrant resolution.
_MIN_QUADRANT_PREFIX_LEN: int = 3


class OverlapQuadrant(StrEnum):
    """Classify the diagnostic quadrant between lexical and semantic overlap."""

    NEAR_DUPLICATE = "Near-Duplicate"
    BOILERPLATE = "Boilerplate / Style"
    LATENT_COLLISION = "Latent Collision"
    DISTINCT = "Distinct"

    @classmethod
    def _missing_(cls, value: object) -> OverlapQuadrant | None:
        if not isinstance(value, str):
            return None
        tokens = tuple(tokenize(value.replace("_", "-")))
        if not tokens:
            return None
        for member in cls:
            member_name_tokens = tuple(tokenize(member.name.replace("_", "-")))
            if tokens in (member_name_tokens, tuple(tokenize(member.value))):
                return member
        if len(tokens) == 1 and len(tokens[0]) >= _MIN_QUADRANT_PREFIX_LEN:
            prefix_matches = [
                m
                for m in cls
                if (val_tokens := tokenize(m.value)) and val_tokens[0].startswith(tokens[0])
            ]
            if len(prefix_matches) == 1:
                return prefix_matches[0]
        return None


def classify_overlap_quadrant(
    lexical_ratio: float,
    semantic_similarity: float,
    lex_high: float = 0.5,
    sem_high: float = 0.75,
) -> OverlapQuadrant:
    """Classify the relationship between lexical and semantic overlap into a diagnostic quadrant."""
    match (lexical_ratio >= lex_high, semantic_similarity >= sem_high):
        case (True, True):
            return OverlapQuadrant.NEAR_DUPLICATE
        case (True, False):
            return OverlapQuadrant.BOILERPLATE
        case (False, True):
            return OverlapQuadrant.LATENT_COLLISION
        case _:
            return OverlapQuadrant.DISTINCT
