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

"""Test dense semantic and hybrid retrieval scorers and RRF fusion."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import reach.retrieval
from reach.models import Skill
from reach.retrieval import (
    Bm25Scorer,
    DenseScorer,
    HybridScorer,
    build_scorer,
    compute_rrf,
    cosine_similarity,
    directional_projection,
)


def _make_skill(name: str, desc: str) -> Skill:
    """Create a minimal Skill object for testing."""
    return Skill(
        name=name,
        description=desc,
        path=Path(f"/skills/{name}"),
    )


_OVERSHOOT_VEC = [
    -5.627240503927933,
    0.10710576206724731,
    -9.469280606322727,
    -6.02324698626703,
    2.997688755590463,
    0.8988296120643327,
    -5.591187559186066,
    1.7853136775181753,
    6.1886091335565325,
    -9.87002480643878,
]


@pytest.mark.parametrize(
    ("v1", "v2", "expected"),
    [
        ([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], 1.0),
        ([1.0, 0.0], [0.0, 1.0], 0.0),
        ([0.0, 0.0], [1.0, 2.0], 0.0),
        (_OVERSHOOT_VEC, _OVERSHOOT_VEC, 1.0),
        (_OVERSHOOT_VEC, [-x for x in _OVERSHOOT_VEC], -1.0),
    ],
)
def test_cosine_similarity(v1: list[float], v2: list[float], expected: float) -> None:
    """Verify cosine similarity handles identical, orthogonal, zero, and overshoot vectors."""
    sim = cosine_similarity(v1, v2)
    assert -1.0 <= sim <= 1.0
    assert pytest.approx(sim) == expected


def test_directional_projection_asymmetry() -> None:
    """Verify directional projection measures proportion of target covered by candidate."""
    # Target is specialized: [1.0, 0.0]
    # Candidate is broad: [1.0, 1.0]
    target = [1.0, 0.0]
    candidate = [1.0, 1.0]

    # Target projecting onto candidate
    proj_t_to_c = directional_projection(target, candidate)
    # Candidate projecting onto target
    proj_c_to_t = directional_projection(candidate, target)

    assert proj_t_to_c != proj_c_to_t
    assert pytest.approx(proj_t_to_c) == 1.0  # Target vector fully covered by candidate
    assert pytest.approx(proj_c_to_t) == 0.5  # Candidate vector partially covered by target


def test_compute_rrf_fuses_rankings() -> None:
    """Verify Reciprocal Rank Fusion combines two independent ranking lists."""
    # List 1: A (rank 1), B (rank 2), C (rank 3)
    # List 2: B (rank 1), C (rank 2), A (rank 3)
    rankings = [
        ["skill-a", "skill-b", "skill-c"],
        ["skill-b", "skill-c", "skill-a"],
    ]
    fused = compute_rrf(rankings, k=60)

    # Expected scores:
    # skill-b: 1/(60+2) + 1/(60+1) = 1/62 + 1/61 ≈ 0.016129 + 0.016393 = 0.032522
    # skill-a: 1/(60+1) + 1/(60+3) = 1/61 + 1/63 ≈ 0.016393 + 0.015873 = 0.032266
    # skill-c: 1/(60+3) + 1/(60+2) = 1/63 + 1/62 ≈ 0.015873 + 0.016129 = 0.032002
    assert fused[0][0] == "skill-b"
    assert fused[1][0] == "skill-a"
    assert fused[2][0] == "skill-c"


def test_compute_rrf_tie_breaking_alphabetical() -> None:
    """Verify RRF breaks ties deterministically using alphabetical skill name."""
    rankings = [
        ["skill-z", "skill-a"],
        ["skill-a", "skill-z"],
    ]
    fused = compute_rrf(rankings, k=60)
    # Both have identical RRF score (1/61 + 1/62), so skill-a should precede skill-z
    assert fused[0][0] == "skill-a"
    assert fused[1][0] == "skill-z"


def test_dense_scorer_with_precomputed_vectors() -> None:
    """Verify DenseScorer ranks candidates using vector similarity."""
    target = _make_skill("pdf-parser", "Extract tables from PDF documents.")
    cand1 = _make_skill("doc-extractor", "Parse structured tables from PDF files.")
    cand2 = _make_skill("image-editor", "Crop and rotate JPEG photos.")

    vectors = {
        "pdf-parser": [1.0, 0.9, 0.0],
        "doc-extractor": [0.95, 0.85, 0.0],
        "image-editor": [0.0, 0.1, 1.0],
    }

    scorer = DenseScorer(vectors=vectors)
    ranked = scorer.rank(target, [cand1, cand2])

    assert len(ranked) == 2
    assert ranked[0][0] == "doc-extractor"
    assert ranked[1][0] == "image-editor"
    assert ranked[0][1] > ranked[1][1]


def test_dense_scorer_pairwise_similarity() -> None:
    """Verify DenseScorer computes pairwise cosine similarity across skills."""
    skills = [
        _make_skill("skill-a", "Skill A description"),
        _make_skill("skill-b", "Skill B description"),
        _make_skill("skill-c", "Skill C description"),
    ]
    vectors = {
        "skill-a": [1.0, 0.0],
        "skill-b": [0.99, 0.05],  # High cosine similarity to skill-a
        "skill-c": [0.0, 1.0],  # Orthogonal vector to skill-a
    }
    scorer = DenseScorer(vectors=vectors)
    pairs = scorer.pairwise_similarity(skills)

    # Pairs are sorted by similarity descending
    assert len(pairs) == 3  # (a,b), (a,c), (b,c)
    top_pair = pairs[0]
    assert {top_pair[0], top_pair[1]} == {"skill-a", "skill-b"}
    assert top_pair[2] > 0.95


def test_hybrid_scorer_fuses_lexical_and_dense() -> None:
    """Verify HybridScorer captures both lexical keyword matches and semantic matches."""
    target = _make_skill("pdf-tables", "Extract tabular data from PDF files.")
    # Lexical match (shares keywords "extract", "pdf")
    lexical_rival = _make_skill("pdf-tool", "Extract text strings from PDF documents.")
    # Semantic match (different words: "parse", "spreadsheets", but conceptually identical)
    semantic_rival = _make_skill("sheet-parser", "Convert document tables into spreadsheets.")
    # Unrelated
    unrelated = _make_skill("audio-player", "Play mp3 audio streams.")

    skills = [target, lexical_rival, semantic_rival, unrelated]

    # Provide vectors where sheet-parser is semantically close to pdf-tables
    vectors = {
        "pdf-tables": [1.0, 0.8, 0.0],
        "sheet-parser": [0.95, 0.80, 0.0],
        "pdf-tool": [0.5, 0.2, 0.0],
        "audio-player": [0.0, 0.0, 1.0],
    }

    hybrid = HybridScorer.from_skills_and_vectors(
        skills=skills,
        vectors=vectors,
        rrf_k=60,
    )
    candidates = [lexical_rival, semantic_rival, unrelated]
    ranked = hybrid.rank(target, candidates)

    assert len(ranked) == 3
    # Both lexical and semantic rivals should rank above unrelated
    ranked_names = [name for name, _ in ranked]
    assert ranked_names[2] == "audio-player"
    assert set(ranked_names[:2]) == {"pdf-tool", "sheet-parser"}


def test_hybrid_scorer_no_alphabetical_rrf_bias_for_zero_bm25_matches() -> None:
    """Verify zero-BM25 candidates receive 0 lexical RRF points regardless of alphabetical order."""
    target = _make_skill("cloud-orchestrator", "Deploy and manage container workloads.")
    # Lexical match (shares keywords "deploy", "workloads")
    cand_lex = _make_skill("workload-deployer", "Deploy serverless workloads.")
    # Pure semantic match (no keyword overlap at all, but vector close)
    cand_sem = _make_skill("provision-compute", "Setup virtual machines, host hypervisors.")
    # Irrelevant skills with alphabetically early names and zero keyword overlap
    distractors = [_make_skill(f"aaa-{i}", f"Irrelevant unrelated topic {i}.") for i in range(5)]

    skills = [target, cand_lex, cand_sem, *distractors]
    vectors = {
        "cloud-orchestrator": [1.0, 0.9, 0.0],
        "provision-compute": [0.95, 0.85, 0.0],
        "workload-deployer": [0.7, 0.6, 0.0],
    }
    for i, d in enumerate(distractors):
        vectors[d.name] = [0.0, 0.0, float(i + 1)]

    hybrid = HybridScorer.from_skills_and_vectors(skills, vectors, rrf_k=60)
    ranked = hybrid.rank(target, [cand_lex, cand_sem, *distractors])
    ranked_names = [name for name, _ in ranked]

    # cand_sem should strictly outrank all zero-match distractors
    for d in distractors:
        assert ranked_names.index("provision-compute") < ranked_names.index(d.name)
    assert set(ranked_names[:2]) == {"provision-compute", "workload-deployer"}


def test_build_scorer_factory() -> None:
    """Verify build_scorer constructs requested scorer variants."""
    skills = [
        _make_skill("tool-1", "Tool 1 description"),
        _make_skill("tool-2", "Tool 2 description"),
    ]
    bm25 = build_scorer("bm25", skills)
    assert bm25.__class__.__name__ == "Bm25Scorer"

    with pytest.raises(ValueError, match="unknown scorer"):
        build_scorer("non-existent-scorer", skills)


def test_build_neighborhood_catalogs_with_hybrid_scorer() -> None:
    """Verify build_neighborhood_catalogs respects hybrid scorer rankings."""
    from reach.catalog import build_neighborhood_catalogs

    target = _make_skill("pdf-tables", "Extract tabular data from PDF files.")
    lexical_rival = _make_skill("pdf-tool", "Extract text strings from PDF documents.")
    semantic_rival = _make_skill("sheet-parser", "Convert document tables into spreadsheets.")
    filler = _make_skill("audio-player", "Play mp3 audio streams.")

    skills = [target, lexical_rival, semantic_rival, filler]
    vectors = {
        "pdf-tables": [1.0, 0.8, 0.0],
        "sheet-parser": [0.95, 0.80, 0.0],
        "pdf-tool": [0.5, 0.2, 0.0],
        "audio-player": [0.0, 0.0, 1.0],
    }
    hybrid = HybridScorer.from_skills_and_vectors(skills, vectors, rrf_k=60)
    catalogs = build_neighborhood_catalogs(skills, size=3, rivals=2, scorer=hybrid)
    target_catalog = next(c for c in catalogs if c.id == "neighborhood:pdf-tables")
    # Verify target and both rival categories (lexical and semantic) are included
    assert set(target_catalog.skills) == {"pdf-tables", "pdf-tool", "sheet-parser"}


def test_classify_overlap_quadrant() -> None:
    """Verify dual-axis overlap quadrant classification."""
    from reach.retrieval import classify_overlap_quadrant

    assert classify_overlap_quadrant(0.8, 0.9) == "Near-Duplicate"
    assert classify_overlap_quadrant(0.7, 0.4) == "Boilerplate / Style"
    assert classify_overlap_quadrant(0.2, 0.85) == "Latent Collision"
    assert classify_overlap_quadrant(0.1, 0.3) == "Distinct"


def test_build_scorer_dense_missing_model2vec_raises_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify dense build_scorer raises RuntimeError with pip tip when model2vec is absent."""
    skills = [_make_skill("s1", "desc1")]

    def mock_load(model_name: str) -> Any:
        msg = (
            "model2vec is required for dense semantic scoring. "
            "Install it with: pip install 'skill-reach[semantic]'"
        )
        raise RuntimeError(msg)

    monkeypatch.setattr(reach.retrieval, "_load_model2vec_model", mock_load)

    with pytest.raises(RuntimeError, match=r"pip install 'skill-reach\[semantic\]'"):
        build_scorer("dense", skills)


def test_build_scorer_hybrid_missing_model2vec_falls_back_to_bm25(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify build_scorer('hybrid') falls back to Bm25Scorer when model2vec is missing."""
    skills = [_make_skill("s1", "desc1")]

    def mock_from_skills(*args: Any, **kwargs: Any) -> Any:
        msg = "model2vec is required"
        raise RuntimeError(msg)

    monkeypatch.setattr(reach.retrieval.HybridScorer, "from_skills", mock_from_skills)

    scorer = build_scorer("hybrid", skills)
    assert scorer.__class__.__name__ == "Bm25Scorer"


def test_dense_scorer_zero_vector_handling() -> None:
    """Verify DenseScorer handles zero-length vectors without dividing by zero."""
    s1 = _make_skill("s1", "desc1")
    s2 = _make_skill("s2", "desc2")
    scorer = DenseScorer(vectors={"s1": [0.0, 0.0], "s2": [1.0, 1.0]})

    pairs = scorer.pairwise_similarity([s1, s2])
    assert len(pairs) == 1
    assert pairs[0][2] == 0.0

    ranked = scorer.rank(s1, [s2])
    assert ranked == [("s2", 0.0)]


def test_dense_scorer_rank_text() -> None:
    """Verify DenseScorer ranks candidates against text queries."""
    cand1 = _make_skill("pdf-parser", "Extract tables from PDF files.")
    cand2 = _make_skill("image-editor", "Crop and rotate images.")
    vectors = {
        "pdf-query": [1.0, 0.9, 0.0],
        "pdf-parser": [0.95, 0.85, 0.0],
        "image-editor": [0.0, 0.1, 1.0],
    }
    scorer = DenseScorer(vectors=vectors)
    ranked = scorer.rank_text("pdf-query", [cand1, cand2])
    assert len(ranked) == 2
    assert ranked[0][0] == "pdf-parser"
    assert ranked[1][0] == "image-editor"
    assert ranked[0][1] > ranked[1][1]


def test_hybrid_scorer_rank_text() -> None:
    """Verify HybridScorer fuses lexical and dense scores for text queries."""
    cand1 = _make_skill("pdf-parser", "Extract tables from PDF files.")
    cand2 = _make_skill("image-editor", "Crop and rotate images.")
    vectors = {
        "pdf": [1.0, 0.9, 0.0],
        "pdf-parser": [0.95, 0.85, 0.0],
        "image-editor": [0.0, 0.1, 1.0],
    }
    lexical = Bm25Scorer.from_skills([cand1, cand2])
    dense = DenseScorer(vectors=vectors)
    hybrid = HybridScorer(lexical=lexical, semantic=dense)

    ranked = hybrid.rank_text("pdf", [cand1, cand2])
    assert len(ranked) == 2
    assert ranked[0][0] == "pdf-parser"
    assert ranked[1][0] == "image-editor"


def test_dense_scorer_memoizes_query_text_vectors(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify DenseScorer caches dynamic query text embeddings across repeated rank_text calls."""
    import reach.retrieval as retrieval_mod

    encode_calls: list[list[str]] = []

    class _FakeModel:
        def encode(self, texts: list[str]) -> list[list[float]]:
            encode_calls.append(list(texts))
            return [[1.0, 0.5, 0.0] for _ in texts]

    monkeypatch.setattr(retrieval_mod, "_load_model2vec_model", lambda _name: _FakeModel())
    cand = _make_skill("pdf-parser", "Extract tables from PDF files.")
    scorer = DenseScorer(vectors={"pdf-parser": [1.0, 0.5, 0.0]})

    scorer.rank_text("parse my invoice pdf", [cand])
    scorer.rank_text("parse my invoice pdf", [cand])
    scorer.score_query("parse my invoice pdf", cand)

    assert len(encode_calls) == 1


def test_load_model2vec_model_offline_and_import_errors(
    monkeypatch: pytest.MonkeyPatch,
    orig_load_model2vec: Any,
) -> None:
    """Verify _load_model2vec_model handles fake model2vec/hf modules and missing model2vec."""
    import sys
    import types

    monkeypatch.setitem(sys.modules, "model2vec", None)
    with pytest.raises(RuntimeError, match="model2vec is required"):
        orig_load_model2vec("missing-model")
    orig_load_model2vec.cache_clear()

    fake_m2v: Any = types.ModuleType("model2vec")
    fake_static: Any = types.SimpleNamespace(from_pretrained=lambda name: {"loaded": name})
    fake_m2v.StaticModel = fake_static
    monkeypatch.setitem(sys.modules, "model2vec", fake_m2v)

    # Case 1: huggingface_hub.utils missing (ImportError suppressed)
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    monkeypatch.setitem(sys.modules, "huggingface_hub.utils", None)
    assert orig_load_model2vec("m1") == {"loaded": "m1"}

    # Case 2: huggingface_hub.utils present with disable_progress_bars
    orig_load_model2vec.cache_clear()
    disabled: list[bool] = []
    fake_hf: Any = types.ModuleType("huggingface_hub")
    fake_hf_utils: Any = types.ModuleType("huggingface_hub.utils")
    fake_hf_utils.disable_progress_bars = lambda: disabled.append(True)
    fake_hf.utils = fake_hf_utils
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)
    monkeypatch.setitem(sys.modules, "huggingface_hub.utils", fake_hf_utils)
    assert orig_load_model2vec("m2") == {"loaded": "m2"}
    assert disabled == [True]


def test_dense_and_hybrid_scorer_directional_and_fallback_branches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify directional projection, on-demand vector fallback, and empty-input guards."""
    import reach.retrieval as retrieval_mod

    assert directional_projection([0.0, 0.0], [1.0, 2.0]) == 0.0

    s1 = _make_skill("s1", "First skill")
    s2 = _make_skill("s2", "Second skill")

    # On-demand vector computation via _load_model2vec_model
    class _ArrayLike:
        def __init__(self, vals: list[float]) -> None:
            self._vals = vals

        def tolist(self) -> list[float]:
            return list(self._vals)

    class _FakeModel:
        def encode(self, texts: list[str]) -> list[_ArrayLike]:
            return [_ArrayLike([1.0, 0.5]) for _ in texts]

    monkeypatch.setattr(retrieval_mod, "_load_model2vec_model", lambda _m: _FakeModel())
    from_skills_scorer = DenseScorer.from_skills([s1, s2], mode="directional")
    assert from_skills_scorer.rank(s1, [s1, s2]) == [("s2", 1.0)]
    assert from_skills_scorer.rank_text("query text", [s1, s2])[0][1] > 0.0
    assert from_skills_scorer.rank_text("", [s1, s2]) == [("s1", 0.0), ("s2", 0.0)]

    # Successful on-demand embedding when vectors={} is initially empty
    ondemand_dir = DenseScorer(vectors={}, mode="directional")
    assert ondemand_dir.rank(s1, [s1, s2]) == [("s2", 1.0)]
    ondemand_cos = DenseScorer(vectors={}, mode="cosine")
    assert ondemand_cos.rank(s1, [s1, s2])[0][0] == "s2"

    # On-demand fallback when _load_model2vec_model raises RuntimeError
    def _fail_load(_m: str) -> Any:
        msg = "offline"
        raise RuntimeError(msg)

    monkeypatch.setattr(retrieval_mod, "_load_model2vec_model", _fail_load)
    empty_dir_scorer = DenseScorer(vectors={}, mode="directional")
    assert empty_dir_scorer.rank(s1, [s2]) == [("s2", 0.0)]
    assert empty_dir_scorer.rank_text("unseen", [s2]) == [("s2", 0.0)]
    assert empty_dir_scorer.score_query("unseen", s2) == 0.0

    empty_cos_scorer = DenseScorer(vectors={})
    assert empty_cos_scorer.rank(s1, [s2]) == [("s2", 0.0)]
    assert empty_cos_scorer.pairwise_similarity([s1, s2]) == []

    hybrid = HybridScorer.from_skills_and_vectors([s1], {"s1": [1.0, 0.0]})
    assert hybrid.rank_text("q", []) == []
    assert hybrid.rank(s1, [s1]) == []


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (123, None),
        ("---", None),
        ("near_duplicate", "Near-Duplicate"),
        ("near", "Near-Duplicate"),
        ("lat", "Latent Collision"),
        ("zzz", None),
    ],
)
def test_overlap_quadrant_missing(
    raw: Any,
    expected: str | None,
) -> None:
    """Verify OverlapQuadrant._missing_ normalizes aliases and rejects unknown inputs."""
    from reach.retrieval import OverlapQuadrant

    if expected is None:
        assert OverlapQuadrant._missing_(raw) is None
    else:
        assert OverlapQuadrant(raw) == expected
