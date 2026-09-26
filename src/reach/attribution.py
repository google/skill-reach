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

"""Analyze per-token lexical attribution to diagnose skill misrouting."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from reach.leak import FUNCTION_WORDS
from reach.retrieval import tokenize

if TYPE_CHECKING:
    from reach.retrieval import Bm25Scorer

type TokenScoreMap = dict[str, float]
type DiagnosticRoles = tuple[TokenAttribution, ...]


class TokenAttribution(BaseModel):
    """Represent an individual query token's scoring bias between two skills."""

    model_config = ConfigDict(frozen=True)

    token: str
    target_score: float = Field(ge=0.0)
    rival_score: float = Field(ge=0.0)
    delta: float
    is_target_gap: bool = False


class QueryAttribution(BaseModel):
    """Diagnose token-level BM25 contributions driving a query towards a rival skill."""

    model_config = ConfigDict(frozen=True)

    query_text: str
    target_skill: str
    rival_skill: str
    target_total: float = Field(ge=0.0)
    rival_total: float = Field(ge=0.0)
    net_bias: float
    drivers: tuple[TokenAttribution, ...] = ()
    anchors: tuple[TokenAttribution, ...] = ()
    unscored_tokens: tuple[str, ...] = ()
    target_semantic: float | None = None
    rival_semantic: float | None = None
    semantic_bias: float | None = None


def attribute_query(
    query_text: str,
    target_skill: str,
    rival_skill: str | None,
    scorer: Bm25Scorer,
    *,
    target_semantic: float | None = None,
    rival_semantic: float | None = None,
) -> QueryAttribution:
    """Compute per-token BM25 score contributions for target vs rival skill.

    Args:
        query_text: The user's query or prompt string.
        target_skill: The ground-truth expected skill name.
        rival_skill: The rival skill name, or None / '(no selection)' for abstention.
        scorer: Bm25Scorer indexed over the resident skill corpus.
        target_semantic: Optional dense semantic similarity for target skill.
        rival_semantic: Optional dense semantic similarity for rival skill.

    Returns:
        A validated QueryAttribution diagnostic report.
    """
    tokens = tokenize(query_text)
    unique_tokens = tuple(dict.fromkeys(tokens))
    target_contribs = {
        k: v
        for k, v in scorer.contributions(tokens, target_skill).items()
        if k not in FUNCTION_WORDS
    }

    is_abstention = rival_skill is None or rival_skill.lower() in (
        "none",
        "(no selection)",
        "(no skill)",
        "",
    )
    rival_name = "(no selection)" if is_abstention or rival_skill is None else rival_skill
    rival_contribs = (
        {}
        if is_abstention
        else {
            k: v
            for k, v in scorer.contributions(tokens, rival_name).items()
            if k not in FUNCTION_WORDS
        }
    )

    scoring_keys = set(target_contribs.keys()) | set(rival_contribs.keys())
    drivers: list[TokenAttribution] = []
    anchors: list[TokenAttribution] = []

    for token in scoring_keys:
        t_score = target_contribs.get(token, 0.0)
        r_score = rival_contribs.get(token, 0.0)
        delta = r_score - t_score
        is_gap = t_score == 0.0 and r_score > 0.0
        attr = TokenAttribution(
            token=token,
            target_score=t_score,
            rival_score=r_score,
            delta=delta,
            is_target_gap=is_gap,
        )
        if delta > 0.0:
            drivers.append(attr)
        elif delta < 0.0:
            anchors.append(attr)

    drivers.sort(key=lambda item: (-item.delta, item.token))
    anchors.sort(key=lambda item: (item.delta, item.token))
    unscored = tuple(t for t in unique_tokens if t not in scoring_keys)

    target_total = sum(target_contribs.values())
    rival_total = sum(rival_contribs.values())

    sem_bias = (
        (rival_semantic - target_semantic)
        if (target_semantic is not None and rival_semantic is not None)
        else None
    )

    return QueryAttribution(
        query_text=query_text,
        target_skill=target_skill,
        rival_skill=rival_name,
        target_total=target_total,
        rival_total=rival_total,
        net_bias=rival_total - target_total,
        drivers=tuple(drivers),
        anchors=tuple(anchors),
        unscored_tokens=unscored,
        target_semantic=target_semantic,
        rival_semantic=rival_semantic,
        semantic_bias=sem_bias,
    )
