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

"""Validate pass-rate drop decomposition into context dilution and skill shadowing components."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from reach.metrics import DecompositionResult, decompose_pass_rate_drop

if TYPE_CHECKING:
    from collections.abc import Callable

    from reach.models import ProbeResult, Query


def test_decomposition_happy_path(
    make_paired_results: Callable[..., tuple[list[ProbeResult], list[ProbeResult], list[Query]]],
) -> None:
    """Verify strict mathematical equality delta = delta_ctx + delta_shd and expected values."""
    base, scaled, queries = make_paired_results(
        both_pass=5,
        both_fail=2,
        ctx_loss=3,
        shd_loss=4,
    )
    result = decompose_pass_rate_drop(base, scaled, queries=queries, seed=42)

    assert isinstance(result, DecompositionResult)
    assert result.sample_size == 14
    assert math.isclose(result.baseline_pass_rate, 12 / 14, rel_tol=1e-9)
    assert math.isclose(result.scaled_pass_rate, 5 / 14, rel_tol=1e-9)
    assert math.isclose(result.delta_total, 7 / 14, rel_tol=1e-9)
    assert math.isclose(result.delta_abstention, 3 / 14, rel_tol=1e-9)
    assert math.isclose(result.delta_collision, 4 / 14, rel_tol=1e-9)

    # Total drop must equal abstention plus collision drops
    assert math.isclose(
        result.delta_total, result.delta_abstention + result.delta_collision, abs_tol=1e-9
    )


def test_decomposition_zero_delta(
    make_paired_results: Callable[..., tuple[list[ProbeResult], list[ProbeResult], list[Query]]],
) -> None:
    """Verify that identical arm performances produce exact zero deltas."""
    base, scaled, queries = make_paired_results(
        both_pass=8,
        both_fail=4,
        ctx_loss=0,
        shd_loss=0,
    )
    result = decompose_pass_rate_drop(base, scaled, queries=queries, seed=42)

    assert result.delta_total == 0.0
    assert result.delta_abstention == 0.0
    assert result.delta_collision == 0.0
    assert result.delta_total_ci[0] <= 0.0 <= result.delta_total_ci[1]


def test_decomposition_multi_attempt_replicates(
    make_paired_results: Callable[..., tuple[list[ProbeResult], list[ProbeResult], list[Query]]],
) -> None:
    """Verify multi-attempt replicates (A=3) satisfy exact delta sum invariant."""
    base, scaled, queries = make_paired_results(
        both_pass=4,
        both_fail=3,
        ctx_loss=3,
        shd_loss=2,
        attempts=3,
    )
    result = decompose_pass_rate_drop(base, scaled, queries=queries, seed=42)

    assert math.isclose(
        result.delta_total, result.delta_abstention + result.delta_collision, abs_tol=1e-9
    )
    assert result.sample_size == 12


def test_decomposition_bootstrap_confidence_intervals(
    make_paired_results: Callable[..., tuple[list[ProbeResult], list[ProbeResult], list[Query]]],
) -> None:
    """Verify 95% bootstrap intervals enclose point estimates with proper ordering."""
    base, scaled, _ = make_paired_results(
        both_pass=15,
        both_fail=5,
        ctx_loss=10,
        shd_loss=10,
        attempts=1,
    )
    result = decompose_pass_rate_drop(base, scaled, iterations=500, seed=123)

    assert result.delta_total_ci[0] <= result.delta_total <= result.delta_total_ci[1]
    assert result.delta_abstention_ci[0] <= result.delta_abstention <= result.delta_abstention_ci[1]
    assert result.delta_collision_ci[0] <= result.delta_collision <= result.delta_collision_ci[1]


def test_decomposition_empty_results() -> None:
    """Verify empty inputs gracefully return zeroed decomposition results."""
    result = decompose_pass_rate_drop([], [])
    assert result.sample_size == 0
    assert result.delta_total == 0.0
    assert result.delta_abstention == 0.0
    assert result.delta_collision == 0.0
    assert result.delta_total_ci == (0.0, 0.0)


def test_decomposition_disjoint_queries_returns_zeroed_result(
    make_paired_results: Callable[..., tuple[list[ProbeResult], list[ProbeResult], list[Query]]],
) -> None:
    """Verify disjoint query sets between baseline and scaled arms return a zeroed result."""
    base, scaled, _ = make_paired_results(both_pass=2, both_fail=1, ctx_loss=1, shd_loss=1)
    disjoint_scaled = [r.model_copy(update={"query_id": f"disjoint-{r.query_id}"}) for r in scaled]
    result = decompose_pass_rate_drop(base, disjoint_scaled)
    assert result.sample_size == 0
    assert result.baseline_pass_rate == 0.0
    assert result.scaled_pass_rate == 0.0
    assert result.delta_total == 0.0
    assert result.delta_abstention == 0.0
    assert result.delta_collision == 0.0


def test_decomposition_bootstrap_matches_scipy_reference(
    make_paired_results: Callable[..., tuple[list[ProbeResult], list[ProbeResult], list[Query]]],
) -> None:
    """Verify bootstrap intervals are consistent with scipy reference bootstrap distribution."""
    import numpy as np
    from scipy.stats import bootstrap

    base, scaled, queries = make_paired_results(
        both_pass=20,
        both_fail=5,
        ctx_loss=10,
        shd_loss=15,
    )
    result = decompose_pass_rate_drop(base, scaled, queries=queries, iterations=1000, seed=42)

    base_dict = {r.query_id: (r.predicted_label == "skill-a") for r in base}
    scaled_dict = {r.query_id: (r.predicted_label == "skill-a") for r in scaled}
    deltas = np.array([float(base_dict[q.id]) - float(scaled_dict[q.id]) for q in queries])

    scipy_res = bootstrap(
        (deltas,),
        np.mean,
        confidence_level=0.95,
        n_resamples=1000,
        random_state=42,
        method="percentile",
    )
    scipy_low = scipy_res.confidence_interval.low
    scipy_high = scipy_res.confidence_interval.high

    # Verify pure-Python bootstrap closely matches scipy percentile bootstrap
    assert math.isclose(result.delta_total_ci[0], scipy_low, abs_tol=0.08)
    assert math.isclose(result.delta_total_ci[1], scipy_high, abs_tol=0.08)


def test_decomposition_marginal_attribution_with_baseline_failures() -> None:
    """Verify baseline partial failures do not contaminate marginal scaled failure attribution."""
    from reach.models import CatalogMode, DisclosureState, InvocationPattern, ProbeResult, Query

    q = Query(id="q1", text="test query", expected_skill="skill-a")
    # Baseline: 3/5 pass, 2/5 fail due to abstention (p_pass=0.6, p_abs=0.4, p_col=0.0)
    base = [
        ProbeResult(
            query_id="q1",
            catalog_id="base",
            catalog_mode=CatalogMode.SWEEP,
            catalog_size=2,
            model="m",
            runtime="fake",
            attempt=i,
            invoked_skills=("skill-a",) if i <= 3 else (),
            invocation_pattern=(
                InvocationPattern.ORACLE_ONLY if i <= 3 else InvocationPattern.ABANDONED
            ),
            disclosure_state=DisclosureState.FULL,
        )
        for i in range(1, 6)
    ]
    # Scaled: 1/5 pass, 2/5 FULL abstention fail, 1/5 truncated fail, 1/5 FULL collision fail
    # (p_pass=0.2, delta_total=0.4: delta_abs=0.0, delta_trunc=0.2, delta_col=0.2)
    scaled = [
        ProbeResult(
            query_id="q1",
            catalog_id="scaled",
            catalog_mode=CatalogMode.SWEEP,
            catalog_size=10,
            model="m",
            runtime="fake",
            attempt=i,
            invoked_skills=("skill-a",) if i == 1 else (() if i <= 4 else ("rival",)),
            invocation_pattern=(
                InvocationPattern.ORACLE_ONLY
                if i == 1
                else (
                    InvocationPattern.ABANDONED if i <= 4 else InvocationPattern.DISTRACTOR_HIJACK
                )
            ),
            disclosure_state=(DisclosureState.NAME_ONLY_ELIDED if i == 4 else DisclosureState.FULL),
        )
        for i in range(1, 6)
    ]
    res = decompose_pass_rate_drop(base, scaled, queries=[q], iterations=50, seed=42)
    assert math.isclose(res.delta_total, 0.4, abs_tol=1e-9)
    assert math.isclose(res.delta_abstention, 0.0, abs_tol=1e-9)
    assert math.isclose(res.delta_truncated, 0.2, abs_tol=1e-9)
    assert math.isclose(res.delta_collision, 0.2, abs_tol=1e-9)
    assert res.delta_truncated_ci[0] <= res.delta_truncated <= res.delta_truncated_ci[1]
    assert math.isclose(
        res.delta_total,
        res.delta_abstention + res.delta_truncated + res.delta_collision,
        abs_tol=1e-9,
    )
