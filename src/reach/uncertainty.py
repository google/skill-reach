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

"""Calculate statistical confidence intervals and power for rate metrics."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from statistics import NormalDist
from typing import Annotated, Self, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "Interval",
    "cluster_wilson_interval",
    "detectable_delta",
    "effective_sample_size",
    "estimate_skill_icc",
    "required_probes",
    "wilson_interval",
]

#: Default confidence level for statistical intervals (95%).
DEFAULT_CONFIDENCE = 0.95

#: Default statistical power for hypothesis comparisons (80%).
DEFAULT_POWER = 0.80

_INTERVAL_BOUNDS_LEN = 2


def _z(tail: float) -> float:
    """Calculate the standard normal deviate leaving tail probability above it."""
    return NormalDist().inv_cdf(1.0 - tail)


def _checked_confidence(confidence: float) -> float:
    """Validate that confidence level lies strictly in (0, 1)."""
    if not 0.0 < confidence < 1.0:
        msg = f"confidence must lie strictly between 0 and 1, got {confidence}"
        raise ValueError(
            msg,
        )
    return confidence


def critical_value(confidence: float = DEFAULT_CONFIDENCE) -> float:
    """Calculate two-sided normal critical value for a given confidence level."""
    return _z((1.0 - _checked_confidence(confidence)) / 2.0)


def ci_span_sigmas(confidence: float = DEFAULT_CONFIDENCE) -> float:
    """Calculate two-sided normal confidence interval span in standard errors (2 * z)."""
    return 2.0 * critical_value(confidence)


def bootstrap_quantiles(confidence: float = DEFAULT_CONFIDENCE) -> tuple[float, float]:
    """Calculate lower and upper tail quantiles for a two-sided confidence interval."""
    alpha = 1.0 - _checked_confidence(confidence)
    return (alpha / 2.0, 1.0 - alpha / 2.0)


class Interval(BaseModel):
    """Represent a statistical confidence interval with lower and upper bounds."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    low: Annotated[float, Field(ge=-1.0, le=1.0)]
    high: Annotated[float, Field(ge=-1.0, le=1.0)]
    confidence: Annotated[float, Field(gt=0.0, lt=1.0)] = DEFAULT_CONFIDENCE

    @classmethod
    def zero(cls, confidence: float = DEFAULT_CONFIDENCE) -> Self:
        """Return a zero-width interval centered at zero."""
        return cls(low=0.0, high=0.0, confidence=confidence)

    @classmethod
    def unit(cls, confidence: float = DEFAULT_CONFIDENCE) -> Self:
        """Return a unit interval [0.0, 1.0]."""
        return cls(low=0.0, high=1.0, confidence=confidence)

    @classmethod
    def from_tuple(
        cls,
        bounds: Interval | Sequence[float],
        confidence: float = DEFAULT_CONFIDENCE,
    ) -> Self:
        """Construct an Interval from an existing Interval or a 2-element sequence."""
        if isinstance(bounds, Interval):
            if bounds.confidence == confidence and type(bounds) is cls:
                return cast(Self, bounds)
            return cls(low=bounds.low, high=bounds.high, confidence=confidence)
        if len(bounds) != _INTERVAL_BOUNDS_LEN:
            msg = f"expected 2 elements for interval bounds, got {len(bounds)}"
            raise ValueError(msg)
        return cls(low=float(bounds[0]), high=float(bounds[1]), confidence=confidence)

    @model_validator(mode="after")
    def _bounds_are_ordered(self) -> Self:
        """Validate that lower bound does not exceed upper bound."""
        if self.low > self.high:
            msg = f"interval bounds are inverted: [{self.low}, {self.high}]"
            raise ValueError(msg)
        return self

    @property
    def width(self) -> float:
        """Return the span (high - low) of the confidence interval."""
        return self.high - self.low

    def excludes(self, rate: float) -> bool:
        """Return True if rate falls strictly outside the interval bounds."""
        return not self.low <= rate <= self.high

    def overlaps(self, other: Interval) -> bool:
        """Return True if this interval intersects with another interval."""
        return self.low <= other.high and other.low <= self.high

    def format_percent(self, digits: int = 1, separator: str = " - ") -> str:
        """Format the interval as a percentage range string '[low% - high%]'."""
        return f"[{self.low * 100:.{digits}f}%{separator}{self.high * 100:.{digits}f}%]"


def _stage_deff(cluster_size: float, icc: float) -> float:
    """Compute single-stage Kish cluster design effect 1 + (m - 1) * rho."""
    if cluster_size <= 1.0:
        return 1.0
    return 1.0 + (cluster_size - 1.0) * max(0.0, min(1.0, icc))


def _design_effect(
    attempts: int,
    intra_cluster_correlation: float,
    queries_per_skill: float = 1.0,
    skill_icc: float = 0.0,
) -> float:
    """Compute two-stage survey cluster design effect across attempts and skills."""
    return _stage_deff(float(attempts), intra_cluster_correlation) * _stage_deff(
        queries_per_skill, skill_icc
    )


def _wilson_from_counts(
    hits: int,
    probes: int,
    effective_n: float,
    confidence: float,
) -> Interval | None:
    """Compute Wilson score confidence interval from raw counts and effective sample size."""
    if probes < 0:
        msg = f"probes cannot be negative, got {probes}"
        raise ValueError(msg)
    if not 0 <= hits <= probes:
        msg = f"{hits} hits is not a possible count out of {probes} probes"
        raise ValueError(msg)
    if not probes:
        return None

    z = critical_value(confidence)
    z2 = z * z
    rate = hits / probes
    eff_hits = rate * effective_n
    denominator = effective_n + z2
    center = (eff_hits + z2 / 2.0) / denominator
    half = z / denominator * math.sqrt(eff_hits * (1.0 - rate) + z2 / 4.0)
    return Interval(
        low=0.0 if hits == 0 else max(0.0, center - half),
        high=1.0 if hits == probes else min(1.0, center + half),
        confidence=confidence,
    )


def wilson_interval(
    hits: int,
    probes: int,
    confidence: float = DEFAULT_CONFIDENCE,
) -> Interval | None:
    """Calculate the Wilson score confidence interval for a binomial proportion."""
    return _wilson_from_counts(hits, probes, float(probes), confidence)


def effective_sample_size(
    sample_size: int,
    attempts: int = 1,
    intra_cluster_correlation: float = 0.6,
    *,
    queries_per_skill: float = 1.0,
    skill_icc: float = 0.0,
) -> int:
    """Calculate survey-style effective sample size adjusting for cluster correlation."""
    if sample_size <= 0:
        return 0
    deff = _design_effect(attempts, intra_cluster_correlation, queries_per_skill, skill_icc)
    return max(1, round(sample_size / deff))


def cluster_wilson_interval(
    hits: int,
    probes: int,
    attempts: int = 1,
    confidence: float = DEFAULT_CONFIDENCE,
    intra_cluster_correlation: float = 0.6,
    *,
    queries_per_skill: float = 1.0,
    skill_icc: float = 0.0,
) -> Interval | None:
    """Calculate Wilson score confidence interval adjusted for cluster design effect."""
    deff = _design_effect(attempts, intra_cluster_correlation, queries_per_skill, skill_icc)
    if deff <= 1.0:
        return wilson_interval(hits, probes, confidence=confidence)
    return _wilson_from_counts(hits, probes, max(1.0, probes / deff), confidence)


def _two_proportion_constant(alpha: float, power: float) -> float:
    """Compute the pooled two-proportion sample size constant."""
    return (_z(alpha / 2.0) + _z(1.0 - power)) ** 2 * 0.5


def required_probes(
    delta: float,
    confidence: float = DEFAULT_CONFIDENCE,
    power: float = DEFAULT_POWER,
) -> int:
    """Calculate sample size per arm required to detect a rate delta with power."""
    if not 0.0 < delta <= 1.0:
        msg = f"delta must lie in (0, 1], got {delta}"
        raise ValueError(msg)
    alpha = 1.0 - _checked_confidence(confidence)
    if not 0.0 < power < 1.0:
        msg = f"power must lie strictly between 0 and 1, got {power}"
        raise ValueError(msg)
    return math.ceil(_two_proportion_constant(alpha, power) / (delta * delta))


def detectable_delta(
    probes: int,
    confidence: float = DEFAULT_CONFIDENCE,
    power: float = DEFAULT_POWER,
) -> float | None:
    """Calculate the minimum detectable rate delta for a given sample size per arm."""
    if probes < 0:
        msg = f"probes cannot be negative, got {probes}"
        raise ValueError(msg)
    if not probes:
        return None
    alpha = 1.0 - _checked_confidence(confidence)
    if not 0.0 < power < 1.0:
        msg = f"power must lie strictly between 0 and 1, got {power}"
        raise ValueError(msg)
    return min(1.0, math.sqrt(_two_proportion_constant(alpha, power) / probes))


_MIN_ANOVA_GROUPS: int = 2


def estimate_skill_icc(
    outcomes_by_skill: Mapping[str, Sequence[float]],
) -> float | None:
    """Estimate intra-skill correlation via one-way random-effects ANOVA across skills.

    Args:
        outcomes_by_skill: Mapping of skill names to sequences of per-query pass rates.

    Returns:
        Estimated intra-class correlation coefficient in [0.0, 1.0], or None when
        fewer than two skills or zero within-skill degrees of freedom are available.
    """
    groups = [[float(v) for v in vals] for vals in outcomes_by_skill.values() if len(vals) >= 1]
    s_skills = len(groups)
    if s_skills < _MIN_ANOVA_GROUPS:
        return None
    total_q = sum(len(g) for g in groups)
    if total_q <= s_skills:
        return None

    grand_mean = sum(sum(g) for g in groups) / total_q
    ssb = 0.0
    ssw = 0.0
    for g in groups:
        q_s = len(g)
        mean_s = sum(g) / q_s
        ssb += q_s * ((mean_s - grand_mean) ** 2)
        ssw += sum((y - mean_s) ** 2 for y in g)

    msb = ssb / (s_skills - 1)
    msw = ssw / (total_q - s_skills)
    if msb == 0.0 and msw == 0.0:
        return 0.0

    q_bar = (total_q - sum(len(g) ** 2 for g in groups) / total_q) / (s_skills - 1)
    denom = msb + (q_bar - 1.0) * msw
    if denom <= 0.0:
        return 0.0
    rho = (msb - msw) / denom
    return round(max(0.0, min(1.0, rho)), 4)
