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
from collections.abc import Sequence
from statistics import NormalDist
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "BOOTSTRAP_QUANTILE_HIGH",
    "BOOTSTRAP_QUANTILE_LOW",
    "DEFAULT_CI_SPAN_SIGMAS",
    "DEFAULT_CONFIDENCE",
    "DEFAULT_POWER",
    "NORMAL_95_CI_SPAN_SIGMAS",
    "Interval",
    "bootstrap_quantiles",
    "ci_span_sigmas",
    "cluster_wilson_interval",
    "critical_value",
    "detectable_delta",
    "effective_sample_size",
    "required_probes",
    "wilson_interval",
]

#: Default confidence level for statistical intervals (95%).
DEFAULT_CONFIDENCE = 0.95

#: Tail quantiles for two-sided bootstrap confidence intervals derived from DEFAULT_CONFIDENCE.
BOOTSTRAP_QUANTILE_LOW: float = (1.0 - DEFAULT_CONFIDENCE) / 2.0
BOOTSTRAP_QUANTILE_HIGH: float = 1.0 - BOOTSTRAP_QUANTILE_LOW

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


#: Multiplier (2 * z) converting confidence interval width to SE, derived from DEFAULT_CONFIDENCE.
DEFAULT_CI_SPAN_SIGMAS: float = ci_span_sigmas(DEFAULT_CONFIDENCE)
NORMAL_95_CI_SPAN_SIGMAS: float = DEFAULT_CI_SPAN_SIGMAS


class Interval(BaseModel):
    """Represent a statistical confidence interval with lower and upper bounds."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    low: Annotated[float, Field(ge=0.0, le=1.0)]
    high: Annotated[float, Field(ge=0.0, le=1.0)]
    confidence: Annotated[float, Field(gt=0.0, lt=1.0)] = DEFAULT_CONFIDENCE

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

    @property
    def center(self) -> float:
        """Return the center point (midpoint) of the confidence interval."""
        return (self.low + self.high) / 2.0

    def excludes(self, rate: float) -> bool:
        """Return True if rate falls strictly outside the interval bounds."""
        return not self.low <= rate <= self.high

    def contains(self, rate: float) -> bool:
        """Return True if rate falls within the interval bounds."""
        return self.low <= rate <= self.high

    def __contains__(self, rate: object) -> bool:
        """Return True if rate falls within the interval bounds."""
        if not isinstance(rate, (int, float)):
            return False
        return self.low <= rate <= self.high

    def overlaps(self, other: Interval) -> bool:
        """Return True if this interval intersects with another interval."""
        return self.low <= other.high and other.low <= self.high

    def intersection(self, other: Interval) -> Interval | None:
        """Calculate the intersection with another interval, returning None if disjoint."""
        if not self.overlaps(other):
            return None
        return Interval(
            low=max(self.low, other.low),
            high=min(self.high, other.high),
            confidence=min(self.confidence, other.confidence),
        )

    def format_percent(self, digits: int = 1, separator: str = " - ") -> str:
        """Format the interval as a percentage range string '[low% - high%]'."""
        return f"[{self.low * 100:.{digits}f}%{separator}{self.high * 100:.{digits}f}%]"

    def as_tuple(self) -> tuple[float, float]:
        """Return interval bounds as a (low, high) float tuple."""
        return (self.low, self.high)

    @classmethod
    def from_tuple(
        cls,
        bounds: Sequence[float],
        confidence: float = DEFAULT_CONFIDENCE,
    ) -> Self:
        """Construct an Interval from a 2-element sequence of bounds."""
        if len(bounds) != _INTERVAL_BOUNDS_LEN:
            msg = f"expected 2 elements for interval bounds, got {len(bounds)}"
            raise ValueError(msg)
        return cls(low=bounds[0], high=bounds[1], confidence=confidence)


def wilson_interval(
    hits: int,
    probes: int,
    confidence: float = DEFAULT_CONFIDENCE,
) -> Interval | None:
    """Calculate the Wilson score confidence interval for a binomial proportion."""
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
    denominator = probes + z2
    center = (hits + z2 / 2.0) / denominator
    half = z / denominator * math.sqrt(hits * (probes - hits) / probes + z2 / 4.0)
    return Interval(
        low=0.0 if hits == 0 else max(0.0, center - half),
        high=1.0 if hits == probes else min(1.0, center + half),
        confidence=confidence,
    )


def effective_sample_size(
    sample_size: int,
    attempts: int = 1,
    intra_cluster_correlation: float = 0.6,
) -> int:
    """Calculate survey-style effective sample size adjusting for repeated attempts."""
    if sample_size <= 0:
        return 0
    if attempts <= 1:
        return sample_size
    icc = max(0.0, min(1.0, intra_cluster_correlation))
    deff = 1.0 + (attempts - 1) * icc
    return max(1, round(sample_size / deff))


def cluster_wilson_interval(
    hits: int,
    probes: int,
    attempts: int = 1,
    confidence: float = DEFAULT_CONFIDENCE,
    intra_cluster_correlation: float = 0.6,
) -> Interval | None:
    """Calculate Wilson score confidence interval adjusted for cluster design effect."""
    if probes <= 0:
        return None
    neff = effective_sample_size(probes, attempts, intra_cluster_correlation)
    rate = hits / probes
    adj_hits = max(0, min(neff, round(rate * neff)))
    return wilson_interval(adj_hits, neff, confidence=confidence)


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
