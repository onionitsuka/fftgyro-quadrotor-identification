"""Robust, interval-constrained smoothing for offline clock alignment.

This module estimates a monotone piecewise-linear clock map.  Every fitted
anchor remains inside its causal exchange interval; a curvature penalty keeps
network-delay jitter from being mistaken for clock drift.  The model never
extrapolates and is not an absolute clock-calibration claim.
"""

from __future__ import annotations

import bisect
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import spsolve


@dataclass(frozen=True)
class SmoothIntervalClock:
    stamps: tuple[int, ...]
    estimates: tuple[int, ...]
    lowers: tuple[int, ...]
    uppers: tuple[int, ...]
    first_ns: int
    last_ns: int
    iterations: int
    maximum_anchor_residual_ns: int

    def map(self, stamp: int) -> tuple[int, int, int]:
        if not self.first_ns <= stamp <= self.last_ns:
            raise ValueError("smooth clock does not extrapolate outside anchors")
        right = bisect.bisect_left(self.stamps, stamp)
        if right == 0:
            return self.estimates[0], self.lowers[0], self.uppers[0]
        if right == len(self.stamps):
            return self.estimates[-1], self.lowers[-1], self.uppers[-1]
        left = right - 1
        weight = (stamp - self.stamps[left]) / (self.stamps[right] - self.stamps[left])

        def blend(values: tuple[int, ...]) -> int:
            return values[left] + round(weight * (values[right] - values[left]))

        estimate, lower, upper = (
            blend(self.estimates),
            blend(self.lowers),
            blend(self.uppers),
        )
        return estimate, min(lower, estimate), max(upper, estimate)


def fit_smooth_interval_clock(
    stamps: Sequence[int],
    lowers: Sequence[int],
    uppers: Sequence[int],
    *,
    curvature_strength: float = 100.0,
    robust_iterations: int = 5,
    first_stamp: int | None = None,
    last_stamp: int | None = None,
) -> SmoothIntervalClock:
    """Fit a smooth monotone map while retaining every causal anchor interval.

    The midpoint term is weighted inversely by interval width and reweighted
    with a Huber rule.  It selects a location inside the identifiable interval;
    the interval itself, rather than that selected location, remains the timing
    uncertainty carried downstream.
    """
    if len(stamps) < 3 or not (len(stamps) == len(lowers) == len(uppers)):
        raise ValueError("need at least three equally sized clock-anchor arrays")
    if curvature_strength <= 0 or robust_iterations < 1:
        raise ValueError("invalid smooth-clock tuning")
    if any(b <= a for a, b in zip(stamps, stamps[1:], strict=False)):
        raise ValueError("clock anchors must strictly increase")
    if any(high < low for low, high in zip(lowers, uppers, strict=True)):
        raise ValueError("invalid clock-anchor interval")
    coverage_first = stamps[0] if first_stamp is None else first_stamp
    coverage_last = stamps[-1] if last_stamp is None else last_stamp
    if coverage_first > stamps[0] or coverage_last < stamps[-1]:
        raise ValueError("clock coverage cannot exclude fitted anchors")

    x = np.asarray(stamps, dtype=np.float64)
    low = np.asarray(lowers, dtype=np.float64)
    high = np.asarray(uppers, dtype=np.float64)
    midpoint = (low + high) / 2.0
    width = np.maximum(high - low, 1.0)

    # Work in milliseconds relative to a weighted affine baseline.  This avoids
    # loss of precision from optimizing epoch-sized nanosecond values directly.
    x_center = x[0]
    x_scale = max(x[-1] - x[0], 1.0)
    xn = (x - x_center) / x_scale
    base_weight = (np.median(width) / width) ** 2
    design = np.column_stack((xn, np.ones(len(xn))))
    weighted = design * np.sqrt(base_weight)[:, None]
    slope, intercept = np.linalg.lstsq(weighted, midpoint * np.sqrt(base_weight), rcond=None)[0]
    baseline = slope * xn + intercept
    target = (midpoint - baseline) / 1e6
    lower_residual = (low - baseline) / 1e6
    upper_residual = (high - baseline) / 1e6
    second_dense = np.zeros((len(xn) - 2, len(xn)))
    for index in range(len(xn) - 2):
        left = xn[index + 1] - xn[index]
        right = xn[index + 2] - xn[index + 1]
        second_dense[index, index] = 1.0 / left
        second_dense[index, index + 1] = -(1.0 / left + 1.0 / right)
        second_dense[index, index + 2] = 1.0 / right
    second_dense /= max(np.median(np.abs(second_dense @ target)), 0.01)
    second = csr_matrix(second_dense)

    weights = base_weight.copy()
    estimate = np.clip(np.zeros(len(xn)), lower_residual, upper_residual)
    huber_ms = max(float(np.median(width) / 2e6), 0.01)
    iterations_used = 0
    for _iteration in range(robust_iterations):
        iterations_used += 1
        normal = diags(weights) + curvature_strength * (second.T @ second)
        estimate = np.clip(
            spsolve(normal.tocsc(), weights * target),
            lower_residual,
            upper_residual,
        )
        residual = np.abs(estimate - target)
        robust = np.minimum(1.0, huber_ms / np.maximum(residual, 1e-12))
        new_weights = base_weight * robust
        if np.max(np.abs(new_weights - weights)) < 1e-6:
            break
        weights = new_weights

    fitted = np.rint(baseline + estimate * 1e6).astype(np.int64)
    if np.any(np.diff(fitted) <= 0):
        raise ValueError("smooth interval clock would reverse time")
    if np.any(fitted < np.asarray(lowers)) or np.any(fitted > np.asarray(uppers)):
        raise ValueError("smooth interval clock escaped causal bounds")
    maximum_residual = int(np.max(np.abs(fitted - np.rint(midpoint).astype(np.int64))))
    return SmoothIntervalClock(
        tuple(int(value) for value in stamps),
        tuple(int(value) for value in fitted),
        tuple(int(value) for value in lowers),
        tuple(int(value) for value in uppers),
        int(coverage_first),
        int(coverage_last),
        iterations_used,
        maximum_residual,
    )
