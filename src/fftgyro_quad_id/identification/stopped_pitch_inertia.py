"""Mechanics and clock primitives for the stopped-rotor Pitch inertia retest."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Px4BootClockMap:
    """Affine PX4 boot-time to aircraft-Pi monotonic-time mapping.

    The fit uses only recorded causal ``SYSTEM_TIME`` exchange intervals.  It
    never shifts signals to improve their correlation.
    """

    px4_origin_us: float
    aircraft_origin_ns: float
    scale_ns_per_us: float
    maximum_anchor_interval_ms: float

    def map_us(self, px4_us: np.ndarray | float) -> np.ndarray:
        values = np.asarray(px4_us, dtype=float)
        return self.aircraft_origin_ns + self.scale_ns_per_us * (values - self.px4_origin_us)


def fit_px4_boot_clock(anchors: list[dict[str, float | int | str]]) -> Px4BootClockMap:
    """Fit two or more PX4/Pi anchors, using interval midpoints as estimates."""
    if len(anchors) < 2:
        raise ValueError("at least two PX4 clock anchors are required")
    boot = np.asarray([float(row["px4_boot_us"]) for row in anchors])
    lower = np.asarray([float(row["aircraft_monotonic_lower_ns"]) for row in anchors])
    upper = np.asarray([float(row["aircraft_monotonic_upper_ns"]) for row in anchors])
    if (
        not np.isfinite(boot).all()
        or not np.isfinite(lower).all()
        or not np.isfinite(upper).all()
        or np.any(upper < lower)
        or np.any(np.diff(boot) <= 0)
    ):
        raise ValueError("invalid or reordered PX4 clock anchors")
    middle = 0.5 * (lower + upper)
    origin_boot = boot[0]
    origin_aircraft = middle[0]
    design = boot - origin_boot
    scale = float(design @ (middle - origin_aircraft) / (design @ design))
    # One PX4 microsecond should be one thousand Pi nanoseconds.  A wider
    # tolerance would conceal a reset, rollover, or mismatched run artifact.
    if not 999.0 <= scale <= 1001.0:
        raise ValueError(f"implausible PX4 boot clock scale {scale:g} ns/us")
    return Px4BootClockMap(
        px4_origin_us=origin_boot,
        aircraft_origin_ns=origin_aircraft,
        scale_ns_per_us=scale,
        maximum_anchor_interval_ms=float(np.max(upper - lower) / 1e6),
    )


def pitch_support_moment_about_cg(
    force_flu_n: np.ndarray,
    moment_flu_nm: np.ndarray,
    r_sensor_to_cg_flu_m: np.ndarray,
) -> np.ndarray:
    """Return support-on-aircraft moment about CG on FRD positive Pitch.

    Leptrino measures aircraft-on-sensor in the confirmed FLU frame.  With
    ``r`` from sensor origin to aircraft CG, projection onto FRD +Y gives
    ``measured My - (r x measured F)y``.
    """
    return support_moment_about_cg_frd_axis(
        force_flu_n, moment_flu_nm, r_sensor_to_cg_flu_m, axis="PITCH"
    )


def support_moment_about_cg_frd_axis(
    force_flu_n: np.ndarray,
    moment_flu_nm: np.ndarray,
    r_sensor_to_cg_flu_m: np.ndarray,
    *,
    axis: str,
) -> np.ndarray:
    """Return support-on-aircraft CG moment on FRD positive Pitch or Yaw.

    For both requested axes, aircraft-on-sensor FLU measurement and the
    FLU-to-FRD sign reversal cancel.  The returned scalar is therefore the
    corresponding component of ``M_measured - r x F_measured``.
    """
    indices = {"PITCH": 1, "YAW": 2}
    normalized_axis = str(axis).upper()
    if normalized_axis not in indices:
        raise ValueError("axis must be PITCH or YAW")
    force = np.asarray(force_flu_n, dtype=float)
    moment = np.asarray(moment_flu_nm, dtype=float)
    arm = np.asarray(r_sensor_to_cg_flu_m, dtype=float)
    if force.ndim != 2 or moment.shape != force.shape or force.shape[1] != 3:
        raise ValueError("force and moment must both have shape (n, 3)")
    if arm.shape != (3,) or not np.isfinite(arm).all():
        raise ValueError("sensor-to-CG arm must contain three finite values")
    index = indices[normalized_axis]
    return moment[:, index] - np.cross(np.broadcast_to(arm, force.shape), force)[:, index]


def fit_inertia_with_run_intercepts(
    acceleration_rad_s2: np.ndarray,
    moment_nm: np.ndarray,
    run_index: np.ndarray,
) -> dict[str, object]:
    """OLS fit of ``moment = run intercept + Iyy * acceleration``."""
    acceleration = np.asarray(acceleration_rad_s2, dtype=float)
    moment = np.asarray(moment_nm, dtype=float)
    runs = np.asarray(run_index)
    if (
        acceleration.ndim != 1
        or moment.shape != acceleration.shape
        or runs.shape != acceleration.shape
    ):
        raise ValueError("fit arrays must be equal-length one-dimensional arrays")
    keep = np.isfinite(acceleration) & np.isfinite(moment)
    acceleration, moment, runs = acceleration[keep], moment[keep], runs[keep]
    labels = np.unique(runs)
    if len(labels) < 1 or len(acceleration) < len(labels) + 20:
        raise ValueError("too few samples for inertia fit")
    dummies = np.column_stack([runs == label for label in labels])
    design = np.column_stack((dummies, acceleration))
    coefficients, _, rank, _ = np.linalg.lstsq(design, moment, rcond=None)
    if rank != design.shape[1]:
        raise ValueError("inertia design matrix is rank deficient")
    predicted = design @ coefficients
    residual = moment - predicted
    dof = len(moment) - design.shape[1]
    residual_variance = float(residual @ residual / dof)
    covariance = residual_variance * np.linalg.inv(design.T @ design)
    inertia = float(coefficients[-1])
    inertia_std = math.sqrt(max(0.0, float(covariance[-1, -1])))
    centered = moment - np.asarray(
        [coefficients[np.flatnonzero(labels == value)[0]] for value in runs]
    )
    denominator = float(centered @ centered)
    return {
        "inertia_kg_m2": inertia,
        "inertia_standard_error_kg_m2": inertia_std,
        "inertia_95pct_ols_kg_m2": [inertia - 1.96 * inertia_std, inertia + 1.96 * inertia_std],
        "intercepts_nm": {
            str(label): float(value) for label, value in zip(labels, coefficients[:-1], strict=True)
        },
        "samples": len(moment),
        "runs": len(labels),
        "rmse_nm": float(np.sqrt(np.mean(residual**2))),
        "r_squared_within_run": (
            1.0 - float(residual @ residual) / denominator if denominator else math.nan
        ),
    }
