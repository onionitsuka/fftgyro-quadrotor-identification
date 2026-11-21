"""Encoder-referenced nested-joint motion; gyro is a rate observation only.

Nominal chain Rz(heading) Ry(outer_pitch) Rz(inner_yaw). A constant independently
calibrated rotation maps PX4 body vectors into the nominal inner-frame axes.
No PX4 attitude, startup quaternion, or per-run angle tare enters this module.
"""

from __future__ import annotations

import numpy as np

from fftgyro_quad_id.filters.angle_rate_kalman import AngleRateNoise, fuse_angle_and_rate


def joint_rates_from_gyro(body_rate, inner_yaw, nominal_from_body=None):
    """Return outer Pitch rate, inner Yaw rate, and unsupported-axis residual.

    For nominal mounting, omega_body = [sin(psi)*theta_dot,
    cos(psi)*theta_dot, psi_dot]. Thus body q is not generally outer Pitch rate.
    All angles/rates use radians. Residual is diagnostic, never an extra Roll DOF.
    """
    body_rate = np.asarray(body_rate, dtype=float)
    if body_rate.shape[-1:] != (3,):
        raise ValueError("body_rate must end in three components")
    mount = np.eye(3) if nominal_from_body is None else np.asarray(nominal_from_body, float)
    if (
        mount.shape != (3, 3)
        or not np.isfinite(mount).all()
        or not np.allclose(mount.T @ mount, np.eye(3), atol=1e-9)
        or not np.isclose(np.linalg.det(mount), 1, atol=1e-9)
    ):
        raise ValueError("nominal_from_body must be a proper rotation matrix")
    p, q, r = np.moveaxis(body_rate @ mount.T, -1, 0)
    psi = np.asarray(inner_yaw, dtype=float)
    return np.stack(
        (np.sin(psi) * p + np.cos(psi) * q, r, np.cos(psi) * p - np.sin(psi) * q), axis=-1
    )


def fuse_joint_blocks(
    time_s,
    encoder_rad,
    gyro_joint_rad_s,
    valid,
    *,
    noise: AngleRateNoise | None = None,
    max_gap_s=0.05,
    min_samples=31,
    edge_trim_s=0.25,
):
    """Reuse the Kalman/RTS filter on each valid block, preserving absolute zero.

    Inputs are already clock-aligned. No long-gap extrapolation, interpolation of
    missing measurements, angle reset, or signal-correlation clock adjustment.
    Returns states [angle, rate, acceleration, projected gyro bias], block IDs,
    and an interior mask for derivative use. Noise values are analysis settings,
    not calibrated confidence bounds for resampled data.
    """
    t, angle, gyro = (np.asarray(a, dtype=float) for a in (time_s, encoder_rad, gyro_joint_rad_s))
    valid = np.asarray(valid, dtype=bool)
    if (
        t.ndim != 1
        or any(a.shape != t.shape for a in (angle, gyro, valid))
        or not np.isfinite(t).all()
        or np.any(np.diff(t) <= 0)
    ):
        raise ValueError("equal-length arrays and finite increasing times required")
    if max_gap_s <= 0 or edge_trim_s < 0 or min_samples < 2:
        raise ValueError("invalid gap, trim or minimum block size")
    valid = valid & np.isfinite(angle) & np.isfinite(gyro)
    state = np.full((len(t), 4), np.nan)
    block_id = np.full(len(t), -1, dtype=int)
    interior = np.zeros(len(t), dtype=bool)
    idx = np.flatnonzero(valid)
    splits = np.flatnonzero((np.diff(idx) > 1) | (np.diff(t[idx]) > max_gap_s)) + 1
    for number, block in enumerate(np.split(idx, splits)):
        if len(block) < min_samples:
            continue
        # The ready encoders are already unwrapped. Do not unwrap separate
        # blocks or subtract a startup angle: preserve their absolute branch.
        estimate = fuse_angle_and_rate(t[block], angle[block], t[block], gyro[block], noise=noise)
        state[block] = estimate.smoothed_state
        block_id[block] = number
        interior[block] = (t[block] >= t[block[0]] + edge_trim_s) & (
            t[block] <= t[block[-1]] - edge_trim_s
        )
    return state, block_id, interior
