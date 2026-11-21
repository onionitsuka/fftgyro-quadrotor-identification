"""Campaign G4.0 kinematics, distinct from aircraft ZYX Euler angles.

The product Frame2 is locked; Frame3 is experimental Pitch and Frame4 is
inner Yaw. In forward/right/down coordinates the nominal body-to-tunnel
rotation is Rz(fixture) Ry(encoder_pitch) Rz(encoder_yaw) Rx(mount_roll).
The terminal Roll is a PX4-derived residual, NOT the full Euler Roll at a
nonzero inner Yaw. No MuJoCo masses, friction, sensor geometry or aero are used.
"""

from __future__ import annotations

import numpy as np

KINEMATICS_SCHEMA = "g40-outer-pitch-inner-yaw-frd-v1"


def rotation(axis: int, angle) -> np.ndarray:
    """Broadcastable active right-handed rotation matrix, radians."""
    angle = np.asarray(angle, dtype=float)
    c, s = np.cos(angle), np.sin(angle)
    result = np.zeros(angle.shape + (3, 3))
    i, j = (axis + 1) % 3, (axis + 2) % 3
    result[..., axis, axis] = 1
    result[..., i, i] = result[..., j, j] = c
    result[..., i, j] = -s
    result[..., j, i] = s
    return result


def body_to_tunnel(pitch, inner_yaw, fixture_heading=0.0, mount_roll=0.0) -> np.ndarray:
    """Body FRD to tunnel forward/right/down; positive Yaw is nose right."""
    return (
        rotation(2, fixture_heading)
        @ rotation(1, pitch)
        @ rotation(2, inner_yaw)
        @ rotation(0, mount_roll)
    )


def airspeed_frd(speed, pitch, inner_yaw, fixture_heading=0.0, mount_roll=0.0):
    """Transform [V,0,0] air-relative velocity, not the opposite wind vector.

    CG translation is not included. Encoder angles may exceed +/-pi: the
    rotation is periodic without resetting the absolute mechanical reference.
    """
    r = body_to_tunnel(pitch, inner_yaw, fixture_heading, mount_roll)
    return r[..., 0, :] * np.asarray(speed)[..., None]


def euler_heading(pitch, inner_yaw, fixture_heading=0.0, mount_roll=0.0):
    """Displayed horizontal nose azimuth; NOT fixture + inner encoder Yaw."""
    r = body_to_tunnel(pitch, inner_yaw, fixture_heading, mount_roll)
    return np.arctan2(r[..., 1, 0], r[..., 0, 0])


def residual_mount_roll(quaternion, pitch, inner_yaw) -> np.ndarray:
    """Remove the known nested-joint Euler Roll from measured PX4 Roll.

    PX4 wxyz is FRD-to-NED. Only gravity direction is used, so magnetic/world
    Yaw never defines wind alignment. This one-parameter residual is an
    approximation, not an independently calibrated full mount rotation.
    """
    q = np.asarray(quaternion, dtype=float)
    if q.shape[-1:] != (4,):
        raise ValueError("PX4 wxyz quaternion required")
    norm = np.linalg.norm(q, axis=-1)
    valid = np.isfinite(q).all(axis=-1) & (np.abs(norm - 1) < 0.01)
    q = q / np.where(valid, norm, 1.0)[..., None]
    w, x, y, z = np.moveaxis(q, -1, 0)
    measured_roll = np.arctan2(2 * (y * z + w * x), 1 - 2 * (x * x + y * y))
    r = body_to_tunnel(pitch, inner_yaw)
    expected_roll = np.arctan2(r[..., 2, 1], r[..., 2, 2])
    difference = measured_roll - expected_roll
    return np.where(valid, np.arctan2(np.sin(difference), np.cos(difference)), np.nan)


def common_encoder_front(metadata: dict, calibration: dict) -> float:
    """Validate the common operator-defined encoder zero, not a pitch fit.

    Only a single common zero is supported here. Differing pitch-dependent
    reference values require an explicit geometry calibration, not interpolation.
    """
    if metadata.get("yaw_sign") != calibration.get("yaw_sign") or metadata.get(
        "offset_yaw_deg"
    ) != calibration.get("existing_encoder_offset_yaw_deg"):
        raise ValueError("encoder Yaw calibration mismatch")
    fronts = {
        float(p["front_experimental_yaw_deg"]) for p in calibration["pitch_calibrations"].values()
    }
    if len(fronts) != 1 or not np.isfinite(next(iter(fronts))):
        raise ValueError("a common mechanical encoder zero is required")
    return next(iter(fronts))
