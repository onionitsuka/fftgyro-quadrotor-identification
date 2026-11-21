"""Gravity in physical aircraft FLU from a PX4 FRD-to-NED quaternion."""

import numpy as np


def gravity_flu(quaternion, gravity=9.80665):
    q = np.asarray(quaternion, dtype=float)
    if (
        q.shape != (4,)
        or not np.isfinite(q).all()
        or not np.isclose(np.linalg.norm(q), 1, atol=0.01)
    ):
        raise ValueError("Expected unit PX4 wxyz attitude quaternion")
    w, x, y, z = q / np.linalg.norm(q)
    # Third row of body-FRD -> world-NED rotation, then FRD -> FLU.
    return gravity * np.array(
        [2 * (x * z - w * y), -2 * (y * z + w * x), -(1 - 2 * (x * x + y * y))]
    )


def reference_gravity_at_zero(session, axis):
    """Undo the signed joint angle at the saved stationary attitude reference."""
    axis = np.asarray(axis, float)
    if not np.isclose(np.linalg.norm(axis), 1):
        raise ValueError("Joint axis must be unit length")
    reference = session["pid"]
    g = gravity_flu(reference["zero_quaternion_wxyz"])
    angle = np.deg2rad(float(reference["zero_angle_deg"]))
    return (
        np.cos(angle) * g
        + np.sin(angle) * np.cross(axis, g)
        + (1 - np.cos(angle)) * axis * np.dot(axis, g)
    ).tolist()


def px4_roll_gravity_at_encoder_level(session, gravity=9.80665):
    """Use PX4 Roll while defining absolute encoder Pitch zero as level.

    The PX4 quaternion supplies the lateral tilt only.  Its Pitch component is
    deliberately discarded because the rig encoder independently defines
    physical Pitch, with zero calibrated as horizontal.
    """
    measured = gravity_flu(session["pid"]["zero_quaternion_wxyz"], gravity)
    horizontal_projection = np.hypot(measured[1], measured[2])
    if horizontal_projection < 0.1 * gravity:
        raise ValueError("PX4 attitude is too close to vertical to identify Roll")
    roll = np.arctan2(-measured[1], -measured[2])
    return [0.0, -gravity * np.sin(roll), -gravity * np.cos(roll)]


def registered_yaw_front_deg(session: dict, metadata: dict, calibration: dict) -> float:
    """Resolve mechanical front; a startup PX4/encoder association is NOT zero.

    ``pid.zero_angle_deg`` locates the initial quaternion in the registered
    encoder coordinates. Subtracting it again changes the aerodynamic heading.
    """
    if metadata.get("offset_yaw_deg") != calibration.get(
        "existing_encoder_offset_yaw_deg"
    ) or metadata.get("yaw_sign") != calibration.get("yaw_sign"):
        raise ValueError("encoder conversion does not match the registered Yaw-front calibration")
    key = f"{float(session.get('fixed_pitch_deg', 0)):g}"
    registered = calibration.get("pitch_calibrations", {}).get(key)
    if registered is None:
        raise ValueError(f"no registered Yaw front at fixed Pitch {key} deg")
    front = float(registered["front_experimental_yaw_deg"])
    if not np.isfinite(front):
        raise ValueError("non-finite registered Yaw front")
    return front
