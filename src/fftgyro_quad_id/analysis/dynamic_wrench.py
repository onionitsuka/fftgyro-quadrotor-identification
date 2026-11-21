"""Aircraft-only Newton--Euler residuals. All vectors use one body frame.

Support means rig-on-aircraft; r_SG points from sensor origin to aircraft CG.
The residual is aero + prop only when there are no other external loads.
"""

from __future__ import annotations

import numpy as np


def vector(value, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite vector of length 3")
    return result


def rotation(value) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if (
        result.shape != (3, 3)
        or not np.isfinite(result).all()
        or not np.allclose(result @ result.T, np.eye(3), atol=1e-8, rtol=0)
        or not np.isclose(np.linalg.det(result), 1.0, atol=1e-8, rtol=0)
    ):
        raise ValueError("sensor_to_body must be a proper orthonormal rotation (det=+1)")
    return result


def inertia_tensor(value) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if (
        result.shape != (3, 3)
        or not np.isfinite(result).all()
        or not np.allclose(result, result.T, atol=1e-10, rtol=0)
    ):
        raise ValueError("inertia must be a finite symmetric 3x3 tensor at the CG")
    principal = np.linalg.eigvalsh(result)
    if principal[0] <= 0 or principal[-1] > principal[:-1].sum() + 1e-10:
        raise ValueError("inertia must be positive definite and satisfy triangle inequalities")
    return result


def support_wrench(
    measured: np.ndarray, sensor_to_body, bias_sensor, convention: str
) -> tuple[np.ndarray, np.ndarray]:
    """Remove six-channel instrument bias BEFORE rotating and applying action sign."""
    measured = np.asarray(measured, dtype=float)
    bias = np.asarray(bias_sensor, dtype=float)
    if measured.ndim != 2 or measured.shape[1] != 6 or not np.isfinite(measured).all():
        raise ValueError("measured wrench must be finite Nx6")
    if bias.shape != (6,) or not np.isfinite(bias).all():
        raise ValueError("bias_sensor must contain six finite values [N,N,N,Nm,Nm,Nm]")
    signs = {"support_on_aircraft": 1.0, "aircraft_on_sensor": -1.0}
    if convention not in signs:
        raise ValueError("specify support_on_aircraft or aircraft_on_sensor")
    matrix = rotation(sensor_to_body)
    corrected = (measured - bias) * signs[convention]
    return corrected[:, :3] @ matrix.T, corrected[:, 3:] @ matrix.T


def balance(
    mass_kg: float,
    gravity_body: np.ndarray,
    acceleration_cg_body: np.ndarray,
    force_support_body: np.ndarray,
    moment_support_sensor_body: np.ndarray,
    r_sensor_cg_body,
    omega_body: np.ndarray,
    alpha_body: np.ndarray,
    inertia_cg,
    rotor_h_body: np.ndarray,
    rotor_dh_body: np.ndarray,
) -> dict[str, np.ndarray]:
    """General 3D balance; dH is the derivative of rotor H components in BODY axes.

    I includes the locked-rotor inertia. H is ONLY the additional rotor spin
    angular momentum relative to the body (no double counting of I*omega).
    a_G is inertial acceleration expressed in body axes, not IMU specific force.
    """
    inertia = inertia_tensor(inertia_cg)
    if not np.isfinite(mass_kg) or mass_kg <= 0:
        raise ValueError("mass_kg must be finite and positive")
    arrays = [
        np.asarray(v, dtype=float)
        for v in (
            gravity_body,
            acceleration_cg_body,
            force_support_body,
            moment_support_sensor_body,
            omega_body,
            alpha_body,
            rotor_h_body,
            rotor_dh_body,
        )
    ]
    shape = arrays[0].shape
    if (
        len(shape) != 2
        or shape[1] != 3
        or any(a.shape != shape or not np.isfinite(a).all() for a in arrays)
    ):
        raise ValueError("all time series must be finite Nx3 with identical shape")
    gravity, acceleration, force, moment, omega, alpha, h, dh = arrays
    r = vector(r_sensor_cg_body, "r_sensor_cg_body")
    transfer = np.cross(r, force)
    support_cg = moment - transfer
    rotational = alpha @ inertia.T + np.cross(omega, omega @ inertia.T)
    rotor = dh + np.cross(omega, h)
    return {
        "gravity_force_N": mass_kg * gravity,
        "translation_force_N": mass_kg * acceleration,
        "support_force_N": force,
        "support_moment_sensor_Nm": moment,
        "origin_transfer_Nm": transfer,
        "support_moment_cg_Nm": support_cg,
        "rotation_moment_Nm": rotational,
        "rotor_moment_Nm": rotor,
        "force_ap_N": mass_kg * (acceleration - gravity) - force,
        "moment_ap_Nm": rotational + rotor - support_cg,
    }


def fixed_axis_balance(
    angle: np.ndarray,
    rate: np.ndarray,
    alpha: np.ndarray,
    measured: np.ndarray,
    config: dict,
    rotor_h: np.ndarray | None = None,
    rotor_dh: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """One fixed rotation axis, arbitrary zero attitude; stationary axis point P.

    With scalar inertia only the external moment projected onto that axis is
    returned. A full CG tensor additionally enables all three moment components.
    Angle must be right-hand rotation about axis_body, measured from g_at_zero.
    """
    axis = vector(config["axis_body"], "axis_body")
    if not np.isclose(np.linalg.norm(axis), 1.0, atol=1e-8, rtol=0):
        raise ValueError("axis_body must be a unit vector")
    angle, rate, alpha = (np.asarray(v, dtype=float) for v in (angle, rate, alpha))
    if angle.ndim != 1 or any(
        a.shape != angle.shape or not np.isfinite(a).all() for a in (angle, rate, alpha)
    ):
        raise ValueError("angle/rate/alpha must be equal-length finite vectors")
    gravity0 = vector(config["gravity_at_zero_body_m_s2"], "gravity_at_zero_body_m_s2")
    if not 9.0 < np.linalg.norm(gravity0) < 10.5:
        raise ValueError("gravity magnitude must be in m/s^2 (between 9 and 10.5)")
    r_sg = vector(config["r_sensor_cg_body_m"], "r_sensor_cg_body_m")
    r_sp = vector(config["r_sensor_pivot_body_m"], "r_sensor_pivot_body_m")
    r_pg = r_sg - r_sp
    omega, acceleration_angle = rate[:, None] * axis, alpha[:, None] * axis
    # Rodrigues R(-angle) maps fixed gravity into the rotating body axes.
    c, s = np.cos(angle)[:, None], np.sin(angle)[:, None]
    gravity = c * gravity0 - s * np.cross(axis, gravity0)
    gravity += (1 - c) * np.dot(axis, gravity0) * axis
    acceleration = np.cross(acceleration_angle, r_pg)
    acceleration += np.cross(omega, np.cross(omega, r_pg))
    force, moment = support_wrench(
        measured, config["sensor_to_body"], config["bias_sensor"], config["load_convention"]
    )
    if len(force) != len(angle):
        raise ValueError("motion and wrench row counts differ")
    if config["rotor_mode"] == "neglect":
        if rotor_h is not None or rotor_dh is not None:
            raise ValueError("rotor data supplied with rotor_mode=neglect")
        rotor_h = rotor_dh = np.zeros_like(omega)
    elif config["rotor_mode"] == "columns":
        if rotor_h is None or rotor_dh is None:
            raise ValueError("rotor_mode=columns needs rotor H and body derivative")
    else:
        raise ValueError("rotor_mode must explicitly be neglect or columns")

    full_tensor = config.get("inertia_cg_kg_m2") is not None
    if full_tensor:
        inertia = inertia_tensor(config["inertia_cg_kg_m2"])
    else:
        scalar = float(config["inertia_axis_kg_m2"])
        if not np.isfinite(scalar) or scalar <= 0:
            raise ValueError("inertia_axis_kg_m2 must be positive")
        # Isotropic placeholder computes the correct AXIS projection only.
        # Transverse moment results are removed below, never reported as known.
        inertia = np.eye(3) * scalar
    result = balance(
        float(config["mass_kg"]),
        gravity,
        acceleration,
        force,
        moment,
        r_sg,
        omega,
        acceleration_angle,
        inertia,
        rotor_h,
        rotor_dh,
    )
    result["acceleration_cg_m_s2"] = acceleration
    for key in ("rotation_moment_Nm", "rotor_moment_Nm", "support_moment_cg_Nm", "moment_ap_Nm"):
        result[key.replace("_Nm", "_axis_Nm")] = result[key] @ axis
    if not full_tensor:
        del result["moment_ap_Nm"]
        del result["rotation_moment_Nm"]
    return result
