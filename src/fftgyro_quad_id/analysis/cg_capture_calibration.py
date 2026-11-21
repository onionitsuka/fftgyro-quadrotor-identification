"""Calibration from leptrino_cg_live captures, whose stored poses are body FLU.

metadata.input_frame describes serial INPUT, not poses[].mean or final_result.
The acquisition calls to_flu before storing means and fitting the CG.
"""

from __future__ import annotations

import numpy as np

from fftgyro_quad_id.analysis.dynamic_wrench import rotation
from fftgyro_quad_id.analysis.leptrino_mass import fit_gravity_circle


def calibration_from_confirmed_flu_capture(metadata: dict, capture_roll_deg: float) -> dict:
    """Physical sensor FLU, known constant Roll, only Pitch varied during CG capture.

    Undo the acquisition software rotation before interpreting any physical axis.
    The force-circle center includes the gravity component parallel to Pitch.
    """
    if (
        metadata.get("state") != "COMPLETE"
        or metadata.get("scope") != "aircraft only"
        or metadata.get("motor_power_disconnected") is not True
    ):
        raise ValueError("Require complete unpowered aircraft-only CG capture")
    if not np.isfinite(capture_roll_deg) or abs(capture_roll_deg) >= 80:
        raise ValueError("Capture Roll must be known and far from a vertical Pitch axis")
    frame = metadata.get("input_frame")
    if frame not in ("element", "flu"):
        raise ValueError("Capture software input-frame convention missing")
    stored_from_sensor = (
        np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        if frame == "element"
        else np.eye(3)
    )
    means = np.asarray([p["mean"] for p in metadata["poses"]], float)
    force = means[:, :3] @ stored_from_sensor
    cg = stored_from_sensor.T @ np.asarray(metadata["final_result"]["cg_m"])
    effective = stored_from_sensor.T @ np.asarray(metadata["final_result"]["effective_bias_nm"])
    planar = force[:, [0, 2]]
    origin = planar.mean(axis=0)
    centered = planar - origin
    design = np.column_stack([2 * centered, np.ones(len(force))])
    fit, _, rank, _ = np.linalg.lstsq(design, np.sum(centered**2, axis=1), rcond=None)
    if rank != 3:
        raise ValueError("Pitch poses insufficient to identify a force circle")
    center = fit[:2] + origin
    radius = float(np.sqrt(fit[2] + np.sum(fit[:2] ** 2)))
    roll = np.deg2rad(capture_roll_deg)
    mass = radius / (9.80665 * np.cos(roll))
    axial_weight = -mass * 9.80665 * np.sin(roll)
    bias_force = np.array([center[0], np.mean(force[:, 1]) - axial_weight, center[1]])
    bias_moment = effective + np.cross(cg, bias_force)
    return {
        "mass_kg": float(mass),
        "capture_roll_deg": float(capture_roll_deg),
        "r_sensor_cg_body_m": cg.tolist(),
        "bias_sensor": np.r_[bias_force, bias_moment].tolist(),
        "pitch_plane_radius_N": radius,
        "gravity_parallel_to_pitch_N_y": float(axial_weight),
        "constant_y_residuals_N": (force[:, 1] - np.mean(force[:, 1])).tolist(),
        "pitch_plane_radius_residuals_N": (
            np.linalg.norm(planar - center, axis=1) - radius
        ).tolist(),
        "basis": "User-confirmed physical sensor FLU and constant capture Roll; Pitch-only poses",
        "limitations": [
            "Three poses do not independently validate calibration accuracy.",
            "Mounting, payload, and instrument zero must transfer to test runs.",
        ],
    }


def calibration_from_cg_capture(metadata: dict, sensor_to_body) -> dict:
    if (
        metadata.get("state") != "COMPLETE"
        or metadata.get("scope") != "aircraft only"
        or metadata.get("motor_power_disconnected") is not True
    ):
        raise ValueError("require complete aircraft-only, unpowered CG capture")
    matrix = rotation(sensor_to_body)
    means = np.asarray([p["mean"] for p in metadata["poses"]], float)
    fit = fit_gravity_circle(means[:, :3])
    cg = np.asarray(metadata["final_result"]["cg_m"], float)
    effective_bias = np.asarray(metadata["final_result"]["effective_bias_nm"], float)
    # The generic circle helper's historical '_sensor' keys use its INPUT frame.
    force_bias_body = fit["circle_center_sensor_N"]
    moment_bias_body = effective_bias + np.cross(cg, force_bias_body)
    return {
        "mass_kg": float(fit["mass_kg"]),
        "stored_pose_frame": "body_flu",
        "serial_input_frame": metadata.get("input_frame"),
        "r_sensor_cg_body_m": cg.tolist(),
        "r_sensor_cg_element_m": (matrix.T @ cg).tolist(),
        "force_bias_body_N": force_bias_body.tolist(),
        "moment_bias_body_Nm": moment_bias_body.tolist(),
        "bias_sensor": np.concatenate(
            (matrix.T @ force_bias_body, matrix.T @ moment_bias_body)
        ).tolist(),
        "assumptions": [
            "Stored means follow leptrino_cg_live's to_flu conversion.",
            "Gravity circle axis is horizontal; constant zero and configuration across poses.",
            "Sensor mounting and zero remain transferable to campaign captures.",
        ],
    }
