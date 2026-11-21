import numpy as np
import pytest

from fftgyro_quad_id.analysis.cg_capture_calibration import (
    calibration_from_cg_capture,
    calibration_from_confirmed_flu_capture,
)
from fftgyro_quad_id.analysis.dynamic_wrench import support_wrench
from fftgyro_quad_id.rig.cg_balance import fit_cg


def test_stored_flu_pose_is_not_rotated_twice():
    rotation = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    cg = np.array([0.003, -0.002, 0.081])
    bf = np.array([-0.9, -1.5, 3.9])
    bm = np.array([0.03, -0.01, 0.08])
    angles = np.deg2rad([-60, -20, 15, 65])
    weight = 1.6 * 9.80665 * np.column_stack((np.sin(angles), np.zeros(4), -np.cos(angles)))
    forces = weight + bf
    moments = np.cross(cg, weight) + bm
    means = np.column_stack((forces, moments))
    meta = {
        "state": "COMPLETE",
        "scope": "aircraft only",
        "motor_power_disconnected": True,
        "input_frame": "element",
        "poses": [{"mean": x.tolist()} for x in means],
        "final_result": fit_cg(means),
    }
    cal = calibration_from_cg_capture(meta, rotation)
    assert np.allclose(cal["r_sensor_cg_body_m"], cg)
    assert np.allclose(cal["bias_sensor"], np.r_[rotation.T @ bf, rotation.T @ bm])
    # Independent unseen pose must balance gravity and have zero CG moment.
    w = np.array([2.0, 0.0, -np.sqrt((1.6 * 9.80665) ** 2 - 4)])
    raw = np.r_[rotation.T @ (w + bf), rotation.T @ (np.cross(cg, w) + bm)]
    f, m = support_wrench(raw[None], rotation, cal["bias_sensor"], "aircraft_on_sensor")
    assert np.allclose(f[0], -w)
    assert np.allclose(m[0] - np.cross(cg, f[0]), 0, atol=1e-12)
    assert cal["mass_kg"] == pytest.approx(1.6)


def test_reject_incomplete_capture():
    with pytest.raises(ValueError):
        calibration_from_cg_capture({}, np.eye(3))


def test_inclined_pitch_axis_does_not_absorb_axial_weight_into_bias():
    roll, mass = np.deg2rad(-3.3), 1.6
    angles = np.deg2rad([-60, 0, 55])
    cg = np.array([-0.0012, -0.0002, 0.0813])
    bf, bm = np.array([1.5, -1.8, 3.9]), np.array([-0.08, -0.03, 0.09])
    weight = (
        mass
        * 9.80665
        * np.column_stack(
            (
                -np.cos(roll) * np.sin(angles),
                np.full(3, -np.sin(roll)),
                -np.cos(roll) * np.cos(angles),
            )
        )
    )
    rotation = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    stored = np.column_stack(((weight + bf) @ rotation.T, (np.cross(cg, weight) + bm) @ rotation.T))
    meta = {
        "state": "COMPLETE",
        "scope": "aircraft only",
        "motor_power_disconnected": True,
        "input_frame": "element",
        "poses": [{"mean": row.tolist()} for row in stored],
        "final_result": fit_cg(stored),
    }
    result = calibration_from_confirmed_flu_capture(meta, -3.3)
    assert result["mass_kg"] == pytest.approx(mass)
    assert np.allclose(result["bias_sensor"], np.r_[bf, bm])
    assert np.allclose(result["r_sensor_cg_body_m"], cg)
    # Independent unseen static pose and a different Roll: recovered force must be zero.
    test_roll, test_pitch = np.deg2rad([8, 17])
    w = (
        mass
        * 9.80665
        * np.array(
            [
                -np.cos(test_roll) * np.sin(test_pitch),
                -np.sin(test_roll),
                -np.cos(test_roll) * np.cos(test_pitch),
            ]
        )
    )
    sample = np.r_[w + bf, np.cross(cg, w) + bm]
    force, moment = support_wrench(
        sample[None], np.eye(3), result["bias_sensor"], "aircraft_on_sensor"
    )
    assert np.allclose(force[0] + w, 0, atol=1e-12)
    assert np.allclose(moment[0] - np.cross(cg, force[0]), 0, atol=1e-12)
