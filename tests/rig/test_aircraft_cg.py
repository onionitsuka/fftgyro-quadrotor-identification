from __future__ import annotations

import numpy as np
import pytest

from fftgyro_quad_id.rig.aircraft_cg import estimate_aircraft_cg, evaluate_axis_alignment


def test_static_wrenches_recover_cg_with_constant_force_and_moment_offsets() -> None:
    angles = np.deg2rad(np.asarray([-45.0, -30.0, -15.0, 0.0, 15.0, 30.0, 45.0]))
    true_force = np.column_stack(
        (np.zeros_like(angles), 9.2 * np.sin(angles), -9.2 * np.cos(angles))
    )
    true_cg = np.asarray([0.012, -0.004, 0.038])
    force_bias = np.asarray([0.3, -0.2, 0.1])
    moment_bias = np.asarray([0.02, -0.01, 0.03])
    measured_force = true_force + force_bias
    measured_moment = np.cross(np.broadcast_to(true_cg, true_force.shape), true_force)
    measured_moment += moment_bias

    result = estimate_aircraft_cg(measured_force, measured_moment)

    np.testing.assert_allclose(result.cg_from_sensor_m, true_cg, atol=1e-12)
    np.testing.assert_allclose(result.moment_residual_rmse_nm, 0.0, atol=1e-12)
    assert result.design_rank == 6
    assert result.pose_count == 7


def test_axis_alignment_removes_only_offset_perpendicular_to_axis() -> None:
    result = evaluate_axis_alignment(
        cg_from_sensor_m=(0.012, -0.004, 0.038),
        axis_direction=(1.0, 0.0, 0.0),
        axis_point_from_sensor_m=(0.002, 0.001, 0.008),
    )

    np.testing.assert_allclose(result.offset_perpendicular_to_axis_m, (0.0, -0.005, 0.03))
    np.testing.assert_allclose(result.correction_to_aircraft_m, (0.0, 0.005, -0.03))
    assert result.offset_along_axis_m == pytest.approx(0.01)


def test_repeated_orientation_is_rejected_as_unidentifiable() -> None:
    force = np.tile(np.asarray([0.0, 0.0, -9.8]), (7, 1))
    moment = np.zeros_like(force)

    with pytest.raises(ValueError, match="not identifiable"):
        estimate_aircraft_cg(force, moment)
