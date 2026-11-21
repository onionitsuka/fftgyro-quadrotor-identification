import numpy as np
import pytest

from fftgyro_quad_id.identification.stopped_pitch_inertia import (
    fit_inertia_with_run_intercepts,
    fit_px4_boot_clock,
    pitch_support_moment_about_cg,
    support_moment_about_cg_frd_axis,
)


def test_px4_clock_uses_recorded_anchors_without_signal_alignment():
    anchors = [
        {
            "px4_boot_us": 1_000_000,
            "aircraft_monotonic_lower_ns": 9_999_000_000,
            "aircraft_monotonic_upper_ns": 10_001_000_000,
        },
        {
            "px4_boot_us": 101_000_000,
            "aircraft_monotonic_lower_ns": 109_999_000_000,
            "aircraft_monotonic_upper_ns": 110_001_000_000,
        },
    ]
    model = fit_px4_boot_clock(anchors)
    assert model.scale_ns_per_us == pytest.approx(1000.0)
    assert model.map_us(51_000_000) == pytest.approx(60_000_000_000)
    assert model.maximum_anchor_interval_ms == 2.0


def test_pitch_moment_shifts_measured_wrench_to_cg_with_confirmed_signs():
    force = np.array([[10.0, 2.0, 3.0]])
    measured_moment = np.array([[0.0, 4.0, 0.0]])
    r_sensor_cg = np.array([0.0, 0.0, 0.1])
    # (r x F)y = 1 N m, hence FRD-positive support moment is 4 - 1.
    calculated = pitch_support_moment_about_cg(force, measured_moment, r_sensor_cg)
    assert calculated == pytest.approx([3.0])


def test_yaw_moment_uses_frd_positive_z_after_cg_transfer():
    force = np.array([[2.0, 10.0, 3.0]])
    measured_moment = np.array([[0.0, 0.0, 4.0]])
    r_sensor_cg = np.array([0.1, 0.0, 0.0])
    # (r x F)z = 1 N m; action/reaction and FLU/FRD signs cancel.
    calculated = support_moment_about_cg_frd_axis(force, measured_moment, r_sensor_cg, axis="YAW")
    assert calculated == pytest.approx([3.0])


def test_joint_fit_recovers_inertia_despite_different_run_zero_offsets():
    rng = np.random.default_rng(4)
    run = np.repeat(np.arange(4), 300)
    acceleration = rng.normal(0, 7, len(run))
    intercept = np.array([0.12, -0.08, 0.03, -0.02])
    moment = intercept[run] + 0.0215 * acceleration + rng.normal(0, 0.002, len(run))
    result = fit_inertia_with_run_intercepts(acceleration, moment, run)
    assert result["inertia_kg_m2"] == pytest.approx(0.0215, abs=3e-5)
    assert result["runs"] == 4
    assert result["r_squared_within_run"] > 0.99
