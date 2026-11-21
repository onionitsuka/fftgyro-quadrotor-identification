import numpy as np
import pytest

from fftgyro_quad_id.analysis.retest_static_reference import design, fit_static_reference


def observations(gravity, noise=0.001):
    parameters = np.array([0.2, -0.1, 0.3, 0.002, -0.004, 0.001, 2.0, 0.06, -0.02])
    means = (design(gravity, 0.08) @ parameters).reshape(-1, 6)
    return dict(
        frame="physical_sensor",
        wind_free_motor_off_stationary=True,
        gravity_m_s2=np.asarray(gravity).tolist(),
        fixed_cg_height_m=0.08,
        raw_wrench_mean=means.tolist(),
        mean_stderr=np.full_like(means, noise).tolist(),
    ), parameters


def test_recovers_known_bias_mass_and_xy_without_gravity_sign_assumption():
    pitch = np.deg2rad([0, -15, -30, -15, 0])
    gravity = np.column_stack([9.80665 * np.sin(pitch), np.zeros(5), -9.80665 * np.cos(pitch)])
    obs, truth = observations(gravity)
    result = fit_static_reference(obs)
    assert result["decision"] == "CONDITIONAL_FIT"
    np.testing.assert_allclose(result["parameters"], truth, atol=1e-10)
    np.testing.assert_allclose(result["cg_sensor_m"], [0.03, -0.01, 0.08])
    assert not result["absolute_values_validated"]
    covariance = np.asarray(result["covariance"])
    assert np.linalg.eigvalsh(covariance).min() > 0
    obs["mean_stderr"] = (np.asarray(obs["mean_stderr"]) * 2).tolist()
    doubled = fit_static_reference(obs)
    np.testing.assert_allclose(doubled["covariance"], covariance * 4)


def test_one_pose_cannot_identify_zero_and_cg():
    obs, _ = observations([[0, 0, -9.80665]] * 5)
    result = fit_static_reference(obs)
    assert result["decision"] == "NOT_IDENTIFIABLE"
    assert result["parameters"] is None and result["covariance"] is None


def test_windy_or_powered_means_not_accepted():
    obs, _ = observations([[0, 0, -9.80665]])
    obs["wind_free_motor_off_stationary"] = False
    with pytest.raises(ValueError, match="wind-free"):
        fit_static_reference(obs)
