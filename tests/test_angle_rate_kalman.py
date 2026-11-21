import numpy as np
import pytest

from fftgyro_quad_id.filters.angle_rate_kalman import (
    AngleRateNoise,
    fuse_angle_and_rate,
    smooth_encoder_angle,
)


def test_smoother_recovers_rate_acceleration_and_gyro_bias():
    rng = np.random.default_rng(42)
    gyro_time = np.arange(0.0, 8.0, 0.004)
    encoder_time = np.arange(0.0, 8.0, 0.01)

    def angle(t):
        return 0.18 * np.sin(2 * np.pi * 0.7 * t)

    def rate(t):
        return 0.18 * 2 * np.pi * 0.7 * np.cos(2 * np.pi * 0.7 * t)

    def acceleration(t):
        return -0.18 * (2 * np.pi * 0.7) ** 2 * np.sin(2 * np.pi * 0.7 * t)

    bias = 0.035
    encoder = angle(encoder_time) + rng.normal(0, np.deg2rad(0.06), len(encoder_time))
    gyro = rate(gyro_time) + bias + rng.normal(0, 0.008, len(gyro_time))
    estimate = fuse_angle_and_rate(
        encoder_time,
        encoder,
        gyro_time,
        gyro,
        noise=AngleRateNoise(jerk_density=5.0, bias_walk_density=0.0002),
    )
    interior = (estimate.time_s > 1.0) & (estimate.time_s < 7.0)
    state = estimate.smoothed_state[interior]
    time = estimate.time_s[interior]
    assert np.sqrt(np.mean((state[:, 0] - angle(time)) ** 2)) < 0.002
    assert np.sqrt(np.mean((state[:, 1] - rate(time)) ** 2)) < 0.015
    assert np.sqrt(np.mean((state[:, 2] - acceleration(time)) ** 2)) < 0.2
    assert abs(np.median(state[:, 3]) - bias) < 0.005
    assert estimate.encoder_updates == len(encoder_time)
    assert estimate.gyro_updates == np.count_nonzero(gyro_time <= encoder_time[-1])


def test_rejects_unaligned_time_ranges():
    with pytest.raises(ValueError, match="do not overlap"):
        fuse_angle_and_rate(
            np.array([0.0, 1.0]),
            np.array([0.0, 0.1]),
            np.array([2.0, 3.0]),
            np.array([0.0, 0.1]),
        )


def test_encoder_only_smoother_recovers_motion_without_clock_fitting():
    time = np.arange(0.0, 8.0, 0.01)
    omega = 2 * np.pi * 0.7
    angle = 0.18 * np.sin(omega * time)
    estimate = smooth_encoder_angle(
        time,
        angle,
        noise=AngleRateNoise(encoder_angle=np.deg2rad(0.01), jerk_density=8.0),
    )
    interior = (time > 1.0) & (time < 7.0)
    expected_rate = 0.18 * omega * np.cos(omega * time[interior])
    expected_acceleration = -0.18 * omega**2 * np.sin(omega * time[interior])
    assert np.sqrt(np.mean((estimate.smoothed_state[interior, 1] - expected_rate) ** 2)) < 0.02
    assert (
        np.sqrt(
            np.mean(
                (estimate.smoothed_state[interior, 2] - expected_acceleration) ** 2
            )
        )
        < 0.25
    )
    assert estimate.gyro_updates == 0
    assert np.isnan(estimate.gyro_innovation_rms_rad_s)
