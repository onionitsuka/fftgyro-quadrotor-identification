"""Asynchronous angle/rate Kalman filtering for rig post-processing.

The state is ``[angle, angular_rate, angular_acceleration, gyro_bias]``.
Encoder angle and PX4 gyro rate may arrive at different sample rates.  A
constant-acceleration model driven by white jerk is used between observations;
an RTS backward pass supplies offline smoothed estimates.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AngleRateNoise:
    """Noise standard deviations in SI units.

    ``jerk_density`` and ``bias_walk_density`` are continuous-time white-noise
    amplitudes per square-root hertz.
    """

    encoder_angle: float = np.deg2rad(0.06)
    gyro_rate: float = 0.01
    jerk_density: float = 5.0
    bias_walk_density: float = 0.001

    def validate(self) -> None:
        values = (
            self.encoder_angle,
            self.gyro_rate,
            self.jerk_density,
            self.bias_walk_density,
        )
        if not all(np.isfinite(value) and value > 0 for value in values):
            raise ValueError("all noise standard deviations must be finite and positive")


@dataclass(frozen=True)
class AngleRateEstimate:
    time_s: np.ndarray
    filtered_state: np.ndarray
    smoothed_state: np.ndarray
    filtered_covariance: np.ndarray
    smoothed_covariance: np.ndarray
    encoder_updates: int
    gyro_updates: int
    encoder_innovation_rms_rad: float
    gyro_innovation_rms_rad_s: float


def _validate_series(name: str, time_s: np.ndarray, values: np.ndarray) -> None:
    if time_s.ndim != 1 or values.ndim != 1 or len(time_s) != len(values):
        raise ValueError(f"{name} times and values must be equal-length one-dimensional arrays")
    if len(time_s) < 2 or not np.isfinite(time_s).all() or not np.isfinite(values).all():
        raise ValueError(f"{name} requires at least two finite samples")
    if np.any(np.diff(time_s) <= 0):
        raise ValueError(f"{name} times must be strictly increasing")


def _transition(dt: float, noise: AngleRateNoise) -> tuple[np.ndarray, np.ndarray]:
    dt2, dt3, dt4, dt5 = dt**2, dt**3, dt**4, dt**5
    transition = np.array(
        [
            [1.0, dt, 0.5 * dt2, 0.0],
            [0.0, 1.0, dt, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    jerk = noise.jerk_density**2 * np.array(
        [
            [dt5 / 20, dt4 / 8, dt3 / 6],
            [dt4 / 8, dt3 / 3, dt2 / 2],
            [dt3 / 6, dt2 / 2, dt],
        ]
    )
    process = np.zeros((4, 4))
    process[:3, :3] = jerk
    process[3, 3] = noise.bias_walk_density**2 * dt
    return transition, process


def _update(
    state: np.ndarray,
    covariance: np.ndarray,
    value: float,
    observation: np.ndarray,
    variance: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    innovation = float(value - observation @ state)
    innovation_variance = float(observation @ covariance @ observation + variance)
    gain = covariance @ observation / innovation_variance
    state = state + gain * innovation
    # Joseph form maintains symmetry and positive semi-definiteness better.
    identity = np.eye(4)
    residual_map = identity - np.outer(gain, observation)
    covariance = residual_map @ covariance @ residual_map.T + variance * np.outer(gain, gain)
    return state, 0.5 * (covariance + covariance.T), innovation


def fuse_angle_and_rate(
    encoder_time_s: np.ndarray,
    encoder_angle_rad: np.ndarray,
    gyro_time_s: np.ndarray,
    gyro_rate_rad_s: np.ndarray,
    *,
    noise: AngleRateNoise | None = None,
) -> AngleRateEstimate:
    """Fuse asynchronous unwrapped encoder angle and body-axis gyro rate.

    Inputs must already use the same time base and physical positive axis.  The
    result is restricted to their common time interval.  No clock alignment or
    coordinate conversion is inferred here.
    """

    noise = noise or AngleRateNoise()
    encoder_time_s = np.asarray(encoder_time_s, dtype=float)
    encoder_angle_rad = np.asarray(encoder_angle_rad, dtype=float)
    gyro_time_s = np.asarray(gyro_time_s, dtype=float)
    gyro_rate_rad_s = np.asarray(gyro_rate_rad_s, dtype=float)
    _validate_series("encoder", encoder_time_s, encoder_angle_rad)
    _validate_series("gyro", gyro_time_s, gyro_rate_rad_s)
    noise.validate()

    start = max(encoder_time_s[0], gyro_time_s[0])
    stop = min(encoder_time_s[-1], gyro_time_s[-1])
    if not start < stop:
        raise ValueError("encoder and gyro time ranges do not overlap")
    encoder_mask = (encoder_time_s >= start) & (encoder_time_s <= stop)
    gyro_mask = (gyro_time_s >= start) & (gyro_time_s <= stop)
    et, ea = encoder_time_s[encoder_mask], encoder_angle_rad[encoder_mask]
    gt, gr = gyro_time_s[gyro_mask], gyro_rate_rad_s[gyro_mask]
    if len(et) < 2 or len(gt) < 2:
        raise ValueError("common interval must contain at least two samples from each sensor")

    event_time = np.unique(np.concatenate((et, gt)))
    state = np.array([np.interp(event_time[0], et, ea), np.interp(event_time[0], gt, gr), 0.0, 0.0])
    covariance = np.diag(
        [noise.encoder_angle**2, noise.gyro_rate**2, noise.jerk_density**2, 0.1**2]
    )
    filtered_state = np.empty((len(event_time), 4))
    filtered_covariance = np.empty((len(event_time), 4, 4))
    predicted_state = np.empty_like(filtered_state)
    predicted_covariance = np.empty_like(filtered_covariance)
    transitions = np.empty_like(filtered_covariance)
    encoder_innovations: list[float] = []
    gyro_innovations: list[float] = []
    encoder_index = gyro_index = 0
    angle_observation = np.array([1.0, 0.0, 0.0, 0.0])
    gyro_observation = np.array([0.0, 1.0, 0.0, 1.0])

    previous = event_time[0]
    for index, stamp in enumerate(event_time):
        dt = float(stamp - previous)
        transition, process = _transition(dt, noise)
        state = transition @ state
        covariance = transition @ covariance @ transition.T + process
        predicted_state[index] = state
        predicted_covariance[index] = covariance
        transitions[index] = transition
        while encoder_index < len(et) and et[encoder_index] == stamp:
            state, covariance, innovation = _update(
                state, covariance, ea[encoder_index], angle_observation, noise.encoder_angle**2
            )
            encoder_innovations.append(innovation)
            encoder_index += 1
        while gyro_index < len(gt) and gt[gyro_index] == stamp:
            state, covariance, innovation = _update(
                state, covariance, gr[gyro_index], gyro_observation, noise.gyro_rate**2
            )
            gyro_innovations.append(innovation)
            gyro_index += 1
        filtered_state[index] = state
        filtered_covariance[index] = covariance
        previous = stamp

    smoothed_state = filtered_state.copy()
    smoothed_covariance = filtered_covariance.copy()
    for index in range(len(event_time) - 2, -1, -1):
        next_index = index + 1
        smoother_gain = (
            filtered_covariance[index]
            @ transitions[next_index].T
            @ np.linalg.inv(predicted_covariance[next_index])
        )
        smoothed_state[index] += smoother_gain @ (
            smoothed_state[next_index] - predicted_state[next_index]
        )
        smoothed_covariance[index] += (
            smoother_gain
            @ (smoothed_covariance[next_index] - predicted_covariance[next_index])
            @ smoother_gain.T
        )
        smoothed_covariance[index] = 0.5 * (
            smoothed_covariance[index] + smoothed_covariance[index].T
        )

    return AngleRateEstimate(
        time_s=event_time,
        filtered_state=filtered_state,
        smoothed_state=smoothed_state,
        filtered_covariance=filtered_covariance,
        smoothed_covariance=smoothed_covariance,
        encoder_updates=len(encoder_innovations),
        gyro_updates=len(gyro_innovations),
        encoder_innovation_rms_rad=float(np.sqrt(np.mean(np.square(encoder_innovations)))),
        gyro_innovation_rms_rad_s=float(np.sqrt(np.mean(np.square(gyro_innovations)))),
    )


def smooth_encoder_angle(
    encoder_time_s: np.ndarray,
    encoder_angle_rad: np.ndarray,
    *,
    noise: AngleRateNoise | None = None,
) -> AngleRateEstimate:
    """Estimate angle, rate, and acceleration from encoder angle alone.

    This is the clock-safe fallback for runs that lack a recorded PX4-boot to
    host-clock mapping.  It uses the same constant-acceleration/white-jerk
    model and RTS backward pass as :func:`fuse_angle_and_rate`, but it does not
    infer a gyro offset from signal correlation.  The fourth state is retained
    for API compatibility and remains an unused zero gyro-bias state.
    """

    noise = noise or AngleRateNoise()
    encoder_time_s = np.asarray(encoder_time_s, dtype=float)
    encoder_angle_rad = np.asarray(encoder_angle_rad, dtype=float)
    _validate_series("encoder", encoder_time_s, encoder_angle_rad)
    noise.validate()

    event_time = encoder_time_s
    state = np.array([encoder_angle_rad[0], 0.0, 0.0, 0.0])
    covariance = np.diag(
        [noise.encoder_angle**2, 0.3**2, noise.jerk_density**2, 1e-12]
    )
    filtered_state = np.empty((len(event_time), 4))
    filtered_covariance = np.empty((len(event_time), 4, 4))
    predicted_state = np.empty_like(filtered_state)
    predicted_covariance = np.empty_like(filtered_covariance)
    transitions = np.empty_like(filtered_covariance)
    innovations: list[float] = []
    angle_observation = np.array([1.0, 0.0, 0.0, 0.0])

    previous = event_time[0]
    for index, (stamp, angle) in enumerate(zip(event_time, encoder_angle_rad, strict=True)):
        dt = float(stamp - previous)
        transition, process = _transition(dt, noise)
        # No gyro observes the fourth state, so do not inject bias random walk.
        process[3, 3] = 0.0
        state = transition @ state
        covariance = transition @ covariance @ transition.T + process
        predicted_state[index] = state
        predicted_covariance[index] = covariance
        transitions[index] = transition
        state, covariance, innovation = _update(
            state, covariance, angle, angle_observation, noise.encoder_angle**2
        )
        innovations.append(innovation)
        filtered_state[index] = state
        filtered_covariance[index] = covariance
        previous = stamp

    smoothed_state = filtered_state.copy()
    smoothed_covariance = filtered_covariance.copy()
    for index in range(len(event_time) - 2, -1, -1):
        next_index = index + 1
        smoother_gain = (
            filtered_covariance[index]
            @ transitions[next_index].T
            @ np.linalg.inv(predicted_covariance[next_index])
        )
        smoothed_state[index] += smoother_gain @ (
            smoothed_state[next_index] - predicted_state[next_index]
        )
        smoothed_covariance[index] += (
            smoother_gain
            @ (smoothed_covariance[next_index] - predicted_covariance[next_index])
            @ smoother_gain.T
        )
        smoothed_covariance[index] = 0.5 * (
            smoothed_covariance[index] + smoothed_covariance[index].T
        )

    return AngleRateEstimate(
        time_s=event_time,
        filtered_state=filtered_state,
        smoothed_state=smoothed_state,
        filtered_covariance=filtered_covariance,
        smoothed_covariance=smoothed_covariance,
        encoder_updates=len(innovations),
        gyro_updates=0,
        encoder_innovation_rms_rad=float(np.sqrt(np.mean(np.square(innovations)))),
        gyro_innovation_rms_rad_s=float("nan"),
    )
