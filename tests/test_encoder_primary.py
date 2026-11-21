import numpy as np
import pytest

from fftgyro_quad_id.filters.encoder_primary import fuse_joint_blocks, joint_rates_from_gyro
from fftgyro_quad_id.frames.rig_kinematics import body_to_tunnel, rotation


def test_joint_projection_matches_rotation_derivative():
    theta, psi, td, pd = -0.5, 1.1, 0.7, -0.9
    mount = rotation(0, 0.12) @ rotation(1, -0.05)
    dt = 1e-6
    r = body_to_tunnel(theta, psi) @ mount
    dr = (
        (body_to_tunnel(theta + dt * td, psi + dt * pd) @ mount)
        - (body_to_tunnel(theta - dt * td, psi - dt * pd) @ mount)
    ) / (2 * dt)
    skew = r.T @ dr
    omega = np.array([skew[2, 1], skew[0, 2], skew[1, 0]])
    np.testing.assert_allclose(joint_rates_from_gyro(omega, psi, mount), [td, pd, 0], atol=1e-9)


def test_inner_yaw_is_not_euler_roll_rate():
    np.testing.assert_allclose(joint_rates_from_gyro([0, 0, 2.0], 1.3), [0, 2, 0])
    np.testing.assert_allclose(joint_rates_from_gyro([2.0, 0, 0], np.pi / 2), [2, 0, 0], atol=1e-12)
    with pytest.raises(ValueError, match="proper rotation"):
        joint_rates_from_gyro([0, 0, 1], 0, np.diag([1, 1, -1]))


def test_encoder_anchor_bias_multiturn_and_gaps():
    t = np.arange(0, 10, 0.01)
    angle = 4 * np.pi + 0.15 * np.sin(2 * t)
    gyro = 0.3 * np.cos(2 * t) + 0.04
    valid = (t < 4) | (t > 6)
    state, blocks, interior = fuse_joint_blocks(t, angle, gyro, valid)
    assert np.isnan(state[~valid]).all()
    assert (blocks[~valid] == -1).all()
    assert blocks[0] != blocks[-1]
    np.testing.assert_allclose(state[interior, 0], angle[interior], atol=0.003)
    np.testing.assert_allclose(state[interior, 1], 0.3 * np.cos(2 * t[interior]), atol=0.015)
    assert abs(np.median(state[interior, 3]) - 0.04) < 0.006
    # Changing observations after a gap cannot change the preceding block.
    changed = angle.copy()
    changed[t > 6] += 2 * np.pi
    other, _, _ = fuse_joint_blocks(t, changed, gyro, valid)
    np.testing.assert_array_equal(other[t < 4], state[t < 4])


def test_long_time_gap_splits_even_when_mask_valid():
    t = np.r_[np.arange(0, 1, 0.01), np.arange(2, 3, 0.01)]
    state, blocks, _ = fuse_joint_blocks(
        t, np.ones(len(t)) * 5, np.zeros(len(t)), np.ones(len(t), bool)
    )
    assert blocks[0] != blocks[-1]
    np.testing.assert_allclose(state[:, 0], 5)
