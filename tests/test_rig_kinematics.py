import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from fftgyro_quad_id.frames.rig_kinematics import (
    airspeed_frd,
    body_to_tunnel,
    common_encoder_front,
    euler_heading,
    residual_mount_roll,
)


@pytest.mark.parametrize("yaw", [0, np.pi / 2, np.pi, -np.pi, 2 * np.pi, 5 * np.pi])
def test_inner_yaw_keeps_axial_flow_at_fixed_outer_pitch(yaw):
    theta = -np.pi / 6
    np.testing.assert_allclose(
        airspeed_frd(10, theta, yaw),
        [10 * np.cos(theta) * np.cos(yaw), -10 * np.cos(theta) * np.sin(yaw), -5],
        atol=1e-12,
    )


def test_fixture_yaw_and_inner_yaw_are_not_interchangeable():
    assert not np.allclose(airspeed_frd(10, -0.5, 0.7), airspeed_frd(10, -0.5, 0, 0.7))
    np.testing.assert_allclose(airspeed_frd(10, 0, 0.7, 0.2), airspeed_frd(10, 0, 0, 0.9))


def test_rotation_and_airspeed_invariants():
    rng = np.random.default_rng(4)
    angles = rng.normal(size=(4, 200))
    r = body_to_tunnel(*angles)
    np.testing.assert_allclose(
        np.swapaxes(r, -1, -2) @ r, np.broadcast_to(np.eye(3), r.shape), atol=1e-14
    )
    np.testing.assert_allclose(np.linalg.det(r), 1, atol=1e-14)
    v = airspeed_frd(7.5, *angles)
    np.testing.assert_allclose(np.linalg.norm(v, axis=1), 7.5, atol=1e-13)
    np.testing.assert_allclose(airspeed_frd(0, *angles), 0)


def test_px4_roll_does_not_double_count_nested_yaw_tilt():
    pitch, yaw, mount = np.deg2rad([-30, 80, 2])
    quat_xyzw = Rotation.from_matrix(body_to_tunnel(pitch, yaw, 0.2, mount)).as_quat()
    np.testing.assert_allclose(
        residual_mount_roll(quat_xyzw[[3, 0, 1, 2]], pitch, yaw), mount, atol=1e-12
    )
    assert not np.isclose(euler_heading(pitch, yaw, 0.2), yaw + 0.2)


def test_common_zero_rejects_changed_calibration():
    cal = {
        "yaw_sign": 1,
        "existing_encoder_offset_yaw_deg": -83.1445,
        "pitch_calibrations": {"0": {"front_experimental_yaw_deg": 0}},
    }
    meta = {"yaw_sign": 1, "offset_yaw_deg": -83.1445}
    assert common_encoder_front(meta, cal) == 0
    with pytest.raises(ValueError):
        common_encoder_front({**meta, "yaw_sign": -1}, cal)
