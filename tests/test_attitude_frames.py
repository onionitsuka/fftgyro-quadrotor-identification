import numpy as np

from fftgyro_quad_id.analysis.attitude_frames import (
    gravity_flu,
    px4_roll_gravity_at_encoder_level,
    reference_gravity_at_zero,
)


def test_nose_up_gravity_and_reference_undo():
    angle = np.deg2rad(30)
    q = [np.cos(angle / 2), 0, np.sin(angle / 2), 0]
    assert np.allclose(gravity_flu(q), [-9.80665 * 0.5, 0, -9.80665 * np.sqrt(3) / 2])
    session = {"pid": {"zero_quaternion_wxyz": q, "zero_angle_deg": 30}}
    assert np.allclose(reference_gravity_at_zero(session, [0, -1, 0]), [0, 0, -9.80665])


def test_yaw_does_not_change_level_gravity():
    q = [np.cos(0.6), 0, 0, np.sin(0.6)]
    assert np.allclose(gravity_flu(q), [0, 0, -9.80665])


def test_roll_right_gravity_has_negative_flu_y():
    q = [np.cos(0.2), np.sin(0.2), 0, 0]
    assert np.allclose(gravity_flu(q), [0, -9.80665 * np.sin(0.4), -9.80665 * np.cos(0.4)])


def test_px4_roll_keeps_encoder_pitch_zero_level():
    roll, pitch = np.deg2rad([8, -17])
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    q = [cr * cp, sr * cp, cr * sp, -sr * sp]
    session = {"pid": {"zero_quaternion_wxyz": q, "zero_angle_deg": 23}}
    expected = [0, -9.80665 * np.sin(roll), -9.80665 * np.cos(roll)]
    assert np.allclose(px4_roll_gravity_at_encoder_level(session), expected)
