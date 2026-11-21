import numpy as np
import pytest

from fftgyro_quad_id.analysis.cg_joint import gravity_cg_design, project_cg


def test_level_pitch_sign_and_yaw_zero():
    assert gravity_cg_design(0.0, 0.0, 2.0, "PITCH") == pytest.approx([-19.6133, 0])
    assert gravity_cg_design(0.0, 0.0, 2.0, "YAW") == pytest.approx([0, 0])


def test_gravity_cross_product_and_common_origin_equilibrium():
    delta = np.array([0.003, -0.002])
    c = gravity_cg_design(np.array([-0.4, -0.2]), np.array([0.2, -0.3]), 1.6, "YAW")
    from fftgyro_quad_id.frames.rig_kinematics import body_to_tunnel

    r = body_to_tunnel(np.array([-0.4, -0.2]), np.array([0.2, -0.3]))
    gravity = 1.6 * 9.80665 * r[:, 2, :] * [1, -1, -1]
    expected = -np.cross(np.r_[delta, 0], gravity)[:, 2]
    np.testing.assert_allclose(c @ delta, expected)


def test_constant_cg_effect_cannot_be_separated_from_bias():
    c = np.column_stack([np.ones(10) * 15, np.zeros(10)])
    result = project_cg(c, np.ones((10, 1)), np.ones(10))
    assert result["rank"] == 0
    assert result["coordinate_identifiable"] == [False, False]
