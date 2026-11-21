import numpy as np

from fftgyro_quad_id.analysis.leptrino_mass import fit_gravity_circle


def test_gravity_circle_recovers_mass_despite_force_zero() -> None:
    mass = 1.6
    gravity = 9.80665
    bias = np.array([-0.9, -1.5, 3.9])
    angles = np.deg2rad([-50.0, 0.0, 47.0, 82.0])
    forces = bias + mass * gravity * np.column_stack(
        (np.zeros(len(angles)), np.sin(angles), -np.cos(angles))
    )
    result = fit_gravity_circle(forces, gravity)
    assert np.isclose(result["mass_kg"], mass)
    assert np.allclose(result["circle_center_sensor_N"], bias)
    assert np.max(np.abs(result["pose_radius_residual_N"])) < 1e-10
