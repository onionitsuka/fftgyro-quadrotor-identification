"""No fitting: compare deployed algebra against the existing load pipeline."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from analysis import Settings, induced_inflow, route_state  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "forward_sim", SCRIPTS / "simulate_graybox_forward_flight.py"
)
sim = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = sim
spec.loader.exec_module(sim)


def test_simulated_motor_map_matches_pipeline_and_inverse():
    from analysis import motor_basis
    motors = [dict(map_coefficients=[3000, 5000, 2000], voltage_exponent=e)
              for e in (.87, .88, .89, .90)]
    u = np.array([.2, .3, .4, .5])
    speed = sim.motor_map(u, 14.4, motors)
    expected = [float((motor_basis(u[i:i+1], np.array([14.4]), m["voltage_exponent"])
                        @ m["map_coefficients"])[0]) * 2*np.pi/60
                for i, m in enumerate(motors)]
    np.testing.assert_allclose(speed, expected)
    np.testing.assert_allclose(sim.inverse_motor_map(speed, 14.4, motors), u, atol=1e-9)


def artifact():
    from dataclasses import asdict

    return dict(
        settings=asdict(Settings(fz_model="sun_graybox")),
        motor_models=[],
        models={
            "measured": {
                out: dict(coefficients=[0.002], selected_terms=[[["mu_x", 1]]])
                for out in sim.OUTPUTS
            }
        },
    )


def test_hover_and_gravity_cancel():
    a = sim.Aircraft(artifact(), "measured")
    x = np.zeros(13)
    x[6] = 1
    dx, _ = sim.rhs(x, a, np.full(4, a.hover_w), np.zeros(4))
    np.testing.assert_allclose(dx, 0, atol=1e-12)


@pytest.mark.parametrize("mu", [[0, 0, 0], [0.1, 0.05, -0.03], [0.1, 0, 0.2]])
def test_momentum_root_matches_pipeline(mu):
    mu = np.array(mu, float)
    root, ambiguous = induced_inflow(mu[None, :], 0.01)
    assert not ambiguous[0]
    assert sim.inflow(mu, 0.01) == pytest.approx(root[0] - mu[2], abs=1e-12)


def test_absolute_algebra_matches_saved_pipeline():
    a = sim.Aircraft(artifact(), "measured", anchored=False)
    cfg = Settings(fz_model="sun_graybox")
    omega, air = np.array([0.1, -0.2, 0.3]), np.array([7.0, 0.5, -1.0])
    rotor = np.array([580.0, 590.0, 570.0, 560.0])
    raw = dict(
        air=np.tile(air, (100, 1)), omega=np.tile(omega, (100, 1)), valid_motion=np.ones(100, bool)
    )
    pipeline = route_state(raw, np.tile(rotor * 60 / (2 * np.pi), (100, 1)), cfg)
    expected = pipeline["physical"][50].copy()
    for j, out in enumerate(sim.OUTPUTS):
        scale = pipeline["force_scale"] if j < 3 else pipeline["moment_scale"]
        expected[j] += (
            scale[50] * pipeline["state"]["mu_x"][50] * 0.002
        )
    np.testing.assert_allclose(a.loads(air, omega, rotor, np.zeros(4)), expected, atol=1e-10)


def test_rotor_acceleration_reaction_sign():
    a = sim.Aircraft(artifact(), "physical")
    dot = np.array([10.0, 0, 0, 0])
    load = a.loads(np.zeros(3), np.zeros(3), np.full(4, a.hover_w), dot)
    assert load[5] == pytest.approx(a.cfg["rotor_inertia_kg_m2"] * 10)


def test_forward_pitch_rotates_thrust_forward():
    q = np.array([np.cos(-0.1 / 2), 0, np.sin(-0.1 / 2), 0])
    force = sim.rotation(q) @ np.array([0, 0, -10])
    assert force[0] > 0


def test_velocity_reference_continuous_and_reaches_ten():
    assert sim.reference(0) == (0, 0)
    assert sim.reference(8) == (10, 0)
    assert sim.reference(30) == (10, 0)


def test_quaternion_derivative_rotates_about_body_y():
    dx = sim.quaternion_derivative(np.array([1.0, 0, 0, 0]), np.array([0.0, 2, 0]))
    np.testing.assert_array_equal(dx, [0, 0, 1, 0])


def test_controller_preserves_level_hover():
    a = sim.Aircraft(artifact(), "physical")
    x, integral = np.zeros(13), np.zeros(6)
    x[6] = 1
    desired, _ = sim.controller(x, integral, a, 0, 0.02, 10)
    np.testing.assert_allclose(desired, a.hover_w, atol=1e-10)
    np.testing.assert_allclose(integral, 0)


def test_tuned_pitch_loop_has_margin_in_provisional_local_model():
    # Diagnostic local slope, NOT proof of full nonlinear/free-flight stability.
    inertia = 0.0200949375623
    kp = inertia * sim.ATTITUDE_FREQUENCY[1] ** 2
    kd = inertia * 2 * sim.ATTITUDE_DAMPING * sim.ATTITUDE_FREQUENCY[1]
    ki = kp * sim.ATTITUDE_INTEGRAL_GAIN
    matrix = np.array(
        [[0, 1, 0], [(0.85 - kp) / inertia, (-0.04 - kd) / inertia, -ki / inertia], [1, 0, 0]]
    )
    assert np.max(np.linalg.eigvals(matrix).real) < 0
