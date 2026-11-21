"""Appendix integrator checks against rigid-body mechanics, not fitted flight data."""
import copy
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("appendix_simulation", ROOT/"sim/appendix/simulate.py")
sim = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = sim
SPEC.loader.exec_module(sim)


@pytest.fixture
def config():
    return json.loads((ROOT/"sim/appendix/model.json").read_text())


def test_free_fall_and_hover_balance(config):
    vehicle = sim.Loads(config, False)
    x = np.zeros(13); x[6] = 1.
    dx, _ = sim.derivative(x, vehicle, np.zeros(4), np.zeros(4), np.zeros(3))
    np.testing.assert_allclose(dx[3:6], [0, 0, sim.G], atol=1e-12)
    x, rotor, _, _ = sim.hover_trim(vehicle, 16.)
    dx, _ = sim.derivative(x, vehicle, rotor, np.zeros(4), np.zeros(3))
    np.testing.assert_allclose(dx, np.zeros(13), atol=1e-9)


def test_rotor_acceleration_conserves_total_angular_momentum(config):
    vehicle = sim.Loads(config, False)
    # Unequal rotor accelerations cause the opposite body yaw acceleration.
    rotor = np.full(4, 550.)
    rotor_dot = np.array([10., 0., 0., 0.])
    x = np.zeros(13); x[6] = 1.
    dx, _ = sim.derivative(x, vehicle, rotor, rotor_dot, np.zeros(3))
    h_dot = -vehicle.cfg.rotor_inertia_kg_m2*(rotor_dot@sim.SPIN)
    assert abs(vehicle.inertia[2]*dx[12]+h_dot) < 1e-12


def test_quaternion_yaw_rotation_is_body_to_world():
    q = np.array([np.cos(np.pi/4), 0., 0., np.sin(np.pi/4)])
    np.testing.assert_allclose(sim.rotation(q)@[1., 0., 0.], [0., 1., 0.], atol=1e-12)


def test_fractional_motor_delay_never_uses_future_command():
    history = [np.full(4, float(i)) for i in range(5)]
    motors = [dict(delay_s=.015) for _ in range(4)]
    np.testing.assert_allclose(sim.delayed_command(.04, history, .01, motors), 2.5)
    np.testing.assert_allclose(sim.delayed_command(0., history[:1], .01, motors), 0.)


@pytest.mark.parametrize("route", ["command_route", "measured"])
def test_identified_trim_is_an_equilibrium_without_stored_reference(config, route):
    vehicle = sim.Loads(config, True, route)
    x, rotor, _, residual = sim.hover_trim(vehicle, 16.)
    assert residual < 1e-7
    dx, _ = sim.derivative(x, vehicle, rotor, np.zeros(4), np.zeros(3))
    np.testing.assert_allclose(dx, np.zeros(13), atol=1e-8)
    air = np.array([3., 0., 0.])
    np.testing.assert_array_equal(vehicle(air, np.zeros(3), rotor, np.zeros(4)),
        sim.Loads(config, True, route)(air, np.zeros(3), rotor, np.zeros(4)))
    with pytest.raises(ValueError, match="extrapolation"):
        vehicle(np.array([12., 0., 0.]), np.zeros(3), rotor, np.zeros(4))


def test_short_hover_does_not_drift_from_numerical_initialization(config):
    c = copy.deepcopy(config)
    c["scenario"].update(target_m_s=0., hover_s=.05, ramp_s=.05, hold_s=.05, after_s=.05)
    for identified in (False, True):
        trace, metrics = sim.simulate(c, identified)
        assert metrics["status"] == "COMPLETED"
        assert trace.shape[1] == len(sim.HEADER)
        np.testing.assert_allclose(trace[:, 2:4], 0., atol=1e-7)
        np.testing.assert_allclose(trace[:, 4], -c["scenario"]["initial_height_m"], atol=1e-7)
        np.testing.assert_allclose(trace[:, 5:8], 0., atol=1e-7)


def test_public_evaluator_matches_original_saved_model_on_varied_states(config):
    """Extraction must preserve signed/reverse flow, local flow and every saved term."""
    import analysis as original
    from scripts.current_graybox_aircraft import instantaneous_data, correction
    from aerodynamics import instantaneous_data as exported_state, correction as exported_correction
    cfg = original.Settings(**config["settings"])
    rng = np.random.default_rng(20261031)
    vehicle = sim.Loads(config, False)
    for _ in range(24):
        air = rng.uniform([-7., -4., -.5], [7., 4., .5])
        rate, rotor, acceleration = rng.uniform(-.5, .5, 3), rng.uniform(400, 700, 4), rng.uniform(-30, 30, 4)
        physical = vehicle.propulsion(rate, rotor, acceleration)
        expected = instantaneous_data(air, rate, rotor, acceleration, cfg, physical)
        actual = exported_state(air, rate, rotor, acceleration, cfg, physical)
        for out in sim.AXES:
            for route in ("command_route", "measured"):
                model = config["models"][route][out]
                np.testing.assert_allclose(exported_correction(actual, cfg, out, model),
                    correction(expected, cfg, out, model), atol=1e-13, rtol=1e-12)
