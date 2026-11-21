"""Appendix integrator checks against rigid-body mechanics, not fitted flight data."""
import copy
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("appendix_simulation", ROOT/"simulate.py")
sim = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = sim
SPEC.loader.exec_module(sim)


@pytest.fixture
def config():
    return json.loads((ROOT/"model.json").read_text())


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
def test_identified_trim_is_an_equilibrium_without_offset(config, route):
    vehicle = sim.Loads(config, True, route)
    x, rotor, _, residual = sim.hover_trim(vehicle, 16.)
    assert residual < 1e-7
    dx, _ = sim.derivative(x, vehicle, rotor, np.zeros(4), np.zeros(3))
    np.testing.assert_allclose(dx, np.zeros(13), atol=1e-8)
    # The load is a pure function of the current state: no stored reference.
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


@pytest.mark.parametrize("voltage", [13., 14.5, 16.])
def test_kaufmann_units_monotonicity_and_inverse(config, voltage):
    commands = np.linspace(.2, .40, 19)[:, None]*np.ones((1,4))
    speeds = np.array([sim.motor_speed(d, voltage, config) for d in commands])
    assert np.all(np.diff(speeds, axis=0) > 0)
    # The saved physical-voltage coefficients must match the normalized basis.
    coef = np.array([m['coefficients_volts_basis'] for m in config['motor_models']])
    d = np.array([.23, .32, .29, .38])
    expected = (coef[:,0]+coef[:,1]*voltage+coef[:,2]*np.sqrt(d)
                +coef[:,3]*d+coef[:,4]*voltage*np.sqrt(d))*2*np.pi/60
    np.testing.assert_allclose(sim.motor_speed(d, voltage, config), expected, atol=1e-10)
    np.testing.assert_allclose(sim.motor_command(expected, voltage, config), d, atol=3e-10)
