"""Finite-jerk reference only: these tests do not certify flight feasibility."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from simulate_koil_pass import PassProfile
from simulate_graybox_forward_flight import ControllerSettings, velocity_acceleration
from run_koil_px4_sitl import yaw_ready, VELOCITY_MASK, POSITION_MASK
from types import SimpleNamespace


@pytest.mark.parametrize("acceleration", [3, 4, 5])
def test_reference_hold_stop_and_distance(acceleration):
    profile = PassProfile(acceleration=acceleration)
    ramp = profile.ramp_duration(5)
    start = profile.hover_s + ramp
    assert profile.reference(0, 5) == (0, 0)
    assert profile.reference(start, 5) == (5, 0)
    assert profile.reference(start + 1.49, 5) == (5, 0)
    assert profile.reference(start + 1.5 + ramp + 0.1, 5) == (0, 0)
    t = np.linspace(0, start + 1.5 + ramp + 1, 20001)
    refs = np.array([profile.reference(time, 5) for time in t])
    assert np.max(np.abs(refs[:, 1])) <= acceleration
    integral = np.sum((refs[1:, 0] + refs[:-1, 0]) * np.diff(t) / 2)
    assert integral == pytest.approx(5 * (ramp + 1.5), abs=1e-5)


def test_acceleration_continuous_at_ramp_breakpoints():
    profile = PassProfile()
    for time in (0, profile.jerk_ramp_s, profile.ramp_duration(5) - profile.jerk_ramp_s, profile.ramp_duration(5)):
        np.testing.assert_allclose(profile.ramp(time - 1e-8, 5), profile.ramp(time + 1e-8, 5), atol=1e-6)


def test_position_hold_uses_reference_endpoint_after_braking():
    profile = PassProfile()
    end = profile.hover_s + 2 * profile.ramp_duration(5) + profile.hold_s
    assert profile.position_reference(end - 0.001, 5) is None
    np.testing.assert_allclose(profile.position_reference(end, 5), [14.75, 0])


def test_two_second_hold_changes_braking_and_endpoint():
    profile = PassProfile(hold_s=2.)
    start = profile.hover_s+profile.ramp_duration(5)
    assert profile.reference(start+1.99,5) == (5,0)
    end = start+2.+profile.ramp_duration(5)
    np.testing.assert_allclose(profile.position_reference(end,5),[17.25,0])


@pytest.mark.parametrize("target,acceleration",[(2,4),(3,2),(3,4),(4,4)])
def test_screening_reference_endpoint(target,acceleration):
    profile = PassProfile(hold_s=1,acceleration=acceleration)
    ramp = profile.ramp_duration(target)
    end = profile.hover_s+2*ramp+1
    assert profile.reference(end,target) == (0,0)
    np.testing.assert_allclose(profile.position_reference(end,target),[target*(ramp+1),0])


def test_offboard_yaw_is_enabled_and_required_before_translation():
    assert not (VELOCITY_MASK & 1024)  # MAVLink YAW_IGNORE must be clear.
    assert not (POSITION_MASK & 1024)
    assert yaw_ready(SimpleNamespace(yaw=.01,yawspeed=.01))
    assert not yaw_ready(SimpleNamespace(yaw=np.pi/2,yawspeed=0))
    assert not yaw_ready(SimpleNamespace(yaw=0,yawspeed=.2))


def test_antiwindup_blocks_growth_but_allows_unwinding():
    settings = ControllerSettings(velocity_kp=3, velocity_ki=1, antiwindup=True)
    accel, integral = velocity_acceleration(np.array([10., -1., 0.]),
        np.array([0., 10., 0.]), np.zeros(3), 0.02, np.ones(3) * 6, settings)
    np.testing.assert_allclose(integral, [0, 9.98, 0])
    np.testing.assert_allclose(accel, [6, 6, 0])


def test_default_velocity_pi_matches_original_formula():
    error, old = np.array([1., -2., .5]), np.zeros(3)
    ff = np.array([4., 0., 0.])
    accel, integral = velocity_acceleration(error, old, ff, 0.02,
        np.array([4., 4., 3.]), ControllerSettings())
    np.testing.assert_allclose(integral, .02 * error)
    np.testing.assert_allclose(accel, np.clip(ff + .9 * error + .15 * integral,
                                           [-4, -4, -3], [4, 4, 3]))
