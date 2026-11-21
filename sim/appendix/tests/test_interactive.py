"""Check the interactive controller/integrator, without opening a browser."""
from pathlib import Path
import csv
import io
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from interactive import Flight


@pytest.mark.parametrize("model,route", [("existing", "command_route"),
                                         ("identified", "command_route"),
                                         ("identified", "measured")])
def test_hover_is_equilibrium(model, route):
    f = Flight(model, route)
    for _ in range(40):
        f.advance({})
    assert f.status == "running"
    assert f.t == 2.0
    np.testing.assert_allclose(f.x[:6], [0, 0, -2, 0, 0, 0], atol=1e-5)
    rows = list(csv.reader(io.StringIO(f.csv())))
    assert len(rows) == 42  # header, initial state, forty 50 ms samples
    assert all(len(row) == len(rows[0]) for row in rows)


@pytest.mark.parametrize("model", ["existing", "identified"])
def test_motion_controls_and_pause_semantics(model):
    f = Flight(model)
    for _ in range(60):
        f.advance(dict(forward=1, right=.3, climb=.2, yaw=.15, speed=1.5))
    assert f.status == "running"
    assert f.x[0] > .3 and f.x[1] > .1
    assert f.x[2] < -2.1
    assert f.snapshot()["angles"][2] > 3
    before = f.x.copy()
    f.snapshot()  # requests for telemetry must not advance time
    np.testing.assert_array_equal(before, f.x)
    for _ in range(100):
        f.advance({})
    assert f.status == "running"
    assert np.linalg.norm(f.x[3:5]) < .6


def test_nonfinite_input_does_not_change_state():
    f = Flight("existing")
    before = f.x.copy()
    with pytest.raises(ValueError):
        f.advance({"wind": float("nan")})
    np.testing.assert_array_equal(before, f.x)


@pytest.mark.parametrize("model", ["existing", "identified"])
def test_mode2_neutral_and_pitch_sign(model):
    f = Flight(model)
    for _ in range(20):
        f.advance({"mode": "attitude"})
    np.testing.assert_allclose(f.x[:6], [0, 0, -2, 0, 0, 0], atol=1e-5)
    for _ in range(20):
        f.advance({"mode": "attitude", "forward": .25})
    assert f.status == "running"
    assert f.snapshot()["angles"][1] < -3  # forward stick -> nose down
    assert f.x[3] > .2


def test_mode2_collective_has_no_altitude_feedback():
    f = Flight("existing")
    for _ in range(20):
        f.advance({"mode": "attitude", "climb": .15})
    assert f.x[5] < -.3
    assert f.height == 2.0  # collective input did not move an altitude setpoint


def test_braking_into_four_m_s_tailwind_stays_controlled():
    # Recorded failure: reverse speed request -3.5 m/s, then release at 4.2 s.
    f = Flight('identified', 'command_route')
    pitches, heights = [], []
    for i in range(154):
        forward = -1.0 if 24 <= i < 83 else 0.0
        s = f.advance(dict(forward=forward, speed=3.5, wind=4.0))
        assert s['status'] == 'running'
        pitches.append(s['angles'][1]); heights.append(-s['position'][2])
    assert min(pitches) > -18  # previously -42.5 deg
    assert min(heights) > 1.8  # previously 0.96 m from 2 m start
    rows = list(csv.DictReader(io.StringIO(f.csv())))
    assert rows[-1]['speed_command_m_s'] == '3.5'
    assert rows[-1]['load_model'] == 'identified'


def test_torque_saturation_is_visible_without_motor_saturation():
    f = Flight('existing')
    f.x[10] = -30.0
    f.attitude_control(f.trim_rotation, f.trim_control[0])
    assert f.torque_saturated and f.snapshot()['saturated']
