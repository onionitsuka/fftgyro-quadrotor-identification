import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/"scripts"))
from hover_pitch_profile import HoverPitchProfile,pitch_quaternion,altitude_thrust


def test_smooth_window_and_unit_quaternion():
    p=HoverPitchProfile(10,2)
    assert p.duration_s == 12
    assert p.reference_deg(2) == p.reference_deg(8) == 0
    assert abs(p.reference_deg(2+1e-5)) < 1e-8
    assert p.reference_deg(5) == pytest.approx(-10)
    assert np.linalg.norm(pitch_quaternion(10)) == pytest.approx(1)


def test_common_altitude_controller_correct_sign_and_bounds():
    hover,_=altitude_thrust(-2,0,0,.02,0)
    below,_=altitude_thrust(-1.5,0,0,.02,0)
    above,_=altitude_thrust(-2.5,0,0,.02,0)
    assert above < hover < below
    thrust,integral=altitude_thrust(10,10,0,.02,10)
    assert thrust == .85 and integral == 0


@pytest.mark.parametrize("amplitude", [20, 30, 45])
def test_large_offline_excitation(amplitude):
    profile = HoverPitchProfile(amplitude, 4)
    assert profile.reference_deg(8) == pytest.approx(-amplitude)
    assert np.linalg.norm(pitch_quaternion(-amplitude)) == pytest.approx(1)
    thrust, integral = altitude_thrust(-2, 0, 0, .02, -amplitude)
    assert thrust == pytest.approx(.54 / np.cos(np.deg2rad(amplitude)))
    assert thrust < .85 and integral == 0
