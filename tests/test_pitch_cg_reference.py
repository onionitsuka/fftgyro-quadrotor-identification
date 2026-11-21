import numpy as np
import pytest

from fftgyro_quad_id.analysis.pitch_cg_reference import (
    infer_forward_shift,
    shift_pitch_origin,
)


def test_stationary_and_powered_consistency():
    dx = 0.006
    weight = 15.6
    stopped = -dx * weight
    estimate = infer_forward_shift(stopped, weight)
    assert estimate == pytest.approx(dx)
    assert shift_pitch_origin(stopped, -weight, dx) == pytest.approx(0)
    # In powered static equilibrium, support + thrust = weight.
    for thrust in (0, 5, 12, 18):
        support = weight - thrust
        measured_old = 0.03 - dx * support
        prop_old = 0.03 + dx * thrust
        measured_new = shift_pitch_origin(measured_old, -support, dx)
        prop_new = shift_pitch_origin(prop_old, thrust, dx)
        assert measured_new == pytest.approx(prop_new)


def test_cross_product_sign():
    r = np.array([0.004, 0, 0])
    force = np.array([0.2, 0.1, 8.0])
    new_flu = -np.cross(r, force)
    assert shift_pitch_origin(0, force[2], r[0]) == pytest.approx(-new_flu[1])


@pytest.mark.parametrize("load", [0, -1, float("nan")])
def test_invalid_reference(load):
    with pytest.raises(ValueError):
        infer_forward_shift(-0.1, load)
