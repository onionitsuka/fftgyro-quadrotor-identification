import numpy as np
import pytest

from fftgyro_quad_id.identification.motor_asymmetric import simulate_asymmetric
from fftgyro_quad_id.identification.motor_single import simulate_first_order


def test_matches_existing_symmetric_model_and_preserves_zoh():
    t = np.arange(0, 3, 0.01)
    target = np.where(t < 1, 3000.0, 1000.0)
    np.testing.assert_allclose(
        simulate_asymmetric(t, target, 0.2, 0.2, 0, 0),
        simulate_first_order(t, target, 0.2, 0),
        atol=1e-10,
    )


def test_asymmetric_decay_and_delay():
    t = np.arange(0, 2, 0.01)
    target = np.where(t < 0.5, 0.0, 1000.0)
    y = simulate_asymmetric(t, target, 0.1, 0.3, 0.1, 0)
    assert y[59] == 0 and y[-1] > 999
    down = simulate_asymmetric(t, np.zeros(len(t)), 0.1, 0.3, 0, 1000)
    np.testing.assert_allclose(down, 1000 * np.exp(-t / 0.3), atol=1e-10)
    with pytest.raises(ValueError):
        simulate_asymmetric(t, target, 0, 0.3, 0, 0)
