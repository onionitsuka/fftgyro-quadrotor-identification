import numpy as np
import pytest

from fftgyro_quad_id.filters.load_lowpass import filter_contiguous_load_blocks


def test_preserves_one_hz_and_rejects_twenty_hz():
    time = np.arange(0, 10, 0.01)
    low = np.sin(2 * np.pi * time)
    high = 0.5 * np.sin(2 * np.pi * 20 * time)
    filtered = filter_contiguous_load_blocks(
        (low + high)[:, None],
        np.arange(len(time)),
        sample_rate_hz=100,
        cutoff_hz=5,
        order=4,
    )[:, 0]
    interior = slice(100, -100)
    assert np.sqrt(np.mean((filtered[interior] - low[interior]) ** 2)) < 0.002


def test_does_not_bridge_source_gap():
    indices = np.r_[np.arange(100), np.arange(200, 300)]
    values = np.r_[np.zeros(100), np.ones(100)][:, None]
    filtered = filter_contiguous_load_blocks(
        values,
        indices,
        sample_rate_hz=100,
        cutoff_hz=5,
    )[:, 0]
    np.testing.assert_allclose(filtered[:100], 0, atol=1e-12)
    np.testing.assert_allclose(filtered[100:], 1, atol=1e-12)


def test_rejects_invalid_cutoff():
    with pytest.raises(ValueError, match="Nyquist"):
        filter_contiguous_load_blocks(
            np.zeros((10, 6)),
            np.arange(10),
            sample_rate_hz=100,
            cutoff_hz=50,
        )
