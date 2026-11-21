import pytest
from fftgyro_quad_id.sync.pi_affine import fit


def exchanges():
    # asymmetric 1/4 ms paths, 80 ppm drift; true server=client*1.00008+3e9
    return [(t, round((t+1_000_000)*1.00008+3e9),
             round((t+1_100_000)*1.00008+3e9), t+5_100_000)
            for t in range(10_000_000_000, 41_000_000_000, 1_000_000_000)]


def test_asymmetric_drift_bounds_contain_truth():
    model = fit(exchanges())
    for t in range(10_000_000_000, 40_000_000_000, 70_000_000):
        est, low, high = model.map(t)
        assert low <= round(t*1.00008+3e9) <= high
        assert low <= est <= high
        assert high-low < 6_000_000


def test_no_extrapolation():
    model = fit(exchanges())
    with pytest.raises(ValueError):
        model.map(model.first_ns-1)


def test_inconsistent_anchor_is_not_dropped():
    rows = exchanges()
    a,b,c,d = rows[15]
    rows[15] = (a,b+100_000_000,c+100_000_000,d)
    with pytest.raises(ValueError, match="affine"):
        fit(rows)


@pytest.mark.parametrize("rows", [[], exchanges()[:2], exchanges()[::-1]])
def test_invalid(rows):
    with pytest.raises(ValueError):
        fit(rows)
