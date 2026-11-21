import pytest

from fftgyro_quad_id.sync.rpm_clock_compare import (
    compare,
    fit_local,
    fit_piecewise_midpoint,
    fit_smooth_interval,
)
from fftgyro_quad_id.sync.rpm_pid import Exchange, fit_clock


def make_pairs(curvature=0.0, count=121):
    pairs = []
    for i in range(count):
        x = i * 0.5
        truth = 10**12 + round((x + curvature * x * x) * 1e9)
        pairs.append(Exchange(i, 1_000_000 + i * 500_000, truth - 40_000, truth + 40_000))
    return pairs


def test_constant_clock_local_continuity_and_bounds():
    model = fit_local(make_pairs())
    for center in model.centers[1:-1]:
        a, b, c = [model.map(center + offset)[0] for offset in (-1, 0, 1)]
        assert 999 <= b - a <= 1001
        assert 999 <= c - b <= 1001
        _, low, high = model.map(center)
        assert low <= 10**12 + (center - 1_000_000) * 1000 <= high


def test_local_tracks_smooth_drift_when_global_rejected():
    pairs = make_pairs(curvature=1e-6)
    with pytest.raises(ValueError, match="affine"):
        fit_clock(pairs)
    model = fit_local(pairs)
    errors = [abs(model.map(e.nano_us)[0] - (e.sent_ns + e.received_ns) // 2) for e in pairs]
    assert max(errors) < 50_000
    summary, _ = compare(pairs, [e.nano_us for e in pairs])
    assert summary["models"]["global"]["fit"] == "REJECTED"
    assert summary["models"]["local"]["fit"] == "OK"
    assert summary["models"]["local"]["held_out"]["violations"] == 0


def test_local_refuses_extrapolation():
    model = fit_local(make_pairs())
    with pytest.raises(ValueError, match="extrapolate"):
        model.map(model.centers[0] - 1)


def test_weighting_does_not_discard_constraints():
    pairs = make_pairs()
    a = fit_clock(pairs)
    b = fit_clock(pairs, weight_by_rtt=True)
    assert a.vertices == b.vertices


def test_dense_exchanges_can_tighten_bounds_without_changing_truth():
    pairs = []
    for i in range(201):
        truth = 10**12 + i * 100_010_000
        delay = 1_000_000 if i % 10 == 0 else 100_000
        pairs.append(Exchange(i, i * 100_000, truth - delay, truth + delay))
    dense, sparse = fit_clock(pairs), fit_clock(pairs[::10])
    for nano in (1_000_000, 10_000_000, 19_000_000):
        _, low, high = dense.map(nano)
        _, sparse_low, sparse_high = sparse.map(nano)
        assert high - low < sparse_high - sparse_low
        assert low <= 10**12 + round(nano * 1000 * 1.0001) <= high


def test_smooth_interval_rejects_midpoint_jitter_but_preserves_bounds():
    pairs = []
    for index in range(41):
        truth = 10**12 + index * 100_000_000
        asymmetric_delay = 300_000 if index % 2 else -300_000
        pairs.append(
            Exchange(
                index,
                1_000_000 + index * 100_000,
                truth - 500_000 + asymmetric_delay,
                truth + 500_000 + asymmetric_delay,
            )
        )
    smooth = fit_smooth_interval(pairs, slack_ns=0)
    midpoint = fit_piecewise_midpoint(pairs, slack_ns=0)
    smooth_values = [smooth.map(pair.nano_us)[0] for pair in pairs]
    midpoint_values = [midpoint.map(pair.nano_us)[0] for pair in pairs]
    smooth_curvature = max(
        abs(a - 2 * b + c)
        for a, b, c in zip(smooth_values, smooth_values[1:], smooth_values[2:], strict=False)
    )
    midpoint_curvature = max(
        abs(a - 2 * b + c)
        for a, b, c in zip(midpoint_values, midpoint_values[1:], midpoint_values[2:], strict=False)
    )
    assert smooth_curvature < midpoint_curvature / 10
    assert all(
        pair.sent_ns <= estimate <= pair.received_ns
        for pair, estimate in zip(pairs, smooth_values, strict=True)
    )
    assert all(b > a for a, b in zip(smooth_values, smooth_values[1:], strict=False))


@pytest.mark.parametrize("window,step", [(0, 1), (6, 4), (float("nan"), 2)])
def test_invalid_window(window, step):
    with pytest.raises(ValueError):
        fit_local(make_pairs(), window_s=window, step_s=step)
