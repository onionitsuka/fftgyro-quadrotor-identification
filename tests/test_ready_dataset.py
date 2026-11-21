from __future__ import annotations

import numpy as np

from fftgyro_quad_id.analysis.ready_dataset import (
    _analysis_rows,
    _linear_with_gap,
    _unwrap_degrees,
    _zoh,
    fit_px4_clock,
)
from fftgyro_quad_id.sync.pi_affine import (
    fit_piecewise_midpoint as fit_piecewise_pi,
)
from fftgyro_quad_id.sync.pi_affine import (
    fit_smooth_interval as fit_smooth_pi,
)
from fftgyro_quad_id.sync.rpm_clock_compare import fit_piecewise_midpoint
from fftgyro_quad_id.sync.rpm_pid import Exchange


def test_piecewise_midpoint_clock_interpolates_causal_brackets() -> None:
    exchanges = [
        Exchange(1, 1000, 1_000_000, 1_020_000),
        Exchange(2, 2000, 2_000_000, 2_040_000),
        Exchange(3, 3000, 3_000_000, 3_060_000),
    ]
    model = fit_piecewise_midpoint(exchanges, slack_ns=0)
    estimate, lower, upper = model.map(1500)
    assert estimate == 1_515_000
    assert lower <= estimate <= upper


def test_piecewise_pi_clock_interpolates_ntp_offsets() -> None:
    model = fit_piecewise_pi(
        [
            (100, 1100, 1110, 120),
            (200, 1200, 1210, 220),
            (300, 1300, 1310, 320),
        ]
    )
    estimate, lower, upper = model.map(160)
    assert estimate == 1155
    assert lower <= estimate <= upper


def test_smooth_pi_clock_stays_inside_exchange_intervals() -> None:
    rows = []
    for index in range(20):
        client = 1_000_000_000 + index * 100_000_000
        server = client + 50_000_000_000
        jitter = 200_000 if index % 2 else -200_000
        rows.append(
            (client, server - 500_000 + jitter, server + 500_000 + jitter, client + 1_000_000)
        )
    model = fit_smooth_pi(rows)
    values = [model.map((row[0] + row[3]) // 2)[0] for row in rows]
    assert all(b > a for a, b in zip(values, values[1:], strict=False))
    for value, row in zip(values, rows, strict=True):
        center = (row[0] + row[3]) // 2
        assert center + row[1] - row[3] <= value <= center + row[2] - row[0]


def test_resampling_refuses_to_bridge_gap() -> None:
    source = np.asarray([0, 10_000_000, 100_000_000], dtype=np.int64)
    values = np.asarray([0.0, 1.0, 10.0])
    grid = np.asarray([5_000_000, 50_000_000], dtype=np.int64)
    interpolated, valid = _linear_with_gap(source, values, grid, max_gap_ns=30_000_000)
    assert valid.tolist() == [True, False]
    assert np.isnan(interpolated[1])


def test_zoh_has_age_limit() -> None:
    values, valid, _ = _zoh(
        np.asarray([0, 10_000_000], dtype=np.int64),
        np.asarray([1.0, 2.0]),
        np.asarray([20_000_000, 50_000_000], dtype=np.int64),
    )
    assert values.tolist() == [2.0, 2.0]
    assert valid.tolist() == [True, False]


def test_encoder_degrees_unwrap_before_interpolation() -> None:
    values = _unwrap_degrees(np.asarray([178.0, 179.0, -179.0, -178.0]))
    assert np.allclose(values, [178.0, 179.0, 181.0, 182.0])


def test_partial_prefix_excludes_segment_25() -> None:
    rows = [
        {"analysis_window": "0", "stage": "SEGMENT", "segment_id": "T-W-005-024"},
        {"analysis_window": "0", "stage": "SEGMENT", "segment_id": "T-W-005-025"},
    ]
    assert _analysis_rows(rows, 765).tolist() == [True, False]


def test_px4_clock_reports_residuals() -> None:
    rows = [
        {
            "px4_boot_ms": str(index),
            "feedback_received_ns": str(1_000_000_000 + index * 1_000_000 + index % 3 * 1000),
        }
        for index in range(1, 101)
    ]
    slope, intercept, details = fit_px4_clock(rows)
    assert abs(slope - 1.0) < 1e-3
    assert intercept > 0
    assert details["pairs"] == 100
