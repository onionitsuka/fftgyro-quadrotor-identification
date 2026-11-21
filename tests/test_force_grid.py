import csv
import hashlib
import json

import pytest

from fftgyro_quad_id.sync.force_grid import LEPTRINO_DELAY_NS_1200_HZ, infer_grid, reconstruct
from fftgyro_quad_id.sync.pi_affine import fit, fit_smooth_interval, read_exchanges


def test_batch_reconstruction_retains_fixed_latency():
    # At 1000 Hz five samples read in each batch, with fixed latency 200 us.
    received = [10_004_200_000] * 5 + [10_009_200_000] * 5
    grid, report = infer_grid(received, 1000)
    assert grid == [10_000_200_000 + i * 1_000_000 for i in range(10)]
    assert all(t <= r for t, r in zip(grid, received, strict=False))
    assert report["advance_from_receive_ms"]["max"] == 4


def test_integer_rounding_and_large_uptime():
    origin = 10**16
    received = [origin + (i * 10**9 + 600) // 1200 for i in range(1201)]
    grid, report = infer_grid(received)
    assert grid == received
    assert grid[-1] - grid[0] == 10**9
    assert report["nominal_span_s"] == 1


def test_holdout_exposes_variable_offset():
    received = [10**10 + i * 10**6 + (2_000_000 if i < 1000 else 0) for i in range(2000)]
    # Avoid clock reversal at block boundary while retaining a faster arrival.
    received[1000] = received[999]
    _, report = infer_grid(received, 1000)
    held = report["alternating_second_holdout"]
    assert held["samples"] == 1000
    assert held["causal_violations"] > 0
    assert held["maximum_excess_ms"] == 2


@pytest.mark.parametrize(
    "received,rate",
    [
        ([], 1200),
        ([1], 1200),
        ([2, 1], 1200),
        ([-1, 2], 1200),
        ([1, 2], 0),
        ([1, 2], 1.2),
        ([1, 2], True),
        ([1, 2], 1200),
    ],
)
def test_bad_input(received, rate):
    with pytest.raises(ValueError):
        infer_grid(received, rate)


@pytest.fixture
def aligned(tmp_path):
    source = tmp_path / "aligned"
    source.mkdir()
    (tmp_path / "clock").mkdir()
    manifest = dict(
        leptrino_boot_id="l",
        leptrino_boot_id_after="l",
        aircraft_boot_id="a",
        aircraft_boot_id_after="a",
    )
    manifest_file = tmp_path / "run.json"
    manifest_file.write_text(json.dumps(manifest))
    clock = tmp_path / "clock/exchanges.csv"
    clock.write_text(
        "client_send_ns,server_receive_ns,server_send_ns,client_receive_ns\n"
        "1000000000,6000100000,6000200000,1000400000\n"
        "2000000000,7000100000,7000200000,2000400000\n"
        "3000000000,8000100000,8000200000,3000400000\n"
    )
    pi = fit(read_exchanges(clock))
    meta = dict(
        state="ALIGNED_CONDITIONAL_NOT_QUALIFIED",
        force_samples=6,
        leptrino_boot_id="l",
        aircraft_boot_id="a",
        source_manifest_sha256=hashlib.sha256(manifest_file.read_bytes()).hexdigest(),
        pi_model=dict(
            client_origin_ns=pi.client_origin_ns,
            server_origin_ns=pi.server_origin_ns,
            vertices=pi.vertices,
        ),
    )
    (source / "summary.json").write_text(json.dumps(meta))
    with (source / "force_aligned.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["index", "receive_monotonic_ns", "payload_hex", "fx_N"])
        for i in range(6):
            payload = bytes([20, 255, 0x32, 0]) + bytes(15) + bytes([i])
            writer.writerow([i, 2_000_000_000 + (i // 3) * 2_500_000, payload.hex(), i])
    return source


def test_full_output_preserves_source_and_refuses_overwrite(aligned):
    before = (aligned / "force_aligned.csv").read_bytes()
    target = aligned.parent / "grid"
    result = reconstruct(aligned, target)
    assert result["state"] == "NOMINAL_GRID_ESTIMATED_WITH_VENDOR_DELAY"
    assert result["vendor_sensor_delay_ns"] == LEPTRINO_DELAY_NS_1200_HZ
    assert (aligned / "force_aligned.csv").read_bytes() == before
    with (target / "force_grid.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [int(r["fx_N"]) for r in rows] == list(range(6))
    assert len({r["nominal_sample_leptrino_ns"] for r in rows}) == 6
    for row in rows:
        assert int(row["nominal_sample_leptrino_ns"]) <= int(row["receive_monotonic_ns"])
        assert (
            int(row["nominal_sample_leptrino_ns"]) - int(row["estimated_measurement_leptrino_ns"])
        ) == LEPTRINO_DELAY_NS_1200_HZ
        assert (
            int(row["nominal_sample_aircraft_clock_lower_ns"])
            <= int(row["nominal_sample_aircraft_estimate_ns"])
            <= int(row["nominal_sample_aircraft_clock_upper_ns"])
        )
    with pytest.raises(FileExistsError):
        reconstruct(aligned, target)


def test_reconstruction_reuses_smooth_interval_model(aligned):
    clock = aligned.parent / "clock/exchanges.csv"
    model = fit_smooth_interval(read_exchanges(clock))
    summary = json.loads((aligned / "summary.json").read_text())
    summary["pi_model"] = {
        "method": "smooth_interval",
        "anchors": len(model.stamps),
        "first_ns": model.first_ns,
        "last_ns": model.last_ns,
    }
    (aligned / "summary.json").write_text(json.dumps(summary))
    result = reconstruct(aligned, aligned.parent / "grid-smooth")
    assert result["state"] == "NOMINAL_GRID_ESTIMATED_WITH_VENDOR_DELAY"


@pytest.mark.parametrize("problem", ["counter", "index", "manifest", "coverage", "clock"])
def test_reject_inconsistent_sources(aligned, problem):
    source = aligned / "force_aligned.csv"
    if problem in ("counter", "index", "coverage"):
        with source.open(newline="") as stream:
            reader = csv.DictReader(stream)
            fields = reader.fieldnames
            rows = list(reader)
        if problem == "counter":
            rows[2]["payload_hex"] = rows[1]["payload_hex"]
        elif problem == "index":
            rows[2]["index"] = "3"
        else:
            for row in rows:
                row["receive_monotonic_ns"] = str(500_000_000 + int(row["index"]) * 833333)
        with source.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    elif problem == "manifest":
        (aligned.parent / "run.json").write_text("{}")
    else:
        clock = aligned.parent / "clock/exchanges.csv"
        clock.write_text(clock.read_text().replace("6000100000", "6000150000"))
    with pytest.raises(ValueError):
        reconstruct(aligned, aligned.parent / "grid")
