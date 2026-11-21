import csv
import hashlib
import json

import pytest

from fftgyro_quad_id.sync.delay_sensitivity import analyze, delta_stats, ordered, preceding, rpm_at


def test_preceding_does_not_look_ahead():
    assert preceding([10, 20, 30], 10) == 0
    assert preceding([10, 20, 30], 19) == 0
    assert preceding([10, 20, 30], 20) == 1


@pytest.mark.parametrize("stamp", [9, 30, 31])
def test_no_extrapolation(stamp):
    with pytest.raises(ValueError):
        preceding([10, 20, 30], stamp)


def test_rpm_interval_average():
    assert rpm_at([0, 10_000_000, 30_000_000], 5_000_000) == (0, 3000)
    assert rpm_at([0, 10_000_000, 30_000_000], 10_000_000) == (1, 1500)


def test_delta_statistics():
    assert delta_stats([-2, 0, 2]) == pytest.approx(
        {"mean_absolute": 4/3, "rms": (8/3)**.5, "maximum_absolute": 2})


@pytest.mark.parametrize("values", [[], [float("nan")], [float("inf")]])
def test_bad_values(values):
    with pytest.raises(ValueError):
        delta_stats(values)


@pytest.mark.parametrize("times", [[], [1], [1, 1], [2, 1]])
def test_bad_times(times):
    with pytest.raises(ValueError):
        ordered(times)


def test_full_scenarios_common_rows_and_earlier_shift(tmp_path):
    grid, aligned = tmp_path / "grid", tmp_path / "aligned"
    grid.mkdir()
    aligned.mkdir()
    (tmp_path / "pid").mkdir()
    manifest = dict(aircraft_boot_id="a", aircraft_boot_id_after="a",
                    leptrino_boot_id="l", leptrino_boot_id_after="l")
    mf = tmp_path / "run.json"
    mf.write_text(json.dumps(manifest))
    am = aligned / "summary.json"
    am.write_text(json.dumps(dict(state="ALIGNED_CONDITIONAL_NOT_QUALIFIED",
                                 aircraft_boot_id="a", leptrino_boot_id="l")))
    (grid / "summary.json").write_text(json.dumps(dict(
        state="NOMINAL_GRID_ESTIMATED_NOT_CALIBRATED", samples=21,
        source_manifest_sha256=hashlib.sha256(mf.read_bytes()).hexdigest(),
        source_sha256={"summary.json": hashlib.sha256(am.read_bytes()).hexdigest()},
        aircraft_boot_id="a", leptrino_boot_id="l")))
    force = grid / "force_grid.csv"
    with force.open("w", newline="") as stream:
        w = csv.writer(stream)
        w.writerow(["index", "nominal_sample_aircraft_estimate_ns",
                    "fx_N", "fy_N", "fz_N", "mx_Nm", "my_Nm", "mz_Nm"])
        for i in range(21):
            w.writerow([i, i * 5_000_000, i, 0, 0, 0, 0, 0])
    with (tmp_path / "pid/control.csv").open("w", newline="") as stream:
        w = csv.writer(stream)
        w.writerow(["sent_ns", "m1", "m2", "m3", "m4"])
        for i in range(11):
            w.writerow([i * 10_000_000, *([i / 100] * 4)])
    with (aligned / "rpm_edges_aligned.csv").open("w", newline="") as stream:
        w = csv.writer(stream)
        w.writerow(["channel", "aircraft_estimate_ns", "aircraft_lower_ns",
                    "aircraft_upper_ns", "within_qt_anchors"])
        for ch in range(1, 5):
            for i in range(11):
                t = i * 10_000_000
                w.writerow([ch, t, t, t, True])
    before = force.read_bytes()
    target = tmp_path / "sensitivity"
    report = analyze(grid, aligned, target)
    assert force.read_bytes() == before
    assert report["common_force_samples"] == 18
    assert report["first_force_index"] == 2
    assert report["last_force_index"] == 19
    assert report["scenarios"][0]["pid_row_changed_percent"] == 0
    assert report["scenarios"][-1]["pid_row_changed_percent"] == 100
    assert report["scenarios"][-1]["command_difference_from_zero_delay"]["m1"]["rms"] == pytest.approx(.01)
    with (target / "pairings.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 90
    baseline = {r["force_index"]: r for r in rows if r["assumed_delay_ms"] == "0"}
    for row in rows:
        base = baseline[row["force_index"]]
        assert row["fx_N"] == base["fx_N"]
        assert int(row["shifted_aircraft_ns"]) == int(base["shifted_aircraft_ns"]) - int(row["assumed_delay_ms"]) * 1_000_000
        assert float(row["rpm_s1"]) == 3000
    with pytest.raises(FileExistsError):
        analyze(grid, aligned, target)
