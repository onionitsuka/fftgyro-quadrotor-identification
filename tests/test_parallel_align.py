import csv
import json

import pytest

from fftgyro_quad_id.sync.parallel_align import align_run, match_pid


@pytest.mark.parametrize(
    "estimate,low,high,expected",
    [
        (15, 12, 19, (0, False)),
        (19, 18, 21, (0, True)),
        (20, 19, 21, (1, True)),
        (10, 9, 11, (0, True)),
        (30, 29, 31, (2, True)),
        (9, 8, 10, ("", "")),
    ],
)
def test_pid_boundary_ambiguity(estimate, low, high, expected):
    assert match_pid([10, 20, 30], estimate, low, high) == expected


@pytest.fixture
def run(tmp_path):
    for folder in ("rpm", "clock", "leptrino", "pid"):
        (tmp_path / folder).mkdir()

    def js(path, data):
        (tmp_path / path).write_text(json.dumps(data), encoding="utf-8")

    js(
        "run.json",
        dict(
            state="CAPTURED",
            aircraft_boot_id="a",
            leptrino_boot_id="l",
            aircraft_boot_id_after="a",
            leptrino_boot_id_after="l",
        ),
    )
    js("clock/summary.json", dict(state="COMPLETE", client_boot_id="l", server_boot_id="a"))
    js("leptrino/summary.json", dict(state="COMPLETE", boot_id="l", samples=8))
    (tmp_path / "rpm/result.txt").write_text("COMPLETE\n")
    with (tmp_path / "rpm/rpm.csv").open("w", newline="") as stream:
        w = csv.writer(stream)
        w.writerow(["host_ns", "kind", "capture_id", "sequence", "nano_us", "value"])
        sequence = 0
        w.writerow([50_000_000_000, "B", 123, sequence, 0, 0])
        sequence += 1
        for i in range(201):
            host = 51_000_000_000 + i * 100_000_000
            nano = 1_000_000 + i * 100_000
            w.writerow([host - 200_000, "Q", 123, 0, 0, i])
            w.writerow([host + 800_000, "T", 123, sequence, nano, i])
            sequence += 1
            w.writerow([host + 800_000, "E", 123, sequence, nano - 500, 1 + i % 4])
            sequence += 1
        w.writerow([71_100_000_000, "Z", 123, sequence, 21_100_000, 0])
    with (tmp_path / "clock/exchanges.csv").open("w", newline="") as stream:
        w = csv.writer(stream)
        w.writerow(["client_send_ns", "server_receive_ns", "server_send_ns", "client_receive_ns"])
        for t in range(10_000_000_000, 31_000_000_000, 1_000_000_000):
            w.writerow([t, t + 50_001_000_000, t + 50_001_100_000, t + 5_100_000])
    with (tmp_path / "leptrino/samples.csv").open("w", newline="") as stream:
        w = csv.writer(stream)
        w.writerow(["index", "receive_monotonic_ns", "fx_N"])
        for i in range(8):
            w.writerow([i, 12_000_000_000 + i * 100_000_000, i])
    (tmp_path / "pid/result.txt").write_text("status=COMPLETE\nrelease_ack=1\n")
    (tmp_path / "pid/control.csv").write_text("sent_ns\n62000000000\n63000000000\n")
    return tmp_path


def test_full_join_and_preserved_times(run):
    summary = align_run(run, run / "aligned")
    assert summary["force_samples"] == 8
    assert sum(summary["rpm_edges_by_channel"].values()) == 201
    assert summary["rpm_edges_outside_anchors"] == 1
    assert summary["pid_rows"] == 2
    assert summary["state"] == "ALIGNED_CONDITIONAL_NOT_QUALIFIED"
    with (run / "aligned/force_aligned.csv").open() as stream:
        first = next(csv.DictReader(stream))
    assert first["receive_monotonic_ns"] == "12000000000"
    assert (
        int(first["aircraft_receive_lower_ns"])
        <= 62_000_000_000
        <= int(first["aircraft_receive_upper_ns"])
    )


def test_full_join_with_smooth_interval_clocks(run):
    summary = align_run(
        run,
        run / "aligned-smooth",
        rpm_model="smooth_interval",
        pi_model="smooth_interval",
    )
    assert summary["rpm_model"] == "smooth_interval"
    assert summary["pi_model"]["method"] == "smooth_interval"
    assert summary["rpm_model_validation"]["selected_fallback"]["optimizer_iterations"] >= 1


def test_fftgyro_encoder_is_mapped_from_windows_to_aircraft_clock(run):
    manifest = json.loads((run / "run.json").read_text())
    manifest["encoder_required"] = True
    (run / "run.json").write_text(json.dumps(manifest))
    for folder in ("encoder", "encoder-clock"):
        (run / folder).mkdir()
    (run / "encoder/metadata.json").write_text(
        json.dumps({"state": "COMPLETE", "encoder_frames": 8})
    )
    (run / "encoder-clock/summary.json").write_text(
        json.dumps({"state": "COMPLETE", "server_boot_id": "a"})
    )
    with (run / "encoder-clock/exchanges.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["client_send_ns", "server_receive_ns", "server_send_ns", "client_receive_ns"]
        )
        for stamp in range(1_000_000_000, 5_000_000_000, 1_000_000_000):
            writer.writerow(
                [stamp, stamp + 60_000_000_000, stamp + 60_000_100_000, stamp + 1_000_000]
            )
    with (run / "encoder/samples.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["receive_monotonic_ns", "experimental_pitch_deg"])
        for index in range(8):
            writer.writerow([2_100_000_000 + index * 100_000_000, index])
    summary = align_run(run, run / "aligned-encoder")
    assert summary["encoder_samples"] == 8
    assert summary["encoder_pid_overlap"] == 8
    with (run / "aligned-encoder/encoder_aligned.csv").open() as stream:
        first = next(csv.DictReader(stream))
    assert first["experimental_pitch_deg"] == "0"
    assert (
        int(first["aircraft_receive_lower_ns"])
        <= 62_100_000_000
        <= int(first["aircraft_receive_upper_ns"])
    )


@pytest.mark.parametrize(
    "key,value",
    [("state", "INCOMPLETE"), ("aircraft_boot_id_after", "reboot"), ("leptrino_boot_id", "wrong")],
)
def test_bad_run_rejected(run, key, value):
    path = run / "run.json"
    data = json.loads(path.read_text())
    data[key] = value
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        align_run(run, run / "aligned")
    assert not (run / "aligned").exists()


def test_force_outside_coverage_rejected(run):
    path = run / "leptrino/samples.csv"
    path.write_text(path.read_text().replace("12000000000", "1000000000"))
    with pytest.raises(ValueError, match="coverage"):
        align_run(run, run / "aligned")


def test_does_not_overwrite(run):
    align_run(run, run / "aligned")
    with pytest.raises(FileExistsError):
        align_run(run, run / "aligned")
