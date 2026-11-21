"""Synthetic acquisition/coverage tests, not validation against flight data."""

import csv
import json
import sys

import numpy as np
import pytest

from fftgyro_quad_id.flight_coverage import DEFAULTS, audit_flight, intervals, main, summarize


def synthetic():
    t = np.arange(0, 20, 0.02)
    zero = np.zeros(len(t))
    data = dict(
        time_s=t,
        airborne=np.ones(len(t)),
        mocap_valid=np.ones(len(t)),
        usable=np.ones(len(t)),
        pitch_deg=zero.copy(),
        roll_deg=zero.copy(),
        yaw_deg=zero.copy(),
        p_deg_s=zero.copy(),
        q_deg_s=zero.copy(),
        r_deg_s=zero.copy(),
        speed_m_s=zero.copy(),
        vz_m_s=zero.copy(),
        u_air_m_s=zero.copy(),
        voltage_v=np.full(len(t), 15.5),
    )
    for i in range(1, 5):
        data[f"rpm{i}"] = np.full(len(t), 4000.0)
        data[f"command{i}"] = np.full(len(t), 0.4)
    meta = dict(
        flight_id="a",
        recording_key="fc-boot-flight-a",
        frames_verified=True,
        sync_verified=True,
        sync_uncertainty_ms=2.0,
        rpm_calibration_verified=True,
        command_mapping_verified=True,
    )
    return data, meta


def test_hover_is_not_yaw_or_pitch_excitation():
    d, m = synthetic()
    r = audit_flight(d, m, DEFAULTS)
    assert r["motions"]["hover"]["status"] == "OBSERVED"
    assert r["motions"]["yaw"]["status"] == "NOT_OBSERVED"
    assert r["motions"]["pitch"]["status"] == "NOT_OBSERVED"


def test_large_roll_is_not_level_hover():
    d, m = synthetic()
    d["roll_deg"][:] = 30
    assert audit_flight(d, m, DEFAULTS)["motions"]["hover"]["status"] == "NOT_OBSERVED"


def test_two_repetitions_not_one_long_flight():
    d, m = synthetic()
    first = audit_flight(d, m, DEFAULTS)
    decision = summarize([first], DEFAULTS)["motion_decisions"]["hover"]
    assert decision["status"] == "ADDITIONAL_MOTION_CANDIDATE"
    second = audit_flight(d, m | dict(flight_id="b", recording_key="fc-boot-flight-b"), DEFAULTS)
    decision = summarize([first, second], DEFAULTS)["motion_decisions"]["hover"]
    assert decision["status"] == "REPEATED_MOTION_AVAILABLE"


def test_copied_flight_is_not_an_independent_repeat():
    d, m = synthetic()
    first = audit_flight(d, m, DEFAULTS)
    with pytest.raises(ValueError, match="Duplicate"):
        summarize([first, first], DEFAULTS)


def test_pitch_and_yaw_excitation_detected():
    d, m = synthetic()
    w = 2 * np.pi * 0.3
    d["pitch_deg"] = 10 * np.sin(w * d["time_s"])
    d["q_deg_s"] = 10 * w * np.cos(w * d["time_s"])
    d["yaw_deg"] = (179 + 30 * np.sin(w * d["time_s"]) + 180) % 360 - 180
    d["r_deg_s"] = 30 * w * np.cos(w * d["time_s"])
    r = audit_flight(d, m, DEFAULTS)
    assert r["motions"]["pitch"]["status"] == "OBSERVED"
    assert r["motions"]["yaw"]["status"] == "OBSERVED"
    assert r["motions"]["yaw"]["angle_span_deg"] < 61


def test_no_false_yaw_span_across_a_gap():
    d, m = synthetic()
    d["time_s"][500:] += 1
    d["yaw_deg"][500:] = 100
    d["r_deg_s"] = 20 * np.sin(d["time_s"])
    assert audit_flight(d, m, DEFAULTS)["motions"]["yaw"]["status"] == "NOT_OBSERVED"
    assert intervals(np.array([0.0, 0.02, 1.0]), np.ones(3, bool), 0.1).sum() == 0.02


def test_missing_rpm_is_not_replaced_by_commands():
    d, m = synthetic()
    del d["rpm3"]
    routes = audit_flight(d, m, DEFAULTS)["route_valid_duration_s"]
    assert routes["measured_rpm"] is None
    assert routes["common_physical_vs_input_model"] is None
    assert routes["input_voltage"] > 19


@pytest.mark.parametrize(
    "change",
    [{"sync_verified": False}, {"sync_uncertainty_ms": 100}, {"rpm_calibration_verified": False}],
)
def test_calibration_and_sync_are_data_work_not_new_flight(change):
    d, m = synthetic()
    r = audit_flight(d, m | change, DEFAULTS)
    assert r["motions"]["hover"]["status"] == "OBSERVED"
    assert r["route_valid_duration_s"]["measured_rpm"] is None


def test_invalid_tracking_does_not_imply_motion_was_absent():
    d, m = synthetic()
    d["mocap_valid"][:] = 0
    r = audit_flight(d, m, DEFAULTS)
    assert summarize([r], DEFAULTS)["motion_decisions"]["hover"]["status"] == "RESOLVE_DATA_FIRST"


def test_nonmonotonic_time_rejected():
    d, m = synthetic()
    d["time_s"][4] = d["time_s"][3]
    with pytest.raises(ValueError, match="strictly increasing"):
        audit_flight(d, m, DEFAULTS)


def test_common_comparison_uses_intersection():
    d, m = synthetic()
    d["rpm1"][:300] = np.nan
    d["voltage_v"][700:] = np.nan
    routes = audit_flight(d, m, DEFAULTS)["route_valid_duration_s"]
    assert routes["common_physical_vs_input_model"] < routes["measured_rpm"]
    assert routes["common_physical_vs_input_model"] < routes["input_voltage"]


def test_mixed_up_frames_blocks_motion_decisions():
    d, m = synthetic()
    r = audit_flight(d, m | {"frames_verified": False}, DEFAULTS)
    assert summarize([r], DEFAULTS)["motion_decisions"]["hover"]["status"] == "RESOLVE_DATA_FIRST"


def test_cli_report_and_no_overwrite(tmp_path, monkeypatch):
    d, m = synthetic()
    csv_path = tmp_path / "flight.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(d)
        writer.writerows(zip(*d.values(), strict=True))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"flights": [m | {"csv": csv_path.name}]}))
    output = tmp_path / "report.json"
    monkeypatch.setattr(sys, "argv", ["audit", str(manifest), "--output", str(output)])
    main()
    report = json.loads(output.read_text())
    assert report["route_decisions"]["common_physical_vs_input_model"]["status"] == "DATA_AVAILABLE"
    assert report["model_validation"] == "NOT_RUN"
    with pytest.raises(FileExistsError):
        main()


def test_cli_rejects_empty_flights(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"flights": []}))
    monkeypatch.setattr(sys, "argv", ["audit", str(manifest), "--output", str(tmp_path / "report")])
    with pytest.raises(ValueError, match="No flights"):
        main()
