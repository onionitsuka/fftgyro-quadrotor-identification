import csv
import json
import struct
from types import SimpleNamespace

import pytest

from fftgyro_quad_id.experiment_runner import immediate_quality as qc
from fftgyro_quad_id.io.leptrino import frame
from fftgyro_quad_id.io.leptrino_raw import RawWriter


def json_file(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def csv_file(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def force_run(root, status=0x40, raw=1):
    json_file(
        root / "leptrino/summary.json",
        dict(
            state="RAW_COMPLETE",
            rated_limits_N_Nm=[100, 100, 100, 2, 2, 2],
            start_command_before_ns=0,
            elapsed_capture_s=1,
        ),
    )
    writer = RawWriter(root / "leptrino/chunks.bin")
    writer.push(0, frame(bytes([4, 255, 50, 0])))
    for i in range(101):
        payload = bytes([20, 255, 50, 0]) + struct.pack("<6h", raw, 2, 3, 4, 5, 6)
        writer.push(i * 10_000_000, frame(payload + bytes([0, 0, status, i % 256])))
    writer.push(1_010_000_000, frame(bytes([4, 255, 51, 0])))
    writer.finish()


@pytest.mark.parametrize(
    "status,raw,decision",
    [(0x40, 1, "PASS"), (0x44, 1, "RETRY"), (0x41, 1, "RETRY"), (0x40, 10001, "RETRY")],
)
def test_raw_force_protocol_faults_and_preservation(tmp_path, status, raw, decision):
    force_run(tmp_path, status, raw)
    before = (tmp_path / "leptrino/chunks.bin").read_bytes()
    result = qc.inspect(tmp_path, {}, ["force"])[0]
    assert result["decision"] == decision
    assert (tmp_path / "leptrino/chunks.bin").read_bytes() == before
    assert not (tmp_path / "leptrino/samples.csv").exists()
    assert "candidate_counter_jumps" not in result["metrics"]


def test_truncated_force_is_retry(tmp_path):
    force_run(tmp_path)
    path = tmp_path / "leptrino/chunks.bin"
    path.write_bytes(path.read_bytes()[:-5])
    assert qc.inspect(tmp_path, {}, ["force"])[0]["decision"] == "RETRY"


def test_missing_evidence_is_never_pass(tmp_path):
    report = qc.summarize(qc.inspect(tmp_path, {}, ["force", "encoder"]))
    assert report["decision"] == "REVIEW"
    assert not report["physics_validated"]
    assert not report["automatic_motor_retry"]
    assert qc.summarize([])["decision"] == "REVIEW"


@pytest.mark.parametrize(
    "policy",
    [
        {"max_clock_bound_ms": float("nan")},
        {"max_receive_gap_ms": -1},
        {"ignored_unknown_key": 2},
        {"max_saturation_fraction": 2},
        {"static_windows_s": [[0, 5]], "max_static_rate_deg_s": 1},
        {"static_windows_s": [[0, 10]]},
    ],
)
def test_invalid_policy_rejected_before_hardware(policy):
    with pytest.raises(ValueError):
        qc.validate_policy(policy)


def encoder_run(root, rate=0):
    json_file(root / "encoder/metadata.json", {"state": "COMPLETE"})
    csv_file(
        root / "encoder/samples.csv",
        [
            dict(
                receive_monotonic_ns=i * 10_000_000,
                experimental_pitch_deg=rate * i / 100,
                experimental_yaw_deg=(179 + i / 100 * rate + 180) % 360 - 180,
            )
            for i in range(1201)
        ],
    )


@pytest.mark.parametrize("rate,decision", [(0, "PASS"), (0.2, "PASS"), (3, "RETRY")])
def test_static_windows_detect_motion_and_unwrap_yaw(tmp_path, rate, decision):
    encoder_run(tmp_path, rate)
    profile = dict(static_windows_s=[[0, 10]], max_static_rate_deg_s=1)
    checks = qc.inspect(tmp_path, {}, ["encoder", "stationary"], profile)
    assert checks[0]["decision"] == "PASS"
    assert checks[1]["decision"] == decision


def test_holding_without_declared_window_is_not_certified(tmp_path):
    encoder_run(tmp_path)
    assert qc.inspect(tmp_path, {}, ["stationary"])[0]["decision"] == "REVIEW"


def pid_run(root, duration=1, saturated=False):
    json_file(
        root / "pid/session.config.json",
        dict(
            segments=[dict(segment_id="s1", duration_s=duration)],
        ),
    )
    (root / "pid/result.txt").write_text("status=COMPLETE\n", encoding="utf-8")
    csv_file(
        root / "pid/control.csv",
        [
            dict(
                sent_ns=1_000_000_000 + i * 10_000_000,
                stage="SEGMENT",
                segment_id="s1",
                segment_time_s=i / 100,
                saturated=int(saturated),
                authority_limited=0,
                angle_deg=0,
                target_deg=0,
                **{f"sent{j}": 0.4 for j in range(1, 5)},
            )
            for i in range(101)
        ],
    )


def test_pid_segments_are_checked_even_with_complete_result(tmp_path):
    pid_run(tmp_path, duration=2)
    assert qc.inspect(tmp_path, {}, ["pid"])[0]["decision"] == "RETRY"


def test_pid_saturation_needs_review_not_automatic_motor_retry(tmp_path):
    pid_run(tmp_path, saturated=True)
    assert qc.inspect(tmp_path, {}, ["pid"])[0]["decision"] == "REVIEW"


def test_explicit_tracking_limit(tmp_path):
    pid_run(tmp_path)
    path = tmp_path / "pid/control.csv"
    rows = qc.table(path)
    rows[0]["angle_deg"] = "10"
    csv_file(path, rows)
    assert (
        qc.inspect(tmp_path, {}, ["pid"], {"max_tracking_error_deg": 2})[0]["decision"] == "RETRY"
    )


def test_rpm_missing_channel_and_idle_exception(tmp_path, monkeypatch):
    pid_run(tmp_path)
    events = [
        SimpleNamespace(kind="E", value=j, host_ns=1_000_000_000 + i * 10_000_000)
        for i in range(101)
        for j in (1, 2, 3)
    ]
    monkeypatch.setattr(qc, "read_capture", lambda *a, **k: (events, [1, 2, 3]))
    monkeypatch.setattr(qc, "fit_clock", lambda *a: None)
    assert qc.inspect(tmp_path, {"idle": False}, ["rpm"])[0]["decision"] == "RETRY"
    assert qc.inspect(tmp_path, {"idle": True}, ["rpm"])[0]["decision"] == "PASS"


def test_clock_boot_mismatch(tmp_path):
    json_file(tmp_path / "clock/summary.json", dict(state="COMPLETE", server_boot_id="old"))
    csv_file(
        tmp_path / "clock/exchanges.csv",
        [dict(offset_low_ns=0, offset_high_ns=100) for _ in range(3)],
    )
    assert qc.inspect(tmp_path, {"aircraft_boot_id": "new"}, ["clock"])[0]["decision"] == "RETRY"


def test_sd_verification_does_not_claim_ulog_content(tmp_path):
    json_file(
        tmp_path / "flightlog/stop.json",
        dict(
            sd_log_verified=True,
            logger_stopped=True,
            dropouts_reported=[0],
            sd_file_bytes=1000,
        ),
    )
    result = qc.inspect(tmp_path, {}, ["flightlog"])[0]
    assert result["decision"] == "PASS"
    assert "本文は未検証" in result["reason"]
