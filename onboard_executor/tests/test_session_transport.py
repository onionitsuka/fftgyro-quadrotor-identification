"""Run the production session executable against a synthetic PX4 on a PTY.

Linux, Python standard library only. Never accepts a hardware port. Qualification
flags apply ONLY to temporary synthetic configurations, not physical apparatus.
Usage: python3 test_session_transport.py EXECUTABLE axis-session.example.json
"""

import csv
import errno
import json
import math
import os
from pathlib import Path
import pty
import select
import signal
import shutil
import struct
import subprocess
import sys
import tempfile
import time


def crc(data: bytes) -> int:
    value = 0xFFFF
    for byte in data:
        tmp = byte ^ (value & 0xFF)
        tmp ^= (tmp << 4) & 0xFF
        value = (value >> 8) ^ (tmp << 8) ^ (tmp << 3) ^ (tmp >> 4)
    return value


def packet(msgid: int, payload: bytes, extra: int) -> bytes:
    header = bytes([len(payload), 0, 0, 0, 1, 1]) + msgid.to_bytes(3, "little")
    return b"\xfd" + header + payload + struct.pack("<H", crc(header + payload + bytes([extra])))


def frames(buffer: bytearray):
    while buffer:
        if buffer[0] != 0xFD:
            del buffer[0]
            continue
        if len(buffer) < 10:
            return
        length = 12 + buffer[1] + (13 if buffer[2] & 1 else 0)
        if len(buffer) < length:
            return
        frame = bytes(buffer[:length])
        del buffer[:length]
        msgid = int.from_bytes(frame[7:10], "little")
        if msgid == 60000:
            assert crc(frame[1:10 + frame[1]] + bytes([50])) == struct.unpack_from("<H", frame, 10 + frame[1])[0]
            yield frame[5], frame[6], frame[10:10 + frame[1]].ljust(29, b"\0")


def check(executable: Path, template: dict, mode: str, props_off: bool = False,
          preserve_segments: bool = False, fixed_commissioning: bool = False,
          pitch_commissioning: bool = False) -> None:
    with tempfile.TemporaryDirectory(prefix="fftgyro-session-pty-") as temp:
        root = Path(temp)
        master, slave = pty.openpty()
        proc = None
        try:
            config = json.loads(json.dumps(template))
            cascade = config["pid"].get("law") == "ANGLE_RATE_PID"
            capture_reference = config["startup"].get("capture_reference", False)
            excitation_start = 500 if cascade else 100
            commissioning = props_off or fixed_commissioning or pitch_commissioning
            config.update(session_id="synthetic-only", runtime_qualified=not commissioning)
            if not preserve_segments:
                config["maximum_duration_s"] = 12 if commissioning else 20
            hardware = config["hardware"]
            hardware.update(uart_device=os.ttyname(slave), log_directory=str(root / "logs"))
            assert hardware["uart_device"].startswith("/dev/pts/")
            for key in hardware:
                if key.endswith("_verified"):
                    hardware[key] = not commissioning
            config["startup"]["control_ready_verified"] = not commissioning
            if pitch_commissioning:
                hardware["motor_mapping_verified"] = True
                hardware["feedback_frame_verified"] = True
                if mode == "nonidentity_reference":
                    # Nontrivial reference: never auto-zero or assume identity.
                    config["pid"]["zero_quaternion_wxyz"] = [math.cos(.4), 0, 0, math.sin(.4)]
            if not preserve_segments:
                segment = config["segments"][0]
                segment.update(duration_s=5, collective=0.1, start_deg=0)
                config["segments"] = [segment]
            path = root / "session.json"
            if mode in ("absolute_anchor_negative", "absolute_anchor_positive"):
                anchor = -26.9 if mode.endswith("negative") else 26.9
                config["startup"]["encoder_absolute_anchor"] = True
                config["pid"]["zero_angle_deg"] = anchor
                half = math.radians(-.5) / 2
                config["pid"]["zero_quaternion_wxyz"] = (
                    [math.cos(half), 0, math.sin(half), 0] if config["pid"]["axis"] == "PITCH"
                    else [math.cos(half), 0, 0, math.sin(half)])
                config["pid"]["angle_min_deg"] = -90
                config["pid"]["angle_max_deg"] = 90
                # Long-ramp Yaw templates need enough validation budget even
                # though the synthetic feedback settles immediately.
                config["maximum_duration_s"] = 80
                config["segments"] = [dict(config["segments"][0], start_deg=anchor+.5, end_deg=anchor+.5)]
            path.write_text(json.dumps(config))
            with (root / "stdout").open("w+") as stdout, (root / "stderr").open("w+") as stderr:
                command = [str(executable), "session-props-off" if props_off else "session-run", str(path)]
                if props_off:
                    command += ["--props-removed", "--fixed", "--operator-stop-ready", "--charged-4s"]
                if fixed_commissioning:
                    command = [str(executable), "session-fixed-commissioning", str(path),
                               "--props-installed", "--fixed", "--operator-stop-ready", "--charged-4s"]
                if pitch_commissioning:
                    command = [str(executable), "session-pitch-hold-commissioning", str(path),
                               "--props-installed", "--pitch-only-free", "--operator-stop-ready", "--charged-4s"]
                proc = subprocess.Popen(command, stdout=stdout, stderr=stderr)
                start = time.monotonic()
                next_attitude = start + 0.05
                buffer = bytearray()
                control_count = 0
                release_seen = False
                first_control = None
                release_time = None
                replay = None
                stop_sent = False
                last_motor_peak = 0.0
                shutdown_drift_started = False
                while proc.poll() is None and time.monotonic() - start < config["maximum_duration_s"] + 5:
                    now = time.monotonic()
                    if now >= next_attitude:
                        if mode != "telemetry_loss" or control_count == 0:
                            angle, rate = 0.0, 0.0
                            if mode == "bad_reference": angle = .75
                            if mode == "micro_vibration_preflight": rate = 3
                            if mode == "moving_preflight": rate = 8
                            if mode == "shutdown_drift" and shutdown_drift_started:
                                # Synthetic wind/rig restoration after output
                                # starts fading: valid angle and rate, but no
                                # longer holdable with vanishing motor thrust.
                                angle, rate = 30, 5
                            if control_count > excitation_start:
                                if mode == "pd_response" and control_count < excitation_start + (50 if cascade else 100):
                                    angle, rate = .2, .5
                                if mode == "relaxed_settling": angle, rate = 3, 3
                                if mode == "angle_limit": angle = config["pid"]["angle_max_deg"] + 1
                                if mode == "rate_limit": rate = config["pid"]["rate_limit_deg_s"] + 1
                                if mode == "unsettled":
                                    if cascade: rate = config["settling"]["rate_deg_s"] + 1
                                    else: angle = config["settling"]["error_deg"] + 1
                            half = math.radians(angle) / 2
                            q = ((math.cos(half), 0, math.sin(half), 0)
                                 if config["pid"]["axis"] == "PITCH"
                                 else (math.cos(half), 0, 0, math.sin(half)))
                            if control_count > excitation_start and mode in ("fixed_offset", "swing_limit"):
                                # Roll then Pitch; fixed_offset reproduces >3 deg in both.
                                roll = math.radians(4 if mode == "fixed_offset" else config["pid"]["swing_limit_deg"] + 1) / 2
                                pitch = math.radians(4 if mode == "fixed_offset" else 0) / 2
                                q = (math.cos(roll)*math.cos(pitch), math.sin(roll)*math.cos(pitch),
                                     math.cos(roll)*math.sin(pitch), math.sin(roll)*math.sin(pitch))
                            if mode == "nonidentity_reference":
                                q = tuple(config["pid"]["zero_quaternion_wxyz"])
                            boot = 1000 + int((now - start) * 1000)
                            if mode == "px4_reboot" and control_count > 100: boot = 1
                            rates = ((0, math.radians(rate), 0) if config["pid"]["axis"] == "PITCH"
                                     else (0, 0, math.radians(rate)))
                            payload = struct.pack("<I7f", boot, *q, *rates)
                            os.write(master, packet(31, payload, 246))
                        if mode == "replayed_ack" and replay is not None and not release_seen:
                            os.write(master, replay)
                        next_attitude = now + 0.01
                    if mode == "operator_stop" and control_count > 100 and not stop_sent:
                        proc.send_signal(signal.SIGTERM)
                        stop_sent = True
                    if not select.select([master], [], [], 0.002)[0]:
                        continue
                    try:
                        buffer.extend(os.read(master, 8192))
                    except OSError as error:
                        if error.errno == errno.EIO:
                            break
                        raise
                    for system, component, payload in frames(buffer):
                        run, sequence, *values = struct.unpack("<II4fHBBB", payload)
                        motors, timeout, target_system, target_component, action = values[:4], *values[4:]
                        assert (target_system, target_component) == (1, 1)
                        assert timeout == 150 and action in (0, 1)
                        assert all(math.isfinite(v) and 0 <= v <= hardware["motor_max"] for v in motors)
                        assert not release_seen, "command sent after release"
                        if action == 1:
                            release_seen = True
                            release_time = time.monotonic()
                            status = 1
                        else:
                            control_count += 1
                            first_control = first_control or time.monotonic()
                            status = 4 if mode == "rejected" else 0
                            peak = max(motors)
                            if (mode == "shutdown_drift" and control_count > 200 and
                                    last_motor_peak > .05 and peak < last_motor_peak - 1e-5):
                                shutdown_drift_started = True
                            last_motor_peak = peak
                        ack = packet(60001, struct.pack("<QIIBBB", int((now - start) * 1e6), run, sequence,
                                                       system, component, status), 221)
                        if action == 0 and mode == "replayed_ack":
                            if replay is None:
                                replay = ack
                            else:
                                continue
                        if (action == 0 and mode == "ack_loss") or (action == 1 and mode == "release_loss"):
                            continue
                        os.write(master, ack)
                proc.wait(timeout=2)
                stdout.seek(0)
                stderr.seek(0)
                diagnostic = stdout.read() + stderr.read()
            directory = root / "logs" / "synthetic-only"
            assert (directory / "result.txt").exists(), diagnostic
            result = (directory / "result.txt").read_text()
            if props_off:
                assert "runtime_qualified=0" in result and "props_off_commissioning=1" in result
                assert "commissioning_zero_quaternion=" in result
            if fixed_commissioning:
                assert "runtime_qualified=0" in result and "fixed_commissioning=1" in result
                assert "fixed_completion_requires_settling=0" in result
                assert "props_off_commissioning=0" in result and "commissioning_zero_quaternion=" in result
            preflight_failure = mode == "moving_preflight" or (mode == "bad_reference" and not capture_reference)
            assert (control_count == 0 if preflight_failure else control_count > 0) and release_seen, diagnostic + result
            success = mode in ("normal", "pd_response", "nonidentity_reference", "fixed_offset",
                               "absolute_anchor_negative", "absolute_anchor_positive",
                               "micro_vibration_preflight",
                               "relaxed_settling", "shutdown_drift")
            success = success or (mode == "bad_reference" and capture_reference)
            expected = {"normal": "status=COMPLETE", "ack_loss": "ACK timeout",
                        "absolute_anchor_negative": "status=COMPLETE",
                        "absolute_anchor_positive": "status=COMPLETE",
                        "replayed_ack": "ACK timeout", "telemetry_loss": "feedback missing/stale",
                        "rejected": "PX4 rejected motor request", "release_loss": "release not acknowledged",
                        "pd_response": "status=COMPLETE", "nonidentity_reference": "status=COMPLETE",
                        "bad_reference": "PITCH preflight not stationary",
                        "moving_preflight": "PITCH preflight not stationary",
                        "angle_limit": "joint angle limit", "rate_limit": "body rate limit",
                        "fixed_offset": "status=COMPLETE", "micro_vibration_preflight": "status=COMPLETE",
                        "swing_limit": "non-target attitude limit",
                        "relaxed_settling": "status=COMPLETE", "shutdown_drift": "status=COMPLETE",
                        "unsettled": "hold did not settle", "operator_stop": "operator stop",
                        "px4_reboot": "feedback fault during Pitch hold"}[mode]
            if mode == "bad_reference" and capture_reference:
                expected = "status=COMPLETE"
            assert expected in result, diagnostic + result
            assert proc.returncode == (0 if success else 1), diagnostic + result
            if mode == "shutdown_drift":
                assert shutdown_drift_started, "synthetic low-thrust attitude drift was not injected"
            if not success:
                assert "status=ABORTED" in result, result
            if mode == "rate_limit":
                fields = dict(line.split("=", 1) for line in result.splitlines() if "=" in line)
                rates = [float(v) for v in fields["failure_body_rate_deg_s"].strip(",").split(",")]
                assert rates[1] > config["pid"]["rate_limit_deg_s"], fields
                assert int(fields["failure_detected_ns"]) >= int(fields["failure_feedback_received_ns"])
            if mode in ("ack_loss", "replayed_ack", "telemetry_loss", "rejected"):
                assert release_time - first_control < 0.5, "fault-to-release exceeded test bound"
            with (directory / "control.csv").open() as stream:
                controls = list(csv.DictReader(stream))
            with (directory / "ack.csv").open() as stream:
                acks = list(csv.DictReader(stream))
            model = config["pid"].get("cascade", {}).get("collective_gain_compensation", {})
            if model.get("enabled"):
                assert None not in controls[0], "CSV header and rows have different lengths"
                for row in controls:
                    if row["control_mode"] == "STOP_MOTOR_FADE":
                        assert row["stage"] == "RAMP_DOWN"
                        assert all(float(row[k]) == 0 for k in ("p_term", "i_term", "d_term"))
                        assert row["tracking_guard_enabled"] == "0" and row["integral_enabled"] == "0"
                        continue
                    collective = float(row["collective"])
                    slope_ref = model["thrust_curve_linear"] + 2 * model["thrust_curve_quadratic"] * model["reference_collective"]
                    slope = model["thrust_curve_linear"] + 2 * model["thrust_curve_quadratic"] * max(collective, model["reference_collective"])
                    scale = float(row["actuator_gain_scale"])
                    assert abs(scale - slope_ref / slope) < 1e-9
                    demand = scale * sum(float(row[k]) for k in ("p_term", "i_term", "d_term")) + float(row["external_delta"])
                    assert abs(float(row["delta_requested"]) - demand) < 1e-9
                    assert bool(int(row["actuator_gain_floor_active"])) == (collective < model["reference_collective"])
                    if row["stage"] in ("STARTUP_RAMP", "COLLECTIVE_CHANGE", "RAMP_DOWN"):
                        assert float(row["trajectory_rate_deg_s"]) == 0, "moving FF leaked into stationary reference"
                if success:
                    assert any(float(r["actuator_gain_scale"]) < .9 for r in controls)
                    assert any(abs(float(r["external_delta"])) > .001 for r in controls)
                    transitions = [r for r in controls if r["stage"] == "TRANSITION"]
                    assert transitions and any(abs(float(r["trajectory_rate_deg_s"])) > .1 for r in transitions)
                    assert any(r["actuator_model_in_range"] == "1" for r in controls)
                    assert controls[-1]["actuator_model_in_range"] == "0"
            if success and any(s.get("collective_step_up") for s in config["segments"]):
                jumps = []
                for previous, current in zip(controls, controls[1:]):
                    before, after = float(previous["sent1"]), float(current["sent1"])
                    if current["stage"] == "COLLECTIVE_STEP_UP":
                        assert after > before
                        jumps.append(after-before)
                    elif current["stage"] == "COLLECTIVE_CHANGE":
                        assert before-after <= hardware["normal_ramp_rate"]*.01+1e-6
                    assert all(abs(float(current[f"sent{i}"])-after)<1e-6 for i in range(1,5))
                assert jumps and max(jumps) > .09, "upward transition remained a ramp"
            assert len(controls) == control_count and len(acks) == control_count + 1
            assert acks[-1]["release"] == "1"
            assert int(acks[-1]["ack_status"]) == (-1 if mode == "release_loss" else 1)
            if success or mode == "operator_stop":
                assert "RAMP_DOWN" in {row["stage"] for row in controls}
                assert all(abs(float(controls[-1][f"sent{i}"])) < 1e-6 for i in range(1, 5))
                shutdown = [r for r in controls if r["control_mode"] == "STOP_MOTOR_FADE"]
                if mode == "shutdown_drift":
                    assert any(float(r["angle_deg"]) > 20 for r in shutdown)
                if config.get("control_mode") == "AXIS_PID" and not pitch_commissioning:
                    assert shutdown, "loaded axis shutdown still tried to hold angle to zero thrust"
                    for previous, current in zip(shutdown, shutdown[1:]):
                        for i in range(1, 5):
                            before = float(previous[f"sent{i}"])
                            after = float(current[f"sent{i}"])
                            assert 0 <= after <= before + 1e-6
                            assert before - after <= hardware["normal_ramp_rate"] * .01 + 1e-5
                    assert all(float(shutdown[-1][f"sent{i}"]) == 0 for i in range(1, 5))
                if preserve_segments and success:
                    assert {r["segment_id"] for r in controls if r["stage"] == "SEGMENT"} == {
                        s["segment_id"] for s in config["segments"]}
                    for segment in config["segments"]:
                        if segment.get("require_settled"):
                            rows = [r for r in controls if r["segment_id"] == segment["segment_id"]
                                    and r["stage"] in ("SEGMENT", "SETTLING_EXTENSION")]
                            assert rows[-1]["settled"] == ("0" if mode == "fixed_offset" else "1")
                    if any(s["kind"] != "HOLD" for s in config["segments"]):
                        assert max(abs(float(r["target_deg"])) for r in controls) > 0.4
                        assert any(abs(float(r["p_term"])) > 0 for r in controls)
            if config.get("control_mode") == "FIXED_HOLD":
                if success and hardware["startup_ramp_rate"] == .02:
                    startup = [r for r in controls if r["stage"] == "STARTUP_RAMP"]
                    assert len(startup) >= 499
                    values = [0.] + [float(r["sent1"]) for r in startup]
                    assert max(b-a for a,b in zip(values, values[1:])) <= .00020001
                    assert max(values) >= .0999
                assert all(r["control_mode"] == "FIXED_HOLD" and r["integral_enabled"] == "0"
                           for r in controls)
                assert all(float(r[key]) == 0 for r in controls
                           for key in ("p_term", "i_term", "d_term", "delta"))
                assert all(len({r[f"sent{i}"] for i in range(1, 5)}) == 1 for r in controls)
            if pitch_commissioning:
                assert "pitch_hold_commissioning=1" in result and "runtime_qualified=0" in result
                assert "commissioning_zero_quaternion=" not in result
                threshold = config["startup"]["integral_enable_collective"]
                assert all((r["integral_enabled"] == "1") == (float(r["collective"]) >= threshold)
                           for r in controls)
                assert all(abs(float(r["i_term"])) <= config["pid"]["integral_limit"] + 1e-9
                           for r in controls)
                low_startup = [r for r in controls if r["stage"] == "STARTUP_RAMP"
                               and float(r["collective"]) < threshold]
                assert all(float(r["i_term"]) == 0 for r in low_startup)
                if mode in ("pd_response", "relaxed_settling"):
                    assert any(float(r["i_term"]) < -1e-6 for r in controls), "Pitch I never accumulated"
                    if mode == "relaxed_settling" and config["pid"]["differential_limit"] > .005:
                        assert any(abs(float(r["delta"])) > .0051 for r in controls), "old Pitch differential cap still active"
                    shutdown = [r for r in controls if r["stage"] == "RAMP_DOWN"
                                and float(r["collective"]) < threshold]
                    assert shutdown and abs(float(shutdown[0]["i_term"])) > 1e-6
                    assert all(abs(float(r["i_term"]) - float(shutdown[0]["i_term"])) < 1e-9
                               for r in shutdown), "low-collective freeze reset or accumulated I"
                assert all(abs(float(r["delta"])) <= config["pid"]["differential_limit"] + 1e-9
                           for r in controls)
                assert all(0 <= float(r[f"sent{i}"]) <= hardware["motor_max"] + 1e-7
                           for r in controls for i in range(1, 5))
                saved = json.loads((directory / "session.config.json").read_text())
                assert saved["pid"]["zero_quaternion_wxyz"] == config["pid"]["zero_quaternion_wxyz"]
                if mode == "pd_response":
                    changed = [r for r in controls if float(r["angle_deg"]) > .1]
                    assert changed and all(float(r["p_term"]) < 0 for r in changed)
                    if not cascade:
                        assert all(float(r["d_term"]) < 0 for r in changed)
                    elif config["pid"]["cascade"]["kd"] == 0:
                        assert all(float(r["d_term"]) == 0 for r in controls)
                    assert all(float(r["sent1"]) < float(r["sent2"]) for r in changed)
                    # Check all four motors against physical front/rear positions,
                    # not just motor 1 vs 2 (which also passes for the old Roll mix).
                    for row in changed:
                        m1, m2, m3, m4 = (float(row[f"sent{i}"]) for i in range(1, 5))
                        assert abs(m1 - m3) < 1e-6 and abs(m2 - m4) < 1e-6
                        assert m1 + m3 - m2 - m4 < 0  # Nose-down correction.
                        assert abs(m2 + m3 - m1 - m4) < 1e-6  # No Roll torque.
            if capture_reference and not preflight_failure:
                assert result.count("session_reference_quaternion=") == 1, "reference recaptured mid-session"
                fields = dict(line.split("=", 1) for line in result.splitlines() if "=" in line)
                reference = [float(v) for v in fields["session_reference_quaternion"].strip(",").split(",")]
                if mode in ("absolute_anchor_negative", "absolute_anchor_positive"):
                    expected_angle = config["pid"]["zero_angle_deg"] + .5
                    assert abs(float(fields["reference_zero_angle_deg"])-expected_angle) < 1e-5, fields
                    assert all(abs(float(r["angle_deg"])-expected_angle) < 1e-5
                               for r in controls if r["control_mode"] != "STOP_MOTOR_FADE")
                    assert all(abs(float(r["target_deg"])-expected_angle) < 1e-5
                               for r in controls if r["stage"] == "STARTUP_RAMP")
                expected_reference = config["pid"]["zero_quaternion_wxyz"] if mode == "nonidentity_reference" else [1, 0, 0, 0]
                if mode == "bad_reference":
                    expected_reference = [math.cos(math.radians(.75) / 2), 0, math.sin(math.radians(.75) / 2), 0]
                assert all(abs(a - b) < 1e-6 for a, b in zip(reference, expected_reference)), fields
                dwell = (int(fields["reference_capture_end_ns"]) - int(fields["reference_capture_start_ns"])) / 1e9
                assert config["startup"]["reference_dwell_s"] <= dwell < 1, fields
                assert all(r["pid_law"] == "ANGLE_RATE_PID" for r in controls
                           if r["control_mode"] != "STOP_MOTOR_FADE")
                assert all(r["pid_law"] == "NONE" for r in controls
                           if r["control_mode"] == "STOP_MOTOR_FADE")
            if cascade and success:
                startup = [r for r in controls if r["stage"] == "STARTUP_RAMP"]
                expected_startup = round(config["startup"]["minimum_duration_s"] * 100)
                assert expected_startup <= len(startup) <= expected_startup + 2, \
                    "configured S startup duration changed"
                values = [0.] + [float(r["collective"]) for r in startup]
                increments = [b-a for a, b in zip(values, values[1:])]
                assert all(-1e-9 <= value <= .00040001 for value in increments)
                assert increments[0] < 1e-7 and increments[-1] < 1e-7
                assert abs(values[-1] - .1) < 1e-7
                if mode in ("normal", "nonidentity_reference", "bad_reference") and len(config["segments"]) == 1:
                    hold = [r for r in controls if r["stage"] in ("SEGMENT", "SETTLING_EXTENSION")]
                    assert 100 <= len(hold) <= 102, "preparation HOLD did not finish in 1 second"
                    assert all(r["preparation_only"] == "1" for r in hold)
                    assert hold[-1]["settled"] == "1"
                    assert release_time - first_control < 9, "short stage timing regressed"
                if len(config["segments"]) > 1 and mode == "pd_response":
                    holds = [[r for r in controls if r["segment_id"] == s["segment_id"] and
                              r["stage"] in ("SEGMENT", "SETTLING_EXTENSION")] for s in config["segments"]]
                    for segment, rows in zip(config["segments"], holds):
                        assert segment["duration_s"] * 100 <= len(rows) <= segment["duration_s"] * 100 + 2
                        assert all(r["preparation_only"] == str(int(segment.get("preparation_only", False))) for r in rows)
                    learned = float(holds[0][-1]["i_term"])
                    assert abs(learned) > 1e-6
                    assert all(abs(float(r["i_term"]) - learned) < 1e-9 for rows in holds[1:] for r in rows), "I reset across segments"
                    assert any(r["analysis_window"] == "1" for rows in holds[1:] for r in rows), "scientific window missing"
            print(f"PASS {mode}: controls={control_count}, release recorded, result/logs checked", flush=True)
        finally:
            if proc is not None and proc.poll() is None:
                proc.kill()
                proc.wait()
            artifact_root = os.environ.get("FFTGYRO_TEST_ARTIFACT_DIR")
            if artifact_root:
                destination = Path(artifact_root) / (mode + "-" + root.name)
                shutil.copytree(root, destination)
                print(f"SYNTHETIC artifacts: {destination}", flush=True)
            os.close(master)
            os.close(slave)


if __name__ == "__main__":
    executable = Path(sys.argv[1]).resolve(strict=True)
    template = json.loads(Path(sys.argv[2]).read_text())
    scenarios = ["normal", "ack_loss", "replayed_ack", "telemetry_loss", "rejected", "release_loss"]
    if (template.get("control_mode", "AXIS_PID") == "AXIS_PID" and
            "--props-off" not in sys.argv[3:] and "--fixed-commissioning" not in sys.argv[3:]):
        scenarios.append("shutdown_drift")
    for scenario in scenarios:
        check(executable, template, scenario, "--props-off" in sys.argv[3:],
              fixed_commissioning="--fixed-commissioning" in sys.argv[3:])
