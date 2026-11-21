"""Read-only admission checks for a separately qualified parallel session."""

import math
import re
import json
import csv
from pathlib import Path


def shutdown_guard_after_complete_waveform(out: Path) -> bool:
    """Accept only a post-waveform body-rate guard with an acknowledged release.

    A free Yaw axis can accelerate in wind while collective is ramped to zero.
    That remains a useful safety abort, but it must not discard a waveform that
    has already finished.  This check is deliberately narrow: every
    non-preparation segment must have reached its declared duration and the
    final PX4 release command must have been acknowledged.
    """
    try:
        result_lines = (out / "pid/result.txt").read_text(encoding="utf-8").splitlines()
        statuses = [line.split("=", 1)[1] for line in result_lines if line.startswith("status=")]
        details = [line.split("=", 1)[1] for line in result_lines if line.startswith("detail=")]
        if not statuses or statuses[-1] != "ABORTED" or not details or details[-1] != "body rate limit":
            return False

        config = json.loads((out / "pid/session.config.json").read_text(encoding="utf-8"))
        expected = {
            segment["segment_id"]: float(segment["duration_s"])
            for segment in config["segments"]
            if not segment.get("preparation_only", False)
        }
        if not expected:
            return False

        observed: dict[str, float] = {}
        last_stage = None
        with (out / "pid/control.csv").open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                last_stage = row.get("stage")
                if row.get("stage") == "SEGMENT" and row.get("segment_id") in expected:
                    observed[row["segment_id"]] = max(
                        observed.get(row["segment_id"], 0.0), float(row["segment_time_s"])
                    )
        if last_stage != "RAMP_DOWN":
            return False
        # At 100 Hz, the last stored sample precedes the mathematical endpoint.
        if any(observed.get(segment_id, -1.0) < duration - 0.05
               for segment_id, duration in expected.items()):
            return False

        with (out / "pid/ack.csv").open(encoding="utf-8", newline="") as stream:
            acknowledgements = list(csv.DictReader(stream))
        return bool(
            acknowledgements
            and acknowledgements[-1].get("release") == "1"
            and acknowledgements[-1].get("ack_status") == "1"
        )
    except (ValueError, OSError, IndexError, KeyError, TypeError, csv.Error):
        return False


def acquisition_artifacts_complete(out: Path) -> bool:
    """Check the four synchronized acquisition products, independent of SSH exit."""
    try:
        base_complete = (
            ("status=COMPLETE" in (out / "pid/result.txt").read_text(encoding="utf-8").splitlines()
             or shutdown_guard_after_complete_waveform(out))
            and (out / "pid/control.csv").stat().st_size > 0
            and (out / "rpm/result.txt").read_text(encoding="utf-8").splitlines()[0] == "COMPLETE"
            and (out / "rpm/rpm.csv").stat().st_size > 0
            and json.loads((out / "leptrino/summary.json").read_text(encoding="utf-8")).get("state")
                in ("COMPLETE", "RAW_COMPLETE")
            and (out / "leptrino/chunks.bin").stat().st_size > 0
            and json.loads((out / "clock/summary.json").read_text(encoding="utf-8")).get("state") == "COMPLETE"
            and (out / "clock/exchanges.csv").stat().st_size > 0
        )
        if not base_complete:
            return False
        manifest_path = out / "run.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
        if not manifest.get("encoder_required", False):
            return True
        encoder = json.loads((out / "encoder/metadata.json").read_text(encoding="utf-8"))
        encoder_clock = json.loads((out / "encoder-clock/summary.json").read_text(encoding="utf-8"))
        return (
            encoder.get("state") == "COMPLETE"
            and encoder.get("encoder_frames", 0) >= 20
            and (out / "encoder/raw.bin").stat().st_size > 0
            and (out / "encoder/samples.csv").stat().st_size > 0
            and encoder_clock.get("state") == "COMPLETE"
            and encoder_clock.get("server_boot_id") == manifest.get("aircraft_boot_id")
            and (out / "encoder-clock/exchanges.csv").stat().st_size > 0
        )
    except (ValueError, OSError, IndexError):
        return False


def check_capture(config: dict, executor: str, seconds: int, *, fixed: bool,
                  axis_ready: bool, props_installed: bool, operator_ready: bool,
                  charged: bool, props_removed: bool = False, fixed_commissioning: bool = False,
                  pitch_commissioning: bool = False, yaw_commissioning: bool = False) -> float:
    if not re.fullmatch(r"/home/fftgyro/[A-Za-z0-9_./-]+", executor) or ".." in executor.split("/"):
        raise ValueError("select an explicit separately deployed executor under /home/fftgyro")
    if not (operator_ready and charged):
        raise ValueError("operator-stop-ready and charged-4s confirmations required")
    if props_installed == props_removed:
        raise ValueError("confirm exactly one of props-installed or props-removed")
    if sum((fixed_commissioning, pitch_commissioning, yaw_commissioning)) > 1:
        raise ValueError("select only one commissioning mode")
    if yaw_commissioning:
        if (config.get("runtime_qualified") is not False or config.get("control_mode") != "AXIS_PID"
                or config.get("pid", {}).get("axis") != "YAW" or fixed or not axis_ready
                or not props_installed):
            raise ValueError("Yaw commissioning requires initial AXIS_PID and only Yaw free")
        h = config["hardware"]
        if any(h.get(flag) is not True for flag in ("motor_mapping_verified", "feedback_frame_verified")):
            raise ValueError("Yaw motor mapping and feedback frame must be verified")
        if h.get("mechanically_locked_verified", False) or config["startup"].get("control_ready_verified", False):
            raise ValueError("Yaw commissioning is not fixed or control-qualified")
        if h.get("uart_device") != "/dev/serial0" or h.get("non_target_pid_enabled") is not False:
            raise ValueError("requires aircraft UART and non-target PID disabled")
        duration = config["maximum_duration_s"]
        limit = 18 if config.get("pid", {}).get("law") == "ANGLE_RATE_PID" else 12
        if isinstance(duration, bool) or not isinstance(duration, (float, int)) or not 0 < duration <= min(limit, seconds - 40):
            raise ValueError("Yaw commissioning duration/recording limit")
        return float(duration)
    if pitch_commissioning:
        if (config.get("runtime_qualified") is not False or config.get("control_mode") != "AXIS_PID"
                or config.get("pid", {}).get("axis") != "PITCH" or fixed or not axis_ready
                or not props_installed):
            raise ValueError("Pitch commissioning requires initial AXIS_PID and only Pitch free")
        h = config["hardware"]
        if any(h.get(flag) is not True for flag in ("motor_mapping_verified", "feedback_frame_verified")):
            raise ValueError("Pitch motor mapping and feedback frame must be verified")
        if h.get("mechanically_locked_verified", False) or config["startup"].get("control_ready_verified", False):
            raise ValueError("Pitch commissioning is not fixed or control-qualified")
        if h.get("uart_device") != "/dev/serial0" or h.get("non_target_pid_enabled") is not False:
            raise ValueError("requires aircraft UART and non-target PID disabled")
        duration = config["maximum_duration_s"]
        limit = 18 if config.get("pid", {}).get("law") == "ANGLE_RATE_PID" else 12
        if isinstance(duration, bool) or not isinstance(duration, (float, int)) or not 0 < duration <= min(limit, seconds - 40):
            raise ValueError("Pitch commissioning duration/recording limit")
        # Dedicated native validator checks the full legacy/cascade/reference envelope.
        return float(duration)
    if fixed_commissioning:
        if (config.get("runtime_qualified") is not False or config.get("control_mode") != "FIXED_HOLD"
                or not fixed or axis_ready or not props_installed):
            raise ValueError("fixed commissioning requires an unqualified FIXED_HOLD and full lock")
        duration = config["maximum_duration_s"]
        rate = config.get("hardware", {}).get("startup_ramp_rate", 0.1)
        slow = not isinstance(rate, bool) and isinstance(rate, (int, float)) and 0 < rate <= .02
        very_slow = slow and rate <= .01
        limit = 18 if very_slow else 13 if slow else 12
        if isinstance(duration, bool) or not isinstance(duration, (float, int)) or not 0 < duration <= min(limit, seconds - 40):
            raise ValueError("fixed commissioning duration/recording limit")
        # Complete motor/feedback/segment bounds are enforced by the named native validator
        # before any capture, and again by the runtime before opening UART.
        return float(duration)
    coupled = config.get("coupled_yaw")
    if coupled is not None:
        if (config.get("control_mode") != "AXIS_PID" or config.get("pid", {}).get("axis") != "PITCH"
                or coupled.get("pid", {}).get("axis") != "YAW"
                or config.get("coupled_trial_status") != "EXPERIMENTAL"):
            raise ValueError("invalid experimental Pitch/Yaw session")
    if config.get("runtime_qualified") is not True and coupled is None:
        raise ValueError("session is a draft: runtime_qualified=false; no capture or motor start")
    if config.get("schema") != "fftgyro-axis-session-v1":
        raise ValueError("unsupported session schema")
    h = config["hardware"]
    for flag in ("motor_mapping_verified", "feedback_frame_verified", "stop_verified",
                 "rpm_logging_verified", "load_logging_verified", "sync_verified"):
        if h.get(flag) is not True:
            raise ValueError(f"missing qualification: {flag}")
    if config.get("control_mode") == "FIXED_INPUT" and h.get("fixed_input_mapping_verified") is not True:
        raise ValueError("missing qualification: fixed_input_mapping_verified")
    if h.get("uart_device") != "/dev/serial0" or h.get("non_target_pid_enabled") is not False:
        raise ValueError("requires aircraft UART and non-target PID disabled")
    mode = config.get("control_mode", "AXIS_PID")
    if mode in ("FIXED_HOLD", "FIXED_INPUT"):
        if not fixed or axis_ready or h.get("mechanically_locked_verified") is not True:
            raise ValueError("fixed session requires full mechanical lock, not axis-ready")
    elif mode == "AXIS_PID":
        if (fixed or not axis_ready or not props_installed
                or config["startup"].get("control_ready_verified") is not True):
            raise ValueError("dynamic session requires axis-ready, not full fixed lock")
    else:
        raise ValueError("unknown control mode")
    duration = config["maximum_duration_s"]
    if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not math.isfinite(duration):
        raise ValueError("invalid session duration")
    if not 0 < duration <= seconds - 40:
        raise ValueError("force recording must exceed the session limit by at least 40 seconds")
    return float(duration)
