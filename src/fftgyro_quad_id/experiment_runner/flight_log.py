from __future__ import annotations

import csv
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


ARMED_EVENT_ID = 10
DISARMED_EVENT_ID = 11


@dataclass(frozen=True, slots=True)
class ArmWindow:
    arm_time_us: int
    disarm_time_us: int

    @property
    def duration_s(self) -> float:
        return (self.disarm_time_us - self.arm_time_us) / 1_000_000.0


@dataclass(frozen=True, slots=True)
class MotorOutputSample:
    time_us: int
    values: tuple[float, ...]
    source_message: str


@dataclass(frozen=True, slots=True)
class FlightExtractionReport:
    arm_window: ArmWindow
    arm_window_count: int
    sample_count: int
    source_message: str
    observed_rate_hz: float | None
    motor_channels: tuple[int, ...]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        document = asdict(self)
        document["arm_window"] = asdict(self.arm_window)
        return document


def find_arm_windows(
    records: Iterable[Mapping[str, Any]],
    *,
    implicit_arm_time_us: int | None = None,
) -> tuple[ArmWindow, ...]:
    transitions: list[tuple[int, bool]] = []
    for record in records:
        message_type = str(record.get("mavpackettype", ""))
        time_us = _time_us(record)
        if time_us is None:
            continue
        if message_type == "EV":
            event_id = int(record.get("Id", -1))
            if event_id == ARMED_EVENT_ID:
                transitions.append((time_us, True))
            elif event_id == DISARMED_EVENT_ID:
                transitions.append((time_us, False))
        elif message_type in {"ARM", "ARMD"} and "ArmState" in record:
            transitions.append((time_us, bool(record["ArmState"])))

    transitions.sort()
    windows: list[ArmWindow] = []
    arm_time = implicit_arm_time_us
    state = implicit_arm_time_us is not None
    for time_us, armed in transitions:
        # Newer logs can contain both ARM and EV records for one transition.
        if armed == state:
            continue
        state = armed
        if armed:
            arm_time = time_us
        elif arm_time is not None and time_us > arm_time:
            windows.append(ArmWindow(arm_time, time_us))
            arm_time = None
    return tuple(windows)


def extract_motor_outputs(
    records: Iterable[Mapping[str, Any]],
    *,
    arm_index: int = 0,
    motor_channels: tuple[int, ...] = (9, 10, 11, 12),
) -> tuple[tuple[MotorOutputSample, ...], FlightExtractionReport]:
    if arm_index < 0:
        raise ValueError("arm_index_must_be_non_negative")
    if not motor_channels or any(channel < 1 or channel > 32 for channel in motor_channels):
        raise ValueError("motor_channels_must_be_in_1_to_32")

    materialized = tuple(records)
    windows = find_arm_windows(materialized)
    boundary_warnings: list[str] = []
    if not windows:
        # With LOG_DISARMED=0 and LOG_FILE_DSRMROT=1, ArduPilot starts the
        # file during the arm transition. The armed EV/ARM record can precede
        # the first writable file record, while the disarm event is present.
        # MRAW is emitted only while armed, so its first sample is a safe,
        # slightly-late boundary for this file layout.
        first_mraw_time = min(
            (
                time_us
                for record in materialized
                if record.get("mavpackettype") == "MRAW"
                and (time_us := _time_us(record)) is not None
            ),
            default=None,
        )
        if first_mraw_time is not None:
            windows = find_arm_windows(
                materialized,
                implicit_arm_time_us=first_mraw_time,
            )
            if windows:
                boundary_warnings.append("arm_boundary_inferred_from_first_armed_mraw")
    if arm_index >= len(windows):
        raise ValueError(
            f"complete_arm_window_not_found:index={arm_index}:count={len(windows)}"
        )
    window = windows[arm_index]

    # MRAW is produced by the optional 100 Hz Lua logger. RCOU is normally 10 Hz.
    has_mraw = any(
        record.get("mavpackettype") == "MRAW"
        and _inside(record, window)
        for record in materialized
    )
    source = "MRAW" if has_mraw else "RCOU"
    samples: list[MotorOutputSample] = []
    for record in materialized:
        if record.get("mavpackettype") != source or not _inside(record, window):
            continue
        time_us = _time_us(record)
        assert time_us is not None
        if source == "MRAW":
            values = tuple(float(record.get(f"M{index}", 0.0)) for index in range(1, 5))
            if len(motor_channels) != 4:
                raise ValueError("MRAW_requires_exactly_four_motor_channels")
        else:
            values = tuple(float(record.get(f"C{channel}", 0.0)) for channel in motor_channels)
        samples.append(MotorOutputSample(time_us, values, source))

    if not samples:
        raise ValueError(f"no_{source.lower()}_samples_inside_arm_window")
    samples.sort(key=lambda sample: sample.time_us)
    deltas = [
        (current.time_us - previous.time_us) / 1_000_000.0
        for previous, current in zip(samples, samples[1:])
        if current.time_us > previous.time_us
    ]
    rate = 1.0 / statistics.median(deltas) if deltas else None
    warnings = boundary_warnings
    if source == "RCOU":
        warnings.append("using_standard_RCOU_nominally_10Hz_install_MRAW_lua_for_transients")
    if rate is not None and rate < 50.0:
        warnings.append("motor_output_rate_below_50Hz")

    report = FlightExtractionReport(
        arm_window=window,
        arm_window_count=len(windows),
        sample_count=len(samples),
        source_message=source,
        observed_rate_hz=rate,
        motor_channels=motor_channels,
        warnings=tuple(warnings),
    )
    return tuple(samples), report


def read_ardupilot_bin(path: Path) -> tuple[dict[str, Any], ...]:
    try:
        from pymavlink import mavutil  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("pymavlink_missing_install_project_hardware_extra") from exc

    connection = mavutil.mavlink_connection(str(path))
    records: list[dict[str, Any]] = []
    while True:
        message = connection.recv_match(blocking=False)
        if message is None:
            break
        message_type = message.get_type()
        if message_type in {"EV", "ARM", "ARMD", "RCOU", "MRAW"}:
            records.append(message.to_dict())
    return tuple(records)


def extract_ardupilot_flight_csv(
    input_path: Path,
    output_path: Path,
    *,
    arm_index: int = 0,
    motor_channels: tuple[int, ...] = (9, 10, 11, 12),
) -> FlightExtractionReport:
    samples, report = extract_motor_outputs(
        read_ardupilot_bin(input_path),
        arm_index=arm_index,
        motor_channels=motor_channels,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "time_s",
                "time_us",
                *(f"motor_{index}_output_raw" for index in range(1, len(motor_channels) + 1)),
                "source_message",
            )
        )
        for sample in samples:
            writer.writerow(
                (
                    f"{(sample.time_us - report.arm_window.arm_time_us) / 1_000_000.0:.9f}",
                    sample.time_us,
                    *sample.values,
                    sample.source_message,
                )
            )
    return report


def _time_us(record: Mapping[str, Any]) -> int | None:
    value = record.get("TimeUS")
    if value is None:
        return None
    return int(value)


def _inside(record: Mapping[str, Any], window: ArmWindow) -> bool:
    time_us = _time_us(record)
    return time_us is not None and window.arm_time_us <= time_us <= window.disarm_time_us
