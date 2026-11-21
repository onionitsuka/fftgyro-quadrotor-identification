from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import statistics


UNO_BAUD = 1_000_000
MICROS_MODULUS = 2**32


@dataclass(frozen=True, slots=True)
class EdgeEvent:
    kind: str
    micros: int
    label: str | None = None


@dataclass(frozen=True, slots=True)
class RpmSample:
    edge_micros: int
    rpm: float


def parse_uno_line(line: str) -> EdgeEvent | None:
    text = line.strip()
    if not text or text in {"event,micros", "event,label,micros"}:
        return None
    fields = text.split(",")
    if len(fields) == 2 and fields[0] in {"R", "S", "O", "F"}:
        kind, label, timestamp = fields[0], None, fields[1]
    elif len(fields) == 3 and fields[0] in {"R", "S", "O", "F", "M"}:
        kind, label, timestamp = fields[0], fields[1] or None, fields[2]
    else:
        raise ValueError(f"invalid_uno_record:{text}")
    if kind == "M" and label not in {
        "MARK_START", "COMMAND_ON", "COMMAND_ACK", "COMMAND_OFF", "MARK_END"
    }:
        raise ValueError(f"invalid_uno_marker:{text}")
    try:
        value = int(timestamp)
    except ValueError as exc:
        raise ValueError(f"invalid_uno_timestamp:{text}") from exc
    if value < 0:
        raise ValueError(f"negative_uno_timestamp:{text}")
    return EdgeEvent(kind, value, label)


def rpm_from_edges(
    edge_micros: Iterable[int], *, marks_per_revolution: int = 2,
    minimum_edge_interval_us: int = 0,
) -> tuple[RpmSample, ...]:
    if marks_per_revolution <= 0:
        raise ValueError("marks_per_revolution_must_be_positive")
    if minimum_edge_interval_us < 0:
        raise ValueError("minimum_edge_interval_us_must_not_be_negative")
    samples: list[RpmSample] = []
    previous: int | None = None
    for current in edge_micros:
        if not 0 <= current < MICROS_MODULUS:
            raise ValueError("edge_timestamp_outside_uint32")
        if previous is not None:
            delta_us = (current - previous) % MICROS_MODULUS
            if delta_us == 0:
                raise ValueError("duplicate_edge_timestamp")
            if delta_us < minimum_edge_interval_us:
                continue
            rpm = 60_000_000.0 / (marks_per_revolution * delta_us)
            samples.append(RpmSample(edge_micros=current, rpm=rpm))
        previous = current
    return tuple(samples)


def reject_inserted_false_edges(
    edge_micros: Iterable[int], *, minimum_edge_interval_us: int = 5_000
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Remove isolated optical insertions while preserving the raw input separately.

    An inserted edge either creates one very short adjacent interval or splits
    one locally typical interval into two shorter pieces. The operation is
    deterministic and returns every rejected timestamp for the audit trail.
    """
    if minimum_edge_interval_us <= 0:
        raise ValueError("minimum_edge_interval_us_must_be_positive")
    cleaned = list(edge_micros)
    if any(right <= left for left, right in zip(cleaned, cleaned[1:])):
        raise ValueError("edge_timestamps_must_be_strictly_increasing")
    rejected: list[int] = []
    changed = True
    while changed and len(cleaned) >= 3:
        changed = False
        intervals = [right - left for left, right in zip(cleaned, cleaned[1:])]
        for index in range(1, len(cleaned) - 1):
            left = cleaned[index] - cleaned[index - 1]
            right = cleaned[index + 1] - cleaned[index]
            combined = left + right
            neighbors = (
                intervals[max(0, index - 5) : max(0, index - 1)]
                + intervals[min(len(intervals), index + 1) : min(len(intervals), index + 5)]
            )
            local = statistics.median(neighbors) if neighbors else max(left, right)
            remove_index: int | None = None
            if left < minimum_edge_interval_us or left < 0.2 * combined:
                remove_index = index
            elif right < minimum_edge_interval_us or right < 0.2 * combined:
                remove_index = index + 1
            elif (
                0.65 * local <= combined <= 1.35 * local
                and max(left, right) < 0.8 * local
            ):
                remove_index = index
            if remove_index is not None:
                rejected.append(cleaned.pop(remove_index))
                changed = True
                break
    return tuple(cleaned), tuple(rejected)


def rpm_from_same_mark_periods(
    edge_micros: Iterable[int], *, marks_per_revolution: int = 2
) -> tuple[RpmSample, ...]:
    """Estimate RPM from a full revolution to cancel unequal mark spacing."""
    if marks_per_revolution <= 0:
        raise ValueError("marks_per_revolution_must_be_positive")
    edges = tuple(edge_micros)
    samples: list[RpmSample] = []
    for index in range(marks_per_revolution, len(edges)):
        delta_us = edges[index] - edges[index - marks_per_revolution]
        if delta_us <= 0:
            raise ValueError("edge_timestamps_must_be_strictly_increasing")
        samples.append(
            RpmSample(edge_micros=edges[index], rpm=60_000_000.0 / delta_us)
        )
    return tuple(samples)


def capture_uno_serial(
    port: str,
    output: Path,
    *,
    duration_s: float | None = None,
    baud: int = UNO_BAUD,
) -> int:
    if duration_s is not None and duration_s <= 0.0:
        raise ValueError("duration_s_must_be_positive")
    if output.exists():
        raise FileExistsError(f"rpm_capture_output_already_exists:{output}")
    try:
        import serial  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("pyserial_missing_install_project_hardware_extra") from exc

    output.parent.mkdir(parents=True, exist_ok=True)
    event_count = 0
    start = time.monotonic()
    with serial.Serial(port, baudrate=baud, timeout=0.2) as device:
        with output.open("x", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("event", "label", "micros"))
            while duration_s is None or time.monotonic() - start < duration_s:
                raw = device.readline()
                if not raw:
                    continue
                event = parse_uno_line(raw.decode("ascii", errors="strict"))
                if event is None:
                    continue
                writer.writerow((event.kind, event.label or "", event.micros))
                stream.flush()
                event_count += 1
                if event.kind == "O":
                    raise RuntimeError("uno_ring_buffer_overflow_run_rejected")
    return event_count
