from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PulseAcknowledgement:
    run_id: str
    sequence: int
    release_time_us: int
    width_us: int


def ping_command() -> bytes:
    return b"PING\n"


def arm_command(run_id: str) -> bytes:
    if not run_id or len(run_id) > 24 or not run_id.replace("-", "").replace("_", "").isalnum():
        raise ValueError("run_id must be 1-24 ASCII letters, digits, '-' or '_'")
    return f"ARM {run_id}\n".encode("ascii")


def pulse_command(sequence: int, width_us: int = 1000) -> bytes:
    if sequence < 0:
        raise ValueError("sequence must be non-negative")
    if not 100 <= width_us <= 10_000:
        raise ValueError("width_us must be in [100, 10000]")
    return f"PULSE {sequence} {width_us}\n".encode("ascii")


def disarm_command() -> bytes:
    return b"DISARM\n"


def parse_pulse_ack(line: str) -> PulseAcknowledgement:
    fields = line.strip().split()
    if len(fields) != 5 or fields[0] != "PULSED":
        raise ValueError(f"not a PULSED acknowledgement: {line!r}")
    try:
        sequence = int(fields[2])
        release_time_us = int(fields[3])
        width_us = int(fields[4])
    except ValueError as exc:
        raise ValueError(f"invalid numeric field in acknowledgement: {line!r}") from exc
    return PulseAcknowledgement(
        run_id=fields[1],
        sequence=sequence,
        release_time_us=release_time_us,
        width_us=width_us,
    )
