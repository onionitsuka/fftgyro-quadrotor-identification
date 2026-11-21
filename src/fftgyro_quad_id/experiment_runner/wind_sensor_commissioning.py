from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any, Iterable


PROTOCOL_ID = "wind_sensor_frame_influence_commissioning"
PROTOCOL_VERSION = "operator-intent-2026-07-16-v1"
DEFAULT_SPEEDS_MPS = (0.0, 5.0, 10.0)
APPARATUS_STATES = (
    "empty_tunnel",
    "frame_body_props_removed",
    "frame_body_props_installed_stopped",
)


class SensorMode(StrEnum):
    SIMULTANEOUS_TWO_SENSOR = "simultaneous-two-sensor"
    SEQUENTIAL_ONE_SENSOR = "sequential-one-sensor"


@dataclass(frozen=True, slots=True)
class WindSensorCommissioningSegment:
    sequence: int
    segment_id: str
    protocol_id: str
    protocol_version: str
    mode: SensorMode
    target_wind_speed_mps: float
    apparatus_state: str
    repeat: int
    measurement_position: str
    bracket_role: str
    near_frame_probe_installed: bool
    record_s: float
    rotors_running: bool = False
    propellers_rotating: bool = False
    primary_sitl_input: str = "freestream_reference"
    diagnostic_quantity: str = "near_frame_over_freestream"
    use_for_sitl_identification: bool = False
    requires_operator_confirmation: bool = True
    executable: bool = False
    blocker: str = "sensor_count_positions_and_calibration_are_unresolved"

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["mode"] = self.mode.value
        return row


@dataclass(frozen=True, slots=True)
class WindSensorCommissioningSummary:
    protocol_id: str
    protocol_version: str
    mode: str
    wind_speeds_mps: tuple[float, ...]
    apparatus_states: tuple[str, ...]
    repeats_per_condition: int
    recording_count: int
    pure_recording_seconds: float
    freestream_recordings: int
    near_frame_recordings: int
    probe_installed_recordings: int
    probe_removed_reference_recordings: int
    frame_influence_is_diagnostic_not_sitl_input: bool
    executable_recordings: int
    blockers: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["wind_speeds_mps"] = list(self.wind_speeds_mps)
        row["apparatus_states"] = list(self.apparatus_states)
        row["blockers"] = list(self.blockers)
        return row


def build_wind_sensor_commissioning_plan(
    mode: str,
    *,
    wind_speeds_mps: tuple[float, ...] = DEFAULT_SPEEDS_MPS,
    repeats: int = 3,
    record_s: float = 30.0,
) -> list[WindSensorCommissioningSegment]:
    try:
        sensor_mode = SensorMode(mode)
    except ValueError as exc:
        raise ValueError(f"unknown wind sensor mode: {mode}") from exc
    if not wind_speeds_mps or any(value < 0 for value in wind_speeds_mps):
        raise ValueError("wind speeds must be non-empty and nonnegative")
    if len(set(wind_speeds_mps)) != len(wind_speeds_mps):
        raise ValueError("wind speeds must be unique")
    if repeats < 1:
        raise ValueError("repeats must be positive")
    if record_s < 30.0:
        raise ValueError("commissioning recordings must be at least 30 seconds")

    segments: list[WindSensorCommissioningSegment] = []

    def add(**kwargs: Any) -> None:
        sequence = len(segments) + 1
        segments.append(
            WindSensorCommissioningSegment(
                sequence=sequence,
                segment_id=f"{PROTOCOL_ID}-{sequence:04d}",
                protocol_id=PROTOCOL_ID,
                protocol_version=PROTOCOL_VERSION,
                mode=sensor_mode,
                **kwargs,
            )
        )

    for speed in wind_speeds_mps:
        for state in APPARATUS_STATES:
            for repeat in range(1, repeats + 1):
                if sensor_mode == SensorMode.SIMULTANEOUS_TWO_SENSOR:
                    add(
                        target_wind_speed_mps=speed,
                        apparatus_state=state,
                        repeat=repeat,
                        measurement_position="freestream_and_near_frame_simultaneous",
                        bracket_role="diagnostic_probe_installed",
                        near_frame_probe_installed=True,
                        record_s=record_s,
                    )
                    add(
                        target_wind_speed_mps=speed,
                        apparatus_state=state,
                        repeat=repeat,
                        measurement_position="freestream_only",
                        bracket_role="interference_reference_probe_removed",
                        near_frame_probe_installed=False,
                        record_s=record_s,
                    )
                else:
                    for position, role in (
                        ("freestream_reference", "before_near_frame"),
                        ("near_frame_diagnostic", "diagnostic"),
                        ("freestream_reference", "after_near_frame"),
                    ):
                        add(
                            target_wind_speed_mps=speed,
                            apparatus_state=state,
                            repeat=repeat,
                            measurement_position=position,
                            bracket_role=role,
                            near_frame_probe_installed=(
                                position == "near_frame_diagnostic"
                            ),
                            record_s=record_s,
                        )
    return segments


def summarize_wind_sensor_commissioning(
    mode: str,
    segments: Iterable[WindSensorCommissioningSegment],
) -> WindSensorCommissioningSummary:
    items = list(segments)
    conditions = {
        (item.target_wind_speed_mps, item.apparatus_state, item.repeat)
        for item in items
    }
    repeat_count = max((item.repeat for item in items), default=0)
    if SensorMode(mode) == SensorMode.SIMULTANEOUS_TWO_SENSOR:
        freestream = len(items)
        near_frame = sum(item.near_frame_probe_installed for item in items)
    else:
        freestream = sum(
            item.measurement_position == "freestream_reference" for item in items
        )
        near_frame = sum(
            item.measurement_position == "near_frame_diagnostic" for item in items
        )
    return WindSensorCommissioningSummary(
        protocol_id=PROTOCOL_ID,
        protocol_version=PROTOCOL_VERSION,
        mode=mode,
        wind_speeds_mps=tuple(sorted({item.target_wind_speed_mps for item in items})),
        apparatus_states=APPARATUS_STATES,
        repeats_per_condition=repeat_count,
        recording_count=len(items),
        pure_recording_seconds=sum(item.record_s for item in items),
        freestream_recordings=freestream,
        near_frame_recordings=near_frame,
        probe_installed_recordings=sum(
            item.near_frame_probe_installed for item in items
        ),
        probe_removed_reference_recordings=sum(
            not item.near_frame_probe_installed for item in items
        ),
        frame_influence_is_diagnostic_not_sitl_input=True,
        executable_recordings=sum(item.executable for item in items),
        blockers=tuple(sorted({item.blocker for item in items if item.blocker})),
    )
