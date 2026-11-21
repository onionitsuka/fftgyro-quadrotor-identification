from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any, Iterable


PROTOCOL_ID = "loadcell_geometry_sync_calibration"
PROTOCOL_VERSION = "pdf-2026-07-11-v1"


class CalibrationKind(StrEnum):
    OFFSET_RESET = "offset_reset"
    ZERO = "zero"
    KNOWN_FORCE = "known_force"
    KNOWN_MOMENT = "known_moment"
    SYNC = "sync"
    COORDINATE = "coordinate"


FORCE_DIRECTIONS = ("+x", "-x", "+y", "-y", "+z", "-z")
GRAVITY_ORIENTATIONS = (
    "+x_up",
    "-x_up",
    "+y_up",
    "-y_up",
    "+z_up",
    "-z_up",
)


@dataclass(frozen=True, slots=True)
class CalibrationSegment:
    sequence: int
    segment_id: str
    protocol_id: str
    protocol_version: str
    kind: CalibrationKind
    purpose: str
    repeat: int
    duration_s: float | None
    force_direction: str | None = None
    force_level_index: int | None = None
    known_force_n: float | None = None
    moment_point_id: str | None = None
    known_moment_nm: float | None = None
    gravity_orientation: str | None = None
    requires_operator_value: bool = False

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["kind"] = self.kind.value
        return row


@dataclass(frozen=True, slots=True)
class CalibrationSummary:
    protocol_id: str
    protocol_version: str
    mode: str
    segment_count: int
    zero_segments: int
    known_force_points: int
    known_moment_points: int
    sync_steps: int
    coordinate_checks: int
    unresolved_segments: int
    fixed_recording_seconds: float
    scheduled_recording_seconds: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_factory_calibrated_plan(
    *,
    sync_step_s: float | None = None,
    coordinate_hold_s: float | None = None,
) -> list[CalibrationSegment]:
    """Operational Phase 2 for a factory-calibrated Leptrino sensor.

    This does not attempt to replace the sensor's factory conversion matrix.
    The LGR101U official tool performs the offset reset; the experiment plan
    records that operator action and independently checks timing and axis signs.
    """
    for name, duration in (
        ("sync_step_s", sync_step_s),
        ("coordinate_hold_s", coordinate_hold_s),
    ):
        if duration is not None and duration <= 0:
            raise ValueError(f"{name} must be positive when specified")

    segments: list[CalibrationSegment] = []

    def add(**kwargs: Any) -> None:
        sequence = len(segments) + 1
        segments.append(
            CalibrationSegment(
                sequence=sequence,
                segment_id=f"{PROTOCOL_ID}-{sequence:04d}",
                protocol_id=PROTOCOL_ID,
                protocol_version=PROTOCOL_VERSION,
                **kwargs,
            )
        )

    add(
        kind=CalibrationKind.OFFSET_RESET,
        purpose="official_tool_offset_reset",
        repeat=1,
        duration_s=0.0,
    )
    add(
        kind=CalibrationKind.ZERO,
        purpose="day_start_zero_after_offset_reset",
        repeat=1,
        duration_s=30.0,
    )
    for repeat in range(1, 11):
        add(
            kind=CalibrationKind.SYNC,
            purpose="motor_step_sync",
            repeat=repeat,
            duration_s=sync_step_s,
            requires_operator_value=sync_step_s is None,
        )
    for orientation in GRAVITY_ORIENTATIONS:
        add(
            kind=CalibrationKind.COORDINATE,
            purpose="axis_sign_sanity_check_not_scale_calibration",
            repeat=1,
            duration_s=coordinate_hold_s,
            gravity_orientation=orientation,
            requires_operator_value=coordinate_hold_s is None,
        )
    add(
        kind=CalibrationKind.ZERO,
        purpose="day_end_zero",
        repeat=1,
        duration_s=30.0,
    )
    return segments


def build_full_validation_plan(
    *,
    force_levels_n: tuple[float, float, float] | None = None,
    force_repeats: int = 3,
    moment_values_nm: tuple[float, ...] | None = None,
    static_hold_s: float | None = None,
    sync_step_s: float | None = None,
    coordinate_hold_s: float | None = None,
) -> list[CalibrationSegment]:
    """Build Phase 2 without inventing lab-specific weights or lever arms.

    Three force levels in six directions and three independent placements give
    the PDF minimum of 54 force points. The 20 moment values must come from the
    actual calibration fixture geometry.
    """
    if force_repeats < 1:
        raise ValueError("force_repeats must be positive")
    if force_levels_n is not None:
        if len(force_levels_n) != 3:
            raise ValueError("exactly three force levels are required")
        if any(value <= 0 for value in force_levels_n):
            raise ValueError("force levels must be positive")
        if len(set(force_levels_n)) != 3:
            raise ValueError("force levels must be distinct")
    if moment_values_nm is not None:
        if len(moment_values_nm) < 20:
            raise ValueError("at least 20 known moment points are required")
        if any(value == 0 for value in moment_values_nm):
            raise ValueError("known moment points must be non-zero")
    for name, duration in (
        ("static_hold_s", static_hold_s),
        ("sync_step_s", sync_step_s),
        ("coordinate_hold_s", coordinate_hold_s),
    ):
        if duration is not None and duration <= 0:
            raise ValueError(f"{name} must be positive when specified")

    segments: list[CalibrationSegment] = []

    def add(**kwargs: Any) -> None:
        sequence = len(segments) + 1
        segments.append(
            CalibrationSegment(
                sequence=sequence,
                segment_id=f"{PROTOCOL_ID}-{sequence:04d}",
                protocol_id=PROTOCOL_ID,
                protocol_version=PROTOCOL_VERSION,
                **kwargs,
            )
        )

    add(
        kind=CalibrationKind.ZERO,
        purpose="day_start_zero",
        repeat=1,
        duration_s=30.0,
    )

    for direction in FORCE_DIRECTIONS:
        for level_index in range(1, 4):
            value = None if force_levels_n is None else force_levels_n[level_index - 1]
            for repeat in range(1, force_repeats + 1):
                add(
                    kind=CalibrationKind.KNOWN_FORCE,
                    purpose="known_force_calibration",
                    repeat=repeat,
                    duration_s=static_hold_s,
                    force_direction=direction,
                    force_level_index=level_index,
                    known_force_n=value,
                    requires_operator_value=value is None or static_hold_s is None,
                )

    moments = moment_values_nm if moment_values_nm is not None else (None,) * 20
    for index, value in enumerate(moments, start=1):
        add(
            kind=CalibrationKind.KNOWN_MOMENT,
            purpose="known_moment_calibration",
            repeat=1,
            duration_s=static_hold_s,
            moment_point_id=f"moment_{index:02d}",
            known_moment_nm=value,
            requires_operator_value=value is None or static_hold_s is None,
        )

    for repeat in range(1, 11):
        add(
            kind=CalibrationKind.SYNC,
            purpose="motor_step_sync",
            repeat=repeat,
            duration_s=sync_step_s,
            requires_operator_value=sync_step_s is None,
        )

    for orientation in GRAVITY_ORIENTATIONS:
        add(
            kind=CalibrationKind.COORDINATE,
            purpose="gravity_coordinate_sign_check",
            repeat=1,
            duration_s=coordinate_hold_s,
            gravity_orientation=orientation,
            requires_operator_value=coordinate_hold_s is None,
        )

    add(
        kind=CalibrationKind.ZERO,
        purpose="day_end_zero",
        repeat=1,
        duration_s=30.0,
    )
    return segments


def summarize(
    segments: Iterable[CalibrationSegment], *, mode: str
) -> CalibrationSummary:
    items = list(segments)
    durations = [item.duration_s for item in items]
    return CalibrationSummary(
        protocol_id=PROTOCOL_ID,
        protocol_version=PROTOCOL_VERSION,
        mode=mode,
        segment_count=len(items),
        zero_segments=sum(item.kind == CalibrationKind.ZERO for item in items),
        known_force_points=sum(
            item.kind == CalibrationKind.KNOWN_FORCE for item in items
        ),
        known_moment_points=sum(
            item.kind == CalibrationKind.KNOWN_MOMENT for item in items
        ),
        sync_steps=sum(item.kind == CalibrationKind.SYNC for item in items),
        coordinate_checks=sum(
            item.kind == CalibrationKind.COORDINATE for item in items
        ),
        unresolved_segments=sum(item.requires_operator_value for item in items),
        fixed_recording_seconds=sum(value for value in durations if value is not None),
        scheduled_recording_seconds=(
            None if any(value is None for value in durations) else sum(durations)
        ),
    )
