from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any, Iterable

from .angle_grid import PROFILES
from .axis_policy import DYNAMIC_EXCITATION_AXES, ROLL_FIXED_DEG


PROTOCOL_6A_ID = "phase6a_wind_fixed_pose"
PROTOCOL_6B_ID = "phase6b_wind_single_axis_dynamic"
PROTOCOL_VERSION = "pdf-2026-07-11-roll-fixed-v1"


class Phase6AKind(StrEnum):
    BETA_SETUP = "beta_setup"
    POSE_TARE = "pose_tare"
    STATIC_MEASUREMENT = "static_measurement"
    DOPT_SELECTION_SLOT = "dopt_selection_slot"


@dataclass(frozen=True, slots=True)
class Phase6ASegment:
    sequence: int
    segment_id: str
    protocol_id: str
    protocol_version: str
    kind: Phase6AKind
    beta_block_index: int
    beta_nominal_deg: float
    alpha_fixture_slot: int | None
    alpha_nominal_deg: float | None
    fixture_id: str | None
    wind_level_index: int | None
    wind_speed_mps: float | None
    collective_level_index: int | None
    collective_percent_hover: float | None
    input_selection_slot: int | None
    differential_input: str | None
    repeat: int | None
    settle_s: float | None
    record_s: float | None
    roll_fixed_deg: float = ROLL_FIXED_DEG
    pitch_fixed: bool = True
    yaw_fixed: bool = True
    beta_dynamic_command_allowed: bool = False
    requires_wind_off: bool = False
    requires_rotors_stopped: bool = False
    requires_operator_confirmation: bool = False
    executable: bool = False
    blocker: str | None = None
    wind_sensor_mode: str = "freestream_only"
    near_frame_probe_must_be_removed: bool = True

    @property
    def duration_s(self) -> float | None:
        if self.settle_s is None or self.record_s is None:
            return None
        return self.settle_s + self.record_s

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["kind"] = self.kind.value
        row["duration_s"] = self.duration_s
        return row


@dataclass(frozen=True, slots=True)
class Phase6ASummary:
    protocol_id: str
    protocol_version: str
    profile: str
    angle_profile: str
    configured_wind_speeds_mps: tuple[float, ...]
    wind_level_count: int
    pdf_target_wind_level_count: int
    wind_level_shortfall: int
    beta_blocks: int
    alpha_fixture_slots: int
    neutral_data_segments: int
    doptimal_selection_slots: int
    total_data_segments: int
    beta_setup_actions: int
    pose_tare_actions: int
    known_measurement_seconds: float
    pdf_5x5_neutral_segments: int
    pdf_5x5_total_segments: int
    count_changed_from_pdf_by_angle_grid: bool
    roll_fixed_deg: float
    dynamic_attitude_test_included: bool
    executable_segments: int
    blockers: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["blockers"] = list(self.blockers)
        row["configured_wind_speeds_mps"] = list(self.configured_wind_speeds_mps)
        return row


def build_phase6a_plan(
    profile: str,
    angle_profile: str = "sun-beta45-coarse",
    *,
    alpha_nominal_deg: tuple[float, ...] | None = None,
    fixture_ids: tuple[str, ...] | None = None,
    wind_speeds_mps: tuple[float, ...] | None = None,
    collective_percent_hover: tuple[float, ...] | None = None,
) -> list[Phase6ASegment]:
    if profile not in {"pilot", "minimum", "recommended"}:
        raise ValueError(f"unknown Phase 6A profile: {profile}")
    try:
        angles = PROFILES[angle_profile]
    except KeyError as exc:
        raise ValueError(f"unknown angle-grid profile: {angle_profile}") from exc

    alpha_count = angles.alpha_level_count
    pdf_wind_count = 2 if profile == "pilot" else 4
    if wind_speeds_mps is not None:
        if not wind_speeds_mps:
            raise ValueError("at least one Phase 6 wind speed is required")
        if any(value <= 0 for value in wind_speeds_mps):
            raise ValueError("Phase 6 wind speeds must be positive; 0 m/s belongs to Phase 5")
        if len(set(wind_speeds_mps)) != len(wind_speeds_mps):
            raise ValueError("Phase 6 wind speeds must be unique")
        wind_count = len(wind_speeds_mps)
    else:
        wind_count = pdf_wind_count
    collective_count = {"pilot": 1, "minimum": 2, "recommended": 3}[profile]
    for name, values, count in (
        ("alpha levels", alpha_nominal_deg, alpha_count),
        ("fixture IDs", fixture_ids, alpha_count),
        ("collective levels", collective_percent_hover, collective_count),
    ):
        if values is not None and len(values) != count:
            raise ValueError(f"exactly {count} {name} are required")

    segments: list[Phase6ASegment] = []

    def add(**kwargs: Any) -> None:
        sequence = len(segments) + 1
        segments.append(
            Phase6ASegment(
                sequence=sequence,
                segment_id=f"{PROTOCOL_6A_ID}-{sequence:05d}",
                protocol_id=PROTOCOL_6A_ID,
                protocol_version=PROTOCOL_VERSION,
                **kwargs,
            )
        )

    for beta_index, beta in enumerate(angles.beta_deg, start=1):
        add(
            kind=Phase6AKind.BETA_SETUP,
            beta_block_index=beta_index,
            beta_nominal_deg=beta,
            alpha_fixture_slot=None,
            alpha_nominal_deg=None,
            fixture_id=None,
            wind_level_index=None,
            wind_speed_mps=None,
            collective_level_index=None,
            collective_percent_hover=None,
            input_selection_slot=None,
            differential_input=None,
            repeat=None,
            settle_s=None,
            record_s=None,
            requires_wind_off=True,
            requires_rotors_stopped=True,
            requires_operator_confirmation=True,
            blocker="beta_angle_and_clearance_require_operator_confirmation",
        )
        for alpha_index in range(alpha_count):
            alpha = None if alpha_nominal_deg is None else alpha_nominal_deg[alpha_index]
            fixture = None if fixture_ids is None else fixture_ids[alpha_index]
            add(
                kind=Phase6AKind.POSE_TARE,
                beta_block_index=beta_index,
                beta_nominal_deg=beta,
                alpha_fixture_slot=alpha_index + 1,
                alpha_nominal_deg=alpha,
                fixture_id=fixture,
                wind_level_index=None,
                wind_speed_mps=None,
                collective_level_index=None,
                collective_percent_hover=None,
                input_selection_slot=None,
                differential_input=None,
                repeat=None,
                settle_s=None,
                record_s=None,
                requires_wind_off=True,
                requires_rotors_stopped=True,
                requires_operator_confirmation=True,
                blocker="pose_angle_fixture_and_tare_duration_require_confirmation",
            )
            for wind_index in range(wind_count):
                wind = None if wind_speeds_mps is None else wind_speeds_mps[wind_index]
                for collective_index in range(collective_count):
                    collective = (
                        None
                        if collective_percent_hover is None
                        else collective_percent_hover[collective_index]
                    )
                    if profile == "recommended":
                        selection_slots = range(1, 4)
                        repeats = (1, 2)
                    else:
                        selection_slots = (0,)
                        repeats = (1,)
                    for selection_slot in selection_slots:
                        for repeat in repeats:
                            add(
                                kind=Phase6AKind.STATIC_MEASUREMENT,
                                beta_block_index=beta_index,
                                beta_nominal_deg=beta,
                                alpha_fixture_slot=alpha_index + 1,
                                alpha_nominal_deg=alpha,
                                fixture_id=fixture,
                                wind_level_index=wind_index + 1,
                                wind_speed_mps=wind,
                                collective_level_index=collective_index + 1,
                                collective_percent_hover=collective,
                                input_selection_slot=(
                                    selection_slot if profile == "recommended" else None
                                ),
                                differential_input=(
                                    None if profile == "recommended" else "neutral"
                                ),
                                repeat=repeat,
                                settle_s=10.0,
                                record_s=10.0,
                                blocker=(
                                    "doptimal_input_selection_is_unresolved"
                                    if profile == "recommended"
                                    else "physical_condition_values_are_unresolved"
                                ),
                            )
                    if profile == "minimum":
                        add(
                            kind=Phase6AKind.DOPT_SELECTION_SLOT,
                            beta_block_index=beta_index,
                            beta_nominal_deg=beta,
                            alpha_fixture_slot=alpha_index + 1,
                            alpha_nominal_deg=alpha,
                            fixture_id=fixture,
                            wind_level_index=wind_index + 1,
                            wind_speed_mps=wind,
                            collective_level_index=collective_index + 1,
                            collective_percent_hover=collective,
                            input_selection_slot=1,
                            differential_input=None,
                            repeat=1,
                            settle_s=10.0,
                            record_s=10.0,
                            blocker="doptimal_input_selection_is_unresolved",
                        )
    return segments


def summarize_phase6a(
    profile: str,
    angle_profile: str,
    segments: Iterable[Phase6ASegment],
) -> Phase6ASummary:
    items = list(segments)
    neutral = sum(
        item.kind == Phase6AKind.STATIC_MEASUREMENT
        and item.differential_input == "neutral"
        for item in items
    )
    selected = sum(
        item.kind in {Phase6AKind.STATIC_MEASUREMENT, Phase6AKind.DOPT_SELECTION_SLOT}
        and item.differential_input != "neutral"
        for item in items
    )
    pdf_neutral = {"pilot": 100, "minimum": 200, "recommended": 0}[profile]
    pdf_total = {"pilot": 100, "minimum": 400, "recommended": 1800}[profile]
    data_items = [
        item
        for item in items
        if item.kind in {Phase6AKind.STATIC_MEASUREMENT, Phase6AKind.DOPT_SELECTION_SLOT}
    ]
    configured_winds = tuple(
        sorted(
            {
                item.wind_speed_mps
                for item in data_items
                if item.wind_speed_mps is not None
            }
        )
    )
    return Phase6ASummary(
        protocol_id=PROTOCOL_6A_ID,
        protocol_version=PROTOCOL_VERSION,
        profile=profile,
        angle_profile=angle_profile,
        configured_wind_speeds_mps=configured_winds,
        wind_level_count=len(
            {item.wind_level_index for item in data_items if item.wind_level_index is not None}
        ),
        pdf_target_wind_level_count=2 if profile == "pilot" else 4,
        wind_level_shortfall=max(
            0,
            (2 if profile == "pilot" else 4)
            - len(
                {
                    item.wind_level_index
                    for item in data_items
                    if item.wind_level_index is not None
                }
            ),
        ),
        beta_blocks=len({item.beta_block_index for item in items}),
        alpha_fixture_slots=len(
            {item.alpha_fixture_slot for item in items if item.alpha_fixture_slot is not None}
        ),
        neutral_data_segments=neutral,
        doptimal_selection_slots=selected,
        total_data_segments=len(data_items),
        beta_setup_actions=sum(item.kind == Phase6AKind.BETA_SETUP for item in items),
        pose_tare_actions=sum(item.kind == Phase6AKind.POSE_TARE for item in items),
        known_measurement_seconds=sum(item.duration_s or 0.0 for item in data_items),
        pdf_5x5_neutral_segments=pdf_neutral,
        pdf_5x5_total_segments=pdf_total,
        count_changed_from_pdf_by_angle_grid=len(data_items) != pdf_total,
        roll_fixed_deg=ROLL_FIXED_DEG,
        dynamic_attitude_test_included=False,
        executable_segments=sum(item.executable for item in data_items),
        blockers=tuple(
            sorted({item.blocker for item in items if item.blocker is not None})
        ),
    )


@dataclass(frozen=True, slots=True)
class Phase6BSegment:
    sequence: int
    segment_id: str
    protocol_id: str
    protocol_version: str
    dynamic_axis: str
    wind_level_index: int
    wind_speed_mps: float | None
    collective_level_index: int
    waveform_slot: int
    waveform: str | None
    repeat: int
    roll_fixed_deg: float = ROLL_FIXED_DEG
    non_excited_axis_fixed: bool = True
    differential_sign_reversal_required: bool = True
    neutral_coast_required: bool = True
    executable: bool = False
    blocker: str = "wind_collective_waveform_and_safety_limits_are_unresolved"
    wind_sensor_mode: str = "freestream_only"
    near_frame_probe_must_be_removed: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class Phase6BSummary:
    protocol_id: str
    protocol_version: str
    profile: str
    dynamic_axes: tuple[str, ...]
    roll_fixed_deg: float
    configured_wind_speeds_mps: tuple[float, ...]
    wind_level_count: int
    pdf_target_wind_level_count: int
    wind_level_shortfall: int
    run_count: int
    pdf_three_axis_run_count: int
    three_axis_run_count_at_configured_winds: int
    run_count_reduction_from_roll_constraint: int
    run_count_reduction_from_wind_shortfall_at_two_axes: int
    executable_runs: int

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["dynamic_axes"] = list(self.dynamic_axes)
        row["configured_wind_speeds_mps"] = list(self.configured_wind_speeds_mps)
        return row


def build_phase6b_plan(
    profile: str, wind_speeds_mps: tuple[float, ...] | None = None
) -> list[Phase6BSegment]:
    if profile not in {"minimum", "recommended"}:
        raise ValueError(f"unknown Phase 6B profile: {profile}")
    collective_count = 3 if profile == "minimum" else 4
    waveform_count = 3 if profile == "minimum" else 4
    if wind_speeds_mps is not None:
        if not wind_speeds_mps or any(value <= 0 for value in wind_speeds_mps):
            raise ValueError("Phase 6 wind speeds must be non-empty and positive")
        if len(set(wind_speeds_mps)) != len(wind_speeds_mps):
            raise ValueError("Phase 6 wind speeds must be unique")
        winds: tuple[float | None, ...] = wind_speeds_mps
    else:
        winds = (None, None, None, None)
    segments: list[Phase6BSegment] = []
    for axis in DYNAMIC_EXCITATION_AXES:
        for wind_index, wind_speed in enumerate(winds, start=1):
            for collective_index in range(1, collective_count + 1):
                for waveform_slot in range(1, waveform_count + 1):
                    for repeat in (1, 2):
                        sequence = len(segments) + 1
                        segments.append(
                            Phase6BSegment(
                                sequence=sequence,
                                segment_id=f"{PROTOCOL_6B_ID}-{sequence:04d}",
                                protocol_id=PROTOCOL_6B_ID,
                                protocol_version=PROTOCOL_VERSION,
                                dynamic_axis=axis,
                                wind_level_index=wind_index,
                                wind_speed_mps=wind_speed,
                                collective_level_index=collective_index,
                                waveform_slot=waveform_slot,
                                waveform=None,
                                repeat=repeat,
                            )
                        )
    return segments


def summarize_phase6b(
    profile: str, segments: Iterable[Phase6BSegment]
) -> Phase6BSummary:
    items = list(segments)
    pdf_count = 216 if profile == "minimum" else 384
    wind_count = len({item.wind_level_index for item in items})
    collective_count = 3 if profile == "minimum" else 4
    waveform_count = 3 if profile == "minimum" else 4
    three_axis_same_winds = 3 * wind_count * collective_count * waveform_count * 2
    two_axis_pdf_winds = 2 * 4 * collective_count * waveform_count * 2
    return Phase6BSummary(
        protocol_id=PROTOCOL_6B_ID,
        protocol_version=PROTOCOL_VERSION,
        profile=profile,
        dynamic_axes=DYNAMIC_EXCITATION_AXES,
        roll_fixed_deg=ROLL_FIXED_DEG,
        configured_wind_speeds_mps=tuple(
            sorted(
                {
                    item.wind_speed_mps
                    for item in items
                    if item.wind_speed_mps is not None
                }
            )
        ),
        wind_level_count=wind_count,
        pdf_target_wind_level_count=4,
        wind_level_shortfall=max(
            0, 4 - wind_count
        ),
        run_count=len(items),
        pdf_three_axis_run_count=pdf_count,
        three_axis_run_count_at_configured_winds=three_axis_same_winds,
        run_count_reduction_from_roll_constraint=three_axis_same_winds - len(items),
        run_count_reduction_from_wind_shortfall_at_two_axes=(
            two_axis_pdf_winds - len(items)
        ),
        executable_runs=sum(item.executable for item in items),
    )
