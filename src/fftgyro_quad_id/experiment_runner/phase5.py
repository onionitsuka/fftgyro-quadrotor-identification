from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any, Iterable

from .angle_grid import PROFILES
from .axis_policy import DYNAMIC_EXCITATION_AXES, ROLL_FIXED_DEG


PROTOCOL_ID = "phase5_no_wind_fixed_pose"
PROTOCOL_VERSION = "pdf-2026-07-11-fixed-beta-v1"
COLLECTIVE_LEVELS_PERCENT = (60.0, 75.0, 90.0, 105.0)


class Phase5Kind(StrEnum):
    BETA_SETUP = "beta_setup"
    POSE_TARE = "pose_tare"
    STATIC_MEASUREMENT = "static_measurement"


@dataclass(frozen=True, slots=True)
class Phase5Segment:
    sequence: int
    segment_id: str
    protocol_id: str
    protocol_version: str
    kind: Phase5Kind
    subtest: str
    beta_block_index: int
    beta_nominal_deg: float
    alpha_fixture_slot: int | None
    alpha_nominal_deg: float | None
    fixture_id: str | None
    repeat: int | None
    collective_percent_hover: float | None
    differential_input: str | None
    settle_s: float | None
    record_s: float | None
    wind_speed_mps: float = 0.0
    beta_dynamic_command_allowed: bool = False
    roll_fixed: bool = True
    roll_fixed_deg: float = ROLL_FIXED_DEG
    pitch_fixed: bool = True
    yaw_fixed: bool = True
    requires_rotors_stopped: bool = False
    requires_operator_confirmation: bool = False
    executable: bool = False
    blocker: str | None = None

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
class Phase5Summary:
    protocol_id: str
    protocol_version: str
    profile: str
    angle_profile: str
    beta_blocks: int
    alpha_fixture_slots: int
    data_segments: int
    beta_setup_actions: int
    pose_tare_actions: int
    known_measurement_seconds: float
    pdf_equivalent_5x5_data_segments: int
    count_changed_from_pdf_by_angle_grid: bool
    all_rotational_dof_mechanically_fixed: bool
    roll_fixed_deg: float
    dynamic_attitude_test_included: bool
    executable_segments: int
    blockers: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["blockers"] = list(self.blockers)
        return row


def build_phase5b_plan(
    profile: str,
    angle_profile: str = "sun-beta45-coarse",
    *,
    alpha_nominal_deg: tuple[float, ...] | None = None,
    fixture_ids: tuple[str, ...] | None = None,
) -> list[Phase5Segment]:
    """Build the fixed 5B portion only; 5C dynamics are deliberately excluded."""
    if profile not in {"pilot", "minimum"}:
        raise ValueError(f"unknown Phase 5B profile: {profile}")
    try:
        angles = PROFILES[angle_profile]
    except KeyError as exc:
        raise ValueError(f"unknown angle-grid profile: {angle_profile}") from exc

    alpha_count = angles.alpha_level_count
    if alpha_nominal_deg is not None and len(alpha_nominal_deg) != alpha_count:
        raise ValueError(f"exactly {alpha_count} measured alpha levels are required")
    if fixture_ids is not None and len(fixture_ids) != alpha_count:
        raise ValueError(f"exactly {alpha_count} fixture IDs are required")

    collective_levels = (90.0,) if profile == "pilot" else COLLECTIVE_LEVELS_PERCENT
    segments: list[Phase5Segment] = []

    def add(**kwargs: Any) -> None:
        sequence = len(segments) + 1
        segments.append(
            Phase5Segment(
                sequence=sequence,
                segment_id=f"{PROTOCOL_ID}-{sequence:05d}",
                protocol_id=PROTOCOL_ID,
                protocol_version=PROTOCOL_VERSION,
                **kwargs,
            )
        )

    alpha_ready = alpha_nominal_deg is not None and fixture_ids is not None
    for beta_index, beta in enumerate(angles.beta_deg, start=1):
        add(
            kind=Phase5Kind.BETA_SETUP,
            subtest="5B_fixed_pose",
            beta_block_index=beta_index,
            beta_nominal_deg=beta,
            alpha_fixture_slot=None,
            alpha_nominal_deg=None,
            fixture_id=None,
            repeat=None,
            collective_percent_hover=None,
            differential_input=None,
            settle_s=None,
            record_s=None,
            requires_rotors_stopped=True,
            requires_operator_confirmation=True,
            blocker="beta_angle_and_clearance_require_operator_confirmation",
        )
        for alpha_index in range(alpha_count):
            alpha = None if alpha_nominal_deg is None else alpha_nominal_deg[alpha_index]
            fixture = None if fixture_ids is None else fixture_ids[alpha_index]
            add(
                kind=Phase5Kind.POSE_TARE,
                subtest="5B_fixed_pose",
                beta_block_index=beta_index,
                beta_nominal_deg=beta,
                alpha_fixture_slot=alpha_index + 1,
                alpha_nominal_deg=alpha,
                fixture_id=fixture,
                repeat=None,
                collective_percent_hover=None,
                differential_input=None,
                settle_s=None,
                record_s=None,
                requires_rotors_stopped=True,
                requires_operator_confirmation=True,
                blocker=(
                    "alpha_fixture_angle_and_tare_duration_are_unresolved"
                    if not alpha_ready
                    else "tare_duration_is_unresolved"
                ),
            )
            for collective in collective_levels:
                for repeat in (1, 2):
                    add(
                        kind=Phase5Kind.STATIC_MEASUREMENT,
                        subtest="5B_fixed_pose",
                        beta_block_index=beta_index,
                        beta_nominal_deg=beta,
                        alpha_fixture_slot=alpha_index + 1,
                        alpha_nominal_deg=alpha,
                        fixture_id=fixture,
                        repeat=repeat,
                        collective_percent_hover=collective,
                        differential_input="neutral",
                        settle_s=10.0,
                        record_s=10.0,
                        blocker=(
                            "beta_clearance_and_tare_are_unresolved"
                            if alpha_ready
                            else "alpha_fixture_angle_and_id_are_unresolved"
                        ),
                        executable=False,
                    )
    return segments


def summarize(
    profile: str,
    angle_profile: str,
    segments: Iterable[Phase5Segment],
) -> Phase5Summary:
    items = list(segments)
    measurements = [item for item in items if item.kind == Phase5Kind.STATIC_MEASUREMENT]
    pdf_equivalent = 50 if profile == "pilot" else 200
    blockers = tuple(sorted({item.blocker for item in items if item.blocker is not None}))
    return Phase5Summary(
        protocol_id=PROTOCOL_ID,
        protocol_version=PROTOCOL_VERSION,
        profile=profile,
        angle_profile=angle_profile,
        beta_blocks=len({item.beta_block_index for item in items}),
        alpha_fixture_slots=len(
            {item.alpha_fixture_slot for item in items if item.alpha_fixture_slot is not None}
        ),
        data_segments=len(measurements),
        beta_setup_actions=sum(item.kind == Phase5Kind.BETA_SETUP for item in items),
        pose_tare_actions=sum(item.kind == Phase5Kind.POSE_TARE for item in items),
        known_measurement_seconds=sum(item.duration_s or 0.0 for item in measurements),
        pdf_equivalent_5x5_data_segments=pdf_equivalent,
        count_changed_from_pdf_by_angle_grid=len(measurements) != pdf_equivalent,
        all_rotational_dof_mechanically_fixed=all(
            item.roll_fixed and item.pitch_fixed and item.yaw_fixed for item in items
        ),
        roll_fixed_deg=ROLL_FIXED_DEG,
        dynamic_attitude_test_included=False,
        executable_segments=sum(item.executable for item in measurements),
        blockers=blockers,
    )


class Phase5CKind(StrEnum):
    ROTOR_STOPPED_FREE_DECAY = "rotor_stopped_free_decay"
    ROTOR_RUNNING_DYNAMIC = "rotor_running_dynamic"


@dataclass(frozen=True, slots=True)
class Phase5CSegment:
    sequence: int
    segment_id: str
    protocol_id: str
    protocol_version: str
    kind: Phase5CKind
    dynamic_axis: str
    initial_angle_level_index: int | None
    initial_angle_deg: float | None
    collective_level_index: int | None
    collective_percent_hover: float | None
    waveform: str | None
    repeat: int
    roll_fixed_deg: float = ROLL_FIXED_DEG
    non_excited_axis_fixed: bool = True
    torque_off: bool = True
    executable: bool = False
    blocker: str = "dynamic_limits_and_physical_condition_values_are_unresolved"

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["kind"] = self.kind.value
        return row


@dataclass(frozen=True, slots=True)
class Phase5CSummary:
    protocol_id: str
    protocol_version: str
    profile: str
    dynamic_axes: tuple[str, ...]
    roll_fixed_deg: float
    rotor_stopped_free_decay_runs: int
    rotor_running_dynamic_runs: int
    total_runs: int
    pdf_three_axis_free_decay_runs: int
    pdf_three_axis_rotor_dynamic_runs: int
    run_count_reduction_from_roll_constraint: int
    executable_runs: int

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["dynamic_axes"] = list(self.dynamic_axes)
        return row


def build_phase5c_plan(profile: str) -> list[Phase5CSegment]:
    if profile not in {"minimum", "recommended"}:
        raise ValueError(f"unknown Phase 5C profile: {profile}")
    free_decay_repeats = 3 if profile == "minimum" else 5
    collective_count = 1 if profile == "minimum" else 3
    waveforms = (
        ("low_frequency_sine", "chirp")
        if profile == "minimum"
        else ("low_frequency_sine", "chirp", "prbs", "small_step")
    )
    segments: list[Phase5CSegment] = []

    def add(**kwargs: Any) -> None:
        sequence = len(segments) + 1
        segments.append(
            Phase5CSegment(
                sequence=sequence,
                segment_id=f"phase5c_no_wind_dynamic-{sequence:04d}",
                protocol_id="phase5c_no_wind_dynamic",
                protocol_version=PROTOCOL_VERSION,
                **kwargs,
            )
        )

    for axis in DYNAMIC_EXCITATION_AXES:
        for angle_level in range(1, 4):
            for repeat in range(1, free_decay_repeats + 1):
                add(
                    kind=Phase5CKind.ROTOR_STOPPED_FREE_DECAY,
                    dynamic_axis=axis,
                    initial_angle_level_index=angle_level,
                    initial_angle_deg=None,
                    collective_level_index=None,
                    collective_percent_hover=None,
                    waveform=None,
                    repeat=repeat,
                )
        for collective_level in range(1, collective_count + 1):
            for waveform in waveforms:
                for repeat in (1, 2):
                    add(
                        kind=Phase5CKind.ROTOR_RUNNING_DYNAMIC,
                        dynamic_axis=axis,
                        initial_angle_level_index=None,
                        initial_angle_deg=None,
                        collective_level_index=collective_level,
                        collective_percent_hover=None,
                        waveform=waveform,
                        repeat=repeat,
                    )
    return segments


def summarize_phase5c(
    profile: str, segments: Iterable[Phase5CSegment]
) -> Phase5CSummary:
    items = list(segments)
    stopped = sum(item.kind == Phase5CKind.ROTOR_STOPPED_FREE_DECAY for item in items)
    running = sum(item.kind == Phase5CKind.ROTOR_RUNNING_DYNAMIC for item in items)
    pdf_stopped = 27 if profile == "minimum" else 45
    # Minimum uses the lower end of the PDF's 12--24 range (one collective).
    pdf_running = 12 if profile == "minimum" else 72
    return Phase5CSummary(
        protocol_id="phase5c_no_wind_dynamic",
        protocol_version=PROTOCOL_VERSION,
        profile=profile,
        dynamic_axes=DYNAMIC_EXCITATION_AXES,
        roll_fixed_deg=ROLL_FIXED_DEG,
        rotor_stopped_free_decay_runs=stopped,
        rotor_running_dynamic_runs=running,
        total_runs=len(items),
        pdf_three_axis_free_decay_runs=pdf_stopped,
        pdf_three_axis_rotor_dynamic_runs=pdf_running,
        run_count_reduction_from_roll_constraint=(pdf_stopped + pdf_running) - len(items),
        executable_runs=sum(item.executable for item in items),
    )
