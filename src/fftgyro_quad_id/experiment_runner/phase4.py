from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from itertools import product
from typing import Any, Iterable


PROTOCOL_ID = "phase4_physical_benchmark"
PROTOCOL_VERSION = "pdf-2026-07-11-v1"
WIND_SPEEDS_MPS = (0.0, 4.0, 6.0, 8.0, 10.0, 12.0)
RIG_ANGLES_DEG = (-20.0, 0.0, 20.0)
BODY_ANGLES_DEG = (-40.0, -20.0, 0.0, 20.0, 40.0)
AXES = ("roll", "pitch", "yaw")
NO_WIND_INITIAL_ANGLES_DEG = (5.0, 10.0, 20.0)


class Phase4Kind(StrEnum):
    STATIC = "static"
    FREE_DECAY = "free_decay"


@dataclass(frozen=True, slots=True)
class Phase4Segment:
    sequence: int
    segment_id: str
    protocol_id: str
    protocol_version: str
    kind: Phase4Kind
    subtest: str
    repeat: int
    settle_s: float
    record_s: float
    wind_speed_mps: float | None = None
    wind_level_index: int | None = None
    alpha_nominal_deg: float | None = None
    beta_nominal_deg: float | None = None
    fixture_id: str | None = None
    propeller_state: str | None = None
    axis: str | None = None
    initial_angle_deg: float | None = None
    initial_angle_level_index: int | None = None
    torque_off: bool = True
    requires_operator_value: bool = False
    wind_sensor_mode: str = "freestream_only"
    near_frame_probe_must_be_removed: bool = True
    use_for_sitl_identification: bool = True

    @property
    def duration_s(self) -> float:
        return self.settle_s + self.record_s

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["kind"] = self.kind.value
        row["duration_s"] = self.duration_s
        return row


@dataclass(frozen=True, slots=True)
class Phase4Summary:
    protocol_id: str
    protocol_version: str
    profile: str
    segment_count: int
    rig_static_segments: int
    no_wind_free_decay_runs: int
    wind_free_decay_runs: int
    body_static_segments: int
    unresolved_segments: int
    pure_recording_seconds: float
    pdf_summary_rig_static_segments: int
    detailed_rig_static_matches_pdf_summary: bool
    source_discrepancies: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["source_discrepancies"] = list(self.source_discrepancies)
        return row


def _minimum_body_poses() -> tuple[tuple[float, float], ...]:
    poses = {(0.0, beta) for beta in (-20.0, 0.0, 20.0)}
    poses.update((alpha, 0.0) for alpha in BODY_ANGLES_DEG)
    return tuple(sorted(poses))


def build_plan(profile: str) -> list[Phase4Segment]:
    if profile not in {"minimum", "recommended"}:
        raise ValueError(f"unknown Phase 4 profile: {profile}")

    segments: list[Phase4Segment] = []

    def add(**kwargs: Any) -> None:
        sequence = len(segments) + 1
        segments.append(
            Phase4Segment(
                sequence=sequence,
                segment_id=f"{PROTOCOL_ID}-{sequence:04d}",
                protocol_id=PROTOCOL_ID,
                protocol_version=PROTOCOL_VERSION,
                **kwargs,
            )
        )

    # Detailed Phase 4.1: 6 winds x 3 alpha x 3 beta x 3 repeats = 162.
    for wind, alpha, beta, repeat in product(
        WIND_SPEEDS_MPS, RIG_ANGLES_DEG, RIG_ANGLES_DEG, range(1, 4)
    ):
        add(
            kind=Phase4Kind.STATIC,
            subtest="rig_static_wind",
            repeat=repeat,
            settle_s=10.0,
            record_s=10.0,
            wind_speed_mps=wind,
            alpha_nominal_deg=alpha,
            beta_nominal_deg=beta,
            # Every alpha fixture needs its real ID and measured angle.
            fixture_id=None,
            requires_operator_value=True,
        )

    # Detailed Phase 4.2 plus the MuJoCo gate fixes the no-wind angles to 5/10/20 deg.
    for axis, angle, repeat in product(
        AXES, NO_WIND_INITIAL_ANGLES_DEG, range(1, 6)
    ):
        add(
            kind=Phase4Kind.FREE_DECAY,
            subtest="rig_free_decay_no_wind",
            repeat=repeat,
            settle_s=0.0,
            record_s=30.0,
            wind_speed_mps=0.0,
            axis=axis,
            initial_angle_deg=angle,
        )

    if profile == "recommended":
        # The PDF gives counts but not the actual four winds or two initial angles.
        for wind_level, axis, angle_level, repeat in product(
            range(1, 5), AXES, range(1, 3), range(1, 4)
        ):
            add(
                kind=Phase4Kind.FREE_DECAY,
                subtest="rig_free_decay_wind",
                repeat=repeat,
                settle_s=0.0,
                record_s=30.0,
                wind_speed_mps=None,
                wind_level_index=wind_level,
                axis=axis,
                initial_angle_deg=None,
                initial_angle_level_index=angle_level,
                requires_operator_value=True,
            )

    if profile == "minimum":
        body_poses = _minimum_body_poses()
    else:
        body_poses = tuple(product(BODY_ANGLES_DEG, BODY_ANGLES_DEG))
    for wind, (alpha, beta), propeller_state, repeat in product(
        WIND_SPEEDS_MPS, body_poses, ("removed", "installed_stopped"), range(1, 3)
    ):
        add(
            kind=Phase4Kind.STATIC,
            subtest="rotor_stopped_body_wind",
            repeat=repeat,
            settle_s=10.0,
            record_s=10.0,
            wind_speed_mps=wind,
            alpha_nominal_deg=alpha,
            beta_nominal_deg=beta,
            fixture_id=None,
            propeller_state=propeller_state,
            requires_operator_value=True,
        )
    return segments


def summarize(profile: str, segments: Iterable[Phase4Segment]) -> Phase4Summary:
    items = list(segments)
    rig_static = sum(item.subtest == "rig_static_wind" for item in items)
    pdf_summary_rig_static = 450
    discrepancies: list[str] = []
    if rig_static != pdf_summary_rig_static:
        discrepancies.append(
            "phase4_detail_defines_162_rig_static_segments_but_time_table_states_450"
        )
    return Phase4Summary(
        protocol_id=PROTOCOL_ID,
        protocol_version=PROTOCOL_VERSION,
        profile=profile,
        segment_count=len(items),
        rig_static_segments=rig_static,
        no_wind_free_decay_runs=sum(
            item.subtest == "rig_free_decay_no_wind" for item in items
        ),
        wind_free_decay_runs=sum(
            item.subtest == "rig_free_decay_wind" for item in items
        ),
        body_static_segments=sum(
            item.subtest == "rotor_stopped_body_wind" for item in items
        ),
        unresolved_segments=sum(item.requires_operator_value for item in items),
        pure_recording_seconds=sum(item.duration_s for item in items),
        pdf_summary_rig_static_segments=pdf_summary_rig_static,
        detailed_rig_static_matches_pdf_summary=rig_static == pdf_summary_rig_static,
        source_discrepancies=tuple(discrepancies),
    )
