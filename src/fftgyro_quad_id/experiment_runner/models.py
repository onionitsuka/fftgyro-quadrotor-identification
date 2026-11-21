from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any


class SegmentKind(StrEnum):
    STEADY = "steady"
    TRANSIENT = "transient"
    SYNC = "sync"


@dataclass(frozen=True, slots=True)
class ExperimentSegment:
    sequence: int
    segment_id: str
    protocol_id: str
    protocol_version: str
    kind: SegmentKind
    purpose: str
    motor_id: str
    voltage_v: float
    repeat: int
    input_start: float
    input_end: float
    pre_hold_s: float
    post_hold_s: float
    analysis_window_s: float | None = None
    direction: str | None = None
    method_id: str = "unspecified"
    condition_id: str = "unspecified"

    @property
    def duration_s(self) -> float:
        return self.pre_hold_s + self.post_hold_s

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["kind"] = self.kind.value
        row["duration_s"] = self.duration_s
        return row


@dataclass(frozen=True, slots=True)
class PlanSummary:
    protocol_id: str
    protocol_version: str
    profile: str
    segment_count: int
    steady_segments: int
    dynamic_up_transitions: int
    controlled_down_transitions: int
    sync_segments: int
    max_up_step: float
    max_down_step: float
    pure_recording_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
