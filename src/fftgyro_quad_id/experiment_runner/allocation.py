from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


PROTOCOL_ID = "allocation_no_wind"
PROTOCOL_VERSION = "pdf-2026-07-11-v1"
COLLECTIVE_LEVELS = (0.30, 0.45, 0.60, 0.75)


@dataclass(frozen=True, slots=True)
class DifferencePattern:
    pattern_id: str
    p: int
    q: int
    r: int


PATTERNS = (
    DifferencePattern("neutral", 0, 0, 0),
    DifferencePattern("plus_p", 1, 0, 0),
    DifferencePattern("minus_p", -1, 0, 0),
    DifferencePattern("plus_q", 0, 1, 0),
    DifferencePattern("minus_q", 0, -1, 0),
    DifferencePattern("plus_r", 0, 0, 1),
    DifferencePattern("minus_r", 0, 0, -1),
    DifferencePattern("plus_p_plus_q", 1, 1, 0),
    DifferencePattern("minus_p_minus_q", -1, -1, 0),
    DifferencePattern("plus_p_minus_q", 1, -1, 0),
    DifferencePattern("minus_p_plus_q", -1, 1, 0),
    DifferencePattern("plus_p_plus_r", 1, 0, 1),
    DifferencePattern("minus_p_minus_r", -1, 0, -1),
)


@dataclass(frozen=True, slots=True)
class AllocationSigns:
    sp: tuple[int, int, int, int]
    sq: tuple[int, int, int, int]
    sr: tuple[int, int, int, int]

    def validate(self) -> None:
        axes = (self.sp, self.sq, self.sr)
        for name, axis in zip(("sp", "sq", "sr"), axes, strict=True):
            if len(axis) != 4 or any(value not in {-1, 1} for value in axis):
                raise ValueError(f"{name} must contain exactly four -1/+1 values")
            if sum(axis) != 0:
                raise ValueError(f"{name} must have two -1 and two +1 values")
        collective = (1, 1, 1, 1)
        vectors = (collective, *axes)
        for index, left in enumerate(vectors):
            for right in vectors[index + 1 :]:
                if sum(a * b for a, b in zip(left, right, strict=True)) != 0:
                    raise ValueError("collective/sp/sq/sr must be mutually orthogonal")


@dataclass(frozen=True, slots=True)
class AllocationSegment:
    sequence: int
    segment_id: str
    protocol_id: str
    protocol_version: str
    collective: float
    delta: float
    pattern_id: str
    coefficient_p: int
    coefficient_q: int
    coefficient_r: int
    repeat: int
    settle_s: float = 5.0
    record_s: float = 10.0

    @property
    def duration_s(self) -> float:
        return self.settle_s + self.record_s

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["duration_s"] = self.duration_s
        return row


@dataclass(frozen=True, slots=True)
class AllocationSummary:
    protocol_id: str
    protocol_version: str
    profile: str
    segment_count: int
    pure_recording_seconds: float
    motor_inputs_resolved: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_plan(delta: float = 0.05, repeats: int = 3) -> list[AllocationSegment]:
    if not 0 < delta <= 0.08:
        raise ValueError("delta must be in (0, 0.08]")
    if repeats < 1:
        raise ValueError("repeats must be positive")
    segments: list[AllocationSegment] = []
    for collective in COLLECTIVE_LEVELS:
        for pattern in PATTERNS:
            for repeat in range(1, repeats + 1):
                sequence = len(segments) + 1
                segments.append(
                    AllocationSegment(
                        sequence=sequence,
                        segment_id=f"{PROTOCOL_ID}-{sequence:04d}",
                        protocol_id=PROTOCOL_ID,
                        protocol_version=PROTOCOL_VERSION,
                        collective=collective,
                        delta=delta,
                        pattern_id=pattern.pattern_id,
                        coefficient_p=pattern.p,
                        coefficient_q=pattern.q,
                        coefficient_r=pattern.r,
                        repeat=repeat,
                    )
                )
    return segments


def summarize(segments: list[AllocationSegment]) -> AllocationSummary:
    return AllocationSummary(
        protocol_id=PROTOCOL_ID,
        protocol_version=PROTOCOL_VERSION,
        profile="pdf-recommended",
        segment_count=len(segments),
        pure_recording_seconds=sum(segment.duration_s for segment in segments),
    )


def resolve_motor_inputs(
    segment: AllocationSegment,
    signs: AllocationSigns,
) -> tuple[float, float, float, float]:
    signs.validate()
    values = tuple(
        segment.collective
        + segment.delta
        * (
            segment.coefficient_p * signs.sp[index]
            + segment.coefficient_q * signs.sq[index]
            + segment.coefficient_r * signs.sr[index]
        )
        for index in range(4)
    )
    if any(not 0.0 <= value <= 1.0 for value in values):
        raise ValueError(f"resolved motor input outside [0, 1]: {values}")
    return values

