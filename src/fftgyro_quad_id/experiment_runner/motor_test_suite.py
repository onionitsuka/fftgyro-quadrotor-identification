"""Convert the motor-single manifest into safely executable contiguous runs."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from .models import ExperimentSegment


@dataclass(frozen=True, slots=True)
class MotorCommandStep:
    segment_id: str
    sequence: int
    purpose: str
    percent: float
    duration_s: float
    input_start_percent: float | None = None
    analysis_window_s: float | None = None
    direction: str | None = None


@dataclass(frozen=True, slots=True)
class MotorTestRun:
    group: int
    phase: str
    motor: int
    voltage_v: float
    repetition: int
    steps: tuple[MotorCommandStep, ...]
    method_id: str = "unspecified"
    condition_id: str = "unspecified"

    @property
    def duration_s(self) -> float:
        return sum(step.duration_s for step in self.steps)

    @property
    def maximum_percent(self) -> float:
        return max(step.percent for step in self.steps)

    @property
    def segment_range(self) -> tuple[int, int]:
        return self.steps[0].sequence, self.steps[-1].sequence

    @property
    def run_suffix(self) -> str:
        voltage = f"{self.voltage_v:g}".replace(".", "p")
        return f"g{self.group:03d}_{self.phase}_m{self.motor}_v{voltage}_r{self.repetition:02d}"


def _phase(segment: ExperimentSegment) -> str:
    if segment.purpose in {"steady_rpm_map", "controlled_static_return"}:
        return "steady"
    if segment.purpose == "hardware_sync_marker":
        return "sync"
    if segment.purpose in {
        "down_pre_hold",
        "down_step_settle",
        "symmetric_baseline",
        "symmetric_up_settle",
        "symmetric_down_settle",
        "thermal_baseline",
        "thermal_down_settle",
        "custom_baseline",
        "custom_step_settle",
        "settling_approach",
    }:
        return "settling"
    return "dynamic"


def _motor_number(motor_id: str) -> int:
    prefix = "motor_"
    if not motor_id.startswith(prefix):
        raise ValueError(f"invalid motor ID: {motor_id}")
    motor = int(motor_id.removeprefix(prefix))
    if not 1 <= motor <= 16:
        raise ValueError(f"motor number out of range: {motor}")
    return motor


def build_motor_test_runs(
    segments: Iterable[ExperimentSegment],
) -> tuple[MotorTestRun, ...]:
    """Group consecutive manifest segments without changing their order.

    A steady sweep must remain contiguous so its controlled descent starts at
    the preceding command. The same applies to each dynamic waveform.
    """

    groups: list[list[ExperimentSegment]] = []
    keys: list[tuple[str, str, str, str, float, int]] = []
    for segment in segments:
        key = (
            _phase(segment),
            segment.method_id,
            segment.condition_id,
            segment.motor_id,
            segment.voltage_v,
            segment.repeat,
        )
        if not keys or key != keys[-1]:
            keys.append(key)
            groups.append([])
        groups[-1].append(segment)

    runs: list[MotorTestRun] = []
    for group_number, (key, items) in enumerate(zip(keys, groups, strict=True), start=1):
        phase, method_id, condition_id, motor_id, voltage_v, repetition = key
        steps = tuple(
            MotorCommandStep(
                segment_id=item.segment_id,
                sequence=item.sequence,
                purpose=item.purpose,
                percent=round(item.input_end * 100.0, 6),
                duration_s=item.duration_s,
                input_start_percent=round(item.input_start * 100.0, 6),
                analysis_window_s=item.analysis_window_s,
                direction=item.direction,
            )
            for item in items
        )
        if any(step.duration_s <= 0 for step in steps):
            raise ValueError(f"run {group_number} contains a non-positive hold")
        runs.append(
            MotorTestRun(
                group=group_number,
                phase=phase,
                motor=_motor_number(motor_id),
                voltage_v=voltage_v,
                repetition=repetition,
                steps=steps,
                method_id=method_id,
                condition_id=condition_id,
            )
        )
    return tuple(runs)


def select_run_range(
    runs: tuple[MotorTestRun, ...], start: int, stop: int | None
) -> tuple[MotorTestRun, ...]:
    if start < 1:
        raise ValueError("start group must be positive")
    final = len(runs) if stop is None else stop
    if final < start or final > len(runs):
        raise ValueError(f"group range must be within 1..{len(runs)}")
    return tuple(run for run in runs if start <= run.group <= final)
