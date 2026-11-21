from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .adapters import ExperimentAdapter
from .models import ExperimentSegment
from .safety import GateResult, SafetyLimits, evaluate_safety


class RunStatus(StrEnum):
    COMPLETED = "completed"
    STOPPED = "stopped"


@dataclass(frozen=True, slots=True)
class RunResult:
    status: RunStatus
    completed_segments: int
    stopped_before_segment: str | None = None
    gate_result: GateResult | None = None


class ExperimentRunner:
    def __init__(
        self,
        adapter: ExperimentAdapter,
        limits: SafetyLimits | None = None,
    ) -> None:
        self.adapter = adapter
        self.limits = limits or SafetyLimits()

    def run_simulation(self, segments: list[ExperimentSegment]) -> RunResult:
        """Validate the state machine without sleeping or physical output."""
        if self.adapter.permits_physical_output:
            raise RuntimeError("run_simulation refuses a physical-output adapter")

        preflight_gate = evaluate_safety(self.adapter.preflight(), self.limits)
        if not preflight_gate.passed:
            self.adapter.stop_all()
            return RunResult(
                status=RunStatus.STOPPED,
                completed_segments=0,
                stopped_before_segment=segments[0].segment_id if segments else None,
                gate_result=preflight_gate,
            )

        completed = 0
        try:
            for segment in segments:
                gate = evaluate_safety(self.adapter.observe(), self.limits)
                if not gate.passed:
                    return RunResult(
                        status=RunStatus.STOPPED,
                        completed_segments=completed,
                        stopped_before_segment=segment.segment_id,
                        gate_result=gate,
                    )
                self.adapter.begin_segment(segment)
                self.adapter.end_segment(segment)
                completed += 1
        finally:
            self.adapter.stop_all()

        return RunResult(
            status=RunStatus.COMPLETED,
            completed_segments=completed,
        )

