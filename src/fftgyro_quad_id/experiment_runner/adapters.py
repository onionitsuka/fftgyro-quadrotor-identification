from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from .models import ExperimentSegment
from .safety import SafetyObservation


class ExperimentAdapter(Protocol):
    """Hardware boundary. Implementations must fail safe on process loss."""

    @property
    def permits_physical_output(self) -> bool: ...

    def preflight(self) -> SafetyObservation: ...

    def begin_segment(self, segment: ExperimentSegment) -> None: ...

    def observe(self) -> SafetyObservation: ...

    def end_segment(self, segment: ExperimentSegment) -> None: ...

    def stop_all(self) -> None: ...


@dataclass(slots=True)
class SimulationAdapter:
    """No-I/O adapter used to verify sequencing and safety behavior."""

    observations: list[SafetyObservation] = field(
        default_factory=lambda: [SafetyObservation()]
    )
    events: list[str] = field(default_factory=list)
    _observation_index: int = 0

    @property
    def permits_physical_output(self) -> bool:
        return False

    def preflight(self) -> SafetyObservation:
        self.events.append("preflight")
        return self._next_observation()

    def begin_segment(self, segment: ExperimentSegment) -> None:
        self.events.append(f"begin:{segment.segment_id}")

    def observe(self) -> SafetyObservation:
        self.events.append("observe")
        return self._next_observation()

    def end_segment(self, segment: ExperimentSegment) -> None:
        self.events.append(f"end:{segment.segment_id}")

    def stop_all(self) -> None:
        self.events.append("stop_all")

    def _next_observation(self) -> SafetyObservation:
        if not self.observations:
            return SafetyObservation()
        index = min(self._observation_index, len(self.observations) - 1)
        value = self.observations[index]
        self._observation_index += 1
        return value

