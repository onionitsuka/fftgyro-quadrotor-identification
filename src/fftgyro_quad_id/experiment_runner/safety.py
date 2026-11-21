from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class GateDecision(StrEnum):
    PASS = "pass"
    STOP = "stop"


@dataclass(frozen=True, slots=True)
class SafetyLimits:
    loadcell_utilization_max: float = 0.80
    voltage_drift_fraction_max: float = 0.02
    telemetry_missing_fraction_max: float = 0.01
    rpm_reference_error_fraction_max: float = 0.02
    sync_residual_s_max: float = 0.001
    motor_temperature_warning_c: float | None = None
    esc_temperature_warning_c: float | None = None


@dataclass(frozen=True, slots=True)
class SafetyObservation:
    emergency_stop: bool = False
    loadcell_utilization: float = 0.0
    baseline_voltage_v: float | None = None
    voltage_v: float | None = None
    telemetry_missing_fraction: float | None = None
    rpm_reference_error_fraction: float | None = None
    sync_residual_s: float | None = None
    motor_temperature_c: float | None = None
    esc_temperature_c: float | None = None


@dataclass(frozen=True, slots=True)
class GateResult:
    decision: GateDecision
    reasons: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return self.decision == GateDecision.PASS


def evaluate_safety(
    observation: SafetyObservation,
    limits: SafetyLimits,
) -> GateResult:
    reasons: list[str] = []

    if observation.emergency_stop:
        reasons.append("emergency_stop")
    if observation.loadcell_utilization >= limits.loadcell_utilization_max:
        reasons.append("loadcell_at_or_above_80_percent")

    if (
        observation.baseline_voltage_v is not None
        and observation.voltage_v is not None
    ):
        if observation.baseline_voltage_v <= 0:
            reasons.append("invalid_baseline_voltage")
        else:
            voltage_drift = abs(
                observation.voltage_v - observation.baseline_voltage_v
            ) / observation.baseline_voltage_v
            if voltage_drift > limits.voltage_drift_fraction_max:
                reasons.append("voltage_drift_above_limit")

    if (
        observation.telemetry_missing_fraction is not None
        and observation.telemetry_missing_fraction
        > limits.telemetry_missing_fraction_max
    ):
        reasons.append("telemetry_missing_above_1_percent")

    if (
        observation.rpm_reference_error_fraction is not None
        and observation.rpm_reference_error_fraction
        > limits.rpm_reference_error_fraction_max
    ):
        reasons.append("rpm_reference_error_above_2_percent")

    if (
        observation.sync_residual_s is not None
        and observation.sync_residual_s > limits.sync_residual_s_max
    ):
        reasons.append("sync_residual_above_1_ms")

    if (
        limits.motor_temperature_warning_c is not None
        and observation.motor_temperature_c is not None
        and observation.motor_temperature_c
        >= limits.motor_temperature_warning_c
    ):
        reasons.append("motor_temperature_at_or_above_warning")

    if (
        limits.esc_temperature_warning_c is not None
        and observation.esc_temperature_c is not None
        and observation.esc_temperature_c >= limits.esc_temperature_warning_c
    ):
        reasons.append("esc_temperature_at_or_above_warning")

    return GateResult(
        decision=GateDecision.STOP if reasons else GateDecision.PASS,
        reasons=tuple(reasons),
    )

