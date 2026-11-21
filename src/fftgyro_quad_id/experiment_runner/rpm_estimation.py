from __future__ import annotations

import math
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path


MODEL_NAME = "voltage_quadratic_command_map_calibrated"
RPM_SOURCE = "estimated_from_new_phase1_voltage_command_model"


@dataclass(frozen=True, slots=True)
class RpmEstimate:
    motor_id: str
    command: float
    voltage_v: float
    estimated_setpoint_rpm: float
    estimated_dynamic_rpm: float | None
    time_constant_s: float
    calibration_data_sha256: str
    rpm_source: str = RPM_SOURCE

    def to_dict(self) -> dict[str, float | str | None]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RpmEstimatorReadiness:
    ready: bool
    motor_ids: tuple[str, ...]
    blockers: tuple[str, ...]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "ready": self.ready,
            "motor_ids": list(self.motor_ids),
            "rpm_source": RPM_SOURCE,
            "blockers": list(self.blockers),
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True, slots=True)
class RpmEstimator:
    command_linear_per_v: float
    command_quadratic_per_v: float
    calibrated_command_min: float
    calibrated_command_max: float
    calibration_rmse_rpm: float
    calibration_data_sha256: str
    time_constants_s: dict[str, float]

    def setpoint_rpm(self, motor_id: str, command: float, voltage_v: float) -> float:
        if motor_id not in self.time_constants_s:
            raise ValueError(f"unknown_motor_id:{motor_id}")
        if not math.isfinite(command) or not 0.0 <= command <= 1.0:
            raise ValueError("command_outside_[0,1]")
        if command != 0.0 and not (
            self.calibrated_command_min <= command <= self.calibrated_command_max
        ):
            raise ValueError(f"{motor_id}:command_outside_calibration_range")
        if not math.isfinite(voltage_v) or voltage_v <= 0.0:
            raise ValueError("voltage_v_must_be_positive")
        rpm = voltage_v * (
            self.command_linear_per_v * command
            + self.command_quadratic_per_v * command * command
        )
        if rpm < 0.0:
            raise ValueError("command_map_produced_negative_rpm")
        return rpm

    def estimate(
        self,
        motor_id: str,
        command: float,
        voltage_v: float,
        *,
        initial_rpm: float | None = None,
        dt_s: float | None = None,
    ) -> RpmEstimate:
        if (initial_rpm is None) != (dt_s is None):
            raise ValueError("initial_rpm_and_dt_s_must_be_given_together")
        setpoint = self.setpoint_rpm(motor_id, command, voltage_v)
        dynamic: float | None = None
        time_constant = self.time_constants_s[motor_id]
        if initial_rpm is not None and dt_s is not None:
            if initial_rpm < 0:
                raise ValueError("initial_rpm_must_be_nonnegative")
            if dt_s <= 0:
                raise ValueError("dt_s_must_be_positive")
            alpha = math.exp(-dt_s / time_constant)
            dynamic = alpha * initial_rpm + (1.0 - alpha) * setpoint
        return RpmEstimate(
            motor_id=motor_id,
            command=command,
            voltage_v=voltage_v,
            estimated_setpoint_rpm=setpoint,
            estimated_dynamic_rpm=dynamic,
            time_constant_s=time_constant,
            calibration_data_sha256=self.calibration_data_sha256,
        )


def _number(value: object, *, field: str) -> float:
    try:
        number = float(str(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid_{field}") from exc
    if not math.isfinite(number):
        raise ValueError(f"nonfinite_{field}")
    return number


def load_rpm_estimator(path: Path) -> tuple[RpmEstimator, RpmEstimatorReadiness]:
    with path.open("rb") as stream:
        document = tomllib.load(stream)

    source = document.get("rpm_source", {})
    motors = document.get("motor_time_constant_s", {})
    blockers: list[str] = []
    warnings: list[str] = []

    model = str(source.get("model", "")).strip()
    if model != MODEL_NAME:
        blockers.append(f"rpm_model_must_be_{MODEL_NAME}")
    calibration_hash = str(source.get("calibration_data_sha256", "")).strip().lower()
    if len(calibration_hash) != 64 or any(
        character not in "0123456789abcdef" for character in calibration_hash
    ):
        blockers.append("calibration_data_sha256_missing_or_invalid")

    numeric_defaults = {
        "command_linear_per_v": 0.0,
        "command_quadratic_per_v": 0.0,
        "calibrated_command_min": -1.0,
        "calibrated_command_max": -1.0,
        "calibration_rmse_rpm": -1.0,
    }
    values: dict[str, float] = {}
    for field, default in numeric_defaults.items():
        try:
            values[field] = _number(source.get(field, default), field=field)
        except ValueError:
            values[field] = default
            blockers.append(f"{field}_invalid")

    if values["command_linear_per_v"] <= 0.0:
        blockers.append("command_linear_per_v_missing_or_invalid")
    if not math.isfinite(values["command_quadratic_per_v"]):
        blockers.append("command_quadratic_per_v_invalid")
    command_min = values["calibrated_command_min"]
    command_max = values["calibrated_command_max"]
    if not 0.0 <= command_min < command_max <= 1.0:
        blockers.append("calibrated_command_range_invalid")
    elif min(
        values["command_linear_per_v"]
        + 2.0 * values["command_quadratic_per_v"] * command_min,
        values["command_linear_per_v"]
        + 2.0 * values["command_quadratic_per_v"] * command_max,
    ) <= 0.0:
        blockers.append("command_map_not_monotonic_in_calibrated_range")
    if values["calibration_rmse_rpm"] < 0.0:
        blockers.append("calibration_rmse_rpm_missing_or_invalid")

    time_constants: dict[str, float] = {}
    if not isinstance(motors, dict) or not motors:
        blockers.append("motor_time_constants_missing")
    else:
        for motor_id, raw_tau in motors.items():
            motor = str(motor_id).strip()
            try:
                tau = _number(raw_tau, field="time_constant_s")
            except ValueError:
                blockers.append(f"{motor or 'unknown_motor'}:time_constant_invalid")
                continue
            if not motor or tau <= 0.0:
                blockers.append(f"{motor or 'unknown_motor'}:time_constant_invalid")
                continue
            time_constants[motor] = tau

    readiness = RpmEstimatorReadiness(
        ready=not blockers,
        motor_ids=tuple(sorted(time_constants)),
        blockers=tuple(dict.fromkeys(blockers)),
        warnings=tuple(warnings),
    )
    estimator = RpmEstimator(
        command_linear_per_v=values["command_linear_per_v"],
        command_quadratic_per_v=values["command_quadratic_per_v"],
        calibrated_command_min=command_min,
        calibrated_command_max=command_max,
        calibration_rmse_rpm=values["calibration_rmse_rpm"],
        calibration_data_sha256=calibration_hash,
        time_constants_s=time_constants,
    )
    return estimator, readiness
