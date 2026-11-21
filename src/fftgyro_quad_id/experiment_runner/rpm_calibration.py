from __future__ import annotations

import csv
import hashlib
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from .rpm_estimation import MODEL_NAME


REQUIRED_COLUMNS = frozenset({"motor_id", "voltage_v", "command", "rpm", "split"})


@dataclass(frozen=True, slots=True)
class _Point:
    motor_id: str
    voltage_v: float
    command: float
    rpm: float
    split: str


@dataclass(frozen=True, slots=True)
class RpmCalibrationResult:
    model: str
    command_linear_per_v: float
    command_quadratic_per_v: float
    calibrated_command_min: float
    calibrated_command_max: float
    train_rmse_rpm: float
    validation_rmse_rpm: float
    validation_max_abs_error_rpm: float
    train_points: int
    validation_points: int
    motor_ids: tuple[str, ...]
    calibration_data_sha256: str

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["motor_ids"] = list(self.motor_ids)
        return result


def _number(raw: object, *, field: str, line: int) -> float:
    try:
        value = float(str(raw))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"line_{line}:invalid_{field}") from exc
    if not math.isfinite(value):
        raise ValueError(f"line_{line}:nonfinite_{field}")
    return value


def _read_points(path: Path) -> tuple[list[_Point], str]:
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    points: list[_Point] = []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or ())
        if missing:
            raise ValueError("calibration_csv_missing_columns:" + ",".join(sorted(missing)))
        for line, row in enumerate(reader, start=2):
            motor_id = str(row["motor_id"] or "").strip()
            split = str(row["split"] or "").strip().lower()
            voltage = _number(row["voltage_v"], field="voltage_v", line=line)
            command = _number(row["command"], field="command", line=line)
            rpm = _number(row["rpm"], field="rpm", line=line)
            if not motor_id:
                raise ValueError(f"line_{line}:motor_id_missing")
            if split not in {"train", "validation"}:
                raise ValueError(f"line_{line}:split_must_be_train_or_validation")
            if voltage <= 0.0:
                raise ValueError(f"line_{line}:voltage_v_must_be_positive")
            if not 0.0 < command <= 1.0:
                raise ValueError(f"line_{line}:command_outside_(0,1]")
            if rpm <= 0.0:
                raise ValueError(f"line_{line}:rpm_must_be_positive")
            points.append(_Point(motor_id, voltage, command, rpm, split))
    return points, digest


def _predict(point: _Point, a1: float, a2: float) -> float:
    return point.voltage_v * (a1 * point.command + a2 * point.command**2)


def _rmse(points: list[_Point], a1: float, a2: float) -> float:
    return math.sqrt(
        sum((_predict(point, a1, a2) - point.rpm) ** 2 for point in points)
        / len(points)
    )


def calibrate_rpm_command_map(path: Path) -> RpmCalibrationResult:
    points, digest = _read_points(path)
    train = [point for point in points if point.split == "train"]
    validation = [point for point in points if point.split == "validation"]
    if len(train) < 6:
        raise ValueError("fewer_than_6_train_points")
    if len(validation) < 3:
        raise ValueError("fewer_than_3_validation_points")
    if len({point.command for point in train}) < 3:
        raise ValueError("fewer_than_3_distinct_train_commands")

    x1 = [point.voltage_v * point.command for point in train]
    x2 = [point.voltage_v * point.command**2 for point in train]
    y = [point.rpm for point in train]
    s11 = sum(value * value for value in x1)
    s12 = sum(left * right for left, right in zip(x1, x2))
    s22 = sum(value * value for value in x2)
    b1 = sum(left * right for left, right in zip(x1, y))
    b2 = sum(left * right for left, right in zip(x2, y))
    determinant = s11 * s22 - s12 * s12
    if determinant <= 1e-12 * max(s11 * s22, 1.0):
        raise ValueError("rpm_calibration_design_matrix_singular")
    a1 = (b1 * s22 - b2 * s12) / determinant
    a2 = (s11 * b2 - s12 * b1) / determinant

    command_min = min(point.command for point in train)
    command_max = max(point.command for point in train)
    if any(not command_min <= point.command <= command_max for point in validation):
        raise ValueError("validation_command_outside_train_range")
    if a1 <= 0.0:
        raise ValueError("fitted_command_linear_per_v_not_positive")
    if min(a1 + 2.0 * a2 * command_min, a1 + 2.0 * a2 * command_max) <= 0.0:
        raise ValueError("fitted_command_map_not_monotonic_in_train_range")

    validation_errors = [
        _predict(point, a1, a2) - point.rpm for point in validation
    ]
    return RpmCalibrationResult(
        model=MODEL_NAME,
        command_linear_per_v=a1,
        command_quadratic_per_v=a2,
        calibrated_command_min=command_min,
        calibrated_command_max=command_max,
        train_rmse_rpm=_rmse(train, a1, a2),
        validation_rmse_rpm=_rmse(validation, a1, a2),
        validation_max_abs_error_rpm=max(abs(error) for error in validation_errors),
        train_points=len(train),
        validation_points=len(validation),
        motor_ids=tuple(sorted({point.motor_id for point in points})),
        calibration_data_sha256=digest,
    )
