from __future__ import annotations

import csv
import math
import tomllib
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


REQUIRED_COLUMNS = frozenset(
    {"motor_id", "voltage_v", "command", "thrust_n", "rpm", "current_a"}
)


@dataclass(frozen=True, slots=True)
class MotorOperatingPoint:
    motor_id: str
    command: float
    estimated_rpm: float
    estimated_current_a: float
    target_thrust_n: float


@dataclass(frozen=True, slots=True)
class CollectiveLevel:
    hover_percent: float
    total_target_thrust_n: float
    estimated_total_current_a: float
    motor_points: tuple[MotorOperatingPoint, ...]

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["motor_points"] = [asdict(point) for point in self.motor_points]
        return row


@dataclass(frozen=True, slots=True)
class CollectiveMappingResult:
    ready: bool
    vehicle_mass_kg: float
    operation_voltage_v: float
    current_limit_a: float
    levels: tuple[CollectiveLevel, ...]
    blockers: tuple[str, ...]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "vehicle_mass_kg": self.vehicle_mass_kg,
            "operation_voltage_v": self.operation_voltage_v,
            "current_limit_a": self.current_limit_a,
            "levels": [level.to_dict() for level in self.levels],
            "blockers": list(self.blockers),
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True, slots=True)
class _Point:
    command: float
    thrust_n: float
    rpm: float
    current_a: float


def _float(raw: str, field: str, line: int) -> float:
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"line {line}: invalid {field}") from exc
    if not math.isfinite(value):
        raise ValueError(f"line {line}: non-finite {field}")
    return value


def _load_curve(
    csv_path: Path,
    *,
    operation_voltage_v: float,
    voltage_tolerance_v: float,
) -> dict[str, list[_Point]]:
    grouped: dict[tuple[str, float], list[tuple[float, float, float]]] = defaultdict(list)
    with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or ())
        missing = REQUIRED_COLUMNS - fields
        if missing:
            raise ValueError(f"missing CSV columns: {','.join(sorted(missing))}")
        for line, row in enumerate(reader, start=2):
            motor_id = str(row["motor_id"]).strip()
            voltage = _float(row["voltage_v"], "voltage_v", line)
            if abs(voltage - operation_voltage_v) > voltage_tolerance_v:
                continue
            command = _float(row["command"], "command", line)
            thrust = _float(row["thrust_n"], "thrust_n", line)
            rpm = _float(row["rpm"], "rpm", line)
            current = _float(row["current_a"], "current_a", line)
            if not motor_id:
                raise ValueError(f"line {line}: motor_id missing")
            if not 0.0 <= command <= 1.0:
                raise ValueError(f"line {line}: command outside [0,1]")
            if min(thrust, rpm, current) < 0:
                raise ValueError(f"line {line}: negative measurement")
            grouped[(motor_id, command)].append((thrust, rpm, current))

    curves: dict[str, list[_Point]] = defaultdict(list)
    for (motor_id, command), samples in grouped.items():
        count = len(samples)
        curves[motor_id].append(
            _Point(
                command=command,
                thrust_n=sum(item[0] for item in samples) / count,
                rpm=sum(item[1] for item in samples) / count,
                current_a=sum(item[2] for item in samples) / count,
            )
        )
    for points in curves.values():
        points.sort(key=lambda item: item.command)
    return dict(curves)


def _interpolate(points: list[_Point], target_thrust_n: float) -> _Point | None:
    for lower, upper in zip(points, points[1:]):
        if lower.thrust_n <= target_thrust_n <= upper.thrust_n:
            span = upper.thrust_n - lower.thrust_n
            if span <= 0:
                continue
            fraction = (target_thrust_n - lower.thrust_n) / span
            return _Point(
                command=lower.command + fraction * (upper.command - lower.command),
                thrust_n=target_thrust_n,
                rpm=lower.rpm + fraction * (upper.rpm - lower.rpm),
                current_a=lower.current_a + fraction * (upper.current_a - lower.current_a),
            )
    return None


def build_collective_mapping(configuration_path: Path) -> CollectiveMappingResult:
    with configuration_path.open("rb") as stream:
        document = tomllib.load(stream)
    vehicle = document.get("vehicle", {})
    operation = document.get("operation", {})
    data = document.get("data", {})
    blockers: list[str] = []
    warnings: list[str] = []

    mass = float(vehicle.get("mass_kg", 0.0))
    gravity = float(vehicle.get("gravity_mps2", 9.80665))
    voltage = float(operation.get("voltage_v", 0.0))
    current_limit = float(operation.get("current_limit_a", 0.0))
    max_command = float(operation.get("max_command", 1.0))
    motor_ids = tuple(str(value).strip() for value in operation.get("motor_ids", []))
    percentages = tuple(float(value) for value in operation.get("hover_percentages", []))
    curve_file = str(data.get("steady_curve_csv", "")).strip()
    tolerance = float(data.get("voltage_tolerance_v", 0.1))

    if mass <= 0:
        blockers.append("vehicle_mass_kg_missing")
    if gravity <= 0:
        blockers.append("gravity_mps2_invalid")
    if voltage <= 0:
        blockers.append("operation_voltage_v_missing")
    if current_limit <= 0:
        blockers.append("current_limit_a_missing")
    if not 0 < max_command <= 1:
        blockers.append("max_command_outside_(0,1]")
    if len(motor_ids) != 4 or len(set(motor_ids)) != 4 or any(not item for item in motor_ids):
        blockers.append("exactly_4_unique_motor_ids_required")
    if not percentages or any(value <= 0 for value in percentages):
        blockers.append("positive_hover_percentages_required")
    if not curve_file:
        blockers.append("steady_curve_csv_missing")

    csv_path = configuration_path.parent / curve_file if curve_file else None
    curves: dict[str, list[_Point]] = {}
    if csv_path is not None:
        if not csv_path.is_file():
            blockers.append("steady_curve_csv_not_found")
        else:
            try:
                curves = _load_curve(
                    csv_path,
                    operation_voltage_v=voltage,
                    voltage_tolerance_v=tolerance,
                )
            except (OSError, ValueError) as exc:
                blockers.append(f"steady_curve_csv_error:{exc}")

    for motor_id in motor_ids:
        points = curves.get(motor_id, [])
        if len(points) < 3:
            blockers.append(f"{motor_id}:fewer_than_3_curve_points_at_operation_voltage")
            continue
        if any(
            upper.thrust_n <= lower.thrust_n
            for lower, upper in zip(points, points[1:])
        ):
            blockers.append(f"{motor_id}:thrust_curve_not_strictly_increasing")

    levels: list[CollectiveLevel] = []
    if not blockers:
        hover_total_thrust = mass * gravity
        for percentage in percentages:
            target_per_motor = hover_total_thrust * percentage / 100.0 / 4.0
            motor_points: list[MotorOperatingPoint] = []
            for motor_id in motor_ids:
                point = _interpolate(curves[motor_id], target_per_motor)
                if point is None:
                    blockers.append(
                        f"{motor_id}:{percentage:g}pct_target_outside_measured_thrust_range"
                    )
                    continue
                if point.command > max_command:
                    blockers.append(
                        f"{motor_id}:{percentage:g}pct_command_exceeds_max_command"
                    )
                motor_points.append(
                    MotorOperatingPoint(
                        motor_id=motor_id,
                        command=point.command,
                        estimated_rpm=point.rpm,
                        estimated_current_a=point.current_a,
                        target_thrust_n=target_per_motor,
                    )
                )
            if len(motor_points) == 4:
                total_current = sum(point.estimated_current_a for point in motor_points)
                if total_current > current_limit:
                    blockers.append(
                        f"{percentage:g}pct_estimated_current_exceeds_limit"
                    )
                levels.append(
                    CollectiveLevel(
                        hover_percent=percentage,
                        total_target_thrust_n=hover_total_thrust * percentage / 100.0,
                        estimated_total_current_a=total_current,
                        motor_points=tuple(motor_points),
                    )
                )

    if curves and len({len(points) for points in curves.values()}) > 1:
        warnings.append("motor_curve_point_counts_differ")
    return CollectiveMappingResult(
        ready=not blockers,
        vehicle_mass_kg=mass,
        operation_voltage_v=voltage,
        current_limit_a=current_limit,
        levels=tuple(levels),
        blockers=tuple(blockers),
        warnings=tuple(warnings),
    )
