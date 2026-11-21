from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from .collective_mapping import build_collective_mapping
from .flight_stack_readiness import FlightStackStage, evaluate_flight_stack
from .lr8450_readiness import Lr8450Stage, evaluate_lr8450
from .power_supply_readiness import SupplyStage, evaluate_supply
from .wind_sensor_readiness import WindSensorStage, evaluate_wind_sensors


@dataclass(frozen=True, slots=True)
class HardwareCheck:
    check_id: str
    state: str
    configuration: str
    blockers: tuple[str, ...]
    next_command: str | None

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["blockers"] = list(self.blockers)
        return row


@dataclass(frozen=True, slots=True)
class HardwareStatus:
    ready: bool
    checks: tuple[HardwareCheck, ...]
    next_check_id: str | None
    next_action: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "checks": [item.to_dict() for item in self.checks],
            "next_check_id": self.next_check_id,
            "next_action": self.next_action,
        }


def _blockers(result: Any) -> tuple[str, ...]:
    return tuple(str(value) for value in getattr(result, "blockers", ()))


def evaluate_hardware_status(
    config_dir: Path, *, deferred: tuple[str, ...] = ()
) -> HardwareStatus:
    deferred_set = set(deferred)
    unknown = deferred_set - {item[0] for item in _definitions(config_dir)}
    if unknown:
        raise ValueError(f"unknown deferred checks: {','.join(sorted(unknown))}")

    checks: list[HardwareCheck] = []
    for check_id, path, evaluator, inspect_command, template in _definitions(config_dir):
        relative = str(path)
        if check_id in deferred_set:
            checks.append(
                HardwareCheck(
                    check_id=check_id,
                    state="deferred",
                    configuration=relative,
                    blockers=(),
                    next_command=None,
                )
            )
            continue

        if not path.is_file():
            checks.append(
                HardwareCheck(
                    check_id=check_id,
                    state="missing",
                    configuration=relative,
                    blockers=("configuration_missing",),
                    next_command=f"cp {template} {path}",
                )
            )
            continue

        try:
            result = evaluator(path)
            result_blockers = _blockers(result)
            ready = bool(getattr(result, "ready", False))
        except (OSError, ValueError, KeyError) as exc:
            ready = False
            result_blockers = (f"configuration_error:{exc}",)
        checks.append(
            HardwareCheck(
                check_id=check_id,
                state="ready" if ready else "blocked",
                configuration=relative,
                blockers=result_blockers,
                next_command=None if ready else inspect_command,
            )
        )

    next_check = next(
        (item for item in checks if item.state in {"missing", "blocked"}), None
    )
    return HardwareStatus(
        ready=next_check is None,
        checks=tuple(checks),
        next_check_id=next_check.check_id if next_check else None,
        next_action=next_check.next_command if next_check else None,
    )


def _definitions(
    config_dir: Path,
) -> tuple[tuple[str, Path, Callable[[Path], Any], str, str], ...]:
    return (
        (
            "power-supply",
            config_dir / "power-supply.toml",
            lambda path: evaluate_supply(path, SupplyStage.IDENTIFY),
            "fftgyro-exp supply-readiness configs/experiment_runner/power-supply.toml --stage identify",
            "configs/experiment_runner/power-supply.example.toml",
        ),
        (
            "flight-stack",
            config_dir / "flight-stack.toml",
            lambda path: evaluate_flight_stack(path, FlightStackStage.IDENTIFY),
            "fftgyro-exp flight-stack-readiness configs/experiment_runner/flight-stack.toml --stage identify",
            "configs/experiment_runner/flight-stack.example.toml",
        ),
        (
            "lr8450",
            config_dir / "lr8450.toml",
            lambda path: evaluate_lr8450(path, Lr8450Stage.IDENTIFY),
            "fftgyro-exp lr8450-readiness configs/experiment_runner/lr8450.toml --stage identify",
            "configs/experiment_runner/lr8450.example.toml",
        ),
        (
            "wind-sensors",
            config_dir / "wind-sensors.toml",
            lambda path: evaluate_wind_sensors(path, WindSensorStage.IDENTIFY),
            "fftgyro-exp wind-sensor-readiness configs/experiment_runner/wind-sensors.toml --stage identify",
            "configs/experiment_runner/wind-sensors.example.toml",
        ),
        (
            "collective-mapping",
            config_dir / "collective-mapping.toml",
            build_collective_mapping,
            "fftgyro-exp collective-map configs/experiment_runner/collective-mapping.toml",
            "configs/experiment_runner/collective-mapping.example.toml",
        ),
    )
