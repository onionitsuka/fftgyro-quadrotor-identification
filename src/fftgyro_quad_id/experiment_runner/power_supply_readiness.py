from __future__ import annotations

import tomllib
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any


class SupplyStage(StrEnum):
    IDENTIFY = "identify"
    MANUAL_RUN = "manual-run"
    REMOTE_RUN = "remote-run"


@dataclass(frozen=True, slots=True)
class SupplyReadinessResult:
    stage: SupplyStage
    ready: bool
    blockers: tuple[str, ...]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage.value,
            "ready": self.ready,
            "blockers": list(self.blockers),
            "warnings": list(self.warnings),
        }


def _positive_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0


def evaluate_supply(path: Path, stage: SupplyStage) -> SupplyReadinessResult:
    with path.open("rb") as stream:
        document = tomllib.load(stream)

    supply = document.get("supply", {})
    limits = document.get("experiment_limits", {})
    serial = document.get("rs232", {})
    verification = document.get("verification", {})
    blockers: list[str] = []
    warnings: list[str] = []

    manufacturer = str(supply.get("manufacturer", "")).strip().upper()
    model = str(supply.get("model", "")).strip()
    if manufacturer != "KIKUSUI":
        blockers.append("manufacturer_not_confirmed_as_kikusui")
    if not model:
        blockers.append("supply_model_unknown")

    rated_voltage = supply.get("rated_voltage_v")
    rated_current = supply.get("rated_current_a")
    if not _positive_number(rated_voltage):
        blockers.append("rated_voltage_unknown_or_invalid")
    if not _positive_number(rated_current):
        blockers.append("rated_current_unknown_or_invalid")

    if stage != SupplyStage.IDENTIFY:
        voltage_max = limits.get("voltage_max_v")
        current_max = limits.get("current_max_a")
        if not _positive_number(voltage_max):
            blockers.append("experiment_voltage_limit_unknown_or_invalid")
        if not _positive_number(current_max):
            blockers.append("experiment_current_limit_unknown_or_invalid")
        if _positive_number(rated_voltage) and _positive_number(voltage_max):
            if voltage_max > rated_voltage:
                blockers.append("experiment_voltage_limit_exceeds_supply_rating")
        if _positive_number(rated_current) and _positive_number(current_max):
            if current_max > rated_current:
                blockers.append("experiment_current_limit_exceeds_supply_rating")
        if not bool(verification.get("propellers_removed", False)):
            blockers.append("propellers_removed_not_confirmed")
        if not bool(verification.get("output_off_verified", False)):
            blockers.append("physical_output_off_not_verified")
        if not bool(verification.get("current_limit_verified", False)):
            blockers.append("physical_current_limit_not_verified")
        if not str(verification.get("emergency_power_removal", "")).strip():
            blockers.append("emergency_power_removal_not_recorded")

    if stage == SupplyStage.REMOTE_RUN:
        if not str(serial.get("port", "")).strip():
            blockers.append("rs232_port_unknown")
        if not _positive_number(serial.get("baud")):
            blockers.append("rs232_baud_unknown_or_invalid")
        if serial.get("data_bits") not in (7, 8):
            blockers.append("rs232_data_bits_unknown_or_invalid")
        if str(serial.get("parity", "")).upper() not in {"N", "E", "O"}:
            blockers.append("rs232_parity_unknown_or_invalid")
        if serial.get("stop_bits") not in (1, 2):
            blockers.append("rs232_stop_bits_unknown_or_invalid")
        if not str(serial.get("flow_control", "")).strip():
            blockers.append("rs232_flow_control_unknown")
        if not str(serial.get("cable_wiring", "")).strip():
            blockers.append("rs232_cable_wiring_unknown")
        if not str(serial.get("command_profile", "")).strip():
            blockers.append("model_specific_command_profile_missing")
        if not bool(verification.get("remote_identity_verified", False)):
            blockers.append("remote_identity_query_not_verified")
        if not bool(verification.get("remote_readback_verified", False)):
            blockers.append("remote_voltage_current_readback_not_verified")
        if not bool(verification.get("remote_output_off_verified", False)):
            blockers.append("remote_output_off_not_verified")
        if not bool(verification.get("disconnect_fails_safe", False)):
            blockers.append("rs232_disconnect_fail_safe_not_verified")

    if stage != SupplyStage.REMOTE_RUN and serial:
        warnings.append("rs232_present_but_remote_control_not_enabled_for_this_stage")

    return SupplyReadinessResult(
        stage=stage,
        ready=not blockers,
        blockers=tuple(blockers),
        warnings=tuple(warnings),
    )
