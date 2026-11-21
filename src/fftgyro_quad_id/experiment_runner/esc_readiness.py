from __future__ import annotations

import tomllib
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any


SUPPORTED_BLUEJAY_MCUS = frozenset({"BB21", "BB51"})


class ReadinessStage(StrEnum):
    INSPECT = "inspect"
    FLASH_ONE = "flash-one"
    DEPLOY_ALL = "deploy-all"


@dataclass(frozen=True, slots=True)
class EscRecord:
    esc_id: str
    motor_id: str
    part_number: str
    mcu: str
    layout: str
    original_firmware: str
    settings_backup: str
    readout_success: bool


@dataclass(frozen=True, slots=True)
class EscReadinessResult:
    stage: ReadinessStage
    ready: bool
    blockers: tuple[str, ...]
    warnings: tuple[str, ...]
    esc_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage.value,
            "ready": self.ready,
            "blockers": list(self.blockers),
            "warnings": list(self.warnings),
            "esc_count": self.esc_count,
        }


def load_inventory(path: Path) -> tuple[dict[str, Any], list[EscRecord]]:
    with path.open("rb") as stream:
        document = tomllib.load(stream)

    bluejay = document.get("bluejay", {})
    records = []
    for raw in document.get("esc", []):
        records.append(
            EscRecord(
                esc_id=str(raw.get("esc_id", "")).strip(),
                motor_id=str(raw.get("motor_id", "")).strip(),
                part_number=str(raw.get("part_number", "")).strip(),
                mcu=str(raw.get("mcu", "")).strip().upper(),
                layout=str(raw.get("layout", "")).strip().upper(),
                original_firmware=str(raw.get("original_firmware", "")).strip(),
                settings_backup=str(raw.get("settings_backup", "")).strip(),
                readout_success=bool(raw.get("readout_success", False)),
            )
        )
    return bluejay, records


def evaluate_inventory(
    inventory_path: Path,
    stage: ReadinessStage,
) -> EscReadinessResult:
    bluejay, records = load_inventory(inventory_path)
    blockers: list[str] = []
    warnings: list[str] = []

    if len(records) != 4:
        blockers.append(f"expected_4_esc_records:found_{len(records)}")

    ids = [record.esc_id for record in records]
    motor_ids = [record.motor_id for record in records]
    if len(set(ids)) != len(ids) or any(not value for value in ids):
        blockers.append("esc_ids_missing_or_not_unique")
    if len(set(motor_ids)) != len(motor_ids) or any(not value for value in motor_ids):
        blockers.append("motor_ids_missing_or_not_unique")

    for record in records:
        prefix = record.esc_id or "unknown_esc"
        if not record.readout_success:
            blockers.append(f"{prefix}:original_firmware_not_read")
        if not record.mcu:
            blockers.append(f"{prefix}:mcu_unknown")
        elif record.mcu not in SUPPORTED_BLUEJAY_MCUS:
            blockers.append(f"{prefix}:unsupported_mcu:{record.mcu}")
        if not record.layout:
            blockers.append(f"{prefix}:layout_unknown")
        if not record.original_firmware:
            blockers.append(f"{prefix}:original_firmware_unknown")

        if stage != ReadinessStage.INSPECT:
            if not record.settings_backup:
                blockers.append(f"{prefix}:settings_backup_not_recorded")
            else:
                backup_path = inventory_path.parent / record.settings_backup
                if not backup_path.is_file():
                    blockers.append(f"{prefix}:settings_backup_file_missing")

    if stage != ReadinessStage.INSPECT:
        if not str(bluejay.get("programmer", "")).strip():
            blockers.append("programmer_not_selected")
        if not str(bluejay.get("recovery_procedure", "")).strip():
            blockers.append("recovery_procedure_not_recorded")
        if not bool(bluejay.get("propellers_removed", False)):
            blockers.append("propellers_removed_not_confirmed")
        if not bool(bluejay.get("current_limited_supply", False)):
            blockers.append("current_limited_supply_not_confirmed")

    if stage == ReadinessStage.DEPLOY_ALL:
        if not bool(bluejay.get("spare_esc_validation_passed", False)):
            blockers.append("spare_esc_validation_not_passed")
        if not str(bluejay.get("validated_bluejay_version", "")).strip():
            blockers.append("validated_bluejay_version_missing")
        if not str(bluejay.get("validated_target", "")).strip():
            blockers.append("validated_bluejay_target_missing")

    if records and len({record.layout for record in records if record.layout}) > 1:
        warnings.append("layouts_differ_between_escs")
    if records and len({record.original_firmware for record in records}) > 1:
        warnings.append("original_firmware_differs_between_escs")

    return EscReadinessResult(
        stage=stage,
        ready=not blockers,
        blockers=tuple(blockers),
        warnings=tuple(warnings),
        esc_count=len(records),
    )

