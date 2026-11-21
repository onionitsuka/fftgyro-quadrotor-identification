from __future__ import annotations

import tomllib
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any


REQUIRED_QUANTITIES = frozenset(
    {"voltage_v", "current_a", "motor_temp_c", "esc_temp_c", "sync"}
)


class Lr8450Stage(StrEnum):
    IDENTIFY = "identify"
    MANUAL_RECORD = "manual-record"
    REMOTE_RECORD = "remote-record"


@dataclass(frozen=True, slots=True)
class Lr8450ReadinessResult:
    stage: Lr8450Stage
    ready: bool
    blockers: tuple[str, ...]
    warnings: tuple[str, ...]
    configured_quantities: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage.value,
            "ready": self.ready,
            "blockers": list(self.blockers),
            "warnings": list(self.warnings),
            "configured_quantities": list(self.configured_quantities),
        }


def _positive(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0


def evaluate_lr8450(path: Path, stage: Lr8450Stage) -> Lr8450ReadinessResult:
    with path.open("rb") as stream:
        document = tomllib.load(stream)

    instrument = document.get("instrument", {})
    recording = document.get("recording", {})
    remote = document.get("remote", {})
    verification = document.get("verification", {})
    channels = document.get("channel", [])
    blockers: list[str] = []
    warnings: list[str] = []

    model = str(instrument.get("model", "")).strip().upper()
    if model not in {"LR8450", "LR8450-01"}:
        blockers.append("instrument_model_unknown_or_not_lr8450")
    if not str(instrument.get("serial_number", "")).strip():
        blockers.append("instrument_serial_number_missing")
    if not str(instrument.get("firmware_version", "")).strip():
        blockers.append("instrument_firmware_version_missing")
    modules = tuple(str(value).strip().upper() for value in instrument.get("modules", []))
    if not modules or any(not value for value in modules):
        blockers.append("installed_input_modules_unknown")

    quantity_names: list[str] = []
    channel_ids: list[str] = []
    for raw in channels:
        channel_id = str(raw.get("channel_id", "")).strip()
        quantity = str(raw.get("quantity", "")).strip().lower()
        channel_ids.append(channel_id)
        quantity_names.append(quantity)
        prefix = channel_id or "unknown_channel"
        if not channel_id:
            blockers.append("channel_id_missing")
        if not quantity:
            blockers.append(f"{prefix}:quantity_missing")
        if not str(raw.get("input_mode", "")).strip():
            blockers.append(f"{prefix}:input_mode_missing")
        if quantity != "sync" and not str(raw.get("unit", "")).strip():
            blockers.append(f"{prefix}:unit_missing")
        if quantity == "current_a":
            if not str(raw.get("transducer_model", "")).strip():
                blockers.append(f"{prefix}:current_transducer_model_missing")
            if not _positive(raw.get("scale_a_per_unit")):
                blockers.append(f"{prefix}:current_scale_missing_or_invalid")
        if quantity in {"motor_temp_c", "esc_temp_c"}:
            if not str(raw.get("sensor_type", "")).strip():
                blockers.append(f"{prefix}:temperature_sensor_type_missing")
        if quantity == "sync" and not bool(raw.get("electrical_level_verified", False)):
            blockers.append(f"{prefix}:sync_electrical_level_not_verified")

    if len(set(channel_ids)) != len(channel_ids) or any(not value for value in channel_ids):
        blockers.append("channel_ids_missing_or_not_unique")
    missing_quantities = REQUIRED_QUANTITIES - set(quantity_names)
    for quantity in sorted(missing_quantities):
        blockers.append(f"required_channel_missing:{quantity}")

    if stage != Lr8450Stage.IDENTIFY:
        interval_ms = recording.get("interval_ms")
        required_max_ms = recording.get("required_interval_max_ms")
        if not _positive(interval_ms):
            blockers.append("recording_interval_missing_or_invalid")
        if not _positive(required_max_ms):
            blockers.append("required_interval_limit_missing_or_invalid")
        if _positive(interval_ms) and _positive(required_max_ms):
            if interval_ms > required_max_ms:
                blockers.append("recording_interval_slower_than_protocol_requirement")
        if not bool(recording.get("local_storage_enabled", False)):
            blockers.append("instrument_local_storage_not_enabled")
        if not str(recording.get("storage_destination", "")).strip():
            blockers.append("local_storage_destination_missing")
        if not bool(verification.get("manual_start_stop_verified", False)):
            blockers.append("manual_start_stop_not_verified")
        if not bool(verification.get("saved_file_readback_verified", False)):
            blockers.append("saved_file_readback_not_verified")
        if not bool(verification.get("power_cycle_preserves_configuration", False)):
            blockers.append("configuration_persistence_not_verified")

    if stage == Lr8450Stage.REMOTE_RECORD:
        interface = str(remote.get("interface", "")).strip().upper()
        if interface not in {"LAN", "USB"}:
            blockers.append("remote_interface_must_be_lan_or_usb")
        if not str(remote.get("address", "")).strip():
            blockers.append("remote_address_missing")
        if not bool(remote.get("communication_commands_enabled", False)):
            blockers.append("communication_commands_not_enabled")
        if bool(remote.get("logger_utility_enabled", False)):
            blockers.append("logger_utility_must_be_disabled_in_command_mode")
        if str(remote.get("delimiter", "")).upper() not in {"LF", "CR+LF"}:
            blockers.append("command_delimiter_unknown")
        if not bool(verification.get("remote_identity_verified", False)):
            blockers.append("remote_identity_not_verified")
        if not bool(verification.get("remote_start_stop_verified", False)):
            blockers.append("remote_start_stop_not_verified")
        if not bool(verification.get("disconnect_keeps_local_recording", False)):
            blockers.append("disconnect_local_recording_behavior_not_verified")

    if stage != Lr8450Stage.REMOTE_RECORD and remote:
        warnings.append("remote_settings_present_but_not_required_for_this_stage")

    return Lr8450ReadinessResult(
        stage=stage,
        ready=not blockers,
        blockers=tuple(blockers),
        warnings=tuple(warnings),
        configured_quantities=tuple(sorted(set(quantity_names) - {""})),
    )
