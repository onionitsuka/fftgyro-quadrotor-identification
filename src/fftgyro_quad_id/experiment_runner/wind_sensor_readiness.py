from __future__ import annotations

import tomllib
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any


ALLOWED_OUTPUT_TYPES = frozenset(
    {"voltage", "current", "pulse", "rs232", "rs485", "usb", "ethernet"}
)
REQUIRED_ROLES = frozenset({"freestream", "near_frame"})


class WindSensorStage(StrEnum):
    IDENTIFY = "identify"
    FRAME_DIAGNOSTIC = "frame-diagnostic"
    MAIN_IDENTIFICATION = "main-identification"


@dataclass(frozen=True, slots=True)
class WindSensorReadinessResult:
    stage: WindSensorStage
    ready: bool
    blockers: tuple[str, ...]
    warnings: tuple[str, ...]
    sensor_ids: tuple[str, ...]
    roles: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage.value,
            "ready": self.ready,
            "blockers": list(self.blockers),
            "warnings": list(self.warnings),
            "sensor_ids": list(self.sensor_ids),
            "roles": list(self.roles),
        }


def _text(value: Any) -> str:
    return str(value or "").strip()


def evaluate_wind_sensors(
    path: Path, stage: WindSensorStage
) -> WindSensorReadinessResult:
    with path.open("rb") as stream:
        document = tomllib.load(stream)

    sensors = document.get("sensor", [])
    acquisition = document.get("acquisition", {})
    placement = document.get("placement", {})
    verification = document.get("verification", {})
    blockers: list[str] = []
    warnings: list[str] = []
    sensor_ids: list[str] = []
    serials: list[str] = []
    channels: list[str] = []
    roles: list[str] = []

    if len(sensors) != 2:
        blockers.append("exactly_two_wind_sensors_required")

    for index, raw in enumerate(sensors, start=1):
        sensor_id = _text(raw.get("sensor_id"))
        prefix = sensor_id or f"sensor_{index}"
        model = _text(raw.get("model"))
        serial = _text(raw.get("serial_number"))
        output_type = _text(raw.get("output_type")).lower()
        channel = _text(raw.get("acquisition_channel"))
        role = _text(raw.get("role")).lower()
        sensor_ids.append(sensor_id)
        serials.append(serial)
        channels.append(channel)
        roles.append(role)

        if not sensor_id:
            blockers.append(f"{prefix}:sensor_id_missing")
        if not model:
            blockers.append(f"{prefix}:model_missing")
        if not serial:
            blockers.append(f"{prefix}:serial_number_missing")
        if output_type not in ALLOWED_OUTPUT_TYPES:
            blockers.append(f"{prefix}:output_type_missing_or_unsupported")
        if not channel:
            blockers.append(f"{prefix}:acquisition_channel_missing")
        if role not in REQUIRED_ROLES:
            blockers.append(f"{prefix}:role_must_be_freestream_or_near_frame")

        if stage != WindSensorStage.IDENTIFY:
            if not _text(raw.get("calibration_evidence")):
                blockers.append(f"{prefix}:calibration_evidence_missing")
            if not bool(raw.get("scale_verified", False)):
                blockers.append(f"{prefix}:scale_not_verified")
            if not bool(raw.get("zero_check_verified", False)):
                blockers.append(f"{prefix}:zero_check_not_verified")

    for values, blocker in (
        (sensor_ids, "sensor_ids_missing_or_not_unique"),
        (serials, "sensor_serials_missing_or_not_unique"),
        (channels, "acquisition_channels_missing_or_not_unique"),
    ):
        if any(not value for value in values) or len(set(values)) != len(values):
            blockers.append(blocker)

    missing_roles = REQUIRED_ROLES - set(roles)
    for role in sorted(missing_roles):
        blockers.append(f"required_sensor_role_missing:{role}")

    if stage != WindSensorStage.IDENTIFY:
        if not _text(acquisition.get("logger")):
            blockers.append("acquisition_logger_missing")
        if not bool(acquisition.get("simultaneous_sampling_verified", False)):
            blockers.append("simultaneous_sampling_not_verified")
        if not bool(placement.get("freestream_position_documented", False)):
            blockers.append("freestream_position_not_documented")
        if not bool(
            placement.get("freestream_outside_apparatus_disturbance_confirmed", False)
        ):
            blockers.append("freestream_position_not_confirmed_undisturbed")

    if stage == WindSensorStage.FRAME_DIAGNOSTIC:
        if not bool(placement.get("near_frame_position_documented", False)):
            blockers.append("near_frame_position_not_documented")
        if not bool(placement.get("near_probe_support_documented", False)):
            blockers.append("near_probe_support_not_documented")
        if not bool(verification.get("probe_installed_removed_pair_planned", False)):
            blockers.append("probe_installed_removed_pair_not_planned")

    if stage == WindSensorStage.MAIN_IDENTIFICATION:
        if not bool(verification.get("near_probe_removed_verified", False)):
            blockers.append("near_probe_removal_not_verified")
        if bool(verification.get("near_probe_installed", False)):
            blockers.append("near_probe_must_not_be_installed_for_main_identification")

    if len(set(_text(raw.get("model")) for raw in sensors)) > 1:
        warnings.append("sensor_models_differ_cross_comparison_required")

    return WindSensorReadinessResult(
        stage=stage,
        ready=not blockers,
        blockers=tuple(dict.fromkeys(blockers)),
        warnings=tuple(warnings),
        sensor_ids=tuple(value for value in sensor_ids if value),
        roles=tuple(value for value in roles if value),
    )
