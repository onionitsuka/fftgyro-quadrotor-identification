from __future__ import annotations

import tomllib
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any


STACK_COMMAND = {
    "ardupilot_copter": "MAV_CMD_DO_MOTOR_TEST",
    "px4": "MAV_CMD_ACTUATOR_TEST",
}


class FlightStackStage(StrEnum):
    IDENTIFY = "identify"
    OUTPUT_CONFIGURE = "output-configure"
    MOTOR_TEST = "motor-test"


@dataclass(frozen=True, slots=True)
class FlightStackReadinessResult:
    stage: FlightStackStage
    ready: bool
    stack: str | None
    version: str | None
    required_motor_test_command: str | None
    blockers: tuple[str, ...]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage.value,
            "ready": self.ready,
            "stack": self.stack,
            "version": self.version,
            "required_motor_test_command": self.required_motor_test_command,
            "blockers": list(self.blockers),
            "warnings": list(self.warnings),
        }


def _text(value: Any) -> str:
    return str(value or "").strip()


def evaluate_flight_stack(
    path: Path, stage: FlightStackStage
) -> FlightStackReadinessResult:
    with path.open("rb") as stream:
        document = tomllib.load(stream)

    autopilot = document.get("autopilot", {})
    outputs = document.get("outputs", {})
    verification = document.get("verification", {})
    blockers: list[str] = []
    warnings: list[str] = []

    board = _text(autopilot.get("board")).lower().replace(" ", "")
    stack = _text(autopilot.get("stack")).lower()
    version = _text(autopilot.get("version"))
    firmware_target = _text(autopilot.get("firmware_target"))

    if board != "pixhawk6x":
        blockers.append("board_not_confirmed_as_pixhawk_6x")
    if stack not in STACK_COMMAND:
        blockers.append("flight_stack_must_be_ardupilot_copter_or_px4")
    if not version:
        blockers.append("flight_stack_version_missing")
    if not firmware_target:
        blockers.append("firmware_target_missing")
    if not bool(verification.get("heartbeat_identity_verified", False)):
        blockers.append("mavlink_heartbeat_identity_not_verified")
    if not _text(verification.get("parameter_backup_file")):
        blockers.append("parameter_backup_file_missing")

    if stage != FlightStackStage.IDENTIFY:
        channels = outputs.get("motor_output_channels", [])
        if (
            not isinstance(channels, list)
            or len(channels) != 4
            or len(set(channels)) != 4
            or any(not isinstance(value, int) or value < 1 for value in channels)
        ):
            blockers.append("exactly_4_unique_positive_motor_output_channels_required")
        if not _text(outputs.get("connector_group")):
            blockers.append("motor_connector_group_missing")
        if str(outputs.get("actuator_protocol", "")).lower() not in {
            "pwm",
            "oneshot",
            "dshot150",
            "dshot300",
            "dshot600",
        }:
            blockers.append("actuator_protocol_not_confirmed")
        for key, blocker in (
            ("output_mapping_verified", "motor_output_mapping_not_verified"),
            ("output_protocol_verified", "actuator_output_protocol_not_verified"),
        ):
            if not bool(verification.get(key, False)):
                blockers.append(blocker)

    required_command = STACK_COMMAND.get(stack)
    if stage == FlightStackStage.MOTOR_TEST:
        configured_command = _text(
            verification.get("motor_test_command")
        ).upper()
        if configured_command != required_command:
            blockers.append("motor_test_command_does_not_match_flight_stack")
        if not bool(verification.get("propellers_removed", False)):
            blockers.append("propellers_removed_not_confirmed")
        if not bool(verification.get("command_ack_accepted", False)):
            blockers.append("motor_test_command_ack_not_verified")
        if not bool(verification.get("timeout_stop_verified", False)):
            blockers.append("motor_test_timeout_stop_not_verified")
        if stack == "ardupilot_copter":
            warnings.append(
                "existing_mavlink_actuator_adapter_is_px4_only_use_ardupilot_motor_test_adapter"
            )

    return FlightStackReadinessResult(
        stage=stage,
        ready=not blockers,
        stack=stack if stack in STACK_COMMAND else None,
        version=version or None,
        required_motor_test_command=required_command,
        blockers=tuple(blockers),
        warnings=tuple(warnings),
    )
