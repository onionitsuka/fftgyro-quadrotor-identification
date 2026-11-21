from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum


MAV_CMD_ACTUATOR_TEST = 310
MAV_COMP_ID_AUTOPILOT1 = 1


class MavResult(IntEnum):
    ACCEPTED = 0
    TEMPORARILY_REJECTED = 1
    DENIED = 2
    UNSUPPORTED = 3
    FAILED = 4
    IN_PROGRESS = 5
    CANCELLED = 6


@dataclass(frozen=True, slots=True)
class CommandLong:
    target_system: int
    target_component: int
    command: int
    confirmation: int
    params: tuple[float, float, float, float, float, float, float]


@dataclass(frozen=True, slots=True)
class ScheduledCommand:
    offset_s: float
    command: CommandLong


def motor_output_function(motor_number: int) -> int:
    """MAVLink ACTUATOR_OUTPUT_FUNCTION_MOTOR1..MOTOR16 are 1..16."""
    if not 1 <= motor_number <= 16:
        raise ValueError("motor_number must be in [1, 16]")
    return motor_number


def actuator_test_command(
    *,
    target_system: int,
    motor_number: int,
    value: float,
    timeout_s: float,
) -> CommandLong:
    if not 1 <= target_system <= 255:
        raise ValueError("target_system must be in [1, 255]")
    if not math.isnan(value) and not 0.0 <= value <= 1.0:
        raise ValueError("motor test value must be NaN or in [0, 1]")
    if not 0.0 <= timeout_s <= 3.0:
        raise ValueError("actuator test timeout must be in [0, 3] seconds")
    return CommandLong(
        target_system=target_system,
        target_component=MAV_COMP_ID_AUTOPILOT1,
        command=MAV_CMD_ACTUATOR_TEST,
        confirmation=0,
        params=(
            value,
            timeout_s,
            0.0,
            0.0,
            float(motor_output_function(motor_number)),
            0.0,
            0.0,
        ),
    )


def stop_motor_command(*, target_system: int, motor_number: int) -> CommandLong:
    # MAV_CMD_ACTUATOR_TEST defines NaN as disarmed/stop for a motor function.
    return actuator_test_command(
        target_system=target_system,
        motor_number=motor_number,
        value=math.nan,
        timeout_s=0.0,
    )


def schedule_motor_hold(
    *,
    target_system: int,
    motor_number: int,
    value: float,
    duration_s: float,
    command_timeout_s: float = 1.0,
    refresh_interval_s: float = 0.5,
) -> list[ScheduledCommand]:
    """Build expiring refreshes followed by an explicit stop command."""
    if duration_s <= 0:
        raise ValueError("duration_s must be positive")
    if command_timeout_s <= 0 or command_timeout_s > 3.0:
        raise ValueError("command_timeout_s must be in (0, 3]")
    if refresh_interval_s <= 0:
        raise ValueError("refresh_interval_s must be positive")
    if refresh_interval_s >= command_timeout_s:
        raise ValueError("refresh interval must be shorter than command timeout")

    hold = actuator_test_command(
        target_system=target_system,
        motor_number=motor_number,
        value=value,
        timeout_s=command_timeout_s,
    )
    commands: list[ScheduledCommand] = []
    offset = 0.0
    while offset < duration_s:
        commands.append(ScheduledCommand(offset_s=round(offset, 9), command=hold))
        offset += refresh_interval_s
    commands.append(
        ScheduledCommand(
            offset_s=duration_s,
            command=stop_motor_command(
                target_system=target_system,
                motor_number=motor_number,
            ),
        )
    )
    return commands


def validate_command_ack(command: int, result: int) -> None:
    if command != MAV_CMD_ACTUATOR_TEST:
        raise RuntimeError(f"unexpected COMMAND_ACK for command {command}")
    try:
        parsed = MavResult(result)
    except ValueError as exc:
        raise RuntimeError(f"unknown MAV_RESULT {result}") from exc
    if parsed != MavResult.ACCEPTED:
        raise RuntimeError(f"actuator test rejected: {parsed.name}")

