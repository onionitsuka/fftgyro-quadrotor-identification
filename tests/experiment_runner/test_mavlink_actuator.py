import math
import unittest

from fftgyro_quad_id.experiment_runner.mavlink_actuator import (
    MAV_CMD_ACTUATOR_TEST,
    MavResult,
    actuator_test_command,
    motor_output_function,
    schedule_motor_hold,
    stop_motor_command,
    validate_command_ack,
)


class MavlinkActuatorTests(unittest.TestCase):
    def test_motor_function_is_not_a_physical_pin_number(self) -> None:
        self.assertEqual(motor_output_function(1), 1)
        self.assertEqual(motor_output_function(4), 4)
        with self.assertRaises(ValueError):
            motor_output_function(0)

    def test_actuator_command_uses_standard_parameter_positions(self) -> None:
        command = actuator_test_command(
            target_system=1,
            motor_number=3,
            value=0.2,
            timeout_s=1.0,
        )
        self.assertEqual(command.command, MAV_CMD_ACTUATOR_TEST)
        self.assertEqual(command.params[0], 0.2)
        self.assertEqual(command.params[1], 1.0)
        self.assertEqual(command.params[4], 3.0)

    def test_stop_uses_nan_and_immediate_restore(self) -> None:
        command = stop_motor_command(target_system=1, motor_number=1)
        self.assertTrue(math.isnan(command.params[0]))
        self.assertEqual(command.params[1], 0.0)

    def test_five_second_hold_refreshes_before_timeout_then_stops(self) -> None:
        commands = schedule_motor_hold(
            target_system=1,
            motor_number=1,
            value=0.1,
            duration_s=5.0,
        )
        self.assertEqual([item.offset_s for item in commands[:-1]], [
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            2.5,
            3.0,
            3.5,
            4.0,
            4.5,
        ])
        self.assertEqual(commands[-1].offset_s, 5.0)
        self.assertTrue(math.isnan(commands[-1].command.params[0]))

    def test_refresh_must_be_faster_than_expiry(self) -> None:
        with self.assertRaises(ValueError):
            schedule_motor_hold(
                target_system=1,
                motor_number=1,
                value=0.1,
                duration_s=5.0,
                command_timeout_s=1.0,
                refresh_interval_s=1.0,
            )

    def test_only_accepted_ack_passes(self) -> None:
        validate_command_ack(MAV_CMD_ACTUATOR_TEST, MavResult.ACCEPTED)
        with self.assertRaises(RuntimeError):
            validate_command_ack(
                MAV_CMD_ACTUATOR_TEST,
                MavResult.TEMPORARILY_REJECTED,
            )


if __name__ == "__main__":
    unittest.main()

