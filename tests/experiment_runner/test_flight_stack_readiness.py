from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fftgyro_quad_id.experiment_runner.flight_stack_readiness import (
    FlightStackStage,
    evaluate_flight_stack,
)


ARDUPILOT = """
[autopilot]
board = "Pixhawk 6X"
stack = "ardupilot_copter"
version = "4.6.1"
firmware_target = "Pixhawk6X"

[outputs]
motor_output_channels = [1, 2, 3, 4]
connector_group = "MAIN"
actuator_protocol = "dshot300"

[verification]
heartbeat_identity_verified = true
parameter_backup_file = "pixhawk6x.param"
output_mapping_verified = true
output_protocol_verified = true
motor_test_command = "MAV_CMD_DO_MOTOR_TEST"
propellers_removed = true
command_ack_accepted = true
timeout_stop_verified = true
"""


class FlightStackReadinessTests(unittest.TestCase):
    def _evaluate(self, text: str, stage: FlightStackStage):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "flight-stack.toml"
            path.write_text(text, encoding="utf-8")
            return evaluate_flight_stack(path, stage)

    def test_ardupilot_requires_motor_test_command(self) -> None:
        result = self._evaluate(ARDUPILOT, FlightStackStage.MOTOR_TEST)
        self.assertTrue(result.ready)
        self.assertEqual(result.required_motor_test_command, "MAV_CMD_DO_MOTOR_TEST")

    def test_px4_requires_actuator_test_command(self) -> None:
        text = (
            ARDUPILOT.replace('stack = "ardupilot_copter"', 'stack = "px4"')
            .replace('version = "4.6.1"', 'version = "1.16.0"')
            .replace('firmware_target = "Pixhawk6X"', 'firmware_target = "px4_fmu-v6x_default"')
            .replace('motor_test_command = "MAV_CMD_DO_MOTOR_TEST"', 'motor_test_command = "MAV_CMD_ACTUATOR_TEST"')
        )
        result = self._evaluate(text, FlightStackStage.MOTOR_TEST)
        self.assertTrue(result.ready)
        self.assertEqual(result.required_motor_test_command, "MAV_CMD_ACTUATOR_TEST")

    def test_unknown_stack_blocks_identification(self) -> None:
        result = self._evaluate(
            ARDUPILOT.replace('stack = "ardupilot_copter"', 'stack = ""'),
            FlightStackStage.IDENTIFY,
        )
        self.assertIn(
            "flight_stack_must_be_ardupilot_copter_or_px4", result.blockers
        )

    def test_wrong_command_is_rejected(self) -> None:
        result = self._evaluate(
            ARDUPILOT.replace(
                'motor_test_command = "MAV_CMD_DO_MOTOR_TEST"',
                'motor_test_command = "MAV_CMD_ACTUATOR_TEST"',
            ),
            FlightStackStage.MOTOR_TEST,
        )
        self.assertIn(
            "motor_test_command_does_not_match_flight_stack", result.blockers
        )
