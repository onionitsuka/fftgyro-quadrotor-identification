from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fftgyro_quad_id.experiment_runner.lr8450_readiness import (
    Lr8450Stage,
    evaluate_lr8450,
)


COMPLETE = """
[instrument]
model = "LR8450"
serial_number = "SERIAL"
firmware_version = "3.01"
modules = ["U8550"]

[recording]
interval_ms = 1.0
required_interval_max_ms = 2.0
local_storage_enabled = true
storage_destination = "SD"

[remote]
interface = "LAN"
address = "192.0.2.10"
communication_commands_enabled = true
logger_utility_enabled = false
delimiter = "LF"

[verification]
manual_start_stop_verified = true
saved_file_readback_verified = true
power_cycle_preserves_configuration = true
remote_identity_verified = true
remote_start_stop_verified = true
disconnect_keeps_local_recording = true

[[channel]]
channel_id = "CH1"
quantity = "voltage_v"
input_mode = "DC voltage"
unit = "V"

[[channel]]
channel_id = "CH2"
quantity = "current_a"
input_mode = "DC voltage"
unit = "A"
transducer_model = "example"
scale_a_per_unit = 10.0

[[channel]]
channel_id = "CH3"
quantity = "motor_temp_c"
input_mode = "thermocouple"
unit = "degC"
sensor_type = "K"

[[channel]]
channel_id = "CH4"
quantity = "esc_temp_c"
input_mode = "thermocouple"
unit = "degC"
sensor_type = "K"

[[channel]]
channel_id = "CH5"
quantity = "sync"
input_mode = "logic"
unit = "logic"
electrical_level_verified = true
"""


class Lr8450ReadinessTests(unittest.TestCase):
    def _evaluate(self, text: str, stage: Lr8450Stage):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "lr8450.toml"
            path.write_text(text, encoding="utf-8")
            return evaluate_lr8450(path, stage)

    def test_complete_remote_configuration_passes(self) -> None:
        self.assertTrue(self._evaluate(COMPLETE, Lr8450Stage.REMOTE_RECORD).ready)

    def test_input_module_is_required(self) -> None:
        result = self._evaluate(
            COMPLETE.replace('modules = ["U8550"]', "modules = []"),
            Lr8450Stage.IDENTIFY,
        )
        self.assertIn("installed_input_modules_unknown", result.blockers)

    def test_all_pdf_quantities_are_required(self) -> None:
        text = COMPLETE.replace('quantity = "sync"', 'quantity = "unused"')
        result = self._evaluate(text, Lr8450Stage.IDENTIFY)
        self.assertIn("required_channel_missing:sync", result.blockers)

    def test_slow_interval_blocks_dynamic_protocol(self) -> None:
        result = self._evaluate(
            COMPLETE.replace("interval_ms = 1.0", "interval_ms = 5.0"),
            Lr8450Stage.MANUAL_RECORD,
        )
        self.assertIn(
            "recording_interval_slower_than_protocol_requirement", result.blockers
        )

    def test_remote_requires_local_recording_survive_disconnect(self) -> None:
        result = self._evaluate(
            COMPLETE.replace(
                "disconnect_keeps_local_recording = true",
                "disconnect_keeps_local_recording = false",
            ),
            Lr8450Stage.REMOTE_RECORD,
        )
        self.assertIn(
            "disconnect_local_recording_behavior_not_verified", result.blockers
        )
