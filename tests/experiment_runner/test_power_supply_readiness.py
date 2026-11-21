from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fftgyro_quad_id.experiment_runner.power_supply_readiness import (
    SupplyStage,
    evaluate_supply,
)


BASE = """
[supply]
manufacturer = "KIKUSUI"
model = "PWR-EXAMPLE"
rated_voltage_v = 30.0
rated_current_a = 20.0

[experiment_limits]
voltage_max_v = 16.7
current_max_a = 10.0

[rs232]
port = "/dev/ttyUSB0"
baud = 9600
data_bits = 8
parity = "N"
stop_bits = 1
flow_control = "none"
cable_wiring = "model-manual-verified"
command_profile = "pwr-example-v1"

[verification]
propellers_removed = true
output_off_verified = true
current_limit_verified = true
emergency_power_removal = "front-panel OUTPUT key and upstream disconnect"
remote_identity_verified = true
remote_readback_verified = true
remote_output_off_verified = true
disconnect_fails_safe = true
"""


class PowerSupplyReadinessTests(unittest.TestCase):
    def _evaluate(self, text: str, stage: SupplyStage):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "supply.toml"
            path.write_text(text, encoding="utf-8")
            return evaluate_supply(path, stage)

    def test_complete_remote_configuration_passes(self) -> None:
        self.assertTrue(self._evaluate(BASE, SupplyStage.REMOTE_RUN).ready)

    def test_unknown_model_blocks_even_identification(self) -> None:
        result = self._evaluate(BASE.replace('model = "PWR-EXAMPLE"', 'model = ""'), SupplyStage.IDENTIFY)
        self.assertIn("supply_model_unknown", result.blockers)

    def test_manual_run_does_not_require_rs232(self) -> None:
        before_rs232 = BASE.split("[rs232]", 1)[0]
        verification = BASE.split("[verification]", 1)[1]
        text = before_rs232 + "[verification]" + verification
        self.assertTrue(self._evaluate(text, SupplyStage.MANUAL_RUN).ready)

    def test_remote_run_requires_verified_disconnect_behavior(self) -> None:
        result = self._evaluate(
            BASE.replace("disconnect_fails_safe = true", "disconnect_fails_safe = false"),
            SupplyStage.REMOTE_RUN,
        )
        self.assertIn("rs232_disconnect_fail_safe_not_verified", result.blockers)

    def test_limits_must_not_exceed_nameplate(self) -> None:
        result = self._evaluate(
            BASE.replace("voltage_max_v = 16.7", "voltage_max_v = 31.0"),
            SupplyStage.MANUAL_RUN,
        )
        self.assertIn("experiment_voltage_limit_exceeds_supply_rating", result.blockers)
