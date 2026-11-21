from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fftgyro_quad_id.experiment_runner.wind_sensor_readiness import (
    WindSensorStage,
    evaluate_wind_sensors,
)


COMPLETE = """
[acquisition]
logger = "LR8450"
simultaneous_sampling_verified = true

[placement]
freestream_position_documented = true
freestream_outside_apparatus_disturbance_confirmed = true
near_frame_position_documented = true
near_probe_support_documented = true

[verification]
probe_installed_removed_pair_planned = true
near_probe_removed_verified = true
near_probe_installed = false

[[sensor]]
sensor_id = "WS-FREE"
model = "MODEL-A"
serial_number = "SERIAL-A"
output_type = "voltage"
acquisition_channel = "CH10"
role = "freestream"
calibration_evidence = "certificate-a.pdf"
scale_verified = true
zero_check_verified = true

[[sensor]]
sensor_id = "WS-NEAR"
model = "MODEL-A"
serial_number = "SERIAL-B"
output_type = "voltage"
acquisition_channel = "CH11"
role = "near_frame"
calibration_evidence = "certificate-b.pdf"
scale_verified = true
zero_check_verified = true
"""


class WindSensorReadinessTests(unittest.TestCase):
    def _evaluate(self, text: str, stage: WindSensorStage):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "wind-sensors.toml"
            path.write_text(text, encoding="utf-8")
            return evaluate_wind_sensors(path, stage)

    def test_complete_frame_diagnostic_configuration_passes(self) -> None:
        result = self._evaluate(COMPLETE, WindSensorStage.FRAME_DIAGNOSTIC)
        self.assertTrue(result.ready)

    def test_both_roles_are_required(self) -> None:
        text = COMPLETE.replace('role = "near_frame"', 'role = "freestream"')
        result = self._evaluate(text, WindSensorStage.IDENTIFY)
        self.assertIn("required_sensor_role_missing:near_frame", result.blockers)

    def test_missing_calibration_blocks_frame_diagnostic(self) -> None:
        text = COMPLETE.replace('calibration_evidence = "certificate-a.pdf"', 'calibration_evidence = ""')
        result = self._evaluate(text, WindSensorStage.FRAME_DIAGNOSTIC)
        self.assertIn("WS-FREE:calibration_evidence_missing", result.blockers)

    def test_main_identification_requires_near_probe_removed(self) -> None:
        text = COMPLETE.replace("near_probe_removed_verified = true", "near_probe_removed_verified = false")
        result = self._evaluate(text, WindSensorStage.MAIN_IDENTIFICATION)
        self.assertIn("near_probe_removal_not_verified", result.blockers)

    def test_different_models_warn(self) -> None:
        text = COMPLETE.replace('model = "MODEL-A"', 'model = "MODEL-B"', 1)
        result = self._evaluate(text, WindSensorStage.IDENTIFY)
        self.assertIn("sensor_models_differ_cross_comparison_required", result.warnings)
