from __future__ import annotations

import io
import math
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from fftgyro_quad_id.experiment_runner.cli import _parser, main
from fftgyro_quad_id.experiment_runner.rpm_estimation import load_rpm_estimator


def _config(tau: float = 0.1) -> str:
    return f"""
[rpm_source]
model = "voltage_quadratic_command_map_calibrated"
command_linear_per_v = 1000.0
command_quadratic_per_v = -200.0
calibrated_command_min = 0.1
calibrated_command_max = 0.8
calibration_rmse_rpm = 10.0
calibration_data_sha256 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"

[motor_time_constant_s]
motor_1 = {tau}
"""


class RpmEstimationTests(unittest.TestCase):
    def _load(self, config: str | None = None):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "rpm.toml"
            path.write_text(config or _config(), encoding="utf-8")
            return load_rpm_estimator(path)

    def test_setpoint_uses_newly_calibrated_voltage_command_map(self) -> None:
        estimator, readiness = self._load()
        self.assertTrue(readiness.ready)
        estimate = estimator.estimate("motor_1", 0.4, 15.7)
        expected = 15.7 * (1000.0 * 0.4 - 200.0 * 0.4**2)
        self.assertEqual(
            estimate.rpm_source,
            "estimated_from_new_phase1_voltage_command_model",
        )
        self.assertAlmostEqual(estimate.estimated_setpoint_rpm, expected)
        self.assertIsNone(estimate.estimated_dynamic_rpm)

    def test_dynamic_rpm_uses_first_order_lag(self) -> None:
        estimator, readiness = self._load()
        self.assertTrue(readiness.ready)
        estimate = estimator.estimate(
            "motor_1", 0.5, 15.7, initial_rpm=0.0, dt_s=0.1
        )
        setpoint = 15.7 * (1000.0 * 0.5 - 200.0 * 0.5**2)
        self.assertAlmostEqual(
            estimate.estimated_dynamic_rpm,
            setpoint * (1.0 - math.exp(-1.0)),
        )

    def test_voltage_is_required_and_changes_setpoint(self) -> None:
        estimator, _ = self._load()
        low = estimator.estimate("motor_1", 0.5, 14.0)
        high = estimator.estimate("motor_1", 0.5, 16.0)
        self.assertGreater(high.estimated_setpoint_rpm, low.estimated_setpoint_rpm)
        with self.assertRaisesRegex(ValueError, "voltage_v_must_be_positive"):
            estimator.estimate("motor_1", 0.5, 0.0)

    def test_command_outside_calibration_range_is_rejected(self) -> None:
        estimator, _ = self._load()
        with self.assertRaisesRegex(ValueError, "command_outside_calibration_range"):
            estimator.estimate("motor_1", 0.9, 15.7)

    def test_time_constant_is_required(self) -> None:
        _, readiness = self._load(config=_config(tau=0.0))
        self.assertIn("motor_1:time_constant_invalid", readiness.blockers)

    def test_cli_keeps_subcommand_and_motor_command_separate(self) -> None:
        args = _parser().parse_args(
            [
                "rpm-estimate",
                "rpm.toml",
                "--motor",
                "motor_1",
                "--command",
                "0.5",
                "--voltage-v",
                "15.7",
            ]
        )
        self.assertEqual(args.command, "rpm-estimate")
        self.assertEqual(args.rpm_command, 0.5)
        self.assertEqual(args.voltage_v, 15.7)

    def test_cli_estimates_rpm_from_valid_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            configuration = Path(temporary) / "rpm.toml"
            configuration.write_text(_config(), encoding="utf-8")
            output = io.StringIO()
            with patch(
                "sys.argv",
                [
                    "fftgyro-exp",
                    "rpm-estimate",
                    str(configuration),
                    "--motor",
                    "motor_1",
                    "--command",
                    "0.4",
                    "--voltage-v",
                    "15.7",
                ],
            ), redirect_stdout(output):
                main()
        self.assertIn('"estimated_setpoint_rpm":', output.getvalue())
