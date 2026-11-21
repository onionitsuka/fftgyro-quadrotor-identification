from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from fftgyro_quad_id.experiment_runner.cli import main
from fftgyro_quad_id.experiment_runner.rpm_calibration import calibrate_rpm_command_map


def _rpm(voltage: float, command: float) -> float:
    return voltage * (1000.0 * command - 200.0 * command**2)


def _csv(*, validation_command: float = 0.5) -> str:
    rows = ["motor_id,voltage_v,command,rpm,split"]
    for index, command in enumerate((0.1, 0.2, 0.3, 0.4, 0.6, 0.8)):
        voltage = 15.0 + 0.1 * index
        rows.append(f"motor_1,{voltage},{command},{_rpm(voltage, command)},train")
    for index, command in enumerate((0.2, validation_command, 0.7)):
        voltage = 15.2 + 0.1 * index
        rows.append(f"motor_1,{voltage},{command},{_rpm(voltage, command)},validation")
    return "\n".join(rows) + "\n"


class RpmCalibrationTests(unittest.TestCase):
    def _calibrate(self, csv_text: str):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "rpm-calibration.csv"
            path.write_text(csv_text, encoding="utf-8")
            return calibrate_rpm_command_map(path)

    def test_fits_voltage_quadratic_map_from_new_measurements(self) -> None:
        result = self._calibrate(_csv())
        self.assertEqual(result.model, "voltage_quadratic_command_map_calibrated")
        self.assertAlmostEqual(result.command_linear_per_v, 1000.0)
        self.assertAlmostEqual(result.command_quadratic_per_v, -200.0)
        self.assertAlmostEqual(result.validation_rmse_rpm, 0.0, places=8)
        self.assertEqual(len(result.calibration_data_sha256), 64)

    def test_requires_a_real_validation_split(self) -> None:
        with self.assertRaisesRegex(ValueError, "fewer_than_3_validation_points"):
            self._calibrate(_csv().replace("validation", "train"))

    def test_rejects_validation_extrapolation(self) -> None:
        with self.assertRaisesRegex(ValueError, "validation_command_outside_train_range"):
            self._calibrate(_csv(validation_command=0.9))

    def test_cli_prints_new_calibration_coefficients_and_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "rpm-calibration.csv"
            path.write_text(_csv(), encoding="utf-8")
            output = io.StringIO()
            with (
                patch("sys.argv", ["fftgyro-exp", "rpm-calibrate", str(path)]),
                redirect_stdout(output),
            ):
                main()
        result = output.getvalue()
        self.assertIn('"command_linear_per_v":', result)
        self.assertIn('"calibration_data_sha256":', result)
