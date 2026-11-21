from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fftgyro_quad_id.experiment_runner.hardware_status import evaluate_hardware_status


class HardwareStatusTests(unittest.TestCase):
    def test_deferred_wind_sensor_does_not_become_next_action(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result = evaluate_hardware_status(
                Path(temporary), deferred=("wind-sensors",)
            )
        wind = next(item for item in result.checks if item.check_id == "wind-sensors")
        self.assertEqual(wind.state, "deferred")
        self.assertEqual(result.next_check_id, "power-supply")

    def test_first_missing_check_returns_copy_command(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result = evaluate_hardware_status(Path(temporary))
        self.assertFalse(result.ready)
        self.assertEqual(result.next_check_id, "power-supply")
        self.assertIn("power-supply.example.toml", result.next_action or "")

    def test_unknown_deferred_check_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(ValueError):
                evaluate_hardware_status(
                    Path(temporary), deferred=("not-a-real-check",)
                )
