from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fftgyro_quad_id.experiment_runner.collective_mapping import build_collective_mapping


def _configuration(current_limit: float = 80.0, mass: float = 4.0) -> str:
    return f"""
[vehicle]
mass_kg = {mass}
gravity_mps2 = 10.0

[operation]
voltage_v = 15.7
current_limit_a = {current_limit}
max_command = 0.8
motor_ids = ["motor_1", "motor_2", "motor_3", "motor_4"]
hover_percentages = [60.0, 75.0, 90.0, 105.0]

[data]
steady_curve_csv = "curve.csv"
voltage_tolerance_v = 0.1
"""


def _curve(*, nonmonotonic_motor: str | None = None) -> str:
    rows = ["motor_id,voltage_v,command,thrust_n,rpm,current_a"]
    for motor in range(1, 5):
        motor_id = f"motor_{motor}"
        for command, thrust, rpm, current in (
            (0.2, 4.0, 2000.0, 2.0),
            (0.4, 8.0, 4000.0, 4.0),
            (0.6, 12.0, 6000.0, 6.0),
            (0.8, 16.0, 8000.0, 8.0),
        ):
            if motor_id == nonmonotonic_motor and command == 0.6:
                thrust = 7.0
            rows.append(f"{motor_id},15.7,{command},{thrust},{rpm},{current}")
    return "\n".join(rows) + "\n"


class CollectiveMappingTests(unittest.TestCase):
    def _files(self, root: Path, config: str, curve: str) -> Path:
        configuration = root / "mapping.toml"
        configuration.write_text(config, encoding="utf-8")
        (root / "curve.csv").write_text(curve, encoding="utf-8")
        return configuration

    def test_four_levels_interpolate_per_motor(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            config = self._files(Path(temp), _configuration(), _curve())
            result = build_collective_mapping(config)
            self.assertTrue(result.ready)
            self.assertEqual(len(result.levels), 4)
            ninety = next(level for level in result.levels if level.hover_percent == 90)
            self.assertAlmostEqual(ninety.total_target_thrust_n, 36.0)
            self.assertTrue(
                all(abs(point.command - 0.45) < 1e-9 for point in ninety.motor_points)
            )

    def test_current_limit_blocks_high_level(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            config = self._files(
                Path(temp), _configuration(current_limit=20.0), _curve()
            )
            result = build_collective_mapping(config)
            self.assertFalse(result.ready)
            self.assertIn("105pct_estimated_current_exceeds_limit", result.blockers)

    def test_target_outside_measured_range_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            config = self._files(Path(temp), _configuration(mass=8.0), _curve())
            result = build_collective_mapping(config)
            self.assertTrue(
                any("target_outside_measured_thrust_range" in item for item in result.blockers)
            )

    def test_nonmonotonic_curve_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            config = self._files(
                Path(temp), _configuration(), _curve(nonmonotonic_motor="motor_2")
            )
            result = build_collective_mapping(config)
            self.assertIn("motor_2:thrust_curve_not_strictly_increasing", result.blockers)
