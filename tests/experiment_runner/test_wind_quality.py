from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fftgyro_quad_id.experiment_runner.wind_quality import (
    WindObservation,
    evaluate_wind_observation,
)


def _configuration(*, primary: str = "sensor_pair_mean") -> str:
    return f"""
[wind_measurement_policy]
primary_reference = "{primary}"
pdf_target_deviation_threshold_fraction = 0.03
q_to_sensor_pair_warning_mps = 0.5
frame_speed_change_warning_fraction = 0.05

[wind_sensor_layout]
sensor1_role = "freestream_reference"
sensor2_role = "near_frame_diagnostic"
freestream_location_documented = true
outside_model_rotor_and_fixture_influence_confirmed = true
outside_wall_boundary_layer_confirmed = true
probe_does_not_obstruct_test_section_confirmed = true
calibration_current = true
empty_tunnel_verification_completed = true
frame_diagnostic_enabled = true
near_frame_location_documented = true
near_frame_outside_rotor_slipstream_confirmed = true
near_frame_probe_does_not_obstruct_flow_confirmed = true
near_frame_body_relative_position_repeatable = true
"""


class WindQualityTests(unittest.TestCase):
    def _evaluate(self, config: str, observation: WindObservation):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "wind.toml"
            path.write_text(config, encoding="utf-8")
            return evaluate_wind_observation(path, observation)

    def test_historical_case_one_passes_with_explicit_mean_policy(self) -> None:
        result = self._evaluate(
            _configuration(primary="sensor1"),
            WindObservation(5.0, 4.5, 5.3, 4.7),
        )
        self.assertTrue(result.ready_for_identification)
        self.assertAlmostEqual(result.model_wind_speed_mps or 0.0, 5.3)
        self.assertAlmostEqual(result.sensor_pair_difference_mps, 0.6)
        self.assertTrue(result.measured_value_required_by_pdf)

    def test_unresolved_primary_blocks_identification(self) -> None:
        result = self._evaluate(
            _configuration(
                primary="unresolved_until_sensor_locations_and_calibration_are_known"
            ),
            WindObservation(5.0, 4.5, 5.3, 4.7),
        )
        self.assertIn("primary_wind_reference_unresolved", result.blockers)
    def test_sensor_difference_is_frame_effect_not_identification_blocker(self) -> None:
        result = self._evaluate(
            _configuration(primary="sensor1"),
            WindObservation(5.0, 4.5, 5.3, 4.7),
        )
        self.assertTrue(result.ready_for_identification)
        self.assertTrue(result.ready_for_frame_diagnostic)
        self.assertAlmostEqual(result.frame_speed_ratio or 0.0, 4.7 / 5.3)
        self.assertIn("near_frame_speed_differs_from_freestream", result.warnings)

    def test_more_than_three_percent_uses_measured_value(self) -> None:
        result = self._evaluate(
            _configuration(primary="sensor2"),
            WindObservation(10.0, 10.1, 9.7, 8.9),
        )
        self.assertTrue(result.measured_value_required_by_pdf)
        self.assertIn(
            "target_deviation_exceeds_pdf_3pct_use_measured_wind",
            result.warnings,
        )

    def test_pair_mean_is_rejected_when_sensors_have_different_roles(self) -> None:
        result = self._evaluate(
            _configuration(primary="sensor_pair_mean"),
            WindObservation(5.0, 4.5, 5.3, 4.7),
        )
        self.assertIn(
            "primary_reference_must_be_freestream_sensor",
            result.blockers,
        )

    def test_second_sensor_is_optional(self) -> None:
        result = self._evaluate(
            _configuration(primary="sensor1"),
            WindObservation(5.0, 4.5, 5.1, None),
        )
        self.assertTrue(result.ready_for_identification)
        self.assertFalse(result.ready_for_frame_diagnostic)
        self.assertIsNone(result.sensor_pair_difference_mps)
        self.assertEqual(result.model_wind_speed_mps, 5.1)
