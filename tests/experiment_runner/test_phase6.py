from __future__ import annotations

import unittest

from fftgyro_quad_id.experiment_runner.phase6 import (
    Phase6AKind,
    build_phase6a_plan,
    build_phase6b_plan,
    summarize_phase6a,
    summarize_phase6b,
)


class Phase6ATests(unittest.TestCase):
    def test_minimum_scales_pdf_grid_to_seven_fixed_beta_blocks(self) -> None:
        segments = build_phase6a_plan("minimum", "sun-beta45-coarse")
        result = summarize_phase6a("minimum", "sun-beta45-coarse", segments)
        self.assertEqual(result.neutral_data_segments, 280)
        self.assertEqual(result.doptimal_selection_slots, 280)
        self.assertEqual(result.total_data_segments, 560)
        self.assertEqual(result.pdf_5x5_total_segments, 400)
        self.assertEqual(result.pose_tare_actions, 35)
        self.assertEqual(result.roll_fixed_deg, 0.0)
        self.assertEqual(result.executable_segments, 0)

    def test_recommended_has_three_selected_inputs_and_two_repeats(self) -> None:
        segments = build_phase6a_plan("recommended", "sun-beta45-coarse")
        result = summarize_phase6a("recommended", "sun-beta45-coarse", segments)
        self.assertEqual(result.total_data_segments, 2520)
        measurements = [
            item for item in segments if item.kind == Phase6AKind.STATIC_MEASUREMENT
        ]
        self.assertEqual({item.input_selection_slot for item in measurements}, {1, 2, 3})
        self.assertEqual({item.repeat for item in measurements}, {1, 2})

    def test_physical_level_lengths_are_not_guessed(self) -> None:
        with self.assertRaises(ValueError):
            build_phase6a_plan("minimum", wind_speeds_mps=(0.0, 5.0, 10.0))

    def test_historical_five_and_ten_mps_form_two_speed_pilot(self) -> None:
        speeds = (5.0, 10.0)
        segments = build_phase6a_plan(
            "minimum", "sun-beta45-coarse", wind_speeds_mps=speeds
        )
        result = summarize_phase6a("minimum", "sun-beta45-coarse", segments)
        self.assertEqual(result.configured_wind_speeds_mps, speeds)
        self.assertEqual(result.wind_level_count, 2)
        self.assertEqual(result.wind_level_shortfall, 2)
        self.assertEqual(result.neutral_data_segments, 140)
        self.assertEqual(result.total_data_segments, 280)


class Phase6BTests(unittest.TestCase):
    def test_roll_is_removed_from_minimum_dynamic_plan(self) -> None:
        segments = build_phase6b_plan("minimum")
        result = summarize_phase6b("minimum", segments)
        self.assertEqual(result.dynamic_axes, ("pitch", "yaw"))
        self.assertEqual(result.roll_fixed_deg, 0.0)
        self.assertEqual(result.run_count, 144)
        self.assertEqual(result.pdf_three_axis_run_count, 216)
        self.assertEqual(result.run_count_reduction_from_roll_constraint, 72)
        self.assertTrue(all(item.dynamic_axis != "roll" for item in segments))

    def test_roll_is_removed_from_recommended_dynamic_plan(self) -> None:
        result = summarize_phase6b("recommended", build_phase6b_plan("recommended"))
        self.assertEqual(result.run_count, 256)
        self.assertEqual(result.run_count_reduction_from_roll_constraint, 128)

    def test_two_historical_winds_reduce_dynamic_pilot(self) -> None:
        segments = build_phase6b_plan("minimum", (5.0, 10.0))
        result = summarize_phase6b("minimum", segments)
        self.assertEqual(result.configured_wind_speeds_mps, (5.0, 10.0))
        self.assertEqual(result.wind_level_count, 2)
        self.assertEqual(result.wind_level_shortfall, 2)
        self.assertEqual(result.run_count, 72)
        self.assertEqual(result.three_axis_run_count_at_configured_winds, 108)
        self.assertEqual(result.run_count_reduction_from_roll_constraint, 36)
        self.assertEqual(result.run_count_reduction_from_wind_shortfall_at_two_axes, 72)
