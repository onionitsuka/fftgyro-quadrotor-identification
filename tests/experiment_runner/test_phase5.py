from __future__ import annotations

import unittest

from fftgyro_quad_id.experiment_runner.phase5 import (
    Phase5CKind,
    Phase5Kind,
    build_phase5b_plan,
    build_phase5c_plan,
    summarize,
    summarize_phase5c,
)


class Phase5Tests(unittest.TestCase):
    def test_sun_coarse_pilot_scales_pdf_count_and_fixes_all_axes(self) -> None:
        segments = build_phase5b_plan("pilot", "sun-beta45-coarse")
        result = summarize("pilot", "sun-beta45-coarse", segments)
        self.assertEqual(result.data_segments, 70)
        self.assertEqual(result.pdf_equivalent_5x5_data_segments, 50)
        self.assertEqual(result.beta_blocks, 7)
        self.assertEqual(result.pose_tare_actions, 35)
        self.assertTrue(result.all_rotational_dof_mechanically_fixed)
        self.assertFalse(result.dynamic_attitude_test_included)
        self.assertEqual(result.executable_segments, 0)

    def test_minimum_is_four_collectives_two_repeats_per_pose(self) -> None:
        segments = build_phase5b_plan("minimum", "sun-beta45-coarse")
        measurements = [
            item for item in segments if item.kind == Phase5Kind.STATIC_MEASUREMENT
        ]
        self.assertEqual(len(measurements), 280)
        self.assertEqual({item.differential_input for item in measurements}, {"neutral"})
        self.assertTrue(all(not item.beta_dynamic_command_allowed for item in segments))

    def test_measured_alpha_and_fixture_ids_do_not_bypass_other_gates(self) -> None:
        alpha = (-31.2, -14.8, 0.1, 15.0, 30.9)
        fixtures = ("A1", "A2", "A3", "A4", "A5")
        segments = build_phase5b_plan(
            "pilot",
            "sun-beta45-coarse",
            alpha_nominal_deg=alpha,
            fixture_ids=fixtures,
        )
        result = summarize("pilot", "sun-beta45-coarse", segments)
        self.assertEqual(result.executable_segments, 0)
        self.assertIn("tare_duration_is_unresolved", result.blockers)
        self.assertIn("beta_clearance_and_tare_are_unresolved", result.blockers)

    def test_alpha_level_count_must_match_fixture_slots(self) -> None:
        with self.assertRaises(ValueError):
            build_phase5b_plan(
                "pilot", "sun-beta45-coarse", alpha_nominal_deg=(0.0,)
            )


class Phase5CTests(unittest.TestCase):
    def test_minimum_excludes_roll_dynamics(self) -> None:
        segments = build_phase5c_plan("minimum")
        result = summarize_phase5c("minimum", segments)
        self.assertEqual(result.dynamic_axes, ("pitch", "yaw"))
        self.assertEqual(result.roll_fixed_deg, 0.0)
        self.assertEqual(result.rotor_stopped_free_decay_runs, 18)
        self.assertEqual(result.rotor_running_dynamic_runs, 8)
        self.assertEqual(result.total_runs, 26)
        self.assertTrue(all(item.dynamic_axis != "roll" for item in segments))

    def test_recommended_excludes_roll_dynamics(self) -> None:
        segments = build_phase5c_plan("recommended")
        result = summarize_phase5c("recommended", segments)
        self.assertEqual(result.rotor_stopped_free_decay_runs, 30)
        self.assertEqual(result.rotor_running_dynamic_runs, 48)
        self.assertEqual(result.run_count_reduction_from_roll_constraint, 39)
        running = [
            item for item in segments if item.kind == Phase5CKind.ROTOR_RUNNING_DYNAMIC
        ]
        self.assertEqual(
            {item.waveform for item in running},
            {"low_frequency_sine", "chirp", "prbs", "small_step"},
        )
