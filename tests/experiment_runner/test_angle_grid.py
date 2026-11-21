from __future__ import annotations

import unittest

from fftgyro_quad_id.experiment_runner.angle_grid import (
    SUN_FORCE_BETA_EDGES_DEG,
    beta_partition,
    summarize,
)


class AngleGridTests(unittest.TestCase):
    def test_pdf_grid_misses_sun_high_sideslip_partition(self) -> None:
        summary = summarize("pdf-5x5")
        self.assertEqual(summary.pose_count, 25)
        self.assertIn("abs_beta_[60,90]", summary.missing_beta_partitions)

    def test_sun_coarse_is_truncated_at_45_with_both_signs(self) -> None:
        summary = summarize("sun-beta45-coarse")
        self.assertEqual(summary.pose_count, 35)
        self.assertEqual(summary.alpha_deg, ())
        self.assertEqual(summary.alpha_level_count, 5)
        self.assertEqual(summary.required_abs_alpha_clearance_deg, None)
        self.assertEqual(summary.missing_beta_partitions, ())
        self.assertTrue(summary.symmetric_signed_beta_coverage)
        self.assertEqual(summary.required_abs_beta_clearance_deg, 45.0)
        self.assertEqual(summary.boundary_holdout_beta_deg, (-45.0, 45.0))
        self.assertTrue(summary.sun_full_90deg_envelope_intentionally_truncated)
        self.assertTrue(summary.beta_is_fixed_per_block)
        self.assertFalse(summary.dynamic_beta_command_allowed)
        self.assertEqual(summary.beta_block_count, 7)
        self.assertTrue(summary.tare_required_after_each_beta_change)
        self.assertTrue(summary.alpha_levels_are_provisional)
        self.assertFalse(summary.alpha_fixture_levels_verified)
        self.assertTrue(summary.all_rotational_dof_mechanically_fixed)
        self.assertEqual(summary.fixed_attitude_axes, ("roll", "pitch", "yaw"))
        self.assertFalse(summary.executable_as_fixed_pose_plan)

    def test_measured_alpha_levels_replace_candidates_only_when_supplied(self) -> None:
        levels = (-31.2, -14.8, 0.1, 15.0, 30.9)
        summary = summarize(
            "sun-beta45-coarse",
            verified_abs_alpha_clearance_deg=32.0,
            verified_abs_beta_clearance_deg=45.0,
            alpha_fixture_levels_verified=True,
            alpha_fixture_levels_deg=levels,
        )
        self.assertEqual(summary.alpha_deg, levels)
        self.assertEqual(summary.required_abs_alpha_clearance_deg, 31.2)
        self.assertTrue(summary.executable_as_fixed_pose_plan)

    def test_sun_fine_covers_six_moment_partitions(self) -> None:
        summary = summarize("sun-beta45-moment")
        self.assertEqual(summary.pose_count, 45)
        self.assertEqual(len(summary.covered_beta_partitions), 3)
        self.assertEqual(summary.missing_beta_partitions, ())

    def test_partition_boundaries_are_deterministic(self) -> None:
        self.assertEqual(
            beta_partition(29.999, SUN_FORCE_BETA_EDGES_DEG),
            "abs_beta_[0,30)",
        )
        self.assertEqual(
            beta_partition(30.0, SUN_FORCE_BETA_EDGES_DEG),
            "abs_beta_[30,60)",
        )
        self.assertEqual(
            beta_partition(-90.0, SUN_FORCE_BETA_EDGES_DEG),
            "abs_beta_[60,90]",
        )

    def test_clearance_must_cover_both_axes(self) -> None:
        self.assertFalse(
            summarize(
                "sun-beta45-coarse",
                verified_abs_alpha_clearance_deg=40,
                verified_abs_beta_clearance_deg=40,
            ).physical_clearance_verified
        )
        self.assertFalse(
            summarize(
                "sun-beta45-coarse",
                verified_abs_alpha_clearance_deg=40,
                verified_abs_beta_clearance_deg=45,
            ).physical_clearance_verified
        )
