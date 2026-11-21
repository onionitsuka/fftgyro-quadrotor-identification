from __future__ import annotations

import unittest

from fftgyro_quad_id.experiment_runner.calibration import (
    build_factory_calibrated_plan,
    build_full_validation_plan,
    summarize,
)


class CalibrationPlanTests(unittest.TestCase):
    def test_pdf_minimum_force_moment_and_sync_counts(self) -> None:
        summary = summarize(build_full_validation_plan(), mode="full")
        self.assertEqual(summary.known_force_points, 54)
        self.assertEqual(summary.known_moment_points, 20)
        self.assertEqual(summary.sync_steps, 10)

    def test_unknown_lab_values_remain_explicitly_unresolved(self) -> None:
        summary = summarize(build_full_validation_plan(), mode="full")
        self.assertEqual(summary.unresolved_segments, 90)
        self.assertIsNone(summary.scheduled_recording_seconds)

    def test_values_resolve_force_and_moment_points(self) -> None:
        plan = build_full_validation_plan(
            force_levels_n=(4.9, 9.8, 19.6),
            moment_values_nm=tuple(0.1 * index for index in range(1, 21)),
            static_hold_s=10.0,
            sync_step_s=6.0,
            coordinate_hold_s=10.0,
        )
        self.assertEqual(summarize(plan, mode="full").unresolved_segments, 0)
        self.assertEqual(
            summarize(plan, mode="full").scheduled_recording_seconds, 920.0
        )

    def test_bad_force_levels_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_full_validation_plan(force_levels_n=(1.0, 1.0, 2.0))

    def test_fewer_than_twenty_moment_points_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_full_validation_plan(moment_values_nm=(0.1,) * 19)

    def test_factory_mode_omits_known_load_recalibration(self) -> None:
        summary = summarize(build_factory_calibrated_plan(), mode="factory")
        self.assertEqual(summary.mode, "factory")
        self.assertEqual(summary.known_force_points, 0)
        self.assertEqual(summary.known_moment_points, 0)
        self.assertEqual(summary.sync_steps, 10)
        self.assertEqual(summary.coordinate_checks, 6)
        self.assertEqual(summary.segment_count, 19)

    def test_factory_mode_is_resolved_after_timing_is_selected(self) -> None:
        plan = build_factory_calibrated_plan(
            sync_step_s=6.0,
            coordinate_hold_s=10.0,
        )
        self.assertEqual(summarize(plan, mode="factory").unresolved_segments, 0)
