from __future__ import annotations

import unittest

from fftgyro_quad_id.experiment_runner.phase4 import build_plan, summarize


class Phase4PlanTests(unittest.TestCase):
    def test_main_phase4_manifest_requires_near_probe_removed(self) -> None:
        segments = build_plan("minimum")
        self.assertTrue(all(item.wind_sensor_mode == "freestream_only" for item in segments))
        self.assertTrue(all(item.near_frame_probe_must_be_removed for item in segments))
        self.assertTrue(all(item.use_for_sitl_identification for item in segments))

    def test_minimum_body_lattice_has_168_segments(self) -> None:
        summary = summarize("minimum", build_plan("minimum"))
        self.assertEqual(summary.rig_static_segments, 162)
        self.assertEqual(summary.no_wind_free_decay_runs, 45)
        self.assertEqual(summary.wind_free_decay_runs, 0)
        self.assertEqual(summary.body_static_segments, 168)
        self.assertEqual(summary.segment_count, 375)

    def test_recommended_counts_follow_detailed_phase4_sections(self) -> None:
        summary = summarize("recommended", build_plan("recommended"))
        self.assertEqual(summary.rig_static_segments, 162)
        self.assertEqual(summary.no_wind_free_decay_runs, 45)
        self.assertEqual(summary.wind_free_decay_runs, 72)
        self.assertEqual(summary.body_static_segments, 600)
        self.assertEqual(summary.segment_count, 879)

    def test_minimum_body_poses_are_seven_unique_conditions(self) -> None:
        body = [
            item
            for item in build_plan("minimum")
            if item.subtest == "rotor_stopped_body_wind"
        ]
        poses = {(item.alpha_nominal_deg, item.beta_nominal_deg) for item in body}
        self.assertEqual(len(poses), 7)
        self.assertIn((-40.0, 0.0), poses)
        self.assertIn((0.0, -20.0), poses)

    def test_unspecified_wind_decay_values_are_not_invented(self) -> None:
        wind_decay = [
            item
            for item in build_plan("recommended")
            if item.subtest == "rig_free_decay_wind"
        ]
        self.assertTrue(all(item.wind_speed_mps is None for item in wind_decay))
        self.assertTrue(all(item.initial_angle_deg is None for item in wind_decay))
        self.assertTrue(all(item.requires_operator_value for item in wind_decay))

    def test_pdf_internal_count_conflict_is_exposed(self) -> None:
        summary = summarize("recommended", build_plan("recommended"))
        self.assertFalse(summary.detailed_rig_static_matches_pdf_summary)
        self.assertEqual(summary.pdf_summary_rig_static_segments, 450)
        self.assertIn(
            "phase4_detail_defines_162_rig_static_segments_but_time_table_states_450",
            summary.source_discrepancies,
        )

    def test_static_and_free_decay_durations_follow_pdf_time_assumptions(self) -> None:
        plan = build_plan("recommended")
        self.assertTrue(
            all(item.duration_s == 20.0 for item in plan if item.kind == "static")
        )
        self.assertTrue(
            all(item.duration_s == 30.0 for item in plan if item.kind == "free_decay")
        )
