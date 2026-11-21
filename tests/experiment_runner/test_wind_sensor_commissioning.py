from __future__ import annotations

import unittest

from fftgyro_quad_id.experiment_runner.wind_sensor_commissioning import (
    build_wind_sensor_commissioning_plan,
    summarize_wind_sensor_commissioning,
)


class WindSensorCommissioningTests(unittest.TestCase):
    def test_two_sensor_mode_pairs_diagnostic_with_probe_removed_reference(self) -> None:
        mode = "simultaneous-two-sensor"
        segments = build_wind_sensor_commissioning_plan(mode)
        result = summarize_wind_sensor_commissioning(mode, segments)
        self.assertEqual(result.recording_count, 54)
        self.assertEqual(result.pure_recording_seconds, 1620.0)
        self.assertEqual(result.freestream_recordings, 54)
        self.assertEqual(result.near_frame_recordings, 27)
        self.assertEqual(result.probe_installed_recordings, 27)
        self.assertEqual(result.probe_removed_reference_recordings, 27)
        self.assertTrue(result.frame_influence_is_diagnostic_not_sitl_input)

    def test_one_sensor_mode_brackets_every_near_measurement(self) -> None:
        mode = "sequential-one-sensor"
        segments = build_wind_sensor_commissioning_plan(mode)
        result = summarize_wind_sensor_commissioning(mode, segments)
        self.assertEqual(result.recording_count, 81)
        self.assertEqual(result.freestream_recordings, 54)
        self.assertEqual(result.near_frame_recordings, 27)
        for index in range(0, len(segments), 3):
            block = segments[index : index + 3]
            self.assertEqual(
                [item.bracket_role for item in block],
                ["before_near_frame", "diagnostic", "after_near_frame"],
            )

    def test_rotors_never_run_during_frame_only_commissioning(self) -> None:
        segments = build_wind_sensor_commissioning_plan("simultaneous-two-sensor")
        self.assertTrue(all(not item.rotors_running for item in segments))
        self.assertTrue(all(not item.propellers_rotating for item in segments))
        self.assertTrue(all(not item.use_for_sitl_identification for item in segments))

    def test_recording_must_be_at_least_thirty_seconds(self) -> None:
        with self.assertRaises(ValueError):
            build_wind_sensor_commissioning_plan(
                "simultaneous-two-sensor", record_s=10.0
            )
