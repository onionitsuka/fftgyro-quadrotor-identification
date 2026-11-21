from __future__ import annotations

import unittest

from fftgyro_quad_id.experiment_runner.wind_sensor_policy import get_wind_sensor_policy


class WindSensorPolicyTests(unittest.TestCase):
    def test_fixed_diagnostic_uses_near_probe_but_not_for_sitl_fit(self) -> None:
        policy = get_wind_sensor_policy("frame_diagnostic_fixed")
        self.assertFalse(policy.use_for_sitl_identification)
        self.assertFalse(policy.near_frame_probe_must_be_removed)

    def test_static_main_identification_removes_near_probe(self) -> None:
        policy = get_wind_sensor_policy("phase6a_static")
        self.assertTrue(policy.use_for_sitl_identification)
        self.assertTrue(policy.near_frame_probe_must_be_removed)

    def test_dynamic_identification_is_freestream_only(self) -> None:
        policy = get_wind_sensor_policy("phase6b_dynamic")
        self.assertEqual(policy.near_frame_sensor_mode, "forbidden_in_dynamic_run")
        self.assertTrue(policy.near_frame_probe_must_be_removed)
