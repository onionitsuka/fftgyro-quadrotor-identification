from __future__ import annotations

import unittest

from fftgyro_quad_id.experiment_runner.axis_policy import (
    DYNAMIC_EXCITATION_AXES,
    ROLL_FIXED_DEG,
    dynamic_axis_allowed,
)


class AxisPolicyTests(unittest.TestCase):
    def test_roll_is_fixed_and_never_a_dynamic_axis(self) -> None:
        self.assertEqual(ROLL_FIXED_DEG, 0.0)
        self.assertEqual(DYNAMIC_EXCITATION_AXES, ("pitch", "yaw"))
        self.assertFalse(dynamic_axis_allowed("roll"))
        self.assertTrue(dynamic_axis_allowed("pitch"))
        self.assertTrue(dynamic_axis_allowed("yaw"))
