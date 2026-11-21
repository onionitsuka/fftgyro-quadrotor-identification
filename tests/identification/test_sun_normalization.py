"""Physical invariants of the adopted Sun normalization."""

import math
import unittest

from fftgyro_quad_id.identification.sun_static import (
    SunStaticConfig,
    rpm_acquisition_to_x500_motor_order,
    sun_motor_inputs,
    wrench_scales,
)


class SunNormalizationTests(unittest.TestCase):
    def test_campaign_rpm_channels_are_reordered_to_motor_order(self) -> None:
        reordered = rpm_acquisition_to_x500_motor_order([101.0, 102.0, 103.0, 104.0])
        self.assertEqual(reordered.tolist(), [104.0, 101.0, 103.0, 102.0])

    def test_equal_speeds_have_zero_differentials(self) -> None:
        values = sun_motor_inputs([400.0] * 4)
        self.assertEqual(values["omega_bar_rad_s"], 400.0)
        for name in ("up", "uq", "ur"):
            self.assertEqual(values[name], 0.0)

    def test_x500_axis_allocation_and_squared_input(self) -> None:
        # A 2:1 RPM ratio gives squared normalized speeds 1.6 and 0.4;
        # the two-versus-two difference must therefore be 2.4.
        for target, speeds in (
            ("up", [1.0, 2.0, 2.0, 1.0]),
            ("uq", [2.0, 1.0, 2.0, 1.0]),
            ("ur", [2.0, 2.0, 1.0, 1.0]),
        ):
            values = sun_motor_inputs(speeds)
            for name in ("up", "uq", "ur"):
                self.assertAlmostEqual(values[name], 2.4 if name == target else 0.0)

    def test_scaling_rpm_preserves_inputs_and_quadruples_wrench_scale(self) -> None:
        speeds = [300.0, 410.0, 370.0, 390.0]
        low = sun_motor_inputs(speeds)
        high = sun_motor_inputs([2.0 * value for value in speeds])
        for name in ("up", "uq", "ur"):
            self.assertAlmostEqual(low[name], high[name])
        cfg = SunStaticConfig()
        f1, m1 = wrench_scales(cfg, low["omega_bar_rad_s"])
        f2, m2 = wrench_scales(cfg, high["omega_bar_rad_s"])
        self.assertAlmostEqual(f2, 4.0 * f1)
        self.assertAlmostEqual(m2, 4.0 * m1)
        self.assertAlmostEqual(m1 / f1, cfg.reference_length_m)

    def test_force_scale_has_four_disks_and_no_half_factor(self) -> None:
        cfg = SunStaticConfig(rotor_radius_m=1.0, air_density_kg_m3=1.0)
        force, _ = wrench_scales(cfg, 1.0)
        self.assertAlmostEqual(force, 4.0 * math.pi)

    def test_invalid_or_stopped_rotors_are_rejected(self) -> None:
        for speeds in ([0.0] * 4, [1, 2, 3], [-1, 2, 3, 4], [math.nan, 1, 1, 1]):
            with self.subTest(speeds=speeds), self.assertRaises(ValueError):
                sun_motor_inputs(speeds)


if __name__ == "__main__":
    unittest.main()
