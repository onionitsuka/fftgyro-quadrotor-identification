import unittest

from fftgyro_quad_id.experiment_runner.motor_single import (
    DOWNWARD_METHODS,
    DYNAMIC_DOWN_STEP,
    DYNAMIC_HOLD_MAX_S,
    DYNAMIC_HOLD_MIN_S,
    DYNAMIC_UP_STEPS,
    METHOD_DOWN_FINAL_FIXED,
    METHOD_DOWN_INITIAL_FIXED,
    METHOD_DOWN_OPERATING_POINT,
    METHOD_DYNAMIC,
    METHOD_LARGE_STEP,
    METHOD_STEADY_MAP,
    METHOD_SYMMETRIC,
    METHOD_SYNC,
    METHOD_THERMAL_COLD,
    METHOD_THERMAL_WARM,
    STEADY_HOLD_S,
    build_custom_step_plan,
    build_dynamic_waveform,
    build_plan,
    summarize,
)


class MotorSinglePlanTests(unittest.TestCase):
    def test_recommended_matches_agreed_voltage_and_repeat_counts(self) -> None:
        segments = build_plan("recommended")
        summary = summarize("recommended", segments)
        self.assertEqual(summary.steady_segments, 5 * 8 * 3)
        dynamic_baselines = [
            item for item in segments if item.purpose == "dynamic_baseline"
        ]
        self.assertEqual(len(dynamic_baselines), 1 * 3)
        self.assertEqual(summary.sync_segments, 1)

    def test_minimum_counts_use_three_static_voltages(self) -> None:
        summary = summarize("minimum", build_plan("minimum"))
        self.assertEqual(summary.steady_segments, 3 * 8 * 2)
        self.assertEqual(summary.sync_segments, 1)

    def test_extended_profile_adds_staged_points_through_one_hundred_percent(self) -> None:
        segments = build_plan("recommended-extended-100")
        summary = summarize("recommended-extended-100", segments)
        steady = [item for item in segments if item.purpose == "steady_rpm_map"]

        self.assertEqual(summary.steady_segments, 5 * 12 * 3)
        self.assertEqual(
            sorted({item.input_end for item in steady}),
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0],
        )

    def test_sequence_and_ids_are_stable(self) -> None:
        segments = build_plan("recommended")
        self.assertEqual(segments[0].sequence, 1)
        self.assertEqual(segments[0].segment_id, "motor_single-00001")
        self.assertEqual(segments[-1].sequence, len(segments))
        self.assertEqual(
            len({segment.segment_id for segment in segments}), len(segments)
        )

    def test_steady_map_is_upward_only_and_uses_last_two_seconds(self) -> None:
        steady = [
            item
            for item in build_plan("recommended")
            if item.purpose == "steady_rpm_map"
        ]
        self.assertTrue(all(item.input_end >= item.input_start for item in steady))
        self.assertEqual(STEADY_HOLD_S, 10.0)
        self.assertTrue(all(item.post_hold_s == STEADY_HOLD_S for item in steady))
        self.assertTrue(all(item.analysis_window_s == 2.0 for item in steady))

    def test_dynamic_waveform_limits_high_output_rises_to_ten_points(self) -> None:
        waveform = build_dynamic_waveform()
        upward = [
            item.end - item.start
            for item in waveform
            if item.purpose == "dynamic_up"
        ]
        downward = [item.start - item.end for item in waveform if item.end < item.start]
        self.assertTrue(upward)
        self.assertEqual(
            {round(step, 2) for step in upward}, set(DYNAMIC_UP_STEPS)
        )
        high_output = [
            item.end - item.start
            for item in waveform
            if item.purpose == "dynamic_up" and item.end > 0.60 + 1e-9
        ]
        self.assertTrue(high_output)
        self.assertTrue(all(abs(step - 0.10) < 1e-9 for step in high_output))
        self.assertAlmostEqual(max(upward), max(DYNAMIC_UP_STEPS))
        self.assertTrue(all(step <= max(DYNAMIC_UP_STEPS) + 1e-9 for step in upward))
        self.assertTrue(downward)
        self.assertTrue(
            all(abs(step - DYNAMIC_DOWN_STEP) < 1e-9 for step in downward)
        )

    def test_boundary_method_uses_ten_point_rises_above_sixty_percent(self) -> None:
        segments = build_plan("recommended", methods=(METHOD_LARGE_STEP,))
        rises = [item for item in segments if item.purpose == "large_step_up"]

        self.assertEqual(len(rises), 1 * 3 * 4)
        self.assertEqual(
            {(item.input_start, item.input_end) for item in rises},
            {(0.20, 0.40), (0.40, 0.60), (0.60, 0.70), (0.70, 0.80)},
        )
        self.assertTrue(
            all(
                item.input_end - item.input_start <= (
                    0.10 if item.input_end > 0.60 + 1e-9 else 0.20
                ) + 1e-9
                for item in rises
            )
        )
        approaches = [
            item for item in segments if item.purpose == "large_step_approach"
        ]
        self.assertTrue(approaches)
        self.assertTrue(
            all(item.input_end - item.input_start <= 0.10 + 1e-9 for item in approaches)
        )

    def test_dynamic_holds_are_within_agreed_range(self) -> None:
        dynamic = [
            item
            for item in build_dynamic_waveform()
            if item.purpose in {"dynamic_up", "dynamic_down"}
        ]
        self.assertTrue(
            all(DYNAMIC_HOLD_MIN_S <= item.hold_s <= DYNAMIC_HOLD_MAX_S for item in dynamic)
        )

    def test_dynamic_waveform_is_identical_across_runs(self) -> None:
        segments = build_plan("recommended")

        def run(voltage: float, repeat: int) -> list[tuple[float, float, float]]:
            return [
                (item.input_start, item.input_end, item.post_hold_s)
                for item in segments
                if item.voltage_v == voltage
                and item.repeat == repeat
                and item.purpose in {"dynamic_up", "dynamic_down", "controlled_down"}
            ]

        self.assertEqual(run(14.7, 1), run(15.7, 2))
        self.assertEqual(run(15.7, 2), run(16.7, 3))

    def test_no_non_emergency_downward_step_exceeds_point_zero_five(self) -> None:
        downward = [
            item.input_start - item.input_end
            for item in build_plan("all-motors-validation")
            if item.input_end < item.input_start
        ]
        self.assertTrue(downward)
        self.assertLessEqual(max(downward), 0.05 + 1e-9)

    def test_each_existing_main_method_can_be_selected_independently(self) -> None:
        expected_runs = {
            METHOD_STEADY_MAP: 15,
            METHOD_DYNAMIC: 3,
            METHOD_LARGE_STEP: 12,
            METHOD_SYNC: 1,
        }
        for method_id, run_count in expected_runs.items():
            with self.subTest(method_id=method_id):
                segments = build_plan("recommended", methods=(method_id,))
                run_keys = {
                    (
                        item.method_id,
                        item.condition_id,
                        item.motor_id,
                        item.voltage_v,
                        item.repeat,
                    )
                    for item in segments
                }
                self.assertEqual(len(run_keys), run_count)
                self.assertEqual({item.method_id for item in segments}, {method_id})

    def test_downward_suite_defines_all_four_methods_and_thermal_brackets(self) -> None:
        segments = build_plan("recommended", methods=DOWNWARD_METHODS)
        methods = list(dict.fromkeys(item.method_id for item in segments))

        self.assertEqual(
            methods,
            [
                METHOD_THERMAL_COLD,
                METHOD_DOWN_FINAL_FIXED,
                METHOD_DOWN_INITIAL_FIXED,
                METHOD_SYMMETRIC,
                METHOD_DOWN_OPERATING_POINT,
                METHOD_THERMAL_WARM,
            ],
        )
        self.assertEqual({item.voltage_v for item in segments}, {16.2})
        self.assertEqual({item.repeat for item in segments}, {1, 2, 3})

    def test_downward_conditions_have_expected_steps_and_settling_holds(self) -> None:
        segments = build_plan("recommended", methods=DOWNWARD_METHODS)
        transitions = {
            (
                item.method_id,
                item.condition_id,
                round(item.input_start * 100),
                round(item.input_end * 100),
            )
            for item in segments
            if item.direction in {"up", "down"}
        }

        self.assertIn((METHOD_DOWN_FINAL_FIXED, "drop05_to50", 55, 50), transitions)
        self.assertIn((METHOD_DOWN_FINAL_FIXED, "drop15_to50", 65, 50), transitions)
        self.assertIn((METHOD_DOWN_INITIAL_FIXED, "drop05_from65", 65, 60), transitions)
        self.assertIn((METHOD_DOWN_INITIAL_FIXED, "drop10_from65", 65, 55), transitions)
        self.assertIn((METHOD_SYMMETRIC, "span15_around50", 50, 65), transitions)
        self.assertIn((METHOD_SYMMETRIC, "span15_around50", 65, 50), transitions)
        self.assertIn(
            (METHOD_DOWN_OPERATING_POINT, "drop10_80to70", 80, 70), transitions
        )
        self.assertNotIn(
            (METHOD_DOWN_OPERATING_POINT, "drop10_60to50", 60, 50), transitions
        )
        measurement = [item for item in segments if item.purpose != "settling_approach"]
        approaches = [item for item in segments if item.purpose == "settling_approach"]
        self.assertTrue(all(item.post_hold_s == 10.0 for item in measurement))
        self.assertTrue(all(item.analysis_window_s == 3.0 for item in measurement))
        self.assertTrue(all(item.post_hold_s == 0.5 for item in approaches))
        self.assertTrue(all(item.analysis_window_s is None for item in approaches))
        self.assertLessEqual(
            max(item.input_end - item.input_start for item in approaches),
            0.10 + 1e-9,
        )

    def test_custom_sequence_preserves_levels_holds_and_repetitions(self) -> None:
        segments = build_custom_step_plan(
            method_name="custom-hysteresis",
            levels_percent=(50, 60, 50),
            hold_s=10.0,
            repetitions=3,
            voltage_v=14.7,
            motor=1,
        )

        self.assertEqual(len(segments), 21)
        self.assertEqual({item.method_id for item in segments}, {"custom-hysteresis"})
        self.assertEqual({item.repeat for item in segments}, {1, 2, 3})
        self.assertEqual(
            [
                round(item.input_end * 100)
                for item in segments
                if item.repeat == 1 and item.purpose != "settling_approach"
            ],
            [50, 60, 50],
        )
        self.assertTrue(
            all(
                item.post_hold_s == 10.0
                for item in segments
                if item.purpose != "settling_approach"
            )
        )

    def test_custom_sequence_rejects_unvalidated_step_sizes(self) -> None:
        with self.assertRaisesRegex(ValueError, "downward step"):
            build_custom_step_plan(
                method_name="unsafe-down",
                levels_percent=(70, 50),
                hold_s=10.0,
                repetitions=1,
                voltage_v=14.7,
                motor=1,
            )
        with self.assertRaisesRegex(ValueError, "upward step"):
            build_custom_step_plan(
                method_name="unsafe-up",
                levels_percent=(20, 60),
                hold_s=10.0,
                repetitions=1,
                voltage_v=14.7,
                motor=1,
            )
        with self.assertRaisesRegex(ValueError, "upward step"):
            build_custom_step_plan(
                method_name="unsafe-high-up",
                levels_percent=(60, 80),
                hold_s=3.0,
                repetitions=1,
                voltage_v=16.2,
                motor=1,
            )

        safe_high = build_custom_step_plan(
            method_name="safe-high-up",
            levels_percent=(60, 70, 80),
            hold_s=3.0,
            repetitions=1,
            voltage_v=16.2,
            motor=1,
        )
        measured = [
            item for item in safe_high if item.purpose != "settling_approach"
        ]
        self.assertEqual([round(item.input_end * 100) for item in measured], [60, 70, 80])


if __name__ == "__main__":
    unittest.main()
