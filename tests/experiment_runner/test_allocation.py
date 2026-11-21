import unittest

from fftgyro_quad_id.experiment_runner.allocation import (
    AllocationSigns,
    build_plan,
    resolve_motor_inputs,
    summarize,
)


class AllocationPlanTests(unittest.TestCase):
    def test_pdf_recommended_plan_has_156_segments(self) -> None:
        segments = build_plan()
        summary = summarize(segments)
        self.assertEqual(summary.segment_count, 156)
        self.assertEqual(summary.pure_recording_seconds, 156 * 15)

    def test_each_collective_has_13_patterns_and_3_repeats(self) -> None:
        segments = build_plan()
        for collective in (0.30, 0.45, 0.60, 0.75):
            selected = [item for item in segments if item.collective == collective]
            self.assertEqual(len(selected), 39)
            self.assertEqual(len({item.pattern_id for item in selected}), 13)

    def test_valid_sign_vectors_resolve_symbolic_roll_input(self) -> None:
        signs = AllocationSigns(
            sp=(1, -1, -1, 1),
            sq=(1, 1, -1, -1),
            sr=(1, -1, 1, -1),
        )
        roll = next(item for item in build_plan() if item.pattern_id == "plus_p")
        values = resolve_motor_inputs(roll, signs)
        self.assertEqual(values, (0.35, 0.25, 0.25, 0.35))

    def test_non_orthogonal_or_unbalanced_signs_are_rejected(self) -> None:
        bad = AllocationSigns(
            sp=(1, 1, 1, -1),
            sq=(1, 1, -1, -1),
            sr=(1, -1, 1, -1),
        )
        neutral = build_plan()[0]
        with self.assertRaises(ValueError):
            resolve_motor_inputs(neutral, bad)

    def test_symbolic_plan_does_not_contain_assumed_motor_commands(self) -> None:
        row = build_plan()[0].to_dict()
        self.assertNotIn("motor_1_input", row)
        self.assertNotIn("sp", row)


if __name__ == "__main__":
    unittest.main()

