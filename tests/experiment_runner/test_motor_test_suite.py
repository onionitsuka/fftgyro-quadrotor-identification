import unittest
from collections import defaultdict

from fftgyro_quad_id.experiment_runner.motor_single import DOWNWARD_METHODS, build_plan
from fftgyro_quad_id.experiment_runner.motor_test_suite import (
    build_motor_test_runs,
    select_run_range,
)


class MotorTestSuiteTest(unittest.TestCase):
    def test_recommended_plan_becomes_31_contiguous_runs(self) -> None:
        runs = build_motor_test_runs(build_plan("recommended"))

        self.assertEqual(len(runs), 31)
        self.assertEqual([run.phase for run in runs].count("steady"), 15)
        self.assertEqual([run.phase for run in runs].count("dynamic"), 15)
        self.assertEqual([run.phase for run in runs].count("sync"), 1)


    def test_steady_sweep_preserves_ascent_and_controlled_descent(self) -> None:
        run = build_motor_test_runs(build_plan("recommended"))[0]

        self.assertEqual(run.segment_range, (1, 22))
        self.assertEqual(
            [step.percent for step in run.steps[:8]],
            [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
        )
        self.assertEqual(
            [step.percent for step in run.steps[8:]],
            [
                75.0, 70.0, 65.0, 60.0, 55.0, 50.0, 45.0,
                40.0, 35.0, 30.0, 25.0, 20.0, 15.0, 10.0,
            ],
        )
        self.assertEqual(run.duration_s, 90.5)


    def test_dynamic_waveform_is_kept_in_one_run(self) -> None:
        run = build_motor_test_runs(build_plan("recommended"))[15]

        self.assertEqual(run.phase, "dynamic")
        self.assertEqual(run.segment_range, (331, 415))
        self.assertEqual(run.steps[0].purpose, "dynamic_baseline")
        self.assertEqual(run.steps[0].percent, 20.0)
        self.assertEqual(run.maximum_percent, 80.0)
        self.assertTrue(60.0 <= run.duration_s <= 65.0)


    def test_range_selection_supports_resume_without_renumbering(self) -> None:
        runs = build_motor_test_runs(build_plan("recommended"))

        selected = select_run_range(runs, 7, 9)

        self.assertEqual([run.group for run in selected], [7, 8, 9])
        self.assertTrue(all(run.voltage_v == 15.7 for run in selected))


    def test_invalid_range_is_rejected(self) -> None:
        runs = build_motor_test_runs(build_plan("recommended"))

        with self.assertRaises(ValueError):
            select_run_range(runs, 9, 7)

    def test_extended_profile_reaches_100_with_controlled_descent(self) -> None:
        run = build_motor_test_runs(build_plan("recommended-extended-100"))[0]

        self.assertEqual(run.maximum_percent, 100.0)
        self.assertEqual(
            [step.percent for step in run.steps[8:12]],
            [85.0, 90.0, 95.0, 100.0],
        )
        descent = [step.percent for step in run.steps[12:]]
        self.assertEqual(descent[0], 95.0)
        self.assertEqual(descent[-1], 10.0)
        self.assertTrue(
            all(
                left - right == 5.0
                for left, right in zip(descent, descent[1:], strict=False)
            )
        )

    def test_downward_suite_becomes_33_primary_voltage_runs(self) -> None:
        all_runs = build_motor_test_runs(
            build_plan("recommended", methods=DOWNWARD_METHODS)
        )
        self.assertEqual(len(all_runs), 33)
        runs = tuple(run for run in all_runs if run.voltage_v == 16.2)

        self.assertEqual(len(runs), 33)
        self.assertEqual(runs[0].method_id, "thermal-reference-cold")
        self.assertEqual(runs[-1].method_id, "thermal-reference-warm")
        self.assertEqual(
            [
                step.percent
                for step in runs[3].steps
                if step.purpose != "settling_approach"
            ],
            [55.0, 50.0],
        )

        symmetric_15 = next(
            run
            for run in runs
            if run.method_id == "symmetric-step"
            and run.condition_id == "span15_around50"
        )
        self.assertEqual(
            [
                step.percent
                for step in symmetric_15.steps
                if step.purpose != "settling_approach"
            ],
            [50.0, 65.0, 50.0],
        )
        self.assertEqual(runs[3].steps[0].percent, 10.0)
        self.assertTrue(
            all(
                right.percent - left.percent <= 10.0 + 1e-9
                for left, right in zip(
                    runs[3].steps,
                    runs[3].steps[1:],
                    strict=False,
                )
                if right.purpose == "settling_approach"
            )
        )
        final_fixed = [run for run in runs if run.method_id == "down-final-fixed"]
        self.assertEqual(
            [(run.repetition, run.condition_id) for run in final_fixed],
            [
                (1, "drop05_to50"),
                (1, "drop15_to50"),
                (2, "drop15_to50"),
                (2, "drop05_to50"),
                (3, "drop05_to50"),
                (3, "drop15_to50"),
            ],
        )

    def test_only_thermal_brackets_intentionally_repeat_a_numeric_run(self) -> None:
        runs = tuple(
            run
            for run in build_motor_test_runs(
                build_plan("recommended", methods=DOWNWARD_METHODS)
            )
            if run.voltage_v == 16.2
        )
        owners: dict[tuple[tuple[float | None, float, float], ...], set[str]] = (
            defaultdict(set)
        )
        for run in runs:
            signature = tuple(
                (step.input_start_percent, step.percent, step.duration_s)
                for step in run.steps
                if step.purpose != "settling_approach"
            )
            owners[signature].add(run.method_id)

        repeated_across_methods = {
            frozenset(methods) for methods in owners.values() if len(methods) > 1
        }
        self.assertEqual(
            repeated_across_methods,
            {
                frozenset(
                    {"thermal-reference-cold", "thermal-reference-warm"}
                )
            },
        )


if __name__ == "__main__":
    unittest.main()
