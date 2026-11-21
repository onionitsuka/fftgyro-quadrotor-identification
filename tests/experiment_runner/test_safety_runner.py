import unittest

from fftgyro_quad_id.experiment_runner.adapters import SimulationAdapter
from fftgyro_quad_id.experiment_runner.motor_single import build_plan
from fftgyro_quad_id.experiment_runner.runner import ExperimentRunner, RunStatus
from fftgyro_quad_id.experiment_runner.safety import (
    SafetyLimits,
    SafetyObservation,
    evaluate_safety,
)


class SafetyGateTests(unittest.TestCase):
    def test_nominal_observation_passes(self) -> None:
        result = evaluate_safety(
            SafetyObservation(
                loadcell_utilization=0.40,
                baseline_voltage_v=15.7,
                voltage_v=15.6,
                telemetry_missing_fraction=0.005,
                rpm_reference_error_fraction=0.01,
                sync_residual_s=0.0005,
            ),
            SafetyLimits(),
        )
        self.assertTrue(result.passed)

    def test_pdf_stop_conditions_are_reported_together(self) -> None:
        result = evaluate_safety(
            SafetyObservation(
                emergency_stop=True,
                loadcell_utilization=0.81,
                baseline_voltage_v=15.0,
                voltage_v=14.0,
                telemetry_missing_fraction=0.02,
                rpm_reference_error_fraction=0.03,
                sync_residual_s=0.002,
                motor_temperature_c=81.0,
                esc_temperature_c=91.0,
            ),
            SafetyLimits(
                motor_temperature_warning_c=80.0,
                esc_temperature_warning_c=90.0,
            ),
        )
        self.assertFalse(result.passed)
        self.assertEqual(len(result.reasons), 8)


class RunnerTests(unittest.TestCase):
    def test_simulation_completes_without_physical_output(self) -> None:
        segments = build_plan("minimum")[:3]
        adapter = SimulationAdapter()
        result = ExperimentRunner(adapter).run_simulation(segments)
        self.assertEqual(result.status, RunStatus.COMPLETED)
        self.assertEqual(result.completed_segments, 3)
        self.assertEqual(adapter.events[-1], "stop_all")

    def test_gate_failure_stops_before_next_segment(self) -> None:
        segments = build_plan("minimum")[:3]
        adapter = SimulationAdapter(
            observations=[
                SafetyObservation(),
                SafetyObservation(),
                SafetyObservation(emergency_stop=True),
            ]
        )
        result = ExperimentRunner(adapter).run_simulation(segments)
        self.assertEqual(result.status, RunStatus.STOPPED)
        self.assertEqual(result.completed_segments, 1)
        self.assertEqual(result.stopped_before_segment, segments[1].segment_id)
        self.assertEqual(adapter.events[-1], "stop_all")


if __name__ == "__main__":
    unittest.main()

