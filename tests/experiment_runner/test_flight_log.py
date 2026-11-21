from __future__ import annotations

import unittest

from fftgyro_quad_id.experiment_runner.flight_log import (
    extract_motor_outputs,
    find_arm_windows,
)


class FlightLogTests(unittest.TestCase):
    def test_extracts_only_complete_arm_window(self) -> None:
        records = (
            {"mavpackettype": "RCOU", "TimeUS": 900_000, "C9": 1000},
            {"mavpackettype": "EV", "TimeUS": 1_000_000, "Id": 10},
            {"mavpackettype": "RCOU", "TimeUS": 1_100_000, "C9": 1100},
            {"mavpackettype": "RCOU", "TimeUS": 1_200_000, "C9": 1200},
            {"mavpackettype": "EV", "TimeUS": 2_000_000, "Id": 11},
            {"mavpackettype": "RCOU", "TimeUS": 2_100_000, "C9": 1000},
        )
        samples, report = extract_motor_outputs(records, motor_channels=(9,))
        self.assertEqual([sample.values for sample in samples], [(1100.0,), (1200.0,)])
        self.assertEqual(report.arm_window.duration_s, 1.0)
        self.assertEqual(report.source_message, "RCOU")
        self.assertIn("motor_output_rate_below_50Hz", report.warnings)

    def test_prefers_100hz_custom_motor_log(self) -> None:
        records = [
            {"mavpackettype": "EV", "TimeUS": 0, "Id": 10},
            {"mavpackettype": "RCOU", "TimeUS": 10_000, "C9": 1500},
        ]
        for index in range(101):
            records.append(
                {
                    "mavpackettype": "MRAW",
                    "TimeUS": index * 10_000,
                    "M1": 1100,
                    "M2": 1200,
                    "M3": 1300,
                    "M4": 1400,
                }
            )
        records.append({"mavpackettype": "EV", "TimeUS": 1_000_000, "Id": 11})
        samples, report = extract_motor_outputs(records)
        self.assertEqual(report.source_message, "MRAW")
        self.assertAlmostEqual(report.observed_rate_hz or 0.0, 100.0)
        self.assertEqual(samples[0].values, (1100.0, 1200.0, 1300.0, 1400.0))

    def test_duplicate_arm_record_and_event_are_collapsed(self) -> None:
        records = (
            {"mavpackettype": "ARM", "TimeUS": 100, "ArmState": 1},
            {"mavpackettype": "EV", "TimeUS": 101, "Id": 10},
            {"mavpackettype": "ARM", "TimeUS": 200, "ArmState": 0},
            {"mavpackettype": "EV", "TimeUS": 201, "Id": 11},
        )
        windows = find_arm_windows(records)
        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0].arm_time_us, 100)
        self.assertEqual(windows[0].disarm_time_us, 200)

    def test_disarm_rotated_log_infers_start_from_first_armed_mraw(self) -> None:
        records = (
            {"mavpackettype": "MRAW", "TimeUS": 1_000, "M1": 1000},
            {"mavpackettype": "MRAW", "TimeUS": 11_000, "M1": 1100},
            {"mavpackettype": "ARM", "TimeUS": 20_000, "ArmState": 0},
            {"mavpackettype": "EV", "TimeUS": 20_001, "Id": 11},
        )
        samples, report = extract_motor_outputs(records)
        self.assertEqual(report.arm_window.arm_time_us, 1_000)
        self.assertEqual(report.arm_window.disarm_time_us, 20_000)
        self.assertEqual(len(samples), 2)
        self.assertIn("arm_boundary_inferred_from_first_armed_mraw", report.warnings)

    def test_missing_disarm_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "complete_arm_window_not_found"):
            extract_motor_outputs(
                (
                    {"mavpackettype": "EV", "TimeUS": 100, "Id": 10},
                    {"mavpackettype": "RCOU", "TimeUS": 110, "C9": 1000},
                ),
                motor_channels=(9,),
            )


if __name__ == "__main__":
    unittest.main()
