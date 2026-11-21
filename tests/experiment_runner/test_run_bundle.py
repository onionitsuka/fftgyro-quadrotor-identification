from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fftgyro_quad_id.experiment_runner.flight_log import ArmWindow, FlightExtractionReport
from fftgyro_quad_id.experiment_runner.run_bundle import (
    NumericSeries,
    create_run_from_config,
    finalize_run,
    normalize_arduino,
    normalize_hioki,
    write_aligned_csv,
)


def _leptrino_csv() -> bytes:
    data = "\n".join(
        f"{index / 10:.1f},0,0,0,0,0,0" for index in range(11)
    )
    text = f"""CFSLGR_DataFile
データ保存[個/秒],10,,Filter,OFF
定格,250,250,250,6,6,6
経過時間[sec],Fx[N],Fy[N],Fz[N],Mx[Nm],My[Nm],Mz[Nm]
{data}
"""
    return text.encode("cp932")


class RunBundleTests(unittest.TestCase):
    def test_finalize_requires_and_bundles_all_four_sources(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "acquisition.toml"
            config.write_text(
                f'''[session]
output_root = "{(root / 'runs').as_posix()}"
protocol_id = "integrated"
protocol_version = "1"
profile = "test"
operator = "operator"
hardware_config_id = "hardware"
esc_inventory_id = "esc"
[flight]
motor_channels = [9, 10, 11, 12]
[alignment]
sample_rate_hz = 10.0
duration_tolerance_s = 0.01
[arduino]
marks_per_revolution = 2
require_arm_and_disarm_markers = true
[hioki]
time_column = "t"
sync_column = "sync"
sync_threshold = 0.5
require_arm_and_disarm_markers = true
[hioki.columns]
voltage_v = "voltage"
''',
                encoding="utf-8",
            )
            flight = root / "flight.BIN"
            flight.write_bytes(b"fake-bin")
            leptrino = root / "leptrino.csv"
            leptrino.write_bytes(_leptrino_csv())
            hioki = root / "hioki.csv"
            hioki.write_text(
                "t,voltage,sync\n0.0,14.8,0\n0.1,14.8,1\n1.0,14.7,0\n1.1,14.7,1\n",
                encoding="utf-8",
            )
            arduino = root / "arduino.csv"
            arduino.write_text(
                "event,micros\nS,100000\nR,200000\nR,300000\nS,1100000\n",
                encoding="utf-8",
            )
            session = create_run_from_config(config, "run_001")

            def fake_extract(*args, **kwargs):
                output = args[1]
                output.write_text(
                    "time_s,time_us,motor_1_output_raw,motor_2_output_raw,"
                    "motor_3_output_raw,motor_4_output_raw,source_message\n"
                    "0.0,1000000,1000,1000,1000,1000,MRAW\n"
                    "1.0,2000000,1200,1200,1200,1200,MRAW\n",
                    encoding="utf-8",
                )
                return FlightExtractionReport(
                    ArmWindow(1_000_000, 2_000_000),
                    1,
                    2,
                    "MRAW",
                    100.0,
                    (9, 10, 11, 12),
                    (),
                )

            with patch(
                "fftgyro_quad_id.experiment_runner.run_bundle.extract_ardupilot_flight_csv",
                side_effect=fake_extract,
            ):
                report = finalize_run(
                    config,
                    session.root,
                    flight_path=flight,
                    leptrino_path=leptrino,
                    hioki_path=hioki,
                    arduino_path=arduino,
                )
            self.assertEqual(report.status, "completed")
            self.assertTrue((session.root / "processed" / "aligned.csv").is_file())
            self.assertTrue((session.root / "raw" / "pixhawk" / "flight.BIN").is_file())
            self.assertTrue((session.root / "source_manifest.json").is_file())
            self.assertFalse((session.root / "RUN_IN_PROGRESS").exists())

    def test_arduino_uses_first_and_last_sync_marker(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "raw.csv"
            raw.write_text(
                "event,micros\nS,1000000\nR,1100000\nR,1200000\nS,2000000\n",
                encoding="utf-8",
            )
            series, duration = normalize_arduino(
                raw,
                root / "normalized.csv",
                marks_per_revolution=2,
                require_two_sync_markers=True,
            )
            self.assertEqual(duration, 1.0)
            self.assertEqual(len(series.rows), 1)
            self.assertEqual(series.rows[0][1], (300.0,))

    def test_arduino_overflow_rejects_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "raw.csv"
            raw.write_text("event,micros\nS,1\nO,1\nS,2\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "overflow"):
                normalize_arduino(
                    raw,
                    root / "normalized.csv",
                    marks_per_revolution=2,
                    require_two_sync_markers=True,
                )

    def test_hioki_sync_edges_define_zero_and_duration(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "hioki.csv"
            with raw.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(("t", "V", "Sync"))
                writer.writerows(
                    ((0.0, 14.8, 0), (0.1, 14.7, 1), (1.0, 14.6, 0), (1.1, 14.5, 1))
                )
            series, duration = normalize_hioki(
                raw,
                root / "normalized.csv",
                {
                    "time_column": "t",
                    "sync_column": "Sync",
                    "sync_threshold": 0.5,
                    "columns": {"voltage_v": "V"},
                    "require_arm_and_disarm_markers": True,
                },
            )
            self.assertAlmostEqual(duration or 0.0, 1.0)
            self.assertEqual(series.rows[0][0], 0.0)

    def test_alignment_uses_previous_for_motor_and_linear_for_analog(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "aligned.csv"
            write_aligned_csv(
                path,
                (
                    NumericSeries(
                        "flight", ("motor",), ((0.0, (1000.0,)), (1.0, (2000.0,))), "previous"
                    ),
                    NumericSeries(
                        "hioki", ("voltage",), ((0.0, (10.0,)), (1.0, (20.0,)))
                    ),
                ),
                duration_s=1.0,
                sample_rate_hz=2.0,
            )
            with path.open(encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[1]["flight_motor"], "1000.0")
            self.assertEqual(rows[1]["hioki_voltage"], "15.0")


if __name__ == "__main__":
    unittest.main()
