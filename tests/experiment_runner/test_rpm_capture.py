import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from fftgyro_quad_id.experiment_runner.rpm_capture import (
    capture_uno_serial,
    parse_uno_line,
    reject_inserted_false_edges,
    rpm_from_same_mark_periods,
    rpm_from_edges,
)


class RpmCaptureTests(unittest.TestCase):
    def test_parses_rpm_and_sync_records(self) -> None:
        rpm = parse_uno_line("R,123\n")
        sync = parse_uno_line("S,456\n")
        self.assertIsNotNone(rpm)
        self.assertIsNotNone(sync)
        assert rpm is not None
        assert sync is not None
        self.assertEqual(rpm.kind, "R")
        self.assertEqual(sync.micros, 456)
        self.assertIsNone(parse_uno_line("event,micros\n"))

    def test_two_marks_convert_edge_period_to_rpm(self) -> None:
        samples = rpm_from_edges((1000, 4000, 7000), marks_per_revolution=2)
        self.assertEqual(len(samples), 2)
        self.assertAlmostEqual(samples[0].rpm, 10_000.0)

    def test_named_marker_is_parsed(self) -> None:
        marker = parse_uno_line("M,COMMAND_ON,12345\n")
        self.assertIsNotNone(marker)
        assert marker is not None
        self.assertEqual(marker.label, "COMMAND_ON")
        self.assertEqual(marker.micros, 12345)

    def test_too_short_false_edge_is_ignored_without_moving_reference(self) -> None:
        samples = rpm_from_edges(
            (10_000, 11_000, 30_000),
            marks_per_revolution=2,
            minimum_edge_interval_us=5_000,
        )
        self.assertEqual(len(samples), 1)
        self.assertAlmostEqual(samples[0].rpm, 1_500.0)

    def test_inserted_edge_is_rejected_and_same_mark_period_cancels_spacing(self) -> None:
        raw = (0, 22_000, 22_900, 44_500, 67_400, 89_000)
        cleaned, rejected = reject_inserted_false_edges(raw)
        self.assertEqual(rejected, (22_900,))
        self.assertEqual(cleaned, (0, 22_000, 44_500, 67_400, 89_000))
        rpm = rpm_from_same_mark_periods(cleaned, marks_per_revolution=2)
        self.assertTrue(all(1_300 < sample.rpm < 1_400 for sample in rpm))

    def test_micros_rollover_is_unwrapped_for_period(self) -> None:
        samples = rpm_from_edges((2**32 - 1000, 2000), marks_per_revolution=2)
        self.assertAlmostEqual(samples[0].rpm, 10_000.0)

    def test_duplicate_edge_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate_edge_timestamp"):
            rpm_from_edges((123, 123))

    def test_capture_never_overwrites_an_existing_raw_file(self) -> None:
        with TemporaryDirectory() as directory:
            output = Path(directory) / "rpm_edges.csv"
            output.write_text("preserve me\n", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "already_exists"):
                capture_uno_serial("unused", output, duration_s=1.0)
            self.assertEqual(output.read_text(encoding="utf-8"), "preserve me\n")


if __name__ == "__main__":
    unittest.main()
