from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fftgyro_quad_id.experiment_runner.lgr101 import LgrFormatError, inspect_lgr_file


def _csv(*, rate: int = 100, filter_setting: str = "OFF", rows: str | None = None) -> bytes:
    data = rows or """0.0000,-0.066,0.098,-0.007,-0.0013,0.0006,-0.0006
0.0100,-0.093,0.100,0.043,-0.0009,-0.0006,0.0010
0.0200,0.132,-0.058,-0.001,0.0004,0.0009,-0.0014
"""
    text = f"""CFSLGR_DataFile
データ保存[個/秒],{rate},,Filter,{filter_setting}
定格,250,250,250,6,6,6
経過時間[sec],Fx[N],Fy[N],Fz[N],Mx[Nm],My[Nm],Mz[Nm]
{data}"""
    return text.replace("\n", "\r\n").encode("cp932")


class Lgr101Tests(unittest.TestCase):
    def _inspect(self, content: bytes, required_rate: float = 100):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "lgr.csv"
            path.write_bytes(content)
            return inspect_lgr_file(path, required_sample_rate_hz=required_rate)

    def test_real_lgr_shape_parses(self) -> None:
        report = self._inspect(_csv())
        self.assertTrue(report.ready)
        self.assertEqual(report.row_count, 3)
        self.assertAlmostEqual(report.observed_sample_rate_hz or 0, 100.0)
        self.assertEqual(report.ratings["fx"], 250.0)

    def test_old_100_hz_log_is_blocked_for_motor_transient(self) -> None:
        report = self._inspect(_csv(), required_rate=500)
        self.assertIn(
            "nominal_sample_rate_below_protocol_requirement", report.blockers
        )

    def test_filter_setting_is_preserved_as_warning(self) -> None:
        report = self._inspect(_csv(filter_setting="10Hz"))
        self.assertIn(
            "logger_low_pass_filter_enabled_record_setting_in_manifest",
            report.warnings,
        )

    def test_sample_gap_is_detected(self) -> None:
        rows = """0.0000,0,0,0,0,0,0
0.0100,0,0,0,0,0,0
0.0300,0,0,0,0,0,0
"""
        report = self._inspect(_csv(rows=rows))
        self.assertEqual(report.estimated_missing_samples, 1)
        self.assertIn("sample_gaps_detected", report.blockers)

    def test_eighty_percent_rating_is_rejected(self) -> None:
        rows = """0.0000,0,0,0,0,0,0
0.0100,200,0,0,0,0,0
"""
        report = self._inspect(_csv(rows=rows))
        self.assertIn("loadcell_at_or_above_80_percent", report.blockers)

    def test_bad_magic_is_rejected(self) -> None:
        with self.assertRaises(LgrFormatError):
            self._inspect(_csv().replace(b"CFSLGR_DataFile", b"other", 1))
