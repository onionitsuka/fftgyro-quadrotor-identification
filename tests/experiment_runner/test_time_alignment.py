from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from fftgyro_quad_id.experiment_runner.time_alignment import align_hioki_current


class TimeAlignmentTests(unittest.TestCase):
    def test_current_onset_defines_offline_zero_and_records_uncertainty(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            timeline = root / "timeline.json"
            timeline.write_text(
                json.dumps(
                    {"events": [
                        {"name": "COMMAND_ON", "host_utc": "2026-07-18T13:50:12+09:00"},
                        {"name": "COMMAND_OFF", "host_utc": "2026-07-18T13:50:14+09:00"},
                    ]}
                ),
                encoding="utf-8",
            )
            hioki = root / "hioki.csv"
            rows = ['"Time","CH1-1-2[A]"']
            for index in range(41):
                current = 0.05 if index < 11 or index > 30 else 0.55
                rows.append(f"'26-07-18 13:50:{10 + index / 10:05.2f}s,{current}")
            hioki.write_text("\n".join(rows) + "\n", encoding="utf-8")
            result = align_hioki_current(hioki, timeline)
            self.assertAlmostEqual(result.host_to_hioki_offset_s, -0.9)
            self.assertAlmostEqual(result.sample_period_s, 0.1)
            self.assertFalse(result.phase_analysis_allowed)
            self.assertGreaterEqual(result.estimated_sync_uncertainty_s, 0.1)


if __name__ == "__main__":
    unittest.main()
