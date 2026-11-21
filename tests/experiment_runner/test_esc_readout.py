from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fftgyro_quad_id.experiment_runner.esc_readout import evaluate_readout


def _interface(*, method: str = "arduino_blheli_bootloader_interface_l") -> str:
    return f"""
[interface]
method = "{method}"
board = "Arduino Uno R3"
mcu = "ATmega328P"
interface_firmware = "Arduino BLHeli Bootloader 16.7"
interface_type = "L"
signal_pin = 11
common_ground_confirmed = true
esc_external_power_confirmed = true
propellers_removed = true
current_limited_supply = true
single_esc_connected = true
c2_pads_used = false
"""


def _read(attempt: int, *, layout: str = "G_H_30", mcu: str = "BB21") -> str:
    return f"""
[[read]]
attempt = {attempt}
esc_id = "esc_1"
part_number = "520081"
mcu = "{mcu}"
layout = "{layout}"
firmware = "BLHeli_S 16.7"
timestamp = "2026-07-16T10:0{attempt}:00+09:00"
settings_file = "evidence/read_{attempt}.txt"
screenshot_file = "evidence/read_{attempt}.png"
debug_log_file = "evidence/read_{attempt}.log"
"""


class EscReadoutTests(unittest.TestCase):
    def _receipt(self, root: Path, rows: str, interface: str | None = None) -> Path:
        evidence = root / "evidence"
        evidence.mkdir()
        for attempt in (1, 2, 3):
            for suffix in ("txt", "png", "log"):
                (evidence / f"read_{attempt}.{suffix}").write_bytes(
                    f"evidence-{attempt}-{suffix}".encode()
                )
        receipt = root / "receipt.toml"
        receipt.write_text((interface or _interface()) + rows, encoding="utf-8")
        return receipt

    def test_three_consistent_reads_pass(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            receipt = self._receipt(root, "".join(_read(i) for i in (1, 2, 3)))
            result = evaluate_readout(receipt)
            self.assertTrue(result.ready)
            self.assertEqual(result.attempt_count, 3)
            self.assertEqual(result.mcu, "BB21")
            self.assertEqual(len(result.evidence_sha256), 9)

    def test_disagreeing_layout_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = _read(1) + _read(2, layout="A_H_20") + _read(3)
            result = evaluate_readout(self._receipt(root, rows))
            self.assertIn("three_reads_disagree:layout", result.blockers)

    def test_c2_is_not_accepted_as_normal_first_read(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            interface = _interface(method="arduino_c2_direct").replace(
                "c2_pads_used = false", "c2_pads_used = true"
            )
            rows = "".join(_read(i) for i in (1, 2, 3))
            result = evaluate_readout(self._receipt(root, rows, interface))
            self.assertIn(
                "normal_first_read_requires_arduino_blheli_bootloader_interface_l",
                result.blockers,
            )
            self.assertIn(
                "c2_direct_programming_is_not_the_normal_first_read_path",
                result.blockers,
            )

    def test_bb10_blocks_bluejay(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = "".join(_read(i, mcu="BB10") for i in (1, 2, 3))
            result = evaluate_readout(self._receipt(root, rows))
            self.assertIn("unsupported_bluejay_mcu:BB10", result.blockers)
