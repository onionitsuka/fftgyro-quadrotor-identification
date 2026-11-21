import tempfile
import unittest
from pathlib import Path

from fftgyro_quad_id.experiment_runner.esc_readiness import (
    ReadinessStage,
    evaluate_inventory,
)


def _inventory(esc_rows: str, bluejay_rows: str = "") -> str:
    return f"""
[bluejay]
programmer = "blheli-usb-linker"
recovery_procedure = "docs/recovery.md"
propellers_removed = true
current_limited_supply = true
spare_esc_validation_passed = true
validated_bluejay_version = "0.21.0"
validated_target = "TEST_TARGET"
{bluejay_rows}
{esc_rows}
"""


def _esc(index: int, mcu: str = "BB21") -> str:
    return f"""
[[esc]]
esc_id = "esc_{index}"
motor_id = "motor_{index}"
part_number = "520081"
mcu = "{mcu}"
layout = "TEST"
original_firmware = "BLHeli_S 16.7"
settings_backup = "backups/esc_{index}.txt"
readout_success = true
"""


class EscReadinessTests(unittest.TestCase):
    def test_inspection_rejects_unknown_mcu_and_layout(self) -> None:
        source = Path("configs/experiment_runner/esc-inventory.example.toml")
        result = evaluate_inventory(source, ReadinessStage.INSPECT)
        self.assertFalse(result.ready)
        self.assertIn("esc_1:mcu_unknown", result.blockers)
        self.assertIn("esc_1:layout_unknown", result.blockers)

    def test_flash_one_requires_real_backup_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            inventory = root / "inventory.toml"
            inventory.write_text(
                _inventory("".join(_esc(index) for index in range(1, 5))),
                encoding="utf-8",
            )
            result = evaluate_inventory(inventory, ReadinessStage.FLASH_ONE)
            self.assertFalse(result.ready)
            self.assertIn("esc_1:settings_backup_file_missing", result.blockers)

    def test_deploy_all_passes_only_after_all_evidence_exists(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            backup_dir = root / "backups"
            backup_dir.mkdir()
            for index in range(1, 5):
                (backup_dir / f"esc_{index}.txt").write_text("backup", encoding="utf-8")
            inventory = root / "inventory.toml"
            inventory.write_text(
                _inventory("".join(_esc(index) for index in range(1, 5))),
                encoding="utf-8",
            )
            result = evaluate_inventory(inventory, ReadinessStage.DEPLOY_ALL)
            self.assertTrue(result.ready)

    def test_bb10_is_not_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            inventory = Path(temp) / "inventory.toml"
            rows = _esc(1, "BB10") + "".join(
                _esc(index) for index in range(2, 5)
            )
            inventory.write_text(_inventory(rows), encoding="utf-8")
            result = evaluate_inventory(inventory, ReadinessStage.INSPECT)
            self.assertIn("esc_1:unsupported_mcu:BB10", result.blockers)


if __name__ == "__main__":
    unittest.main()
