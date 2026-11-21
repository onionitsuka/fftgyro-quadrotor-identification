from __future__ import annotations

import hashlib
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .esc_readiness import SUPPORTED_BLUEJAY_MCUS


NORMAL_READ_METHOD = "arduino_blheli_bootloader_interface_l"


@dataclass(frozen=True, slots=True)
class ReadAttempt:
    attempt: int
    esc_id: str
    part_number: str
    mcu: str
    layout: str
    firmware: str
    timestamp: str
    settings_file: str
    screenshot_file: str
    debug_log_file: str


@dataclass(frozen=True, slots=True)
class EscReadoutResult:
    ready: bool
    method: str
    attempt_count: int
    esc_id: str | None
    mcu: str | None
    layout: str | None
    firmware: str | None
    evidence_sha256: tuple[tuple[str, str], ...]
    blockers: tuple[str, ...]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["evidence_sha256"] = [
            {"path": path, "sha256": digest} for path, digest in self.evidence_sha256
        ]
        row["blockers"] = list(self.blockers)
        row["warnings"] = list(self.warnings)
        return row


def _text(value: Any) -> str:
    return str(value or "").strip()


def _load(path: Path) -> tuple[dict[str, Any], list[ReadAttempt]]:
    with path.open("rb") as stream:
        document = tomllib.load(stream)
    interface = document.get("interface", {})
    attempts = [
        ReadAttempt(
            attempt=int(raw.get("attempt", 0)),
            esc_id=_text(raw.get("esc_id")),
            part_number=_text(raw.get("part_number")),
            mcu=_text(raw.get("mcu")).upper(),
            layout=_text(raw.get("layout")).upper(),
            firmware=_text(raw.get("firmware")),
            timestamp=_text(raw.get("timestamp")),
            settings_file=_text(raw.get("settings_file")),
            screenshot_file=_text(raw.get("screenshot_file")),
            debug_log_file=_text(raw.get("debug_log_file")),
        )
        for raw in document.get("read", [])
    ]
    return interface, attempts


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evaluate_readout(receipt_path: Path) -> EscReadoutResult:
    interface, attempts = _load(receipt_path)
    blockers: list[str] = []
    warnings: list[str] = []
    evidence: list[tuple[str, str]] = []
    method = _text(interface.get("method"))

    if method != NORMAL_READ_METHOD:
        blockers.append("normal_first_read_requires_arduino_blheli_bootloader_interface_l")
    if _text(interface.get("interface_type")).upper() != "L":
        blockers.append("blhelisuite_interface_type_l_not_confirmed")
    if "UNO" not in _text(interface.get("board")).upper():
        blockers.append("arduino_uno_board_not_confirmed")
    if _text(interface.get("mcu")).upper() != "ATMEGA328P":
        blockers.append("atmega328p_not_confirmed")
    if not _text(interface.get("interface_firmware")):
        blockers.append("uno_interface_firmware_name_missing")
    signal_pin = interface.get("signal_pin")
    if not isinstance(signal_pin, int) or not 0 <= signal_pin <= 19:
        blockers.append("uno_signal_pin_missing_or_invalid")
    for key in (
        "common_ground_confirmed",
        "esc_external_power_confirmed",
        "propellers_removed",
        "current_limited_supply",
        "single_esc_connected",
    ):
        if not bool(interface.get(key, False)):
            blockers.append(f"{key}_not_confirmed")
    if bool(interface.get("c2_pads_used", False)):
        blockers.append("c2_direct_programming_is_not_the_normal_first_read_path")

    if len(attempts) != 3:
        blockers.append(f"exactly_3_read_attempts_required:found_{len(attempts)}")
    if {item.attempt for item in attempts} != {1, 2, 3}:
        blockers.append("read_attempt_numbers_must_be_1_2_3")

    required_fields = (
        "esc_id",
        "part_number",
        "mcu",
        "layout",
        "firmware",
        "timestamp",
    )
    for item in attempts:
        for field in required_fields:
            if not getattr(item, field):
                blockers.append(f"read_{item.attempt}:{field}_missing")
        for field in ("settings_file", "screenshot_file", "debug_log_file"):
            relative = getattr(item, field)
            if not relative:
                blockers.append(f"read_{item.attempt}:{field}_missing")
                continue
            evidence_path = receipt_path.parent / relative
            if not evidence_path.is_file() or evidence_path.stat().st_size == 0:
                blockers.append(f"read_{item.attempt}:{field}_not_found_or_empty")
            else:
                evidence.append((relative, _sha256(evidence_path)))

    for field in ("esc_id", "part_number", "mcu", "layout", "firmware"):
        values = {getattr(item, field) for item in attempts if getattr(item, field)}
        if len(values) > 1:
            blockers.append(f"three_reads_disagree:{field}")

    mcu = attempts[0].mcu if attempts and attempts[0].mcu else None
    if mcu and mcu not in SUPPORTED_BLUEJAY_MCUS:
        blockers.append(f"unsupported_bluejay_mcu:{mcu}")
    if attempts and len({item.timestamp for item in attempts if item.timestamp}) < 3:
        warnings.append("read_attempt_timestamps_are_not_unique")

    return EscReadoutResult(
        ready=not blockers,
        method=method,
        attempt_count=len(attempts),
        esc_id=attempts[0].esc_id if attempts else None,
        mcu=mcu,
        layout=attempts[0].layout if attempts else None,
        firmware=attempts[0].firmware if attempts else None,
        evidence_sha256=tuple(evidence),
        blockers=tuple(blockers),
        warnings=tuple(warnings),
    )
