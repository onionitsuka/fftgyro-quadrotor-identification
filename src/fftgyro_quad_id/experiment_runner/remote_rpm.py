"""Windows-side Arduino RPM capture controlled by remote experiment events."""

from __future__ import annotations

import csv
import json
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import serial
from serial.tools import list_ports

from .remote_acquisition import RemoteEvent

ARDUINO_BAUD = 1_000_000
ARDUINO_USB_ID = (0x2341, 0x0043, "95033313031351A0E1E0")
MARKERS = frozenset({"MARK_START", "COMMAND_ON", "COMMAND_ACK", "COMMAND_OFF", "MARK_END"})
MINIMUM_RPM_EDGES = 2


def resolve_arduino_port(requested: str | None) -> str:
    ports = list(list_ports.comports())
    if requested is not None:
        match = next((item for item in ports if item.device == requested), None)
        if match is None:
            raise RuntimeError(f"Arduino is missing: {requested}")
        if (match.vid, match.pid) != ARDUINO_USB_ID[:2]:
            raise RuntimeError(
                f"{requested} is not the expected Arduino Uno: "
                f"vid={match.vid!r}, pid={match.pid!r}"
            )
        return requested

    matches = [
        item.device
        for item in ports
        if (item.vid, item.pid, item.serial_number) == ARDUINO_USB_ID
    ]
    if len(matches) != 1:
        raise RuntimeError(f"cannot uniquely auto-detect Arduino Uno: candidates={matches}")
    return matches[0]


@dataclass
class RunCapture:
    batch_id: str
    run_id: str
    payload: dict[str, Any]
    rows: list[tuple[str, str, str]] = field(default_factory=list)
    command_row_start: int | None = None
    command_on_micros: int | None = None
    command_off_micros: int | None = None


class ArduinoRPMService:
    """Own the Windows Arduino port and translate Mac events into markers."""

    def __init__(self, port: str, root: Path, baud: int = ARDUINO_BAUD) -> None:
        self.port = port
        self.root = root
        self.serial = serial.Serial(port, baud, timeout=0.05)
        self._condition = threading.Condition()
        self._capture: RunCapture | None = None
        self._error: str | None = None
        self._stopping = False
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        # Opening an Uno serial port resets it. Wait for setup() before reading.
        time.sleep(2.0)
        self.serial.reset_input_buffer()
        self._thread.start()

    def close(self) -> None:
        self._stopping = True
        self._thread.join(timeout=1.0)
        self.serial.close()

    def health(self) -> dict[str, Any]:
        with self._condition:
            return {
                "arduino_ready": self._error is None and self.serial.is_open,
                "arduino_port": self.port,
                "active_run_id": None if self._capture is None else self._capture.run_id,
                "error": self._error,
            }

    @staticmethod
    def _parse_line(line: bytes) -> tuple[str, str, str] | None:
        fields = line.decode("ascii", errors="ignore").strip().split(",")
        if len(fields) == 2 and fields[0] in {"R", "S", "O", "F"}:
            return fields[0], "", fields[1]
        if len(fields) == 3 and fields[0] in {"R", "S", "O", "F", "M"}:
            return fields[0], fields[1], fields[2]
        return None

    def _read_loop(self) -> None:
        while not self._stopping:
            try:
                record = self._parse_line(self.serial.readline())
            except (OSError, serial.SerialException) as error:
                with self._condition:
                    self._error = f"Arduino serial read failed: {error}"
                    self._condition.notify_all()
                return
            if record is None:
                continue
            with self._condition:
                if self._capture is not None:
                    self._capture.rows.append(record)
                    if record[0] == "O":
                        self._error = "Arduino ring-buffer overflow; run rejected"
                self._condition.notify_all()

    def _require_ready(self) -> None:
        if self._error is not None:
            raise RuntimeError(self._error)
        if not self.serial.is_open:
            raise RuntimeError("Arduino serial port is closed")

    @staticmethod
    def _require_event_matches_capture(
        event: RemoteEvent,
        capture: RunCapture | None,
    ) -> RunCapture:
        if capture is None:
            raise RuntimeError(f"{event.event_type} received with no active run")
        if (event.batch_id, event.run_id) != (capture.batch_id, capture.run_id):
            raise RuntimeError(
                f"{event.event_type} targets {event.batch_id}/{event.run_id}, but "
                f"active run is {capture.batch_id}/{capture.run_id}"
            )
        return capture

    def _write_marker(self, name: str, event: RemoteEvent) -> int:
        if name not in MARKERS:
            raise ValueError(name)
        with self._condition:
            self._require_ready()
            capture = self._require_event_matches_capture(event, self._capture)
            start_index = len(capture.rows)
            self.serial.write((name + "\n").encode("ascii"))
            self.serial.flush()
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline:
                self._require_ready()
                for record in capture.rows[start_index:]:
                    if record[0] == "M" and record[1] == name:
                        return int(record[2])
                self._condition.wait(timeout=min(0.1, deadline - time.monotonic()))
        raise RuntimeError(f"Arduino did not echo marker: {name}")

    def _start(self, event: RemoteEvent) -> dict[str, Any]:
        if event.run_id is None:
            raise RuntimeError("RUN_START has no run_id")
        run_dir = self.root / event.batch_id / "runs" / event.run_id
        if run_dir.exists():
            raise RuntimeError(f"refusing to reuse existing run directory: {run_dir}")
        with self._condition:
            self._require_ready()
            if self._capture is not None:
                raise RuntimeError(f"Arduino capture already active: {self._capture.run_id}")
            self._capture = RunCapture(event.batch_id, event.run_id, event.payload)
        micros = self._write_marker("MARK_START", event)
        return {
            "batch_id": event.batch_id,
            "run_id": event.run_id,
            "arduino_marker": "MARK_START",
            "arduino_micros": micros,
        }

    def _finish(self, event: RemoteEvent, *, aborted: bool = False) -> dict[str, Any]:
        with self._condition:
            capture = self._require_event_matches_capture(event, self._capture)
            self._require_ready()
            if not aborted and (
                capture.command_row_start is None
                or capture.command_on_micros is None
                or capture.command_off_micros is None
            ):
                raise RuntimeError("RUN_END received before COMMAND_ON and COMMAND_OFF")
        end_micros = self._write_marker("MARK_END", event)
        with self._condition:
            capture = self._require_event_matches_capture(event, self._capture)
            self._require_ready()
            rows = list(capture.rows)
            start = capture.command_row_start
            self._capture = None
        if start is None:
            start = len(rows)

        run_dir = self.root / capture.batch_id / "runs" / capture.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        rpm_path = run_dir / "rpm.csv"
        metadata_path = run_dir / "rpm_metadata.json"
        rpm_edges = sum(1 for event, _, _ in rows[start:] if event == "R")
        confirmed = rpm_edges >= MINIMUM_RPM_EDGES
        with rpm_path.open("x", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(("event", "label", "micros"))
            writer.writerows(rows)
        metadata_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "batch_id": capture.batch_id,
                    "run_id": capture.run_id,
                    "run": capture.payload,
                    "rpm_edges": rpm_edges,
                    "minimum_edges": MINIMUM_RPM_EDGES,
                    "execution_confirmed": confirmed,
                    "aborted": aborted,
                    "command_on_micros": capture.command_on_micros,
                    "command_off_micros": capture.command_off_micros,
                    "mark_end_micros": end_micros,
                    "created_utc": datetime.now(UTC).isoformat(),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return {
            "batch_id": capture.batch_id,
            "run_id": capture.run_id,
            "rpm_file": str(rpm_path),
            "metadata_file": str(metadata_path),
            "rpm_edges": rpm_edges,
            "minimum_edges": MINIMUM_RPM_EDGES,
            "execution_confirmed": confirmed,
            "aborted": aborted,
        }

    def handle_event(self, event: RemoteEvent) -> dict[str, Any]:
        if event.event_type == "RUN_START":
            return self._start(event)
        if event.event_type == "COMMAND_ON":
            with self._condition:
                capture = self._require_event_matches_capture(event, self._capture)
                if capture.command_on_micros is not None:
                    raise RuntimeError("COMMAND_ON already recorded for active run")
            micros = self._write_marker("COMMAND_ON", event)
            with self._condition:
                capture = self._require_event_matches_capture(event, self._capture)
                capture.command_row_start = len(capture.rows)
                capture.command_on_micros = micros
            return {
                "batch_id": event.batch_id,
                "run_id": event.run_id,
                "arduino_marker": "COMMAND_ON",
                "arduino_micros": micros,
            }
        if event.event_type == "COMMAND_OFF":
            with self._condition:
                capture = self._require_event_matches_capture(event, self._capture)
                if capture.command_on_micros is None:
                    raise RuntimeError("COMMAND_OFF received before COMMAND_ON")
                if capture.command_off_micros is not None:
                    raise RuntimeError("COMMAND_OFF already recorded for active run")
            micros = self._write_marker("COMMAND_OFF", event)
            with self._condition:
                capture = self._require_event_matches_capture(event, self._capture)
                capture.command_off_micros = micros
            return {
                "batch_id": event.batch_id,
                "run_id": event.run_id,
                "arduino_marker": "COMMAND_OFF",
                "arduino_micros": micros,
            }
        if event.event_type == "RUN_END":
            return self._finish(event)
        if event.event_type == "RUN_ABORT":
            return self._finish(event, aborted=True)
        return {}
