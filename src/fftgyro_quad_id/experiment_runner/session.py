from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic_ns
from typing import Any


RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
RAW_SOURCES = (
    "pixhawk",
    "pi_executor",
    "fftgyro_acquisition",
    "loadcell_lgr101u",
    "aux_lr8450",
    "sync_uno",
    "wind_tunnel",
    "operator",
)


@dataclass(frozen=True, slots=True)
class RunMetadata:
    run_id: str
    protocol_id: str
    protocol_version: str
    profile: str
    created_utc: str
    operator: str
    hardware_config_id: str
    esc_inventory_id: str
    fixture_id: str | None = None
    notes: str = ""


class ExperimentSession:
    """Append-only run metadata and event journal."""

    def __init__(self, root: Path, metadata: RunMetadata) -> None:
        self.root = root
        self.metadata = metadata
        self.events_path = root / "events.jsonl"
        self._finalized = False

    @classmethod
    def create(
        cls,
        output_root: Path,
        *,
        run_id: str,
        protocol_id: str,
        protocol_version: str,
        profile: str,
        operator: str,
        hardware_config_id: str,
        esc_inventory_id: str,
        fixture_id: str | None = None,
        notes: str = "",
    ) -> "ExperimentSession":
        if not RUN_ID_PATTERN.fullmatch(run_id):
            raise ValueError(
                "run_id must be 1-64 letters, digits, underscores or hyphens"
            )
        required = {
            "protocol_id": protocol_id,
            "protocol_version": protocol_version,
            "profile": profile,
            "operator": operator,
            "hardware_config_id": hardware_config_id,
            "esc_inventory_id": esc_inventory_id,
        }
        missing = [name for name, value in required.items() if not value.strip()]
        if missing:
            raise ValueError(f"required metadata missing: {', '.join(missing)}")

        run_root = output_root / run_id
        # Atomic mkdir is the overwrite guard. A previous run is never reused.
        run_root.mkdir(parents=True, exist_ok=False)
        try:
            raw_root = run_root / "raw"
            raw_root.mkdir()
            for source in RAW_SOURCES:
                (raw_root / source).mkdir()

            metadata = RunMetadata(
                run_id=run_id,
                protocol_id=protocol_id,
                protocol_version=protocol_version,
                profile=profile,
                created_utc=datetime.now(UTC).isoformat(),
                operator=operator,
                hardware_config_id=hardware_config_id,
                esc_inventory_id=esc_inventory_id,
                fixture_id=fixture_id,
                notes=notes,
            )
            _write_new_json(run_root / "run.json", asdict(metadata))
            session = cls(run_root, metadata)
            session.append_event("session_created", {"raw_sources": RAW_SOURCES})
            return session
        except BaseException:
            # Leave evidence of a partially-created run rather than deleting it.
            failure_path = run_root / "SESSION_CREATION_FAILED"
            failure_path.touch(exist_ok=True)
            raise

    @classmethod
    def open_existing(cls, run_root: Path) -> "ExperimentSession":
        metadata_path = run_root / "run.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"run_metadata_missing:{metadata_path}")
        if (run_root / "final.json").exists():
            raise RuntimeError(f"run_already_finalized:{run_root}")
        document = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata = RunMetadata(**document)
        return cls(run_root, metadata)

    def append_event(self, event_type: str, payload: dict[str, Any]) -> None:
        if self._finalized:
            raise RuntimeError("cannot append to a finalized session")
        if not event_type or any(character.isspace() for character in event_type):
            raise ValueError("event_type must be a non-empty token")
        record = {
            "run_id": self.metadata.run_id,
            "event_type": event_type,
            "utc": datetime.now(UTC).isoformat(),
            "monotonic_ns": monotonic_ns(),
            "payload": payload,
        }
        encoded = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        with self.events_path.open("a", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())

    def raw_source_dir(self, source: str) -> Path:
        if source not in RAW_SOURCES:
            raise ValueError(f"unknown raw source: {source}")
        return self.root / "raw" / source

    def finalize(self, *, status: str, summary: dict[str, Any]) -> Path:
        if self._finalized:
            raise RuntimeError("session already finalized")
        if status not in {"completed", "stopped", "aborted", "failed"}:
            raise ValueError(f"invalid final status: {status}")
        self.append_event("session_finalizing", {"status": status})
        final_path = self.root / "final.json"
        _write_new_json(
            final_path,
            {
                "run_id": self.metadata.run_id,
                "status": status,
                "finalized_utc": datetime.now(UTC).isoformat(),
                "summary": summary,
            },
        )
        self._finalized = True
        return final_path


def _write_new_json(path: Path, document: dict[str, Any]) -> None:
    encoded = json.dumps(document, ensure_ascii=False, indent=2) + "\n"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())

