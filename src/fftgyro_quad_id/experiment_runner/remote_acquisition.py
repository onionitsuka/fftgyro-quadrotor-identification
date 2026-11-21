"""Cross-computer event transport for the experiment acquisition workflow.

The transport intentionally carries coordination events only.  It does not
pretend that LAN delivery provides hardware-grade synchronisation; the Windows
receiver records both sender and receiver clocks so the residual uncertainty
remains explicit in the experiment record.
"""

from __future__ import annotations

import hmac
import json
import os
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

PROTOCOL = "fftgyro-remote-acquisition"
PROTOCOL_VERSION = 2
EVENT_TYPES = frozenset(
    {
        "BATCH_START",
        "RUN_START",
        "COMMAND_ON",
        "COMMAND_OFF",
        "RUN_END",
        "RUN_ABORT",
        "BATCH_END",
    }
)
_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
_MAX_BODY_BYTES = 1_048_576


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds")


def validate_identifier(value: str, label: str) -> str:
    if not _ID_PATTERN.fullmatch(value):
        raise ValueError(
            f"invalid {label}: use 1-96 ASCII letters, digits, '.', '_' or '-'"
        )
    return value


@dataclass(frozen=True)
class RemoteEvent:
    batch_id: str
    event_type: str
    sequence: int
    sender_utc: str
    sender_monotonic_ns: int
    run_id: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    protocol: str = PROTOCOL
    protocol_version: int = PROTOCOL_VERSION

    @property
    def event_id(self) -> str:
        scope = self.run_id if self.run_id is not None else "__batch__"
        return f"{self.batch_id}:{scope}:{self.event_type}:{self.sequence}"

    def validate(self) -> None:
        if self.protocol != PROTOCOL or self.protocol_version != PROTOCOL_VERSION:
            raise ValueError("unsupported remote-acquisition protocol")
        validate_identifier(self.batch_id, "batch_id")
        if self.run_id is not None:
            validate_identifier(self.run_id, "run_id")
        if self.event_type not in EVENT_TYPES:
            raise ValueError(f"unsupported event_type: {self.event_type}")
        if self.sequence < 1:
            raise ValueError("sequence must be positive")
        if self.event_type in {
            "RUN_START", "COMMAND_ON", "COMMAND_OFF", "RUN_END",
            "RUN_ABORT",
        }:
            if self.run_id is None:
                raise ValueError(f"run_id is required for {self.event_type}")
        if not isinstance(self.payload, dict):
            raise ValueError("payload must be an object")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "protocol": self.protocol,
            "protocol_version": self.protocol_version,
            "event_id": self.event_id,
            "batch_id": self.batch_id,
            "run_id": self.run_id,
            "event_type": self.event_type,
            "sequence": self.sequence,
            "sender_utc": self.sender_utc,
            "sender_monotonic_ns": self.sender_monotonic_ns,
            "payload": self.payload,
        }

    @classmethod
    def create(
        cls,
        *,
        batch_id: str,
        event_type: str,
        sequence: int,
        run_id: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> RemoteEvent:
        return cls(
            batch_id=batch_id,
            run_id=run_id,
            event_type=event_type,
            sequence=sequence,
            sender_utc=utc_now(),
            sender_monotonic_ns=time.monotonic_ns(),
            payload=payload or {},
        )

    @classmethod
    def from_dict(cls, document: dict[str, Any]) -> RemoteEvent:
        event = cls(
            protocol=str(document.get("protocol", "")),
            protocol_version=int(document.get("protocol_version", -1)),
            batch_id=str(document["batch_id"]),
            run_id=(None if document.get("run_id") is None else str(document["run_id"])),
            event_type=str(document["event_type"]),
            sequence=int(document["sequence"]),
            sender_utc=str(document["sender_utc"]),
            sender_monotonic_ns=int(document["sender_monotonic_ns"]),
            payload=dict(document.get("payload", {})),
        )
        event.validate()
        supplied_id = document.get("event_id")
        if supplied_id is not None and supplied_id != event.event_id:
            raise ValueError(
                "event_id does not match batch_id, run_id, event_type and sequence"
            )
        return event


class EventIdentityCollision(RuntimeError):
    """An event ID was reused for different event content."""


class EventJournal:
    """Append-only, retry-safe receiver journal."""

    def __init__(
        self,
        root: Path | str,
        event_handler: Callable[[RemoteEvent], dict[str, Any]] | None = None,
    ):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._seen: dict[str, set[str]] = {}
        self._results: dict[str, dict[str, Any]] = {}
        self._events: dict[str, dict[str, Any]] = {}
        self._event_handler = event_handler

    def batch_dir(self, batch_id: str) -> Path:
        validate_identifier(batch_id, "batch_id")
        return self.root / batch_id

    def _load_seen(self, batch_id: str) -> set[str]:
        if batch_id in self._seen:
            return self._seen[batch_id]
        result: set[str] = set()
        path = self.batch_dir(batch_id) / "remote_events.jsonl"
        if path.exists():
            for line_number, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(),
                start=1,
            ):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    if not isinstance(record, dict):
                        raise TypeError("record is not an object")
                    event_document = dict(record["event"])
                    event = RemoteEvent.from_dict(event_document)
                    if event.batch_id != batch_id:
                        raise ValueError(
                            f"event batch_id {event.batch_id!r} does not match directory"
                        )
                    event_id = event.event_id
                    event_result = record.get("event_result", {})
                    if not isinstance(event_result, dict):
                        raise TypeError("event_result is not an object")
                    if event_id in result:
                        raise ValueError(f"duplicate journal event_id {event_id}")
                    result.add(event_id)
                    self._events[event_id] = event_document
                    self._results[event_id] = dict(event_result)
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                    raise RuntimeError(
                        f"invalid acquisition journal {path} line {line_number}: {error}"
                    ) from error
        self._seen[batch_id] = result
        return result

    def record(self, event: RemoteEvent) -> dict[str, Any]:
        event.validate()
        event_document = event.to_dict()
        received_utc = utc_now()
        received_monotonic_ns = time.monotonic_ns()
        with self._lock:
            seen = self._load_seen(event.batch_id)
            duplicate = event.event_id in seen
            event_result = dict(self._results.get(event.event_id, {}))
            if duplicate and self._events.get(event.event_id) != event_document:
                raise EventIdentityCollision(
                    f"event_id collision with different content: {event.event_id}"
                )
            if not duplicate:
                if self._event_handler is not None:
                    event_result = self._event_handler(event)
                self._events[event.event_id] = event_document
                self._results[event.event_id] = event_result
                directory = self.batch_dir(event.batch_id)
                directory.mkdir(parents=True, exist_ok=True)
                record = {
                    "event": event_document,
                    "receiver_utc": received_utc,
                    "receiver_monotonic_ns": received_monotonic_ns,
                    "event_result": event_result,
                }
                path = directory / "remote_events.jsonl"
                with path.open("a", encoding="utf-8", newline="\n") as stream:
                    stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                seen.add(event.event_id)
                if event.event_type == "RUN_START" and event.run_id:
                    (directory / "runs" / event.run_id).mkdir(parents=True, exist_ok=True)
        return {
            "ok": True,
            "protocol": PROTOCOL,
            "protocol_version": PROTOCOL_VERSION,
            "event_id": event.event_id,
            "batch_id": event.batch_id,
            "run_id": event.run_id,
            "event_type": event.event_type,
            "sequence": event.sequence,
            "duplicate": duplicate,
            "receiver_utc": received_utc,
            "receiver_monotonic_ns": received_monotonic_ns,
            "event_result": event_result,
        }


class AcquisitionHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        journal: EventJournal,
        token: str | None = None,
        health_provider: Callable[[], dict[str, Any]] | None = None,
    ):
        super().__init__(server_address, AcquisitionRequestHandler)
        self.journal = journal
        self.token = token
        self.health_provider = health_provider


class AcquisitionRequestHandler(BaseHTTPRequestHandler):
    server: AcquisitionHTTPServer

    def log_message(self, format: str, *args: object) -> None:
        return

    def _authorised(self) -> bool:
        expected = self.server.token
        if not expected:
            return True
        supplied = self.headers.get("X-FFTGYRO-Token", "")
        return hmac.compare_digest(supplied, expected)

    def _json(self, status: HTTPStatus, document: dict[str, Any]) -> None:
        body = json.dumps(document, ensure_ascii=False).encode("utf-8")
        self.send_response(status.value)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path != "/health":
            self._json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "not found"})
            return
        if not self._authorised():
            self._json(HTTPStatus.UNAUTHORIZED, {"ok": False, "error": "unauthorised"})
            return
        health = {
            "ok": True,
            "protocol": PROTOCOL,
            "protocol_version": PROTOCOL_VERSION,
            "receiver_utc": utc_now(),
        }
        if self.server.health_provider is not None:
            health["acquisition"] = self.server.health_provider()
        self._json(HTTPStatus.OK, health)

    def do_POST(self) -> None:
        if self.path != "/events":
            self._json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "not found"})
            return
        if not self._authorised():
            self._json(HTTPStatus.UNAUTHORIZED, {"ok": False, "error": "unauthorised"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 1 or length > _MAX_BODY_BYTES:
                raise ValueError("invalid request size")
            document = json.loads(self.rfile.read(length).decode("utf-8"))
            response = self.server.journal.record(RemoteEvent.from_dict(document))
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
            self._json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(error)})
            return
        except EventIdentityCollision as error:
            self._json(HTTPStatus.CONFLICT, {"ok": False, "error": str(error)})
            return
        except RuntimeError as error:
            self._json(
                HTTPStatus.SERVICE_UNAVAILABLE,
                {"ok": False, "error": str(error)},
            )
            return
        self._json(HTTPStatus.OK, response)


class BridgeClient:
    def __init__(self, base_url: str, token: str | None = None, timeout: float = 3.0):
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout

    def _request(self, request: Request) -> dict[str, Any]:
        if self.token:
            request.add_header("X-FFTGYRO-Token", self.token)
        try:
            with urlopen(request, timeout=self.timeout) as response:
                document = json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            try:
                detail = json.loads(error.read().decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                detail = {"error": str(error)}
            raise RuntimeError(
                f"Windows acquisition bridge rejected request "
                f"(HTTP {error.code}): {detail}"
            ) from error
        except (URLError, TimeoutError, OSError) as error:
            raise RuntimeError(f"Windows acquisition bridge unavailable: {error}") from error
        if not isinstance(document, dict):
            raise RuntimeError(
                f"Windows acquisition bridge returned non-object JSON: {document!r}"
            )
        if not document.get("ok"):
            raise RuntimeError(f"Windows acquisition bridge rejected request: {document}")
        return document

    def health(self) -> dict[str, Any]:
        document = self._request(Request(f"{self.base_url}/health", method="GET"))
        self._validate_protocol(document)
        return document

    def send(
        self,
        event: RemoteEvent,
        *,
        allow_duplicate: bool = False,
    ) -> dict[str, Any]:
        body = json.dumps(event.to_dict(), ensure_ascii=False).encode("utf-8")
        request = Request(
            f"{self.base_url}/events",
            data=body,
            headers={"Content-Type": "application/json; charset=utf-8"},
            method="POST",
        )
        document = self._request(request)
        self._validate_protocol(document)
        expected = {
            "event_id": event.event_id,
            "batch_id": event.batch_id,
            "run_id": event.run_id,
            "event_type": event.event_type,
            "sequence": event.sequence,
        }
        mismatches = {
            key: {"expected": value, "received": document.get(key)}
            for key, value in expected.items()
            if document.get(key) != value
        }
        if mismatches:
            raise RuntimeError(
                f"Windows acquisition bridge response identity mismatch: {mismatches}"
            )
        if document.get("duplicate") and not allow_duplicate:
            raise RuntimeError(
                f"Windows acquisition bridge reported duplicate event {event.event_id}; "
                "refusing to reuse cached acquisition data"
            )
        return document

    @staticmethod
    def _validate_protocol(document: dict[str, Any]) -> None:
        received = (document.get("protocol"), document.get("protocol_version"))
        expected = (PROTOCOL, PROTOCOL_VERSION)
        if received != expected:
            raise RuntimeError(
                "Windows acquisition bridge protocol mismatch: "
                f"expected {expected[0]} v{expected[1]}, "
                f"received {received[0]!r} v{received[1]!r}; "
                "update and restart the Windows bridge before testing"
            )
