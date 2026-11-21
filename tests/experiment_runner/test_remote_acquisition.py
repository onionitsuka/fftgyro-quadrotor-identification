import json
import tempfile
import threading
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from fftgyro_quad_id.experiment_runner.remote_acquisition import (
    PROTOCOL,
    PROTOCOL_VERSION,
    AcquisitionHTTPServer,
    BridgeClient,
    EventIdentityCollision,
    EventJournal,
    RemoteEvent,
)


class RemoteAcquisitionTest(unittest.TestCase):
    def test_journal_is_retry_safe(self):
        with tempfile.TemporaryDirectory() as temporary:
            journal = EventJournal(Path(temporary))
            event = RemoteEvent.create(
                batch_id="batch-001",
                run_id="batch-001-r01",
                event_type="RUN_START",
                sequence=2,
                payload={"repetition": 1},
            )
            first = journal.record(event)
            second = journal.record(event)

            self.assertFalse(first["duplicate"])
            self.assertTrue(second["duplicate"])
            path = Path(temporary) / "batch-001" / "remote_events.jsonl"
            self.assertEqual(len(path.read_text(encoding="utf-8").splitlines()), 1)
            record = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(record["event"]["run_id"], "batch-001-r01")

    def test_run_event_requires_run_id(self):
        event = RemoteEvent.create(
            batch_id="batch-001", event_type="COMMAND_ON", sequence=1
        )
        with self.assertRaises(ValueError):
            event.to_dict()

    def test_handler_result_is_returned_for_duplicate(self):
        with tempfile.TemporaryDirectory() as temporary:
            handled = []

            def handler(event: RemoteEvent) -> dict[str, object]:
                handled.append(event.event_id)
                return {"rpm_edges": 42}

            journal = EventJournal(Path(temporary), event_handler=handler)
            event = RemoteEvent.create(
                batch_id="batch-001",
                run_id="batch-001-r01",
                event_type="RUN_END",
                sequence=8,
            )
            first = journal.record(event)
            second = journal.record(event)

            self.assertEqual(handled, [event.event_id])
            self.assertEqual(first["event_result"], {"rpm_edges": 42})
            self.assertEqual(second["event_result"], {"rpm_edges": 42})

    def test_same_sequence_in_different_runs_is_not_a_duplicate(self):
        with tempfile.TemporaryDirectory() as temporary:
            handled = []

            def handler(event: RemoteEvent) -> dict[str, object]:
                handled.append(event.run_id)
                return {"run_id": event.run_id}

            journal = EventJournal(Path(temporary), event_handler=handler)
            first = RemoteEvent.create(
                batch_id="batch-001",
                run_id="run-001",
                event_type="RUN_START",
                sequence=101,
            )
            second = RemoteEvent.create(
                batch_id="batch-001",
                run_id="run-002",
                event_type="RUN_START",
                sequence=101,
            )

            first_result = journal.record(first)
            second_result = journal.record(second)

            self.assertNotEqual(first.event_id, second.event_id)
            self.assertFalse(first_result["duplicate"])
            self.assertFalse(second_result["duplicate"])
            self.assertEqual(handled, ["run-001", "run-002"])

    def test_event_id_reuse_with_changed_content_is_rejected_after_restart(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            event = RemoteEvent.create(
                batch_id="batch-001",
                run_id="run-001",
                event_type="RUN_START",
                sequence=101,
                payload={"percent": 10},
            )
            EventJournal(root).record(event)
            collision = replace(event, payload={"percent": 20})

            with self.assertRaises(EventIdentityCollision):
                EventJournal(root).record(collision)

    def test_corrupt_journal_is_rejected_instead_of_ignored(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            batch = root / "batch-001"
            batch.mkdir()
            (batch / "remote_events.jsonl").write_text("{truncated\n", encoding="utf-8")
            event = RemoteEvent.create(
                batch_id="batch-001",
                run_id="run-001",
                event_type="RUN_START",
                sequence=101,
            )

            with self.assertRaisesRegex(RuntimeError, "line 1"):
                EventJournal(root).record(event)

    def test_old_protocol_journal_cannot_be_extended(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            batch = root / "batch-001"
            batch.mkdir()
            old_event = RemoteEvent.create(
                batch_id="batch-001",
                run_id="run-old",
                event_type="RUN_START",
                sequence=101,
            ).to_dict()
            old_event["protocol_version"] = PROTOCOL_VERSION - 1
            (batch / "remote_events.jsonl").write_text(
                json.dumps({"event": old_event, "event_result": {}}) + "\n",
                encoding="utf-8",
            )
            new_event = RemoteEvent.create(
                batch_id="batch-001",
                run_id="run-new",
                event_type="RUN_START",
                sequence=101,
            )

            with self.assertRaisesRegex(RuntimeError, "unsupported remote-acquisition protocol"):
                EventJournal(root).record(new_event)

    def test_health_rejects_old_windows_protocol(self):
        client = BridgeClient("http://windows.invalid")
        with patch.object(
            client,
            "_request",
            return_value={
                "ok": True,
                "protocol": PROTOCOL,
                "protocol_version": PROTOCOL_VERSION - 1,
            },
        ):
            with self.assertRaisesRegex(RuntimeError, "update and restart"):
                client.health()

    def test_send_rejects_response_for_another_run(self):
        client = BridgeClient("http://windows.invalid")
        event = RemoteEvent.create(
            batch_id="batch-001",
            run_id="run-001",
            event_type="RUN_START",
            sequence=101,
        )
        wrong_response = {
            "ok": True,
            "protocol": PROTOCOL,
            "protocol_version": PROTOCOL_VERSION,
            "event_id": event.event_id,
            "batch_id": event.batch_id,
            "run_id": "run-002",
            "event_type": event.event_type,
            "sequence": event.sequence,
        }
        with patch.object(client, "_request", return_value=wrong_response):
            with self.assertRaisesRegex(RuntimeError, "identity mismatch"):
                client.send(event)

    def test_send_rejects_cached_duplicate_by_default(self):
        client = BridgeClient("http://windows.invalid")
        event = RemoteEvent.create(
            batch_id="batch-001",
            run_id="run-001",
            event_type="RUN_END",
            sequence=104,
        )
        duplicate_response = {
            "ok": True,
            "protocol": PROTOCOL,
            "protocol_version": PROTOCOL_VERSION,
            "event_id": event.event_id,
            "batch_id": event.batch_id,
            "run_id": event.run_id,
            "event_type": event.event_type,
            "sequence": event.sequence,
            "duplicate": True,
        }
        with patch.object(client, "_request", return_value=duplicate_response):
            with self.assertRaisesRegex(RuntimeError, "cached acquisition data"):
                client.send(event)

    def test_http_server_and_client_exchange_validated_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            server = AcquisitionHTTPServer(
                ("127.0.0.1", 0),
                EventJournal(Path(temporary)),
            )
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                client = BridgeClient(f"http://127.0.0.1:{server.server_port}")
                health = client.health()
                self.assertEqual(health["protocol_version"], PROTOCOL_VERSION)
                event = RemoteEvent.create(
                    batch_id="batch-001",
                    run_id="run-001",
                    event_type="RUN_START",
                    sequence=101,
                )
                first = client.send(event)
                duplicate = client.send(event, allow_duplicate=True)
                self.assertEqual(first["run_id"], "run-001")
                self.assertFalse(first["duplicate"])
                self.assertTrue(duplicate["duplicate"])
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2.0)


if __name__ == "__main__":
    unittest.main()
