import tempfile
import threading
import unittest
from pathlib import Path

from fftgyro_quad_id.experiment_runner.remote_acquisition import EventJournal, RemoteEvent
from fftgyro_quad_id.experiment_runner.remote_rpm import ArduinoRPMService


class FakeSerial:
    def __init__(self, service: ArduinoRPMService) -> None:
        self.service = service
        self.is_open = True
        self.micros = 1000

    def write(self, data: bytes) -> int:
        marker = data.decode("ascii").strip()
        capture = self.service._capture
        assert capture is not None
        capture.rows.append(("M", marker, str(self.micros)))
        self.micros += 1000
        self.service._condition.notify_all()
        return len(data)

    def flush(self) -> None:
        pass


class RemoteRPMTest(unittest.TestCase):
    def make_service(self, root: Path) -> ArduinoRPMService:
        service = ArduinoRPMService.__new__(ArduinoRPMService)
        service.port = "COM4"
        service.root = root
        service._condition = threading.Condition()
        service._capture = None
        service._error = None
        service.serial = FakeSerial(service)
        return service

    @staticmethod
    def event(
        event_type: str,
        sequence: int,
        run_id: str = "batch-001-r01",
    ) -> RemoteEvent:
        return RemoteEvent.create(
            batch_id="batch-001",
            run_id=run_id,
            event_type=event_type,
            sequence=sequence,
            payload={"motor": 1, "percent": 10},
        )

    def test_remote_run_writes_rpm_and_confirms_edges(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            service = self.make_service(root)

            service.handle_event(self.event("RUN_START", 1))
            service.handle_event(self.event("COMMAND_ON", 2))
            assert service._capture is not None
            service._capture.rows.extend([("R", "", "3000"), ("R", "", "4000")])
            service.handle_event(self.event("COMMAND_OFF", 3))
            result = service.handle_event(self.event("RUN_END", 4))

            self.assertTrue(result["execution_confirmed"])
            self.assertEqual(result["rpm_edges"], 2)
            self.assertEqual(result["run_id"], "batch-001-r01")
            self.assertTrue(Path(result["rpm_file"]).exists())
            self.assertIsNone(service._capture)

    def test_abort_releases_capture_before_command_on(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            service = self.make_service(Path(temporary))

            service.handle_event(self.event("RUN_START", 1))
            result = service.handle_event(self.event("RUN_ABORT", 2))

            self.assertTrue(result["aborted"])
            self.assertFalse(result["execution_confirmed"])
            self.assertIsNone(service._capture)

    def test_event_for_another_run_cannot_mark_active_capture(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            service = self.make_service(Path(temporary))
            service.handle_event(self.event("RUN_START", 101, "run-001"))

            with self.assertRaisesRegex(RuntimeError, "active run is"):
                service.handle_event(self.event("COMMAND_ON", 102, "run-002"))

    def test_run_end_requires_command_off(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            service = self.make_service(Path(temporary))
            service.handle_event(self.event("RUN_START", 101))
            service.handle_event(self.event("COMMAND_ON", 102))

            with self.assertRaisesRegex(RuntimeError, "COMMAND_ON and COMMAND_OFF"):
                service.handle_event(self.event("RUN_END", 104))

            self.assertIsNotNone(service._capture)
            aborted = service.handle_event(self.event("RUN_ABORT", 105))
            self.assertTrue(aborted["aborted"])
            self.assertIsNone(service._capture)

    def test_existing_run_directory_cannot_be_reused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "batch-001" / "runs" / "batch-001-r01").mkdir(parents=True)
            service = self.make_service(root)

            with self.assertRaisesRegex(RuntimeError, "existing run directory"):
                service.handle_event(self.event("RUN_START", 101))

    def test_twenty_four_runs_may_reuse_sequences_without_result_aliasing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            service = self.make_service(root)
            journal = EventJournal(root, event_handler=service.handle_event)
            results = []

            for index in range(24):
                run_id = f"run-{index + 1:02d}"
                journal.record(self.event("RUN_START", 101, run_id))
                journal.record(self.event("COMMAND_ON", 102, run_id))
                assert service._capture is not None
                service._capture.rows.extend(
                    [("R", "", str(10_000 + index * 10)), ("R", "", str(10_001 + index * 10))]
                )
                journal.record(self.event("COMMAND_OFF", 103, run_id))
                results.append(journal.record(self.event("RUN_END", 104, run_id)))

            self.assertEqual(
                [result["event_result"]["run_id"] for result in results],
                [f"run-{index + 1:02d}" for index in range(24)],
            )
            self.assertEqual(len(list(root.rglob("rpm.csv"))), 24)
            journal_path = root / "batch-001" / "remote_events.jsonl"
            self.assertEqual(len(journal_path.read_text().splitlines()), 24 * 4)


if __name__ == "__main__":
    unittest.main()
