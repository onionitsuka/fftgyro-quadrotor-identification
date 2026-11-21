import json
import tempfile
import unittest
from pathlib import Path

from fftgyro_quad_id.experiment_runner.session import ExperimentSession, RAW_SOURCES


def _create(root: Path, run_id: str = "run_001") -> ExperimentSession:
    return ExperimentSession.create(
        root,
        run_id=run_id,
        protocol_id="motor_single",
        protocol_version="pdf-2026-07-11-v1",
        profile="minimum",
        operator="operator_1",
        hardware_config_id="hardware_001",
        esc_inventory_id="esc_inventory_001",
    )


class ExperimentSessionTests(unittest.TestCase):
    def test_create_writes_metadata_and_raw_directories(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            session = _create(Path(temp))
            metadata = json.loads((session.root / "run.json").read_text())
            self.assertEqual(metadata["run_id"], "run_001")
            for source in RAW_SOURCES:
                self.assertTrue(session.raw_source_dir(source).is_dir())

    def test_existing_run_is_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _create(root)
            with self.assertRaises(FileExistsError):
                _create(root)

    def test_events_are_append_only_json_lines(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            session = _create(Path(temp))
            session.append_event("segment_started", {"segment_id": "s1"})
            records = [
                json.loads(line)
                for line in session.events_path.read_text().splitlines()
            ]
            self.assertEqual(records[0]["event_type"], "session_created")
            self.assertEqual(records[1]["payload"]["segment_id"], "s1")
            self.assertLessEqual(records[0]["monotonic_ns"], records[1]["monotonic_ns"])

    def test_finalize_is_single_use_and_locks_event_append(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            session = _create(Path(temp))
            final_path = session.finalize(
                status="stopped",
                summary={"reason": "emergency_stop"},
            )
            document = json.loads(final_path.read_text())
            self.assertEqual(document["status"], "stopped")
            with self.assertRaises(RuntimeError):
                session.append_event("late_event", {})
            with self.assertRaises(RuntimeError):
                session.finalize(status="completed", summary={})

    def test_invalid_run_id_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError):
                _create(Path(temp), run_id="run id with spaces")


if __name__ == "__main__":
    unittest.main()

