import argparse
import importlib.util
import json
from pathlib import Path


def load_module():
    path = Path(__file__).resolve().parents[1] / "scripts/capture_frame_encoder.py"
    spec = importlib.util.spec_from_file_location("capture_frame_encoder_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def encoder_frame() -> bytes:
    frame = bytearray(b" " * 32)
    frame[0:4] = bytes((0x7A, 2, 0, ord(",")))
    for start, value in ((4, "10.0000"), (13, "-2.5000"), (22, "30.0000")):
        frame[start:start + 8] = value.rjust(8).encode("ascii")
    frame[12] = frame[21] = frame[30] = ord(",")
    frame[31] = 0x7B
    return bytes(frame)


def test_capture_requires_mode_two_and_writes_validated_products(tmp_path, monkeypatch):
    module = load_module()
    stop = tmp_path / "stop"

    class Port:
        in_waiting = 32

        def __init__(self):
            self.reads = 0

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, _size):
            self.reads += 1
            if self.reads == 100:
                stop.touch()
            return encoder_frame()

    monkeypatch.setattr(module, "open_serial_without_reset", lambda *_: Port())
    output = tmp_path / "encoder"
    ready = tmp_path / "ready"
    args = argparse.Namespace(
        output=output, ready_file=ready, stop_file=stop, port="COM5", baud=576000,
        seconds=5, minimum_rate_hz=1, offset_roll_deg=-147.1289,
        offset_yaw_deg=-83.1445, pitch_sign=-1.0, yaw_sign=1.0,
    )
    assert module.capture(args) == 0
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["state"] == "COMPLETE"
    assert metadata["encoder_frames"] == 100
    assert metadata["motor_commands_sent"] is False
    assert not ready.exists()
    assert len((output / "samples.csv").read_text().splitlines()) == 101

