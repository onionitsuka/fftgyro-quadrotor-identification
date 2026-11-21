"""Exercise the C++ receiver against a synthetic Nano over a pseudoterminal.

Run: python3 onboard_executor/tests/test_rpm_transport.py /tmp/fftgyro-rpm-test
The test executable uses FFTGYRO_RPM_TEST_PTY only to skip the macOS baud ioctl.
"""

import binascii
import csv
import os
import pty
import select
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def frame(kind, capture, sequence, stamp, value=0):
    payload = struct.pack("<2sBcIIII", b"\xa5\x5a", 1, kind, capture, sequence, stamp, value)
    return payload + struct.pack("<H", binascii.crc_hqx(payload, 0xFFFF))


def check(executable, mode):
    with tempfile.TemporaryDirectory() as temp:
        master, slave = pty.openpty()
        output = Path(temp) / "capture"
        proc = subprocess.Popen(
            [executable, os.ttyname(slave), str(output), "0.3"]
            + (["100"] if mode == "dense" else []),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        command = b""
        capture = 0
        sequence = 0
        try:
            deadline = time.monotonic() + 8
            while proc.poll() is None and time.monotonic() < deadline:
                if not select.select([master], [], [], 0.05)[0]:
                    continue
                command += os.read(master, 4096)
                while b"\n" in command:
                    line, command = command.split(b"\n", 1)
                    if line.startswith(b"B"):
                        capture = int(line[1:], 16)
                        sequence = 1
                        os.write(master, frame(b"B", capture, 0, 0xFFFFFF00))
                        for channel in range(1, 5):
                            message = frame(
                                b"E",
                                capture,
                                sequence,
                                (0xFFFFFF00 + channel * 100) & 0xFFFFFFFF,
                                channel,
                            )
                            if mode == "crc" and channel == 2:
                                message = message[:-1] + bytes([message[-1] ^ 1])
                            if mode != "gap" or channel != 2:
                                # Arbitrary USB fragmentation must not alter packets.
                                os.write(master, message[:7])
                                os.write(master, message[7:])
                            sequence += 1
                        if mode == "overflow":
                            os.write(master, frame(b"H", capture, sequence, 2000, 3))
                            sequence += 1
                        if mode == "reset":
                            os.write(master, frame(b"B", capture ^ 2, 0, 0))
                    elif line.startswith(b"T"):
                        os.write(master, frame(b"T", capture, sequence, 3000, int(line[1:], 16)))
                        sequence += 1
                    elif line == b"Z" and capture and mode != "missing_end":
                        os.write(master, frame(b"Z", capture, sequence, 4000))
                        sequence += 1
            stdout, stderr = proc.communicate(timeout=1)
            result = (output / "result.txt").read_text()
            if mode in ("normal", "dense"):
                assert proc.returncode == 0, (stdout, stderr)
                assert result.startswith("COMPLETE\n"), result
                with (output / "rpm.csv").open() as stream:
                    rows = list(csv.DictReader(stream))
                    events = [r for r in rows if r["kind"] == "E"]
                assert [int(r["value"]) for r in events] == [1, 2, 3, 4]
                assert int(events[-1]["nano_us"]) == 144  # Raw rollover preserved.
                queries = [r for r in rows if r["kind"] == "Q"]
                replies = [r for r in rows if r["kind"] == "T"]
                assert [r["value"] for r in queries] == [r["value"] for r in replies]
                if mode == "dense":
                    assert len(queries) >= 2
                    assert "sync_interval_ms=100\n" in result
            else:
                assert proc.returncode == 1 and result.startswith("INCOMPLETE\n"), (
                    stdout,
                    stderr,
                    result,
                )
            print(f"PASS {mode}")
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
            os.close(master)
            os.close(slave)


if __name__ == "__main__":
    for interval in ("0", "49", "1001", "nan", "100x"):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "not-created"
            rejected = subprocess.run([sys.argv[1], "/not-a-device", str(output), "1", interval],
                                      capture_output=True)
            assert rejected.returncode != 0 and not output.exists()
    print("PASS invalid sync intervals", flush=True)
    for scenario in ("normal", "dense", "crc", "gap", "overflow", "reset", "missing_end"):
        check(sys.argv[1], scenario)
