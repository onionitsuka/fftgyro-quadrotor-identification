"""Bounded Leptrino USB acquisition; also runnable as a standalone Python file.

Protocol follows the existing FFTGYRO force-sensor.js implementation.
Receive timestamps are NOT sensor sampling timestamps or cross-Pi synchronization.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import socket
import struct
import time


STATUS_FAULT_MASK = 0x07
RATED_RAW = 10_000


def status_flags(status: int) -> dict[str, bool | int]:
    """Decode the documented status bits; reserved bits remain observable."""
    if not 0 <= status <= 0xFF:
        raise ValueError("status must be one byte")
    return {
        "compensation_rom_error": bool(status & 0x01),
        "sensor_error": bool(status & 0x02),
        "rated_overrange": bool(status & 0x04),
        "external_input": bool(status & 0x20),
        "reserved_bits": status & 0xD8,
    }


def frame(payload: bytes) -> bytes:
    checksum = 3
    for value in payload:
        checksum ^= value
    return b"\x10\x02" + payload.replace(b"\x10", b"\x10\x10") + b"\x10\x03" + bytes([checksum])


class Decoder:
    def __init__(self) -> None:
        self.state = 0
        self.payload = bytearray()
        self.errors = 0

    def feed(self, data: bytes) -> list[bytes]:
        result = []
        for value in data:
            if self.state == 0:
                self.state = 1 if value == 16 else 0
            elif self.state == 1:
                if value == 2:
                    self.payload.clear()
                    self.state = 2
                else:
                    self.state = 1 if value == 16 else 0
            elif self.state == 2:
                if value == 16:
                    self.state = 3
                else:
                    self.payload.append(value)
            elif self.state == 3:
                if value == 16:
                    self.payload.append(value)
                    self.state = 2
                elif value == 3:
                    self.state = 4
                else:
                    self.errors += 1
                    self.payload.clear()
                    self.state = 2 if value == 2 else 0
            else:
                checksum = 3
                for byte in self.payload:
                    checksum ^= byte
                if checksum == value:
                    result.append(bytes(self.payload))
                else:
                    self.errors += 1
                self.state = 0
            if len(self.payload) > 255:
                self.errors += 1
                self.payload.clear()
                self.state = 0
        return result


def checked(payload: bytes, command: int, minimum: int) -> bytes:
    if len(payload) < minimum or payload[0] != len(payload) or payload[2] != command or payload[3] != 0:
        raise ValueError(f"Unexpected response: {payload.hex()}")
    return payload


def parse_info(payload: bytes) -> dict:
    checked(payload, 0x2A, 38)
    return dict(zip(("product", "serial", "firmware", "frequency_hz"),
                    (payload[a:b].decode("ascii").strip("\x00 ")
                     for a, b in ((4, 20), (20, 28), (28, 32), (32, 38)))))


def parse_limits(payload: bytes) -> tuple[float, ...]:
    checked(payload, 0x2B, 28)
    limits = struct.unpack_from("<6f", payload, 4)
    if not all(math.isfinite(x) and x > 0 for x in limits):
        raise ValueError("Invalid rated limits")
    return limits


def parse_sample(payload: bytes, limits: tuple[float, ...]) -> tuple:
    checked(payload, 0x32, 20)
    raw = struct.unpack_from("<6h", payload, 4)
    return raw, tuple(x * limit / 10000 for x, limit in zip(raw, limits)), payload[18]


def capture(args: argparse.Namespace) -> int:
    import serial  # Optional on development machines; apt install python3-serial on Pi.

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    decoder = Decoder()
    metadata = {"state": "INCOMPLETE", "port": args.port, "duration_requested_s": args.seconds,
                "hostname": socket.gethostname(), "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "timestamp_basis": "host monotonic receive; shared within USB read batch",
                "cross_pi_synchronized": False, "zero_reset_sent": False,
                "measurement_quality": "UNVERIFIED: trailing status byte interpretation and calibration pending",
                "loss_detection": "No verified sensor sequence counter; zero detected errors does not prove no loss"}

    def save() -> None:
        temporary = out / "summary.tmp"
        with temporary.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        temporary.replace(out / "summary.json")

    save()
    count = 0
    status_count = 0
    status_fault_count = 0
    status_faults = {"compensation_rom_error": 0, "sensor_error": 0, "rated_overrange": 0}
    max_abs_raw = [0] * 6
    overrange_consistency = {
        "flag_and_raw_overrange": 0,
        "flag_without_raw_overrange": 0,
        "raw_overrange_without_flag": 0,
        "neither": 0,
    }
    status_histogram = {}
    sums = [0.0] * 6
    stopped = False
    port = None
    previous_handlers = {}

    def interrupt(signum, unused) -> None:
        raise KeyboardInterrupt(f"Signal {signum}")

    for sig in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[sig] = signal.signal(sig, interrupt)
    try:
        port = serial.Serial(port=None, baudrate=460800, timeout=0.05, write_timeout=1, exclusive=True)
        port.dtr = False
        port.rts = False
        port.port = args.port
        port.open()
        time.sleep(0.02)
        with (out / "received.bin").open("xb") as binary, (out / "samples.csv").open("x", newline="", encoding="utf-8") as output:
            writer = csv.writer(output)
            writer.writerow(["index", "receive_monotonic_ns", "batch", "fx_N", "fy_N", "fz_N", "mx_Nm", "my_Nm", "mz_Nm",
                             "raw_fx", "raw_fy", "raw_fz", "raw_mx", "raw_my", "raw_mz", "status", "payload_hex"])

            def send(command: int) -> None:
                message = frame(bytes([4, 255, command, 0]))
                if port.write(message) != len(message):
                    raise OSError("Short command write")

            def receive() -> tuple[int, list[bytes]]:
                data = port.read(min(max(port.in_waiting, 1), 4096))
                stamp = time.monotonic_ns()
                binary.write(data)
                return stamp, decoder.feed(data)

            def request(command: int, minimum: int) -> bytes:
                send(command)
                deadline = time.monotonic() + 2
                while time.monotonic() < deadline:
                    _, messages = receive()
                    for payload in messages:
                        if len(payload) >= 3 and payload[2] == command:
                            return checked(payload, command, minimum)
                raise TimeoutError(f"No response to {command:#x}")

            request(0x33, 4)
            info = parse_info(request(0x2A, 38))
            metadata["device"] = info
            if info["serial"] != args.expected_serial or info["product"] != args.expected_product:
                raise ValueError(f"Wrong sensor: {info}")
            limits = parse_limits(request(0x2B, 28))
            metadata["rated_limits_N_Nm"] = limits
            before = time.monotonic_ns()
            metadata["utc_anchor_ns"] = time.time_ns()
            after = time.monotonic_ns()
            metadata["monotonic_anchor_ns"] = (before + after) // 2
            metadata["anchor_bracket_ns"] = after - before
            save()
            send(0x32)
            start = time.monotonic()
            last_sample = start
            first_stamp = None
            last_stamp = None
            batch = 0
            while time.monotonic() - start < args.seconds:
                stamp, messages = receive()
                batch += 1
                for payload in messages:
                    if len(payload) >= 3 and payload[2] == 0x32:
                        if len(payload) == 4:
                            checked(payload, 0x32, 4)
                            continue
                        raw, values, status = parse_sample(payload, limits)
                        writer.writerow([count, stamp, batch, *values, *raw, status, payload.hex()])
                        count += 1
                        status_count += int(status != 0)
                        flags = status_flags(status)
                        status_fault_count += int(bool(status & STATUS_FAULT_MASK))
                        for name in status_faults:
                            status_faults[name] += int(flags[name])
                        max_abs_raw = [max(old, abs(value))
                                       for old, value in zip(max_abs_raw, raw)]
                        raw_overrange = any(abs(value) > RATED_RAW for value in raw)
                        if flags["rated_overrange"] and raw_overrange:
                            key = "flag_and_raw_overrange"
                        elif flags["rated_overrange"]:
                            key = "flag_without_raw_overrange"
                        elif raw_overrange:
                            key = "raw_overrange_without_flag"
                        else:
                            key = "neither"
                        overrange_consistency[key] += 1
                        status_histogram[str(status)] = status_histogram.get(str(status), 0) + 1
                        sums = [a + b for a, b in zip(sums, values)]
                        first_stamp = stamp if first_stamp is None else first_stamp
                        last_stamp = stamp
                        last_sample = time.monotonic()
                if time.monotonic() - last_sample > 1:
                    raise TimeoutError("No force samples for one second")
            metadata.update(elapsed_s=time.monotonic() - start, first_receive_ns=first_stamp, last_receive_ns=last_stamp)
            request(0x33, 4)
            stopped = True
            for file in (binary, output):
                file.flush()
                os.fsync(file.fileno())
            if count == 0 or decoder.errors:
                raise ValueError("Empty capture or framing errors")
            metadata["state"] = "COMPLETE"
    except (Exception, KeyboardInterrupt) as exc:
        metadata["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if port is not None and port.is_open:
            if not stopped:
                try:
                    port.write(frame(bytes([4, 255, 0x33, 0])))
                except Exception as exc:
                    metadata["stop_write_error"] = str(exc)
            port.close()
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
        metadata.update(samples=count, nonzero_status_samples=status_count,
                        documented_status_fault_samples=status_fault_count,
                        documented_status_fault_counts=status_faults,
                        max_abs_raw_by_axis=max_abs_raw,
                        overrange_flag_vs_raw=overrange_consistency,
                        status_byte_histogram=status_histogram, framing_errors=decoder.errors,
                        stop_acknowledged=stopped, mean_N_Nm=[x / count for x in sums] if count else None)
        if metadata.get("state") == "COMPLETE":
            metadata["measurement_quality"] = (
                "INVALID_DOCUMENTED_STATUS_FAULT" if status_fault_count
                else "STATUS_OK; calibration and timing validation remain separate"
            )
        save()
    print(json.dumps(metadata, indent=2))
    return 0 if metadata["state"] == "COMPLETE" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True)
    parser.add_argument("--output", required=True, help="New directory; never overwrites a run")
    parser.add_argument("--seconds", type=float, default=5)
    parser.add_argument("--expected-serial", required=True)
    parser.add_argument("--expected-product", default="ASM0551012R0A00")
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or not 0 < args.seconds <= 3600:
        parser.error("seconds must be finite and in (0, 3600]")
    return capture(args)


if __name__ == "__main__":
    raise SystemExit(main())
