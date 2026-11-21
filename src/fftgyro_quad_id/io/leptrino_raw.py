"""Bounded USB receiver with queued raw storage and post-stop CSV decoding.

Standalone on Pi: keep leptrino.py beside this file. No motor/zero commands.
File format: magic, then repeated <uint64 receive_ns,uint32 size> and USB bytes.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import signal
import socket
import struct
import threading
import time

try:
    from .leptrino import (RATED_RAW, STATUS_FAULT_MASK, Decoder, checked, frame, parse_info,
                           parse_limits, parse_sample, status_flags)
except ImportError:
    from leptrino import (RATED_RAW, STATUS_FAULT_MASK, Decoder, checked, frame, parse_info,
                          parse_limits, parse_sample, status_flags)

MAGIC = b"LEPTRAW1"
HEADER = struct.Struct("<QI")


class RawWriter:
    def __init__(self, path: Path):
        self.items = queue.Queue(maxsize=4096)  # <=16 MiB payload, bounded item count
        self.done = threading.Event()
        self.error = None
        self.peak_queue = 0
        self.max_write_ns = 0
        self.file = path.open("xb")
        self.file.write(MAGIC)
        self.worker = threading.Thread(target=self._run, name="raw-storage")
        self.worker.start()

    def push(self, stamp: int, data: bytes):
        if self.error:
            raise OSError(self.error)
        if not data:
            return
        if len(data) > 4096:
            raise ValueError("chunk too large")
        try:
            self.items.put_nowait((stamp, data))
        except queue.Full:
            raise OSError("raw storage queue full; capture is incomplete") from None
        self.peak_queue = max(self.peak_queue, self.items.qsize())

    def _run(self):
        try:
            synced = time.monotonic()
            while not self.done.is_set() or not self.items.empty():
                try:
                    stamp, data = self.items.get(timeout=0.05)
                except queue.Empty:
                    continue
                before = time.monotonic_ns()
                self.file.write(HEADER.pack(stamp, len(data)))
                self.file.write(data)
                if time.monotonic()-synced >= 1:
                    self.file.flush()
                    os.fsync(self.file.fileno())
                    synced = time.monotonic()
                self.max_write_ns = max(self.max_write_ns, time.monotonic_ns()-before)
            self.file.flush()
            os.fsync(self.file.fileno())
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
        finally:
            self.file.close()

    def finish(self):
        self.done.set()
        self.worker.join()
        if self.error:
            raise OSError(self.error)


def chunks(path: Path):
    with path.open("rb") as source:
        if source.read(len(MAGIC)) != MAGIC:
            raise ValueError("invalid raw file magic")
        previous = -1
        while header := source.read(HEADER.size):
            if len(header) != HEADER.size:
                raise ValueError("truncated raw header")
            stamp, size = HEADER.unpack(header)
            if stamp < previous or not 0 < size <= 4096:
                raise ValueError("invalid chunk timestamp/size")
            previous = stamp
            data = source.read(size)
            if len(data) != size:
                raise ValueError("truncated raw chunk")
            yield stamp, data


def decode(path: Path, output: Path, limits: tuple, start_ns: int,
           *, counter_jump_limit: int | None = None) -> dict:
    decoder = Decoder()
    count = 0
    previous_stamp = None
    previous_counter = None
    max_gap = 0
    jumps = []
    jump_count = 0
    statuses = {}
    fault_samples = 0
    fault_counts = {"compensation_rom_error": 0, "sensor_error": 0, "rated_overrange": 0}
    max_abs_raw = [0] * 6
    overrange_consistency = {
        "flag_and_raw_overrange": 0,
        "flag_without_raw_overrange": 0,
        "raw_overrange_without_flag": 0,
        "neither": 0,
    }
    started = False
    stopped = False
    with output.open("x", newline="", encoding="utf-8") as target:
        writer = csv.writer(target)
        writer.writerow(["index", "receive_monotonic_ns", "batch", "fx_N", "fy_N", "fz_N",
                         "mx_Nm", "my_Nm", "mz_Nm", "raw_fx", "raw_fy", "raw_fz",
                         "raw_mx", "raw_my", "raw_mz", "status", "payload_hex"])
        for batch, (stamp, data) in enumerate(chunks(path)):
            for payload in decoder.feed(data):
                if stamp < start_ns:
                    continue
                if len(payload) >= 3 and payload[2] == 0x33:
                    checked(payload, 0x33, 4)
                    stopped = True
                if len(payload) < 3 or payload[2] != 0x32:
                    continue
                if stopped:
                    raise ValueError("sample after stop acknowledgement")
                if len(payload) == 4:
                    checked(payload, 0x32, 4)
                    started = True
                    continue
                raw, values, status = parse_sample(payload, limits)
                writer.writerow([count, stamp, batch, *values, *raw, status, payload.hex()])
                statuses[str(status)] = statuses.get(str(status), 0)+1
                flags = status_flags(status)
                fault_samples += int(bool(status & STATUS_FAULT_MASK))
                for name in fault_counts:
                    fault_counts[name] += int(flags[name])
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
                if previous_stamp is not None:
                    max_gap = max(max_gap, stamp-previous_stamp)
                if previous_counter is not None and (payload[19]-previous_counter)%256 != 1:
                    jump_count += 1
                    if counter_jump_limit is None or len(jumps) < counter_jump_limit:
                        jumps.append({"index": count, "delta_mod256": (payload[19]-previous_counter)%256})
                previous_counter, previous_stamp = payload[19], stamp
                count += 1
        target.flush()
        os.fsync(target.fileno())
    if decoder.errors or decoder.state != 0 or not started or not stopped or not count:
        raise ValueError(f"incomplete/invalid stream: errors={decoder.errors}, state={decoder.state}, "
                         f"start={started}, stop={stopped}, samples={count}")
    return dict(samples=count, framing_errors=decoder.errors, max_receive_gap_ms=max_gap/1e6,
                candidate_counter_jumps=jumps, candidate_counter_jump_count=jump_count,
                status_byte_histogram=statuses,
                documented_status_fault_samples=fault_samples,
                documented_status_fault_counts=fault_counts,
                max_abs_raw_by_axis=max_abs_raw,
                overrange_flag_vs_raw=overrange_consistency,
                candidate_counter_note=("Byte 19 is reserved/indeterminate in vendor protocol Ver1.31; "
                                        "its observed modulo-256 pattern is diagnostic only"))


def capture(args) -> int:
    import serial

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    summary = dict(state="INCOMPLETE", hostname=socket.gethostname(),
                   boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                   source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   timestamp_basis="CLOCK_MONOTONIC USB read completion; not sensor sample time",
                   measurement_quality="UNVERIFIED", seconds_requested=args.seconds,
                   zero_reset_sent=False, motor_commands_sent=False)

    def save():
        with (out/"summary.tmp").open("w", encoding="utf-8") as target:
            json.dump(summary, target, indent=2)
            target.flush()
            os.fsync(target.fileno())
        (out/"summary.tmp").replace(out/"summary.json")

    save()
    storage = None
    port = None
    stopped = False
    stream_active = False
    stop_requested = False
    def interrupt(signum, unused):
        nonlocal stop_requested
        if stream_active:
            stop_requested = True
            summary["operator_stop_signal"] = signum
            return
        raise KeyboardInterrupt(f"signal {signum}")
    handlers = {sig: signal.signal(sig, interrupt) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        port = serial.Serial(port=None, baudrate=460800, timeout=0.05, write_timeout=1, exclusive=True)
        port.dtr = port.rts = False
        port.port = args.port
        port.open()
        storage = RawWriter(out/"chunks.bin")

        def send(command):
            data = frame(bytes([4, 255, command, 0]))
            if port.write(data) != len(data):
                raise OSError("short command write")

        def receive():
            # Blocking on one byte keeps latency low; drain already queued bytes next.
            data = port.read(min(max(port.in_waiting, 1), 4096))
            stamp = time.monotonic_ns()
            storage.push(stamp, data)
            return stamp, data

        def request(command, minimum):
            parser = Decoder()
            send(command)
            deadline = time.monotonic()+2
            while time.monotonic() < deadline:
                stamp, data = receive()
                for payload in parser.feed(data):
                    if len(payload) >= 3 and payload[2] == command:
                        return checked(payload, command, minimum)
            raise TimeoutError(f"no response to {command:#x}")

        request(0x33, 4)
        info = parse_info(request(0x2a, 38))
        if info["serial"] != args.expected_serial or info["product"] != args.expected_product:
            raise ValueError(f"wrong sensor: {info}")
        limits = parse_limits(request(0x2b, 28))
        summary.update(device=info, rated_limits_N_Nm=limits)
        start_ns = time.monotonic_ns()
        summary["start_command_before_ns"] = start_ns
        send(0x32)
        stream_active = True
        deadline = time.monotonic()+args.seconds
        last_data = time.monotonic()
        while time.monotonic() < deadline and not stop_requested:
            stamp, data = receive()
            if data:
                last_data = time.monotonic()
            elif time.monotonic()-last_data > 1:
                raise TimeoutError("no USB bytes for one second")
        summary["stop_command_before_ns"] = time.monotonic_ns()
        request(0x33, 4)
        stopped = True
        stream_active = False
        summary["stop_acknowledged_ns"] = time.monotonic_ns()
        summary["stopped_early"] = stop_requested
        summary["elapsed_capture_s"] = (summary["stop_command_before_ns"]-start_ns)/1e9
        port.close()
        storage.finish()
        summary.update(raw_writer_peak_queue=storage.peak_queue,
                       raw_writer_max_write_ms=storage.max_write_ns/1e6)
        summary["raw_storage_complete"] = True
        summary["state"] = "RAW_COMPLETE"
        if args.decode_on_pi:
            summary.update(decode(out/"chunks.bin", out/"samples.csv", limits, start_ns))
            summary["state"] = "COMPLETE" if not summary["candidate_counter_jumps"] else "INCOMPLETE"
            summary["measurement_quality"] = (
                "INVALID_DOCUMENTED_STATUS_FAULT"
                if summary["documented_status_fault_samples"]
                else "STATUS_OK; calibration and timing validation remain separate"
            )
    except (Exception, KeyboardInterrupt) as exc:
        summary["state"] = "INCOMPLETE"
        summary["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if port is not None and port.is_open:
            if not stopped:
                try:
                    port.write(frame(bytes([4,255,0x33,0])))
                except Exception as exc:
                    summary["stop_error"] = str(exc)
            port.close()
        if storage is not None:
            try:
                storage.finish()
            except Exception as exc:
                summary.update(state="INCOMPLETE", storage_error=str(exc))
        summary["stop_acknowledged"] = stopped
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        save()
    print(json.dumps(summary, indent=2), flush=True)
    return int(summary["state"] not in ("COMPLETE", "RAW_COMPLETE"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seconds", type=float, default=40)
    parser.add_argument("--expected-serial", required=True)
    parser.add_argument("--expected-product", default="ASM0551012R0A00")
    parser.add_argument("--decode-on-pi", action="store_true",
                        help="Optional slow post-processing; normally decode on Windows")
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or not 0 < args.seconds <= 3600:
        parser.error("seconds must be (0,3600]")
    raise SystemExit(capture(args))


if __name__ == "__main__":
    main()
