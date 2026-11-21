"""Capture and validate the FFT GYRO frame-encoder stream on Windows.

Raw bytes, receive timestamps, and decoded product/experimental angles are
retained. This process never sends a motor or frame-motor command.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys
import time

import serial

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fftgyro_quad_id.rig.free_decay import FrameReader, parse_encoder_frame, wrap_degrees


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True)
    parser.add_argument("--baud", type=int, default=576000, choices=(57600, 576000))
    parser.add_argument("--seconds", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ready-file", type=Path, required=True)
    parser.add_argument("--stop-file", type=Path, required=True)
    parser.add_argument("--minimum-rate-hz", type=float, default=80.0)
    parser.add_argument("--offset-roll-deg", type=float, default=-147.1289)
    parser.add_argument("--offset-yaw-deg", type=float, default=-83.1445)
    parser.add_argument("--pitch-sign", type=float, default=-1.0, choices=(-1.0, 1.0))
    parser.add_argument("--yaw-sign", type=float, default=1.0, choices=(-1.0, 1.0))
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or not 5 <= args.seconds <= 1800:
        parser.error("seconds must be in [5, 1800]")
    if not math.isfinite(args.minimum_rate_hz) or not 1 <= args.minimum_rate_hz <= 200:
        parser.error("minimum-rate-hz must be in [1, 200]")
    return args


def open_serial_without_reset(port_name: str, baud: int) -> serial.Serial:
    port = serial.Serial(port=None, baudrate=baud, timeout=0.02)
    port.dtr = False
    port.rts = False
    port.port = port_name
    port.open()
    return port


def write_json(path: Path, document: dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(document, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())


def capture(args: argparse.Namespace) -> int:
    args.output.mkdir(parents=True, exist_ok=False)
    if args.ready_file.exists() or args.stop_file.exists():
        raise FileExistsError("encoder ready/stop marker already exists")
    metadata_path = args.output / "metadata.json"
    metadata: dict[str, object] = {
        "state": "STARTING",
        "port": args.port,
        "baud": args.baud,
        "watchdog_seconds": args.seconds,
        "minimum_rate_hz": args.minimum_rate_hz,
        "timestamp_basis": "Windows time.time_ns and time.monotonic_ns at serial read completion",
        "sensor_sample_time_available": False,
        "axis_mapping": "product Frame3/Roll -> experimental Pitch; Frame4/Yaw -> experimental Yaw",
        "offset_roll_deg": args.offset_roll_deg,
        "offset_yaw_deg": args.offset_yaw_deg,
        "pitch_sign": args.pitch_sign,
        "yaw_sign": args.yaw_sign,
        "motor_commands_sent": False,
        "utc_start_ns": time.time_ns(),
        "monotonic_start_ns": time.monotonic_ns(),
    }
    write_json(metadata_path, metadata)
    byte_count = chunk_count = frame_count = non_encoder_frames = 0
    reader = FrameReader()
    started = time.monotonic()
    first_valid_ns: int | None = None
    last_valid_ns: int | None = None
    stopped_by_request = False
    try:
        with open_serial_without_reset(args.port, args.baud) as port, (
            args.output / "raw.bin"
        ).open("xb") as raw_file, (args.output / "receive_times.csv").open(
            "x", newline="", encoding="utf-8"
        ) as timing_file, (args.output / "samples.csv").open(
            "x", newline="", encoding="utf-8"
        ) as sample_file:
            timing_writer = csv.writer(timing_file)
            sample_writer = csv.writer(sample_file)
            timing_writer.writerow(("receive_utc_ns", "receive_monotonic_ns", "offset", "length"))
            sample_writer.writerow((
                "receive_utc_ns", "receive_monotonic_ns", "raw_roll_deg", "raw_pitch_deg",
                "raw_yaw_deg", "experimental_pitch_deg", "experimental_yaw_deg",
            ))
            metadata["state"] = "WAITING_FOR_MODE_2"
            write_json(metadata_path, metadata)
            deadline = started + args.seconds
            last_sample_flush = started
            while time.monotonic() < deadline:
                if args.stop_file.exists():
                    stopped_by_request = True
                    break
                data = port.read(max(1, port.in_waiting))
                receive_monotonic_ns = time.monotonic_ns()
                receive_utc_ns = time.time_ns()
                if data:
                    offset = raw_file.tell()
                    raw_file.write(data)
                    timing_writer.writerow((receive_utc_ns, receive_monotonic_ns, offset, len(data)))
                    byte_count += len(data)
                    chunk_count += 1
                for frame in reader.push(data):
                    if frame[1] != 2:
                        non_encoder_frames += 1
                        continue
                    decoded = parse_encoder_frame(frame)
                    if decoded is None:
                        continue
                    raw_roll, raw_pitch, raw_yaw = decoded
                    pitch = args.pitch_sign * wrap_degrees(raw_roll + args.offset_roll_deg)
                    yaw = args.yaw_sign * wrap_degrees(raw_yaw + args.offset_yaw_deg)
                    sample_writer.writerow((receive_utc_ns, receive_monotonic_ns, raw_roll,
                                             raw_pitch, raw_yaw, pitch, yaw))
                    frame_count += 1
                    last_valid_ns = receive_monotonic_ns
                    if first_valid_ns is None:
                        first_valid_ns = receive_monotonic_ns
                        metadata["state"] = "CAPTURING"
                        write_json(metadata_path, metadata)
                        args.ready_file.write_text(str(receive_monotonic_ns), encoding="ascii")
                        print("FRAME_ENCODER_CAPTURE_READY mode=2", flush=True)
                elapsed = time.monotonic() - started
                # The coordinator reads the stationary preflight window before motor start.
                # Flush only the small decoded stream, not raw/fsync per sample.
                if time.monotonic() - last_sample_flush >= .1:
                    sample_file.flush()
                    last_sample_flush = time.monotonic()
                if first_valid_ns is None and elapsed >= 3:
                    detail = ("motor mode 1 received; select encoder mode 2" if non_encoder_frames else
                              f"COM port opened but received zero bytes; verify live encoder values in "
                              f"Test Tool Nx mode 2, then disconnect and close Test Tool Nx"
                              if byte_count == 0 else
                              f"no valid mode-2 frame; bytes={byte_count}; discarded={reader.discarded_bytes}")
                    raise RuntimeError(detail)
                if last_valid_ns is not None and receive_monotonic_ns - last_valid_ns > 1_000_000_000:
                    raise RuntimeError("validated encoder stream stopped for more than 1 second")
            raw_file.flush(); os.fsync(raw_file.fileno())
            timing_file.flush(); os.fsync(timing_file.fileno())
            sample_file.flush(); os.fsync(sample_file.fileno())
        end_ns = time.monotonic_ns()
        elapsed_capture_s = max(0.0, (end_ns - (first_valid_ns or end_ns)) / 1e9)
        rate_hz = frame_count / elapsed_capture_s if elapsed_capture_s > 0 else 0.0
        if not stopped_by_request:
            raise TimeoutError("encoder watchdog reached before coordinated stop")
        if frame_count < 20 or rate_hz < args.minimum_rate_hz:
            raise RuntimeError(
                f"encoder rate below requirement: frames={frame_count}, rate={rate_hz:.3f} Hz"
            )
    except BaseException as error:
        metadata.update(
            state="INCOMPLETE", error=f"{type(error).__name__}: {error}",
            utc_end_ns=time.time_ns(), byte_count=byte_count, chunk_count=chunk_count,
            encoder_frames=frame_count, non_encoder_frames=non_encoder_frames,
            discarded_bytes=reader.discarded_bytes,
        )
        write_json(metadata_path, metadata)
        raise
    finally:
        args.ready_file.unlink(missing_ok=True)

    metadata.update(
        state="COMPLETE", utc_end_ns=time.time_ns(), byte_count=byte_count,
        chunk_count=chunk_count, encoder_frames=frame_count,
        non_encoder_frames=non_encoder_frames, discarded_bytes=reader.discarded_bytes,
        elapsed_capture_s=elapsed_capture_s, measured_rate_hz=rate_hz,
        coordinated_stop=True, raw_storage_complete=True,
    )
    write_json(metadata_path, metadata)
    print(
        f"FRAME_ENCODER_CAPTURE_COMPLETE frames={frame_count} rate_hz={rate_hz:.3f}",
        flush=True,
    )
    return 0


def main() -> int:
    return capture(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
