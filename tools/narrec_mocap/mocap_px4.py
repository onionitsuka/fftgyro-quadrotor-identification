#!/usr/bin/env python3
"""Record complete NatNet frames; optionally supply vision pose to PX4.

No arm, flight mode, parameter or actuator commands are implemented.
NatNet decoder is supplied by MAVProxy; its private complete-frame hook is
checked at startup because the public rigid-body callback omits validity.
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
import socket
import time

# MAVLink 2 is needed to carry the covariance and reset counter extensions.
os.environ["MAVLINK20"] = "1"

import numpy as np
from scipy.spatial.transform import Rotation


FIELDS = ["rx_unix_ns", "rx_monotonic_ns", "frame", "server_timestamp_s",
          "camera_mid_exposure_ticks", "server_data_received_ticks", "server_transmit_ticks",
          "rigid_body_id", "tracking_valid", "motive_recording", "mean_error_m",
          "x_m", "y_m", "z_m", "qx", "qy", "qz", "qw", "sent_vision", "reason"]


def load_config(path: Path) -> dict:
    c = json.loads(path.read_text(encoding="utf-8"))
    for key in ("world_from_motive", "rigid_from_body"):
        a = np.asarray(c[key], dtype=float)
        if (a.shape != (3, 3) or not np.isfinite(a).all()
                or not np.allclose(a.T @ a, np.eye(3), atol=1e-6)
                or not np.isclose(np.linalg.det(a), 1, atol=1e-6)):
            raise ValueError(f"{key}: expected a proper rotation matrix")
    for key in ("origin_motive_m", "body_to_rigid_m", "bounds_min_m", "bounds_max_m"):
        a = np.asarray(c[key], dtype=float)
        if a.shape != (3,) or not np.isfinite(a).all():
            raise ValueError(f"{key}: expected three finite values")
    if np.any(np.asarray(c["bounds_min_m"]) >= c["bounds_max_m"]):
        raise ValueError("invalid bounds")
    for key in ("position_sigma_m", "angle_sigma_rad", "max_age_s", "send_hz"):
        if not math.isfinite(c[key]) or c[key] <= 0:
            raise ValueError(f"{key}: expected a positive finite value")
    if not 30 <= c["send_hz"] <= 50:
        raise ValueError("send_hz must be 30..50 for the planned PX4 route")
    return c


def transform_pose(c: dict, pos: list, quat_xyzw: list) -> tuple[np.ndarray, np.ndarray]:
    p, q = np.asarray(pos, dtype=float), np.asarray(quat_xyzw, dtype=float)
    if p.shape != (3,) or q.shape != (4,) or not np.isfinite(np.r_[p, q]).all():
        raise ValueError("missing/nonfinite pose")
    if not 0.95 <= np.linalg.norm(q) <= 1.05:
        raise ValueError("bad quaternion norm")
    a = np.asarray(c["world_from_motive"])
    r = a @ Rotation.from_quat(q).as_matrix() @ np.asarray(c["rigid_from_body"])
    # body_to_rigid is the vector from FC/body origin to the rigid origin, in FRD.
    body_pos = a @ (p - np.asarray(c["origin_motive_m"])) - r @ np.asarray(c["body_to_rigid_m"])
    return body_pos, Rotation.from_matrix(r).as_euler("xyz")


def eligible(c: dict, row: dict, now_ns: int) -> tuple[bool, str]:
    if not row["tracking_valid"]:
        return False, "untracked_or_missing"
    if not 0 <= (now_ns - row["rx_monotonic_ns"]) / 1e9 <= c["max_age_s"]:
        return False, "stale"
    try:
        p, _ = transform_pose(c, [row[k] for k in ("x_m", "y_m", "z_m")],
                              [row[k] for k in ("qx", "qy", "qz", "qw")])
    except ValueError:
        return False, "invalid_pose"
    if np.any(p < c["bounds_min_m"]) or np.any(p > c["bounds_max_m"]):
        return False, "out_of_bounds"
    return True, "valid"


def frame_row(data, rigid_id: int, unix_ns: int, mono_ns: int) -> dict:
    bodies = [b for b in data.rigid_body_data.rigid_body_list if b.id_num == rigid_id]
    if len(bodies) > 1:
        raise ValueError("duplicate rigid body ID")
    b = bodies[0] if bodies else None
    row = dict.fromkeys(FIELDS, "")
    row.update(rx_unix_ns=unix_ns, rx_monotonic_ns=mono_ns,
               frame=data.prefix_data.frame_number,
               server_timestamp_s=data.suffix_data.timestamp, rigid_body_id=rigid_id,
               camera_mid_exposure_ticks=data.suffix_data.stamp_camera_mid_exposure,
               server_data_received_ticks=data.suffix_data.stamp_data_received,
               server_transmit_ticks=data.suffix_data.stamp_transmit,
               tracking_valid=bool(b and b.tracking_valid),
               motive_recording=data.suffix_data.is_recording,
               mean_error_m=b.error if b else "", sent_vision=False)
    if b:
        row.update(zip(("x_m", "y_m", "z_m"), b.pos))
        row.update(zip(("qx", "qy", "qz", "qw"), b.rot))
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--server", required=True, help="Motive LAN IPv4")
    ap.add_argument("--client", required=True, help="this PC LAN IPv4")
    ap.add_argument("--multicast", action="store_true", help="default is unicast")
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True, help="new run directory")
    ap.add_argument("--duration", type=float, default=60)
    ap.add_argument("--mavlink", help="optional connection, e.g. /dev/ttyUSB0 or udpin:127.0.0.1:14551")
    ap.add_argument("--baud", type=int, default=921600)
    ap.add_argument("--send-vision", action="store_true", help="send VISION_POSITION_ESTIMATE only")
    args = ap.parse_args()
    c = load_config(args.config)
    if args.duration <= 0 or not math.isfinite(args.duration):
        ap.error("duration must be positive and finite")
    if args.send_vision and (not args.mavlink or c["bench_transform_verified"] is not True):
        ap.error("send-vision requires mavlink and bench_transform_verified=true")
    from MAVProxy.modules.mavproxy_optitrack.NatNetClient import NatNetClient

    fifo: queue.Queue = queue.Queue(maxsize=2048)
    errors: queue.Queue = queue.Queue()
    unpack = getattr(NatNetClient, "_NatNetClient__unpack_mocap_data", None)
    if unpack is None:
        raise RuntimeError("unsupported MAVProxy decoder: complete-frame hook missing")

    class Capture(NatNetClient):
        def _NatNetClient__data_thread_function(self, in_socket, stop, gprint_level):
            # Finite recv timeout makes shutdown work even after LAN disconnect.
            # Capture decode failures in the main loop instead of losing a thread.
            in_socket.settimeout(0.2)
            while not stop():
                try:
                    raw, _ = in_socket.recvfrom(65536)
                except socket.timeout:
                    continue
                except OSError as exc:
                    if not stop():
                        errors.put_nowait(repr(exc))
                    return
                try:
                    self._NatNetClient__process_message(raw, 0)
                except Exception as exc:
                    errors.put_nowait(repr(exc))
                    return

        def _NatNetClient__unpack_mocap_data(self, *a):
            unix_ns, mono_ns = time.time_ns(), time.monotonic_ns()
            try:
                result = unpack(self, *a)
                fifo.put_nowait(frame_row(result[1], c["rigid_body_id"], unix_ns, mono_ns))
                return result
            except Exception as exc:
                errors.put_nowait(repr(exc))
                raise

    args.output.mkdir(parents=True, exist_ok=False)
    meta = {"arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "config": c, "start_unix_ns": time.time_ns(),
            "timestamp_policy": "receive time; server timestamp separately retained; not hardware synchronized",
            "decoder_sha256": hashlib.sha256(Path(__import__(NatNetClient.__module__, fromlist=["x"]).__file__).read_bytes()).hexdigest()}
    (args.output / "capture.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    conn = None
    if args.mavlink:
        from pymavlink import mavutil
        conn = mavutil.mavlink_connection(args.mavlink, baud=args.baud, source_system=245, source_component=197)
        hb = conn.wait_heartbeat(timeout=10)
        if hb is None or hb.autopilot != mavutil.mavlink.MAV_AUTOPILOT_PX4:
            raise RuntimeError("PX4 heartbeat not received; check routing/system ID")
    client = Capture()
    client.set_server_address(args.server)
    client.set_client_address(args.client)
    client.set_use_multicast(args.multicast)
    client.set_print_level(0)
    count = sent = 0
    last_frame = None
    last_server_time = None
    last_send = 0.0
    start = last_status = last_sync = time.monotonic()
    covariance = [0.0] * 21
    for i in (0, 6, 11):
        covariance[i] = c["position_sigma_m"] ** 2
    for i in (15, 18, 20):
        covariance[i] = c["angle_sigma_rad"] ** 2
    failure = None
    try:
        with (args.output / "mocap.csv").open("x", newline="", encoding="utf-8") as f, \
                (args.output / "telemetry.jsonl").open("x", encoding="utf-8") as tf:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            if not client.run():
                raise RuntimeError("NatNet socket startup failed")
            while time.monotonic() - start < args.duration:
                if not errors.empty():
                    raise RuntimeError("NatNet decode/record failure: " + errors.get_nowait())
                try:
                    row = fifo.get(timeout=0.01)
                except queue.Empty:
                    row = None
                if row is not None:
                    good, reason = eligible(c, row, time.monotonic_ns())
                    if (last_frame is not None and row["frame"] <= last_frame
                            or last_server_time is not None and row["server_timestamp_s"] <= last_server_time):
                        good, reason = False, "frame_or_clock_reset_restart_required"
                    last_frame = row["frame"]
                    last_server_time = row["server_timestamp_s"]
                    now = time.monotonic()
                    if good and args.send_vision and now - last_send >= 1 / c["send_hz"]:
                        p, e = transform_pose(c, [row[k] for k in ("x_m", "y_m", "z_m")],
                                              [row[k] for k in ("qx", "qy", "qz", "qw")])
                        conn.mav.vision_position_estimate_send(row["rx_unix_ns"] // 1000, *p, *e, covariance, 0)
                        row["sent_vision"] = True
                        last_send = now
                        sent += 1
                    row["reason"] = reason
                    writer.writerow(row)
                    count += 1
                    if args.send_vision and reason == "frame_or_clock_reset_restart_required":
                        raise RuntimeError(reason)
                if conn:
                    for _ in range(20):
                        msg = conn.recv_match(blocking=False)
                        if msg is None:
                            break
                        tf.write(json.dumps({"rx_unix_ns": time.time_ns(), "message": msg.to_dict()}, default=str) + "\n")
                now = time.monotonic()
                if now - last_sync >= 1:
                    for stream in (f, tf):
                        stream.flush()
                        os.fsync(stream.fileno())
                    last_sync = now
                if now - last_status >= 2:
                    print(f"elapsed={now-start:.1f}s frames={count} vision_sent={sent} queue={fifo.qsize()}", flush=True)
                    last_status = now
            if count == 0:
                raise RuntimeError("no mocap frames; recording NOT verified")
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        failure = repr(exc)
        raise
    finally:
        if getattr(client, "data_socket", None) and getattr(client, "command_socket", None):
            client.shutdown()
        if conn:
            conn.close()
        meta.update(frames=count, vision_sent=sent, error=failure,
                    end_unix_ns=time.time_ns(), server_version=client.get_server_version(),
                    natnet_version=client.get_nat_net_version_server())
        (args.output / "capture.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {count} frames to {args.output}. Motive take and FC SD log must be saved separately.")


if __name__ == "__main__":
    main()
