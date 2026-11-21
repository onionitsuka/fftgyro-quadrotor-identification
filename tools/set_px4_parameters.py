#!/usr/bin/env python3
"""Set and verify explicitly named PX4 MAVLink parameters."""

from __future__ import annotations

import argparse
import json
import math
import struct
import time
from pathlib import Path

from pymavlink import mavutil


def parameter_name(message: object) -> str:
    value = message.param_id
    if isinstance(value, bytes):
        return value.split(b"\0", 1)[0].decode("ascii")
    return value.rstrip("\0")


def decode(value: float, parameter_type: int) -> int | float:
    if parameter_type == mavutil.mavlink.MAV_PARAM_TYPE_REAL32:
        return float(value)
    formats = {
        mavutil.mavlink.MAV_PARAM_TYPE_UINT8: "B",
        mavutil.mavlink.MAV_PARAM_TYPE_INT8: "b",
        mavutil.mavlink.MAV_PARAM_TYPE_UINT16: "H",
        mavutil.mavlink.MAV_PARAM_TYPE_INT16: "h",
        mavutil.mavlink.MAV_PARAM_TYPE_UINT32: "I",
        mavutil.mavlink.MAV_PARAM_TYPE_INT32: "i",
    }
    code = formats[parameter_type]
    raw = struct.pack("<f", float(value))
    return int(struct.unpack("<" + code, raw[: struct.calcsize(code)])[0])


def encode(value: int | float, parameter_type: int) -> float:
    if parameter_type == mavutil.mavlink.MAV_PARAM_TYPE_REAL32:
        return float(value)
    formats = {
        mavutil.mavlink.MAV_PARAM_TYPE_UINT8: "B",
        mavutil.mavlink.MAV_PARAM_TYPE_INT8: "b",
        mavutil.mavlink.MAV_PARAM_TYPE_UINT16: "H",
        mavutil.mavlink.MAV_PARAM_TYPE_INT16: "h",
        mavutil.mavlink.MAV_PARAM_TYPE_UINT32: "I",
        mavutil.mavlink.MAV_PARAM_TYPE_INT32: "i",
    }
    code = formats[parameter_type]
    raw = struct.pack("<" + code, int(value)).ljust(4, b"\0")
    return struct.unpack("<f", raw)[0]


def read_parameter(link: mavutil.mavfile, system: int, component: int, name: str):
    link.mav.param_request_read_send(system, component, name.encode("ascii"), -1)
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        message = link.recv_match(type="PARAM_VALUE", blocking=True, timeout=0.3)
        if message is not None and parameter_name(message) == name:
            return decode(float(message.param_value), int(message.param_type)), int(message.param_type)
    raise RuntimeError(f"parameter read timeout: {name}")


def same(actual: int | float, expected: int | float) -> bool:
    if isinstance(actual, float) or isinstance(expected, float):
        return math.isclose(float(actual), float(expected), rel_tol=1e-6, abs_tol=1e-6)
    return actual == expected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("device")
    parser.add_argument("settings", type=Path, help="JSON object of parameter names to values")
    parser.add_argument("--baud", type=int, default=115200)
    args = parser.parse_args()
    requested = json.loads(args.settings.read_text(encoding="utf-8"))

    link = mavutil.mavlink_connection(args.device, baud=args.baud, autoreconnect=False)
    heartbeat = link.wait_heartbeat(timeout=5)
    if heartbeat is None or heartbeat.autopilot != mavutil.mavlink.MAV_AUTOPILOT_PX4:
        raise RuntimeError("PX4 heartbeat not found")
    if heartbeat.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED:
        raise RuntimeError("refusing to change parameters while armed")
    system = heartbeat.get_srcSystem()
    component = heartbeat.get_srcComponent()

    report = {}
    for name, target in requested.items():
        before, parameter_type = read_parameter(link, system, component, name)
        link.mav.param_set_send(
            system,
            component,
            name.encode("ascii"),
            encode(target, parameter_type),
            parameter_type,
        )
        after, returned_type = read_parameter(link, system, component, name)
        if returned_type != parameter_type or not same(after, target):
            raise RuntimeError(f"verification failed: {name}={after}, expected {target}")
        report[name] = {"before": before, "after": after, "type": parameter_type}
        print(f"{name}: {before} -> {after}", flush=True)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
