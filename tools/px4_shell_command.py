#!/usr/bin/env python3
"""Run one PX4 NSH command through MAVLink SERIAL_CONTROL."""

from __future__ import annotations

import argparse
import time

from pymavlink import mavutil


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("device")
    parser.add_argument("command")
    parser.add_argument("--timeout", type=float, default=5.0)
    args = parser.parse_args()

    link = mavutil.mavlink_connection(args.device, baud=115200, autoreconnect=False)
    if link.wait_heartbeat(timeout=5) is None:
        raise RuntimeError("no MAVLink heartbeat")

    payload = (args.command.rstrip("\r\n") + "\n").encode("ascii")
    if len(payload) > 70:
        raise ValueError("command is longer than one SERIAL_CONTROL frame")
    flags = (
        mavutil.mavlink.SERIAL_CONTROL_FLAG_EXCLUSIVE
        | mavutil.mavlink.SERIAL_CONTROL_FLAG_RESPOND
        | mavutil.mavlink.SERIAL_CONTROL_FLAG_MULTI
    )
    link.mav.serial_control_send(
        mavutil.mavlink.SERIAL_CONTROL_DEV_SHELL,
        flags,
        0,
        0,
        len(payload),
        list(payload.ljust(70, b"\0")),
    )

    deadline = time.monotonic() + args.timeout
    output = bytearray()
    while time.monotonic() < deadline:
        message = link.recv_match(type="SERIAL_CONTROL", blocking=True, timeout=0.3)
        if message is None:
            continue
        if int(message.device) != mavutil.mavlink.SERIAL_CONTROL_DEV_SHELL:
            continue
        count = int(message.count)
        if count:
            output.extend(bytes(message.data[:count]))
            deadline = time.monotonic() + 0.8
    print(output.decode("utf-8", errors="replace"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
