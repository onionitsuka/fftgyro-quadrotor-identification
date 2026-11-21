#!/usr/bin/env python3
"""Read-only live monitor for PX4 RC kill-switch verification."""

from __future__ import annotations

import argparse
import math
import time

from pymavlink import mavutil


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("device")
    parser.add_argument("--seconds", type=float, default=60.0)
    args = parser.parse_args()

    link = mavutil.mavlink_connection(args.device, baud=115200, autoreconnect=False)
    heartbeat = link.wait_heartbeat(timeout=5)
    if heartbeat is None:
        raise RuntimeError("no MAVLink heartbeat")

    system = heartbeat.get_srcSystem()
    component = heartbeat.get_srcComponent()
    for message_id, interval_us in (
        (mavutil.mavlink.MAVLINK_MSG_ID_HEARTBEAT, 200_000),
        (mavutil.mavlink.MAVLINK_MSG_ID_RC_CHANNELS, 100_000),
        (mavutil.mavlink.MAVLINK_MSG_ID_BATTERY_STATUS, 500_000),
    ):
        link.mav.command_long_send(
            system,
            component,
            mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
            0,
            message_id,
            interval_us,
            0,
            0,
            0,
            0,
            0,
        )

    deadline = time.monotonic() + args.seconds
    last_ch8 = None
    last_ch9 = None
    last_armed = None
    last_voltage_decivolt = None
    print("monitor ready: toggle the configured kill switch ON, then OFF", flush=True)
    while time.monotonic() < deadline:
        message = link.recv_match(
            type=["RC_CHANNELS", "HEARTBEAT", "BATTERY_STATUS", "STATUSTEXT"],
            blocking=True,
            timeout=0.5,
        )
        if message is None:
            continue
        if message.get_type() == "RC_CHANNELS":
            ch8 = int(message.chan8_raw)
            ch9 = int(message.chan9_raw)
            if ch8 != last_ch8 or ch9 != last_ch9:
                print(f"RC8={ch8} us RC9={ch9} us", flush=True)
                last_ch8 = ch8
                last_ch9 = ch9
        elif message.get_type() == "HEARTBEAT":
            armed = bool(message.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
            if armed != last_armed:
                print(f"ARMED={armed}", flush=True)
                last_armed = armed
        elif message.get_type() == "BATTERY_STATUS":
            cells = [int(value) for value in message.voltages if 0 < int(value) < 65_535]
            voltage = sum(cells) / 1000.0 if cells else float("nan")
            decivolt = round(voltage * 10) if math.isfinite(voltage) else None
            if decivolt != last_voltage_decivolt:
                print(f"BATTERY={voltage:.2f} V", flush=True)
                last_voltage_decivolt = decivolt
        elif message.get_type() == "STATUSTEXT":
            print(f"PX4: {message.text}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
