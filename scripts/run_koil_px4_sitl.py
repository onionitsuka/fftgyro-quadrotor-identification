"""Localhost-only PX4/Gazebo comparison. Never accepts serial or remote hardware endpoints.

標準X500とPX4標準制御の参考比較。前の簡易モデルと機体諸元・制御器は異なる。
LOCAL_POSITION_NEDはPX4推定状態。実機・Gazebo真値の計測とは称さない。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from pathlib import Path

from simulate_koil_pass import PassProfile
from hover_pitch_profile import HoverPitchProfile, pitch_quaternion, altitude_thrust

# PX4 requires paired x/y acceleration setpoints. ay=0 is valid, not ignored.
VELOCITY_MASK = 1 | 2 | 32 | 256 | 2048
POSITION_MASK = 8 | 16 | 32 | 64 | 128 | 256 | 2048


def yaw_ready(attitude) -> bool:
    """Do not start translation before the explicit zero-yaw target settles."""
    return abs(attitude.yaw) < math.radians(5) and abs(attitude.yawspeed) < math.radians(5)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=14540)
    parser.add_argument("--warmup-s", type=float, default=8)
    parser.add_argument("--model-label", default="PX4 stock Gazebo x500")
    parser.add_argument("--hold-s", type=float, default=1.5)
    parser.add_argument("--target-v", type=float, default=5.)
    parser.add_argument("--acceleration", type=float, default=4.)
    parser.add_argument("--altitude", type=float, default=2.)
    parser.add_argument("--pitch-amplitude-deg",type=float,default=0.)
    parser.add_argument("--pitch-period-s",type=float,default=2.)
    args = parser.parse_args()
    if not math.isfinite(args.altitude) or args.altitude <= 0:
        parser.error("altitude must be finite and positive")
    if not math.isfinite(args.hold_s) or args.hold_s <= 0:
        parser.error("hold-s must be finite and positive")
    if not all(math.isfinite(v) and v > 0 for v in (args.target_v,args.acceleration)) or args.target_v <= .2*args.acceleration:
        parser.error("Positive target/acceleration and trapezoidal ramp required")
    if not math.isfinite(args.pitch_amplitude_deg) or not 0 <= args.pitch_amplitude_deg <= 45 or not math.isfinite(args.pitch_period_s) or args.pitch_period_s <= 0:
        parser.error("Offline Pitch amplitude must be 0..45 deg and period positive")
    script_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    from pymavlink import mavutil
    args.output.mkdir(parents=True, exist_ok=False)
    mav = mavutil.mavlink
    connection = mavutil.mavlink_connection(f"udpin:127.0.0.1:{args.port}", source_system=245)
    if not connection.wait_heartbeat(timeout=25):
        raise RuntimeError("No localhost SITL heartbeat")
    system, component = connection.target_system, connection.target_component

    def command(cmd, *params):
        connection.mav.command_long_send(system, component, cmd, 0, *(list(params) + [0] * (7 - len(params))))

    for msg in (mav.MAVLINK_MSG_ID_LOCAL_POSITION_NED, mav.MAVLINK_MSG_ID_ATTITUDE):
        command(mav.MAV_CMD_SET_MESSAGE_INTERVAL, msg, 20000)
    profile = PassProfile(hold_s=args.hold_s,acceleration=args.acceleration)
    duration = profile.hover_s + 2 * profile.ramp_duration(args.target_v) + profile.hold_s + 12
    pitch_profile = HoverPitchProfile(args.pitch_amplitude_deg,args.pitch_period_s) if args.pitch_amplitude_deg else None
    if pitch_profile:
        duration = pitch_profile.duration_s
    altitude_integral = 0.
    last_command_boot = None
    start = time.monotonic()
    last_send = last_heartbeat = -1.0
    local = attitude = None
    origin = None
    mission_boot_ms = None
    stable_since = None
    rows, armed, offboard = [], False, False
    armed_request = False
    statuses = []
    try:
        while time.monotonic() - start < 100:
            now = time.monotonic() - start
            message = connection.recv_match(blocking=True, timeout=.01)
            if message:
                kind = message.get_type()
                if kind == "LOCAL_POSITION_NED":
                    local = message
                elif kind == "ATTITUDE":
                    attitude = message
                elif kind == "HEARTBEAT":
                    armed = bool(message.base_mode & mav.MAV_MODE_FLAG_SAFETY_ARMED)
                    offboard = ((message.custom_mode >> 16) & 255) == 6
                elif kind == "STATUSTEXT":
                    statuses.append(message.text)
            if now - last_heartbeat > 1:
                connection.mav.heartbeat_send(mav.MAV_TYPE_GCS, mav.MAV_AUTOPILOT_INVALID, 0, 0, mav.MAV_STATE_ACTIVE)
                last_heartbeat = now
            mission_t = ((local.time_boot_ms - mission_boot_ms) / 1000
                         if local is not None and mission_boot_ms is not None else None)
            if mission_t is not None and mission_t >= duration:
                break
            if now - last_send >= .04:
                timestamp = local.time_boot_ms if local else 0
                speed, acceleration = profile.reference(mission_t or 0, args.target_v)
                goal = profile.position_reference(mission_t, args.target_v) if mission_t is not None else None
                if pitch_profile is not None and mission_t is not None:
                    pitch_goal = pitch_profile.reference_deg(mission_t)
                    dt = (timestamp-last_command_boot)/1000 if last_command_boot is not None else 0.
                    thrust,altitude_integral = altitude_thrust(local.z,local.vz,altitude_integral,dt,pitch_goal)
                    connection.mav.set_attitude_target_send(timestamp,system,component,7,
                        pitch_quaternion(pitch_goal),0,0,0,thrust)
                    last_command_boot = timestamp
                elif mission_t is None or goal is not None:
                    # Full position hold for takeoff and after reference braking.
                    xy = origin[:2] + goal if goal is not None else [0., 0.]
                    mask = POSITION_MASK
                    connection.mav.set_position_target_local_ned_send(timestamp, system, component,
                        mav.MAV_FRAME_LOCAL_NED, mask, xy[0], xy[1], -args.altitude, 0, 0, 0, 0, 0, 0, 0, 0)
                else:
                    # vx/vy and vertical position, with reference ax feedforward.
                    mask = VELOCITY_MASK
                    connection.mav.set_position_target_local_ned_send(timestamp, system, component,
                        mav.MAV_FRAME_LOCAL_NED, mask, 0, 0, -args.altitude, speed, 0, 0, acceleration, 0, 0, 0, 0)
                last_send = now
            if now > args.warmup_s and not armed_request:
                command(mav.MAV_CMD_DO_SET_MODE, mav.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, 6)
                command(mav.MAV_CMD_COMPONENT_ARM_DISARM, 1)
                armed_request = True
            if armed_request and (not armed or not offboard) and now > args.warmup_s + 10:
                raise RuntimeError(f"SITL did not arm/enter Offboard: {statuses[-12:]}")
            if local is None or attitude is None:
                continue
            if mission_boot_ms is None and armed and offboard:
                settled = (abs(local.z + args.altitude) < .1 and
                           (local.vx**2 + local.vy**2 + local.vz**2) ** .5 < .15 and yaw_ready(attitude))
                stable_since = now if settled and stable_since is None else stable_since if settled else None
                if stable_since is not None and now - stable_since >= 1:
                    origin = [local.x, local.y, local.z]
                    mission_boot_ms = local.time_boot_ms
            elif mission_boot_ms is not None and message and message.get_type() == "LOCAL_POSITION_NED":
                if not armed or not offboard:
                    raise RuntimeError("SITL left the armed Offboard state during the comparison")
                t = (local.time_boot_ms - mission_boot_ms) / 1000
                if rows and t <= rows[-1]["time_s"]:
                    continue
                rows.append(dict(time_s=t, x=local.x - origin[0], y=local.y - origin[1],
                    height=origin[2] - local.z, vx=local.vx, vy=local.vy, vz=local.vz,
                    pitch_deg=attitude.pitch * 180 / 3.141592653589793,
                    roll_deg=math.degrees(attitude.roll), yaw_deg=math.degrees(attitude.yaw),
                    pitch_rate_deg_s=math.degrees(attitude.pitchspeed),
                    reference_pitch_deg=pitch_profile.reference_deg(t) if pitch_profile else float("nan"),
                    yaw_rate_deg_s=math.degrees(attitude.yawspeed), reference_yaw_deg=0.,
                    reference_vx=profile.reference(t, args.target_v)[0]))
        if not rows or rows[-1]["time_s"] < duration - .1:
            raise RuntimeError(f"Incomplete SITL mission: {statuses[-12:]}")
        with (args.output / "px4_gz_x500.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        (args.output / "summary.json").write_text(json.dumps(dict(status="COMPLETED",
            comparison_valid=True, script_sha256=script_sha,
            model=args.model_label, observation="LOCAL_POSITION_NED estimator, not ground truth",
            controller=("PX4 Offboard attitude/rate control; common external altitude PI; no XY hold" if pitch_profile
                        else "PX4 Offboard velocity/acceleration, z position; final xyz position; gains set by launcher"),
            matched=("Same velocity and acceleration reference and duration only" if args.model_label == "PX4 stock Gazebo x500"
                     else "See launcher summary for aircraft/controller matching"),
            unmatched=("Mass, inertia, propulsion, aerodynamic loads, feedback controller and estimator" if args.model_label == "PX4 stock Gazebo x500"
                       else "See launcher summary; do not infer loads-only equivalence"),
            duration_s=duration, hold_s=profile.hold_s, target_vx_m_s=args.target_v,
            acceleration_m_s2=args.acceleration, samples=len(rows), statuses=statuses,
            yaw_target_deg=0., initial_yaw_gate="Within 5 deg and 5 deg/s for 1 s",
            excitation=("Hann-windowed Pitch cosine; horizontal position not held; fixed common altitude PI" if pitch_profile else "Forward velocity pass"),
            max_abs_yaw_error_deg=max(abs(r["yaw_deg"]) for r in rows)), ensure_ascii=False, indent=2) + "\n")
        (args.output / "mission_clock.json").write_text(json.dumps(dict(
            mission_boot_ms=mission_boot_ms, estimator_origin_ned_m=origin,
            duration_s=duration, hold_s=profile.hold_s,target_vx_m_s=args.target_v,
            acceleration_m_s2=args.acceleration), indent=2) + "\n")
    finally:
        # Local simulation only: leave it in Land; launcher owns process cleanup.
        command(mav.MAV_CMD_NAV_LAND)
        connection.close()


if __name__ == "__main__":
    main()
