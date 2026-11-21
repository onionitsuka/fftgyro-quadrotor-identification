"""Run one frozen plant with the same real PX4 SITL binary and mission.

Only loopback TCP/UDP; starts a new isolated PX4 rootfs, never uses hardware.
The baseline is matched thrust/torque physics, not unmodified Gazebo X500.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from matched_px4_plant import Plant, common_parameters, GAZEBO_AERO
from simulate_graybox_forward_flight import euler_deg


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv=None, *, plant_factory=Plant, parameters_factory=common_parameters, metadata=None,
         mission_command_factory=None, max_simulation_s=100, before_step=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--px4-root", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--route", choices=("physical", "gazebo_aero", "measured", "command_route"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-v",type=float,default=5.)
    parser.add_argument("--acceleration",type=float,default=4.)
    parser.add_argument("--hold-s",type=float,default=1.5)
    parser.add_argument("--altitude",type=float,default=2.)
    args = parser.parse_args(argv)
    from pymavlink import mavutil
    mav = mavutil.mavlink
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    artifact = json.loads(args.models.read_text())
    schedule = json.loads(args.schedule.read_text())
    if schedule["model_sha256"] != sha(args.models):
        raise ValueError("Coefficient/schedule mismatch")
    plant = plant_factory(artifact, schedule, args.route)
    build = args.px4_root / "build/px4_sitl_default"
    rootfs = args.output / "rootfs"
    rootfs.mkdir()
    (rootfs / "etc").symlink_to((build / "etc").resolve(), target_is_directory=True)
    params = parameters_factory(artifact)
    startup = args.output / "startup.sh"
    startup.write_text("#!/bin/sh\n. etc/init.d-posix/rcS\n" +
                       "\n".join(f"param set {k} {v:.12g}" for k, v in params.items()) +
                       "\nparam show\necho MATCHED_PARAMETERS_READY\n")
    env = dict(os.environ, PX4_SYS_AUTOSTART="10016", PX4_SIM_MODEL="iris")
    log = (args.output / "px4.log").open("w")
    process = subprocess.Popen([str(build / "bin/px4"), "-d", "-s", str(startup)],
                               cwd=rootfs, env=env, stdout=log, stderr=subprocess.STDOUT)
    mission = None
    bridge = None
    rows = []
    wall_start = time.monotonic()
    mission_log = (args.output / "mission.log").open("w")
    try:
        bridge = mavutil.mavlink_connection("tcpin:127.0.0.1:4560", source_system=250)
        command = np.zeros(4)
        armed = False
        for step in range(round(max_simulation_s / plant.dt)):
            elapsed = step * plant.dt
            timestamp = 1000000 + int(round(elapsed * 1e6))
            sensor = plant.sensor_values()
            for sensor_id in (1, 2, 0):
                bridge.mav.hil_sensor_send(timestamp, *sensor["accel"], *sensor["gyro"],
                    *sensor["mag"], sensor["pressure_hpa"], 0, 488 - plant.state[2], 20,
                    63 if sensor_id == 2 else 8191, sensor_id)
            if step % 25 == 0:
                speed = float(np.linalg.norm(plant.state[3:5]))
                cog = int(round(np.degrees(np.arctan2(plant.state[4], plant.state[3])) % 360 * 100))
                bridge.mav.hil_gps_send(timestamp, 3, sensor["lat"], sensor["lon"], sensor["alt_mm"],
                    20, 30, int(round(speed * 100)), *(int(round(v * 100)) for v in plant.state[3:6]), cog, 14)
            # Ground truth is published independently of the EKF feedback.
            if step % 5 == 0:
                bridge.mav.hil_state_quaternion_send(timestamp, plant.state[6:10].tolist(),
                    *plant.state[10:13], sensor["lat"], sensor["lon"], sensor["alt_mm"],
                    *(int(round(v * 100)) for v in plant.state[3:6]), 0, 0,
                    *(int(round(v / 9.80665 * 1000)) for v in sensor["accel"]))
            message = bridge.recv_match(type="HIL_ACTUATOR_CONTROLS", blocking=True, timeout=.1)
            if message:
                command = np.array(message.controls[:4])
                armed = bool(message.mode & mav.MAV_MODE_FLAG_SAFETY_ARMED)
                if np.max(command) > 1.001 or np.min(command) < -.001:
                    raise ValueError("Unexpected non-normalized PX4 motor outputs")
            if before_step is not None:
                before_step(plant, elapsed, args.output)
            plant.step(command, armed)
            if step % 5 == 0:
                x = plant.state
                rows.append(dict(boot_time_s=timestamp / 1e6, time_s=elapsed,
                    x=float(x[0]), y=float(x[1]), height=float(-x[2]),
                    vx=float(x[3]), vy=float(x[4]), vz=float(x[5]), pitch_deg=float(euler_deg(x[6:10])[1]),
                    roll_deg=float(euler_deg(x[6:10])[0]),yaw_deg=float(euler_deg(x[6:10])[2]),
                    **dict(zip(("Fx","Fy","Fz","Mx","My","Mz"),map(float,plant.load))),
                    **{f"command_M{j+1}": float(command[j]) for j in range(4)},
                    **{f"rpm_M{j+1}": float(plant.rotor[j] * 60 / (2 * np.pi)) for j in range(4)}))
            if mission is None and elapsed > 2:
                mission_command = [sys.executable, str(Path(__file__).with_name("run_koil_px4_sitl.py")),
                    "--output", str(args.output / "mission"), "--warmup-s", "20",
                    "--target-v",str(args.target_v),"--acceleration",str(args.acceleration),
                    "--hold-s",str(args.hold_s),"--altitude",str(args.altitude)]
                if mission_command_factory is not None:
                    mission_command = mission_command_factory(args.output)
                mission = subprocess.Popen(mission_command,
                    stdout=mission_log, stderr=subprocess.STDOUT)
            if mission is not None and mission.poll() is not None:
                if mission.returncode:
                    raise RuntimeError("PX4 mission failed; see mission.log")
                break
            if process.poll() is not None:
                raise RuntimeError("PX4 exited; see px4.log")
            # Real-time pacing keeps the common 25-Hz Offboard sender synchronized.
            wait = wall_start + (step + 1) * plant.dt - time.monotonic()
            if wait > 0:
                time.sleep(wait)
        else:
            raise RuntimeError(f"Incomplete mission after {max_simulation_s} simulated seconds")
        with (args.output / "plant_full.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        summary = dict(status="COMPLETED", route=args.route,
            observation="Full plant ground truth and separate PX4 estimator trace",
            model_sha256=sha(args.models), schedule_sha256=sha(args.schedule),
            binary_sha256=sha(build / "bin/px4"), common_parameters=params,
            common_settings=artifact["settings"], common_motor_models=artifact["motor_models"],
            dt_s=plant.dt, wind_m_s=[0,0,0], voltage_V=16,
            sensor_noise="Ideal IMU/GPS; seeded Gaussian mag SD 0.0005 gauss and baro SD 0.001 hPa, seed 20261005, same per step",
            changed="Aircraft.loads only; same integration, motors, PX4 binary, gains and sensors",
            limitations=["Thrust/torque physical baseline, NOT native Gazebo X500",
                "Frozen initial-hover anchor and conditional inertia; not free-flight validated",
                "Distinct flight roots with identical starting configuration; EKF/integrators evolve separately",
                "Same reference, not identical closed-loop motor inputs"])
        if metadata:
            summary.update(metadata)
        log.flush()
        # Full PX4 gains/allocation snapshot, not just the explicitly changed values.
        observed = {}
        for name, value in re.findall(r"([A-Z][A-Z0-9_]+)\s+\[[0-9,]+\]\s*:\s*(\S+)",
                                      (args.output / "px4.log").read_text()):
            if name.startswith(("MC_", "MPC_", "CA_", "PWM_MAIN_")):
                observed[name] = float(value)
        if not observed:
            raise ValueError("No PX4 controller/allocation parameter snapshot")
        summary["controller_parameters"] = observed
        summary["gazebo_aero"] = GAZEBO_AERO if args.route == "gazebo_aero" else None
        summary["controller_parameters_sha256"] = hashlib.sha256(
            json.dumps(observed, sort_keys=True).encode()).hexdigest()
        (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    finally:
        if rows:
            with (args.output / "plant_full.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        if mission and mission.poll() is None:
            mission.terminate()
            mission.wait(timeout=5)
        if bridge:
            bridge.close()
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        log.close()
        mission_log.close()


if __name__ == "__main__":
    main()
