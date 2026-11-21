"""Native Gazebo X500 sensors/physics with an isolated, real-aircraft SDF variant.

No hardware endpoints, stock SDF edits or global process kills. Native motor
mapping/response intentionally differs from the matched Python plant.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

from build_native_gazebo_model import build
from matched_px4_plant import common_parameters


def initial_pose_enu(yaw_ned: float = 0.) -> str:
    """ENU east/CCW heading -> PX4 NED north/CW heading."""
    return f"0,0,0,0,0,{math.pi / 2 - yaw_ned:.16g}"


def stop(process: subprocess.Popen | None) -> None:
    if process is None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--px4-root", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--route", choices=("baseline","measured","command_route"), default="baseline")
    parser.add_argument("--schedule", type=Path)
    parser.add_argument("--hold-s", type=float, default=1.5)
    parser.add_argument("--target-v", type=float, default=5.)
    parser.add_argument("--acceleration", type=float, default=4.)
    parser.add_argument("--pitch-amplitude-deg",type=float,default=0.)
    parser.add_argument("--pitch-period-s",type=float,default=2.)
    args = parser.parse_args()
    if not math.isfinite(args.pitch_amplitude_deg) or not 0 <= args.pitch_amplitude_deg <= 45 or not math.isfinite(args.pitch_period_s) or args.pitch_period_s <= 0:
        parser.error("Offline Pitch amplitude must be 0..45 deg and period positive")
    if not math.isfinite(args.hold_s) or args.hold_s <= 0:
        parser.error("hold-s must be finite and positive")
    if not all(math.isfinite(v) and v > 0 for v in (args.target_v,args.acceleration)) or args.target_v <= .2*args.acceleration:
        parser.error("Positive target/acceleration and trapezoidal ramp required")
    args.output = args.output.resolve()
    args.px4_root = args.px4_root.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    artifact = json.loads(args.models.read_text())
    variant = build(args.px4_root, artifact, args.output / "models")
    if args.route != "baseline":
        if args.schedule is None:
            raise ValueError("Identified route requires a frozen schedule")
        schedule = json.loads(args.schedule.read_text())
        if schedule["model_sha256"] != hashlib.sha256(args.models.read_bytes()).hexdigest():
            raise ValueError("Schedule/model hash mismatch")
    # Both baseline and identified runs load the same system/telemetry hooks.
    sdf_path = args.output / "models/fftgyro_x500/model.sdf"
    tree = ET.parse(sdf_path)
    plugin = ET.SubElement(tree.getroot().find("model"),"plugin",
        filename="gz-sim-python-system-loader-system",name="gz::sim::systems::PythonSystemLoader")
    for key,value in dict(module_name="gazebo_identified_system", coefficient_route=args.route,
        models=str(args.models.resolve()), output=str(args.output),
        schedule=str(args.schedule.resolve()) if args.schedule else "").items():
        ET.SubElement(plugin,key).text = value
    tree.write(sdf_path,encoding="utf-8",xml_declaration=True)
    variant["native_sdf_sha256"] = variant["generated_sdf_sha256"]
    variant["generated_sdf_sha256"] = hashlib.sha256(sdf_path.read_bytes()).hexdigest()
    (sdf_path.parent / "manifest.json").write_text(json.dumps(variant,indent=2)+"\n")
    simulation = args.px4_root / "Tools/simulation/gz"
    binary = args.px4_root / "build/px4_sitl_default/bin/px4"
    rootfs = args.output / "rootfs"
    rootfs.mkdir()
    (rootfs / "etc").symlink_to(binary.parent.parent / "etc", target_is_directory=True)
    params = common_parameters(artifact)
    startup = args.output / "startup.sh"
    startup.write_text("#!/bin/sh\n. etc/init.d-posix/rcS\n" +
        "\n".join(f"param set {k} {v:.12g}" for k,v in params.items()) + "\nparam show\n")
    env = dict(os.environ, PX4_SYS_AUTOSTART="4001", PX4_SIM_MODEL="fftgyro_x500",
        PX4_GZ_MODEL_POSE=initial_pose_enu(),
        PX4_GZ_STANDALONE="1", PX4_GZ_WORLD="default", PX4_GZ_NO_FOLLOW="1", HEADLESS="1",
        PX4_GZ_MODELS=str(args.output / "models"), PX4_GZ_WORLDS=str(simulation / "worlds"),
        GZ_SIM_RESOURCE_PATH=f"{args.output / 'models'}:{simulation / 'models'}",
        GZ_SIM_SYSTEM_PLUGIN_PATH=str(binary.parent.parent / "src/modules/simulation/gz_plugins"),
        GZ_SIM_SERVER_CONFIG_PATH=str(args.px4_root / "src/modules/simulation/gz_bridge/server.config"),
        GZ_PARTITION="fftgyro_" + hashlib.sha256(str(args.output).encode()).hexdigest()[:12],
        GZ_IP="127.0.0.1")
    env["GZ_SIM_SYSTEM_PLUGIN_PATH"] += ":" + str(Path(__file__).resolve().parent)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parent) + ":" + env.get("PYTHONPATH","")
    gazebo = px4 = mission = None
    with (args.output / "gazebo.log").open("w") as gz_log, (args.output / "px4.log").open("w") as px_log:
        try:
            gazebo = subprocess.Popen(["gz", "sim", "-r", "-s", str(simulation / "worlds/default.sdf")],
                env=env, stdout=gz_log, stderr=subprocess.STDOUT, start_new_session=True)
            px4 = subprocess.Popen([str(binary), "-d", "-s", str(startup)], cwd=rootfs,
                env=env, stdout=px_log, stderr=subprocess.STDOUT, start_new_session=True)
            with (args.output / "mission.log").open("w") as mission_log:
                mission = subprocess.Popen([sys.executable, str(Path(__file__).with_name("run_koil_px4_sitl.py")),
                    "--output", str(args.output / "mission"), "--warmup-s", "20",
                    "--hold-s", str(args.hold_s),
                    "--target-v",str(args.target_v),"--acceleration",str(args.acceleration),
                    "--pitch-amplitude-deg",str(args.pitch_amplitude_deg),"--pitch-period-s",str(args.pitch_period_s),
                    "--model-label", f"Native Gazebo X500 common engine: {args.route}"],
                    stdout=mission_log, stderr=subprocess.STDOUT, start_new_session=True)
                deadline = time.monotonic()+140
                while mission.poll() is None and time.monotonic()<deadline:
                    if (args.output / "plugin_failure.json").exists():
                        raise RuntimeError("Identified Gazebo plugin failed; see plugin_failure.json")
                    time.sleep(.25)
                if mission.poll() != 0:
                    raise RuntimeError("Native Gazebo mission failed; see logs")
            if px4.poll() is not None or gazebo.poll() is not None:
                raise RuntimeError("Simulator exited unexpectedly")
            if not (args.output / "plugin_ready.json").exists():
                raise RuntimeError("Gazebo coefficient plugin did not load")
            if (args.output / "plugin_failure.json").exists():
                raise RuntimeError("Gazebo coefficient plugin failed at mission end")
            summary = dict(status="COMPLETED", route=args.route, sdf_variant=variant,
                mission_profile=dict(target_vx_m_s=args.target_v,hold_s=args.hold_s,acceleration_m_s2=args.acceleration,jerk_ramp_s=.2,hover_s=2,settle_s=12,
                    initial_yaw_ned_deg=0.,target_yaw_ned_deg=0.,spawn_pose_enu=env["PX4_GZ_MODEL_POSE"]),
                plugin=json.loads((args.output / "plugin_ready.json").read_text()),
                schedule_sha256=hashlib.sha256(args.schedule.read_bytes()).hexdigest() if args.schedule else None,
                common_parameters=params, model_sha256=hashlib.sha256(args.models.read_bytes()).hexdigest(),
                binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                observation="Mission CSV is PX4 estimator; native truth retained in ULog",
                limitations=["Native common engine; compare only matching native-route runs",
                    "Same explicit controller gains; simulator airframe defaults may differ",
                    "Low-RPM run-up uses native loads; identified residual begins above half nominal hover speed",
                    "Identified route has frozen assumed hover anchor; absolute aero unvalidated",
                    "Native joint slowdown angular momentum not experimentally calibrated"])
            if args.pitch_amplitude_deg:
                summary["mission_profile"] = dict(kind="hover_pitch",amplitude_deg=args.pitch_amplitude_deg,
                    period_s=args.pitch_period_s,cycles=3,hover_s=2,recovery_s=4,
                    waveform="Hann-windowed cosine",horizontal_position_hold=False,
                    altitude_controller=dict(thrust_ff=.54,kp=.10,kd=.08,ki=.03,integral_limit=2,command_bounds=[.15,.85]),
                    initial_yaw_ned_deg=0.,target_yaw_ned_deg=0.,spawn_pose_enu=env["PX4_GZ_MODEL_POSE"])
            px_log.flush()
            summary["controller_parameters"] = {k:float(v) for k,v in re.findall(
                r"([A-Z][A-Z0-9_]+)\s+\[[0-9,]+\]\s*:\s*(\S+)", (args.output / "px4.log").read_text())
                if k.startswith(("MC_", "MPC_", "CA_", "PWM_MAIN_"))}
            summary["world_sdf_sha256"] = hashlib.sha256((simulation / "worlds/default.sdf").read_bytes()).hexdigest()
            summary["server_config_sha256"] = hashlib.sha256(Path(env["GZ_SIM_SERVER_CONFIG_PATH"]).read_bytes()).hexdigest()
            (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        finally:
            stop(mission)
            stop(px4)
            stop(gazebo)


if __name__ == "__main__":
    main()
