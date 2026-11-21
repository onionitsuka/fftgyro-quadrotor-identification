"""Extract native Gazebo ground truth, not EKF estimates, on the mission clock."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    from pyulog import ULog
    logs = list((args.run / "rootfs/log").rglob("*.ulg"))
    if len(logs) != 1:
        raise ValueError("Expected exactly one native ULog")
    source = logs[0]
    ulog = ULog(str(source), message_name_filter_list=["vehicle_local_position_groundtruth"])
    data = ulog.get_dataset("vehicle_local_position_groundtruth").data
    clock = json.loads((args.run / "mission/mission_clock.json").read_text())
    time = data["timestamp_sample"] / 1e6 - clock["mission_boot_ms"] / 1000
    if time[0] > 0 or time[-1] < clock["duration_s"] - .1 or np.any(np.diff(time) <= 0) or np.max(np.diff(time)) > .1:
        raise ValueError("Insufficient/missing ground-truth clock coverage")
    select = (time >= 0) & (time <= clock["duration_s"])
    rows = {"time_s": time[select]}
    for key in ("x","y","z","vx","vy","vz"):
        rows[key] = data[key][select].astype(float)
        if key in ("x","y","z"):
            rows[key] -= np.interp(0,time,data[key])
    rows["height"] = -rows.pop("z")
    speed = np.sqrt(sum(rows[k]**2 for k in ("vx","vy","vz")))
    dt = np.diff(rows["time_s"])
    distance = float(np.sum(dt * (speed[1:]+speed[:-1])/2))
    output = args.run / "native_ground_truth.csv"
    if output.exists():
        raise FileExistsError(output)
    with output.open("w",newline="") as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows))
        writer.writeheader()
        writer.writerows({k:float(v[j]) for k,v in rows.items()} for j in range(len(dt)+1))
    (args.run / "native_ground_truth_summary.json").write_text(json.dumps(dict(
        observation="Gazebo ground truth published by PX4 gz_bridge, not EKF",
        ulog_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        max_vx_m_s=float(rows["vx"].max()), max_forward_displacement_m=float(rows["x"].max()),
        cumulative_3d_distance_m=distance, max_abs_height_change_m=float(np.abs(rows["height"]).max()),
        time_window_s=[float(rows["time_s"][0]),float(rows["time_s"][-1])],
        transformation="Recorded mission clock; translate true position at t=0; no speed smoothing or mean removal",
        limitations="One native baseline run; native motor mapping/sensors/physics differ from Python plant"
    ),indent=2)+"\n")


if __name__ == "__main__":
    main()
