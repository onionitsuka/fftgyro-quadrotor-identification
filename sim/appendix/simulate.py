#!/usr/bin/env python3
"""付録用の6自由度シミュレーション（保存モデルのみ・オフライン）。

読む順序: 1 荷重 → 2 ホバートリム → 3 指令とRPM → 4 運動方程式
          → 5 共通制御器 → 6 時間積分 → 7 出力。
座標: 世界NED、機体FRD、quaternion=[w,x,y,z]は機体から世界への回転。
係数を再同定せず、飛行ログ・実測RPM・実機通信を使わない。
荷重の候補項評価は、既存解析から切り出した純粋関数を使う。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from aerodynamics import instantaneous_data, correction

G = 9.80665
AXES = ("Fx", "Fy", "Fz", "Mx", "My", "Mz")
ROLL = np.array([-1., 1., 1., -1.])
PITCH = np.array([1., -1., 1., -1.])
SPIN = np.array([1., 1., -1., -1.])


def rotation(q: np.ndarray) -> np.ndarray:
    """機体ベクトルを世界へ移す回転行列。"""
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def attitude(roll: float, pitch: float) -> np.ndarray:
    """Yaw=0の初期quaternion。"""
    cr, sr = np.cos(roll/2), np.sin(roll/2)
    cp, sp = np.cos(pitch/2), np.sin(pitch/2)
    return np.array([cr*cp, sr*cp, cr*sp, -sr*sp])


# 1. 推進系の既存モデルと同定された6軸荷重
class Loads:
    def __init__(self, config: dict, identified: bool, route: str | None = None):
        self.config, self.identified = config, identified
        self.route = route or config["default_route"]
        self.cfg = SimpleNamespace(**config["settings"])
        self.mass = self.cfg.mass_kg
        self.inertia = np.asarray(self.cfg.inertia_kg_m2)

    def propulsion(self, rate, rotor, rotor_dot):
        """回転数二乗、重心への移送、ローター反作用とジャイロ。"""
        c, squared = self.cfg, rotor**2
        force = np.array([0., 0., -c.kf*squared.sum()])
        moment = np.array([c.kf*c.arm_xy_m*(squared@ROLL),
                           c.kf*c.arm_xy_m*(squared@PITCH), c.kq*(squared@SPIN)])
        shift = (np.asarray(c.cg_flu_m)-c.propulsion_reference_cg_flu_m)*[1, -1, -1]
        h = np.array([0., 0., -c.rotor_inertia_kg_m2*(rotor@SPIN)])
        h_dot = np.array([0., 0., -c.rotor_inertia_kg_m2*(rotor_dot@SPIN)])
        moment -= np.cross(shift, force) + h_dot + np.cross(rate, h)
        return np.r_[force, moment]

    def residual(self, air, rate, rotor, rotor_dot):
        """保存係数による補正。風速によらず1つの絶対値関数で、初期基準や定数補正を持たない。"""
        limit = self.config["airspeed_limit_m_s"]
        if np.linalg.norm(air) > limit+1e-10:
            raise ValueError(f"airspeed exceeds {limit} m/s; no extrapolation")
        physical = self.propulsion(rate, rotor, rotor_dot)
        data = instantaneous_data(air, rate, rotor, rotor_dot, self.cfg, physical)
        models = self.config["models"][self.route]
        result = np.array([correction(data, self.cfg, out, models[out]) for out in AXES])
        if self.config.get("roll_transfer", False):
            # Roll試験がないため，機体の90°対称を仮定してPitch補正をRollへ流用する（Mxの補正のみ）．
            # (+90°)：(u,v,w)->(-v,u,w)，(p,q,r)->(-q,p,r)，ロータ[3,4,2,1]；(-90°)は逆向きで符号反転．
            plus = self._pitch_residual(air[[1, 0, 2]]*[-1, 1, 1], rate[[1, 0, 2]]*[-1, 1, 1],
                                        rotor[[2, 3, 1, 0]], rotor_dot[[2, 3, 1, 0]])
            minus = self._pitch_residual(air[[1, 0, 2]]*[1, -1, 1], rate[[1, 0, 2]]*[1, -1, 1],
                                         rotor[[3, 2, 0, 1]], rotor_dot[[3, 2, 0, 1]])
            result[AXES.index("Mx")] = (plus - minus)/2
        return result

    def _pitch_residual(self, air, rate, rotor, rotor_dot):
        physical = self.propulsion(rate, rotor, rotor_dot)
        data = instantaneous_data(air, rate, rotor, rotor_dot, self.cfg, physical)
        return correction(data, self.cfg, "My", self.config["models"][self.route]["My"])

    def __call__(self, air, rate, rotor, rotor_dot):
        physical = self.propulsion(rate, rotor, rotor_dot)
        if not self.identified:
            return physical
        return physical + self.residual(air, rate, rotor, rotor_dot)


# 2. 各モデル自身の釣合いから初期ホバー状態を求める
def hover_trim(vehicle: Loads, voltage: float):
    """未知数はRoll/Pitchと4基RPM。荷重係数を変更せずF=−mg、M=0を解く。"""
    c = vehicle.cfg
    nominal = np.sqrt(vehicle.mass*G/(4*c.kf))
    low, high = motor_speed(np.full(4, .1), voltage, vehicle.config), motor_speed(
        np.ones(4), voltage, vehicle.config)

    def state(z):
        q, rotor = attitude(*z[:2]), nominal*np.exp(z[2:])
        return q, rotor

    def residual(z):
        q, rotor = state(z)
        wrench = vehicle(np.zeros(3), np.zeros(3), rotor, np.zeros(4))
        force = rotation(q)@wrench[:3] + np.array([0., 0., vehicle.mass*G])
        return np.r_[force/(vehicle.mass*G), wrench[3:]/(vehicle.mass*G*c.arm_xy_m)]

    bounds = (np.r_[[-.35, -.35], np.log(low/nominal)],
              np.r_[[.35, .35], np.log(high/nominal)])
    solved = least_squares(residual, np.zeros(6), bounds=bounds,
                           xtol=1e-11, ftol=1e-11, gtol=1e-11, max_nfev=100)
    error = float(np.max(np.abs(residual(solved.x))))
    if not solved.success or error > 1e-7:
        raise ValueError(f"no supported hover trim: dimensionless residual={error:.3g}")
    q, rotor = state(solved.x)
    state0 = np.r_[np.zeros(6), q, np.zeros(3)]
    c = vehicle.cfg
    nominal_control = np.r_[c.kf*np.sum(rotor**2),
                            c.kf*c.arm_xy_m*(rotor**2@ROLL),
                            c.kf*c.arm_xy_m*(rotor**2@PITCH), c.kq*(rotor**2@SPIN)]
    return state0, rotor, nominal_control, error


# 3. 指令＋電圧 → 定常RPM → 一次遅れ（旧設定の遅延も読込可能）
def motor_speed(command, voltage, config):
    motors = config["motor_models"]
    u = np.maximum(command, 0.)
    if all(m.get("kind") == "kaufmann2023" for m in motors):
        coef = np.array([m["coefficients"] for m in motors])
        v = voltage / 16
        rpm = (coef[:, 0] + coef[:, 1]*v + coef[:, 2]*np.sqrt(u)
               + coef[:, 3]*u + coef[:, 4]*v*np.sqrt(u))
    else:  # Read archived configurations for reproducible comparisons.
        coef = np.array([m["map_coefficients"] for m in motors])
        exponent = np.array([m.get("voltage_exponent", 1.) for m in motors])
        rpm = (voltage/16)**exponent*(coef[:, 0]*u + coef[:, 1]*np.sqrt(u) + coef[:, 2]*u*u)
    return rpm*2*np.pi/60


def motor_command(goal, voltage, config):
    lo, hi = np.full(4, .1), np.ones(4)
    for _ in range(32):
        mid = (lo+hi)/2
        lower = motor_speed(mid, voltage, config) < goal
        lo, hi = np.where(lower, mid, lo), np.where(lower, hi, mid)
    return (lo+hi)/2


def delayed_command(time, history, dt, motors):
    """計算刻みごとの指令を遅延時刻へ線形補間。開始前はトリム指令。"""
    stamps = (time-np.array([m["delay_s"] for m in motors]))/dt
    index = np.clip(stamps, 0., len(history)-1.)
    lo, hi = np.floor(index).astype(int), np.ceil(index).astype(int)
    fraction = index-lo
    return np.array([(1-fraction[j])*history[lo[j]][j]+fraction[j]*history[hi[j]][j]
                     for j in range(4)])


# 4. 重心まわりの6自由度剛体運動（支持作用なし）
def derivative(x, vehicle, rotor, rotor_dot, wind):
    r, rate = rotation(x[6:10]), x[10:13]
    air = r.T@(x[3:6]-wind)
    wrench = vehicle(air, rate, rotor, rotor_dot)
    acceleration = r@wrench[:3]/vehicle.mass + np.array([0., 0., G])
    angular_acceleration = (wrench[3:]-np.cross(rate, vehicle.inertia*rate))/vehicle.inertia
    q = x[6:10]
    dq = .5*np.r_[-q[1:]@rate, q[0]*rate+np.cross(q[1:], rate)]
    return np.r_[x[3:6], acceleration, dq, angular_acceleration], wrench


# 5. 両モデルで同じ速度・高度・姿勢制御器を使う（PX4ではない）
def reference(time, scenario):
    t, v, ramp = time-scenario["hover_s"], scenario["target_m_s"], scenario["ramp_s"]
    if t <= 0:
        return 0., 0.
    if t < ramp:
        return v*(1-np.cos(np.pi*t/ramp))/2, v*np.pi*np.sin(np.pi*t/ramp)/(2*ramp)
    t -= ramp
    if t <= scenario["hold_s"]:
        return v, 0.
    t -= scenario["hold_s"]
    if t < ramp:
        return v*(1+np.cos(np.pi*t/ramp))/2, -v*np.pi*np.sin(np.pi*t/ramp)/(2*ramp)
    return 0., 0.


def control(x, integral, vehicle, time, dt, scenario, trim_rotation, trim_control):
    c, gains = vehicle.cfg, vehicle.config["controller"]
    speed, feedforward = reference(time, scenario)
    target_velocity = np.array([speed, 0., -gains["height_kp"]*(x[2]+scenario["initial_height_m"])])
    error = target_velocity-x[3:6]
    limits = np.array([gains["horizontal_acceleration_limit_m_s2"]]*2 +
                      [gains["vertical_acceleration_limit_m_s2"]])
    candidate = np.clip(integral[:3]+dt*error, -20, 20)
    raw = np.array([feedforward, 0, 0])+gains["velocity_kp"]*error+gains["velocity_ki"]*candidate
    blocked = (np.abs(raw)>limits)&(raw*error>0)
    integral[:3] = np.where(blocked, integral[:3], candidate)
    acceleration = np.clip(np.array([feedforward, 0, 0])+gains["velocity_kp"]*error+
                           gains["velocity_ki"]*integral[:3], -limits, limits)
    force = vehicle.mass*(acceleration-np.array([0., 0., G]))
    bz = -force/np.linalg.norm(force)
    by = np.cross(bz, [1., 0., 0.]); by /= np.linalg.norm(by)
    desired = np.column_stack([np.cross(by, bz), by, bz])@trim_rotation
    actual = rotation(x[6:10])
    skew = .5*(desired.T@actual-actual.T@desired)
    error_attitude = np.array([skew[2, 1], skew[0, 2], skew[1, 0]])
    integral[3:] = np.clip(integral[3:]+dt*error_attitude, -2, 2)
    frequency = np.asarray(gains["attitude_frequency_rad_s"])
    torque = trim_control[1:] - vehicle.inertia*(frequency**2*(error_attitude+
        gains["attitude_ki"]*integral[3:])+2*gains["attitude_damping"]*frequency*x[10:13])
    torque += np.cross(x[10:13], vehicle.inertia*x[10:13])
    torque = np.clip(torque, -np.array(gains["torque_limit_Nm"]), gains["torque_limit_Nm"])
    thrust = trim_control[0]*np.linalg.norm(force)/(vehicle.mass*G)*max(0., desired[:, 2]@actual[:, 2])
    allocation = np.vstack([np.full(4, c.kf), c.kf*c.arm_xy_m*ROLL,
                            c.kf*c.arm_xy_m*PITCH, c.kq*SPIN])
    squared = np.linalg.solve(allocation, np.r_[thrust, torque])
    low, high = motor_speed(np.full(4, .1), scenario["voltage_V"], vehicle.config), motor_speed(
        np.ones(4), scenario["voltage_V"], vehicle.config)
    saturated = bool(np.any((squared < low**2)|(squared > high**2)))
    goal = np.sqrt(np.clip(squared, low**2, high**2))
    return motor_command(goal, scenario["voltage_V"], vehicle.config), saturated


# 6. 一次遅れを解析的に更新し、剛体運動は陽的中点法で積分
def simulate(config: dict, identified: bool, route: str | None = None):
    scenario = config["scenario"]
    dt, voltage = scenario["dt_s"], scenario["voltage_V"]
    duration = scenario["hover_s"]+2*scenario["ramp_s"]+scenario["hold_s"]+scenario["after_s"]
    vehicle = Loads(config, identified, route)
    x, rotor, trim_control, trim_error = hover_trim(vehicle, voltage)
    x[2] = -scenario["initial_height_m"]
    trim_rotation, initial_rotor = rotation(x[6:10]), rotor.copy()
    initial_command = motor_command(rotor, voltage, config)
    integral, history, rows = np.zeros(6), [], []
    wind = np.asarray(scenario["wind_ned_m_s"])
    status, failure_time = "COMPLETED", None
    for step in range(round(duration/dt)+1):
        time = step*dt
        command, saturated = control(x, integral, vehicle, time, dt, scenario, trim_rotation, trim_control)
        history.append(command)
        # Pre-t=0 command is the equilibrium command, including nonzero delay.
        if step == 0:
            history[0] = initial_command
        delayed = delayed_command(time, history, dt, config["motor_models"])
        goal = motor_speed(delayed, voltage, config)
        tau = np.array([m["tau_up_s"] if goal[j]>=rotor[j] else m["tau_down_s"]
                        for j, m in enumerate(config["motor_models"])])
        half_rotor = goal+(rotor-goal)*np.exp(-dt/(2*tau))
        try:
            dx, wrench = derivative(x, vehicle, rotor, (goal-rotor)/tau, wind)
            midpoint = x+dt*dx/2
            dx_mid, _ = derivative(midpoint, vehicle, half_rotor, (goal-half_rotor)/tau, wind)
            rows.append(np.r_[time, reference(time, scenario)[0], x, rotor*60/(2*np.pi),
                              command, wrench, int(saturated)])
            if step == round(duration/dt):
                break
            x += dt*dx_mid
            x[6:10] /= np.linalg.norm(x[6:10])
            rotor = goal+(rotor-goal)*np.exp(-dt/tau)
            if not np.isfinite(x).all() or np.linalg.norm(x[3:6]) > 20 or abs(x[2]) > 20:
                raise ValueError("state outside simulation limits")
            if np.rad2deg(np.arccos(np.clip(rotation(x[6:10])[2, 2], -1, 1))) > 60:
                raise ValueError("tilt exceeds 60 degrees")
        except (ValueError, OverflowError, FloatingPointError) as error:
            status, failure_time = str(error), time
            break
    trace = np.asarray(rows)
    if not len(trace):
        raise ValueError(f"no valid simulation samples: {status}")
    return trace, dict(status=status, failure_time_s=failure_time, end_time_s=float(trace[-1, 0]),
        initial_rotor_rpm=(initial_rotor*60/(2*np.pi)).tolist(), initial_command=initial_command.tolist(),
        initial_roll_pitch_deg=np.rad2deg([np.arctan2(trim_rotation[2, 1], trim_rotation[2, 2]),
                                          np.arcsin(-trim_rotation[2, 0])]).tolist(),
        trim_dimensionless_residual=trim_error, constant_force_bias_added=False,
        load_route=vehicle.route if identified else "none (existing model)",
        max_speed_m_s=float(np.max(np.linalg.norm(trace[:, 5:8], axis=1))),
        saturation_fraction=float(trace[:, -1].mean()))


# 7. CSV・計算条件・図を保存（実測との精度検証は別に行う）
HEADER = ["time_s", "reference_vx_m_s", "x_m", "y_m", "z_m", "vx_m_s", "vy_m_s", "vz_m_s",
          "qw", "qx", "qy", "qz", "p_rad_s", "q_rad_s", "r_rad_s",
          *[f"rpm_M{i}" for i in range(1, 5)], *[f"command_M{i}" for i in range(1, 5)],
          *AXES, "saturated"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plot(output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    fonts = {f.name for f in font_manager.fontManager.ttflist}
    font = next((n for n in ("Hiragino Sans", "Noto Sans CJK JP", "IPAexGothic") if n in fonts), "sans-serif")
    with plt.rc_context({"font.size": 14, "font.family": font, "axes.unicode_minus": False, "pdf.fonttype": 42}):
        fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
        for label, color, style in [("existing", "#D55E00", "--"), ("identified", "#0072B2", "-")]:
            data = np.genfromtxt(output/f"{label}.csv", delimiter=",", names=True)
            t = data["time_s"]
            name = "既存モデル" if label == "existing" else "同定モデル"
            axes[0, 0].plot(t, data["vx_m_s"], color=color, ls=style, lw=2, label=name)
            axes[0, 1].plot(t, -data["z_m"], color=color, ls=style, lw=2)
            pitch = np.rad2deg([np.arcsin(np.clip(-rotation(np.array(q))[2, 0], -1, 1))
                               for q in zip(data["qw"], data["qx"], data["qy"], data["qz"])])
            axes[1, 0].plot(t, pitch, color=color, ls=style, lw=2)
            axes[1, 1].plot(data["x_m"], -data["z_m"], color=color, ls=style, lw=2)
            if label == "existing":
                axes[0, 0].plot(t, data["reference_vx_m_s"], color="0.4", ls=":", lw=1.5, label="目標速度")
        for ax, ylabel in zip(axes.flat, ["前方速度 [m/s]", "高度 [m]", "Pitch [度]", "高度 [m]"]):
            ax.set(xlabel="時間 [s]", ylabel=ylabel); ax.grid(alpha=.2)
        axes[1, 1].set_xlabel("前方移動距離 [m]")
        axes[0, 0].legend()
        summary = json.loads((output/"summary.json").read_text())
        if any(m["status"] != "COMPLETED" for m in summary["cases"].values()):
            fig.suptitle("途中終了した計算は線の終端まで（理由はsummary.json）")
        fig.savefig(output/"comparison.png", dpi=180, facecolor="white")
        fig.savefig(output/"comparison.pdf", facecolor="white")
        plt.close(fig)
    (output/"figure_manifest.json").write_text(json.dumps(dict(
        source_csv_sha256={n: sha(output/f"{n}.csv") for n in ("existing", "identified")},
        transformations="none; quaternion converted to Pitch; NED z sign reversed for height",
        uncertainty="deterministic scenario; no statistical uncertainty estimated",
        missing="stopped runs end at the last computed sample; never extended",
        size_inches=[11, 7], png_dpi=180,
        description="既存モデルと同定モデルの速度、高度、Pitch、前方距離の計算例。実測との比較ではない。"
    ), ensure_ascii=False, indent=2)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("model.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", type=float, help="前方目標速度[m/s]。既定3。")
    parser.add_argument("--dt", type=float, help="積分刻み[s]。既定0.005。")
    parser.add_argument("--route", choices=("command_route", "measured"),
                        help="同定荷重の学習経路。既定は指令・電圧から推算したRPMで学習したcommand_route。")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    if args.plot_only:
        plot(args.output); return
    config = json.loads(args.config.read_text())
    if args.route is not None: config["default_route"] = args.route
    if args.target is not None: config["scenario"]["target_m_s"] = args.target
    if args.dt is not None: config["scenario"]["dt_s"] = args.dt
    scenario = config["scenario"]
    if not 0 <= scenario["target_m_s"] <= 5 or not 0 < scenario["dt_s"] <= .01:
        parser.error("Use target 0..5 m/s and dt 0..0.01 s")
    if np.any(np.asarray(scenario["wind_ned_m_s"]) != 0):
        parser.error("This appendix example initializes at windless hover")
    for relative, expected in config["provenance"]["evaluation_source_sha256"].items():
        if sha(HERE/relative) != expected: parser.error(f"Load evaluator changed: {relative}")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/"config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2)+"\n")
    summary = dict(type="DETERMINISTIC_OFFLINE_SIMULATION", validation=False,
        rpm_source=config["simulation_rpm_source"], load_route=config["default_route"], code_sha256=sha(__file__),
        runtime=dict(python=sys.version, numpy=version("numpy"), scipy=version("scipy")), cases={})
    for identified, label in [(False, "existing"), (True, "identified")]:
        trace, metric = simulate(config, identified)
        with (args.output/f"{label}.csv").open("w", newline="") as stream:
            writer = csv.writer(stream); writer.writerow(HEADER); writer.writerows(trace)
        summary["cases"][label] = metric
        (args.output/"summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n")
        print(label, metric["status"], f"end={metric['end_time_s']:.3f}s", flush=True)
    raise SystemExit(0 if all(m["status"] == "COMPLETED" for m in summary["cases"].values()) else 2)


if __name__ == "__main__":
    main()
