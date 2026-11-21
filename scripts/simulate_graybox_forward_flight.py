"""保存済み6軸モデルの自由飛行ストレス試験。PX4/SITL/実飛行検証ではない。

FRD機体、NED世界。0風、一定電圧、3秒ホバー→5秒で10m/s→保持。
初期ホバーのみ既知のmg/零トルクへアンカー。その後の定数更新/再fit/荷重clipなし。
両係数経路とも保存した4モーターmap・遅延・上下時定数でRPMを自由応答させる。
実RPM係数経路も未来の実測RPMは使わず、仮定したシミュレーションRPMを使う。
簡単な共通速度/高度/姿勢PIDを使用。支持トルクやrig慣性は入れない。
numpyだけで遠隔計算でき、--plot-onlyでは既存CSVから日本語図を作る。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

OUTPUTS = ("Fx", "Fy", "Fz", "Mx", "My", "Mz")
SPIN = np.array([1.0, 1.0, -1.0, -1.0])
ROLL = np.array([-1.0, 1.0, 1.0, -1.0])
PITCH = np.array([1.0, -1.0, 1.0, -1.0])
G = 9.80665
# Provisional simulation PID only; never written to the real Pixhawk.
# Stronger restoring stiffness than the local fitted Pitch slope near 10 m/s.
ATTITUDE_FREQUENCY = np.array([7.8, 7.8, 2.5])
ATTITUDE_DAMPING = 0.9
ATTITUDE_INTEGRAL_GAIN = 0.25
ATTITUDE_INTEGRAL_LIMIT = 2.0
VELOCITY_INTEGRAL_GAIN = 0.15
VELOCITY_INTEGRAL_LIMIT = 20.0  # Still permits up to 3 m/s² of steady compensation.


@dataclass(frozen=True)
class ControllerSettings:
    """Offline controller only. Defaults preserve the original comparison."""
    velocity_kp: float = 0.9
    velocity_ki: float = VELOCITY_INTEGRAL_GAIN
    position_kp: float = 0.8
    antiwindup: bool = False
    attitude_frequency: float = 7.8
    attitude_damping: float = ATTITUDE_DAMPING


def velocity_acceleration(
    error: np.ndarray, integral: np.ndarray, feedforward: np.ndarray, dt: float,
    limits: np.ndarray, settings: ControllerSettings,
) -> tuple[np.ndarray, np.ndarray]:
    """PI with optional conditional integration when an axis is saturated."""
    candidate = np.clip(integral + dt * error,
                        -VELOCITY_INTEGRAL_LIMIT, VELOCITY_INTEGRAL_LIMIT)
    raw = feedforward + settings.velocity_kp * error + settings.velocity_ki * candidate
    if settings.antiwindup:
        blocked = (np.abs(raw) > limits) & (raw * error > 0)
        candidate = np.where(blocked, integral, candidate)
        raw = feedforward + settings.velocity_kp * error + settings.velocity_ki * candidate
    return np.clip(raw, -limits, limits), candidate


def rotation(q: np.ndarray) -> np.ndarray:
    """Scalar-first unit quaternion: body -> world."""
    w, x, y, z = q / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def quaternion_derivative(q: np.ndarray, rate: np.ndarray) -> np.ndarray:
    return 0.5 * np.r_[-q[1:] @ rate, q[0] * rate + np.cross(q[1:], rate)]


def inflow(mu: np.ndarray, ct: float) -> float:
    """Same positive unique momentum root as graybox_pipeline.induced_inflow."""
    h2, z = float(mu[0] ** 2 + mu[1] ** 2), float(mu[2])

    def flux(nu: float) -> float:
        return 2 * nu * math.sqrt(h2 + (nu - z) ** 2)

    disc = z * z - 8 * h2
    if z > 0 and disc > 0:
        nmax, nmin = (3 * z - math.sqrt(disc)) / 4, (3 * z + math.sqrt(disc)) / 4
        if flux(nmin) - 1e-12 <= ct <= flux(nmax) + 1e-12:
            raise ValueError("Ambiguous quasi-steady inflow; no arbitrary root selected")
    lo, hi = 0.0, max(z, 0) + math.sqrt(ct / 2) + 1
    for _ in range(48):
        mid = (lo + hi) / 2
        if flux(mid) < ct:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2 - z


class Aircraft:
    def __init__(self, artifact: dict, route: str, *, anchored: bool = True):
        self.cfg, self.route = artifact["settings"], route
        self.models = artifact["models"].get(route, {})
        self.motors = artifact["motor_models"]
        self.mass = self.cfg["mass_kg"]
        self.inertia = np.array(self.cfg["inertia_kg_m2"])
        self.hover_w = math.sqrt(self.mass * G / (4 * self.cfg["kf"]))
        self.anchor = np.zeros(6)
        self.unanchored_hover = self.loads(
            np.zeros(3), np.zeros(3), np.full(4, self.hover_w), np.zeros(4)
        )
        if anchored:
            # Explicit assumed initial absolute load, NOT measured hover calibration.
            self.anchor = np.r_[0, 0, -self.mass * G, 0, 0, 0] - self.unanchored_hover

    def state(self, air: np.ndarray, omega: np.ndarray, rotor: np.ndarray) -> tuple[dict, float]:
        c = self.cfg
        bar2 = float(np.mean(rotor**2))
        denominator = math.sqrt(bar2) * c["rotor_radius_m"]
        mu = air / denominator
        rates = omega * c["arm_xy_m"] / denominator
        up, uq, ur = [(rotor**2) @ signs / bar2 for signs in (ROLL, PITCH, SPIN)]
        advance = float(np.linalg.norm(mu))
        beta = abs(math.atan2(mu[1], abs(mu[0])))
        state = dict(zip(("mu_x", "mu_y", "mu_z"), mu, strict=True))
        state.update(zip(("pbar", "qbar", "rbar"), rates, strict=True))
        state.update(
            up=up,
            uq=uq,
            ur=ur,
            yaw_rate=omega[2],
            ur_absur=ur * abs(ur),
            sun_low=float(advance <= 0.05),
            sun_beta_0_30=float(advance > 0.05 and beta < np.pi / 6),
            sun_beta_30_60=float(advance > 0.05 and np.pi / 6 <= beta < np.pi / 3),
            sun_beta_60_90=float(advance > 0.05 and beta >= np.pi / 3),
            mu_h2=mu[0] ** 2 + mu[1] ** 2,
        )
        state.update(
            {
                "abs_" + k: abs(state[k])
                for k in ("mu_x", "mu_y", "pbar", "qbar", "rbar", "up", "uq", "ur")
            }
        )
        if self.route != "physical":
            ct = c["kf"] / (c["rho"] * np.pi * c["rotor_radius_m"] ** 4)
            state["inflow"] = inflow(mu, ct)
        scale = c["rho"] * 4 * np.pi * c["rotor_radius_m"] ** 2 * denominator**2
        return state, scale

    def loads(
        self, air: np.ndarray, omega: np.ndarray, rotor: np.ndarray, rotor_dot: np.ndarray
    ) -> np.ndarray:
        c, w2 = self.cfg, rotor**2
        physical = np.array(
            [
                0.0,
                0.0,
                -c["kf"] * w2.sum(),
                c["kf"] * c["arm_xy_m"] * (w2 @ ROLL),
                c["kf"] * c["arm_xy_m"] * (w2 @ PITCH),
                c["kq"] * (w2 @ SPIN),
            ]
        )
        shift = (np.array(c["cg_flu_m"]) - c["propulsion_reference_cg_flu_m"]) * [1, -1, -1]
        physical[3:] -= np.cross(shift, physical[:3])
        h = np.array([0, 0, -c["rotor_inertia_kg_m2"] * (rotor @ SPIN)])
        dh = np.array([0, 0, -c["rotor_inertia_kg_m2"] * (rotor_dot @ SPIN)])
        physical[3:] += -dh - np.cross(omega, h)
        if self.route != "physical":
            state, force_scale = self.state(air, omega, rotor)
            for i, out in enumerate(OUTPUTS):
                model = self.models[out]
                correction = sum(
                    coef * math.prod(state[key] ** power for key, power in term)
                    for coef, term in zip(
                        model["coefficients"], model["selected_terms"], strict=True
                    )
                )
                physical[i] += force_scale * (1 if i < 3 else c["arm_xy_m"]) * correction
        return physical + self.anchor


def motor_map(u: np.ndarray, voltage: float, motors: list) -> np.ndarray:
    coeff = np.array([m["map_coefficients"] for m in motors])
    exponent = np.array([m.get("voltage_exponent", 1.0) for m in motors])
    u = np.maximum(u, 0.)
    rpm = (voltage / 16) ** exponent * (coeff[:, 0] * u + coeff[:, 1] * np.sqrt(u) + coeff[:, 2] * u * u)
    return rpm * 2 * np.pi / 60


def inverse_motor_map(goal: np.ndarray, voltage: float, motors: list) -> np.ndarray:
    lo, hi = np.full(4, 0.1), np.ones(4)
    for _ in range(32):
        mid = (lo + hi) / 2
        lower = motor_map(mid, voltage, motors) < goal
        lo, hi = np.where(lower, mid, lo), np.where(lower, hi, mid)
    return (lo + hi) / 2


def reference(time: float, target: float = 10.0) -> tuple[float, float]:
    phase = np.clip((time - 3) / 5, 0, 1)
    velocity = target * (phase - np.sin(2 * np.pi * phase) / (2 * np.pi))
    acceleration = target / 5 * (1 - np.cos(2 * np.pi * phase)) if 0 < phase < 1 else 0.0
    return float(velocity), float(acceleration)


def euler_deg(q: np.ndarray) -> np.ndarray:
    r = rotation(q)
    return np.rad2deg(
        [
            np.arctan2(r[2, 1], r[2, 2]),
            np.arcsin(np.clip(-r[2, 0], -1, 1)),
            np.arctan2(r[1, 0], r[0, 0]),
        ]
    )


def controller(
    x: np.ndarray, integral: np.ndarray, vehicle: Aircraft, time: float, dt: float, target: float,
    *, reference_function: Callable[[float, float], tuple[float, float]] = reference,
    horizontal_accel_limit: float = 2.5,
    controller_settings: ControllerSettings | None = None,
    position_reference: Callable[[float, float], np.ndarray | None] | None = None,
    height_reference: Callable[[float], tuple[float, float]] | None = None,
) -> tuple[np.ndarray, dict]:
    settings = controller_settings or ControllerSettings()
    speed, feedforward = reference_function(time, target)
    goal = position_reference(time, target) if position_reference else None
    height, climb = height_reference(time) if height_reference else (0., 0.)
    desired_velocity = np.array([speed, 0, -climb-1.2*(x[2]+height)])
    if goal is not None:
        desired_velocity[:2] = np.clip(settings.position_kp * (goal - x[:2]), -2, 2)
        feedforward = 0.0
    error = desired_velocity - x[3:6]
    accel, integral[:3] = velocity_acceleration(
        error, integral[:3], np.array([feedforward, 0, 0]), dt,
        np.array([horizontal_accel_limit, horizontal_accel_limit, 3.0]), settings,
    )
    desired_force = vehicle.mass * (accel - np.array([0, 0, G]))
    bz = -desired_force / np.linalg.norm(desired_force)
    by = np.cross(bz, [1, 0, 0])
    by /= np.linalg.norm(by)
    desired = np.column_stack([np.cross(by, bz), by, bz])
    actual = rotation(x[6:10])
    skew = 0.5 * (desired.T @ actual - actual.T @ desired)
    error_att = np.array([skew[2, 1], skew[0, 2], skew[1, 0]])
    # Common controller disturbance rejection, NOT a per-step load-model anchor.
    integral[3:] = np.clip(
        integral[3:] + dt * error_att, -ATTITUDE_INTEGRAL_LIMIT, ATTITUDE_INTEGRAL_LIMIT
    )
    rate = x[10:13]
    frequencies = np.array([settings.attitude_frequency, settings.attitude_frequency, 2.5])
    # Conservative generic yaw loop: avoid exciting the provisional motor/Ip lag.
    torque = -vehicle.inertia * (
        frequencies**2 * (error_att + ATTITUDE_INTEGRAL_GAIN * integral[3:])
        + 2 * settings.attitude_damping * frequencies * rate
    )
    torque += np.cross(rate, vehicle.inertia * rate)
    torque = np.clip(torque, [-0.6, -0.6, -0.1], [0.6, 0.6, 0.1])
    thrust = max(0, float(-desired_force @ actual[:, 2]))
    c = vehicle.cfg
    allocator = np.vstack(
        [
            np.full(4, c["kf"]),
            c["kf"] * c["arm_xy_m"] * ROLL,
            c["kf"] * c["arm_xy_m"] * PITCH,
            c["kq"] * SPIN,
        ]
    )
    squared = np.linalg.solve(allocator, np.r_[thrust, torque])
    return np.sqrt(np.maximum(squared, 0)), dict(
        reference_vx=speed, desired_thrust=thrust,
        position_hold=int(goal is not None),
        reference_x=float(goal[0]) if goal is not None else float("nan"),
    )


def rhs(
    x: np.ndarray, aircraft: Aircraft, rotor: np.ndarray, rotor_dot: np.ndarray,
    *, ground_contact: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    r = rotation(x[6:10])
    rate = x[10:13]
    load = aircraft.loads(r.T @ x[3:6], rate, rotor, rotor_dot)
    dv = r @ load[:3] / aircraft.mass + np.array([0, 0, G])
    dw = (load[3:] - np.cross(rate, aircraft.inertia * rate)) / aircraft.inertia
    if ground_contact and x[2] >= -1e-8 and x[5] >= 0 and dv[2] >= 0:
        # Ideal flat support: reaction cancels gravity/load and contact torques.
        # Release as soon as upward acceleration is possible; not a landing-gear model.
        return np.zeros(13), load
    return np.r_[x[3:6], dv, quaternion_derivative(x[6:10], rate), dw], load


def simulate(
    artifact: dict,
    route: str,
    duration: float,
    dt: float,
    voltage: float,
    target: float,
    aircraft: Aircraft | None = None,
    *,
    reference_function: Callable[[float, float], tuple[float, float]] = reference,
    horizontal_accel_limit: float = 2.5,
    controller_settings: ControllerSettings | None = None,
    position_reference: Callable[[float, float], np.ndarray | None] | None = None,
    height_reference: Callable[[float], tuple[float, float]] | None = None,
    ground_start: bool = False,
) -> tuple[list[dict], dict]:
    aircraft = aircraft or Aircraft(artifact, route)
    x = np.zeros(13)
    x[6] = 1
    rotor = np.zeros(4) if ground_start else np.full(4, aircraft.hover_w)
    integral = np.zeros(6)
    hover_command = inverse_motor_map(np.full(4, aircraft.hover_w), voltage, aircraft.motors)
    history, rows = [], []
    desired, control = controller(
        x, integral, aircraft, 0, 0, target,
        reference_function=reference_function, horizontal_accel_limit=horizontal_accel_limit,
        controller_settings=controller_settings, position_reference=position_reference,
        height_reference=height_reference,
    )
    command, status = hover_command.copy(), "COMPLETED"
    low = motor_map(np.full(4, 0.1), voltage, aircraft.motors)
    high = motor_map(np.ones(4), voltage, aircraft.motors)
    for step in range(round(duration / dt) + 1):
        time = step * dt
        if step % max(1, round(0.02 / dt)) == 0:
            desired, control = controller(
                x, integral, aircraft, time, 0.02, target,
                reference_function=reference_function,
                horizontal_accel_limit=horizontal_accel_limit,
                controller_settings=controller_settings, position_reference=position_reference,
                height_reference=height_reference,
            )
            command = inverse_motor_map(desired, voltage, aircraft.motors)
        history.append(command.copy())
        saturated = bool(np.any((desired < low) | (desired > high)))
        desired = np.clip(desired, low, high)
        delayed = np.array(
            [
                (0. if ground_start and step < round(m["delay_s"] / dt) else
                 history[max(0, step - round(m["delay_s"] / dt))][j])
                for j, m in enumerate(aircraft.motors)
            ]
        )
        goal = motor_map(delayed, voltage, aircraft.motors)
        tau = np.array(
            [
                m["tau_up_s"] if goal[j] >= rotor[j] else m["tau_down_s"]
                for j, m in enumerate(aircraft.motors)
            ]
        )
        next_rotor = goal + (rotor - goal) * np.exp(-dt / tau)
        mean_rotor, rotor_dot = (rotor + next_rotor) / 2, (next_rotor - rotor) / dt
        try:
            k1, load = rhs(x, aircraft, mean_rotor, rotor_dot, ground_contact=ground_start)
            k2, _ = rhs(x + dt * k1 / 2, aircraft, mean_rotor, rotor_dot, ground_contact=ground_start)
        except (ValueError, OverflowError) as error:
            status = str(error)
            break
        if step % max(1, round(0.02 / dt)) == 0:
            angles = euler_deg(x[6:10])
            row = dict(
                time_s=time,
                vx=x[3],
                vy=x[4],
                vz=x[5],
                height=-x[2],
                x=x[0],
                y=x[1],
                roll_deg=angles[0],
                pitch_deg=angles[1],
                yaw_deg=angles[2],
                p_deg_s=float(np.rad2deg(x[10])),
                q_deg_s=float(np.rad2deg(x[11])),
                r_deg_s=float(np.rad2deg(x[12])),
                thrust=-load[2],
                speed=float(np.linalg.norm(x[3:6])),
                saturated=int(saturated),
                voltage=voltage,
                **control,
            )
            if ground_start:
                hgoal, _ = height_reference(time) if height_reference else (0., 0.)
                row.update(reference_height=hgoal,ground_contact=int(x[2]>=-1e-8 and np.all(k1==0)),
                           ground_normal_force_N=max(0.,aircraft.mass*G+(rotation(x[6:10])@load[:3])[2])
                           if x[2]>=-1e-8 and np.all(k1==0) else 0.)
            row.update(zip(OUTPUTS, load, strict=True))
            row.update({f"rpm_M{j + 1}": mean_rotor[j] * 60 / (2 * np.pi) for j in range(4)})
            row.update({f"command_M{j + 1}": command[j] for j in range(4)})
            rows.append(row)
        x += dt * k2
        if ground_start and x[2] > 0:
            x[2], x[5] = 0., min(0., x[5])
        x[6:10] /= np.linalg.norm(x[6:10])
        rotor = next_rotor
        if not np.isfinite(x).all() or np.linalg.norm(x[3:6]) > 40 or abs(x[2]) > 5:
            status = "STOPPED: nonfinite, speed>40m/s, or height deviation>5m"
            break
        if np.max(np.abs(euler_deg(x[6:10])[:2])) > 70:
            status = "STOPPED: roll/pitch>70deg"
            break
    metrics = dict(
        status=status,
        end_time_s=rows[-1]["time_s"],
        final_vx=rows[-1]["vx"],
        final_height=rows[-1]["height"],
        final_pitch_deg=rows[-1]["pitch_deg"],
        max_abs_height=max(abs(r["height"]) for r in rows),
        max_vx=max(r["vx"] for r in rows),
        saturation_fraction=float(np.mean([r["saturated"] for r in rows])),
        assumed_hover_rpm=aircraft.hover_w * 60 / (2 * np.pi),
        assumed_hover_command=hover_command.tolist(),
        unanchored_hover_load=aircraft.unanchored_hover.tolist(),
        fixed_hover_anchor=aircraft.anchor.tolist(),
    )
    return rows, metrics


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    fonts = {f.name for f in font_manager.fontManager.ttflist}
    plt.rcParams["font.family"] = next(
        (f for f in ("Hiragino Sans", "Noto Sans CJK JP", "IPAexGothic") if f in fonts),
        "sans-serif",
    )
    plt.rcParams["axes.unicode_minus"] = False
    summary = json.loads((output / "summary.json").read_text())
    scheduled = bool(summary.get("speed_schedule"))
    names = {
        "physical": "推進物理モデルのみ",
        "measured": "実RPMで同定した荷重モデル",
        "command_route": "入力経路で同定した荷重モデル",
    }
    if scheduled:
        names.update(
            measured="速度別基準＋実RPM係数の局所変化",
            command_route="速度別基準＋入力経路係数の局所変化",
        )
    fig, axes = plt.subplots(3, 2, figsize=(13, 11), constrained_layout=True)
    fields = [
        ("vx", "前方速度 [m/s]"),
        ("height", "開始高度からの変化 [m]"),
        ("pitch_deg", "Pitch [度]"),
        ("thrust", "機体z方向の上向き力 [N]"),
        ("rpm_M1", "M1 RPM（例）"),
        ("yaw_deg", "Yaw [度]"),
    ]
    for route, label in names.items():
        with (output / f"{route}.csv").open() as stream:
            records = list(csv.DictReader(stream))
        t = np.array([float(r["time_s"]) for r in records])
        for ax, (key, title) in zip(axes.flat, fields, strict=True):
            values = [float(r[key]) for r in records]
            (line,) = ax.plot(t, values, label=label, lw=1.5)
            if summary["trajectories"][route]["status"] != "COMPLETED":
                ax.scatter(t[-1], values[-1], color=line.get_color(), s=20, zorder=3)
            ax.set(xlabel="時間 [s]", ylabel=title)
            ax.grid(alpha=0.25)
    t = np.linspace(0, summary["duration_s"], 301)
    axes[0, 0].plot(
        t, [reference(v, summary["target_vx"])[0] for v in t], "k--", label="目標速度"
    )
    axes[0, 1].axhline(0, color="k", ls="--", lw=1)
    axes[1, 1].axhline(
        json.loads((output / "summary.json").read_text())["weight_N"],
        color="k",
        ls="--",
        lw=1,
        label="重量",
    )
    axes[0, 0].legend(fontsize=9)
    fig.suptitle(
        (
            "速度別モデルを線形補間：ホバー → 前方10 m/s\n"
            if scheduled
            else "ホバー → 前方10 m/s：6自由度の試作シミュレーション\n"
        )
        + "初期ホバーのみ仮定で補正／共通モーター応答・試作PID／点は計算を途中停止した位置",
        fontsize=14,
    )
    fig.savefig(output / "forward_flight.png", dpi=160)
    plt.close(fig)
    with (output / "speed_load_scan.csv").open() as stream:
        scan = list(csv.DictReader(stream))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for route, label in names.items():
        records = [r for r in scan if r["route"] == route]
        v = [float(r["speed_m_s"]) for r in records]
        axes[0].plot(v, [float(r["upward_force_N"]) for r in records], label=label)
        axes[1].plot(v, [float(r["Fx"]) for r in records], label=label)
    axes[0].axhline(0, color="k", ls="--")
    axes[0].set(ylabel="機体z方向の上向き力 [N]", xlabel="前方対気速度 [m/s]")
    axes[1].set(ylabel="機体Fx [N]", xlabel="前方対気速度 [m/s]")
    for ax in axes:
        ax.grid(alpha=0.25)
    axes[1].legend(fontsize=8)
    fig.suptitle("同じホバーRPM・水平姿勢で速度だけ変更：速度間の移植診断（飛行軌道ではない）")
    fig.savefig(output / "speed_load_scan.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dt", type=float, default=0.004)
    parser.add_argument("--duration", type=float, default=30)
    parser.add_argument("--voltage", type=float, default=16)
    parser.add_argument("--target", type=float, default=10)
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--schedule", type=Path, help="速度ノードの基準＋局所動的変化を線形補間")
    args = parser.parse_args()
    if args.plot_only:
        plot(args.output)
        return
    if args.models is None or args.dt <= 0 or args.duration <= 0:
        parser.error("models and positive dt/duration required")
    artifact = json.loads(args.models.read_text())
    schedule = json.loads(args.schedule.read_text()) if args.schedule else None
    if (
        schedule
        and schedule["model_sha256"] != hashlib.sha256(args.models.read_bytes()).hexdigest()
    ):
        raise ValueError("Schedule and saved coefficient artifact differ")
    args.output.mkdir(parents=True, exist_ok=True)
    summary = dict(
        status="ASSUMED_HOVER_FREE_FLIGHT_STRESS_TEST_NOT_VALIDATION",
        model_sha256=hashlib.sha256(args.models.read_bytes()).hexdigest(),
        source=str(args.models),
        dt_s=args.dt,
        duration_s=args.duration,
        target_vx=args.target,
        controller_attitude_natural_frequency_rad_s=ATTITUDE_FREQUENCY.tolist(),
        controller_attitude_damping_ratio=ATTITUDE_DAMPING,
        controller_attitude_integral_gain_per_s=ATTITUDE_INTEGRAL_GAIN,
        controller_attitude_integral_limit_rad_s=ATTITUDE_INTEGRAL_LIMIT,
        controller_velocity_integral_limit_m=VELOCITY_INTEGRAL_LIMIT,
        controller_velocity_integral_gain_per_s2=VELOCITY_INTEGRAL_GAIN,
        voltage=args.voltage,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        wind_world=[0, 0, 0],
        weight_N=artifact["settings"]["mass_kg"] * G,
        trajectories={},
        fz_speed_common_mode={},
        speed_schedule=schedule,
        limitations=[
            "Initial absolute hover load assumed, not independently measured",
            "Dynamic coefficients frozen; scheduled static baselines fit T only; no load clipping",
            "Generic controller, not PX4 SITL or recorded experiment PID",
            "All cases use the same fitted motor maps/lag; no future measured RPM is used",
            "Roll/cross-axis translation and 0-to-10 speed changes not validated",
            "Instantaneous deployment omits offline 0.5s SG/acausal filtering",
            "Saved inertia and motor/rotor dynamics are conditional",
            "Physical baseline contains no air drag; not PX4 or Sun reduced model",
        ],
    )
    scan = []
    for route in ("physical", "measured", "command_route"):
        if schedule and route != "physical":
            from speed_scheduled_graybox import ScheduledAircraft

            aircraft = ScheduledAircraft(artifact, route, schedule)
        else:
            aircraft = Aircraft(artifact, route)
        rows, metrics = simulate(
            artifact, route, args.duration, args.dt, args.voltage, args.target, aircraft=aircraft
        )
        write_csv(args.output / f"{route}.csv", rows)
        summary["trajectories"][route] = metrics
        print(route, json.dumps(metrics), flush=True)
        if route != "physical":
            fz = artifact["models"][route]["Fz"]
            terms = {
                tuple((k, p) for k, p in t): c
                for t, c in zip(fz["selected_terms"], fz["coefficients"], strict=True)
            }
            common = (terms[(("mu_h2", 1),)] + terms[(("mu_z", 2),)]) / 2
            factor = artifact["settings"]["rho"] * 4 * np.pi * aircraft.cfg["rotor_radius_m"] ** 2
            summary["fz_speed_common_mode"][route] = dict(
                coefficient=common,
                dimensional_N_per_m2_s2=factor * common,
                contribution_at_10m_s_N=factor * common * 100,
                applied_as_cross_speed_term=not bool(schedule),
                identity="S_F*c*(mu_h2+mu_z^2) = 4*rho*pi*R^2*c*|air|^2",
                interpretation=(
                    "Nearly constant within fixed-speed runs; moving-reference subtraction "
                    "can conceal it. Not proof of identified absolute speed dependence."
                ),
            )
        for speed in np.linspace(0, 10, 101):
            load = aircraft.loads(
                np.array([speed, 0, 0]), np.zeros(3), np.full(4, aircraft.hover_w), np.zeros(4)
            )
            scan.append(dict(route=route, speed_m_s=speed, upward_force_N=-load[2], Fx=load[0]))
    write_csv(args.output / "speed_load_scan.csv", scan)
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
