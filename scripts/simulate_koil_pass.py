"""KOIL短距離通過のオフライン比較。実機指令・PX4 SITLではない。

既存6自由度/モーター/PIDと速度補間モデルを再利用し、再fitしない。
5 m/sの指令保持と、実速度が5 m/s付近にいる時間を区別する。
比較基準は空力抵抗を持たない既存推進物理モデルであり、PX4標準ではない。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from simulate_graybox_forward_flight import Aircraft, ControllerSettings, simulate, write_csv
from speed_scheduled_graybox import ScheduledAircraft


@dataclass(frozen=True)
class PassProfile:
    acceleration: float = 4.0  # peak reference acceleration [m/s²]
    hold_s: float = 1.5
    jerk_ramp_s: float = 0.2
    hover_s: float = 2.0

    def ramp_duration(self, target: float) -> float:
        if self.acceleration <= 0 or target <= self.acceleration * self.jerk_ramp_s:
            raise ValueError("Positive acceleration and trapezoidal ramp required")
        return target / self.acceleration + self.jerk_ramp_s

    def ramp(self, time: float, target: float) -> tuple[float, float]:
        """Trapezoidal acceleration: finite jerk, continuous speed and acceleration."""
        a, j = self.acceleration, self.jerk_ramp_s
        duration = self.ramp_duration(target)
        if time <= 0:
            return 0.0, 0.0
        if time >= duration:
            return target, 0.0
        if time < j:
            return a * time**2 / (2 * j), a * time / j
        if time <= duration - j:
            return a * (time - j / 2), a
        remaining = duration - time
        return target - a * remaining**2 / (2 * j), a * remaining / j

    def reference(self, time: float, target: float) -> tuple[float, float]:
        t = time - self.hover_s
        duration = self.ramp_duration(target)
        if t <= duration:
            return self.ramp(t, target)
        if t < duration + self.hold_s:
            return target, 0.0
        v, a = self.ramp(t - duration - self.hold_s, target)
        return target - v, -a

    def position_reference(self, time: float, target: float) -> np.ndarray | None:
        """After braking, hold the integrated reference endpoint, not measured position."""
        end = self.hover_s + 2 * self.ramp_duration(target) + self.hold_s
        if time < end:
            return None
        return np.array([target * (self.ramp_duration(target) + self.hold_s), 0.0])


def diagnostics(rows: list[dict], profile: PassProfile, target: float) -> dict:
    t = np.array([r["time_s"] for r in rows])
    values = lambda key: np.array([r[key] for r in rows])
    hold_start = profile.hover_s + profile.ramp_duration(target)
    brake_start = hold_start + profile.hold_s
    brake_end = brake_start + profile.ramp_duration(target)
    in_hold = (t >= hold_start) & (t < brake_start)
    close = np.abs(values("vx") - target) <= 0.25
    sample_dt = float(np.median(np.diff(t)))
    runs = np.diff(np.r_[False, close, False].astype(int))
    lengths = np.flatnonzero(runs == -1) - np.flatnonzero(runs == 1)
    hold_close = close & in_hold
    hold_runs = np.diff(np.r_[False, hold_close, False].astype(int))
    hold_lengths = np.flatnonzero(hold_runs == -1) - np.flatnonzero(hold_runs == 1)
    slow = (np.linalg.norm(np.array([values(k) for k in ("vx", "vy", "vz")]), axis=0) < 0.2)
    stop_time = None
    count = max(1, round(1 / sample_dt))
    for start in np.flatnonzero((t >= brake_end) & slow):
        if start + count <= len(t) and slow[start:start + count].all():
            stop_time = float(t[start])
            break
    x_span = float(np.ptp(values("x")))
    return dict(
        commanded_hold_start_s=hold_start,
        commanded_brake_start_s=brake_start,
        commanded_brake_end_s=brake_end,
        ideal_reference_distance_m=target * (profile.ramp_duration(target) + profile.hold_s),
        commanded_hold_actual_vx_range_m_s=[float(values("vx")[in_hold].min()),
                                            float(values("vx")[in_hold].max())] if in_hold.any() else None,
        actual_time_within_5_percent_s=float(close.sum() * sample_dt),
        longest_contiguous_time_within_5_percent_s=float(lengths.max() * sample_dt) if len(lengths) else 0,
        hold_time_within_5_percent_s=float(hold_close.sum() * sample_dt),
        longest_hold_time_within_5_percent_s=float(hold_lengths.max() * sample_dt) if len(hold_lengths) else 0,
        near_target_tolerance_m_s=0.25,
        stop_speed_threshold_m_s=0.2,
        stop_persistence_s=1.0,
        stop_time_s=stop_time,
        x_range_m=[float(values("x").min()), float(values("x").max())],
        x_span_m=x_span,
        required_length_with_assumed_3m_each_end_m=x_span + 6,
        geometric_span_within_17m=bool(x_span <= 17),
        max_abs_y_m=float(np.abs(values("y")).max()),
        max_abs_pitch_deg=float(np.abs(values("pitch_deg")).max()),
        max_abs_q_deg_s=float(np.abs(values("q_deg_s")).max()),
        speed_tracking_rmse_m_s=float(np.sqrt(np.mean((values("vx") - values("reference_vx"))**2))),
        final_speed_m_s=float(np.linalg.norm([rows[-1][k] for k in ("vx", "vy", "vz")])),
        final_endpoint_error_m=float(np.linalg.norm(
            np.array([rows[-1]["x"], rows[-1]["y"]]) -
            np.array([target * (profile.ramp_duration(target) + profile.hold_s), 0.0]))),
    )


def plot(output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    fonts = {f.name for f in font_manager.fontManager.ttflist}
    plt.rcParams.update({"font.family": next((f for f in ("Hiragino Sans", "Noto Sans CJK JP", "IPAexGothic") if f in fonts), "sans-serif"),
                         "font.size": 13, "axes.unicode_minus": False})
    summary = json.loads((output / "summary.json").read_text())
    for case in summary["cases"]:
        fig, axes = plt.subplots(3, 2, figsize=(14, 12), layout="constrained")
        fields = [("vx", "前方速度 [m/s]"), ("x", "開始位置からの前方移動 [m]"),
                  ("pitch_deg", "Pitch [度]"), ("q_deg_s", "Pitch角速度 [度/s]"),
                  ("height", "開始高度からの変化 [m]"), ("y", "横方向移動 [m]")]
        styles = [("physical", "既存推進物理モデル（抵抗なし）", "#333333", "--"),
                  ("measured", "同定モデル：実RPM係数経路", "#0072B2", "-"),
                  ("command_route", "同定モデル：入力係数経路", "#D55E00", "-.")]
        for route, label, color, style in styles:
            if route not in case["routes"]:
                continue
            with (output / f"a{case['acceleration_m_s2']:g}_{route}.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            t = [float(r["time_s"]) for r in rows]
            for ax, (key, title) in zip(axes.flat, fields, strict=True):
                ax.plot(t, [float(r[key]) for r in rows], color=color, ls=style, label=label, lw=1.8)
                ax.set(xlabel="時間 [s]", ylabel=title)
                ax.grid(alpha=0.2)
            metric = case["routes"][route]
            for ax in axes.flat:
                ax.axvspan(metric["commanded_hold_start_s"], metric["commanded_brake_start_s"], color="gray", alpha=0.07)
        axes[0, 0].plot(t, [float(r["reference_vx"]) for r in rows], "k:", lw=1.5, label="目標速度（灰帯：1.5秒保持）")
        axes[0, 0].legend(fontsize=10)
        axes[0, 1].axhline(17, color="#666666", ls=":", label="仮の使用長17 m")
        axes[0, 1].legend(fontsize=10)
        mode = "速度PI調整＋制動後位置保持" if summary.get("position_hold") else "共通試作PID"
        fig.suptitle(f"5 m/sを1.5秒指令：加減速ピーク{case['acceleration_m_s2']:g} m/s²\n"
                     f"速度別線形補間・{mode}／無風・16 V／実飛行・PX4標準SITLの検証ではない", fontsize=16)
        fig.savefig(output / f"comparison_a{case['acceleration_m_s2']:g}.png", dpi=150, facecolor="white")
        plt.close(fig)


def plot_control_comparison(output: Path, baseline: Path) -> None:
    """Same axes/time horizon; preserve both the successful and adverse responses."""
    import matplotlib.pyplot as plt
    summary = json.loads((output / "summary.json").read_text())
    case = summary["cases"][0]
    acceleration = case["acceleration_m_s2"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), layout="constrained")
    fields = [("vx", "前方速度 [m/s]"), ("x", "開始位置からの前方移動 [m]"),
              ("pitch_deg", "Pitch [度]"), ("height", "開始高度からの変化 [m]")]
    sources = [(baseline, "measured", "調整前：実RPM係数経路", "#666666", "--"),
               (output, "measured", "調整後：実RPM係数経路", "#0072B2", "-"),
               (output, "command_route", "調整後：入力係数経路", "#D55E00", "-.")]
    for folder, route, label, color, style in sources:
        with (folder / f"a{acceleration:g}_{route}.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        for ax, (key, unit) in zip(axes.flat, fields, strict=True):
            ax.plot([float(r["time_s"]) for r in rows], [float(r[key]) for r in rows],
                    label=label, color=color, ls=style, lw=2)
            ax.set(xlabel="時間 [s]", ylabel=unit)
            ax.grid(alpha=0.2)
    axes[0, 0].plot([float(r["time_s"]) for r in rows],
                    [float(r["reference_vx"]) for r in rows], "k:", label="目標速度")
    metric = case["routes"]["measured"]
    axes[0, 1].axhline(metric["ideal_reference_distance_m"], color="k", ls=":", label="位置保持目標")
    axes[0, 1].axhline(17, color="#999999", ls=":", label="仮の使用長17 m")
    for ax in axes.flat:
        ax.axvspan(metric["commanded_hold_start_s"], metric["commanded_brake_start_s"],
                   color="gray", alpha=0.07)
    axes[0, 0].legend(fontsize=11)
    axes[0, 1].legend(fontsize=11)
    fig.suptitle("制御調整前後：同じモデル・同じ5 m/s／1.5秒指令・同じ計算時間\n"
                 "灰帯：速度保持指令／無風・16 V／オフライン予測、実飛行検証ではない", fontsize=16)
    fig.savefig(output / "control_before_after.png", dpi=150, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path)
    parser.add_argument("--schedule", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--accelerations", type=float, nargs="+", default=[3, 4, 5])
    parser.add_argument("--dt", type=float, default=0.004)
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--baseline-plot", type=Path, help="Baseline output for before/after plot")
    parser.add_argument("--velocity-kp", type=float, default=0.9)
    parser.add_argument("--velocity-ki", type=float, default=0.15)
    parser.add_argument("--attitude-frequency", type=float, default=7.8)
    parser.add_argument("--attitude-damping", type=float, default=0.9)
    parser.add_argument("--position-hold", action="store_true")
    parser.add_argument("--antiwindup", action="store_true")
    parser.add_argument("--accel-headroom", type=float, default=0.0,
                        help="Controller acceleration allowance beyond reference peak [m/s²]")
    parser.add_argument("--settle-s", type=float, default=6.0)
    parser.add_argument("--jerk-ramp-s", type=float, default=0.2)
    parser.add_argument("--routes", nargs="+", choices=("physical", "measured", "command_route"),
                        default=["physical", "measured", "command_route"])
    args = parser.parse_args()
    if args.plot_only:
        plot(args.output)
        if args.baseline_plot:
            plot_control_comparison(args.output, args.baseline_plot)
        return
    if min(args.velocity_kp, args.velocity_ki, args.accel_headroom, args.settle_s) < 0 or args.jerk_ramp_s <= 0:
        parser.error("Controller gains, headroom and settling duration must be nonnegative")
    tuning = ControllerSettings(velocity_kp=args.velocity_kp, velocity_ki=args.velocity_ki,
                                antiwindup=args.antiwindup, attitude_frequency=args.attitude_frequency,
                                attitude_damping=args.attitude_damping)
    if min(args.attitude_frequency, args.attitude_damping) <= 0:
        parser.error("Attitude frequency and damping must be positive")
    artifact = json.loads(args.models.read_text())
    schedule = json.loads(args.schedule.read_text())
    model_sha = hashlib.sha256(args.models.read_bytes()).hexdigest()
    if model_sha != schedule["model_sha256"]:
        raise ValueError("Coefficient/schedule hash mismatch")
    args.output.mkdir(parents=True, exist_ok=False)
    summary = dict(status="EXPLORATORY_NOT_FLIGHT_CLEARANCE", model_sha256=model_sha,
                   schedule_sha256=hashlib.sha256(args.schedule.read_bytes()).hexdigest(),
                   script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   dynamics_sha256=hashlib.sha256(Path(simulate.__code__.co_filename).read_bytes()).hexdigest(),
                   model_source=str(args.models), schedule_source=str(args.schedule), dt_s=args.dt,
                   target_vx_m_s=5, commanded_hold_s=1.5, wind_world_m_s=[0, 0, 0], voltage_V=16,
                   assumed_field_length_m=23, assumed_end_margin_m=3,
                   controller_settings=asdict(tuning), position_hold=args.position_hold,
                   settling_duration_s=args.settle_s,
                   simulated_routes=args.routes,
                   limitations=["Physical comparator is thrust/torque only, NOT PX4 standard SITL",
                                "Controller tuning on the simulated plant is exploratory, not independent validation",
                                "Initial hover load anchored by assumption; absolute aero baseline uncertain",
                                "Speed-node linear interpolation, not independent 2m/s identification",
                                "Both coefficient routes use simulated motor RPM, not future measurements",
                                "Same controller per route; separately closed-loop, NOT identical motor inputs",
                                "Position hold after reference braking" if args.position_hold else
                                "No position-hold after braking; speed regulation only",
                                "No wind, navigation errors, battery sag, ground/wall effects or uncertainty bands",
                                "3m margins are hypothetical; vehicle size and operational clearance unassessed"], cases=[])
    for acceleration in args.accelerations:
        profile = PassProfile(acceleration=acceleration, jerk_ramp_s=args.jerk_ramp_s)
        duration = profile.hover_s + 2 * profile.ramp_duration(5) + profile.hold_s + args.settle_s
        limit = acceleration + args.accel_headroom
        case = dict(acceleration_m_s2=acceleration, controller_accel_limit_m_s2=limit,
                    acceleration_jerk_ramp_s=profile.jerk_ramp_s, duration_s=duration, routes={})
        for route in args.routes:
            aircraft = Aircraft(artifact, route) if route == "physical" else ScheduledAircraft(artifact, route, schedule)
            rows, metrics = simulate(artifact, route, duration, args.dt, 16, 5, aircraft,
                                     reference_function=profile.reference, horizontal_accel_limit=limit,
                                     controller_settings=tuning,
                                     position_reference=profile.position_reference if args.position_hold else None)
            metrics.update(diagnostics(rows, profile, 5))
            write_csv(args.output / f"a{acceleration:g}_{route}.csv", rows)
            case["routes"][route] = metrics
            print(acceleration, route, json.dumps(metrics), flush=True)
        summary["cases"].append(case)
        (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
