"""Current absolute/increment models in the existing offline 6-DOF simulator.

Not PX4/Gazebo SITL, not flight validation. All RPM is simulation-generated.
No parameter search, sensor-zero update, aerodynamic clipping or hidden trim.
"""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import pickle
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import analysis as a
from scripts import simulate_graybox_forward_flight as flight
from scripts.current_graybox_aircraft import CurrentAircraft
from scripts.simulate_koil_pass import PassProfile, diagnostics


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def takeoff_height(time):
    """1 s on ground, 3 s smooth rise to 1.5 m, then altitude hold."""
    phase = float(np.clip((time-1.)/3.,0.,1.))
    height = 1.5*(phase-np.sin(2*np.pi*phase)/(2*np.pi))
    climb = .5*(1-np.cos(2*np.pi*phase)) if 0 < phase < 1 else 0.
    return height, climb


def run(args):
    import __main__
    __main__.Settings = a.Settings
    # Operator-specified checkpoint produced by this repository, never external pickle.
    if args.inputs:
        saved = a.read_json(args.inputs)
        cfg, motors = a.Settings(**saved["settings"]), saved["motor_models"]
    else:
        with args.trusted_prepared_checkpoint.open("rb") as stream:
            _, profile, cfg, _, _, _ = pickle.load(stream)
        motors = profile["motor_models"]
    cfg = replace(cfg, fz_model="sun_graybox", yaw_motion_model="sun_blended")
    baseline, latest, extra = [a.read_json(p) for p in (args.baseline, args.latest, args.extra)]
    artifact = dict(settings=asdict(cfg), motor_models=motors, models={})
    controller = flight.ControllerSettings(antiwindup=True)
    trajectory = PassProfile(acceleration=2.5, hold_s=1.5, hover_s=6. if args.takeoff else 2.)
    args.output.mkdir(parents=True, exist_ok=False)
    a.write_json(args.output/"inputs.json", dict(settings=asdict(cfg), motor_models=motors,
        controller=asdict(controller), trajectory=asdict(trajectory),
        sources={str(p):sha(p) for p in (args.baseline, args.latest, args.extra)},
        prepared_checkpoint=str(args.trusted_prepared_checkpoint) if args.trusted_prepared_checkpoint else None,
        saved_inputs=str(args.inputs) if args.inputs else None))
    records = []
    for conditioned in (False, True):
        condition = "hover_conditioned" if conditioned else "absolute"
        if condition not in args.conditions:
            continue
        for label, bundle in (("physics", None), ("adopted", baseline), ("latest", latest)):
            if label not in args.variants:
                continue
            routes = ("physical",) if bundle is None else args.routes
            for route in routes:
                vehicle = (flight.Aircraft(artifact, "physical", anchored=False) if bundle is None else
                    CurrentAircraft(cfg, motors, bundle, extra, route, conditioned=conditioned,
                                    startup_blend=args.takeoff))
                rows, metric = flight.simulate(artifact, route, args.duration, args.dt,
                    args.voltage, args.target, aircraft=vehicle,
                    reference_function=trajectory.reference, horizontal_accel_limit=2.5,
                    controller_settings=controller, position_reference=trajectory.position_reference,
                    height_reference=takeoff_height if args.takeoff else None, ground_start=args.takeoff)
                filename = f"{condition}-{label}-{route}.csv"
                flight.write_csv(args.output/filename, rows)
                metric.update(diagnostics(rows, trajectory, args.target))
                if args.takeoff:
                    altitude_error=np.array([r["height"]-r["reference_height"] for r in rows])
                    metric.update(max_altitude_tracking_error_m=float(np.max(np.abs(altitude_error))),
                        altitude_tracking_rmse_m=float(np.sqrt(np.mean(altitude_error**2))),
                        first_liftoff_s=next((r["time_s"] for r in rows if r["height"]>.02),None))
                records.append(dict(condition=condition, model=label, route=route,
                                    file=filename, metrics=metric))
                print(condition, label, route, metric["status"],
                      "distance", metric["x_span_m"], "height", metric["max_abs_height"], flush=True)
                a.write_json(args.output/"progress.json", dict(cases=records))
    a.write_json(args.output/"summary.json", dict(
        status="EXPLORATORY_OFFLINE_FREE_FLIGHT", cases=records,
        target_m_s=args.target, duration_s=args.duration, dt_s=args.dt, voltage=args.voltage,
        takeoff=args.takeoff,trajectory=asdict(trajectory),
        source_code={str(p):sha(p) for p in (Path(__file__), Path(flight.__file__))},
        limitations=[
            "Not PX4 SITL or standard Gazebo; shared provisional offline PID and quadratic propulsion baseline.",
            "Every route uses generated RPM from the SAME fitted motor dynamics; no measured flight RPM exists.",
            "Measured/command labels name the training route, not the simulation RPM source.",
            "Absolute composition uses a fixed initial-hover reference; no per-step reset or re-fit.",
            "hover_conditioned adds a disclosed fixed mg/zero-moment anchor, not independent absolute validation.",
            "Fx/Fy/Mx reuse the same existing extra-axis model in both identified candidates.",
            "Support is not guaranteed by speed alone; joint RPM/inflow/rate coverage remains unvalidated.",
            "Mass/inertia/rotor inertia and input latency retain experimental estimation limitations.",
            "Stopped traces are retained, not replaced, extended or clipped."]))
    if args.takeoff:
        record=a.read_json(args.output/"summary.json")
        record["limitations"].extend([
            "Ideal pinned level ground support until upward force exceeds weight; no landing-gear compliance/friction model.",
            "Startup below 0.3 hover angular speed uses propulsion physics only; linearly blend to fitted loads by 0.5.",
            "Startup transition thresholds are explicit engineering assumptions, NOT identified low-RPM aerodynamics."])
        a.write_json(args.output/"summary.json",record)


def plot(output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    fonts = {f.name for f in font_manager.fontManager.ttflist}
    plt.rcParams.update({"font.family": next((f for f in ("Hiragino Sans", "Noto Sans CJK JP")
                        if f in fonts), "sans-serif"), "font.size":14,
                        "axes.unicode_minus":False, "pdf.fonttype":42})
    summary = a.read_json(output/"summary.json")
    styles = dict(physics=("#333333", "--", "推進物理モデル"),
                  adopted=("#0072B2", "-", "旧同定モデル"),
                  latest=("#D55E00", "-.", "最新候補の同定モデル"))
    fields = [("vx", "前方速度 [m/s]",1), ("x", "前方移動距離 [m]",1),
              ("pitch_deg", "Pitch [度]",1), ("height", "高度変化 [m]",1),
              ("yaw_deg", "Yaw [度]",1), ("My", "Pitchモーメント [mN·m]",1000)]
    for condition in ("absolute", "hover_conditioned"):
        for route in a.ROUTES:
            fig, axes = plt.subplots(3,2,figsize=(13,11),layout="constrained")
            selected = [r for r in summary["cases"] if r["condition"] == condition and
                        (r["route"] == route or r["model"] == "physics")]
            if not selected:
                plt.close(fig)
                continue
            notes = []
            for record in selected:
                rows = a.read_csv(output/record["file"])
                t = np.array([float(r["time_s"]) for r in rows])
                color, style, label = styles[record["model"]]
                for ax,(field, ylabel, scale) in zip(axes.flat, fields):
                    ax.plot(t,[float(r[field])*scale for r in rows],color=color,ls=style,lw=2,label=label)
                    ax.set(xlabel="時間 [s]",ylabel=ylabel,xlim=(0,summary["duration_s"]))
                    ax.grid(alpha=.2)
                if record["metrics"]["status"] != "COMPLETED":
                    notes.append(label+"："+record["metrics"]["status"])
            full_t = np.linspace(0,summary["duration_s"],600)
            profile = PassProfile(**summary.get("trajectory",dict(acceleration=2.5,hold_s=1.5)))
            axes[0,0].plot(full_t,[profile.reference(t,summary["target_m_s"])[0] for t in full_t],
                           color="#777777",ls=":",label="目標速度")
            axes[0,0].legend(fontsize=11)
            if summary.get("takeoff"):
                axes[1,1].plot(full_t,[takeoff_height(t)[0] for t in full_t],color="#777777",ls=":",label="目標高度")
                axes[1,1].set_ylabel("地上からの高度 [m]")
                axes[1,1].legend(fontsize=11)
            anchor_label = "初期ホバーのみ釣り合い補正" if condition == "hover_conditioned" else "絶対値：初期荷重の補正なし"
            route_label = "実測RPMで学習した係数" if route == "measured" else "入力＋電圧経路で学習した係数"
            start_label="離陸→1.5 mホバー" if summary.get("takeoff") else "ホバー"
            fig.suptitle(f"{start_label}→目標{summary['target_m_s']:g} m/s→減速｜{anchor_label}\n"
                         f"{route_label}／全RPMはシミュレーション生成／PX4 SITLではない",fontsize=16)
            if notes:
                fig.supxlabel("途中終了：線の終端以降は未計算（理由はsummary.json）",fontsize=12)
            for extension in ("png","pdf"):
                fig.savefig(output/f"{condition}-{route}.{extension}",dpi=160,facecolor="white")
            plt.close(fig)
    a.write_json(output/"figure_manifest.json",dict(
        source="summary.json and all per-case CSVs",smoothing="none",
        uncertainty="Deterministic scenario only; no uncertainty intervals estimated",
        missing="Stopped traces end explicitly; no interpolation beyond their endpoint",
        dimensions_inches=[13,11],png_dpi=160,
        alt_text="各モデルの前方速度、距離、Pitch、Yaw、高度、Pitchモーメントの時系列比較。"))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--plot-only", action="store_true")
    p.add_argument("--trusted-prepared-checkpoint", type=Path)
    p.add_argument("--inputs", type=Path, help="Reuse exported settings/motors; no pickle/raw data needed")
    for key in ("baseline","latest","extra"):
        p.add_argument("--"+key,type=Path)
    p.add_argument("--target",type=float,default=5.)
    p.add_argument("--duration",type=float,default=14.)
    p.add_argument("--dt",type=float,default=.005)
    p.add_argument("--voltage",type=float,default=16.)
    p.add_argument("--takeoff",action="store_true",help="Start grounded with stopped motors and climb to 1.5 m")
    p.add_argument("--variants",nargs="+",choices=("physics","adopted","latest"),
                   default=["physics","adopted","latest"])
    p.add_argument("--routes",nargs="+",choices=a.ROUTES,default=list(a.ROUTES))
    p.add_argument("--conditions",nargs="+",choices=("absolute","hover_conditioned"),
                   default=["absolute","hover_conditioned"])
    args=p.parse_args()
    if args.plot_only:
        plot(args.output)
    else:
        if (not args.inputs and not args.trusted_prepared_checkpoint) or any(
                getattr(args,k) is None for k in ("baseline","latest","extra")):
            p.error("Model paths and trusted checkpoint are required")
        if not (0 < args.dt <= .01 and 0 < args.target <= 11.5 and args.duration > 0 and args.voltage > 0):
            p.error("Invalid scenario or integration settings")
        run(args)
