#!/usr/bin/env python3
"""FFTGYRO 最終解析（2026-10-13時点の結果を固定）。

読む順番: 1 読込・同期 → 2 姿勢 → 3 センサ自己荷重 → 4 RPM/物理
→ 5 空力項 → 6 同定 → 7 Yaw基準移送 → 8 全体実行・誤差計算。

通常: python analysis.py --bundle INPUT_BUNDLE --output NEW_OUTPUT
係数再推定: 上記に --refit を追加（項、正則化、foldは変更しない）。
文献RPM写像: --refit --motor-model KAUFMANN_JSON（風洞校正。停止・始動は未検証）。
理論項の再選択: --refit --reselect-physics（既存採用項＋理論候補、ridgeは固定）。
生ログ再読込: --raw-root RAW --aligned-root CLOCK_CACHE を追加。
運動補正した停止基準: --zero-recovery RECOVERY_SUMMARY。
追加基準で再fit: 上記に --refit --drift-prior DRIFT_PRIOR を付ける。
図・表の出力: python analysis_output.py --input NEW_OUTPUT --output REPORT
このファイルは計算結果を保存するだけ。描画は行わない。

通常入力は生ログから作った同期済み信号。予測値や再構成荷重は入力しない。
復号/時計合わせだけ src/fftgyro_quad_id/sync を使用。他の解析scriptは不要。
独立停止基準、採用マスク、校正値、採用項、既存foldは外部bundleに固定。
既定はMy/Mz/Fz。--six-axisでFx/Fy/Mxの項選択・条件別検証を追加。
物理比較は必ず実測RPM。command_routeは入力・電圧だけからRPMを計算。
無基準runは差分学習のみ。有風Yawの基準移送は独立絶対値検証ではない。
過去データの反復探索を経たgrouped OOFであり、新規盲検試験ではない。
オフラインRTS/zero-phase filteringは未来情報を使用。飛行制御用ではない。
"""

from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass, replace
import hashlib
from itertools import product
import json
from pathlib import Path
import re
import sys
from typing import Mapping, Sequence
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.linalg import solve_triangular
from scipy.optimize import lsq_linear, minimize_scalar
from scipy.signal import butter, savgol_filter, sosfiltfilt

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))
OUTPUTS = ("Fx", "Fy", "Fz", "Mx", "My", "Mz")
CHANNELS = ("My", "Mz", "Fz")
EXTRA_CHANNELS = ("Fx", "Fy", "Mx")
ROUTES = ("measured", "command_route")
SIGNS = np.array([1.0, 1.0, -1.0, -1.0])
FLU_TO_FRD = np.array([1.0, -1.0, -1.0])
SUN_REGIONS = ("sun_low", "sun_beta_0_30", "sun_beta_30_60", "sun_beta_60_90")
THEORY_VERSION = "body-rotor-physics-v1"
THEORY_ANGLES = ("1", "nx", "ny", "nz", "nx_ny", "nx_nz", "ny_nz", "ny2", "nz2")
THEORY_ROTOR_LOADS = ("plane_linear", "plane_quadratic", "axial", "horizontal2", "axial2")


# ===========================================================================
# 1. 設定と生ログの読み込み・時計同期
# ===========================================================================
@dataclass
class Settings:
    hz: int = 100
    stride: int = 10  # 平滑化後のみ間引く。既定10Hzで同定
    load_cutoff_hz: float = 5.0
    consistent_load_filter: bool = False  # 比較帯域整合の感度候補。実データ採用は未確認
    accept_leptrino_0x44: bool = True  # 2026-10-05本人指定：全runの0x44を解析に採用
    smooth_seconds: float = 0.5
    warmup_seconds: float = 2.0  # 指令RPMの初期値依存を評価から除く
    mass_kg: float = 1.5940849591374202  # Leptrino 3姿勢の重力円fit
    cg_flu_m: tuple = (-0.00118167487305559, -0.000193481696162535, 0.0812596628766431)
    propulsion_reference_cg_flu_m: tuple = (
        -0.00118167487305559,
        -0.000193481696162535,
        0.0812596628766431,
    )  # 既存のローター腕長が基準にしたCG。校正CGへ移す際も固定。
    # 2026-10-26最新指定：Pの推定から9/26を除外。9/13の質量・CGは維持。
    # Pは静止試験でなく確認済み寸法80+14.6mm。凍結bundleの旧設定は別途保持。
    pivot_flu_m: tuple = (0.0, 0.0, 0.0946)
    bias_sensor: tuple = (
        1.533289031909923,
        -0.8941354342513528,
        3.897356241489841,
        -0.0026839816200249916,
        -0.03530655222219087,
        0.08712733366048599,
    )
    inertia_kg_m2: tuple = (0.0169, 0.020094937562291817, 0.026162349956215518)
    rotor_inertia_kg_m2: float = 3.060838931314845e-5  # 別個体由来、未検証
    kf: float = 1.0611738388343621e-5  # 機体固定試験由来 N/(rad/s)^2
    kq: float = 2.0065025697826264e-7  # 同上 Nm/(rad/s)^2
    rotor_radius_m: float = 0.127
    arm_xy_m: float = 0.1768
    rho: float = 1.225
    encoder_sigma_rad: float = np.deg2rad(0.06)
    gyro_sigma_rad_s: float = 0.01
    jerk_density: float = 5.0
    bias_walk: float = 0.001
    pitch_encoder_sign: float = -1.0
    pitch_encoder_offset_deg: float = -147.1289
    axis_roll_deg: float = 0.0  # Pitch/Yaw軸より支持側の一定傾斜。既定値は従来互換。
    axis_pitch_offset_deg: float = 0.0  # raw encoder設定を変更せず、姿勢変換だけ補正。
    max_terms: int = 6  # 以下5設定はlegacy_cv専用。sun_publicでは使わない
    sun_max_terms: int = 18  # 旧Yaw領域分割の4領域合計上限
    shortlist: int = 8
    fit_samples_per_run: int = 300
    improvement: float = 0.01
    selection_method: str = "sun_public"  # Sun公開実装のF/PSE。legacy_cvは旧結果再現用
    free_propulsion: bool = False  # TrueならKF/KQに相当する項も再推定
    fz_model: str = "legacy"  # legacy / sun_reduced / sun_graybox
    yaw_motion_model: str = (
        "rotor_local"  # legacy / rate / rotor_local / sun_piecewise / sun_blended
    )
    sun_beta_halfwidth_deg: float = 5.0  # 事前固定。30/60度の±5度でC2接続
    sun_mu_halfwidth: float = 0.005  # 低速境界 .05 の±.005。現データで近傍検証不可
    clock_special_runs: tuple = (34, 639)
    prefix_segment_max: tuple = ((765, 24),)  # 既存採用判断: e0765は24区間まで

    def __post_init__(self):
        if self.fz_model not in ("legacy", "sun_reduced", "sun_graybox"):
            raise ValueError("unknown Fz model")
        if self.fz_model != "legacy" and self.selection_method != "sun_public":
            raise ValueError("Sun Fz base requires sun_public selection")
        if self.selection_method not in ("sun_public", "legacy_cv"):
            raise ValueError("unknown selection method")
        if self.yaw_motion_model not in (
            "legacy",
            "rate",
            "rotor_local",
            "sun_piecewise",
            "sun_blended",
        ):
            raise ValueError("unknown yaw motion model")
        if not 0 < self.sun_beta_halfwidth_deg <= 15 or not 0 < self.sun_mu_halfwidth < 0.05:
            raise ValueError("invalid Sun blending half-width")
        if not (self.hz > 0 and self.stride > 0 and self.smooth_seconds > 0):
            raise ValueError("sampling/filter settings must be positive")
        if not 0 < self.load_cutoff_hz < self.hz / 2:
            raise ValueError("load cutoff must be below Nyquist")
        if self.mass_kg <= 0 or np.any(np.asarray(self.inertia_kg_m2) <= 0):
            raise ValueError("mass/inertia must be positive")
        if not np.isfinite([self.axis_roll_deg, self.axis_pitch_offset_deg]).all():
            raise ValueError("axis corrections must be finite")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def reported_wind(session, record):
    """Per-run acquisition confirmation beats an older campaign-wide report."""
    for key in ("measured_wind_mps", "acquisition_wind_mps", "reported_wind_mps"):
        if record.get(key) is not None:
            return float(record[key]), key
    return float(session.get("wind_m_s", 0)), "nominal_fallback_unverified"


def col(rows, key, dtype=float):
    return np.array([row.get(key, "nan") or "nan" for row in rows], dtype=dtype)


def blocks(valid):
    ids = np.flatnonzero(valid)
    return [x for x in np.split(ids, np.flatnonzero(np.diff(ids) != 1) + 1) if len(x)]


def interpolate(time, value, grid, max_gap, hold=False):
    """短い間隔だけ内挿。指令は直前値保持。端点外/長い欠損はNaN。"""
    time, value = np.asarray(time), np.asarray(value)
    good = (
        np.isfinite(value).all(axis=tuple(range(1, value.ndim)))
        if value.ndim > 1
        else np.isfinite(value)
    )
    time, value = time[good], value[good]
    result = np.full((len(grid),) + value.shape[1:], np.nan)
    if len(time) < 2:
        return result
    if np.any(np.diff(time) <= 0):
        raise ValueError("時計の逆転/重複を確認してください")
    right = np.searchsorted(time, grid, side="right")
    left = np.clip(right - 1, 0, len(time) - 1)
    right = np.clip(right, 0, len(time) - 1)
    valid = (grid >= time[0]) & (grid <= time[-1])
    if hold:
        valid &= grid - time[left] <= max_gap
        result[valid] = value[left[valid]]
    else:
        span = time[right] - time[left]
        exact = grid == time[left]
        valid &= exact | ((span > 0) & (span <= max_gap))
        weight = np.divide(grid - time[left], span, out=np.zeros(len(grid)), where=span > 0)
        weight = weight.reshape((-1,) + (1,) * (value.ndim - 1))
        result[valid] = (value[left] + weight * (value[right] - value[left]))[valid]
    return result


def raw_paths(run, work, aligned_root, cfg):
    """既存の通信形式/時計推定を再利用する唯一の境界。生ログには書き込まない。

    align_run: boot ID/往復通信でPi・Nano・encoder時計を対応、binary荷重を復号。
    reconstruct: t_i=min(receive_i-i/1200)+i/1200-4.13ms を共通時計へ写像。
    時計を荷重fitに合わせて動かさない。summaryに残る未確定遅延は未確定のまま。
    """
    from fftgyro_quad_id.sync.force_grid import reconstruct
    from fftgyro_quad_id.sync.parallel_align import align_run

    if aligned_root:
        folder = Path(aligned_root) / run.name
    else:
        folder = work / "raw-cache" / run.name
    aligned, force = folder / "aligned", folder / "force-grid"
    if not (aligned / "summary.json").exists():
        if aligned_root:
            raise FileNotFoundError(aligned)
        number = int(run.name.split("_e")[1].split("_")[0])
        special = number in cfg.clock_special_runs
        align_run(
            run,
            aligned,
            rpm_model="smooth_interval" if special else "local_or_smooth",
            pi_model="smooth_interval" if special else "affine_or_smooth",
            accept_incomplete=True,
        )
    meta = read_json(aligned / "summary.json")
    digest = hashlib.sha256((run / "run.json").read_bytes()).hexdigest()
    if meta.get("source_manifest_sha256") != digest:
        raise ValueError("同期cacheと生ログが不一致")
    if not (force / "summary.json").exists():
        if aligned_root:
            raise FileNotFoundError(force)
        reconstruct(aligned, force, run_root=run)
    return aligned, force, meta


def analysis_rows(controls, number, cfg, adoption):
    """既存採用判断を保存。始動/移動/停止は解析対象に含めない。"""
    selected = np.array([r.get("analysis_window", "").lower() in ("true", "1") for r in controls])
    source = "explicit_analysis_window"
    if not selected.any() and (adoption or {}).get("motion_adoption", "").startswith("ACCEPT"):
        # 旧収録形式で全て0だったrun。採用CSVのある場合だけ既存ready_datasetと同じ規則。
        selected = np.array([r.get("stage") == "SEGMENT" for r in controls])
        source = "adopted_stage_SEGMENT_fallback"
    for restricted, limit in cfg.prefix_segment_max:
        if number == restricted:
            suffix = [
                int(m.group(1)) if (m := re.search(r"-(\d+)$", r["segment_id"])) else 10**9
                for r in controls
            ]
            selected &= np.asarray(suffix) <= limit
    if not selected.any():
        raise ValueError("明示的なanalysis_windowがありません")
    return selected, source


def load_raw(run, work, aligned_root, cfg, adoption=None):
    session = read_json(run / "resolved-session.json")
    axis = session["pid"]["axis"].upper()
    controls = read_csv(run / "pid/control.csv")
    ct = col(controls, "sent_ns", np.int64)
    number = int(run.name.split("_e")[1].split("_")[0])
    selected, mask_source = analysis_rows(controls, number, cfg, adoption)
    aligned, force_folder, clock_info = raw_paths(run, work, aligned_root, cfg)
    step = round(1e9 / cfg.hz)
    # 起動部も含め、入力RPMルートを評価区間の手前から連続シミュレーションする。
    grid = np.arange(((ct[0] + step - 1) // step) * step, ct[-1] + 1, step, dtype=np.int64)
    t = (grid - grid[0]) / 1e9
    source = np.clip(np.searchsorted(ct, grid, side="right") - 1, 0, len(ct) - 1)
    segment = np.array([row["segment_id"] for row in controls])[source]
    command = interpolate(
        ct,
        np.column_stack([col(controls, f"sent{i}") for i in range(1, 5)]),
        grid,
        30_000_000,
        True,
    )
    analysis = selected[source] & (grid - ct[source] <= 30_000_000)
    # PX4 boot time → aircraft monotonic: drift fit + 最速1%受信を因果的基準にする。
    px, received = col(controls, "px4_boot_ms") * 1e6, col(controls, "feedback_received_ns")
    ok = np.isfinite(px) & np.isfinite(received) & (px > 0) & (received > 0)
    if ok.sum() < 20:
        raise ValueError("PX4時計対が不足")
    origin = px[ok][0]
    slope, intercept = np.polyfit(px[ok] - origin, received[ok], 1)
    residual = received[ok] - (slope * (px[ok] - origin) + intercept)
    intercept += np.quantile(residual, 0.01) - slope * origin
    from pyulog import ULog

    files = sorted((run / "flightlog").glob("*.ulg"))
    if len(files) != 1:
        raise ValueError("ULogを1本に特定できません")
    log = ULog(
        str(files[0]), message_name_filter_list=["vehicle_angular_velocity", "battery_status"]
    )

    def topic(name):
        candidates = [x for x in log.data_list if x.name == name]
        if len(candidates) != 1:
            raise ValueError(f"{name}: topic/instanceが一意でない")
        return candidates[0].data

    gyro_data = topic("vehicle_angular_velocity")
    battery = topic("battery_status")
    gt = slope * np.asarray(gyro_data["timestamp_sample"], float) * 1000 + intercept
    gyro = interpolate(
        gt, np.column_stack([gyro_data[f"xyz[{i}]"] for i in range(3)]), grid, 30_000_000
    )
    bt = slope * np.asarray(battery["timestamp"], float) * 1000 + intercept
    voltage = interpolate(bt, battery["voltage_v"], grid, 1_000_000_000)
    for dropout in log.dropouts:
        end = slope * dropout.timestamp * 1000 + intercept
        bad = (grid >= end - dropout.duration * 1e6 - 50e6) & (grid <= end + 50e6)
        gyro[bad] = np.nan
        voltage[bad] = np.nan
    # 生RPM edge差から計算。2 edges/rev。並び替えはこの1回だけ。
    edges = read_csv(aligned / "rpm_edges_aligned.csv")
    rpm = []
    for channel in range(1, 5):
        rows = [r for r in edges if int(r["channel"]) == channel and r["aircraft_estimate_ns"]]
        times = col(rows, "aircraft_estimate_ns", np.int64)
        if len(times) < 2 or np.any(np.diff(times) <= 0):
            raise ValueError("RPM edge時計を確認")
        values = 30e9 / np.diff(times)
        values[(values < 100) | (values > 20000)] = np.nan
        rpm.append(interpolate(times[1:], values, grid, 50_000_000))
    rpm = np.column_stack(rpm)[:, [3, 0, 2, 1]]
    encoder = [
        r
        for r in read_csv(aligned / "encoder_aligned.csv")
        if r.get("aircraft_receive_estimate_ns")
        and r.get("within_clock_anchors", "true").lower() in ("true", "1")
    ]
    et = col(encoder, "aircraft_receive_estimate_ns", np.int64)
    if np.any(np.diff(et) < 0):
        raise ValueError("encoder受信時計が逆転")
    unique_time, group, counts = np.unique(et, return_inverse=True, return_counts=True)
    angles = []
    for key in ("experimental_pitch_deg", "experimental_yaw_deg"):
        value = np.unwrap(np.deg2rad(col(encoder, key)))
        # Windowsの同一serial受信batchに複数フレームがある場合は角度平均。
        value = np.bincount(group, weights=value) / counts
        angles.append(interpolate(unique_time, value, grid, 30_000_000))
    metadata = read_json(run / "encoder/metadata.json")
    if (
        metadata.get("pitch_sign") != cfg.pitch_encoder_sign
        or metadata.get("offset_roll_deg") != cfg.pitch_encoder_offset_deg
    ):
        raise ValueError("Pitch encoderの水平ゼロ校正が本試験と異なります")
    calibration = read_json(
        REPO / "configs/experiment_runner/yaw-front-encoder-calibration-20260916.json"
    )
    if (
        metadata.get("yaw_sign") != calibration["yaw_sign"]
        or metadata.get("offset_yaw_deg") != calibration["existing_encoder_offset_yaw_deg"]
    ):
        raise ValueError("encoder水平/正面ゼロ校正の対応を確認")
    front = {v["front_experimental_yaw_deg"] for v in calibration["pitch_calibrations"].values()}
    if len(front) != 1:
        raise ValueError("Yaw正面校正が一意でない")
    angles[1] -= np.deg2rad(front.pop())
    # 各grid時刻を中心とする10ms平均。status異常、欠落、USB受信gapは除外。
    force = read_csv(force_folder / "force_grid.csv")
    ft = col(force, "estimated_measurement_aircraft_ns", np.int64)
    bins = (ft - grid[0] + step // 2) // step
    inside = (bins >= 0) & (bins < len(grid))
    bins = bins[inside]
    count = np.bincount(bins, minlength=len(grid))
    load = []
    for key in ("fx_N", "fy_N", "fz_N", "mx_Nm", "my_Nm", "mz_Nm"):
        total = np.bincount(bins, weights=col(force, key)[inside], minlength=len(grid))
        load.append(np.divide(total, count, out=np.full(len(grid), np.nan), where=count >= 6))
    load = np.column_stack(load)
    status = np.zeros(len(grid), dtype=np.uint32)
    np.bitwise_or.at(status, bins, col(force, "status", np.uint32)[inside])
    # 2026-10-05本人指定：全runで0x44採用。strict設定なら旧採用CSVの例外だけ維持。
    override = (
        cfg.accept_leptrino_0x44
        or (adoption or {}).get("load_adoption") == "ACCEPT_WITH_0X44_OVERRIDE"
    )
    bad_status = ~leptrino_status_valid(status, accept_0x44=override)
    load[bad_status] = np.nan
    receipt = col(force, "receive_monotonic_ns", np.int64)
    for k in np.flatnonzero(np.diff(receipt) > 30_000_000):
        end = ft[k + 1]
        start = end - (receipt[k + 1] - receipt[k])
        load[(grid >= start) & (grid <= end)] = np.nan
    return dict(
        run=run.name,
        role=session["role"],
        axis=axis,
        wind=float(session.get("wind_m_s", 0)),
        heading=np.deg2rad(session.get("heading_deg", 0)),
        t=t,
        segment=segment,
        analysis=analysis,
        command=command,
        voltage=voltage,
        rpm=rpm,
        gyro=gyro,
        angles=np.column_stack(angles),
        load=load,
        grid_origin_ns=int(grid[0]),
        first_reference_window=first_hold_window(session, controls),
        provenance=dict(
            source=str(run),
            manifest_sha256=hashlib.sha256((run / "run.json").read_bytes()).hexdigest(),
            control_sha256=hashlib.sha256((run / "pid/control.csv").read_bytes()).hexdigest(),
            clock_state=clock_info["state"],
            analysis_mask_source=mask_source,
            px4_clock_slope=float(slope),
            status_0x44_override=override,
            status_0x44_adoption_scope="all_runs_user_20261005"
            if cfg.accept_leptrino_0x44
            else "legacy_adoption_csv",
            status_0x44_grid_rows=int(np.count_nonzero(status == 0x44)),
            encoder_duplicate_receipts=int(len(et) - len(unique_time)),
            status_fault_rows=int(bad_status.sum()),
            status_histogram={str(int(s)): int((status == s).sum()) for s in np.unique(status)},
        ),
    )


def first_hold_window(session, controls):
    """First numbered HOLD immediately preceding excitation, not entry/ramp.

    No force/rate/fit-error search for a nicer reference later in the run.
    """
    segments = session.get("segments", [])
    for i, segment in enumerate(segments):
        if segment.get("preparation_only") or segment.get("kind") == "HOLD":
            continue
        if not i or segments[i - 1].get("kind") != "HOLD":
            return None
        sid = segments[i - 1]["segment_id"]
        stamps = [
            int(r["sent_ns"])
            for r in controls
            if r["segment_id"] == sid and r["stage"] == "SEGMENT"
        ]
        if not stamps:
            return None
        hi = max(stamps) - 100_000_000
        return dict(
            hold_segment_id=sid,
            window_start_ns=max(min(stamps) + 250_000_000, hi - 1_500_000_000),
            window_end_ns=hi,
        )
    return None


def leptrino_status_valid(status, *, accept_0x44=True):
    """Analysis adoption only. Preserve flags; never accept 0x45/46/other faults.

    0x44 includes the documented rated-overrange bit. User-authorized inclusion
    is not sensor calibration, a healthy-status claim, or a hardware QC override.
    """
    status = np.asarray(status, dtype=np.uint32)
    return ((status & 0x07) == 0) | ((status == 0x44) & accept_0x44)


# ===========================================================================
# 2. エンコーダーとgyroの融合・力学再構成
# ===========================================================================
def fuse_motion(angle, rate, dt, cfg):
    """状態 x=[角度,角速度,角加速度,gyro bias]。gyroの姿勢角は使用しない。"""
    n = len(angle)
    F = np.array([[1, dt, dt * dt / 2, 0], [0, 1, dt, 0], [0, 0, 1, 0], [0, 0, 0, 1.0]])
    Q = np.zeros((4, 4))
    Q[3, 3] = cfg.bias_walk**2 * dt
    Q[:3, :3] = cfg.jerk_density**2 * np.array(
        [
            [dt**5 / 20, dt**4 / 8, dt**3 / 6],
            [dt**4 / 8, dt**3 / 3, dt**2 / 2],
            [dt**3 / 6, dt**2 / 2, dt],
        ]
    )
    H = np.array([[1.0, 0, 0, 0], [0, 1, 0, 1]])
    R = np.diag([cfg.encoder_sigma_rad**2, cfg.gyro_sigma_rad_s**2])
    x = np.array([angle[0], rate[0], 0.0, 0.0])
    P = np.diag([R[0, 0], R[1, 1], 25.0, 0.01])
    xs, ps, xp, pp = np.zeros((n, 4)), np.zeros((n, 4, 4)), np.zeros((n, 4)), np.zeros((n, 4, 4))
    for i in range(n):
        if i:
            x = F @ x
            P = F @ P @ F.T + Q
        xp[i], pp[i] = x, P
        K = np.linalg.solve(H @ P @ H.T + R, H @ P).T
        x = x + K @ (np.array([angle[i], rate[i]]) - H @ x)
        A = np.eye(4) - K @ H
        P = A @ P @ A.T + K @ R @ K.T
        xs[i], ps[i] = x, P
    for i in range(n - 2, -1, -1):
        gain = np.linalg.solve(pp[i + 1], F @ ps[i]).T
        xs[i] += gain @ (xs[i + 1] - xp[i + 1])
    return xs


def smooth(value, cfg):
    width = max(5, round(cfg.smooth_seconds * cfg.hz) | 1)
    return savgol_filter(value, width, 3, axis=0)


def load_bandwidth_filter(value, cfg):
    """欠損を跨がず、6軸荷重のButterworth＋Savgolを一度だけ適用。"""
    value = np.asarray(value, dtype=float)
    filtered = np.full_like(value, np.nan)
    finite = np.isfinite(value).all(axis=tuple(range(1, value.ndim)))
    sos = butter(4, cfg.load_cutoff_hz, fs=cfg.hz, output="sos")
    minimum = max(51, round(2 * cfg.smooth_seconds * cfg.hz))
    for ix in blocks(finite):
        if len(ix) >= minimum:
            filtered[ix] = smooth(sosfiltfilt(sos, value[ix], axis=0), cfg)
    return filtered


def rotation(axis, angle):
    angle = np.asarray(angle)
    out = np.zeros(angle.shape + (3, 3))
    i, j = (axis + 1) % 3, (axis + 2) % 3
    out[..., axis, axis] = 1
    out[..., i, i] = out[..., j, j] = np.cos(angle)
    out[..., j, i] = np.sin(angle)
    out[..., i, j] = -np.sin(angle)
    return out


def rig_rotation(theta, psi, heading, cfg):
    """Body FRD -> tunnel: upstream axis tilt, not terminal IMU misalignment.

    Angles theta/psi/heading are radians. Constant upstream rotations change
    gravity and wind projections, not body-frame joint omega/alpha.
    """
    return (
        rotation(2, heading)
        @ rotation(0, np.deg2rad(cfg.axis_roll_deg))
        @ rotation(1, np.asarray(theta) + np.deg2rad(cfg.axis_pitch_offset_deg))
        @ rotation(2, psi)
    )


def prepare_motion(raw, cfg):
    n = len(raw["t"])
    state = np.full((n, 2, 4), np.nan)
    load = np.full((n, 6), np.nan)
    valid = np.isfinite(np.column_stack([raw["angles"], raw["gyro"], raw["load"]])).all(axis=1)
    edge = np.zeros(n, bool)
    dt = 1 / cfg.hz
    for ix in blocks(valid):
        if len(ix) < max(51, round(2 * cfg.smooth_seconds * cfg.hz)):
            continue
        yaw = raw["angles"][ix, 1]
        g = raw["gyro"][ix]
        joint = np.column_stack([np.sin(yaw) * g[:, 0] + np.cos(yaw) * g[:, 1], g[:, 2]])
        for j in range(2):
            state[ix, j] = fuse_motion(raw["angles"][ix, j], joint[:, j], dt, cfg)
        load[ix] = load_bandwidth_filter(raw["load"][ix], cfg)
        trim = max(round(cfg.smooth_seconds * cfg.hz), 25)
        edge[ix[trim:-trim]] = True
    theta, psi = state[:, 0, 0], state[:, 1, 0]
    td, pd = state[:, 0, 1], state[:, 1, 1]
    tdd, pdd = state[:, 0, 2], state[:, 1, 2]
    omega = np.column_stack([np.sin(psi) * td, np.cos(psi) * td, pd])
    alpha = np.column_stack(
        [np.sin(psi) * tdd + np.cos(psi) * pd * td, np.cos(psi) * tdd - np.sin(psi) * pd * td, pdd]
    )
    R = rig_rotation(theta, psi, raw["heading"], cfg)
    raw.update(
        omega=omega,
        alpha=alpha,
        rotation=R,
        filtered_load=load,
        valid_motion=edge,
        joint_angle=state[:, :, 0],
    )
    return raw


def reconstruct_load(raw, cfg):
    measured = raw["filtered_load"] - np.asarray(cfg.bias_sensor)
    # 作用反作用を反転し、FLU→FRD。重心への移し替えを先に符号付きで計算。
    force = -measured[:, :3] * FLU_TO_FRD
    moment = -measured[:, 3:] * FLU_TO_FRD
    cg = np.asarray(cfg.cg_flu_m) * FLU_TO_FRD
    pg = cg - np.asarray(cfg.pivot_flu_m) * FLU_TO_FRD
    support_cg = moment - np.cross(cg, force)
    w, a = raw["omega"], raw["alpha"]
    v_cg = np.cross(w, pg)
    a_cg = np.cross(a, pg) + np.cross(w, v_cg)
    gravity = 9.80665 * raw["rotation"][:, 2, :]
    inertia = np.asarray(cfg.inertia_kg_m2)
    # 既定は従来のSGのみ。感度候補では慣性・重力にも荷重と同じ全フィルタを適用。
    # filtered_loadへ再適用はしない（荷重の二重処理を避ける）。
    dynamic = np.column_stack(
        [cfg.mass_kg * (a_cg - gravity), a * inertia + np.cross(w, w * inertia)]
    )
    if cfg.consistent_load_filter:
        dyn = load_bandwidth_filter(dynamic, cfg)
    else:
        dyn = np.full_like(dynamic, np.nan)
        for ix in blocks(np.isfinite(dynamic).all(axis=1)):
            if len(ix) > round(cfg.smooth_seconds * cfg.hz) + 4:
                dyn[ix] = smooth(dynamic[ix], cfg)
    target = dyn - np.column_stack([force, support_cg])
    # 対気速度=機体速度−風速度。風洞の流れはtunnelの-X方向。
    air = raw["wind"] * raw["rotation"][:, 0, :] + v_cg
    # alpha/betaは表示用。ホバリング近傍の角度特異性を避け、回帰にはmuの3成分を使う。
    flow_angles = np.column_stack(
        [np.arctan2(air[:, 2], air[:, 0]), np.arctan2(air[:, 1], np.hypot(air[:, 0], air[:, 2]))]
    )
    flow_angles[np.linalg.norm(air, axis=1) < 0.2] = np.nan
    raw.update(
        target=target,
        air=air,
        flow_angles=flow_angles,
        support=np.column_stack([force, support_cg]),
    )
    return raw


# ===========================================================================
# 3. Leptrino自身の荷重を機体から分離
# ===========================================================================
def parallel_axis(mass, position):
    r = np.asarray(position)
    return mass * ((r @ r) * np.eye(3) - np.outer(r, r))


def sensor_parameters(summary):
    c = summary["models"]["rigid"]["coefficients"]
    mass = c["mass_kg"]
    h = np.array([c["first_moment_" + a + "_kgm"] for a in "xyz"])
    inertia = np.array([[c["I" + min(i, j) + max(i, j) + "_kgm2"] for j in "xyz"] for i in "xyz"])
    center = h / mass
    central = inertia - parallel_axis(mass, center)
    # Roll inertia is poorly excited. Project the central second-moment matrix
    # onto PSD; this is an explicit physical regularization, NOT a new estimate.
    second = 0.5 * np.trace(central) * np.eye(3) - central
    eig, vectors = np.linalg.eigh(second)
    second_valid = (vectors * np.maximum(eig, 0)) @ vectors.T
    central_valid = np.trace(second_valid) * np.eye(3) - second_valid
    inertia_valid = central_valid + parallel_axis(mass, center)
    d = np.diag(FLU_TO_FRD)
    return dict(
        mass=mass,
        h=d @ h,
        inertia=d @ inertia_valid @ d,
        raw_inertia=d @ inertia @ d,
        center=d @ center,
        audit=dict(
            mass_kg=mass,
            center_sensor_FLU_m=center.tolist(),
            raw_inertia_sensor_FLU_kgm2=inertia.tolist(),
            regularized_inertia_sensor_FLU_kgm2=inertia_valid.tolist(),
            central_second_moment_eigenvalues=eig.tolist(),
            method="PSD projection of central second moment; mass and first moment unchanged",
            physical_inertia_independently_identified=False,
        ),
    )


def split_calibration(cfg, sensor):
    old_cg = np.asarray(cfg.cg_flu_m) * FLU_TO_FRD
    mass = cfg.mass_kg - sensor["mass"]
    cg = (cfg.mass_kg * old_cg - sensor["h"]) / mass
    total_origin = np.diag(cfg.inertia_kg_m2) + parallel_axis(cfg.mass_kg, old_cg)
    inertia = total_origin - sensor["inertia"] - parallel_axis(mass, cg)
    if mass <= 0 or np.linalg.eigvalsh(inertia).min() <= 0:
        raise ValueError("Nonphysical separated aircraft parameters")
    new = replace(
        cfg, mass_kg=mass, cg_flu_m=tuple(cg * FLU_TO_FRD), inertia_kg_m2=tuple(np.diag(inertia))
    )
    return new, inertia


def self_wrench(gravity, origin_acceleration, omega, alpha, sensor):
    effective_g = gravity - origin_acceleration
    h, inertia = sensor["h"], sensor["inertia"]
    force = sensor["mass"] * effective_g - np.cross(alpha, h) - np.cross(omega, np.cross(omega, h))
    moment = np.cross(h, effective_g) - alpha @ inertia.T - np.cross(omega, omega @ inertia.T)
    return np.column_stack((force, moment))


def mechanics_filter(values, cfg):
    """Use exactly the production mechanics bandwidth on both removed parts."""
    if cfg.consistent_load_filter:
        return load_bandwidth_filter(values, cfg)
    out = np.full_like(values, np.nan)
    for ix in blocks(np.isfinite(values).all(axis=1)):
        if len(ix) > round(cfg.smooth_seconds * cfg.hz) + 4:
            out[ix] = smooth(values[ix], cfg)
    return out


def shift_wrench(wrench, old_to_new):
    shifted = np.asarray(wrench).copy()
    shifted[:, 3:] -= np.cross(old_to_new, shifted[:, :3])
    return shifted


def reconstruct_body(raw, old_cfg, sensor):
    """Return body-only wrench and an independent aggregate-identity audit.

    The identity F_new=F_old, M_new=M_old-(r_new-r_old)×F_old is checked.
    This check prevents a manufactured improvement by subtracting self gravity
    while retaining the old aggregate weight/first moment.
    """
    cfg, inertia = split_calibration(old_cfg, sensor)
    reconstruct_load(raw, old_cfg)
    old_target = raw["target"].copy()
    w, a = raw["omega"], raw["alpha"]
    gravity = 9.80665 * raw["rotation"][:, 2, :]
    origin = -np.asarray(cfg.pivot_flu_m) * FLU_TO_FRD
    a0 = np.cross(a, origin) + np.cross(w, np.cross(w, origin))
    self_load = self_wrench(gravity, a0, w, a, sensor)
    filtered_self = mechanics_filter(self_load, cfg)
    cg = np.asarray(cfg.cg_flu_m) * FLU_TO_FRD
    pg = cg - np.asarray(cfg.pivot_flu_m) * FLU_TO_FRD
    velocity = np.cross(w, pg)
    acceleration = np.cross(a, pg) + np.cross(w, velocity)
    dynamic = np.column_stack(
        (cfg.mass_kg * (acceleration - gravity), a @ inertia.T + np.cross(w, w @ inertia.T))
    )
    measured = (raw["filtered_load"] - np.asarray(old_cfg.bias_sensor)) * np.tile(FLU_TO_FRD, 2)
    support_origin = -measured + filtered_self
    support_cg = shift_wrench(support_origin, cg)
    target = mechanics_filter(dynamic, cfg) - support_cg
    shift = cg - np.asarray(old_cfg.cg_flu_m) * FLU_TO_FRD
    expected = shift_wrench(old_target, shift)
    np.testing.assert_allclose(target, expected, atol=2e-11, rtol=1e-11, equal_nan=True)
    raw.update(
        target=target, air=raw["wind"] * raw["rotation"][:, 0, :] + velocity, support=support_cg
    )
    # Non-adopted sensitivity: historical inertia might already be aircraft-only.
    retained = np.diag(old_cfg.inertia_kg_m2) - inertia
    inertia_sensitivity = mechanics_filter(
        np.column_stack((np.zeros_like(w), a @ retained.T + np.cross(w, w @ retained.T))), cfg
    )
    return cfg, old_target, filtered_self, inertia_sensitivity


# ===========================================================================
# 4. 指令・電圧からのRPM推算と推進物理モデル
# ===========================================================================
def motor_basis(command, voltage, exponent=1.0):
    """RPM∞=(V/16)^γ (a u+b sqrt(u)+c u²); old artifacts default to γ=1."""
    u = np.maximum(command, 0.0)
    return (voltage[:, None] / 16) ** exponent * np.column_stack([u, np.sqrt(u), u * u])


def validate_training_roles(roles):
    roles = tuple(roles)
    if not roles or not set(roles) <= {"T", "D"}:
        raise ValueError("Training roles must be T and/or D; H is forbidden")
    return roles


def fit_motors(runs, cfg, voltage_model="power_law", *, training_roles=("T",)):
    training_roles = validate_training_roles(training_roles)
    training = [r for r in runs if r["role"] in training_roles]
    if len(training) < 3:
        raise ValueError("モーターとGray-boxの同定にはT runを3本以上指定")
    models = []
    dt = 1 / cfg.hz
    for motor in range(4):
        xs = []
        ys = []
        steady_inputs = []
        series = []
        for r in training:
            u = r["command"][:, motor]
            v = r["voltage"]
            rpm = r["rpm"][:, motor]
            ok = np.isfinite(u) & np.isfinite(v) & np.isfinite(rpm) & (u >= 0.1) & (v > 5)
            steady = ok & (np.abs(np.gradient(u, dt)) < 0.03)
            idx = np.flatnonzero(steady)[:: cfg.stride]
            if len(idx) < 10:
                continue
            weight = 1 / np.sqrt(len(idx))
            xs.append(motor_basis(u[idx], v[idx]) * weight)
            ys.append(rpm[idx] * weight)
            steady_inputs.append((u[idx], v[idx], weight))
            series.append((u, v, rpm, ok))
        if not xs:
            raise ValueError(f"M{motor + 1}の入力/RPM学習データなし")
        coef = lsq_linear(np.vstack(xs), np.concatenate(ys), bounds=(0, np.inf)).x
        exponent = 1.0
        if voltage_model == "power_law":
            if np.ptp(np.concatenate([v for _, v, _ in steady_inputs])) < 1e-6:
                raise ValueError("電圧一定ではγを識別できません。proportionalを指定してください")

            # Profile γ using T only. At each γ, fit the same nonnegative 3-term map.
            def solve_map(gamma):
                x = np.vstack([motor_basis(u, v, gamma) * weight for u, v, weight in steady_inputs])
                y = np.concatenate(ys)
                scale = np.linalg.norm(x, axis=0)
                c = lsq_linear(x / scale, y, bounds=(0, np.inf)).x / scale
                return c, float(np.sum((x @ c - y) ** 2))

            opt = minimize_scalar(
                lambda gamma: solve_map(gamma)[1], bounds=(-2, 3), method="bounded"
            )
            exponent = float(opt.x)
            coef, _ = solve_map(exponent)
            if exponent < -1.99 or exponent > 2.99:
                raise ValueError(f"M{motor + 1}: 電圧指数が探索境界。自動採用しません")
        elif voltage_model != "proportional":
            raise ValueError("Unknown motor voltage model")
        best = None
        for delay in (0.0, 0.02, 0.05, 0.1):
            before = []
            after = []
            goals = []
            for u, v, rpm, ok in series:
                lag = round(delay / dt)
                index = np.arange(1, len(u))
                delayed = index - lag
                good = ok[index] & ok[index - 1] & (delayed >= 0) & ok[np.maximum(delayed, 0)]
                index = index[good]
                delayed = delayed[good]
                before.extend(rpm[index - 1])
                after.extend(rpm[index])
                goals.extend(motor_basis(u[delayed], v[delayed], exponent) @ coef)
            before, after, goals = map(np.asarray, (before, after, goals))
            taus = []
            loss = 0.0
            for up in (True, False):
                take = (goals >= before) == up
                if take.sum() < 20:
                    raise ValueError("上昇/下降応答を同定できる点が不足")

                def objective(tau, goal=goals[take], previous=before[take], actual=after[take]):
                    pred = goal + (previous - goal) * np.exp(-dt / tau)
                    return float(np.mean((actual - pred) ** 2))

                opt = minimize_scalar(objective, bounds=(0.01, 1.0), method="bounded")
                taus.append(float(opt.x))
                loss += float(opt.fun) * take.sum()
            item = dict(
                map_coefficients=coef.tolist(),
                voltage_exponent=exponent,
                static_map_model=voltage_model,
                tau_up_s=taus[0],
                tau_down_s=taus[1],
                delay_s=delay,
                training_one_step_mse=loss / len(before),
            )
            if best is None or item["training_one_step_mse"] < best["training_one_step_mse"]:
                best = item
        models.append(best)
        # 低速の指令変化だけでは時定数を分離できない。境界値を実機定数と誤認しない。
        best["identifiability_warning"] = (
            "時定数が探索境界付近。遅延/観測ノイズとの分離は未確認、実機定数とは断定不可。"
            if min(best["tau_up_s"], best["tau_down_s"]) < 0.011
            or max(best["tau_up_s"], best["tau_down_s"]) > 0.99
            else "有効時定数。入力の励起帯域と同期誤差に依存する。"
        )
    return models


def predict_rpm(command, voltage, models, cfg):
    """自由応答: 初期RPMは指令map。評価の実測RPMによるリセットは一切しない。"""
    if any(m.get("kind") == "kaufmann2023" for m in models):
        from fftgyro_quad_id.identification.motor_literature import predict
        if len(models) != 4 or not all(m.get("kind") == "kaufmann2023" for m in models):
            raise ValueError("Kaufmannモデルは4基共通の構造で指定してください")
        taus = {m["tau_s"] for m in models}
        if len(taus) != 1 or any(m.get("delay_s", 0) != 0 for m in models):
            raise ValueError("文献の構造は共通の一次遅れ・純遅延なしです")
        return predict(command, voltage, models, dt=1/cfg.hz, tau=taus.pop(),
                       warmup=cfg.warmup_seconds)
    prediction = np.full_like(command, np.nan)
    valid = np.isfinite(command).all(axis=1) & np.isfinite(voltage) & (voltage > 5)
    warm = np.zeros(len(command), bool)
    dt = 1 / cfg.hz
    for ix in blocks(valid):
        for j, m in enumerate(models):
            target = motor_basis(
                command[ix, j], voltage[ix], m.get("voltage_exponent", 1.0)
            ) @ np.asarray(m["map_coefficients"])
            lag = round(m["delay_s"] / dt)
            y = float(target[0])
            for k, row in enumerate(ix):
                goal = target[max(0, k - lag)]
                tau = m["tau_up_s"] if goal >= y else m["tau_down_s"]
                y = goal + (y - goal) * np.exp(-dt / tau)
                prediction[row, j] = y
        warm[ix[round(cfg.warmup_seconds / dt) :]] = True
    return prediction, warm


def route_state(raw, rpm, cfg):
    w = rpm * 2 * np.pi / 60
    valid = np.isfinite(w).all(axis=1) & (w > 0).all(axis=1) & raw["valid_motion"]
    filtered = np.full_like(w, np.nan)
    derivative = np.full_like(w, np.nan)
    physical_speed = np.full_like(w, np.nan)
    physical_derivative = np.full_like(w, np.nan)
    for ix in blocks(valid):
        if len(ix) < round(cfg.smooth_seconds * cfg.hz) + 5:
            continue
        filtered[ix] = smooth(w[ix], cfg)
        derivative[ix] = np.gradient(filtered[ix], 1 / cfg.hz, axis=0)
        if cfg.consistent_load_filter:
            # featuresは従来SG-RPM。物理荷重は未SG-RPMから先に組み立て、
            # その6軸出力へ共通フィルタを一回だけ適用する（SGを二重にしない）。
            physical_speed[ix] = w[ix]
            physical_derivative[ix] = np.gradient(w[ix], 1 / cfg.hz, axis=0)
        trim = max(2, round(cfg.smooth_seconds * cfg.hz / 2))
        filtered[np.r_[ix[:trim], ix[-trim:]]] = np.nan
        derivative[np.r_[ix[:trim], ix[-trim:]]] = np.nan
    w = filtered
    w2 = w * w
    bar = np.sqrt(w2.mean(axis=1))
    denom = bar * cfg.rotor_radius_m
    mu = raw["air"] / denom[:, None]
    pqr = raw["omega"] * cfg.arm_xy_m / denom[:, None]
    up = w2 @ np.array([-1, 1, 1, -1]) / bar**2
    uq = w2 @ np.array([1, -1, 1, -1]) / bar**2
    ur = w2 @ SIGNS / bar**2
    force_scale = cfg.rho * 4 * np.pi * cfg.rotor_radius_m**2 * denom**2
    moment_scale = force_scale * cfg.arm_xy_m
    if not cfg.consistent_load_filter:
        physical_speed, physical_derivative = w, derivative
    physical_squared = physical_speed**2
    # Rotor相対角運動量hは、採用した反作用方向と逆。-dh-omega×hが機体への反作用。
    h = np.zeros((len(w), 3))
    dh = h.copy()
    h[:, 2] = -cfg.rotor_inertia_kg_m2 * (physical_speed @ SIGNS)
    dh[:, 2] = -cfg.rotor_inertia_kg_m2 * (physical_derivative @ SIGNS)
    reaction = -dh - np.cross(raw["omega"], h)
    physical = np.column_stack(
        [
            np.zeros((len(w), 2)),
            -cfg.kf * physical_squared.sum(axis=1),
            cfg.kf * cfg.arm_xy_m * physical_squared @ np.array([-1, 1, 1, -1]),
            cfg.kf * cfg.arm_xy_m * physical_squared @ np.array([1, -1, 1, -1]),
            cfg.kq * physical_squared @ SIGNS,
        ]
    )
    # 既存の腕長は基準CGまわり。新CGへM'=M-(r_new-r_ref)×Fを適用。
    cg_shift = (
        np.asarray(cfg.cg_flu_m) - np.asarray(cfg.propulsion_reference_cg_flu_m)
    ) * FLU_TO_FRD
    physical[:, 3:] -= np.cross(cg_shift, physical[:, :3])
    physical[:, 3:] += reaction
    if cfg.consistent_load_filter:
        physical = load_bandwidth_filter(physical, cfg)
        physical[~np.isfinite(w).all(axis=1)] = np.nan
    state = dict(
        mu_x=mu[:, 0],
        mu_y=mu[:, 1],
        mu_z=mu[:, 2],
        abs_mu_x=np.abs(mu[:, 0]),
        abs_mu_y=np.abs(mu[:, 1]),
        pbar=pqr[:, 0],
        qbar=pqr[:, 1],
        rbar=pqr[:, 2],
        abs_pbar=np.abs(pqr[:, 0]),
        abs_qbar=np.abs(pqr[:, 1]),
        abs_rbar=np.abs(pqr[:, 2]),
        up=up,
        uq=uq,
        ur=ur,
        abs_up=np.abs(up),
        abs_uq=np.abs(uq),
        abs_ur=np.abs(ur),
    )
    rotor_flow = rotor_local_flow(raw, w, cfg)
    state.update(rotor_local_yaw(raw, w, cfg, flow=rotor_flow))
    state.update(sun_yaw_regions(mu, pqr[:, 2], ur, raw["omega"][:, 2], cfg))
    if cfg.fz_model != "legacy":
        # Ct,h from independent existing KF, NOT from this run's measured Fz.
        ct_hover = cfg.kf / (cfg.rho * np.pi * cfg.rotor_radius_m**4)
        induced, ambiguous = induced_inflow(mu, ct_hover)
        state.update(mu_h2=mu[:, 0] ** 2 + mu[:, 1] ** 2, inflow=induced - mu[:, 2])
        raw.setdefault("inflow_audit", []).append(
            dict(
                ct_hover=ct_hover,
                finite_rows=int(np.isfinite(induced).sum()),
                ambiguous_rows=int(ambiguous.sum()),
                basis="Existing fixed-test KF; not independently measured free-flight hover Ct",
            )
        )
    return dict(
        state=state,
        physical=physical,
        force_scale=force_scale,
        moment_scale=moment_scale,
        rpm=rpm,
        rotor_flow=rotor_flow,
        air_frd_m_s=raw["air"],
        omega_body_rad_s=raw["omega"],
    )


def induced_inflow(mu, ct_hover):
    """Sun Eq.34: 2*nu*sqrt(mu_h²+(nu-mu_z)²)=Ct,h.

    Positive roots only. Near-axial descent can have three roots; mask those
    rows instead of choosing a root using observed forces or fitting error.
    This is a quasi-steady momentum model, not a vortex-ring/dynamic-inflow model.
    """
    if not np.isfinite(ct_hover) or ct_hover <= 0:
        raise ValueError("Ct,h must be positive")
    mu = np.asarray(mu, dtype=float)
    h2, z = mu[:, 0] ** 2 + mu[:, 1] ** 2, mu[:, 2]
    finite = np.isfinite(mu).all(axis=1)
    disc = z * z - 8 * h2
    extrema = finite & (z > 0) & (disc > 0)
    root = np.sqrt(np.maximum(disc, 0))
    nmax, nmin = (3 * z - root) / 4, (3 * z + root) / 4

    def flux(nu):
        return 2 * nu * np.sqrt(h2 + (nu - z) ** 2)

    ambiguous = extrema & (ct_hover >= flux(nmin) - 1e-12) & (ct_hover <= flux(nmax) + 1e-12)
    lo = np.zeros(len(mu))
    hi = np.maximum(z, 0) + np.sqrt(ct_hover / 2) + 1
    # The upper bracket always has flux > Ct. A unique root is bracketed even
    # when flux has extrema, provided Ct is outside their ambiguous interval.
    for _ in range(64):
        mid = (lo + hi) / 2
        lower = flux(mid) < ct_hover
        lo = np.where(lower, mid, lo)
        hi = np.where(lower, hi, mid)
    nu = (lo + hi) / 2
    nu[~finite | ambiguous] = np.nan
    return nu, ambiguous


def monomials(variables, degree):
    return [
        tuple((name, power) for name, power in zip(variables, powers, strict=True) if power)
        for powers in product(range(degree + 1), repeat=len(variables))
        if sum(powers) <= degree
    ]


def smooth_switch(value, boundary, halfwidth):
    """Quintic transition: weights and first/second derivatives join continuously."""
    t = np.clip((value - boundary + halfwidth) / (2 * halfwidth), 0.0, 1.0)
    return t**3 * (10 - 15 * t + 6 * t**2)


def sun_yaw_regions(mu, rbar, ur, yaw_rate, cfg=None):
    """Sun Eq.2/Table12--13: beta=asin(v/sqrt(u²+v²)), not aircraft yaw.

    atan2(v,abs(u)) is equivalent and handles reverse flow. mu_x still retains
    the forward/backward distinction. At zero horizontal flow beta is undefined;
    use zero only as a partition convention, and expose that limitation.
    """
    advance = np.linalg.norm(mu, axis=1)
    beta = np.arctan2(mu[:, 1], np.abs(mu[:, 0]))
    ab = np.abs(beta)
    low = advance <= 0.05
    result = dict(
        sun_low=low.astype(float),
        sun_beta_0_30=((~low) & (ab < np.pi / 6)).astype(float),
        sun_beta_30_60=((~low) & (ab >= np.pi / 6) & (ab < np.pi / 3)).astype(float),
        sun_beta_60_90=((~low) & (ab >= np.pi / 3)).astype(float),
        sun_beta=beta,
        sun_mu=advance,
        yaw_rate=yaw_rate,
        ur_absur=ur * np.abs(ur),
        rbar_absrbar=rbar * np.abs(rbar),
        muy_absmuy=mu[:, 1] * np.abs(mu[:, 1]),
    )
    if cfg is not None and cfg.yaw_motion_model == "sun_blended":
        high = smooth_switch(advance, 0.05, cfg.sun_mu_halfwidth)
        a = smooth_switch(np.rad2deg(ab), 30.0, cfg.sun_beta_halfwidth_deg)
        b = smooth_switch(np.rad2deg(ab), 60.0, cfg.sun_beta_halfwidth_deg)
        result.update(
            sun_low=1 - high,
            sun_beta_0_30=high * (1 - a),
            sun_beta_30_60=high * (a - b),
            sun_beta_60_90=high * b,
        )
    return result


def rotor_local_flow(raw, rotor_speed, cfg):
    """Each rotor's local freestream, normalized by its own |Omega_i| R.

    mu_h,i = hypot(V_x,i, V_y,i)/(|Omega_i| R); lambda_i = V_z,i/(|Omega_i| R).
    lambda is axial freestream, NOT total inflow including the induced wake.
    Zero/missing speed or nonfinite ratios stay unavailable; no RPM floor or
    measured-RPM substitution is used. Geometry is the existing nominal plane.
    """
    speed = np.abs(np.asarray(rotor_speed, dtype=float))
    if speed.ndim != 2 or speed.shape[1] != 4:
        raise ValueError("Expected four rotor angular speeds per sample")
    if cfg.rotor_radius_m <= 0 or cfg.arm_xy_m <= 0:
        raise ValueError("Rotor radius and arm length must be positive")
    shift = (np.asarray(cfg.cg_flu_m) - cfg.propulsion_reference_cg_flu_m) * FLU_TO_FRD
    positions = cfg.arm_xy_m * np.array([[1, 1, 0], [-1, -1, 0], [1, -1, 0], [-1, 1, 0]]) - shift
    velocity = raw["air"][:, None, :] + np.cross(raw["omega"][:, None, :], positions)
    tip_speed = speed * cfg.rotor_radius_m
    available = np.isfinite(tip_speed) & (tip_speed > 0) & np.isfinite(velocity).all(axis=2)
    mu = np.full_like(velocity, np.nan)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        np.divide(velocity, tip_speed[:, :, None], out=mu, where=available[:, :, None])
        available &= np.isfinite(mu).all(axis=2)
        mu[~available] = np.nan
        # S_total = sum_i rho*A*(Omega_i*R)^2.  Consequently averaging
        # (Omega_i^2 / mean(Omega^2))*f(mu_i) preserves each rotor's load scale.
        mean_squared = np.mean(speed**2, axis=1)
        weight = np.full_like(speed, np.nan)
        np.divide(
            speed**2,
            mean_squared[:, None],
            out=weight,
            where=np.isfinite(mean_squared[:, None]) & (mean_squared[:, None] > 0),
        )
    return dict(
        positions_frd_m=positions,
        velocity_frd_m_s=velocity,
        omega_abs_rad_s=speed,
        mu=mu,
        mu_h=np.hypot(mu[:, :, 0], mu[:, :, 1]),
        lambda_axial=mu[:, :, 2],
        omega_squared_weight=weight,
        available=available,
    )


def rotor_local_yaw(raw, rotor_speed, cfg, *, flow=None):
    """Rotor-plane drag and empirical axial-flow torque; no induced-wake model.

    v_i = v_air,CG + omega_body cross r_i (body relative to air, FRD).
    Nominal rotor plane passes through the reference CG; its height is not measured.
    Positive linear/quadratic coefficients give in-plane forces opposing local flow.
    Each rotor uses its own |Omega_i| R. Its contribution is weighted by
    Omega_i^2/mean(Omega^2) before summing with the existing moment scale.
    For these linear/quadratic features this is algebraically equivalent to
    the previous shared-RMS implementation; old fitted coefficients stay valid.
    """
    flow = rotor_local_flow(raw, rotor_speed, cfg) if flow is None else flow
    positions = flow["positions_frd_m"]
    plane = flow["mu"].copy()
    plane[:, :, 2] = 0
    weighted_plane = plane * flow["omega_squared_weight"][:, :, None]
    linear = -np.cross(positions / cfg.arm_xy_m, weighted_plane)[:, :, 2].mean(axis=1)
    quadratic = -np.cross(
        positions / cfg.arm_xy_m, np.linalg.norm(plane, axis=2)[:, :, None] * weighted_plane
    )[:, :, 2].mean(axis=1)
    axial = np.mean(SIGNS * flow["omega_squared_weight"] * flow["lambda_axial"], axis=1)
    return dict(local_yaw_linear=linear, local_yaw_quadratic=quadratic, local_yaw_axial=axial)


# ===========================================================================
# 5. 採用済みの空力候補項（各RPM経路で計算）
# ===========================================================================
def theory_feature_names(output):
    """Deterministic candidates, not selected from observed loads or run labels."""
    if output not in OUTPUTS:
        raise ValueError("Unknown wrench component")
    if output.startswith("F"):
        names = ["body_drag__" + k for k in THEORY_ANGLES]
        names += ["body_normal_y", "body_normal_z"]
    else:
        axis = "xyz".index(output[-1].lower())
        names = ["body_pressure__" + k for k in THEORY_ANGLES]
        names += [f"body_drag_arm_{a}__{k}" for j, a in enumerate("xyz")
                  if j != axis for k in THEORY_ANGLES]
        names += ["damping_body", "damping_rotor", "damping_quadratic"]
    names += ["rotor_" + k for k in THEORY_ROTOR_LOADS]
    # These are corrections to, not replacements for, the frozen propulsion.
    names += [f"rotor_thrust_{i}" for i in range(1, 5)] if output in ("Fz", "Mx", "My") else []
    names += [f"rotor_torque_{i}" for i in range(1, 5)] if output == "Mz" else []
    return tuple("theory__" + name for name in names)


def theory_feature_bank(data, cfg, output):
    """Dimensional N / N m bases; coefficients are dimensionless corrections.

    S_ref=4*pi*R^2 is a normalization area, NOT the measured frontal area.
    n=V/|V|=(cos(alpha)cos(beta), sin(beta), sin(alpha)cos(beta)).
    Direction cosines avoid Euler-heading dependence and angle branch jumps.
    Sun 2019 Eqs.25/32 motivate rotor flow loads; vector drag, bounded angular
    modulation and pressure-arm bases are explicit extensions, not Sun's formula.
    No sensor force, zero, run ID or measured RPM from another route is read.
    """
    velocity = np.asarray(data["air_frd_m_s"], float)
    rate = np.asarray(data["omega_body_rad_s"], float)
    if velocity.ndim != 2 or velocity.shape[1] != 3 or rate.shape != velocity.shape:
        raise ValueError("Expected matching N by 3 body airspeed and angular rate")
    speed = np.linalg.norm(velocity, axis=1)
    direction = np.divide(velocity, speed[:, None], out=np.zeros_like(velocity),
                          where=speed[:, None] != 0)
    nx, ny, nz = direction.T
    angular = dict(zip(THEORY_ANGLES,
                       (np.ones(len(speed)), nx, ny, nz, nx*ny, nx*nz, ny*nz, ny**2, nz**2)))
    area = np.pi * cfg.rotor_radius_m**2
    length = cfg.arm_xy_m
    pressure = 0.5 * cfg.rho * (4 * area) * speed**2
    drag = -0.5 * cfg.rho * (4 * area) * speed[:, None] * velocity
    axis = "xyz".index(output[-1].lower())
    force_output = output.startswith("F")
    bank = {}
    if force_output:
        bank.update({"body_drag__" + k: drag[:, axis] * v for k, v in angular.items()})
        for j, name in ((1, "y"), (2, "z")):
            # Normal load perpendicular to the freestream; odd in incidence.
            normal = np.eye(3)[j] - direction[:, j, None] * direction
            bank["body_normal_" + name] = pressure * direction[:, j] * normal[:, axis]
    else:
        bank.update({"body_pressure__" + k: pressure * length * v for k, v in angular.items()})
        for j, name in enumerate("xyz"):
            if j == axis:
                continue
            arm_moment = np.cross(length * np.eye(3)[j], drag)[:, axis]
            bank.update({f"body_drag_arm_{name}__{k}": arm_moment * v for k, v in angular.items()})
        bank["damping_body"] = -0.5 * cfg.rho * (4 * area) * length**2 * speed * rate[:, axis]
        bank["damping_quadratic"] = -0.5 * cfg.rho * (4 * area) * length**3 * rate[:, axis] * np.abs(rate[:, axis])

    flow = data["rotor_flow"]
    omega = flow["omega_abs_rad_s"]  # own-route, filtered rotor angular speeds
    local = flow["velocity_frd_m_s"]
    plane = local.copy()
    plane[:, :, 2] = 0
    horizontal = np.linalg.norm(plane, axis=2)
    rotor_forces = {
        "plane_linear": -cfg.rho * area * cfg.rotor_radius_m * omega[:, :, None] * plane,
        "plane_quadratic": -cfg.rho * area * horizontal[:, :, None] * plane,
    }
    for name, value in (("axial", cfg.rotor_radius_m * omega * local[:, :, 2]),
                        ("horizontal2", horizontal**2),
                        ("axial2", local[:, :, 2] * np.abs(local[:, :, 2]))):
        rotor_forces[name] = np.zeros_like(local)
        rotor_forces[name][:, :, 2] = -cfg.rho * area * value
    positions = flow["positions_frd_m"]
    for name, forces in rotor_forces.items():
        loads = forces if force_output else np.cross(positions, forces)
        bank["rotor_" + name] = loads[:, :, axis].sum(axis=1)
    if not force_output:
        bank["damping_rotor"] = -cfg.rho * area * cfg.rotor_radius_m * length**2 * omega.sum(axis=1) * rate[:, axis]
    if output in ("Fz", "Mx", "My"):
        thrust = -cfg.rho * area * cfg.rotor_radius_m**2 * omega**2
        for i in range(4):
            factor = 1.0 if force_output else np.cross(positions[i], [0., 0., 1.])[axis]
            bank[f"rotor_thrust_{i+1}"] = factor * thrust[:, i]
    if output == "Mz":
        for i in range(4):
            bank[f"rotor_torque_{i+1}"] = SIGNS[i] * cfg.rho * area * cfg.rotor_radius_m**3 * omega[:, i]**2
    bank = {"theory__" + k: v for k, v in bank.items()}
    assert set(bank) == set(theory_feature_names(output))
    return bank


def evaluate_names(state, scale, names):
    """Evaluate every monomial BEFORE taking any time/reference average."""
    features = {}
    for name in names:
        factors = []
        for factor in name.split("*"):
            if factor == "1":
                continue
            key, power = factor.split("^") if "^" in factor else (factor, "1")
            factors.append(np.asarray(state[key], float) ** int(power))
        features[name] = (
            np.asarray(scale, float) * np.prod(factors, axis=0)
            if factors
            else np.asarray(scale, float).copy()
        )
    return features


def rpm_weights(rpm, balanced):
    """Relative loss weights only; not statistical confidence weights."""
    rpm = np.asarray(rpm, float)
    if not np.isfinite(rpm).all() or (rpm < 0).any():
        raise ValueError("Finite nonnegative own-route RPM required")
    if not balanced:
        return np.ones(len(rpm))
    bins = np.floor(rpm / 750).astype(int)
    _, inverse, count = np.unique(bins, return_inverse=True, return_counts=True)
    weight = np.minimum(len(rpm) / (len(count) * count[inverse]), 4.0)
    return weight / weight.mean()


def speed_features(data, output):
    """Extra shared rotor-speed/flow terms; no run, clock or target features."""
    state = data["state"]
    scale = data["moment_scale"]
    # Use smoothed scale-consistent speed; caller stores it explicitly.
    speed = data["rms_speed_ratio"] - 1
    products = (
        (
            ("mu_x",),
            ("mu_z",),
            ("mu_x", "mu_x"),
            ("mu_x", "mu_z"),
            ("mu_z", "mu_z"),
            ("abs_mu_y",),
            ("qbar",),
            ("mu_x", "qbar"),
        )
        if output == "My"
        else (
            ("mu_y",),
            ("mu_x", "mu_y"),
            ("mu_z",),
            ("mu_y", "mu_z"),
            ("ur",),
            ("rbar",),
            ("local_yaw_linear",),
            ("local_yaw_axial",),
        )
    )
    feature = {
        f"speed_dev__{'_'.join(keys)}": scale * speed * np.prod([state[k] for k in keys], axis=0)
        for keys in products
    }
    # Common RPM-squared term permits a shared rotor-group imbalance. The
    # differential term corrects the fixed thrust/drag coefficient, not zeros.
    feature["rotor_common_scale"] = scale.copy()
    gain = "uq" if output == "My" else "ur"
    feature["rotor_differential_gain"] = scale * state[gain]
    return feature


def feature_dictionary(output, spec):
    """Physics/Sun variable families, no data-dependent prior term reuse."""
    if output == "Fz":
        # Total-flow pressure is nearly constant in a fixed-speed run. It must
        # remain in the anchored-mean model even when contrasts cannot select it.
        # state['inflow'] is total inflow lambda = nu - mu_z. Keep the
        # linear candidate; lambda**2 is an additional, freely selected term.
        # It is NOT (lambda-lambda_0)**2. Reference contrasts are formed only
        # after multiplying by the time-varying force scale Q_F.
        basic = ["prop_gain", "mu_h2", "mu_total2", "inflow_delta", "inflow^2"]
        extra = (
            []
            if spec == "compact"
            else [
                "mu_z",
                "mu_x*mu_z",
                "mu_y*mu_z",
                "mu_z^2",
                "abs_qbar",
                "abs_rbar",
                "abs_uq*mu_h2",
                "abs_ur*mu_h2",
                "mu_h2*mu_z",
                "inflow_delta*mu_h2",
            ]
        )
        return basic + [key for key in extra if key != "mu_z^2"]
    degree = 2 if spec == "compact" else 3
    if output == "My":
        steady = monomials(("mu_x", "abs_mu_y", "mu_z"), degree)[1:]
        terms = steady + [
            a + ((key, 1),)
            for a in monomials(("mu_x", "mu_z"), 1)
            for key in ("qbar", "uq")
            if a or key == "qbar"
        ]
        if spec == "expanded":
            terms += [((k, 1),) for k in ("abs_qbar",)] + [
                ((k, 1), ("qbar", 1)) for k in ("mu_x", "mu_z")
            ]
    else:
        steady = monomials(("mu_x", "mu_y", "mu_z"), degree)[1:]
        dynamic = [
            ((k, 1),)
            for k in (
                "rbar",
                "rbar_absrbar",
                "ur_absur",
                "local_yaw_linear",
                "local_yaw_quadratic",
                "local_yaw_axial",
            )
        ]
        interactions = [((k, 1), (b, 1)) for k in ("mu_x", "mu_y", "mu_z") for b in ("rbar", "ur")]
        terms = steady + dynamic + interactions
    terms = list(dict.fromkeys(terms))
    names = [
        "*".join(key + (f"^{power}" if power != 1 else "") for key, power in term) for term in terms
    ]
    if output == "Mz" and spec == "expanded":
        # Same 30/60-degree domains, fixed C2 transition, all branches retained.
        return [name + "*" + region for region in SUN_REGIONS for name in names]
    return names


def evaluate_features(state, physical, scale, output, spec, cfg):
    state = dict(state)
    if output == "Fz":
        state["inflow_delta"] = state["inflow"] - np.sqrt(
            cfg.kf / (cfg.rho * np.pi * cfg.rotor_radius_m**4) / 2
        )
        state["mu_total2"] = state["mu_x"] ** 2 + state["mu_y"] ** 2 + state["mu_z"] ** 2
    features = {}
    for name in feature_dictionary(output, spec):
        if name == "prop_gain":
            features[name] = physical.copy()
            continue
        value = scale.copy()
        for factor in name.split("*"):
            key, power = (factor.split("^") + [1])[:2] if "^" in factor else (factor, 1)
            value *= state[key] ** int(power)
        features[name] = value
    return features


def selected_banks(data, cfg, output, rules):
    j = OUTPUTS.index(output)
    scale = data["force_scale"] if j < 3 else data["moment_scale"]
    banks = {}
    for spec, names in rules.items():
        if output in EXTRA_CHANNELS:
            banks[spec] = extra_feature_bank(data, output, spec, cfg)
            continue
        regular = [
            k for k in names if k != "intercept" and not k.startswith(("rotor_", "speed_dev__", "theory__"))
        ]
        if output == "Fz":
            values = evaluate_features(
                data["state"], data["physical"][:, j], scale, output, spec, cfg
            )
        else:
            values = evaluate_names(data["state"], scale, regular)
            values.update(speed_features(data, output))
        if any(k.startswith("theory__") for k in names):
            values.update(theory_feature_bank(data, cfg, output))
        banks[spec] = {k: values[k] for k in names if k != "intercept"}
    return banks


def extra_feature_names(output, spec):
    """小さな物理量辞書。時刻・run番号・荷重・零点は説明変数にしない。"""
    if spec == "sun_inplane":
        # Sun 2019 Eq.29--30: one common-flow polynomial, without a second
        # per-rotor representation of the same in-plane drag effect.
        variables = {"Fx": ("mu_x", "abs_mu_y", "mu_z"),
                     "Fy": ("abs_mu_x", "mu_y", "mu_z")}[output]
        return ["*".join(k + (f"^{p}" if p != 1 else "") for k, p in term) or "1"
                for term in monomials(variables, 3)]
    theory = spec.startswith("theory_")
    spec = spec.removeprefix("theory_")
    variables, rates = {
        "Fx": (("mu_x", "abs_mu_y", "mu_z"), ("qbar", "rbar")),
        "Fy": (("abs_mu_x", "mu_y", "mu_z"), ("pbar", "rbar")),
        "Mx": (("abs_mu_x", "mu_y", "mu_z"), ("pbar", "up")),
    }[output]
    terms = monomials(variables, 2 if spec == "compact" else 3)
    terms += [a + ((rate, 1),) for a in monomials(variables, 1) for rate in rates]
    names = ["*".join(k + (f"^{p}" if p != 1 else "") for k, p in t) or "1" for t in terms]
    if output.startswith("F"):
        names += ["rotor_inplane_linear", "rotor_inplane_quadratic"]
    if theory:
        names += theory_feature_names(output)
    return list(dict.fromkeys(names))


def extra_feature_bank(data, output, spec, cfg=None):
    scale = data["force_scale"] if output.startswith("F") else data["moment_scale"]
    names = extra_feature_names(output, spec)
    bank = evaluate_names(data["state"], scale, [k for k in names if not k.startswith(("rotor_", "theory__"))])
    if output.startswith("F") and spec != "sun_inplane":
        # 各ローターの局所流 / (|Omega_i| R)。機体角速度の効果も含む。
        f = data["rotor_flow"]
        j = OUTPUTS.index(output)
        mu = f["mu"][:, :, j]
        weighted = -f["omega_squared_weight"] * mu
        bank["rotor_inplane_linear"] = scale * weighted.mean(axis=1)
        bank["rotor_inplane_quadratic"] = scale * (f["mu_h"] * weighted).mean(axis=1)
    if spec.startswith("theory_"):
        if cfg is None:
            raise ValueError("Theoretical candidates require explicit geometry/settings")
        bank.update(theory_feature_bank(data, cfg, output))
    return bank


def spline_weights(beta):
    knots = np.sin(np.deg2rad([0.0, 30.0, 60.0, 90.0])) ** 2
    coordinate = np.clip(np.sin(beta) ** 2, 0.0, 1.0)
    return CubicSpline(knots, np.eye(4), bc_type="natural")(coordinate)


def continuous_bank(data):
    """Shared degree<=3 flow model, rate/input terms, and smooth local changes.

    The omitted first spline basis is represented by the corresponding shared
    term: partition of unity prevents adding four duplicate constant effects.
    Front/back and left/right remain distinguishable through signed mu_x/y.
    """
    s = data["state"]
    scale = data["moment_scale"]
    bank = {}
    for i in range(4):
        for j in range(4 - i):
            for k in range(4 - i - j):
                name = f"flow_{i}{j}{k}"
                value = scale * s["mu_x"] ** i * s["mu_y"] ** j * s["mu_z"] ** k
                bank[name] = value
                if i + j + k <= 2:
                    bank[name + "_r"] = value * s["rbar"]
                    bank[name + "_u"] = value * s["ur"]
    for key in ("local_yaw_linear", "local_yaw_quadratic", "local_yaw_axial", "rbar_absrbar"):
        bank[key] = scale * s[key]
    speed = data["rms_speed_ratio"] - 1
    for key in ("mu_y", "ur", "rbar", "local_yaw_axial"):
        bank["speed_" + key] = scale * speed * s[key]
    weights = spline_weights(s["sun_beta"])
    high = 1 - s["sun_low"]
    for key in ("mu_y", "mu_x", "mu_z", "ur", "rbar", "local_yaw_linear", "local_yaw_axial"):
        for j in (1, 2, 3):
            bank[f"local{j}_{key}"] = scale * s[key] * high * weights[:, j]
    return bank


def reference_contrast(values, indices, reference):
    if not len(reference):
        raise ValueError("An explicit reference-index array is required")
    return values[indices] - values[reference].mean(axis=0)


def training_matrix(runs, route, bank, names, *, extra_ratio, balanced, noise):
    """Original loss unchanged; extra stage has 0.25 of its run's total weight.

    Extra stage duration cannot dominate a run simply by being longer. The
    original baseline's condition/repetition and RPM weights are reproduced.
    This is a working prediction loss, not independent Gaussian observations.
    """
    repeats = Counter(r["condition_id"] for r in runs)
    xx, yy = [], []
    for r in runs:
        d = r["route"][route]
        n = r["original_count"]
        x = np.column_stack([d[bank][k] for k in names])
        y = r["y"] - d["physical"]
        for sl, ratio in ((slice(0, n), 1.0), (slice(n, len(y)), extra_ratio)):
            count = len(y[sl])
            if not count or not ratio:
                continue
            weight = rpm_weights(d["rpm"][sl], balanced)
            factor = (
                np.sqrt(min(20.0, n / 5) * ratio * weight / (count * repeats[r["condition_id"]]))
                / noise
            )
            xx.append(x[sl] * factor[:, None])
            yy.append(y[sl] * factor)
    return np.vstack(xx), np.concatenate(yy)


def predict_yaw_delta(run, model, route):
    d = run["route"][route]
    result = d["physical"].copy()
    for key, value in model["coefficient_by_name"].items():
        result += value * d[model["bank"]][key]
    return result


def training_runs(runs, held_fold):
    """Preparation and original points travel together with their whole run."""
    return [r for r in runs if held_fold == "final" or r["fold"] != held_fold]


def physics_bank(data, cfg, signs):
    """Return dimensional moment regressors; empirical coefficients are shared."""
    flow = data["rotor_flow"]
    omega = flow["omega_abs_rad_s"]
    weight = cfg.rho * np.pi * cfg.rotor_radius_m**5 * omega**2
    axial = flow["lambda_axial"]
    horizontal = flow["mu_h"]
    result = {}
    # Permit repeatable motor/propeller imbalance without any run-specific gain.
    for i in range(4):
        result[f"rotor{i + 1}_torque"] = signs[i] * weight[:, i]
    for name, value in (
        ("axial", axial),
        ("axial2", axial**2),
        ("horizontal2", horizontal**2),
        ("horizontal_axial", horizontal * axial),
        ("horizontal3", horizontal**3),
    ):
        result["torque_" + name] = np.sum(signs * weight * value, axis=1)
    # Flow-direction terms use the FULL azimuth, including front/back. Dynamic
    # pressure scaling makes them vanish at zero flow; no direction discontinuity
    # remains in the moment at zero horizontal speed.
    s = data["state"]
    mx, my, mz = (s[k] for k in ("mu_x", "mu_y", "mu_z"))
    angle = np.arctan2(my, mx)
    pressure_scale = data["moment_scale"] * (mx**2 + my**2)
    for n in (1, 2, 3, 4):
        for trig, fun in (("sin", np.sin), ("cos", np.cos)):
            result[f"body_h{n}_{trig}"] = pressure_scale * fun(n * angle)
    for name, value in (("axial", mz), ("yaw_input", s["ur"]), ("yaw_rate", s["rbar"])):
        for n in (1, 2):
            result[f"body_h{n}_sin_{name}"] = pressure_scale * np.sin(n * angle) * value
            result[f"body_h{n}_cos_{name}"] = pressure_scale * np.cos(n * angle) * value
    return result


# ===========================================================================
# 6. 独立停止基準・共分散付き回帰
# ===========================================================================
def anchor_mean(anchors: list[dict], times: np.ndarray) -> np.ndarray | None:
    """No observations -> no absolute mean, not an arbitrary zero estimate."""
    if not anchors:
        return None
    anchors = sorted(anchors, key=lambda a: a["center_ns"])
    t = np.array([a["center_ns"] / 60e9 for a in anchors])
    b = np.array([a["bias_sensor_FLU"] for a in anchors])
    return np.column_stack([np.interp(times, t, b[:, j]) for j in range(6)])


def apply_zero_recovery(profile, recovery, profile_sha256):
    """Validated motor-off anchors extend support; old references stay fixed.

    New references are conditional on mechanics/contact assumptions and receive
    their own label. This does not turn model-predicted initial loads into truth.
    """
    if recovery.get('schema') != 'fftgyro-motion-zero-audit-v1':
        raise ValueError('Unknown zero recovery schema')
    if recovery['profile_sha256'] != profile_sha256:
        raise ValueError('Zero recovery used a different mechanics/input profile')
    old_keys={(x['epoch'],x['aircraft_boot_id']) for x in profile['anchors']}
    passed={tuple(x['key']) for x in recovery['epochs'] if x['accepted'] and not x['old_anchored']}
    anchors=recovery['accepted_anchors']
    for x in anchors:
        key=(x['epoch'],x['aircraft_boot_id'])
        if key in old_keys or key not in passed or x['wind'] != 0 or not x['window_consistent']:
            raise ValueError('Recovered reference is not an accepted new motor-off epoch')
    new_keys={(x['epoch'],x['aircraft_boot_id']) for x in anchors}
    changed=[]
    for r in profile['runs']:
        if tuple(r['key']) in new_keys:
            if r['anchored']:
                raise ValueError('Refusing to replace an existing zero')
            r['anchored']=True
            r['absolute_reference_kind']='motion_corrected_absolute'
            changed.append(r['run'])
    if set(changed) != set(recovery['newly_supported_runs']):
        raise ValueError('Recovered run mapping differs from the audited mapping')
    profile['anchors']=profile['anchors']+anchors
    return dict(new_runs=changed, new_epochs=len(new_keys),
                reference_kind='motion_corrected_absolute', coefficients_refitted=False,
                limitation='Conditional stopped-motion calibration; unobserved drift and shared mechanics error remain')


def interpolated_zero_covariance(anchor_minutes, run_minutes, observation_cov, slope_sigma):
    """Covariance of error about the SAME held linear zero used in reconstruction.

    Time is minutes. A zero-mean uncertain initial drift rate and integrated
    Wiener rate define K(t,s)=ts+min(t,s)^2 max(t,s)/2-min(t,s)^3/6.
    The covariance is (Q-H A) K (Q-H A)' + H R H'. This is an error
    sensitivity model, not a confidence guarantee or a fit to powered loads.
    Axes of the drift prior are independent; observed six-axis covariance is
    retained. Returns [run, sensor_axis, run, sensor_axis].
    """
    at, qt = np.asarray(anchor_minutes, float), np.asarray(run_minutes, float)
    sigma = np.asarray(slope_sigma, float)
    if len(at) < 2 or np.any(np.diff(at) <= 0) or sigma.shape != (6,) or np.any(sigma < 0):
        raise ValueError('Invalid recovered zero covariance inputs')
    obs = np.asarray(observation_cov, float)
    if obs.shape != (len(at), 6, 6) or not np.isfinite(obs).all():
        raise ValueError('Expected one finite six-axis covariance per anchor')
    h = np.column_stack([np.interp(qt, at, col) for col in np.eye(len(at))])
    origin = min(at.min(), qt.min())
    def kernel(x, y):
        x, y = x[:, None]-origin, y[None, :]-origin
        lo, hi = np.minimum(x, y), np.maximum(x, y)
        return x*y + lo**2*hi/2 - lo**3/6
    k = kernel(qt, qt)-h@kernel(at, qt)-kernel(qt, at)@h.T+h@kernel(at, at)@h.T
    k = (k+k.T)/2
    cov = np.einsum('ij,ab->iajb', k, np.diag(sigma**2))
    cov += np.einsum('ik,jk,kab->iajb', h, h, obs)
    return cov


def extend_recovered_calibration(profile, recovery, prior):
    """Keep the original covariance block; add independent new power epochs.

    One-second blocks estimate anchor-mean noise. Cross-run motor-off residual
    RMSE adds a shared epoch-level working floor at the CG; it is not reduced
    by collecting more powered samples. Geometry is frozen, not identified.
    Only GLS / zero_prior_weighted fits use this covariance; other losses keep
    their original weights. Never update the zero mean from model residuals.
    """
    cal = profile['calibration']
    old_ids = list(cal['ids'])
    added = [r for r in profile['runs'] if r.get('absolute_reference_kind') == 'motion_corrected_absolute']
    new_ids = [r['run'] for r in added]
    if set(old_ids) & set(new_ids) or set(new_ids) != set(recovery['newly_supported_runs']):
        raise ValueError('Recovered calibration IDs overlap or differ from audit')
    cfg, _ = split_calibration(Settings(**profile['settings']), sensor_parameters(profile['sensor_summary']))
    cg = np.array(cfg.cg_flu_m)*FLU_TO_FRD
    transform = shift_wrench(np.diag(np.tile(FLU_TO_FRD, 2)), cg).T
    nold, ntotal = len(old_ids), len(old_ids)+len(added)
    extended = {}
    for output in ('My', 'Mz', 'Fz'):
        matrix = np.zeros((ntotal, ntotal))
        matrix[:nold, :nold] = cal['covariance'][output]
        extended[output] = matrix
    floor = np.array([.02, .02, .01, .0002, .0002, .0002])
    epochs = []
    for epoch in recovery['epochs']:
        if not epoch['accepted'] or epoch['old_anchored']:
            continue
        key = tuple(epoch['key'])
        aa = sorted([a for a in recovery['accepted_anchors']
                     if (a['epoch'], a['aircraft_boot_id']) == key], key=lambda a:a['center_ns'])
        inds = [i for i, r in enumerate(added) if tuple(r['key']) == key]
        if not inds:
            continue
        observations = []
        for a in aa:
            blocks = np.asarray(a['block_bias_sensor_FLU'])
            if len(blocks) < 2:
                raise ValueError('Recovered anchor lacks independent time blocks')
            observations.append(np.cov(blocks.T)/len(blocks)+np.diag(floor**2))
        native = interpolated_zero_covariance(
            [a['center_ns']/60e9 for a in aa], [added[i]['center'] for i in inds],
            observations, prior['slope_prior_sigma'])
        target = np.ix_(np.array(inds)+nold, np.array(inds)+nold)
        for output, axis in (('Fz', 2), ('My', 4), ('Mz', 5)):
            mapped = np.einsum('a,iajb,b->ij', transform[axis], native, transform[axis])
            mapped += epoch['leave_run_out_rmse'][axis]**2
            extended[output][target] = (mapped+mapped.T)/2
        epochs.append(dict(key=list(key), runs=len(inds), anchors=len(aa),
                           shared_CG_floor=epoch['leave_run_out_rmse']))
    cal['ids'] = old_ids+new_ids
    cal['covariance'] = extended
    return dict(old_runs=nold, new_runs=len(added), epochs=epochs,
                zero_mean_updated=False, geometry_frozen=True,
                method='Held-linear interpolation error + integrated drift-rate prior + shared motor-off epoch floor',
                limitation='Working covariance only; unobserved drift and common geometry errors are not independently validated')


def extend_foundation_training(profile):
    """Add recovered absolute targets, excluding the entire validation fold."""
    audit = {}
    for key, model in profile['models']['foundation'].items():
        fold, route, output = key.split('/')
        if output == 'Fz':
            continue  # Fz already uses all eligible runs, unanchored ones as contrasts.
        additions = [r['run'] for r in profile['runs']
                     if r.get('absolute_reference_kind') == 'motion_corrected_absolute'
                     and output in r['channels'] and r['channels'][output]['reference_available']
                     and (fold == 'final' or str(r['fold']) != fold)]
        old = set(model['training_runs'])
        model['training_runs'] = sorted(old | set(additions))
        audit[key] = dict(previous=len(old), added=len(set(additions)-old), total=len(model['training_runs']))
    return audit


@dataclass(frozen=True)
class RunBlock:
    run_id: str
    role: str
    features: Mapping[str, np.ndarray]
    residual: np.ndarray
    anchored: bool
    noise_sd: float
    effective_samples: float | None = None


@dataclass(frozen=True)
class RegressionBlocks:
    x: np.ndarray
    y: np.ndarray
    names: tuple[str, ...]
    audit: dict
    column_norms: np.ndarray | None = None


def _contrasts(values: np.ndarray) -> np.ndarray:
    """Householder contrasts preserve centered SSE, without an N-by-N matrix."""
    v = np.ones(len(values)) / np.sqrt(len(values))
    v[0] -= 1
    return (values - np.outer(v, 2 * (v @ values) / (v @ v)))[1:]


def assemble_blocks(
    runs: Sequence[RunBlock],
    feature_names: Sequence[str],
    *,
    mean_covariance: np.ndarray | None = None,
    covariance_run_ids: Sequence[str] | None = None,
    training_roles: Sequence[str] = ("T",),
    train_run_ids: Sequence[str] | None = None,
    within_correlation: float = 1.0,
    include_intercept: bool = True,
    model_discrepancy_sd: float = 0.0,
) -> RegressionBlocks:
    """Build contrasts and anchored means, using explicitly selected roles.

    ``mean_covariance`` is zero-error covariance between run means, in load
    units squared; observational mean variance is added separately. Its IDs
    may include held-out runs, but only the selected training submatrix enters
    the regression. Without explicit IDs, it follows anchored training order.
    Explicit ``train_run_ids`` replaces historical role selection, allowing
    grouped cross-validation to train across roles without held-out leakage.

    ``within_correlation`` approximates reduced effective independent samples:
    n_eff=n/within_correlation unless overridden on a run. It is a weighting
    approximation, not a claim that contrasts remove serial correlation.
    ``model_discrepancy_sd`` is an independent run-mean model-error floor, not
    sensor-zero drift. It changes anchored-mean covariance only, never contrasts
    or absolute support. Its value must be specified from training data only.
    """
    names = tuple(feature_names)
    if len(names) != len(set(names)) or "intercept" in names:
        raise ValueError("Feature names must be unique and exclude 'intercept'")
    if not np.isfinite(within_correlation) or within_correlation < 1:
        raise ValueError("within_correlation must be finite and >= 1")
    if not np.isfinite(model_discrepancy_sd) or model_discrepancy_sd < 0:
        raise ValueError("model_discrepancy_sd must be finite and nonnegative")
    if train_run_ids is None:
        selected = [r for r in runs if r.role in training_roles]
    else:
        requested_ids = set(train_run_ids)
        if requested_ids - {r.run_id for r in runs}:
            raise ValueError("Explicit training IDs include absent runs")
        selected = [r for r in runs if r.run_id in requested_ids]
    if not selected or len({r.run_id for r in selected}) != len(selected):
        raise ValueError("Need nonempty, uniquely identified training runs")
    anchors = [r.run_id for r in selected if r.anchored]
    has_intercept = bool(include_intercept and anchors)
    columns = (("intercept",) if has_intercept else ()) + names
    contrast_x, contrast_y, mean_x, mean_y, mean_variance = [], [], [], [], []
    contributions = []
    for run in selected:
        y = np.asarray(run.residual, dtype=float)
        if y.ndim != 1 or len(y) < 2 or not np.isfinite(y).all():
            raise ValueError(f"{run.run_id}: residual requires >= 2 finite samples")
        xx = [np.asarray(run.features[name], dtype=float) for name in names]
        if any(v.shape != y.shape or not np.isfinite(v).all() for v in xx):
            raise ValueError(f"{run.run_id}: features must match finite residual samples")
        if has_intercept:
            xx.insert(0, np.ones(len(y)))
        x = np.column_stack(xx) if xx else np.empty((len(y), 0))
        sigma = float(run.noise_sd)
        effective = float(
            run.effective_samples
            if run.effective_samples is not None
            else len(y) / within_correlation
        )
        if not np.isfinite(sigma) or sigma <= 0:
            raise ValueError(f"{run.run_id}: noise_sd must be positive")
        if not np.isfinite(effective) or not 0 < effective <= len(y):
            raise ValueError(f"{run.run_id}: effective_samples must be in (0, n]")
        joint = _contrasts(np.column_stack([y, x]))
        # Exact known constant columns have no contrast information.
        constant = np.ptp(x, axis=0) == 0
        joint[:, 1:][:, constant] = 0
        weight = np.sqrt(effective / len(y)) / sigma
        contrast_x.append(joint[:, 1:] * weight)
        contrast_y.append(joint[:, 0] * weight)
        if run.anchored:
            mean_x.append(x.mean(axis=0))
            mean_y.append(float(y.mean()))
            mean_variance.append(sigma**2 / effective + model_discrepancy_sd**2)
        contributions.append(
            dict(
                run=run.run_id,
                role=run.role,
                anchored=bool(run.anchored),
                samples=len(y),
                effective_samples=effective,
                contrast_rows=len(y) - 1,
                noise_sd=sigma,
            )
        )
    mean_condition = None
    if anchors:
        if mean_covariance is None:
            if covariance_run_ids is not None:
                raise ValueError("covariance_run_ids requires mean_covariance")
            cov = np.zeros((len(anchors), len(anchors)))
        else:
            ids = list(covariance_run_ids) if covariance_run_ids is not None else anchors
            matrix = np.asarray(mean_covariance, dtype=float)
            if len(ids) != len(set(ids)) or matrix.shape != (len(ids), len(ids)):
                raise ValueError("Mean covariance shape or run IDs mismatch")
            indices = [ids.index(run_id) for run_id in anchors]
            cov = matrix[np.ix_(indices, indices)]
            if not np.isfinite(cov).all() or not np.allclose(cov, cov.T, rtol=1e-10, atol=1e-14):
                raise ValueError("Training mean covariance must be finite and symmetric")
            eigenvalues = np.linalg.eigvalsh(cov)
            tolerance = max(float(np.max(np.abs(eigenvalues))), np.finfo(float).tiny) * 1e-10
            if eigenvalues[0] < -tolerance:
                raise ValueError("Training mean covariance must be positive semidefinite")
        total_cov = cov + np.diag(mean_variance)
        chol = np.linalg.cholesky(total_cov)
        contrast_x.append(solve_triangular(chol, np.asarray(mean_x), lower=True))
        contrast_y.append(solve_triangular(chol, np.asarray(mean_y), lower=True))
        mean_condition = float(np.linalg.cond(total_cov))
    audit = dict(
        training_roles=sorted({r.role for r in selected}),
        training_selection="explicit_run_ids" if train_run_ids is not None else "historical_roles",
        training_runs=[r.run_id for r in selected],
        absolute_training_runs=anchors,
        contrast_only_training_runs=[r.run_id for r in selected if not r.anchored],
        absolute_prediction_supported=bool(anchors),
        intercept_supported=has_intercept,
        contributions=contributions,
        mean_rows=len(anchors),
        contrast_rows=sum(len(y) for y in contrast_y[: len(selected)]),
        mean_covariance_condition_number=mean_condition,
        within_correlation=float(within_correlation),
        model_discrepancy_sd=float(model_discrepancy_sd),
        model_discrepancy_policy="Independent anchored-run-mean model error, not a zero update",
    )
    return RegressionBlocks(np.vstack(contrast_x), np.concatenate(contrast_y), columns, audit)


def subset_blocks(blocks: RegressionBlocks, names: Sequence[str]) -> RegressionBlocks:
    """Reuse the identical transformed rows for a selected candidate subset."""
    selected = tuple(names)
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("Selected columns must be nonempty and unique")
    indices = [blocks.names.index(name) for name in selected]
    audit = dict(blocks.audit, intercept_supported="intercept" in selected)
    # Freeze parent norms before advanced indexing changes array memory layout.
    parent_norms = (
        np.linalg.norm(blocks.x, axis=0) if blocks.column_norms is None else blocks.column_norms
    )
    norms = parent_norms[indices]
    return RegressionBlocks(blocks.x[:, indices], blocks.y, selected, audit, norms)


def compress_regression_blocks(blocks: RegressionBlocks) -> RegressionBlocks:
    """Compress a selected design by QR without changing any least-square loss.

    QR of the AUGMENTED [X,y] retains X'X, X'y and y'y, including target energy
    orthogonal to X. A design-only QR with Q'y would lose that residual energy.
    Precompression column scales and original sample count stay frozen. Use
    this after term selection; its shorter row count is not a new Sun/PSE N.
    """
    x, y = np.asarray(blocks.x, float), np.asarray(blocks.y, float)
    if x.ndim != 2 or y.shape != (len(x),) or x.shape[1] != len(blocks.names):
        raise ValueError("Regression shape mismatch")
    if not len(y) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Regression blocks must be nonempty and finite")
    norms = (
        np.linalg.norm(x, axis=0)
        if blocks.column_norms is None
        else np.asarray(blocks.column_norms).copy()
    )
    reduced = np.linalg.qr(np.column_stack([x, y]), mode="r")
    audit = dict(
        blocks.audit,
        regression_compression="augmented_QR_loss_preserving",
        regression_original_rows=blocks.audit.get("regression_original_rows", len(y)),
        regression_compressed_rows=len(reduced),
    )
    return RegressionBlocks(reduced[:, :-1], reduced[:, -1], blocks.names, audit, norms)


def fit_regularized(
    blocks: RegressionBlocks,
    *,
    ridge: float = 0.0,
    bounds: Mapping[str, tuple[float, float]] | None = None,
    ridge_exempt: Sequence[str] = ("intercept",),
) -> dict:
    """Fit normalized training columns; bounds apply to original coefficients.

    The objective is ||X beta-y||² + ridge*||D beta_scaled||², with column
    L2 norms frozen from these training blocks. Zero-norm columns are removed;
    numerical rank and singular values describe the unregularized design.
    """
    x, y = np.asarray(blocks.x, float), np.asarray(blocks.y, float)
    if x.ndim != 2 or y.shape != (len(x),) or x.shape[1] != len(blocks.names):
        raise ValueError("Regression shape mismatch")
    if not np.isfinite(x).all() or not np.isfinite(y).all() or len(y) == 0:
        raise ValueError("Regression blocks must be nonempty and finite")
    if not np.isfinite(ridge) or ridge < 0:
        raise ValueError("ridge must be finite and nonnegative")
    requested = dict(bounds or {})
    if set(requested) - set(blocks.names):
        raise ValueError("Bounds reference unknown columns")
    scales = (
        np.linalg.norm(x, axis=0)
        if blocks.column_norms is None
        else np.asarray(blocks.column_norms, float).copy()
    )
    if scales.shape != (x.shape[1],) or not np.isfinite(scales).all() or (scales < 0).any():
        raise ValueError("Frozen column norms must be finite, nonnegative and match columns")
    take = scales > 0
    names = [name for name, keep in zip(blocks.names, take) if keep]
    dropped = [name for name, keep in zip(blocks.names, take) if not keep]
    for name in dropped:
        lo, hi = requested.get(name, (-np.inf, np.inf))
        if not lo <= 0 <= hi:
            raise ValueError(f"{name}: zero-information coefficient cannot satisfy bounds")
    if not names:
        raise ValueError("No informative regression columns")
    scales = scales[take]
    normalized = x[:, take] / scales
    singular = np.linalg.svd(normalized, compute_uv=False)
    original_rows = int(blocks.audit.get("regression_original_rows", len(y)))
    tolerance = singular[0] * max(original_rows, len(names)) * np.finfo(float).eps
    rank = int(np.sum(singular > tolerance))
    penalty = np.asarray([0.0 if name in ridge_exempt else 1.0 for name in names])
    if ridge:
        design = np.vstack([normalized, np.sqrt(ridge) * np.diag(penalty)])
        target = np.concatenate([y, np.zeros(len(names))])
    else:
        design, target = normalized, y
    lower, upper = [], []
    for name, scale in zip(names, scales):
        lo, hi = requested.get(name, (-np.inf, np.inf))
        if np.isnan(lo) or np.isnan(hi) or lo >= hi:
            raise ValueError(f"{name}: bounds require lower < upper")
        lower.append(lo * scale)
        upper.append(hi * scale)
    if np.isneginf(lower).all() and np.isposinf(upper).all():
        # Preserve the original design's SVD cutoff after row compression.
        rcond = np.finfo(float).eps * max(original_rows + (len(names) if ridge else 0), len(names))
        coefficient = np.linalg.lstsq(design, target, rcond=rcond)[0]
        solver = dict(method="numpy_lstsq", success=True)
    else:
        solution = lsq_linear(
            design, target, bounds=(lower, upper), tol=1e-10, lsmr_tol="auto", max_iter=1000
        )
        if not solution.success:
            raise RuntimeError(f"Bounded regression failed: {solution.message}")
        coefficient = solution.x
        solver = dict(
            method="scipy_lsq_linear",
            success=True,
            iterations=int(solution.nit),
            optimality=float(solution.optimality),
            active_mask=solution.active_mask.tolist(),
        )
    residual = normalized @ coefficient - y
    raw_coefficient = coefficient / scales
    return dict(
        feature_names=names,
        coefficients=raw_coefficient.tolist(),
        coefficient_by_name=dict(zip(names, raw_coefficient.tolist())),
        column_scales=scales.tolist(),
        scaled_coefficients=coefficient.tolist(),
        dropped_zero_norm_columns=dropped,
        ridge=float(ridge),
        ridge_exempt=list(ridge_exempt),
        numerical_rank=rank,
        normalized_singular_values=singular.tolist(),
        normalized_condition_number=(
            float(singular[0] / singular[-1]) if rank == len(names) else None
        ),
        weighted_residual_sse=float(residual @ residual),
        bounds={
            name: [float(v) if np.isfinite(v) else None for v in requested[name]]
            for name in names
            if name in requested
        },
        solver=solver,
        **blocks.audit,
    )


def predict_correction(model: Mapping, features: Mapping[str, np.ndarray]) -> np.ndarray:
    """Predict in original units with frozen training coefficients/scales."""
    names = list(model["feature_names"])
    size = len(next(iter(features.values()))) if features else 1
    if not names:
        return np.zeros(size)
    columns = [
        np.ones(size) if name == "intercept" else np.asarray(features[name], float)
        for name in names
    ]
    if any(column.shape != (size,) for column in columns):
        raise ValueError("Prediction features must be equally sized one-dimensional arrays")
    # Using frozen scales also makes serialized prediction reproducible.
    return (np.column_stack(columns) / np.asarray(model["column_scales"])) @ np.asarray(
        model["scaled_coefficients"]
    )


def select_extra_terms(matrix, *, first_candidate=None):
    """既存Sun公開版のF入>5/F出<4/PSE 1%規則を、QRで短縮して適用。

    重み付き行数をF/PSEのNとする作業上の選択。独立標本数とは呼ばない。
    元の定数ベクトルもQRへ入れるため、圧縮後もPearson相関が変わらない。
    候補辞書・条件等重み・基準差分は本研究の設定であり、Sun原実験とは別。
    """
    x, y = matrix.x, matrix.y
    if first_candidate is not None and first_candidate not in matrix.names:
        raise ValueError("First candidate missing from dictionary")
    n, count = x.shape
    if n < 3 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Term selection needs >=3 finite training rows")
    scales = np.linalg.norm(x, axis=0)
    z = np.linalg.qr(np.column_stack([x / np.maximum(scales, 1e-30), y, np.ones(n)]), mode="r")
    xq, target, constant = z[:, :count], z[:, count], z[:, count + 1]
    pool = scales > 1e-25
    required = [i for i, name in enumerate(matrix.names) if name == "intercept"]
    active, history = required.copy(), []
    pool[active] = False
    variance = float(np.var(y))

    def centered(v):
        return (
            v - np.outer(constant, constant @ v / n)
            if v.ndim == 2
            else v - constant * (constant @ v / n)
        )

    def solve(ids):
        a = xq[:, ids]
        norms = np.linalg.norm(a, axis=0)
        if np.any(norms <= 1e-25):
            raise ValueError("Zero regressor")
        if ids:
            coef, _, rank, singular = np.linalg.lstsq(a / norms, target, rcond=1e-10)
            if rank != len(ids) or singular[0] / singular[-1] > 1e6:
                raise ValueError("Unidentifiable regressors")
            prediction = (a / norms) @ coef
        else:
            prediction = np.zeros_like(target)
        sse = float(np.sum((target - prediction) ** 2))
        return prediction, sse, sse / n + variance * len(ids) / n

    prediction, sse, previous = solve(active)
    initial, last_out = previous, None
    first_test_pending = first_candidate is not None
    reason = "candidate_pool_exhausted"
    while pool.any():
        if n - len(active) - 1 <= 0:
            reason = "no_residual_degrees_of_freedom"
            break
        q = np.linalg.qr(xq[:, active], mode="reduced")[0] if active else np.empty((len(z), 0))
        residual = centered(target - prediction)
        rn = np.linalg.norm(residual)
        if rn <= 32 * np.finfo(float).eps * max(np.linalg.norm(target), 1e-30):
            reason = "zero_residual_variance"
            break
        candidates = np.flatnonzero(pool)
        v = centered(xq[:, candidates] - q @ (q.T @ xq[:, candidates]))
        norms = np.linalg.norm(v, axis=0)
        correlations = np.where(
            norms > 1e-10, np.abs(v.T @ residual) / (np.maximum(norms, 1e-30) * rn), -np.inf
        )
        if not np.isfinite(correlations).any():
            reason = "no_independent_correlated_candidates"
            break
        j = int(candidates[np.argmax(correlations)])
        # Sun Eq.29--30 tests mu_x / mu_y first. F/PSE and backward deletion
        # still apply: this is not a mandatory retained regressor.
        if first_test_pending:
            first_test_pending = False
            first = matrix.names.index(first_candidate)
            positions = np.flatnonzero(candidates == first)
            if len(positions) and np.isfinite(correlations[positions[0]]):
                j = first
        before = active.copy()
        try:
            newprediction, newsse, _ = solve(active + [j])
        except ValueError:
            pool[j] = False
            history.append(dict(action="numerical_rejection", column=j))
            continue
        df = n - len(active) - 1
        fin = max(0.0, sse - newsse) / max(newsse / df, np.finfo(float).tiny)
        if fin <= 5:
            reason = "forward_F_not_above_5"
            break
        pool[j] = False
        if last_out is not None and np.array_equal(x[:, j], x[:, last_out]):
            reason = "repeated_eliminated_column"
            break
        active, prediction, sse = active + [j], newprediction, newsse
        backward = [
            float("inf")
            if k in required
            else max(0.0, solve([i for i in active if i != k])[1] - sse)
            / max(sse / df, np.finfo(float).tiny)
            for k in active
        ]
        weakest = int(np.argmin(backward))
        last_out = active[weakest] if backward[weakest] < 4 else None
        if last_out is not None:
            active.remove(last_out)
        prediction, sse, pse = solve(active)
        accepted = pse < 0.99 * previous
        history.append(dict(added=j, removed=last_out, F_in=float(fin), pse=pse, accepted=accepted))
        if not accepted:
            active = before
            reason = "PSE_not_1_percent_better_rollback"
            break
        previous = pse
        if pse < 0.001 * initial:
            reason = "PSE_below_0_001_initial"
            break
    return tuple(matrix.names[i] for i in active), dict(
        source="https://github.com/SihaoSun/stepwise_regression/tree/b04095ccb9f0e5dcafae43db6ec1a4320cbd0b16/func",
        candidate_names=list(matrix.names),
        history=history,
        stop_reason=reason,
        training_rows=n,
        weighting="condition_equal; weighted rows, not independent samples",
        F_in=5,
        F_out=4,
        PSE_ratio=0.99,
        term_cap=None,
        first_candidate_tested=first_candidate,
    )


def extra_matrix(runs, route, output, spec, family):
    """独立基準のあるrunだけで絶対基礎、全runの基準差分で変化を学習。"""
    rr = [
        r
        for r in runs
        if r["channels"][output]["reference_available"] and (family == "increment" or r["anchored"])
    ]
    if not rr:
        raise ValueError("No training support for " + family)
    names = tuple(extra_feature_names(output, spec))
    if family == "foundation":
        names = ("intercept",) + names
    repeats = Counter(r["condition_id"] for r in rr)
    xx, yy = [], []
    for r in rr:
        ch = r["channels"][output]
        d, take = ch[route], ch["take"]
        n = int(take.sum())
        x = np.column_stack(
            [np.ones(n) if k == "intercept" else d["features"][spec][k][take] for k in names]
        )
        y = (ch["truth"] - d["physical"])[take]
        if family == "increment":
            x -= np.array([d["reference_features"][spec][k] for k in names])
            y -= ch["reference"] - d["reference_physical"]
        weight = 1 / np.sqrt(n * repeats[r["condition_id"]])
        xx.append(x * weight)
        yy.append(y * weight)
    return RegressionBlocks(
        np.vstack(xx),
        np.concatenate(yy),
        names,
        dict(
            training_runs=[r["run"] for r in rr],
            loss="condition_equal_" + family,
            absolute_prediction_supported=family == "foundation",
            zero_updated=False,
        ),
    )


def extra_predictions(run, output, route, foundation, increment):
    ch = run["channels"][output]
    d, take = ch[route], ch["take"]
    initial = reference_prediction(d, foundation)
    predicted = (
        d["physical"][take] + predict_correction(increment, d["features"][increment["spec"]])[take]
    )
    delta = predicted - reference_prediction(d, increment)
    return initial + delta, delta


def fit_extra_channels(runs, *, reselect_physics=False):
    """追加3軸だけ新規選択。外側5条件foldの内側3条件foldで辞書/ridgeを選ぶ。"""
    models = {}
    folds = ["final", *sorted({r["fold"] for r in runs})]
    ridges = (0.0, 0.01, 0.1, 1.0)

    def candidates(train, route, output, spec, family):
        matrix = extra_matrix(train, route, output, spec, family)
        names, selection = select_extra_terms(matrix)
        small = compress_regression_blocks(subset_blocks(matrix, names)) if names else matrix
        for ridge in ridges:
            model = (
                fit_regularized(small, ridge=ridge)
                if names
                else dict(
                    feature_names=[],
                    coefficients=[],
                    coefficient_by_name={},
                    column_scales=[],
                    scaled_coefficients=[],
                    numerical_rank=0,
                    normalized_condition_number=None,
                    ridge=ridge,
                    **small.audit,
                )
            )
            model.update(spec=spec, output=output, selection=selection)
            yield model

    for output in EXTRA_CHANNELS:
        for route in ROUTES:
            for fold in folds:
                train = [
                    r
                    for r in training_runs(runs, fold)
                    if r["channels"][output]["reference_available"]
                ]
                conditions = sorted({r["condition_id"] for r in train})
                inner = {c: i % 3 for i, c in enumerate(conditions)}
                for family in ("foundation", "increment"):
                    scores = defaultdict(list)
                    for split in range(3):
                        tt = [r for r in train if inner[r["condition_id"]] != split]
                        vv = [
                            r
                            for r in train
                            if inner[r["condition_id"]] == split
                            and (family == "increment" or r["anchored"])
                        ]
                        if not vv:
                            raise ValueError("Inner validation lacks absolute/contrast support")
                        specs = ("theory_compact", "theory_expanded") if reselect_physics else ("compact", "expanded")
                        for spec in specs:
                            for model in candidates(tt, route, output, spec, family):
                                by_condition = defaultdict(list)
                                for r in vv:
                                    ch, d = r["channels"][output], r["channels"][output][route]
                                    refpred = reference_prediction(d, model)
                                    if family == "foundation":
                                        error = np.array([refpred - ch["reference"]])
                                    else:
                                        prediction = d["physical"] + predict_correction(
                                            model, d["features"][spec]
                                        )
                                        error = (
                                            prediction - refpred - ch["truth"] + ch["reference"]
                                        )[ch["take"]]
                                    by_condition[r["condition_id"]].append(float(np.mean(error**2)))
                                scores[spec, model["ridge"]].extend(
                                    float(np.mean(v)) for v in by_condition.values()
                                )
                    spec, ridge = min(scores, key=lambda k: (float(np.mean(scores[k])), k))
                    model = next(
                        m
                        for m in candidates(train, route, output, spec, family)
                        if m["ridge"] == ridge
                    )
                    model["inner_cv"] = [
                        dict(
                            spec=s,
                            ridge=l,
                            condition_equal_mse=float(np.mean(v)),
                            conditions=len(v),
                        )
                        for (s, l), v in scores.items()
                    ]
                    model["selection_target"] = (
                        "initial_absolute_reference"
                        if family == "foundation"
                        else "reference_increment"
                    )
                    key = f"{fold}/{route}/{output}/{family}"
                    models[key] = model
                    print("EXTRA FIT", key, spec, ridge, len(model["feature_names"]), flush=True)
    return models


def evaluate_extra_channels(runs, models, *, outputs=None):
    outputs = EXTRA_CHANNELS if outputs is None else tuple(outputs)
    if not outputs or not set(outputs) <= set(EXTRA_CHANNELS):
        raise ValueError("Extra-channel evaluation needs Fx, Fy and/or Mx")
    rows, arrays = [], {}
    for r in runs:
        for output in outputs:
            ch = r["channels"][output]
            if not ch["reference_available"]:
                continue
            take, name = ch["take"], r["run"]
            for mode, fold in (("all_data_in_sample", "final"), ("grouped_OOF", r["fold"])):
                for route in ROUTES:
                    key = f"{fold}/{route}/{output}/"
                    foundation, increment = models[key + "foundation"], models[key + "increment"]
                    for m in (foundation, increment):
                        check_fold(m, r, mode, runs)
                    prediction, delta = extra_predictions(r, output, route, foundation, increment)
                    prefix = f"{mode}/{output}/{route}/{name}/"
                    arrays[prefix + "absolute_prediction"] = prediction
                    variants = [
                        (
                            "increment",
                            ch["truth"][take] - ch["reference"],
                            delta,
                            ch["physical"][take] - ch["physical_reference"],
                        )
                    ]
                    if r["anchored"]:
                        variants.append(
                            (
                                r.get("absolute_reference_kind", "independent_absolute"),
                                ch["truth"][take],
                                prediction,
                                ch["physical"][take],
                            )
                        )
                    for reference, truth, predicted, physical in variants:
                        rows.append(
                            dict(
                                run=name,
                                mode=mode,
                                output=output,
                                route=route,
                                reference=reference,
                                wind=r["wind"],
                                condition_id=r["condition_id"],
                                physical=error_metrics(truth, physical),
                                identified=error_metrics(truth, predicted),
                            )
                        )
                        arrays.update(
                            {
                                prefix + reference + "/" + k: v
                                for k, v in dict(
                                    time=r["time"][take],
                                    truth=truth,
                                    prediction=predicted,
                                    physical=physical,
                                ).items()
                            }
                        )
    return rows, arrays


def frozen_fit(runs, cal, route, output, model, cfg, *, matrix_only=False):
    """Same selected columns, training IDs, loss weights and ridge as before."""
    train = set(model["training_runs"])
    rr = [r for r in runs if r["run"] in train]
    if len(rr) != len(train):
        raise ValueError("Training run missing")
    names = tuple(model["feature_names"])
    spec = model["spec"]
    candidate = model.get("predictive_candidate", dict(loss="gls", balanced=False))
    repeats = Counter(r["condition_id"] for r in rr)
    if candidate["loss"] == "gls":
        blocks = []
        features = tuple(k for k in names if k != "intercept")
        for r in rr:
            ch = r["channels"][output]
            take = ch["take"]
            d = ch[route]
            blocks.append(
                RunBlock(
                    r["run"],
                    r["role"],
                    {k: d["features"][spec][k][take] for k in features},
                    (ch["truth"] - d["physical"])[take],
                    r["anchored"],
                    cal["noise"][output] * np.sqrt(repeats[r["condition_id"]]),
                    effective_samples=min(20.0, float(take.sum()) / 5),
                )
            )
        matrix = assemble_blocks(
            blocks,
            features,
            mean_covariance=cal["covariance"][output],
            covariance_run_ids=cal["ids"],
            train_run_ids=list(train),
            include_intercept="intercept" in names,
            model_discrepancy_sd=model.get("model_discrepancy_sd", 0.0),
        )
        matrix = subset_blocks(matrix, names)
    else:
        xx, yy = [], []
        for r in rr:
            ch = r["channels"][output]
            d = ch[route]
            take = ch["take"]
            x = np.column_stack([d["features"][spec][k][take] for k in names])
            residual = (ch["truth"] - d["physical"])[take]
            if ch["reference_available"]:
                x = x - np.array([d["reference_features"][spec][k] for k in names])
                residual = residual - (ch["reference"] - d["reference_physical"])
            else:
                x = x - x.mean(axis=0)
                residual = residual - residual.mean()
            fz = r["channels"]["Fz"][route]["physical"][take]
            rpm = np.sqrt(np.maximum(-fz / (4 * cfg.kf), 0)) * 60 / (2 * np.pi)
            weight = rpm_weights(rpm, candidate["balanced"])
            weight = (
                np.sqrt(
                    min(20.0, len(residual) / 5)
                    * weight
                    / (len(residual) * repeats[r["condition_id"]])
                )
                / cal["noise"][output]
            )
            xx.append(x * weight[:, None])
            yy.append(residual * weight)
        matrix = RegressionBlocks(
            np.vstack(xx),
            np.concatenate(yy),
            names,
            dict(training_runs=[r["run"] for r in rr], loss="reference_delta"),
        )
    if matrix_only:
        return matrix
    bounds = {
        k: tuple(
            -np.inf if x is None and i == 0 else np.inf if x is None else x for i, x in enumerate(v)
        )
        for k, v in model.get("bounds", {}).items()
    }
    result = fit_regularized(
        compress_regression_blocks(matrix), ridge=model["ridge"], bounds=bounds
    )
    result.update(
        output=output,
        spec=spec,
        training_runs=sorted(train),
        frozen_structure=True,
        predictive_candidate=candidate,
        model_discrepancy_sd=model.get("model_discrepancy_sd", 0.0),
    )
    return result


def assemble_foundation(runs, cal, route, output, template):
    """Absolute point loss, optionally mixed with the deployed initial target.

    reference_fraction=0 preserves the original loss. At 1, each training run
    contributes its observed initial load and own-route initial features only.
    Reference loads are training targets, NEVER prediction inputs.
    """
    ids = set(template["training_runs"])
    rr = [r for r in runs if r["run"] in ids]
    if len(rr) != len(ids) or not rr:
        raise ValueError("Missing foundation training runs")
    if any(not r["anchored"] or output not in r["channels"] for r in rr):
        raise ValueError("Absolute foundation requires independent zero anchors")
    names = tuple(template["feature_names"])
    spec = template["spec"]
    counts = Counter(r["condition_id"] for r in rr)
    # 初期の選択結果はfoundation_loss、最終保存モデルはlossに同じ設定を記録。
    weighting = template.get("foundation_loss", template.get("loss", "equal_run"))
    if weighting == "equal_run_absolute_point":
        weighting = "equal_run"
    fraction = float(template.get('reference_fraction', 0.))
    if not 0. <= fraction <= 1.:
        raise ValueError('reference_fraction must be in [0, 1]')
    xx, yy, audit = [], [], []
    for r in rr:
        ch = r["channels"][output]
        d = ch[route]
        take = ch["take"]
        n = int(take.sum())
        weight = 1 / np.sqrt(n)
        if weighting in ("condition_equal", "zero_prior_weighted"):
            weight /= np.sqrt(counts[r["condition_id"]])
        elif weighting != "equal_run":
            raise ValueError("Unknown foundation loss")
        if weighting == "zero_prior_weighted":
            i = cal["ids"].index(r["run"])
            weight /= np.sqrt(cal["covariance"][output][i, i] + cal["noise"][output] ** 2)
        x = np.column_stack(
            [np.ones(n) if k == "intercept" else d["features"][spec][k][take] for k in names]
        )
        y = ch["truth"][take] - d["physical"][take]
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("Invalid samples; do not silently remove")
        if fraction:
            if not ch['reference_available'] or d['reference_features'][spec] is None:
                raise ValueError('Training run has no initial reference')
            xref = np.array([1. if k == 'intercept' else d['reference_features'][spec][k] for k in names])
            yref = ch['reference']-d['reference_physical']
            # Original weight contains 1/sqrt(n); keep total run weight fixed.
            reference_weight = np.sqrt(fraction*n)
            if fraction == 1.:
                x, y = xref[None, :]*reference_weight, np.array([yref])*reference_weight
            else:
                x = np.vstack([x*np.sqrt(1-fraction), xref*reference_weight])
                y = np.r_[y*np.sqrt(1-fraction), yref*reference_weight]
        xx.append(x * weight)
        yy.append(y * weight)
        audit.append(dict(run=r["run"], samples=n, weight=float(weight)))
    return np.vstack(xx), np.concatenate(yy), names, audit


def fit_foundation(runs, cal, route, output, template):
    x, y, names, audit = assemble_foundation(runs, cal, route, output, template)
    if template.get("bounds"):
        raise ValueError("Unexpected foundation coefficient bounds")
    blocks = RegressionBlocks(x, y, names, dict(training_runs=[a["run"] for a in audit]))
    model = fit_regularized(
        blocks,
        ridge=template["ridge"],
        ridge_exempt=tuple(template.get("ridge_exempt", ("intercept",))),
    )
    model.update(
        spec=template["spec"],
        output=output,
        absolute_training_runs=model["training_runs"],
        absolute_prediction_supported=True,
        frozen_structure=True,
        contributions=audit,
        loss=template.get("foundation_loss", template.get("loss", "equal_run_absolute_point")),
        zero_updated=False,
    )
    if 'reference_fraction' in template:
        model['reference_fraction'] = template['reference_fraction']
    return model


def tune_foundation(runs, cal, route, output, template, baseline):
    """Small, training-only 3-condition-fold loss/ridge search; terms are fixed.

    Score the initial load actually used by the absolute+increment composition.
    Average repeats within condition, then conditions within reference class,
    then the old/new classes equally. No outer-held-out loads enter selection.
    Accept only >=2% lower inner MSE, with neither reference class >5% worse.
    These are development selection rules, not statistical significance claims.
    """
    train_ids = set(template['training_runs'])
    rr = [r for r in runs if r['run'] in train_ids]
    conditions = sorted({r['condition_id'] for r in rr},
                        key=lambda c: hashlib.sha256(('foundation-polish-v1:'+c).encode()).hexdigest())
    if len(conditions) < 6:
        result = dict(baseline)
        result['foundation_polish'] = dict(selected=False, reason='fewer_than_six_training_conditions')
        return result
    folds = {c:i % 3 for i,c in enumerate(conditions)}
    original = dict(loss=template.get('foundation_loss',template.get('loss','equal_run_absolute_point')),
                    reference_fraction=float(template.get('reference_fraction',0.)), ridge=template['ridge'])
    candidates = [original]
    for fraction in (0., .5, 1.):
        for loss in ('equal_run', 'condition_equal'):
            for ridge in (0., .001, .01, .1, 1.):
                candidate = dict(loss=loss, reference_fraction=fraction, ridge=ridge)
                if candidate not in candidates:
                    candidates.append(candidate)

    inner = []
    scores = [[] for _ in candidates]
    for fold in range(3):
        training = [r for r in rr if folds[r['condition_id']] != fold]
        validation = [r for r in rr if folds[r['condition_id']] == fold]
        inner.append(dict(fold=fold, training_runs=[r['run'] for r in training],
                          validation_runs=[r['run'] for r in validation]))
        # QR once per loss/fraction; changing ridge needs only a small solve.
        designs = {}
        for i, candidate in enumerate(candidates):
            key = (candidate['loss'], candidate['reference_fraction'])
            trial = dict(template, **candidate, foundation_loss=candidate['loss'],
                         training_runs=[r['run'] for r in training])
            if key not in designs:
                x,y,names,audit = assemble_foundation(training,cal,route,output,trial)
                designs[key] = compress_regression_blocks(RegressionBlocks(
                    x,y,names,dict(training_runs=[r['run'] for r in training])))
            model = fit_regularized(designs[key], ridge=candidate['ridge'],
                                    ridge_exempt=tuple(template.get('ridge_exempt',('intercept',))))
            model['spec'] = template['spec']
            for r in validation:
                ch = r['channels'][output]
                error = reference_prediction(ch[route],model)-ch['reference']
                scores[i].append(dict(condition=r['condition_id'],
                    reference=r.get('absolute_reference_kind','independent_absolute'), squared_error=error**2))
    evaluations = []
    for candidate, rows in zip(candidates,scores):
        per_condition = defaultdict(list)
        for row in rows:
            per_condition[row['reference'],row['condition']].append(row['squared_error'])
        groups = defaultdict(list)
        for (reference,_), values in per_condition.items():
            groups[reference].append(float(np.mean(values)))
        class_mse = {reference:float(np.mean(values)) for reference,values in groups.items()}
        evaluations.append(dict(candidate=candidate, class_mse=class_mse,
                                score=float(np.mean(list(class_mse.values())))))
    base = evaluations[0]
    eligible = [i for i,e in enumerate(evaluations)
                if all(e['class_mse'][ref] <= 1.05*value+1e-16 for ref,value in base['class_mse'].items())]
    best = min(eligible,key=lambda i:evaluations[i]['score'])
    if evaluations[best]['score'] >= .98*base['score']:
        best = 0
    candidate = candidates[best]
    if best:
        trial = dict(template, **candidate, foundation_loss=candidate['loss'])
        result = fit_foundation(rr,cal,route,output,trial)
    else:
        result = dict(baseline)
    result['foundation_polish'] = dict(
        selected=best!=0, candidate=candidate, baseline_score=base['score'],
        selected_score=evaluations[best]['score'], candidates=evaluations, inner_folds=inner,
        selection_training_runs=sorted(train_ids), terms_changed=False, zero_updated=False,
        score='condition-then-reference-class equal initial-load MSE',
        acceptance='inner MSE improvement >=2%; neither reference class MSE >5% worse',
        limitation='Conditional on frozen calibration and previously selected terms; outer development folds are not a new blind test')
    return result


def reference_prediction(data, model):
    """Only explanatory state and own-route RPM. No measured reference load."""
    features = data["reference_features"][model["spec"]]
    if features is None:
        raise ValueError("No recorded initial explanatory state")
    bank = {k: np.atleast_1d(v) for k, v in features.items()}
    return float(data["reference_physical"] + predict_correction(model, bank)[0])


def fit_yaw_matrix(x, y, names, ridge, bank, *, local_penalty=1.0):
    if not names:
        return dict(
            bank=bank, feature_names=[], coefficient_by_name={}, ridge=ridge, numerical_rank=0
        )
    blocks = compress_regression_blocks(RegressionBlocks(x, y, tuple(names), {}))
    # fit_regularized penalizes beta*column_norms. Increasing only the frozen
    # norm gives an exactly stronger local penalty without changing X beta.
    norms = blocks.column_norms.copy()
    for j, name in enumerate(names):
        if name.startswith("local") and name[5:6].isdigit():
            norms[j] *= local_penalty
    adjusted = RegressionBlocks(blocks.x, blocks.y, blocks.names, blocks.audit, norms)
    model = fit_regularized(adjusted, ridge=ridge)
    model.update(bank=bank, local_penalty=local_penalty)
    return model


# ===========================================================================
# 7. 有風Yawの基準移送（独立な絶対値検証ではない）
# ===========================================================================
def bracket(target: dict, anchors: list[dict], *, exclude_conditions=()) -> dict:
    """No power/clock crossing, extrapolation, or selection by load residual."""
    candidates = [
        a
        for a in anchors
        if a["run"] != target["run"]
        and tuple(a["key"]) == tuple(target["key"])
        and a["condition_id"] not in exclude_conditions
    ]
    before = [a for a in candidates if a["minute"] < target["minute"]]
    after = [a for a in candidates if a["minute"] > target["minute"]]
    if not before or not after:
        return dict(status="one_sided" if candidates else "no_anchor_in_epoch")
    left, right = max(before, key=lambda a: a["minute"]), min(after, key=lambda a: a["minute"])
    fraction = (target["minute"] - left["minute"]) / (right["minute"] - left["minute"])
    return dict(
        status="interpolated",
        runs=[left["run"], right["run"]],
        weights=[1.0 - fraction, fraction],
        gap_minutes=right["minute"] - left["minute"],
        distances_minutes=[target["minute"] - left["minute"], right["minute"] - target["minute"]],
    )


def blend(values, weights):
    w = np.asarray(weights, float)
    x = np.asarray(values, float)
    if not len(w) or len(x) != len(w) or not np.isfinite(x).all() or not np.isfinite(w).all():
        raise ValueError("Finite aligned values required")
    if np.any(w < 0) or not np.isclose(w.sum(), 1.0):
        raise ValueError("Only convex interpolation is allowed")
    return w @ x


def compose(model, uncalibrated_load, offset, foundation_constant):
    """One constant per run; predictions never use that target run's load."""
    f, y = np.asarray(model), np.asarray(uncalibrated_load)
    return f + foundation_constant, y - offset


# ===========================================================================
# 8. 実行順序: 信号 → 力学 → 説明変数 → fit → 予測 → 誤差・保存
# ===========================================================================
def window_indices(time, state, stride=1):
    """記録済みの窓。荷重の大小を見て窓を選び直さない。"""
    ix = np.flatnonzero((time >= state["start"] - 1e-9) & (time <= state["end"] + 1e-9))
    if not len(ix):
        raise ValueError("Recorded reference window is absent")
    return ix[::stride]


def on_grid(time, requested):
    ix = np.searchsorted(time, requested)
    np.testing.assert_allclose(time[ix], requested, atol=1e-10, rtol=0)
    return ix


def transfer_windows(entries):
    """2経路の予測窓が異なっても、観測の基準窓は実測RPM側に統一。"""
    donors = defaultdict(dict)
    targets = {}
    for entry in entries:
        if entry["route"] == "measured":
            targets[entry["run"]] = entry["anchor_state"]
        for d in entry["donors"]:
            key = (d["state"]["start"], d["state"]["end"])
            donors[d["run"]][key] = d["state"]
    return donors, targets


def prepare_inputs(args, profile):
    """再構成や予測の保存値を使わず、信号と固定校正から再計算する。"""
    cfg = Settings(**profile["settings"])
    sensor = sensor_parameters(profile["sensor_summary"])
    body_cfg, _ = split_calibration(cfg, sensor)
    np.testing.assert_allclose(
        body_cfg.cg_flu_m, profile["separated_settings"]["cg_flu_m"], atol=1e-12
    )
    rules = defaultdict(lambda: defaultdict(set))
    for family in ("foundation", "increment"):
        for key, model in profile["models"][family].items():
            _, route, output = key.split("/")
            if family == "increment" and output == "Mz":
                continue  # 最新Yawは別の連続モデル
            rules[route, output][model["spec"]].update(model["feature_names"])
            if getattr(args, "reselect_physics", False):
                rules[route, output][model["spec"]].update(theory_feature_names(output))
    if getattr(args, "six_axis", False):
        for route in ROUTES:
            for output in EXTRA_CHANNELS:
                specs = ("theory_compact", "theory_expanded") if getattr(args, "reselect_physics", False) else ("compact", "expanded")
                for spec in specs:
                    rules[route, output][spec].update(extra_feature_names(output, spec))
    anchors = defaultdict(list)
    for a in profile["anchors"]:
        anchors[a["epoch"], a["aircraft_boot_id"]].append(a)
    donor_windows, target_windows = transfer_windows(profile["transfer"]["direct"])
    targets = set(profile["transfer"]["target_runs"])
    prepared, yaw_runs, transfer = [], [], {"donors": {}, "targets": {}}
    for count, info in enumerate(profile["runs"]):
        name = info["run"]
        path = args.bundle / "signals" / f"{name}.npz"
        if hashlib.sha256(path.read_bytes()).hexdigest() != info["signals_sha256"]:
            raise ValueError("Input signal changed: " + name)
        with np.load(path, allow_pickle=False) as f:
            raw = {k: f[k] for k in f.files}
        raw.update(
            {
                k: info[k]
                for k in ("run", "role", "axis", "wind", "heading", "grid_origin_ns", "provenance")
            }
        )
        if args.raw_root:
            fresh = load_raw(
                args.raw_root / name,
                args.output / "clock-cache",
                args.aligned_root,
                cfg,
                {"motion_adoption": "ACCEPT_frozen_manifest"},
            )
            np.testing.assert_array_equal(fresh["t"], raw["t"])
            for key in ("command", "voltage", "rpm", "gyro", "angles", "load"):
                np.testing.assert_allclose(fresh[key], raw[key], atol=1e-10, rtol=0, equal_nan=True)
                raw[key] = fresh[key]
        prepare_motion(raw, cfg)
        rpm_estimated, _ = predict_rpm(raw["command"], raw["voltage"], profile["motor_models"], cfg)
        zero = anchor_mean(anchors[tuple(info["key"])], np.array([info["center"]]))
        if (zero is not None) != info["anchored"]:
            raise ValueError("Independent zero availability changed: " + name)
        run_cfg = replace(cfg, bias_sensor=tuple(zero[0])) if zero is not None else cfg
        fc, _, _, _ = reconstruct_body(raw, run_cfg, sensor)
        ix = on_grid(raw["t"], raw["selected_time"])
        ref = raw["reference"]
        run = dict(info, time=raw["t"][ix], channels={})
        if getattr(args, "systematic_search", False):
            run["search_data"] = {}
        channel_metadata = dict(info["channels"])
        if getattr(args, "six_axis", False):
            channel_metadata.update({k: dict(info["channels"]["Fz"]) for k in EXTRA_CHANNELS})
        for output, meta in channel_metadata.items():
            j = OUTPUTS.index(output)
            run["channels"][output] = dict(
                meta,
                take=raw["take_" + ("Fz" if output in EXTRA_CHANNELS else output)],
                truth=raw["target"][ix, j].copy(),
                reference=float(raw["target"][ref, j].mean())
                if meta["reference_available"]
                else None,
            )
        yrun = None
        if "yaw" in info:
            yi = on_grid(raw["t"], raw["yaw_time"])
            ym = info["yaw"]
            ri = np.flatnonzero(ref) if ym["reference_available"] else yi[: ym["original_count"]]
            yrun = dict(
                info,
                **{k: v for k, v in ym.items() if k not in info},
                time=raw["t"][yi],
                y=reference_contrast(raw["target"][:, 5], yi, ri),
                route={},
            )
        # 6軸の再構成と、同定対象/独立基準の有無は別々に保存する。
        np.savez_compressed(
            args.output / "reconstructed" / f"{name}.npz",
            time=raw["t"][ix],
            wrench=raw["target"][ix],
            omega=raw["omega"][ix],
            alpha=raw["alpha"][ix],
            rpm_measured=raw["rpm"][ix],
            rpm_estimated=rpm_estimated[ix],
            independent_absolute_reference=np.array(info["anchored"]),
        )
        if name in targets:
            saved = raw["target"].copy()
            reconstruct_body(raw, replace(cfg, bias_sensor=(0.0,) * 6), sensor)
            load0 = raw["target"][:, 5].copy()
            raw["target"] = saved
            ti = yi[: info["yaw"]["original_count"]]
            transfer["targets"][name] = dict(
                run=name,
                key=info["key"],
                condition_id=info["condition_id"],
                minute=info["center"],
                fold=info["fold"],
                time=raw["t"][ti],
                load0=load0[ti],
                route={},
            )
            if name in target_windows:
                ai = window_indices(raw["t"], target_windows[name])
                transfer["targets"][name].update(
                    anchor_observed=float(load0[ai].mean()),
                    anchor_minute=float((raw["grid_origin_ns"] + raw["t"][ai].mean() * 1e9) / 60e9),
                )
        for route in ROUTES:
            rpm = raw["rpm"] if route == "measured" else rpm_estimated
            data = route_state(
                raw, rpm.copy(), replace(fc, fz_model="sun_graybox", yaw_motion_model="sun_blended")
            )
            data["rms_speed_ratio"] = np.sqrt(
                np.maximum(-data["physical"][:, 2] / (4 * fc.kf), 0)
            ) / (4000 * 2 * np.pi / 60)
            if getattr(args, "systematic_search", False):
                from analysis_search import retain_state
                run["search_data"][route] = retain_state(data, ix, ref,
                    yi if yrun is not None else None, ri if yrun is not None else None)
            for output, ch in run["channels"].items():
                j = OUTPUTS.index(output)
                bank = selected_banks(data, fc, output, rules[route, output])
                d = dict(
                    physical=data["physical"][ix, j],
                    reference_physical=float(data["physical"][ref, j].mean())
                    if ch["reference_available"]
                    else None,
                    features={s: {k: v[ix] for k, v in b.items()} for s, b in bank.items()},
                    reference_features={
                        s: {k: float(v[ref].mean()) for k, v in b.items()}
                        if ch["reference_available"]
                        else None
                        for s, b in bank.items()
                    },
                )
                ch[route] = d
                if route == "measured":
                    ch.update(physical=d["physical"], physical_reference=d["reference_physical"])
            if yrun is not None or name in donor_windows:
                cb = continuous_bank(data)
                ab = dict(cb, **physics_bank(data, fc, SIGNS))
                if getattr(args, "reselect_physics", False):
                    added = theory_feature_bank(data, fc, "Mz")
                    cb.update(added)
                    ab.update(added)
                if yrun is not None:
                    yrun["route"][route] = dict(
                        continuous={k: reference_contrast(v, yi, ri) for k, v in cb.items()},
                        augmented={k: reference_contrast(v, yi, ri) for k, v in ab.items()},
                        physical=reference_contrast(data["physical"][:, 5], yi, ri),
                        rpm=data["rms_speed_ratio"][yi] * 4000,
                    )
                if name in targets:
                    t = transfer["targets"][name]
                    t["route"][route] = dict(
                        physical=data["physical"][ti, 5], bank={k: v[ti] for k, v in ab.items()}
                    )
                    if name in target_windows:
                        t["route"][route].update(
                            anchor_physical=float(data["physical"][ai, 5].mean()),
                            anchor_bank={k: float(v[ai].mean()) for k, v in ab.items()},
                        )
                for window_key, state in donor_windows[name].items():
                    wi = window_indices(raw["t"], state, cfg.stride)
                    transfer["donors"][name, route, window_key] = dict(
                        load=float(raw["target"][wi, 5].mean()),
                        physical=float(data["physical"][wi, 5].mean()),
                        bank={k: float(v[wi].mean()) for k, v in ab.items()},
                    )
        prepared.append(run)
        if yrun is not None:
            yaw_runs.append(yrun)
        if count % 20 == 0:
            print("PREPARE", count + 1, len(profile["runs"]), flush=True)
    return body_cfg, prepared, yaw_runs, transfer


def select_theory_model(matrix, template):
    """Select from prior terms + theory on TRAINING rows, then retain old ridge.

    Existing physically bounded corrections keep their bounds if selected.
    No new sign constraint is inferred for the total aerodynamic residual.
    """
    names, audit = select_extra_terms(matrix)
    if names:
        bounds = {k: tuple(-np.inf if v is None and j == 0 else np.inf if v is None else v
                            for j, v in enumerate(values))
                  for k, values in template.get("bounds", {}).items() if k in names}
        model = fit_regularized(compress_regression_blocks(subset_blocks(matrix, names)),
                                ridge=template["ridge"], bounds=bounds,
                                ridge_exempt=template.get("ridge_exempt", ("intercept",)))
    else:
        model = dict(feature_names=[], coefficients=[], coefficient_by_name={}, column_scales=[],
                     scaled_coefficients=[], ridge=template["ridge"], numerical_rank=0,
                     normalized_condition_number=None, bounds={}, **matrix.audit)
    return {**template, **model, "selection": audit, "frozen_structure": False,
            "theory_candidates": THEORY_VERSION, "hyperparameters_reselected": False,
            "zero_updated": False}


def refit_models(profile, runs, yaw_runs, cal, cfg, *, polish_foundation=False, reselect_physics=False):
    """原則は凍結設定の再fit。任意でMy/Mz基礎の損失・ridgeだけ学習内選択。"""
    result = {"foundation": {}, "increment": {}, "yaw": {}}
    for family in ("foundation", "increment"):
        for key, old in profile["models"][family].items():
            fold, route, output = key.split("/")
            if family == "increment" and output == "Mz":
                continue
            print("FIT", family, key, flush=True)
            if reselect_physics:
                trial = dict(old, feature_names=list(dict.fromkeys(
                    [*old["feature_names"], *theory_feature_names(output)])))
                if family == "foundation" and output != "Fz":
                    x, y, names, contributions = assemble_foundation(runs, cal, route, output, trial)
                    matrix = RegressionBlocks(x, y, names, dict(training_runs=old["training_runs"],
                        absolute_training_runs=old["training_runs"], absolute_prediction_supported=True,
                        contributions=contributions))
                else:
                    matrix = frozen_fit(runs, cal, route, output, trial, cfg, matrix_only=True)
                new = select_theory_model(matrix, trial)
            elif family == "foundation" and output != "Fz":
                new = fit_foundation(runs, cal, route, output, old)
                if polish_foundation:
                    new = tune_foundation(runs,cal,route,output,old,new)
                    print('POLISH', key, new['foundation_polish'].get('candidate','baseline'), flush=True)
            else:
                new = frozen_fit(runs, cal, route, output, old, cfg)
            result[family][key] = new
    for key, old in profile["models"]["yaw"].items():
        fold, route = key.split("/")
        train = [r for r in yaw_runs if r["run"] in old["training_runs"]]
        if len(train) != len(old["training_runs"]):
            raise ValueError("Missing Yaw training run")
        names = old["feature_names"]
        if reselect_physics:
            names = list(dict.fromkeys([*names, *theory_feature_names("Mz")]))
        x, y = training_matrix(
            train,
            route,
            old["bank"],
            names,
            extra_ratio=old["extra_ratio"],
            balanced=False,
            noise=profile["yaw_noise"],
        )
        if reselect_physics:
            selected, audit = select_extra_terms(RegressionBlocks(x, y, tuple(names), {}))
            indices = [names.index(k) for k in selected]
            # Reselection treats old splines and added theory as peers after
            # column normalization; the legacy-only local multiplier is removed.
            new = fit_yaw_matrix(x[:, indices], y, selected, old["ridge"], old["bank"],
                                 local_penalty=1.0)
            new.update(selection=audit, frozen_structure=False, theory_candidates=THEORY_VERSION,
                       hyperparameters_reselected=False, zero_updated=False,
                       local_penalty=1.0, candidate_penalty_policy="uniform_normalized_columns")
        else:
            new = fit_yaw_matrix(
                x, y, names, old["ridge"], old["bank"], local_penalty=old["local_penalty"]
            )
        new.update(training_runs=old["training_runs"], extra_ratio=old["extra_ratio"])
        result["yaw"][key] = new
        print("FIT yaw", key, flush=True)
    return result


def error_metrics(truth, predicted):
    error = np.asarray(predicted) - truth
    if not len(error) or not np.isfinite(error).all():
        raise ValueError("Invalid comparison; no silent point removal")
    return dict(
        samples=len(error),
        rmse=float(np.sqrt(np.mean(error**2))),
        bias=float(error.mean()),
        centered_rmse=float(error.std()),
    )


def check_fold(model, run, mode, runs):
    if mode == "all_data_in_sample":
        return
    train = set(model["training_runs"])
    if run["run"] in train or any(
        r["condition_id"] == run["condition_id"] for r in runs if r["run"] in train
    ):
        raise ValueError("Training/validation condition overlap")


def evaluate(runs, yaw_runs, models):
    """My/Mz=初期状態からの基礎予測+増分。実測初期荷重で予測を合わせない。"""
    yindex = {r["run"]: r for r in yaw_runs}
    rows, arrays = [], {}
    for r in runs:
        for output, ch in r["channels"].items():
            if output in EXTRA_CHANNELS:
                continue  # 追加3軸はevaluate_extra_channelsで評価
            if not ch["reference_available"]:
                continue
            take = ch["take"]
            name = r["run"]
            time = r["time"][take]
            for mode, fold in (("all_data_in_sample", "final"), ("grouped_OOF", r["fold"])):
                for route in ROUTES:
                    key = f"{fold}/{route}/{output}"
                    d = ch[route]
                    foundation = models["foundation"][key]
                    check_fold(foundation, r, mode, runs)
                    initial = reference_prediction(d, foundation)
                    if output == "Mz":
                        model = models["yaw"][f"{fold}/{route}"]
                        yr = yindex[name]
                        check_fold(model, r, mode, runs)
                        delta = predict_yaw_delta(yr, model, route)[: yr["original_count"]]
                        np.testing.assert_allclose(
                            time, yr["time"][: yr["original_count"]], atol=1e-10
                        )
                    else:
                        model = models["increment"][key] if output == "My" else foundation
                        check_fold(model, r, mode, runs)
                        absolute = (
                            d["physical"][take]
                            + predict_correction(model, d["features"][model["spec"]])[take]
                        )
                        delta = absolute - reference_prediction(d, model)
                    prediction = initial + delta
                    prefix = f"{mode}/{output}/{route}/{name}/"
                    variants = (
                        (
                            "increment",
                            ch["truth"][take] - ch["reference"],
                            delta,
                            ch["physical"][take] - ch["physical_reference"],
                        ),
                    )
                    if r["anchored"]:
                        variants += (
                            (
                                r.get("absolute_reference_kind", "independent_absolute"),
                                ch["truth"][take],
                                prediction,
                                ch["physical"][take],
                            ),
                        )
                    for reference, truth, predicted, physical in variants:
                        row = dict(
                            run=name,
                            mode=mode,
                            output=output,
                            route=route,
                            reference=reference,
                            wind=r["wind"],
                            condition_id=r["condition_id"],
                            physical=error_metrics(truth, physical),
                            identified=error_metrics(truth, predicted),
                        )
                        rows.append(row)
                        p = prefix + reference + "/"
                        arrays.update(
                            {
                                p + k: v
                                for k, v in dict(
                                    time=time, truth=truth, prediction=predicted, physical=physical
                                ).items()
                            }
                        )
                    # Unsupported absolute predictions are saved, never scored as absolute truth.
                    arrays[prefix + "absolute_prediction"] = prediction
    return rows, arrays


def yaw_absolute(physical, bank, model):
    return physical + sum(c * bank[k] for k, c in model["coefficient_by_name"].items())


def transfer_yaw(profile, transfer, models, rows, arrays):
    """Pitch基準の差を移送し、同じ電源/時計区間の前後だけで線形補間。

    c_j = Σ donor_weight * (donor_Mz - model(donor_state))
    b_j = mean(target_load0 at anchor) - mean(model at anchor) - c_j
    prediction_k = model_k + interpolate(c_j)
    assisted_observation_k = load0_k - interpolate(b_j)
    同じbを両RPM経路と物理比較に適用。run内でbを変化させない。
    """
    nodes = {}
    targets = transfer["targets"]
    donors = transfer["donors"]
    for entry in profile["transfer"]["direct"]:
        name, route = entry["run"], entry["route"]
        t = targets[name]
        node = nodes.setdefault(
            name,
            dict(
                run=name,
                key=t["key"],
                condition_id=t["condition_id"],
                minute=t["anchor_minute"],
                constant={},
                offset={},
            ),
        )
        for fold in ("final", *range(5)):
            model = models["yaw"][f"{fold}/{route}"]
            constant = 0.0
            for d in entry["donors"]:
                window = (d["state"]["start"], d["state"]["end"])
                v = donors[d["run"], route, window]
                constant += d["weight"] * (
                    v["load"] - yaw_absolute(v["physical"], v["bank"], model)
                )
            node["constant"][f"{fold}/{route}"] = float(constant)
            if route == "measured":
                v = t["route"][route]
                node["offset"][str(fold)] = float(
                    t["anchor_observed"]
                    - yaw_absolute(v["anchor_physical"], v["anchor_bank"], model)
                    - constant
                )
    coverage = []
    for name, t in targets.items():
        plan = (
            dict(status="direct", runs=[name], weights=[1.0], gap_minutes=0.0)
            if name in nodes
            else bracket(t, list(nodes.values()))
        )
        coverage.append(dict(run=name, **plan))
        if plan["status"] not in ("direct", "interpolated"):
            continue
        for mode, fold in (("all_data_in_sample", "final"), ("condition_model", t["fold"])):
            use = (
                plan
                if mode == "all_data_in_sample" or plan["status"] == "direct"
                else bracket(t, list(nodes.values()), exclude_conditions=[t["condition_id"]])
            )
            if use["status"] not in ("direct", "interpolated"):
                continue
            offset = float(
                blend([nodes[n]["offset"][str(fold)] for n in use["runs"]], use["weights"])
            )
            truth = t["load0"] - offset
            physical = t["route"]["measured"]["physical"]
            for route in ROUTES:
                constant = blend(
                    [nodes[n]["constant"][f"{fold}/{route}"] for n in use["runs"]], use["weights"]
                )
                d = t["route"][route]
                prediction = (
                    yaw_absolute(d["physical"], d["bank"], models["yaw"][f"{fold}/{route}"])
                    + constant
                )
                reference = "assisted_" + plan["status"]
                rows.append(
                    dict(
                        run=name,
                        mode=mode,
                        output="Mz",
                        route=route,
                        reference=reference,
                        wind=11.5,
                        condition_id=t["condition_id"],
                        physical=error_metrics(truth, physical),
                        identified=error_metrics(truth, prediction),
                    )
                )
                prefix = f"{mode}/Mz/{route}/{name}/{reference}/"
                arrays.update(
                    {
                        prefix + k: v
                        for k, v in dict(
                            time=t["time"], truth=truth, prediction=prediction, physical=physical
                        ).items()
                    }
                )
    return dict(
        nodes=list(nodes.values()),
        coverage=coverage,
        limitation="Model-assisted absolute level; not independent sensor-zero or absolute validation",
    )


def summarize(rows):
    groups = defaultdict(list)
    for r in rows:
        ref = "assisted_absolute" if r["reference"].startswith("assisted_") else r["reference"]
        for wind in ("all", r["wind"]):
            groups[r["mode"], r["output"], r["route"], ref, wind].append(r)
    result = []
    for (mode, output, route, reference, wind), rr in groups.items():
        item = dict(
            mode=mode, output=output, route=route, reference=reference, wind=wind, runs=len(rr)
        )
        for variant in ("physical", "identified"):
            n = sum(r[variant]["samples"] for r in rr)
            item[variant] = dict(
                run_equal_rmse=float(np.sqrt(np.mean([r[variant]["rmse"] ** 2 for r in rr]))),
                sample_rmse=float(
                    np.sqrt(sum(r[variant]["rmse"] ** 2 * r[variant]["samples"] for r in rr) / n)
                ),
                run_equal_bias=float(np.mean([r[variant]["bias"] for r in rr])),
                samples=n,
            )
        result.append(item)
    return result


def verify_legacy(root, rows, arrays):
    """移行の検算専用。保存予測はここ以外の計算には使わない。"""
    source = np.load(root / "latest-absolute-refit-20261012-v1/result/predictions.npz")
    common = np.load(root / "empty-sensor-refit-20261011-v1/result/predictions.npz")
    yaw = np.load(root / "yaw-cancellation-reselect-20261012-v1/predictions.npz")
    assisted = np.load(
        root / "pitch-yaw-local-anchor-20261013-v1/time-interpolation-v1/predictions.npz"
    )
    maximum = 0.0
    checked = 0
    for r in rows:
        mode, o, route, name, ref = (r[k] for k in ("mode", "output", "route", "run", "reference"))
        dest = f"{mode}/{o}/{route}/{name}/{ref}/"
        if ref == "independent_absolute":
            archive = source
            p = f"{mode}/{o}/{route}/{name}/"
            fields = {
                "prediction": "updated",
                "truth": "truth",
                "physical": "physical",
                "time": "time",
            }
        elif ref == "increment":
            if o == "Mz":
                archive = yaw
                p = f"{'in_sample' if mode == 'all_data_in_sample' else mode}/{route}/{name}/"
                fields = {
                    "prediction": "reselected",
                    "truth": "truth",
                    "physical": "physical",
                    "time": "time",
                }
            else:
                archive = common
                p = f"{mode}/{o}/{route}/{name}/"
                fields = {
                    "prediction": "new",
                    "truth": "truth_new",
                    "physical": "physical_new",
                    "time": "time",
                }
        else:
            archive = assisted
            p = f"{'in_sample' if mode == 'all_data_in_sample' else mode}/{name}/"
            fields = {
                "prediction": route,
                "truth": "observed",
                "physical": "physical",
                "time": "time",
            }
        for k, v in fields.items():
            a, b = arrays[dest + k], archive[p + v]
            np.testing.assert_allclose(a, b, atol=2e-8, rtol=2e-8, err_msg=dest + k)
            maximum = max(maximum, float(np.max(abs(a - b))))
            checked += 1
    for archive in (source, common, yaw, assisted):
        archive.close()
    return dict(arrays_checked=checked, maximum_absolute_difference=maximum, tolerance=2e-8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle", type=Path, required=True, help="export_final_analysis_bundle.pyで凍結した入力"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="新規ディレクトリ（既存結果は上書きしない）"
    )
    parser.add_argument("--refit", action="store_true", help="採用項を固定して係数を再推定")
    parser.add_argument("--motor-model", type=Path,
                        help="Kaufmann式の校正JSON。荷重係数との整合のため--refit必須")
    parser.add_argument("--reselect-physics", action="store_true",
                        help="既存採用項＋機体抗力・風向・局所流入・減衰の理論候補から項を再選択")
    parser.add_argument("--systematic-search", action="store_true",
                        help="全既存候補＋対称な交差項を復帰し、絶対/差分の内側CVで選択")
    parser.add_argument("--search-outputs", nargs="+", choices=OUTPUTS,
                        help="系統探索する成分。省略時はMy/Mz/Fz（six-axis時は全6軸）")
    parser.add_argument("--polish-foundation", action="store_true",
                        help="My/Mz基礎の初期荷重重み・条件重み・ridgeを学習内3条件foldで選ぶ")
    parser.add_argument("--zero-recovery", type=Path,
                        help="運動補正した無風停止基準のsummary.json")
    parser.add_argument("--drift-prior", type=Path,
                        help="追加基準で再fitする際の独立ドリフト試験prior.json")
    parser.add_argument(
        "--six-axis", action="store_true", help="Fx/Fy/Mxの項選択・条件別検証を追加"
    )
    parser.add_argument("--raw-root", type=Path, help="指定時は生ログを再読込し同期済み信号と照合")
    parser.add_argument("--aligned-root", type=Path, help="検証済みの機器間同期cache")
    parser.add_argument("--verify-legacy", type=Path, help="移行検算だけに使う旧結果root")
    args = parser.parse_args()
    if args.motor_model and (not args.refit or args.verify_legacy):
        raise ValueError('--motor-model requires --refit and cannot verify legacy predictions')
    if args.polish_foundation and not args.refit:
        raise ValueError('--polish-foundation requires --refit')
    if args.reselect_physics and (not args.refit or args.polish_foundation):
        raise ValueError('--reselect-physics requires --refit and cannot be combined with --polish-foundation')
    if args.systematic_search and (not args.refit or args.reselect_physics or args.polish_foundation):
        raise ValueError('--systematic-search requires --refit, without other search flags')
    if args.systematic_search and args.six_axis and args.search_outputs and not set(EXTRA_CHANNELS) <= set(args.search_outputs):
        raise ValueError('--six-axis systematic search must include Fx, Fy and Mx')
    profile = read_json(args.bundle / "profile.json")
    if profile["schema"] != "fftgyro-final-20261013-v1":
        raise ValueError("Unknown input schema")
    if args.motor_model:
        motor = read_json(args.motor_model)
        if motor.get("schema") != "fftgyro-literature-motor-v1" or motor.get("flight_training") is not False:
            raise ValueError("風洞校正済みの文献モーターモデルJSONが必要です")
        if len(motor["models"]) != 4 or not all(m["kind"] == "kaufmann2023" for m in motor["models"]):
            raise ValueError("Kaufmann式の4基の係数が必要です")
        profile["motor_models"] = [dict(m, tau_s=motor["tau_s"], delay_s=motor["delay_s"])
                                   for m in motor["models"]]
        profile["motor_model_source"] = dict(path=str(args.motor_model), source=motor["source"],
            sha256=hashlib.sha256(args.motor_model.read_bytes()).hexdigest(),
            training_runs=motor["training_runs"],
            note="Motor calibration split must be distinguished from downstream load-validation folds")
    recovery_audit=None
    if args.zero_recovery:
        if args.six_axis:
            raise ValueError('Recovered reference fits currently cover My/Mz/Fz only')
        if args.refit and not args.drift_prior:
            raise ValueError('Recovered-reference refit requires --drift-prior')
        recovery = read_json(args.zero_recovery)
        recovery_audit=apply_zero_recovery(profile,recovery,
            hashlib.sha256((args.bundle/'profile.json').read_bytes()).hexdigest())
    args.output.mkdir(parents=True, exist_ok=False)
    if args.motor_model:
        write_json(args.output / "motor-model.json", motor)
    (args.output / "reconstructed").mkdir()
    cal = profile["calibration"]
    with np.load(args.bundle / "zero_covariance.npz", allow_pickle=False) as f:
        cal["covariance"] = {k: f[k] for k in f.files}
    if args.zero_recovery and args.refit:
        recovery_audit['calibration'] = extend_recovered_calibration(profile, recovery, read_json(args.drift_prior))
        recovery_audit['training'] = extend_foundation_training(profile)
        recovery_audit['coefficients_refitted'] = True
        recovery_audit['drift_prior_sha256'] = hashlib.sha256(args.drift_prior.read_bytes()).hexdigest()
        np.savez_compressed(args.output/'zero_covariance.npz', **cal['covariance'])
        write_json(args.output/'calibration.json', dict(ids=cal['ids'], noise=cal['noise'],
                   audit=recovery_audit['calibration']))
    cfg, runs, yaw_runs, transfer = prepare_inputs(args, profile)
    if args.systematic_search:
        # Trusted, self-created local checkpoint: never load third-party pickle files.
        import pickle
        with (args.output/"search-prepared.pickle").open("wb") as stream:
            pickle.dump((args, profile, cfg, runs, yaw_runs, transfer), stream, protocol=5)
    models = refit_models(profile, runs, yaw_runs, cal, cfg,
                          polish_foundation=args.polish_foundation,
                          reselect_physics=args.reselect_physics) if args.refit and not args.systematic_search else profile["models"]
    if args.systematic_search:
        from analysis_search import fit_systematic
        requested = args.search_outputs or (OUTPUTS if args.six_axis else CHANNELS)
        if any(k in EXTRA_CHANNELS for k in requested) and not args.six_axis:
            raise ValueError('Extra channels require --six-axis')
        models = fit_systematic(profile, runs, yaw_runs, cal, cfg, requested, args.output)
    rows, arrays = evaluate(runs, yaw_runs, models)
    transfer_audit = (dict(status="not_recomputed_systematic_search",
        reason="Model-assisted absolute references are not independent validation; unchanged stop references are scored.")
        if args.systematic_search else transfer_yaw(profile, transfer, models, rows, arrays))
    if args.six_axis:
        if not args.systematic_search:
            models = dict(models, extra=fit_extra_channels(runs, reselect_physics=args.reselect_physics))
        extra_rows, extra_arrays = evaluate_extra_channels(runs, models["extra"])
        rows.extend(extra_rows)
        arrays.update(extra_arrays)
    write_json(args.output / "yaw-transfer.json", transfer_audit)
    np.savez_compressed(args.output / "predictions.npz", **arrays)
    verification = verify_legacy(args.verify_legacy, rows, arrays) if args.verify_legacy else None
    summary = dict(
        result_schema="fftgyro-analysis-result-v1",
        profile=profile["schema"],
        coefficient_refit=args.refit,
        motor_model_source=profile.get("motor_model_source"),
        foundation_polish=args.polish_foundation,
        zero_recovery=recovery_audit,
        term_selection_repeated=args.six_axis or args.reselect_physics or args.systematic_search,
        term_selection_scope=(list(requested) if args.systematic_search else
            list(CHANNELS if args.reselect_physics else ()) + list(EXTRA_CHANNELS if args.six_axis else ())),
        theoretical_candidates=THEORY_VERSION if args.reselect_physics else None,
        systematic_search=args.systematic_search,
        systematic_search_outputs=(list(requested) if args.systematic_search else []),
        systematic_search_code_sha256=(hashlib.sha256(Path(__file__).with_name("analysis_search.py").read_bytes()).hexdigest()
                                      if args.systematic_search else None),
        theoretical_selection_policy=("Prior selected terms + explicit theory; Sun F/PSE on training folds. "
            "My/Mz/Fz ridge/loss remain frozen; Yaw local multiplier is 1 (uniform normalized-column penalty). "
            "No new nested hyperparameter search. "
            "No term is forced; coefficients of corrections need not be nonnegative; not a passive model guarantee."
            if args.reselect_physics else None),
        extra_channels_refit=args.six_axis,
        extra_channels_method=(
            "Sun F/PSE within inner 3-condition-fold dictionary/ridge selection; outer frozen 5 folds. "
            "Independent anchored initial foundation + reference increment; condition-equal losses. "
            "Calibration covariance is not used as a fit weight for Fx/Fy/Mx."
            if args.six_axis
            else None
        ),
        signal_runs=len(runs),
        yaw_training_runs=len(yaw_runs),
        per_run=rows,
        metrics=summarize(rows),
        source_profile_sha256=hashlib.sha256(
            (args.bundle / "profile.json").read_bytes()
        ).hexdigest(),
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        verification=verification,
        limitations=[
            "Independent absolute and Pitch-assisted absolute references must not be pooled as independent validation.",
            "Run-constant zero; endpoint holding for independent stop anchors, no extrapolation for assisted Yaw.",
            ("Theory candidates reselected with fixed calibration and My/Mz/Fz hyperparameters; not a new blind test."
             if args.reselect_physics else
             "Recovered calibration covariance extended with independent drift prior; terms/old covariance are fixed."
             if args.zero_recovery and args.refit else
             "Calibration covariance and model structures are frozen external inputs, not newly estimated here."),
            ("My/Mz foundation loss and ridge tuned within training conditions; increments/Fz unchanged by polishing."
             if args.polish_foundation else "Foundation/increment structure and ridge selected by inner condition CV."
             if args.systematic_search else "Foundation loss/ridge selection not repeated."),
            (
                "Six component fits; Mx is a cross-axis response, NOT independently excited Roll dynamics. "
                "Fx/Fy physical baseline is zero, NOT an independently validated drag model."
                if args.six_axis
                else "Final fits cover My/Mz/Fz; all six components are reconstructed, not all six identified."
            ),
            "Repeatedly explored grouped OOF is not a fresh blind test. No free-flight validation is claimed.",
        ],
    )
    write_json(args.output / "summary.json", summary)
    write_json(args.output / "models.json", models)
    print("DONE", args.output, flush=True)


if __name__ == "__main__":
    main()
