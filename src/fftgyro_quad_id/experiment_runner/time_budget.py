"""Offline time budget from the registered sessions, not their watchdog limits.

Mirrors session.cpp reference endpoints and session_runner.cpp transitions.
Continuous-time estimate; a small tick allowance covers 100 Hz boundaries.
This is neither a scheduler nor proof that a physical hold will settle.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import math
from pathlib import Path
import struct

from .production_console import REPO, catalog, current_selection, read_json, resolve, session_for, sha


def endpoint(segment: dict, *, end: bool) -> float:
    kind = segment["kind"]
    if kind in ("MOVE", "SCAN"):
        return segment["end_deg"] if end else segment["start_deg"]
    if kind == "HOLD":
        # HOLD's end_deg is optional and is not used by the executor.
        return segment["start_deg"]
    if kind == "SINE":
        if segment.get("fade_s", 0) > 0:
            return segment["start_deg"]
        t = segment["duration_s"] if end else 0
        phase = segment.get("phase_rad", 0) + 2 * math.pi * segment["frequency_hz"] * t
        return segment["start_deg"] + segment["amplitude_deg"] * math.sin(phase)
    raise ValueError(f"Unsupported segment kind: {kind}")


def fixed_reference(segment: dict, t: float) -> list[float]:
    value = segment["fixed_input"]
    if segment["kind"] != "SINE":
        return value
    x = min(1.0, max(0.0, min(t, segment["duration_s"] - t) / segment["fade_s"]))
    envelope = x ** 3 * (10 + x * (-15 + 6 * x))
    wave = envelope * math.sin(2 * math.pi * segment["frequency_hz"] * t + segment.get("phase_rad", 0))
    return [v + a * wave for v, a in zip(value, segment["fixed_amplitude"])]


def float32(value: float) -> float:
    return struct.unpack("f", struct.pack("f", value))[0]


@dataclass(frozen=True)
class SessionTime:
    segments_s: float
    ramps_s: float
    transitions_s: float
    tick_allowance_s: float
    maximum_settling_extension_s: float

    @property
    def normal_s(self) -> float:
        return self.segments_s + self.ramps_s + self.transitions_s + self.tick_allowance_s


def estimate_session(c: dict) -> SessionTime:
    segments, hardware, startup = c["segments"], c["hardware"], c["startup"]
    fixed = c["control_mode"] == "FIXED_INPUT"
    if c["control_mode"] not in ("FIXED_INPUT", "FIXED_HOLD", "AXIS_PID"):
        raise ValueError("Unsupported control mode")
    first, last = segments[0], segments[-1]
    up = first["collective"] / hardware["startup_ramp_rate"]
    if fixed or startup["ramp_shape"] == "QUINTIC":
        up = max(startup["minimum_duration_s"], 1.875 * up)
    elif startup["ramp_shape"] != "LINEAR":
        raise ValueError("Unsupported startup ramp")
    down = max(c["shutdown"]["minimum_duration_s"],
               1.875 * last["collective"] / hardware["normal_ramp_rate"])
    transitions = 0.0
    previous_angle = (c["pid"]["zero_angle_deg"] if startup.get("capture_reference")
                      else endpoint(first, end=False))
    previous_input = first.get("fixed_input")
    previous_collective = first["collective"]
    for segment in segments:
        if fixed:
            # Validated production fixed sines have faded endpoints.
            if segment["kind"] == "SINE" and segment.get("fade_s", 0) <= 0:
                raise ValueError("Unfaded fixed sine needs a separate endpoint estimate")
            target = fixed_reference(segment, 0)
            distance = max(abs(float32(sum(m * b for m, b in zip(motor, target))) -
                               float32(sum(m * a for m, a in zip(motor, previous_input))))
                           for motor in hardware["fixed_input_mix"])
            if distance > 1e-8:
                transitions += max(1.0, 1.875 * distance / hardware["normal_ramp_rate"])
            # Fixed playback sends t < duration, NOT the exact endpoint. A small
            # sine tail can still trigger the executor's minimum 1 s transition.
            ticks = math.ceil((segment["duration_s"] - 1e-8) / .01)
            previous_input = fixed_reference(segment, max(0, ticks - 1) * .01)
        else:
            transitions += 1.875 * abs(endpoint(segment, end=False) - previous_angle) / c["transition_rate_deg_s"]
            # Collective change may overlap an angle transition; this conservative
            # estimate counts it separately rather than assuming overlap.
            transitions += abs(segment["collective"] - previous_collective) / hardware["normal_ramp_rate"]
            previous_angle = endpoint(segment, end=True)
        previous_collective = segment["collective"]
    extension = sum(max(0, c["settling"]["maximum_hold_s"] - s["duration_s"])
                    for s in segments if s.get("require_settled"))
    return SessionTime(sum(s["duration_s"] for s in segments), up + down,
                       transitions, .1 + .03 * len(segments), extension)


def nonnegative(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("Must be a finite nonnegative number")
    return number


def measured_overheads(root: Path, plan_hash: str, expected: dict[int, tuple[str, float]]) -> list[float]:
    """Only successful current-version receipts; no old/partial run extrapolation."""
    values = []
    for path in root.glob("*/operator-result.json"):
        try:
            receipt = read_json(path)
            binding_hash, normal_s = expected[receipt["number"]]
            elapsed = receipt["automatic_elapsed_s"]
            if (receipt.get("plan_sha256") == plan_hash and receipt.get("session_sha256") == binding_hash
                    and receipt.get("capture_complete") and receipt.get("log_status") == "SD_CONFIRMED"
                    and type(elapsed) in (float, int) and math.isfinite(elapsed) and elapsed > 0):
                values.append(elapsed - normal_s)
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description="本番所要時間の再計算（接続・回転なし）")
    parser.add_argument("--config", type=Path, default=REPO / "configs/experiment_runner/production.json")
    parser.add_argument("--target-hours", type=nonnegative, default=50)
    parser.add_argument("--overhead-seconds", type=nonnegative, default=25,
                        help="1 runの通信・基準取得・計測起動・終了・回収の仮定秒数")
    parser.add_argument("--settling-seconds", type=nonnegative, default=0,
                        help="1 run平均の追加整定時間（規定HOLDとは別）")
    parser.add_argument("--manual-hours", type=nonnegative, default=6,
                        help="準備・機械/風速変更・電池交換・操作者待ち・再取得等の総予算")
    parser.add_argument("--measured-root", type=Path, help="成功runの実測付帯＋追加整定時間も表示")
    parser.add_argument("--full-plan", action="store_true", help="比較用：除外前の全番号を見積り（接続・回転なし）")
    args = parser.parse_args()
    settings = read_json(args.config)
    path = resolve(settings["plan"])
    plan_hash = sha(path)
    groups = defaultdict(list)
    expected = {}
    selection = current_selection(settings, plan_hash)
    rows = catalog(read_json(path)) if args.full_plan else selection.rows
    for row in rows:
        if args.full_plan:
            # Active bindings may point at shortened sessions. Use the retained
            # immutable 948-session build for a like-for-like source comparison.
            legacy = REPO / "build/production-sessions" / f"{row['number']:04d}.json"
            c = read_json(legacy)
            session_hash = sha(legacy)
        else:
            _, c, _ = session_for(row, settings, plan_hash)
            session_hash = settings["sessions"][str(row["number"])]["sha256"]
        timing = estimate_session(c)
        groups[row["wind_mps"]].append(timing)
        expected[row["number"]] = (session_hash, timing.normal_s)
    count = sum(len(v) for v in groups.values())
    normal = sum(t.normal_s for v in groups.values() for t in v)
    overhead = args.overhead_seconds + args.settling_seconds
    total = (normal + count * overhead) / 3600 + args.manual_hours
    print("風速[m/s]  本数  波形・ランプ・移動[h]  自動処理見積[h]")
    for wind, runs in groups.items():
        seconds = sum(t.normal_s for t in runs)
        print(f"{wind:9g}  {len(runs):4d}  {seconds / 3600:20.2f}  {(seconds + len(runs) * overhead) / 3600:16.2f}")
    print(f"合計 {count} 本 ／ 実行 {normal / 3600:.2f} h ＋ 自動付帯 {count * overhead / 3600:.2f} h"
          f" ＋ 現場作業等 {args.manual_hours:.2f} h ＝ {total:.2f} h")
    print(f"目標 {args.target_hours:g} hまでの余裕: {args.target_hours - total:+.2f} h")
    allowed = ((args.target_hours - args.manual_hours) * 3600 - normal) / count
    print(f"現場作業予算を残すための付帯＋追加整定の上限: 平均 {allowed:.1f} 秒/run")
    print("元計画全反復の比較見積り。" if args.full_plan else
          f"現行選択 {selection.selection_id} ／ 元{selection.source_count}本から{len(selection.omitted)}本を除外。")
    print("付帯時間は未実測の仮定。watchdog上限まで待つ計算ではありません。")
    print("再試行・未整定・通信遅延により超過します。50時間以内の保証ではありません。")
    if args.measured_root:
        observed = measured_overheads(args.measured_root, plan_hash, expected)
        if observed:
            average = sum(observed) / len(observed)
            print(f"現行設定の成功run {len(observed)} 回: 実測付帯＋追加整定の平均 {average:.1f} 秒/run")
            print(f"同じ平均を仮置きした総時間: {(normal + count * average) / 3600 + args.manual_hours:.2f} h")
            print("取得済み条件の標本であり、未実施の風速・軸・再試行を代表する保証はありません。")
        else:
            print("所要時間を記録した現行設定の成功runはまだありません。")
    return 0
