"""Post-stop measurement QC. Standard library only; never commands hardware.

PASS means the checks below passed, not identified physics or safe flight.
Missing evidence is REVIEW, documented corruption/incomplete acquisition is RETRY.
The same functions run on the Pis when bulk downloads are deferred.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
import tempfile
from pathlib import Path

from ..io.leptrino_raw import decode
from ..sync.rpm_pid import fit_clock, read_capture
from .session_capture import shutdown_guard_after_complete_waveform

DEFAULT_POLICY = {
    "max_receive_gap_ms": 100.0,
    "max_clock_bound_ms": 20.0,
    "max_rpm_edge_gap_ms": 500.0,
    "max_saturation_fraction": 0.05,
}


def validate_policy(policy: dict | None) -> dict:
    result = {**DEFAULT_POLICY, **(policy or {})}
    allowed = {
        *DEFAULT_POLICY,
        "max_tracking_error_deg",
        "static_windows_s",
        "max_static_rate_deg_s",
    }
    if set(result) - allowed:
        raise ValueError(f"Unknown QC policy keys: {sorted(set(result) - allowed)}")
    for key, value in result.items():
        if key == "static_windows_s":
            if not isinstance(value, list) or any(
                not isinstance(w, list)
                or len(w) != 2
                or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in w)
                or w[0] < 0
                or w[1] - w[0] < 10
                for w in value
            ):
                raise ValueError("Static windows need [start,end] seconds, length >=10s")
        elif not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"QC limit {key} must be positive and finite")
    if result["max_saturation_fraction"] > 1:
        raise ValueError("Saturation fraction must be <=1")
    if result.get("static_windows_s") and "max_static_rate_deg_s" not in result:
        raise ValueError("Static windows require an explicit max_static_rate_deg_s")
    return result


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def table(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Empty CSV: {path.name}")
    return rows


def timing(rows: list[dict], key: str, max_gap_ms: float) -> dict:
    stamps = [int(row[key]) for row in rows]
    gaps = [b - a for a, b in zip(stamps, stamps[1:], strict=False)]
    if any(g < 0 for g in gaps) or stamps[0] < 0 or stamps[-1] <= stamps[0]:
        raise ValueError("Invalid/reversed/zero-span timestamps")
    return {
        "samples": len(rows),
        "duration_s": (stamps[-1] - stamps[0]) / 1e9,
        "max_gap_ms": max(gaps, default=0) / 1e6,
        "gap_limit_exceeded": max(gaps, default=0) / 1e6 > max_gap_ms,
    }


def inspect(
    root: Path, manifest: dict, components: list[str], policy: dict | None = None
) -> list[dict]:
    """Component reports can be merged across hosts; no cross-clock raw subtraction."""
    limits = validate_policy(policy)
    checks = []

    def check(name, action):
        try:
            decision, metrics, reason = action()
        except FileNotFoundError as error:
            decision, metrics, reason = "REVIEW", {}, f"未取得/未回収: {error.filename}"
        except (OSError, ValueError, KeyError, TypeError, IndexError, csv.Error) as error:
            decision, metrics, reason = "RETRY", {}, f"記録の不整合: {error}"
        checks.append(dict(name=name, decision=decision, metrics=metrics, reason=reason))

    def encoder():
        meta = load(root / "encoder/metadata.json")
        rows = table(root / "encoder/samples.csv")
        metrics = timing(rows, "receive_monotonic_ns", limits["max_receive_gap_ms"])
        for row in rows:
            for axis in ("experimental_pitch_deg", "experimental_yaw_deg"):
                if not math.isfinite(float(row[axis])):
                    raise ValueError("Nonfinite encoder angle")
        if meta.get("state") != "COMPLETE":
            return "RETRY", metrics, "エンコーダー収録未完了"
        rate = (len(rows) - 1) / metrics["duration_s"]
        metrics["rate_hz"] = rate
        if rate < manifest.get("encoder_minimum_rate_hz", 80) or metrics["gap_limit_exceeded"]:
            return "RETRY", metrics, "エンコーダーの低レート/長い欠測"
        return "PASS", metrics, "角度・時刻・レートを確認"

    def stationary():
        windows = limits.get("static_windows_s", [])
        if not windows:
            return "REVIEW", {}, "静止姿勢の区間指定なし。保持成立は未判定"
        rows = table(root / "encoder/samples.csv")
        origin = int(rows[0]["receive_monotonic_ns"])
        results = []
        for begin, end in windows:
            part = [
                r for r in rows if begin <= (int(r["receive_monotonic_ns"]) - origin) / 1e9 <= end
            ]
            if len(part) < 20:
                raise ValueError("Insufficient static-window samples")
            span = (
                int(part[-1]["receive_monotonic_ns"]) - int(part[0]["receive_monotonic_ns"])
            ) / 1e9
            if span < end - begin - 0.1:
                raise ValueError("Incomplete static window")
            # 0.2s differences reduce encoder quantization; wrapped angles are unwrapped.
            metrics = dict(window_s=[begin, end], axes={})
            for axis in ("experimental_pitch_deg", "experimental_yaw_deg"):
                angles = [float(part[0][axis])]
                for row in part[1:]:
                    angles.append(angles[-1] + (float(row[axis]) - angles[-1] + 180) % 360 - 180)
                rates, old = [], 0
                for i in range(1, len(part)):
                    dt = (
                        int(part[i]["receive_monotonic_ns"])
                        - int(part[old]["receive_monotonic_ns"])
                    ) / 1e9
                    if dt >= 0.2:
                        rates.append(abs((angles[i] - angles[old]) / dt))
                        old = i
                metrics["axes"][axis] = dict(
                    mean_deg=statistics.mean(angles),
                    std_deg=statistics.pstdev(angles),
                    max_rate_deg_s=max(rates, default=math.inf),
                )
            results.append(metrics)
        moving = any(
            m["max_rate_deg_s"] > limits["max_static_rate_deg_s"]
            for w in results
            for m in w["axes"].values()
        )
        return (
            ("RETRY" if moving else "PASS"),
            {"windows": results},
            (
                "停止姿勢窓に動きあり"
                if moving
                else "指定停止姿勢窓の静止を確認（CG/零点の識別とは別）"
            ),
        )

    def clock(folder, expected_boot):
        summary = load(root / folder / "summary.json")
        rows = table(root / folder / "exchanges.csv")
        widths = []
        for row in rows:
            low, high = int(row["offset_low_ns"]), int(row["offset_high_ns"])
            if low > high:
                raise ValueError("Reversed clock bounds")
            widths.append((high - low) / 1e6)
        if not expected_boot:
            return "REVIEW", {}, "対応するboot IDなし"
        if summary.get("state") != "COMPLETE" or summary.get("server_boot_id") != expected_boot:
            return "RETRY", {}, "時計収録未完了/boot ID不一致"
        metrics = dict(exchanges=len(rows), max_bound_width_ms=max(widths))
        if len(rows) < 3 or max(widths) > limits["max_clock_bound_ms"]:
            return "REVIEW", metrics, "時計交換が少ない/時刻の不確かさが大きい"
        return "PASS", metrics, "時計交換・boot ID整合（センサ固有遅延の確定ではない）"

    def force():
        summary = load(root / "leptrino/summary.json")
        if summary.get("state") not in ("COMPLETE", "RAW_COMPLETE"):
            return "RETRY", {}, "Leptrino収録未完了"
        rated = summary["rated_limits_N_Nm"]
        if len(rated) != 6 or any(not math.isfinite(float(x)) or x <= 0 for x in rated):
            raise ValueError("Invalid measured sensor ratings")
        # Existing protocol decoder checks framing, START/STOP and truncation.
        # Reserved byte 19 is NOT used as a verified dropped-sample counter.
        with tempfile.TemporaryDirectory(prefix="fftgyro-qc-") as temp:
            metrics = decode(
                root / "leptrino/chunks.bin",
                Path(temp) / "samples.csv",
                tuple(rated),
                int(summary["start_command_before_ns"]),
                counter_jump_limit=0,
            )
            first = last = None
            with (Path(temp) / "samples.csv").open(newline="", encoding="utf-8") as stream:
                for row in csv.DictReader(stream):
                    stamp = int(row["receive_monotonic_ns"])
                    first = stamp if first is None else first
                    last = stamp
            metrics["duration_s"] = (last - first) / 1e9 if first is not None else 0
        metrics.pop("candidate_counter_jumps")
        metrics["reserved_byte_pattern_changes"] = metrics.pop("candidate_counter_jump_count")
        elapsed = float(summary["elapsed_capture_s"])
        if elapsed <= 0 or metrics["duration_s"] < elapsed - 0.5:
            return "RETRY", metrics, "荷重データが収録時間を覆っていない"
        if (
            metrics["documented_status_fault_samples"]
            or max(metrics["max_abs_raw_by_axis"]) > 10000
        ):
            return "RETRY", metrics, "荷重計status異常/定格超過。0x44も正常扱いしない"
        if metrics["max_receive_gap_ms"] > limits["max_receive_gap_ms"]:
            return "RETRY", metrics, "荷重の長い受信欠測"
        return "PASS", metrics, "生6軸・status・定格・フレーム完了を確認"

    def pid():
        config = load(root / "pid/session.config.json")
        result = (root / "pid/result.txt").read_text(encoding="utf-8")
        guard = shutdown_guard_after_complete_waveform(root)
        if "status=COMPLETE" not in result.splitlines() and not guard:
            return "RETRY", {}, "運動/固定波形が未完了"
        rows = table(root / "pid/control.csv")
        metrics = timing(rows, "sent_ns", limits["max_receive_gap_ms"])
        observed = {}
        selected = []
        scientific_segments = {
            s["segment_id"] for s in config["segments"] if not s.get("preparation_only", False)
        }
        for row in rows:
            if row.get("stage") == "SEGMENT":
                observed[row["segment_id"]] = max(
                    observed.get(row["segment_id"], 0), float(row["segment_time_s"])
                )
                if row["segment_id"] in scientific_segments:
                    selected.append(row)
        if not selected:
            raise ValueError("No programmed segment rows")
        for segment in config["segments"]:
            if (
                not segment.get("preparation_only")
                and observed.get(segment["segment_id"], -1) < segment["duration_s"] - 0.05
            ):
                raise ValueError(f"Segment incomplete: {segment['segment_id']}")
        saturated = sum(
            r.get("saturated") == "1" or r.get("authority_limited") == "1" for r in selected
        )
        metrics["saturation_fraction"] = saturated / len(selected)
        for row in selected:
            for j in range(1, 5):
                if not math.isfinite(float(row[f"sent{j}"])):
                    raise ValueError("Nonfinite motor command")
        tracking_limit = limits.get(
            "max_tracking_error_deg", config.get("pid", {}).get("tracking_limit_deg")
        )
        dynamic = config.get("control_mode", "AXIS_PID") != "FIXED_INPUT"
        if dynamic and tracking_limit is not None:
            if not all("angle_deg" in r and "target_deg" in r for r in selected):
                return "REVIEW", metrics, "追従判定用angle_deg/target_degなし"
            errors = [
                abs((float(r["angle_deg"]) - float(r["target_deg"]) + 180) % 360 - 180)
                for r in selected
            ]
            if not all(math.isfinite(e) for e in errors):
                raise ValueError("Nonfinite tracking error")
            metrics["max_tracking_error_deg"] = max(errors)
            if not math.isfinite(float(tracking_limit)) or tracking_limit <= 0:
                raise ValueError("Invalid qualified tracking limit")
            metrics["tracking_limit_deg"] = tracking_limit
            if max(errors) > tracking_limit:
                return "RETRY", metrics, "事前指定した追従誤差を超過"
        if metrics["gap_limit_exceeded"]:
            return "RETRY", metrics, "制御ログに長い欠測"
        if guard or metrics["saturation_fraction"] > limits["max_saturation_fraction"]:
            return "REVIEW", metrics, "波形後の停止guard/入力飽和。自動で再運転しない"
        if dynamic and tracking_limit is None:
            return "REVIEW", metrics, "動的波形完了だが、追従の事前閾値がない"
        return "PASS", metrics, "全指定区間を完了。保持静止/校正成立は別判定"

    def rpm():
        unanswered = []
        events, exchanges = read_capture(root / "rpm/rpm.csv", unanswered=unanswered)
        metrics = dict(events=len(events), exchanges=len(exchanges), unanswered=len(unanswered))
        try:
            fit_clock(exchanges)
            metrics["affine_clock_feasible"] = True
        except ValueError as error:
            metrics["affine_clock_feasible"] = False
            metrics["clock_model_issue"] = str(error)
        if manifest.get("idle"):
            if not metrics["affine_clock_feasible"] or unanswered:
                return "REVIEW", metrics, "停止試験のRPM時計モデル/未応答を確認する必要あり"
            return "PASS", metrics, "停止試験なのでRPMエッジなしを許容"
        rows = [r for r in table(root / "pid/control.csv") if r.get("stage") == "SEGMENT"]
        begin, end = int(rows[0]["sent_ns"]), int(rows[-1]["sent_ns"])
        channel_gaps = {}
        for channel in range(1, 5):
            stamps = sorted(
                e.host_ns
                for e in events
                if e.kind == "E" and e.value == channel and begin <= e.host_ns <= end
            )
            if len(stamps) < 2:
                return "RETRY", metrics, f"RPM{channel}の計測エッジ不足"
            gaps = [
                stamps[0] - begin,
                end - stamps[-1],
                *[b - a for a, b in zip(stamps, stamps[1:], strict=False)],
            ]
            channel_gaps[str(channel)] = max(gaps) / 1e6
        metrics["max_channel_gap_ms"] = channel_gaps
        if max(channel_gaps.values()) > limits["max_rpm_edge_gap_ms"]:
            return "RETRY", metrics, "本区間でRPMチャンネル欠測"
        if unanswered or not metrics["affine_clock_feasible"]:
            return "REVIEW", metrics, "RPM時計応答未回収/アフィン時計仮定が不成立"
        return "PASS", metrics, "4チャンネルの本区間エッジとNano時計モデルを確認"

    def flightlog():
        stop = load(root / "flightlog/stop.json")
        if not stop.get("sd_log_verified") or not stop.get("logger_stopped"):
            return "RETRY", {}, "PX4 SDログ保存/停止が確認できない"
        drops = stop.get("dropouts_reported")
        if drops is None:
            return "REVIEW", {}, "PX4 dropout情報なし（ULog内容は別途確認）"
        if any(drops):
            return "REVIEW", {"dropouts_reported": drops}, "PX4 dropoutあり。欠測位置を確認"
        return (
            "PASS",
            {"sd_file_bytes": stop.get("sd_file_bytes")},
            "SD保存を確認。ULog本文は未検証",
        )

    actions = {
        "encoder": encoder,
        "stationary": stationary,
        "force": force,
        "pid": pid,
        "rpm": rpm,
        "clock": lambda: clock("clock", manifest.get("aircraft_boot_id")),
        "encoder-clock": lambda: clock("encoder-clock", manifest.get("aircraft_boot_id")),
        "flightlog": flightlog,
    }
    for component in components:
        check(component, actions[component])
    return checks


def summarize(checks: list[dict], policy: dict | None = None) -> dict:
    decision = (
        "RETRY"
        if any(c["decision"] == "RETRY" for c in checks)
        else ("REVIEW" if not checks or any(c["decision"] != "PASS" for c in checks) else "PASS")
    )
    return dict(
        schema="fftgyro-immediate-qc-v1",
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        decision=decision,
        checks=checks,
        policy=validate_policy(policy),
        physics_validated=False,
        automatic_motor_retry=False,
        scope="計測・波形の成立。零点/CG識別、空力モデル、実飛行の成立は別評価",
    )


def render(report: dict) -> str:
    label = {"PASS": "計測QC合格", "REVIEW": "要確認", "RETRY": "再取得候補"}
    lines = [f"試験直後判定: {label[report['decision']]}"]
    for item in report["checks"]:
        lines.append(f"  [{item['decision']}] {item['name']}: {item['reason']}")
    lines.append("※零点・重心の識別成功/モデル精度の合格ではありません。自動再運転しません。")
    return "\n".join(lines)
