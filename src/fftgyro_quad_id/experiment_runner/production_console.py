"""Number-driven operator console; delegates hardware work to run_sync_bench.

No gain inference or promotion of draft sessions. ULogs stay on the PX4 SD card.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import unicodedata

from .session_capture import acquisition_artifacts_complete, check_capture
from .run_selection import RunSelection, select_runs
from .timing_policy import apply_timing_policy
from . import wind_acquisition

REPO = Path(__file__).resolve().parents[3]
GROUPS = {"B": "基準入力・各モーター差動", "G": "回転域別差動", "A": "差動振幅",
          "M": "複合差動", "X": "固定・入力応答", "FX": "固定・高周波入力",
          "N": "通常の角度往復", "SIDE": "斜め風・角度往復", "RPM": "回転域別の角度往復",
          "E": "角度往復＋独立差動", "F1": "小振幅・高周波", "F2": "振幅拡大",
          "F3": "高速移動・停止", "F": "高速混合", "J": "中心角間の移動",
          "S": "Yaw近傍走査", "W": "Yaw全周走査", "PILOT": "予備試験（本番の取得数に含めない）"}

ANALYSIS_DECISIONS = {
    "ACCEPT_WITH_MASKS",
    "HOLD_LOAD_STATUS",
    "REQUIRE_RERUN",
    "VERIFIED_LATER_ATTEMPT",
}
ANALYSIS_PROGRESS_ACCEPTED = ANALYSIS_DECISIONS - {"REQUIRE_RERUN"}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve(value: str) -> Path:
    p = Path(value).expanduser()
    return p if p.is_absolute() else REPO / p


def catalog(plan: dict) -> list[dict]:
    """Freeze numbering by wind block and original within-block plan order."""
    order = {v: i for i, v in enumerate([0, 2.5, 5, 7.5, 10, 8.75, 3.75])}
    rows = sorted(plan["conditions"], key=lambda r: order[r["wind_mps"]])
    slots = []
    for row in rows:
        for rep in range(1, row["repeats"] + 1):
            slots.append(dict(row, number=len(slots) + 1, repetition=rep))
    return slots


def current_selection(settings: dict, plan_hash: str) -> RunSelection:
    path = resolve(settings["plan"])
    if sha(path) != plan_hash or settings["plan_sha256"] != plan_hash:
        raise ValueError("条件表が変更され、番号対応が一致しません。")
    return select_runs(catalog(read_json(path)), settings.get("execution_selection"), plan_hash, REPO)


def execution_row(row: dict, selection: RunSelection) -> dict:
    """Apply the active timing policy without changing frozen numbering/progress."""
    if row["role"] == "PILOT" or row["number"] > 948:
        return row
    if row["number"] not in selection.omitted:
        return next(r for r in selection.rows if r["number"] == row["number"])
    if row["wind_mps"] == 0:
        return apply_timing_policy(row, selection.timing_policy)
    return row


def load_supplements(settings: dict, plan_hash: str) -> list[dict]:
    """Register pinned extra-number catalogs without changing the frozen 319 selection."""
    rows: list[dict] = []
    bindings: dict[str, dict] = {}
    occupied = {int(number) for number in settings.get("sessions", {})}
    occupied.update(row["number"] for row in settings.get("pilots", []))
    for reference in settings.get("supplement_catalogs", []):
        path = resolve(reference["path"])
        if sha(path) != reference["sha256"]:
            raise ValueError(f"補足条件設定が登録時から変更されています: {path.name}")
        catalog_config = read_json(path)
        # Supplemental numbers are independent of the main-run acquisition
        # subset. Keep their original selection reference as provenance rather
        # than requiring a new catalog release when only that subset changes.
        if catalog_config.get("plan_sha256") != plan_hash:
            raise ValueError(f"補足条件の元計画が一致しません: {path.name}")
        first, last = reference["first_number"], reference["last_number"]
        selected = sorted((row for row in catalog_config.get("pilots", [])
                           if first <= row["number"] <= last), key=lambda row: row["number"])
        if [row["number"] for row in selected] != list(range(first, last + 1)):
            raise ValueError(f"補足番号が欠落または重複しています: {path.name}")
        for row in selected:
            number = row["number"]
            if number in occupied:
                raise ValueError(f"補足番号が既存番号と衝突しています: {number}")
            binding = catalog_config.get("sessions", {}).get(str(number))
            if not binding:
                raise ValueError(f"補足番号のsessionが未登録です: {number}")
            session = resolve(binding["path"])
            if sha(session) != binding["sha256"]:
                raise ValueError(f"補足sessionのhashが一致しません: {number}")
            occupied.add(number)
            bindings[str(number)] = binding
            rows.append(row)
    settings["sessions"] = {**settings.get("sessions", {}), **bindings}
    return rows


def describe(row: dict) -> str:
    axis = {"fixed": "全軸固定", "pitch": "Pitchのみ解放／他軸固定",
            "yaw": "Yawのみ解放／他軸固定", "pitch_yaw": "Pitch・Yaw同時解放／Frame2固定"}[row["axis"]]
    planned = row.get("planned_repeats", row["repeats"])
    details = [f"実験 {row['number']:04d}  {row['id']}  反復 {row['repetition']}/{planned}",
               f"{GROUPS.get(row['group'], row['group'])}  区分={row['role']}",
               f"風速 {row['wind_mps']:g} m/s ／ {axis}",
               f"Pitch条件 {row['pitch_deg']:+g}° ／ 装置方位 {row['heading_deg']:+g}°",
               f"規定波形 {row['nominal_seconds']:.1f} 秒（起動・停止・収録待ちは別）"]
    if "source_nominal_seconds" in row and row["source_nominal_seconds"] != row.get("nominal_seconds"):
        details[-1] += f" ／ 元計画 {row['source_nominal_seconds']:.1f} 秒"
    if row["axis"] == "pitch":
        details.append("Pitch角度はエンコーダー既存ゼロ＝水平0°基準（開始姿勢からの相対角ではありません）")
    elif row["axis"] == "yaw":
        details.append("Yaw角度は固定試験の正面＝0°基準です（開始時は瞬時値を使用・静止不要）")
    if planned != row["repeats"]:
        # Keep immutable source numbering while displaying the effective count.
        details[0] = details[0].rsplit(" ", 1)[0] + (
            f" 1/{planned} (source repetition {row['repetition']})"
        )
        details.append(f"反復削減対象：元計画{row['repeats']}回 → 現行{planned}回（条件単独の試行間ばらつきは未評価）")
    for field, title, unit in [("frequency_hz", "周波数", "Hz"), ("amplitude_deg", "片振幅", "°"),
                               ("rpm_ratio", "基準RPM比", "倍")]:
        if field in row:
            details.append(f"{title}: {row[field]} {unit}")
    return "\n".join(details)


def describe_session(row: dict, session: dict) -> str:
    """Show the registered waveform time while preserving the source condition row."""
    segments = session["segments"]
    seconds = sum(segment["duration_s"] for segment in segments)
    if (len(segments) >= 2 and segments[0]["kind"] == "HOLD" and segments[-1]["kind"] == "HOLD"
            and segments[0]["duration_s"] == segments[-1]["duration_s"] == 1):
        seconds -= 2  # Entry and final hold are not included in catalog nominal time.
    display = dict(row)
    if row["number"] in {374, 377, 386, 452, 757, 758, 763, 765, 833, 913} and abs(seconds - row["nominal_seconds"]) > 0.05:
        display["source_nominal_seconds"] = row["nominal_seconds"]
        display["nominal_seconds"] = seconds
    return describe(display)


def session_for(row: dict, settings: dict, plan_hash: str) -> tuple[Path, dict, int]:
    if settings["plan_sha256"] != plan_hash:
        raise ValueError("条件表が登録時から変わっています。番号対応を再確認してください。")
    binding = settings.get("sessions", {}).get(str(row["number"]))
    if not binding:
        raise ValueError("この番号の実行設定は未登録です。条件別の確認済みJSONを登録してください。")
    path = resolve(binding["path"])
    if sha(path) != binding["sha256"]:
        raise ValueError("実行設定が登録時から変更されています。再登録が必要です。")
    c = read_json(path)
    if c.get("role") != row["role"] or c.get("repetition") != row["repetition"]:
        raise ValueError("設定のT/D/H区分・反復が実験番号と一致しません。")
    if c.get("wind_m_s") != row["wind_mps"] or c.get("heading_deg") != row["heading_deg"]:
        raise ValueError("設定の風速・方位が実験番号と一致しません。")
    if {s.get("condition_id") for s in c["segments"]} != {row["id"]}:
        raise ValueError("設定内の条件IDが実験番号と一致しません。")
    fixed = row["axis"] == "fixed"
    if (c.get("control_mode") in ("FIXED_HOLD", "FIXED_INPUT")) != fixed:
        raise ValueError("固定／動的モードが実験番号と一致しません。")
    if fixed and row["role"] != "PILOT" and c.get("control_mode") != "FIXED_INPUT":
        raise ValueError("正式固定条件にはFIXED_INPUTが必要です。")
    if row["axis"] == "pitch_yaw" and not c.get("coupled_yaw"):
        raise ValueError("連成用executor設定がありません")
    if not fixed and row["axis"] != "pitch_yaw" and c["pid"]["axis"] != row["axis"].upper():
        raise ValueError("PID対象軸が実験番号と一致しません。")
    if row["axis"] in ("fixed", "yaw") and c.get("fixed_pitch_deg") != row["pitch_deg"]:
        raise ValueError("機械固定Pitch角が一致しません。")
    # Source conditions preserve absolute centers and waveform identity; reference
    # capture alone cannot establish a wind-tunnel absolute attitude.
    sources = c.get("source_conditions", [])
    if row["role"] != "PILOT" and sources != [{k: v for k, v in row.items() if k not in ("number", "repetition", "planned_repeats")}]:
        raise ValueError("source_conditionsが正本条件と一致しません。波形生成元を確認してください。")
    seconds = math.ceil(c["maximum_duration_s"] + 40)
    if not 20 <= seconds <= 1800:
        raise ValueError("収録上限1800秒を超えます。条件の分割設定が必要です。")
    executor = settings["coupled_executor"] if row["axis"] == "pitch_yaw" else settings["executor"]
    check_capture(c, executor, seconds, fixed=fixed, axis_ready=not fixed,
                  props_installed=True, operator_ready=True, charged=True)
    return path, c, seconds


def log_verdict(run: dict) -> tuple[str, str]:
    logs = run.get("flightlog", {})
    start, stop = logs.get("start", {}), logs.get("stop", {})
    evidence = (start.get("complete") and start.get("utc_verified") and start.get("logger_running")
                and stop.get("complete") and stop.get("logger_stopped")
                and stop.get("same_log_running_before_stop") and stop.get("sd_log_verified")
                and stop.get("ulog_path") == start.get("ulog_path")
                and any(n > 0 for n in stop.get("written_kib", [])))
    if not evidence:
        return "UNVERIFIED", "PX4ログ：未確認（SD回収後に照合が必要）"
    drops = stop.get("dropouts_reported", [])
    if not drops:
        return "QUALITY_UNKNOWN", "PX4ログ：SD書込確認、dropout情報なし"
    if any(drops):
        return "DROPOUT", f"PX4ログ：SD書込確認、dropout報告 {max(drops)} 件"
    return "SD_CONFIRMED", "PX4ログ：SD書込確認／報告dropout 0（内容検証はSD回収後）"


def capture_complete(out: Path, manifest: dict, returncode: int) -> bool:
    if returncode != 0 or manifest.get("state") != "CAPTURED":
        return False
    if manifest.get("bulk_download_deferred"):
        return manifest.get("remote_artifacts_verified") is True
    return acquisition_artifacts_complete(out)


def failure_detail(out: Path) -> str | None:
    """Return the executor's final failure reason for immediate operator display."""
    try:
        aircraft_lines = (out / "aircraft-process.txt").read_text(
            encoding="utf-8", errors="replace"
        ).splitlines()
    except OSError:
        aircraft_lines = []
    sd_errors = [line.strip().removeprefix("RuntimeError: ") for line in aircraft_lines
                 if "PX4 microSD free space" in line or
                 "PX4 microSD capacity missing or ambiguous" in line]
    if sd_errors:
        return sd_errors[-1]
    try:
        lines = (out / "pid/result.txt").read_text(encoding="utf-8").splitlines()
    except OSError:
        lines = []
    details = [line.removeprefix("detail=") for line in lines if line.startswith("detail=")]
    if details and details[-1]:
        return details[-1]
    try:
        run_error = read_json(out / "run.json").get("error")
    except (OSError, ValueError):
        run_error = None
    if run_error:
        # Coordinator errors may contain an embedded traceback.  The final line
        # is the actionable root cause and fits on the operator screen.
        return str(run_error).strip().splitlines()[-1]
    try:
        start = read_json(out / "flightlog/start.json")
    except (OSError, ValueError):
        return None
    return start.get("error")


def completed(root: Path, plan_hash: str) -> set[int]:
    numbers = set()
    for path in root.glob("*/operator-result.json"):
        try:
            r = read_json(path)
            # ULog content is intentionally verified later during the single SD
            # import.  A complete synchronized capture must advance the console
            # now, while its independent log_status remains visible in metadata.
            if (r.get("plan_sha256") == plan_hash and wind_acquisition.is_main_receipt(r) and
                    (r.get("capture_complete") or acquisition_artifacts_complete(path.parent))):
                numbers.add(r["number"])
        except (ValueError, OSError, KeyError):
            continue
    numbers.update(adjudicated_shutdown_waveforms(root, plan_hash))
    decisions = analysis_adjudications(root, plan_hash)
    numbers.update(number for number, decision in decisions.items()
                   if decision in ANALYSIS_PROGRESS_ACCEPTED)
    # A reviewed REQUIRE_RERUN decision overrides a superficially complete
    # receipt. This is needed when the physical condition was invalidated after
    # capture, while preserving the original attempt and its receipt.
    numbers.difference_update(number for number, decision in decisions.items()
                              if decision == "REQUIRE_RERUN")
    return numbers


def analysis_adjudications(root: Path, plan_hash: str) -> dict[int, str]:
    """Load reviewed analysis decisions without altering original run records.

    Conflicting records fail conservatively: REQUIRE_RERUN wins. Other
    conflicts are ignored for progress instead of silently choosing one
    research judgment.
    """
    grouped: dict[int, set[str]] = {}
    rerun_sources: dict[int, list[str | None]] = {}
    for path in (root / "adjudications").glob("*.json"):
        try:
            record = read_json(path)
            if (record.get("schema") != "fftgyro-analysis-adjudication-v1" or
                    record.get("plan_sha256") != plan_hash):
                continue
            sources = record.get("source_attempts", {})
            for group in record.get("groups", []):
                decision = group.get("decision")
                if decision not in ANALYSIS_DECISIONS:
                    continue
                for raw_number in group.get("numbers", []):
                    number = int(raw_number)
                    if number > 0:
                        grouped.setdefault(number, set()).add(decision)
                        if decision == "REQUIRE_RERUN":
                            rerun_sources.setdefault(number, []).append(
                                sources.get(f"{number:04d}") or sources.get(str(number)))
        except (OSError, ValueError, KeyError, TypeError):
            continue

    decisions: dict[int, str] = {}
    for number, values in grouped.items():
        if "REQUIRE_RERUN" in values:
            requirements = rerun_sources.get(number, [])
            completed_later = []
            for receipt_path in root.glob("*/operator-result.json"):
                try:
                    receipt = read_json(receipt_path)
                    if (receipt.get("plan_sha256") == plan_hash
                            and int(receipt.get("number")) == number
                            and (receipt.get("capture_complete")
                                 or acquisition_artifacts_complete(receipt_path.parent))):
                        completed_later.append(receipt_path.parent.name)
                except (OSError, ValueError, KeyError, TypeError):
                    continue
            if (requirements and all(requirement for requirement in requirements)
                    and completed_later
                    and max(completed_later) > max(requirements)):
                decisions[number] = "VERIFIED_LATER_ATTEMPT"
            else:
                decisions[number] = "REQUIRE_RERUN"
        elif len(values) == 1:
            decisions[number] = next(iter(values))
    return decisions


def print_analysis_status(decisions: dict[int, str]) -> None:
    labels = {
        "REQUIRE_RERUN": "再試験必要",
        "HOLD_LOAD_STATUS": "荷重採用保留（再試験はまだしない）",
        "ACCEPT_WITH_MASKS": "区間除外条件で採用",
        "VERIFIED_LATER_ATTEMPT": "後続attemptで取得済み",
    }
    print("取得済みデータの採否判定:")
    for decision in ("REQUIRE_RERUN", "HOLD_LOAD_STATUS",
                     "ACCEPT_WITH_MASKS", "VERIFIED_LATER_ATTEMPT"):
        numbers = sorted(number for number, value in decisions.items() if value == decision)
        rendered = " ".join(f"{number:04d}" for number in numbers) or "なし"
        print(f"  {labels[decision]}: {rendered}")


def adjudicated_shutdown_waveforms(root: Path, plan_hash: str) -> set[int]:
    """Count user-adjudicated trial-waveform success without rewriting raw receipts.

    This is not a claim that ULog content or measured loads passed quality
    checks. The original INCOMPLETE state and shutdown failure remain in each
    run's immutable capture records.
    """
    numbers: set[int] = set()
    for path in (root / "adjudications").glob("*.json"):
        try:
            record = read_json(path)
            if (record.get("schema") != "fftgyro-shutdown-waveform-adjudication-v1" or
                    record.get("plan_sha256") != plan_hash or
                    record.get("decision") not in {
                        "ACCEPT_WAVEFORM_FOR_SEQUENCE", "ACCEPT_WAVEFORM_SUCCESS"
                    }):
                continue
            for entry in record.get("runs", []):
                run_name = entry.get("run", "")
                if not re.fullmatch(r"\d{8}_\d{6}_e\d{4}_a\d{3}", run_name):
                    continue
                folder = root / run_name
                receipt = read_json(folder / "operator-result.json")
                manifest = read_json(folder / "run.json")
                process = (folder / "aircraft-process.txt").read_text(
                    encoding="utf-8", errors="replace")
                if (receipt.get("plan_sha256") == plan_hash and wind_acquisition.is_main_receipt(receipt) and
                        receipt.get("number") == entry.get("number") and
                        receipt.get("condition_id") == entry.get("condition_id") and
                        receipt.get("capture_complete") is False and
                        receipt.get("returncode") == 1 and
                        manifest.get("run") == run_name and
                        manifest.get("state") == "INCOMPLETE" and
                        "aircraft supervisor transport exited 1" in manifest.get("error", "") and
                        "sustained saturation" in process and
                        "PID stopped, exit=1" in process and
                        (folder / "encoder" / "samples.csv").stat().st_size > 0):
                    numbers.add(receipt["number"])
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return numbers


def reserve(root: Path, number: int) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    prefix = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_e{number:04d}"
    for attempt in range(1, 10000):
        path = root / f"{prefix}_a{attempt:03d}"
        try:
            path.mkdir()
            return path
        except FileExistsError:
            continue
    raise RuntimeError("保存先の採番に失敗しました。")


@contextmanager
def operator_lock(root: Path):
    """OS releases the lock on exit/crash; the tiny marker may remain."""
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".operator.lock").open("a+b") as stream:
        if stream.tell() == 0:
            stream.write(b"0"); stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise ValueError("別の本番操作画面がこの保存先で試験を操作中です。") from error
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def command(settings: dict, row: dict, session: Path, seconds: int, out: Path) -> list[str]:
    args = [sys.executable, "-u", str(REPO / "scripts/run_sync_bench.py"),
            "--data-root", str(out.parent), "--run", out.name, "--prepared-output",
            "--session-config", str(session), "--session-executor",
            settings.get("coupled_executor", settings["executor"]) if row["axis"] == "pitch_yaw" else settings["executor"],
            "--seconds", str(seconds), "--fixed" if row["axis"] == "fixed" else "--axis-ready",
            "--props-installed", "--operator-stop-ready", "--charged-4s"]
    args.extend(["--battery-min-v", str(settings.get("battery_min_v", 14.5)),
                 "--battery-max-v", str(settings.get("battery_max_v", 16.9))])
    args.extend(["--encoder-port", settings.get("encoder_port", "COM5"),
                 "--encoder-baud", str(settings.get("encoder_baud", 576000)),
                 "--encoder-minimum-rate-hz", str(settings.get("encoder_minimum_rate_hz", 80.0)),
                 "--windows-clock-port", str(settings.get("windows_clock_port", 43188))])
    if row["axis"] in ("yaw", "pitch_yaw"):
        calibration = settings["yaw_front_calibration"]
        args.extend(["--yaw-front-calibration", str(resolve(calibration["path"])),
                     "--yaw-front-calibration-sha256", calibration["sha256"]])
    if settings.get("defer_bulk_download", False):
        args.append("--defer-bulk-download")
    if settings.get("retest_context"):
        args.extend(["--retest-context", str(resolve(settings["retest_context"]))])
    if settings.get("qc_profile"):
        args.extend(["--qc-profile", str(resolve(settings["qc_profile"]))])
    if settings.get("require_qc_pass", False):
        args.append("--require-qc-pass")
    for key in ("aircraft_host", "leptrino_host", "aircraft_host_key_alias", "leptrino_host_key_alias",
                "clock_server_ip", "clock_client_visible_ip", "windows_client_visible_ip"):
        args.extend(["--" + key.replace("_", "-"), settings[key]])
    return args


def terminal_line(text: str, columns: int) -> str:
    """Pad by display cells, including Japanese; never wrap the last column."""
    budget = max(1, columns - 1)
    used = 0
    result = []
    for char in text:
        width = 2 if unicodedata.east_asian_width(char) in ("W", "F") else 1
        if used + width > budget:
            break
        result.append(char)
        used += width
    return "\r" + "".join(result) + " " * (budget - used)


def monitor(args: list[str], output: Path) -> int:
    """One updating terminal line, full diagnostic output retained on disk."""
    child_environment = os.environ.copy()
    # The Windows console uses CP932, while the coordinator output is consumed
    # through a pipe.  Make that pipe UTF-8 explicitly instead of relying on the
    # active console code page.
    child_environment["PYTHONUTF8"] = "1"
    process_options = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
                       if os.name == "nt" else {"start_new_session": True})
    proc = subprocess.Popen(args, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="strict", env=child_environment,
                            **process_options)
    lines: queue.Queue = queue.Queue()
    def consume():
        for line in proc.stdout:
            lines.put(line)
        lines.put(None)
    threading.Thread(target=consume, daemon=True).start()
    began = time.monotonic()
    phase = "接続・設定検証"
    battery = ""
    eof = False
    last = ""
    stop_sent = False
    with output.open("x", encoding="utf-8") as log:
        while not eof or proc.poll() is None:
            try:
                line = lines.get(timeout=.25)
                if line is None:
                    eof = True
                else:
                    log.write(line); log.flush()
                    if line.startswith("PROGRESS "):
                        phase = line[len("PROGRESS "):].strip()
                        match = re.search(r"電圧確認\s+([0-9.]+)\s+V", phase)
                        if match:
                            battery = f"電圧 {float(match.group(1)):.3f} V"
                    elif "Force stream active" in line:
                        phase = "力計収録中 → RPM・PX4ログ開始 → モーター試験"
                    elif "Aircraft bench finished" in line:
                        phase = "機体側処理終了／後段収録・保存確認中"
            except queue.Empty:
                pass
            except KeyboardInterrupt:
                # The child owns the ordered remote cleanup.  It is in a separate
                # process group, so explicitly forward one interrupt and then wait
                # for its cleanup result instead of terminating it abruptly.
                if not stop_sent:
                    if os.name == "nt":
                        proc.send_signal(signal.CTRL_BREAK_EVENT)
                    else:
                        os.killpg(proc.pid, signal.SIGINT)
                    stop_sent = True
                phase = "中断要求送信済み／安全停止・ログ終了待ち"
            elapsed = int(time.monotonic() - began)
            shown = phase if not battery or "電圧確認" in phase else f"{phase} | {battery}"
            if sys.stdout.isatty():
                text = f"経過 {elapsed // 60:02d}:{elapsed % 60:02d}  {shown}"
                print(terminal_line(text, shutil.get_terminal_size().columns), end="", flush=True)
            elif shown != last:
                print(shown, flush=True)
            last = shown
    print()
    return proc.wait()


def execute(row: dict, settings: dict, plan_hash: str, root: Path) -> None:
    selection = current_selection(settings, plan_hash)
    row = execution_row(row, selection)
    path, config, seconds = session_for(row, settings, plan_hash)
    wind = wind_acquisition.condition(row, settings)
    description = describe_session(row, config)
    if wind:
        description = description.replace(f"風速 {row['wind_mps']:g} m/s", f"風速 {wind['acquisition_wind_mps']:g} m/s")
    print("\n" + description)
    if wind:
        print(wind_acquisition.label(wind))
        print(f"風速記録参照: {wind['wind_log_reference'] or '未指定（実風速は別途照合）'}")
    values = sorted({s["collective"] for s in config["segments"]})
    print(f"共通入力 {values} ／ 収録の監視上限 {seconds} 秒（正常時は試験終了後に収録も終了）")
    print(f"電圧インターロック {settings.get('battery_min_v', 14.5):g}～"
          f"{settings.get('battery_max_v', 16.9):g} V（開始直前にPX4で自動実測）")
    print(f"実行設定: {path.name}")
    if config.get("coupled_yaw"):
        print("連成制御は実機未確認。Pitch水平0度・Yaw正面0度を同じ観測窓で対応付けます。")
    elif config.get("supplement_condition_status") == "UNMEASURED":
        print("追加条件：既存制御profileは認定済み／この風・姿勢・速度での達成は未確認。実測ログで別判定。")
    if rate_design := config.get("rate_design"):
        print(f"角速度指令ピーク {rate_design['planned_peak_deg_s']}°/s ／ "
              f"現PIDの内部目標上限 {rate_design['controller_target_cap_deg_s']}°/s"
              "（到達保証ではありません）")
    labels = {"HOLD": "保持", "MOVE": "移動", "SINE": "正弦波", "SCAN": "走査"}
    for segment in config["segments"][:8]:
        if config.get("coupled_yaw"):
            yaw = config["coupled_yaw"]["segments"][config["segments"].index(segment)]
            print(f"  {segment['duration_s']:g}秒："
                  f"Pitch {segment['start_deg']:+g}° ±{segment.get('amplitude_deg', 0):g}° / {segment.get('frequency_hz', 0):g}Hz、"
                  f"Yaw {yaw['start_deg']:+g}° ±{yaw.get('amplitude_deg', 0):g}° / {yaw.get('frequency_hz', 0):g}Hz")
        elif config.get("control_mode") == "FIXED_INPUT":
            value = segment["fixed_input"]
            amplitude = segment.get("fixed_amplitude", [0, 0, 0, 0])
            print(f"  {labels.get(segment['kind'], segment['kind'])}: "
                  f"C/P/R/Y={value}  振幅={amplitude}  {segment['duration_s']:g}秒")
        else:
            end = segment.get("end_deg", segment["start_deg"])
            print(f"  {labels.get(segment['kind'], segment['kind'])}: "
                  f"{segment['start_deg']:+g} → {end:+g}°  {segment['duration_s']:g}秒")
    if len(config["segments"]) > 8:
        print(f"  …計{len(config['segments'])}区間（完全な波形は保存設定に記録）")
    print("表示の風速・固定状態を合わせ、LiPo接続・電圧確認・即時停止の準備後、Enterで開始。qで戻る。")
    if input("開始 > ").strip() != "":
        return
    # Recheck after operator preparation; never start with a changed selection.
    current_selection(settings, plan_hash)
    out = reserve(root, row["number"])
    # Snapshot prevents a later edit from changing the displayed command.
    snapshot = out / "operator-session.json"
    snapshot.write_bytes(path.read_bytes())
    if sha(snapshot) != settings["sessions"][str(row["number"])]["sha256"]:
        raise ValueError("開始直前に設定が変わりました。番号を選び直してください。")
    if selection.document_bytes is not None:
        (out / "operator-selection.json").write_bytes(selection.document_bytes)
    receipt = dict(number=row["number"], condition_id=row["id"], repetition=row["repetition"],
                   original_repeats=row["repeats"], planned_repeats=row.get("planned_repeats", row["repeats"]),
                   selection_id=selection.selection_id, selection_sha256=selection.sha256,
                   plan_sha256=plan_hash, session_sha256=sha(snapshot), capture_complete=False,
                   log_status="UNVERIFIED", operator_confirmed_at=datetime.now().isoformat())
    if wind:
        wind["operator_confirmed_at"] = receipt["operator_confirmed_at"]
        receipt["wind_condition"] = wind
        (out / "wind-condition.json").write_text(json.dumps(wind, indent=2), encoding="utf-8")
    (out / "operator-result.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    started = time.monotonic()
    rc = monitor(command(settings, row, snapshot, seconds, out), out / "console.log")
    elapsed = time.monotonic() - started
    manifest = read_json(out / "run.json") if (out / "run.json").exists() else {}
    if wind and (out / "run.json").exists():
        manifest["wind_condition"] = wind
        (out / "run.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    status, message = log_verdict(manifest)
    good = capture_complete(out, manifest, rc)
    receipt.update(capture_complete=good, log_status=status, returncode=rc,
                   automatic_elapsed_s=elapsed,
                   ulog_path=manifest.get("flightlog", {}).get("start", {}).get("ulog_path"))
    (out / "operator-result.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print("収録完了" if good else "未完了：自動再試行はしません。console.logとrun.jsonを確認してください。")
    if good and manifest.get("bulk_download_deferred"):
        print("高速後処理：完了票を確認済み。RPM/PID/Leptrino生データは両PiのSDに保存し、一括回収待ちです。")
    if not good and (detail := failure_detail(out)):
        print(f"中止理由: {detail}")
    print(message)
    preflight = manifest.get("flightlog", {}).get("start", {}).get("battery_preflight", {})
    postflight = manifest.get("flightlog", {}).get("stop", {}).get("battery_postflight", {})
    storage = manifest.get("pi_storage_before", {})
    if all(storage.get(name, {}).get("available_bytes") is not None
           for name in ("aircraft", "leptrino")):
        aircraft_gib = storage["aircraft"]["available_bytes"] / (1024 ** 3)
        leptrino_gib = storage["leptrino"]["available_bytes"] / (1024 ** 3)
        print(f"Pi SD free at start: aircraft {aircraft_gib:.1f} GiB / Leptrino {leptrino_gib:.1f} GiB")
    pixhawk_sd = manifest.get("flightlog", {}).get("start", {}).get("sd_capacity_preflight", {})
    if pixhawk_sd.get("available_reported") is not None:
        print(f"Pixhawk SD free at start: {pixhawk_sd['available_reported']} (MAVLink storage report)")
    if preflight.get("voltage_v") is not None:
        ending = (f" → 終了 {postflight['voltage_v']:.3f} V"
                  if postflight.get("voltage_v") is not None else "")
        print(f"LiPo電圧: 開始 {preflight['voltage_v']:.3f} V{ending}")
    print(f"今回の自動処理: {elapsed / 60:.1f} 分（接続・試験・回収込み／操作者待ちは含まない）")
    print(f"SD上のULog: {receipt['ulog_path'] or '未確認'}")
    print(f"保存先: {out}")


def main() -> int:
    parser = argparse.ArgumentParser(description="本番：実験番号を入力して条件表示・実行・次条件案内")
    parser.add_argument("number", nargs="?", type=int)
    parser.add_argument("--config", type=Path, default=REPO / "configs/experiment_runner/production.json")
    parser.add_argument("--preview", action="store_true", help="接続・回転せず条件だけ表示")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--wind-mps", type=float, choices=(10, 11.5), help="旧10 m/s番号の取得風速を選択")
    parser.add_argument("--wind-scope", choices=("main", "additional_true10", "additional_fresh5"),
                        help="同じ番号を既存計画・追加10・再測定5のどの取得枠へ保存するか")
    parser.add_argument("--wind-log-reference", help="今回の実風速記録ID/パス（自動検証はしません）")
    parser.add_argument("--wind-status", action="store_true", help="10/11.5 m/sの区分別取得状況（接続なし）")
    parser.add_argument("--list-omitted", action="store_true", help="反復削減による実施対象外番号と理由（接続なし）")
    parser.add_argument("--quality-status", action="store_true",
                        help="取得済みデータの採用・保留・再試験判定（接続なし）")
    parser.add_argument("--bind", type=Path, help="指定番号に確認済みsessionを登録（実機接続なし）")
    args = parser.parse_args()
    settings = read_json(args.config)
    original_sessions = dict(settings.get("sessions", {}))
    plan_path = resolve(settings["plan"])
    plan_hash = sha(plan_path)
    try:
        selection = current_selection(settings, plan_hash)
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))
    try:
        supplements = load_supplements(settings, plan_hash)
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))
    original_rows = catalog(read_json(plan_path))
    rows = selection.rows + settings.get("pilots", []) + supplements
    supplement_numbers = {row["number"] for row in supplements}
    if args.list_omitted:
        original = {r["number"]: r for r in catalog(read_json(plan_path))}
        for number, reason in selection.omitted.items():
            print(f"{number:04d}  {original[number]['id']} r{original[number]['repetition']}  対象外：{reason}")
        print(f"実施対象外 {len(selection.omitted)} 本（完了・取得済みではありません）")
        return 0
    # The selection controls the recommended sequence and its progress count,
    # not whether a frozen original experiment number may be run explicitly.
    # This keeps diagnostic/cross-check runs addressable without renumbering or
    # silently adding them back to the active acquisition scope.
    lookup = {r["number"]: r for r in original_rows}
    lookup.update({r["number"]: r for r in rows})
    if args.number is not None and args.number not in lookup:
        parser.error("番号がありません。--listで登録番号を確認してください。")
    if args.bind:
        if args.number is None:
            parser.error("登録対象の実験番号が必要です。")
        if args.number in supplement_numbers:
            parser.error("補足番号の認定設定は元の別カタログで版管理してください。")
        path = args.bind.resolve()
        binding = dict(path=str(path), sha256=sha(path))
        settings.setdefault("sessions", {})[str(args.number)] = binding
        session_for(lookup[args.number], settings, plan_hash)
        # The supplemental bindings are resolved from pinned catalogs, not copied
        # into the frozen production config when a normal session is rebound.
        settings["sessions"] = {**original_sessions, str(args.number): binding}
        args.config.write_text(json.dumps(settings, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"登録完了：実験 {args.number:04d} ／ {path.name}（実機未実行）")
        return 0
    root = resolve(settings["data_root"])
    wind_policy = wind_acquisition.policy(settings)
    if args.wind_mps is not None:
        settings["_wind_mps_override"] = args.wind_mps
    if args.wind_scope is not None:
        settings["_wind_scope_override"] = args.wind_scope
    if args.wind_log_reference:
        settings["_wind_log_reference"] = args.wind_log_reference
    if args.wind_status:
        if wind_policy is None:
            print("風速区分設定なし")
            return 0
        additional_done = wind_acquisition.additional_completed(root, plan_hash, wind_policy)
        fresh5_done = wind_acquisition.additional_completed(
            root, plan_hash, wind_policy, "additional_fresh5")
        legacy_done = completed(root, plan_hash)
        print("番号 / 旧計画枠11.5 m/s（申告・区間未照合） / 追加10 m/s枠")
        for number in wind_policy["additional_numbers"]:
            print(f"{number:04d} / {'収録済' if number in legacy_done else '未収録'} / "
                  f"{'収録済' if number in additional_done else '未収録'}")
        print(f"追加10 m/s: {len(additional_done)}/{len(wind_policy['additional_numbers'])}本収録済。実風速・内容の品質検証は別です。")
        print("\n番号 / 旧5.6 m/s申告枠 / 再測定5.0 m/s枠")
        for number in wind_policy["fresh5_numbers"]:
            print(f"{number:04d} / {'収録済' if number in legacy_done else '未収録'} / "
                  f"{'収録済' if number in fresh5_done else '未収録'}")
        print(f"再測定5.0 m/s: {len(fresh5_done)}/{len(wind_policy['fresh5_numbers'])}本収録済。")
        return 0
    if args.quality_status:
        print_analysis_status(analysis_adjudications(root, plan_hash))
        return 0
    if args.list:
        for r in rows:
            wind = wind_acquisition.condition(r, settings)
            actual = wind['acquisition_wind_mps'] if wind else r['wind_mps']
            extra = "  補足（達成条件はrunで確認）" if r["number"] in supplement_numbers else ""
            print(f"{r['number']:04d}  {r['id']} r{r['repetition']}  {actual:g}m/s  {r['axis']}  Pitch{r['pitch_deg']:+g}°{extra}")
        return 0
    if args.preview:
        row = execution_row(lookup[args.number or 1], selection)
        wind = None
        try:
            path, config, _ = session_for(row, settings, plan_hash)
            wind = wind_acquisition.condition(row, settings)
            description = describe_session(row, config)
            if wind:
                description = description.replace(f"風速 {row['wind_mps']:g} m/s", f"風速 {wind['acquisition_wind_mps']:g} m/s")
            print(description)
            if wind:
                print(wind_acquisition.label(wind))
            print(f"実行設定検証OK: {path}")
            if config.get("supplement_condition_status") == "UNMEASURED":
                print("追加条件の達成は未確認：実角速度・姿勢追従・荷重は取得後に判定します。")
            if rate_design := config.get("rate_design"):
                print(f"角速度指令ピーク {rate_design['planned_peak_deg_s']}°/s ／ "
                      f"現PIDの内部目標上限 {rate_design['controller_target_cap_deg_s']}°/s")
        except (ValueError, OSError, KeyError) as error:
            print(describe(row))
            print(f"実行準備: {error}")
        if wind_policy and wind and wind["progress_scope"] in (
                "additional_true10", "additional_fresh5"):
            order = (wind_policy["additional_numbers"]
                     if wind["progress_scope"] == "additional_true10"
                     else wind_policy["fresh5_numbers"])
            index = order.index(row["number"]) if row["number"] in order else len(order)
            following = lookup[order[index + 1]] if index + 1 < len(order) else None
        else:
            following = next((r for r in selection.rows if r["number"] > row["number"]), None)
        if following:
            following_wind = wind_acquisition.condition(following, settings)
            next_description = describe(following)
            if following_wind:
                next_description = next_description.replace(f"風速 {following['wind_mps']:g} m/s",
                    f"風速 {following_wind['acquisition_wind_mps']:g} m/s")
            print("\n次の番号の条件:\n" + next_description)
        return 0
    print("FFT GYRO 本番操作 ／ PX4ログはSDへ保存し、後で一括回収")
    print(f"現行 {len(selection.rows)} ラン ／ 元計画 {selection.source_count} ラン ／ "
          f"対象外 {len(selection.omitted)} ラン（元番号を維持・条件選択と反復削減を含む）")
    if supplements:
        print(f"追加番号 {len(supplements)} ラン（登録catalogの番号、達成条件はrunで別判定）")
    if wind_policy:
        print("旧5.6 m/s申告枠・再測定5.0 m/s枠・旧11.5 m/s申告枠・追加10 m/s枠を別集計します（--wind-status）。")
    startup_decisions = analysis_adjudications(root, plan_hash)
    rerun_numbers = sorted(number for number, decision in startup_decisions.items()
                           if decision == "REQUIRE_RERUN")
    hold_count = sum(decision == "HOLD_LOAD_STATUS" for decision in startup_decisions.values())
    if rerun_numbers or hold_count:
        rendered = " ".join(f"{number:04d}" for number in rerun_numbers) or "なし"
        print(f"データ精査：再試験 {rendered} ／ 荷重採用保留 {hold_count}本（--quality-statusで詳細）")
    requested = args.number
    while True:
        done = completed(root, plan_hash)
        additional_done = wind_acquisition.additional_completed(root, plan_hash, wind_policy)
        fresh5_done = wind_acquisition.additional_completed(
            root, plan_hash, wind_policy, "additional_fresh5")
        shutdown_adjudicated = adjudicated_shutdown_waveforms(root, plan_hash)
        analysis_decisions = analysis_adjudications(root, plan_hash)
        if all(r["number"] in done for r in selection.rows):
            print(f"現行計画の{len(selection.rows)}番号の収録・SD書込確認が完了しました。"
                  f"除外{len(selection.omitted)}番号は取得済みとは数えません。SD内容の照合は別途必要です。")
        if wind_policy:
            print(f"追加10 m/s枠：{len(additional_done)}/{len(wind_policy['additional_numbers'])}本収録済"
                  "（旧11.5 m/sとは別集計・実風速記録の検証は別）")
            print(f"再測定5.0 m/s枠：{len(fresh5_done)}/{len(wind_policy['fresh5_numbers'])}本収録済")
        try:
            value = str(requested) if requested is not None else input("\n実験番号 / q=終了 > ").strip()
            requested = None
            if value.lower() == "q":
                return 0
            if value.isdigit() and int(value) in selection.omitted:
                print("現行計画の対象外番号を追加試験として実行します。"
                      "保存されますが、現行計画の完了数には加算しません。")
            if not value.isdigit() or int(value) not in lookup:
                print("番号がありません。--listで現行の対象番号を確認してください。予備試験は9001以降です。")
                continue
            row = lookup[int(value)]
            wind = wind_acquisition.condition(row, settings)
            extra_wind = wind and wind["progress_scope"] == "additional_true10"
            extra_fresh5 = wind and wind["progress_scope"] == "additional_fresh5"
            analysis_decision = analysis_decisions.get(row["number"])
            if row["number"] in supplement_numbers:
                # Draft check runs before operator lock, output reservation or SSH.
                session_for(row, settings, plan_hash)
            if extra_wind:
                print("追加10 m/s枠は" + ("収録済みです。別attemptで再収録します。" if row["number"] in additional_done
                      else "未収録です。旧11.5 m/sの完了票・採用判定とは別に保存します。"))
            elif extra_fresh5:
                print("再測定5 m/s枠は" + ("収録済みです。別attemptで再収録します。"
                      if row["number"] in fresh5_done
                      else "未収録です。旧5.6 m/s申告データの完了票・採用判定とは別に保存します。"))
            elif analysis_decision == "REQUIRE_RERUN":
                print("この番号はデータ精査で再試験必要と判定されています。既存attemptは保持したまま、今回を別attemptとして保存します。")
            elif analysis_decision == "HOLD_LOAD_STATUS":
                print("この番号は収録済みですが、Leptrino 0x44の意味が確定するまで荷重データの採用保留です。現時点では再試験しません。")
            elif analysis_decision == "ACCEPT_WITH_MASKS":
                print("この番号は欠測区間を解析から除外する条件で採用済みです。再実行は別attemptです。")
            elif analysis_decision == "VERIFIED_LATER_ATTEMPT":
                print("この番号は後続の正常attemptで取得済みです。再実行は別attemptです。")
            elif row["number"] in done:
                if row["number"] in shutdown_adjudicated:
                    print("この番号は試験波形成功として取得済みです。停止ランプ中止とULog・荷重品質の未照合は記録に残っています。再実行は別attemptです。")
                else:
                    print("この番号は取得済みです。実行すると別attemptとして保存します。")
            with operator_lock(root):
                execute(row, settings, plan_hash, root)
        except (EOFError, KeyboardInterrupt):
            print("\n操作画面を終了します。")
            return 0
        except (ValueError, OSError, KeyError, subprocess.SubprocessError) as error:
            print(f"開始できません／処理未完了: {error}")


if __name__ == "__main__":
    raise SystemExit(main())
