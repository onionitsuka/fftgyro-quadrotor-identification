"""Windows SSH coordinator for idle, props-off, or qualified fixed/dynamic sessions.

No arming or PX4 parameter changes. Pitch dynamic sessions pair the existing
horizontal encoder zero with PX4 attitude before motor start.
Every motor trial sets/verifies PX4 UTC before starting its standard ULog.
Requires explicit fresh operator confirmations. Raw data must be outside the repository.
"""
import argparse
import base64
import csv
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import re
import signal
import shutil
import subprocess
import sys
import tempfile
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fftgyro_quad_id.experiment_runner.session_capture import acquisition_artifacts_complete, check_capture
from fftgyro_quad_id.experiment_runner.immediate_quality import (
    inspect as inspect_quality, render as render_quality, summarize as summarize_quality,
    validate_policy,
)
from fftgyro_quad_id.experiment_runner.coupled_reference import resolve_coupled_anchor
from fftgyro_quad_id.experiment_runner.retest_reference import prepare_context
from fftgyro_quad_id.experiment_runner.empty_sensor_decay import (
    ManualDecayGuide, STAGES as EMPTY_SENSOR_STAGES, STATIC_STAGE as EMPTY_STATIC_STAGE,
    validate_condition as validate_empty_condition,
)
from fftgyro_quad_id.experiment_runner.pitch_reference import (
    needs_pitch_anchor, needs_yaw_anchor, resolve_pitch_anchor, resolve_yaw_anchor,
)

AIR = "fftgyro@192.0.2.195"
LEP = "fftgyro@192.0.2.80"
# ConnectTimeout protects setup.  Do not use OpenSSH ServerAlive during a run:
# the experiment has its own bounded deadline and remote RPM/controller watchdogs,
# while a false ServerAlive failure would tear down an otherwise healthy capture.
OPTIONS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=8"]
HOST_KEY_ALIASES = {LEP: "leptrino-pi.local"}
CLOCK_PRE_ROLL_S = 1
RPM_READY_HOLD_S = 1
RPM_POST_ROLL_S = 0.25
FORCE_POST_ROLL_S = 0.25
ENCODER_POST_ROLL_S = 0.25
# Human-facing angle/target telemetry only.  The controller and CSV logger run
# independently at their native rate; 5 Hz keeps the terminal responsive
# without adding SSH sessions or changing the commanded waveform.
AIRCRAFT_PROGRESS_INTERVAL_S = 0.2
MIN_PI_STORAGE_FREE_BYTES = 1024 ** 3
REMOTE_COMPLETION_POLL_S = 1
SFTP_ATTEMPTS = 3
SFTP_RETRY_DELAY_S = 1
_EMERGENCY_CLEANUP = None
CONSOLE_RESERVATION_FILES = frozenset(
    {
        "operator-result.json",
        "operator-selection.json",
        "operator-session.json",
        "console.log",
        "wind-condition.json",
    }
)


def remote_quality(host, base, manifest, components, policy):
    """Read remote raw data after closure; return only QC metrics, not bulk files."""
    source_root = Path(__file__).resolve().parents[1] / "src" / "fftgyro_quad_id"
    names = (
        "io/leptrino.py", "io/leptrino_raw.py", "sync/rpm_pid.py",
        "experiment_runner/session_capture.py", "experiment_runner/immediate_quality.py",
    )
    files = {name: base64.b64encode((source_root / name).read_bytes()).decode() for name in names}
    payload = base64.b64encode(json.dumps(dict(
        files=files, manifest=manifest, components=components, policy=policy, root=base,
    )).encode()).decode()
    program = f"""
import base64,json,sys,tempfile
from pathlib import Path
payload=json.loads(base64.b64decode({payload!r}))
with tempfile.TemporaryDirectory(prefix='fftgyro-qc-code-') as directory:
 root=Path(directory)
 for name in ('', 'io', 'sync', 'experiment_runner'):
  folder=root/'fftgyro_quad_id'/name
  folder.mkdir(parents=True,exist_ok=True)
  (folder/'__init__.py').write_text('')
 for name,encoded in payload['files'].items():
  (root/'fftgyro_quad_id'/name).write_bytes(base64.b64decode(encoded))
 sys.path.insert(0,str(root))
 from fftgyro_quad_id.experiment_runner.immediate_quality import inspect
 print(json.dumps(inspect(Path(payload['root']),payload['manifest'],payload['components'],payload['policy'])))
"""
    completed = subprocess.run(
        remote(host, "python3 -"), input=program, text=True, capture_output=True,
        check=True, timeout=120,
    )
    checks = json.loads(completed.stdout)
    if (not isinstance(checks, list) or len(checks) != len(components)
            or {c.get("name") for c in checks} != set(components)
            or any(c.get("decision") not in ("PASS", "REVIEW", "RETRY") for c in checks)):
        raise ValueError("Incomplete/malformed remote QC response")
    return checks


def immediate_quality(out, manifest, policy):
    local = ["encoder", "encoder-clock"]
    if policy.get("static_windows_s") or manifest.get("qc_stationary_required"):
        local.append("stationary")
    aircraft = ["rpm"]
    if not manifest.get("idle"):
        aircraft.append("pid")
    if not manifest.get("idle") or manifest.get("idle_px4_log"):
        aircraft.append("flightlog")
    checks = inspect_quality(out, manifest, local, policy)
    if manifest.get("bulk_download_deferred"):
        with ThreadPoolExecutor(max_workers=2) as pool:
            pending = {
                pool.submit(remote_quality, host, manifest["remote_base"], manifest, parts, policy): parts
                for host, parts in ((AIR, aircraft), (LEP, ["force", "clock"]))
            }
            for future, parts in pending.items():
                try:
                    checks.extend(future.result())
                except (OSError, ValueError, subprocess.SubprocessError) as error:
                    checks.extend(dict(name=part, decision="REVIEW", metrics={},
                                       reason=f"Pi側QCを回収できない: {type(error).__name__}") for part in parts)
    else:
        checks.extend(inspect_quality(out, manifest, [*aircraft, "force", "clock"], policy))
    checks.append(dict(name="acquisition", decision="PASS" if manifest.get("state") == "CAPTURED" else "RETRY",
                       metrics={}, reason=f"収録状態: {manifest.get('state')}"))
    return summarize_quality(checks, policy)


def unused_console_reservation(path: Path) -> bool:
    """Return true only for a console-created directory not yet used by capture."""
    return (
        path.is_dir()
        and not ({entry.name for entry in path.iterdir()} - CONSOLE_RESERVATION_FILES)
        and (path / "operator-result.json").is_file()
    )

REFERENCE_CAPTURE_SCRIPT = r'''import json
import math
import time
from pymavlink import mavutil

# pymavlink 2.4.49 can cache a plain message with _instances=None before
# receiving the instance-keyed form of the same PX4 message type.  Repair only
# that stale cache entry before the stock helper processes the new message.
_original_add_message = mavutil.add_message
def _add_message(messages, message_type, message):
    field = getattr(message, "_instance_field", None)
    value = getattr(message, field, None) if field else None
    cached = messages.get(message_type)
    if value is not None and cached is not None and getattr(cached, "_instances", None) is None:
        del messages[message_type]
    return _original_add_message(messages, message_type, message)
mavutil.add_message = _add_message

link = mavutil.mavlink_connection("/dev/serial0", baud=921600,
                                  source_system=246, source_component=191)
samples = []
try:
    heartbeat = link.wait_heartbeat(timeout=5)
    if heartbeat is None or heartbeat.get_srcSystem() != 1:
        raise RuntimeError("PX4 heartbeat missing during reference capture")
    if heartbeat.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED:
        raise RuntimeError("PX4 armed during reference capture")
    link.mav.command_long_send(1, 1, mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
                               0, 31, 10000, 0, 0, 0, 0, 0)
    end = time.monotonic() + 5
    while time.monotonic() < end:
        message = link.recv_match(blocking=True, timeout=.2)
        if not message or message.get_srcSystem() != 1 or message.get_srcComponent() != 1:
            continue
        if (message.get_type() == "HEARTBEAT" and
                message.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED):
            raise RuntimeError("PX4 became armed during reference capture")
        if message.get_type() == "ATTITUDE_QUATERNION":
            samples.append({"host_ns": time.monotonic_ns(), **message.to_dict()})
    if len(samples) < 20:
        raise RuntimeError("insufficient reference samples")
    quaternions = []
    for sample in samples:
        quaternion = [sample[f"q{i}"] for i in range(1, 5)]
        norm = math.sqrt(sum(value * value for value in quaternion))
        if not .9 < norm < 1.1:
            raise RuntimeError("invalid reference quaternion")
        quaternion = [value / norm for value in quaternion]
        if quaternions and sum(a * b for a, b in zip(quaternion, quaternions[0])) < 0:
            quaternion = [-value for value in quaternion]
        quaternions.append(quaternion)
    mean = [sum(q[index] for q in quaternions) / len(quaternions) for index in range(4)]
    norm = math.sqrt(sum(value * value for value in mean))
    mean = [value / norm for value in mean]
    summary = {
        "quaternion_wxyz": mean,
        "samples": len(samples),
        "armed": False,
        "max_reference_deviation_deg": max(
            math.degrees(2 * math.acos(min(1, abs(sum(a * b for a, b in zip(q, mean))))))
            for q in quaternions),
        "max_body_rate_deg_s": [
            max(abs(math.degrees(sample[key])) for sample in samples)
            for key in ("rollspeed", "pitchspeed", "yawspeed")],
        "calibration_changed": False,
        "motor_commands_sent": False,
        "reference_scope": "Current resting pose for this run; native stationary preflight still required"
    }
    print(json.dumps({"summary": summary, "raw_samples": samples}), flush=True)
finally:
    link.close()
'''

# Yaw control uses the pose at the end of the capture as its per-run zero.  A
# long stationary average is unnecessary and makes wind-tunnel operation more
# vulnerable to Wi-Fi/SSH stalls.
YAW_REFERENCE_CAPTURE_SCRIPT = (
    REFERENCE_CAPTURE_SCRIPT
    .replace("end = time.monotonic() + 5", "end = time.monotonic() + 1")
    .replace("if len(samples) < 20:", "if len(samples) < 5:")
)


def transport_options(host):
    options = list(OPTIONS)
    alias = HOST_KEY_ALIASES.get(host)
    if alias:
        options.extend(("-o", f"HostKeyAlias={alias}"))
    return options


def remote(host, command):
    return ["ssh", *transport_options(host), host, command]


def request_remote_stop(host, stop_path, attempts=2):
    """Create one owned stop marker, retrying transient SSH transport stalls."""
    if (not re.fullmatch(r"/[A-Za-z0-9_./-]+", stop_path) or
            ".." in stop_path.split("/") or attempts < 1):
        raise ValueError("invalid remote stop request")
    last_error = None
    for attempt in range(attempts):
        try:
            return subprocess.run(remote(host, f"touch {stop_path}"), check=True, timeout=8,
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as error:
            last_error = error
            if attempt + 1 < attempts:
                time.sleep(.25)
    raise last_error


def remote_file_exists(host: str, path: str, timeout: float = 5) -> bool:
    """Probe one absolute owned marker without trusting the long-lived SSH channel."""
    if not re.fullmatch(r"/[A-Za-z0-9_./-]+", path) or ".." in path.split("/"):
        raise ValueError("invalid remote marker path")
    try:
        result = subprocess.run(remote(host, f"test -f {path}"), timeout=timeout,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def collect_sftp_artifacts(host: str, base: str, names: tuple[str, ...], out: Path,
                           attempts: int = SFTP_ATTEMPTS) -> None:
    """Download one Pi's exact artifacts via a retried, all-or-nothing staging area."""
    if attempts < 1 or not names:
        raise ValueError("at least one SFTP attempt and artifact name are required")
    destinations = [out / name for name in names]
    existing = [path for path in destinations if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite collected artifact: {existing[0]}")
    label = "機体Pi" if host == AIR else "Leptrino Pi"
    last_error = None
    for attempt in range(1, attempts + 1):
        staging = Path(tempfile.mkdtemp(prefix="fftgyro-sftp-"))
        try:
            print(f"SFTP回収: {label} ({attempt}/{attempts})", flush=True)
            batch = "".join(f'get -R "{base}/{name}" "{name}"\n' for name in names)
            subprocess.run(["sftp", "-b", "-", *transport_options(host), host], input=batch,
                           text=True, cwd=staging, check=True, timeout=180,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            missing = [name for name in names if not (staging / name).exists()]
            if missing:
                raise RuntimeError(f"SFTP completed without expected artifacts: {missing}")
            for name, destination in zip(names, destinations):
                (staging / name).replace(destination)
            return
        except (OSError, RuntimeError, subprocess.CalledProcessError,
                subprocess.TimeoutExpired) as error:
            last_error = error
            if attempt < attempts:
                print(f"SFTP一時失敗: {label}。{SFTP_RETRY_DELAY_S}秒後に再試行します。",
                      flush=True)
                time.sleep(SFTP_RETRY_DELAY_S)
        finally:
            shutil.rmtree(staging, ignore_errors=True)
    raise last_error


def collect_sftp_receipts(host: str, base: str, names: dict[str, str], out: Path,
                          attempts: int = 2) -> None:
    """Fetch only small completion records; bulk streams remain verified on Pi SD."""
    if attempts < 1 or not names:
        raise ValueError("at least one receipt and SFTP attempt are required")
    if any(not re.fullmatch(r"[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*", name)
           for name in names):
        raise ValueError("invalid remote receipt path")
    destination = out / "remote-receipts"
    destination.mkdir(exist_ok=True)
    if any((destination / local).exists() for local in names.values()):
        raise FileExistsError("refusing to overwrite a remote completion receipt")
    last_error = None
    for attempt in range(1, attempts + 1):
        staging = Path(tempfile.mkdtemp(prefix="fftgyro-receipts-"))
        try:
            for local in names.values():
                (staging / local).parent.mkdir(parents=True, exist_ok=True)
            batch = "".join(
                f'get "{base}/{remote_name}" "{local_name}"\n'
                for remote_name, local_name in names.items()
            )
            subprocess.run(["sftp", "-b", "-", *transport_options(host), host], input=batch,
                           text=True, cwd=staging, check=True, timeout=20,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            for local in names.values():
                source = staging / local
                if not source.is_file():
                    raise RuntimeError(f"SFTP completed without receipt: {local}")
            for local in names.values():
                target = destination / local
                target.parent.mkdir(parents=True, exist_ok=True)
                (staging / local).replace(target)
            return
        except (OSError, RuntimeError, subprocess.CalledProcessError,
                subprocess.TimeoutExpired) as error:
            last_error = error
            if attempt < attempts:
                time.sleep(.25)
        finally:
            shutil.rmtree(staging, ignore_errors=True)
    raise last_error


def read_boots():
    """Independent read-only checks; neither Pi waits for the other handshake."""
    def read(host):
        return subprocess.check_output(remote(host, "cat /proc/sys/kernel/random/boot_id"),
                                       timeout=15, text=True).strip()
    with ThreadPoolExecutor(max_workers=2) as pool:
        values = list(pool.map(read, (AIR, LEP)))
    return dict(zip(("aircraft_boot_id", "leptrino_boot_id"), values))


def read_storage_capacity():
    """Read each Pi data filesystem's capacity without transferring run data."""
    data_path = "/home/fftgyro/research-data/fftgyro"

    def read(host):
        output = subprocess.check_output(
            remote(host, f"df -Pk {data_path} | tail -n 1"),
            timeout=15,
            text=True,
        ).strip()
        fields = output.split()
        if len(fields) < 6:
            raise RuntimeError(f"invalid df response from {host}: {output!r}")
        try:
            total_bytes = int(fields[1]) * 1024
            available_bytes = int(fields[3]) * 1024
        except ValueError as error:
            raise RuntimeError(f"invalid df values from {host}: {output!r}") from error
        if total_bytes <= 0 or available_bytes < 0:
            raise RuntimeError(f"invalid filesystem capacity from {host}: {output!r}")
        return {
            "path": data_path,
            "filesystem": fields[0],
            "total_bytes": total_bytes,
            "available_bytes": available_bytes,
            "available_percent": 100.0 * available_bytes / total_bytes,
        }

    with ThreadPoolExecutor(max_workers=2) as pool:
        values = list(pool.map(read, (AIR, LEP)))
    return dict(zip(("aircraft", "leptrino"), values))


def storage_progress(storage):
    def free_gib(name):
        return storage[name]["available_bytes"] / (1024 ** 3)

    return (
        f"Pi SD free: aircraft {free_gib('aircraft'):.1f} GiB / "
        f"Leptrino {free_gib('leptrino'):.1f} GiB"
    )


def storage_below_limit(storage, minimum_bytes=MIN_PI_STORAGE_FREE_BYTES):
    """Return Pi names whose data filesystem has at most the reserved free space."""
    return [
        name for name in ("aircraft", "leptrino")
        if storage[name]["available_bytes"] <= minimum_bytes
    ]


def wait_clock_ready(process, log_path, timeout=30):
    """Wait for actual remote bind, not a fixed sleep after starting SSH."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("clock responder exited before ready")
        if "Temporary clock responder ready" in log_path.read_text(encoding="utf-8"):
            return
        time.sleep(0.1)
    raise RuntimeError("clock responder readiness timeout; no acquisition started")


def wait_local_ready(process, ready_path: Path, log_path: Path, label: str, timeout=15):
    """Require a validated local stream/exchange before any motor command can start."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            detail = log_path.read_text(encoding="utf-8", errors="replace").strip()
            raise RuntimeError(f"{label} exited before ready: {detail[-1000:]}")
        if ready_path.is_file():
            return
        time.sleep(.05)
    raise RuntimeError(f"{label} readiness timeout; no motor start")


def _main():
    global AIR, LEP, HOST_KEY_ALIASES, _EMERGENCY_CLEANUP
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--prepared-output", action="store_true", help="Use a console-reserved output directory")
    parser.add_argument("--idle", action="store_true",help="No PID or motor command")
    parser.add_argument(
        "--idle-px4-log",
        action="store_true",
        help="With --idle, record a normal PX4 ULog without arming or commanding motors",
    )
    parser.add_argument("--session-props-off", action="store_true", help="Bounded production SessionRunner hold")
    parser.add_argument("--fixed", action="store_true")
    parser.add_argument("--session-config", type=Path, help="Separately qualified fixed/dynamic session; never promotes drafts")
    parser.add_argument("--session-executor", default="", help="Explicit separate Pi executable for --session-config")
    parser.add_argument("--axis-ready", action="store_true", help="Only the selected dynamic axis is free; other axes locked")
    parser.add_argument("--props-installed", action="store_true")
    parser.add_argument("--fixed-commissioning", action="store_true", help="Bounded unqualified fixed hold only")
    parser.add_argument("--pitch-hold-commissioning", action="store_true", help="Initial bounded Pitch PD hold only")
    parser.add_argument("--yaw-hold-commissioning", action="store_true", help="Initial bounded Yaw cascade hold only")
    parser.add_argument("--seconds",type=int,default=80,help="Capture watchdog ceiling; 20..1800 s (idle: requested duration)")
    parser.add_argument("--aircraft-host", default=AIR)
    parser.add_argument("--leptrino-host", default=LEP)
    parser.add_argument("--aircraft-host-key-alias", default="fftgyro-pi.local")
    parser.add_argument("--leptrino-host-key-alias", default="leptrino-pi.local")
    parser.add_argument(
        "--clock-server-ip",
        default="192.0.2.195",
        help="Aircraft Pi IPv4 address used for clock-server bind and Leptrino probe",
    )
    parser.add_argument(
        "--clock-client-visible-ip",
        default="192.0.2.80",
        help="Leptrino probe source IPv4 as visible to the aircraft clock server",
    )
    parser.add_argument("--encoder-port", default="COM5")
    parser.add_argument("--encoder-baud", type=int, default=576000, choices=(57600, 576000))
    parser.add_argument("--encoder-minimum-rate-hz", type=float, default=80.0)
    parser.add_argument("--yaw-front-calibration", type=Path)
    parser.add_argument("--yaw-front-calibration-sha256", default="")
    parser.add_argument("--windows-client-visible-ip", default="192.0.2.1")
    parser.add_argument("--windows-clock-port", type=int, default=43188)
    parser.add_argument("--battery-min-v", type=float, default=14.5)
    parser.add_argument("--battery-max-v", type=float, default=16.9)
    parser.add_argument("--defer-bulk-download", action="store_true",
                        help="Verify remote persistence now and download bulk streams later")
    parser.add_argument("--qc-profile", type=Path, help="取得前に固定したQC閾値・停止姿勢窓JSON")
    parser.add_argument("--qc-stationary-required", action="store_true",
                        help="停止校正等で保持窓の静止判定を必須化。窓未指定は要確認")
    parser.add_argument("--require-qc-pass", action="store_true",
                        help="収録完了とは別に、QC要確認/再取得候補を非0終了コードにする")
    parser.add_argument("--retest-context", type=Path,
                        help="再試験の構成・前校正・A段階JSON。不一致は接続前に停止")
    parser.add_argument("--empty-sensor-condition",
                        help="空のLeptrinoの姿勢・静止/減衰・反復メタデータJSON。公称角を実角にしない")
    for flag in ("props-removed", "operator-stop-ready", "charged-4s"):
        parser.add_argument("--"+flag, action="store_true")
    args = parser.parse_args()
    try:
        qc_policy = validate_policy(json.loads(args.qc_profile.read_text(encoding="utf-8"))
                                    if args.qc_profile else None)
    except (OSError, ValueError) as error:
        parser.error(f"Invalid QC profile: {error}")
    if not (math.isfinite(args.battery_min_v) and math.isfinite(args.battery_max_v) and
            0 < args.battery_min_v < args.battery_max_v):
        parser.error("battery voltage limits must satisfy 0 < min < max")
    if not math.isfinite(args.encoder_minimum_rate_hz) or not 1 <= args.encoder_minimum_rate_hz <= 200:
        parser.error("encoder minimum rate must be in [1, 200] Hz")
    if not 1024 <= args.windows_clock_port <= 65535:
        parser.error("windows clock port must be in [1024, 65535]")
    if args.idle_px4_log and not args.idle:
        parser.error("--idle-px4-log requires --idle")
    AIR = args.aircraft_host
    LEP = args.leptrino_host
    HOST_KEY_ALIASES = {
        AIR: args.aircraft_host_key_alias,
        LEP: args.leptrino_host_key_alias,
    }
    if not re.fullmatch(r"[a-z0-9_-]{1,80}", args.run):
        parser.error("invalid run name")
    if not 20 <= args.seconds <= 1800:
        parser.error("duration must be 20..1800 s")
    source_session_text = args.session_config.read_text(encoding="utf-8") if args.session_config else None
    session_text = source_session_text
    session_limit = 0.0
    expected_session_seconds = 0.0
    if session_text is not None:
        if args.idle or args.session_props_off:
            parser.error("session-config cannot be combined with idle/session-props-off")
        try:
            session_limit = check_capture(json.loads(session_text), args.session_executor, args.seconds,
                fixed=args.fixed, axis_ready=args.axis_ready, props_installed=args.props_installed,
                operator_ready=args.operator_stop_ready, charged=args.charged_4s,
                props_removed=args.props_removed,
                fixed_commissioning=args.fixed_commissioning,
                pitch_commissioning=args.pitch_hold_commissioning,
                yaw_commissioning=args.yaw_hold_commissioning)
            expected_session_seconds = float(
                json.loads(session_text).get("nominal_condition_seconds", session_limit)
            ) + 30.0
        except (ValueError, KeyError, TypeError) as error:
            parser.error(str(error))
    source_session = json.loads(session_text) if session_text is not None else None
    retest_context = None
    if args.retest_context:
        try:
            retest_context = prepare_context(
                args.retest_context, idle=args.idle, fixed=args.fixed,
                stationary=args.qc_stationary_required, px4_log=args.idle_px4_log,
                session=source_session,
            )
            if args.qc_profile is None:
                raise ValueError("retest requires a predeclared --qc-profile")
            args.require_qc_pass = True
        except (OSError, ValueError, TypeError, KeyError) as error:
            parser.error(f"Retest reference preflight: {error}")
        print(f"PROGRESS 再試験 {retest_context['stage']}／構成 {retest_context['configuration_id']}"
              "／前校正・順序確認（絶対値の妥当性は別判定）", flush=True)
    empty_condition = None
    if args.empty_sensor_condition:
        try:
            empty_condition = json.loads(args.empty_sensor_condition)
            validate_empty_condition(empty_condition, retest_context)
        except (ValueError, TypeError, KeyError) as error:
            parser.error(f"Empty sensor condition: {error}")
    elif retest_context and retest_context["stage"] in EMPTY_SENSOR_STAGES:
        parser.error("empty-sensor recording requires --empty-sensor-condition metadata")
    yaw_front_calibration = None
    if source_session is not None and (needs_yaw_anchor(source_session) or source_session.get("coupled_yaw")):
        if args.yaw_front_calibration is None or not re.fullmatch(r"[0-9a-f]{64}", args.yaw_front_calibration_sha256):
            parser.error("Yaw AXIS_PID requires a SHA-pinned fixed-test encoder front calibration")
        calibration_bytes = args.yaw_front_calibration.read_bytes()
        if hashlib.sha256(calibration_bytes).hexdigest() != args.yaw_front_calibration_sha256:
            parser.error("Yaw front calibration hash mismatch")
        yaw_front_calibration = json.loads(calibration_bytes)
        if yaw_front_calibration.get("schema") != "fftgyro-yaw-front-encoder-calibration-v1":
            parser.error("invalid Yaw front calibration schema")
    if source_session is None and (args.axis_ready or args.session_executor or args.fixed_commissioning or args.pitch_hold_commissioning or args.yaw_hold_commissioning or (args.props_installed and not args.idle)):
        parser.error("axis-ready/session-executor require session-config; props-installed also permits idle")
    if args.props_installed and args.props_removed:
        parser.error("props-installed and props-removed are mutually exclusive")
    if not args.idle and session_text is None and not (args.props_removed and args.operator_stop_ready and args.charged_4s):
        parser.error("motor bench requires all three fresh operator confirmations")
    if args.session_props_off and (args.idle or not args.fixed):
        parser.error("session props-off requires fixed rig and cannot be idle")
    seconds=args.seconds
    out = args.data_root/args.run
    if args.prepared_output:
        if not unused_console_reservation(out):
            parser.error("prepared output is not an unused console reservation")
    else:
        out.mkdir(parents=True, exist_ok=False)
    print("PROGRESS 接続・設定検証", flush=True)
    reference = None
    native_reference = bool(session_text and json.loads(session_text).get("startup", {}).get("capture_reference"))
    if (args.pitch_hold_commissioning or args.yaw_hold_commissioning) and not native_reference:
        capture = subprocess.run(
            remote(AIR, "/home/fftgyro/venvs/mavlink/bin/python -"),
            input=REFERENCE_CAPTURE_SCRIPT, text=True, capture_output=True, timeout=20)
        if capture.returncode:
            (out / "reference-capture-error.txt").write_text(capture.stderr, encoding="utf-8")
            raise RuntimeError("per-run attitude reference capture failed; no acquisition or motor start")
        try:
            reference = json.loads(capture.stdout)
            quaternion = reference["summary"]["quaternion_wxyz"]
            if (len(quaternion) != 4 or
                    any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                        for v in quaternion) or
                    not .999 <= math.sqrt(sum(v * v for v in quaternion)) <= 1.001):
                raise ValueError("invalid quaternion")
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            (out / "reference-capture-error.txt").write_text(capture.stdout + capture.stderr, encoding="utf-8")
            raise RuntimeError("invalid per-run attitude reference response; no acquisition or motor start") from error
        (out / "attitude-reference.json").write_text(json.dumps(reference, indent=2), encoding="utf-8")
        resolved = json.loads(session_text)
        resolved["pid"]["zero_quaternion_wxyz"] = quaternion
        resolved.setdefault("operator_context", {})["reference_record"] = "attitude-reference.json"
        resolved["operator_context"]["reference_note"] = (
            "Automatically measured for this run; not a sensor calibration. Native stationary preflight remains mandatory.")
        session_text = json.dumps(resolved, indent=2)
        print("Per-run attitude reference captured: samples=" + str(reference["summary"]["samples"]), flush=True)
    if session_text is not None:
        (out / "source-session.json").write_text(source_session_text, encoding="utf-8")
        (out / "resolved-session.json").write_text(session_text, encoding="utf-8")
    session_encoded = base64.b64encode(session_text.encode()).decode() if session_text is not None else ""
    trial_log_source = (Path(__file__).resolve().parents[1] /
                        "src/fftgyro_quad_id/experiment_runner/px4_trial_log.py").read_bytes()
    trial_log_encoded = base64.b64encode(trial_log_source).decode()
    clock_probe_source = (Path(__file__).resolve().parents[1] /
                          "src/fftgyro_quad_id/sync/pi_clock_probe.py").read_bytes()
    clock_probe_encoded = base64.b64encode(clock_probe_source).decode()
    leptrino_raw_source = (Path(__file__).resolve().parents[1] /
                           "src/fftgyro_quad_id/io/leptrino_raw.py").read_bytes()
    leptrino_raw_encoded = base64.b64encode(leptrino_raw_source).decode()
    ram_capture_encoded = base64.b64encode((Path(__file__).resolve().parents[1] /
        'src/fftgyro_quad_id/io/ram_capture.py').read_bytes()).decode()
    leptrino_protocol_source = (Path(__file__).resolve().parents[1] /
                                "src/fftgyro_quad_id/io/leptrino.py").read_bytes()
    leptrino_protocol_encoded = base64.b64encode(leptrino_protocol_source).decode()
    if session_text is not None:
        validation_command = ("session-yaw-hold-commissioning-validate" if args.yaw_hold_commissioning else
                              "session-pitch-hold-commissioning-validate" if args.pitch_hold_commissioning else
                              "session-fixed-commissioning-validate" if args.fixed_commissioning else "session-validate")
        # Validate the resolved configuration with the actual binary before acquisition.
        validation = f"""
import base64,subprocess,tempfile
from pathlib import Path
with tempfile.TemporaryDirectory(prefix='fftgyro-session-validate-') as directory:
 p=Path(directory)/'session.json'; p.write_bytes(base64.b64decode('{session_encoded}'))
 raise SystemExit(subprocess.call(['{args.session_executor}','{validation_command}',str(p)]))
"""
        subprocess.run(remote(AIR,"python3 -"), input=validation, text=True, check=True, timeout=20)
    base = "/home/fftgyro/research-data/fftgyro/"+args.run
    boots = read_boots()
    storage_before = read_storage_capacity()
    print("PROGRESS " + storage_progress(storage_before), flush=True)
    manifest = dict(**boots, run=args.run, state="STARTING", props_removed=args.props_removed,
                    operator_stop_ready=args.operator_stop_ready, charged_4s_operator_confirmed=args.charged_4s,
                    battery_allowed_min_v=args.battery_min_v,
                    battery_allowed_max_v=args.battery_max_v,
                    aircraft_host=AIR, leptrino_host=LEP,
                    aircraft_host_key_alias=args.aircraft_host_key_alias,
                    leptrino_host_key_alias=args.leptrino_host_key_alias,
                    remote_base=base,
                    pi_storage_before=storage_before,
                    pi_storage_minimum_free_bytes=MIN_PI_STORAGE_FREE_BYTES,
                    bulk_download_deferred=args.defer_bulk_download,
                    encoder_required=True, encoder_port=args.encoder_port,
                    encoder_baud=args.encoder_baud,
                    encoder_minimum_rate_hz=args.encoder_minimum_rate_hz,
                    windows_clock_port=args.windows_clock_port,
                    windows_client_visible_ip=args.windows_client_visible_ip,
                    force_seconds=seconds, idle=args.idle,
                    immediate_qc_policy=qc_policy,
                    qc_stationary_required=args.qc_stationary_required,
                    immediate_qc_profile_sha256=hashlib.sha256(args.qc_profile.read_bytes()).hexdigest()
                    if args.qc_profile else None,
                    idle_px4_log=args.idle_px4_log,
                    fixed=args.fixed, session_props_off=args.session_props_off,
                    axis_ready=args.axis_ready, props_installed=args.props_installed,
                    fixed_commissioning=args.fixed_commissioning,
                    pitch_hold_commissioning=args.pitch_hold_commissioning,
                    yaw_hold_commissioning=args.yaw_hold_commissioning,
                    session_executor=args.session_executor,
                    session_source_sha256=hashlib.sha256(source_session_text.encode()).hexdigest() if source_session_text else None,
                    session_resolved_sha256=hashlib.sha256(session_text.encode()).hexdigest() if session_text else None,
                    per_run_attitude_reference=args.pitch_hold_commissioning or args.yaw_hold_commissioning,
                    native_reference_capture=native_reference,
                    flightlog_utc_before_logger_on=(not args.idle or args.idle_px4_log),
                    attitude_reference_summary=reference["summary"] if reference else None,
                    motor_profile="NONE" if args.idle else
                    ("bounded commissioning " if args.fixed_commissioning or args.pitch_hold_commissioning or args.yaw_hold_commissioning else "qualified SessionRunner ") + json.loads(session_text).get("control_mode", "AXIS_PID")
                    if session_text is not None else
                    "SessionRunner props-off: HOLD 5 s, total <=12 s, collective <=0.1, individual <=0.12"
                    if args.session_props_off else "existing bench-pid: 6 s, collective <=0.1, individual <=0.12")
    def save():
        (out/"run.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    def fail(message):
        manifest.update(state="INCOMPLETE", error=message)
        save()
        raise RuntimeError(message)
    manifest["retest_context"] = retest_context
    manifest["empty_sensor_condition"] = empty_condition
    manual_decay_guide = (ManualDecayGuide(
        static_capture=retest_context["stage"] == EMPTY_STATIC_STAGE,
        pose_label=empty_condition["pose_label"],
        excitation_label=empty_condition.get("excitation_label") or "",
    ) if empty_condition else None)
    if manual_decay_guide:
        manifest["manual_empty_sensor_decay"] = manual_decay_guide.metadata()
        manifest["manual_motion_requires_offline_contact_review"] = True
    manifest["capture_setup_utc_ns"] = time.time_ns()
    manifest["retest_context_sha256"] = hashlib.sha256(args.retest_context.read_bytes()).hexdigest() \
        if args.retest_context else None
    save()
    low_storage = storage_below_limit(storage_before)
    if low_storage:
        manifest["pi_storage_preflight"] = "BLOCKED"
        fail(
            "Pi SD free space at or below 1 GiB on " + ", ".join(low_storage)
            + "; no acquisition or motor test started. " + storage_progress(storage_before)
        )
    manifest["pi_storage_preflight"] = "PASSED"
    save()
    processes = []
    workers = {}
    logs = []
    def launch(host, script, name):
        log = (out/(name+"-process.txt")).open("x",encoding="utf-8")
        logs.append(log)
        proc = subprocess.Popen(remote(host,"python3 -"),stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT,text=True,encoding="utf-8")
        proc.stdin.write(script)
        proc.stdin.close()
        processes.append(proc)
        workers[name] = proc
        return proc
    server_stop_path = f"/tmp/fftgyro-clock-stop-{uuid.uuid4().hex}"
    air_stop_path = f"/tmp/fftgyro-air-stop-{uuid.uuid4().hex}"
    air_runtime = f"/dev/shm/fftgyro-{args.run}-{uuid.uuid4().hex}"
    encoder_ready = out / "encoder-ready"
    encoder_stop = out / "encoder-stop"
    windows_clock_ready = out / "encoder-clock-ready"
    windows_clock_stop = out / "encoder-clock-stop"
    encoder = None
    windows_clock = None

    cleanup_started = False
    def emergency_cleanup(error):
        """Best-effort ordered stop for an interrupted local coordinator."""
        nonlocal cleanup_started
        if cleanup_started:
            return
        cleanup_started = True
        cleanup_errors = []

        def stop_worker(name, host, stop_path, timeout):
            process = workers.get(name)
            if process is None:
                return
            # Ctrl+C is delivered to the whole local process group and can end
            # ssh.exe before its remote worker.  Likewise, an exited local SSH
            # process is not sufficient evidence that every remote child and
            # bound UDP socket has closed.  Always send this run's owned marker.
            try:
                request_remote_stop(host, stop_path)
            except Exception as exc:
                cleanup_errors.append(f"{name} stop request: {type(exc).__name__}: {exc}")
            if process.poll() is None:
                try:
                    process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    cleanup_errors.append(f"{name} did not close within {timeout} s")
                    process.terminate()
                    try:
                        process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()

        def stop_local(process, stop_path, name, timeout):
            if process is None or process.poll() is not None:
                return
            try:
                stop_path.touch(exist_ok=True)
                process.wait(timeout=timeout)
            except Exception as exc:
                cleanup_errors.append(f"{name} close: {type(exc).__name__}: {exc}")
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()

        # First ask the aircraft supervisor to abort-ramp/release and close its
        # PX4 log/RPM stream.  Force then closes before the final clock bracket.
        stop_worker("aircraft", AIR, air_stop_path, 20)
        stop_local(encoder, encoder_stop, "frame encoder", 10)
        stop_worker("leptrino", LEP, f"{base}/capture-stop", 20)
        stop_local(windows_clock, windows_clock_stop, "Windows clock probe", 10)
        stop_worker("server", AIR, server_stop_path, 20)
        for process in processes:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        for log in logs:
            if not log.closed:
                log.close()
        manifest.update(state="INCOMPLETE", error=f"{type(error).__name__}: {error}",
                        emergency_cleanup_attempted=True,
                        emergency_cleanup_errors=cleanup_errors,
                        process_returncodes=[process.returncode for process in processes])
        try:
            save()
        except Exception as exc:
            # Preserve the triggering exception.  A storage failure is still
            # reported to stderr when run.json itself cannot be updated.
            print(f"Emergency cleanup record failed: {type(exc).__name__}: {exc}",
                  file=sys.stderr, flush=True)

    _EMERGENCY_CLEANUP = emergency_cleanup
    server = launch(AIR, f"""
import base64,subprocess,tempfile,time
from pathlib import Path
stop=Path('{server_stop_path}')
with tempfile.TemporaryDirectory(prefix='fftgyro-clock-worker-') as directory:
 source=Path(directory)/'pi_clock_probe.py'
 source.write_bytes(base64.b64decode('{clock_probe_encoded}'))
 children=[]
 try:
  children=[
   subprocess.Popen(['python3',str(source),'serve','--bind','{args.clock_server_ip}','--peer','{args.clock_client_visible_ip}','--port','43187','--seconds','{seconds+60}','--stop-file',str(stop)]),
   subprocess.Popen(['python3',str(source),'serve','--bind','{args.clock_server_ip}','--peer','{args.windows_client_visible_ip}','--port','{args.windows_clock_port}','--seconds','{seconds+60}','--stop-file',str(stop)])]
  while not stop.exists() and all(child.poll() is None for child in children): time.sleep(.05)
  if not stop.exists() and any(child.returncode for child in children if child.returncode is not None):
   raise RuntimeError('one clock responder exited')
 finally:
  for child in children:
   if child.poll() is None: child.terminate()
  for child in children:
   try: child.wait(timeout=3)
   except subprocess.TimeoutExpired: child.kill(); child.wait()
  stop.unlink(missing_ok=True)
""", "server")
    print("PROGRESS Pi間時計同期・力計収録の準備", flush=True)
    try:
        wait_clock_ready(server, out/"server-process.txt")
    except RuntimeError as error:
        fail(str(error))
    encoder_log_path = out / "encoder-process.txt"
    encoder_log = encoder_log_path.open("x", encoding="utf-8")
    logs.append(encoder_log)
    encoder = subprocess.Popen(
        [sys.executable, "-u", str(Path(__file__).with_name("capture_frame_encoder.py")),
         "--port", args.encoder_port, "--baud", str(args.encoder_baud),
         "--seconds", str(seconds + 30), "--minimum-rate-hz", str(args.encoder_minimum_rate_hz),
         "--output", str(out / "encoder"), "--ready-file", str(encoder_ready),
         "--stop-file", str(encoder_stop)],
        stdout=encoder_log, stderr=subprocess.STDOUT, text=True, encoding="utf-8")
    windows_clock_log_path = out / "encoder-clock-process.txt"
    windows_clock_log = windows_clock_log_path.open("x", encoding="utf-8")
    logs.append(windows_clock_log)
    windows_clock = subprocess.Popen(
        [sys.executable, "-u", str(Path(__file__).resolve().parents[1] /
         "src/fftgyro_quad_id/sync/pi_clock_probe.py"), "probe",
         "--peer", args.clock_server_ip, "--port", str(args.windows_clock_port),
         "--seconds", str(seconds + 30), "--hz", "10", "--output", str(out / "encoder-clock"),
         "--ready-file", str(windows_clock_ready), "--stop-file", str(windows_clock_stop)],
        stdout=windows_clock_log, stderr=subprocess.STDOUT, text=True, encoding="utf-8")
    try:
        wait_local_ready(encoder, encoder_ready, encoder_log_path, "FFT GYRO encoder")
        wait_local_ready(windows_clock, windows_clock_ready, windows_clock_log_path,
                         "Windows clock probe")
    except RuntimeError as error:
        fail(str(error))
    print("PROGRESS FFT GYRO encoder / Windows clock sync ready", flush=True)
    absolute_axis = ("Pitch" if session_text is not None and needs_pitch_anchor(json.loads(session_text)) else
                     "Yaw" if session_text is not None and needs_yaw_anchor(json.loads(session_text)) else None)
    if absolute_axis is not None:
        label = "Pitch水平基準" if absolute_axis == "Pitch" else "Yaw正面瞬時基準"
        print(f"PROGRESS {label}：エンコーダーとPX4の静止姿勢を対応付け中", flush=True)
        try:
            begin_ns = time.monotonic_ns()
            capture_script = (REFERENCE_CAPTURE_SCRIPT if absolute_axis == "Pitch"
                              else YAW_REFERENCE_CAPTURE_SCRIPT)
            capture_timeout = 20 if absolute_axis == "Pitch" else 12
            captured = None
            for attempt in range(2):
                try:
                    captured = subprocess.run(
                        remote(AIR, "/home/fftgyro/venvs/mavlink/bin/python -"),
                        input=capture_script, text=True, capture_output=True,
                        timeout=capture_timeout)
                    break
                except subprocess.TimeoutExpired:
                    if attempt == 1:
                        raise
                    time.sleep(.5)
            assert captured is not None
            end_ns = time.monotonic_ns()
            if captured.returncode:
                raise RuntimeError("Pitch reference capture failed: " + captured.stderr[-1000:])
            reference = json.loads(captured.stdout)
            # Preserve rejected evidence as well, so a pre-motor refusal can be
            # diagnosed without guessing which stationary criterion fired.
            attempt_name = "pitch-horizontal-reference-attempt.json" if absolute_axis == "Pitch" else "yaw-front-reference-attempt.json"
            (out / attempt_name).write_text(
                json.dumps(reference, indent=2), encoding="utf-8")
            time.sleep(.2)  # decoded encoder CSV flush only; no motor process exists yet
            with (out / "encoder/samples.csv").open(encoding="utf-8", newline="") as stream:
                samples = list(csv.DictReader(stream))
            encoder_metadata = json.loads((out / "encoder/metadata.json").read_text(encoding="utf-8"))
            if json.loads(session_text).get("coupled_yaw"):
                resolved, evidence = resolve_coupled_anchor(
                    json.loads(session_text), reference, samples, encoder_metadata,
                    yaw_front_calibration, begin_ns, end_ns)
            elif absolute_axis == "Pitch":
                resolved, evidence = resolve_pitch_anchor(
                    json.loads(session_text), reference, samples, encoder_metadata, begin_ns, end_ns)
            else:
                resolved, evidence = resolve_yaw_anchor(
                    json.loads(session_text), reference, samples, encoder_metadata,
                    yaw_front_calibration, begin_ns, end_ns)
            session_text = json.dumps(resolved, indent=2)
            new_encoded = base64.b64encode(session_text.encode()).decode()
            validation = validation.replace(session_encoded, new_encoded)
            session_encoded = new_encoded
            subprocess.run(remote(AIR, "python3 -"), input=validation,
                           text=True, check=True, timeout=20)
            evidence_name = "pitch-horizontal-reference.json" if absolute_axis == "Pitch" else "yaw-front-reference.json"
            (out / evidence_name).write_text(
                json.dumps(evidence, indent=2), encoding="utf-8")
            (out / "resolved-session.json").write_text(session_text, encoding="utf-8")
            manifest.update(angle_reference_axis=absolute_axis,
                            per_run_attitude_reference=True,
                            attitude_reference_summary=reference['summary'],
                            initial_absolute_angle_deg=(evidence['initial_pitch_deg'] if absolute_axis == "Pitch"
                                                        else evidence['initial_front_relative_yaw_deg']),
                            session_resolved_sha256=hashlib.sha256(session_text.encode()).hexdigest())
            if absolute_axis == "Pitch":
                manifest["pitch_angle_basis"] = "encoder_horizontal_zero"
            else:
                manifest["yaw_angle_basis"] = "fixed_test_encoder_front_instantaneous"
                manifest["yaw_front_calibration_sha256"] = args.yaw_front_calibration_sha256
            save()
            initial = evidence['initial_pitch_deg'] if absolute_axis == "Pitch" else evidence['initial_front_relative_yaw_deg']
            print(f"PROGRESS {label}：開始 {initial:+.2f}° / "
                  f"最初の目標 {evidence['first_target_deg']:+.2f}° / "
                  f"移動量 {evidence['initial_move_deg']:+.2f}°", flush=True)
        except (ValueError, KeyError, OSError, subprocess.SubprocessError, RuntimeError) as error:
            fail(f"absolute {absolute_axis} reference failed before motor start: " + str(error))
    force_launch = time.monotonic()
    lep = launch(LEP, f"""
import base64,json,subprocess,time
from pathlib import Path
p=Path('{base}'); p.mkdir(parents=True,exist_ok=False)
source=p/'worker-source'; source.mkdir()
clock_source=source/'pi_clock_probe.py'; clock_source.write_bytes(base64.b64decode('{clock_probe_encoded}'))
force_source=source/'leptrino_raw.py'; force_source.write_bytes(base64.b64decode('{leptrino_raw_encoded}'))
force_protocol=source/'leptrino.py'; force_protocol.write_bytes(base64.b64decode('{leptrino_protocol_encoded}'))
exec(base64.b64decode('{ram_capture_encoded}'))
spool=RamCapture(p,{seconds})
capture=spool.runtime
print('RAM_CAPTURE_READY '+str(capture),flush=True)
clock=None
force=None
try:
 clock=subprocess.Popen(['python3',str(clock_source),'probe','--peer','{args.clock_server_ip}','--seconds','{seconds+30}','--hz','10','--output',str(capture/'clock'),'--ready-file',str(p/'clock-ready'),'--stop-file',str(p/'clock-stop')])
 deadline=time.monotonic()+10
 while not (p/'clock-ready').exists():
  if clock.poll() is not None or time.monotonic()>deadline: raise RuntimeError('no valid clock exchange before force')
  time.sleep(.05)
 time.sleep({CLOCK_PRE_ROLL_S})
 if clock.poll() is not None: raise RuntimeError('clock probe ended before force')
 force=subprocess.Popen(['python3',str(force_source),'--port','/dev/serial/by-id/usb-STMicroelectronics_STM32_Virtual_COM_Port_2506002-if00','--expected-serial','2506002','--seconds','{seconds}','--output',str(capture/'leptrino')])
 deadline=time.monotonic()+{seconds+20}
 size=None; ready=False; requested=False
 while force.poll() is None:
  spool.check()
  if clock.poll() is not None: raise RuntimeError('clock ended during force capture')
  if time.monotonic()>deadline: raise RuntimeError('force capture deadline')
  if (p/'capture-stop').exists():
   requested=True
   force.terminate()
   break
  raw=capture/'leptrino/chunks.bin'
  if not ready and raw.exists():
   current=raw.stat().st_size
   if size is not None and current>size and current>1024:
    print('FORCE_STREAM_READY',flush=True); ready=True
   size=current
  time.sleep(.1)
 rc=force.wait(timeout=10)
 if not {args.idle!r} and not requested: raise RuntimeError('force ended before coordinator stop')
 # Stop acknowledgement and writer drain precede the final clock bracket.
 (p/'clock-stop').touch(exist_ok=False)
 cc=clock.wait(timeout=5)
 if rc or cc: raise RuntimeError('force or clock capture incomplete')
 summary=json.loads((capture/'clock/summary.json').read_text())
 if not summary.get('coordinated_stop') or not summary.get('final_exchange_after_stop'):
  raise RuntimeError('clock final exchange did not follow force closure')
finally:
 for child in (force,clock):
  if child is not None and child.poll() is None:
   child.terminate()
   try: child.wait(timeout=5)
   except subprocess.TimeoutExpired: child.kill(); child.wait()
 print('RAM_PERSIST_START',flush=True)
 spool.persist()
 print('RAM_PERSIST_VERIFIED',flush=True)
""", "leptrino")
    # Check actual raw file progress before allowing the motor bench to start.
    deadline = time.monotonic()+70
    while time.monotonic() < deadline:
        if lep.poll() is not None or server.poll() is not None:
            fail("force/clock process stopped before motor start")
        # Read the existing SSH worker's readiness event; no extra SSH per poll.
        if "FORCE_STREAM_READY" in (out/"leptrino-process.txt").read_text(encoding="utf-8"):
            break
        time.sleep(.1)
    else:
        fail("force raw stream not progressing; no motor start")
    if session_text is not None and time.monotonic() - force_launch > seconds - session_limit - 10:
        fail("insufficient force recording time remains; no motor start")
    print("Force stream active; starting RPM" + (" (motor OFF)" if args.idle else ", then bounded motor commissioning"),flush=True)
    air = launch(AIR, f"""
import subprocess,time,json,base64,csv,shutil
from pathlib import Path
p=Path('{base}'); p.mkdir(parents=True,exist_ok=False)
runtime=Path('{air_runtime}'); runtime.mkdir(parents=True,exist_ok=False)
stop_request=Path('{air_stop_path}'); stop_request.unlink(missing_ok=True)
completion_marker=p/'aircraft-complete'
failure_marker=p/'aircraft-failed'
rpm=subprocess.Popen(['/home/fftgyro/commissioning/rpm-sync-20260909/fftgyro-rpm','/dev/serial/by-id/usb-Arduino_LLC_Arduino_Nano_Every_D5035FB0515452344D202020FF09141A-if00',str(runtime/'rpm'),'{seconds if args.idle else int(session_limit+15) if session_text is not None else 40}','100'])
pid=None
log_stop_attempted=False
worker_complete=False
worker_error='aircraft worker ended before completion'
def flightlog(action):
 helper=base64.b64decode('{trial_log_encoded}').decode('utf-8')
 output=p/'flightlog'/(action+'.json')
 subprocess.run(['/home/fftgyro/venvs/mavlink/bin/python','-c',helper,action,
                 '--output',str(output),'--battery-min-v','{args.battery_min_v}',
                 '--battery-max-v','{args.battery_max_v}'],check=True,timeout=10 if action=='stop' else 35)
 result=json.loads(output.read_text())
 if action=='start':
  sd=result.get('sd_capacity_preflight',{{}})
  if sd.get('passed'):
   print('PROGRESS Pixhawk SD空き '+str(sd['available_reported'])+'／開始許可',flush=True)
  battery=result.get('battery_preflight',{{}})
  if battery.get('passed'):
   print('PROGRESS 電圧確認 '+format(float(battery['voltage_v']),'.3f')+' V／開始許可',flush=True)
 return result
def finish_flightlog():
 global log_stop_attempted
 start_record=p/'flightlog/start.json'
 if not start_record.exists() or not json.loads(start_record.read_text()).get('logger_on_attempted'):
  return
 log_stop_attempted=True
 stop_record=p/'flightlog/stop.json'
 last_error=None
 for attempt in range(1,3):
  if stop_record.exists():
   try:
    existing=json.loads(stop_record.read_text())
    if existing.get('complete') and existing.get('logger_stopped'):
     return
   except (OSError,ValueError):
    pass
   serial=1
   while (p/'flightlog'/('stop-failed-'+str(serial)+'.json')).exists(): serial+=1
   stop_record.replace(p/'flightlog'/('stop-failed-'+str(serial)+'.json'))
  try:
   flightlog('stop')
   return
  except subprocess.CalledProcessError as error:
   last_error=error
   if attempt<2: time.sleep(.25)
 raise last_error
try:
 csv=runtime/'rpm/rpm.csv'
 deadline=time.monotonic()+7
 while time.monotonic()<deadline and rpm.poll() is None:
  if csv.exists() and csv.stat().st_size>100: break
  time.sleep(.1)
 else: raise RuntimeError('RPM not ready; no motor start')
 if {args.idle!r}:
  if {args.idle_px4_log!r}:
   print('PROGRESS PX4 SDログ記録開始（非ARM・モーター指令なし）',flush=True)
   flightlog('start')
  print('PROGRESS IDLE_CAPTURE_READY Leptrino・encoder・clock・RPM'+('・PX4 ULog' if {args.idle_px4_log!r} else '')+'収録中',flush=True)
  rr=rpm.wait(timeout={seconds+15})
  if rr: raise RuntimeError('RPM incomplete')
  if {args.idle_px4_log!r}:
   print('PROGRESS PX4 SDログ記録終了',flush=True)
   finish_flightlog()
  worker_complete=True
  raise SystemExit(0)
 time.sleep({RPM_READY_HOLD_S})
 # Own the UART only before the executor opens it. Every start includes fresh UTC set/readback.
 print('PROGRESS PX4時計設定・SDログ記録開始',flush=True)
 flightlog('start')
 if rpm.poll() is not None: raise RuntimeError('RPM ended during FlightLog setup; no motor start')
 if {session_text is not None!r}:
  config=json.loads(base64.b64decode('{session_encoded}'))
  (p/'source-session.json').write_bytes(base64.b64decode('{session_encoded}'))
  config['session_id']='pid'
  config['hardware']['log_directory']=str(runtime)
  config_path=p/'run-session.json'
  config_path.write_text(json.dumps(config,indent=2))
  command=['{args.session_executor}','session-fixed-commissioning' if {args.fixed_commissioning!r} else 'session-run',str(config_path)]
  if {args.fixed_commissioning!r}: command+=['--props-installed','--fixed','--operator-stop-ready','--charged-4s']
  if {args.pitch_hold_commissioning!r}:
   command=['{args.session_executor}','session-pitch-hold-commissioning',str(config_path),
            '--props-installed','--pitch-only-free','--operator-stop-ready','--charged-4s']
  if {args.yaw_hold_commissioning!r}:
   command=['{args.session_executor}','session-yaw-hold-commissioning',str(config_path),
            '--props-installed','--yaw-only-free','--operator-stop-ready','--charged-4s']
  pid=subprocess.Popen(command)
 elif {args.session_props_off!r}:
  release=Path('/home/fftgyro/commissioning/executor-managed/releases/props-off-v7-20260911-01')
  binary=release/'fftgyro-executor'
  template=release/'session.json'
  config=json.loads(template.read_text())
  config['session_id']='pid'
  config['parent_series_id']='{args.run}'
  config['hardware']['log_directory']=str(p)
  config_path=p/'props-off-session.json'
  config_path.write_text(json.dumps(config,indent=2))
  pid=subprocess.Popen([str(binary),'session-props-off',str(config_path),'--props-removed','--fixed','--operator-stop-ready','--charged-4s'])
 else:
  pid=subprocess.Popen(['/home/fftgyro/commissioning/parallel-precheck-20260908/bench-pid','/dev/serial0',str(p/'pid'),'--props-off-spin'])
 deadline=time.monotonic()+{session_limit+5 if session_text is not None else 18 if args.session_props_off else 10}
 last_size=csv.stat().st_size; last_progress=time.monotonic()
 next_display=0
 while pid.poll() is None:
  current=csv.stat().st_size
  if current!=last_size: last_size=current; last_progress=time.monotonic()
  if stop_request.exists():
   pid.terminate(); raise RuntimeError('coordinator requested safe stop')
  if rpm.poll() is not None or time.monotonic()-last_progress>2.5 or time.monotonic()>deadline:
   pid.terminate(); raise RuntimeError('RPM/controller watchdog stop')
  if time.monotonic()>=next_display:
   next_display=time.monotonic()+{AIRCRAFT_PROGRESS_INTERVAL_S}
   control=runtime/'pid/control.csv'
   if control.exists():
    try:
     import csv as csv_module
     with control.open('rb') as stream:
      header=next(csv_module.reader([stream.readline().decode().strip()]))
      # The writer flushes while a row may be observed concurrently.  Search
      # backwards for the newest complete, schema-matching row instead of
      # assuming tail[-2] is usable.  The latter left the terminal showing the
      # final STARTUP_RAMP sample for the rest of long experiments.
      stream.seek(max(stream.tell(),control.stat().st_size-65536))
      tail=stream.read().decode().splitlines()
     state=None
     for line in reversed(tail):
      values=next(csv_module.reader([line]))
      if len(values)==len(header):
       candidate=dict(zip(header,values))
       if candidate.get('stage') and candidate.get('angle_deg') and candidate.get('target_deg'):
        state=candidate
        break
     if state is not None:
      labels=dict(STARTUP_RAMP='起動ランプ',RAMP_DOWN='停止ランプ',
                  TRANSITION='目標移動',SEGMENT='実験波形',SETTLING_EXTENSION='整定待ち',
                  COLLECTIVE_CHANGE='共通推力変更',HOLD='保持',MOVE='移動',SINE='往復')
      stage=labels.get(state['stage'],state['stage'])
      extra=''
      if 'config' in locals() and config.get('coupled_yaw') and state.get('yaw_angle_deg'):
       extra=' | Yaw '+format(float(state['yaw_angle_deg']),'+.1f')+' / '+format(float(state['yaw_target_deg']),'+.1f')+' deg'
      print('PROGRESS '+stage+' | 角度 '+format(float(state['angle_deg']),'+.1f')+' / 目標 '+format(float(state['target_deg']),'+.1f')+' deg'+extra,flush=True)
     else:
      print('PROGRESS 制御実行中（進捗CSVの更新待ち）',flush=True)
    except (OSError,ValueError,KeyError,UnicodeError,csv_module.Error):
     # Never leave the operator looking at a stale phase label.  Detailed
     # control logging continues independently and is validated after the run.
     print('PROGRESS 制御実行中（進捗CSVの読取待ち）',flush=True)
  time.sleep(.02)
 print('PID stopped, exit='+str(pid.returncode),flush=True)
 print('PROGRESS モーター出力停止／RPM・PX4ログ終了処理',flush=True)
 finish_flightlog()
 # The deployed RPM recorder handles SIGTERM as a coordinated close: it drains
 # the Nano stream, writes result.txt=COMPLETE, and exits zero.  This was checked
 # against the production binary; waiting for its conservative duration would
 # leave the UI apparently stuck long after motor release.
 time.sleep({RPM_POST_ROLL_S})
 if rpm.poll() is None: rpm.terminate()
 rr=rpm.wait(timeout=5)
 shutdown_guard_capture=False
 if pid.returncode and {session_text is not None!r}:
  try:
   import csv as records_csv
   result=(runtime/'pid/result.txt').read_text().splitlines()
   statuses=[line.split('=',1)[1] for line in result if line.startswith('status=')]
   details=[line.split('=',1)[1] for line in result if line.startswith('detail=')]
   expected={{segment['segment_id']:float(segment['duration_s']) for segment in config['segments']
             if not segment.get('preparation_only',False)}}
   observed={{}}; last_stage=None
   with (runtime/'pid/control.csv').open(newline='') as stream:
    for row in records_csv.DictReader(stream):
     last_stage=row.get('stage')
     if row.get('stage')=='SEGMENT' and row.get('segment_id') in expected:
      observed[row['segment_id']]=max(observed.get(row['segment_id'],0.0),float(row['segment_time_s']))
   with (runtime/'pid/ack.csv').open(newline='') as stream:
    acknowledgements=list(records_csv.DictReader(stream))
   shutdown_guard_capture=(
    statuses and statuses[-1]=='ABORTED' and details and details[-1]=='body rate limit'
    and last_stage=='RAMP_DOWN' and expected
    and all(observed.get(segment_id,-1.0)>=duration-.05 for segment_id,duration in expected.items())
    and acknowledgements and acknowledgements[-1].get('release')=='1'
    and acknowledgements[-1].get('ack_status')=='1')
  except (OSError,ValueError,KeyError,TypeError,records_csv.Error):
   shutdown_guard_capture=False
 if (pid.returncode and not shutdown_guard_capture) or rr:
  raise RuntimeError('PID or RPM incomplete')
 capture_state='CAPTURE_COMPLETE_SHUTDOWN_GUARD' if shutdown_guard_capture else 'COMPLETE'
 (runtime/'pid/capture-result.json').write_text(json.dumps({{'state':capture_state}},indent=2)+'\\n')
 if shutdown_guard_capture:
  print('CAPTURE complete; shutdown body-rate guard followed by acknowledged release',flush=True)
 worker_complete=True
except BaseException as error:
 worker_error=type(error).__name__+': '+str(error)
 raise
finally:
 for child in (pid,rpm):
  if child is not None and child.poll() is None:
   child.terminate()
   try: child.wait(timeout=3)
   except subprocess.TimeoutExpired: child.kill(); child.wait()
 finish_flightlog()
 stop_request.unlink(missing_ok=True)
 for name in ('rpm','pid'):
  source=runtime/name
  if source.exists() and not (p/name).exists(): shutil.copytree(source,p/name)
 shutil.rmtree(runtime,ignore_errors=True)
 if worker_complete:
  # This source is embedded in the outer Python f-string above.  Keep the
  # backslash escaped so the remote script receives one valid string literal.
  completion_marker.write_text('COMPLETE\\n')
 else:
  failure_marker.write_text(worker_error+'\\n')
""", "aircraft")
    air_launch_monotonic = time.monotonic()
    air_deadline = air_launch_monotonic + (seconds+30 if args.idle or session_text is not None else 65)
    last_progress_line = ""
    remote_completion_verified = False
    remote_failure_verified = False
    # Avoid repeated SSH handshakes during measurement. Start independent
    # completion probes only near the earliest plausible supervisor finish.
    completion_probe_delay = (
        max(10.0, seconds - 10.0) if args.idle
        else max(10.0, min(session_limit - 5.0, expected_session_seconds)) if session_text is not None
        else 10.0
    )
    next_completion_probe = air_launch_monotonic + completion_probe_delay
    while air.poll() is None:
        if time.monotonic() > air_deadline:
            raise subprocess.TimeoutExpired("aircraft capture", seconds+30)
        if encoder.poll() is not None:
            raise RuntimeError("FFT GYRO encoder capture ended during the experiment")
        if windows_clock.poll() is not None:
            raise RuntimeError("Windows clock probe ended during the experiment")
        progress_lines = (out / "aircraft-process.txt").read_text(encoding="utf-8", errors="replace").splitlines()
        progress = next((line for line in reversed(progress_lines) if line.startswith("PROGRESS ")), "")
        if progress and progress != last_progress_line:
            print(progress, flush=True)
            last_progress_line = progress
        if manual_decay_guide and (manual_decay_guide.ready_monotonic_ns is not None
                                  or "IDLE_CAPTURE_READY" in progress):
            events = manual_decay_guide.advance(time.monotonic_ns(), time.time_ns())
            for event in events:
                print("PROGRESS " + event["message"], flush=True)
            if events:
                manifest["manual_empty_sensor_decay"] = manual_decay_guide.metadata()
                save()
        if time.monotonic() >= next_completion_probe:
            next_completion_probe = time.monotonic() + REMOTE_COMPLETION_POLL_S
            if remote_file_exists(AIR, f"{base}/aircraft-complete"):
                remote_completion_verified = True
                # The remote supervisor has finalized RPM/PID artifacts and
                # released UART. Only the stale local SSH client may remain.
                try:
                    air.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    air.terminate()
                    air.wait(timeout=3)
                break
            if remote_file_exists(AIR, f"{base}/aircraft-failed"):
                remote_failure_verified = True
                # The remote worker has already stopped motors, closed the PX4
                # logger, and copied its diagnostic artifacts.  Do not leave
                # the operator staring at a stale progress line until the
                # conservative acquisition timeout expires.
                try:
                    air.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    air.terminate()
                    air.wait(timeout=3)
                break
        try:
            air.wait(timeout=.25)
        except subprocess.TimeoutExpired:
            pass
    # A failed local SSH transport is not proof that the remote supervisor has
    # stopped.  Enter the ordered emergency cleanup instead of treating the
    # disconnect as normal completion and racing SFTP against remote finalizers.
    if remote_failure_verified:
        raise RuntimeError(
            "aircraft worker reported incomplete; requesting owned safe stop "
            "before artifact recovery"
        )
    if air.returncode != 0 and not remote_completion_verified:
        raise RuntimeError(
            f"aircraft supervisor transport exited {air.returncode}; "
            "requesting owned safe stop before artifact recovery"
        )
    print("Aircraft bench finished; closing encoder/force/RPM/clock captures",flush=True)
    # Requests go to this run's workers, not potentially reused OS PIDs.
    time.sleep(ENCODER_POST_ROLL_S)
    encoder_stop.touch(exist_ok=False)
    if encoder.wait(timeout=10):
        raise RuntimeError("FFT GYRO encoder capture did not close completely")
    time.sleep(FORCE_POST_ROLL_S)
    request_remote_stop(LEP, f"{base}/capture-stop")
    lep.wait(timeout=20)  # Closed RAM capture is copied, fsynced and verified on SD.
    windows_clock_stop.touch(exist_ok=False)
    if windows_clock.wait(timeout=10):
        raise RuntimeError("Windows clock probe did not close completely")
    request_remote_stop(AIR, server_stop_path)
    server.wait(timeout=8)
    for log in logs:
        log.close()
    for key, after in read_boots().items():
        manifest[key+"_after"] = after
        if after != boots[key]:
            raise RuntimeError("Pi rebooted during run")
    process_success = [p.returncode == 0 or (p is air and remote_completion_verified)
                       for p in processes]
    manifest.update(state="CAPTURED" if all(process_success) else "INCOMPLETE",
                    process_returncodes=[p.returncode for p in processes],
                    aircraft_completion_marker_verified=remote_completion_verified,
                    encoder_process_returncode=encoder.returncode,
                    windows_clock_process_returncode=windows_clock.returncode,
                    encoder=json.loads((out / "encoder/metadata.json").read_text(encoding="utf-8")),
                    encoder_clock=json.loads((out / "encoder-clock/summary.json").read_text(encoding="utf-8")))
    if (manifest["encoder"].get("state") != "COMPLETE" or
            manifest["encoder_clock"].get("state") != "COMPLETE" or
            manifest["encoder_clock"].get("server_boot_id") != manifest["aircraft_boot_id"]):
        manifest["state"] = "INCOMPLETE"
    save()
    for marker in (encoder_stop, windows_clock_stop):
        marker.unlink(missing_ok=True)
    if args.defer_bulk_download:
        print("PROGRESS 完了票を確認中／生データはPiのSDに保持", flush=True)
        aircraft_receipts = {"rpm/result.txt": "air-rpm-result.txt"}
        if not args.idle or args.idle_px4_log:
            aircraft_receipts.update({
                "flightlog/start.json": "air-flightlog-start.json",
                "flightlog/stop.json": "air-flightlog-stop.json",
            })
        if not args.idle:
            aircraft_receipts.update({
                "pid/result.txt": "air-pid-result.txt",
                "pid/capture-result.json": "air-pid-capture-result.json",
            })
        leptrino_receipts = {
            "ram-storage.json": "leptrino-ram-storage.json",
            "clock/summary.json": "leptrino-clock-summary.json",
            "leptrino/summary.json": "leptrino-force-summary.json",
        }
        # These are only a few kilobytes and use disjoint staging files, so both
        # handshakes can safely overlap without bulk-transfer contention.
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = (
                pool.submit(collect_sftp_receipts, AIR, base, aircraft_receipts, out),
                pool.submit(collect_sftp_receipts, LEP, base, leptrino_receipts, out),
            )
            for future in futures:
                future.result()
        receipts = out / "remote-receipts"
        manifest['leptrino_ram_storage'] = json.loads(
            (receipts / 'leptrino-ram-storage.json').read_text(encoding="utf-8"))
        manifest["flightlog"] = {
            "start": json.loads((receipts / "air-flightlog-start.json").read_text(encoding="utf-8")),
            "stop": json.loads((receipts / "air-flightlog-stop.json").read_text(encoding="utf-8")),
        } if (not args.idle or args.idle_px4_log) else {}
        remote_verified = (
            manifest['leptrino_ram_storage'].get('state') == 'PERSISTED_VERIFIED'
            and json.loads((receipts / 'leptrino-clock-summary.json').read_text(encoding="utf-8")).get('state') == 'COMPLETE'
            and json.loads((receipts / 'leptrino-force-summary.json').read_text(encoding="utf-8")).get('state') in ('COMPLETE', 'RAW_COMPLETE')
            and (receipts / 'air-rpm-result.txt').read_text(encoding="utf-8").splitlines()[0] == 'COMPLETE'
            and (args.idle or json.loads(
                (receipts / 'air-pid-capture-result.json').read_text(encoding="utf-8")
            ).get('state') in ('COMPLETE', 'CAPTURE_COMPLETE_SHUTDOWN_GUARD'))
        )
        manifest.update(
            remote_artifacts_verified=remote_verified,
            deferred_artifact_names={
                "aircraft": (["flightlog", "rpm", "pid"] if not args.idle else
                             ["flightlog", "rpm"] if args.idle_px4_log else ["rpm"]),
                "leptrino": ["clock", "leptrino", "ram-storage.json"],
            },
        )
        if not remote_verified:
            manifest["state"] = "INCOMPLETE"
    else:
        # Full synchronous collection remains available for commissioning and recovery.
        aircraft_names = (("flightlog", "rpm") if args.idle_px4_log else ("rpm",)) \
            if args.idle else ("flightlog", "rpm", "pid")
        collect_sftp_artifacts(AIR, base, aircraft_names, out)
        collect_sftp_artifacts(LEP, base, ("clock", "leptrino", "ram-storage.json"), out)
        manifest['leptrino_ram_storage'] = json.loads((out / 'ram-storage.json').read_text())
    ram_verified = manifest['leptrino_ram_storage'].get('state') == 'PERSISTED_VERIFIED'
    if not ram_verified:
        manifest['state'] = 'INCOMPLETE'
    if (not args.idle or args.idle_px4_log) and not args.defer_bulk_download:
        manifest["flightlog"] = {action: json.loads((out / "flightlog" / (action + ".json")).read_text())
                                 for action in ("start", "stop") if (out / "flightlog" / (action + ".json")).exists()}
        if manifest["state"] == "INCOMPLETE" and ram_verified and acquisition_artifacts_complete(out):
            manifest["state"] = "CAPTURED"
            manifest["artifact_recovery"] = {
                "reason": "all synchronized acquisition artifacts completed despite a worker transport exit",
                "process_returncodes": manifest["process_returncodes"],
            }
        save()
    save()
    print("PROGRESS 完了票確認終了／生データはPiのSDに保持" if args.defer_bulk_download
          else "PROGRESS 収録データ回収終了／PX4 ULog本体はSDに保持", flush=True)
    print(json.dumps(manifest,indent=2),flush=True)
    _EMERGENCY_CLEANUP = None
    print("PROGRESS 試験直後QC：保存済みの生荷重・RPM・姿勢・時計を確認中", flush=True)
    try:
        report = immediate_quality(out, manifest, qc_policy)
    except Exception as error:
        # A QC failure must never relabel a successfully persisted raw capture.
        report = summarize_quality([dict(name="qc-engine", decision="REVIEW", metrics={},
                                        reason=f"QC処理失敗: {type(error).__name__}: {error}")], qc_policy)
    with (out / "immediate-qc.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(render_quality(report), flush=True)
    manifest["immediate_qc"] = dict(decision=report["decision"], path="immediate-qc.json",
                                  physics_validated=False)
    manifest["capture_closed_utc_ns"] = time.time_ns()
    save()
    if args.require_qc_pass and report["decision"] != "PASS":
        return {"REVIEW": 2, "RETRY": 3}[report["decision"]]
    return int(manifest["state"] != "CAPTURED")


def main():
    """Ensure local exceptions cannot bypass the ordered remote cleanup."""
    global _EMERGENCY_CLEANUP
    break_signal = getattr(signal, "SIGBREAK", None)
    previous_break_handler = (signal.signal(break_signal, signal.default_int_handler)
                              if break_signal is not None else None)
    try:
        return _main()
    except BaseException as error:
        cleanup = _EMERGENCY_CLEANUP
        if cleanup is not None:
            try:
                cleanup(error)
            except BaseException as cleanup_error:
                print(f"Emergency cleanup failed: {type(cleanup_error).__name__}: {cleanup_error}",
                      file=sys.stderr, flush=True)
        raise
    finally:
        _EMERGENCY_CLEANUP = None
        if break_signal is not None:
            signal.signal(break_signal, previous_break_handler)


if __name__ == "__main__":
    raise SystemExit(main())
