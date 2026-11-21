"""Mock shell/clock/launcher only. Never connects to a flight controller."""

import ast
import base64
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from fftgyro_quad_id.experiment_runner import px4_trial_log as logger

EPOCH = 1789045200


class FakeShell:
    def __init__(self, overrides=None, parameters=None, storage=None):
        self.commands = []
        self.overrides = overrides or {}
        self.parameters = parameters or dict(logger.LIGHTWEIGHT_LOGGING_PARAMETERS)
        self.storage = storage or SimpleNamespace(
            status=2, storage_id=2, storage_count=1, name="microSD",
            total_capacity=32768.0, used_capacity=1024.0,
            available_capacity=31744.0,
        )

    def parameter(self, name):
        return self.parameters[name]

    def storage_information(self):
        return self.storage

    def system_time_anchor(self):
        return {
            "px4_boot_us": 12_345_000,
            "aircraft_monotonic_lower_ns": 70_000_000_000,
            "aircraft_monotonic_upper_ns": 70_001_000_000,
        }

    def command(self, command):
        self.commands.append(command)
        if command in self.overrides:
            return self.overrides[command]
        if command == "logger status":
            if "logger on" in self.commands and "logger off" not in self.commands:
                return "Full File Logging Running:\nLog file: /fs/microsd/log/2026-09-10/20260910_001001_001.ulg"
            return "Not logging"
        if command.startswith("system_time set "):
            return "Successfully set system time"
        if command == "system_time get":
            return f"Unix epoch time: {EPOCH}"
        return ""


def test_pymavlink_instance_workaround_rebuilds_missing_instance_cache():
    calls = []

    class Message:
        _instance_field = "instance"

        def __init__(self, instance):
            self.instance = instance
            self._instances = None

    def original(messages, message_type, message):
        calls.append((message_type, message.instance))
        if message_type not in messages:
            message._instances = {}
            messages[message_type] = message
        messages[message_type]._instances[message.instance] = message

    mavutil = SimpleNamespace(add_message=original)
    logger.install_pymavlink_instance_workaround(mavutil)
    messages = {"BATTERY_STATUS": Message(None)}
    incoming = Message(1)
    mavutil.add_message(messages, "BATTERY_STATUS", incoming)
    assert calls == [("BATTERY_STATUS", 1)]
    assert messages["BATTERY_STATUS"]._instances[1] is incoming


def battery_message(message_type, millivolts, battery_id=0):
    values = [millivolts] + [65535] * 9
    return SimpleNamespace(
        get_type=lambda: message_type,
        get_srcSystem=lambda: 1,
        get_srcComponent=lambda: 1,
        voltage_battery=millivolts,
        voltages=values,
        id=battery_id,
    )


def test_battery_preflight_uses_live_px4_samples_and_records_limits(monkeypatch):
    messages = iter([battery_message("SYS_STATUS", 14770) for _ in range(3)])
    shell = SimpleNamespace(link=SimpleNamespace(recv_match=lambda **kwargs: next(messages)))
    record = {}
    saved = []
    measurement = logger.verify_battery_preflight(
        shell, record, lambda: saved.append(dict(record)), 14.5, 16.9)
    assert measurement["voltage_v"] == pytest.approx(14.77)
    assert measurement["passed"] is True
    assert measurement["sources"] == ["SYS_STATUS"]
    assert saved[-1]["battery_preflight"]["allowed_min_v"] == 14.5


def test_battery_preflight_blocks_before_logger_when_outside_limits(monkeypatch):
    monkeypatch.setattr(logger, "sample_battery_voltage", lambda shell: {
        "voltage_v": 14.2, "observed_min_v": 14.2, "observed_max_v": 14.2,
        "samples": 3, "sources": ["BATTERY_STATUS"],
    })
    record = {}
    with pytest.raises(RuntimeError, match="no logger or motor start"):
        logger.verify_battery_preflight(object(), record, lambda: None, 14.5, 16.9)
    assert record["battery_preflight"]["passed"] is False


def test_battery_voltage_rejects_secondary_instance_and_sentinel():
    assert logger._battery_voltage_v(battery_message("BATTERY_STATUS", 14770, 1)) is None
    assert logger._battery_voltage_v(battery_message("BATTERY_STATUS", 65535)) is None


@pytest.mark.parametrize(
    ("available_mib", "passed"),
    [(14336, True), (1024, False), (1000, False), (1025, True)],
)
def test_pixhawk_sd_preflight_blocks_at_one_gib(available_mib, passed):
    storage = SimpleNamespace(
        status=2, storage_id=2, storage_count=1, name="microSD",
        total_capacity=32768.0, used_capacity=32768.0 - available_mib,
        available_capacity=float(available_mib),
    )
    shell = FakeShell(storage=storage)
    record = {}
    saved = []
    if passed:
        logger.verify_sd_capacity_preflight(shell, record, lambda: saved.append(dict(record)))
    else:
        with pytest.raises(RuntimeError, match="no logger or motor start"):
            logger.verify_sd_capacity_preflight(shell, record, lambda: saved.append(dict(record)))
    assert record["sd_capacity_preflight"]["passed"] is passed
    assert saved and shell.commands == []


def test_pixhawk_capacity_uses_storage_information_not_wrapped_df():
    shell = FakeShell(
        overrides={"df -h": "vfat 883M 416K 882M /fs/microsd"},
        storage=SimpleNamespace(
            status=2, storage_id=2, storage_count=1, name="KIOXIA",
            total_capacity=29563.0, used_capacity=1.0,
            available_capacity=29562.0,
        ),
    )
    record = {}
    logger.verify_sd_capacity_preflight(shell, record, lambda: None)
    assert record["sd_capacity_preflight"]["passed"] is True
    assert record["sd_capacity_preflight"]["source"] == "MAVLink STORAGE_INFORMATION"
    assert "df -h" not in shell.commands


def test_storage_information_retries_transient_timeout():
    sends = []
    storage = SimpleNamespace(
        get_type=lambda: "STORAGE_INFORMATION",
        get_srcSystem=lambda: 1,
        get_srcComponent=lambda: 1,
        storage_count=1,
    )

    class Link:
        mav = SimpleNamespace(command_long_send=lambda *args: sends.append(args))

        def recv_match(self, **kwargs):
            return storage if len(sends) == 2 else None

    shell = logger.Shell.__new__(logger.Shell)
    shell.link = Link()
    # First request reaches its deadline; the second receives the response.
    clock = iter([0.0, 0.1, 1.1, 2.0, 2.1]).__next__
    assert shell.storage_information(timeout_s=1, attempts=3, monotonic=clock) is storage
    assert len(sends) == 2


def test_system_time_anchor_brackets_px4_boot_clock_on_pi_monotonic():
    sends = []
    message = SimpleNamespace(
        get_type=lambda: "SYSTEM_TIME",
        get_srcSystem=lambda: 1,
        get_srcComponent=lambda: 1,
        time_boot_ms=12_345,
        time_unix_usec=1_789_045_200_123_456,
    )

    class Link:
        mav = SimpleNamespace(command_long_send=lambda *args: sends.append(args))

        def recv_match(self, **kwargs):
            return message

    shell = logger.Shell.__new__(logger.Shell)
    shell.link = Link()
    clock = iter([70_000_000_000, 70_000_100_000, 70_000_200_000]).__next__
    anchor = shell.system_time_anchor(monotonic_ns=clock)
    assert anchor["px4_boot_us"] == 12_345_000
    assert anchor["aircraft_monotonic_lower_ns"] == 70_000_000_000
    assert anchor["aircraft_monotonic_upper_ns"] == 70_000_200_000
    assert anchor["interval_width_ms"] == pytest.approx(0.2)
    assert len(sends) == 1


@pytest.mark.parametrize("status,total,used,available", [
    (1, 32768.0, 0.0, 32768.0),
    (2, float("nan"), 0.0, 1000.0),
    (2, 32768.0, 0.0, -1.0),
])
def test_pixhawk_sd_preflight_rejects_unready_or_invalid_storage(
        status, total, used, available):
    shell = FakeShell(storage=SimpleNamespace(
        status=status, storage_id=2, storage_count=1, name="microSD",
        total_capacity=total, used_capacity=used, available_capacity=available,
    ))
    record = {}
    with pytest.raises(RuntimeError, match="not ready or invalid"):
        logger.verify_sd_capacity_preflight(shell, record, lambda: None)
    assert "logger on" not in shell.commands


def test_lightweight_logging_profile_is_recorded_and_accepted():
    shell = FakeShell()
    record = {}
    saved = []
    result = logger.verify_logging_profile_preflight(
        shell, record, lambda: saved.append(dict(record)))
    assert result["passed"] is True
    assert result["actual"] == {
        "SDLOG_MODE": 0,
        "SDLOG_PROFILE": 9,
        "SDLOG_BACKEND": 1,
        "SDLOG_MISSION": 0,
    }
    assert saved[-1]["logging_profile_preflight"]["passed"] is True


def test_heavy_logging_profile_blocks_before_logger_or_motor_start():
    shell = FakeShell(parameters={
        "SDLOG_MODE": 0,
        "SDLOG_PROFILE": 57,
        "SDLOG_BACKEND": 3,
        "SDLOG_MISSION": 0,
    })
    record = {}
    with pytest.raises(RuntimeError, match="set parameters and reboot Pixhawk"):
        logger.verify_logging_profile_preflight(shell, record, lambda: None)
    assert record["logging_profile_preflight"]["mismatches"] == {
        "SDLOG_PROFILE": {"actual": 57, "expected": 9},
        "SDLOG_BACKEND": {"actual": 3, "expected": 1},
    }
    assert "logger on" not in shell.commands


def start(shell, record=None, now=None, save=None):
    record = record if record is not None else {}
    now = now or iter([EPOCH + .1, EPOCH + .2]).__next__
    logger.start_logger(shell, record, save or (lambda: None), now=now, monotonic=lambda: 10.)
    return record


def test_every_start_sets_and_verifies_before_logger_on():
    for _ in range(2):
        shell = FakeShell()
        saved = []
        record = {}
        start(shell, record, save=lambda: saved.append(dict(record)))
        assert shell.commands == ["logger status", f"system_time set {EPOCH}",
                                  "system_time get", "logger on", "logger status"]
        assert saved[0]["utc_verified"] and saved[0]["logger_on_attempted"]
        assert record["logger_running"] and record["ulog_path"].endswith(".ulg")
        assert record["px4_clock_anchor"]["px4_boot_us"] == 12_345_000


@pytest.mark.parametrize("overrides", [
    {"logger status": "Full File Logging Running:"},
    {"logger status": "logger: command not found"},
    {f"system_time set {EPOCH}": "Failed to set system time"},
    {"system_time get": "read timeout"},
    {"system_time get": f"Unix epoch time: {EPOCH - 10}"},
    {"system_time get": f"Unix epoch time: {EPOCH + 100}"},
    {"system_time get": f"Unix epoch time: {EPOCH}\nUnix epoch time: {EPOCH}"},
])
def test_bad_clock_or_existing_log_never_starts(overrides):
    shell = FakeShell(overrides)
    with pytest.raises(RuntimeError):
        start(shell)
    assert "logger on" not in shell.commands


def test_wall_clock_step_prevents_start():
    shell = FakeShell()
    with pytest.raises(RuntimeError, match="stepped"):
        start(shell, now=iter([EPOCH, EPOCH + 10]).__next__)
    assert "logger on" not in shell.commands


def test_stop_does_not_touch_utc():
    shell = FakeShell()
    record = {}
    logger.stop_logger(shell, record)
    assert shell.commands == ["logger status", "logger off", "logger status"]
    assert record["logger_stopped"]


@pytest.mark.parametrize("listing,expected", [
    (" -rw-rw-rw-  42000 20260910_001001_001.ulg\n", True),
    (" -rw-rw-rw-42000 /fs/microsd/log/2026-09-10/20260910_001001_001.ulg\n", True),
    (" -rw-rw-rw-  0 20260910_001001_001.ulg\n", False),
    ("ls: no such file\n", False),
    ("nsh> ls -l /fs/microsd/log/2026-09-10/20260910_001001_001.ulg\n", False),
    (" -rw-rw-rw-  42000 other.ulg\n", False),
])
def test_sd_size_evidence(listing, expected):
    path = "/fs/microsd/log/2026-09-10/20260910_001001_001.ulg"
    shell = FakeShell({"ls -l " + path: listing})
    record = {}
    logger.verify_sd_log(shell, record, {"ulog_path": path})
    assert record["sd_log_verified"] is expected
    assert record["content_verified"] is False


def test_mib_logger_progress_is_normalized_to_kib():
    path = "/fs/microsd/log/2026-09-10/20260910_001001_001.ulg"
    class MiBShell(FakeShell):
        def command(self, command):
            if command == "logger status" and "logger off" not in self.commands:
                self.commands.append(command)
                return f"Full File Logging Running:\nLog file: {path}\nWrote 13.08 MiB\ndropouts: 0"
            return super().command(command)

    shell = MiBShell({"ls -l " + path: f" -rw-rw-rw-13715374 {path}\n"})
    record = {}
    logger.stop_logger(shell, record, start_record={"ulog_path": path})
    assert record["written_kib"] == [13.08 * 1024]
    assert record["sd_log_verified"] is True


def test_status_failure_cannot_skip_logger_off():
    class BrokenStatus(FakeShell):
        def command(self, cmd):
            if cmd == "logger status" and not self.commands:
                self.commands.append(cmd)
                raise RuntimeError("timeout")
            return super().command(cmd)
    shell = BrokenStatus()
    record = {}
    logger.stop_logger(shell, record, start_record={})
    assert "logger off" in shell.commands
    assert record["logger_stopped"]
    assert not record["sd_log_verified"]


def test_sd_path_injection_not_sent():
    shell = FakeShell()
    record = {}
    logger.verify_sd_log(shell, record, {"ulog_path": "/fs/microsd/log/a.ulg; reboot"})
    assert not shell.commands
    assert not record["sd_log_verified"]


def test_shell_waits_for_actual_marker_not_command_echo(monkeypatch):
    monkeypatch.setattr(logger.secrets, "token_hex", lambda _: "0123456789abcdef")
    chunks = iter(["nsh> echo FG0123456789abcdef\r\n", "Unix epoch time: 123\r\n",
                   "FG0123456789abcdef\r\nnsh> "])
    sent = []
    reads = []

    def receive(**kwargs):
        data = next(chunks).encode()
        reads.append(data)
        return SimpleNamespace(get_srcSystem=lambda: 1, get_srcComponent=lambda: 1,
                               get_type=lambda: "SERIAL_CONTROL", device=10, data=list(data),
                               count=len(data))

    shell = logger.Shell.__new__(logger.Shell)
    shell.link = SimpleNamespace(mav=SimpleNamespace(serial_control_send=lambda *v: sent.append(v)),
                                 recv_match=receive)
    response = shell.command("system_time get")
    assert len(reads) == 3 and "Unix epoch time: 123" in response
    assert sent[0][:2] == (10, 2)


def test_cascade_capture_admits_native_fifteen_second_budget():
    from fftgyro_quad_id.experiment_runner.session_capture import check_capture
    root = Path(__file__).resolve().parents[2]
    config = json.loads((root / "onboard_executor/config/pitch-cascade-commissioning.example.json").read_text())
    config["hardware"].update(motor_mapping_verified=True, feedback_frame_verified=True)
    assert check_capture(config, "/home/fftgyro/synthetic/fftgyro-executor", 80,
                         fixed=False, axis_ready=True, props_installed=True, operator_ready=True,
                         charged=True, pitch_commissioning=True) == 15


@pytest.mark.parametrize("value,ok", [("yes\n", True), ("true\n", True), ("no\n", False), ("", False)])
def test_pi_clock_sync_required(monkeypatch, value, ok):
    monkeypatch.setattr(logger.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=value))
    if ok:
        logger.require_synced_utc()
    else:
        with pytest.raises(RuntimeError, match="NTP"):
            logger.require_synced_utc()


def generated_air_script(tmp_path, idle=False):
    root = Path(__file__).resolve().parents[2]
    tree = ast.parse((root / "scripts/run_sync_bench.py").read_text(encoding="utf-8"))
    call = next(n for n in ast.walk(tree) if isinstance(n, ast.Call) and
                isinstance(n.func, ast.Name) and n.func.id == "launch" and
                len(n.args) == 3 and isinstance(n.args[2], ast.Constant) and
                n.args[2].value == "aircraft")
    config = {"hardware": {}, "session_id": "input"}
    args = SimpleNamespace(idle=idle, idle_px4_log=False,
                           session_executor="/synthetic/executor", fixed_commissioning=False,
                           pitch_hold_commissioning=True, yaw_hold_commissioning=False,
                           session_props_off=False, run="synthetic",
                           battery_min_v=14.5, battery_max_v=16.9)
    env = dict(base=(tmp_path / "run").as_posix(), args=args, seconds=80, session_limit=15,
               session_text=json.dumps(config), trial_log_encoded=base64.b64encode(b"# fake").decode(),
               session_encoded=base64.b64encode(json.dumps(config).encode()).decode(),
               RPM_READY_HOLD_S=1, RPM_POST_ROLL_S=1,
               AIRCRAFT_PROGRESS_INTERVAL_S=2,
               air_stop_path=(tmp_path / "air-stop").as_posix(),
               air_runtime=(tmp_path / "runtime").as_posix())
    return eval(compile(ast.Expression(call.args[1]), "air-template", "eval"), env)


@pytest.mark.parametrize("failure,expected", [
    (None, ["rpm", "start", "pid", "stop"]),
    ("clock", ["rpm", "start"]),
    ("after_on", ["rpm", "start", "stop"]),
])
def test_coordinator_orders_logger_and_motor_processes(tmp_path, monkeypatch, failure, expected):
    events = []

    class Process:
        def __init__(self, command):
            self.returncode = None
            if command[0].endswith("fftgyro-rpm"):
                events.append("rpm")
                path = Path(command[2])
                path.mkdir()
                (path / "rpm.csv").write_text("ready" * 30)
            else:
                events.append("pid")
                config = json.loads(Path(command[2]).read_text())
                (Path(config["hardware"]["log_directory"]) / "pid").mkdir()
                self.returncode = 0

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            self.returncode = 0
            return 0

        def terminate(self):
            self.returncode = 0

    def run(command, **kwargs):
        action = command[3]
        events.append(action)
        output = Path(command[command.index("--output") + 1])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({"logger_on_attempted": failure != "clock",
                                      "complete": not (action == "start" and failure),
                                      "logger_stopped": action == "stop",
                                      "battery_preflight": {"passed": True, "voltage_v": 16.0}}))
        if action == "start" and failure:
            raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(subprocess, "Popen", Process)
    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(logger.time, "sleep", lambda _: None)
    script = generated_air_script(tmp_path)
    if failure:
        with pytest.raises(subprocess.CalledProcessError):
            exec(compile(script, "synthetic-air-script", "exec"), {})
    else:
        exec(compile(script, "synthetic-air-script", "exec"), {})
    assert events == expected
