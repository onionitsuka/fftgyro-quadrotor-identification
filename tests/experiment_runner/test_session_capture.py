import copy

import pytest

from fftgyro_quad_id.experiment_runner.session_capture import check_capture


def qualified(mode="AXIS_PID"):
    # Synthetic admission fixture, never written as an apparatus configuration.
    return dict(schema="fftgyro-axis-session-v1", runtime_qualified=True,
                control_mode=mode, maximum_duration_s=60,
                hardware=dict(uart_device="/dev/serial0", non_target_pid_enabled=False,
                              motor_mapping_verified=True, feedback_frame_verified=True,
                              stop_verified=True, rpm_logging_verified=True,
                              load_logging_verified=True, sync_verified=True,
                              mechanically_locked_verified=True,
                              fixed_input_mapping_verified=True),
                startup=dict(control_ready_verified=True))


def admit(c, **changes):
    options = dict(fixed=c["control_mode"] in ("FIXED_HOLD", "FIXED_INPUT"),
                   axis_ready=c["control_mode"] == "AXIS_PID", props_installed=True,
                   operator_ready=True, charged=True)
    options.update(changes)
    return check_capture(c, "/home/fftgyro/commissioning/test/fftgyro-executor", 100, **options)


@pytest.mark.parametrize("rate,duration,accepted", [(.02,13,True),(.1,13,False),(.02,14,False),
                                                  (0,13,False),(True,13,False),("0.02",13,False)])
def test_slow_fixed_duration(rate, duration, accepted):
    c = qualified("FIXED_HOLD")
    c["runtime_qualified"] = False
    c["maximum_duration_s"] = duration
    c["hardware"]["startup_ramp_rate"] = rate
    if accepted:
        assert admit(c, fixed_commissioning=True) == duration
    else:
        with pytest.raises(ValueError):
            admit(c, fixed_commissioning=True)


@pytest.mark.parametrize("mode", ["FIXED_HOLD", "AXIS_PID"])
def test_admission_preserves_configuration(mode):
    c = qualified(mode)
    original = copy.deepcopy(c)
    assert admit(c) == 60
    assert c == original


def test_rate_target_cap_mismatch_is_not_a_start_interlock():
    c = qualified("AXIS_PID")
    c["pid"] = {"axis": "PITCH", "cascade": {"target_rate_limit_deg_s": 30}}
    c["segments"] = [{"kind": "MOVE", "start_deg": -5, "end_deg": 5,
                      "duration_s": 0.3125}]
    assert admit(c) == 60


def test_qualified_fixed_hold_admits_explicit_props_removed():
    c = qualified("FIXED_HOLD")
    assert admit(c, props_installed=False, props_removed=True) == 60


def test_dynamic_session_rejects_props_removed():
    with pytest.raises(ValueError):
        admit(qualified("AXIS_PID"), props_installed=False, props_removed=True)


@pytest.mark.parametrize("flag", ["motor_mapping_verified", "feedback_frame_verified", "stop_verified",
                                  "rpm_logging_verified", "load_logging_verified", "sync_verified"])
def test_missing_qualification(flag):
    c = qualified()
    c["hardware"][flag] = False
    with pytest.raises(ValueError, match="qualification"):
        admit(c)


@pytest.mark.parametrize("field,value", [("runtime_qualified", False), ("runtime_qualified", "true"),
                                        ("control_mode", "TYPO"), ("maximum_duration_s", 61),
                                        ("maximum_duration_s", float("nan")),
                                        ("maximum_duration_s", True)])
def test_draft_or_bad_limits(field, value):
    c = qualified()
    c[field] = value
    with pytest.raises(ValueError):
        admit(c)


@pytest.mark.parametrize("mode,options", [("AXIS_PID", {"fixed": True}),
                                         ("AXIS_PID", {"axis_ready": False}),
                                         ("FIXED_HOLD", {"fixed": False}),
                                         ("FIXED_HOLD", {"axis_ready": True}),
                                         ("FIXED_HOLD", {"props_installed": False}),
                                         ("AXIS_PID", {"operator_ready": False}),
                                         ("AXIS_PID", {"charged": False})])
def test_wrong_apparatus_or_missing_confirmation(mode, options):
    with pytest.raises(ValueError):
        admit(qualified(mode), **options)


def test_fixed_lock_and_dynamic_authority_are_separate():
    c = qualified("FIXED_HOLD")
    c["startup"]["control_ready_verified"] = False
    assert admit(c) == 60
    c["hardware"]["mechanically_locked_verified"] = False
    with pytest.raises(ValueError):
        admit(c)


def test_fixed_input_requires_its_four_mode_mapping_qualification():
    c = qualified("FIXED_INPUT")
    assert admit(c) == 60
    c["hardware"]["fixed_input_mapping_verified"] = False
    with pytest.raises(ValueError, match="fixed_input_mapping_verified"):
        admit(c)
    c = qualified()
    c["startup"]["control_ready_verified"] = False
    with pytest.raises(ValueError):
        admit(c)


@pytest.mark.parametrize("mode", ["AXIS_PID", "FIXED_HOLD", "IDLE", "COMMISSIONING", "PITCH_COMMISSIONING"])
def test_coordinator_generated_workers_without_network(tmp_path, monkeypatch, mode):
    import importlib.util
    import io
    import json
    from pathlib import Path
    import sys
    from types import SimpleNamespace

    path = Path(__file__).resolve().parents[2] / "scripts/run_sync_bench.py"
    spec = importlib.util.spec_from_file_location("sync_bench_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    scripts = []

    class Input(io.StringIO):
        def close(self):
            scripts.append(self.getvalue())
            super().close()

    class Process:
        def __init__(self, *args, **kwargs):
            self.stdin = Input() if kwargs.get("stdin") == module.subprocess.PIPE else None
            self.returncode = None
            argv = args[0]
            if isinstance(argv, list) and "capture_frame_encoder.py" in " ".join(argv):
                output = Path(argv[argv.index("--output") + 1])
                output.mkdir()
                (output / "raw.bin").write_bytes(b"frame")
                (output / "samples.csv").write_text("receive_monotonic_ns\n1\n")
                (output / "metadata.json").write_text(json.dumps(
                    {"state": "COMPLETE", "encoder_frames": 100, "measured_rate_hz": 100}))
                Path(argv[argv.index("--ready-file") + 1]).write_text("ready")
            if (isinstance(argv, list) and "pi_clock_probe.py" in " ".join(argv)
                    and "probe" in argv):
                output = Path(argv[argv.index("--output") + 1])
                output.mkdir()
                (output / "exchanges.csv").write_text("synthetic")
                (output / "summary.json").write_text(json.dumps(
                    {"state": "COMPLETE", "server_boot_id": "boot-id"}))
                Path(argv[argv.index("--ready-file") + 1]).write_text("ready")
            kwargs["stdout"].write("Temporary clock responder ready\nFORCE_STREAM_READY\n")
            kwargs["stdout"].flush()

        def poll(self):
            return self.returncode

        def wait(self, **kwargs):
            self.returncode = 0
            return 0

    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        if command[0] == 'sftp':
            staging = Path(kwargs['cwd'])
            for line in kwargs.get('input', '').splitlines():
                name = line.rsplit('"', 2)[1]
                target = staging / name
                if name == 'ram-storage.json':
                    target.write_text(json.dumps({'state': 'PERSISTED_VERIFIED'}))
                else:
                    target.mkdir()
        if kwargs.get("input") and command[0] != "sftp":
            compile(kwargs["input"], "remote-validation", "exec")
        if "/home/fftgyro/venvs/mavlink/bin/python -" in command:
            reference = {"summary": {"quaternion_wxyz": [.5, .5, .5, .5],
                                      "samples": 500, "armed": False,
                                      "max_reference_deviation_deg": .1,
                                      "max_body_rate_deg_s": [.1, .2, .1],
                                      "calibration_changed": False,
                                      "motor_commands_sent": False},
                         "raw_samples": []}
            return SimpleNamespace(returncode=0, stdout=json.dumps(reference), stderr="")
        return SimpleNamespace(returncode=0, stdout=str(4096 * len(calls)), stderr="")

    monkeypatch.setattr(module.subprocess, "Popen", Process)
    monkeypatch.setattr(module.subprocess, "run", run)
    def check_output(command, **kwargs):
        if "df -Pk" in " ".join(command):
            return "/dev/mmcblk0p2 60000000 1000000 59000000 2% /home/fftgyro/research-data/fftgyro\n"
        return "boot-id\n"
    monkeypatch.setattr(module.subprocess, "check_output", check_output)
    monkeypatch.setattr(module.time, "sleep", lambda _: None)
    config = tmp_path / "source.json"
    data = qualified("FIXED_HOLD" if mode == "COMMISSIONING" else mode)
    if mode == "COMMISSIONING":
        data.update(runtime_qualified=False, maximum_duration_s=12)
    if mode == "PITCH_COMMISSIONING":
        data.update(runtime_qualified=False, maximum_duration_s=12, control_mode="AXIS_PID", pid={"axis": "PITCH"})
        data["hardware"]["mechanically_locked_verified"] = False
        data["startup"]["control_ready_verified"] = False
    config.write_text(json.dumps(data))
    arguments = ["run_sync_bench.py", "--data-root", str(tmp_path), "--run", "mock",
                 "--seconds", "100", "--props-installed"]
    arguments += (["--idle", "--fixed"] if mode == "IDLE" else [
        "--session-config", str(config), "--session-executor", "/home/fftgyro/commissioning/test/executor",
        "--fixed" if mode in ("FIXED_HOLD", "COMMISSIONING") else "--axis-ready", "--operator-stop-ready", "--charged-4s"])
    if mode == "COMMISSIONING":
        arguments.append("--fixed-commissioning")
    if mode == "PITCH_COMMISSIONING":
        arguments.append("--pitch-hold-commissioning")
    monkeypatch.setattr(sys, "argv", arguments)
    assert module.main() == 0
    assert len(scripts) == 3
    for script in scripts:
        compile(script, "remote-worker", "exec")
    aircraft = scripts[-1]
    leptrino = scripts[-2]
    manifest = json.loads((tmp_path / "mock/run.json").read_text())
    assert manifest["state"] == "CAPTURED" and manifest["props_installed"]
    assert "capture-stop" in leptrino
    assert "capture.pids" not in leptrino
    assert leptrino.index("rc=force.wait(timeout=10)") < leptrino.index("(p/'clock-stop').touch")
    assert leptrino.index("clock-ready") < leptrino.index("force=subprocess.Popen")
    assert "force_protocol=source/'leptrino.py'" in leptrino
    assert not any("stat -c" in str(command) for command, _ in calls)
    assert not any("kill -TERM" in str(command) for command, _ in calls)
    transfers = [(command, opts) for command, opts in calls if command[0] == "sftp"]
    assert len(transfers) == 2
    assert transfers[0][0][-1] == module.AIR
    assert transfers[1][0][-1] == module.LEP
    for command, opts in transfers:
        assert command[1:3] == ["-b", "-"] and opts["cwd"] != tmp_path / "mock"
        assert "source-session" not in opts["input"] and "*" not in opts["input"]
        assert all(line.startswith('get -R "') for line in opts["input"].splitlines())
    batches = "".join(opts["input"] for _, opts in transfers)
    assert '/clock" "clock"' in batches and '/leptrino" "leptrino"' in batches
    # Execute the generated collector supervisor with synthetic child processes.
    # Its parent request must close force before asking clock for the final pair.
    remote_base = "/home/fftgyro/research-data/fftgyro/mock"
    owned = tmp_path / "synthetic-leptrino-worker"
    worker_text = leptrino.replace(remote_base, owned.as_posix())
    ram_root = tmp_path / 'ram'
    ram_root.mkdir()
    worker_text = worker_text.replace('spool=RamCapture(p,100)',
        f"spool=RamCapture(p,100,Path({ram_root.as_posix()!r}))")
    events = []

    class Child:
        def __init__(self, argv):
            self.kind = "clock" if "probe" in argv else "force"
            self.output = Path(argv[argv.index('--output') + 1])
            self.output.mkdir()
            self.returncode = None
            self.stopping = False
            if self.kind == "clock":
                (owned / "clock-ready").write_text("synthetic exchange")
            else:
                (owned / "capture-stop").touch()

        def poll(self):
            return self.returncode

        def terminate(self):
            self.stopping = True
            events.append(self.kind + " stop")

        def wait(self, **kwargs):
            if self.kind == "force":
                assert self.stopping
                assert not (owned / "clock-stop").exists()
            else:
                assert (owned / "clock-stop").exists()
                assert events[-1] == "force closed"
                (self.output / "summary.json").write_text(json.dumps(
                    dict(coordinated_stop=True, final_exchange_after_stop=True)))
            events.append(self.kind + " closed")
            self.returncode = 0
            return 0

    with monkeypatch.context() as worker_patch:
        worker_patch.setattr(module.subprocess, "Popen", Child)
        exec(compile(worker_text, "synthetic-collector-supervisor", "exec"), {})
    assert events == ["force stop", "force closed", "clock closed"]
    assert json.loads((owned / 'ram-storage.json').read_text())['state'] == 'PERSISTED_VERIFIED'
    assert not list(ram_root.iterdir())
    assert "if rpm.poll() is None: rpm.terminate()" in aircraft
    assert "rpm.wait(timeout=5)" in aircraft
    assert "handles SIGTERM as a coordinated close" in aircraft
    assert "PROGRESS モーター出力停止／RPM・PX4ログ終了処理" in aircraft
    assert "if stop_request.exists():" in aircraft
    assert "coordinator requested safe stop" in aircraft
    if mode == "IDLE":
        assert manifest["motor_profile"] == "NONE" and manifest["idle"]
        assert "if True:\n  if False:" in aircraft
        assert "rr=rpm.wait" in aircraft
        assert not (tmp_path / "mock/source-session.json").exists()
    else:
        expected_limit = 17.0 if mode in ("COMMISSIONING", "PITCH_COMMISSIONING") else 65.0
        assert "'session-run'" in aircraft and f"deadline=time.monotonic()+{expected_limit}" in aircraft
        assert ("bounded commissioning" if mode in ("COMMISSIONING", "PITCH_COMMISSIONING") else mode) in manifest["motor_profile"]
        if mode == "PITCH_COMMISSIONING":
            assert "'session-pitch-hold-commissioning'" in aircraft and "'--pitch-only-free'" in aircraft
            assert "session-pitch-hold-commissioning-validate" in calls[1][1]["input"]
            assert manifest["pitch_hold_commissioning"] and not manifest["fixed"]
            assert manifest["per_run_attitude_reference"]
            assert manifest["attitude_reference_summary"]["samples"] == 500
            assert json.loads((tmp_path / "mock/resolved-session.json").read_text())["pid"]["zero_quaternion_wxyz"] == [.5, .5, .5, .5]
            assert (tmp_path / "mock/attitude-reference.json").exists()
        assert (tmp_path / "mock/source-session.json").read_text() == config.read_text()


def test_main_wrapper_runs_registered_cleanup_and_preserves_error(monkeypatch):
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "scripts/run_sync_bench.py"
    spec = importlib.util.spec_from_file_location("sync_bench_cleanup_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    events = []

    def fail_after_start():
        module._EMERGENCY_CLEANUP = lambda error: events.append((type(error).__name__, str(error)))
        raise RuntimeError("synthetic coordinator failure")

    monkeypatch.setattr(module, "_main", fail_after_start)
    with pytest.raises(RuntimeError, match="synthetic coordinator failure"):
        module.main()
    assert events == [("RuntimeError", "synthetic coordinator failure")]
    assert module._EMERGENCY_CLEANUP is None


def test_cleanup_failure_never_masks_original_error(monkeypatch, capsys):
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "scripts/run_sync_bench.py"
    spec = importlib.util.spec_from_file_location("sync_bench_cleanup_failure_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def fail_after_start():
        def broken_cleanup(unused):
            raise OSError("cleanup transport failed")
        module._EMERGENCY_CLEANUP = broken_cleanup
        raise RuntimeError("original coordinator failure")

    monkeypatch.setattr(module, "_main", fail_after_start)
    with pytest.raises(RuntimeError, match="original coordinator failure"):
        module.main()
    assert "cleanup transport failed" in capsys.readouterr().err
    assert module._EMERGENCY_CLEANUP is None


def test_fixed_commissioning_does_not_promote_flags():
    c = qualified("FIXED_HOLD")
    c.update(runtime_qualified=False, maximum_duration_s=12)
    for key in c["hardware"]:
        if key.endswith("_verified"):
            c["hardware"][key] = False
    original = copy.deepcopy(c)
    assert admit(c, fixed_commissioning=True) == 12
    assert c == original
    for key, value in (("runtime_qualified", True), ("control_mode", "AXIS_PID"),
                       ("maximum_duration_s", 13), ("maximum_duration_s", float("nan"))):
        bad = copy.deepcopy(c)
        bad[key] = value
        with pytest.raises(ValueError):
            admit(bad, fixed_commissioning=True)


def test_pitch_commissioning_preserves_draft_and_requires_mapping():
    c = qualified()
    c.update(runtime_qualified=False, maximum_duration_s=12, pid={"axis": "PITCH"})
    c["hardware"]["mechanically_locked_verified"] = False
    c["startup"]["control_ready_verified"] = False
    for flag in ("stop_verified", "rpm_logging_verified", "load_logging_verified", "sync_verified"):
        c["hardware"][flag] = False
    original = copy.deepcopy(c)
    assert admit(c, pitch_commissioning=True) == 12
    assert c == original
    for options in ({"fixed": True}, {"axis_ready": False}, {"fixed_commissioning": True},
                    {"props_installed": False}, {"charged": False}, {"operator_ready": False}):
        with pytest.raises(ValueError):
            admit(c, pitch_commissioning=True, **options)
    for field, value in (("runtime_qualified", True), ("maximum_duration_s", 13),
                         ("maximum_duration_s", float("nan")), ("maximum_duration_s", True)):
        bad = copy.deepcopy(c)
        bad[field] = value
        with pytest.raises(ValueError):
            admit(bad, pitch_commissioning=True)
    for flag in ("motor_mapping_verified", "feedback_frame_verified"):
        bad = copy.deepcopy(c)
        bad["hardware"][flag] = False
        with pytest.raises(ValueError):
            admit(bad, pitch_commissioning=True)
    bad = copy.deepcopy(c)
    bad["pid"]["axis"] = "YAW"
    with pytest.raises(ValueError):
        admit(bad, pitch_commissioning=True)


def test_clock_readiness_waits_for_remote_bind(tmp_path, monkeypatch):
    import importlib.util
    from pathlib import Path
    from types import SimpleNamespace
    path = Path(__file__).resolve().parents[2] / "scripts/run_sync_bench.py"
    spec = importlib.util.spec_from_file_location("readiness_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    log = tmp_path / "server.txt"
    log.write_text("")
    monkeypatch.setattr(module.time, "monotonic", lambda: 0)
    sleeps = []
    def start_after_wait(seconds):
        sleeps.append(seconds)
        log.write_text("Temporary clock responder ready\n")
    monkeypatch.setattr(module.time, "sleep", start_after_wait)
    module.wait_clock_ready(SimpleNamespace(poll=lambda: None), log)
    assert sleeps == [0.1]
    with pytest.raises(RuntimeError, match="exited"):
        module.wait_clock_ready(SimpleNamespace(poll=lambda: 1), log)
    log.write_text("")
    clock = iter([0, 31])
    monkeypatch.setattr(module.time, "monotonic", lambda: next(clock))
    with pytest.raises(RuntimeError, match="timeout"):
        module.wait_clock_ready(SimpleNamespace(poll=lambda: None), log)
