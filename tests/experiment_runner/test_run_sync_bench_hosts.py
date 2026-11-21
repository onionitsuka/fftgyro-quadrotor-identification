import importlib.util
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("run_sync_bench", ROOT / "scripts/run_sync_bench.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_leptrino_uses_stable_ip_with_known_host_alias() -> None:
    command = MODULE.remote(MODULE.LEP, "true")
    assert MODULE.LEP == "fftgyro@192.0.2.80"
    assert "HostKeyAlias=leptrino-pi.local" in command


def test_aircraft_does_not_reuse_leptrino_host_key_alias() -> None:
    command = MODULE.remote(MODULE.AIR, "true")
    assert "HostKeyAlias=leptrino-pi.local" not in command


def test_deferred_qc_runs_readonly_packaged_checks(monkeypatch):
    import json
    from types import SimpleNamespace

    calls = []

    def run(command, **options):
        calls.append((command, options))
        compile(options["input"], "qc-worker", "exec")
        assert "TemporaryDirectory" in options["input"]
        assert "payload['root']" in options["input"]
        return SimpleNamespace(stdout=json.dumps([
            dict(name="rpm", decision="PASS", metrics={}, reason="test")
        ]))

    monkeypatch.setattr(MODULE.subprocess, "run", run)
    result = MODULE.remote_quality(MODULE.AIR, "/capture/run", {}, ["rpm"], None)
    assert result[0]["decision"] == "PASS"
    assert calls[0][0][-1] == "python3 -"
    assert calls[0][1]["timeout"] == 120


def test_remote_qc_failure_does_not_become_pass_or_mutate_capture(tmp_path, monkeypatch):
    import json

    manifest = dict(state="CAPTURED", bulk_download_deferred=True, remote_base="/capture/run",
                    idle=True)
    monkeypatch.setattr(MODULE, "inspect_quality", lambda *a: [])

    def remote_failure(*args):
        raise subprocess.TimeoutExpired("ssh", 120)

    monkeypatch.setattr(MODULE, "remote_quality", remote_failure)
    result = MODULE.immediate_quality(tmp_path, manifest, {})
    assert result["decision"] == "REVIEW"
    assert manifest["state"] == "CAPTURED"
    assert not result["automatic_motor_retry"]


def test_run_transport_does_not_abort_on_serveralive_false_failure() -> None:
    command = MODULE.remote(MODULE.AIR, "true")
    assert not any("ServerAlive" in value for value in command)
    assert "ConnectTimeout=8" in command


def test_aircraft_progress_is_throttled_without_reducing_capture_rates() -> None:
    assert MODULE.AIRCRAFT_PROGRESS_INTERVAL_S == .2


def test_idle_px4_log_has_an_all_sensor_ready_marker() -> None:
    source = (ROOT / "scripts/run_sync_bench.py").read_text(encoding="utf-8")
    assert "--idle-px4-log" in source
    assert "PROGRESS IDLE_CAPTURE_READY" in source
    assert source.index("flightlog('start')", source.index("if {args.idle!r}")) < source.index(
        "PROGRESS IDLE_CAPTURE_READY"
    )


def test_reference_capture_repairs_pymavlink_instance_cache_before_heartbeat() -> None:
    script = MODULE.REFERENCE_CAPTURE_SCRIPT
    assert "mavutil.add_message = _add_message" in script
    assert script.index("mavutil.add_message = _add_message") < script.index(
        "mavutil.mavlink_connection"
    )


def test_storage_capacity_reads_both_pis_in_parallel(monkeypatch) -> None:
    calls = []

    def check_output(command, **options):
        calls.append((command, options))
        return "/dev/mmcblk0p2 1000000 250000 750000 25% /\n"

    monkeypatch.setattr(MODULE.subprocess, "check_output", check_output)
    capacity = MODULE.read_storage_capacity()

    assert len(calls) == 2
    assert {command[-2] for command, _ in calls} == {MODULE.AIR, MODULE.LEP}
    assert all("df -Pk" in command[-1] for command, _ in calls)
    assert capacity["aircraft"]["available_bytes"] == 750000 * 1024
    assert capacity["leptrino"]["available_percent"] == 75.0
    assert "aircraft 0.7 GiB" in MODULE.storage_progress(capacity)


def test_storage_capacity_rejects_malformed_df(monkeypatch) -> None:
    monkeypatch.setattr(MODULE.subprocess, "check_output", lambda *args, **kwargs: "bad\n")
    with pytest.raises(RuntimeError, match="invalid df response"):
        MODULE.read_storage_capacity()


@pytest.mark.parametrize(
    ("aircraft_free", "leptrino_free", "expected"),
    [
        (2 * 1024 ** 3, 2 * 1024 ** 3, []),
        (1024 ** 3, 2 * 1024 ** 3, ["aircraft"]),
        (2 * 1024 ** 3, 1024 ** 3 - 1, ["leptrino"]),
        (1024 ** 3 - 1, 1024 ** 3, ["aircraft", "leptrino"]),
    ],
)
def test_storage_limit_blocks_either_pi_at_one_gib(
    aircraft_free, leptrino_free, expected
) -> None:
    storage = {
        "aircraft": {"available_bytes": aircraft_free},
        "leptrino": {"available_bytes": leptrino_free},
    }
    assert MODULE.storage_below_limit(storage) == expected


def test_console_selection_snapshot_is_an_unused_reservation_file(tmp_path) -> None:
    for name in (
        "operator-result.json",
        "operator-selection.json",
        "operator-session.json",
        "console.log",
        "wind-condition.json",
    ):
        (tmp_path / name).write_text("{}", encoding="utf-8")
    assert MODULE.unused_console_reservation(tmp_path)
    (tmp_path / "run.json").write_text("{}", encoding="utf-8")
    assert not MODULE.unused_console_reservation(tmp_path)


def test_remote_stop_retries_one_transport_timeout(monkeypatch) -> None:
    calls = []

    def run(command, **options):
        calls.append((command, options))
        if len(calls) == 1:
            raise subprocess.TimeoutExpired(command, options["timeout"])
        return object()

    monkeypatch.setattr(MODULE.subprocess, "run", run)
    monkeypatch.setattr(MODULE.time, "sleep", lambda _: None)
    MODULE.request_remote_stop(MODULE.LEP, "/tmp/fftgyro-owned-stop")
    assert len(calls) == 2
    assert calls[-1][0][-1] == "touch /tmp/fftgyro-owned-stop"


def test_remote_completion_marker_uses_fresh_short_connection(monkeypatch) -> None:
    calls = []

    def run(command, **options):
        calls.append((command, options))
        return type("Result", (), {"returncode": 0})()

    monkeypatch.setattr(MODULE.subprocess, "run", run)
    assert MODULE.remote_file_exists(
        MODULE.AIR, "/home/fftgyro/research-data/fftgyro/run/aircraft-complete"
    )
    assert calls[0][0][-1].endswith("/aircraft-complete")
    assert calls[0][1]["timeout"] == 5


def test_remote_completion_marker_rejects_path_escape() -> None:
    with pytest.raises(ValueError):
        MODULE.remote_file_exists(MODULE.AIR, "/tmp/../unsafe")


def test_sftp_collection_retries_then_promotes_complete_batch(monkeypatch, tmp_path) -> None:
    calls = []

    def run(command, **options):
        calls.append((command, options))
        if len(calls) == 1:
            raise subprocess.TimeoutExpired(command, options["timeout"])
        staging = Path(options["cwd"])
        (staging / "rpm").mkdir()
        (staging / "rpm" / "rpm.csv").write_text("sample", encoding="utf-8")
        return object()

    monkeypatch.setattr(MODULE.subprocess, "run", run)
    monkeypatch.setattr(MODULE.time, "sleep", lambda _: None)
    MODULE.collect_sftp_artifacts(
        MODULE.AIR, "/home/fftgyro/research-data/fftgyro/test", ("rpm",), tmp_path
    )

    assert len(calls) == 2
    assert (tmp_path / "rpm" / "rpm.csv").read_text(encoding="utf-8") == "sample"
    assert all(Path(options["cwd"]) != tmp_path for _, options in calls)


def test_sftp_collection_leaves_no_partial_artifacts_after_three_failures(
        monkeypatch, tmp_path) -> None:
    calls = []

    def run(command, **options):
        calls.append((command, options))
        staging = Path(options["cwd"])
        (staging / "rpm").mkdir()
        raise subprocess.TimeoutExpired(command, options["timeout"])

    monkeypatch.setattr(MODULE.subprocess, "run", run)
    monkeypatch.setattr(MODULE.time, "sleep", lambda _: None)
    with pytest.raises(subprocess.TimeoutExpired):
        MODULE.collect_sftp_artifacts(
            MODULE.AIR, "/home/fftgyro/research-data/fftgyro/test", ("rpm",), tmp_path
        )

    assert len(calls) == 3
    assert not (tmp_path / "rpm").exists()


def test_receipt_collection_fetches_only_named_small_files(monkeypatch, tmp_path) -> None:
    calls = []

    def run(command, **options):
        calls.append((command, options))
        staging = Path(options["cwd"])
        (staging / "air-rpm-result.txt").write_text("COMPLETE\n", encoding="utf-8")
        return object()

    monkeypatch.setattr(MODULE.subprocess, "run", run)
    MODULE.collect_sftp_receipts(
        MODULE.AIR,
        "/home/fftgyro/research-data/fftgyro/test",
        {"rpm/result.txt": "air-rpm-result.txt"},
        tmp_path,
    )
    assert calls[0][1]["timeout"] == 20
    assert 'get "/home/fftgyro/research-data/fftgyro/test/rpm/result.txt"' in calls[0][1]["input"]
    assert (tmp_path / "remote-receipts/air-rpm-result.txt").read_text() == "COMPLETE\n"


def test_pitch_session_passes_missing_session_guard_without_hardware(tmp_path, monkeypatch):
    import sys

    session = tmp_path / "pitch.json"
    session.write_text('{"control_mode":"AXIS_PID","pid":{"axis":"PITCH"}}', encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["run_sync_bench.py", "--data-root", str(tmp_path),
        "--run", "pitch-argument-check", "--session-config", str(session),
        "--session-executor", "/test/executor", "--axis-ready", "--props-installed",
        "--operator-stop-ready", "--charged-4s", "--prepared-output"])
    monkeypatch.setattr(MODULE, "check_capture", lambda *args, **kwargs: 30.0)

    def reached_output_preflight(*args, **kwargs):
        raise RuntimeError("reached output preflight without hardware")

    monkeypatch.setattr(MODULE, "unused_console_reservation", reached_output_preflight)
    with pytest.raises(RuntimeError, match="reached output preflight without hardware"):
        MODULE._main()


def test_invalid_retest_context_blocks_before_output_or_hardware(tmp_path, monkeypatch):
    import sys

    path = tmp_path / "context.json"
    path.write_text('{"schema":"fftgyro-retest-context-v1"}', encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["run_sync_bench.py", "--data-root", str(tmp_path),
                        "--run", "bad-context", "--idle", "--retest-context", str(path)])

    def no_hardware(*args, **kwargs):
        raise AssertionError("must not touch hardware on invalid context")

    monkeypatch.setattr(MODULE.subprocess, "run", no_hardware)
    monkeypatch.setattr(MODULE.subprocess, "Popen", no_hardware)
    with pytest.raises(SystemExit) as result:
        MODULE._main()
    assert result.value.code == 2
    assert not (tmp_path / "bad-context").exists()
