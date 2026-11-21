from __future__ import annotations

import csv
import hashlib
import json
import struct
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fftgyro_quad_id.experiment_runner.integrated_rig import (
    IntegratedRigError,
    IntegratedRigRunner,
    SshConfig,
    SshTransport,
    _PX4_PARAM_DECODER_SCRIPT,
    _materialize_run_start_pitch_reference,
    _matches_physical_safety_confirmation,
    _probe_acquisition,
    _reboot_pixhawk,
    _validate_acquisition_csv,
    _validate_capture_ready,
    _validate_capture_result,
    _validate_executor_csv,
    _validate_live_interlock,
    choose_ssh_host,
    inspect_executor_config,
    inspect_safe_executor_config,
    load_integrated_rig_config,
    run_preflight,
    select_experiment,
)


def _executor_document(*, safe: bool = False) -> dict[str, object]:
    return {
        "run_name": "safe_locked" if safe else "test_run",
        "metadata": {"fixture_mode": "fixed"},
        "uart": {
            "device": "/dev/serial0",
            "baud": 921600,
            "target_system": 1,
            "target_component": 1,
        },
        "sample_rate_hz": 100,
        "timeout_ms": 150,
        "collective": 0.0 if safe else 0.1,
        "motor_min": 0.0,
        "motor_max": 0.4,
        "axis_mix": [0.0, 0.0, 0.0, 0.0],
        "motor_mapping_verified": not safe,
        "collective_verified": not safe,
        "startup_ramp_rate": 0.2,
        "normal_ramp_rate": 0.1,
        "abort_ramp_rate": 0.5,
        "log_directory": "/var/lib/fftgyro-executor",
        "gpio": {"enabled": False},
        "waveform": {
            "type": "HOLD4",
            "amplitude": 0.0,
            "preroll_s": 1.0,
            "active_s": 2.0,
            "posthold_s": 1.0,
            "edge_s": 0.0,
            "fade_s": 0.0,
        },
    }


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _configuration(root: Path) -> Path:
    capture = root / "capture.js"
    capture.write_text("// fixture", encoding="utf-8")
    processing = root / "processing.js"
    processing.write_text("// processing fixture", encoding="utf-8")
    capture_hash = hashlib.sha256(capture.read_bytes()).hexdigest()
    processing_hash = hashlib.sha256(processing.read_bytes()).hexdigest()
    safe = root / "safe.json"
    powered = root / "powered.json"
    _write_json(safe, _executor_document(safe=True))
    _write_json(powered, _executor_document())
    config = root / "rig.toml"
    config.write_text(
        f"""
[session]
output_root = "{(root / "data").as_posix()}"
minimum_free_mb = 1
operator = "operator"
hardware_config_id = "hardware"
fixture_id = "fixture"
protocol_version = "1.0"

[ssh]
host_candidates = ["user@10.0.0.2", "user@pi.local"]
connect_timeout_s = 1
accept_new_host_keys = true

[pi]
expected_hostname = "pi"
expected_user = "user"
executor_binary = "/usr/local/bin/fftgyro-executor"
executor_sha256 = "{'a' * 64}"
safe_executor_config = "safe.json"
executor_service = "fftgyro-executor.service"
executor_config = "/etc/fftgyro-executor/run.json"
log_directory = "/var/lib/fftgyro-executor"
uart_device = "/dev/serial0"
uart_realpath = "/dev/ttyAMA0"
uart_baud = 921600
mavlink_python = "/venv/bin/python"
minimum_free_mb = 1
minimum_uptime_s = 1

[acquisition]
node_executable = "{Path(sys.executable).as_posix()}"
electron_as_node = false
script = "capture.js"
script_sha256 = "{capture_hash}"
processing_module = "processing.js"
processing_sha256 = "{processing_hash}"
encoder_port = "auto"
encoder_baud = 576000
g_version = "G 3.5"
encoder_offset_x_deg = 0
encoder_offset_y_deg = 0
encoder_offset_z_deg = 0
force_port = "auto"
expected_force_product = "ASM0551012R0A00"
expected_force_serial = "2506002"
expected_force_frequency_hz = 1200
expected_force_limit_n = 100
expected_moment_limit_nm = 2
clock_mode = "device_sync"
minimum_encoder_rate_hz = 100
minimum_force_rate_hz = 1000
maximum_sync_delta_us = 100000
pre_roll_s = 1
post_roll_s = 1
ready_timeout_s = 2

[safety]
minimum_voltage_v = 14.5
maximum_voltage_v = 17
maximum_motor_command = 0.4
require_disarmed = true
require_rc_signal = true
rc_kill_channel = 8
rc_kill_on_min_raw = 1750
rc_kill_on_max_raw = 2200
confirmation_phrase = "SAFE"
run_confirmation_phrase = "RUN"

[px4]
expected_system_id = 1
expected_component_id = 1
expected_autopilot = 12
expected_vehicle_type = 2
expected_flight_git_prefix = "d6f12ad1c4"
heartbeat_stability_s = 2

[px4.parameters]
COM_MOT_TEST_EN = 1
RC_MAP_KILL_SW = 8

[flight_log]
enabled = true
retrieve_every_runs = 5
retrieve_at_batch_end = true
retrieval_mode = "deferred"

[[experiments]]
number = 1
name = "test_run"
purpose = "exercise the integrated test fixture"
operator_check = "verify the fixture is safe"
enabled = true
executor_config = "powered.json"
""",
        encoding="utf-8",
    )
    return config


def _external_plan_configuration(root: Path, experiments: str) -> Path:
    config = _configuration(root)
    document = config.read_text(encoding="utf-8")
    root_document, _, _inline_experiments = document.partition("[[experiments]]")
    config.write_text(
        'experiments_file = "plan.toml"\n' + root_document,
        encoding="utf-8",
    )
    (root / "plan.toml").write_text(experiments, encoding="utf-8")
    return config


class IntegratedRigConfigTests(unittest.TestCase):
    def test_run_start_pitch_reference_materializes_stable_uart_mean(self) -> None:
        document = _executor_document()
        document["metadata"] = {
            "angle_guard_reference": "RUN_START_PX4_PITCH",
        }
        document["angle_guard"] = {
            "enabled": True,
            "attitude_pitch_sign": 1.0,
            "attitude_center_deg": 0.0,
        }
        reference = _materialize_run_start_pitch_reference(
            document,
            {
                "pitch_mean_deg": 3.25,
                "pitch_min_deg": 3.20,
                "pitch_max_deg": 3.30,
                "pitch_rate_abs_max_dps": 0.8,
            },
            path=Path("dynamic.json"),
        )
        self.assertIsNotNone(reference)
        self.assertEqual(
            document["angle_guard"]["attitude_center_deg"],
            3.25,
        )
        self.assertAlmostEqual(reference["pitch_span_deg"], 0.1)

    def test_run_start_pitch_reference_rejects_moving_fixture(self) -> None:
        document = _executor_document()
        document["metadata"] = {
            "angle_guard_reference": "RUN_START_PX4_PITCH",
        }
        document["angle_guard"] = {
            "enabled": True,
            "attitude_pitch_sign": 1.0,
            "attitude_center_deg": 0.0,
        }
        with self.assertRaisesRegex(
            IntegratedRigError,
            "not stationary enough",
        ):
            _materialize_run_start_pitch_reference(
                document,
                {
                    "pitch_mean_deg": 3.0,
                    "pitch_min_deg": 0.0,
                    "pitch_max_deg": 4.0,
                    "pitch_rate_abs_max_dps": 11.0,
                },
                path=Path("dynamic.json"),
            )

    def test_physical_confirmation_accepts_full_width_form(self) -> None:
        self.assertTrue(_matches_physical_safety_confirmation("k", "k"))
        self.assertTrue(_matches_physical_safety_confirmation("ｋ", "k"))
        self.assertFalse(_matches_physical_safety_confirmation("RUN", "k"))

    def test_external_plan_loads_rich_numbered_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = _external_plan_configuration(
                root,
                """
[[experiments]]
number = 1
name = "fixed_pitch_train"
purpose = "identify the fixed-pitch wind response"
operator_check = "verify the -15 degree fixture"
enabled = true
executor_config = "powered.json"
day = 2
block = "D2_W5_FIXED"
priority = "essential"
dataset_role = "training"
apparatus = "aircraft_rotor_running"
fixture_mode = "fixed_pitch"
pitch_deg = -15
wind_speed_mps = 5
voltage_min_v = 15.0
voltage_max_v = 16.5
repeat = 2
depends_on = []
blocked_by = []
""",
            )
            config = load_integrated_rig_config(config_path)
            spec = config.experiments[0]
            self.assertEqual(config.experiment_source, root / "plan.toml")
            self.assertEqual(spec.executor_config, root / "powered.json")
            self.assertEqual(spec.day, 2)
            self.assertEqual(spec.block, "D2_W5_FIXED")
            self.assertEqual(spec.dataset_role, "training")
            self.assertEqual(spec.apparatus, "aircraft_rotor_running")
            self.assertEqual(spec.fixture_mode, "fixed_pitch")
            self.assertEqual(spec.pitch_deg, -15.0)
            self.assertEqual(spec.wind_speed_mps, 5.0)
            self.assertEqual(spec.repeat, 2)

            catalog = IntegratedRigRunner(config).experiment_catalog()
            self.assertEqual(catalog[0]["mode"], "executor")
            self.assertTrue(catalog[0]["executable_config_valid"])
            self.assertEqual(catalog[0]["voltage_min_v"], 15.0)
            self.assertEqual(catalog[0]["pitch_deg"], -15.0)

    def test_external_plan_is_mutually_exclusive_with_inline_experiments(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = _configuration(root)
            (root / "plan.toml").write_text(
                config.read_text(encoding="utf-8").partition("[[experiments]]")[2],
                encoding="utf-8",
            )
            config.write_text(
                'experiments_file = "plan.toml"\n'
                + config.read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "do not mix"):
                load_integrated_rig_config(config)

    def test_disabled_missing_executor_is_visible_but_nonblocking(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = load_integrated_rig_config(
                _external_plan_configuration(
                    root,
                    """
[[experiments]]
number = 1
name = "yaw_future"
purpose = "reserve a yaw-coupling condition"
operator_check = "do not run until yaw mapping is verified"
enabled = false
day = 2
block = "D2_YAW_OPTIONAL"
priority = "optional"
dataset_role = "pilot"
apparatus = "aircraft_rotor_running"
fixture_mode = "pitch_yaw_free"
wind_speed_mps = 10
blocked_by = ["yaw motor mix is not verified"]
""",
                )
            )
            catalog = IntegratedRigRunner(config).experiment_catalog()
            self.assertEqual(catalog[0]["mode"], "executor")
            self.assertFalse(catalog[0]["executable_config_valid"])
            self.assertIn("not configured", catalog[0]["validation_error"])

            def unreachable_run(args, **_kwargs):
                return subprocess.CompletedProcess(args, 255, "", "unreachable")

            report = run_preflight(
                config,
                run_command=unreachable_run,
                experiment_number=1,
            )
            executor_check = next(
                check
                for check in report.checks
                if check.name == "executor_config:yaw_future"
            )
            self.assertFalse(executor_check.passed)
            self.assertFalse(executor_check.blocking)

    def test_enabled_invalid_executor_still_raises_from_catalog(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "invalid.json").write_text("{", encoding="utf-8")
            config = load_integrated_rig_config(
                _external_plan_configuration(
                    root,
                    """
[[experiments]]
number = 1
name = "invalid_enabled"
purpose = "prove enabled experiments fail closed"
operator_check = "none"
enabled = true
executor_config = "invalid.json"
""",
                )
            )
            with self.assertRaises(json.JSONDecodeError):
                IntegratedRigRunner(config).experiment_catalog()

    def test_plan_condition_constraints_fail_closed(self) -> None:
        cases = {
            "fixed pitch": (
                'fixture_mode = "fixed_pitch"\n',
                "pitch_deg is required",
            ),
            "free pitch value": (
                'fixture_mode = "pitch_free"\npitch_deg = 15\n',
                "pitch_deg must be absent",
            ),
            "blocked enabled": (
                'blocked_by = ["not commissioned"]\n',
                "requires enabled=false",
            ),
            "future dependency": (
                "depends_on = [1]\n",
                "only earlier",
            ),
        }
        for name, (extra, message) in cases.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                config_path = _external_plan_configuration(
                    root,
                    f"""
[[experiments]]
number = 1
name = "condition"
purpose = "constraint test"
operator_check = "constraint test"
enabled = true
executor_config = "powered.json"
{extra}
""",
                )
                with self.assertRaisesRegex(ValueError, message):
                    load_integrated_rig_config(config_path)

    def test_load_resolves_paths_and_validates_configs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = load_integrated_rig_config(_configuration(root))
            self.assertEqual(config.acquisition.g_version, "G 3.5")
            self.assertEqual(config.pi.safe_executor_config, root / "safe.json")
            self.assertEqual(config.experiments[0].number, 1)
            self.assertEqual(
                config.experiments[0].executor_config,
                root / "powered.json",
            )
            self.assertEqual(inspect_safe_executor_config(root / "safe.json"), "safe_locked")
            summary = inspect_executor_config(
                root / "powered.json",
                expected_uart_device="/dev/serial0",
                expected_uart_baud=921600,
                expected_target_system=1,
                expected_target_component=1,
                maximum_motor_command=0.4,
            )
            self.assertEqual(summary.run_name, "test_run")
            self.assertAlmostEqual(summary.expected_duration_s, 5.5)

    def test_experiment_number_follows_configuration_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = _configuration(root)
            original = config_path.read_text(encoding="utf-8")
            config_path.write_text(
                original.replace("number = 1", "number = 2"),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ValueError,
                r"experiments\[1\]\.number must be 1",
            ):
                load_integrated_rig_config(config_path)

    def test_select_experiment_rejects_unknown_and_disabled_numbers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = _configuration(root)
            config = load_integrated_rig_config(config_path)
            self.assertEqual(select_experiment(config, 1).name, "test_run")
            with self.assertRaisesRegex(
                IntegratedRigError,
                "unknown experiment number 2",
            ):
                select_experiment(config, 2)

            config_path.write_text(
                config_path.read_text(encoding="utf-8").replace(
                    "enabled = true",
                    "enabled = false",
                ),
                encoding="utf-8",
            )
            disabled = load_integrated_rig_config(config_path)
            with self.assertRaisesRegex(IntegratedRigError, "is disabled"):
                select_experiment(disabled, 1)
            self.assertEqual(
                select_experiment(disabled, 1, require_enabled=False).number,
                1,
            )

    def test_catalog_reports_number_and_derived_command_limits(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = load_integrated_rig_config(_configuration(Path(temporary)))
            catalog = IntegratedRigRunner(config).experiment_catalog()
            self.assertEqual(len(catalog), 1)
            self.assertEqual(catalog[0]["number"], 1)
            self.assertEqual(catalog[0]["purpose"], "exercise the integrated test fixture")
            self.assertAlmostEqual(catalog[0]["peak_motor_command"], 0.1)

    def test_safe_config_must_be_fail_locked(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "unsafe.json"
            _write_json(path, _executor_document())
            with self.assertRaisesRegex(ValueError, "motor_mapping_verified=false"):
                inspect_safe_executor_config(path)

    def test_executor_peak_over_integrated_limit_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "powered.json"
            document = _executor_document()
            document["motor_max"] = 0.8
            document["collective"] = 0.5
            _write_json(path, document)
            with self.assertRaisesRegex(ValueError, "safety limit"):
                inspect_executor_config(
                    path,
                    expected_uart_device="/dev/serial0",
                    expected_uart_baud=921600,
                    expected_target_system=1,
                    expected_target_component=1,
                    maximum_motor_command=0.4,
                )

    def test_executor_peak_includes_axis_bias(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "powered.json"
            document = _executor_document()
            document["collective"] = 0.30
            document["motor_max"] = 0.40
            document["axis_mix"] = [1.0, -1.0, 1.0, -1.0]
            document["axis_bias"] = 0.02
            document["waveform"]["amplitude"] = 0.01
            _write_json(path, document)
            summary = inspect_executor_config(
                path,
                expected_uart_device="/dev/serial0",
                expected_uart_baud=921600,
                expected_target_system=1,
                expected_target_component=1,
                maximum_motor_command=0.40,
            )
            self.assertAlmostEqual(summary.peak_motor_command, 0.33)

    def test_string_boolean_and_nonfinite_values_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = _configuration(root)
            original = config_path.read_text(encoding="utf-8")
            config_path.write_text(
                original.replace(
                    "require_disarmed = true",
                    'require_disarmed = "false"',
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "must be true or false"):
                load_integrated_rig_config(config_path)

            config_path.write_text(
                original.replace(
                    "maximum_motor_command = 0.4",
                    'maximum_motor_command = "nan"',
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "must be between"):
                load_integrated_rig_config(config_path)


class IntegratedRigTransportTests(unittest.TestCase):
    def test_ssh_falls_back_without_relaxing_host_key_check(self) -> None:
        calls: list[list[str]] = []

        def fake_run(args, **_kwargs):
            calls.append(list(args))
            if args[-2] == "user@10.0.0.2":
                return subprocess.CompletedProcess(args, 255, "", "timeout")
            return subprocess.CompletedProcess(args, 0, "FFTGYRO_SSH_OK", "")

        host, errors = choose_ssh_host(
            SshConfig(
                host_candidates=("user@10.0.0.2", "user@pi.local"),
                connect_timeout_s=1,
                identity_file=None,
                accept_new_host_keys=True,
            ),
            run_command=fake_run,
        )
        self.assertEqual(host, "user@pi.local")
        self.assertEqual(len(errors), 1)
        self.assertTrue(
            all("StrictHostKeyChecking=accept-new" in call for call in calls)
        )
        self.assertTrue(all("ConnectionAttempts=1" in call for call in calls))

    def test_acquisition_probe_uses_result_protocol_and_both_sources(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = load_integrated_rig_config(_configuration(Path(temporary)))
            payload = {
                "status": "ok",
                "mode": "probe",
                "clockSync": {"confirmed": True, "deltaUs": 20},
                "timestampBasis": {"mode": "device_sync"},
                "encoder": {"port": "COM17", "rate": {"count": 100}},
                "force": {"port": "COM9", "rate": {"count": 1200}},
            }

            def fake_run(args, **_kwargs):
                return subprocess.CompletedProcess(
                    args,
                    0,
                    "READY {}\nRESULT " + json.dumps(payload) + "\n",
                    "",
                )

            checks = _probe_acquisition(config, run_command=fake_run)
            self.assertTrue(checks)
            self.assertTrue(all(check.passed for check in checks))

    def test_ssh_command_timeout_becomes_a_rig_error(self) -> None:
        def fake_run(args, **_kwargs):
            raise subprocess.TimeoutExpired(args, 1)

        transport = SshTransport(
            SshConfig(
                host_candidates=("user@pi",),
                connect_timeout_s=1,
                identity_file=None,
                accept_new_host_keys=True,
            ),
            "user@pi",
            run_command=fake_run,
        )
        with self.assertRaisesRegex(IntegratedRigError, "SSH transport failed"):
            transport.run("true")

    def test_ssh_monitor_retry_recovers_from_one_timeout(self) -> None:
        calls = 0

        def fake_run(args, **_kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise subprocess.TimeoutExpired(args, 1)
            return subprocess.CompletedProcess(args, 0, "active\n", "")

        transport = SshTransport(
            SshConfig(
                host_candidates=("user@pi",),
                connect_timeout_s=1,
                identity_file=None,
                accept_new_host_keys=True,
            ),
            "user@pi",
            run_command=fake_run,
        )
        with patch(
            "fftgyro_quad_id.experiment_runner.integrated_rig.time.sleep",
            return_value=None,
        ):
            result = transport.run(
                "systemctl is-active service || true",
                check=False,
                transport_retries=3,
            )
        self.assertEqual(result.stdout.strip(), "active")
        self.assertEqual(calls, 2)

    def test_ssh_monitor_retry_recovers_from_connection_exit(self) -> None:
        calls = 0

        def fake_run(args, **_kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                return subprocess.CompletedProcess(args, 255, "", "reset")
            return subprocess.CompletedProcess(args, 0, "inactive\n", "")

        transport = SshTransport(
            SshConfig(
                host_candidates=("user@pi",),
                connect_timeout_s=1,
                identity_file=None,
                accept_new_host_keys=True,
            ),
            "user@pi",
            run_command=fake_run,
        )
        with patch(
            "fftgyro_quad_id.experiment_runner.integrated_rig.time.sleep",
            return_value=None,
        ):
            result = transport.run(
                "systemctl is-active service || true",
                check=False,
                transport_retries=3,
            )
        self.assertEqual(result.stdout.strip(), "inactive")
        self.assertEqual(calls, 2)


class IntegratedRigLiveInterlockTests(unittest.TestCase):
    @staticmethod
    def _telemetry(config) -> dict[str, object]:
        channels = [1500] * 18
        channels[7] = 1900
        return {
            "heartbeat": True,
            "heartbeat_count": 5,
            "identity_stable": True,
            "system": 1,
            "component": 1,
            "autopilot": 12,
            "vehicle_type": 2,
            "armed": False,
            "flight_sw_version": 0,
            "flight_custom_version": "000000c4d12af1d6",
            "flight_git_prefix": "d6f12ad1c4",
            "voltage_source": "SYS_STATUS",
            "voltage_v": 16.2,
            "voltage_min_v": 16.1,
            "voltage_max_v": 16.3,
            "voltage_samples": 4,
            "current_a": 0.5,
            "rc_received": True,
            "rc_channels": channels,
            "boot_time_samples": 5,
            "boot_time_monotonic": True,
            "parameters": dict(config.px4.required_parameters),
        }

    def test_live_interlock_accepts_verified_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = load_integrated_rig_config(_configuration(Path(temporary)))
            result = _validate_live_interlock(config, self._telemetry(config))
            self.assertEqual(result["rc_kill_state"], "ON")
            self.assertEqual(result["rc_kill_raw"], 1900)
            self.assertEqual(result["voltage_source"], "SYS_STATUS")

    def test_live_interlock_enforces_selected_experiment_voltage_range(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = load_integrated_rig_config(_configuration(Path(temporary)))
            spec = replace(
                config.experiments[0],
                voltage_min_v=16.15,
                voltage_max_v=16.25,
            )
            with self.assertRaisesRegex(
                IntegratedRigError,
                r"allowed=16\.15\.\.16\.25",
            ):
                _validate_live_interlock(config, self._telemetry(config), spec)

            telemetry = self._telemetry(config)
            telemetry["voltage_min_v"] = 16.16
            telemetry["voltage_max_v"] = 16.24
            result = _validate_live_interlock(config, telemetry, spec)
            self.assertEqual(result["allowed_voltage_min_v"], 16.15)
            self.assertEqual(result["allowed_voltage_max_v"], 16.25)

    def test_post_run_interlock_skips_kill_but_still_requires_disarmed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = load_integrated_rig_config(_configuration(Path(temporary)))
            telemetry = self._telemetry(config)
            telemetry["rc_channels"][7] = 1000
            with self.assertRaisesRegex(
                IntegratedRigError,
                "RC Kill is not confirmed ON",
            ):
                _validate_live_interlock(config, telemetry)
            result = _validate_live_interlock(
                config,
                telemetry,
                require_rc_kill_on=False,
                check_rc_kill=False,
            )
            self.assertEqual(result["rc_kill_state"], "NOT_CHECKED")
            telemetry["armed"] = True
            with self.assertRaisesRegex(IntegratedRigError, "became armed"):
                _validate_live_interlock(
                    config,
                    telemetry,
                    require_rc_kill_on=False,
                    check_rc_kill=False,
                )

    def test_live_interlock_rejects_each_safety_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = load_integrated_rig_config(_configuration(Path(temporary)))
            cases: dict[str, tuple[str, object]] = {
                "kill": ("rc_channels", [1500] * 18),
                "voltage": ("voltage_min_v", 14.0),
                "firmware": ("flight_git_prefix", "0011223344"),
                "armed": ("armed", True),
                "clock": ("boot_time_monotonic", False),
            }
            for name, (field, value) in cases.items():
                with self.subTest(name=name):
                    telemetry = self._telemetry(config)
                    telemetry[field] = value
                    with self.assertRaises(IntegratedRigError):
                        _validate_live_interlock(config, telemetry)

            telemetry = self._telemetry(config)
            telemetry["parameters"] = {
                **dict(config.px4.required_parameters),
                "RC_MAP_KILL_SW": 10.0,
            }
            with self.assertRaisesRegex(IntegratedRigError, "parameter drift"):
                _validate_live_interlock(config, telemetry)

    def test_capture_ready_rejects_port_collision_and_bad_sync(self) -> None:
        ready = {
            "runId": "run-1",
            "encoder": {"port": "COM17"},
            "force": {"port": "COM9"},
            "clockSync": {"confirmed": True, "deltaUs": 50},
            "timestampBasis": {"mode": "device_sync"},
        }
        _validate_capture_ready(ready, "run-1", maximum_sync_delta_us=100)
        with self.assertRaisesRegex(IntegratedRigError, "same port"):
            _validate_capture_ready(
                {**ready, "force": {"port": "com17"}},
                "run-1",
                maximum_sync_delta_us=100,
            )
        with self.assertRaisesRegex(IntegratedRigError, "clock sync"):
            _validate_capture_ready(
                {
                    **ready,
                    "clockSync": {
                        "confirmed": True,
                        "deltaUs": float("nan"),
                    },
                },
                "run-1",
                maximum_sync_delta_us=100,
            )

    def test_capture_ready_accepts_declared_pc_receive_clock(self) -> None:
        ready = {
            "runId": "run-1",
            "encoder": {"port": "COM17"},
            "force": {"port": "COM9"},
            "clockSync": {"requested": False, "confirmed": False},
            "timestampBasis": {
                "mode": "pc_receive",
                "clock": "windows_monotonic_anchored_epoch_us",
                "hardwareSynchronized": False,
            },
        }
        _validate_capture_ready(
            ready,
            "run-1",
            maximum_sync_delta_us=100,
            clock_mode="pc_receive",
        )
        with self.assertRaisesRegex(IntegratedRigError, "receive-clock"):
            _validate_capture_ready(
                {
                    **ready,
                    "timestampBasis": {
                        **ready["timestampBasis"],
                        "hardwareSynchronized": True,
                    },
                },
                "run-1",
                maximum_sync_delta_us=100,
                clock_mode="pc_receive",
            )

    def test_safe_restore_requires_watchdog_cancellation(self) -> None:
        class FakeTransport:
            def __init__(self) -> None:
                self.commands: list[str] = []

            def run(self, command: str, **_kwargs):
                self.commands.append(command)
                return None

        with tempfile.TemporaryDirectory() as temporary:
            config = load_integrated_rig_config(_configuration(Path(temporary)))
            runner = IntegratedRigRunner(config)
            transport = FakeTransport()
            runner._transport = transport  # type: ignore[assignment]
            runner._restore_safe_executor_config(
                "/tmp/run.safe.json",
                "/tmp/run.json",
                "fftgyro-config-restore-run",
            )
            self.assertEqual(len(transport.commands), 2)
            self.assertIn("install -o root", transport.commands[0])
            self.assertIn("systemctl stop", transport.commands[1])
            self.assertIn('case \"$timer_state\"', transport.commands[1])
            self.assertIn("rm -f", transport.commands[1])
            self.assertNotIn("|| true; rm -f", transport.commands[1])

    def test_verified_pixhawk_reboot_result_is_required(self) -> None:
        class FakeTransport:
            def __init__(self, payload: dict[str, object]) -> None:
                self.payload = payload
                self.commands: list[str] = []

            def run(self, command: str, **kwargs):
                self.commands.append(command)
                self.input_text = kwargs["input_text"]
                return SimpleNamespace(
                    returncode=0,
                    stdout=json.dumps(self.payload) + "\n",
                    stderr="",
                )

        with tempfile.TemporaryDirectory() as temporary:
            config = load_integrated_rig_config(_configuration(Path(temporary)))
            verified = {
                "verified": True,
                "command_ack_result": 0,
                "old_boot_ms": 120000,
                "new_boot_ms": 4000,
                "recovered_heartbeats": 3,
            }
            transport = FakeTransport(verified)
            self.assertEqual(
                _reboot_pixhawk(config, transport),  # type: ignore[arg-type]
                verified,
            )
            self.assertIn(config.pi.uart_device, transport.commands[0])
            self.assertIn(
                "MAV_CMD_PREFLIGHT_REBOOT_SHUTDOWN",
                transport.input_text,
            )

            rejected = FakeTransport(
                {
                    "verified": False,
                    "command_ack_result": 4,
                    "reason": "rejected",
                }
            )
            with self.assertRaisesRegex(IntegratedRigError, "not verified"):
                _reboot_pixhawk(config, rejected)  # type: ignore[arg-type]

    def test_px4_bytewise_integer_parameters_are_decoded(self) -> None:
        namespace = {"struct": struct}
        exec(_PX4_PARAM_DECODER_SCRIPT, namespace)
        decode = namespace["decode_parameter_value"]
        encoded_int32 = struct.unpack("<f", struct.pack("<i", 921600))[0]
        self.assertEqual(
            decode(SimpleNamespace(param_type=6, param_value=encoded_int32)),
            921600.0,
        )
        self.assertAlmostEqual(
            decode(SimpleNamespace(param_type=9, param_value=0.75)),
            0.75,
        )

    def test_shared_pi_config_is_backed_up_before_safe_restore(self) -> None:
        class FakeBackupTransport:
            host = "user@pi"

            def __init__(self, content: bytes) -> None:
                self.content = content

            def run(self, _command: str, **_kwargs):
                digest = hashlib.sha256(self.content).hexdigest()
                return SimpleNamespace(stdout=digest + "\n")

            def download(self, _remote: str, local: Path) -> None:
                local.parent.mkdir(parents=True, exist_ok=True)
                local.write_bytes(self.content)

        with tempfile.TemporaryDirectory() as temporary:
            config = load_integrated_rig_config(_configuration(Path(temporary)))
            runner = IntegratedRigRunner(config)
            content = b'{"run_name":"shared-current-setting"}\n'
            runner._transport = FakeBackupTransport(content)  # type: ignore[assignment]
            backup_path, digest = runner._backup_installed_executor_config()
            self.assertEqual(backup_path.read_bytes(), content)
            self.assertEqual(digest, hashlib.sha256(content).hexdigest())
            manifest = json.loads(
                (backup_path.parent / "backup.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["sha256"], digest)

    def test_capture_only_run_never_requests_executor_start(self) -> None:
        class CaptureOnlyTransport:
            host = "user@pi"

            def __init__(self, telemetry: dict[str, object]) -> None:
                self.telemetry = telemetry
                self.commands: list[str] = []

            def run(self, command: str, **kwargs):
                self.commands.append(command)
                if kwargs.get("input_text") is not None:
                    return SimpleNamespace(
                        returncode=0,
                        stdout=json.dumps(self.telemetry) + "\n",
                        stderr="",
                    )
                if "python3 -c" in command:
                    return SimpleNamespace(
                        returncode=0,
                        stdout=json.dumps(
                            {
                                "realtime_ns": 1_000_000_000,
                                "monotonic_ns": 500_000_000,
                            }
                        )
                        + "\n",
                        stderr="",
                    )
                return SimpleNamespace(returncode=0, stdout="", stderr="")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = _external_plan_configuration(
                root,
                """
[[experiments]]
number = 1
name = "rotor_stopped_reference"
purpose = "measure the body and fixture baseline"
operator_check = "remove propellers and keep RC Kill ON"
enabled = true
day = 1
block = "D1_W0_REFERENCE"
priority = "essential"
dataset_role = "reference"
apparatus = "aircraft_rotor_stopped"
fixture_mode = "fixed_pitch"
pitch_deg = 0
wind_speed_mps = 0
voltage_min_v = 15.0
voltage_max_v = 16.5
capture_only_duration_s = 1.0
""",
            )
            config = load_integrated_rig_config(config_path)
            spec = config.experiments[0]
            runner = IntegratedRigRunner(
                config,
                input_function=lambda _prompt: "RUN",
            )
            transport = CaptureOnlyTransport(self._telemetry(config))
            runner._transport = transport  # type: ignore[assignment]

            def fake_start(output_path: Path, *, duration_s: float, run_id: str):
                self.assertEqual(duration_s, 1.0)
                output_path.write_text("capture fixture\n", encoding="utf-8")
                return SimpleNamespace(
                    ready={
                        "runId": run_id,
                        "encoder": {"port": "COM17"},
                        "force": {"port": "COM9"},
                        "clockSync": {"confirmed": True, "deltaUs": 10},
                        "timestampBasis": {"mode": "device_sync"},
                    },
                    run_id=run_id,
                )

            def fake_wait(capture, *, timeout_s: float):
                self.assertEqual(timeout_s, 21.0)
                return {
                    "status": "ok",
                    "runId": capture.run_id,
                    "clockSync": {"confirmed": True, "deltaUs": 10},
                    "timestampBasis": {"mode": "device_sync"},
                    "encoder": {"rate": {"count": 100}},
                    "force": {"rate": {"count": 1200}},
                }

            with (
                patch.object(runner, "_start_capture", side_effect=fake_start),
                patch.object(runner, "_wait_capture", side_effect=fake_wait),
                patch(
                    "fftgyro_quad_id.experiment_runner.integrated_rig."
                    "_validate_acquisition_csv"
                ),
            ):
                run_root = runner._run_one(spec, index=1, total=1)

            self.assertTrue(run_root.name.startswith("E001_"))
            self.assertTrue((run_root / "experiment-plan.toml").is_file())
            commands = "\n".join(transport.commands)
            self.assertNotIn("systemctl start", commands)
            self.assertNotIn("executor_start_requested", commands)
            events = (run_root / "events.jsonl").read_text(encoding="utf-8")
            self.assertNotIn('"event_type":"executor_start_requested"', events)
            self.assertIn('"mode":"capture_only"', events)


class IntegratedRigLogValidationTests(unittest.TestCase):
    def test_complete_executor_csv_requires_zero_release(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "executor.csv"
            header = [
                "sequence",
                "scheduled_ns",
                "sent_ns",
                "requested_motor1",
                "requested_motor2",
                "requested_motor3",
                "requested_motor4",
                "ack_status",
                "action",
            ]
            with path.open("w", encoding="utf-8", newline="") as stream:
                stream.write("# run_id=1,result=COMPLETE,detail=\n")
                writer = csv.DictWriter(stream, fieldnames=header)
                writer.writeheader()
                writer.writerow(
                    {
                        "sequence": 0,
                        "scheduled_ns": 1,
                        "sent_ns": 2,
                        "requested_motor1": 0.1,
                        "requested_motor2": 0.1,
                        "requested_motor3": 0.1,
                        "requested_motor4": 0.1,
                        "ack_status": 0,
                        "action": "CONTROL",
                    }
                )
                writer.writerow(
                    {
                        "sequence": 1,
                        "scheduled_ns": 3,
                        "sent_ns": 4,
                        "requested_motor1": 0,
                        "requested_motor2": 0,
                        "requested_motor3": 0,
                        "requested_motor4": 0,
                        "ack_status": 1,
                        "action": "RELEASE",
                    }
                )
            _validate_executor_csv(path)

    def test_offboard_executor_csv_requires_zero_thrust_disarmed_end(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "offboard.csv"
            header = [
                "sequence",
                "scheduled_ns",
                "sent_ns",
                "stage",
                "target_pitch_deg",
                "target_pitch_rate_dps",
                "thrust",
                "measured_pitch_deg",
                "measured_pitch_rate_dps",
                "armed",
                "offboard",
            ]
            with path.open("w", encoding="utf-8", newline="") as stream:
                stream.write("# result=COMPLETE,detail=,control_mode=ATTITUDE\n")
                writer = csv.DictWriter(stream, fieldnames=header)
                writer.writeheader()
                writer.writerow(
                    {
                        "sequence": 0,
                        "scheduled_ns": 1,
                        "sent_ns": 2,
                        "stage": "ACTIVE",
                        "target_pitch_deg": 5,
                        "target_pitch_rate_dps": 0,
                        "thrust": 0.3,
                        "measured_pitch_deg": 4.8,
                        "measured_pitch_rate_dps": 1,
                        "armed": 1,
                        "offboard": 1,
                    }
                )
                writer.writerow(
                    {
                        "sequence": 1,
                        "scheduled_ns": 3,
                        "sent_ns": 4,
                        "stage": "DISARM_WAIT",
                        "target_pitch_deg": 0,
                        "target_pitch_rate_dps": 0,
                        "thrust": 0,
                        "measured_pitch_deg": 0,
                        "measured_pitch_rate_dps": 0,
                        "armed": 0,
                        "offboard": 1,
                    }
                )
            _validate_executor_csv(path, executor_kind="offboard")

    def test_acquisition_csv_requires_real_rows_from_both_sources(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "acquisition.csv"
            headers = [
                "Origen trigger",
                "Trigger PC us",
                "ESP propio us",
                "ESP sync us",
                "Enc Roll deg",
                "Enc Pitch deg",
                "Enc Yaw deg",
                "Fuerza PC us",
                "Fuerza Fx N",
                "Fuerza Fy N",
                "Fuerza Fz N",
                "Fuerza Mx Nm",
                "Fuerza My Nm",
                "Fuerza Mz Nm",
            ]
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=headers)
                writer.writeheader()
                timestamp = 1_000_000
                for source in ("Encoder/ESP", "Sensor de Fuerza"):
                    for _ in range(10):
                        writer.writerow(
                            {
                                key: (
                                    source
                                    if key == "Origen trigger"
                                    else timestamp
                                    if key == "Trigger PC us"
                                    else 0.0
                                )
                                for key in headers
                            }
                        )
                        timestamp += 1
            _validate_acquisition_csv(path)

    def test_capture_result_rejects_bad_clock_sync(self) -> None:
        result = {
            "status": "ok",
            "runId": "run",
            "clockSync": {"confirmed": True, "deltaUs": 200_000},
            "timestampBasis": {"mode": "device_sync"},
            "encoder": {"rate": {"count": 100}},
            "force": {"rate": {"count": 1200}},
        }
        with self.assertRaisesRegex(IntegratedRigError, "delta"):
            _validate_capture_result(
                result,
                "run",
                maximum_sync_delta_us=100_000,
            )


if __name__ == "__main__":
    unittest.main()
