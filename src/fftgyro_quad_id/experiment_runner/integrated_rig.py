from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import queue
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
import tomllib
import unicodedata
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, IO, Sequence

from .session import ExperimentSession


SAFE_RUN_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
SAFE_BLOCK_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
SAFE_CONDITION_ID = re.compile(r"^[A-Z][A-Z0-9_-]{0,15}$")
SAFE_REMOTE_NAME = re.compile(r"^[A-Za-z0-9_.@:/-]+$")
RUN_PRIORITIES = frozenset({"essential", "validation", "optional"})
DATASET_ROLES = frozenset(
    {
        "calibration",
        "training",
        "model_selection",
        "final_holdout",
        "pilot",
        "reference",
    }
)
APPARATUS_TYPES = frozenset(
    {"rig_only", "aircraft_rotor_stopped", "aircraft_rotor_running"}
)
FIXTURE_MODES = frozenset({"fixed_pitch", "pitch_free", "pitch_yaw_free"})
EXECUTOR_KINDS = frozenset({"direct_motor", "offboard"})


class IntegratedRigError(RuntimeError):
    """A condition that makes a powered integrated run unsafe or incomplete."""


@dataclass(frozen=True, slots=True)
class SessionConfig:
    output_root: Path
    minimum_free_mb: int
    operator: str
    hardware_config_id: str
    fixture_id: str
    protocol_version: str


@dataclass(frozen=True, slots=True)
class SshConfig:
    host_candidates: tuple[str, ...]
    connect_timeout_s: float
    identity_file: Path | None
    accept_new_host_keys: bool


@dataclass(frozen=True, slots=True)
class PiConfig:
    expected_hostname: str
    expected_user: str
    executor_binary: str
    executor_sha256: str
    safe_executor_config: Path
    executor_service: str
    executor_config: str
    log_directory: str
    uart_device: str
    uart_realpath: str
    uart_baud: int
    mavlink_python: str
    minimum_free_mb: int
    minimum_uptime_s: float
    direct_executor_binary: str | None = None
    direct_executor_sha256: str | None = None
    offboard_executor_binary: str | None = None
    offboard_executor_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class AcquisitionConfig:
    node_executable: str
    electron_as_node: bool
    script: Path
    script_sha256: str
    processing_module: Path
    processing_sha256: str
    g_version: str
    encoder_offset_x_deg: float
    encoder_offset_y_deg: float
    encoder_offset_z_deg: float
    encoder_port: str
    encoder_baud: int
    force_port: str
    expected_force_product: str
    expected_force_serial: str
    expected_force_frequency_hz: float
    expected_force_limit_n: float
    expected_moment_limit_nm: float
    clock_mode: str
    maximum_sync_delta_us: int
    minimum_encoder_rate_hz: float
    minimum_force_rate_hz: float
    pre_roll_s: float
    post_roll_s: float
    ready_timeout_s: float


@dataclass(frozen=True, slots=True)
class SafetyConfig:
    minimum_voltage_v: float
    maximum_voltage_v: float
    maximum_motor_command: float
    require_disarmed: bool
    require_rc_signal: bool
    rc_kill_channel: int
    rc_kill_on_min_raw: int
    rc_kill_on_max_raw: int
    confirmation_phrase: str
    run_confirmation_phrase: str


@dataclass(frozen=True, slots=True)
class FlightLogConfig:
    enabled: bool
    retrieve_every_runs: int
    retrieve_at_batch_end: bool
    retrieval_mode: str
    reboot_pixhawk_after_run: bool
    reboot_timeout_s: float
    reboot_stability_s: float


@dataclass(frozen=True, slots=True)
class Px4Config:
    expected_system_id: int
    expected_component_id: int
    expected_autopilot: int
    expected_vehicle_type: int
    expected_flight_git_prefix: str
    heartbeat_stability_s: float
    required_parameters: tuple[tuple[str, float], ...]


@dataclass(frozen=True, slots=True)
class IntegratedRunSpec:
    number: int
    name: str
    purpose: str
    operator_check: str
    enabled: bool
    executor_config: Path | None
    condition_id: str = ""
    attempt: int = 1
    executor_kind: str = "direct_motor"
    capture_only_duration_s: float | None = None
    day: int = 1
    block: str = "legacy"
    priority: str = "essential"
    dataset_role: str = "pilot"
    apparatus: str = "aircraft_rotor_running"
    fixture_mode: str = "pitch_free"
    pitch_deg: float | None = None
    beta_deg: float = 0.0
    mount_instance: int = 1
    wind_speed_mps: float = 0.0
    voltage_min_v: float = 0.0
    voltage_max_v: float = math.inf
    repeat: int = 1
    depends_on: tuple[int, ...] = ()
    blocked_by: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class IntegratedRigConfig:
    path: Path
    experiment_source: Path
    session: SessionConfig
    ssh: SshConfig
    pi: PiConfig
    acquisition: AcquisitionConfig
    safety: SafetyConfig
    px4: Px4Config
    flight_log: FlightLogConfig
    experiments: tuple[IntegratedRunSpec, ...]


@dataclass(frozen=True, slots=True)
class PreflightCheck:
    name: str
    status: str
    detail: str
    blocking: bool

    @property
    def passed(self) -> bool:
        return self.status == "pass"


@dataclass(frozen=True, slots=True)
class PreflightReport:
    ready: bool
    ssh_host: str | None
    voltage_v: float | None
    current_a: float | None
    rc_kill_raw: int | None
    checks: tuple[PreflightCheck, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ExecutorConfigSummary:
    run_name: str
    expected_duration_s: float
    peak_motor_command: float
    sample_rate_hz: int
    executor_kind: str = "direct_motor"
    control_mode: str | None = None
    axis: str | None = None


@dataclass(frozen=True, slots=True)
class CommandResult:
    args: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str


RunCommand = Callable[..., subprocess.CompletedProcess[str]]


def load_integrated_rig_config(path: Path) -> IntegratedRigConfig:
    config_path = path.expanduser().resolve()
    with config_path.open("rb") as stream:
        document = tomllib.load(stream)
    base = config_path.parent

    session_doc = _table(document, "session")
    ssh_doc = _table(document, "ssh")
    pi_doc = _table(document, "pi")
    acquisition_doc = _table(document, "acquisition")
    safety_doc = _table(document, "safety")
    px4_doc = _table(document, "px4")
    safety_minimum_voltage_v = _positive_float(
        safety_doc.get("minimum_voltage_v", 14.5),
        "safety.minimum_voltage_v",
    )
    safety_maximum_voltage_v = _positive_float(
        safety_doc.get("maximum_voltage_v", 17.0),
        "safety.maximum_voltage_v",
    )
    parameters_doc = px4_doc.get("parameters")
    if not isinstance(parameters_doc, dict) or not parameters_doc:
        raise ValueError("px4.parameters must be a non-empty table")
    flight_doc = document.get("flight_log", {})
    if not isinstance(flight_doc, dict):
        raise ValueError("configuration table flight_log must be a table")
    experiments_doc = document.get("experiments")
    legacy_runs_doc = document.get("runs")
    experiments_file_text = _optional_text(document.get("experiments_file"))
    configured_sources = sum(
        source is not None
        for source in (experiments_doc, legacy_runs_doc, experiments_file_text)
    )
    if configured_sources > 1:
        raise ValueError("use [[experiments]] only; do not mix it with legacy [[runs]]")
    experiment_source = config_path
    experiments_base = base
    if experiments_file_text is not None:
        experiment_source = _resolve_path(base, experiments_file_text)
        with experiment_source.open("rb") as stream:
            experiment_document = tomllib.load(stream)
        runs_doc = experiment_document.get("experiments")
        if experiment_document.get("runs") is not None:
            raise ValueError(
                "external experiment plan must use [[experiments]], not [[runs]]"
            )
        experiments_base = experiment_source.parent
    else:
        runs_doc = experiments_doc if experiments_doc is not None else legacy_runs_doc
    if not isinstance(runs_doc, list) or not runs_doc:
        raise ValueError(
            "configuration requires at least one [[experiments]] table"
        )

    output_root = _resolve_path(
        base,
        _required_text(session_doc, "output_root"),
    )
    identity_text = _optional_text(ssh_doc.get("identity_file"))
    identity_file = _resolve_path(base, identity_text) if identity_text else None
    capture_script = _resolve_path(
        base,
        _required_text(acquisition_doc, "script"),
    )
    processing_module = _resolve_path(
        base,
        _required_text(acquisition_doc, "processing_module"),
    )

    candidates_value = ssh_doc.get("host_candidates")
    if not isinstance(candidates_value, list) or not candidates_value:
        raise ValueError("ssh.host_candidates must be a non-empty array")
    host_candidates = tuple(str(value).strip() for value in candidates_value)
    if any(not SAFE_REMOTE_NAME.fullmatch(value) for value in host_candidates):
        raise ValueError("ssh.host_candidates contains an unsafe value")

    run_specs: list[IntegratedRunSpec] = []
    names: set[str] = set()
    condition_attempts: set[tuple[str, int]] = set()
    for index, value in enumerate(runs_doc, start=1):
        if not isinstance(value, dict):
            raise ValueError(f"experiments[{index}] must be a table")
        number = _positive_int(
            value.get("number", index),
            f"experiments[{index}].number",
        )
        if number != index:
            raise ValueError(
                f"experiments[{index}].number must be {index} "
                "to match the configured order"
            )
        name = _required_text(value, "name")
        if not SAFE_RUN_NAME.fullmatch(name):
            raise ValueError(
                f"experiments[{index}].name is not a safe run name: {name}"
            )
        if name in names:
            raise ValueError(f"duplicate run name: {name}")
        names.add(name)
        condition_id = str(value.get("condition_id", f"E{number:03d}")).strip()
        if not SAFE_CONDITION_ID.fullmatch(condition_id):
            raise ValueError(
                f"experiments[{index}].condition_id is invalid: {condition_id}"
            )
        attempt = _strict_positive_int(
            value.get("attempt", 1),
            f"experiments[{index}].attempt",
        )
        condition_attempt = (condition_id, attempt)
        if condition_attempt in condition_attempts:
            raise ValueError(
                f"duplicate condition/attempt: {condition_id}/{attempt}"
            )
        condition_attempts.add(condition_attempt)
        enabled = _boolean(
            value.get("enabled", True),
            f"experiments[{index}].enabled",
        )
        apparatus = _enum_text(
            value.get("apparatus", "aircraft_rotor_running"),
            f"experiments[{index}].apparatus",
            APPARATUS_TYPES,
        )
        fixture_mode = _enum_text(
            value.get("fixture_mode", "pitch_free"),
            f"experiments[{index}].fixture_mode",
            FIXTURE_MODES,
        )
        pitch_deg = (
            _finite_float(value.get("pitch_deg"), f"experiments[{index}].pitch_deg")
            if "pitch_deg" in value
            else None
        )
        if fixture_mode == "fixed_pitch" and pitch_deg is None:
            raise ValueError(
                f"experiments[{index}].pitch_deg is required for fixed_pitch"
            )
        if fixture_mode != "fixed_pitch" and pitch_deg is not None:
            raise ValueError(
                f"experiments[{index}].pitch_deg must be absent for {fixture_mode}"
            )
        voltage_min_v = _bounded_float(
            value.get("voltage_min_v", safety_minimum_voltage_v),
            f"experiments[{index}].voltage_min_v",
            safety_minimum_voltage_v,
            safety_maximum_voltage_v,
        )
        voltage_max_v = _bounded_float(
            value.get("voltage_max_v", safety_maximum_voltage_v),
            f"experiments[{index}].voltage_max_v",
            safety_minimum_voltage_v,
            safety_maximum_voltage_v,
        )
        if voltage_min_v >= voltage_max_v:
            raise ValueError(
                f"experiments[{index}] voltage_min_v must be less than voltage_max_v"
            )
        depends_on = _positive_int_tuple(
            value.get("depends_on", []),
            f"experiments[{index}].depends_on",
        )
        if len(set(depends_on)) != len(depends_on):
            raise ValueError(f"experiments[{index}].depends_on contains duplicates")
        if any(dependency >= number for dependency in depends_on):
            raise ValueError(
                f"experiments[{index}].depends_on must contain only earlier "
                "experiment numbers"
            )
        blocked_by = _text_tuple(
            value.get("blocked_by", []),
            f"experiments[{index}].blocked_by",
        )
        if blocked_by and enabled:
            raise ValueError(
                f"experiments[{index}].blocked_by requires enabled=false"
            )
        executor_text = _optional_text(value.get("executor_config"))
        executor_kind = _enum_text(
            value.get("executor_kind", "direct_motor"),
            f"experiments[{index}].executor_kind",
            EXECUTOR_KINDS,
        )
        capture_only_duration_s = (
            _positive_float(
                value.get("capture_only_duration_s"),
                f"experiments[{index}].capture_only_duration_s",
            )
            if "capture_only_duration_s" in value
            else None
        )
        capture_only_allowed = apparatus in {
            "rig_only",
            "aircraft_rotor_stopped",
        }
        if executor_text is None and enabled and not (
            capture_only_allowed and capture_only_duration_s is not None
        ):
            raise ValueError(
                f"experiments[{index}].executor_config is required for enabled "
                "motor-running experiments"
            )
        if capture_only_duration_s is not None:
            if not capture_only_allowed:
                raise ValueError(
                    f"experiments[{index}].capture_only_duration_s is only valid "
                    "for rig_only or aircraft_rotor_stopped"
                )
            if executor_text is not None:
                raise ValueError(
                    f"experiments[{index}] must not combine executor_config with "
                    "capture_only_duration_s"
                )
        run_specs.append(
            IntegratedRunSpec(
                number=number,
                condition_id=condition_id,
                attempt=attempt,
                name=name,
                purpose=_required_text(value, "purpose"),
                operator_check=_required_text(value, "operator_check"),
                enabled=enabled,
                executor_config=(
                    _resolve_path(experiments_base, executor_text)
                    if executor_text is not None
                    else None
                ),
                executor_kind=executor_kind,
                day=_strict_positive_int(
                    value.get("day", 1),
                    f"experiments[{index}].day",
                ),
                block=_block_name(
                    value.get("block", "legacy"),
                    f"experiments[{index}].block",
                ),
                priority=_enum_text(
                    value.get("priority", "essential"),
                    f"experiments[{index}].priority",
                    RUN_PRIORITIES,
                ),
                dataset_role=_enum_text(
                    value.get("dataset_role", "pilot"),
                    f"experiments[{index}].dataset_role",
                    DATASET_ROLES,
                ),
                apparatus=apparatus,
                fixture_mode=fixture_mode,
                pitch_deg=pitch_deg,
                beta_deg=_bounded_float(
                    value.get("beta_deg", 0.0),
                    f"experiments[{index}].beta_deg",
                    -30.0,
                    30.0,
                ),
                mount_instance=_strict_positive_int(
                    value.get("mount_instance", 1),
                    f"experiments[{index}].mount_instance",
                ),
                wind_speed_mps=_bounded_float(
                    value.get("wind_speed_mps", 0.0),
                    f"experiments[{index}].wind_speed_mps",
                    0.0,
                    15.0,
                ),
                voltage_min_v=voltage_min_v,
                voltage_max_v=voltage_max_v,
                repeat=_strict_positive_int(
                    value.get("repeat", 1),
                    f"experiments[{index}].repeat",
                ),
                depends_on=depends_on,
                blocked_by=blocked_by,
                capture_only_duration_s=capture_only_duration_s,
            )
        )

    component_values: dict[str, tuple[str | None, str | None]] = {}
    for component in ("direct", "offboard"):
        binary = _optional_text(pi_doc.get(f"{component}_executor_binary"))
        digest = _optional_text(pi_doc.get(f"{component}_executor_sha256"))
        if (binary is None) != (digest is None):
            raise ValueError(
                f"pi.{component}_executor_binary and "
                f"pi.{component}_executor_sha256 must be configured together"
            )
        component_values[component] = (binary, digest)

    config = IntegratedRigConfig(
        path=config_path,
        experiment_source=experiment_source,
        session=SessionConfig(
            output_root=output_root,
            minimum_free_mb=_positive_int(
                session_doc.get("minimum_free_mb", 2048),
                "session.minimum_free_mb",
            ),
            operator=_required_text(session_doc, "operator"),
            hardware_config_id=_required_text(session_doc, "hardware_config_id"),
            fixture_id=_required_text(session_doc, "fixture_id"),
            protocol_version=str(session_doc.get("protocol_version", "1.0")).strip(),
        ),
        ssh=SshConfig(
            host_candidates=host_candidates,
            connect_timeout_s=_positive_float(
                ssh_doc.get("connect_timeout_s", 5.0),
                "ssh.connect_timeout_s",
            ),
            identity_file=identity_file,
            accept_new_host_keys=_boolean(
                ssh_doc.get("accept_new_host_keys", True),
                "ssh.accept_new_host_keys",
            ),
        ),
        pi=PiConfig(
            expected_hostname=_safe_remote_token(
                _required_text(pi_doc, "expected_hostname"),
                "pi.expected_hostname",
            ),
            expected_user=_safe_remote_token(
                _required_text(pi_doc, "expected_user"),
                "pi.expected_user",
            ),
            executor_binary=_safe_remote_path(
                _required_text(pi_doc, "executor_binary"),
                "pi.executor_binary",
            ),
            executor_sha256=_sha256_text(
                _required_text(pi_doc, "executor_sha256"),
                "pi.executor_sha256",
            ),
            safe_executor_config=_resolve_path(
                base,
                _required_text(pi_doc, "safe_executor_config"),
            ),
            executor_service=_safe_remote_token(
                _required_text(pi_doc, "executor_service"),
                "pi.executor_service",
            ),
            executor_config=_safe_remote_path(
                _required_text(pi_doc, "executor_config"),
                "pi.executor_config",
            ),
            log_directory=_safe_remote_path(
                _required_text(pi_doc, "log_directory"),
                "pi.log_directory",
            ),
            uart_device=_safe_remote_path(
                _required_text(pi_doc, "uart_device"),
                "pi.uart_device",
            ),
            uart_realpath=_safe_remote_path(
                _required_text(pi_doc, "uart_realpath"),
                "pi.uart_realpath",
            ),
            uart_baud=_positive_int(pi_doc.get("uart_baud", 921600), "pi.uart_baud"),
            mavlink_python=_safe_remote_path(
                _required_text(pi_doc, "mavlink_python"),
                "pi.mavlink_python",
            ),
            minimum_free_mb=_positive_int(
                pi_doc.get("minimum_free_mb", 256),
                "pi.minimum_free_mb",
            ),
            minimum_uptime_s=_positive_float(
                pi_doc.get("minimum_uptime_s", 60.0),
                "pi.minimum_uptime_s",
            ),
            direct_executor_binary=(
                _safe_remote_path(component_values["direct"][0], "pi.direct_executor_binary")
                if component_values["direct"][0] is not None else None
            ),
            direct_executor_sha256=(
                _sha256_text(component_values["direct"][1], "pi.direct_executor_sha256")
                if component_values["direct"][1] is not None else None
            ),
            offboard_executor_binary=(
                _safe_remote_path(
                    component_values["offboard"][0],
                    "pi.offboard_executor_binary",
                )
                if component_values["offboard"][0] is not None else None
            ),
            offboard_executor_sha256=(
                _sha256_text(
                    component_values["offboard"][1],
                    "pi.offboard_executor_sha256",
                )
                if component_values["offboard"][1] is not None else None
            ),
        ),
        acquisition=AcquisitionConfig(
            node_executable=_resolve_executable(
                base,
                str(acquisition_doc.get("node_executable", "node")).strip(),
            ),
            electron_as_node=_boolean(
                acquisition_doc.get("electron_as_node", False),
                "acquisition.electron_as_node",
            ),
            script=capture_script,
            script_sha256=_sha256_text(
                _required_text(acquisition_doc, "script_sha256"),
                "acquisition.script_sha256",
            ),
            processing_module=processing_module,
            processing_sha256=_sha256_text(
                _required_text(acquisition_doc, "processing_sha256"),
                "acquisition.processing_sha256",
            ),
            g_version=_g_version(
                _required_text(acquisition_doc, "g_version")
            ),
            encoder_offset_x_deg=_finite_float(
                acquisition_doc.get("encoder_offset_x_deg", 0.0),
                "acquisition.encoder_offset_x_deg",
            ),
            encoder_offset_y_deg=_finite_float(
                acquisition_doc.get("encoder_offset_y_deg", 0.0),
                "acquisition.encoder_offset_y_deg",
            ),
            encoder_offset_z_deg=_finite_float(
                acquisition_doc.get("encoder_offset_z_deg", 0.0),
                "acquisition.encoder_offset_z_deg",
            ),
            encoder_port=str(acquisition_doc.get("encoder_port", "auto")).strip(),
            encoder_baud=_positive_int(
                acquisition_doc.get("encoder_baud", 576000),
                "acquisition.encoder_baud",
            ),
            force_port=str(acquisition_doc.get("force_port", "auto")).strip(),
            expected_force_product=_required_text(
                acquisition_doc,
                "expected_force_product",
            ),
            expected_force_serial=_required_text(
                acquisition_doc,
                "expected_force_serial",
            ),
            expected_force_frequency_hz=_positive_float(
                acquisition_doc.get("expected_force_frequency_hz"),
                "acquisition.expected_force_frequency_hz",
            ),
            expected_force_limit_n=_positive_float(
                acquisition_doc.get("expected_force_limit_n"),
                "acquisition.expected_force_limit_n",
            ),
            expected_moment_limit_nm=_positive_float(
                acquisition_doc.get("expected_moment_limit_nm"),
                "acquisition.expected_moment_limit_nm",
            ),
            clock_mode=_acquisition_clock_mode(
                str(acquisition_doc.get("clock_mode", "device_sync")).strip()
            ),
            maximum_sync_delta_us=_positive_int(
                acquisition_doc.get("maximum_sync_delta_us", 100_000),
                "acquisition.maximum_sync_delta_us",
            ),
            minimum_encoder_rate_hz=_positive_float(
                acquisition_doc.get("minimum_encoder_rate_hz", 100.0),
                "acquisition.minimum_encoder_rate_hz",
            ),
            minimum_force_rate_hz=_positive_float(
                acquisition_doc.get("minimum_force_rate_hz", 1000.0),
                "acquisition.minimum_force_rate_hz",
            ),
            pre_roll_s=_nonnegative_float(
                acquisition_doc.get("pre_roll_s", 2.0),
                "acquisition.pre_roll_s",
            ),
            post_roll_s=_nonnegative_float(
                acquisition_doc.get("post_roll_s", 2.0),
                "acquisition.post_roll_s",
            ),
            ready_timeout_s=_positive_float(
                acquisition_doc.get("ready_timeout_s", 15.0),
                "acquisition.ready_timeout_s",
            ),
        ),
        safety=SafetyConfig(
            minimum_voltage_v=safety_minimum_voltage_v,
            maximum_voltage_v=safety_maximum_voltage_v,
            maximum_motor_command=_bounded_float(
                safety_doc.get("maximum_motor_command", 0.8),
                "safety.maximum_motor_command",
                0.0,
                1.0,
            ),
            require_disarmed=_boolean(
                safety_doc.get("require_disarmed", True),
                "safety.require_disarmed",
            ),
            require_rc_signal=_boolean(
                safety_doc.get("require_rc_signal", True),
                "safety.require_rc_signal",
            ),
            rc_kill_channel=_positive_int(
                safety_doc.get("rc_kill_channel", 8),
                "safety.rc_kill_channel",
            ),
            rc_kill_on_min_raw=_positive_int(
                safety_doc.get("rc_kill_on_min_raw"),
                "safety.rc_kill_on_min_raw",
            ),
            rc_kill_on_max_raw=_positive_int(
                safety_doc.get("rc_kill_on_max_raw"),
                "safety.rc_kill_on_max_raw",
            ),
            confirmation_phrase=_required_text(safety_doc, "confirmation_phrase"),
            run_confirmation_phrase=_required_text(
                safety_doc,
                "run_confirmation_phrase",
            ),
        ),
        px4=Px4Config(
            expected_system_id=_positive_int(
                px4_doc.get("expected_system_id", 1),
                "px4.expected_system_id",
            ),
            expected_component_id=_nonnegative_int(
                px4_doc.get("expected_component_id", 1),
                "px4.expected_component_id",
            ),
            expected_autopilot=_nonnegative_int(
                px4_doc.get("expected_autopilot", 12),
                "px4.expected_autopilot",
            ),
            expected_vehicle_type=_nonnegative_int(
                px4_doc.get("expected_vehicle_type", 2),
                "px4.expected_vehicle_type",
            ),
            expected_flight_git_prefix=_hex_text(
                _required_text(
                    px4_doc,
                    "expected_flight_git_prefix",
                ),
                "px4.expected_flight_git_prefix",
                minimum_length=8,
                maximum_length=10,
            ),
            heartbeat_stability_s=_positive_float(
                px4_doc.get("heartbeat_stability_s", 5.0),
                "px4.heartbeat_stability_s",
            ),
            required_parameters=tuple(
                sorted(
                    (
                        _px4_parameter_name(name),
                        float(expected),
                    )
                    for name, expected in parameters_doc.items()
                )
            ),
        ),
        flight_log=FlightLogConfig(
            enabled=_boolean(
                flight_doc.get("enabled", True),
                "flight_log.enabled",
            ),
            retrieve_every_runs=_nonnegative_int(
                flight_doc.get("retrieve_every_runs", 5),
                "flight_log.retrieve_every_runs",
            ),
            retrieve_at_batch_end=_boolean(
                flight_doc.get("retrieve_at_batch_end", True),
                "flight_log.retrieve_at_batch_end",
            ),
            retrieval_mode=str(flight_doc.get("retrieval_mode", "deferred")).strip(),
            reboot_pixhawk_after_run=_boolean(
                flight_doc.get("reboot_pixhawk_after_run", False),
                "flight_log.reboot_pixhawk_after_run",
            ),
            reboot_timeout_s=_positive_float(
                flight_doc.get("reboot_timeout_s", 30.0),
                "flight_log.reboot_timeout_s",
            ),
            reboot_stability_s=_positive_float(
                flight_doc.get("reboot_stability_s", 3.0),
                "flight_log.reboot_stability_s",
            ),
        ),
        experiments=tuple(run_specs),
    )
    _validate_integrated_config(config)
    return config


def select_experiment(
    config: IntegratedRigConfig,
    experiment_number: int,
    *,
    require_enabled: bool = True,
) -> IntegratedRunSpec:
    for spec in config.experiments:
        if spec.number == experiment_number:
            if require_enabled and not spec.enabled:
                raise IntegratedRigError(
                    f"experiment {experiment_number} ({spec.name}) is disabled "
                    "in the configuration"
                )
            return spec
    available = ", ".join(str(spec.number) for spec in config.experiments)
    raise IntegratedRigError(
        f"unknown experiment number {experiment_number}; available: {available}"
    )


def _experiment_metadata(spec: IntegratedRunSpec) -> dict[str, Any]:
    return {
        "number": spec.number,
        "condition_id": spec.condition_id or f"E{spec.number:03d}",
        "attempt": spec.attempt,
        "enabled": spec.enabled,
        "name": spec.name,
        "purpose": spec.purpose,
        "operator_check": spec.operator_check,
        "mode": (
            "capture_only"
            if spec.capture_only_duration_s is not None
            else "executor"
        ),
        "capture_only_duration_s": spec.capture_only_duration_s,
        "day": spec.day,
        "block": spec.block,
        "priority": spec.priority,
        "dataset_role": spec.dataset_role,
        "apparatus": spec.apparatus,
        "fixture_mode": spec.fixture_mode,
        "pitch_deg": spec.pitch_deg,
        "beta_deg": spec.beta_deg,
        "mount_instance": spec.mount_instance,
        "wind_speed_mps": spec.wind_speed_mps,
        "voltage_min_v": spec.voltage_min_v,
        "voltage_max_v": spec.voltage_max_v,
        "repeat": spec.repeat,
        "depends_on": list(spec.depends_on),
        "blocked_by": list(spec.blocked_by),
        "executor_kind": spec.executor_kind,
    }


def _experiment_condition_text(spec: IntegratedRunSpec) -> str:
    pitch = (
        f"{spec.pitch_deg:g} deg"
        if spec.pitch_deg is not None
        else "free"
    )
    return (
        f"day={spec.day}, block={spec.block}, apparatus={spec.apparatus}, "
        f"condition={spec.condition_id or f'E{spec.number:03d}'}, "
        f"attempt={spec.attempt}, fixture={spec.fixture_mode}, pitch={pitch}, "
        f"beta={spec.beta_deg:g} deg, mount={spec.mount_instance}, "
        f"wind={spec.wind_speed_mps:g} m/s, "
        f"voltage={spec.voltage_min_v:g}..{spec.voltage_max_v:g} V, "
        f"repeat={spec.repeat}, role={spec.dataset_role}, "
        f"priority={spec.priority}"
    )


def inspect_executor_config(
    path: Path,
    *,
    expected_uart_device: str,
    expected_uart_baud: int,
    expected_target_system: int,
    expected_target_component: int,
    maximum_motor_command: float,
    expected_executor_kind: str | None = None,
) -> ExecutorConfigSummary:
    document = json.loads(path.read_text(encoding="utf-8"))
    return _inspect_executor_document(
        document,
        path=path,
        expected_uart_device=expected_uart_device,
        expected_uart_baud=expected_uart_baud,
        expected_target_system=expected_target_system,
        expected_target_component=expected_target_component,
        maximum_motor_command=maximum_motor_command,
        expected_executor_kind=expected_executor_kind,
    )


def _materialize_run_start_pitch_reference(
    document: dict[str, Any],
    telemetry: dict[str, Any],
    *,
    path: Path,
) -> dict[str, Any] | None:
    metadata = document.get("metadata")
    if not isinstance(metadata, dict):
        return None
    reference_mode = str(metadata.get("angle_guard_reference", "")).strip()
    if not reference_mode:
        return None
    if reference_mode != "RUN_START_PX4_PITCH":
        raise IntegratedRigError(
            f"unsupported angle_guard_reference {reference_mode!r}: {path}"
        )
    angle_guard = document.get("angle_guard")
    if not isinstance(angle_guard, dict) or angle_guard.get("enabled") is not True:
        raise IntegratedRigError(
            f"run-start pitch reference requires an enabled angle_guard: {path}"
        )
    pitch_mean_deg = _optional_float(telemetry.get("pitch_mean_deg"))
    pitch_min_deg = _optional_float(telemetry.get("pitch_min_deg"))
    pitch_max_deg = _optional_float(telemetry.get("pitch_max_deg"))
    pitch_rate_abs_max_dps = _optional_float(
        telemetry.get("pitch_rate_abs_max_dps")
    )
    if (
        pitch_mean_deg is None
        or pitch_min_deg is None
        or pitch_max_deg is None
        or pitch_rate_abs_max_dps is None
    ):
        raise IntegratedRigError(
            "live PX4 pitch reference was unavailable; no executor start was requested"
        )
    pitch_span_deg = pitch_max_deg - pitch_min_deg
    if pitch_span_deg > 3.0 or pitch_rate_abs_max_dps > 10.0:
        raise IntegratedRigError(
            "pitch was not stationary enough to establish the run-start zero: "
            f"span={pitch_span_deg:.3f} deg, "
            f"max_rate={pitch_rate_abs_max_dps:.3f} deg/s"
        )
    pitch_sign = _optional_float(angle_guard.get("attitude_pitch_sign"))
    if pitch_sign not in {-1.0, 1.0}:
        raise IntegratedRigError(
            f"invalid angle_guard.attitude_pitch_sign: {path}"
        )
    center_deg = pitch_sign * pitch_mean_deg
    angle_guard["attitude_center_deg"] = center_deg
    metadata["materialized_attitude_center_deg"] = center_deg
    metadata["materialized_pitch_mean_deg"] = pitch_mean_deg
    metadata["materialized_pitch_span_deg"] = pitch_span_deg
    metadata["materialized_pitch_rate_abs_max_dps"] = pitch_rate_abs_max_dps
    return {
        "mode": reference_mode,
        "attitude_center_deg": center_deg,
        "pitch_mean_deg": pitch_mean_deg,
        "pitch_min_deg": pitch_min_deg,
        "pitch_max_deg": pitch_max_deg,
        "pitch_span_deg": pitch_span_deg,
        "pitch_rate_abs_max_dps": pitch_rate_abs_max_dps,
    }


def _inspect_executor_document(
    document: Any,
    *,
    path: Path,
    expected_uart_device: str,
    expected_uart_baud: int,
    expected_target_system: int,
    expected_target_component: int,
    maximum_motor_command: float,
    expected_executor_kind: str | None = None,
) -> ExecutorConfigSummary:
    if not isinstance(document, dict):
        raise ValueError(f"executor config must contain an object: {path}")
    run_name = str(document.get("run_name", "")).strip()
    if not SAFE_RUN_NAME.fullmatch(run_name):
        raise ValueError(f"executor run_name is missing or unsafe: {path}")
    uart = document.get("uart")
    waveform = document.get("waveform")
    if not isinstance(uart, dict) or not isinstance(waveform, dict):
        raise ValueError(f"executor config is missing uart or waveform: {path}")
    if str(uart.get("device")) != expected_uart_device:
        raise ValueError(
            f"executor uart device mismatch: {uart.get('device')} != {expected_uart_device}"
        )
    if int(uart.get("baud", 0)) != expected_uart_baud:
        raise ValueError(
            f"executor uart baud mismatch: {uart.get('baud')} != {expected_uart_baud}"
        )
    if int(uart.get("target_system", -1)) != expected_target_system:
        raise ValueError(
            "executor target system mismatch: "
            f"{uart.get('target_system')} != {expected_target_system}"
        )
    if int(uart.get("target_component", -1)) != expected_target_component:
        raise ValueError(
            "executor target component mismatch: "
            f"{uart.get('target_component')} != {expected_target_component}"
        )
    executor_kind = (
        "offboard"
        if "control_mode" in document and "offboard" in document
        else "direct_motor"
    )
    if (
        expected_executor_kind is not None
        and executor_kind != expected_executor_kind
    ):
        raise ValueError(
            f"executor kind mismatch: {executor_kind} != "
            f"{expected_executor_kind}: {path}"
        )
    if executor_kind == "offboard":
        return _inspect_offboard_executor_document(
            document,
            path=path,
            run_name=run_name,
            waveform=waveform,
            maximum_motor_command=maximum_motor_command,
        )
    if document.get("motor_mapping_verified") is not True:
        raise ValueError(f"motor_mapping_verified is not true: {path}")
    if document.get("collective_verified") is not True:
        raise ValueError(f"collective_verified is not true: {path}")
    metadata = document.get("metadata", {})
    if (
        isinstance(metadata, dict)
        and metadata.get("trim_required") is True
        and document.get("axis_bias_verified") is not True
    ):
        raise ValueError(
            f"axis_bias_verified is not true for trim-required config: {path}"
        )

    sample_rate_hz = int(document.get("sample_rate_hz", 0))
    if sample_rate_hz != 100:
        raise ValueError(f"executor sample_rate_hz must be 100: {path}")
    timeout_ms = int(document.get("timeout_ms", 0))
    if not 1 <= timeout_ms <= 150:
        raise ValueError(f"executor timeout_ms must be 1..150: {path}")

    collective = _bounded_float(document.get("collective", 0.0), "collective", 0.0, 1.0)
    motor_min = _bounded_float(document.get("motor_min", 0.0), "motor_min", 0.0, 1.0)
    motor_max = _bounded_float(document.get("motor_max", 0.0), "motor_max", 0.0, 1.0)
    axis_mix = document.get("axis_mix")
    if not isinstance(axis_mix, list) or len(axis_mix) != 4:
        raise ValueError(f"executor axis_mix must contain four values: {path}")
    axis_values = tuple(
        _bounded_float(value, f"axis_mix[{index}]", -1.0, 1.0)
        for index, value in enumerate(axis_mix)
    )
    axis_bias = _bounded_float(
        float(document.get("axis_bias", 0.0)),
        "axis_bias",
        -1.0,
        1.0,
    )
    amplitude = _bounded_float(
        abs(float(waveform.get("amplitude", 0.0))),
        "waveform.amplitude",
        0.0,
        1.0,
    )
    requested = tuple(
        collective + abs(value) * (abs(axis_bias) + amplitude)
        for value in axis_values
    )
    peak = max((collective, *requested))
    if motor_min > motor_max:
        raise ValueError(f"invalid executor motor bounds: {path}")
    if peak > motor_max + 1e-12:
        raise ValueError(f"waveform peak {peak:g} exceeds executor motor_max {motor_max:g}")
    if peak > maximum_motor_command + 1e-12:
        raise ValueError(
            f"waveform peak {peak:g} exceeds integrated safety limit "
            f"{maximum_motor_command:g}"
        )

    startup_rate = _positive_float(
        document.get("startup_ramp_rate"),
        "startup_ramp_rate",
    )
    normal_rate = _positive_float(
        document.get("normal_ramp_rate"),
        "normal_ramp_rate",
    )
    waveform_duration = sum(
        _nonnegative_float(waveform.get(key, 0.0), f"waveform.{key}")
        for key in ("preroll_s", "active_s", "posthold_s")
    )
    expected_duration = peak / startup_rate + waveform_duration + peak / normal_rate
    return ExecutorConfigSummary(
        run_name=run_name,
        expected_duration_s=expected_duration,
        peak_motor_command=peak,
        sample_rate_hz=sample_rate_hz,
        executor_kind="direct_motor",
    )


def _inspect_offboard_executor_document(
    document: dict[str, Any],
    *,
    path: Path,
    run_name: str,
    waveform: dict[str, Any],
    maximum_motor_command: float,
) -> ExecutorConfigSummary:
    if int(document.get("schema_version", 0)) != 1:
        raise ValueError(f"offboard schema_version must be 1: {path}")
    control_mode = str(document.get("control_mode", ""))
    if control_mode not in {"ATTITUDE", "BODY_RATE"}:
        raise ValueError(f"invalid offboard control_mode: {path}")
    axis = str(document.get("axis", "PITCH"))
    if axis not in {"PITCH", "YAW"}:
        raise ValueError(f"invalid offboard axis: {path}")
    if axis == "YAW" and control_mode != "BODY_RATE":
        raise ValueError(f"yaw Offboard experiments require BODY_RATE: {path}")
    sample_rate_hz = int(document.get("sample_rate_hz", 0))
    if sample_rate_hz != 100:
        raise ValueError(f"offboard sample_rate_hz must be 100: {path}")
    waveform_type = str(waveform.get("type", ""))
    if waveform_type not in {
        "HOLD",
        "PULSE",
        "SINE",
        "LOG_CHIRP",
        "ANGLE_SEQUENCE",
    }:
        raise ValueError(f"invalid offboard waveform type: {path}")

    authorization = document.get("authorization")
    offboard = document.get("offboard")
    if not isinstance(authorization, dict) or not isinstance(offboard, dict):
        raise ValueError(
            f"offboard config is missing authorization or offboard: {path}"
        )
    for field in (
        "allow_arming",
        "rc_kill_tested",
        "mechanical_restraint_verified",
    ):
        if authorization.get(field) is not True:
            raise ValueError(f"authorization.{field} is not true: {path}")
    safety_mode = authorization.get("mode")
    if safety_mode is None:
        safety_mode = (
            "AUTOMATIC_OFFBOARD_LOSS"
            if authorization.get("offboard_loss_action_tested") is True
            else "RC_KILL_PRIMARY"
        )
    if safety_mode == "AUTOMATIC_OFFBOARD_LOSS":
        if authorization.get("offboard_loss_action_tested") is not True:
            raise ValueError(
                f"automatic Offboard-loss action is not verified: {path}"
            )
    elif safety_mode == "RC_KILL_PRIMARY":
        for field in (
            "propellers_installed_verified",
            "dedicated_rc_kill_operator_verified",
            "rc_kill_response_rehearsal_verified",
        ):
            if authorization.get(field) is not True:
                raise ValueError(f"authorization.{field} is not true: {path}")
    else:
        raise ValueError(f"invalid authorization.mode: {path}")

    thrust = _bounded_float(
        offboard.get("thrust"), "offboard.thrust", 0.0, 1.0
    )
    if thrust > maximum_motor_command + 1e-12:
        raise ValueError(
            f"offboard thrust {thrust:g} exceeds integrated safety limit "
            f"{maximum_motor_command:g}"
        )
    startup_s = _positive_float(
        offboard.get("startup_ramp_s"), "offboard.startup_ramp_s"
    )
    shutdown_s = _positive_float(
        offboard.get("shutdown_ramp_s"), "offboard.shutdown_ramp_s"
    )
    prestream_s = _positive_float(
        offboard.get("prestream_s"), "offboard.prestream_s"
    )
    mode_timeout_s = _positive_float(
        offboard.get("mode_timeout_s"), "offboard.mode_timeout_s"
    )
    arm_timeout_s = _positive_float(
        offboard.get("arm_timeout_s"), "offboard.arm_timeout_s"
    )
    max_abs_pitch_deg = _bounded_float(
        offboard.get("max_abs_pitch_deg"),
        "offboard.max_abs_pitch_deg",
        0.01,
        180.0,
    )
    _bounded_float(
        offboard.get("max_abs_pitch_rate_dps"),
        "offboard.max_abs_pitch_rate_dps",
        0.01,
        500.0,
    )
    _bounded_float(
        offboard.get("max_abs_yaw_deg", 15.0),
        "offboard.max_abs_yaw_deg",
        0.01,
        180.0,
    )
    _bounded_float(
        offboard.get("max_abs_yaw_rate_dps", 40.0),
        "offboard.max_abs_yaw_rate_dps",
        0.01,
        500.0,
    )
    active_s = _positive_float(
        waveform.get("active_s"), "waveform.active_s"
    )
    if waveform_type == "ANGLE_SEQUENCE":
        if axis != "PITCH" or control_mode != "ATTITUDE":
            raise ValueError(
                f"ANGLE_SEQUENCE requires PITCH ATTITUDE control: {path}"
            )
        targets = waveform.get("targets_deg")
        if not isinstance(targets, list) or not targets:
            raise ValueError(
                f"ANGLE_SEQUENCE targets_deg must be a nonempty list: {path}"
            )
        parsed_targets = [
            _bounded_float(
                target,
                f"waveform.targets_deg[{index}]",
                -max_abs_pitch_deg,
                max_abs_pitch_deg,
            )
            for index, target in enumerate(targets)
        ]
        if any(abs(target) >= max_abs_pitch_deg for target in parsed_targets):
            raise ValueError(
                f"ANGLE_SEQUENCE target reaches the pitch abort limit: {path}"
            )
        transition_s = _positive_float(
            waveform.get("transition_s"), "waveform.transition_s"
        )
        hold_s = _positive_float(
            waveform.get("hold_s"), "waveform.hold_s"
        )
        repetitions = int(waveform.get("repetitions", 1))
        if not 1 <= repetitions <= 10:
            raise ValueError(
                f"ANGLE_SEQUENCE repetitions must be 1..10: {path}"
            )
        sequence_duration = (
            len(parsed_targets) * repetitions * (transition_s + hold_s)
        )
        if abs(active_s - sequence_duration) > 0.005:
            raise ValueError(
                "ANGLE_SEQUENCE active_s does not match its move/hold "
                f"segments: {path}"
            )
    waveform_duration = sum(
        _nonnegative_float(waveform.get(key, 0.0), f"waveform.{key}")
        for key in ("preroll_s", "active_s", "posthold_s")
    )
    expected_duration = (
        prestream_s
        + mode_timeout_s
        + arm_timeout_s
        + startup_s
        + waveform_duration
        + shutdown_s
    )
    return ExecutorConfigSummary(
        run_name=run_name,
        expected_duration_s=expected_duration,
        peak_motor_command=thrust,
        sample_rate_hz=sample_rate_hz,
        executor_kind="offboard",
        control_mode=control_mode,
        axis=axis,
    )


def inspect_safe_executor_config(path: Path) -> str:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"safe executor config must contain an object: {path}")
    if document.get("motor_mapping_verified") is not False:
        raise ValueError("safe executor config must set motor_mapping_verified=false")
    if document.get("collective_verified") is not False:
        raise ValueError("safe executor config must set collective_verified=false")
    return str(document.get("run_name", "safe-locked"))


def choose_ssh_host(
    config: SshConfig,
    *,
    run_command: RunCommand = subprocess.run,
) -> tuple[str | None, tuple[str, ...]]:
    errors: list[str] = []
    for candidate in config.host_candidates:
        args = _ssh_base_args(config) + [candidate, "printf FFTGYRO_SSH_OK"]
        try:
            completed = run_command(
                args,
                text=True,
                capture_output=True,
                timeout=config.connect_timeout_s + 2.0,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            errors.append(f"{candidate}: {type(error).__name__}: {error}")
            continue
        if completed.returncode == 0 and completed.stdout.strip() == "FFTGYRO_SSH_OK":
            return candidate, tuple(errors)
        detail = completed.stderr.strip() or completed.stdout.strip() or "connection failed"
        errors.append(f"{candidate}: {detail}")
    return None, tuple(errors)


class SshTransport:
    def __init__(
        self,
        config: SshConfig,
        host: str,
        *,
        run_command: RunCommand = subprocess.run,
    ) -> None:
        self.config = config
        self.host = host
        self._run_command = run_command

    def run(
        self,
        remote_command: str,
        *,
        timeout: float = 15.0,
        input_text: str | None = None,
        check: bool = True,
        transport_retries: int = 0,
    ) -> CommandResult:
        args = _ssh_base_args(self.config) + [self.host, remote_command]
        if transport_retries < 0:
            raise ValueError("transport_retries must be nonnegative")
        for attempt in range(transport_retries + 1):
            try:
                completed = self._run_command(
                    args,
                    text=True,
                    input=input_text,
                    capture_output=True,
                    timeout=timeout,
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired) as error:
                if attempt < transport_retries:
                    time.sleep(0.25 * (attempt + 1))
                    continue
                raise IntegratedRigError(
                    f"SSH transport failed for {self.host}: "
                    f"{type(error).__name__}: {error}"
                ) from error
            if completed.returncode == 255 and attempt < transport_retries:
                time.sleep(0.25 * (attempt + 1))
                continue
            break
        result = CommandResult(
            args=tuple(str(value) for value in args),
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )
        if check and result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip()
            raise IntegratedRigError(
                f"SSH command failed ({result.returncode}): {detail}"
            )
        return result

    def upload(self, local_path: Path, remote_path: str) -> None:
        args = _scp_base_args(self.config) + [
            str(local_path),
            f"{self.host}:{remote_path}",
        ]
        try:
            completed = self._run_command(
                args,
                text=True,
                capture_output=True,
                timeout=30.0,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise IntegratedRigError(
                f"SCP upload transport failed for {self.host}: "
                f"{type(error).__name__}: {error}"
            ) from error
        if completed.returncode != 0:
            raise IntegratedRigError(
                "SCP upload failed: "
                + (completed.stderr.strip() or completed.stdout.strip())
            )

    def download(self, remote_path: str, local_path: Path) -> None:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        args = _scp_base_args(self.config) + [
            f"{self.host}:{remote_path}",
            str(local_path),
        ]
        try:
            completed = self._run_command(
                args,
                text=True,
                capture_output=True,
                timeout=120.0,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise IntegratedRigError(
                f"SCP download transport failed for {self.host}: "
                f"{type(error).__name__}: {error}"
            ) from error
        if completed.returncode != 0:
            raise IntegratedRigError(
                "SCP download failed: "
                + (completed.stderr.strip() or completed.stdout.strip())
            )


def run_preflight(
    config: IntegratedRigConfig,
    *,
    run_command: RunCommand = subprocess.run,
    experiment_number: int | None = None,
) -> PreflightReport:
    checks: list[PreflightCheck] = list(
        _local_preflight_checks(config, run_command=run_command)
    )
    selected_spec = (
        select_experiment(config, experiment_number, require_enabled=False)
        if experiment_number is not None
        else None
    )
    experiments = (
        (selected_spec,)
        if selected_spec is not None
        else config.experiments
    )
    for spec in experiments:
        if spec.capture_only_duration_s is not None:
            checks.append(
                PreflightCheck(
                    f"executor_config:{spec.name}",
                    "pass",
                    (
                        "capture_only; executor remains inactive; "
                        f"expected={spec.capture_only_duration_s:.1f}s"
                    ),
                    spec.enabled,
                )
            )
            continue
        try:
            if spec.executor_config is None:
                raise ValueError("executor_config is not configured")
            summary = inspect_executor_config(
                spec.executor_config,
                expected_uart_device=config.pi.uart_device,
                expected_uart_baud=config.pi.uart_baud,
                expected_target_system=config.px4.expected_system_id,
                expected_target_component=config.px4.expected_component_id,
                maximum_motor_command=config.safety.maximum_motor_command,
                expected_executor_kind=spec.executor_kind,
            )
            checks.append(
                PreflightCheck(
                    f"executor_config:{spec.name}",
                    "pass",
                    (
                        f"{summary.run_name}, kind={summary.executor_kind}, "
                        f"peak={summary.peak_motor_command:.3f}, "
                        f"expected={summary.expected_duration_s:.1f}s"
                    ),
                    spec.enabled,
                )
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            checks.append(
                PreflightCheck(
                    f"executor_config:{spec.name}",
                    "fail",
                    str(error),
                    spec.enabled,
                )
            )

    try:
        safe_name = inspect_safe_executor_config(config.pi.safe_executor_config)
        checks.append(
            PreflightCheck(
                "safe_executor_config",
                "pass",
                f"{safe_name}: {config.pi.safe_executor_config}",
                True,
            )
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        checks.append(
            PreflightCheck("safe_executor_config", "fail", str(error), True)
        )

    node_path = shutil.which(config.acquisition.node_executable)
    if node_path is None:
        checks.append(
            PreflightCheck(
                "node_runtime",
                "fail",
                f"not found: {config.acquisition.node_executable}",
                True,
            )
        )
    else:
        checks.append(PreflightCheck("node_runtime", "pass", node_path, True))
    for executable in ("ssh", "scp"):
        resolved = shutil.which(executable)
        checks.append(
            PreflightCheck(
                f"{executable}_client",
                "pass" if resolved else "fail",
                resolved or "not found",
                True,
            )
        )
    checks.append(
        PreflightCheck(
            "acquisition_script",
            (
                "pass"
                if (
                    config.acquisition.script.is_file()
                    and _sha256_file(config.acquisition.script)
                    == config.acquisition.script_sha256
                )
                else "fail"
            ),
            (
                f"{config.acquisition.script}; expected_sha256="
                f"{config.acquisition.script_sha256}"
            ),
            True,
        )
    )
    checks.append(
        PreflightCheck(
            "acquisition_processing",
            (
                "pass"
                if (
                    config.acquisition.processing_module.is_file()
                    and _sha256_file(config.acquisition.processing_module)
                    == config.acquisition.processing_sha256
                )
                else "fail"
            ),
            (
                f"{config.acquisition.processing_module}; expected_sha256="
                f"{config.acquisition.processing_sha256}"
            ),
            True,
        )
    )

    acquisition_probe = _probe_acquisition(config, run_command=run_command)
    checks.extend(acquisition_probe)

    ssh_host: str | None = None
    transport: SshTransport | None = None
    remote: dict[str, Any] | None = None
    ssh_errors: list[str] = []
    # The remote preflight is itself a stronger SSH reachability check than a
    # separate "printf" probe.  Combining them avoids opening two consecutive
    # SSH sessions before every experiment while retaining candidate fallback.
    for candidate in config.ssh.host_candidates:
        candidate_transport = SshTransport(
            config.ssh,
            candidate,
            run_command=run_command,
        )
        try:
            candidate_remote = _remote_preflight(config, candidate_transport)
        except (
            IntegratedRigError,
            json.JSONDecodeError,
            OSError,
            ValueError,
        ) as error:
            ssh_errors.append(
                f"{candidate}: {type(error).__name__}: {error}"
            )
            continue
        ssh_host = candidate
        transport = candidate_transport
        remote = candidate_remote
        break
    if ssh_host is None or transport is None or remote is None:
        checks.append(
            PreflightCheck(
                "ssh",
                "fail",
                " | ".join(ssh_errors) or "no SSH candidate succeeded",
                True,
            )
        )
        return PreflightReport(
            ready=False,
            ssh_host=None,
            voltage_v=None,
            current_a=None,
            rc_kill_raw=None,
            checks=tuple(checks),
        )
    checks.append(PreflightCheck("ssh", "pass", ssh_host, True))

    remote_checks = (
        (
            "pi_identity",
            (
                remote.get("hostname") == config.pi.expected_hostname
                and remote.get("user") == config.pi.expected_user
            ),
            (
                f"{remote.get('user')}@{remote.get('hostname')}; expected "
                f"{config.pi.expected_user}@{config.pi.expected_hostname}"
            ),
        ),
        ("pi_executor", bool(remote.get("executor_exists")), config.pi.executor_binary),
        (
            "pi_executor_hash",
            remote.get("executor_sha256") == config.pi.executor_sha256,
            (
                f"{remote.get('executor_sha256')}; expected "
                f"{config.pi.executor_sha256}"
            ),
        ),
        (
            "pi_safe_config_installed",
            (
                remote.get("executor_config_sha256")
                == _sha256_file(config.pi.safe_executor_config)
            ),
            (
                f"{remote.get('executor_config_sha256')}; expected "
                f"{_sha256_file(config.pi.safe_executor_config)}"
            ),
        ),
        ("pi_uart", bool(remote.get("uart_exists")), config.pi.uart_device),
        (
            "pi_uart_mapping",
            remote.get("uart_realpath") == config.pi.uart_realpath,
            (
                f"{remote.get('uart_realpath')}; expected "
                f"{config.pi.uart_realpath}"
            ),
        ),
        ("pi_mavlink_python", bool(remote.get("mavlink_python_exists")), config.pi.mavlink_python),
        (
            "pi_service_inactive",
            remote.get("service_state") == "inactive",
            f"state={remote.get('service_state')}",
        ),
        (
            "pi_service_disabled",
            remote.get("service_enabled_state") == "disabled",
            f"enabled={remote.get('service_enabled_state')}",
        ),
        (
            "pi_service_user",
            remote.get("service_user") == config.pi.expected_user,
            (
                f"user={remote.get('service_user')}; expected="
                f"{config.pi.expected_user}"
            ),
        ),
        (
            "pi_service_exec_start",
            (
                config.pi.executor_binary in str(remote.get("service_exec_start"))
                and " run " in str(remote.get("service_exec_start"))
                and config.pi.executor_config in str(remote.get("service_exec_start"))
            ),
            str(remote.get("service_exec_start")),
        ),
        (
            "pi_service_exec_start_pre",
            (
                config.pi.executor_binary in str(remote.get("service_exec_start_pre"))
                and " validate " in str(remote.get("service_exec_start_pre"))
                and config.pi.executor_config
                in str(remote.get("service_exec_start_pre"))
            ),
            str(remote.get("service_exec_start_pre")),
        ),
        (
            "pi_service_no_restart",
            remote.get("service_restart") == "no",
            f"Restart={remote.get('service_restart')}",
        ),
        (
            "pi_no_executor_process",
            not bool(remote.get("executor_processes")),
            f"processes={remote.get('executor_processes') or 'none'}",
        ),
        (
            "pi_no_stale_restore_timer",
            not bool(remote.get("active_restore_timers")),
            f"timers={remote.get('active_restore_timers') or 'none'}",
        ),
        (
            "pi_fuser_available",
            bool(remote.get("fuser_available")),
            "required to prove UART exclusivity",
        ),
        (
            "pi_uart_free",
            not bool(remote.get("uart_users")),
            f"users={remote.get('uart_users') or 'none'}",
        ),
        (
            "pi_serial_getty_disabled",
            remote.get("serial_getty_state") != "active",
            f"state={remote.get('serial_getty_state')}",
        ),
        (
            "pi_disk_space",
            int(remote.get("free_mb", 0)) >= config.pi.minimum_free_mb,
            f"free={remote.get('free_mb')}MB required={config.pi.minimum_free_mb}MB",
        ),
        (
            "pi_uptime",
            float(remote.get("uptime_s", 0.0)) >= config.pi.minimum_uptime_s,
            (
                f"uptime={float(remote.get('uptime_s', 0.0)):.1f}s "
                f"required={config.pi.minimum_uptime_s:.1f}s"
            ),
        ),
        (
            "pi_rootfs_writable",
            bool(remote.get("rootfs_writable")),
            f"root_options={remote.get('root_options')}",
        ),
        (
            "pi_no_current_throttling",
            int(remote.get("throttled_flags", 0)) & 0xF == 0,
            f"get_throttled=0x{int(remote.get('throttled_flags', 0)):x}",
        ),
        (
            "pi_vcgencmd_available",
            bool(remote.get("vcgencmd_available")),
            "required to check current undervoltage/throttling",
        ),
        (
            "pi_no_uploader",
            not bool(remote.get("uploader_processes")),
            f"processes={remote.get('uploader_processes') or 'none'}",
        ),
        (
            "pi_no_filesystem_errors",
            not bool(remote.get("filesystem_errors")),
            f"errors={remote.get('filesystem_errors') or 'none'}",
        ),
        (
            "pi_sudo",
            bool(remote.get("sudo_systemctl_ok")),
            "passwordless scoped systemctl access",
        ),
        (
            "pi_safe_restore_scheduler",
            bool(remote.get("sudo_systemd_run_ok")),
            "passwordless transient restore timer support",
        ),
    )
    component_checks: list[tuple[str, bool, str]] = []
    for component, expected_binary, expected_digest in (
        (
            "direct",
            config.pi.direct_executor_binary,
            config.pi.direct_executor_sha256,
        ),
        (
            "offboard",
            config.pi.offboard_executor_binary,
            config.pi.offboard_executor_sha256,
        ),
    ):
        if expected_binary is None or expected_digest is None:
            continue
        actual = remote.get(f"{component}_executor_sha256")
        component_checks.extend(
            (
                (
                    f"pi_{component}_executor",
                    bool(remote.get(f"{component}_executor_exists")),
                    expected_binary,
                ),
                (
                    f"pi_{component}_executor_hash",
                    actual == expected_digest,
                    f"{actual}; expected {expected_digest}",
                ),
            )
        )
    for name, passed, detail in (*remote_checks, *component_checks):
        checks.append(
            PreflightCheck(name, "pass" if passed else "fail", str(detail), True)
        )

    voltage_v: float | None = None
    current_a: float | None = None
    rc_kill_raw: int | None = None
    if all(check.passed for check in checks if check.name in {"pi_uart", "pi_uart_free"}):
        try:
            telemetry = _probe_uart(config, transport)
            voltage_v = _optional_float(telemetry.get("voltage_v"))
            current_a = _optional_float(telemetry.get("current_a"))
            rc_values = telemetry.get("rc_channels")
            if isinstance(rc_values, list) and len(rc_values) >= config.safety.rc_kill_channel:
                rc_kill_raw = int(rc_values[config.safety.rc_kill_channel - 1])
            checks.append(
                PreflightCheck(
                    "pixhawk_heartbeat",
                    "pass" if telemetry.get("heartbeat") else "fail",
                    (
                        f"system={telemetry.get('system')} "
                        f"component={telemetry.get('component')} "
                        f"autopilot={telemetry.get('autopilot')} "
                        f"type={telemetry.get('vehicle_type')} "
                        f"armed={telemetry.get('armed')}"
                    ),
                    True,
                )
            )
            identity_ok = (
                int(telemetry.get("system", -1)) == config.px4.expected_system_id
                and int(telemetry.get("component", -1))
                == config.px4.expected_component_id
                and int(telemetry.get("autopilot", -1))
                == config.px4.expected_autopilot
                and int(telemetry.get("vehicle_type", -1))
                == config.px4.expected_vehicle_type
            )
            checks.append(
                PreflightCheck(
                    "pixhawk_identity",
                    "pass" if identity_ok else "fail",
                    (
                        f"actual={telemetry.get('system')}/"
                        f"{telemetry.get('component')}/"
                        f"{telemetry.get('autopilot')}/"
                        f"{telemetry.get('vehicle_type')}; expected="
                        f"{config.px4.expected_system_id}/"
                        f"{config.px4.expected_component_id}/"
                        f"{config.px4.expected_autopilot}/"
                        f"{config.px4.expected_vehicle_type}"
                    ),
                    True,
                )
            )
            flight_git_prefix = str(
                telemetry.get("flight_git_prefix") or ""
            ).lower()
            firmware_ok = flight_git_prefix.startswith(
                config.px4.expected_flight_git_prefix
            )
            checks.append(
                PreflightCheck(
                    "pixhawk_firmware_base",
                    "pass" if firmware_ok else "fail",
                    (
                        f"git_prefix={flight_git_prefix or 'unavailable'}; "
                        f"expected={config.px4.expected_flight_git_prefix}; "
                        f"raw_custom_version="
                        f"{telemetry.get('flight_custom_version') or 'unavailable'}"
                    ),
                    True,
                )
            )
            stable_ok = (
                bool(telemetry.get("identity_stable"))
                and int(telemetry.get("heartbeat_count", 0)) >= 2
                and bool(telemetry.get("boot_time_monotonic"))
                and int(telemetry.get("boot_time_samples", 0)) >= 2
            )
            checks.append(
                PreflightCheck(
                    "pixhawk_stable",
                    "pass" if stable_ok else "fail",
                    (
                        f"heartbeats={telemetry.get('heartbeat_count')} "
                        f"boot_samples={telemetry.get('boot_time_samples')} "
                        f"boot_monotonic={telemetry.get('boot_time_monotonic')}"
                    ),
                    True,
                )
            )
            actual_parameters = telemetry.get("parameters")
            if not isinstance(actual_parameters, dict):
                actual_parameters = {}
            for parameter_name, expected in config.px4.required_parameters:
                actual = _optional_float(actual_parameters.get(parameter_name))
                parameter_ok = (
                    actual is not None and abs(actual - expected) <= 1e-3
                )
                checks.append(
                    PreflightCheck(
                        f"px4_param:{parameter_name}",
                        "pass" if parameter_ok else "fail",
                        f"actual={actual} expected={expected}",
                        True,
                    )
                )
            disarmed_ok = not bool(telemetry.get("armed"))
            checks.append(
                PreflightCheck(
                    "pixhawk_disarmed",
                    "pass" if disarmed_ok or not config.safety.require_disarmed else "fail",
                    f"armed={telemetry.get('armed')}",
                    config.safety.require_disarmed,
                )
            )
            voltage_min_v = _optional_float(telemetry.get("voltage_min_v"))
            voltage_max_v = _optional_float(telemetry.get("voltage_max_v"))
            allowed_voltage_min_v = (
                selected_spec.voltage_min_v
                if selected_spec is not None
                else config.safety.minimum_voltage_v
            )
            allowed_voltage_max_v = (
                selected_spec.voltage_max_v
                if selected_spec is not None
                else config.safety.maximum_voltage_v
            )
            voltage_ok = (
                voltage_v is not None
                and voltage_min_v is not None
                and voltage_max_v is not None
                and int(telemetry.get("voltage_samples", 0)) >= 2
                and voltage_min_v >= allowed_voltage_min_v
                and voltage_max_v <= allowed_voltage_max_v
            )
            checks.append(
                PreflightCheck(
                    "battery_voltage",
                    "pass" if voltage_ok else "fail",
                    (
                        f"{voltage_v:.3f}V; allowed "
                        f"{allowed_voltage_min_v:.2f}.."
                        f"{allowed_voltage_max_v:.2f}V; "
                        f"observed={voltage_min_v}..{voltage_max_v}V; "
                        f"source={telemetry.get('voltage_source')}"
                        if voltage_v is not None
                        else "battery voltage unavailable"
                    ),
                    True,
                )
            )
            rc_ok = (
                bool(telemetry.get("rc_received"))
                and rc_kill_raw is not None
                and 800 <= rc_kill_raw <= 2200
            )
            checks.append(
                PreflightCheck(
                    "rc_signal",
                    "pass" if rc_ok or not config.safety.require_rc_signal else "fail",
                    (
                        f"received={rc_ok}; CH{config.safety.rc_kill_channel}="
                        f"{rc_kill_raw if rc_kill_raw is not None else 'unavailable'} "
                        f"rssi={telemetry.get('rc_rssi')}"
                    ),
                    config.safety.require_rc_signal,
                )
            )
            kill_on_ok = (
                rc_ok
                and rc_kill_raw is not None
                and config.safety.rc_kill_on_min_raw
                <= rc_kill_raw
                <= config.safety.rc_kill_on_max_raw
            )
            checks.append(
                PreflightCheck(
                    "rc_kill_on",
                    "pass" if kill_on_ok else "fail",
                    (
                        f"CH{config.safety.rc_kill_channel}={rc_kill_raw}; "
                        f"expected Kill ON range="
                        f"{config.safety.rc_kill_on_min_raw}.."
                        f"{config.safety.rc_kill_on_max_raw}"
                    ),
                    True,
                )
            )
            if selected_spec is not None:
                try:
                    selected_interlock = _validate_live_interlock(
                        config,
                        telemetry,
                        selected_spec,
                    )
                except IntegratedRigError as error:
                    checks.append(
                        PreflightCheck(
                            "selected_experiment_interlock",
                            "fail",
                            str(error),
                            selected_spec.enabled,
                        )
                    )
                else:
                    checks.append(
                        PreflightCheck(
                            "selected_experiment_interlock",
                            "pass",
                            (
                                f"experiment={selected_spec.number}; allowed_voltage="
                                f"{selected_interlock['allowed_voltage_min_v']}.."
                                f"{selected_interlock['allowed_voltage_max_v']}V"
                            ),
                            selected_spec.enabled,
                        )
                    )
        except (IntegratedRigError, json.JSONDecodeError, OSError, ValueError) as error:
            checks.append(PreflightCheck("pixhawk_uart", "fail", str(error), True))

    ready = all(check.passed or not check.blocking for check in checks)
    return PreflightReport(
        ready=ready,
        ssh_host=ssh_host,
        voltage_v=voltage_v,
        current_a=current_a,
        rc_kill_raw=rc_kill_raw,
        checks=tuple(checks),
    )


class IntegratedRigRunner:
    def __init__(
        self,
        config: IntegratedRigConfig,
        *,
        run_command: RunCommand = subprocess.run,
        input_function: Callable[[str], str] = input,
        output: Callable[[str], None] = print,
    ) -> None:
        self.config = config
        self._run_command = run_command
        self._input = input_function
        self._output = output
        self._transport: SshTransport | None = None
        self._capture: subprocess.Popen[str] | None = None

    def preflight(self, experiment_number: int | None = None) -> PreflightReport:
        return run_preflight(
            self.config,
            run_command=self._run_command,
            experiment_number=experiment_number,
        )

    def run(
        self,
        experiment_number: int,
        *,
        dry_run: bool = False,
    ) -> tuple[Path, ...]:
        spec = select_experiment(self.config, experiment_number)
        report = self.preflight(experiment_number)
        self._print_preflight(report, compact=report.ready)
        if not report.ready or report.ssh_host is None:
            raise IntegratedRigError("preflight failed; no motor command was sent")
        if dry_run:
            self._output("DRY RUN COMPLETE: no motor command was sent")
            return ()

        summary = self._inspect_experiment(spec)
        execution_text = (
            f"計測のみ {summary.expected_duration_s:.1f} s（モーター停止を維持）"
            if spec.capture_only_duration_s is not None
            else (
                f"最大モーター指令 {summary.peak_motor_command:.3f}, "
                f"推定時間 {summary.expected_duration_s:.1f} s"
            )
        )
        response = self._input(
            "\n"
            f"実験 {spec.number}: {spec.name}\n"
            f"目的: {spec.purpose}\n"
            f"条件: {_experiment_condition_text(spec)}\n"
            f"直前確認: {spec.operator_check}\n"
            f"実行内容: {execution_text}\n"
            "プロペラ、固定具、ストッパー、退避、RC Kill ONを確認して "
            f"{self.config.safety.confirmation_phrase} と入力: "
        ).strip()
        if not _matches_physical_safety_confirmation(
            response,
            self.config.safety.confirmation_phrase,
        ):
            raise IntegratedRigError("physical safety confirmation did not match")

        self._transport = SshTransport(
            self.config.ssh,
            report.ssh_host,
            run_command=self._run_command,
        )
        completed_roots: list[Path] = []
        try:
            root = self._run_one(
                spec,
                index=spec.number,
                total=len(self.config.experiments),
            )
            completed_roots.append(root)
        except BaseException:
            # Capture-only experiments never install or start an executor.
            # Emitting an unconfirmed software-stop alarm after a transport
            # failure is therefore misleading; RC Kill remains ON throughout.
            if spec.capture_only_duration_s is None:
                self._emergency_software_stop()
            raise
        finally:
            self._stop_capture()
        return tuple(completed_roots)

    def experiment_catalog(self) -> tuple[dict[str, Any], ...]:
        catalog: list[dict[str, Any]] = []
        for spec in self.config.experiments:
            item = _experiment_metadata(spec)
            item["executor_config"] = (
                str(spec.executor_config)
                if spec.executor_config is not None
                else None
            )
            try:
                summary = self._inspect_experiment(spec)
            except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
                if spec.enabled:
                    raise
                item.update(
                    {
                        "executable_config_valid": False,
                        "validation_error": str(error),
                        "executor_run_name": None,
                        "peak_motor_command": None,
                        "expected_duration_s": spec.capture_only_duration_s,
                    }
                )
            else:
                item.update(
                    {
                        "executable_config_valid": True,
                        "validation_error": None,
                        "executor_run_name": summary.run_name,
                        "executor_kind": summary.executor_kind,
                        "control_mode": summary.control_mode,
                        "axis": summary.axis,
                        "peak_motor_command": summary.peak_motor_command,
                        "expected_duration_s": summary.expected_duration_s,
                    }
                )
            catalog.append(item)
        return tuple(catalog)

    def _inspect_experiment(
        self,
        spec: IntegratedRunSpec,
    ) -> ExecutorConfigSummary:
        if spec.capture_only_duration_s is not None:
            return ExecutorConfigSummary(
                run_name=f"{spec.name}_capture_only",
                expected_duration_s=spec.capture_only_duration_s,
                peak_motor_command=0.0,
                sample_rate_hz=0,
            )
        if spec.executor_config is None:
            raise ValueError(f"executor_config is not configured for {spec.name}")
        return inspect_executor_config(
            spec.executor_config,
            expected_uart_device=self.config.pi.uart_device,
            expected_uart_baud=self.config.pi.uart_baud,
            expected_target_system=self.config.px4.expected_system_id,
            expected_target_component=self.config.px4.expected_component_id,
            maximum_motor_command=self.config.safety.maximum_motor_command,
            expected_executor_kind=spec.executor_kind,
        )

    def status(self) -> dict[str, Any]:
        host, errors = choose_ssh_host(self.config.ssh, run_command=self._run_command)
        if host is None:
            return {"reachable": False, "errors": list(errors)}
        transport = SshTransport(self.config.ssh, host, run_command=self._run_command)
        command = (
            "printf 'SERVICE=%s\\n' \"$(systemctl is-active "
            f"{shlex.quote(self.config.pi.executor_service)} || true)\"; "
            "printf 'CONFIG=%s\\n' \"$(sha256sum "
            f"{shlex.quote(self.config.pi.executor_config)} "
            "2>/dev/null | cut -d' ' -f1)\"; "
            "printf 'LATEST=%s\\n' \"$(find "
            f"{shlex.quote(self.config.pi.log_directory)} -maxdepth 1 -type f "
            "-name '*.csv' -printf '%T@ %f\\n' 2>/dev/null | sort -nr | head -1)\""
        )
        result = transport.run(command, check=False)
        values = {
            key: value
            for line in result.stdout.splitlines()
            if "=" in line
            for key, value in (line.split("=", 1),)
        }
        installed_hash = values.get("CONFIG") or None
        return {
            "reachable": True,
            "ssh_host": host,
            "service_state": values.get("SERVICE") or "unknown",
            "installed_config_sha256": installed_hash,
            "safe_config_installed": (
                installed_hash == _sha256_file(self.config.pi.safe_executor_config)
            ),
            "latest_log": values.get("LATEST") or None,
        }

    def _close_px4_ulog_by_reboot(
        self,
        session: ExperimentSession,
        spec: IntegratedRunSpec,
    ) -> dict[str, Any]:
        assert self._transport is not None
        safe_sha256 = _sha256_file(self.config.pi.safe_executor_config)
        executor_pattern = (
            "^" + re.escape(self.config.pi.executor_binary) + r"([[:space:]]|$)"
        )
        self._transport.run(
            f"test \"$(systemctl is-active "
            f"{shlex.quote(self.config.pi.executor_service)} || true)\" = inactive && "
            f"! pgrep -f {shlex.quote(executor_pattern)} >/dev/null && "
            f"! fuser {shlex.quote(self.config.pi.uart_device)} "
            ">/dev/null 2>&1 && "
            f"test \"$(sha256sum "
            f"{shlex.quote(self.config.pi.executor_config)} "
            "| cut -d' ' -f1)\" = "
            f"{shlex.quote(safe_sha256)}",
            timeout=10.0,
        )
        interlock = _validate_live_interlock(
            self.config,
            _probe_uart(self.config, self._transport),
            spec,
            require_rc_kill_on=False,
            check_rc_kill=False,
        )
        session.append_event(
            "pixhawk_reboot_interlock",
            {
                **interlock,
                "executor_state": "inactive",
                "safe_executor_config_sha256": safe_sha256,
                "purpose": "close_SDLOG_MODE_2_ulog",
            },
        )
        reboot = _reboot_pixhawk(self.config, self._transport)
        session.append_event("pixhawk_reboot_verified", reboot)
        return reboot

    def stop(self) -> dict[str, Any]:
        host, errors = choose_ssh_host(self.config.ssh, run_command=self._run_command)
        if host is None:
            raise IntegratedRigError("cannot reach Pi: " + " | ".join(errors))
        transport = SshTransport(self.config.ssh, host, run_command=self._run_command)
        self._transport = transport
        self._stop_executor_and_confirm()
        backup_path, local_backup_hash = self._backup_installed_executor_config()
        remote_safe = f"/tmp/fftgyro-manual-stop-{int(time.time())}.safe.json"
        transport.upload(self.config.pi.safe_executor_config, remote_safe)
        self._restore_safe_executor_config(remote_safe, None, None)
        return {
            "ssh_host": host,
            "service_state": "inactive",
            "safe_config_restored": True,
            "previous_config_backup": str(backup_path),
            "previous_config_sha256": local_backup_hash,
        }

    def _backup_installed_executor_config(self) -> tuple[Path, str]:
        assert self._transport is not None
        backup_root = (
            self.config.session.output_root
            / "_maintenance"
            / (
                datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
                + "_before_safe_restore"
            )
        )
        backup_root.mkdir(parents=True, exist_ok=False)
        backup_path = backup_root / "pi-installed-run.json"
        remote_hash_result = self._transport.run(
            f"sha256sum {shlex.quote(self.config.pi.executor_config)} "
            "| cut -d' ' -f1"
        )
        remote_hash = remote_hash_result.stdout.strip().lower()
        if not re.fullmatch(r"[0-9a-f]{64}", remote_hash):
            raise IntegratedRigError(
                "could not establish the installed Pi config hash; "
                "safe restore was not attempted"
            )
        self._transport.download(self.config.pi.executor_config, backup_path)
        local_backup_hash = _sha256_file(backup_path)
        if local_backup_hash != remote_hash:
            raise IntegratedRigError(
                "installed Pi config backup hash mismatch; "
                "safe restore was not attempted"
            )
        _write_json_new(
            backup_root / "backup.json",
            {
                "created_at_utc": datetime.now(UTC).isoformat(),
                "ssh_host": self._transport.host,
                "remote_path": self.config.pi.executor_config,
                "backup_path": str(backup_path),
                "sha256": local_backup_hash,
            },
        )
        return backup_path, local_backup_hash

    def _run_one(self, spec: IntegratedRunSpec, *, index: int, total: int) -> Path:
        assert self._transport is not None
        if spec.capture_only_duration_s is not None:
            return self._run_capture_only(spec, index=index, total=total)
        if spec.executor_config is None:
            raise IntegratedRigError(
                f"executor_config is not configured for powered experiment {spec.name}"
            )
        executor_document = json.loads(spec.executor_config.read_bytes())
        metadata = executor_document.get("metadata")
        reference_mode = (
            str(metadata.get("angle_guard_reference", "")).strip()
            if isinstance(metadata, dict)
            else ""
        )
        angle_guard_reference = None
        if reference_mode:
            angle_guard_reference = _materialize_run_start_pitch_reference(
                executor_document,
                _probe_uart(self.config, self._transport),
                path=spec.executor_config,
            )
        executor_config_bytes = (
            json.dumps(executor_document, indent=2, ensure_ascii=False) + "\n"
        ).encode("utf-8")
        executor_summary = _inspect_executor_document(
            executor_document,
            path=spec.executor_config,
            expected_uart_device=self.config.pi.uart_device,
            expected_uart_baud=self.config.pi.uart_baud,
            expected_target_system=self.config.px4.expected_system_id,
            expected_target_component=self.config.px4.expected_component_id,
            maximum_motor_command=self.config.safety.maximum_motor_command,
            expected_executor_kind=spec.executor_kind,
        )
        run_id = _unique_run_id(
            spec.name,
            experiment_number=spec.number,
            condition_id=spec.condition_id,
            attempt=spec.attempt,
        )
        session = ExperimentSession.create(
            self.config.session.output_root,
            run_id=run_id,
            protocol_id="px4_fftgyro_integrated_rig",
            protocol_version=self.config.session.protocol_version,
            profile=spec.name,
            operator=self.config.session.operator,
            hardware_config_id=self.config.session.hardware_config_id,
            esc_inventory_id="not_applicable_four_motor_installed",
            fixture_id=self.config.session.fixture_id,
            notes="FFTGYRO encoder + Leptrino + Pi executor + PX4 ULog",
        )
        (session.root / "RUN_IN_PROGRESS").touch(exist_ok=False)
        local_config = session.root / "executor.config.json"
        with local_config.open("xb") as stream:
            stream.write(executor_config_bytes)
            stream.flush()
            os.fsync(stream.fileno())
        shutil.copy2(self.config.path, session.root / "integrated-run.toml")
        if self.config.experiment_source != self.config.path:
            shutil.copy2(
                self.config.experiment_source,
                session.root / "experiment-plan.toml",
            )
        session.append_event(
            "run_preparing",
            {
                **_experiment_metadata(spec),
                "experiment_number": spec.number,
                "experiment_count": total,
                "executor": asdict(executor_summary),
                "ssh_host": self._transport.host,
            },
        )
        if angle_guard_reference is not None:
            session.append_event(
                "run_start_pitch_reference",
                angle_guard_reference,
            )

        acquisition_output = (
            session.raw_source_dir("fftgyro_acquisition") / "acquisition.csv"
        )
        capture_seconds = (
            self.config.acquisition.pre_roll_s
            + executor_summary.expected_duration_s
            + self.config.acquisition.post_roll_s
        )
        remote_temp = f"/tmp/{run_id}.json"
        remote_safe = f"/tmp/{run_id}.safe.json"
        remote_before: set[str] = set()
        final_status = "failed"
        summary: dict[str, Any] = {}
        clock_brackets: list[dict[str, Any]] = []
        remote_test_uploaded = False
        remote_safe_uploaded = False
        restore_unit: str | None = None
        executor_may_be_running = False
        try:
            remote_before = self._remote_log_names()
            self._transport.upload(local_config, remote_temp)
            remote_test_uploaded = True
            self._transport.upload(
                self.config.pi.safe_executor_config,
                remote_safe,
            )
            remote_safe_uploaded = True
            remote_test_sha256 = _sha256_file(local_config)
            remote_safe_sha256 = _sha256_file(self.config.pi.safe_executor_config)
            validate_command = (
                f"test \"$(systemctl is-active "
                f"{shlex.quote(self.config.pi.executor_service)} || true)\" = inactive && "
                f"test \"$(sha256sum {shlex.quote(remote_temp)} | cut -d' ' -f1)\" = "
                f"{shlex.quote(remote_test_sha256)} && "
                f"test \"$(sha256sum {shlex.quote(remote_safe)} | cut -d' ' -f1)\" = "
                f"{shlex.quote(remote_safe_sha256)} && "
                f"sudo -n {shlex.quote(self.config.pi.executor_binary)} validate "
                f"{shlex.quote(remote_temp)}"
            )
            self._transport.run(validate_command, timeout=20.0)
            live_interlock = _validate_live_interlock(
                self.config,
                _probe_uart(self.config, self._transport),
                spec,
            )
            session.append_event("live_run_interlock", live_interlock)
            clock_brackets.append(self._sample_pi_clock("before"))
            session.append_event("pi_clock_bracket", clock_brackets[-1])

            prompt = self._input(
                f"\n[実験 {index}/{total}] {spec.name}: 接続・設定準備完了。"
                f"\n条件: {_experiment_condition_text(spec)}"
                f"\n直前確認: {spec.operator_check}\n"
                "RC Killを解除し、周囲の退避を再確認して "
                f"{self.config.safety.run_confirmation_phrase} と入力: "
            ).strip()
            if prompt != self.config.safety.run_confirmation_phrase:
                raise IntegratedRigError("run-start confirmation did not match")

            executor_pattern = (
                "^"
                + re.escape(self.config.pi.executor_binary)
                + r"([[:space:]]|$)"
            )
            final_remote_interlock = (
                f"test \"$(systemctl is-active "
                f"{shlex.quote(self.config.pi.executor_service)} || true)\" = inactive && "
                f"! pgrep -f {shlex.quote(executor_pattern)} >/dev/null && "
                f"! fuser {shlex.quote(self.config.pi.uart_device)} "
                ">/dev/null 2>&1 && "
                "test -z \"$(systemctl list-units --all "
                "--state=active,activating --no-legend "
                "'fftgyro-config-restore-*.timer')\" && "
                f"test \"$(sha256sum {shlex.quote(remote_temp)} | cut -d' ' -f1)\" = "
                f"{shlex.quote(remote_test_sha256)} && "
                f"test \"$(sha256sum {shlex.quote(remote_safe)} | cut -d' ' -f1)\" = "
                f"{shlex.quote(remote_safe_sha256)}"
            )
            # Complete the retryable, read-only remote guard before starting
            # the fixed-duration acquisition. Otherwise a hotspot reconnect
            # consumes the recording window and can make acquisition finish
            # while an otherwise healthy executor is still winding down.
            self._transport.run(
                final_remote_interlock,
                timeout=10.0,
                transport_retries=3,
            )

            capture = self._start_capture(
                acquisition_output,
                duration_s=capture_seconds,
                run_id=run_id,
            )
            session.append_event(
                "acquisition_ready",
                {"duration_s": capture_seconds, **capture.ready},
            )
            if self.config.acquisition.pre_roll_s:
                time.sleep(self.config.acquisition.pre_roll_s)
            if capture.process.poll() is not None:
                raise IntegratedRigError(
                    "FFTGYRO/Leptrino acquisition stopped during pre-roll; "
                    "no executor start was requested"
                )
            session.append_event(
                "acquisition_preroll_complete",
                {"pre_roll_s": self.config.acquisition.pre_roll_s},
            )
            session.append_event("executor_start_requested", {})
            restore_unit = f"fftgyro-config-restore-{run_id}"[:80]
            restore_delay_s = math.ceil(
                executor_summary.expected_duration_s + 30.0
            )
            watchdog_command = (
                f"systemctl stop {shlex.quote(self.config.pi.executor_service)} && "
                f"test \"$(systemctl is-active "
                f"{shlex.quote(self.config.pi.executor_service)} || true)\" = inactive && "
                f"test \"$(sha256sum {shlex.quote(remote_safe)} | cut -d' ' -f1)\" = "
                f"{shlex.quote(remote_safe_sha256)} && "
                f"/usr/bin/install -o root -g fftgyro -m 0640 "
                f"{shlex.quote(remote_safe)} "
                f"{shlex.quote(self.config.pi.executor_config)}"
            )
            executor_may_be_running = True
            self._transport.run(
                "sudo -n systemctl reset-failed "
                f"{shlex.quote(self.config.pi.executor_service)} || true; "
                "sudo -n systemd-run --quiet "
                f"--unit={shlex.quote(restore_unit)} "
                f"--on-active={restore_delay_s}s "
                f"/bin/sh -c {shlex.quote(watchdog_command)} && "
                f"systemctl is-active --quiet "
                f"{shlex.quote(restore_unit + '.timer')} && "
                f"sudo -n install -o root -g fftgyro -m 0640 "
                f"{shlex.quote(remote_temp)} "
                f"{shlex.quote(self.config.pi.executor_config)} && "
                "sudo -n systemctl start "
                f"{shlex.quote(self.config.pi.executor_service)}",
                timeout=15.0,
            )
            session.append_event("executor_config_installed", {})
            self._wait_for_executor(
                timeout_s=executor_summary.expected_duration_s + 20.0,
                capture=capture,
            )
            executor_may_be_running = False
            session.append_event("executor_stopped", {})

            capture_result = self._wait_capture(
                capture,
                timeout_s=capture_seconds + 20.0,
            )
            _validate_capture_result(
                capture_result,
                run_id,
                clock_mode=self.config.acquisition.clock_mode,
                maximum_sync_delta_us=(
                    self.config.acquisition.maximum_sync_delta_us
                ),
            )
            session.append_event("acquisition_finished", capture_result)
            self._capture = None
            clock_brackets.append(self._sample_pi_clock("after"))
            session.append_event("pi_clock_bracket", clock_brackets[-1])

            remote_after = self._remote_log_names()
            created = sorted(remote_after - remote_before)
            csv_candidates = [
                name
                for name in created
                if name.startswith(executor_summary.run_name + "_")
                and name.endswith(".csv")
            ]
            if len(csv_candidates) != 1:
                raise IntegratedRigError(
                    "expected exactly one new Pi executor CSV, got "
                    + repr(csv_candidates)
                )
            remote_csv_name = csv_candidates[0]
            pi_raw = session.raw_source_dir("pi_executor")
            local_csv = pi_raw / "executor.csv"
            local_remote_config = pi_raw / "executor.config.json"
            self._transport.download(
                f"{self.config.pi.log_directory}/{remote_csv_name}",
                local_csv,
            )
            remote_config_name = remote_csv_name.removesuffix(".csv") + ".config.json"
            self._transport.download(
                f"{self.config.pi.log_directory}/{remote_config_name}",
                local_remote_config,
            )
            if _sha256_file(local_remote_config) != _sha256_file(local_config):
                raise IntegratedRigError(
                    "Pi executor config snapshot does not match the commanded "
                    "configuration"
                )
            _validate_executor_csv(
                local_csv,
                executor_kind=executor_summary.executor_kind,
                executor_axis=executor_summary.axis,
            )
            _validate_acquisition_csv(acquisition_output)

            if self.config.flight_log.enabled:
                session.append_event(
                    "px4_ulog_pending",
                    {
                        "retrieval_mode": self.config.flight_log.retrieval_mode,
                        "retrieve_every_runs": self.config.flight_log.retrieve_every_runs,
                        "retrieve_at_batch_end": self.config.flight_log.retrieve_at_batch_end,
                    },
                )

            source_manifest = {
                "fftgyro_acquisition": _file_manifest(acquisition_output, session.root),
                "pi_executor": _file_manifest(local_csv, session.root),
                "pi_executor_config": _file_manifest(
                    local_remote_config,
                    session.root,
                ),
            }
            _write_json_new(session.root / "source_manifest.json", source_manifest)
            summary = {
                "experiment": _experiment_metadata(spec),
                "executor": asdict(executor_summary),
                "acquisition": capture_result,
                "acquisition_code": {
                    "script": str(self.config.acquisition.script),
                    "script_sha256": _sha256_file(
                        self.config.acquisition.script
                    ),
                    "processing_module": str(
                        self.config.acquisition.processing_module
                    ),
                    "processing_sha256": _sha256_file(
                        self.config.acquisition.processing_module
                    ),
                },
                "source_manifest": source_manifest,
                "clock_alignment": {
                    "method": "ssh_realtime_monotonic_bracket",
                    "brackets": clock_brackets,
                    "hardware_sync_verified": False,
                    "warning": (
                        "PC/Pi software clock brackets are not a substitute for "
                        "a measured common hardware sync edge"
                    ),
                },
                "flight_log_status": (
                    "pending_on_pixhawk_sd"
                    if self.config.flight_log.enabled
                    else "disabled"
                ),
            }
            final_status = "completed"
            return session.root
        except KeyboardInterrupt:
            final_status = "aborted"
            summary = {"reason": "keyboard_interrupt"}
            raise
        except BaseException as error:
            final_status = "failed"
            summary = {
                "reason": f"{type(error).__name__}: {error}",
            }
            session.append_event("run_failed", summary)
            raise
        finally:
            active_error = sys.exc_info()[0] is not None
            stop_failure: BaseException | None = None
            if executor_may_be_running:
                try:
                    self._stop_executor_and_confirm()
                    executor_may_be_running = False
                except BaseException as stop_error:
                    stop_failure = stop_error
                    session.append_event(
                        "executor_stop_unconfirmed",
                        {
                            "error": f"{type(stop_error).__name__}: {stop_error}",
                            "required_action": "RC_KILL_ON",
                        },
                    )
                    self._output(
                        "\n*** EXECUTOR停止未確認: RC KILLを今すぐONにしてください ***"
                    )
            self._stop_capture()
            restore_failure: BaseException | None = None
            try:
                if remote_safe_uploaded and not executor_may_be_running:
                    self._restore_safe_executor_config(
                        remote_safe,
                        remote_temp if remote_test_uploaded else None,
                        restore_unit,
                    )
            except BaseException as restore_error:
                restore_failure = restore_error
                session.append_event(
                    "safe_config_restore_failed",
                    {
                        "error": (
                            f"{type(restore_error).__name__}: {restore_error}"
                        )
                    },
                )
                if final_status == "completed":
                    final_status = "failed"
                    summary = {
                        **summary,
                        "safe_config_restore_error": str(restore_error),
                    }
            reboot_failure: BaseException | None = None
            if (
                final_status == "completed"
                and restore_failure is None
                and stop_failure is None
                and self.config.flight_log.enabled
                and self.config.flight_log.reboot_pixhawk_after_run
            ):
                try:
                    reboot = self._close_px4_ulog_by_reboot(session, spec)
                    summary = {
                        **summary,
                        "flight_log_status": (
                            "closed_on_pixhawk_sd_by_verified_reboot"
                        ),
                        "pixhawk_reboot": reboot,
                    }
                except BaseException as reboot_error:
                    reboot_failure = reboot_error
                    final_status = "failed"
                    summary = {
                        **summary,
                        "flight_log_status": (
                            "data_complete_but_ulog_boundary_unverified"
                        ),
                        "pixhawk_reboot_error": (
                            f"{type(reboot_error).__name__}: {reboot_error}"
                        ),
                    }
                    session.append_event(
                        "pixhawk_reboot_failed",
                        {
                            "error": (
                                f"{type(reboot_error).__name__}: {reboot_error}"
                            ),
                            "next_run_blocked": True,
                        },
                    )
            marker = session.root / "RUN_IN_PROGRESS"
            if marker.exists():
                marker.unlink()
            if not (session.root / "final.json").exists():
                session.finalize(status=final_status, summary=summary)
            if stop_failure is not None and not active_error:
                raise IntegratedRigError(
                    "executor stop/release could not be confirmed; use RC Kill: "
                    f"{stop_failure}"
                ) from stop_failure
            if restore_failure is not None and not active_error:
                raise IntegratedRigError(
                    "test completed but the known-safe Pi config could not be "
                    f"restored: {restore_failure}"
                ) from restore_failure
            if reboot_failure is not None and not active_error:
                raise IntegratedRigError(
                    "test data completed, but the Pixhawk reboot/ULog boundary "
                    f"could not be verified: {reboot_failure}"
                ) from reboot_failure

    def _run_capture_only(
        self,
        spec: IntegratedRunSpec,
        *,
        index: int,
        total: int,
    ) -> Path:
        assert self._transport is not None
        duration_s = spec.capture_only_duration_s
        if duration_s is None:
            raise IntegratedRigError("capture-only duration is not configured")
        run_id = _unique_run_id(
            spec.name,
            experiment_number=spec.number,
            condition_id=spec.condition_id,
            attempt=spec.attempt,
        )
        session = ExperimentSession.create(
            self.config.session.output_root,
            run_id=run_id,
            protocol_id="px4_fftgyro_integrated_rig",
            protocol_version=self.config.session.protocol_version,
            profile=spec.name,
            operator=self.config.session.operator,
            hardware_config_id=self.config.session.hardware_config_id,
            esc_inventory_id="not_applicable_capture_only",
            fixture_id=self.config.session.fixture_id,
            notes=(
                "FFTGYRO encoder + Leptrino acquisition only; "
                "Pi executor inhibited and RC Kill kept ON"
            ),
        )
        (session.root / "RUN_IN_PROGRESS").touch(exist_ok=False)
        shutil.copy2(self.config.path, session.root / "integrated-run.toml")
        if self.config.experiment_source != self.config.path:
            shutil.copy2(
                self.config.experiment_source,
                session.root / "experiment-plan.toml",
            )
        session.append_event(
            "run_preparing",
            {
                **_experiment_metadata(spec),
                "experiment_number": spec.number,
                "experiment_count": total,
                "executor": None,
                "ssh_host": self._transport.host,
                "motor_inhibition": "RC_KILL_ON_AND_EXECUTOR_INACTIVE",
            },
        )

        acquisition_output = (
            session.raw_source_dir("fftgyro_acquisition") / "acquisition.csv"
        )
        final_status = "failed"
        summary: dict[str, Any] = {}
        clock_brackets: list[dict[str, Any]] = []
        try:
            safe_sha256 = _sha256_file(self.config.pi.safe_executor_config)
            executor_pattern = (
                "^"
                + re.escape(self.config.pi.executor_binary)
                + r"([[:space:]]|$)"
            )
            self._transport.run(
                f"test \"$(systemctl is-active "
                f"{shlex.quote(self.config.pi.executor_service)} || true)\" = inactive && "
                f"! pgrep -f {shlex.quote(executor_pattern)} >/dev/null && "
                f"! fuser {shlex.quote(self.config.pi.uart_device)} "
                ">/dev/null 2>&1 && "
                f"test \"$(sha256sum "
                f"{shlex.quote(self.config.pi.executor_config)} "
                "| cut -d' ' -f1)\" = "
                f"{shlex.quote(safe_sha256)}",
                timeout=10.0,
                transport_retries=3,
            )
            live_interlock = _validate_live_interlock(
                self.config,
                _probe_uart(self.config, self._transport),
                spec,
            )
            session.append_event(
                "live_run_interlock",
                {
                    **live_interlock,
                    "executor_state": "inactive",
                    "safe_executor_config_sha256": safe_sha256,
                    "rc_kill_required_state_during_capture": "ON",
                },
            )
            clock_brackets.append(self._sample_pi_clock("before"))
            session.append_event("pi_clock_bracket", clock_brackets[-1])

            prompt = self._input(
                f"\n[計測のみ {index}/{total}] {spec.name}\n"
                f"条件: {_experiment_condition_text(spec)}\n"
                f"直前確認: {spec.operator_check}\n"
                "RC KillをONのまま維持し、モーターが停止していることを確認して "
                f"{self.config.safety.run_confirmation_phrase} と入力: "
            ).strip()
            if prompt != self.config.safety.run_confirmation_phrase:
                raise IntegratedRigError(
                    "capture-only start confirmation did not match"
                )

            capture = self._start_capture(
                acquisition_output,
                duration_s=duration_s,
                run_id=run_id,
            )
            session.append_event(
                "acquisition_ready",
                {
                    "duration_s": duration_s,
                    "capture_only": True,
                    **capture.ready,
                },
            )
            capture_result = self._wait_capture(
                capture,
                timeout_s=duration_s + 20.0,
            )
            _validate_capture_result(
                capture_result,
                run_id,
                clock_mode=self.config.acquisition.clock_mode,
                maximum_sync_delta_us=(
                    self.config.acquisition.maximum_sync_delta_us
                ),
            )
            session.append_event("acquisition_finished", capture_result)
            self._capture = None
            _validate_acquisition_csv(acquisition_output)
            clock_brackets.append(self._sample_pi_clock("after"))
            session.append_event("pi_clock_bracket", clock_brackets[-1])

            if self.config.flight_log.enabled:
                session.append_event(
                    "px4_ulog_pending",
                    {
                        "retrieval_mode": self.config.flight_log.retrieval_mode,
                        "retrieve_every_runs": (
                            self.config.flight_log.retrieve_every_runs
                        ),
                        "retrieve_at_batch_end": (
                            self.config.flight_log.retrieve_at_batch_end
                        ),
                    },
                )

            source_manifest = {
                "fftgyro_acquisition": _file_manifest(
                    acquisition_output,
                    session.root,
                )
            }
            _write_json_new(session.root / "source_manifest.json", source_manifest)
            summary = {
                "experiment": _experiment_metadata(spec),
                "executor": None,
                "motor_inhibition": "RC_KILL_ON_AND_EXECUTOR_INACTIVE",
                "acquisition": capture_result,
                "acquisition_code": {
                    "script": str(self.config.acquisition.script),
                    "script_sha256": _sha256_file(
                        self.config.acquisition.script
                    ),
                    "processing_module": str(
                        self.config.acquisition.processing_module
                    ),
                    "processing_sha256": _sha256_file(
                        self.config.acquisition.processing_module
                    ),
                },
                "source_manifest": source_manifest,
                "clock_alignment": {
                    "method": "ssh_realtime_monotonic_bracket",
                    "brackets": clock_brackets,
                    "hardware_sync_verified": False,
                    "warning": (
                        "PC/Pi software clock brackets are not a substitute for "
                        "a measured common hardware sync edge"
                    ),
                },
                "flight_log_status": (
                    "pending_on_pixhawk_sd"
                    if self.config.flight_log.enabled
                    else "disabled"
                ),
            }
            final_status = "completed"
            return session.root
        except KeyboardInterrupt:
            final_status = "aborted"
            summary = {"reason": "keyboard_interrupt"}
            raise
        except BaseException as error:
            summary = {"reason": f"{type(error).__name__}: {error}"}
            session.append_event("run_failed", summary)
            raise
        finally:
            self._stop_capture()
            reboot_failure: BaseException | None = None
            if (
                final_status == "completed"
                and self.config.flight_log.enabled
                and self.config.flight_log.reboot_pixhawk_after_run
            ):
                try:
                    reboot = self._close_px4_ulog_by_reboot(session, spec)
                    summary = {
                        **summary,
                        "flight_log_status": (
                            "closed_on_pixhawk_sd_by_verified_reboot"
                        ),
                        "pixhawk_reboot": reboot,
                    }
                except BaseException as reboot_error:
                    reboot_failure = reboot_error
                    final_status = "failed"
                    summary = {
                        **summary,
                        "flight_log_status": (
                            "data_complete_but_ulog_boundary_unverified"
                        ),
                        "pixhawk_reboot_error": (
                            f"{type(reboot_error).__name__}: {reboot_error}"
                        ),
                    }
                    session.append_event(
                        "pixhawk_reboot_failed",
                        {
                            "error": (
                                f"{type(reboot_error).__name__}: {reboot_error}"
                            ),
                            "next_run_blocked": True,
                        },
                    )
            marker = session.root / "RUN_IN_PROGRESS"
            if marker.exists():
                marker.unlink()
            if not (session.root / "final.json").exists():
                session.finalize(status=final_status, summary=summary)
            if reboot_failure is not None and sys.exc_info()[0] is None:
                raise IntegratedRigError(
                    "capture completed, but the Pixhawk reboot/ULog boundary "
                    f"could not be verified: {reboot_failure}"
                ) from reboot_failure

    def _start_capture(
        self,
        output_path: Path,
        *,
        duration_s: float,
        run_id: str,
    ) -> "_CaptureProcess":
        output_path.parent.mkdir(parents=True, exist_ok=True)
        args = [
            self.config.acquisition.node_executable,
            str(self.config.acquisition.script),
            "--duration-ms",
            str(round(duration_s * 1000)),
            "--output",
            str(output_path),
            "--run-id",
            run_id,
            "--encoder-port",
            self.config.acquisition.encoder_port,
            "--g-version",
            self.config.acquisition.g_version,
            "--encoder-offset-x",
            str(self.config.acquisition.encoder_offset_x_deg),
            "--encoder-offset-y",
            str(self.config.acquisition.encoder_offset_y_deg),
            "--encoder-offset-z",
            str(self.config.acquisition.encoder_offset_z_deg),
            "--encoder-baud",
            str(self.config.acquisition.encoder_baud),
            "--force-port",
            self.config.acquisition.force_port,
            "--expected-force-product",
            self.config.acquisition.expected_force_product,
            "--expected-force-serial",
            self.config.acquisition.expected_force_serial,
            "--expected-force-frequency",
            str(self.config.acquisition.expected_force_frequency_hz),
            "--expected-force-limit-n",
            str(self.config.acquisition.expected_force_limit_n),
            "--expected-force-limit-nm",
            str(self.config.acquisition.expected_moment_limit_nm),
            "--minimum-encoder-rate-hz",
            str(self.config.acquisition.minimum_encoder_rate_hz),
            "--minimum-force-rate-hz",
            str(self.config.acquisition.minimum_force_rate_hz),
            "--ready-timeout-ms",
            str(round(self.config.acquisition.ready_timeout_s * 1000)),
        ]
        if self.config.acquisition.clock_mode == "pc_receive":
            args.append("--no-sync-time")
        creationflags = (
            subprocess.CREATE_NEW_PROCESS_GROUP
            if os.name == "nt"
            else 0
        )
        process = subprocess.Popen(
            args,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            creationflags=creationflags,
            env=_acquisition_environment(self.config.acquisition),
        )
        self._capture = process
        capture = _CaptureProcess(process)
        try:
            ready = capture.wait_ready(self.config.acquisition.ready_timeout_s)
            _validate_capture_ready(
                ready,
                run_id,
                clock_mode=self.config.acquisition.clock_mode,
                maximum_sync_delta_us=(
                    self.config.acquisition.maximum_sync_delta_us
                ),
            )
            capture.ready = ready
        except BaseException:
            try:
                capture.stop()
            except BaseException:
                pass
            self._capture = None
            raise
        return capture

    def _wait_capture(
        self,
        capture: "_CaptureProcess",
        *,
        timeout_s: float,
    ) -> dict[str, Any]:
        return capture.wait_result(timeout_s)

    def _stop_capture(self) -> None:
        process = self._capture
        self._capture = None
        if process is None or process.poll() is not None:
            return
        try:
            if os.name == "nt" and hasattr(signal, "CTRL_BREAK_EVENT"):
                process.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                process.terminate()
            process.wait(timeout=10.0)
        except (OSError, subprocess.TimeoutExpired) as graceful_error:
            try:
                process.kill()
                process.wait(timeout=5.0)
            except (OSError, subprocess.TimeoutExpired) as force_error:
                self._output(
                    "WARNING: acquisition process could not be confirmed stopped: "
                    f"graceful={graceful_error}; force={force_error}"
                )

    def _wait_for_executor(
        self,
        *,
        timeout_s: float,
        capture: "_CaptureProcess",
    ) -> None:
        assert self._transport is not None
        deadline = time.monotonic() + timeout_s
        observed_active = False
        while time.monotonic() < deadline:
            capture_code = capture.process.poll()
            if capture_code is not None:
                self._stop_executor_and_confirm()
                raise IntegratedRigError(
                    "FFTGYRO/Leptrino acquisition stopped while the executor "
                    f"was running (exit={capture_code}); software stop requested"
                )
            try:
                result = self._transport.run(
                    "systemctl is-active "
                    + shlex.quote(self.config.pi.executor_service)
                    + " || true",
                    timeout=5.0,
                    check=False,
                    transport_retries=3,
                )
            except (IntegratedRigError, OSError, subprocess.TimeoutExpired) as error:
                raise IntegratedRigError(
                    "SSH monitoring was lost while the executor may still be "
                    "running. RC KILLを今すぐONにしてください。"
                ) from error
            state = result.stdout.strip()
            if state in {"active", "activating"}:
                observed_active = True
            elif state == "failed":
                detail = self._transport.run(
                    "journalctl -u "
                    + shlex.quote(self.config.pi.executor_service)
                    + " -n 30 --no-pager",
                    check=False,
                ).stdout
                raise IntegratedRigError("Pi executor failed:\n" + detail)
            elif observed_active and state == "inactive":
                return
            elif not observed_active and state == "inactive":
                # A short run can complete between the start call and first poll.
                result_check = self._transport.run(
                    "systemctl show "
                    + shlex.quote(self.config.pi.executor_service)
                    + " -p Result --value",
                    check=False,
                ).stdout.strip()
                if result_check in {"success", ""}:
                    return
            time.sleep(0.25)
        raise IntegratedRigError("Pi executor did not stop before the bounded timeout")

    def _stop_executor_and_confirm(self) -> None:
        assert self._transport is not None
        result = self._transport.run(
            "sudo -n systemctl stop "
            + shlex.quote(self.config.pi.executor_service)
            + "; sudo -n systemctl reset-failed "
            + shlex.quote(self.config.pi.executor_service)
            + " || true",
            timeout=20.0,
            check=False,
        )
        if result.returncode != 0:
            raise IntegratedRigError(
                "systemctl stop failed: "
                + (result.stderr.strip() or result.stdout.strip())
            )
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            state = self._transport.run(
                "systemctl is-active "
                + shlex.quote(self.config.pi.executor_service)
                + " || true",
                timeout=5.0,
                check=False,
            ).stdout.strip()
            if state == "inactive":
                return
            if state == "failed":
                raise IntegratedRigError("executor entered failed state during stop")
            time.sleep(0.25)
        raise IntegratedRigError("executor did not become inactive after stop")

    def _remote_log_names(self) -> set[str]:
        assert self._transport is not None
        command = (
            "find "
            + shlex.quote(self.config.pi.log_directory)
            + " -maxdepth 1 -type f -printf '%f\\n'"
        )
        last_error: IntegratedRigError | None = None
        for attempt in range(3):
            try:
                result = self._transport.run(
                    command,
                    timeout=10.0,
                    check=True,
                )
                return {
                    line.strip()
                    for line in result.stdout.splitlines()
                    if line.strip()
                }
            except IntegratedRigError as error:
                last_error = error
                if attempt < 2:
                    time.sleep(0.5 * (attempt + 1))
        assert last_error is not None
        raise IntegratedRigError(
            "remote executor log listing failed after 3 attempts: "
            + str(last_error)
        ) from last_error

    def _sample_pi_clock(self, label: str) -> dict[str, Any]:
        assert self._transport is not None
        pc_send_epoch_ns = time.time_ns()
        pc_send_monotonic_ns = time.monotonic_ns()
        result = self._transport.run(
            "python3 -c \"import json,time; "
            "print(json.dumps({'realtime_ns':time.time_ns(),"
            "'monotonic_ns':time.monotonic_ns()}))\"",
            timeout=5.0,
            transport_retries=3,
        )
        pc_receive_monotonic_ns = time.monotonic_ns()
        pc_receive_epoch_ns = time.time_ns()
        remote = _last_json_object(result.stdout)
        midpoint_epoch_ns = (pc_send_epoch_ns + pc_receive_epoch_ns) // 2
        pi_realtime_ns = int(remote["realtime_ns"])
        pi_monotonic_ns = int(remote["monotonic_ns"])
        return {
            "label": label,
            "pc_send_epoch_ns": pc_send_epoch_ns,
            "pc_receive_epoch_ns": pc_receive_epoch_ns,
            "pc_rtt_ns": pc_receive_monotonic_ns - pc_send_monotonic_ns,
            "pi_realtime_ns": pi_realtime_ns,
            "pi_monotonic_ns": pi_monotonic_ns,
            "pc_minus_pi_realtime_ns": midpoint_epoch_ns - pi_realtime_ns,
            "pi_realtime_minus_monotonic_ns": pi_realtime_ns - pi_monotonic_ns,
        }

    def _restore_safe_executor_config(
        self,
        remote_safe: str,
        remote_test: str | None,
        restore_unit: str | None,
    ) -> None:
        assert self._transport is not None
        expected = _sha256_file(self.config.pi.safe_executor_config)
        cleanup = shlex.quote(remote_safe)
        if remote_test is not None:
            cleanup += " " + shlex.quote(remote_test)
        command = (
            f"sudo -n install -o root -g fftgyro -m 0640 "
            f"{shlex.quote(remote_safe)} "
            f"{shlex.quote(self.config.pi.executor_config)} && "
            f"actual=$(sha256sum {shlex.quote(self.config.pi.executor_config)} "
            "| cut -d' ' -f1) && "
            f"test \"$actual\" = {shlex.quote(expected)}"
        )
        self._transport.run(command, timeout=15.0)
        if restore_unit is not None:
            timer_unit = restore_unit + ".timer"
            service_unit = restore_unit + ".service"
            cancel_script = (
                "set -eu; "
                f"systemctl stop {shlex.quote(timer_unit)} "
                f"{shlex.quote(service_unit)} 2>/dev/null || true; "
                f"timer_state=$(systemctl is-active {shlex.quote(timer_unit)} "
                "2>/dev/null || true); "
                f"service_state=$(systemctl is-active {shlex.quote(service_unit)} "
                "2>/dev/null || true); "
                "case \"$timer_state\" in inactive|failed|unknown) ;; *) exit 31;; esac; "
                "case \"$service_state\" in inactive|failed|unknown) ;; *) exit 32;; esac; "
                f"rm -f {cleanup}"
            )
            self._transport.run(
                "sudo -n /bin/sh -c " + shlex.quote(cancel_script),
                timeout=15.0,
            )
        else:
            self._transport.run(f"rm -f {cleanup}", timeout=10.0)

    def _emergency_software_stop(self) -> None:
        self._stop_capture()
        if self._transport is None:
            return
        try:
            result = self._transport.run(
                "sudo -n systemctl stop "
                + shlex.quote(self.config.pi.executor_service),
                timeout=20.0,
                check=False,
            )
            if result.returncode != 0:
                self._output(
                    "\n*** SOFTWARE STOP未確認: RC KILLを今すぐONにしてください ***"
                )
        except BaseException as error:
            self._output(
                "\n*** SSH断または停止失敗: RC KILLを今すぐONにしてください ***"
            )
            self._output(f"stop error: {type(error).__name__}: {error}")

    def _print_preflight(
        self,
        report: PreflightReport,
        *,
        compact: bool = False,
    ) -> None:
        self._output("\nIntegrated rig preflight")
        if compact:
            warnings = [
                check
                for check in report.checks
                if not check.passed and not check.blocking
            ]
            self._output(
                "  READY "
                f"checks={sum(check.passed for check in report.checks)}/"
                f"{len(report.checks)} "
                f"ssh={report.ssh_host} "
                f"voltage={report.voltage_v:.3f}V "
                f"CH8={report.rc_kill_raw}"
            )
            for check in warnings:
                self._output(f"  WARN {check.name}: {check.detail}")
            return
        for check in report.checks:
            marker = "PASS" if check.passed else ("WARN" if not check.blocking else "FAIL")
            self._output(f"  {marker:4} {check.name}: {check.detail}")
        self._output("READY" if report.ready else "BLOCKED")


class _CaptureProcess:
    def __init__(self, process: subprocess.Popen[str]) -> None:
        self.process = process
        self.lines: list[str] = []
        self.messages: queue.Queue[str] = queue.Queue()
        self.ready: dict[str, Any] = {}
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    def _read(self) -> None:
        stream: IO[str] | None = self.process.stdout
        if stream is None:
            return
        for line in stream:
            cleaned = line.rstrip("\r\n")
            self.lines.append(cleaned)
            self.messages.put(cleaned)

    def wait_ready(self, timeout_s: float) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise IntegratedRigError(
                    "integrated acquisition exited before READY:\n"
                    + "\n".join(self.lines[-30:])
                )
            try:
                line = self.messages.get(timeout=min(0.25, deadline - time.monotonic()))
            except queue.Empty:
                continue
            if line.startswith("READY "):
                return _parse_prefixed_json(line, "READY ")
        raise IntegratedRigError(
            f"integrated acquisition did not become ready within {timeout_s:g}s"
        )

    def wait_result(self, timeout_s: float) -> dict[str, Any]:
        try:
            returncode = self.process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired as error:
            self.stop()
            raise IntegratedRigError("integrated acquisition timed out") from error
        self.reader.join(timeout=2.0)
        result_lines = [line for line in self.lines if line.startswith("RESULT ")]
        if returncode != 0:
            raise IntegratedRigError(
                f"integrated acquisition failed ({returncode}):\n"
                + "\n".join(self.lines[-30:])
            )
        if len(result_lines) != 1:
            raise IntegratedRigError(
                "integrated acquisition did not emit exactly one RESULT record"
            )
        return _parse_prefixed_json(result_lines[0], "RESULT ")

    def stop(self) -> None:
        if self.process.poll() is not None:
            return
        try:
            if os.name == "nt" and hasattr(signal, "CTRL_BREAK_EVENT"):
                self.process.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                self.process.terminate()
            self.process.wait(timeout=10.0)
        except (OSError, subprocess.TimeoutExpired):
            try:
                self.process.kill()
                self.process.wait(timeout=5.0)
            except (OSError, subprocess.TimeoutExpired):
                pass


def _local_preflight_checks(
    config: IntegratedRigConfig,
    *,
    run_command: RunCommand,
) -> tuple[PreflightCheck, ...]:
    checks: list[PreflightCheck] = []
    output_root = config.session.output_root
    try:
        output_root.mkdir(parents=True, exist_ok=True)
        probe = output_root / f".fftgyro-write-probe-{os.getpid()}"
        with probe.open("x", encoding="ascii") as stream:
            stream.write("write-test\n")
            stream.flush()
            os.fsync(stream.fileno())
        probe.unlink()
        writable = True
        detail = str(output_root)
    except OSError as error:
        writable = False
        detail = f"{output_root}: {error}"
    checks.append(
        PreflightCheck("local_output_writable", "pass" if writable else "fail", detail, True)
    )

    free_mb = 0
    try:
        free_mb = shutil.disk_usage(output_root).free // (1024 * 1024)
    except OSError:
        pass
    checks.append(
        PreflightCheck(
            "local_disk_space",
            "pass" if free_mb >= config.session.minimum_free_mb else "fail",
            f"free={free_mb}MB required={config.session.minimum_free_mb}MB",
            True,
        )
    )

    repository_root = _find_repository_root(config.path)
    inside_repository = (
        repository_root is not None
        and _is_relative_to(output_root, repository_root)
    )
    checks.append(
        PreflightCheck(
            "raw_data_outside_git",
            "fail" if inside_repository else "pass",
            (
                f"output={output_root}; repo={repository_root}"
                if repository_root is not None
                else f"output={output_root}; repository root not detected"
            ),
            True,
        )
    )

    if os.name == "nt":
        command = [
            "powershell",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            (
                "Get-CimInstance Win32_Process | "
                "Where-Object { $_.ProcessId -ne $PID -and $_.CommandLine -match "
                "'px_uploader|uploader\\.py|px4.*upload' } | "
                "ForEach-Object { $_.ProcessId.ToString() + ':' + $_.CommandLine }"
            ),
        ]
        try:
            result = run_command(
                command,
                text=True,
                capture_output=True,
                timeout=10.0,
                check=False,
            )
            uploader_lines = [
                line.strip()
                for line in result.stdout.splitlines()
                if line.strip() and str(os.getpid()) not in line
            ]
            checks.append(
                PreflightCheck(
                    "windows_no_uploader",
                    "pass" if result.returncode == 0 and not uploader_lines else "fail",
                    "none" if not uploader_lines else " | ".join(uploader_lines),
                    True,
                )
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            checks.append(
                PreflightCheck(
                    "windows_no_uploader",
                    "warn",
                    f"could not inspect processes: {error}",
                    False,
                )
            )
    return tuple(checks)


def _probe_acquisition(
    config: IntegratedRigConfig,
    *,
    run_command: RunCommand,
) -> tuple[PreflightCheck, ...]:
    if not config.acquisition.script.is_file():
        return ()
    if shutil.which(config.acquisition.node_executable) is None:
        return ()
    args = [
        config.acquisition.node_executable,
        str(config.acquisition.script),
        "--probe",
        "--encoder-port",
        config.acquisition.encoder_port,
        "--g-version",
        config.acquisition.g_version,
        "--encoder-offset-x",
        str(config.acquisition.encoder_offset_x_deg),
        "--encoder-offset-y",
        str(config.acquisition.encoder_offset_y_deg),
        "--encoder-offset-z",
        str(config.acquisition.encoder_offset_z_deg),
        "--encoder-baud",
        str(config.acquisition.encoder_baud),
        "--force-port",
        config.acquisition.force_port,
        "--expected-force-product",
        config.acquisition.expected_force_product,
        "--expected-force-serial",
        config.acquisition.expected_force_serial,
        "--expected-force-frequency",
        str(config.acquisition.expected_force_frequency_hz),
        "--expected-force-limit-n",
        str(config.acquisition.expected_force_limit_n),
        "--expected-force-limit-nm",
        str(config.acquisition.expected_moment_limit_nm),
        "--minimum-encoder-rate-hz",
        str(config.acquisition.minimum_encoder_rate_hz),
        "--minimum-force-rate-hz",
        str(config.acquisition.minimum_force_rate_hz),
    ]
    if config.acquisition.clock_mode == "pc_receive":
        args.append("--no-sync-time")
    try:
        completed = run_command(
            args,
            text=True,
            capture_output=True,
            timeout=config.acquisition.ready_timeout_s,
            check=False,
            env=_acquisition_environment(config.acquisition),
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return (
            PreflightCheck("acquisition_devices", "fail", str(error), True),
        )
    if completed.returncode != 0:
        return (
            PreflightCheck(
                "acquisition_devices",
                "fail",
                completed.stderr.strip() or completed.stdout.strip(),
                True,
            ),
        )
    probe_lines = [
        line
        for line in completed.stdout.splitlines()
        if line.startswith("RESULT ")
    ]
    if len(probe_lines) != 1:
        return (
            PreflightCheck(
                "acquisition_devices",
                "fail",
                "probe did not emit exactly one RESULT record",
                True,
            ),
        )
    try:
        result = _parse_prefixed_json(probe_lines[0], "RESULT ")
    except (json.JSONDecodeError, ValueError) as error:
        return (
            PreflightCheck("acquisition_devices", "fail", str(error), True),
        )
    checks: list[PreflightCheck] = []
    clock_sync = result.get("clockSync")
    timestamp_basis = result.get("timestampBasis")
    sync_delta = (
        _optional_float(clock_sync.get("deltaUs"))
        if isinstance(clock_sync, dict)
        else None
    )
    if config.acquisition.clock_mode == "device_sync":
        sync_ok = (
            result.get("status") == "ok"
            and isinstance(clock_sync, dict)
            and bool(clock_sync.get("confirmed"))
            and sync_delta is not None
            and abs(sync_delta) <= config.acquisition.maximum_sync_delta_us
            and isinstance(timestamp_basis, dict)
            and timestamp_basis.get("mode") == "device_sync"
        )
        sync_detail = (
            f"mode=device_sync delta={sync_delta}us allowed="
            f"{config.acquisition.maximum_sync_delta_us}us"
        )
    else:
        sync_ok = (
            result.get("status") == "ok"
            and isinstance(clock_sync, dict)
            and clock_sync.get("requested") is False
            and clock_sync.get("confirmed") is False
            and isinstance(timestamp_basis, dict)
            and timestamp_basis.get("mode") == "pc_receive"
            and timestamp_basis.get("clock")
                == "windows_monotonic_anchored_epoch_us"
            and timestamp_basis.get("hardwareSynchronized") is False
        )
        sync_detail = (
            "mode=pc_receive; common Windows high-resolution receive clock; "
            "not hardware synchronized"
        )
    checks.append(
        PreflightCheck(
            "acquisition_clock_basis",
            "pass" if sync_ok else "fail",
            sync_detail,
            True,
        )
    )
    for key, label in (("encoder", "fftgyro_encoder"), ("force", "leptrino_force")):
        value = result.get(key)
        ready = (
            result.get("status") == "ok"
            and result.get("mode") == "probe"
            and isinstance(value, dict)
            and bool(value.get("port"))
        )
        checks.append(
            PreflightCheck(
                label,
                "pass" if ready else "fail",
                json.dumps(value, ensure_ascii=False, separators=(",", ":")),
                True,
            )
        )
    distinct = (
        isinstance(result.get("encoder"), dict)
        and isinstance(result.get("force"), dict)
        and result["encoder"].get("port") != result["force"].get("port")
    )
    checks.append(
        PreflightCheck(
            "acquisition_ports_distinct",
            "pass" if distinct else "fail",
            (
                f"encoder={result.get('encoder', {}).get('port')} "
                f"force={result.get('force', {}).get('port')}"
            ),
            True,
        )
    )
    return tuple(checks)


def _remote_preflight(
    config: IntegratedRigConfig,
    transport: SshTransport,
) -> dict[str, Any]:
    args = (
        config.pi.executor_binary,
        config.pi.executor_service,
        config.pi.log_directory,
        config.pi.uart_device,
        config.pi.mavlink_python,
        config.pi.executor_config,
        config.pi.direct_executor_binary or "",
        config.pi.offboard_executor_binary or "",
    )
    command = "python3 - " + " ".join(shlex.quote(value) for value in args)
    result = transport.run(
        command,
        input_text=_REMOTE_PREFLIGHT_SCRIPT,
        timeout=15.0,
    )
    return _last_json_object(result.stdout)


def _probe_uart(
    config: IntegratedRigConfig,
    transport: SshTransport,
) -> dict[str, Any]:
    parameters = json.dumps(
        dict(config.px4.required_parameters),
        separators=(",", ":"),
    )
    command = (
        f"{shlex.quote(config.pi.mavlink_python)} - "
        f"{shlex.quote(config.pi.uart_device)} {config.pi.uart_baud} "
        f"{config.px4.heartbeat_stability_s:g} {shlex.quote(parameters)}"
    )
    result = transport.run(
        command,
        input_text=_UART_PROBE_SCRIPT,
        timeout=(
            10.0
            + 2.0 * len(config.px4.required_parameters)
            + config.px4.heartbeat_stability_s
        ),
    )
    return _last_json_object(result.stdout)


def _reboot_pixhawk(
    config: IntegratedRigConfig,
    transport: SshTransport,
) -> dict[str, Any]:
    command = (
        f"{shlex.quote(config.pi.mavlink_python)} - "
        f"{shlex.quote(config.pi.uart_device)} {config.pi.uart_baud} "
        f"{config.flight_log.reboot_timeout_s:g} "
        f"{config.flight_log.reboot_stability_s:g} "
        f"{config.px4.expected_system_id} "
        f"{config.px4.expected_component_id}"
    )
    result = transport.run(
        command,
        input_text=_PIXHAWK_REBOOT_SCRIPT,
        timeout=config.flight_log.reboot_timeout_s + 10.0,
        check=False,
    )
    try:
        reboot = _last_json_object(result.stdout)
    except (ValueError, json.JSONDecodeError) as error:
        raise IntegratedRigError(
            "Pixhawk reboot command produced no verifiable result"
        ) from error
    if not reboot.get("verified"):
        raise IntegratedRigError(
            "Pixhawk reboot was not verified: " + json.dumps(reboot, sort_keys=True)
        )
    return reboot


def _validate_live_interlock(
    config: IntegratedRigConfig,
    telemetry: dict[str, Any],
    spec: IntegratedRunSpec | None = None,
    *,
    require_rc_kill_on: bool = True,
    check_rc_kill: bool = True,
) -> dict[str, Any]:
    failures: list[str] = []
    identity = (
        int(telemetry.get("system", -1)),
        int(telemetry.get("component", -1)),
        int(telemetry.get("autopilot", -1)),
        int(telemetry.get("vehicle_type", -1)),
    )
    expected_identity = (
        config.px4.expected_system_id,
        config.px4.expected_component_id,
        config.px4.expected_autopilot,
        config.px4.expected_vehicle_type,
    )
    if not telemetry.get("heartbeat") or identity != expected_identity:
        failures.append(
            f"Pixhawk identity mismatch: actual={identity}, expected={expected_identity}"
        )
    if not (
        telemetry.get("identity_stable")
        and int(telemetry.get("heartbeat_count", 0)) >= 2
        and telemetry.get("boot_time_monotonic")
        and int(telemetry.get("boot_time_samples", 0)) >= 2
    ):
        failures.append("Pixhawk heartbeat/boot clock was not stable")
    flight_git_prefix = str(
        telemetry.get("flight_git_prefix") or ""
    ).lower()
    if not flight_git_prefix.startswith(
        config.px4.expected_flight_git_prefix
    ):
        failures.append(
            "PX4 firmware base mismatch: "
            f"{flight_git_prefix or 'unavailable'}"
        )
    if config.safety.require_disarmed and telemetry.get("armed"):
        failures.append("Pixhawk became armed")

    voltage_v = _optional_float(telemetry.get("voltage_v"))
    voltage_min_v = _optional_float(telemetry.get("voltage_min_v"))
    voltage_max_v = _optional_float(telemetry.get("voltage_max_v"))
    allowed_voltage_min_v = (
        spec.voltage_min_v
        if spec is not None
        else config.safety.minimum_voltage_v
    )
    allowed_voltage_max_v = (
        spec.voltage_max_v
        if spec is not None
        else config.safety.maximum_voltage_v
    )
    if not (
        voltage_v is not None
        and voltage_min_v is not None
        and voltage_max_v is not None
        and int(telemetry.get("voltage_samples", 0)) >= 2
        and voltage_min_v >= allowed_voltage_min_v
        and voltage_max_v <= allowed_voltage_max_v
    ):
        failures.append(
            "battery range invalid: "
            f"last={voltage_v}, observed={voltage_min_v}..{voltage_max_v}, "
            f"allowed={allowed_voltage_min_v}.."
            f"{allowed_voltage_max_v}V"
        )

    rc_values = telemetry.get("rc_channels")
    rc_kill_raw: int | None = None
    if isinstance(rc_values, list) and len(rc_values) >= config.safety.rc_kill_channel:
        rc_kill_raw = int(rc_values[config.safety.rc_kill_channel - 1])
    rc_available = bool(telemetry.get("rc_received")) and rc_kill_raw is not None
    rc_kill_on = bool(
        rc_available
        and config.safety.rc_kill_on_min_raw
        <= rc_kill_raw
        <= config.safety.rc_kill_on_max_raw
    )
    if check_rc_kill and not rc_available:
        failures.append(
            f"RC signal is unavailable: CH{config.safety.rc_kill_channel}="
            f"{rc_kill_raw}"
        )
    elif check_rc_kill and require_rc_kill_on and not rc_kill_on:
        failures.append(
            f"RC Kill is not confirmed ON: CH{config.safety.rc_kill_channel}="
            f"{rc_kill_raw}, expected={config.safety.rc_kill_on_min_raw}.."
            f"{config.safety.rc_kill_on_max_raw}"
        )

    actual_parameters = telemetry.get("parameters")
    if not isinstance(actual_parameters, dict):
        actual_parameters = {}
    for parameter_name, expected in config.px4.required_parameters:
        actual = _optional_float(actual_parameters.get(parameter_name))
        if actual is None or abs(actual - expected) > 1e-3:
            failures.append(
                f"PX4 parameter drift: {parameter_name}={actual}, expected={expected}"
            )
    if failures:
        raise IntegratedRigError(
            "live run interlock failed; no executor start was requested: "
            + " | ".join(failures)
        )
    return {
        "verified_at_utc": datetime.now(UTC).isoformat(),
        "voltage_v": voltage_v,
        "voltage_min_v": voltage_min_v,
        "voltage_max_v": voltage_max_v,
        "voltage_source": telemetry.get("voltage_source"),
        "allowed_voltage_min_v": allowed_voltage_min_v,
        "allowed_voltage_max_v": allowed_voltage_max_v,
        "current_a": _optional_float(telemetry.get("current_a")),
        "rc_kill_channel": config.safety.rc_kill_channel,
        "rc_kill_raw": rc_kill_raw,
        "rc_kill_state": (
            ("ON" if rc_kill_on else "OFF")
            if check_rc_kill
            else "NOT_CHECKED"
        ),
        "flight_sw_version": telemetry.get("flight_sw_version"),
        "flight_git_prefix": flight_git_prefix,
        "flight_custom_version": telemetry.get("flight_custom_version"),
    }


_REMOTE_PREFLIGHT_SCRIPT = r"""
import getpass
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys

(
    executor,
    service,
    log_dir,
    uart,
    mav_python,
    executor_config,
    direct_executor,
    offboard_executor,
) = sys.argv[1:9]

def state(unit):
    result = subprocess.run(
        ["systemctl", "is-active", unit],
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() or "unknown"

def unit_property(unit, name):
    result = subprocess.run(
        ["systemctl", "show", unit, f"--property={name}", "--value"],
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip()

fuser = shutil.which("fuser")
users = ""
if fuser:
    result = subprocess.run(
        [fuser, uart],
        text=True,
        capture_output=True,
        check=False,
    )
    users = (result.stdout + " " + result.stderr).strip()

sudo = subprocess.run(
    ["sudo", "-n", "systemctl", "is-active", service],
    text=True,
    capture_output=True,
    check=False,
)
sudo_systemd_run = subprocess.run(
    ["sudo", "-n", "systemd-run", "--version"],
    text=True,
    capture_output=True,
    check=False,
)
enabled = subprocess.run(
    ["systemctl", "is-enabled", service],
    text=True,
    capture_output=True,
    check=False,
)
executor_process = subprocess.run(
    ["pgrep", "-af", executor],
    text=True,
    capture_output=True,
    check=False,
)
executor_processes = [
    line for line in executor_process.stdout.splitlines()
    if (
        line.strip()
        and "pgrep -af" not in line
        and not line.lstrip().startswith(str(os.getpid()) + " ")
    )
]

free_mb = 0
try:
    free_mb = shutil.disk_usage(log_dir).free // (1024 * 1024)
except OSError:
    pass

executor_hash = None
try:
    digest = hashlib.sha256()
    with open(executor, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    executor_hash = digest.hexdigest()
except OSError:
    pass

def executable_digest(path):
    if not path:
        return (False, None)
    exists = os.path.isfile(path) and os.access(path, os.X_OK)
    try:
        digest = hashlib.sha256()
        with open(path, "rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return (exists, digest.hexdigest())
    except OSError:
        return (exists, None)

direct_executor_exists, direct_executor_hash = executable_digest(direct_executor)
offboard_executor_exists, offboard_executor_hash = executable_digest(offboard_executor)

executor_config_hash = None
try:
    digest = hashlib.sha256()
    with open(executor_config, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    executor_config_hash = digest.hexdigest()
except OSError:
    pass

uptime_s = 0.0
try:
    with open("/proc/uptime", "r", encoding="ascii") as stream:
        uptime_s = float(stream.read().split()[0])
except (OSError, ValueError, IndexError):
    pass

root_options = ""
findmnt = shutil.which("findmnt")
if findmnt:
    mounted = subprocess.run(
        [findmnt, "-n", "-o", "OPTIONS", "/"],
        text=True,
        capture_output=True,
        check=False,
    )
    root_options = mounted.stdout.strip()

throttled_flags = 0
vcgencmd = shutil.which("vcgencmd")
if vcgencmd:
    throttled = subprocess.run(
        [vcgencmd, "get_throttled"],
        text=True,
        capture_output=True,
        check=False,
    ).stdout.strip()
    if "=" in throttled:
        try:
            throttled_flags = int(throttled.split("=", 1)[1], 0)
        except ValueError:
            throttled_flags = -1

uploader = subprocess.run(
    ["pgrep", "-af", "px_uploader|uploader\\.py"],
    text=True,
    capture_output=True,
    check=False,
)
uploader_processes = [
    line for line in uploader.stdout.splitlines()
    if line.strip() and "pgrep -af" not in line
]

restore_timer_result = subprocess.run(
    [
        "systemctl",
        "list-units",
        "--all",
        "--state=active,activating",
        "--no-legend",
        "fftgyro-config-restore-*.timer",
    ],
    text=True,
    capture_output=True,
    check=False,
)
active_restore_timers = [
    line.strip()
    for line in restore_timer_result.stdout.splitlines()
    if line.strip()
]

filesystem_errors = []
journalctl = shutil.which("journalctl")
if journalctl:
    journal = subprocess.run(
        [journalctl, "-b", "-p", "err", "--no-pager", "-n", "200"],
        text=True,
        capture_output=True,
        check=False,
    )
    needles = ("ext4-fs error", "i/o error", "read-only file system", "mmc")
    filesystem_errors = [
        line.strip()
        for line in journal.stdout.splitlines()
        if any(needle in line.lower() for needle in needles)
    ][:10]

print(json.dumps({
    "hostname": socket.gethostname(),
    "user": getpass.getuser(),
    "executor_exists": os.path.isfile(executor) and os.access(executor, os.X_OK),
    "executor_sha256": executor_hash,
    "direct_executor_exists": direct_executor_exists,
    "direct_executor_sha256": direct_executor_hash,
    "offboard_executor_exists": offboard_executor_exists,
    "offboard_executor_sha256": offboard_executor_hash,
    "executor_config_sha256": executor_config_hash,
    "mavlink_python_exists": os.path.isfile(mav_python) and os.access(mav_python, os.X_OK),
    "uart_exists": os.path.exists(uart),
    "uart_realpath": os.path.realpath(uart),
    "uart_users": users,
    "fuser_available": bool(fuser),
    "service_state": state(service),
    "service_enabled_state": enabled.stdout.strip() or "unknown",
    "service_user": unit_property(service, "User"),
    "service_exec_start": unit_property(service, "ExecStart"),
    "service_exec_start_pre": unit_property(service, "ExecStartPre"),
    "service_restart": unit_property(service, "Restart"),
    "executor_processes": executor_processes,
    "active_restore_timers": active_restore_timers,
    "serial_getty_state": state("serial-getty@serial0.service"),
    "sudo_systemctl_ok": sudo.returncode in (0, 3),
    "sudo_systemd_run_ok": sudo_systemd_run.returncode == 0,
    "free_mb": free_mb,
    "uptime_s": uptime_s,
    "root_options": root_options,
    "rootfs_writable": "rw" in root_options.split(","),
    "vcgencmd_available": bool(vcgencmd),
    "throttled_flags": throttled_flags,
    "uploader_processes": uploader_processes,
    "filesystem_errors": filesystem_errors,
}))
"""


_PX4_PARAM_DECODER_SCRIPT = r"""
def decode_parameter_value(message):
    param_type = int(message.param_type)
    raw_float = float(message.param_value)
    if param_type == 9:  # MAV_PARAM_TYPE_REAL32
        return raw_float
    packed = struct.pack("<f", raw_float)
    formats = {
        1: "<B",  # MAV_PARAM_TYPE_UINT8
        2: "<b",  # MAV_PARAM_TYPE_INT8
        3: "<H",  # MAV_PARAM_TYPE_UINT16
        4: "<h",  # MAV_PARAM_TYPE_INT16
        5: "<I",  # MAV_PARAM_TYPE_UINT32
        6: "<i",  # MAV_PARAM_TYPE_INT32
    }
    format_string = formats.get(param_type)
    if format_string is None:
        raise RuntimeError(f"unsupported MAV_PARAM_TYPE: {param_type}")
    size = struct.calcsize(format_string)
    return float(struct.unpack(format_string, packed[:size])[0])
"""


_UART_PROBE_SCRIPT = r"""
import json
import math
import struct
import sys
import time
from pymavlink import mavutil
""" + _PX4_PARAM_DECODER_SCRIPT + r"""
device = sys.argv[1]
baud = int(sys.argv[2])
stability_s = float(sys.argv[3])
required_parameters = json.loads(sys.argv[4])
link = mavutil.mavlink_connection(
    device,
    baud=baud,
    source_system=250,
    source_component=191,
)
heartbeat = link.wait_heartbeat(timeout=5)
if heartbeat is None:
    raise SystemExit("heartbeat timeout")
source_system = heartbeat.get_srcSystem()
source_component = heartbeat.get_srcComponent()

for message_id, interval_us in (
    (mavutil.mavlink.MAVLINK_MSG_ID_SYS_STATUS, 200000),
    (mavutil.mavlink.MAVLINK_MSG_ID_BATTERY_STATUS, 200000),
    (mavutil.mavlink.MAVLINK_MSG_ID_RC_CHANNELS, 200000),
    (mavutil.mavlink.MAVLINK_MSG_ID_ATTITUDE, 200000),
):
    link.mav.command_long_send(
        source_system,
        source_component,
        mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
        0,
        message_id,
        interval_us,
        0, 0, 0, 0, 0,
    )

parameters = {}
for parameter_name in required_parameters:
    encoded_name = parameter_name.encode("ascii")
    link.mav.param_request_read_send(
        source_system,
        source_component,
        encoded_name,
        -1,
    )
    parameter_deadline = time.monotonic() + 2
    while time.monotonic() < parameter_deadline:
        message = link.recv_match(type="PARAM_VALUE", blocking=True, timeout=0.25)
        if message is None:
            continue
        if (
            message.get_srcSystem() != source_system
            or message.get_srcComponent() != source_component
        ):
            continue
        received_name = message.param_id
        if isinstance(received_name, bytes):
            received_name = received_name.decode("ascii", errors="replace")
        received_name = str(received_name).rstrip("\x00")
        if received_name == parameter_name:
            parameters[parameter_name] = decode_parameter_value(message)
            break

link.mav.command_long_send(
    source_system,
    source_component,
    mavutil.mavlink.MAV_CMD_REQUEST_MESSAGE,
    0,
    mavutil.mavlink.MAVLINK_MSG_ID_AUTOPILOT_VERSION,
    0, 0, 0, 0, 0, 0,
)

sys_status_voltages = []
battery_status_voltages = []
currents = []
rc_channels = None
rc_rssi = None
flight_sw_version = None
flight_custom_version = None
flight_git_prefix = None
heartbeat_count = 1
identity_stable = True
armed_seen = bool(
    heartbeat.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
)
boot_times_ms = []
attitude_pitch_deg = []
attitude_pitch_rate_dps = []
deadline = time.monotonic() + stability_s
while time.monotonic() < deadline:
    message = link.recv_match(
        type=[
            "HEARTBEAT",
            "SYS_STATUS",
            "BATTERY_STATUS",
            "RC_CHANNELS",
            "ATTITUDE",
            "AUTOPILOT_VERSION",
        ],
        blocking=True,
        timeout=0.5,
    )
    if message is None:
        continue
    if (
        message.get_srcSystem() != source_system
        or message.get_srcComponent() != source_component
    ):
        continue
    if message.get_type() == "HEARTBEAT":
        heartbeat_count += 1
        armed_seen = armed_seen or bool(
            message.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
        )
        identity_stable = identity_stable and (
            message.get_srcSystem() == source_system
            and message.get_srcComponent() == source_component
            and int(message.autopilot) == int(heartbeat.autopilot)
            and int(message.type) == int(heartbeat.type)
        )
    elif message.get_type() == "SYS_STATUS":
        if message.voltage_battery not in (0, 65535):
            sys_status_voltages.append(message.voltage_battery / 1000.0)
        if message.current_battery != -1:
            currents.append(message.current_battery / 100.0)
    elif message.get_type() == "BATTERY_STATUS":
        values = [value for value in message.voltages if value not in (0, 65535)]
        if values:
            battery_status_voltages.append(sum(values) / 1000.0)
        if message.current_battery != -1:
            currents.append(message.current_battery / 100.0)
    elif message.get_type() == "RC_CHANNELS":
        rc_channels = [
            int(getattr(message, f"chan{index}_raw"))
            for index in range(1, 19)
        ]
        rc_rssi = int(message.rssi)
    elif message.get_type() == "ATTITUDE":
        boot_times_ms.append(int(message.time_boot_ms))
        attitude_pitch_deg.append(math.degrees(float(message.pitch)))
        attitude_pitch_rate_dps.append(math.degrees(float(message.pitchspeed)))
    elif message.get_type() == "AUTOPILOT_VERSION":
        flight_sw_version = int(message.flight_sw_version)
        custom_version_bytes = bytes(message.flight_custom_version)
        flight_custom_version = custom_version_bytes.hex()
        # PX4 stores a masked uint64 Git version in little-endian order and
        # overwrites bytes 0..2 with vendor version bytes. Bytes 3..7 therefore
        # preserve the first five Git SHA bytes in reverse order.
        flight_git_prefix = custom_version_bytes[3:8][::-1].hex()

selected_voltages = (
    sys_status_voltages
    if sys_status_voltages
    else battery_status_voltages
)
voltage_source = (
    "SYS_STATUS"
    if sys_status_voltages
    else "BATTERY_STATUS_CELL_SUM"
    if battery_status_voltages
    else None
)

print(json.dumps({
    "heartbeat": True,
    "heartbeat_count": heartbeat_count,
    "identity_stable": identity_stable,
    "system": source_system,
    "component": source_component,
    "autopilot": int(heartbeat.autopilot),
    "vehicle_type": int(heartbeat.type),
    "armed": armed_seen,
    "flight_sw_version": flight_sw_version,
    "flight_custom_version": flight_custom_version,
    "flight_git_prefix": flight_git_prefix,
    "voltage_source": voltage_source,
    "voltage_v": selected_voltages[-1] if selected_voltages else None,
    "voltage_min_v": min(selected_voltages) if selected_voltages else None,
    "voltage_max_v": max(selected_voltages) if selected_voltages else None,
    "voltage_samples": len(selected_voltages),
    "current_a": currents[-1] if currents else None,
    "current_samples": len(currents),
    "rc_received": rc_channels is not None,
    "rc_rssi": rc_rssi,
    "rc_channels": rc_channels,
    "boot_time_samples": len(boot_times_ms),
    "boot_time_monotonic": all(
        later >= earlier
        for earlier, later in zip(boot_times_ms, boot_times_ms[1:])
    ),
    "pitch_deg": attitude_pitch_deg[-1] if attitude_pitch_deg else None,
    "pitch_mean_deg": (
        sum(attitude_pitch_deg) / len(attitude_pitch_deg)
        if attitude_pitch_deg else None
    ),
    "pitch_min_deg": min(attitude_pitch_deg) if attitude_pitch_deg else None,
    "pitch_max_deg": max(attitude_pitch_deg) if attitude_pitch_deg else None,
    "pitch_rate_dps": (
        attitude_pitch_rate_dps[-1] if attitude_pitch_rate_dps else None
    ),
    "pitch_rate_abs_max_dps": (
        max(abs(value) for value in attitude_pitch_rate_dps)
        if attitude_pitch_rate_dps else None
    ),
    "parameters": parameters,
}))
link.close()
"""


_PIXHAWK_REBOOT_SCRIPT = r"""
import json
import sys
import time
from pymavlink import mavutil

device = sys.argv[1]
baud = int(sys.argv[2])
timeout_s = float(sys.argv[3])
stability_s = float(sys.argv[4])
expected_system = int(sys.argv[5])
expected_component = int(sys.argv[6])

link = mavutil.mavlink_connection(
    device,
    baud=baud,
    source_system=250,
    source_component=191,
)
heartbeat = link.wait_heartbeat(timeout=5)
if heartbeat is None:
    raise SystemExit("heartbeat timeout before Pixhawk reboot")
source_system = heartbeat.get_srcSystem()
source_component = heartbeat.get_srcComponent()
if (source_system, source_component) != (expected_system, expected_component):
    raise SystemExit(
        "Pixhawk identity mismatch before reboot: "
        + repr((source_system, source_component))
    )

def request_attitude():
    link.mav.command_long_send(
        source_system,
        source_component,
        mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
        0,
        mavutil.mavlink.MAVLINK_MSG_ID_ATTITUDE,
        100000,
        0, 0, 0, 0, 0,
    )

request_attitude()
old_boot_ms = None
sample_deadline = time.monotonic() + 3.0
while time.monotonic() < sample_deadline:
    message = link.recv_match(type="ATTITUDE", blocking=True, timeout=0.25)
    if message is None:
        continue
    if (
        message.get_srcSystem() == source_system
        and message.get_srcComponent() == source_component
    ):
        old_boot_ms = int(message.time_boot_ms)
        if old_boot_ms > 1000:
            break
if old_boot_ms is None:
    raise SystemExit("ATTITUDE boot clock unavailable before Pixhawk reboot")

link.mav.command_long_send(
    source_system,
    source_component,
    mavutil.mavlink.MAV_CMD_PREFLIGHT_REBOOT_SHUTDOWN,
    0,
    1,
    0, 0, 0, 0, 0, 0,
)

ack_result = None
ack_deadline = time.monotonic() + 4.0
last_heartbeat_at = time.monotonic()
while time.monotonic() < ack_deadline:
    message = link.recv_match(
        type=["COMMAND_ACK", "HEARTBEAT"],
        blocking=True,
        timeout=0.2,
    )
    if message is None:
        continue
    if (
        message.get_srcSystem() != source_system
        or message.get_srcComponent() != source_component
    ):
        continue
    if message.get_type() == "HEARTBEAT":
        last_heartbeat_at = time.monotonic()
    elif (
        int(message.command)
        == mavutil.mavlink.MAV_CMD_PREFLIGHT_REBOOT_SHUTDOWN
    ):
        ack_result = int(message.result)
        break

accepted_results = {
    mavutil.mavlink.MAV_RESULT_ACCEPTED,
    mavutil.mavlink.MAV_RESULT_IN_PROGRESS,
}
if ack_result not in accepted_results:
    print(json.dumps({
        "verified": False,
        "command_ack_result": ack_result,
        "old_boot_ms": old_boot_ms,
        "reason": "reboot command was not accepted",
    }))
    link.close()
    raise SystemExit(2)

deadline = time.monotonic() + timeout_s
gap_observed = False
reset_observed = False
new_boot_times_ms = []
recovered_heartbeats = 0
attitude_requested_after_recovery = False
while time.monotonic() < deadline:
    message = link.recv_match(
        type=["HEARTBEAT", "ATTITUDE"],
        blocking=True,
        timeout=0.25,
    )
    now = time.monotonic()
    if now - last_heartbeat_at >= 1.5:
        gap_observed = True
    if message is None:
        continue
    if (
        message.get_srcSystem() != source_system
        or message.get_srcComponent() != source_component
    ):
        continue
    if message.get_type() == "HEARTBEAT":
        last_heartbeat_at = now
        if reset_observed:
            recovered_heartbeats += 1
        if gap_observed and not attitude_requested_after_recovery:
            request_attitude()
            attitude_requested_after_recovery = True
    elif message.get_type() == "ATTITUDE":
        boot_ms = int(message.time_boot_ms)
        if boot_ms + 1000 < old_boot_ms:
            reset_observed = True
            new_boot_times_ms.append(boot_ms)
        elif reset_observed:
            new_boot_times_ms.append(boot_ms)
    if (
        reset_observed
        and recovered_heartbeats >= 2
        and len(new_boot_times_ms) >= 2
        and new_boot_times_ms[-1] >= new_boot_times_ms[-2]
        and (new_boot_times_ms[-1] - new_boot_times_ms[0])
        >= round(stability_s * 1000)
    ):
        break

verified = (
    reset_observed
    and recovered_heartbeats >= 2
    and len(new_boot_times_ms) >= 2
    and all(
        later >= earlier
        for earlier, later in zip(new_boot_times_ms, new_boot_times_ms[1:])
    )
)
result = {
    "verified": verified,
    "command": "MAV_CMD_PREFLIGHT_REBOOT_SHUTDOWN",
    "command_ack_result": ack_result,
    "old_boot_ms": old_boot_ms,
    "new_boot_ms": new_boot_times_ms[-1] if new_boot_times_ms else None,
    "heartbeat_gap_observed": gap_observed,
    "recovered_heartbeats": recovered_heartbeats,
    "stability_s": stability_s,
}
print(json.dumps(result))
link.close()
if not verified:
    raise SystemExit(3)
"""


def _ssh_base_args(config: SshConfig) -> list[str]:
    args = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectionAttempts=1",
        "-o",
        f"ConnectTimeout={max(1, round(config.connect_timeout_s))}",
        "-o",
        (
            "StrictHostKeyChecking=accept-new"
            if config.accept_new_host_keys
            else "StrictHostKeyChecking=yes"
        ),
    ]
    if config.identity_file is not None:
        args.extend(["-i", str(config.identity_file)])
    return args


def _acquisition_environment(config: AcquisitionConfig) -> dict[str, str]:
    environment = dict(os.environ)
    if config.electron_as_node:
        environment["ELECTRON_RUN_AS_NODE"] = "1"
    return environment


def _scp_base_args(config: SshConfig) -> list[str]:
    args = [
        "scp",
        "-q",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectionAttempts=1",
        "-o",
        f"ConnectTimeout={max(1, round(config.connect_timeout_s))}",
        "-o",
        (
            "StrictHostKeyChecking=accept-new"
            if config.accept_new_host_keys
            else "StrictHostKeyChecking=yes"
        ),
    ]
    if config.identity_file is not None:
        args.extend(["-i", str(config.identity_file)])
    return args


def _validate_integrated_config(config: IntegratedRigConfig) -> None:
    if not config.session.protocol_version:
        raise ValueError("session.protocol_version must not be empty")
    if config.safety.minimum_voltage_v >= config.safety.maximum_voltage_v:
        raise ValueError("minimum voltage must be less than maximum voltage")
    if not 1 <= config.safety.rc_kill_channel <= 18:
        raise ValueError("safety.rc_kill_channel must be 1..18")
    if not (
        800
        <= config.safety.rc_kill_on_min_raw
        < config.safety.rc_kill_on_max_raw
        <= 2200
    ):
        raise ValueError(
            "safety RC Kill ON raw range must satisfy "
            "800 <= min < max <= 2200"
        )
    if not config.safety.confirmation_phrase.strip():
        raise ValueError("safety.confirmation_phrase must not be empty")
    if not config.safety.run_confirmation_phrase.strip():
        raise ValueError("safety.run_confirmation_phrase must not be empty")
    if config.flight_log.retrieval_mode not in {"deferred", "uart"}:
        raise ValueError("flight_log.retrieval_mode must be deferred or uart")
    if config.flight_log.retrieval_mode == "uart":
        raise ValueError(
            "flight_log.retrieval_mode=uart is not enabled until closed-ULog "
            "selection and transfer are verified; use deferred"
        )
    if config.ssh.identity_file is not None and not config.ssh.identity_file.is_file():
        raise ValueError(f"SSH identity file does not exist: {config.ssh.identity_file}")
    if not config.pi.safe_executor_config.is_file():
        raise ValueError(
            f"safe executor config does not exist: {config.pi.safe_executor_config}"
        )
    if not config.acquisition.script.is_file():
        raise ValueError(f"acquisition script does not exist: {config.acquisition.script}")
    if not config.acquisition.processing_module.is_file():
        raise ValueError(
            "acquisition processing module does not exist: "
            f"{config.acquisition.processing_module}"
        )
    for spec in config.experiments:
        if (
            spec.enabled
            and spec.executor_config is not None
            and not spec.executor_config.is_file()
        ):
            raise ValueError(f"executor config does not exist: {spec.executor_config}")


def _validate_executor_csv(
    path: Path,
    *,
    executor_kind: str = "direct_motor",
    executor_axis: str | None = None,
) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise IntegratedRigError(f"Pi executor CSV is missing or empty: {path}")
    with path.open("r", encoding="utf-8") as stream:
        first = stream.readline().strip()
        reader = csv.DictReader(stream)
        header = reader.fieldnames or []
        rows = [row for row in reader]
    direct_required = {
        "sequence",
        "scheduled_ns",
        "sent_ns",
        "requested_motor1",
        "requested_motor2",
        "requested_motor3",
        "requested_motor4",
        "ack_status",
        "action",
    }
    offboard_required = {
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
    }
    if executor_axis == "YAW":
        offboard_required.update(
            {
                "target_yaw_deg",
                "target_yaw_rate_dps",
                "measured_yaw_deg",
                "measured_yaw_rate_dps",
            }
        )
    required = (
        offboard_required if executor_kind == "offboard" else direct_required
    )
    if "result=COMPLETE" not in first:
        raise IntegratedRigError(f"Pi executor did not report COMPLETE: {first}")
    missing = sorted(required - set(header))
    if missing:
        raise IntegratedRigError(f"Pi executor CSV columns missing: {missing}")
    if len(rows) < 2:
        raise IntegratedRigError("Pi executor CSV contains too few data rows")
    sequences = [int(row["sequence"]) for row in rows]
    expected_sequences = list(range(sequences[0], sequences[0] + len(sequences)))
    if sequences != expected_sequences:
        raise IntegratedRigError("Pi executor sequence is not contiguous")
    last = rows[-1]
    if executor_kind == "offboard":
        if last.get("stage") not in {"ZERO_THRUST", "DISARM_WAIT"}:
            raise IntegratedRigError(
                "Offboard executor CSV does not end in a zero-thrust stage"
            )
        thrust = float(last["thrust"])
        if not math.isfinite(thrust) or abs(thrust) > 1e-9:
            raise IntegratedRigError(
                f"Offboard executor final thrust is not zero: {thrust}"
            )
        if int(last["armed"]) != 0:
            raise IntegratedRigError("Offboard executor final sample is still armed")
        return
    if last.get("action") != "RELEASE":
        raise IntegratedRigError("Pi executor CSV does not end with RELEASE")
    for motor in range(1, 5):
        value = float(last[f"requested_motor{motor}"])
        if not math.isfinite(value) or abs(value) > 1e-9:
            raise IntegratedRigError(
                f"Pi executor final motor{motor} request is not zero: {value}"
            )
    if int(last["ack_status"]) != 1:
        raise IntegratedRigError(
            f"Pi executor RELEASE was not acknowledged: {last['ack_status']}"
        )


def _validate_acquisition_csv(path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise IntegratedRigError(f"acquisition CSV is missing or empty: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        header = reader.fieldnames or []
        rows = list(reader)
    required_labels = {
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
    }
    missing = sorted(required_labels - set(header))
    if missing:
        raise IntegratedRigError(f"acquisition CSV columns missing: {missing}")
    if len(rows) < 20:
        raise IntegratedRigError("acquisition CSV contains too few rows")
    trigger_counts = {
        "Encoder/ESP": 0,
        "Sensor de Fuerza": 0,
    }
    timestamps: list[int] = []
    for row in rows:
        trigger = row["Origen trigger"]
        if trigger in trigger_counts:
            trigger_counts[trigger] += 1
        timestamp = int(row["Trigger PC us"])
        if timestamp <= 0:
            raise IntegratedRigError("acquisition contains a non-positive timestamp")
        timestamps.append(timestamp)
        if trigger == "Encoder/ESP":
            for label in ("Enc Roll deg", "Enc Pitch deg", "Enc Yaw deg"):
                value = float(row[label])
                if not math.isfinite(value):
                    raise IntegratedRigError(
                        f"acquisition contains non-finite {label}: {row[label]}"
                    )
        elif trigger == "Sensor de Fuerza":
            for label in (
                "Fuerza Fx N",
                "Fuerza Fy N",
                "Fuerza Fz N",
                "Fuerza Mx Nm",
                "Fuerza My Nm",
                "Fuerza Mz Nm",
            ):
                value = float(row[label])
                if not math.isfinite(value):
                    raise IntegratedRigError(
                        f"acquisition contains non-finite {label}: {row[label]}"
                    )
    if any(count < 10 for count in trigger_counts.values()):
        raise IntegratedRigError(
            f"acquisition source rows are incomplete: {trigger_counts}"
        )
    if timestamps != sorted(timestamps):
        raise IntegratedRigError("acquisition trigger timestamps are not monotonic")


def _validate_capture_ready(
    ready: dict[str, Any],
    run_id: str,
    *,
    maximum_sync_delta_us: int,
    clock_mode: str = "device_sync",
) -> None:
    if ready.get("runId") != run_id:
        raise IntegratedRigError(
            f"acquisition READY run ID mismatch: {ready.get('runId')} != {run_id}"
        )
    encoder = ready.get("encoder")
    force = ready.get("force")
    if not isinstance(encoder, dict) or not encoder.get("port"):
        raise IntegratedRigError("acquisition READY is missing the FFTGYRO port")
    if not isinstance(force, dict) or not force.get("port"):
        raise IntegratedRigError("acquisition READY is missing the Leptrino port")
    if str(encoder["port"]).casefold() == str(force["port"]).casefold():
        raise IntegratedRigError(
            "acquisition READY resolved FFTGYRO and Leptrino to the same port"
        )
    clock_sync = ready.get("clockSync")
    sync_delta = (
        _optional_float(clock_sync.get("deltaUs"))
        if isinstance(clock_sync, dict)
        else None
    )
    timestamp_basis = ready.get("timestampBasis")
    if clock_mode == "device_sync":
        if (
            not isinstance(clock_sync, dict)
            or not clock_sync.get("confirmed")
            or sync_delta is None
            or abs(sync_delta) > maximum_sync_delta_us
            or not isinstance(timestamp_basis, dict)
            or timestamp_basis.get("mode") != "device_sync"
        ):
            raise IntegratedRigError(
                "acquisition READY device clock sync is invalid: "
                f"delta={sync_delta}us, allowed={maximum_sync_delta_us}us"
            )
    elif (
        not isinstance(clock_sync, dict)
        or clock_sync.get("requested") is not False
        or clock_sync.get("confirmed") is not False
        or not isinstance(timestamp_basis, dict)
        or timestamp_basis.get("mode") != "pc_receive"
        or timestamp_basis.get("clock")
            != "windows_monotonic_anchored_epoch_us"
        or timestamp_basis.get("hardwareSynchronized") is not False
    ):
        raise IntegratedRigError(
            "acquisition READY PC receive-clock declaration is invalid"
        )


def _validate_capture_result(
    result: dict[str, Any],
    run_id: str,
    *,
    maximum_sync_delta_us: int,
    clock_mode: str = "device_sync",
) -> None:
    if result.get("status") != "ok":
        raise IntegratedRigError(
            "integrated acquisition result is not ok: "
            + json.dumps(result, ensure_ascii=False)
        )
    if result.get("runId") != run_id:
        raise IntegratedRigError(
            f"acquisition run ID mismatch: {result.get('runId')} != {run_id}"
        )
    clock_sync = result.get("clockSync")
    timestamp_basis = result.get("timestampBasis")
    if clock_mode == "device_sync":
        if not isinstance(clock_sync, dict) or not clock_sync.get("confirmed"):
            raise IntegratedRigError(
                "FFTGYRO device clock synchronization was not confirmed"
            )
        sync_delta = _optional_float(clock_sync.get("deltaUs"))
        if sync_delta is None or abs(sync_delta) > maximum_sync_delta_us:
            raise IntegratedRigError(
                "FFTGYRO clock synchronization delta is outside the allowed "
                f"range: {sync_delta}us > {maximum_sync_delta_us}us"
            )
        if (
            not isinstance(timestamp_basis, dict)
            or timestamp_basis.get("mode") != "device_sync"
        ):
            raise IntegratedRigError(
                "acquisition result is missing the device-sync time basis"
            )
    elif (
        not isinstance(clock_sync, dict)
        or clock_sync.get("requested") is not False
        or clock_sync.get("confirmed") is not False
        or not isinstance(timestamp_basis, dict)
        or timestamp_basis.get("mode") != "pc_receive"
        or timestamp_basis.get("clock")
            != "windows_monotonic_anchored_epoch_us"
        or timestamp_basis.get("hardwareSynchronized") is not False
    ):
        raise IntegratedRigError(
            "acquisition result is missing the PC receive-clock time basis"
        )
    for source in ("encoder", "force"):
        source_result = result.get(source)
        if not isinstance(source_result, dict):
            raise IntegratedRigError(f"acquisition result is missing {source}")
        rate = source_result.get("rate")
        if not isinstance(rate, dict) or int(rate.get("count", 0)) < 10:
            raise IntegratedRigError(f"acquisition result has too few {source} samples")


def _file_manifest(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(root)),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _unique_run_id(
    name: str,
    *,
    experiment_number: int | None = None,
    condition_id: str | None = None,
    attempt: int = 1,
) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    condition = condition_id or (
        f"E{experiment_number:03d}" if experiment_number is not None else ""
    )
    prefix = f"{condition}_A{attempt:02d}_" if condition else ""
    return f"{prefix}{stamp}_{name}"[:64]


def _write_json_new(path: Path, document: dict[str, Any]) -> None:
    encoded = json.dumps(document, ensure_ascii=False, indent=2) + "\n"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def _last_json_object(text: str) -> dict[str, Any]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        raise ValueError("command returned no JSON")
    value = json.loads(lines[-1])
    if not isinstance(value, dict):
        raise ValueError("command JSON result is not an object")
    return value


def _matches_physical_safety_confirmation(text: str, expected: str) -> bool:
    return (
        unicodedata.normalize("NFKC", text).strip()
        == unicodedata.normalize("NFKC", expected).strip()
    )


def _parse_prefixed_json(line: str, prefix: str) -> dict[str, Any]:
    if not line.startswith(prefix):
        raise ValueError(f"line does not start with {prefix!r}")
    value = json.loads(line[len(prefix) :])
    if not isinstance(value, dict):
        raise ValueError(f"{prefix.strip()} payload must be an object")
    return value


def _table(document: dict[str, Any], name: str) -> dict[str, Any]:
    value = document.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"configuration table {name} is missing")
    return value


def _required_text(document: dict[str, Any], key: str) -> str:
    value = str(document.get(key, "")).strip()
    if not value:
        raise ValueError(f"configuration value {key} is missing")
    return value


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _block_name(value: Any, name: str) -> str:
    text = str(value).strip()
    if not SAFE_BLOCK_NAME.fullmatch(text):
        raise ValueError(
            f"{name} must start with an ASCII letter or digit and contain only "
            "ASCII letters, digits, underscore, hyphen, or period"
        )
    return text


def _enum_text(value: Any, name: str, choices: frozenset[str]) -> str:
    text = str(value).strip()
    if text not in choices:
        raise ValueError(f"{name} must be one of: {', '.join(sorted(choices))}")
    return text


def _text_tuple(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be an array of strings")
    result: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str):
            raise ValueError(f"{name}[{index}] must be a string")
        text = item.strip()
        if (
            not text
            or len(text) > 256
            or any(ord(character) < 32 for character in text)
        ):
            raise ValueError(f"{name}[{index}] must be non-empty safe text")
        result.append(text)
    return tuple(result)


def _positive_int_tuple(value: Any, name: str) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be an array of positive integers")
    return tuple(
        _strict_positive_int(item, f"{name}[{index}]")
        for index, item in enumerate(value)
    )


def _resolve_path(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def _resolve_executable(base: Path, value: str) -> str:
    if not value:
        raise ValueError("acquisition.node_executable must not be empty")
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return str(candidate.resolve())
    if any(separator in value for separator in ("/", "\\")) or value.startswith("."):
        return str((base / candidate).resolve())
    return value


def _safe_remote_path(value: str, name: str) -> str:
    if not value.startswith("/") or not SAFE_REMOTE_NAME.fullmatch(value):
        raise ValueError(f"{name} must be a safe absolute POSIX path")
    return value


def _safe_remote_token(value: str, name: str) -> str:
    if "/" in value or not SAFE_REMOTE_NAME.fullmatch(value):
        raise ValueError(f"{name} must be a safe service/token name")
    return value


def _sha256_text(value: str, name: str) -> str:
    normalized = value.lower()
    if not re.fullmatch(r"[0-9a-f]{64}", normalized):
        raise ValueError(f"{name} must be a 64-character SHA-256 digest")
    return normalized


def _hex_text(
    value: str,
    name: str,
    *,
    minimum_length: int,
    maximum_length: int,
) -> str:
    normalized = value.strip().lower()
    if not (
        minimum_length <= len(normalized) <= maximum_length
        and len(normalized) % 2 == 0
        and re.fullmatch(r"[0-9a-f]+", normalized)
    ):
        raise ValueError(
            f"{name} must be an even-length hexadecimal string "
            f"({minimum_length}..{maximum_length} characters)"
        )
    return normalized


def _boolean(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be true or false")
    return value


def _px4_parameter_name(value: Any) -> str:
    name = str(value).strip()
    if not re.fullmatch(r"[A-Z0-9_]{1,16}", name):
        raise ValueError(f"unsafe PX4 parameter name: {name!r}")
    return name


def _g_version(value: str) -> str:
    if value not in {"G 3.5", "G 4.0"}:
        raise ValueError("acquisition.g_version must be 'G 3.5' or 'G 4.0'")
    return value


def _acquisition_clock_mode(value: str) -> str:
    if value not in {"device_sync", "pc_receive"}:
        raise ValueError(
            "acquisition.clock_mode must be 'device_sync' or 'pc_receive'"
        )
    return value


def _find_repository_root(path: Path) -> Path | None:
    for candidate in (path.parent, *path.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _positive_float(value: Any, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or not number > 0:
        raise ValueError(f"{name} must be positive")
    return number


def _finite_float(value: Any, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _nonnegative_float(value: Any, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{name} must be non-negative")
    return number


def _bounded_float(value: Any, name: str, minimum: float, maximum: float) -> float:
    number = float(value)
    if not math.isfinite(number) or not minimum <= number <= maximum:
        raise ValueError(f"{name} must be between {minimum:g} and {maximum:g}")
    return number


def _positive_int(value: Any, name: str) -> int:
    number = int(value)
    if number <= 0:
        raise ValueError(f"{name} must be positive")
    return number


def _strict_positive_int(value: Any, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    number = int(value)
    if number < 0:
        raise ValueError(f"{name} must be non-negative")
    return number


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None
