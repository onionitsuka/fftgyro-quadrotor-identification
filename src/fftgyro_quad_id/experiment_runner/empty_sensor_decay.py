"""Unpowered, hand-spun empty-sensor captures; no prescribed angle or release key."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

KIND = "manual-empty-sensor-decay"
STAGE = "E-empty-decay"
STATIC_STAGE = "E-empty-static"
STAGES = (STAGE, STATIC_STAGE)
DEFAULT_SECONDS = 90
DEFAULT_STATIC_SECONDS = 30
PRE_STATIC_S = 10
PUSH_WINDOW_S = 10
POSES = (
    ("level", "水平付近", 0),
    ("nose-down", "前下がり付近", -30),
    ("nose-up", "前上がり付近", 30),
)


def validate_seconds(seconds: int, *, static: bool = False) -> None:
    minimum = 20 if static else 60
    if isinstance(seconds, bool) or not isinstance(seconds, int) or not minimum <= seconds <= 300:
        raise ValueError(f"empty-sensor recording must be {minimum}..300 seconds")


def build_plan(seconds: int = DEFAULT_SECONDS) -> dict:
    validate_seconds(seconds)
    rows = []

    def add(axis, pose, *, direction=None, repetition=None, baseline_role=None):
        pose_id, label, pitch = pose
        static = baseline_role is not None
        suffix = f"{direction.lower()}-{repetition}" if not static else baseline_role
        # Retain the previous horizontal-Yaw and Pitch IDs.
        prefix = f"E-{axis.lower()}" if pose_id in ("level", "pitch-free") else f"E-yaw-{pose_id}"
        identifier = f"E-static-{pose_id}-{suffix}" if static else f"{prefix}-{suffix}"
        rows.append(
            dict(
                id=identifier,
                wind_m_s=0,
                split="T",
                stage=STATIC_STAGE if static else STAGE,
                axis=axis,
                pair_id=None,
                acquisition_mode="static" if static else "decay",
                pose_id=pose_id,
                pose_label=label,
                nominal_fixed_pitch_deg=pitch,
                nominal_angles_are_measured=False,
                fixed_yaw_pose="existing_lock_position" if axis == "PITCH" else None,
                selected_axis_free=not static,
                all_axes_locked=static,
                baseline_role=baseline_role,
                direction_label=direction,
                repetition=repetition,
                excitation_label=(
                    None if static else ("usual", "stronger", "stronger-repeat")[repetition - 1]
                ),
                seconds=DEFAULT_STATIC_SECONDS if static else seconds,
                location_id="outside_wind_tunnel",
                source_session=None,
                purpose="empty_sensor_pose_output" if static else "empty_sensor_motion_output",
                motor_commands_sent=False,
                target_angle_deg=None,
                target_rate_deg_s=None,
            )
        )

    for pose in POSES:
        add("PITCH", pose, baseline_role="pre")
    for pose in POSES:
        for direction in ("POS", "NEG"):
            for repetition in (1, 2, 3):
                add("YAW", pose, direction=direction, repetition=repetition)
    for direction in ("POS", "NEG"):
        for repetition in (1, 2, 3):
            add(
                "PITCH",
                ("pitch-free", "Pitch自由・Yawは既存の固定方位", None),
                direction=direction,
                repetition=repetition,
            )
    for pose in POSES:
        add("PITCH", pose, baseline_role="post")
    return dict(
        schema="fftgyro-bayesian-retest-runner-v1",
        plan_kind=KIND,
        production_config="configs/experiment_runner/production.json",
        acquisition_backend="scripts/run_sync_bench.py",
        status="PREPARATION_ONLY",
        powered_counts_by_wind={"0": 0},
        manual_counts_by_axis={"YAW": 18, "PITCH": 6},
        static_pose_count=6,
        default_static_qc="configs/experiment_runner/empty-sensor-static-qc.json",
        additional_T_max_blocks=0,
        aircraft_removed=True,
        sensor_measurement_side_empty=True,
        empty_yaw_rig_wind_contribution="neglected_by_user_assumption",
        protocol=dict(
            pre_static_s=PRE_STATIC_S,
            manual_push_window_s=PUSH_WINDOW_S,
            release_enter_required=False,
            hand_release_automatically_detected=False,
            full_settling_required=False,
            actual_direction_from_unwrapped_encoder=True,
            existing_physical_limits_unchanged=True,
            static_all_axes_mechanically_locked=True,
            automatic_tare=False,
            pitch_equilibrium_not_prescribed=True,
            arbitrary_fixed_yaw_heading_required=False,
            effective_sensor_mass_not_assumed=True,
        ),
        runs=rows,
    )


def idle_arguments(settings: dict, seconds: int, *, static: bool = False) -> list[str]:
    """Only the established idle recorder; no session, PID, arming or battery flag."""
    validate_seconds(seconds, static=static)
    args = ["--idle", "--seconds", str(seconds)]
    for key in (
        "aircraft_host",
        "leptrino_host",
        "aircraft_host_key_alias",
        "leptrino_host_key_alias",
        "clock_server_ip",
        "clock_client_visible_ip",
        "windows_client_visible_ip",
        "windows_clock_port",
        "encoder_port",
        "encoder_baud",
        "encoder_minimum_rate_hz",
    ):
        if key in settings:
            args.extend(("--" + key.replace("_", "-"), str(settings[key])))
    return args


def condition_metadata(row: dict) -> dict:
    """Preserve the selected plan row in run.json even with a custom output name."""
    keys = (
        "id",
        "stage",
        "axis",
        "acquisition_mode",
        "pose_id",
        "pose_label",
        "nominal_fixed_pitch_deg",
        "nominal_angles_are_measured",
        "fixed_yaw_pose",
        "selected_axis_free",
        "all_axes_locked",
        "baseline_role",
        "direction_label",
        "repetition",
        "excitation_label",
        "wind_m_s",
        "location_id",
        "split",
        "motor_commands_sent",
        "target_angle_deg",
        "target_rate_deg_s",
    )
    return {"schema": "fftgyro-empty-sensor-condition-v1", **{key: row[key] for key in keys}}


def validate_condition(value: dict, context: dict | None) -> None:
    """Check labels and setup without treating nominal posture as measured truth."""
    if not isinstance(value, dict) or value.get("schema") != "fftgyro-empty-sensor-condition-v1":
        raise ValueError("invalid empty-sensor condition schema")
    if context is None or context.get("stage") not in STAGES:
        raise ValueError("empty-sensor condition requires an empty-sensor context")
    for key in (
        "stage",
        "axis",
        "pose_id",
        "wind_m_s",
        "location_id",
        "split",
        "selected_axis_free",
        "all_axes_locked",
    ):
        if value.get(key) != context.get(key):
            raise ValueError(f"empty-sensor context mismatch: {key}")
    if value.get("motor_commands_sent") is not False or value.get("wind_m_s") != 0:
        raise ValueError("empty-sensor condition must be unpowered and wind-free")
    if value.get("nominal_angles_are_measured") is not False:
        raise ValueError("nominal pose must not be labelled as a measured angle")
    if value.get("target_angle_deg") is not None or value.get("target_rate_deg_s") is not None:
        raise ValueError("empty-sensor capture must have no angle/rate target")
    static = value["stage"] == STATIC_STAGE
    if value.get("acquisition_mode") != ("static" if static else "decay"):
        raise ValueError("empty-sensor acquisition mode/stage mismatch")
    if static:
        if (
            value.get("baseline_role") not in ("pre", "post")
            or value.get("direction_label") is not None
        ):
            raise ValueError("static pose needs a pre/post role, not a spin direction")
    elif value.get("direction_label") not in ("POS", "NEG") or value.get("repetition") not in (
        1,
        2,
        3,
    ):
        raise ValueError("manual decay requires direction and repetition labels")
    poses = {pose[0]: pose[2] for pose in POSES}
    pitch_free = not static and value["axis"] == "PITCH"
    expected_pose = "pitch-free" if pitch_free else value.get("pose_id")
    if expected_pose not in (*poses, "pitch-free") or value.get("pose_id") != expected_pose:
        raise ValueError("invalid empty-sensor pose")
    expected_pitch = None if pitch_free else poses.get(expected_pose)
    if value.get("nominal_fixed_pitch_deg") != expected_pitch:
        raise ValueError("nominal fixed-pitch label mismatch")
    pitch = value.get("nominal_fixed_pitch_deg")
    if pitch is not None and (
        isinstance(pitch, bool) or not isinstance(pitch, (int, float)) or not math.isfinite(pitch)
    ):
        raise ValueError("invalid nominal fixed-pitch angle")


@dataclass
class ManualDecayGuide:
    """Record instruction times, not inferred hand-contact/release ground truth."""

    ready_monotonic_ns: int | None = None
    last_phase: int = -1
    events: list[dict] = field(default_factory=list)
    static_capture: bool = False
    pose_label: str = ""
    excitation_label: str = ""

    def advance(self, now_ns: int, utc_ns: int) -> list[dict]:
        if self.ready_monotonic_ns is None:
            self.ready_monotonic_ns = now_ns
        elapsed = (now_ns - self.ready_monotonic_ns) / 1e9
        phase = (
            0
            if self.static_capture or elapsed < PRE_STATIC_S
            else 1
            if elapsed < PRE_STATIC_S + PUSH_WINDOW_S
            else 2
        )
        if phase <= self.last_phase:
            return []
        names = ("STATIC_INSTRUCTION", "MANUAL_PUSH_INSTRUCTION", "NO_TOUCH_INSTRUCTION")
        messages = (
            "計測準備完了：まず10秒触らず記録。静止しなくても手で押さえないでください。",
            "今から10秒以内に手で回して放してください。角度・初速合わせ、Enterは不要。"
            + {
                "usual": " 今回は通常の勢い。",
                "stronger": " 今回は可動域内で少し強め。",
                "stronger-repeat": " 今回は強めの再現。",
            }.get(self.excitation_label, ""),
            "以後は触らず減衰を記録します。まだ回っていても、そのまま離れてください。",
        )
        if self.static_capture:
            names = ("STATIC_POSE_INSTRUCTION",)
            messages = (
                f"計測準備完了：{self.pose_label}。全軸を機械固定したまま、最後まで触らず記録します。",
            )
        event = dict(
            kind=names[phase],
            utc_ns=utc_ns,
            local_monotonic_ns=now_ns,
            elapsed_after_ready_s=elapsed,
            message=messages[phase],
            hand_release_observed=False,
        )
        # If a display update is late, never issue an already expired push instruction.
        self.last_phase = phase
        self.events.append(event)
        return [event]

    def metadata(self) -> dict:
        return dict(
            events=list(self.events),
            ready_local_monotonic_ns=self.ready_monotonic_ns,
            phase_reached=self.last_phase,
            hand_release_automatically_detected=False,
            instruction_times_are_not_release_times=True,
            contact_interval_requires_offline_review=True,
            zero_or_inertia_validated=False,
            acquisition_mode="static" if self.static_capture else "decay",
            pose_label=self.pose_label,
        )
