"""Offline commissioning profiles. Never infer mapping, gains or qualification."""

from __future__ import annotations

import copy
import math
from typing import Any


def build_commissioning(profile: dict[str, Any], kind: str) -> dict[str, Any]:
    """Build a separate PILOT hold / small-motion draft from an axis profile.

    The axis, quaternion, mapping and gains come from the supplied profile.
    These are candidate values, not apparatus facts; every qualification is reset.
    """
    if kind not in {"fixed", "hold", "sine", "move", "scan"}:
        raise ValueError("unknown commissioning kind")
    if profile["pid"]["axis"] not in {"PITCH", "YAW"}:
        raise ValueError("supply an explicit Pitch or Yaw profile")
    if kind == "scan" and profile["pid"]["axis"] != "YAW":
        raise ValueError("scan commissioning is Yaw only")
    c = copy.deepcopy(profile)
    axis = c["pid"]["axis"].lower()
    c.update(schema="fftgyro-axis-session-v1", role="PILOT", repetition=1,
             session_id=f"commissioning-{axis}-{kind}", parent_series_id="commissioning-draft",
             wind_m_s=0, heading_deg=0, runtime_qualified=False,
             control_mode="FIXED_HOLD" if kind == "fixed" else "AXIS_PID",
             maximum_duration_s=12 if kind == "fixed" else 60, transition_rate_deg_s=1,
             profile_kind="draft_requires_apparatus_qualification")
    for section in ("hardware", "startup"):
        for key in c[section]:
            if key.endswith("_verified"):
                c[section][key] = False
    c["hardware"]["mechanically_locked_verified"] = False
    c["hardware"]["non_target_pid_enabled"] = False
    c["settling"]["maximum_hold_s"] = 5
    c["startup"]["integral_enable_collective"] = 0.1
    # A draft is not an escalation to the normal profile's 0.8 ceiling.
    c["hardware"].update(motor_min=0, motor_max=0.12, startup_ramp_rate=0.1,
                         normal_ramp_rate=0.1, abort_ramp_rate=0.1)
    center = c["pid"]["zero_angle_deg"]
    if not math.isfinite(center):
        raise ValueError("nonfinite center")
    if kind == "fixed":
        c["pid"].update(kp=0, ki=0, kd=0)
    parts: list[dict[str, Any]] = []

    def add(wave: str, duration: float, start: float, **extra: Any) -> None:
        label = f"{kind}-{len(parts) + 1:02}"
        parts.append(dict(condition_id="COMMISSIONING", segment_id=label, kind=wave,
                          duration_s=duration, start_deg=start, end_deg=start,
                          collective=0.1, **extra))

    def hold(angle: float) -> None:
        add("HOLD", 5, angle, hold_id=f"hold-{len(parts) + 1:02}", require_settled=True)

    hold(center)
    if kind == "sine":
        # Entry level from the canonical plan: 0.5 Hz / 0.5 deg, 8 core cycles.
        add("SINE", 18, center, amplitude_deg=0.5, frequency_hz=0.5,
            phase_rad=-math.pi, fade_s=1, analysis_start_s=5, analysis_end_s=17)
        hold(center)
    elif kind in {"move", "scan"}:
        previous = center
        for target in (center + 0.5, center - 0.5, center):
            duration = (1.875 * abs(target - previous) if kind == "move"
                        else abs(target - previous) + 1)
            add("MOVE" if kind == "move" else "SCAN", max(2, duration), previous)
            parts[-1]["end_deg"] = target
            hold(target)
            previous = target
    c["segments"] = parts
    # Never quietly clamp targets to the profile; reject an incompatible envelope.
    for s in parts:
        extent = s.get("amplitude_deg", 0)
        if (min(s["start_deg"] - extent, s["end_deg"]) < c["pid"]["angle_min_deg"] or
                max(s["start_deg"] + extent, s["end_deg"]) > c["pid"]["angle_max_deg"]):
            raise ValueError("commissioning motion outside supplied angle envelope")
    return c
