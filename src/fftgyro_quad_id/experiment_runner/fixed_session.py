"""Deterministic expansion of formal fixed C/P/R/Y experiment conditions.

This module is offline only.  It never qualifies hardware or starts motors.
The one-second transitions are executed by the native FIXED_INPUT runner and
are therefore not duplicated as analysis segments here.
"""
from __future__ import annotations

import copy
import math
from typing import Any

MODES = {"C": 0, "P": 1, "R": 2, "Y": 3}


def _vector(c: float, **modes: float) -> list[float]:
    value = [c, 0.0, 0.0, 0.0]
    for name, amount in modes.items():
        value[MODES[name]] = amount
    return value


def _base(row: dict[str, Any], profile: dict[str, Any]) -> float:
    ratio = row.get("rpm_ratio", 0.95 if row.get("positive_angle") else 1.0)
    key = format(ratio, "g")
    try:
        return float(profile["collective_by_rpm_ratio"][key])
    except KeyError as error:
        raise ValueError(f"Missing measured/configured base command for RPM ratio {key}") from error


def expand_fixed_condition(row: dict[str, Any], profile: dict[str, Any], repetition: int) -> list[dict[str, Any]]:
    if row["axis"] != "fixed" or row["group"] not in {"B", "G", "A", "M", "X", "FX", "FIX"}:
        raise ValueError("Not a supported fixed-input condition")
    if not 1 <= repetition <= row["repeats"]:
        raise ValueError("Repetition does not belong to this condition")
    base = _base(row, profile)
    amplitudes = profile["fixed_amplitudes"]
    small, medium, large = (float(amplitudes[k]) for k in ("small", "medium", "large"))
    direction = -1 if repetition == 2 else 1
    phase = {"T": 0.0, "D": math.pi / 3, "H": math.pi / 5}[row["role"]]
    parts: list[dict[str, Any]] = []
    compact = bool(row.get("timing_policy"))
    hold_seconds = 3 if compact else 5

    def add(kind: str, seconds: float, value: list[float], *, amplitude: list[float] | None = None,
            frequency: float = 0.0, fade: float = 0.0, analysis_start: float = 0.0) -> None:
        serial = len(parts) + 1
        part = dict(condition_id=row["id"], segment_id=f"{row['id']}-{serial:03}", kind=kind,
                    duration_s=seconds, start_deg=0.0, end_deg=0.0, collective=value[0],
                    fixed_input=value, require_settled=False,
                    analysis_start_s=analysis_start, analysis_end_s=seconds)
        if kind == "HOLD":
            part["hold_id"] = f"{row['id']}-h{serial:03}"
        else:
            part.update(fixed_amplitude=amplitude, frequency_hz=frequency,
                        phase_rad=phase, fade_s=fade)
        parts.append(part)

    zero = _vector(base)
    def alternating(points: list[list[float]]) -> None:
        add("HOLD", hold_seconds, zero)
        for point in points:
            add("HOLD", hold_seconds, point)
            add("HOLD", hold_seconds, zero)

    group = row["group"]
    if group == "B":
        low = float(profile["collective_by_rpm_ratio"]["0.8"])
        high = float(profile["collective_by_rpm_ratio"]["1.2"])
        alternating([_vector(high), _vector(low)] +
                    [_vector(base, **{mode: sign * medium}) for mode in "PRY" for sign in (1, -1)])
    elif group == "G":
        low = float(profile["collective_by_rpm_ratio"]["0.8"])
        high = float(profile["collective_by_rpm_ratio"]["1.2"])
        for domain in ([low, high] if direction > 0 else [high, low]):
            local = _vector(domain)
            add("HOLD", hold_seconds, local)
            for mode in "PRY":
                amount = float(amplitudes.get(f"{mode.lower()}_maximum", large))
                for sign in (direction, -direction):
                    add("HOLD", hold_seconds, _vector(domain, **{mode: sign * amount}))
                    add("HOLD", hold_seconds, local)
    elif group == "A":
        points = [_vector(base, **{mode: sign * amount})
                  for mode in "PRY" for amount in (small, large) for sign in (direction, -direction)]
        alternating(points)
    elif group == "M":
        half = float(amplitudes.get("combined", large / 2))
        points = []
        for left, right in (("P", "Y"), ("R", "P"), ("R", "Y")):
            for a, b in ((1, 1), (1, -1), (-1, 1), (-1, -1))[::direction]:
                points.append(_vector(base, **{left: a * half, right: b * half}))
        alternating(points)
    elif group in {"X", "FX"}:
        mode = row["mode"]
        amount = float(amplitudes.get("c_small", small)) if mode == "C" else small
        amp = [0.0] * 4
        amp[MODES[mode]] = amount
        if group == "X":
            # Four 1 s native transitions + 13 s endpoint holds = specified 17 s step block.
            add("HOLD", hold_seconds, zero)
            add("HOLD", 4.5 if compact else 6.5, _vector(base, **({mode: direction * amount} if mode != "C" else {}))
                if mode != "C" else _vector(base + direction * amount))
            add("HOLD", hold_seconds, zero)
            add("HOLD", 4.5 if compact else 6.5, _vector(base, **({mode: -direction * amount} if mode != "C" else {}))
                if mode != "C" else _vector(base - direction * amount))
            add("HOLD", hold_seconds, zero)
            for frequency in (0.1, 0.3, 0.5):
                add("SINE", (5 if compact else 6) / frequency, zero, amplitude=amp, frequency=frequency,
                    fade=min(1.0, 1.0 / frequency), analysis_start=2 / frequency)
            add("HOLD", hold_seconds, zero)
        else:
            add("HOLD", hold_seconds, zero)
            for frequency in (0.7, 0.8, 1.0, 1.5, 2.0):
                add("SINE", (5 if compact else 6) / frequency, zero, amplitude=amp, frequency=frequency,
                    fade=min(1.0, 1.0 / frequency), analysis_start=2 / frequency)
                add("HOLD", hold_seconds, zero)
    else:  # Formal fixed D/H: eight single endpoints plus four signed compound endpoints.
        single = float(amplitudes.get("diagnostic", 0.0125 if row["role"] == "D" else 0.015))
        compound = float(amplitudes.get("diagnostic_combined", single * (0.5 if row["role"] == "D" else 0.75)))
        c_delta = float(amplitudes.get("c_diagnostic", medium))
        points = [_vector(base + sign * c_delta) for sign in (direction, -direction)]
        points += [_vector(base, **{mode: sign * single}) for mode in "PRY" for sign in (direction, -direction)]
        points += [_vector(base, P=a * compound, Y=b * compound)
                   for a, b in ((direction, direction), (direction, -direction),
                                (-direction, direction), (-direction, -direction))]
        alternating(points)

    return parts


def build_fixed_session(row: dict[str, Any], profile: dict[str, Any], repetition: int,
                        session_id: str | None = None,
                        qualification_evidence: dict[str, Any] | None = None) -> dict[str, Any]:
    c = copy.deepcopy(profile)
    c.update(schema="fftgyro-axis-session-v1", session_id=session_id or f"e-{row['id']}-r{repetition}",
             parent_series_id=row["id"], role=row["role"], repetition=repetition,
             wind_m_s=row["wind_mps"], heading_deg=row["heading_deg"],
             fixed_pitch_deg=row["pitch_deg"], runtime_qualified=qualification_evidence is not None,
             control_mode="FIXED_INPUT", source_conditions=[copy.deepcopy(row)])
    c["profile_kind"] = ("qualified_fixed_input" if qualification_evidence is not None
                         else "full_candidate_requires_fixed_input_qualification")
    if qualification_evidence is not None:
        c.setdefault("hardware", {})["fixed_input_mapping_verified"] = True
        c["qualification_evidence"] = copy.deepcopy(qualification_evidence)
    c["segments"] = expand_fixed_condition(row, profile, repetition)
    # PX4's world-frame yaw is arbitrary after boot.  A formal fixed-input run
    # must therefore measure the locked rig pose at the start of each run and
    # apply attitude/rate guards to motion relative to that pose.  Using the
    # identity quaternion here can reject a motionless rig before the first
    # motor command when its reported yaw happens to exceed the swing limit.
    c.setdefault("startup", {})["capture_reference"] = True
    # Nominal plan seconds include one second for every changed endpoint.
    transitions = sum(a["fixed_input"] != b["fixed_input"] for a, b in zip(c["segments"], c["segments"][1:]))
    prescribed = sum(s["duration_s"] for s in c["segments"]) + transitions
    if not math.isclose(prescribed, row["nominal_seconds"], abs_tol=1e-5):
        raise ValueError(f"{row['id']}: expanded {prescribed}s differs from plan {row['nominal_seconds']}s")
    c["nominal_condition_seconds"] = row["nominal_seconds"]
    c["maximum_duration_s"] = prescribed + 90
    return c
