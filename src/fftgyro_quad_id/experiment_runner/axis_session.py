"""Expand frozen dynamic conditions into an offline, reviewable executor session.

Never connects to hardware. Qualified gains, motor mapping and command-to-RPM
settings are supplied separately; none are inferred from the experiment plan.
"""

from __future__ import annotations

import copy
import math
from typing import Any


def planned_motion_peak_deg_s(segments: list[dict[str, Any]]) -> float:
    """Lower bound on requested peak rate; enough to reject known cap mismatch."""
    peak = 0.0
    for segment in segments:
        kind = segment.get("kind")
        if kind == "MOVE":
            peak = max(peak, 1.875 * abs(segment["end_deg"] - segment["start_deg"]) / segment["duration_s"])
        elif kind == "SINE":
            peak = max(peak, 2 * math.pi * segment["frequency_hz"] * abs(segment["amplitude_deg"]))
        elif kind == "SCAN":
            peak = max(peak, abs(segment["end_deg"] - segment["start_deg"]) / max(segment["duration_s"] - 1, 1e-9))
    return peak


def requested_rate_cap_mismatch(session: dict[str, Any]) -> tuple[float, float] | None:
    """Return (requested lower-bound peak, controller cap) for a known mismatch."""
    if session.get("control_mode") != "AXIS_PID":
        return None
    cascade = session.get("pid", {}).get("cascade", {})
    cap = cascade.get("target_rate_limit_deg_s")
    if not isinstance(cap, (int, float)) or isinstance(cap, bool) or cap <= 0:
        return None
    peak = planned_motion_peak_deg_s(session.get("segments", []))
    return (peak, float(cap)) if peak > cap + 1e-6 else None


def expand_condition(
    row: dict[str, Any], collective: float, repetition: int, external_delta: float = 0.00125
) -> list[dict[str, Any]]:
    """Full candidate profile only. No automatic safety promotion/demotion."""
    if row["axis"] not in {"pitch", "yaw"}:
        raise ValueError("Fixed input conditions require the fixed-input executor, not angle PID")
    if not 1 <= repetition <= row["repeats"]:
        raise ValueError("Repetition does not belong to this condition")
    role, group = row["role"], row["group"]
    compact = bool(row.get("timing_policy"))
    center = row["pitch_deg"] if row["axis"] == "pitch" else 0.0
    phase = {"T": 0, "D": math.pi / 3, "H": math.pi / 5}[role]
    direction = -1 if repetition == 2 else 1
    phase += math.pi if direction < 0 else 0
    parts: list[dict[str, Any]] = []

    def add(kind: str, seconds: float, start: float, **extra: Any) -> None:
        serial = len(parts) + 1
        parts.append(
            dict(
                condition_id=row["id"],
                segment_id=f"{row['id']}-{serial:03}",
                kind=kind,
                duration_s=seconds,
                start_deg=start,
                collective=collective,
                **extra,
            )
        )

    def hold(angle: float, seconds: float = 5, *, settled: bool = False) -> None:
        add(
            "HOLD",
            seconds,
            angle,
            hold_id=f"{row['id']}-h{len(parts) + 1:03}",
            require_settled=settled,
        )

    def sines(specs: list[tuple[float, float, float, float | None]], hold_total: float) -> None:
        # 1 s fades are charged to the adjacent prescribed holds. Core cycles
        # remain intact, with the first 2 cycles (or 20 s for E) marked transient.
        each_hold = hold_total / (len(specs) + 1)
        fade = 1.0
        if each_hold < 2 * fade:
            raise ValueError("Insufficient adjacent hold time for sine fades")
        hold(center, each_hold - fade)
        for index, (frequency, amplitude, core_seconds, ext_frequency) in enumerate(specs):
            omega = 2 * math.pi * frequency
            ext_phase = phase + math.pi / 2
            add(
                "SINE",
                core_seconds + 2 * fade,
                center,
                amplitude_deg=amplitude,
                frequency_hz=frequency,
                phase_rad=phase - omega * fade,
                fade_s=fade,
                external_amplitude=external_delta if ext_frequency else 0,
                external_frequency_hz=ext_frequency or 0,
                external_phase_rad=ext_phase - 2 * math.pi * (ext_frequency or 0) * fade,
                analysis_start_s=fade + (20 if ext_frequency else 2 / frequency),
                analysis_end_s=fade + core_seconds,
            )
            hold(center, each_hold - (fade if index == len(specs) - 1 else 2 * fade))

    def moves(path: list[float], rates: list[float], loops: int, dwell: float) -> None:
        hold(path[0], dwell, settled=True)
        for rate in rates:
            for _ in range(loops):
                for start, end in zip(path, path[1:], strict=False):
                    seconds = 1.875 * abs(end - start) / rate
                    add(
                        "MOVE",
                        seconds,
                        start,
                        end_deg=end,
                        analysis_start_s=0,
                        analysis_end_s=seconds,
                    )
                    hold(end, dwell, settled=True)

    def fast_moves(rates: list[float], loops: int) -> None:
        delta = row.get("fast_move_deg", 10 if row["axis"] == "pitch" else 30)
        if isinstance(delta, bool) or not isinstance(delta, (int, float)) or not math.isfinite(delta) or delta <= 0:
            raise ValueError("fast_move_deg must be a positive finite angle")
        start, end = center - direction * delta / 2, center + direction * delta / 2
        moves([start, end, start], rates, loops, 5)

    if role == "T" and group in {"N", "SIDE", "RPM"}:
        f = row["frequency_hz"]
        sines([(f, row["amplitude_deg"], (6 if compact else 8) / f, None)], 6 if compact else 10)
    elif role == "T" and group == "E":
        specs = [(0.2, 1, 40 if compact else 60, 0.35), (0.35, 1, 40 if compact else 60, 0.2)]
        sines(specs[::direction], 9 if compact else 15)
    elif role == "T" and group == "F1":
        sines([(f, 1, (6 if compact else 8) / f, None) for f in [0.7, 0.8, 1][::direction]], 12 if compact else 20)
    elif role == "T" and group == "F2":
        amps = row["amplitudes_deg"][::direction]
        sines([(0.5, a, 12 if compact else 16, None) for a in amps], (len(amps) + 1) * (3 if compact else 5))
    elif role == "T" and group == "F3":
        fast_moves(row["rates_dps"][::direction], 3)
    elif group == "J":
        if role == "T":
            path, rates, dwell = [-7.5, -15, -30, -15, -7.5], [5, 15], 5
        elif role == "D":
            path, rates, dwell = [-15, -7.5, -30, -15], [7.5, 12.5], 7
        else:
            path, rates, dwell = [-15, -27.5, -10, -22.5, -15], [7.5, 12.5], 6
        if direction < 0:
            path = path[::-1]
        moves(path, rates[::direction], 2, dwell)
        if role == "H":
            parts[0]["duration_s"] = 5
    elif group in {"S", "W"}:
        if group == "S":
            if role == "T":
                path = [0, *range(5, 31, 5), *range(25, -31, -5), *range(-25, 1, 5)]
            elif role == "D":
                path = [0, 10, 20, 30, 0, -10, -20, -30, 0]
            else:
                path = [0, 7.5, 22.5, 27.5, 0, -7.5, -22.5, -27.5, 0]
        else:
            corners = {
                "T": [45, 90, 135, 180],
                "D": [60, 120, 150, 180],
                "H": [67.5, 112.5, 157.5, 180],
            }[role]
            positive = [*corners, *corners[-2::-1], 0]
            path = [0, *positive, *[-v for v in positive]]
        path = [direction * a for a in path]
        # Settling-checked scan holds cannot use the compact 3 s dwell:
        # the native executor requires >= 5 s for non-preparation holds.
        dwell = {"T": 5, "D": 7, "H": 6}[role]
        hold(path[0], dwell, settled=True)
        for start, end in zip(path, path[1:], strict=False):
            add("SCAN", abs(end - start) / 2 + 1, start, end_deg=end)
            hold(end, dwell, settled=True)
    elif role in {"D", "H"} and group in {"N", "SIDE"}:
        specs = (
            [(0.2, 2, 40, None), (0.4, 4, 20, None)]
            if role == "D"
            else [(0.25, 1.5, 32, None), (0.4, 4, 20, None)]
        )
        ext = (0.25, 1, 60, 0.45) if role == "D" else (0.45, 1, 60, 0.25)
        if role == "H":
            sines([ext], 10)
            sines(specs[::-1][::direction], 20)
        else:
            sines(specs[::direction], 20)
            sines([ext], 10)
    elif role in {"D", "H"} and group == "F":
        fast = [(0.75, 0.75, 8 / 0.75, None), (0.9, 0.75, 8 / 0.9, None)]
        if role == "H":
            fast.reverse()
        sines(fast[::direction], 15)
        sines([(0.5, 8 if row["axis"] == "pitch" else 15, 16, None)], 10)
        rates = (
            {"D": [20, 40], "H": [22.5, 45]}
            if row["axis"] == "pitch"
            else {"D": [30, 60], "H": [45, 75]}
        )[role]
        fast_moves(rates[::direction], 2)
    else:
        raise ValueError(f"Unsupported condition: {role}-{group}")

    duration = sum(p["duration_s"] for p in parts)
    remaining = row["nominal_seconds"] - duration
    if remaining > 1e-5 and role in {"D", "H"} and group in {"N", "SIDE", "F", "J"}:
        hold(parts[-1]["start_deg"], remaining)
    elif abs(remaining) > 1e-5:
        raise ValueError(
            f"{row['id']}: expanded {duration}s differs from plan {row['nominal_seconds']}s"
        )
    if role == "T" and group == "N" and row["frequency_hz"] == 0.5 and row["amplitude_deg"] == 5:
        for part in parts:
            part["purpose_tags"] = ["N", "F2-shared-5deg-baseline"]
    return parts


def _settling_hold(
    condition: str, label: str, angle: float, collective: float, duration_s: float = 5
) -> dict[str, Any]:
    return dict(
        condition_id=condition,
        segment_id=label,
        kind="HOLD",
        duration_s=duration_s,
        start_deg=angle,
        collective=collective,
        hold_id=label,
        require_settled=True,
        preparation_only=True,
    )


def build_session(
    rows: list[dict[str, Any]],
    profile: dict[str, Any],
    session_id: str,
    parent_series_id: str,
    repetition: int,
    qualification_evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not rows or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Select unique condition IDs")
    first = rows[0]

    def key(r: dict[str, Any]) -> tuple[Any, ...]:
        return (
            r["role"],
            r["axis"],
            r["wind_mps"],
            r["heading_deg"],
            r["pitch_deg"] if r["axis"] == "yaw" else None,
        )

    if any(key(r) != key(first) for r in rows):
        raise ValueError(
            "Wind, axis, heading, role, or Yaw fixed Pitch changes require a separate session"
        )
    if first["axis"] not in {"pitch", "yaw"}:
        raise ValueError("This builder is for dynamic conditions only")
    if profile["pid"]["axis"] != first["axis"].upper():
        raise ValueError("PID profile axis does not match selected conditions")
    preparation_s = profile.get("preparation_hold_s", 5)
    if (
        isinstance(preparation_s, bool)
        or not isinstance(preparation_s, (float, int))
        or not 1 <= preparation_s <= 5
    ):
        raise ValueError("preparation_hold_s must be between 1 and 5 seconds")
    c = copy.deepcopy(profile)
    # Generation never authorizes a new physical condition or copies a live-run permission.
    c.update(
        schema="fftgyro-axis-session-v1",
        session_id=session_id,
        parent_series_id=parent_series_id,
        role=first["role"],
        repetition=repetition,
        wind_m_s=first["wind_mps"],
        heading_deg=first["heading_deg"],
        fixed_pitch_deg=first["pitch_deg"] if first["axis"] == "yaw" else 0,
        runtime_qualified=qualification_evidence is not None,
        control_mode="AXIS_PID",
        segments=[],
    )
    c["source_conditions"] = copy.deepcopy(rows)
    c["profile_kind"] = ("qualified_dynamic" if qualification_evidence is not None
                         else "full_candidate_requires_qualification")
    if qualification_evidence is not None:
        c["qualification_evidence"] = copy.deepcopy(qualification_evidence)
    c["nominal_condition_seconds"] = sum(r["nominal_seconds"] for r in rows)
    commands = profile["collective_by_rpm_ratio"]
    for row in rows:
        key_rpm = format(row.get("rpm_ratio", 1), "g")
        if key_rpm not in commands:
            raise ValueError(f"Missing measured/configured base command for RPM ratio {key_rpm}")
        parts = expand_condition(
            row, commands[key_rpm], repetition, profile.get("external_delta", 0.00125)
        )
        # Only these extra preparation holds may be shortened, never the planned analysis segments.
        start = parts[0]["start_deg"]
        c["segments"].append(
            _settling_hold(row["id"], f"{row['id']}-entry", start, commands[key_rpm], preparation_s)
        )
        c["segments"].extend(parts)
    end = c["segments"][-1]
    c["segments"].append(
        _settling_hold(end["condition_id"], "session-final", end["start_deg"], end["collective"], preparation_s)
    )
    c["expanded_seconds_excluding_transitions_and_ramps"] = sum(
        s["duration_s"] for s in c["segments"]
    )
    mismatch = requested_rate_cap_mismatch(c)
    if mismatch is not None:
        # Advisory metadata only: the requested peak is not an achieved rate.
        # Qualification and admission remain governed by their existing checks.
        c["rate_cap_mismatch"] = {
            "requested_peak_lower_bound_deg_s": mismatch[0],
            "controller_target_cap_deg_s": mismatch[1],
        }
    if c["expanded_seconds_excluding_transitions_and_ramps"] >= c["maximum_duration_s"]:
        raise ValueError("Session exceeds configured duration; split at condition boundaries")
    return c
