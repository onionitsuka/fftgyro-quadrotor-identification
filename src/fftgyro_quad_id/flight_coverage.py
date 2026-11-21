"""Offline audit of repeated free-flight motion, NOT flight approval or model validation.

Input: a manifest plus one normalized, time-aligned CSV per physical flight.
Raw Motive/ULog/RPM adapters are deliberately separate. No motor commands,
fitting, residual-based exclusions, RPM substitution or automatic time alignment.
Run: python -m fftgyro_quad_id.flight_coverage manifest.json --output new-report.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

# Provisional collection targets, not experimentally established sufficiency limits.
DEFAULTS = {
    "max_gap_s": 0.10,
    "max_sync_uncertainty_ms": 10.0,
    "repeat_flights": 2,
    "minimum_motion_valid_s": 10.0,
    "hover_contiguous_s": 8.0,
    "hover_speed_max_m_s": 0.25,
    "hover_vertical_max_m_s": 0.15,
    "hover_pitch_max_deg": 5.0,
    "hover_roll_max_deg": 5.0,
    "hover_rate_max_deg_s": 5.0,
    "pitch_span_deg": 10.0,
    "pitch_rate_deg_s": 10.0,
    "yaw_span_deg": 20.0,
    "yaw_rate_deg_s": 15.0,
    "rate_each_sign_s": 0.5,
    "translation_speed_m_s": 1.0,
    "translation_each_sign_s": 1.0,
    "vertical_speed_m_s": 0.3,
    "vertical_each_sign_s": 0.5,
}
MOTIONS = ("hover", "pitch", "yaw", "forward_reverse", "climb_descent")
FEATURES = ("q_deg_s", "r_deg_s", "u_air_m_s", "pitch_diff", "yaw_diff")


def intervals(t: np.ndarray, mask: np.ndarray, max_gap: float) -> np.ndarray:
    """Credit only intervals with two valid endpoints; never bridge a gap."""
    dt = np.diff(t)
    return np.where(mask[:-1] & mask[1:] & (dt <= max_gap + 1e-9), dt, 0.0)


def longest_run(credited: np.ndarray) -> float:
    current = longest = 0.0
    for value in credited:
        current = current + float(value) if value > 0 else 0.0
        longest = max(longest, current)
    return longest


def bool_column(data: dict, name: str) -> np.ndarray:
    if name not in data or not np.isin(data[name], [0, 1]).all():
        raise ValueError(f"{name}: explicit 0/1 values required")
    return data[name].astype(bool)


def correlation_audit(data: dict, valid: np.ndarray) -> dict:
    """Raw regressor diagnostic, not identifiability of the final gray-box basis."""
    names = [name for name in FEATURES if name in data]
    if not names:
        return {"status": "NO_REGRESSORS"}
    matrix = np.column_stack([data[name] for name in names])
    matrix = matrix[valid & np.isfinite(matrix).all(axis=1)]
    if len(matrix) < 3:
        return {"status": "TOO_FEW_SAMPLES"}
    deviation = matrix.std(axis=0)
    active = deviation > 1e-10
    constant = [name for name, keep in zip(names, active, strict=True) if not keep]
    names = [name for name, keep in zip(names, active, strict=True) if keep]
    if not names:
        return {"status": "CONSTANT_ONLY", "constant_columns": constant}
    standardized = (matrix[:, active] - matrix[:, active].mean(axis=0)) / deviation[active]
    singular = np.linalg.svd(standardized, compute_uv=False)
    tolerance = max(standardized.shape) * np.finfo(float).eps * singular[0]
    corr = standardized.T @ standardized / len(standardized)
    return {
        "status": "EXPLORATORY_ONLY",
        "columns": names,
        "constant_columns": constant,
        "rank": int(np.count_nonzero(singular > tolerance)),
        "singular_values": singular.tolist(),
        "strong_pairs_abs_r_ge_0_95": [
            {"a": names[i], "b": names[j], "r": float(corr[i, j])}
            for i in range(len(names))
            for j in range(i + 1, len(names))
            if abs(corr[i, j]) >= 0.95
        ],
        "note": "Interpolated samples are not independent; this is not a significance test.",
    }


def audit_flight(data: dict, metadata: dict, rules: dict) -> dict:
    t = np.asarray(data.get("time_s", []), dtype=float)
    if len(t) < 2 or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError("time_s must be finite, strictly increasing, and contain >=2 rows")
    data = {key: np.asarray(value, dtype=float) for key, value in data.items()}
    if any(len(value) != len(t) for value in data.values()):
        raise ValueError("All columns must have the same length")
    validity = "pose_valid" if "pose_valid" in data else "mocap_valid"
    base = bool_column(data, "airborne") & bool_column(data, validity)
    # 'usable' must encode acquisition faults, never disagreement with a load model.
    base &= bool_column(data, "usable")
    gap = rules["max_gap_s"]

    def exposure(mask):
        return float(intervals(t, mask, gap).sum())

    results = {}
    geometry = metadata.get("frames_verified") is True

    def available(names):
        return geometry and all(name in data for name in names)

    def motion(name, columns, criterion):
        if not available(columns):
            results[name] = {"status": "UNKNOWN_SOURCE", "required_columns": columns}
            return
        valid = base.copy()
        for column in columns:
            valid &= np.isfinite(data[column])
        minimum = (
            rules["hover_contiguous_s"] if name == "hover" else rules["minimum_motion_valid_s"]
        )
        if exposure(valid) < minimum:
            results[name] = {
                "status": "INSUFFICIENT_VALID_DATA",
                "valid_duration_s": exposure(valid),
            }
            return
        enough, details = criterion(valid)
        results[name] = {
            "status": "OBSERVED" if enough else "NOT_OBSERVED",
            "valid_duration_s": exposure(valid),
            **details,
        }

    def hover(valid):
        slow = data["speed_m_s"] <= rules["hover_speed_max_m_s"]
        slow &= abs(data["vz_m_s"]) <= rules["hover_vertical_max_m_s"]
        slow &= abs(data["pitch_deg"]) <= rules["hover_pitch_max_deg"]
        slow &= abs(data["roll_deg"]) <= rules["hover_roll_max_deg"]
        for rate in ("p_deg_s", "q_deg_s", "r_deg_s"):
            slow &= abs(data[rate]) <= rules["hover_rate_max_deg_s"]
        seconds = longest_run(intervals(t, valid & slow, gap))
        return seconds >= rules["hover_contiguous_s"], {"longest_hover_s": seconds}

    def oscillation(valid, angle, rate, span_min, rate_min, wrapped=False):
        selected = data[angle][valid]
        if wrapped:
            # Never unwrap through tracking loss or a recording gap.
            connected = intervals(t, valid, gap) > 0
            starts = np.r_[0, np.flatnonzero(~connected) + 1]
            ends = np.r_[starts[1:], len(t)]
            spans = [
                float(np.ptp(np.degrees(np.unwrap(np.radians(data[angle][a:b])))))
                for a, b in zip(starts, ends, strict=True)
                if b - a >= 2 and valid[a:b].all()
            ]
            span = max(spans, default=0.0)
        else:
            span = (
                float(np.percentile(selected, 95) - np.percentile(selected, 5))
                if len(selected)
                else 0.0
            )
        positive = exposure(valid & (data[rate] >= rate_min))
        negative = exposure(valid & (data[rate] <= -rate_min))
        enough = (
            span >= span_min
            and min(positive, negative) >= rules["rate_each_sign_s"]
            and exposure(valid) >= rules["minimum_motion_valid_s"]
        )
        return enough, {
            "angle_span_deg": span,
            "positive_rate_s": positive,
            "negative_rate_s": negative,
        }

    def bidirectional(valid, column, threshold, seconds):
        positive = exposure(valid & (data[column] >= threshold))
        negative = exposure(valid & (data[column] <= -threshold))
        return min(positive, negative) >= seconds, {"positive_s": positive, "negative_s": negative}

    motion(
        "hover",
        ["speed_m_s", "vz_m_s", "roll_deg", "pitch_deg", "p_deg_s", "q_deg_s", "r_deg_s"],
        hover,
    )
    motion(
        "pitch",
        ["pitch_deg", "q_deg_s"],
        lambda valid: oscillation(
            valid, "pitch_deg", "q_deg_s", rules["pitch_span_deg"], rules["pitch_rate_deg_s"]
        ),
    )
    motion(
        "yaw",
        ["yaw_deg", "r_deg_s"],
        lambda valid: oscillation(
            valid, "yaw_deg", "r_deg_s", rules["yaw_span_deg"], rules["yaw_rate_deg_s"], True
        ),
    )
    motion(
        "forward_reverse",
        ["u_air_m_s"],
        lambda valid: bidirectional(
            valid, "u_air_m_s", rules["translation_speed_m_s"], rules["translation_each_sign_s"]
        ),
    )
    motion(
        "climb_descent",
        ["vz_m_s"],
        lambda valid: bidirectional(
            valid, "vz_m_s", rules["vertical_speed_m_s"], rules["vertical_each_sign_s"]
        ),
    )

    sync = metadata.get("sync_uncertainty_ms")
    sync_ok = (
        metadata.get("sync_verified") is True
        and isinstance(sync, (float, int))
        and np.isfinite(sync)
        and 0 <= sync <= rules["max_sync_uncertainty_ms"]
    )
    rpm_names = [f"rpm{i}" for i in range(1, 5)]
    cmd_names = [f"command{i}" for i in range(1, 5)] + ["voltage_v"]

    def route_duration(names, calibrated):
        if not geometry or not sync_ok or not calibrated or any(name not in data for name in names):
            return None
        valid = base.copy()
        for name in names:
            valid &= np.isfinite(data[name])
            valid &= (
                (data[name] >= 0) & (data[name] <= 1)
                if name.startswith("command")
                else data[name] > 0
            )
        return exposure(valid)

    rpm_ready = metadata.get("rpm_calibration_verified") is True
    command_ready = metadata.get("command_mapping_verified") is True
    return {
        "flight_id": metadata["flight_id"],
        "recording_key": metadata["recording_key"],
        "pose_source": metadata.get("pose_source", "normalized_mocap"),
        "valid_airborne_duration_s": exposure(base),
        "motions": results,
        "largest_record_interval_s": float(np.diff(t).max()),
        "route_valid_duration_s": {
            "measured_rpm": route_duration(rpm_names, rpm_ready),
            "input_voltage": route_duration(cmd_names, command_ready),
            "common_physical_vs_input_model": route_duration(
                rpm_names + cmd_names, rpm_ready and command_ready
            ),
        },
        "synchronization_ready": sync_ok,
        "regressor_diagnostic": correlation_audit(data, base),
    }


def summarize(flights: list[dict], rules: dict) -> dict:
    for key in ("flight_id", "recording_key"):
        if len({f[key] for f in flights}) != len(flights):
            raise ValueError("Duplicate physical flight; do not count twice")
    decisions = {}
    for name in MOTIONS:
        observed = [f["flight_id"] for f in flights if f["motions"][name]["status"] == "OBSERVED"]
        unknown = [
            f["flight_id"]
            for f in flights
            if f["motions"][name]["status"] in {"UNKNOWN_SOURCE", "INSUFFICIENT_VALID_DATA"}
        ]
        needed = max(0, rules["repeat_flights"] - len(observed))
        status = (
            "REPEATED_MOTION_AVAILABLE"
            if not needed
            else "RESOLVE_DATA_FIRST"
            if unknown
            else "ADDITIONAL_MOTION_CANDIDATE"
        )
        decisions[name] = {
            "status": status,
            "observed_flights": observed,
            "unknown_flights": unknown,
            "additional_flights_if_unknowns_do_not_qualify": needed,
        }
    routes = {}
    for name in ("measured_rpm", "input_voltage", "common_physical_vs_input_model"):
        ready = [
            f["flight_id"]
            for f in flights
            if (f["route_valid_duration_s"][name] or 0) >= rules["minimum_motion_valid_s"]
        ]
        routes[name] = {
            "ready_flights": ready,
            "status": "DATA_AVAILABLE" if ready else "RESOLVE_DATA_FIRST",
        }
    return {
        "motion_decisions": decisions,
        "route_decisions": routes,
        "flights": flights,
        "rules": rules,
        "model_validation": "NOT_RUN",
        "flight_authorization": "NOT_PROVIDED",
        "notes": [
            "Thresholds are provisional collection targets, not proof of model accuracy.",
            "Repeated motion is not proof of repeated identical inputs or independent errors.",
            "Synchronization/calibration failures require data work first, "
            "not automatically new flight.",
            "u_air_m_s = [R_NED_to_FRD*(ground_velocity-wind_velocity)]_x. "
            "In still air it equals body ground velocity. "
            "Beta is undefined at zero airspeed.",
            "Measured-RPM physical baseline and input/voltage model must use "
            "identical valid comparison samples.",
            "No free-flight force sensor is assumed; motion prediction is not "
            "direct six-axis load validation.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if not manifest.get("flights"):
        raise ValueError("No flights provided; missing data is not evidence of missing motion")
    overrides = manifest.get("rules", {})
    if set(overrides) - DEFAULTS.keys():
        raise ValueError("Unknown collection rule")
    rules = DEFAULTS | overrides
    if any(
        isinstance(value, bool) or not np.isfinite(value) or value <= 0 for value in rules.values()
    ) or not isinstance(rules["repeat_flights"], int):
        raise ValueError("Invalid collection rules")
    flights, ids, recordings, hashes = [], set(), set(), set()
    for meta in manifest["flights"]:
        path = args.manifest.parent / meta["csv"]
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if meta["flight_id"] in ids or meta["recording_key"] in recordings or sha in hashes:
            raise ValueError("Duplicate physical flight or identical CSV; do not count twice")
        ids.add(meta["flight_id"])
        recordings.add(meta["recording_key"])
        hashes.add(sha)
        with path.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
                raise ValueError(
                    "Normalized CSV needs unique column names; raw Motive is not this schema"
                )
            rows = list(reader)
            data = {
                name: np.array([float(row[name]) if row[name] else np.nan for row in rows])
                for name in reader.fieldnames
            }
        result = audit_flight(data, meta, rules)
        result["csv_sha256"] = sha
        flights.append(result)
    report = summarize(flights, rules)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(args.output)


if __name__ == "__main__":
    main()
