"""Traceable waveform-duration policy layered on the immutable source plan."""
from __future__ import annotations

import copy


COMPACT_319_TIMING_POLICY = "compact-hold3-six-cycle-20260915-v1"


def _scan_seconds(row: dict, dwell: float) -> float:
    if row["group"] == "S":
        if row["role"] == "T":
            path = [0, *range(5, 31, 5), *range(25, -31, -5), *range(-25, 1, 5)]
        elif row["role"] == "D":
            path = [0, 10, 20, 30, 0, -10, -20, -30, 0]
        else:
            path = [0, 7.5, 22.5, 27.5, 0, -7.5, -22.5, -27.5, 0]
    else:
        corners = {
            "T": [45, 90, 135, 180],
            "D": [60, 120, 150, 180],
            "H": [67.5, 112.5, 157.5, 180],
        }[row["role"]]
        positive = [*corners, *corners[-2::-1], 0]
        path = [0, *positive, *[-value for value in positive]]
    return len(path) * dwell + sum(abs(end - start) / 2 + 1 for start, end in zip(path, path[1:]))


def effective_nominal_seconds(row: dict, policy: str) -> float:
    if policy != COMPACT_319_TIMING_POLICY:
        raise ValueError(f"Unsupported timing policy: {policy}")
    group = row["group"]
    if row["axis"] == "fixed":
        hold_counts = {"B": 17, "G": 26, "A": 25, "M": 25, "FIX": 25}
        if group in hold_counts:
            count = hold_counts[group]
            return float(count * 3 + count - 1)
        if group == "X":
            return 3 + 4.5 + 3 + 4.5 + 3 + sum(5 / f for f in (.1, .3, .5)) + 3 + 4
        if group == "FX":
            return 6 * 3 + sum(5 / f for f in (.7, .8, 1, 1.5, 2))
    if row["role"] == "T" and group in {"N", "SIDE", "RPM"}:
        return 6 / row["frequency_hz"] + 6
    if row["role"] == "T" and group == "E":
        return 89.0
    if row["role"] == "T" and group == "F1":
        return 12 + sum(6 / f for f in (.7, .8, 1))
    if row["role"] == "T" and group == "F2":
        count = len(row["amplitudes_deg"])
        return count * 12 + (count + 1) * 3
    if row["role"] == "T" and group in {"S", "W"}:
        # Native settling validation requires at least 5 s for analysis holds.
        return _scan_seconds(row, 5)
    return float(row["nominal_seconds"])


def apply_timing_policy(row: dict, policy: str | None) -> dict:
    if not policy:
        return row
    if row["role"] == "T" and row["group"] in {"S", "W"}:
        # These scans retain their original 5 s settling holds and source row.
        if policy != COMPACT_319_TIMING_POLICY:
            raise ValueError(f"Unsupported timing policy: {policy}")
        return row
    result = copy.deepcopy(row)
    result["source_nominal_seconds"] = row["nominal_seconds"]
    result["timing_policy"] = policy
    result["nominal_seconds"] = round(effective_nominal_seconds(row, policy), 6)
    return result
