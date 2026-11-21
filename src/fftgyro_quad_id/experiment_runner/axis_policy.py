from __future__ import annotations


# Apparatus constraint confirmed by the operator on 2026-07-16.
ROLL_FIXED_DEG = 0.0
STATIC_FIXED_AXES = ("roll", "pitch", "yaw")
DYNAMIC_EXCITATION_AXES = ("pitch", "yaw")
DYNAMIC_FORBIDDEN_AXES = ("roll",)


def dynamic_axis_allowed(axis: str) -> bool:
    return axis in DYNAMIC_EXCITATION_AXES
