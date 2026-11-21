from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from random import Random

from .models import ExperimentSegment, PlanSummary, SegmentKind

PROTOCOL_ID = "motor_single"
PROTOCOL_VERSION = "ordinary-dshot-ee-spy401-2026-07-23-v6"

METHOD_STEADY_MAP = "steady-map"
METHOD_DOWN_FINAL_FIXED = "down-final-fixed"
METHOD_DOWN_INITIAL_FIXED = "down-initial-fixed"
METHOD_SYMMETRIC = "symmetric-step"
METHOD_DOWN_OPERATING_POINT = "down-operating-point"
METHOD_THERMAL_COLD = "thermal-reference-cold"
METHOD_THERMAL_WARM = "thermal-reference-warm"
METHOD_DYNAMIC = "dynamic-asymmetric"
METHOD_LARGE_STEP = "large-step-boundary"
METHOD_SYNC = "sync-check"

DEFAULT_METHODS = (
    METHOD_STEADY_MAP,
    METHOD_DYNAMIC,
    METHOD_LARGE_STEP,
    METHOD_SYNC,
)
DOWNWARD_METHODS = (
    METHOD_THERMAL_COLD,
    METHOD_DOWN_FINAL_FIXED,
    METHOD_DOWN_INITIAL_FIXED,
    METHOD_SYMMETRIC,
    METHOD_DOWN_OPERATING_POINT,
    METHOD_THERMAL_WARM,
)
TEST_METHODS = frozenset(DEFAULT_METHODS + DOWNWARD_METHODS)

STEADY_INPUTS = (
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
)
EXTENDED_STEADY_INPUTS = STEADY_INPUTS + (0.85, 0.90, 0.95, 1.00)

DYNAMIC_BANDS = (
    (0.20, 0.40, 0.20),
    (0.40, 0.60, 0.20),
    (0.60, 0.80, 0.10),
)
# Complete sawtooth cycles can overshoot this per-band target. With the fixed
# seed, three bands plus band transitions total approximately 60 seconds.
DYNAMIC_BAND_TARGET_S = 16.0
# ESC thermal damage and melted insulation were observed during the former
# large-rise protocol; the electrical root cause has not been established.
# A measured 60 -> 80 percent transition reached 17.4 A and pulled the
# nominal 16.2 V supply down to 13 V. Use +20 points only through 60 percent;
# above 60 percent, reach 80 percent through consecutive +10 point rises.
DYNAMIC_UP_STEPS = (0.10, 0.20)
DYNAMIC_DOWN_STEP = 0.05
DYNAMIC_HOLD_MIN_S = 0.4
DYNAMIC_HOLD_MAX_S = 1.0
DYNAMIC_SEED = 20260716
LARGE_STEP_CONDITIONS = (
    (0.20, 0.40),
    (0.40, 0.60),
    (0.60, 0.70),
    (0.70, 0.80),
)
LARGE_STEP_BASELINE_HOLD_S = 3.0
LARGE_STEP_RESPONSE_HOLD_S = 2.0
CONTROLLED_RETURN_HOLD_S = 0.75
STEADY_HOLD_S = 10.0
SETTLING_PRE_HOLD_S = 10.0
SETTLING_POST_HOLD_S = 10.0
SETTLING_ANALYSIS_WINDOW_S = 3.0
SETTLING_REPEATS = 3
MAX_CUSTOM_UP_STEP_PERCENT = 20.0
MAX_CUSTOM_HIGH_UP_STEP_PERCENT = 10.0
CUSTOM_HIGH_TARGET_PERCENT = 60.0
MAX_CUSTOM_DOWN_STEP_PERCENT = 15.0
APPROACH_START = 0.10
APPROACH_STEP = 0.10
APPROACH_HOLD_S = 0.5
_METHOD_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]{0,47}$")
FULL_TEST_VOLTAGES_V = (14.7, 15.2, 15.7, 16.2, 16.7)
PRIMARY_RESPONSE_VOLTAGES_V = (16.2,)


@dataclass(frozen=True, slots=True)
class MotorSingleProfile:
    name: str
    motor_ids: tuple[str, ...]
    steady_voltages_v: tuple[float, ...]
    dynamic_voltages_v: tuple[float, ...]
    steady_repeats: int
    dynamic_repeats: int
    steady_inputs: tuple[float, ...] = STEADY_INPUTS
    include_sync_check: bool = True


PROFILES = {
    "recommended": MotorSingleProfile(
        name="recommended",
        motor_ids=("motor_1",),
        steady_voltages_v=FULL_TEST_VOLTAGES_V,
        dynamic_voltages_v=PRIMARY_RESPONSE_VOLTAGES_V,
        steady_repeats=3,
        dynamic_repeats=3,
    ),
    "recommended-extended-100": MotorSingleProfile(
        name="recommended-extended-100",
        motor_ids=("motor_1",),
        steady_voltages_v=FULL_TEST_VOLTAGES_V,
        dynamic_voltages_v=PRIMARY_RESPONSE_VOLTAGES_V,
        steady_repeats=3,
        dynamic_repeats=3,
        steady_inputs=EXTENDED_STEADY_INPUTS,
    ),
    "minimum": MotorSingleProfile(
        name="minimum",
        motor_ids=("motor_1",),
        steady_voltages_v=(14.7, 15.7, 16.7),
        dynamic_voltages_v=(15.7,),
        steady_repeats=2,
        dynamic_repeats=3,
    ),
    "all-motors-validation": MotorSingleProfile(
        name="all-motors-validation",
        motor_ids=("motor_1", "motor_2", "motor_3", "motor_4"),
        steady_voltages_v=FULL_TEST_VOLTAGES_V,
        dynamic_voltages_v=PRIMARY_RESPONSE_VOLTAGES_V,
        steady_repeats=3,
        dynamic_repeats=3,
    ),
}


def _direction(start: float, end: float) -> str:
    if end > start:
        return "up"
    if end < start:
        return "down"
    return "hold"


@dataclass(frozen=True, slots=True)
class CommandStep:
    start: float
    end: float
    hold_s: float
    purpose: str


def _rounded_command(value: float) -> float:
    return round(value + 1e-12, 2)


def _approach_levels(target: float) -> tuple[float, ...]:
    levels: list[float] = []
    value = APPROACH_START
    while value < target - 1e-9:
        levels.append(_rounded_command(value))
        value = _rounded_command(value + APPROACH_STEP)
    return tuple(levels)


def _controlled_descent(start: float, end: float) -> list[CommandStep]:
    if start < end:
        raise ValueError("controlled descent cannot increase command")
    steps: list[CommandStep] = []
    current = _rounded_command(start)
    while current > end + 1e-9:
        next_value = _rounded_command(max(end, current - DYNAMIC_DOWN_STEP))
        steps.append(
            CommandStep(
                start=current,
                end=next_value,
                hold_s=CONTROLLED_RETURN_HOLD_S,
                purpose="controlled_down",
            )
        )
        current = next_value
    return steps


def build_dynamic_waveform(seed: int = DYNAMIC_SEED) -> tuple[CommandStep, ...]:
    """Build one reproducible 60 s-class asymmetric excitation waveform.

    Identified upward jumps are +0.20 through 60 percent and +0.10 above
    60 percent. Every downward
    transition is exactly one adjacent -0.05 step so the fixture is never
    unloaded by a large command drop. Band changes are labelled separately
    from identified steps. The same waveform is reused across voltages and
    repeats.
    """

    rng = Random(seed)
    steps: list[CommandStep] = []
    current = DYNAMIC_BANDS[0][0]
    for lower, upper, up_step in DYNAMIC_BANDS:
        if current < lower - 1e-9:
            steps.append(
                CommandStep(
                    current,
                    lower,
                    DYNAMIC_HOLD_MIN_S,
                    "dynamic_band_transition",
                )
            )
            current = lower
        elif current > lower + 1e-9:
            descent = _controlled_descent(current, lower)
            steps.extend(descent)
            current = lower

        elapsed = 0.0
        while elapsed < DYNAMIC_BAND_TARGET_S:
            if current + up_step > upper + 1e-9:
                descent = _controlled_descent(current, lower)
                steps.extend(descent)
                elapsed += sum(item.hold_s for item in descent)
                current = lower
                continue
            target = _rounded_command(current + up_step)
            hold_s = round(rng.uniform(DYNAMIC_HOLD_MIN_S, DYNAMIC_HOLD_MAX_S), 3)
            steps.append(CommandStep(current, target, hold_s, "dynamic_up"))
            elapsed += hold_s
            current = target

            # The high band reaches 80 percent as 60 -> 70 -> 80 rather than
            # the empirically invalid 60 -> 80 transition.
            if current < upper - 1e-9:
                continue

            descent = _controlled_descent(current, lower)
            for item in descent:
                hold_s = round(
                    rng.uniform(DYNAMIC_HOLD_MIN_S, DYNAMIC_HOLD_MAX_S), 3
                )
                steps.append(CommandStep(item.start, item.end, hold_s, "dynamic_down"))
                elapsed += hold_s
            current = lower

    steps.extend(_controlled_descent(current, 0.20))
    return tuple(steps)


def build_plan(
    profile_name: str,
    methods: Iterable[str] | None = None,
) -> list[ExperimentSegment]:
    try:
        profile = PROFILES[profile_name]
    except KeyError as exc:
        raise ValueError(f"unknown profile: {profile_name}") from exc

    selected_methods = tuple(DEFAULT_METHODS if methods is None else methods)
    if not selected_methods:
        raise ValueError("select at least one motor test method")
    unknown = set(selected_methods).difference(TEST_METHODS)
    if unknown:
        raise ValueError(f"unknown motor test methods: {sorted(unknown)}")
    if len(set(selected_methods)) != len(selected_methods):
        raise ValueError("motor test methods must be unique")
    enabled = set(selected_methods)

    segments: list[ExperimentSegment] = []

    def add(
        *,
        kind: SegmentKind,
        purpose: str,
        motor_id: str,
        voltage_v: float,
        repeat: int,
        input_start: float,
        input_end: float,
        pre_hold_s: float,
        post_hold_s: float,
        direction: str,
        method_id: str,
        condition_id: str,
        analysis_window_s: float | None = None,
    ) -> None:
        sequence = len(segments) + 1
        segments.append(
            ExperimentSegment(
                sequence=sequence,
                segment_id=f"{PROTOCOL_ID}-{sequence:05d}",
                protocol_id=PROTOCOL_ID,
                protocol_version=PROTOCOL_VERSION,
                kind=kind,
                purpose=purpose,
                motor_id=motor_id,
                voltage_v=voltage_v,
                repeat=repeat,
                input_start=input_start,
                input_end=input_end,
                pre_hold_s=pre_hold_s,
                post_hold_s=post_hold_s,
                analysis_window_s=analysis_window_s,
                direction=direction,
                method_id=method_id,
                condition_id=condition_id,
            )
        )

    dynamic_waveform = build_dynamic_waveform()

    def add_settling_run(
        *,
        method_id: str,
        condition_id: str,
        motor_id: str,
        voltage_v: float,
        repeat: int,
        levels: tuple[float, ...],
        purposes: tuple[str, ...],
    ) -> None:
        if len(levels) != len(purposes) or not levels:
            raise ValueError("settling run levels and purposes must be non-empty and aligned")
        previous = 0.0
        for level in _approach_levels(levels[0]):
            add(
                kind=SegmentKind.TRANSIENT,
                purpose="settling_approach",
                motor_id=motor_id,
                voltage_v=voltage_v,
                repeat=repeat,
                input_start=previous,
                input_end=level,
                pre_hold_s=0.0,
                post_hold_s=APPROACH_HOLD_S,
                direction=_direction(previous, level),
                method_id=method_id,
                condition_id=condition_id,
            )
            previous = level
        for index, (level, purpose) in enumerate(
            zip(levels, purposes, strict=True)
        ):
            hold_s = SETTLING_PRE_HOLD_S if index == 0 else SETTLING_POST_HOLD_S
            add(
                kind=SegmentKind.TRANSIENT,
                purpose=purpose,
                motor_id=motor_id,
                voltage_v=voltage_v,
                repeat=repeat,
                input_start=previous,
                input_end=level,
                pre_hold_s=0.0,
                post_hold_s=hold_s,
                analysis_window_s=SETTLING_ANALYSIS_WINDOW_S,
                direction=_direction(previous, level),
                method_id=method_id,
                condition_id=condition_id,
            )
            previous = level

    def add_downward_methods(motor_id: str) -> None:
        def balanced(
            items: tuple[tuple[float, ...], ...],
            repeat: int,
        ) -> tuple[tuple[float, ...], ...]:
            offset = (repeat - 1) % len(items)
            return items[offset:] + items[:offset]

        for voltage_v in profile.dynamic_voltages_v:
            if METHOD_THERMAL_COLD in enabled:
                for repeat in range(1, SETTLING_REPEATS + 1):
                    add_settling_run(
                        method_id=METHOD_THERMAL_COLD,
                        condition_id="drop10_60to50",
                        motor_id=motor_id,
                        voltage_v=voltage_v,
                        repeat=repeat,
                        levels=(0.60, 0.50),
                        purposes=("thermal_baseline", "thermal_down_settle"),
                    )

            if METHOD_DOWN_FINAL_FIXED in enabled:
                final_conditions: tuple[tuple[float, ...], ...] = (
                    (0.55,),
                    (0.65,),
                )
                for repeat in range(1, SETTLING_REPEATS + 1):
                    for (start,) in balanced(final_conditions, repeat):
                        drop = round((start - 0.50) * 100)
                        add_settling_run(
                            method_id=METHOD_DOWN_FINAL_FIXED,
                            condition_id=f"drop{drop:02d}_to50",
                            motor_id=motor_id,
                            voltage_v=voltage_v,
                            repeat=repeat,
                            levels=(start, 0.50),
                            purposes=("down_pre_hold", "down_step_settle"),
                        )

            if METHOD_DOWN_INITIAL_FIXED in enabled:
                initial_conditions: tuple[tuple[float, ...], ...] = (
                    (0.60,),
                    (0.55,),
                )
                for repeat in range(1, SETTLING_REPEATS + 1):
                    for (target,) in balanced(initial_conditions, repeat):
                        drop = round((0.65 - target) * 100)
                        add_settling_run(
                            method_id=METHOD_DOWN_INITIAL_FIXED,
                            condition_id=f"drop{drop:02d}_from65",
                            motor_id=motor_id,
                            voltage_v=voltage_v,
                            repeat=repeat,
                            levels=(0.65, target),
                            purposes=("down_pre_hold", "down_step_settle"),
                        )

            if METHOD_SYMMETRIC in enabled:
                symmetric_conditions: tuple[tuple[float, ...], ...] = (
                    (0.55,),
                    (0.60,),
                    (0.65,),
                )
                for repeat in range(1, SETTLING_REPEATS + 1):
                    for (high,) in balanced(symmetric_conditions, repeat):
                        span = round((high - 0.50) * 100)
                        add_settling_run(
                            method_id=METHOD_SYMMETRIC,
                            condition_id=f"span{span:02d}_around50",
                            motor_id=motor_id,
                            voltage_v=voltage_v,
                            repeat=repeat,
                            levels=(0.50, high, 0.50),
                            purposes=(
                                "symmetric_baseline",
                                "symmetric_up_settle",
                                "symmetric_down_settle",
                            ),
                        )

            if METHOD_DOWN_OPERATING_POINT in enabled:
                operating_conditions: tuple[tuple[float, ...], ...] = (
                    (0.40, 0.30),
                    (0.80, 0.70),
                )
                for repeat in range(1, SETTLING_REPEATS + 1):
                    for start, target in balanced(operating_conditions, repeat):
                        add_settling_run(
                            method_id=METHOD_DOWN_OPERATING_POINT,
                            condition_id=(
                                f"drop10_{round(start * 100):02d}to"
                                f"{round(target * 100):02d}"
                            ),
                            motor_id=motor_id,
                            voltage_v=voltage_v,
                            repeat=repeat,
                            levels=(start, target),
                            purposes=("down_pre_hold", "down_step_settle"),
                        )

    def add_warm_reference(motor_id: str) -> None:
        if METHOD_THERMAL_WARM not in enabled:
            return
        for voltage_v in profile.dynamic_voltages_v:
            for repeat in range(1, SETTLING_REPEATS + 1):
                add_settling_run(
                    method_id=METHOD_THERMAL_WARM,
                    condition_id="drop10_60to50",
                    motor_id=motor_id,
                    voltage_v=voltage_v,
                    repeat=repeat,
                    levels=(0.60, 0.50),
                    purposes=("thermal_baseline", "thermal_down_settle"),
                )

    for motor_id in profile.motor_ids:
        # When a single voltage is selected by the runner, the cold reference
        # starts that voltage batch and the warm reference below ends it.
        add_downward_methods(motor_id)

        if METHOD_STEADY_MAP in enabled:
            for voltage_v in profile.steady_voltages_v:
                for repeat in range(1, profile.steady_repeats + 1):
                    previous = profile.steady_inputs[0]
                    condition_id = (
                        "sweep-10to100" if max(profile.steady_inputs) > 0.80
                        else "sweep-10to80"
                    )
                    for value in profile.steady_inputs:
                        add(
                            kind=SegmentKind.STEADY,
                            purpose="steady_rpm_map",
                            motor_id=motor_id,
                            voltage_v=voltage_v,
                            repeat=repeat,
                            input_start=previous,
                            input_end=value,
                            pre_hold_s=0.0,
                            post_hold_s=STEADY_HOLD_S,
                            analysis_window_s=2.0,
                            direction=_direction(previous, value),
                            method_id=METHOD_STEADY_MAP,
                            condition_id=condition_id,
                        )
                        previous = value

                    for step in _controlled_descent(previous, profile.steady_inputs[0]):
                        add(
                            kind=SegmentKind.TRANSIENT,
                            purpose="controlled_static_return",
                            motor_id=motor_id,
                            voltage_v=voltage_v,
                            repeat=repeat,
                            input_start=step.start,
                            input_end=step.end,
                            pre_hold_s=0.0,
                            post_hold_s=step.hold_s,
                            direction="down",
                            method_id=METHOD_STEADY_MAP,
                            condition_id=condition_id,
                        )

        if (
            METHOD_DYNAMIC in enabled
            or METHOD_LARGE_STEP in enabled
            or METHOD_SYNC in enabled
        ):
            for voltage_v in profile.dynamic_voltages_v:
                if METHOD_DYNAMIC in enabled:
                    for repeat in range(1, profile.dynamic_repeats + 1):
                        add(
                            kind=SegmentKind.TRANSIENT,
                            purpose="dynamic_baseline",
                            motor_id=motor_id,
                            voltage_v=voltage_v,
                            repeat=repeat,
                            input_start=0.20,
                            input_end=0.20,
                            pre_hold_s=0.0,
                            post_hold_s=3.0,
                            direction="hold",
                            method_id=METHOD_DYNAMIC,
                            condition_id="bands-20to80",
                        )
                        for step in dynamic_waveform:
                            add(
                                kind=SegmentKind.TRANSIENT,
                                purpose=step.purpose,
                                motor_id=motor_id,
                                voltage_v=voltage_v,
                                repeat=repeat,
                                input_start=step.start,
                                input_end=step.end,
                                pre_hold_s=0.0,
                                post_hold_s=step.hold_s,
                                direction=_direction(step.start, step.end),
                                method_id=METHOD_DYNAMIC,
                                condition_id="bands-20to80",
                            )

                if METHOD_LARGE_STEP in enabled:
                    for repeat in range(1, profile.dynamic_repeats + 1):
                        for start, target in LARGE_STEP_CONDITIONS:
                            rise_points = round((target - start) * 100)
                            condition_id = (
                                f"rise{rise_points:02d}_{round(start * 100):02d}to"
                                f"{round(target * 100):02d}"
                            )
                            previous = 0.0
                            for level in _approach_levels(start):
                                add(
                                    kind=SegmentKind.TRANSIENT,
                                    purpose="large_step_approach",
                                    motor_id=motor_id,
                                    voltage_v=voltage_v,
                                    repeat=repeat,
                                    input_start=previous,
                                    input_end=level,
                                    pre_hold_s=0.0,
                                    post_hold_s=APPROACH_HOLD_S,
                                    direction=_direction(previous, level),
                                    method_id=METHOD_LARGE_STEP,
                                    condition_id=condition_id,
                                )
                                previous = level
                            add(
                                kind=SegmentKind.TRANSIENT,
                                purpose="large_step_baseline",
                                motor_id=motor_id,
                                voltage_v=voltage_v,
                                repeat=repeat,
                                input_start=previous,
                                input_end=start,
                                pre_hold_s=0.0,
                                post_hold_s=LARGE_STEP_BASELINE_HOLD_S,
                                direction=_direction(previous, start),
                                method_id=METHOD_LARGE_STEP,
                                condition_id=condition_id,
                            )
                            add(
                                kind=SegmentKind.TRANSIENT,
                                purpose="large_step_up",
                                motor_id=motor_id,
                                voltage_v=voltage_v,
                                repeat=repeat,
                                input_start=start,
                                input_end=target,
                                pre_hold_s=0.0,
                                post_hold_s=LARGE_STEP_RESPONSE_HOLD_S,
                                direction="up",
                                method_id=METHOD_LARGE_STEP,
                                condition_id=condition_id,
                            )

                if METHOD_SYNC in enabled and profile.include_sync_check:
                    add(
                        kind=SegmentKind.SYNC,
                        purpose="hardware_sync_marker",
                        motor_id=motor_id,
                        voltage_v=voltage_v,
                        repeat=1,
                        input_start=0.20,
                        input_end=0.20,
                        pre_hold_s=1.0,
                        post_hold_s=1.0,
                        direction="hold",
                        method_id=METHOD_SYNC,
                        condition_id="marker-20",
                    )

        add_warm_reference(motor_id)

    return segments


def build_custom_step_plan(
    *,
    method_name: str,
    levels_percent: Iterable[float],
    hold_s: float,
    repetitions: int,
    voltage_v: float,
    motor: int,
) -> list[ExperimentSegment]:
    """Build a guarded custom command sequence while preserving run metadata."""
    if not _METHOD_NAME_PATTERN.fullmatch(method_name):
        raise ValueError("method_name must use lowercase letters, digits and hyphens")
    levels = tuple(float(value) for value in levels_percent)
    if not levels or any(not 0.0 < value <= 100.0 for value in levels):
        raise ValueError("custom levels must be within (0, 100]")
    if hold_s <= 0.0:
        raise ValueError("custom hold must be positive")
    if repetitions < 1:
        raise ValueError("custom repetitions must be positive")
    if not 1 <= motor <= 16:
        raise ValueError("custom motor must be within [1, 16]")
    if voltage_v <= 0.0:
        raise ValueError("custom voltage must be positive")
    for left, right in zip(levels, levels[1:], strict=False):
        delta = right - left
        maximum_up = (
            MAX_CUSTOM_HIGH_UP_STEP_PERCENT
            if right > CUSTOM_HIGH_TARGET_PERCENT + 1e-9
            else MAX_CUSTOM_UP_STEP_PERCENT
        )
        if delta > maximum_up + 1e-9:
            raise ValueError(
                f"custom upward step {delta:g}% to {right:g}% exceeds "
                f"{maximum_up:g}%"
            )
        if -delta > MAX_CUSTOM_DOWN_STEP_PERCENT + 1e-9:
            raise ValueError(
                f"custom downward step {-delta:g}% exceeds "
                f"{MAX_CUSTOM_DOWN_STEP_PERCENT:g}%"
            )

    condition_id = "levels-" + "-".join(f"{value:g}".replace(".", "p") for value in levels)
    segments: list[ExperimentSegment] = []
    motor_id = f"motor_{motor}"
    for repeat in range(1, repetitions + 1):
        first_level = levels[0] / 100.0
        previous = 0.0
        for level in _approach_levels(first_level):
            sequence = len(segments) + 1
            segments.append(
                ExperimentSegment(
                    sequence=sequence,
                    segment_id=f"{PROTOCOL_ID}-{sequence:05d}",
                    protocol_id=PROTOCOL_ID,
                    protocol_version=PROTOCOL_VERSION,
                    kind=SegmentKind.TRANSIENT,
                    purpose="settling_approach",
                    motor_id=motor_id,
                    voltage_v=voltage_v,
                    repeat=repeat,
                    input_start=previous,
                    input_end=level,
                    pre_hold_s=0.0,
                    post_hold_s=APPROACH_HOLD_S,
                    direction=_direction(previous, level),
                    method_id=method_name,
                    condition_id=condition_id,
                )
            )
            previous = level
        for index, value_percent in enumerate(levels):
            value = value_percent / 100.0
            sequence = len(segments) + 1
            segments.append(
                ExperimentSegment(
                    sequence=sequence,
                    segment_id=f"{PROTOCOL_ID}-{sequence:05d}",
                    protocol_id=PROTOCOL_ID,
                    protocol_version=PROTOCOL_VERSION,
                    kind=SegmentKind.TRANSIENT,
                    purpose="custom_baseline" if index == 0 else "custom_step_settle",
                    motor_id=motor_id,
                    voltage_v=voltage_v,
                    repeat=repeat,
                    input_start=previous,
                    input_end=value,
                    pre_hold_s=0.0,
                    post_hold_s=hold_s,
                    analysis_window_s=min(SETTLING_ANALYSIS_WINDOW_S, hold_s),
                    direction=_direction(previous, value),
                    method_id=method_name,
                    condition_id=condition_id,
                )
            )
            previous = value
    return segments


def summarize(profile_name: str, segments: Iterable[ExperimentSegment]) -> PlanSummary:
    items = list(segments)
    counts = Counter(item.purpose for item in items)
    upward_steps = [
        item.input_end - item.input_start
        for item in items
        if item.input_end > item.input_start
    ]
    downward_steps = [
        item.input_start - item.input_end
        for item in items
        if item.input_end < item.input_start
    ]
    return PlanSummary(
        protocol_id=PROTOCOL_ID,
        protocol_version=PROTOCOL_VERSION,
        profile=profile_name,
        segment_count=len(items),
        steady_segments=counts["steady_rpm_map"],
        dynamic_up_transitions=counts["dynamic_up"],
        controlled_down_transitions=len(downward_steps),
        sync_segments=counts["hardware_sync_marker"],
        max_up_step=round(max(upward_steps, default=0.0), 2),
        max_down_step=round(max(downward_steps, default=0.0), 2),
        pure_recording_seconds=round(sum(item.duration_s for item in items), 3),
    )
