from __future__ import annotations

import math
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


PRIMARY_REFERENCES = frozenset(
    {"q_derived", "sensor1", "sensor2", "sensor_pair_mean"}
)


@dataclass(frozen=True, slots=True)
class WindObservation:
    target_mps: float
    q_derived_mps: float
    sensor1_mps: float
    sensor2_mps: float | None = None


@dataclass(frozen=True, slots=True)
class WindQualityResult:
    ready_for_identification: bool
    ready_for_frame_diagnostic: bool
    target_mps: float
    q_derived_mps: float
    sensor1_mps: float
    sensor2_mps: float | None
    sensor_pair_mean_mps: float | None
    sensor_pair_difference_mps: float | None
    frame_speed_delta_mps: float | None
    frame_speed_ratio: float | None
    frame_speed_change_fraction: float | None
    primary_reference: str
    model_wind_speed_mps: float | None
    target_error_fraction: float | None
    measured_value_required_by_pdf: bool
    blockers: tuple[str, ...]
    frame_diagnostic_blockers: tuple[str, ...]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["blockers"] = list(self.blockers)
        row["frame_diagnostic_blockers"] = list(self.frame_diagnostic_blockers)
        row["warnings"] = list(self.warnings)
        return row


def _finite_nonnegative(value: float) -> bool:
    return math.isfinite(value) and value >= 0.0


def evaluate_wind_observation(
    configuration_path: Path,
    observation: WindObservation,
) -> WindQualityResult:
    with configuration_path.open("rb") as stream:
        document = tomllib.load(stream)
    policy = document.get("wind_measurement_policy", {})
    layout = document.get("wind_sensor_layout", {})
    blockers: list[str] = []
    frame_blockers: list[str] = []
    warnings: list[str] = []

    for name, value in asdict(observation).items():
        if value is not None and not _finite_nonnegative(value):
            blockers.append(f"{name}_must_be_finite_and_nonnegative")

    primary = str(policy.get("primary_reference", "")).strip()
    if primary not in PRIMARY_REFERENCES:
        blockers.append("primary_wind_reference_unresolved")

    sensor_roles = {
        "sensor1": str(layout.get("sensor1_role", "")).strip(),
        "sensor2": str(layout.get("sensor2_role", "")).strip(),
    }
    freestream_sensor = next(
        (name for name, role in sensor_roles.items() if role == "freestream_reference"),
        None,
    )
    if freestream_sensor is None:
        blockers.append("freestream_reference_sensor_not_assigned")
    if freestream_sensor is not None and primary in PRIMARY_REFERENCES:
        if primary != freestream_sensor:
            blockers.append("primary_reference_must_be_freestream_sensor")
    for key in (
        "freestream_location_documented",
        "outside_model_rotor_and_fixture_influence_confirmed",
        "outside_wall_boundary_layer_confirmed",
        "probe_does_not_obstruct_test_section_confirmed",
        "calibration_current",
        "empty_tunnel_verification_completed",
    ):
        if not bool(layout.get(key, False)):
            blockers.append(f"{key}_not_confirmed")
    frame_enabled = bool(layout.get("frame_diagnostic_enabled", False))
    if frame_enabled:
        if sensor_roles.get("sensor2") != "near_frame_diagnostic":
            frame_blockers.append("sensor2_near_frame_diagnostic_role_not_assigned")
        if observation.sensor2_mps is None:
            frame_blockers.append("near_frame_sensor_value_missing")
        for key in (
            "near_frame_location_documented",
            "near_frame_outside_rotor_slipstream_confirmed",
            "near_frame_probe_does_not_obstruct_flow_confirmed",
            "near_frame_body_relative_position_repeatable",
        ):
            if not bool(layout.get(key, False)):
                frame_blockers.append(f"{key}_not_confirmed")
    else:
        frame_blockers.append("frame_diagnostic_not_enabled")
    target_threshold = float(
        policy.get("pdf_target_deviation_threshold_fraction", 0.03)
    )
    if not 0 < target_threshold < 1:
        blockers.append("target_deviation_threshold_fraction_invalid")

    if observation.sensor2_mps is None:
        pair_mean = None
        pair_difference = None
    else:
        pair_mean = (observation.sensor1_mps + observation.sensor2_mps) / 2.0
        pair_difference = abs(observation.sensor1_mps - observation.sensor2_mps)
    if observation.sensor2_mps is not None and observation.sensor1_mps > 0:
        frame_delta = observation.sensor2_mps - observation.sensor1_mps
        frame_ratio = observation.sensor2_mps / observation.sensor1_mps
        frame_change = frame_delta / observation.sensor1_mps
    else:
        frame_delta = None
        frame_ratio = None
        frame_change = None

    values = {
        "q_derived": observation.q_derived_mps,
        "sensor1": observation.sensor1_mps,
        "sensor2": observation.sensor2_mps,
        "sensor_pair_mean": pair_mean,
    }
    model_speed = values.get(primary)
    if model_speed is None:
        target_error = None
        measured_required = False
    elif observation.target_mps > 0:
        target_error = abs(model_speed - observation.target_mps) / observation.target_mps
        measured_required = target_error > target_threshold
        if measured_required:
            warnings.append("target_deviation_exceeds_pdf_3pct_use_measured_wind")
    else:
        target_error = None
        measured_required = model_speed > 0
        if measured_required:
            warnings.append("nonzero_measured_wind_during_zero_wind_condition")

    q_to_pair = abs(observation.q_derived_mps - observation.sensor1_mps)
    q_pair_warning = float(policy.get("q_to_sensor_pair_warning_mps", 0.0))
    if q_pair_warning > 0 and q_to_pair > q_pair_warning:
        warnings.append("q_derived_and_sensor_pair_mean_differ")
    frame_warning_fraction = float(
        policy.get("frame_speed_change_warning_fraction", 0.0)
    )
    if (
        frame_change is not None
        and frame_warning_fraction > 0
        and abs(frame_change) > frame_warning_fraction
    ):
        warnings.append("near_frame_speed_differs_from_freestream")

    return WindQualityResult(
        ready_for_identification=not blockers,
        ready_for_frame_diagnostic=not frame_blockers,
        target_mps=observation.target_mps,
        q_derived_mps=observation.q_derived_mps,
        sensor1_mps=observation.sensor1_mps,
        sensor2_mps=observation.sensor2_mps,
        sensor_pair_mean_mps=pair_mean,
        sensor_pair_difference_mps=pair_difference,
        frame_speed_delta_mps=frame_delta,
        frame_speed_ratio=frame_ratio,
        frame_speed_change_fraction=frame_change,
        primary_reference=primary,
        model_wind_speed_mps=model_speed,
        target_error_fraction=target_error,
        measured_value_required_by_pdf=measured_required,
        blockers=tuple(blockers),
        frame_diagnostic_blockers=tuple(frame_blockers),
        warnings=tuple(warnings),
    )
