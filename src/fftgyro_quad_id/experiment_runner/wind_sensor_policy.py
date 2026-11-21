from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ProtocolWindSensorPolicy:
    protocol: str
    freestream_sensor_required: bool
    near_frame_sensor_mode: str
    near_frame_probe_must_be_removed: bool
    use_for_sitl_identification: bool
    purpose: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


POLICIES = {
    "frame_diagnostic_fixed": ProtocolWindSensorPolicy(
        protocol="frame_diagnostic_fixed",
        freestream_sensor_required=True,
        near_frame_sensor_mode="diagnostic_installed_with_probe_ab_reference",
        near_frame_probe_must_be_removed=False,
        use_for_sitl_identification=False,
        purpose="quantify local frame flow and near-probe interference",
    ),
    "phase4_static": ProtocolWindSensorPolicy(
        protocol="phase4_static",
        freestream_sensor_required=True,
        near_frame_sensor_mode="forbidden_in_main_run",
        near_frame_probe_must_be_removed=True,
        use_for_sitl_identification=True,
        purpose="body and rig aerodynamic benchmark without probe contamination",
    ),
    "phase6a_static": ProtocolWindSensorPolicy(
        protocol="phase6a_static",
        freestream_sensor_required=True,
        near_frame_sensor_mode="forbidden_in_main_run",
        near_frame_probe_must_be_removed=True,
        use_for_sitl_identification=True,
        purpose="fixed wind gray-box identification without probe contamination",
    ),
    "phase6b_dynamic": ProtocolWindSensorPolicy(
        protocol="phase6b_dynamic",
        freestream_sensor_required=True,
        near_frame_sensor_mode="forbidden_in_dynamic_run",
        near_frame_probe_must_be_removed=True,
        use_for_sitl_identification=True,
        purpose="dynamic wind identification with freestream only",
    ),
    "phase6c_dynamic": ProtocolWindSensorPolicy(
        protocol="phase6c_dynamic",
        freestream_sensor_required=True,
        near_frame_sensor_mode="forbidden_in_dynamic_run",
        near_frame_probe_must_be_removed=True,
        use_for_sitl_identification=True,
        purpose="coupled dynamic validation with freestream only",
    ),
}


def get_wind_sensor_policy(protocol: str) -> ProtocolWindSensorPolicy:
    try:
        return POLICIES[protocol]
    except KeyError as exc:
        raise ValueError(f"unknown wind sensor policy: {protocol}") from exc
