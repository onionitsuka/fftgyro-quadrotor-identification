"""Estimate aircraft center of gravity from static six-axis wrenches."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class CgEstimate:
    """CG position and fit diagnostics in the wrench sensor coordinate system."""

    cg_from_sensor_m: tuple[float, float, float]
    moment_bias_nm: tuple[float, float, float]
    cg_standard_error_m: tuple[float, float, float]
    moment_residual_rmse_nm: tuple[float, float, float]
    design_rank: int
    design_condition_number: float
    pose_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class AxisAlignment:
    """CG displacement from a rotation axis, in the same coordinates as the CG."""

    axis_unit: tuple[float, float, float]
    axis_point_from_sensor_m: tuple[float, float, float]
    offset_perpendicular_to_axis_m: tuple[float, float, float]
    offset_along_axis_m: float
    correction_to_aircraft_m: tuple[float, float, float]
    perpendicular_distance_m: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _vectors(values: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.ndim != 2 or result.shape[1] != 3:
        raise ValueError(f"{name} must have shape (pose_count, 3)")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _tuple3(values: np.ndarray) -> tuple[float, float, float]:
    vector = np.asarray(values, dtype=float).reshape(3)
    return float(vector[0]), float(vector[1]), float(vector[2])


def _cross_product_design(force_n: np.ndarray) -> np.ndarray:
    """Return matrices C(F) for which C(F) r = r cross F."""

    result = np.zeros((len(force_n), 3, 3))
    fx, fy, fz = force_n.T
    result[:, 0, 1] = fz
    result[:, 0, 2] = -fy
    result[:, 1, 0] = -fz
    result[:, 1, 2] = fx
    result[:, 2, 0] = fy
    result[:, 2, 1] = -fx
    return result


def estimate_aircraft_cg(force_n: np.ndarray, moment_nm: np.ndarray) -> CgEstimate:
    """Fit ``M_measured = r_CG/S cross F_measured + constant_bias``.

    One row must represent one steady pose mean. A constant force offset is
    absorbed by the fitted moment intercept, so a common tare is sufficient.
    The force and moment vectors must use the same sensor coordinate system.
    """

    force = _vectors(force_n, "force_n")
    moment = _vectors(moment_nm, "moment_nm")
    if force.shape != moment.shape:
        raise ValueError("force_n and moment_nm must have the same shape")
    if len(force) < 3:
        raise ValueError("at least three steady poses are required")

    cross = _cross_product_design(force)
    design = np.concatenate((cross, np.broadcast_to(np.eye(3), cross.shape)), axis=2)
    design_2d = design.reshape(-1, 6)
    measured = moment.reshape(-1)
    rank = int(np.linalg.matrix_rank(design_2d))
    if rank < 6:
        raise ValueError(
            "CG is not identifiable: use more distinct orientations with one common tare"
        )

    parameters, _, _, singular_values = np.linalg.lstsq(design_2d, measured, rcond=None)
    predicted = (design_2d @ parameters).reshape(-1, 3)
    residual = moment - predicted
    degrees_of_freedom = measured.size - parameters.size
    standard_error = np.full(3, np.nan)
    if degrees_of_freedom > 0:
        residual_variance = float(np.sum(residual**2) / degrees_of_freedom)
        covariance = residual_variance * np.linalg.inv(design_2d.T @ design_2d)
        standard_error = np.sqrt(np.diag(covariance)[:3])

    return CgEstimate(
        cg_from_sensor_m=_tuple3(parameters[:3]),
        moment_bias_nm=_tuple3(parameters[3:]),
        cg_standard_error_m=_tuple3(standard_error),
        moment_residual_rmse_nm=_tuple3(np.sqrt(np.mean(residual**2, axis=0))),
        design_rank=rank,
        design_condition_number=float(singular_values[0] / singular_values[-1]),
        pose_count=len(force),
    )


def evaluate_axis_alignment(
    cg_from_sensor_m: tuple[float, float, float],
    axis_direction: tuple[float, float, float],
    axis_point_from_sensor_m: tuple[float, float, float],
) -> AxisAlignment:
    """Project the estimated CG onto an arbitrary rotation-axis line."""

    cg = np.asarray(cg_from_sensor_m, dtype=float)
    direction = np.asarray(axis_direction, dtype=float)
    point = np.asarray(axis_point_from_sensor_m, dtype=float)
    if not np.all(np.isfinite(np.concatenate((cg, direction, point)))):
        raise ValueError("CG and axis geometry must contain only finite values")
    norm = float(np.linalg.norm(direction))
    if norm <= 0.0:
        raise ValueError("axis_direction must be non-zero")
    axis = direction / norm
    displacement = cg - point
    along = float(displacement @ axis)
    perpendicular = displacement - along * axis
    correction = -perpendicular
    return AxisAlignment(
        axis_unit=_tuple3(axis),
        axis_point_from_sensor_m=_tuple3(point),
        offset_perpendicular_to_axis_m=_tuple3(perpendicular),
        offset_along_axis_m=along,
        correction_to_aircraft_m=_tuple3(correction),
        perpendicular_distance_m=float(np.linalg.norm(perpendicular)),
    )
