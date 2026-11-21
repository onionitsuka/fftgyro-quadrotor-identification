"""Static multi-pose wrench fit. Coordinates: body FLU, units: N, Nm, m.

No mass or sensor force-zero assumption: M = r cross F + constant bias.
Only pose means are fit; 1200 Hz samples are not independent repetitions.
"""
from __future__ import annotations

import numpy as np

try:
    from fftgyro_quad_id.rig import aircraft_cg as _aircraft_cg
except ModuleNotFoundError:  # Standalone Pi deployment with four files together.
    import aircraft_cg as _aircraft_cg

AIRCRAFT_CG_SOURCE_PATH = _aircraft_cg.__file__
estimate_aircraft_cg = _aircraft_cg.estimate_aircraft_cg
evaluate_axis_alignment = _aircraft_cg.evaluate_axis_alignment

# Confirmed apparatus dimension equation [mm]:
#   57 + upper-surface reference = 2 + requested length - 33
FORMULA_LEFT_FIXED_MM = 57.0
FORMULA_RIGHT_BASE_MM = 2.0
FORMULA_RIGHT_SUBTRACTED_MM = 33.0
# ASM0551012R0A00, SASM055001 p.4: focal depth from the aircraft-side
# upper face is 14.6 mm (mounting-face interpretation corrected by user,
# 2026-10-26). Do not subtract this from the 20.1 mm total housing height.
SENSOR_ORIGIN_BELOW_UPPER_SURFACE_MM = 14.6


def to_flu(values, frame: str) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    if v.shape[-1] != 6 or not np.isfinite(v).all():
        raise ValueError("Expected six finite wrench components")
    if frame == "flu":
        return v.copy()
    if frame == "element":
        return v[..., [1, 0, 2, 4, 3, 5]] * [1, -1, 1, 1, -1, 1]
    raise ValueError("frame must be element or flu")


def correction_mm(cg_m, axis_direction, axis_point_mm) -> np.ndarray:
    """Whole-aircraft translation relative to an arbitrary Frame3 axis line."""
    point_m = tuple(np.asarray(axis_point_mm, dtype=float) * 1.0e-3)
    alignment = evaluate_axis_alignment(tuple(cg_m), tuple(axis_direction), point_m)
    return 1000.0 * np.asarray(alignment.correction_to_aircraft_m)


def upper_surface_height_mm(sensor_origin_height_mm: float) -> float:
    """Convert a fitted S-to-CG height to height above the sensor's upper face."""
    value = float(sensor_origin_height_mm)
    if not np.isfinite(value):
        raise ValueError("sensor_origin_height_mm must be finite")
    return value - SENSOR_ORIGIN_BELOW_UPPER_SURFACE_MM


def requested_length_mm(upper_surface_reference_mm: float) -> float:
    """Calculate the physical adjustment dimension from the confirmed equation."""
    value = float(upper_surface_reference_mm)
    if not np.isfinite(value):
        raise ValueError("upper_surface_reference_mm must be finite")
    return (
        FORMULA_LEFT_FIXED_MM
        + value
        - FORMULA_RIGHT_BASE_MM
        + FORMULA_RIGHT_SUBTRACTED_MM
    )


def fit_cg(wrenches, *, max_condition=30., max_residual_nm=0.01) -> dict:
    w = np.asarray(wrenches, dtype=float)
    if w.size == 0:
        w = np.empty((0, 6))
    if w.ndim != 2 or w.shape[1] != 6 or not np.isfinite(w).all():
        raise ValueError("Expected finite Nx6 pose means")
    result = {"state": "NEED_POSES", "pose_count": len(w), "cg_m": None}
    if len(w) < 3:
        return result
    force = w[:, :3] - w[:, :3].mean(axis=0)
    a = np.array([[[0, z, -y], [-z, 0, x], [y, -x, 0]] for x, y, z in force]).reshape(-1, 3)
    rank = int(np.linalg.matrix_rank(a))
    singular = np.linalg.svd(a, compute_uv=False)
    result["singular_values_N"] = singular.tolist()
    if rank < 3 or singular[-1] < 0.1:
        result["state"] = "NEED_ANGLE_SPAN"
        return result
    condition = float(singular[0] / singular[-1])
    try:
        estimate = estimate_aircraft_cg(w[:, :3], w[:, 3:])
    except ValueError:
        result["state"] = "NEED_ANGLE_SPAN"
        return result
    r = np.asarray(estimate.cg_from_sensor_m)
    bias = np.asarray(estimate.moment_bias_nm)
    errors = w[:, 3:] - np.cross(r, w[:, :3]) - bias
    residual = float(np.max(np.linalg.norm(errors, axis=1)))
    result.update(condition=condition, max_residual_nm=residual, effective_bias_nm=bias.tolist())
    if condition > max_condition:
        result["state"] = "NEED_ANGLE_SPAN"
    elif residual > max_residual_nm:
        result["state"] = "INCONSISTENT_POSES"
    else:
        result.update(
            state="ESTIMATE",
            cg_m=r.tolist(),
            cg_standard_error_m=list(estimate.cg_standard_error_m),
            design_rank=estimate.design_rank,
        )
    return result


def static_mean(samples, *, force_std=0.1, torque_std=0.005) -> tuple:
    """Basic motion/drift screen, not proof of static force or calibrated accuracy."""
    w = np.asarray(samples, dtype=float)
    if w.ndim != 2 or w.shape[1] != 6 or len(w) < 100 or not np.isfinite(w).all():
        return None, "WAIT_SAMPLES"
    thresholds = np.array([force_std] * 3 + [torque_std] * 3)
    if np.any(w.std(axis=0) > thresholds):
        return None, "MOVING_OR_NOISY"
    n = len(w) // 3
    if np.any(np.abs(w[:n].mean(axis=0) - w[-n:].mean(axis=0)) > thresholds * 2):
        return None, "DRIFT"
    return w.mean(axis=0), "STATIC_CANDIDATE"


def display_text(result: dict, age_s: float, *, live: bool) -> str:
    """Format the CG estimate using the confirmed physical adjustment equation."""
    state = result["state"]
    if result.get("cg_m") is None:
        labels = {
            "NEED_POSES": "姿勢不足（3姿勢以上）",
            "NEED_ANGLE_SPAN": "角度の広がり不足",
            "INCONSISTENT_POSES": "姿勢間の荷重が不整合",
        }
        return f"{labels.get(state, state)} | 保存姿勢 {result['pose_count']} | 求めたい長さ --"

    cg_mm = 1000.0 * np.asarray(result["cg_m"], dtype=float)
    front_mm = float(cg_mm[0])
    sensor_origin_mm = float(cg_mm[2])
    upper_surface_mm = upper_surface_height_mm(sensor_origin_mm)
    target_mm = requested_length_mm(upper_surface_mm)
    prefix = "計測中の暫定推定" if live else "保存済み推定（調整後はrで再計測）"
    front_text = f"{'前' if front_mm >= 0 else '後'}へ {abs(front_mm):.1f} mm"
    formula = f"57+({upper_surface_mm:.1f})=2+L-33"
    return (
        f"{prefix} | 重心 {front_text} / S基準 {sensor_origin_mm:+.1f} mm / "
        f"上面基準 {upper_surface_mm:+.1f} mm | "
        f"求めたい長さ L={target_mm:.1f} mm [{formula}] | 更新から {age_s:.1f} s | "
        f"残差 {result['max_residual_nm']:.4f} Nm（精度保証ではない）"
    )
