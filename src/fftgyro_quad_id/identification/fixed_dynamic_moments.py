"""Staged, fixed-test-anchored moment models with explicit observation targets.

All moments are FRD-positive, about the aircraft CG. Static coefficients are
estimated using FIXED_INPUT T runs only; D selects complexity. Dynamic terms
vanish at zero rate. HOLD subtraction is a dimensional difference, not a tare.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict
from itertools import product

import numpy as np

from fftgyro_quad_id.frames.rig_kinematics import airspeed_frd
from fftgyro_quad_id.identification.sun_static import SunStaticConfig


def states(
    rpm_motor: np.ndarray,
    wind: float,
    pitch: np.ndarray,
    heading: np.ndarray,
    roll: np.ndarray,
    rate: np.ndarray,
    *,
    inner_yaw=0.0,
) -> dict[str, np.ndarray]:
    """Sun inputs using separate fixture heading and absolute inner Yaw."""
    cfg = SunStaticConfig()
    rpm = np.asarray(rpm_motor, float)
    if rpm.ndim != 2 or rpm.shape[1] != 4 or not np.isfinite(rpm).all():
        raise ValueError("finite Nx4 MOTOR-ORDER RPM required")
    if np.any(rpm <= 0):
        raise ValueError("all four rotors must have positive RPM")
    omega2 = (rpm * (2 * np.pi / 60)) ** 2
    rms2 = omega2.mean(axis=1)
    rms = np.sqrt(rms2)
    normalized = omega2 / rms2[:, None]
    u, v, w = np.moveaxis(airspeed_frd(wind, pitch, inner_yaw, heading, roll), -1, 0)
    denominator = rms * cfg.rotor_radius_m
    force_scale = (
        cfg.air_density_kg_m3 * cfg.rotor_count * np.pi * cfg.rotor_radius_m**2 * denominator**2
    )
    return {
        "Q": cfg.reference_length_m * force_scale,
        "omega2": rms2,
        "mu_x": u / denominator,
        "mu_y": v / denominator,
        "mu_z": w / denominator,
        "uq": normalized @ np.array([1.0, -1.0, 1.0, -1.0]),
        "ur": normalized @ np.array([1.0, 1.0, -1.0, -1.0]),
        "rate_bar": np.asarray(rate) * cfg.reference_length_m / denominator,
        "u_m_s": u,
        "v_m_s": v,
        "w_m_s": w,
    }


def _polynomials(s: Mapping[str, np.ndarray], axis: str, degree: int):
    if axis not in {"PITCH", "YAW"}:
        raise ValueError(axis)
    x, y, z = s["mu_x"], s["mu_y"], s["mu_z"]
    for i, j, k in product(range(degree + 1), repeat=3):
        # Pitch: Sun P5(mu_x,mu_z) P2(abs(mu_y)). Yaw: P5(mu_x,mu_y,mu_z).
        if axis == "PITCH":
            if i + k > degree or j > min(2, degree):
                continue
        elif i + j + k > degree:
            continue
        lateral = np.abs(y) if axis == "PITCH" else y
        yield (i, j, k), x**i * lateral**j * z**k


def static_design(s: Mapping[str, np.ndarray], axis: str, degree: int = 5):
    """Dimensional Sun static basis; the pure squared-difference term is protected."""
    inp = s["uq" if axis == "PITCH" else "ur"]
    columns, names = [], []
    for powers, values in _polynomials(s, axis, degree):
        for exponent in range(2 if axis == "PITCH" else 4):
            if powers == (0, 0, 0) and exponent == 1:
                continue  # Supplied independently through KF*arm or KQ.
            names.append(f"s:{powers[0]},{powers[1]},{powers[2]}:{exponent}")
            columns.append(s["Q"] * values * inp**exponent)
    return np.column_stack(columns), names


def dynamic_design(s: Mapping[str, np.ndarray], axis: str, degree: int = 1):
    """Rate-dependent Sun subset; no constant, angle-only or acceleration term."""
    inp = s["uq" if axis == "PITCH" else "ur"]
    columns, names = [], []
    for powers, values in _polynomials(s, axis, degree):
        for exponent in range(1 if degree == 0 else 2):
            names.append(f"d:{powers[0]},{powers[1]},{powers[2]}:{exponent}")
            columns.append(s["Q"] * values * inp**exponent * s["rate_bar"])
    return np.column_stack(columns), names


def physical_moment(s: Mapping[str, np.ndarray], axis: str, kf: float, kq: float):
    gain = kf * SunStaticConfig().reference_length_m if axis == "PITCH" else kq
    return gain * s["omega2"] * s["uq" if axis == "PITCH" else "ur"]


def observed_moment(
    support: np.ndarray,
    acceleration: np.ndarray,
    spin_reaction_acceleration: np.ndarray,
    inertia: float,
    ip: float,
):
    """I*acceleration - support - rotor inertial reaction (not spin H sign)."""
    return inertia * acceleration - support - ip * spin_reaction_acceleration


def run_weights(runs: np.ndarray) -> np.ndarray:
    counts = Counter(map(str, runs))
    return np.asarray([len(runs) / (len(counts) * counts[str(run)]) for run in runs])


def metrics(observed: np.ndarray, predicted: np.ndarray, runs: np.ndarray) -> dict:
    if not len(observed):
        return {"samples": 0, "runs": 0}
    residual = observed - predicted
    centered = observed - observed.mean()
    sst = float(centered @ centered)
    rmses = [float(np.sqrt(np.mean(residual[runs == run] ** 2))) for run in np.unique(runs)]
    return {
        "samples": len(observed),
        "runs": len(rmses),
        "rmse_Nm": float(np.sqrt(np.mean(residual**2))),
        "run_equal_rmse_Nm": float(np.sqrt(np.mean(np.square(rmses)))),
        "bias_Nm": float(np.mean(residual)),
        "mae_Nm": float(np.mean(np.abs(residual))),
        "r2": 1 - float(residual @ residual) / sst if sst > 1e-20 else None,
    }


def fit_ridge(x: np.ndarray, y: np.ndarray, runs: np.ndarray, alpha: float):
    weights = run_weights(runs)
    scale = np.sqrt(np.average(x * x, axis=0, weights=weights))
    active = scale > max(float(scale.max(initial=0)) * 1e-12, 1e-15)
    scale = np.where(active, scale, 1.0)
    z = x[:, active] / scale[active]
    gram = (z.T * weights) @ z / weights.sum()
    rhs = (z.T * weights) @ y / weights.sum()
    beta = np.zeros(x.shape[1])
    beta[active] = np.linalg.lstsq(gram + alpha * np.eye(z.shape[1]), rhs, rcond=1e-12)[0]
    beta /= scale
    singular = np.linalg.svd(z * np.sqrt(weights[:, None]), compute_uv=False)
    rank = int(np.sum(singular > (singular[0] * 1e-10))) if len(singular) else 0
    return beta, {
        "active_columns": int(active.sum()),
        "standardized_rank": rank,
        "standardized_singular_values": singular.tolist(),
        "standardized_condition_number": (
            float(singular[0] / singular[-1]) if len(singular) and singular[-1] > 0 else None
        ),
    }


def feature_subset(names: list[str], axis: str, degree: int) -> np.ndarray:
    selected = []
    for i, name in enumerate(names):
        powers = tuple(map(int, name.split(":")[1].split(",")))
        allowed = (
            powers[0] + powers[2] <= degree and powers[1] <= min(2, degree)
            if axis == "PITCH"
            else sum(powers) <= degree
        )
        if allowed:
            selected.append(i)
    return np.asarray(selected, dtype=int)


def select_model(
    x: np.ndarray,
    target: np.ndarray,
    base: np.ndarray,
    roles: np.ndarray,
    runs: np.ndarray,
    names: list[str],
    axis: str,
    *,
    dynamic: bool = False,
) -> dict:
    """Fit T, select by D only. H target values are never read here."""
    train, dev = roles == "T", roles == "D"
    if not train.any() or not dev.any():
        raise ValueError("T and D are required")
    trials = [
        {
            "degree": -1,
            "ridge": None,
            "D_run_equal_rmse_Nm": metrics(target[dev], base[dev], runs[dev])["run_equal_rmse_Nm"],
        }
    ]
    best = {"coefficients": np.zeros(len(names)), "trial": trials[0], "diagnostics": {}}
    for degree in (0, 1, 2) if dynamic else (1, 2, 3, 5):
        indices = feature_subset(names, axis, degree)
        if dynamic and degree == 0:
            # Pure rate only; rate*input is reserved for richer models.
            indices = np.asarray([i for i in indices if names[i].endswith(":0")])
        for alpha in (1e-6, 1e-4, 1e-2, 0.1, 1.0, 10.0):
            beta, diagnostics = fit_ridge(
                x[train][:, indices], (target - base)[train], runs[train], alpha
            )
            prediction = base[dev] + x[dev][:, indices] @ beta
            score = metrics(target[dev], prediction, runs[dev])["run_equal_rmse_Nm"]
            trial = {"degree": degree, "ridge": alpha, "D_run_equal_rmse_Nm": score}
            trials.append(trial)
            if score < best["trial"]["D_run_equal_rmse_Nm"]:
                full = np.zeros(len(names))
                full[indices] = beta
                best = {"coefficients": full, "trial": trial, "diagnostics": diagnostics}
    return {
        "axis": axis,
        "feature_names": names,
        "coefficients": best["coefficients"].tolist(),
        "selected": best["trial"],
        "diagnostics": best["diagnostics"],
        "trials": trials,
        "selection": "T coefficients; D run-equal RMSE; H never used for selection",
        "normalization": asdict(SunStaticConfig()),
    }
