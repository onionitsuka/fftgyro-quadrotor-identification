"""Offline discrete BO of four-run training blocks, not apparatus control.

Rewards are changes on one frozen development set, normalized by its initial
axis RMSE. Training size is a covariate because marginal gains change as T grows.
The small fixed-kernel GP is a planning approximation, not a calibrated oracle.
"""

from __future__ import annotations

import numpy as np
from scipy.special import ndtr


CONDITIONS = [(axis, wind) for wind in (0, 5, 10) for axis in ("Pitch", "Yaw")]
SEQUENCE = ["F-before", "D-slow", "D-fast", "F-after"]


def features(axis: str, wind: int, training_runs: int) -> list[float]:
    return [float(axis == "Yaw"), wind / 10, training_runs / 48]


def kernel(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    distance = ((a[:, None, :] - b[None, :, :]) / [0.45, 0.5, 0.5]) ** 2
    return 0.05**2 * np.exp(-0.5 * distance.sum(axis=2))


def validate(history: dict) -> list[dict]:
    if history.get("evaluation_role") != "D":
        raise ValueError("Only D may guide acquisition; H must remain sealed")
    if not history.get("evaluation_id"):
        raise ValueError("A frozen development-set identifier is required")
    records = history.get("observations", [])
    if not records and history.get("baseline_rmse") is None:
        return records
    baseline = np.asarray(history["baseline_rmse"], dtype=float)
    if baseline.shape != (2,) or not np.isfinite(baseline).all() or (baseline <= 0).any():
        raise ValueError("baseline_rmse must contain positive Pitch, Yaw values")
    for index, row in enumerate(records):
        if (row["axis"], row["wind_m_s"]) not in CONDITIONS:
            raise ValueError("Unknown condition; do not invent waveform limits")
        if row.get("role") != "T" or row.get("qc_pass") is not True:
            raise ValueError("Only complete QC-passed T blocks may be rewarded")
        if row.get("evaluation_id") != history["evaluation_id"]:
            raise ValueError("Development set changed during optimization")
        if row.get("powered_runs") != 4 or row.get("training_runs_before") != 4 * index:
            raise ValueError("History must be chronological, complete four-run A blocks")
        for key in ("rmse_before", "rmse_after", "gain_se"):
            value = np.asarray(row[key], dtype=float)
            if value.shape != (2,) or not np.isfinite(value).all() or (value < 0).any():
                raise ValueError(f"Invalid {key}: requires nonnegative Pitch, Yaw pair")
        if index and not np.allclose(row["rmse_before"], records[index - 1]["rmse_after"]):
            raise ValueError("RMSE trajectory is discontinuous")
        if index == 0 and not np.allclose(row["rmse_before"], baseline):
            raise ValueError("First evaluation must match the fixed baseline")
    return records


def propose(history: dict, max_extra_blocks: int = 6, threshold: float = 0.01) -> dict:
    """Recommend one batch. Threshold is fraction of initial D RMSE, not percent points."""
    if max_extra_blocks < 0 or not 0 < threshold < 1:
        raise ValueError("Invalid budget or meaningful-improvement threshold")
    rows = validate(history)
    n = len(rows)
    seen = {(row["axis"], row["wind_m_s"]) for row in rows}
    remaining = [c for c in CONDITIONS if c not in seen]
    counts = {
        "mandatory_T": 24, "fixed_M": 9, "development_D": 8,
        "sealed_H": 24, "separate_HT": 30, "Z_S": "separate; not powered",
        "base_graybox": 65, "base_full_scope": 95,
        "maximum_full_scope": 95 + 4 * max_extra_blocks,
        "completed_T": 4 * n, "full_scope_with_mandatory_coverage": 71 + 4 * (n + len(remaining)),
    }
    result = {
        "status": "PREPARATION_ONLY", "runtime_qualified": False,
        "counts": counts, "minimum_run_count_proven": False,
        "objective": "equal-weight normalized D RMSE gain, both axes",
        "evaluation_scope": "fixed D at 5 m/s; not all-wind validation",
        "meaningful_gain_fraction_of_initial_rmse": threshold,
        "candidate_ranking": [], "next_block": None,
    }
    if remaining:
        result["decision"] = "BOOTSTRAP_COVERAGE"
        selected = remaining[0]
    elif n >= 6 + max_extra_blocks:
        result["decision"] = "BUDGET_LIMIT_NOT_PRECISION_SUCCESS"
        return result
    else:
        baseline = np.asarray(history["baseline_rmse"])
        gains = np.asarray([(np.array(r["rmse_before"]) - r["rmse_after"]) / baseline
                            for r in rows])
        noise = np.asarray([np.array(r["gain_se"]) / baseline for r in rows])
        x = np.asarray([features(r["axis"], r["wind_m_s"], r["training_runs_before"])
                        for r in rows])
        candidates = np.asarray([features(a, w, 4 * n) for a, w in CONDITIONS])
        cross = kernel(candidates, x)
        mean, variance = [], []
        for axis_index in range(2):
            covariance = kernel(x, x) + np.diag(noise[:, axis_index]**2 + 1e-8)
            mean.append(cross @ np.linalg.solve(covariance, gains[:, axis_index]))
            variance.append(np.maximum(0, np.diag(kernel(candidates, candidates)) -
                                      (cross * np.linalg.solve(covariance, cross.T).T).sum(1)))
        means = np.asarray(mean).T
        variances = np.asarray(variance).T
        # Independent output GPs: joint covariance is not estimated from few batches.
        joint_mean = means.mean(axis=1)
        joint_sd = np.sqrt(variances.sum(axis=1)) / 2
        z = (joint_mean - threshold) / np.maximum(joint_sd, 1e-12)
        expected = (joint_mean - threshold) * ndtr(z) + joint_sd * np.exp(-z*z/2) / np.sqrt(2*np.pi)
        # Penalize a candidate when either output risks meaningful degradation.
        stable_probability = ndtr((means + threshold) / np.maximum(np.sqrt(variances), 1e-12)).prod(1)
        acquisition = np.maximum(expected, 0) * stable_probability / 4
        for index, (axis, wind) in enumerate(CONDITIONS):
            result["candidate_ranking"].append({
                "axis": axis, "wind_m_s": wind,
                "predicted_normalized_gain": means[index].tolist(),
                "gain_upper_95": (means[index] + 1.96*np.sqrt(variances[index])).tolist(),
                "expected_joint_gain_per_run": float(acquisition[index]),
            })
        result["candidate_ranking"].sort(key=lambda r: r["expected_joint_gain_per_run"], reverse=True)
        plateau = np.all(gains[-3:] < threshold)
        upper_small = np.all(means + 1.96 * np.sqrt(variances) < threshold)
        if n >= 9 and plateau and upper_small:
            result["decision"] = "STOP_ADDITIONAL_T_HEURISTIC"
            return result
        result["decision"] = "BO_SELECT_ADDITIONAL_T"
        best = result["candidate_ranking"][0]
        selected = (best["axis"], best["wind_m_s"])
    result["next_block"] = {
        "axis": selected[0], "wind_m_s": selected[1], "role": "T",
        "location": "outside_wind_tunnel" if selected[1] == 0 else "wind_tunnel",
        "powered_runs": 4, "sequence": SEQUENCE,
        "qualified_session_paths": None,
        "constraint": "same battery/configuration/location/wind; Z/S and QC retained",
    }
    return result
