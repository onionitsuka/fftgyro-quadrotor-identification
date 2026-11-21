"""Conditional, fixed-height CG identification from wind-off stopped poses.

All vectors are in the same sensor FLU frame. Wrench convention is load on
sensor: M = bias + r x gravity_force. Gravity is predicted from independent
mass/pose, never the biased force readout. Bounds are physical assumptions,
not additional observations. No powered/aerodynamic data are accepted here.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linprog, lsq_linear


def fit_stopped_cg_xy(
    gravity_force_n,
    moment_nm,
    configuration_ids,
    zero_epoch_ids,
    nominal_cg_m,
    xy_bounds_m,
    moment_tolerance_nm,
    measured_force_n=None,
    force_tolerance_n=None,
) -> dict:
    """Fit xy per configuration and all available zero-bias components per epoch.

    tolerance is a per-component bound on ALL observation/model error, not an
    automatically inferred confidence interval. Profile intervals solve LPs
    with all other CGs/biases free within their stated bounds. The input must
    already have passed independent stopped/wind-off/quality checks.
    """
    force = np.asarray(gravity_force_n, dtype=float)
    moment = np.asarray(moment_nm, dtype=float)
    nominal = np.asarray(nominal_cg_m, dtype=float)
    bounds = np.asarray(xy_bounds_m, dtype=float)
    if force.ndim != 2 or force.shape[1] != 3 or not len(force):
        raise ValueError("gravity_force_n must be nonempty N x 3")
    n = len(force)
    if moment.shape != force.shape or nominal.shape != (3,) or bounds.shape != (2, 2):
        raise ValueError("Expected moments N x 3, nominal CG (3,), xy bounds (2,2)")
    if not all(np.isfinite(x).all() for x in (force, moment, nominal, bounds)):
        raise ValueError("Nonfinite input")
    if np.any(bounds[:, 0] >= bounds[:, 1]):
        raise ValueError("Each CG lower bound must be below its upper bound")
    if len(configuration_ids) != n or len(zero_epoch_ids) != n:
        raise ValueError("One explicit configuration and zero epoch per pose required")
    if any(not isinstance(x, str) or not x for x in [*configuration_ids, *zero_epoch_ids]):
        raise ValueError("IDs must be nonempty strings; unknown grouping needs separate hypotheses")
    tolerance = np.broadcast_to(np.asarray(moment_tolerance_nm, float), (n, 3)).copy()
    if not np.isfinite(tolerance).all() or np.any(tolerance <= 0):
        raise ValueError("Moment tolerance must be finite and positive")
    groups = list(dict.fromkeys(configuration_ids))
    epochs = list(dict.fromkeys(zero_epoch_ids))
    force_result = None
    if measured_force_n is not None:
        measured_force = np.asarray(measured_force_n, dtype=float)
        if measured_force.shape != force.shape or not np.isfinite(measured_force).all():
            raise ValueError("measured_force_n must be finite N x 3")
        if force_tolerance_n is None:
            raise ValueError("force_tolerance_n required with measured_force_n")
        force_tolerance = np.broadcast_to(np.asarray(force_tolerance_n, float), (n, 3)).copy()
        if not np.isfinite(force_tolerance).all() or np.any(force_tolerance <= 0):
            raise ValueError("force_tolerance_n must be finite and positive")
        force_result = {}
        for epoch in epochs:
            mask = np.asarray(zero_epoch_ids) == epoch
            delta = measured_force[mask] - force[mask]
            lower_force = np.max(delta - force_tolerance[mask], axis=0)
            upper_force = np.min(delta + force_tolerance[mask], axis=0)
            feasible_force = bool(np.all(lower_force <= upper_force + 1e-12))
            candidate = (lower_force + upper_force) / 2
            residual_force = delta - candidate
            normalized = np.abs(residual_force) / force_tolerance[mask]
            force_result[epoch] = {
                "bias_n": candidate.tolist() if feasible_force else None,
                "feasible_bias_interval_n": (
                    np.column_stack((lower_force, upper_force)).tolist() if feasible_force else None
                ),
                "maximum_normalized_residual": float(np.max(normalized)),
                "consistent_with_tolerance": feasible_force,
                "pose_count": int(np.count_nonzero(mask)),
            }
    k = 2 * len(groups)
    cg = np.zeros((n, 3, k))
    bias = np.zeros((n, 3, 3 * len(epochs)))
    for i, (group, epoch) in enumerate(zip(configuration_ids, zero_epoch_ids, strict=True)):
        j = 2 * groups.index(group)
        cg[i, :, j] = np.cross([1.0, 0.0, 0.0], force[i])
        cg[i, :, j + 1] = np.cross([0.0, 1.0, 0.0], force[i])
        b = 3 * epochs.index(epoch)
        bias[i, :, b : b + 3] = np.eye(3)
    c = cg.reshape(-1, k)
    b = bias.reshape(-1, 3 * len(epochs))
    y = (moment - np.cross(nominal, force)).ravel()
    w = 1 / tolerance.ravel()
    cw, bw = c * w[:, None], b * w[:, None]
    # Remove the space explainable by arbitrary sensor moment biases.
    projected = cw - bw @ np.linalg.lstsq(bw, cw, rcond=None)[0]
    _, singular, vh = np.linalg.svd(projected, full_matrices=True)
    threshold = max(np.linalg.norm(cw, 2), 1.0) * 1e-10
    rank = int(np.count_nonzero(singular > threshold))
    null = vh[rank:]
    identifiable = np.linalg.norm(null, axis=0) < 1e-8
    design = np.column_stack((c, b))
    lower = np.r_[np.tile(bounds[:, 0], len(groups)), np.full(b.shape[1], -np.inf)]
    upper = np.r_[np.tile(bounds[:, 1], len(groups)), np.full(b.shape[1], np.inf)]
    fit = lsq_linear(design * w[:, None], y * w, bounds=(lower, upper), tol=1e-12)
    if not fit.success:
        raise RuntimeError(f"Bounded least squares failed: {fit.message}")
    residual = design @ fit.x - y
    aw = design * w[:, None]
    yw = y * w
    constraint = np.vstack((aw, -aw))
    limit = np.r_[yw + 1, -yw + 1]
    lp_bounds = list(zip(lower, upper, strict=True))
    feasible = linprog(
        np.zeros(design.shape[1]), A_ub=constraint, b_ub=limit, bounds=lp_bounds, method="highs"
    )
    if feasible.status not in (0, 2):
        raise RuntimeError(f"Feasibility check failed: {feasible.message}")
    intervals = []
    if feasible.success:
        for j in range(k):
            objective = np.zeros(design.shape[1])
            objective[j] = 1
            extrema = []
            for sign in (1, -1):
                solved = linprog(
                    sign * objective, A_ub=constraint, b_ub=limit, bounds=lp_bounds, method="highs"
                )
                if not solved.success:
                    raise RuntimeError(f"Profile failed: {solved.message}")
                extrema.append(float(solved.x[j]))
            intervals.append(extrema)
    at_bound = np.isclose(fit.x[:k], lower[:k], atol=1e-8, rtol=0) | np.isclose(
        fit.x[:k], upper[:k], atol=1e-8, rtol=0
    )
    ls_within_tolerance = bool(np.max(np.abs(residual * w)) <= 1 + 1e-7)
    estimates = {}
    for i, group in enumerate(groups):
        entry = {}
        for axis, j in zip(("x", "y"), (2 * i, 2 * i + 1), strict=True):
            valid = bool(
                feasible.success and ls_within_tolerance and identifiable[j] and not at_bound[j]
            )
            entry[axis] = {
                "data_identifiable": bool(identifiable[j]),
                "delta_estimate_m": float(fit.x[j]) if valid else None,
                "feasible_delta_interval_m": intervals[j] if feasible.success else None,
                "bound_active_at_fit": bool(at_bound[j]),
            }
        estimates[group] = entry
    result = {
        "state": (
            "INCONSISTENT_WITH_TOLERANCE"
            if not feasible.success
            else "PARTIALLY_IDENTIFIED"
            if rank < k
            else "BOUND_ACTIVE"
            if at_bound.any()
            else "FEASIBLE_INTERVAL_ONLY"
            if not ls_within_tolerance
            else "CONDITIONAL_ESTIMATE"
        ),
        "configurations": estimates,
        "nominal_cg_m": nominal.tolist(),
        "fixed_height_m": float(nominal[2]),
        "projected_cg_rank": rank,
        "cg_parameter_count": k,
        "projected_singular_values": singular.tolist(),
        "nullspace_cg_directions": null.tolist(),
        "parameter_order": [f"{g}:delta_{axis}_m" for g in groups for axis in ("x", "y")],
        "residual_rmse_nm": float(np.sqrt(np.mean(residual**2))),
        "maximum_normalized_residual": float(np.max(np.abs(residual * w))),
        "moment_bias_nm": (
            {e: fit.x[k + 3 * i : k + 3 * i + 3].tolist() for i, e in enumerate(epochs)}
            if rank == k and feasible.success and ls_within_tolerance
            else None
        ),
        "interval_meaning": (
            "Feasible set under supplied deterministic error/CG bounds; NOT a confidence interval"
        ),
        "assumptions": [
            "Wind-off, rotor-stopped, static, no other aircraft support",
            "Independent mass and pose; correct wrench action sign and FLU frame",
            "Constant sensor bias within each explicit zero epoch",
            "CG height fixed; xy constant within each explicit configuration",
            "No aerodynamic fit or automatic correction of historical runs",
        ],
    }
    if force_result is not None:
        result["force_bias_n"] = force_result
        result["six_axis_equations_consistent"] = bool(
            result["state"] == "CONDITIONAL_ESTIMATE"
            and all(v["consistent_with_tolerance"] for v in force_result.values())
            and all(
                axis["delta_estimate_m"] is not None
                for configuration in estimates.values()
                for axis in configuration.values()
            )
        )
        result["six_axis_note"] = (
            "Equation consistency is not calibration certification: inspect CG profile intervals, "
            "pose excitation, sensor drift, grouping evidence and independent validation."
        )
    return result
