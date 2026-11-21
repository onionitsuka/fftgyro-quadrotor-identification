"""Published command/voltage -> RPM maps, fitted without flight data.

Kaufmann et al. (2023), Nature, doi:10.1038/s41586-023-06419-4, Eqs. 1, 6.
Wildmann & Wetz (2022), AMT, doi:10.5194/amt-15-5465-2022, Eqs. 1, 2.
Shastry & Paley (2023), IJ RNC, doi:10.1002/rnc.6935, Eqs. 8, 9 (inverted).
Coefficients are recalibrated for the actual motors; paper coefficients are not transferred.
"""

import numpy as np
from scipy.optimize import least_squares, lsq_linear, minimize_scalar
from scipy.signal import lfilter


def basis(command, voltage, kind, exponent=1.0):
    u, v = np.asarray(command), np.asarray(voltage)
    if np.any(u < 0):
        raise ValueError("Normalized motor commands must be nonnegative")
    if kind == "kaufmann2023":
        # V/16 is only numerical scaling; exactly the published five-term span.
        return np.column_stack((np.ones_like(u), v / 16, np.sqrt(u), u, v / 16 * np.sqrt(u)))
    if kind == "wildmann2022":
        return (v * u)[:, None]
    if kind == "current_power":
        return (v[:, None] / 16) ** exponent * np.column_stack((u, np.sqrt(u), u * u))
    raise ValueError(kind)


def steady_rpm(command, voltage, model):
    """No measured RPM enters this function; result is RPM, not rad/s."""
    u, v = np.asarray(command), np.asarray(voltage)
    if model["kind"] == "shastry2023":
        a, b, v0 = model["coefficients"]
        effective = np.maximum(u * v - v0, 0)
        return 5000 * 2 * effective / (b + np.sqrt(b * b + 4 * a * effective))
    return basis(u, v, model["kind"], model.get("voltage_exponent", 1.0)) @ np.asarray(model["coefficients"])


def fit_static(command, voltage, rpm, weight, kind):
    """weight is a least-squares row multiplier, not a statistical variance."""
    u, v, y, w = map(np.asarray, (command, voltage, rpm, weight))
    if not (np.isfinite(np.column_stack((u, v, y, w))).all() and np.all(w > 0)):
        raise ValueError("Finite calibration data with positive weights required")
    if kind == "shastry2023":
        x = np.column_stack(((y / 5000) ** 2, y / 5000, np.ones_like(y)))
        upper = float(np.min(u * v) * .999)
        initial = lsq_linear(x * w[:, None], u * v * w,
                             bounds=([1e-8, 1e-8, 0], [np.inf, np.inf, upper])).x
        def residual(c):
            return (steady_rpm(u, v, {"kind": kind, "coefficients": c}) - y) * w
        fit = least_squares(residual, initial, bounds=([1e-8, 1e-8, 0], [np.inf, np.inf, upper]),
                            x_scale="jac", max_nfev=300)
        if not fit.success:
            raise RuntimeError(fit.message)
        return {"kind": kind, "coefficients": fit.x.tolist(), "rpm_scale": 5000,
                "voltage_equation": "d*V=A*(n/5000)^2+B*(n/5000)+V0",
                "fit_cost": float(np.sum(residual(fit.x) ** 2))}

    def solve(gamma):
        x = basis(u, v, kind, gamma) * w[:, None]
        scale = np.linalg.norm(x, axis=0)
        z = x / scale
        if kind == "kaufmann2023":
            c, _, rank, singular = np.linalg.lstsq(z, y * w, rcond=None)
        else:
            c = lsq_linear(z, y * w, bounds=(0, np.inf), tol=1e-10).x
            singular = np.linalg.svd(z, compute_uv=False)
            rank = np.linalg.matrix_rank(z)
        coef = c / scale
        return coef, float(np.sum((x @ coef - y * w) ** 2)), int(rank), singular

    gamma = 1.0
    if kind == "current_power":
        fit = minimize_scalar(lambda g: solve(g)[1], bounds=(-2, 3), method="bounded")
        gamma = float(fit.x)
    coef, cost, rank, singular = solve(gamma)
    result = {"kind": kind, "coefficients": coef.tolist(), "fit_cost": cost,
              "rank": rank, "scaled_condition_number": float(singular[0] / singular[-1])}
    if kind == "current_power":
        result["voltage_exponent"] = gamma
    if kind == "kaufmann2023":
        result["basis"] = ["1", "V/16", "sqrt(d)", "d", "V/16*sqrt(d)"]
        result["coefficients_volts_basis"] = (coef / np.array([1, 16, 1, 1, 16])).tolist()
    return result


def valid_blocks(valid):
    edges = np.flatnonzero(np.diff(np.r_[False, valid, False]))
    return list(zip(edges[::2], edges[1::2]))


def first_order(goal, dt, tau, delay=0.0):
    """Causal exact ZOH recursion with map initialization, then fractional delay.

    Uses the same sample convention as the established 100-Hz analysis. No RPM
    observations or future commands initialize/reset the simulated state.
    """
    if tau <= 0 or delay < 0 or dt <= 0:
        raise ValueError("Positive dt/tau and nonnegative delay required")
    g = np.asarray(goal)
    alpha = np.exp(-dt / tau)
    y = lfilter([1-alpha], [1, -alpha], g, axis=0, zi=(alpha*g[0])[None, ...])[0]
    n = np.arange(len(y)) - delay / dt
    lo = np.floor(n).astype(int)
    f = n - lo
    return (1-f)[:, None]*y[np.clip(lo, 0, len(y)-1)] + f[:, None]*y[np.clip(lo+1, 0, len(y)-1)]


def predict(command, voltage, models, dt=.01, tau=.053, delay=0.0, warmup=2.0):
    u, v = np.asarray(command), np.asarray(voltage)
    result = np.full_like(u, np.nan, dtype=float)
    warm = np.zeros(len(u), bool)
    valid = np.isfinite(u).all(axis=1) & (u >= 0).all(axis=1) & np.isfinite(v) & (v > 5)
    for start, end in valid_blocks(valid):
        goals = np.column_stack([steady_rpm(u[start:end, j], v[start:end], m) for j, m in enumerate(models)])
        result[start:end] = first_order(goals, dt, tau, delay)
        warm[start+int(np.ceil(warmup/dt)):end] = True
    return result, warm
