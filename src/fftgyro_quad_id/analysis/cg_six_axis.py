"""Six-axis static gravity calibration with explicit shared bias epochs.

Sensor FLU, load on sensor; independent mass/pose gravity and fixed CG height.
With gravity known the force and moment blocks are algebraically separable:
adding force readings checks the calibration but cannot invent CG information.
"""

import numpy as np
from scipy.optimize import lsq_linear


def fit_six_axis(gravity, wrench, groups, epochs, nominal, bound_mm=15.0):
    gravity, wrench, nominal = map(lambda x: np.asarray(x, float), (gravity, wrench, nominal))
    n = len(gravity)
    if gravity.shape != (n, 3) or wrench.shape != (n, 6) or nominal.shape != (3,) or n == 0:
        raise ValueError("Expected nonempty gravity Nx3, wrench Nx6, nominal (3,)")
    if (
        len(groups) != n
        or len(epochs) != n
        or not all(isinstance(v, str) and v for v in [*groups, *epochs])
    ):
        raise ValueError("Explicit group/epoch per pose required")
    if not all(np.isfinite(v).all() for v in (gravity, wrench, nominal)):
        raise ValueError("Nonfinite data")
    if not np.isfinite(bound_mm) or bound_mm <= 0:
        raise ValueError("Positive diagnostic CG bound required")
    gs, es = list(dict.fromkeys(groups)), list(dict.fromkeys(epochs))
    k = 2 * len(gs)
    design = np.zeros((n, 6, k + 6 * len(es)))
    base = np.column_stack((gravity, np.cross(nominal, gravity)))
    for i, (g, e) in enumerate(zip(groups, epochs, strict=True)):
        j, b = 2 * gs.index(g), k + 6 * es.index(e)
        design[i, 3:, j] = np.cross([1, 0, 0], gravity[i]) / 1000
        design[i, 3:, j + 1] = np.cross([0, 1, 0], gravity[i]) / 1000
        design[i, :, b : b + 6] = np.eye(6)
    # Unit scales only. Blocks are independent given mass/pose and zero cross covariance.
    scale = np.array([0.1] * 3 + [0.005] * 3)
    a = (design / scale[None, :, None]).reshape(n * 6, -1)
    y = ((wrench - base) / scale).ravel()
    lower = np.r_[np.full(k, -bound_mm), np.full(6 * len(es), -np.inf)]
    fit = lsq_linear(a, y, bounds=(lower, -lower), tol=1e-12, max_iter=500)
    if not fit.success:
        raise RuntimeError(fit.message)
    residual = wrench - base - (design @ fit.x)
    delta = fit.x[:k].reshape(-1, 2)
    return {
        "delta_xy_mm": dict(zip(gs, delta.tolist(), strict=True)),
        "bias_6": dict(zip(es, fit.x[k:].reshape(-1, 6).tolist(), strict=True)),
        "rmse_6": np.sqrt(np.mean(residual**2, axis=0)).tolist(),
        "residual_6": residual.tolist(),
        "bound_hits": int(np.sum(np.abs(delta) >= bound_mm - 1e-5)),
        "cg_bound_mm": bound_mm,
        "nominal_cg_m": nominal.tolist(),
    }


def predict_six_axis(gravity, group, epoch, fit):
    gravity = np.asarray(gravity, float)
    delta = np.r_[np.asarray(fit["delta_xy_mm"][group]) / 1000, 0]
    r = np.array(fit["nominal_cg_m"]) + delta
    return np.r_[gravity, np.cross(r, gravity)] + fit["bias_6"][epoch]
