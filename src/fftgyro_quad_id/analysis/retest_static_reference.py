"""Conditional static calibration in physical sensor axes; not an accuracy certificate.

Inputs are settled-window means and gravity acceleration expressed in the SAME
sensor frame.  Never pass powered/windy means, a moving mean, or gravity-removed
forces. Known CG height is a fixed assumption. Noise excludes mounting, gravity
direction, height, manufacturer scale and correlated drift systematic errors.
"""

from __future__ import annotations

import numpy as np


def design(gravity_m_s2, height_m: float) -> np.ndarray:
    """Parameters: six additive biases, mass, mass*x, mass*y."""
    gravity = np.asarray(gravity_m_s2, dtype=float)
    if gravity.ndim != 2 or gravity.shape[1] != 3 or not np.isfinite(gravity).all():
        raise ValueError("gravity must be finite n by 3 in physical sensor axes")
    if not np.isfinite(height_m) or np.any(np.linalg.norm(gravity, axis=1) < 9):
        raise ValueError("height must be finite; gravity must include full gravity magnitude")
    matrix = np.zeros((len(gravity), 6, 9))
    matrix[:, :, :6] = np.eye(6)
    matrix[:, :3, 6] = gravity
    matrix[:, 3:, 6] = np.cross([0, 0, height_m], gravity)
    matrix[:, 3:, 7] = np.cross([1, 0, 0], gravity)
    matrix[:, 3:, 8] = np.cross([0, 1, 0], gravity)
    return matrix.reshape(-1, 9)


def fit_static_reference(observations: dict) -> dict:
    """Weighted fit; rank deficiency returns no unique parameters/covariance.

    stderr is the uncertainty of each WINDOW MEAN, not raw-sample SD. Account
    for serial correlation when obtaining it. Equal samples are not repeats.
    """
    if (
        observations.get("frame") != "physical_sensor"
        or observations.get("wind_free_motor_off_stationary") is not True
    ):
        raise ValueError(
            "require confirmed wind-free, motor-off settled physical-sensor observations"
        )
    gravity = observations["gravity_m_s2"]
    matrix = design(gravity, observations["fixed_cg_height_m"])
    wrench = np.asarray(observations["raw_wrench_mean"], dtype=float)
    noise = np.asarray(observations["mean_stderr"], dtype=float)
    expected = (matrix.shape[0] // 6, 6)
    if wrench.shape != expected or noise.shape != expected:
        raise ValueError("raw_wrench_mean and mean_stderr must be n by 6 (N, N*m)")
    if not np.isfinite(wrench).all() or not np.isfinite(noise).all() or np.any(noise <= 0):
        raise ValueError("require finite means and positive mean standard errors")
    weighted = matrix / noise.reshape(-1, 1)
    scales = np.linalg.norm(weighted, axis=0)
    if np.any(scales == 0):
        rank = int(np.linalg.matrix_rank(weighted))
        singular = []
    else:
        singular = np.linalg.svd(weighted / scales, compute_uv=False).tolist()
        rank = int(np.sum(np.asarray(singular) > singular[0] * 1e-10))
    report = dict(
        rank=rank,
        parameter_count=9,
        normalized_singular_values=singular,
        absolute_values_validated=False,
        uncertainty_scope="window-mean noise conditional on gravity, height and sensor calibration",
        unaccounted_systematics=[
            "gravity direction",
            "height",
            "scale/cross-axis",
            "correlated drift",
        ],
        parameters=None,
        covariance=None,
    )
    if rank < 9:
        report["decision"] = "NOT_IDENTIFIABLE"
        return report
    normalized = weighted / scales
    estimate = (
        np.linalg.lstsq(normalized, wrench.reshape(-1) / noise.reshape(-1), rcond=None)[0] / scales
    )
    _, s, vt = np.linalg.svd(normalized, full_matrices=False)
    covariance = (vt.T / s**2) @ vt / np.outer(scales, scales)
    residual = (matrix @ estimate - wrench.reshape(-1)).reshape(-1, 6)
    report.update(
        decision="CONDITIONAL_FIT" if estimate[6] > 0 else "NONPHYSICAL_MASS",
        normalized_condition_number=singular[0] / singular[-1],
        parameters=estimate.tolist(),
        covariance=covariance.tolist(),
        parameter_names=["bFx", "bFy", "bFz", "bMx", "bMy", "bMz", "mass", "mass_x", "mass_y"],
        residual_rmse=np.sqrt(np.mean(residual**2, axis=0)).tolist(),
    )
    if estimate[6] > 0:
        report["cg_sensor_m"] = [
            estimate[7] / estimate[6],
            estimate[8] / estimate[6],
            observations["fixed_cg_height_m"],
        ]
        jacobian = np.zeros((2, 9))
        jacobian[:, 6] = -estimate[7:9] / estimate[6] ** 2
        jacobian[0, 7] = jacobian[1, 8] = 1 / estimate[6]
        cg_covariance = jacobian @ covariance @ jacobian.T
        report["cg_xy_covariance_m2"] = cg_covariance.tolist()
        report["cg_xy_std_m"] = np.sqrt(np.maximum(np.diag(cg_covariance), 0)).tolist()
    # Uncertainty of estimated baseline at each observed pose, not total load error.
    variances = np.einsum("ij,jk,ik->i", matrix, covariance, matrix)
    report["baseline_std_N_Nm"] = np.sqrt(np.maximum(variances, 0)).reshape(-1, 6).tolist()
    return report
