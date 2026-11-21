"""Quasi-static CG sensitivity at a fixed reference point, in FRD moments."""

import numpy as np

from fftgyro_quad_id.frames.rig_kinematics import body_to_tunnel


def gravity_cg_design(pitch_rad, yaw_rad, mass_kg, axis):
    """Rows for a dot (delta_r_FLU cross gravity_FLU), in Nm/m.

    y_nominal = prop_at_nominal + aero_at_nominal + C*delta_r + residual_bias.
    This is a static equilibrium equation, not a dynamic CG correction.
    """
    if axis not in ("PITCH", "YAW") or not np.isfinite(mass_kg) or mass_kg <= 0:
        raise ValueError("Positive mass and PITCH/YAW required")
    r = body_to_tunnel(pitch_rad, yaw_rad)
    gravity = mass_kg * 9.80665 * r[..., 2, :] * [1, -1, -1]
    direction = np.array([0, -1, 0] if axis == "PITCH" else [0, 0, -1])
    return np.stack(
        [np.cross([1, 0, 0], gravity) @ direction, np.cross([0, 1, 0], gravity) @ direction],
        axis=-1,
    )


def project_cg(cg, nuisance, weights):
    """Unregularized nuisance projection: priors never manufacture data rank."""
    c = np.asarray(cg) * np.sqrt(weights)[:, None]
    z = np.asarray(nuisance) * np.sqrt(weights)[:, None]
    u, s, _ = np.linalg.svd(z, full_matrices=False)
    rank_z = int(np.sum(s > s[0] * 1e-10)) if len(s) and s[0] else 0
    projected = c - u[:, :rank_z] @ (u[:, :rank_z].T @ c)
    _, singular, vh = np.linalg.svd(projected, full_matrices=True)
    threshold = max(np.linalg.norm(c, 2), 1.0) * 1e-9
    rank = int(np.sum(singular > threshold))
    null = vh[rank:]
    fractions = np.linalg.norm(projected, axis=0) / np.maximum(np.linalg.norm(c, axis=0), 1e-30)
    return {
        "rank": rank,
        "columns": c.shape[1],
        "singular_values": singular.tolist(),
        "remaining_column_fraction": fractions.tolist(),
        "coordinate_identifiable": (np.linalg.norm(null, axis=0) < 1e-7).tolist(),
    }
