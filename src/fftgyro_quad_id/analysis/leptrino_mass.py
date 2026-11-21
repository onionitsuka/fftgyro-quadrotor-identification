"""Aircraft mass from the gravity circle traced by static Leptrino pose forces."""

from __future__ import annotations

import numpy as np


def fit_gravity_circle(force_pose_means_N, gravity_m_s2: float = 9.80665) -> dict:
    """Fit the force-space circle produced by rotation about a horizontal axis.

    A constant six-axis sensor zero translates all force points but does not
    change the circle radius.  For a horizontal rotation axis, gravity is
    perpendicular to the axis and the radius is exactly ``mass * gravity``.
    """
    force = np.asarray(force_pose_means_N, dtype=float)
    if force.ndim != 2 or force.shape[1] != 3 or len(force) < 3:
        raise ValueError("at least three 3D force pose means are required")
    if not np.isfinite(force).all():
        raise ValueError("force pose means must be finite")
    if not np.isfinite(gravity_m_s2) or gravity_m_s2 <= 0:
        raise ValueError("gravity_m_s2 must be finite and positive")

    origin = force.mean(axis=0)
    _, singular, right = np.linalg.svd(force - origin, full_matrices=False)
    if singular[1] <= 1e-9:
        raise ValueError("pose forces do not span a circle plane")
    basis = right[:2].T
    normal = np.cross(basis[:, 0], basis[:, 1])
    normal /= np.linalg.norm(normal)
    coordinates = (force - origin) @ basis
    design = np.column_stack((2 * coordinates, np.ones(len(coordinates))))
    rhs = np.sum(coordinates**2, axis=1)
    solution, _, rank, _ = np.linalg.lstsq(design, rhs, rcond=None)
    if rank < 3:
        raise ValueError("pose forces cannot identify a circle")
    center_2d, constant = solution[:2], solution[2]
    radius_squared = constant + float(center_2d @ center_2d)
    if radius_squared <= 0:
        raise ValueError("fitted circle radius is not positive")
    center = origin + basis @ center_2d
    radii = np.linalg.norm(force - center, axis=1)
    plane_residual = (force - origin) @ normal
    radius = float(np.sqrt(radius_squared))
    return {
        "mass_kg": radius / gravity_m_s2,
        "gravity_force_radius_N": radius,
        "circle_center_sensor_N": center,
        "circle_normal_sensor": normal,
        "pose_radius_residual_N": radii - radius,
        "plane_residual_N": plane_residual,
        "singular_values_N": singular,
    }
