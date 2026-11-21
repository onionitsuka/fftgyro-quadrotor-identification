"""Conditional forward CG correction; sensor/aircraft axes FLU, pitch moment FRD.

Only dx is identifiable under the hypothesis of unchanged sensor bias, dz and
mass, and zero external aerodynamic/propulsive wrench in the stopped reference.
"""

import numpy as np


def infer_forward_shift(stopped_pitch_nm: float, support_z_n: float) -> float:
    """Choose dx such that stopped pitch + dx * support_z = 0."""
    if not np.isfinite([stopped_pitch_nm, support_z_n]).all() or support_z_n <= 1:
        raise ValueError("Need finite stationary load and positive upward support > 1 N")
    return -stopped_pitch_nm / support_z_n


def shift_pitch_origin(moment_frd, force_z_flu, dx_m):
    """External wrench at G+dx: My_FRD_new = My_FRD_old - dx * Fz_FLU.

    For reconstructed measured moment, pass negative support force. The
    rotational inertia term is held fixed: this is a quasi-static diagnostic,
    not a full dynamic battery/inertia update.
    """
    return np.asarray(moment_frd) - dx_m * np.asarray(force_z_flu)
