"""Command-target RPM dynamics with distinct rise/fall constants."""

import numpy as np


def simulate_asymmetric(time_s, target_rpm, tau_up_s, tau_down_s, delay_s, initial_rpm):
    """ZOH input, nonnegative delay; initialize once, never reset at transitions."""
    t, target = np.asarray(time_s, float), np.asarray(target_rpm, float)
    if (
        t.ndim != 1
        or target.shape != t.shape
        or len(t) < 2
        or not np.isfinite(t).all()
        or not np.isfinite(target).all()
        or np.any(np.diff(t) <= 0)
        or not np.isfinite(initial_rpm)
        or not all(np.isfinite([tau_up_s, tau_down_s, delay_s]))
        or min(tau_up_s, tau_down_s) <= 0
        or delay_s < 0
    ):
        raise ValueError("finite increasing time, targets and positive time constants required")
    delayed = target[np.maximum(np.searchsorted(t, t - delay_s, side="right") - 1, 0)]
    result = np.empty(len(t))
    result[0] = initial_rpm
    for i, dt in enumerate(np.diff(t), 1):
        tau = tau_up_s if delayed[i - 1] >= result[i - 1] else tau_down_s
        result[i] = delayed[i - 1] + (result[i - 1] - delayed[i - 1]) * np.exp(-dt / tau)
    return result
