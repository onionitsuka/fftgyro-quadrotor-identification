"""Offline zero-phase filtering for synchronized multi-axis load samples."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt


def filter_contiguous_load_blocks(
    values: np.ndarray,
    source_indices: np.ndarray,
    *,
    sample_rate_hz: float,
    cutoff_hz: float,
    order: int = 4,
) -> np.ndarray:
    """Low-pass NxC values without filtering across invalid source gaps."""
    loads = np.asarray(values, dtype=float)
    indices = np.asarray(source_indices, dtype=np.int64)
    if loads.ndim != 2 or len(loads) != len(indices) or not np.isfinite(loads).all():
        raise ValueError("values must be finite NxC and match source_indices")
    if len(indices) and (np.any(np.diff(indices) <= 0) or indices[0] < 0):
        raise ValueError("source_indices must be strictly increasing and nonnegative")
    if not 0 < cutoff_hz < sample_rate_hz / 2:
        raise ValueError("cutoff_hz must lie between zero and Nyquist")
    if order < 1:
        raise ValueError("order must be positive")
    if not len(loads):
        return loads.copy()
    sos = butter(order, cutoff_hz, btype="lowpass", fs=sample_rate_hz, output="sos")
    split = np.flatnonzero(np.diff(indices) > 1) + 1
    output = np.empty_like(loads)
    for positions in np.split(np.arange(len(indices)), split):
        # Keep each valid island independent. Short islands use the longest
        # reflection padding they can support instead of contaminating a gap.
        padlen = min(3 * (2 * len(sos) + 1), len(positions) - 1)
        if padlen < 1:
            output[positions] = loads[positions]
        else:
            output[positions] = sosfiltfilt(sos, loads[positions], axis=0, padlen=padlen)
    return output
