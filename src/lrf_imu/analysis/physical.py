"""Acceleration-magnitude plausibility metrics from the historical analysis."""

from __future__ import annotations

from typing import Any

import numpy as np

GRAVITY_M_S2 = 9.80665


def _windows(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 3 or array.shape[1] < 3:
        raise ValueError("windows must have shape [samples, channels>=3, time]")
    if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
        raise ValueError("windows must contain only finite numeric values")
    return array


def acceleration_magnitude(windows: np.ndarray) -> np.ndarray:
    """Return the Euclidean magnitude of the first three acceleration channels."""

    array = _windows(windows)
    return np.sqrt(np.sum(np.square(array[:, :3, :], dtype=np.float64), axis=1))


def acceleration_magnitude_summary(
    windows: np.ndarray,
    *,
    gravity_m_s2: float = GRAVITY_M_S2,
    max_g: float = 10.0,
) -> dict[str, Any]:
    """Summarize magnitudes and count points strictly above ``max_g``.

    Historical code compares each acceleration-magnitude time point with
    ``10 * 9.80665`` using ``>`` (not ``>=``). Inputs must already be in
    physical ``m/s^2`` coordinates.
    """

    if gravity_m_s2 <= 0 or max_g <= 0:
        raise ValueError("gravity_m_s2 and max_g must be positive")
    values = acceleration_magnitude(windows).reshape(-1)
    threshold = float(gravity_m_s2 * max_g)
    count = int(np.count_nonzero(values > threshold))
    return {
        "schema_version": "m3e.physical-magnitude.1",
        "coordinate_system": "physical_m_s2",
        "acceleration_channels": [0, 1, 2],
        "point_count": int(values.size),
        "threshold_g": float(max_g),
        "gravity_m_s2": float(gravity_m_s2),
        "threshold_m_s2": threshold,
        "comparison": "strictly_greater_than",
        "count_above_threshold": count,
        "pct_above_threshold": float(100.0 * count / values.size),
        "mean_m_s2": float(np.mean(values)),
        "std_m_s2_population": float(np.std(values, ddof=0)),
        "median_m_s2": float(np.median(values)),
        "p95_m_s2": float(np.percentile(values, 95)),
        "p99_m_s2": float(np.percentile(values, 99)),
        "max_m_s2": float(np.max(values)),
    }
