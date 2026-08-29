"""Small, descriptive signal metrics for Paper 3 sanity checks."""

from __future__ import annotations
from typing import Any
import numpy as np


def require_windows(value: Any, name: str = "windows") -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 3 or array.shape[1:] != (3, 160):
        raise ValueError(f"{name} must have shape [N,3,160], got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return array.astype(np.float32, copy=False)


def finite_counts(value: Any) -> dict[str, int | float]:
    array = np.asarray(value)
    finite = np.isfinite(array)
    return {
        "total_values": int(array.size),
        "nan_values": int(np.isnan(array).sum()),
        "inf_values": int(np.isinf(array).sum()),
        "finite_values": int(finite.sum()),
        "finite_percent": float(100 * finite.mean()) if array.size else 100.0,
    }


def _one_channel(values: np.ndarray) -> dict[str, float]:
    flat = values.reshape(-1).astype(np.float64)
    magnitude = np.sqrt(np.sum(values.astype(np.float64) ** 2, axis=0))
    return {
        "mean": float(flat.mean()),
        "std": float(flat.std()),
        "minimum": float(flat.min()),
        "maximum": float(flat.max()),
        "rms": float(np.sqrt(np.mean(flat**2))),
        "variance": float(flat.var()),
        "iqr": float(np.percentile(flat, 75) - np.percentile(flat, 25)),
        "signal_magnitude_mean": float(magnitude.mean()),
        "signal_magnitude_std": float(magnitude.std()),
    }


def signal_summary(windows: Any) -> dict[str, Any]:
    array = require_windows(windows)
    channels = [_one_channel(array[:, index, :]) for index in range(3)]
    variances = np.var(array.astype(np.float64), axis=(1, 2))
    return {
        "channels": channels,
        "mean_window_variance": float(variances.mean()) if variances.size else 0.0,
        "median_window_variance": float(np.median(variances))
        if variances.size
        else 0.0,
        "fraction_near_constant": float(np.mean(variances <= 1e-12))
        if variances.size
        else 1.0,
        "finite": finite_counts(array),
    }


def spectral_summary(windows: Any, sampling_hz: float = 50.0) -> dict[str, Any]:
    array = require_windows(windows)
    spectra = []
    for index in range(3):
        values = array[:, index, :].astype(np.float64)
        freqs = np.fft.rfftfreq(values.shape[-1], 1.0 / float(sampling_hz))
        power = np.mean(
            np.abs(np.fft.rfft(values - values.mean(axis=-1, keepdims=True), axis=-1))
            ** 2,
            axis=0,
        )
        total = float(
            np.trapezoid(power, freqs)
            if hasattr(np, "trapezoid")
            else np.trapz(power, freqs)
        )
        denom = float(power.sum())
        spectra.append(
            {
                "total_spectral_power": total,
                "dominant_frequency_hz": float(freqs[int(np.argmax(power))]),
                "spectral_centroid_hz": float((freqs * power).sum() / denom)
                if denom > 0
                else 0.0,
                "relative_power_0_2hz": float(
                    power[(freqs >= 0) & (freqs <= 2)].sum() / denom
                )
                if denom > 0
                else 0.0,
                "relative_power_2_10hz": float(
                    power[(freqs > 2) & (freqs <= 10)].sum() / denom
                )
                if denom > 0
                else 0.0,
                "relative_power_10_25hz": float(
                    power[(freqs > 10) & (freqs <= 25)].sum() / denom
                )
                if denom > 0
                else 0.0,
            }
        )
    return {"channels": spectra, "sampling_hz": float(sampling_hz)}


def compare_summaries(real: Any, synthetic: Any) -> dict[str, Any]:
    real_array, synthetic_array = (
        require_windows(real, "real"),
        require_windows(synthetic, "synthetic"),
    )
    real_s, syn_s = signal_summary(real_array), signal_summary(synthetic_array)
    errors = {
        "mse": float(
            np.mean(
                (
                    real_array.astype(np.float64).mean(axis=0)
                    - synthetic_array.astype(np.float64).mean(axis=0)
                )
                ** 2
            )
        ),
        "mae": float(
            np.mean(
                np.abs(
                    real_array.astype(np.float64).mean(axis=0)
                    - synthetic_array.astype(np.float64).mean(axis=0)
                )
            )
        ),
    }
    return {
        "real": real_s,
        "synthetic": syn_s,
        "reconstruction_error_on_channel_means": errors,
        "interpretation": "descriptive gross mismatch and collapse detection; not distributional equivalence",
    }


def feature_vector(windows: Any, sampling_hz: float = 50.0) -> np.ndarray:
    summary, spectral = signal_summary(windows), spectral_summary(windows, sampling_hz)
    values = []
    for channel, spec in zip(summary["channels"], spectral["channels"]):
        values.extend(
            [
                channel["mean"],
                channel["std"],
                channel["rms"],
                channel["signal_magnitude_mean"],
                spec["total_spectral_power"],
                spec["spectral_centroid_hz"],
            ]
        )
    return np.asarray(values, dtype=np.float64)


__all__ = [
    "compare_summaries",
    "feature_vector",
    "finite_counts",
    "require_windows",
    "signal_summary",
    "spectral_summary",
]
