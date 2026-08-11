"""Welch PSD aggregation and paper-relevant spectral comparisons."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def _windows(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 3 or min(array.shape) == 0:
        raise ValueError("windows must have non-empty [samples, channels, time] shape")
    if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
        raise ValueError("windows must contain only finite numeric values")
    return array


def compute_psd(
    windows: np.ndarray,
    *,
    sampling_hz: float = 50.0,
    nperseg: int = 160,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Welch PSD for every window/channel as in the source analysis."""

    if sampling_hz <= 0 or nperseg <= 0:
        raise ValueError("sampling_hz and nperseg must be positive")
    array = _windows(windows)
    try:
        from scipy.signal import welch  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover - exercised without analysis extra
        raise ImportError("spectral analysis requires scipy; install lrf-imu[analysis]") from exc
    segment = min(int(nperseg), int(array.shape[-1]))
    frequencies, psd = welch(array, fs=float(sampling_hz), nperseg=segment, axis=-1)
    return np.asarray(frequencies), np.asarray(psd)


def aggregate_fold_psd(fold_psds: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Return fold mean and population SD, matching the historical summarizer."""

    if not fold_psds:
        raise ValueError("at least one fold PSD is required")
    arrays = [np.asarray(value, dtype=np.float64) for value in fold_psds]
    if any(value.shape != arrays[0].shape for value in arrays):
        raise ValueError("all fold PSD arrays must have identical shapes")
    stacked = np.stack(arrays, axis=0)
    return np.mean(stacked, axis=0), np.std(stacked, axis=0, ddof=0)


def _band_power(values: np.ndarray, frequencies: np.ndarray, low: float, high: float) -> float:
    mask = (frequencies >= low) & (frequencies <= high)
    if np.count_nonzero(mask) < 2:
        return 0.0
    return float(np.trapz(values[mask], frequencies[mask]))


def spectral_statistics(
    frequencies: np.ndarray,
    real_mean: np.ndarray,
    synthetic_mean: np.ndarray,
    *,
    real_std: np.ndarray | None = None,
    synthetic_std: np.ndarray | None = None,
    channel_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Compare channel-level natural-log PSD and paper frequency bands.

    Correlations use ``corrcoef(log(real), log(synthetic))`` on bins where both
    PSDs are positive. Band ratios are synthetic/real trapezoidal power.
    """

    freq = np.asarray(frequencies, dtype=np.float64)
    real = np.asarray(real_mean, dtype=np.float64)
    synthetic = np.asarray(synthetic_mean, dtype=np.float64)
    if real.ndim != 2 or synthetic.shape != real.shape or freq.ndim != 1:
        raise ValueError("mean PSDs must be [channels, frequencies]")
    if real.shape[1] != freq.size or not np.isfinite(real).all() or not np.isfinite(synthetic).all():
        raise ValueError("PSD arrays must align with finite frequencies")
    if real_std is not None or synthetic_std is not None:
        if real_std is None or synthetic_std is None:
            raise ValueError("real_std and synthetic_std must be supplied together")
        rstd = np.asarray(real_std, dtype=np.float64)
        sstd = np.asarray(synthetic_std, dtype=np.float64)
        if rstd.shape != real.shape or sstd.shape != real.shape:
            raise ValueError("PSD standard deviations must match mean PSD shape")
    else:
        rstd = None
        sstd = None
    names = list(channel_names or [f"channel_{index}" for index in range(real.shape[0])])
    if len(names) != real.shape[0]:
        raise ValueError("channel_names length must match channel count")

    records: list[dict[str, Any]] = []
    for index, name in enumerate(names):
        rmean, smean = real[index], synthetic[index]
        mask = (rmean > 0) & (smean > 0)
        correlation = float("nan")
        if np.count_nonzero(mask) >= 2:
            correlation = float(np.corrcoef(np.log(rmean[mask]), np.log(smean[mask]))[0, 1])
        ratios: dict[str, float] = {}
        for key, low, high in (("0_25hz", 0.0, 25.0), ("10_25hz", 10.0, 25.0), ("0_2hz", 0.0, 2.0)):
            real_power = _band_power(rmean, freq, low, high)
            synthetic_power = _band_power(smean, freq, low, high)
            ratios[key] = float(synthetic_power / real_power) if real_power > 0 else float("nan")
        record: dict[str, Any] = {
            "channel": str(name),
            "log_psd_correlation": correlation,
            "band_power_ratio_0_25hz": ratios["0_25hz"],
            "band_power_ratio_10_25hz": ratios["10_25hz"],
            "band_power_ratio_0_2hz": ratios["0_2hz"],
        }
        if rstd is not None and sstd is not None:
            std_mask = rstd[index] > 0
            record["std_ratio"] = (
                float(np.mean(sstd[index, std_mask] / rstd[index, std_mask]))
                if np.any(std_mask)
                else float("nan")
            )
        records.append(record)

    numeric_keys = [key for key in records[0] if key != "channel"]
    means = {
        key: float(np.mean([record[key] for record in records]))
        for key in numeric_keys
    }
    high_frequency_ratio = means["band_power_ratio_10_25hz"]
    return {
        "schema_version": "m3e.spectral-statistics.1",
        "welch": {"sampling_hz": None, "frequency_bins": int(freq.size)},
        "logarithm": "natural_log",
        "positive_bins_only": True,
        "channels": records,
        "mean": means,
        "high_frequency_band_hz": [10.0, 25.0],
        "high_frequency_attenuation_observed": bool(high_frequency_ratio < 1.0),
        "interpretation_scope": "descriptive_signal_comparison_not_clinical_validation",
    }
