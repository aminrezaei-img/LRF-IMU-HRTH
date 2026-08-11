"""Nine-setting segmentation sensitivity aggregation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

WINDOW_GRID = (
    (80, 20), (80, 40), (80, 80),
    (160, 20), (160, 40), (160, 80),
    (240, 20), (240, 40), (240, 80),
)

SCALAR_METRICS = (
    "trtr_macro_f1",
    "tstr_macro_f1",
    "macro_f1_retention",
    "trtr_accuracy",
    "tstr_accuracy",
    "aug_macro_f1",
    "psd_log_cosine",
    "hf_psd_ratio",
)


def _mean_sample_sd(values: Sequence[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("metric values must be non-empty and finite")
    return float(np.mean(array)), float(np.std(array, ddof=1)) if array.size > 1 else 0.0


def summarize_fold_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize one segmentation setting with fold-wise sample SD."""

    if not records:
        raise ValueError("at least one fold record is required")
    summary: dict[str, Any] = {"n_folds": len(records), "sd_ddof": 1}
    for metric in SCALAR_METRICS:
        values = [float(record[metric]) for record in records if metric in record]
        if values:
            mean, sd = _mean_sample_sd(values)
            summary[f"{metric}_mean"] = mean
            summary[f"{metric}_sd"] = sd
    classes = sorted({
        str(name)
        for record in records
        for name in (record.get("tstr_per_class_f1") or {})
    })
    for class_name in classes:
        values = [
            float(record["tstr_per_class_f1"][class_name])
            for record in records
            if class_name in (record.get("tstr_per_class_f1") or {})
        ]
        mean, sd = _mean_sample_sd(values)
        summary[f"tstr_f1_{class_name}_mean"] = mean
        summary[f"tstr_f1_{class_name}_sd"] = sd
    return summary


def summarize_sensitivity_grid(
    records_by_setting: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    require_complete_grid: bool = True,
) -> dict[str, Any]:
    """Aggregate fold records for the source 3x3 window/hop grid."""

    expected = {f"WIN{window}_HOP{hop}" for window, hop in WINDOW_GRID}
    supplied = set(records_by_setting)
    if require_complete_grid and supplied != expected:
        missing = sorted(expected - supplied)
        extra = sorted(supplied - expected)
        raise ValueError(f"sensitivity grid mismatch; missing={missing}, extra={extra}")
    settings: dict[str, Any] = {}
    for window, hop in WINDOW_GRID:
        name = f"WIN{window}_HOP{hop}"
        if name not in records_by_setting:
            continue
        settings[name] = {
            "window_samples": window,
            "hop_samples": hop,
            **summarize_fold_records(records_by_setting[name]),
        }
    return {
        "schema_version": "m3e.window-sensitivity.1",
        "grid": [[window, hop] for window, hop in WINDOW_GRID],
        "setting_count": len(settings),
        "aggregation": "fold_mean_and_sample_sd",
        "settings": settings,
    }
