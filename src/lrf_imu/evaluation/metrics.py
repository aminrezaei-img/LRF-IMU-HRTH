"""Source-faithful classification metrics for the LRF-IMU evaluation path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

PAPER_LABELS = (0, 1, 2, 3)
CONFUSION_NONZERO_EPSILON = 1e-6
RETENTION_DENOMINATOR_EPSILON = 1e-8


@dataclass(frozen=True)
class ClassificationMetrics:
    accuracy: float
    macro_f1: float
    per_class_f1: tuple[float, ...]
    confusion: tuple[tuple[int, ...], ...]
    confusion_normalized: tuple[tuple[float, ...], ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "accuracy": self.accuracy,
            "f1_macro": self.macro_f1,
            "per_class_f1": list(self.per_class_f1),
            "confusion": [list(row) for row in self.confusion],
            "confusion_normalized": [list(row) for row in self.confusion_normalized],
        }


def _sklearn_metrics():
    try:
        from sklearn.metrics import (  # type: ignore[import-untyped]
            accuracy_score,
            confusion_matrix,
            f1_score,
        )
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "evaluation requires scikit-learn; install the 'evaluation' extra"
        ) from exc
    return accuracy_score, confusion_matrix, f1_score


def classification_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    *,
    labels: Sequence[int] = PAPER_LABELS,
) -> ClassificationMetrics:
    """Match historical accuracy/F1/confusion calls and label ordering."""

    accuracy_score, confusion_matrix, f1_score = _sklearn_metrics()
    truth = np.asarray(y_true, dtype=np.int64).reshape(-1)
    predicted = np.asarray(y_pred, dtype=np.int64).reshape(-1)
    if truth.shape != predicted.shape:
        raise ValueError("y_true and y_pred must have identical one-dimensional shapes")
    ordered = tuple(int(label) for label in labels)
    raw = confusion_matrix(truth, predicted, labels=list(ordered)).astype(np.int64)
    with np.errstate(divide="ignore", invalid="ignore"):
        normalized = raw.astype(np.float64) / raw.sum(axis=1, keepdims=True)
    per_class = f1_score(
        truth, predicted, average=None, labels=list(ordered), zero_division=0
    )
    macro = f1_score(
        truth, predicted, average="macro", labels=list(ordered), zero_division=0
    )
    return ClassificationMetrics(
        accuracy=float(accuracy_score(truth, predicted)),
        macro_f1=float(macro),
        per_class_f1=tuple(float(value) for value in per_class),
        confusion=tuple(tuple(int(value) for value in row) for row in raw),
        confusion_normalized=tuple(
            tuple(float(value) for value in row) for row in normalized
        ),
    )


def retention_ratio(value: float, trtr_value: float) -> float:
    """Return the source expression ``value/(TRTR+1e-8)`` for one fold."""

    return float(value) / (float(trtr_value) + RETENTION_DENOMINATOR_EPSILON)


def mean_sample_sd(values: Iterable[float]) -> tuple[float, float]:
    """Return NaN-filtered mean and sample SD (``ddof=1``)."""

    array = np.asarray(tuple(values), dtype=np.float64)
    array = array[~np.isnan(array)]
    if array.size == 0:
        return float("nan"), float("nan")
    if array.size == 1:
        return float(array.mean()), 0.0
    return float(array.mean()), float(array.std(ddof=1))


def aggregate_confusions(
    matrices: Iterable[Sequence[Sequence[float]] | np.ndarray],
    *,
    nonzero_epsilon: float = CONFUSION_NONZERO_EPSILON,
) -> Mapping[str, Any]:
    """Use historical ``nanmean`` and non-zero-fold cell-count semantics."""

    arrays = [np.asarray(matrix, dtype=np.float64) for matrix in matrices]
    if not arrays:
        raise ValueError("at least one confusion matrix is required")
    first_shape = arrays[0].shape
    if any(array.shape != first_shape for array in arrays):
        raise ValueError("all confusion matrices must have the same shape")
    stack = np.stack(arrays, axis=0)
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(stack, axis=0)
    count = np.sum(stack > float(nonzero_epsilon), axis=0).astype(np.int64)
    return {
        "mean": mean.tolist(),
        "nonzero_fold_count": count.tolist(),
        "fold_count": int(stack.shape[0]),
        "semantics": "nanmean; count is folds with cell > 1e-6",
    }


def summarize_fold_records(
    records: Iterable[Mapping[str, Any]],
    *,
    scenario_order: Sequence[str] = ("trtr", "scarce", "tstr", "tstr_scarce"),
) -> list[dict[str, Any]]:
    """Summarize folds; retention is always a mean of fold-wise ratios."""

    rows = list(records)
    summary: list[dict[str, Any]] = []
    for scenario in scenario_order:
        selected = [row for row in rows if row.get("scenario") == scenario]
        if not selected:
            continue
        accuracy_mean, accuracy_sd = mean_sample_sd(row["accuracy"] for row in selected)
        macro_mean, macro_sd = mean_sample_sd(row["f1_macro"] for row in selected)
        retention_mean, retention_sd = mean_sample_sd(
            row["retention_ratio"] for row in selected
        )
        summary.append({
            "scenario": scenario,
            "fold_count": len(selected),
            "accuracy_mean": accuracy_mean,
            "accuracy_sd": accuracy_sd,
            "f1_macro_mean": macro_mean,
            "f1_macro_sd": macro_sd,
            "retention_mean": retention_mean,
            "retention_sd": retention_sd,
            "sd_ddof": 1,
            "retention_aggregation": "mean of fold-wise ratios",
        })
    return summary


__all__ = [
    "CONFUSION_NONZERO_EPSILON", "ClassificationMetrics", "PAPER_LABELS",
    "RETENTION_DENOMINATOR_EPSILON", "aggregate_confusions",
    "classification_metrics", "mean_sample_sd", "retention_ratio",
    "summarize_fold_records",
]
