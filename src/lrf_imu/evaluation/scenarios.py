"""Historical four-scenario classifier population construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import numpy as np

SCENARIO_ORDER = ("trtr", "scarce", "tstr", "tstr_scarce")
SCENARIO_DISPLAY_NAMES = {
    "trtr": "TRTR (Full Real)",
    "scarce": "Scarce (2/class)",
    "tstr": "TSTR (Synthetic Only)",
    "tstr_scarce": "TSTR + Scarce (Augmented)",
}


@dataclass(frozen=True)
class ScenarioPopulation:
    name: str
    windows: np.ndarray
    labels: np.ndarray
    real_indices: tuple[int, ...] = ()
    synthetic_indices: tuple[int, ...] = ()

    def metadata(self) -> dict[str, Any]:
        labels, counts = np.unique(self.labels, return_counts=True)
        return {
            "scenario": self.name,
            "train_size": int(self.labels.size),
            "class_counts": {
                str(int(label)): int(count) for label, count in zip(labels, counts)
            },
            "real_index_count": len(self.real_indices),
            "synthetic_index_count": len(self.synthetic_indices),
        }


def ensure_nct(windows: np.ndarray, channels: Optional[int] = None) -> np.ndarray:
    array = np.asarray(windows)
    if array.ndim != 3:
        raise ValueError("windows must be a three-dimensional N,C,T array")
    if channels is not None:
        if array.shape[1] == channels:
            return array
        if array.shape[2] == channels:
            return np.transpose(array, (0, 2, 1))
    if array.shape[2] in (3, 6) and array.shape[1] not in (3, 6):
        return np.transpose(array, (0, 2, 1))
    return array


def flatten_windows(windows: np.ndarray) -> np.ndarray:
    array = np.asarray(windows)
    if array.ndim != 3:
        raise ValueError("windows must be a three-dimensional N,C,T array")
    return array.reshape(array.shape[0], -1)


def _select_per_class_with_indices(
    windows: np.ndarray,
    labels: np.ndarray,
    n_per_class: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    if n_per_class <= 0:
        indices = tuple(range(len(labels)))
        return windows, labels, indices
    selected_groups: list[np.ndarray] = []
    for label in sorted(np.unique(labels)):
        candidate_indices = np.where(labels == label)[0]
        if len(candidate_indices) > n_per_class:
            candidate_indices = rng.choice(
                candidate_indices, size=n_per_class, replace=False
            )
        selected_groups.append(np.asarray(candidate_indices, dtype=np.int64))
    if not selected_groups:
        return windows, labels, tuple(range(len(labels)))
    selected = np.concatenate(selected_groups)
    selected_labels = np.concatenate([
        np.full(group.size, int(labels[group[0]]), dtype=np.int64)
        for group in selected_groups
    ])
    return windows[selected], selected_labels, tuple(int(value) for value in selected)


def build_scenario_populations(
    real_windows: np.ndarray,
    real_labels: Sequence[int] | np.ndarray,
    synthetic_windows: Optional[np.ndarray],
    synthetic_labels: Optional[Sequence[int] | np.ndarray],
    *,
    seed: int = 42,
    scarce_per_class: int = 2,
    synthetic_per_class: int = 500,
    class_labels: Sequence[int] = (0, 1, 2, 3),
    channels: Optional[int] = None,
) -> Mapping[str, ScenarioPopulation]:
    """Preserve source RNG consumption: cap synthetic first, select scarce second."""

    real_x = ensure_nct(np.asarray(real_windows), channels)
    real_y = np.asarray(real_labels, dtype=np.int64).reshape(-1)
    if real_x.shape[0] != real_y.size:
        raise ValueError("real windows and labels have different lengths")
    if scarce_per_class < 0 or synthetic_per_class <= 0:
        raise ValueError("scarce_per_class must be non-negative and synthetic_per_class positive")
    rng = np.random.default_rng(int(seed))

    if synthetic_windows is None or synthetic_labels is None:
        synthetic_x = np.empty((0,) + real_x.shape[1:], dtype=real_x.dtype)
        synthetic_y = np.empty((0,), dtype=np.int64)
        synthetic_indices: tuple[int, ...] = ()
    else:
        source_x = ensure_nct(np.asarray(synthetic_windows), channels)
        source_y = np.asarray(synthetic_labels, dtype=np.int64).reshape(-1)
        if source_x.shape[0] != source_y.size:
            raise ValueError("synthetic windows and labels have different lengths")
        synthetic_x, synthetic_y, synthetic_indices = _select_per_class_with_indices(
            source_x, source_y, int(synthetic_per_class), rng
        )

    scarce_groups: list[np.ndarray] = []
    for label in class_labels:
        candidates = np.where(real_y == int(label))[0]
        if candidates.size >= scarce_per_class:
            selected = rng.choice(candidates, scarce_per_class, replace=False)
        else:
            selected = candidates
        scarce_groups.append(np.asarray(selected, dtype=np.int64))
    scarce_indices_array = (
        np.concatenate(scarce_groups) if scarce_groups else np.empty((0,), dtype=np.int64)
    )
    scarce_x = real_x[scarce_indices_array]
    scarce_y = np.concatenate([
        np.full(group.size, int(label), dtype=np.int64)
        for label, group in zip(class_labels, scarce_groups)
    ])
    scarce_indices = tuple(int(value) for value in scarce_indices_array)

    populations: dict[str, ScenarioPopulation] = {
        "trtr": ScenarioPopulation(
            "trtr", real_x, real_y, real_indices=tuple(range(real_y.size))
        ),
        "scarce": ScenarioPopulation(
            "scarce", scarce_x, scarce_y, real_indices=scarce_indices
        ),
    }
    if synthetic_y.size:
        populations["tstr"] = ScenarioPopulation(
            "tstr", synthetic_x, synthetic_y, synthetic_indices=synthetic_indices
        )
        populations["tstr_scarce"] = ScenarioPopulation(
            "tstr_scarce",
            np.concatenate([synthetic_x, scarce_x], axis=0),
            np.concatenate([synthetic_y, scarce_y], axis=0),
            real_indices=scarce_indices,
            synthetic_indices=synthetic_indices,
        )
    return populations


__all__ = [
    "SCENARIO_DISPLAY_NAMES", "SCENARIO_ORDER", "ScenarioPopulation",
    "build_scenario_populations", "ensure_nct", "flatten_windows",
]
