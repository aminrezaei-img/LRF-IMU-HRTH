"""Preparation bridge from HARTH-family recordings to the LRF-IMU VAE."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from .audit import audit_window_duplicates
from .harth import HarthSubject, TARGET_CLASS_NAMES, load_harth_subjects
from .normalization import ChannelStandardizer
from .windowing import make_windows


@dataclass(frozen=True)
class PreparedHarthData:
    """In-memory ten-class HARTH preparation result."""
    subjects: Dict[str, HarthSubject]
    train_windows: np.ndarray
    validation_windows: np.ndarray
    held_out_test_windows: np.ndarray
    train_labels: np.ndarray
    validation_labels: np.ndarray
    held_out_test_labels: np.ndarray
    train_subjects: Tuple[str, ...]
    validation_subjects: Tuple[str, ...]
    held_out_subject: str
    normalizer: ChannelStandardizer
    metadata: dict

    @property
    def summary(self) -> dict:
        return dict(self.metadata)


def prepare_harth_data(
    data_root: str | Path,
    *,
    composition: str = "harth_walking_speed",
    held_out_subject: str | None = None,
    window_length: int = 160,
    hop_length: int = 40,
    validation_fraction: float = 0.15,
    seed: int = 42,
    rate_tolerance_hz: float = 0.25,
) -> PreparedHarthData:
    """Read, map, window, split, normalize, and audit a HARTH-family fold.

    Splits are by namespaced subject key (for example ``harth:S006``), never
    by windows. Excluded labels terminate runs, unlike the legacy REALDISP
    compatibility mode.
    """
    if not 0 < float(validation_fraction) < 1:
        raise ValueError("validation_fraction must be in (0, 1)")
    loaded = load_harth_subjects(data_root, composition, rate_tolerance_hz=rate_tolerance_hz)
    keys = tuple(loaded)
    if held_out_subject is None:
        held_out_subject = keys[0]
    if held_out_subject not in loaded:
        raise KeyError("held_out_subject must be one of the discovered namespaced subjects")

    windows: Dict[str, np.ndarray] = {}
    labels: Dict[str, np.ndarray] = {}
    for key, subject in loaded.items():
        x, y = make_windows(
            subject.signals,
            subject.encoded_labels,
            window_length=window_length,
            hop_length=hop_length,
            allowed_labels=range(len(TARGET_CLASS_NAMES)),
            label_encoding={i: i for i in range(len(TARGET_CLASS_NAMES))},
            mode="strict_original_contiguity",
            channel_count=3,
        )
        windows[key], labels[key] = x, y

    pool = [key for key in keys if key != held_out_subject]
    rng = np.random.RandomState(int(seed))
    rng.shuffle(pool)
    n_val = min(max(1, int(len(pool) * float(validation_fraction))), max(0, len(pool) - 1))
    validation_subjects = tuple(pool[:n_val])
    train_subjects = tuple(pool[n_val:])

    def concatenate(selected: Tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
        if selected:
            return np.concatenate([windows[key] for key in selected]), np.concatenate([labels[key] for key in selected])
        return np.empty((0, 3, int(window_length)), np.float32), np.empty((0,), np.int64)

    train, train_y = concatenate(train_subjects)
    validation, validation_y = concatenate(validation_subjects)
    test, test_y = concatenate((held_out_subject,))
    if train.shape[0] == 0 or validation.shape[0] == 0:
        raise ValueError("the selected HARTH fold needs non-empty train and validation windows")
    normalizer = ChannelStandardizer(channels=3).fit_training(
        train, training_subjects=train_subjects, validation_subjects=validation_subjects,
        held_out_subject=held_out_subject,
    )
    train, validation, test = normalizer.transform(train), normalizer.transform(validation), normalizer.transform(test)
    audit = audit_window_duplicates(train, validation, test, include_within_split=True, raise_on_duplicate=True)
    per_subject = {
        key: {"dataset": subject.dataset, "subject_id": subject.subject_id,
              "window_count": int(windows[key].shape[0]),
              "encoded_class_counts": {name: int(np.count_nonzero(labels[key] == i)) for i, name in enumerate(TARGET_CLASS_NAMES)},
              "excluded_row_count": subject.excluded_row_count,
              "sample_rate_hz": subject.sample_rate_hz}
        for key, subject in loaded.items()
    }
    metadata = {
        "schema_version": "harth.ten_class.prepare.1", "dataset": "HARTH-family",
        "composition": composition, "channels": ["thigh_x", "thigh_y", "thigh_z"],
        "channel_count": 3, "sampling_frequency_hz": 50.0,
        "target_class_names": list(TARGET_CLASS_NAMES), "target_class_count": 10,
        "window": {"length": int(window_length), "hop": int(hop_length), "padding": False},
        "split": {"protocol": "leave_one_subject_out", "subject_key": "dataset:subject_id",
                  "held_out_subject": held_out_subject, "train_subjects": list(train_subjects),
                  "validation_subjects": list(validation_subjects), "seed": int(seed),
                  "validation_fraction": float(validation_fraction)},
        "counts": {"train": int(train.shape[0]), "validation": int(validation.shape[0]), "held_out_test": int(test.shape[0])},
        "per_subject": per_subject, "normalization": normalizer.to_metadata(), "audit": audit,
        "raw_values_persisted": False, "participant_data_persisted": False,
    }
    return PreparedHarthData(loaded, train, validation, test, train_y, validation_y, test_y,
                             train_subjects, validation_subjects, held_out_subject, normalizer, metadata)


__all__ = ["PreparedHarthData", "prepare_harth_data"]
