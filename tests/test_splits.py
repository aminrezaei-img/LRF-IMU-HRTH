"""Focused tests for canonical LOSO and split-stage separation."""

from pathlib import Path
import sys

import numpy as np
import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from lrf_imu.data.splits import (  # noqa: E402
    CANONICAL_SUBJECTS,
    CNN_WINDOW_VALIDATION_FRACTION,
    DEFAULT_SPLIT_SEED,
    DuplicateSubjectError,
    MissingSubjectError,
    UnknownSubjectError,
    VAE_SUBJECT_VALIDATION_FRACTION,
    canonical_loso_folds,
    split_cnn_windows,
    split_vae_subjects,
    split_vae_windows,
    validate_subjects,
)


def _compact_subject_windows():
    counts = {1: 7, 2: 8, 3: 8, 5: 8}
    windows = {
        subject: np.full((count, 6, 4), float(subject), dtype=np.float32)
        for subject, count in counts.items()
    }
    labels = {
        subject: np.arange(count, dtype=np.int64) % 4
        for subject, count in counts.items()
    }
    return windows, labels


def test_canonical_cohort_and_loso_fold_count_are_locked():
    assert CANONICAL_SUBJECTS == (1, 2, 3, 5, 8, 9, 10, 11, 12, 13, 14, 16)
    folds = canonical_loso_folds()
    assert len(folds) == 12
    assert folds[0].held_out_subject == 1
    assert folds[-1].held_out_subject == 16
    assert all(len(fold.training_subjects) == 11 for fold in folds)
    assert all(
        fold.held_out_subject not in fold.training_subjects for fold in folds
    )


def test_source_randomstate_subject_split_matches_compact_snapshot():
    result = split_vae_subjects(
        ["01", "02", "03", "05"],
        "05",
        val_fraction=VAE_SUBJECT_VALIDATION_FRACTION,
        seed=DEFAULT_SPLIT_SEED,
    )
    assert result.validation_subjects == (1,)
    assert result.train_subjects == (2, 3)
    assert result.held_out_subject == 5
    assert set(result.train_subjects).isdisjoint(result.validation_subjects)
    assert 5 not in result.train_subjects + result.validation_subjects
    assert result.metadata.validation_unit == "subject"
    assert result.metadata.random_state == "numpy.random.RandomState"


def test_compact_materialized_split_is_exact_16_7_8_and_subject_disjoint():
    windows, labels = _compact_subject_windows()
    result = split_vae_windows(
        windows,
        "05",
        labels_by_subject=labels,
        cohort=["01", "02", "03", "05"],
        seed=42,
    )

    assert result.train_windows.shape == (16, 6, 4)
    assert result.validation_windows.shape == (7, 6, 4)
    assert result.held_out_test_windows.shape == (8, 6, 4)
    assert result.train_windows.dtype == np.float32
    assert result.train_labels.shape == (16,)
    assert result.validation_labels.shape == (7,)
    assert result.held_out_test_labels.shape == (8,)
    assert result.metadata.window_counts == {
        "train": 16,
        "validation": 7,
        "held_out_test": 8,
    }
    assert result.metadata.window_counts_by_subject == {1: 7, 2: 8, 3: 8, 5: 8}
    assert result.metadata.train_subjects == (2, 3)
    assert result.metadata.validation_subjects == (1,)
    assert result.metadata.held_out_subject == 5
    assert np.all(result.train_windows[:, 0, 0] != 1.0)
    assert np.all(result.train_windows[:, 0, 0] != 5.0)
    assert np.all(result.validation_windows[:, 0, 0] == 1.0)
    assert np.all(result.held_out_test_windows[:, 0, 0] == 5.0)


def test_complete_cohort_guard_reports_missing_subjects():
    with pytest.raises(MissingSubjectError, match="missing canonical subject"):
        validate_subjects([1, 2, 3], require_complete_cohort=True)


def test_duplicate_subjects_are_rejected_after_normalization():
    with pytest.raises(DuplicateSubjectError):
        validate_subjects([1, "01"], require_complete_cohort=False)


def test_unknown_subjects_are_rejected():
    with pytest.raises(UnknownSubjectError):
        validate_subjects([1, 7], require_complete_cohort=False)


def test_missing_held_out_subject_is_rejected():
    windows, labels = _compact_subject_windows()
    with pytest.raises(MissingSubjectError, match="held_out_subject"):
        split_vae_windows(
            {1: windows[1], 2: windows[2], 3: windows[3]},
            5,
            labels_by_subject={1: labels[1], 2: labels[2], 3: labels[3]},
        )


def test_cnn_window_validation_is_separate_and_uses_020():
    windows = np.arange(20 * 6 * 4, dtype=np.float32).reshape(20, 6, 4)
    labels = np.asarray([3, 4, 1, 3, 33, 33, 4, 1, 33, 1, 3, 1, 33, 3, 33, 3, 1, 4, 4, 4])
    result = split_cnn_windows(windows, labels, seed=42)

    assert VAE_SUBJECT_VALIDATION_FRACTION == 0.15
    assert CNN_WINDOW_VALIDATION_FRACTION == 0.20
    assert result.metadata.validation_fraction == 0.20
    assert result.metadata.validation_unit == "window"
    assert result.metadata.protocol == "cnn_internal_window_validation"
    assert result.metadata.separate_from_vae_subject_split is True
    assert result.train_windows.shape == (16, 6, 4)
    assert result.validation_windows.shape == (4, 6, 4)
    assert set(result.train_indices).isdisjoint(result.validation_indices)
    assert set(result.train_indices).union(result.validation_indices) == set(range(20))
    assert sorted(np.unique(result.validation_labels, return_counts=True)[1].tolist()) == [1, 1, 1, 1]


def test_vae_split_does_not_accept_cnn_fraction_as_implicit_protocol():
    windows, labels = _compact_subject_windows()
    result = split_vae_windows(
        windows,
        5,
        labels_by_subject=labels,
        val_fraction=CNN_WINDOW_VALIDATION_FRACTION,
    )
    assert result.metadata.validation_unit == "subject"
    assert result.metadata.validation_fraction == 0.20
    assert result.metadata.protocol == "vae_subject_loso"
    assert result.metadata.validation_subjects == (1,)


def test_split_rejects_misaligned_labels_and_mismatched_window_shapes():
    windows, labels = _compact_subject_windows()
    bad_labels = dict(labels)
    bad_labels[1] = np.zeros(6, dtype=np.int64)
    with pytest.raises(ValueError, match="labels for subject 01"):
        split_vae_windows(windows, 5, labels_by_subject=bad_labels)

    bad_windows = dict(windows)
    bad_windows[2] = np.zeros((8, 3, 4), dtype=np.float32)
    with pytest.raises(ValueError, match=r"share \[C, T\]"):
        split_vae_windows(bad_windows, 5, labels_by_subject=labels)


def test_empty_subject_partition_has_typed_shape_without_leakage():
    windows = {
        1: np.ones((2, 2, 3), dtype=np.float32),
        2: np.ones((2, 2, 3), dtype=np.float32) * 2,
    }
    labels = {subject: np.zeros(2, dtype=np.int64) for subject in windows}
    result = split_vae_windows(windows, 2, labels_by_subject=labels)
    assert result.validation_windows.shape == (0, 2, 3)
    assert result.validation_labels.shape == (0,)
    assert result.train_windows.shape == (2, 2, 3)
    assert result.held_out_test_windows.shape == (2, 2, 3)
