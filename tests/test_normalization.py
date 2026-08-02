"""Focused synthetic tests for the public training-only standardizer."""

import json
from pathlib import Path

import numpy as np
import pytest

from lrf_imu.data.normalization import (
    ChannelStandardizer,
    ChannelValidationError,
    NonFiniteDataError,
    NotFittedError,
    ShapeValidationError,
)


FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "synthetic"


def _load_fixture(name):
    return json.loads((FIXTURE_ROOT / name).read_text(encoding="utf-8"))


def _training_fixture():
    cases = _load_fixture("standardization_cases.json")
    pattern = np.asarray(cases["time_pattern"], dtype=np.float32)
    channels = [
        base + np.tile(pattern, 1)
        for base in cases["channel_bases"]
    ]
    one_subject = np.asarray(channels, dtype=np.float32)
    return np.stack([one_subject, one_subject], axis=0)


def _offset_fixture(means, *, dtype=np.float32):
    pattern = np.asarray([-1, 1, -1, 1], dtype=dtype)
    return np.stack(
        [
            np.asarray([mean + pattern for mean in means], dtype=dtype),
            np.asarray([mean - pattern for mean in means], dtype=dtype),
        ],
        axis=0,
    )


def test_fixture_means_and_population_stds_are_per_channel():
    cases = _load_fixture("standardization_cases.json")
    training = _training_fixture()

    standardizer = ChannelStandardizer(channels=6).fit_training(training)

    np.testing.assert_allclose(
        standardizer.mean.reshape(-1), cases["expected_train_means"], rtol=0, atol=0
    )
    np.testing.assert_allclose(
        standardizer.std.reshape(-1), cases["expected_train_stds"], rtol=0, atol=0
    )
    assert standardizer.mean.shape == (1, 6, 1)
    assert standardizer.std.shape == (1, 6, 1)


def test_validation_test_and_synthetic_use_training_statistics_without_leakage():
    cases = _load_fixture("standardization_cases.json")
    training = _training_fixture()
    standardizer = ChannelStandardizer(channels=6).fit_training(training)

    held_out = _offset_fixture(cases["held_out_channel_means"])
    validation = _offset_fixture([100 + value for value in cases["channel_bases"]])
    synthetic = _offset_fixture(
        cases["synthetic_channel_means"], dtype=np.float64
    )

    transformed_test = standardizer.transform(held_out)
    transformed_validation = standardizer.apply(validation)
    transformed_synthetic = standardizer.transform(synthetic)

    # The output is deliberately not re-centered per split.  With train SD 1,
    # these means expose that all three inputs use the train means.
    np.testing.assert_allclose(
        transformed_test.mean(axis=(0, 2)),
        np.asarray(cases["held_out_channel_means"])
        - np.asarray(cases["expected_train_means"]),
    )
    np.testing.assert_allclose(
        transformed_validation.mean(axis=(0, 2)),
        100 + np.asarray(cases["channel_bases"])
        - np.asarray(cases["expected_train_means"]),
    )
    assert transformed_synthetic.dtype == np.float64
    np.testing.assert_allclose(
        standardizer.mean.reshape(-1), cases["expected_train_means"]
    )


def test_constant_channel_uses_the_standard_deviation_floor():
    training = _training_fixture()
    training[:, 0, :] = 7.0
    standardizer = ChannelStandardizer(channels=6).fit_training(training)

    assert standardizer.std[0, 0, 0] == pytest.approx(1e-8, rel=0, abs=0)
    transformed = standardizer.transform(training)
    assert np.all(transformed[:, 0, :] == 0)
    np.testing.assert_allclose(standardizer.inverse_transform(transformed), training)


def test_inverse_transform_round_trips_each_supported_split():
    standardizer = ChannelStandardizer(channels=6).fit_training(_training_fixture())
    split_arrays = (
        _training_fixture(),
        _offset_fixture([10, 20, 30, 40, 50, 60]),
        _offset_fixture([20, 30, 40, 50, 60, 70]),
    )

    for windows in split_arrays:
        transformed = standardizer.transform(windows)
        reconstructed = standardizer.inverse(transformed)
        np.testing.assert_allclose(reconstructed, windows, rtol=1e-6, atol=1e-6)


def test_shape_and_dtype_contract_is_strict_and_empty_batches_preserve_shape():
    standardizer = ChannelStandardizer(channels=6).fit_training(_training_fixture())
    windows = _training_fixture()

    transformed = standardizer.transform(windows)
    assert transformed.shape == windows.shape
    assert transformed.dtype == np.float32

    empty = np.empty((0, 6, 4), dtype=np.float32)
    empty_transformed = standardizer.transform(empty)
    assert empty_transformed.shape == empty.shape
    assert empty_transformed.dtype == empty.dtype

    with pytest.raises(ShapeValidationError):
        standardizer.transform(np.zeros((2, 6, 4, 1), dtype=np.float32))
    with pytest.raises(ShapeValidationError):
        standardizer.transform(np.zeros((2, 6, 0), dtype=np.float32))
    with pytest.raises(ShapeValidationError):
        standardizer.fit_training(np.zeros((0, 6, 4), dtype=np.float32))


def test_mismatched_channel_count_is_rejected_at_fit_and_apply():
    with pytest.raises(ChannelValidationError):
        ChannelStandardizer(channels=6).fit_training(
            np.zeros((2, 3, 4), dtype=np.float32)
        )

    standardizer = ChannelStandardizer().fit_training(_training_fixture())
    with pytest.raises(ChannelValidationError):
        standardizer.transform(np.zeros((2, 3, 4), dtype=np.float32))


def test_unfitted_use_and_nonfinite_data_are_rejected():
    standardizer = ChannelStandardizer(channels=6)
    windows = _training_fixture()

    with pytest.raises(NotFittedError):
        standardizer.transform(windows)
    with pytest.raises(NotFittedError):
        standardizer.inverse_transform(windows)
    with pytest.raises(NotFittedError):
        standardizer.to_metadata()

    nan_windows = windows.copy()
    nan_windows[0, 0, 0] = np.nan
    with pytest.raises(NonFiniteDataError):
        standardizer.fit_training(nan_windows)

    standardizer.fit_training(windows)
    with pytest.raises(NonFiniteDataError):
        standardizer.transform(nan_windows)
    with pytest.raises(NonFiniteDataError):
        standardizer.inverse_transform(nan_windows)


def test_metadata_is_json_safe_and_restores_a_fitted_transform():
    standardizer = ChannelStandardizer(channels=6).fit_training(
        _training_fixture(),
        training_subjects=[1, 2],
        validation_subjects=[3],
        held_out_subject=5,
    )

    metadata = standardizer.to_metadata()
    payload = json.dumps(metadata, sort_keys=True)
    restored = ChannelStandardizer.from_json(payload)

    assert metadata["fit_axes"] == [0, 2]
    assert metadata["ddof"] == 0
    assert metadata["training_only"] is True
    assert metadata["fit_stage"] == "post_validation_training_subjects_only"
    assert metadata["channels"] == 6
    assert metadata["mean_shape"] == [1, 6, 1]
    assert metadata["std_shape"] == [1, 6, 1]
    assert metadata["training_subjects"] == [1, 2]
    assert metadata["validation_subjects"] == [3]
    assert metadata["held_out_subject"] == 5
    np.testing.assert_allclose(
        restored.transform(_offset_fixture([10, 20, 30, 40, 50, 60])),
        standardizer.transform(_offset_fixture([10, 20, 30, 40, 50, 60])),
    )


def test_training_metadata_rejects_subject_overlap():
    with pytest.raises(ValueError, match="disjoint"):
        ChannelStandardizer(channels=6).fit_training(
            _training_fixture(), training_subjects=[1], validation_subjects=[1]
        )
    with pytest.raises(ValueError, match="held_out_subject"):
        ChannelStandardizer(channels=6).fit_training(
            _training_fixture(), training_subjects=[1], held_out_subject=1
        )
