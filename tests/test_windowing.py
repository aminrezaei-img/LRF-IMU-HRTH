"""Focused regression tests for the public filtering/windowing contract."""

from pathlib import Path
import sys

import numpy as np
import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from lrf_imu.data.windowing import WindowingError, make_windows  # noqa: E402


def _signals(length: int, channels: int = 2) -> np.ndarray:
    return np.arange(length * channels, dtype=np.float64).reshape(length, channels)


def test_default_filter_before_runs_bridges_excluded_label_gap() -> None:
    signals = _signals(6, channels=2)
    labels = np.array([1, 1, 99, 99, 1, 1])
    windows, encoded = make_windows(signals, labels, window_length=4, hop_length=2)
    expected = signals[[0, 1, 4, 5]].T.astype(np.float32)
    assert windows.shape == (1, 2, 4)
    np.testing.assert_array_equal(windows[0], expected)
    np.testing.assert_array_equal(encoded, np.array([0], dtype=np.int64))


def test_strict_original_contiguity_keeps_excluded_rows_as_boundaries() -> None:
    signals = _signals(6, channels=2)
    labels = np.array([1, 1, 99, 99, 1, 1])
    windows, encoded = make_windows(
        signals, labels, window_length=2, hop_length=1,
        strict_original_contiguity=True,
    )
    assert windows.shape == (2, 2, 2)
    np.testing.assert_array_equal(
        windows[:, 0, :], np.array([[0, 2], [8, 10]], dtype=np.float32)
    )
    np.testing.assert_array_equal(encoded, np.array([0, 0], dtype=np.int64))
    compatibility, _ = make_windows(signals, labels, window_length=2, hop_length=1)
    assert compatibility.shape == (3, 2, 2)


def test_strict_mode_can_be_selected_by_explicit_name() -> None:
    labels = np.array([1, 1, 99, 99, 1, 1])
    strict, _ = make_windows(
        _signals(6), labels, window_length=4, hop_length=1,
        mode="strict_original_contiguity",
    )
    assert strict.shape == (0, 2, 4)


def test_windows_do_not_cross_activity_boundaries_and_use_complete_windows_only() -> None:
    signals = _signals(12, channels=1)
    labels = np.array([1] * 6 + [3] * 6)
    windows, encoded = make_windows(signals, labels, window_length=4, hop_length=2)
    assert windows.shape == (4, 1, 4)
    np.testing.assert_array_equal(encoded, np.array([0, 0, 1, 1], dtype=np.int64))
    np.testing.assert_array_equal(
        windows[:, 0, :],
        np.array(
            [[0, 1, 2, 3], [2, 3, 4, 5], [6, 7, 8, 9], [8, 9, 10, 11]],
            dtype=np.float32,
        ),
    )


def test_short_and_empty_inputs_return_configurable_empty_shape() -> None:
    short_windows, short_labels = make_windows(
        _signals(3, channels=3), np.array([1, 1, 1]), window_length=4, hop_length=2
    )
    assert short_windows.shape == (0, 3, 4)
    assert short_windows.dtype == np.float32
    assert short_labels.shape == (0,)
    assert short_labels.dtype == np.int64
    empty_windows, empty_labels = make_windows(
        np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int64),
        window_length=7, hop_length=3, channel_count=4,
    )
    assert empty_windows.shape == (0, 4, 7)
    assert empty_windows.dtype == np.float32
    assert empty_labels.shape == (0,)
    assert empty_labels.dtype == np.int64


def test_orientation_dtype_and_raw_to_encoded_label_mapping() -> None:
    signals = _signals(16, channels=3)
    labels = np.repeat(np.array([1, 3, 4, 33], dtype=np.float64), 4)
    windows, encoded = make_windows(signals, labels, window_length=4, hop_length=1)
    assert windows.shape == (4, 3, 4)
    assert windows.dtype == np.float32
    assert encoded.dtype == np.int64
    np.testing.assert_array_equal(windows[0], signals[:4].T.astype(np.float32))
    np.testing.assert_array_equal(encoded, np.array([0, 1, 2, 3], dtype=np.int64))


def test_positive_window_and_hop_options_and_source_aliases() -> None:
    signals = _signals(4)
    labels = np.ones(4, dtype=np.int64)
    windows, _ = make_windows(signals, labels, win=4, hop=2)
    assert windows.shape == (1, 2, 4)
    for kwargs in (
        {"window_length": 0}, {"hop_length": 0}, {"window_length": -1},
        {"hop_length": -1}, {"window_length": 4.5},
    ):
        with pytest.raises(WindowingError):
            make_windows(signals, labels, **kwargs)


@pytest.mark.parametrize(
    "signals, labels, expected_message",
    [
        (_signals(3), np.ones(2), "same number of time samples"),
        (np.ones((2, 3, 1)), np.ones(2), r"shape \(time, channels\)"),
        (_signals(2), np.array([1, 1.5]), "integer-valued"),
        (_signals(2), np.array([1, np.nan]), "finite"),
        (np.array([[1, "bad"]], dtype=object), np.array([1]), "numeric"),
    ],
)
def test_malformed_inputs_raise_clear_errors(signals, labels, expected_message) -> None:
    with pytest.raises(WindowingError, match=expected_message):
        make_windows(signals, labels, window_length=2, hop_length=1)


def test_mode_conflicts_and_unknown_activity_encoding_are_rejected() -> None:
    signals = _signals(2)
    labels = np.array([7, 7])
    with pytest.raises(WindowingError, match="missing allowed"):
        make_windows(signals, labels, window_length=2, hop_length=1, allowed_labels=[7])
    with pytest.raises(WindowingError, match="conflicts"):
        make_windows(
            signals, np.array([1, 1]), window_length=2, hop_length=1,
            strict_original_contiguity=True, mode="filter_before_runs",
        )
