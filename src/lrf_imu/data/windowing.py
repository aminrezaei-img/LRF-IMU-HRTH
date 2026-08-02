"""Pure filtering and window construction for the public data boundary.

The compatibility path intentionally follows the paper-specific preprocessing
order recorded in the release contracts: rows whose labels are not one of the
four target activities are removed first, then contiguous runs are built from
the remaining target labels.  Consequently, an excluded-label gap can bridge
two portions of the same target activity.  This behavior is deliberately
scoped to the compatibility mode and is not a general REALDISP continuity
rule.

``strict_original_contiguity`` is the explicit alternative.  In that mode an
excluded raw row terminates the current run, so excluded rows can never bridge
two otherwise equal activity runs.  The default remains the compatibility
mode.

Inputs are time-major ``(T, C)`` arrays.  Outputs are ``(B, C, T_window)``
float32 windows and int64 encoded labels.  The functions do not mutate input
arrays, pad short runs, write files, or otherwise perform I/O.
"""

from __future__ import annotations

import operator
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, cast

import numpy as np


DEFAULT_WINDOW_LENGTH = 160
DEFAULT_HOP_LENGTH = 40

# These are raw activity codes.  They are intentionally kept distinct from
# the encoded labels returned by ``make_windows``.
TARGET_LABELS = (1, 3, 4, 33)
RAW_TO_ENCODED: Dict[int, int] = {1: 0, 3: 1, 4: 2, 33: 3}

# Descriptive aliases make the contract easy to discover without coupling this
# low-level module to the separately migrated activity-schema module.
DEFAULT_ALLOWED_LABELS = TARGET_LABELS
LABEL_ENCODING = RAW_TO_ENCODED


class WindowingError(ValueError):
    """Raised when a windowing input or option violates the public contract."""


def _positive_int(value: object, name: str) -> int:
    """Return an integer option, rejecting booleans and non-integral values."""

    if isinstance(value, (bool, np.bool_)):
        raise WindowingError("{} must be a positive integer".format(name))
    try:
        result = operator.index(cast(Any, value))
    except TypeError as exc:
        raise WindowingError("{} must be a positive integer".format(name)) from exc
    if result <= 0:
        raise WindowingError("{} must be a positive integer".format(name))
    return int(result)


def _resolve_length_alias(
    value: object,
    alias: Optional[object],
    default: int,
    name: str,
    alias_name: str,
) -> int:
    """Resolve a source-style ``win``/``hop`` spelling without ambiguity."""

    if alias is not None:
        alias_value = _positive_int(alias, alias_name)
        value_is_default = value == default
        value_checked = _positive_int(value, name)
        if not value_is_default and value_checked != alias_value:
            raise WindowingError(
                "{} and {} specify different values".format(name, alias_name)
            )
        value = alias_value
    return _positive_int(value, name)


def _coerce_signals(
    signals: object,
    labels_length: int,
    channel_count: Optional[object],
) -> np.ndarray:
    """Validate and convert time-major signal data to float32."""

    try:
        array = np.asarray(signals)
    except Exception as exc:  # pragma: no cover - NumPy supplies the detail.
        raise WindowingError("signals must be a numeric two-dimensional array") from exc

    if array.ndim == 1 and array.size == 0 and channel_count is not None:
        channels = _positive_int(channel_count, "channel_count")
        array = array.reshape(0, channels)
    elif array.ndim != 2:
        raise WindowingError("signals must have shape (time, channels)")

    if channel_count is not None:
        channels = _positive_int(channel_count, "channel_count")
        if array.shape[1] != channels:
            raise WindowingError(
                "channel_count={} does not match signals.shape[1]={}".format(
                    channels, array.shape[1]
                )
            )
    elif array.shape[1] <= 0:
        raise WindowingError("signals must contain at least one channel")

    if array.shape[0] != labels_length:
        raise WindowingError(
            "signals and labels must have the same number of time samples "
            "(got {} and {})".format(array.shape[0], labels_length)
        )

    try:
        converted = np.asarray(array, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as exc:
        raise WindowingError("signals must contain numeric values") from exc
    if not np.isfinite(converted).all():
        raise WindowingError("signals must contain only finite numeric values")
    return converted


def _coerce_labels(labels: object) -> np.ndarray:
    """Validate raw activity codes while accepting integer-valued log floats."""

    try:
        array = np.asarray(labels)
    except Exception as exc:  # pragma: no cover - NumPy supplies the detail.
        raise WindowingError("labels must be a one-dimensional numeric array") from exc
    if array.ndim != 1:
        raise WindowingError("labels must have shape (time,)")
    try:
        numeric = np.asarray(array, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise WindowingError("labels must contain numeric activity codes") from exc
    if not np.isfinite(numeric).all():
        raise WindowingError("labels must contain only finite numeric activity codes")
    if not np.equal(numeric, np.trunc(numeric)).all():
        raise WindowingError("labels must contain integer-valued activity codes")
    try:
        return numeric.astype(np.int64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise WindowingError("labels contain values outside the int64 range") from exc


def _normalise_allowed_labels(allowed_labels: Optional[Iterable[object]]) -> Tuple[int, ...]:
    if allowed_labels is None:
        allowed_labels = TARGET_LABELS
    if isinstance(allowed_labels, (str, bytes)):
        raise WindowingError("allowed_labels must be an iterable of integer codes")
    try:
        result = tuple(_positive_or_zero_int(label, "allowed label") for label in allowed_labels)
    except TypeError as exc:
        raise WindowingError("allowed_labels must be an iterable of integer codes") from exc
    if len(set(result)) != len(result):
        raise WindowingError("allowed_labels must not contain duplicates")
    return result


def _positive_or_zero_int(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise WindowingError("{} must be an integer".format(name))
    try:
        result = operator.index(cast(Any, value))
    except TypeError as exc:
        raise WindowingError("{} must be an integer".format(name)) from exc
    if result < 0:
        raise WindowingError("{} must not be negative".format(name))
    return int(result)


def _normalise_label_encoding(
    label_encoding: Optional[Mapping[object, object]],
    allowed_labels: Sequence[int],
) -> Dict[int, int]:
    source = RAW_TO_ENCODED if label_encoding is None else label_encoding
    if not isinstance(source, Mapping):
        raise WindowingError("label_encoding must be a mapping of raw to encoded labels")
    try:
        result = {
            _positive_or_zero_int(raw, "raw label"): _positive_or_zero_int(encoded, "encoded label")
            for raw, encoded in source.items()
        }
    except AttributeError as exc:
        raise WindowingError("label_encoding must be a mapping of raw to encoded labels") from exc
    missing = [label for label in allowed_labels if label not in result]
    if missing:
        raise WindowingError(
            "label_encoding is missing allowed raw label(s): {}".format(missing)
        )
    return result


def _resolve_strict_mode(
    strict_original_contiguity: object,
    mode: Optional[str],
    compatibility_mode: Optional[str],
) -> bool:
    if not isinstance(strict_original_contiguity, (bool, np.bool_)):
        raise WindowingError("strict_original_contiguity must be a boolean")
    selected = [value for value in (mode, compatibility_mode) if value is not None]
    if len(selected) > 1 and selected[0] != selected[1]:
        raise WindowingError("mode and compatibility_mode specify different modes")
    selected_mode = selected[0] if selected else None
    if selected_mode is None:
        return bool(strict_original_contiguity)
    if not isinstance(selected_mode, str):
        raise WindowingError("mode must be a supported string")
    normalised = selected_mode.strip().lower().replace("-", "_")
    if normalised in {"filter_before_runs", "compatibility", "default"}:
        mode_is_strict = False
    elif normalised in {"strict_original_contiguity", "strict", "original_contiguity"}:
        mode_is_strict = True
    else:
        raise WindowingError(
            "unsupported mode {!r}; use 'filter_before_runs' or "
            "'strict_original_contiguity'".format(selected_mode)
        )
    if bool(strict_original_contiguity) and not mode_is_strict:
        raise WindowingError("strict_original_contiguity conflicts with compatibility mode")
    return mode_is_strict


def _empty_result(channel_count: int, window_length: int) -> Tuple[np.ndarray, np.ndarray]:
    return (
        np.empty((0, channel_count, window_length), dtype=np.float32),
        np.empty((0,), dtype=np.int64),
    )


def _window_run(
    run_signals: np.ndarray,
    raw_label: int,
    window_length: int,
    hop_length: int,
    label_encoding: Mapping[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Window one single-activity run without padding."""

    run_length, channels = run_signals.shape
    if run_length < window_length:
        return _empty_result(channels, window_length)
    starts = range(0, run_length - window_length + 1, hop_length)
    count = len(range(0, run_length - window_length + 1, hop_length))
    windows = np.empty((count, channels, window_length), dtype=np.float32)
    for output_index, start in enumerate(starts):
        # The input is time-major; the public window contract is channel-major.
        windows[output_index] = run_signals[start : start + window_length].T
    encoded = np.full(
        (count,), label_encoding[raw_label], dtype=np.int64
    )
    return windows, encoded


def _compatibility_runs(
    signals: np.ndarray,
    labels: np.ndarray,
    allowed_labels: Sequence[int],
) -> List[Tuple[np.ndarray, int]]:
    """Filter excluded rows, then split the filtered sequence by label."""

    keep = np.isin(labels, np.asarray(allowed_labels, dtype=np.int64))
    filtered_signals = signals[keep]
    filtered_labels = labels[keep]
    if filtered_labels.size == 0:
        return []

    boundaries = np.flatnonzero(filtered_labels[1:] != filtered_labels[:-1]) + 1
    starts = np.concatenate((np.asarray([0], dtype=np.int64), boundaries))
    stops = np.concatenate((boundaries, np.asarray([filtered_labels.size], dtype=np.int64)))
    return [
        (filtered_signals[start:stop], int(filtered_labels[start]))
        for start, stop in zip(starts, stops)
    ]


def _strict_runs(
    signals: np.ndarray,
    labels: np.ndarray,
    allowed_labels: Sequence[int],
) -> List[Tuple[np.ndarray, int]]:
    """Split on excluded raw rows and on every target-label transition."""

    allowed = set(allowed_labels)
    runs: List[Tuple[np.ndarray, int]] = []
    start: Optional[int] = None
    current: Optional[int] = None
    for index, raw_label in enumerate(labels):
        label = int(raw_label)
        if label not in allowed:
            if start is not None:
                assert current is not None
                runs.append((signals[start:index], int(current)))
                start = None
                current = None
            continue
        if start is None:
            start = index
            current = label
        elif label != current:
            assert current is not None
            runs.append((signals[start:index], int(current)))
            start = index
            current = label
    if start is not None:
        assert current is not None
        runs.append((signals[start:], int(current)))
    return runs


def make_windows(
    signals: object,
    labels: object,
    window_length: object = DEFAULT_WINDOW_LENGTH,
    hop_length: object = DEFAULT_HOP_LENGTH,
    *,
    allowed_labels: Optional[Iterable[object]] = None,
    label_encoding: Optional[Mapping[object, object]] = None,
    strict_original_contiguity: object = False,
    channel_count: Optional[object] = None,
    mode: Optional[str] = None,
    compatibility_mode: Optional[str] = None,
    win: Optional[object] = None,
    hop: Optional[object] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Filter labels and return complete, activity-bounded windows.

    Parameters
    ----------
    signals:
        Numeric time-major array with shape ``(T, C)``.
    labels:
        Raw activity-code vector with shape ``(T,)``.  Integer-valued floats,
        as produced by numeric tab-separated log loading, are accepted.
    window_length, hop_length:
        Positive sample counts.  ``win`` and ``hop`` are accepted as explicit
        source-compatible keyword aliases.
    allowed_labels, label_encoding:
        Raw target codes and their raw-to-encoded mapping.  Defaults are the
        four paper-specific codes ``1, 3, 4, 33`` mapped to ``0..3``.
    strict_original_contiguity:
        When true, excluded raw rows terminate runs.  It is false by default.
    channel_count:
        Optional channel count used to validate non-empty inputs and to recover
        the shape of an empty one-dimensional signal container.
    mode, compatibility_mode:
        Explicit string spellings for either ``filter_before_runs`` (default)
        or ``strict_original_contiguity``.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        ``(windows, encoded_labels)`` with shapes ``(B, C, window_length)``
        and ``(B,)`` and dtypes float32 and int64 respectively.
    """

    window_length = _resolve_length_alias(
        window_length, win, DEFAULT_WINDOW_LENGTH, "window_length", "win"
    )
    hop_length = _resolve_length_alias(
        hop_length, hop, DEFAULT_HOP_LENGTH, "hop_length", "hop"
    )
    strict = _resolve_strict_mode(
        strict_original_contiguity, mode, compatibility_mode
    )
    allowed = _normalise_allowed_labels(allowed_labels)
    encoding = _normalise_label_encoding(label_encoding, allowed)
    raw_labels = _coerce_labels(labels)
    signal_array = _coerce_signals(signals, raw_labels.size, channel_count)
    channels = int(signal_array.shape[1])

    runs = (
        _strict_runs(signal_array, raw_labels, allowed)
        if strict
        else _compatibility_runs(signal_array, raw_labels, allowed)
    )
    if not runs:
        return _empty_result(channels, window_length)

    window_parts: List[np.ndarray] = []
    label_parts: List[np.ndarray] = []
    for run_signals, raw_label in runs:
        windows, encoded = _window_run(
            run_signals, raw_label, window_length, hop_length, encoding
        )
        if encoded.size:
            window_parts.append(windows)
            label_parts.append(encoded)
    if not window_parts:
        return _empty_result(channels, window_length)
    return (
        np.concatenate(window_parts, axis=0).astype(np.float32, copy=False),
        np.concatenate(label_parts, axis=0).astype(np.int64, copy=False),
    )


def window_signal(*args: Any, **kwargs: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Descriptive alias for :func:`make_windows`."""

    return make_windows(*args, **kwargs)


def make_windows_per_subject(*args: Any, **kwargs: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Compatibility alias used by callers that process one subject at a time."""

    return make_windows(*args, **kwargs)


__all__ = [
    "DEFAULT_ALLOWED_LABELS",
    "DEFAULT_HOP_LENGTH",
    "DEFAULT_WINDOW_LENGTH",
    "LABEL_ENCODING",
    "RAW_TO_ENCODED",
    "TARGET_LABELS",
    "WindowingError",
    "make_windows",
    "make_windows_per_subject",
    "window_signal",
]
