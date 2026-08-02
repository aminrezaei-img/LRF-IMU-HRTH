"""Exact-window integrity checks for public data splits.

The historical implementation checked only train versus validation windows and
hashed each window with ``sha1(np.ascontiguousarray(window).view(np.uint8))``.
This module keeps that byte-level policy for parity while making the public
default cover train/validation, train/held-out-test, and validation/held-out-
test.  The audit is intentionally an integrity check, not a privacy or
near-duplicate analysis.

No input window values are returned or written by this module.  Reports contain
only shapes, counts, dtypes, and exact-match digests.
"""

from collections import Counter
import hashlib
import json
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np


HASH_ALGORITHM = "sha1"
"""Digest algorithm used by the observed source and synthetic contract."""

PUBLIC_AUDIT_SCOPE = "public_all_split_pairs"
"""Name of the expanded public audit scope."""

HISTORICAL_TRAIN_VALIDATION_ONLY = "historical_train_validation_only"
"""Explicit compatibility scope for the observed legacy audit."""

LEGACY_TRAIN_VALIDATION_ALIAS = "train_validation"
"""Short historical alias retained for parity callers."""

CANONICAL_BYTES_POLICY = (
    "np.ascontiguousarray(window).view(np.uint8).tobytes(order='C')"
)

_PUBLIC_COMPARISONS = (
    "train_validation",
    "train_held_out_test",
    "validation_held_out_test",
)


class DuplicateWindowError(ValueError):
    """Raised by strict audit calls when an exact duplicate is found.

    The JSON-safe report is available as :attr:`summary` so callers can retain
    evidence without serializing any raw windows.
    """

    def __init__(self, summary: Mapping[str, Any]) -> None:
        self.summary = dict(summary)
        comparisons = summary.get("duplicate_comparisons", [])
        count = len(comparisons)
        super().__init__(
            "Exact duplicate windows detected in {} comparison(s).".format(count)
        )


def _array_shape(value: np.ndarray) -> List[int]:
    """Return a JSON-safe list for a NumPy shape."""

    return [int(size) for size in value.shape]


def _coerce_window(window: Any, name: str = "window") -> np.ndarray:
    """Validate and return one numeric ``[C, T]`` window."""

    try:
        array = np.asarray(window)
    except Exception as exc:
        raise ValueError("{} could not be converted to an array".format(name)) from exc

    if array.ndim != 2:
        raise ValueError(
            "{} must have shape [C, T]; got {}".format(name, _array_shape(array))
        )
    if array.shape[0] <= 0 or array.shape[1] <= 0:
        raise ValueError(
            "{} must have positive channel and time dimensions; got {}".format(
                name, _array_shape(array)
            )
        )
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(
            "{} must have a numeric dtype; got {}".format(name, array.dtype)
        )
    return array


def _coerce_windows(windows: Any, split_name: str) -> np.ndarray:
    """Validate and return a split with shape ``[N, C, T]``.

    Empty splits are valid when they retain the channel/time dimensions, e.g.
    ``np.empty((0, 6, 160), dtype=np.float32)``.
    """

    try:
        array = np.asarray(windows)
    except Exception as exc:
        raise ValueError(
            "{} windows could not be converted to an array".format(split_name)
        ) from exc

    if array.ndim != 3:
        raise ValueError(
            "{} windows must have shape [N, C, T]; got {}".format(
                split_name, _array_shape(array)
            )
        )
    if array.shape[1] <= 0 or array.shape[2] <= 0:
        raise ValueError(
            "{} windows must have positive C and T dimensions; got {}".format(
                split_name, _array_shape(array)
            )
        )
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(
            "{} windows must have a numeric dtype; got {}".format(
                split_name, array.dtype
            )
        )
    return array


def canonical_window_bytes(window: Any) -> bytes:
    """Return the source-compatible canonical bytes for one exact window.

    The array dtype and byte order are preserved.  Only C-contiguity is
    canonicalized; numeric values are not rounded or converted.  Shape and
    labels are deliberately not included because the observed source hashes
    only the contiguous window payload.
    """

    array = _coerce_window(window)
    contiguous = np.ascontiguousarray(array)
    return contiguous.view(np.uint8).tobytes(order="C")


def hash_window(window: Any) -> str:
    """Return the lowercase SHA-1 digest of one exact window."""

    return hashlib.sha1(canonical_window_bytes(window)).hexdigest()


def hash_windows(windows: Any) -> List[str]:
    """Hash all windows in a validated split in deterministic input order."""

    array = _coerce_windows(windows, "windows")
    return [hash_window(window) for window in array]


# Private/source-shaped aliases make parity probes straightforward while the
# named public functions above remain the preferred API.
_hash_window = hash_window
canonical_bytes = canonical_window_bytes


def _split_metadata(array: np.ndarray) -> Dict[str, Any]:
    """Build JSON-safe shape/count metadata for one split."""

    return {
        "count": int(array.shape[0]),
        "shape": _array_shape(array),
        "window_shape": [int(array.shape[1]), int(array.shape[2])],
        "dtype": str(array.dtype),
    }


def _hash_counts(array: np.ndarray) -> Counter:
    """Count exact hashes without retaining window payloads."""

    return Counter(hash_window(window) for window in array)


def _pair_report(
    left_name: str,
    right_name: str,
    left_hashes: Counter,
    right_hashes: Counter,
) -> Dict[str, Any]:
    """Build a deterministic report for one pair of split hash multisets."""

    duplicate_hashes = sorted(set(left_hashes).intersection(right_hashes))
    left_matches = sum(left_hashes[digest] for digest in duplicate_hashes)
    right_matches = sum(right_hashes[digest] for digest in duplicate_hashes)
    return {
        "left_split": left_name,
        "right_split": right_name,
        "duplicate_hashes": duplicate_hashes,
        "unique_duplicate_count": int(len(duplicate_hashes)),
        "left_matching_window_count": int(left_matches),
        "right_matching_window_count": int(right_matches),
        "has_duplicates": bool(duplicate_hashes),
        "comparison_type": "exact_hash_intersection",
        "near_duplicate_detection": False,
    }


def _within_split_report(split_name: str, hashes: Counter) -> Dict[str, Any]:
    """Build an optional report for repeated hashes within one split."""

    duplicate_counts = {
        digest: int(count)
        for digest, count in sorted(hashes.items())
        if count > 1
    }
    return {
        "split": split_name,
        "duplicate_hashes": list(duplicate_counts),
        "duplicate_hash_counts": duplicate_counts,
        "unique_duplicate_count": int(len(duplicate_counts)),
        "duplicate_window_count": int(sum(count - 1 for count in duplicate_counts.values())),
        "has_duplicates": bool(duplicate_counts),
        "comparison_type": "exact_hash_repetition",
        "near_duplicate_detection": False,
    }


def _normalise_mode(mode: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """Return canonical mode and the alias supplied by a caller."""

    if mode is None:
        return None, None
    if not isinstance(mode, str):
        raise ValueError(
            "compatibility_mode must be a named string or None; got {}".format(
                type(mode).__name__
            )
        )
    if mode in {"public", PUBLIC_AUDIT_SCOPE}:
        return None, mode
    if mode in {
        HISTORICAL_TRAIN_VALIDATION_ONLY,
        LEGACY_TRAIN_VALIDATION_ALIAS,
        "legacy_train_validation_only",
    }:
        return HISTORICAL_TRAIN_VALIDATION_ONLY, mode
    raise ValueError(
        "Unknown compatibility_mode {!r}; use {!r} or None".format(
            mode, HISTORICAL_TRAIN_VALIDATION_ONLY
        )
    )


def _validate_common_shape(
    split_arrays: Mapping[str, np.ndarray],
) -> Tuple[int, int]:
    """Require all present splits to share the same ``[C, T]`` shape."""

    shapes = {name: tuple(array.shape[1:]) for name, array in split_arrays.items()}
    expected = next(iter(shapes.values()))
    mismatches = {
        name: list(shape)
        for name, shape in shapes.items()
        if shape != expected
    }
    if mismatches:
        expected_list = [int(size) for size in expected]
        raise ValueError(
            "All split windows must share [C, T] shape {}; mismatched splits: {}".format(
                expected_list, mismatches
            )
        )
    return int(expected[0]), int(expected[1])


def audit_window_duplicates(
    train_windows: Any,
    validation_windows: Any,
    held_out_test_windows: Optional[Any] = None,
    *,
    include_within_split: bool = False,
    compatibility_mode: Optional[str] = None,
    raise_on_duplicate: bool = False,
) -> Dict[str, Any]:
    """Audit exact duplicate windows across public split boundaries.

    Parameters
    ----------
    train_windows, validation_windows, held_out_test_windows:
        Numeric arrays with shape ``[N, C, T]``.  Public mode requires all
        three arrays so the expanded scope cannot silently regress to the
        historical two-way check.  The explicitly named
        :data:`HISTORICAL_TRAIN_VALIDATION_ONLY` mode accepts only train and
        validation and compares that pair.
    include_within_split:
        Also report repeated exact hashes within each supplied split.
    compatibility_mode:
        ``None`` (the public all-pairs scope), or
        ``"historical_train_validation_only"``.  The short
        ``"train_validation"`` alias is accepted for fixture/parity callers,
        but output metadata always records the canonical historical name.
    raise_on_duplicate:
        If true, raise :class:`DuplicateWindowError` after constructing the
        JSON-safe report.  The default returns the report so callers can
        inspect all affected comparisons; strict compatibility callers can
        opt into the observed source's raising behavior.

    Returns
    -------
    dict
        A JSON-safe report containing only metadata and SHA-1 digests.
    """

    if not isinstance(include_within_split, bool):
        raise TypeError("include_within_split must be a boolean")
    if not isinstance(raise_on_duplicate, bool):
        raise TypeError("raise_on_duplicate must be a boolean")

    mode, supplied_alias = _normalise_mode(compatibility_mode)
    train = _coerce_windows(train_windows, "train")
    validation = _coerce_windows(validation_windows, "validation")

    split_arrays = {"train": train, "validation": validation}
    if mode is None:
        if held_out_test_windows is None:
            raise ValueError(
                "held_out_test_windows is required for the public all-pairs audit; "
                "use compatibility_mode={!r} for historical train/validation-only coverage".format(
                    HISTORICAL_TRAIN_VALIDATION_ONLY
                )
            )
        split_arrays["held_out_test"] = _coerce_windows(
            held_out_test_windows, "held_out_test"
        )

    channels, time_steps = _validate_common_shape(split_arrays)
    split_hashes = {
        name: _hash_counts(array) for name, array in split_arrays.items()
    }
    split_info = {
        name: _split_metadata(array) for name, array in split_arrays.items()
    }

    comparison_inputs: Tuple[Tuple[str, str], ...]
    if mode is None:
        comparison_names = list(_PUBLIC_COMPARISONS)
        comparison_inputs = (
            ("train", "validation"),
            ("train", "held_out_test"),
            ("validation", "held_out_test"),
        )
        scope = PUBLIC_AUDIT_SCOPE
        scope_description = (
            "Intentional public-release improvement over the observed "
            "train-versus-validation-only audit: all three split boundaries "
            "are checked for exact window reuse."
        )
        historical_scope_used = False
    else:
        comparison_names = ["train_validation"]
        comparison_inputs = (("train", "validation"),)
        scope = HISTORICAL_TRAIN_VALIDATION_ONLY
        scope_description = (
            "Compatibility coverage matching the observed historical audit; "
            "held-out-test boundaries are intentionally not evaluated."
        )
        historical_scope_used = True

    comparisons = {
        name: _pair_report(left, right, split_hashes[left], split_hashes[right])
        for name, (left, right) in zip(comparison_names, comparison_inputs)
    }

    if include_within_split:
        within_split = {
            name: _within_split_report(name, split_hashes[name])
            for name in split_arrays
        }
    else:
        within_split = {}

    duplicate_records = [
        {"comparison": name, "hash": digest}
        for name in comparison_names
        for digest in comparisons[name]["duplicate_hashes"]
    ]
    within_duplicate_records = [
        {"split": name, "hash": digest}
        for name, report in within_split.items()
        for digest in report["duplicate_hashes"]
    ]

    has_cross_split_duplicates = bool(duplicate_records)
    has_within_split_duplicates = bool(within_duplicate_records)
    metadata = {
        "scope": scope,
        "scope_description": scope_description,
        "comparisons": list(comparison_names),
        "algorithm": "SHA-1",
        "hash_algorithm": HASH_ALGORITHM,
        "digest_format": "lowercase hexadecimal",
        "canonical_bytes_policy": CANONICAL_BYTES_POLICY,
        "canonical_order": "C-contiguous",
        "dtype_policy": "preserve dtype and byte order; no numeric conversion",
        "shape_policy": "inputs are [N, C, T]; each hashed window is [C, T]; shape is not hashed",
        "window_shape": [channels, time_steps],
        "counts": {
            name: int(info["count"]) for name, info in split_info.items()
        },
        "shapes": {
            name: list(info["shape"]) for name, info in split_info.items()
        },
        "dtypes": {
            name: info["dtype"] for name, info in split_info.items()
        },
        "include_within_split": include_within_split,
        "compatibility_mode": mode,
        "compatibility_alias": supplied_alias if mode is not None else None,
        "historical_scope_used": historical_scope_used,
        "public_scope_improvement": not historical_scope_used,
        "historical_scope": LEGACY_TRAIN_VALIDATION_ALIAS,
        "near_duplicate_detection": False,
        "privacy_claim": False,
        "privacy_statement": (
            "Exact-byte integrity only; this report is not a privacy guarantee."
        ),
        "raw_values_persisted": False,
    }

    summary = {
        "schema_version": "3A.exact-window-duplicate-audit.1",
        "scope": scope,
        "hash_algorithm": HASH_ALGORITHM,
        "canonical_bytes_policy": CANONICAL_BYTES_POLICY,
        "metadata": metadata,
        "splits": split_info,
        "comparisons": comparisons,
        "within_split": within_split,
        "duplicate_hashes": {
            name: list(comparisons[name]["duplicate_hashes"])
            for name in comparison_names
        },
        "duplicate_comparisons": duplicate_records,
        "within_split_duplicates": within_duplicate_records,
        "has_cross_split_duplicates": has_cross_split_duplicates,
        "has_within_split_duplicates": has_within_split_duplicates,
        "has_duplicates": has_cross_split_duplicates or has_within_split_duplicates,
        "passed": not (has_cross_split_duplicates or has_within_split_duplicates),
        "raw_values_persisted": False,
        "near_duplicate_detection": False,
        "privacy_claim": False,
    }

    if raise_on_duplicate and summary["has_duplicates"]:
        raise DuplicateWindowError(summary)
    return summary


def audit_train_validation_only_compatibility(
    train_windows: Any,
    validation_windows: Any,
    *,
    include_within_split: bool = False,
    raise_on_duplicate: bool = True,
) -> Dict[str, Any]:
    """Run the explicitly named historical train/validation-only audit.

    This adapter mirrors the observed source boundary and defaults to its
    historical fail-fast behavior.  It must not be used as the public default
    because it cannot establish held-out-test non-overlap.
    """

    return audit_window_duplicates(
        train_windows,
        validation_windows,
        compatibility_mode=HISTORICAL_TRAIN_VALIDATION_ONLY,
        include_within_split=include_within_split,
        raise_on_duplicate=raise_on_duplicate,
    )


# Readable aliases for callers migrating from the source-shaped terminology.
audit_train_validation_only = audit_train_validation_only_compatibility
audit_split_duplicates = audit_window_duplicates


def assert_no_duplicate_windows(summary: Mapping[str, Any]) -> Dict[str, Any]:
    """Raise :class:`DuplicateWindowError` when a report is not clean."""

    if not isinstance(summary, Mapping):
        raise TypeError("summary must be a mapping returned by an audit function")
    if bool(summary.get("has_duplicates", False)):
        raise DuplicateWindowError(summary)
    return dict(summary)


def audit_summary_to_json(summary: Mapping[str, Any], *, indent: Optional[int] = 2) -> str:
    """Serialize an audit report with strict JSON-safe settings."""

    if not isinstance(summary, Mapping):
        raise TypeError("summary must be a mapping returned by an audit function")
    return json.dumps(
        dict(summary),
        indent=indent,
        sort_keys=True,
        allow_nan=False,
    )


to_json = audit_summary_to_json


__all__ = [
    "CANONICAL_BYTES_POLICY",
    "DuplicateWindowError",
    "HASH_ALGORITHM",
    "HISTORICAL_TRAIN_VALIDATION_ONLY",
    "LEGACY_TRAIN_VALIDATION_ALIAS",
    "PUBLIC_AUDIT_SCOPE",
    "assert_no_duplicate_windows",
    "audit_split_duplicates",
    "audit_summary_to_json",
    "audit_train_validation_only",
    "audit_train_validation_only_compatibility",
    "audit_window_duplicates",
    "canonical_bytes",
    "canonical_window_bytes",
    "hash_window",
    "hash_windows",
    "to_json",
]
