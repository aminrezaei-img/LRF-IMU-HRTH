"""Synthetic-only regression tests for exact-window integrity auditing."""

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
DUPLICATE_FIXTURE = (
    REPOSITORY_ROOT / "tests" / "fixtures" / "synthetic" / "duplicate_audit_cases.json"
)
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from lrf_imu.data.audit import (  # noqa: E402
    CANONICAL_BYTES_POLICY,
    DuplicateWindowError,
    HASH_ALGORITHM,
    HISTORICAL_TRAIN_VALIDATION_ONLY,
    PUBLIC_AUDIT_SCOPE,
    audit_summary_to_json,
    audit_train_validation_only_compatibility,
    audit_window_duplicates,
    canonical_window_bytes,
    hash_window,
    hash_windows,
)


def _window(start: float) -> np.ndarray:
    """Return a deterministic synthetic [C, T] float32 window."""

    return np.array(
        [[start, start + 1.0, start + 2.0], [start + 3.0, start + 4.0, start + 5.0]],
        dtype=np.float32,
    )


def _split(*windows: np.ndarray) -> np.ndarray:
    return np.stack(windows, axis=0).astype(np.float32, copy=False)


def _fixture_split(names, windows):
    if not names:
        return np.empty((0, 2, 3), dtype=np.float32)
    return _split(*(windows[name] for name in names))


def test_canonical_bytes_match_observed_sha1_policy_and_fortran_input() -> None:
    window = _window(1.0)
    expected_bytes = np.ascontiguousarray(window).view(np.uint8).tobytes()

    assert canonical_window_bytes(window) == expected_bytes
    assert hash_window(window) == "5baa3a1be4e6d56160aa961c0da63c0de7ede5d7"
    assert hash_window(np.asfortranarray(window)) == hash_window(window)
    assert HASH_ALGORITHM == "sha1"
    assert "ascontiguousarray" in CANONICAL_BYTES_POLICY


def test_hash_windows_preserves_input_order_without_raw_values() -> None:
    windows = _split(_window(1.0), _window(7.0))

    assert hash_windows(windows) == [
        "5baa3a1be4e6d56160aa961c0da63c0de7ede5d7",
        "56a79bb78d508786fe03b970fec2802ab65e2297",
    ]


def test_synthetic_duplicate_fixture_cases_drive_clean_and_duplicate_audits() -> None:
    fixture = json.loads(DUPLICATE_FIXTURE.read_text(encoding="utf-8"))
    assert fixture["hash_algorithm"] == HASH_ALGORITHM
    windows = {"window_a": _window(1.0), "window_b": _window(7.0)}

    for case in fixture["cases"]:
        train = _fixture_split(case["train"], windows)
        validation = _fixture_split(case["validation"], windows)
        held_out_test = _fixture_split(case["held_out_test"], windows)
        summary = audit_window_duplicates(
            train,
            validation,
            held_out_test,
            raise_on_duplicate=False,
        )
        expected_pairs = {
            item.split(":", 1)[0] for item in case["expected_duplicates"]
        }
        actual_pairs = {
            item["comparison"] for item in summary["duplicate_comparisons"]
        }
        assert actual_pairs == expected_pairs
        assert summary["passed"] is (not expected_pairs)
        assert summary["metadata"]["counts"] == {
            "train": len(case["train"]),
            "validation": len(case["validation"]),
            "held_out_test": len(case["held_out_test"]),
        }


def test_public_clean_audit_covers_all_three_cross_split_boundaries() -> None:
    train = _split(_window(1.0))
    validation = _split(_window(7.0))
    held_out_test = _split(_window(13.0))

    summary = audit_window_duplicates(train, validation, held_out_test)

    assert summary["scope"] == PUBLIC_AUDIT_SCOPE
    assert summary["metadata"]["comparisons"] == [
        "train_validation",
        "train_held_out_test",
        "validation_held_out_test",
    ]
    assert summary["metadata"]["public_scope_improvement"] is True
    assert summary["metadata"]["historical_scope"] == "train_validation"
    assert summary["metadata"]["algorithm"] == "SHA-1"
    assert summary["metadata"]["hash_algorithm"] == "sha1"
    assert summary["metadata"]["window_shape"] == [2, 3]
    assert summary["metadata"]["counts"] == {
        "train": 1,
        "validation": 1,
        "held_out_test": 1,
    }
    assert summary["metadata"]["shapes"] == {
        "train": [1, 2, 3],
        "validation": [1, 2, 3],
        "held_out_test": [1, 2, 3],
    }
    assert summary["passed"] is True
    assert summary["has_duplicates"] is False
    assert all(not report["has_duplicates"] for report in summary["comparisons"].values())


@pytest.mark.parametrize(
    "duplicate_pair,held_out_test",
    [
        ("train_validation", _split(_window(13.0))),
        ("train_held_out_test", _split(_window(1.0))),
        ("validation_held_out_test", _split(_window(7.0))),
    ],
)
def test_public_audit_reports_each_cross_split_duplicate(
    duplicate_pair: str, held_out_test: np.ndarray
) -> None:
    train = _split(_window(1.0))
    validation = _split(_window(7.0))

    if duplicate_pair == "train_validation":
        validation = _split(_window(1.0))

    summary = audit_window_duplicates(train, validation, held_out_test)

    assert summary["passed"] is False
    assert summary["has_cross_split_duplicates"] is True
    assert summary["comparisons"][duplicate_pair]["duplicate_hashes"]
    assert {
        record["comparison"] for record in summary["duplicate_comparisons"]
    } == {duplicate_pair}
    assert summary["raw_values_persisted"] is False


def test_optional_within_split_reporting_is_exact_and_counted() -> None:
    repeated = _window(1.0)
    train = _split(repeated, repeated, _window(7.0))
    validation = _split(_window(13.0))
    held_out_test = _split(_window(19.0))

    summary = audit_window_duplicates(
        train,
        validation,
        held_out_test,
        include_within_split=True,
    )

    report = summary["within_split"]["train"]
    digest = hash_window(repeated)
    assert report["duplicate_hashes"] == [digest]
    assert report["duplicate_hash_counts"] == {digest: 2}
    assert report["duplicate_window_count"] == 1
    assert summary["has_within_split_duplicates"] is True
    assert summary["passed"] is False


def test_historical_compatibility_scope_is_explicit_and_does_not_claim_test_coverage() -> None:
    train = _split(_window(1.0))
    validation = _split(_window(7.0))

    summary = audit_window_duplicates(
        train,
        validation,
        compatibility_mode=HISTORICAL_TRAIN_VALIDATION_ONLY,
    )

    assert summary["scope"] == HISTORICAL_TRAIN_VALIDATION_ONLY
    assert summary["metadata"]["comparisons"] == ["train_validation"]
    assert summary["metadata"]["historical_scope_used"] is True
    assert summary["metadata"]["public_scope_improvement"] is False
    assert "held_out_test" not in summary["comparisons"]


def test_historical_adapter_preserves_strict_duplicate_failure_with_json_report() -> None:
    train = _split(_window(1.0))
    validation = _split(_window(1.0))

    with pytest.raises(DuplicateWindowError) as caught:
        audit_train_validation_only_compatibility(train, validation)

    report = caught.value.summary
    assert report["scope"] == HISTORICAL_TRAIN_VALIDATION_ONLY
    assert report["has_duplicates"] is True
    parsed = json.loads(audit_summary_to_json(report))
    assert parsed["comparisons"]["train_validation"]["unique_duplicate_count"] == 1


def test_strict_public_mode_can_fail_after_recording_all_duplicate_evidence() -> None:
    train = _split(_window(1.0))
    validation = _split(_window(7.0))
    held_out_test = _split(_window(1.0))

    with pytest.raises(DuplicateWindowError) as caught:
        audit_window_duplicates(
            train,
            validation,
            held_out_test,
            raise_on_duplicate=True,
        )

    assert caught.value.summary["comparisons"]["train_held_out_test"]["has_duplicates"]


def test_public_mode_requires_held_out_test_and_shape_errors_are_explicit() -> None:
    valid = _split(_window(1.0))

    with pytest.raises(ValueError, match="held_out_test_windows is required"):
        audit_window_duplicates(valid, valid)
    with pytest.raises(ValueError, match=r"shape \[N, C, T\]"):
        audit_window_duplicates(valid[0], valid, valid)
    with pytest.raises(ValueError, match=r"share \[C, T\] shape"):
        audit_window_duplicates(valid, _split(np.ones((3, 4), dtype=np.float32)), valid)
    nonnumeric = np.empty((1, 2, 3), dtype=object)
    nonnumeric.fill("not numeric")
    with pytest.raises(TypeError, match="numeric dtype"):
        audit_window_duplicates(nonnumeric, valid, valid)
    with pytest.raises(ValueError, match="positive C and T"):
        audit_window_duplicates(
            np.empty((0, 0, 3), dtype=np.float32), valid, valid
        )


def test_summary_is_json_safe_and_does_not_make_privacy_or_near_duplicate_claim() -> None:
    summary = audit_window_duplicates(
        _split(_window(1.0)),
        _split(_window(7.0)),
        _split(_window(13.0)),
        include_within_split=False,
    )

    encoded = audit_summary_to_json(summary)
    decoded = json.loads(encoded)
    assert decoded == summary
    assert summary["raw_values_persisted"] is False
    assert summary["privacy_claim"] is False
    assert summary["near_duplicate_detection"] is False
    assert summary["within_split"] == {}
    assert "participant" not in encoded.lower()
    assert "5baa3a1be4e6d56160aa961c0da63c0de7ede5d7" not in encoded
    assert hashlib.sha1(canonical_window_bytes(_window(1.0))).hexdigest() == hash_window(
        _window(1.0)
    )
