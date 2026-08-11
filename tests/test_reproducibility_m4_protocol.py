from __future__ import annotations

import json
from pathlib import Path

import pytest

from lrf_imu import reproducibility
from lrf_imu.reproducibility import _reference_comparison


def test_reduced_smoke_sample_count_is_labeled_protocol_mismatch(
    tmp_path: Path,
) -> None:
    reference = {
        "gate_c_6ch_rf": {
            "folds": [
                {
                    "held_out_subject": 1,
                    "scenarios": {
                        "trtr": {
                            "f1_macro": {
                                "reference": 1.0,
                                "comparison_basis": "unrounded_json",
                            }
                        }
                    },
                }
            ]
        }
    }
    path = tmp_path / "reference.json"
    path.write_text(json.dumps(reference), encoding="utf-8")
    current = [
        {
            "held_out_subject": 1,
            "records": [
                {
                    "scenario": "trtr",
                    "f1_macro": 1.0,
                    "accuracy": 1.0,
                    "retention_ratio": 1.0,
                }
            ],
        }
    ]
    comparison = _reference_comparison(
        str(path),
        sensor="six_channel",
        classifier="rf",
        folds=current,
        samples_per_class=1,
    )
    assert comparison is not None
    assert comparison["status"] == "protocol_mismatch_descriptive_only"
    assert comparison["historical_samples_per_class"] == 500
    assert comparison["current_samples_per_class"] == 1
    assert comparison["parity_claim"] is False


def test_resume_rejects_incompatible_implementation_version(
    tmp_path: Path,
) -> None:
    payload = {
        "schema_version": reproducibility.MANIFEST_SCHEMA,
        "implementation_version": "m4.incompatible-fixture",
        "run_fingerprint": "same",
        "folds": {},
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="incompatible implementation version"):
        reproducibility._load_resume_manifest(path, {"run_fingerprint": "same"})
