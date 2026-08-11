from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys

import numpy as np
import pytest

from lrf_imu.cli import build_parser
from lrf_imu.reproducibility import (
    CANONICAL_SUBJECTS,
    _checkpoint_path,
    _reference_comparison,
    run_reproduce_core,
)


def _args(tmp_path: Path, *extra: str):
    parser = build_parser()
    return parser.parse_args(
        [
            "reproduce-core",
            "--data-root",
            str(tmp_path / "data"),
            "--checkpoint-root",
            str(tmp_path / "checkpoints"),
            "--output-root",
            str(tmp_path / "output"),
            *extra,
        ]
    )


def _windows(samples_per_class: int, channels: int = 3):
    labels = np.repeat(np.arange(4, dtype=np.int64), samples_per_class)
    windows = np.zeros((labels.size, channels, 160), dtype=np.float32)
    for index, label in enumerate(labels):
        windows[index].fill(float(label * 10 + index % samples_per_class))
    return windows, labels


def _prepared() -> SimpleNamespace:
    train_x, train_y = _windows(5)
    test_x, test_y = _windows(2)
    return SimpleNamespace(
        train_windows=train_x,
        train_labels=train_y,
        validation_windows=np.empty((0, 3, 160), dtype=np.float32),
        validation_labels=np.empty((0,), dtype=np.int64),
        held_out_test_windows=test_x,
        held_out_test_labels=test_y,
        sensor_schema=SimpleNamespace(channel_count=3),
        split=SimpleNamespace(train_subjects=(2, 3, 5), validation_subjects=(8,)),
    )


def test_twelve_fold_dry_run_is_foreign_cwd_safe_and_no_write(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    output = tmp_path / "not-created"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(root / "src")
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    command = [
        sys.executable,
        "-B",
        "-m",
        "lrf_imu",
        "reproduce-core",
        "--data-root",
        str(tmp_path / "not-read"),
        "--checkpoint-root",
        str(tmp_path / "not-read-either"),
        "--output-root",
        str(output),
        "--all-folds",
        "--dry-run",
    ]
    completed = subprocess.run(
        command,
        cwd=foreign,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert [fold["held_out_subject"] for fold in payload["folds"]] == list(
        CANONICAL_SUBJECTS
    )
    assert payload["write_permission_requested"] is False
    assert output.exists() is False


def test_non_dry_run_requires_explicit_write_permission_before_loading(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import lrf_imu.reproducibility as reproduction

    monkeypatch.setattr(
        reproduction,
        "prepare_data",
        lambda **kwargs: pytest.fail("data must not be read without permission"),
    )
    args = _args(tmp_path, "--held-out-subject", "1", "--scenario", "trtr")
    with pytest.raises(ValueError, match="--write-results"):
        run_reproduce_core(args)
    assert (tmp_path / "output").exists() is False


def test_checkpoint_root_supports_source_and_model_weight_roots(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    expected = (
        source_root
        / "Results"
        / "model_weights"
        / "flow_weights"
        / "6CH"
        / "full"
        / "subject_01"
        / "flow_unet_best.pt"
    )
    expected.parent.mkdir(parents=True)
    expected.write_bytes(b"safe fixture, not a checkpoint")
    args = _args(
        tmp_path,
        "--held-out-subject",
        "1",
        "--checkpoint-root",
        str(source_root),
        "--dry-run",
    )
    assert _checkpoint_path(args, 1, "flow", require_exists=True) == expected.resolve()


def test_synthetic_smoke_writes_structured_manifest_then_resumes_without_work(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import lrf_imu.reproducibility as reproduction

    synthetic_x, synthetic_y = _windows(3)
    calls = {"prepare": 0, "generate": 0}
    checkpoint_root = tmp_path / "checkpoints" / "model_weights"
    vae_checkpoint = (
        checkpoint_root / "vae_weights/3CH/ablation/subject_01/vae_s4_z48.pt"
    )
    flow_checkpoint = (
        checkpoint_root / "flow_weights/3CH/ablation/subject_01/flow_unet_best.pt"
    )
    for checkpoint in (vae_checkpoint, flow_checkpoint):
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
    vae_checkpoint.write_bytes(b"vae fixture")
    flow_checkpoint.write_bytes(b"flow fixture")


    def fake_prepare(**kwargs):
        calls["prepare"] += 1
        return _prepared()

    def fake_generate(*args, **kwargs):
        calls["generate"] += 1
        return (
            synthetic_x,
            synthetic_y,
            {
                "status": "fresh",
                "array_path": str(tmp_path / "external.npz"),
                "manifest_path": str(tmp_path / "external.manifest.json"),
                "identity": {
                    "vae_checkpoint_sha256": reproduction.sha256_file(vae_checkpoint),
                    "flow_checkpoint_sha256": reproduction.sha256_file(flow_checkpoint),
                    "implementation_version": reproduction.IMPLEMENTATION_VERSION,
                },
                "array_sha256": "a" * 64,
            },
        )

    monkeypatch.setattr(reproduction, "prepare_data", fake_prepare)
    monkeypatch.setattr(reproduction, "_materialize_synthetic", fake_generate)
    args = _args(
        tmp_path,
        "--held-out-subject",
        "1",
        "--sensor",
        "three_channel",
        "--samples-per-class",
        "3",
        "--write-results",
    )
    assert run_reproduce_core(args) == 0
    manifest_path = tmp_path / "output" / "reproduce_core_manifest.json"
    report_path = tmp_path / "output" / "reproduce_core_report.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert manifest["folds"]["subject_01"]["status"] == "completed"
    assert manifest["runtime"]["versions"]["numpy"]
    assert manifest["plan"]["config"]["sha256"]
    assert report["summary"] and report["exact_paper_reproduction"] is False
    assert calls == {"prepare": 1, "generate": 1}

    monkeypatch.setattr(
        reproduction,
        "prepare_data",
        lambda **kwargs: pytest.fail("completed fold must be resumed"),
    )
    monkeypatch.setattr(
        reproduction,
        "_materialize_synthetic",
        lambda *args, **kwargs: pytest.fail("completed cache must not regenerate"),
    )
    resumed = _args(
        tmp_path,
        "--held-out-subject",
        "1",
        "--sensor",
        "three_channel",
        "--samples-per-class",
        "3",
        "--write-results",
        "--resume",
    )
    assert run_reproduce_core(resumed) == 0
    after = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert after["status"] == "completed"
    assert after["attempts"][-1]["status"] == "skipped_completed"
    assert calls == {"prepare": 1, "generate": 1}

    vae_checkpoint.write_bytes(b"same path, changed checkpoint content")
    monkeypatch.setattr(reproduction, "prepare_data", fake_prepare)
    monkeypatch.setattr(reproduction, "_materialize_synthetic", fake_generate)
    assert run_reproduce_core(resumed) == 0
    changed = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert changed["folds"]["subject_01"]["attempt_count"] == 2
    assert calls == {"prepare": 2, "generate": 2}


def test_failure_is_recorded_for_safe_retry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import lrf_imu.reproducibility as reproduction

    monkeypatch.setattr(
        reproduction,
        "prepare_data",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("fixture interruption")),
    )
    args = _args(
        tmp_path,
        "--held-out-subject",
        "1",
        "--scenario",
        "trtr",
        "--write-results",
    )
    with pytest.raises(RuntimeError, match="fixture interruption"):
        run_reproduce_core(args)
    manifest = json.loads(
        (tmp_path / "output" / "reproduce_core_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["status"] == "failed"
    assert manifest["failure"]["retry_with_resume"] is True
    assert manifest["folds"]["subject_01"]["status"] == "failed"


def test_reference_comparison_reports_signed_fold_differences_without_parity_claim(
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
                                "reference": 0.9,
                                "comparison_basis": "unrounded_json",
                            },
                            "accuracy": {
                                "reference": 0.8,
                                "comparison_basis": "unrounded_json",
                            },
                            "retention_ratio": {
                                "reference": 1.0,
                                "comparison_basis": "unrounded_json",
                            },
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
                    "f1_macro": 0.85,
                    "accuracy": 0.82,
                    "retention_ratio": 1.0,
                }
            ],
        }
    ]
    comparison = _reference_comparison(
        str(path), sensor="six_channel", classifier="rf", folds=current
    )
    assert comparison is not None
    assert comparison["status"] == "compared_without_parity_threshold"
    assert comparison["parity_claim"] is False
    assert comparison["comparison_count"] == 3
    by_metric = {item["metric"]: item for item in comparison["comparisons"]}
    assert by_metric["f1_macro"]["signed_difference"] == pytest.approx(-0.05)
