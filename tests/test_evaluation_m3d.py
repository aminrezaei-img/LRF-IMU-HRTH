from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from lrf_imu.evaluation import (
    SyntheticCacheIdentity,
    aggregate_confusions,
    build_scenario_populations,
    cache_paths,
    classification_metrics,
    cnn_state_geometry,
    evaluate_scenarios,
    mean_sample_sd,
    retention_ratio,
    stratified_train_validation_indices,
    summarize_fold_records,
)


def _separable_windows(samples_per_class: int, *, channels: int = 3) -> tuple[np.ndarray, np.ndarray]:
    labels = np.repeat(np.arange(4, dtype=np.int64), samples_per_class)
    windows = np.empty((labels.size, channels, 8), dtype=np.float32)
    for index, label in enumerate(labels):
        windows[index].fill(float(label * 10 + index % samples_per_class))
    return windows, labels


def test_metric_contract_labels_zero_division_and_confusion() -> None:
    truth = np.array([0, 0, 1, 1, 2, 3])
    predicted = np.array([0, 1, 1, 1, 0, 3])
    measured = classification_metrics(truth, predicted)
    assert measured.accuracy == pytest.approx(4 / 6)
    assert measured.confusion == ((1, 1, 0, 0), (0, 2, 0, 0), (1, 0, 0, 0), (0, 0, 0, 1))
    assert len(measured.per_class_f1) == 4
    assert measured.per_class_f1[2] == 0.0


def test_retention_is_foldwise_and_sd_is_sample_sd() -> None:
    assert retention_ratio(0.8, 1.0) == pytest.approx(0.8 / 1.00000001)
    assert mean_sample_sd([1.0, 2.0, 3.0]) == pytest.approx((2.0, 1.0))
    records = [
        {"scenario": "trtr", "accuracy": 1.0, "f1_macro": 0.8, "retention_ratio": 1.0},
        {"scenario": "trtr", "accuracy": 1.0, "f1_macro": 1.0, "retention_ratio": 1.0},
        {"scenario": "tstr", "accuracy": 0.7, "f1_macro": 0.4, "retention_ratio": 0.5},
        {"scenario": "tstr", "accuracy": 0.9, "f1_macro": 0.9, "retention_ratio": 0.9},
    ]
    summary = {row["scenario"]: row for row in summarize_fold_records(records)}
    assert summary["tstr"]["retention_mean"] == pytest.approx(0.7)
    assert summary["tstr"]["retention_mean"] != pytest.approx(0.65 / 0.9)
    assert summary["tstr"]["sd_ddof"] == 1


def test_confusion_nanmean_and_nonzero_fold_count_semantics() -> None:
    first = np.array([[1.0, 0.0], [np.nan, np.nan]])
    second = np.array([[0.5, 0.5], [0.0, 1.0]])
    aggregate = aggregate_confusions([first, second])
    assert np.asarray(aggregate["mean"])[0].tolist() == pytest.approx([0.75, 0.25])
    assert aggregate["nonzero_fold_count"] == [[2, 1], [0, 1]]
    assert aggregate["fold_count"] == 2


def test_scenario_population_rng_order_and_no_validation_leak() -> None:
    real_x, real_y = _separable_windows(5)
    synthetic_x, synthetic_y = _separable_windows(6)
    populations = build_scenario_populations(
        real_x, real_y, synthetic_x, synthetic_y,
        seed=42, scarce_per_class=2, synthetic_per_class=3, channels=3,
    )
    assert populations["trtr"].labels.size == 20
    assert populations["scarce"].labels.size == 8
    assert populations["tstr"].labels.size == 12
    assert populations["tstr_scarce"].labels.size == 20
    rng = np.random.default_rng(42)
    for label in range(4):
        rng.choice(np.where(synthetic_y == label)[0], size=3, replace=False)
    expected_scarce = np.concatenate([
        rng.choice(np.where(real_y == label)[0], 2, replace=False) for label in range(4)
    ])
    assert populations["scarce"].real_indices == tuple(expected_scarce)


def test_random_forest_predictions_and_all_four_metrics_are_deterministic() -> None:
    train_x, train_y = _separable_windows(8)
    test_x, test_y = _separable_windows(2)
    synthetic_x, synthetic_y = _separable_windows(8)
    first = evaluate_scenarios(
        train_x, train_y, test_x, test_y, synthetic_x, synthetic_y,
        classifier="rf", channels=3, synthetic_per_class=8,
    )
    second = evaluate_scenarios(
        train_x, train_y, test_x, test_y, synthetic_x, synthetic_y,
        classifier="rf", channels=3, synthetic_per_class=8,
    )
    assert first == second
    assert [row["scenario"] for row in first["records"]] == [
        "trtr", "scarce", "tstr", "tstr_scarce"
    ]
    assert all(row["f1_macro"] == 1.0 for row in first["records"])
    assert first["classifier_training_population"].startswith("public VAE-safe")


def test_cnn_geometry_and_source_split_contract() -> None:
    geometry = cnn_state_geometry(6, 160)
    assert geometry["conv1.weight"] == (32, 6, 5)
    assert geometry["conv2.weight"] == (64, 32, 5)
    assert geometry["conv3.weight"] == (128, 64, 5)
    assert geometry["fc1.weight"] == (256, 128 * 80)
    assert geometry["fc3.weight"] == (4, 128)
    labels = np.repeat(np.arange(4), 10)
    train, validation = stratified_train_validation_indices(labels, seed=42)
    assert len(train) == 32
    assert validation is not None and len(validation) == 8
    assert set(labels[validation]) == {0, 1, 2, 3}


def test_cache_identity_contains_every_required_discriminator(tmp_path: Path) -> None:
    identity = SyntheticCacheIdentity(
        sensor_configuration="six_channel", held_out_subject=1,
        vae_checkpoint_sha256="a" * 64, flow_checkpoint_sha256="b" * 64,
        config_identity="paper-six", seed=42, steps=10, samples_per_class=500,
        implementation_version="8aeeda2",
    )
    paths = cache_paths(tmp_path, identity)
    assert len(identity.key) == 64
    assert identity.key in paths["array"].name
    assert "subject_01" in paths["manifest"].parts


def test_reference_map_is_machine_readable_and_covers_expected_families() -> None:
    root = Path(__file__).resolve().parents[1]
    payload = json.loads((root / "contracts" / "evaluation_reference_map.json").read_text(encoding="utf-8"))
    assert payload["canonical_folds"] == [1, 2, 3, 5, 8, 9, 10, 11, 12, 13, 14, 16]
    assert any(item.get("files_read") == 108 for item in payload["artifact_families"])
    assert payload["source_semantics"]["rf"] == {
        "n_estimators": 100, "random_state": 42, "n_jobs": 1,
        "other_parameters": "sklearn defaults",
    }


def test_cli_dry_run_is_no_write_and_foreign_cwd_safe(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    output = tmp_path / "must_not_exist"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(root / "src")
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    command = [
        sys.executable, "-B", "-m", "lrf_imu", "evaluate-loso",
        "--data-root", str(tmp_path / "not-read"), "--output-root", str(output),
        "--write-results", "--dry-run",
    ]
    completed = subprocess.run(
        command, cwd=foreign, env=environment, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert len(payload["folds"]) == 12
    assert payload["dry_run"] is True
    assert not output.exists()


def test_import_does_not_load_sklearn_or_torch() -> None:
    command = [
        sys.executable, "-B", "-c",
        "import sys, lrf_imu.evaluation; print(int('sklearn' in sys.modules), int('torch' in sys.modules))",
    ]
    completed = subprocess.run(
        command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "0 0"


def test_cnn_run_seeds_once_and_consumes_model_rng_sequentially(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch
    import lrf_imu.evaluation.core as evaluation_core

    train_x, train_y = _separable_windows(3)
    test_x, test_y = _separable_windows(1)
    synthetic_x, synthetic_y = _separable_windows(3)
    draws: list[float] = []
    seed_calls: list[int] = []
    original_seed = evaluation_core.seed_cnn_run

    def tracked_seed(seed: int) -> None:
        seed_calls.append(seed)
        original_seed(seed)

    def fake_predict(*args, **kwargs):
        draws.append(float(torch.rand(())))
        return np.asarray(test_y)

    monkeypatch.setattr(evaluation_core, "seed_cnn_run", tracked_seed)
    monkeypatch.setattr(evaluation_core, "predict_cnn", fake_predict)
    first = evaluate_scenarios(
        train_x,
        train_y,
        test_x,
        test_y,
        synthetic_x,
        synthetic_y,
        classifier="cnn",
        channels=3,
        synthetic_per_class=3,
    )
    first_draws = tuple(draws)
    draws.clear()
    second = evaluate_scenarios(
        train_x,
        train_y,
        test_x,
        test_y,
        synthetic_x,
        synthetic_y,
        classifier="cnn",
        channels=3,
        synthetic_per_class=3,
    )
    assert seed_calls == [42, 42]
    assert len(set(first_draws)) == 4
    assert tuple(draws) == first_draws
    assert first == second
    assert torch.backends.cudnn.benchmark is False
    assert torch.backends.cudnn.deterministic is True


@pytest.mark.parametrize("scenario", ["scarce", "tstr", "tstr_scarce"])
def test_scenario_only_runs_trtr_internally_and_returns_requested_scope(scenario: str) -> None:
    train_x, train_y = _separable_windows(5)
    test_x, test_y = _separable_windows(2)
    synthetic_x, synthetic_y = _separable_windows(5)
    needs_synthetic = scenario in {"tstr", "tstr_scarce"}
    result = evaluate_scenarios(
        train_x,
        train_y,
        test_x,
        test_y,
        synthetic_x if needs_synthetic else None,
        synthetic_y if needs_synthetic else None,
        classifier="rf",
        scenarios=(scenario,),
        channels=3,
        synthetic_per_class=5,
    )
    assert result["requested_scenarios"] == [scenario]
    assert result["executed_scenarios"] == ["trtr", scenario]
    assert result["internal_prerequisite_scenarios"] == ["trtr"]
    assert [record["scenario"] for record in result["records"]] == [scenario]
    assert result["records"][0]["retention_ratio"] == pytest.approx(1.0 / 1.00000001)


def test_resume_preserves_fold_schema_and_mixed_or_all_resumed_aggregate(tmp_path: Path) -> None:
    from lrf_imu.evaluation.cli import _loso_payload, _resume_fold

    record = {
        "scenario": "trtr",
        "accuracy": 1.0,
        "f1_macro": 1.0,
        "retention_ratio": 1.0,
    }
    fold = {
        "schema_version": "m3d.fold-evaluation.1",
        "records": [record],
        "held_out_subject": 1,
        "execution": {"status": "fresh", "resumed": False, "target": None},
    }
    target = tmp_path / "subject_01_rf_evaluation.json"
    target.write_text(json.dumps(fold), encoding="utf-8")
    resumed = _resume_fold(target)
    assert resumed["records"] == fold["records"]
    assert resumed["execution"]["status"] == "resumed"
    fresh = {**fold, "held_out_subject": 2}
    mixed = _loso_payload([fresh, resumed], dry_run=False)
    all_resumed = _loso_payload([resumed, {**resumed, "held_out_subject": 2}], dry_run=False)
    assert mixed["summary"][0]["fold_count"] == 2
    assert all_resumed["summary"][0]["fold_count"] == 2


def test_write_results_requires_output_before_data_loading(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import argparse
    import lrf_imu.evaluation.cli as evaluation_cli

    monkeypatch.setattr(
        evaluation_cli,
        "prepare_data",
        lambda **kwargs: pytest.fail("data loading must not start"),
    )
    args = argparse.Namespace(
        scenario=["trtr"],
        output_root=None,
        write_results=True,
        classifier="rf",
        data_root=str(tmp_path / "unread"),
        checkpoint_root=None,
        sensor="six_channel",
        dry_run=False,
        resume=False,
        overwrite=False,
        config=None,
        seed=42,
        scarce_per_class=2,
        samples_per_class=500,
        device="cpu",
        synthetic_cache=None,
    )
    with pytest.raises(ValueError, match="--output-root"):
        evaluation_cli.evaluate_one(args, 1)


def test_cache_manifest_rejects_wrong_fold_sensor_or_checksum(tmp_path: Path) -> None:
    from lrf_imu.evaluation.cache import (
        load_validated_synthetic_cache,
        write_cache_manifest,
    )

    array = tmp_path / "steps10_synthetic.npz"
    windows, labels = _separable_windows(2, channels=6)
    np.savez_compressed(array, X_syn=windows, y_syn=labels)
    identity = SyntheticCacheIdentity(
        sensor_configuration="six_channel",
        held_out_subject=1,
        vae_checkpoint_sha256="a" * 64,
        flow_checkpoint_sha256="b" * 64,
        config_identity="six_channel_160_40_release_default",
        seed=42,
        steps=10,
        samples_per_class=2,
        implementation_version="fixture",
    )
    write_cache_manifest(tmp_path / "manifest.json", identity, array_path=array)
    loaded_x, loaded_y, manifest = load_validated_synthetic_cache(
        array,
        sensor_configuration="six_channel",
        held_out_subject=1,
        seed=42,
        steps=10,
        samples_per_class=2,
    )
    assert loaded_x.shape == windows.shape and loaded_y.tolist() == labels.tolist()
    assert manifest["identity"]["held_out_subject"] == 1
    with pytest.raises(ValueError, match="requested fold"):
        load_validated_synthetic_cache(
            array,
            sensor_configuration="six_channel",
            held_out_subject=2,
            seed=42,
            steps=10,
            samples_per_class=2,
        )
    with pytest.raises(ValueError, match="requested fold"):
        load_validated_synthetic_cache(
            array,
            sensor_configuration="three_channel",
            held_out_subject=1,
            seed=42,
            steps=10,
            samples_per_class=2,
        )
    with array.open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ValueError, match="checksum"):
        load_validated_synthetic_cache(
            array,
            sensor_configuration="six_channel",
            held_out_subject=1,
            seed=42,
            steps=10,
            samples_per_class=2,
        )

def test_parity_report_has_fold_level_per_class_and_confusion_evidence() -> None:
    report_path = Path(__file__).parents[1] / "contracts" / "evaluation_parity_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8-sig"))

    for gate_name in ("gate_c_6ch_rf", "gate_d_3ch_rf", "gate_e_6ch_cnn"):
        gate = report[gate_name]
        assert len(gate["folds"]) == 12
        for fold in gate["folds"]:
            assert set(fold["scenarios"]) == {"trtr", "scarce", "tstr", "tstr_scarce"}
            for scenario in fold["scenarios"].values():
                per_class = scenario["per_class_f1"]["values"]
                assert len(per_class) == 4
                assert [value["label"] for value in per_class] == [0, 1, 2, 3]
                assert all({"reference", "regenerated"} <= set(value) for value in per_class)
            confusion = fold["scenarios"]["tstr_scarce"]["confusion_normalized"]
            assert len(confusion["reference"]) == len(confusion["regenerated"]) == 4

        aggregate = gate["tstr_scarce_confusion_aggregate"]
        assert aggregate["verification"]["nanmean_recomputed_independently"] is True
        assert (
            aggregate["verification"][
                "cm_count_nonzero_fold_semantics_recomputed_independently"
            ]
            is True
        )
        assert "cm_count=folds" in aggregate["reference"]["semantics"]
        assert "cm_count=folds" in aggregate["regenerated"]["semantics"]

    invalidated = report["gate_e_6ch_cnn"]["prior_evidence_invalidated"]
    assert invalidated["excluded"] is True
    assert "reset" in invalidated["reason"]
