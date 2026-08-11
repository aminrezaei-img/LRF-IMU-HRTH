from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import pytest

from lrf_imu.analysis import (
    POSTHOC_AUDIT_THREAT_MODEL,
    TRUE_HOLDOUT_THREAT_MODEL,
    WINDOW_GRID,
    acceleration_magnitude,
    acceleration_magnitude_summary,
    aggregate_fold_psd,
    class_conditional_latent_parameters,
    reconstruction_success,
    sample_class_conditional_latents,
    spectral_statistics,
    summarize_membership_records,
    summarize_sensitivity_grid,
    vae_only_random_forest_metrics,
)


def test_physical_threshold_uses_first_three_channels_and_strict_comparison() -> None:
    threshold = 10.0 * 9.80665
    windows = np.zeros((1, 6, 3), dtype=np.float64)
    windows[0, 0] = [threshold, np.nextafter(threshold, np.inf), 0.0]
    windows[0, 3:] = 1e9
    magnitude = acceleration_magnitude(windows)
    assert magnitude.tolist()[0][:2] == pytest.approx([threshold, np.nextafter(threshold, np.inf)])
    result = acceleration_magnitude_summary(windows)
    assert result["count_above_threshold"] == 1
    assert result["point_count"] == 3
    assert result["comparison"] == "strictly_greater_than"


def test_psd_log_correlation_band_ratios_and_attenuation() -> None:
    frequencies = np.array([0.0, 1.0, 2.0, 10.0, 15.0, 25.0])
    real = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
    synthetic = real * 0.5
    result = spectral_statistics(frequencies, real, synthetic, channel_names=["ACCX"])
    channel = result["channels"][0]
    assert channel["log_psd_correlation"] == pytest.approx(1.0)
    assert channel["band_power_ratio_0_25hz"] == pytest.approx(0.5)
    assert channel["band_power_ratio_10_25hz"] == pytest.approx(0.5)
    assert result["high_frequency_attenuation_observed"] is True


def test_fold_psd_uses_population_sd_like_historical_summarizer() -> None:
    mean, sd = aggregate_fold_psd([np.array([[1.0]]), np.array([[3.0]])])
    assert mean.item() == 2.0
    assert sd.item() == 1.0


def test_nine_setting_sensitivity_uses_sample_sd() -> None:
    records = {
        f"WIN{window}_HOP{hop}": [
            {"trtr_macro_f1": 1.0, "tstr_macro_f1": 0.5, "tstr_per_class_f1": {"walking": 0.4}},
            {"trtr_macro_f1": 1.0, "tstr_macro_f1": 0.7, "tstr_per_class_f1": {"walking": 0.8}},
        ]
        for window, hop in WINDOW_GRID
    }
    summary = summarize_sensitivity_grid(records)
    assert summary["setting_count"] == 9
    baseline = summary["settings"]["WIN160_HOP40"]
    assert baseline["tstr_macro_f1_mean"] == pytest.approx(0.6)
    assert baseline["tstr_macro_f1_sd"] == pytest.approx(np.std([0.5, 0.7], ddof=1))
    assert baseline["sd_ddof"] == 1


def test_latent_gaussian_uses_population_std_and_one_rng_across_classes() -> None:
    labels = np.repeat(np.arange(4), 2)
    latents = np.arange(8 * 2, dtype=np.float32).reshape(8, 2, 1)
    means, stds = class_conditional_latent_parameters(latents, labels)
    expected = latents[:2].reshape(2, -1).std(axis=0, ddof=0) + 1e-6
    assert stds[0] == pytest.approx(expected)
    first, first_labels = sample_class_conditional_latents(latents, labels, samples_per_class=3, seed=42)
    second, second_labels = sample_class_conditional_latents(latents, labels, samples_per_class=3, seed=42)
    assert np.array_equal(first, second)
    assert np.array_equal(first_labels, second_labels)
    rng = np.random.default_rng(42)
    expected_first = means[0] + stds[0] * rng.standard_normal((3, means.shape[1])).astype(np.float32)[0]
    assert first[0].reshape(-1) == pytest.approx(expected_first)


def test_vae_only_metric_path_is_rf_trtr_and_tstr() -> None:
    labels = np.repeat(np.arange(4), 5)
    windows = np.repeat(labels[:, None, None].astype(np.float32), 3 * 8, axis=1).reshape(-1, 3, 8)
    result = vae_only_random_forest_metrics(windows, labels, windows, labels, windows, labels)
    assert [record["scenario"] for record in result["records"]] == ["trtr", "tstr"]
    assert all(record["f1_macro"] == 1.0 for record in result["records"])
    assert result["generation_method"] == "per_class_diagonal_gaussian_latent"
    assert result["flow_steps"] == 0


def test_privacy_threat_models_are_separate_and_not_averaged() -> None:
    true_records = [
        {"attack": "blackbox_sample_access_min_distance_to_synthetic", "roc_auc": 0.48},
        {"attack": "blackbox_sample_access_min_distance_to_synthetic", "roc_auc": 0.52},
    ]
    posthoc_records = [{"best_attack_auc": 0.51}, {"best_attack_auc": 0.53}]
    true = summarize_membership_records(true_records, threat_model=TRUE_HOLDOUT_THREAT_MODEL)
    posthoc = summarize_membership_records(posthoc_records, threat_model=POSTHOC_AUDIT_THREAT_MODEL)
    assert true["roc_auc_mean"] == pytest.approx(0.5)
    assert posthoc["roc_auc_mean"] == pytest.approx(0.52)
    assert true["threat_model"] != posthoc["threat_model"]
    assert true["setup"]["nonmembers"] == "windows_excluded_from_flow_training_before_fit"
    assert posthoc["may_be_combined_with_other_mia_setup"] is False
    with pytest.raises(ValueError, match="best_attack_auc"):
        summarize_membership_records(true_records, threat_model=POSTHOC_AUDIT_THREAT_MODEL)


def test_reconstruction_criterion_is_strict_ten_percent_of_random_baseline() -> None:
    result = reconstruction_success([0.099, 0.1, 0.101], random_baseline_l2=1.0)
    assert result["threshold_l2"] == pytest.approx(0.1)
    assert result["successful_count"] == 1
    assert result["success_rate_pct"] == pytest.approx(100 / 3)


def test_analysis_import_is_lazy_for_optional_dependencies() -> None:
    command = [
        sys.executable,
        "-B",
        "-c",
        "import sys,lrf_imu.analysis; print(int('torch' in sys.modules),int('scipy' in sys.modules),int('sklearn' in sys.modules))",
    ]
    completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "0 0 0"


def test_vae_only_dry_run_is_foreign_cwd_safe_and_writes_nothing() -> None:
    root = Path(__file__).resolve().parents[1]
    foreign = Path(tempfile.gettempdir()).resolve()
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(root / "src")
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    command = [
        sys.executable, "-B", "-m", "lrf_imu", "evaluate-vae-only",
        "--data-root", str(foreign / "m3e-not-read"),
        "--vae-checkpoint", str(foreign / "m3e-not-read.pt"),
        "--held-out-subject", "1", "--dry-run",
    ]
    completed = subprocess.run(
        command, cwd=foreign, env=environment, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["dry_run"] is True
    assert payload["execution"]["output_written"] is False
