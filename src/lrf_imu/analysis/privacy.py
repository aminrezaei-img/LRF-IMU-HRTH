"""Explicitly separated historical privacy threat-model summaries."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

TRUE_HOLDOUT_THREAT_MODEL = "true_training_window_holdout_min_distance_to_synthetic"
POSTHOC_AUDIT_THREAT_MODEL = "posthoc_training_split_best_distance_or_vae_reconstruction"
RECONSTRUCTION_THREAT_MODEL = "latent_optimization_reconstruction_vs_random_baseline"


def _sample_summary(values: Sequence[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("values must be non-empty and finite")
    return float(np.mean(array)), float(np.std(array, ddof=1)) if array.size > 1 else 0.0


def summarize_membership_records(
    records: Sequence[Mapping[str, Any]],
    *,
    threat_model: str,
) -> dict[str, Any]:
    """Summarize one MIA setup without mixing the two historical designs."""

    if not records:
        raise ValueError("at least one membership record is required")
    if threat_model == TRUE_HOLDOUT_THREAT_MODEL:
        key = "roc_auc"
        expected_attack = "blackbox_sample_access_min_distance_to_synthetic"
        for record in records:
            if record.get("attack", expected_attack) != expected_attack:
                raise ValueError("record does not match the true-holdout attack")
        setup = {
            "membership_population": "training_subject_windows",
            "nonmembers": "windows_excluded_from_flow_training_before_fit",
            "attack": expected_attack,
        }
    elif threat_model == POSTHOC_AUDIT_THREAT_MODEL:
        key = "best_attack_auc"
        for record in records:
            if key not in record and "l4_best_auc" not in record:
                raise ValueError("post-hoc records require best_attack_auc or l4_best_auc")
        setup = {
            "membership_population": "posthoc_split_of_observed_training_population",
            "nonmembers": "posthoc_split_not_proven_excluded_from_flow_training",
            "attack": "maximum_of_distance_and_vae_reconstruction_auc",
        }
    else:
        raise ValueError("unknown membership threat model")
    if threat_model == POSTHOC_AUDIT_THREAT_MODEL:
        values = [
            float(record["best_attack_auc"] if "best_attack_auc" in record else record["l4_best_auc"])
            for record in records
        ]
    else:
        values = [float(record[key]) for record in records]
    mean, sd = _sample_summary(values)
    return {
        "schema_version": "m3e.membership-inference.1",
        "threat_model": threat_model,
        "setup": setup,
        "fold_count": len(values),
        "roc_auc_mean": mean,
        "roc_auc_sample_sd": sd,
        "roc_auc_min": float(np.min(values)),
        "roc_auc_max": float(np.max(values)),
        "interpretation": "attack_specific_empirical_result_not_an_anonymization_guarantee",
        "may_be_combined_with_other_mia_setup": False,
    }


def reconstruction_success(
    optimization_l2: Sequence[float],
    *,
    random_baseline_l2: float,
) -> dict[str, Any]:
    """Apply the source's strict 10%-of-random-baseline success criterion."""

    values = np.asarray(optimization_l2, dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all() or random_baseline_l2 <= 0:
        raise ValueError("finite optimization distances and positive random baseline required")
    threshold = float(0.10 * random_baseline_l2)
    successes = int(np.count_nonzero(values < threshold))
    return {
        "threat_model": RECONSTRUCTION_THREAT_MODEL,
        "criterion": "optimization_l2_strictly_less_than_0.10_times_random_baseline_l2",
        "threshold_l2": threshold,
        "attempt_count": int(values.size),
        "successful_count": successes,
        "success_rate_pct": float(100.0 * successes / values.size),
    }


def summarize_reconstruction_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize already executed Level-6 fold records without a privacy guarantee."""

    if not records:
        raise ValueError("at least one reconstruction record is required")
    rates = [float(record["success_rate"]) for record in records]
    successes = sum(int(record["n_successful"]) for record in records)
    attempts = sum(int(record["n_attempts"]) for record in records)
    mean, sd = _sample_summary(rates)
    return {
        "schema_version": "m3e.reconstruction-attack.1",
        "threat_model": RECONSTRUCTION_THREAT_MODEL,
        "criterion": "optimization_l2_strictly_less_than_0.10_times_random_baseline_l2",
        "fold_count": len(records),
        "fold_success_rate_mean_pct": mean,
        "fold_success_rate_sample_sd_pct": sd,
        "total_successful": successes,
        "total_attempts": attempts,
        "pooled_success_rate_pct": float(100.0 * successes / attempts) if attempts else 0.0,
        "interpretation": "attack_specific_empirical_result_not_a_privacy_or_anonymization_guarantee",
    }
