"""Portable evaluation APIs; optional sklearn/Torch imports remain lazy."""

from .cache import (
    SyntheticCacheIdentity,
    cache_paths,
    discover_cache_manifest,
    load_synthetic_cache,
    load_validated_synthetic_cache,
    sha256_file,
    validate_cache_manifest,
)
from .classifiers import (
    CNNTrainingSpec, RandomForestSpec, build_har_cnn, cnn_state_geometry,
    predict_cnn, predict_random_forest, seed_cnn_run, stratified_train_validation_indices,
)
from .core import evaluate_scenarios
from .metrics import (
    ClassificationMetrics, aggregate_confusions, classification_metrics,
    mean_sample_sd, retention_ratio, summarize_fold_records,
)
from .scenarios import SCENARIO_ORDER, ScenarioPopulation, build_scenario_populations

__all__ = [
    "CNNTrainingSpec", "ClassificationMetrics", "RandomForestSpec",
    "SCENARIO_ORDER", "ScenarioPopulation", "SyntheticCacheIdentity",
    "aggregate_confusions", "build_har_cnn", "build_scenario_populations",
    "cache_paths", "classification_metrics", "cnn_state_geometry",
    "discover_cache_manifest",
    "evaluate_scenarios", "load_synthetic_cache", "load_validated_synthetic_cache",
    "mean_sample_sd",
    "predict_cnn", "predict_random_forest", "retention_ratio", "seed_cnn_run",
    "sha256_file",
    "stratified_train_validation_indices", "summarize_fold_records",
    "validate_cache_manifest",
]
