"""In-memory source-compatible evaluation across the four paper scenarios."""

from __future__ import annotations

from typing import Any, Iterable, Optional, Sequence

import numpy as np

from .classifiers import (
    CNNTrainingSpec,
    RandomForestSpec,
    predict_cnn,
    predict_random_forest,
    seed_cnn_run,
)
from .metrics import PAPER_LABELS, classification_metrics, retention_ratio
from .scenarios import SCENARIO_ORDER, build_scenario_populations, ensure_nct


def evaluate_scenarios(
    real_train_windows: np.ndarray,
    real_train_labels: np.ndarray,
    real_test_windows: np.ndarray,
    real_test_labels: np.ndarray,
    synthetic_windows: Optional[np.ndarray],
    synthetic_labels: Optional[np.ndarray],
    *,
    classifier: str = "rf",
    scenarios: Iterable[str] = SCENARIO_ORDER,
    channels: Optional[int] = None,
    seed: int = 42,
    scarce_per_class: int = 2,
    synthetic_per_class: int = 500,
    labels: Sequence[int] = PAPER_LABELS,
    device: str = "cpu",
    cnn_spec: Optional[CNNTrainingSpec] = None,
) -> dict[str, Any]:
    """Evaluate requested populations; run TRTR internally when retention needs it."""

    supplied = tuple(str(value).lower() for value in scenarios)
    unknown = sorted(set(supplied).difference(SCENARIO_ORDER))
    if unknown:
        raise ValueError("unknown evaluation scenarios: {}".format(unknown))
    requested = tuple(name for name in SCENARIO_ORDER if name in supplied)
    if not requested:
        raise ValueError("at least one evaluation scenario is required")
    execution_scenarios = requested if "trtr" in requested else ("trtr",) + requested

    populations = build_scenario_populations(
        real_train_windows,
        real_train_labels,
        synthetic_windows,
        synthetic_labels,
        seed=seed,
        scarce_per_class=scarce_per_class,
        synthetic_per_class=synthetic_per_class,
        class_labels=labels,
        channels=channels,
    )
    missing = [name for name in execution_scenarios if name not in populations]
    if missing:
        raise ValueError("synthetic inputs are required for scenarios: {}".format(missing))
    test_x = ensure_nct(np.asarray(real_test_windows), channels)
    test_y = np.asarray(real_test_labels, dtype=np.int64).reshape(-1)
    if test_x.shape[0] != test_y.size:
        raise ValueError("test windows and labels have different lengths")

    classifier_name = classifier.strip().lower()
    if classifier_name in {"random_forest", "randomforest"}:
        classifier_name = "rf"
    if classifier_name in {"1dcnn", "deep"}:
        classifier_name = "cnn"
    if classifier_name not in {"rf", "cnn"}:
        raise ValueError("classifier must be rf or cnn")
    selected_spec = cnn_spec or CNNTrainingSpec(seed=seed)
    if classifier_name == "cnn":
        seed_cnn_run(selected_spec.seed)

    executed_records: list[dict[str, Any]] = []
    for scenario in execution_scenarios:
        population = populations[scenario]
        if classifier_name == "rf":
            predictions = predict_random_forest(
                population.windows,
                population.labels,
                test_x,
                spec=RandomForestSpec(random_state=seed),
            )
        else:
            predictions = predict_cnn(
                population.windows,
                population.labels,
                test_x,
                channels=int(channels or test_x.shape[1]),
                sequence_length=int(test_x.shape[2]),
                num_classes=len(labels),
                device=device,
                spec=selected_spec,
            )
        measured = classification_metrics(test_y, predictions, labels=labels)
        executed_records.append(
            {"scenario": scenario, **population.metadata(), **measured.as_dict()}
        )

    trtr = next(record for record in executed_records if record["scenario"] == "trtr")
    for record in executed_records:
        record["retention_ratio"] = (
            1.0
            if record["scenario"] == "trtr"
            else retention_ratio(record["f1_macro"], trtr["f1_macro"])
        )
    records = [record for record in executed_records if record["scenario"] in requested]
    return {
        "schema_version": "m3d.fold-evaluation.1",
        "classifier": classifier_name,
        "seed": int(seed),
        "labels": [int(label) for label in labels],
        "requested_scenarios": list(requested),
        "executed_scenarios": list(execution_scenarios),
        "internal_prerequisite_scenarios": [
            name for name in execution_scenarios if name not in requested
        ],
        "classifier_training_population": (
            "public VAE-safe train partition only; VAE subject validation excluded"
        ),
        "cnn_internal_validation": (
            "window-level 0.20 stratified split within each scenario population"
        ),
        "records": records,
        "test_count": int(test_y.size),
    }


__all__ = ["evaluate_scenarios"]