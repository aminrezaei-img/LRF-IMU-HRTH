"""VAE-only class-conditional latent-Gaussian ablation."""

from __future__ import annotations

from typing import Any

import numpy as np


def class_conditional_latent_parameters(
    latent_means: np.ndarray,
    labels: np.ndarray,
    *,
    num_classes: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit source-compatible flattened diagonal Gaussians per class."""

    latents = np.asarray(latent_means)
    target = np.asarray(labels)
    if latents.ndim != 3 or target.shape != (latents.shape[0],):
        raise ValueError("latent_means must be [samples, channels, time] with one label per sample")
    means, standard_deviations = [], []
    for class_id in range(num_classes):
        selected = latents[target == class_id]
        if selected.size == 0:
            raise ValueError(f"no training samples for class {class_id}")
        flattened = selected.reshape(selected.shape[0], -1)
        means.append(flattened.mean(axis=0))
        standard_deviations.append(flattened.std(axis=0, ddof=0) + 1e-6)
    return np.stack(means), np.stack(standard_deviations)


def sample_class_conditional_latents(
    latent_means: np.ndarray,
    labels: np.ndarray,
    *,
    samples_per_class: int = 500,
    seed: int = 42,
    num_classes: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample one shared NumPy RNG sequentially across ascending class IDs."""

    if samples_per_class <= 0:
        raise ValueError("samples_per_class must be positive")
    latents = np.asarray(latent_means)
    means, standard_deviations = class_conditional_latent_parameters(
        latents, labels, num_classes=num_classes
    )
    rng = np.random.default_rng(seed)
    sampled, sampled_labels = [], []
    for class_id in range(num_classes):
        noise = rng.standard_normal((samples_per_class, means.shape[1])).astype(np.float32)
        flat = means[class_id] + standard_deviations[class_id] * noise
        sampled.append(flat.reshape(samples_per_class, latents.shape[1], latents.shape[2]))
        sampled_labels.append(np.full(samples_per_class, class_id, dtype=np.int64))
    return np.concatenate(sampled).astype(np.float32), np.concatenate(sampled_labels)


def generate_vae_only_samples(
    vae: Any,
    training_windows: np.ndarray,
    training_labels: np.ndarray,
    *,
    samples_per_class: int = 500,
    seed: int = 42,
    num_classes: int = 4,
    device: str = "cpu",
    batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode posterior means, fit per-class Gaussians, and decode in batches."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised without training extra
        raise ImportError("VAE-only analysis requires torch; install lrf-imu[training]") from exc
    windows = np.asarray(training_windows, dtype=np.float32)
    labels = np.asarray(training_labels, dtype=np.int64)
    vae.eval()
    encoded = []
    with torch.no_grad():
        for start in range(0, len(windows), batch_size):
            inputs = torch.from_numpy(windows[start : start + batch_size]).to(device)
            mean, _ = vae.encode(inputs)
            encoded.append(mean.detach().cpu().numpy())
    latent_samples, synthetic_labels = sample_class_conditional_latents(
        np.concatenate(encoded), labels,
        samples_per_class=samples_per_class, seed=seed, num_classes=num_classes,
    )
    decoded = []
    with torch.no_grad():
        for start in range(0, len(latent_samples), batch_size):
            inputs = torch.from_numpy(latent_samples[start : start + batch_size]).to(device)
            decoded.append(vae.decode(inputs).detach().cpu().numpy())
    return np.concatenate(decoded).astype(np.float32), synthetic_labels


def vae_only_random_forest_metrics(
    training_windows: np.ndarray,
    training_labels: np.ndarray,
    test_windows: np.ndarray,
    test_labels: np.ndarray,
    synthetic_windows: np.ndarray,
    synthetic_labels: np.ndarray,
    *,
    seed: int = 42,
) -> dict[str, Any]:
    """Run source-compatible RF TRTR and VAE-only TSTR metrics."""

    from ..evaluation.core import evaluate_scenarios

    result = evaluate_scenarios(
        training_windows,
        training_labels,
        test_windows,
        test_labels,
        synthetic_windows,
        synthetic_labels,
        classifier="rf",
        scenarios=("trtr", "tstr"),
        channels=int(np.asarray(training_windows).shape[1]),
        seed=seed,
        synthetic_per_class=int(np.min(np.bincount(np.asarray(synthetic_labels), minlength=4))),
    )
    result.update({
        "schema_version": "m3e.vae-only-ablation.1",
        "ablation": "vae_only",
        "generation_method": "per_class_diagonal_gaussian_latent",
        "flow_steps": 0,
        "rf": {"n_estimators": 100, "random_state": seed, "n_jobs": 1},
    })
    return result
