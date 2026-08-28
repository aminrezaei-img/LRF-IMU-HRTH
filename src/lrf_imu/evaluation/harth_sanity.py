"""Paper 3 HARTH signal sanity evaluation (no classifier or benchmark)."""

from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path
from typing import Any
import numpy as np
import torch
from ..checkpoints import load_flow_checkpoint, load_vae_checkpoint
from ..data.harth_pipeline import prepare_harth_data
from ..generation.flow import sample_reverse_euler
from .signal_metrics import (
    compare_summaries,
    feature_vector,
    finite_counts,
    signal_summary,
    spectral_summary,
)

CLASS_NAMES = (
    "walking_slow",
    "walking_moderate",
    "walking_brisk",
    "running",
    "stair_climbing",
    "cycling_seated",
    "cycling_standing",
    "sitting",
    "standing",
    "lying",
)


def _hash(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1048576), b""):
            h.update(b)
    return h.hexdigest()


def _write_report(
    output: Path,
    title: str,
    payload: dict[str, Any],
    rows: list[dict[str, Any]],
    filename: str,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / filename.replace(".csv", ".json")).write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    report = output / "signal_validation_report.md"
    with report.open("a", encoding="utf-8") as handle:
        handle.write("# LRF-IMU HARTH signal sanity\n\n")
        handle.write(f"- schema: `{payload.get('schema_version')}`\n")
        handle.write(f"- dataset: `{payload.get('dataset')}`\n")
        handle.write(f"- held-out subject: `{payload.get('held_out_subject')}`\n")
        handle.write(
            "- interpretation: descriptive gross-mismatch and collapse diagnostics; not equivalence or benchmark evidence.\n\n"
        )
    if rows:
        keys = list(rows[0])
        with (output / filename).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)


def evaluate_harth_vae(
    *,
    data_root: str,
    composition: str,
    held_out_subject: str,
    config: str,
    vae_checkpoint: str,
    output_dir: str,
    seed: int = 42,
    max_windows: int | None = None,
) -> dict[str, Any]:
    prepared = prepare_harth_data(
        data_root, composition=composition, held_out_subject=held_out_subject, seed=seed
    )
    real = (
        prepared.held_out_test_windows
        if max_windows is None
        else prepared.held_out_test_windows[:max_windows]
    )
    if real.shape[0] == 0:
        raise ValueError("held-out subject has no evaluation windows")
    model, inspection = load_vae_checkpoint(
        vae_checkpoint, channels=3, latent_channels=48, down_levels=2, device="cpu"
    )
    model.eval()
    recon = []
    latent_mu = []
    latent_logvar = []
    losses = []
    with torch.no_grad():
        for start in range(0, len(real), 128):
            x = torch.from_numpy(real[start : start + 128])
            y, mu, lv = model(x, deterministic=True)
            if (
                not torch.isfinite(y).all()
                or not torch.isfinite(mu).all()
                or not torch.isfinite(lv).all()
            ):
                raise ValueError("VAE produced NaN or Inf")
            recon.append(y.numpy())
            latent_mu.append(mu.numpy())
            latent_logvar.append(lv.numpy())
            losses.append(float(torch.mean((y - x) ** 2)))
    reconstructed = np.concatenate(recon)
    mu = np.concatenate(latent_mu)
    lv = np.concatenate(latent_logvar)
    payload = {
        "schema_version": "paper3.harth-vae-sanity.1",
        "checkpoint": inspection.to_mapping(),
        "checkpoint_sha256": _hash(vae_checkpoint),
        "config": str(Path(config).resolve()),
        "dataset": composition,
        "held_out_subject": held_out_subject,
        "seed": seed,
        "input_geometry": [len(real), 3, 160],
        "latent_geometry": [len(real), 48, 40],
        "reconstruction_geometry": list(reconstructed.shape),
        "input_finite": finite_counts(real),
        "reconstruction_finite": finite_counts(reconstructed),
        "mean_mse": float(np.mean(losses)),
        "input_summary": signal_summary(real),
        "reconstruction_summary": signal_summary(reconstructed),
        "spectral_input": spectral_summary(real),
        "spectral_reconstruction": spectral_summary(reconstructed),
        "latent": {
            "mean_abs": float(np.mean(np.abs(mu))),
            "mean_variance": float(np.var(mu)),
            "mean_std": float(np.mean(np.exp(0.5 * lv))),
            "near_zero_variance_fraction": float(
                np.mean(np.var(mu, axis=(0, 2)) <= 1e-12)
            ),
        },
        "comparison": compare_summaries(real, reconstructed),
        "hard_failure": False,
    }
    out = Path(output_dir)
    _write_report(
        out,
        "VAE",
        payload,
        [
            {
                "channel": CLASS_NAMES[i] if i < 3 else str(i),
                **{k: v for k, v in c.items()},
            }
            for i, c in enumerate(payload["comparison"]["real"]["channels"])
        ],
        "vae_reconstruction_metrics.csv",
    )
    (out / "vae_sanity_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    return payload


def evaluate_harth_flow(
    *,
    data_root: str,
    composition: str,
    held_out_subject: str,
    config: str,
    vae_checkpoint: str,
    flow_checkpoint: str,
    output_dir: str,
    seed: int = 42,
    samples_per_class: int = 100,
) -> dict[str, Any]:
    prepared = prepare_harth_data(
        data_root, composition=composition, held_out_subject=held_out_subject, seed=seed
    )
    vae, vi = load_vae_checkpoint(
        vae_checkpoint, channels=3, latent_channels=48, device="cpu"
    )
    flow, fi = load_flow_checkpoint(
        flow_checkpoint, channels=3, latent_channels=48, num_classes=10, device="cpu"
    )
    generated = []
    rows = []
    features = []
    for class_id, name in enumerate(CLASS_NAMES):
        labels = torch.full((samples_per_class,), class_id, dtype=torch.long)
        z = sample_reverse_euler(
            flow, labels, (48, 40), num_steps=10, seed=seed + class_id, device="cpu"
        )
        with torch.no_grad():
            samples = vae.decode(z).numpy().astype(np.float32)
        if not np.isfinite(samples).all():
            raise ValueError("Flow/VAE generated NaN or Inf")
        generated.append(samples)
        features.append(feature_vector(samples))
        real = prepared.held_out_test_windows[prepared.held_out_test_labels == class_id]
        row = {
            "class_id": class_id,
            "class_name": name,
            "generated_windows": samples_per_class,
            "generated_fraction_near_constant": signal_summary(samples)[
                "fraction_near_constant"
            ],
            "generated_rms": float(
                np.mean([x["rms"] for x in signal_summary(samples)["channels"]])
            ),
        }
        if len(real):
            row.update(
                {
                    "real_windows": len(real),
                    "real_rms": float(
                        np.mean([x["rms"] for x in signal_summary(real)["channels"]])
                    ),
                }
            )
        rows.append(row)
    matrix = np.linalg.norm(
        np.asarray(features)[:, None, :] - np.asarray(features)[None, :, :], axis=2
    )
    payload = {
        "schema_version": "paper3.harth-flow-sanity.1",
        "checkpoint_sha256": _hash(flow_checkpoint),
        "vae_checkpoint_sha256": _hash(vae_checkpoint),
        "flow_checkpoint": fi.to_mapping(),
        "vae_checkpoint": vi.to_mapping(),
        "config": str(Path(config).resolve()),
        "dataset": composition,
        "held_out_subject": held_out_subject,
        "seed": seed,
        "num_classes": 10,
        "class_names": list(CLASS_NAMES),
        "samples_per_class": samples_per_class,
        "generated_geometry": [samples_per_class, 3, 160],
        "generated_finite": finite_counts(np.concatenate(generated)),
        "class_metrics": rows,
        "class_feature_distance": matrix.tolist(),
        "all_class_features_identical": bool(np.allclose(matrix, 0)),
        "walking_speed_ordering": {
            "classes": ["walking_slow", "walking_moderate", "walking_brisk"],
            "strict_monotonicity_required": False,
        },
        "hard_failure": False,
    }
    out = Path(output_dir)
    _write_report(out, "Flow", payload, rows, "flow_class_metrics.csv")
    np.savetxt(out / "class_feature_distance.csv", matrix, delimiter=",")
    (out / "flow_sanity_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    return payload


__all__ = ["CLASS_NAMES", "evaluate_harth_vae", "evaluate_harth_flow"]
