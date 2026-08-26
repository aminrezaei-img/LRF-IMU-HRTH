"""Trainable HARTH 3-channel VAE/10-class Flow orchestration."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from ..checkpoints import load_vae_checkpoint, save_vae_checkpoint
from ..config import load_config
from ..data.harth_pipeline import prepare_harth_data
from ..models.flow import LatentDiffusionUNet1D
from ..models.vae import LatentVAE1D
from .flow import train_flow
from .vae import VAEProfile, profile_from_config, resolve_device, train_vae


class LimitedLoader:
    def __init__(self, loader: DataLoader, limit: Optional[int]):
        self.loader, self.limit = loader, limit
    def __iter__(self):
        for index, batch in enumerate(self.loader):
            if self.limit is not None and index >= self.limit:
                break
            yield batch
    def __len__(self):
        return min(len(self.loader), self.limit) if self.limit is not None else len(self.loader)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool, limit: Optional[int]) -> LimitedLoader:
    return LimitedLoader(DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y)), batch_size=batch_size, shuffle=shuffle), limit)


def _profile(config: Any, epochs: Optional[int]) -> VAEProfile:
    profile = profile_from_config(config.vae)
    return replace(profile, max_epochs=int(epochs)) if epochs is not None else profile


def train_harth_vae(*, data_root: str, config_path: str, composition: str, held_out_subject: str, output_dir: str, seed: int = 42, epochs: Optional[int] = None, max_train_batches: Optional[int] = None, max_val_batches: Optional[int] = None) -> dict:
    config_file = Path(config_path).expanduser()
    if not config_file.is_absolute():
        config_file = (Path.cwd() / config_file).resolve()
    config = load_config(config_file, base_dir=config_file.parent)
    prepared = prepare_harth_data(data_root, composition=composition, held_out_subject=held_out_subject, seed=seed)
    output = Path(output_dir).expanduser().resolve(); output.mkdir(parents=True, exist_ok=True)
    model = LatentVAE1D(in_ch=3, z_ch=48, down_levels=2)
    profile = _profile(config, epochs)
    result = train_vae(model, _loader(prepared.train_windows, prepared.train_labels, profile.batch_size, True, max_train_batches), _loader(prepared.validation_windows, prepared.validation_labels, profile.batch_size, False, max_val_batches), profile=profile, checkpoint_path=output / "vae_s3_z48.pt", device=config.runtime.device, seed=seed, drop_last=False)
    metadata = {"schema_version": "harth.vae-run.1", "dataset": composition, "held_out_subject": held_out_subject, "channels": 3, "window_length": 160, "latent_channels": 48, "latent_time": 40, "seed": seed, "normalization": prepared.normalizer.to_metadata(), "class_names": list(prepared.metadata["target_class_names"]), "source_label_maps": "src.lrf_imu.data.harth.SOURCE_LABEL_MAPS", "preparation": prepared.summary, "training": result.to_mapping(), "checkpoint": "vae_s3_z48.pt"}
    (output / "vae_run_meta.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


def train_harth_flow(*, data_root: str, config_path: str, composition: str, held_out_subject: str, vae_checkpoint: str, output_dir: str, seed: int = 42, epochs: Optional[int] = None, max_train_batches: Optional[int] = None, max_val_batches: Optional[int] = None) -> dict:
    config_file = Path(config_path).expanduser()
    if not config_file.is_absolute():
        config_file = (Path.cwd() / config_file).resolve()
    config = load_config(config_file, base_dir=config_file.parent)
    prepared = prepare_harth_data(data_root, composition=composition, held_out_subject=held_out_subject, seed=seed)
    vae, inspection = load_vae_checkpoint(vae_checkpoint, channels=3, latent_channels=48, down_levels=2, device="cpu")
    vae.eval()
    def encode(x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return vae.encode(torch.from_numpy(x))[0].numpy().astype(np.float32)
    # train_flow performs the frozen-VAE encoding internally; loaders therefore
    # carry normalized [B,3,160] windows rather than precomputed latents.
    train_z, val_z = prepared.train_windows, prepared.validation_windows
    model = LatentDiffusionUNet1D(in_ch=48, model_ch=config.flow.base_width, num_classes=10)
    flow_epochs = int(epochs if epochs is not None else config.flow.training.epochs)
    output = Path(output_dir).expanduser().resolve(); output.mkdir(parents=True, exist_ok=True)
    result = train_flow(model, vae, _loader(train_z, prepared.train_labels, config.flow.training.batch_size, True, max_train_batches), _loader(val_z, prepared.validation_labels, config.flow.training.batch_size, False, max_val_batches), epochs=flow_epochs, learning_rate=config.flow.training.learning_rate, grad_clip=config.flow.training.grad_clip, weight_decay=config.flow.training.weight_decay, device=resolve_device(config.runtime.device), seed=seed, checkpoint_path=output / "flow_unet_best.pt", config={"channels": 3, "latent_dim": 48, "model_ch": config.flow.base_width, "num_classes": 10, "class_names": list(prepared.metadata["target_class_names"]), "dataset": composition, "held_out_subject": held_out_subject, "vae_checkpoint_sha256": _sha256(Path(vae_checkpoint))})
    metadata = {"schema_version": "harth.flow-run.1", "dataset": composition, "held_out_subject": held_out_subject, "channels": 3, "latent_geometry": [48, 40], "num_classes": 10, "class_names": list(prepared.metadata["target_class_names"]), "vae_checkpoint_sha256": _sha256(Path(vae_checkpoint)), "seed": seed, "training": result.to_mapping(), "checkpoint": "flow_unet_best.pt"}
    (output / "flow_run_meta.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


def class_id(value: int | str) -> int:
    if isinstance(value, str):
        names = {name: i for i, name in enumerate(("walking_slow", "walking_moderate", "walking_brisk", "running", "stair_climbing", "cycling_seated", "cycling_standing", "sitting", "standing", "lying"))}
        if value not in names: raise ValueError("unknown HARTH activity name: {}".format(value))
        return names[value]
    result = int(value)
    if result < 0 or result >= 10: raise ValueError("HARTH class ID must be in [0, 9]")
    return result


def generate_harth_window(flow_checkpoint: str, vae_checkpoint: str, activity: int | str, *, seed: int = 42, device: str = "cpu") -> tuple[np.ndarray, dict]:
    from ..generation.flow import sample_reverse_euler
    vae, vi = load_vae_checkpoint(vae_checkpoint, channels=3, device=device)
    from ..checkpoints import load_flow_checkpoint
    flow, fi = load_flow_checkpoint(flow_checkpoint, channels=3, latent_channels=48, num_classes=10, device=device)
    label = class_id(activity); labels = torch.tensor([label], device=device)
    latent = sample_reverse_euler(flow, labels, (48, 40), num_steps=10, seed=seed, device=device)
    with torch.no_grad(): sample = vae.decode(latent).detach().cpu().numpy().astype(np.float32)
    return sample, {"class_id": label, "class_name": ("walking_slow", "walking_moderate", "walking_brisk", "running", "stair_climbing", "cycling_seated", "cycling_standing", "sitting", "standing", "lying")[label], "shape": list(sample.shape), "vae_checkpoint": vi.to_mapping(), "flow_checkpoint": fi.to_mapping()}

__all__ = ["class_id", "generate_harth_window", "train_harth_flow", "train_harth_vae"]
