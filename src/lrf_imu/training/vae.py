"""Copied VAE augmentation, loss, profile, and training semantics."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, fields, replace
from pathlib import Path
import random
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple, Union, cast

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset

from ..checkpoints import save_vae_checkpoint
from ..config import AugmentationConfig, VAEConfig, VAETrainingConfig
from ..models.vae import LatentVAE1D


MIGRATION_PROVENANCE = {
    "VAE_logic.py": {
        "original_relative_path": "VAE/VAE_logic.py",
        "original_sha256": "3C989BB8242236D3107AE75A1533622D955D6E739D0B44103636897D01E80505",
        "public_destination": "src/lrf_imu/training/vae.py",
        "copied_minimal_modifications": (
            "Copied augment_batch, compute_vae_loss, AdamW/beta/validation/early-stop "
            "semantics; removed dataset-specific orchestration, plots, logs, and "
            "import-time environment mutation; made CPU AMP conditional."
        ),
    },
    "Run_VAE_Pretraings.ps1": {
        "original_relative_path": "VAE/Run_VAE_Pretraings.ps1",
        "original_sha256": "2BE8A6D0645431E36E336DB6AA0C4174668F53A6720036B56414E5F0709C1B3D",
        "public_destination": "src/lrf_imu/training/vae.py",
        "copied_minimal_modifications": (
            "Represented the wrapper's observed 6CH/3CH namespace, seed, CPU/CUDA, "
            "AMP, optimizer, and explicit-checkpoint behavior as Python APIs; no "
            "PowerShell path discovery or implicit filesystem writes were copied."
        ),
    },
}


@dataclass(frozen=True)
class VAEProfile:
    """An evidence-labelled training profile; none claims exact reproduction."""

    name: str
    batch_size: int
    learning_rate: float
    max_epochs: int
    early_stop_min_epochs: int
    early_stop_patience: int
    early_stop_min_delta: float
    l2_weight: float
    l1_weight: float
    beta_init: float
    beta_min: float
    beta_decay: float
    use_amp_bf16: bool
    grad_clip: float
    augmentation: AugmentationConfig
    num_workers: int = 0
    pin_memory: bool = True
    exact_paper_reproduction: bool = False

    def to_mapping(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "max_epochs": self.max_epochs,
            "early_stop_min_epochs": self.early_stop_min_epochs,
            "early_stop_patience": self.early_stop_patience,
            "early_stop_min_delta": self.early_stop_min_delta,
            "l2_weight": self.l2_weight,
            "l1_weight": self.l1_weight,
            "beta_init": self.beta_init,
            "beta_min": self.beta_min,
            "beta_decay": self.beta_decay,
            "use_amp_bf16": self.use_amp_bf16,
            "grad_clip": self.grad_clip,
            "augmentation": {
                "enabled": self.augmentation.enabled,
                "jitter": self.augmentation.jitter,
                "scale": self.augmentation.scale,
                "time_mask": self.augmentation.time_mask,
            },
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "exact_paper_reproduction": self.exact_paper_reproduction,
        }


_OBSERVED_WRAPPER_PROFILE = VAEProfile(
    name="observed_wrapper_compatibility",
    batch_size=256,
    learning_rate=0.001,
    max_epochs=1000,
    early_stop_min_epochs=200,
    early_stop_patience=100,
    early_stop_min_delta=1e-4,
    l2_weight=0.5,
    l1_weight=0.1,
    beta_init=0.08,
    beta_min=0.04,
    beta_decay=0.995,
    use_amp_bf16=True,
    grad_clip=1.0,
    augmentation=AugmentationConfig(
        enabled=True, jitter=0.008, scale=0.04, time_mask=0.05
    ),
    num_workers=0,
    pin_memory=True,
)

_OLDER_MANUSCRIPT_PROFILE = VAEProfile(
    name="older_manuscript_reported",
    batch_size=128,
    learning_rate=0.001,
    max_epochs=1000,
    early_stop_min_epochs=50,
    early_stop_patience=30,
    early_stop_min_delta=1e-4,
    l2_weight=1.0,
    l1_weight=0.1,
    beta_init=0.005,
    beta_min=0.00001,
    beta_decay=0.7,
    use_amp_bf16=True,
    grad_clip=1.0,
    augmentation=AugmentationConfig(enabled=False, jitter=0.0, scale=0.0, time_mask=0.0),
    num_workers=4,
    pin_memory=True,
)

_PROFILE_NAMES = {
    _OBSERVED_WRAPPER_PROFILE.name: _OBSERVED_WRAPPER_PROFILE,
    _OLDER_MANUSCRIPT_PROFILE.name: _OLDER_MANUSCRIPT_PROFILE,
}
_CUSTOM_FIELDS = frozenset(
    field.name
    for field in fields(VAEProfile)
    if field.name not in {"name", "exact_paper_reproduction"}
)


def select_vae_profile(
    name: str = "observed_wrapper_compatibility",
    *,
    overrides: Optional[Mapping[str, Any]] = None,
) -> VAEProfile:
    """Select one named profile or create a transparent custom variant."""

    normalized = str(name).strip().lower()
    if normalized in _PROFILE_NAMES:
        if overrides:
            raise ValueError("named VAE profiles do not accept custom overrides")
        return _PROFILE_NAMES[normalized]
    if normalized != "custom":
        raise ValueError(
            "unknown VAE profile {}; choose observed_wrapper_compatibility, "
            "older_manuscript_reported, or custom".format(name)
        )

    values: Dict[str, Any] = dict(overrides or {})
    unknown = set(values).difference(_CUSTOM_FIELDS)
    if unknown:
        raise ValueError("unknown custom VAE profile field(s): {}".format(sorted(unknown)))
    if isinstance(values.get("augmentation"), Mapping):
        values["augmentation"] = AugmentationConfig(**dict(values["augmentation"]))
    values.setdefault("name", "custom")
    values.pop("name", None)
    values["exact_paper_reproduction"] = False
    return replace(_OBSERVED_WRAPPER_PROFILE, name="custom", **values)


def profile_from_config(config: VAEConfig) -> VAEProfile:
    """Map the release VAE config to its named or custom executable profile."""

    if config.schedule_profile != "custom":
        return select_vae_profile(config.schedule_profile)
    training = config.training
    return select_vae_profile(
        "custom",
        overrides={
            "batch_size": training.batch_size,
            "learning_rate": training.learning_rate,
            "max_epochs": training.max_epochs,
            "early_stop_min_epochs": training.early_stop_min_epochs,
            "early_stop_patience": training.early_stop_patience,
            "l2_weight": training.l2_weight,
            "l1_weight": training.l1_weight,
            "beta_init": training.beta_init,
            "beta_min": training.beta_min,
            "beta_decay": training.beta_decay,
            "use_amp_bf16": training.use_amp_bf16,
            "grad_clip": training.grad_clip,
            "augmentation": training.augmentation,
        },
    )


def augment_batch(
    x: torch.Tensor,
    aug: Optional[Union[AugmentationConfig, Mapping[str, float]]] = None,
) -> torch.Tensor:
    """Apply the source wrapper's jitter, per-channel scale, and time mask."""

    if aug is None:
        cfg: Mapping[str, Any] = {
            "jitter": 0.008,
            "scale": 0.04,
            "time_mask": 0.05,
        }
    elif isinstance(aug, AugmentationConfig):
        cfg = {
            "jitter": aug.jitter,
            "scale": aug.scale,
            "time_mask": aug.time_mask,
        }
    else:
        cfg = aug

    if cfg.get("jitter", 0) > 0:
        x = x + torch.randn_like(x) * float(cfg["jitter"])

    if cfg.get("scale", 0) > 0:
        scale = 1.0 + torch.randn(
            x.size(0), x.size(1), 1, device=x.device, dtype=x.dtype
        ) * float(cfg["scale"])
        x = x * scale

    if cfg.get("time_mask", 0) > 0:
        time_steps = x.size(-1)
        mask = (
            torch.rand(x.size(0), 1, time_steps, device=x.device)
            < float(cfg["time_mask"])
        )
        x = x.masked_fill(mask, 0.0)
    return x


def compute_vae_loss(
    x0: torch.Tensor,
    x_hat: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float,
    weights: Mapping[str, float],
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Return the copied weighted reconstruction plus KL loss."""

    l2 = F.mse_loss(x_hat, x0)
    l1 = (x_hat - x0).abs().mean()
    recon = weights.get("l2", 1.0) * l2 + weights.get("l1", 0.1) * l1
    kl = 0.5 * torch.mean(torch.exp(logvar) + mu**2 - 1.0 - logvar)
    loss = recon + beta * kl
    logs = {
        "l2": float(l2.detach().cpu()),
        "l1": float(l1.detach().cpu()),
        "kl": float(kl.detach().cpu()),
        "beta": float(beta),
    }
    return loss, logs


def set_seed(seed: int = 42) -> None:
    """Set the source wrapper's random generators without changing the environment."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def resolve_device(device: Union[str, torch.device] = "auto") -> torch.device:
    requested = str(device).lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    if requested not in {"cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    return torch.device(requested)


def _autocast(device: torch.device, enabled: bool):
    if not enabled or device.type != "cuda":
        return nullcontext()
    try:
        if hasattr(torch.cuda, "is_bf16_supported") and not torch.cuda.is_bf16_supported():
            return nullcontext()
    except Exception:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def _batch_input(batch: Any) -> torch.Tensor:
    if isinstance(batch, Mapping):
        if "x" not in batch:
            raise ValueError("mapping batches must contain an 'x' input")
        value = batch["x"]
    elif isinstance(batch, (tuple, list)):
        if not batch:
            raise ValueError("empty training batch")
        value = batch[0]
    else:
        value = batch
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)
    if not value.is_floating_point():
        value = value.float()
    return value


def _loader(
    data: Union[DataLoader, Dataset, Iterable[Any]],
    *,
    batch_size: int,
    shuffle: bool,
    drop_last: bool,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    if isinstance(data, DataLoader):
        return data
    return DataLoader(
        cast(Dataset, data),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )


@dataclass(frozen=True)
class TrainingResult:
    """Machine-readable summary of one in-memory VAE training run."""

    best_validation_loss: float
    best_epoch: int
    epochs_completed: int
    train_losses: Tuple[float, ...]
    validation_losses: Tuple[float, ...]
    profile: str
    device: str
    checkpoint_path: Optional[str]

    def to_mapping(self) -> Dict[str, Any]:
        return {
            "best_validation_loss": self.best_validation_loss,
            "best_epoch": self.best_epoch,
            "epochs_completed": self.epochs_completed,
            "train_loss_count": len(self.train_losses),
            "validation_loss_count": len(self.validation_losses),
            "profile": self.profile,
            "device": self.device,
            "checkpoint_path": self.checkpoint_path,
            "loss_values_included": False,
        }


def _profile_from_training_config(training_config: VAETrainingConfig) -> VAEProfile:
    return select_vae_profile(
        "custom",
        overrides={
            "batch_size": training_config.batch_size,
            "learning_rate": training_config.learning_rate,
            "max_epochs": training_config.max_epochs,
            "early_stop_min_epochs": training_config.early_stop_min_epochs,
            "early_stop_patience": training_config.early_stop_patience,
            "l2_weight": training_config.l2_weight,
            "l1_weight": training_config.l1_weight,
            "beta_init": training_config.beta_init,
            "beta_min": training_config.beta_min,
            "beta_decay": training_config.beta_decay,
            "use_amp_bf16": training_config.use_amp_bf16,
            "grad_clip": training_config.grad_clip,
            "augmentation": training_config.augmentation,
        },
    )


def train_vae(
    model: LatentVAE1D,
    train_data: Union[DataLoader, Dataset, Iterable[Any]],
    validation_data: Union[DataLoader, Dataset, Iterable[Any]],
    *,
    profile: Optional[VAEProfile] = None,
    training_config: Optional[VAETrainingConfig] = None,
    checkpoint_path: Optional[Union[str, Path]] = None,
    device: Union[str, torch.device] = "auto",
    seed: int = 42,
    num_workers: Optional[int] = None,
    pin_memory: Optional[bool] = None,
    drop_last: bool = True,
) -> TrainingResult:
    """Train with the copied wrapper semantics and optional explicit checkpoint."""

    if profile is not None and training_config is not None:
        raise ValueError("provide profile or training_config, not both")
    if profile is None:
        profile = (
            _profile_from_training_config(training_config)
            if training_config is not None
            else select_vae_profile()
        )
    if profile.batch_size <= 0 or profile.max_epochs <= 0:
        raise ValueError("VAE profile batch_size and max_epochs must be positive")
    if not 0 < profile.beta_min <= profile.beta_init:
        raise ValueError("VAE beta schedule is invalid")

    target_device = resolve_device(device)
    set_seed(seed)
    model = model.to(target_device)
    worker_count = profile.num_workers if num_workers is None else int(num_workers)
    use_pin_memory = profile.pin_memory if pin_memory is None else bool(pin_memory)
    if worker_count < 0:
        raise ValueError("num_workers must not be negative")

    train_loader = _loader(
        train_data,
        batch_size=profile.batch_size,
        shuffle=True,
        drop_last=drop_last,
        num_workers=worker_count,
        pin_memory=use_pin_memory,
    )
    validation_loader = _loader(
        validation_data,
        batch_size=profile.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=worker_count,
        pin_memory=use_pin_memory,
    )

    if len(train_loader) == 0 or len(validation_loader) == 0:
        raise ValueError("VAE train and validation loaders must both contain batches")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=profile.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=1e-4,
    )
    weights = {"l2": profile.l2_weight, "l1": profile.l1_weight}
    best_validation = float("inf")
    best_epoch = 0
    no_improve = 0
    beta = float(profile.beta_init)
    best_state: Optional[Dict[str, torch.Tensor]] = None
    train_losses = []
    validation_losses = []

    for epoch in range(1, profile.max_epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            inputs = _batch_input(batch).to(target_device, non_blocking=True)
            if profile.augmentation.enabled:
                inputs = augment_batch(inputs, profile.augmentation)
            optimizer.zero_grad(set_to_none=True)
            with _autocast(target_device, profile.use_amp_bf16):
                reconstruction, mu, logvar = model(inputs, deterministic=True)
                loss, _ = compute_vae_loss(
                    inputs, reconstruction, mu, logvar, beta, weights
                )
            loss.backward()
            if profile.grad_clip > 0:
                clip_grad_norm_(model.parameters(), profile.grad_clip)
            optimizer.step()
            train_loss += float(loss.detach().cpu())
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        validation_loss = 0.0
        with torch.no_grad():
            for batch in validation_loader:
                inputs = _batch_input(batch).to(target_device, non_blocking=True)
                with _autocast(target_device, profile.use_amp_bf16):
                    reconstruction, mu, logvar = model(inputs, deterministic=True)
                    loss, _ = compute_vae_loss(
                        inputs,
                        reconstruction,
                        mu,
                        logvar,
                        beta=0.0,
                        weights=weights,
                    )
                validation_loss += float(loss.detach().cpu())
        validation_loss /= len(validation_loader)
        validation_losses.append(validation_loss)

        beta = max(profile.beta_min, beta * profile.beta_decay)
        improved = (best_validation - validation_loss) > profile.early_stop_min_delta
        if improved:
            best_validation = validation_loss
            best_epoch = epoch
            no_improve = 0
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            if checkpoint_path is not None:
                save_vae_checkpoint(model, checkpoint_path)
        else:
            no_improve += 1

        if (
            epoch >= profile.early_stop_min_epochs
            and no_improve >= profile.early_stop_patience
        ):
            break

    if best_state is None:
        raise ValueError("VAE training did not produce a validation checkpoint")
    model.load_state_dict(best_state, strict=True)

    return TrainingResult(
        best_validation_loss=float(best_validation),
        best_epoch=int(best_epoch),
        epochs_completed=len(train_losses),
        train_losses=tuple(float(value) for value in train_losses),
        validation_losses=tuple(float(value) for value in validation_losses),
        profile=profile.name,
        device=str(target_device),
        checkpoint_path=str(checkpoint_path) if checkpoint_path is not None else None,
    )


__all__ = [
    "MIGRATION_PROVENANCE",
    "TrainingResult",
    "VAEProfile",
    "augment_batch",
    "compute_vae_loss",
    "profile_from_config",
    "resolve_device",
    "select_vae_profile",
    "set_seed",
    "train_vae",
]
