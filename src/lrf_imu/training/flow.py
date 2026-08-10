"""Rectified Flow training primitives with the operational equations intact."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F


MODEL_TIME_SCALE = 1000.0
PAPER_NUM_STEPS = 10
PAPER_SEED = 42
LATENT_CHANNELS = 48
LATENT_TIME_STEPS = 40


class FlowTrainingError(ValueError):
    """Raised when a flow training or sampling contract is invalid."""


@dataclass(frozen=True)
class FlowMatchingBatch:
    """Synthetic-safe view of one flow-matching training draw."""

    z0: torch.Tensor
    z1: torch.Tensor
    t: torch.Tensor
    zt: torch.Tensor
    target: torch.Tensor

    def to_mapping(self) -> Dict[str, Any]:
        return {
            "z0_shape": list(self.z0.shape),
            "z1_shape": list(self.z1.shape),
            "zt_shape": list(self.zt.shape),
            "target_shape": list(self.target.shape),
            "batch_size": int(self.z0.shape[0]),
            "latent_channels": int(self.z0.shape[1]),
            "latent_time_steps": int(self.z0.shape[2]),
            "equations": {
                "z0": "VAE posterior mean",
                "z1": "N(0,I)",
                "zt": "(1-t)z0 + t z1",
                "target": "z1-z0",
                "model_time": "1000*t",
                "loss": "MSE(pred,target)",
            },
            "tensor_values_included": False,
        }


def _validate_latent_tensor(value: torch.Tensor, name: str) -> None:
    if not isinstance(value, torch.Tensor) or value.ndim != 3:
        raise FlowTrainingError(
            "{} must be a tensor with shape [batch, latent_channels, time]".format(name)
        )
    if value.shape[0] <= 0 or value.shape[1] <= 0 or value.shape[2] <= 0:
        raise FlowTrainingError("{} must have positive dimensions".format(name))


def _batch_times(
    t: Optional[torch.Tensor],
    batch_size: int,
    *,
    device: torch.device,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    if t is None:
        values = torch.rand(batch_size, device=device, generator=generator)
    else:
        values = torch.as_tensor(t, device=device, dtype=torch.float32).reshape(-1)
        if values.numel() == 1 and batch_size != 1:
            values = values.expand(batch_size)
        if values.numel() != batch_size:
            raise FlowTrainingError("t must have one value per batch item")
    if torch.any(values < 0.0) or torch.any(values > 1.0):
        raise FlowTrainingError("flow interpolation times must lie in [0,1]")
    return values


def interpolate_latents(
    z0: torch.Tensor,
    z1: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    """Apply ``zt = (1-t)z0 + t z1`` without changing tensor geometry."""

    _validate_latent_tensor(z0, "z0")
    _validate_latent_tensor(z1, "z1")
    if z0.shape != z1.shape:
        raise FlowTrainingError("z0 and z1 must have identical shapes")
    times = torch.as_tensor(t, device=z0.device, dtype=z0.dtype).reshape(-1)
    if times.numel() == 1 and z0.shape[0] != 1:
        times = times.expand(z0.shape[0])
    if times.numel() != z0.shape[0]:
        raise FlowTrainingError("t must have one value per batch item")
    return (1.0 - times[:, None, None]) * z0 + times[:, None, None] * z1


def _randn_like(value: torch.Tensor, generator: Optional[torch.Generator]) -> torch.Tensor:
    kwargs: Dict[str, Any] = {
        "device": value.device,
        "dtype": value.dtype,
    }
    if generator is not None:
        kwargs["generator"] = generator
    return torch.randn(value.shape, **kwargs)


def make_flow_matching_batch(
    z0: torch.Tensor,
    *,
    t: Optional[torch.Tensor] = None,
    z1: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
) -> FlowMatchingBatch:
    """Draw the exact conditional flow-matching tuple used by training."""

    _validate_latent_tensor(z0, "z0")
    noise = _randn_like(z0, generator) if z1 is None else z1
    _validate_latent_tensor(noise, "z1")
    if noise.shape != z0.shape:
        raise FlowTrainingError("z1 must have the same shape as z0")
    times = _batch_times(
        t,
        int(z0.shape[0]),
        device=z0.device,
        generator=generator,
    )
    zt = interpolate_latents(z0, noise, times)
    target = noise - z0
    return FlowMatchingBatch(z0=z0, z1=noise, t=times, zt=zt, target=target)


# Short aliases make the equations easy to discover for downstream callers.
flow_matching_batch = make_flow_matching_batch
flow_interpolation = interpolate_latents


def compute_flow_matching_loss(
    predicted_velocity: torch.Tensor,
    target_velocity: torch.Tensor,
) -> torch.Tensor:
    """Return the operational ``MSE(pred,target)`` objective."""

    if predicted_velocity.shape != target_velocity.shape:
        raise FlowTrainingError("predicted and target velocities must have identical shapes")
    return F.mse_loss(predicted_velocity, target_velocity)


compute_flow_loss = compute_flow_matching_loss


def model_time(t: torch.Tensor) -> torch.Tensor:
    """Convert normalized flow time to the source model's ``1000*t`` input."""

    return torch.as_tensor(t) * MODEL_TIME_SCALE


def reverse_euler_step(
    latent: torch.Tensor,
    velocity: torch.Tensor,
    dt: float,
) -> torch.Tensor:
    """Perform one explicit reverse Euler update ``z <- z - dt*v``."""

    if latent.shape != velocity.shape:
        raise FlowTrainingError("latent and velocity shapes must match")
    step = float(dt)
    if not np.isfinite(step) or step <= 0.0:
        raise FlowTrainingError("reverse Euler dt must be a finite positive number")
    return latent - velocity * step


reverse_euler_update = reverse_euler_step


def _module_device(module: Any) -> torch.device:
    try:
        return next(module.parameters()).device
    except (AttributeError, StopIteration):
        return torch.device("cpu")


class RectifiedFlowIMU:
    """Minimal VAE-conditioned Rectified Flow contract from the source code."""

    def __init__(
        self,
        vae: Any,
        unet: Any,
        conditioner: Any = None,
        device: Union[str, torch.device, None] = None,
    ) -> None:
        del conditioner  # The source conditioner was removed from the operational path.
        selected_device = _module_device(unet) if device is None else torch.device(device)
        self.device = selected_device
        self.vae = vae.to(selected_device) if hasattr(vae, "to") else vae
        self.unet = unet.to(selected_device) if hasattr(unet, "to") else unet
        vae_channels = getattr(self.vae, "input_channels", getattr(self.vae, "in_ch", None))
        flow_channels = getattr(self.unet, "flow_channels", None)
        if vae_channels is not None and flow_channels is not None and int(vae_channels) != int(flow_channels):
            raise FlowTrainingError(
                "paired VAE/flow channel mismatch: VAE has {}, flow has {}".format(
                    int(vae_channels), int(flow_channels)
                )
            )
        if hasattr(self.vae, "eval"):
            self.vae.eval()
        if hasattr(self.vae, "parameters"):
            for parameter in self.vae.parameters():
                parameter.requires_grad_(False)

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``z0`` as the VAE posterior mean, never a sampled posterior."""

        encoded = self.vae.encode(x)
        if isinstance(encoded, (tuple, list)):
            return encoded[0]
        if isinstance(encoded, Mapping):
            for key in ("mu", "mean", "posterior_mean"):
                if key in encoded:
                    return encoded[key]
        if isinstance(encoded, torch.Tensor):
            return encoded
        raise FlowTrainingError("VAE encode must return a posterior mean tensor")

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.vae.decode(z)

    def make_training_batch(
        self,
        x0: torch.Tensor,
        *,
        t: Optional[torch.Tensor] = None,
        z1: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> FlowMatchingBatch:
        inputs = x0.to(self.device)
        z0 = self.encode(inputs)
        return make_flow_matching_batch(z0, t=t, z1=z1, generator=generator)

    def training_loss(
        self,
        x0: torch.Tensor,
        labels: torch.Tensor,
        *,
        t: Optional[torch.Tensor] = None,
        z1: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        return_batch: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, FlowMatchingBatch]]:
        """Compute the six source equations in their operational order."""

        batch = self.make_training_batch(
            x0,
            t=t,
            z1=z1,
            generator=generator,
        )
        class_ids = torch.as_tensor(
            labels,
            device=self.device,
            dtype=torch.long,
        ).reshape(-1)
        if class_ids.numel() != batch.z0.shape[0]:
            raise FlowTrainingError("labels must have one value per batch item")
        predicted = self.unet(
            batch.zt,
            batch.t * MODEL_TIME_SCALE,
            class_ids,
        )
        loss = compute_flow_matching_loss(predicted, batch.target)
        return (loss, batch) if return_batch else loss

    @torch.no_grad()
    def sample_latent(
        self,
        labels: torch.Tensor,
        shape: Sequence[int],
        num_steps: Optional[int] = PAPER_NUM_STEPS,
        *,
        seed: Optional[int] = None,
        noise: Optional[torch.Tensor] = None,
        noise_scale: float = 1.0,
        record_every: Optional[int] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]]:
        """Integrate from ``t=1`` to ``t=0`` with explicit reverse Euler steps."""

        class_ids = torch.as_tensor(labels, device=self.device, dtype=torch.long).reshape(-1)
        if class_ids.numel() <= 0:
            raise FlowTrainingError("at least one class label is required")
        steps = PAPER_NUM_STEPS if num_steps is None else int(num_steps)
        if steps < 1:
            raise FlowTrainingError("num_steps must be >= 1")
        if len(tuple(shape)) != 2 or any(int(value) <= 0 for value in shape):
            raise FlowTrainingError("shape must be [latent_channels, latent_time_steps]")
        scale = float(noise_scale)
        if not np.isfinite(scale) or scale <= 0.0:
            raise FlowTrainingError("noise_scale must be positive and finite")

        generator = None
        if seed is not None:
            try:
                generator = torch.Generator(device=self.device.type)
            except (RuntimeError, TypeError):
                generator = torch.Generator()
            generator.manual_seed(int(seed))
        if noise is None:
            initial = torch.randn(
                (class_ids.shape[0], int(shape[0]), int(shape[1])),
                device=self.device,
                generator=generator,
            ) * scale
        else:
            initial = noise.to(self.device)
            if tuple(initial.shape) != (
                int(class_ids.shape[0]),
                int(shape[0]),
                int(shape[1]),
            ):
                raise FlowTrainingError("provided noise shape does not match labels and shape")
        latent = initial
        dt = 1.0 / float(steps)
        states = []
        interval = None if record_every is None else int(record_every)
        if interval is not None and interval < 1:
            raise FlowTrainingError("record_every must be >= 1")
        if interval is not None:
            states.append(latent.clone())
        for index in range(steps):
            current_t = 1.0 - float(index) * dt
            times = torch.full(
                (class_ids.shape[0],),
                current_t,
                device=self.device,
                dtype=latent.dtype,
            )
            velocity = self.unet(latent, times * MODEL_TIME_SCALE, class_ids)
            latent = reverse_euler_step(latent, velocity, dt)
            completed = index + 1
            if interval is not None and (
                completed == steps or completed % interval == 0
            ):
                states.append(latent.clone())
        if interval is not None:
            return latent, tuple(states)
        return latent

    @torch.no_grad()
    def sample(
        self,
        labels: torch.Tensor,
        num_steps: Optional[int] = PAPER_NUM_STEPS,
        *,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """Sample and decode one native VAE window per requested label."""

        channels = int(
            getattr(self.vae, "input_channels", getattr(self.vae, "in_ch", 6))
        )
        dummy = torch.zeros(1, channels, 160, device=self.device)
        latent_shape = tuple(self.encode(dummy).shape[1:])
        latent = self.sample_latent(
            labels,
            latent_shape,
            num_steps=num_steps,
            seed=seed,
        )
        if isinstance(latent, tuple):
            latent = latent[0]
        return self.decode(latent)


@dataclass(frozen=True)
class FlowTrainingResult:
    """Small metadata-only result returned by :func:`train_flow`."""

    device: str
    epochs_ran: int
    best_epoch: int
    best_val_loss: float
    train_losses: Tuple[float, ...]
    val_losses: Tuple[float, ...]
    checkpoint_path: Optional[str]

    def to_mapping(self) -> Dict[str, Any]:
        return {
            "device": self.device,
            "epochs_ran": self.epochs_ran,
            "best_epoch": self.best_epoch,
            "best_val_loss": self.best_val_loss,
            "train_loss_count": len(self.train_losses),
            "validation_loss_count": len(self.val_losses),
            "checkpoint_path": self.checkpoint_path,
            "tensor_values_included": False,
        }


def _batch_pair(batch: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
        return batch[0], batch[1]
    raise FlowTrainingError("flow loaders must yield (input, class_label) batches")


def _loss_tensor(value: Union[torch.Tensor, Tuple[torch.Tensor, FlowMatchingBatch]]) -> torch.Tensor:
    if isinstance(value, tuple):
        raise FlowTrainingError("flow training loss unexpectedly returned its matching batch")
    return value


def _checkpoint_payload(
    model: Any,
    optimizer: Any,
    *,
    epoch: int,
    history: Mapping[str, Any],
    config: Mapping[str, Any],
    val_loss: float,
) -> Dict[str, Any]:
    """Build the exact six-key operational flow checkpoint boundary."""

    return {
        "config": dict(config),
        "epoch": int(epoch),
        "history": dict(history),
        "opt": optimizer.state_dict(),
        "unet": model.state_dict(),
        "val_loss": float(val_loss),
    }


def train_flow(
    model: Any,
    vae: Any,
    train_loader: Iterable[Any],
    validation_loader: Iterable[Any],
    *,
    epochs: int = 1,
    learning_rate: float = 5e-4,
    grad_clip: float = 1.0,
    weight_decay: float = 1e-4,
    device: Union[str, torch.device] = "cpu",
    seed: int = PAPER_SEED,
    checkpoint_path: Optional[Union[str, Path]] = None,
    config: Optional[Mapping[str, Any]] = None,
) -> FlowTrainingResult:
    """Run the minimal source-compatible AdamW flow loop for synthetic probes."""

    if int(epochs) < 1:
        raise FlowTrainingError("epochs must be >= 1")
    if float(learning_rate) <= 0.0:
        raise FlowTrainingError("learning_rate must be positive")
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    target_device = torch.device(device)
    model = model.to(target_device)
    trainer = RectifiedFlowIMU(vae, model, device=target_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(learning_rate),
        betas=(0.9, 0.95),
        weight_decay=float(weight_decay),
    )
    history: Dict[str, Any] = {"train_loss": [], "val_loss": [], "epochs": []}
    best_loss = float("inf")
    best_epoch = 0

    for epoch in range(1, int(epochs) + 1):
        model.train()
        train_values = []
        for raw_batch in train_loader:
            inputs, labels = _batch_pair(raw_batch)
            optimizer.zero_grad(set_to_none=True)
            loss = _loss_tensor(trainer.training_loss(inputs, labels))
            loss.backward()
            if float(grad_clip) > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            optimizer.step()
            train_values.append(float(loss.detach().cpu()))
        if not train_values:
            raise FlowTrainingError("train loader yielded no batches")

        model.eval()
        val_values = []
        with torch.no_grad():
            for raw_batch in validation_loader:
                inputs, labels = _batch_pair(raw_batch)
                validation_loss = _loss_tensor(trainer.training_loss(inputs, labels))
                val_values.append(float(validation_loss.detach().cpu()))
        if not val_values:
            raise FlowTrainingError("validation loader yielded no batches")
        train_loss = float(sum(train_values) / len(train_values))
        val_loss = float(sum(val_values) / len(val_values))
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["epochs"].append(epoch)
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            if checkpoint_path is not None:
                destination = Path(checkpoint_path).expanduser().resolve()
                if destination.exists() and destination.is_dir():
                    raise FlowTrainingError("checkpoint_path must be a file path")
                destination.parent.mkdir(parents=True, exist_ok=True)
                payload_config = dict(config or {})
                payload_config.setdefault("model_ch", int(getattr(model, "model_ch", 256)))
                payload_config.setdefault("latent_dim", int(getattr(model, "in_ch", 48)))
                payload_config.setdefault("num_classes", int(getattr(model, "num_classes", 4)))
                torch.save(
                    _checkpoint_payload(
                        model,
                        optimizer,
                        epoch=epoch,
                        history=history,
                        config=payload_config,
                        val_loss=val_loss,
                    ),
                    destination,
                )

    return FlowTrainingResult(
        device=str(target_device),
        epochs_ran=int(epochs),
        best_epoch=best_epoch,
        best_val_loss=best_loss,
        train_losses=tuple(history["train_loss"]),
        val_losses=tuple(history["val_loss"]),
        checkpoint_path=None if checkpoint_path is None else str(Path(checkpoint_path).expanduser().resolve()),
    )


__all__ = [
    "FlowMatchingBatch",
    "FlowTrainingError",
    "FlowTrainingResult",
    "LATENT_CHANNELS",
    "LATENT_TIME_STEPS",
    "MODEL_TIME_SCALE",
    "PAPER_NUM_STEPS",
    "PAPER_SEED",
    "RectifiedFlowIMU",
    "compute_flow_loss",
    "compute_flow_matching_loss",
    "flow_interpolation",
    "flow_matching_batch",
    "interpolate_latents",
    "make_flow_matching_batch",
    "model_time",
    "reverse_euler_step",
    "reverse_euler_update",
    "train_flow",
]
