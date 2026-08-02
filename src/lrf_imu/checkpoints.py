"""Safe, explicit-path boundaries for public VAE checkpoints."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch

from .models.vae import LatentVAE1D, SUPPORTED_CHANNELS


PathLike = Union[str, Path]
CHECKPOINT_ROOT_KEY = "vae"
CHECKPOINT_INPUT_LENGTH = 160
CHECKPOINT_DEFAULT_DOWN_LEVELS = 2
# Retain the historical public constant; inspection derives geometry dynamically.
CHECKPOINT_LATENT_TIME_STEPS = CHECKPOINT_INPUT_LENGTH // (2**CHECKPOINT_DEFAULT_DOWN_LEVELS)


class CheckpointError(ValueError):
    """Raised when an explicit checkpoint fails the public schema boundary."""


@dataclass(frozen=True)
class CheckpointInspection:
    """Safe checkpoint metadata; tensor values are intentionally excluded."""

    checkpoint_path: str
    channels: int
    latent_channels: int
    latent_stride: int
    input_length: int
    latent_time_steps: int
    state_dict_keys: Tuple[str, ...]
    tensor_shapes: Dict[str, Tuple[int, ...]]
    tensor_dtypes: Dict[str, str]

    def to_mapping(self) -> Dict[str, Any]:
        return {
            "checkpoint_path": self.checkpoint_path,
            "channels": self.channels,
            "latent_channels": self.latent_channels,
            "latent_stride": self.latent_stride,
            "input_length": self.input_length,
            "latent_time_steps": self.latent_time_steps,
            "state_dict_key_count": len(self.state_dict_keys),
            "state_dict_keys": list(self.state_dict_keys),
            "tensor_shapes": {
                key: list(shape) for key, shape in self.tensor_shapes.items()
            },
            "tensor_dtypes": dict(self.tensor_dtypes),
            "payload_values_included": False,
        }


def _explicit_checkpoint_path(path: PathLike) -> Path:
    if path is None or str(path).strip() == "":
        raise CheckpointError("an explicit checkpoint path is required")
    candidate = Path(path).expanduser()
    if not candidate.is_file():
        raise CheckpointError("checkpoint path does not name a file: {}".format(candidate))
    return candidate.resolve()


def _safe_load(path: Path) -> Any:
    """Load only tensor-safe payloads; never fall back to arbitrary pickle."""

    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError as exc:
        raise CheckpointError(
            "this PyTorch version does not expose weights_only loading; refusing unsafe checkpoint loading"
        ) from exc
    except Exception as exc:
        raise CheckpointError("could not safely load checkpoint") from exc


def expected_state_dict_shapes(
    channels: int,
    latent_channels: int = 48,
    down_levels: int = 2,
    hidden: int = 160,
) -> Dict[str, Tuple[int, ...]]:
    """Return the exact source-compatible parameter schema."""

    if channels not in SUPPORTED_CHANNELS:
        raise CheckpointError("channels must be 3 or 6")
    if latent_channels <= 0 or hidden <= 0 or down_levels < 0:
        raise CheckpointError("invalid VAE checkpoint geometry")

    shapes: Dict[str, Tuple[int, ...]] = {}

    def add_conv(prefix: str, out_channels: int, in_channels: int, kernel: int) -> None:
        shapes[prefix + ".weight"] = (out_channels, in_channels, kernel)
        shapes[prefix + ".bias"] = (out_channels,)

    add_conv("enc_in.0", hidden, channels, 7)
    for level in range(down_levels):
        base = level * 4
        add_conv("downs.{}".format(base), hidden, hidden, 4)
        add_conv("downs.{}".format(base + 2), hidden, hidden, 3)
    add_conv("mu_head", latent_channels, hidden, 1)
    add_conv("lv_head", latent_channels, hidden, 1)
    add_conv("dec_in.0", hidden, latent_channels, 1)
    for level in range(down_levels):
        base = level * 4
        # ConvTranspose1d has the same shape here because both hidden widths
        # are 160; the parameter names remain those emitted by PyTorch.
        add_conv("ups.{}".format(base), hidden, hidden, 4)
        add_conv("ups.{}".format(base + 2), hidden, hidden, 3)
    add_conv("dec_out", channels, hidden, 1)
    return shapes


def _state_dict_from_payload(payload: Any) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise CheckpointError("checkpoint payload must be a mapping")
    if set(payload.keys()) != {CHECKPOINT_ROOT_KEY}:
        raise CheckpointError("checkpoint root keys must be exactly ['vae']")
    state_dict = payload[CHECKPOINT_ROOT_KEY]
    if not isinstance(state_dict, Mapping):
        raise CheckpointError("checkpoint root 'vae' must contain a state-dict mapping")
    return state_dict


def _infer_channels(state_dict: Mapping[str, Any]) -> int:
    try:
        decoder_weight = state_dict["dec_out.weight"]
        decoder_channels = int(decoder_weight.shape[0])
    except (KeyError, AttributeError, IndexError, TypeError, ValueError) as exc:
        raise CheckpointError("checkpoint is missing channel-bearing VAE tensors") from exc
    if decoder_channels not in SUPPORTED_CHANNELS:
        raise CheckpointError("checkpoint channel count must be 3 or 6")
    return decoder_channels


def _validate_state_dict(
    state_dict: Mapping[str, Any],
    *,
    channels: int,
    latent_channels: int,
    down_levels: int,
) -> Dict[str, Tuple[int, ...]]:
    expected = expected_state_dict_shapes(
        channels,
        latent_channels=latent_channels,
        down_levels=down_levels,
    )
    actual_keys = set(state_dict.keys())
    expected_keys = set(expected.keys())
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        extra = sorted(actual_keys - expected_keys)
        raise CheckpointError(
            "checkpoint state-dict keys do not match the VAE schema (missing={}, extra={})".format(
                missing, extra
            )
        )
    for key, shape in expected.items():
        value = state_dict[key]
        if not isinstance(value, torch.Tensor):
            raise CheckpointError("checkpoint tensor {} is not a tensor".format(key))
        if tuple(value.shape) != shape:
            raise CheckpointError(
                "checkpoint tensor {} has shape {}, expected {}".format(
                    key, tuple(value.shape), shape
                )
            )
    return expected


def _latent_time_steps_for_geometry(input_length: int, down_levels: int) -> int:
    """Derive the latent temporal length from the public input geometry."""

    if input_length <= 0 or down_levels < 0:
        raise CheckpointError("invalid VAE checkpoint geometry")
    latent_stride = 2**int(down_levels)
    if input_length % latent_stride != 0:
        raise CheckpointError(
            "input length {} is not divisible by latent stride {}".format(
                input_length, latent_stride
            )
        )
    return input_length // latent_stride


def inspect_vae_checkpoint(
    checkpoint_path: PathLike,
    *,
    channels: Optional[int] = None,
    latent_channels: int = 48,
    down_levels: int = 2,
) -> CheckpointInspection:
    """Validate and summarize one explicit weights-only checkpoint."""

    path = _explicit_checkpoint_path(checkpoint_path)
    payload = _safe_load(path)
    state_dict = _state_dict_from_payload(payload)
    inferred_channels = _infer_channels(state_dict)
    selected_channels = inferred_channels if channels is None else int(channels)
    if selected_channels not in SUPPORTED_CHANNELS:
        raise CheckpointError("channels must be 3 or 6")
    if selected_channels != inferred_channels:
        raise CheckpointError(
            "checkpoint contains {} channels but {} were requested".format(
                inferred_channels, selected_channels
            )
        )
    expected = _validate_state_dict(
        state_dict,
        channels=selected_channels,
        latent_channels=latent_channels,
        down_levels=down_levels,
    )
    latent_stride = 2**int(down_levels)
    latent_time_steps = _latent_time_steps_for_geometry(
        CHECKPOINT_INPUT_LENGTH, int(down_levels)
    )
    shapes = {key: tuple(state_dict[key].shape) for key in expected}
    dtypes = {key: str(state_dict[key].dtype) for key in expected}
    return CheckpointInspection(
        checkpoint_path=str(path),
        channels=selected_channels,
        latent_channels=int(latent_channels),
        latent_stride=latent_stride,
        input_length=CHECKPOINT_INPUT_LENGTH,
        latent_time_steps=latent_time_steps,
        state_dict_keys=tuple(expected.keys()),
        tensor_shapes=shapes,
        tensor_dtypes=dtypes,
    )


def load_vae_checkpoint(
    checkpoint_path: PathLike,
    *,
    channels: Optional[int] = None,
    latent_channels: int = 48,
    down_levels: int = 2,
    device: Union[str, torch.device] = "cpu",
) -> Tuple[LatentVAE1D, CheckpointInspection]:
    """Safely load one validated checkpoint into a copied public model."""

    path = _explicit_checkpoint_path(checkpoint_path)
    payload = _safe_load(path)
    state_dict = _state_dict_from_payload(payload)
    inferred_channels = _infer_channels(state_dict)
    selected_channels = inferred_channels if channels is None else int(channels)
    if selected_channels != inferred_channels:
        raise CheckpointError(
            "checkpoint contains {} channels but {} were requested".format(
                inferred_channels, selected_channels
            )
        )
    _validate_state_dict(
        state_dict,
        channels=selected_channels,
        latent_channels=latent_channels,
        down_levels=down_levels,
    )
    model = LatentVAE1D(
        in_ch=selected_channels,
        z_ch=latent_channels,
        down_levels=down_levels,
    )
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    inspection = inspect_vae_checkpoint(
        path,
        channels=selected_channels,
        latent_channels=latent_channels,
        down_levels=down_levels,
    )
    return model, inspection


def save_vae_checkpoint(model: LatentVAE1D, checkpoint_path: PathLike) -> Path:
    """Save the source-compatible root-key schema to an explicit path."""

    path = _explicit_save_path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({CHECKPOINT_ROOT_KEY: model.state_dict()}, path)
    return path


def _explicit_save_path(path: PathLike) -> Path:
    if path is None or str(path).strip() == "":
        raise CheckpointError("an explicit checkpoint path is required")
    candidate = Path(path).expanduser()
    if candidate.exists() and candidate.is_dir():
        raise CheckpointError("checkpoint path must be a file path")
    return candidate.resolve()


__all__ = [
    "CHECKPOINT_INPUT_LENGTH",
    "CHECKPOINT_LATENT_TIME_STEPS",
    "CHECKPOINT_ROOT_KEY",
    "CheckpointError",
    "CheckpointInspection",
    "expected_state_dict_shapes",
    "inspect_vae_checkpoint",
    "load_vae_checkpoint",
    "save_vae_checkpoint",
]
