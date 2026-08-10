"""Safe explicit-path boundaries for public VAE and Rectified Flow checkpoints."""

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
CHECKPOINT_LATENT_TIME_STEPS = CHECKPOINT_INPUT_LENGTH // (2**CHECKPOINT_DEFAULT_DOWN_LEVELS)


class CheckpointError(ValueError):
    """Raised when an explicit checkpoint fails the public schema boundary."""


@dataclass(frozen=True)
class CheckpointInspection:
    """Safe VAE checkpoint metadata; tensor values are intentionally excluded."""

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


def _explicit_save_path(path: PathLike) -> Path:
    if path is None or str(path).strip() == "":
        raise CheckpointError("an explicit checkpoint path is required")
    candidate = Path(path).expanduser()
    if candidate.exists() and candidate.is_dir():
        raise CheckpointError("checkpoint path must be a file path")
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
    """Return the exact source-compatible VAE parameter schema."""

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
        channels = int(state_dict["dec_out.weight"].shape[0])
    except (KeyError, AttributeError, IndexError, TypeError, ValueError) as exc:
        raise CheckpointError("checkpoint is missing channel-bearing VAE tensors") from exc
    if channels not in SUPPORTED_CHANNELS:
        raise CheckpointError("checkpoint channel count must be 3 or 6")
    return channels


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
    if input_length <= 0 or down_levels < 0:
        raise CheckpointError("invalid VAE checkpoint geometry")
    stride = 2**int(down_levels)
    if input_length % stride != 0:
        raise CheckpointError(
            "input length {} is not divisible by latent stride {}".format(
                input_length, stride
            )
        )
    return input_length // stride


def inspect_vae_checkpoint(
    checkpoint_path: PathLike,
    *,
    channels: Optional[int] = None,
    latent_channels: int = 48,
    down_levels: int = 2,
) -> CheckpointInspection:
    """Validate and summarize one explicit weights-only VAE checkpoint."""

    path = _explicit_checkpoint_path(checkpoint_path)
    state_dict = _state_dict_from_payload(_safe_load(path))
    inferred = _infer_channels(state_dict)
    selected = inferred if channels is None else int(channels)
    if selected not in SUPPORTED_CHANNELS:
        raise CheckpointError("channels must be 3 or 6")
    if selected != inferred:
        raise CheckpointError(
            "checkpoint contains {} channels but {} were requested".format(inferred, selected)
        )
    expected = _validate_state_dict(
        state_dict,
        channels=selected,
        latent_channels=latent_channels,
        down_levels=down_levels,
    )
    stride = 2**int(down_levels)
    return CheckpointInspection(
        checkpoint_path=str(path),
        channels=selected,
        latent_channels=int(latent_channels),
        latent_stride=stride,
        input_length=CHECKPOINT_INPUT_LENGTH,
        latent_time_steps=_latent_time_steps_for_geometry(CHECKPOINT_INPUT_LENGTH, int(down_levels)),
        state_dict_keys=tuple(expected.keys()),
        tensor_shapes={key: tuple(state_dict[key].shape) for key in expected},
        tensor_dtypes={key: str(state_dict[key].dtype) for key in expected},
    )


def load_vae_checkpoint(
    checkpoint_path: PathLike,
    *,
    channels: Optional[int] = None,
    latent_channels: int = 48,
    down_levels: int = 2,
    device: Union[str, torch.device] = "cpu",
) -> Tuple[LatentVAE1D, CheckpointInspection]:
    """Safely load one validated VAE checkpoint into a copied public model."""

    path = _explicit_checkpoint_path(checkpoint_path)
    state_dict = _state_dict_from_payload(_safe_load(path))
    inferred = _infer_channels(state_dict)
    selected = inferred if channels is None else int(channels)
    if selected != inferred:
        raise CheckpointError(
            "checkpoint contains {} channels but {} were requested".format(inferred, selected)
        )
    _validate_state_dict(
        state_dict,
        channels=selected,
        latent_channels=latent_channels,
        down_levels=down_levels,
    )
    model = LatentVAE1D(in_ch=selected, z_ch=latent_channels, down_levels=down_levels)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    return model, inspect_vae_checkpoint(
        path,
        channels=selected,
        latent_channels=latent_channels,
        down_levels=down_levels,
    )


def save_vae_checkpoint(model: LatentVAE1D, checkpoint_path: PathLike) -> Path:
    """Save the source-compatible VAE root-key schema to an explicit path."""

    path = _explicit_save_path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({CHECKPOINT_ROOT_KEY: model.state_dict()}, path)
    return path


FLOW_CHECKPOINT_ROOT_KEYS = frozenset({"config", "epoch", "history", "opt", "unet", "val_loss"})
FLOW_CHECKPOINT_LATENT_CHANNELS = 48
FLOW_CHECKPOINT_NUM_CLASSES = 4
FLOW_CHECKPOINT_DEFAULT_WIDTH = 256


@dataclass(frozen=True)
class FlowCheckpointInspection:
    """Safe flow-checkpoint metadata; tensor values are never returned."""

    checkpoint_path: str
    channels: int
    latent_channels: int
    num_classes: int
    model_width: int
    width_profile: str
    state_dict_keys: Tuple[str, ...]
    tensor_shapes: Dict[str, Tuple[int, ...]]
    tensor_dtypes: Dict[str, str]
    root_keys: Tuple[str, ...]

    @property
    def classes(self) -> int:
        return self.num_classes

    @property
    def model_ch(self) -> int:
        return self.model_width

    def to_mapping(self) -> Dict[str, Any]:
        return {
            "checkpoint_path": self.checkpoint_path,
            "channels": self.channels,
            "latent_channels": self.latent_channels,
            "latent_time_steps": 40,
            "num_classes": self.num_classes,
            "classes": self.num_classes,
            "model_width": self.model_width,
            "model_ch": self.model_width,
            "width_profile": self.width_profile,
            "root_keys": list(self.root_keys),
            "state_dict_key_count": len(self.state_dict_keys),
            "state_dict_keys": list(self.state_dict_keys),
            "tensor_shapes": {key: list(shape) for key, shape in self.tensor_shapes.items()},
            "tensor_dtypes": dict(self.tensor_dtypes),
            "payload_values_included": False,
            "exact_paper_reproduction": False,
            "paper_width_conflict_unresolved": True,
        }


FlowCheckpointError = CheckpointError


def _flow_config_int(config: Mapping[str, Any], keys: Tuple[str, ...], label: str) -> Optional[int]:
    for key in keys:
        if key in config:
            try:
                return int(config[key])
            except (TypeError, ValueError) as exc:
                raise CheckpointError("flow checkpoint config {} must be an integer".format(label)) from exc
    return None


def _flow_state_dict_from_payload(payload: Any) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise CheckpointError("flow checkpoint payload must be a mapping")
    actual = set(payload.keys())
    expected_root = set(FLOW_CHECKPOINT_ROOT_KEYS)
    if actual != expected_root:
        raise CheckpointError(
            "flow checkpoint root keys must be exactly "
            "['config', 'epoch', 'history', 'opt', 'unet', 'val_loss'] "
            "(missing={}, extra={})".format(
                sorted(expected_root - actual), sorted(actual - expected_root)
            )
        )
    if not isinstance(payload["config"], Mapping):
        raise CheckpointError("flow checkpoint config must be a mapping")
    if not isinstance(payload["history"], Mapping):
        raise CheckpointError("flow checkpoint history must be a mapping")
    if not isinstance(payload["opt"], Mapping):
        raise CheckpointError("flow checkpoint opt must be a mapping")
    if not isinstance(payload["unet"], Mapping):
        raise CheckpointError("flow checkpoint unet must be a state-dict mapping")
    try:
        if int(payload["epoch"]) < 0:
            raise CheckpointError("flow checkpoint epoch must be non-negative")
        float(payload["val_loss"])
    except (TypeError, ValueError) as exc:
        raise CheckpointError("flow checkpoint epoch and val_loss must be numeric") from exc
    return payload["unet"]


def expected_flow_state_dict_shapes(
    *,
    latent_channels: int = FLOW_CHECKPOINT_LATENT_CHANNELS,
    model_width: int = FLOW_CHECKPOINT_DEFAULT_WIDTH,
    num_classes: int = FLOW_CHECKPOINT_NUM_CLASSES,
    channel_mult: Tuple[int, ...] = (1, 2, 4),
) -> Dict[str, Tuple[int, ...]]:
    """Return the exact state-dict schema emitted by the public flow U-Net."""

    from .models.flow import LatentDiffusionUNet1D

    model = LatentDiffusionUNet1D(
        in_ch=int(latent_channels),
        model_ch=int(model_width),
        channel_mult=channel_mult,
        num_classes=int(num_classes),
    )
    return {key: tuple(value.shape) for key, value in model.state_dict().items()}


def _flow_geometry(state_dict: Mapping[str, Any], config: Mapping[str, Any]) -> Tuple[int, int, int]:
    try:
        input_weight = state_dict["input_proj.weight"]
        label_weight = state_dict["label_embed.weight"]
        output_weight = state_dict["out_proj.weight"]
        state_values = (
            int(input_weight.shape[1]),
            int(input_weight.shape[0]),
            int(label_weight.shape[0]),
        )
        if int(output_weight.shape[0]) != state_values[0]:
            raise CheckpointError("flow input and output latent channel counts do not match")
    except (KeyError, AttributeError, IndexError, TypeError, ValueError) as exc:
        raise CheckpointError("flow checkpoint is missing channel-, width-, or class-bearing tensors") from exc
    config_values = (
        _flow_config_int(config, ("latent_dim", "latent_channels", "latent_dim_channels"), "latent channels"),
        _flow_config_int(config, ("model_ch", "model_width", "base_width", "width"), "model width"),
        _flow_config_int(config, ("num_classes", "classes", "class_count"), "class count"),
    )
    selected = tuple(state if configured is None else configured for state, configured in zip(state_values, config_values))
    if selected != state_values:
        raise CheckpointError("flow checkpoint config does not match unet tensor geometry")
    return state_values


def _flow_channels(config: Mapping[str, Any], requested: Optional[int]) -> int:
    value = _flow_config_int(config, ("channels", "input_channels", "n_channels"), "channels")
    if value is None:
        if requested is None:
            raise CheckpointError("flow checkpoint config must declare paired VAE channels (3 or 6)")
        value = int(requested)
    if value not in SUPPORTED_CHANNELS:
        raise CheckpointError("paired VAE/flow channels must be 3 or 6")
    if requested is not None and value != int(requested):
        raise CheckpointError(
            "flow checkpoint contains {} channels but {} were requested".format(value, int(requested))
        )
    return value


def _flow_profile(config: Mapping[str, Any], width: int) -> str:
    from .models.flow import CUSTOM_PROFILE, DEFAULT_MODEL_WIDTH, HISTORICAL_CHECKPOINT_PROFILE, select_flow_profile

    name = config.get("width_profile")
    if name is None:
        name = HISTORICAL_CHECKPOINT_PROFILE if width == DEFAULT_MODEL_WIDTH else CUSTOM_PROFILE
    try:
        return select_flow_profile(str(name), model_ch=width).name
    except ValueError as exc:
        raise CheckpointError(str(exc)) from exc


def _validate_flow_state_dict(
    state_dict: Mapping[str, Any],
    *,
    latent_channels: int,
    model_width: int,
    num_classes: int,
) -> Dict[str, Tuple[int, ...]]:
    expected = expected_flow_state_dict_shapes(
        latent_channels=latent_channels,
        model_width=model_width,
        num_classes=num_classes,
    )
    actual_keys = set(state_dict.keys())
    expected_keys = set(expected.keys())
    if actual_keys != expected_keys:
        raise CheckpointError(
            "flow checkpoint state-dict keys do not match the U-Net schema "
            "(missing={}, extra={})".format(
                sorted(expected_keys - actual_keys), sorted(actual_keys - expected_keys)
            )
        )
    for key, shape in expected.items():
        value = state_dict[key]
        if not isinstance(value, torch.Tensor):
            raise CheckpointError("flow checkpoint tensor {} is not a tensor".format(key))
        if tuple(value.shape) != shape:
            raise CheckpointError(
                "flow checkpoint tensor {} has shape {}, expected {}".format(key, tuple(value.shape), shape)
            )
    return expected


def inspect_flow_checkpoint(
    checkpoint_path: PathLike,
    *,
    channels: Optional[int] = None,
    vae_channels: Optional[int] = None,
    latent_channels: int = FLOW_CHECKPOINT_LATENT_CHANNELS,
    num_classes: int = FLOW_CHECKPOINT_NUM_CLASSES,
    classes: Optional[int] = None,
    model_width: Optional[int] = None,
    model_ch: Optional[int] = None,
    width_profile: Optional[str] = None,
) -> FlowCheckpointInspection:
    """Safely inspect one explicit six-key flow checkpoint."""

    path = _explicit_checkpoint_path(checkpoint_path)
    payload = _safe_load(path)
    state_dict = _flow_state_dict_from_payload(payload)
    config = payload["config"]
    requested_channels = channels if channels is not None else vae_channels
    paired_channels = _flow_channels(config, requested_channels)
    inferred_latent, inferred_width, inferred_classes = _flow_geometry(state_dict, config)
    requested_latent = int(latent_channels)
    requested_classes = int(num_classes if classes is None else classes)
    if inferred_latent != requested_latent:
        raise CheckpointError("flow checkpoint latent channels {} do not match requested {}".format(inferred_latent, requested_latent))
    if inferred_classes != requested_classes:
        raise CheckpointError("flow checkpoint class count {} does not match requested {}".format(inferred_classes, requested_classes))
    selected_width = inferred_width if model_width is None else int(model_width)
    if model_ch is not None:
        if model_width is not None and int(model_width) != int(model_ch):
            raise CheckpointError("model_width and model_ch must agree")
        selected_width = int(model_ch)
    if selected_width != inferred_width:
        raise CheckpointError("flow checkpoint model width {} does not match requested {}".format(inferred_width, selected_width))
    profile = _flow_profile(config, inferred_width)
    if width_profile is not None:
        from .models.flow import select_flow_profile

        try:
            requested_profile = select_flow_profile(width_profile, model_ch=inferred_width).name
        except ValueError as exc:
            raise CheckpointError(str(exc)) from exc
        if requested_profile != profile:
            raise CheckpointError("flow checkpoint width profile {} does not match requested {}".format(profile, requested_profile))
    expected = _validate_flow_state_dict(
        state_dict,
        latent_channels=inferred_latent,
        model_width=inferred_width,
        num_classes=inferred_classes,
    )
    ordered = tuple(sorted(expected))
    return FlowCheckpointInspection(
        checkpoint_path=str(path),
        channels=paired_channels,
        latent_channels=inferred_latent,
        num_classes=inferred_classes,
        model_width=inferred_width,
        width_profile=profile,
        state_dict_keys=ordered,
        tensor_shapes={key: tuple(state_dict[key].shape) for key in ordered},
        tensor_dtypes={key: str(state_dict[key].dtype) for key in ordered},
        root_keys=tuple(sorted(FLOW_CHECKPOINT_ROOT_KEYS)),
    )


def load_flow_checkpoint(
    checkpoint_path: PathLike,
    *,
    channels: Optional[int] = None,
    vae_channels: Optional[int] = None,
    latent_channels: int = FLOW_CHECKPOINT_LATENT_CHANNELS,
    num_classes: int = FLOW_CHECKPOINT_NUM_CLASSES,
    classes: Optional[int] = None,
    model_width: Optional[int] = None,
    model_ch: Optional[int] = None,
    width_profile: Optional[str] = None,
    device: Union[str, torch.device] = "cpu",
) -> Tuple[Any, FlowCheckpointInspection]:
    """Safely load one validated flow U-Net with no arbitrary-pickle fallback."""

    inspection = inspect_flow_checkpoint(
        checkpoint_path,
        channels=channels,
        vae_channels=vae_channels,
        latent_channels=latent_channels,
        num_classes=num_classes,
        classes=classes,
        model_width=model_width,
        model_ch=model_ch,
        width_profile=width_profile,
    )
    payload = _safe_load(_explicit_checkpoint_path(checkpoint_path))
    state_dict = _flow_state_dict_from_payload(payload)
    from .models.flow import LatentDiffusionUNet1D

    model = LatentDiffusionUNet1D(
        in_ch=inspection.latent_channels,
        model_ch=inspection.model_width,
        num_classes=inspection.num_classes,
    )
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    setattr(model, "flow_channels", inspection.channels)
    setattr(model, "checkpoint_width_profile", inspection.width_profile)
    setattr(model, "checkpoint_path", str(inspection.checkpoint_path))
    return model, inspection


def save_flow_checkpoint(
    model: Any,
    checkpoint_path: PathLike,
    *,
    channels: int,
    epoch: int = 0,
    history: Optional[Mapping[str, Any]] = None,
    optimizer: Any = None,
    val_loss: float = float("nan"),
    config: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Save the exact six-key flow schema to an explicitly named path."""

    if int(channels) not in SUPPORTED_CHANNELS:
        raise CheckpointError("paired VAE/flow channels must be 3 or 6")
    path = _explicit_save_path(checkpoint_path)
    model_width = int(getattr(model, "model_ch", FLOW_CHECKPOINT_DEFAULT_WIDTH))
    latent_channels = int(getattr(model, "in_ch", FLOW_CHECKPOINT_LATENT_CHANNELS))
    num_classes = int(getattr(model, "num_classes", FLOW_CHECKPOINT_NUM_CLASSES))
    payload_config = dict(config or {})
    payload_config.setdefault("channels", int(channels))
    payload_config.setdefault("latent_dim", latent_channels)
    payload_config.setdefault("model_ch", model_width)
    payload_config.setdefault("num_classes", num_classes)
    payload_config.setdefault("width_profile", getattr(model, "checkpoint_width_profile", None))
    payload_config = {key: value for key, value in payload_config.items() if value is not None}
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": payload_config,
            "epoch": int(epoch),
            "history": dict(history or {}),
            "opt": {} if optimizer is None else optimizer.state_dict(),
            "unet": model.state_dict(),
            "val_loss": float(val_loss),
        },
        path,
    )
    return path


__all__ = [
    "CHECKPOINT_INPUT_LENGTH",
    "CHECKPOINT_LATENT_TIME_STEPS",
    "CHECKPOINT_ROOT_KEY",
    "CheckpointError",
    "CheckpointInspection",
    "FLOW_CHECKPOINT_DEFAULT_WIDTH",
    "FLOW_CHECKPOINT_LATENT_CHANNELS",
    "FLOW_CHECKPOINT_NUM_CLASSES",
    "FLOW_CHECKPOINT_ROOT_KEYS",
    "FlowCheckpointError",
    "FlowCheckpointInspection",
    "expected_state_dict_shapes",
    "expected_flow_state_dict_shapes",
    "inspect_vae_checkpoint",
    "inspect_flow_checkpoint",
    "load_vae_checkpoint",
    "load_flow_checkpoint",
    "save_vae_checkpoint",
    "save_flow_checkpoint",
]
