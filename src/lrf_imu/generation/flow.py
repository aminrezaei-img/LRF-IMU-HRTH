"""Paper sampling and website trajectory profiles for the public flow API."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from ..training.flow import (
    MODEL_TIME_SCALE,
    PAPER_NUM_STEPS,
    PAPER_SEED,
    reverse_euler_step,
)


SAMPLING_RATE_HZ = 50.0
NATIVE_WINDOW_SAMPLES = 160
PAPER_SAMPLES_PER_CLASS = 500
WEBSITE_NUM_STEPS = 100
WEBSITE_RECORD_EVERY = 2
WEBSITE_OVERLAP_SAMPLES = 40
WEBSITE_DURATION_SECONDS = 10.0
WEBSITE_SCHEMA_VERSION = "m3c.website-trajectory.1"
PAPER_SCHEMA_VERSION = "m3c.paper-samples.1"


class FlowGenerationError(ValueError):
    """Raised when generation inputs or profile boundaries are invalid."""


@dataclass(frozen=True)
class EulerTrajectory:
    """Recorded reverse-Euler states without hidden tensor serialization."""

    states: Tuple[torch.Tensor, ...]
    stored_steps: Tuple[int, ...]
    flow_times: Tuple[float, ...]

    @property
    def state_count(self) -> int:
        return len(self.states)

    def to_mapping(self) -> Dict[str, Any]:
        shape = list(self.states[0].shape) if self.states else None
        return {
            "state_count": self.state_count,
            "stored_steps": list(self.stored_steps),
            "flow_times": list(self.flow_times),
            "state_shape": shape,
            "tensor_values_included": False,
        }


def seed_everything(seed: int) -> None:
    """Set deterministic CPU/GPU RNG streams for an explicit generation seed."""

    value = int(seed)
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def class_labels(
    *,
    num_classes: int = 4,
    samples_per_class: int = PAPER_SAMPLES_PER_CLASS,
    device: Optional[Union[str, torch.device]] = None,
) -> torch.Tensor:
    """Return paper/TSTR label order: all 500 windows for class 0, then class 1, etc."""

    classes = int(num_classes)
    count = int(samples_per_class)
    if classes < 1:
        raise FlowGenerationError("num_classes must be positive")
    if count < 1:
        raise FlowGenerationError("samples_per_class must be positive")
    return torch.arange(classes, device=device, dtype=torch.long).repeat_interleave(count)


def paper_sampling_metadata(
    *,
    samples_per_class: int = PAPER_SAMPLES_PER_CLASS,
    num_steps: int = PAPER_NUM_STEPS,
    seed: int = PAPER_SEED,
) -> Dict[str, Any]:
    """Describe paper sampling without claiming exact paper reproduction."""

    if int(num_steps) < 1:
        raise FlowGenerationError("num_steps must be positive")
    return {
        "profile": "paper",
        "solver": "explicit_reverse_euler",
        "num_steps": int(num_steps),
        "seed": int(seed),
        "samples_per_class": int(samples_per_class),
        "latent_shape": ["B", 48, 40],
        "start_time": 1.0,
        "end_time": 0.0,
        "website_trajectory": False,
        "exact_paper_reproduction": False,
        "paper_width_conflict_unresolved": True,
    }


def _selected_device(model: Any, device: Optional[Union[str, torch.device]]) -> torch.device:
    if device is not None:
        return torch.device(device)
    try:
        return next(model.parameters()).device
    except (AttributeError, StopIteration):
        return torch.device("cpu")


def _flow_model(flow_or_model: Any) -> Any:
    return getattr(flow_or_model, "unet", flow_or_model)


def _generator_for(device: torch.device, seed: Optional[int]) -> Optional[torch.Generator]:
    if seed is None:
        return None
    try:
        generator = torch.Generator(device=device.type)
    except (RuntimeError, TypeError):
        generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator


@torch.no_grad()
def sample_reverse_euler_trajectory(
    unet: Any,
    labels: torch.Tensor,
    latent_shape: Sequence[int],
    *,
    num_steps: int = PAPER_NUM_STEPS,
    seed: Optional[int] = None,
    noise: Optional[torch.Tensor] = None,
    noise_scale: float = 1.0,
    record_every: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> EulerTrajectory:
    """Integrate independently seeded latent windows and optionally record states."""

    model = _flow_model(unet)
    target_device = _selected_device(model, device)
    class_ids = torch.as_tensor(labels, device=target_device, dtype=torch.long).reshape(-1)
    if class_ids.numel() <= 0:
        raise FlowGenerationError("at least one label is required")
    shape = tuple(int(value) for value in latent_shape)
    if len(shape) != 2 or any(value <= 0 for value in shape):
        raise FlowGenerationError("latent_shape must be [latent_channels, latent_time_steps]")
    steps = int(num_steps)
    if steps < 1:
        raise FlowGenerationError("num_steps must be >= 1")
    interval = None if record_every is None else int(record_every)
    if interval is not None and interval < 1:
        raise FlowGenerationError("record_every must be >= 1")
    scale = float(noise_scale)
    if not np.isfinite(scale) or scale <= 0.0:
        raise FlowGenerationError("noise_scale must be positive and finite")

    generator = _generator_for(target_device, seed)
    expected_shape = (int(class_ids.shape[0]), shape[0], shape[1])
    if noise is None:
        state = torch.randn(expected_shape, device=target_device, generator=generator) * scale
    else:
        state = noise.to(target_device)
        if tuple(state.shape) != expected_shape:
            raise FlowGenerationError("noise shape does not match labels and latent_shape")

    stored_steps: List[int] = [0]
    flow_times: List[float] = [1.0]
    states: List[torch.Tensor] = [state.clone()]
    dt = 1.0 / float(steps)
    for index in range(steps):
        current_time = 1.0 - float(index) * dt
        times = torch.full(
            (class_ids.shape[0],),
            current_time,
            device=target_device,
            dtype=state.dtype,
        )
        velocity = model(state, times * MODEL_TIME_SCALE, class_ids)
        state = reverse_euler_step(state, velocity, dt)
        completed = index + 1
        if interval is None or completed % interval == 0 or completed == steps:
            stored_steps.append(completed)
            flow_times.append(max(0.0, 1.0 - completed * dt))
            states.append(state.clone())
    return EulerTrajectory(tuple(states), tuple(stored_steps), tuple(flow_times))


@torch.no_grad()
def sample_reverse_euler(
    unet: Any,
    labels: torch.Tensor,
    latent_shape: Sequence[int] = (48, 40),
    *,
    num_steps: int = PAPER_NUM_STEPS,
    seed: Optional[int] = PAPER_SEED,
    noise: Optional[torch.Tensor] = None,
    noise_scale: float = 1.0,
    device: Optional[Union[str, torch.device]] = None,
) -> torch.Tensor:
    """Return the final state of the explicit reverse-Euler paper sampler."""

    trajectory = sample_reverse_euler_trajectory(
        unet,
        labels,
        latent_shape,
        num_steps=num_steps,
        seed=seed,
        noise=noise,
        noise_scale=noise_scale,
        device=device,
    )
    return trajectory.states[-1]


@torch.no_grad()
def generate_paper_latents(
    flow_or_model: Any,
    *,
    samples_per_class: int = PAPER_SAMPLES_PER_CLASS,
    num_steps: int = PAPER_NUM_STEPS,
    seed: int = PAPER_SEED,
    latent_shape: Optional[Sequence[int]] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """Generate paper/TSTR latent windows with 500/class and seed 42 defaults."""

    target_device = _selected_device(_flow_model(flow_or_model), device)
    labels = class_labels(
        samples_per_class=samples_per_class,
        device=target_device,
    )
    if latent_shape is None:
        latent_shape = (48, 40)
        if hasattr(flow_or_model, "encode"):
            channels = int(
                getattr(
                    getattr(flow_or_model, "vae", None),
                    "input_channels",
                    getattr(getattr(flow_or_model, "vae", None), "in_ch", 6),
                )
            )
            dummy = torch.zeros(1, channels, NATIVE_WINDOW_SAMPLES, device=target_device)
            latent_shape = tuple(flow_or_model.encode(dummy).shape[1:])
    latents = sample_reverse_euler(
        _flow_model(flow_or_model),
        labels,
        latent_shape,
        num_steps=num_steps,
        seed=seed,
        device=target_device,
    )
    return latents, labels, paper_sampling_metadata(
        samples_per_class=samples_per_class,
        num_steps=num_steps,
        seed=seed,
    )


generate_samples = generate_paper_latents


def website_seed(base_seed: int, subject_id: int, activity_id: int) -> int:
    """Apply the source website seed formula exactly."""

    return int(base_seed) + int(subject_id) * 1000 + int(activity_id) * 100


trajectory_seed = website_seed


def required_segment_count(
    target_samples: int,
    native_samples: int = NATIVE_WINDOW_SAMPLES,
    overlap_samples: int = WEBSITE_OVERLAP_SAMPLES,
) -> int:
    """Return the independent native-window count needed for one website signal."""

    target = int(target_samples)
    native = int(native_samples)
    overlap = int(overlap_samples)
    if target <= 0 or native <= 0:
        raise FlowGenerationError("target and native sample counts must be positive")
    if target <= native:
        return 1
    if overlap < 0 or overlap >= native:
        raise FlowGenerationError("overlap_samples must be in [0,native_samples)")
    hop = native - overlap
    return 1 + int(math.ceil((target - native) / float(hop)))


def overlap_add_windows(
    windows: np.ndarray,
    target_samples: int,
    overlap_samples: int = WEBSITE_OVERLAP_SAMPLES,
) -> np.ndarray:
    """Join ``[segments,channels,native]`` windows with linear overlap-add."""

    array = np.asarray(windows)
    if array.ndim != 3:
        raise FlowGenerationError(
            "windows must have shape [segments, channels, native_samples]"
        )
    n_segments, n_channels, native_samples = array.shape
    target = int(target_samples)
    overlap = int(overlap_samples)
    if target <= 0:
        raise FlowGenerationError("target_samples must be positive")
    if n_segments == 1:
        if target > native_samples:
            raise FlowGenerationError("one native window cannot fill the requested duration")
        return array[0, :, :target].astype(np.float32, copy=False)
    if overlap < 0 or overlap >= native_samples:
        raise FlowGenerationError(
            "overlap_samples must be in [0, {})".format(native_samples)
        )
    hop = native_samples - overlap
    total_samples = native_samples + (n_segments - 1) * hop
    if target > total_samples:
        raise FlowGenerationError("native windows do not cover target_samples")
    output = np.zeros((n_channels, total_samples), dtype=np.float64)
    weights = np.zeros(total_samples, dtype=np.float64)
    for segment_index in range(n_segments):
        start = segment_index * hop
        end = start + native_samples
        segment_weight = np.ones(native_samples, dtype=np.float64)
        if overlap > 0 and segment_index > 0:
            segment_weight[:overlap] = np.linspace(
                0.0, 1.0, overlap, endpoint=False, dtype=np.float64
            )
        if overlap > 0 and segment_index < n_segments - 1:
            segment_weight[-overlap:] = np.linspace(
                1.0, 0.0, overlap, endpoint=False, dtype=np.float64
            )
        output[:, start:end] += array[segment_index].astype(np.float64) * segment_weight[None, :]
        weights[start:end] += segment_weight
    output /= np.maximum(weights, 1e-12)[None, :]
    return output[:, :target].astype(np.float32)


linear_overlap_add = overlap_add_windows


def _decode_native_states(
    vae: Any,
    states: Iterable[torch.Tensor],
    *,
    standardizer: Any = None,
) -> np.ndarray:
    decoded_states = []
    with torch.no_grad():
        for state in states:
            decoded = vae.decode(state)
            values = decoded.detach().cpu().numpy()
            if standardizer is not None:
                values = standardizer.inverse(values)
            decoded_states.append(np.asarray(values, dtype=np.float32))
    return np.stack(decoded_states, axis=0)


def website_sampling_metadata(
    *,
    base_seed: int,
    subject_id: int,
    activity_id: int,
    num_steps: int = WEBSITE_NUM_STEPS,
    record_every: int = WEBSITE_RECORD_EVERY,
    duration_seconds: float = WEBSITE_DURATION_SECONDS,
    overlap_samples: int = WEBSITE_OVERLAP_SAMPLES,
    native_window_samples: int = NATIVE_WINDOW_SAMPLES,
) -> Dict[str, Any]:
    """Return the website profile separately from paper/TSTR sampling metadata."""

    steps = int(num_steps)
    interval = int(record_every)
    if steps < 1 or interval < 1:
        raise FlowGenerationError("website num_steps and record_every must be positive")
    target_samples = int(round(float(duration_seconds) * SAMPLING_RATE_HZ))
    segments = required_segment_count(target_samples, native_window_samples, overlap_samples)
    stored_steps = list(range(0, steps + 1, interval))
    if stored_steps[-1] != steps:
        stored_steps.append(steps)
    return {
        "profile": "website_trajectory",
        "solver": "explicit_reverse_euler",
        "num_steps": steps,
        "record_every": interval,
        "state_count": len(stored_steps),
        "stored_steps": stored_steps,
        "duration_seconds": float(duration_seconds),
        "target_samples": target_samples,
        "native_window_samples": int(native_window_samples),
        "native_segment_count": segments,
        "overlap_samples": int(overlap_samples),
        "base_seed": int(base_seed),
        "subject_id": int(subject_id),
        "activity_id": int(activity_id),
        "seed": website_seed(base_seed, subject_id, activity_id),
        "seed_formula": "base_seed + subject_id * 1000 + activity_id * 100",
        "paper_sampling_steps": PAPER_NUM_STEPS,
        "paper_samples_per_class": PAPER_SAMPLES_PER_CLASS,
        "paper_tstr_separation": True,
        "exact_paper_reproduction": False,
    }


def build_website_trajectory_payload(
    native_state_windows: np.ndarray,
    *,
    subject_id: int,
    activity_id: int,
    base_seed: int = PAPER_SEED,
    duration_seconds: float = WEBSITE_DURATION_SECONDS,
    overlap_samples: int = WEBSITE_OVERLAP_SAMPLES,
    num_steps: int = WEBSITE_NUM_STEPS,
    record_every: int = WEBSITE_RECORD_EVERY,
    json_precision: int = 5,
    activity_name: Optional[str] = None,
    vae_checkpoint_name: Optional[str] = None,
    flow_checkpoint_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Build JSON-safe website trajectory metadata and signals from native windows."""

    native = np.asarray(native_state_windows)
    if native.ndim != 4:
        raise FlowGenerationError(
            "native_state_windows must have shape [states, segments, channels, samples]"
        )
    metadata = website_sampling_metadata(
        base_seed=base_seed,
        subject_id=subject_id,
        activity_id=activity_id,
        num_steps=num_steps,
        record_every=record_every,
        duration_seconds=duration_seconds,
        overlap_samples=overlap_samples,
        native_window_samples=native.shape[-1],
    )
    expected_states = len(metadata["stored_steps"])
    if native.shape[0] != expected_states:
        raise FlowGenerationError(
            "native_state_windows contains {} states; website profile records {}".format(
                native.shape[0], expected_states
            )
        )
    signals = np.stack(
        [
            overlap_add_windows(
                native[index],
                metadata["target_samples"],
                overlap_samples,
            )
            for index in range(native.shape[0])
        ],
        axis=0,
    )
    activity_label = activity_name or "class_{}".format(int(activity_id))
    payload = {
        "schema_version": WEBSITE_SCHEMA_VERSION,
        "profile": "website_trajectory",
        "subject": "subject_{:02d}".format(int(subject_id)),
        "activity": {
            "id": int(activity_id),
            "name": activity_label,
        },
        "signal": {
            "sampling_rate_hz": SAMPLING_RATE_HZ,
            "samples": int(metadata["target_samples"]),
            "duration_seconds": float(duration_seconds),
            "native_window_samples": int(native.shape[-1]),
            "native_segment_count": int(native.shape[1]),
            "overlap_samples": int(overlap_samples),
            "construction": "overlap_add_of_independent_native_windows",
        },
        "generation": metadata,
        "provenance": {
            "paper_tstr_samples": False,
            "vae_checkpoint_name": vae_checkpoint_name,
            "flow_checkpoint_name": flow_checkpoint_name,
        },
        "stored_steps": list(metadata["stored_steps"]),
        "flow_times": [
            round(max(0.0, 1.0 - float(step) / float(num_steps)), int(json_precision))
            for step in metadata["stored_steps"]
        ],
        "signals": np.round(
            signals.astype(np.float64), decimals=int(json_precision)
        ).tolist(),
    }
    return payload


@torch.no_grad()
def generate_website_trajectory(
    flow_or_model: Any,
    vae: Any,
    *,
    subject_id: int,
    activity_id: int,
    base_seed: int = PAPER_SEED,
    duration_seconds: float = WEBSITE_DURATION_SECONDS,
    overlap_samples: int = WEBSITE_OVERLAP_SAMPLES,
    num_steps: int = WEBSITE_NUM_STEPS,
    record_every: int = WEBSITE_RECORD_EVERY,
    noise_scale: float = 1.0,
    standardizer: Any = None,
    json_precision: int = 5,
    activity_name: Optional[str] = None,
    vae_checkpoint_name: Optional[str] = None,
    flow_checkpoint_name: Optional[str] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> Dict[str, Any]:
    """Generate independent native windows and return the website-only payload."""

    model = _flow_model(flow_or_model)
    target_device = _selected_device(model, device)
    channels = int(
        getattr(vae, "input_channels", getattr(vae, "in_ch", 6))
    )
    flow_channels = getattr(model, "flow_channels", None)
    if flow_channels is not None and int(flow_channels) != channels:
        raise FlowGenerationError(
            "paired VAE/flow channel mismatch: VAE has {}, flow has {}".format(
                channels, int(flow_channels)
            )
        )
    dummy = torch.zeros(1, channels, NATIVE_WINDOW_SAMPLES, device=target_device)
    if hasattr(flow_or_model, "encode"):
        latent_shape = tuple(flow_or_model.encode(dummy).shape[1:])
    else:
        encoded = vae.encode(dummy)
        latent_shape = tuple((encoded[0] if isinstance(encoded, (tuple, list)) else encoded).shape[1:])
    target_samples = int(round(float(duration_seconds) * SAMPLING_RATE_HZ))
    segments = required_segment_count(
        target_samples,
        NATIVE_WINDOW_SAMPLES,
        overlap_samples,
    )
    computed_seed = website_seed(base_seed, subject_id, activity_id)
    labels = torch.full(
        (segments,),
        int(activity_id),
        dtype=torch.long,
        device=target_device,
    )
    trajectory = sample_reverse_euler_trajectory(
        model,
        labels,
        latent_shape,
        num_steps=num_steps,
        seed=computed_seed,
        noise_scale=noise_scale,
        record_every=record_every,
        device=target_device,
    )
    native = _decode_native_states(vae, trajectory.states, standardizer=standardizer)
    return build_website_trajectory_payload(
        native,
        subject_id=subject_id,
        activity_id=activity_id,
        base_seed=base_seed,
        duration_seconds=duration_seconds,
        overlap_samples=overlap_samples,
        num_steps=num_steps,
        record_every=record_every,
        json_precision=json_precision,
        activity_name=activity_name,
        vae_checkpoint_name=vae_checkpoint_name,
        flow_checkpoint_name=flow_checkpoint_name,
    )


def write_trajectory_payload(
    payload: Mapping[str, Any],
    output_path: Union[str, Path],
    *,
    overwrite: bool = False,
) -> Path:
    """Write one explicit website JSON path; no implicit result roots are used."""

    destination = Path(output_path).expanduser().resolve()
    if destination.exists() and destination.is_dir():
        raise FlowGenerationError("trajectory output must be a file path")
    if destination.exists() and not overwrite:
        raise FileExistsError("output exists and overwrite is false: {}".format(destination))
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(dict(payload), ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    return destination


export_website_trajectories = write_trajectory_payload


__all__ = [
    "EulerTrajectory",
    "FlowGenerationError",
    "NATIVE_WINDOW_SAMPLES",
    "PAPER_NUM_STEPS",
    "PAPER_SAMPLES_PER_CLASS",
    "PAPER_SCHEMA_VERSION",
    "PAPER_SEED",
    "SAMPLING_RATE_HZ",
    "WEBSITE_DURATION_SECONDS",
    "WEBSITE_NUM_STEPS",
    "WEBSITE_OVERLAP_SAMPLES",
    "WEBSITE_RECORD_EVERY",
    "WEBSITE_SCHEMA_VERSION",
    "build_website_trajectory_payload",
    "class_labels",
    "export_website_trajectories",
    "generate_paper_latents",
    "generate_samples",
    "generate_website_trajectory",
    "linear_overlap_add",
    "overlap_add_windows",
    "paper_sampling_metadata",
    "required_segment_count",
    "sample_reverse_euler",
    "sample_reverse_euler_trajectory",
    "seed_everything",
    "trajectory_seed",
    "website_sampling_metadata",
    "website_seed",
    "write_trajectory_payload",
]
