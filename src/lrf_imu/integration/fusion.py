"""Exact-duration, provenance-preserving DayForge signal fusion primitives.

This module intentionally exposes a small generator protocol.  Real VAE/Flow
inference can be plugged in later; unit tests use a deterministic fake generator.
Intervals are half-open: [start_time, end_time).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .physical_state import CLASS_NAMES

SAMPLE_RATE_HZ = 50
WINDOW_SAMPLES = 160
LATENT_CHANNELS = 48
LATENT_TIME = 40
NUM_CLASSES = 10
CHANNEL_NAMES = ("thigh_x", "thigh_y", "thigh_z")


class FusionError(ValueError):
    pass


@dataclass(frozen=True)
class StitchConfig:
    overlap_samples: int = 40
    method: str = "linear_crossfade"

    def __post_init__(self):
        if self.method not in {"linear_crossfade", "cosine_crossfade"}:
            raise FusionError("unsupported stitch method")
        if not 0 <= self.overlap_samples < WINDOW_SAMPLES:
            raise FusionError("overlap_samples must be in [0, 160)")


@dataclass(frozen=True)
class SegmentResult:
    record: dict[str, Any]
    signal: np.ndarray
    provenance: dict[str, Any]
    stitch_audit: dict[str, Any]


def target_samples(duration_seconds: Any, sample_rate: int = SAMPLE_RATE_HZ) -> int:
    value = float(duration_seconds)
    if not np.isfinite(value) or value < 0:
        raise FusionError("duration must be finite and non-negative")
    return int(np.floor(value * sample_rate + 0.5))


def stable_seed(
    global_seed: int,
    persona: Any,
    date: Any,
    interval: Any,
    class_id: int,
    window_index: int | None = None,
) -> int:
    parts = [str(global_seed), str(persona), str(date), str(interval), str(class_id)]
    if window_index is not None:
        parts.append(str(window_index))
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


def _fade(k: int, method: str) -> tuple[np.ndarray, np.ndarray]:
    if k == 0:
        return np.empty(0), np.empty(0)
    x = np.linspace(0.0, 1.0, k, endpoint=True)
    if method == "cosine_crossfade":
        incoming = (1.0 - np.cos(np.pi * x)) / 2.0
    else:
        incoming = x
    return 1.0 - incoming, incoming


def stitch_windows(
    windows: Sequence[np.ndarray], target: int, config: StitchConfig = StitchConfig()
) -> tuple[np.ndarray, list[int]]:
    """Crossfade independently generated [3,160] windows and return exact [3,target]."""
    if target < 0:
        raise FusionError("target must be non-negative")
    if target == 0:
        return np.empty((0, 3), dtype=np.float32), []
    if not windows:
        raise FusionError("at least one window is required")
    clean = []
    for window in windows:
        a = np.asarray(window, dtype=np.float32)
        if a.shape != (3, WINDOW_SAMPLES):
            raise FusionError("generator must return [3, 160]")
        if not np.isfinite(a).all():
            raise FusionError("generator returned NaN or Inf")
        clean.append(a)
    assembled = clean[0].T.copy()
    boundaries = []
    for index, nxt in enumerate(clean[1:], 1):
        overlap = min(config.overlap_samples, assembled.shape[0], WINDOW_SAMPLES - 1)
        if overlap:
            out_w, in_w = _fade(overlap, config.method)
            assembled[-overlap:] = (
                assembled[-overlap:] * out_w[:, None]
                + nxt[:, :overlap].T * in_w[:, None]
            )
            assembled = np.concatenate([assembled, nxt[:, overlap:].T], axis=0)
            boundaries.append(int(assembled.shape[0] - WINDOW_SAMPLES + overlap))
        else:
            boundaries.append(int(assembled.shape[0]))
            assembled = np.concatenate([assembled, nxt.T], axis=0)
    return assembled[:target].astype(np.float32, copy=False), boundaries


def _duration_from_times(record: Mapping[str, Any]) -> float:
    start, end = (
        record.get("start_time", record.get("start")),
        record.get("end_time", record.get("end")),
    )
    try:
        if isinstance(start, str) and isinstance(end, str):
            return (
                datetime.fromisoformat(end.replace("Z", "+00:00"))
                - datetime.fromisoformat(start.replace("Z", "+00:00"))
            ).total_seconds()
        return float(
            record["duration_seconds"]
            if "duration_seconds" in record
            else record["duration"]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise FusionError("interval has invalid start/end or duration") from exc


def generate_segment(
    record: Mapping[str, Any],
    generator: Callable[..., np.ndarray],
    *,
    global_seed: int = 42,
    vae_checkpoint: str | None = None,
    flow_checkpoint: str | None = None,
    normalization_metadata: str | None = None,
    stitch: StitchConfig = StitchConfig(),
) -> SegmentResult | dict[str, Any]:
    """Generate one mapped interval; unsupported intervals are metadata-only."""
    source = dict(record)
    n = target_samples(_duration_from_times(source))
    eligible = bool(source.get("imu_eligible", False))
    base = {
        "persona_id": source.get("persona_id"),
        "date": source.get("date"),
        "resolved_interval_id": source.get("resolved_interval_id"),
        "source_episode_id": source.get("source_episode_id"),
        "start_time": source.get("start_time"),
        "end_time": source.get("end_time"),
        "duration_seconds": _duration_from_times(source),
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "sample_count": n,
        "semantic_activity": source.get("semantic_activity"),
        "physical_state_class_id": source.get("physical_state_class_id"),
        "physical_state_class_name": source.get("physical_state_class_name"),
        "mobility_mode": source.get("mobility_mode"),
        "imu_available": False,
        "imu_unavailable_reason": source.get("imu_unavailable_reason"),
        "status": "IMU_UNAVAILABLE",
    }
    if not eligible:
        return base
    cls = source.get("physical_state_class_id")
    if not isinstance(cls, int) or not 0 <= cls < NUM_CLASSES:
        raise FusionError("eligible interval has invalid class ID")
    interval_seed = stable_seed(
        global_seed, base["persona_id"], base["date"], base["resolved_interval_id"], cls
    )
    count = max(1, int(np.ceil(max(n, 1) / (WINDOW_SAMPLES - stitch.overlap_samples))))
    if n < WINDOW_SAMPLES:
        count = 1
    windows = []
    seeds = []
    try:
        for index in range(count):
            seed = stable_seed(
                interval_seed,
                base["persona_id"],
                base["date"],
                base["resolved_interval_id"],
                cls,
                index,
            )
            seeds.append(seed)
            try:
                window = generator(
                    class_id=cls,
                    seed=seed,
                    window_index=index,
                    window_length=WINDOW_SAMPLES,
                )
            except TypeError:
                window = generator(cls, seed, index)
            windows.append(window)
        signal, boundaries = stitch_windows(windows, n, stitch)
    except Exception as exc:
        return {
            **base,
            "status": "IMU_GENERATION_FAILED",
            "imu_unavailable_reason": None,
            "failure_type": type(exc).__name__,
            "failure_message": str(exc),
            "generation_seed": interval_seed,
        }
    diffs = np.abs(np.diff(signal, axis=0)) if len(signal) > 1 else np.empty((0, 3))
    jumps = [
        diffs[b - 1 : b + 1].max(axis=0).tolist()
        for b in boundaries
        if 0 < b < len(signal)
    ]
    provenance = {
        "mapping_provenance": source.get("mapping_provenance"),
        "mapping_rule": source.get("mapping_rule"),
        "mapping_version": source.get("mapping_version"),
        "vae_checkpoint": vae_checkpoint,
        "flow_checkpoint": flow_checkpoint,
        "normalization_metadata": normalization_metadata,
        "global_seed": global_seed,
        "interval_seed": interval_seed,
        "window_seeds": seeds,
        "stitch_method": stitch.method,
        "overlap_samples": stitch.overlap_samples,
        "interval_convention": "[start,end)",
        "software_revision": None,
    }
    base.update(
        {
            "imu_available": True,
            "status": "GENERATED",
            "imu_unavailable_reason": None,
            "segment_id": f"{base['persona_id']}:{base['date']}:{base['resolved_interval_id']}",
            "generated_windows": count,
            "cropped_samples": max(0, WINDOW_SAMPLES - n) if n < WINDOW_SAMPLES else 0,
        }
    )
    return SegmentResult(
        base,
        signal,
        provenance,
        {
            "boundary_count": len(boundaries),
            "stitch_jumps": jumps,
            "ordinary_jump_mean": float(diffs.mean()) if diffs.size else 0.0,
        },
    )


def segment_timestamps(record: Mapping[str, Any]) -> np.ndarray:
    """Return integer tick timestamps relative to the owning interval."""
    n = int(record["sample_count"])
    return np.arange(n, dtype=np.int64)


def validate_segment_timeline(result: SegmentResult) -> None:
    if result.signal.shape != (result.record["sample_count"], 3):
        raise FusionError("sample count/channel shape mismatch")
    if not np.isfinite(result.signal).all():
        raise FusionError("generated segment contains NaN/Inf")
    ticks = segment_timestamps(result.record)
    if len(ticks) and (ticks[0] != 0 or ticks[-1] >= result.record["sample_count"]):
        raise FusionError("segment timestamp bounds invalid")


def validate_checkpoint_contract(
    vae_metadata: Mapping[str, Any],
    flow_metadata: Mapping[str, Any],
    normalization_metadata: Mapping[str, Any],
) -> None:
    checks = {
        "VAE input channels": (vae_metadata.get("channels"), 3),
        "VAE window length": (vae_metadata.get("input_length"), 160),
        "VAE latent channels": (vae_metadata.get("latent_channels"), 48),
        "VAE latent time": (vae_metadata.get("latent_time_steps"), 40),
        "Flow classes": (flow_metadata.get("num_classes"), 10),
        "Flow latent channels": (flow_metadata.get("latent_channels"), 48),
        "Flow latent time": (flow_metadata.get("latent_time_steps"), 40),
    }
    for name, (actual, expected) in checks.items():
        if actual != expected:
            raise FusionError(f"{name} incompatible: expected {expected}, got {actual}")
    if not normalization_metadata or not all(
        k in normalization_metadata for k in ("mean", "std")
    ):
        raise FusionError("normalization metadata with mean and std is required")


def audit_segments(
    results: Sequence[SegmentResult | Mapping[str, Any]],
) -> dict[str, Any]:
    generated = [x for x in results if isinstance(x, SegmentResult)]
    unavailable = [x for x in results if not isinstance(x, SegmentResult)]

    def duration(item: SegmentResult | Mapping[str, Any]) -> float:
        if isinstance(item, SegmentResult):
            return float(item.record.get("duration_seconds", 0))
        return float(item.get("duration_seconds", 0))

    return {
        "intervals": len(results),
        "generated_intervals": len(generated),
        "unsupported_intervals": len(unavailable),
        "generated_duration_seconds": sum(duration(x) for x in generated),
        "unsupported_duration_seconds": sum(duration(x) for x in unavailable),
        "generated_samples": sum(int(x.record["sample_count"]) for x in generated),
        "generation_failures": sum(
            1
            for x in results
            if (x.record if isinstance(x, SegmentResult) else x).get("status")
            == "IMU_GENERATION_FAILED"
        ),
        "per_class_duration_seconds": {
            name: sum(
                duration(x)
                for x in generated
                if x.record.get("physical_state_class_name") == name
            )
            for name in CLASS_NAMES
        },
    }


__all__ = [
    "CHANNEL_NAMES",
    "FusionError",
    "LATENT_CHANNELS",
    "LATENT_TIME",
    "NUM_CLASSES",
    "SAMPLE_RATE_HZ",
    "StitchConfig",
    "SegmentResult",
    "audit_segments",
    "generate_segment",
    "stable_seed",
    "stitch_windows",
    "target_samples",
    "validate_checkpoint_contract",
    "validate_segment_timeline",
    "segment_timestamps",
]
