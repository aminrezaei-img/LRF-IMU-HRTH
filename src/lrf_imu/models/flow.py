"""Torch-backed Rectified Flow U-Net compatibility surface.

The implementation keeps the layer names and ordering of the operational
``models/unet_1d.py`` source so that observed ``flow_unet_best.pt`` files can
be validated and loaded without copying the original repository into the
public package.  The public default is the observed 256-wide checkpoint
profile; the manuscript-reported 128-wide profile is available only when
selected explicitly because the two pieces of evidence conflict.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch
from torch import nn
import torch.nn.functional as F


LATENT_CHANNELS = 48
NUM_CLASSES = 4
DEFAULT_MODEL_WIDTH = 256
MANUSCRIPT_MODEL_WIDTH = 128
DEFAULT_CHANNEL_MULT = (1, 2, 4)
NORMALIZATION_GROUPS = 8
SHORT_KERNEL = 3
LONG_KERNEL = 31
SE_REDUCTION = 4

HISTORICAL_CHECKPOINT_PROFILE = "historical_checkpoint_compatibility_256"
MANUSCRIPT_REPORTED_PROFILE = "manuscript_reported_128"
CUSTOM_PROFILE = "custom"
EXACT_PAPER_REPRODUCTION = False


class FlowModelError(ValueError):
    """Raised when a flow model configuration is incompatible with the contract."""


@dataclass(frozen=True)
class FlowWidthProfile:
    """Named, explicit width evidence for the public flow model."""

    name: str
    model_ch: int
    evidence: str
    checkpoint_compatible: bool
    exact_paper_reproduction: bool = EXACT_PAPER_REPRODUCTION

    @property
    def width(self) -> int:
        """Compatibility alias for callers that call the width ``width``."""

        return self.model_ch

    @property
    def base_width(self) -> int:
        """Compatibility alias for configuration objects."""

        return self.model_ch

    def to_mapping(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "model_ch": self.model_ch,
            "width": self.model_ch,
            "evidence": self.evidence,
            "checkpoint_compatible": self.checkpoint_compatible,
            "manuscript_width_conflict": True,
            "exact_paper_reproduction": self.exact_paper_reproduction,
        }


_PROFILE_ALIASES = {
    HISTORICAL_CHECKPOINT_PROFILE: HISTORICAL_CHECKPOINT_PROFILE,
    "historical_checkpoint_compatibility": HISTORICAL_CHECKPOINT_PROFILE,
    "observed_wrapper_compatibility": HISTORICAL_CHECKPOINT_PROFILE,
    "observed_checkpoint_compatibility": HISTORICAL_CHECKPOINT_PROFILE,
    MANUSCRIPT_REPORTED_PROFILE: MANUSCRIPT_REPORTED_PROFILE,
    "manuscript_reported": MANUSCRIPT_REPORTED_PROFILE,
    "manuscript": MANUSCRIPT_REPORTED_PROFILE,
    CUSTOM_PROFILE: CUSTOM_PROFILE,
}


def _validate_width(model_ch: int) -> int:
    width = int(model_ch)
    if width < NORMALIZATION_GROUPS or width % NORMALIZATION_GROUPS:
        raise FlowModelError(
            "flow model width must be a positive multiple of 8 for GroupNorm(8)"
        )
    if width % SE_REDUCTION:
        raise FlowModelError(
            "flow model width must be divisible by the squeeze-excitation reduction 4"
        )
    return width


def select_flow_profile(
    name: Optional[str] = None,
    overrides: Optional[Mapping[str, Any]] = None,
    *,
    model_ch: Optional[int] = None,
    width: Optional[int] = None,
) -> FlowWidthProfile:
    """Resolve a width only from an explicit named profile or custom override.

    ``None`` intentionally selects the observed 256-wide checkpoint profile.
    The manuscript-reported 128-wide value is never silently substituted for
    an observed checkpoint width, and all returned profiles explicitly retain
    ``exact_paper_reproduction=False``.
    """

    raw_name = HISTORICAL_CHECKPOINT_PROFILE if name is None else str(name).strip()
    normalized = _PROFILE_ALIASES.get(raw_name.lower(), raw_name)
    values = dict(overrides or {})
    requested_width = model_ch
    if requested_width is None:
        requested_width = width
    if requested_width is None:
        for key in ("model_ch", "model_width", "base_width", "width"):
            if key in values:
                requested_width = int(values[key])
                break

    if normalized == HISTORICAL_CHECKPOINT_PROFILE:
        selected = DEFAULT_MODEL_WIDTH
        if requested_width is not None and int(requested_width) != selected:
            raise FlowModelError(
                "historical_checkpoint_compatibility_256 requires model width 256; "
                "select manuscript_reported_128 or custom explicitly for another width"
            )
        return FlowWidthProfile(
            HISTORICAL_CHECKPOINT_PROFILE,
            selected,
            "observed wrapper and checkpoint metadata",
            checkpoint_compatible=True,
        )

    if normalized == MANUSCRIPT_REPORTED_PROFILE:
        selected = MANUSCRIPT_MODEL_WIDTH
        if requested_width is not None and int(requested_width) != selected:
            raise FlowModelError(
                "manuscript_reported_128 requires model width 128; use custom for another width"
            )
        return FlowWidthProfile(
            MANUSCRIPT_REPORTED_PROFILE,
            selected,
            "manuscript-reported width; not an observed checkpoint claim",
            checkpoint_compatible=False,
        )

    if normalized != CUSTOM_PROFILE:
        raise FlowModelError(
            "unknown flow width profile {!r}; expected {}, {}, or {}".format(
                raw_name,
                HISTORICAL_CHECKPOINT_PROFILE,
                MANUSCRIPT_REPORTED_PROFILE,
                CUSTOM_PROFILE,
            )
        )

    selected = DEFAULT_MODEL_WIDTH if requested_width is None else int(requested_width)
    selected = _validate_width(selected)
    return FlowWidthProfile(
        CUSTOM_PROFILE,
        selected,
        "caller-selected width",
        checkpoint_compatible=False,
    )


def sine_time_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """Return the source-compatible Transformer-style sine/cosine embedding."""

    if dim <= 0:
        raise FlowModelError("time embedding dimension must be positive")
    values = timesteps.reshape(-1).float()
    if dim == 1:
        return torch.sin(values[:, None])
    half = dim // 2
    denominator = max(half - 1, 1)
    freqs = torch.exp(
        -math.log(10000.0)
        * torch.arange(start=0, end=half, device=values.device).float()
        / denominator
    )
    args = values[:, None] * freqs[None]
    embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat(
            [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
        )
    return embedding


def _group_count(in_ch: int, out_ch: int) -> int:
    """Use the source's grouped-convolution count, with a safe custom-width fallback."""

    candidate = min(int(in_ch), int(out_ch))
    if in_ch % candidate == 0 and out_ch % candidate == 0:
        return candidate
    return math.gcd(int(in_ch), int(out_ch)) or 1


class ResBlock1D(nn.Module):
    """Source-compatible residual block with local and long grouped kernels."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        *,
        norm_groups: int = NORMALIZATION_GROUPS,
        long_kernel: int = LONG_KERNEL,
        se_reduction: int = SE_REDUCTION,
    ) -> None:
        super().__init__()
        in_ch = int(in_ch)
        out_ch = int(out_ch)
        if in_ch <= 0 or out_ch <= 0:
            raise FlowModelError("residual block channels must be positive")
        if out_ch % int(norm_groups):
            raise FlowModelError(
                "residual block output channels must be divisible by GroupNorm groups"
            )
        if out_ch < int(se_reduction):
            raise FlowModelError("residual block output channels are too small for SE")
        if int(long_kernel) <= 0 or int(long_kernel) % 2 == 0:
            raise FlowModelError("long kernel must be a positive odd integer")

        self.conv1 = nn.Conv1d(in_ch, out_ch, SHORT_KERNEL, padding=1)
        self.norm1 = nn.GroupNorm(int(norm_groups), out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, SHORT_KERNEL, padding=1)
        self.norm2 = nn.GroupNorm(int(norm_groups), out_ch)
        reduction_channels = max(1, out_ch // int(se_reduction))
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(out_ch, reduction_channels, 1),
            nn.SiLU(),
            nn.Conv1d(reduction_channels, out_ch, 1),
            nn.Sigmoid(),
        )
        groups = _group_count(in_ch, out_ch)
        self.bigconv = nn.Conv1d(
            in_ch,
            out_ch,
            int(long_kernel),
            padding=int(long_kernel) // 2,
            groups=groups,
        )
        self.shortcut = (
            nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.conv1(x)
        hidden = self.norm1(hidden)
        hidden = F.silu(hidden)
        hidden = self.conv2(hidden)
        hidden = self.norm2(hidden)
        hidden = hidden * self.se(hidden)
        hidden = hidden + self.bigconv(x)
        return self.shortcut(x) + hidden


class DownLayer(nn.Module):
    """Residual block followed by source-compatible average-pool downsampling."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.resblock = ResBlock1D(in_ch, out_ch)
        self.downsample = nn.AvgPool1d(2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = self.resblock(x)
        skip = hidden
        return self.downsample(hidden), skip


class UpLayer(nn.Module):
    """Nearest-neighbor x2 upsampling followed by skip concatenation."""

    def __init__(self, deep_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.resblock = ResBlock1D(deep_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        hidden = self.upsample(x)
        if hidden.shape[-1] != skip.shape[-1]:
            hidden = F.interpolate(hidden, size=skip.shape[-1], mode="nearest")
        return self.resblock(torch.cat([hidden, skip], dim=1))


class LatentDiffusionUNet1D(nn.Module):
    """Conditional 1D velocity model for latent tensors shaped ``[B,48,40]``."""

    def __init__(
        self,
        in_ch: int = LATENT_CHANNELS,
        model_ch: int = DEFAULT_MODEL_WIDTH,
        channel_mult: Sequence[int] = DEFAULT_CHANNEL_MULT,
        num_classes: int = NUM_CLASSES,
        *,
        latent_channels: Optional[int] = None,
        model_width: Optional[int] = None,
        classes: Optional[int] = None,
        width_profile: Optional[str] = None,
    ) -> None:
        super().__init__()

        if latent_channels is not None:
            if int(in_ch) != LATENT_CHANNELS and int(in_ch) != int(latent_channels):
                raise FlowModelError("in_ch and latent_channels must agree")
            in_ch = int(latent_channels)
        if model_width is not None:
            if int(model_ch) != DEFAULT_MODEL_WIDTH and int(model_ch) != int(model_width):
                raise FlowModelError("model_ch and model_width must agree")
            model_ch = int(model_width)
        if classes is not None:
            if int(num_classes) != NUM_CLASSES and int(num_classes) != int(classes):
                raise FlowModelError("num_classes and classes must agree")
            num_classes = int(classes)
        if width_profile is not None:
            # The constructor default is the observed profile.  An explicitly
            # selected manuscript profile is allowed to choose its own width;
            # an explicitly supplied model_ch still has to agree with it.
            profile = (
                select_flow_profile(width_profile)
                if model_ch == DEFAULT_MODEL_WIDTH and model_width is None
                else select_flow_profile(width_profile, model_ch=model_ch)
            )
            model_ch = profile.model_ch

        in_ch = int(in_ch)
        model_ch = _validate_width(model_ch)
        num_classes = int(num_classes)
        multipliers = tuple(int(value) for value in channel_mult)
        if in_ch <= 0:
            raise FlowModelError("latent input/output channels must be positive")
        if num_classes <= 0:
            raise FlowModelError("num_classes must be positive")
        if len(multipliers) < 2 or any(value <= 0 for value in multipliers):
            raise FlowModelError("channel_mult must contain at least two positive values")
        for multiplier in multipliers:
            _validate_width(model_ch * multiplier)

        self.in_ch = in_ch
        self.latent_channels = in_ch
        self.model_ch = model_ch
        self.model_width = model_ch
        self.num_classes = num_classes
        self.channel_mult = multipliers
        self.exact_paper_reproduction = EXACT_PAPER_REPRODUCTION

        self.time_embed = nn.Sequential(
            nn.Linear(model_ch, model_ch * 4),
            nn.SiLU(),
            nn.Linear(model_ch * 4, model_ch),
        )
        self.label_embed = nn.Embedding(num_classes, model_ch)
        self.input_proj = nn.Conv1d(in_ch, model_ch, 1)

        current_channels = model_ch
        self.enc_convs = nn.ModuleList()
        self.downs = nn.ModuleList()
        for multiplier in multipliers[:-1]:
            out_channels = model_ch * multiplier
            self.enc_convs.append(nn.Conv1d(current_channels, out_channels, 1))
            self.downs.append(DownLayer(out_channels, out_channels))
            current_channels = out_channels

        bottle_channels = model_ch * multipliers[-1]
        self.bottle_conv = nn.Conv1d(current_channels, bottle_channels, 1)
        self.bottle_resblock = ResBlock1D(bottle_channels, bottle_channels)
        current_channels = bottle_channels

        self.ups = nn.ModuleList()
        for multiplier in reversed(multipliers[:-1]):
            skip_channels = model_ch * multiplier
            out_channels = model_ch * multiplier
            self.ups.append(UpLayer(current_channels, skip_channels, out_channels))
            current_channels = out_channels

        self.out_proj = nn.Conv1d(current_channels, in_ch, 1)

    def _conditioning_inputs(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 3:
            raise FlowModelError("flow input must have shape [batch, latent_channels, time]")
        if x.size(1) != self.in_ch:
            raise FlowModelError(
                "latent channel count {} does not match flow model {}".format(
                    x.size(1), self.in_ch
                )
            )
        times = torch.as_tensor(t, device=x.device, dtype=torch.float32).reshape(-1)
        if times.numel() == 1 and x.size(0) != 1:
            times = times.expand(x.size(0))
        if times.numel() != x.size(0):
            raise FlowModelError("time conditioning must have one value per batch item")
        class_ids = torch.as_tensor(labels, device=x.device, dtype=torch.long).reshape(-1)
        if class_ids.numel() != x.size(0):
            raise FlowModelError("class conditioning must have one value per batch item")
        if torch.any(class_ids < 0) or torch.any(class_ids >= self.num_classes):
            raise FlowModelError(
                "class labels must be in [0, {}) for this flow model".format(
                    self.num_classes
                )
            )
        return times, class_ids

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Predict velocity; ``t`` is the source-scaled model time ``1000*t``."""

        times, class_ids = self._conditioning_inputs(x, t, labels)
        embedding = self.time_embed(sine_time_embedding(times, self.model_ch))
        embedding = embedding + self.label_embed(class_ids)
        hidden = self.input_proj(x) + embedding.unsqueeze(-1)

        skips = []
        for conv, down in zip(self.enc_convs, self.downs):
            hidden = conv(hidden)
            hidden, skip = down(hidden)
            skips.append(skip)

        hidden = self.bottle_conv(hidden)
        hidden = self.bottle_resblock(hidden)
        for up in self.ups:
            hidden = up(hidden, skips.pop())
        return self.out_proj(hidden)


# Source and public callers use both names.  Keeping one implementation avoids
# the duplicate model drift that existed between the two training scripts.
FlowUNet1D = LatentDiffusionUNet1D
RectifiedFlowUNet1D = LatentDiffusionUNet1D


def flow_model_metadata(
    *,
    channels: int,
    latent_channels: int = LATENT_CHANNELS,
    num_classes: int = NUM_CLASSES,
    model_ch: int = DEFAULT_MODEL_WIDTH,
    width_profile: Optional[str] = None,
) -> Dict[str, Any]:
    """Return JSON-safe model metadata without exposing parameters."""

    if channels not in (3, 6):
        raise FlowModelError("paired VAE/flow channels must be 3 or 6")
    if int(num_classes) <= 0:
        raise FlowModelError("num_classes must be positive")
    if width_profile is None:
        profile = (
            HISTORICAL_CHECKPOINT_PROFILE
            if int(model_ch) == DEFAULT_MODEL_WIDTH
            else CUSTOM_PROFILE
        )
    else:
        profile = (
            select_flow_profile(width_profile).name
            if int(model_ch) == DEFAULT_MODEL_WIDTH
            else select_flow_profile(width_profile, model_ch=model_ch).name
        )
    selected = (
        select_flow_profile(profile)
        if profile == MANUSCRIPT_REPORTED_PROFILE and int(model_ch) == DEFAULT_MODEL_WIDTH
        else select_flow_profile(profile, model_ch=model_ch)
    )
    if int(latent_channels) != LATENT_CHANNELS:
        raise FlowModelError("public flow latent channel count must be 48")
    if int(num_classes) <= 0:
        raise FlowModelError("num_classes must be positive")
    return {
        "channels": int(channels),
        "latent_channels": int(latent_channels),
        "latent_time_steps": 40,
        "num_classes": int(num_classes),
        "model_ch": selected.model_ch,
        "width_profile": selected.name,
        "architecture": {
            "channel_multipliers": list(DEFAULT_CHANNEL_MULT),
            "residual_short_kernel": SHORT_KERNEL,
            "residual_long_kernel": LONG_KERNEL,
            "normalization_groups": NORMALIZATION_GROUPS,
            "se_reduction": SE_REDUCTION,
            "downsampling": "avg_pool_factor_2",
            "upsampling": "nearest_factor_2",
        },
        "exact_paper_reproduction": EXACT_PAPER_REPRODUCTION,
        "paper_width_conflict_unresolved": True,
    }


__all__ = [
    "CUSTOM_PROFILE",
    "DEFAULT_CHANNEL_MULT",
    "DEFAULT_MODEL_WIDTH",
    "EXACT_PAPER_REPRODUCTION",
    "FlowModelError",
    "FlowUNet1D",
    "FlowWidthProfile",
    "HISTORICAL_CHECKPOINT_PROFILE",
    "LATENT_CHANNELS",
    "LatentDiffusionUNet1D",
    "LONG_KERNEL",
    "MANUSCRIPT_MODEL_WIDTH",
    "MANUSCRIPT_REPORTED_PROFILE",
    "NUM_CLASSES",
    "RectifiedFlowUNet1D",
    "ResBlock1D",
    "DownLayer",
    "UpLayer",
    "flow_model_metadata",
    "select_flow_profile",
    "sine_time_embedding",
]
