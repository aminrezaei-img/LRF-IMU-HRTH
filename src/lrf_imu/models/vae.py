"""The copied operational one-dimensional latent VAE.

The module deliberately keeps the source module's layer names and ordering so
that public models can consume the observed ``{"vae": state_dict}`` boundary.
This is an evidence-labelled compatibility implementation, not an exact paper
reproduction.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn


SUPPORTED_CHANNELS = (3, 6)

MIGRATION_PROVENANCE = {
    "original_relative_path": "VAE/VAE_logic.py",
    "original_sha256": "3C989BB8242236D3107AE75A1533622D955D6E739D0B44103636897D01E80505",
    "public_destination": "src/lrf_imu/models/vae.py",
    "copied_minimal_modifications": (
        "Copied LatentVAE1D layer order, names, dimensions, activations, and "
        "forward equations; added public 3CH/6CH validation and keyword aliases; "
        "removed source import-time environment mutation and unrelated pipeline code."
    ),
}


class LatentVAE1D(nn.Module):
    """Time-compressive 1D VAE used by the LRF-IMU latent pipeline.

    The default public geometry is ``[B, C, 160] -> [B, 48, 40]`` and back,
    with separately constructed 3-channel and 6-channel models.  Other lengths
    divisible by the configured stride remain useful for compact synthetic
    probes.
    """

    def __init__(
        self,
        in_ch: Optional[int] = None,
        z_ch: int = 48,
        hidden: int = 160,
        down_levels: int = 2,
        *,
        input_channels: Optional[int] = None,
        latent_channels: Optional[int] = None,
    ) -> None:
        super().__init__()

        if in_ch is None:
            in_ch = input_channels
        elif input_channels is not None and int(in_ch) != int(input_channels):
            raise ValueError("in_ch and input_channels must agree")
        if in_ch is None:
            raise TypeError("input_channels or in_ch is required")

        if latent_channels is not None:
            if z_ch != 48 and int(z_ch) != int(latent_channels):
                raise ValueError("z_ch and latent_channels must agree")
            z_ch = int(latent_channels)

        in_ch = int(in_ch)
        z_ch = int(z_ch)
        hidden = int(hidden)
        down_levels = int(down_levels)
        if in_ch not in SUPPORTED_CHANNELS:
            raise ValueError(
                "LatentVAE1D supports only independent 3-channel or 6-channel models"
            )
        if z_ch <= 0 or hidden <= 0:
            raise ValueError("latent and hidden channels must be positive")
        if down_levels < 0:
            raise ValueError("down_levels must not be negative")

        self.in_ch = in_ch
        self.z_ch = z_ch
        self.input_channels = in_ch
        self.latent_channels = z_ch
        self.down_levels = down_levels
        self.latent_stride = 2**down_levels

        # These names and module positions are intentionally copied from the
        # immutable source.  Do not rename them without a checkpoint migration.
        self.enc_in = nn.Sequential(
            nn.Conv1d(in_ch, hidden, kernel_size=7, padding=3),
            nn.GELU(),
        )

        downs = []
        for _ in range(down_levels):
            downs += [
                nn.Conv1d(hidden, hidden, kernel_size=4, stride=2, padding=1),
                nn.GELU(),
                nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
                nn.GELU(),
            ]
        self.downs = nn.Sequential(*downs)
        self.mu_head = nn.Conv1d(hidden, z_ch, kernel_size=1)
        self.lv_head = nn.Conv1d(hidden, z_ch, kernel_size=1)

        self.dec_in = nn.Sequential(
            nn.Conv1d(z_ch, hidden, kernel_size=1),
            nn.GELU(),
        )
        ups = []
        for _ in range(down_levels):
            ups += [
                nn.ConvTranspose1d(hidden, hidden, kernel_size=4, stride=2, padding=1),
                nn.GELU(),
                nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
                nn.GELU(),
            ]
        self.ups = nn.Sequential(*ups)
        self.dec_out = nn.Conv1d(hidden, in_ch, kernel_size=1)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the posterior mean and log-variance."""

        if x.ndim != 3:
            raise ValueError("VAE input must have shape [batch, channels, time]")
        if x.size(1) != self.in_ch:
            raise ValueError(
                "input channel count {} does not match model channel count {}".format(
                    x.size(1), self.in_ch
                )
            )
        if x.size(-1) <= 0 or x.size(-1) % self.latent_stride != 0:
            raise ValueError(
                "input time length must be a positive multiple of the latent stride {}".format(
                    self.latent_stride
                )
            )

        h = self.enc_in(x)
        h = self.downs(h)
        mu = self.mu_head(h)
        logvar = self.lv_head(h)
        return mu, logvar

    def reparameterize(
        self, mu: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        """Sample ``mu + exp(0.5 * logvar) * randn_like(mu)``."""

        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def reparam(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Source-compatible alias for :meth:`reparameterize`."""

        return self.reparameterize(mu, logvar)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode a latent tensor through the copied decoder."""

        if z.ndim != 3:
            raise ValueError("latent input must have shape [batch, latent_channels, time]")
        if z.size(1) != self.z_ch:
            raise ValueError(
                "latent channel count {} does not match model channel count {}".format(
                    z.size(1), self.z_ch
                )
            )
        h = self.dec_in(z)
        h = self.ups(h)
        return self.dec_out(h)

    def forward(
        self, x: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return reconstruction, posterior mean, and posterior log-variance."""

        mu, logvar = self.encode(x)
        z = mu if deterministic else self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar


__all__ = ["LatentVAE1D", "MIGRATION_PROVENANCE", "SUPPORTED_CHANNELS"]
