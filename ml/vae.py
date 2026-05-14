"""Convolutional VAE for 512×512 RGB displacement maps."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _down_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.LeakyReLU(0.2, inplace=True),
    )


def _up_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.LeakyReLU(0.2, inplace=True),
    )


class DisplacementVAE(nn.Module):
    """
    Expects input x in roughly [-1, 1] (e.g. (img/255)*2 - 1).
    Spatial path: 512 → 8 over 6 stride-2 stages (2^6 = 64).
    """

    def __init__(self, in_channels: int = 3, latent_dim: int = 128, base: int = 48) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.in_channels = in_channels

        c1, c2, c3, c4, c5, c6 = base, base * 2, base * 4, base * 6, base * 8, base * 8
        self.enc = nn.Sequential(
            _down_block(in_channels, c1),
            _down_block(c1, c2),
            _down_block(c2, c3),
            _down_block(c3, c4),
            _down_block(c4, c5),
            _down_block(c5, c6),
        )
        flat = 8 * 8 * c6
        self.fc_mu = nn.Linear(flat, latent_dim)
        self.fc_logvar = nn.Linear(flat, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, flat)

        self.dec = nn.Sequential(
            _up_block(c6, c5),
            _up_block(c5, c4),
            _up_block(c4, c3),
            _up_block(c3, c2),
            _up_block(c2, c1),
            nn.ConvTranspose2d(c1, in_channels, kernel_size=4, stride=2, padding=1),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.enc(x)
        h = h.flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc_decode(z)
        h = h.view(z.size(0), -1, 8, 8)
        return self.dec(h)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    @torch.inference_mode()
    def encode_mu(self, x: torch.Tensor) -> torch.Tensor:
        mu, _ = self.encode(x)
        return mu


def vae_loss(recon: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    recon_loss = F.l1_loss(recon, x, reduction="mean")
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl, recon_loss, kl
