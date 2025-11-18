"""
Optional priors fusion architectures (spec Section 4.3).

For now this exposes a simple PriorsFusionUNet wrapper that concatenates
features from two branches (raw S2 and priors) early in the network.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .unet2d import UNet2D


class PriorsFusionUNet(nn.Module):
  def __init__(self, n_raw: int, n_priors: int, base_channels: int = 64, depth: int = 4, num_classes: int = 1):
    super().__init__()
    self.n_raw = n_raw
    self.n_priors = n_priors
    self.raw_proj = nn.Conv2d(n_raw, n_raw, kernel_size=1)
    self.prior_proj = nn.Conv2d(n_priors, n_priors, kernel_size=1)
    self.unet = UNet2D(in_channels=n_raw + n_priors, base_channels=base_channels, depth=depth, num_classes=num_classes)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # x: (B, n_raw+n_priors, H, W)
    raw = x[:, : self.n_raw]
    priors = x[:, self.n_raw :]
    raw_f = self.raw_proj(raw)
    prior_f = self.prior_proj(priors)
    fused = torch.cat([raw_f, prior_f], dim=1)
    return self.unet(fused)


