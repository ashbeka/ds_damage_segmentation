"""
Data augmentation utilities for Phase 2 (spec Section 3.2).
"""
from __future__ import annotations

from typing import Callable, List, Tuple

import torch


class RandomFlipRotate:
  """Random flips and 90-degree rotations applied consistently to (x, y, valid)."""

  def __call__(self, x: torch.Tensor, y: torch.Tensor, valid: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # x: (C,H,W), y: (1,H,W), valid: (1,H,W)
    if torch.rand(()) < 0.5:
      x = torch.flip(x, dims=[2])
      y = torch.flip(y, dims=[2])
      valid = torch.flip(valid, dims=[2])
    if torch.rand(()) < 0.5:
      x = torch.flip(x, dims=[1])
      y = torch.flip(y, dims=[1])
      valid = torch.flip(valid, dims=[1])
    # random rotation k * 90deg
    k = int(torch.randint(0, 4, (1,)))
    if k:
      x = torch.rot90(x, k, dims=[1, 2])
      y = torch.rot90(y, k, dims=[1, 2])
      valid = torch.rot90(valid, k, dims=[1, 2])
    return x, y, valid


class RandomGaussianNoise:
  """Add low-level Gaussian noise to spectral channels only."""

  def __init__(self, sigma: float = 0.01):
    self.sigma = sigma

  def __call__(self, x: torch.Tensor, y: torch.Tensor, valid: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    noise = torch.randn_like(x) * self.sigma
    x = x + noise
    return x, y, valid


class ComposeTransforms:
  def __init__(self, transforms: List[Callable]):
    self.transforms = transforms

  def __call__(self, x: torch.Tensor, y: torch.Tensor, valid: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    for t in self.transforms:
      x, y, valid = t(x, y, valid)
    return x, y, valid


