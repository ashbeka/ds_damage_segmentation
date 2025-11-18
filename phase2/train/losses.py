"""
Segmentation losses for Phase 2 (spec Section 5.2).
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class BCEDiceLoss(nn.Module):
  def __init__(self, bce_weight: float = 1.0, dice_weight: float = 1.0, smooth: float = 1.0):
    super().__init__()
    self.bce_weight = bce_weight
    self.dice_weight = dice_weight
    self.bce = nn.BCEWithLogitsLoss(reduction="mean")
    self.smooth = smooth

  def forward(self, logits: torch.Tensor, targets: torch.Tensor, valid_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # logits: (B,1,H,W), targets: (B,1,H,W), valid_mask: (B,1,H,W)
    probs = torch.sigmoid(logits)
    valid = valid_mask.bool()
    if valid.any():
      bce = self.bce(logits[valid], targets[valid])
    else:
      bce = logits.new_tensor(0.0)

    # dice over valid pixels
    probs_flat = probs[valid]
    targets_flat = targets[valid]
    intersection = (probs_flat * targets_flat).sum()
    denom = probs_flat.sum() + targets_flat.sum()
    dice = 1.0 - (2.0 * intersection + self.smooth) / (denom + self.smooth)

    loss = self.bce_weight * bce + self.dice_weight * dice
    return loss, bce.detach(), dice.detach()


