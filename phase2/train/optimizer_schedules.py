"""
Optimizer and LR scheduler utilities (spec Section 5.3).
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch.optim import Optimizer


def build_optimizer(params, cfg: Dict) -> Optimizer:
  name = cfg.get("name", "adamw").lower()
  lr = cfg.get("lr", 1e-3)
  weight_decay = cfg.get("weight_decay", 1e-4)
  if name == "adamw":
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
  if name == "adam":
    return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
  raise ValueError(f"Unknown optimizer: {name}")


def build_scheduler(optimizer: Optimizer, cfg: Dict):
  name = cfg.get("name", "cosine").lower()
  if name == "cosine":
    t_max = cfg.get("T_max", 50)
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)
  if name == "step":
    step_size = cfg.get("step_size", 20)
    gamma = cfg.get("gamma", 0.1)
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
  if name == "plateau":
    factor = cfg.get("factor", 0.5)
    patience = cfg.get("patience", 5)
    return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=factor, patience=patience)
  return None


