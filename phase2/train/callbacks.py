"""
Simple training callbacks: checkpointing and metric logging.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import torch


class ModelCheckpoint:
  def __init__(self, out_dir: Path, monitor: str = "val_iou", mode: str = "max"):
    self.out_dir = Path(out_dir)
    self.monitor = monitor
    self.mode = mode
    self.best = None
    self.out_dir.mkdir(parents=True, exist_ok=True)

  def step(self, metrics: Dict[str, float], model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int):
    value = metrics.get(self.monitor)
    if value is None:
      return
    if self.best is None or (self.mode == "max" and value > self.best) or (self.mode == "min" and value < self.best):
      self.best = value
      state = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "monitor": self.monitor,
        "best_value": self.best,
      }
      torch.save(state, self.out_dir / "best.ckpt")


class MetricsLogger:
  def __init__(self, out_dir: Path):
    self.out_dir = Path(out_dir)
    self.out_dir.mkdir(parents=True, exist_ok=True)
    self.records = []

  def log(self, epoch: int, train_metrics: Dict[str, float], val_metrics: Dict[str, float]):
    rec = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
    self.records.append(rec)
    with (self.out_dir / "train_log.json").open("w", encoding="utf-8") as f:
      json.dump(self.records, f, indent=2)


