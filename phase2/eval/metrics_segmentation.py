"""
Segmentation metrics for Phase 2 (spec Section 6.1).
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score


def binary_stats(pred: np.ndarray, target: np.ndarray, valid_mask: np.ndarray) -> Dict[str, float]:
  vm = valid_mask.astype(bool)
  p = pred.astype(bool)[vm]
  t = target.astype(bool)[vm]
  tp = float(np.logical_and(p, t).sum())
  fp = float(np.logical_and(p, ~t).sum())
  fn = float(np.logical_and(~p, t).sum())
  tn = float(np.logical_and(~p, ~t).sum())
  eps = 1e-6
  precision = tp / (tp + fp + eps)
  recall = tp / (tp + fn + eps)
  iou = tp / (tp + fp + fn + eps)
  f1 = 2 * precision * recall / (precision + recall + eps)
  acc = (tp + tn) / (tp + tn + fp + fn + eps)
  return {
    "tp": tp,
    "fp": fp,
    "fn": fn,
    "tn": tn,
    "precision": precision,
    "recall": recall,
    "iou": iou,
    "f1": f1,
    "acc": acc,
  }


def auroc_score(probs: np.ndarray, target: np.ndarray, valid_mask: np.ndarray) -> float:
  vm = valid_mask.astype(bool)
  if vm.sum() == 0:
    return float("nan")
  y_true = target[vm].astype(np.uint8)
  y_score = probs[vm].astype(np.float32)
  try:
    return float(roc_auc_score(y_true, y_score))
  except Exception:
    return float("nan")


