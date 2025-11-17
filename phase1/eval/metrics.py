"""
Metrics for Phase 1 (spec Section 6).
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from sklearn import metrics as skm


Array = np.ndarray


def binary_metrics(pred: Array, target: Array, valid_mask: Optional[Array] = None) -> Dict[str, float]:
    if target.ndim == 3 and target.shape[0] == 1:
        target = target[0]
    if pred.ndim == 3 and pred.shape[0] == 1:
        pred = pred[0]
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: {pred.shape} vs {target.shape}")
    if valid_mask is None:
        valid_mask = np.ones_like(target, dtype=bool)
    pred_flat = pred.astype(bool)[valid_mask]
    tgt_flat = target.astype(bool)[valid_mask]
    tp = np.logical_and(pred_flat, tgt_flat).sum()
    fp = np.logical_and(pred_flat, ~tgt_flat).sum()
    fn = np.logical_and(~pred_flat, tgt_flat).sum()
    tn = np.logical_and(~pred_flat, ~tgt_flat).sum()
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "iou": float(iou),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def auroc_score(scores: Array, target: Array, valid_mask: Optional[Array] = None) -> float:
    if target.ndim == 3 and target.shape[0] == 1:
        target = target[0]
    if valid_mask is None:
        valid_mask = np.ones_like(target, dtype=bool)
    s = scores[valid_mask].reshape(-1)
    y = target[valid_mask].reshape(-1)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(skm.roc_auc_score(y, s))
