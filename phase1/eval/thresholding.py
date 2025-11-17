"""
Thresholding utilities (spec Section 5).
"""
from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import numpy as np

from eval.metrics import binary_metrics


def otsu_threshold(scores: np.ndarray, valid_mask: np.ndarray) -> float:
    """Compute Otsu threshold on valid pixels."""
    vals = scores[valid_mask].astype(np.float64)
    if vals.size == 0:
        return 0.0
    hist, bin_edges = np.histogram(vals, bins=256, range=(vals.min(), vals.max()))
    hist = hist.astype(np.float64)
    prob = hist / np.sum(hist)
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * bin_edges[:-1])
    mu_t = mu[-1]
    sigma_b_squared = (mu_t * omega - mu) ** 2 / (omega * (1 - omega) + 1e-12)
    idx = np.argmax(sigma_b_squared)
    return bin_edges[:-1][idx]


def apply_threshold(scores: np.ndarray, threshold: float) -> np.ndarray:
    return (scores >= threshold).astype(np.uint8)


def grid_search_threshold(
    train_scores: List[np.ndarray],
    train_labels: List[np.ndarray],
    valid_masks: List[np.ndarray],
    grid: Tuple[float, float, float],
    criterion: str = "iou",
) -> float:
    """
    Grid search threshold on train split to maximize IoU or F1.
    """
    thr_min, thr_max, thr_step = grid
    best_thr = thr_min
    best_score = -1.0
    thresholds = np.arange(thr_min, thr_max + 1e-6, thr_step)
    for thr in thresholds:
        agg = []
        for s, y, m in zip(train_scores, train_labels, valid_masks):
            pred = apply_threshold(s, thr)
            metrics = binary_metrics(pred, y, valid_mask=m)
            agg.append(metrics[criterion])
        mean_metric = float(np.mean(agg)) if agg else -1.0
        if mean_metric > best_score:
            best_score = mean_metric
            best_thr = thr
    return float(best_thr)
