"""
Change Vector Analysis baseline (spec Section 4.2.2).

We compute the same L2 magnitude as pixel differencing; thresholding is handled
upstream (Otsu/global). Normalization is min-max per tile.
"""
from __future__ import annotations

import numpy as np

Array = np.ndarray


def cva_score(x1: Array, x2: Array, valid_mask: Array) -> Array:
    diff = x2 - x1
    score = np.linalg.norm(diff, axis=0)
    score[~valid_mask] = 0.0
    if np.any(valid_mask):
        v = score[valid_mask]
        vmin, vmax = v.min(), v.max()
        if vmax > vmin:
            score = (score - vmin) / (vmax - vmin)
        else:
            score[:] = 0.0
    return score.astype(np.float32)
