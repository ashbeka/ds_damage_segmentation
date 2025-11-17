"""
PCA-diff baseline (spec Section 4.2.3).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.decomposition import PCA

Array = np.ndarray


def pca_diff_score(
    x1: Array,
    x2: Array,
    valid_mask: Array,
    rank_S: Optional[int] = None,
    variance_threshold: float = 0.95,
    random_state: int = 1234,
) -> Array:
    """
    Compute PCA on the difference image and return combined magnitude of top PCs.
    """
    diff = x2 - x1
    mat = diff[:, valid_mask].T  # samples x features
    if mat.shape[0] == 0:
        raise RuntimeError("No valid pixels for PCA-diff.")
    n_components = rank_S if rank_S is not None else min(mat.shape[0], mat.shape[1])
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=random_state)
    pca.fit(mat)
    # Select components by variance threshold if applicable
    if variance_threshold is not None:
        cs = np.cumsum(pca.explained_variance_ratio_)
        keep = max(1, int(np.searchsorted(cs, variance_threshold) + 1))
    else:
        keep = pca.components_.shape[0]
    comps = pca.transform(mat)[:, :keep]
    mags = np.linalg.norm(comps, axis=1)
    score = np.zeros_like(valid_mask, dtype=np.float32)
    score[valid_mask] = mags
    # Normalize per tile
    if mags.size > 0:
        vmin, vmax = mags.min(), mags.max()
        if vmax > vmin:
            score[valid_mask] = (mags - vmin) / (vmax - vmin)
        else:
            score[valid_mask] = 0.0
    return score
