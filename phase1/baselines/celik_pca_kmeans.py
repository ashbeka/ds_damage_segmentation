"""
Celik 2009 local PCA + k-means baseline (spec Section 4.2.4).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


Array = np.ndarray


def _extract_patches(diff: Array, patch_size: int) -> Array:
    """
    Extract sliding patches (with reflect padding) and return features of shape (N, C*patch*patch).
    """
    pad = patch_size // 2
    padded = np.pad(diff, ((0, 0), (pad, pad), (pad, pad)), mode="reflect")
    windows = np.lib.stride_tricks.sliding_window_view(padded, window_shape=(patch_size, patch_size), axis=(1, 2))
    # windows shape: (C, H, W, patch, patch)
    c, h, w, _, _ = windows.shape
    feats = windows.reshape(c, h, w, patch_size * patch_size)
    feats = feats.transpose(1, 2, 0, 3).reshape(h * w, c * patch_size * patch_size)
    return feats


def celik_score(
    x1: Array,
    x2: Array,
    patch_size: int = 9,
    pca_energy: float = 0.9,
    kmeans_init: str = "k-means++",
    max_iter: int = 100,
    valid_mask: Optional[Array] = None,
    random_state: int = 1234,
    downsample_max_side: Optional[int] = None,
) -> Array:
    """
    Apply Celik local PCA + k-means; returns a normalized change score map.
    """
    if patch_size % 2 == 0:
        raise ValueError("patch_size must be odd.")
    diff = x2 - x1
    # optional downsample for stability on large tiles
    if downsample_max_side is not None:
        h, w = diff.shape[1:]
        max_side = max(h, w)
        if max_side > downsample_max_side:
            step = int(np.ceil(max_side / downsample_max_side))
            diff = diff[:, ::step, ::step]
            if valid_mask is not None:
                valid_mask = valid_mask[::step, ::step]
    feats = _extract_patches(diff, patch_size)
    if valid_mask is not None:
        flat_mask = valid_mask.reshape(-1)
    else:
        flat_mask = np.ones(feats.shape[0], dtype=bool)

    feats_valid = feats[flat_mask]
    if feats_valid.shape[0] == 0:
        raise RuntimeError("No valid pixels for Celik baseline.")

    pca = PCA(n_components=min(feats_valid.shape), svd_solver="randomized", random_state=random_state)
    pca.fit(feats_valid)
    cs = np.cumsum(pca.explained_variance_ratio_)
    keep = max(1, int(np.searchsorted(cs, pca_energy) + 1))
    proj = pca.transform(feats_valid)[:, :keep]
    mags = np.linalg.norm(proj, axis=1)
    # Cluster on projected features
    kmeans = KMeans(n_clusters=2, init=kmeans_init, max_iter=max_iter, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(proj)
    cluster_scores = [mags[labels == k].mean() for k in range(2)]
    change_cluster = int(np.argmax(cluster_scores))
    score_flat = np.zeros_like(flat_mask, dtype=np.float32)
    score_flat[flat_mask] = (labels == change_cluster).astype(np.float32)

    # Smoothen by injecting magnitude as confidence and min-max normalize
    score_vals = np.zeros_like(flat_mask, dtype=np.float32)
    score_vals[flat_mask] = mags
    if mags.size > 0:
        vmin, vmax = mags.min(), mags.max()
        norm_mags = (mags - vmin) / (vmax - vmin + 1e-8) if vmax > vmin else np.zeros_like(mags)
        score_vals[flat_mask] = norm_mags
    score_flat = score_flat * score_vals

    score = score_flat.reshape(diff.shape[1], diff.shape[2])
    return score
