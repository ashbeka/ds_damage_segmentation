"""
IR-MAD baseline (stretch goal).

Implements a lightweight Iteratively Reweighted Multivariate Alteration Detection:
1) Estimate canonical directions via eigen-decomposition of a generalized covariance matrix.
2) Compute MAD variates and reweight observations based on chi-square distances.
3) Repeat for a small number of iterations.

This implementation supports optional subsampling for stability on large tiles.
"""
from __future__ import annotations

import numpy as np
import scipy.linalg as la

Array = np.ndarray


def ir_mad_score(
    x1: Array,
    x2: Array,
    valid_mask: Array,
    iters: int = 3,
    downsample_max_pixels: int = 200000,
    eps: float = 1e-6,
) -> Array:
    """
    Compute IR-MAD change magnitude. Returns min-max normalized score.
    """
    # Flatten valid pixels
    v = valid_mask.reshape(-1)
    mat1 = x1.reshape(x1.shape[0], -1)[:, v]
    mat2 = x2.reshape(x2.shape[0], -1)[:, v]
    n = mat1.shape[1]
    # Optional subsample for coefficient estimation
    if n > downsample_max_pixels:
        idx = np.random.choice(n, size=downsample_max_pixels, replace=False)
        samp1 = mat1[:, idx]
        samp2 = mat2[:, idx]
    else:
        samp1 = mat1
        samp2 = mat2

    w = np.ones(samp1.shape[1], dtype=np.float64)
    A = np.eye(samp1.shape[0])

    for _ in range(max(1, iters)):
        # Weighted centering
        w_norm = w / (np.sum(w) + eps)
        m1 = np.sum(samp1 * w_norm, axis=1, keepdims=True)
        m2 = np.sum(samp2 * w_norm, axis=1, keepdims=True)
        x1c = samp1 - m1
        x2c = samp2 - m2
        # Weighted covariance
        c11 = (x1c * w_norm) @ x1c.T
        c22 = (x2c * w_norm) @ x2c.T
        c12 = (x1c * w_norm) @ x2c.T
        c21 = c12.T
        # Solve generalized eigenproblem for canonical variates
        try:
            eigvals, eigvecs = la.eigh(
                c12 @ la.inv(c22 + eps * np.eye(c22.shape[0])),
                c11 + eps * np.eye(c11.shape[0]),
                check_finite=False,
            )
        except Exception:
            continue
        # Sort descending
        idx_sort = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx_sort]
        A = eigvecs[:, idx_sort]
        # Compute MAD variates on sample (difference of canonical projections)
        mads = (A.T @ samp1) - (A.T @ samp2)
        chi2 = np.sum((mads / (np.sqrt(np.maximum(eigvals, eps))[:, None] + eps)) ** 2, axis=0)
        w = 1.0 - np.exp(-0.5 * chi2)  # higher weight to likely change

    # Apply final transform to full matrices
    mads_full = (A.T @ mat1) - (A.T @ mat2)
    mag = np.sqrt(np.sum(mads_full ** 2, axis=0))
    score = np.zeros(valid_mask.size, dtype=np.float32)
    score[v] = mag
    # Min-max normalize
    if np.any(score > 0):
        s_min, s_max = score[v].min(), score[v].max()
        if s_max > s_min:
            score[v] = (score[v] - s_min) / (s_max - s_min + eps)
        else:
            score[v] = 0.0
    return score.reshape(valid_mask.shape)
