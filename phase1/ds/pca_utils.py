"""
PCA helpers for DS (spec Section 4.1).

Functions here handle rank selection by variance retention, orthonormalization,
and projector utilities used by DS scoring.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from sklearn.decomposition import PCA


Array = np.ndarray


@dataclass
class PCABasis:
    basis: Array  # shape (d, r)
    explained_variance_ratio: Array  # shape (r,)
    rank: int


def fit_pca_basis(
    x: Array,
    rank: Optional[int] = None,
    variance_threshold: Optional[float] = 0.95,
    random_state: int = 1234,
    use_randomized: bool = True,
) -> PCABasis:
    """
    Fit PCA on X (d, n) and return a basis with either fixed rank or energy threshold.
    """
    if x.ndim != 2:
        raise ValueError(f"Expected (d, n) matrix, got {x.shape}")
    # sklearn expects samples x features, so transpose
    samples = x.T
    solver = "randomized" if use_randomized else "full"
    # Fit full PCA once, then truncate to desired rank/energy
    pca_full = PCA(n_components=min(samples.shape), svd_solver=solver, random_state=random_state)
    pca_full.fit(samples)

    if variance_threshold is not None:
        cumsum = np.cumsum(pca_full.explained_variance_ratio_)
        r = int(np.searchsorted(cumsum, variance_threshold) + 1)
    elif rank is not None:
        r = int(rank)
    else:
        r = pca_full.components_.shape[0]

    r = max(1, min(r, pca_full.components_.shape[0]))
    basis = pca_full.components_[:r].T  # (d, r)
    explained = pca_full.explained_variance_ratio_[:r]
    return PCABasis(basis=basis.astype(np.float32), explained_variance_ratio=explained.astype(np.float32), rank=r)


def orthonormalize(mat: Array) -> Array:
    """Return an orthonormal basis spanning columns of `mat` via QR."""
    if mat.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got {mat.shape}")
    q, _ = np.linalg.qr(mat)
    return q


def residual_projector(basis: Array) -> Array:
    """
    Compute residual projector R = I - P where P = basis basis^T.
    basis is assumed orthonormal (d, r).
    """
    d = basis.shape[0]
    return np.eye(d, dtype=basis.dtype) - basis @ basis.T


def project(basis: Array, x: Array) -> Array:
    """
    Project data matrix (d, n) onto basis (d, r) -> (r, n).
    """
    return basis.T @ x


def reconstruct(basis: Array, coeffs: Array) -> Array:
    """Reconstruct from projection coefficients."""
    return basis @ coeffs


def cross_residual_energy(residual_proj: Array, x: Array) -> Array:
    """Compute squared residual norms for each column of x."""
    rx = residual_proj @ x
    return np.sum(rx * rx, axis=0)


def difference_subspace(phi: Array, psi: Array) -> Array:
    """
    Construct difference subspace D = orth([R_psi * phi, R_phi * psi]).
    """
    if phi.shape[0] != psi.shape[0]:
        raise ValueError("Bases must have same dimensionality.")
    r_phi = residual_projector(phi)
    r_psi = residual_projector(psi)
    stacked = np.concatenate([r_psi @ phi, r_phi @ psi], axis=1)
    return orthonormalize(stacked)
