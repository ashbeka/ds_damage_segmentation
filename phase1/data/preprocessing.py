"""
Preprocessing and normalization utilities (spec Section 3).

Implements:
- NODATA handling with configurable value and min_valid_bands.
- Bandwise z-score statistics (fit on OSCD train) with JSON serialization.
- Reshape helpers between (C, H, W) and (C, N) while tracking valid indices.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np


Array = np.ndarray


def build_valid_mask(
    x: Array,
    nodata_value: Optional[float] = 0.0,
    min_valid_bands: int = 3,
    existing_mask: Optional[Array] = None,
) -> Array:
    """
    Build a boolean valid-pixel mask given a spectral cube.

    A pixel is valid if at least `min_valid_bands` channels are not equal to `nodata_value`.
    If `existing_mask` is provided, it is AND-ed with the computed validity.
    """
    if x.ndim != 3:
        raise ValueError(f"Expected (C,H,W) array, got shape {x.shape}")
    _, h, w = x.shape
    mask = np.ones((h, w), dtype=bool) if existing_mask is None else existing_mask.astype(bool).copy()
    if nodata_value is not None:
        non_nodata = np.sum(x != nodata_value, axis=0)
        mask &= non_nodata >= min_valid_bands
    return mask


def vectorize_cube(x: Array, valid_mask: Optional[Array] = None) -> Tuple[Array, Tuple[Array, Array]]:
    """
    Reshape (C, H, W) -> (C, N) selecting pixels where valid_mask is True.
    Returns the flattened matrix and the (row_idx, col_idx) used for reconstruction.
    """
    if x.ndim != 3:
        raise ValueError(f"Expected (C,H,W) array, got shape {x.shape}")
    if valid_mask is None:
        h, w = x.shape[1:]
        valid_mask = np.ones((h, w), dtype=bool)
    rows, cols = np.where(valid_mask)
    mat = x[:, rows, cols].reshape(x.shape[0], -1)
    return mat, (rows, cols)


def devectorize_cube(
    mat: Array,
    idx: Tuple[Array, Array],
    shape_hw: Tuple[int, int],
    fill_value: float = np.nan,
) -> Array:
    """
    Reconstruct (C, H, W) array from a (C, N) matrix and coordinate indices.
    """
    c = mat.shape[0]
    h, w = shape_hw
    out = np.full((c, h, w), fill_value, dtype=mat.dtype)
    out[:, idx[0], idx[1]] = mat
    return out


@dataclass
class BandStats:
    mean: Array
    std: Array
    eps: float = 1e-6

    def normalize(self, x: Array) -> Array:
        """Apply bandwise z-score normalization."""
        if x.shape[0] != self.mean.shape[0]:
            raise ValueError(f"Channel mismatch: x has {x.shape[0]}, stats have {self.mean.shape[0]}")
        return (x - self.mean[:, None, None]) / (self.std[:, None, None] + self.eps)

    def to_dict(self) -> dict:
        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "eps": self.eps,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BandStats":
        return cls(mean=np.array(data["mean"], dtype=np.float32), std=np.array(data["std"], dtype=np.float32), eps=float(data.get("eps", 1e-6)))


def compute_band_stats(
    cubes: Iterable[Array],
    nodata_value: Optional[float] = 0.0,
    min_valid_bands: int = 3,
    eps: float = 1e-6,
) -> BandStats:
    """
    Compute global bandwise mean/std over a collection of (C,H,W) cubes, ignoring NODATA.
    """
    sums = None
    sumsqs = None
    counts = None

    for cube in cubes:
        if cube.ndim != 3:
            raise ValueError(f"Expected (C,H,W), got {cube.shape}")
        mask = build_valid_mask(cube, nodata_value=nodata_value, min_valid_bands=min_valid_bands)
        mat, _ = vectorize_cube(cube, mask)
        if mat.size == 0:
            continue
        if sums is None:
            c = cube.shape[0]
            sums = np.zeros((c,), dtype=np.float64)
            sumsqs = np.zeros((c,), dtype=np.float64)
            counts = np.zeros((c,), dtype=np.int64)
        sums += mat.sum(axis=1)
        sumsqs += (mat ** 2).sum(axis=1)
        counts += mat.shape[1]

    if counts is None or np.any(counts == 0):
        raise RuntimeError("No valid pixels found to compute band stats.")

    mean = sums / counts
    var = np.maximum(sumsqs / counts - mean ** 2, 0.0)
    std = np.sqrt(var + eps)
    return BandStats(mean=mean.astype(np.float32), std=std.astype(np.float32), eps=eps)


def save_band_stats(stats: BandStats, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(stats.to_dict(), f, indent=2)


def load_band_stats(path: Path) -> BandStats:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return BandStats.from_dict(data)


def apply_normalization(
    x: Array,
    stats: BandStats,
    valid_mask: Optional[Array] = None,
    nodata_value: Optional[float] = 0.0,
    fill_value: float = 0.0,
) -> Tuple[Array, Array]:
    """
    Apply bandwise normalization and return normalized cube + valid mask used.
    Invalid pixels (if any) are set to `fill_value` after normalization for downstream convenience.
    """
    if valid_mask is None:
        valid_mask = build_valid_mask(x, nodata_value=nodata_value)
    normed = stats.normalize(x.astype(np.float32))
    if valid_mask is not None:
        normed[:, ~valid_mask] = fill_value
    return normed, valid_mask
