"""
Overlay utilities for Phase 2 visualization (spec Section 7.2).
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np


def robust_percentile_scale(arr: np.ndarray, valid_mask: np.ndarray | None = None, p_low: float = 2, p_high: float = 98) -> np.ndarray:
  """
  Scale an array to [0,1] using robust percentiles over valid pixels.
  """
  if valid_mask is not None:
    vals = arr[valid_mask]
  else:
    vals = arr.reshape(-1, arr.shape[-1]) if arr.ndim == 3 else arr.flatten()
  if vals.size == 0:
    return np.zeros_like(arr, dtype=np.float32)
  lo, hi = np.percentile(vals, [p_low, p_high])
  scaled = (arr - lo) / max(hi - lo, 1e-6)
  return np.clip(scaled, 0, 1).astype(np.float32)


def rgb_from_s2(cube: np.ndarray, band_order: List[str], valid_mask: np.ndarray | None = None) -> np.ndarray:
  idx = {b: i for i, b in enumerate(band_order)}
  r = cube[idx["B04"]]
  g = cube[idx["B03"]]
  b = cube[idx["B02"]]
  rgb = np.stack([r, g, b], axis=-1)
  rgb = robust_percentile_scale(rgb, valid_mask=valid_mask)
  return rgb


def overlay_mask_on_rgb(rgb: np.ndarray, mask: np.ndarray, color: Tuple[float, float, float] = (1.0, 0.0, 0.0), alpha: float = 0.4) -> np.ndarray:
  out = rgb.copy()
  m = mask.astype(bool)
  out[m] = (1 - alpha) * out[m] + alpha * np.array(color)
  return out


def tile_panels_to_grid(fig, axes_arr, images: List[Tuple[np.ndarray, str, str]]):
  """
  Utility to fill a grid of axes with images.
  images: list of (img, title, cmap_name).
  """
  for ax, (img, title, cmap_name) in zip(axes_arr.ravel(), images):
    if img is None:
      ax.axis("off")
      continue
    if cmap_name:
      ax.imshow(img, cmap=cmap_name)
    else:
      ax.imshow(img)
    ax.set_title(title)
    ax.axis("off")
  fig.tight_layout()
