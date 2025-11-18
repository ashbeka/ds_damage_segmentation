"""
OSCD segmentation dataset for Phase 2 (spec Phase 2 Section 3).

Wraps the Phase 1 OSCD loader and adds:
- stacking of pre/post S2 bands,
- optional priors (DS / PCA-diff / pixel-diff) loaded from saved change maps,
- patch extraction and basic augmentations.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
  sys.path.append(str(ROOT))

from phase1.data.oscd_dataset import OSCDEvaluatorDataset
from phase1.data.preprocessing import apply_normalization, load_band_stats

from .transforms import ComposeTransforms, RandomFlipRotate, RandomGaussianNoise


Array = np.ndarray
def _load_priors(
  change_maps_root: Path,
  split: str,
  city: str,
  priors_cfg: Dict[str, bool],
) -> List[Array]:
  priors: List[Array] = []
  for method_key, enabled in priors_cfg.items():
    if not enabled:
      continue
    method_name = {
      "ds_projection": "ds_projection",
      "pca_diff": "pca_diff",
      "pixel_diff": "pixel_diff",
    }.get(method_key)
    if method_name is None:
      continue
    score_path = change_maps_root / split / method_name / f"{city}_score.npy"
    if not score_path.exists():
      raise FileNotFoundError(f"Missing prior score map at {score_path}")
    scores = np.load(score_path).astype(np.float32)
    s_min, s_max = float(scores.min()), float(scores.max())
    if s_max > s_min:
      scores = (scores - s_min) / (s_max - s_min)
    else:
      scores = np.zeros_like(scores, dtype=np.float32)
    priors.append(scores[None, ...])
  return priors


class OSCDSegmentationDataset(Dataset):
  """
  Patch-wise segmentation dataset built on top of the Phase 1 OSCD loader.
  """

  def __init__(
    self,
    oscd_root: Path,
    split: str,
    cfg: Dict,
    phase1_change_maps_root: Optional[Path] = None,
    stats_path: Optional[Path] = None,
  ):
    self.oscd_root = Path(oscd_root)
    self.split = split
    self.cfg = cfg
    self.phase1_change_maps_root = Path(phase1_change_maps_root) if phase1_change_maps_root else None

    band_order: List[str] = cfg["dataset"].get("band_order", [])
    if not band_order:
      # reuse Phase 1 default band order if not provided
      band_order = [
        "B01",
        "B02",
        "B03",
        "B04",
        "B05",
        "B06",
        "B07",
        "B08",
        "B09",
        "B10",
        "B11",
        "B12",
        "B8A",
      ]
    self.band_order = band_order

    self.patch_size: int = int(cfg["dataset"].get("patch_size", 256))
    self.patch_overlap: int = int(cfg["dataset"].get("patch_overlap", 64))

    # Phase 1 dataset to get tiles
    self.oscd_ds = OSCDEvaluatorDataset(
      self.oscd_root,
      split,
      band_order,
      nodata_value=0.0,
      min_valid_bands=cfg["dataset"].get("min_valid_bands", 3),
      stats_path=None,
      val_cities=cfg["dataset"]["split"].get("val", []),
      val_from_train=0,
    )

    # band stats for normalization (reuse Phase 1 stats)
    if stats_path is None:
      stats_path = ROOT / "phase1" / "data" / "oscd_band_stats.json"
    self.stats = load_band_stats(Path(stats_path))

    # precompute patch index list (city, y, x)
    self.patches: List[Tuple[str, int, int]] = []
    cities_for_split = cfg["dataset"]["split"][split]
    for city in cities_for_split:
      sample = self.oscd_ds.load_city(city)
      h, w = sample.x_pre.shape[1:]
      stride = max(1, self.patch_size - self.patch_overlap)

      def _positions(length: int, window: int, stride_val: int) -> List[int]:
        positions = list(range(0, max(1, length - window + 1), stride_val))
        last = length - window
        if positions:
          if positions[-1] != last:
            positions.append(max(0, last))
        else:
          positions = [0]
        return positions

      ys = _positions(h, self.patch_size, stride)
      xs = _positions(w, self.patch_size, stride)
      for y in ys:
        for x in xs:
          self.patches.append((city, y, x))

    # build transforms
    aug_cfg = cfg["dataset"].get("augmentations", {})
    tfs = []
    if aug_cfg.get("flip", True):
      tfs.append(RandomFlipRotate())
    if aug_cfg.get("noise", False):
      tfs.append(RandomGaussianNoise())
    self.transforms = ComposeTransforms(tfs) if tfs else None

  def __len__(self) -> int:
    return len(self.patches)

  def __getitem__(self, idx: int):
    city, y0, x0 = self.patches[idx]
    sample = self.oscd_ds.load_city(city)

    # normalize pre/post
    x1_norm, vm = apply_normalization(sample.x_pre, self.stats, valid_mask=sample.valid_mask, nodata_value=0.0)
    x2_norm, _ = apply_normalization(sample.x_post, self.stats, valid_mask=sample.valid_mask, nodata_value=0.0)

    feats = []
    feats_cfg = self.cfg["features"]
    if feats_cfg.get("use_raw_s2", True):
      if feats_cfg.get("use_pre_post_stack", True):
        feats.append(x1_norm)
        feats.append(x2_norm)
      else:
        feats.append(x2_norm - x1_norm)

    if self.phase1_change_maps_root is not None:
      priors_cfg = feats_cfg.get("priors", {})
      if priors_cfg:
        priors = _load_priors(self.phase1_change_maps_root, self.split, city, priors_cfg)
        feats.extend(priors)

    x_full = np.concatenate(feats, axis=0).astype(np.float32)
    y_full = (sample.y.astype(np.float32) if sample.y is not None else np.zeros((1, x_full.shape[1], x_full.shape[2]), dtype=np.float32))
    valid_full = vm.astype(np.float32)

    sl_y = slice(y0, y0 + self.patch_size)
    sl_x = slice(x0, x0 + self.patch_size)
    x_patch = x_full[:, sl_y, sl_x]
    y_patch = y_full[:, sl_y, sl_x]
    v_patch = valid_full[sl_y, sl_x]

    x_tensor = torch.from_numpy(x_patch)
    y_tensor = torch.from_numpy(y_patch)
    v_tensor = torch.from_numpy(v_patch[None, ...])

    if self.transforms is not None:
      x_tensor, y_tensor, v_tensor = self.transforms(x_tensor, y_tensor, v_tensor)

    return {"city": city, "split": self.split, "x": x_tensor, "y": y_tensor, "valid": v_tensor}
