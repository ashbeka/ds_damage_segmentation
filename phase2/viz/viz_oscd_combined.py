"""
Combined OSCD visualization: Phase 1 DS/PCA-diff priors + Phase 2 segmentation.

For each city, produces a single multi-panel figure showing:
- Pre RGB
- Post RGB
- GT overlay
- DS projection map (from Phase 1 change maps)
- PCA-diff map (from Phase 1 change maps)
- Segmentation probability map (Phase 2 model)
- DS mask (Otsu on DS projection)
- Segmentation mask (thresholded prob)

This is intended as an interpretability tool to see how priors and
segmentation outputs relate to each other and to the ground truth.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
  sys.path.append(str(ROOT))

from phase1.data.oscd_dataset import OSCDEvaluatorDataset
from phase1.data.preprocessing import apply_normalization, load_band_stats
from models.unet2d import UNet2D
from models.unet2d_resnet_backbone import UNet2DResNetBackbone
from models.priors_fusion_heads import PriorsFusionUNet
from data.oscd_seg_dataset import OSCDSegmentationDataset
from viz.overlay_utils import overlay_mask_on_rgb, rgb_from_s2


def otsu_threshold(scores: np.ndarray, valid_mask: np.ndarray) -> float:
  """Compute Otsu threshold on valid pixels (local copy to avoid phase1 import issues)."""
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
  return float(bin_edges[:-1][idx])


def parse_args():
  ap = argparse.ArgumentParser()
  ap.add_argument("--config", type=Path, required=True, help="Segmentation config YAML (Phase 2)")
  ap.add_argument("--oscd_root", type=Path, required=True, help="Path to OSCD root (phase1/data/raw/OSCD)")
  ap.add_argument(
    "--phase1_change_maps_root",
    type=Path,
    required=True,
    help="Root of Phase 1 change maps (e.g., phase1/outputs/oscd_saved/oscd_change_maps)",
  )
  ap.add_argument("--checkpoint", type=Path, required=True, help="Segmentation checkpoint (best.ckpt)")
  ap.add_argument("--output_dir", type=Path, required=True, help="Where to save combined figures")
  ap.add_argument(
    "--cities",
    default="test",
    help="Comma-separated city list or one of: train|val|test|all (based on OSCD split)",
  )
  return ap.parse_args()


def load_config(path: Path):
  with path.open("r", encoding="utf-8") as f:
    return yaml.safe_load(f)


def infer_channel_counts(cfg: dict) -> dict:
  feats = cfg["features"]
  n_raw = 0
  if feats.get("use_raw_s2", True):
    if feats.get("use_pre_post_stack", True):
      n_raw += 26
    else:
      n_raw += 13
  priors = feats.get("priors", {})
  n_priors = 0
  for _, enabled in priors.items():
    if enabled:
      n_priors += 1
  return {"n_raw": n_raw, "n_priors": n_priors, "total": n_raw + n_priors}


def build_model(cfg: dict, in_channels: int) -> torch.nn.Module:
  model_cfg = cfg["model"]
  model_type = model_cfg.get("type", "unet2d")
  num_classes = model_cfg.get("num_classes", 1)
  if model_type == "unet2d":
    base_ch = model_cfg.get("base_channels", 64)
    depth = model_cfg.get("depth", 4)
    return UNet2D(in_channels=in_channels, base_channels=base_ch, depth=depth, num_classes=num_classes)
  if model_type == "unet2d_resnet":
    pretrained = model_cfg.get("pretrained", False)
    return UNet2DResNetBackbone(in_channels=in_channels, num_classes=num_classes, pretrained=pretrained)
  if model_type == "priors_fusion_unet":
    counts = infer_channel_counts(cfg)
    n_raw, n_priors = counts["n_raw"], counts["n_priors"]
    base_ch = model_cfg.get("base_channels", 64)
    depth = model_cfg.get("depth", 4)
    return PriorsFusionUNet(n_raw=n_raw, n_priors=n_priors, base_channels=base_ch, depth=depth, num_classes=num_classes)
  raise NotImplementedError(f"Unknown model type: {model_type}")


def load_ds_pca_maps(change_root: Path, split_name: str, city: str):
  """Load DS projection and PCA-diff score maps from Phase 1 change maps."""
  ds_path = change_root / split_name / "ds_projection" / f"{city}_score.npy"
  pca_path = change_root / split_name / "pca_diff" / f"{city}_score.npy"
  ds = np.load(ds_path).astype(np.float32)
  pca = np.load(pca_path).astype(np.float32)
  return ds, pca


def normalize_map(m: np.ndarray, valid_mask: np.ndarray | None = None) -> np.ndarray:
  m = m.astype(np.float32)
  if valid_mask is not None:
    vals = m[valid_mask]
  else:
    vals = m.flatten()
  if vals.size == 0:
    return np.zeros_like(m, dtype=np.float32)
  lo, hi = np.percentile(vals, [2, 98])
  scaled = (m - lo) / max(hi - lo, 1e-6)
  return np.clip(scaled, 0, 1).astype(np.float32)


def compute_segmentation_probs(cfg: dict, oscd_root: Path, change_root: Path, ckpt_path: Path, device: torch.device):
  """Run the segmentation model over the test split and return per-city prob maps."""
  counts = infer_channel_counts(cfg)
  in_ch = counts["total"]

  model = build_model(cfg, in_ch).to(device)
  ckpt = torch.load(ckpt_path, map_location=device)
  model.load_state_dict(ckpt["model_state"])
  model.eval()

  # Build patch dataset for test split
  ds = OSCDSegmentationDataset(oscd_root, "test", cfg, change_root)
  loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

  city_preds: dict[str, list[tuple[np.ndarray, int, int]]] = {}
  for batch, (city, y0, x0) in zip(loader, ds.patches):
    x = batch["x"].to(device)
    with torch.no_grad():
      logits = model(x)
      if logits.shape[-2:] != x.shape[-2:]:
        logits = torch.nn.functional.interpolate(
          logits, size=x.shape[-2:], mode="bilinear", align_corners=False
        )
      probs = torch.sigmoid(logits).squeeze(0).squeeze(0).cpu().numpy()
    if city not in city_preds:
      city_preds[city] = []
    city_preds[city].append((probs, y0, x0))

  return city_preds


def main():
  args = parse_args()
  cfg = load_config(args.config)
  out_dir = args.output_dir
  out_dir.mkdir(parents=True, exist_ok=True)

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  # stats and band order from Phase 1
  stats = load_band_stats(ROOT / "phase1" / "data" / "oscd_band_stats.json")
  band_order = cfg["dataset"].get(
    "band_order",
    ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B09", "B10", "B11", "B12", "B8A"],
  )

  # Base datasets for full tiles
  ds_train = OSCDEvaluatorDataset(args.oscd_root, "train", band_order)
  ds_val = OSCDEvaluatorDataset(args.oscd_root, "val", band_order)
  ds_test = OSCDEvaluatorDataset(args.oscd_root, "test", band_order)
  split_map = {"train": ds_train, "val": ds_val, "test": ds_test}

  # Resolve which cities to visualize
  if args.cities.lower() == "all":
    targets: list[tuple[str, str]] = []
    for split_name, dset in split_map.items():
      targets.extend([(split_name, c) for c in dset.cities])
  elif args.cities.lower() in split_map:
    split_name = args.cities.lower()
    targets = [(split_name, c) for c in split_map[split_name].cities]
  else:
    # explicit list; infer split per city from split_map
    names = [c.strip() for c in args.cities.split(",") if c.strip()]
    inv = {}
    for sn, dset in split_map.items():
      for c in dset.cities:
        inv[c] = sn
    targets = [(inv.get(c, "test"), c) for c in names]

  # Compute segmentation probabilities for test split (patch-based)
  city_probs = compute_segmentation_probs(cfg, args.oscd_root, args.phase1_change_maps_root, args.checkpoint, device)

  # Create combined figures
  change_root = args.phase1_change_maps_root
  for split_name, city in targets:
    if split_name != "test":
      # we only have segmentation probs computed for test split in this helper
      continue
    if city not in city_probs:
      continue

    dset = split_map[split_name]
    sample = dset.load_city(city)

    # Normalized pre/post for consistency (not strictly needed for viz)
    x1n, vm = apply_normalization(sample.x_pre, stats, valid_mask=sample.valid_mask, nodata_value=0.0)
    _x2n, _ = apply_normalization(sample.x_post, stats, valid_mask=sample.valid_mask, nodata_value=0.0)

    rgb_pre = rgb_from_s2(sample.x_pre, band_order, valid_mask=sample.valid_mask)
    rgb_post = rgb_from_s2(sample.x_post, band_order, valid_mask=sample.valid_mask)
    gt = (sample.y[0] > 0) if sample.y is not None else None

    # DS and PCA maps from Phase 1
    ds_proj_raw, pca_raw = load_ds_pca_maps(change_root, split_name, city)
    ds_proj = normalize_map(ds_proj_raw, valid_mask=vm)
    pca_map = normalize_map(pca_raw, valid_mask=vm)
    ds_mask = ds_proj_raw >= otsu_threshold(ds_proj_raw, vm)

    # Reconstruct segmentation prob map from patches
    h, w = sample.x_pre.shape[1:]
    prob_full = np.zeros((h, w), dtype=np.float32)
    counts = np.zeros((h, w), dtype=np.float32)
    for probs, y0, x0 in city_probs[city]:
      ys = slice(y0, y0 + probs.shape[0])
      xs = slice(x0, x0 + probs.shape[1])
      region = prob_full[ys, xs]
      rh, rw = region.shape
      ph, pw = probs.shape
      h_eff = min(rh, ph)
      w_eff = min(rw, pw)
      prob_full[ys.start : ys.start + h_eff, xs.start : xs.start + w_eff] += probs[:h_eff, :w_eff]
      counts[ys.start : ys.start + h_eff, xs.start : xs.start + w_eff] += 1.0
    counts[counts == 0] = 1.0
    prob_full /= counts
    seg_mask = (prob_full >= 0.5).astype(np.uint8)

    # Build figure: 3x3 grid as described
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    fig.suptitle(f"{city} ({split_name}) – DS/PCA vs segmentation", fontsize=12)

    # Row 1: pre, post, GT
    axes[0, 0].imshow(rgb_pre)
    axes[0, 0].set_title("Pre RGB")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(rgb_post)
    axes[0, 1].set_title("Post RGB")
    axes[0, 1].axis("off")

    if gt is not None:
      axes[0, 2].imshow(overlay_mask_on_rgb(rgb_pre, gt, color=(1, 0, 0)))
      axes[0, 2].set_title("GT overlay")
    else:
      axes[0, 2].imshow(rgb_pre)
      axes[0, 2].set_title("GT overlay (none)")
    axes[0, 2].axis("off")

    # Row 2: DS proj, PCA-diff, seg prob
    im_ds = axes[1, 0].imshow(ds_proj, cmap="inferno")
    axes[1, 0].set_title("DS projection (Phase 1)")
    axes[1, 0].axis("off")
    fig.colorbar(im_ds, ax=axes[1, 0])

    im_pca = axes[1, 1].imshow(pca_map, cmap="inferno")
    axes[1, 1].set_title("PCA-diff (Phase 1)")
    axes[1, 1].axis("off")
    fig.colorbar(im_pca, ax=axes[1, 1])

    im_seg = axes[1, 2].imshow(prob_full, cmap="inferno")
    axes[1, 2].set_title("Segmentation prob (Phase 2)")
    axes[1, 2].axis("off")
    fig.colorbar(im_seg, ax=axes[1, 2])

    # Row 3: DS mask, seg mask, blank for future metrics
    axes[2, 0].imshow(ds_mask, cmap="gray", vmin=0, vmax=1)
    axes[2, 0].set_title("DS mask (Otsu)")
    axes[2, 0].axis("off")

    axes[2, 1].imshow(seg_mask, cmap="gray", vmin=0, vmax=1)
    axes[2, 1].set_title("Seg mask (0.5)")
    axes[2, 1].axis("off")

    axes[2, 2].axis("off")

    fig.tight_layout()
    fig.savefig(out_dir / f"{city}_combined_summary.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
  main()
