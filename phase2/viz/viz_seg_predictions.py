"""
Visualization of segmentation predictions on OSCD tiles (spec Section 7.1).
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


def parse_args():
  ap = argparse.ArgumentParser()
  ap.add_argument("--config", type=Path, required=True)
  ap.add_argument("--oscd_root", type=Path, required=True)
  ap.add_argument("--phase1_change_maps_root", type=Path, required=True)
  ap.add_argument("--checkpoint", type=Path, required=True)
  ap.add_argument("--output_dir", type=Path, required=True)
  ap.add_argument("--cities", default="test")
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


def main():
  args = parse_args()
  cfg = load_config(args.config)
  out_dir = args.output_dir
  out_dir.mkdir(parents=True, exist_ok=True)

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  # reuse stats from Phase 1
  stats = load_band_stats(ROOT / "phase1" / "data" / "oscd_band_stats.json")
  band_order = cfg["dataset"].get(
    "band_order",
    ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B09", "B10", "B11", "B12", "B8A"],
  )

  counts = infer_channel_counts(cfg)
  in_ch = counts["total"]

  model = build_model(cfg, in_ch).to(device)
  ckpt = torch.load(args.checkpoint, map_location=device)
  model.load_state_dict(ckpt["model_state"])
  model.eval()

  ds_all = OSCDEvaluatorDataset(args.oscd_root, "train", band_order)
  ds_val = OSCDEvaluatorDataset(args.oscd_root, "val", band_order)
  ds_test = OSCDEvaluatorDataset(args.oscd_root, "test", band_order)
  split_map = {"train": ds_all, "val": ds_val, "test": ds_test}
  cities = []
  if args.cities.lower() == "all":
    for d in split_map.values():
      cities.extend(d.cities)
  elif args.cities.lower() in split_map:
    cities = split_map[args.cities.lower()].cities
  else:
    cities = [c.strip() for c in args.cities.split(",") if c.strip()]

  # build dataset for predictions to reuse patching logic
  seg_ds = OSCDSegmentationDataset(args.oscd_root, "test", cfg, args.phase1_change_maps_root)
  loader = DataLoader(seg_ds, batch_size=1, shuffle=False, num_workers=0)

  # map city -> list of (probs, coords)
  city_preds = {}
  for batch, (city, y0, x0) in zip(loader, seg_ds.patches):
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

  for split_name, dset in split_map.items():
    for city in dset.cities:
      if city not in city_preds:
        continue
      sample = dset.load_city(city)
      x1n, vm = apply_normalization(sample.x_pre, stats, valid_mask=sample.valid_mask, nodata_value=0.0)
      x2n, _ = apply_normalization(sample.x_post, stats, valid_mask=sample.valid_mask, nodata_value=0.0)
      rgb_pre = rgb_from_s2(sample.x_pre, band_order, valid_mask=sample.valid_mask)
      rgb_post = rgb_from_s2(sample.x_post, band_order, valid_mask=sample.valid_mask)
      gt = (sample.y[0] > 0) if sample.y is not None else None

      # reconstruct probs from patches
      h, w = sample.x_pre.shape[1:]
      prob_full = np.zeros((h, w), dtype=np.float32)
      counts = np.zeros((h, w), dtype=np.float32)
      for probs, y0, x0 in city_preds[city]:
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
      pred_mask = (prob_full >= 0.5).astype(np.uint8)

      fig, axes = plt.subplots(2, 3, figsize=(12, 8))
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

      im = axes[1, 0].imshow(prob_full, cmap="inferno")
      axes[1, 0].set_title("Prediction prob")
      axes[1, 0].axis("off")
      fig.colorbar(im, ax=axes[1, 0])

      axes[1, 1].imshow(pred_mask, cmap="gray", vmin=0, vmax=1)
      axes[1, 1].set_title("Pred mask (0.5)")
      axes[1, 1].axis("off")

      axes[1, 2].axis("off")
      fig.suptitle(f"{city} ({split_name}) segmentation", fontsize=12)
      fig.tight_layout()
      fig.savefig(out_dir / f"{city}_seg_summary.png", dpi=150)
      plt.close(fig)


if __name__ == "__main__":
  main()
