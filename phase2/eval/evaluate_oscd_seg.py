"""
Evaluate trained OSCD segmentation models (spec Section 6.2).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from data.oscd_seg_dataset import OSCDSegmentationDataset
from models.unet2d import UNet2D
from models.unet2d_resnet_backbone import UNet2DResNetBackbone
from models.priors_fusion_heads import PriorsFusionUNet
from eval.metrics_segmentation import auroc_score, binary_stats


def parse_args():
  ap = argparse.ArgumentParser()
  ap.add_argument("--config", type=Path, required=True)
  ap.add_argument("--oscd_root", type=Path, required=True)
  ap.add_argument("--phase1_change_maps_root", type=Path, required=True)
  ap.add_argument("--checkpoint", type=Path, required=True)
  ap.add_argument("--output_dir", type=Path, required=True)
  return ap.parse_args()


def load_config(path: Path) -> Dict:
  with path.open("r", encoding="utf-8") as f:
    return yaml.safe_load(f)


def build_model(cfg: Dict, in_channels: int) -> torch.nn.Module:
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


def infer_channel_counts(cfg: Dict) -> Dict[str, int]:
  feats = cfg["features"]
  n_raw = 0
  if feats.get("use_raw_s2", True):
    if feats.get("use_pre_post_stack", True):
      n_raw += 26
    else:
      n_raw += 13
  priors = feats.get("priors", {})
  n_priors = 0
  for key, enabled in priors.items():
    if enabled:
      n_priors += 1
  return {"n_raw": n_raw, "n_priors": n_priors, "total": n_raw + n_priors}


def main():
  args = parse_args()
  cfg = load_config(args.config)
  out_dir = args.output_dir
  out_dir.mkdir(parents=True, exist_ok=True)

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  counts = infer_channel_counts(cfg)
  in_ch = counts["total"]
  model = build_model(cfg, in_ch).to(device)

  ckpt = torch.load(args.checkpoint, map_location=device)
  model.load_state_dict(ckpt["model_state"])
  model.eval()

  results: Dict[str, Dict[str, Dict]] = {}

  for split in ["val", "test"]:
    ds = OSCDSegmentationDataset(
      args.oscd_root,
      split,
      cfg,
      phase1_change_maps_root=args.phase1_change_maps_root,
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    split_metrics: Dict[str, Dict] = {}
    city_stats: Dict[str, Dict[str, list]] = {}

    for batch in loader:
      city = batch["city"]
      if isinstance(city, (list, tuple)):
        city = city[0]
      x = batch["x"].to(device)
      y = batch["y"].to(device)
      valid = batch["valid"].to(device)
      with torch.no_grad():
        logits = model(x)
        if logits.shape[-2:] != y.shape[-2:]:
          logits = torch.nn.functional.interpolate(
            logits, size=y.shape[-2:], mode="bilinear", align_corners=False
          )
        probs = torch.sigmoid(logits)
      probs_np = probs.squeeze(1).cpu().numpy()
      y_np = y.squeeze(1).cpu().numpy()
      valid_np = valid.squeeze(1).cpu().numpy()
      pred_mask = (probs_np >= 0.5).astype(np.uint8)
      stats = binary_stats(pred_mask, y_np, valid_np)
      au = auroc_score(probs_np, y_np, valid_np)
      stats["auroc"] = au
      if city not in city_stats:
        city_stats[city] = {"f1": [], "iou": [], "auroc": []}
      city_stats[city]["f1"].append(stats["f1"])
      city_stats[city]["iou"].append(stats["iou"])
      city_stats[city]["auroc"].append(stats["auroc"])

    # aggregate per city
    split_res = {}
    for city, vals in city_stats.items():
      split_res[city] = {k: float(np.mean(v)) for k, v in vals.items()}
    # aggregate mean over cities
    if split_res:
      mean_iou = float(np.mean([v["iou"] for v in split_res.values()]))
      mean_f1 = float(np.mean([v["f1"] for v in split_res.values()]))
      mean_auroc = float(np.mean([v["auroc"] for v in split_res.values()]))
    else:
      mean_iou = mean_f1 = mean_auroc = float("nan")
    results[split] = {
      "per_city": split_res,
      "summary": {"mean_iou": mean_iou, "mean_f1": mean_f1, "mean_auroc": mean_auroc},
    }

  with (out_dir / "oscd_seg_eval_results.json").open("w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)


if __name__ == "__main__":
  main()
