"""
Train OSCD segmentation model for Phase 2 (spec Section 5).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict

import torch
import yaml
from torch.utils.data import DataLoader

try:
  from tqdm import tqdm
except Exception:  # pragma: no cover
  def tqdm(x, *args, **kwargs):
    return x

from data.oscd_seg_dataset import OSCDSegmentationDataset
from models.unet2d import UNet2D
from models.unet2d_resnet_backbone import UNet2DResNetBackbone
from models.priors_fusion_heads import PriorsFusionUNet
from train.losses import BCEDiceLoss
from train.optimizer_schedules import build_optimizer, build_scheduler
from train.callbacks import ModelCheckpoint, MetricsLogger
from eval.metrics_segmentation import auroc_score, binary_stats


def parse_args():
  ap = argparse.ArgumentParser()
  ap.add_argument("--config", type=Path, required=True)
  ap.add_argument("--oscd_root", type=Path, required=True)
  ap.add_argument("--phase1_change_maps_root", type=Path, required=True)
  ap.add_argument("--output_dir", type=Path, required=True)
  ap.add_argument("--max_epochs", type=int, default=None, help="Optional cap on number of epochs (for quick tests).")
  return ap.parse_args()


def load_config(path: Path) -> Dict:
  with path.open("r", encoding="utf-8") as f:
    return yaml.safe_load(f)


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


def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
  model.eval()
  all_f1 = []
  all_iou = []
  all_au = []
  with torch.no_grad():
    for batch in loader:
      x = batch["x"].to(device)
      y = batch["y"].to(device)
      valid = batch["valid"].to(device)
      logits = model(x)
      if logits.shape[-2:] != y.shape[-2:]:
        logits = torch.nn.functional.interpolate(
          logits, size=y.shape[-2:], mode="bilinear", align_corners=False
        )
      probs = torch.sigmoid(logits)
      probs_np = probs.squeeze(1).cpu().numpy()
      y_np = y.squeeze(1).cpu().numpy()
      valid_np = valid.squeeze(1).cpu().numpy()
      pred_mask = (probs_np >= 0.5).astype("uint8")
      stats = binary_stats(pred_mask, y_np, valid_np)
      all_f1.append(stats["f1"])
      all_iou.append(stats["iou"])
      all_au.append(auroc_score(probs_np, y_np, valid_np))
  model.train()
  return {
    "f1": float(sum(all_f1) / max(len(all_f1), 1)),
    "iou": float(sum(all_iou) / max(len(all_iou), 1)),
    "auroc": float(sum(all_au) / max(len(all_au), 1)),
  }


def main():
  args = parse_args()
  cfg = load_config(args.config)
  out_dir = args.output_dir
  out_dir.mkdir(parents=True, exist_ok=True)

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  counts = infer_channel_counts(cfg)
  in_ch = counts["total"]
  model = build_model(cfg, in_ch).to(device)

  train_ds = OSCDSegmentationDataset(
    args.oscd_root,
    "train",
    cfg,
    phase1_change_maps_root=args.phase1_change_maps_root,
  )
  val_ds = OSCDSegmentationDataset(
    args.oscd_root,
    "val",
    cfg,
    phase1_change_maps_root=args.phase1_change_maps_root,
  )

  num_workers = cfg["training"].get("num_workers", 0)
  train_loader = DataLoader(train_ds, batch_size=cfg["training"]["batch_size"], shuffle=True, num_workers=num_workers)
  val_loader = DataLoader(val_ds, batch_size=cfg["training"]["batch_size"], shuffle=False, num_workers=num_workers)

  optimizer = build_optimizer(model.parameters(), cfg["training"]["optimizer"])
  scheduler = build_scheduler(optimizer, cfg["training"]["scheduler"])

  loss_cfg = cfg["training"]["loss"]
  criterion = BCEDiceLoss(bce_weight=loss_cfg.get("bce_weight", 1.0), dice_weight=loss_cfg.get("dice_weight", 1.0))

  ckpt_cb = ModelCheckpoint(out_dir, monitor="val_iou", mode="max")
  log_cb = MetricsLogger(out_dir)

  epochs = cfg["training"]["epochs"]
  if args.max_epochs is not None:
    epochs = min(epochs, args.max_epochs)
  for epoch in range(1, epochs + 1):
    model.train()
    start_epoch = time.perf_counter()
    train_loss_vals = []
    for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False):
      x = batch["x"].to(device)
      y = batch["y"].to(device)
      valid = batch["valid"].to(device)
      optimizer.zero_grad()
      logits = model(x)
      if logits.shape[-2:] != y.shape[-2:]:
        logits = torch.nn.functional.interpolate(
          logits, size=y.shape[-2:], mode="bilinear", align_corners=False
        )
      loss, _, _ = criterion(logits, y, valid)
      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
      optimizer.step()
      train_loss_vals.append(loss.item())
    train_loss = float(sum(train_loss_vals) / max(len(train_loss_vals), 1))

    val_metrics = evaluate(model, val_loader, device)
    epoch_time = time.perf_counter() - start_epoch
    train_metrics = {"loss": train_loss}
    log_cb.log(epoch, train_metrics, val_metrics)
    ckpt_cb.step({"val_iou": val_metrics["iou"]}, model, optimizer, epoch)
    if scheduler is not None:
      if cfg["training"]["scheduler"].get("name", "cosine").lower() == "plateau":
        scheduler.step(val_metrics["iou"])
      else:
        scheduler.step()

    print(
      f"Epoch {epoch}/{epochs} "
      f"- train_loss: {train_loss:.4f} "
      f"- val_iou: {val_metrics['iou']:.3f} "
      f"- val_f1: {val_metrics['f1']:.3f} "
      f"- time: {epoch_time:.1f}s"
    )

  meta = {
    "config": str(args.config),
    "oscd_root": str(args.oscd_root),
    "phase1_change_maps_root": str(args.phase1_change_maps_root),
  }
  with (out_dir / "run_metadata.json").open("w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)


if __name__ == "__main__":
  main()
