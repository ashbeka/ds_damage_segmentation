"""
Visualize OSCD tiles with ground truth, naive diffs, and DS maps.

Produces multi-panel figures per city:
- Pre RGB
- Post RGB
- GT overlay on pre
- RGB-only pixel diff (L2)
- Full-band pixel diff (L2 on normalized 13 bands)
- DS projection map (current subspace variant)

Optional:
- Annotate per-city metrics if an eval results JSON is provided.

Usage:
    python -m eval.visualize_oscd_examples \
        --config configs/oscd_default.yaml \
        --oscd_root data/raw/OSCD \
        --output_dir outputs/oscd_figs \
        --cities beirut,valencia \
        [--metrics_json outputs/oscd_eval_results.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import yaml

from baselines.pca_diff import pca_diff_score
from baselines.pixel_diff import pixel_l2_difference
from data.oscd_dataset import OSCDEvaluatorDataset
from data.preprocessing import apply_normalization, load_band_stats
from ds.ds_scores import DSConfig, compute_ds_scores
from eval.thresholding import otsu_threshold
from eval.utils import suppress_rasterio_warnings

suppress_rasterio_warnings()


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path, help="OSCD config YAML")
    ap.add_argument("--oscd_root", required=True, type=Path, help="Path to OSCD root")
    ap.add_argument("--output_dir", required=True, type=Path, help="Where to save figures")
    ap.add_argument("--cities", default="test", help="Comma list of cities or one of: train|val|test|all")
    ap.add_argument("--metrics_json", type=Path, default=None, help="Optional oscd_eval_results.json to annotate metrics")
    return ap.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def robust_percentile_scale(arr: np.ndarray, valid_mask: Optional[np.ndarray] = None, p_low=2, p_high=98) -> np.ndarray:
    """Scale an array to [0,1] using robust percentiles over valid pixels."""
    if valid_mask is not None:
        vals = arr[valid_mask]
    else:
        vals = arr if arr.ndim == 1 else arr.reshape(-1, arr.shape[-1]) if arr.ndim == 3 else arr.flatten()
    if vals.size == 0:
        return np.zeros_like(arr, dtype=np.float32)
    lo, hi = np.percentile(vals, [p_low, p_high])
    scaled = (arr - lo) / max(hi - lo, 1e-6)
    return np.clip(scaled, 0, 1).astype(np.float32)


def get_rgb(img: np.ndarray, band_order: List[str], valid_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Extract RGB (B04,B03,B02) from a (C,H,W) cube; mask invalid and scale robustly."""
    idx = {b: i for i, b in enumerate(band_order)}
    r = img[idx["B04"]]
    g = img[idx["B03"]]
    b = img[idx["B02"]]
    rgb = np.stack([r, g, b], axis=-1)
    rgb = robust_percentile_scale(rgb, valid_mask=valid_mask)
    if valid_mask is not None:
        vm3 = valid_mask[..., None]
        rgb = np.where(vm3, rgb, np.nan)
    return rgb


def normalize_map(m: np.ndarray, valid_mask: Optional[np.ndarray] = None) -> np.ndarray:
    return robust_percentile_scale(m.astype(np.float32), valid_mask=valid_mask)


def load_metrics(metrics_json: Optional[Path]) -> Dict[str, Dict[str, dict]]:
    if metrics_json is None or not metrics_json.exists():
        return {}
    with metrics_json.open("r", encoding="utf-8") as f:
        return json.load(f)


def format_metrics(metrics: dict, city: str, method: str) -> str:
    if not metrics:
        return ""
    # metrics structure: split -> method -> per_tile(list)
    for split, methods in metrics.items():
        if method not in methods:
            continue
        per_tile = methods[method].get("per_tile", [])
        for entry in per_tile:
            if entry.get("city") == city:
                m_global = entry.get("global", {})
                return f"F1g {m_global.get('f1', float('nan')):.2f}, IoUg {m_global.get('iou', float('nan')):.2f}"
    return ""


def main():
    args = parse_args()
    cfg = load_config(args.config)
    outdir = args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)

    band_order = cfg["dataset"]["band_order"]
    stats = load_band_stats(Path(cfg["normalization"]["stats_path"]))
    ds_cfg = DSConfig(
        rank_r=cfg["ds"].get("rank_r", 6),
        variance_threshold=cfg["ds"].get("variance_threshold"),
        use_randomized_pca=cfg["ds"].get("use_randomized_pca", True),
        score_normalization=cfg["ds"]["score"].get("normalization", "percentile_99"),
        percentile=cfg["ds"]["score"].get("percentile", 99.0),
        nodata_value=cfg["dataset"].get("nodata_value", 0.0),
        subspace_variant=cfg["ds"].get("subspace_variant", "residual"),
    )

    # Determine city list
    ds_train = OSCDEvaluatorDataset(
        args.oscd_root,
        "train",
        band_order,
        nodata_value=cfg["dataset"].get("nodata_value", 0.0),
        min_valid_bands=cfg["dataset"].get("min_valid_bands", 3),
    )
    ds_val = OSCDEvaluatorDataset(
        args.oscd_root,
        "val",
        band_order,
        nodata_value=cfg["dataset"].get("nodata_value", 0.0),
        min_valid_bands=cfg["dataset"].get("min_valid_bands", 3),
    )
    ds_test = OSCDEvaluatorDataset(
        args.oscd_root,
        "test",
        band_order,
        nodata_value=cfg["dataset"].get("nodata_value", 0.0),
        min_valid_bands=cfg["dataset"].get("min_valid_bands", 3),
    )

    split_map = {"train": ds_train, "val": ds_val, "test": ds_test}
    if args.cities.lower() in ["train", "val", "test", "all"]:
        target_cities: List[str] = []
        if args.cities.lower() == "all":
            for d in split_map.values():
                target_cities.extend(d.cities)
        else:
            target_cities = split_map[args.cities.lower()].cities
    else:
        target_cities = [c.strip() for c in args.cities.split(",") if c.strip()]

    metrics_json = args.metrics_json
    metrics_data = load_metrics(metrics_json) if metrics_json else {}

    # Build index split -> dataset for lookup
    city_to_split = {}
    for split_name, dset in split_map.items():
        for c in dset.cities:
            city_to_split[c] = (split_name, dset)

    for city in target_cities:
        if city not in city_to_split:
            print(f"Skipping unknown city: {city}")
            continue
        split_name, dset = city_to_split[city]
        sample = dset.load_city(city)

        # Raw RGB
        rgb_pre = get_rgb(sample.x_pre, band_order, valid_mask=sample.valid_mask)
        rgb_post = get_rgb(sample.x_post, band_order, valid_mask=sample.valid_mask)

        # Normalize
        x1_norm, vm = apply_normalization(
            sample.x_pre, stats, valid_mask=sample.valid_mask, nodata_value=cfg["dataset"].get("nodata_value", 0.0)
        )
        x2_norm, _ = apply_normalization(
            sample.x_post, stats, valid_mask=sample.valid_mask, nodata_value=cfg["dataset"].get("nodata_value", 0.0)
        )

        # Maps
        rgb_indices = [band_order.index(b) for b in ["B04", "B03", "B02"]]
        rgb_diff = np.linalg.norm(x2_norm[rgb_indices] - x1_norm[rgb_indices], axis=0)
        full_diff = pixel_l2_difference(x1_norm, x2_norm, vm)

        ds_scores = compute_ds_scores(x1_norm, x2_norm, valid_mask=vm, cfg=ds_cfg)
        ds_proj = ds_scores["projection"]

        # Optional: PCA-diff map
        try:
            pca_map = pca_diff_score(x1_norm, x2_norm, vm, variance_threshold=cfg["baselines"]["pca_diff"].get("variance_threshold", 0.95))
            pca_map = normalize_map(pca_map)
        except Exception:
            pca_map = None

        # Ground truth
        gt = sample.y[0] > 0 if sample.y is not None else None

        # Normalize maps for display
        rgb_diff_viz = normalize_map(rgb_diff, valid_mask=vm)
        full_diff_viz = normalize_map(full_diff, valid_mask=vm)
        ds_proj_viz = normalize_map(ds_proj, valid_mask=vm)

        # Thresholded DS (global threshold from config grid not known per-city; use Otsu for visualization)
        ds_mask = ds_proj >= otsu_threshold(ds_proj, vm)

        # Plot
        fig, axes = plt.subplots(3, 3, figsize=(12, 12))
        fig.suptitle(f"{city} ({split_name}) - DS variant: {ds_cfg.subspace_variant}", fontsize=12)

        def imshow(ax, img, title, cmap=None):
            ax.imshow(img, cmap=cmap)
            ax.set_title(title)
            ax.axis("off")

        imshow(axes[0, 0], rgb_pre, "Pre RGB")
        imshow(axes[0, 1], rgb_post, "Post RGB")
        if gt is not None:
            axes[0, 2].imshow(rgb_pre)
            axes[0, 2].imshow(gt, cmap="Reds", alpha=0.4)
            axes[0, 2].set_title("GT overlay")
            axes[0, 2].axis("off")
        else:
            imshow(axes[0, 2], rgb_pre, "GT overlay (none)")

        imshow(axes[1, 0], rgb_diff_viz, "RGB diff (L2)")
        imshow(axes[1, 1], full_diff_viz, "Full-band diff (L2)")
        imshow(axes[1, 2], ds_proj_viz, "DS projection")

        # Third row: DS mask vs GT, and optional PCA-diff
        axes[2, 0].imshow(ds_mask, cmap="gray", vmin=0, vmax=1)
        axes[2, 0].set_title("DS mask (Otsu)")
        axes[2, 0].axis("off")

        if pca_map is not None:
            imshow(axes[2, 1], pca_map, "PCA-diff")
        else:
            axes[2, 1].axis("off")

        # Metrics annotation
        metrics_text = format_metrics(metrics_data, city, "ds_projection")
        axes[2, 2].axis("off")
        axes[2, 2].text(0.05, 0.5, f"Metrics (DS):\n{metrics_text}", fontsize=10)

        fig.tight_layout()
        fig.savefig(outdir / f"{city}_summary.png", dpi=150)
        plt.close(fig)
        print(f"Saved figure for {city} -> {outdir / (city + '_summary.png')}")


if __name__ == "__main__":
    main()
