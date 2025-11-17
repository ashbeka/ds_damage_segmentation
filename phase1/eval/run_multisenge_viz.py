"""
Run DS visualization on MultiSenGE S2 patches (spec Section 4.1 & 6).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
import rasterio
from eval.utils import suppress_rasterio_warnings
from data.preprocessing import reindex_stats

suppress_rasterio_warnings()

from data.multisenge_dataset import load_pair, scan_multisenge_s2, select_pairs
from data.preprocessing import BandStats, apply_normalization, load_band_stats
from ds.ds_scores import DSConfig, compute_ds_scores, sliding_window_ds


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--multisenge_root", required=True, type=Path)
    ap.add_argument("--output_dir", required=True, type=Path)
    return ap.parse_args()


def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_png(score: np.ndarray, out_path: Path, title: str):
    plt.figure(figsize=(6, 6))
    plt.imshow(score, cmap="inferno")
    plt.title(title)
    plt.axis("off")
    plt.colorbar()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def subset_stats(stats: BandStats, stats_order, desired_order) -> BandStats:
    """Reindex band stats from stats_order to desired_order (subset)."""
    idx_map = {b: i for i, b in enumerate(stats_order)}
    mean = []
    std = []
    for b in desired_order:
        if b not in idx_map:
            raise ValueError(f"Band {b} not found in stats_order")
        mean.append(stats.mean[idx_map[b]])
        std.append(stats.std[idx_map[b]])
    return BandStats(mean=np.array(mean, dtype=np.float32), std=np.array(std, dtype=np.float32), eps=stats.eps)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    stats = load_band_stats(Path(cfg["normalization"]["stats_path"]))
    desired_order = cfg["dataset"].get("band_order")
    stats_order = cfg["normalization"].get("stats_band_order", desired_order)
    if desired_order is not None and stats_order is not None and stats_order != desired_order:
        stats = subset_stats(stats, stats_order, desired_order)
    samples = scan_multisenge_s2(args.multisenge_root)
    pairs = select_pairs(
        samples,
        n_dates_required=cfg["dataset"].get("n_dates_required", 2),
        max_patches=cfg["dataset"].get("max_patches_for_phase1"),
        pair_strategy=cfg["dataset"].get("pair_strategy", "earliest_latest"),
    )

    ds_cfg = DSConfig(
        rank_r=cfg["ds"].get("rank_r", 6),
        variance_threshold=cfg["ds"].get("variance_threshold"),
        use_randomized_pca=cfg["ds"].get("use_randomized_pca", True),
        score_normalization=cfg["ds"]["score"].get("normalization", "percentile_99"),
        percentile=cfg["ds"]["score"].get("percentile", 99.0),
        nodata_value=cfg["dataset"].get("nodata_value", 0.0),
    )
    window_cfg = cfg["ds"].get("window", {})

    log_entries = []
    for patch_id, t1, t2 in pairs:
        x1, x2, mask = load_pair(
            t1,
            t2,
            nodata_value=cfg["dataset"].get("nodata_value", 0.0),
            min_valid_bands=cfg["dataset"].get("min_valid_bands", 3),
        )
        x1n, vm = apply_normalization(x1, stats, valid_mask=mask, nodata_value=cfg["dataset"].get("nodata_value", 0.0))
        x2n, _ = apply_normalization(x2, stats, valid_mask=mask, nodata_value=cfg["dataset"].get("nodata_value", 0.0))
        if window_cfg.get("enabled", False):
            ds_out = sliding_window_ds(
                x1n,
                x2n,
                window_size=window_cfg.get("size", 64),
                stride=window_cfg.get("stride", 32),
                aggregator=window_cfg.get("aggregator", "mean"),
                cfg=ds_cfg,
            )
        else:
            ds_out = compute_ds_scores(x1n, x2n, valid_mask=vm, cfg=ds_cfg)

        proj = ds_out["projection"]
        cross = ds_out["cross_residual"]

        formats = cfg.get("output", {}).get("formats", ["png"])
        if "png" in formats:
            save_png(proj, outdir / f"{patch_id}_proj.png", f"{patch_id} DS projection")
            save_png(cross, outdir / f"{patch_id}_cross.png", f"{patch_id} DS cross residual")
        if "geotiff" in formats:
            meta = {
                "driver": "GTiff",
                "dtype": "float32",
                "count": 1,
                "width": proj.shape[1],
                "height": proj.shape[0],
            }
            with rasterio.open(outdir / f"{patch_id}_proj.tif", "w", **meta) as dst:
                dst.write(proj.astype(np.float32), 1)
            with rasterio.open(outdir / f"{patch_id}_cross.tif", "w", **meta) as dst:
                dst.write(cross.astype(np.float32), 1)
        log_entries.append({"patch_id": patch_id, "t1": str(t1), "t2": str(t2)})

    with open(outdir / "multisenge_viz_log.json", "w", encoding="utf-8") as f:
        json.dump(log_entries, f, indent=2)


if __name__ == "__main__":
    main()
