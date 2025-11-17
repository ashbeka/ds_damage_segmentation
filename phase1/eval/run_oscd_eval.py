"""
Run OSCD evaluation for DS + baselines (spec Sections 4–6).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import yaml

from baselines.celik_pca_kmeans import celik_score
from baselines.cva import cva_score
from baselines.pca_diff import pca_diff_score
from baselines.pixel_diff import pixel_l2_difference
from data.oscd_dataset import OSCDEvaluatorDataset, fit_or_load_band_stats
from data.preprocessing import BandStats, apply_normalization
from ds.ds_scores import DSConfig, compute_ds_scores, sliding_window_ds
from eval.metrics import auroc_score, binary_metrics
from eval.thresholding import apply_threshold, grid_search_threshold, otsu_threshold


def git_hash() -> str:
    try:
        import subprocess

        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode("utf-8")
            .strip()
        )
    except Exception:
        return "unknown"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--oscd_root", required=True, type=Path)
    ap.add_argument("--output_dir", required=True, type=Path)
    ap.add_argument("--no_window", action="store_true", help="Disable DS sliding window regardless of config.")
    ap.add_argument("--disable_celik", action="store_true", help="Disable Celik baseline regardless of config.")
    return ap.parse_args()


def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def run_methods_on_tile(
    x1: np.ndarray,
    x2: np.ndarray,
    valid_mask: np.ndarray,
    ds_cfg: DSConfig,
    baseline_cfg: dict,
    ds_window_cfg: dict,
    disable_celik: bool = False,
) -> Dict[str, np.ndarray]:
    scores = {}
    # DS
    if ds_cfg:
        if ds_window_cfg.get("enabled", False):
            sw = ds_window_cfg
            ds_out = sliding_window_ds(
                x1,
                x2,
                window_size=int(sw.get("size", 64)),
                stride=int(sw.get("stride", 32)),
                aggregator=sw.get("aggregator", "mean"),
                cfg=ds_cfg,
            )
        else:
            ds_out = compute_ds_scores(x1, x2, valid_mask=valid_mask, cfg=ds_cfg)
        scores["ds_projection"] = ds_out["projection"]
        scores["ds_cross_residual"] = ds_out["cross_residual"]
    # Baselines
    if baseline_cfg.get("pixel_diff", {}).get("enabled", True):
        scores["pixel_diff"] = pixel_l2_difference(x1, x2, valid_mask)
    if baseline_cfg.get("cva", {}).get("enabled", True):
        scores["cva"] = cva_score(x1, x2, valid_mask)
    if baseline_cfg.get("pca_diff", {}).get("enabled", True):
        pca_args = baseline_cfg["pca_diff"]
        scores["pca_diff"] = pca_diff_score(
            x1,
            x2,
            valid_mask,
            rank_S=pca_args.get("rank_S"),
            variance_threshold=pca_args.get("variance_threshold", 0.95),
        )
    if baseline_cfg.get("celik", {}).get("enabled", True) and not disable_celik:
        ck = baseline_cfg["celik"]
        scores["celik"] = celik_score(
            x1,
            x2,
            patch_size=ck.get("patch_size", 9),
            pca_energy=ck.get("pca_energy", 0.9),
            kmeans_init=ck.get("kmeans_init", "k-means++"),
            max_iter=ck.get("max_iter", 100),
            valid_mask=valid_mask,
            downsample_max_side=ck.get("downsample_max_side"),
        )
    return scores


def _positions(length: int, window: int, stride: int):
    positions = list(range(0, max(1, length - window + 1), stride))
    last = length - window
    if positions:
        if positions[-1] != last:
            positions.append(max(0, last))
    else:
        positions = [0]
    return positions


def eval_split(
    dataset: OSCDEvaluatorDataset,
    stats: BandStats,
    cfg: dict,
    force_no_window: bool = False,
    disable_celik: bool = False,
) -> Dict[str, dict]:
    ds_cfg = DSConfig(
        rank_r=cfg["ds"].get("rank_r", 6),
        variance_threshold=cfg["ds"].get("variance_threshold"),
        use_randomized_pca=cfg["ds"].get("use_randomized_pca", True),
        score_normalization=cfg["ds"]["score"].get("normalization", "percentile_99"),
        percentile=cfg["ds"]["score"].get("percentile", 99.0),
        nodata_value=cfg["dataset"].get("nodata_value", 0.0),
    )
    results = {}
    for sample in dataset:
        x1_norm, vm1 = apply_normalization(
            sample.x_pre,
            stats,
            valid_mask=sample.valid_mask,
            nodata_value=cfg["dataset"].get("nodata_value", 0.0),
        )
        x2_norm, _ = apply_normalization(
            sample.x_post,
            stats,
            valid_mask=sample.valid_mask,
            nodata_value=cfg["dataset"].get("nodata_value", 0.0),
        )
        methods_runtime = {}
        methods_scores = {}
        tiling_cfg = cfg["dataset"].get("internal_tiling", {})
        if tiling_cfg.get("enabled", False):
            h, w = x1_norm.shape[1:]
            tile_sz = int(tiling_cfg.get("tile_size", 512))
            stride = int(tiling_cfg.get("stride", tile_sz))
            counts = np.zeros((h, w), dtype=np.float32)
            accum: Dict[str, np.ndarray] = {}
            runtimes: Dict[str, float] = {}
            for y in _positions(h, tile_sz, stride):
                for x in _positions(w, tile_sz, stride):
                    sl_y = slice(y, y + tile_sz)
                    sl_x = slice(x, x + tile_sz)
                    patch_vm = vm1[sl_y, sl_x]
                    start = time.perf_counter()
                    patch_scores = run_methods_on_tile(
                        x1_norm[:, sl_y, sl_x],
                        x2_norm[:, sl_y, sl_x],
                        valid_mask=patch_vm,
                        ds_cfg=ds_cfg,
                        baseline_cfg=cfg["baselines"],
                        ds_window_cfg=cfg["ds"].get("window", {}) if not force_no_window else {"enabled": False},
                        disable_celik=disable_celik,
                    )
                    elapsed = time.perf_counter() - start
                    for m, s in patch_scores.items():
                        if m not in accum:
                            accum[m] = np.zeros((h, w), dtype=np.float32)
                            runtimes[m] = 0.0
                        accum[m][sl_y, sl_x] += s
                        runtimes[m] += elapsed
                    counts[sl_y, sl_x] += 1.0
            for m, acc in accum.items():
                methods_scores[m] = np.divide(acc, counts, out=np.zeros_like(acc), where=counts > 0)
                methods_runtime[m] = runtimes[m]
        else:
            start = time.perf_counter()
            scores = run_methods_on_tile(
                x1_norm,
                x2_norm,
                valid_mask=vm1,
                ds_cfg=ds_cfg,
                baseline_cfg=cfg["baselines"],
                ds_window_cfg=cfg["ds"].get("window", {}) if not force_no_window else {"enabled": False},
                disable_celik=disable_celik,
            )
            runtime_tile = time.perf_counter() - start
            for m, s in scores.items():
                methods_scores[m] = s
                methods_runtime[m] = runtime_tile

        results[sample.city] = {"scores": methods_scores, "runtimes": methods_runtime, "label": sample.y, "mask": vm1}
    return results


def aggregate_metrics(
    split_results: Dict[str, dict],
    threshold_cfg: dict,
) -> Dict[str, dict]:
    methods = next(iter(split_results.values()))["scores"].keys()
    out: Dict[str, dict] = {m: {} for m in methods}
    # Collect train data for calibration if available
    return out


def main():
    args = parse_args()
    cfg = load_config(args.config)
    ensure_dir(args.output_dir)
    stats = fit_or_load_band_stats(
        args.oscd_root,
        cfg["dataset"]["band_order"],
        Path(cfg["normalization"]["stats_path"]),
        nodata_value=cfg["dataset"].get("nodata_value", 0.0),
        min_valid_bands=cfg["dataset"].get("min_valid_bands", 3),
    )

    datasets = {
        "train": OSCDEvaluatorDataset(
            args.oscd_root,
            "train",
            cfg["dataset"]["band_order"],
            nodata_value=cfg["dataset"].get("nodata_value", 0.0),
            min_valid_bands=cfg["dataset"].get("min_valid_bands", 3),
            stats_path=cfg["normalization"]["stats_path"],
            val_cities=cfg["dataset"].get("val_cities"),
            val_from_train=cfg["dataset"].get("val_from_train", 0),
        ),
        "val": OSCDEvaluatorDataset(
            args.oscd_root,
            "val",
            cfg["dataset"]["band_order"],
            nodata_value=cfg["dataset"].get("nodata_value", 0.0),
            min_valid_bands=cfg["dataset"].get("min_valid_bands", 3),
            stats_path=cfg["normalization"]["stats_path"],
            val_cities=cfg["dataset"].get("val_cities"),
            val_from_train=cfg["dataset"].get("val_from_train", 0),
        ),
        "test": OSCDEvaluatorDataset(
            args.oscd_root,
            "test",
            cfg["dataset"]["band_order"],
            nodata_value=cfg["dataset"].get("nodata_value", 0.0),
            min_valid_bands=cfg["dataset"].get("min_valid_bands", 3),
            stats_path=cfg["normalization"]["stats_path"],
            val_cities=cfg["dataset"].get("val_cities"),
            val_from_train=cfg["dataset"].get("val_from_train", 0),
        ),
    }

    split_results: Dict[str, Dict[str, dict]] = {}
    for split_name, ds in datasets.items():
        print(f"Processing split: {split_name} ({len(ds)} tiles)")
        split_results[split_name] = eval_split(ds, stats, cfg, force_no_window=args.no_window, disable_celik=args.disable_celik)

    # Threshold calibration and metric aggregation
    threshold_cfg = cfg["thresholding"]
    grid = (
        threshold_cfg["methods"][1]["grid"]["min"],
        threshold_cfg["methods"][1]["grid"]["max"],
        threshold_cfg["methods"][1]["grid"]["step"],
    )
    primary_metric = threshold_cfg.get("primary_metric", "iou")

    summary: Dict[str, dict] = {}
    train_res = split_results["train"]
    methods = next(iter(train_res.values()))["scores"].keys()

    global_thresholds: Dict[str, float] = {}
    for method in methods:
        train_scores = [train_res[c]["scores"][method] for c in train_res]
        train_labels = [train_res[c]["label"] for c in train_res]
        valid_masks = [train_res[c]["mask"] for c in train_res]
        thr = grid_search_threshold(train_scores, train_labels, valid_masks, grid, criterion=primary_metric)
        global_thresholds[method] = thr

    summary: Dict[str, dict] = {}
    for split_name, res in split_results.items():
        split_metrics: Dict[str, dict] = {}
        for method in methods:
            aurocs = []
            per_tile = []
            runtimes = []
            f1_otsu = []
            iou_otsu = []
            f1_global = []
            iou_global = []
            for city, rec in res.items():
                score = rec["scores"][method]
                y = rec["label"]
                m = rec["mask"]
                if y is not None:
                    au = auroc_score(score, y, valid_mask=m)
                    aurocs.append(au)
                    # per-tile Otsu
                    t_otsu = otsu_threshold(score, m)
                    pred_o = apply_threshold(score, t_otsu)
                    metrics_o = binary_metrics(pred_o, y, valid_mask=m)
                    f1_otsu.append(metrics_o["f1"])
                    iou_otsu.append(metrics_o["iou"])
                    # global threshold
                    t_g = global_thresholds[method]
                    pred_g = apply_threshold(score, t_g)
                    metrics_g = binary_metrics(pred_g, y, valid_mask=m)
                    f1_global.append(metrics_g["f1"])
                    iou_global.append(metrics_g["iou"])
                    per_tile.append({"city": city, "otsu": metrics_o, "global": metrics_g})
                runtimes.append(rec["runtimes"][method])
            split_metrics[method] = {
                "auroc_mean": float(np.nanmean(aurocs)) if aurocs else float("nan"),
                "f1_otsu_mean": float(np.mean(f1_otsu)) if f1_otsu else float("nan"),
                "iou_otsu_mean": float(np.mean(iou_otsu)) if iou_otsu else float("nan"),
                "f1_global_mean": float(np.mean(f1_global)) if f1_global else float("nan"),
                "iou_global_mean": float(np.mean(iou_global)) if iou_global else float("nan"),
                "runtime_avg_sec": float(np.mean(runtimes)) if runtimes else 0.0,
                "per_tile": per_tile,
                "threshold_global": global_thresholds[method],
            }
        summary[split_name] = split_metrics

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    out_path = out_root / "oscd_eval_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    # concise CSV summary
    csv_lines = ["split,method,auroc_mean,f1_otsu_mean,iou_otsu_mean,f1_global_mean,iou_global_mean,runtime_avg_sec,threshold_global"]
    for split_name, split_metrics in summary.items():
        for method, vals in split_metrics.items():
            csv_lines.append(
                f"{split_name},{method},{vals['auroc_mean']},{vals['f1_otsu_mean']},{vals['iou_otsu_mean']},{vals['f1_global_mean']},{vals['iou_global_mean']},{vals['runtime_avg_sec']},{vals['threshold_global']}"
            )
    with open(out_root / "oscd_eval_summary.csv", "w", encoding="utf-8") as f:
        f.write("\n".join(csv_lines))
    # metadata log
    meta = {
        "config": str(args.config),
        "oscd_root": str(args.oscd_root),
        "output_dir": str(args.output_dir),
        "git_hash": git_hash(),
        "args": {
            "no_window": args.no_window,
            "disable_celik": args.disable_celik,
        },
    }
    with open(out_root / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved results to {out_path} and summary CSV to oscd_eval_summary.csv")


if __name__ == "__main__":
    main()
