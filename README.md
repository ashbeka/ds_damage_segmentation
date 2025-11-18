# Phase 1 – Difference-Subspace Change Detection (Sentinel-2)

Implementation repo for **Phase 1**: DS change detection with classical unsupervised baselines on OSCD (quantitative) and MultiSenGE S2 (qualitative). Full requirements are in `docs/spec_phase1_ds_oscd.md`.

## What’s here
- Core method: Difference-Subspace (DS) change detector.
- Baselines: pixel diff, CVA, PCA-diff, Celik (local PCA+k-means), IR-MAD (implemented, off by default).
- Datasets: OSCD (labeled) for metrics; MultiSenGE S2 (unlabeled) for qualitative viz.

## Layout
- `configs/`: default hyperparams for OSCD and MultiSenGE.
- `ds/`: PCA helpers and DS scoring.
- `baselines/`: classical baselines.
- `data/`: dataset loaders and preprocessing (band stats, masking).
- `eval/`: metrics, thresholding, OSCD eval + MultiSenGE viz CLIs.
- `scripts/`: optional helpers.
- `outputs/`: results (ignored by git).

## Setup
1. From repo root: create/activate Python ≥3.9 env (e.g., `.venv`).
2. Install deps: `pip install -r phase1/requirements.txt`.

## How to run
- OSCD eval (DS + baselines, Celik on, IR-MAD off by default):
  ```
  python -m eval.run_oscd_eval --config configs/oscd_default.yaml --oscd_root data/raw/OSCD --output_dir outputs/oscd_run
  ```
  Flags: `--no_window` (speed), `--disable_celik` (speed), enable IR-MAD in config if desired.
- MultiSenGE viz (qualitative DS maps, PNG + GeoTIFF):
  ```
  python -m eval.run_multisenge_viz --config configs/multisenge_default.yaml --multisenge_root data/raw/MultiSenGE/s2 --output_dir outputs/multisenge_viz
  ```

## Visualization utilities
- OSCD example figures (pre/post RGB, GT overlay, RGB/full diff, DS map, DS mask, optional PCA-diff):
  ```
  python -m eval.visualize_oscd_examples \
    --config configs/oscd_default.yaml \
    --oscd_root data/raw/OSCD \
    --output_dir outputs/oscd_figs \
    --cities beirut,valencia \
    --metrics_json outputs/oscd_eval_results.json  # optional
  ```
- MultiSenGE DS maps: see `outputs/multisenge_viz/` (PNG + GeoTIFF) after running the viz script above.

## Notes
- OSCD official splits are encoded; band orders fixed (13-band OSCD, 10-band MultiSenGE).
- Band stats fitted on OSCD train; reused for all runs.
- Previews for a few OSCD cities are in `outputs/oscd_previews/` (ignored by git).
- Rasterio geotransform warnings are suppressed in runners.

See the spec for full methodology; this README stays practical.
