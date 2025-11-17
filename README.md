# Phase 1 – Difference-Subspace Change Detection (Sentinel-2)

Implementation repo for **Phase 1** of the project: DS change detection with classical unsupervised baselines on OSCD (quantitative) and MultiSenGE S2 (qualitative). Full requirements live in `docs/spec_phase1_ds_oscd.md`.

## Status
- Implements Phase 1 per spec; code is organized under `configs/`, `ds/`, `baselines/`, `data/`, `eval/`, and `scripts/`.
- Dataset roots discovered locally (please confirm): `data/raw/OSCD/` for OSCD tiles and masks, `data/raw/MultiSenGE/s2/` for Sentinel-2 patches.
- DS sliding window enabled by default; Celik baseline enabled with a reduced (safe) patch size and optional downsample (adjust in configs). MultiSenGE uses 10 S2 bands; OSCD stats are subselected accordingly in viz. MultiSenGE viz emits both PNG and GeoTIFF by default.

## Setup
1. Create/activate Python ≥3.9 env.
2. Install deps: `pip install -r requirements.txt`.

## Running (once components are in place)
- OSCD evaluation: `python -m eval.run_oscd_eval --config configs/oscd_default.yaml --oscd_root <path> --output_dir <dir>`.
- MultiSenGE visualization: `python -m eval.run_multisenge_viz --config configs/multisenge_default.yaml --multisenge_root <path> --output_dir <dir>`.

See the spec for expected behaviors, metrics, and outputs. This README stays practical; methodology and design decisions remain in the spec/notes.


# Phase 1 - to do later