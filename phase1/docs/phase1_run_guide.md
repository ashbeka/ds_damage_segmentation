# Phase 1 Run Guide – DS & Baselines

This file is a **practical cheat‑sheet** for running Phase 1 code
without re‑reading the full report/spec. It focuses on:

- Which configs exist.
- How to run OSCD evaluations.
- How to run MultiSenGE visualizations.
- Where outputs go and how Phase 2 consumes them.

All commands below assume you start from the **Phase 1 directory**:

```bash
cd DS_damage_segmentation/phase1
```

---

## 1. OSCD – DS + Baselines Evaluation

### 1.1 Core script and configs

- Script: `phase1/eval/run_oscd_eval.py`
- Main configs:
  - `phase1/configs/oscd_default.yaml`
    - OSCD dataset paths and splits.
    - DS config (projection/cross‑residual, rank, variant).
    - Baseline toggles (pixel diff, CVA, PCA‑diff, Celik, IR‑MAD).
  - `phase1/configs/oscd_variant_eig.yaml`
    - Same as above but using the **eigen‑based DS** variant.

### 1.2 Typical OSCD evaluation command

Run DS + baselines on OSCD with the default config:

```bash
python -m eval.run_oscd_eval \
  --config configs/oscd_default.yaml \
  --oscd_root data/raw/OSCD \
  --output_dir outputs/oscd_saved
```

Important flags:

- `--no_window`: forces DS to run **without** sliding windows, even if
  the config enables them.
- `--disable_celik`: disables the Celik local PCA+k‑means baseline (it
  is relatively slow).
- `--save_change_maps`: saves per‑tile change maps (scores + binary masks)
  into `output_dir/oscd_change_maps/...` – this is what Phase 2 uses as
  priors.

Example with change‑map saving and Celik disabled:

```bash
python -m eval.run_oscd_eval \
  --config configs/oscd_default.yaml \
  --oscd_root data/raw/OSCD \
  --output_dir outputs/oscd_saved \
  --save_change_maps \
  --disable_celik
```

### 1.3 Outputs

For an `--output_dir` like `phase1/outputs/oscd_saved`, the script
produces:

- `oscd_eval_results.json`
  - Full metrics per split, per method, per tile.
- `oscd_eval_summary.csv`
  - Concise per‑method summary (mean AUROC, F1, IoU, runtimes).
- `run_metadata.json`
  - Config path, OSCD root, git hash, and flags used.
- (If `--save_change_maps` is set)
  - `oscd_change_maps/{split}/{method}/{city}_score.npy`
  - `oscd_change_maps/{split}/{method}/{city}_mask.png`

These change‑map `.npy` files are consumed by Phase 2 as priors
(`ds_projection`, `pca_diff`, etc.).

---

## 2. OSCD – Per‑City DS/PCA Visualization

### 2.1 Script

- Script: `eval/visualize_oscd_examples.py`
- Purpose: create multi‑panel PNGs per city showing:
  - Pre RGB, Post RGB, GT overlay,
  - RGB L2 diff, full‑band diff,
  - DS projection, DS mask, PCA‑diff, and optional metrics.

### 2.2 Example command

```bash
python -m eval.visualize_oscd_examples \
  --config configs/oscd_default.yaml \
  --oscd_root data/raw/OSCD \
  --output_dir outputs/oscd_figs_all \
  --cities test
```

You can also pass a comma‑separated list of cities, e.g.:

```bash
--cities beirut,valencia,chongqing
```

Optional:

- `--metrics_json outputs/oscd_saved/oscd_eval_results.json`
  - Adds per‑city DS metrics text to the figure.

Output example:

- `phase1/outputs/oscd_figs_all/chongqing_summary.png`

---

## 3. MultiSenGE – DS Visualization

### 3.1 Script and config

- Script: `phase1/eval/run_multisenge_viz.py`
- Config: `phase1/configs/multisenge_default.yaml`
  - MultiSenGE root paths.
  - S2 band order and statistics.
  - DS configuration and pair sampling strategy.

### 3.2 Example command

```bash
python -m phase1.eval.run_multisenge_viz \
  --config phase1/configs/multisenge_default.yaml \
  --output_dir phase1/outputs/multisenge_ds_viz
```

Outputs:

- Per‑patch DS score PNGs and GeoTIFFs under
  `phase1/outputs/multisenge_ds_viz/...`.

These are useful for visually understanding DS behavior on many unlabeled
patches; Phase 2 and Phase 3 can re‑use them conceptually as priors or
pretraining targets.

---

## 4. How Phase 2 Uses Phase 1 Outputs

Phase 2 expects the OSCD change maps produced by Phase 1 to live at:

- `phase1/outputs/oscd_saved/oscd_change_maps/{split}/{method}/{city}_score.npy`

and uses them as priors:

- `ds_projection` → DS projection score map.
- `pca_diff` → PCA‑diff score map.
- Optionally `pixel_diff` etc. if enabled.

The path is configured via:

- In Phase 2 configs: `phase1.change_maps_root`
  - e.g. `phase1/outputs/oscd_saved/oscd_change_maps`

As long as you keep Phase 1 outputs in that location, Phase 2 training
and visualization scripts will find them automatically.
