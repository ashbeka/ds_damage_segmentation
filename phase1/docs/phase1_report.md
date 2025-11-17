# Phase 1 – DS‑Only Change Detection on Sentinel‑2 (MultiSenGE + OSCD)

> This document summarizes everything implemented in **Phase 1** of the project:
> the research idea, methods, datasets, code structure, evaluation, and outputs.  
> It is intended as the “handoff spec” for Phase 2 and for future assistants
> who need full context.

---

## 1. High‑Level Goal & Role in the Overall Project

### 1.1 Overall research context

The full research project is about **disaster damage detection**. Eventually we
want robust, label‑efficient segmentation models for damage types and change
types, combining satellite imagery (Sentinel‑2, possibly Sentinel‑1, UAV, etc.)
and classical subspace/matrix methods.

**Phase 1** is intentionally narrow:

- Implement and understand **Difference‑Subspace (DS)** change detection on
  Sentinel‑2.
- Compare DS to **classical unsupervised baselines** (pixel differencing, CVA,
  PCA‑diff, Celik local PCA+k‑means, IR‑MAD).
- Use:
  - **OSCD (Onera Satellite Change Detection)** as the **labeled benchmark** to
    quantify performance (AUROC, F1, IoU, etc.).
  - **MultiSenGE S2** as an **unlabeled testbed** to visualize DS behavior on
    many real patches.

This phase does **not** train deep neural nets. Its outputs (continuous change
maps, DS scores, PCA‑diff maps) are meant to be **priors/inputs for Phase 2**,
where we will build segmentation/damage models.

---

## 2. Repository Layout (Phase 1)

From the parent repo (`DS_damage_segmentation`), Phase 1 lives in `phase1/`.

Important directories and files:

- `phase1/docs/`
  - `spec_phase1_ds_oscd.md` – canonical spec for Phase 1 (datasets, DS math,
    baselines, metrics, milestones).
- `phase1/configs/`
  - `oscd_default.yaml` – OSCD dataset + DS/baseline/threshold/output config
    (residual DS by default).
  - `oscd_variant_eig.yaml` – OSCD config that uses eigen‑based DS variant.
  - `multisenge_default.yaml` – MultiSenGE S2 config (band order, DS settings,
    pair strategy, output formats).
- `phase1/ds/`
  - `pca_utils.py` – PCA/subspace utilities:
    - `fit_pca_basis` – PCA basis (equivalent to lab’s `cvtPCA`).
    - `residual_projector`, `difference_subspace` (Option A DS).
    - `difference_subspace_eig` (Option B DS).
  - `ds_scores.py` – DS scoring:
    - `DSConfig` (rank, variant, normalization).
    - `compute_ds_scores` – DS projection & cross‑residual per tile.
    - `sliding_window_ds` – local DS with aggregation.
- `phase1/baselines/`
  - `pixel_diff.py` – pixel L2 difference.
  - `cva.py` – CVA (same magnitude as pixel_diff, separate name).
  - `pca_diff.py` – PCA on difference image `X2 − X1`.
  - `celik_pca_kmeans.py` – local PCA + k‑means (Celik 2009).
  - `ir_mad.py` – Iteratively Reweighted MAD (stretch baseline).
- `phase1/data/`
  - `oscd_dataset.py` – OSCD dataset loader, official splits, band stacking,
    band‑stats fitting, GT mask loading.
  - `multisenge_dataset.py` – MultiSenGE S2 loader (grouping by patch ID, pair
    strategies).
  - `preprocessing.py` – NODATA masks, bandwise z‑score stats, cube↔matrix
    reshape, and stats reindexing.
- `phase1/eval/`
  - `metrics.py` – binary metrics and AUROC.
  - `thresholding.py` – per‑tile Otsu and global threshold grid search.
  - `run_oscd_eval.py` – main OSCD evaluation driver (metrics + optional
    change‑map saving).
  - `run_multisenge_viz.py` – MultiSenGE DS visualization driver (PNG + GeoTIFF).
  - `visualize_oscd_examples.py` – OSCD per‑city summary figures (pre/post,
    GT, naive diffs, DS maps).
  - `utils.py` – warning suppression for rasterio’s geotransform messages.
- `phase1/outputs/` (ignored by git)
  - Various OSCD eval runs (`oscd_*`) with JSON/CSV.
  - Saved OSCD change maps (`oscd_change_maps`) when enabled.
  - MultiSenGE DS visualizations (`multisenge_viz/`).
  - OSCD per‑city summary figures (`oscd_figs_all/` and `oscd_previews*`).

There is also a `reference_code/` directory at the repo root with:

- `reference_code/DS/` – shape‑based subspace and DS experiments by your
  seniors (on 3D motion data).
- `reference_code/Subspace Toolbox/cvtToolBox/` – MATLAB‑style subspace
  toolbox (PCA, subspace bases, canonical angles, etc.).

These are **read‑only references** for us, not part of Phase 1’s executable
pipeline.

---

## 3. Datasets

### 3.1 OSCD – Labeled Benchmark

- **Tiles:** 24 Sentinel‑2 locations.
- **Data:**
  - Pre and post Sentinel‑2 images, pre‑registered, 13 bands each.
  - Binary change masks (0 = no change, 255 or >0 = change).
- **Representation in code:**
  - Each city → `(X_pre, X_post, Y_change, valid_mask)`:
    - `X_*`: `(13, H, W)` float32.
    - `Y_change`: `(1, H, W)` uint8, values `{0,1}` after binarization.
    - `valid_mask`: `(H, W)` bool (NODATA handling).
- **Splits (encoded in `oscd_dataset.py`):**
  - Train: `beirut, bercy, bordeaux, cupertino, hongkong, mumbai,
             nantes, paris, pisa, rennes, saclay_e`.
  - Test:  `brasilia, chongqing, dubai, lasvegas, milano, montpellier,
             norcia, rio, saclay_w, valencia`.
  - Val: configurable subset of train (`val_cities` or `val_from_train` in
    `configs/oscd_default.yaml`).

**Ground truth masks:**

- Paths like:
  - `phase1/data/raw/OSCD/onera_satellite_change_detection dataset__train_labels/<city>/cm/`
  - `phase1/data/raw/OSCD/onera_satellite_change_detection dataset__test_labels/<city>/cm/`
- Each directory usually contains:
  - `cm.png` – mask with 0 background and 255 change.
  - `<city>-cm.tif` – alternative TIFF mask.
- Loader `_load_mask` prefers the PNG, falls back to TIFF, and binarizes via
  `(mask > 0)`.

### 3.2 MultiSenGE S2 – Unlabeled Testbed

- ~8k multi‑temporal S2 patches (256×256) over Grand‑Est.
- We use S2 L2A only (S1 ignored in Phase 1).
- Each patch has multiple dates; we group them by tile/coords and then select
  pairs (or other sequences) for DS visualization.

**Representation:**

- Each patch path encodes tile, date, and offsets (e.g.
  `31TFN_20200731_S2_4626_514.tif`).
- Loader groups by `(tile, x, y)` and sorts by date.
- Currently we use:
  - Pair strategy `earliest_latest` by default (first vs last date).
  - Alternatives supported: `adjacent`, `first_mid_last`.

---

## 4. Preprocessing & Normalization

### 4.1 Band order

- OSCD and MultiSenGE band orders are encoded in configs:
  - OSCD: `[B01, B02, B03, B04, B05, B06, B07, B08, B09, B10, B11, B12, B8A]`.
  - MultiSenGE (S2 only): `[B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12]`.

### 4.2 NODATA & valid mask

- `build_valid_mask` in `preprocessing.py`:
  - NODATA value configurable (commonly 0).
  - `min_valid_bands` threshold: a pixel must have at least that many bands
    ≠ `nodata_value` to be considered valid.
- `apply_normalization`:
  - Computes valid_mask if not provided.
  - Normalizes only valid pixels and sets invalid to a fill value (0 by
    default) after normalization.

### 4.3 Bandwise z-score

- `fit_or_load_band_stats` in `oscd_dataset.py`:
  - On first use:
    - Scans OSCD train cities (pre & post), stacks all 13 bands, filters
      valid pixels.
    - Computes global per‑band mean/std and writes to
      `data/oscd_band_stats.json`.
  - Later runs:
    - Just load `BandStats` via `load_band_stats`.

- MultiSenGE:
  - Reuses OSCD stats but reindexes them from OSCD’s 13‑band order to
    MultiSenGE’s 10‑band order via a helper.

### 4.4 Reshaping

- `vectorize_cube` / `devectorize_cube` convert:
  - `(C,H,W)` ↔ `(C,N)` (C=d bands, N=H×W), tracking valid pixel indices so
    we can project back to the full image grid.

---

## 5. DS Theory & Implementation

### 5.1 Subspace construction

For a given tile:

- Let `X1, X2 ∈ R^{d×n}` be normalized spectral matrices (d=13 bands), columns
  are pixel vectors at times `t1` and `t2`. We center/standardize bands via the
  global z‑score stats.
- We fit PCA bases:

  ```python
  phi_basis = fit_pca_basis(X1, rank=rank_r or variance_threshold)
  psi_basis = fit_pca_basis(X2, rank=rank_r or variance_threshold)
  Φ = phi_basis.basis  # shape (d, r)
  Ψ = psi_basis.basis  # shape (d, r)
  ```
- fit_pca_basis is a sklearn‑PCA equivalent to the lab’s MATLAB
cvtPCA/cvtBasisVectorSVD:
    - It transposes to (n×d) for sklearn, centers internally, and uses either:
        - fixed rank rank, or
        - energy threshold variance_threshold (e.g. 0.95).

### 5.2 Difference subspace options
Let P_Φ = ΦΦᵀ, P_Ψ = ΨΨᵀ. Residual projectors: R_Φ = I−P_Φ,
R_Ψ = I−P_Ψ.

We support two DS constructions:

Option A – residual‑stacked (Phase 1 default)
In ds/pca_utils.py:
def difference_subspace(phi, psi):
    r_phi = residual_projector(phi)  # I - ΦΦᵀ
    r_psi = residual_projector(psi)  # I - ΨΨᵀ
    stacked = np.concatenate([r_psi @ phi, r_phi @ psi], axis=1)
    return orthonormalize(stacked)   # QR

Interpretation:

- R_ΨΦ = “part of Φ that sticks out of Ψ.”
- R_ΦΨ = “part of Ψ that sticks out of Φ.”
- The difference subspace is generated by these “difference pieces” bundled together and orthonormalized.

This is Option (a) in appendix_ds_math.tex.

Option B – eigen‑based (sum‑of‑projectors, paper/toolbox style)
In ds/pca_utils.py:

def difference_subspace_eig(phi, psi, eps=1e-6):
    P_phi = phi @ phi.T
    P_psi = psi @ psi.T
    G = P_phi + P_psi
    eigvals, eigvecs = np.linalg.eigh(G)
    idx = np.where((eigvals > eps) & (eigvals < 1.0 - eps))[0]
    if idx.size == 0:
        return difference_subspace(phi, psi)
    D = eigvecs[:, idx]
    return orthonormalize(D)

Interpretation:

- G = P_Φ + P_Ψ:
    - Eigenvalues near 2 → common subspace.
    - Eigenvalues in (0,1) → difference subspace (in one, not both).
    - Eigenvalues near 0 → outside both.
- This matches the generalized DS construction from the DS papers and from your seniors’ 3D shape DS code in reference_code/DS/utils.py.

We expose this via DSConfig.subspace_variant:

- residual (default): Option A.
- eig: Option B.

### 5.3 DS scores
Given a pixel pair (x1, x2) and difference subspace D:

- Projection energy (DS projection):

p = || Dᵀ (x2 − x1) ||².

- Cross‑residual (illumination‑robust):

r_cross = || R_Ψ x2 ||² + || R_Φ x1 ||².

Implementation:

- _compute_ds_matrix_scores in ds_scores.py:
    - Builds diff = X2 − X1.
    - Computes proj_coeff = Dᵀ diff and projection_energy = sum(proj_coeff²).
    - Computes cross_residual using residual_projector and cross_residual_energy.
We also support sliding‑window DS via sliding_window_ds, recomputing DS
locally on windows and aggregating (mean or max).

## 6. Baselines
All baselines operate on normalized S2 data, per tile.

- pixel_diff (L2 difference):

    - s_pix = ||x2 − x1||₂ across all bands.
    - We also visualize an RGB‑only diff in the OSCD summary figures (using
[B04,B03,B02]).

- cva (CVA):

    - Same magnitude as pixel_diff (we reuse the same L2 norm), but historically associated with particular thresholding practices. In our pipeline it passes through the same Otsu/global thresholding framework.

- pca_diff:

1. Build D = X2 − X1.
2.2PCA on D, rank S such that ~95% variance is retained.
3. Use magnitude of the projection onto top S PCs as the score.
4. Normalize per tile to [0,1].

- celik_pca_kmeans (Celik 2009):

    - For each pixel:
        - Extract h×h patch from difference image (or stacked bands).
        - Flatten and PCA to a low dimensional subspace (energy ≥ 0.9).
        - Run k‑means with k=2 over projected patch features.
        - The cluster with higher mean score is “change”.
    - Implementation:
        - Patch size (patch_size) and downsampling (downsample_max_side) are configurable to control runtime/memory. 
        - We found Celik can be heavy; downsampling and smaller patches are used for stability.

- ir_mad (IR‑MAD, stretch):

    - Iteratively reweighted canonical variates between X1, X2.
    - Computes MAD variates and reweights based on chi-square distances.
    - Implemented in baselines/ir_mad.py.
    - Disabled by default; when enabled, it tends to be slower and weaker than PCA‑diff/DS on OSCD.


## 7. Thresholding & Metrics

### 7.1 Thresholds
We do not use ground truth for per-tile thresholds (Otsu is purely
unsupervised). We use train labels only for global calibration.

- Otsu per tile (unsupervised):

    - Compute Otsu threshold on the score histogram per tile (on normalized scores).
    - Produces tile‑specific thresholds and masks.

- Global fixed threshold (train‑calibrated):
    - Search over t ∈ [0.05,0.95] with step 0.05.
    - For each method, pick t* that maximizes either IoU or F1 on train.
    - Apply t* to val/test; no leakage from val/test labels into calibration.
Optionally, a manual “fixed thresholds from YAML” mode can override grid search
if we decide to lock in thresholds.

### 7.2 Metrics
Per tile / per split:

- Binary metrics:
    - IoU, F1, Precision, Recall.
- Continuous:
    - AUROC of the raw scores vs GT labels.
- Efficiency:
    - Runtime per tile (approximate, aggregate per method).
Implementation:

- eval/metrics.py:

    - binary_metrics(pred, target, valid_mask) calculates TP/FP/FN/TN and derived metrics.
    - auroc_score(scores, target, valid_mask) uses sklearn metrics.

- run_oscd_eval.py:
    - Aggregates per‑city metrics into means per split/method.
    - Writes:
        - oscd_eval_results.json – full structure, including per‑tile “otsu” and “global” metrics.
        - oscd_eval_summary.csv – compressed view (AUROC/F1/IoU + runtimes).
        - run_metadata.json – config path, data root, output dir, CLI flags, git hash (if available).

## 8. Saving Change Maps
By default, the OSCD eval used to compute scores in memory only. We added an
option to save them per tile/method:

- Controlled by:

    - CLI: --save_change_maps, or
    - Config: output.save_change_maps: true in oscd_default.yaml.

- When enabled:

    - For each split (train,val,test), method, and city:
        - Save score map as .npy:
            - outputs/<run>/oscd_change_maps/{split}/{method}/{city}_score.npy
        - Save mask map as .png:
            - outputs/<run>/oscd_change_maps/{split}/{method}/{city}_mask.png
        - Mask PNG uses the global threshold for that method.

These can be used in Phase 2 as priors/inputs (e.g., joint with RGB bands or as
extra channels).

## 9. Visualization
### 9.1 OSCD Previews (quick‑look)
We created a few quick preview figures for individual cities (e.g.,
outputs/oscd_previews/beirut_preview.png) showing:

- Pre RGB
- DS projection map
- DS Otsu mask

These were for initial sanity checks.

### 9.2 OSCD Summary Figures (new consolidated view)
Script: eval/visualize_oscd_examples.py

Per city (train/val/test or selected subset), it generates a 3×3 panel:

1. Pre RGB (robust percentile scaling, masked; 13‑band S2 → B04/B03/B02).
2. Post RGB.
3. GT overlay on pre (red mask over RGB).
4. RGB diff (L2; B04,B03,B02 only, on normalized data).
5. Full-band diff (L2 over all 13 normalized bands).
6. DS projection map (normalized to [0,1]; current DS variant).
7. DS mask (Otsu), shown as a binary image.
8. PCA‑diff map (if available, normalized).
9. Metrics text (for DS) if an oscd_eval_results.json is provided.

Command example:
python -m eval.visualize_oscd_examples \
  --config configs/oscd_default.yaml \
  --oscd_root data/raw/OSCD \
  --output_dir outputs/oscd_figs_all \
  --cities all \
  --metrics_json outputs/oscd_eval_results.json

These figures directly address the review request to “show the ground truth
alongside difference images and DS maps” and to compare naive diff vs DS.

### 9.3 MultiSenGE DS Maps
Script: eval/run_multisenge_viz.py

Per selected patch pair, saves:

- DS projection map (*_proj.png + *_proj.tif).
- DS cross‑residual map (*_cross.png + *_cross.tif).
- A log file listing patch IDs and dates.

These are used qualitatively to inspect DS behavior on a large variety of
unlabeled S2 scenes.

## 10. Results (OSCD – Example Test Split Summary)
Numbers are from representative runs; exact values may differ slightly as
config/flags change, but the relative ranking is stable.

### 10.1 Residual DS + baselines (full run with Celik and IR‑MAD)
From outputs/oscd_saved/oscd_eval_summary.csv (test split):

- ds_projection:
    - AUROC ≈ 0.755
    - F1 (Otsu) ≈ 0.275, IoU (Otsu) ≈ 0.178
    - F1 (global) ≈ 0.236, IoU (global) ≈ 0.142
- ds_cross_residual:
    - AUROC ≈ 0.527; F1/IoU much lower – weaker than projection.
- pixel_diff / cva:
    - AUROC ≈ 0.756
    - F1 (Otsu) ≈ 0.258, IoU ≈ 0.172
    - Global F1 ≈ 0.157, IoU ≈ 0.088
- pca_diff:
    - AUROC ≈ 0.813 (strongest baseline).
    - F1 (Otsu) ≈ 0.277, IoU ≈ 0.185.
    - Global F1 ≈ 0.235, IoU ≈ 0.144.
- celik:
    - AUROC ≈ 0.65–0.66 depending on downsample.
    - F1/IoU slightly below DS/PCA‑diff but still meaningful.
- ir_mad (when enabled):
    - AUROC ≈ 0.70; F1/IoU lower than DS/PCA‑diff.
    - Serves mostly as an extra classical reference.

### 10.2 Eig vs residual DS (Option B vs Option A)
Using configs/oscd_variant_eig.yaml and outputs/oscd_eig_nowindow:

- With sliding window off and Celik disabled:
    - ds_projection (eig):
        - AUROC ≈ 0.75506 (nearly identical to residual).
        - F1/IoU Otsu and global also differ only at 3rd decimal place.
- Conclusion: on OSCD, the eig‑based DS and residual‑stacked DS behave
numerically almost identically, confirming that both constructions are
equivalent in practice for this dataset.

## 11. “Naive pixel differencing vs DS” – Key Takeaways
From the summary figures (outputs/oscd_figs_all/) and metrics:

- RGB diff:

    - Intuitive but uses only 3 bands.
    - Works well when change is dominantly visible in RGB (e.g., strong contrast differences).
    - Sensitive to illumination / color balancing issues; ignores NIR/SWIR spectral differences.
- Full-band pixel diff:

    - Benefits from all 13 bands.
    - Captures changes in NIR/SWIR that RGB misses (vegetation, soil moisture, etc.).
    - Still treats each band equally and doesn’t exploit subspace structure.

- DS / PCA-diff:

    - Align better with GT in many OSCD tiles:
        - PCA‑diff has the highest AUROC overall.
        - DS projection is competitive with pixel diff and sometimes cleaner, especially in reducing noise/outliers.
    - They model structure in the spectral space:
        - DS uses subspaces and difference subspaces to focus on directions where the spectral manifolds changed.
        - PCA‑diff finds dominant directions of change in the difference image.
These visuals + metrics motivate why we move beyond naive RGB diff and why DS /
PCA‑diff are valuable for Phase 2.

## 12. How to Reproduce Phase 1
From the repo root:
python -m venv .venv
# activate .venv...
pip install -r phase1/requirements.txt
cd phase1

OSCD eval (residual DS + baselines):
python -m eval.run_oscd_eval \
  --config configs/oscd_default.yaml \
  --oscd_root data/raw/OSCD \
  --output_dir outputs/oscd_run

Optional: save change maps:
python -m eval.run_oscd_eval \
  --config configs/oscd_default.yaml \
  --oscd_root data/raw/OSCD \
  --output_dir outputs/oscd_saved \
  --save_change_maps

OSCD eval with eig DS (Option B):
python -m eval.run_oscd_eval \
  --config configs/oscd_variant_eig.yaml \
  --oscd_root data/raw/OSCD \
  --output_dir outputs/oscd_eig_nowindow \
  --disable_celik \
  --no_window

MultiSenGE DS visualization:
python -m eval.run_multisenge_viz \
  --config configs/multisenge_default.yaml \
  --multisenge_root data/raw/MultiSenGE/s2 \
  --output_dir outputs/multisenge_viz

OSCD summary figures for all cities:
python -m eval.visualize_oscd_examples \
  --config configs/oscd_default.yaml \
  --oscd_root data/raw/OSCD \
  --output_dir outputs/oscd_figs_all \
  --cities all \
  --metrics_json outputs/oscd_run/oscd_eval_results.json

## 13. Phase 1 Status & Phase 2 Handoff
Status:

- DS and classical baselines are implemented and validated against OSCD.
- DS is implemented in both residual‑stacked and eig (paper/toolbox) forms.
- Metrics and visualizations are in place to illustrate:
        - How DS compares to naive diff, PCA‑diff, Celik, and IR‑MAD.
        - How DS behaves qualitatively on OSCD and MultiSenGE.

Handoff for Phase 2:

- Inputs available:
    - OSCD change maps (score + mask per method/tile).
    - DS, PCA‑diff, pixel diff maps on MultiSenGE S2.
- Phase 2 can:
    - Use DS/PCA‑diff as extra channels or priors for damage/change segmentation models.
    - Design SSC/temporal methods (e.g., geodesic DS, multi‑date DS acceleration) on top of the existing DS machinery.
    - Leverage the per‑tile figures and metrics as baseline performance/context.

This document should give Phase 2 (and any new assistant) enough context to
understand exactly what Phase 1 implemented, why, and how to build on top of it.