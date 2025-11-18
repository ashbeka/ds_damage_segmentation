# Phase 1 – DS-Only Change Detection on Sentinel-2 (MultiSenGE + OSCD)

**Links to matrix IDs:**  
- Core: D11-EXP-003, D11-METH-SUB-003..006, D11-METH-SUB-011, D11-METRIC-003  
- Baselines: D11-EXP-005, gaps: “Simple DS evaluation on a candidate dataset”, “Unsupervised CD baselines (CVA, PCA-diff, IR-MAD)”, “Celik 2009 baseline”

---

## 1. Scope and Goal

**Goal (Phase 1):**  
Implement and evaluate a **label-free Difference-Subspace (DS) change-detection module** for Sentinel-2 imagery, using:

- **MultiSenGE** as an unlabeled testbed to visualize DS behavior on multi-temporal patches. :contentReference[oaicite:0]{index=0}  
- **Onera Satellite Change Detection (OSCD)** as the main labeled benchmark (13-band S2 pairs + binary change masks). :contentReference[oaicite:1]{index=1}  

We compare DS against classical **unsupervised change detection baselines**:

- Pixel differencing (bandwise)  
- CVA (Change Vector Analysis)  
- PCA-diff (PCA on difference image)  
- Celik 2009 PCA + k-means (local PCA on h×h windows + k=2 clustering) :contentReference[oaicite:2]{index=2}  
- (Stretch) IR-MAD if time permits

Outputs:

- **Continuous change maps** and **thresholded binary masks** for OSCD  
- **Quantitative metrics:** AUROC, F1, IoU, Precision, Recall, runtime per tile (for OSCD)  
- **Qualitative DS maps** over MultiSenGE patches (for understanding behavior, not metrics)

This phase **does not** train deep nets; it prepares the DS module and baselines that will later act as priors/inputs for segmentation (Phase 2).

---

## 2. Datasets and Splits

### 2.1 OSCD (primary benchmark)

- 24 Sentinel-2 locations (pairs of pre/post S2 tiles, 13 bands each) with **registered** images and binary change masks (0 = no change, 255 = change). :contentReference[oaicite:3]{index=3}  
- Use the **official train/val/test split** from the OSCD authors/baseline repo (or TorchGeo defaults if they match). :contentReference[oaicite:4]{index=4}  

**Working representation:**

- Keep all 13 bands; resample to a common grid if needed (dataset already provides registered pairs).  
- Store each pair as `(X_pre, X_post, Y_change)` with shapes:  
  - `X_*`: 13 × H × W (float32)  
  - `Y_change`: 1 × H × W (uint8, {0,1})

**Config parameters (OSCD):**

- `bands_oscd`: `["B01", ..., "B12"]` (full set, we can later decide band subsets for ablations)  
- `tile_size`: use full OSCD tile size (no tiling) for baseline; allow optional internal tiling for memory (`512 × 512`, stride 256).  
- `split_source`: `"official_oscd_split"`  

### 2.2 MultiSenGE (unlabeled dev/testbed)

- ~8,157 patches of **256×256 S2 L2A** (and S1 VV/VH) over Grand-Est; we will **only use S2** for Phase 1. :contentReference[oaicite:5]{index=5}  
- Multi-temporal patches with multiple dates per location; metadata files provide acquisition dates and tile IDs. :contentReference[oaicite:6]{index=6}  

**Usage in Phase 1:**

- Select a small **subset** of patches (e.g., 50–200) with at least 2–3 dates each.  
- For each patch, construct 2-date pairs (t1,t2) or 3-date triplets (t1,t2,t3) to test:
  - DS projection-energy maps
  - Cross-residual maps
  - (Later) second-order DS / subspace “velocity” and “acceleration”  

There is **no ground truth**; we only store DS maps as GeoTIFF/PNG and visually inspect them.

**Config parameters (MultiSenGE):**

- `patch_size`: 256 × 256  
- `n_dates_required`: ≥ 2  
- `dates_per_side`: choose earliest vs latest dates in patch or domain-specific windows (e.g., first/last N dates)  
- `max_patches_for_phase1`: e.g., 200 (to limit runtime)

---

## 3. Preprocessing and Normalization

**Common steps (OSCD + MultiSenGE):**

1. **Load and stack bands** in fixed order `[B1,...,B12]` (or dataset’s standard 13-band order).  
2. **Mask invalid pixels / NODATA** using dataset masks if provided; else treat zero-only pixels as candidate NODATA, but keep this heuristic configurable.
3. **Bandwise z-score normalization:**
   - Compute global mean and std per band on **OSCD train** (and a small subset of MultiSenGE)  
   - Apply same `(x - μ)/σ` normalization to all datasets.
4. Optional **SCL / cloud mask** if available:
   - For OSCD, if no SCL is provided, leave cloud handling for later ablations; Phase 1 focuses on base DS behavior.
5. **Vectorization:**
   - For each pair, reshape 13×H×W → 13×N matrix (`N = H×W`), keeping track of 2D indices for re-projection.

Config block (pseudo-YAML):

```yaml
normalization:
  type: zscore
  fit_dataset: oscd_train
  ignore_nodata: true

nodata:
  enabled: true
  value: 0
  min_valid_bands: 3
```
## 4. Methods

### 4.1 Difference-Subspace (DS) Change Detection

**Inputs:**  
Normalized matrices \(X_1, X_2 \in \mathbb{R}^{d \times n}\) (d = 13 bands, n = valid pixels) for times \(t_1, t_2\).

---

#### **Step 1 — PCA Bases**

Compute rank-r PCA for each time:

- \(X_1 = U_1 \Sigma_1 V_1^\top,\quad \Phi = U_1[:, :r]\)
- \(X_2 = U_2 \Sigma_2 V_2^\top,\quad \Psi = U_2[:, :r]\)

Choose r using either:
- energy retention ≥ 95%, or  
- a fixed small value (e.g., \(r = 5\) to \(8\)).

Randomized / incremental PCA is allowed for large tiles.

---

#### **Step 2 — Difference Subspace (Residual Stacking)**  
(see *appendix_ds_math.tex*)

Residual projectors:

- \(P_\Phi = \Phi \Phi^\top,\quad P_\Psi = \Psi \Psi^\top\)
- \(R_\Phi = I - P_\Phi,\quad R_\Psi = I - P_\Psi\)

Difference subspace:

- \(D = \mathrm{orth}([\,R_\Psi \Phi,\; R_\Phi \Psi\,])\)

(QR or SVD may be used for orthonormalization.)

---

#### **Step 3 — Pixel-wise Scores**

For each pixel vector \((x_1, x_2)\):

**(a) Projection-energy score**
\[
p = \left\| D^\top (x_2 - x_1) \right\|^2
\]

**(b) Cross-residual score** (illumination-robust)
\[
r_{\text{cross}} = \| R_\Psi x_2 \|^2 + \| R_\Phi x_1 \|^2
\]

---

#### **Step 4 — Score Normalization**

Normalize scores to [0, 1] per tile using:
- min-max scaling, or  
- percentile clipping (99th percentile recommended).

---

#### **Step 5 — Sliding-Window DS (Local DS)** *(optional but recommended)*

Window parameters:
- window size \(h \in \{64, 96\}\)
- stride \(s \in \{32, 48\}\)

Procedure for each window \(w\):

1. Extract local matrices \(X_1(w)\), \(X_2(w)\).  
2. Recompute Steps 1–3 → obtain \(p_w\), \(r_{\text{cross},w}\).  
3. Aggregate scores at each pixel across overlapping windows  
   using **mean** or **max** fusion.

---

#### **Default Hyperparameters**

```yaml
ds:
  rank_r: 6
  window:
    enabled: true
    size: 64
    stride: 32
  score:
    use_projection: true
    use_cross_residual: true
    normalize: percentile_99
```
### 4.2 Baselines

All baselines operate on **normalized Sentinel-2 data**, per tile.

---

#### **1. Pixel Differencing (L2 norm)**

For each pixel:

\[
s_{\text{pix}} = \| x_2 - x_1 \|_2
\]

Simple magnitude of spectral difference.

---

#### **2. CVA (Change Vector Analysis)**

Same magnitude as pixel differencing:

\[
s_{\text{cva}} = \| x_2 - x_1 \|_2
\]

But thresholding is done using:

- **Otsu thresholding**, or  
- Gaussian assumption on unchanged pixels.

---

#### **3. PCA-diff**

1. Build a difference matrix/image:
   \[
   D = X_2 - X_1
   \]
2. Apply PCA to \(D\) (rank \(S\) capturing ~95% variance; typically \(S \approx 3\)–\(5\)).  
3. Use:
   - either the first principal component  
   - or the combined magnitude across top PCs  

Normalize final score to [0, 1].

---

#### **4. Celik 2009 — Local PCA + k-means**

This is the classical and widely cited **local PCA + k-means** change detector.

For each pixel:

1. Extract an \(h \times h\) **local patch** from the difference image  
   (or from stacked \((X_1, X_2)\) bands).
2. Flatten patch → vector.  
3. Project onto a **local** PCA space of dimension \(S\)  
   (energy ≥ 90%).  
4. Run **k-means with \(k=2\)** over all projected vectors in the tile.  
5. Assign the cluster with **higher mean score** as “change”.

**Recommended defaults (to validate empirically):**

- Window size \(h \in \{7, 9, 11\}\)  
- PCA energy ≥ 0.90  
- \(S\) determined by energy threshold

---

#### **5. IR-MAD (Stretch Goal)**

- Optional  
- Implement only if time allows  
- Otherwise mark as “future baseline”

---

#### **Baseline Hyperparameter Configuration**

```yaml
baselines:
  pixel_diff:
    enabled: true

  cva:
    enabled: true

  pca_diff:
    enabled: true
    rank_S: 3

  celik:
    enabled: true
    patch_size: 9
    pca_energy: 0.9
    kmeans_init: "k-means++"
    max_iter: 100

  ir_mad:
    enabled: false   # stretch goal
```
## 5. Thresholding and Calibration (OSCD Only)

All change-score maps (DS and baselines) are treated as **continuous change indices**.  
To obtain binary maps for OSCD, we consider two thresholding strategies:

---

### **5.1 Per-Tile Otsu Thresholding (Unsupervised)**

- Compute Otsu threshold **independently per tile**.  
- Produces a binary mask:
  \[
  \hat{Y} = \mathbf{1}\big(s \ge t_{\text{otsu}}\big)
  \]

This method does not use any training labels.

---

### **5.2 Global Threshold Estimated from OSCD Train**

A single global threshold \(t^\*\) is selected using **only the OSCD train split**.

Two estimation options:

#### **Option A — Logistic / Temperature Calibration**
Fit a simple calibration function (e.g., logistic) using train labels to map raw scores → calibrated probabilities.  
Select the threshold that maximizes F1 or IoU.

#### **Option B — Grid Search (Recommended, Simple & Robust)**

Search thresholds:
\[
t \in [0.05, 0.95] \text{ with step } 0.05
\]

Choose \(t^\*\) that maximizes:
- F1 score, or  
- IoU score  

on **OSCD train**.

---

### **5.3 Evaluation Protocol**

- **Train split** is used **only** for threshold calibration.  
- Evaluate the calibrated threshold on **val** and **test** splits.  
- For DS and all baselines, also report **AUROC** (threshold-free evaluation).

---

### **5.4 Thresholding Configuration**

```yaml
thresholding:
  methods:
    - name: otsu_per_tile

    - name: global_fixed
      source: oscd_train
      criterion: f1        # or iou
      grid:
        min: 0.05
        max: 0.95
        step: 0.05

  primary_metric: "iou"
```
## 6. Metrics and Logging

### **OSCD (Labeled Benchmark)**

#### **Binary Metrics**
- IoU  
- F1  
- Precision  
- Recall  

#### **Continuous Metrics**
- AUROC for:
  - DS projection score  
  - DS cross-residual score  
  - All baselines  

#### **Compute Metrics**
- Runtime per tile  
- (Optional) Peak memory usage  

---

### **MultiSenGE (Unlabeled Testbed)**

- No quantitative metrics  
- Save DS maps as:
  - PNG  
  - or GeoTIFF  with colorbars
- Maintain:
  - A visual gallery  
  - Parameter logs (JSON/YAML)

---

## 7. Implementation Details (for the Implementation Repo)

### **Software Stack**

- Python ≥ 3.9  
- `numpy`  
- `scikit-learn`  
- `scipy`  
- `rasterio`  
- `matplotlib`  

**Optional:**  
- `torchgeo` (for OSCD loaders)
- huggingface datasets for OSCD_MSI variant

---

### **Suggested Repository Structure**

ds-phase1/
  configs/
    oscd_default.yaml
    multisenge_default.yaml
  ds/
    __init__.py
    pca_utils.py        # PCA, whitening, incremental PCA
    ds_scores.py        # DS projection, cross-residual, sliding-window DS
  baselines/
    __init__.py
    pixel_diff.py
    cva.py
    pca_diff.py
    celik_pca_kmeans.py
  data/
    __init__.py
    oscd_dataset.py     # wrappers around OSCD
    multisenge_dataset.py
  eval/
    __init__.py
    metrics.py
    thresholding.py
    run_oscd_eval.py
    run_multisenge_viz.py
  scripts/
    __init__.py
    make_oscd_figs.sh
    make_multisenge_gallery.sh
  requirements.txt
  README.md
---

### **Reproducibility**
- Global random seed (e.g., 1234) for PCA sampling, k-means, etc.
- Log config + git hash per run in a simple JSON log file.

---

## 8. Milestones for Phase 1

### **M1 – Data Sanity Check**
- Download OSCD and a small MultiSenGE subset.  
- Confirm:
  - band ordering  
  - array shapes  
  - basic per-band statistics  
  - presence/handling of NODATA  

---

### **M2 – DS Core Implementation**
- Implement **global PCA + DS + projection score** for a single tile.  
- Visualize DS maps on:
  - 1–2 OSCD tiles  
  - a few MultiSenGE patches  

---

### **M3 – Baselines Implementation**
- Implement:
  - Pixel differencing  
  - CVA  
  - PCA-diff  
  - Celik (local PCA + k-means)  
- Run baselines on 1–2 OSCD tiles.  
- Verify outputs visually.

---

### **M4 – Full OSCD Evaluation**
- Run **DS + all baselines** on all OSCD **train/val/test** tiles.  
- Calibrate thresholds using **train** split.  
- Report:
  - AUROC  
  - F1  
  - IoU  
  - Precision  
  - Recall  
  - Runtime per tile  
- Evaluate on **val** and **test** splits.

---

### **M5 – MultiSenGE Gallery**
- Run DS on selected MultiSenGE patches.  
- Save:
  - DS maps (PNG / GeoTIFF)  
  - A visual gallery  
- Add brief qualitative observations (patterns, stability, temporal behavior).

---

### **M6 – Feedback Loop Into Notes Repo**
- Summarize:
  - quantitative results  
  - qualitative insights  
- Update:
  - `coverage_matrix.csv` (mark IDs resolved/in progress)  
  - `gaps_towatch.md` (close or refine items)  
- Add any new insights as small ADRs if necessary.

