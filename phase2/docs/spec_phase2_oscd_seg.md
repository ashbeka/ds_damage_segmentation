# Phase 2 – DS / PCA-Diff Priors for Change & Damage-Oriented Segmentation (OSCD-First)

> This is the Phase 2 implementation spec to hand off to Codex (VS Code).
> It assumes Phase 1 is complete as described in your Phase 1 report.
> Core playground = OSCD, with explicit hooks to later plug in a true damage dataset (e.g., xBD / xBD-S12) without redesign.

---

## 0. Scope & Design Philosophy

### 0.1 High-level goal

Phase 2 moves from unsupervised change scores → supervised segmentation:

- On OSCD:
  - Learn pixel-wise binary change segmentation models using:
    - Raw S2 bands only.
    - DS / PCA-diff / pixel-diff priors as extra channels.
    - Combinations of priors (e.g., PCA-diff+DS).
  - Use OSCD primarily as a change detection benchmark but treat this as the first step of a damage-oriented pipeline.

- For future true damage dataset (xBD, xBD-S12, etc.):
  - Design everything so we can swap OSCD → damage dataset with:
    - Minimal changes to model code.
    - New dataset adapter + new config.

Phase 2 does not yet tackle temporal SSC / progression seriously; that’s for Phase 3. Here we focus on:

1. A solid, well-engineered segmentation baseline.
2. Rigorous comparison of feature paths:
   - Path A: Raw S2 only.
   - Path B: Raw S2 + DS or PCA-diff.
   - Path C: Priors only (e.g., PCA-diff alone).
   - Path D: DS + PCA-diff combined.

### 0.2 Constraints & assumptions

- Hardware: no constraints (powerful PC, GPU assumed).

- Phase 1 artifacts:

  - You can recompute or load:

    - oscd_change_maps/{split}/{method}/{city}_score.npy

    - oscd_change_maps/{split}/{method}/{city}_mask.png

  - oscd_eval_results.json & oscd_eval_summary.csv are available.

- Code style:

  - Phase 2 lives in phase2:

  - Clear configs/, data/, models/, train/, eval/, viz/.

  - CLI entry points like python -m train.train_oscd_seg.
---

## 1. Repository Layout for Phase 2

## 1.1 New directory tree (under repo root)
DS_damage_segmentation/
  phase1/
  phase2/
    configs/
      oscd_seg_baseline.yaml
      oscd_seg_priors.yaml
      oscd_seg_ablation.yaml
      damage_dataset_template.yaml      # placeholder for xBD/xBD-S12
    data/
      oscd_seg_dataset.py
      damage_dataset_adapter.py         # stub/template
      transforms.py
    models/
      unet2d.py
      unet2d_resnet_backbone.py         # optional, for stronger baseline
      priors_fusion_heads.py
    train/
      train_oscd_seg.py
      optimizer_schedules.py
      losses.py
      callbacks.py
    eval/
      evaluate_oscd_seg.py
      metrics_segmentation.py
      compare_priors_effect.py
    viz/
      viz_seg_predictions.py
      overlay_utils.py
    docs/
      spec_phase2_oscd_seg.md           # this spec (or final version)

---

## 2. Targets & Tasks

### 2.1 Primary learning task (OSCD)

- **Label:** OSCD binary change mask (`Y_change ∈ {0,1}` per pixel).

- **Input options (feature paths):**

1. **Raw S2 only (baseline)**  
   - Input tensor per pixel:  
     `X_raw = concat([pre_S2_bands, post_S2_bands]) ∈ ℝ^{26}` (13+13).  
   - Model: U-Net-style CNN predicting change probability per pixel.

2. **Raw S2 + single prior**  
   - Variants:
     - **PCA-diff** prior:
       - Add a single channel `pca_diff_score` (Phase 1 saved `pca_diff` score maps).
     - **DS projection** prior:
       - Add `ds_projection_score`.
   - Input shape example: `[26 raw + 1 prior] = 27` channels.

3. **Raw S2 + multiple priors**  
   - Example:
     - `X = [pre(13), post(13), ds_proj(1), pca_diff(1), pixel_diff(1)]` → 29 channels.
   - This is your **PCA-diff + DS** path.

4. **Priors only**  
   - For sanity:
     - e.g. `[ds_proj, pca_diff]` (1 or 2 channels).
   - Tests how strong priors are without raw spectral info.

- **Outputs:**
  - Map `P_change(x) ∈ [0,1]` for each pixel.
  - Optionally: multi-output for change/no-change logits.

---

### 2.2 Future damage-dataset path (design now, implement later)

- Design everything so that for a dataset like **xBD / xBD-S12**:
  - `damage_dataset_adapter.py` produces:
    - `X_pre`, `X_post` (S2-like bands or RGB).
    - `Y_damage` with **multi-class damage levels** or binary damage.
  - The **model definition** is unchanged, only:
    - Number of input channels.
    - Number of classes in the final layer.

We **explicitly note in spec & configs**:

> "Phase 2 uses OSCD as a change benchmark, but is designed to support extension to a true damage dataset (xBD/xBD-S12)."

---

## 3. Data Pipeline – OSCD Segmentation

### 3.1 Dataset class: `OSCDSegmentationDataset`

File: `phase2/data/oscd_seg_dataset.py`

- **Inputs from Phase 1:**
  - Use the **same OSCD root**:
    - `phase1/data/raw/OSCD/…`
  - Reuse Phase 1 loader logic:
    - Call into `phase1/data/oscd_dataset.py` via import, or directly re-implement minimal loader in Phase 2.
  - Access **change maps** (priors) from saved Phase 1 runs:
    - `phase1/outputs/oscd_saved/oscd_change_maps/{split}/{method}/{city}_score.npy`.

- **Responsibilities:**

  1. Given `split ∈ {train,val,test}` and `city`, load:
     - `X_pre, X_post` (normalized 13×H×W).
     - `Y_change` (binary mask 1×H×W).
     - `valid_mask` from Phase 1.

  2. Stack channels according to config:
     - Example config snippet:

       ```yaml
       features:
         use_raw_s2: true
         use_pre_post_stack: true   # pre||post vs diff-only
         priors:
           ds_projection: true
           pca_diff: true
           pixel_diff: false
       ```

     - Use this to decide which channels to include.

  3. If priors enabled:
     - Load `score.npy` for each enabled method.
     - Normalize to [0,1] (if not already).
     - Add as 1×H×W channels.

  4. Tiling / patches:
     - Because OSCD tiles are large and your GPU can handle more, but for batching and augmentation we still use patches:
       - Configurable `patch_size` (e.g., 256×256).
       - Option `overlap` for sliding crops (e.g., 64).
     - Dataset returns **patches**:
       - `X_patch: (C, patch_h, patch_w)`
       - `Y_patch: (1, patch_h, patch_w)`
       - `valid_patch` (same size) for masking.

  5. Augmentations:
     - Implement in `transforms.py`:
       - Random horizontal/vertical flips.
       - Random rotations (90°, 180°, 270°).
       - Optional small scaling / jitter (keep registration).
       - Optional Gaussian noise.
     - All applied consistently to pre/post and priors.

---

### 3.2 Data transforms module

File: `phase2/data/transforms.py`

- Define composable transforms:

  ```python
  class RandomFlipRotate:
      # apply to (X, Y, valid_mask)

  class RandomGaussianNoise:
      # add low-level noise to spectral bands, not to masks

  class NormalizePriors:
      # ensure priors channels are in [0,1]
- Use them in the dataset __getitem__.

### 3.3 Dataloaders

In train/train_oscd_seg.py:

- build_dataloaders(config):

  - train_loader, val_loader.

  - batch_size configurable (start with 8 or 16 patches).

  - num_workers based on CPU.


---

## 4. Models

### 4.1 Core model: U-Net-like 2D segmentation

File: `phase2/models/unet2d.py`

- **Input:** `C_in = N_raw_bands + N_prior_channels`.

- **Output:** one or more logits per pixel (`C_out = 1` for binary).

- **Architecture:**
  - **Encoder:** 4–5 levels:
    - Conv (3×3) → BN → ReLU → Conv → BN → ReLU → MaxPool.
    - Channel progression: 64 → 128 → 256 → 512 → 1024.
  - **Decoder:**
    - UpConv (2×2) → concat with encoder feature → Conv×2 etc.
  - **Final conv:** `Conv(64→1)` + sigmoid (in `forward` or use BCEWithLogitsLoss).

- **Make `C_in` dynamic:**
  - Pass from config or args; internal conv1 uses `in_channels=C_in`.

---

### 4.2 Optional stronger baseline: U-Net with ResNet encoder

File: `phase2/models/unet2d_resnet_backbone.py`

- Use a standard ResNet-34/50 as encoder (e.g. via torchvision).
- Replace first conv to accept `C_in` channels.
- Decoder as above (bilinear upsampling + conv or transposed conv).
- This is a stretch baseline if simple U-Net underperforms.

---

### 4.3 Priors fusion heads (optional advanced)

File: `phase2/models/priors_fusion_heads.py`

- Design for future experiments where priors are processed by their own branch:

```python
class PriorsFusionUNet(nn.Module):
    def __init__(self, n_raw, n_priors, ...):
        # branch for raw S2
        # branch for priors (1x1 or small conv stack)
        # fuse early (concat) or mid-level
```
Not necessary for first implementation, but spec it as Phase 2b.

---

## 5. Training Loop

### 5.1 Training script

File: phase2/train/train_oscd_seg.py

- CLI example:

```bash
python -m train.train_oscd_seg \
  --config configs/oscd_seg_priors.yaml \
  --oscd_root phase1/data/raw/OSCD \
  --phase1_change_maps_root phase1/outputs/oscd_saved/oscd_change_maps \
  --output_dir phase2/outputs/oscd_seg_run1
```
- Responsibilities:

  1. Parse config + CLI overrides.
  2. Build datasets/dataloaders.
  3. Instantiate model with correct C_in.
  4. Define optimizer, scheduler, losses.
  5. Training loop with:
    - Mixed precision (amp) if GPU supports it.
    - Gradient clipping (optional).
    - Periodic validation and checkpointing.

### 5.2 Loss functions

File: phase2/train/losses.py

- Primary losses:
  - BCEWithLogitsLoss for per-pixel binary classification (with mask).
  - Dice loss or Soft IoU to address class imbalance.

- Combined loss:
total_loss = λ_bce * bce_loss + λ_dice * dice_loss

- Start with λ_bce = 1.0, λ_dice = 1.0.
- Optionally add focal loss later

### 5.3 Optimizer & schedule

File: phase2/train/optimizer_schedules.py

- Default:
  - Optimizer: AdamW
    - lr = 1e-3, weight_decay = 1e-4.
  - Scheduler:
    - Cosine annealing or step LR.
    - Or ReduceLROnPlateau on val IoU.
- Config example:

```optimizer:
  name: adamw
  lr: 0.001
  weight_decay: 0.0001

scheduler:
  name: cosine
  T_max: 50
```
### 5.4 Callbacks & training utilities

File: phase2/train/callbacks.py

- Implement:

  - ModelCheckpoint:
    - Save best model based on val IoU (and maybe best AUROC).
  - EarlyStopping (optional):
    - Patience in epochs on val IoU.
  - LoggingCallback:
    - Write metrics to CSV and optionally tensorboard-style logs.

---

## 6. Evaluation for Phase 2

### 6.1 Segmentation metrics module

File: `phase2/eval/metrics_segmentation.py`

- For each predicted mask and ground truth:

  - Binary mask via threshold `p >= 0.5` (configurable).

  - Compute:

    - Pixel metrics:
      - TP/FP/FN/TN, IoU, F1, Precision, Recall.

    - AUROC:
      - Use continuous probabilities vs GT labels.

    - Per-city stats:
      - Aggregate metrics per tile (city).

    - Split-level mean metrics:
      - `mean IoU`, `mean F1`, etc.

- Accept optional `valid_mask` to ignore NODATA.

---

### 6.2 Evaluation script

File: `phase2/eval/evaluate_oscd_seg.py`

- CLI example:

```bash
python -m eval.evaluate_oscd_seg \
  --config configs/oscd_seg_priors.yaml \
  --oscd_root phase1/data/raw/OSCD \
  --phase1_change_maps_root phase1/outputs/oscd_saved/oscd_change_maps \
  --checkpoint phase2/outputs/oscd_seg_run1/best.ckpt \
  --output_dir phase2/outputs/oscd_eval_run1
```
- Responsibilities:

  1. Rebuild dataset and model with the same config.
  2. Load checkpoint.
  3. Run inference on val and test splits.
  4. Compute metrics per tile and per split.
  5. Save:
    - oscd_seg_eval_results.json:
      - split → city → metrics.
    - oscd_seg_eval_summary.csv:
      - split, model_name, features_config, mean_iou, mean_f1, auroc, ....

### 6.3 Priors effect comparison script

File: phase2/eval/compare_priors_effect.py

- Goal: systematic ablation of feature paths.
- This script:
  - Loads multiple oscd_seg_eval_summary.csv files (one per experiment).
  - Extracts metrics for:
    - features_config = "raw_only"
    - features_config = "raw+ds"
    - features_config = "raw+pca"
    - features_config = "raw+ds+pca"
  - Produces a small table + optional bar chart (per metric).
  - Writes to:
    - phase2/outputs/oscd_priors_ablation_summary.csv.
  - Config should contain a human-readable string like:

experiment_tag: "raw+ds+pca"

Saved into run_metadata.json for easy grouping.
---

## 7. Visualization in Phase 2

### 7.1 Prediction overlays on OSCD tiles

File: `phase2/viz/viz_seg_predictions.py`

- **Inputs:**
  - Pre RGB.
  - GT change mask.
  - Predicted probability map.
  - Derived binary mask (thresholded).

- **Outputs:**
  - For each city, composite PNG with panels:
    1. Pre RGB.
    2. Post RGB.
    3. GT change mask overlay.
    4. Prior map (e.g., PCA-diff) from Phase 1.
    5. Model prediction probability map.
    6. Predicted binary mask.
    7. Optional DS map from Phase 1 for side-by-side.

- **CLI example:**

```bash
python -m viz.viz_seg_predictions \
  --config configs/oscd_seg_priors.yaml \
  --oscd_root phase1/data/raw/OSCD \
  --phase1_change_maps_root phase1/outputs/oscd_saved/oscd_change_maps \
  --checkpoint phase2_outputs/oscd_seg_run1/best.ckpt \
  --output_dir phase2_outputs/oscd_seg_figs \
  --cities all
```
### 7.2 Overlay utilities

File: phase2/viz/overlay_utils.py

- Helper functions:
  - rgb_from_s2(bands) (reusing Phase 1’s robust percentile scaling).
  - overlay_mask_on_rgb(rgb, mask, color).
  - tile_panels_to_grid(images).

---

## 8. Config Design

### 8.1 Baseline config (raw S2 only)

File: phase2/configs/oscd_seg_baseline.yaml
Key fields:
```
dataset:
  name: oscd
  root: phase1/data/raw/OSCD
  split:
    train: [beirut, bercy, bordeaux, cupertino, hongkong, mumbai, nantes, paris, pisa, rennes, saclay_e]
    val:   [beirut, cupertino, mumbai]     # example subset
    test:  [brasilia, chongqing, dubai, lasvegas, milano, montpellier, norcia, rio, saclay_w, valencia]
  patch_size: 256
  patch_overlap: 64
  augmentations:
    flip: true
    rotate90: true
    noise: true

features:
  use_raw_s2: true
  use_pre_post_stack: true
  priors:
    ds_projection: false
    pca_diff: false
    pixel_diff: false

model:
  type: unet2d
  base_channels: 64
  depth: 4
  # C_in is inferred from features

training:
  epochs: 50
  batch_size: 8
  optimizer:
    name: adamw
    lr: 0.001
    weight_decay: 0.0001
  scheduler:
    name: cosine
    T_max: 50
  loss:
    bce_weight: 1.0
    dice_weight: 1.0

experiment_tag: "oscd_seg_raw_only"
```

### 8.2 Priors config (raw + DS + PCA-diff)

File: phase2/configs/oscd_seg_priors.yaml

Differences:
```
features:
  use_raw_s2: true
  use_pre_post_stack: true
  priors:
    ds_projection: true
    pca_diff: true
    pixel_diff: false

phase1:
  change_maps_root: phase1/outputs/oscd_saved/oscd_change_maps

experiment_tag: "oscd_seg_raw+ds+pca"
```
### 8.3 Ablation config file

File: phase2/configs/oscd_seg_ablation.yaml

- Optional to define multiple experiments in one YAML and have a meta-script iterate them.

---

## 9. Experimental Plan (Must-Run Experiments)

### 9.1 Core experiments on OSCD

1. **E0 – Raw only baseline**
   - Config: `oscd_seg_baseline.yaml`.
   - Input: [pre, post], 26 channels.
   - Output: baseline IoU/F1/AUROC.

2. **E1 – Raw + DS projection**
   - Config: copy baseline, set `priors.ds_projection = true`.
   - Input: 27 channels (pre, post, ds_projection).
   - Evaluate improvement vs E0.

3. **E2 – Raw + PCA-diff**
   - `priors.pca_diff = true`, `ds_projection = false`.
   - Compare to E1: which prior helps more?

4. **E3 – Raw + DS + PCA-diff (combined path)**
   - `priors.ds_projection = true`, `priors.pca_diff = true`.
   - This is your **“PCA-diff + DS together”** path.
   - Check if combination > max(E1, E2).

5. **E4 – Priors only**
   - `use_raw_s2 = false`, `priors.ds_projection = true`, `priors.pca_diff = true`.
   - Understand pure prior power, and how much raw S2 helps.

For each experiment:
- Run training ≥ 3 seeds if possible (or at least 2) to estimate variance.
- Save metrics in a consistent format.
- Use `compare_priors_effect.py` to summarize.

---

### 9.2 Stretch experiments

- **Stronger model** (ResNet backbone):
  - Repeat E0/E2/E3 with `unet2d_resnet_backbone` to see if priors matter even with a powerful backbone.

- **Pretraining on MultiSenGE (optional)**:
  - Train the U-Net in a self-supervised fashion using DS/PCA-diff maps as pseudo labels on MultiSenGE.
  - Then fine-tune on OSCD.
  - This is a bridge to a future Phase 2b/3.

## 10. Extension to True Damage Dataset (Design Hooks)

### 10.1 Adapter stub

File: phase2/data/damage_dataset_adapter.py

- provide a skeleton class:

```
class DamageDatasetAdapter(torch.utils.data.Dataset):
    def __init__(self, root, split, config):
        # TODO: implement for xBD/xBD-S12
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx):
        # return (X, Y_damage, valid_mask)
        pass
```
- Document where this will be used:
  - Another training script: train/train_damage_seg.py.
  - Config: configs/damage_dataset_template.yaml.

## 10.2 Spec note for future

In the spec & damage template config, explicitly note:

“When a damage-labeled dataset (e.g., xBD-S12) is available, we will (1) adapt band order & normalization to match Phase 1/2, (2) reuse the same U-Net architecture and priors, and (3) extend Y to multi-class damage levels.”

## 11. Phase 2 Completion Criteria

Phase 2 is “complete enough” when:

1. **Code:**

   - `phase2/` compiles and runs end-to-end for OSCD:
     - Training, evaluation, visualization.
   - Configs exist and are documented.

2. **Experiments:**

   - E0–E3 are run at least once with recorded metrics.
   - There is a **comparison table** summarizing:
     - For each experiment: mean IoU, F1, AUROC on OSCD test split.

3. **Artifacts:**

   - `oscd_seg_eval_summary.csv` + `oscd_priors_ablation_summary.csv`.
   - OSCD per-city prediction figures for at least a subset of cities.

4. **Research story:**

   - We can answer clearly:
     - Does PCA-diff prior help?
     - Does DS prior help?
     - Is PCA-diff + DS better than either alone?
   - And we can **reuse the code** for a damage dataset adapter in the next phase.
