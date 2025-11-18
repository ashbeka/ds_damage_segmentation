# Phase 2 Run Guide – OSCD Segmentation with DS/PCA Priors

This file is a **practical guide** for Phase 2:

- Which configs correspond to which experiments.
- How to train, evaluate, and visualize.
- How to run ResNet and PriorsFusionUNet variants.
- How to (later) try longer training and ImageNet pretraining.

All commands assume you start in the **Phase 2 directory**:

```bash
cd DS_damage_segmentation/phase2
```

OSCD root and Phase‑1 change maps are always:

- `--oscd_root ../phase1/data/raw/OSCD`
- `--phase1_change_maps_root ../phase1/outputs/oscd_saved/oscd_change_maps`

---

## 1. Configs and Experiments

Key configs in `phase2/configs/`:

- `oscd_seg_baseline.yaml`
  - U‑Net, raw S2 only (`E0`).
- `oscd_seg_E1_raw_ds.yaml`
  - U‑Net, raw + DS prior (`E1`).
- `oscd_seg_E2_raw_pca.yaml`
  - U‑Net, raw + PCA‑diff prior (`E2`).
- `oscd_seg_priors.yaml`
  - U‑Net, raw + DS + PCA‑diff (`E3`).
- `oscd_seg_baseline_resnet.yaml`
  - ResNet‑U‑Net, raw only (stronger baseline).
- `oscd_seg_priors_resnet.yaml`
  - ResNet‑U‑Net, raw + DS + PCA‑diff.
- `oscd_seg_priors_fusion.yaml`
  - PriorsFusionUNet, raw + DS + PCA‑diff (two‑branch fusion).

All configs share:

- `dataset.*` – OSCD splits, patch size, augmentations.
- `features.*` – which priors are enabled.
- `model.*` – model type and channels.
- `training.*` – epochs, batch size, optimizer, scheduler, loss.
- `phase1.change_maps_root` – where priors from Phase 1 live.

---

## 2. Training – Core U‑Net Experiments

### 2.1 E0 – U‑Net raw only

```bash
python -m train.train_oscd_seg \
  --config configs/oscd_seg_baseline.yaml \
  --oscd_root ../phase1/data/raw/OSCD \
  --phase1_change_maps_root ../phase1/outputs/oscd_saved/oscd_change_maps \
  --output_dir outputs/oscd_seg_E0_raw
```

### 2.2 E1 – U‑Net raw + DS

```bash
python -m train.train_oscd_seg \
  --config configs/oscd_seg_E1_raw_ds.yaml \
  --oscd_root ../phase1/data/raw/OSCD \
  --phase1_change_maps_root ../phase1/outputs/oscd_saved/oscd_change_maps \
  --output_dir outputs/oscd_seg_E1_raw_ds
```

### 2.3 E2 – U‑Net raw + PCA‑diff

```bash
python -m train.train_oscd_seg \
  --config configs/oscd_seg_E2_raw_pca.yaml \
  --oscd_root ../phase1/data/raw/OSCD \
  --phase1_change_maps_root ../phase1/outputs/oscd_saved/oscd_change_maps \
  --output_dir outputs/oscd_seg_E2_raw_pca
```

### 2.4 E3 – U‑Net raw + DS + PCA‑diff

```bash
python -m train.train_oscd_seg \
  --config configs/oscd_seg_priors.yaml \
  --oscd_root ../phase1/data/raw/OSCD \
  --phase1_change_maps_root ../phase1/outputs/oscd_saved/oscd_change_maps \
  --output_dir outputs/oscd_seg_E3_raw_ds_pca
```

You can use `--max_epochs N` to do quick test runs without editing the
config.

---

## 3. Training – ResNet and PriorsFusionUNet

### 3.1 ResNet‑U‑Net raw only

Config: `configs/oscd_seg_baseline_resnet.yaml`

```bash
python -m train.train_oscd_seg \
  --config configs/oscd_seg_baseline_resnet.yaml \
  --oscd_root ../phase1/data/raw/OSCD \
  --phase1_change_maps_root ../phase1/outputs/oscd_saved/oscd_change_maps \
  --output_dir outputs/oscd_seg_E0_raw_resnet
```

### 3.2 ResNet‑U‑Net raw + DS + PCA‑diff

Config: `configs/oscd_seg_priors_resnet.yaml`

```bash
python -m train.train_oscd_seg \
  --config configs/oscd_seg_priors_resnet.yaml \
  --oscd_root ../phase1/data/raw/OSCD \
  --phase1_change_maps_root ../phase1/outputs/oscd_saved/oscd_change_maps \
  --output_dir outputs/oscd_seg_E3_raw_ds_pca_resnet
```

### 3.3 PriorsFusionUNet (raw + DS + PCA‑diff)

Config: `configs/oscd_seg_priors_fusion.yaml`

```bash
python -m train.train_oscd_seg \
  --config configs/oscd_seg_priors_fusion.yaml \
  --oscd_root ../phase1/data/raw/OSCD \
  --phase1_change_maps_root ../phase1/outputs/oscd_saved/oscd_change_maps \
  --output_dir outputs/oscd_seg_E3_raw_ds_pca_fusion
```

---

## 4. Evaluation

Script: `phase2/eval/evaluate_oscd_seg.py`

Example: evaluate U‑Net raw‑only model (`E0`) and save metrics:

```bash
python -m eval.evaluate_oscd_seg \
  --config configs/oscd_seg_baseline.yaml \
  --oscd_root ../phase1/data/raw/OSCD \
  --phase1_change_maps_root ../phase1/outputs/oscd_saved/oscd_change_maps \
  --checkpoint outputs/oscd_seg_E0_raw/best.ckpt \
  --output_dir outputs/oscd_seg_E0_raw_eval
```

Same pattern for other runs; just swap config, checkpoint, and output
dir:

- ResNet raw‑only:
  - `--config configs/oscd_seg_baseline_resnet.yaml`
  - `--checkpoint outputs/oscd_seg_E0_raw_resnet/best.ckpt`
  - `--output_dir outputs/oscd_seg_E0_raw_resnet_eval`
- PriorsFusionUNet:
  - `--config configs/oscd_seg_priors_fusion.yaml`
  - `--checkpoint outputs/oscd_seg_E3_raw_ds_pca_fusion/best.ckpt`
  - `--output_dir outputs/oscd_seg_E3_raw_ds_pca_fusion_eval`

Outputs include:

- `oscd_seg_eval_results.json` – per‑city metrics.
- `oscd_seg_eval_summary.csv` – split‑level metrics table.

Use `phase2/eval/compare_priors_effect.py` to aggregate multiple
experiments into a single priors‑ablation CSV.

---

## 5. Visualization

### 5.1 Per‑city segmentation summaries

Script: `phase2/viz/viz_seg_predictions.py`

Example (ResNet raw‑only, test cities):

```bash
python -m viz.viz_seg_predictions \
  --config configs/oscd_seg_baseline_resnet.yaml \
  --oscd_root ../phase1/data/raw/OSCD \
  --phase1_change_maps_root ../phase1/outputs/oscd_saved/oscd_change_maps \
  --checkpoint outputs/oscd_seg_E0_raw_resnet/best.ckpt \
  --output_dir outputs/oscd_seg_figs_E0_resnet \
  --cities test
```

This produces per‑city figures like
`outputs/oscd_seg_figs_E0_resnet/chongqing_seg_summary.png` showing:

- Pre RGB, Post RGB, GT overlay.
- Segmentation probability map and binary mask.

Same command works for other configs (swap config/checkpoint/output).

### 5.2 Combined DS/PCA + segmentation figures

Script: `phase2/viz/viz_oscd_combined.py`

ResNet raw‑only combined figures (test split):

```bash
python -m viz.viz_oscd_combined \
  --config configs/oscd_seg_baseline_resnet.yaml \
  --oscd_root ../phase1/data/raw/OSCD \
  --phase1_change_maps_root ../phase1/outputs/oscd_saved/oscd_change_maps \
  --checkpoint outputs/oscd_seg_E0_raw_resnet/best.ckpt \
  --output_dir outputs/oscd_combined_resnet \
  --cities test
```

PriorsFusionUNet combined figures:

```bash
python -m viz.viz_oscd_combined \
  --config configs/oscd_seg_priors_fusion.yaml \
  --oscd_root ../phase1/data/raw/OSCD \
  --phase1_change_maps_root ../phase1/outputs/oscd_saved/oscd_change_maps \
  --checkpoint outputs/oscd_seg_E3_raw_ds_pca_fusion/best.ckpt \
  --output_dir outputs/oscd_combined_fusion \
  --cities test
```

Each per‑city PNG (e.g.
`outputs/oscd_combined_resnet/brasilia_combined_summary.png`) shows:

- Row 1: Pre RGB, Post RGB, GT overlay.
- Row 2: DS projection, PCA‑diff, segmentation probability.
- Row 3: DS mask (Otsu), segmentation mask (0.5 threshold).

These are the main interpretability tools for comparing DS/PCA priors to
segmentation behavior.

---

## 6. Future Tweaks (longer training, ImageNet)

High‑level ideas are described in detail in
`phase2/docs/phase2_report.md` (Section 7). Short summary:

- **Longer training + more seeds**:
  - Increase `training.epochs` in configs (e.g. 100–150).
  - Re‑run key experiments (E0/E3, ResNet baselines) with new
    `--output_dir` names (e.g. `..._long`, `..._seed1`).
- **ImageNet pretraining (ResNet)**:
  - In `oscd_seg_baseline_resnet.yaml` /
    `oscd_seg_priors_resnet.yaml`, set:
    ```yaml
    model:
      type: unet2d_resnet
      num_classes: 1
      pretrained: true
    ```
  - Re‑run training with a possibly lower learning rate and a new
    `--output_dir` (e.g.
    `outputs/oscd_seg_E0_raw_resnet_imagenet`).
- **MultiSenGE pretraining / advanced priors usage**:
  - Left for a Phase‑2b/3 extension; see the report for the conceptual
    plan (pseudo‑label pretraining, loss weighting, mid‑layer fusion).

This run guide, together with `phase2_report.md`, should give you enough
structure to revisit and extend Phase 2 experiments at any later time
without re‑deriving all the commands.+
