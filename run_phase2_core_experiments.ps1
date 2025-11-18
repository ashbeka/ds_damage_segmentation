# run_phase2_core_experiments.ps1
Set-Location "E:\research_projects\DS_damage_segmentation\phase2"

function Run-CommandWithRetry {
    param(
        [string]$Command,
        [int]$MaxRetries = 1,
        [string]$Name = ""
    )
    $attempt = 0
    while ($attempt -le $MaxRetries) {
        $attemptNum = $attempt + 1
        Write-Host "[$Name] Running (attempt $attemptNum/$($MaxRetries+1)):`n $Command" -ForegroundColor Cyan
        # Use & to invoke the command line
        cmd.exe /c $Command
        if ($LASTEXITCODE -eq 0) {
            Write-Host "[$Name] Succeeded." -ForegroundColor Green
            return $true
        } else {
            Write-Host "[$Name] Failed with exit code $LASTEXITCODE." -ForegroundColor Yellow
        }
        $attempt++
    }
    Write-Host "[$Name] Giving up after $($MaxRetries+1) attempts." -ForegroundColor Red
    return $false
}

# Helper to run one experiment (train + eval)
function Run-Experiment {
    param(
        [string]$TrainCmd,
        [string]$EvalCmd,
        [string]$Tag
    )

    $okTrain = Run-CommandWithRetry -Command $TrainCmd -MaxRetries 1 -Name "$Tag train"
    if (-not $okTrain) {
        Write-Host "[$Tag] Training failed after retries; skipping evaluation and moving to next experiment." -ForegroundColor Red
        return
    }

    $okEval = Run-CommandWithRetry -Command $EvalCmd -MaxRetries 1 -Name "$Tag eval"
    if (-not $okEval) {
        Write-Host "[$Tag] Evaluation failed after retries; moving to next experiment." -ForegroundColor Red
    }
}

# E0 – raw only
$trainE0 = 'python -m train.train_oscd_seg --config configs/oscd_seg_baseline.yaml --oscd_root ../phase1/data/raw/OSCD --phase1_change_maps_root ../phase1/outputs/oscd_saved/oscd_change_maps --output_dir ../phase2_outputs/oscd_seg_E0_raw'
$evalE0  = 'python -m eval.evaluate_oscd_seg --config configs/oscd_seg_baseline.yaml --oscd_root ../phase1/data/raw/OSCD --phase1_change_maps_root ../phase1/outputs/oscd_saved/oscd_change_maps --checkpoint ../phase2_outputs/oscd_seg_E0_raw/best.ckpt --output_dir ../phase2_outputs/oscd_seg_E0_raw_eval'
Run-Experiment -TrainCmd $trainE0 -EvalCmd $evalE0 -Tag "E0_raw_only"

# E1 – raw + DS
$trainE1 = 'python -m train.train_oscd_seg --config configs/oscd_seg_E1_raw_ds.yaml --oscd_root ../phase1/data/raw/OSCD --phase1_change_maps_root ../phase1/outputs/oscd_saved/oscd_change_maps --output_dir ../phase2_outputs/oscd_seg_E1_raw_ds'
$evalE1  = 'python -m eval.evaluate_oscd_seg --config configs/oscd_seg_E1_raw_ds.yaml --oscd_root ../phase1/data/raw/OSCD --phase1_change_maps_root ../phase1/outputs/oscd_saved/oscd_change_maps --checkpoint ../phase2_outputs/oscd_seg_E1_raw_ds/best.ckpt --output_dir ../phase2_outputs/oscd_seg_E1_raw_ds_eval'
Run-Experiment -TrainCmd $trainE1 -EvalCmd $evalE1 -Tag "E1_raw+ds"

# E2 – raw + PCA-diff
$trainE2 = 'python -m train.train_oscd_seg --config configs/oscd_seg_E2_raw_pca.yaml --oscd_root ../phase1/data/raw/OSCD --phase1_change_maps_root ../phase1/outputs/oscd_saved/oscd_change_maps --output_dir ../phase2_outputs/oscd_seg_E2_raw_pca'
$evalE2  = 'python -m eval.evaluate_oscd_seg --config configs/oscd_seg_E2_raw_pca.yaml --oscd_root ../phase1/data/raw/OSCD --phase1_change_maps_root ../phase1/outputs/oscd_saved/oscd_change_maps --checkpoint ../phase2_outputs/oscd_seg_E2_raw_pca/best.ckpt --output_dir ../phase2_outputs/oscd_seg_E2_raw_pca_eval'
Run-Experiment -TrainCmd $trainE2 -EvalCmd $evalE2 -Tag "E2_raw+pca"

# E3 – raw + DS + PCA-diff
$trainE3 = 'python -m train.train_oscd_seg --config configs/oscd_seg_priors.yaml --oscd_root ../phase1/data/raw/OSCD --phase1_change_maps_root ../phase1/outputs/oscd_saved/oscd_change_maps --output_dir ../phase2_outputs/oscd_seg_E3_raw_ds_pca'
$evalE3  = 'python -m eval.evaluate_oscd_seg --config configs/oscd_seg_priors.yaml --oscd_root ../phase1/data/raw/OSCD --phase1_change_maps_root ../phase1/outputs/oscd_saved/oscd_change_maps --checkpoint ../phase2_outputs/oscd_seg_E3_raw_ds_pca/best.ckpt --output_dir ../phase2_outputs/oscd_seg_E3_raw_ds_pca_eval'
Run-Experiment -TrainCmd $trainE3 -EvalCmd $evalE3 -Tag "E3_raw+ds+pca"

Write-Host "All scheduled experiments attempted." -ForegroundColor Cyan
