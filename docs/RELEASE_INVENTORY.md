# LRF-IMU Release Inventory

## Summary

This inventory covers the read-only research tree at:

`D:/PAPER2/CODEX_CLEANUP - Copy/BACKUP THE NIGHT BEFORE SUBMISSION/VERSION AT THE TIME OF SUBMISSION/LRF-IMU-P3`

The release-audit workspace is:

`C:/Users/AminR/Documents/Codex/2026-07-30/we-are-preparing-the-accepted-lrf`

The accepted paper is reported by the user as published in `Machine Learning: Health` with DOI `10.1088/3049-477X/ae91ef`. Publisher-side verification of this DOI was not established from the web search performed in this session, so the DOI is recorded as user supplied.

No experiment logic, preprocessing code, checkpoints, generated datasets, result files, or existing configuration files were modified during this audit.

## Repository Structure

Important directories in the inspected source tree:

| Path | Purpose | Public release posture |
| --- | --- | --- |
| `.claude/` | Local assistant or workflow state. | Exclude. Not research logic. |
| `LRF/` | Rectified Flow support code, including sampling logic. | Candidate for public release after path and license audit. |
| `VAE/` | VAE preprocessing, model training, and fold-safe preparation logic. | Candidate for public release after separating dataset paths and raw data assumptions. |
| `models/` | Model definitions, including the 1D U-Net velocity-field backbone. | Candidate for public release. |
| `utils/` | Helper modules used across scripts. | Candidate for public release after dependency audit. |
| `Results/` | Generated outputs, checkpoints, synthetic caches, plots, tables, logs, and privacy audit outputs. | Do not commit wholesale. Curate only small, non-sensitive summary tables if policy allows. |
| `manuscript/` | Manuscript TeX files and response-related source material. | Do not publish wholesale. Contains publication history and likely author or review material. |
| `__pycache__/` | Python cache files. | Exclude. |

Top-level files that matter for reproduction:

| Path | Purpose |
| --- | --- |
| `Readme.md` | Short project overview and high-level pipeline pointer. |
| `RUNNING_INSTRUCTIONS.md` | Most complete observed run guide for the 42-step paper pipeline. |
| `renaming_paths_vae.md` | Important provenance document describing renamed output paths, legacy names, and superseded scripts. |
| `1_train_flow.ps1` | Main PowerShell wrapper for Rectified Flow training. |
| `1_Rectified_Flow_training.py` | Main Python training entry point for the Rectified Flow model. |
| `rectified_flow_training.py` | Alternate or duplicate flow training script. Needs comparison before release. |
| `2_generate_synthetic_all_folds.ps1` | Main synthetic-cache generation wrapper. |
| `2_generate_synthetic_only.py` | Python generation script for synthetic windows. |
| `3_run_loso_evaluation.ps1` | Main LOSO evaluation wrapper. |
| `3_run_full_losocv.py` | Python LOSO evaluation script. |
| `TSTR.py` | Main Train-on-Synthetic, Test-on-Real evaluation and classifier implementation. |
| `run_tstr_all_subjects.ps1` | Fold-wise TSTR classification wrapper. |
| `windowing_sensitivity_grid_3x3.ps1` | 3x3 window and hop grid wrapper. |
| `window_grid_aug_quality.py` | Window-grid augmentation quality summary script. |
| `vae_ablation.ps1` | VAE-only ablation wrapper. |
| `vae_ablation_loso.py` | VAE-only ablation implementation. |

## Canonical Entry Points

The most complete canonical pipeline is documented in `RUNNING_INSTRUCTIONS.md`. The commands below are the observed canonical entry points, not newly validated reruns.

### Preprocessing and VAE Training

- `VAE/VAE_logic.py`
- `VAE/Run_VAE_Pretraings.ps1`

Observed command:

```powershell
.\VAE\Run_VAE_Pretraings.ps1 -IdealDir "D:/PAPER2/SALVAGED_PARTS/ideal_logs" -VariantTag "6CH/full" -Subjects @(1,2,3,5,8,9,10,11,12,13,14,16)
```

### Rectified Flow Training

- `1_train_flow.ps1`
- `1_Rectified_Flow_training.py`
- `models/unet_1d.py`
- `LRF/rectified_flow.py`

Observed command:

```powershell
.\1_train_flow.ps1
```

### Synthetic Cache Generation

- `2_generate_synthetic_all_folds.ps1`
- `2_generate_synthetic_only.py`

Observed command:

```powershell
.\2_generate_synthetic_all_folds.ps1
```

### LOSO Utility Evaluation

- `3_run_loso_evaluation.ps1`
- `3_run_full_losocv.py`
- `TSTR.py`

Observed command:

```powershell
.\3_run_loso_evaluation.ps1
```

### TSTR Classification Details

- `run_tstr_all_subjects.ps1`
- `TSTR.py`
- `8_1_summarize_tstr_classification_across_folds.py`
- `13_classification_violin_confusion_PATCHED.py`

Observed command:

```powershell
.\run_tstr_all_subjects.ps1
```

### Window and Hop Sensitivity

- `windowing_sensitivity_grid_3x3.ps1`
- `window_grid_aug_quality.py`

Observed command:

```powershell
.\windowing_sensitivity_grid_3x3.ps1
python window_grid_aug_quality.py
```

### VAE-Only Component Ablation

- `vae_ablation.ps1`
- `vae_ablation_loso.py`

Observed command:

```powershell
.\vae_ablation.ps1
python vae_ablation_loso.py
```

### Physics, Spectral, Manifold, Coverage, and Privacy Analyses

Canonical scripts listed in `RUNNING_INSTRUCTIONS.md`:

- `6_phyiscs_plausibility.py`
- `6_1_physics_summary_across_folds.py`
- `7_cluster_analysis.py`
- `7_1_summarize_cluster_analysis_across_folds.py`
- `9_psd_freq_analysis_data_efficiency.py`
- `9_1_summarize_psd_efficiency_across_folds.py`
- `10_coverage_population_analysis.py`
- `10_1_summarize_coverage_across_folds.py`
- `10_2_coverage_summary_all_folds_v6.py`
- `11_combined_manifold.py`
- `11_1_summarize_combined_manifold_across_folds.py`
- `14_privacy_audit.py`
- `14_1_reconstruction_attack.py`
- `14_2_summarize_privacy_audit.py`
- `membership_inference_holdout.py`
- `eval_membership_inference_holdout.py`
- `summarize_membership_inference_holdout.py`

## Important Generated Artifacts

Observed `Results/` directory size is approximately 33.18 GB. It contains 396 `.pt` files totaling approximately 31.51 GB, 446 `.npz` files totaling approximately 993 MB, and many generated figures, logs, and summary tables.

Key result directories:

| Path | Role |
| --- | --- |
| `Results/model_weights/vae_weights/6CH/full/` | Fold-specific VAE checkpoints and metadata. |
| `Results/model_weights/flow_weights/6CH/full/` | Fold-specific Rectified Flow checkpoints and summaries. |
| `Results/synthetic_weights/6CH/full/` | Fold-matched cached synthetic windows. |
| `Results/loso/` | LOSO classifier outputs, summaries, and confusion matrices. |
| `Results/tstr_classification/6CH/full/` | Full-IMU TSTR classification summaries. |
| `Results/tstr_classification/3CH/ablation/` | Accelerometer-only TSTR classification summaries. |
| `Results/sensitivity_grid_analysis/` | 3x3 window/hop grid checkpoints, caches, plots, and summary tables. |
| `Results/physics_plausibility/` | Acceleration, correlation, and physics validation outputs. |
| `Results/psd_analysis/` | Power spectral density and data-efficiency outputs. |
| `Results/distinctness_and_population_coverage/` | MMD, C2ST, UMAP, PCA, coverage, and novelty outputs. |
| `Results/privacy_audit/` | Privacy, nearest-neighbour, reconstruction-attack, and audit summaries. |
| `Results/membership_inference/` | Membership inference holdout outputs. |

## Duplicated, Obsolete, Experimental, or Ambiguous Files

These files require explicit handling before a clean public release:

| Path | Reason |
| --- | --- |
| `rectified_flow_training.py` | Appears to duplicate or alternate `1_Rectified_Flow_training.py`. Needs diff-based triage. |
| `13_classification_violin_confusion.py` | Legacy version; `13_classification_violin_confusion_PATCHED.py` is named as the patched version. |
| `reviewer_response_vae_ablation.tex` | Review response artifact, not release code. |
| `Response_to_Reviewers...docx` and similar response files | Review and author material, not public code. |
| `DRAFT.IPYNB` | Notebook draft with unclear status. |
| `Results/.../*.log` | Logs contain absolute local paths and should not be published unreviewed. |
| Paths containing `SPECTRAL_LOSS_OFF` in metadata | Historical naming remains in some metadata after path cleanup. Needs explanation or migration note. |

## Files Needed to Reproduce Paper Results

At minimum, a reproducible release needs:

- Source dataset access instructions for REALDISP, not redistributed raw logs.
- `VAE/VAE_logic.py` and `VAE/Run_VAE_Pretraings.ps1`.
- `1_Rectified_Flow_training.py`, `1_train_flow.ps1`, `LRF/rectified_flow.py`, and `models/unet_1d.py`.
- `2_generate_synthetic_only.py` and `2_generate_synthetic_all_folds.ps1`.
- `TSTR.py`, `3_run_full_losocv.py`, `3_run_loso_evaluation.ps1`, and `run_tstr_all_subjects.ps1`.
- Analysis scripts named in `RUNNING_INSTRUCTIONS.md`.
- Paper result summary CSV/TEX files under `Results/` for audit, preferably curated into a separate `paper_artifacts/` directory.
- Checkpoints only as optional release assets outside Git, after privacy and size review.

## Known Inventory Gaps

- The inspected source tree is not a git repository, so original commit provenance is unavailable from local git history.
- The manuscript references a `figures/` directory, but no such directory exists inside the inspected `LRF-IMU-P3` tree. Result figures exist under `Results/...`; the exact submission figure-copy folder was not found.
- Some final manuscript values appear to be manually transferred or post-processed from summaries rather than regenerated into one final table file.
- The exact accepted manuscript source is inferred as `manuscript/Manuscript_02_06_2026.tex`; this should be checked against the publisher proof.
