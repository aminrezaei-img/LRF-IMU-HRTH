# LRF-IMU Reproducibility Audit

## Summary

This audit records reproducibility assumptions and evidence before any public-release cleanup. It is based on local repository evidence from the inspected `LRF-IMU-P3` tree and does not modify research artifacts.

## Dataset and Preprocessing Assumptions

Dataset:

- REALDISP benchmark dataset.
- Ideal-placement logs only.
- Right-thigh IMU only.
- Six channels: triaxial accelerometer and triaxial gyroscope.
- Sampling rate: 50 Hz.
- Local raw log root observed in scripts: `D:/PAPER2/SALVAGED_PARTS/ideal_logs`.
- Raw logs are outside the source tree and must not be redistributed without checking dataset license and terms.

Participants:

- Included subjects: `1, 2, 3, 5, 8, 9, 10, 11, 12, 13, 14, 16`.
- Number of LOSO folds: 12.
- Reason recorded in manuscript: only participants with complete data across the four selected activities were included.

Activity labels:

| Encoded class | REALDISP code | Activity |
| --- | --- | --- |
| 0 | 1 | walking |
| 1 | 3 | running |
| 2 | 4 | jump_up |
| 3 | 33 | cycling |

Source evidence:

- `VAE/VAE_logic.py` defines label column `119`, right-thigh columns `80..85`, target codes `(1, 3, 4, 33)`, and mapping to four encoded classes.
- `paper_release_reference.yaml` records these values without changing them.

Preprocessing caveat:

- The four-class preprocessing filters target labels before constructing contiguous runs. This reproduces the paper-specific task but should be reviewed before reusing the parser for full REALDISP activity modeling, because excluded labels can affect how continuity should be interpreted.

## LOSO Procedure

The paper uses Leave-One-Subject-Out cross-validation:

- For each fold, one subject is held out as test.
- VAE and Rectified Flow models are trained only on the remaining 11 subjects.
- Standardization parameters are fitted only on the training subjects.
- Downstream classifiers are trained only on training-subject data or synthetic data generated from fold-trained models.
- CNN validation is drawn from the training pool and does not use the held-out subject.
- A SHA-1 duplicate and overlap audit is reported in the manuscript as finding no identical train-test windows.

## Windowing and Normalization

Main setting:

- Window length: 160 samples.
- Window duration: 3.2 s.
- Hop: 40 samples.
- Hop duration: 0.8 s.
- Overlap: 75 percent.

Sensitivity grid:

- Window lengths: 80, 160, 240 samples.
- Hops: 20, 40, 80 samples.
- Main paper setting: 160/40.

Normalization:

- Per-channel z-score standardization.
- Fit on training windows only within each LOSO fold.
- Same fold-specific mean and standard deviation applied to held-out and synthetic windows.

## Random Seeds

Observed seed values:

| Use | Seed |
| --- | --- |
| Global seed | 42 |
| Python hash seed in PowerShell wrappers | 42 |
| Synthetic generation | 42 |
| TSTR and classifier sampling | 42 |
| Subsampling for diagnostics | 42 |
| Subject split seed in VAE helper | 42 |

## VAE Hyperparameters

Architecture:

- Input shape: `B x 6 x 160`.
- Latent shape in main setting: `B x 48 x 40`.
- Latent stride: 4.
- Two downsampling stages and two upsampling stages.
- Deterministic reconstruction pass uses posterior mean while KL is still computed.

Training values observed in scripts and metadata:

| Parameter | Observed value |
| --- | --- |
| Batch size in checkpoint metadata | 256 |
| Learning rate default | 0.001 |
| Max epochs | 1000 |
| Early-stop minimum epochs in wrapper | 200 |
| Early-stop patience in wrapper | 100 |
| Mixed precision | bfloat16 AMP enabled |
| Safe subject split | enabled |
| Spectral loss | disabled |
| FFT weight | 0.0 |
| Gradient clipping | 1.0 |
| L2 reconstruction weight in wrapper | 0.5 |
| L1 reconstruction weight in wrapper | 0.1 |
| Beta init in wrapper | 0.08 |
| Beta min in wrapper | 0.04 |
| Beta decay in wrapper | 0.995 |
| Augmentation | enabled in observed wrapper context |
| Augmentation jitter | 0.008 |
| Augmentation scale | 0.04 |
| Augmentation time mask | 0.05 |

Uncertainty:

- The manuscript text includes VAE loss values `w_L2=1.0`, `w_L1=0.1`, `beta_init=5e-3`, `beta_min=1e-5`, and `beta_decay=0.7`. Later wrapper evidence used in the revision-era run guide shows different VAE settings. This should be resolved before claiming exact code-to-paper hyperparameter identity.

## Rectified Flow Hyperparameters

Training:

| Parameter | Observed value |
| --- | --- |
| Latent channels | 48 |
| Latent time steps | 40 |
| Epochs | 300 |
| Learning rate | 0.0005 |
| Optimizer | AdamW |
| Betas | 0.9, 0.95 |
| Weight decay | 0.0001 |
| Gradient clipping | 1.0 |
| Early-stop patience | 50 |
| Configured batch size in `1_train_flow.ps1` | 128 |
| Auto-batch | enabled |
| Subject 01 recorded effective batch size | 512 |
| Configured `FLOW_MODEL_CH` | 256 |

Architecture evidence:

- `models/unet_1d.py` implements `LatentDiffusionUNet1D`.
- Conditioning combines sinusoidal time embedding and learned class embedding.
- Residual blocks use short kernel size 3, long grouped kernel size 31, GroupNorm with 8 groups, and squeeze-excitation with reduction ratio 4.
- Downsampling uses average pooling by factor 2.
- Upsampling uses nearest-neighbour interpolation by factor 2.

Uncertainty:

- The manuscript supplementary table reports base width `C=128`; the observed `1_train_flow.ps1` sets `FLOW_MODEL_CH=256`, and `train_summary.json` for subject 01 records `model_ch=256`. This discrepancy must be resolved or explained before public reproducibility claims are finalized.

## Sampling Settings

Observed synthetic generation settings:

- Euler ODE sampler.
- Integrates from noise at `t=1` to data latent at `t=0`.
- Number of Euler steps: 10.
- Seed: 42.
- Synthetic windows per class for TSTR: 500.
- Synthetic cache root: `Results/synthetic_weights/6CH/full`.
- VAE decoder is frozen during generation.

## Classifier and Evaluation Settings

Random Forest:

- Source: `TSTR.py`.
- `RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)`.
- Input: flattened standardized window, shape `N x C*T`.
- Output: class probabilities through `predict_proba`.

1D CNN:

- Source: `TSTR.py`.
- Conv1d input channels to 32, kernel 5, padding 2.
- Conv1d 32 to 64, kernel 5, padding 2.
- MaxPool1d kernel 2.
- Conv1d 64 to 128, kernel 5, padding 2.
- FC layers: `128*(T/2) -> 256 -> 128 -> num_classes`.
- Dropout: 0.3 after the first two FC layers.
- Optimizer: Adam.
- Learning rate: 0.001.
- Weight decay: 0.0001.
- Epochs: 80.
- Patience: 10.
- Batch size: 64.
- Validation fraction: 0.2.

Evaluation scenarios:

- TRTR: train on real training subjects, test on held-out real subject.
- Low-data baseline: two real windows per class.
- TSTR: train on synthetic windows only, test on held-out real subject.
- Synthetic augmentation: low-data real subset plus synthetic windows.

Primary metric:

- Macro F1, summarized as mean plus standard deviation across 12 LOSO folds.

## Software Environment

Environment evidence is incomplete. Scripts imply:

- Windows PowerShell wrappers.
- Python environment with PyTorch, NumPy, pandas, scikit-learn, SciPy, matplotlib, seaborn, and likely tslearn or related DTW tools.
- CUDA/GPU expected for model training.
- Thread-count environment variables are set in some PowerShell wrappers for reproducibility.

Missing:

- No locked `requirements.txt`, `environment.yml`, `pip freeze`, or conda export was found during this audit.
- Public release should create a tested environment file from a clean machine or container.

## Commands by Paper Experiment

The commands below are observed run-guide commands, not rerun during this audit.

| Paper component | Command or script |
| --- | --- |
| VAE fold training | `.\VAE\Run_VAE_Pretraings.ps1 -IdealDir "D:/PAPER2/SALVAGED_PARTS/ideal_logs" -VariantTag "6CH/full" -Subjects @(1,2,3,5,8,9,10,11,12,13,14,16)` |
| Rectified Flow fold training | `.\1_train_flow.ps1` |
| Synthetic cache generation | `.\2_generate_synthetic_all_folds.ps1` |
| Main LOSO evaluation | `.\3_run_loso_evaluation.ps1` |
| TSTR per-subject analysis | `.\run_tstr_all_subjects.ps1` |
| Window grid | `.\windowing_sensitivity_grid_3x3.ps1` then `python window_grid_aug_quality.py` |
| VAE-only ablation | `.\vae_ablation.ps1` then `python vae_ablation_loso.py` |
| Physics plausibility | `python 6_phyiscs_plausibility.py` then `python 6_1_physics_summary_across_folds.py` |
| PSD and data efficiency | `python 9_psd_freq_analysis_data_efficiency.py` then `python 9_1_summarize_psd_efficiency_across_folds.py` |
| Manifold analysis | `python 11_combined_manifold.py` then `python 11_1_summarize_combined_manifold_across_folds.py` |
| Coverage analysis | `python 10_coverage_population_analysis.py`, `python 10_1_summarize_coverage_across_folds.py`, `python 10_2_coverage_summary_all_folds_v6.py` |
| Privacy audit | `python 14_privacy_audit.py`, `python 14_1_reconstruction_attack.py`, `python 14_2_summarize_privacy_audit.py` |
| Membership inference | `python membership_inference_holdout.py`, `python eval_membership_inference_holdout.py`, `python summarize_membership_inference_holdout.py` |

## Not Verified from Repository Evidence

- Exact accepted publisher proof source file.
- Independent DOI verification from publisher pages.
- Complete software environment.
- Whether all final manuscript figure files are present in the inspected source tree. The manuscript references a missing `figures/` directory.
- The exact provenance of some manually transferred table values, especially the 6CH/3CH utility table and some distributional similarity standard deviations.
- Whether old path names containing `SPECTRAL_LOSS_OFF` reflect only naming history or different experimental settings.
