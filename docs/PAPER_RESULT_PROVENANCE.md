# Paper Result Provenance

## Summary

This file maps accepted-manuscript claims to observed scripts and artifacts. It distinguishes direct artifact provenance from partial provenance where a script and upstream summary exist but the final table appears manually copied, post-processed, or not present as a complete regenerated artifact.

Provenance labels:

- `Complete`: script and final summary artifact were found and values align with manuscript-level reporting.
- `Partial`: source scripts and related outputs exist, but final transfer, rounding, or post-processing cannot be fully reconstructed from one artifact.
- `Unestablished`: complete source artifact was not found in the inspected tree.

## Manuscript Source

Accepted source candidate:

- `manuscript/Manuscript_02_06_2026.tex`

Status:

- This is the latest manuscript-like TeX source found in the inspected tree.
- It must still be checked against the publisher proof for exact final identity.
- The manuscript references a `figures/` directory that was not present in the inspected source tree.

## Headline Results

| Result or claim | Manuscript value | Source scripts | Source artifacts | Status |
| --- | --- | --- | --- | --- |
| Main RF TRTR macro F1 | `0.985 +/- 0.021` | `3_run_loso_evaluation.ps1`, `3_run_full_losocv.py`, `TSTR.py` | `Results/loso/loso_summary_by_classifier.csv`, `Results/loso/table_loso_overall.tex` | Complete |
| Main RF TSTR macro F1 | `0.956 +/- 0.081` | `3_run_loso_evaluation.ps1`, `3_run_full_losocv.py`, `TSTR.py` | `Results/loso/loso_summary_by_classifier.csv`, `Results/loso/table_loso_overall.tex` | Complete |
| TSTR retention | `97.1% +/- 8.6%` | Same as above | `Results/loso/loso_summary_by_classifier.csv` | Complete |
| Across 3x3 grid, TSTR macro F1 at least `0.950` | Rounded minimum from grid | `windowing_sensitivity_grid_3x3.ps1`, `window_grid_aug_quality.py` | `Results/sensitivity_grid_analysis/summary/window_grid_aug_quality_summary.csv` and `.tex` | Complete |
| Across 3x3 grid, augmentation macro F1 at least `0.931` | Rounded minimum from grid | Same as above | `Results/sensitivity_grid_analysis/summary/window_grid_aug_quality_summary.csv` and `.tex` | Complete |
| VAE-only TSTR macro F1 | `0.443 +/- 0.171` | `vae_ablation.ps1`, `vae_ablation_loso.py` | `Results/vae_ablation/vae_ablation.csv` and per-subject JSON files | Partial |
| No synthetic acceleration above `10g` | `0` exceedances | `6_phyiscs_plausibility.py`, `6_1_physics_summary_across_folds.py` | `Results/physics_plausibility/allfolds/physics_metrics_across_folds.tex` | Complete |
| Synthetic acceleration magnitude mean | `11.31 +/- 0.38 m/s^2` | Same as above | Same as above | Complete |
| Real acceleration magnitude mean | `12.94 +/- 0.77 m/s^2` | Same as above | Same as above | Complete |
| KS distance for acceleration magnitude | `0.129 +/- 0.038` | Same as above | Same as above | Complete |
| Overall mean absolute correlation difference | `0.158 +/- 0.048` | Same as above | Same as above | Complete |
| Mean log-PSD correlation | `0.966` | `9_psd_freq_analysis_data_efficiency.py`, `9_1_summarize_psd_efficiency_across_folds.py` | `Results/psd_analysis/allfolds/psd_spectral_stats_6ch.csv` | Complete |
| Mean PSD band-power ratio 0 to 25 Hz | `0.555` | Same as above | Same as above | Complete |
| Mean PSD ratio 10 to 25 Hz | `0.445` | Same as above | Same as above | Complete |

## Tables

| Manuscript table | Content | Source artifacts | Status | Notes |
| --- | --- | --- | --- | --- |
| Main LOSO utility table | RF/CNN TRTR, TSTR, retention, low-data augmentation | `Results/loso/loso_summary_by_classifier.csv`, `Results/loso/table_loso_overall.tex` | Complete for RF headline values, partial for all displayed CNN variants | CNN values differ across main LOSO and sensor-reduction summaries; check exact table source before release. |
| Window/stride grid table | 9 grid settings, TRTR, TSTR, retention, augmentation, PSD, high-frequency PSD ratio | `Results/sensitivity_grid_analysis/summary/window_grid_aug_quality_summary.csv`, `.tex` | Complete | Values align with the final manuscript table after rounding. |
| VAE-only ablation table | Full LRF-IMU versus VAE-only latent sampling | `Results/vae_ablation/vae_ablation.csv`; scripts `vae_ablation_loso.py` and `sensitivity_summary.py` | Partial | Fold-level CSV exists. A final complete generated LaTeX table was not confirmed. |
| Reproducibility tables | Data split, VAE, flow, classifiers, runtime | `manuscript/Manuscript_02_06_2026.tex`, scripts named in this audit | Partial | These are manuscript tables. Some values conflict with observed wrapper settings, especially VAE loss schedule and flow base width. |
| Generative quality 6CH versus 3CH | Diversity, novelty, coverage, PCA area ratio | `aggregate_generative_quality_6v3.py`, 6CH and 3CH coverage outputs | Partial | Expected comparison file `Results/distinctness_and_population_coverage/ablation_comparison/generative_quality_6v3_comparison.csv` was not found. |
| TSTR classification 6CH versus 3CH | RF and CNN TRTR, scarce, TSTR, TSTR plus scarce | `Results/tstr_classification/6CH/full/allfolds/*.csv`, `Results/tstr_classification/3CH/ablation/allfolds/*.csv`, `make_tstr_table_6v3.py` | Partial | Root `tstr_utility_6v3.tex` is an incomplete stub in this inspected tree. |

## Figures

| Figure | Manuscript include path | Observed generation or output source | Status |
| --- | --- | --- | --- |
| LRF-IMU pipeline | `figures/figure1_pipeline.pdf` | Not found in inspected tree | Unestablished |
| VAE architecture | `figures/NeuralBlocks_minimal.pdf` | Not found in inspected tree | Unestablished |
| RF training objective | `figures/3a_Training_objective.pdf` | Not found in inspected tree | Unestablished |
| Conditional U-Net architecture | `figures/Architecture.pdf` | Not found in inspected tree | Unestablished |
| ODE sampling schematic | `3b_Inference_ODE.tex` | Not found in inspected tree | Unestablished |
| Physics validation | Result figures under `Results/physics_plausibility/...` | `6_phyiscs_plausibility.py`, `6_1_physics_summary_across_folds.py` | Complete for result figure outputs |
| PSD figure | `figures/Fig_PSD_Frequency_Analysis_all_6ch.pdf` in manuscript | `Results/psd_analysis/allfolds/Fig_PSD_Frequency_Analysis_all_6ch.pdf` | Partial | Result figure found under `Results/`; manuscript copy path not found. |
| Data-efficiency figure | `figures/Fig_Data_Efficiency_all_6ch.pdf` in manuscript | `Results/psd_analysis/allfolds/` outputs | Partial | Source data found; exact manuscript copy path not found. |
| Combined manifold figure | `figures/Fig_Combined_Manifold_ALL.pdf` in manuscript | `Results/distinctness_and_population_coverage/6CH/full/Combined_Manifold_ALL/` | Partial | Generated result figure likely exists under `Results/`; exact manuscript copy path not found. |
| Coverage summary figure | `figures/Fig_Coverage_Summary_AllFolds.pdf` in manuscript | `Results/distinctness_and_population_coverage/6CH/full/Coverage_Summary_updated/` or related coverage outputs | Partial | Folder naming drift needs reconciliation. |

## Distinctness and Coverage Provenance

Distributional similarity:

- Scripts: `11_combined_manifold.py`, `11_1_summarize_combined_manifold_across_folds.py`.
- Artifacts:
  - `Results/distinctness_and_population_coverage/6CH/full/Combined_Manifold_ALL/combined_manifold_metrics_per_fold.csv`
  - `Results/distinctness_and_population_coverage/6CH/full/Combined_Manifold_ALL/combined_manifold_metrics_across_folds.csv`

Audit issue:

- The summary CSV observed during this audit gives signal-space MMD mean close to `4.51e-3`, but the standard deviation in the CSV appears larger than the manuscript-reported `0.63e-3`. C2ST standard deviations likewise need checking. This may be due to a different summary convention, manual rounding, or a later artifact not present in the inspected tree.

Coverage and novelty:

- Scripts: `10_coverage_population_analysis.py`, `10_1_summarize_coverage_across_folds.py`, `10_2_coverage_summary_all_folds_v6.py`.
- Artifacts:
  - `Results/distinctness_and_population_coverage/6CH/full/Distinctnes_population_coverage_all/coverage_metrics_across_folds.csv`
  - `Results/distinctness_and_population_coverage/6CH/full/Distinctnes_population_coverage_all/coverage_metrics_across_folds.tex`

Audit issue:

- Fold-mean nearest-neighbour summaries and pooled nearest-neighbour summaries are both discussed in the manuscript. The release should label which statistics are fold-level and which are pooled.

## Privacy and Membership Inference Provenance

Privacy audit:

- Scripts:
  - `14_privacy_audit.py`
  - `14_1_reconstruction_attack.py`
  - `14_2_summarize_privacy_audit.py`
- Artifacts:
  - `Results/privacy_audit/summary/privacy_audit_summary.tex`
  - `Results/privacy_audit/summary/reconstruction_attack_summary.csv`

Membership inference holdout:

- Scripts:
  - `membership_inference_holdout.py`
  - `eval_membership_inference_holdout.py`
  - `summarize_membership_inference_holdout.py`
- Artifacts:
  - `Results/membership_inference/r/merged_fulltrain_summary.csv`
  - `Results/membership_inference/r/merged_fulltrain_summary.tex`

Audit issue:

- Two membership-inference summaries exist. The privacy audit summary reports best attack AUC around `0.515 +/- 0.016`; the holdout membership table reports mean AUC `0.495 +/- 0.020` and max fold AUC `0.520`. Both may be valid for different threat models, but the public release must avoid mixing them without explanation.

## Regenerated Versus Manual or Post-Processed Values

Clearly regenerated or directly summarized:

- Main RF LOSO values from `Results/loso/loso_summary_by_classifier.csv`.
- Window-grid values from `Results/sensitivity_grid_analysis/summary/window_grid_aug_quality_summary.csv`.
- Physics plausibility values from `Results/physics_plausibility/allfolds/physics_metrics_across_folds.tex`.
- PSD summary values from `Results/psd_analysis/allfolds/psd_spectral_stats_6ch.csv`.
- Membership holdout table from `Results/membership_inference/r/merged_fulltrain_summary.tex`.

Likely manual or partially post-processed:

- Data-efficiency table values in manuscript.
- 6CH/3CH TSTR utility table, because the found `tstr_utility_6v3.tex` is incomplete.
- 6CH/3CH generative quality comparison table, because the expected final CSV was not found.
- Some distributional similarity standard deviations in the manuscript.

## Results with Incomplete Provenance

The following should be explicitly reverified before public release claims:

- Exact source of the final accepted PDF figures copied under `figures/`.
- Exact accepted manuscript source file.
- VAE hyperparameter schedule discrepancy between manuscript text and observed wrappers.
- Flow base-width discrepancy between manuscript supplement and observed `FLOW_MODEL_CH`.
- 6CH/3CH generative quality comparison values.
- 6CH/3CH TSTR utility table final formatting and source values.
