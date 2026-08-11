# Milestone 3E analysis parity

`exact_paper_reproduction=false`. The machine-readable record is
`contracts/analysis_parity_report.json`.

| Analysis | Status | Execution result |
| --- | --- | --- |
| VAE-only ablation | PARTIAL | All 12 6CH folds ran with the surviving checkpoints. Historical TSTR macro-F1 was 0.443404 ± 0.171059; regenerated was 0.448935 ± 0.177147. Every TSTR fold differed. TRTR matched exactly, fold by fold. The historical JSONs name a `VAE_weights/6CH/SPECTRAL_LOSS_OFF` checkpoint lineage that is absent; the run used the surviving `Results/model_weights/vae_weights/6CH/full` lineage. |
| Window sensitivity | PASS | The public aggregator read 108 fold JSONs for the full 3 × 3 grid. All 171 comparable numeric cells matched the stored summary exactly (maximum absolute difference 0). |
| Physical >10g | PARTIAL | A subject-01 historical synthetic cache was inverse-standardized with the public fold normalizer. Zero of 320,000 acceleration-magnitude points exceeded 10g; maximum magnitude was 66.123 m/s². The stored per-fold summary reports 0% for every fold, but the other 11 raw caches were not newly executed. |
| Spectral | PARTIAL | Public spectral statistics over the stored all-fold PSD curves reproduced mean log-PSD correlation 0.966280814 (historical 0.9663) and 10–25 Hz power ratio 0.445482737 (historical 0.4455). The ratio documents high-frequency attenuation. Raw fold-window PSDs were not regenerated. |
| Privacy | PARTIAL | Public summaries reproduced true-holdout MIA 0.495199 ± 0.020137 and the distinct post-hoc best-attack audit 0.514989 ± 0.016021. Reconstruction was 0/240 actually optimized attempts. The historical configuration selected 50 targets per fold but optimized only the first 20. Attacks were not rerun. |
| 3CH VAE-only | BLOCKED | No mapped historical 3CH VAE-only fold artifacts exist for comparison. |

## Interpretation boundaries

The two membership-inference values answer different threat models and are not
combined. Reconstruction success uses the strict criterion `optimized L2 <
0.10 × random-baseline L2`. These are attack-specific observations, not an
anonymization or privacy guarantee. The spectral results are descriptive signal
comparisons, not clinical validation.

No checkpoint, generated sample, participant window, plot, or historical Results
payload is included in the repository.
