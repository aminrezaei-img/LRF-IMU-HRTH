# Milestone 3E handoff

## Scope delivered

Milestone 3E adds small numerical analysis modules and no plotting migration:

- VAE-only latent-Gaussian generation and RF evaluation;
- nine-setting segmentation aggregation;
- physical acceleration-magnitude and strict >10g checks;
- Welch/log-PSD and band-power comparison;
- explicitly separate true-holdout MIA, post-hoc MIA, and reconstruction summaries;
- no-write-by-default CLI commands with explicit JSON output permission.

## Execution

All 12 VAE-only 6CH folds were run with 500 synthetic samples per class, seed
42, posterior means, population latent SD plus `1e-6`, one sequential NumPy RNG,
and RF(100, seed 42, one job). TRTR was exact; TSTR remains PARTIAL because the
checkpoint path recorded by the historical ablation is unavailable and the
surviving checkpoint lineage produces fold-level differences.

The sensitivity grid reproduced 171/171 comparable cells exactly. Subject-01
physical execution found 0/320,000 points above 10g. Stored aggregate PSD curves
reproduced 0.966280814 log-PSD correlation and 0.445482737 high-frequency power
ratio. Stored privacy folds reproduced both distinct MIA summaries and zero
reconstruction successes while preserving 600 configured targets versus 240
actual optimization attempts.

## External-only evidence and limits

Metadata-only execution JSON is under the user-selected external validation
root. Temporary fold inputs contain no arrays and may be deleted after review.
Historical Results, checkpoints, REALDISP, and the audit tree remained read only.
No attack was rerun, no 3CH VAE-only historical reference was available, and no
privacy, anonymization, clinical, or exact-paper claim is made.

Next milestone should compose these already validated surfaces; it must not
reinterpret PARTIAL as PASS or silently replace the unavailable VAE-only
checkpoint lineage.
