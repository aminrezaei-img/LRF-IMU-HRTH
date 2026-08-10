# Model card

## Milestone 3B status

An executable compatibility VAE boundary is included, but no trained model or
checkpoint is included. The public LatentVAE1D preserves the observed 6CH/3CH
state-dict geometry and is usable for explicit CPU encode/reconstruct probes
when an external checkpoint is supplied. It does not establish paper benchmark
parity or exact reproduction.

## Audited method identity

The paper audit describes LRF-IMU as a two-stage latent pipeline: a VAE
compresses six-channel IMU windows and a class-conditional Rectified Flow
operates in the latent space before decoding. M3B migrates only the VAE model,
training semantics, and safe checkpoint boundary. Rectified Flow, sampling,
classifier evaluation, and full result generation remain outside this release.

## VAE compatibility evidence

The public model supports independent declared 3CH and 6CH namespaces with
[batch, channels, 160] -> [batch, 48, 40] geometry. Synthetic original/public
comparisons and both named subject-01 checkpoints produced exact zero maximum
errors for deterministic outputs. One external REALDISP fold also matched
exactly on normalized VAE reconstruction outputs. See docs/VAE_PARITY_REPORT.md
for hashes, tolerances, and limitations.

## Intended and out-of-scope use

The current contents are intended for release-boundary review, configuration
inspection, and future reproducibility work. They are not intended for clinical
decision-making, participant monitoring, deployment, or claims about model
quality.

## Data and limitations

The audited evidence covers a controlled REALDISP subset: ideal placement, one
right-thigh sensor, four activities, 12 listed subjects, and leave-one-subject-
out evaluation. It does not establish performance for other placements,
activities, populations, sampling rates, or deployment conditions. REALDISP
data and derived artifacts are not included.

The audit retains unresolved discrepancies in the VAE loss/KL schedule and
Rectified Flow base width, as well as incomplete provenance for some paper
tables and figures. Historical validation metrics also do not exactly match the
M3B public one-fold evaluation. These issues must be resolved before a later
release makes exact reproduction or model-performance claims.

## Milestone 3C Flow model card supplement

### Model boundary

The Flow model is a class-conditional 1-D latent velocity field over `[batch,48,40]`. The public implementation mirrors the source U-Net convention: sinusoidal time embedding, four-class embedding, average-pool downsampling, nearest-neighbor upsampling, residual/SE blocks, and explicit reverse Euler integration. Historical subject-01 checkpoints use width 256 and 89 state tensors.

The checkpoint loader uses `weights_only=True` where supported, requires the exact six-key Flow root schema (`config`, `epoch`, `history`, `opt`, `unet`, `val_loss`), validates tensor geometry, and keeps checkpoint payload values out of reports. A Flow checkpoint and VAE checkpoint must have the same declared 3CH or 6CH width/channel pairing; cross-pairing is rejected.

### Sampling profiles

The paper/TSTR profile is ten reverse-Euler steps with the paper seed convention. The website profile is separate: 100 steps, every second state retained (51 states), native 160-sample windows, 40-sample linear overlap-add, four independent segments for ten seconds, and seed `base + subject*1000 + activity*100`. Website signals are explicitly marked `website_trajectory` and `paper_tstr_samples=false`.

### Scientific status

Source/public parity passed on synthetic probes, both historical checkpoint widths/channels, and one held-out REALDISP fold. This validates implementation parity, not a claim of exact paper reproduction. `exact_paper_reproduction=false`; the historical width-256 checkpoint versus manuscript/source width-128 discrepancy remains unresolved. No TSTR/evaluation script was migrated and no generated artifacts are included.
## Milestone 3D evaluation supplement

The release now includes RF and CNN evaluation logic, but no evaluation model,
checkpoint, participant data, or generated sample. All-fold 6CH/3CH RF and 6CH
CNN runs were executed externally. Deterministic evaluation matches historical
metrics when supplied the same stored cache. Fresh CPU-generated TSTR samples
do not claim equivalence to historical CUDA-generated samples; fold-level
differences and runtime details are retained in the evaluation parity report.

The results apply only to the audited REALDISP cohort and protocol. They do not
establish clinical validity, anonymization, deployment fitness, or universal
performance. The 3CH result is empirical compatibility for a reconstructed
parser paired with separately trained 3CH checkpoints, not lineage proof.
