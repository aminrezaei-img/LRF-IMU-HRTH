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
