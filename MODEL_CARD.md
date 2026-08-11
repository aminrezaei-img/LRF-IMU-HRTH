# Model card

## Model and release status

LRF-IMU is a two-stage, class-conditioned latent generative pipeline. A VAE
maps 160-sample IMU windows to a 48-by-40 latent representation; a Rectified
Flow U-Net transports class-conditioned latent noise toward data before the VAE
decoder returns a window. The paper/TSTR sampler uses ten reverse-Euler steps.

The public model supports separately trained 6CH and 3CH configurations.
Historical checkpoints are not included. The 3CH preprocessing path is an
explicit reconstruction of accelerometer columns, paired only with separately
trained 3CH checkpoints; it is not an inference-time channel drop and is not
proof of exact historical parser lineage.

Public/original deterministic VAE and Flow operations matched exactly on
synthetic inputs, historical 6CH/3CH checkpoints, and a real fold. Core
evaluation and analysis have exact, partial, and blocked components documented
in [`docs/RESULTS_REPRODUCTION.md`](docs/RESULTS_REPRODUCTION.md).
`exact_paper_reproduction=false` remains unchanged.

## Intended and out-of-scope use

The current contents are intended for research reproduction, controlled IMU
generation experiments, evaluation, and inspection of the documented release
boundary. They are not intended for clinical
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
M3B public one-fold evaluation. These issues do not invalidate the explicitly
bounded M3D results reported with their historical and runtime qualifiers. They
must be resolved before claiming exact paper reproduction, artifact equivalence,
or performance beyond the audited protocol.

## Milestone 3C Flow model card supplement

### Model boundary

The Flow model is a class-conditional 1-D latent velocity field over `[batch,48,40]`. The public implementation mirrors the source U-Net convention: sinusoidal time embedding, four-class embedding, average-pool downsampling, nearest-neighbor upsampling, residual/SE blocks, and explicit reverse Euler integration. Historical subject-01 checkpoints use width 256 and 89 state tensors.

The checkpoint loader uses `weights_only=True` where supported, requires the exact six-key Flow root schema (`config`, `epoch`, `history`, `opt`, `unet`, `val_loss`), validates tensor geometry, and keeps checkpoint payload values out of reports. A Flow checkpoint and VAE checkpoint must have the same declared 3CH or 6CH width/channel pairing; cross-pairing is rejected.

### Sampling profiles

The paper/TSTR profile is ten reverse-Euler steps with the paper seed convention. The website profile is separate: 100 steps, every second state retained (51 states), native 160-sample windows, 40-sample linear overlap-add, four independent segments for ten seconds, and seed `base + subject*1000 + activity*100`. Website signals are explicitly marked `website_trajectory` and `paper_tstr_samples=false`.

### Scientific status

Source/public parity passed on synthetic probes, historical width-256 6CH and
3CH checkpoints, and one held-out REALDISP fold. Width 128 was exercised only
through synthetic/manuscript compatibility tests; no historical width-128
checkpoint was validated. This validates implementation parity, not a claim of
exact paper reproduction. `exact_paper_reproduction=false`; the width-256
checkpoint versus manuscript/source width-128 discrepancy remains unresolved.
Evaluation was outside the Milestone 3C parity gate and was added later without
changing the accepted model implementation. No generated artifacts are included.

## Milestone 3D evaluation supplement

The release now includes RF and CNN evaluation logic, but no evaluation model,
checkpoint, participant data, or generated sample. All-fold 6CH/3CH RF and 6CH
CNN runs were executed externally. Deterministic evaluation matches historical
metrics when supplied the same stored cache. Fresh CPU-generated TSTR samples
were produced in a different device/runtime context from the historical
CUDA-associated samples. The evidence does not isolate device as the sole cause;
fold-level differences and runtime details remain in the parity report.

The results apply only to the audited REALDISP cohort and protocol. They do not
establish clinical validity, anonymization, deployment fitness, or universal
performance. The 3CH result is empirical compatibility for a reconstructed
parser paired with separately trained 3CH checkpoints, not lineage proof.

## Milestone 3E analysis supplement

The release includes numerical, no-plot analysis helpers for VAE-only ablation,
segmentation sensitivity, acceleration plausibility, PSD comparison, and three
explicit privacy threat-model summaries. It includes no checkpoint, participant
window, synthetic array, trained evaluator, or historical Results payload.

The sensitivity aggregation is exact against stored fold artifacts. VAE-only
TSTR, physical, spectral, and privacy evidence remains explicitly PARTIAL for
the reasons recorded in `docs/ANALYSIS_PARITY_REPORT.md`. In particular, the
two MIA setups are not interchangeable, zero reconstruction success is not an
anonymization guarantee, and high-frequency attenuation is documented rather
than hidden. `exact_paper_reproduction=false`.
