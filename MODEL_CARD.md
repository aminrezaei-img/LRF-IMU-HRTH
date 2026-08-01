# Model card

## Milestone 1 status

No trained model, checkpoint, or executable scientific model implementation is
included in this release. The configuration layer describes later migration
targets; it is not a usable generative model and does not establish benchmark
parity.

## Audited method identity

The paper audit describes LRF-IMU as a two-stage latent pipeline: a VAE
compresses six-channel IMU windows and a class-conditional Rectified Flow
operates in the latent space before decoding. The scientific implementation,
preprocessing, sampling, and evaluation code remain outside Milestone 1.

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
tables and figures. Those issues must be resolved before a later release makes
exact reproduction or model-performance claims.
