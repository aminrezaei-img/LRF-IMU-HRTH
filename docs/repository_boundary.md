# Repository boundary

## Core responsibility

LRF-IMU-HARTH owns the sensor-model lineage:

1. verify and load HARTH plus Adult Walking Speed;
2. encode the fixed ten-class taxonomy;
3. create subject-safe train, validation, and held-out partitions;
4. fit and record training-only normalisation;
5. train and evaluate the three-channel VAE;
6. train and evaluate the ten-class latent Rectified Flow; and
7. generate independent class-conditioned 160-sample windows.

The accepted production assets are one global VAE/Flow pair. Their use by
several people or several cohorts never creates new persona-specific model
weights.

## Explicitly external

The following are application responsibilities and must not enter the core
source, configuration, examples, or command-line interface:

- semantic or contextual records;
- physical-state assignment from another dataset;
- signal-eligibility decisions;
- participant/persona and date discovery;
- exact-duration interval scheduling;
- multi-window stitching or crossfade policy;
- multimodal fusion and linkage indexes; and
- cohort-level generation, resume, finalisation, and freeze reports.

A downstream integration may import the model API or execute the model CLI,
but it must pin this repository release and the artifact SHA-256 values. It
must not vendor or silently modify model code and still claim the same model
identity.

## GitHub layout

Normal Git contains code, configuration, tests, documentation, and the small
artifact manifest. GitHub Release `paper3-harth-production-v1` contains the
large immutable files:

- production VAE checkpoint;
- production Rectified Flow checkpoint;
- the exact training-data archive with upstream licence and README.

Clean model run summaries, production normalisation, and the frozen
configuration are small tracked repository resources. Their recovered source
hashes are recorded, while private workstation paths are excluded.

Generated signals and downstream datasets are not LRF model artifacts and are
not part of this release.
