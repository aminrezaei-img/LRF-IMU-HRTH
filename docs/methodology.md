# Methodology

## Paper 3 model path

The Paper 3 replacement path uses HARTH and Adult Walking Speed recordings as
the source for a controlled ten-class thigh-accelerometer generator:

```text
HARTH + Adult Walking Speed
            ↓
   subject-level split and windows
            ↓
 training-subject-only normalization
            ↓
       VAE encoding
            ↓
 latent Rectified Flow training
            ↓
 class-conditioned latent sampling
            ↓
       VAE decoding
            ↓
  synthetic three-axis accelerometer signal
```

The production geometry is three channels, 160 samples per window, 40-sample
hop, and 50 Hz. The VAE latent representation is 48 channels by 40 time
steps. The Flow model is conditioned on ten fixed HARTH-compatible classes.

## Paper 3 application path

DayForge provides semantic and contextual evidence. It does not assign HARTH
classes or generate signals. The LRF bridge applies a separate deterministic
mapping and then invokes the existing exact-duration generator:

```text
DayForge resolved intervals and handoff evidence
            ↓
 conservative Module B physical-state mapping
            ↓
 Module C exact-duration class-conditioned generation
            ↓
       multi-window stitching
            ↓
     synthetic sensor timeline
```

Realized mobility and explicit physical evidence remain stronger than a
contextual hint. The derived in-bed opportunity is sensor-facing contextual
evidence only; it is not physiological sleep.

## Historical boundary

The repository also contains the earlier REALDISP method-development and
parity path. REALDISP results, six-channel configurations, and the Paper 3
HARTH-family replacement are separate evidence boundaries. A result from one
path should not be silently presented as a result from the other.

See [architecture](architecture.md), [data and taxonomy](data_and_taxonomy.md),
and [validation](validation.md).
