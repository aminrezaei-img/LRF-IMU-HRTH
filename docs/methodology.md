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

## Application boundary

The model accepts a canonical class ID and a seed and returns an independent
synthetic window. It does not infer classes from contextual records, decide
which external intervals are eligible, or assemble participant timelines.
Keeping those decisions outside the model repository prevents a downstream
cohort policy from becoming part of the HARTH training method.

## Historical boundary

The repository also contains the earlier REALDISP method-development and
parity path. REALDISP results, six-channel configurations, and the Paper 3
HARTH-family replacement are separate evidence boundaries. A result from one
path should not be silently presented as a result from the other.

See [architecture](architecture.md), [data and taxonomy](data_and_taxonomy.md),
the [repository boundary](repository_boundary.md), and
[validation](validation.md).
