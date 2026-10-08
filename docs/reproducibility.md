# Reproducibility

## Frozen scientific state

The production model lineage is:

```text
training code:             81b123eab079f5cde7400fee5620e3bffb85a673
accepted generation code: 150b4de6e58365fdda5fc7279192c136d4e8b064
training data:             dad2cfbe89a26f72f19770419469ac037de200df
```

The Paper 3 composition is `harth_walking_speed`, seed 42, held-out subject
`harth:S006`, three channels, 160-sample windows, 40-sample hop, and 50 Hz.

The core preprocessing, training, and model-window generation files are
unchanged between the training commit and this cleaned release branch.
Application-specific histories remain available in Git but are not part of
the core release surface.

## Runtime record

The validated production runtime used Conda `py311`, Python 3.11.11, PyTorch
2.5.1, Torch CUDA 12.1, and an NVIDIA RTX 4070 Laptop GPU. Record the actual
environment, device, driver, seed, configuration, and checkpoint hashes for a
new run.

## Checkpoint identity

The production VAE and Flow hashes are recorded in
[checkpoints](checkpoints.md). A run should verify those hashes before loading
the files and retain them in its metadata.

## Reproducibility levels

- **Artifact reproduction:** download the accepted files and verify their
  exact SHA-256 identities.
- **Pipeline reproduction:** use the frozen data snapshot, code, configuration,
  subject split, and seed to repeat preprocessing and training.
- **Numerical reproduction:** compare outputs only under a fully recorded
  runtime. GPU kernels and library versions can prevent bitwise-identical
  retraining even when the method is equivalent.

## Research paths

1. **Existing checkpoints:** fetch the production assets, verify hashes,
   generate a small window, and inspect metadata.
2. **Full model path:** extract the frozen HARTH-family data, train the VAE and
   Flow, evaluate them, and generate class-conditioned windows.

No participant/cohort data, production checkpoints, generated arrays, or
private machine paths are stored in normal Git history. The exact model and
training-data files are versioned as GitHub Release assets for this repository.
