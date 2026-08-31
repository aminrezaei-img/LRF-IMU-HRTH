# Reproducibility

## Frozen scientific state

The LRF scientific freeze is:

```text
commit: 150b4de6e58365fdda5fc7279192c136d4e8b064
tag:    paper3_lrf_dayforge_handoff_v1
```

The Paper 3 composition is `harth_walking_speed`, seed 42, held-out subject
`harth:S006`, three channels, 160-sample windows, 40-sample hop, and 50 Hz.

The DayForge scientific handoff is identified separately by its scientific
commit/tag and its read-only semantic and derived evidence roots. Packaging
must not alter either source.

## Runtime record

The validated production runtime used Conda `py311`, Python 3.11.11, PyTorch
2.5.1, Torch CUDA 12.1, and an NVIDIA RTX 4070 Laptop GPU. Record the actual
environment, device, driver, seed, configuration, and checkpoint hashes for a
new run.

## Checkpoint identity

The production VAE and Flow hashes are recorded in
[checkpoints](checkpoints.md). A run should verify those hashes before loading
the files and retain them in its metadata.

## Reproducibility tests

The final Module C validation exercised:

- same-seed output: all compared files and arrays were byte/exact identical
  under the tested runtime;
- different-seed output: all compared generated arrays differed while
  remaining finite;
- exact sample counts and `[start,end)` timestamps;
- deterministic per-window seeds;
- provenance and boundary-jump audits; and
- no mutation of the DayForge source cohort.

These are runtime-specific reproducibility observations, not a claim that all
hardware, CUDA kernels, or future library versions will be bitwise identical.

## Research paths

1. **Existing checkpoints:** install the package, supply compatible checkpoint
   paths, verify hashes, generate a small window, and inspect metadata.
2. **Full model path:** prepare external HARTH-family data, train VAE and Flow,
   evaluate with Module A, and generate class-conditioned windows.
3. **Paper 3 interface:** supply frozen DayForge outputs, run Module B mapping,
   then invoke Module C only for explicitly selected intervals.

No participant data, production checkpoints, generated arrays, or private
machine paths are stored in this repository.
