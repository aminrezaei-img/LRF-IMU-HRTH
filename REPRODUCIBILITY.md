# Reproducibility status

## What this milestone establishes

Milestone 1 provides:

- portable roots and configuration loading under `src/lrf_imu/`;
- three evidence-labeled YAML variants;
- the pinned audit references at commit
  `f38bebce36c4f21d857dc084ac8d06759c2c012d`; and
- safety checks that reject local-path markers, secrets, and prohibited
  generated artifacts outside the exact historical-reference exception.

## What it does not establish

This tree does not contain the REALDISP data, preprocessing implementation,
VAE or Flow implementation, generation code, evaluation code, checkpoints,
synthetic caches, result summaries, paper figures, or manuscript source. It
therefore cannot reproduce the scientific results from a clean checkout and
must not claim exact paper equivalence.

The configuration values are compatibility defaults selected from audited
manuscript, wrapper, and checkpoint evidence. The unresolved VAE schedule and
Flow-width discrepancies remain visible in each paper config's evidence block.

## Audited facts for later work

Use the dataset and preprocessing boundary in [DATA_ACCESS.md](DATA_ACCESS.md):
REALDISP ideal placement, right-thigh six-channel input, 50 Hz, four audited
activity codes, 12 listed LOSO subjects, 160/40 main windowing, and training-
participant-only standardization. These facts describe the audited task, not a
general-purpose REALDISP loader.

The audit also records a source-state drift: the current immutable source
`Results/` tree was observed at 4,667 files and 34,811,675,494 bytes, compared
with the pinned audit baseline of 4,619 files and 34,791,553,468 bytes. The
additional 48 website-trajectory JSON files and four rewritten flow-trajectory
images are excluded from this milestone and were not copied.

## Verification boundary

The release smoke checks should:

1. parse all three YAML files;
2. load each config with `device="cpu"` and a dotted override;
3. compile/import the configuration package without creating cache files;
4. run `tests/test_no_absolute_paths.py` with the cache provider disabled and a
   temporary base directory outside the repository; and
5. verify the seven locked/reference hashes and repository/source integrity.

No scientific training or evaluation command is part of this milestone.
