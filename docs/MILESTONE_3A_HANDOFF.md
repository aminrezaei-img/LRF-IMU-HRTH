# Milestone 3A handoff

## Status

M3A is an uncommitted public-release integration candidate. The source
workspace for transport is the verified mirror associated with this handoff;
the root integration tree is not a Git worktree. The patch is generated against
the authoritative public branch release/v1.0.0-candidate at the pinned HEAD
recorded by the parent task and is validated with a check-only apply.

M3A integrates only the seven completed data-preparation lanes:

1. activities and schema;
2. REALDISP discovery and loading;
3. activity-bounded windowing;
4. VAE subject and CNN window splits;
5. training-only normalization;
6. duplicate auditing; and
7. pipeline, package exports, CLI, tests, and pyproject metadata.

No VAE, Flow, classifier, generation, evaluation, checkpoint, or model migration
is included. Milestone 3B was not started.

## Scientific and safety decisions

- The raw schema is exactly 120 columns. Right-thigh 6CH uses columns 80..85,
  the label is column 119, and raw activity codes 1/3/4/33 encode to 0/1/2/3.
- Production geometry is 160 samples with a 40-sample hop. The default is
  filter-before-runs; strict original contiguity is opt-in.
- The VAE-safe subject split uses 0.15. The compact fixture demonstrates 16/7/8
  parity with held-out 05, validation 01, and training 02/03.
- Standardization uses training-only population standard deviation ddof=0.
- Duplicate identity is SHA-1 over canonical exact-window bytes. Public auditing
  covers every split pair; the historical train/validation-only adapter is
  explicit.
- The 3CH path is PUBLIC_RECONSTRUCTION_REQUIRED: columns 80..82 are a separate
  accelerometer schema and never an inference-time drop from 6CH.
- The pipeline is metadata-only at the filesystem boundary. Participant-derived
  arrays, raw logs, checkpoints, caches, and results are not serialized or
  included. There is no exact-paper claim.

## Cleanup

The transport mirror excludes build output, egg-info, Python caches, pytest
caches, pytest basetemps, replacement files, and the populated data-directory
gitkeep placeholder. No participant artifact or raw dataset file is present.

## Validation record

The final suite command is:

    python -B -m pytest -p no:cacheprovider --basetemp <external-basetemp>

The final run collected 109 tests and completed with 109 passed and 0 warnings.
The final checks also cover:

- Ruff and mypy when available, plus compile/import safety.
- JSON, YAML, TOML, and CFF parsing.
- CLI help, dry-run, validate-only, explicit metadata write, overwrite refusal,
  and foreign-cwd invocation.
- Both compact 6CH and reconstructed 3CH flows, including 16/7/8 parity.
- No raw or participant-derived window arrays in serialized metadata.
- No prohibited extensions, caches, temporary roots, duplicate source copies, or
  hardcoded workstation paths in operational release files.
- Public and audit Git status/HEAD pins and immutable-source inventory hashes
  unchanged.

The patch is ready for the root task to apply. This handoff does not commit,
push, or modify any authoritative repository.
## Correction pass: dependency and wheel portability

The acceptance correction pass is limited to three operational findings. The
runtime dependency is declared consistently as PyYAML plus numpy>=1.20 in both
install metadata files. The lower bound is unpinned and selected as a
minimum-safe declaration for the supported Python floor, not as an invented
historical lock.

The default six-channel YAML is intentional package data under
src/lrf_imu/resources/configs/paper/. The root configs/paper/ copies remain
human-facing evidence. tests/test_packaging.py parses the metadata, checks the
resource glob, and compares normalized bytes to prevent silent drift.
Installed-wheel probes exercise both the console script and the module entry
point from a foreign working directory.

The split parser now exposes and serializes
vae_subject_validation_fraction: 0.15 and
classifier_window_validation_fraction: 0.20. The old
validation_fraction: 0.20 remains an explicit classifier/window alias and must
match the named classifier value. The pipeline reads the VAE field for the
subject split instead of using an unconditional literal. This correction does
not migrate VAE/model code, add participant artifacts, or make an exact-paper
claim.