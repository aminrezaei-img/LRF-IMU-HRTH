# Reproducibility status

## What this milestone establishes

The release history provides:

- portable roots and configuration loading under `src/lrf_imu/`;
- three evidence-labeled YAML variants;
- the pinned audit references at commit
  `f38bebce36c4f21d857dc084ac8d06759c2c012d`; and
- safety checks that reject local-path markers, secrets, and prohibited
  generated artifacts outside the exact historical-reference exception.

## What it does not establish

This tree does not contain the REALDISP data, historical checkpoints, Flow
implementation, generation/classifier evaluation code, synthetic caches, result
summaries, paper figures, or manuscript source. M3B adds a public VAE boundary
and verifies it against the immutable implementation and named external
checkpoints, but it cannot reproduce the scientific results from a clean
checkout and must not claim exact paper equivalence.

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

## Milestone 3A evidence

The seven data-preparation lanes are covered by synthetic-only fixtures and pure
contract tests. The compact parity flow reports 16 training, 7 validation, and 8
held-out windows with held-out subject 05, validation subject 01, and training
subjects 02 and 03. It exercises both 6CH and the explicit reconstructed 3CH path,
the default filter-before-runs mode and strict contiguity, training-only ddof=0
normalization, all-pair duplicate auditing, and SHA-1 exact-window identity.

The final integration command is:

    python -B -m pytest -p no:cacheprovider --basetemp <external-basetemp>

The M3A run collected 109 tests and completed with 109 passed and 0 warnings.
Additional checks covered CLI help, foreign-working-directory dry-run and
validate-only behavior, explicit metadata writing, overwrite refusal, JSON-safe
metadata, import/compile safety, path and artifact scanners, and 6CH/3CH compact
flows. The output policy was verified to serialize no raw or participant-derived
window arrays.

These results establish a portable public preparation boundary, not a rerun of
the participant study. No participant artifacts, checkpoints, VAE/Flow migration,
full training, full evaluation, or exact-paper equivalence claim is included.
## Milestone 3A correction pass

The executable package declares the same two core runtime requirements in
pyproject.toml and requirements.txt: PyYAML and the unpinned lower-bound
declaration numpy>=1.20. The lower bound is a current minimum-safe
compatibility choice for the supported Python floor, not a claim about the
historical research environment.

The six-channel default YAML is also included as intentional package data.
The root paper YAMLs remain human-facing evidence, while
tests/test_packaging.py guards normalized byte synchronization with the
installed-resource copies. The parser and pipeline preserve
split.classifier_window_validation_fraction: 0.20 for the classifier/window
protocol and read split.vae_subject_validation_fraction: 0.15 for the
VAE-safe subject split. The legacy split.validation_fraction: 0.20 key remains
an explicit classifier/window alias.

A clean wheel probe must run both the lrf-imu prepare-data console script and
python -m lrf_imu prepare-data from a foreign working directory with no
repository checkout on the import path. These probes use the synthetic fixture
only and do not establish participant-level reproducibility.