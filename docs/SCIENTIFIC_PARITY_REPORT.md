# Milestone 3A scientific parity report

## Scope and evidence posture

This report describes the public, synthetic-only M3A preparation boundary. It
records compatibility behavior selected from the Milestone 2A/2B contracts and
the tests that lock that behavior. It does not assert that the immutable
scientific source was copied, that participant artifacts are reproducible, or
that the public tree is an exact paper reproduction.

## Locked preparation invariants

| Contract | Public behavior | Evidence |
| --- | --- | --- |
| Raw layout | Exactly 120 numeric tab-separated columns; right-thigh signal columns 80..85; activity label column 119 | Schema and REALDISP loader tests |
| Activity vocabulary | Raw 1, 3, 4, 33 map to encoded 0, 1, 2, 3 | Activity and windowing tests |
| Windowing | Complete 160/40 windows; no padding or activity-boundary crossing | Windowing and compact pipeline tests |
| Run semantics | filter-before-runs is the default; strict original contiguity is explicit | Windowing and CLI tests |
| VAE-safe split | Subject-level validation fraction 0.15 | Split metadata and compact fixture tests |
| Compact parity | 16 train, 7 validation, 8 held-out; held-out 05, validation 01, train 02/03 in the compact assignment | Synthetic fixture and pipeline tests |
| Normalization | Fit on training windows only, population standard deviation ddof=0, floor 1e-8 | Normalization and pipeline tests |
| Duplicate identity | SHA-1 over canonical exact-window bytes, with cross-split public audit | Duplicate audit tests |
| 3CH path | Explicit reconstruction over columns 80..82, separate model schema, no inference-time channel drop | Schema, loader, and CLI tests |
| Output safety | Participant-derived arrays stay in memory; metadata is the only permitted artifact | Pipeline and CLI tests |

## Historical compatibility versus public safety

The public boundary retains historical compatibility where it is meaningful:
the four-class vocabulary, filter-before-runs default, 160/40 geometry, and
training-only population standardization are named and tested. It also makes
intentional safety changes: roots are explicit, discovery is direct-child and
nonrecursive, the default duplicate audit covers all split pairs, writes require
explicit permission, overwrite is opt-in, and raw or participant-derived arrays
are never serialized. The train/validation-only audit remains available only as
a labeled historical compatibility adapter.

## Three-channel lineage

The 3CH result is PUBLIC_RECONSTRUCTION_REQUIRED. The public route selects the
accelerometer columns 80, 81, and 82 and is a separately trained input schema.
It does not project a 6CH input at inference time and does not recover a
historical VAE checkpoint or accepted-paper training lineage. Fresh 3CH training
and evaluation would be required for a scientific claim.

## Validation evidence

The integrated synthetic suite collected 109 tests and completed with 109
passed and 0 warnings. It covered the seven M3A lanes, compact 6CH and 3CH
flows, CLI help and foreign-cwd invocation, dry-run, validate-only, explicit
metadata writing, overwrite refusal, import/compile safety, and artifact/path
scanners.

No full participant-data run, VAE training, Flow training, classifier training,
generation, evaluation, checkpoint migration, or exact-paper comparison was
performed. Those are intentionally outside M3A.
## Correction pass: explicit split and installed-package contracts

The paper YAMLs preserve the historical classifier/window value of 0.20 and
now name it explicitly as split.classifier_window_validation_fraction.
They separately name the VAE subject-level value as
split.vae_subject_validation_fraction: 0.15. The parser retains
split.validation_fraction: 0.20 only as a checked classifier/window alias,
and the pipeline metadata reports both values while passing 0.15 to the
subject-level split. This resolves the runtime contradiction without
relabeling the compatibility evidence.

The default six-channel profile is packaged intentionally for wheel execution.
The root YAMLs remain human-facing evidence and a synchronization test
compares normalized bytes with the packaged copies. NumPy is declared
consistently as numpy>=1.20 beside PyYAML in both core dependency files.
These are packaging and configuration guarantees; they do not establish
participant-data reproducibility, VAE/model migration, or exact-paper parity.