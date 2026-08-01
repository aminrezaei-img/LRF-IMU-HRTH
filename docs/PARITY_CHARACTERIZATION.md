# Milestone 2A parity characterization

## Status

Milestone 2A is characterization-only. It records the evidence needed before
scientific migration; it does not implement preprocessing, the VAE, Flow,
generation, evaluation, or any paper experiment.

This milestone makes no exact-paper-reproduction, full-rerun, checkpoint-parity,
or accepted-manuscript-equivalence claim. No scientific module was copied into
`src/lrf_imu`. No REALDISP data, participant artifacts, checkpoints, synthetic
caches, results, logs, or manuscript history were added to the public tree.

## Evidence model

The characterization keeps evidence tiers separate because they answer different
questions:

| Tier | Meaning | Permitted conclusion |
| --- | --- | --- |
| Locked audit | Pinned audit repository and locked release references | What was deliberately recorded as the Milestone 1 boundary |
| Observed immutable source | Read-only source inspection | What the inspected source currently does or names |
| Safe synthetic probe | Dummy arrays/tensors only | Narrow behavioral semantics, not scientific results |
| Checkpoint schema only | Keys, metadata, and shapes from weights-only inspection | Shape compatibility clues, not training provenance |
| Manuscript variants | Separate manuscript-era statements | Variant claims, not a selected authority |
| Current tested runtime | Current CPU imports and dummy execution | Present portability evidence, not historical environment proof |
| Public compatibility config | Milestone 1 YAML/configuration layer | Portable compatibility defaults, not scientific implementation |

The authoritative contract files are:

- `contracts/source_inventory.json`
- `contracts/data_preprocessing_contract.json`
- `contracts/vae_contract.json`
- `contracts/rectified_flow_contract.json`
- `contracts/evaluation_contract.json`
- `contracts/runtime_contract.json`

The companion interpretation documents are:

- `docs/SCIENTIFIC_SOURCE_CONTRACTS.md`
- `docs/CANONICAL_ENTRYPOINTS_AND_DUPLICATES.md`
- `docs/MIGRATION_PLAN.md`
- `docs/MILESTONE_2A_HANDOFF.md`

## Stable cross-contract facts

The following values are mutually consistent across the contracts:

- REALDISP ideal-placement, right-thigh, six-channel observed path at 50 Hz.
- Source codes `1, 3, 4, 33` map to encoded labels `0, 1, 2, 3` for walking,
  running, jump_up, and cycling.
- The subject set is `[1, 2, 3, 5, 8, 9, 10, 11, 12, 13, 14, 16]`, giving 12
  LOSO folds.
- The main window/hop is `160/40` samples, or `3.2/0.8` seconds.
- The observed VAE geometry is `B,C,160 -> B,48,40` with stride 4.
- Paper Flow sampling is reverse explicit Euler, 10 steps, seed 42, 500
  samples per class.
- Evaluation uses macro-F1 as the primary metric, labels `0..3`,
  `zero_division=0`, and fold sample SD with `ddof=1`.

These facts describe compatibility contracts. They do not prove that every
manuscript number came from this exact code path.

## Deliberately unresolved findings

### Flow width

Manuscript evidence includes `C=128`; later manuscript, wrapper, and observed
checkpoint-shape evidence includes `C=256`. Milestone 2A does not choose between
them. Exact Flow migration is gated on a written width decision plus
state-dict/forward parity tests.

### VAE schedule

Older manuscript/runtime defaults report `1.0/0.1` reconstruction weights and
beta `0.005 -> 0.00001` with decay `0.7`. Later manuscript/wrapper evidence
reports `0.5/0.1` and beta `0.08 -> 0.04` with decay `0.995`. Checkpoints contain
weights only and cannot settle this training provenance question.

### Three-channel lineage

The source preprocessing path observed during characterization always extracts
and returns six channels and ignores `ABLATION_ACC_ONLY`, while a structurally
three-channel checkpoint exists and public/manuscript policy requires separate
three-channel training. The exact 3CH lineage is blocked. No inference-time
channel dropping is implied.

### Split and leakage semantics

The safe VAE subject validation split uses `0.15`; the CNN internal validation
stage uses `0.20`. They are separate stages. The observed duplicate audit checks
train versus validation windows and raises on overlap; it does not prove
train-versus-held-out-test non-overlap.

### Topology and runtime

The documented Flow trainer and the combined LOSO import target are byte-identical
files with different callers. The source has no historical environment lock,
contains hardcoded roots and PowerShell orchestration, and has an undefined
`supplementWin240` wrapper variable. These are migration gates, not reasons to
infer missing evidence.

## What the public repository establishes

The public repository remains a small, auditable compatibility boundary. It
contains configuration and path primitives, evidence-labeled documents, and
these Milestone 2A contracts. It still does not contain scientific execution
code or participant-derived artifacts. The next migration remains gated by the
prerequisites in `docs/MIGRATION_PLAN.md`.

## Verification boundary

Validation for this milestone is limited to JSON parsing and cross-contract
consistency, path and artifact scanning, the existing public skeleton tests,
Git scope/integrity checks, and safe static/dummy probes. Full REALDISP
experiments, training, generation, evaluation, and checkpoint parity are not
tested and are not claimed.
