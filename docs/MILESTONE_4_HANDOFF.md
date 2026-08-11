# Milestone 4 handoff

## Result

Milestone 4 adds a thin `reproduce-core` command around the accepted public
preparation, checkpoint, paper-generation, evaluation, aggregation, and
reference-comparison functions. It adds no model, preprocessing, classifier,
or metric logic.

`exact_paper_reproduction` remains `false`.

## Execution contract

A non-dry run requires all three explicit filesystem roots and
`--write-results`. The command never copies REALDISP or checkpoints. Generated
arrays, their checksum manifests, fold metrics, and the run manifest are
written only below the selected external output root.

```text
python -m lrf_imu reproduce-core \
  --data-root <REALDISP-ideal-log-root> \
  --checkpoint-root <historical-source-or-model-weights-root> \
  --output-root <external-output-root> \
  --sensor six_channel \
  --held-out-subject 1 \
  --classifier rf \
  --write-results
```

Use `--all-folds` instead of `--held-out-subject` to select the canonical 12
folds. `--scenario` may be repeated. `--dry-run` performs no data/checkpoint
read and no write. `--resume` validates the run fingerprint, completed result
hashes, and synthetic-cache identity/checksum before skipping work.

The optional `--reference-report` accepts the Milestone 3D evaluation parity
JSON. It records historical/current signed and absolute differences fold by
fold. It intentionally applies no proximity threshold and makes no new parity
claim.

## External artifacts

- `reproduce_core_manifest.json`: run identity, configuration hash, runtime,
  attempts, timing, checkpoint/cache identities, interruption/failure state,
  and result hashes.
- `reproduce_core_report.json`: fold results, independently aggregated summary,
  and optional reference comparison.
- `evaluation/<sensor>/<classifier>/subject_XX.json`: small metric/count record.
- `generated/<sensor>/subject_XX/*.npz`: generated standardized windows; external
  scratch only, never a Git artifact.
- adjacent `*.manifest.json`: VAE/Flow/config hashes and paper-sampler identity.

Manifest and result JSON writes, plus generated NPZ replacement, are atomic.
An interrupted/failed run records a retryable state; resuming reuses only
validated completed work.

## Verification scope

Focused tests cover the no-write dry-run, explicit write gate, canonical source
and model-weight checkpoint layouts, synthetic composition, completed-fold
resume, retryable failure state, and threshold-free reference comparison.

Milestone 4 does not retrain models, change ten-step generation, change
classifier populations, distribute checkpoints, or resolve the known
manuscript/wrapper discrepancies.

## Execution evidence

The 12-fold six-channel RF plan ran from a foreign working directory and made
no output directory.

A real subject-01 CPU smoke used only public package code, REALDISP ideal logs,
historical paired 6CH VAE/Flow checkpoints, and ten reverse-Euler steps. It
generated one window per class to exercise orchestration economically. The fold
counts were 2,399 classifier-training, 280 VAE-validation, and 243 held-out
windows. All four RF scenarios completed. The reduced-sample comparison was
explicitly labeled `protocol_mismatch_descriptive_only`; its metrics are a smoke,
not a paper-result rerun.

A second identical invocation with `--resume` validated the run fingerprint
and skipped the completed fold. Checkpoint/cache hashes and 12 descriptive
metric comparisons remain in external metadata. After verification, the one
14,714-byte generated NPZ was deleted; its checksum manifest was retained.

Final verification: 7 focused tests passed; the complete suite passed 165 tests
with 1 optional skip. Ruff, mypy, compile/import, foreign-CWD CLI, diff, path,
and artifact scans passed. No data, checkpoints, generated arrays, or Results
payload entered the staging repository.
