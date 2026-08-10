# Milestone 3D handoff

## Delivered

- Source-compatible RF/CNN classifiers, four-scenario population construction,
  metrics/retention/confusion aggregation, validated external cache identity,
  and one-fold/all-fold CLI commands.
- A machine-readable historical artifact map and fold-level parity report.
- Fresh public generation and RF execution for all 12 folds in both 6CH and
  separately trained 3CH configurations.
- A corrected fresh 6CH CNN execution for all 12 folds, using one run-level
  seed before the historical scenario order. The previous reset-per-scenario
  run is invalidated and excluded.

## Execution boundary

All REALDISP windows, checkpoint payloads, generated arrays, predictions,
trained CNN states, and verbose logs remained under an external validation root
or the immutable source tree. No historical Result was overwritten. The public
repository contains only code, tests, documentation, and small metadata/metric
JSON.

Every accepted synthetic cache requires an adjacent manifest identifying the
sensor, held-out subject, VAE/Flow SHA-256, config, seed, steps, samples/class,
implementation, and array SHA-256. A cache is rejected if any requested
identity or checksum differs. Fresh correction-run arrays were deleted after
their manifests, checksums, and metrics were retained.

## Commands

```text
python -m lrf_imu evaluate --data-root <realdisp-root> --sensor six_channel --classifier rf --held-out-subject 1 --synthetic-cache <external-cache> --scenario tstr
python -m lrf_imu evaluate-loso --data-root <realdisp-root> --sensor six_channel --classifier rf --synthetic-root <external-cache-root> --output-root <external-output> --write-results --resume
```

The CLI never generates or writes implicitly. TSTR requires an explicit cache
path/root. `--write-results` is rejected before loading data unless an explicit
output root is supplied. A scenario-only TSTR/scarce request executes TRTR
internally for the fold-wise retention denominator but returns only the
requested scenario. Fresh and resumed fold JSON share the same schema, so
mixed/all-resumed LOSO aggregation is valid.

## Remaining limitations

- Historical caches were generated on CUDA; fresh execution was CPU-only.
  Fold-level RF differences are therefore recorded, not hidden behind close
  aggregate means.
- CNN training is runtime-sensitive and is not labelled exact historical
  parity. The corrected 6CH run is fold-level statistical/result agreement.
- Optional 3CH CNN was not run after the required 6CH CNN completed.
- The reconstructed public 3CH parser remains empirical compatibility evidence,
  not proof of exact historical lineage.
- Existing VAE schedule and Flow width conflicts remain unresolved.

See `contracts/evaluation_parity_report.json` and
`docs/EVALUATION_PARITY_REPORT.md` for the concise evidence.
