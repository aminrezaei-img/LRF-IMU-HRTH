# Milestone 3D evaluation parity report

Milestone 3D migrated the active four-scenario evaluator and executed the
public preprocessing, VAE, Flow, generation, RF, and CNN surfaces against the
immutable REALDISP/checkpoint/result references. Large and participant- or
checkpoint-derived payloads remained outside Git.

## Protocol

- Folds: subjects 01, 02, 03, 05, 08, 09, 10, 11, 12, 13, 14, and 16.
- Scenarios: TRTR, scarce real (up to two/class), TSTR (500/class), and TSTR +
  scarce.
- RF: 100 trees, seed 42, one job, otherwise scikit-learn defaults.
- CNN: one run-level seed before the ordered TRTR, scarce, TSTR, TSTR + scarce
  sequence. Model initialization consumes the sequential global Torch RNG;
  each scenario's DataLoader uses its own seed-42 generator. CuDNN benchmark is
  disabled and deterministic mode enabled.
- Metrics: labels 0--3, macro F1, accuracy, per-class F1, and fold-wise
  retention. Aggregate SD uses `ddof=1`.
- Confusion aggregation: `nanmean` across normalized fold matrices;
  `cm_count[i,j]` is the number of folds whose normalized cell exceeds `1e-6`,
  not a participant or sample count.
- Paper sampler: seed 42, batch size 100, ten reverse-Euler steps.
- Classifier training used the VAE-safe training partition and excluded the
  subject-level VAE validation partition, matching the active historical
  evaluator. CNN then used its separate 0.20 window-level validation split.

## Results

| Surface | Historical macro-F1 | Regenerated macro-F1 | Verdict |
| --- | ---: | ---: | --- |
| 6CH RF TRTR | 0.985067 +/- 0.021051 | 0.985060 +/- 0.021056 | Matches stored precision; deterministic real-only path |
| 6CH RF scarce | 0.400442 +/- 0.087752 | 0.400442 +/- 0.087757 | Matches stored precision |
| 6CH RF TSTR | 0.955942 +/- 0.081307 | 0.961957 +/- 0.061189 | Partial; fold-level differences retained |
| 6CH RF TSTR + scarce | 0.950708 +/- 0.087462 | 0.959769 +/- 0.061493 | Partial; fold-level differences retained |
| 3CH RF TRTR | 0.980425 | 0.980421 +/- 0.026714 | Strong empirical compatibility |
| 3CH RF TSTR | 0.980042 | 0.978561 +/- 0.057279 | Partial empirical compatibility |
| 6CH CNN TRTR | 0.989590 +/- 0.031887 | 0.995706 +/- 0.011248 | Stochastic/runtime agreement, not exact |
| 6CH CNN scarce | 0.340242 +/- 0.189706 | 0.238208 +/- 0.106294 | Partial; fold-level differences retained |
| 6CH CNN TSTR | 0.850340 +/- 0.167847 | 0.894868 +/- 0.146825 | Partial stochastic/runtime agreement |
| 6CH CNN TSTR + scarce | 0.857833 +/- 0.144714 | 0.906602 +/- 0.147315 | Partial stochastic/runtime agreement |

Fresh 6CH RF TSTR retention was `0.976976 +/- 0.065137`; fresh 3CH RF TSTR
retention was `0.998303 +/- 0.057464`. Corrected 6CH CNN TSTR retention was
`0.899301 +/- 0.150700`, and CNN TSTR + scarce retention was
`0.910125 +/- 0.144982`. Retention was calculated per fold before aggregation.

An earlier CNN execution incorrectly reset the global Torch RNG at each
scenario boundary. Its metric JSON is explicitly invalidated and excluded from
the parity contract. The corrected 12-fold run generated fresh samples through
the public VAE/Flow path, seeded once per fold before model construction, and
ran scenarios in historical order.

## Device-specific generation diagnosis

Historical subject-01 cache metadata records CUDA generation. This validation
environment had PyTorch 2.7.1 CPU only. CPU and CUDA use different random-number
streams, so seed 42 does not imply identical initial noise across devices.
Fresh subject-01 6CH RF TSTR macro-F1 was `0.823290`, versus the historical
`0.739663`.

Using the immutable historical subject-01 cache with the public evaluator
reproduced the unrounded historical result exactly: accuracy `0.7489711934`,
macro-F1 `0.7396629077`, retention `0.7396629003`, and per-class F1
`[0.964706, 0.715909, 0.827586, 0.450450]`. This isolates the discrepancy to
fresh runtime/device generation, not public preprocessing, RF, or metrics.

The machine-readable contract contains every fold's reference/regenerated
metrics, labeled per-class F1 values, normalized TSTR + scarce confusion
matrix, aggregate `nanmean`/`cm_count` verification, population counts, and
cache identity. It contains no windows, tensors, predictions, checkpoints, or
generated arrays.

## Status

- Gate A evaluation unit parity: PASS.
- Gate B subject-01 6CH RF: PARTIAL for fresh CPU generation; exact evaluator
  parity with the historical cache.
- Gate C all-fold 6CH RF: PARTIAL, with explicit fold-level differences.
- Gate D all-fold 3CH RF: PARTIAL strong empirical compatibility; historical
  parser lineage remains unresolved.
- Gate E all-fold 6CH CNN: PARTIAL stochastic/runtime agreement after the
  corrected run-level seed execution. Optional 3CH CNN was not run.

`exact_paper_reproduction=false` remains unchanged.
