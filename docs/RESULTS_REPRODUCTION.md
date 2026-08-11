# Results reproduction

This page is the concise scientific status for the local release candidate.
It distinguishes implementation parity, deterministic result reproduction,
runtime-sensitive agreement, and work that remains partial or blocked.
`exact_paper_reproduction=false` remains the correct repository setting.

## What was validated

The public preprocessing, VAE, Rectified Flow, generation, evaluation, and
analysis code was exercised with license-safe fixtures and external REALDISP
data/checkpoints. Data, checkpoints, generated arrays, and historical
`Results/` payloads are not included in Git.

| Surface | Status | Meaning |
| --- | --- | --- |
| Preprocessing | Exact contract parity | Six-channel behavior and the explicit reconstructed three-channel path passed fixture and real-fold checks. |
| VAE | Exact implementation parity | Public/original deterministic operations matched on synthetic inputs, historical 6CH/3CH checkpoints, and a real fold. |
| Rectified Flow | Exact implementation parity | Public/original velocity, Euler, ten-step latent, and decoding operations matched for historical 6CH/3CH checkpoints. |
| Historical subject-01 RF cache | Exact evaluator parity | The public evaluator reproduced the stored unrounded metrics when given the same immutable cache. |
| Fresh all-fold RF generation | Partial result reproduction | CPU and historical CUDA random streams differ; every nonzero fold difference is retained. |
| CNN | Statistical/runtime agreement | Training is runtime-sensitive and is not described as bitwise parity. |
| Paper analyses | Mixed PASS/PARTIAL/BLOCKED | Each analysis is labelled below; no aggregate proximity is promoted to exact parity. |

## Core LOSO evaluation

Values are mean macro-F1 +/- sample SD across the 12 held-out subjects.

| Experiment | Historical | Regenerated | Verdict |
| --- | ---: | ---: | --- |
| 6CH RF TRTR | 0.985067 +/- 0.021051 | 0.985060 +/- 0.021056 | Real-only path matches stored precision |
| 6CH RF scarce real | 0.400442 +/- 0.087752 | 0.400442 +/- 0.087757 | Matches stored precision |
| 6CH RF TSTR | 0.955942 +/- 0.081307 | 0.961957 +/- 0.061189 | Partial; fold differences retained |
| 6CH RF TSTR + scarce | 0.950708 +/- 0.087462 | 0.959769 +/- 0.061493 | Partial; fold differences retained |
| 3CH RF TRTR | 0.980425 | 0.980421 +/- 0.026714 | Strong empirical compatibility |
| 3CH RF TSTR | 0.980042 | 0.978561 +/- 0.057279 | Partial empirical compatibility |
| 6CH CNN TRTR | 0.989590 +/- 0.031887 | 0.995706 +/- 0.011248 | Statistical/runtime agreement |
| 6CH CNN scarce real | 0.340242 +/- 0.189706 | 0.238208 +/- 0.106294 | Partial |
| 6CH CNN TSTR | 0.850340 +/- 0.167847 | 0.894868 +/- 0.146825 | Partial statistical/runtime agreement |
| 6CH CNN TSTR + scarce | 0.857833 +/- 0.144714 | 0.906602 +/- 0.147315 | Partial statistical/runtime agreement |

Fresh 6CH RF TSTR retention was `0.976976 +/- 0.065137`; fresh 3CH RF
retention was `0.998303 +/- 0.057464`. Retention was computed per fold before
aggregation. The optional 3CH CNN run was not completed.

Subject 01 demonstrates the runtime distinction directly. Fresh CPU 6CH RF
TSTR macro-F1 was `0.823290`, while the historical CUDA-generated result was
`0.7396629077`. Supplying that exact historical cache to the public evaluator
reproduced accuracy `0.7489711934`, macro-F1 `0.7396629077`, and retention
`0.7396629003`. This diagnoses generation-device RNG as the source of the
fresh-run difference, not the public RF or metric implementation.

Full fold-level values and signed differences are in
[`contracts/evaluation_parity_report.json`](../contracts/evaluation_parity_report.json).

## Paper-relevant analyses

| Analysis | Result | Status |
| --- | --- | --- |
| VAE-only 6CH TSTR | Historical `0.443404 +/- 0.171059`; regenerated `0.448935 +/- 0.177147` | PARTIAL: all 12 folds ran, but the historical checkpoint lineage is unavailable and every TSTR fold differed |
| Segmentation sensitivity | 171/171 numeric cells across nine settings matched; maximum absolute difference 0 | PASS |
| Acceleration >10g | 0/320,000 points for newly executed subject 01; stored summaries report zero for all folds | PARTIAL: only one raw fold rerun |
| Spectral comparison | log-PSD correlation `0.966280814`; 10-25 Hz synthetic/real power ratio `0.445482737` | PARTIAL: recomputed from stored aggregate PSD curves; high-frequency attenuation is explicit |
| Membership inference | true holdout `0.495199 +/- 0.020137`; separate post-hoc audit `0.514989 +/- 0.016021` | PARTIAL: stored attacks summarized, not rerun |
| Reconstruction attack | 0/240 optimized attempts succeeded | PARTIAL: stored run; 600 targets were configured but the source loop optimized only 240 |
| 3CH VAE-only | No mapped historical comparison artifact | BLOCKED |

The two membership-inference values represent different threat models and are
not combined. Zero successes under one reconstruction criterion is not proof of
anonymization or a general privacy guarantee.

## Reproduce with external assets

Install the relevant extras:

```text
python -m pip install ".[training,evaluation,analysis]"
```

Inspect checkpoint structure without printing tensor values:

```text
python -m lrf_imu inspect-vae-checkpoint --checkpoint <vae-checkpoint> --channels 6
python -m lrf_imu inspect-flow-checkpoint --checkpoint <flow-checkpoint>
```

Run one RF fold using an already generated, externally stored cache:

```text
python -m lrf_imu evaluate --data-root <realdisp-root> --sensor six_channel --classifier rf --held-out-subject 1 --synthetic-cache <external-cache> --scenario trtr --scenario tstr
```

Run the public end-to-end composition with historical external checkpoints:

```text
python -m lrf_imu reproduce-core --data-root <realdisp-root> --checkpoint-root <checkpoint-root> --output-root <external-output> --sensor six_channel --held-out-subject 1 --classifier rf --write-results
```

Replace `--held-out-subject 1` with `--all-folds` for the canonical cohort.
Use `--dry-run` to inspect a plan without reading data/checkpoints or writing,
and `--resume` to continue a checksum-compatible external run.

Historical checkpoint distribution is not part of this candidate. A clean
checkout can run synthetic smokes, but historical result reproduction requires
user-supplied REALDISP access and matching external VAE/Flow checkpoints.

## Why exact paper reproduction remains false

The historical VAE schedule differs between manuscript and wrapper evidence;
Flow width is 128 in one manuscript lineage but 256 in historical checkpoints;
fresh CPU and historical CUDA generation use different random streams; the
historical 3CH parser lineage is incomplete; some analysis/table provenance is
partial; and the accepted figure package is not identified. See
[`KNOWN_DISCREPANCIES.md`](KNOWN_DISCREPANCIES.md).

This status does not prevent a technically valid local code candidate. It does
prevent claims that every paper number, table, or figure is exactly regenerated.
