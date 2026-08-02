# Milestone 3B handoff

## Scope

M3B integrates the copied operational one-dimensional VAE, its compatible
training/loss profile surface, and an explicit safe checkpoint boundary. It
does not migrate Rectified Flow, classifier/TSTR evaluation, historical
checkpoints, participant arrays, Results, or raw data.

## Gates

| Gate | Result | Evidence |
| --- | --- | --- |
| A: synthetic CPU smoke | PASS | 3CH/6CH shape, deterministic reconstruction, fixed-seed stochastic behavior, and invalid-channel rejection. |
| B: original/public numeric parity | PASS | Isolated namespaces; 26-key initialized state and all compared outputs have max error 0.0 for 3CH and 6CH. |
| C: historical checkpoint parity | PASS | Both named subject-01 checkpoints load with weights_only=True, root vae, 26 tensors, expected shapes, and deterministic max error 0.0; cross-channel mismatch rejected. |
| D: one real fold | PASS | External REALDISP subject 01 held out; 2399/280/243 train/validation/test windows; training-only normalization; 243 held-out normalized windows match with max/mean error 0.0. |
| E: VAE-only evaluation | PARTIAL | Public validation reconstruction ran without Flow and was compared with the stored JSON; scalar metric differences remain, and historical TSTR/classifier orchestration was not run. |

## Safety and limitations

All public CLI probes use explicit paths and metadata-only output. No checkpoint
payload, participant window, or tensor value is stored in the public parity
report. The external validation directory contains only concise metadata JSON.
The required schedule conflict and metric mismatch remain unresolved. The
release must continue to report exact paper reproduction as false.

See docs/VAE_PARITY_REPORT.md and contracts/vae_parity_report.json for the
machine-readable evidence.
