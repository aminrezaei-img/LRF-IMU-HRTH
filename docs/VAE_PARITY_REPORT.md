# VAE parity report

This report records Milestone 3B integration evidence. It is license-safe and
metadata-only: no checkpoint payload, tensor value, participant window, or
machine-specific absolute path is included. The complete machine-readable
record is contracts/vae_parity_report.json.

## Implementations

- Original model: <immutable-source>/VAE/VAE_logic.py, SHA-256
  3C989BB8242236D3107AE75A1533622D955D6E739D0B44103636897D01E80505.
- Public model: src/lrf_imu/models/vae.py, SHA-256
  43F0F81E629E3A1277835487E68C0D24441093EB4C9CE5AD39DC6DD41DAFDDEF.
- Public training surface: src/lrf_imu/training/vae.py, SHA-256
  F474B69C26B6DFF8B5AADA61D37F68B4E83E525BAD0B13E8B50012A26C6493B9.

Public implementation hashes use SHA-256 over UTF-8 file bytes after CRLF is
normalized to LF; no other bytes are changed. Immutable-source hashes are
byte hashes of the source files as stored, with their original line endings
preserved for provenance.

The minimal modifications are explicit channel/shape validation, source
compatible keyword aliases, lazy optional exports, removal of import-time
environment mutation and unrelated dataset orchestration, and a safe
weights_only=True checkpoint boundary. Layer order, parameter names,
dimensions, activations, and forward equations are preserved.

## Numeric evidence

- Gate A: CPU shape contracts are [2, 6, 160] -> [2, 48, 40] -> [2, 6, 160]
  and [2, 3, 160] -> [2, 48, 40] -> [2, 3, 160]; deterministic and fixed-seed
  stochastic checks pass, and unsupported channels are rejected.
- Gate B: isolated original/public initialization and synthetic input compare
  posterior mean, log-variance, deterministic reconstruction, decoded mean,
  and fixed-seed latent sampling for both channel sets. Every maximum absolute
  error is 0.0 with tolerance 0.0.
- Gate C: both named subject-01 checkpoints expose root key vae, 26 tensors,
  latent geometry 48x40, and expected channel-bearing shapes. All compared
  deterministic outputs have maximum absolute error 0.0; a 6CH checkpoint
  requested as 3CH is rejected.
- Gate D: one external REALDISP fold uses subject 01 as held-out, 2399 training,
  280 validation, and 243 held-out windows, with training-only ddof=0
  normalization. The 243 normalized held-out inputs produce latent shape
  [243, 48, 40] and reconstruction shape [243, 6, 160]; original/public
  maximum and mean absolute differences are 0.0. Aggregate MSE is
  0.0647426471 and L1 is 0.1358768344 for both implementations.

Gate E is PARTIAL. The public validation reconstruction completed without Flow,
but comparison with the stored subject-01 validation JSON gives absolute
differences MSE 9.8871e-05, L1 1.1033e-04, FFT 3.4705e-04, and KL
8.4436e-04. The historical VAE-only TSTR/classifier wrapper was not run and
no source Results artifact was copied.

## Reproduction posture

The observed wrapper schedule and older manuscript schedule remain in conflict:
0.5/0.1, 0.08/0.04/0.995 versus 1.0/0.1, 0.005/0.00001/0.7 for the
reconstruction and KL terms. The public profile preserves both evidence
variants and sets exact_paper_reproduction to false.
