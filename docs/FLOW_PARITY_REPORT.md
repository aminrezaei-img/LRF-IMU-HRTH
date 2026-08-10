# Flow parity report - Milestone 3C

## Executive result

All requested scientific gates passed on CPU without serializing tensor values or generated artifacts. This is implementation parity evidence, not an exact-paper claim: `exact_paper_reproduction=false`, and the 128/256 width conflict remains unresolved.

## Source and public convention

The source reference uses `models/unet_1d.py`, `LRF/rectified_flow.py`, and the byte-identical training scripts `1_Rectified_Flow_training.py` and `rectified_flow_training.py`. The public boundary preserves the source convention: posterior-mean `z0`, Gaussian `z1`, `zt=(1-t)z0+t*z1`, target `z1-z0`, model time `1000*t`, MSE velocity loss, and reverse Euler `z <- z-v*dt`. Raw SHA-256 hashes for all source and final staged implementation files are in `contracts/flow_parity_report.json`.

## Gate A - synthetic Flow smoke

Both public width profiles, 128 and 256, constructed and ran on CPU with input/output shape `[2,48,40]`. Time and class conditioning changed the output, interpolation matched the equation with max/mean error 0/0, target and loss were finite, one reverse Euler step was finite, ten-step trajectories were finite, and repeated seed 42 sampling matched with max/mean error 0/0.

## Gate B - isolated original/public numerical parity

The original source U-Net and public U-Net were loaded in isolated namespaces with identical initialization/state, latent inputs, times, and labels. At both widths, time embedding, class-conditioned forward, interpolation, target velocity, MSE loss, one Euler step, and ten-step sampled latent all reported max absolute error 0 and mean absolute error 0 at tolerance `1e-6`. State key counts were 89 and key sets matched.

## Gate C - historical checkpoint parity

The 6CH full and 3CH ablation subject-01 Flow checkpoints were loaded with weights-only loading. Each has exact root keys `config`, `epoch`, `history`, `opt`, `unet`, `val_loss`, 89 float32 U-Net tensors, width 256, latent channels 48, latent time steps 40, and four classes. Matching subject-01 VAE checkpoints have 26 state tensors, latent channels 48, stride 4, and native length 160. Flow velocity, Euler, ten-step latent, and paired decoder comparisons all had max/mean error 0/0. A 128-width request and a cross-width/channel VAE/Flow pairing were explicitly rejected.

Safe checkpoint metadata, relative source paths, byte sizes, payload hashes, epochs, history/optimizer key summaries, validation losses, and state-shape digests are recorded without payload values in the contract and external evidence.

## Gate D - deterministic generation parity

For each channel set, classes 0-3 were sampled once with seed 42 and the ten-step paper sampler. Initial noise, final latent, and decoded standardized outputs were finite and matched original/public operations with max/mean error 0/0. Shapes were `[4,48,40]` for latents and `[4,6,160]` or `[4,3,160]` for decoded standardized windows. Inverse-normalized physical output was not available for this synthetic-only gate and was checked in Gate E.

## Gate E - one real fold

The REALDISP ideal-log root was prepared through the public M3A path with subject 01 held out. The fold yielded 2,399 training, 280 validation, and 243 held-out windows; held-out class counts were cycling 86, jump_up 12, running 63, and walking 82. Six-channel windows were `[6,160]`; the train-only per-channel standardizer had mean/std shapes `[1,6,1]`, fitted over 2,399 windows, and was not persisted. Four class samples generated with the matching subject-01 VAE/Flow pair had zero max/mean error for latent, decoded standardized, and inverse-normalized physical outputs.

## Paper and website separation

Paper/TSTR generation is the ten-step `paper` profile. Website export is a separate `website_trajectory` profile with 100 steps, every second state retained, 51 states, native 160-sample windows, 40-sample linear overlap-add, four segments, ten seconds, and seed formula `base + subject*1000 + activity*100`. Its metadata asserts `paper_tstr_samples=false`; website output is never labeled paper/TSTR.

## Files and evidence

The minimal public implementation scope is the Flow model, training, generation, checkpoint, CLI, and package export files plus `tests/test_flow_m3c.py`; the seven named M3C documents and `contracts/flow_parity_report.json` are the only documentation/evidence changes. No TSTR/evaluation scripts, checkpoints, Results, participant artifacts, arrays, `.pt`, `.npz`, `.pkl`, or caches are included. External metadata is at `<external-validation-root>/m3c_validation.json`.

## Verification and limitations

Focused M3C/M3B tests: 20 passed, 1 skipped. Complete public suite: 129 passed, 1 skipped. Ruff and mypy passed; compile/import and structured-file parsing passed; artifact/path scanners passed. The Windows pytest ACL workaround used an external basetemp, no cache provider, disabled bytecode, and a temporary isolated fixture outside staging. Exact-paper reproduction remains false, and the width conflict is intentionally unresolved.

Validation did not mutate the protected audit or immutable source; public HEAD remained unchanged while the intentional 18-file M3C delta was applied and left uncommitted for review.
