# Milestone 3C handoff

Status: complete and ready for integration review.

## Scope

This handoff integrates the staged public Flow implementation while preserving accepted M3B VAE behavior. Validation did not mutate the protected audit or immutable source; public HEAD remained unchanged while the intentional 18-file M3C delta was applied and left uncommitted for review. It does not migrate TSTR/evaluation scripts, copy checkpoints or generated arrays, or resolve the scientific 128/256 discrepancy.

## Gates

| Gate | Result | Evidence |
| --- | --- | --- |
| A synthetic smoke | PASS | Widths 128 and 256; `[B,48,40]`; class/time conditioning, source equations, one Euler step, ten steps, fixed seed; formula/determinism max/mean error 0/0 |
| B original/public parity | PASS | Widths 128 and 256; time embedding, forward, interpolation, target, loss, Euler, and ten-step latent max/mean error 0/0 at `1e-6` tolerance |
| C historical checkpoint parity | PASS | Subject-01 6CH and 3CH Flow checkpoints; velocity, Euler, ten-step latent, and paired VAE decoder max/mean error 0/0 |
| D deterministic generation | PASS | Four classes, one per class, seed 42, ten steps; initial noise/latent/decoded outputs finite and max/mean error 0/0 for both channel sets |
| E real fold | PASS | REALDISP, subject 01 held out; standardizer/class counts, latent/decoded/physical output parity max/mean error 0/0 |

## Historical checkpoint record

Both Flow payloads have exact root keys `config`, `epoch`, `history`, `opt`, `unet`, `val_loss`; 89 float32 U-Net tensors; width 256; latent channels 48; latent time steps 40; four classes; and profile `historical_checkpoint_compatibility_256`.

| Pair | Flow size / SHA-256 | VAE size / SHA-256 | Safe metadata |
| --- | --- | --- | --- |
| 6CH full subject 01 | 185,506,434 / `4e28a9dfd28fb21fab8df38a3e6b26c5c2bc24481be2cbc5e8ec50818e194841` | 3,005,794 / `7f11c6d4f6e5ab968a22f075ea5cfae1b2f32a1322a0656f019871eed2ba8e2b` | epoch 228; val_loss 0.0583379752933979; state-shape digest `f80fcbe44b54999c8020fa125efadb7430e743c96500320f561cc5107fd0ec06` |
| 3CH ablation subject 01 | 185,506,434 / `8580a15fb3ce75d52e88de8a4151ba33e3e6db6762a2040f81a9269c031a2726` | 2,990,434 / `2e2431c91ff6ecdfdaa6866ca5fed7846cec05d539b6eb5966e6dbe2117b2bc2` | epoch 228; val_loss 0.034181322902441025; state-shape digest `f80fcbe44b54999c8020fa125efadb7430e743c96500320f561cc5107fd0ec06` |

Cross-channel VAE/Flow pairing and requested width mismatch were rejected. Tensor values were never printed or serialized.

## Website profile

The exporter is separate from paper/TSTR generation: `website_trajectory`, 100 reverse-Euler steps, `record_every=2`, 51 states, native 160-sample windows, four independent segments, 40-sample linear overlap-add, ten seconds, and seed `base + subject*1000 + activity*100` (subject 01/activity 0/base 42 -> 1042). It reports `paper_tstr_samples=false` and never labels its output paper/TSTR.

## Verification and protected state

The focused M3C/M3B tests passed 20 with one skip; the complete public suite passed 129 with one skip. Validation did not mutate the protected audit or immutable source. Public HEAD remained unchanged at `ad21022354e5061ceeb888547eb34910ea730381` while the intentional 18-file M3C delta was applied and left uncommitted for review; the audit remained at `f38bebce36c4f21d857dc084ac8d06759c2c012d`. The immutable source was read only.

The metadata-only run is at `<external-validation-root>/m3c_validation.json`. The unapplied patch is prepared at the requested integration path and is validated with `git apply --check --cached --binary --whitespace=error` against the public baseline HEAD.

## Remaining risks

`exact_paper_reproduction=false`. The source/manuscript width-128 convention and historical width-256 checkpoints remain an unresolved scientific discrepancy. The real-fold check validates in-memory operations and writes no windows or generated arrays.
