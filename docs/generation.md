# Generation

## HARTH window generation

Generation is class-conditioned in the latent space. A seeded noise tensor is
transported by the Flow model with reverse Euler steps and decoded by the
compatible VAE. The canonical small-window command is:

```bash
python -m lrf_imu generate-harth \
  --flow-checkpoint <flow-checkpoint> \
  --vae-checkpoint <vae-checkpoint> \
  --activity sitting \
  --seed 42 \
  --device cpu
```

The output is one decoded window with shape `[1, 3, 160]`, representing 3.2 s
at 50 Hz. Activity can be a canonical class name or an ID from 0 through 9.
The command returns metadata rather than embedding raw tensor values in its
JSON response.

## Determinism

The seed is part of the generation contract. Per-window and per-interval
applications derive stable seeds from the global seed and source identity.
Same-seed comparisons are meaningful only when the runtime, device, model
files, and preprocessing metadata are held fixed. Different devices or
backend kernels may produce valid but non-bitwise-identical results.

## Exact-duration application

For DayForge intervals, use Module C rather than repeatedly calling the
single-window command. Module C computes the target sample count from the
interval duration, crops short outputs, synthesizes long intervals from
independent windows, and stitches them with the existing crossfade policy.
See [stitching and fusion](stitching_and_fusion.md).

Generated arrays and runtime outputs should be stored outside normal Git
history. Their manifests should retain checkpoint hashes, mapping evidence,
seed derivation, interval bounds, and finite-value checks.
