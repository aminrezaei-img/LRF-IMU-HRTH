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
Use `--output <file.npz>` to save the `samples` array and metadata. Without an
output path, the command performs generation and prints metadata without
serializing tensor values.

## Determinism

The seed is part of the generation contract. Same-seed comparisons are
meaningful only when the runtime, device, model files, and preprocessing
metadata are held fixed. Different devices or backend kernels may produce
valid but non-bitwise-identical results.

## Boundary

The core generator produces independent native windows. Exact-duration
assembly, overlap/crossfade policies, interval identity, and multimodal
linkage belong to a separate application repository. The core release does
not impose those policies.

Generated arrays and runtime outputs stay outside normal Git history. Their
metadata should retain checkpoint hashes, class ID, seed, runtime, shape, and
finite-value checks.
