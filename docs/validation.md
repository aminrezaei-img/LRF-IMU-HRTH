# Validation

Validation is split by responsibility and evidence level.

## Module A

VAE and Flow sanity evaluation checks:

- checkpoint schema and geometry;
- finite input, latent, reconstruction, and generated values;
- NaN/Inf counts;
- reconstruction mean, standard deviation, RMS, MAE, and MSE;
- latent variance and near-constant diagnostics; and
- descriptive spectral summaries.

These reports are sanity/resemblance checks. They are not distributional
equivalence claims and should not be substituted for a scientific benchmark.

## Module B

Mapping validation checks parser identity and time alignment, fixed class IDs,
evidence precedence, unavailable reasons, conflict flags, provenance, and
source immutability. The mapping summary keeps baseline, physical-hint, and
combined coverage visible.

## Module C

Fusion validation checks:

- exact target sample count;
- short-interval cropping;
- long-interval multi-window generation;
- deterministic seed derivation;
- `[start,end)` timestamp alignment;
- finite arrays and expected channel geometry;
- stitching boundary jumps;
- provenance completeness;
- unsupported intervals without arrays; and
- generation failures separately from intentional unavailability.

## Commands

```bash
python -m pytest -p no:cacheprovider -q
python -m compileall -q src
git diff --check
python -m ruff check src tests
```

The repository also provides `scripts/validate_release.sh` and
`scripts/validate_release.ps1` for the same bounded checks plus CLI help. The
legacy REALDISP parity and TSTR/TRTR evidence remain documented in the
historical reports; they are not silently relabeled as Paper 3 HARTH results.
