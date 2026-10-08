# Validation

Validation is split by responsibility and evidence level.

## Model validation

VAE and Flow sanity evaluation checks:

- checkpoint schema and geometry;
- finite input, latent, reconstruction, and generated values;
- NaN/Inf counts;
- reconstruction mean, standard deviation, RMS, MAE, and MSE;
- latent variance and near-constant diagnostics; and
- descriptive spectral summaries.

These reports are sanity/resemblance checks. They are not distributional
equivalence claims and should not be substituted for a scientific benchmark.

## Release validation

The release checks also require:

- exactly one global VAE/Flow checkpoint pair in the artifact manifest;
- byte-count and SHA-256 verification for every release asset;
- no application-specific source/configuration/script surface;
- no raw data, checkpoints, or generated arrays in normal Git history;
- working preprocessing, training, evaluation, generation, artifact-fetch,
  and artifact-verification command help; and
- portable documentation links and paths.

## Commands

```bash
python -m pytest -p no:cacheprovider -q
python scripts/check_syntax.py
git diff --check
python -m ruff check src/lrf_imu/artifacts.py src/lrf_imu/cli.py scripts/check_syntax.py \
  tests/test_core_release_boundary.py tests/test_production_artifacts.py
```

The repository also provides `scripts/validate_release.sh` and
`scripts/validate_release.ps1` for the same bounded checks plus CLI help. The
legacy REALDISP parity and TSTR/TRTR evidence remain documented in the
historical reports; they are not silently relabeled as Paper 3 HARTH results.
