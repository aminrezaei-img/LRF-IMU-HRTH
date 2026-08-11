# Reproducibility status

## Current evidence boundary

This candidate contains portable preprocessing, VAE, Rectified Flow,
generation, RF/CNN evaluation, paper-relevant numerical analyses, and a thin
`reproduce-core` orchestration command. Synthetic contract tests and external
REALDISP/checkpoint runs establish exact implementation parity for the
deterministic scientific operations described in the parity reports.

A clean checkout does not contain REALDISP, historical VAE/Flow checkpoints,
generated arrays, trained evaluators, or historical `Results/` payloads.
License-safe smokes run without those assets; historical result reproduction
requires user-supplied data access and matching external checkpoints.

Evidence is reported at four levels:

1. exact public/original implementation parity on identical inputs and states;
2. exact result parity when the same immutable historical artifact is supplied;
3. statistical/runtime or partial reproduction when fresh execution differs;
4. blocked where the necessary historical lineage is unavailable.

No aggregate closeness overrides a differing fold. The authoritative values are
in [`docs/RESULTS_REPRODUCTION.md`](docs/RESULTS_REPRODUCTION.md), with
machine-readable fold details in `contracts/`. Configuration values remain
compatibility defaults where manuscript, wrapper, and checkpoint evidence
conflict. `exact_paper_reproduction=false` remains unchanged.

## Audited data boundary

Use the dataset and preprocessing boundary in [DATA_ACCESS.md](DATA_ACCESS.md):
REALDISP ideal placement, right-thigh six-channel input, 50 Hz, four audited
activity codes, 12 listed LOSO subjects, 160/40 main windowing, and training-
participant-only standardization. These facts describe the audited task, not a
general-purpose REALDISP loader.

The audit also records a source-state drift: the current immutable source
`Results/` tree was observed at 4,667 files and 34,811,675,494 bytes, compared
with the pinned audit baseline of 4,619 files and 34,791,553,468 bytes. The
additional 48 website-trajectory JSON files and four rewritten flow-trajectory
images are excluded from this milestone and were not copied.

## Configuration and safety verification boundary

The release smoke checks should:

1. parse all three YAML files;
2. load each config with `device="cpu"` and a dotted override;
3. compile/import the configuration package without creating cache files;
4. run `tests/test_no_absolute_paths.py` with the cache provider disabled and a
   temporary base directory outside the repository; and
5. verify the seven locked/reference hashes and repository/source integrity.

These checks validate configuration and release safety. Scientific result evidence
is reported separately below and in
[docs/RESULTS_REPRODUCTION.md](docs/RESULTS_REPRODUCTION.md).

## Milestone 3A evidence

The seven data-preparation lanes are covered by synthetic-only fixtures and pure
contract tests. The compact parity flow reports 16 training, 7 validation, and 8
held-out windows with held-out subject 05, validation subject 01, and training
subjects 02 and 03. It exercises both 6CH and the explicit reconstructed 3CH path,
the default filter-before-runs mode and strict contiguity, training-only ddof=0
normalization, all-pair duplicate auditing, and SHA-1 exact-window identity.

The final integration command is:

    python -B -m pytest -p no:cacheprovider --basetemp <external-basetemp>

The M3A run collected 109 tests and completed with 109 passed and 0 warnings.
Additional checks covered CLI help, foreign-working-directory dry-run and
validate-only behavior, explicit metadata writing, overwrite refusal, JSON-safe
metadata, import/compile safety, path and artifact scanners, and 6CH/3CH compact
flows. The output policy was verified to serialize no raw or participant-derived
window arrays.

These results establish a portable public preparation boundary, not a rerun of
the participant study. No participant artifacts, checkpoints, VAE/Flow migration,
full training, full evaluation, or exact-paper equivalence claim is included.
## Milestone 3A correction pass

The executable package declares the same two core runtime requirements in
pyproject.toml and requirements.txt: PyYAML and the unpinned lower-bound
declaration numpy>=1.21.3. The lower bound is a current minimum-safe
compatibility choice for the supported Python floor, not a claim about the
historical research environment.

The six-channel default YAML is also included as intentional package data.
The root paper YAMLs remain human-facing evidence, while
tests/test_packaging.py guards normalized byte synchronization with the
installed-resource copies. The parser and pipeline preserve
split.classifier_window_validation_fraction: 0.20 for the classifier/window
protocol and read split.vae_subject_validation_fraction: 0.15 for the
VAE-safe subject split. The legacy split.validation_fraction: 0.20 key remains
an explicit classifier/window alias.

A clean wheel probe must run both the lrf-imu prepare-data console script and
python -m lrf_imu prepare-data from a foreign working directory with no
repository checkout on the import path. These probes use the synthetic fixture
only and do not establish participant-level reproducibility.

## Milestone 3C reproducibility record

### Runtime and commands

The verification used the bundled Anaconda Python runtime because the literal launcher is environment-dependent. All numerical checks were CPU-only and deterministic. Paths below are placeholders so this document is portable.

```text
python -m lrf_imu flow-smoke
python -m lrf_imu inspect-flow-checkpoint --checkpoint <source-root>/Results/model_weights/flow_weights/6CH/full/subject_01/flow_unet_best.pt
python -m lrf_imu inspect-flow-checkpoint --checkpoint <source-root>/Results/model_weights/flow_weights/3CH/ablation/subject_01/flow_unet_best.pt
python -m lrf_imu generate --config <stage>/configs/paper/six_channel_160_40.yaml --vae-checkpoint <6CH-vae> --flow-checkpoint <6CH-flow> --class-id 0 --count 1 --steps 10 --seed 42 --device cpu
python -m lrf_imu export-trajectories --flow-checkpoint <6CH-flow> --vae-checkpoint <6CH-vae> --subject 1 --activity 0 --base-seed 42 --device cpu
```

The generate and exporter commands were run without an output path. Both exited 0, reported `output_written=false`, and included no tensor values. A foreign-cwd run with a missing-Torch shim exited 2 with one CLI error line and no traceback.

### Gates

| Gate | Scope | Result | Numeric evidence |
| --- | --- | --- | --- |
| A | Widths 128 and 256; `[B,48,40]`; conditioning, interpolation, target, loss, one Euler step, ten-step trajectory, fixed seed | PASS | Formula and determinism max/mean error 0/0; finite CPU tensors |
| B | Isolated original/public namespaces at widths 128 and 256 | PASS | Time embedding, class forward, interpolation, target, loss, Euler, and ten-step latent max/mean error 0/0; tolerance `1e-6` |
| C | Historical subject-01 6CH and 3CH checkpoints plus paired VAEs | PASS | Velocity, Euler, ten-step latent, and paired decoder max/mean error 0/0; 89 Flow keys; wrong-width and cross-width pairings rejected |
| D | Four classes, one sample per class, paper sampler, seed 42 | PASS | Initial noise finite; latent and decoded standardized max/mean error 0/0 for 3CH and 6CH |
| E | REALDISP fold with subject 01 held out | PASS | Standardized and inverse-normalized physical outputs max/mean error 0/0; no arrays serialized |

Gate E used the public M3A preparation contract and the accepted six-channel VAE configuration. The fold contained 2,399 training, 280 validation, and 243 held-out windows of shape `[channels,160]`; held-out class counts were cycling 86, jump_up 12, running 63, walking 82. The per-channel train-only standardizer was fitted over 2,399 windows, with mean/std shapes `[1,6,1]`, and persisted statistics remained false.

### Verification

Focused M3C plus M3B tests: 20 passed, 1 skipped. Complete public suite: 129 passed, 1 skipped. Ruff passed on changed modules and the M3C tests; mypy passed on the five changed Python modules. Compile/import and JSON/YAML/TOML/CFF parsing are recorded in the parity contract.

The metadata-only evidence output is `<external-validation-root>/m3c_validation.json`. It contains checkpoint hashes, sizes, root-key/schema metadata, shapes, and aggregate error statistics, never tensor payloads, generated arrays, windows, or participant artifacts.
## Milestone 3D evaluation record

The public evaluator completed all twelve RF folds for 6CH and separately
trained 3CH checkpoints, and all twelve 6CH CNN folds. Each fold recorded the
classifier-training subjects, excluded VAE-validation subjects, real/test/class
counts, synthetic count, classifier settings, and fold-level historical
comparison. Retention was computed fold-wise and aggregate SD used `ddof=1`.

The fresh runtime was Python 3.12.4, PyTorch 2.7.1 CPU, scikit-learn 1.4.2,
and NumPy 1.26.4. Historical cache metadata identifies CUDA generation; CPU
and CUDA RNG streams differ. Consequently the public evaluator is exact on
the immutable historical subject-01 cache, while fresh CPU TSTR results are
PARTIAL and retain every nonzero fold-level difference.

The corrected 6CH CNN run seeds once per fold before model construction and
executes scenarios in historical order; model initializations therefore consume
the sequential global Torch RNG. Per-scenario DataLoaders retain independent
seed-42 generators. The previous reset-per-scenario run is invalidated and
excluded. Cache manifests validate sensor/fold/config/checkpoint/sample identity
and the generated-array SHA-256. The full metadata-only record is
`contracts/evaluation_parity_report.json`.

## Milestone 3E analysis record

The VAE-only ablation executed all 12 six-channel folds with the source protocol.
Historical versus regenerated TSTR macro-F1 was `0.443404 ± 0.171059` versus
`0.448935 ± 0.177147`; every synthetic fold differed because the checkpoint
lineage named by the historical result files is unavailable. TRTR reproduced
exactly at `0.985060 ± 0.021056`.

The public sensitivity aggregator reproduced 171/171 stored numeric cells
exactly. A public physical execution found 0/320,000 subject-01 synthetic points
above 10g. Stored all-fold PSD curves reproduced log-PSD correlation
`0.966280814` and the attenuated 10–25 Hz power ratio `0.445482737`.

Privacy summaries preserve separate threat models: true-holdout MIA was
`0.495199 ± 0.020137`; the post-hoc best-attack audit was `0.514989 ± 0.016021`.
Reconstruction had zero successes among 240 actual optimizations; the historical
configuration records 600 selected targets because only the first 20 of 50 per
fold entered the optimization loop. These are not privacy guarantees.

## Milestone 4 orchestration record

The orchestration layer contains no new scientific logic. It calls the accepted
public modules in this order:

```text
prepare -> validate/load paired checkpoints -> ten-step paper generation
        -> scenario evaluation -> fold aggregation -> optional reference comparison
```

A 12-fold no-write plan is available from any working directory:

```text
python -m lrf_imu reproduce-core --data-root <realdisp-root> --checkpoint-root <checkpoint-root> --output-root <external-output> --all-folds --dry-run
```

For an executable fold, add `--write-results`. Use `--resume` after a clean
interruption or recorded failure. Resume is accepted only when the run
fingerprint matches; a completed fold is skipped only when its result checksum
matches, and a generated cache is reused only when its adjacent identity
manifest and array checksum validate.

The structured external output contains:

- configuration SHA-256 and package/runtime metadata;
- fold-specific VAE and Flow SHA-256 identities for generated caches;
- paper sampler steps, seed, sample count, and standardized coordinate system;
- preparation population counts but no participant windows;
- fold duration, attempt count, failure/interruption status, and retry state;
- metric-only fold JSON and an independently aggregated report.

The command does not copy data or checkpoints and has no implicit dependency on
the immutable research source. The source tree is one possible explicit
checkpoint root only. Generated NPZ caches are external runtime artifacts and
must not be added to Git.

Reference comparison is descriptive. It preserves signed and absolute
fold-level differences from the selected historical report and applies no
acceptance tolerance. `exact_paper_reproduction` remains `false`.
