# LRF-IMU

LRF-IMU is a class-conditioned latent Rectified Flow framework for generating
wearable accelerometer and gyroscope windows. This local release candidate
contains portable preprocessing, VAE, Flow, generation, evaluation, analysis,
and end-to-end orchestration code for the accepted REALDISP study protocol. It addresses controlled generation and utility evaluation of activity-labelled IMU windows under a participant-held-out research design.

## Release status

The public/original implementations passed exact parity checks for
preprocessing, VAE, Rectified Flow, deterministic generation operations, and a
real held-out fold. Core LOSO evaluation and paper analyses were then rerun or
compared fold by fold. Some fresh results are partial because historical CUDA
and current CPU random streams differ or historical artifact lineage is
incomplete. See [results reproduction](docs/RESULTS_REPRODUCTION.md).

`exact_paper_reproduction=false` is intentional. The repository does not
include REALDISP data, historical checkpoints, generated arrays, trained
evaluation models, or historical `Results/` payloads. Licensing also requires
a human decision, so this is a technically validated local candidate rather
than an authorized public release. See
[`LICENSE_DECISIONS.md`](LICENSE_DECISIONS.md).

Start here:

- [Data access and local layout](DATA_ACCESS.md)
- [Reproducibility workflow](REPRODUCIBILITY.md)
- [Model card and use limits](MODEL_CARD.md)
- [Regenerated results and exact limitations](docs/RESULTS_REPRODUCTION.md)
- [Release checklist](docs/RELEASE_CHECKLIST.md)

## Paper identity

The paper record used by this release candidate is:

- **Title:** *A latent rectified flow approach to generate synthetic wearable
  data – a LABDA solution*
- **Authors:** Amin Rezaei, Morten Kjærgaard, and Jasper Schipperijn
- **Journal:** *Machine Learning: Health*
- **DOI:** [10.1088/3049-477X/ae91ef](https://doi.org/10.1088/3049-477X/ae91ef)

The DOI and paper identity are recorded from supplied release metadata and the
audited manuscript candidate. Publisher-side identity verification was not
established in the audit, so this repository does not make a final publication
or equivalence claim.

## What is included

- Portable YAML configurations for six-channel, accelerometer-only, and
  window/hop-grid variants.
- Portable REALDISP preparation, LOSO splits, and training-only normalization.
- Compatible 6CH and separately trained 3CH VAE and Rectified Flow models.
- Ten-step paper generation and a separately labelled website trajectory export.
- RF/CNN evaluation, paper-relevant numerical analyses, and `reproduce-core`.
- Contract tests, parity reports, package metadata, CI, and artifact scanners.

Configuration values remain evidence-labelled compatibility defaults where
historical sources disagree. The detailed limitations are preserved in
[`docs/KNOWN_DISCREPANCIES.md`](docs/KNOWN_DISCREPANCIES.md).

## Audited REALDISP scope

The audit records the controlled subset used by the paper candidate:

| Item | Audited value |
| --- | --- |
| Dataset | REALDISP benchmark |
| Placement and sensor | Ideal placement, right-thigh IMU |
| Channels | `ax`, `ay`, `az`, `gx`, `gy`, `gz` |
| Sampling rate | 50 Hz |
| Activity codes | 1 walking, 3 running, 4 jump_up, 33 cycling |
| Subjects/folds | 1, 2, 3, 5, 8, 9, 10, 11, 12, 13, 14, 16; 12 LOSO folds |
| Main window/hop | 160/40 samples (3.2/0.8 seconds) |
| Standardization | Per-channel z-score fitted on training participants within each fold |

REALDISP data, preprocessed windows, checkpoints, synthetic caches, trained
models, logs, and manuscript history are not included.

## Install

Python 3.10 or newer is supported. The base runtime dependencies are
PyYAML>=6.0 and NumPy>=1.21.3. These are minimum release bounds, not historical
environment pins. Optional extras provide pytest (`test`), PyTorch
(`training`), scikit-learn (`evaluation`), SciPy (`analysis`), and release
tools (`dev`).

```text
python -m pip install -e ".[training,evaluation,analysis,test]"
```

Run the license-safe CPU smokes without REALDISP or historical checkpoints:

```text
python -m lrf_imu vae-smoke
python -m lrf_imu flow-smoke
```

## Prepare external REALDISP data

```text
python -m lrf_imu prepare-data --data-root <realdisp-root> --held-out-subject 1 --sensor-configuration six_channel --validate-only
```

The data root must contain the user-obtained ideal-placement logs described in
[`DATA_ACCESS.md`](DATA_ACCESS.md). Validation and dry-run modes do not write.
Participant-derived arrays stay outside the repository.

## Inspect checkpoints and generate a window

```text
python -m lrf_imu inspect-vae-checkpoint --checkpoint <vae-checkpoint> --channels 6
python -m lrf_imu inspect-flow-checkpoint --checkpoint <flow-checkpoint>
python -m lrf_imu generate --config configs/paper/six_channel_160_40.yaml --vae-checkpoint <vae-checkpoint> --flow-checkpoint <flow-checkpoint> --class-id 0 --count 1 --steps 10 --seed 42 --device cpu
```

Historical checkpoints are required for historical reproduction but are not
distributed by this repository. Generation reports metadata unless the user
provides an explicit output path. The paper sampler uses ten reverse-Euler
steps. Website trajectory export is a distinct 100-step visualization profile,
not the sampler used for reported TSTR results.

## Verification

The focused safety test is `tests/test_no_absolute_paths.py`. It rejects
machine-specific paths, secrets, and prohibited generated artifacts outside
the exact historical-reference exception. Run the full suite from an external
temporary directory after installing the `test` extra:

```text
python -B -m pytest -q -p no:cacheprovider --basetemp <external-temp>
```

The project intentionally has no `LICENSE` file yet; see
[LICENSE_DECISIONS.md](LICENSE_DECISIONS.md).

## Evaluate one LOSO fold

Evaluation reads REALDISP and synthetic caches from explicit external paths. A
single RF fold can be run without writing results:

```text
python -m lrf_imu evaluate --data-root <realdisp-root> --sensor six_channel --classifier rf --held-out-subject 1 --synthetic-cache <external-cache> --scenario trtr --scenario tstr
```

The synthetic cache must have its checksum-validated adjacent identity manifest.
Add an explicit output root and `--write-results` only when output is intended.
See [results reproduction](docs/RESULTS_REPRODUCTION.md) for the exact, partial,
and runtime-sensitive evidence labels.

## Reproduce the core LOSO evaluation

`reproduce-core` composes the accepted public preparation, checkpoint,
ten-step generation, evaluation, aggregation, and optional historical-reference
comparison stages. It supports one fold or the canonical 12 folds, dry-run,
and checksum-validated resume. A real run writes only to the explicit external
output root and requires `--write-results`:

```text
python -m lrf_imu reproduce-core --data-root <realdisp-root> --checkpoint-root <historical-source-or-model-weights-root> --output-root <external-output> --sensor six_channel --held-out-subject 1 --classifier rf --write-results
```

Generated sample caches remain external and are never suitable Git artifacts.
The run manifest records config/checkpoint/cache hashes, runtime, seed, timing,
attempts, interruption/failure state, and result hashes. An optional
`--reference-report contracts/evaluation_parity_report.json` records fold-level
differences without converting proximity into an exact-parity claim. See
`docs/MILESTONE_4_HANDOFF.md`.
