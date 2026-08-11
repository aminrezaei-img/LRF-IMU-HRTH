# LRF-IMU

**Latent Rectified Flow for synthetic wearable IMU generation**

LRF-IMU is the research code accompanying:

**A latent rectified flow approach to generate synthetic wearable data – a LABDA solution**  
Amin Rezaei, Morten Kjærgaard, Jasper Schipperijn  
*Machine Learning: Health*  
[https://doi.org/10.1088/3049-477X/ae91ef](https://doi.org/10.1088/3049-477X/ae91ef)

LRF-IMU generates class-conditioned wearable accelerometer and gyroscope signals with a variational autoencoder (VAE) and latent Rectified Flow. The study was evaluated on right-thigh REALDISP recordings using a 12-fold leave-one-subject-out design.

## Overview

The generation pipeline is:

```text
IMU window
    ↓
VAE encoder
    ↓
latent representation [48 × 40]
    ↓
class-conditioned Rectified Flow
    ↓
VAE decoder
    ↓
synthetic IMU window
```

The repository supports:

- **6-channel IMU:** triaxial accelerometer + triaxial gyroscope;
- **3-channel IMU:** separately trained accelerometer-only configuration;
- four activity classes: walking, running, jump-up, and cycling;
- 50 Hz signals;
- 160-sample windows (3.2 s) with a 40-sample hop (0.8 s);
- 12-fold participant-held-out evaluation;
- Random Forest and CNN downstream evaluation;
- segmentation-sensitivity, spectral, physical-plausibility, and privacy analyses.

The paper-generation profile uses **10 reverse-Euler Rectified Flow steps**. A separate 100-step trajectory profile is provided for website visualization and is not used for the reported TSTR results.

## Published study

For the six-channel Random Forest evaluation, the published study reported:

| Evaluation | Macro-F1 |
| --- | ---: |
| Real training → real test (TRTR) | 0.985 ± 0.021 |
| Synthetic training → real test (TSTR) | 0.956 ± 0.081 |
| TSTR retention | 97.1% |
| 2 real samples/class | 0.400 ± 0.088 |
| Synthetic augmentation | 0.951 ± 0.087 |

See [`docs/RESULTS_REPRODUCTION.md`](docs/RESULTS_REPRODUCTION.md) for fold-level reproduction results and the distinction between exact, runtime-sensitive, partial, and blocked comparisons.

## Reproducibility status

The public implementation was compared directly with the original research implementation and historical model checkpoints.

| Component | Status |
| --- | --- |
| Preprocessing and LOSO construction | Exact contract parity |
| VAE operations | Exact numerical parity |
| Historical 6CH/3CH VAE checkpoints | Exact numerical parity |
| Rectified Flow operations | Exact numerical parity |
| Historical 6CH/3CH Flow checkpoints | Exact numerical parity |
| Deterministic 10-step generation | Exact numerical parity |
| Evaluation using the same historical synthetic cache | Exact evaluator parity |
| Fresh CPU-generated TSTR evaluation | Partial/runtime-sensitive reproduction |
| Segmentation-sensitivity grid | Exact reproduction |

Fresh CPU generation and the historical CUDA-associated caches were produced in different runtime/device contexts, so bitwise synthetic-sample parity is not expected. The public evaluator reproduces the stored historical subject-01 result exactly when supplied the same immutable historical cache; fresh-generation differences are retained rather than hidden.

Several historical configuration and artifact-lineage ambiguities remain documented in [`docs/KNOWN_DISCREPANCIES.md`](docs/KNOWN_DISCREPANCIES.md). Therefore:

```text
exact_paper_reproduction = false
```

This does not change the directly verified implementation and checkpoint parity reported above.

## Installation

Clone the repository:

```bash
git clone https://github.com/aminsens/LRF-IMU.git
cd LRF-IMU
```

Install the full research environment:

```bash
python -m pip install -e ".[training,evaluation,analysis,test]"
```

Python 3.10 or newer is supported. Core configuration and preprocessing require NumPy and PyYAML; PyTorch, scikit-learn, SciPy, and testing tools are provided through optional package extras.

## Quick start

The model implementations can be tested without REALDISP or historical checkpoints:

```bash
python -m lrf_imu vae-smoke
python -m lrf_imu flow-smoke
```

Both commands run on CPU.

## REALDISP data

REALDISP is **not distributed with this repository**.

Obtain the dataset separately and validate a participant-held-out fold with:

```bash
python -m lrf_imu prepare-data \
    --data-root <realdisp-root> \
    --held-out-subject 1 \
    --sensor-configuration six_channel \
    --validate-only
```

The audited study configuration uses:

- ideal placement;
- right-thigh sensor;
- accelerometer + gyroscope channels;
- subjects `1, 2, 3, 5, 8, 9, 10, 11, 12, 13, 14, 16`;
- walking, running, jump-up, and cycling.

See [`DATA_ACCESS.md`](DATA_ACCESS.md) for the expected file layout and access boundary.

## Historical checkpoints

Historical VAE and Rectified Flow checkpoints are **not included** in the repository.

If you have compatible checkpoints, inspect them with:

```bash
python -m lrf_imu inspect-vae-checkpoint \
    --checkpoint <vae-checkpoint> \
    --channels 6

python -m lrf_imu inspect-flow-checkpoint \
    --checkpoint <flow-checkpoint>
```

The public loaders validate checkpoint structure, tensor geometry, sensor configuration, and model compatibility before execution.

## Generate synthetic IMU

With matching VAE and Flow checkpoints:

```bash
python -m lrf_imu generate \
    --config configs/paper/six_channel_160_40.yaml \
    --vae-checkpoint <vae-checkpoint> \
    --flow-checkpoint <flow-checkpoint> \
    --class-id 0 \
    --count 1 \
    --steps 10 \
    --seed 42 \
    --device cpu
```

Generation is no-write by default. Arrays are written only when an explicit output path and write permission are supplied.

## Evaluate one LOSO fold

```bash
python -m lrf_imu evaluate \
    --data-root <realdisp-root> \
    --sensor six_channel \
    --classifier rf \
    --held-out-subject 1 \
    --synthetic-cache <external-cache> \
    --scenario trtr \
    --scenario tstr
```

Synthetic caches are validated against their adjacent identity manifests before evaluation.

## Reproduce the core experiment

The public workflow composes:

```text
prepare
  → load and validate checkpoints
  → generate synthetic IMU
  → evaluate scenarios
  → aggregate folds
  → optionally compare with historical references
```

Example:

```bash
python -m lrf_imu reproduce-core \
    --data-root <realdisp-root> \
    --checkpoint-root <checkpoint-root> \
    --output-root <external-output> \
    --sensor six_channel \
    --held-out-subject 1 \
    --classifier rf \
    --write-results
```

Use `--all-folds` for the canonical 12-fold experiment and `--resume` for checksum-validated continuation of interrupted runs.

Generated arrays and participant-derived data remain outside the repository. See [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) for the full workflow and evidence boundaries.

## Repository structure

```text
src/lrf_imu/
├── data/          REALDISP preparation and LOSO splitting
├── models/        VAE and Rectified Flow models
├── training/      training objectives and utilities
├── generation/    latent-flow sampling
├── evaluation/    RF/CNN evaluation
└── analysis/      sensitivity, spectral, physical and privacy analyses

configs/           experiment configurations
contracts/         machine-readable parity and provenance records
docs/              reproduction reports and scientific limitations
tests/             synthetic fixtures and regression tests
```

The repository does **not** contain:

- REALDISP data;
- participant-derived windows;
- historical VAE or Flow checkpoints;
- generated synthetic datasets;
- trained evaluation models;
- historical `Results/` payloads;
- manuscript or publisher assets.

## Documentation

For most users:

- [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) — complete reproduction workflow;
- [`DATA_ACCESS.md`](DATA_ACCESS.md) — REALDISP preparation and access boundary;
- [`MODEL_CARD.md`](MODEL_CARD.md) — model scope and limitations;
- [`docs/RESULTS_REPRODUCTION.md`](docs/RESULTS_REPRODUCTION.md) — regenerated results;
- [`docs/KNOWN_DISCREPANCIES.md`](docs/KNOWN_DISCREPANCIES.md) — unresolved historical differences.

The `contracts/` directory and remaining files under `docs/` preserve detailed parity and provenance evidence for readers who need the full audit trail.

## Citation

If you use LRF-IMU, please cite the associated paper:

> Rezaei A, Kjærgaard M, Schipperijn J.  
> *A latent rectified flow approach to generate synthetic wearable data – a LABDA solution.*  
> *Machine Learning: Health*, 2026.  
> DOI: [10.1088/3049-477X/ae91ef](https://doi.org/10.1088/3049-477X/ae91ef)

Machine-readable citation metadata is provided in [`CITATION.cff`](CITATION.cff).

## Scope and limitations

The released models and evaluations cover the documented REALDISP protocol: one right-thigh placement, four activities, and 12 participants. The results do not establish performance for other sensor placements, populations, sampling rates, activities, clinical applications, or deployment settings.

The repository does not claim that generated samples are anonymized or that synthetic data provides a universal privacy guarantee.
