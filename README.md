# LRF-IMU-HARTH

LRF-IMU-HARTH is the model repository for a class-conditioned latent
Rectified Flow generator of synthetic right-thigh accelerometry. This core
release contains the complete HARTH-family path from frozen raw data through
preprocessing, VAE training, Flow training, evaluation, and model-only signal
generation.

Application-specific mapping, cohort scheduling, timeline fusion, and
participant-day indexing are deliberately outside this repository. The core
model neither knows nor needs the downstream cohort that requests a signal.

## Frozen production lineage

The accepted model was trained once and then reused unchanged by downstream
studies. It is one global checkpoint pair—not one model per persona and not a
separate set of “210-day weights”.

| Artifact | SHA-256 |
| --- | --- |
| `vae_s3_z48.pt` | `6B118182E14FF04CBD57D66A76986BF3568561F0FD42D02257F8036A0138AAD9` |
| `flow_unet_best.pt` | `F1058EB1FB94809B74B8DFFC24E2697F8C73B046CF2A8E7D24FFF60EC6D63164` |
| `normalization_harth_s006.json` | `F8F5E51085211BE731895744CB002815158CB5E59172E8D0BF7064A77274295B` |

Scientific identities:

- training code: `81b123eab079f5cde7400fee5620e3bffb85a673`;
- accepted generation code: `150b4de6e58365fdda5fc7279192c136d4e8b064`;
- training data: `ntnu-ai-lab/harth-ml-experiments` at
  `dad2cfbe89a26f72f19770419469ac037de200df`;
- composition: `harth_walking_speed`;
- held-out subject: `harth:S006`;
- global seed: `42`.

The model-training, preprocessing, and fixed-window generation files in this
branch are byte-identical to those at the training commit. Repository cleanup
does not change the scientific implementation.

## Pipeline

```text
HARTH + Adult Walking Speed CSV files
                  ↓
       schema and label validation
                  ↓
  subject-safe LOSO split (held out S006)
                  ↓
  160-sample windows / 40-sample hop / 50 Hz
                  ↓
 training-subject-only per-axis normalization
                  ↓
        three-channel VAE training
                  ↓
       latent representation [48, 40]
                  ↓
 ten-class latent Rectified Flow training
                  ↓
 reverse-Euler sampling and VAE decoding
                  ↓
 synthetic window [batch, 3, 160]
```

## Fixed ten-class taxonomy

| ID | Class |
| ---: | --- |
| 0 | `walking_slow` |
| 1 | `walking_moderate` |
| 2 | `walking_brisk` |
| 3 | `running` |
| 4 | `stair_climbing` |
| 5 | `cycling_seated` |
| 6 | `cycling_standing` |
| 7 | `sitting` |
| 8 | `standing` |
| 9 | `lying` |

## Installation

The validated production environment used Python 3.11.11, PyTorch 2.5.1,
CUDA 12.1, and an NVIDIA GeForce RTX 4070 Laptop GPU. A reconstructed and
subsequently validated environment description is provided at
[`environment/production-reconstructed.yml`](environment/production-reconstructed.yml).
It is not represented as a lost byte-for-byte historical environment lock.

```bash
python -m pip install -e ".[training,evaluation,analysis,test]"
```

## Acquire the frozen release assets

Large files are GitHub Release assets rather than Git blobs. After the
`paper3-harth-production-v1` release is published, download and verify the
complete bundle with:

```bash
python -m lrf_imu fetch-production-artifacts \
  --output-dir production-artifacts
```

Download only the exact model checkpoint pair:

```bash
python -m lrf_imu fetch-production-artifacts \
  --output-dir production-artifacts \
  --models-only
```

Verify an existing bundle without downloading:

```bash
python -m lrf_imu verify-production-artifacts \
  --artifact-dir production-artifacts
```

Every file is checked against the packaged manifest at
[`src/lrf_imu/resources/manifests/production_harth_v1.json`](src/lrf_imu/resources/manifests/production_harth_v1.json).
A mismatched file is rejected before use.

The clean scientific metadata and training summary are in that manifest. The
configuration and normalisation are packaged in Git at
`configs/paper/harth_10class_160_40.yaml` and
`src/lrf_imu/resources/normalization/harth_s006.json`. Recovered metadata with
private workstation paths is preserved only in the private lineage lock and
is intentionally not uploaded.

The training-data ZIP contains the exact 55 tracked CSV files used by the
model, plus the upstream README and MIT licence. Extract it so that the
resulting data root contains `harth/` and `adult_walking_speed/`.

## Reproduce preprocessing

```bash
python -m lrf_imu prepare-harth-data \
  --data-root <extracted-data-root> \
  --composition harth_walking_speed \
  --held-out-subject harth:S006 \
  --window-length 160 \
  --hop-length 40 \
  --seed 42
```

The accepted run produced 159,575 training windows, 18,533 validation windows,
and 8,497 held-out windows. These counts are an integrity check for the frozen
data/configuration, not values that the software forces.

## Train the VAE and Flow

```bash
python -m lrf_imu train-harth-vae \
  --data-root <extracted-data-root> \
  --composition harth_walking_speed \
  --held-out-subject harth:S006 \
  --config configs/paper/harth_10class_160_40.yaml \
  --output-dir output/vae \
  --seed 42

python -m lrf_imu train-harth-flow \
  --data-root <extracted-data-root> \
  --composition harth_walking_speed \
  --held-out-subject harth:S006 \
  --config configs/paper/harth_10class_160_40.yaml \
  --vae-checkpoint output/vae/vae_s3_z48.pt \
  --output-dir output/flow \
  --seed 42
```

The Flow training loop intentionally follows the accepted fixed 300-epoch
schedule. See [training](docs/training.md) for the preserved scheduling note.
GPU training can be scientifically equivalent without producing a
byte-identical checkpoint across all hardware and library versions; the exact
accepted checkpoint pair is therefore published as immutable release assets.

## Evaluate and generate

```bash
python -m lrf_imu evaluate-harth-vae \
  --data-root <extracted-data-root> \
  --composition harth_walking_speed \
  --held-out-subject harth:S006 \
  --config configs/paper/harth_10class_160_40.yaml \
  --vae-checkpoint production-artifacts/vae_s3_z48.pt \
  --output-dir output/vae-evaluation

python -m lrf_imu evaluate-harth-flow \
  --data-root <extracted-data-root> \
  --composition harth_walking_speed \
  --held-out-subject harth:S006 \
  --config configs/paper/harth_10class_160_40.yaml \
  --vae-checkpoint production-artifacts/vae_s3_z48.pt \
  --flow-checkpoint production-artifacts/flow_unet_best.pt \
  --output-dir output/flow-evaluation \
  --samples-per-class 100

python -m lrf_imu generate-harth \
  --flow-checkpoint production-artifacts/flow_unet_best.pt \
  --vae-checkpoint production-artifacts/vae_s3_z48.pt \
  --activity sitting \
  --seed 42 \
  --device cuda
```

Generation returns one `[1, 3, 160]` window and its checkpoint/class/seed
metadata. The Python API is `lrf_imu.training.harth.generate_harth_window`.

## Validation

```bash
python -m pytest -p no:cacheprovider -q
python scripts/check_syntax.py
python -m ruff check src/lrf_imu/artifacts.py src/lrf_imu/cli.py scripts/check_syntax.py \
  tests/test_core_release_boundary.py tests/test_production_artifacts.py
git diff --check
```

PowerShell and Bash validation wrappers are available in `scripts/`.

## Repository boundary

This repository stops at model-only windows. It contains no contextual
mapping policy, participant/persona data, interval eligibility decisions,
cohort scheduler, long-interval stitching, multimodal fusion, or fused output.
Those responsibilities belong to a separately versioned integration project,
which consumes this release by tag and SHA-256 identity.

See [repository boundary](docs/repository_boundary.md),
[architecture](docs/architecture.md), [training](docs/training.md),
[generation](docs/generation.md), [reproducibility](docs/reproducibility.md),
and [checkpoints](docs/checkpoints.md).

## Citation and licence status

Citation metadata is provided in [CITATION.cff](CITATION.cff). The upstream
training-data snapshot retains its own README and MIT licence. This software
repository currently has no owner-selected licence; redistribution rights for
the software are therefore not implied until the repository owner adds one.

The software and generated data are research outputs, not clinical devices,
sleep detectors, privacy guarantees, or measurements from real participants.
