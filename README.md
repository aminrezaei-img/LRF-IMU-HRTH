# LRF-IMU-HARTH

LRF-IMU-HARTH is a research software release for generating synthetic thigh
accelerometer windows with a class-conditioned latent Rectified Flow model.
It contains the Paper 3 HARTH-family replacement path, VAE and Flow training
and evaluation, conservative DayForge-to-HARTH mapping, and exact-duration
sensor synthesis with stitching and provenance.

The repository also retains the earlier REALDISP-oriented implementation and
its parity records. Those paths are documented separately; the Paper 3 path is
the `harth_walking_speed` composition described below.

## Paper 3 pipeline

```text
HARTH + Adult Walking Speed
            ↓
      preprocessing
            ↓
  ten-class physical-state taxonomy
            ↓
            VAE
            ↓
   latent Rectified Flow
            ↓
 synthetic 3-axis thigh accelerometer window
```

The Paper 3 application layer is separate from model training:

```text
DayForge semantic/contextual evidence
            ↓
     Module B physical-state mapping
            ↓
    Module C exact-duration generation
            ↓
       stitching and fusion
            ↓
   synthetic accelerometer timeline
```

The final DayForge-to-LRF multimodal orchestration is downstream of this
repository's core generator. It is not required to install or use the
generator itself.

## Features

- HARTH plus Adult Walking Speed preprocessing with subject-level LOSO splits.
- Three-channel thigh input at 50 Hz with 160-sample windows and 40-sample hop.
- A VAE with latent geometry `[batch, 48, 40]`.
- Ten-class latent Rectified Flow generation and Module A signal sanity checks.
- Module B mapping for realized mobility, `physical_state_hint`, and the
  derived `in_bed_or_lying_opportunity` handoff.
- Module C exact-duration generation, deterministic per-window seeds,
  multi-window stitching, provenance, and failure audits.
- Metadata-only validation and reproducibility records; participant data and
  large model files remain external.

## Scientific scope

The Paper 3 baseline uses HARTH and Adult Walking Speed. HAR70+ is not part of
the default `harth_walking_speed` composition. The model is a research
generator, not a clinical instrument, sleep detector, anonymization guarantee,
or deployment-ready monitoring system. Synthetic signals should be evaluated
for the intended task and should not be treated as measurements from a real
participant.

The production freeze is the annotated tag
`paper3_lrf_dayforge_handoff_v1` at commit
`150b4de6e58365fdda5fc7279192c136d4e8b064`. Packaging does not change that
scientific behavior.

## HARTH taxonomy

The class IDs are fixed and must be preserved:

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

See [data and taxonomy](docs/data_and_taxonomy.md) for source-label and
exclusion details.

## Installation

The package supports Python 3.10 or newer. The core package needs PyYAML and
NumPy. Training uses PyTorch; evaluation and analysis add the optional
scikit-learn and SciPy dependencies.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[training,evaluation,analysis,test]"
```

On Windows PowerShell, activate the environment with
`.venv\Scripts\Activate.ps1`. The validated production runs used the existing
Conda `py311` environment with CUDA-enabled PyTorch; do not infer CUDA support
from the host alone.

## Quick start

These no-data checks exercise the model interfaces on CPU:

```bash
python -m lrf_imu vae-smoke
python -m lrf_imu flow-smoke
```

For a small decoded HARTH window, supply compatible VAE and Flow checkpoints:

```bash
python -m lrf_imu generate-harth \
  --flow-checkpoint <flow-checkpoint> \
  --vae-checkpoint <vae-checkpoint> \
  --activity sitting \
  --seed 42 \
  --device cpu
```

The command returns metadata for one `[1, 3, 160]` window. It does not write
raw arrays unless a caller explicitly captures or extends the output workflow.

## Data preparation

Place the acquired HARTH-family data outside the repository and use the
canonical composition:

```bash
python -m lrf_imu prepare-harth-data \
  --data-root <harth-family-root> \
  --composition harth_walking_speed \
  --held-out-subject harth:S006 \
  --window-length 160 \
  --hop-length 40 \
  --seed 42
```

The production baseline uses three thigh accelerometer channels, 50 Hz,
training-subject-only per-channel z-score normalization, and exact duplicate
audits across train, validation, and held-out windows. See
[data and taxonomy](docs/data_and_taxonomy.md) and
[reproducibility](docs/reproducibility.md).

The validated external run recorded 55 namespaced subjects: 46 training, 8
validation, and held-out `harth:S006`, with 159,575 training, 18,533
validation, and 8,497 held-out windows. These numbers are a provenance record,
not values to force when using another dataset snapshot.

## Training and evaluation

Use the frozen Paper 3 configuration:

```bash
python -m lrf_imu train-harth-vae \
  --data-root <harth-family-root> \
  --composition harth_walking_speed \
  --held-out-subject harth:S006 \
  --config configs/paper/harth_10class_160_40.yaml \
  --output-dir <vae-output> \
  --seed 42

python -m lrf_imu train-harth-flow \
  --data-root <harth-family-root> \
  --composition harth_walking_speed \
  --held-out-subject harth:S006 \
  --config configs/paper/harth_10class_160_40.yaml \
  --vae-checkpoint <vae-checkpoint> \
  --output-dir <flow-output> \
  --seed 42
```

Module A sanity evaluation is explicit and descriptive:

```bash
python -m lrf_imu evaluate-harth-vae \
  --data-root <harth-family-root> \
  --composition harth_walking_speed \
  --held-out-subject harth:S006 \
  --config configs/paper/harth_10class_160_40.yaml \
  --vae-checkpoint <vae-checkpoint> \
  --output-dir <vae-report>

python -m lrf_imu evaluate-harth-flow \
  --data-root <harth-family-root> \
  --composition harth_walking_speed \
  --held-out-subject harth:S006 \
  --config configs/paper/harth_10class_160_40.yaml \
  --vae-checkpoint <vae-checkpoint> \
  --flow-checkpoint <flow-checkpoint> \
  --output-dir <flow-report> \
  --samples-per-class 100
```

The production Flow configuration records `early_stop_patience`, but the
current `train_flow` loop executes its configured fixed schedule. This is
documented behavior of the frozen baseline, not a reason to alter the release.
See [training](docs/training.md).

## DayForge integration

Module B consumes resolved DayForge mobility intervals and optional read-only
evidence roots:

```bash
python -m lrf_imu map-dayforge-physical-states \
  --dayforge-root <validated-dayforge-root> \
  --derived-root <in-bed-handoff-root> \
  --config configs/paper/dayforge_harth_mapping.yaml \
  --output-dir <mapping-output>
```

The mapping CLI writes a CSV, JSON summary, and Markdown report. The JSON
summary includes baseline, hint-enabled, and combined coverage views. The
mapping rules are conservative: walking hints do not select a speed class,
cycling hints do not infer cycling posture, passive transport is unavailable,
and in-bed opportunity is not physiological sleep.

Module C can then be exercised for one selected person-day:

```bash
python -m lrf_imu synthesize-dayforge \
  --dayforge-root <validated-dayforge-root> \
  --mapping-root <mapping-output> \
  --vae-checkpoint <vae-checkpoint> \
  --flow-checkpoint <flow-checkpoint> \
  --normalization-metadata <normalization-json> \
  --output-dir <fusion-output> \
  --persona <persona-id> \
  --date <YYYY-MM-DD> \
  --seed 42 \
  --device cuda
```

Do not interpret this interface as a command to generate the full DayForge
cohort. See [DayForge mapping](docs/dayforge_mapping.md) and
[stitching and fusion](docs/stitching_and_fusion.md).

## Reproducible runners

The thin wrappers call the canonical CLI and validate checkpoint files before
execution:

```bash
bash scripts/run_lrf_imu.sh \
  --vae-checkpoint <vae-checkpoint> \
  --flow-checkpoint <flow-checkpoint> \
  --class sitting \
  --seed 42 \
  --output output/example.json

bash scripts/run_paper3_dayforge.sh \
  --dayforge-root <validated-dayforge-root> \
  --derived-root <in-bed-handoff-root> \
  --mapping-output output/mapping
```

PowerShell equivalents are provided beside the Bash wrappers. Training uses
the canonical commands above; no second training implementation is included.
See [generation](docs/generation.md) and [validation](docs/validation.md).

## Validation and output structure

Run the release checks from a checkout:

```bash
bash scripts/validate_release.sh
```

The checks cover tests, compilation, CLI help, and static repository hygiene.
The release produces metadata such as `vae_run_meta.json`,
`flow_run_meta.json`, `mapping_summary.json`, segment manifests, and signal
validation reports. Generated arrays, participant data, checkpoints, and
runtime logs belong outside normal Git history.

See:

- [methodology](docs/methodology.md)
- [architecture](docs/architecture.md)
- [data and taxonomy](docs/data_and_taxonomy.md)
- [training](docs/training.md)
- [generation](docs/generation.md)
- [DayForge mapping](docs/dayforge_mapping.md)
- [stitching and fusion](docs/stitching_and_fusion.md)
- [reproducibility](docs/reproducibility.md)
- [validation](docs/validation.md)
- [checkpoints](docs/checkpoints.md)
- [model card](MODEL_CARD.md)
- [data access](DATA_ACCESS.md)

## Citation

Please cite the paper and this software release. Machine-readable metadata is
provided in [CITATION.cff](CITATION.cff).

```bibtex
@article{rezaei2026lrfimu,
  title     = {A latent rectified flow approach to generate synthetic wearable data -- a LABDA solution},
  author    = {Rezaei, Amin and Kjærgaard, Morten and Schipperijn, Jasper},
  journal   = {Machine Learning: Health},
  year      = {2026},
  doi       = {10.1088/3049-477X/ae91ef}
}
```

## Limitations and development status

This is a code-and-documentation release. HARTH-family data, DayForge data,
production checkpoints, and generated IMU arrays are not bundled. No public
checkpoint download URL or DOI is invented here. The project has no license
file; a license decision is required before redistribution under a chosen
open-source license.

The Paper 3 scientific modules are frozen and production-tested. Future work
may publish model artifacts through an appropriate research repository, but
that publication is separate from this source release. Do not change the
taxonomy, evidence hierarchy, exact-duration semantics, or checkpoint lineage
as part of packaging work.
