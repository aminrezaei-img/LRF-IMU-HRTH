# Architecture

## Repository modules

| Concern | Source path | Responsibility |
| --- | --- | --- |
| Data preparation | `src/lrf_imu/data/` | Discovery, label encoding, splits, windows, normalization, and audits |
| Models | `src/lrf_imu/models/` | VAE and latent Flow neural networks |
| Training | `src/lrf_imu/training/` | VAE/Flow objectives, loops, checkpoints, and HARTH orchestration |
| Generation | `src/lrf_imu/generation/` | Latent reverse-Euler sampling |
| Evaluation | `src/lrf_imu/evaluation/harth_sanity.py` | Descriptive HARTH reconstruction and generation sanity reports |
| Artifact integrity | `src/lrf_imu/artifacts.py` | Release-asset acquisition and SHA-256 verification |
| CLI | `src/lrf_imu/cli.py` and `evaluation/cli.py` | Public command-line interfaces |
| Configuration | `configs/paper/` | Frozen human-readable experiment contracts |
| Provenance | training metadata and production manifest | Checkpoint, seed, source, and output identity |

## Data flow

```text
raw external data
      ↓
data pipeline → normalized [N, C, 160] windows
      ↓                         ↓
  VAE encoder              class labels
      ↓                         ↓
   [N, 48, 40] ← latent Flow training
      ↓
   reverse Euler sampler
      ↓
  VAE decoder → [N, C, 160] synthetic windows
```

The public core ends at independent model windows. Mapping another dataset to
the fixed taxonomy, deciding whether an interval is eligible, assembling long
signals, and building multimodal indexes are downstream application concerns.
They are intentionally absent from the package and consume it only through
the documented model API and immutable artifact hashes.

## Configuration boundary

`configs/paper/harth_10class_160_40.yaml` defines the Paper 3 model geometry,
windowing, split, normalization, VAE, Flow, and sampling defaults. No core
configuration file contains participant data, cohort policy, or
machine-specific paths.

The release-asset boundary is defined independently in
`src/lrf_imu/resources/manifests/production_harth_v1.json`. Large artifacts
are fetched from the repository release and accepted only after byte-count and
SHA-256 verification.
