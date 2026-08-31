# Architecture

## Repository modules

| Concern | Source path | Responsibility |
| --- | --- | --- |
| Data preparation | `src/lrf_imu/data/` | Discovery, label encoding, splits, windows, normalization, and audits |
| Models | `src/lrf_imu/models/` | VAE and latent Flow neural networks |
| Training | `src/lrf_imu/training/` | VAE/Flow objectives, loops, checkpoints, and HARTH orchestration |
| Generation | `src/lrf_imu/generation/` | Latent reverse-Euler sampling |
| Module A | `src/lrf_imu/evaluation/harth_sanity.py` | Descriptive HARTH reconstruction and generation sanity reports |
| Module B | `src/lrf_imu/integration/dayforge.py` and `physical_state.py` | Read-only DayForge evidence parsing and conservative HARTH mapping |
| Module C | `src/lrf_imu/integration/fusion.py` | Exact-duration generation, stitching, provenance, and failure audits |
| CLI | `src/lrf_imu/cli.py` and `evaluation/cli.py` | Public command-line interfaces |
| Configuration | `configs/paper/` | Frozen human-readable experiment and mapping contracts |
| Provenance | training metadata and integration manifests | Checkpoint, seed, source, and output identity |

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

The DayForge bridge is an independent application layer. It consumes resolved
interval records and optional evidence roots, writes mapping metadata, and
passes only eligible classes to the existing Module C interface.

## Configuration boundary

`configs/paper/harth_10class_160_40.yaml` defines the Paper 3 model geometry,
windowing, split, normalization, VAE, Flow, and sampling defaults.
`configs/paper/dayforge_harth_mapping.yaml` defines the mapping policy and
whether the two optional evidence sources are consumed. No configuration file
contains participant data or machine-specific paths.
