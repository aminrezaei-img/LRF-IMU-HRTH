# LRF-IMU

This repository is the public LRF-IMU release boundary through Milestone 3B. It
contains the portable M3A data-preparation boundary and an evidence-labelled
VAE/checkpoint boundary. The scientific implementation and release decisions
remain under review where the evidence is incomplete.

## Paper identity

The paper record used by this skeleton is:

- **Title:** *A latent rectified flow approach to generate synthetic wearable
  data – a LABDA solution*
- **Authors:** Amin Rezaei, Morten Kjærgaard, and Jasper Schipperijn
- **Journal:** *Machine Learning: Health*
- **DOI:** [10.1088/3049-477X/ae91ef](https://doi.org/10.1088/3049-477X/ae91ef)

The DOI and paper identity are recorded from supplied release metadata and the
audited manuscript candidate. Publisher-side identity verification was not
established in the audit, so this repository does not make a final publication
or equivalence claim.

## What Milestone 1 contains

- Portable YAML configurations for the observed six-channel, accelerometer-only,
  and window/hop-grid variants.
- `src/lrf_imu/` configuration and path primitives only.
- Seven byte-preserved audit/locked-reference files under `configs/locked/` and
  `docs/`.
- Provisional release, citation, data-access, model-card, and contribution
  documents.
- Durable empty directories for later scripts, tests, paper artifacts, and
  scientific subpackages.

The configuration values are evidence-labeled compatibility defaults. They are
not an exact paper-reproduction implementation.

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

REALDISP data, preprocessed windows, checkpoints, synthetic caches, results,
logs, and manuscript history are not included. See [DATA_ACCESS.md](DATA_ACCESS.md)
for the provisional access boundary and [REPRODUCIBILITY.md](REPRODUCIBILITY.md)
for the evidence boundary.

## Install and inspect the configuration layer

The base runtime dependencies are PyYAML and NumPy (numpy>=1.20). The lower
bound is an unpinned, minimum-safe runtime declaration for the supported Python
floor and APIs used by this package; it is not a historical environment claim.
The test extra supplies pytest and the optional training extra supplies PyTorch
for VAE operations.
From the repository root,
install the configuration layer and test tools with:

```text
python -m pip install -e ".[test]"
```

The configs accept portable root overrides and a CPU device selection:

```python
from pathlib import Path

from lrf_imu import load_config

config = load_config(
    Path("configs/paper/six_channel_160_40.yaml"),
    base_dir=Path("."),
    data_root=Path("/path/to/realdisp"),
    device="cpu",
    subject=16,
    fold=16,
)
print(config.paths.data_root)
```


The default six-channel profile is packaged as an intentional runtime resource,
so both the lrf-imu console script and python -m lrf_imu prepare-data can load
it from an installed wheel and from a foreign working directory. The
human-facing copies remain under configs/paper/; a packaging test compares
their normalized bytes with the packaged resources.

The split configuration distinguishes the VAE subject validation fraction
(split.vae_subject_validation_fraction: 0.15) from the classifier/window
fraction (split.classifier_window_validation_fraction: 0.20). The historical
split.validation_fraction: 0.20 key is retained as an explicitly documented
classifier/window alias and must agree with the named classifier value.
Milestone 3A adds a metadata-only data-preparation boundary; it does not
migrate VAE, Flow, classifier, generation, or evaluation models. Later work must
preserve the audit discrepancies and data-release restrictions instead of
treating this boundary as proof of scientific parity.

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

## Milestone 3A data-preparation boundary

Milestone 3A integrates seven contract-driven lanes under src/lrf_imu/data/:
activities and schemas, REALDISP discovery/loading, activity-bounded windowing,
subject/window splits, training-only normalization, duplicate auditing, and a
metadata-only pipeline with the lrf-imu prepare-data CLI.

The locked compatibility behavior is:

- 120 tab-separated raw columns, right-thigh signal columns 80 through 85, and
  label column 119; raw activity codes 1, 3, 4, and 33 map to encoded labels
  0, 1, 2, and 3.
- 160-sample windows with a 40-sample hop. The default filter_before_runs mode
  filters the four-class vocabulary before run detection;
  strict_original_contiguity is available when gaps must remain boundaries.
- VAE-safe subject validation uses 0.15 and the compact fixture reproduces 16/7/8
  windows with held-out subject 05, validation subject 01, and training subjects
  02 and 03.
- Standardization is fit on training windows only with population standard
  deviation (ddof=0). Duplicate identity uses canonical exact-window bytes and
  SHA-1.

The public safety boundary is intentional: preparation reads explicit external data
roots, keeps participant-derived arrays in memory, audits all split pairs by
default, and writes only prepare_data_metadata.json after an explicit output
permission flag. Dry-run and validate-only modes never write. The reconstructed 3CH
accelerometer path selects columns 80 through 82 as a separately trained schema; it
is not an inference-time drop from a 6CH input and is not historical lineage evidence.

M3A has no participant artifacts, checkpoints, result payloads, or VAE/model
migration, and it makes no exact-paper reproduction claim. M3B adds only the
public VAE implementation, safe checkpoint inspection/loading, and no-write
CPU/reconstruction entry points; it does not add checkpoints or participant
artifacts. See docs/MILESTONE_3B_HANDOFF.md and docs/VAE_PARITY_REPORT.md for the
evidence summary.

## Milestone 3B VAE boundary

Install the optional VAE dependency before using the model commands:

~~~text
python -m pip install -e ".[test,training]"
~~~

The public model accepts independent declared 6CH or 3CH inputs and preserves
the observed geometry [batch, channels, 160] -> [batch, 48, 40]. It rejects
unsupported channel counts and cross-channel checkpoint use. The following
commands are explicit-path, CPU-safe metadata/reconstruction probes; they do
not write checkpoints, tensors, or participant windows:

~~~text
PYTHONPATH=src python -m lrf_imu vae-smoke
PYTHONPATH=src python -m lrf_imu inspect-vae-checkpoint --checkpoint <external-checkpoint> --channels 6
PYTHONPATH=src python -m lrf_imu reconstruct --config configs/paper/six_channel_160_40.yaml --checkpoint <external-checkpoint> --input <safe-npy-or-npz> --device cpu
~~~

The VAE copy is compatibility evidence, not an exact paper-reproduction
claim. Historical checkpoints, raw data, and Results outputs remain external
and are never copied into this tree.