# LRF-IMU

This repository is the Milestone 1 boundary for a public LRF-IMU release. It
establishes a small, auditable home for portable configuration and path
primitives while the scientific implementation and release decisions remain
under review.

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

The current package has one runtime dependency, PyYAML. The `test` optional
extra supplies pytest for the repository test suite. From the repository root,
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

No model, preprocessing, generation, or evaluation entry point is promised by
this milestone. Later migration work must preserve the audit discrepancies and
the data-release restrictions instead of treating these configs as proof of
scientific parity.

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
