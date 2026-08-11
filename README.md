# LRF-IMU

This repository is the public LRF-IMU release boundary through Milestone 3B. It
contains the portable M3A data-preparation boundary and an evidence-labelled
VAE/checkpoint boundary. The scientific implementation and release decisions
remain under review where the evidence is incomplete.

## Paper identity

The paper record used by this skeleton is:

- **Title:** *A latent rectified flow approach to generate synthetic wearable
  data â€“ a LABDA solution*
- **Authors:** Amin Rezaei, Morten KjÃ¦rgaard, and Jasper Schipperijn
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

The base runtime dependencies are PyYAML>=6.0 and NumPy>=1.21.3. These lower
bound is an unpinned, minimum-safe runtime declaration for the supported Python
floor and APIs used by this package; it is not a historical environment claim.
Optional extras provide pytest (`test`), PyTorch (`training`), scikit-learn
(`evaluation`), SciPy (`analysis`), and release tools (`dev`).
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

## Milestone 3C: Rectified Flow integration

The public package now exposes the staged Flow implementation through a lazy, metadata-first boundary. It preserves the accepted M3B VAE behavior and uses the source equations:

```text
zt = (1-t) z0 + t z1
target = z1 - z0
model time = 1000 t
reverse Euler: z <- z - v dt
```

The paper profile is fixed at ten reverse-Euler steps. Historical subject-01 Flow checkpoints are supported only with their separately trained, matching 6CH or 3CH VAE checkpoint. The historical checkpoints validate as width 256, latent channels 48, four classes, and 89 U-Net state tensors.

The safe command boundaries are:

```text
python -m lrf_imu flow-smoke
python -m lrf_imu inspect-flow-checkpoint --checkpoint <6CH-flow-checkpoint>
python -m lrf_imu generate --config <config> --vae-checkpoint <vae> --flow-checkpoint <flow> --class-id 0 --count 1 --steps 10 --seed 42 --device cpu
python -m lrf_imu export-trajectories --flow-checkpoint <flow> --vae-checkpoint <vae> --subject 1 --activity 0 --base-seed 42 --device cpu
```

Generation and inspection report shapes, finite-value status, hashes, and schema metadata only. They do not print tensor values, write output implicitly, or copy checkpoints or participant artifacts into the release tree.

Website trajectories use a deliberately separate profile: 100 reverse-Euler steps, `record_every=2`, 51 states, native 160-sample windows, 40-sample linear overlap-add, ten seconds at 50 Hz, and seed `base + subject*1000 + activity*100`. Website output is never labeled as paper/TSTR output.

Milestone 3C validation passed Gates A-E and the website contract. The detailed metadata-only evidence is external at `<external-validation-root>/m3c_validation.json`; the committed parity contract is `contracts/flow_parity_report.json`.

`exact_paper_reproduction` remains `false`, and the 128-versus-256 width conflict remains explicitly unresolved. See `docs/FLOW_PARITY_REPORT.md` and `docs/KNOWN_DISCREPANCIES.md`.
## Milestone 3D: evaluation

The package exposes source-compatible RF/CNN evaluation for one fold or the
canonical 12-fold cohort. Inputs and outputs are always explicit:

```text
python -m lrf_imu evaluate --data-root <realdisp-root> --sensor six_channel --classifier rf --held-out-subject 1 --synthetic-cache <external-cache> --scenario trtr --scenario tstr
python -m lrf_imu evaluate-loso --data-root <realdisp-root> --sensor six_channel --classifier rf --synthetic-root <external-cache-root> --output-root <external-output> --write-results --resume
```

Fresh CPU evaluation completed all 12 RF folds for 6CH and separate 3CH
models, plus a corrected all-fold 6CH CNN run seeded once per fold before the
historical scenario order. Historical subject-01 RF metrics reproduce exactly
when the immutable historical cache is supplied. Fresh CPU generation is not
bitwise equivalent to historical CUDA generation, so affected TSTR folds are
reported as partial rather than exact.

TSTR/scarce-only requests execute TRTR internally for retention but return only
the requested scenarios. Synthetic caches require a validated adjacent
identity/checksum manifest. Write mode requires an explicit output root, and
fresh/resumed fold results share one aggregation schema. See
`docs/EVALUATION_PARITY_REPORT.md`.

## Milestone 3E: paper-relevant analyses

Install the analysis and evaluation extras for these commands:

```text
python -m pip install -e ".[training,evaluation,analysis]"
python -m lrf_imu evaluate-vae-only --data-root <realdisp-root> --vae-checkpoint <checkpoint> --held-out-subject 1 --dry-run
python -m lrf_imu analyze-sensitivity --input <fold-records.json>
python -m lrf_imu analyze-physical --input <physical-windows.npz> --units m_s2
python -m lrf_imu analyze-spectral --real <real-windows.npz> --synthetic <synthetic-windows.npz>
python -m lrf_imu analyze-privacy --input <fold-records.json> --threat-model <explicit-model>
```

Commands print metadata by default. Writing requires both `--write-results` and
an explicit `--output`; participant or synthetic arrays are never written by
these commands. The nine-setting sensitivity summary reproduced exactly. Other
analysis results are explicitly PARTIAL where raw attacks/folds were not rerun
or the historical checkpoint lineage is unavailable. See
`docs/ANALYSIS_PARITY_REPORT.md`.

## End-to-end core reproduction

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
