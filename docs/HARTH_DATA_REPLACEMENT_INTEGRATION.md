# HARTH-family replacement for the REALDISP data path

## Purpose

This document is the implementation handoff for replacing the current
REALDISP-specific preprocessing path in LRF-IMU with a HARTH-family input
pipeline. It is written so another agent can fetch and inspect the required
source material without guessing what the VAE integration needs.

The replacement must preserve the downstream LRF-IMU contract while changing
the dataset boundary. It must not copy raw participant data into this
repository.

## Source repository inspected

Repository:

```text
https://github.com/aminrezaei-img/harth-ml-experiments
```

Inspected revision at the time of writing:

```text
dad2cfb (grafted, origin/main) Added Readme information about walking speed
```

Relevant source directories:

```text
harth/                  HARTH v2.0; 31 subjects, S006.csv ... S038.csv
har70plus/              HAR70+; 18 older-adult subjects, 501.csv ... 518.csv
adult_walking_speed/    24 walking-speed subjects, 01.csv ... 24.csv
experiments/            Baseline ML experiments and configurations
```

The repository CSV files use the common schema:

```text
timestamp,back_x,back_y,back_z,thigh_x,thigh_y,thigh_z,label
```

The sample files inspected contain timestamps at approximately 50 Hz and
three accelerometer axes on the back and thigh. There is no gyroscope stream
in this schema.

## Target activity taxonomy

The replacement model must use exactly these ten encoded classes, in this
order unless a later decision explicitly changes it:

| Encoded ID | Supercategory | Canonical name | Required source labels |
|---:|---|---|---|
| 0 | locomotion | walking_slow | adult_walking_speed: 101 |
| 1 | locomotion | walking_moderate | adult_walking_speed: 102 |
| 2 | locomotion | walking_brisk | adult_walking_speed: 103 |
| 3 | locomotion | running | adult_walking_speed: 2; HARTH: 2 |
| 4 | locomotion | stair_climbing | HARTH: 4 and/or 5, subject to the decision below |
| 5 | cycling | cycling_seated | HARTH: 13 |
| 6 | cycling | cycling_standing | HARTH: 14 |
| 7 | posture_stationary | sitting | HARTH: 7; HAR70+: 7 |
| 8 | posture_stationary | standing | HARTH: 6; HAR70+: 6 |
| 9 | posture_stationary | lying | HARTH: 8; HAR70+: 8 |

### Important source-label facts

HARTH documents these relevant labels:

```text
1   walking
2   running
4   stairs ascending
5   stairs descending
6   standing
7   sitting
8   lying
13  cycling seated
14  cycling standing
```

HAR70+ documents walking-related and posture/stair labels, but does not
provide the cycling classes required by the target taxonomy:

```text
1 walking
4 stairs ascending
5 stairs descending
6 standing
7 sitting
8 lying
```

The walking-speed dataset provides the speed-specific classes required to
create `walking_slow`, `walking_moderate`, and `walking_brisk`:

```text
101 slow-walking
102 moderate-walking
103 brisk-walking
2   running
```

The source repository's `adult_walking_speed` data is therefore required for
the three walking-speed classes. HARTH is required for cycling and provides
the main source for the posture/stationary and stair classes. HAR70+ is an
optional additional population/domain, not a complete replacement by itself.

## Decisions that must be made before implementation

### 1. Stair direction policy

The requested taxonomy says `stair climbing`, while HARTH distinguishes
ascending and descending. The recommended first integration policy is:

```text
HARTH label 4 (stairs ascending) → stair_climbing
HARTH label 5 (stairs descending) → stair_climbing
HAR70+ label 4 and 5             → stair_climbing, if HAR70+ is included
```

The adapter must preserve the original label and direction in provenance
metadata even if both are mapped to one model class. If the scientific intent
is ascending-only, change the mapping to label 4 only before implementation;
do not silently discard label 5.

### 2. Dataset composition policy

To produce all ten requested classes, the recommended initial dataset is:

```text
HARTH + adult_walking_speed
```

HAR70+ should be enabled through an explicit option rather than silently
merged, because it changes the population/domain and does not add cycling.
The implementation should support a named composition such as:

```text
harth_walking_speed
harth_walking_speed_har70plus
```

Do not mix datasets by default without recording dataset membership per
subject and checking for subject-ID collisions.

### 3. Generic HARTH walking

HARTH label 1 is generic walking and has no slow/moderate/brisk speed label.
It must not be silently mapped to one of the three speed-specific classes.
For the initial ten-class taxonomy, either:

- exclude HARTH label 1; or
- add a separate `walking_generic` class and explicitly expand the taxonomy.

The recommended choice is to exclude it initially, because inventing a speed
class would corrupt the requested semantics.

### 4. Cycling inactive labels

HARTH also contains labels 130 and 140 (`cycling seated/standing, inactive`).
They should not be silently combined with labels 13 and 14. The recommended
initial policy is to exclude 130 and 140, then add them only through an
explicit policy such as:

```text
130 → cycling_seated
140 → cycling_standing
```

if the scientific question intentionally includes inactive cycling.

## Required adapter contract

The other agent should fetch and return the following information and/or
implementation pieces from the HARTH-family repository:

1. **Exact file schema** for all three relevant dataset folders, including:
   - column names and order;
   - timestamp parsing format;
   - label column type and observed label values;
   - sample-rate verification method;
   - missing-value and duplicate-timestamp behavior.
2. **Subject discovery rules** for:
   - `harth/S*.csv`;
   - `adult_walking_speed/*.csv`;
   - optional `har70plus/*.csv`.
3. **Dataset-specific label tables** with original integer labels, names, and
   the proposed ten-class mapping above.
4. **The exact signal channel selection** for the LRF-IMU replacement. The
   recommended default is right-thigh acceleration only:

   ```text
   [thigh_x, thigh_y, thigh_z]
   ```

   This is a 3-channel input and must use a separately trained VAE/Flow model;
   it cannot reuse a six-channel REALDISP checkpoint.
5. **A dataset-neutral reader** returning, per subject:

   ```python
   signals:    float32 ndarray with shape [time, 3]
   raw_labels: int32/int64 ndarray with shape [time]
   timestamps: optional datetime array with shape [time]
   metadata:   dataset name, subject ID, source file, sample rate, and label map
   ```
6. **A deterministic label normalization layer** that maps original labels to
   the ten encoded IDs and rejects unsupported labels unless they are
   explicitly configured as excluded.
7. **A sample-rate policy**:
   - verify timestamps rather than assuming 50 Hz;
   - define what happens when the rate is not exactly 50 Hz;
   - resample before windowing if necessary, using a documented method;
   - record the original and effective rates in metadata.
8. **A subject identity namespace** such as `(dataset_name, subject_id)`.
   Numeric IDs alone are unsafe because HARTH, HAR70+, and walking-speed files
   overlap or use different conventions.
9. **A provenance-preserving composition manifest** listing every included
   subject and source dataset, without storing raw values.
10. **Synthetic tests or small non-participant fixtures** covering every target
    class, label remapping, subject discovery, timestamp/rate validation,
    boundary handling, and mixed-dataset subject identity.

The agent should not fetch or commit the full CSV datasets into LRF-IMU.
Source code, schemas, tiny synthetic fixtures, and metadata contracts are
acceptable; participant data must remain external.

## LRF-IMU integration design

The current REALDISP path is coupled primarily in:

```text
src/lrf_imu/data/realdisp.py
src/lrf_imu/data/schema.py
src/lrf_imu/data/pipeline.py
src/lrf_imu/data/activities.py
src/lrf_imu/data/windowing.py
src/lrf_imu/data/splits.py
src/lrf_imu/data/normalization.py
```

The recommended replacement is an adapter rather than another dataset-specific
copy of `realdisp.py`:

```text
src/lrf_imu/data/datasets.py       # neutral subject-record protocol
src/lrf_imu/data/harth.py          # HARTH/HAR70+/walking-speed reader
src/lrf_imu/data/label_maps.py     # source → ten-class mapping
src/lrf_imu/data/sampling.py       # timestamp/rate validation/resampling
src/lrf_imu/data/pipeline.py       # dataset composition dispatch
```

The existing downstream components should remain reusable where their
contracts are still valid:

```text
make_windows(...)          # activity-bounded [N,C,T] windows
split_vae_windows(...)     # subject-safe VAE split
ChannelStandardizer(...)   # training-only per-channel z-score
train_vae(...)             # VAE optimization
LatentVAE1D(...)           # model geometry
```

The new adapter must produce time-major signals. `make_windows()` converts
them into the VAE's channel-major representation:

```text
adapter output: [time, 3]
VAE input:      [N, 3, T]
```

## Recommended replacement preprocessing contract

For the first implementation, use this explicit contract:

```yaml
dataset:
  composition: harth_walking_speed
  source_repository: https://github.com/aminrezaei-img/harth-ml-experiments
  sample_rate_hz: 50
  verify_timestamps: true
  resampling: reject_or_explicitly_resample
  channel_set: thigh_accelerometer
  channels: [thigh_x, thigh_y, thigh_z]
  channel_count: 3

labels:
  encoded_class_count: 10
  mapping:
    0: walking_slow
    1: walking_moderate
    2: walking_brisk
    3: running
    4: stair_climbing
    5: cycling_seated
    6: cycling_standing
    7: sitting
    8: standing
    9: lying
  exclude:
    - HARTH label 1 generic walking
    - HARTH labels 3 shuffling
    - HARTH labels 130 and 140 inactive cycling
    - any transition/unknown/unmapped labels

window:
  samples: 160
  hop: 40
  complete_windows_only: true
  padding: false
  boundary_policy: split_on_excluded_or_unmapped_rows

split:
  protocol: leave_one_subject_out
  subject_key: [dataset, subject_id]
  validation_unit: subject
  vae_subject_validation_fraction: 0.15
  seed: 42

normalization:
  method: per_channel_zscore
  fit_on: training_subjects_only
  ddof: 0
  minimum_standard_deviation: 1e-8
```

The window length and hop can remain 160/40 because all relevant source data
is documented at 50 Hz. This produces 3.2-second windows with 0.8-second hop,
matching the current VAE geometry. The new ten-class model must use:

```text
LatentVAE1D(in_ch=3, z_ch=48, down_levels=2)
```

and a separately trained three-channel Rectified Flow model.

## Splitting and leakage requirements

The composition must be split by namespaced subject, never by random windows.
All windows from one `(dataset, subject_id)` must remain in exactly one of:

```text
VAE train subjects
VAE validation subjects
held-out test subject
```

For mixed datasets, add a composition/domain report containing at least:

- number of subjects per source dataset;
- number of subjects per split and source dataset;
- per-class window counts per source dataset;
- whether every class occurs in the training partition;
- whether any subject identity appears in more than one split;
- timestamp/rate validation results;
- original-to-encoded label counts;
- excluded-label counts.

The existing duplicate-window audit should continue to check train/validation,
train/test, and validation/test pairs. It should operate on standardized
windows and remain independent of source dataset.

## Expected changes in LRF-IMU after the adapter arrives

The integration agent should then:

1. Add a dataset selector to configuration and CLI, without breaking the
   existing REALDISP compatibility path.
2. Add a HARTH-family configuration with `input_channels: 3` and
   `num_classes: 10`.
3. Replace the hard-coded REALDISP activity mapping in the selected pipeline
   with the namespaced ten-class mapping.
4. Generalize activity metadata so class names and count come from the active
   dataset configuration rather than fixed four-class constants.
5. Generalize `make_windows()` so allowed labels and encoding are supplied by
   the active adapter, while retaining the current REALDISP compatibility
   default.
6. Generalize `split_vae_windows()` to accept namespaced subject keys or a
   stable canonical subject token while retaining backward compatibility for
   REALDISP integer IDs.
7. Ensure the VAE and Flow class-conditioning dimensions are 10 for the new
   model; existing four-class checkpoints must be rejected as incompatible.
8. Keep normalization training-only and record its statistics/provenance.
9. Add parity-style synthetic tests for the new adapter and end-to-end compact
   preparation tests producing `[N, 3, 160]` windows and labels in `[0, 9]`.
10. Update documentation and generated metadata so the active dataset,
    composition, label mapping, source revision, subject namespace, and
    effective sampling rate are explicit.

## Acceptance criteria for the fetched handoff

The handoff is complete only if the other agent provides:

- exact source revision and paths inspected;
- a verified schema for each included CSV family;
- an unambiguous ten-class label map;
- a written stair-direction and inactive-cycling decision;
- a written HARTH generic-walking decision;
- a sample-rate/timestamp policy;
- a namespaced subject split policy;
- a 3-channel `[time, channels]` adapter contract;
- synthetic fixtures/tests for all ten labels and mixed-source handling;
- no raw participant CSVs committed to LRF-IMU;
- enough metadata for reproducible VAE preprocessing and later Flow/evaluation
  integration.
