# Data access

## Dataset boundary

The audited paper task uses the REALDISP benchmark. The dataset is external to
this repository and is not redistributed here. Users must obtain it directly
from the dataset custodian under the terms that apply to their intended use.
This repository does not grant access to REALDISP or decide its license terms.

The audited subset is:

- ideal-placement recordings from one right-thigh IMU;
- 50 Hz sampling with six channels: `ax`, `ay`, `az`, `gx`, `gy`, `gz`;
- original activity codes 1, 3, 4, and 33, mapped to walking, running, jump_up,
  and cycling;
- subjects 1, 2, 3, 5, 8, 9, 10, 11, 12, 13, 14, and 16;
- 12 leave-one-subject-out folds;
- 160-sample windows with a 40-sample hop; and
- per-channel z-score standardization fitted on training participants within
  each fold.

The audited source notes that this paper-specific four-class preprocessing
filters target labels before constructing contiguous runs. That choice is
useful for reproducing the paper task but should be reviewed before a general
REALDISP parser is designed.

## Official primary source and citation

Official primary source: [REALDISP Activity Recognition Dataset at the UCI
Machine Learning Repository](https://archive.ics.uci.edu/dataset/305/realdisp%2Bactivity%2Brecognition%2Bdataset)

Formal citation:

> Banos, O., Toth, M., & Amft, O. (2012). REALDISP Activity Recognition Dataset [Dataset]. UCI Machine Learning Repository. DOI 10.24432/C5GP6D.

Users remain responsible for reviewing the current dataset terms and citation
requirements at the official record before downloading or using the dataset.
This repository neither auto-downloads nor redistributes REALDISP.

## Local setup

Keep acquired data outside Git and outside generated-output directories. Pass a
user-selected `data_root` to a configuration loader; do not encode a personal
or machine-specific path in a config, script, notebook, or documentation file.
Do not commit raw logs, preprocessed windows, or participant-derived synthetic
data. Check the current dataset terms before sharing any derived artifact.

## Milestone 3A preparation contract

The public loader accepts only explicit direct-child subject*_ideal.log files.
Each row must have exactly 120 numeric tab-separated columns. The six-channel
path selects right-thigh columns 80..85; the explicit reconstructed 3CH path selects
80..82. Column 119 is retained as a raw activity code and is mapped only through
the four-class vocabulary 1 -> 0, 3 -> 1, 4 -> 2, and 33 -> 3.

The default public compatibility mode is filter-before-runs, preserving the
historical paper-task behavior for short excluded-label gaps. Strict original
contiguity is an explicit opt-in. Complete 160/40 windows do not cross activity
boundaries and are never padded. The VAE subject-level validation fraction is
0.15; the separate CNN window-level fraction remains 0.20.

Normalization is fitted on the training partition only, using population standard
deviation (ddof=0) with a 1e-8 floor. Duplicate checks use SHA-1 over canonical
exact-window bytes and the public default checks train/validation, train/test, and
validation/test boundaries; a historical train/validation-only adapter remains
explicitly labeled as compatibility behavior.

REALDISP and all participant-derived arrays remain outside this repository. The
preparation pipeline returns arrays in memory but serializes only JSON-safe metadata
with raw values, labels, signals, and windows excluded.