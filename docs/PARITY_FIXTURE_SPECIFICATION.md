# Deterministic Parity Fixture Specification

The fixture set is a small, synthetic, license-safe test boundary. It is
designed to test contracts and compatibility behavior without embedding
participant-derived signals, checkpoints, trained weights, or result payloads.

## Recipe

- Subjects are the synthetic identifiers 01, 02, 03, and 05.
- Each subject has 24 rows, 120 tab-separated columns, and no header.
- Columns 80 through 85 are the synthetic right-thigh six-channel block.
- Column 119 is the activity label. Allowed labels are 1, 3, 4, and 33.
- Nonselected columns use a deterministic subject/column formula. Selected
  channels use a deterministic subject/row/channel formula.
- Subject 01 contains a filtered gap pattern 1, 1, 99, 99, 1, 1 to exercise
  filter-before-runs bridging. The other subjects contain contiguous allowed
  runs.
- Compact windows use four samples with stride two. Production compatibility
  metadata remains 160 samples with stride 40 at 50 Hz.
- The safe subject split is 16/7/8: subject 01 yields seven compact windows and
  subjects 02, 03, and 05 yield eight each.
- The VAE subject validation fraction is 0.15. The CNN window validation
  fraction is 0.20. These are separate split contracts.

## Preprocessing invariants

Filtering occurs before run detection and gap bridging. A short two-row
disallowed gap between allowed rows remains within the bridged run. Windows
never cross activity boundaries. Standardization uses training-only means and
population standard deviation (ddof 0) over axes 0 and 2, with a floor of
1e-8. Inverse transformation is required to round-trip the synthetic probe.

The three-channel contract is a distinct selection of columns 80, 81, and 82.
It must not be implemented as a silent drop from a six-channel tensor.

## Duplicate and evaluation probes

Duplicate auditing covers train/validation and train/held-out-test comparisons,
with a legacy train-validation-only alias retained as a compatibility warning.
The fixture metrics use encoded labels 0, 1, 2, 3 with the explicit raw-code mapping 1 to 0, 3 to 1, 4 to 2, and 33 to 3. They include y_true/y_pred, macro-F1 with zero division handled explicitly, confusion counts, sample standard deviation (ddof 1), and population standard deviation (ddof 0).

## Sampling and visualization

The paper sampling probe is reverse Euler with 10 steps and seed 42. The website
visualization probe remains separate: 100 stored steps, 500 native samples,
10-second duration, 50 Hz, and overlap-add reconstruction. The website's
trajectory seed is derived from the documented seed formula. Website 100-step
visualization must not be substituted for the paper's 10-step sampling.

All fixture files are UTF-8 text, deterministic, human-auditable, and covered by
SHA-256 checksums in the manifest. The checksum file excludes itself.
