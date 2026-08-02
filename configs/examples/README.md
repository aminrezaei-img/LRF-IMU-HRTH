# Configuration examples

The paper configurations are portable YAML inputs for the initial release
framework. Relative roots (`data`, `outputs`, `checkpoints`, and `results`)
are resolved against the caller's current directory unless `base_dir` is
provided. No configuration contains a personal or machine-specific path.

```python
from pathlib import Path

from lrf_imu import load_config

config = load_config(
    "configs/paper/six_channel_160_40.yaml",
    base_dir=Path("."),
    data_root=Path("/path/to/realdisp"),
    output_root=Path("/path/to/outputs"),
    checkpoint_root=Path("/path/to/checkpoints"),
    results_root=Path("/path/to/results"),
    subject=16,
    fold=16,
    device="cpu",
    seed=7,
)

print(config.paths.data_root)
print(config.paths.fold(config.split.held_out_subject).checkpoint_root)
```

The same controls are available after loading with
`config.with_overrides(...)`. Dotted keys are supported for additional
CLI-ready settings, for example:

```python
config = config.with_overrides({"sampling.steps": 20, "runtime.device": "cpu"})
```

The three supplied variants have distinct purposes:

- `paper/six_channel_160_40.yaml` declares six right-thigh accelerometer and
  gyroscope channels.
- `paper/accelerometer_only_160_40.yaml` declares a separately trained
  accelerometer-only model. It must not reuse a six-channel checkpoint with
  channels dropped at inference time.
- `paper/sensitivity_grid.yaml` declares the 3x3 six-channel window/hop grid;
  each grid point requires its own training and fold-specific checkpoints.

Every paper configuration contains the four evidence tiers:

- `manuscript_reported`
- `observed_wrapper`
- `observed_checkpoint_metadata`
- `release_default`

The unresolved VAE schedule and flow base-width discrepancies are retained in
`evidence.conflicts`. The selected values are labeled compatibility defaults;
the configs intentionally do not claim exact reproduction of the paper.

This milestone adds configuration and path primitives only. It does not move
the research preprocessing, model, sampling, or evaluation implementations.

## Packaged runtime resources and split fractions

The root configs/paper/ files are human-facing evidence and reviewable
configuration examples. The matching files under
src/lrf_imu/resources/configs/paper/ are intentional package data for wheel
portability, not replacement configurations. tests/test_packaging.py compares
their normalized bytes and parses both sides so configuration drift fails in
the test suite.

Each paper profile names both validation protocols:

- split.vae_subject_validation_fraction: 0.15 is the VAE subject-level value.
- split.classifier_window_validation_fraction: 0.20 is the classifier/window
  value.
- split.validation_fraction: 0.20 remains as a legacy classifier/window
  alias for compatibility. It is not silently relabeled as the VAE value, and
  the parser rejects a mismatch with the named classifier value.