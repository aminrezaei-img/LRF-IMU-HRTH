# Training

## Environment

Use an existing Python environment with the required dependencies. The
validated production run used Conda `py311`, Python 3.11.11, PyTorch 2.5.1,
CUDA 12.1, and an NVIDIA RTX 4070 Laptop GPU. Training can use another
compatible CUDA environment, but the runtime and checkpoint provenance should
be recorded.

Install the package with:

```bash
python -m pip install -e ".[training,evaluation,analysis,test]"
```

## Paper 3 configuration

Use `configs/paper/harth_10class_160_40.yaml` without changing the first
production baseline. Important values are:

- input: three channels, 160 samples, 50 Hz;
- latent: 48 channels and 40 time steps;
- VAE: batch size 256, learning rate 0.001, maximum 1,000 epochs;
- VAE loss: L2 weight 0.5, L1 weight 0.1, beta 0.08 with minimum 0.04 and
  decay 0.995;
- augmentation: enabled with jitter 0.008, scale 0.04, and time mask 0.05;
- Flow: 300 epochs, learning rate 0.0005, batch size 128, AdamW, and ten
  classes.

## Commands

```bash
python -m lrf_imu prepare-harth-data \
  --data-root <harth-family-root> \
  --composition harth_walking_speed \
  --held-out-subject harth:S006

python -m lrf_imu train-harth-vae \
  --data-root <harth-family-root> \
  --composition harth_walking_speed \
  --held-out-subject harth:S006 \
  --config configs/paper/harth_10class_160_40.yaml \
  --output-dir <vae-output> --seed 42

python -m lrf_imu train-harth-flow \
  --data-root <harth-family-root> \
  --composition harth_walking_speed \
  --held-out-subject harth:S006 \
  --config configs/paper/harth_10class_160_40.yaml \
  --vae-checkpoint <vae-checkpoint> \
  --output-dir <flow-output> --seed 42
```

Module A evaluation commands are documented in [validation](validation.md).

## Fixed-schedule note

The Flow configuration contains `early_stop_patience`, but the current frozen
`train_flow` implementation executes its configured epoch schedule. The
validated production baseline therefore ran 300 epochs. This limitation is
documented rather than silently corrected during packaging.

Do not retrain or alter the production checkpoints as part of release
packaging. See [checkpoints](checkpoints.md).
