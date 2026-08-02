# Public Compatibility Profile

This profile is a transparent boundary for public contract tests. It captures
the behavior that can be named without claiming exact source equivalence.

## Data and splits

The public production shape is 50 Hz, 160-sample windows, stride 40, and six
right-thigh channels. The synthetic compact probe uses four-sample windows and
stride two. Safe subject-wise fixture counts are 16 training, 7 validation,
and 8 held-out windows. VAE validation is subject-level at 0.15; CNN validation
is window-level at 0.20.

## VAE

The compatibility profile uses six input channels, a latent shape of
[batch, 48, 40], batch size 256, maximum 1000 epochs, minimum 200 epochs,
patience 100, with L2 weight 0.5 and L1 weight 0.1. The later wrapper beta schedule is 0.08 to 0.04 with
decay 0.995. The older schedule and loss weights are retained as unresolved
evidence rather than silently discarded.

## Rectified flow

The compatibility profile uses latent width 256, channel multipliers 1, 2,
and 4, configured batch size 128 with automatic selection enabled, safety
fraction 0.75, reverse Euler sampling, paper step count 10, interval 1 to 0,
and seed 42. The source also exposes an observed subject-01 effective batch
of 512 and a manuscript width of 128; both remain source conflicts where
applicable.

## Evaluation and classifiers

The random-forest probe is 100 trees, seed 42, and one job. The CNN probe uses
channels 32, 64, 128; kernel size 5 with padding 2; pooling after the second
block; fully connected widths 256, 128, 4; dropout 0.3; 80 epochs;
patience 10; batch 64; learning rate 0.001; weight decay 0.0001; and
validation fraction 0.20 without class weights.

Metrics use the encoded evaluation order 0, 1, 2, 3, macro-F1, zero-division
handling, and sample standard deviation for fold dispersion. The raw-code mapping
is 1 → 0 walking, 3 → 1 running, 4 → 2 jump_up, and 33 → 3 cycling. Exact paper
reproduction and full experimental completion are both false.

Website visualization uses 100 Euler steps over 500 native samples/windows; this is separate from the paper sampler's 10 steps.
