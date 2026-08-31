# Stitching and fusion

Module C already implements exact-duration, segmented DayForge-to-IMU
synthesis. Packaging documents its existing behavior; it does not introduce a
second generator.

## Duration contract

For an eligible interval:

```text
N = round(duration_seconds × 50)
```

Intervals use half-open `[start,end)` semantics. The generated signal contains
exactly `N` samples with three channels. The final segment is cropped to the
target length when necessary.

## Window policy

- Short intervals generate a window and crop it to the target sample count.
- One-window intervals use one generated window.
- Long intervals use multiple independently generated windows.
- A repeated single synthetic tile is never used to fill a long interval.
- Adjacent windows are joined with the existing linear crossfade overlap.
- Boundary jumps and finite values are recorded in the stitch audit.

Each window has a deterministic seed derived from the global seed and interval
identity. Segment manifests record the source interval, class, target samples,
window seeds, overlap, stitch method, checkpoint paths, and mapping
provenance.

## Failure semantics

Intentional unavailability—such as unsupported transport, ambiguous posture,
or non-realized mobility—is recorded separately from an actual IMU generation
failure. Unsupported intervals do not produce arrays. Generation failures are
listed in `generation_failures.json` and do not get silently reclassified as
unavailable.

Use the canonical command for a bounded person-day:

```bash
python -m lrf_imu synthesize-dayforge \
  --dayforge-root <validated-dayforge-root> \
  --mapping-root <mapping-output> \
  --vae-checkpoint <vae-checkpoint> \
  --flow-checkpoint <flow-checkpoint> \
  --normalization-metadata <normalization-json> \
  --output-dir <fusion-output> \
  --persona <persona-id> --date <YYYY-MM-DD> --seed 42
```

Do not use this documentation as authorization to synthesize an entire
cohort. Select the intended person-days explicitly and preserve the resulting
audit files outside normal Git history.
