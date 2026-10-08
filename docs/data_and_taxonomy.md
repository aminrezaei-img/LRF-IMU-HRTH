# Data and taxonomy

## Paper 3 source cohorts

The default Paper 3 composition is `harth_walking_speed`:

- HARTH recordings;
- Adult Walking Speed recordings; and
- no HAR70+ recordings in the default composition.

The source data is external and must be supplied with `--data-root`. The
production contract uses a namespaced subject key, leave-one-subject-out
held-out policy, three thigh accelerometer channels, 50 Hz sampling, 160-sample
windows, and a 40-sample hop.

The validated production partition used 55 namespaced subjects: 46 training,
8 validation, and held-out `harth:S006`. It produced 159,575 training,
18,533 validation, and 8,497 held-out windows. These are recorded validation
values, not defaults to force on another data snapshot.

## Fixed ten-class taxonomy

| ID | Name | Meaning in this release |
| ---: | --- | --- |
| 0 | `walking_slow` | Route-speed-derived slow walking |
| 1 | `walking_moderate` | Route-speed-derived moderate walking |
| 2 | `walking_brisk` | Route-speed-derived brisk walking |
| 3 | `running` | Explicit running evidence |
| 4 | `stair_climbing` | Explicit stair evidence |
| 5 | `cycling_seated` | Existing generic bicycle-travel policy |
| 6 | `cycling_standing` | Explicit source class where available |
| 7 | `sitting` | Explicit or approved sitting evidence |
| 8 | `standing` | Explicit or approved standing evidence |
| 9 | `lying` | Explicit lying or conservative in-bed opportunity |

The taxonomy is fixed. There is no `sleep` class.

## Preprocessing exclusions

The adapter preserves source-specific label policies and constructs windows
only from supported, physically contiguous runs. Unsupported labels and
activity gaps do not become windows spanning incompatible states. Class
frequency is observed rather than rebalanced; the release does not introduce
weighted sampling.

Normalization is per-channel z-score normalization fit on training subjects
only. The persisted geometry is `[1, 3, 1]`; validation and held-out data do
not contribute to the fitted statistics.

See [data access](../DATA_ACCESS.md) for the separate REALDISP path and
[training](training.md) for the Paper 3 command.
