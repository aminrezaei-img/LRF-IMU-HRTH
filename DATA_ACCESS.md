# Data access

## Production HARTH-family snapshot

The production ten-class model used two folders from
`ntnu-ai-lab/harth-ml-experiments`:

- `harth/`;
- `adult_walking_speed/`.

The frozen source commit is:

```text
dad2cfbe89a26f72f19770419469ac037de200df
```

GitHub Release `paper3-harth-production-v1` contains a ZIP archive of exactly
the 55 tracked CSV files used by the pipeline, plus the upstream `README.md`
and MIT `LICENSE`. The archive SHA-256 is recorded in the packaged production
artifact manifest. `har70plus/` and unrelated experiment outputs are not part
of the production composition.

Acquire and verify the archive with:

```bash
python -m lrf_imu fetch-production-artifacts \
  --output-dir production-artifacts
```

After extraction, point `--data-root` at the directory containing both source
folders:

```text
<data-root>/
  LICENSE
  README.md
  harth/
    S006.csv
    ...
  adult_walking_speed/
    01.csv
    ...
```

The loader uses direct-child discovery inside each declared folder. It does
not recursively search a wider workspace and does not silently include other
datasets.

## Production preparation contract

- composition: `harth_walking_speed`;
- sensor: right-thigh accelerometer, three axes;
- sampling rate: 50 Hz;
- window length/hop: 160/40 samples;
- split key: namespaced `dataset:subject_id`;
- held-out subject: `harth:S006`;
- validation fraction: 0.15 at subject level;
- seed: 42;
- normalisation: per-channel z-score fitted on training subjects only;
- duplicate audit: exact window-byte checks within and across all partitions.

The accepted snapshot produced 159,575 training, 18,533 validation, and 8,497
held-out windows. A different count is a reason to trace the data/configuration
identity rather than force the expected values.

## Licensing and citation

The release archive preserves the upstream licence and README verbatim.
Anyone redistributing or using the snapshot remains responsible for following
the upstream attribution and dataset citation requirements. The data licence
does not supply a licence for this software repository.

## Legacy REALDISP path

The repository retains the earlier four-class REALDISP implementation and its
parity evidence. REALDISP is not bundled in the HARTH production release.
Users of that legacy path must obtain the dataset from the
[UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/305/realdisp%2Bactivity%2Brecognition%2Bdataset)
under its current terms.

The two data lineages must not be mixed: the production HARTH checkpoint pair
is three-channel and ten-class, while historical REALDISP checkpoints can be
six-channel and four-class.
