# Paper 3 HARTH–DayForge integration runbook

This runbook describes the frozen runtime sequence. It does not train models or execute DayForge.

For the final handoff evidence contract, use
[`dayforge_mapping.md`](dayforge_mapping.md). The mapping command accepts the
semantic DayForge root and the separate derived in-bed handoff root; both are
read-only inputs.

Set the package path for a checkout:

```bash
export PYTHONPATH=src
```

## Runtime sequence

1. Prepare HARTH data:

```bash
python -m lrf_imu prepare-harth-data \
  --data-root <HARTH_ROOT> \
  --composition harth_walking_speed \
  --held-out-subject harth:S006
```

2. Train the HARTH VAE (runtime phase only):

```bash
python -m lrf_imu train-harth-vae \
  --data-root <HARTH_ROOT> \
  --composition harth_walking_speed \
  --held-out-subject harth:S006 \
  --config configs/paper/harth_10class_160_40.yaml \
  --output-dir <VAE_OUTPUT>
```

3. Run VAE sanity evaluation:

```bash
python -m lrf_imu evaluate-harth-vae \
  --data-root <HARTH_ROOT> \
  --composition harth_walking_speed \
  --held-out-subject harth:S006 \
  --config configs/paper/harth_10class_160_40.yaml \
  --vae-checkpoint <VAE_CHECKPOINT> \
  --output-dir <VAE_REPORT>
```

4. Train the HARTH Flow (runtime phase only):

```bash
python -m lrf_imu train-harth-flow \
  --data-root <HARTH_ROOT> \
  --composition harth_walking_speed \
  --held-out-subject harth:S006 \
  --config configs/paper/harth_10class_160_40.yaml \
  --vae-checkpoint <VAE_CHECKPOINT> \
  --output-dir <FLOW_OUTPUT>
```

5. Run Flow sanity evaluation:

```bash
python -m lrf_imu evaluate-harth-flow \
  --data-root <HARTH_ROOT> \
  --composition harth_walking_speed \
  --held-out-subject harth:S006 \
  --config configs/paper/harth_10class_160_40.yaml \
  --vae-checkpoint <VAE_CHECKPOINT> \
  --flow-checkpoint <FLOW_CHECKPOINT> \
  --output-dir <FLOW_REPORT>
```

6. Map validated DayForge intervals without modifying DayForge:

```bash
python -m lrf_imu map-dayforge-physical-states \
  --dayforge-root <VALIDATED_DAYFORGE_ROOT> \
  --derived-root <IN_BED_HANDOFF_ROOT> \
  --config configs/paper/dayforge_harth_mapping.yaml \
  --output-dir <MAPPING_OUTPUT>
```

7. Inspect `mapping_summary.json` and `mapping_report.md` before synthesis.

8. Dry-run one person-day fusion:

```bash
python -m lrf_imu synthesize-dayforge \
  --dayforge-root <VALIDATED_DAYFORGE_ROOT> \
  --mapping-root <MAPPING_OUTPUT> \
  --output-dir <FUSION_OUTPUT> \
  --persona <PERSONA_ID> --date <YYYY-MM-DD> --dry-run
```

9. Generate one person-day using existing checkpoints and normalization metadata:

```bash
python -m lrf_imu synthesize-dayforge \
  --dayforge-root <VALIDATED_DAYFORGE_ROOT> \
  --mapping-root <MAPPING_OUTPUT> \
  --vae-checkpoint <VAE_CHECKPOINT> \
  --flow-checkpoint <FLOW_CHECKPOINT> \
  --normalization-metadata <NORMALIZATION_JSON> \
  --output-dir <FUSION_OUTPUT> \
  --persona <PERSONA_ID> --date <YYYY-MM-DD> --seed 42
```

10. Review `fusion_summary.json`, `unsupported_intervals.csv`, generated segment manifests, and `fusion_validation_report.md`; only then scale to a selected cohort.

The commands above are interfaces only. No real cohort result, checkpoint result, or coverage number is asserted by this repository freeze.
