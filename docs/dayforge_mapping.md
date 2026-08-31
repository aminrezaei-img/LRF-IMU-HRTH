# DayForge mapping

Module B is a read-only bridge from DayForge resolved intervals to the fixed
HARTH physical-state taxonomy. DayForge semantic activity is evidence, not
HARTH ground truth.

## Evidence hierarchy

```text
strong realized mobility or explicit physical evidence
                         ↓
              physical_state_hint
                         ↓
       non-conflicting derived in-bed opportunity
                         ↓
                      unavailable
```

The mapping CLI can consume the semantic DayForge root and the separate
`in_bed_or_lying_opportunity` handoff root:

```bash
python -m lrf_imu map-dayforge-physical-states \
  --dayforge-root <validated-dayforge-root> \
  --derived-root <in-bed-handoff-root> \
  --config configs/paper/dayforge_harth_mapping.yaml \
  --output-dir <mapping-output>
```

The output includes `physical_state_mapping.csv`, `mapping_summary.json`, and
`mapping_report.md`. The summary reports baseline, hint-only, and combined
coverage so any change in coverage remains visible.

## Guardrails

- Realized walking uses route speed; a walking hint alone never selects slow,
  moderate, or brisk walking.
- Realized bicycle travel follows the existing cycling policy; a cycling hint
  alone never infers `cycling_standing`.
- Passive transport remains `UNSUPPORTED_PASSIVE_TRANSPORT`, even when a
  passenger may be seated.
- Failed or non-realized movement cannot fabricate locomotion from a hint.
- Generic work, childcare, mixed, and unknown activities do not receive
  arbitrary posture fallbacks.
- Direct sitting, standing, lying, running, and stair hints are mapped only
  when stronger evidence does not contradict them.
- Derived in-bed opportunity can support `lying` only when stronger evidence is
  absent or compatible. It is not physiological sleep and does not create a
  sleep class.

## Provenance

Each mapped record retains source activity, physical hint and its source,
derived evidence IDs and source files, the final class, mapping rule, mapping
source, and conflict indicator. The relevant source labels are
`realized_mobility`, `existing_explicit_semantic_evidence`,
`physical_state_hint`, and `derived_in_bed_opportunity`.

Unavailable reason categories used by the implementation include:

```text
AMBIGUOUS_PHYSICAL_STATE
CYCLING_POSTURE_UNRESOLVED
MIXED_OR_UNSUPPORTED_ACTIVITY
NONREALIZED_MOBILITY
UNSUPPORTED_PASSIVE_TRANSPORT
WALK_SPEED_UNAVAILABLE
```

The derived root and DayForge inputs are never written by the mapping code.
