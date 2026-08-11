# Milestone 6 handoff

## Result

The public documentation now leads with a first-time researcher workflow rather
than the internal milestone chronology. Scientific code, configurations, tests,
CI, and accepted parity contracts were not changed.

## Researcher-facing path

- `README.md`: scope, status, install, data preparation, model smokes,
  checkpoint inspection, generation, one-fold evaluation, and core LOSO entry.
- `DATA_ACCESS.md`: external REALDISP responsibility and expected ideal-log
  layout.
- `REPRODUCIBILITY.md`: evidence levels, executable workflow, and limits.
- `MODEL_CARD.md`: model identity, intended use, evaluation status, and
  prohibited overclaims.
- `docs/RESULTS_REPRODUCTION.md`: authoritative headline and analysis values,
  including every partial/blocked distinction.
- `docs/RELEASE_CHECKLIST.md`: technical completion versus human licensing and
  release decisions.
- `docs/KNOWN_DISCREPANCIES.md`: unresolved historical evidence.
- `llms.txt`: concise machine-oriented repository map.

## Preserved scientific boundaries

Historical checkpoints are not distributed. REALDISP is not redistributed.
The 3CH parser is an explicit public reconstruction paired with separately
trained 3CH models, not proven historical parser lineage. Paper sampling is ten
reverse-Euler steps; website trajectories use a distinct 100-step visualization
profile. Privacy observations are threat-model-specific and do not establish
anonymization. `exact_paper_reproduction=false` remains unchanged.

## Release posture

The documentation preserves the Milestone 5 verdict:

**CONDITIONAL GO - CODE TECHNICALLY RELEASE READY, LICENSING DECISION REQUIRED.**

No licence grant, remote, tag, push, upload, or publication action is part of
this milestone.
