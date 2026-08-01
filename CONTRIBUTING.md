# Contributing

This repository is at a deliberately narrow Milestone 1 boundary. Changes
should keep it auditable, portable, and honest about what has not yet been
migrated.

## Before opening a change

- Keep raw REALDISP data, preprocessed windows, checkpoints, caches, results,
  logs, manuscript history, and secrets outside the repository.
- Do not edit the seven byte-preserved files listed in
  `docs/ORIGINAL_TO_PUBLIC_MAPPING.md` unless a later milestone explicitly
  changes the archival decision and rechecks their hashes.
- Use configurable roots and neutral placeholders; never commit workstation
  paths or personal identifiers.
- Do not add scientific model, preprocessing, generation, or evaluation code
  under the Milestone 1 scope.
- Avoid new dependencies unless the release scope and license review are
  updated at the same time.

## Suggested checks

From the repository root, run the focused scanner and the configuration smoke
checks described in `REPRODUCIBILITY.md`. Keep temporary test directories
outside the repository and remove artifacts created by the checks.

## Pull requests

Describe the scope, evidence used, files changed, and known limitations. A
change should not imply exact paper reproduction unless the unresolved audit
discrepancies have been resolved with authoritative evidence. Use the pull
request template for the safety and verification checklist.
