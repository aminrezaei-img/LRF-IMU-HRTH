# Milestone 2A handoff

## Handoff status

Milestone 2A is complete as a characterization-only deliverable. The public
branch remains `release/parity-characterization`. The commit parent for this
handoff is:

`cc401f8dbe381fe3efe534219e6a912f955279a8`

The one new follow-up commit contains the eleven files listed below. Its exact
hash is reported by the final Git verification and handoff message; this file
does not self-reference a mutable hash.

## Files created

### Contracts

- `contracts/source_inventory.json`
- `contracts/data_preprocessing_contract.json`
- `contracts/vae_contract.json`
- `contracts/rectified_flow_contract.json`
- `contracts/evaluation_contract.json`
- `contracts/runtime_contract.json`

### Documents

- `docs/PARITY_CHARACTERIZATION.md`
- `docs/SCIENTIFIC_SOURCE_CONTRACTS.md`
- `docs/CANONICAL_ENTRYPOINTS_AND_DUPLICATES.md`
- `docs/MIGRATION_PLAN.md`
- `docs/MILESTONE_2A_HANDOFF.md`

No other public files are part of Milestone 2A. No Milestone 1 audit copy was
modified.

## Evidence handed forward

- Audit repository: branch `release/audit-and-inventory`, commit
  `f38bebce36c4f21d857dc084ac8d06759c2c012d`, clean at inspection.
- Immutable source: read-only inspection; selected manifest hashes and source
  hashes remained unchanged. The source tree is not treated as a Git worktree.
- Public repository before this change: branch
  `release/parity-characterization`, clean at parent
  `cc401f8dbe381fe3efe534219e6a912f955279a8`.
- The observed source tree baseline was independently checked as 4,667 files
  and 34,811,675,494 bytes; existing documented drift remains outside this
  milestone and was not copied.

## Validation performed

The pre-commit gate is:

1. Parse every `contracts/*.json` with Python.
2. Check required top-level fields and cross-contract subjects, labels, shapes,
   seeds, and sampler/evaluation steps.
3. Run the existing public tests with Python `-B`, cache provider disabled, and
   an external temporary base directory.
4. Run the repository absolute-path/artifact scanner and a targeted scan of new
   files for source-workstation markers.
5. Run `git diff --check`, inspect status and diff scope, and verify only the
   eleven required files are new.
6. Verify no prohibited extensions, directories, scientific `src` changes,
   remotes, or tags.
7. Recheck audit/source/public pins and selected immutable-source hashes.

The final handoff response records the exact command outputs and commit hash.

## Assumptions and blockers

- This is not an exact paper reproduction and no full REALDISP experiment was
  run.
- The Flow width conflict remains `128` manuscript versus `256` later/wrapper/
  checkpoint-shape evidence.
- The VAE schedule remains split between older defaults and later wrapper values.
- The observed VAE preprocessing path is six-channel only despite a structural
  three-channel checkpoint and separate-model policy.
- The VAE-safe subject split `0.15` and CNN internal split `0.20` are distinct.
- The observed duplicate audit is train-versus-validation only.
- The source has no historical environment lock; the current CPU runtime is not
  historical proof.
- Flow trainer duplication, PowerShell portability, undefined
  `supplementWin240`, optional `tslearn`, and partial table provenance remain
  migration gates.

## Recommended next milestone

Milestone 2B should resolve the evidence gates and build synthetic-only parity
fixtures before any scientific implementation migration. Its first decisions
should be Flow width, VAE schedule, 3CH lineage, safe-split interface, and the
trusted-checkpoint policy. It should keep raw data, participant artifacts,
checkpoints, results, logs, and manuscript history outside the public Git tree.

## Commit protocol

The follow-up commit must be the second commit on the branch, must not amend or
rewrite the Milestone 1 parent, and must use the Lore trailers required by the
repository instructions. No remote, tag, push, or publication action is part of
this handoff.
