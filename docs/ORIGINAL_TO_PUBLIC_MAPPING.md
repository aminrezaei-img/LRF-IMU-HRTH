# Original-to-Public Mapping

## Milestone 1 scope

This mapping records the copied or locked artifacts currently present in the
Milestone 1 public-release tree. The audit source of truth is commit
`f38bebce36c4f21d857dc084ac8d06759c2c012d` on
`release/audit-and-inventory`. The immutable scientific source tree was read
only; it was not modified.

The current copied/locked set is seven files: the locked reference YAML, the
verbatim run guide, and the five committed audit documents. The audit
repository's `release_manifest.json` is not present in the public tree and is
therefore not represented as a copied artifact here. Scaffold `.gitkeep`
files are placeholders, not copied or locked science artifacts.

No scientific implementation code was migrated in Milestone 1. In particular,
no model, preprocessing, generation, evaluation, checkpoint, raw-data,
synthetic-data, or `Results/` artifact was copied. Consequently, this mapping
does not establish behavioral equivalence and no code-equivalence claim exists.

## Status and verification terms

- **Copied verbatim:** file bytes were copied without content edits. A changed
  public filename or directory is recorded as a path-only difference.
- **Locked reference, synthesized then copied verbatim:** the audit-created
  artifact has no one-to-one original science-tree file; the public copy is
  byte-identical to the committed audit artifact.
- **Behavioral/hash verification:** these Milestone 1 artifacts are
  documentation or non-executing references. Verification therefore means
  direct byte comparison and SHA-256 equality, not execution of scientific
  behavior. No scientific implementation was run for this mapping.

SHA-256 values below were recomputed directly with SHA-256 over the source and
public files. All seven copied artifacts matched byte-for-byte.

## Artifact mapping

| Original/reference path | Public path | Status | Reason | Behavioral/hash verification | Known differences | Original SHA-256 | Public SHA-256 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Audit repository, commit `f38bebc`: `configs/locked/paper_release_reference.yaml` (synthesized from immutable-source evidence; no direct source-tree counterpart) | `configs/locked/paper_release_reference.yaml` | Locked reference, synthesized then copied verbatim | Preserve observed dataset, split, seed, hyperparameter, command, and uncertainty values as a non-executing audit reference. | Non-executing YAML; direct byte comparison: equal. | This is an audit synthesis, not an original research file and not an operational configuration. It deliberately records historical local paths and unresolved values. | `0999bf84429b9ac7e3b58969db83dab14f4097eef45515c453179d4daca3541d` | `0999bf84429b9ac7e3b58969db83dab14f4097eef45515c453179d4daca3541d` |
| Immutable source tree: `RUNNING_INSTRUCTIONS.md` | `configs/locked/RUNNING_INSTRUCTIONS.verbatim.md` | Copied verbatim; path/name only | Preserve the historical 42-stage run guide before any portability or scientific migration work. | Non-executing Markdown; direct byte comparison with the immutable source: equal. No command was executed. | Directory and filename changed; contents, including historical paths and commands, were not rewritten. | `1d6b6c48bf9742c630c40a858f5684140c7c7848983fdfab1901df2f5f9453a6` | `1d6b6c48bf9742c630c40a858f5684140c7c7848983fdfab1901df2f5f9453a6` |
| Audit repository, commit `f38bebc`: `docs/PAPER_RESULT_PROVENANCE.md` | `docs/PAPER_RESULT_PROVENANCE.md` | Copied verbatim | Carry forward the committed claim-to-script/result provenance map. | Documentation; direct byte comparison: equal. | Moved from the audit repository into the public-release tree; embedded historical paths and uncertainty statements remain. | `c8d818b67d015ef408ed3454491dbae2991d269e677bfd49e9498276de8bb61f` | `c8d818b67d015ef408ed3454491dbae2991d269e677bfd49e9498276de8bb61f` |
| Audit repository, commit `f38bebc`: `docs/PUBLIC_RELEASE_RISKS.md` | `docs/PUBLIC_RELEASE_RISKS.md` | Copied verbatim | Preserve the committed privacy, data, licensing, path, and large-artifact risk register. | Documentation; direct byte comparison: equal. | Moved from the audit repository; no risk statement or embedded historical path was edited. | `1a592a32d5be42ed903fca164bd1627d543e2dfd6c273075dec2397088e62dd8` | `1a592a32d5be42ed903fca164bd1627d543e2dfd6c273075dec2397088e62dd8` |
| Audit repository, commit `f38bebc`: `docs/RELEASE_HANDOFF.md` | `docs/RELEASE_HANDOFF.md` | Copied verbatim | Preserve the audit handoff, immutable-reference list, release order, and unresolved decisions for later milestones. | Documentation; direct byte comparison: equal. | Moved from the audit repository; it remains a historical handoff and is not an operational entry point. | `f5b91555b4958cea94ade7ce48346a37f0ef3a5a64b70dd93fafb1a20d31f24f` | `f5b91555b4958cea94ade7ce48346a37f0ef3a5a64b70dd93fafb1a20d31f24f` |
| Audit repository, commit `f38bebc`: `docs/RELEASE_INVENTORY.md` | `docs/RELEASE_INVENTORY.md` | Copied verbatim | Preserve the audited repository structure, entry-point inventory, generated-artifact inventory, and known gaps. | Documentation; direct byte comparison: equal. | Moved from the audit repository; it continues to describe the immutable source and audit state rather than a runnable public package. | `70d438cd1941b32ef86ee4afbeb2c25c57acd0f06bf516bf1839aff1fa1cc67c` | `70d438cd1941b32ef86ee4afbeb2c25c57acd0f06bf516bf1839aff1fa1cc67c` |
| Audit repository, commit `f38bebc`: `docs/REPRODUCIBILITY_AUDIT.md` | `docs/REPRODUCIBILITY_AUDIT.md` | Copied verbatim | Preserve the observed dataset, preprocessing, split, model, sampling, classifier, environment, and reproducibility assumptions. | Documentation; direct byte comparison: equal. | Moved from the audit repository; unresolved discrepancies remain intentionally unresolved. | `e49744e73cb5154c317a6ff7810b11f851244b190a6e2132e9140b4dd92a14d3` | `e49744e73cb5154c317a6ff7810b11f851244b190a6e2132e9140b4dd92a14d3` |

## Intentional scanner compatibility distinction

The historical audit and locked-reference files are an intentional scanner
compatibility exception. The verbatim run guide, locked YAML, and copied audit
provenance documents may retain original workstation references because
changing them would destroy the historical evidence and invalidate the
recorded hashes. The locked YAML explicitly says it is a non-executing
inventory copy; it is not a release default or executable configuration.

This exception is narrow and explicit: no operational code, executable
configuration, or operational default may contain a source-drive/project-root
literal, a user-home literal, a local account name, or another
machine-specific path. A future scanner must allow these named
historical-reference files only under this documented exception and must fail
on such paths elsewhere. Do not conceal the exception by silently rewriting
the locked evidence; parameterizing operational paths is a later milestone.

## Files deliberately not mapped as copied artifacts

The following were not copied into the public tree in Milestone 1: scientific
source modules and wrappers; preprocessing or evaluation code; raw REALDISP
logs; checkpoints; `.pt`, `.npz`, or `.pkl` files; synthetic caches; `Results/`
outputs; manuscript/review history; logs; `.claude/`; and the current
audit-to-source drift described in `KNOWN_DISCREPANCIES.md`. No equivalence
claim can be made for any of these items from this milestone.

## Milestone 3A public implementations

Milestone 3A adds public contract implementations rather than one-to-one copies
of immutable-source modules. The mapping is:

| Historical/evidence contract | Public M3A surface | Status |
| --- | --- | --- |
| Four-class activity vocabulary and 120-column right-thigh layout | src/lrf_imu/data/activities.py, schema.py | Synthetic contract locked |
| REALDISP ideal-log discovery and selected-channel loading | src/lrf_imu/data/realdisp.py | External-root, nonrecursive, no participant data copied |
| Filter-before-runs and complete-window construction | src/lrf_imu/data/windowing.py | Compatibility default plus explicit strict mode |
| VAE-safe subject split and separate CNN split | src/lrf_imu/data/splits.py | 0.15 and 0.20 remain distinct |
| Training-only standardization | src/lrf_imu/data/normalization.py | Population ddof=0, floor 1e-8 |
| Exact-window duplicate audit | src/lrf_imu/data/audit.py | SHA-1 canonical bytes; public all-pair scope |
| Preparation orchestration and safe output | src/lrf_imu/data/pipeline.py, cli.py | Metadata-only; explicit write permission |

The M3A code is a public safety implementation derived from the locked contracts
and synthetic fixtures. It is not a claim that the historical scientific source
was copied, that participant artifacts are reproducible from this tree, or that
any VAE/model migration occurred. The reconstructed 3CH path is explicitly
separate and remains a fresh-training requirement.
## M3A configuration-resource mapping

| Human-facing evidence path | Installed runtime path | Status and guard |
| --- | --- | --- |
| configs/paper/six_channel_160_40.yaml, accelerometer_only_160_40.yaml, and sensitivity_grid.yaml | src/lrf_imu/resources/configs/paper/ with the same filenames | Intentional package-data copies for wheel portability. tests/test_packaging.py compares normalized bytes and parses both sides. This is configuration-resource synchronization, not a scientific source-copy or exact-paper claim. |

The root YAMLs remain reviewable evidence. The packaged copies only remove the
repository-checkout dependency from the default runtime path. No participant
artifact, checkpoint, VAE/model implementation, or machine-specific path is
introduced by this mapping.