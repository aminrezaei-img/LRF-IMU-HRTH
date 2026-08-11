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
## Milestone 3B VAE mapping

M3B adds the following public compatibility surfaces. The source tree and
historical checkpoint payloads remain read-only external evidence; no payload is
copied into the public tree.

| Historical/evidence path | Public path | Status | Minimal change and guard |
| --- | --- | --- | --- |
| <immutable-source>/VAE/VAE_logic.py (3C989BB8...) | src/lrf_imu/models/vae.py | Copied model semantics with boundary guards | Layer order, names, dimensions, activations, and forward equations preserved; 3CH/6CH validation and keyword aliases added. |
| <immutable-source>/VAE/VAE_logic.py and Run_VAE_Pretraings.ps1 | src/lrf_imu/training/vae.py | Compatibility training surface | Augmentation, loss, profile, optimizer, and checkpoint semantics represented without import-time path mutation or dataset orchestration. |
| Observed vae state-dict checkpoint boundary | src/lrf_imu/checkpoints.py | Safe loader/inspector | weights_only=True, exact root key, 26-key schema, channel/shape checks, and explicit paths. |
| Historical VAE CLI/evaluation entry points | src/lrf_imu/cli.py | No-write public probes | CPU smoke, safe checkpoint inspection, and deterministic reconstruction only. |

Behavioral parity is established for synthetic inputs, both named external
subject-01 checkpoints, and one prepared external REALDISP fold. This mapping
does not establish training-environment identity, historical metric identity,
Flow/TSTR parity, or exact paper reproduction.

## Milestone 3C Flow mapping supplement

| Source evidence | Public staging implementation | Integration decision |
| --- | --- | --- |
| `models/unet_1d.py` | `src/lrf_imu/models/flow.py` | Preserve the source block topology and expose explicit 128/256 profiles; historical loading selects width 256 only when schema confirms it |
| `LRF/rectified_flow.py` | `src/lrf_imu/training/flow.py` | Preserve posterior-mean/noise interpolation, `1000*t` model time, MSE target, and reverse Euler equations |
| `1_Rectified_Flow_training.py` | `src/lrf_imu/training/flow.py` | Use one public implementation; the source duplicate `rectified_flow_training.py` is byte-identical and is not maintained as a second copy |
| Source generation call used by `TSTR.py` | `src/lrf_imu/generation/flow.py` | Keep the paper sampler metadata-only and separate from website trajectory generation |
| `1_train_flow.ps1` and running instructions | `src/lrf_imu/checkpoints.py` and CLI | Load explicit paired checkpoint files with weights-only validation and safe metadata inspection |
| `8_export_website_trajectories.py` behavior | `export-trajectories` command/profile | Use a distinct website profile; never label it paper/TSTR |

The public integration changes are limited to Flow model/training/generation boundaries, checkpoint validation, lazy package exports, CLI aliases and validation, one M3C test module, and the permitted M3C documents/contract. Existing M3B VAE modules, contracts, tests, and behavior remain intact.

Raw SHA-256 hashes for the source evidence and final staged files are recorded in `contracts/flow_parity_report.json`. The source U-Net, rectified-flow equations, and duplicate-training-script identity were checked against the named source files; no source payloads or Results artifacts were copied.
## Milestone 3D evaluation mapping

| Source evidence | Public path | Decision |
| --- | --- | --- |
| `TSTR.py` and active patched four-scenario evaluator | `src/lrf_imu/evaluation/{core,scenarios,classifiers}.py` | Preserve population, RF/CNN, seed, and classifier-validation semantics with portable in-memory functions |
| Historical metric/retention/confusion helpers | `src/lrf_imu/evaluation/metrics.py` | Preserve labels 0--3, zero-division behavior, fold-wise retention, `ddof=1`, and nanmean/nonzero-cell aggregation |
| Synthetic-cache wrappers | `src/lrf_imu/evaluation/cache.py` | Explicit external cache identity and weights-free metadata; no implicit source root |
| Active one-fold/all-fold wrappers | `src/lrf_imu/evaluation/cli.py` and root CLI | Explicit data/cache/output roots, dry-run, write permission, overwrite refusal, and resume |
| Immutable stored evaluation artifacts | `contracts/evaluation_reference_map.json` and `evaluation_parity_report.json` | Small relative-path/metric evidence only; no Result payload copied |

Gate A fixtures, all-fold execution, and fold-level historical comparisons are
documented in `docs/EVALUATION_PARITY_REPORT.md`.

## Milestone 3E analysis mapping

| Source evidence | Public path | Decision |
| --- | --- | --- |
| `vae_ablation_loso.py` | `src/lrf_imu/analysis/ablation.py` | Preserve posterior-mean encoding, flattened class-diagonal Gaussian, population SD + `1e-6`, sequential NumPy RNG, and RF protocol; portable public data/checkpoint boundary |
| `window_grid_aug_quality.py` and stored 108-fold grid | `src/lrf_imu/analysis/sensitivity.py` | Numeric aggregation only; nine settings, fold mean and sample SD; no plotting |
| `6_phyiscs_plausibility.py` | `src/lrf_imu/analysis/physical.py` | First three channels, physical m/s², gravity 9.80665, strict `>10g` |
| `9_1_summarize_psd_efficiency_across_folds.py` | `src/lrf_imu/analysis/spectral.py` | Welch/log-PSD and trapezoidal band ratios; explicitly retain high-frequency attenuation |
| `membership_inference_holdout.py` | `src/lrf_imu/analysis/privacy.py` | True-training-holdout minimum-distance MIA kept separate |
| `12_privacy_memorization_audit.py` | `src/lrf_imu/analysis/privacy.py` | Post-hoc best-attack MIA and reconstruction criterion kept distinct; no privacy guarantee |
| Historical command wrappers | `src/lrf_imu/analysis/cli.py` and root CLI | Explicit inputs, metadata output by default, explicit write permission |

Selected immutable-source and result-artifact hashes, execution scope, and
known differences are recorded in `contracts/analysis_parity_report.json`.
No original file was copied unchanged, and no Results payload entered Git.
