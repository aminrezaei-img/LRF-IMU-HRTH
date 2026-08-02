# Known Discrepancies

## Scope and release posture

This register preserves unresolved evidence from the committed audit at
`f38bebce36c4f21d857dc084ac8d06759c2c012d` and read-only inspection of the
immutable source tree. It does not select a winning manuscript, wrapper, or
artifact value. `Unknown`, `not observed`, and `not applicable` are deliberate
states, not omissions.

Milestone 1 is documentation-only. No scientific implementation code,
preprocessing, model, evaluation, checkpoint, raw data, synthetic cache, or
`Results/` output was migrated. These discrepancies therefore remain release
blockers for exact paper-reproduction claims; no code-equivalence claim exists.

## Summary

| ID | Issue | Current decision |
| --- | --- | --- |
| S1 | VAE reconstruction weights and KL schedule differ across manuscript variants and observed wrappers. | Use the observed-wrapper values only as provisional compatibility defaults in executable release configs to minimize behavioral change; they are not authoritative manuscript values, do not resolve the discrepancy, and do not support exact-paper-equivalence claims. |
| S2 | Rectified Flow base width differs across manuscript variants and observed wrapper/checkpoint metadata. | Use the observed-wrapper value only as a provisional compatibility default in executable release configs to minimize behavioral change; it is not an authoritative manuscript value, does not resolve the discrepancy, and does not support exact-paper-equivalence claims. |
| S3 | 6CH/3CH generative-quality and TSTR utility tables have partial or manual/post-processed provenance. | Do not copy or cite as regenerated release artifacts until the final inputs and outputs are recovered. |
| S4 | Distributional-similarity signal-space standard deviations do not match the observed summary CSV. | Preserve manuscript and observed summary values; reconcile aggregation before using the SDs as reproduction evidence. |
| S5 | Exact final accepted-manuscript figure-copy paths are missing from the inspected source tree. | Do not copy figures or claim accepted-figure equivalence. |
| O1 | Audit-to-current D: drift after the 2026-07-30 audit. | Exclude, do not review as Milestone 1 evidence, and do not copy. |

## S1 — VAE loss weights and KL schedule

### Manuscript value

The older/highlighted manuscript variants recorded in the source tree report

- `w_L2=1.0`, `w_L1=0.1`;
- `beta_init=5e-3`, `beta_min=1e-5`, and `beta_decay=0.7`.

The later candidate `manuscript/Manuscript_02_06_2026.tex` records
`w_L2=0.5`, `w_L1=0.1`, and `beta_init/beta_min/beta_decay=0.08/0.04/0.995`.
The committed audit treats the accepted manuscript identity as unresolved, so
these are preserved as manuscript-version evidence rather than reconciled.

### Observed wrapper/code value

`VAE/Run_VAE_Pretraings.ps1` sets
`VAE_L2_WEIGHT=0.5`, `VAE_L1_WEIGHT=0.1`, `VAE_BETA_INIT=0.08`,
`VAE_BETA_MIN=0.04`, and `VAE_BETA_DECAY=0.995`. The locked YAML and
`REPRODUCIBILITY_AUDIT.md` record the same observed wrapper values.

### Observed checkpoint metadata

The observed `Results/model_weights/vae_weights/6CH/full/subject_01/vae_run_meta.json`
records `batch_size=256`, `max_epochs=1000`, `latent_stride=4`, `latent_dim=48`,
and bfloat16 AMP enabled. It does **not** record `w_L2`, `w_L1`, or the beta
schedule. The metadata therefore cannot select between the manuscript value
sets.

### Available evidence

- `docs/REPRODUCIBILITY_AUDIT.md` and `configs/locked/paper_release_reference.yaml`
  preserve the discrepancy.
- `VAE/Run_VAE_Pretraings.ps1` contains the observed wrapper settings.
- `manuscript/Manuscript_Highlighted_Revisions.tex` and
  `manuscript/Manuscript_old.tex` contain the older `1.0/0.1` and
  `5e-3/1e-5/0.7` values.
- `manuscript/Manuscript_02_06_2026.tex` contains the later wrapper-matching
  values.
- The checkpoint metadata above was inspected read-only and was not copied.

### Reproduction impact

The reconstruction weighting and KL schedule can change VAE training and
therefore downstream latent and generation results. A run using the older
manuscript values cannot be treated as the same run as one using the observed
wrapper values without an authoritative run-to-manuscript link.

### Current release decision

Keep both value sets intact. The active executable release configs use the
observed-wrapper values only as provisional compatibility defaults to minimize
behavioral change. They are not authoritative manuscript values, do not resolve
this discrepancy, and do not support exact-paper-equivalence claims. Leave VAE
migration outside Milestone 1.

### Evidence needed to resolve

Obtain the accepted publisher proof or an authoritative final manuscript
source, the exact configuration used for the reported checkpoints, and a
checkpoint/run manifest that records the loss weights and beta schedule. Only
after those are linked can a release default and parity test be selected.

## S2 — Rectified Flow base width

### Manuscript value

The older/highlighted manuscript variants report U-Net base width `C=128` and
channel widths `128 -> 256 -> 512 -> 256 -> 128`. The later candidate
`manuscript/Manuscript_02_06_2026.tex` reports `C=256`. The accepted proof or
authoritative version is not established, so the committed audit's `C=128`
versus observed `C=256` discrepancy remains open even though one later source
candidate matches the wrapper.

### Observed wrapper/code value

`1_train_flow.ps1` sets `FLOW_MODEL_CH=256`. The corresponding observed
architecture uses the 256-base channel progression, and the locked YAML records
the same wrapper value. `models/unet_1d.py` is an immutable candidate
implementation, not a Milestone 1 migration.

### Observed checkpoint metadata

`Results/model_weights/flow_weights/6CH/full/subject_01/train_summary.json`
records `model_ch=256`, `batch_size=512`, `latent_dim=48`, `latent_stride=4`,
and `total_params=15447728`. This is subject-01 metadata only; it does not
prove that every historical fold or the accepted manuscript used the same
architecture.

### Available evidence

- `docs/REPRODUCIBILITY_AUDIT.md` and `configs/locked/paper_release_reference.yaml`
  preserve the `C=128` versus `C=256` discrepancy.
- `1_train_flow.ps1` directly sets `FLOW_MODEL_CH=256`.
- `manuscript/Manuscript_Highlighted_Revisions.tex` and
  `manuscript/Manuscript_old.tex` contain `C=128`.
- `manuscript/Manuscript_02_06_2026.tex` contains `C=256`.
- The subject-01 flow summary above was inspected read-only and was not
  copied.

### Reproduction impact

Base width changes the U-Net parameterization, memory use, and learned velocity
field. A `C=128` implementation and a `C=256` implementation are not
interchangeable for exact checkpoint or numerical reproduction.

### Current release decision

Keep both value sets intact. The active executable release configs use the
observed-wrapper value only as a provisional compatibility default to minimize
behavioral change. It is not an authoritative manuscript value, does not resolve
this discrepancy, and does not support exact-paper-equivalence claims. Do not
copy the checkpoints or migrate Flow implementation code in Milestone 1.

### Evidence needed to resolve

Obtain the accepted proof/source, the exact Flow configuration for the reported
run, and per-fold run manifests or checkpoint metadata containing the model
width and state-dict architecture. Record the authoritative file hashes before
choosing a public configuration.

## S3 — Partial provenance for 6CH/3CH comparison tables

### Manuscript value

The highlighted/revision manuscript variant reports the following comparison
values, but the final generated table artifacts were not established:

- Generative quality: diversity `22.17 +/- 0.82` (6CH) versus
  `15.68 +/- 0.52` (3CH), minimum novelty distance `9.24 +/- 2.43` versus
  `4.80 +/- 0.49`, coverage ratio `101.3% +/- 5.0%` versus
  `99.3% +/- 2.3%`, and PCA area ratio `0.868 +/- 0.092` versus
  `0.889 +/- 0.072`, with manuscript retention values `70.7%`, `52.0%`,
  `98.0%`, and `102.5%`, respectively.
- Low-data TSTR/augmentation: the manuscript reports RF scarce-data
  `0.40 +/- 0.09` (6CH) and `0.47 +/- 0.08` (3CH), CNN scarce-data
  `0.34 +/- 0.19` and `0.44 +/- 0.20`, RF augmented `0.95 +/- 0.09` and
  `0.98 +/- 0.06`, and CNN augmented `0.86 +/- 0.15` and `0.97 +/- 0.06`.

These are manuscript-variant values, not newly asserted results and not a
claim that the public tree can regenerate them.

### Observed wrapper/code value

The source contains `aggregate_generative_quality_6v3.py` and
`make_tstr_table_6v3.py`, which are intended to aggregate 6CH/3CH inputs. The
expected generative comparison file
`Results/distinctness_and_population_coverage/ablation_comparison/generative_quality_6v3_comparison.csv`
is missing in the inspected source tree. The `tstr_utility_6v3.tex` file is
present but is a 379-byte header-only table stub with no rows. Upstream per-fold
summary files exist, but no complete final comparison output and no
authoritative rounding/transfer path were established.

### Observed checkpoint metadata

Not observed and not established for the table rows. The committed audit does
not connect a checkpoint manifest to the 6CH/3CH comparison values. Checkpoint
directories are excluded from this public tree, so no checkpoint metadata is
being inferred or copied.

### Available evidence

- `docs/PAPER_RESULT_PROVENANCE.md` labels both the generative-quality and
  TSTR 6CH/3CH tables `Partial`.
- `aggregate_generative_quality_6v3.py` and `make_tstr_table_6v3.py` are
  present in the immutable source tree.
- The expected generative comparison CSV is absent.
- `tstr_utility_6v3.tex` contains the table header but no data rows.
- Manuscript variants contain the values listed above.

### Reproduction impact

The exact table values, fold inputs, rounding, and any manual post-processing
cannot be reconstructed from one complete generated artifact. Re-running a
similarly named script or using a visible upstream CSV would not by itself
prove that the manuscript table was reproduced.

### Current release decision

Do not copy these result tables, treat their manuscript values as unresolved
provenance, and do not claim that the public release regenerates the 6CH/3CH
comparison. No result or checkpoint artifact was migrated in Milestone 1.

### Evidence needed to resolve

Recover the exact final comparison CSV/TEX files, all 6CH and 3CH per-fold
inputs, the script revisions and command lines used to generate them, the
sampling/configuration manifests, and the authoritative accepted manuscript
table. Hash and compare those artifacts before making any reproduction claim.

## S4 — Distributional-similarity standard deviations

### Manuscript value

The manuscript reports, across LOSO folds:

- signal-space MMD squared: `(4.51 +/- 0.63) x 10^-3`;
- signal-space C2ST: `87.96 +/- 1.42%`;
- latent-space MMD squared: `(3.20 +/- 1.83) x 10^-2`;
- latent-space C2ST: `92.37 +/- 3.85%`.

### Observed wrapper/code value

The observed `Results/distinctness_and_population_coverage/6CH/full/Combined_Manifold_ALL/combined_manifold_metrics_across_folds.csv`
contains:

```text
metric,mean,sd
mmd_signal,0.00451356497605578,0.002977557839774243
c2st_signal_mean,0.8796818528861557,0.06364243859264916
mmd_latent,0.03196123686938951,0.01803509623239333
c2st_latent_mean,0.9237223426795395,0.03847410883539999
```

The signal-space mean is close to the manuscript mean, but the observed
signal-space SDs (`0.00298` for MMD and `0.06364` for C2ST) are larger than the
manuscript SDs (`0.00063` and `0.0142`). The latent-space values are numerically
consistent with the manuscript values. This difference may reflect an
aggregation convention, a later artifact, or another source version; no cause
has been selected.

### Observed checkpoint metadata

Not applicable/not observed for this summary. The metric CSV does not contain
checkpoint metadata, and the committed audit does not establish a checkpoint
lineage for the reported SD convention.

### Available evidence

- `docs/PAPER_RESULT_PROVENANCE.md` explicitly flags the MMD/C2ST SD issue.
- The source CSV above was inspected read-only; its SHA-256 is
  `d490002323548ee3369c019345b254041fde0abb16795c2175985d45bab398cb`.
- `manuscript/Manuscript_02_06_2026.tex` and related variants report the
  manuscript values above.
- The all-fold summarization entry point is recorded in the locked run guide;
  its exact historical revision and aggregation convention have not been
  linked to the manuscript proof.

### Reproduction impact

The mean signal-space MMD is close, but the reported variability and C2ST
interpretation cannot be reproduced from the observed summary CSV as written.
Using the wrong SD convention could misstate uncertainty without changing the
underlying fold means.

### Current release decision

Preserve both manuscript and observed summary values. Do not overwrite either,
copy the result CSV, or claim exact reproduction of the distributional-
similarity uncertainty until the aggregation is reconciled.

### Evidence needed to resolve

Recover the per-fold MMD/C2ST records, the exact summarizer revision and
arguments, and the accepted manuscript/proof table. Determine whether the
reported SD is fold-level, pooled, bootstrap, or another convention, then
recompute from the same records and record hashes.

## S5 — Missing final accepted-manuscript figure-copy paths

### Manuscript value

The candidate manuscript includes figures from a `figures/` directory,
including `figure1_pipeline.pdf`, `NeuralBlocks_minimal.pdf`,
`3a_Training_objective.pdf`, `Architecture_revised.pdf`,
`3b_Inference_ODE.tex`, `figure2_training_curves_updated_2.pdf`,
`Fig4_Flow_Trajectory_6Channel_steps10.pdf`,
`Fig3_Physics_Validation_AllFolds.pdf`,
`figure5_structural_consistency_subject02.pdf`,
`fig_tstr_summary_combined.pdf`, `Fig_5_violin_Confusion_Matrices.pdf`,
`Fig_Combined_Manifold_ALL.pdf`, `Fig_Coverage_Summary_AllFolds.pdf`,
`Fig_PSD_Frequency_Analysis_all_6ch.pdf`, and
`Fig_Data_Efficiency_all_6ch.pdf`. The exact accepted publisher figure package
and its source path are not established.

### Observed wrapper/code value

The audit found generated result figures under `Results/...`, and the locked
run guide names result-generation scripts and output locations. The inspected
source tree has no top-level `figures/` directory, so no source-to-submission
copy operation or final figure manifest was observed. A generated result path
is not treated as the accepted manuscript copy path.

### Observed checkpoint metadata

Not applicable/not observed. Figure-copy identity is not established by model
checkpoint metadata, and no checkpoint or generated figure was copied in
Milestone 1.

### Available evidence

- `docs/PAPER_RESULT_PROVENANCE.md` marks the pipeline/architecture figures
  unestablished and several result figures partial.
- The candidate manuscript contains the `figures/...` include paths listed
  above.
- A read-only check found the expected top-level `figures/` directory missing.
- Result-generation locations are documented in
  `configs/locked/RUNNING_INSTRUCTIONS.verbatim.md` and the audit inventory.

### Reproduction impact

The public release cannot identify the exact PDFs/PNGs submitted to the
publisher, their hashes, or whether an observed `Results/` figure is the same
artifact. Generating a similarly named figure does not establish accepted-
figure equivalence.

### Current release decision

Do not copy figures, do not copy `Results/`, and do not claim that the public
tree contains or regenerates the accepted publisher figure package. Keep the
figure identity unresolved.

### Evidence needed to resolve

Obtain the publisher proof or submission archive, the final figure directory or
manifest, original figure files with SHA-256 hashes, and a mapping from each
manuscript include path to its source-generation command. Perform a metadata,
privacy, and rights review before any figure is considered for release.

## O1 — Audit-to-current D: drift after the 2026-07-30 audit

This is an operational provenance issue, not a sixth scientific claim.

### Manuscript value

Not applicable. These files were not part of the committed audit evidence and
are not used here to add or revise a manuscript result.

### Observed wrapper/code value

Compared with the 2026-07-30 audit, the current immutable D: tree contains 48
new website trajectory JSON files under
`Results/trajectories_website/subject_XX/` (four activities for each of the
12 audited subjects). It also contains four rewritten Flow-trajectory image
artifacts, all dated 2026-07-31:

- `Results/flow_trajectory/subject_01/Fig4_Flow_Trajectory_6Channel_steps10.pdf`
- `Results/flow_trajectory/subject_01/Fig4_Flow_Trajectory_6Channel_steps10.png`
- `Results/flow_trajectory/subject_02/Fig4_Flow_Trajectory_6Channel_steps10.pdf`
- `Results/flow_trajectory/subject_02/Fig4_Flow_Trajectory_6Channel_steps10.png`

The 48 JSON files and four image files are audit-to-current drift, not
Milestone 1 release inputs. Their generator configuration, lineage, and
relationship to the audited artifacts are not established here.

### Observed checkpoint metadata

Not observed/not established. No checkpoint, run manifest, or model lineage was
attached to these post-audit files for this register.

### Available evidence

- The committed audit inventory and manifest are dated 2026-07-30.
- Direct read-only file inspection of D: found the 48 JSON files with
  2026-07-31 timestamps and the four listed flow-trajectory files with
  2026-07-31 timestamps.
- The audit comparison explicitly records this 48-file/four-image drift.

### Reproduction impact

These post-audit outputs cannot be assumed to match the audited source state,
paper figure package, or documented sampling provenance. Including them would
mix reviewed and unreviewed states and would weaken the audit-to-public mapping.

### Current release decision

Treat all 48 website trajectory JSON files and all four rewritten
flow-trajectory image files as **excluded, unreviewed, and not copied**. Do not
use them as Milestone 1 evidence and do not repair the source tree by deleting
or reverting them.

### Evidence needed to resolve

If these outputs are considered later, recover the exact exporter/plotting
script revision, configuration, seed, model/checkpoint lineage, generation
timestamps, output hashes, schema/visual checks, and privacy/rights review.
Compare them against the 2026-07-30 audit snapshot before deciding whether
they are release-safe or scientifically relevant.

## Intentional scanner compatibility distinction

The copied historical audit and locked-reference files may retain original
workstation references so that their evidence remains verbatim and hashable.
This is a documented compatibility exception, not permission for operational
paths. No operational code, executable configuration, or default may contain a
source-drive/project-root literal, a user-home literal, a local account name,
or another workstation-specific path. Milestone 1 contains no scientific
implementation code or operational defaults; future migrated code must
parameterize all such paths and must fail a scanner when they appear outside
the named historical-reference files.

## Milestone 3A implementation status

M3A closes the public data-preparation interface without closing the scientific
evidence gaps. The synthetic contract suite validates the 120-column schema,
right-thigh 80..85 selection, label 119, raw-to-encoded mapping 1/3/4/33 ->
0/1/2/3, 160/40 windows, filter-before-runs compatibility, strict contiguity,
0.15 VAE subject splitting, compact 16/7/8 parity, training-only ddof=0
standardization, and SHA-1 exact-window duplicate identity.

The following remain intentional discrepancies or boundaries:

- The public default is metadata-only and external-root based. It does not write
  raw logs, participant-derived windows, checkpoints, synthetic caches, or result
  payloads, even where historical wrappers wrote local artifacts.
- The 3CH path is a public reconstruction over columns 80..82 for a separate model.
  It is not an inference-time drop from 6CH and does not recover historical VAE
  or checkpoint lineage.
- VAE schedule, Flow width, model/checkpoint provenance, full evaluation provenance,
  and exact accepted-manuscript identity remain unresolved. No exact-paper claim is
  supported, and no VAE/model migration is part of M3A.
- The public duplicate audit checks all split pairs by default; the historical
  train/validation-only behavior is retained only as an explicitly named adapter.

No participant artifacts were added by M3A. The existing historical-reference
scanner exception remains narrow and applies only to named provenance documents.
## M3A correction status: validation fractions and package defaults

The paper-facing YAMLs retain split.validation_fraction: 0.20 as historical
classifier/window compatibility evidence. It is now accompanied by the
explicit split.classifier_window_validation_fraction: 0.20 field and the
separate VAE subject-level field
split.vae_subject_validation_fraction: 0.15. The parser rejects disagreement
between the legacy alias and the named classifier value, while the pipeline
reads the explicit VAE value. The 0.20 value is not silently relabeled.

The installed package carries synchronized copies of the paper YAMLs as
intentional package data so its default config does not walk to a repository
root. The root YAMLs remain human-facing evidence and synchronization is
test-guarded. This resolves the packaging/runtime discrepancy only. It does
not resolve VAE schedule, Flow width, checkpoint lineage, participant-data
access, or exact-paper identity.
## Milestone 3B status: VAE compatibility versus historical metrics

M3B resolves the implementation-level VAE state and forward-compatibility
boundary for the named 3CH and 6CH subject-01 checkpoints. The public model and
the immutable model have exact zero maximum error for synthetic deterministic
outputs, checkpoint-loaded deterministic outputs, and the one-fold comparison
on identical normalized inputs. This does not resolve the training or
evaluation environment used to produce historical scalar metrics.

The public one-fold validation reconstruction was compared with the stored
subject-01 validation JSON. The public versus stored absolute differences were
MSE 9.8871e-05, L1 1.1033e-04, FFT 3.4705e-04, and KL 8.4436e-04.
Possible contributors include historical preprocessing/runtime details; the
available evidence does not isolate one cause. Gate E therefore remains
PARTIAL, and these differences must not be silently relabeled as exact
historical metric parity.

The VAE schedule conflict remains open: the observed wrapper records L2/L1
weights 0.5/0.1 and beta 0.08/0.04/0.995, while older manuscript evidence
records 1.0/0.1 and 0.005/0.00001/0.7. M3B preserves both evidence sets and
keeps exact_paper_reproduction: false.
