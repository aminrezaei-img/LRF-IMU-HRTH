# Canonical entrypoints and duplicates

## Reading rule

Canonicality is evidence-tiered, not absolute. The order used here is:

1. locked run guide and audit references;
2. observed wrapper call sites;
3. current source defaults and imports;
4. manuscript and historical naming.

An entrypoint can be the observed path for a workflow and still be blocked for
public migration.

## Wrapper-to-module map

| Caller | Target | Classification | Migration posture |
| --- | --- | --- | --- |
| `<immutable-source>/VAE/Run_VAE_Pretraings.ps1` | `<immutable-source>/VAE/VAE_logic.py` | observed VAE wrapper path | wrap after schedule/3CH gates |
| `<immutable-source>/1_train_flow.ps1` | `<immutable-source>/1_Rectified_Flow_training.py` | documented direct Flow path | blocked pending width and duplicate decision |
| `<immutable-source>/2_generate_synthetic_all_folds.ps1` | `<immutable-source>/2_generate_synthetic_only.py` | generation wrapper path | minimal-edit after path gate |
| `<immutable-source>/2_generate_synthetic_only.py` | `<immutable-source>/TSTR.py` | generation/evaluation dependency | wrap with standardized-coordinate tests |
| `<immutable-source>/3_run_loso_evaluation.ps1` | `<immutable-source>/TSTR.py` | evaluation wrapper path | wrap |
| `<immutable-source>/3_run_full_losocv.py` | `<immutable-source>/rectified_flow_training.py` | live import of duplicate | blocked until caller audit |
| `<immutable-source>/run_tstr_all_subjects.ps1` | `<immutable-source>/8_tstr_classification_figure_org_PATCHED_v3.py` | current patched four-scenario path | wrap and preserve provenance |

## Flow trainer duplicate

`1_Rectified_Flow_training.py` and `rectified_flow_training.py` are byte-identical
operational duplicates. Both have SHA-256
`9DA925EC82F0704C4DEC2E55C37730327D1AEEA460F8A1ABB3197BE928D2CFB4`.

The prefixed file is the documented direct wrapper target. The unprefixed file
is imported by the combined LOSO script. Milestone 2A does not select one as a
global authority. Future work should audit every caller, select one source of
truth, and retain a tested compatibility alias only if a live caller requires
the old name.

## Patched versus legacy analysis

The patched four-scenario evaluator
`8_tstr_classification_figure_org_PATCHED_v3.py` is the current observed path.
It must remain distinct from `TSTR.py`, which implements the core TRTR/TSTR
evaluation path.

The patched violin/confusion script
`13_classification_violin_confusion_PATCHED.py` is current relative to the
unpatched `13_classification_violin_confusion.py`. The unpatched file remains a
legacy comparison candidate only; it is not a migration target without result
provenance.

## Historical names and defaults

`SPECTRAL_LOSS_OFF` survives in historical output names and path documentation.
Current VAE loss inspection found reconstruction L2/L1 plus KL and no active
spectral-loss term. The name is therefore historical metadata, not proof of a
different training objective.

Wrapper values and Python defaults differ in VAE schedule, batch size, early
stopping, and some Flow orchestration settings. The later wrapper-compatible
profile is retained in public configs as a compatibility choice only. It is not
selected as accepted-manuscript authority.

`VAE/main.py` is an alternate QA/import surface with different defaults and no
current wrapper evidence establishing it as a release entrypoint. A fallback
`data.realdisp` import in coverage analysis is unreachable unless the primary
VAE import fails; no such package is part of the public tree.

## Entrypoint migration rules

- Keep wrappers thin and make interpreter, roots, device, and artifact policy
  explicit.
- Do not migrate a Flow trainer before resolving the duplicate and width gate.
- Do not treat a website trajectory exporter as the paper sampler. The website
  path uses 100 Euler steps, `record_every=2`, 10-second overlap-add output, and
  derived seeds; paper sampling uses 10 steps, seed 42, and 500 per class.
- Keep the patched and unpatched analysis paths separate in provenance records.
- Keep historical names visible only as documentary lineage.
