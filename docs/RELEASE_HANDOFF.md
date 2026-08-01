# Release Handoff

## Summary

This handoff is for a future coding agent preparing the LRF-IMU accepted paper code for public release. The accepted DOI is user supplied as `10.1088/3049-477X/ae91ef`. The inspected source tree is:

`D:/PAPER2/CODEX_CLEANUP - Copy/BACKUP THE NIGHT BEFORE SUBMISSION/VERSION AT THE TIME OF SUBMISSION/LRF-IMU-P3`

The current branch in the release-audit workspace is:

`release/audit-and-inventory`

The original source tree was treated as read-only during this audit.

## What Has Been Verified

- The current audit workspace is a git repository on branch `release/audit-and-inventory`.
- The inspected `LRF-IMU-P3` tree is not itself a git repository.
- The main pipeline is documented in `RUNNING_INSTRUCTIONS.md`.
- The fold subjects are `1, 2, 3, 5, 8, 9, 10, 11, 12, 13, 14, 16`.
- The paper-specific task uses REALDISP ideal-placement, right-thigh, four activities: walking, running, jump_up, and cycling.
- Main windowing is 160 samples with hop 40.
- The release workspace now contains a locked reference YAML and a verbatim copy of the run guide.
- The major result folders and large artifacts have been inventoried.
- Multiple provenance gaps and release risks have been identified before cleanup.

## What Remains Uncertain

- Exact identity of the accepted publisher source versus `manuscript/Manuscript_02_06_2026.tex`.
- Independent publisher-side verification of DOI from web search in this session.
- Complete environment lock file.
- Exact final manuscript figure-copy directory, since manuscript include paths reference `figures/` but that folder was not found in the inspected tree.
- VAE hyperparameter discrepancy between manuscript text and observed wrapper settings.
- Flow base-width discrepancy between manuscript supplement and observed `FLOW_MODEL_CH=256`.
- Whether all final 6CH/3CH tables can be regenerated from visible final artifacts without manual transfer.
- Whether any derived checkpoints or synthetic caches are safe to share publicly.

## Immutable Reference Artifacts

Do not modify these before making a separate archival copy:

```text
RUNNING_INSTRUCTIONS.md
Readme.md
renaming_paths_vae.md
manuscript/Manuscript_02_06_2026.tex
VAE/VAE_logic.py
VAE/Run_VAE_Pretraings.ps1
1_train_flow.ps1
1_Rectified_Flow_training.py
2_generate_synthetic_all_folds.ps1
2_generate_synthetic_only.py
3_run_loso_evaluation.ps1
3_run_full_losocv.py
TSTR.py
run_tstr_all_subjects.ps1
windowing_sensitivity_grid_3x3.ps1
window_grid_aug_quality.py
vae_ablation.ps1
vae_ablation_loso.py
Results/model_weights/
Results/synthetic_weights/
Results/loso/
Results/sensitivity_grid_analysis/
Results/privacy_audit/
Results/membership_inference/
Results/distinctness_and_population_coverage/
```

In this audit workspace, the durable locked copy is:

```text
configs/locked/paper_release_reference.yaml
configs/locked/RUNNING_INSTRUCTIONS.verbatim.md
```

## Recommended Release Architecture

Do not mirror the full research folder. Build a curated release:

```text
lrf-imu/
    README.md
    LICENSE
    CITATION.cff
    requirements.txt or environment.yml
    configs/
        paper_6ch_160_40.yaml
        sensitivity_grid.yaml
    src/
        lrf_imu/
            data/
            vae/
            flow/
            evaluation/
            plotting/
    scripts/
        prepare_realdisp.py
        train_vae_loso.py
        train_flow_loso.py
        generate_synthetic_loso.py
        evaluate_tstr_loso.py
        reproduce_paper_summaries.py
    docs/
    tests/
```

Release approach:

- Keep raw REALDISP data outside the repository.
- Make all paths configurable.
- Keep checkpoints and synthetic caches outside Git.
- Provide exact commands to reproduce tables from raw dataset access.
- Include a small smoke-test fixture only if it is synthetic and license safe.

## Safest Order of Work

1. Archive the full original `LRF-IMU-P3` tree outside the public release workflow.
2. Resolve the VAE and flow hyperparameter discrepancies documented in `REPRODUCIBILITY_AUDIT.md`.
3. Decide what result artifacts, if any, are safe to publish.
4. Create a clean source-only package in a new directory or repository.
5. Parameterize all absolute paths.
6. Remove review, manuscript-history, assistant-state, cache, log, checkpoint, and generated-output files.
7. Add environment lock file and dependency license notes.
8. Add smoke tests for windowing, LOSO split integrity, model shape checks, and output schema checks.
9. Run one lightweight reproduction pass on a small local subset or mock fixture.
10. Only then consider full reproduction commands and optional external artifact hosting.

## Agent Notes

- Do not "fix" scientific scripts while doing the first cleanup pass. Preserve behavior first.
- If refactoring begins, write tests around old behavior before changing structure.
- Keep `renaming_paths_vae.md` close at hand; it explains historical path drift.
- Treat any path with `SPECTRAL_LOSS_OFF` as historical naming until verified.
- Treat all files under `Results/` as generated or participant-derived until reviewed.
