# Public Release Risks

## Summary

The inspected project contains publishable research code mixed with raw-data paths, large checkpoints, generated participant-derived artifacts, review materials, local assistant state, and historical scripts. A public repository should be curated rather than mirroring the full tree.

## Data and Participant Risks

Raw REALDISP logs:

- Local path: `D:/PAPER2/SALVAGED_PARTS/ideal_logs`.
- These files are outside the source tree but are referenced by scripts.
- Do not redistribute raw logs in the public repository.
- Provide dataset acquisition instructions and preprocessing commands instead.

Participant-derived generated files:

- Synthetic caches under `Results/synthetic_weights`.
- Checkpoints under `Results/model_weights`.
- Privacy audit artifacts under `Results/privacy_audit`.
- Membership inference artifacts under `Results/membership_inference`.
- Large `.npz` samples under analysis folders.

Risk:

- Even if not raw data, these may encode participant-derived movement patterns.
- Public release requires a privacy and license decision before sharing any derived data or checkpoints.

## Private or Identifiable Material

Potentially sensitive areas:

| Path or pattern | Risk |
| --- | --- |
| `manuscript/` | Author details, review history, funding, acknowledgements, and unreleased manuscript versions. |
| `reviewer*`, `Response_to_Reviewers*`, `Author_Response*` | Peer-review material and author identity context. |
| `.claude/` | Local assistant state and workflow traces. |
| `Results/**/*.log` | Absolute paths, machine context, and run metadata. |
| Files containing `D:/PAPER2`, `D:\PAPER2`, `C:/Users`, or `AminR` | Local identity and workstation paths. |

Required action before release:

- Search for absolute local paths and author names.
- Remove peer-review documents.
- Replace local path defaults with CLI arguments or documented placeholders.

## Credentials and Secrets

No credential file was specifically identified during this audit, but no full secret scan was completed. Before public release:

- Run a secret scanner over the curated release tree.
- Check `.env`, notebook outputs, logs, hidden directories, and metadata files.
- Do not publish local assistant or editor state.

## Large Binary Risks

Large generated artifacts:

- `Results/model_weights`: approximately 5.83 GB in this inspected tree subset.
- `Results/sensitivity_grid_analysis`: approximately 24.10 GB.
- `Results/membership_inference`: approximately 2.91 GB.
- All `.pt` files together: approximately 31.51 GB.
- All `.npz` files together: approximately 993 MB.

Risk:

- These should not enter Git.
- If any are shared, use release assets, Zenodo, OSF, institutional storage, or a model/data repository with an explicit privacy and license note.

## Dataset Redistribution Restrictions

REALDISP should be treated as an external dataset:

- Do not include raw REALDISP logs.
- Do not include preprocessed raw windows unless redistribution terms permit it.
- Public code should document how users can obtain the dataset and reproduce the preprocessing locally.
- Include citation and dataset access instructions in the final release README.

## Third-Party Code and Licenses

Before release:

- Check all imported dependencies and their licenses.
- Confirm whether any code was copied from third-party repositories or notebooks.
- Add a project license only after confirming compatibility with dependencies, dataset terms, and institutional policy.
- Include attribution for REALDISP and all cited datasets or baselines.

## Internal SENS, LABDA, and Review Material

The project includes internal context from the manuscript and review process. Do not publish:

- Reviewer response drafts.
- Submission checklists.
- Marked manuscript PDFs.
- Funding placeholders or internal programme names unless already public and approved.
- Any private SENS or LABDA operational material.

## Files That Must Not Enter a Public Repository

Strong exclude candidates:

```text
.claude/
__pycache__/
*.pyc
*.pkl
*.pt
*.npz
*.log
Results/
manuscript/
reviewer*
Response_to_Reviewers*
Author_Response*
DRAFT.IPYNB
```

Conditional include candidates:

- Curated summary CSV/TEX files from `Results/` only if they contain no restricted data, no author-sensitive paths, and no participant-identifiable material.
- Selected final figures only if generated from release-safe summaries and checked for metadata.
- Checkpoints only as separate non-Git artifacts after privacy review.

## Release-Safe Architecture

Recommended public tree:

```text
lrf-imu/
    README.md
    LICENSE
    CITATION.cff
    requirements.txt or environment.yml
    configs/
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
        reproduce_paper_tables.py
    docs/
    tests/
```

Do not create this architecture until the audit discrepancies are resolved and immutable artifacts are preserved.

## Required Pre-Release Checks

- Full path and identity scan.
- Secret scan.
- License review.
- Dataset redistribution review.
- Large-file policy decision.
- Metadata scan for PDFs, DOCX, notebooks, and images.
- Reproduction smoke test from a clean checkout using only documented paths.
