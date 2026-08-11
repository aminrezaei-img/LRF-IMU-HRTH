# Release notes

## 1.0.0 — Code-only public release

Version 1.0.0 publishes the validated research-code surface:

- portable REALDISP preprocessing and LOSO orchestration;
- compatible 6CH and separately trained 3CH VAE and Rectified Flow implementations;
- ten-step paper generation and distinct website trajectory export;
- RF/CNN evaluation, paper-relevant analyses, reproducibility commands, tests, CI, and safety scanners; and
- software citation metadata with a preferred citation to *A latent rectified flow approach to generate synthetic wearable data – a LABDA solution*, Machine Learning: Health, DOI 10.1088/3049-477X/ae91ef.

The release is code-only. It excludes REALDISP and participant-derived data, checkpoints, generated datasets, trained evaluators, historical Results payloads, manuscript files, and figures. `exact_paper_reproduction=false` remains unchanged; fold-level differences and unresolved historical lineage remain documented.

## 0.1.0 — Initial audited boundary — 2026-08-01

The initial boundary established portable configuration and path primitives, paper and sensitivity YAMLs, audit/reference records, and provisional public documentation before scientific migration. It deliberately excluded scientific source code and all restricted or large research artifacts.

The pinned audit recorded 4,619 `Results/` files totaling 34,791,553,468 bytes. The later immutable source observation contained 4,667 files and 34,811,675,494 bytes, including 48 later website-trajectory JSON files and four rewritten flow-trajectory images. Those differences remain documented and excluded.
