# Release checklist

This checklist describes the local candidate. It does not authorize publication.

## Technical candidate

- [x] Portable preprocessing, VAE, Flow, generation, evaluation, analysis, and thin reproducibility orchestration are present.
- [x] Six-channel and separately trained three-channel model paths are distinct.
- [x] REALDISP, checkpoints, generated arrays, trained evaluators, and historical Results payloads are excluded from Git.
- [x] Package build, wheel/sdist install, foreign-working-directory commands, tests, and repository safety scans passed in Milestone 5.
- [x] CI requires neither REALDISP, historical checkpoints, CUDA, nor private results.
- [x] Paper generation is identified as ten-step reverse Euler; website trajectories are a separate 100-step visualization profile.
- [x] Result claims distinguish exact implementation parity, exact evaluator parity, statistical/runtime agreement, partial reproduction, and blocked work.
- [x] `exact_paper_reproduction=false` remains in force.
- [x] No remote, tag, push, upload, or publication action was performed.

## Human decisions required before public release

- [ ] Confirm code ownership and institutional authority to license the public code.
- [ ] Review copied/adapted third-party code and dependency licence compatibility.
- [ ] Confirm REALDISP terms for the documented workflow and any derived artifacts.
- [ ] Decide whether historical model weights may be released; no checkpoint is included now.
- [ ] Decide whether generated data may be distributed.
- [ ] Resolve manuscript/figure rights and the accepted figure-package identity.
- [ ] Select and approve a code licence, then add a `LICENSE` file.
- [ ] Review final citation and release metadata.
- [ ] Obtain explicit human approval before adding a remote, tag, or publishing.

## Current verdict

**CONDITIONAL GO - CODE TECHNICALLY RELEASE READY, LICENSING DECISION REQUIRED.**

`LICENSE_DECISIONS.md` is a decision record, not a licence grant. Until the
human items above are resolved, the repository is a local candidate only.
