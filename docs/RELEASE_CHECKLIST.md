# Release checklist

This checklist governs the code-only public release.

## Validated repository

- [x] Portable preprocessing, VAE, Flow, generation, evaluation, analysis, and thin reproducibility orchestration are present.
- [x] Six-channel and separately trained three-channel model paths are distinct.
- [x] REALDISP, checkpoints, generated arrays, trained evaluators, and historical Results payloads are excluded from Git.
- [x] Package build, wheel/sdist install, foreign-working-directory commands, tests, and repository safety scans passed.
- [x] CI requires neither REALDISP, historical checkpoints, CUDA, nor private results.
- [x] Paper generation is ten-step reverse Euler; website trajectories are a separate 100-step visualization profile.
- [x] Result claims distinguish exact implementation parity, exact evaluator parity, statistical/runtime agreement, partial reproduction, and blocked work.
- [x] `exact_paper_reproduction=false` remains in force.

## Publication scope

- [x] The public scope is repository code, configuration, documentation, and tests.
- [x] The release does not include or grant rights to REALDISP, participant data, checkpoints, generated datasets, trained models, historical Results payloads, manuscript files, or figures.
- [x] Historical checkpoints remain external and are not published by this repository.
- [x] Citation metadata identifies software version 1.0.0 and the published paper separately.
- [x] No remote, tag, push, upload, or publication action was performed during this preparation pass.

## Verdict

**GO — CODE-ONLY PUBLIC RELEASE READY**
