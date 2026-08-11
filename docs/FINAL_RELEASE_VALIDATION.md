# Final local release-candidate validation

## Verdict

**CONDITIONAL GO — TECHNICALLY READY, HUMAN DECISION REQUIRED**

The local code candidate passed the software, scientific-smoke, packaging,
and repository-safety gates described below. Public release still requires a
human licensing and rights decision. Historical checkpoints are not
distributed. `exact_paper_reproduction=false` remains in force.

No remote, tag, push, upload, or publication action was performed.

## Candidate identity

Validation began from commit
`32689e113fc3a607dd9de8eb72c7b7ceb3532d5b` on
`release/v1.0.0-candidate`. The disposable clone and authoritative repository
were clean, with 11 commits, zero remotes, and zero tags before this report was
added. The audit repository remained clean at
`f38bebce36c4f21d857dc084ac8d06759c2c012d`. The immutable scientific source
and historical Results tree were read only.

## Scientific status

| Milestone | Final status | Evidence boundary |
| --- | --- | --- |
| 3A preprocessing | PASS | Portable REALDISP parsing/windowing, LOSO split, train-only normalization, 6CH, and explicitly reconstructed separate 3CH path were accepted. |
| 3B VAE | PASS | Public/original operations and historical 6CH/3CH checkpoint execution were numerically identical for the accepted parity inputs. |
| 3C Rectified Flow | PASS | Public/original Flow operations, checkpoint loading, ten-step generation, and one real fold were numerically identical for accepted parity inputs. The width-history conflict remains documented. |
| 3D evaluation | PARTIAL historical-result agreement | All 12 RF folds ran for 6CH and 3CH; all 12 6CH CNN folds ran. Exact evaluator parity was established, while fresh CPU generation differs from historical CUDA RNG streams. Every nonzero fold-level difference remains in the contract. |
| 3E analyses | PASS/PARTIAL/BLOCKED by analysis | Nine-setting sensitivity was exact. VAE-only, physical, PSD, and privacy scopes retain their documented partial limits; 3CH VAE-only remains blocked by missing historical artifacts. |
| 4 orchestration | PASS | Thin prepare/load/generate/evaluate/aggregate/compare orchestration supports dry-run, explicit writes, interruption records, and identity-checked resume. |
| 5 packaging/CI/safety | PASS locally | Package build/install and local CI-equivalent checks passed. GitHub Actions was not run remotely because nothing was pushed. |
| 6 documentation | PASS | Researcher workflow, data/checkpoint boundaries, result levels, discrepancies, and prohibited overclaims are documented. |
| 7 fresh validation | PASS | Fresh-clone installs, tests, smokes, real subject-01 orchestration, resume, and final repository audit passed as qualified below. |

### Regenerated core evaluation evidence

The committed fold-level report is
`contracts/evaluation_parity_report.json`, SHA-256
`9015ce352556560ac8db8f67513671d89a859c8ead8ca8c21191a94e05172682`.
The external M3D record retained 24 checksum manifests (12 6CH and 12 3CH)
and zero generated NPZ arrays at final validation.

| Evaluation | Regenerated macro-F1, mean +/- sample SD | Interpretation |
| --- | ---: | --- |
| 6CH RF TRTR | 0.985060 +/- 0.021056 | Matches the unrounded historical evaluator evidence within stored precision. |
| 6CH RF scarce | 0.400442 +/- 0.087757 | Fold-level comparison retained. |
| 6CH RF TSTR | 0.961957 +/- 0.061189 | Historical 0.955942 +/- 0.081307; PARTIAL because fresh CPU and historical CUDA generation streams differ. |
| 6CH RF TSTR retention | 0.976976 +/- 0.065137 | Fold-wise ratios, not a ratio of aggregate means. |
| 6CH RF TSTR + scarce | 0.959769 +/- 0.061493 | Historical 0.950708 +/- 0.087462; fold-level differences retained. |
| 3CH RF TRTR | 0.980421 +/- 0.026714 | Strong empirical compatibility; not proof of exact historical parser lineage. |
| 3CH RF TSTR | 0.978561 +/- 0.057279 | Historical 0.980042 orientation; fold-level differences retained. |
| 3CH RF TSTR retention | 0.998303 +/- 0.057464 | Separately trained 3CH VAE/Flow checkpoints; no inference-time channel drop. |
| 6CH CNN TRTR | 0.995706 +/- 0.011248 | Statistical/runtime reproduction, not bitwise historical parity. |
| 6CH CNN TSTR | 0.894868 +/- 0.146825 | Historical fold-reference mean 0.850340 +/- 0.167847; stochastic runtime agreement only. |

The corrected CNN run seeds once at each fold boundary and consumes model
initialization RNG in the historical scenario order. Three-channel CNN was
not run. Rounded headline values were not used as a parity gate.

The analysis contract
`contracts/analysis_parity_report.json` has SHA-256
`095b30257bdd99bae5e22505c7ae5f6a0997be2ac57c1e0682ed39088efaa21f`.
It records VAE-only 6CH TSTR 0.448935 +/- 0.177147 versus historical
0.443404 +/- 0.171059 (PARTIAL checkpoint lineage), sensitivity 171/171
cells exact, subject-01 physical 0/320,000 points above 10g, stored-curve
log-PSD correlation 0.966280814, true-holdout MIA 0.495199 +/- 0.020137,
separate post-hoc MIA 0.514989 +/- 0.016021, and 0/240 successful recorded
reconstruction attempts. These observations are not anonymization, privacy,
clinical-validity, or universal-superiority guarantees.

## Fresh install and build

A local clone was made without hard links. Its automatic `origin` was removed.
Validation environments had no `PYTHONPATH`, editable install, repository
working-directory dependency, or inherited system site-packages.

- Base install passed in an empty Python 3.12.4 environment.
- All declared training, evaluation, analysis, test, and development surfaces
  passed with the documented tested runtime: PyTorch 2.7.1+cpu, NumPy 1.26.4,
  scikit-learn 1.4.2, and SciPy 1.12.0.
- `pip check` passed.
- Wheel and sdist built successfully. The wheel was 145,420 bytes with SHA-256
  `244a86826ee761ce311963dbf1453438249ec5a8e61baf6d32d49a14cd2ef64b`;
  the sdist was 162,705 bytes with SHA-256
  `b2a10c5c6d872bf772ac18c2609d47181f2e84179d22bb9f8c6c530cc8414244`.
- Separate empty environments installed the wheel and sdist. Foreign-CWD
  package import, console entry point, `python -m lrf_imu`, and command help
  passed for both.
- The unrestricted `torch>=2.0` resolver selected PyTorch 2.13.0 on this host,
  whose `c10.dll` did not initialize. Revalidation on the repository's
  documented tested PyTorch 2.7.1 CPU runtime passed. This is recorded as a
  host/runtime compatibility limit, not historical environment evidence.

## Executed validation

The full installed-package suite passed **169 tests with 1 optional skip** in
21.45 seconds from a foreign working directory using an external pytest
basetemp. The focused path/secret/artifact scanner passed 5 tests; Ruff passed.

Installed-package commands also verified:

- synthetic `prepare-data`: 16 train, 7 VAE-validation, and 8 held-out windows;
- `vae-smoke`: deterministic 6CH and 3CH `[B,C,160] -> [B,48,40] -> [B,C,160]`;
- `flow-smoke`: conditioned `[1,48,40] -> [1,48,40]`, with the unresolved width
  distinction reported rather than hidden;
- safe 6CH historical VAE and Flow checkpoint key/shape inspection;
- one-fold evaluation dry-run with no write;
- canonical 12-fold `reproduce-core --dry-run` with no data/checkpoint read and
  no output directory;
- foreign-CWD base, wheel, sdist, and full-runtime imports and CLI surfaces.

### Installed-package real subject-01 gate

Only installed public-package code was imported. REALDISP ideal logs and the
historical paired 6CH checkpoints were read externally. The run prepared 2,399
classifier-training, 280 VAE-validation, and 243 held-out windows; loaded the
paired VAE/Flow checkpoints; generated one synthetic window per class using
seed 42 and ten reverse-Euler steps; and completed all four RF scenarios in
76.53 seconds. Its status is deliberately
`protocol_mismatch_descriptive_only`: one sample per class is an orchestration
smoke, not the paper's 500-per-class rerun.

The external manifest/report/result SHA-256 values were respectively
`fa7b4070daac7e88899830a49bf14a0af13130c510f7525af8b9b5ec50d3b918`,
`7c8ee0d93797407de2a58dfE80c35e1d66d57fd3cc49e212873813bb5b095b65`,
and `d83106877648bca86f0f933b1517cc2895284c3c21a436791200c692427f532c`.
An identical `--resume` invocation recorded `skipped_completed`. The temporary
generated NPZ was deleted after its checksum
`509ddb58aec012314e10f220f76b1fde77a5e41b67bdc2666bc3266d76732d20`
was retained in the manifest.

## Final repository audit

The clean candidate contained 157 tracked files. The largest tracked file was
680,476 bytes; no tracked file exceeded 5 MiB. There were zero tracked
checkpoint/generated-array/database/pickle payload extensions and zero
tracked `Results`, checkpoint, generated, cache, build, or distribution
payload directories. `git count-objects -vH` reported 367 loose objects,
931.18 KiB, zero packs, and zero garbage. Git integrity, worktree, index,
path/secret/artifact scanning, and ignored-residue checks passed. The
authoritative repository retained zero remotes and zero tags.

## Human decisions and remaining limits

Before publication, a human must resolve code ownership and institutional
licensing authority, third-party compatibility, REALDISP terms, model-weight
and generated-data release rights, and manuscript/figure rights; select a
licence; review final citation/release metadata; and explicitly authorize any
remote, tag, upload, or publication. Historical checkpoints remain external.
The VAE schedule disagreement, Flow width history, partial table provenance,
CUDA-versus-CPU generation-stream difference, reconstructed 3CH parser
lineage, and exact accepted-manuscript artifact provenance remain unresolved.
They continue to prevent an exact-paper claim, so
`exact_paper_reproduction=false` remains unchanged.

The final local verdict is therefore **CONDITIONAL GO — TECHNICALLY READY,
HUMAN DECISION REQUIRED**. This verdict authorizes no publication action.
