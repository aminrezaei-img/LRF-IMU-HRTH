# Milestone 2A migration plan

## Gate

Milestone 2A authorizes characterization only. The public repository remains a
configuration, contract, and path boundary. Scientific migration is a later
milestone and is gated by the evidence and synthetic parity tests below.

No migration step may add scientific modules to `src/lrf_imu` without an
explicitly reviewed follow-up scope. No step may add raw data, checkpoints,
results, logs, synthetic caches, participant artifacts, or manuscript history.

## Decision categories

| Category | Meaning |
| --- | --- |
| `safe-copy` | Copy a small evidence-stable contract/utility only after artifact and path review. |
| `wrap` | Preserve an observed boundary while parameterizing roots, device, and outputs. |
| `minimal-edit` | Repair portability/configuration without changing the scientific algorithm. |
| `rewrite-with-parity-tests` | Build a new public execution boundary only after synthetic regression tests lock behavior. |
| `blocked` | Do not migrate until the named evidence or safety gate is resolved. |

## Component matrix

| Component | Category | Prerequisites | Required tests |
| --- | --- | --- | --- |
| REALDISP discovery and six-channel schema | `safe-copy` | Keep raw data external; freeze file/column contract | synthetic 120-column fixture; nonrecursive discovery; dtype/shape checks |
| Four-class filter and run construction | `safe-copy` | Label as paper-specific; document filter-before-runs caveat | excluded-label gap fixture; short-run and label-map tests |
| Windowing 160/40 and 80/160/240 grid | `safe-copy` | Preserve complete-window/no-padding behavior | boundary, hop, short-run, and grid-count tests |
| Z-score standardization | `minimal-edit` | Record fit subjects, axes, ddof, and minimum std | leakage and fit/apply fixture; inverse-transform test |
| VAE-safe subject split | `wrap` | Repair return-count/alias mismatch; retain 0.15 | subject-disjoint split; held-out exclusion; metadata test |
| Legacy standard split | `blocked` | Resolve seven-versus-eight return contract | caller and return-shape regression tests |
| 6CH VAE geometry | `safe-copy` | Keep `[B,C,160] -> [B,48,40]` schema | deterministic/stochastic and state-shape tests |
| VAE schedule and beta profile | `blocked` | Resolve manuscript/wrapper/runtime evidence | explicit profile-selection test; provenance record |
| VAE 3CH path | `blocked` | Recover exact preprocessing/training lineage | independent 3CH fixture; 3CH checkpoint load; no channel-drop test |
| VAE training module | `rewrite-with-parity-tests` | Split import-safe model, data, and orchestration boundaries | dummy forward/loss; checkpoint selection; CPU and device tests |
| Flow equations and sampler | `safe-copy` | Freeze latent, labels, seed, steps, and coordinate system | interpolation, target, reverse Euler, seeded dummy sampler |
| Flow U-Net | `blocked` | Resolve C=128 versus C=256 and checkpoint authority | width-specific state-dict and forward tests |
| Flow trainer duplicate | `blocked` | Audit callers; select source of truth or alias | importer graph; byte-equivalence; one-batch training smoke test |
| Flow/generation wrappers | `wrap` | Parameterize roots and artifact policy | dry-run command graph; exit-code and path tests |
| Generation entrypoint | `minimal-edit` | Separate paper sampler from website exporter | dummy checkpoint schema; count/seed; standardized output tests |
| Website trajectory exporter | `safe-copy` | Keep website-only label and outputs external | 100-step record and linear overlap-add fixture |
| Core TSTR evaluator | `wrap` | Freeze evaluation scenarios and split semantics | dummy RF/CNN; metric and scenario dispatch tests |
| Patched four-scenario evaluator | `wrap` | Preserve distinction from core TSTR and provenance | scarce cap, augmentation, and fold aggregation tests |
| CNN implementation | `wrap` | Keep internal validation 0.20 separate from VAE 0.15 | architecture, dropout, optimizer, and split tests |
| Metrics/confusion summaries | `minimal-edit` | Explicitly name sample SD and cm_count semantics | macro-F1, retention, nanmean, and cm_count fixtures |
| VAE-only ablation | `wrap` | Recover final table provenance; keep flow_used=false | latent Gaussian and no-flow path tests |
| Optional analysis (`tslearn`) | `blocked` | Lock optional dependency and data boundary | dependency gate; dummy analysis fixture |
| Runtime/dependency declaration | `rewrite-with-parity-tests` | Produce a clean current lock, not a historical claim | fresh-environment install; import and CPU dummy suite |
| Serialization/loading | `minimal-edit` | Define trusted checkpoint policy; no public binaries | weights-only schema; strictness; unsafe fallback rejection |
| PowerShell orchestration | `wrap` | Fix undefined `supplementWin240`; portable roots | parse/dry-run; failure propagation; artifact allowlist |

## Order of work for the next milestone

1. Resolve the evidence gates: Flow width, VAE schedule, 3CH lineage, safe split
   interface, and final evaluation provenance.
2. Create synthetic-only fixtures for data, VAE geometry, Flow sampling, and
   evaluation metrics; do not use participant data.
3. Lock a current runtime and optional dependency policy. Treat the historical
   Python/CUDA/GPU statement as documentary until independently reproduced.
4. Implement the smallest portable boundary, beginning with contracts and thin
   wrappers; keep scientific modules out of the public tree until approved.
5. Run parity tests and review artifact scans before any scientific source or
   checkpoint decision.

## Exit criteria

The next migration milestone is not complete until:

- all blocked decisions have written evidence or remain explicitly blocked;
- the public tree contains no prohibited artifact or machine-specific path;
- synthetic parity tests pass under a clean, documented runtime;
- no exact-paper or full-rerun claim is made without direct provenance; and
- the audit and immutable source remain unchanged.

## Milestone 3A status

Milestone 3A implements the seven data-preparation lanes that were authorized by
the synthetic parity boundary:

1. activities and raw/schema validation;
2. direct-child REALDISP loading;
3. activity-bounded 160/40 windowing with filter-before-runs default and strict
   contiguity opt-in;
4. VAE subject-level and separate CNN window-level split interfaces;
5. training-only population-standard-deviation normalization;
6. SHA-1 exact-window duplicate auditing across public split pairs; and
7. a metadata-only pipeline, package exports, and CLI safety boundary.

These are public contract implementations backed by synthetic fixtures; they are
not copies of participant-derived scientific modules. Historical compatibility is
preserved where useful, while public safety intentionally requires explicit roots,
separate 3CH configuration, no inference-time channel dropping, no implicit
writes, and no serialized participant arrays. The public duplicate audit covers
every split pair by default; the train/validation-only adapter is retained only
as a labeled historical compatibility mode.

The 3CH route is PUBLIC_RECONSTRUCTION_REQUIRED: it is a separate accelerometer
schema for fresh downstream training and does not establish historical checkpoint
or VAE lineage. VAE, Flow, model, generation, and evaluation migration remains
blocked and is outside M3A; Milestone 3B is not started by this handoff.
### M3A correction constraints

The M3A runtime dependency gate is satisfied by a current unpinned lower bound,
numpy>=1.20, declared consistently with PyYAML in pyproject.toml and
requirements.txt. It is not a historical environment reconstruction.
The default configuration is package data with a byte-synchronization test;
the root paper YAMLs remain evidence and examples.

The split contract names the VAE subject validation fraction as 0.15 and the
classifier/window fraction as 0.20. The legacy 0.20 key remains an explicit
classifier/window alias. Any future migration must preserve these protocols
separately and must not infer VAE validation from the generic historical key.