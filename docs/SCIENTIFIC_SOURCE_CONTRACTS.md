# Scientific source contracts

## Purpose and boundary

These contracts translate the Milestone 2A read-only characterization into
portable, reviewable statements. They are not scientific implementation code,
do not contain raw data or checkpoints, and do not establish exact paper
reproduction.

The public repository records what is safe to carry forward as a contract while
preserving what remains uncertain. A contract can be stable for compatibility
without being authoritative for the accepted manuscript.

## Evidence tiers

1. **Locked audit** — the pinned audit branch and its release manifest, locked
   YAML, run guide, inventory, provenance, and risk documents.
2. **Observed immutable source** — direct inspection of source modules, wrappers,
   documentation, and selected hashes in `<immutable-source>`.
3. **Safe synthetic probe** — dummy logs, arrays, and tensors executed in an
   external temporary directory with bytecode disabled.
4. **Checkpoint schema only** — weights-only inspection of root keys, metadata,
   and tensor shapes. It cannot identify training settings that were not saved.
5. **Manuscript variants** — separate manuscript-era statements; no variant is
   silently promoted to paper authority.
6. **Current tested runtime** — the current CPU environment used for probes; it
   is not a historical lock.
7. **Public compatibility config** — Milestone 1's portable YAML values, which
   intentionally carry `exact_paper_reproduction: false`.

## Contract index

| Contract | Main responsibility | Current status |
| --- | --- | --- |
| `source_inventory.json` | Source topology, callers, duplicates, patched and historical candidates | Characterized; migration remains gated |
| `data_preprocessing_contract.json` | REALDISP schema, four-class path, windows, splits, normalization, leakage audit | 6CH observed; 3CH and alias path blocked |
| `vae_contract.json` | LatentVAE1D geometry, schedules, checkpoint shapes, training behavior | Geometry stable; schedule and 3CH lineage unresolved |
| `rectified_flow_contract.json` | Flow equations, U-Net shape evidence, paper sampler, website exporter | Sampler stable; base width unresolved |
| `evaluation_contract.json` | Scenarios, RF/CNN settings, metrics, confusion aggregation, provenance | Core path characterized; selected tables partial/blocked |
| `runtime_contract.json` | Current probe versions, historical claim status, portability and serialization risks | Current probe only; no historical lock |

## Cross-contract invariants

The six JSON contracts intentionally repeat a small set of values so each file
can be reviewed independently. These invariants must remain synchronized:

| Field | Required value |
| --- | --- |
| Subjects | `1, 2, 3, 5, 8, 9, 10, 11, 12, 13, 14, 16` |
| Class labels | `0 walking`, `1 running`, `2 jump_up`, `3 cycling` |
| Main window/hop | `160/40` samples |
| Sampling rate | `50 Hz` |
| VAE latent | `[B,48,40]` from `[B,C,160]` |
| Paper Flow sampler | reverse Euler, `10` steps, seed `42` |
| Synthetic count | `500` per class when not reduced by cache metadata |
| Primary metric | macro-F1 over labels `0..3` |
| Fold SD | sample SD, `ddof=1` |

The `0.15` VAE-safe subject split and `0.20` CNN internal split are not an
invariant that may be merged. They belong to different stages.

## Evidence classifications

- `canonical_observed` means the behavior is directly supported by the current
  inspected source or locked compatibility evidence.
- `wrapper` means orchestration is observed, but the wrapper does not prove the
  underlying scientific provenance.
- `patched_current` means a later/current patched analysis candidate was selected
  by observed run-guide or caller evidence; it is not a universal canonical.
- `historical_or_superseded` means the file or name remains useful for lineage,
  not that it should be executed or migrated.
- `ambiguous` means multiple live or documented candidates remain.
- `blocked` means a required evidence, safety, dependency, or parity gate is
  missing.

## Do not over-read checkpoint evidence

The VAE checkpoint root key is `vae`; the Flow checkpoint includes `unet` among
its root keys. The observed VAE state shapes support separate 6CH and 3CH
models. The Flow shapes support the later/wrapper width profile. None of these
facts proves the schedule, optimizer history, source revision, data split, or
accepted-manuscript identity. Checkpoint binaries therefore remain outside
public Git.

## Migration interpretation

The category in `docs/MIGRATION_PLAN.md` is a decision boundary:

- `safe-copy` is limited to a small, evidence-stable contract or utility.
- `wrap` preserves an existing boundary while making roots and invocation
  explicit.
- `minimal-edit` is reserved for configuration/path/portability repairs that
  leave the scientific algorithm unchanged.
- `rewrite-with-parity-tests` is required for monoliths or new public execution
  surfaces whose behavior must be locked by synthetic fixtures.
- `blocked` means no source migration is authorized by this characterization.

No category authorizes copying participant data, checkpoints, results, logs,
manuscript history, or scientific modules into `src/lrf_imu`.
