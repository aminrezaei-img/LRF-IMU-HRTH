# Milestone 2B Handoff

## Release boundary

The Milestone 2B candidate is prepared as an uncommitted, mechanically
reviewable patch. It adds evidence resolutions, sanitized contracts,
deterministic fixtures, and pure-Python contract tests. It does not add
scientific implementation under src/lrf_imu, migrate models, copy participant
data, copy checkpoints, or claim exact paper reproduction.

The patch is intended for branch release/v1.0.0-candidate at starting commit
8445c9cff1b93e92844b94394258635b5d25fd54 with parent
cc401f8dbe381fe3efe534219e6a912f955279a8.

## Deliverables

- Evidence-gate, three-channel, fixture-specification, compatibility-profile,
  and handoff documents.
- Four JSON contracts for configuration decisions, channel lineage, parity
  fixtures, and reference behavior snapshots.
- A deterministic synthetic fixture tree with manifest, readme, raw JSON/log
  material, metadata probes, and SHA-256 checksums.
- Contract and fixture-integrity tests with no new dependencies.
- A narrowly scoped scanner exception for the synthetic log fixtures and a
  matching gitignore exception.

## Accepted evidence posture

The snapshot set is 19 total, 14 PASS, and 5 HOLD. The exact three-channel
outcome is PUBLIC_RECONSTRUCTION_REQUIRED. The safe split is 16/7/8, with
subject 01 yielding seven compact windows and subjects 02, 03, and 05 yielding
eight each. The 0.15 VAE subject split and 0.20 CNN window split remain
separate. Filter-before-runs gap bridging and the 8-versus-7 alias mismatch are
preserved as observed compatibility behavior and an unresolved gate,
respectively.

The website exporter digest is recorded as
9F1210C6034695061E83A648F2941CA9F6E0E8A057F547FE320D3F24D967F3EE. The
fixture-design report's two-character transposition is documented as an
evidence-hygiene discrepancy, not source drift.

## Verification boundary

The patch bundle is checked for JSON validity, deterministic fixture hashes,
raw shape and formula integrity, split and standardization invariants,
duplicate-audit probes, website overlap-add behavior, snapshot counts, path
hygiene, forbidden artifacts, and the complete public pytest suite in isolated
temporary locations. The public repository itself remains unchanged until the
orchestrator applies the patch and requests the follow-up acceptance review.

## Suggested commit metadata

Subject: Lock Milestone 2B evidence and synthetic parity boundary

Constraint: Public release may contain contracts and synthetic fixtures only.
Rejected: Migrating scientific implementation or copying source outputs |
would exceed the Milestone 2B evidence boundary.
Confidence: high
Scope-risk: moderate
Reversibility: clean
Directive: Resolve all HOLDs before making an exact-reproduction claim or
training/evaluation claim.
Tested: JSON parsing, contract tests, fixture integrity tests, path scanner,
diff checks, and clean patch application in a disposable verification copy.
Not-tested: Full scientific training and exact manuscript reproduction.
