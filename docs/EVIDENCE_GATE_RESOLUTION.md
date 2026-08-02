# Milestone 2B Evidence-Gate Resolution

Status: public candidate evidence boundary prepared for independent review. This
document records compatibility decisions; it does not certify exact reproduction
of the paper or completion of the full experiments.

## Pinned authorities

- Public candidate branch: release/v1.0.0-candidate at commit
  8445c9cff1b93e92844b94394258635b5d25fd54.
- Public parent: cc401f8dbe381fe3efe534219e6a912f955279a8.
- Audit inventory: release/audit-and-inventory at
  f38bebce36c4f21d857dc084ac8d06759c2c012d.
- Immutable source: the locked submission-time source tree identified by the
  source-relative paths and hashes in the contracts.
- Investigation evidence is retained by report identifier and sanitized
  source-relative path. Workstation paths are intentionally excluded.

## Decision register

| Area | Public candidate decision | Evidence posture |
| --- | --- | --- |
| Accepted manuscript identity | Keep the candidate manuscript identity as a HOLD until the release owner resolves the accepted-source pin. | Historical source material and audit inventory are not interchangeable. |
| VAE compatibility profile | Record the later wrapper profile: loss weights 0.5 and 0.1; beta 0.08 to 0.04 with decay 0.995. | Compatibility profile, not exact-paper claim. |
| VAE legacy schedule | Preserve the older 1.0 and 0.1 weights and 0.005 to 0.00001 beta schedule with decay 0.7 as unresolved evidence. | HOLD. |
| Rectified-flow width | Use width 256 as the public compatibility value because it is the later configuration-backed profile. | Manuscript width 128 remains unresolved. |
| VAE batch | Resolve to 256. | Configuration-backed. |
| Flow batch | Preserve configured batch 128, automatic selection, and the observed subject-01 effective 512 as compatibility evidence. | The effective batch observation is not a universal rule. |
| Training duration | Keep VAE 1000 maximum, 200 minimum, and patience 100 as the compatibility profile. Keep flow cap 300, direct patience 50, and full-LOSO minimum 20 as separate observations. | Flow duration and patience conflicts remain HOLD where sources disagree. |
| Sampling | Freeze reverse Euler with 10 paper steps, 100 website Euler steps over 500 native samples/windows, interval 1 to 0, and seed 42 for the public fixture boundary. | Paper and website visualization are separate protocols. |
| Three-channel lineage | Outcome is exactly PUBLIC_RECONSTRUCTION_REQUIRED. | The observed six-channel path and intended three-channel path are documented separately. |
| Splits | The safe fixture split is 16/7/8. Keep VAE 0.15 subject validation separate from CNN 0.20 window validation. | Do not retain the temporary 14/7/7 arithmetic concern. |
| Filtering | Preserve filter-before-runs gap bridging as observed compatibility behavior. | This is an observed behavior, not an endorsement of a new scientific implementation. |
| Alias | Keep the 8-versus-7 safe/legacy return-length mismatch as a HOLD. | The public fixture contract returns eight items. |
| Website source hash | Use the independently recomputed digest for 8_export_website_trajectories.py. | The copied fixture-design report contains a two-character transcription error; this is evidence hygiene, not source drift. |

## Evidence tiers

Tier A is directly observed in immutable source or a pinned audit artifact.
Tier B is a cross-checked configuration or wrapper behavior used for public
compatibility. Tier C is a reconstruction or synthetic fixture expectation.
A Tier C fixture never upgrades an unresolved source conflict.

The snapshot set remains 19 total: 14 PASS and 5 HOLD. The HOLDs are the
three-channel projection lineage, the safe eight-item versus legacy alias gate,
the VAE schedule conflict, the flow base-width conflict, and the website-source
hash transcription discrepancy.

Exact paper reproduction remains false. No result, metric, model artifact,
participant record, checkpoint, or full experiment is included in this release
boundary.
