# Production release `paper3-harth-production-v1`

This release freezes the model-only HARTH lineage. It contains no contextual
mapping, cohort outputs, persona data, timeline stitching, or multimodal
fusion artifacts.

## GitHub Release assets

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `vae_s3_z48.pt` | 2,990,434 | `6B118182E14FF04CBD57D66A76986BF3568561F0FD42D02257F8036A0138AAD9` |
| `flow_unet_best.pt` | 185,505,026 | `F1058EB1FB94809B74B8DFFC24E2697F8C73B046CF2A8E7D24FFF60EC6D63164` |
| `harth_walking_speed_data_dad2cfbe89a26f72f19770419469ac037de200df.zip` | 385,975,231 | `AFCAAB04C151E0C182FC85F96DF98762F942280ED26C802E2A6B25BD2FE1C651` |

The ZIP contains 55 CSV files plus the upstream README and MIT licence. The
checkpoint files are the one global pair later reused unchanged by downstream
cohorts; they are not persona-specific.

`SHA256SUMS.txt` accompanies the release for manual verification. The
authoritative machine-readable manifest is packaged with the source at
`src/lrf_imu/resources/manifests/production_harth_v1.json`.

## Scientific identities

- training code: `81b123eab079f5cde7400fee5620e3bffb85a673`;
- accepted generation code: `150b4de6e58365fdda5fc7279192c136d4e8b064`;
- training data: `dad2cfbe89a26f72f19770419469ac037de200df`;
- VAE best epoch: 185 of 285 completed;
- Flow best epoch: 226 of 300 completed;
- held-out subject: `harth:S006`;
- seed: 42.

The full acquisition, verification, preprocessing, training, evaluation, and
native-window generation commands are in the root README.
