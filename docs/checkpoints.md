# Checkpoints

Production checkpoints are external artifacts. They are not added to normal
Git history by this release.

## VAE

Expected filename: `vae_s3_z48.pt`
SHA-256:

```text
6B118182E14FF04CBD57D66A76986BF3568561F0FD42D02257F8036A0138AAD9
```

## Rectified Flow

Expected filename: `flow_unet_best.pt`
SHA-256:

```text
F1058EB1FB94809B74B8DFFC24E2697F8C73B046CF2A8E7D24FFF60EC6D63164
```

## Verification

PowerShell:

```powershell
Get-FileHash -Algorithm SHA256 <vae-checkpoint>
Get-FileHash -Algorithm SHA256 <flow-checkpoint>
```

Bash:

```bash
sha256sum <vae-checkpoint> <flow-checkpoint>
```

The VAE and Flow must use the same declared channel configuration and latent
geometry. The Paper 3 production pair is three-channel, `[48,40]` latent
geometry, and ten classes. Do not silently pair these files with the earlier
REALDISP six-channel checkpoints.

No public Paper 3 checkpoint download URL is asserted here. Future artifact
publication can use a research repository or release asset while preserving
these hashes.
