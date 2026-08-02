# Three-Channel Lineage

The exact public outcome for the three-channel question is
PUBLIC_RECONSTRUCTION_REQUIRED. The repository must not silently claim that
three-channel training, checkpoint lineage, and evaluation were recovered from
the submission-time artifacts.

## Observed paths

| Stage | Six-channel evidence | Three-channel interpretation |
| --- | --- | --- |
| Input selection | The observed parser path reads right-thigh columns 80 through 85. | The intended three-channel contract uses columns 80, 81, and 82 only. |
| VAE input | The checkpoint-facing schema is shaped as [batch, 6, 160]. | A separate reconstruction target is [batch, 3, 160]; it is not produced by an inference-time drop from six channels. |
| VAE latent | The public shape contract is [batch, 48, 40]. | The latent shape is unchanged for the synthetic three-channel probe, but the weights and training lineage are not implied to match. |
| Decoder | The observed six-channel decoder shape is [6, 160, 1]. | The three-channel decoder shape is [3, 160, 1]. |
| Classifier/evaluation | Flow, TSTR, and aggregate paths do not provide one consistent three-channel lineage. | Public reconstruction and fresh validation are required before a claim can be made. |

The observed parser also ignores the ABLATION_ACC_ONLY namespace in the relevant
source path. The wrapper selects the three-channel namespace for one public
compatibility route, but that selection does not establish that the associated
checkpoint was trained through the same path.

## Contract boundary

The public contract records:

- six-channel observed indices 80 through 85;
- three-channel intended indices 80, 81, and 82;
- explicit separation of the two schemas;
- no implicit channel drop at inference time;
- checkpoint input and decoder shape gates;
- a fresh three-channel reconstruction and evaluation gate.

A shape-compatible synthetic probe is useful for testing serialization and
interface assumptions. It is not participant data, a checkpoint, or evidence of
scientific parity. The public release therefore keeps the lineage outcome
blocked and marks the associated snapshot HOLD.
