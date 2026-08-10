"""Focused synthetic Milestone 3C tests for the public flow boundary."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch
from torch import nn


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from lrf_imu.checkpoints import (  # noqa: E402
    CheckpointError,
    inspect_flow_checkpoint,
    save_flow_checkpoint,
)
from lrf_imu.generation.flow import (  # noqa: E402
    build_website_trajectory_payload,
    linear_overlap_add,
    paper_sampling_metadata,
    sample_reverse_euler,
    website_sampling_metadata,
    website_seed,
)
from lrf_imu.models.flow import (  # noqa: E402
    LatentDiffusionUNet1D,
    select_flow_profile,
)
from lrf_imu.models.vae import LatentVAE1D  # noqa: E402
from lrf_imu.training.flow import (  # noqa: E402
    RectifiedFlowIMU,
    compute_flow_matching_loss,
    interpolate_latents,
    make_flow_matching_batch,
    reverse_euler_step,
)


@pytest.mark.parametrize("width", [128, 256])
def test_flow_unet_latent_geometry_and_architecture(width: int) -> None:
    torch.manual_seed(3)
    model = LatentDiffusionUNet1D(model_ch=width).eval()
    assert model.model_ch == width
    assert model.label_embed.num_embeddings == 4
    assert model.downs[0].downsample.__class__.__name__ == "AvgPool1d"
    assert model.ups[0].upsample.mode == "nearest"
    assert model.bottle_resblock.bigconv.kernel_size == (31,)
    assert model.bottle_resblock.bigconv.groups == width * 4
    inputs = torch.randn(1, 48, 40)
    with torch.no_grad():
        output = model(inputs, torch.tensor([250.0]), torch.tensor([0]))
    assert tuple(output.shape) == (1, 48, 40)


def test_conditioning_interpolation_velocity_and_loss() -> None:
    z0 = torch.ones(1, 2, 2)
    z1 = torch.full_like(z0, 5.0)
    t = torch.tensor([0.25])
    zt = interpolate_latents(z0, z1, t)
    assert torch.equal(zt, torch.full_like(zt, 2.0))
    batch = make_flow_matching_batch(z0, t=t, z1=z1)
    assert torch.equal(batch.target, torch.full_like(batch.target, 4.0))
    predicted = torch.ones_like(batch.target)
    assert torch.equal(compute_flow_matching_loss(predicted, batch.target), torch.tensor(9.0))
    assert torch.equal(reverse_euler_step(z1, batch.target, 0.5), torch.full_like(z1, 3.0))


def test_rectified_flow_uses_vae_posterior_mean_and_model_time() -> None:
    vae = LatentVAE1D(in_ch=3, z_ch=48, down_levels=2).eval()
    flow = LatentDiffusionUNet1D(model_ch=32, num_classes=4)
    trainer = RectifiedFlowIMU(vae, flow, device="cpu")
    x0 = torch.randn(1, 3, 160)
    z1 = torch.zeros(1, 48, 40)
    loss, batch = trainer.training_loss(
        x0,
        torch.tensor([1]),
        t=torch.tensor([0.25]),
        z1=z1,
        return_batch=True,
    )
    assert tuple(batch.z0.shape) == (1, 48, 40)
    assert torch.equal(batch.zt, batch.z0 * 0.75)
    assert loss.ndim == 0


class _ZeroVelocity(nn.Module):
    def forward(self, z: torch.Tensor, t: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        del t, labels
        return torch.zeros_like(z)


def test_ten_step_sampling_is_deterministic_and_starts_at_noise() -> None:
    model = _ZeroVelocity()
    labels = torch.tensor([0, 3])
    first = sample_reverse_euler(model, labels, (2, 2), num_steps=10, seed=42)
    second = sample_reverse_euler(model, labels, (2, 2), num_steps=10, seed=42)
    assert torch.equal(first, second)
    assert tuple(first.shape) == (2, 2, 2)
    assert torch.equal(first, sample_reverse_euler(model, labels, (2, 2), num_steps=10, seed=42))


def test_flow_checkpoint_schema_width_channel_and_class_validation(tmp_path: Path) -> None:
    model = LatentDiffusionUNet1D(model_ch=32, num_classes=4)
    checkpoint = tmp_path / "flow.pt"
    save_flow_checkpoint(model, checkpoint, channels=6, epoch=2, history={"val_loss": [1.0]})
    inspection = inspect_flow_checkpoint(checkpoint, channels=6, width_profile="custom", model_ch=32)
    assert inspection.channels == 6
    assert inspection.latent_channels == 48
    assert inspection.num_classes == 4
    assert inspection.model_width == 32
    assert inspection.to_mapping()["payload_values_included"] is False
    assert set(inspection.root_keys) == {"config", "epoch", "history", "opt", "unet", "val_loss"}

    with pytest.raises(CheckpointError, match="contains 6 channels but 3"):
        inspect_flow_checkpoint(checkpoint, channels=3)
    with pytest.raises(CheckpointError, match="latent channels"):
        inspect_flow_checkpoint(checkpoint, latent_channels=32)
    with pytest.raises(CheckpointError, match="class count"):
        inspect_flow_checkpoint(checkpoint, num_classes=3)
    with pytest.raises(CheckpointError, match="model width"):
        inspect_flow_checkpoint(checkpoint, model_ch=64)

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    payload.pop("opt")
    bad_root = tmp_path / "bad-root.pt"
    torch.save(payload, bad_root)
    with pytest.raises(CheckpointError, match="root keys"):
        inspect_flow_checkpoint(bad_root)


def test_paper_and_website_profiles_are_distinct_and_seeded() -> None:
    paper = paper_sampling_metadata()
    website = website_sampling_metadata(base_seed=42, subject_id=2, activity_id=1)
    assert paper["num_steps"] == 10
    assert paper["samples_per_class"] == 500
    assert paper["website_trajectory"] is False
    assert website["num_steps"] == 100
    assert website["record_every"] == 2
    assert website["state_count"] == 51
    assert website["native_segment_count"] == 4
    assert website["seed"] == 2142
    assert website["paper_tstr_separation"] is True
    assert website_seed(42, 2, 1) == 2142

    windows = np.asarray([[[0, 1, 2, 3]], [[10, 11, 12, 13]]], dtype=np.float32)
    joined = linear_overlap_add(windows, target_samples=6, overlap_samples=2)
    assert np.array_equal(joined, np.asarray([[0, 1, 2, 7, 12, 13]], dtype=np.float32))

    native = np.zeros((51, 4, 1, 160), dtype=np.float32)
    payload = build_website_trajectory_payload(native, subject_id=2, activity_id=1)
    assert payload["generation"]["state_count"] == 51
    assert payload["provenance"]["paper_tstr_samples"] is False
    assert len(payload["signals"]) == 51


def test_flow_width_profiles_are_explicit() -> None:
    observed = select_flow_profile()
    manuscript = select_flow_profile("manuscript_reported_128")
    custom = select_flow_profile("custom", overrides={"model_ch": 32})
    assert observed.name == "historical_checkpoint_compatibility_256"
    assert observed.model_ch == 256
    assert manuscript.name == "manuscript_reported_128"
    assert manuscript.model_ch == 128
    assert custom.name == "custom"
    assert custom.model_ch == 32
    assert observed.exact_paper_reproduction is False
    with pytest.raises(ValueError, match="requires model width 256"):
        select_flow_profile("historical_checkpoint_compatibility_256", model_ch=128)


def test_cli_help_smoke_and_torch_free_core_import(tmp_path: Path) -> None:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(SOURCE_ROOT) + os.pathsep + environment.get("PYTHONPATH", "")
    python = sys.executable
    help_result = subprocess.run(
        [python, "-B", "-m", "lrf_imu", "--help"],
        cwd=str(tmp_path),
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert help_result.returncode == 0
    for command in ("flow-smoke", "inspect-flow-checkpoint", "generate", "export-trajectories"):
        assert command in help_result.stdout

    smoke_result = subprocess.run(
        [python, "-B", "-m", "lrf_imu", "flow-smoke", "--width-profile", "manuscript_reported_128"],
        cwd=str(tmp_path),
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert smoke_result.returncode == 0, smoke_result.stderr
    smoke = json.loads(smoke_result.stdout)
    assert smoke["output_shape"] == [1, 48, 40]
    assert smoke["tensor_values_included"] is False

    core_result = subprocess.run(
        [python, "-B", "-c", "import sys; import lrf_imu; assert 'torch' not in sys.modules"],
        cwd=str(tmp_path),
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert core_result.returncode == 0, core_result.stderr


def test_cli_generate_json_serializes_final_output_metadata(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from lrf_imu.cli import main

    checkpoint = tmp_path / "flow.pt"
    save_flow_checkpoint(
        LatentDiffusionUNet1D(model_ch=32),
        checkpoint,
        channels=6,
        config={"width_profile": "custom"},
    )
    output = tmp_path / "generate.json"

    exit_code = main(
        [
            "generate",
            "--flow-checkpoint",
            str(checkpoint),
            "--latent-only",
            "--channels",
            "6",
            "--latent-channels",
            "48",
            "--classes",
            "4",
            "--width-profile",
            "custom",
            "--model-ch",
            "32",
            "--class-id",
            "0",
            "--count",
            "1",
            "--steps",
            "1",
            "--seed",
            "42",
            "--device",
            "cpu",
            "--output",
            str(output),
        ]
    )
    assert exit_code == 0
    stdout_payload = json.loads(capsys.readouterr().out)
    disk_payload = json.loads(output.read_text(encoding="utf-8"))
    expected_path = str(output.resolve())
    assert disk_payload["output_written"] is True
    assert disk_payload["output_path"] == expected_path
    assert stdout_payload["output_written"] is True
    assert stdout_payload["output_path"] == expected_path

def test_cli_missing_torch_returns_code_two_without_traceback(monkeypatch, capsys) -> None:
    monkeypatch.setitem(sys.modules, "torch", None)
    from lrf_imu.cli import main

    exit_code = main(["flow-smoke"])
    captured = capsys.readouterr()
    assert exit_code == 2
    assert "optional PyTorch" in captured.err
    assert "Traceback" not in captured.err
