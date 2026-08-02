"""Focused Milestone 3B tests for the copied public VAE boundary."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType
from typing import Dict

import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from lrf_imu.checkpoints import (  # noqa: E402
    CheckpointError,
    inspect_vae_checkpoint,
    save_vae_checkpoint,
)
from lrf_imu.cli import main as cli_main  # noqa: E402
from lrf_imu.models.vae import LatentVAE1D  # noqa: E402
from lrf_imu.training.vae import (  # noqa: E402
    compute_vae_loss,
    select_vae_profile,
    train_vae,
)


@pytest.mark.parametrize("channels", [6, 3])
def test_public_vae_geometry_and_determinism(channels: int) -> None:
    torch.manual_seed(11)
    model = LatentVAE1D(in_ch=channels, z_ch=48, down_levels=2).cpu().eval()
    inputs = torch.randn(2, channels, 160)

    with torch.no_grad():
        reconstruction_a, mu, logvar = model(inputs, deterministic=True)
        reconstruction_b, _, _ = model(inputs, deterministic=True)
        torch.manual_seed(19)
        stochastic_a, _, _ = model(inputs, deterministic=False)
        torch.manual_seed(19)
        stochastic_b, _, _ = model(inputs, deterministic=False)

    assert tuple(mu.shape) == (2, 48, 40)
    assert tuple(logvar.shape) == (2, 48, 40)
    assert tuple(reconstruction_a.shape) == (2, channels, 160)
    assert torch.equal(reconstruction_a, reconstruction_b)
    assert torch.equal(stochastic_a, stochastic_b)
    assert not torch.equal(stochastic_a, reconstruction_a)


def test_decoder_and_invalid_channels() -> None:
    model = LatentVAE1D(input_channels=6, latent_channels=48).cpu().eval()
    latent = torch.randn(2, 48, 40)
    with torch.no_grad():
        decoded = model.decode(latent)
    assert tuple(decoded.shape) == (2, 6, 160)

    with pytest.raises(ValueError, match="only independent 3-channel or 6-channel"):
        LatentVAE1D(input_channels=4)
    with pytest.raises(ValueError, match="input channel count"):
        model(torch.randn(1, 3, 160), deterministic=True)


def test_loss_matches_copied_equations() -> None:
    x0 = torch.tensor([[[1.0, 2.0]]])
    x_hat = torch.tensor([[[2.0, 0.0]]])
    mu = torch.tensor([[[0.5, -0.5]]])
    logvar = torch.tensor([[[0.1, -0.2]]])
    loss, logs = compute_vae_loss(
        x0, x_hat, mu, logvar, beta=0.2, weights={"l2": 0.5, "l1": 0.1}
    )
    l2 = torch.nn.functional.mse_loss(x_hat, x0)
    l1 = (x_hat - x0).abs().mean()
    kl = 0.5 * torch.mean(torch.exp(logvar) + mu**2 - 1.0 - logvar)
    expected = 0.5 * l2 + 0.1 * l1 + 0.2 * kl
    assert torch.equal(loss, expected)
    assert logs["beta"] == 0.2


def test_checkpoint_root_shape_and_channel_validation(tmp_path: Path) -> None:
    model = LatentVAE1D(in_ch=6, z_ch=48, down_levels=2).cpu()
    checkpoint = tmp_path / "vae_s4_z48.pt"
    save_vae_checkpoint(model, checkpoint)
    inspection = inspect_vae_checkpoint(checkpoint, channels=6)
    assert inspection.channels == 6
    assert len(inspection.state_dict_keys) == 26
    assert inspection.tensor_shapes["enc_in.0.weight"] == (160, 6, 7)
    assert inspection.tensor_shapes["dec_out.weight"] == (6, 160, 1)
    assert inspection.to_mapping()["payload_values_included"] is False

    with pytest.raises(CheckpointError, match="contains 6 channels but 3"):
        inspect_vae_checkpoint(checkpoint, channels=3)

    state = dict(model.state_dict())
    bad_root = tmp_path / "bad-root.pt"
    torch.save({"model": state}, bad_root)
    with pytest.raises(CheckpointError, match=r"exactly \['vae'\]"):
        inspect_vae_checkpoint(bad_root)

    bad_shape = tmp_path / "bad-shape.pt"
    state["enc_in.0.weight"] = torch.zeros(160, 3, 7)
    torch.save({"vae": state}, bad_shape)
    with pytest.raises(CheckpointError, match="enc_in.0.weight"):
        inspect_vae_checkpoint(bad_shape, channels=6)


def test_checkpoint_latent_length_tracks_configured_down_levels(tmp_path: Path) -> None:
    model = LatentVAE1D(in_ch=6, z_ch=48, down_levels=3).cpu()
    checkpoint = tmp_path / "vae_s8_z48.pt"
    save_vae_checkpoint(model, checkpoint)

    inspection = inspect_vae_checkpoint(checkpoint, channels=6, down_levels=3)

    assert inspection.input_length == 160
    assert inspection.latent_stride == 8
    assert inspection.latent_time_steps == 20


def test_cli_reports_missing_optional_torch_without_traceback(monkeypatch, capsys) -> None:
    monkeypatch.setitem(sys.modules, "torch", None)

    exit_code = cli_main(["vae-smoke"])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "optional PyTorch" in captured.err
    assert "lrf-imu[training]" in captured.err
    assert "Traceback" not in captured.err


def test_profile_selection_is_explicit_and_non_reproduction() -> None:
    observed = select_vae_profile()
    older = select_vae_profile("older_manuscript_reported")
    custom = select_vae_profile("custom", overrides={"batch_size": 4, "beta_init": 0.2})

    assert observed.name == "observed_wrapper_compatibility"
    assert (observed.batch_size, observed.beta_init, observed.beta_min) == (256, 0.08, 0.04)
    assert (older.l2_weight, older.beta_init, older.beta_min, older.beta_decay) == (
        1.0,
        0.005,
        0.00001,
        0.7,
    )
    assert older.augmentation.enabled is False
    assert custom.name == "custom"
    assert custom.batch_size == 4
    assert custom.beta_init == 0.2
    assert observed.exact_paper_reproduction is False
    assert older.exact_paper_reproduction is False
    assert custom.exact_paper_reproduction is False


def test_cpu_training_smoke_and_explicit_checkpoint(tmp_path: Path) -> None:
    profile = select_vae_profile(
        "custom",
        overrides={
            "batch_size": 2,
            "max_epochs": 1,
            "early_stop_min_epochs": 1,
            "early_stop_patience": 5,
            "use_amp_bf16": False,
            "augmentation": {"enabled": False, "jitter": 0.0, "scale": 0.0, "time_mask": 0.0},
        },
    )
    train = TensorDataset(torch.randn(4, 3, 160), torch.zeros(4, dtype=torch.long))
    validation = TensorDataset(torch.randn(2, 3, 160), torch.zeros(2, dtype=torch.long))
    checkpoint = tmp_path / "trained" / "vae.pt"
    result = train_vae(
        LatentVAE1D(in_ch=3, z_ch=48, down_levels=2),
        train,
        validation,
        profile=profile,
        checkpoint_path=checkpoint,
        device="cpu",
        pin_memory=False,
        seed=7,
    )
    assert result.device == "cpu"
    assert result.best_epoch == 1
    assert checkpoint.is_file()
    assert inspect_vae_checkpoint(checkpoint, channels=3).channels == 3


def _load_original_vae_module() -> ModuleType:
    source_root_value = os.environ.get("LRF_IMU_VAE_SOURCE_ROOT")
    if not source_root_value:
        pytest.skip("set LRF_IMU_VAE_SOURCE_ROOT for immutable-source parity")
    source_root = Path(source_root_value)
    source_file = source_root / "VAE_logic.py"
    if not source_file.is_file():
        pytest.skip("immutable VAE_logic.py is not available")

    module_name = "lrf_imu_immutable_vae_logic_for_parity"
    spec = importlib.util.spec_from_file_location(module_name, source_file)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not create immutable-source import spec")
    module = importlib.util.module_from_spec(spec)
    previous_path = list(sys.path)
    previous_env = dict(os.environ)
    previous_utils = sys.modules.get("utils")
    sys.path.insert(0, str(source_root))
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = previous_path
        os.environ.clear()
        os.environ.update(previous_env)
        sys.modules.pop(module_name, None)
        if previous_utils is None:
            sys.modules.pop("utils", None)
        else:
            sys.modules["utils"] = previous_utils
    return module


def test_original_public_numeric_parity() -> None:
    original_module = _load_original_vae_module()
    torch.manual_seed(123)
    original = original_module.LatentVAE1D(in_ch=6, z_ch=48, hidden=160, down_levels=2).cpu().eval()
    torch.manual_seed(123)
    public = LatentVAE1D(in_ch=6, z_ch=48, hidden=160, down_levels=2).cpu().eval()
    assert tuple(original.state_dict()) == tuple(public.state_dict())

    state_errors = [
        float((original.state_dict()[key] - public.state_dict()[key]).abs().max())
        for key in original.state_dict()
    ]
    torch.manual_seed(321)
    inputs = torch.randn(2, 6, 160)
    with torch.no_grad():
        original_mean, original_mu, original_logvar = original(inputs, deterministic=True)
        public_mean, public_mu, public_logvar = public(inputs, deterministic=True)
        original_decoded = original.decode(original_mu)
        public_decoded = public.decode(public_mu)
        torch.manual_seed(99)
        original_sample = original.reparam(original_mu, original_logvar)
        torch.manual_seed(99)
        public_sample = public.reparam(public_mu, public_logvar)

    tensors = [
        original_mu - public_mu,
        original_logvar - public_logvar,
        original_mean - public_mean,
        original_decoded - public_decoded,
        original_sample - public_sample,
    ]
    max_errors = [float(error.abs().max()) for error in tensors]
    assert max(state_errors + max_errors) <= 0.0

    report_path_value = os.environ.get("LRF_IMU_VAE_PARITY_REPORT")
    if report_path_value:
        report = {
            "schema_version": "m3b.synthetic-vae-parity.1",
            "license_safe": True,
            "checkpoint_payloads": False,
            "source_relative_path": "<immutable-source>/VAE/VAE_logic.py",
            "source_sha256": "3C989BB8242236D3107AE75A1533622D955D6E739D0B44103636897D01E80505",
            "public_destination": "src/lrf_imu/models/vae.py",
            "input_shape": [2, 6, 160],
            "latent_shape": [2, 48, 40],
            "reconstruction_shape": [2, 6, 160],
            "max_abs_errors": {
                "initialized_state": max(state_errors),
                "posterior_mean": max_errors[0],
                "posterior_logvar": max_errors[1],
                "mean_reconstruction": max_errors[2],
                "decoded_mean": max_errors[3],
                "fixed_seed_sample": max_errors[4],
            },
            "tolerance": 0.0,
            "numeric_parity": True,
        }
        Path(report_path_value).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def _subprocess_environment() -> Dict[str, str]:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(SOURCE_ROOT) + os.pathsep + environment.get("PYTHONPATH", "")
    return environment


def test_cli_help_smoke_inspect_and_reconstruct(tmp_path: Path) -> None:
    environment = _subprocess_environment()
    help_result = subprocess.run(
        [sys.executable, "-B", "-m", "lrf_imu", "--help"],
        cwd=str(tmp_path),
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert help_result.returncode == 0
    assert "vae-smoke" in help_result.stdout
    assert "inspect-vae-checkpoint" in help_result.stdout
    assert "reconstruct" in help_result.stdout

    smoke_result = subprocess.run(
        [sys.executable, "-B", "-m", "lrf_imu", "vae-smoke"],
        cwd=str(tmp_path),
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert smoke_result.returncode == 0, smoke_result.stderr
    smoke = json.loads(smoke_result.stdout)
    assert [entry["channels"] for entry in smoke["results"]] == [6, 3]
    assert all(entry["deterministic_mean_equal"] for entry in smoke["results"])
    assert all(entry["fixed_seed_stochastic_equal"] for entry in smoke["results"])

    checkpoint = tmp_path / "vae.pt"
    save_vae_checkpoint(LatentVAE1D(in_ch=6, z_ch=48, down_levels=2), checkpoint)
    input_path = tmp_path / "input.npy"
    np.save(input_path, np.zeros((1, 6, 160), dtype=np.float32))

    inspect_result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            "lrf_imu",
            "inspect-vae-checkpoint",
            "--checkpoint",
            str(checkpoint),
            "--channels",
            "6",
        ],
        cwd=str(tmp_path),
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert inspect_result.returncode == 0, inspect_result.stderr
    inspection = json.loads(inspect_result.stdout)
    assert inspection["channels"] == 6
    assert inspection["state_dict_key_count"] == 26
    assert inspection["payload_values_included"] is False

    reconstruct_result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            "lrf_imu",
            "reconstruct",
            "--config",
            str(REPOSITORY_ROOT / "configs" / "paper" / "six_channel_160_40.yaml"),
            "--checkpoint",
            str(checkpoint),
            "--input",
            str(input_path),
            "--device",
            "cpu",
        ],
        cwd=str(tmp_path),
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert reconstruct_result.returncode == 0, reconstruct_result.stderr
    reconstruction = json.loads(reconstruct_result.stdout)
    assert reconstruction["reconstruction_shape"] == [1, 6, 160]
    assert reconstruction["deterministic_posterior_mean"] is True
    assert reconstruction["output_written"] is False
