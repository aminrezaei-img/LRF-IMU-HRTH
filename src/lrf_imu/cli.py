"""Command-line entry points for the public LRF-IMU package."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
import numpy as np
from pathlib import Path
import sys
from typing import Any, Optional, Sequence

from .data.pipeline import (
    METADATA_FILENAME,
    PreparedData,
    PreparationError,
    prepare_data,
    write_metadata_summary,
)


def _positive_int(value: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if result <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def _non_negative_int(value: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a non-negative integer") from exc
    if result < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return result


def _channel_count(value: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("channels must be 3 or 6") from exc
    if result not in (3, 6):
        raise argparse.ArgumentTypeError("channels must be 3 or 6")
    return result


def build_parser() -> argparse.ArgumentParser:
    """Build the parser without reading configuration or touching the filesystem."""

    parser = argparse.ArgumentParser(
        prog="lrf-imu",
        description=(
            "Portable, metadata-only orchestration for the paper-specific "
            "REALDISP preparation boundary."
        ),
    )
    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")
    prepare = subparsers.add_parser(
        "prepare-data",
        help="discover, window, split, normalize, and audit one fold in memory",
        description=(
            "Prepare one explicit REALDISP LOSO fold in memory. Participant-derived "
            "windows are never written; metadata writing requires both --output-root "
            "and --write-metadata."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    prepare.add_argument(
        "--config",
        metavar="PATH",
        help="YAML configuration path; defaults to the package six-channel profile",
    )
    prepare.add_argument(
        "--data-root",
        metavar="PATH",
        help="explicit external directory containing direct-child subject*_ideal.log files",
    )
    prepare.add_argument(
        "--output-root",
        metavar="PATH",
        help="explicit directory for the single metadata JSON artifact",
    )
    prepare.add_argument(
        "--held-out-subject",
        type=_positive_int,
        metavar="ID",
        help="LOSO test subject ID; defaults to the configured subject or first discovered subject",
    )
    prepare.add_argument(
        "--sensor-configuration",
        metavar="MODE_OR_YAML",
        help="six_channel, three_channel/accelerometer_only, or a YAML configuration path",
    )
    prepare.add_argument(
        "--seed",
        type=_non_negative_int,
        metavar="N",
        help="deterministic VAE-safe subject split seed",
    )
    prepare.add_argument(
        "--window-length",
        type=_positive_int,
        metavar="SAMPLES",
        help="complete window length in samples",
    )
    prepare.add_argument(
        "--hop-length",
        type=_positive_int,
        metavar="SAMPLES",
        help="window hop in samples",
    )
    prepare.add_argument(
        "--compatibility-mode",
        metavar="MODE",
        default="filter_before_runs",
        help=(
            "filter_before_runs (default), strict_original_contiguity, public, "
            "or historical_train_validation_only"
        ),
    )
    prepare.add_argument(
        "--validate-only",
        action="store_true",
        help="run the in-memory checks and print metadata without writing",
    )
    prepare.add_argument(
        "--dry-run",
        action="store_true",
        help="show the JSON-safe preparation summary without writing",
    )
    prepare.add_argument(
        "--write-metadata",
        action="store_true",
        help="explicitly authorize writing prepare_data_metadata.json",
    )
    prepare.add_argument(
        "--overwrite",
        action="store_true",
        help="allow replacing an existing metadata JSON artifact",
    )

    subparsers.add_parser(
        "vae-smoke",
        help="run a no-write CPU shape and determinism smoke for both VAE channel sets",
        description=(
            "Run the public VAE on synthetic CPU tensors. This command never reads "
            "a checkpoint or writes an artifact."
        ),
    )

    inspect = subparsers.add_parser(
        "inspect-vae-checkpoint",
        help="validate one explicit weights-only VAE checkpoint and print safe metadata",
        description=(
            "Inspect one explicit VAE checkpoint without printing tensor values or "
            "copying checkpoint payloads."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    inspect.add_argument("--checkpoint", required=True, metavar="PATH")
    inspect.add_argument("--channels", type=_channel_count, metavar="N")
    inspect.add_argument(
        "--sensor-configuration",
        metavar="MODE_OR_YAML",
        help="six_channel, three_channel/accelerometer_only, or an explicit YAML config path",
    )

    reconstruct = subparsers.add_parser(
        "reconstruct",
        help="deterministically reconstruct one explicit safe .npy/.npz input",
        description=(
            "Load one explicit YAML config, weights-only checkpoint, and safe NumPy "
            "input; print shapes and safe metadata without writing output."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    reconstruct.add_argument("--config", required=True, metavar="PATH")
    reconstruct.add_argument("--checkpoint", required=True, metavar="PATH")
    reconstruct.add_argument(
        "--input",
        required=True,
        metavar="PATH",
        help="explicit .npy input or .npz containing one array (prefer key 'input')",
    )
    reconstruct.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="execution device; CUDA is used only when explicitly available",
    )
    return parser


def _execution_mode(args: argparse.Namespace) -> str:
    if args.dry_run:
        return "dry_run"
    if args.validate_only:
        return "validate_only"
    return "prepare"


def _summary_for_stdout(
    prepared: PreparedData,
    *,
    args: argparse.Namespace,
    metadata_written: bool,
) -> dict[str, Any]:
    summary = deepcopy(prepared.summary)
    summary["execution"] = {
        "mode": _execution_mode(args),
        "metadata_written": bool(metadata_written),
        "write_permission_requested": bool(args.write_metadata),
        "output_root_explicit": bool(args.output_root),
        "artifact_filename": METADATA_FILENAME if metadata_written else None,
        "participant_windows_serialized": False,
    }
    return summary


def _run_prepare_data(args: argparse.Namespace) -> int:
    if args.write_metadata and not args.output_root and not (args.dry_run or args.validate_only):
        raise PreparationError(
            "--write-metadata requires an explicit --output-root path"
        )

    prepared = prepare_data(
        data_root=args.data_root,
        config_path=args.config,
        held_out_subject=args.held_out_subject,
        sensor_configuration=args.sensor_configuration,
        seed=args.seed,
        window_length=args.window_length,
        hop_length=args.hop_length,
        compatibility_mode=args.compatibility_mode,
    )

    metadata_written = False
    # Dry-run and validation-only are hard no-write modes, even if a caller
    # also supplied the permission flag and an output root.
    if args.write_metadata and args.output_root and not (args.dry_run or args.validate_only):
        write_metadata_summary(
            prepared.summary,
            Path(args.output_root),
            overwrite=args.overwrite,
        )
        metadata_written = True

    payload = _summary_for_stdout(
        prepared,
        args=args,
        metadata_written=metadata_written,
    )
    print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False))
    return 0


def _sensor_configuration_channels(value: str) -> int:
    normalized = value.strip().lower().replace("-", "_")
    if normalized in {"six_channel", "6ch", "full"}:
        return 6
    if normalized in {"three_channel", "3ch", "accelerometer_only", "ablation"}:
        return 3
    path = Path(value).expanduser()
    if not path.is_file():
        raise ValueError(
            "sensor configuration must be six_channel, three_channel, accelerometer_only, or an explicit YAML path"
        )
    from .config import load_config

    return int(load_config(path, base_dir=path.parent).vae.input_channels)


def _requested_channels(args: argparse.Namespace) -> Optional[int]:
    values = []
    if getattr(args, "channels", None) is not None:
        values.append(int(args.channels))
    if getattr(args, "sensor_configuration", None):
        values.append(_sensor_configuration_channels(args.sensor_configuration))
    if values and any(value != values[0] for value in values[1:]):
        raise ValueError("--channels and --sensor-configuration select different channel counts")
    return values[0] if values else None


def _run_vae_smoke() -> int:
    import torch

    from .models.vae import LatentVAE1D
    from .training.vae import set_seed

    reports = []
    for channels in (6, 3):
        set_seed(42)
        model = LatentVAE1D(in_ch=channels, z_ch=48, down_levels=2).to("cpu")
        model.eval()
        inputs = torch.randn(2, channels, 160)
        with torch.no_grad():
            mean_a, mu, logvar = model(inputs, deterministic=True)
            mean_b, _, _ = model(inputs, deterministic=True)
            torch.manual_seed(7)
            stochastic_a, _, _ = model(inputs, deterministic=False)
            torch.manual_seed(7)
            stochastic_b, _, _ = model(inputs, deterministic=False)
        reports.append(
            {
                "channels": channels,
                "device": "cpu",
                "input_shape": list(inputs.shape),
                "latent_mean_shape": list(mu.shape),
                "latent_logvar_shape": list(logvar.shape),
                "reconstruction_shape": list(mean_a.shape),
                "deterministic_mean_equal": bool(torch.equal(mean_a, mean_b)),
                "fixed_seed_stochastic_equal": bool(torch.equal(stochastic_a, stochastic_b)),
                "payload_values_included": False,
            }
        )
    print(json.dumps({"command": "vae-smoke", "results": reports}, sort_keys=True))
    return 0


def _run_inspect_vae_checkpoint(args: argparse.Namespace) -> int:
    from .checkpoints import inspect_vae_checkpoint

    inspection = inspect_vae_checkpoint(
        args.checkpoint,
        channels=_requested_channels(args),
    )
    print(json.dumps(inspection.to_mapping(), indent=2, sort_keys=True))
    return 0


def _load_safe_input(path_value: str) -> np.ndarray:
    path = Path(path_value).expanduser()
    if not path.is_file():
        raise ValueError("input path does not name a file: {}".format(path))
    try:
        if path.suffix.lower() == ".npy":
            array = np.load(path, allow_pickle=False)
        elif path.suffix.lower() == ".npz":
            with np.load(path, allow_pickle=False) as archive:
                if "input" in archive.files:
                    array = archive["input"]
                elif len(archive.files) == 1:
                    array = archive[archive.files[0]]
                else:
                    raise ValueError(".npz input must contain one array or an 'input' array")
        else:
            raise ValueError("safe reconstruction input must be .npy or .npz")
    except (OSError, ValueError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith(".npz input"):
            raise
        raise ValueError("could not safely read NumPy input") from exc
    array = np.asarray(array)
    if array.ndim != 3:
        raise ValueError("input must have shape [batch, channels, time]")
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError("input must contain numeric values")
    return np.asarray(array, dtype=np.float32)


def _run_reconstruct(args: argparse.Namespace) -> int:
    import torch

    from .checkpoints import load_vae_checkpoint
    from .config import load_config
    from .training.vae import profile_from_config, resolve_device

    config_path = Path(args.config).expanduser().resolve()
    config = load_config(config_path, base_dir=config_path.parent)
    if config.window.samples != 160 or config.vae.latent_time_steps != 40:
        raise ValueError("reconstruct supports the public 160-sample/40-step VAE geometry")
    stride = int(config.vae.latent_stride)
    down_levels = int(round(math.log2(stride)))
    if 2**down_levels != stride:
        raise ValueError("vae.latent_stride must be a power of two")

    requested_device = args.device
    if requested_device == "auto":
        requested_device = config.runtime.device
    device = resolve_device(requested_device)
    model, inspection = load_vae_checkpoint(
        args.checkpoint,
        channels=config.vae.input_channels,
        latent_channels=config.vae.latent_dim_channels,
        down_levels=down_levels,
        device=device,
    )
    array = _load_safe_input(args.input)
    expected_channels = config.vae.input_channels
    if tuple(array.shape[1:]) != (expected_channels, 160):
        raise ValueError(
            "input shape {} does not match [batch, {}, 160]".format(
                tuple(array.shape), expected_channels
            )
        )
    inputs = torch.from_numpy(array).to(device)
    model.eval()
    with torch.no_grad():
        reconstruction, mu, logvar = model(inputs, deterministic=True)
    profile = profile_from_config(config.vae)
    payload = {
        "command": "reconstruct",
        "config_path": str(config_path),
        "checkpoint": inspection.to_mapping(),
        "input_shape": list(inputs.shape),
        "latent_mean_shape": list(mu.shape),
        "latent_logvar_shape": list(logvar.shape),
        "reconstruction_shape": list(reconstruction.shape),
        "device": str(device),
        "deterministic_posterior_mean": True,
        "profile": profile.to_mapping(),
        "exact_paper_reproduction": False,
        "output_written": False,
        "tensor_values_included": False,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the package CLI and return a process exit code."""

    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.command is None:
        parser.print_help()
        return 0
    try:
        if args.command == "prepare-data":
            return _run_prepare_data(args)
        if args.command == "vae-smoke":
            return _run_vae_smoke()
        if args.command == "inspect-vae-checkpoint":
            return _run_inspect_vae_checkpoint(args)
        if args.command == "reconstruct":
            return _run_reconstruct(args)
        parser.error("unknown command: {}".format(args.command))
    except (PreparationError, FileExistsError, NotADirectoryError, OSError, ValueError) as exc:
        print("lrf-imu: error: {}".format(exc), file=sys.stderr)
        return 2
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            print(
                "lrf-imu: error: VAE commands require optional PyTorch; "
                "install lrf-imu[training]",
                file=sys.stderr,
            )
            return 2
        raise
    return 2


__all__ = ["build_parser", "main"]
