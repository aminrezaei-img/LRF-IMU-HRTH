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


def _add_flow_geometry_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--channels", type=_channel_count, metavar="N")
    parser.add_argument("--latent-channels", type=_positive_int, default=48, metavar="N")
    parser.add_argument("--classes", "--num-classes", type=_positive_int, default=4, metavar="N")
    parser.add_argument(
        "--width-profile",
        default="historical_checkpoint_compatibility_256",
        help="historical_checkpoint_compatibility_256, manuscript_reported_128, or custom",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the parser without reading configuration or importing Torch."""

    parser = argparse.ArgumentParser(
        prog="lrf-imu",
        description=(
            "Portable metadata-first orchestration for the public LRF-IMU "
            "data, VAE, and Rectified Flow boundaries."
        ),
    )
    parser.add_argument("--version", action="version", version="%(prog)s 1.0.0")
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
    prepare.add_argument("--config", metavar="PATH")
    prepare.add_argument("--data-root", metavar="PATH")
    prepare.add_argument("--output-root", metavar="PATH")
    prepare.add_argument("--held-out-subject", type=_positive_int, metavar="ID")
    prepare.add_argument("--sensor-configuration", metavar="MODE_OR_YAML")
    prepare.add_argument("--seed", type=_non_negative_int, metavar="N")
    prepare.add_argument("--window-length", type=_positive_int, metavar="SAMPLES")
    prepare.add_argument("--hop-length", type=_positive_int, metavar="SAMPLES")
    prepare.add_argument(
        "--compatibility-mode",
        metavar="MODE",
        default="filter_before_runs",
        help="filter_before_runs (default), strict_original_contiguity, public, or historical_train_validation_only",
    )
    prepare.add_argument("--validate-only", action="store_true")
    prepare.add_argument("--dry-run", action="store_true")
    prepare.add_argument("--write-metadata", action="store_true")
    prepare.add_argument("--overwrite", action="store_true")

    harth_prepare = subparsers.add_parser(
        "prepare-harth-data",
        help="prepare the HARTH-family ten-class thigh-accelerometer input",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    harth_prepare.add_argument("--data-root", required=True, metavar="PATH")
    harth_prepare.add_argument("--composition", default="harth_walking_speed", choices=(
        "harth", "harth_walking_speed", "harth_walking_speed_har70plus"
    ))
    harth_prepare.add_argument("--held-out-subject", metavar="DATASET:ID")
    harth_prepare.add_argument("--window-length", type=_positive_int, default=160)
    harth_prepare.add_argument("--hop-length", type=_positive_int, default=40)
    harth_prepare.add_argument("--seed", type=_non_negative_int, default=42)

    subparsers.add_parser(
        "vae-smoke",
        help="run a no-write CPU shape and determinism smoke for both VAE channel sets",
        description="Run the public VAE on synthetic CPU tensors without reading or writing artifacts.",
    )

    inspect_vae = subparsers.add_parser(
        "inspect-vae-checkpoint",
        help="validate one explicit weights-only VAE checkpoint and print safe metadata",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    inspect_vae.add_argument("--checkpoint", required=True, metavar="PATH")
    inspect_vae.add_argument("--channels", type=_channel_count, metavar="N")
    inspect_vae.add_argument("--sensor-configuration", metavar="MODE_OR_YAML")

    reconstruct = subparsers.add_parser(
        "reconstruct",
        help="deterministically reconstruct one explicit safe .npy/.npz input",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    reconstruct.add_argument("--config", required=True, metavar="PATH")
    reconstruct.add_argument("--checkpoint", required=True, metavar="PATH")
    reconstruct.add_argument("--input", required=True, metavar="PATH")
    reconstruct.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")

    flow_smoke = subparsers.add_parser(
        "flow-smoke",
        help="run a no-write synthetic CPU shape and conditioning smoke for Rectified Flow",
        description=(
            "Build the explicit flow U-Net profile on synthetic latent tensors. "
            "No checkpoints, participant data, or tensor values are written or printed."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    flow_smoke.add_argument(
        "--width-profile",
        default="historical_checkpoint_compatibility_256",
        help="historical_checkpoint_compatibility_256, manuscript_reported_128, or custom",
    )
    flow_smoke.add_argument("--model-ch", type=_positive_int, default=None)
    flow_smoke.add_argument("--batch-size", type=_positive_int, default=1)
    flow_smoke.add_argument("--device", choices=("cpu",), default="cpu")

    inspect_flow = subparsers.add_parser(
        "inspect-flow-checkpoint",
        help="validate one explicit six-key weights-only flow checkpoint",
        description="Inspect a flow checkpoint without printing tensor values or copying payloads.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    inspect_flow.add_argument("--checkpoint", required=True, metavar="PATH")
    _add_flow_geometry_arguments(inspect_flow)
    inspect_flow.add_argument("--model-ch", type=_positive_int, default=None)

    generate = subparsers.add_parser(
        "generate",
        help="sample paper/TSTR latent windows from one explicit flow checkpoint",
        description=(
            "Use the paper profile: noise at t=1, ten reverse Euler steps, seed 42, "
            "and 500 windows per class by default. Output is written only when --output is explicit."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    generate.add_argument(
        "--config",
        metavar="PATH",
        help="optional release YAML used to validate the paired channel and latent geometry",
    )
    generate.add_argument("--flow-checkpoint", required=True, metavar="PATH")
    generate.add_argument("--vae-checkpoint", metavar="PATH")
    generate.add_argument("--latent-only", action="store_true")
    _add_flow_geometry_arguments(generate)
    generate.add_argument("--model-ch", type=_positive_int, default=None)
    generate.add_argument("--class-id", type=_non_negative_int, default=None, metavar="ID")
    generate.add_argument(
        "--samples-per-class",
        "--count",
        dest="samples_per_class",
        type=_positive_int,
        default=500,
        metavar="N",
        help="number of windows for every class, or for --class-id when selected",
    )
    generate.add_argument(
        "--num-steps",
        "--steps",
        dest="num_steps",
        type=_positive_int,
        default=10,
        metavar="N",
    )
    generate.add_argument("--seed", type=_non_negative_int, default=42)
    generate.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    generate.add_argument("--output", metavar="PATH", help="explicit .npz or .json output path")

    export = subparsers.add_parser(
        "export-trajectories",
        help="generate website-only 100-step trajectories with explicit output paths",
        description=(
            "Use the distinct website profile: 100 reverse Euler steps, record every 2 steps, "
            "51 states, independent native windows, 40-sample overlap, and ten seconds."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    export.add_argument("--flow-checkpoint", required=True, metavar="PATH")
    export.add_argument("--vae-checkpoint", required=True, metavar="PATH")
    export.add_argument("--subject", required=True, type=_positive_int, metavar="ID")
    export.add_argument("--activity", required=True, type=_non_negative_int, metavar="ID")
    export.add_argument("--activity-name", default=None)
    _add_flow_geometry_arguments(export)
    export.add_argument("--model-ch", type=_positive_int, default=None)
    export.add_argument("--base-seed", type=_non_negative_int, default=42)
    export.add_argument("--num-steps", type=_positive_int, default=100)
    export.add_argument("--record-every", type=_positive_int, default=2)
    export.add_argument("--duration-seconds", type=float, default=10.0)
    export.add_argument("--overlap-samples", type=_non_negative_int, default=40)
    export.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    export.add_argument("--output", metavar="PATH", help="explicit JSON output path")
    from .evaluation.cli import add_evaluation_parsers

    add_evaluation_parsers(subparsers)
    from .analysis.cli import add_analysis_parsers

    add_analysis_parsers(subparsers)
    from .reproducibility import add_reproduce_parser

    add_reproduce_parser(subparsers)
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
        raise PreparationError("--write-metadata requires an explicit --output-root path")
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
    if args.write_metadata and args.output_root and not (args.dry_run or args.validate_only):
        write_metadata_summary(prepared.summary, Path(args.output_root), overwrite=args.overwrite)
        metadata_written = True
    print(json.dumps(_summary_for_stdout(prepared, args=args, metadata_written=metadata_written), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False))
    return 0


def _sensor_configuration_channels(value: str) -> int:
    normalized = value.strip().lower().replace("-", "_")
    if normalized in {"six_channel", "6ch", "full"}:
        return 6
    if normalized in {"three_channel", "3ch", "accelerometer_only", "ablation"}:
        return 3
    path = Path(value).expanduser()
    if not path.is_file():
        raise ValueError("sensor configuration must be six_channel, three_channel, accelerometer_only, or an explicit YAML path")
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


def _run_prepare_harth_data(args: argparse.Namespace) -> int:
    from .data.harth_pipeline import prepare_harth_data

    prepared = prepare_harth_data(
        args.data_root,
        composition=args.composition,
        held_out_subject=args.held_out_subject,
        window_length=args.window_length,
        hop_length=args.hop_length,
        seed=args.seed,
    )
    print(json.dumps({
        "command": "prepare-harth-data",
        "execution": {"metadata_only": True, "participant_windows_serialized": False},
        **prepared.summary,
    }, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False))
    return 0


def _run_vae_smoke() -> int:
    import torch
    from .models.vae import LatentVAE1D
    from .training.vae import set_seed

    reports = []
    for channels in (6, 3):
        set_seed(42)
        model = LatentVAE1D(in_ch=channels, z_ch=48, down_levels=2).to("cpu").eval()
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

    inspection = inspect_vae_checkpoint(args.checkpoint, channels=_requested_channels(args))
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
    if array.ndim != 3 or not np.issubdtype(array.dtype, np.number):
        raise ValueError("input must be a numeric array with shape [batch, channels, time]")
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
    requested_device = config.runtime.device if args.device == "auto" else args.device
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
        raise ValueError("input shape {} does not match [batch, {}, 160]".format(tuple(array.shape), expected_channels))
    inputs = torch.from_numpy(array).to(device)
    with torch.no_grad():
        reconstruction, mu, logvar = model(inputs, deterministic=True)
    print(json.dumps({
        "command": "reconstruct",
        "config_path": str(config_path),
        "checkpoint": inspection.to_mapping(),
        "input_shape": list(inputs.shape),
        "latent_mean_shape": list(mu.shape),
        "latent_logvar_shape": list(logvar.shape),
        "reconstruction_shape": list(reconstruction.shape),
        "device": str(device),
        "deterministic_posterior_mean": True,
        "profile": profile_from_config(config.vae).to_mapping(),
        "exact_paper_reproduction": False,
        "output_written": False,
        "tensor_values_included": False,
    }, indent=2, sort_keys=True))
    return 0


def _run_flow_smoke(args: argparse.Namespace) -> int:
    import torch
    from .models.flow import (
        LatentDiffusionUNet1D,
        flow_model_metadata,
        select_flow_profile,
    )

    if args.model_ch is None:
        profile = select_flow_profile(args.width_profile)
    elif args.width_profile == "historical_checkpoint_compatibility_256":
        profile = select_flow_profile("custom", model_ch=args.model_ch)
    else:
        profile = select_flow_profile(args.width_profile, model_ch=args.model_ch)
    model = LatentDiffusionUNet1D(in_ch=48, model_ch=profile.model_ch, num_classes=4).to(args.device).eval()
    batch_size = int(args.batch_size)
    inputs = torch.randn(batch_size, 48, 40, device=args.device)
    times = torch.full((batch_size,), 250.0, device=args.device)
    labels = torch.arange(batch_size, device=args.device) % 4
    with torch.no_grad():
        output = model(inputs, times, labels)
        class_changed = model(inputs, times, (labels + 1) % 4)
        time_changed = model(inputs, times + 100.0, labels)
    metadata = flow_model_metadata(channels=6, model_ch=profile.model_ch, width_profile=profile.name)
    metadata.update({
        "command": "flow-smoke",
        "input_shape": list(inputs.shape),
        "output_shape": list(output.shape),
        "class_conditioning_changes_output": bool(not torch.equal(output, class_changed)),
        "time_conditioning_changes_output": bool(not torch.equal(output, time_changed)),
        "device": str(args.device),
        "tensor_values_included": False,
    })
    print(json.dumps(metadata, sort_keys=True))
    return 0


def _run_inspect_flow(args: argparse.Namespace) -> int:
    from .checkpoints import inspect_flow_checkpoint

    inspection = inspect_flow_checkpoint(
        args.checkpoint,
        channels=args.channels,
        latent_channels=args.latent_channels,
        num_classes=args.classes,
        model_ch=args.model_ch,
        width_profile=args.width_profile,
    )
    print(json.dumps(inspection.to_mapping(), indent=2, sort_keys=True))
    return 0


def _run_generate(args: argparse.Namespace) -> int:
    import torch
    from .checkpoints import load_flow_checkpoint, load_vae_checkpoint
    from .generation.flow import (
        generate_paper_latents,
        paper_sampling_metadata,
        sample_reverse_euler,
    )

    config_summary = None
    if args.config:
        from .config import load_config

        config_path = Path(args.config).expanduser().resolve()
        config = load_config(config_path, base_dir=config_path.parent)
        configured_channels = int(config.vae.input_channels)
        if args.channels is None:
            args.channels = configured_channels
        elif int(args.channels) != configured_channels:
            raise ValueError(
                "config VAE channels {} do not match --channels {}".format(
                    configured_channels, int(args.channels)
                )
            )
        configured_latent_channels = int(config.vae.latent_dim_channels)
        if int(args.latent_channels) != configured_latent_channels:
            raise ValueError(
                "config latent channels {} do not match --latent-channels {}".format(
                    configured_latent_channels, int(args.latent_channels)
                )
            )
        configured_classes = int(config.flow.architecture.num_classes)
        if int(args.classes) != configured_classes:
            raise ValueError(
                "config class count {} does not match --classes {}".format(
                    configured_classes, int(args.classes)
                )
            )
        config_summary = {
            "path": str(config_path),
            "channels": configured_channels,
            "latent_channels": configured_latent_channels,
            "num_classes": configured_classes,
            "width_profile": str(config.flow.width_profile),
            "exact_paper_reproduction": bool(config.release.exact_paper_reproduction),
        }
    model, flow_inspection = load_flow_checkpoint(
        args.flow_checkpoint,
        channels=args.channels,
        latent_channels=args.latent_channels,
        num_classes=args.classes,
        model_ch=args.model_ch,
        width_profile=args.width_profile,
        device=args.device,
    )
    if args.class_id is not None and int(args.class_id) >= int(args.classes):
        raise ValueError(
            "class-id must be in [0, {}) for this flow model".format(int(args.classes))
        )
    if not args.vae_checkpoint and not args.latent_only:
        raise ValueError("--vae-checkpoint is required unless --latent-only is selected")
    if args.class_id is None:
        latents, labels, metadata = generate_paper_latents(
            model,
            samples_per_class=args.samples_per_class,
            num_steps=args.num_steps,
            seed=args.seed,
            latent_shape=(args.latent_channels, 40),
            device=args.device,
        )
    else:
        labels = torch.full(
            (int(args.samples_per_class),),
            int(args.class_id),
            dtype=torch.long,
            device=torch.device(args.device),
        )
        latents = sample_reverse_euler(
            model,
            labels,
            (args.latent_channels, 40),
            num_steps=args.num_steps,
            seed=args.seed,
            device=args.device,
        )
        metadata = paper_sampling_metadata(
            samples_per_class=args.samples_per_class,
            num_steps=args.num_steps,
            seed=args.seed,
        )
        metadata.update({
            "class_id": int(args.class_id),
            "class_count": 1,
        })
    decoded = None
    vae_inspection = None
    if args.vae_checkpoint:
        vae, vae_inspection = load_vae_checkpoint(
            args.vae_checkpoint,
            channels=flow_inspection.channels,
            latent_channels=args.latent_channels,
            down_levels=2,
            device=args.device,
        )
        with torch.no_grad():
            decoded = vae.decode(latents)
    payload = {
        "command": "generate",
        "profile": "paper",
        "config": config_summary,
        "flow_checkpoint": flow_inspection.to_mapping(),
        "vae_checkpoint": None if vae_inspection is None else vae_inspection.to_mapping(),
        "sampling": metadata,
        "class_id": None if args.class_id is None else int(args.class_id),
        "latent_shape": list(latents.shape),
        "label_shape": list(labels.shape),
        "decoded_shape": None if decoded is None else list(decoded.shape),
        "device": str(args.device),
        "output_written": False,
        "tensor_values_included": False,
        "website_trajectory": False,
    }
    if args.output:
        destination = Path(args.output).expanduser().resolve()
        if destination.exists() and destination.is_dir():
            raise ValueError("--output must be a file path")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.suffix.lower() not in (".json", ".npz"):
            raise ValueError("--output must end in .json or .npz")
        payload["output_written"] = True
        payload["output_path"] = str(destination)
        if destination.suffix.lower() == ".json":
            destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        else:
            arrays = {"latents": latents.detach().cpu().numpy(), "labels": labels.detach().cpu().numpy()}
            if decoded is not None:
                arrays["samples"] = decoded.detach().cpu().numpy()
            np.savez_compressed(destination, **arrays)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _run_export_trajectories(args: argparse.Namespace) -> int:
    from .checkpoints import load_flow_checkpoint, load_vae_checkpoint
    from .generation.flow import generate_website_trajectory, write_trajectory_payload

    flow_model, flow_inspection = load_flow_checkpoint(
        args.flow_checkpoint,
        channels=args.channels,
        latent_channels=args.latent_channels,
        num_classes=args.classes,
        model_ch=args.model_ch,
        width_profile=args.width_profile,
        device=args.device,
    )
    vae, vae_inspection = load_vae_checkpoint(
        args.vae_checkpoint,
        channels=flow_inspection.channels,
        latent_channels=args.latent_channels,
        down_levels=2,
        device=args.device,
    )
    payload = generate_website_trajectory(
        flow_model,
        vae,
        subject_id=args.subject,
        activity_id=args.activity,
        base_seed=args.base_seed,
        duration_seconds=args.duration_seconds,
        overlap_samples=args.overlap_samples,
        num_steps=args.num_steps,
        record_every=args.record_every,
        activity_name=args.activity_name,
        vae_checkpoint_name=Path(args.vae_checkpoint).name,
        flow_checkpoint_name=Path(args.flow_checkpoint).name,
        device=args.device,
    )
    if args.output:
        destination = write_trajectory_payload(payload, args.output, overwrite=False)
        output_written = True
        output_path = str(destination)
    else:
        output_written = False
        output_path = None
    metadata = dict(payload)
    metadata.pop("signals", None)
    metadata.update({
        "command": "export-trajectories",
        "flow_checkpoint": flow_inspection.to_mapping(),
        "vae_checkpoint": vae_inspection.to_mapping(),
        "output_written": output_written,
        "output_path": output_path,
        "tensor_values_included": False,
        "website_trajectory": True,
    })
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


def _optional_torch_error(exc: ImportError) -> bool:
    message = str(exc).lower()
    return getattr(exc, "name", None) == "torch" or "no module named 'torch'" in message or "no module named \"torch\"" in message


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
        if args.command == "prepare-harth-data":
            return _run_prepare_harth_data(args)
        if args.command == "vae-smoke":
            return _run_vae_smoke()
        if args.command == "inspect-vae-checkpoint":
            return _run_inspect_vae_checkpoint(args)
        if args.command == "reconstruct":
            return _run_reconstruct(args)
        if args.command == "flow-smoke":
            return _run_flow_smoke(args)
        if args.command == "inspect-flow-checkpoint":
            return _run_inspect_flow(args)
        if args.command == "generate":
            return _run_generate(args)
        if args.command == "export-trajectories":
            return _run_export_trajectories(args)
        if args.command == "evaluate":
            from .evaluation.cli import run_evaluate

            return run_evaluate(args)
        if args.command == "evaluate-loso":
            from .evaluation.cli import run_evaluate_loso

            return run_evaluate_loso(args)
        if args.command == "reproduce-core":
            from .reproducibility import run_reproduce_core

            return run_reproduce_core(args)
        from .analysis.cli import ANALYSIS_COMMANDS, run_analysis_command

        if args.command in ANALYSIS_COMMANDS:
            return run_analysis_command(args)
        parser.error("unknown command: {}".format(args.command))
    except (PreparationError, FileExistsError, NotADirectoryError, OSError, ValueError) as exc:
        print("lrf-imu: error: {}".format(exc), file=sys.stderr)
        return 2
    except ImportError as exc:
        if _optional_torch_error(exc):
            print(
                "lrf-imu: error: flow/VAE commands require optional PyTorch; install lrf-imu[training]",
                file=sys.stderr,
            )
            return 2
        raise
    return 2


__all__ = ["build_parser", "main"]
