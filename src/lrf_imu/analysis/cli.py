"""No-write-by-default CLI composition for Milestone 3E analyses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .physical import acceleration_magnitude_summary
from .privacy import (
    POSTHOC_AUDIT_THREAT_MODEL,
    RECONSTRUCTION_THREAT_MODEL,
    TRUE_HOLDOUT_THREAT_MODEL,
    summarize_membership_records,
    summarize_reconstruction_records,
)
from .sensitivity import summarize_sensitivity_grid


def _positive(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def _add_output(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output", metavar="JSON")
    parser.add_argument("--write-results", action="store_true")
    parser.add_argument("--overwrite", action="store_true")


def add_analysis_parsers(subparsers: Any) -> None:
    physical = subparsers.add_parser(
        "analyze-physical",
        help="summarize physical acceleration magnitude and the strict >10g check",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    physical.add_argument("--input", required=True, metavar="NPY_OR_NPZ")
    physical.add_argument("--array-key", default="samples")
    physical.add_argument("--units", required=True, choices=("m_s2",))
    physical.add_argument("--max-g", type=float, default=10.0)
    _add_output(physical)

    spectral = subparsers.add_parser(
        "analyze-spectral",
        help="compare real/synthetic Welch PSD without plotting",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    spectral.add_argument("--real", required=True, metavar="NPY_OR_NPZ")
    spectral.add_argument("--synthetic", required=True, metavar="NPY_OR_NPZ")
    spectral.add_argument("--real-key", default="samples")
    spectral.add_argument("--synthetic-key", default="samples")
    spectral.add_argument("--sampling-hz", type=float, default=50.0)
    spectral.add_argument("--nperseg", type=_positive, default=160)
    spectral.add_argument("--channel-names", metavar="COMMA_LIST")
    _add_output(spectral)

    sensitivity = subparsers.add_parser(
        "analyze-sensitivity",
        help="aggregate the nine-setting window/hop grid from fold JSON records",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sensitivity.add_argument("--input", required=True, metavar="JSON")
    _add_output(sensitivity)

    privacy = subparsers.add_parser(
        "analyze-privacy",
        help="summarize one explicit historical privacy threat model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    privacy.add_argument("--input", required=True, metavar="JSON")
    privacy.add_argument(
        "--threat-model",
        required=True,
        choices=(TRUE_HOLDOUT_THREAT_MODEL, POSTHOC_AUDIT_THREAT_MODEL, RECONSTRUCTION_THREAT_MODEL),
    )
    _add_output(privacy)

    vae = subparsers.add_parser(
        "evaluate-vae-only",
        help="run one VAE-only latent-Gaussian RF ablation fold",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    vae.add_argument("--data-root", required=True, metavar="PATH")
    vae.add_argument("--config", metavar="YAML")
    vae.add_argument("--vae-checkpoint", required=True, metavar="PT")
    vae.add_argument("--held-out-subject", required=True, type=_positive)
    vae.add_argument("--sensor", choices=("six_channel", "three_channel"), default="six_channel")
    vae.add_argument("--samples-per-class", type=_positive, default=500)
    vae.add_argument("--seed", type=int, default=42)
    vae.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    vae.add_argument("--batch-size", type=_positive, default=256)
    vae.add_argument("--dry-run", action="store_true")
    _add_output(vae)


def _load_array(path_value: str, key: str) -> np.ndarray:
    path = Path(path_value).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"array input does not exist: {path}")
    if path.suffix.lower() == ".npy":
        return np.asarray(np.load(path, allow_pickle=False))
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            if key not in archive.files:
                raise ValueError(f"array key {key!r} not found in {path.name}")
            return np.asarray(archive[key])
    raise ValueError("analysis arrays must be explicit .npy or .npz files")


def _load_json(path_value: str) -> Any:
    path = Path(path_value).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"JSON input does not exist: {path}")
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def _emit(payload: dict[str, Any], args: argparse.Namespace) -> int:
    output = getattr(args, "output", None)
    write = bool(getattr(args, "write_results", False))
    if output and not write:
        raise ValueError("--output requires --write-results")
    if write and not output:
        raise ValueError("--write-results requires an explicit --output JSON path")
    payload["execution"] = {
        "output_written": False,
        "write_permission_requested": write,
        "participant_or_synthetic_arrays_written": False,
    }
    if write:
        destination = Path(str(output)).expanduser().resolve()
        if destination.suffix.lower() != ".json":
            raise ValueError("analysis output must be a .json file")
        if destination.exists() and not args.overwrite:
            raise FileExistsError(f"analysis output exists; use --overwrite: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload["execution"]["output_written"] = True
        payload["execution"]["output_path"] = str(destination)
        mode = "w" if args.overwrite else "x"
        with destination.open(mode, encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0


def _run_physical(args: argparse.Namespace) -> int:
    payload = acceleration_magnitude_summary(
        _load_array(args.input, args.array_key), max_g=float(args.max_g)
    )
    payload["command"] = "analyze-physical"
    return _emit(payload, args)


def _run_spectral(args: argparse.Namespace) -> int:
    from .spectral import compute_psd, spectral_statistics

    real = _load_array(args.real, args.real_key)
    synthetic = _load_array(args.synthetic, args.synthetic_key)
    if real.shape[1] != synthetic.shape[1]:
        raise ValueError("real and synthetic channel counts differ")
    frequencies, real_psd = compute_psd(real, sampling_hz=args.sampling_hz, nperseg=args.nperseg)
    synthetic_frequencies, synthetic_psd = compute_psd(
        synthetic, sampling_hz=args.sampling_hz, nperseg=args.nperseg
    )
    if not np.array_equal(frequencies, synthetic_frequencies):
        raise ValueError("real and synthetic Welch frequency grids differ")
    real_mean = np.mean(real_psd, axis=0)
    real_std = np.std(real_psd, axis=0, ddof=0)
    synthetic_mean = np.mean(synthetic_psd, axis=0)
    synthetic_std = np.std(synthetic_psd, axis=0, ddof=0)
    names = args.channel_names.split(",") if args.channel_names else None
    payload = spectral_statistics(
        frequencies, real_mean, synthetic_mean,
        real_std=real_std, synthetic_std=synthetic_std, channel_names=names,
    )
    payload["command"] = "analyze-spectral"
    payload["welch"].update({
        "sampling_hz": float(args.sampling_hz),
        "nperseg": min(args.nperseg, real.shape[-1]),
    })
    payload["input_window_counts"] = {"real": int(real.shape[0]), "synthetic": int(synthetic.shape[0])}
    return _emit(payload, args)


def _run_sensitivity(args: argparse.Namespace) -> int:
    source = _load_json(args.input)
    records = source.get("settings", source) if isinstance(source, dict) else source
    if not isinstance(records, dict):
        raise ValueError("sensitivity JSON must map setting names to fold-record lists")
    payload = summarize_sensitivity_grid(records)
    payload["command"] = "analyze-sensitivity"
    return _emit(payload, args)


def _run_privacy(args: argparse.Namespace) -> int:
    source = _load_json(args.input)
    records = source.get("records", source) if isinstance(source, dict) else source
    if not isinstance(records, list):
        raise ValueError("privacy JSON must be a record list or contain a records list")
    if args.threat_model == RECONSTRUCTION_THREAT_MODEL:
        payload = summarize_reconstruction_records(records)
    else:
        payload = summarize_membership_records(records, threat_model=args.threat_model)
    payload["command"] = "analyze-privacy"
    return _emit(payload, args)


def _vae_plan(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema_version": "m3e.vae-only-plan.1",
        "command": "evaluate-vae-only",
        "data_root": str(Path(args.data_root).expanduser().resolve()),
        "config": str(Path(args.config).expanduser().resolve()) if args.config else None,
        "vae_checkpoint": str(Path(args.vae_checkpoint).expanduser().resolve()),
        "held_out_subject": int(args.held_out_subject),
        "sensor": args.sensor,
        "samples_per_class": int(args.samples_per_class),
        "seed": int(args.seed),
        "device": args.device,
        "dry_run": bool(args.dry_run),
        "output_target": str(Path(args.output).expanduser().resolve()) if args.output else None,
    }


def _run_vae_only(args: argparse.Namespace) -> int:
    plan = _vae_plan(args)
    if args.dry_run:
        return _emit(plan, args)
    from ..checkpoints import load_vae_checkpoint
    from ..data.pipeline import prepare_data
    from .ablation import generate_vae_only_samples, vae_only_random_forest_metrics

    prepared = prepare_data(
        data_root=args.data_root,
        config_path=args.config,
        held_out_subject=args.held_out_subject,
        sensor_configuration=args.sensor,
        seed=args.seed,
        compatibility_mode="filter_before_runs",
    )
    vae, inspection = load_vae_checkpoint(
        args.vae_checkpoint,
        channels=prepared.sensor_schema.channel_count,
        latent_channels=48,
        down_levels=2,
        device=args.device,
    )
    synthetic_windows, synthetic_labels = generate_vae_only_samples(
        vae, prepared.train_windows, prepared.train_labels,
        samples_per_class=args.samples_per_class, seed=args.seed,
        device=args.device, batch_size=args.batch_size,
    )
    payload = vae_only_random_forest_metrics(
        prepared.train_windows, prepared.train_labels,
        prepared.held_out_test_windows, prepared.held_out_test_labels,
        synthetic_windows, synthetic_labels, seed=args.seed,
    )
    payload.update({
        "command": "evaluate-vae-only",
        "plan": plan,
        "checkpoint": inspection.to_mapping(),
        "provenance": {
            "train_subjects": list(prepared.split.train_subjects),
            "vae_validation_subjects": list(prepared.split.validation_subjects),
            "train_window_count": int(prepared.train_labels.size),
            "validation_window_count": int(prepared.validation_labels.size),
            "test_window_count": int(prepared.held_out_test_labels.size),
            "synthetic_count_by_class": {
                str(label): int(np.count_nonzero(synthetic_labels == label)) for label in range(4)
            },
        },
        "synthetic_arrays_retained": False,
        "exact_paper_reproduction": False,
    })
    return _emit(payload, args)


def run_analysis_command(args: argparse.Namespace) -> int:
    runners = {
        "analyze-physical": _run_physical,
        "analyze-spectral": _run_spectral,
        "analyze-sensitivity": _run_sensitivity,
        "analyze-privacy": _run_privacy,
        "evaluate-vae-only": _run_vae_only,
    }
    return runners[args.command](args)


ANALYSIS_COMMANDS = frozenset({
    "analyze-physical", "analyze-spectral", "analyze-sensitivity",
    "analyze-privacy", "evaluate-vae-only",
})

__all__ = ["ANALYSIS_COMMANDS", "add_analysis_parsers", "run_analysis_command"]
