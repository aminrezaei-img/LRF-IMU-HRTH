"""Command-line entry points for the public LRF-IMU package."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
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
        parser.error("unknown command: {}".format(args.command))
    except (PreparationError, FileExistsError, NotADirectoryError, OSError, ValueError) as exc:
        print("lrf-imu: error: {}".format(exc), file=sys.stderr)
        return 2
    return 2


__all__ = ["build_parser", "main"]
