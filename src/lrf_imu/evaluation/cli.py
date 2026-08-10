"""CLI composition for portable one-fold and LOSO evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from ..data.pipeline import prepare_data
from .cache import load_validated_synthetic_cache
from .core import evaluate_scenarios
from .metrics import summarize_fold_records

CANONICAL_SUBJECTS = (1, 2, 3, 5, 8, 9, 10, 11, 12, 13, 14, 16)


def _positive(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", metavar="PATH")
    parser.add_argument("--data-root", required=True, metavar="PATH")
    parser.add_argument("--checkpoint-root", metavar="PATH")
    parser.add_argument(
        "--sensor", choices=("six_channel", "three_channel"), default="six_channel"
    )
    parser.add_argument("--classifier", choices=("rf", "cnn"), default="rf")
    parser.add_argument(
        "--scenario",
        action="append",
        choices=("trtr", "scarce", "tstr", "tstr_scarce"),
        help=(
            "repeat to select returned scenarios; TRTR is run internally when needed "
            "for retention; default returns all four"
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--samples-per-class", type=_positive, default=500)
    parser.add_argument("--scarce-per-class", type=_positive, default=2)
    parser.add_argument("--output-root", metavar="PATH")
    parser.add_argument("--write-results", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")


def add_evaluation_parsers(subparsers: Any) -> None:
    evaluate = subparsers.add_parser(
        "evaluate",
        help="evaluate one LOSO fold from explicit external inputs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_common(evaluate)
    evaluate.add_argument("--held-out-subject", required=True, type=_positive)
    evaluate.add_argument("--synthetic-cache", metavar="NPZ")

    loso = subparsers.add_parser(
        "evaluate-loso",
        help="evaluate the canonical 12 folds with resumable external outputs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_common(loso)
    loso.add_argument("--synthetic-root", metavar="PATH")


def _scenarios(args: argparse.Namespace) -> tuple[str, ...]:
    return tuple(args.scenario or ("trtr", "scarce", "tstr", "tstr_scarce"))


def _target(args: argparse.Namespace, subject: int) -> Path | None:
    if not args.output_root:
        return None
    return Path(args.output_root).expanduser().resolve() / (
        "subject_{:02d}_{}_evaluation.json".format(subject, args.classifier)
    )


def _synthetic_for_fold(args: argparse.Namespace, subject: int) -> Path | None:
    explicit = getattr(args, "synthetic_cache", None)
    if explicit:
        return Path(explicit).expanduser().resolve()
    root_value = getattr(args, "synthetic_root", None)
    if not root_value:
        return None
    root = Path(root_value).expanduser().resolve()
    candidates = (
        root / "subject_{:02d}".format(subject) / "steps10_synthetic.npz",
        root / "subject_{:02d}.npz".format(subject),
    )
    return next((candidate for candidate in candidates if candidate.is_file()), candidates[0])


def _validate_write_contract(args: argparse.Namespace) -> None:
    if args.write_results and not args.output_root:
        raise ValueError("--write-results requires an explicit --output-root")


def _write_result(target: Path, result: dict[str, Any], args: argparse.Namespace) -> None:
    if not args.write_results:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not args.overwrite:
        raise FileExistsError("result exists; use --resume or --overwrite: {}".format(target))
    mode = "w" if args.overwrite else "x"
    with target.open(mode, encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _resume_fold(target: Path) -> dict[str, Any]:
    with target.open("r", encoding="utf-8") as handle:
        resumed = json.load(handle)
    if not isinstance(resumed, dict) or not isinstance(resumed.get("records"), list):
        raise ValueError("resumed fold does not match the m3d fold-evaluation schema")
    resumed["execution"] = {
        "status": "resumed",
        "resumed": True,
        "target": str(target),
    }
    return resumed


def evaluate_one(args: argparse.Namespace, subject: int) -> dict[str, Any]:
    _validate_write_contract(args)
    scenarios = _scenarios(args)
    target = _target(args, subject)
    synthetic_path = _synthetic_for_fold(args, subject)
    plan = {
        "schema_version": "m3d.evaluation-plan.1",
        "held_out_subject": subject,
        "sensor": args.sensor,
        "classifier": args.classifier,
        "requested_scenarios": list(scenarios),
        "data_root": str(Path(args.data_root).expanduser().resolve()),
        "checkpoint_root": (
            str(Path(args.checkpoint_root).expanduser().resolve())
            if args.checkpoint_root
            else None
        ),
        "synthetic_cache": str(synthetic_path) if synthetic_path else None,
        "output_target": str(target) if target else None,
        "write_permission_requested": bool(args.write_results),
        "dry_run": bool(args.dry_run),
    }
    if args.dry_run:
        return plan
    if target and target.exists() and args.resume:
        return _resume_fold(target)

    needs_synthetic = any(name in {"tstr", "tstr_scarce"} for name in scenarios)
    if needs_synthetic and synthetic_path is None:
        raise ValueError("TSTR scenarios require --synthetic-cache/--synthetic-root")
    synthetic_windows = None
    synthetic_labels = None
    cache_identity = None
    if needs_synthetic:
        assert synthetic_path is not None
        synthetic_windows, synthetic_labels, manifest = load_validated_synthetic_cache(
            synthetic_path,
            sensor_configuration=args.sensor,
            held_out_subject=subject,
            seed=args.seed,
            steps=10,
            samples_per_class=args.samples_per_class,
        )
        cache_identity = manifest["identity"]

    prepared = prepare_data(
        data_root=args.data_root,
        config_path=args.config,
        held_out_subject=subject,
        sensor_configuration=args.sensor,
        seed=args.seed,
        compatibility_mode="filter_before_runs",
    )
    result = evaluate_scenarios(
        prepared.train_windows,
        prepared.train_labels,
        prepared.held_out_test_windows,
        prepared.held_out_test_labels,
        synthetic_windows,
        synthetic_labels,
        classifier=args.classifier,
        scenarios=scenarios,
        channels=prepared.sensor_schema.channel_count,
        seed=args.seed,
        scarce_per_class=args.scarce_per_class,
        synthetic_per_class=args.samples_per_class,
        device=args.device,
    )
    result.update(
        {
            "held_out_subject": subject,
            "sensor_configuration": args.sensor,
            "train_subjects": list(prepared.split.train_subjects),
            "vae_validation_subjects": list(prepared.split.validation_subjects),
            "train_window_count": int(prepared.train_labels.size),
            "vae_validation_window_count": int(prepared.validation_labels.size),
            "test_window_count": int(prepared.held_out_test_labels.size),
            "cache_identity": cache_identity,
            "plan": plan,
            "execution": {
                "status": "fresh",
                "resumed": False,
                "target": str(target) if target else None,
            },
        }
    )
    if target is not None:
        _write_result(target, result, args)
    return result


def _loso_payload(results: Sequence[dict[str, Any]], *, dry_run: bool) -> dict[str, Any]:
    if dry_run:
        return {"folds": list(results), "dry_run": True}
    records = [record for result in results for record in result["records"]]
    return {
        "folds": list(results),
        "summary": summarize_fold_records(records),
        "dry_run": False,
    }


def run_evaluate(args: argparse.Namespace) -> int:
    result = evaluate_one(args, int(args.held_out_subject))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def run_evaluate_loso(args: argparse.Namespace) -> int:
    results = [evaluate_one(args, subject) for subject in CANONICAL_SUBJECTS]
    print(json.dumps(_loso_payload(results, dry_run=args.dry_run), indent=2, sort_keys=True))
    return 0


__all__ = [
    "CANONICAL_SUBJECTS",
    "add_evaluation_parsers",
    "evaluate_one",
    "run_evaluate",
    "run_evaluate_loso",
]