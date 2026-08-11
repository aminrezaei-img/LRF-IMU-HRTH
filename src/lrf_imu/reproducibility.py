"""Thin, resumable composition of the accepted public scientific stages.

This module adds orchestration only.  Preparation, checkpoint validation,
paper-profile generation, scenario construction, classifiers, and aggregation
remain owned by their Milestone 3 modules.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np

from .config import ExperimentConfig, load_config
from .data.pipeline import PreparedData, prepare_data
from .evaluation.cache import (
    SyntheticCacheIdentity,
    cache_paths,
    load_validated_synthetic_cache,
    sha256_file,
    write_cache_manifest,
)
from .evaluation.core import evaluate_scenarios
from .evaluation.metrics import summarize_fold_records


CANONICAL_SUBJECTS = (1, 2, 3, 5, 8, 9, 10, 11, 12, 13, 14, 16)
SCENARIO_ORDER = ("trtr", "scarce", "tstr", "tstr_scarce")
MANIFEST_SCHEMA = "m4.reproduce-core-manifest.1"
REPORT_SCHEMA = "m4.reproduce-core-report.1"
IMPLEMENTATION_VERSION = "m4.public-orchestrator.1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _positive(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def add_reproduce_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "reproduce-core",
        help="compose prepare, checkpoint load, paper generation, evaluation, and aggregation",
        description=(
            "Run the accepted public stages without copying data/checkpoints. "
            "A non-dry run requires an explicit output root and --write-results."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-root", required=True, metavar="PATH")
    parser.add_argument("--checkpoint-root", required=True, metavar="PATH")
    parser.add_argument("--output-root", required=True, metavar="PATH")
    parser.add_argument("--config", metavar="PATH")
    parser.add_argument(
        "--sensor", choices=("six_channel", "three_channel"), default="six_channel"
    )
    subject = parser.add_mutually_exclusive_group(required=True)
    subject.add_argument("--held-out-subject", type=_positive, metavar="ID")
    subject.add_argument("--all-folds", action="store_true")
    parser.add_argument("--classifier", choices=("rf", "cnn"), default="rf")
    parser.add_argument(
        "--scenario",
        action="append",
        choices=SCENARIO_ORDER,
        help="repeat to select scenarios; the default runs all four",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples-per-class", type=_positive, default=500)
    parser.add_argument("--scarce-per-class", type=_positive, default=2)
    parser.add_argument(
        "--vae-checkpoint",
        metavar="PATH",
        help="single-fold explicit override; otherwise canonical paths below --checkpoint-root are used",
    )
    parser.add_argument(
        "--flow-checkpoint",
        metavar="PATH",
        help="single-fold explicit override; otherwise canonical paths below --checkpoint-root are used",
    )
    parser.add_argument(
        "--reference-report",
        metavar="JSON",
        help="optional M3D fold-reference report to compare without declaring tolerance-based parity",
    )
    parser.add_argument("--write-results", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")


def _sensor_layout(sensor: str) -> tuple[str, str, int, str]:
    if sensor == "six_channel":
        return "6CH", "full", 6, "six_channel_160_40.yaml"
    if sensor == "three_channel":
        return "3CH", "ablation", 3, "accelerometer_only_160_40.yaml"
    raise ValueError("sensor must be six_channel or three_channel")


def _subjects(args: argparse.Namespace) -> tuple[int, ...]:
    values = CANONICAL_SUBJECTS if args.all_folds else (int(args.held_out_subject),)
    unknown = sorted(set(values).difference(CANONICAL_SUBJECTS))
    if unknown:
        raise ValueError(
            "held-out subject is not in the canonical cohort: {}".format(unknown)
        )
    return tuple(values)


def _scenarios(args: argparse.Namespace) -> tuple[str, ...]:
    requested = tuple(args.scenario or SCENARIO_ORDER)
    return tuple(name for name in SCENARIO_ORDER if name in requested)


def _needs_synthetic(scenarios: Iterable[str]) -> bool:
    return any(name in {"tstr", "tstr_scarce"} for name in scenarios)


def _selected_config(args: argparse.Namespace) -> tuple[ExperimentConfig, Path, str]:
    _, _, channels, default_name = _sensor_layout(args.sensor)
    request = args.config or "configs/paper/{}".format(default_name)
    config = load_config(request, base_dir=Path.cwd(), seed=args.seed)
    if int(config.vae.input_channels) != channels:
        raise ValueError(
            "configuration has {} VAE channels but {} requires {}".format(
                config.vae.input_channels, args.sensor, channels
            )
        )
    if config.release.exact_paper_reproduction:
        raise ValueError("reproduce-core refuses exact_paper_reproduction=true")
    if config.config_path is None:
        raise ValueError("configuration has no resolved source path")
    path = config.config_path.resolve()
    return config, path, sha256_file(path)


def _checkpoint_candidates(
    root: Path, sensor: str, subject: int, kind: str
) -> tuple[Path, ...]:
    code, profile, _, _ = _sensor_layout(sensor)
    filename = "vae_s4_z48.pt" if kind == "vae" else "flow_unet_best.pt"
    family = "vae_weights" if kind == "vae" else "flow_weights"
    suffix = Path(family) / code / profile / "subject_{:02d}".format(subject) / filename
    return tuple(
        dict.fromkeys(
            (
                root / "Results" / "model_weights" / suffix,
                root / "model_weights" / suffix,
                root / suffix,
            )
        )
    )


def _checkpoint_path(
    args: argparse.Namespace,
    subject: int,
    kind: str,
    *,
    require_exists: bool,
) -> Path:
    explicit = getattr(args, "{}_checkpoint".format(kind), None)
    if explicit:
        if args.all_folds:
            raise ValueError("explicit checkpoint overrides are single-fold only")
        candidate = Path(explicit).expanduser().resolve()
        if require_exists and not candidate.is_file():
            raise FileNotFoundError(
                "{} checkpoint does not exist: {}".format(kind, candidate)
            )
        return candidate
    root = Path(args.checkpoint_root).expanduser().resolve()
    candidates = _checkpoint_candidates(root, args.sensor, subject, kind)
    existing = [path.resolve() for path in candidates if path.is_file()]
    if len(existing) > 1 and len({str(path) for path in existing}) > 1:
        raise ValueError(
            "multiple canonical {} checkpoints found for subject {:02d}: {}".format(
                kind, subject, existing
            )
        )
    if existing:
        return existing[0]
    if require_exists:
        raise FileNotFoundError(
            "{} checkpoint not found for subject {:02d}; checked {}".format(
                kind, subject, [str(path) for path in candidates]
            )
        )
    return candidates[0].resolve()


def _runtime_metadata(device: str) -> dict[str, Any]:
    versions: dict[str, Optional[str]] = {"numpy": np.__version__}
    try:
        import sklearn  # type: ignore[import-untyped]

        versions["scikit_learn"] = sklearn.__version__
    except ImportError:
        versions["scikit_learn"] = None
    try:
        import torch

        versions["torch"] = torch.__version__
        cuda = {
            "available": bool(torch.cuda.is_available()),
            "runtime": torch.version.cuda,
        }
    except ImportError:
        versions["torch"] = None
        cuda = {"available": False, "runtime": None}
    return {
        "python": sys.version.split()[0],
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "requested_device": device,
        "versions": versions,
        "cuda": cuda,
    }


def _json_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_npz(path: Path, windows: np.ndarray, labels: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, X_syn=windows, y_syn=labels)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _plan(
    args: argparse.Namespace, *, config_path: Path, config_sha: str
) -> dict[str, Any]:
    subjects = _subjects(args)
    scenarios = _scenarios(args)
    plan = {
        "schema_version": "m4.reproduce-core-plan.1",
        "sensor": args.sensor,
        "subjects": list(subjects),
        "classifier": args.classifier,
        "scenarios": list(scenarios),
        "seed": int(args.seed),
        "samples_per_class": int(args.samples_per_class),
        "scarce_per_class": int(args.scarce_per_class),
        "device": args.device,
        "data_root": str(Path(args.data_root).expanduser().resolve()),
        "checkpoint_root": str(Path(args.checkpoint_root).expanduser().resolve()),
        "output_root": str(Path(args.output_root).expanduser().resolve()),
        "config": {"path": str(config_path), "sha256": config_sha},
        "reference_report": (
            str(Path(args.reference_report).expanduser().resolve())
            if args.reference_report
            else None
        ),
        "write_permission_requested": bool(args.write_results),
        "resume_requested": bool(args.resume),
        "dry_run": bool(args.dry_run),
        "folds": [],
    }
    for subject in subjects:
        fold: dict[str, Any] = {
            "held_out_subject": subject,
            "vae_checkpoint": None,
            "flow_checkpoint": None,
            "synthetic_generation_required": _needs_synthetic(scenarios),
        }
        if _needs_synthetic(scenarios):
            fold["vae_checkpoint"] = str(
                _checkpoint_path(args, subject, "vae", require_exists=False)
            )
            fold["flow_checkpoint"] = str(
                _checkpoint_path(args, subject, "flow", require_exists=False)
            )
        plan["folds"].append(fold)
    fingerprint_input = {
        key: plan[key]
        for key in (
            "sensor",
            "subjects",
            "classifier",
            "scenarios",
            "seed",
            "samples_per_class",
            "scarce_per_class",
            "device",
            "data_root",
            "checkpoint_root",
            "output_root",
            "config",
            "reference_report",
        )
    }
    plan["run_fingerprint"] = _json_hash(fingerprint_input)
    return plan


def _new_manifest(plan: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": MANIFEST_SCHEMA,
        "implementation_version": IMPLEMENTATION_VERSION,
        "exact_paper_reproduction": False,
        "run_fingerprint": plan["run_fingerprint"],
        "status": "running",
        "started_at": _utc_now(),
        "updated_at": _utc_now(),
        "completed_at": None,
        "runtime": _runtime_metadata(str(plan["device"])),
        "plan": dict(plan),
        "folds": {},
        "attempts": [],
        "report_path": None,
    }


def _load_resume_manifest(path: Path, plan: Mapping[str, Any]) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            "--resume requires an existing manifest: {}".format(path)
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != MANIFEST_SCHEMA:
        raise ValueError("resume manifest has an unsupported schema")
    if payload.get("implementation_version") != IMPLEMENTATION_VERSION:
        raise ValueError(
            "resume manifest has an incompatible implementation version"
        )
    if payload.get("run_fingerprint") != plan.get("run_fingerprint"):
        raise ValueError("resume manifest does not match the requested run")
    if not isinstance(payload.get("folds"), dict):
        raise ValueError("resume manifest has no fold state")
    payload["status"] = "running"
    payload["completed_at"] = None
    payload["updated_at"] = _utc_now()
    return payload


def _write_cache_manifest_atomic(
    path: Path,
    identity: SyntheticCacheIdentity,
    *,
    array_path: Path,
    metadata: Mapping[str, Any],
) -> None:
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        temporary.unlink()
    write_cache_manifest(
        temporary,
        identity,
        array_path=array_path,
        extra=metadata,
        overwrite=False,
    )
    os.replace(temporary, path)


def _materialize_synthetic(
    args: argparse.Namespace,
    *,
    subject: int,
    config_sha: str,
    cache_root: Path,
    resume: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    import torch

    from .checkpoints import load_flow_checkpoint, load_vae_checkpoint
    from .generation.flow import generate_paper_latents

    _, _, channels, _ = _sensor_layout(args.sensor)
    vae_checkpoint = _checkpoint_path(args, subject, "vae", require_exists=True)
    flow_checkpoint = _checkpoint_path(args, subject, "flow", require_exists=True)
    vae_sha = sha256_file(vae_checkpoint)
    flow_sha = sha256_file(flow_checkpoint)
    identity = SyntheticCacheIdentity(
        sensor_configuration=args.sensor,
        held_out_subject=subject,
        vae_checkpoint_sha256=vae_sha,
        flow_checkpoint_sha256=flow_sha,
        config_identity=config_sha,
        seed=int(args.seed),
        steps=10,
        samples_per_class=int(args.samples_per_class),
        implementation_version=IMPLEMENTATION_VERSION,
    )
    locations = cache_paths(cache_root, identity)
    array_path = locations["array"]
    manifest_path = locations["manifest"]
    if resume and array_path.is_file() and manifest_path.is_file():
        windows, labels, cache_manifest = load_validated_synthetic_cache(
            array_path,
            sensor_configuration=args.sensor,
            held_out_subject=subject,
            seed=args.seed,
            steps=10,
            samples_per_class=args.samples_per_class,
        )
        return (
            windows,
            labels,
            {
                "status": "resumed",
                "array_path": str(array_path),
                "manifest_path": str(manifest_path),
                "identity": cache_manifest["identity"],
                "array_sha256": cache_manifest["array_sha256"],
            },
        )

    flow, flow_inspection = load_flow_checkpoint(
        flow_checkpoint,
        channels=channels,
        latent_channels=48,
        num_classes=4,
        width_profile="historical_checkpoint_compatibility_256",
        device=args.device,
    )
    vae, vae_inspection = load_vae_checkpoint(
        vae_checkpoint,
        channels=channels,
        latent_channels=48,
        down_levels=2,
        device=args.device,
    )
    latents, generated_labels, sampling = generate_paper_latents(
        flow,
        samples_per_class=args.samples_per_class,
        num_steps=10,
        seed=args.seed,
        latent_shape=(48, 40),
        device=args.device,
    )
    with torch.no_grad():
        windows_tensor = vae.decode(latents)
    windows = windows_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
    labels_array = (
        generated_labels.detach().cpu().numpy().astype(np.int64, copy=False)
    )
    _atomic_npz(array_path, windows, labels_array)
    _write_cache_manifest_atomic(
        manifest_path,
        identity,
        array_path=array_path,
        metadata={
            "sampling": sampling,
            "decoded_shape": list(windows.shape),
            "flow_checkpoint": flow_inspection.to_mapping(),
            "vae_checkpoint": vae_inspection.to_mapping(),
            "coordinate_system": "standardized_no_inverse",
        },
    )
    return (
        windows,
        labels_array,
        {
            "status": "fresh",
            "array_path": str(array_path),
            "manifest_path": str(manifest_path),
            "identity": identity.as_dict(),
            "array_sha256": sha256_file(array_path),
        },
    )


def _evaluate_fold(
    args: argparse.Namespace,
    *,
    prepared: PreparedData,
    subject: int,
    synthetic_windows: Optional[np.ndarray],
    synthetic_labels: Optional[np.ndarray],
) -> dict[str, Any]:
    result = evaluate_scenarios(
        prepared.train_windows,
        prepared.train_labels,
        prepared.held_out_test_windows,
        prepared.held_out_test_labels,
        synthetic_windows,
        synthetic_labels,
        classifier=args.classifier,
        scenarios=_scenarios(args),
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
        }
    )
    return result


def _reference_comparison(
    reference_path: Optional[str],
    *,
    sensor: str,
    classifier: str,
    folds: Sequence[Mapping[str, Any]],
    samples_per_class: int = 500,
) -> Optional[dict[str, Any]]:
    if reference_path is None:
        return None
    path = Path(reference_path).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    gate_name = {
        ("six_channel", "rf"): "gate_c_6ch_rf",
        ("three_channel", "rf"): "gate_d_3ch_rf",
        ("six_channel", "cnn"): "gate_e_6ch_cnn",
    }.get((sensor, classifier))
    if gate_name is None or not isinstance(payload.get(gate_name), dict):
        return {
            "status": "reference_unavailable_for_requested_profile",
            "source_path": str(path),
            "source_sha256": sha256_file(path),
            "comparisons": [],
        }
    reference_folds = {
        int(item["held_out_subject"]): item
        for item in payload[gate_name].get("folds", [])
    }
    comparisons = []
    for fold in folds:
        subject = int(fold["held_out_subject"])
        source_fold = reference_folds.get(subject)
        if source_fold is None:
            continue
        records = {item["scenario"]: item for item in fold["records"]}
        for scenario, current in records.items():
            source_scenario = source_fold.get("scenarios", {}).get(scenario, {})
            for metric in ("f1_macro", "accuracy", "retention_ratio"):
                source_metric = source_scenario.get(metric)
                if (
                    not isinstance(source_metric, dict)
                    or source_metric.get("reference") is None
                ):
                    continue
                historical = float(source_metric["reference"])
                regenerated = float(current[metric])
                comparisons.append(
                    {
                        "held_out_subject": subject,
                        "scenario": scenario,
                        "metric": metric,
                        "historical_reference": historical,
                        "current_run": regenerated,
                        "signed_difference": regenerated - historical,
                        "absolute_difference": abs(regenerated - historical),
                        "comparison_basis": source_metric.get("comparison_basis"),
                    }
                )
    same_sample_protocol = int(samples_per_class) == 500
    return {
        "status": (
            "compared_without_parity_threshold"
            if same_sample_protocol
            else "protocol_mismatch_descriptive_only"
        ),
        "source_path": str(path),
        "source_sha256": sha256_file(path),
        "gate": gate_name,
        "historical_samples_per_class": 500,
        "current_samples_per_class": int(samples_per_class),
        "comparison_count": len(comparisons),
        "comparisons": comparisons,
        "parity_claim": False,
    }


def _fold_result_path(
    output_root: Path, args: argparse.Namespace, subject: int
) -> Path:
    return (
        output_root
        / "evaluation"
        / args.sensor
        / args.classifier
        / "subject_{:02d}.json".format(subject)
    )


def _resume_completed_fold(
    manifest: Mapping[str, Any],
    output_root: Path,
    args: argparse.Namespace,
    subject: int,
) -> Optional[dict[str, Any]]:
    state = manifest.get("folds", {}).get("subject_{:02d}".format(subject))
    if not isinstance(state, Mapping) or state.get("status") != "completed":
        return None
    if _needs_synthetic(_scenarios(args)):
        cache = state.get("synthetic_cache")
        identity = cache.get("identity") if isinstance(cache, Mapping) else None
        if not isinstance(identity, Mapping):
            return None
        expected = {
            "vae_checkpoint_sha256": sha256_file(
                _checkpoint_path(args, subject, "vae", require_exists=True)
            ),
            "flow_checkpoint_sha256": sha256_file(
                _checkpoint_path(args, subject, "flow", require_exists=True)
            ),
            "implementation_version": IMPLEMENTATION_VERSION,
        }
        if any(identity.get(key) != value for key, value in expected.items()):
            return None
    target = _fold_result_path(output_root, args, subject)
    if not target.is_file() or sha256_file(target) != state.get("result_sha256"):
        return None
    result = json.loads(target.read_text(encoding="utf-8"))
    result["execution"] = {"status": "resumed", "result_path": str(target)}
    return result


def run_reproduce_core(args: argparse.Namespace) -> int:
    config, config_path, config_sha = _selected_config(args)
    plan = _plan(args, config_path=config_path, config_sha=config_sha)
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    if not args.write_results:
        raise ValueError(
            "non-dry reproduce-core requires explicit --write-results permission"
        )

    output_root = Path(args.output_root).expanduser().resolve()
    manifest_path = output_root / "reproduce_core_manifest.json"
    report_path = output_root / "reproduce_core_report.json"
    if output_root.exists() and not output_root.is_dir():
        raise NotADirectoryError(
            "output root is not a directory: {}".format(output_root)
        )
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = (
        _load_resume_manifest(manifest_path, plan)
        if args.resume
        else _new_manifest(plan)
    )
    if manifest_path.exists() and not args.resume:
        raise FileExistsError("run manifest exists; use --resume or a new output root")
    _atomic_json(manifest_path, manifest)

    fold_results: list[dict[str, Any]] = []
    run_started = time.perf_counter()
    try:
        for subject in _subjects(args):
            fold_key = "subject_{:02d}".format(subject)
            resumed = _resume_completed_fold(manifest, output_root, args, subject)
            if resumed is not None:
                fold_results.append(resumed)
                manifest["attempts"].append(
                    {
                        "held_out_subject": subject,
                        "attempt": "resume",
                        "status": "skipped_completed",
                        "at": _utc_now(),
                    }
                )
                manifest["updated_at"] = _utc_now()
                _atomic_json(manifest_path, manifest)
                continue

            previous = manifest.get("folds", {}).get(fold_key, {})
            attempt = int(previous.get("attempt_count", 0)) + 1
            fold_started = time.perf_counter()
            state = {
                "status": "running",
                "attempt_count": attempt,
                "started_at": _utc_now(),
                "completed_at": None,
                "failure": None,
            }
            manifest["folds"][fold_key] = state
            manifest["attempts"].append(
                {
                    "held_out_subject": subject,
                    "attempt": attempt,
                    "status": "started",
                    "at": state["started_at"],
                }
            )
            manifest["updated_at"] = _utc_now()
            _atomic_json(manifest_path, manifest)

            prepared = prepare_data(
                data_root=args.data_root,
                config_path=config_path,
                held_out_subject=subject,
                sensor_configuration=args.sensor,
                seed=args.seed,
                compatibility_mode="filter_before_runs",
            )
            synthetic_windows = None
            synthetic_labels = None
            cache_record = None
            if _needs_synthetic(_scenarios(args)):
                synthetic_windows, synthetic_labels, cache_record = (
                    _materialize_synthetic(
                        args,
                        subject=subject,
                        config_sha=config_sha,
                        cache_root=output_root / "generated",
                        resume=bool(args.resume),
                    )
                )
            result = _evaluate_fold(
                args,
                prepared=prepared,
                subject=subject,
                synthetic_windows=synthetic_windows,
                synthetic_labels=synthetic_labels,
            )
            result["execution"] = {
                "status": "fresh",
                "attempt": attempt,
                "duration_seconds": time.perf_counter() - fold_started,
            }
            target = _fold_result_path(output_root, args, subject)
            _atomic_json(target, result)
            state.update(
                {
                    "status": "completed",
                    "completed_at": _utc_now(),
                    "duration_seconds": time.perf_counter() - fold_started,
                    "preparation": {
                        "config_sha256": config_sha,
                        "train_windows": int(prepared.train_labels.size),
                        "vae_validation_windows": int(prepared.validation_labels.size),
                        "test_windows": int(prepared.held_out_test_labels.size),
                        "metadata_only": True,
                    },
                    "synthetic_cache": cache_record,
                    "result_path": str(target),
                    "result_sha256": sha256_file(target),
                }
            )
            fold_results.append(result)
            manifest["updated_at"] = _utc_now()
            _atomic_json(manifest_path, manifest)

        records = [record for fold in fold_results for record in fold["records"]]
        report = {
            "schema_version": REPORT_SCHEMA,
            "exact_paper_reproduction": False,
            "run_fingerprint": plan["run_fingerprint"],
            "sensor": args.sensor,
            "classifier": args.classifier,
            "seed": int(args.seed),
            "subjects": list(_subjects(args)),
            "scenarios": list(_scenarios(args)),
            "folds": fold_results,
            "summary": summarize_fold_records(records),
            "reference_comparison": _reference_comparison(
                args.reference_report,
                sensor=args.sensor,
                classifier=args.classifier,
                folds=fold_results,
                samples_per_class=args.samples_per_class,
            ),
            "duration_seconds": time.perf_counter() - run_started,
            "safety": {
                "data_copied": False,
                "checkpoints_copied": False,
                "source_repository_required": False,
                "generated_arrays_location": str(output_root / "generated"),
                "explicit_write_permission": True,
            },
        }
        _atomic_json(report_path, report)
        manifest["status"] = "completed"
        manifest["completed_at"] = _utc_now()
        manifest["updated_at"] = _utc_now()
        manifest["duration_seconds"] = time.perf_counter() - run_started
        manifest["report_path"] = str(report_path)
        manifest["report_sha256"] = sha256_file(report_path)
        _atomic_json(manifest_path, manifest)
        print(
            json.dumps(
                {
                    "command": "reproduce-core",
                    "status": "completed",
                    "manifest_path": str(manifest_path),
                    "report_path": str(report_path),
                    "fold_count": len(fold_results),
                    "summary": report["summary"],
                    "exact_paper_reproduction": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    except BaseException as exc:
        if isinstance(exc, (SystemExit, GeneratorExit)):
            raise
        manifest["status"] = (
            "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        )
        manifest["updated_at"] = _utc_now()
        manifest["failure"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "at": _utc_now(),
            "retry_with_resume": True,
        }
        running = [
            value
            for value in manifest.get("folds", {}).values()
            if isinstance(value, dict) and value.get("status") == "running"
        ]
        for value in running:
            value["status"] = manifest["status"]
            value["failure"] = dict(manifest["failure"])
        _atomic_json(manifest_path, manifest)
        raise


__all__ = [
    "CANONICAL_SUBJECTS",
    "IMPLEMENTATION_VERSION",
    "MANIFEST_SCHEMA",
    "REPORT_SCHEMA",
    "add_reproduce_parser",
    "run_reproduce_core",
]
