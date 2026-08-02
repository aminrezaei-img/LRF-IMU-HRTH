"""Side-effect-free in-memory preparation for the public data boundary.

The pipeline composes the staged discovery, schema, windowing, split,
normalization, and duplicate-audit primitives.  It deliberately keeps all
participant-derived arrays in memory and exposes only JSON-safe metadata for
reporting.  Persisting that metadata is a separate, explicitly guarded
operation used by the CLI.

This module is a compatibility implementation for the characterized
paper-specific four-class REALDISP path.  It is not an exact-paper-reproduction
claim, and the reconstructed three-channel route is always labelled as a
separate schema requiring separately trained downstream models.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, Union, cast

import numpy as np

from ..config import (
    DEFAULT_CONFIG_PATH,
    PACKAGE_CONFIG_DIR,
    PACKAGE_CONFIG_FILENAMES,
    ConfigError,
    ExperimentConfig,
    load_config,
)
from .activities import (
    ACTIVITY_NAMES,
    ENCODED_LABEL_TO_NAME,
    validate_activity_mapping,
)
from .audit import (
    HISTORICAL_TRAIN_VALIDATION_ONLY,
    PUBLIC_AUDIT_SCOPE,
    audit_window_duplicates,
)
from .normalization import ChannelStandardizer
from .realdisp import RealDISPError, load_realdisp_subjects
from .schema import (
    SensorMode,
    SensorSchema,
    normalize_sensor_mode,
    sensor_schema_for_mode,
)
from .splits import (
    CANONICAL_SUBJECTS,
    VaeSplitResult,
    split_vae_windows,
)
from .windowing import make_windows


METADATA_FILENAME = "prepare_data_metadata.json"
PREPARATION_SCHEMA_VERSION = "3A.prepare-data-summary.1"
DEFAULT_COMPATIBILITY_MODE = "filter_before_runs"
DEFAULT_AUDIT_MODE = PUBLIC_AUDIT_SCOPE

PathValue = Union[str, os.PathLike]


class PreparationError(ValueError):
    """Raised when a preparation request cannot satisfy the public contract."""


@dataclass(frozen=True)
class PreparedData:
    """In-memory prepared fold with a metadata-only public summary.

    The three split arrays and ``windows_by_subject`` are intentionally
    caller-visible for downstream in-process training, but no method on this
    object serializes them.  Use :func:`write_metadata_summary` to persist the
    summary returned by :attr:`summary`.
    """

    config: ExperimentConfig
    config_path: Optional[Path]
    data_root: Path
    sensor_schema: SensorSchema
    available_subjects: Tuple[int, ...]
    windows_by_subject: Mapping[int, np.ndarray]
    labels_by_subject: Mapping[int, np.ndarray]
    split: VaeSplitResult
    train_windows: np.ndarray
    validation_windows: np.ndarray
    held_out_test_windows: np.ndarray
    train_labels: np.ndarray
    validation_labels: np.ndarray
    held_out_test_labels: np.ndarray
    normalizer: Optional[ChannelStandardizer]
    audit: Mapping[str, Any]
    metadata: Mapping[str, Any]

    @property
    def summary(self) -> Dict[str, Any]:
        """Return a detached JSON-safe metadata summary, never signal arrays."""

        return deepcopy(dict(self.metadata))

    def to_metadata(self) -> Dict[str, Any]:
        """Alias for :attr:`summary` used by artifact-writing callers."""

        return self.summary

    def as_dict(self) -> Dict[str, Any]:
        """Return metadata only; array payloads are intentionally omitted."""

        return self.summary


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _package_root() -> Path:
    """Return the source checkout root when the human-facing configs exist."""

    candidate = Path(__file__).resolve().parents[3]
    if (candidate / "configs" / "paper").is_dir():
        return candidate
    return Path(__file__).resolve().parents[1]


def _resolve_config_path(config_path: Optional[PathValue]) -> Path:
    """Resolve checkout evidence or the packaged default configuration."""

    package_root = _package_root()
    if config_path is None:
        source_candidate = (
            package_root / "configs" / "paper" / "six_channel_160_40.yaml"
        ).resolve()
        return source_candidate if source_candidate.is_file() else DEFAULT_CONFIG_PATH.resolve()

    requested = Path(config_path).expanduser()
    if requested.is_absolute():
        return requested.resolve()

    cwd_candidate = (Path.cwd() / requested).resolve()
    if cwd_candidate.is_file():
        return cwd_candidate
    package_candidate = (package_root / requested).resolve()
    if package_candidate.is_file():
        return package_candidate
    if requested.parent.name == "paper" and requested.name in PACKAGE_CONFIG_FILENAMES:
        packaged_candidate = (PACKAGE_CONFIG_DIR / requested.name).resolve()
        if packaged_candidate.is_file():
            return packaged_candidate
    # Keep the candidate in the error raised by load_config. This preserves a
    # useful diagnostic for unsupported external paths.
    return package_candidate


def _config_base_dir(config_path: Path) -> Path:
    package_root = _package_root().resolve()
    resolved = config_path.resolve()
    if _is_relative_to(resolved, PACKAGE_CONFIG_DIR.resolve()):
        return Path.cwd().resolve()
    return package_root if _is_relative_to(resolved, package_root) else resolved.parent


def _path_label(path: Optional[Path], *, package_root: Optional[Path] = None) -> str:
    """Return a repository-relative or opaque external path label."""

    if path is None:
        return "<in-memory-config>"
    resolved = path.expanduser().resolve()
    if _is_relative_to(resolved, PACKAGE_CONFIG_DIR.resolve()):
        return "<packaged-config>/{}".format(resolved.name)
    root = (package_root or _package_root()).resolve()
    if _is_relative_to(resolved, root):
        return resolved.relative_to(root).as_posix()
    name = resolved.name or "unnamed"
    return "<external-config>/{}".format(name)


def _opaque_path_descriptor(path: Path, *, kind: str) -> Dict[str, Any]:
    """Describe a machine path without persisting its absolute spelling."""

    normalized = str(path.expanduser().resolve()).replace("\\", "/").casefold()
    return {
        "kind": kind,
        "path": "<{}>".format(kind),
        "path_sha256": hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
    }


def _canonical_file_sha256(path: Path) -> str:
    """Hash normalized text bytes so CRLF checkout settings do not matter."""

    payload = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(payload).hexdigest()


def _mapping_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise PreparationError("{} must be a positive integer".format(name))
    try:
        integer = int(value)
    except (TypeError, ValueError) as exc:
        raise PreparationError("{} must be a positive integer".format(name)) from exc
    if integer != value or integer <= 0:
        raise PreparationError("{} must be a positive integer".format(name))
    return integer


def _non_negative_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise PreparationError("{} must be a non-negative integer".format(name))
    try:
        integer = int(value)
    except (TypeError, ValueError) as exc:
        raise PreparationError("{} must be a non-negative integer".format(name)) from exc
    if integer != value or integer < 0:
        raise PreparationError("{} must be a non-negative integer".format(name))
    return integer


def _normalise_subject(subject: Any, name: str = "held_out_subject") -> int:
    if isinstance(subject, bool):
        raise PreparationError("{} must be an integer subject ID".format(name))
    if isinstance(subject, str):
        token = subject.strip().lower()
        if token.startswith("subject"):
            token = token[7:]
        if not token.isdigit():
            raise PreparationError("{} must look like 1, 01, or subject01".format(name))
        subject = int(token)
    try:
        integer = int(subject)
    except (TypeError, ValueError) as exc:
        raise PreparationError("{} must be an integer subject ID".format(name)) from exc
    if integer != subject or integer <= 0:
        raise PreparationError("{} must be a positive integer subject ID".format(name))
    return integer


def _normalise_sensor_override(value: Optional[str]) -> Tuple[Optional[SensorMode], Optional[Path]]:
    if value is None:
        return None, None
    if not isinstance(value, str) or not value.strip():
        raise PreparationError("sensor configuration must be a named mode or YAML path")
    candidate = Path(value).expanduser()
    if candidate.is_file() and candidate.suffix.lower() in {".yaml", ".yml"}:
        return None, candidate.resolve()
    package_candidate = (PACKAGE_CONFIG_DIR / candidate.name).resolve()
    if (
        candidate.name in PACKAGE_CONFIG_FILENAMES
        and package_candidate.is_file()
        and package_candidate.suffix.lower() in {".yaml", ".yml"}
    ):
        return None, package_candidate
    try:
        return normalize_sensor_mode(value, strict=False), None
    except Exception as exc:
        raise PreparationError(
            "unsupported sensor configuration {!r}; use six_channel, three_channel, "
            "accelerometer_only, or a YAML config path".format(value)
        ) from exc


def _load_pipeline_config(
    config_path: Optional[PathValue],
    *,
    data_root: Optional[PathValue],
    seed: Optional[int],
    sensor_configuration: Optional[str],
) -> Tuple[ExperimentConfig, Path]:
    selected_path = _resolve_config_path(config_path)
    sensor_mode, sensor_config_path = _normalise_sensor_override(sensor_configuration)
    if sensor_config_path is not None:
        selected_path = sensor_config_path

    base_dir = _config_base_dir(selected_path)
    try:
        config = load_config(
            selected_path,
            base_dir=base_dir,
            data_root=Path(data_root) if data_root is not None else None,
            seed=seed,
        )
    except ConfigError as exc:
        raise PreparationError(str(exc)) from exc

    if sensor_mode is not None:
        schema = sensor_schema_for_mode(sensor_mode)
        try:
            config = config.with_overrides(
                overrides={
                    "sensor.variant": sensor_mode.value,
                    "sensor.channel_indices": list(schema.channel_indices),
                    "sensor.training_mode": "separate_model",
                    "sensor.inference_policy": "declared_channel_set_only",
                    "sensor.allow_channel_drop": False,
                    "channels": list(schema.channel_names),
                    "vae.input_channels": schema.channel_count,
                }
            )
        except ConfigError as exc:
            raise PreparationError("invalid sensor configuration override: {}".format(exc)) from exc

    if config.release.exact_paper_reproduction:
        raise PreparationError(
            "exact_paper_reproduction=true is not permitted by the public preparation boundary"
        )
    return config, selected_path


def _resolve_data_root(config: ExperimentConfig, data_root: Optional[PathValue]) -> Path:
    if data_root is not None:
        root = Path(data_root).expanduser()
        if not root.is_absolute():
            root = Path.cwd() / root
        return root.resolve()
    return config.paths.data_root.resolve()


def _resolve_modes(compatibility_mode: Optional[str]) -> Tuple[str, str]:
    value = compatibility_mode or DEFAULT_COMPATIBILITY_MODE
    if not isinstance(value, str) or not value.strip():
        raise PreparationError("compatibility mode must be a supported string")
    normalized = value.strip().lower().replace("-", "_")
    window_modes = {
        "filter_before_runs": "filter_before_runs",
        "compatibility": "filter_before_runs",
        "default": "filter_before_runs",
        "strict_original_contiguity": "strict_original_contiguity",
        "strict": "strict_original_contiguity",
        "original_contiguity": "strict_original_contiguity",
    }
    audit_modes = {
        "public": PUBLIC_AUDIT_SCOPE,
        PUBLIC_AUDIT_SCOPE: PUBLIC_AUDIT_SCOPE,
        HISTORICAL_TRAIN_VALIDATION_ONLY: HISTORICAL_TRAIN_VALIDATION_ONLY,
        "legacy_train_validation_only": HISTORICAL_TRAIN_VALIDATION_ONLY,
        "train_validation": HISTORICAL_TRAIN_VALIDATION_ONLY,
    }
    if normalized in window_modes:
        return window_modes[normalized], DEFAULT_AUDIT_MODE
    if normalized in audit_modes:
        return DEFAULT_COMPATIBILITY_MODE, audit_modes[normalized]
    raise PreparationError(
        "unsupported compatibility mode {!r}; use filter_before_runs, "
        "strict_original_contiguity, public, or historical_train_validation_only".format(
            compatibility_mode
        )
    )


def _class_counts(labels: np.ndarray) -> Dict[str, int]:
    counts: Dict[str, int] = {name: 0 for name in ACTIVITY_NAMES}
    for label, count in zip(*np.unique(labels, return_counts=True)):
        encoded = int(label)
        name = ENCODED_LABEL_TO_NAME.get(encoded, str(encoded))
        counts[name] = int(count)
    return counts


def _shape(array: np.ndarray) -> list[int]:
    return [int(value) for value in array.shape]


def _normalizer_summary(normalizer: Optional[ChannelStandardizer], *, train_count: int) -> Dict[str, Any]:
    if normalizer is None or not normalizer.fitted:
        return {
            "method": "per_channel_zscore",
            "fit_axes": [0, 2],
            "ddof": 0,
            "minimum_standard_deviation": 1e-8,
            "training_only": True,
            "fit_stage": "post_validation_training_subjects_only",
            "fit_status": "not_fitted_empty_training_split",
            "fit_sample_count": int(train_count),
            "metadata_sha256": _mapping_sha256(
                {"fit_status": "not_fitted_empty_training_split", "fit_sample_count": int(train_count)}
            ),
        }
    full = normalizer.to_metadata()
    metadata_hash = _mapping_sha256(full)
    return {
        "method": full["method"],
        "fit_axes": list(full["fit_axes"]),
        "ddof": int(full["ddof"]),
        "minimum_standard_deviation": float(full["minimum_standard_deviation"]),
        "training_only": bool(full["training_only"]),
        "fit_stage": full["fit_stage"],
        "fit_status": "fitted",
        "fit_sample_count": int(full["fit_sample_count"]),
        "channels": int(full["channels"]),
        "mean_shape": list(full["mean_shape"]),
        "std_shape": list(full["std_shape"]),
        "training_subjects": list(full["training_subjects"] or []),
        "validation_subjects": list(full["validation_subjects"] or []),
        "held_out_subject": full["held_out_subject"],
        "metadata_sha256": metadata_hash,
        "statistics_persisted": False,
    }


def _build_metadata(
    *,
    config: ExperimentConfig,
    config_path: Optional[Path],
    data_root: Path,
    sensor_schema: SensorSchema,
    available_subjects: Tuple[int, ...],
    input_files: Mapping[int, Path],
    windows_by_subject: Mapping[int, np.ndarray],
    labels_by_subject: Mapping[int, np.ndarray],
    split: VaeSplitResult,
    train_windows: np.ndarray,
    validation_windows: np.ndarray,
    held_out_test_windows: np.ndarray,
    train_labels: np.ndarray,
    validation_labels: np.ndarray,
    held_out_test_labels: np.ndarray,
    normalizer: Optional[ChannelStandardizer],
    audit: Mapping[str, Any],
    window_length: int,
    hop_length: int,
    windowing_mode: str,
    audit_mode: str,
) -> Dict[str, Any]:
    config_hash = (
        _canonical_file_sha256(config_path)
        if config_path is not None and config_path.is_file()
        else _mapping_sha256(config.to_mapping())
    )
    file_metadata = {
        "{:02d}".format(int(subject)): {
            "filename": path.name,
            "sha256": _canonical_file_sha256(path),
        }
        for subject, path in sorted(input_files.items())
    }
    per_subject = {
        "{:02d}".format(int(subject)): {
            "window_count": int(windows_by_subject[subject].shape[0]),
            "window_shape": _shape(windows_by_subject[subject])[1:],
            "window_dtype": str(windows_by_subject[subject].dtype),
            "label_count": int(labels_by_subject[subject].shape[0]),
            "class_counts": _class_counts(labels_by_subject[subject]),
        }
        for subject in sorted(windows_by_subject)
    }
    split_metadata = split.metadata.as_dict()
    return {
        "schema_version": PREPARATION_SCHEMA_VERSION,
        "command": "prepare-data",
        "exact_paper_reproduction": False,
        "config": {
            "identity_sha256": config_hash,
            "path": _path_label(config_path),
            "name": config.name,
            "profile": config.release.profile,
            "exact_paper_reproduction": bool(config.release.exact_paper_reproduction),
        },
        "data": {
            "dataset": config.sensor.dataset,
            "placement": config.sensor.placement,
            "sampling_frequency_hz": float(config.sampling_frequency_hz),
            "raw_column_count": 120,
            "data_root": _opaque_path_descriptor(data_root, kind="external-data-root"),
            "available_subjects": ["{:02d}".format(subject) for subject in available_subjects],
            "input_files": file_metadata,
        },
        "sensor": {
            "mode": sensor_schema.mode.value,
            "channel_indices": list(sensor_schema.channel_indices),
            "channel_names": list(sensor_schema.channel_names),
            "channel_count": sensor_schema.channel_count,
            "training_mode": "separate_model",
            "inference_time_channel_drop": False,
            "inference_policy": "declared_channel_set_only",
            "reconstructed_three_channel": bool(sensor_schema.reconstructed_three_channel),
            "three_channel_lineage": (
                "PUBLIC_RECONSTRUCTION_REQUIRED"
                if sensor_schema.is_three_channel
                else "observed_six_channel_schema"
            ),
            "downstream_models": "separate_model_per_declared_channel_set",
        },
        "preprocessing": {
            "compatibility_mode": windowing_mode,
            "filter_before_runs": windowing_mode == "filter_before_runs",
            "complete_windows_only": True,
            "padding": False,
            "window_length": int(window_length),
            "hop_length": int(hop_length),
            "window_seconds": float(window_length / config.sampling_frequency_hz),
            "hop_seconds": float(hop_length / config.sampling_frequency_hz),
            "per_subject": per_subject,
        },
        "split": {
            "protocol": split_metadata["protocol"],
            "validation_unit": split_metadata["validation_unit"],
            # Preserve the historical metadata key while naming both protocols.
            "validation_fraction": split_metadata["validation_fraction"],
            "vae_subject_validation_fraction": float(
                config.split.vae_subject_validation_fraction
            ),
            "classifier_window_validation_fraction": float(
                config.split.classifier_window_validation_fraction
            ),
            "seed": int(split_metadata["seed"]),
            "random_state": split_metadata["random_state"],
            "canonical_cohort": split_metadata["canonical_cohort"],
            "available_subjects": split_metadata["available_subjects"],
            "train_subjects": split_metadata["train_subjects"],
            "validation_subjects": split_metadata["validation_subjects"],
            "held_out_subject": split_metadata["held_out_subject"],
            "window_counts_by_subject": split_metadata["window_counts_by_subject"],
            "counts": {
                "train": int(train_windows.shape[0]),
                "validation": int(validation_windows.shape[0]),
                "held_out_test": int(held_out_test_windows.shape[0]),
            },
            "shapes": {
                "train": _shape(train_windows),
                "validation": _shape(validation_windows),
                "held_out_test": _shape(held_out_test_windows),
            },
            "dtypes": {
                "train": str(train_windows.dtype),
                "validation": str(validation_windows.dtype),
                "held_out_test": str(held_out_test_windows.dtype),
            },
            "label_counts": {
                "train": _class_counts(train_labels),
                "validation": _class_counts(validation_labels),
                "held_out_test": _class_counts(held_out_test_labels),
            },
            "cnn_validation_fraction": float(
                config.split.classifier_window_validation_fraction
            ),
            "cnn_split_not_run": True,
        },
        "normalization": _normalizer_summary(normalizer, train_count=int(train_windows.shape[0])),
        "audit": dict(audit),
        "modes": {
            "compatibility_mode": windowing_mode,
            "audit_mode": audit_mode,
            "audit_scope": audit.get("scope"),
            "audit_public_scope_improvement": bool(
                audit.get("metadata", {}).get("public_scope_improvement", False)
            ),
        },
        "seed": int(config.seed),
        "window": {"length": int(window_length), "hop": int(hop_length)},
        "safety": {
            "raw_signals_persisted": False,
            "participant_derived_windows_persisted": False,
            "raw_values_persisted": False,
            "metadata_only_output": True,
            "allowed_output_files": [METADATA_FILENAME],
            "hashes_only_for_derived_payloads": True,
        },
    }


def prepare_data(
    data_root: Optional[PathValue] = None,
    *,
    config: Optional[ExperimentConfig] = None,
    config_path: Optional[PathValue] = None,
    held_out_subject: Optional[Union[int, str]] = None,
    sensor_configuration: Optional[str] = None,
    seed: Optional[int] = None,
    window_length: Optional[int] = None,
    hop_length: Optional[int] = None,
    compatibility_mode: str = DEFAULT_COMPATIBILITY_MODE,
    audit_mode: Optional[str] = None,
    include_within_split: bool = True,
    raise_on_duplicate: bool = True,
) -> PreparedData:
    """Prepare one safe LOSO fold entirely in memory.

    ``data_root`` is an explicit external input.  The function reads only the
    direct child ``subject*_ideal.log`` files selected by the staged loader,
    produces complete activity-bounded windows, materializes the VAE-safe
    subject split, fits normalization on its training partition, and audits
    exact standardized windows across split boundaries.

    No output path is accepted here and no filesystem write is performed.
    Use :func:`write_metadata_summary` only after obtaining explicit output
    permission at the CLI boundary.
    """

    if config is not None and not isinstance(config, ExperimentConfig):
        raise PreparationError("config must be an ExperimentConfig instance")
    resolved_config_path: Optional[Path]
    if config is None:
        config, resolved_config_path = _load_pipeline_config(
            config_path,
            data_root=Path(data_root) if data_root is not None else None,
            seed=seed,
            sensor_configuration=sensor_configuration,
        )
    else:
        resolved_config_path = config.config_path
        if seed is not None:
            try:
                config = config.with_overrides(seed=_non_negative_int(seed, "seed"))
            except ConfigError as exc:
                raise PreparationError(str(exc)) from exc
        sensor_mode, sensor_config_path = _normalise_sensor_override(sensor_configuration)
        if sensor_config_path is not None:
            config, resolved_config_path = _load_pipeline_config(
                sensor_config_path,
                data_root=Path(data_root) if data_root is not None else None,
                seed=seed,
                sensor_configuration=None,
            )
        elif sensor_mode is not None:
            schema = sensor_schema_for_mode(sensor_mode)
            try:
                config = config.with_overrides(
                    overrides={
                        "sensor.variant": sensor_mode.value,
                        "sensor.channel_indices": list(schema.channel_indices),
                        "sensor.training_mode": "separate_model",
                        "sensor.inference_policy": "declared_channel_set_only",
                        "sensor.allow_channel_drop": False,
                        "channels": list(schema.channel_names),
                        "vae.input_channels": schema.channel_count,
                    }
                )
            except ConfigError as exc:
                raise PreparationError("invalid sensor configuration override: {}".format(exc)) from exc

    if config.release.exact_paper_reproduction:
        raise PreparationError(
            "exact_paper_reproduction=true is not permitted by the public preparation boundary"
        )
    root = _resolve_data_root(config, data_root)
    if not root.exists():
        raise PreparationError("REALDISP data root does not exist: {}".format(root))
    if not root.is_dir():
        raise PreparationError("REALDISP data root is not a directory: {}".format(root))

    mode, default_audit_mode = _resolve_modes(compatibility_mode)
    selected_audit_mode = default_audit_mode if audit_mode is None else audit_mode
    _, selected_audit_mode = _resolve_modes(selected_audit_mode)
    if selected_audit_mode == DEFAULT_AUDIT_MODE:
        selected_audit_mode = PUBLIC_AUDIT_SCOPE
    elif selected_audit_mode == HISTORICAL_TRAIN_VALIDATION_ONLY:
        selected_audit_mode = HISTORICAL_TRAIN_VALIDATION_ONLY
    else:  # pragma: no cover - _resolve_modes constrains this branch.
        raise PreparationError("unsupported audit mode: {}".format(selected_audit_mode))

    sensor_schema = sensor_schema_for_mode(config.sensor.variant, strict=False)
    if tuple(config.sensor.channel_indices) != tuple(sensor_schema.channel_indices):
        raise PreparationError(
            "configuration channel_indices do not match the declared sensor mode {}".format(
                sensor_schema.mode.value
            )
        )
    if tuple(config.channels) != tuple(sensor_schema.channel_names):
        raise PreparationError(
            "configuration channels do not match the declared sensor mode {}".format(
                sensor_schema.mode.value
            )
        )
    validate_activity_mapping()

    selected_window_length = _positive_int(
        config.window.samples if window_length is None else window_length,
        "window_length",
    )
    selected_hop_length = _positive_int(
        config.window.hop if hop_length is None else hop_length,
        "hop_length",
    )
    selected_seed = _non_negative_int(config.seed if seed is None else seed, "seed")

    variant = "three_channel" if sensor_schema.is_three_channel else "six_channel"
    try:
        loaded = load_realdisp_subjects(root, variant=variant)
    except (OSError, ValueError, TypeError, RealDISPError) as exc:
        raise PreparationError("could not discover/load REALDISP logs: {}".format(exc)) from exc
    if not loaded:
        raise PreparationError("no REALDISP subjects were loaded from the data root")

    available_subjects = tuple(sorted(int(subject) for subject in loaded))
    configured_subjects = set(int(subject) for subject in config.subjects)
    unknown_configured = sorted(configured_subjects.difference(CANONICAL_SUBJECTS))
    if unknown_configured:
        raise PreparationError(
            "configuration contains subjects outside the canonical cohort: {}".format(
                unknown_configured
            )
        )
    if any(subject not in CANONICAL_SUBJECTS for subject in available_subjects):
        raise PreparationError(
            "discovered subject IDs are outside the canonical paper cohort: {}".format(
                list(available_subjects)
            )
        )

    if held_out_subject is None:
        held_out = config.split.held_out_subject
    else:
        held_out = _normalise_subject(held_out_subject)
    if held_out is None:
        held_out = available_subjects[0]
    held_out = _normalise_subject(held_out)
    if held_out not in available_subjects:
        raise PreparationError(
            "held-out subject {:02d} was not discovered; available subjects are {}".format(
                held_out, ["{:02d}".format(subject) for subject in available_subjects]
            )
        )

    windows_by_subject: Dict[int, np.ndarray] = {}
    labels_by_subject: Dict[int, np.ndarray] = {}
    input_files = {subject: loaded[subject].path for subject in available_subjects}
    for subject in available_subjects:
        subject_data = loaded[subject]
        windows, labels = make_windows(
            subject_data.signals,
            subject_data.raw_labels,
            window_length=selected_window_length,
            hop_length=selected_hop_length,
            mode=mode,
            channel_count=sensor_schema.channel_count,
        )
        windows_by_subject[subject] = windows
        labels_by_subject[subject] = labels

    try:
        split = split_vae_windows(
            cast(Mapping[Union[int, str], Any], windows_by_subject),
            held_out,
            labels_by_subject=cast(Mapping[Union[int, str], Any], labels_by_subject),
            cohort=available_subjects,
            val_fraction=config.split.vae_subject_validation_fraction,
            seed=selected_seed,
            require_complete_cohort=False,
        )
    except Exception as exc:
        raise PreparationError("could not materialize the VAE-safe subject split: {}".format(exc)) from exc

    train_windows = split.train_windows.astype(np.float32, copy=False)
    validation_windows = split.validation_windows.astype(np.float32, copy=False)
    held_out_test_windows = split.held_out_test_windows.astype(np.float32, copy=False)
    train_labels = (
        split.train_labels.astype(np.int64, copy=False)
        if split.train_labels is not None
        else np.empty((0,), dtype=np.int64)
    )
    validation_labels = (
        split.validation_labels.astype(np.int64, copy=False)
        if split.validation_labels is not None
        else np.empty((0,), dtype=np.int64)
    )
    held_out_test_labels = (
        split.held_out_test_labels.astype(np.int64, copy=False)
        if split.held_out_test_labels is not None
        else np.empty((0,), dtype=np.int64)
    )

    normalizer: Optional[ChannelStandardizer]
    if train_windows.shape[0] > 0:
        normalizer = ChannelStandardizer(channels=sensor_schema.channel_count).fit_training(
            train_windows,
            training_subjects=split.train_subjects,
            validation_subjects=split.validation_subjects,
            held_out_subject=held_out,
        )
        train_windows = normalizer.transform(train_windows)
        validation_windows = normalizer.transform(validation_windows)
        held_out_test_windows = normalizer.transform(held_out_test_windows)
    else:
        # A compact synthetic fixture can be intentionally run with the
        # production 160-sample window.  Preserve its empty [0,C,T] contract
        # and metadata without pretending that statistics were fitted.
        normalizer = None

    try:
        audit = audit_window_duplicates(
            train_windows,
            validation_windows,
            held_out_test_windows,
            include_within_split=include_within_split,
            compatibility_mode=selected_audit_mode,
            raise_on_duplicate=raise_on_duplicate,
        )
    except Exception as exc:
        if hasattr(exc, "summary"):
            raise PreparationError("duplicate audit failed: {}".format(exc)) from exc
        raise PreparationError("duplicate audit could not be completed: {}".format(exc)) from exc

    metadata = _build_metadata(
        config=config,
        config_path=resolved_config_path,
        data_root=root,
        sensor_schema=sensor_schema,
        available_subjects=available_subjects,
        input_files=input_files,
        windows_by_subject=windows_by_subject,
        labels_by_subject=labels_by_subject,
        split=split,
        train_windows=train_windows,
        validation_windows=validation_windows,
        held_out_test_windows=held_out_test_windows,
        train_labels=train_labels,
        validation_labels=validation_labels,
        held_out_test_labels=held_out_test_labels,
        normalizer=normalizer,
        audit=audit,
        window_length=selected_window_length,
        hop_length=selected_hop_length,
        windowing_mode=mode,
        audit_mode=selected_audit_mode,
    )
    return PreparedData(
        config=config,
        config_path=resolved_config_path,
        data_root=root,
        sensor_schema=sensor_schema,
        available_subjects=available_subjects,
        windows_by_subject=windows_by_subject,
        labels_by_subject=labels_by_subject,
        split=split,
        train_windows=train_windows,
        validation_windows=validation_windows,
        held_out_test_windows=held_out_test_windows,
        train_labels=train_labels,
        validation_labels=validation_labels,
        held_out_test_labels=held_out_test_labels,
        normalizer=normalizer,
        audit=audit,
        metadata=metadata,
    )


def prepare_realdisp_data(*args: Any, **kwargs: Any) -> PreparedData:
    """Descriptive alias for :func:`prepare_data`."""

    return prepare_data(*args, **kwargs)


def prepare_realdisp_fold(*args: Any, **kwargs: Any) -> PreparedData:
    """Explicit fold-oriented alias retained for orchestration callers."""

    return prepare_data(*args, **kwargs)


_FORBIDDEN_METADATA_KEYS = frozenset(
    {
        "raw_signals",
        "signals",
        "raw_labels",
        "labels",
        "windows",
        "train_windows",
        "validation_windows",
        "held_out_test_windows",
        "window_values",
        "participant_derived_windows",
        "raw_values",
    }
)


def _validate_metadata_summary(summary: Mapping[str, Any]) -> None:
    if summary.get("schema_version") != PREPARATION_SCHEMA_VERSION:
        raise ValueError("summary is not a 3A prepare-data metadata summary")
    safety = summary.get("safety")
    if not isinstance(safety, Mapping):
        raise ValueError("metadata summary is missing its safety contract")
    required_false = (
        "raw_signals_persisted",
        "participant_derived_windows_persisted",
        "raw_values_persisted",
    )
    if any(safety.get(key) is not False for key in required_false):
        raise ValueError("metadata summary permits participant-derived payload persistence")
    if safety.get("metadata_only_output") is not True:
        raise ValueError("metadata summary is not marked metadata-only")
    if safety.get("hashes_only_for_derived_payloads") is not True:
        raise ValueError("metadata summary is not hash-only for derived payloads")

    def walk(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                if str(key).casefold() in _FORBIDDEN_METADATA_KEYS:
                    raise ValueError(
                        "metadata summary contains a forbidden participant-derived key: {}".format(
                            key
                        )
                    )
                walk(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                walk(child)

    walk(summary)


def write_metadata_summary(
    summary: Mapping[str, Any],
    output_root: PathValue,
    *,
    overwrite: bool = False,
    filename: str = METADATA_FILENAME,
) -> Path:
    """Write exactly one JSON metadata artifact after explicit authorization.

    The caller must provide the output root and call this function explicitly;
    the in-memory pipeline never invokes it.  ``overwrite=False`` uses
    exclusive file creation and therefore cannot silently replace an artifact.
    """

    if not isinstance(summary, Mapping):
        raise TypeError("summary must be a mapping")
    if not isinstance(output_root, (str, os.PathLike, Path)):
        raise TypeError("output_root must be an explicit filesystem path")
    if not isinstance(filename, str) or not filename or Path(filename).name != filename:
        raise ValueError("filename must be a single JSON filename")
    if Path(filename).suffix.lower() != ".json":
        raise ValueError("metadata filename must use the .json extension")
    if not isinstance(overwrite, bool):
        raise TypeError("overwrite must be a boolean")

    _validate_metadata_summary(summary)
    payload = json.dumps(
        dict(summary),
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    )
    root = Path(output_root).expanduser()
    if root.exists() and not root.is_dir():
        raise NotADirectoryError("output_root is not a directory: {}".format(root))
    root.mkdir(parents=True, exist_ok=True)
    target = root / filename
    if target.exists() and not overwrite:
        raise FileExistsError(
            "metadata artifact already exists; pass --overwrite to replace it: {}".format(
                target
            )
        )
    mode = "w" if overwrite else "x"
    with target.open(mode, encoding="utf-8", newline="\n") as handle:
        handle.write(payload)
        handle.write("\n")
    return target


__all__ = [
    "DEFAULT_AUDIT_MODE",
    "DEFAULT_COMPATIBILITY_MODE",
    "METADATA_FILENAME",
    "PREPARATION_SCHEMA_VERSION",
    "PreparedData",
    "PreparationError",
    "prepare_data",
    "prepare_realdisp_data",
    "prepare_realdisp_fold",
    "write_metadata_summary",
]
