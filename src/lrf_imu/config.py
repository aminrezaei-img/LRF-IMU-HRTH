"""Central, portable configuration for the initial LRF-IMU release.

This module intentionally stops at configuration and path resolution.  It does
not import model, preprocessing, evaluation, or hardware-specific code.  The
YAML files are therefore useful to later migration work without silently
claiming that a release default is an exact reproduction of the paper.
"""

from copy import deepcopy
from dataclasses import fields, is_dataclass
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import yaml

from .paths import PathLike, ProjectPaths, paths_from_mapping


DEFAULT_CONFIG_PATH = Path("configs/paper/six_channel_160_40.yaml")


class ConfigError(ValueError):
    """Raised when a release configuration is missing or internally invalid."""


class EvidenceTier(str, Enum):
    """Evidence labels used by the audit and the release configurations."""

    MANUSCRIPT_REPORTED = "manuscript_reported"
    OBSERVED_WRAPPER = "observed_wrapper"
    OBSERVED_CHECKPOINT_METADATA = "observed_checkpoint_metadata"
    RELEASE_DEFAULT = "release_default"


REQUIRED_EVIDENCE_TIERS = tuple(item.value for item in EvidenceTier)


def _mapping(value: Any, name: str) -> Dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ConfigError("{} must be a mapping".format(name))
    return dict(value)


def _tuple_of(value: Any, name: str, cast: Any = str) -> Tuple[Any, ...]:
    if value is None:
        return tuple()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ConfigError("{} must be a sequence".format(name))
    try:
        return tuple(cast(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ConfigError("{} contains an invalid value".format(name)) from exc


def _positive_int(value: Any, name: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError("{} must be an integer".format(name)) from exc
    if result <= 0:
        raise ConfigError("{} must be positive".format(name))
    return result


def _non_negative_int(value: Any, name: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError("{} must be an integer".format(name)) from exc
    if result < 0:
        raise ConfigError("{} must not be negative".format(name))
    return result


def _float(value: Any, name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError("{} must be numeric".format(name)) from exc


def _bool(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1"}:
            return True
        if lowered in {"false", "no", "0"}:
            return False
    raise ConfigError("{} must be a boolean".format(name))


@dataclass(frozen=True)
class ReleaseMetadata:
    """How a configuration should be described in public documentation."""

    profile: str = "compatibility_default"
    exact_paper_reproduction: bool = False
    note: str = (
        "Compatibility defaults are evidence-labeled and are not an exact "
        "paper-reproduction claim."
    )


@dataclass(frozen=True)
class EvidenceConfig:
    """Evidence tiers and unresolved discrepancies retained with a config."""

    tiers: Dict[str, str] = field(default_factory=dict)
    defaults: Dict[str, Any] = field(default_factory=dict)
    conflicts: Dict[str, Any] = field(default_factory=dict)
    notes: Tuple[str, ...] = tuple()

    def source_for(self, parameter: str) -> Optional[str]:
        """Return the evidence tier recorded for a parameter, if present."""

        value = self.defaults.get(parameter)
        if isinstance(value, Mapping):
            value = value.get("tier")
        return str(value) if value is not None else None


@dataclass(frozen=True)
class SensorConfig:
    dataset: str = "REALDISP"
    placement: str = "ideal"
    name: str = "right_thigh"
    label_column: int = 119
    channel_indices: Tuple[int, ...] = tuple()
    variant: str = "six_channel"
    training_mode: str = "separate_model"
    inference_policy: str = "declared_channel_set_only"
    allow_channel_drop: bool = False


@dataclass(frozen=True)
class ActivitySpec:
    code: int
    name: str


@dataclass(frozen=True)
class WindowConfig:
    samples: int = 160
    hop: int = 40
    grid_samples: Tuple[int, ...] = tuple()
    grid_hops: Tuple[int, ...] = tuple()


@dataclass(frozen=True)
class NormalizationConfig:
    method: str = "zscore"
    fit_on: str = "training_subjects"
    axes: Tuple[str, ...] = ("sample", "time")
    apply_to_synthetic: bool = True


@dataclass(frozen=True)
class SplitConfig:
    protocol: str = "leave_one_subject_out"
    held_out_subject: Optional[int] = None
    fold_id: Optional[int] = None
    validation_fraction: float = 0.2


@dataclass(frozen=True)
class SensitivityGridConfig:
    enabled: bool = False
    window_samples: Tuple[int, ...] = tuple()
    hop_samples: Tuple[int, ...] = tuple()
    train_separately_per_setting: bool = True


@dataclass(frozen=True)
class AugmentationConfig:
    enabled: bool = True
    jitter: float = 0.008
    scale: float = 0.04
    time_mask: float = 0.05


@dataclass(frozen=True)
class VAETrainingConfig:
    batch_size: int = 256
    learning_rate: float = 0.001
    max_epochs: int = 1000
    early_stop_min_epochs: int = 200
    early_stop_patience: int = 100
    use_amp_bf16: bool = True
    grad_clip: float = 1.0
    l2_weight: float = 0.5
    l1_weight: float = 0.1
    beta_init: float = 0.08
    beta_min: float = 0.04
    beta_decay: float = 0.995
    use_spectral_loss: bool = False
    fft_weight: float = 0.0
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)


@dataclass(frozen=True)
class VAEConfig:
    input_channels: int = 6
    latent_dim_channels: int = 48
    latent_stride: int = 4
    latent_time_steps: int = 40
    deterministic_reconstruction_pass: bool = True
    schedule_profile: str = "observed_wrapper_compatibility"
    training: VAETrainingConfig = field(default_factory=VAETrainingConfig)


@dataclass(frozen=True)
class FlowTrainingConfig:
    epochs: int = 300
    learning_rate: float = 0.0005
    early_stop_patience: int = 50
    grad_clip: float = 1.0
    optimizer: str = "AdamW"
    optimizer_betas: Tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.0001
    batch_size: int = 128
    auto_batch: bool = True


@dataclass(frozen=True)
class FlowArchitectureConfig:
    channel_multipliers: Tuple[int, ...] = (1, 2, 4)
    residual_block_kernel_short: int = 3
    residual_block_kernel_long: int = 31
    normalization_groups: int = 8
    squeeze_excitation_reduction: int = 4
    downsampling: str = "avg_pool_factor_2"
    upsampling: str = "nearest_factor_2"
    num_classes: int = 4


@dataclass(frozen=True)
class FlowConfig:
    latent_dim_channels: int = 48
    latent_stride: int = 4
    base_width: int = 256
    width_profile: str = "observed_wrapper_compatibility"
    training: FlowTrainingConfig = field(default_factory=FlowTrainingConfig)
    architecture: FlowArchitectureConfig = field(default_factory=FlowArchitectureConfig)

    @property
    def model_ch(self) -> int:
        """Compatibility alias for the observed wrapper/checkpoint metadata."""

        return self.base_width


@dataclass(frozen=True)
class SamplingConfig:
    ode_solver: str = "Euler"
    steps: int = 10
    start_time: float = 1.0
    end_time: float = 0.0
    windows_per_class: int = 500
    seed: Optional[int] = 42


@dataclass(frozen=True)
class RandomForestConfig:
    estimator: str = "RandomForestClassifier"
    n_estimators: int = 100
    random_state: int = 42
    n_jobs: int = 1
    input_representation: str = "flattened_standardized_window"


@dataclass(frozen=True)
class CNNConfig:
    conv_channels: Tuple[int, int, int] = (32, 64, 128)
    kernel_size: int = 5
    pool_kernel_size: int = 2
    fc_hidden: Tuple[int, int] = (256, 128)
    dropout: float = 0.3
    epochs: int = 80
    patience: int = 10
    batch_size: int = 64
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    validation_fraction: float = 0.2


@dataclass(frozen=True)
class ClassifierConfig:
    random_forest: RandomForestConfig = field(default_factory=RandomForestConfig)
    cnn: CNNConfig = field(default_factory=CNNConfig)
    primary: str = "random_forest"


@dataclass(frozen=True)
class RuntimeConfig:
    device: str = "auto"


@dataclass(frozen=True)
class ExperimentConfig:
    """Validated configuration object consumed by later pipeline migrations."""

    name: str
    data_root: Path
    output_root: Path
    checkpoint_root: Path
    results_root: Path
    sensor: SensorConfig
    channels: Tuple[str, ...]
    subjects: Tuple[int, ...]
    activity_codes: Tuple[int, ...]
    activity_mapping: Dict[int, ActivitySpec]
    sampling_frequency_hz: float
    window: WindowConfig
    normalization: NormalizationConfig
    seed: int
    vae: VAEConfig
    flow: FlowConfig
    sampling: SamplingConfig
    classifiers: ClassifierConfig
    split: SplitConfig
    runtime: RuntimeConfig
    sensitivity_grid: SensitivityGridConfig
    release: ReleaseMetadata
    evidence: EvidenceConfig
    config_path: Optional[Path] = field(default=None, compare=False, repr=False)
    base_dir: Optional[Path] = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        if not self.name:
            raise ConfigError("name must not be empty")
        if not self.channels:
            raise ConfigError("channels must not be empty")
        if len(self.channels) != self.vae.input_channels:
            raise ConfigError(
                "vae.input_channels ({}) must equal the declared channel count ({})".format(
                    self.vae.input_channels, len(self.channels)
                )
            )
        if len(self.sensor.channel_indices) != len(self.channels):
            raise ConfigError(
                "sensor.channel_indices must have one entry per declared channel"
            )
        if self.sensor.allow_channel_drop:
            raise ConfigError(
                "channel-dropping inference is not supported; train the declared "
                "channel variant separately"
            )
        if not self.subjects or len(set(self.subjects)) != len(self.subjects):
            raise ConfigError("subjects must be a non-empty list of unique IDs")
        if len(self.activity_mapping) != len(self.activity_codes):
            raise ConfigError("activity_mapping and activity_codes must have equal length")
        mapped_codes = tuple(
            self.activity_mapping[index].code
            for index in sorted(self.activity_mapping)
        )
        if mapped_codes != self.activity_codes:
            raise ConfigError(
                "activity_mapping codes must match activity_codes in encoded-class order"
            )
        if self.sampling_frequency_hz <= 0:
            raise ConfigError("sampling_frequency_hz must be positive")
        if self.window.samples <= 0 or self.window.hop <= 0:
            raise ConfigError("window.samples and window.hop must be positive")
        if self.window.hop > self.window.samples:
            raise ConfigError("window.hop must not exceed window.samples")
        if self.split.held_out_subject is not None and self.split.held_out_subject not in self.subjects:
            raise ConfigError("split.held_out_subject must be one of subjects")
        if self.split.fold_id is not None and self.split.fold_id not in self.subjects:
            raise ConfigError("split.fold_id must be one of subjects")
        missing_tiers = set(REQUIRED_EVIDENCE_TIERS).difference(self.evidence.tiers)
        if missing_tiers:
            raise ConfigError(
                "evidence.tiers is missing: {}".format(", ".join(sorted(missing_tiers)))
            )
        if self.release.exact_paper_reproduction:
            raise ConfigError(
                "initial release configs must not claim exact paper reproduction"
            )

    @property
    def device(self) -> str:
        return self.runtime.device

    @property
    def window_duration_seconds(self) -> float:
        """Return the configured window duration in seconds."""

        return self.window.samples / self.sampling_frequency_hz

    @property
    def paths(self) -> ProjectPaths:
        """Resolve configured roots without creating directories."""

        base = self.base_dir if self.base_dir is not None else Path.cwd()
        return paths_from_mapping(
            {
                "data_root": self.data_root,
                "output_root": self.output_root,
                "checkpoint_root": self.checkpoint_root,
                "results_root": self.results_root,
            },
            base_dir=base,
        )

    @property
    def project_paths(self) -> ProjectPaths:
        """Descriptive alias for :attr:`paths`."""

        return self.paths

    def with_overrides(
        self,
        overrides: Optional[Mapping[str, Any]] = None,
        *,
        data_root: Optional[PathLike] = None,
        output_root: Optional[PathLike] = None,
        checkpoint_root: Optional[PathLike] = None,
        results_root: Optional[PathLike] = None,
        subject: Optional[int] = None,
        fold: Optional[int] = None,
        sensor: Optional[Union[SensorConfig, Mapping[str, Any]]] = None,
        device: Optional[str] = None,
        seed: Optional[int] = None
    ) -> "ExperimentConfig":
        """Return a validated copy with CLI-style overrides applied.

        ``overrides`` accepts dotted keys such as ``window.hop`` or
        ``runtime.device``.  The named arguments cover the common command-line
        controls explicitly requested for the initial framework.
        """

        data = self.to_mapping()
        if overrides:
            data = apply_overrides(data, overrides)
        for key, value in (
            ("data_root", data_root),
            ("output_root", output_root),
            ("checkpoint_root", checkpoint_root),
            ("results_root", results_root),
        ):
            if value is not None:
                data[key] = value
        _apply_global_seed(data, seed, overrides)
        if subject is not None:
            data.setdefault("split", {})["held_out_subject"] = int(subject)
        if fold is not None:
            data.setdefault("split", {})["fold_id"] = int(fold)
        if device is not None:
            data.setdefault("runtime", {})["device"] = str(device)
        if sensor is not None:
            if isinstance(sensor, SensorConfig):
                sensor = _serialize(sensor)
            if not isinstance(sensor, Mapping):
                raise ConfigError("sensor override must be a mapping or SensorConfig")
            data.setdefault("sensor", {}).update(dict(sensor))
        return ExperimentConfig.from_mapping(
            data,
            config_path=self.config_path,
            base_dir=self.base_dir,
        )

    def to_mapping(self) -> Dict[str, Any]:
        """Serialize the portable portion of this object for override handling."""

        data = _serialize(self)
        data.pop("config_path", None)
        data.pop("base_dir", None)
        return data

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any],
        *,
        config_path: Optional[PathLike] = None,
        base_dir: Optional[PathLike] = None
    ) -> "ExperimentConfig":
        data = _mapping(raw, "config")
        sensor_data = _mapping(data.get("sensor"), "sensor")
        channels = _tuple_of(data.get("channels"), "channels", str)
        if not channels:
            raise ConfigError("channels must be declared")

        indices = _tuple_of(
            sensor_data.get("channel_indices"),
            "sensor.channel_indices",
            int,
        )
        if not indices:
            raise ConfigError("sensor.channel_indices must be declared")

        activity_mapping = _parse_activity_mapping(data)
        activity_codes = _tuple_of(
            data.get("activity_codes"),
            "activity_codes",
            int,
        )
        if not activity_codes:
            activity_codes = tuple(
                activity_mapping[index].code
                for index in sorted(activity_mapping)
            )

        window_data = _mapping(data.get("window"), "window")
        grid_data = _mapping(
            window_data.get("grid", data.get("sensitivity_grid")),
            "window.grid",
        )
        window = WindowConfig(
            samples=_positive_int(window_data.get("samples", 160), "window.samples"),
            hop=_positive_int(window_data.get("hop", 40), "window.hop"),
            grid_samples=_tuple_of(
                grid_data.get("window_samples", grid_data.get("samples")),
                "window.grid.window_samples",
                int,
            ),
            grid_hops=_tuple_of(
                grid_data.get("hop_samples", grid_data.get("hops")),
                "window.grid.hop_samples",
                int,
            ),
        )

        normalization_data = _mapping(data.get("normalization"), "normalization")
        normalization = NormalizationConfig(
            method=str(normalization_data.get("method", "zscore")),
            fit_on=str(normalization_data.get("fit_on", "training_subjects")),
            axes=_tuple_of(
                normalization_data.get("axes", ("sample", "time")),
                "normalization.axes",
                str,
            ),
            apply_to_synthetic=_bool(
                normalization_data.get("apply_to_synthetic", True),
                "normalization.apply_to_synthetic",
            ),
        )

        vae = _parse_vae(data.get("vae"), len(channels), window.samples)
        flow = _parse_flow(data.get("flow"), vae, len(activity_mapping))
        sampling = _parse_sampling(data.get("sampling"), data.get("seed", 42))
        classifiers = _parse_classifiers(data.get("classifiers"), int(data.get("seed", 42)))

        split_data = _mapping(data.get("split"), "split")
        split = SplitConfig(
            protocol=str(split_data.get("protocol", "leave_one_subject_out")),
            held_out_subject=_optional_int(split_data.get("held_out_subject"), "split.held_out_subject"),
            fold_id=_optional_int(split_data.get("fold_id"), "split.fold_id"),
            validation_fraction=_fraction(
                split_data.get("validation_fraction", 0.2),
                "split.validation_fraction",
            ),
        )

        sensitivity_data = _mapping(
            data.get("sensitivity_grid"),
            "sensitivity_grid",
        )
        sensitivity = SensitivityGridConfig(
            enabled=_bool(sensitivity_data.get("enabled", bool(sensitivity_data)), "sensitivity_grid.enabled"),
            window_samples=_tuple_of(
                sensitivity_data.get("window_samples", window.grid_samples),
                "sensitivity_grid.window_samples",
                int,
            ),
            hop_samples=_tuple_of(
                sensitivity_data.get("hop_samples", window.grid_hops),
                "sensitivity_grid.hop_samples",
                int,
            ),
            train_separately_per_setting=_bool(
                sensitivity_data.get("train_separately_per_setting", True),
                "sensitivity_grid.train_separately_per_setting",
            ),
        )

        release_data = _mapping(data.get("release"), "release")
        release = ReleaseMetadata(
            profile=str(release_data.get("profile", "compatibility_default")),
            exact_paper_reproduction=_bool(
                release_data.get("exact_paper_reproduction", False),
                "release.exact_paper_reproduction",
            ),
            note=str(release_data.get("note", ReleaseMetadata.note)),
        )

        evidence_data = _mapping(data.get("evidence"), "evidence")
        evidence = _parse_evidence(evidence_data)
        runtime_data = _mapping(data.get("runtime"), "runtime")
        runtime = RuntimeConfig(device=str(runtime_data.get("device", data.get("device", "auto"))))

        subjects = _tuple_of(data.get("subjects"), "subjects", int)
        if not subjects:
            raise ConfigError("subjects must be declared")

        configured_base = Path(base_dir).expanduser().resolve() if base_dir is not None else Path.cwd()
        configured_config = (
            Path(config_path).expanduser().resolve()
            if config_path is not None
            else None
        )

        return cls(
            name=str(data.get("name", "lrf_imu_experiment")),
            data_root=Path(data.get("data_root", "data")),
            output_root=Path(data.get("output_root", "outputs")),
            checkpoint_root=Path(data.get("checkpoint_root", "checkpoints")),
            results_root=Path(data.get("results_root", "results")),
            sensor=SensorConfig(
                dataset=str(sensor_data.get("dataset", data.get("dataset", "REALDISP"))),
                placement=str(sensor_data.get("placement", "ideal")),
                name=str(sensor_data.get("name", "right_thigh")),
                label_column=_non_negative_int(
                    sensor_data.get("label_column", 119),
                    "sensor.label_column",
                ),
                channel_indices=indices,
                variant=str(sensor_data.get("variant", "six_channel")),
                training_mode=str(sensor_data.get("training_mode", "separate_model")),
                inference_policy=str(
                    sensor_data.get("inference_policy", "declared_channel_set_only")
                ),
                allow_channel_drop=_bool(
                    sensor_data.get("allow_channel_drop", False),
                    "sensor.allow_channel_drop",
                ),
            ),
            channels=channels,
            subjects=subjects,
            activity_codes=activity_codes,
            activity_mapping=activity_mapping,
            sampling_frequency_hz=_float(
                data.get("sampling_frequency_hz", 50),
                "sampling_frequency_hz",
            ),
            window=window,
            normalization=normalization,
            seed=_non_negative_int(data.get("seed", 42), "seed"),
            vae=vae,
            flow=flow,
            sampling=sampling,
            classifiers=classifiers,
            split=split,
            runtime=runtime,
            sensitivity_grid=sensitivity,
            release=release,
            evidence=evidence,
            config_path=configured_config,
            base_dir=configured_base,
        )


def _optional_int(value: Any, name: str) -> Optional[int]:
    if value is None or value == "":
        return None
    return _non_negative_int(value, name)


def _fraction(value: Any, name: str) -> float:
    result = _float(value, name)
    if result < 0 or result >= 1:
        raise ConfigError("{} must be in [0, 1)".format(name))
    return result


def _parse_activity_mapping(data: Mapping[str, Any]) -> Dict[int, ActivitySpec]:
    raw = data.get("activity_mapping")
    if raw is None:
        raw = data.get("activities")
    raw_mapping = _mapping(raw, "activity_mapping")
    if not raw_mapping:
        raise ConfigError("activity_mapping must be declared")
    result: Dict[int, ActivitySpec] = {}
    for key, value in raw_mapping.items():
        index = _non_negative_int(key, "activity_mapping key")
        item = _mapping(value, "activity_mapping[{}]".format(index))
        code = _non_negative_int(
            item.get("code", item.get("realdisp_code")),
            "activity_mapping[{}].code".format(index),
        )
        name = str(item.get("name", "class_{}".format(index)))
        result[index] = ActivitySpec(code=code, name=name)
    return result


def _parse_evidence(data: Mapping[str, Any]) -> EvidenceConfig:
    tiers_data = _mapping(data.get("tiers"), "evidence.tiers")
    tiers = {str(key): str(value) for key, value in tiers_data.items()}
    return EvidenceConfig(
        tiers=tiers,
        defaults=deepcopy(_mapping(data.get("defaults"), "evidence.defaults")),
        conflicts=deepcopy(_mapping(data.get("conflicts"), "evidence.conflicts")),
        notes=_tuple_of(data.get("notes"), "evidence.notes", str),
    )


def _parse_vae(raw: Any, channel_count: int, window_samples: int) -> VAEConfig:
    data = _mapping(raw, "vae")
    training_data = _mapping(data.get("training"), "vae.training")
    beta_data = _mapping(
        training_data.get("beta_schedule", data.get("beta_schedule")),
        "vae.training.beta_schedule",
    )
    augmentation_data = _mapping(
        training_data.get("augmentation", data.get("augmentation")),
        "vae.training.augmentation",
    )
    augmentation = AugmentationConfig(
        enabled=_bool(augmentation_data.get("enabled", True), "vae.augmentation.enabled"),
        jitter=_float(augmentation_data.get("jitter", 0.008), "vae.augmentation.jitter"),
        scale=_float(augmentation_data.get("scale", 0.04), "vae.augmentation.scale"),
        time_mask=_float(
            augmentation_data.get("time_mask", 0.05),
            "vae.augmentation.time_mask",
        ),
    )
    training = VAETrainingConfig(
        batch_size=_positive_int(training_data.get("batch_size", 256), "vae.training.batch_size"),
        learning_rate=_float(training_data.get("learning_rate", 0.001), "vae.training.learning_rate"),
        max_epochs=_positive_int(training_data.get("max_epochs", 1000), "vae.training.max_epochs"),
        early_stop_min_epochs=_non_negative_int(
            training_data.get("early_stop_min_epochs", 200),
            "vae.training.early_stop_min_epochs",
        ),
        early_stop_patience=_positive_int(
            training_data.get("early_stop_patience", 100),
            "vae.training.early_stop_patience",
        ),
        use_amp_bf16=_bool(training_data.get("use_amp_bf16", True), "vae.training.use_amp_bf16"),
        grad_clip=_float(training_data.get("grad_clip", 1.0), "vae.training.grad_clip"),
        l2_weight=_float(training_data.get("l2_weight", 0.5), "vae.training.l2_weight"),
        l1_weight=_float(training_data.get("l1_weight", 0.1), "vae.training.l1_weight"),
        beta_init=_float(beta_data.get("init", training_data.get("beta_init", 0.08)), "vae.training.beta_init"),
        beta_min=_float(beta_data.get("min", training_data.get("beta_min", 0.04)), "vae.training.beta_min"),
        beta_decay=_float(beta_data.get("decay", training_data.get("beta_decay", 0.995)), "vae.training.beta_decay"),
        use_spectral_loss=_bool(
            training_data.get("use_spectral_loss", False),
            "vae.training.use_spectral_loss",
        ),
        fft_weight=_float(training_data.get("fft_weight", 0.0), "vae.training.fft_weight"),
        augmentation=augmentation,
    )
    latent_stride = _positive_int(data.get("latent_stride", 4), "vae.latent_stride")
    return VAEConfig(
        input_channels=_positive_int(data.get("input_channels", channel_count), "vae.input_channels"),
        latent_dim_channels=_positive_int(
            data.get("latent_dim_channels", 48),
            "vae.latent_dim_channels",
        ),
        latent_stride=latent_stride,
        latent_time_steps=_positive_int(
            data.get("latent_time_steps", window_samples // latent_stride),
            "vae.latent_time_steps",
        ),
        deterministic_reconstruction_pass=_bool(
            data.get("deterministic_reconstruction_pass", True),
            "vae.deterministic_reconstruction_pass",
        ),
        schedule_profile=str(
            data.get("schedule_profile", "observed_wrapper_compatibility")
        ),
        training=training,
    )


def _parse_flow(raw: Any, vae: VAEConfig, class_count: int) -> FlowConfig:
    data = _mapping(raw, "flow")
    training_data = _mapping(data.get("training"), "flow.training")
    architecture_data = _mapping(data.get("architecture"), "flow.architecture")
    flow_training = FlowTrainingConfig(
        epochs=_positive_int(training_data.get("epochs", 300), "flow.training.epochs"),
        learning_rate=_float(training_data.get("learning_rate", 0.0005), "flow.training.learning_rate"),
        early_stop_patience=_positive_int(
            training_data.get("early_stop_patience", 50),
            "flow.training.early_stop_patience",
        ),
        grad_clip=_float(training_data.get("grad_clip", 1.0), "flow.training.grad_clip"),
        optimizer=str(training_data.get("optimizer", "AdamW")),
        optimizer_betas=_tuple_of(
            training_data.get("optimizer_betas", (0.9, 0.95)),
            "flow.training.optimizer_betas",
            float,
        ),
        weight_decay=_float(training_data.get("weight_decay", 0.0001), "flow.training.weight_decay"),
        batch_size=_positive_int(training_data.get("batch_size", 128), "flow.training.batch_size"),
        auto_batch=_bool(training_data.get("auto_batch", True), "flow.training.auto_batch"),
    )
    if len(flow_training.optimizer_betas) != 2:
        raise ConfigError("flow.training.optimizer_betas must contain two values")
    architecture = FlowArchitectureConfig(
        channel_multipliers=_tuple_of(
            architecture_data.get("channel_multipliers", (1, 2, 4)),
            "flow.architecture.channel_multipliers",
            int,
        ),
        residual_block_kernel_short=_positive_int(
            architecture_data.get("residual_block_kernel_short", 3),
            "flow.architecture.residual_block_kernel_short",
        ),
        residual_block_kernel_long=_positive_int(
            architecture_data.get("residual_block_kernel_long", 31),
            "flow.architecture.residual_block_kernel_long",
        ),
        normalization_groups=_positive_int(
            architecture_data.get("normalization_groups", 8),
            "flow.architecture.normalization_groups",
        ),
        squeeze_excitation_reduction=_positive_int(
            architecture_data.get("squeeze_excitation_reduction", 4),
            "flow.architecture.squeeze_excitation_reduction",
        ),
        downsampling=str(architecture_data.get("downsampling", "avg_pool_factor_2")),
        upsampling=str(architecture_data.get("upsampling", "nearest_factor_2")),
        num_classes=_positive_int(
            architecture_data.get("num_classes", class_count),
            "flow.architecture.num_classes",
        ),
    )
    return FlowConfig(
        latent_dim_channels=_positive_int(
            data.get("latent_dim_channels", vae.latent_dim_channels),
            "flow.latent_dim_channels",
        ),
        latent_stride=_positive_int(
            data.get("latent_stride", vae.latent_stride),
            "flow.latent_stride",
        ),
        base_width=_positive_int(
            data.get("base_width", data.get("model_ch", 256)),
            "flow.base_width",
        ),
        width_profile=str(
            data.get("width_profile", "observed_wrapper_compatibility")
        ),
        training=flow_training,
        architecture=architecture,
    )


def _parse_sampling(raw: Any, seed: Any) -> SamplingConfig:
    data = _mapping(raw, "sampling")
    configured_seed = data.get("seed", seed)
    return SamplingConfig(
        ode_solver=str(data.get("ode_solver", "Euler")),
        steps=_positive_int(data.get("steps", 10), "sampling.steps"),
        start_time=_float(data.get("start_time", 1.0), "sampling.start_time"),
        end_time=_float(data.get("end_time", 0.0), "sampling.end_time"),
        windows_per_class=_positive_int(
            data.get("windows_per_class", 500),
            "sampling.windows_per_class",
        ),
        seed=_optional_int(configured_seed, "sampling.seed"),
    )


def _parse_classifiers(raw: Any, seed: int) -> ClassifierConfig:
    data = _mapping(raw, "classifiers")
    rf_data = _mapping(data.get("random_forest"), "classifiers.random_forest")
    cnn_data = _mapping(data.get("cnn"), "classifiers.cnn")
    random_forest = RandomForestConfig(
        estimator=str(rf_data.get("estimator", "RandomForestClassifier")),
        n_estimators=_positive_int(
            rf_data.get("n_estimators", 100),
            "classifiers.random_forest.n_estimators",
        ),
        random_state=_non_negative_int(
            rf_data.get("random_state", seed),
            "classifiers.random_forest.random_state",
        ),
        n_jobs=int(rf_data.get("n_jobs", 1)),
        input_representation=str(
            rf_data.get("input_representation", "flattened_standardized_window")
        ),
    )
    cnn = CNNConfig(
        conv_channels=_tuple_of(
            cnn_data.get("conv_channels", (32, 64, 128)),
            "classifiers.cnn.conv_channels",
            int,
        ),
        kernel_size=_positive_int(cnn_data.get("kernel_size", 5), "classifiers.cnn.kernel_size"),
        pool_kernel_size=_positive_int(
            cnn_data.get("pool_kernel_size", 2),
            "classifiers.cnn.pool_kernel_size",
        ),
        fc_hidden=_tuple_of(
            cnn_data.get("fc_hidden", (256, 128)),
            "classifiers.cnn.fc_hidden",
            int,
        ),
        dropout=_fraction(cnn_data.get("dropout", 0.3), "classifiers.cnn.dropout"),
        epochs=_positive_int(cnn_data.get("epochs", 80), "classifiers.cnn.epochs"),
        patience=_positive_int(cnn_data.get("patience", 10), "classifiers.cnn.patience"),
        batch_size=_positive_int(cnn_data.get("batch_size", 64), "classifiers.cnn.batch_size"),
        learning_rate=_float(cnn_data.get("learning_rate", 0.001), "classifiers.cnn.learning_rate"),
        weight_decay=_float(cnn_data.get("weight_decay", 0.0001), "classifiers.cnn.weight_decay"),
        validation_fraction=_fraction(
            cnn_data.get("validation_fraction", 0.2),
            "classifiers.cnn.validation_fraction",
        ),
    )
    primary = str(data.get("primary", "random_forest"))
    if primary not in {"random_forest", "cnn"}:
        raise ConfigError("classifiers.primary must be 'random_forest' or 'cnn'")
    return ClassifierConfig(random_forest=random_forest, cnn=cnn, primary=primary)


def apply_overrides(
    mapping: Mapping[str, Any],
    overrides: Mapping[str, Any],
) -> Dict[str, Any]:
    """Apply dotted CLI-style keys to a copied config mapping.

    Unknown intermediate sections are rejected so a misspelled override does
    not silently create an unused setting.
    """

    result = deepcopy(dict(mapping))
    for dotted_key, value in overrides.items():
        if not isinstance(dotted_key, str) or not dotted_key.strip():
            raise ConfigError("override keys must be non-empty strings")
        parts = [part for part in dotted_key.split(".") if part]
        cursor: Dict[str, Any] = result
        for part in parts[:-1]:
            if part not in cursor or not isinstance(cursor[part], Mapping):
                raise ConfigError("Unknown override path: {}".format(dotted_key))
            cursor = cursor[part]
        if parts[-1] not in cursor:
            raise ConfigError("Unknown override key: {}".format(dotted_key))
        cursor[parts[-1]] = value
    return result


def _apply_global_seed(
    data: Dict[str, Any],
    seed: Optional[int],
    overrides: Optional[Mapping[str, Any]],
) -> None:
    """Apply one named seed while preserving explicit component overrides."""

    if seed is None:
        return
    # The named seed is global by default. Explicit dotted component
    # overrides remain the mechanism for intentionally distinct seeds.
    data["seed"] = seed
    explicit_overrides = overrides or {}
    if "sampling.seed" not in explicit_overrides:
        data.setdefault("sampling", {})["seed"] = seed
    if "classifiers.random_forest.random_state" not in explicit_overrides:
        data.setdefault("classifiers", {}).setdefault("random_forest", {})[
            "random_state"
        ] = seed


def load_config(
    config_path: Optional[PathLike] = None,
    *,
    overrides: Optional[Mapping[str, Any]] = None,
    base_dir: Optional[PathLike] = None,
    data_root: Optional[PathLike] = None,
    output_root: Optional[PathLike] = None,
    checkpoint_root: Optional[PathLike] = None,
    results_root: Optional[PathLike] = None,
    subject: Optional[int] = None,
    fold: Optional[int] = None,
    sensor: Optional[Union[SensorConfig, Mapping[str, Any]]] = None,
    device: Optional[str] = None,
    seed: Optional[int] = None,
) -> ExperimentConfig:
    """Load a YAML config and apply portable, CLI-ready overrides.

    Relative config paths and roots are interpreted against ``base_dir``; if it
    is omitted, the caller's current working directory is used.  The function
    never creates roots or imports model code.
    """

    base = Path(base_dir).expanduser().resolve() if base_dir is not None else Path.cwd()
    requested = Path(config_path or DEFAULT_CONFIG_PATH).expanduser()
    path = requested if requested.is_absolute() else base / requested
    path = path.resolve()
    if not path.is_file():
        raise ConfigError("Configuration file does not exist: {}".format(path))
    try:
        with path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
    except yaml.YAMLError as exc:
        raise ConfigError("Could not parse YAML config {}".format(path)) from exc
    data = _mapping(raw, "config")
    if overrides:
        if "runtime.device" in overrides and "runtime" not in data:
            data["runtime"] = {"device": data.get("device", "auto")}
        data = apply_overrides(data, overrides)
    for key, value in (
        ("data_root", data_root),
        ("output_root", output_root),
        ("checkpoint_root", checkpoint_root),
        ("results_root", results_root),
    ):
        if value is not None:
            data[key] = value
    _apply_global_seed(data, seed, overrides)
    if subject is not None:
        data.setdefault("split", {})["held_out_subject"] = int(subject)
    if fold is not None:
        data.setdefault("split", {})["fold_id"] = int(fold)
    if device is not None:
        data.setdefault("runtime", {})["device"] = str(device)
    if sensor is not None:
        if isinstance(sensor, SensorConfig):
            sensor = _serialize(sensor)
        if not isinstance(sensor, Mapping):
            raise ConfigError("sensor override must be a mapping or SensorConfig")
        data.setdefault("sensor", {}).update(dict(sensor))
    return ExperimentConfig.from_mapping(
        data,
        config_path=path,
        base_dir=base,
    )


def _serialize(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {
            item.name: _serialize(getattr(value, item.name))
            for item in fields(value)
        }
    if isinstance(value, Mapping):
        return {key: _serialize(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_serialize(item) for item in value]
    if isinstance(value, list):
        return [_serialize(item) for item in value]
    return value
