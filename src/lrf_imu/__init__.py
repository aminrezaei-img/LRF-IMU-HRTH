"""Portable configuration and path primitives for the LRF-IMU release."""

from .config import (
    DEFAULT_CONFIG_PATH,
    PACKAGE_CONFIG_DIR,
    REQUIRED_EVIDENCE_TIERS,
    ActivitySpec,
    ClassifierConfig,
    ConfigError,
    CNNConfig,
    EvidenceConfig,
    EvidenceTier,
    ExperimentConfig,
    FlowConfig,
    NormalizationConfig,
    ReleaseMetadata,
    RuntimeConfig,
    SamplingConfig,
    SensorConfig,
    SplitConfig,
    VAEConfig,
    WindowConfig,
    apply_overrides,
    load_config,
)
from .paths import FoldPaths, ProjectPaths, fold_paths, paths_from_mapping


__version__ = "0.1.0"


__all__ = [
    "ActivitySpec",
    "ClassifierConfig",
    "ConfigError",
    "CNNConfig",
    "DEFAULT_CONFIG_PATH",
    "EvidenceConfig",
    "PACKAGE_CONFIG_DIR",
    "EvidenceTier",
    "ExperimentConfig",
    "FlowConfig",
    "FoldPaths",
    "NormalizationConfig",
    "ProjectPaths",
    "REQUIRED_EVIDENCE_TIERS",
    "ReleaseMetadata",
    "RuntimeConfig",
    "SamplingConfig",
    "SensorConfig",
    "SplitConfig",
    "VAEConfig",
    "WindowConfig",
    "apply_overrides",
    "fold_paths",
    "load_config",
    "paths_from_mapping",
]
