"""Portable configuration, data, and lazy VAE exports for LRF-IMU."""

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

_VAE_EXPORTS = frozenset(
    {
        "CheckpointError",
        "CheckpointInspection",
        "LatentVAE1D",
        "TrainingResult",
        "VAEProfile",
        "augment_batch",
        "compute_vae_loss",
        "expected_state_dict_shapes",
        "inspect_vae_checkpoint",
        "load_vae_checkpoint",
        "profile_from_config",
        "save_vae_checkpoint",
        "select_vae_profile",
        "train_vae",
    }
)


def __getattr__(name: str):
    """Load optional torch-backed exports only when a caller requests them."""

    if name not in _VAE_EXPORTS:
        raise AttributeError("module {!r} has no attribute {!r}".format(__name__, name))
    from .checkpoints import (
        CheckpointError,
        CheckpointInspection,
        expected_state_dict_shapes,
        inspect_vae_checkpoint,
        load_vae_checkpoint,
        save_vae_checkpoint,
    )
    from .models.vae import LatentVAE1D
    from .training.vae import (
        TrainingResult,
        VAEProfile,
        augment_batch,
        compute_vae_loss,
        profile_from_config,
        select_vae_profile,
        train_vae,
    )
    exports = {
        "CheckpointError": CheckpointError,
        "CheckpointInspection": CheckpointInspection,
        "LatentVAE1D": LatentVAE1D,
        "TrainingResult": TrainingResult,
        "VAEProfile": VAEProfile,
        "augment_batch": augment_batch,
        "compute_vae_loss": compute_vae_loss,
        "expected_state_dict_shapes": expected_state_dict_shapes,
        "inspect_vae_checkpoint": inspect_vae_checkpoint,
        "load_vae_checkpoint": load_vae_checkpoint,
        "profile_from_config": profile_from_config,
        "save_vae_checkpoint": save_vae_checkpoint,
        "select_vae_profile": select_vae_profile,
        "train_vae": train_vae,
    }
    globals().update(exports)
    return exports[name]


__all__ = [
    "ActivitySpec",
    "ClassifierConfig",
    "CheckpointError",
    "CheckpointInspection",
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
    "LatentVAE1D",
    "ProjectPaths",
    "REQUIRED_EVIDENCE_TIERS",
    "ReleaseMetadata",
    "RuntimeConfig",
    "SamplingConfig",
    "SensorConfig",
    "SplitConfig",
    "TrainingResult",
    "VAEConfig",
    "VAEProfile",
    "WindowConfig",
    "apply_overrides",
    "augment_batch",
    "compute_vae_loss",
    "expected_state_dict_shapes",
    "fold_paths",
    "inspect_vae_checkpoint",
    "load_config",
    "load_vae_checkpoint",
    "paths_from_mapping",
    "profile_from_config",
    "save_vae_checkpoint",
    "select_vae_profile",
    "train_vae",
]
