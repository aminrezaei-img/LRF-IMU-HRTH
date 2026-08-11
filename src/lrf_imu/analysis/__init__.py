"""Portable, numerical-only analyses used by the public paper workflow.

Heavy optional dependencies are imported only when the corresponding function
is executed.  Importing :mod:`lrf_imu.analysis` therefore remains safe in the
base installation.
"""

from .ablation import (
    class_conditional_latent_parameters,
    sample_class_conditional_latents,
    vae_only_random_forest_metrics,
)
from .physical import (
    GRAVITY_M_S2,
    acceleration_magnitude,
    acceleration_magnitude_summary,
)
from .privacy import (
    POSTHOC_AUDIT_THREAT_MODEL,
    RECONSTRUCTION_THREAT_MODEL,
    TRUE_HOLDOUT_THREAT_MODEL,
    reconstruction_success,
    summarize_membership_records,
    summarize_reconstruction_records,
)
from .sensitivity import WINDOW_GRID, summarize_sensitivity_grid
from .spectral import aggregate_fold_psd, compute_psd, spectral_statistics

__all__ = [
    "GRAVITY_M_S2",
    "POSTHOC_AUDIT_THREAT_MODEL",
    "RECONSTRUCTION_THREAT_MODEL",
    "TRUE_HOLDOUT_THREAT_MODEL",
    "WINDOW_GRID",
    "acceleration_magnitude",
    "acceleration_magnitude_summary",
    "aggregate_fold_psd",
    "class_conditional_latent_parameters",
    "compute_psd",
    "reconstruction_success",
    "sample_class_conditional_latents",
    "spectral_statistics",
    "summarize_membership_records",
    "summarize_reconstruction_records",
    "summarize_sensitivity_grid",
    "vae_only_random_forest_metrics",
]
