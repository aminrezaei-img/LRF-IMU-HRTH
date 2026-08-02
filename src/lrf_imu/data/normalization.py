"""Training-only per-channel z-score normalization for IMU windows.

The public data contract represents windows as ``[N, C, T]``.  Statistics are
fit per channel by reducing the batch and time axes, then reused unchanged for
every other split in the fold.  This module deliberately does not discover
subjects or load data: callers must pass the post-validation training windows
to :meth:`ChannelStandardizer.fit_training`.

The implementation mirrors the observed ``SimpleStandardizer`` in
``VAE/VAE_logic.py`` while making the state boundary explicit, validating
inputs, and providing JSON-safe metadata for reproducible fold artifacts.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, Dict, Iterable, Optional, Tuple, Union

import numpy as np


FIT_AXES: Tuple[int, int] = (0, 2)
DEFAULT_EPSILON = 1e-8
NORMALIZATION_SCHEMA_VERSION = "3A.normalization.1"
APPLY_TO = (
    "train",
    "validation",
    "held_out_test",
    "synthetic_in_standardized_coordinate_system",
)


class NormalizationError(ValueError):
    """Base class for invalid normalization input or metadata."""


class NotFittedError(RuntimeError):
    """Raised when a transform is requested before training statistics exist."""


class ShapeValidationError(NormalizationError):
    """Raised when an input is not a non-empty-channel ``[N, C, T]`` array."""


class ChannelValidationError(NormalizationError):
    """Raised when an input channel count differs from the fitted contract."""


class NonFiniteDataError(NormalizationError):
    """Raised when an input or serialized statistic contains a non-finite value."""


SubjectId = Union[int, str]


def _validate_epsilon(value: Any) -> float:
    """Validate and normalize the minimum standard deviation."""

    try:
        epsilon = float(value)
    except (TypeError, ValueError) as exc:
        raise NormalizationError("epsilon must be a finite positive number") from exc
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise NormalizationError("epsilon must be a finite positive number")
    return epsilon


def _validate_channel_count(value: Any, *, field_name: str = "channels") -> int:
    """Return a positive Python integer channel count."""

    if isinstance(value, bool):
        raise ChannelValidationError("{} must be a positive integer".format(field_name))
    try:
        count = int(value)
    except (TypeError, ValueError) as exc:
        raise ChannelValidationError(
            "{} must be a positive integer".format(field_name)
        ) from exc
    if count != value or count <= 0:
        raise ChannelValidationError("{} must be a positive integer".format(field_name))
    return count


def _coerce_windows(
    windows: Any,
    *,
    name: str,
    allow_empty_batch: bool,
) -> np.ndarray:
    """Validate an array-shaped window collection without changing its dtype."""

    try:
        array = np.asarray(windows)
    except Exception as exc:  # pragma: no cover - NumPy-specific conversion errors
        raise ShapeValidationError(
            "{} must be a numeric array with shape [N, C, T]".format(name)
        ) from exc

    if array.ndim != 3:
        raise ShapeValidationError(
            "{} must have shape [N, C, T], got ndim {} and shape {}".format(
                name, array.ndim, tuple(array.shape)
            )
        )
    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(
        array.dtype, np.complexfloating
    ):
        raise ShapeValidationError(
            "{} must contain real numeric values, got dtype {}".format(
                name, array.dtype
            )
        )
    if array.shape[0] == 0 and not allow_empty_batch:
        raise ShapeValidationError("{} must contain at least one window".format(name))
    if array.shape[1] <= 0 or array.shape[2] <= 0:
        raise ShapeValidationError(
            "{} must have positive channel and time dimensions, got shape {}".format(
                name, tuple(array.shape)
            )
        )
    try:
        finite = bool(np.all(np.isfinite(array)))
    except TypeError as exc:  # pragma: no cover - covered by dtype guard above
        raise ShapeValidationError(
            "{} must contain real numeric values".format(name)
        ) from exc
    if not finite:
        raise NonFiniteDataError("{} must contain only finite values".format(name))
    return array


def _output_dtype(input_dtype: np.dtype) -> np.dtype:
    """Preserve floating input precision; use float32 for integer inputs."""

    dtype = np.dtype(input_dtype)
    if np.issubdtype(dtype, np.floating):
        return dtype
    return np.dtype(np.float32)


def _subject_ids(
    values: Optional[Union[SubjectId, Iterable[SubjectId]]], *, name: str
) -> Optional[Tuple[SubjectId, ...]]:
    """Normalize optional subject metadata to JSON-safe, unique IDs."""

    if values is None:
        return None
    if isinstance(values, (str, int)) and not isinstance(values, bool):
        candidate_values = [values]
    else:
        try:
            candidate_values = list(values)  # type: ignore[arg-type]
        except TypeError as exc:
            raise NormalizationError(
                "{} must be a subject ID or an iterable of subject IDs".format(name)
            ) from exc

    normalized = []
    for value in candidate_values:
        if isinstance(value, np.integer) and not isinstance(value, np.bool_):
            value = int(value)
        elif isinstance(value, str):
            value = str(value)
        elif isinstance(value, int) and not isinstance(value, bool):
            value = int(value)
        else:
            raise NormalizationError(
                "{} must contain only string or integer subject IDs".format(name)
            )
        normalized.append(value)

    if len(set(normalized)) != len(normalized):
        raise NormalizationError("{} must not contain duplicate subject IDs".format(name))
    return tuple(normalized)


def _validate_subject_separation(
    training_subjects: Optional[Tuple[SubjectId, ...]],
    validation_subjects: Optional[Tuple[SubjectId, ...]],
    held_out_subject: Optional[SubjectId],
) -> None:
    """Reject metadata that would document an obvious split leakage."""

    training_set = set(training_subjects or ())
    validation_set = set(validation_subjects or ())
    if training_set.intersection(validation_set):
        raise NormalizationError(
            "training_subjects and validation_subjects must be disjoint"
        )
    if held_out_subject is not None and held_out_subject in training_set:
        raise NormalizationError("held_out_subject must not be a training subject")
    if held_out_subject is not None and held_out_subject in validation_set:
        raise NormalizationError("held_out_subject must not be a validation subject")


def _statistic_vector(value: Any, *, name: str, channels: int) -> np.ndarray:
    """Read flat or broadcast-shaped serialized channel statistics."""

    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise NormalizationError("{} must be numeric".format(name)) from exc
    if array.size != channels:
        raise ChannelValidationError(
            "{} must contain exactly {} channel values, got shape {}".format(
                name, channels, tuple(array.shape)
            )
        )
    if not np.all(np.isfinite(array)):
        raise NonFiniteDataError("{} must contain only finite values".format(name))
    return array.reshape(1, channels, 1).copy()


class ChannelStandardizer:
    """Fit and apply a training-only per-channel z-score transform.

    Parameters
    ----------
    channels:
        Optional expected number of channels.  If omitted, the channel count
        is inferred from the training windows and then fixed for the lifetime
        of the instance.
    epsilon:
        Minimum standard deviation.  Population standard deviations below this
        value are replaced by ``epsilon``.

    Notes
    -----
    ``fit_training`` is the explicit public fitting entry point.  ``fit`` is a
    compatibility spelling with the same training-only semantics; neither
    method accepts validation, held-out, or synthetic arrays for fitting.
    """

    def __init__(
        self,
        channels: Optional[int] = None,
        epsilon: float = DEFAULT_EPSILON,
        *,
        expected_channels: Optional[int] = None,
        min_std: Optional[float] = None,
    ) -> None:
        if channels is not None and expected_channels is not None:
            if _validate_channel_count(channels) != _validate_channel_count(
                expected_channels, field_name="expected_channels"
            ):
                raise ChannelValidationError(
                    "channels and expected_channels must agree"
                )
        if expected_channels is not None:
            channels = expected_channels
        if min_std is not None:
            epsilon = min_std

        self._expected_channels = (
            _validate_channel_count(channels) if channels is not None else None
        )
        self._epsilon = _validate_epsilon(epsilon)
        self._channels: Optional[int] = None
        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None
        self._fit_subjects: Optional[Tuple[SubjectId, ...]] = None
        self._validation_subjects: Optional[Tuple[SubjectId, ...]] = None
        self._held_out_subject: Optional[SubjectId] = None
        self._fit_sample_count: Optional[int] = None

    @property
    def fitted(self) -> bool:
        """Whether training statistics have been fitted or restored."""

        return self._mean is not None and self._std is not None

    @property
    def is_fitted(self) -> bool:
        """Alias for :attr:`fitted` used by estimator-style callers."""

        return self.fitted

    @property
    def epsilon(self) -> float:
        """The standard-deviation floor."""

        return self._epsilon

    @property
    def channels(self) -> Optional[int]:
        """The fixed channel count, or the constructor expectation before fit."""

        return self._channels if self._channels is not None else self._expected_channels

    @property
    def n_channels(self) -> Optional[int]:
        """Alias for :attr:`channels`."""

        return self.channels

    def _require_fitted(self) -> Tuple[np.ndarray, np.ndarray, int]:
        if not self.fitted or self._mean is None or self._std is None or self._channels is None:
            raise NotFittedError(
                "ChannelStandardizer is not fitted; call fit_training() first"
            )
        return self._mean, self._std, self._channels

    def _validate_channels(self, array: np.ndarray, *, name: str) -> None:
        expected = self.channels
        if expected is not None and array.shape[1] != expected:
            raise ChannelValidationError(
                "{} has {} channels; expected {}".format(
                    name, array.shape[1], expected
                )
            )

    def fit_training(
        self,
        training_windows: Any,
        *,
        training_subjects: Optional[Union[SubjectId, Iterable[SubjectId]]] = None,
        validation_subjects: Optional[Union[SubjectId, Iterable[SubjectId]]] = None,
        held_out_subject: Optional[SubjectId] = None,
    ) -> "ChannelStandardizer":
        """Fit statistics on post-validation training windows only.

        The arrays belonging to validation, held-out-test, and synthetic
        splits are intentionally not accepted as separate fitting arguments.
        They must be passed later to :meth:`transform`, which reuses this
        immutable fold state.
        """

        array = _coerce_windows(
            training_windows, name="training_windows", allow_empty_batch=False
        )
        self._validate_channels(array, name="training_windows")

        fit_subjects = _subject_ids(training_subjects, name="training_subjects")
        val_subjects = _subject_ids(
            validation_subjects, name="validation_subjects"
        )
        held_out = _subject_ids(
            held_out_subject, name="held_out_subject"
        )
        held_out_id = held_out[0] if held_out is not None else None
        _validate_subject_separation(fit_subjects, val_subjects, held_out_id)

        # Compute into locals so a failed fit never replaces an existing
        # fitted state.  Float64 accumulation keeps metadata stable for both
        # float32 windows and integer synthetic probes.
        values = np.asarray(array, dtype=np.float64)
        mean = np.mean(values, axis=FIT_AXES, keepdims=True, dtype=np.float64)
        std = np.std(
            values, axis=FIT_AXES, keepdims=True, ddof=0, dtype=np.float64
        )
        std = np.maximum(std, self._epsilon)
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)):
            raise NonFiniteDataError("computed normalization statistics are not finite")

        self._channels = int(array.shape[1])
        self._mean = mean.reshape(1, self._channels, 1)
        self._std = std.reshape(1, self._channels, 1)
        self._fit_subjects = fit_subjects
        self._validation_subjects = val_subjects
        self._held_out_subject = held_out_id
        self._fit_sample_count = int(array.shape[0])
        return self

    def fit(self, training_windows: Any, **kwargs: Any) -> "ChannelStandardizer":
        """Compatibility spelling for :meth:`fit_training`."""

        return self.fit_training(training_windows, **kwargs)

    # Explicit alias for callers that prefer the name used in the contract.
    fit_on_training = fit_training

    def _transform(self, windows: Any, *, inverse: bool, name: str) -> np.ndarray:
        mean, std, _ = self._require_fitted()
        array = _coerce_windows(windows, name=name, allow_empty_batch=True)
        self._validate_channels(array, name=name)
        values = np.asarray(array, dtype=np.float64)
        if inverse:
            transformed = values * std + mean
        else:
            transformed = (values - mean) / std
        if not np.all(np.isfinite(transformed)):
            raise NonFiniteDataError(
                "{} produced non-finite values".format(name)
            )
        return transformed.astype(_output_dtype(array.dtype), copy=False)

    def transform(self, windows: Any) -> np.ndarray:
        """Apply the fitted training statistics to any compatible split."""

        return self._transform(windows, inverse=False, name="windows")

    def apply(self, windows: Any) -> np.ndarray:
        """Alias for :meth:`transform` for pipeline-style callers."""

        return self.transform(windows)

    def fit_transform(self, training_windows: Any, **kwargs: Any) -> np.ndarray:
        """Fit on training windows and return their standardized values."""

        self.fit_training(training_windows, **kwargs)
        return self.transform(training_windows)

    def inverse_transform(self, windows: Any) -> np.ndarray:
        """Map standardized windows back to the original channel coordinates."""

        return self._transform(windows, inverse=True, name="standardized_windows")

    def inverse(self, windows: Any) -> np.ndarray:
        """Alias for :meth:`inverse_transform`."""

        return self.inverse_transform(windows)

    @property
    def mean(self) -> np.ndarray:
        """Copy of the broadcast-shaped fitted means ``[1, C, 1]``."""

        mean, _, _ = self._require_fitted()
        return mean.copy()

    @property
    def mean_(self) -> np.ndarray:
        """Estimator-style alias for :attr:`mean`."""

        return self.mean

    @property
    def std(self) -> np.ndarray:
        """Copy of the floored broadcast-shaped fitted SDs ``[1, C, 1]``."""

        _, std, _ = self._require_fitted()
        return std.copy()

    @property
    def std_(self) -> np.ndarray:
        """Estimator-style alias for :attr:`std`."""

        return self.std

    @property
    def scale(self) -> np.ndarray:
        """Alias for :attr:`std`."""

        return self.std

    @property
    def scale_(self) -> np.ndarray:
        """Estimator-style alias for :attr:`scale`."""

        return self.scale

    def to_metadata(self) -> Dict[str, Any]:
        """Return a JSON-safe description of the fitted transform."""

        mean, std, channels = self._require_fitted()
        return {
            "schema_version": NORMALIZATION_SCHEMA_VERSION,
            "method": "per_channel_zscore",
            "normalization_method": "zscore",
            "fit_axes": [int(axis) for axis in FIT_AXES],
            "ddof": 0,
            "epsilon": float(self._epsilon),
            "minimum_standard_deviation": float(self._epsilon),
            "fit_stage": "post_validation_training_subjects_only",
            "fit_on": "training_subjects",
            "training_only": True,
            "apply_to": list(APPLY_TO),
            "channels": int(channels),
            "mean": [float(value) for value in mean.reshape(-1)],
            "std": [float(value) for value in std.reshape(-1)],
            "mean_shape": [1, int(channels), 1],
            "std_shape": [1, int(channels), 1],
            "fit_sample_count": int(self._fit_sample_count or 0),
            "training_subjects": (
                list(self._fit_subjects) if self._fit_subjects is not None else None
            ),
            "validation_subjects": (
                list(self._validation_subjects)
                if self._validation_subjects is not None
                else None
            ),
            "held_out_subject": self._held_out_subject,
        }

    @property
    def metadata(self) -> Dict[str, Any]:
        """Property form of :meth:`to_metadata`."""

        return self.to_metadata()

    def to_dict(self) -> Dict[str, Any]:
        """Alias for :meth:`to_metadata`."""

        return self.to_metadata()

    def to_json(self, **json_kwargs: Any) -> str:
        """Serialize fitted metadata as JSON."""

        return json.dumps(self.to_metadata(), sort_keys=True, **json_kwargs)

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, Any]) -> "ChannelStandardizer":
        """Restore a fitted standardizer from JSON-compatible metadata."""

        if not isinstance(metadata, Mapping):
            raise NormalizationError("metadata must be a mapping")

        method = metadata.get("method", metadata.get("normalization_method"))
        if method not in ("per_channel_zscore", "zscore"):
            raise NormalizationError(
                "unsupported normalization method: {}".format(method)
            )
        axes = metadata.get("fit_axes", metadata.get("axes", list(FIT_AXES)))
        if tuple(axes) != FIT_AXES:
            raise NormalizationError("fit_axes must be [0, 2]")
        try:
            ddof = int(metadata.get("ddof", metadata.get("standard_deviation_ddof", 0)))
        except (TypeError, ValueError) as exc:
            raise NormalizationError("ddof must be zero") from exc
        if ddof != 0:
            raise NormalizationError("ddof must be zero")

        epsilon = _validate_epsilon(
            metadata.get(
                "epsilon", metadata.get("minimum_standard_deviation", DEFAULT_EPSILON)
            )
        )
        channels_value = metadata.get("channels", metadata.get("n_channels"))
        if channels_value is None:
            raise ChannelValidationError("metadata must declare channels")
        channels = _validate_channel_count(channels_value)
        mean_value = metadata.get("mean", metadata.get("mean_"))
        std_value = metadata.get(
            "std", metadata.get("scale", metadata.get("std_"))
        )
        if mean_value is None or std_value is None:
            raise NormalizationError("metadata must include mean and std")
        mean = _statistic_vector(mean_value, name="mean", channels=channels)
        std = _statistic_vector(std_value, name="std", channels=channels)
        if np.any(std <= 0):
            raise NormalizationError("std must be strictly positive")
        std = np.maximum(std, epsilon)

        expected_mean_shape = metadata.get("mean_shape")
        expected_std_shape = metadata.get("std_shape")
        expected_shape = [1, channels, 1]
        if expected_mean_shape is not None and list(expected_mean_shape) != expected_shape:
            raise ShapeValidationError("metadata mean_shape must be [1, channels, 1]")
        if expected_std_shape is not None and list(expected_std_shape) != expected_shape:
            raise ShapeValidationError("metadata std_shape must be [1, channels, 1]")

        fit_subjects = _subject_ids(
            metadata.get("training_subjects", metadata.get("fit_subjects")),
            name="training_subjects",
        )
        validation_subjects = _subject_ids(
            metadata.get("validation_subjects"), name="validation_subjects"
        )
        held_out = _subject_ids(
            metadata.get("held_out_subject"), name="held_out_subject"
        )
        held_out_id = held_out[0] if held_out is not None else None
        _validate_subject_separation(fit_subjects, validation_subjects, held_out_id)

        result = cls(channels=channels, epsilon=epsilon)
        result._channels = channels
        result._mean = mean
        result._std = std
        result._fit_subjects = fit_subjects
        result._validation_subjects = validation_subjects
        result._held_out_subject = held_out_id
        fit_count = metadata.get("fit_sample_count", 0)
        if isinstance(fit_count, bool):
            raise NormalizationError("fit_sample_count must be a non-negative integer")
        try:
            fit_count_int = int(fit_count)
        except (TypeError, ValueError) as exc:
            raise NormalizationError(
                "fit_sample_count must be a non-negative integer"
            ) from exc
        if fit_count_int < 0 or fit_count_int != fit_count:
            raise NormalizationError("fit_sample_count must be a non-negative integer")
        result._fit_sample_count = fit_count_int
        return result

    @classmethod
    def from_dict(cls, metadata: Mapping[str, Any]) -> "ChannelStandardizer":
        """Alias for :meth:`from_metadata`."""

        return cls.from_metadata(metadata)

    @classmethod
    def from_json(cls, payload: str) -> "ChannelStandardizer":
        """Restore a fitted standardizer from a JSON string."""

        try:
            metadata = json.loads(payload)
        except (TypeError, ValueError) as exc:
            raise NormalizationError("payload must contain valid JSON metadata") from exc
        return cls.from_metadata(metadata)


def fit_training_normalizer(
    training_windows: Any,
    *,
    channels: Optional[int] = None,
    epsilon: float = DEFAULT_EPSILON,
    training_subjects: Optional[Union[SubjectId, Iterable[SubjectId]]] = None,
    validation_subjects: Optional[Union[SubjectId, Iterable[SubjectId]]] = None,
    held_out_subject: Optional[SubjectId] = None,
) -> ChannelStandardizer:
    """Construct and fit a :class:`ChannelStandardizer` on training windows."""

    return ChannelStandardizer(channels=channels, epsilon=epsilon).fit_training(
        training_windows,
        training_subjects=training_subjects,
        validation_subjects=validation_subjects,
        held_out_subject=held_out_subject,
    )


# Names used by adjacent estimator-style code and by the characterized source.
Standardizer = ChannelStandardizer
PerChannelZScore = ChannelStandardizer
SimpleStandardizer = ChannelStandardizer


__all__ = [
    "APPLY_TO",
    "ChannelStandardizer",
    "ChannelValidationError",
    "DEFAULT_EPSILON",
    "FIT_AXES",
    "NORMALIZATION_SCHEMA_VERSION",
    "NonFiniteDataError",
    "NormalizationError",
    "NotFittedError",
    "PerChannelZScore",
    "ShapeValidationError",
    "SimpleStandardizer",
    "Standardizer",
    "fit_training_normalizer",
]
