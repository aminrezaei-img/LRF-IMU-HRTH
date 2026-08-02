"""Deterministic, subject-safe split contracts for the public data layer.

The public split API deliberately returns typed result objects instead of the
historical positional tuple.  Subject-level validation for the VAE and
window-level validation for the CNN are separate protocols and have separate
result metadata.

Only subject identifiers and caller-owned arrays cross this module boundary.
The module does not discover files, read participant data, mutate global random
state, or write split artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from numbers import Integral
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np


SubjectId = int
SubjectKey = Union[int, str]


# These values are the locked paper-specific cohort, not a claim that all
# REALDISP subjects or activities are supported by the public release.
CANONICAL_SUBJECTS: Tuple[SubjectId, ...] = (
    1,
    2,
    3,
    5,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    16,
)
CANONICAL_PAPER_SUBJECTS = CANONICAL_SUBJECTS
DEFAULT_SPLIT_SEED = 42
VAE_SUBJECT_VALIDATION_FRACTION = 0.15
CNN_WINDOW_VALIDATION_FRACTION = 0.20


_SUBJECT_TEXT_RE = re.compile(r"^(?:subject)?0*(\d+)$", re.IGNORECASE)


class SplitError(ValueError):
    """Base class for invalid or unsafe split inputs."""


class DuplicateSubjectError(SplitError):
    """Raised when two input keys normalize to the same subject ID."""


class MissingSubjectError(SplitError):
    """Raised when a required subject or complete cohort member is absent."""


class UnknownSubjectError(SplitError):
    """Raised when a subject is outside the canonical paper cohort."""


class InvalidSplitError(SplitError):
    """Raised when an otherwise known input cannot form a safe split."""


@dataclass(frozen=True)
class LosoFold:
    """One canonical leave-one-subject-out fold."""

    held_out_subject: SubjectId
    training_subjects: Tuple[SubjectId, ...]


@dataclass(frozen=True)
class SubjectSplit:
    """Subject assignment for the VAE-safe protocol.

    ``train_subjects`` and ``validation_subjects`` retain the deterministic
    post-shuffle order used when windows are concatenated.  The held-out
    subject is never present in either tuple.
    """

    train_subjects: Tuple[SubjectId, ...]
    validation_subjects: Tuple[SubjectId, ...]
    held_out_subject: SubjectId
    metadata: "SplitMetadata"


@dataclass(frozen=True)
class SplitMetadata:
    """JSON-safe metadata describing one VAE subject-level split."""

    protocol: str
    validation_unit: str
    validation_fraction: float
    seed: int
    random_state: str
    canonical_cohort: Tuple[SubjectId, ...]
    available_subjects: Tuple[SubjectId, ...]
    train_subjects: Tuple[SubjectId, ...]
    validation_subjects: Tuple[SubjectId, ...]
    held_out_subject: SubjectId
    window_counts_by_subject: Mapping[SubjectId, int]
    window_counts: Mapping[str, int]
    shapes: Mapping[str, Tuple[int, ...]]

    def as_dict(self) -> Dict[str, Any]:
        """Return a stable JSON-compatible representation."""

        return {
            "protocol": self.protocol,
            "validation_unit": self.validation_unit,
            "validation_fraction": float(self.validation_fraction),
            "seed": int(self.seed),
            "random_state": self.random_state,
            "canonical_cohort": [_format_subject(sid) for sid in self.canonical_cohort],
            "available_subjects": [_format_subject(sid) for sid in self.available_subjects],
            "train_subjects": [_format_subject(sid) for sid in self.train_subjects],
            "validation_subjects": [_format_subject(sid) for sid in self.validation_subjects],
            "held_out_subject": _format_subject(self.held_out_subject),
            "window_counts_by_subject": {
                _format_subject(sid): int(count)
                for sid, count in sorted(self.window_counts_by_subject.items())
            },
            "window_counts": {
                str(name): int(count) for name, count in self.window_counts.items()
            },
            "shapes": {
                str(name): [int(value) for value in shape]
                for name, shape in self.shapes.items()
            },
        }


@dataclass(frozen=True)
class VaeSplitResult:
    """Materialized VAE-safe LOSO split.

    This object is the complete public return contract.  It intentionally has
    no positional tuple protocol, so callers cannot accidentally consume the
    historical seven-versus-eight-item return signatures.
    """

    train_windows: np.ndarray
    validation_windows: np.ndarray
    held_out_test_windows: np.ndarray
    train_labels: Optional[np.ndarray]
    validation_labels: Optional[np.ndarray]
    held_out_test_labels: Optional[np.ndarray]
    metadata: SplitMetadata

    @property
    def train_subjects(self) -> Tuple[SubjectId, ...]:
        return self.metadata.train_subjects

    @property
    def validation_subjects(self) -> Tuple[SubjectId, ...]:
        return self.metadata.validation_subjects

    @property
    def held_out_subject(self) -> SubjectId:
        return self.metadata.held_out_subject

    def as_dict(self) -> Dict[str, Any]:
        """Return metadata and shapes without serializing signal values."""

        return {
            "metadata": self.metadata.as_dict(),
            "has_labels": self.train_labels is not None,
        }


@dataclass(frozen=True)
class CnnWindowSplitMetadata:
    """Metadata for the distinct CNN window-level validation protocol."""

    protocol: str
    validation_unit: str
    validation_fraction: float
    seed: int
    stratified: bool
    separate_from_vae_subject_split: bool
    train_count: int
    validation_count: int

    def as_dict(self) -> Dict[str, Any]:
        return {
            "protocol": self.protocol,
            "validation_unit": self.validation_unit,
            "validation_fraction": float(self.validation_fraction),
            "seed": int(self.seed),
            "stratified": bool(self.stratified),
            "separate_from_vae_subject_split": bool(
                self.separate_from_vae_subject_split
            ),
            "train_count": int(self.train_count),
            "validation_count": int(self.validation_count),
        }


@dataclass(frozen=True)
class CnnWindowSplit:
    """Materialized window-level CNN train/validation split."""

    train_windows: np.ndarray
    validation_windows: np.ndarray
    train_labels: np.ndarray
    validation_labels: np.ndarray
    train_indices: np.ndarray
    validation_indices: np.ndarray
    metadata: CnnWindowSplitMetadata


def _format_subject(subject_id: SubjectId) -> str:
    return "{:02d}".format(int(subject_id))


def _normalize_subject_id(value: SubjectKey, *, field_name: str) -> SubjectId:
    if isinstance(value, bool):
        raise InvalidSplitError("{} must be an integer subject ID, not bool".format(field_name))

    if isinstance(value, Integral):
        subject_id = int(value)
    elif isinstance(value, str):
        match = _SUBJECT_TEXT_RE.fullmatch(value.strip())
        if match is None:
            raise InvalidSplitError(
                "{} must look like 1, 01, or subject01; got {!r}".format(
                    field_name, value
                )
            )
        subject_id = int(match.group(1))
    else:
        raise InvalidSplitError(
            "{} must be an integer or subject string; got {!r}".format(
                field_name, value
            )
        )

    if subject_id not in CANONICAL_SUBJECTS:
        raise UnknownSubjectError(
            "{}={} is outside the canonical paper cohort {}".format(
                field_name, subject_id, list(CANONICAL_SUBJECTS)
            )
        )
    return subject_id


def _normalize_subject_sequence(
    subjects: Iterable[SubjectKey],
    *,
    field_name: str,
    require_complete_cohort: bool,
) -> Tuple[SubjectId, ...]:
    if isinstance(subjects, (str, bytes)):
        raise InvalidSplitError("{} must be an iterable of subject IDs".format(field_name))

    normalized: List[SubjectId] = []
    seen = set()
    try:
        iterator = iter(subjects)
    except TypeError as exc:
        raise InvalidSplitError("{} must be an iterable of subject IDs".format(field_name)) from exc

    for raw_subject in iterator:
        subject_id = _normalize_subject_id(raw_subject, field_name=field_name)
        if subject_id in seen:
            raise DuplicateSubjectError(
                "{} contains duplicate subject {}".format(field_name, _format_subject(subject_id))
            )
        seen.add(subject_id)
        normalized.append(subject_id)

    if not normalized:
        raise MissingSubjectError("{} must contain at least one subject".format(field_name))

    normalized_tuple = tuple(sorted(normalized))
    if require_complete_cohort:
        missing = tuple(sid for sid in CANONICAL_SUBJECTS if sid not in seen)
        if missing:
            raise MissingSubjectError(
                "{} is missing canonical subject(s): {}".format(
                    field_name, ", ".join(_format_subject(sid) for sid in missing)
                )
            )
    return normalized_tuple


def validate_subjects(
    subjects: Iterable[SubjectKey] = CANONICAL_SUBJECTS,
    *,
    require_complete_cohort: bool = True,
) -> Tuple[SubjectId, ...]:
    """Validate and normalize subject IDs.

    Production LOSO callers should keep the default complete-cohort guard.
    Synthetic compact fixtures can pass ``require_complete_cohort=False``
    while still receiving duplicate and unknown-ID checks.
    """

    return _normalize_subject_sequence(
        subjects,
        field_name="subjects",
        require_complete_cohort=require_complete_cohort,
    )


def canonical_loso_folds() -> Tuple[LosoFold, ...]:
    """Return all 12 canonical LOSO folds in ascending held-out order."""

    return tuple(
        LosoFold(
            held_out_subject=held_out,
            training_subjects=tuple(
                sid for sid in CANONICAL_SUBJECTS if sid != held_out
            ),
        )
        for held_out in CANONICAL_SUBJECTS
    )


def _validate_fraction(value: float, *, name: str) -> float:
    try:
        fraction = float(value)
    except (TypeError, ValueError) as exc:
        raise InvalidSplitError("{} must be a finite fraction in (0, 1)".format(name)) from exc
    if not math.isfinite(fraction) or not 0.0 < fraction < 1.0:
        raise InvalidSplitError("{} must be a finite fraction in (0, 1)".format(name))
    return fraction


def _validate_seed(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise InvalidSplitError("seed must be an integer")
    return int(seed)


def _select_vae_subjects(
    subjects: Tuple[SubjectId, ...],
    held_out_subject: SubjectKey,
    *,
    val_fraction: float,
    seed: int,
) -> SubjectSplit:
    held_out = _normalize_subject_id(held_out_subject, field_name="held_out_subject")
    if held_out not in subjects:
        raise MissingSubjectError(
            "held_out_subject={} is not present in the supplied subject cohort".format(
                _format_subject(held_out)
            )
        )

    fraction = _validate_fraction(val_fraction, name="val_fraction")
    normalized_seed = _validate_seed(seed)
    training_pool = [sid for sid in subjects if sid != held_out]
    if not training_pool:
        raise InvalidSplitError("a LOSO split needs at least one non-held-out subject")

    # This is intentionally RandomState rather than default_rng: it is the
    # source-compatible MT19937 shuffle used by VAE_logic.py.
    rng = np.random.RandomState(normalized_seed)
    shuffled = list(training_pool)
    rng.shuffle(shuffled)

    if len(shuffled) <= 1:
        validation_subjects: Tuple[SubjectId, ...] = ()
        train_subjects = tuple(shuffled)
    else:
        n_validation = max(1, int(len(shuffled) * fraction))
        n_validation = min(n_validation, len(shuffled) - 1)
        validation_subjects = tuple(shuffled[:n_validation])
        train_subjects = tuple(shuffled[n_validation:])

    metadata = SplitMetadata(
        protocol="vae_subject_loso",
        validation_unit="subject",
        validation_fraction=fraction,
        seed=normalized_seed,
        random_state="numpy.random.RandomState",
        canonical_cohort=CANONICAL_SUBJECTS,
        available_subjects=subjects,
        train_subjects=train_subjects,
        validation_subjects=validation_subjects,
        held_out_subject=held_out,
        window_counts_by_subject={},
        window_counts={"train": 0, "validation": 0, "held_out_test": 0},
        shapes={"train": (), "validation": (), "held_out_test": ()},
    )
    return SubjectSplit(
        train_subjects=train_subjects,
        validation_subjects=validation_subjects,
        held_out_subject=held_out,
        metadata=metadata,
    )


def split_vae_subjects(
    subjects: Iterable[SubjectKey] = CANONICAL_SUBJECTS,
    held_out_subject: SubjectKey = 1,
    *,
    val_fraction: float = VAE_SUBJECT_VALIDATION_FRACTION,
    seed: int = DEFAULT_SPLIT_SEED,
    require_complete_cohort: bool = False,
) -> SubjectSplit:
    """Create a deterministic VAE-safe subject-level LOSO assignment.

    ``require_complete_cohort=True`` is the production guard for all twelve
    canonical subjects.  It is optional only so the four-subject synthetic
    fixture can exercise the same algorithm without inventing missing data.
    """

    normalized = _normalize_subject_sequence(
        subjects,
        field_name="subjects",
        require_complete_cohort=require_complete_cohort,
    )
    return _select_vae_subjects(
        normalized,
        held_out_subject,
        val_fraction=val_fraction,
        seed=seed,
    )


def _normalize_subject_mapping(
    values: Mapping[SubjectKey, Any],
    *,
    field_name: str,
) -> Dict[SubjectId, Any]:
    if not isinstance(values, Mapping):
        raise InvalidSplitError("{} must be a mapping keyed by subject ID".format(field_name))

    normalized: Dict[SubjectId, Any] = {}
    for raw_subject, value in values.items():
        subject_id = _normalize_subject_id(raw_subject, field_name=field_name)
        if subject_id in normalized:
            raise DuplicateSubjectError(
                "{} contains duplicate subject {} after normalization".format(
                    field_name, _format_subject(subject_id)
                )
            )
        normalized[subject_id] = value

    if not normalized:
        raise MissingSubjectError("{} must contain at least one subject".format(field_name))
    return normalized


def _validate_window_mapping(
    subject_windows: Mapping[SubjectKey, Any],
    *,
    labels_by_subject: Optional[Mapping[SubjectKey, Any]],
    cohort: Optional[Iterable[SubjectKey]],
    require_complete_cohort: bool,
) -> Tuple[Dict[SubjectId, np.ndarray], Optional[Dict[SubjectId, np.ndarray]], Tuple[SubjectId, ...]]:
    windows_by_subject_raw = _normalize_subject_mapping(
        subject_windows,
        field_name="subject_windows",
    )
    available = tuple(sorted(windows_by_subject_raw))

    if cohort is not None:
        requested = _normalize_subject_sequence(
            cohort,
            field_name="cohort",
            require_complete_cohort=require_complete_cohort,
        )
        missing = tuple(sid for sid in requested if sid not in windows_by_subject_raw)
        extra = tuple(sid for sid in windows_by_subject_raw if sid not in requested)
        if missing:
            raise MissingSubjectError(
                "subject_windows is missing cohort subject(s): {}".format(
                    ", ".join(_format_subject(sid) for sid in missing)
                )
            )
        if extra:
            raise UnknownSubjectError(
                "subject_windows contains subject(s) outside cohort: {}".format(
                    ", ".join(_format_subject(sid) for sid in extra)
                )
            )
        available = requested
    elif require_complete_cohort:
        missing = tuple(sid for sid in CANONICAL_SUBJECTS if sid not in windows_by_subject_raw)
        if missing:
            raise MissingSubjectError(
                "subject_windows is missing canonical subject(s): {}".format(
                    ", ".join(_format_subject(sid) for sid in missing)
                )
            )
        available = CANONICAL_SUBJECTS

    windows_by_subject: Dict[SubjectId, np.ndarray] = {}
    common_shape: Optional[Tuple[int, int]] = None
    common_dtype: Optional[np.dtype] = None
    for subject_id in available:
        array = np.asarray(windows_by_subject_raw[subject_id])
        if array.ndim != 3:
            raise InvalidSplitError(
                "subject {} windows must have shape [N, C, T], got {}".format(
                    _format_subject(subject_id), tuple(array.shape)
                )
            )
        shape = (int(array.shape[1]), int(array.shape[2]))
        if common_shape is None:
            common_shape = shape
            common_dtype = array.dtype
        elif shape != common_shape:
            raise InvalidSplitError(
                "all subject windows must share [C, T]; subject {} has {} but expected {}".format(
                    _format_subject(subject_id), shape, common_shape
                )
            )
        windows_by_subject[subject_id] = array

    labels_by_subject_normalized: Optional[Dict[SubjectId, np.ndarray]] = None
    if labels_by_subject is not None:
        labels_raw = _normalize_subject_mapping(
            labels_by_subject,
            field_name="labels_by_subject",
        )
        missing_labels = tuple(sid for sid in available if sid not in labels_raw)
        extra_labels = tuple(sid for sid in labels_raw if sid not in available)
        if missing_labels:
            raise MissingSubjectError(
                "labels_by_subject is missing subject(s): {}".format(
                    ", ".join(_format_subject(sid) for sid in missing_labels)
                )
            )
        if extra_labels:
            raise UnknownSubjectError(
                "labels_by_subject contains subject(s) not in subject_windows: {}".format(
                    ", ".join(_format_subject(sid) for sid in extra_labels)
                )
            )
        labels_by_subject_normalized = {}
        for subject_id in available:
            labels = np.asarray(labels_raw[subject_id])
            if labels.ndim != 1:
                raise InvalidSplitError(
                    "labels for subject {} must have shape [N], got {}".format(
                        _format_subject(subject_id), tuple(labels.shape)
                    )
                )
            if labels.shape[0] != windows_by_subject[subject_id].shape[0]:
                raise InvalidSplitError(
                    "labels for subject {} have {} rows but windows have {}".format(
                        _format_subject(subject_id),
                        labels.shape[0],
                        windows_by_subject[subject_id].shape[0],
                    )
                )
            labels_by_subject_normalized[subject_id] = labels

    # The local assignments above intentionally retain arrays as views; the
    # materializer below concatenates into fresh outputs and never mutates the
    # caller's mapping or arrays.
    del common_dtype
    return windows_by_subject, labels_by_subject_normalized, available


def _materialize_partition(
    subject_ids: Sequence[SubjectId],
    windows_by_subject: Mapping[SubjectId, np.ndarray],
    labels_by_subject: Optional[Mapping[SubjectId, np.ndarray]],
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    arrays = [windows_by_subject[sid] for sid in subject_ids]
    if arrays:
        windows = np.concatenate(arrays, axis=0)
    else:
        first = next(iter(windows_by_subject.values()))
        windows = np.empty((0, first.shape[1], first.shape[2]), dtype=first.dtype)

    if labels_by_subject is None:
        return windows, None
    label_arrays = [labels_by_subject[sid] for sid in subject_ids]
    if label_arrays:
        labels = np.concatenate(label_arrays, axis=0)
    else:
        first_labels = next(iter(labels_by_subject.values()))
        labels = np.empty((0,), dtype=first_labels.dtype)
    return windows, labels


def split_vae_windows(
    subject_windows: Mapping[SubjectKey, Any],
    held_out_subject: SubjectKey,
    *,
    labels_by_subject: Optional[Mapping[SubjectKey, Any]] = None,
    cohort: Optional[Iterable[SubjectKey]] = None,
    val_fraction: float = VAE_SUBJECT_VALIDATION_FRACTION,
    seed: int = DEFAULT_SPLIT_SEED,
    require_complete_cohort: bool = False,
) -> VaeSplitResult:
    """Materialize a VAE-safe LOSO split from subject-keyed windows.

    Inputs must be ``[N_subject_windows, C, T]`` arrays.  The compact fixture
    may pass ``cohort=(1, 2, 3, 5)``; production callers should pass the full
    canonical cohort or set ``require_complete_cohort=True``.  Labels are
    optional but, when supplied, must be one-dimensional and subject-aligned.
    """

    windows_by_subject, labels_by_subject_normalized, available = _validate_window_mapping(
        subject_windows,
        labels_by_subject=labels_by_subject,
        cohort=cohort,
        require_complete_cohort=require_complete_cohort,
    )
    assignment = _select_vae_subjects(
        available,
        held_out_subject,
        val_fraction=val_fraction,
        seed=seed,
    )

    train_windows, train_labels = _materialize_partition(
        assignment.train_subjects,
        windows_by_subject,
        labels_by_subject_normalized,
    )
    validation_windows, validation_labels = _materialize_partition(
        assignment.validation_subjects,
        windows_by_subject,
        labels_by_subject_normalized,
    )
    held_out_test_windows, held_out_test_labels = _materialize_partition(
        (assignment.held_out_subject,),
        windows_by_subject,
        labels_by_subject_normalized,
    )

    counts_by_subject = {
        sid: int(windows_by_subject[sid].shape[0]) for sid in available
    }
    metadata = SplitMetadata(
        protocol=assignment.metadata.protocol,
        validation_unit=assignment.metadata.validation_unit,
        validation_fraction=assignment.metadata.validation_fraction,
        seed=assignment.metadata.seed,
        random_state=assignment.metadata.random_state,
        canonical_cohort=assignment.metadata.canonical_cohort,
        available_subjects=assignment.metadata.available_subjects,
        train_subjects=assignment.train_subjects,
        validation_subjects=assignment.validation_subjects,
        held_out_subject=assignment.held_out_subject,
        window_counts_by_subject=counts_by_subject,
        window_counts={
            "train": int(train_windows.shape[0]),
            "validation": int(validation_windows.shape[0]),
            "held_out_test": int(held_out_test_windows.shape[0]),
        },
        shapes={
            "train": tuple(int(value) for value in train_windows.shape),
            "validation": tuple(int(value) for value in validation_windows.shape),
            "held_out_test": tuple(int(value) for value in held_out_test_windows.shape),
        },
    )
    return VaeSplitResult(
        train_windows=train_windows,
        validation_windows=validation_windows,
        held_out_test_windows=held_out_test_windows,
        train_labels=train_labels,
        validation_labels=validation_labels,
        held_out_test_labels=held_out_test_labels,
        metadata=metadata,
    )


def make_vae_loso_split(*args: Any, **kwargs: Any) -> VaeSplitResult:
    """Explicitly named synonym for :func:`split_vae_windows`."""

    return split_vae_windows(*args, **kwargs)


def _validate_cnn_inputs(windows: Any, labels: Any) -> Tuple[np.ndarray, np.ndarray]:
    window_array = np.asarray(windows)
    label_array = np.asarray(labels)
    if window_array.ndim != 3:
        raise InvalidSplitError(
            "CNN windows must have shape [N, C, T], got {}".format(
                tuple(window_array.shape)
            )
        )
    if label_array.ndim != 1:
        raise InvalidSplitError(
            "CNN labels must have shape [N], got {}".format(tuple(label_array.shape))
        )
    if window_array.shape[0] != label_array.shape[0]:
        raise InvalidSplitError(
            "CNN windows and labels disagree on N: {} versus {}".format(
                window_array.shape[0], label_array.shape[0]
            )
        )
    if window_array.shape[0] < 2:
        raise InvalidSplitError("CNN window split needs at least two windows")
    return window_array, label_array


def _stratified_indices(
    labels: np.ndarray,
    *,
    validation_count: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Return deterministic class-balanced indices without sklearn.

    The public package has no sklearn dependency.  When every class can
    support a stratified allocation, one seeded RandomState shuffle is used
    per sorted class; otherwise the documented fallback is a seeded global
    permutation.  The fallback is still deterministic and is recorded in
    metadata rather than silently pretending to be stratified.
    """

    labels = np.asarray(labels)
    classes, counts = np.unique(labels, return_counts=True)
    n_classes = int(classes.size)
    if n_classes == 0:
        raise InvalidSplitError("CNN labels must contain at least one class")

    n_train = int(labels.size - validation_count)
    can_stratify = (
        validation_count >= n_classes
        and n_train >= n_classes
        and bool(np.all(counts >= 2))
    )
    rng = np.random.RandomState(seed)
    if not can_stratify:
        shuffled = rng.permutation(labels.size)
        validation_indices = np.asarray(shuffled[:validation_count], dtype=np.int64)
        train_indices = np.asarray(shuffled[validation_count:], dtype=np.int64)
        return train_indices, validation_indices, False

    exact = (counts.astype(np.float64) * validation_count) / float(labels.size)
    allocation = np.floor(exact).astype(np.int64)
    allocation[allocation == 0] = 1
    if int(allocation.sum()) > validation_count:
        # Remove excess from the classes furthest above their required one
        # sample, with stable class-order tie breaking.
        while int(allocation.sum()) > validation_count:
            candidates = np.flatnonzero(allocation > 1)
            if candidates.size == 0:
                break
            candidate = min(
                candidates.tolist(),
                key=lambda index: (exact[index] - allocation[index], int(index)),
            )
            allocation[candidate] -= 1
    elif int(allocation.sum()) < validation_count:
        remaining = validation_count - int(allocation.sum())
        order = sorted(
            range(n_classes),
            key=lambda index: (-(exact[index] - math.floor(exact[index])), index),
        )
        for index in order[:remaining]:
            allocation[index] += 1

    train_parts: List[np.ndarray] = []
    validation_parts: List[np.ndarray] = []
    for class_index, class_value in enumerate(classes):
        class_indices = np.flatnonzero(labels == class_value).astype(np.int64)
        shuffled_class = rng.permutation(class_indices)
        n_class_validation = int(allocation[class_index])
        validation_parts.append(shuffled_class[:n_class_validation])
        train_parts.append(shuffled_class[n_class_validation:])

    train_indices = np.concatenate(train_parts).astype(np.int64, copy=False)
    validation_indices = np.concatenate(validation_parts).astype(np.int64, copy=False)
    return train_indices, validation_indices, True


def split_cnn_windows(
    windows: Any,
    labels: Any,
    *,
    validation_fraction: float = CNN_WINDOW_VALIDATION_FRACTION,
    seed: int = DEFAULT_SPLIT_SEED,
) -> CnnWindowSplit:
    """Split windows for the downstream CNN at the distinct 0.20 stage.

    This function accepts windows, not subject groups.  It must not be used as
    the VAE subject validation split; that separation is explicit in the
    returned metadata and result type.
    """

    window_array, label_array = _validate_cnn_inputs(windows, labels)
    fraction = _validate_fraction(
        validation_fraction,
        name="CNN validation_fraction",
    )
    normalized_seed = _validate_seed(seed)
    validation_count = max(1, int(math.ceil(window_array.shape[0] * fraction)))
    if validation_count >= window_array.shape[0]:
        raise InvalidSplitError("CNN validation fraction leaves no training windows")

    train_indices, validation_indices, stratified = _stratified_indices(
        label_array,
        validation_count=validation_count,
        seed=normalized_seed,
    )
    metadata = CnnWindowSplitMetadata(
        protocol="cnn_internal_window_validation",
        validation_unit="window",
        validation_fraction=fraction,
        seed=normalized_seed,
        stratified=stratified,
        separate_from_vae_subject_split=True,
        train_count=int(train_indices.size),
        validation_count=int(validation_indices.size),
    )
    return CnnWindowSplit(
        train_windows=window_array[train_indices],
        validation_windows=window_array[validation_indices],
        train_labels=label_array[train_indices],
        validation_labels=label_array[validation_indices],
        train_indices=train_indices,
        validation_indices=validation_indices,
        metadata=metadata,
    )


def split_cnn_train_validation(*args: Any, **kwargs: Any) -> CnnWindowSplit:
    """Explicit synonym for :func:`split_cnn_windows`."""

    return split_cnn_windows(*args, **kwargs)


__all__ = [
    "CANONICAL_PAPER_SUBJECTS",
    "CANONICAL_SUBJECTS",
    "CNN_WINDOW_VALIDATION_FRACTION",
    "CnnWindowSplit",
    "CnnWindowSplitMetadata",
    "DEFAULT_SPLIT_SEED",
    "DuplicateSubjectError",
    "InvalidSplitError",
    "LosoFold",
    "MissingSubjectError",
    "SplitError",
    "SplitMetadata",
    "SubjectSplit",
    "UnknownSubjectError",
    "VAE_SUBJECT_VALIDATION_FRACTION",
    "VaeSplitResult",
    "canonical_loso_folds",
    "make_vae_loso_split",
    "split_cnn_train_validation",
    "split_cnn_windows",
    "split_vae_subjects",
    "split_vae_windows",
    "validate_subjects",
]
