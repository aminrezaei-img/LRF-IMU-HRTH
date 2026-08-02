"""Portable discovery and loading for the audited REALDISP log subset.

Only the paper-specific ideal-placement/right-thigh input schema is covered
here.  The raw dataset remains an explicit external input; this module never
downloads data, consults environment variables, or creates files at import
time.

The source project always selected six right-thigh channels.  The public
three-channel path below is an explicit reconstruction of the intended
accelerometer-only schema (columns 80--82), not a claim that the historical
parser recovered that path.  Six- and three-channel arrays therefore have
separate, named entry points and must be used with separately trained models.
"""

from __future__ import annotations

import csv
import importlib
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple, Union


PathLike = Union[str, os.PathLike]


# These exports are intentionally stable and mirror the public schema module.
# The values are also used as a compatibility fallback while the sibling
# schema/activity modules are imported lazily (which keeps package import
# side-effect free and permits focused use during staged integration).
EXPECTED_COLUMN_COUNT = 120
LABEL_COLUMN_INDEX = 119
RIGHT_THIGH_SIX_CHANNEL_INDICES = (80, 81, 82, 83, 84, 85)
RIGHT_THIGH_ACCELEROMETER_INDICES = (80, 81, 82)
SIX_CHANNEL_NAMES = ("ax", "ay", "az", "gx", "gy", "gz")
THREE_CHANNEL_NAMES = ("ax", "ay", "az")
DEFAULT_SAMPLING_FREQUENCY_HZ = 50
DEFAULT_WINDOW_SAMPLES = 160
DEFAULT_HOP_SAMPLES = 40

_SUBJECT_FILENAME_RE = re.compile(
    r"^subject(?P<subject_id>[0-9]+)_ideal\.log$", re.IGNORECASE
)
_RAW_TO_ENCODED_FALLBACK = {1: 0, 3: 1, 4: 2, 33: 3}


class RealDISPError(ValueError):
    """Base error for malformed or unsupported REALDISP input."""


class RealDISPLogError(RealDISPError):
    """Raised when a log is not a valid numeric 120-column TSV."""


@dataclass(frozen=True)
class RealDISPSubject:
    """A loaded subject's selected channels and raw activity labels.

    ``signals`` is shaped ``[samples, channels]`` and is always ``float32``.
    ``raw_labels`` retains the source activity codes (1, 3, 4, 33, or other
    source codes such as a synthetic fixture's excluded-label marker); it is
    not the encoded classifier label space.
    """

    subject_id: int
    path: Path
    signals: Any
    raw_labels: Any


def _numpy() -> Any:
    """Import numpy only when an array operation is requested."""

    try:
        return importlib.import_module("numpy")
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on runtime
        raise ImportError(
            "REALDISP loading requires the already-supported numpy runtime "
            "dependency; install the project's test/scientific environment."
        ) from exc


def _optional_module(name: str) -> Optional[Any]:
    """Return a sibling contract module without making it an import side effect."""

    try:
        return importlib.import_module("." + name, package=__package__)
    except ModuleNotFoundError as exc:
        # During staged milestone integration the sibling task may not have
        # landed yet.  A missing sibling module is safe to handle with the
        # frozen contract fallback; errors raised from an existing module are
        # not swallowed.
        if exc.name in {
            (str(__package__) + "." + name),
            name,
        }:
            return None
        raise


def _first_attribute(module: Optional[Any], names: Sequence[str], default: Any) -> Any:
    if module is None:
        return default
    for name in names:
        if hasattr(module, name):
            return getattr(module, name)
    return default


def _schema_values() -> Tuple[int, int, Tuple[int, ...], Tuple[int, ...]]:
    """Read schema constants from ``schema.py`` with contract-safe fallbacks."""

    schema = _optional_module("schema")
    expected = int(
        _first_attribute(
            schema,
            (
                "EXPECTED_COLUMN_COUNT",
                "EXPECTED_RAW_COLUMN_COUNT",
                "REQUIRED_RAW_COLUMNS",
                "RAW_COLUMN_COUNT",
            ),
            EXPECTED_COLUMN_COUNT,
        )
    )
    label = int(
        _first_attribute(
            schema,
            (
                "LABEL_COLUMN_INDEX",
                "ACTIVITY_LABEL_COLUMN_INDEX",
                "LABEL_INDEX",
                "RAW_LABEL_COLUMN",
            ),
            LABEL_COLUMN_INDEX,
        )
    )
    six = tuple(
        int(index)
        for index in _first_attribute(
            schema,
            (
                "RIGHT_THIGH_SIX_CHANNEL_INDICES",
                "RIGHT_THIGH_CHANNEL_INDICES",
                "SIX_CHANNEL_INDICES",
            ),
            RIGHT_THIGH_SIX_CHANNEL_INDICES,
        )
    )
    three = tuple(
        int(index)
        for index in _first_attribute(
            schema,
            (
                "RIGHT_THIGH_ACCELEROMETER_INDICES",
                "THREE_CHANNEL_INDICES",
                "ACCELEROMETER_ONLY_INDICES",
                "ACCELEROMETER_CHANNEL_INDICES",
            ),
            RIGHT_THIGH_ACCELEROMETER_INDICES,
        )
    )
    if expected <= 0:
        raise RealDISPError("schema expected column count must be positive")
    if label < 0 or label >= expected:
        raise RealDISPError(
            "schema label column index must be within the expected raw column count"
        )
    if len(six) != 6 or len(three) != 3:
        raise RealDISPError("REALDISP schema must declare six- and three-channel selections")
    if len(set(six)) != len(six) or len(set(three)) != len(three):
        raise RealDISPError("REALDISP channel selections must not contain duplicates")
    if any(index < 0 or index >= expected for index in six + three):
        raise RealDISPError("REALDISP channel selections must be valid raw column indices")
    return expected, label, six, three


def _activity_mapping() -> Dict[int, int]:
    """Read the raw-to-encoded mapping from ``activities.py`` if available."""

    activities = _optional_module("activities")
    candidate = _first_attribute(
        activities,
        (
            "RAW_TO_ENCODED_LABEL",
            "RAW_ACTIVITY_CODE_TO_LABEL",
            "ACTIVITY_CODE_TO_LABEL",
            "CODE_TO_ENCODED_LABEL",
            "RAW_TO_ENCODED",
        ),
        _RAW_TO_ENCODED_FALLBACK,
    )
    if callable(candidate):
        # A callable activity encoder is not a mapping and should not be
        # guessed at here; use the frozen four-class contract instead.
        return dict(_RAW_TO_ENCODED_FALLBACK)
    try:
        mapping = {int(key): int(value) for key, value in dict(candidate).items()}
    except (TypeError, ValueError, AttributeError) as exc:
        raise RealDISPError("activities.py must expose an integer raw-to-encoded mapping") from exc
    if not mapping:
        raise RealDISPError("activities.py raw-to-encoded activity mapping is empty")
    return mapping


def extract_subject_id(filename: PathLike) -> int:
    """Extract and normalize a subject ID from an exact ideal-log filename.

    Matching is anchored at both ends and case-insensitive.  Leading zeros
    are normalized by converting the captured digits to ``int``.  Paths and
    bare filenames are both accepted; directory names never participate in
    the match.
    """

    name = Path(filename).name
    match = _SUBJECT_FILENAME_RE.fullmatch(name)
    if match is None:
        raise ValueError(
            "Expected an exact subject*_ideal.log filename; got {!r}".format(name)
        )
    return int(match.group("subject_id"))


def _normalize_subject_allowlist(values: Optional[Iterable[Any]]) -> Optional[Tuple[int, ...]]:
    if values is None:
        return None
    if isinstance(values, (str, bytes)):
        pieces = [piece.strip() for piece in str(values).split(",") if piece.strip()]
        values = pieces
    normalized = []
    for value in values:
        if isinstance(value, bool):
            raise TypeError("subject IDs must be integers, not booleans")
        try:
            integer = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("subject IDs must be integer-like values") from exc
        if str(value).strip() != str(integer) and not isinstance(value, int):
            # Accept ordinary zero-padded strings such as "01", but reject
            # fractional values that int() would silently truncate.
            try:
                if float(value) != float(integer):
                    raise ValueError
            except (TypeError, ValueError) as exc:
                raise ValueError("subject IDs must be integer-like values") from exc
        normalized.append(integer)
    return tuple(sorted(set(normalized)))


def discover_subject_logs(
    root: PathLike,
    subjects: Optional[Iterable[int]] = None,
    *,
    allowed_subjects: Optional[Iterable[int]] = None,
    allowlist: Optional[Iterable[int]] = None,
    subject_ids: Optional[Iterable[int]] = None,
) -> Dict[int, Path]:
    """Discover direct-child ``subject*_ideal.log`` files under ``root``.

    Discovery is deliberately nonrecursive and deterministic.  All matching
    files are normalized before an optional allowlist is applied, so duplicate
    IDs can never be silently hidden by a filter.  Missing allowlisted IDs are
    reported instead of silently omitted.
    """

    root_path = Path(root) if root is not None else None
    if root_path is None:
        raise TypeError("REALDISP data root is required")
    root_path = root_path.expanduser()
    if not root_path.exists():
        raise FileNotFoundError("REALDISP data root does not exist: {}".format(root_path))
    if not root_path.is_dir():
        raise NotADirectoryError("REALDISP data root is not a directory: {}".format(root_path))

    supplied_allowlists = [
        ("subjects", subjects),
        ("allowed_subjects", allowed_subjects),
        ("allowlist", allowlist),
        ("subject_ids", subject_ids),
    ]
    supplied = [(name, value) for name, value in supplied_allowlists if value is not None]
    if len(supplied) > 1:
        raise TypeError("provide only one of subjects, allowed_subjects, allowlist, or subject_ids")
    requested = _normalize_subject_allowlist(supplied[0][1] if supplied else None)

    discovered: Dict[int, Path] = {}
    for path in sorted(root_path.iterdir(), key=lambda item: item.name.casefold()):
        if not path.is_file() or _SUBJECT_FILENAME_RE.fullmatch(path.name) is None:
            continue
        subject_id = extract_subject_id(path.name)
        previous = discovered.get(subject_id)
        if previous is not None:
            raise ValueError(
                "Duplicate normalized subject ID {} for {!s} and {!s}".format(
                    subject_id, previous, path
                )
            )
        discovered[subject_id] = path

    if not discovered:
        raise FileNotFoundError(
            "No direct-child subject*_ideal.log files found under: {}".format(root_path)
        )

    if requested is not None:
        missing = sorted(set(requested) - set(discovered))
        if missing:
            raise FileNotFoundError(
                "Requested REALDISP subject IDs not found under {}: {}".format(
                    root_path, missing
                )
            )
        discovered = {subject_id: discovered[subject_id] for subject_id in requested}
    return dict(sorted(discovered.items()))


def discover_realdisp_logs(
    root: PathLike,
    subjects: Optional[Iterable[int]] = None,
    *,
    allowed_subjects: Optional[Iterable[int]] = None,
    allowlist: Optional[Iterable[int]] = None,
    subject_ids: Optional[Iterable[int]] = None,
) -> Dict[int, Path]:
    """Named public alias for :func:`discover_subject_logs`."""

    return discover_subject_logs(
        root,
        subjects,
        allowed_subjects=allowed_subjects,
        allowlist=allowlist,
        subject_ids=subject_ids,
    )


def load_realdisp_log(path: PathLike) -> Any:
    """Load one numeric, headerless, tab-separated REALDISP log.

    The returned array is two-dimensional ``float64`` with exactly 120
    columns.  Signals and labels are cast to their public dtypes by the
    extraction functions.  Row and column locations are included in errors so
    malformed files can be repaired without inspecting participant data.
    """

    np = _numpy()
    file_path = Path(path) if path is not None else None
    if file_path is None:
        raise TypeError("REALDISP log path is required")
    file_path = file_path.expanduser()
    if not file_path.exists():
        raise FileNotFoundError("REALDISP log does not exist: {}".format(file_path))
    if not file_path.is_file():
        raise IsADirectoryError("REALDISP log path is not a file: {}".format(file_path))
    expected_columns, _, _, _ = _schema_values()

    rows = []
    try:
        with file_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle, delimiter="\t")
            for row_number, row in enumerate(reader, start=1):
                if not row:
                    raise RealDISPLogError(
                        "Empty row {} in REALDISP log {}".format(row_number, file_path.name)
                    )
                if len(row) != expected_columns:
                    raise RealDISPLogError(
                        "Expected {} tab-separated columns in {} at row {}; got {}".format(
                            expected_columns, file_path.name, row_number, len(row)
                        )
                    )
                parsed_row = []
                for column_number, token in enumerate(row, start=1):
                    try:
                        value = float(token.strip())
                    except (TypeError, ValueError) as exc:
                        raise RealDISPLogError(
                            "Non-numeric value in {} at row {}, column {}: {!r}".format(
                                file_path.name, row_number, column_number, token
                            )
                        ) from exc
                    if not math.isfinite(value):
                        raise RealDISPLogError(
                            "Non-finite numeric value in {} at row {}, column {}: {!r}".format(
                                file_path.name, row_number, column_number, token
                            )
                        )
                    parsed_row.append(value)
                rows.append(parsed_row)
    except UnicodeDecodeError as exc:
        raise RealDISPLogError("Could not decode REALDISP log {} as UTF-8".format(file_path)) from exc
    except OSError as exc:
        raise OSError("Could not read REALDISP log {}: {}".format(file_path, exc)) from exc

    if not rows:
        raise RealDISPLogError("REALDISP log is empty: {}".format(file_path))
    return np.asarray(rows, dtype=np.float64)


def _coerce_raw_matrix(data: Any) -> Any:
    np = _numpy()
    if isinstance(data, (str, os.PathLike, Path)):
        matrix = load_realdisp_log(data)
    else:
        try:
            matrix = np.asarray(data)
        except (TypeError, ValueError) as exc:
            raise RealDISPLogError("REALDISP raw data must be a numeric 2-D array") from exc
        if matrix.ndim != 2:
            raise RealDISPLogError(
                "REALDISP raw data must be a 2-D array; got {} dimensions".format(matrix.ndim)
            )
        expected_columns, _, _, _ = _schema_values()
        if matrix.shape[1] != expected_columns:
            raise RealDISPLogError(
                "Expected {} columns in REALDISP raw data; got {}".format(
                    expected_columns, matrix.shape[1]
                )
            )
        try:
            matrix = matrix.astype(np.float64, copy=False)
        except (TypeError, ValueError) as exc:
            raise RealDISPLogError("REALDISP raw data must contain numeric values") from exc
        if not np.isfinite(matrix).all():
            raise RealDISPLogError("REALDISP raw data contains non-finite numeric values")
    return matrix


def extract_raw_activity_labels(data: Any) -> Any:
    """Extract source activity codes from zero-based column 119 as ``int32``."""

    np = _numpy()
    matrix = _coerce_raw_matrix(data)
    _, label_index, _, _ = _schema_values()
    labels = np.asarray(matrix[:, label_index], dtype=np.float64)
    if not np.equal(labels, np.floor(labels)).all():
        raise RealDISPLogError("REALDISP activity labels must be integer-valued")
    return labels.astype(np.int32, copy=False)


def _channel_indices(variant: str) -> Tuple[int, ...]:
    if not isinstance(variant, str):
        raise TypeError("REALDISP channel variant must be a string")
    key = variant.strip().lower().replace("-", "_").replace(" ", "_")
    six_aliases = {"six", "6ch", "6_channel", "six_channel", "full", "full_6ch"}
    three_aliases = {
        "three",
        "3ch",
        "3_channel",
        "three_channel",
        "accelerometer",
        "accelerometer_only",
        "accel",
        "acc_only",
    }
    _, _, six, three = _schema_values()
    if key in six_aliases:
        return six
    if key in three_aliases:
        return three
    raise ValueError(
        "Unknown REALDISP channel variant {!r}; choose 'six_channel' or "
        "'accelerometer_only' (the explicit reconstructed 3CH path)".format(variant)
    )


def extract_sensor_channels(data: Any, variant: str = "six_channel") -> Any:
    """Extract an explicit right-thigh channel schema as ``float32``.

    ``six_channel`` selects columns ``80..85`` in ``ax, ay, az, gx, gy, gz``
    order.  ``accelerometer_only``/``three_channel`` selects columns ``80..82``
    in ``ax, ay, az`` order as a separately trained 3CH reconstruction.  It is
    not an inference-time drop from an already extracted six-channel array.
    """

    np = _numpy()
    matrix = _coerce_raw_matrix(data)
    indices = _channel_indices(variant)
    return np.asarray(matrix[:, indices], dtype=np.float32)


def load_subject_data(path: PathLike, variant: str = "six_channel") -> Tuple[Any, Any]:
    """Load one log and return ``(signals, raw_activity_labels)``."""

    raw = load_realdisp_log(path)
    return extract_sensor_channels(raw, variant), extract_raw_activity_labels(raw)


def load_realdisp_subjects(
    root: PathLike,
    variant: str = "six_channel",
    subjects: Optional[Iterable[int]] = None,
    *,
    allowed_subjects: Optional[Iterable[int]] = None,
    allowlist: Optional[Iterable[int]] = None,
    subject_ids: Optional[Iterable[int]] = None,
) -> Dict[int, RealDISPSubject]:
    """Discover and load selected subjects without writing participant data."""

    discovered = discover_subject_logs(
        root,
        subjects,
        allowed_subjects=allowed_subjects,
        allowlist=allowlist,
        subject_ids=subject_ids,
    )
    loaded = {}
    for subject_id, path in discovered.items():
        signals, raw_labels = load_subject_data(path, variant=variant)
        loaded[subject_id] = RealDISPSubject(
            subject_id=subject_id,
            path=path,
            signals=signals,
            raw_labels=raw_labels,
        )
    return loaded


def encode_activity_labels(raw_labels: Any) -> Any:
    """Map raw REALDISP activity codes to encoded labels without filtering."""

    np = _numpy()
    mapping = _activity_mapping()
    labels = np.asarray(raw_labels)
    if labels.ndim != 1:
        raise ValueError("raw activity labels must be a one-dimensional array")
    try:
        integer_labels = labels.astype(np.int64, copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("raw activity labels must be integer-valued") from exc
    if not np.equal(labels, integer_labels).all():
        raise ValueError("raw activity labels must be integer-valued")
    unsupported = sorted(set(int(item) for item in integer_labels) - set(mapping))
    if unsupported:
        raise ValueError(
            "Unsupported REALDISP activity code(s): {}; expected one of {}".format(
                unsupported, sorted(mapping)
            )
        )
    return np.asarray([mapping[int(item)] for item in integer_labels], dtype=np.int64)


# Compatibility spellings retained as thin aliases for the observed source
# vocabulary.  They do not introduce a global root or an implicit data path.
realdisp_discover_ideal_logs = discover_subject_logs
realdisp_load_log = load_realdisp_log
extract_channels = extract_sensor_channels
extract_right_thigh_channels = extract_sensor_channels
extract_activity_labels = extract_raw_activity_labels
extract_raw_labels = extract_raw_activity_labels
load_realdisp_subject = load_subject_data
def realdisp_list_subjects(root: PathLike, subjects: Optional[Iterable[int]] = None):
    return sorted(discover_subject_logs(root, subjects=subjects).keys())


__all__ = [
    "DEFAULT_HOP_SAMPLES",
    "DEFAULT_SAMPLING_FREQUENCY_HZ",
    "DEFAULT_WINDOW_SAMPLES",
    "EXPECTED_COLUMN_COUNT",
    "LABEL_COLUMN_INDEX",
    "RealDISPError",
    "RealDISPLogError",
    "RealDISPSubject",
    "RIGHT_THIGH_ACCELEROMETER_INDICES",
    "RIGHT_THIGH_SIX_CHANNEL_INDICES",
    "SIX_CHANNEL_NAMES",
    "THREE_CHANNEL_NAMES",
    "discover_realdisp_logs",
    "discover_subject_logs",
    "encode_activity_labels",
    "extract_activity_labels",
    "extract_channels",
    "extract_raw_activity_labels",
    "extract_raw_labels",
    "extract_right_thigh_channels",
    "extract_sensor_channels",
    "extract_subject_id",
    "load_realdisp_log",
    "load_realdisp_subject",
    "load_realdisp_subjects",
    "load_subject_data",
    "realdisp_discover_ideal_logs",
    "realdisp_list_subjects",
    "realdisp_load_log",
]
