"""HARTH-family CSV adapter for the ten-class LRF-IMU replacement.

The adapter reads the public CSV schema used by the HARTH, HAR70+, and adult
walking-speed datasets. Raw participant files remain external inputs. The
returned signals are time-major thigh accelerometer arrays, ready for the
shared LRF-IMU windowing and normalization stages.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Tuple

import numpy as np


HARTH_COLUMNS = ("timestamp", "back_x", "back_y", "back_z", "thigh_x", "thigh_y", "thigh_z", "label")
TARGET_SAMPLE_RATE_HZ = 50.0
TARGET_CLASS_NAMES = (
    "walking_slow", "walking_moderate", "walking_brisk", "running",
    "stair_climbing", "cycling_seated", "cycling_standing", "sitting",
    "standing", "lying",
)

# Source labels are intentionally dataset-qualified: label 1 means different
# things only if a future dataset is added with a different vocabulary.
SOURCE_LABEL_MAPS: Mapping[str, Mapping[int, int]] = {
    "adult_walking_speed": {101: 0, 102: 1, 103: 2, 2: 3},
    "harth": {2: 3, 4: 4, 5: 4, 13: 5, 14: 6, 7: 7, 6: 8, 8: 9},
    "har70plus": {4: 4, 5: 4, 7: 7, 6: 8, 8: 9},
}
EXCLUDED_SOURCE_LABELS: Mapping[str, Tuple[int, ...]] = {
    "harth": (1, 3, 130, 140),
    "har70plus": (1, 3),
    "adult_walking_speed": (),
}


class HARTHError(ValueError):
    """Raised for malformed HARTH-family input or incompatible options."""


@dataclass(frozen=True)
class HarthSubject:
    """One source recording with namespaced identity and time-major signals."""

    dataset: str
    subject_id: str
    key: str
    path: Path
    signals: np.ndarray
    raw_labels: np.ndarray
    encoded_labels: np.ndarray
    timestamps: Tuple[datetime, ...]
    sample_rate_hz: float
    excluded_row_count: int


def _dataset_for_directory(path: Path) -> str:
    name = path.name.casefold()
    aliases = {"harth": "harth", "har70plus": "har70plus", "adult_walking_speed": "adult_walking_speed"}
    if name not in aliases:
        raise HARTHError("unknown HARTH-family dataset directory: {}".format(path))
    return aliases[name]


def _parse_timestamp(value: str, path: Path, row: int) -> datetime:
    try:
        return datetime.fromisoformat(value.strip())
    except ValueError as exc:
        raise HARTHError("invalid timestamp in {} at row {}".format(path.name, row)) from exc


def _read_csv(path: Path, dataset: str, *, rate_tolerance_hz: float) -> HarthSubject:
    if not path.is_file():
        raise FileNotFoundError("HARTH-family CSV does not exist: {}".format(path))
    timestamps = []
    signals = []
    raw_labels = []
    expected = set(HARTH_COLUMNS)
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or set(reader.fieldnames) != expected or tuple(reader.fieldnames) != HARTH_COLUMNS:
                raise HARTHError("{} must have exactly the HARTH columns in documented order".format(path.name))
            for row_number, row in enumerate(reader, start=2):
                timestamp = _parse_timestamp(row["timestamp"], path, row_number)
                try:
                    values = [float(row[name]) for name in ("thigh_x", "thigh_y", "thigh_z")]
                    raw_label_float = float(row["label"])
                except (TypeError, ValueError) as exc:
                    raise HARTHError("non-numeric signal or label in {} at row {}".format(path.name, row_number)) from exc
                if not np.isfinite(values).all() or not np.isfinite(raw_label_float) or raw_label_float != int(raw_label_float):
                    raise HARTHError("non-finite or non-integer data in {} at row {}".format(path.name, row_number))
                timestamps.append(timestamp)
                signals.append(values)
                raw_labels.append(int(raw_label_float))
    except UnicodeDecodeError as exc:
        raise HARTHError("could not decode {} as UTF-8".format(path)) from exc
    if len(signals) < 2:
        raise HARTHError("{} must contain at least two samples".format(path.name))
    times = np.asarray([(item - timestamps[0]).total_seconds() for item in timestamps], dtype=np.float64)
    deltas = np.diff(times)
    if np.any(deltas <= 0):
        raise HARTHError("timestamps must be strictly increasing in {}".format(path.name))
    rate = float(1.0 / np.median(deltas))
    if abs(rate - TARGET_SAMPLE_RATE_HZ) > float(rate_tolerance_hz):
        raise HARTHError("{} has effective sample rate {:.6g} Hz; expected 50 Hz".format(path.name, rate))
    mapping = SOURCE_LABEL_MAPS[dataset]
    raw = np.asarray(raw_labels, dtype=np.int64)
    encoded = np.full(raw.shape, -1, dtype=np.int64)
    for source, target in mapping.items():
        encoded[raw == source] = target
    return HarthSubject(
        dataset=dataset,
        subject_id=path.stem,
        key="{}:{}".format(dataset, path.stem),
        path=path,
        signals=np.asarray(signals, dtype=np.float32),
        raw_labels=raw.astype(np.int32),
        encoded_labels=encoded,
        timestamps=tuple(timestamps),
        sample_rate_hz=rate,
        excluded_row_count=int(np.count_nonzero(encoded < 0)),
    )


def discover_harth_files(root: str | Path, composition: str = "harth_walking_speed") -> Dict[str, Path]:
    """Discover selected CSVs under a HARTH-family root deterministically."""
    base = Path(root).expanduser().resolve()
    if not base.is_dir():
        raise NotADirectoryError("HARTH data root is not a directory: {}".format(base))
    compositions = {
        "harth": ("harth",),
        "harth_walking_speed": ("harth", "adult_walking_speed"),
        "harth_walking_speed_har70plus": ("harth", "adult_walking_speed", "har70plus"),
    }
    if composition not in compositions:
        raise HARTHError("unknown composition {}; choose {}".format(composition, ", ".join(compositions)))
    result: Dict[str, Path] = {}
    for dataset in compositions[composition]:
        directory = base / dataset
        if not directory.is_dir():
            raise FileNotFoundError("required dataset directory is missing: {}".format(directory))
        for path in sorted(directory.glob("*.csv"), key=lambda item: item.name.casefold()):
            key = "{}:{}".format(dataset, path.stem)
            if key in result:
                raise HARTHError("duplicate namespaced subject {}".format(key))
            result[key] = path
    if not result:
        raise FileNotFoundError("no HARTH-family CSV files found under {}".format(base))
    return dict(sorted(result.items()))


def load_harth_subjects(
    root: str | Path,
    composition: str = "harth_walking_speed",
    *,
    rate_tolerance_hz: float = 0.25,
    subjects: Optional[Iterable[str]] = None,
) -> Dict[str, HarthSubject]:
    """Load namespaced HARTH-family subjects and preserve source provenance."""
    files = discover_harth_files(root, composition)
    if subjects is not None:
        requested = tuple(str(value) for value in subjects)
        missing = sorted(set(requested) - set(files))
        if missing:
            raise FileNotFoundError("requested HARTH subjects not found: {}".format(missing))
        files = {key: files[key] for key in requested}
    loaded = {}
    for key, path in files.items():
        dataset = key.split(":", 1)[0]
        loaded[key] = _read_csv(path, dataset, rate_tolerance_hz=rate_tolerance_hz)
    return loaded


def target_label_mapping() -> Dict[int, str]:
    """Return the stable encoded ten-class vocabulary."""
    return {index: name for index, name in enumerate(TARGET_CLASS_NAMES)}


__all__ = [
    "EXCLUDED_SOURCE_LABELS", "HARTH_COLUMNS", "HARTHError", "HarthSubject",
    "SOURCE_LABEL_MAPS", "TARGET_CLASS_NAMES", "TARGET_SAMPLE_RATE_HZ",
    "discover_harth_files", "load_harth_subjects", "target_label_mapping",
]
