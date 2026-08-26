"""Synthetic regression tests for the HARTH-family replacement path."""
from __future__ import annotations

import csv
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from lrf_imu.data.harth import TARGET_CLASS_NAMES, discover_harth_files, load_harth_subjects
from lrf_imu.data.harth_pipeline import prepare_harth_data


def _write(path: Path, labels: list[int], *, with_index: bool = False, offset: int = 0) -> None:
    header = ["timestamp", "back_x", "back_y", "back_z", "thigh_x", "thigh_y", "thigh_z", "label"]
    if with_index:
        header = ["timestamp", "index", *header[1:]]
    start = datetime(2024, 1, 1)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for index, label in enumerate(labels):
            row = [(start + timedelta(seconds=index / 50)).isoformat(timespec="milliseconds")]
            if with_index:
                row.append(index)
            row.extend([0, 0, 0, index + offset, index + offset + 1, index + offset + 2, label])
            writer.writerow(row)


def _fixture(root: Path) -> None:
    (root / "harth").mkdir()
    (root / "adult_walking_speed").mkdir()
    # A long contiguous block per class makes complete 160/40 windows.
    _write(root / "harth" / "S006.csv", sum(([2] * 200, [4] * 200, [5] * 200, [13] * 200, [14] * 200, [7] * 200, [6] * 200, [8] * 200), []), with_index=True, offset=10000)
    _write(root / "harth" / "S008.csv", [2] * 200 + [4] * 200 + [13] * 200 + [7] * 200 + [6] * 200 + [8] * 200, offset=20000)
    _write(root / "adult_walking_speed" / "01.csv", [101] * 200 + [102] * 200 + [103] * 200 + [2] * 200, offset=30000)


def test_harth_discovery_mapping_and_optional_index(tmp_path: Path) -> None:
    _fixture(tmp_path)
    files = discover_harth_files(tmp_path)
    assert set(files) == {"harth:S006", "harth:S008", "adult_walking_speed:01"}
    loaded = load_harth_subjects(tmp_path)
    assert set(loaded) == set(files)
    assert loaded["harth:S006"].signals.shape[1] == 3
    assert set(np.unique(loaded["adult_walking_speed:01"].encoded_labels)) == set(range(4))


def test_harth_pipeline_has_ten_class_contract_and_isolation(tmp_path: Path) -> None:
    _fixture(tmp_path)
    prepared = prepare_harth_data(tmp_path, held_out_subject="harth:S006")
    assert prepared.train_windows.shape[1:] == (3, 160)
    assert prepared.train_windows.dtype == np.float32
    assert set(np.unique(np.concatenate([prepared.train_labels, prepared.validation_labels, prepared.held_out_test_labels]))) == set(range(10))
    assert set(prepared.train_subjects).isdisjoint(prepared.validation_subjects)
    assert prepared.held_out_subject not in prepared.train_subjects + prepared.validation_subjects
    assert prepared.normalizer.mean.shape == (1, 3, 1)
    assert prepared.summary["audit"]["passed"] is True
    assert prepared.summary["target_class_names"] == list(TARGET_CLASS_NAMES)
