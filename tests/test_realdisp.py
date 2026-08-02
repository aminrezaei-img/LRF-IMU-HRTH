"""Focused synthetic-contract tests for REALDISP discovery and loading."""

from pathlib import Path
import shutil

import numpy as np
import pytest

from lrf_imu.data.realdisp import (
    EXPECTED_COLUMN_COUNT,
    RealDISPLogError,
    discover_subject_logs,
    extract_raw_activity_labels,
    extract_sensor_channels,
    extract_subject_id,
    load_realdisp_log,
    load_subject_data,
)


FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "synthetic" / "raw"


def _copy_fixture(source_name: str, destination: Path, *, destination_name: str = None) -> Path:
    target = destination / (destination_name or source_name)
    shutil.copyfile(FIXTURE_ROOT / source_name, target)
    return target


def test_subject_filename_parsing_is_anchored_and_case_insensitive():
    assert extract_subject_id("subject01_ideal.log") == 1
    assert extract_subject_id("SUBJECT001_IDEAL.LOG") == 1
    assert extract_subject_id(Path("nested") / "subject05_ideal.log") == 5

    with pytest.raises(ValueError, match="exact subject"):
        extract_subject_id("prefix_subject01_ideal.log")
    with pytest.raises(ValueError, match="exact subject"):
        extract_subject_id("subject01_ideal.log.bak")


def test_discovery_is_direct_child_only_and_deterministic(tmp_path):
    _copy_fixture("subject01_ideal.log", tmp_path)
    _copy_fixture("subject02_ideal.log", tmp_path)
    nested = tmp_path / "nested"
    nested.mkdir()
    _copy_fixture("subject03_ideal.log", nested)

    discovered = discover_subject_logs(tmp_path)

    assert list(discovered) == [1, 2]
    assert all(path.parent == tmp_path for path in discovered.values())


def test_duplicate_normalized_subject_ids_are_rejected(tmp_path):
    _copy_fixture("subject01_ideal.log", tmp_path)
    _copy_fixture("subject01_ideal.log", tmp_path, destination_name="subject001_ideal.log")

    with pytest.raises(ValueError, match="Duplicate normalized subject ID 1"):
        discover_subject_logs(tmp_path)


def test_explicit_allowlist_filters_and_requires_requested_subjects(tmp_path):
    _copy_fixture("subject01_ideal.log", tmp_path)
    _copy_fixture("subject02_ideal.log", tmp_path)
    _copy_fixture("subject05_ideal.log", tmp_path)

    assert list(discover_subject_logs(tmp_path, allowed_subjects=[5, 1])) == [1, 5]
    assert list(discover_subject_logs(tmp_path, allowlist="02")) == [2]

    with pytest.raises(FileNotFoundError, match="not found"):
        discover_subject_logs(tmp_path, subjects=[9])


def test_numeric_loader_enforces_exact_120_columns(tmp_path):
    source = FIXTURE_ROOT / "subject01_ideal.log"
    raw = load_realdisp_log(source)

    assert raw.shape == (24, EXPECTED_COLUMN_COUNT)
    assert raw.dtype == np.float64

    malformed = tmp_path / "subject01_ideal.log"
    malformed.write_text(
        "\n".join("\t".join(line.split("\t")[:-1]) for line in source.read_text().splitlines())
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(RealDISPLogError, match="Expected 120.*119"):
        load_realdisp_log(malformed)


def test_numeric_loader_reports_nonnumeric_cells(tmp_path):
    source = FIXTURE_ROOT / "subject01_ideal.log"
    malformed = tmp_path / "subject01_ideal.log"
    lines = source.read_text(encoding="utf-8").splitlines()
    first_row = lines[0].split("\t")
    first_row[0] = "not-a-number"
    lines[0] = "\t".join(first_row)
    malformed.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(RealDISPLogError, match="Non-numeric.*row 1.*column 1"):
        load_realdisp_log(malformed)


def test_six_and_reconstructed_three_channel_extraction_preserve_order():
    raw = load_realdisp_log(FIXTURE_ROOT / "subject01_ideal.log")

    six = extract_sensor_channels(raw, "six_channel")
    three = extract_sensor_channels(raw, "accelerometer_only")
    labels = extract_raw_activity_labels(raw)

    assert six.shape == (24, 6)
    assert three.shape == (24, 3)
    assert six.dtype == np.float32
    assert three.dtype == np.float32
    np.testing.assert_array_equal(six[0], np.arange(1000, 1006, dtype=np.float32))
    np.testing.assert_array_equal(three, six[:, :3])
    np.testing.assert_array_equal(labels[:6], np.array([1, 1, 99, 99, 1, 1], dtype=np.int32))


def test_subject_loader_returns_selected_channels_and_raw_labels():
    signals, labels = load_subject_data(FIXTURE_ROOT / "subject02_ideal.log")

    assert signals.shape == (24, 6)
    assert labels.shape == (24,)
    assert labels.dtype == np.int32
    np.testing.assert_array_equal(signals[0], np.arange(2000, 2006, dtype=np.float32))
