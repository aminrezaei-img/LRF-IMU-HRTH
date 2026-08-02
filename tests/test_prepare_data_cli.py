"""Milestone 3A orchestration and CLI safety tests."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
FIXTURE_ROOT = REPOSITORY_ROOT / "tests" / "fixtures" / "synthetic"
RAW_ROOT = FIXTURE_ROOT / "raw"
CONFIG_PATH = REPOSITORY_ROOT / "configs" / "paper" / "six_channel_160_40.yaml"
PYTHON = Path(sys.executable)

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from lrf_imu.data.pipeline import (  # noqa: E402
    METADATA_FILENAME,
    prepare_data,
    write_metadata_summary,
)


def _compact_kwargs():
    return {
        "data_root": RAW_ROOT,
        "config_path": CONFIG_PATH,
        "held_out_subject": 5,
        "window_length": 4,
        "hop_length": 2,
    }


def _subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = str(SOURCE_ROOT)
    return env


def _run_module(cwd: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(PYTHON), "-B", "-m", "lrf_imu", *arguments],
        cwd=cwd,
        env=_subprocess_env(),
        text=True,
        capture_output=True,
        check=False,
    )


def test_compact_pipeline_is_in_memory_and_metadata_only() -> None:
    prepared = prepare_data(**_compact_kwargs())

    assert prepared.train_windows.shape == (16, 6, 4)
    assert prepared.validation_windows.shape == (7, 6, 4)
    assert prepared.held_out_test_windows.shape == (8, 6, 4)
    assert prepared.train_windows.dtype == np.float32
    assert prepared.train_labels.dtype == np.int64
    assert prepared.summary["split"]["counts"] == {
        "train": 16,
        "validation": 7,
        "held_out_test": 8,
    }
    assert prepared.summary["audit"]["scope"] == "public_all_split_pairs"
    assert prepared.summary["audit"]["passed"] is True
    assert prepared.summary["normalization"]["training_only"] is True
    assert prepared.summary["safety"]["participant_derived_windows_persisted"] is False

    serialized = json.dumps(prepared.summary, sort_keys=True)
    assert "1000.0" not in serialized
    assert "1050.0" not in serialized
    assert prepared.summary["safety"]["raw_signals_persisted"] is False


def test_three_channel_pipeline_is_explicit_and_separate() -> None:
    prepared = prepare_data(
        **_compact_kwargs(),
        sensor_configuration="accelerometer_only",
    )

    assert prepared.train_windows.shape == (16, 3, 4)
    assert prepared.sensor_schema.channel_indices == (80, 81, 82)
    assert prepared.summary["sensor"]["reconstructed_three_channel"] is True
    assert prepared.summary["sensor"]["inference_time_channel_drop"] is False
    assert prepared.summary["sensor"]["training_mode"] == "separate_model"
    assert prepared.summary["sensor"]["three_channel_lineage"] == (
        "PUBLIC_RECONSTRUCTION_REQUIRED"
    )


def test_strict_contiguity_is_distinct_from_default_compatibility() -> None:
    compatibility = prepare_data(**_compact_kwargs())
    strict = prepare_data(
        **_compact_kwargs(),
        compatibility_mode="strict_original_contiguity",
    )

    assert compatibility.summary["preprocessing"]["compatibility_mode"] == (
        "filter_before_runs"
    )
    assert strict.summary["preprocessing"]["compatibility_mode"] == (
        "strict_original_contiguity"
    )
    assert strict.summary["sensor"]["channel_count"] == 6


def test_production_window_contract_keeps_empty_shape_without_fabricating_statistics() -> None:
    prepared = prepare_data(
        data_root=RAW_ROOT,
        config_path=CONFIG_PATH,
        held_out_subject=5,
    )

    assert prepared.train_windows.shape == (0, 6, 160)
    assert prepared.validation_windows.shape == (0, 6, 160)
    assert prepared.held_out_test_windows.shape == (0, 6, 160)
    assert prepared.normalizer is None
    assert prepared.summary["normalization"]["fit_status"] == (
        "not_fitted_empty_training_split"
    )


def test_metadata_writer_requires_explicit_output_and_protects_existing_artifact(
    tmp_path: Path,
) -> None:
    summary = prepare_data(**_compact_kwargs()).summary
    output_root = tmp_path / "metadata"

    artifact = write_metadata_summary(summary, output_root)
    assert artifact == output_root / METADATA_FILENAME
    original = artifact.read_bytes()
    persisted = json.loads(artifact.read_text(encoding="utf-8"))
    assert persisted["safety"]["participant_derived_windows_persisted"] is False
    assert "1000.0" not in artifact.read_text(encoding="utf-8")
    assert sorted(path.name for path in output_root.iterdir()) == [METADATA_FILENAME]

    with pytest.raises(FileExistsError):
        write_metadata_summary(summary, output_root)
    assert artifact.read_bytes() == original

    write_metadata_summary(summary, output_root, overwrite=True)
    assert artifact.exists()


def test_module_help_and_foreign_cwd_are_portable(tmp_path: Path) -> None:
    root_help = _run_module(tmp_path, "--help")
    assert root_help.returncode == 0
    assert "prepare-data" in root_help.stdout

    command_help = _run_module(tmp_path, "prepare-data", "--help")
    assert command_help.returncode == 0
    assert "--write-metadata" in command_help.stdout

    result = _run_module(
        tmp_path,
        "prepare-data",
        "--config",
        str(CONFIG_PATH),
        "--data-root",
        str(RAW_ROOT),
        "--held-out-subject",
        "5",
        "--window-length",
        "4",
        "--hop-length",
        "2",
        "--dry-run",
    )
    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert summary["execution"]["mode"] == "dry_run"
    assert summary["split"]["counts"] == {
        "train": 16,
        "validation": 7,
        "held_out_test": 8,
    }
    assert "C:\\Users" not in result.stdout
    assert "D:\\" not in result.stdout


def test_cli_dry_run_and_validate_only_never_write(tmp_path: Path) -> None:
    output_root = tmp_path / "should-not-exist"
    common = (
        "prepare-data",
        "--config",
        str(CONFIG_PATH),
        "--data-root",
        str(RAW_ROOT),
        "--held-out-subject",
        "5",
        "--window-length",
        "4",
        "--hop-length",
        "2",
        "--output-root",
        str(output_root),
        "--write-metadata",
    )
    dry_run = _run_module(tmp_path, *common, "--dry-run")
    validate_only = _run_module(tmp_path, *common, "--validate-only")
    assert dry_run.returncode == 0, dry_run.stderr
    assert validate_only.returncode == 0, validate_only.stderr
    assert json.loads(dry_run.stdout)["execution"]["metadata_written"] is False
    assert json.loads(validate_only.stdout)["execution"]["metadata_written"] is False
    assert not output_root.exists()


def test_cli_write_permission_is_required_and_overwrite_is_explicit(tmp_path: Path) -> None:
    output_root = tmp_path / "artifacts"
    base = (
        "prepare-data",
        "--config",
        str(CONFIG_PATH),
        "--data-root",
        str(RAW_ROOT),
        "--held-out-subject",
        "5",
        "--window-length",
        "4",
        "--hop-length",
        "2",
        "--output-root",
        str(output_root),
    )

    no_permission = _run_module(tmp_path, *base)
    assert no_permission.returncode == 0, no_permission.stderr
    assert not output_root.exists()

    missing_root_permission = _run_module(
        tmp_path,
        "prepare-data",
        "--data-root",
        str(RAW_ROOT),
        "--held-out-subject",
        "5",
        "--window-length",
        "4",
        "--hop-length",
        "2",
        "--write-metadata",
    )
    assert missing_root_permission.returncode == 2
    assert "--output-root" in missing_root_permission.stderr

    first_write = _run_module(tmp_path, *base, "--write-metadata")
    assert first_write.returncode == 0, first_write.stderr
    artifact = output_root / METADATA_FILENAME
    original = artifact.read_bytes()

    protected = _run_module(tmp_path, *base, "--write-metadata")
    assert protected.returncode == 2
    assert "overwrite" in protected.stderr.lower()
    assert artifact.read_bytes() == original

    overwrite = _run_module(tmp_path, *base, "--write-metadata", "--overwrite")
    assert overwrite.returncode == 0, overwrite.stderr
    assert sorted(path.name for path in output_root.iterdir()) == [METADATA_FILENAME]


def test_cli_rejects_exact_paper_reproduction_config_without_writing(tmp_path: Path) -> None:
    bad_config = tmp_path / "bad.yaml"
    bad_config.write_text(
        CONFIG_PATH.read_text(encoding="utf-8").replace(
            "exact_paper_reproduction: false",
            "exact_paper_reproduction: true",
            1,
        ),
        encoding="utf-8",
    )
    result = _run_module(
        tmp_path,
        "prepare-data",
        "--config",
        str(bad_config),
        "--data-root",
        str(RAW_ROOT),
        "--dry-run",
    )
    assert result.returncode == 2
    assert "exact paper reproduction" in result.stderr.lower()
