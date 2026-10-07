"""Tests for checksum-gated production artifact acquisition."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from lrf_imu.artifacts import (
    ArtifactError,
    acquire_artifacts,
    load_artifact_manifest,
    verify_artifact_directory,
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest().upper()


def _write_manifest(path: Path, payloads: dict[str, bytes]) -> Path:
    manifest = {
        "schema_version": "lrf-imu-harth.production-artifacts.1",
        "release_base_url": "https://example.invalid/release",
        "artifacts": [
            {
                "filename": filename,
                "bytes": len(payload),
                "sha256": _sha256(payload),
                "role": "checkpoint",
            }
            for filename, payload in payloads.items()
        ],
    }
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_local_acquisition_copies_only_verified_artifacts(tmp_path: Path) -> None:
    payloads = {"vae.pt": b"vae", "flow.pt": b"flow"}
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    for filename, payload in payloads.items():
        (source / filename).write_bytes(payload)
    manifest_path = _write_manifest(tmp_path / "manifest.json", payloads)

    result = acquire_artifacts(
        load_artifact_manifest(manifest_path),
        output,
        source_dir=source,
    )

    assert result["verified"] == 2
    assert result["downloaded_or_copied"] == 2
    assert verify_artifact_directory(
        load_artifact_manifest(manifest_path), output
    )["verified"] == 2


def test_acquisition_rejects_hash_mismatch_without_replacing_destination(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    output.mkdir()
    (source / "vae.pt").write_bytes(b"wrong")
    destination = output / "vae.pt"
    destination.write_bytes(b"keep")
    manifest_path = _write_manifest(tmp_path / "manifest.json", {"vae.pt": b"right"})

    with pytest.raises(ArtifactError, match="SHA-256 mismatch"):
        acquire_artifacts(
            load_artifact_manifest(manifest_path),
            output,
            source_dir=source,
            overwrite=True,
        )

    assert destination.read_bytes() == b"keep"
    assert not (output / "vae.pt.part").exists()


def test_verify_rejects_missing_required_artifact(tmp_path: Path) -> None:
    manifest_path = _write_manifest(tmp_path / "manifest.json", {"flow.pt": b"flow"})
    with pytest.raises(ArtifactError, match="missing artifact"):
        verify_artifact_directory(load_artifact_manifest(manifest_path), tmp_path)
