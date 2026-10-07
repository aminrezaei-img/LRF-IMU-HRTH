"""Checksum-gated acquisition of the frozen HARTH production artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import shutil
from typing import Any, Iterable, Mapping
from urllib.parse import quote
from urllib.request import urlopen


SCHEMA_VERSION = "lrf-imu-harth.production-artifacts.1"
DEFAULT_MANIFEST_PATH = (
    Path(__file__).resolve().parent
    / "resources"
    / "manifests"
    / "production_harth_v1.json"
)
_SHA256_PATTERN = re.compile(r"^[0-9A-F]{64}$")


class ArtifactError(ValueError):
    """Raised when a production artifact violates the frozen manifest."""


def sha256_file(path: str | Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Return an uppercase SHA-256 digest without loading the file into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest().upper()


def load_artifact_manifest(path: str | Path | None = None) -> dict[str, Any]:
    """Load and validate the production artifact manifest."""

    manifest_path = DEFAULT_MANIFEST_PATH if path is None else Path(path)
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ArtifactError(f"could not read artifact manifest: {manifest_path}") from exc
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ArtifactError("unsupported production artifact manifest schema")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ArtifactError("artifact manifest must contain a non-empty artifacts list")

    names: set[str] = set()
    for index, artifact in enumerate(artifacts):
        if not isinstance(artifact, dict):
            raise ArtifactError(f"artifact entry {index} is not an object")
        filename = artifact.get("filename")
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise ArtifactError(f"artifact entry {index} has an unsafe filename")
        if filename in names:
            raise ArtifactError(f"duplicate artifact filename: {filename}")
        names.add(filename)
        digest = artifact.get("sha256")
        if not isinstance(digest, str) or not _SHA256_PATTERN.fullmatch(
            digest.upper()
        ):
            raise ArtifactError(f"artifact {filename} has an invalid SHA-256")
        artifact["sha256"] = digest.upper()
        byte_count = artifact.get("bytes")
        if not isinstance(byte_count, int) or byte_count < 0:
            raise ArtifactError(f"artifact {filename} has an invalid byte count")
        if not isinstance(artifact.get("role"), str):
            raise ArtifactError(f"artifact {filename} has no role")
    payload["manifest_path"] = str(manifest_path.resolve())
    return payload


def _selected_artifacts(
    manifest: Mapping[str, Any], roles: Iterable[str] | None
) -> list[Mapping[str, Any]]:
    selected_roles = None if roles is None else frozenset(roles)
    artifacts = manifest["artifacts"]
    return [
        artifact
        for artifact in artifacts
        if selected_roles is None or artifact["role"] in selected_roles
    ]


def _verify_file(path: Path, artifact: Mapping[str, Any]) -> None:
    if not path.is_file():
        raise ArtifactError(f"missing artifact: {artifact['filename']}")
    actual_size = path.stat().st_size
    if actual_size != artifact["bytes"]:
        raise ArtifactError(
            f"byte-count mismatch for {artifact['filename']}: "
            f"expected {artifact['bytes']}, got {actual_size}"
        )
    actual_digest = sha256_file(path)
    if actual_digest != artifact["sha256"]:
        raise ArtifactError(
            f"SHA-256 mismatch for {artifact['filename']}: "
            f"expected {artifact['sha256']}, got {actual_digest}"
        )


def verify_artifact_directory(
    manifest: Mapping[str, Any],
    artifact_dir: str | Path,
    *,
    roles: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Require every selected artifact to match its frozen size and digest."""

    root = Path(artifact_dir)
    selected = _selected_artifacts(manifest, roles)
    for artifact in selected:
        _verify_file(root / artifact["filename"], artifact)
    return {
        "artifact_dir": str(root.resolve()),
        "verified": len(selected),
        "roles": sorted({artifact["role"] for artifact in selected}),
        "status": "PASS",
    }


def acquire_artifacts(
    manifest: Mapping[str, Any],
    output_dir: str | Path,
    *,
    source_dir: str | Path | None = None,
    base_url: str | None = None,
    roles: Iterable[str] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Copy or download selected artifacts, accepting only exact hashes."""

    if source_dir is not None and base_url is not None:
        raise ArtifactError("source_dir and base_url are mutually exclusive")
    if source_dir is None:
        base_url = base_url or manifest.get("release_base_url")
        if not isinstance(base_url, str) or not base_url:
            raise ArtifactError("no release base URL is available")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    source = None if source_dir is None else Path(source_dir)
    selected = _selected_artifacts(manifest, roles)
    changed = 0
    reused = 0

    for artifact in selected:
        filename = artifact["filename"]
        destination = output / filename
        if destination.exists():
            try:
                _verify_file(destination, artifact)
            except ArtifactError:
                if not overwrite:
                    raise
            else:
                reused += 1
                continue

        partial = destination.with_name(destination.name + ".part")
        try:
            if source is not None:
                source_path = source / filename
                if not source_path.is_file():
                    raise ArtifactError(f"missing source artifact: {filename}")
                with source_path.open("rb") as input_handle, partial.open(
                    "wb"
                ) as output_handle:
                    shutil.copyfileobj(input_handle, output_handle)
            else:
                url = base_url.rstrip("/") + "/" + quote(filename)
                with urlopen(url) as response, partial.open("wb") as output_handle:
                    shutil.copyfileobj(response, output_handle)
            _verify_file(partial, artifact)
            partial.replace(destination)
            changed += 1
        except Exception:
            partial.unlink(missing_ok=True)
            raise

    verified = verify_artifact_directory(manifest, output, roles=roles)
    return {
        **verified,
        "downloaded_or_copied": changed,
        "reused": reused,
    }


__all__ = [
    "ArtifactError",
    "DEFAULT_MANIFEST_PATH",
    "SCHEMA_VERSION",
    "acquire_artifacts",
    "load_artifact_manifest",
    "sha256_file",
    "verify_artifact_directory",
]
