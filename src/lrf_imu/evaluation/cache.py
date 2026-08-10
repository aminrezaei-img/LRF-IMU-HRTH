"""External synthetic-cache identity and manifest helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Union

import numpy as np

PathValue = Union[str, Path]
_REQUIRED_IDENTITY_FIELDS = {
    "sensor_configuration",
    "held_out_subject",
    "vae_checkpoint_sha256",
    "flow_checkpoint_sha256",
    "config_identity",
    "seed",
    "steps",
    "samples_per_class",
    "implementation_version",
}


@dataclass(frozen=True)
class SyntheticCacheIdentity:
    sensor_configuration: str
    held_out_subject: int
    vae_checkpoint_sha256: str
    flow_checkpoint_sha256: str
    config_identity: str
    seed: int
    steps: int
    samples_per_class: int
    implementation_version: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def key(self) -> str:
        payload = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sha256_file(path: PathValue) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cache_paths(root: PathValue, identity: SyntheticCacheIdentity) -> Mapping[str, Path]:
    base = Path(root).expanduser().resolve() / identity.sensor_configuration
    base = base / "subject_{:02d}".format(identity.held_out_subject)
    return {
        "array": base / (identity.key + ".npz"),
        "manifest": base / (identity.key + ".manifest.json"),
    }


def discover_cache_manifest(array_path: PathValue) -> Path:
    """Find one unambiguous adjacent manifest without searching unrelated roots."""

    source = Path(array_path).expanduser().resolve()
    candidates = (
        source.with_suffix(".manifest.json"),
        source.parent / "manifest.json",
        source.parent / (source.name + ".manifest.json"),
    )
    existing = list(dict.fromkeys(path for path in candidates if path.is_file()))
    if not existing:
        raise FileNotFoundError(
            "synthetic cache requires an adjacent identity manifest: {}".format(source.name)
        )
    if len(existing) > 1:
        raise ValueError("multiple adjacent synthetic-cache manifests are ambiguous")
    return existing[0]


def validate_cache_manifest(
    array_path: PathValue,
    manifest_path: PathValue,
    *,
    sensor_configuration: str,
    held_out_subject: int,
    seed: int,
    steps: int,
    samples_per_class: int,
) -> dict[str, Any]:
    """Validate fold/sensor/protocol identity and the array checksum before loading."""

    array = Path(array_path).expanduser().resolve()
    manifest = Path(manifest_path).expanduser().resolve()
    with manifest.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or not isinstance(payload.get("identity"), dict):
        raise ValueError("synthetic-cache manifest must contain an identity object")
    identity = payload["identity"]
    missing = sorted(_REQUIRED_IDENTITY_FIELDS.difference(identity))
    if missing:
        raise ValueError("synthetic-cache identity is incomplete: {}".format(missing))
    expected = {
        "sensor_configuration": sensor_configuration,
        "held_out_subject": int(held_out_subject),
        "seed": int(seed),
        "steps": int(steps),
        "samples_per_class": int(samples_per_class),
    }
    mismatches = {
        key: {"expected": value, "observed": identity.get(key)}
        for key, value in expected.items()
        if identity.get(key) != value
    }
    if mismatches:
        raise ValueError(
            "synthetic-cache identity does not match requested fold: {}".format(mismatches)
        )
    for key in (
        "vae_checkpoint_sha256",
        "flow_checkpoint_sha256",
        "config_identity",
        "implementation_version",
    ):
        if not isinstance(identity.get(key), str) or not identity[key].strip():
            raise ValueError("synthetic-cache identity field is empty: {}".format(key))
    filename = payload.get("array_filename")
    if filename is not None and filename != array.name:
        raise ValueError("synthetic-cache manifest names a different array file")
    expected_sha = payload.get("array_sha256", payload.get("cache_sha256"))
    if not isinstance(expected_sha, str) or len(expected_sha) != 64:
        raise ValueError("synthetic-cache manifest requires a SHA-256 checksum")
    if sha256_file(array).lower() != expected_sha.lower():
        raise ValueError("synthetic-cache checksum does not match its manifest")
    return payload


def load_synthetic_cache(path: PathValue) -> tuple[np.ndarray, np.ndarray]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError("synthetic cache does not exist: {}".format(source))
    with np.load(source, allow_pickle=False) as payload:
        if "X_syn" not in payload.files or "y_syn" not in payload.files:
            raise ValueError("synthetic cache must contain X_syn and y_syn")
        windows = np.asarray(payload["X_syn"], dtype=np.float32)
        labels = np.asarray(payload["y_syn"], dtype=np.int64).reshape(-1)
    if windows.ndim != 3 or windows.shape[0] != labels.size:
        raise ValueError("synthetic cache arrays have incompatible shapes")
    return windows, labels


def load_validated_synthetic_cache(
    path: PathValue,
    *,
    sensor_configuration: str,
    held_out_subject: int,
    seed: int,
    steps: int,
    samples_per_class: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    manifest = discover_cache_manifest(path)
    payload = validate_cache_manifest(
        path,
        manifest,
        sensor_configuration=sensor_configuration,
        held_out_subject=held_out_subject,
        seed=seed,
        steps=steps,
        samples_per_class=samples_per_class,
    )
    windows, labels = load_synthetic_cache(path)
    return windows, labels, payload


def write_cache_manifest(
    path: PathValue,
    identity: SyntheticCacheIdentity,
    *,
    array_path: PathValue,
    extra: Mapping[str, Any] | None = None,
    overwrite: bool = False,
) -> Path:
    """Write metadata only after the caller supplies an explicit target path."""

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        raise FileExistsError("manifest exists and overwrite is false: {}".format(destination))
    payload = {
        "schema_version": "m3d.synthetic-cache-manifest.1",
        "cache_key": identity.key,
        "identity": identity.as_dict(),
        "array_filename": Path(array_path).name,
        "array_sha256": sha256_file(array_path),
    }
    if extra:
        payload["metadata"] = dict(extra)
    mode = "w" if overwrite else "x"
    with destination.open(mode, encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return destination


__all__ = [
    "SyntheticCacheIdentity",
    "cache_paths",
    "discover_cache_manifest",
    "load_synthetic_cache",
    "load_validated_synthetic_cache",
    "sha256_file",
    "validate_cache_manifest",
    "write_cache_manifest",
]