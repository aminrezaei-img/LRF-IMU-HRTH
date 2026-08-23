"""Safety checks for paths, secrets, and generated artifacts in the release tree.

The historical-path compatibility exception is intentionally exact and narrow:
the copied audit and locked-reference files preserve provenance from the source
workspace.  It documents a compatibility exception, not a scientific
resolution, and must not be widened to cover active release content.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
from typing import Iterator

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MAX_PUBLIC_FILE_BYTES = 5 * 1024 * 1024
SYNTHETIC_FIXTURE_ROOT = "tests/fixtures/synthetic"
SYNTHETIC_LOG_ALLOWLIST = frozenset(
    {
        "tests/fixtures/synthetic/raw/subject01_ideal.log",
        "tests/fixtures/synthetic/raw/subject02_ideal.log",
        "tests/fixtures/synthetic/raw/subject03_ideal.log",
        "tests/fixtures/synthetic/raw/subject05_ideal.log",
    }
)
SYNTHETIC_FIXTURE_FILE_ALLOWLIST = frozenset(
    {
        "tests/fixtures/synthetic/README.md",
        "tests/fixtures/synthetic/SHA256SUMS",
        "tests/fixtures/synthetic/channel_selection.json",
        "tests/fixtures/synthetic/cnn_split_case.json",
        "tests/fixtures/synthetic/duplicate_audit_cases.json",
        "tests/fixtures/synthetic/evaluation_reference.json",
        "tests/fixtures/synthetic/fixture_manifest.json",
        "tests/fixtures/synthetic/flow_probe.json",
        "tests/fixtures/synthetic/loso_split_cases.json",
        "tests/fixtures/synthetic/metadata_summary.csv",
        "tests/fixtures/synthetic/preprocessing_cases.json",
        "tests/fixtures/synthetic/raw/compact_rows.json",
        "tests/fixtures/synthetic/raw/subject01_ideal.log",
        "tests/fixtures/synthetic/raw/subject02_ideal.log",
        "tests/fixtures/synthetic/raw/subject03_ideal.log",
        "tests/fixtures/synthetic/raw/subject05_ideal.log",
        "tests/fixtures/synthetic/standardization_cases.json",
        "tests/fixtures/synthetic/vae_probe.json",
        "tests/fixtures/synthetic/website_overlap_add.json",
        "tests/fixtures/synthetic/website_trajectory.json",
    }
)
SYNTHETIC_FIXTURE_DIRECTORY_ALLOWLIST = frozenset(
    {
        "tests/fixtures/synthetic",
        "tests/fixtures/synthetic/raw",
    }
)

# These are the only historical copies allowed to retain source-workstation
# markers.  Matching is by exact repository-relative path, never by directory
# or filename wildcard.
HISTORICAL_PATH_COMPATIBILITY_EXCEPTION = frozenset(
    {
        "configs/locked/RUNNING_INSTRUCTIONS.verbatim.md",
        "configs/locked/paper_release_reference.yaml",
        "docs/PAPER_RESULT_PROVENANCE.md",
        "docs/PUBLIC_RELEASE_RISKS.md",
        "docs/RELEASE_HANDOFF.md",
        "docs/RELEASE_INVENTORY.md",
        "docs/REPRODUCIBILITY_AUDIT.md",
    }
)


# Build each marker from pieces so this test does not contain the complete
# machine-specific strings it is designed to detect.
_DRIVE = "D" + ":"
_USER_DRIVE = "C" + ":"
_SOURCE_FOLDER = "PAPER" + "2"
_USER_FOLDER = "Users"
_ACCOUNT = "A" + "minR"
PATH_MARKER_PATTERNS = (
    (
        "source drive/data marker",
        re.compile(re.escape(_DRIVE + "/" + _SOURCE_FOLDER), re.IGNORECASE),
    ),
    (
        "Windows source drive/data marker",
        re.compile(re.escape(_DRIVE + "\\" + _SOURCE_FOLDER), re.IGNORECASE),
    ),
    (
        "user-root marker",
        re.compile(re.escape(_USER_DRIVE + "/" + _USER_FOLDER), re.IGNORECASE),
    ),
    (
        "Windows user-root marker",
        re.compile(re.escape(_USER_DRIVE + "\\" + _USER_FOLDER), re.IGNORECASE),
    ),
    (
        "local account marker",
        re.compile(
            rf"(?<![A-Za-z0-9]){re.escape(_ACCOUNT)}(?![A-Za-z0-9])",
            re.IGNORECASE,
        ),
    ),
)


FORBIDDEN_FILE_SUFFIXES = frozenset(
    {
        ".bin",
        ".ckpt",
        ".db",
        ".h5",
        ".hdf5",
        ".joblib",
        ".key",
        ".log",
        ".onnx",
        ".npz",
        ".npy",
        ".p12",
        ".pkl",
        ".pickle",
        ".pt",
        ".pth",
        ".pyc",
        ".pyo",
        ".safetensors",
        ".sqlite",
        ".sqlite3",
    }
)

FORBIDDEN_DIRECTORY_NAMES = frozenset(
    {
        ".agents",
        ".cache",
        ".claude",
        ".codex",
        ".ipynb_checkpoints",
        ".omx",
        ".pytest_cache",
        "__pycache__",
        "cache",
        "caches",
        "checkpoint",
        "checkpoints",
        "generated",
        "logs",
        "manuscript",
        "raw",
        "raw-data",
        "raw_data",
        "results",
        "review",
        "review-history",
        "review_history",
        "synthetic",
        "synthetic-data",
        "synthetic_data",
        "synthetic_weights",
    }
)

_SECRET_FILE_SUFFIXES = frozenset({".jks", ".pem", ".pfx", ".secret"})
_PRIVATE_KEY_BEGIN = "-----" + "BEGIN " + "PRIVATE KEY" + "-----"
_SECRET_KEYS = (
    "api" + "_key",
    "auth" + "_token",
    "client" + "_secret",
    "pass" + "word",
    "private" + "_key",
    "secret" + "_key",
    "access" + "_token",
)
_SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)\b(?:"
    + "|".join(re.escape(key) for key in _SECRET_KEYS)
    + r")\s*[:=]\s*[\"']?[^\s\"'#]{8,}"
)
_KNOWN_TOKEN_RES = (
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
)


def _relative_path(path: Path, repository_root: Path = REPOSITORY_ROOT) -> str:
    """Return a stable path representation for diagnostics and allowlisting."""

    return path.relative_to(repository_root).as_posix()


def _is_under(relative_path: str, root: str) -> bool:
    return relative_path == root or relative_path.startswith(root + "/")


def _is_allowed_synthetic_log(relative_path: str) -> bool:
    return relative_path in SYNTHETIC_LOG_ALLOWLIST


def _synthetic_fixture_tree_findings(
    repository_root: Path = REPOSITORY_ROOT,
) -> list[str]:
    """Reject every physical synthetic-fixture path outside the manifest tree."""

    fixture_root = repository_root / SYNTHETIC_FIXTURE_ROOT
    findings: list[str] = []
    if not fixture_root.is_dir():
        return [f"{SYNTHETIC_FIXTURE_ROOT}: missing synthetic fixture directory"]

    manifest_path = fixture_root / "fixture_manifest.json"
    if not manifest_path.is_file():
        findings.append(
            f"{_relative_path(manifest_path, repository_root)}: missing fixture manifest"
        )
    else:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            findings.append(
                f"{_relative_path(manifest_path, repository_root)}: invalid fixture manifest: {exc}"
            )
        else:
            expected_manifest_files = {
                path[len(SYNTHETIC_FIXTURE_ROOT) + 1 :]
                for path in SYNTHETIC_FIXTURE_FILE_ALLOWLIST
                if path != f"{SYNTHETIC_FIXTURE_ROOT}/SHA256SUMS"
            }
            declared_files = {
                Path(path).as_posix() for path in manifest.get("files", [])
            }
            if declared_files != expected_manifest_files:
                findings.append(
                    f"{_relative_path(manifest_path, repository_root)}: manifest file list does not match the exact public allowlist"
                )

    for current_root, directory_names, file_names in os.walk(fixture_root):
        current_path = Path(current_root)
        for name in directory_names:
            path = current_path / name
            relative_path = _relative_path(path, repository_root)
            if relative_path not in SYNTHETIC_FIXTURE_DIRECTORY_ALLOWLIST:
                findings.append(
                    f"{relative_path}: unmanifested synthetic fixture directory"
                )
        for name in file_names:
            path = current_path / name
            relative_path = _relative_path(path, repository_root)
            if relative_path not in SYNTHETIC_FIXTURE_FILE_ALLOWLIST:
                findings.append(f"{relative_path}: unmanifested synthetic fixture file")
    return findings


def _iter_tree_entries(repository_root: Path = REPOSITORY_ROOT) -> Iterator[Path]:
    """Yield physical release-tree entries, excluding only VCS internals."""

    for current_root, directory_names, file_names in os.walk(repository_root):
        directory_names[:] = [
            name for name in directory_names if name.casefold() != ".git"
        ]
        current_path = Path(current_root)
        yield from (current_path / name for name in directory_names)
        yield from (current_path / name for name in file_names)


def _iter_files(repository_root: Path = REPOSITORY_ROOT) -> Iterator[Path]:
    for path in _iter_tree_entries(repository_root):
        if path.is_file():
            yield path


def _read_text(path: Path) -> str | None:
    """Read UTF text while leaving binary artifacts to the artifact guard."""

    data = path.read_bytes()
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16")
    if b"\x00" in data:
        return None
    return data.decode("utf-8", errors="replace")


def _line_evidence(relative_path: str, line_number: int, line: str, detail: str) -> str:
    return f"{relative_path}:{line_number}: {detail}: {line.rstrip()!r}"


def _sensitive_filename_reason(path: Path) -> str | None:
    name = path.name.casefold()
    suffix = path.suffix.casefold()

    if name == ".env" or (name.startswith(".env.") and name != ".env.example"):
        return "environment file"
    if name in {
        "credentials",
        "credentials.json",
        "secret",
        "secrets",
        "secrets.json",
        "token.json",
    }:
        return "credential/secret filename"
    if name.startswith(("id_rsa", "id_ed25519")):
        return "private-key filename"
    if suffix in _SECRET_FILE_SUFFIXES:
        return "private-key/secret-file suffix"
    if name == "draft.ipynb":
        return "draft notebook"
    if (
        name.startswith("reviewer")
        or name.startswith("response_to_reviewers")
        or name.startswith("author_response")
    ):
        return "review-history filename"
    return None


def _secret_findings(
    path: Path, text: str, repository_root: Path = REPOSITORY_ROOT
) -> list[str]:
    relative_path = _relative_path(path, repository_root)
    findings: list[str] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if _PRIVATE_KEY_BEGIN.casefold() in line.casefold():
            findings.append(
                _line_evidence(
                    relative_path,
                    line_number,
                    line,
                    "private-key material",
                )
            )
        for pattern in (*_KNOWN_TOKEN_RES, _SECRET_ASSIGNMENT_RE):
            match = pattern.search(line)
            if match:
                findings.append(
                    _line_evidence(
                        relative_path,
                        line_number,
                        line,
                        f"secret-like value {match.group(0)!r}",
                    )
                )
    return findings


def test_no_absolute_or_personal_paths_in_public_text() -> None:
    findings: list[str] = []

    for path in _iter_files():
        relative_path = _relative_path(path)
        if relative_path in HISTORICAL_PATH_COMPATIBILITY_EXCEPTION:
            continue
        if path.suffix.casefold() in FORBIDDEN_FILE_SUFFIXES:
            continue
        text = _read_text(path)
        if text is None:
            continue
        for line_number, line in enumerate(text.splitlines(), start=1):
            for marker_name, marker_pattern in PATH_MARKER_PATTERNS:
                if marker_pattern.search(line):
                    findings.append(
                        _line_evidence(
                            relative_path,
                            line_number,
                            line,
                            f"forbidden machine-specific marker {marker_name}",
                        )
                    )

    if findings:
        pytest.fail(
            "Found machine-specific path or identity markers outside the exact "
            "historical compatibility exception:\n- " + "\n- ".join(findings)
        )


def test_local_account_marker_requires_token_boundaries() -> None:
    marker_patterns = dict(PATH_MARKER_PATTERNS)
    account_pattern = marker_patterns["local account marker"]
    local_path = _USER_DRIVE + "\\" + _USER_FOLDER + "\\" + _ACCOUNT + "\\project"
    public_identity = "https://huggingface.co/" + _ACCOUNT + "ezaei/LRF-IMU"

    assert account_pattern.search(local_path)
    assert not account_pattern.search(public_identity)


def _collect_safety_findings(repository_root: Path = REPOSITORY_ROOT) -> list[str]:
    findings = _synthetic_fixture_tree_findings(repository_root)

    for path in _iter_tree_entries(repository_root):
        relative_path = _relative_path(path, repository_root)
        if (
            path.is_dir()
            and path.name.casefold() in FORBIDDEN_DIRECTORY_NAMES
            and not _is_under(relative_path, SYNTHETIC_FIXTURE_ROOT)
        ):
            findings.append(f"{relative_path}: forbidden directory")
            continue
        if not path.is_file():
            continue

        suffix = path.suffix.casefold()
        synthetic_log = suffix == ".log" and _is_allowed_synthetic_log(relative_path)
        if suffix in FORBIDDEN_FILE_SUFFIXES and not synthetic_log:
            findings.append(
                f"{relative_path}: forbidden generated/binary file suffix {suffix!r}"
            )
        filename_reason = _sensitive_filename_reason(path)
        if filename_reason:
            findings.append(f"{relative_path}: forbidden {filename_reason}")

        if suffix in FORBIDDEN_FILE_SUFFIXES and not synthetic_log:
            continue
        text = _read_text(path)
        if text is not None:
            findings.extend(_secret_findings(path, text, repository_root))
    return findings


def test_no_secrets_or_forbidden_generated_artifacts() -> None:
    findings = _collect_safety_findings()
    if findings:
        pytest.fail(
            "Found secrets, sensitive files, or forbidden generated artifacts:\n- "
            + "\n- ".join(findings)
        )


def test_synthetic_log_allowlist_rejects_unlisted_paths() -> None:
    allowed = {
        "tests/fixtures/synthetic/raw/subject01_ideal.log",
        "tests/fixtures/synthetic/raw/subject02_ideal.log",
        "tests/fixtures/synthetic/raw/subject03_ideal.log",
        "tests/fixtures/synthetic/raw/subject05_ideal.log",
    }
    assert all(_is_allowed_synthetic_log(path) for path in allowed)
    assert not _is_allowed_synthetic_log(
        "tests/fixtures/synthetic/raw/subject99_ideal.log"
    )
    assert not _is_allowed_synthetic_log(
        "tests/fixtures/synthetic/raw/extra_forbidden.log"
    )


def test_synthetic_fixture_tree_rejects_ignored_unmanifested_probes() -> None:
    source_root = REPOSITORY_ROOT / SYNTHETIC_FIXTURE_ROOT
    sandbox_root = REPOSITORY_ROOT / ".scanner-negative-probe"
    assert not sandbox_root.exists()
    fixture_target = sandbox_root / SYNTHETIC_FIXTURE_ROOT
    shutil.copytree(source_root, fixture_target)
    (sandbox_root / ".gitignore").write_text(
        "**/checkpoints/\n*.pt\n*.log\n", encoding="utf-8"
    )
    probes = {
        "tests/fixtures/synthetic/raw/subject99_ideal.log": b"probe\n",
        "tests/fixtures/synthetic/raw/model.pt": b"checkpoint probe",
        "tests/fixtures/synthetic/raw/checkpoints/metadata.json": b"{}",
        "tests/fixtures/synthetic/raw/nested/extra.json": b"{}",
        "tests/fixtures/synthetic/unmanifested.json": b"{}",
    }
    try:
        for relative_path, payload in probes.items():
            probe = sandbox_root / relative_path
            probe.parent.mkdir(parents=True, exist_ok=True)
            probe.write_bytes(payload)
        findings = _collect_safety_findings(sandbox_root)
        assert findings
        for relative_path in probes:
            assert any(
                finding.startswith(relative_path + ":") for finding in findings
            ), relative_path
    finally:
        shutil.rmtree(sandbox_root, ignore_errors=True)
    assert not sandbox_root.exists()


def test_no_physical_release_file_exceeds_five_mibibytes() -> None:
    findings = [
        f"{_relative_path(path)}: {path.stat().st_size} bytes"
        for path in _iter_files()
        if path.stat().st_size > MAX_PUBLIC_FILE_BYTES
    ]
    if findings:
        pytest.fail(
            "Found public repository files larger than 5 MiB:\n- "
            + "\n- ".join(findings)
        )
