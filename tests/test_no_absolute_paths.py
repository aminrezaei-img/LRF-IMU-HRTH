"""Safety checks for paths, secrets, and generated artifacts in the release tree.

The historical-path compatibility exception is intentionally exact and narrow:
the copied audit and locked-reference files preserve provenance from the source
workspace.  It documents a compatibility exception, not a scientific
resolution, and must not be widened to cover active release content.
"""

from __future__ import annotations

import os
from pathlib import Path
import re
from typing import Iterator

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

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
PATH_MARKERS = (
    ("source drive/data marker", _DRIVE + "/" + _SOURCE_FOLDER),
    ("Windows source drive/data marker", _DRIVE + "\\" + _SOURCE_FOLDER),
    ("user-root marker", _USER_DRIVE + "/" + _USER_FOLDER),
    ("Windows user-root marker", _USER_DRIVE + "\\" + _USER_FOLDER),
    ("local account marker", _ACCOUNT),
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


def _relative_path(path: Path) -> str:
    """Return a stable path representation for diagnostics and allowlisting."""

    return path.relative_to(REPOSITORY_ROOT).as_posix()


def _iter_tree_entries() -> Iterator[Path]:
    """Yield release-tree files and directories, excluding VCS internals."""

    for current_root, directory_names, file_names in os.walk(REPOSITORY_ROOT):
        directory_names[:] = [
            name for name in directory_names if name.casefold() != ".git"
        ]
        current_path = Path(current_root)
        yield from (current_path / name for name in directory_names)
        yield from (current_path / name for name in file_names)


def _iter_files() -> Iterator[Path]:
    for path in _iter_tree_entries():
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


def _secret_findings(path: Path, text: str) -> list[str]:
    relative_path = _relative_path(path)
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
    folded_markers = tuple((name, marker.casefold()) for name, marker in PATH_MARKERS)

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
            folded_line = line.casefold()
            for marker_name, marker in folded_markers:
                if marker in folded_line:
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
            "historical compatibility exception:\n- "
            + "\n- ".join(findings)
        )


def test_no_secrets_or_forbidden_generated_artifacts() -> None:
    findings: list[str] = []

    for path in _iter_tree_entries():
        relative_path = _relative_path(path)
        if path.is_dir() and path.name.casefold() in FORBIDDEN_DIRECTORY_NAMES:
            findings.append(f"{relative_path}: forbidden directory")
            continue
        if not path.is_file():
            continue

        suffix = path.suffix.casefold()
        if suffix in FORBIDDEN_FILE_SUFFIXES:
            findings.append(
                f"{relative_path}: forbidden generated/binary file suffix {suffix!r}"
            )
        filename_reason = _sensitive_filename_reason(path)
        if filename_reason:
            findings.append(f"{relative_path}: forbidden {filename_reason}")

        if suffix in FORBIDDEN_FILE_SUFFIXES:
            continue
        text = _read_text(path)
        if text is not None:
            findings.extend(_secret_findings(path, text))

    if findings:
        pytest.fail(
            "Found secrets, sensitive files, or forbidden generated artifacts:\n- "
            + "\n- ".join(findings)
        )
