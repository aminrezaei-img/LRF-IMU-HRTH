"""Regression tests for dependency and wheel-portable configuration contracts."""

from __future__ import annotations

import re
from pathlib import Path
import sys

import pytest
import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from lrf_imu import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    PACKAGE_CONFIG_DIR,
    ConfigError,
    ExperimentConfig,
    load_config,
)
from lrf_imu.data.pipeline import prepare_data  # noqa: E402


PAPER_CONFIG_NAMES = (
    "six_channel_160_40.yaml",
    "accelerometer_only_160_40.yaml",
    "sensitivity_grid.yaml",
)
COMPACT_CONFIG = REPOSITORY_ROOT / "configs" / "paper" / PAPER_CONFIG_NAMES[0]
RAW_ROOT = REPOSITORY_ROOT / "tests" / "fixtures" / "synthetic" / "raw"


def _normalised_bytes(path: Path) -> bytes:
    return path.read_bytes().replace(b"\r\n", b"\n")


def _declared_project_dependencies() -> set[str]:
    text = (REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(
        r"^dependencies\s*=\s*\[(?P<body>.*?)\]",
        text,
        flags=re.MULTILINE | re.DOTALL,
    )
    assert match is not None
    return set(re.findall(r'"([^"]+)"', match.group("body")))


def test_runtime_dependencies_and_package_data_are_declared_consistently() -> None:
    project_dependencies = _declared_project_dependencies()
    requirement_lines = {
        line.strip()
        for line in (REPOSITORY_ROOT / "requirements.txt").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }

    assert project_dependencies == requirement_lines
    assert project_dependencies == {"PyYAML>=6.0", "numpy>=1.21.3"}

    pyproject = (REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'lrf_imu = ["resources/configs/paper/*.yaml"]' in pyproject


def test_packaged_paper_configs_are_intentional_synchronized_resources() -> None:
    assert DEFAULT_CONFIG_PATH.resolve() == (
        PACKAGE_CONFIG_DIR / "six_channel_160_40.yaml"
    ).resolve()

    for name in PAPER_CONFIG_NAMES:
        root_path = REPOSITORY_ROOT / "configs" / "paper" / name
        packaged_path = PACKAGE_CONFIG_DIR / name
        assert packaged_path.is_file()
        assert _normalised_bytes(packaged_path) == _normalised_bytes(root_path)
        assert isinstance(
            yaml.safe_load(packaged_path.read_text(encoding="utf-8")), dict
        )


def test_default_config_load_is_independent_of_current_working_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    config = load_config()
    assert config.config_path is not None
    assert config.config_path.resolve() == DEFAULT_CONFIG_PATH.resolve()


def test_paper_configs_preserve_explicit_vae_and_classifier_fractions() -> None:
    for name in PAPER_CONFIG_NAMES[:2]:
        config = load_config(
            REPOSITORY_ROOT / "configs" / "paper" / name,
            base_dir=REPOSITORY_ROOT,
        )
        assert config.split.vae_subject_validation_fraction == 0.15
        assert config.split.classifier_window_validation_fraction == 0.20
        assert config.split.validation_fraction == 0.20
        serialized_split = config.to_mapping()["split"]
        assert serialized_split["vae_subject_validation_fraction"] == 0.15
        assert serialized_split["classifier_window_validation_fraction"] == 0.20


def test_pipeline_propagates_configured_vae_fraction_and_preserves_classifier_fraction() -> None:
    prepared = prepare_data(
        data_root=RAW_ROOT,
        config_path=COMPACT_CONFIG,
        held_out_subject=5,
        window_length=4,
        hop_length=2,
    )

    assert prepared.split.metadata.validation_fraction == 0.15
    assert prepared.summary["split"]["validation_fraction"] == 0.15
    assert prepared.summary["split"]["vae_subject_validation_fraction"] == 0.15
    assert prepared.summary["split"]["classifier_window_validation_fraction"] == 0.20
    assert prepared.summary["split"]["cnn_validation_fraction"] == 0.20


def test_legacy_classifier_fraction_must_match_explicit_classifier_fraction() -> None:
    config = load_config(COMPACT_CONFIG, base_dir=REPOSITORY_ROOT)
    mapping = config.to_mapping()
    mapping["split"]["validation_fraction"] = 0.20
    mapping["split"]["classifier_window_validation_fraction"] = 0.25

    with pytest.raises(ConfigError, match="legacy classifier/window alias"):
        ExperimentConfig.from_mapping(mapping)
def test_pipeline_reads_non_default_vae_fraction_from_explicit_config(
    tmp_path: Path,
) -> None:
    source = COMPACT_CONFIG.read_text(encoding="utf-8")
    old_value = "vae_subject_validation_fraction: 0.15"
    assert source.count(old_value) == 1
    custom_config = tmp_path / "sentinel_fraction.yaml"
    custom_config.write_text(
        source.replace(old_value, "vae_subject_validation_fraction: 0.10", 1),
        encoding="utf-8",
    )

    prepared = prepare_data(
        data_root=RAW_ROOT,
        config_path=custom_config,
        held_out_subject=5,
        window_length=4,
        hop_length=2,
    )

    assert prepared.config.split.vae_subject_validation_fraction == 0.10
    assert prepared.split.metadata.validation_fraction == 0.10
    assert prepared.summary["split"]["vae_subject_validation_fraction"] == 0.10
    assert prepared.summary["split"]["classifier_window_validation_fraction"] == 0.20

def test_release_metadata_and_extras_are_explicit() -> None:
    pyproject = (REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert 'requires-python = ">=3.10"' in pyproject
    assert 'readme = "README.md"' in pyproject
    assert 'requires = ["setuptools>=69", "wheel"]' in pyproject
    assert 'license =' not in pyproject
    for author in ("Amin Rezaei", "Morten Kjærgaard", "Jasper Schipperijn"):
        assert f'{{name = "{author}"}}' in pyproject
    for extra in (
        'test = ["pytest>=7"]',
        'training = ["torch>=2.0"]',
        'evaluation = ["scikit-learn>=1.2"]',
        'analysis = ["scipy>=1.10"]',
        'dev = ["build>=1.0", "mypy>=1.8", "pytest>=7", "ruff>=0.6"]',
    ):
        assert extra in pyproject


def test_citation_preserves_exact_scientific_identity() -> None:
    citation = yaml.safe_load(
        (REPOSITORY_ROOT / "CITATION.cff").read_text(encoding="utf-8")
    )

    assert citation["title"] == (
        "A latent rectified flow approach to generate synthetic wearable data "
        "– a LABDA solution"
    )
    assert citation["journal"] == "Machine Learning: Health"
    assert citation["doi"] == "10.1088/3049-477X/ae91ef"
    assert [
        f'{author["given-names"]} {author["family-names"]}'
        for author in citation["authors"]
    ] == ["Amin Rezaei", "Morten Kjærgaard", "Jasper Schipperijn"]
