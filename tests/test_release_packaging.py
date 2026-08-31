import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
DOC_NAMES = (
    "methodology.md",
    "architecture.md",
    "data_and_taxonomy.md",
    "training.md",
    "generation.md",
    "dayforge_mapping.md",
    "stitching_and_fusion.md",
    "reproducibility.md",
    "validation.md",
    "checkpoints.md",
)


def test_paper3_documentation_surface_exists():
    for name in DOC_NAMES:
        path = ROOT / "docs" / name
        assert path.is_file(), name
        assert re.search(r"^# ", path.read_text(encoding="utf-8"), re.MULTILINE)


def test_agent_guide_and_readme_describe_the_frozen_paper3_contract():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    llms = (ROOT / "llms.txt").read_text(encoding="utf-8")
    for text in (readme, llms):
        assert "HARTH" in text
        assert "DayForge" in text
        assert "physical_state_hint" in text
        assert "in_bed_or_lying_opportunity" in text
        assert "walking_slow" in text
        assert "cycling_standing" in text
        assert "10" in text
    assert "paper3_lrf_dayforge_handoff_v1" in llms


def test_citation_describes_this_repository():
    citation = yaml.safe_load((ROOT / "CITATION.cff").read_text(encoding="utf-8"))
    assert citation["type"] == "software"
    assert citation["repository-code"].endswith("aminrezaei-img/LRF-IMU-HRTH")
    assert citation["preferred-citation"]["doi"] == "10.1088/3049-477X/ae91ef"


def test_examples_are_small_and_machine_independent():
    examples = ROOT / "examples"
    assert examples.is_dir()
    files = [path for path in examples.rglob("*") if path.is_file()]
    assert files
    assert all(path.stat().st_size < 100_000 for path in files)
    for path in files:
        text = path.read_text(encoding="utf-8")
        assert not re.search(r"(?:[A-Za-z]:[\\/]|/home/|/Users/)", text)


@pytest.mark.parametrize(
    "wrapper",
    ["run_lrf_imu.sh", "run_lrf_imu.ps1", "run_paper3_dayforge.sh", "run_paper3_dayforge.ps1"],
)
def test_reproducible_wrappers_have_help_and_canonical_cli(wrapper):
    path = ROOT / "scripts" / wrapper
    assert path.is_file(), wrapper
    text = path.read_text(encoding="utf-8")
    assert "--help" in text or "-Help" in text
    assert "python -m lrf_imu" in text
    assert not re.search(r"(?:[A-Za-z]:[\\/]|/home/|/Users/)", text)


def test_powerShell_generation_wrapper_help_resolves():
    wrapper = ROOT / "scripts" / "run_lrf_imu.ps1"
    result = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(wrapper),
            "-Help",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "generate-harth" in result.stdout


def test_powerShell_dayforge_wrapper_help_resolves():
    wrapper = ROOT / "scripts" / "run_paper3_dayforge.ps1"
    result = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(wrapper),
            "-Help",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "map-dayforge-physical-states" in result.stdout


def test_powerShell_generation_wrapper_dry_run_resolves():
    wrapper = ROOT / "scripts" / "run_lrf_imu.ps1"
    result = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(wrapper),
            "-VaeCheckpoint",
            "pyproject.toml",
            "-FlowCheckpoint",
            "pyproject.toml",
            "-Class",
            "sitting",
            "-DryRun",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "dry_run=true" in result.stdout


def test_markdown_relative_links_resolve():
    markdown_files = [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]
    link_pattern = re.compile(r"\[[^]]+\]\(([^)]+)\)")
    for markdown in markdown_files:
        for target in link_pattern.findall(markdown.read_text(encoding="utf-8")):
            if target.startswith(("http://", "https://", "#", "mailto:")):
                continue
            target_path = target.split("#", 1)[0]
            assert (markdown.parent / target_path).resolve().is_file(), (
                markdown,
                target,
            )


def test_pyproject_exposes_editable_install_and_cli():
    import tomllib

    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert pyproject["project"]["name"] == "lrf-imu"
    assert pyproject["project"]["scripts"]["lrf-imu"] == "lrf_imu.cli:main"
    assert "torch>=2.0" in pyproject["project"]["optional-dependencies"]["training"]
    assert sys.version_info >= (3, 10)
