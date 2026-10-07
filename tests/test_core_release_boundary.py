"""Regression gates for the model-only LRF-IMU-HARTH release boundary."""

from __future__ import annotations

import json
from pathlib import Path

from lrf_imu.cli import build_parser


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = (
    ROOT
    / "src"
    / "lrf_imu"
    / "resources"
    / "manifests"
    / "production_harth_v1.json"
)

FORBIDDEN_APPLICATION_PATHS = (
    "configs/paper/dayforge_harth_mapping.yaml",
    "docs/PAPER3_HARTH_DAYFORGE_RUNBOOK.md",
    "docs/dayforge_mapping.md",
    "docs/stitching_and_fusion.md",
    "examples/example_interval_manifest.json",
    "examples/example_mapping_input.json",
    "examples/example_physical_state_mapping.csv",
    "scripts/run_paper3_dayforge.ps1",
    "scripts/run_paper3_dayforge.sh",
    "src/lrf_imu/integration",
    "tests/test_dayforge_fusion.py",
    "tests/test_dayforge_handoff.py",
)

FORBIDDEN_APPLICATION_TOKENS = (
    "dayforge",
    "physical_state_hint",
    "in_bed_or_lying_opportunity",
    "map-dayforge-physical-states",
    "synthesize-dayforge",
)


def test_application_layer_is_absent_from_core_release() -> None:
    present = [path for path in FORBIDDEN_APPLICATION_PATHS if (ROOT / path).exists()]
    assert present == []


def test_core_cli_exposes_model_pipeline_only() -> None:
    help_text = build_parser().format_help().casefold()
    for command in (
        "prepare-harth-data",
        "train-harth-vae",
        "train-harth-flow",
        "generate-harth",
    ):
        assert command in help_text
    for token in FORBIDDEN_APPLICATION_TOKENS:
        assert token not in help_text


def test_active_core_surfaces_do_not_reference_application_contracts() -> None:
    roots = (
        ROOT / "src",
        ROOT / "configs",
        ROOT / "scripts",
        ROOT / "examples",
    )
    findings: list[str] = []
    for root in roots:
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.casefold() in {".png", ".jpg"}:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore").casefold()
            for token in FORBIDDEN_APPLICATION_TOKENS:
                if token in text:
                    findings.append(f"{path.relative_to(ROOT).as_posix()}: {token}")
    assert findings == []


def test_production_manifest_names_exactly_one_global_checkpoint_pair() -> None:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "lrf-imu-harth.production-artifacts.1"
    assert payload["training_code_commit"] == (
        "81b123eab079f5cde7400fee5620e3bffb85a673"
    )
    assert payload["training_data_commit"] == (
        "dad2cfbe89a26f72f19770419469ac037de200df"
    )

    artifacts = payload["artifacts"]
    checkpoints = [item for item in artifacts if item["role"] == "checkpoint"]
    assert [item["filename"] for item in checkpoints] == [
        "vae_s3_z48.pt",
        "flow_unet_best.pt",
    ]
    assert [item["sha256"] for item in checkpoints] == [
        "6B118182E14FF04CBD57D66A76986BF3568561F0FD42D02257F8036A0138AAD9",
        "F1058EB1FB94809B74B8DFFC24E2697F8C73B046CF2A8E7D24FFF60EC6D63164",
    ]
    assert payload["checkpoint_scope"] == "one_global_pair"
    assert payload["persona_specific_checkpoints"] is False
    assert payload["cohort_outputs_included"] is False
    assert [item["role"] for item in artifacts].count("training_data") == 1
    assert len(artifacts) == 3
    normalization = ROOT / payload["packaged_resources"]["normalization"]
    assert normalization.is_file()
    normalization_payload = json.loads(normalization.read_text(encoding="utf-8"))
    assert normalization_payload["mean"] == [
        -0.5969877198736515,
        -0.10592963376410314,
        -0.5342781783351523,
    ]
    assert normalization_payload["std"] == [
        0.6554309322235737,
        0.39473405281278284,
        0.6472765870108628,
    ]
    assert normalization_payload["numeric_values_changed"] is False
