import copy
import json
import numpy as np
import pytest

from lrf_imu.integration.fusion import (
    FusionError,
    StitchConfig,
    audit_segments,
    generate_segment,
    stable_seed,
    stitch_windows,
    target_samples,
    validate_checkpoint_contract,
)
from lrf_imu.integration.physical_state import load_mapping_config, map_interval
from lrf_imu.integration.fusion_cli import _result_payload


def rec(seconds=1, eligible=True, cls=0):
    return {
        "persona_id": "p1",
        "date": "2026-01-01",
        "resolved_interval_id": "i1",
        "source_episode_id": "e1",
        "start_time": "2026-01-01T00:00:00+00:00",
        "end_time": f"2026-01-01T00:00:{seconds:02d}+00:00",
        "duration_seconds": seconds,
        "semantic_activity": "walk",
        "mobility_mode": "walk",
        "physical_state_class_id": cls,
        "physical_state_class_name": "walking_slow",
        "imu_eligible": eligible,
        "mapping_rule": "fixture",
        "mapping_provenance": {"x": 1},
    }


def fake(**kwargs):
    return np.full((3, 160), kwargs["window_index"] + 1, dtype=np.float32)


@pytest.mark.parametrize(
    "seconds,expected", [(1, 50), (3.2, 160), (10, 500), (333, 16650)]
)
def test_sample_count(seconds, expected):
    assert target_samples(seconds) == expected


def test_short_and_long_exact_and_no_tiling():
    short = generate_segment(rec(1), fake)
    assert short.signal.shape == (50, 3)
    long = generate_segment(rec(10), fake)
    assert long.signal.shape == (500, 3)
    assert len(np.unique(long.signal[:, 0])) > 1
    assert long.record["generated_windows"] > 1


def test_seed_is_stable_and_window_specific():
    assert stable_seed(42, "p", "d", "i", 0) == stable_seed(42, "p", "d", "i", 0)
    assert stable_seed(42, "p", "d", "i", 0, 0) != stable_seed(42, "p", "d", "i", 0, 1)
    assert np.array_equal(
        generate_segment(rec(10), fake).signal, generate_segment(rec(10), fake).signal
    )


def test_unsupported_never_calls_generator():
    called = []

    def bad(**kwargs):
        called.append(1)
        raise AssertionError

    result = generate_segment(rec(1, False), bad)
    assert result["status"] == "IMU_UNAVAILABLE" and not called


def test_failure_is_distinct():
    def fail(**kwargs):
        raise RuntimeError("boom")

    result = generate_segment(rec(1), fail)
    assert result["status"] == "IMU_GENERATION_FAILED"


def test_stitch_is_finite_exact_and_weights_bounded():
    out, boundaries = stitch_windows(
        [np.zeros((3, 160)), np.ones((3, 160))], 200, StitchConfig(40)
    )
    assert out.shape == (200, 3) and np.isfinite(out).all() and boundaries
    assert np.all((out[120:160] >= 0) & (out[120:160] <= 1))


def test_source_not_mutated_and_provenance():
    source = rec(1)
    before = copy.deepcopy(source)
    result = generate_segment(
        source, fake, vae_checkpoint="v.pt", flow_checkpoint="f.pt"
    )
    assert source == before
    assert result.provenance["mapping_provenance"] == {"x": 1}
    assert result.provenance["window_seeds"]


def test_checkpoint_contract_rejects_wrong_geometry():
    good = {
        "channels": 3,
        "input_length": 160,
        "latent_channels": 48,
        "latent_time_steps": 40,
    }
    flow = {"num_classes": 10, "latent_channels": 48, "latent_time_steps": 40}
    validate_checkpoint_contract(good, flow, {"mean": [0, 0, 0], "std": [1, 1, 1]})
    with pytest.raises(FusionError):
        validate_checkpoint_contract(
            {**good, "channels": 6}, flow, {"mean": [0], "std": [1]}
        )


def test_mapping_config_controls_cycling_class_and_whitelist(tmp_path):
    config_path = tmp_path / "mapping.yaml"
    config_path.write_text(
        "mapping_version: custom\n"
        "cycling:\n"
        "  generic_route_class: cycling_standing\n"
        "sitting_whitelist: [desk_work]\n",
        encoding="utf-8",
    )
    config = load_mapping_config(config_path)
    cycling = map_interval({"interval_type": "travel", "mobility_mode": "bike"}, config)
    sitting = map_interval({"semantic_activity": "desk_work"}, config)
    assert cycling["physical_state_class_id"] == 6
    assert sitting["physical_state_class_id"] == 7


def test_audit_distinguishes_unavailable_from_generation_failure():
    unavailable = generate_segment(rec(1, False), fake)

    def fail(**kwargs):
        raise RuntimeError("boom")

    failed = generate_segment(rec(1), fail)
    summary = audit_segments([unavailable, failed])
    assert summary["unsupported_intervals"] == 2
    assert summary["generation_failures"] == 1


def test_unavailable_result_has_serializable_payload():
    result = generate_segment(rec(1, False), fake)
    payload = _result_payload(result, "segment_000001")
    assert payload["record"]["status"] == "IMU_UNAVAILABLE"
    json.dumps(payload)
