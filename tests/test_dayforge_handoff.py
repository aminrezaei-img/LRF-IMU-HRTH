import json

import pytest

from lrf_imu.integration.dayforge import load_resolved_intervals
from lrf_imu.integration.physical_state import MappingConfig, map_interval


def record(
    *,
    activity="work",
    mode=None,
    interval_type="DWELL",
    realization="REALIZED_EXACT",
    hint=None,
    distance=None,
    route_duration=None,
):
    value = {
        "persona_id": "p1",
        "date": "2026-01-01",
        "interval_id": "p1:2026-01-01:0:dwell",
        "source_episode_index": 0,
        "interval_type": interval_type,
        "start_time": "2026-01-01T00:00:00",
        "end_time": "2026-01-01T01:00:00",
        "duration_seconds": 3600,
        "activity_type": activity,
        "mode": mode,
        "realization_status": realization,
        "route_distance_m": distance,
        "route_duration_s": route_duration,
    }
    if hint is not None:
        value["physical_state_hint"] = hint
    return value


def with_derived(value, hint=None):
    result = dict(value)
    result["in_bed_or_lying_opportunity"] = True
    result["in_bed_or_lying_opportunity_evidence"] = [
        {
            "interval_id": "p1:2026-01-01:0:in_bed_opportunity:0",
            "source": "in_bed_opportunity.json",
        }
    ]
    if hint is not None:
        result["physical_state_hint"] = hint
    return result


@pytest.mark.parametrize(
    ("hint", "expected_id", "expected_name"),
    [
        ("sitting", 7, "sitting"),
        ("standing", 8, "standing"),
        ("lying", 9, "lying"),
        ("running", 3, "running"),
        ("stairs", 4, "stair_climbing"),
    ],
)
def test_physical_state_hint_maps_explicit_states(hint, expected_id, expected_name):
    mapped = map_interval(record(hint=hint))
    assert mapped["physical_state_class_id"] == expected_id
    assert mapped["physical_state_class_name"] == expected_name
    assert mapped["mapping_source"] == "physical_state_hint"


@pytest.mark.parametrize("hint", ["walking", "mixed", "unknown"])
def test_non_specific_hints_do_not_create_a_class(hint):
    mapped = map_interval(record(hint=hint))
    assert mapped["physical_state_class_id"] is None
    assert mapped["imu_eligible"] is False


def test_walking_hint_without_route_speed_is_unavailable():
    mapped = map_interval(record(activity="walk", hint="walking"))
    assert mapped["physical_state_class_id"] is None
    assert mapped["imu_unavailable_reason"] == "WALK_SPEED_UNAVAILABLE"


def test_cycling_hint_never_infers_cycling_standing():
    mapped = map_interval(record(hint="cycling"))
    assert mapped["physical_state_class_id"] != 6
    assert mapped["imu_eligible"] is False


def test_realized_walking_wins_over_contradictory_hint():
    mapped = map_interval(
        record(
            activity="walk",
            mode="walk",
            interval_type="TRAVEL",
            hint="sitting",
            distance=1000,
            route_duration=1000,
        )
    )
    assert mapped["physical_state_class_id"] == 0
    assert mapped["mapping_source"] == "realized_mobility"
    assert mapped["mapping_conflict"] is True


def test_passive_vehicle_is_not_sitting_even_with_sitting_hint():
    mapped = map_interval(
        record(activity="commute", mode="car", interval_type="TRAVEL", hint="sitting")
    )
    assert mapped["physical_state_class_id"] is None
    assert mapped["mapping_status"] == "UNSUPPORTED_PASSIVE_TRANSPORT"


def test_failed_mobility_does_not_fabricate_walking_from_hint():
    mapped = map_interval(
        record(
            activity="walk",
            mode="walk",
            interval_type="TRAVEL",
            realization="INFEASIBLE_ROUTE",
            hint="walking",
        )
    )
    assert mapped["physical_state_class_id"] is None
    assert mapped["mapping_status"] == "NONREALIZED_MOBILITY"


@pytest.mark.parametrize("hint", [None, "unknown", "mixed"])
def test_derived_in_bed_opportunity_maps_to_lying_without_conflicting_hint(hint):
    mapped = map_interval(with_derived(record(activity="idle_at_home"), hint))
    assert mapped["physical_state_class_id"] == 9
    assert mapped["mapping_source"] == "derived_in_bed_opportunity"
    assert mapped["mapping_conflict"] is False


def test_derived_in_bed_with_lying_hint_maps_to_lying():
    mapped = map_interval(with_derived(record(activity="idle_at_home"), "lying"))
    assert mapped["physical_state_class_id"] == 9
    assert mapped["mapping_source"] == "physical_state_hint"


@pytest.mark.parametrize(
    ("value", "expected_id", "source"),
    [
        (
            record(
                activity="walk",
                mode="walk",
                interval_type="TRAVEL",
                distance=1000,
                route_duration=1000,
            ),
            0,
            "realized_mobility",
        ),
        (record(activity="running"), 3, "existing_explicit_semantic_evidence"),
        (
            record(activity="commute", mode="bike", interval_type="TRAVEL"),
            5,
            "realized_mobility",
        ),
    ],
)
def test_derived_in_bed_does_not_override_stronger_movement(value, expected_id, source):
    mapped = map_interval(with_derived(value))
    assert mapped["physical_state_class_id"] == expected_id
    assert mapped["mapping_source"] == source
    assert mapped["mapping_conflict"] is True


@pytest.mark.parametrize(("hint", "expected_id"), [("sitting", 7), ("standing", 8)])
def test_derived_in_bed_conflict_preserves_explicit_posture_hint(hint, expected_id):
    mapped = map_interval(with_derived(record(activity="work"), hint))
    assert mapped["physical_state_class_id"] == expected_id
    assert mapped["mapping_source"] == "physical_state_hint"
    assert mapped["mapping_conflict"] is True
    assert mapped["mapping_provenance"]["physical_state_hint"] == hint


def test_mapping_source_provenance_is_complete_for_derived_evidence():
    mapped = map_interval(with_derived(record(activity="idle_at_home")))
    provenance = mapped["mapping_provenance"]
    assert provenance["mapping_source"] == "derived_in_bed_opportunity"
    assert provenance["final_harth_class"] == "lying"
    assert provenance["derived_evidence_id"].endswith(":0")
    assert provenance["conflict"] is False


def test_mapping_flags_can_reproduce_baseline_without_new_evidence():
    baseline = MappingConfig(
        use_physical_state_hint=False,
        use_derived_in_bed_opportunity=False,
    )
    mapped = map_interval(record(activity="work", hint="sitting"), baseline)
    assert mapped["physical_state_class_id"] is None


def test_loader_aligns_diary_hint_and_derived_interval_without_mutating_inputs(tmp_path):
    source_root = tmp_path / "source"
    mobility_path = (
        source_root
        / "person_days"
        / "p1"
        / "2026-01-01"
        / "mobility"
        / "p1"
        / "2026-01-01.mobility.json"
    )
    diary_path = (
        source_root
        / "person_days"
        / "p1"
        / "2026-01-01"
        / "diaries"
        / "p1"
        / "2026-01-01.diary.json"
    )
    unrelated_path = source_root / "manifests" / "other_records.json"
    derived_root = tmp_path / "derived"
    derived_path = (
        derived_root
        / "person_days"
        / "p1"
        / "2026-01-01"
        / "in_bed_opportunity.json"
    )
    mobility = record(activity="idle_at_home")
    mobility_payload = {
        "persona_id": "p1",
        "date": "2026-01-01",
        "resolved_intervals": [mobility],
        "transition_outcomes": [
            {"persona_id": "p1", "date": "2026-01-01", "transition_id": "t0"}
        ],
    }
    diary_payload = {
        "persona_id": "p1",
        "date": "2026-01-01",
        "episodes": [
            {
                "activity_type": "idle_at_home",
                "start_time": "2026-01-01T00:00:00",
                "end_time": "2026-01-01T01:00:00",
                "physical_state_hint": "unknown",
            }
        ],
    }
    derived_payload = {
        "persona_id": "p1",
        "date": "2026-01-01",
        "schema_version": "sensor_in_bed_opportunity_v1",
        "derived_state": "in_bed_or_lying_opportunity",
        "code_commit": "b5cba4f673904ce5a88fdd47381e8d6ff5f5df2b",
        "code_tag": "research_ema_v1_reprofix4",
        "derived_intervals": [
            {
                "interval_id": "p1:2026-01-01:0:in_bed_opportunity:0",
                "source_episode_index": 0,
                "source_interval_start": "2026-01-01T00:00:00",
                "source_interval_end": "2026-01-01T01:00:00",
                "start_time": "2026-01-01T00:00:00",
                "end_time": "2026-01-01T01:00:00",
                "duration_seconds": 3600,
                "derived_state": "in_bed_or_lying_opportunity",
            }
        ],
    }
    for path, payload in (
        (mobility_path, mobility_payload),
        (diary_path, diary_payload),
        (
            unrelated_path,
            {
                "records": [
                    {
                        "persona_id": "p1",
                        "date": "2026-01-01",
                        "transition_id": "t0",
                    }
                ]
            },
        ),
        (derived_path, derived_payload),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    before = json.dumps(mobility_payload, sort_keys=True)
    rows = load_resolved_intervals(
        source_root,
        derived_root=derived_root,
        persona="p1",
        date="2026-01-01",
    )
    assert len(rows) == 1
    assert all(row.get("interval_id") for row in rows)
    intervals = [row for row in rows if row.get("interval_id") == mobility["interval_id"]]
    assert len(intervals) == 1
    enriched = intervals[0]
    assert enriched["physical_state_hint"] == "unknown"
    assert enriched["physical_state_hint_provenance"]["source_episode_index"] == 0
    assert enriched["in_bed_or_lying_opportunity"] is True
    assert enriched["in_bed_or_lying_opportunity_evidence"][0]["interval_id"].endswith(":0")
    assert json.dumps(mobility_payload, sort_keys=True) == before
