"""Deterministic, conservative DayForge-to-HARTH physical-state mapping."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import yaml


CLASS_NAMES = (
    "walking_slow",
    "walking_moderate",
    "walking_brisk",
    "running",
    "stair_climbing",
    "cycling_seated",
    "cycling_standing",
    "sitting",
    "standing",
    "lying",
)
PHYSICAL_STATE_HINT_VALUES = frozenset(
    {
        "walking",
        "running",
        "cycling",
        "stairs",
        "sitting",
        "standing",
        "lying",
        "mixed",
        "unknown",
    }
)
HINT_CLASS_IDS = {
    "running": 3,
    "stairs": 4,
    "sitting": 7,
    "standing": 8,
    "lying": 9,
}
PASSIVE_MODES = {
    "car",
    "vehicle",
    "bus",
    "transit",
    "train",
    "tram",
    "taxi",
    "passenger",
}
NONREALIZED = {
    "INFEASIBLE_TIME",
    "INFEASIBLE_MODE",
    "INFEASIBLE_ROUTE",
    "INFEASIBLE_NO_POI",
    "UNRESOLVED_STRUCTURE",
}


@dataclass(frozen=True)
class MappingConfig:
    version: str = "dayforge_harth_v1"
    slow_max_kmh: float = 4.0
    brisk_min_kmh: float = 5.5
    cycling_class: int = 5
    sitting_whitelist: tuple[str, ...] = ()
    standing_whitelist: tuple[str, ...] = ()
    use_physical_state_hint: bool = True
    use_derived_in_bed_opportunity: bool = True


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _enabled(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return True
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"true", "yes", "1"}:
            return True
        if normalized in {"false", "no", "0"}:
            return False
    raise ValueError(f"{name}.enabled must be a boolean")


def _class_id(value: Any, name: str) -> int:
    if isinstance(value, str):
        normalized = _text(value)
        if normalized not in CLASS_NAMES:
            raise ValueError(f"{name} must name one of the HARTH classes")
        return CLASS_NAMES.index(normalized)
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a HARTH class name or ID") from exc
    if result < 0 or result >= len(CLASS_NAMES):
        raise ValueError(f"{name} must be in [0, {len(CLASS_NAMES) - 1}]")
    return result


def load_mapping_config(path: str | Path) -> MappingConfig:
    """Load the small YAML policy used by the DayForge mapping CLI."""
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"mapping configuration does not exist: {config_path}")
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
    except yaml.YAMLError as exc:
        raise ValueError(
            f"could not parse mapping configuration: {config_path}"
        ) from exc
    if not isinstance(raw, Mapping):
        raise ValueError("mapping configuration must be a mapping")

    walking = _mapping(raw.get("walking_speed"), "walking_speed")
    cycling = _mapping(raw.get("cycling"), "cycling")
    sitting = tuple(_text(value) for value in raw.get("sitting_whitelist", ()))
    standing = tuple(_text(value) for value in raw.get("standing_whitelist", ()))
    if not all(isinstance(value, str) for value in sitting + standing):
        raise ValueError("posture whitelists must contain strings")
    hint = _mapping(raw.get("physical_state_hint"), "physical_state_hint")
    derived = _mapping(
        raw.get(
            "derived_in_bed_opportunity",
            raw.get("in_bed_or_lying_opportunity"),
        ),
        "derived_in_bed_opportunity",
    )
    return MappingConfig(
        version=str(raw.get("mapping_version", "dayforge_harth_v1")),
        slow_max_kmh=float(walking.get("slow_max_kmh", 4.0)),
        brisk_min_kmh=float(walking.get("brisk_min_kmh", 5.5)),
        cycling_class=_class_id(
            cycling.get("generic_route_class", "cycling_seated"),
            "cycling.generic_route_class",
        ),
        sitting_whitelist=sitting,
        standing_whitelist=standing,
        use_physical_state_hint=_enabled(hint.get("enabled"), "physical_state_hint"),
        use_derived_in_bed_opportunity=_enabled(
            derived.get("enabled"), "derived_in_bed_opportunity"
        ),
    )


def _text(value: Any) -> str:
    return (
        ""
        if value is None
        else str(value).strip().casefold().replace("-", "_").replace(" ", "_")
    )


def _get(record: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in record:
            return record[name]
    return None


def _speed(record: Mapping[str, Any]) -> Optional[float]:
    distance = _get(record, "route_distance_m", "distance_m", "route_distance")
    duration = _get(record, "route_duration_s", "duration_route_s", "route_duration")
    try:
        if distance is None or duration is None or float(duration) <= 0:
            return None
        return float(distance) / float(duration) * 3.6
    except (TypeError, ValueError):
        return None


def _bool_value(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().casefold() in {"true", "yes", "1"}
    return bool(value)


def _derived_evidence(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    value = record.get("in_bed_or_lying_opportunity_evidence", ())
    if isinstance(value, Mapping):
        return [dict(value)]
    if isinstance(value, list):
        return [dict(item) for item in value if isinstance(item, Mapping)]
    return []


def map_interval(
    record: Mapping[str, Any], config: MappingConfig = MappingConfig()
) -> dict[str, Any]:
    """Map one immutable resolved interval with conservative evidence precedence."""
    out = dict(record)
    mode = _text(_get(record, "mobility_mode", "mode", "transport_mode"))
    activity_value = _get(
        record, "semantic_activity", "activity", "activity_type", "semantic"
    )
    activity = _text(activity_value)
    interval_type = _text(_get(record, "interval_type", "type"))
    realization = str(
        _get(record, "realization_status", "realization", "status") or ""
    ).upper()
    raw_hint = record.get("physical_state_hint")
    hint = _text(raw_hint) if raw_hint is not None else None
    if hint not in PHYSICAL_STATE_HINT_VALUES:
        hint = None
    derived = _bool_value(
        record.get(
            "in_bed_or_lying_opportunity",
            record.get("derived_in_bed_opportunity", False),
        )
    )
    derived_evidence = _derived_evidence(record)
    speed = _speed(record)

    cls: int | None = None
    name: str | None = None
    status = "UNSUPPORTED_ACTIVITY"
    rule = "UNAVAILABLE"
    confidence: str | None = None
    reason: str | None = "UNSUPPORTED_ACTIVITY"
    mapping_source = "none"
    conflict = bool(record.get("physical_state_hint_conflict", False))
    conflict_reason: str | None = None

    def mark_conflict(message: str) -> None:
        nonlocal conflict, conflict_reason
        conflict = True
        if conflict_reason is None:
            conflict_reason = message

    def set_class(
        class_id: int,
        class_name: str,
        mapping_status: str,
        mapping_rule: str,
        source: str,
        mapping_confidence: str | None,
    ) -> None:
        nonlocal cls, name, status, rule, mapping_source, confidence, reason
        cls = class_id
        name = class_name
        status = mapping_status
        rule = mapping_rule
        mapping_source = source
        confidence = mapping_confidence
        reason = None

    def hint_is_incompatible_with(compatible: set[str | None]) -> bool:
        return hint is not None and hint not in compatible

    if realization in NONREALIZED or any(token in realization for token in NONREALIZED):
        status = reason = "NONREALIZED_MOBILITY"
        rule = "REALIZATION_GATE"
        mapping_source = "realized_mobility"
        if hint in {"walking", "running", "cycling", "stairs"} or derived:
            mark_conflict("non-realized mobility blocks weaker sensor evidence")
    elif mode in PASSIVE_MODES:
        status = reason = "UNSUPPORTED_PASSIVE_TRANSPORT"
        rule = "PASSIVE_TRANSPORT_EXCLUSION"
        mapping_source = "realized_mobility"
        if hint is not None or derived:
            mark_conflict("passive transport cannot be relabeled as a body state")
    elif interval_type in {"travel", "move", "movement", "transit"} and mode in {
        "walk",
        "walking",
        "pedestrian",
    }:
        mapping_source = "realized_mobility"
        if speed is None:
            status = reason = "WALK_SPEED_UNAVAILABLE"
            rule = "ROUTE_SPEED_REQUIRED"
        else:
            class_id = (
                0
                if speed <= config.slow_max_kmh
                else (2 if speed >= config.brisk_min_kmh else 1)
            )
            set_class(
                class_id,
                CLASS_NAMES[class_id],
                "ROUTE_SPEED_MAPPING",
                "ROUTE_SPEED_TO_WALKING_CLASS",
                "realized_mobility",
                "HIGH",
            )
        if hint_is_incompatible_with({None, "walking"}) or derived:
            mark_conflict("realized walking evidence outranks conflicting handoff evidence")
    elif activity in {"running", "run"} or mode in {"run", "running"}:
        set_class(
            3,
            CLASS_NAMES[3],
            "DIRECT_MOVEMENT_MAPPING",
            "EXPLICIT_RUNNING",
            "existing_explicit_semantic_evidence",
            "HIGH",
        )
        if hint_is_incompatible_with({None, "running"}) or derived:
            mark_conflict("explicit running evidence outranks derived in-bed evidence")
    elif activity in {
        "stairs",
        "stair_climbing",
        "stair_climb",
        "stair_climbing_up",
        "stair_climbing_down",
    }:
        set_class(
            4,
            CLASS_NAMES[4],
            "DIRECT_MOVEMENT_MAPPING",
            "EXPLICIT_STAIRS",
            "existing_explicit_semantic_evidence",
            "HIGH",
        )
        if hint_is_incompatible_with({None, "stairs"}) or derived:
            mark_conflict("explicit stair evidence outranks derived in-bed evidence")
    elif interval_type in {"travel", "move", "movement", "transit"} and mode in {
        "bike",
        "bicycle",
        "cycling",
        "cycle",
    }:
        set_class(
            config.cycling_class,
            CLASS_NAMES[config.cycling_class],
            "GENERIC_CYCLING_ASSUMPTION",
            "GENERIC_BICYCLE_TRAVEL_TO_CYCLING_SEATED",
            "realized_mobility",
            "MEDIUM",
        )
        if hint_is_incompatible_with({None, "cycling"}) or derived:
            mark_conflict("realized bicycle travel outranks derived in-bed evidence")
    elif activity in {"sleep", "lying_in_bed", "lying"} and mode not in {
        "walk",
        "walking",
        "bike",
        "bicycle",
        "car",
        "bus",
    }:
        set_class(
            9,
            CLASS_NAMES[9],
            "DIRECT_SEMANTIC_MAPPING",
            "SLEEP_OR_LYING_IN_BED_TO_LYING",
            "existing_explicit_semantic_evidence",
            "HIGH",
        )
        if hint_is_incompatible_with({None, "lying"}):
            mark_conflict("explicit lying evidence outranks conflicting hint")
    elif config.use_physical_state_hint and hint in HINT_CLASS_IDS:
        class_id = HINT_CLASS_IDS[hint]
        set_class(
            class_id,
            CLASS_NAMES[class_id],
            "PHYSICAL_STATE_HINT_MAPPING",
            f"PHYSICAL_STATE_HINT_TO_{CLASS_NAMES[class_id].upper()}",
            "physical_state_hint",
            "HIGH",
        )
        if derived and hint != "lying":
            mark_conflict("explicit physical-state hint outranks derived in-bed evidence")
    elif config.use_physical_state_hint and hint == "walking":
        status = reason = "WALK_SPEED_UNAVAILABLE"
        rule = "PHYSICAL_STATE_HINT_REQUIRES_ROUTE_SPEED"
        mapping_source = "physical_state_hint"
        if derived:
            mark_conflict("walking hint cannot be replaced by derived lying evidence")
    elif config.use_physical_state_hint and hint == "cycling":
        status = reason = "CYCLING_POSTURE_UNRESOLVED"
        rule = "PHYSICAL_STATE_HINT_REQUIRES_EXPLICIT_CYCLING_POSTURE"
        mapping_source = "physical_state_hint"
        if derived:
            mark_conflict("cycling hint cannot be replaced by derived lying evidence")
    elif activity in config.sitting_whitelist:
        set_class(
            7,
            CLASS_NAMES[7],
            "CONTEXTUAL_POSTURE_MAPPING",
            "APPROVED_SITTING_WHITELIST",
            "existing_explicit_semantic_evidence",
            "MEDIUM",
        )
        if derived:
            mark_conflict("approved sitting evidence outranks derived in-bed evidence")
    elif activity in config.standing_whitelist:
        set_class(
            8,
            CLASS_NAMES[8],
            "CONTEXTUAL_POSTURE_MAPPING",
            "APPROVED_STANDING_WHITELIST",
            "existing_explicit_semantic_evidence",
            "MEDIUM",
        )
        if derived:
            mark_conflict("approved standing evidence outranks derived in-bed evidence")
    elif config.use_derived_in_bed_opportunity and derived:
        set_class(
            9,
            CLASS_NAMES[9],
            "DERIVED_IN_BED_MAPPING",
            "DERIVED_IN_BED_OPPORTUNITY_TO_LYING",
            "derived_in_bed_opportunity",
            "MEDIUM",
        )
    elif activity in {"gym", "gym_session", "exercise_destination"}:
        reason = status = "MIXED_OR_UNSUPPORTED_ACTIVITY"
        rule = "CONSERVATIVE_MIXED_ACTIVITY_POLICY"
    elif activity:
        reason = status = "AMBIGUOUS_PHYSICAL_STATE"
        rule = "NO_GENERIC_POSTURE_FALLBACK"

    if derived and hint in {"sitting", "standing"} and mapping_source == "physical_state_hint":
        mark_conflict("explicit sitting or standing hint conflicts with in-bed opportunity")

    evidence_id = None
    evidence_source = None
    if derived_evidence:
        evidence_id = derived_evidence[0].get("interval_id")
        evidence_source = derived_evidence[0].get(
            "source_file", derived_evidence[0].get("source")
        )
    mapping_provenance = {
        "semantic_activity": _get(
            record, "semantic_activity", "activity", "activity_type", "semantic"
        ),
        "mobility_mode": _get(record, "mobility_mode", "mode", "transport_mode"),
        "route_distance_m": _get(
            record, "route_distance_m", "distance_m", "route_distance"
        ),
        "route_duration_s": _get(
            record, "route_duration_s", "duration_route_s", "route_duration"
        ),
        "realization_status": _get(
            record, "realization_status", "realization", "status"
        ),
        "mapping_source": mapping_source,
        "source_activity": activity_value,
        "physical_state_hint": hint,
        "physical_state_hint_evidence": record.get("physical_state_hint_provenance"),
        "derived_evidence_id": evidence_id,
        "derived_evidence_source": evidence_source,
        "final_harth_class": name,
        "policy_reason": rule,
        "conflict": conflict,
        "conflict_reason": conflict_reason,
    }
    out.update(
        {
            "persona_id": record.get("persona_id", record.get("persona")),
            "date": record.get("date", record.get("day")),
            "resolved_interval_id": record.get(
                "resolved_interval_id", record.get("interval_id")
            ),
            "source_episode_id": record.get(
                "source_episode_id", record.get("episode_id")
            ),
            "interval_type": record.get("interval_type", record.get("type")),
            "semantic_activity": record.get(
                "semantic_activity", record.get("activity", record.get("activity_type"))
            ),
            "start_time": record.get("start_time", record.get("start")),
            "end_time": record.get("end_time", record.get("end")),
            "duration_seconds": record.get("duration_seconds", record.get("duration")),
            "mobility_mode": record.get(
                "mobility_mode", record.get("mode", record.get("transport_mode"))
            ),
            "route_distance_m": record.get(
                "route_distance_m",
                record.get("distance_m", record.get("route_distance")),
            ),
            "route_duration_s": record.get(
                "route_duration_s",
                record.get("duration_route_s", record.get("route_duration")),
            ),
            "route_speed_kmh": speed,
            "physical_state_hint": hint,
            "physical_state_hint_provenance": record.get(
                "physical_state_hint_provenance"
            ),
            "in_bed_or_lying_opportunity": derived,
            "in_bed_or_lying_opportunity_evidence": derived_evidence,
            "derived_in_bed_opportunity": derived,
            "physical_state_class_id": cls,
            "physical_state_class_name": name,
            "mapping_status": status,
            "mapping_rule": rule,
            "mapping_confidence": confidence,
            "mapping_source": mapping_source,
            "mapping_conflict": conflict,
            "mapping_conflict_reason": conflict_reason,
            "imu_eligible": cls is not None,
            "imu_unavailable_reason": reason,
            "mapping_version": config.version,
            "mapping_provenance": mapping_provenance,
        }
    )
    return out


__all__ = [
    "CLASS_NAMES",
    "HINT_CLASS_IDS",
    "MappingConfig",
    "PHYSICAL_STATE_HINT_VALUES",
    "load_mapping_config",
    "map_interval",
]
