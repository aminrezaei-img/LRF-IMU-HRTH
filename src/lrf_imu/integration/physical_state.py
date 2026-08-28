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


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


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


def map_interval(
    record: Mapping[str, Any], config: MappingConfig = MappingConfig()
) -> dict[str, Any]:
    """Map one immutable resolved interval; never mutates ``record``."""
    out = dict(record)
    mode = _text(_get(record, "mobility_mode", "mode", "transport_mode"))
    activity = _text(
        _get(record, "semantic_activity", "activity", "activity_type", "semantic")
    )
    interval_type = _text(_get(record, "interval_type", "type"))
    realization = str(
        _get(record, "realization_status", "realization", "status") or ""
    ).upper()
    speed = _speed(record)
    cls = None
    name = None
    status = "UNSUPPORTED_ACTIVITY"
    rule = "UNAVAILABLE"
    confidence = None
    reason = "UNSUPPORTED_ACTIVITY"
    if realization in NONREALIZED or any(token in realization for token in NONREALIZED):
        status = reason = "NONREALIZED_MOBILITY"
        rule = "REALIZATION_GATE"
    elif mode in PASSIVE_MODES:
        status = reason = "UNSUPPORTED_PASSIVE_TRANSPORT"
        rule = "PASSIVE_TRANSPORT_EXCLUSION"
    elif interval_type in {"travel", "move", "movement", "transit"} and mode in {
        "walk",
        "walking",
        "pedestrian",
    }:
        if speed is None:
            status = reason = "WALK_SPEED_UNAVAILABLE"
            rule = "ROUTE_SPEED_REQUIRED"
        else:
            cls = (
                0
                if speed <= config.slow_max_kmh
                else (2 if speed >= config.brisk_min_kmh else 1)
            )
            name = CLASS_NAMES[cls]
            status = "ROUTE_SPEED_MAPPING"
            rule = "ROUTE_SPEED_TO_WALKING_CLASS"
            confidence = "HIGH"
            reason = None
    elif activity in {"running", "run"} or mode in {"run", "running"}:
        cls = 3
        name = CLASS_NAMES[cls]
        status = "DIRECT_MOVEMENT_MAPPING"
        rule = "EXPLICIT_RUNNING"
        confidence = "HIGH"
        reason = None
    elif activity in {
        "stairs",
        "stair_climbing",
        "stair_climb",
        "stair_climbing_up",
        "stair_climbing_down",
    }:
        cls = 4
        name = CLASS_NAMES[cls]
        status = "DIRECT_MOVEMENT_MAPPING"
        rule = "EXPLICIT_STAIRS"
        confidence = "HIGH"
        reason = None
    elif interval_type in {"travel", "move", "movement", "transit"} and mode in {
        "bike",
        "bicycle",
        "cycling",
        "cycle",
    }:
        cls = config.cycling_class
        name = CLASS_NAMES[cls]
        status = "GENERIC_CYCLING_ASSUMPTION"
        rule = "GENERIC_BICYCLE_TRAVEL_TO_CYCLING_SEATED"
        confidence = "MEDIUM"
        reason = None
    elif activity in {"sleep", "lying_in_bed", "lying"} and mode not in {
        "walk",
        "walking",
        "bike",
        "bicycle",
        "car",
        "bus",
    }:
        cls = 9
        name = CLASS_NAMES[cls]
        status = "DIRECT_SEMANTIC_MAPPING"
        rule = "SLEEP_OR_LYING_IN_BED_TO_LYING"
        confidence = "HIGH"
        reason = None
    elif activity in config.sitting_whitelist:
        cls = 7
        name = CLASS_NAMES[cls]
        status = "CONTEXTUAL_POSTURE_MAPPING"
        rule = "APPROVED_SITTING_WHITELIST"
        confidence = "MEDIUM"
        reason = None
    elif activity in config.standing_whitelist:
        cls = 8
        name = CLASS_NAMES[cls]
        status = "CONTEXTUAL_POSTURE_MAPPING"
        rule = "APPROVED_STANDING_WHITELIST"
        confidence = "MEDIUM"
        reason = None
    elif activity in {"gym", "gym_session", "exercise_destination"}:
        reason = status = "MIXED_OR_UNSUPPORTED_ACTIVITY"
        rule = "CONSERVATIVE_MIXED_ACTIVITY_POLICY"
    elif activity:
        reason = status = "AMBIGUOUS_PHYSICAL_STATE"
        rule = "NO_GENERIC_POSTURE_FALLBACK"
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
            "physical_state_class_id": cls,
            "physical_state_class_name": name,
            "mapping_status": status,
            "mapping_rule": rule,
            "mapping_confidence": confidence,
            "imu_eligible": cls is not None,
            "imu_unavailable_reason": reason,
            "mapping_version": config.version,
            "mapping_provenance": {
                "semantic_activity": _get(
                    record, "semantic_activity", "activity", "activity_type", "semantic"
                ),
                "mobility_mode": _get(
                    record, "mobility_mode", "mode", "transport_mode"
                ),
                "route_distance_m": _get(
                    record, "route_distance_m", "distance_m", "route_distance"
                ),
                "route_duration_s": _get(
                    record, "route_duration_s", "duration_route_s", "route_duration"
                ),
                "realization_status": _get(
                    record, "realization_status", "realization", "status"
                ),
            },
        }
    )
    return out


__all__ = ["CLASS_NAMES", "MappingConfig", "load_mapping_config", "map_interval"]
