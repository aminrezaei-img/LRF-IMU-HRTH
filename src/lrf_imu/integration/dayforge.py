"""Read-only adapters for final DayForge and sensor-handoff JSON outputs."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


PHYSICAL_STATE_HINT_VALUES = frozenset(
    {"walking", "running", "cycling", "stairs", "sitting", "standing", "lying", "mixed", "unknown"}
)
DERIVED_IN_BED_STATE = "in_bed_or_lying_opportunity"


class DayForgeInputError(ValueError):
    pass


def _records(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [x for x in value if isinstance(x, dict)]
    if isinstance(value, dict):
        for key in (
            "resolved_intervals",
            "intervals",
            "mobility_intervals",
            "records",
            "diary",
        ):
            if isinstance(value.get(key), (list, dict)):
                return _records(value[key])
        # A persona/day mapping may contain nested lists.
        result = []
        for child in value.values():
            result.extend(_records(child))
        return result
    return []


def discover_dayforge_json(root: str | Path) -> list[Path]:
    path = Path(root).expanduser()
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"DayForge root does not exist: {path}")
    return sorted(path.rglob("*.json"), key=lambda x: x.as_posix().casefold())


def _matching_files(root: str | Path, suffix: str) -> list[Path]:
    path = Path(root).expanduser()
    if path.is_file():
        return [path] if path.name.casefold().endswith(suffix.casefold()) else []
    if not path.is_dir():
        raise FileNotFoundError(f"DayForge root does not exist: {path}")
    return sorted(
        (
            item
            for item in path.rglob("*")
            if item.is_file() and item.name.casefold().endswith(suffix.casefold())
        ),
        key=lambda x: x.as_posix().casefold(),
    )


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DayForgeInputError(f"could not read DayForge JSON {path}") from exc


def _identity(payload: dict[str, Any]) -> tuple[str | None, str | None]:
    persona = payload.get("persona_id", payload.get("persona", payload.get("person_id")))
    day = payload.get("date", payload.get("day"))
    return (
        str(persona) if persona is not None else None,
        str(day) if day is not None else None,
    )


def _interval_id(record: dict[str, Any]) -> Any:
    value = record.get("resolved_interval_id")
    if value in (None, ""):
        value = record.get("interval_id")
    return value


def _normalise_hint(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().casefold().replace("-", "_").replace(" ", "_")
    return text if text in PHYSICAL_STATE_HINT_VALUES else None


def _parse_time(value: Any) -> datetime | None:
    if value is None:
        return None
    try:
        parsed = value if isinstance(value, datetime) else datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is not None:
        return parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed


def _overlaps(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_start = _parse_time(left.get("start_time"))
    left_end = _parse_time(left.get("end_time"))
    right_start = _parse_time(right.get("start_time"))
    right_end = _parse_time(right.get("end_time"))
    if None in (left_start, left_end, right_start, right_end):
        return True
    return left_start < right_end and right_start < left_end


def _episode_items(payload: dict[str, Any]) -> list[dict[str, Any]]:
    episodes = payload.get("episodes")
    if isinstance(episodes, list):
        return [item for item in episodes if isinstance(item, dict)]
    grounded = payload.get("grounded_episodes")
    if isinstance(grounded, list):
        return [
            item.get("episode", item)
            for item in grounded
            if isinstance(item, dict)
            and isinstance(item.get("episode", item), dict)
        ]
    return []


def load_physical_state_hints(
    root: str | Path,
    *,
    persona: str | None = None,
    date: str | None = None,
) -> list[dict[str, Any]]:
    """Read canonical episode hints, preferring immutable semantic diaries."""
    preferred: dict[tuple[str | None, str | None], Path] = {}
    paths = _matching_files(root, ".diary.json")
    fallback_paths = _matching_files(root, ".grounded.json")
    for path in [*paths, *fallback_paths]:
        payload = _read_json(path)
        if not isinstance(payload, dict):
            continue
        identity = _identity(payload)
        if identity not in preferred or path.name.casefold().endswith(".diary.json"):
            preferred[identity] = path

    result: list[dict[str, Any]] = []
    for path in preferred.values():
        payload = _read_json(path)
        if not isinstance(payload, dict):
            continue
        pid, day = _identity(payload)
        if persona is not None and pid != str(persona):
            continue
        if date is not None and day != str(date):
            continue
        for index, episode in enumerate(_episode_items(payload)):
            hint = episode.get("physical_state_hint")
            if hint is None:
                continue
            result.append(
                {
                    "persona_id": pid,
                    "date": day,
                    "source_episode_index": episode.get("source_episode_index", index),
                    "physical_state_hint": hint,
                    "source_activity": episode.get("activity_type", episode.get("activity")),
                    "start_time": episode.get("start_time", episode.get("start")),
                    "end_time": episode.get("end_time", episode.get("end")),
                    "source_file": str(path),
                    "hint_value_valid": _normalise_hint(hint) is not None,
                }
            )
    return result


def load_in_bed_opportunities(
    root: str | Path,
    *,
    persona: str | None = None,
    date: str | None = None,
) -> list[dict[str, Any]]:
    """Read derived in-bed opportunity intervals without changing their source."""
    result: list[dict[str, Any]] = []
    for path in _matching_files(root, "in_bed_opportunity.json"):
        payload = _read_json(path)
        if not isinstance(payload, dict):
            continue
        pid, day = _identity(payload)
        if persona is not None and pid != str(persona):
            continue
        if date is not None and day != str(date):
            continue
        intervals = payload.get("derived_intervals", [])
        if not isinstance(intervals, list):
            continue
        for interval in intervals:
            if not isinstance(interval, dict):
                continue
            item = dict(interval)
            item.update(
                {
                    "persona_id": pid,
                    "date": day,
                    "source_file": str(path),
                    "handoff_schema_version": payload.get("schema_version"),
                    "handoff_code_commit": payload.get("code_commit"),
                    "handoff_code_tag": payload.get("code_tag"),
                    "source_diary_path": payload.get("source_diary_path"),
                    "source_diary_sha256": payload.get("source_diary_sha256"),
                }
            )
            result.append(item)
    return result


def _source_episode_index(record: dict[str, Any]) -> int | None:
    value = record.get("source_episode_index")
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _match_evidence(
    record: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    pid, day = _identity(record)
    index = _source_episode_index(record)
    same_identity = [
        item
        for item in candidates
        if item.get("persona_id") == pid and item.get("date") == day
    ]
    if index is not None:
        exact = [
            item
            for item in same_identity
            if _source_episode_index(item) == index and _overlaps(record, item)
        ]
        if exact:
            return exact
    return [item for item in same_identity if _overlaps(record, item)]


def _enrich_record(
    record: dict[str, Any],
    hints: list[dict[str, Any]],
    derived: list[dict[str, Any]],
) -> dict[str, Any]:
    item = dict(record)
    hint_matches = _match_evidence(item, hints)
    hint_values = {
        _normalise_hint(candidate.get("physical_state_hint"))
        for candidate in hint_matches
        if _normalise_hint(candidate.get("physical_state_hint")) is not None
    }
    if len(hint_values) == 1:
        item["physical_state_hint"] = next(iter(hint_values))
    elif len(hint_values) > 1:
        item["physical_state_hint"] = None
        item["physical_state_hint_conflict"] = True
    elif "physical_state_hint" not in item:
        item["physical_state_hint"] = None
    if hint_matches:
        item["physical_state_hint_provenance"] = (
            hint_matches[0] if len(hint_matches) == 1 else hint_matches
        )

    derived_matches = _match_evidence(item, derived)
    if derived_matches:
        item["in_bed_or_lying_opportunity"] = True
        item["in_bed_or_lying_opportunity_evidence"] = derived_matches
    else:
        item.setdefault("in_bed_or_lying_opportunity", False)
        item.setdefault("in_bed_or_lying_opportunity_evidence", [])
    item["derived_in_bed_opportunity"] = bool(item["in_bed_or_lying_opportunity"])
    return item


def load_resolved_intervals(
    root: str | Path,
    *,
    persona: str | None = None,
    date: str | None = None,
    max_person_days: int | None = None,
    derived_root: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Load final JSON records without modifying source payloads."""
    result = []
    seen_days = set()
    for path in discover_dayforge_json(root):
        payload = _read_json(path)
        for record in _records(payload):
            item = dict(record)
            if _interval_id(item) in (None, ""):
                continue
            item.setdefault("_source_file", str(path))
            pid = item.get("persona_id", item.get("persona", item.get("person_id")))
            day = item.get("date", item.get("day"))
            if persona is not None and str(pid) != str(persona):
                continue
            if date is not None and str(day) != str(date):
                continue
            key = (str(pid), str(day))
            if (
                max_person_days
                and key not in seen_days
                and len(seen_days) >= max_person_days
            ):
                continue
            seen_days.add(key)
            result.append(item)
    result.sort(
        key=lambda x: (
            str(x.get("persona_id", x.get("persona", ""))),
            str(x.get("date", x.get("day", ""))),
            str(_interval_id(x)),
        )
    )
    hints = load_physical_state_hints(root, persona=persona, date=date)
    derived = (
        load_in_bed_opportunities(derived_root, persona=persona, date=date)
        if derived_root is not None
        else []
    )
    if hints or derived:
        result = [_enrich_record(item, hints, derived) for item in result]
    return result


__all__ = [
    "DERIVED_IN_BED_STATE",
    "DayForgeInputError",
    "PHYSICAL_STATE_HINT_VALUES",
    "discover_dayforge_json",
    "load_in_bed_opportunities",
    "load_physical_state_hints",
    "load_resolved_intervals",
]
