"""Read-only adapter for final DayForge MobilityResolvedDiary JSON outputs."""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any


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


def load_resolved_intervals(
    root: str | Path,
    *,
    persona: str | None = None,
    date: str | None = None,
    max_person_days: int | None = None,
) -> list[dict[str, Any]]:
    """Load final JSON records without modifying source payloads."""
    result = []
    seen_days = set()
    for path in discover_dayforge_json(root):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise DayForgeInputError(f"could not read DayForge JSON {path}") from exc
        for record in _records(payload):
            item = dict(record)
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
            str(x.get("resolved_interval_id", x.get("interval_id", ""))),
        )
    )
    return result


__all__ = ["DayForgeInputError", "discover_dayforge_json", "load_resolved_intervals"]
