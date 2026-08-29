"""Coverage and immutability audit for physical-state bridge records."""

from __future__ import annotations
from collections import Counter
from typing import Any, Iterable
from .physical_state import CLASS_NAMES


def audit_mappings(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(records)
    total = sum(
        float(r.get("duration_seconds", r.get("duration", 0)) or 0) for r in rows
    )
    mapped = [r for r in rows if r.get("imu_eligible") is True]
    mapped_duration = sum(
        float(r.get("duration_seconds", r.get("duration", 0)) or 0) for r in mapped
    )
    by_class = {name: {"intervals": 0, "duration_seconds": 0.0} for name in CLASS_NAMES}
    for r in mapped:
        cid = r["physical_state_class_id"]
        b = by_class[CLASS_NAMES[cid]]
        b["intervals"] += 1
        b["duration_seconds"] += float(
            r.get("duration_seconds", r.get("duration", 0)) or 0
        )
    return {
        "total_resolved_intervals": len(rows),
        "total_resolved_duration_seconds": total,
        "mapped_intervals": len(mapped),
        "mapped_duration_seconds": mapped_duration,
        "unavailable_intervals": len(rows) - len(mapped),
        "unavailable_duration_seconds": total - mapped_duration,
        "interval_mapping_coverage": len(mapped) / len(rows) if rows else 0.0,
        "duration_mapping_coverage": mapped_duration / total if total else 0.0,
        "by_class": by_class,
        "by_reason": dict(
            Counter(
                r.get("imu_unavailable_reason")
                for r in rows
                if not r.get("imu_eligible")
            ),
        ),
        "by_status": dict(Counter(r.get("mapping_status") for r in rows)),
    }


__all__ = ["audit_mappings"]
