"""Command implementation for the exact-duration fusion CLI."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .dayforge import load_resolved_intervals
from .fusion import (
    SegmentResult,
    StitchConfig,
    audit_segments,
    generate_segment,
    target_samples,
    validate_checkpoint_contract,
)


def _mapping_records(root: str) -> list[dict]:
    path = Path(root)
    if path.is_file():
        files = [path]
    else:
        files = sorted(path.rglob("physical_state_mapping.csv"))
    if not files:
        raise ValueError("mapping root contains no physical_state_mapping.csv")
    rows = []
    for f in files:
        with f.open(newline="", encoding="utf-8") as h:
            rows.extend(dict(r) for r in csv.DictReader(h))
    for r in rows:
        for key in ("physical_state_class_id",):
            if r.get(key) not in (None, ""):
                r[key] = int(r[key])
        if r.get("imu_eligible") is not None:
            r["imu_eligible"] = r["imu_eligible"].lower() == "true"
    return rows


def _resolved_interval_id(record: Mapping[str, Any]) -> str | None:
    value = record.get("resolved_interval_id")
    if value in (None, ""):
        value = record.get("interval_id")
    if value in (None, ""):
        return None
    return str(value)


def _result_record(result: SegmentResult | Mapping[str, Any]) -> dict[str, Any]:
    return dict(result.record if isinstance(result, SegmentResult) else result)


def _result_payload(
    result: SegmentResult | Mapping[str, Any], segment_id: str
) -> dict[str, Any]:
    """Return a JSON-safe manifest for either a generated or unavailable result."""
    payload: dict[str, Any] = {"record": _result_record(result)}
    if isinstance(result, SegmentResult):
        payload.update(
            {
                "segment_id": segment_id,
                "provenance": result.provenance,
                "stitch_audit": result.stitch_audit,
                "array_file": segment_id + ".npz",
            }
        )
    return payload


def run_synthesize(args) -> int:
    from .fusion import FusionError

    source = load_resolved_intervals(
        args.dayforge_root,
        persona=args.persona,
        date=args.date,
        max_person_days=args.max_person_days,
    )
    mappings = _mapping_records(args.mapping_root)
    by_id = {}
    for mapping in mappings:
        interval_id = _resolved_interval_id(mapping)
        if interval_id is None:
            continue
        by_id[
            (
                str(mapping.get("persona_id")),
                str(mapping.get("date")),
                interval_id,
            )
        ] = mapping
    records = []
    for raw in source:
        interval_id = _resolved_interval_id(raw)
        if interval_id is None:
            continue
        key = (
            str(raw.get("persona_id", raw.get("persona"))),
            str(raw.get("date", raw.get("day"))),
            interval_id,
        )
        if key not in by_id:
            raise FusionError(f"no physical mapping for interval {key[2]}")
        merged = dict(raw)
        merged.update(by_id[key])
        records.append(merged)
    estimates = []
    for r in records:
        n = target_samples(r.get("duration_seconds", r.get("duration")))
        estimates.append(
            {
                "interval_id": r.get("resolved_interval_id"),
                "target_samples": n,
                "eligible": bool(r.get("imu_eligible")),
                "windows": 1
                if n < 160
                else int(np.ceil(n / (160 - args.stitch_overlap))),
            }
        )
    if not 0 <= args.stitch_overlap < 160:
        raise FusionError("--stitch-overlap must be in [0, 160)")
    if args.dry_run:
        print(
            json.dumps(
                {
                    "command": "synthesize-dayforge",
                    "dry_run": True,
                    "intervals": len(records),
                    "estimated_generated_samples": sum(
                        x["target_samples"] for x in estimates if x["eligible"]
                    ),
                    "estimates": estimates,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if (
        not args.vae_checkpoint
        or not args.flow_checkpoint
        or not args.normalization_metadata
    ):
        raise FusionError(
            "--vae-checkpoint, --flow-checkpoint, and --normalization-metadata are required unless --dry-run"
        )
    norm = json.loads(Path(args.normalization_metadata).read_text(encoding="utf-8"))
    from ..checkpoints import load_flow_checkpoint, load_vae_checkpoint
    from ..generation.flow import sample_reverse_euler
    import torch

    flow, fi = load_flow_checkpoint(
        args.flow_checkpoint,
        channels=3,
        latent_channels=48,
        num_classes=10,
        device=args.device,
    )
    vae, vi = load_vae_checkpoint(
        args.vae_checkpoint,
        channels=3,
        latent_channels=48,
        down_levels=2,
        device=args.device,
    )
    mean = np.asarray(norm["mean"], dtype=np.float32)
    std = np.asarray(norm["std"], dtype=np.float32)
    if (
        mean.shape not in {(3,), (3, 1)}
        or std.shape != mean.shape
        or not np.isfinite(std).all()
        or np.any(std <= 0)
    ):
        raise FusionError(
            "normalization metadata must contain finite positive 3-channel mean/std"
        )
    validate_checkpoint_contract(vi.to_mapping(), fi.to_mapping(), norm)

    def generator(*, class_id, seed, window_index, window_length):
        labels = torch.tensor([class_id], device=args.device)
        z = sample_reverse_euler(
            flow, labels, (48, 40), num_steps=10, seed=seed, device=args.device
        )
        with torch.no_grad():
            x = vae.decode(z).detach().cpu().numpy()[0]
        return x * std.reshape(-1, 1) + mean.reshape(-1, 1)

    results = []
    for r in records:
        results.append(
            generate_segment(
                r,
                generator,
                global_seed=args.seed,
                vae_checkpoint=str(Path(args.vae_checkpoint).resolve()),
                flow_checkpoint=str(Path(args.flow_checkpoint).resolve()),
                normalization_metadata=str(Path(args.normalization_metadata).resolve()),
                stitch=StitchConfig(args.stitch_overlap),
            )
        )
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for index, result in enumerate(results):
        key = f"segment_{index:06d}"
        record = _result_record(result)
        directory = out / str(record.get("persona_id")) / str(record.get("date"))
        directory.mkdir(parents=True, exist_ok=True)
        payload = _result_payload(result, key)
        if isinstance(result, SegmentResult):
            np.savez_compressed(directory / (key + ".npz"), signal=result.signal)
        (directory / (key + ".json")).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    summary = audit_segments(results)
    (out / "fusion_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (out / "unsupported_intervals.csv").open(
        "w", newline="", encoding="utf-8"
    ) as h:
        rows = [_result_record(x) for x in results if not isinstance(x, SegmentResult)]
        writer = csv.DictWriter(
            h,
            fieldnames=[
                "resolved_interval_id",
                "status",
                "imu_unavailable_reason",
                "duration_seconds",
            ],
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)
    failures = [
        _result_record(x)
        for x in results
        if _result_record(x).get("status") == "IMU_GENERATION_FAILED"
    ]
    (out / "generation_failures.json").write_text(
        json.dumps(failures, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (out / "fusion_validation_report.md").write_text(
        "# Fusion validation report\n\n"
        + json.dumps(summary, indent=2, sort_keys=True)
        + "\n\nGeneration failures are distinct from intentional IMU unavailability.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0
