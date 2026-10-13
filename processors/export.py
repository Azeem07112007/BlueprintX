"""
processors/export.py
Blueprint-X — Redesign

JSON export: schema v3.0.0

Public API:
    payload = build_payload(pipeline_result)
    path    = save_json(payload, output_path)
    default = suggest_path(image_path)
"""

from __future__ import annotations

import json
import datetime
from pathlib import Path
from typing import Dict, Any


SCHEMA_VERSION = "3.0.0"


class ExportError(Exception):
    pass


def build_payload(result) -> Dict[str, Any]:
    """Build the complete export dict from a PipelineResult."""
    now = datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"
    h, w = result.original_bgr.shape[:2]

    walls = []
    if result.wall_result:
        walls = [seg.to_dict() for seg in result.wall_result.wall_segments]

    rooms = []
    if result.room_result:
        rooms = [r.to_dict() for r in result.room_result.rooms]

    doors = []
    if result.door_result:
        doors = [d.to_dict() for d in result.door_result.doors]

    topology = {"nodes": [], "edges": []}
    if result.topology_result:
        topology = result.topology_result.to_dict()

    return {
        "schema_version": SCHEMA_VERSION,
        "metadata": {
            "file_name": Path(result.image_path).name if result.image_path else "unknown",
            "file_path": str(result.image_path) if result.image_path else "",
            "exported_at_utc": now,
            "tool": "Blueprint-X",
            "pipeline_version": "2.0",
        },
        "floor_plan": {
            "width_px": w,
            "height_px": h,
        },
        "walls": walls,
        "rooms": rooms,
        "doors": doors,
        "windows": [],       # Future milestone
        "topology": topology,
        "statistics": {
            "total_rooms": result.room_result.estimated_room_count if result.room_result else 0,
            "total_walls": len(walls),
            "total_doors": result.door_result.total_doors if result.door_result else 0,
            "estimated_floor_area_px2": round(result.room_result.total_area_px if result.room_result else 0, 1),
        },
    }


def save_json(payload: Dict[str, Any], output_path: str, indent: int = 2) -> Path:
    """Write payload to a JSON file."""
    out = Path(output_path)
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=indent, ensure_ascii=False)
    except OSError as e:
        raise ExportError(f"Could not write export file: {e}") from e
    return out.resolve()


def suggest_path(image_path: str) -> str:
    """Suggest a JSON output path next to the source image."""
    p = Path(image_path)
    return str(p.parent / f"{p.stem}_blueprint_v3.json")
