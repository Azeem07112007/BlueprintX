"""
modules/json_exporter.py
Blueprint-X Prototype - Milestone 1

Serialises detection results into a structured JSON file.
The schema is intentionally forward-compatible so future milestones can
add doors, windows, materials, and 3-D data without breaking parsers.
"""

import json
import datetime
from pathlib import Path
from typing import Optional, Dict, Any

from modules.wall_detector import WallDetectionResult
from modules.room_detector import RoomDetectionResult
from modules.image_loader import FloorPlanImage


# ---------------------------------------------------------------------------
# Schema version – bump when making breaking changes to the JSON format
# ---------------------------------------------------------------------------
SCHEMA_VERSION = "2.0.0"


class ExportError(Exception):
    """Raised when the export cannot be completed."""
    pass


def build_export_payload(analysis: Any) -> Dict[str, Any]:
    """
    Build the complete export dictionary from a BlueprintAnalysis object.
    """
    now_iso = datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"

    payload: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "metadata": {
            **analysis.floor_plan.metadata(),
            "exported_at_utc": now_iso,
            "tool": "Blueprint-X Prototype",
            "milestone": 1.5,
        },
        "floor_plan": {
            "width_px": analysis.floor_plan.width,
            "height_px": analysis.floor_plan.height,
        },
        "walls": [seg.to_dict() for seg in analysis.walls.walls],
        "rooms": [room.to_dict() for room in analysis.rooms.rooms],
        "doors": [door.to_dict() for door in analysis.doors_windows.doors],
        "windows": [window.to_dict() for window in analysis.doors_windows.windows],
        "stairs": [stair.to_dict() for stair in analysis.doors_windows.stairs],
        "furniture": [f.to_dict() for f in analysis.furniture],
        
        "statistics": {
            "total_rooms": analysis.rooms.estimated_room_count,
            "total_doors": analysis.doors_windows.total_doors,
            "total_windows": analysis.doors_windows.total_windows,
            "total_staircases": analysis.doors_windows.total_stairs,
            "total_furniture": len(analysis.furniture),
            "estimated_floor_area_px2": round(analysis.estimated_floor_area_px2, 1),
            "reconstruction_confidence": round(analysis.reconstruction_confidence, 3),
        }
    }

    return payload


def export_to_json(
    payload: Dict[str, Any],
    output_path: str,
    indent: int = 2,
) -> Path:
    """
    Write the payload to a JSON file on disk.

    Args:
        payload:     Dict produced by build_export_payload().
        output_path: Destination file path (will create directories if needed).
        indent:      JSON indentation level.

    Returns:
        Resolved Path of the written file.

    Raises:
        ExportError: If the file cannot be written.
    """
    out_path = Path(output_path)

    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=indent, ensure_ascii=False)
    except OSError as exc:
        raise ExportError(f"Could not write export file: {exc}") from exc

    return out_path.resolve()


def suggest_output_path(floor_plan: FloorPlanImage) -> str:
    """
    Generate a sensible default output path next to the source image.
    E.g. /home/user/plans/house.png  ->  /home/user/plans/house_blueprint.json
    """
    stem = floor_plan.file_path.stem
    parent = floor_plan.file_path.parent
    return str(parent / f"{stem}_blueprint.json")
