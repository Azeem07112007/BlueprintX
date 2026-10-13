"""
backend/engine/export/exporter.py
Blueprint-X — Milestone 5

Serialises the Building model to the canonical JSON format (schema v2.1.0).
Includes metric units, coordinate transform specification, 3D parameters,
player spawn information, and architectural validation reports.
"""
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from backend.models.dataclasses import Building
from backend.engine.model3d.geometry_validator import GeometryValidator

SCHEMA_VERSION = "2.1.0"
MILESTONE      = 5


def compute_player_spawn(building: Building, px_per_m: float = 100.0) -> Dict[str, Any]:
    """Determines a safe first-person spawn location OUTSIDE in front of the house building."""
    w, h = building.image_size
    width_m = round(w / px_per_m, 2)

    # Spawn player outside the house model (in front of the main entrance / south side)
    spawn_x = round(width_m * 0.5, 2)
    spawn_y = 0.0   # Ground level
    spawn_z = -4.5  # 4.5m outside in front of the house facing the building

    return {
        "room_id": None,
        "room_label": "Exterior Ground",
        "position_m": [spawn_x, spawn_y, spawn_z],
        "rotation_deg": 0.0,
    }


class Exporter:
    def export(self, building: Building, output_path: str) -> None:
        """Serializes the Building dataclass to a canonical JSON file (schema v2.1.0)."""
        px_per_m = building.model_3d.px_per_m if building.model_3d else 100.0
        w_px, h_px = building.image_size

        # Run validation check
        validator = GeometryValidator()
        validation_report = validator.validate_architecture_report(building, px_per_m=px_per_m)

        # Player spawn point calculation
        spawn = compute_player_spawn(building, px_per_m=px_per_m)

        payload: Dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "milestone": MILESTONE,
            "units": "meters",
            "coordinate_system": {
                "source": "image",
                "world_axis": "Y_UP",
                "image_axis": "X_RIGHT_Y_DOWN",
                "origin_offset_m": [0.0, 0.0, 0.0],
                "px_per_m": round(px_per_m, 2),
            },
            "building": {
                "width_m": round(w_px / px_per_m, 2),
                "depth_m": round(h_px / px_per_m, 2),
                "wall_height_m": building.model_3d.floor_height_m if building.model_3d else 2.8,
                "floor_height_m": 0.1,
            },
            "metadata": {
                "source": Path(building.source_path).name,
                "source_path": str(building.source_path),
                "image_size": {
                    "width":  w_px,
                    "height": h_px,
                },
                "scale": building.scale or "1:100",
                "exported_at_utc": datetime.now(timezone.utc).isoformat(),
            },
            "statistics": {
                "total_walls":     len(building.walls),
                "outer_walls":     sum(1 for w in building.walls if w.wall_type == "outer"),
                "internal_walls":  sum(1 for w in building.walls if w.wall_type == "internal"),
                "total_rooms":     len(building.rooms),
                "total_doors":     len(building.doors),
                "total_windows":   len(building.windows),
                "total_furniture": len(building.furniture),
            },
            "player_spawn": spawn,
            "validation": validation_report,
            "walls":       [w.to_dict() for w in building.walls],
            "doors":       [d.to_dict() for d in building.doors],
            "windows":     [w.to_dict() for w in building.windows],
            "furniture":   [f.to_dict() for f in building.furniture],
            "columns":     [c.to_dict() for c in building.columns],
            "stairs":      [s.to_dict() for s in building.stairs],
            "text_blocks": [t.to_dict() for t in building.text_blocks],
            "rooms":       [r.to_dict() for r in building.rooms],
            "topology": {
                "nodes": [n.to_dict() for n in building.topology_nodes],
                "edges": [e.to_dict() for e in building.topology_edges],
            },
            "model_3d": {
                "available": building.model_3d is not None,
                "vertex_count":   building.model_3d.vertex_count   if building.model_3d else 0,
                "triangle_count": building.model_3d.triangle_count if building.model_3d else 0,
                "floor_height_m": building.model_3d.floor_height_m if building.model_3d else 2.8,
                "px_per_m":       building.model_3d.px_per_m       if building.model_3d else 100.0,
                "center":         list(building.model_3d.center)   if building.model_3d else [0, 0, 0],
                "bounds_min":     list(building.model_3d.bounds_min) if building.model_3d else [0, 0, 0],
                "bounds_max":     list(building.model_3d.bounds_max) if building.model_3d else [0, 0, 0],
                "export_hint":    "Use Export OBJ / Export GLB for full 3D geometry",
            },
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

