"""
modules/door_window_detector.py
Blueprint-X Prototype - Milestone 1.5

Detects doors, windows, and staircases from structural edge maps.
"""

import cv2
import numpy as np
import math
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional

@dataclass
class DoorMarker:
    center: Tuple[int, int]
    radius: int
    angle_deg: float
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "center": [int(self.center[0]), int(self.center[1])],
            "radius_px": int(self.radius),
            "angle_deg": round(float(self.angle_deg), 1),
            "confidence": round(float(self.confidence), 3)
        }

@dataclass
class WindowMarker:
    start: Tuple[int, int]
    end: Tuple[int, int]
    thickness_px: int = 5
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start": [int(self.start[0]), int(self.start[1])],
            "end": [int(self.end[0]), int(self.end[1])],
            "thickness_px": int(self.thickness_px),
            "confidence": round(float(self.confidence), 3)
        }

@dataclass
class StaircaseRegion:
    bounding_box: Tuple[int, int, int, int]  # x, y, w, h
    step_count: int = 0
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bounding_box": {
                "x": int(self.bounding_box[0]),
                "y": int(self.bounding_box[1]),
                "width": int(self.bounding_box[2]),
                "height": int(self.bounding_box[3]),
            },
            "step_count": int(self.step_count),
            "confidence": round(float(self.confidence), 3)
        }

@dataclass
class DoorWindowResult:
    doors: List[DoorMarker]
    windows: List[WindowMarker]
    stairs: List[StaircaseRegion]

    @property
    def total_doors(self) -> int:
        return len(self.doors)

    @property
    def total_windows(self) -> int:
        return len(self.windows)

    @property
    def total_stairs(self) -> int:
        return len(self.stairs)

def detect_doors_and_windows(
    structural_mask: np.ndarray, 
    bgr_image: np.ndarray,
    adjacency_edges: List[Dict[str, Any]] = None
) -> DoorWindowResult:
    """
    Detect doors (arcs), windows (parallel lines across gaps), and stairs (uniform steps).
    Uses the structural mask to limit search space.
    """
    doors = []
    windows = []
    stairs = []
    
    # 1. Topological Door Detection
    # Instead of relying on drawn arcs (HoughCircles), we use the physical 
    # structural gaps identified by the room detector's adjacency graph.
    if adjacency_edges:
        for edge in adjacency_edges:
            if edge.get("connection_type") == "gap" and "center" in edge:
                cx, cy = edge["center"]
                doors.append(DoorMarker(
                    center=(cx, cy),
                    radius=15, # Fixed radius for visual representation of topological node
                    angle_deg=0.0
                ))
            
    # 2. Window & Stair heuristics (Placeholder for now, requires complex parallel line matching)
    # In a full implementation, we would:
    # - Windows: Find pairs of parallel lines crossing wall gaps
    # - Stairs: Find clusters of evenly spaced parallel lines
    
    return DoorWindowResult(doors=doors, windows=windows, stairs=stairs)
