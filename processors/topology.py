"""
processors/topology.py
Blueprint-X — Redesign

Stage 5: Adjacency graph construction.

Builds a room connectivity graph:
  - Each room → node  {id, centroid, area}
  - Two rooms sharing a wall → edge  {from, to, type="wall"}
  - Two rooms connected through a door → edge  {from, to, type="door", door_center}

Public API:
    result = build_topology(room_result, door_result, wall_result)
    result.nodes     — List[TopologyNode]
    result.edges     — List[TopologyEdge]
    result.debug     — dict of debug images
"""

from __future__ import annotations

import cv2
import numpy as np
import math
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class TopologyNode:
    room_id: int
    centroid: Tuple[int, int]
    area_px: float
    color_bgr: Tuple[int, int, int]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "room_id": int(self.room_id),
            "centroid": [int(self.centroid[0]), int(self.centroid[1])],
            "area_px": round(float(self.area_px), 1),
        }


@dataclass
class TopologyEdge:
    source_id: int
    target_id: int
    edge_type: str = "wall"          # "wall" | "door"
    door_center: Optional[Tuple[int, int]] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "source": f"room_{self.source_id}",
            "target": f"room_{self.target_id}",
            "type": self.edge_type,
        }
        if self.door_center:
            d["door_center"] = [int(self.door_center[0]), int(self.door_center[1])]
        return d


@dataclass
class TopologyResult:
    nodes: List[TopologyNode] = field(default_factory=list)
    edges: List[TopologyEdge] = field(default_factory=list)
    debug: Dict[str, np.ndarray] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _rooms_share_wall(
    room_a,
    room_b,
    wall_mask: np.ndarray,
    dilation_px: int = 10,
) -> bool:
    """
    Check if two rooms share a wall by dilating both room masks
    and checking if they overlap at wall pixels.
    """
    h, w = wall_mask.shape[:2]
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (dilation_px, dilation_px))

    mask_a = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask_a, [room_a.polygon], -1, 255, cv2.FILLED)
    mask_a_d = cv2.dilate(mask_a, k, iterations=1)

    mask_b = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask_b, [room_b.polygon], -1, 255, cv2.FILLED)
    mask_b_d = cv2.dilate(mask_b, k, iterations=1)

    # Overlap in dilated regions AND at wall pixels
    both_dilated = cv2.bitwise_and(mask_a_d, mask_b_d)
    wall_overlap = cv2.bitwise_and(both_dilated, wall_mask)

    return cv2.countNonZero(wall_overlap) > 20


def _find_door_between(
    room_a,
    room_b,
    doors,
    proximity: float = 60.0,
) -> Optional[Tuple[int, int]]:
    """
    Check if a detected door connects two rooms.
    Returns the door center if found, else None.
    """
    cx_a, cy_a = room_a.centroid
    cx_b, cy_b = room_b.centroid
    mid_x = (cx_a + cx_b) // 2
    mid_y = (cy_a + cy_b) // 2

    for door in doors:
        dcx, dcy = door.center
        dist = math.hypot(dcx - mid_x, dcy - mid_y)
        if dist < proximity:
            return door.center

    return None


def _draw_topology(
    original_bgr: np.ndarray,
    nodes: List[TopologyNode],
    edges: List[TopologyEdge],
) -> np.ndarray:
    """Render the topology graph on a dimmed copy of the original."""
    canvas = cv2.addWeighted(original_bgr, 0.25, np.zeros_like(original_bgr), 0.75, 0)

    node_map: Dict[int, TopologyNode] = {n.room_id: n for n in nodes}

    # Draw edges (paths through doors)
    for edge in edges:
        if edge.source_id not in node_map or edge.target_id not in node_map:
            continue
        pt1 = node_map[edge.source_id].centroid
        pt2 = node_map[edge.target_id].centroid
        
        if edge.edge_type == "door" and edge.door_center:
            color = (0, 220, 80)
            cv2.line(canvas, pt1, edge.door_center, color, 3, cv2.LINE_AA)
            cv2.line(canvas, edge.door_center, pt2, color, 3, cv2.LINE_AA)
            cv2.circle(canvas, edge.door_center, 6, (0, 255, 100), -1, cv2.LINE_AA)

    # Draw nodes
    for node in nodes:
        cx, cy = node.centroid
        cv2.circle(canvas, (cx, cy), 14, node.color_bgr, -1, cv2.LINE_AA)
        cv2.circle(canvas, (cx, cy), 14, (255, 255, 255), 2, cv2.LINE_AA)
        label = f"R{node.room_id}"
        font = cv2.FONT_HERSHEY_DUPLEX
        (tw, th), _ = cv2.getTextSize(label, font, 0.45, 1)
        cv2.putText(canvas, label, (cx - tw // 2, cy + th // 2),
                    font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    return canvas


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_topology(
    room_result,     # RoomResult
    door_result,     # DoorResult
    wall_result,     # WallResult
    original_bgr: np.ndarray,
) -> TopologyResult:
    """
    Build the room adjacency topology graph.

    Args:
        room_result:   RoomResult from room_detection
        door_result:   DoorResult from door_detection
        wall_result:   WallResult from wall_detection
        original_bgr:  Original image for debug rendering

    Returns:
        TopologyResult
    """
    result = TopologyResult()
    debug: Dict[str, np.ndarray] = {}

    rooms = room_result.rooms
    doors = door_result.doors if door_result else []
    wall_mask = wall_result.wall_mask if wall_result else None

    # Build edges
    edges: List[TopologyEdge] = []
    connected_room_ids = set()
    n = len(rooms)

    for i in range(n):
        for j in range(i + 1, n):
            ra, rb = rooms[i], rooms[j]

            # We only care about rooms connected by a door to form a path graph
            door_center = _find_door_between(ra, rb, doors)

            if door_center:
                edges.append(TopologyEdge(
                    source_id=ra.room_id,
                    target_id=rb.room_id,
                    edge_type="door",
                    door_center=door_center,
                ))
                connected_room_ids.add(ra.room_id)
                connected_room_ids.add(rb.room_id)

    result.edges = edges

    # Build nodes (only keep rooms that are part of the path graph)
    nodes = [
        TopologyNode(
            room_id=r.room_id,
            centroid=r.centroid,
            area_px=r.area_px,
            color_bgr=r.color_bgr,
        )
        for r in rooms if r.room_id in connected_room_ids
    ]
    result.nodes = nodes

    # Debug visualization
    debug["14_topology"] = _draw_topology(original_bgr, nodes, edges)
    result.debug = debug

    return result
