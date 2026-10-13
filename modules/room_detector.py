"""
modules/room_detector.py
Blueprint-X Prototype - Milestone 1

Estimates room count and produces a simplified 2-D floor layout by:
  1. Using the binary wall mask from the wall detector
  2. Finding enclosed contours (candidate room polygons)
  3. Filtering by area to discard noise and the overall building footprint
  4. Generating a colour-coded 2-D reconstruction image for the right panel

Future milestones can extend this with:
  - Door / window detection to refine room connectivity
  - Room labelling (bedroom, kitchen, etc.) via ML or rule-based inference
  - 3-D extrusion of room polygons
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import math


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class RoomContour:
    """
    Represents one detected room polygon.
    Coordinates are (x, y) image-space points.
    """
    polygon: np.ndarray          # shape (N, 1, 2)  – raw cv2 contour
    area_px: float
    bounding_box: Tuple[int, int, int, int]   # x, y, w, h
    centroid: Tuple[int, int]
    room_id: int = 0

    def to_dict(self) -> Dict[str, Any]:
        approx = cv2.approxPolyDP(self.polygon, epsilon=3.0, closed=True)
        points = [[int(p[0]), int(p[1])] for p in approx[:, 0, :]]
        return {
            "room_id": int(self.room_id),
            "centroid": [int(self.centroid[0]), int(self.centroid[1])],
            "area_px": round(float(self.area_px), 1),
            "bounding_box": {
                "x": int(self.bounding_box[0]),
                "y": int(self.bounding_box[1]),
                "width": int(self.bounding_box[2]),
                "height": int(self.bounding_box[3]),
            },
            "polygon_points": points,
        }


@dataclass
class RoomAdjacencyGraph:
    edges: List[Dict[str, str]] = field(default_factory=list)

    def to_list(self) -> List[Dict[str, str]]:
        return self.edges


@dataclass
class RoomDetectionResult:
    """Aggregated output of the room detector."""
    rooms: List[RoomContour] = field(default_factory=list)
    adjacency_graph: RoomAdjacencyGraph = field(default_factory=RoomAdjacencyGraph)
    reconstruction_bgr: Optional[np.ndarray] = None   # right panel image

    @property
    def estimated_room_count(self) -> int:
        return len(self.rooms)


# ---------------------------------------------------------------------------
# Colour palette for room fills
# ---------------------------------------------------------------------------

_ROOM_PALETTE = [
    (70,  130, 180),   # steel blue
    (34,  139,  34),   # forest green
    (218, 112, 214),   # orchid
    (210, 105,  30),   # chocolate
    (72,  209, 204),   # medium turquoise
    (255, 165,   0),   # orange
    (147, 112, 219),   # medium purple
    (220,  20,  60),   # crimson
    (0,   206, 209),   # dark turquoise
    (154, 205,  50),   # yellow green
]


def _get_room_color(idx: int) -> Tuple[int, int, int]:
    """Cycle through the palette; return BGR colour."""
    r, g, b = _ROOM_PALETTE[idx % len(_ROOM_PALETTE)]
    return (b, g, r)   # BGR for OpenCV


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------

def _compute_centroid(contour: np.ndarray) -> Tuple[int, int]:
    M = cv2.moments(contour)
    if M["m00"] == 0:
        x, y, w, h = cv2.boundingRect(contour)
        return (x + w // 2, y + h // 2)
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_rooms(
    binary_wall_mask: np.ndarray,
    original_bgr: np.ndarray,
    min_room_area_fraction: float = 0.003, # Increased to suppress noise
    max_room_area_fraction: float = 0.90,
) -> RoomDetectionResult:
    """
    Detect rooms from a binary structural mask.

    Strategy:
      - Invert the structural mask (enclosed white regions = rooms)
      - Erode slightly to separate nearby rooms
      - Find contours and filter by area
      - Generate a coloured reconstruction overlay

    Args:
        binary_wall_mask:        Output of wall_detector._extract_structural_mask().
        original_bgr:            Original floor plan for reference dimensions.
        min_room_area_fraction:  Rooms smaller than this fraction of total
                                 image area are discarded (noise).
        max_room_area_fraction:  Regions larger than this (the building
                                 exterior) are discarded.

    Returns:
        RoomDetectionResult
    """
    result = RoomDetectionResult()
    h, w = binary_wall_mask.shape[:2]
    total_area = h * w

    # Dilate wall mask to close small gaps, then invert to get "room" pixels
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    thick_walls = cv2.dilate(binary_wall_mask, kernel, iterations=2)
    interior = cv2.bitwise_not(thick_walls)

    # Remove tiny specks
    small_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    interior = cv2.morphologyEx(interior, cv2.MORPH_OPEN, small_kernel, iterations=1)

    # Find contours in the interior (room regions)
    contours, hierarchy = cv2.findContours(
        interior, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )

    min_area = total_area * min_room_area_fraction
    max_area = total_area * max_room_area_fraction

    room_contours: List[RoomContour] = []
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        x, y, cw, ch = cv2.boundingRect(cnt)
        cx, cy = _compute_centroid(cnt)

        room_contours.append(RoomContour(
            polygon=cnt,
            area_px=area,
            bounding_box=(x, y, cw, ch),
            centroid=(cx, cy),
            room_id=len(room_contours) + 1
        ))

    # --- Extract adjacency graph ---
    adjacency_edges = []
    num_rooms = len(room_contours)
    if num_rooms > 1:
        room_masks = []
        # Dilate rooms by ~7px to bridge wall gaps
        adj_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        
        for r in room_contours:
            rmask = np.zeros(binary_wall_mask.shape[:2], dtype=np.uint8)
            cv2.drawContours(rmask, [r.polygon], -1, 255, cv2.FILLED)
            room_masks.append(cv2.dilate(rmask, adj_kernel, iterations=1))
            
        for i in range(num_rooms):
            for j in range(i + 1, num_rooms):
                intersection = cv2.bitwise_and(room_masks[i], room_masks[j])
                if np.count_nonzero(intersection) > 50: # Need solid overlap
                    # Calculate center of the gap (door candidate)
                    M = cv2.moments(intersection)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                    else:
                        cx, cy = 0, 0
                        
                    adjacency_edges.append({
                        "source": f"room_{room_contours[i].room_id}",
                        "target": f"room_{room_contours[j].room_id}",
                        "connection_type": "gap",
                        "center": [cx, cy]
                    })
                    
    result.rooms = room_contours
    result.adjacency_graph = RoomAdjacencyGraph(edges=adjacency_edges)

    # --- Build 2-D reconstruction image ---
    recon = _build_reconstruction(
        binary_wall_mask=binary_wall_mask,
        room_contours=room_contours,
        img_size=(w, h),
        original_bgr=original_bgr,
    )
    result.reconstruction_bgr = recon

    return result


def _build_reconstruction(
    binary_wall_mask: np.ndarray,
    room_contours: List[RoomContour],
    img_size: Tuple[int, int],
    original_bgr: np.ndarray,
) -> np.ndarray:
    """
    Render the simplified 2-D floor plan:
      - Dark background
      - Coloured filled room polygons
      - White wall skeleton overlay
      - Room ID labels at centroids
    """
    w, h = img_size

    # Dark background
    canvas = np.full((h, w, 3), 25, dtype=np.uint8)

    # Fill rooms with semi-transparent colour
    overlay = canvas.copy()
    for room in room_contours:
        color = _get_room_color(room.room_id - 1)
        cv2.drawContours(overlay, [room.polygon], -1, color, thickness=cv2.FILLED)

    cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0, canvas)

    # Draw wall skeleton (white)
    wall_vis = cv2.cvtColor(binary_wall_mask, cv2.COLOR_GRAY2BGR)
    wall_mask_bool = binary_wall_mask > 0
    canvas[wall_mask_bool] = (240, 240, 240)

    # Room contour outlines
    for room in room_contours:
        color = _get_room_color(room.room_id - 1)
        cv2.drawContours(canvas, [room.polygon], -1, color, 2)

    # Room labels
    for room in room_contours:
        cx, cy = room.centroid
        label = f"R{room.room_id}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.55
        thickness = 1

        (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)
        tx = cx - tw // 2
        ty = cy + th // 2

        # Dark shadow
        cv2.putText(canvas, label, (tx + 1, ty + 1), font, scale,
                    (0, 0, 0), thickness + 1, cv2.LINE_AA)
        # Light text
        cv2.putText(canvas, label, (tx, ty), font, scale,
                    (255, 255, 255), thickness, cv2.LINE_AA)

    # Footer label
    footer = f"2D Reconstruction  |  {len(room_contours)} room(s) detected"
    cv2.putText(canvas, footer, (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1, cv2.LINE_AA)

    return canvas
