"""
processors/door_detection.py
Blueprint-X — Redesign

Stage 3: Door opening detection.

Strategy:
  1. Find gaps in the skeleton wall mask (breaks in wall continuity)
  2. Detect arc shapes (door swing indicators) with HoughCircles on grayscale
  3. Cross-reference: gap + nearby arc = high-confidence door
  4. Gaps alone = medium-confidence door
  5. Record all doors; do NOT modify wall mask (doors are stored separately)

Public API:
    result = detect_doors(wall_result, original_bgr)
    result.doors       — List[DoorOpening]
    result.debug       — dict of debug images
"""

from __future__ import annotations

import cv2
import numpy as np
import math
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class DoorOpening:
    """A detected door opening in the wall mask."""
    center: Tuple[int, int]          # pixel center of the opening
    width_px: int                    # approximate width of opening
    angle_deg: float = 0.0           # wall angle at this opening
    confidence: float = 0.8
    has_arc: bool = False            # whether a swing arc was detected

    def to_dict(self) -> Dict[str, Any]:
        return {
            "center": [int(self.center[0]), int(self.center[1])],
            "width_px": int(self.width_px),
            "angle_deg": round(float(self.angle_deg), 1),
            "confidence": round(float(self.confidence), 3),
            "has_arc": bool(self.has_arc),
        }


@dataclass
class DoorResult:
    """All outputs from the door detection stage."""
    doors: List[DoorOpening] = field(default_factory=list)
    debug: Dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def total_doors(self) -> int:
        return len(self.doors)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _find_arc_centers(gray: np.ndarray) -> List[Tuple[int, int, int]]:
    """
    Detect circular arcs (door swing indicators) using HoughCircles.
    Returns list of (cx, cy, radius).
    Door arcs are typically 20–120px radius in floor plan scale.
    """
    blurred = cv2.GaussianBlur(gray, (7, 7), 2)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=25,
        param1=80,
        param2=25,
        minRadius=15,
        maxRadius=120,
    )
    if circles is None:
        return []
    return [(int(c[0]), int(c[1]), int(c[2])) for c in circles[0]]


def _find_wall_gaps(
    skeleton: np.ndarray,
    wall_mask: np.ndarray,
    min_gap_px: int = 12,
    max_gap_px: int = 80,
) -> List[Tuple[int, int, int]]:
    """
    Find gaps in the skeleton that represent door openings.

    Method:
      - Dilate skeleton then XOR with original → bridge pixels
      - Find contours of bridge regions and filter by width
    Returns list of (cx, cy, gap_width).
    """
    # Dilate to bridge gaps
    bridge_k = cv2.getStructuringElement(cv2.MORPH_RECT, (max_gap_px, max_gap_px))
    dilated = cv2.dilate(skeleton, bridge_k, iterations=1)

    # Subtract original skeleton to get only gap-fill areas
    gap_fill = cv2.subtract(dilated, skeleton)

    # Also intersect with inverted wall mask (actual empty space)
    inv_wall = cv2.bitwise_not(wall_mask)
    gap_in_wall = cv2.bitwise_and(gap_fill, inv_wall)

    contours, _ = cv2.findContours(gap_in_wall, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    gaps: List[Tuple[int, int, int]] = []
    for cnt in contours:
        x, y, gw, gh = cv2.boundingRect(cnt)
        gap_dim = min(gw, gh)  # use shorter dimension as gap width

        if gap_dim < min_gap_px or gap_dim > max_gap_px:
            continue

        # Centroid
        M = cv2.moments(cnt)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = x + gw // 2, y + gh // 2

        gaps.append((cx, cy, gap_dim))

    return gaps


def _match_gaps_to_arcs(
    gaps: List[Tuple[int, int, int]],
    arcs: List[Tuple[int, int, int]],
    proximity: float = 60.0,
) -> List[DoorOpening]:
    """
    Cross-reference gaps with arcs. Gaps near an arc are high-confidence doors.
    Remaining unmatched gaps are medium-confidence doors.
    """
    doors: List[DoorOpening] = []
    used_arcs = set()

    for cx, cy, gap_w in gaps:
        best_arc = None
        best_dist = proximity + 1

        for i, (ax, ay, ar) in enumerate(arcs):
            dist = math.hypot(cx - ax, cy - ay)
            if dist < best_dist:
                best_dist = dist
                best_arc = i

        has_arc = best_arc is not None and best_dist <= proximity
        if has_arc:
            used_arcs.add(best_arc)

        doors.append(DoorOpening(
            center=(cx, cy),
            width_px=gap_w,
            angle_deg=0.0,
            confidence=0.92 if has_arc else 0.65,
            has_arc=has_arc,
        ))

    # Add arc-only detections (arc present but no corresponding gap found)
    for i, (ax, ay, ar) in enumerate(arcs):
        if i not in used_arcs:
            doors.append(DoorOpening(
                center=(ax, ay),
                width_px=ar,
                angle_deg=0.0,
                confidence=0.55,
                has_arc=True,
            ))

    return doors


def _draw_door_debug(original_bgr: np.ndarray, doors: List[DoorOpening]) -> np.ndarray:
    """Draw detected doors on original image."""
    debug = original_bgr.copy()
    for door in doors:
        cx, cy = door.center
        r = max(door.width_px // 2, 8)
        color = (0, 220, 80) if door.confidence >= 0.8 else (0, 165, 255)
        cv2.circle(debug, (cx, cy), r, color, 2, cv2.LINE_AA)
        cv2.drawMarker(debug, (cx, cy), color, cv2.MARKER_CROSS, 12, 2)
        label = f"D {door.confidence:.0%}"
        cv2.putText(debug, label, (cx + r + 3, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
    return debug


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_doors(
    wall_result,          # WallResult from wall_detection
    original_bgr: np.ndarray,
) -> DoorResult:
    """
    Detect door openings in the floor plan.

    Args:
        wall_result:    WallResult with wall_mask and skeleton_mask
        original_bgr:   Original BGR image for arc detection

    Returns:
        DoorResult
    """
    result = DoorResult()
    debug: Dict[str, np.ndarray] = {}

    gray = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY)

    # 1. Find arc shapes in grayscale
    arcs = _find_arc_centers(gray)

    # 2. Find gaps in skeleton
    skeleton = wall_result.skeleton_mask
    wall_mask = wall_result.wall_mask

    if skeleton is None or wall_mask is None:
        result.debug = debug
        return result

    gaps = _find_wall_gaps(skeleton, wall_mask)

    # 3. Cross-reference
    doors = _match_gaps_to_arcs(gaps, arcs)

    # Filter low-confidence detections with no arc if too many gaps
    if len(doors) > 20:
        doors = [d for d in doors if d.confidence >= 0.65]

    result.doors = doors

    # Debug image
    debug["10_door_detection"] = _draw_door_debug(original_bgr, doors)

    result.debug = debug
    return result
