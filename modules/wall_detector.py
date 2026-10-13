"""
modules/wall_detector.py
Blueprint-X Prototype - Milestone 1

Detects outer walls, internal walls, and produces:
  - An annotated BGR image for the center panel
  - A list of WallSegment objects for JSON export and the 2-D reconstructor

Pipeline:
  1. Grayscale + Gaussian blur  -> reduce noise
  2. Adaptive threshold         -> binarise (walls = dark lines on light background)
  3. Morphological closing      -> connect broken wall pixels
  4. Canny edge detection       -> precise edge map
  5. Hough Line Transform (P)   -> extract line segments
  6. Classify lines as outer / internal based on proximity to image border
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional
import math


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class WallSegment:
    """
    Represents one detected wall segment.
    The coordinate system is (col, row) == (x, y) in image space.
    Future milestones will add thickness, material, and elevation fields.
    """
    start: Tuple[int, int]   # (x1, y1)
    end: Tuple[int, int]     # (x2, y2)
    wall_type: str = "internal"   # "outer" | "internal"
    thickness_px: int = 2
    confidence: float = 1.0

    def length(self) -> float:
        dx = self.end[0] - self.start[0]
        dy = self.end[1] - self.start[1]
        return math.sqrt(dx * dx + dy * dy)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start": [int(self.start[0]), int(self.start[1])],
            "end": [int(self.end[0]), int(self.end[1])],
            "wall_type": self.wall_type,
            "length_px": round(float(self.length()), 2),
            "confidence": round(float(self.confidence), 3),
        }


@dataclass
class WallDetectionResult:
    """Aggregated output of the wall detector."""
    walls: List[WallSegment] = field(default_factory=list)
    annotated_bgr: Optional[np.ndarray] = None        # center panel image
    binary_mask: Optional[np.ndarray] = None          # full binary (walls + furniture)
    structural_mask: Optional[np.ndarray] = None      # thick walls only
    edge_map: Optional[np.ndarray] = None
    scale_factor: float = 1.0

    @property
    def total_wall_segments(self) -> int:
        return len(self.walls)

    @property
    def outer_wall_count(self) -> int:
        return sum(1 for w in self.walls if w.wall_type == "outer")

    @property
    def internal_wall_count(self) -> int:
        return sum(1 for w in self.walls if w.wall_type == "internal")


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------

def _binarise(gray: np.ndarray) -> np.ndarray:
    """
    Convert grayscale floor plan to binary (walls dark, background white).
    Uses adaptive thresholding to handle uneven lighting / scan quality.
    """
    # Gaussian blur to remove small noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive threshold: walls become white (255), background black (0)
    binary = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=15,
        C=4,
    )

    # Morphological closing: connect nearby wall fragments
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    return closed


def _extract_structural_mask(binary: np.ndarray) -> np.ndarray:
    """
    Filter out thin elements (furniture edges, text, dimension lines)
    by eroding the mask with a large kernel, then dilating back to
    recover thick wall structures.
    """
    # Erode to destroy thin lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    eroded = cv2.erode(binary, kernel, iterations=2)
    
    # Dilate back to original thickness
    reconstructed = cv2.dilate(eroded, kernel, iterations=2)
    return reconstructed


def _merge_similar_lines(
    lines: List[Tuple[int, int, int, int]],
    angle_tol: float = 10.0,
    dist_tol: float = 15.0,
) -> List[Tuple[int, int, int, int]]:
    """
    Merge line segments that are nearly parallel and close to each other.
    Reduces duplicate Hough detections from thick walls.
    """
    if not lines:
        return []

    def angle(x1, y1, x2, y2) -> float:
        return math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180

    def midpoint(x1, y1, x2, y2) -> Tuple[float, float]:
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    merged: List[Tuple[int, int, int, int]] = []
    used = [False] * len(lines)

    for i, li in enumerate(lines):
        if used[i]:
            continue
        group = [li]
        ai = angle(*li)
        mi = midpoint(*li)

        for j, lj in enumerate(lines):
            if i == j or used[j]:
                continue
            aj = angle(*lj)
            mj = midpoint(*lj)

            da = abs(ai - aj)
            da = min(da, 180 - da)
            dd = math.hypot(mi[0] - mj[0], mi[1] - mj[1])

            if da < angle_tol and dd < dist_tol:
                group.append(lj)
                used[j] = True

        used[i] = True
        # Keep the longest segment in the group
        best = max(group, key=lambda s: math.hypot(s[2]-s[0], s[3]-s[1]))
        merged.append(best)

    return merged


def _classify_wall(
    x1: int, y1: int, x2: int, y2: int,
    img_w: int, img_h: int,
    border_margin: float = 0.12,
) -> str:
    """
    Classify a wall segment as 'outer' or 'internal'.
    Lines near the image border are considered outer walls.
    """
    margin_x = img_w * border_margin
    margin_y = img_h * border_margin

    near_border = (
        min(x1, x2) < margin_x
        or max(x1, x2) > img_w - margin_x
        or min(y1, y2) < margin_y
        or max(y1, y2) > img_h - margin_y
    )
    return "outer" if near_border else "internal"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_walls(
    bgr_image: np.ndarray,
    min_line_length: int = 60,  # Increased from 40 to ignore furniture edges
    max_line_gap: int = 20,
    hough_threshold: int = 50,
) -> WallDetectionResult:
    """
    Main entry point. Accepts a BGR OpenCV image and returns a
    WallDetectionResult with annotated image and wall segments.

    Args:
        bgr_image:        Input floor plan (BGR, uint8).
        min_line_length:  Minimum Hough line segment length (pixels).
        max_line_gap:     Maximum gap to still be considered one line.
        hough_threshold:  Hough accumulator threshold.

    Returns:
        WallDetectionResult
    """
    result = WallDetectionResult()
    h, w = bgr_image.shape[:2]

    # --- 1. Pre-process ---
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    binary = _binarise(gray)
    result.binary_mask = binary
    
    structural = _extract_structural_mask(binary)
    result.structural_mask = structural

    # --- 2. Edge detection (on structural mask only) ---
    edges = cv2.Canny(structural, threshold1=50, threshold2=150, apertureSize=3)
    # Dilate edges slightly to improve Hough line continuity
    edge_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    edges = cv2.dilate(edges, edge_kernel, iterations=1)
    result.edge_map = edges

    # --- 3. Probabilistic Hough Line Transform ---
    raw_lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=hough_threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )

    raw_list: List[Tuple[int, int, int, int]] = []
    if raw_lines is not None:
        for line in raw_lines:
            x1, y1, x2, y2 = line[0]
            raw_list.append((x1, y1, x2, y2))

    # Cap the number of lines to prevent O(N^2) hang in _merge_similar_lines
    if len(raw_list) > 1500:
        raw_list.sort(key=lambda s: math.hypot(s[2]-s[0], s[3]-s[1]), reverse=True)
        raw_list = raw_list[:1500]

    # --- 4. Merge near-duplicate lines ---
    clean_lines = _merge_similar_lines(raw_list)

    # --- 5. Build WallSegment objects & classify ---
    wall_segments: List[WallSegment] = []
    for (x1, y1, x2, y2) in clean_lines:
        wtype = _classify_wall(x1, y1, x2, y2, w, h)
        seg = WallSegment(
            start=(x1, y1),
            end=(x2, y2),
            wall_type=wtype,
        )
        wall_segments.append(seg)

    result.walls = wall_segments

    # --- 6. Draw annotated image ---
    annotated = bgr_image.copy()

    # Draw outer walls in cyan, internal in orange
    OUTER_COLOR  = (220, 200, 0)    # BGR cyan-ish
    INNER_COLOR  = (0, 140, 255)    # BGR orange

    for seg in wall_segments:
        color = OUTER_COLOR if seg.wall_type == "outer" else INNER_COLOR
        thickness = 3 if seg.wall_type == "outer" else 2
        cv2.line(annotated, seg.start, seg.end, color, thickness, cv2.LINE_AA)

    # Legend
    _draw_legend(annotated)

    result.annotated_bgr = annotated
    return result


def _draw_legend(img: np.ndarray) -> None:
    """Overlay a small legend on the annotated image."""
    h, w = img.shape[:2]
    overlay = img.copy()
    box_x, box_y = 10, 10
    box_w, box_h = 200, 60
    cv2.rectangle(overlay, (box_x, box_y), (box_x + box_w, box_y + box_h), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)

    # Outer wall sample
    cv2.line(img, (box_x + 10, box_y + 18), (box_x + 40, box_y + 18), (220, 200, 0), 3)
    cv2.putText(img, "Outer Wall", (box_x + 48, box_y + 23),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA)

    # Inner wall sample
    cv2.line(img, (box_x + 10, box_y + 42), (box_x + 40, box_y + 42), (0, 140, 255), 2)
    cv2.putText(img, "Internal Wall", (box_x + 48, box_y + 47),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA)
