"""
processors/visualization.py — Blueprint-X Redesign v2

Renders all UI tab views from the PipelineResult.
All wall rendering now uses vector segments, not raster blobs.

Debug key order follows the new pipeline stages.
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import Dict, Optional, List

from processors.wall_detection import WallSegment


ALL_DEBUG_KEYS = [
    "01_grayscale",
    "02_binary_raw",
    "03_no_tiny_blobs",
    "04_after_open",
    "05_after_close",
    "06_cleaned_binary",
    "07_skeleton",
    "08_hough_raw",
    "09_wall_segments",
    "11_planar_graph",
    "12_all_faces",
    "13_room_segmentation",
    "14_topology",
]

DEBUG_LABELS = {
    "01_grayscale":           "Grayscale",
    "02_binary_raw":          "Binary Threshold",
    "03_no_tiny_blobs":       "Tiny Blobs Removed",
    "04_after_open":          "After Open (Text Removed)",
    "05_after_close":         "After Close",
    "06_cleaned_binary":      "Cleaned Binary (Structural)",
    "07_skeleton":            "Skeleton",
    "08_hough_raw":           "Hough Raw Segments",
    "09_wall_segments":       "Wall Vectors",
    "11_planar_graph":        "Planar Graph",
    "12_all_faces":           "All Graph Faces",
    "13_room_segmentation":   "Room Segmentation",
    "14_topology":            "Topology Graph",
}


def render_original(result) -> np.ndarray:
    return result.original_bgr.copy()


def render_walls(result) -> np.ndarray:
    """
    Vector wall view:
      Outer walls in gold
      Internal walls in orange
      Door gaps in green
    """
    bgr = result.original_bgr
    canvas = cv2.addWeighted(bgr, 0.40, np.zeros_like(bgr), 0.60, 0)

    segments: List[WallSegment] = (
        result.wall_result.wall_segments if result.wall_result else []
    )
    for seg in segments:
        color   = (0, 210, 255) if seg.wall_type == "outer" else (0, 130, 255)
        thickness = 4 if seg.wall_type == "outer" else 2
        p1 = (int(seg.start[0]), int(seg.start[1]))
        p2 = (int(seg.end[0]),   int(seg.end[1]))
        cv2.line(canvas, p1, p2, color, thickness, cv2.LINE_AA)
        cv2.circle(canvas, p1, 4, (0, 255, 180), -1)
        cv2.circle(canvas, p2, 4, (0, 255, 180), -1)

    door_gaps = result.wall_result.door_gaps if result.wall_result else []
    for gap in door_gaps:
        cx, cy = int(gap.center[0]), int(gap.center[1])
        cv2.circle(canvas, (cx, cy), int(gap.width_px // 2), (0, 220, 80), 2, cv2.LINE_AA)
        cv2.drawMarker(canvas, (cx, cy), (0, 220, 80), cv2.MARKER_CROSS, 12, 2)

    _draw_legend(canvas, [
        ((0, 210, 255), "Outer Wall"),
        ((0, 130, 255), "Internal Wall"),
        ((0, 220,  80), "Door Gap"),
    ])
    return canvas


def render_rooms(result) -> np.ndarray:
    if result.room_result and result.room_result.segmentation is not None:
        return result.room_result.segmentation.copy()
    return _placeholder(result.original_bgr, "Run analysis first")


def render_overlay(result, alpha: float = 0.65) -> np.ndarray:
    bgr = result.original_bgr.copy()
    overlay = bgr.copy()

    rooms = result.room_result.rooms if result.room_result else []
    for room in rooms:
        cv2.drawContours(overlay, [room.polygon], -1, room.color_bgr, cv2.FILLED)
    cv2.addWeighted(overlay, alpha * 0.5, bgr, 1.0 - alpha * 0.5, 0, bgr)

    segments = result.wall_result.wall_segments if result.wall_result else []
    for seg in segments:
        color = (0, 210, 255) if seg.wall_type == "outer" else (0, 130, 255)
        p1 = (int(seg.start[0]), int(seg.start[1]))
        p2 = (int(seg.end[0]),   int(seg.end[1]))
        cv2.line(bgr, p1, p2, color, 3 if seg.wall_type == "outer" else 2, cv2.LINE_AA)

    for room in rooms:
        cx, cy = room.centroid
        label = f"R{room.room_id}"
        cv2.putText(bgr, label, (cx - 10, cy + 5),
                    cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return bgr


def render_topology(result) -> np.ndarray:
    if result.topology_result and "14_topology" in result.topology_result.debug:
        return result.topology_result.debug["14_topology"].copy()
    return _placeholder(result.original_bgr, "No topology data")


def render_debug_stage(result, key: str) -> np.ndarray:
    for attr in ["preprocess_result", "wall_result", "room_result", "topology_result"]:
        r = getattr(result, attr, None)
        if r is not None and hasattr(r, "debug") and key in r.debug:
            img = r.debug[key]
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            return img.copy()
    return _placeholder(result.original_bgr, f"Stage: {key}")


def _placeholder(ref: np.ndarray, text: str) -> np.ndarray:
    h, w = ref.shape[:2]
    canvas = np.full((h, w, 3), 20, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_DUPLEX
    (tw, th), _ = cv2.getTextSize(text, font, 0.7, 1)
    cv2.putText(canvas, text, (w // 2 - tw // 2, h // 2 + th // 2),
                font, 0.7, (80, 80, 80), 1, cv2.LINE_AA)
    return canvas


def _draw_legend(canvas: np.ndarray, items: list) -> None:
    pad, row_h = 8, 20
    legend_h = pad * 2 + len(items) * row_h
    legend_w = 210
    roi = canvas[pad:pad + legend_h, pad:pad + legend_w]
    bg = np.full_like(roi, 20)
    cv2.addWeighted(bg, 0.6, roi, 0.4, 0, roi)
    canvas[pad:pad + legend_h, pad:pad + legend_w] = roi
    for i, (color, label) in enumerate(items):
        y = pad * 2 + i * row_h
        x = pad * 2
        cv2.line(canvas, (x, y + 6), (x + 24, y + 6), color, 3, cv2.LINE_AA)
        cv2.putText(canvas, label, (x + 30, y + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)
