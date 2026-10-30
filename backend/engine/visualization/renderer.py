"""
backend/engine/visualization/renderer.py
Blueprint-X — Milestone 2

Renders all UI views from the canonical Building model.

Views:
  render_walls()       — Wall vectors overlaid on dimmed blueprint
  render_rooms()       — Segmentation mask + wall lines + room labels + areas
  render_overlay()     — Original image + translucent room fills + walls + labels
                         with proper door arc drawing for gap-detected doors
  render_topology()    — Abstract room graph on dark background
  render_semantics()   — Full semantic annotation: labels, door markers,
                         window markers, adjacency lines, legend panel, area labels
"""
from __future__ import annotations

import cv2
import math
import numpy as np
from typing import Dict, List, Optional, Tuple

from backend.models.dataclasses import Building, Room


# ── Palette helpers ───────────────────────────────────────────────────────────

_OUTER_WALL   = (0, 210, 255)   # bright cyan
_INNER_WALL   = (0, 130, 255)   # blue
_DOOR_COLOR   = (0, 210, 80)    # green
_WINDOW_COLOR = (0, 200, 255)   # sky blue
_FURNITURE_COLOR = (220, 140, 60) # warm amber
_NODE_BORDER  = (255, 255, 255)
_EDGE_COLOR   = (120, 180, 255)


def _put_text_centered(
    img: np.ndarray,
    text: str,
    cx: int,
    cy: int,
    font=cv2.FONT_HERSHEY_DUPLEX,
    scale: float = 0.52,
    color: Tuple = (255, 255, 255),
    thickness: int = 1,
    shadow: bool = True,
) -> None:
    """Draw text centred at (cx, cy) with optional drop shadow."""
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    tx, ty = int(cx) - tw // 2, int(cy) + th // 2
    if shadow:
        cv2.putText(img, text, (tx + 1, ty + 1), font, scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(img, text, (tx, ty), font, scale, color, thickness, cv2.LINE_AA)


def _draw_walls(canvas: np.ndarray, building: Building, outer_thickness: int = 4, inner_thickness: int = 2) -> None:
    for wall in building.walls:
        color = _OUTER_WALL if wall.wall_type == "outer" else _INNER_WALL
        thick = outer_thickness if wall.wall_type == "outer" else inner_thickness
        p1 = (int(wall.start[0]), int(wall.start[1]))
        p2 = (int(wall.end[0]), int(wall.end[1]))
        cv2.line(canvas, p1, p2, color, thick, cv2.LINE_AA)


def _draw_door_markers(canvas: np.ndarray, building: Building) -> None:
    for door in building.doors:
        if door.source == "gap":
            cx, cy = int(door.center[0]), int(door.center[1])
            r = max(4, int(door.width_px // 2))
            cv2.ellipse(canvas, (cx, cy), (r, r), 0, 0, -90, _DOOR_COLOR, 2, cv2.LINE_AA)
            cv2.line(canvas, (cx, cy), (cx + r, cy), _DOOR_COLOR, 2, cv2.LINE_AA)
            cv2.line(canvas, (cx, cy), (cx, cy - r), _DOOR_COLOR, 1, cv2.LINE_AA)
        elif door.bbox:
            x, y, w, h = door.bbox
            cv2.rectangle(canvas, (x, y), (x + w, y + h), _DOOR_COLOR, 2, cv2.LINE_AA)
            _put_text_centered(canvas, "D", x + w // 2, y + h // 2, scale=0.4, color=_DOOR_COLOR)


def _draw_window_markers(canvas: np.ndarray, building: Building) -> None:
    for win in building.windows:
        if win.bbox:
            x, y, w, h = win.bbox
            cv2.rectangle(canvas, (x, y), (x + w, y + h), _WINDOW_COLOR, 2, cv2.LINE_AA)
            _put_text_centered(canvas, "W", x + w // 2, y + h // 2, scale=0.4, color=_WINDOW_COLOR)


def _draw_furniture_markers(canvas: np.ndarray, building: Building) -> None:
    for f in building.furniture:
        if f.bbox:
            x, y, w, h = f.bbox
            cv2.rectangle(canvas, (x, y), (x + w, y + h), _FURNITURE_COLOR, 1, cv2.LINE_AA)


def _area_label(room: Room) -> str:
    """Return 'X.X m²' if area is known, else empty string."""
    if room.area_m2 and room.area_m2 > 0:
        return f"{room.area_m2:.1f} m\u00b2"
    return ""


class Renderer:
    """Provides rendering functions for the UI views based on the Building model."""

    def __init__(
        self,
        original_bgr: np.ndarray,
        building: Building,
        wall_color_bgr: Optional[Tuple[int, int, int]] = None,
        wall_thickness_scale: float = 1.0,
        include_furniture: bool = True,
    ):
        self.bgr = original_bgr.copy()
        self.building = building
        self.wall_color_bgr = wall_color_bgr
        self.wall_thickness_scale = max(0.2, float(wall_thickness_scale))
        self.include_furniture = include_furniture

    def _draw_walls(self, canvas: np.ndarray, outer_thickness: int = 4, inner_thickness: int = 2) -> None:
        outer_t = max(1, int(round(outer_thickness * self.wall_thickness_scale)))
        inner_t = max(1, int(round(inner_thickness * self.wall_thickness_scale)))
        for wall in self.building.walls:
            if self.wall_color_bgr is not None:
                color = self.wall_color_bgr if wall.wall_type == "outer" else tuple(max(0, int(c * 0.82)) for c in self.wall_color_bgr)
            else:
                color = _OUTER_WALL if wall.wall_type == "outer" else _INNER_WALL
            thick = outer_t if wall.wall_type == "outer" else inner_t
            p1 = (int(wall.start[0]), int(wall.start[1]))
            p2 = (int(wall.end[0]), int(wall.end[1]))
            cv2.line(canvas, p1, p2, color, thick, cv2.LINE_AA)

    def _draw_furniture_markers(self, canvas: np.ndarray) -> None:
        if not self.include_furniture:
            return
        for f in self.building.furniture:
            if f.bbox:
                x, y, w, h = f.bbox
                cv2.rectangle(canvas, (x, y), (x + w, y + h), _FURNITURE_COLOR, 1, cv2.LINE_AA)

    # ── Wall view ─────────────────────────────────────────────────────────────

    def render_walls(self) -> np.ndarray:
        """Dimmed blueprint with detected wall vectors and door gap markers."""
        canvas = cv2.addWeighted(self.bgr, 0.38, np.zeros_like(self.bgr), 0.62, 0)
        self._draw_walls(canvas, outer_thickness=4, inner_thickness=2)

        # Endpoint dots
        for wall in self.building.walls:
            cv2.circle(canvas, (int(wall.start[0]), int(wall.start[1])), 4, (0, 255, 180), -1)
            cv2.circle(canvas, (int(wall.end[0]),   int(wall.end[1])),   4, (0, 255, 180), -1)

        _draw_door_markers(canvas, self.building)
        _draw_window_markers(canvas, self.building)
        return canvas

    # ── Room segmentation view ────────────────────────────────────────────────

    def render_rooms(self, segmentation_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Coloured room polygons with wall lines and room labels (+ area) overlaid."""
        h, w = self.bgr.shape[:2]

        if segmentation_mask is not None and segmentation_mask.shape[:2] == (h, w):
            canvas = segmentation_mask.copy()
        else:
            # Rebuild from building rooms if no mask supplied
            canvas = np.zeros((h, w, 3), dtype=np.uint8)
            for room in self.building.rooms:
                cv2.fillPoly(canvas, [room.polygon_np], room.color_bgr)

        # Draw wall lines on top so room boundaries are sharp
        self._draw_walls(canvas, outer_thickness=3, inner_thickness=2)

        # Room labels (+ area below)
        for room in self.building.rooms:
            cx, cy = room.centroid
            label = room.label if room.label else f"R{room.room_id}"
            _put_text_centered(canvas, label, int(cx), int(cy), scale=0.48, shadow=True)
            area_str = _area_label(room)
            if area_str:
                _put_text_centered(canvas, area_str, int(cx), int(cy) + 16,
                                   scale=0.36, color=(220, 230, 255), shadow=True)

        return canvas

    # ── Overlay view ──────────────────────────────────────────────────────────

    def render_overlay(self, alpha: float = 0.65) -> np.ndarray:
        """Original blueprint + translucent room fills + wall vectors + labels."""
        overlay = self.bgr.copy()
        out     = self.bgr.copy()

        for room in self.building.rooms:
            cv2.fillPoly(overlay, [room.polygon_np], room.color_bgr)

        cv2.addWeighted(overlay, alpha * 0.5, out, 1.0 - alpha * 0.5, 0, out)

        self._draw_walls(out, outer_thickness=3, inner_thickness=2)
        _draw_door_markers(out, self.building)

        for room in self.building.rooms:
            cx, cy = room.centroid
            label = room.label if room.label else f"R{room.room_id}"
            _put_text_centered(out, label, int(cx), int(cy), scale=0.50, shadow=True)

        return out

    # ── Topology view ─────────────────────────────────────────────────────────

    def render_topology(self) -> np.ndarray:
        """Abstract room adjacency graph on a dark background."""
        canvas = cv2.addWeighted(self.bgr, 0.20, np.zeros_like(self.bgr), 0.80, 0)
        nodes = {n.room_id: n for n in self.building.topology_nodes}

        # Edges
        for edge in self.building.topology_edges:
            if edge.source_id not in nodes or edge.target_id not in nodes:
                continue
            pt1 = (int(nodes[edge.source_id].centroid[0]), int(nodes[edge.source_id].centroid[1]))
            pt2 = (int(nodes[edge.target_id].centroid[0]), int(nodes[edge.target_id].centroid[1]))

            if edge.edge_type == "door" and edge.door_center:
                dc = (int(edge.door_center[0]), int(edge.door_center[1]))
                cv2.line(canvas, pt1, dc, _DOOR_COLOR, 3, cv2.LINE_AA)
                cv2.line(canvas, dc, pt2, _DOOR_COLOR, 3, cv2.LINE_AA)
                cv2.circle(canvas, dc, 6, (0, 255, 100), -1, cv2.LINE_AA)
            else:
                cv2.line(canvas, pt1, pt2, _EDGE_COLOR, 2, cv2.LINE_AA)

        # Nodes
        for node in self.building.topology_nodes:
            cx, cy = int(node.centroid[0]), int(node.centroid[1])
            room = next((r for r in self.building.rooms if r.room_id == node.room_id), None)
            color = room.color_bgr if room else (128, 128, 128)
            cv2.circle(canvas, (cx, cy), 16, color,        -1,  cv2.LINE_AA)
            cv2.circle(canvas, (cx, cy), 16, _NODE_BORDER,  2,  cv2.LINE_AA)

            lbl = room.label[:8] if (room and room.label) else f"R{node.room_id}"
            _put_text_centered(canvas, lbl, cx, cy, scale=0.40, shadow=False)

        return canvas

    # ── Semantics view ────────────────────────────────────────────────────────

    def render_semantics(self) -> np.ndarray:
        """
        Full semantic annotation view.

        Shows:
          • Coloured room polygons (semi-transparent over original)
          • Room labels in larger text with a dark badge behind each
          • Area label (m²) below the badge when available
          • Door markers (green quarter-arc sweeps / rectangles)
          • Window markers (sky-blue rectangles)
          • Adjacency lines between room centroids
          • Right-hand legend strip listing every room name + area
        """
        h, w = self.bgr.shape[:2]
        LEGEND_W = max(200, w // 5)

        # Canvas = original at 55% + coloured fills at 45%
        overlay = self.bgr.copy()
        canvas  = self.bgr.copy()
        for room in self.building.rooms:
            cv2.fillPoly(overlay, [room.polygon_np], room.color_bgr)
        cv2.addWeighted(overlay, 0.45, canvas, 0.55, 0, canvas)

        # Extend canvas width for legend
        legend = np.full((h, LEGEND_W, 3), 18, dtype=np.uint8)  # very dark bg

        # Adjacency lines between centroids (thin)
        nodes = {n.room_id: n for n in self.building.topology_nodes}
        for edge in self.building.topology_edges:
            if edge.source_id not in nodes or edge.target_id not in nodes:
                continue
            pt1 = (int(nodes[edge.source_id].centroid[0]), int(nodes[edge.source_id].centroid[1]))
            pt2 = (int(nodes[edge.target_id].centroid[0]), int(nodes[edge.target_id].centroid[1]))
            cv2.line(canvas, pt1, pt2, (80, 80, 200), 1, cv2.LINE_AA)

        # Wall lines
        self._draw_walls(canvas, outer_thickness=3, inner_thickness=2)

        # Door / window / furniture markers
        _draw_door_markers(canvas, self.building)
        _draw_window_markers(canvas, self.building)
        self._draw_furniture_markers(canvas)

        # Room labels with badge backing
        font_scale = max(0.42, min(0.65, w / 1400.0))
        font_scale_area = max(0.32, font_scale - 0.10)

        for room in self.building.rooms:
            cx, cy = int(room.centroid[0]), int(room.centroid[1])
            label = room.label if room.label else f"Room {room.room_id}"
            area_str = _area_label(room)

            # Compute badge size (wider if area label is shown)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, font_scale, 1)
            badge_h = th + 8
            if area_str:
                (aw, ah), _ = cv2.getTextSize(area_str, cv2.FONT_HERSHEY_SIMPLEX, font_scale_area, 1)
                badge_w = max(tw, aw) + 10
                badge_h = th + ah + 14
            else:
                badge_w = tw + 10

            pad = 5
            bx0 = cx - badge_w // 2
            by0 = cy - badge_h // 2 - pad
            bx1 = cx + badge_w // 2
            by1 = cy + badge_h // 2 + pad

            # Dark badge
            cv2.rectangle(canvas, (bx0, by0), (bx1, by1), (10, 10, 10), -1)
            cv2.rectangle(canvas, (bx0, by0), (bx1, by1), room.color_bgr, 1)

            # Room label
            label_cy = cy if not area_str else cy - ah // 2 - 3
            _put_text_centered(canvas, label, cx, label_cy,
                               scale=font_scale, shadow=False, color=(235, 235, 235))

            # Area label below
            if area_str:
                _put_text_centered(
                    canvas, area_str, cx, label_cy + th + 6,
                    font=cv2.FONT_HERSHEY_SIMPLEX,
                    scale=font_scale_area, shadow=False,
                    color=(180, 210, 255),
                )

        # ── Legend strip ──────────────────────────────────────────────────────
        cv2.putText(legend, "ROOMS", (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (160, 160, 160), 1, cv2.LINE_AA)
        cv2.line(legend, (10, 30), (LEGEND_W - 10, 30), (50, 50, 80), 1)

        y_cur = 50
        total_area = sum(r.area_m2 for r in self.building.rooms if r.area_m2 > 0)

        for room in self.building.rooms:
            if y_cur + 30 > h:
                break
            c = room.color_bgr
            cv2.rectangle(legend, (10, y_cur - 9), (24, y_cur + 5), c, -1)
            cv2.rectangle(legend, (10, y_cur - 9), (24, y_cur + 5), (80, 80, 80), 1)

            lbl = room.label if room.label else f"Room {room.room_id}"
            lbl_trunc = lbl[:20]
            cv2.putText(legend, lbl_trunc, (30, y_cur + 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34, (200, 200, 200), 1, cv2.LINE_AA)

            # Area sub-label
            area_str = _area_label(room)
            if area_str:
                cv2.putText(legend, area_str, (30, y_cur + 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.30, (130, 160, 220), 1, cv2.LINE_AA)
                y_cur += 30
            else:
                y_cur += 22

        # Total area footer
        if total_area > 0 and y_cur + 30 < h:
            cv2.line(legend, (10, y_cur), (LEGEND_W - 10, y_cur), (50, 50, 80), 1)
            y_cur += 14
            total_str = f"Total: {total_area:.1f} m\u00b2"
            cv2.putText(legend, total_str, (10, y_cur),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, (180, 210, 255), 1, cv2.LINE_AA)

        # Separator line before legend
        cv2.line(legend, (0, 0), (0, h), (50, 50, 80), 2)

        # Merge legend onto right side of canvas
        combined = np.concatenate([canvas, legend], axis=1)
        return combined
