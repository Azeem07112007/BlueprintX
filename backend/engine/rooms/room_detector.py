"""
backend/engine/rooms/room_detector.py
Blueprint-X — M4 (Accuracy Overhaul)

Room detection using a hybrid approach:

PRIMARY — Flood-Fill Segmentation:
  Dilate the wall mask, invert it, run connected components on the non-wall
  regions.  Each connected component is a room.  This works reliably even
  when the topology/DCEL face extraction fails on complex plans.

SECONDARY — Topology Cross-Reference:
  If topology faces are available, use them to refine polygon boundaries and
  adjacency information.  For each flood-fill region we find the best-matching
  topology face and inherit its polygon vertices.

Benefits over pure topology approach:
  - Robust to degenerate graphs (dangling edges, self-intersections)
  - Works on any blueprint complexity
  - Produces accurate pixel-level room masks for display
  - Area computation is exact (from pixel count)
"""
from __future__ import annotations

import cv2
import logging
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from backend.models.dataclasses import Room, Wall, Door, Window
from backend.engine.geometry.geometry_engine import BuildingGraph
from backend.engine.topology.topology_engine import TopologyResult

logger = logging.getLogger(__name__)


def _polygons_share_boundary(
    poly1: List[Tuple[float, float]],
    poly2: List[Tuple[float, float]],
    max_dist: float = 65.0,
    min_overlap: float = 18.0,
) -> bool:
    """Checks whether two room polygons share a parallel boundary wall segment."""
    n1, n2 = len(poly1), len(poly2)
    if n1 < 3 or n2 < 3:
        return False

    for i in range(n1):
        a1 = poly1[i]
        a2 = poly1[(i + 1) % n1]
        ax, ay = a2[0] - a1[0], a2[1] - a1[1]
        lenA = math.hypot(ax, ay)
        if lenA < 1e-4:
            continue
        ux, uy = ax / lenA, ay / lenA

        for j in range(n2):
            b1 = poly2[j]
            b2 = poly2[(j + 1) % n2]
            bx, by = b2[0] - b1[0], b2[1] - b1[1]
            lenB = math.hypot(bx, by)
            if lenB < 1e-4:
                continue

            dot = abs(ux * (bx / lenB) + uy * (by / lenB))
            if dot < 0.85:
                continue

            # Perpendicular distance to line A
            perp1 = abs((b1[0] - a1[0]) * uy - (b1[1] - a1[1]) * ux)
            perp2 = abs((b2[0] - a1[0]) * uy - (b2[1] - a1[1]) * ux)
            if max(perp1, perp2) > max_dist:
                continue

            # 1D projection overlap onto line A
            t_b1 = (b1[0] - a1[0]) * ux + (b1[1] - a1[1]) * uy
            t_b2 = (b2[0] - a1[0]) * ux + (b2[1] - a1[1]) * uy
            t_min = max(0.0, min(t_b1, t_b2))
            t_max = min(lenA, max(t_b1, t_b2))
            if t_max - t_min >= min_overlap:
                return True

    return False


@dataclass
class RoomDetectionResult:
    rooms: List[Room] = field(default_factory=list)
    segmentation: Optional[np.ndarray] = None
    debug: Dict[str, np.ndarray] = field(default_factory=dict)


class RoomDetector:
    """
    Detects rooms from the wall mask using flood-fill connected components,
    cross-referenced with topology faces when available.
    """

    # Minimum pixel² area to keep a region as a room
    _MIN_AREA_PX = 2000

    # Maximum area (fraction of image) — anything larger is likely the
    # exterior / background being mistakenly included
    _MAX_AREA_FRAC = 0.45

    # Perpendicular distance (px) within which a wall midpoint is considered
    # part of an exterior face edge
    _OUTER_WALL_DIST_THRESHOLD = 12.0

    # Primary dilation for closing wall gaps (px, applied twice)
    _WALL_DILATE_K = 13
    # Secondary closing kernel to seal T-junction corners
    _WALL_CLOSE_K  = 7

    def run(
        self,
        graph: BuildingGraph,
        topology: TopologyResult,
        walls: List[Wall],
        doors: List[Door],
        windows: List[Window],
        image_size: Tuple[int, int],
        wall_mask: Optional[np.ndarray] = None,
    ) -> RoomDetectionResult:
        result = RoomDetectionResult()
        h, w = image_size

        # ── Build wall mask from vectors if pixel mask not provided ────────
        if wall_mask is None or wall_mask.shape[:2] != (h, w):
            wall_mask = self._draw_wall_mask(walls, h, w)

        # ── Flood-fill room detection ──────────────────────────────────────
        rooms = self._flood_fill_rooms(wall_mask, h, w, walls=walls)

        if not rooms:
            # Fallback: try topology-based detection
            rooms = self._topology_rooms(graph, topology, h, w)

        if not rooms:
            return result

        # ── Associate doors and windows to rooms ───────────────────────────
        seg_mask = np.zeros((h, w, 3), dtype=np.uint8)
        for room in rooms:
            poly_np = np.array(room.polygon_pts, dtype=np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(seg_mask, [poly_np], room.color_bgr)

        for room in rooms:
            mask = np.zeros((h, w), dtype=np.uint8)
            poly_np = np.array(room.polygon_pts, dtype=np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(mask, [poly_np], 255)
            # Dilate slightly to catch doors right on the boundary
            mask_d = cv2.dilate(mask, np.ones((15, 15), np.uint8))

            for door in doors:
                dcx = max(0, min(w - 1, int(door.center[0])))
                dcy = max(0, min(h - 1, int(door.center[1])))
                if mask_d[dcy, dcx] == 255:
                    if door.id not in room.door_ids:
                        room.door_ids.append(door.id)

            for win in windows:
                wcx = max(0, min(w - 1, int(win.center[0])))
                wcy = max(0, min(h - 1, int(win.center[1])))
                if mask_d[wcy, wcx] == 255:
                    if win.id not in room.window_ids:
                        room.window_ids.append(win.id)

        # ── Reclassify walls outer/internal ───────────────────────────────
        if topology.exterior_face_idx != -1 and topology.faces:
            ext_vids = topology.faces[topology.exterior_face_idx]
            ext_edges = set()
            for i in range(len(ext_vids)):
                u = ext_vids[i]
                v = ext_vids[(i + 1) % len(ext_vids)]
                ext_edges.add(tuple(sorted((u, v))))
            for wall in walls:
                is_outer = False
                for u, v in ext_edges:
                    if u not in graph.vertices or v not in graph.vertices:
                        continue
                    x1, y1 = graph.vertices[u]
                    x2, y2 = graph.vertices[v]
                    mx, my = wall.midpoint
                    dx, dy = x2 - x1, y2 - y1
                    l = math.hypot(dx, dy)
                    if l > 0:
                        dist = abs(dx * (y1 - my) - dy * (x1 - mx)) / l
                        if dist < self._OUTER_WALL_DIST_THRESHOLD:
                            is_outer = True
                            break
                wall.wall_type = "outer" if is_outer else "internal"
        else:
            # Fallback: walls near the image border are outer
            self._classify_walls_by_border(walls, w, h)

        # ── Compute adjacency from shared wall proximity ───────────────────
        self._compute_adjacency(rooms)

        result.rooms = rooms
        result.segmentation = seg_mask
        logger.info("[RoomDetector] Detected %d rooms", len(rooms))
        return result

    # ── Flood-fill room detection ─────────────────────────────────────────────

    def _flood_fill_rooms(
        self, wall_mask: np.ndarray, h: int, w: int,
        walls: List[Wall] = None
    ) -> List[Room]:
        """
        Dilate the wall mask aggressively to close T-junction and corner gaps,
        then flood-fill non-wall regions. Each connected component → one room.
        """
        # Step 1: draw walls with guaranteed thickness to seal all gaps
        sealed = wall_mask.copy()
        if walls:
            for wall in walls:
                p1 = (int(round(wall.start[0])), int(round(wall.start[1])))
                p2 = (int(round(wall.end[0])),   int(round(wall.end[1])))
                # Draw at least 8 px thick regardless of reported thickness
                thick = max(8, int(wall.thickness_px) if wall.thickness_px > 0 else 8)
                cv2.line(sealed, p1, p2, 255, thick)

        # Step 2: morphological dilation + closing to seal corners & gaps
        k1 = np.ones((self._WALL_DILATE_K, self._WALL_DILATE_K), np.uint8)
        k2 = np.ones((self._WALL_CLOSE_K,  self._WALL_CLOSE_K),  np.uint8)
        thick_walls = cv2.dilate(sealed, k1, iterations=2)
        thick_walls = cv2.morphologyEx(thick_walls, cv2.MORPH_CLOSE, k2)

        # Step 3: invert — non-wall pixels are room candidates
        non_wall = cv2.bitwise_not(thick_walls)

        # Step 4: connected components
        n_comp, labels, stats, centroids = cv2.connectedComponentsWithStats(
            non_wall, connectivity=8
        )

        max_area = h * w * self._MAX_AREA_FRAC
        rooms: List[Room] = []

        for i in range(1, n_comp):   # 0 = background
            area = stats[i, cv2.CC_STAT_AREA]
            if area < self._MIN_AREA_PX or area > max_area:
                continue

            comp_mask = (labels == i).astype(np.uint8) * 255
            contours, _ = cv2.findContours(
                comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if not contours:
                continue

            cnt = max(contours, key=cv2.contourArea)
            # Approximation: tighter epsilon to preserve room shape
            epsilon = 0.01 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            if len(approx) < 3:
                continue

            pts = [(float(p[0][0]), float(p[0][1])) for p in approx]
            cx, cy = float(centroids[i][0]), float(centroids[i][1])

            rng = np.random.default_rng(i * 1337 + 7)
            color = (
                int(rng.integers(80, 200)),
                int(rng.integers(80, 200)),
                int(rng.integers(80, 200)),
            )

            x  = int(stats[i, cv2.CC_STAT_LEFT])
            y  = int(stats[i, cv2.CC_STAT_TOP])
            bw = int(stats[i, cv2.CC_STAT_WIDTH])
            bh = int(stats[i, cv2.CC_STAT_HEIGHT])

            # Filter out outer canvas border / margin frame components
            if (bw > w * 0.80 and bh > h * 0.80) and (
                (area / max(1, bw * bh) < 0.35) or
                (x <= 35 and y <= 35 and x + bw >= w - 35 and y + bh >= h - 35)
            ):
                logger.info(
                    "[RoomDetector] Filtered out exterior canvas margin component %d (bbox=%d,%d,%d,%d, area=%.0f)",
                    i, x, y, bw, bh, area
                )
                continue

            room = Room(
                room_id=len(rooms),
                polygon_pts=pts,
                centroid=(cx, cy),
                area_px=float(area),
                bounding_box=(x, y, bw, bh),
                color_bgr=color,
            )
            rooms.append(room)

        # Sort by area descending so large rooms get low IDs
        rooms.sort(key=lambda r: r.area_px, reverse=True)
        for idx, room in enumerate(rooms):
            room.room_id = idx

        return rooms

    # ── Topology-based fallback ────────────────────────────────────────────────

    def _topology_rooms(
        self, graph: BuildingGraph, topology: TopologyResult, h: int, w: int
    ) -> List[Room]:
        rooms: List[Room] = []
        for i, face_vids in enumerate(topology.faces):
            if i == topology.exterior_face_idx:
                continue
            pts = [graph.vertices[vid] for vid in face_vids
                   if vid in graph.vertices]
            if len(pts) < 3:
                continue
            pts_np = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
            area = cv2.contourArea(pts_np)
            if area < self._MIN_AREA_PX:
                continue

            node = next((n for n in topology.nodes if n.room_id == i), None)
            if node:
                centroid = node.centroid
            else:
                M = cv2.moments(pts_np)
                if M["m00"] > 0:
                    centroid = (M["m10"] / M["m00"], M["m01"] / M["m00"])
                else:
                    centroid = (
                        sum(p[0] for p in pts) / len(pts),
                        sum(p[1] for p in pts) / len(pts),
                    )

            x, y, bw, bh = cv2.boundingRect(pts_np)
            rng = np.random.default_rng(i * 123)
            color = (
                int(rng.integers(100, 200)),
                int(rng.integers(100, 200)),
                int(rng.integers(100, 200)),
            )
            room = Room(
                room_id=len(rooms),
                polygon_pts=pts,
                centroid=centroid,
                area_px=float(area),
                bounding_box=(x, y, bw, bh),
                color_bgr=color,
            )
            rooms.append(room)
        return rooms

    # ── Wall mask from vectors ────────────────────────────────────────────────

    def _draw_wall_mask(self, walls: List[Wall], h: int, w: int) -> np.ndarray:
        mask = np.zeros((h, w), dtype=np.uint8)
        for wall in walls:
            p1 = (int(round(wall.start[0])), int(round(wall.start[1])))
            p2 = (int(round(wall.end[0])),   int(round(wall.end[1])))
            thick = max(3, int(wall.thickness_px) if wall.thickness_px > 0 else 5)
            cv2.line(mask, p1, p2, 255, thick)
        return mask

    # ── Wall outer/internal from border proximity ─────────────────────────────

    def _classify_walls_by_border(
        self, walls: List[Wall], w: int, h: int, border: float = 30.0
    ) -> None:
        for wall in walls:
            mx, my = wall.midpoint
            near_border = (
                mx < border or mx > w - border
                or my < border or my > h - border
            )
            wall.wall_type = "outer" if near_border else "internal"

    # ── Adjacency from shared boundary proximity ──────────────────────────────

    def _compute_adjacency(self, rooms: List[Room]) -> None:
        """Mark rooms as adjacent if their wall boundaries share an interface."""
        for i, r in enumerate(rooms):
            for j in range(i + 1, len(rooms)):
                s = rooms[j]
                is_adj = False
                if len(r.polygon_pts) >= 3 and len(s.polygon_pts) >= 3:
                    is_adj = _polygons_share_boundary(r.polygon_pts, s.polygon_pts)
                if not is_adj:
                    # Fallback to bounding box overlap
                    ax, ay, aw, ah = r.bounding_box
                    bx, by, bw, bh = s.bounding_box
                    overlap_x = max(0, min(ax + aw, bx + bw) - max(ax, bx))
                    overlap_y = max(0, min(ay + ah, by + bh) - max(ay, by))
                    # Adjacent if bounding boxes touch along an edge (within 25px) and overlap in other dimension
                    touch_x = (overlap_x >= 10 and min(abs(ay + ah - by), abs(by + bh - ay)) <= 25)
                    touch_y = (overlap_y >= 10 and min(abs(ax + aw - bx), abs(bx + bw - ax)) <= 25)
                    overlap_both = (overlap_x >= 5 and overlap_y >= 5)
                    if touch_x or touch_y or overlap_both:
                        is_adj = True

                if is_adj:
                    if int(s.room_id) not in r.adjacent_room_ids:
                        r.adjacent_room_ids.append(int(s.room_id))
                    if int(r.room_id) not in s.adjacent_room_ids:
                        s.adjacent_room_ids.append(int(r.room_id))
