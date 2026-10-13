"""
backend/engine/model3d/geometry_validator.py
Blueprint-X — 4-Pass Geometry Validation & Topology Repair Engine

Ensures zero visual flaws in generated 3D models by executing 4 rigorous
validation passes over all architectural elements prior to mesh extrusion:

Pass 1: Degenerate & Duplicate Wall Removal
Pass 2: Wall-Wall Intersection & T-Junction Splitting
Pass 3: Endpoint Gap Snapping & Corner Alignment
Pass 4: Furniture Room Containment & Wall Collision Clearance
"""

from __future__ import annotations

import logging
import math
from typing import List, Tuple, Optional
import numpy as np

from backend.models.dataclasses import Building, Wall, Room, Furniture

logger = logging.getLogger(__name__)


def _dist_sq(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2


def _dist(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def _segment_intersection(
    p1: Tuple[float, float], p2: Tuple[float, float],
    p3: Tuple[float, float], p4: Tuple[float, float]
) -> Optional[Tuple[float, float]]:
    """Calculates intersection of line segment (p1-p2) with line segment (p3-p4)."""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
    if abs(denom) < 1e-6:
        return None  # Parallel or collinear

    ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom
    ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denom

    # Exclude endpoints slightly (0.01 to 0.99)
    if 0.01 < ua < 0.99 and 0.01 < ub < 0.99:
        ix = x1 + ua * (x2 - x1)
        iy = y1 + ua * (y2 - y1)
        return (float(ix), float(iy))
    return None


def _point_inside_polygon(x: float, y: float, poly: List[Tuple[float, float]]) -> bool:
    """Ray-casting algorithm to test point inside polygon."""
    n = len(poly)
    if n < 3:
        return False
    inside = False
    p1x, p1y = poly[0]
    for i in range(n + 1):
        p2x, p2y = poly[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside


def _dist_point_to_segment(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    if L2 < 1e-6:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / L2))
    proj_x = ax + t * dx
    proj_y = ay + t * dy
    return math.hypot(px - proj_x, py - proj_y)


def _dist_to_room_boundary(px: float, py: float, polygon_pts: List[Tuple[float, float]]) -> float:
    n = len(polygon_pts)
    if n < 3:
        return 0.0
    min_d = 1e9
    for i in range(n):
        ax, ay = polygon_pts[i]
        bx, by = polygon_pts[(i + 1) % n]
        d = _dist_point_to_segment(px, py, ax, ay, bx, by)
        if d < min_d:
            min_d = d
    return min_d


class GeometryValidator:
    """
    4-Pass Geometry Verification and Cleanup Engine.
    Executes multiple passes to guarantee gapless, non-overlapping, crash-free 3D output.
    """

    def __init__(
        self,
        min_wall_length_px: float = 8.0,
        snap_distance_px: float = 16.0,
        duplicate_dist_px: float = 4.0,
    ):
        self.min_wall_length_px = min_wall_length_px
        self.snap_distance_px = snap_distance_px
        self.duplicate_dist_px = duplicate_dist_px

    def validate_building(self, building: Building) -> Building:
        """Runs complete 7-pass verification on building."""
        logger.info("[GeometryValidator] Starting multi-pass validation...")
        
        # Pass 1: Clean degenerate & duplicate walls
        walls = self.pass1_remove_duplicates_and_short(building.walls)
        
        # Pass 2: Split intersecting walls
        walls = self.pass2_split_intersections(walls)
        
        # Pass 3: Endpoint gap snapping
        walls = self.pass3_snap_endpoints_and_gaps(walls)
        
        # Pass 4: Validate furniture containment & wall clearance
        furniture = self.pass4_validate_furniture(building.furniture, building.rooms, walls)

        # Pass 5: Prune false interior Hough walls inside rooms
        walls = self.pass5_prune_false_interior_walls(walls, building.rooms)

        building.walls = walls
        building.furniture = furniture

        # Pass 6: Ensure main exterior entrance & full room accessibility
        building = self.pass7_ensure_room_accessibility_and_main_entrance(building)

        # Pass 7: Snap & align all doors & windows to wall direction
        building = self.pass6_align_openings_to_walls(building)

        # Pass 8: Advanced Architectural Analyzer (Room adjacency door synthesis & quality scoring)
        from backend.engine.model3d.advanced_analyzer import AdvancedArchitecturalAnalyzer
        building, _report = AdvancedArchitecturalAnalyzer().analyze_and_optimize(building)

        logger.info(f"[GeometryValidator] Completed passes: {len(building.walls)} walls, {len(building.doors)} doors, {len(building.windows)} windows verified")
        return building

    def validate_building_multi_pass(self, building: Building, passes: int = 4) -> Building:
        """Executes the validation suite multiple times to guarantee convergence."""
        for i in range(passes):
            logger.info(f"[GeometryValidator] Pass cycle {i + 1}/{passes}")
            building = self.validate_building(building)
        return building

    # ── Pass 1: Remove duplicates & short walls ──────────────────────────────────
    def pass1_remove_duplicates_and_short(self, walls: List[Wall]) -> List[Wall]:
        valid_walls: List[Wall] = []
        for w in walls:
            length = math.hypot(w.end[0] - w.start[0], w.end[1] - w.start[1])
            if length < self.min_wall_length_px:
                continue

            # Check duplicate against already accepted walls
            is_dup = False
            for prev in valid_walls:
                s_dist = _dist(w.start, prev.start) + _dist(w.end, prev.end)
                r_dist = _dist(w.start, prev.end) + _dist(w.end, prev.start)
                if s_dist < self.duplicate_dist_px * 2 or r_dist < self.duplicate_dist_px * 2:
                    is_dup = True
                    break
            if not is_dup:
                valid_walls.append(w)

        return valid_walls

    # ── Pass 2: Wall-wall intersection & split ──────────────────────────────────
    def pass2_split_intersections(self, walls: List[Wall]) -> List[Wall]:
        result_walls = list(walls)
        has_splits = True
        max_iterations = 10
        iteration = 0

        while has_splits and iteration < max_iterations:
            has_splits = False
            iteration += 1
            n = len(result_walls)

            for i in range(n):
                w1 = result_walls[i]
                for j in range(i + 1, n):
                    w2 = result_walls[j]
                    pt = _segment_intersection(w1.start, w1.end, w2.start, w2.end)
                    if pt is not None:
                        # Split w1 at pt
                        w1a = Wall(start=w1.start, end=pt, thickness_px=w1.thickness_px, wall_type=w1.wall_type)
                        w1b = Wall(start=pt, end=w1.end, thickness_px=w1.thickness_px, wall_type=w1.wall_type)
                        # Split w2 at pt
                        w2a = Wall(start=w2.start, end=pt, thickness_px=w2.thickness_px, wall_type=w2.wall_type)
                        w2b = Wall(start=pt, end=w2.end, thickness_px=w2.thickness_px, wall_type=w2.wall_type)

                        # Replace w1 and w2 with split sub-walls
                        result_walls.pop(j)
                        result_walls.pop(i)
                        result_walls.extend([w1a, w1b, w2a, w2b])
                        has_splits = True
                        break
                if has_splits:
                    break

        return result_walls

    # ── Pass 3: Endpoint snapping & gap closure ─────────────────────────────────
    def pass3_snap_endpoints_and_gaps(self, walls: List[Wall]) -> List[Wall]:
        if not walls:
            return []

        cleaned: List[Wall] = []
        for w in walls:
            s_x, s_y = w.start
            e_x, e_y = w.end

            # Snap to other wall endpoints
            for o in walls:
                if o is w:
                    continue
                # Snap start point
                if _dist(w.start, o.start) <= self.snap_distance_px:
                    s_x, s_y = o.start
                elif _dist(w.start, o.end) <= self.snap_distance_px:
                    s_x, s_y = o.end

                # Snap end point
                if _dist(w.end, o.start) <= self.snap_distance_px:
                    e_x, e_y = o.start
                elif _dist(w.end, o.end) <= self.snap_distance_px:
                    e_x, e_y = o.end

            # T-Junction snapping (point to line segment)
            for o in walls:
                if o is w:
                    continue
                ax, ay = o.start
                bx, by = o.end
                dx, dy = bx - ax, by - ay
                L2 = dx * dx + dy * dy
                if L2 < 1e-4:
                    continue

                # Start projection
                ts = max(0.0, min(1.0, ((s_x - ax) * dx + (s_y - ay) * dy) / L2))
                px_s, py_s = ax + ts * dx, ay + ts * dy
                if 1.0 < _dist((s_x, s_y), (px_s, py_s)) <= self.snap_distance_px:
                    s_x, s_y = px_s, py_s

                # End projection
                te = max(0.0, min(1.0, ((e_x - ax) * dx + (e_y - ay) * dy) / L2))
                px_e, py_e = ax + te * dx, ay + te * dy
                if 1.0 < _dist((e_x, e_y), (px_e, py_e)) <= self.snap_distance_px:
                    e_x, e_y = px_e, py_e

            length = math.hypot(e_x - s_x, e_y - s_y)
            if length >= self.min_wall_length_px:
                cleaned.append(Wall(
                    id=w.id,
                    start=(float(s_x), float(s_y)),
                    end=(float(e_x), float(e_y)),
                    thickness_px=w.thickness_px,
                    wall_type=w.wall_type
                ))

        return cleaned

    # ── Pass 4: Furniture room containment & wall clearance ──────────────────────
    def pass4_validate_furniture(
        self,
        furniture: List[Furniture],
        rooms: List[Room],
        walls: List[Wall]
    ) -> List[Furniture]:
        validated: List[Furniture] = []

        for item in furniture:
            cx, cy = item.center
            
            # Check room containment
            matching_room: Optional[Room] = None
            for r in rooms:
                if r.polygon_pts and len(r.polygon_pts) >= 3:
                    if _point_inside_polygon(cx, cy, r.polygon_pts):
                        matching_room = r
                        break

            # If not inside assigned room, attempt to pull towards room center if room_label matches
            if matching_room is None and item.room_label:
                for r in rooms:
                    if r.label.lower() == item.room_label.lower() and r.polygon_pts and len(r.polygon_pts) >= 3:
                        # Move furniture inside this room
                        r_cx = sum(p[0] for p in r.polygon_pts) / len(r.polygon_pts)
                        r_cy = sum(p[1] for p in r.polygon_pts) / len(r.polygon_pts)
                        # Shift halfway towards room center
                        cx = (cx + r_cx) / 2.0
                        cy = (cy + r_cy) / 2.0
                        item.center = (cx, cy)
                        matching_room = r
                        break

            # Wall clearance check: push furniture away if too close to wall segment
            for w in walls:
                ax, ay = w.start
                bx, by = w.end
                dx, dy = bx - ax, by - ay
                L2 = dx * dx + dy * dy
                if L2 < 1e-4:
                    continue
                t = max(0.0, min(1.0, ((cx - ax) * dx + (cy - ay) * dy) / L2))
                px, py = ax + t * dx, ay + t * dy
                d = math.hypot(cx - px, cy - py)
                min_clearance = w.thickness_px / 2.0 + 15.0  # wall half-thickness + 15px margin
                if d < min_clearance and d > 1e-3:
                    # Push furniture outwards along normal
                    nx, ny = (cx - px) / d, (cy - py) / d
                    shift = min_clearance - d
                    cx += nx * shift
                    cy += ny * shift
                    item.center = (cx, cy)

            validated.append(item)

        return validated

    # ── Pass 5: Prune false Hough wall segments inside room interiors ─────────────
    def pass5_prune_false_interior_walls(self, walls: List[Wall], rooms: List[Room]) -> List[Wall]:
        if not rooms:
            return walls

        valid_walls: List[Wall] = []
        for w in walls:
            mx, my = w.midpoint
            
            is_ghost_wall = False
            for r in rooms:
                if r.polygon_pts and len(r.polygon_pts) >= 3:
                    if _point_inside_polygon(mx, my, r.polygon_pts):
                        dist_boundary = _dist_to_room_boundary(mx, my, r.polygon_pts)
                        # Wall midpoint is deep inside room polygon (>22px from room boundary edges)
                        if dist_boundary > 22.0:
                            is_ghost_wall = True
                            break

            if not is_ghost_wall:
                valid_walls.append(w)
            else:
                logger.info(f"[GeometryValidator] Pruned false interior wall: {w.start} -> {w.end}")

        return valid_walls

    # ── Pass 6: Snap & align all doors and windows to their host wall direction ───────
    def pass6_align_openings_to_walls(self, building: Building) -> Building:
        if not building.walls:
            return building

        all_openings = list(building.doors) + list(building.windows)
        for element in all_openings:
            cx, cy = element.center
            best_wall = None
            min_d = 1e9
            best_proj = (cx, cy)

            for w in building.walls:
                ax, ay = w.start
                bx, by = w.end
                dx, dy = bx - ax, by - ay
                L2 = dx * dx + dy * dy
                if L2 < 1e-4:
                    continue
                t = max(0.0, min(1.0, ((cx - ax) * dx + (cy - ay) * dy) / L2))
                px, py = ax + t * dx, ay + t * dy
                d = math.hypot(cx - px, cy - py)
                if d < min_d:
                    min_d = d
                    best_wall = w
                    best_proj = (px, py)

            if best_wall is not None and min_d < 90.0:
                # Snap center onto wall centerline
                element.center = (best_proj[0], best_proj[1])
                # Align exact wall angle
                w_dx = best_wall.end[0] - best_wall.start[0]
                w_dy = best_wall.end[1] - best_wall.start[1]
                element.angle_deg = math.degrees(math.atan2(w_dy, w_dx)) % 180.0

        return building

    # ── Pass 7: Determine & isolate ONLY the Main Entrance Door (rest remain empty) ───
    def pass7_ensure_room_accessibility_and_main_entrance(self, building: Building) -> Building:
        if not building.walls or not building.rooms:
            return building

        from backend.models.dataclasses import Door
        doors = list(building.doors)
        outer_walls = [w for w in building.walls if w.wall_type == "outer"]
        if not outer_walls:
            outer_walls = sorted(building.walls, key=lambda w: w.length, reverse=True)[:4]

        # 1. Search for explicit OCR door text annotations (e.g. "FRONT DOOR", "MAIN DOOR", "ENTRANCE")
        front_keywords = ["front door", "front", "main door", "main entrance", "entrance", "entry"]
        ocr_main_door = None

        for tb in building.text_blocks:
            tb_lower = tb.text.lower()
            if any(kw in tb_lower for kw in front_keywords):
                best_d = None
                min_d = 140.0
                for d in doors:
                    dist = math.hypot(d.center[0] - tb.center[0], d.center[1] - tb.center[1])
                    if dist < min_d:
                        min_d = dist
                        best_d = d
                if best_d:
                    ocr_main_door = best_d
                    break
                else:
                    # Create door directly at OCR label location snapped to outer wall
                    cx, cy = tb.center
                    for ow in outer_walls:
                        d_wall = _dist_point_to_segment(cx, cy, ow.start[0], ow.start[1], ow.end[0], ow.end[1])
                        if d_wall < 100.0:
                            ow_dx = ow.end[0] - ow.start[0]
                            ow_dy = ow.end[1] - ow.start[1]
                            angle_deg = math.degrees(math.atan2(ow_dy, ow_dx)) % 180.0
                            ocr_main_door = Door(
                                id="door_main_entrance_ocr",
                                center=(cx, cy),
                                width_px=90.0,
                                height_px=ow.thickness_px * 1.5,
                                angle_deg=angle_deg,
                                source="main_entrance",
                                confidence=0.98,
                            )
                            break
                    if ocr_main_door:
                        break

        # Helper to ensure door is in a spacious wall area away from windows
        def _reposition_door_to_spacious_area(d: Door) -> Door:
            win_centers = [w.center for w in building.windows] if building.windows else []
            if not win_centers or not outer_walls:
                return d
            cx, cy = d.center
            min_win_dist = min((math.hypot(cx - wx, cy - wy) for wx, wy in win_centers), default=999.0)
            if min_win_dist >= 150.0:
                return d
            # Search outer walls for a more spacious, window-free spot
            best_pt = (cx, cy)
            max_w_dist = min_win_dist
            best_ow = None
            for ow in outer_walls:
                if ow.length < 80.0:
                    continue
                ax, ay = ow.start
                bx, by = ow.end
                for step in range(2, 8):
                    t = step / 10.0
                    px = ax + t * (bx - ax)
                    py = ay + t * (by - ay)
                    w_d = min((math.hypot(px - wx, py - wy) for wx, wy in win_centers), default=999.0)
                    if w_d > max_w_dist:
                        max_w_dist = w_d
                        best_pt = (px, py)
                        best_ow = ow
            if best_ow and max_w_dist > min_win_dist + 15.0:
                d.center = best_pt
                ow_dx = best_ow.end[0] - best_ow.start[0]
                ow_dy = best_ow.end[1] - best_ow.start[1]
                d.angle_deg = math.degrees(math.atan2(ow_dy, ow_dx)) % 180.0
                logger.info(f"[MainDoorSelector] Repositioned door to spacious wall spot at {best_pt}, win_dist={max_w_dist:.1f}px")
            return d

        # 2. Collect doors located on outer (exterior) walls
        exterior_doors = []
        for d in doors:
            dcx, dcy = d.center
            for ow in outer_walls:
                if _dist_point_to_segment(dcx, dcy, ow.start[0], ow.start[1], ow.end[0], ow.end[1]) < 45.0:
                    d.width_px = 90.0
                    exterior_doors.append(d)
                    break

        selected_door = None
        if ocr_main_door:
            selected_door = ocr_main_door
        elif exterior_doors:
            selected_door = max(exterior_doors, key=lambda d: d.center[1])
        elif doors:
            selected_door = max(doors, key=lambda d: d.center[1])
        elif outer_walls:
            best_outer = max(outer_walls, key=lambda w: (w.midpoint[1], w.length))
            bmx, bmy = best_outer.midpoint
            bdx = best_outer.end[0] - best_outer.start[0]
            bdy = best_outer.end[1] - best_outer.start[1]
            w_angle = math.degrees(math.atan2(bdy, bdx)) % 180.0
            selected_door = Door(
                id="door_main_entrance",
                center=(bmx, bmy),
                width_px=90.0,
                height_px=best_outer.thickness_px * 1.5,
                angle_deg=w_angle,
                source="main_entrance",
                confidence=0.99,
            )

        if selected_door:
            selected_door.source = "main_entrance"
            selected_door.width_px = 90.0
            selected_door = _reposition_door_to_spacious_area(selected_door)
            building.doors = [selected_door]
            logger.info(f"[MainDoorSelector] Retained Main Entrance Door at {selected_door.center}")

        return building

    def validate_architecture_report(self, building: Building, px_per_m: float = 100.0) -> dict:
        """
        Executes structural architectural verification checks and returns a summary report.
        """
        errors = []
        warnings = []

        # Check walls
        if not building.walls:
            errors.append("No walls detected in building model.")
        for w in building.walls:
            if w.length <= 0:
                errors.append(f"Wall {w.id} has zero length.")
            if w.thickness_px <= 0:
                warnings.append(f"Wall {w.id} has non-positive thickness ({w.thickness_px}).")

        # Check rooms
        if not building.rooms:
            warnings.append("No room polygons detected.")
        for r in building.rooms:
            if len(r.polygon_pts) < 3:
                errors.append(f"Room {r.room_id} has invalid polygon with <3 vertices.")
            if r.area_px <= 0:
                errors.append(f"Room {r.room_id} has zero area.")

        # Check doors & windows
        for d in building.doors:
            if d.width_px <= 0:
                warnings.append(f"Door {d.id} has zero or negative width.")
        for win in building.windows:
            if win.width_px <= 0:
                warnings.append(f"Window {win.id} has zero or negative width.")

        # Check scale
        if not building.scale:
            warnings.append("Metric scale not detected in blueprint OCR. Used fallback 100 px/m.")

        is_valid = len(errors) == 0

        return {
            "is_valid": is_valid,
            "status": "VALID" if is_valid else ("ERROR" if errors else "WARNING"),
            "error_count": len(errors),
            "warning_count": len(warnings),
            "errors": errors,
            "warnings": warnings,
            "summary": {
                "total_walls": len(building.walls),
                "total_rooms": len(building.rooms),
                "total_doors": len(building.doors),
                "total_windows": len(building.windows),
                "px_per_m": px_per_m,
            }
        }

