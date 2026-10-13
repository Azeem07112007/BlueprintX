"""
backend/engine/detection/window_synthesizer.py
Blueprint-X — Automatic Window Synthesizer & Exterior Wall Inspector

Detects explicit window patterns on outer walls, and automatically synthesizes
realistic 3D exterior windows for rooms facing the outside environment.
Ensures every house model has proper 3D windows, glass panes, sills, and frames.
"""
from __future__ import annotations

import math
import logging
from typing import List, Tuple, Dict, Any

from backend.models.dataclasses import Building, Wall, Window, Room

logger = logging.getLogger(__name__)


class WindowSynthesizer:
    """
    Detects and synthesizes exterior 3D windows for architectural floor plans.
    """

    @staticmethod
    def _extract_ocr_windows(building: Building) -> List[Window]:
        """Extracts explicit window locations from OCR text blocks labeled with WINDOW specs."""
        ocr_windows: List[Window] = []
        for tb in building.text_blocks:
            is_win = getattr(tb, "is_window_label", False) or ("window" in tb.text.lower())
            if not is_win:
                continue

            cx, cy = tb.center
            # Check if near existing window
            if any(math.hypot(w.center[0] - cx, w.center[1] - cy) < 50.0 for w in building.windows):
                continue

            # Snap to nearest wall
            nearest_wall = None
            min_dist = 80.0
            proj_pt = (cx, cy)

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
                if d < min_dist:
                    min_dist = d
                    nearest_wall = w
                    proj_pt = (px, py)

            angle_deg = 0.0
            width_px = 90.0
            height_px = 15.0
            if nearest_wall:
                w_dx = nearest_wall.end[0] - nearest_wall.start[0]
                w_dy = nearest_wall.end[1] - nearest_wall.start[1]
                angle_deg = math.degrees(math.atan2(w_dy, w_dx)) % 180.0
                height_px = nearest_wall.thickness_px * 1.2

            win_bbox = tb.bbox or (int(proj_pt[0] - 45), int(proj_pt[1] - 8), 90, 16)
            ocr_win = Window(
                center=proj_pt,
                width_px=width_px,
                height_px=height_px,
                angle_deg=angle_deg,
                bbox=win_bbox,
                source="ocr_annotation",
                confidence=tb.confidence,
            )
            ocr_windows.append(ocr_win)

        return ocr_windows

    @staticmethod
    def synthesize_windows(building: Building, px_per_m: float = 100.0) -> List[Window]:
        """
        Scans outer walls and room boundaries. Generates explicit Window objects
        for all exterior-facing walls.
        """
        # First, extract any explicit windows annotated in OCR text blocks
        ocr_wins = WindowSynthesizer._extract_ocr_windows(building)
        if ocr_wins:
            building.windows.extend(ocr_wins)
            logger.info(f"[WindowSynthesizer] Extracted {len(ocr_wins)} explicit window annotations from blueprint text.")

        windows: List[Window] = list(building.windows)
        
        # If explicit windows are already present in the blueprint data, do not create extra ones on your own
        if len(windows) > 0:
            logger.info(f"[WindowSynthesizer] Using {len(windows)} explicit blueprint windows (skipping auto-synthesis).")
            return windows

        existing_centers = [(w.center[0], w.center[1]) for w in windows]

        outer_walls = [w for w in building.walls if w.wall_type == "outer"]
        if not outer_walls:
            # Fall back to longest perimeter walls
            outer_walls = sorted(building.walls, key=lambda w: w.length, reverse=True)[:8]

        for wall in outer_walls:
            if wall.length < 80.0:    # skip tiny wall fragments
                continue

            x1, y1 = wall.start
            x2, y2 = wall.end
            mid_x, mid_y = wall.midpoint

            # Check if a window already exists on this wall
            has_existing = False
            for cx, cy in existing_centers:
                dist = math.hypot(cx - mid_x, cy - mid_y)
                if dist < wall.length * 0.45:
                    has_existing = True
                    break

            if has_existing:
                continue

            # Check if this outer wall has door openings
            near_door = False
            for door in building.doors:
                dcx, dcy = door.center
                if math.hypot(dcx - mid_x, dcy - mid_y) < 50.0:
                    near_door = True
                    break
            if near_door:
                continue

            # Calculate window width: 40% of wall length, capped between 0.8m and 2.2m
            win_w_px = min(max(wall.length * 0.42, 60.0), 220.0)
            win_h_px = wall.thickness_px * 1.2

            dx = (x2 - x1) / wall.length
            dy = (y2 - y1) / wall.length
            angle_deg = math.degrees(math.atan2(dy, dx)) % 180

            win_x = mid_x - (win_w_px / 2.0) * dx
            win_y = mid_y - (win_w_px / 2.0) * dy

            win_bbox = (
                int(min(mid_x - win_w_px/2, mid_x + win_w_px/2)),
                int(min(mid_y - win_h_px/2, mid_y + win_h_px/2)),
                int(win_w_px),
                int(max(win_h_px, 12.0))
            )

            new_win = Window(
                center=(mid_x, mid_y),
                width_px=win_w_px,
                height_px=win_h_px,
                angle_deg=angle_deg,
                bbox=win_bbox,
                source="synthesizer",
                confidence=0.92,
            )
            windows.append(new_win)
            existing_centers.append((mid_x, mid_y))

        logger.info(f"[WindowSynthesizer] Synthesized {len(windows) - len(building.windows)} exterior windows. Total: {len(windows)}")
        return windows
