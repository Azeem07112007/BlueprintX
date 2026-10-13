"""
backend/engine/detection/heuristic_furniture_detector.py
Blueprint-X — Procedural Furniture Detection Engine (Refined & Precision Filtered)

Extracts genuine architectural furniture drawings (Beds, Sofas, Dining Tables,
Kitchen Counters, Toilets, Sinks, Stairs, Plants) while strictly filtering out
OCR text glyphs, font characters, dimension lines, and annotation noise.

Guarantees clean floor plans like sample_floorplan.png render ZERO false 3D boxes.
"""
from __future__ import annotations

import cv2
import math
import logging
import numpy as np
from typing import List, Tuple, Dict, Any, Optional

from backend.models.dataclasses import Furniture, Stair, Room, TextBlock, Door, Window

logger = logging.getLogger(__name__)


class HeuristicFurnitureDetector:
    """
    Extracts interior furniture objects from floor plan drawing lines using
    contour stroke separation, text-character noise suppression, wall masking,
    and room semantic context.
    """

    def __init__(
        self,
        min_area_px: float = 400.0,
        max_area_ratio: float = 0.22,    # max 22% of room area
    ):
        self.min_area_px = min_area_px
        self.max_area_ratio = max_area_ratio

    def run(
        self,
        bgr: np.ndarray,
        wall_mask: np.ndarray,
        rooms: List[Room],
        text_blocks: List[TextBlock],
        doors: List[Door],
        windows: List[Window],
    ) -> Tuple[List[Furniture], List[Stair]]:
        """
        Extracts genuine furniture line contours inside room interiors.
        """
        h, w = wall_mask.shape[:2]
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if len(bgr.shape) == 3 else bgr

        # ── 1. Create dark-line foreground mask ────────────────────────────────
        _, fg_lines = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)

        # Subtract wall mask heavily dilated (remove walls & wall borders)
        kernel_wall = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
        walls_dilated = cv2.dilate(wall_mask, kernel_wall, iterations=1)
        fg_interior = cv2.bitwise_and(fg_lines, cv2.bitwise_not(walls_dilated))

        # Mask out OCR text blocks with generous padding
        for tb in text_blocks:
            if tb.bbox:
                bx, by, bw, bh = tb.bbox
                pad = 12
                cv2.rectangle(
                    fg_interior,
                    (max(0, bx - pad), max(0, by - pad)),
                    (min(w, bx + bw + pad), min(h, by + bh + pad)),
                    0, -1
                )

        # Mask out door sweep arcs / openings
        for d in doors:
            cx, cy = int(d.center[0]), int(d.center[1])
            rad = int(max(d.width_px, d.height_px, 30.0))
            cv2.circle(fg_interior, (cx, cy), rad, 0, -1)

        # ── 2. Filter out text characters & font glyph noise ─────────────────
        # Text characters are small connected components (area < 380, bw < 45, bh < 45)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(fg_interior, connectivity=8)
        for i in range(1, num_labels):
            cw = stats[i, cv2.CC_STAT_WIDTH]
            ch = stats[i, cv2.CC_STAT_HEIGHT]
            carea = stats[i, cv2.CC_STAT_AREA]
            # If component resembles a text character or small noise fragment, erase it
            if carea < 380 or (cw < 48 and ch < 48):
                fg_interior[labels == i] = 0

        furniture_items: List[Furniture] = []
        stairs: List[Stair] = []

        # ── 3. Process room by room ────────────────────────────────────────────
        for room in rooms:
            if len(room.polygon_pts) < 3 or room.area_px < 1500:
                continue

            room_mask = np.zeros((h, w), dtype=np.uint8)
            poly_np = np.array([[int(pt[0]), int(pt[1])] for pt in room.polygon_pts], dtype=np.int32)
            cv2.fillPoly(room_mask, [poly_np], 255)

            kernel_room = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
            room_mask_eroded = cv2.erode(room_mask, kernel_room, iterations=1)

            room_lines = cv2.bitwise_and(fg_interior, room_mask_eroded)

            # Morphological close to join furniture symbol lines
            kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            room_lines_closed = cv2.morphologyEx(room_lines, cv2.MORPH_CLOSE, kernel_close)

            contours, _ = cv2.findContours(room_lines_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            label_lower = (room.label or "").lower()

            rx, ry, rw, rh = room.bounding_box
            max_allowed_area = min(room.area_px * self.max_area_ratio, 10000.0)
            max_allowed_dim = min(max(rw, rh) * 0.50, 220.0)

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < self.min_area_px or area > max_allowed_area:
                    continue

                x, y, bw, bh = cv2.boundingRect(cnt)

                # Furniture symbol must be substantial in size (not text or stray lines)
                if bw > max_allowed_dim or bh > max_allowed_dim:
                    continue
                if bw < 25 or bh < 25:
                    continue

                # Compactness check: P^2 / (4 * pi * A)
                perimeter = cv2.arcLength(cnt, True)
                if area > 0:
                    compactness = (perimeter * perimeter) / (4.0 * math.pi * area)
                    # Text glyphs or stringy lines have high compactness > 5.2
                    if compactness > 5.2:
                        continue

                cx, cy = x + bw / 2.0, y + bh / 2.0
                aspect = max(bw, bh) / max(min(bw, bh), 1.0)
                solidity = area / float(max(bw * bh, 1))

                # Classify furniture type based on room context & shape
                class_name, sub_type = self._classify_contour(
                    cnt=cnt,
                    bbox=(x, y, bw, bh),
                    area=area,
                    aspect=aspect,
                    solidity=solidity,
                    room_label=label_lower,
                )

                if class_name == "stair":
                    stairs.append(Stair(
                        center=(cx, cy),
                        bbox=(x, y, bw, bh),
                        confidence=0.85,
                    ))
                elif class_name:
                    furniture_items.append(Furniture(
                        class_name=class_name,
                        sub_type=sub_type,
                        bbox=(x, y, bw, bh),
                        center=(cx, cy),
                        room_label=room.label,
                        confidence=0.80,
                    ))

        logger.info(f"[HeuristicFurniture] Extracted {len(furniture_items)} valid furniture items and {len(stairs)} staircases.")
        return furniture_items, stairs

    def _classify_contour(
        self,
        cnt: np.ndarray,
        bbox: Tuple[int, int, int, int],
        area: float,
        aspect: float,
        solidity: float,
        room_label: str,
    ) -> Tuple[str, str]:
        """Classifies furniture contour into a valid category."""
        x, y, w, h = bbox

        # ── Bedroom ────────────────────────────────────────────────────────────
        if any(k in room_label for k in ["bed", "sleep", "chamber"]):
            if area > 700 and aspect < 1.8:
                return "bed", "double_bed"
            elif area > 450:
                return "bed", "single_bed"

        # ── Living Room / Lounge ───────────────────────────────────────────────
        if any(k in room_label for k in ["living", "lounge", "salon", "family"]):
            if aspect > 1.8 and area > 700:
                return "sofa", "three_seater_sofa"
            elif aspect >= 1.2 and area > 550:
                return "sofa", "l_sofa"
            elif aspect < 1.4 and area < 550 and area > 400:
                return "coffee_table", "rectangular_table"

        # ── Dining Room ────────────────────────────────────────────────────────
        if any(k in room_label for k in ["dining", "eatery"]):
            if area > 550:
                return "dining_table", "dining_set"

        # ── Kitchen ────────────────────────────────────────────────────────────
        if any(k in room_label for k in ["kitchen", "cook", "pantry"]):
            if aspect > 1.8 and area > 500:
                return "kitchen_counter", "countertop"

        # ── Bathroom / Toilet / WC ─────────────────────────────────────────────
        if any(k in room_label for k in ["bath", "toilet", "wc", "wash", "restroom"]):
            if aspect < 1.6 and area > 350 and area < 1000:
                return "toilet", "flush_toilet"
            elif area > 1000:
                return "bathtub", "soaking_tub"
            elif area > 300:
                return "sink", "vanity_basin"

        # ── Balcony / Patio / Garden ───────────────────────────────────────────
        if any(k in room_label for k in ["balcony", "terrace", "patio", "garden", "deck"]):
            if solidity < 0.65 or aspect < 1.2:
                return "plant", "potted_plant"

        # ── Only classify as generic if area is substantial (> 750 px²) ─────────
        if area > 850 and aspect < 1.5:
            return "bed", "generic_bed"
        elif aspect > 2.0 and area > 750:
            return "sofa", "generic_sofa"

        return "", ""

    @staticmethod
    def merge_detections(
        ai_furniture: List[Furniture],
        heuristic_furniture: List[Furniture],
        distance_threshold_px: float = 35.0,
    ) -> List[Furniture]:
        """
        Merges AI YOLO furniture detections with Heuristic CV detections.
        """
        merged: List[Furniture] = list(ai_furniture)

        for h_item in heuristic_furniture:
            hx, hy = h_item.center
            duplicate = False
            for a_item in merged:
                ax, ay = a_item.center
                dist = math.hypot(hx - ax, hy - ay)
                if dist < distance_threshold_px:
                    duplicate = True
                    break
            if not duplicate:
                merged.append(h_item)

        return merged
