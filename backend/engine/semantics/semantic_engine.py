"""
backend/engine/semantics/semantic_engine.py
Blueprint-X — Milestone 2

Fuses OCR text labels with geometric rooms.
Assigns room labels based on physical proximity of text blocks to room centroids.
When OCR is unavailable, applies size-based heuristic labels.

Also computes real-world room areas (area_m2) using the building scale.
"""
from __future__ import annotations

import math
from typing import List, Optional

from backend.models.dataclasses import Room, TextBlock


class SemanticEngine:
    """
    Applies semantic labels to rooms based on OCR output.
    Falls back to size-based heuristic labels when OCR is unavailable.
    Also computes area_m2 for every room given a pixels-per-metre scale.
    """

    def run(
        self,
        rooms: List[Room],
        text_blocks: List[TextBlock],
        px_per_m: Optional[float] = None,
    ) -> None:
        if not rooms:
            return

        # 1. Match each OCR room label to the closest room centroid
        if text_blocks:
            room_labels = [tb for tb in text_blocks if tb.is_room_label]
            for label_block in room_labels:
                bx, by = label_block.center
                best_room = None
                min_dist = float("inf")

                for room in rooms:
                    # Check if text is inside the room's bounding box (with padding)
                    rx, ry, rw, rh = room.bounding_box
                    if rx - 20 <= bx <= rx + rw + 20 and ry - 20 <= by <= ry + rh + 20:
                        cx, cy = room.centroid
                        dist = math.hypot(cx - bx, cy - by)
                        if dist < min_dist:
                            min_dist = dist
                            best_room = room

                if best_room and not best_room.label:
                    # Capitalize nicely (e.g. "master bedroom" -> "Master Bedroom")
                    best_room.label = label_block.text.title()

        # 2. Auto-classify all unlabeled rooms via size-based heuristic fallback
        for room in rooms:
            if room.label:
                continue  # Already labeled by OCR

            if room.area_px < 2000:
                room.label = "Closet"
            elif room.area_px < 5000:
                room.label = "Bathroom / WC"
            elif room.area_px < 12000:
                room.label = "Small Room"
            elif room.area_px < 25000:
                room.label = "Room"
            elif room.area_px < 60000:
                room.label = "Living Space"
            else:
                room.label = "Open Plan"

        # 3. Compute real-world area for every room when scale is available.
        #    area_m2 = area_px / px_per_m²
        if px_per_m and px_per_m > 0:
            px_per_m2 = px_per_m ** 2
            for room in rooms:
                room.area_m2 = round(room.area_px / px_per_m2, 2)
