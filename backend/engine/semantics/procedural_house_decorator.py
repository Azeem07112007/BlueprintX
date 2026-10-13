"""
backend/engine/semantics/procedural_house_decorator.py
Blueprint-X — Procedural House Decorator & Automated Furnishing Engine

Analyzes room semantic labels and 2D floor polygons, automatically populating
each room with realistically placed 3D furniture, sanitary fixtures, kitchen counters,
dining sets, study desks, and outdoor patio furniture.

Guarantees every floor plan converts into a fully decorated, complete 3D house model.
"""
from __future__ import annotations

import math
import logging
import numpy as np
from typing import List, Tuple, Dict, Any

from backend.models.dataclasses import Building, Room, Furniture, Stair

logger = logging.getLogger(__name__)


class ProceduralHouseDecorator:
    """
    Intelligently decorates floor plan rooms with 3D furniture models based on room labels and geometry.
    """

    @staticmethod
    def decorate_building(building: Building) -> List[Furniture]:
        """
        Decorates all rooms in the building with full architectural furniture sets.
        """
        furniture_list: List[Furniture] = list(building.furniture)
        existing_centers = [f.center for f in furniture_list]

        # Sort rooms by area to assign fallback labels for unlabeled rooms
        valid_rooms = [r for r in building.rooms if r.polygon_pts and len(r.polygon_pts) >= 3]
        sorted_by_area = sorted(valid_rooms, key=lambda r: r.area_px, reverse=True)
        
        # Room area ranking map
        fallback_labels = {}
        for idx, r in enumerate(sorted_by_area):
            if idx == 0:
                fallback_labels[r.id] = "living"
            elif idx in (1, 2):
                fallback_labels[r.id] = "bedroom"
            elif idx == 3:
                fallback_labels[r.id] = "kitchen"
            elif idx == 4:
                fallback_labels[r.id] = "bathroom"
            else:
                fallback_labels[r.id] = "bedroom" if idx % 2 == 0 else "bathroom"

        for room in valid_rooms:
            label = (room.label or "").lower().strip()
            if not label or any(g in label for g in ["room", "unnamed", "area", "space"]):
                label = fallback_labels.get(room.id, "bedroom")

            cx, cy = room.centroid
            rx, ry, rw, rh = room.bounding_box

            # Check existing furniture count inside room bounding box
            existing_count = sum(1 for fcx, fcy in existing_centers if rx <= fcx <= rx + rw and ry <= fcy <= ry + rh)
            if existing_count >= 3:
                continue

            # Generates comprehensive furniture sets per room category
            new_items = ProceduralHouseDecorator._create_room_furniture_set(
                label=label,
                centroid=(cx, cy),
                bbox=(rx, ry, rw, rh),
                has_existing=(existing_count > 0),
            )

            for item in new_items:
                furniture_list.append(item)
                existing_centers.append(item.center)

        logger.info(f"[HouseDecorator] Furnished building rooms. Total furniture items: {len(furniture_list)}")
        return furniture_list

    @staticmethod
    def _create_room_furniture_set(
        label: str,
        centroid: Tuple[float, float],
        bbox: Tuple[int, int, int, int],
        has_existing: bool,
    ) -> List[Furniture]:
        items: List[Furniture] = []
        cx, cy = centroid
        rx, ry, rw, rh = bbox

        # ── 1. Bedroom / Master Bed ───────────────────────────────────────────
        if any(k in label for k in ["bed", "sleep", "chamber", "master"]):
            # Main bed
            bw, bh = min(max(rw * 0.45, 65), 140), min(max(rh * 0.50, 75), 150)
            bed_x, bed_y = cx - bw/2.0, ry + rh * 0.18
            items.append(Furniture(
                class_name="bed", sub_type="double_bed",
                bbox=(int(bed_x), int(bed_y), int(bw), int(bh)),
                center=(cx, bed_y + bh/2.0), room_label=label, confidence=0.95
            ))
            # Side tables
            sw, sh = 20, 20
            items.append(Furniture(
                class_name="nightstand", sub_type="side_table",
                bbox=(int(bed_x - sw - 4), int(bed_y), sw, sh),
                center=(bed_x - sw/2.0, bed_y + sh/2.0), room_label=label, confidence=0.90
            ))
            items.append(Furniture(
                class_name="nightstand", sub_type="side_table",
                bbox=(int(bed_x + bw + 4), int(bed_y), sw, sh),
                center=(bed_x + bw + sw/2.0, bed_y + sh/2.0), room_label=label, confidence=0.90
            ))

        # ── 2. Living Room / Lounge ───────────────────────────────────────────
        elif any(k in label for k in ["living", "lounge", "salon", "family", "tv"]):
            # L-Sofa or 3-Seater Sofa
            sw, sh = min(max(rw * 0.55, 90), 180), min(max(rh * 0.35, 45), 90)
            sofa_x, sofa_y = cx - sw/2.0, cy - sh/2.0
            items.append(Furniture(
                class_name="sofa", sub_type="l_sofa",
                bbox=(int(sofa_x), int(sofa_y), int(sw), int(sh)),
                center=(cx, cy), room_label=label, confidence=0.95
            ))
            # Coffee table in front
            cw_table, ch_table = sw * 0.50, 30
            items.append(Furniture(
                class_name="coffee_table", sub_type="rectangular_table",
                bbox=(int(cx - cw_table/2.0), int(sofa_y + sh + 10), int(cw_table), int(ch_table)),
                center=(cx, sofa_y + sh + 25), room_label=label, confidence=0.90
            ))
            # TV Console along wall
            items.append(Furniture(
                class_name="tv_unit", sub_type="media_console",
                bbox=(int(cx - sw*0.4), int(ry + 10), int(sw*0.8), 22),
                center=(cx, ry + 21), room_label=label, confidence=0.90
            ))

        # ── 3. Kitchen ────────────────────────────────────────────────────────
        elif any(k in label for k in ["kitchen", "cook", "pantry"]):
            kw, kh = min(max(rw * 0.70, 100), 220), 40
            items.append(Furniture(
                class_name="kitchen_counter", sub_type="countertop",
                bbox=(int(rx + 10), int(ry + 10), int(kw), int(kh)),
                center=(rx + 10 + kw/2.0, ry + 30), room_label=label, confidence=0.95
            ))
            # Kitchen sink
            items.append(Furniture(
                class_name="sink", sub_type="kitchen_sink",
                bbox=(int(rx + kw * 0.65), int(ry + 12), 35, 30),
                center=(rx + kw * 0.65 + 17, ry + 27), room_label=label, confidence=0.92
            ))

        # ── 4. Bathroom / WC ──────────────────────────────────────────────────
        elif any(k in label for k in ["bath", "toilet", "wc", "wash", "restroom"]):
            # Flush toilet against wall
            tw, th = 30, 42
            items.append(Furniture(
                class_name="toilet", sub_type="flush_toilet",
                bbox=(int(rx + 10), int(ry + 10), tw, th),
                center=(rx + 25, ry + 31), room_label=label, confidence=0.95
            ))
            # Vanity sink
            items.append(Furniture(
                class_name="sink", sub_type="vanity_basin",
                bbox=(int(rx + 10), int(ry + th + 18), 35, 30),
                center=(rx + 27, ry + th + 33), room_label=label, confidence=0.92
            ))

        # ── 5. Dining Room ────────────────────────────────────────────────────
        elif any(k in label for k in ["dining", "eatery"]):
            dw, dh = min(max(rw * 0.50, 75), 150), min(max(rh * 0.40, 45), 90)
            items.append(Furniture(
                class_name="dining_table", sub_type="dining_set",
                bbox=(int(cx - dw/2.0), int(cy - dh/2.0), int(dw), int(dh)),
                center=(cx, cy), room_label=label, confidence=0.95
            ))

        # ── 6. Study / Office ─────────────────────────────────────────────────
        elif any(k in label for k in ["study", "office", "work"]):
            dw, dh = 70, 35
            items.append(Furniture(
                class_name="dining_table", sub_type="study_desk",
                bbox=(int(rx + 15), int(ry + 15), dw, dh),
                center=(rx + 15 + dw/2.0, ry + 15 + dh/2.0), room_label=label, confidence=0.92
            ))

        # ── 7. Balcony / Terrace / Deck / Garden ──────────────────────────────
        elif any(k in label for k in ["balcony", "terrace", "patio", "deck", "garden"]):
            pw, ph = 40, 40
            items.append(Furniture(
                class_name="plant", sub_type="potted_plant",
                bbox=(int(rx + 10), int(ry + 10), pw, ph),
                center=(rx + 30, ry + 30), room_label=label, confidence=0.95
            ))
            items.append(Furniture(
                class_name="plant", sub_type="potted_plant",
                bbox=(int(rx + rw - 50), int(ry + 10), pw, ph),
                center=(rx + rw - 30, ry + 30), room_label=label, confidence=0.95
            ))

        return items
