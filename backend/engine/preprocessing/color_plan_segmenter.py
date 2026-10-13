"""
backend/engine/preprocessing/color_plan_segmenter.py
Blueprint-X — Color-Aware Rendered Floor Plan Segmenter

Isolates true structural dark-grey wall bands from professional color-rendered floor plans.
Uses HSV saturation filtering and morphological thickness erosion to completely eliminate
colored furniture (sofas, tables, patio decks, green gardens) and thin stair/text strokes
from the wall mask.
"""
from __future__ import annotations

import cv2
import numpy as np
from typing import Tuple


class ColorPlanSegmenter:
    """
    Precision color-space & thickness segmenter for rendered architectural plans.
    """

    @staticmethod
    def segment_walls(bgr: np.ndarray) -> np.ndarray:
        """
        Returns a clean binary mask (255=wall, 0=background) for colored floor plans.
        """
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        H, S, V = cv2.split(hsv)

        # ── 1. Color Saturation & Value Filtering ─────────────────────────────
        # Walls are neutral dark-grey (Saturation <= 32, Value between 25 and 125)
        wall_color_mask = cv2.inRange(hsv, (0, 0, 25), (180, 32, 125))

        # ── 2. Morphological Thickness Erosion ────────────────────────────────
        # Structural walls are thick (>= 10px). Erase thin stair lines & text fonts (1-4px)
        kernel_erase_thin = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        eroded = cv2.erode(wall_color_mask, kernel_erase_thin, iterations=1)

        # Dilate back to restore original wall thickness
        restored = cv2.dilate(eroded, kernel_erase_thin, iterations=1)

        # ── 3. Close small wall breaks & window gaps ──────────────────────────
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        closed = cv2.morphologyEx(restored, cv2.MORPH_CLOSE, kernel_close)

        # ── 4. Area filtering (remove small non-wall artifacts < 300px) ──────
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
        clean_wall_mask = np.zeros_like(closed)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= 300:
                clean_wall_mask[labels == i] = 255

        return clean_wall_mask
