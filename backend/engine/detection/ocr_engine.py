"""
backend/engine/detection/ocr_engine.py
Blueprint-X — Milestone 2

OCR engine for extracting text from blueprints.
Uses EasyOCR (primary) with graceful fallback if not installed.

Extracts:
  - Room labels (Kitchen, Bathroom, etc.)
  - Dimension values (3.5m, 2400mm, etc.)
  - Scale notation (1:100, Scale 1:50, etc.)
  - Other annotations

Also generates an OCR mask to blank text regions before geometry analysis.
"""
from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from backend.models.dataclasses import TextBlock

logger = logging.getLogger(__name__)

# ── Known room-label keywords ──────────────────────────────────────────────────
ROOM_KEYWORDS = {
    "bedroom", "bed", "living", "lounge", "kitchen", "dining", "bathroom",
    "bath", "toilet", "wc", "study", "office", "garage", "storage", "store",
    "hall", "hallway", "corridor", "passage", "laundry", "utility", "pantry",
    "porch", "balcony", "deck", "master", "ensuite", "gym", "library",
}

# ── Dimension pattern ──────────────────────────────────────────────────────────
DIM_PATTERN = re.compile(
    r"(\d+[\.,]\d+\s*(m|mm|cm|ft|'|\"|\"|in)?|\d+\s*(m|mm|cm|ft|'|\"|\"|in))",
    re.IGNORECASE,
)

# ── Scale pattern ──────────────────────────────────────────────────────────────
SCALE_PATTERN = re.compile(r"1\s*[:\-]\s*(\d+)", re.IGNORECASE)

# ── Window pattern ─────────────────────────────────────────────────────────────
WINDOW_PATTERN = re.compile(r"\b(window|win)\b", re.IGNORECASE)


@dataclass
class OCRResult:
    text_blocks: List[TextBlock] = field(default_factory=list)
    ocr_mask: Optional[np.ndarray] = None      # white mask over text regions
    scale: str = ""
    debug: Dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def room_labels(self) -> List[TextBlock]:
        return [t for t in self.text_blocks if t.is_room_label]

    @property
    def dimensions(self) -> List[TextBlock]:
        return [t for t in self.text_blocks if t.is_dimension]

    @property
    def window_labels(self) -> List[TextBlock]:
        return [t for t in self.text_blocks if t.is_window_label]


class OCREngine:
    """
    Extracts text from a blueprint image using EasyOCR.
    Falls back gracefully if easyocr is not installed.
    """

    def __init__(self, languages: List[str] = None):
        self._reader = None
        self._languages = languages or ["en"]
        self._load_reader()

    def _load_reader(self):
        try:
            import easyocr
            self._reader = easyocr.Reader(self._languages, gpu=False, verbose=False)
            logger.info("[OCREngine] EasyOCR reader loaded.")
        except ImportError:
            logger.info("[OCREngine] EasyOCR not installed — running in dummy mode.")
        except Exception as e:
            logger.warning(f"[OCREngine] Failed to load EasyOCR: {e}")

    @property
    def is_loaded(self) -> bool:
        return self._reader is not None

    def run(self, bgr: np.ndarray) -> OCRResult:
        result = OCRResult()
        h, w = bgr.shape[:2]
        result.ocr_mask = np.zeros((h, w), dtype=np.uint8)

        if not self.is_loaded:
            result.debug["ocr_01_mask"] = self._dummy_overlay(bgr)
            return result

        # Run OCR
        try:
            raw = self._reader.readtext(bgr, detail=1)
        except Exception as e:
            logger.error(f"[OCREngine] OCR failed: {e}")
            result.debug["ocr_01_mask"] = self._dummy_overlay(bgr, error=str(e))
            return result

        text_blocks: List[TextBlock] = []
        scale = ""

        for bbox_pts, text, conf in raw:
            if conf < 0.3:
                continue

            text = text.strip()
            if not text:
                continue

            # Compute axis-aligned bounding box from quad points
            xs = [int(p[0]) for p in bbox_pts]
            ys = [int(p[1]) for p in bbox_pts]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            bx, by = x_min, y_min
            bw, bh = x_max - x_min, y_max - y_min
            cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2

            is_room = self._is_room_label(text)
            is_dim = bool(DIM_PATTERN.search(text))
            is_win = bool(WINDOW_PATTERN.search(text))
            scale_match = SCALE_PATTERN.search(text)
            is_scale = scale_match is not None
            if is_scale and not scale:
                scale = f"1:{scale_match.group(1)}"

            block = TextBlock(
                text=text,
                bbox=(bx, by, bw, bh),
                center=(cx, cy),
                is_room_label=is_room,
                is_dimension=is_dim,
                is_scale=is_scale,
                is_window_label=is_win,
                confidence=float(conf),
            )
            text_blocks.append(block)

            # Add to OCR mask
            cv2.fillPoly(result.ocr_mask, [np.array(bbox_pts, dtype=np.int32)], 255)

        result.text_blocks = text_blocks
        result.scale = scale

        # Debug visualization
        result.debug["ocr_01_mask"] = self._draw_ocr_debug(bgr, text_blocks, result.ocr_mask)
        return result

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _is_room_label(self, text: str) -> bool:
        lower = text.lower()
        return any(kw in lower for kw in ROOM_KEYWORDS)

    def _draw_ocr_debug(
        self,
        bgr: np.ndarray,
        blocks: List[TextBlock],
        mask: np.ndarray,
    ) -> np.ndarray:
        canvas = bgr.copy()
        for block in blocks:
            if block.bbox:
                x, y, bw, bh = block.bbox
                color = (0, 220, 80) if block.is_room_label else (200, 200, 0) if block.is_dimension else (180, 180, 180)
                cv2.rectangle(canvas, (x, y), (x + bw, y + bh), color, 1)
                cv2.putText(canvas, block.text[:20], (x, y - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
        # Overlay mask tinted
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        mask_bgr[:, :, 0] = 0  # zero blue
        canvas = cv2.addWeighted(canvas, 0.8, mask_bgr, 0.2, 0)
        return canvas

    def _dummy_overlay(self, bgr: np.ndarray, error: str = "") -> np.ndarray:
        """Return the original image with a small non-destructive info badge.

        Matches the badge style used in AIDetector._no_model_overlay so all
        'module not available' debug views look consistent.
        """
        canvas = bgr.copy()
        msg = f"OCR: EasyOCR not installed{' | ' + error[:40] if error else ''}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale, thick = 0.45, 1
        (tw, th), baseline = cv2.getTextSize(msg, font, scale, thick)
        pad = 6
        x0, y0 = 10, 10
        cv2.rectangle(canvas,
                      (x0, y0),
                      (x0 + tw + pad * 2, y0 + th + baseline + pad * 2),
                      (30, 30, 30), -1)
        cv2.rectangle(canvas,
                      (x0, y0),
                      (x0 + tw + pad * 2, y0 + th + baseline + pad * 2),
                      (80, 80, 80), 1)
        cv2.putText(canvas, msg,
                    (x0 + pad, y0 + th + pad),
                    font, scale, (180, 180, 180), thick, cv2.LINE_AA)
        return canvas

