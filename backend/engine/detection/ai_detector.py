"""
backend/engine/detection/ai_detector.py
Blueprint-X — Milestone 2

YOLO11 Segmentation detector for architectural elements.
Detects: doors, windows, furniture, stairs, columns, toilets, fixtures, etc.

If best.pt is not found, returns a graceful empty result.
This ensures the application always runs even without a trained model.
"""
from __future__ import annotations

import os
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from backend.models.dataclasses import DetectedObject, Door, Window, Furniture, Column, Stair

logger = logging.getLogger(__name__)

# ── Class name → semantic role mapping ────────────────────────────────────────
CLASS_ROLES = {
    "door":              "door",
    "window":            "window",
    "furniture":         "furniture",
    "chair":             "furniture",
    "table":             "furniture",
    "sofa":              "furniture",
    "bed":               "furniture",
    "stairs":            "stair",
    "column":            "column",
    "toilet":            "furniture",
    "sink":              "furniture",
    "bathtub":           "furniture",
    "kitchen_counter":   "furniture",
    "electrical":        "furniture",
    "text_block":        "text",
    "dimension":         "text",
}

# Default weights path
DEFAULT_WEIGHTS = Path(__file__).parent.parent.parent.parent / "training" / "weights" / "best.pt"


@dataclass
class AIDetectionResult:
    model_loaded: bool = False
    model_path: str = ""
    detections: List[DetectedObject] = field(default_factory=list)
    doors: List[Door] = field(default_factory=list)
    windows: List[Window] = field(default_factory=list)
    furniture: List[Furniture] = field(default_factory=list)
    columns: List[Column] = field(default_factory=list)
    stairs: List[Stair] = field(default_factory=list)
    debug: Dict[str, np.ndarray] = field(default_factory=dict)


class AIDetector:
    """
    YOLO11 segmentation detector with graceful dummy fallback.

    Usage:
        detector = AIDetector()
        result = detector.run(bgr_image)
        if not result.model_loaded:
            print("No model — using geometric detection only")
    """

    def __init__(self, weights_path: Optional[str] = None):
        self._model = None
        self._model_path = ""
        self._class_names: List[str] = []

        path = Path(weights_path) if weights_path else DEFAULT_WEIGHTS
        if path.exists():
            self._load_model(str(path))
        else:
            logger.info(
                f"[AIDetector] No weights found at {path}. "
                "Running in dummy mode (geometric detection only)."
            )

    def _load_model(self, path: str):
        try:
            from ultralytics import YOLO
            self._model = YOLO(path)
            self._model_path = path
            if hasattr(self._model, "names"):
                self._class_names = list(self._model.names.values())
            logger.info(f"[AIDetector] Loaded YOLO model: {path}")
        except Exception as e:
            logger.warning(f"[AIDetector] Failed to load model: {e}")
            self._model = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def run(
        self,
        bgr: np.ndarray,
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.45,
    ) -> AIDetectionResult:
        result = AIDetectionResult(
            model_loaded=self.is_loaded,
            model_path=self._model_path,
        )

        if not self.is_loaded:
            result.debug["ai_01_detections"] = self._no_model_overlay(bgr)
            return result

        # Run YOLO inference
        try:
            predictions = self._model.predict(
                bgr,
                conf=conf_threshold,
                iou=iou_threshold,
                verbose=False,
            )
        except Exception as e:
            logger.error(f"[AIDetector] Inference failed: {e}")
            result.debug["ai_01_detections"] = self._no_model_overlay(bgr, error=str(e))
            return result

        # Parse predictions
        raw_detections: List[DetectedObject] = []
        if predictions:
            pred = predictions[0]
            boxes = pred.boxes
            masks = pred.masks  # may be None for bbox-only models

            if boxes is not None:
                for idx in range(len(boxes)):
                    cls_id = int(boxes.cls[idx].item())
                    conf = float(boxes.conf[idx].item())
                    x1, y1, x2, y2 = boxes.xyxy[idx].tolist()
                    x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
                    cx, cy = x + w / 2, y + h / 2

                    class_name = (
                        self._class_names[cls_id]
                        if cls_id < len(self._class_names)
                        else f"class_{cls_id}"
                    )

                    mask = None
                    if masks is not None and idx < len(masks.data):
                        mask_data = masks.data[idx].cpu().numpy()
                        mask = (mask_data * 255).astype(np.uint8)

                    raw_detections.append(DetectedObject(
                        class_name=class_name,
                        confidence=conf,
                        bbox=(x, y, w, h),
                        center=(cx, cy),
                        mask=mask,
                    ))

        result.detections = raw_detections

        # Categorise into typed objects
        for det in raw_detections:
            role = CLASS_ROLES.get(det.class_name.lower(), "furniture")
            if role == "door":
                x, y, w, h = det.bbox
                result.doors.append(Door(
                    center=det.center,
                    width_px=float(w),
                    height_px=float(h),
                    angle_deg=0.0,
                    bbox=det.bbox,
                    mask=det.mask,
                    source="yolo",
                    confidence=det.confidence,
                ))
            elif role == "window":
                x, y, w, h = det.bbox
                result.windows.append(Window(
                    center=det.center,
                    width_px=float(w),
                    height_px=float(h),
                    bbox=det.bbox,
                    mask=det.mask,
                    source="yolo",
                    confidence=det.confidence,
                ))
            elif role == "column":
                result.columns.append(Column(
                    center=det.center,
                    bbox=det.bbox,
                    confidence=det.confidence,
                ))
            elif role == "stair":
                result.stairs.append(Stair(
                    center=det.center,
                    bbox=det.bbox,
                    confidence=det.confidence,
                ))
            else:
                result.furniture.append(Furniture(
                    class_name=det.class_name,
                    center=det.center,
                    bbox=det.bbox,
                    mask=det.mask,
                    confidence=det.confidence,
                ))

        # Debug visualization
        result.debug["ai_01_detections"] = self._draw_detections(bgr, raw_detections)
        return result

    def _draw_detections(self, bgr: np.ndarray, detections: List[DetectedObject]) -> np.ndarray:
        canvas = bgr.copy()
        palette = {
            "door":     (0, 220, 80),
            "window":   (255, 200, 0),
            "stairs":   (180, 100, 255),
            "column":   (0, 180, 255),
            "furniture":(200, 100, 60),
        }
        for det in detections:
            role = CLASS_ROLES.get(det.class_name.lower(), "furniture")
            color = palette.get(role, (180, 180, 180))
            x, y, w, h = det.bbox
            cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 2)
            label = f"{det.class_name} {det.confidence:.2f}"
            cv2.putText(canvas, label, (x, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
            if det.mask is not None:
                try:
                    mask_rs = cv2.resize(det.mask, (bgr.shape[1], bgr.shape[0]))
                    overlay = canvas.copy()
                    overlay[mask_rs > 128] = np.array(color, dtype=np.uint8)
                    cv2.addWeighted(overlay, 0.3, canvas, 0.7, 0, canvas)
                except Exception:
                    pass
        return canvas

    def _no_model_overlay(self, bgr: np.ndarray, error: str = "") -> np.ndarray:
        """Return the original image with a small non-destructive info badge.

        Previously this darkened the whole image and put grey text in the
        centre — that made every debug view look broken even though the
        geometric pipeline still runs without a YOLO model.  Now we just
        draw a small pill badge in the top-left corner so the image stays
        readable.
        """
        canvas = bgr.copy()
        msg = f"AI: geometric-only mode{' | ' + error[:40] if error else ''}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale, thick = 0.45, 1
        (tw, th), baseline = cv2.getTextSize(msg, font, scale, thick)
        pad = 6
        x0, y0 = 10, 10
        # Draw pill background
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
