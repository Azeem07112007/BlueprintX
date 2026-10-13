"""
modules/furniture_detector.py
Blueprint-X Prototype - Milestone 1.5

Detects furniture by analysing the residual binary image (full - structural).
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

@dataclass
class FurnitureItem:
    label: str
    bounding_box: Tuple[int, int, int, int]  # x, y, w, h
    centroid: Tuple[int, int]
    area_px: float
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "bounding_box": {
                "x": int(self.bounding_box[0]),
                "y": int(self.bounding_box[1]),
                "width": int(self.bounding_box[2]),
                "height": int(self.bounding_box[3]),
            },
            "centroid": [int(self.centroid[0]), int(self.centroid[1])],
            "area_px": round(float(self.area_px), 1),
            "confidence": round(float(self.confidence), 3)
        }

def detect_furniture(full_binary: np.ndarray, structural_mask: np.ndarray) -> List[FurnitureItem]:
    """
    Detect furniture by subtracting the structural mask from the full binary mask.
    The remaining blobs are classified by shape/area heuristics.
    """
    # 1. Obtain residual mask (features that are NOT walls)
    # Full binary: walls + furniture are white (255)
    # Structural mask: only walls are white (255)
    residual = cv2.subtract(full_binary, structural_mask)
    
    # 2. Clean up residual (remove tiny noise like text/hatching)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(residual, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # 3. Find contours
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    furniture_items = []
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 400: # Ignore small noise/text
            continue
            
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / h if h > 0 else 0
        
        # Calculate centroid
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = x + w//2, y + h//2
            
        # Basic rule-based classification
        label = "Other"
        if 2000 <= area <= 15000 and 1.3 <= aspect_ratio <= 2.5:
            label = "Bed"
        elif 1500 <= area <= 8000 and 1.5 <= aspect_ratio <= 3.0:
            label = "Sofa"
        elif 2000 <= area <= 12000 and (0.8 <= aspect_ratio <= 1.2 or 1.4 <= aspect_ratio <= 1.6):
            label = "Dining Table"
        elif 400 <= area <= 1500:
            label = "Chair"
        elif 1000 <= area <= 6000 and aspect_ratio > 3.0:
            label = "Kitchen Counter"
        
        furniture_items.append(FurnitureItem(
            label=label,
            bounding_box=(x, y, w, h),
            centroid=(cx, cy),
            area_px=area
        ))
        
    return furniture_items
