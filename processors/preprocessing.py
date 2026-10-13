"""
processors/preprocessing.py  — REWRITE (Stage 1)
Blueprint-X Topology Pipeline

Goal: Extract ONLY structural walls. Remove everything else:
  - Text / room labels
  - Dimension lines and annotations
  - Furniture outlines
  - Door swing arcs
  - Hatching / decorative lines

Strategy:
  Walls in architectural drawings are the THICKEST elements.
  A morphological opening with a kernel slightly smaller than wall thickness
  destroys every element thinner than walls while preserving wall blobs.
  Then a close reconnects any tiny breaks caused by door gaps or scanner noise.
  Finally, small isolated blobs (that somehow survived opening) are removed
  by a connected-component area filter.
"""

from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Dict


# ─── Result ────────────────────────────────────────────────────────────────

@dataclass
class PreprocessResult:
    grayscale:      np.ndarray = field(default=None)
    binary_raw:     np.ndarray = field(default=None)   # after adaptive threshold
    cleaned_binary: np.ndarray = field(default=None)   # structural walls only
    debug: Dict[str, np.ndarray] = field(default_factory=dict)


# ─── Implementation ─────────────────────────────────────────────────────────

def preprocess(bgr: np.ndarray) -> PreprocessResult:
    """
    Extract structural-only wall pixels from a floor plan BGR image.

    The output cleaned_binary contains ONLY thick structural wall pixels.
    Text, furniture, dimension lines, and fixtures are removed.
    """
    result = PreprocessResult()
    debug: Dict[str, np.ndarray] = {}

    # ── Step 1  Grayscale ────────────────────────────────────────────────
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    result.grayscale = gray
    debug["01_grayscale"] = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # ── Step 2  Thresholding ──────────────────────────────────────────────
    # We use adaptive thresholding to find all edges, BUT we also apply a hard global
    # threshold to explicitly remove light gray lines (like door thresholds, pixel val ~200)
    # while preserving solid black walls and windows.
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    adaptive_binary = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=15,
        C=4,
    )
    _, hard_binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    binary = cv2.bitwise_and(adaptive_binary, hard_binary)
    result.binary_raw = binary
    debug["01_binary"] = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    # ── Step 3  Remove tiny blobs (isolated text dots, scanner specks) ───
    # Connected components smaller than this are noise.
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    min_blob_px = 80
    no_tiny = np.zeros_like(binary)
    for i in range(1, n_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_blob_px:
            no_tiny[labels == i] = 255
    debug["03_no_tiny_blobs"] = cv2.cvtColor(no_tiny, cv2.COLOR_GRAY2BGR)

    # ── Step 4  Morphological OPEN  (destroy thin elements) ───────────────
    # Walls in floor plans are typically 3-10 px thick.
    # A 3x3 opening erodes by 1 px → elements thinner than ~2 px are destroyed.
    # Text strokes (1-2 px) and dimension lines (1 px) → gone.
    # Door arcs (thin curves, 1-2 px) → gone.
    # Walls (≥3 px core remaining) → survive, then dilate back.
    open_k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    structural = cv2.morphologyEx(no_tiny, cv2.MORPH_OPEN, open_k)
    debug["04_after_open"] = cv2.cvtColor(structural, cv2.COLOR_GRAY2BGR)

    # ── Step 5  Final cleanup ─────────────────────────────────────────────
    # (Removed 7x7 close because it bridges small valid door gaps)

    # ── Step 6  Second area filter (remove any surviving thin fragments) ──
    n2, labels2, stats2, _ = cv2.connectedComponentsWithStats(
        structural, connectivity=8
    )
    final_mask = np.zeros_like(structural)
    for i in range(1, n2):
        # We use a small threshold (50) here because removing door lines
        # breaks the walls into isolated segments, some of which are short.
        if stats2[i, cv2.CC_STAT_AREA] >= 50:
            final_mask[labels2 == i] = 255
    result.cleaned_binary = final_mask
    debug["06_cleaned_binary"] = cv2.cvtColor(final_mask, cv2.COLOR_GRAY2BGR)

    result.debug = debug
    return result
