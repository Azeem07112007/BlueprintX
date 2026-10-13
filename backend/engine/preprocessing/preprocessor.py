"""
backend/engine/preprocessing/preprocessor.py
Blueprint-X — Milestone 2+

Preprocesses a raw blueprint image into a clean structural binary mask.

Two adaptive pipelines are selected automatically based on blueprint style:

  ── Simple (line-drawing) pipeline ──────────────────────────────────────────
  Classic black-on-white architectural drawings.
  1. Grayscale
  2. Deskew (Hough-angle correction)
  3. Normalize brightness (CLAHE)
  4. Resize to standard resolution
  5. Dual threshold (adaptive + hard global)
  6. Remove tiny blobs
  7. Morphological open (destroy thin elements)
  8. Second area filter

  ── Rendered (color floor plan) pipeline ────────────────────────────────────
  Professionally rendered plans with light floors, colored furniture, dark
  grey walls (~#3a3a3a to #606060) on a cream/light background.
  1. Grayscale + Deskew + Resize  (same)
  2. Dark-pixel isolation  (L-channel < threshold in LAB space)
  3. Canny edge detection on grayscale
  4. Edge + dark-pixel union
  5. Large morphological close to connect broken wall segments
  6. Remove tiny blobs + area filter
"""
from __future__ import annotations

import cv2
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple


@dataclass
class PreprocessResult:
    grayscale: Optional[np.ndarray] = None
    binary_raw: Optional[np.ndarray] = None
    cleaned_binary: Optional[np.ndarray] = None   # structural walls only (for Hough)
    room_seg_mask: Optional[np.ndarray] = None     # pixel-level wall mask for room flood-fill
    scale_factor: float = 1.0                      # resize ratio applied
    skew_angle_deg: float = 0.0
    blueprint_style: str = "simple"                # "simple" | "rendered"
    debug: Dict[str, np.ndarray] = field(default_factory=dict)


class Preprocessor:
    """
    Stateless adaptive preprocessor. Call .run(bgr) to get a PreprocessResult.
    Automatically chooses between the simple and rendered pipelines.
    """

    # ── Shared configuration ─────────────────────────────────────────────────
    MAX_DIM = 2048

    # ── Simple pipeline config ───────────────────────────────────────────────
    ADAPTIVE_BLOCK   = 15
    ADAPTIVE_C       = 4
    HARD_THRESH      = 150   # pixel values ABOVE this → background (bright = white paper)
    MIN_BLOB_PX      = 80
    OPEN_KERNEL      = 3
    MIN_STRUCT_PX    = 50

    # ── Rendered pipeline config ─────────────────────────────────────────────
    # Walls in rendered plans are typically dark grey (L < 90 in LAB)
    RENDERED_L_THRESH        = 90    # LAB L channel: < this means "dark" = wall candidate
    RENDERED_CANNY_LO        = 30
    RENDERED_CANNY_HI        = 120
    RENDERED_CLOSE_KERNEL    = 5     # morphological close to bridge edge gaps
    RENDERED_MIN_BLOB_PX     = 150
    RENDERED_MIN_STRUCT_PX   = 100
    RENDERED_OPEN_KERNEL     = 2     # smaller open so we don't erase thick walls

    # ── Style-detection thresholds ───────────────────────────────────────────
    # A plan is "rendered" if it has genuine colour variation between channels
    # (coloured furniture/walls) OR a distinctly light background that is NOT
    # pure greyscale.
    #
    # Simple line drawings are nearly pure greyscale (color_std ≈ 0, pixel-level
    # BGR channels all very close) on a near-white background.  We keep them in
    # the fast "simple" pipeline.
    COLORFUL_STD_THRESH    = 12.0   # std of per-channel MEAN across B,G,R channels
    PIXEL_COLOR_STD_THRESH = 15.0   # mean per-pixel std across BGR — true colour test
    LIGHT_BG_MEDIAN_THRESH = 180    # median grey value > this → likely light bg
    DARK_STRUCT_FRAC       = 0.03   # at least 3% of pixels must be "wall-dark"

    def run(self, bgr: np.ndarray) -> PreprocessResult:
        result = PreprocessResult()
        debug: Dict[str, np.ndarray] = {}

        # ── Step 1  Grayscale ─────────────────────────────────────────────
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        result.grayscale = gray
        debug["01_grayscale"] = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # ── Step 2  Deskew ────────────────────────────────────────────────
        gray, angle = self._deskew(gray)
        result.skew_angle_deg = angle
        debug["02_deskewed"] = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # ── Step 3  CLAHE brightness normalization ────────────────────────────────────────
        # Use a smaller tile for complex/high-density images so local contrast
        # enhancement doesn't blur together adjacent thin walls.
        tile = (6, 6) if self._is_complex_density(gray) else (8, 8)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=tile)
        gray_norm = clahe.apply(gray)
        debug["03_normalized"] = cv2.cvtColor(gray_norm, cv2.COLOR_GRAY2BGR)

        # ── Step 4  Resize to standard resolution ─────────────────────────
        gray_norm, scale = self._resize(gray_norm)
        gray_resized, _ = self._resize(gray)          # also resize original gray
        bgr_resized, _ = self._resize_bgr(bgr)
        result.scale_factor = scale
        debug["04_resized"] = cv2.cvtColor(gray_norm, cv2.COLOR_GRAY2BGR)

        # ── Step 5  Detect blueprint style ────────────────────────────────
        style = self._detect_style(bgr_resized, gray_resized)
        result.blueprint_style = style

        # ── Step 6  Branch on style ───────────────────────────────────────
        if style == "rendered":
            binary, cleaned = self._pipeline_rendered(bgr_resized, gray_norm, debug)
        else:
            binary, cleaned = self._pipeline_simple(gray_norm, debug)

        result.binary_raw = binary
        result.cleaned_binary = cleaned

        # ── Step 7  Room segmentation mask ───────────────────────────────────
        # This is a SEPARATE mask optimised for room flood-fill.
        # We bypass the morphological-open / area-filter steps that destroy
        # thin internal walls, and instead use a direct pixel threshold + dilation.
        result.room_seg_mask = self._make_room_seg_mask(
            bgr_resized, gray_norm, style
        )
        debug["09_room_seg_mask"] = cv2.cvtColor(result.room_seg_mask, cv2.COLOR_GRAY2BGR)

        result.debug = debug
        return result

    # ── Style Detection ───────────────────────────────────────────────────────

    def _detect_style(self, bgr: np.ndarray, gray: np.ndarray) -> str:
        """Heuristically decide if this is a 'rendered' colour floor plan or a
        'simple' line-drawing blueprint.

        Key insight:
          - Simple line drawings are PURE GREYSCALE — B, G, R values are
            nearly identical at every pixel (per-pixel std ≈ 0).
          - Rendered colour plans have coloured furniture, tinted floors and
            walls — individual pixels have divergent BGR values (per-pixel
            std is high even if the global mean per channel is similar).

        Detection logic:
          A) High per-pixel colour std → definitely rendered (has genuine colour)
          B) High channel-mean std → rendered (global colour tint)
          C) EITHER A or B, BUT only when the image is NOT near-pure-greyscale
        """
        b, g, r = cv2.split(bgr.astype(np.float32))

        # Per-channel global means
        channel_means = [float(b.mean()), float(g.mean()), float(r.mean())]
        color_std = float(np.std(channel_means))

        # Per-pixel colour std (mean of std across B/G/R at each pixel)
        # Stack channels → shape (H, W, 3), compute std per pixel
        pixel_stack = np.stack([b, g, r], axis=2)
        pixel_color_std = float(pixel_stack.std(axis=2).mean())

        median_lum = float(np.median(gray))
        dark_frac = float(np.sum(gray < 80)) / gray.size

        # A plan is "rendered" if it has genuine pixel-level colour variation
        has_color = (
            pixel_color_std > self.PIXEL_COLOR_STD_THRESH   # true per-pixel colour
            or color_std > self.COLORFUL_STD_THRESH          # global tint
        )
        # Also flag as rendered if it has a light background with meaningful
        # dark structure, BUT ONLY when some colour is present (not pure grey)
        has_light_bg_with_structure = (
            median_lum > self.LIGHT_BG_MEDIAN_THRESH
            and dark_frac > self.DARK_STRUCT_FRAC
            and pixel_color_std > 5.0   # must have at least a hint of colour
        )

        is_rendered = has_color or has_light_bg_with_structure
        return "rendered" if is_rendered else "simple"

    # ── Simple Pipeline ───────────────────────────────────────────────────────

    def _is_complex_density(self, gray: np.ndarray) -> bool:
        """Return True if the image has high structural line density."""
        # Quick Canny to estimate edge density
        edges = cv2.Canny(gray, 40, 120)
        density = float(np.count_nonzero(edges)) / gray.size
        return density > 0.06

    def _pipeline_simple(
        self,
        gray: np.ndarray,
        debug: Dict,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Classic adaptive-threshold pipeline for black-on-white line drawings."""
        is_complex = self._is_complex_density(gray)

        # Slight blur — less blur for complex plans to preserve thin partitions
        blur_k = (3, 3) if is_complex else (5, 5)
        blurred = cv2.GaussianBlur(gray, blur_k, 0)

        # Adaptive block size: smaller for complex so text/hatching doesn't bleed
        block_size = 13 if is_complex else self.ADAPTIVE_BLOCK
        c_val = 3 if is_complex else self.ADAPTIVE_C
        adaptive_binary = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=self.ADAPTIVE_BLOCK,
            C=self.ADAPTIVE_C,
        )
        _, hard_binary = cv2.threshold(gray, self.HARD_THRESH, 255, cv2.THRESH_BINARY_INV)
        binary = cv2.bitwise_and(adaptive_binary, hard_binary)
        debug["05_binary"] = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

        # Remove tiny blobs
        no_tiny = self._area_filter(binary, self.MIN_BLOB_PX)
        debug["06_no_tiny"] = cv2.cvtColor(no_tiny, cv2.COLOR_GRAY2BGR)

        # Morphological open
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (self.OPEN_KERNEL, self.OPEN_KERNEL))
        structural = cv2.morphologyEx(no_tiny, cv2.MORPH_OPEN, k)
        debug["07_after_open"] = cv2.cvtColor(structural, cv2.COLOR_GRAY2BGR)

        # Second area filter
        final = self._area_filter(structural, self.MIN_STRUCT_PX)
        debug["08_cleaned_binary"] = cv2.cvtColor(final, cv2.COLOR_GRAY2BGR)

        return binary, final

    # ── Rendered Pipeline ─────────────────────────────────────────────────────

    def _pipeline_rendered(
        self,
        bgr: np.ndarray,
        gray: np.ndarray,
        debug: Dict,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Multi-cue pipeline for rendered colour floor plans.

        Walls in professional renders are typically:
          • Dark grey in RGB (~#404040)
          • Low L value in LAB colour space  (L < 90)
          • Strong edges (Canny) because of high contrast against light floors

        Strategy:
          1. Dark-pixel mask (LAB L-channel threshold)
          2. Canny edge map
          3. Union of both cues
          4. Morphological close to bridge gaps across wall thickness
          5. Blob area filtering
        """

        # ── Color-Aware Structural Wall Segmentation ─────────────────────────
        from backend.engine.preprocessing.color_plan_segmenter import ColorPlanSegmenter

        final = ColorPlanSegmenter.segment_walls(bgr)
        binary = final.copy()

        debug["05_binary"] = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        debug["06_no_tiny"] = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        debug["07_after_open"] = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        debug["08_cleaned_binary"] = cv2.cvtColor(final, cv2.COLOR_GRAY2BGR)

        return binary, final

    # ── Room Segmentation Mask ────────────────────────────────────────────────

    def _make_room_seg_mask(
        self,
        bgr: np.ndarray,
        gray: np.ndarray,
        style: str,
    ) -> np.ndarray:
        """
        Build a dense, gap-free wall mask specifically for room flood-fill.

        For simple (black-on-white) plans: directly threshold dark pixels
        with Otsu + hard threshold union, then dilate moderately.
        No morphological open/area-filter so thin internal walls are preserved.

        For rendered (colour) plans: use LAB L-channel threshold + Canny union.
        """
        if style == "simple":
            # Direct dark-pixel extraction (no destructive morphological ops)
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            # Otsu global threshold
            _, otsu = cv2.threshold(
                blurred, 0, 255,
                cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
            )
            # Hard threshold for very dark pixels
            _, hard = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY_INV)
            # Adaptive threshold to catch thin light-grey walls
            adapt = cv2.adaptiveThreshold(
                blurred, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                blockSize=11, C=3,
            )
            # Union of all three cues
            raw = cv2.bitwise_or(otsu, hard)
            raw = cv2.bitwise_or(raw, adapt)

            # Remove tiny dots (text, hatching) without killing wall lines
            raw = self._area_filter(raw, min_area=30)

            # Dilate to seal T-junction gaps and corner touches
            k_close = np.ones((9, 9), np.uint8)
            sealed  = cv2.dilate(raw, k_close, iterations=2)
            sealed  = cv2.morphologyEx(sealed, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

        else:
            # Rendered plan: LAB L-channel + Canny
            lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
            l_ch = lab[:, :, 0]
            _, dark = cv2.threshold(l_ch, 90, 255, cv2.THRESH_BINARY_INV)
            edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 30, 120)
            raw = cv2.bitwise_or(dark, edges)
            raw = self._area_filter(raw, min_area=50)
            k_close = np.ones((11, 11), np.uint8)
            sealed  = cv2.dilate(raw, k_close, iterations=2)
            sealed  = cv2.morphologyEx(sealed, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

        return sealed

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _area_filter(self, binary: np.ndarray, min_area: int) -> np.ndarray:
        """Remove connected components smaller than min_area pixels."""
        n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
        out = np.zeros_like(binary)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                out[labels == i] = 255
        return out

    def _deskew(self, gray: np.ndarray) -> Tuple[np.ndarray, float]:
        """Detect dominant line angle and rotate to correct skew."""
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=200)
        if lines is None:
            return gray, 0.0

        angles = []
        # OpenCV 5 changed HoughLines output shape from (N,1,2) to (N,2)
        for rho, theta in lines.reshape(-1, 2):
            angle_deg = math.degrees(theta)
            if angle_deg < 45:
                angles.append(angle_deg)
            elif angle_deg > 135:
                angles.append(angle_deg - 180)

        if not angles:
            return gray, 0.0

        median_angle = float(np.median(angles))
        if abs(median_angle) < 0.5 or abs(median_angle) > 10:
            return gray, 0.0

        h, w = gray.shape
        M = cv2.getRotationMatrix2D((w / 2, h / 2), median_angle, 1.0)
        corrected = cv2.warpAffine(
            gray, M, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return corrected, median_angle

    def _resize(self, gray: np.ndarray) -> Tuple[np.ndarray, float]:
        """Resize so max dimension <= MAX_DIM. Returns (resized, scale_factor)."""
        h, w = gray.shape
        max_dim = max(h, w)
        if max_dim <= self.MAX_DIM:
            return gray, 1.0
        scale = self.MAX_DIM / max_dim
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return resized, scale

    def _resize_bgr(self, bgr: np.ndarray) -> Tuple[np.ndarray, float]:
        """Resize BGR image so max dimension <= MAX_DIM."""
        h, w = bgr.shape[:2]
        max_dim = max(h, w)
        if max_dim <= self.MAX_DIM:
            return bgr, 1.0
        scale = self.MAX_DIM / max_dim
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return resized, scale
