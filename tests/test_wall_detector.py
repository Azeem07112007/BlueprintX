"""
tests/test_wall_detector.py
Blueprint-X — Unit tests for WallDetector and Preprocessor.
"""
import sys
import math
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from backend.engine.detection.wall_detector import WallDetector
from backend.engine.preprocessing.preprocessor import Preprocessor
from backend.models.dataclasses import Wall


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_synthetic_binary(width: int = 400, height: int = 300) -> np.ndarray:
    """Create a simple binary floor plan: outer rectangle + one internal wall."""
    img = np.zeros((height, width), dtype=np.uint8)
    # Outer walls (thick lines)
    cv2.rectangle(img, (20, 20), (width - 20, height - 20), 255, 8)
    # One internal wall splitting the room horizontally
    cv2.line(img, (20, height // 2), (width - 20, height // 2), 255, 5)
    return img


def _make_synthetic_bgr(width: int = 400, height: int = 300) -> np.ndarray:
    """Grey BGR image matching the synthetic binary."""
    gray = _make_synthetic_binary(width, height)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


# ── Preprocessor tests ────────────────────────────────────────────────────────

class TestPreprocessor:
    def setup_method(self):
        self.pp = Preprocessor()

    def test_run_returns_result(self):
        bgr = _make_synthetic_bgr()
        res = self.pp.run(bgr)
        assert res is not None

    def test_grayscale_shape(self):
        bgr = _make_synthetic_bgr(320, 240)
        res = self.pp.run(bgr)
        assert res.grayscale is not None
        assert res.grayscale.ndim == 2

    def test_cleaned_binary_not_none(self):
        bgr = _make_synthetic_bgr()
        res = self.pp.run(bgr)
        assert res.cleaned_binary is not None

    def test_binary_has_white_pixels(self):
        """A synthetic floor plan must have white (wall) pixels in cleaned binary."""
        bgr = _make_synthetic_bgr()
        res = self.pp.run(bgr)
        white_px = np.sum(res.cleaned_binary == 255)
        assert white_px > 100, "Cleaned binary should contain wall pixels"

    def test_scale_factor_lte_one_for_small_image(self):
        """Images smaller than MAX_DIM should not be upscaled (scale_factor == 1.0)."""
        bgr = _make_synthetic_bgr(400, 300)
        res = self.pp.run(bgr)
        assert res.scale_factor == 1.0

    def test_blueprint_style_is_valid(self):
        bgr = _make_synthetic_bgr()
        res = self.pp.run(bgr)
        assert res.blueprint_style in ("simple", "rendered")

    def test_debug_images_populated(self):
        bgr = _make_synthetic_bgr()
        res = self.pp.run(bgr)
        required = ["01_grayscale", "02_deskewed", "05_binary", "08_cleaned_binary"]
        for key in required:
            assert key in res.debug, f"Missing debug image: {key}"


# ── WallDetector tests ────────────────────────────────────────────────────────

class TestWallDetector:
    def setup_method(self):
        self.wd = WallDetector()

    def test_run_returns_result(self):
        binary = _make_synthetic_binary()
        bgr    = _make_synthetic_bgr()
        res = self.wd.run(binary, bgr)
        assert res is not None

    def test_detects_walls(self):
        binary = _make_synthetic_binary()
        bgr    = _make_synthetic_bgr()
        res = self.wd.run(binary, bgr)
        assert len(res.walls) > 0, "Should detect walls in synthetic plan"

    def test_walls_have_positive_length(self):
        binary = _make_synthetic_binary()
        bgr    = _make_synthetic_bgr()
        res = self.wd.run(binary, bgr)
        for wall in res.walls:
            assert wall.length > 0, f"Wall has zero length: {wall}"

    def test_wall_type_valid(self):
        binary = _make_synthetic_binary()
        bgr    = _make_synthetic_bgr()
        res = self.wd.run(binary, bgr)
        for wall in res.walls:
            assert wall.wall_type in ("outer", "internal")

    def test_wall_thickness_positive(self):
        binary = _make_synthetic_binary()
        bgr    = _make_synthetic_bgr()
        res = self.wd.run(binary, bgr)
        for wall in res.walls:
            assert wall.thickness_px > 0, "Thickness must be positive"
            assert wall.thickness_px <= 40, "Thickness should not exceed 40 px"

    def test_no_walls_on_blank_image(self):
        binary = np.zeros((300, 400), dtype=np.uint8)
        bgr    = np.zeros((300, 400, 3), dtype=np.uint8)
        res = self.wd.run(binary, bgr)
        assert len(res.walls) == 0, "Blank image should yield no walls"

    def test_debug_images_present(self):
        binary = _make_synthetic_binary()
        bgr    = _make_synthetic_bgr()
        res = self.wd.run(binary, bgr)
        for key in ("wd_01_skeleton", "wd_02_hough", "wd_03_walls"):
            assert key in res.debug

    def test_estimate_thickness_sanity(self):
        """The thickness estimator should return ~8 px for a synthetic 8-px thick wall."""
        binary = np.zeros((200, 200), dtype=np.uint8)
        cv2.line(binary, (10, 100), (190, 100), 255, 8)
        t = self.wd._estimate_thickness(binary, 10, 100, 190, 100, 200, 200)
        # Should be close to 8 (within +/- 3)
        assert 5 <= t <= 11, f"Expected ~8 px, got {t}"

    def test_door_gaps_detected(self):
        """A gap in a long wall should be flagged as a door gap."""
        binary = np.zeros((200, 600), dtype=np.uint8)
        cv2.line(binary, (20, 100), (270, 100), 255, 5)   # left segment
        cv2.line(binary, (320, 100), (580, 100), 255, 5)  # right segment (50 px gap)
        bgr = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        res = self.wd.run(binary, bgr)
        assert len(res.door_gaps) >= 1, "Should detect the gap as a door"


# ── Wall dataclass tests ──────────────────────────────────────────────────────

class TestWallDataclass:
    def test_length_horizontal(self):
        w = Wall(start=(0.0, 0.0), end=(100.0, 0.0))
        assert abs(w.length - 100.0) < 1e-6

    def test_length_diagonal(self):
        w = Wall(start=(0.0, 0.0), end=(3.0, 4.0))
        assert abs(w.length - 5.0) < 1e-6

    def test_midpoint(self):
        w = Wall(start=(0.0, 0.0), end=(100.0, 100.0))
        assert w.midpoint == (50.0, 50.0)

    def test_angle_horizontal(self):
        w = Wall(start=(0.0, 0.0), end=(100.0, 0.0))
        assert abs(w.angle_deg - 0.0) < 1e-6

    def test_angle_vertical(self):
        w = Wall(start=(0.0, 0.0), end=(0.0, 100.0))
        assert abs(w.angle_deg - 90.0) < 1e-6

    def test_to_dict_keys(self):
        w = Wall(start=(10.0, 20.0), end=(110.0, 20.0), wall_type="outer")
        d = w.to_dict()
        for key in ("id", "start", "end", "length_px", "thickness_px", "angle_deg", "wall_type", "confidence"):
            assert key in d, f"Missing key in Wall.to_dict(): {key}"
