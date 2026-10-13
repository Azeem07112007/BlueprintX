"""
tests/test_furniture.py
Unit tests for HeuristicFurnitureDetector and 3D furniture mesh generation.
"""
import pytest
import numpy as np

from backend.models.dataclasses import Building, Room, Furniture, Stair, TextBlock, Door, Window
from backend.engine.detection.heuristic_furniture_detector import HeuristicFurnitureDetector
from backend.engine.model3d.furniture_builder import Furniture3DBuilder
from backend.engine.model3d.builder import Model3DBuilder


class TestHeuristicFurnitureDetector:
    def test_detector_initialization(self):
        detector = HeuristicFurnitureDetector()
        assert detector.min_area_px == 400.0

    def test_detection_in_bedroom(self):
        detector = HeuristicFurnitureDetector()
        h, w = 400, 400
        bgr = np.full((h, w, 3), 255, dtype=np.uint8)
        wall_mask = np.zeros((h, w), dtype=np.uint8)

        # Draw a synthetic bed inside a bedroom region
        cv2_bed_x, cv2_bed_y, cv2_bed_w, cv2_bed_h = 100, 100, 80, 100
        wall_mask[cv2_bed_y:cv2_bed_y+cv2_bed_h, cv2_bed_x:cv2_bed_x+cv2_bed_w] = 0

        # Create room polygon encompassing bed
        room = Room(
            room_id=1,
            polygon_pts=[(50, 50), (250, 50), (250, 250), (50, 250)],
            centroid=(150, 150),
            area_px=40000,
            label="Bedroom",
        )

        items, stairs = detector.run(
            bgr=bgr,
            wall_mask=wall_mask,
            rooms=[room],
            text_blocks=[],
            doors=[],
            windows=[],
        )
        assert isinstance(items, list)
        assert isinstance(stairs, list)

    def test_merge_detections_deduplication(self):
        ai_furn = [Furniture(class_name="sofa", center=(100.0, 100.0), bbox=(80, 80, 40, 40))]
        h_furn = [
            Furniture(class_name="sofa", center=(102.0, 101.0), bbox=(81, 81, 40, 40)),
            Furniture(class_name="table", center=(250.0, 250.0), bbox=(230, 230, 40, 40)),
        ]

        merged = HeuristicFurnitureDetector.merge_detections(ai_furn, h_furn, distance_threshold_px=30.0)
        assert len(merged) == 2  # duplicate sofa suppressed, table kept
        assert merged[0].class_name == "sofa"
        assert merged[1].class_name == "table"


class TestFurniture3DBuilder:
    def test_build_bed_3d(self):
        furn = Furniture(class_name="bed", bbox=(10, 10, 50, 60))
        def dummy_to_world(px, py, elev):
            return np.array([px / 100.0, elev, py / 100.0], dtype=np.float32)

        v, n, c, idx, base = Furniture3DBuilder.build_furniture(furn, dummy_to_world, 100.0, 0)
        assert len(v) > 0
        assert len(n) == len(v)
        assert len(c) == len(v)
        assert len(idx) > 0
        assert base > 0

    def test_build_sofa_3d(self):
        furn = Furniture(class_name="sofa", bbox=(20, 20, 80, 40))
        def dummy_to_world(px, py, elev):
            return np.array([px / 100.0, elev, py / 100.0], dtype=np.float32)

        v, n, c, idx, base = Furniture3DBuilder.build_furniture(furn, dummy_to_world, 100.0, 0)
        assert len(v) > 0
        assert len(idx) > 0

    def test_build_toilet_3d(self):
        furn = Furniture(class_name="toilet", bbox=(5, 5, 20, 25))
        def dummy_to_world(px, py, elev):
            return np.array([px / 100.0, elev, py / 100.0], dtype=np.float32)

        v, n, c, idx, base = Furniture3DBuilder.build_furniture(furn, dummy_to_world, 100.0, 0)
        assert len(v) > 0
        assert len(idx) > 0

    def test_build_staircase_3d(self):
        stair = Stair(bbox=(0, 0, 40, 100))
        def dummy_to_world(px, py, elev):
            return np.array([px / 100.0, elev, py / 100.0], dtype=np.float32)

        v, n, c, idx, base = Furniture3DBuilder.build_staircase(stair, dummy_to_world, 2.8, 0)
        assert len(v) > 0
        assert len(idx) > 0

    def test_build_balcony_railing(self):
        room = Room(
            room_id=1,
            polygon_pts=[(0, 0), (100, 0), (100, 100), (0, 100)],
            label="Balcony",
        )
        def dummy_to_world(px, py, elev):
            return np.array([px / 100.0, elev, py / 100.0], dtype=np.float32)

        v, n, c, idx, base = Furniture3DBuilder.build_balcony_railing(room, dummy_to_world, 1.05, 0)
        assert len(v) > 0
        assert len(idx) > 0
