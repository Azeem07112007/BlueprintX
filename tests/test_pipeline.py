"""
tests/test_pipeline.py
Blueprint-X — Integration tests for the full pipeline.

Run with:
    D:\\BlueprintX\\venv313\\Scripts\\python.exe -m pytest tests/ -v
"""
import json
import os
import sys
from pathlib import Path

import pytest

# Ensure project root is on sys.path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


SAMPLE_IMAGE = ROOT / "sample_floorplan.png"


@pytest.fixture(scope="module")
def pipeline_result():
    """Run the full pipeline once and share the result across tests."""
    from pipeline import BlueprintPipeline
    p = BlueprintPipeline()
    return p.run(str(SAMPLE_IMAGE))


# ── Smoke tests ───────────────────────────────────────────────────────────────

class TestPipelineSmoke:
    def test_sample_image_exists(self):
        assert SAMPLE_IMAGE.exists(), f"sample_floorplan.png not found at {SAMPLE_IMAGE}"

    def test_pipeline_completes(self, pipeline_result):
        """Pipeline must complete without raising an exception."""
        assert pipeline_result is not None

    def test_building_not_empty(self, pipeline_result):
        b = pipeline_result.building
        assert b is not None
        assert b.source_path != ""

    def test_image_size_set(self, pipeline_result):
        w, h = pipeline_result.building.image_size
        assert w > 0 and h > 0

    def test_walls_detected(self, pipeline_result):
        walls = pipeline_result.building.walls
        assert len(walls) > 0, "Expected at least one wall segment"

    def test_rooms_detected(self, pipeline_result):
        rooms = pipeline_result.building.rooms
        assert len(rooms) > 0, "Expected at least one room"

    def test_walls_have_valid_geometry(self, pipeline_result):
        for wall in pipeline_result.building.walls:
            sx, sy = wall.start
            ex, ey = wall.end
            assert wall.length > 0, f"Wall has zero length: {wall}"
            assert wall.wall_type in ("outer", "internal"), f"Invalid wall type: {wall.wall_type}"

    def test_rooms_have_valid_geometry(self, pipeline_result):
        for room in pipeline_result.building.rooms:
            assert room.area_px > 0, f"Room {room.room_id} has zero area"
            assert len(room.polygon_pts) >= 3, "Room polygon must have at least 3 points"

    def test_rooms_have_labels(self, pipeline_result):
        """Every room must have a non-empty label (from OCR or heuristic fallback)."""
        for room in pipeline_result.building.rooms:
            assert room.label, f"Room {room.room_id} (area={room.area_px:.0f}) has no label"


    def test_debug_images_populated(self, pipeline_result):
        required_keys = ["01_grayscale", "05_binary", "08_cleaned_binary",
                         "wd_01_skeleton", "wd_02_hough", "wd_03_walls",
                         "ai_01_detections", "ocr_01_mask"]
        for k in required_keys:
            assert k in pipeline_result.debug_images, f"Missing debug image: {k}"
            assert pipeline_result.debug_images[k] is not None


# ── Export tests ──────────────────────────────────────────────────────────────

class TestExport:
    def test_export_produces_valid_json(self, pipeline_result, tmp_path):
        from backend.engine.export.exporter import Exporter
        out = tmp_path / "test_export.json"
        Exporter().export(pipeline_result.building, str(out))
        assert out.exists()
        data = json.loads(out.read_text(encoding="utf-8"))
        assert isinstance(data, dict)

    def test_export_schema_version(self, pipeline_result, tmp_path):
        from backend.engine.export.exporter import Exporter
        out = tmp_path / "export.json"
        Exporter().export(pipeline_result.building, str(out))
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["schema_version"] == "2.1.0"
        assert data["milestone"] == 5
        assert data["units"] == "meters"

    def test_export_has_required_keys(self, pipeline_result, tmp_path):
        from backend.engine.export.exporter import Exporter
        out = tmp_path / "export.json"
        Exporter().export(pipeline_result.building, str(out))
        data = json.loads(out.read_text(encoding="utf-8"))
        for key in ("walls", "rooms", "doors", "topology", "metadata", "statistics", "coordinate_system", "player_spawn", "validation"):
            assert key in data, f"Missing key in export: {key}"

    def test_export_walls_list(self, pipeline_result, tmp_path):
        from backend.engine.export.exporter import Exporter
        out = tmp_path / "export.json"
        Exporter().export(pipeline_result.building, str(out))
        data = json.loads(out.read_text(encoding="utf-8"))
        assert isinstance(data["walls"], list)
        if data["walls"]:
            wall = data["walls"][0]
            assert "start" in wall and "end" in wall
            assert "wall_type" in wall
            assert "length_px" in wall

    def test_export_rooms_list(self, pipeline_result, tmp_path):
        from backend.engine.export.exporter import Exporter
        out = tmp_path / "export.json"
        Exporter().export(pipeline_result.building, str(out))
        data = json.loads(out.read_text(encoding="utf-8"))
        assert isinstance(data["rooms"], list)
        if data["rooms"]:
            room = data["rooms"][0]
            assert "centroid" in room
            assert "area_px" in room
            assert "polygon_points" in room

    def test_export_topology_has_nodes(self, pipeline_result, tmp_path):
        from backend.engine.export.exporter import Exporter
        out = tmp_path / "export.json"
        Exporter().export(pipeline_result.building, str(out))
        data = json.loads(out.read_text(encoding="utf-8"))
        assert "nodes" in data["topology"]
        assert "edges" in data["topology"]

    def test_export_statistics_block(self, pipeline_result, tmp_path):
        from backend.engine.export.exporter import Exporter
        out = tmp_path / "export.json"
        Exporter().export(pipeline_result.building, str(out))
        data = json.loads(out.read_text(encoding="utf-8"))
        stats = data["statistics"]
        assert "total_walls" in stats
        assert "total_rooms" in stats
        assert stats["total_walls"] == len(pipeline_result.building.walls)
        assert stats["total_rooms"] == len(pipeline_result.building.rooms)


# ── Blueprint Learning Rules tests ────────────────────────────────────────────

class TestBlueprintRules:
    def test_ocr_window_label_detection(self):
        """Verify that OCR text blocks matching window patterns are flagged as window labels."""
        from backend.models.dataclasses import TextBlock
        tb = TextBlock(text="3'x4' WINDOW", center=(100.0, 100.0), is_window_label=True)
        assert tb.is_window_label is True

    def test_explicit_windows_preserved(self):
        """Verify that when explicit windows exist, WindowSynthesizer does not add synthesized ones."""
        from backend.models.dataclasses import Building, Wall, Window, TextBlock
        from backend.engine.detection.window_synthesizer import WindowSynthesizer

        b = Building()
        b.walls = [
            Wall(start=(0, 0), end=(200, 0), wall_type="outer"),
            Wall(start=(200, 0), end=(200, 200), wall_type="outer"),
            Wall(start=(200, 200), end=(0, 200), wall_type="outer"),
            Wall(start=(0, 200), end=(0, 0), wall_type="outer"),
        ]
        b.windows = [
            Window(center=(100, 0), width_px=40, height_px=10, source="explicit_blueprint")
        ]

        wins = WindowSynthesizer.synthesize_windows(b)
        assert len(wins) == 1
        assert wins[0].source == "explicit_blueprint"

    def test_ocr_annotated_windows_used(self):
        """Verify explicit windows from OCR annotations are extracted and used without extra auto-synthesis."""
        from backend.models.dataclasses import Building, Wall, TextBlock
        from backend.engine.detection.window_synthesizer import WindowSynthesizer

        b = Building()
        b.walls = [
            Wall(start=(0, 0), end=(200, 0), wall_type="outer"),
            Wall(start=(200, 0), end=(200, 200), wall_type="outer"),
            Wall(start=(200, 200), end=(0, 200), wall_type="outer"),
            Wall(start=(0, 200), end=(0, 0), wall_type="outer"),
        ]
        b.text_blocks = [
            TextBlock(text="3'x4' WINDOW", center=(100, 5), bbox=(80, 0, 40, 10), is_window_label=True)
        ]

        wins = WindowSynthesizer.synthesize_windows(b)
        assert len(wins) == 1
        assert wins[0].source == "ocr_annotation"

    def test_enclosed_building_mesh(self, pipeline_result):
        """Verify that the generated 3D model forms an enclosed building mesh."""
        model_3d = pipeline_result.building.model_3d
        assert model_3d is not None
        assert model_3d.vertex_count > 0
        assert model_3d.triangle_count > 0
        # Verify bounding box is 3D enclosed (non-zero extents in X, Y, Z)
        bmin, bmax = model_3d.bounds_min, model_3d.bounds_max
        assert (bmax[0] - bmin[0]) > 1.0   # Width X
        assert (bmax[1] - bmin[1]) > 1.0   # Height Y
        assert (bmax[2] - bmin[2]) > 1.0   # Depth Z

    def test_only_main_entrance_door_retained(self):
        """Verify that automatically determined Main Entrance Door is retained and internal doors are empty."""
        from backend.models.dataclasses import Building, Wall, Door, Room, TextBlock
        from backend.engine.model3d.geometry_validator import GeometryValidator

        b = Building()
        b.walls = [
            Wall(start=(0, 0), end=(200, 0), wall_type="outer"),
            Wall(start=(200, 0), end=(200, 200), wall_type="outer"),
            Wall(start=(200, 200), end=(0, 200), wall_type="outer"),
            Wall(start=(0, 200), end=(0, 0), wall_type="outer"),
        ]
        b.rooms = [
            Room(room_id=1, polygon_pts=[(5, 5), (195, 5), (195, 195), (5, 195)])
        ]
        b.doors = [
            Door(center=(50, 100), source="internal_door"),
            Door(center=(100, 200), source="front_door"),
        ]
        b.text_blocks = [
            TextBlock(text="FRONT DOOR", center=(100, 195))
        ]

        b = GeometryValidator().pass7_ensure_room_accessibility_and_main_entrance(b)
        assert len(b.doors) == 1
        assert b.doors[0].source == "main_entrance"
        assert b.doors[0].center == (100, 200)
