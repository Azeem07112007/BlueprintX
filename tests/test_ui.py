"""
tests/test_ui.py
Unit tests for BlueprintX UI logic and launch handlers.
"""
from unittest.mock import MagicMock, patch
import pytest

from backend.models.dataclasses import Building, Building3D, Wall, Room
from pipeline import PipelineResult


def test_launch_unity_imports(tmp_path):
    """Test that _on_launch_unity runs without NameError (ObjExporter defined)."""
    with patch("PySide6.QtWidgets.QMainWindow.__init__", return_value=None):
        from ui.main_window import BlueprintXWindow
        win = BlueprintXWindow.__new__(BlueprintXWindow)

        # Mock result and status_badge
        bld = Building(source_path="dummy.png", image_size=(100, 100))
        bld.walls = [Wall(start=(0, 0), end=(100, 0), wall_type="outer")]
        bld.rooms = [Room(polygon_pts=[(0, 0), (100, 0), (100, 100), (0, 100)], centroid=(50, 50), area_px=10000, label="Room")]
        bld.model_3d = Building3D(
            vertices=__import__("numpy").zeros((4, 3), dtype=__import__("numpy").float32),
            normals=__import__("numpy").zeros((4, 3), dtype=__import__("numpy").float32),
            colors=__import__("numpy").ones((4, 3), dtype=__import__("numpy").float32),
            indices=__import__("numpy").array([[0, 1, 2]], dtype=__import__("numpy").int32),
            vertex_count=4,
            triangle_count=1,
        )
        win._result = PipelineResult(
            image_path="dummy.png",
            original_bgr=None,
            building=bld,
            debug_images={},
            renderer=MagicMock(),
        )
        win.status_badge = MagicMock()
        win.progress_msg = MagicMock()

        # Mock subprocess and shutil to prevent actual launches during test
        with patch("subprocess.Popen"), patch("shutil.copy"):
            win._on_launch_unity()
            assert win.status_badge.set_state.called
