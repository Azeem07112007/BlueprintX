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


def test_ui_customization_handlers():
    """Test that furniture toggle, wall width, and color setters execute and update state."""
    with patch("PySide6.QtWidgets.QMainWindow.__init__", return_value=None):
        from ui.main_window import BlueprintXWindow
        win = BlueprintXWindow.__new__(BlueprintXWindow)

        bld = Building(source_path="dummy.png", image_size=(100, 100))
        bld.walls = [Wall(start=(0, 0), end=(100, 0), wall_type="outer")]
        bld.rooms = [Room(polygon_pts=[(0, 0), (100, 0), (100, 100), (0, 100)], centroid=(50, 50), area_px=10000, label="Room")]

        win._result = PipelineResult(
            image_path="dummy.png",
            original_bgr=__import__("numpy").zeros((100, 100, 3), dtype=__import__("numpy").uint8),
            building=bld,
            debug_images={},
            renderer=MagicMock(),
        )
        win.status_badge = MagicMock()
        win.progress_msg = MagicMock()
        win.stat_verts = MagicMock()
        win.stat_tris = MagicMock()
        win._viewer3d = None
        win.viewer = MagicMock()
        win.tabs = MagicMock()
        win.tabs.currentIndex.return_value = 0

        win.slider_floor_h = MagicMock()
        win.slider_floor_h.value.return_value = 28
        win.slider_wall_w = MagicMock()
        win.slider_wall_w.value.return_value = 35
        win.chk_ceiling = MagicMock()
        win.chk_ceiling.isChecked.return_value = False

        win.btn_toggle_furniture = MagicMock()
        win._lbl_wall_w = MagicMock()
        win.lbl_wall_color = MagicMock()
        win.color_swatch = MagicMock()

        win._include_furniture = True
        win._custom_wall_color_rgb = None
        win._custom_wall_color_hex = None

        # Test furniture toggle
        win._on_toggle_furniture()
        assert win._include_furniture is False
        assert win.status_badge.set_state.called

        win._on_toggle_furniture()
        assert win._include_furniture is True

        # Test wall width handler
        win._on_wall_width_changed(45)
        assert win._lbl_wall_w.setText.called

        # Test wall color setting
        win._set_wall_color(0.5, 0.6, 0.7, "#809AB3", "Slate")
        assert win._custom_wall_color_rgb == (0.5, 0.6, 0.7)
        assert win._custom_wall_color_hex == "#809AB3"
        assert win.lbl_wall_color.setText.called

        # Test reset wall color
        win._on_reset_wall_color()
        assert win._custom_wall_color_rgb is None
        assert win._custom_wall_color_hex is None
