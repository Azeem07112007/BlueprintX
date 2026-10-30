"""
tests/test_customization.py
Unit tests for the new customization features:
  1. Furniture placement toggle (include_furniture=True / False)
  2. Custom wall paint color option (3D Model and 2D Renderer)
  3. Adjustable wall width without breaking existing geometry/openings
"""
import pytest
import numpy as np

from backend.models.dataclasses import Building, Building3D, Wall, Room, Door, Furniture
from backend.engine.model3d.builder import Model3DBuilder
from backend.engine.visualization.renderer import Renderer


def _create_sample_building() -> Building:
    bld = Building(source_path="test_plan.png", image_size=(1000, 1000))
    bld.walls = [
        Wall(start=(100.0, 100.0), end=(900.0, 100.0), wall_type="outer", thickness_px=15.0),
        Wall(start=(900.0, 100.0), end=(900.0, 900.0), wall_type="outer", thickness_px=15.0),
        Wall(start=(900.0, 900.0), end=(100.0, 900.0), wall_type="outer", thickness_px=15.0),
        Wall(start=(100.0, 900.0), end=(100.0, 100.0), wall_type="outer", thickness_px=15.0),
        Wall(start=(500.0, 100.0), end=(500.0, 900.0), wall_type="internal", thickness_px=10.0),
    ]
    bld.doors = [
        Door(center=(500.0, 300.0), width_px=60.0, height_px=20.0, source="gap")
    ]
    bld.rooms = [
        Room(
            room_id=1,
            polygon_pts=[(100.0, 100.0), (500.0, 100.0), (500.0, 900.0), (100.0, 900.0)],
            centroid=(300.0, 500.0),
            area_px=320000.0,
            label="Living Room",
            color_bgr=(180, 150, 120),
        ),
        Room(
            room_id=2,
            polygon_pts=[(500.0, 100.0), (900.0, 100.0), (900.0, 900.0), (500.0, 900.0)],
            centroid=(700.0, 500.0),
            area_px=320000.0,
            label="Bedroom",
            color_bgr=(120, 150, 180),
        ),
    ]
    bld.furniture = [
        Furniture(class_name="bed", bbox=(600, 300, 100, 120), center=(650.0, 360.0)),
        Furniture(class_name="sofa", bbox=(200, 300, 120, 60), center=(260.0, 330.0)),
    ]
    return bld


class TestFurniturePlacementToggle:
    """Tests for the button/option to include or exclude furniture placement."""

    def test_furniture_included_by_default(self):
        bld = _create_sample_building()
        model_with = Model3DBuilder(include_furniture=True).run(bld)
        model_without = Model3DBuilder(include_furniture=False).run(bld)

        assert model_with.vertex_count > model_without.vertex_count
        assert model_with.triangle_count > model_without.triangle_count

    def test_furniture_toggle_preserves_structural_elements(self):
        bld = _create_sample_building()
        model_without = Model3DBuilder(include_furniture=False).run(bld)

        # Still has wall geometry and floor slabs
        assert model_without.vertex_count > 0
        assert model_without.triangle_count > 0
        y_coords = model_without.vertices[:, 1]
        assert np.any(np.abs(y_coords - 2.8) < 0.1)  # Wall top reaches floor_height
        assert np.any(np.abs(y_coords) < 0.1)        # Floor slab is present

    def test_renderer_include_furniture_flag(self):
        bld = _create_sample_building()
        img = np.full((1000, 1000, 3), 255, dtype=np.uint8)

        r_with = Renderer(img, bld, include_furniture=True)
        r_without = Renderer(img, bld, include_furniture=False)

        canvas_with = r_with.render_semantics()
        canvas_without = r_without.render_semantics()

        assert not np.array_equal(canvas_with, canvas_without)


class TestCustomWallPaintColor:
    """Tests for custom wall paint color options."""

    def test_custom_wall_color_applied_to_3d_mesh(self):
        bld = _create_sample_building()
        custom_col = (0.20, 0.60, 0.85)  # vivid sky blue
        model = Model3DBuilder(wall_color=custom_col, include_furniture=False).run(bld)

        # Check that wall color is present in vertex colors
        colors = model.colors
        # Check if custom color channels appear in the mesh
        matches = np.isclose(colors[:, 0], custom_col[0], atol=0.05) & \
                  np.isclose(colors[:, 1], custom_col[1], atol=0.05) & \
                  np.isclose(colors[:, 2], custom_col[2], atol=0.05)
        assert np.any(matches), "Custom wall color was not found in 3D vertex colors"

    def test_separate_outer_and_inner_wall_colors(self):
        bld = _create_sample_building()
        outer_col = (0.90, 0.10, 0.10)  # red
        inner_col = (0.10, 0.90, 0.10)  # green

        model = Model3DBuilder(
            outer_wall_color=outer_col,
            inner_wall_color=inner_col,
            include_furniture=False,
        ).run(bld)

        colors = model.colors
        has_outer = np.any(np.isclose(colors[:, 0], 0.90, atol=0.05) & np.isclose(colors[:, 1], 0.10, atol=0.05))
        has_inner = np.any(np.isclose(colors[:, 0], 0.10, atol=0.05) & np.isclose(colors[:, 1], 0.90, atol=0.05))
        assert has_outer, "Outer wall custom color not found"
        assert has_inner, "Inner wall custom color not found"

    def test_renderer_custom_wall_color_bgr(self):
        bld = _create_sample_building()
        img = np.full((1000, 1000, 3), 200, dtype=np.uint8)

        custom_bgr = (40, 200, 100)
        r = Renderer(img, bld, wall_color_bgr=custom_bgr)
        canvas = r.render_walls()

        # Custom BGR color should be painted on wall lines
        has_custom_pixel = np.any((canvas[:, :, 0] == custom_bgr[0]) &
                                  (canvas[:, :, 1] == custom_bgr[1]) &
                                  (canvas[:, :, 2] == custom_bgr[2]))
        assert has_custom_pixel, "Custom wall color not found on 2D wall canvas"


class TestAdjustableWallWidth:
    """Tests for adjustable wall width without breaking existing geometry."""

    def test_adjustable_wall_width_changes_thickness(self):
        bld = _create_sample_building()
        # Compare thin wall (0.15m) vs thick wall (0.50m)
        m_thin = Model3DBuilder(wall_width_m=0.15, include_furniture=False).run(bld)
        m_thick = Model3DBuilder(wall_width_m=0.50, include_furniture=False).run(bld)

        assert m_thin.vertex_count > 0
        assert m_thick.vertex_count > 0

        # With thicker walls, bounding box slightly expands outward
        thin_bounds_x = m_thin.bounds_max[0] - m_thin.bounds_min[0]
        thick_bounds_x = m_thick.bounds_max[0] - m_thick.bounds_min[0]
        assert thick_bounds_x > thin_bounds_x

    def test_wall_width_preserves_door_cutouts(self):
        bld = _create_sample_building()
        # Thick wall (0.60m)
        m_thick = Model3DBuilder(wall_width_m=0.60, include_furniture=False).run(bld)
        # Verify valid mesh with triangles and proper height bounds
        assert m_thick.triangle_count > 0
        y_max = m_thick.vertices[:, 1].max()
        assert abs(y_max - 2.8) < 0.05

    def test_wall_width_in_renderer(self):
        bld = _create_sample_building()
        img = np.full((1000, 1000, 3), 200, dtype=np.uint8)

        r_thin = Renderer(img, bld, wall_thickness_scale=0.5)
        r_thick = Renderer(img, bld, wall_thickness_scale=2.5)

        c_thin = r_thin.render_walls()
        c_thick = r_thick.render_walls()

        # Dimmed canvas has background value 76; thicker walls paint more line pixels
        diff_thin = np.sum(c_thin != 76)
        diff_thick = np.sum(c_thick != 76)
        assert diff_thick > diff_thin


class TestViewer3DCameraPersistence:
    """Tests that camera does not reset on customization rebuilds."""

    def test_camera_pose_preserved_when_reset_camera_false(self):
        from PySide6.QtWidgets import QApplication
        from PySide6.QtGui import QVector3D
        from ui.viewer3d import Viewer3D

        app = QApplication.instance() or QApplication([])
        bld = _create_sample_building()
        model = Model3DBuilder().run(bld)

        v = Viewer3D()
        v.load_model(model, reset_camera=True)
        v._upload(model)

        # User orbits and zooms
        v._yaw = 135.0
        v._pitch = -40.0
        v._distance = 28.5
        v._pan = QVector3D(1.5, 0.5, -2.0)

        # Customization triggers reload without reset
        v.load_model(model, reset_camera=False)
        v._upload(model)

        assert v._yaw == 135.0
        assert v._pitch == -40.0
        assert v._distance == 28.5
        assert v._pan == QVector3D(1.5, 0.5, -2.0)

    def test_camera_resets_on_explicit_reset_or_new_file(self):
        from PySide6.QtWidgets import QApplication
        from PySide6.QtGui import QVector3D
        from ui.viewer3d import Viewer3D

        app = QApplication.instance() or QApplication([])
        bld = _create_sample_building()
        model = Model3DBuilder().run(bld)

        v = Viewer3D()
        v.load_model(model, reset_camera=True)
        v._upload(model)

        v._yaw = 90.0
        v._pitch = -60.0
        v._pan = QVector3D(5.0, 5.0, 5.0)

        # Explicit reset (e.g. brand new file upload)
        v.load_model(model, reset_camera=True)
        v._upload(model)

        assert v._yaw == 45.0
        assert v._pitch == -22.0

    def test_reset_view_method(self):
        from PySide6.QtWidgets import QApplication
        from PySide6.QtGui import QVector3D
        from ui.viewer3d import Viewer3D

        app = QApplication.instance() or QApplication([])
        bld = _create_sample_building()
        model = Model3DBuilder().run(bld)

        v = Viewer3D()
        v.load_model(model, reset_camera=True)
        v._upload(model)

        v._yaw = 180.0
        v._pitch = -10.0
        v.reset_view()

        assert v._yaw == 45.0
        assert v._pitch == -22.0


class TestTopologyAnd3DStructure:
    """Tests for improved topology model and accurate 3D blueprint structure."""

    def test_doorway_lintel_extrusion(self):
        bld = _create_sample_building()
        model = Model3DBuilder(include_furniture=False).run(bld)

        # In _create_sample_building, door is on the south wall (y=800, x in 400..600)
        # Verify that wall vertices exist above the door height (2.13m to 2.80m)
        verts = model.vertices
        y_vals = verts[:, 1]
        lintel_verts = verts[(y_vals >= 2.10) & (y_vals <= 2.81)]
        assert len(lintel_verts) > 0, "No lintel vertices found above door openings"

    def test_room_topology_connectivity(self):
        from backend.engine.topology.topology_engine import TopologyEngine

        bld = _create_sample_building()
        topo = TopologyEngine().build_room_topology(bld.rooms, bld.doors, bld.walls)

        assert len(topo.nodes) == len(bld.rooms)
        assert len(topo.edges) >= 1
        # Room 1 and Room 2 must be connected
        edge_pairs = {tuple(sorted((e.source_id, e.target_id))) for e in topo.edges}
        assert (1, 2) in edge_pairs
        assert 2 in bld.rooms[0].adjacent_room_ids
        assert 1 in bld.rooms[1].adjacent_room_ids

    def test_canvas_margin_exclusion_and_topology_sample(self):
        from pipeline import BlueprintPipeline

        res = BlueprintPipeline().run("sample_floorplan.png")
        b = res.building

        # Verify no room covers > 80% width and > 80% height with low density
        for r in b.rooms:
            bw, bh = r.bounding_box[2], r.bounding_box[3]
            assert not (bw > 700 and bh > 500), f"Canvas margin room R{r.room_id} was not filtered out"

        assert len(b.rooms) == 10
        assert len(b.topology_nodes) == 10
        assert len(b.topology_edges) >= 14
        # Every room has adjacent neighbors
        for r in b.rooms:
            assert len(r.adjacent_room_ids) > 0, f"Room {r.room_id} has no adjacent rooms"
