"""
tests/test_model3d.py
Blueprint-X — Unit + integration tests for the 3D model pipeline.

Run with:
    D:\\BlueprintX\\venv313\\Scripts\\pytest.exe tests/test_model3d.py -v
"""
import json
import sys
import math
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from backend.models.dataclasses import Building, Building3D, Wall, Room, Door
from backend.engine.model3d.builder import Model3DBuilder
from backend.engine.model3d.obj_exporter import ObjExporter


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_wall(x1, y1, x2, y2, wtype="internal") -> Wall:
    return Wall(start=(float(x1), float(y1)), end=(float(x2), float(y2)), wall_type=wtype)


def _make_room(pts, label="Room") -> Room:
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    import math
    n = len(pts)
    area = abs(sum(
        pts[i][0] * pts[(i+1)%n][1] - pts[(i+1)%n][0] * pts[i][1]
        for i in range(n)
    )) / 2
    r = Room(
        polygon_pts=pts,
        centroid=(cx, cy),
        area_px=area,
        label=label,
        color_bgr=(100, 150, 200),
    )
    return r


def _simple_building() -> Building:
    """A minimal 10×10 m room with 4 outer walls."""
    bld = Building(source_path="test", image_size=(1000, 1000))
    bld.walls = [
        _make_wall(100, 100, 900, 100, "outer"),   # top
        _make_wall(900, 100, 900, 900, "outer"),   # right
        _make_wall(900, 900, 100, 900, "outer"),   # bottom
        _make_wall(100, 900, 100, 100, "outer"),   # left
    ]
    bld.rooms = [
        _make_room([(100, 100), (900, 100), (900, 900), (100, 900)], "Living Room")
    ]
    return bld


@pytest.fixture(scope="module")
def simple_model():
    bld = _simple_building()
    return Model3DBuilder(floor_height_m=2.8, px_per_m=100.0).run(bld)


@pytest.fixture(scope="module")
def pipeline_model():
    """Full pipeline result — reuse from test_pipeline if possible."""
    from pipeline import BlueprintPipeline
    sample = ROOT / "sample_floorplan.png"
    if not sample.exists():
        pytest.skip("sample_floorplan.png not found")
    res = BlueprintPipeline().run(str(sample))
    return res.building.model_3d


# ── Building3D Dataclass ──────────────────────────────────────────────────────

class TestBuilding3DDataclass:
    def test_default_empty(self):
        m = Building3D()
        assert m.vertex_count == 0
        assert m.triangle_count == 0

    def test_interleaved_shape(self):
        m = Building3D(
            vertices=np.zeros((4, 3), np.float32),
            normals=np.zeros((4, 3), np.float32),
            colors=np.ones((4, 3), np.float32),
            vertex_count=4,
        )
        il = m.interleaved()
        assert il.shape == (4, 9)
        assert il.dtype == np.float32

    def test_interleaved_empty(self):
        m = Building3D()
        assert m.interleaved().shape == (0, 9)

    def test_flat_indices_uint32(self):
        m = Building3D(
            indices=np.array([[0, 1, 2], [0, 2, 3]], np.int32),
            triangle_count=2,
        )
        fi = m.flat_indices()
        assert fi.shape == (6,)

    def test_bounds_fields(self):
        m = Building3D(
            bounds_min=(-5.0, 0.0, -5.0),
            bounds_max=(5.0, 2.8, 5.0),
        )
        assert m.bounds_min[1] == 0.0
        assert m.bounds_max[1] == 2.8


# ── Model3DBuilder ────────────────────────────────────────────────────────────

class TestModel3DBuilder:
    def test_produces_building3d(self, simple_model):
        assert isinstance(simple_model, Building3D)

    def test_has_vertices(self, simple_model):
        assert simple_model.vertex_count > 0

    def test_has_triangles(self, simple_model):
        assert simple_model.triangle_count > 0

    def test_vertex_array_shape(self, simple_model):
        assert simple_model.vertices.ndim == 2
        assert simple_model.vertices.shape[1] == 3

    def test_normal_array_shape(self, simple_model):
        assert simple_model.normals.shape == simple_model.vertices.shape

    def test_color_array_shape(self, simple_model):
        assert simple_model.colors.shape == simple_model.vertices.shape

    def test_colors_in_range(self, simple_model):
        assert simple_model.colors.min() >= 0.0
        assert simple_model.colors.max() <= 1.0

    def test_indices_valid_range(self, simple_model):
        n = simple_model.vertex_count
        assert simple_model.indices.max() < n
        assert simple_model.indices.min() >= 0

    def test_bounds_computed(self, simple_model):
        bmin = np.array(simple_model.bounds_min)
        bmax = np.array(simple_model.bounds_max)
        assert (bmax >= bmin).all()

    def test_center_inside_bounds(self, simple_model):
        c = np.array(simple_model.center)
        bmin = np.array(simple_model.bounds_min)
        bmax = np.array(simple_model.bounds_max)
        assert (c >= bmin - 1e-3).all()
        assert (c <= bmax + 1e-3).all()

    def test_floor_height_respected(self, simple_model):
        """Y-axis of vertices should span from ~slab bottom to ~floor_height_m."""
        y_vals = simple_model.vertices[:, 1]
        # Ground slab bottom is at -0.06 m; walls top at floor_height_m
        assert y_vals.min() >= -0.10           # slab bottom with a little tolerance
        assert abs(y_vals.max() - 2.8) < 0.05  # ceiling at y≈2.8

    def test_wall_quad_geometry(self):
        """4 walls → at least 4 quads × 2 tris × 2 faces = 16+ triangles."""
        bld = _simple_building()
        m = Model3DBuilder(floor_height_m=3.0, px_per_m=100.0).run(bld)
        assert m.triangle_count >= 16

    def test_empty_building_returns_empty_model(self):
        bld = Building(source_path="x", image_size=(500, 500))
        m = Model3DBuilder().run(bld)
        assert m.vertex_count == 0

    def test_floor_slab_present(self, simple_model):
        """Y≈0 vertices should exist (floor slab)."""
        y = simple_model.vertices[:, 1]
        assert np.any(np.abs(y) < 0.1), "No floor-level geometry found"

    def test_ceiling_slab_present(self, simple_model):
        """Y≈floor_height_m vertices should exist (ceiling slab)."""
        h = simple_model.floor_height_m
        y = simple_model.vertices[:, 1]
        assert np.any(np.abs(y - h) < 0.1), "No ceiling-level geometry found"

    def test_door_gap_removes_wall_section(self):
        """
        A door gap splits the wall at the gap position, producing two separate
        wall sub-segments.  The total geometry therefore has MORE quads than a
        single uninterrupted wall — but the gap itself has no geometry.
        """
        bld_no_door = Building(source_path="x", image_size=(500, 300))
        bld_no_door.walls = [_make_wall(50, 150, 450, 150)]

        bld_door = Building(source_path="x", image_size=(500, 300))
        bld_door.walls = [_make_wall(50, 150, 450, 150)]
        bld_door.doors = [Door(center=(250, 150), width_px=60, source="gap")]

        m_no_door = Model3DBuilder(px_per_m=100).run(bld_no_door)
        m_door    = Model3DBuilder(px_per_m=100).run(bld_door)

        # Wall boxes now have 6 faces per sub-segment (added bottom face)
        # Each face = 1 quad = 2 tris  →  12 tris per sub-segment
        # Plus ground slab contributes 12 tris (6 faces × 2)
        # The key assertion is: WITH a door gap we have MORE tris than without
        assert m_door.triangle_count > m_no_door.triangle_count, \
            "Door gap should produce 2 sub-segments (more tris than 1 continuous wall)"

    def test_custom_floor_height(self):
        bld = _simple_building()
        m = Model3DBuilder(floor_height_m=4.5, px_per_m=100).run(bld)
        assert abs(m.floor_height_m - 4.5) < 1e-6
        y = m.vertices[:, 1]
        assert y.max() >= 4.4

    def test_scale_parsing(self):
        bld = _simple_building()
        bld.scale = "1:50"
        m = Model3DBuilder(px_per_m=100).run(bld)   # 50 should override 100
        assert abs(m.px_per_m - 50.0) < 1e-6

    def test_interleaved_output(self, simple_model):
        il = simple_model.interleaved()
        assert il.shape[0] == simple_model.vertex_count
        assert il.shape[1] == 9
        assert il.dtype == np.float32


# ── ObjExporter ───────────────────────────────────────────────────────────────

class TestObjExporter:
    def test_export_creates_file(self, simple_model, tmp_path):
        out = str(tmp_path / "test.obj")
        ObjExporter().export(simple_model, out)
        assert Path(out).exists()

    def test_mtl_file_created(self, simple_model, tmp_path):
        out = str(tmp_path / "test.obj")
        ObjExporter().export(simple_model, out)
        mtl = tmp_path / "test.mtl"
        assert mtl.exists()

    def test_obj_has_vertices(self, simple_model, tmp_path):
        out = str(tmp_path / "test.obj")
        ObjExporter().export(simple_model, out)
        lines = Path(out).read_text().splitlines()
        v_lines = [l for l in lines if l.startswith("v ")]
        assert len(v_lines) == simple_model.vertex_count

    def test_obj_has_faces(self, simple_model, tmp_path):
        out = str(tmp_path / "test.obj")
        ObjExporter().export(simple_model, out)
        lines = Path(out).read_text().splitlines()
        f_lines = [l for l in lines if l.startswith("f ")]
        assert len(f_lines) == simple_model.triangle_count

    def test_obj_has_normals(self, simple_model, tmp_path):
        out = str(tmp_path / "test.obj")
        ObjExporter().export(simple_model, out)
        lines = Path(out).read_text().splitlines()
        vn_lines = [l for l in lines if l.startswith("vn ")]
        assert len(vn_lines) == simple_model.vertex_count

    def test_obj_1_indexed_faces(self, simple_model, tmp_path):
        out = str(tmp_path / "test.obj")
        ObjExporter().export(simple_model, out)
        lines = Path(out).read_text().splitlines()
        f_lines = [l for l in lines if l.startswith("f ")]
        for fl in f_lines:
            parts = fl.split()[1:]
            for p in parts:
                idx = int(p.split("//")[0])
                assert idx >= 1, "OBJ face indices must be 1-based"

    def test_export_empty_model(self, tmp_path):
        out = str(tmp_path / "empty.obj")
        ObjExporter().export(Building3D(), out)
        # Should not raise; file may not exist or be empty

    def test_obj_has_mtllib_reference(self, simple_model, tmp_path):
        out = str(tmp_path / "model.obj")
        ObjExporter().export(simple_model, out)
        content = Path(out).read_text()
        assert "mtllib" in content

    def test_mtl_has_newmtl(self, simple_model, tmp_path):
        out = str(tmp_path / "model.obj")
        ObjExporter().export(simple_model, out)
        mtl = (tmp_path / "model.mtl").read_text()
        assert "newmtl" in mtl
        assert "Kd" in mtl


# ── GLB Exporter ──────────────────────────────────────────────────────────────

class TestGlbExporter:
    def test_export_returns_path_or_none(self, simple_model, tmp_path):
        from backend.engine.model3d.glb_exporter import GlbExporter
        out = str(tmp_path / "test.glb")
        result = GlbExporter().export(simple_model, out)
        # Either succeeds (trimesh installed) or returns None gracefully
        assert result is None or Path(result).exists()

    def test_export_creates_glb_if_trimesh_available(self, simple_model, tmp_path):
        try:
            import trimesh
            has_trimesh = True
        except ImportError:
            has_trimesh = False

        if not has_trimesh:
            pytest.skip("trimesh not installed")

        from backend.engine.model3d.glb_exporter import GlbExporter
        out = str(tmp_path / "test.glb")
        result = GlbExporter().export(simple_model, out)
        assert result is not None
        assert Path(result).exists()
        assert Path(result).stat().st_size > 0


# ── Pipeline integration ──────────────────────────────────────────────────────

class TestPipelineModel3D:
    def test_pipeline_produces_model3d(self, pipeline_model):
        assert pipeline_model is not None
        assert isinstance(pipeline_model, Building3D)

    def test_pipeline_model_has_geometry(self, pipeline_model):
        assert pipeline_model.vertex_count > 0
        assert pipeline_model.triangle_count > 0

    def test_pipeline_model_bounds_valid(self, pipeline_model):
        bmin = np.array(pipeline_model.bounds_min)
        bmax = np.array(pipeline_model.bounds_max)
        assert (bmax > bmin).all()

    def test_pipeline_json_has_model3d_key(self, tmp_path):
        from pipeline import BlueprintPipeline
        from backend.engine.export.exporter import Exporter
        sample = ROOT / "sample_floorplan.png"
        if not sample.exists():
            pytest.skip("sample_floorplan.png not found")
        res = BlueprintPipeline().run(str(sample))
        out = tmp_path / "full_export.json"
        Exporter().export(res.building, str(out))
        data = json.loads(out.read_text())
        m3d = data["model_3d"]
        assert m3d["available"] is True
        assert m3d["vertex_count"] > 0
        assert m3d["triangle_count"] > 0
        assert "floor_height_m" in m3d
        assert "px_per_m" in m3d
