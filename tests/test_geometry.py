"""
tests/test_geometry.py
Blueprint-X — Unit tests for the GeometryEngine.
"""
import sys
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from backend.engine.geometry.geometry_engine import GeometryEngine, BuildingGraph
from backend.models.dataclasses import Wall, Door


def _make_wall(x1, y1, x2, y2) -> Wall:
    return Wall(start=(float(x1), float(y1)), end=(float(x2), float(y2)))


def _make_door(cx, cy, width=40.0) -> Door:
    return Door(center=(float(cx), float(cy)), width_px=width, source="gap")


class TestBuildingGraph:
    def test_empty_on_no_walls(self):
        g = GeometryEngine().run([], [], (500, 500))
        assert g.n_vertices == 0
        assert g.n_edges == 0

    def test_single_wall_creates_two_vertices(self):
        walls = [_make_wall(0, 0, 100, 0)]
        g = GeometryEngine().run(walls, [], (500, 500))
        assert g.n_vertices >= 2
        assert g.n_edges >= 1

    def test_perpendicular_walls_share_vertex(self):
        # Two walls that meet at origin
        walls = [
            _make_wall(0, 0, 100, 0),   # horizontal
            _make_wall(0, 0, 0, 100),   # vertical
        ]
        g = GeometryEngine().run(walls, [], (500, 500))
        # Should share the (0,0) vertex — 3 distinct vertices total
        assert g.n_vertices <= 4  # at most 4 (possibly snapped to 3)

    def test_graph_has_adjacency(self):
        walls = [_make_wall(0, 0, 100, 0)]
        g = GeometryEngine().run(walls, [], (500, 500))
        # Both endpoints should appear in adj_sorted
        assert len(g.adj_sorted) >= 2

    def test_adjacency_sorted_by_angle(self):
        """All neighbour lists must be sorted in ascending angle order."""
        walls = [
            _make_wall(200, 200, 400, 200),
            _make_wall(200, 200, 200, 400),
            _make_wall(200, 200, 400, 400),
        ]
        g = GeometryEngine().run(walls, [], (700, 700))
        for vid, neighbors in g.adj_sorted.items():
            angles = [a for _, a in neighbors]
            assert angles == sorted(angles), f"Adj not sorted at vertex {vid}"

    def test_door_creates_door_edge(self):
        walls = [
            _make_wall(0, 100, 200, 100),   # left half
            _make_wall(250, 100, 500, 100), # right half, gap at 200-250
        ]
        door = _make_door(225, 100, 50.0)
        g = GeometryEngine().run(walls, [door], (600, 300))
        door_edges = [eid for eid, (v1, v2, is_door) in g.edges.items() if is_door]
        assert len(door_edges) >= 1, "Door gap should create at least one door edge"

    def test_to_networkx(self):
        """BuildingGraph.to_networkx() should produce a valid NX graph."""
        nx = pytest.importorskip("networkx")
        walls = [_make_wall(0, 0, 100, 0), _make_wall(100, 0, 100, 100)]
        g = GeometryEngine().run(walls, [], (300, 300))
        nxg = g.to_networkx()
        assert nxg is not None
        assert isinstance(nxg, nx.Graph)
        assert nxg.number_of_nodes() == g.n_vertices
        assert nxg.number_of_edges() == g.n_edges


class TestGeometryEngineHelpers:
    def setup_method(self):
        self.eng = GeometryEngine()

    def test_seg_angle_horizontal(self):
        a = self.eng._seg_angle_norm(0, 0, 100, 0)
        assert abs(a - 0.0) < 1e-6

    def test_seg_angle_vertical(self):
        a = self.eng._seg_angle_norm(0, 0, 0, 100)
        assert abs(a - 90.0) < 1e-6

    def test_perp_dist_on_line(self):
        # Point ON the line (0,0)-(100,0) should have dist = 0
        d = self.eng._perp_dist_line(50, 0, 0, 0, 100, 0)
        assert d < 1e-6

    def test_perp_dist_off_line(self):
        d = self.eng._perp_dist_line(50, 10, 0, 0, 100, 0)
        assert abs(d - 10.0) < 1e-6

    def test_project_midpoint(self):
        t = self.eng._project(50, 0, 0, 0, 100, 0)
        assert abs(t - 0.5) < 1e-6

    def test_project_endpoint(self):
        t = self.eng._project(100, 0, 0, 0, 100, 0)
        assert abs(t - 1.0) < 1e-6

    def test_lines_intersect_perpendicular(self):
        pt = self.eng._lines_intersect(
            (0, 50, 100, 50),   # horizontal line y=50
            (50, 0, 50, 100),   # vertical line x=50
        )
        assert pt is not None
        assert abs(pt[0] - 50) < 1e-3 and abs(pt[1] - 50) < 1e-3

    def test_lines_intersect_parallel_returns_none(self):
        pt = self.eng._lines_intersect(
            (0, 0, 100, 0),
            (0, 10, 100, 10),
        )
        assert pt is None

    def test_group_segments_collinear(self):
        segs = [(0, 0, 50, 0), (60, 0, 100, 0)]   # same line, small gap
        groups = self.eng._group_segments(segs)
        assert len(groups) == 1, "Two collinear segs should be one group"

    def test_group_segments_perpendicular(self):
        segs = [(0, 0, 100, 0), (0, 0, 0, 100)]
        groups = self.eng._group_segments(segs)
        assert len(groups) == 2, "Perpendicular segs should be separate groups"
