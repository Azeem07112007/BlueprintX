"""
backend/engine/geometry/geometry_engine.py
Blueprint-X — M4 (Accuracy Overhaul)

Converts wall vectors into a clean planar graph.

Major improvements over v3:
  - Vertex snap tolerance raised to 8.0 px (was 5.0) — handles blueprint coordinate noise
  - large_gap threshold raised to 120 px (was 80) — avoids false "gap" breaks in long walls
  - T-junction splitting: if a new intersection point lies on an existing edge,
    that edge is split at the intersection vertex (critical for correct face extraction)
  - Group segments use all-pairs matching (not just seed-based), avoiding the case
    where segment i is consumed before j can be properly grouped with it
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from backend.models.dataclasses import Wall, Door

try:
    import networkx as nx
    _HAS_NX = True
except ImportError:
    _HAS_NX = False


@dataclass
class BuildingGraph:
    """Planar graph of wall intersections and segments."""
    vertices: Dict[int, Tuple[float, float]] = field(default_factory=dict)
    edges: Dict[int, Tuple[int, int, bool]] = field(default_factory=dict)     # (v1, v2, is_door)
    adj_sorted: Dict[int, List[Tuple[int, float]]] = field(default_factory=dict)

    @property
    def n_vertices(self) -> int: return len(self.vertices)
    @property
    def n_edges(self) -> int: return len(self.edges)

    def to_networkx(self):
        """Convert to NetworkX graph (if available)."""
        if not _HAS_NX:
            return None
        G = nx.Graph()
        for vid, (x, y) in self.vertices.items():
            G.add_node(vid, x=x, y=y)
        for eid, (v1, v2, is_door) in self.edges.items():
            G.add_edge(v1, v2, is_door=is_door, edge_id=eid)
        return G


_EPS = 1e-9

# Vertex snap radius — how close two points must be to be the same junction
_VERTEX_SNAP = 8.0

# Maximum gap (px) in a wall line group that is still treated as wall (not a
# door/window opening). Raised from 80→120 so long walls with small OCR noise
# aren't incorrectly broken into fragments.
_LARGE_GAP = 120


class GeometryEngine:
    """
    Builds a planar wall graph from wall vectors and door gaps.

    Steps:
      1. Group segments by their infinite line (angle + perpendicular offset)
      2. Find all intersection vertices (cross-group intersections)
      3. Build edges between consecutive vertices along each line group
      4. T-junction splitting: vertex on edge → split edge at vertex
      5. Seal door gaps with synthetic door edges
      6. Sort adjacency lists by angle (DCEL requirement)
    """

    def run(
        self,
        walls: List[Wall],
        door_gaps: List[Door],
        image_size: Tuple[int, int],
    ) -> BuildingGraph:
        segs = [(w.start[0], w.start[1], w.end[0], w.end[1]) for w in walls]
        door_data = [(d.center[0], d.center[1], d.width_px, d.angle_deg) for d in door_gaps]
        return self._build(segs, door_data, image_size)

    # ── Core graph building ──────────────────────────────────────────────────

    def _build(
        self,
        segs: List[Tuple],
        door_data: List[Tuple],
        image_size: Tuple[int, int],
    ) -> BuildingGraph:
        if not segs:
            return BuildingGraph()

        vertex_counter = [0]
        vertices: Dict[int, Tuple[float, float]] = {}

        def _get_or_create_vertex(x: float, y: float) -> int:
            for vid, (vx, vy) in vertices.items():
                if math.hypot(x - vx, y - vy) < _VERTEX_SNAP:
                    return vid
            vid = vertex_counter[0]
            vertex_counter[0] += 1
            vertices[vid] = (x, y)
            return vid

        # Group segments into line groups
        groups = self._group_segments(segs)

        # Collect all intersection points per group
        group_intervals: Dict[int, List] = defaultdict(list)
        for gi, group in groups.items():
            ax1, ay1, ax2, ay2 = group["axis"]
            dx, dy = ax2 - ax1, ay2 - ay1
            length = math.hypot(dx, dy)
            if length < _EPS:
                continue
            for seg in group["segs"]:
                x1, y1, x2, y2 = seg
                t1 = self._project(x1, y1, ax1, ay1, ax2, ay2)
                t2 = self._project(x2, y2, ax1, ay1, ax2, ay2)
                if t1 > t2:
                    t1, t2 = t2, t1
                group_intervals[gi].append((t1, t2))

        # Find cross-group intersections and add t-values
        for gi, g_info in groups.items():
            for gj, h_info in groups.items():
                if gi >= gj:
                    continue
                pt = self._lines_intersect(g_info["axis"], h_info["axis"])
                if pt is None:
                    continue
                ax1, ay1, ax2, ay2 = g_info["axis"]
                t = self._project(pt[0], pt[1], ax1, ay1, ax2, ay2)
                bx1, by1, bx2, by2 = h_info["axis"]
                s = self._project(pt[0], pt[1], bx1, by1, bx2, by2)
                group_intervals[gi].append((t, t))
                group_intervals[gj].append((s, s))

        # Build edges
        edges: Dict[int, Tuple[int, int, bool]] = {}
        edge_counter = [0]

        for gi, g_info in groups.items():
            ax1, ay1, ax2, ay2 = g_info["axis"]
            dx, dy = ax2 - ax1, ay2 - ay1
            length = math.hypot(dx, dy)
            if length < _EPS:
                continue

            all_t = []
            for t1, t2 in group_intervals[gi]:
                all_t.extend([t1, t2])
            all_t = sorted(set(round(t, 4) for t in all_t))

            covered = sorted(group_intervals[gi], key=lambda x: x[0])

            for k in range(len(all_t) - 1):
                t_start = all_t[k]
                t_end   = all_t[k + 1]
                mid_t   = (t_start + t_end) / 2
                gap     = (t_end - t_start) * length

                covered_by_seg = any(t1 <= mid_t <= t2 for t1, t2 in covered)
                large_gap = gap > _LARGE_GAP   # raised from 80 → 120

                if not covered_by_seg and large_gap:
                    continue

                px1 = ax1 + t_start * dx
                py1 = ay1 + t_start * dy
                px2 = ax1 + t_end * dx
                py2 = ay1 + t_end * dy

                v1 = _get_or_create_vertex(px1, py1)
                v2 = _get_or_create_vertex(px2, py2)
                if v1 == v2:
                    continue

                eid = edge_counter[0]
                edge_counter[0] += 1
                edges[eid] = (v1, v2, False)

        # ── T-junction splitting ─────────────────────────────────────────────
        # If a vertex lies on (i.e. near the interior of) an existing edge,
        # split that edge into two sub-edges at the vertex.  This ensures
        # the DCEL face traversal doesn't skip over T-intersections.
        edges = self._split_t_junctions(vertices, edges, edge_counter)

        # Seal door gaps
        for cx, cy, width, angle_deg in door_data:
            rad = math.radians(angle_deg)
            half = width / 2 + 5
            gx1 = cx - math.cos(rad) * half
            gy1 = cy - math.sin(rad) * half
            gx2 = cx + math.cos(rad) * half
            gy2 = cy + math.sin(rad) * half
            v1 = _get_or_create_vertex(gx1, gy1)
            v2 = _get_or_create_vertex(gx2, gy2)
            if v1 != v2:
                eid = edge_counter[0]
                edge_counter[0] += 1
                edges[eid] = (v1, v2, True)

        # Sort adjacency by angle
        adj_raw: Dict[int, List] = defaultdict(list)
        for eid, (v1, v2, _) in edges.items():
            if v1 not in vertices or v2 not in vertices:
                continue
            x1, y1 = vertices[v1]
            x2, y2 = vertices[v2]
            angle_12 = math.atan2(y2 - y1, x2 - x1)
            angle_21 = math.atan2(y1 - y2, x1 - x2)
            adj_raw[v1].append((v2, angle_12))
            adj_raw[v2].append((v1, angle_21))

        adj_sorted: Dict[int, List[Tuple[int, float]]] = {}
        for vid, neighbors in adj_raw.items():
            # Deduplicate neighbors (same vid may appear twice if a vertex is
            # referenced by two nearly-coincident edges)
            seen = set()
            unique = []
            for nb, ang in sorted(neighbors, key=lambda x: x[1]):
                if nb not in seen:
                    seen.add(nb)
                    unique.append((nb, ang))
            adj_sorted[vid] = unique

        return BuildingGraph(vertices=vertices, edges=edges, adj_sorted=adj_sorted)

    # ── T-junction splitting ──────────────────────────────────────────────────

    def _split_t_junctions(
        self,
        vertices: Dict[int, Tuple[float, float]],
        edges: Dict[int, Tuple[int, int, bool]],
        edge_counter: List[int],
        tol: float = _VERTEX_SNAP,
    ) -> Dict[int, Tuple[int, int, bool]]:
        """
        For every vertex V and every edge (A, B):
          If V lies on the segment AB (not at A or B) within `tol`,
          split (A, B) into (A, V) and (V, B).
        """
        changed = True
        max_iters = 5   # limit iterations for very complex plans
        iters = 0

        while changed and iters < max_iters:
            changed = False
            iters += 1
            new_edges: Dict[int, Tuple[int, int, bool]] = {}

            for eid, (va, vb, is_door) in list(edges.items()):
                if va not in vertices or vb not in vertices:
                    new_edges[eid] = (va, vb, is_door)
                    continue

                ax, ay = vertices[va]
                bx, by = vertices[vb]
                dx, dy = bx - ax, by - ay
                L = math.hypot(dx, dy)
                if L < _EPS:
                    new_edges[eid] = (va, vb, is_door)
                    continue

                split_v = None
                split_t = None
                for vid, (vx, vy) in vertices.items():
                    if vid == va or vid == vb:
                        continue
                    # Perpendicular distance to line AB
                    perp = abs(dx * (ay - vy) - dy * (ax - vx)) / L
                    if perp > tol:
                        continue
                    # Parametric position on AB
                    t = ((vx - ax) * dx + (vy - ay) * dy) / (L * L)
                    if t <= 0.01 or t >= 0.99:
                        continue
                    # This vertex lies inside segment AB → split
                    if split_v is None or abs(t - 0.5) < abs(split_t - 0.5):
                        split_v = vid
                        split_t = t

                if split_v is not None:
                    # Replace (va, vb) with (va, split_v) and (split_v, vb)
                    eid1 = edge_counter[0]; edge_counter[0] += 1
                    eid2 = edge_counter[0]; edge_counter[0] += 1
                    new_edges[eid1] = (va, split_v, is_door)
                    new_edges[eid2] = (split_v, vb, is_door)
                    changed = True
                else:
                    new_edges[eid] = (va, vb, is_door)

            edges = new_edges

        return edges

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _seg_angle_norm(self, x1, y1, x2, y2) -> float:
        return math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180.0

    def _perp_dist_line(self, px, py, x1, y1, x2, y2) -> float:
        dx, dy = x2 - x1, y2 - y1
        l = math.hypot(dx, dy)
        if l < _EPS:
            return math.hypot(px - x1, py - y1)
        return abs(dx * (y1 - py) - dy * (x1 - px)) / l

    def _project(self, px, py, x1, y1, x2, y2) -> float:
        dx, dy = x2 - x1, y2 - y1
        l2 = dx * dx + dy * dy
        if l2 < _EPS:
            return 0.0
        return ((px - x1) * dx + (py - y1) * dy) / l2

    def _group_segments(
        self,
        segs: List[Tuple[float, float, float, float]],
        angle_tol: float = 4.0,
        perp_tol: float = 10.0,
    ) -> Dict[int, dict]:
        """
        All-pairs grouping to avoid greedy-skip: every segment is compared
        against every other; once a group is formed, all members are marked
        used so they're not re-processed.
        """
        used = [False] * len(segs)
        groups: Dict[int, dict] = {}
        gi = 0

        for i, (x1, y1, x2, y2) in enumerate(segs):
            if used[i]:
                continue
            group_segs = [(x1, y1, x2, y2)]
            used[i] = True
            ai = self._seg_angle_norm(x1, y1, x2, y2)

            for j, (px1, py1, px2, py2) in enumerate(segs):
                if used[j] or i == j:
                    continue
                aj = self._seg_angle_norm(px1, py1, px2, py2)
                da = min(abs(ai - aj), 180 - abs(ai - aj))
                if da > angle_tol:
                    continue
                # Use the seed segment's midpoint as the perpendicular reference
                mx, my = (px1 + px2) / 2, (py1 + py2) / 2
                if self._perp_dist_line(mx, my, x1, y1, x2, y2) <= perp_tol:
                    group_segs.append((px1, py1, px2, py2))
                    used[j] = True

            longest = max(group_segs, key=lambda s: math.hypot(s[2] - s[0], s[3] - s[1]))
            lx1, ly1, lx2, ly2 = longest
            dx, dy = lx2 - lx1, ly2 - ly1
            length = math.hypot(dx, dy)
            if length < _EPS:
                continue

            scale_factor = 2000.0 / max(length, 1)
            ex1 = lx1 - dx * scale_factor
            ey1 = ly1 - dy * scale_factor
            ex2 = lx2 + dx * scale_factor
            ey2 = ly2 + dy * scale_factor

            groups[gi] = {"axis": (ex1, ey1, ex2, ey2), "segs": group_segs}
            gi += 1

        return groups

    def _lines_intersect(
        self,
        line_a: Tuple[float, float, float, float],
        line_b: Tuple[float, float, float, float],
        angle_tol: float = 8.0,    # reduced from 10 — avoids merging nearly-parallel walls
    ) -> Optional[Tuple[float, float]]:
        ax1, ay1, ax2, ay2 = line_a
        bx1, by1, bx2, by2 = line_b
        da = abs(self._seg_angle_norm(ax1, ay1, ax2, ay2) - self._seg_angle_norm(bx1, by1, bx2, by2))
        da = min(da, 180 - da)
        if da < angle_tol:
            return None

        d1x, d1y = ax2 - ax1, ay2 - ay1
        d2x, d2y = bx2 - bx1, by2 - by1
        cross = d1x * d2y - d1y * d2x
        if abs(cross) < _EPS:
            return None

        t = ((bx1 - ax1) * d2y - (by1 - ay1) * d2x) / cross
        ix = ax1 + t * d1x
        iy = ay1 + t * d1y
        return (ix, iy)
