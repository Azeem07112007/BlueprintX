"""
processors/graph_builder.py  — v3 (Stages 3 + 4)
Blueprint-X Topology Pipeline

Correct approach to building the planar graph:

1. Group segments by their infinite line (angle + perpendicular distance).
2. For each line, sort all segments by projection → find all endpoints.
3. Connect consecutive endpoints along the line with sub-edges.
4. Add perpendicular intersections between lines as shared vertices.
5. Seal door gaps with synthetic edges.
6. Build adjacency sorted by angle.

This correctly forms a planar graph even when the outer wall is split into
many pieces (top, right, bottom, left) and when interior walls only partially
span the floor plan.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set

from processors.wall_detection import WallSegment, DoorGap


# ─── Data model ──────────────────────────────────────────────────────────────

@dataclass
class WallGraph:
    vertices:   Dict[int, Tuple[float, float]]
    edges:      Dict[int, Tuple[int, int, bool]]       # (v1,v2,is_door)
    adj_sorted: Dict[int, List[Tuple[int, float]]]     # (neighbor, angle_rad)

    @property
    def n_vertices(self): return len(self.vertices)
    @property
    def n_edges(self):    return len(self.edges)


_EPS = 1e-9


# ─── Geometry helpers ─────────────────────────────────────────────────────────

def _seg_angle_norm(x1,y1,x2,y2) -> float:
    """Normalised angle ∈ [0°, 180°)."""
    return math.degrees(math.atan2(y2-y1, x2-x1)) % 180.0


def _perp_dist_line(px,py, x1,y1,x2,y2) -> float:
    """Perpendicular distance from (px,py) to the infinite LINE through the segment."""
    dx,dy = x2-x1, y2-y1
    d = math.hypot(dx,dy)
    if d < _EPS:
        return math.hypot(px-x1, py-y1)
    return abs(dx*(y1-py) - dy*(x1-px)) / d


def _proj_scalar(px,py, x1,y1,x2,y2) -> float:
    """Scalar projection of (px,py) onto line direction, measured in pixels from (x1,y1)."""
    dx,dy = x2-x1,y2-y1
    d = math.hypot(dx,dy)
    if d < _EPS: return 0.0
    return ((px-x1)*dx + (py-y1)*dy) / d


def _lines_intersect(
    x1:float,y1:float, x2:float,y2:float,
    x3:float,y3:float, x4:float,y4:float,
) -> Optional[Tuple[float,float]]:
    """Intersection of two INFINITE lines. Returns None if parallel."""
    denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
    if abs(denom) < _EPS:
        return None
    t = ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)) / denom
    return (x1+t*(x2-x1), y1+t*(y2-y1))


# ─── Line grouping ────────────────────────────────────────────────────────────

class _LineGroup:
    """All collinear segments that lie on the same infinite line."""
    __slots__ = ('angle','ref_x','ref_y','ref_dx','ref_dy','projections','is_door_list')

    def __init__(self, angle:float, x1:float,y1:float,x2:float,y2:float):
        self.angle = angle
        self.ref_x, self.ref_y = x1, y1
        dx,dy = x2-x1, y2-y1
        d = math.hypot(dx,dy)
        self.ref_dx = dx/d if d > _EPS else 1.0
        self.ref_dy = dy/d if d > _EPS else 0.0
        self.projections: List[Tuple[float,float]] = []  # [(t_start, t_end), ...]
        self.is_door_list: List[bool] = []

    def add(self, x1:float,y1:float, x2:float,y2:float, is_door:bool=False):
        t1 = self._proj(x1,y1)
        t2 = self._proj(x2,y2)
        if t1 > t2: t1,t2 = t2,t1
        self.projections.append((t1,t2))
        self.is_door_list.append(is_door)

    def _proj(self, px:float,py:float) -> float:
        return (px-self.ref_x)*self.ref_dx + (py-self.ref_y)*self.ref_dy

    def point_at(self, t:float) -> Tuple[float,float]:
        return (self.ref_x + t*self.ref_dx, self.ref_y + t*self.ref_dy)


def _group_segments(
    segs_with_flags: List[Tuple[float,float,float,float,bool]],
    angle_tol: float = 5.0,
    perp_tol:  float = 8.0,
) -> List[_LineGroup]:
    """
    Group collinear segments into _LineGroup objects.
    Two segments are collinear if their angles differ by < angle_tol degrees
    and the perpendicular distance between them is < perp_tol.
    """
    groups: List[_LineGroup] = []

    for x1,y1,x2,y2,is_door in segs_with_flags:
        a = _seg_angle_norm(x1,y1,x2,y2)
        placed = False
        for g in groups:
            da = min(abs(a-g.angle), 180-abs(a-g.angle))
            if da > angle_tol:
                continue
            # Midpoint perpendicular distance
            mx,my = (x1+x2)/2, (y1+y2)/2
            # Reference point on g's line
            rx,ry = g.ref_x, g.ref_y
            rdx,rdy = g.ref_dx, g.ref_dy
            perp = abs(rdx*(ry-my) - rdy*(rx-mx))   # cross product = perp dist
            if perp > perp_tol:
                continue
            g.add(x1,y1,x2,y2,is_door)
            placed = True
            break
        if not placed:
            g = _LineGroup(a,x1,y1,x2,y2)
            g.add(x1,y1,x2,y2,is_door)
            groups.append(g)

    return groups


# ─── Vertex pool ─────────────────────────────────────────────────────────────

class _VP:
    def __init__(self, snap:float=10.0):
        self._pts: Dict[int,Tuple[float,float]] = {}
        self._nid = 0
        self._snap = snap

    def add(self, x:float, y:float) -> int:
        for vid,(vx,vy) in self._pts.items():
            if math.hypot(x-vx,y-vy) <= self._snap:
                self._pts[vid] = ((vx+x)/2,(vy+y)/2)
                return vid
        nid = self._nid
        self._pts[nid] = (x,y)
        self._nid += 1
        return nid

    @property
    def pts(self): return dict(self._pts)


# ─── Public API ──────────────────────────────────────────────────────────────

def build_graph(
    wall_segments: List[WallSegment],
    door_gaps:     List[DoorGap],
    image_shape:   Tuple[int,int],
    snap_dist:     float = 12.0,
) -> WallGraph:
    """
    Build a planar graph from wall vectors + sealed door gaps.

    Algorithm:
      1. Group all segments (walls + door seals) by collinear line.
      2. Find ALL pairwise intersections between different line groups.
      3. Pool all intersection + endpoint positions, snapping nearby ones.
      4. For each line group, collect all vertices on that line and
         sort them by projection → create edges between consecutive vertices.
      5. Build angle-sorted adjacency for face traversal.
    """
    wall_raw = [(s.start[0],s.start[1],s.end[0],s.end[1],False)
                for s in wall_segments]
    door_raw = [(g.start[0],g.start[1],g.end[0],g.end[1],True)
                for g in door_gaps]
    all_raw  = wall_raw + door_raw

    # ── Step 1: group collinear segments ─────────────────────────────────
    groups = _group_segments(all_raw, angle_tol=5.0, perp_tol=8.0)

    # ── Step 2: find all line×line intersections ──────────────────────────
    pool = _VP(snap=snap_dist)

    # All segment endpoints → vertices
    for x1,y1,x2,y2,_ in all_raw:
        pool.add(x1,y1)
        pool.add(x2,y2)

    # All pairwise line intersections (between DIFFERENT groups)
    for i,gi in enumerate(groups):
        for j,gj in enumerate(groups):
            if j <= i: continue
            # Only intersect non-parallel lines
            da = min(abs(gi.angle-gj.angle), 180-abs(gi.angle-gj.angle))
            if da < 5.0: continue   # parallel / same direction

            # Representative points on each infinite line
            ix1,iy1 = gi.ref_x, gi.ref_y
            ix2,iy2 = ix1+gi.ref_dx*100, iy1+gi.ref_dy*100
            jx1,jy1 = gj.ref_x, gj.ref_y
            jx2,jy2 = jx1+gj.ref_dx*100, jy1+gj.ref_dy*100

            pt = _lines_intersect(ix1,iy1,ix2,iy2,jx1,jy1,jx2,jy2)
            if pt is None: continue

            # Only add if the intersection is near a segment in EACH group
            h_img,w_img = image_shape
            px,py = pt
            if px < -50 or px > w_img+50 or py < -50 or py > h_img+50:
                continue

            # Check if pt is plausibly on at least one segment in each group
            def near_group(g:_LineGroup, px:float,py:float, margin:float=40.0)->bool:
                t = g._proj(px,py)
                for (t1,t2) in g.projections:
                    if t1-margin <= t <= t2+margin:
                        return True
                return False

            if near_group(gi,px,py) and near_group(gj,px,py):
                pool.add(px,py)

    vertices = pool.pts

    # ── Step 3: for each line group, create edges between consecutive vertices
    edge_set: Set[Tuple[int,int]] = set()
    edge_list: List[Tuple[int,int,bool]] = []

    def add_edge(v1:int,v2:int,is_door:bool=False):
        k = (min(v1,v2),max(v1,v2))
        if k not in edge_set and v1!=v2:
            edge_set.add(k)
            edge_list.append((v1,v2,is_door))

    for g in groups:
        if not g.projections:
            continue

        # Collect all vertices that lie on this line (within perp_tol)
        on_line: List[Tuple[float,int]] = []  # (t, vid)
        for vid,(vx,vy) in vertices.items():
            perp = _perp_dist_line(vx,vy, g.ref_x,g.ref_y,
                                   g.ref_x+g.ref_dx,g.ref_y+g.ref_dy)
            if perp > snap_dist:
                continue
            t = g._proj(vx,vy)
            # Check t is within the extent of this group's segments (with margin)
            all_t = [t_val for pair in g.projections for t_val in pair]
            t_min = min(all_t)
            t_max = max(all_t)
            if t_min - snap_dist <= t <= t_max + snap_dist:
                on_line.append((t,vid))

        if len(on_line) < 2:
            continue

        on_line.sort(key=lambda x:x[0])

        # Deduplicate (same t within snap)
        deduped = [on_line[0]]
        for tv in on_line[1:]:
            if abs(tv[0]-deduped[-1][0]) > snap_dist:
                deduped.append(tv)

        # Determine if each sub-edge is a door based on which gap it falls in
        # A segment is a door if it falls entirely within a door projection
        all_door_projs = [(t1,t2) for k,(t1,t2) in enumerate(g.projections)
                          if g.is_door_list[k]]

        def is_door_edge(ta:float, tb:float) -> bool:
            mid = (ta+tb)/2
            for (dt1,dt2) in all_door_projs:
                if dt1 <= mid <= dt2:
                    return True
            return False

        # Create edges between consecutive vertices along this line.
        # Rule: create an edge unless the gap is much larger than a door
        # (i.e., > 80px AND not covered by any segment or door seal).
        MAX_GAP_NO_EDGE = 80.0   # gaps larger than this without a segment = don't create edge

        for k in range(len(deduped)-1):
            ta,va = deduped[k]
            tb,vb = deduped[k+1]
            gap_len = abs(tb-ta)

            # Check if any segment (wall or door) covers the midpoint of this sub-edge
            mid_t = (ta+tb)/2
            covered = False
            for idx_s,(st1,st2) in enumerate(g.projections):
                # Covered if midpoint falls within segment's projection ± snap
                if st1-snap_dist <= mid_t <= st2+snap_dist:
                    covered = True
                    break

            # If not covered by a segment, only skip if the gap is very large
            if not covered and gap_len > MAX_GAP_NO_EDGE:
                continue

            door_flag = is_door_edge(ta,tb)
            add_edge(va,vb,door_flag)

    # ── Step 4: adjacency sorted by angle ─────────────────────────────────
    vertices = pool.pts
    adj: Dict[int,List[Tuple[int,float]]] = defaultdict(list)
    for v1,v2,_ in edge_list:
        if v1 not in vertices or v2 not in vertices:
            continue
        x1,y1 = vertices[v1]; x2,y2 = vertices[v2]
        adj[v1].append((v2, math.atan2(y2-y1,x2-x1)))
        adj[v2].append((v1, math.atan2(y1-y2,x1-x2)))

    for vid in adj:
        seen: set = set()
        uniq = []
        for n_,a in sorted(adj[vid], key=lambda x:x[1]):
            if n_ not in seen:
                seen.add(n_); uniq.append((n_,a))
        adj[vid] = uniq

    edge_dict = {eid:(v1,v2,d) for eid,(v1,v2,d) in enumerate(edge_list)}
    return WallGraph(vertices=vertices, edges=edge_dict, adj_sorted=dict(adj))
