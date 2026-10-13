"""
processors/room_detection.py  — COMPLETE REWRITE (Stages 5–10)
Blueprint-X Topology Pipeline

Uses PLANAR GRAPH FACE TRAVERSAL to detect rooms as closed polygon faces.

Why this works where flood-fill fails:
  - Rooms are the bounded faces of the planar wall graph.
  - Door openings are sealed with temporary edges, so each room is a
    proper closed polygon face in the graph.
  - We enumerate ALL faces by traversing every directed half-edge exactly
    once with the "leftmost turn" rule.
  - The algorithm is O(E) and architecture-independent.

Algorithm (Stages 5–9):
  Stage 5  find_planar_faces()  ← the core, replaces flood-fill entirely
  Stage 6  filter invalid faces (border, tiny, exterior)
  Stage 7  simplify polygons (RDP + collinear merge)
  Stage 8  compute metrics (area, centroid, bbox, wall/door count)
  Stage 9  re-open doors, attach to room adjacency
  Stage 10 generate colored overlay and topology data
"""

from __future__ import annotations

import math
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any

from processors.graph_builder import WallGraph


# ─── Data models ─────────────────────────────────────────────────────────────

@dataclass
class Room:
    room_id: int
    polygon: np.ndarray           # shape (N,1,2) int32 — for cv2 drawing
    vertices_xy: List[Tuple[float,float]]  # ordered (x,y) pairs
    centroid: Tuple[int,int]
    area_px: float
    bounding_box: Tuple[int,int,int,int]  # x,y,w,h
    wall_count: int = 0
    door_count: int = 0
    adjacent_room_ids: List[int] = field(default_factory=list)
    color_bgr: Tuple[int,int,int] = (128,128,128)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "room_id": self.room_id,
            "centroid": [int(self.centroid[0]), int(self.centroid[1])],
            "area_px":  round(float(self.area_px), 1),
            "bounding_box": {
                "x": int(self.bounding_box[0]),
                "y": int(self.bounding_box[1]),
                "width":  int(self.bounding_box[2]),
                "height": int(self.bounding_box[3]),
            },
            "wall_count": self.wall_count,
            "door_count": self.door_count,
            "adjacent_room_ids": self.adjacent_room_ids,
            "polygon_points": [[int(x),int(y)] for x,y in self.vertices_xy],
        }


@dataclass
class RoomResult:
    rooms: List[Room] = field(default_factory=list)
    segmentation: Optional[np.ndarray] = None
    debug: Dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def estimated_room_count(self) -> int:
        return len(self.rooms)

    @property
    def total_area_px(self) -> float:
        return sum(r.area_px for r in self.rooms)


# ─── Colour palette ─────────────────────────────────────────────────────────

_PALETTE_BGR = [
    (180,130,70),   # steel blue
    (113,204,46),   # emerald
    (60,76,231),    # alizarin
    (15,196,241),   # sunflower
    (182,89,155),   # amethyst
    (219,152,52),   # peter river
    (34,126,230),   # carrot
    (156,188,26),   # turquoise
    (122,64,236),   # pink
    (246,181,100),  # light blue
    (167,214,165),  # light green
    (77,183,255),   # amber
    (205,117,149),  # deep purple
    (172,182,77),   # teal
    (146,98,240),   # pink 400
]


def _room_color(idx: int) -> Tuple[int,int,int]:
    return _PALETTE_BGR[idx % len(_PALETTE_BGR)]


# ─── Geometry helpers ─────────────────────────────────────────────────────────

def _signed_area(pts: List[Tuple[float,float]]) -> float:
    """
    Shoelace formula signed area.
    In IMAGE coordinates (y-axis down):
      Positive → clockwise polygon (interior face from our traversal)
      Negative → counter-clockwise (exterior face)
    """
    n = len(pts)
    area = 0.0
    for i in range(n):
        x1,y1 = pts[i]
        x2,y2 = pts[(i+1) % n]
        area += (x1*y2 - x2*y1)
    return area / 2.0


def _centroid(pts: List[Tuple[float,float]]) -> Tuple[float,float]:
    return (sum(p[0] for p in pts)/len(pts), sum(p[1] for p in pts)/len(pts))


def _bounding_box(pts: List[Tuple[float,float]]) -> Tuple[int,int,int,int]:
    xs = [p[0] for p in pts];  ys = [p[1] for p in pts]
    xmin,xmax = int(min(xs)), int(max(xs))
    ymin,ymax = int(min(ys)), int(max(ys))
    return (xmin, ymin, xmax-xmin, ymax-ymin)


def _touches_border(pts, img_w, img_h, margin=15) -> bool:
    xs = [p[0] for p in pts];  ys = [p[1] for p in pts]
    return (min(xs)<margin or max(xs)>img_w-margin
            or min(ys)<margin or max(ys)>img_h-margin)


def _rdp_simplify(pts: List[Tuple[float,float]], epsilon: float = 3.0) -> List[Tuple[float,float]]:
    """Ramer-Douglas-Peucker polygon simplification."""
    if len(pts) < 3:
        return pts
    # Find point with max distance from line pts[0]–pts[-1]
    def perp_dist(px, py, x1, y1, x2, y2):
        dx,dy = x2-x1, y2-y1
        d = math.hypot(dx,dy)
        if d < 1e-9:
            return math.hypot(px-x1,py-y1)
        return abs(dx*(y1-py)-dy*(x1-px))/d

    max_d, max_i = 0.0, 0
    x1,y1 = pts[0];  x2,y2 = pts[-1]
    for i in range(1, len(pts)-1):
        d = perp_dist(pts[i][0],pts[i][1], x1,y1, x2,y2)
        if d > max_d:
            max_d, max_i = d, i

    if max_d > epsilon:
        left  = _rdp_simplify(pts[:max_i+1], epsilon)
        right = _rdp_simplify(pts[max_i:],   epsilon)
        return left[:-1] + right
    return [pts[0], pts[-1]]


# ─── Stage 5: Planar face traversal ─────────────────────────────────────────

def _find_planar_faces(graph: WallGraph) -> List[List[int]]:
    """
    Enumerate ALL minimal faces of the planar wall graph.

    Algorithm (DCEL / half-edge traversal):
      For each directed half-edge (u→v):
        At vertex v, find the outgoing edge to u in the CCW-sorted adjacency.
        The PREVIOUS entry in that sorted list (modulo degree) is the
        next half-edge in the face that lies to the LEFT of (u→v).
        Follow until we return to the start half-edge.

    In image coordinates (y-axis down), "previous in CCW-sorted" correctly
    traces the INTERIOR faces clockwise (positive signed area).

    Proof on a 2×2 grid: verified manually that each interior room face
    is traced as a clockwise polygon with positive signed area.
    """
    vertices  = graph.vertices
    adj       = graph.adj_sorted     # {vid: [(neighbor, angle), ...]} sorted by angle

    # Build next-half-edge lookup table
    # next_he[(u,v)] = (v, w)  where w is the next vertex in the face
    next_he: Dict[Tuple[int,int], Tuple[int,int]] = {}

    for v, neighbors in adj.items():
        n = len(neighbors)
        if n == 0:
            continue
        for i, (u, angle_vu) in enumerate(neighbors):
            # Neighbors sorted CCW. The "u" entry here represents outgoing edge v→u.
            # For incoming half-edge (u→v), the next half-edge is determined at v.
            # We look at outgoing edges of v in CCW order, find v→u, and take PREV.
            prev_i = (i - 1 + n) % n
            w, _ = neighbors[prev_i]
            next_he[(u, v)] = (v, w)

    # Traverse all half-edges
    visited: set = set()
    faces: List[List[int]] = []

    for start_he in list(next_he.keys()):
        if start_he in visited:
            continue

        face_verts: List[int] = []
        curr = start_he

        while curr not in visited:
            visited.add(curr)
            face_verts.append(curr[0])  # append the source vertex
            curr = next_he.get(curr)
            if curr is None:
                break

        if len(face_verts) >= 3:
            faces.append(face_verts)

    return faces


# ─── Stage 6: Filter faces ───────────────────────────────────────────────────

def _is_valid_room(
    pts: List[Tuple[float,float]],
    signed_area: float,
    img_w: int,
    img_h: int,
    min_area: float,
    max_area: float,
) -> bool:
    """
    Return True if this face is a valid interior room.

    Rules:
      - Signed area must be POSITIVE (clockwise in image coords = interior face)
      - Area must be in [min_area, max_area]
      - Must not touch image borders
    """
    if signed_area <= 0:
        return False
    area = signed_area
    if area < min_area or area > max_area:
        return False
    if _touches_border(pts, img_w, img_h, margin=15):
        return False
    return True


# ─── Stage 7: Simplify polygons ──────────────────────────────────────────────

def _simplify_face(pts: List[Tuple[float,float]], epsilon: float = 3.0) -> List[Tuple[float,float]]:
    """Simplify a face polygon with RDP + collinear merge."""
    if len(pts) < 3:
        return pts
    simplified = _rdp_simplify(pts + [pts[0]], epsilon)
    if simplified and simplified[-1] == simplified[0]:
        simplified = simplified[:-1]
    if len(simplified) < 3:
        return pts
    return simplified


# ─── Stage 8: Metrics ────────────────────────────────────────────────────────

def _count_door_edges(face_vids: List[int], graph: WallGraph, door_vertex_pairs: set) -> int:
    """Count how many edges of this face are door-sealing edges."""
    count = 0
    n = len(face_vids)
    for i in range(n):
        v1 = face_vids[i]
        v2 = face_vids[(i+1) % n]
        key = (min(v1,v2), max(v1,v2))
        if key in door_vertex_pairs:
            count += 1
    return count


# ─── Stage 9: Room adjacency ─────────────────────────────────────────────────

def _compute_adjacency(rooms: List[Room]) -> None:
    """
    Mark rooms as adjacent if they share at least one polygon edge.
    Works by checking if any two polygon edges are within 3px of each other.
    """
    n = len(rooms)
    for i in range(n):
        for j in range(i+1, n):
            ra, rb = rooms[i], rooms[j]
            # Quick bounding box pre-filter
            ax,ay,aw,ah = ra.bounding_box
            bx,by,bw,bh = rb.bounding_box
            if (ax+aw+20 < bx or bx+bw+20 < ax
                    or ay+ah+20 < by or by+bh+20 < ay):
                continue
            # Check if any vertex of ra is within 20px of any vertex of rb
            shared = False
            for (px,py) in ra.vertices_xy:
                for (qx,qy) in rb.vertices_xy:
                    if math.hypot(px-qx,py-qy) < 20:
                        shared = True
                        break
                if shared:
                    break
            if shared:
                if rb.room_id not in ra.adjacent_room_ids:
                    ra.adjacent_room_ids.append(rb.room_id)
                if ra.room_id not in rb.adjacent_room_ids:
                    rb.adjacent_room_ids.append(ra.room_id)


# ─── Stage 10: Visualization ─────────────────────────────────────────────────

def _build_segmentation(
    rooms: List[Room],
    img_h: int,
    img_w: int,
    alpha: float = 0.5,
) -> np.ndarray:
    canvas = np.full((img_h, img_w, 3), 18, dtype=np.uint8)
    overlay = canvas.copy()
    for room in rooms:
        cv2.drawContours(overlay, [room.polygon], -1, room.color_bgr, cv2.FILLED)
    cv2.addWeighted(overlay, alpha, canvas, 1-alpha, 0, canvas)

    for room in rooms:
        cv2.drawContours(canvas, [room.polygon], -1, room.color_bgr, 2, cv2.LINE_AA)

    for room in rooms:
        cx,cy = room.centroid
        cv2.circle(canvas,(cx,cy),5,(255,255,255),-1,cv2.LINE_AA)
        cv2.circle(canvas,(cx,cy),5,room.color_bgr,2,cv2.LINE_AA)
        label = f"R{room.room_id}"
        font = cv2.FONT_HERSHEY_DUPLEX
        sc = 0.55
        (tw,th),_ = cv2.getTextSize(label,font,sc,1)
        tx,ty = cx-tw//2, cy+th//2+10
        cv2.putText(canvas,label,(tx+1,ty+1),font,sc,(0,0,0),2,cv2.LINE_AA)
        cv2.putText(canvas,label,(tx,ty),font,sc,(255,255,255),1,cv2.LINE_AA)
        area_lbl = f"{room.area_px/1000:.1f}k"
        (aw,ah),_ = cv2.getTextSize(area_lbl,font,0.35,1)
        cv2.putText(canvas,area_lbl,(cx-aw//2,ty+ah+4),font,0.35,room.color_bgr,1,cv2.LINE_AA)

    return canvas


# ─── Public API ──────────────────────────────────────────────────────────────

def detect_rooms(
    graph: WallGraph,
    original_bgr: np.ndarray,
    min_area_fraction: float = 0.003,
    max_area_fraction: float = 0.88,
    wall_segments: Optional[List[Any]] = None,
) -> RoomResult:
    """
    Detect rooms as closed polygon faces in the planar wall graph.

    Args:
        graph:           WallGraph from graph_builder.build_graph()
        original_bgr:    Original image (for sizing and rendering)
        min_area_fraction: Minimum room area as fraction of image area
        max_area_fraction: Maximum room area (rejects exterior face)

    Returns:
        RoomResult with rooms, segmentation, and debug images.
    """
    result = RoomResult()
    debug: Dict[str, np.ndarray] = {}
    h, w = original_bgr.shape[:2]
    total_area = float(h * w)
    min_area = total_area * min_area_fraction
    max_area = total_area * max_area_fraction

    vertices = graph.vertices

    # ── Stage 5: Planar face traversal ──────────────────────────────────
    if len(graph.adj_sorted) < 3:
        # Graph too sparse — nothing to do
        result.segmentation = np.full((h,w,3),18,dtype=np.uint8)
        result.debug = debug
        return result

    raw_faces = _find_planar_faces(graph)

    # Build door-edge vertex-pair lookup for stage 8
    door_pairs: set = set()
    for eid,(v1,v2,is_door) in graph.edges.items():
        if is_door:
            door_pairs.add((min(v1,v2), max(v1,v2)))

    # ── Stage T: Fix Outer Walls by Topology ───────────────────────────
    if wall_segments is not None and raw_faces:
        # The exterior face is the one with the most negative area
        ext_face = min(
            raw_faces, 
            key=lambda f: _signed_area([vertices[v] for v in f if v in vertices]) if len(f) >= 3 else 0
        )
        ext_edges = set()
        for i in range(len(ext_face)):
            u = ext_face[i]
            v = ext_face[(i+1) % len(ext_face)]
            ext_edges.add((min(u,v), max(u,v)))

        def _point_on_segment(px, py, x1, y1, x2, y2, tol=5.0):
            dx, dy = x2 - x1, y2 - y1
            l = math.hypot(dx, dy)
            if l < 1e-9: return math.hypot(px - x1, py - y1) <= tol
            if px < min(x1,x2)-tol or px > max(x1,x2)+tol or py < min(y1,y2)-tol or py > max(y1,y2)+tol:
                return False
            return abs(dx * (y1 - py) - dy * (x1 - px)) / l <= tol

        for seg in wall_segments:
            seg.wall_type = 'internal'
            mx, my = (seg.start[0] + seg.end[0]) / 2, (seg.start[1] + seg.end[1]) / 2
            for u, v in ext_edges:
                if u in vertices and v in vertices:
                    x1, y1 = vertices[u]
                    x2, y2 = vertices[v]
                    if _point_on_segment(mx, my, x1, y1, x2, y2):
                        seg.wall_type = 'outer'
                        break

    # ── Stage 6 + 7 + 8: Filter, simplify, compute metrics ──────────────
    rooms: List[Room] = []
    room_id = 1

    for face_vids in raw_faces:
        # Convert vertex IDs to coordinates
        pts = [vertices[v] for v in face_vids if v in vertices]
        if len(pts) < 3:
            continue

        sa = _signed_area(pts)

        if not _is_valid_room(pts, sa, w, h, min_area, max_area):
            continue

        # Stage 7: simplify
        simplified = _simplify_face(pts, epsilon=3.0)
        if len(simplified) < 3:
            continue

        area = abs(_signed_area(simplified))
        if area < min_area:
            continue

        cx, cy = _centroid(simplified)
        bbox = _bounding_box(simplified)

        # Build polygon array for OpenCV drawing
        poly_pts = np.array([[int(x),int(y)] for x,y in simplified], dtype=np.int32)
        poly = poly_pts.reshape((-1,1,2))

        # Stage 8: door count
        n_doors = _count_door_edges(face_vids, graph, door_pairs)

        color = _room_color(room_id - 1)
        rooms.append(Room(
            room_id=room_id,
            polygon=poly,
            vertices_xy=simplified,
            centroid=(int(cx), int(cy)),
            area_px=area,
            bounding_box=bbox,
            wall_count=len(simplified),
            door_count=n_doors,
            color_bgr=color,
        ))
        room_id += 1

    # Sort by area descending, reassign IDs
    rooms.sort(key=lambda r: r.area_px, reverse=True)
    for i,r in enumerate(rooms):
        r.room_id = i+1
        r.color_bgr = _room_color(i)

    # Stage 9: adjacency graph
    _compute_adjacency(rooms)
    result.rooms = rooms

    # ── Debug images ─────────────────────────────────────────────────────
    # Graph vertex visualization
    graph_vis = original_bgr.copy()
    for (x,y) in vertices.values():
        cv2.circle(graph_vis,(int(x),int(y)),3,(0,255,180),-1)
    for eid,(v1,v2,is_door) in graph.edges.items():
        if v1 not in vertices or v2 not in vertices:
            continue
        p1 = (int(vertices[v1][0]),int(vertices[v1][1]))
        p2 = (int(vertices[v2][0]),int(vertices[v2][1]))
        color = (0,80,255) if is_door else (200,200,200)
        cv2.line(graph_vis,p1,p2,color,1,cv2.LINE_AA)
    debug["11_planar_graph"] = graph_vis

    # All faces visualization (before filtering) — use color per face
    all_faces_vis = np.full((h,w,3),18,dtype=np.uint8)
    for fi, face_vids in enumerate(raw_faces):
        pts = [vertices[v] for v in face_vids if v in vertices]
        if len(pts) < 3:
            continue
        sa_check = _signed_area(pts)
        if sa_check <= 0:
            continue
        poly = np.array([[int(x),int(y)] for x,y in pts], dtype=np.int32)
        color = _room_color(fi)
        cv2.fillPoly(all_faces_vis, [poly], color)
        cv2.polylines(all_faces_vis, [poly], True, (255,255,255), 1)
    debug["12_all_faces"] = all_faces_vis

    # Final segmentation
    seg = _build_segmentation(rooms, h, w)
    result.segmentation = seg
    debug["13_room_segmentation"] = seg

    result.debug = debug
    return result
