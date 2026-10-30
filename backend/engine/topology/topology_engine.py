"""
backend/engine/topology/topology_engine.py
Blueprint-X — Milestone 2

Extracts planar faces (rooms) from the BuildingGraph.
Builds a NetworkX room adjacency graph.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from backend.engine.geometry.geometry_engine import BuildingGraph
from backend.models.dataclasses import TopologyEdge, TopologyNode, Room, Door, Wall

try:
    import networkx as nx
    _HAS_NX = True
except ImportError:
    _HAS_NX = False


@dataclass
class TopologyResult:
    faces: List[List[int]] = field(default_factory=list)               # Lists of vertex IDs
    nodes: List[TopologyNode] = field(default_factory=list)
    edges: List[TopologyEdge] = field(default_factory=list)
    debug: Dict[str, np.ndarray] = field(default_factory=dict)
    exterior_face_idx: int = -1


class TopologyEngine:
    """
    Extracts planar faces from the BuildingGraph to form rooms.
    Generates a node-link topology graph.
    """

    def run(
        self,
        graph: BuildingGraph,
        door_gaps: List,
    ) -> TopologyResult:
        result = TopologyResult()

        if graph.n_vertices == 0:
            return result

        # 1. Find all closed faces using DCEL leftmost-turn traversal
        faces = self._find_planar_faces(graph)
        if not faces:
            return result

        # 2. Identify the exterior face (polygon with negative signed area)
        exterior_idx = -1
        min_area = float("inf")
        for i, face_vids in enumerate(faces):
            area = self._signed_polygon_area(face_vids, graph.vertices)
            if area < min_area:
                min_area = area
                exterior_idx = i

        result.faces = faces
        result.exterior_face_idx = exterior_idx

        # 3. Create Nodes
        nodes = []
        for i, face_vids in enumerate(faces):
            if i == exterior_idx:
                continue
            cx, cy = self._polygon_centroid(face_vids, graph.vertices)
            nodes.append(TopologyNode(room_id=i, centroid=(cx, cy)))

        # 4. Create Edges (paths through doors)
        edges = []
        # Find which faces share which edges
        edge_to_faces = self._map_edges_to_faces(faces)

        # Check door gaps to link faces
        for eid, (v1, v2, is_door) in graph.edges.items():
            if not is_door:
                continue
            
            # Find the two faces that share this door edge
            shared = []
            for face_idx, vids in enumerate(faces):
                if face_idx == exterior_idx:
                    continue
                # check if v1, v2 are adjacent in face_vids
                for k in range(len(vids)):
                    if (vids[k] == v1 and vids[(k+1)%len(vids)] == v2) or \
                       (vids[k] == v2 and vids[(k+1)%len(vids)] == v1):
                        shared.append(face_idx)
                        break

            if len(shared) == 2:
                # Add a topology edge
                x1, y1 = graph.vertices[v1]
                x2, y2 = graph.vertices[v2]
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                edges.append(TopologyEdge(
                    source_id=shared[0],
                    target_id=shared[1],
                    edge_type="door",
                    door_center=(cx, cy),
                ))

        result.nodes = nodes
        result.edges = edges
        return result

    def build_room_topology(
        self,
        rooms: List[Room],
        doors: Optional[List[Door]] = None,
        walls: Optional[List[Wall]] = None,
    ) -> TopologyResult:
        """
        Builds a comprehensive, connected room adjacency & door connectivity topology graph
        from room polygons and doors.
        """
        result = TopologyResult()
        valid_rooms = [r for r in rooms if r.polygon_pts and len(r.polygon_pts) >= 3]
        if not valid_rooms:
            return result

        nodes = [
            TopologyNode(room_id=r.room_id, centroid=r.centroid, label=r.label)
            for r in valid_rooms
        ]

        edges: List[TopologyEdge] = []
        added_pairs = set()

        # 1. Door connectivity
        doors = doors or []
        for d in doors:
            dcx, dcy = d.center
            connected_rooms = []
            for idx, r in enumerate(valid_rooms):
                if self._dist_point_to_polygon(dcx, dcy, r.polygon_pts) < 50.0:
                    connected_rooms.append(idx)

            if len(connected_rooms) >= 2:
                for i in range(len(connected_rooms)):
                    for j in range(i + 1, len(connected_rooms)):
                        rm1 = valid_rooms[connected_rooms[i]]
                        rm2 = valid_rooms[connected_rooms[j]]
                        r1, r2 = rm1.room_id, rm2.room_id
                        if r1 != r2:
                            pair = tuple(sorted((r1, r2)))
                            if pair not in added_pairs:
                                added_pairs.add(pair)
                                edges.append(TopologyEdge(
                                    source_id=r1,
                                    target_id=r2,
                                    edge_type="door",
                                    door_center=d.center,
                                ))
                            if r2 not in rm1.adjacent_room_ids:
                                rm1.adjacent_room_ids.append(r2)
                            if r1 not in rm2.adjacent_room_ids:
                                rm2.adjacent_room_ids.append(r1)
                            if d.id not in rm1.door_ids:
                                rm1.door_ids.append(d.id)
                            if d.id not in rm2.door_ids:
                                rm2.door_ids.append(d.id)
                            if r1 not in d.connected_rooms:
                                d.connected_rooms.append(r1)
                            if r2 not in d.connected_rooms:
                                d.connected_rooms.append(r2)

        # 2. Shared wall boundary adjacency
        for i in range(len(valid_rooms)):
            r1 = valid_rooms[i]
            for j in range(i + 1, len(valid_rooms)):
                r2 = valid_rooms[j]
                pair = tuple(sorted((r1.room_id, r2.room_id)))
                if pair in added_pairs:
                    continue

                is_adj, mid_pt = self._check_room_adjacency(r1.polygon_pts, r2.polygon_pts)
                if is_adj:
                    added_pairs.add(pair)
                    edges.append(TopologyEdge(
                        source_id=r1.room_id,
                        target_id=r2.room_id,
                        edge_type="door",
                        door_center=mid_pt,
                    ))
                    if r2.room_id not in r1.adjacent_room_ids:
                        r1.adjacent_room_ids.append(r2.room_id)
                    if r1.room_id not in r2.adjacent_room_ids:
                        r2.adjacent_room_ids.append(r1.room_id)

        result.nodes = nodes
        result.edges = edges
        return result

    @staticmethod
    def _dist_point_to_polygon(px: float, py: float, poly: List[Tuple[float, float]]) -> float:
        n = len(poly)
        if n < 3:
            return 1e9
        min_d = 1e9
        for i in range(n):
            ax, ay = poly[i]
            bx, by = poly[(i + 1) % n]
            dx, dy = bx - ax, by - ay
            L2 = dx * dx + dy * dy
            if L2 < 1e-6:
                d = math.hypot(px - ax, py - ay)
            else:
                t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / L2))
                d = math.hypot(px - (ax + t * dx), py - (ay + t * dy))
            if d < min_d:
                min_d = d
        return min_d

    @staticmethod
    def _check_room_adjacency(
        poly1: List[Tuple[float, float]],
        poly2: List[Tuple[float, float]],
        max_dist: float = 65.0,
        min_overlap: float = 18.0,
    ) -> Tuple[bool, Optional[Tuple[float, float]]]:
        n1, n2 = len(poly1), len(poly2)
        if n1 < 3 or n2 < 3:
            return False, None

        for i in range(n1):
            a1 = poly1[i]
            a2 = poly1[(i + 1) % n1]
            ax, ay = a2[0] - a1[0], a2[1] - a1[1]
            lenA = math.hypot(ax, ay)
            if lenA < 1e-4:
                continue
            ux, uy = ax / lenA, ay / lenA

            for j in range(n2):
                b1 = poly2[j]
                b2 = poly2[(j + 1) % n2]
                bx, by = b2[0] - b1[0], b2[1] - b1[1]
                lenB = math.hypot(bx, by)
                if lenB < 1e-4:
                    continue

                dot = abs(ux * (bx / lenB) + uy * (by / lenB))
                if dot < 0.85:
                    continue

                # Perpendicular distance to line A
                perp1 = abs((b1[0] - a1[0]) * uy - (b1[1] - a1[1]) * ux)
                perp2 = abs((b2[0] - a1[0]) * uy - (b2[1] - a1[1]) * ux)
                if max(perp1, perp2) > max_dist:
                    continue

                # 1D projection overlap
                t_b1 = (b1[0] - a1[0]) * ux + (b1[1] - a1[1]) * uy
                t_b2 = (b2[0] - a1[0]) * ux + (b2[1] - a1[1]) * uy
                t_min = max(0.0, min(t_b1, t_b2))
                t_max = min(lenA, max(t_b1, t_b2))

                overlap = t_max - t_min
                if overlap >= min_overlap:
                    mid_t = (t_min + t_max) / 2.0
                    cx = a1[0] + ux * mid_t
                    cy = a1[1] + uy * mid_t
                    return True, (cx, cy)

        return False, None

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _find_planar_faces(self, graph: BuildingGraph) -> List[List[int]]:
        visited_half_edges: Set[Tuple[int, int]] = set()
        faces: List[List[int]] = []

        for u in graph.adj_sorted:
            for (v, _) in graph.adj_sorted[u]:
                edge = (u, v)
                if edge in visited_half_edges:
                    continue

                face = [u]
                curr = u
                nxt = v

                while True:
                    if (curr, nxt) in visited_half_edges:
                        break
                    visited_half_edges.add((curr, nxt))
                    face.append(nxt)

                    neighbors = graph.adj_sorted[nxt]
                    if len(neighbors) == 1:
                        # Dead end, turn around
                        next_node = neighbors[0][0]
                        curr, nxt = nxt, next_node
                        if curr == u and nxt == v:
                            break
                        continue

                    idx = -1
                    for i, (nn, _) in enumerate(neighbors):
                        if nn == curr:
                            idx = i
                            break

                    if idx == -1:
                        break

                    # Pick the leftmost turn (idx - 1 in CW sorted angle)
                    next_idx = (idx - 1) % len(neighbors)
                    next_node = neighbors[next_idx][0]

                    curr, nxt = nxt, next_node
                    if curr == u and nxt == v:
                        break

                if len(face) > 2 and face[0] == face[-1]:
                    face.pop()
                    faces.append(face)

        return faces

    def _signed_polygon_area(self, vids: List[int], vertices: Dict[int, Tuple[float, float]]) -> float:
        area = 0.0
        n = len(vids)
        for i in range(n):
            j = (i + 1) % n
            x1, y1 = vertices[vids[i]]
            x2, y2 = vertices[vids[j]]
            area += (x1 * y2 - x2 * y1)
        return area / 2.0

    def _polygon_centroid(self, vids: List[int], vertices: Dict[int, Tuple[float, float]]) -> Tuple[float, float]:
        cx = 0.0
        cy = 0.0
        area = 0.0
        n = len(vids)
        for i in range(n):
            j = (i + 1) % n
            x1, y1 = vertices[vids[i]]
            x2, y2 = vertices[vids[j]]
            f = x1 * y2 - x2 * y1
            cx += (x1 + x2) * f
            cy += (y1 + y2) * f
            area += f
        area *= 0.5
        if abs(area) < 1e-9:
            # Fallback for degenerate polygons
            return (
                sum(vertices[v][0] for v in vids) / n,
                sum(vertices[v][1] for v in vids) / n
            )
        cx = cx / (6.0 * area)
        cy = cy / (6.0 * area)
        return cx, cy

    def _map_edges_to_faces(self, faces: List[List[int]]) -> Dict[Tuple[int, int], List[int]]:
        mapping = defaultdict(list)
        for face_idx, vids in enumerate(faces):
            n = len(vids)
            for i in range(n):
                u = vids[i]
                v = vids[(i + 1) % n]
                edge = tuple(sorted((u, v)))
                mapping[edge].append(face_idx)
        return mapping
