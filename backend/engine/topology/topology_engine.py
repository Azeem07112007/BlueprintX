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
from backend.models.dataclasses import TopologyEdge, TopologyNode

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
