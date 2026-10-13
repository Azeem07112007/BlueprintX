"""
backend/models/dataclasses.py
Blueprint-X — Milestone 2

Canonical data models for all structural elements.
Every object has: id, geometry, confidence, metadata.
All downstream modules (topology, semantics, export, UI) consume these.
"""
from __future__ import annotations

import uuid
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ─── 3-D primitive types ──────────────────────────────────────────────────────
Vec3 = Tuple[float, float, float]


# ─── Base ─────────────────────────────────────────────────────────────────────

@dataclass
class BaseElement:
    """All blueprint elements inherit from this."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


# ─── Structural Elements ───────────────────────────────────────────────────────

@dataclass
class Wall(BaseElement):
    """A single vector wall segment."""
    start: Tuple[float, float] = (0.0, 0.0)
    end: Tuple[float, float] = (0.0, 0.0)
    thickness_px: float = 5.0
    wall_type: str = "internal"   # "outer" | "internal"

    @property
    def length(self) -> float:
        dx, dy = self.end[0] - self.start[0], self.end[1] - self.start[1]
        return math.hypot(dx, dy)

    @property
    def angle_deg(self) -> float:
        dx, dy = self.end[0] - self.start[0], self.end[1] - self.start[1]
        return math.degrees(math.atan2(dy, dx)) % 180

    @property
    def midpoint(self) -> Tuple[float, float]:
        return ((self.start[0] + self.end[0]) / 2, (self.start[1] + self.end[1]) / 2)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "start": [round(self.start[0], 1), round(self.start[1], 1)],
            "end": [round(self.end[0], 1), round(self.end[1], 1)],
            "length_px": round(self.length, 2),
            "thickness_px": round(self.thickness_px, 1),
            "angle_deg": round(self.angle_deg, 1),
            "wall_type": self.wall_type,
            "confidence": round(self.confidence, 3),
        }


@dataclass
class Door(BaseElement):
    """A detected door opening."""
    center: Tuple[float, float] = (0.0, 0.0)
    width_px: float = 0.0
    height_px: float = 0.0
    angle_deg: float = 0.0
    bbox: Optional[Tuple[int, int, int, int]] = None   # x,y,w,h
    mask: Optional[np.ndarray] = None
    has_arc: bool = False
    source: str = "gap"   # "gap" (geometric) | "yolo" (AI)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "center": [round(self.center[0], 1), round(self.center[1], 1)],
            "width_px": round(self.width_px, 1),
            "height_px": round(self.height_px, 1),
            "angle_deg": round(self.angle_deg, 1),
            "has_arc": self.has_arc,
            "source": self.source,
            "confidence": round(self.confidence, 3),
        }


@dataclass
class Window(BaseElement):
    """A detected window."""
    center: Tuple[float, float] = (0.0, 0.0)
    width_px: float = 0.0
    height_px: float = 0.0
    angle_deg: float = 0.0
    bbox: Optional[Tuple[int, int, int, int]] = None
    mask: Optional[np.ndarray] = None
    source: str = "yolo"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "center": [round(self.center[0], 1), round(self.center[1], 1)],
            "width_px": round(self.width_px, 1),
            "height_px": round(self.height_px, 1),
            "angle_deg": round(self.angle_deg, 1),
            "source": self.source,
            "confidence": round(self.confidence, 3),
        }


@dataclass
class Furniture(BaseElement):
    """Any detected furniture or fixture."""
    class_name: str = "furniture"      # e.g. "bed", "sofa", "dining_table", "kitchen_counter", "toilet", "sink", "stair", "plant", "tv_unit"
    bbox: Optional[Tuple[int, int, int, int]] = None
    center: Tuple[float, float] = (0.0, 0.0)
    rotation_deg: float = 0.0
    room_label: str = ""
    sub_type: str = ""
    mask: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "class": self.class_name,
            "center": [round(self.center[0], 1), round(self.center[1], 1)],
            "bbox": list(self.bbox) if self.bbox else None,
            "rotation_deg": round(self.rotation_deg, 1),
            "room_label": self.room_label,
            "sub_type": self.sub_type,
            "confidence": round(self.confidence, 3),
        }


@dataclass
class Column(BaseElement):
    """A structural column."""
    center: Tuple[float, float] = (0.0, 0.0)
    radius_px: float = 10.0
    bbox: Optional[Tuple[int, int, int, int]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "center": [round(self.center[0], 1), round(self.center[1], 1)],
            "radius_px": round(self.radius_px, 1),
            "confidence": round(self.confidence, 3),
        }


@dataclass
class Stair(BaseElement):
    """A staircase."""
    bbox: Optional[Tuple[int, int, int, int]] = None
    center: Tuple[float, float] = (0.0, 0.0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "center": [round(self.center[0], 1), round(self.center[1], 1)],
            "bbox": list(self.bbox) if self.bbox else None,
            "confidence": round(self.confidence, 3),
        }


@dataclass
class TextBlock(BaseElement):
    """An OCR-extracted text block."""
    text: str = ""
    bbox: Optional[Tuple[int, int, int, int]] = None
    center: Tuple[float, float] = (0.0, 0.0)
    is_room_label: bool = False
    is_dimension: bool = False
    is_scale: bool = False
    is_window_label: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "center": [round(self.center[0], 1), round(self.center[1], 1)],
            "bbox": list(self.bbox) if self.bbox else None,
            "is_room_label": self.is_room_label,
            "is_dimension": self.is_dimension,
            "is_scale": self.is_scale,
            "is_window_label": self.is_window_label,
            "confidence": round(self.confidence, 3),
        }


# ─── Room ──────────────────────────────────────────────────────────────────────

@dataclass
class Room(BaseElement):
    """A detected room polygon."""
    room_id: int = 0
    polygon_pts: List[Tuple[float, float]] = field(default_factory=list)
    centroid: Tuple[float, float] = (0.0, 0.0)
    area_px: float = 0.0
    area_m2: float = 0.0       # computed from area_px + scale; 0.0 if scale unknown
    bounding_box: Tuple[int, int, int, int] = (0, 0, 0, 0)
    label: str = ""            # "Kitchen", "Bathroom", etc. from OCR/semantic
    color_bgr: Tuple[int, int, int] = (128, 128, 128)
    wall_ids: List[str] = field(default_factory=list)
    door_ids: List[str] = field(default_factory=list)
    window_ids: List[str] = field(default_factory=list)
    adjacent_room_ids: List[int] = field(default_factory=list)

    @property
    def polygon_np(self) -> np.ndarray:
        pts = np.array([[int(x), int(y)] for x, y in self.polygon_pts], dtype=np.int32)
        return pts.reshape((-1, 1, 2))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "room_id": self.room_id,
            "label": self.label,
            "centroid": [round(self.centroid[0], 1), round(self.centroid[1], 1)],
            "area_px": round(self.area_px, 1),
            "area_m2": round(self.area_m2, 2) if self.area_m2 > 0 else None,
            "bounding_box": {
                "x": self.bounding_box[0], "y": self.bounding_box[1],
                "width": self.bounding_box[2], "height": self.bounding_box[3],
            },
            "polygon_points": [[round(x, 1), round(y, 1)] for x, y in self.polygon_pts],
            "polygon_flat": [round(c, 1) for pt in self.polygon_pts for c in pt],
            "adjacent_room_ids": self.adjacent_room_ids,
            "door_ids": self.door_ids,
            "window_ids": self.window_ids,
            "confidence": round(self.confidence, 3),
        }


# ─── Topology ─────────────────────────────────────────────────────────────────

@dataclass
class TopologyNode:
    room_id: int
    centroid: Tuple[float, float]
    label: str = ""
    color_bgr: Tuple[int, int, int] = (128, 128, 128)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "room_id": self.room_id,
            "centroid": [round(self.centroid[0], 1), round(self.centroid[1], 1)],
            "label": self.label,
        }


@dataclass
class TopologyEdge:
    source_id: int
    target_id: int
    edge_type: str = "door"    # "door" | "passage"
    door_center: Optional[Tuple[float, float]] = None
    door_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "source": self.source_id,
            "target": self.target_id,
            "type": self.edge_type,
        }
        if self.door_center:
            d["door_center"] = [round(self.door_center[0], 1), round(self.door_center[1], 1)]
        return d


# ─── Top-Level Building ────────────────────────────────────────────────────────

@dataclass
class Building:
    """The root object. All Blueprint-X analysis produces a Building."""
    source_path: str = ""
    image_size: Tuple[int, int] = (0, 0)    # (width, height)
    scale: str = ""                           # e.g. "1:100"

    walls: List[Wall] = field(default_factory=list)
    doors: List[Door] = field(default_factory=list)
    windows: List[Window] = field(default_factory=list)
    furniture: List[Furniture] = field(default_factory=list)
    columns: List[Column] = field(default_factory=list)
    stairs: List[Stair] = field(default_factory=list)
    rooms: List[Room] = field(default_factory=list)
    text_blocks: List[TextBlock] = field(default_factory=list)

    topology_nodes: List[TopologyNode] = field(default_factory=list)
    topology_edges: List[TopologyEdge] = field(default_factory=list)

    # ── Milestone 4 ──────────────────────────────────────────────────────────
    model_3d: Optional["Building3D"] = None


# ─── 3-D Building Model ───────────────────────────────────────────────────────

@dataclass
class Building3D:
    """
    GPU-ready 3-D representation of a floor plan.

    All geometry is stored as flat numpy arrays so it can be uploaded
    directly to an OpenGL VBO without conversion.

    Coordinate system:
        +X right   (blueprint X / scale)
        +Y up      (elevation)
        +Z depth   (blueprint Y / scale, note Y-flip)

    Interleaved VBO layout per vertex:  [x, y, z,  nx, ny, nz,  r, g, b]
    """
    vertices:  np.ndarray = field(default_factory=lambda: np.zeros((0, 3), np.float32))  # (N, 3)
    normals:   np.ndarray = field(default_factory=lambda: np.zeros((0, 3), np.float32))  # (N, 3)
    colors:    np.ndarray = field(default_factory=lambda: np.zeros((0, 3), np.float32))  # (N, 3) 0-1 RGB
    indices:   np.ndarray = field(default_factory=lambda: np.zeros((0, 3), np.int32))    # (M, 3) triangles

    floor_height_m: float = 2.8
    px_per_m:       float = 100.0
    vertex_count:   int   = 0
    triangle_count: int   = 0
    center:         Vec3  = (0.0, 0.0, 0.0)   # world-space centre for camera targeting
    bounds_min:     Vec3  = (0.0, 0.0, 0.0)
    bounds_max:     Vec3  = (0.0, 0.0, 0.0)

    def interleaved(self) -> np.ndarray:
        """Return (N, 9) float32 array: [pos(3), norm(3), color(3)] per vertex."""
        if self.vertex_count == 0:
            return np.zeros((0, 9), np.float32)
        return np.concatenate([self.vertices, self.normals, self.colors], axis=1).astype(np.float32)

    def flat_indices(self) -> np.ndarray:
        """Return a flat 1-D int32 index array suitable for glDrawElements."""
        return self.indices.flatten().astype(np.int32)


# ─── AI Detection Container ───────────────────────────────────────────────────

@dataclass
class DetectedObject:
    """Raw YOLO detection before semantic fusion."""
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]        # x, y, w, h
    center: Tuple[float, float]
    mask: Optional[np.ndarray] = None      # segmentation mask if available
