"""
processors/wall_detection.py  — REWRITE (Stage 2)
Blueprint-X Topology Pipeline

Goal: Convert the structural wall mask into a vector representation.

Pipeline:
  1. Skeletonize   → 1-pixel centerline of every wall
  2. HoughLinesP   → raw line segments on skeleton
  3. Merge         → collapse collinear / near-duplicate segments
  4. Extend        → snap nearby endpoints (T-junctions, corners)
  5. Gap detect    → find door-sized gaps between collinear segments
  6. Classify      → outer / internal by border proximity

Output: WallResult
  .wall_segments   List[WallSegment]   — clean vector walls
  .door_gaps       List[DoorGap]       — gaps that are likely doors
  .skeleton_mask   np.ndarray
  .wall_mask       np.ndarray          — original thick mask
  .debug           Dict[str, ndarray]
"""

from __future__ import annotations

import cv2
import math
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional


# ─── Data models ────────────────────────────────────────────────────────────

@dataclass
class WallSegment:
    start: Tuple[float, float]
    end: Tuple[float, float]
    wall_type: str = "internal"   # "outer" | "internal"
    thickness_px: float = 5.0

    def length(self) -> float:
        dx, dy = self.end[0]-self.start[0], self.end[1]-self.start[1]
        return math.hypot(dx, dy)

    def angle_deg(self) -> float:
        dx, dy = self.end[0]-self.start[0], self.end[1]-self.start[1]
        return math.degrees(math.atan2(dy, dx)) % 180

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start":     [round(self.start[0],1), round(self.start[1],1)],
            "end":       [round(self.end[0],1),   round(self.end[1],1)],
            "wall_type": self.wall_type,
            "length_px": round(self.length(), 2),
            "angle_deg": round(self.angle_deg(), 1),
        }


@dataclass
class DoorGap:
    """A gap between two collinear wall segments — likely a door opening."""
    start: Tuple[float, float]   # gap start point
    end: Tuple[float, float]     # gap end point
    width_px: float
    angle_deg: float             # wall orientation at this gap
    center: Tuple[float, float] = field(init=False)

    def __post_init__(self):
        self.center = (
            (self.start[0]+self.end[0])/2,
            (self.start[1]+self.end[1])/2,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "center": [round(self.center[0],1), round(self.center[1],1)],
            "width_px": round(self.width_px, 1),
            "angle_deg": round(self.angle_deg, 1),
        }


@dataclass
class WallResult:
    wall_segments: List[WallSegment] = field(default_factory=list)
    door_gaps: List[DoorGap] = field(default_factory=list)
    wall_mask: Optional[np.ndarray] = None
    skeleton_mask: Optional[np.ndarray] = None
    closed_mask: Optional[np.ndarray] = None   # kept for backward compat
    debug: Dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def total_wall_segments(self): return len(self.wall_segments)
    @property
    def outer_wall_count(self): return sum(1 for w in self.wall_segments if w.wall_type=="outer")
    @property
    def internal_wall_count(self): return sum(1 for w in self.wall_segments if w.wall_type=="internal")


# ─── Skeletonization ────────────────────────────────────────────────────────

def _skeletonize(binary: np.ndarray) -> np.ndarray:
    try:
        return cv2.ximgproc.thinning(
            binary, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN
        )
    except AttributeError:
        pass
    # Iterative morphological fallback
    skel = np.zeros_like(binary)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    img = binary.copy()
    while True:
        eroded = cv2.erode(img, element)
        temp   = cv2.dilate(eroded, element)
        temp   = cv2.subtract(img, temp)
        skel   = cv2.bitwise_or(skel, temp)
        img    = eroded.copy()
        if cv2.countNonZero(img) == 0:
            break
    return skel


# ─── Hough extraction ───────────────────────────────────────────────────────

def _hough_on_skeleton(skeleton: np.ndarray) -> List[Tuple[float,float,float,float]]:
    """Run probabilistic Hough on the skeleton, return (x1,y1,x2,y2) list."""
    # Dilate slightly so Hough finds more continuous segments
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    dilated = cv2.dilate(skeleton, k)
    raw = cv2.HoughLinesP(
        dilated,
        rho=1, theta=np.pi/180,
        threshold=25,
        minLineLength=20,
        maxLineGap=6,
    )
    if raw is None:
        return []
    return [(float(l[0][0]), float(l[0][1]), float(l[0][2]), float(l[0][3]))
            for l in raw]


# ─── Segment merging ────────────────────────────────────────────────────────

def _seg_angle(x1,y1,x2,y2) -> float:
    return math.degrees(math.atan2(y2-y1, x2-x1)) % 180


def _perpendicular_distance(px, py, x1, y1, x2, y2) -> float:
    dx, dy = x2-x1, y2-y1
    length = math.hypot(dx, dy)
    if length < 1e-9:
        return math.hypot(px-x1, py-y1)
    return abs(dx*(y1-py) - dy*(x1-px)) / length


def _project_point_onto_line(px, py, x1, y1, x2, y2) -> float:
    """Return the scalar t such that (x1,y1)+t*(x2-x1,y2-y1) is the projection."""
    dx, dy = x2-x1, y2-y1
    length2 = dx*dx + dy*dy
    if length2 < 1e-9:
        return 0.0
    return ((px-x1)*dx + (py-y1)*dy) / length2


def _merge_collinear_segments(
    segs: List[Tuple[float,float,float,float]],
    angle_tol: float = 5.0,
    perp_tol: float = 8.0,
    gap_tol: float = 10.0,
) -> List[Tuple[float,float,float,float]]:
    """
    Cluster near-collinear segments and merge overlapping projections.
    Returns merged segments and (separately identified) large gaps = doors.
    """
    if not segs:
        return []

    # Normalise direction so angle ∈ [0,180)
    normalised = []
    for x1,y1,x2,y2 in segs:
        a = _seg_angle(x1,y1,x2,y2)
        if x1 > x2 or (x1 == x2 and y1 > y2):
            x1,y1,x2,y2 = x2,y2,x1,y1
        normalised.append((x1,y1,x2,y2,a))

    used = [False]*len(normalised)
    merged = []

    for i, (x1,y1,x2,y2,ai) in enumerate(normalised):
        if used[i]:
            continue
        group = [(x1,y1,x2,y2)]
        used[i] = True

        for j, (px1,py1,px2,py2,aj) in enumerate(normalised):
            if used[j] or i == j:
                continue
            da = min(abs(ai-aj), 180-abs(ai-aj))
            if da > angle_tol:
                continue
            # Check perpendicular distance of midpoint of j from line of i
            mx, my = (px1+px2)/2, (py1+py2)/2
            dist = _perpendicular_distance(mx, my, x1,y1,x2,y2)
            if dist <= perp_tol:
                group.append((px1,py1,px2,py2))
                used[j] = True

        # Project all group segments onto the dominant axis of group
        # and merge overlapping intervals
        if not group:
            continue

        # Use the longest segment as the axis
        longest = max(group, key=lambda s: math.hypot(s[2]-s[0],s[3]-s[1]))
        ax1,ay1,ax2,ay2 = longest
        dx,dy = ax2-ax1, ay2-ay1
        length = math.hypot(dx,dy)
        if length < 1e-9:
            continue

        # Collect all projection intervals
        intervals = []
        for (sx1,sy1,sx2,sy2) in group:
            t1 = _project_point_onto_line(sx1,sy1, ax1,ay1,ax2,ay2)
            t2 = _project_point_onto_line(sx2,sy2, ax1,ay1,ax2,ay2)
            if t1 > t2:
                t1,t2 = t2,t1
            intervals.append((t1,t2))

        # Sort and merge intervals with gap tolerance
        intervals.sort()
        curr_t1, curr_t2 = intervals[0]
        for t1,t2 in intervals[1:]:
            if t1 <= curr_t2 + gap_tol/length:
                curr_t2 = max(curr_t2, t2)
            else:
                # Emit current merged segment
                ex1 = ax1 + curr_t1*dx; ey1 = ay1 + curr_t1*dy
                ex2 = ax1 + curr_t2*dx; ey2 = ay1 + curr_t2*dy
                merged.append((ex1,ey1,ex2,ey2))
                curr_t1,curr_t2 = t1,t2
        ex1 = ax1 + curr_t1*dx; ey1 = ay1 + curr_t1*dy
        ex2 = ax1 + curr_t2*dx; ey2 = ay1 + curr_t2*dy
        merged.append((ex1,ey1,ex2,ey2))

    return merged


# ─── Gap detection (doors) ──────────────────────────────────────────────────

def _detect_door_gaps(
    segs: List[Tuple[float,float,float,float]],
    angle_tol: float = 5.0,
    perp_tol: float = 8.0,
    gap_min: float = 14.0,
    gap_max: float = 65.0,
) -> List[DoorGap]:
    """
    Find gaps between collinear wall segments that are in the door-width range.
    These gaps represent door openings.
    """
    if not segs:
        return []

    normalised = []
    for x1,y1,x2,y2 in segs:
        a = _seg_angle(x1,y1,x2,y2)
        if x1 > x2 or (x1==x2 and y1>y2):
            x1,y1,x2,y2 = x2,y2,x1,y1
        normalised.append((x1,y1,x2,y2,a))

    used = [False]*len(normalised)
    gaps: List[DoorGap] = []

    for i,(ax1,ay1,ax2,ay2,ai) in enumerate(normalised):
        group = [(ax1,ay1,ax2,ay2)]
        for j,(px1,py1,px2,py2,aj) in enumerate(normalised):
            if i==j: continue
            da = min(abs(ai-aj),180-abs(ai-aj))
            if da>angle_tol: continue
            mx,my = (px1+px2)/2,(py1+py2)/2
            dist = _perpendicular_distance(mx,my,ax1,ay1,ax2,ay2)
            if dist<=perp_tol:
                group.append((px1,py1,px2,py2))

        if len(group)<2:
            continue

        dx,dy = ax2-ax1,ay2-ay1
        length = math.hypot(dx,dy)
        if length<1e-9: continue

        intervals = []
        for (sx1,sy1,sx2,sy2) in group:
            t1 = _project_point_onto_line(sx1,sy1,ax1,ay1,ax2,ay2)
            t2 = _project_point_onto_line(sx2,sy2,ax1,ay1,ax2,ay2)
            if t1>t2: t1,t2=t2,t1
            intervals.append((t1,t2))
        intervals.sort()

        # Find gaps between consecutive intervals
        curr_t2 = intervals[0][1]
        for t1,t2 in intervals[1:]:
            gap_len = (t1 - curr_t2) * length
            if gap_min <= gap_len <= gap_max:
                # Gap found
                g_x1 = ax1 + curr_t2*dx;  g_y1 = ay1 + curr_t2*dy
                g_x2 = ax1 + t1*dx;        g_y2 = ay1 + t1*dy
                gaps.append(DoorGap(
                    start=(g_x1,g_y1),
                    end=(g_x2,g_y2),
                    width_px=gap_len,
                    angle_deg=ai,
                ))
            curr_t2 = max(curr_t2,t2)

    # Deduplicate (same centre within 15px → same gap)
    deduped: List[DoorGap] = []
    for g in gaps:
        dup = False
        for d in deduped:
            if math.hypot(g.center[0]-d.center[0], g.center[1]-d.center[1]) < 15:
                dup=True; break
        if not dup:
            deduped.append(g)

    return deduped


# ─── Endpoint snapping ──────────────────────────────────────────────────────

def _snap_endpoints(
    segs: List[Tuple[float,float,float,float]],
    snap_dist: float = 14.0,
) -> List[Tuple[float,float,float,float]]:
    """
    Snap nearby segment endpoints to each other to form proper T/L-junctions.
    Only snaps endpoints that are on the same wall line (similar angle and close).
    """
    pts = []
    for seg in segs:
        pts.append((seg[0],seg[1]))
        pts.append((seg[2],seg[3]))

    # For each endpoint, find others within snap_dist and take centroid
    n = len(pts)
    snapped = list(pts)

    for i in range(n):
        cluster = [pts[i]]
        for j in range(n):
            if i==j: continue
            if math.hypot(pts[i][0]-pts[j][0], pts[i][1]-pts[j][1]) < snap_dist:
                cluster.append(pts[j])
        if len(cluster)>1:
            cx = sum(p[0] for p in cluster)/len(cluster)
            cy = sum(p[1] for p in cluster)/len(cluster)
            snapped[i] = (cx,cy)

    result = []
    for k,(seg) in enumerate(segs):
        x1,y1 = snapped[k*2]
        x2,y2 = snapped[k*2+1]
        result.append((x1,y1,x2,y2))
    return result


# ─── Debug overlay ──────────────────────────────────────────────────────────

def _draw_segments_debug(bgr, segments: List[WallSegment]) -> np.ndarray:
    canvas = cv2.addWeighted(bgr, 0.35, np.zeros_like(bgr), 0.65, 0)
    for seg in segments:
        color = (0,130,255)
        thick = 2
        p1 = (int(seg.start[0]),int(seg.start[1]))
        p2 = (int(seg.end[0]),  int(seg.end[1]))
        cv2.line(canvas, p1, p2, color, thick, cv2.LINE_AA)
        cv2.circle(canvas, p1, 4, (0,255,180), -1)
        cv2.circle(canvas, p2, 4, (0,255,180), -1)
    return canvas


# ─── Public API ─────────────────────────────────────────────────────────────

def detect_walls(
    cleaned_binary: np.ndarray,
    original_bgr: np.ndarray,
) -> WallResult:
    """
    Convert the structural-only wall mask into a vector representation.

    Returns WallResult with wall_segments and door_gaps.
    """
    result = WallResult()
    debug: Dict[str, np.ndarray] = {}
    h, w = cleaned_binary.shape[:2]

    result.wall_mask = cleaned_binary

    # ── Skeletonize ───────────────────────────────────────────────────────
    skeleton = _skeletonize(cleaned_binary)
    result.skeleton_mask = skeleton
    skel_vis = original_bgr.copy()
    skel_vis[skeleton>0] = (0,255,180)
    debug["07_skeleton"] = skel_vis

    # ── Hough ─────────────────────────────────────────────────────────────
    raw_segs = _hough_on_skeleton(skeleton)
    raw_vis = original_bgr.copy()
    for x1,y1,x2,y2 in raw_segs:
        cv2.line(raw_vis,(int(x1),int(y1)),(int(x2),int(y2)),(80,80,255),1,cv2.LINE_AA)
    debug["08_hough_raw"] = raw_vis

    # ── Merge collinear segments ──────────────────────────────────────────
    merged = _merge_collinear_segments(raw_segs, angle_tol=5, perp_tol=8, gap_tol=10)

    # ── Detect door gaps (before endpoint snapping changes the geometry) ──
    door_gaps = _detect_door_gaps(merged, gap_min=14, gap_max=100)
    result.door_gaps = door_gaps

    # ── Snap endpoints to form proper junctions ───────────────────────────
    snapped = _snap_endpoints(merged, snap_dist=14)

    # ── Build WallSegment objects ─────────────────────────────────────────
    segments: List[WallSegment] = []
    for x1,y1,x2,y2 in snapped:
        if math.hypot(x2-x1,y2-y1) < 10:
            continue
        segments.append(WallSegment(start=(x1,y1), end=(x2,y2), wall_type='internal'))

    result.wall_segments = segments
    debug["09_wall_segments"] = _draw_segments_debug(original_bgr, segments)

    # Keep closed mask for backward compatibility with visualization
    gap_k = cv2.getStructuringElement(cv2.MORPH_RECT,(12,12))
    result.closed_mask = cv2.morphologyEx(cleaned_binary, cv2.MORPH_CLOSE, gap_k)

    result.debug = debug
    return result
