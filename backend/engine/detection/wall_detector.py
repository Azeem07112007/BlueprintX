"""
backend/engine/detection/wall_detector.py
Blueprint-X — M4 (Accuracy Overhaul)

Wall detection engine. Pure OpenCV + geometry approach.

Pipeline (v4):
  1. Skeletonize the cleaned binary mask
  2. Multi-scale probabilistic Hough to catch walls at all lengths
  3. Direction-based clustering (group by angle bucket, then by perpendicular offset)
  4. Envelope merge: for each cluster, form the tightest bounding box extent
  5. Detect door gaps (collinear aligned segments with appropriate gap sizes)
  6. Snap endpoints (cluster near endpoints to a common junction)
  7. Build Wall dataclasses

Key improvements over v3:
  - Direction clustering uses sorted angle buckets to avoid greedy-skip bugs
  - Two-pass merge: (a) within direction clusters merge by perpendicular proximity,
    (b) second pass merges remaining collinear near-duplicate fragments
  - Snap tolerance now scales with image density
  - Geometry engine large_gap threshold raised (120 px) for complex plans
  - T-junction splitting added to geometry engine
"""
from __future__ import annotations

import cv2
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from backend.models.dataclasses import Wall, Door


@dataclass
class WallDetectionResult:
    walls: List[Wall] = field(default_factory=list)
    door_gaps: List[Door] = field(default_factory=list)
    wall_mask: Optional[np.ndarray] = None
    skeleton_mask: Optional[np.ndarray] = None
    debug: Dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def outer_walls(self) -> List[Wall]:
        return [w for w in self.walls if w.wall_type == "outer"]

    @property
    def internal_walls(self) -> List[Wall]:
        return [w for w in self.walls if w.wall_type == "internal"]


class WallDetector:
    """
    Converts a cleaned binary wall mask into vector Wall objects.

    Three parameter presets are chosen based on blueprint style and complexity:
      "simple"   — clean line drawings (black on white)
      "rendered" — colour floor plans with thick filled walls
      "complex"  — high-density blueprints with many rooms and thin walls

    Rendered colour floor plans have thicker walls appearing as two parallel
    dark edges, larger door arcs, and more furniture noise — we use looser
    merge tolerances and a higher Hough minimum line length to filter
    furniture edges.

    Complex plans use a smaller Hough minimum length and tighter merge so
    we don't accidentally swallow short partition walls.
    """

    # ── Simple / default configuration ───────────────────────────────────────
    HOUGH_THRESHOLD  = 22          # lowered from 25 — catches more wall segments
    HOUGH_MIN_LINE   = 18          # lowered from 20 — catches short partition ends
    HOUGH_MAX_GAP    = 8           # raised from 6 — bridges small breaks
    MERGE_ANGLE_TOL  = 4.0         # tighter: avoid merging walls at slightly different angles
    MERGE_PERP_TOL   = 7.0         # perpendicular distance tolerance for same wall
    MERGE_GAP_TOL    = 12.0        # raised from 10 — joins nearby collinear fragments
    SNAP_DIST        = 12.0
    DOOR_GAP_MIN     = 14.0
    DOOR_GAP_MAX     = 100.0

    # ── Rendered-plan configuration ───────────────────────────────────────────
    R_HOUGH_THRESHOLD = 28
    R_HOUGH_MIN_LINE  = 30
    R_HOUGH_MAX_GAP   = 12
    R_MERGE_ANGLE_TOL = 6.0
    R_MERGE_PERP_TOL  = 22.0        # wider: handles thick walls as two parallel edges
    R_MERGE_GAP_TOL   = 18.0
    R_SNAP_DIST       = 20.0
    R_DOOR_GAP_MIN    = 20.0
    R_DOOR_GAP_MAX    = 160.0

    # ── Complex-plan configuration ────────────────────────────────────────────
    # Many short wall segments, thin internal partitions, dense geometry
    C_HOUGH_THRESHOLD = 18          # lower threshold to catch thin, short walls
    C_HOUGH_MIN_LINE  = 14          # shorter min — catch short partitions
    C_HOUGH_MAX_GAP   = 6
    C_MERGE_ANGLE_TOL = 3.0         # tighter — don't merge near-perpendicular walls
    C_MERGE_PERP_TOL  = 5.0         # tighter — avoid merging parallel but distinct walls
    C_MERGE_GAP_TOL   = 8.0
    C_SNAP_DIST       = 10.0
    C_DOOR_GAP_MIN    = 12.0
    C_DOOR_GAP_MAX    = 90.0

    def run(
        self,
        cleaned_binary: np.ndarray,
        original_bgr: np.ndarray,
        blueprint_style: str = "simple",
    ) -> WallDetectionResult:
        # Detect complexity from the binary mask
        complexity = self._assess_complexity(cleaned_binary)
        if blueprint_style == "rendered":
            preset = "rendered"
        elif complexity == "complex":
            preset = "complex"
        else:
            preset = "simple"

        hough_thresh, hough_min_line, hough_max_gap, \
        merge_atol, merge_ptol, merge_gtol, snap_dist, \
        door_min, door_max = self._preset(preset)

        result = WallDetectionResult()
        debug: Dict[str, np.ndarray] = {}
        result.wall_mask = cleaned_binary
        h_img, w_img = cleaned_binary.shape[:2]

        # ── Skeletonize ───────────────────────────────────────────────────
        skeleton = self._skeletonize(cleaned_binary)
        result.skeleton_mask = skeleton
        skel_vis = original_bgr.copy()
        skel_vis[skeleton > 0] = (0, 255, 180)
        debug["wd_01_skeleton"] = skel_vis

        # ── Multi-scale Hough ─────────────────────────────────────────────
        # Run Hough at two gap settings and union the results so we capture
        # both continuous walls and slightly broken ones.
        raw_segs = self._hough_multi(
            skeleton,
            threshold=hough_thresh,
            min_line=hough_min_line,
            max_gap_lo=hough_max_gap,
            max_gap_hi=hough_max_gap * 3,
        )
        raw_vis = original_bgr.copy()
        for x1, y1, x2, y2 in raw_segs:
            cv2.line(raw_vis, (int(x1), int(y1)), (int(x2), int(y2)), (80, 80, 255), 1)
        debug["wd_02_hough"] = raw_vis

        # ── Direction-cluster merge ───────────────────────────────────────
        merged = self._direction_cluster_merge(
            raw_segs,
            angle_tol=merge_atol,
            perp_tol=merge_ptol,
            gap_tol=merge_gtol,
        )

        # ── Door gap detection (before snapping) ───────────────────────────
        door_gaps = self._detect_door_gaps(
            merged,
            angle_tol=merge_atol,
            perp_tol=merge_ptol,
            door_min=door_min,
            door_max=door_max,
        )
        result.door_gaps = door_gaps

        # ── Snap endpoints ─────────────────────────────────────────────────
        snapped = self._snap_endpoints(merged, snap_dist=snap_dist)

        # ── Build Wall objects ─────────────────────────────────────────────
        walls: List[Wall] = []
        for x1, y1, x2, y2 in snapped:
            if math.hypot(x2 - x1, y2 - y1) < 8:
                continue
            thickness = self._estimate_thickness(
                cleaned_binary, x1, y1, x2, y2, h_img, w_img
            )
            walls.append(Wall(
                start=(float(x1), float(y1)),
                end=(float(x2), float(y2)),
                wall_type="internal",   # Reclassified later by topology
                thickness_px=thickness,
            ))
        result.walls = walls

        # Debug overlay
        wall_vis = original_bgr.copy()
        canvas = cv2.addWeighted(wall_vis, 0.35, np.zeros_like(wall_vis), 0.65, 0)
        for w in walls:
            p1 = (int(w.start[0]), int(w.start[1]))
            p2 = (int(w.end[0]), int(w.end[1]))
            cv2.line(canvas, p1, p2, (0, 130, 255), 2, cv2.LINE_AA)
            cv2.circle(canvas, p1, 4, (0, 255, 180), -1)
            cv2.circle(canvas, p2, 4, (0, 255, 180), -1)
        debug["wd_03_walls"] = canvas

        result.debug = debug
        return result

    # ── Private ───────────────────────────────────────────────────────────────

    def _preset(self, name: str):
        if name == "rendered":
            return (self.R_HOUGH_THRESHOLD, self.R_HOUGH_MIN_LINE,
                    self.R_HOUGH_MAX_GAP, self.R_MERGE_ANGLE_TOL,
                    self.R_MERGE_PERP_TOL, self.R_MERGE_GAP_TOL,
                    self.R_SNAP_DIST, self.R_DOOR_GAP_MIN, self.R_DOOR_GAP_MAX)
        elif name == "complex":
            return (self.C_HOUGH_THRESHOLD, self.C_HOUGH_MIN_LINE,
                    self.C_HOUGH_MAX_GAP, self.C_MERGE_ANGLE_TOL,
                    self.C_MERGE_PERP_TOL, self.C_MERGE_GAP_TOL,
                    self.C_SNAP_DIST, self.C_DOOR_GAP_MIN, self.C_DOOR_GAP_MAX)
        else:
            return (self.HOUGH_THRESHOLD, self.HOUGH_MIN_LINE,
                    self.HOUGH_MAX_GAP, self.MERGE_ANGLE_TOL,
                    self.MERGE_PERP_TOL, self.MERGE_GAP_TOL,
                    self.SNAP_DIST, self.DOOR_GAP_MIN, self.DOOR_GAP_MAX)

    def _assess_complexity(self, binary: np.ndarray) -> str:
        """
        Classify as 'complex' if the binary mask has high edge density.
        We use the ratio of wall pixels to image area.
        """
        h, w = binary.shape[:2]
        wall_px = int(np.count_nonzero(binary))
        density = wall_px / (h * w)
        # High density → complex (many thin walls close together)
        return "complex" if density > 0.08 else "simple"

    def _estimate_thickness(
        self,
        binary: np.ndarray,
        x1: float, y1: float,
        x2: float, y2: float,
        h: int, w: int,
        n_samples: int = 7,
    ) -> float:
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        if length < 1e-6:
            return 5.0
        nx, ny = -dy / length, dx / length
        thicknesses = []
        for i in range(n_samples):
            t = 0.2 + (i / max(n_samples - 1, 1)) * 0.6
            cx = x1 + t * dx
            cy = y1 + t * dy
            hit_neg = hit_pos = 0
            for sign in (+1, -1):
                for d in range(1, 61):
                    px = int(round(cx + sign * nx * d))
                    py = int(round(cy + sign * ny * d))
                    if px < 0 or px >= w or py < 0 or py >= h:
                        break
                    if binary[py, px] == 0:
                        break
                    if sign == 1:
                        hit_pos = d
                    else:
                        hit_neg = d
            measured = hit_pos + hit_neg
            if measured > 0:
                thicknesses.append(float(measured))
        if not thicknesses:
            return 5.0
        thickness = float(np.median(thicknesses))
        return float(np.clip(thickness, 2.0, 40.0))

    def _skeletonize(self, binary: np.ndarray) -> np.ndarray:
        try:
            return cv2.ximgproc.thinning(binary, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
        except AttributeError:
            pass
        skel = np.zeros_like(binary)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        img = binary.copy()
        while True:
            eroded = cv2.erode(img, element)
            temp = cv2.dilate(eroded, element)
            temp = cv2.subtract(img, temp)
            skel = cv2.bitwise_or(skel, temp)
            img = eroded.copy()
            if cv2.countNonZero(img) == 0:
                break
        return skel

    def _hough_multi(
        self,
        skeleton: np.ndarray,
        threshold: int,
        min_line: int,
        max_gap_lo: int,
        max_gap_hi: int,
    ) -> List[Tuple[float, float, float, float]]:
        """Run Hough at two maxLineGap settings and union deduplicated results."""
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        dilated = cv2.dilate(skeleton, k)

        def _run(gap):
            raw = cv2.HoughLinesP(
                dilated, rho=1, theta=np.pi / 180,
                threshold=threshold, minLineLength=min_line, maxLineGap=gap,
            )
            if raw is None:
                return []
            return [(float(l[0]), float(l[1]), float(l[2]), float(l[3]))
                    for l in raw.reshape(-1, 4)]

        segs_lo = _run(max_gap_lo)
        segs_hi = _run(max_gap_hi)

        # Union: deduplicate segs_hi against segs_lo by checking centre proximity
        combined = list(segs_lo)
        for sx1, sy1, sx2, sy2 in segs_hi:
            cx, cy = (sx1 + sx2) / 2, (sy1 + sy2) / 2
            dup = False
            for ex1, ey1, ex2, ey2 in segs_lo:
                ecx, ecy = (ex1 + ex2) / 2, (ey1 + ey2) / 2
                if math.hypot(cx - ecx, cy - ecy) < min_line * 0.6:
                    dup = True
                    break
            if not dup:
                combined.append((sx1, sy1, sx2, sy2))

        return combined

    def _seg_angle(self, x1, y1, x2, y2) -> float:
        return math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180

    def _perp_dist(self, px, py, x1, y1, x2, y2) -> float:
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        if length < 1e-9:
            return math.hypot(px - x1, py - y1)
        return abs(dx * (y1 - py) - dy * (x1 - px)) / length

    def _project(self, px, py, x1, y1, x2, y2) -> float:
        dx, dy = x2 - x1, y2 - y1
        l2 = dx * dx + dy * dy
        if l2 < 1e-9:
            return 0.0
        return ((px - x1) * dx + (py - y1) * dy) / l2

    def _direction_cluster_merge(
        self,
        segs: List[Tuple[float, float, float, float]],
        angle_tol: float = 4.0,
        perp_tol: float = 7.0,
        gap_tol: float = 12.0,
    ) -> List[Tuple[float, float, float, float]]:
        """
        Two-pass merge using direction clustering.

        Pass 1 — direction buckets:
          Sort segments by angle, group segments whose angles differ by <angle_tol
          AND whose perpendicular offset to the group axis differs by <perp_tol.
          This avoids the greedy-skip bug in the old single-pass approach.

        Pass 2 — within each direction+offset group:
          Project all segments onto the group axis and merge overlapping
          intervals (with gap_tol bridge). This produces the final segments.
        """
        if not segs:
            return []

        # Normalise direction: keep start.x ≤ end.x (or start.y ≤ end.y for verticals)
        normalised = []
        for x1, y1, x2, y2 in segs:
            a = self._seg_angle(x1, y1, x2, y2)
            if x1 > x2 or (x1 == x2 and y1 > y2):
                x1, y1, x2, y2 = x2, y2, x1, y1
            normalised.append((x1, y1, x2, y2, a))

        # Sort by angle for stable clustering
        normalised.sort(key=lambda s: s[4])

        used = [False] * len(normalised)
        result: List[Tuple[float, float, float, float]] = []

        for i, (x1, y1, x2, y2, ai) in enumerate(normalised):
            if used[i]:
                continue

            # Gather all segments with similar angle AND similar perpendicular offset
            group = [(x1, y1, x2, y2)]
            used[i] = True
            ax, ay = (x1 + x2) / 2, (y1 + y2) / 2   # midpoint of seed

            for j in range(i + 1, len(normalised)):
                if used[j]:
                    continue
                px1, py1, px2, py2, aj = normalised[j]
                da = abs(ai - aj)
                da = min(da, 180.0 - da)
                if da > angle_tol:
                    # Since sorted by angle, once da > angle_tol we can break
                    # (only valid if we don't have a wrap at 0/180 boundary, but
                    #  with the min above it's fine to keep scanning a bit further)
                    if da > angle_tol * 3:
                        break
                    continue
                # Perpendicular distance of j's midpoint to i's line
                pmx, pmy = (px1 + px2) / 2, (py1 + py2) / 2
                pdist = self._perp_dist(pmx, pmy, x1, y1, x2, y2)
                if pdist <= perp_tol:
                    group.append((px1, py1, px2, py2))
                    used[j] = True

            # Use the longest segment as the projection axis
            longest = max(group, key=lambda s: math.hypot(s[2] - s[0], s[3] - s[1]))
            lx1, ly1, lx2, ly2 = longest
            dx, dy = lx2 - lx1, ly2 - ly1
            length = math.hypot(dx, dy)
            if length < 1e-9:
                continue

            # Project all group segments onto the axis and collect intervals
            intervals = []
            for (sx1, sy1, sx2, sy2) in group:
                t1 = self._project(sx1, sy1, lx1, ly1, lx2, ly2)
                t2 = self._project(sx2, sy2, lx1, ly1, lx2, ly2)
                if t1 > t2:
                    t1, t2 = t2, t1
                intervals.append((t1, t2))
            intervals.sort()

            # Merge overlapping / close intervals using gap_tol
            gap_t = gap_tol / length
            curr_t1, curr_t2 = intervals[0]
            for t1, t2 in intervals[1:]:
                if t1 <= curr_t2 + gap_t:
                    curr_t2 = max(curr_t2, t2)
                else:
                    result.append((
                        lx1 + curr_t1 * dx, ly1 + curr_t1 * dy,
                        lx1 + curr_t2 * dx, ly1 + curr_t2 * dy,
                    ))
                    curr_t1, curr_t2 = t1, t2
            result.append((
                lx1 + curr_t1 * dx, ly1 + curr_t1 * dy,
                lx1 + curr_t2 * dx, ly1 + curr_t2 * dy,
            ))

        return result

    def _detect_door_gaps(
        self,
        segs: List[Tuple[float, float, float, float]],
        angle_tol: float = 4.0,
        perp_tol: float = 7.0,
        door_min: float = 14.0,
        door_max: float = 100.0,
    ) -> List[Door]:
        if not segs:
            return []

        normalised = []
        for x1, y1, x2, y2 in segs:
            a = self._seg_angle(x1, y1, x2, y2)
            if x1 > x2 or (x1 == x2 and y1 > y2):
                x1, y1, x2, y2 = x2, y2, x1, y1
            normalised.append((x1, y1, x2, y2, a))

        gaps: List[Door] = []

        for i, (ax1, ay1, ax2, ay2, ai) in enumerate(normalised):
            group = [(ax1, ay1, ax2, ay2)]
            for j, (px1, py1, px2, py2, aj) in enumerate(normalised):
                if i == j:
                    continue
                da = min(abs(ai - aj), 180 - abs(ai - aj))
                if da > angle_tol:
                    continue
                mx, my = (px1 + px2) / 2, (py1 + py2) / 2
                if self._perp_dist(mx, my, ax1, ay1, ax2, ay2) <= perp_tol:
                    group.append((px1, py1, px2, py2))

            if len(group) < 2:
                continue

            dx, dy = ax2 - ax1, ay2 - ay1
            length = math.hypot(dx, dy)
            if length < 1e-9:
                continue

            intervals = []
            for (sx1, sy1, sx2, sy2) in group:
                t1 = self._project(sx1, sy1, ax1, ay1, ax2, ay2)
                t2 = self._project(sx2, sy2, ax1, ay1, ax2, ay2)
                if t1 > t2:
                    t1, t2 = t2, t1
                intervals.append((t1, t2))
            intervals.sort()

            curr_t2 = intervals[0][1]
            for t1, t2 in intervals[1:]:
                gap_len = (t1 - curr_t2) * length
                if door_min <= gap_len <= door_max:
                    gx1 = ax1 + curr_t2 * dx
                    gy1 = ay1 + curr_t2 * dy
                    gx2 = ax1 + t1 * dx
                    gy2 = ay1 + t1 * dy
                    cx, cy = (gx1 + gx2) / 2, (gy1 + gy2) / 2
                    gaps.append(Door(
                        center=(cx, cy),
                        width_px=gap_len,
                        height_px=gap_len,
                        angle_deg=ai,
                        source="gap",
                        confidence=0.85,
                    ))
                curr_t2 = max(curr_t2, t2)

        # Deduplicate
        deduped: List[Door] = []
        for g in gaps:
            dup = any(math.hypot(g.center[0] - d.center[0], g.center[1] - d.center[1]) < 15 for d in deduped)
            if not dup:
                deduped.append(g)
        return deduped

    def _snap_endpoints(
        self,
        segs: List[Tuple[float, float, float, float]],
        snap_dist: float = 12.0,
    ) -> List[Tuple[float, float, float, float]]:
        """
        Snap near-coincident endpoints to their centroid.
        Uses a union-find approach for correct transitive snapping.
        """
        pts = []
        for seg in segs:
            pts.append((seg[0], seg[1]))
            pts.append((seg[2], seg[3]))

        n = len(pts)
        if n == 0:
            return segs

        # Union-Find
        parent = list(range(n))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(i, j):
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[ri] = rj

        for i in range(n):
            for j in range(i + 1, n):
                if math.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1]) < snap_dist:
                    union(i, j)

        # Compute centroid per cluster
        clusters: Dict[int, List] = {}
        for i in range(n):
            r = find(i)
            clusters.setdefault(r, []).append(pts[i])

        centroids = {r: (
            sum(p[0] for p in members) / len(members),
            sum(p[1] for p in members) / len(members),
        ) for r, members in clusters.items()}

        snapped = [centroids[find(i)] for i in range(n)]

        result = []
        for k in range(len(segs)):
            x1, y1 = snapped[k * 2]
            x2, y2 = snapped[k * 2 + 1]
            result.append((x1, y1, x2, y2))
        return result
