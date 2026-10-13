"""
backend/engine/model3d/wall_cleaner.py
Blueprint-X — Wall segment post-processor.

Takes noisy, fragmented Hough wall detections and produces clean,
continuous, properly-joined wall segments suitable for 3-D extrusion.

Pipeline
────────
1. Filter  — drop segments shorter than min_length_px
2. Merge   — cluster collinear segments; merge overlapping/near intervals
3. Snap    — connect wall endpoints within snap_dist of each other
"""
from __future__ import annotations

import math
from typing import List, Tuple

from backend.models.dataclasses import Wall


# ── Geometry helpers ──────────────────────────────────────────────────────────

def _length(w: Wall) -> float:
    return math.hypot(w.end[0]-w.start[0], w.end[1]-w.start[1])


def _length_coords(x1, y1, x2, y2) -> float:
    return math.hypot(x2-x1, y2-y1)


def _make_wall(x1, y1, x2, y2, wtype, thick) -> Wall:
    return Wall(start=(float(x1), float(y1)),
                end  =(float(x2), float(y2)),
                wall_type   = wtype,
                thickness_px= thick)


# ── WallCleaner ───────────────────────────────────────────────────────────────

class WallCleaner:
    """
    Parameters
    ----------
    min_length_px  : float  Drop segments shorter than this.
    angle_tol_deg  : float  Max angle difference for "same direction".
    offset_tol_px  : float  Max perpendicular offset for "same line".
    gap_tol_px     : float  Merge segments with gap smaller than this.
    snap_dist_px   : float  Snap endpoints within this distance.
    """

    def __init__(
        self,
        min_length_px : float = 25.0,
        angle_tol_deg : float = 8.0,
        offset_tol_px : float = 14.0,
        gap_tol_px    : float = 35.0,
        snap_dist_px  : float = 22.0,
    ):
        self.min_length_px = min_length_px
        self.angle_tol_deg = angle_tol_deg
        self.offset_tol_px = offset_tol_px
        self.gap_tol_px    = gap_tol_px
        self.snap_dist_px  = snap_dist_px

    def clean(self, walls: List[Wall]) -> List[Wall]:
        walls = [w for w in walls if _length(w) >= self.min_length_px]
        if not walls:
            return []
        walls = self._merge_collinear(walls)
        walls = self._snap_endpoints(walls)
        walls = self._extend_t_junctions(walls)
        walls = self._deduplicate(walls)
        return walls

    def _extend_t_junctions(self, walls: List[Wall]) -> List[Wall]:
        """Project endpoints near another wall segment directly onto that segment to close T-junction gaps."""
        result = []
        for i, w in enumerate(walls):
            s_x, s_y = w.start
            e_x, e_y = w.end
            for j, o in enumerate(walls):
                if i == j:
                    continue
                ax, ay = o.start
                bx, by = o.end
                dx, dy = bx - ax, by - ay
                L2 = dx*dx + dy*dy
                if L2 < 1e-4:
                    continue

                # Test start point projection onto segment o
                t_s = max(0.0, min(1.0, ((s_x - ax)*dx + (s_y - ay)*dy) / L2))
                px_s, py_s = ax + t_s*dx, ay + t_s*dy
                dist_s = math.hypot(s_x - px_s, s_y - py_s)
                if 1.0 < dist_s <= self.snap_dist_px:
                    s_x, s_y = px_s, py_s

                # Test end point projection onto segment o
                t_e = max(0.0, min(1.0, ((e_x - ax)*dx + (e_y - ay)*dy) / L2))
                px_e, py_e = ax + t_e*dx, ay + t_e*dy
                dist_e = math.hypot(e_x - px_e, e_y - py_e)
                if 1.0 < dist_e <= self.snap_dist_px:
                    e_x, e_y = px_e, py_e

            result.append(_make_wall(s_x, s_y, e_x, e_y, w.wall_type, w.thickness_px))
        return result

    # ── Step 2 — merge collinear ──────────────────────────────────────────────

    def _merge_collinear(self, walls: List[Wall]) -> List[Wall]:
        # Precompute per-wall properties
        props = []
        for w in walls:
            x1, y1 = w.start
            x2, y2 = w.end
            dx, dy  = x2-x1, y2-y1
            L       = math.hypot(dx, dy)
            angle   = math.degrees(math.atan2(dy, dx)) % 180   # [0, 180)
            nx, ny  = -dy/L, dx/L                               # unit normal
            offset  = x1*nx + y1*ny                             # signed dist from origin
            props.append(dict(
                w=w, angle=angle, nx=nx, ny=ny,
                dx=dx/L, dy=dy/L,
                x1=x1, y1=y1, x2=x2, y2=y2, L=L, offset=offset,
            ))

        used      = [False] * len(props)
        clusters  = []

        for i, pi in enumerate(props):
            if used[i]:
                continue
            cluster = [i]
            used[i] = True
            for j, pj in enumerate(props):
                if j <= i or used[j]:
                    continue
                # Angle similarity — handle both the 0°/180° wrap-around AND the
                # case where one angle is 0° and the other is ~180°.
                da = abs(pi['angle'] - pj['angle'])
                da = min(da, 180.0 - da, abs(da - 180.0))  # three-way min for robustness
                if da > self.angle_tol_deg:
                    continue
                # Perpendicular offset similarity (use i's normal)
                off_j = pj['x1']*pi['nx'] + pj['y1']*pi['ny']
                if abs(off_j - pi['offset']) > self.offset_tol_px:
                    continue
                cluster.append(j)
                used[j] = True
            clusters.append(cluster)

        result = []
        for cluster in clusters:
            result.extend(self._merge_cluster(cluster, props))
        return result

    def _merge_cluster(self, indices: list, props: list) -> List[Wall]:
        if len(indices) == 1:
            return [props[indices[0]]['w']]

        ref      = props[indices[0]]
        ax, ay   = ref['dx'], ref['dy']   # unit wall direction
        ox, oy   = ref['x1'], ref['y1']   # reference origin on the line

        intervals = []
        for idx in indices:
            p  = props[idx]
            w  = p['w']
            t1 = (p['x1']-ox)*ax + (p['y1']-oy)*ay
            t2 = (p['x2']-ox)*ax + (p['y2']-oy)*ay
            intervals.append((min(t1, t2), max(t1, t2), w))

        intervals.sort(key=lambda t: t[0])

        # Merge overlapping / near-adjacent intervals
        merged = [list(intervals[0])]
        for t1, t2, w in intervals[1:]:
            last = merged[-1]
            if t1 <= last[1] + self.gap_tol_px:
                # Extend the merged interval
                merged[-1][1] = max(last[1], t2)
                # Prefer outer wall type; take thicker wall thickness
                if w.wall_type == "outer":
                    merged[-1][2] = w
            else:
                merged.append([t1, t2, w])

        result = []
        for t1, t2, w in merged:
            nx1 = ox + t1*ax;  ny1 = oy + t1*ay
            nx2 = ox + t2*ax;  ny2 = oy + t2*ay
            seg_len = _length_coords(nx1, ny1, nx2, ny2)
            if seg_len >= self.min_length_px:
                result.append(_make_wall(nx1, ny1, nx2, ny2,
                                         w.wall_type, w.thickness_px))
        return result

    # ── Step 3 — snap endpoints ───────────────────────────────────────────────

    def _snap_endpoints(self, walls: List[Wall]) -> List[Wall]:
        if not walls:
            return walls

        # Collect all endpoints
        pts = []   # [x, y, wall_idx, 0=start/1=end]
        for i, w in enumerate(walls):
            pts.append([w.start[0], w.start[1], i, 0])
            pts.append([w.end[0],   w.end[1],   i, 1])

        snapped_x = [p[0] for p in pts]
        snapped_y = [p[1] for p in pts]

        sd_sq = self.snap_dist_px ** 2

        # For each endpoint find the nearest OTHER endpoint within snap_dist
        # and snap both to their midpoint (one-pass, greedy)
        done = [False] * len(pts)
        for i in range(len(pts)):
            if done[i]:
                continue
            wi = pts[i][2]
            best = -1
            best_d = sd_sq
            for j in range(i+1, len(pts)):
                if done[j] or pts[j][2] == wi:
                    continue
                d = (pts[i][0]-pts[j][0])**2 + (pts[i][1]-pts[j][1])**2
                if d < best_d:
                    best_d = d
                    best = j
            if best >= 0:
                mx = (snapped_x[i] + snapped_x[best]) / 2
                my = (snapped_y[i] + snapped_y[best]) / 2
                snapped_x[i]    = snapped_x[best] = mx
                snapped_y[i]    = snapped_y[best] = my
                done[i] = done[best] = True

        # Reconstruct walls from snapped endpoints
        # Gather modifications per wall
        mods: dict[int, dict] = {}
        for k, (ox, oy, wi, end) in enumerate(pts):
            if wi not in mods:
                mods[wi] = {}
            mods[wi][end] = (snapped_x[k], snapped_y[k])

        result = []
        for i, w in enumerate(walls):
            s = mods.get(i, {}).get(0, w.start)
            e = mods.get(i, {}).get(1, w.end)
            if _length_coords(*s, *e) >= self.min_length_px:
                result.append(_make_wall(*s, *e, w.wall_type, w.thickness_px))
        return result

    # ── Step 4 — deduplicate ──────────────────────────────────────────────────

    def _deduplicate(self, walls: List[Wall]) -> List[Wall]:
        """Remove walls whose endpoints are within snap_dist of an already-kept wall."""
        kept: List[Wall] = []
        tol = self.snap_dist_px

        for w in walls:
            sx, sy = w.start
            ex, ey = w.end
            duplicate = False
            for k in kept:
                ksx, ksy = k.start
                kex, key = k.end
                # Check both orientations
                d_fwd = (math.hypot(sx - ksx, sy - ksy) + math.hypot(ex - kex, ey - key))
                d_rev = (math.hypot(sx - kex, sy - key) + math.hypot(ex - ksx, ey - ksy))
                if min(d_fwd, d_rev) < tol:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(w)
        return kept
