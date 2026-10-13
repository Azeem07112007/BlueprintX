"""
backend/engine/model3d/builder.py
Blueprint-X — M4 (Fixed + Enhanced)

Wall source strategy (v4):
  1. Take raw Hough walls from building.walls
  2. Run WallCleaner: filter short → merge collinear → snap endpoints
  3. Extrude each cleaned segment as a solid box with:
       • Door gap cut-outs (horizontal split at ground level)
       • Window cut-outs (mid-height opening, ~60-90 cm sill + 120 cm head)
  4. Room polygons used for floor/ceiling slabs
  5. Thin ground base slab so building sits on a visible surface

Coordinate system: +X right, +Y up, +Z depth (Y-flipped from blueprint)
"""
from __future__ import annotations

import logging
import math
from typing import List, Optional, Tuple

import numpy as np

from backend.models.dataclasses import Building, Building3D, Room, Wall, Window
from backend.engine.model3d.wall_cleaner import WallCleaner
from backend.engine.model3d.furniture_builder import Furniture3DBuilder
from backend.engine.model3d.geometry_validator import GeometryValidator

logger = logging.getLogger(__name__)

# ── Colours (linear RGB 0-1) ──────────────────────────────────────────────────
_OUTER_WALL   = np.array([0.88, 0.78, 0.32], np.float32)   # warm gold
_INNER_WALL   = np.array([0.52, 0.64, 0.82], np.float32)   # soft steel blue
_TOP_OUTER    = np.array([0.95, 0.90, 0.60], np.float32)   # lighter gold cap
_TOP_INNER    = np.array([0.70, 0.80, 0.96], np.float32)   # lighter blue cap
_CEILING_COL  = np.array([0.96, 0.96, 0.97], np.float32)
_BASE_COL     = np.array([0.14, 0.14, 0.16], np.float32)   # dark ground slab
_UP           = np.array([0.0, 1.0, 0.0], np.float32)
_DOWN         = np.array([0.0, -1.0, 0.0], np.float32)


def _norm(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else _UP.copy()


def _quad(v0, v1, v2, v3, n, c, base):
    """Two CCW triangles for quad (v0 v1 v2 v3)."""
    i = base
    return (
        [v0, v1, v2, v3], [n, n, n, n], [c, c, c, c],
        [np.array([i, i+1, i+2], np.int32), np.array([i, i+2, i+3], np.int32)],
        base + 4,
    )


def _is_clockwise(pts: List[Tuple[float, float]]) -> bool:
    area = 0.0
    for i in range(len(pts)):
        j = (i + 1) % len(pts)
        area += pts[i][0] * pts[j][1] - pts[j][0] * pts[i][1]
    return area < 0


def _triangulate_polygon_2d(pts: List[Tuple[float, float]]) -> List[Tuple[int, int, int]]:
    """Ear clipping algorithm for simple 2D polygons (convex & concave)."""
    n = len(pts)
    if n < 3:
        return []
    if n == 3:
        return [(0, 1, 2)]

    cw = _is_clockwise(pts)
    indices = list(range(n))
    if cw:
        indices.reverse()

    def is_ear(i_prev, i_curr, i_next, remaining):
        ax, ay = pts[i_prev]
        bx, by = pts[i_curr]
        cx, cy = pts[i_next]
        cross = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
        if cross <= 0:
            return False
        for k in remaining:
            if k in (i_prev, i_curr, i_next):
                continue
            px, py = pts[k]
            v0x, v0y = cx - ax, cy - ay
            v1x, v1y = bx - ax, by - ay
            v2x, v2y = px - ax, py - ay
            dot00 = v0x * v0x + v0y * v0y
            dot01 = v0x * v1x + v0y * v1y
            dot02 = v0x * v2x + v0y * v2y
            dot11 = v1x * v1x + v1y * v1y
            dot12 = v1x * v2x + v1y * v2y
            denom = dot00 * dot11 - dot01 * dot01
            if abs(denom) < 1e-6:
                continue
            u = (dot11 * dot02 - dot01 * dot12) / denom
            v = (dot00 * dot12 - dot01 * dot02) / denom
            if u >= 0 and v >= 0 and (u + v) < 1.0:
                return False
        return True

    triangles = []
    count = 0
    max_loops = n * n
    while len(indices) > 3 and count < max_loops:
        count += 1
        ear_found = False
        m = len(indices)
        for i in range(m):
            i_prev = indices[(i - 1) % m]
            i_curr = indices[i]
            i_next = indices[(i + 1) % m]
            if is_ear(i_prev, i_curr, i_next, indices):
                triangles.append((i_prev, i_curr, i_next))
                indices.pop(i)
                ear_found = True
                break
        if not ear_found:
            break

    if len(indices) == 3:
        triangles.append((indices[0], indices[1], indices[2]))

    return triangles


class Model3DBuilder:
    """
    Parameters
    ----------
    floor_height_m  Wall height (m), default 2.8.
    px_per_m        Pixel-per-metre fallback, default 100.
    wall_min_outer  Minimum visual wall thickness outer (m), default 0.30.
    wall_min_inner  Minimum visual wall thickness inner (m), default 0.20.
    show_ceiling    Whether to add ceiling slabs (default False = open top).
    win_sill_frac   Window sill height as fraction of floor_height (default 0.25).
    win_head_frac   Window head height as fraction of floor_height (default 0.75).
    """

    def __init__(
        self,
        floor_height_m : float = 2.8,
        px_per_m       : float = 100.0,
        wall_min_outer : float = 0.30,
        wall_min_inner : float = 0.20,
        show_ceiling   : bool  = False,
        win_sill_frac  : float = 0.25,
        win_head_frac  : float = 0.75,
    ):
        self.floor_height_m = floor_height_m
        self._px_per_m      = px_per_m
        self.wall_min_outer = wall_min_outer
        self.wall_min_inner = wall_min_inner
        self.show_ceiling   = show_ceiling
        self.win_sill_frac  = win_sill_frac
        self.win_head_frac  = win_head_frac
        self._cleaner       = WallCleaner()

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self, building: Building) -> Building3D:
        px_per_m = self._resolve_scale(building)
        cx, cy   = self._pixel_centroid(building)

        def to_world(px, py, elev):
            return np.array([(px-cx)/px_per_m, elev, -(py-cy)/px_per_m], np.float32)

        verts, norms, cols, idxs = [], [], [], []
        base = 0

        # ── 0. 4-Pass Geometry Validation & Flaw Removal ──────────────────────
        building = GeometryValidator().validate_building_multi_pass(building, passes=4)

        # ── Clean walls (Hough → merge → snap) ───────────────────────────────
        clean_walls = self._cleaner.clean(building.walls)
        logger.info(f"[Builder] raw={len(building.walls)} → clean={len(clean_walls)} walls")

        door_segs = [(d.center[0], d.center[1], (d.width_px / 2.0) + 8.0)
                     for d in building.doors]

        # Window proximity lookup: (cx_px, cy_px, half_width_px)
        win_segs = [(w.center[0], w.center[1], max(w.width_px, w.height_px) / 2)
                    for w in building.windows]

        # ── 1. Wall boxes ─────────────────────────────────────────────────────
        for w in clean_walls:
            boxes, base = self._wall_box(w, door_segs, win_segs, to_world, px_per_m, base)
            for v, n, c, i in boxes:
                verts.extend(v); norms.extend(n); cols.extend(c); idxs.extend(i)

        # ── 2. Room floor slabs (at y=0.015 to clear ground slab) ──────────────
        for room in building.rooms:
            tris, base = self._slab(room, to_world, 0.015, base, _room_color(room), up=True)
            for v, n, c, i in tris:
                verts.extend(v); norms.extend(n); cols.extend(c); idxs.extend(i)

        # ── 3. Ground base slab (thin dark platform) ──────────────────────────
        base_tris, base = self._ground_slab(building, to_world, px_per_m, base)
        for v, n, c, i in base_tris:
            verts.extend(v); norms.extend(n); cols.extend(c); idxs.extend(i)

        # ── 4. Furniture 3D Mesh Generation ──────────────────────────────────
        for f in building.furniture:
            fv, fn, fc, fi, base = Furniture3DBuilder.build_furniture(f, to_world, px_per_m, base)
            verts.extend(fv); norms.extend(fn); cols.extend(fc); idxs.extend(fi)

        # ── 5. Staircase 3D Mesh Generation ─────────────────────────────────
        for s in building.stairs:
            sv, sn, sc, si, base = Furniture3DBuilder.build_staircase(s, to_world, self.floor_height_m, base)
            verts.extend(sv); norms.extend(sn); cols.extend(sc); idxs.extend(si)

        # ── 6. Balcony Railings ────────────────────────────────────────────────
        for room in building.rooms:
            r_lbl = (room.label or "").lower()
            if any(b in r_lbl for b in ["balcony", "terrace", "patio", "deck"]):
                rv, rn, rc, ri, base = Furniture3DBuilder.build_balcony_railing(room, to_world, 1.05, base)
                verts.extend(rv); norms.extend(rn); cols.extend(rc); idxs.extend(ri)

        # ── 4. Optional ceiling ───────────────────────────────────────────────
        if self.show_ceiling:
            for room in building.rooms:
                tris, base = self._slab(room, to_world, self.floor_height_m,
                                        base, _CEILING_COL, up=False)
                for v, n, c, i in tris:
                    verts.extend(v); norms.extend(n); cols.extend(c); idxs.extend(i)

        if not verts:
            return Building3D()

        V = np.stack(verts, 0).astype(np.float32)
        N = np.stack(norms, 0).astype(np.float32)
        C = np.stack(cols,  0).astype(np.float32)
        I = np.stack(idxs,  0).astype(np.int32)
        bmin, bmax = V.min(0), V.max(0)

        return Building3D(
            vertices=V, normals=N, colors=C, indices=I,
            floor_height_m = self.floor_height_m,
            px_per_m       = px_per_m,
            vertex_count   = len(V),
            triangle_count = len(I),
            center         = tuple(((bmin+bmax)/2).tolist()),
            bounds_min     = tuple(bmin.tolist()),
            bounds_max     = tuple(bmax.tolist()),
        )

    # ── Solid wall box ────────────────────────────────────────────────────────

    def _wall_box(self, wall: Wall, door_segs, win_segs, to_world, px_per_m, base):
        x1, y1 = wall.start
        x2, y2 = wall.end
        L = math.hypot(x2-x1, y2-y1)
        if L < 5.0:
            return [], base

        dx, dy = (x2-x1)/L, (y2-y1)/L
        h      = self.floor_height_m

        raw_t = wall.thickness_px / px_per_m if wall.thickness_px > 0 else 0.0
        if wall.wall_type == "outer":
            half_t = max(self.wall_min_outer, raw_t) / 2
            col, top_col = _OUTER_WALL, _TOP_OUTER
        else:
            half_t = max(self.wall_min_inner, raw_t) / 2
            col, top_col = _INNER_WALL, _TOP_INNER
        half_t_px = half_t * px_per_m

        # Perpendicular in pixel space (-dy, dx is "left" of wall direction)
        perp_x, perp_y = -dy, dx

        # World-space unit perpendicular: cross(wall_dir_world, up)
        # wall_dir_world = (dx, 0, -dy)
        # cross with (0,1,0) = (dy, 0, dx)
        wp = _norm(np.array([dy, 0., dx], np.float32))
        wd = _norm(np.array([dx, 0., -dy], np.float32))

        # Split wall at door gaps (horizontal, full-height openings)
        door_excl  = self._collect_excl(x1, y1, L, dx, dy, door_segs,  perp_tol=40.0)
        # Split wall at window openings (mid-height)
        win_excl   = self._collect_excl(x1, y1, L, dx, dy, win_segs,   perp_tol=35.0)

        door_segs_h = self._merge_excl(door_excl)   # full-height cuts
        win_segs_h  = self._merge_excl(win_excl)    # mid-height cuts (sill→head)

        # Generate wall column spans free of any full-height openings
        col_spans = self._subtract_excl([(0., 1.)], door_segs_h)

        results = []
        px_ = perp_x * half_t_px
        py_ = perp_y * half_t_px

        for ta, tb in col_spans:
            if tb - ta < 1e-4:
                continue
            ax, ay = x1 + ta*L*dx, y1 + ta*L*dy
            bx, by = x1 + tb*L*dx, y1 + tb*L*dy

            def c4(px, py, el): return to_world(px, py, el)

            ai  = c4(ax+px_, ay+py_, 0);   ai_t = c4(ax+px_, ay+py_, h)
            ao  = c4(ax-px_, ay-py_, 0);   ao_t = c4(ax-px_, ay-py_, h)
            bi  = c4(bx+px_, by+py_, 0);   bi_t = c4(bx+px_, by+py_, h)
            bo  = c4(bx-px_, by-py_, 0);   bo_t = c4(bx-px_, by-py_, h)

            # Determine which sub-spans of [ta,tb] have window cut-outs
            local_wins = [(max(ta, wa), min(tb, wb))
                          for wa, wb in win_segs_h
                          if wa < tb and wb > ta]
            local_wins = [(a, b) for a, b in local_wins if b > a + 1e-4]

            if local_wins:
                # Sub-spans without windows (full height)
                solid_spans = self._subtract_excl([(ta, tb)], local_wins)
                for sa, sb in solid_spans:
                    if sb - sa < 1e-4:
                        continue
                    sax, say = x1 + sa*L*dx, y1 + sa*L*dy
                    sbx, sby = x1 + sb*L*dx, y1 + sb*L*dy
                    ai2  = c4(sax+px_, say+py_, 0);  ai2_t = c4(sax+px_, say+py_, h)
                    ao2  = c4(sax-px_, say-py_, 0);  ao2_t = c4(sax-px_, say-py_, h)
                    bi2  = c4(sbx+px_, sby+py_, 0);  bi2_t = c4(sbx+px_, sby+py_, h)
                    bo2  = c4(sbx-px_, sby-py_, 0);  bo2_t = c4(sbx-px_, sby-py_, h)
                    for q in [
                        (ai2, bi2, bi2_t, ai2_t,  wp,   col),
                        (bo2, ao2, ao2_t, bo2_t, -wp,   col),
                        (ao2, ai2, ai2_t, ao2_t, -wd,   col),
                        (bi2, bo2, bo2_t, bi2_t,  wd,   col),
                        (ai2_t,bi2_t,bo2_t,ao2_t, _UP,  top_col),
                        (ao2,  bo2, bi2,  ai2,   _DOWN, col),
                    ]:
                        v, n, c_, i, base = _quad(*q[:4], q[4], q[5], base)
                        results.append((v, n, c_, i))

                # Sub-spans with windows: sill piece + head piece
                sill_h = h * self.win_sill_frac
                head_h = h * self.win_head_frac
                for wa, wb in local_wins:
                    if wb - wa < 1e-4:
                        continue
                    wax, way = x1 + wa*L*dx, y1 + wa*L*dy
                    wbx, wby = x1 + wb*L*dx, y1 + wb*L*dy

                    # Sill piece (floor → sill_h)
                    for (e0, e1) in [(0.0, sill_h), (head_h, h)]:
                        ai3  = c4(wax+px_, way+py_, e0);  ai3_t = c4(wax+px_, way+py_, e1)
                        ao3  = c4(wax-px_, way-py_, e0);  ao3_t = c4(wax-px_, way-py_, e1)
                        bi3  = c4(wbx+px_, wby+py_, e0);  bi3_t = c4(wbx+px_, wby+py_, e1)
                        bo3  = c4(wbx-px_, wby-py_, e0);  bo3_t = c4(wbx-px_, wby-py_, e1)
                        cap = _UP if e1 == h else None
                        for q in [
                            (ai3, bi3, bi3_t, ai3_t,  wp,   col),
                            (bo3, ao3, ao3_t, bo3_t, -wp,   col),
                            (ao3, ai3, ai3_t, ao3_t, -wd,   col),
                            (bi3, bo3, bo3_t, bi3_t,  wd,   col),
                        ]:
                            v, n, c_, i, base = _quad(*q[:4], q[4], q[5], base)
                            results.append((v, n, c_, i))
                        if cap is not None:
                            v, n, c_, i, base = _quad(ai3_t, bi3_t, bo3_t, ao3_t, cap, top_col, base)
                            results.append((v, n, c_, i))
                        # Bottom face (floor visible from below)
                        if e0 == 0.0:
                            v, n, c_, i, base = _quad(ao3, bo3, bi3, ai3, _DOWN, col, base)
                            results.append((v, n, c_, i))

                    # 3D Glass Window Pane (sill_h → head_h)
                    _COL_GLASS = np.array([0.40, 0.75, 0.90], np.float32)
                    ai_g  = c4(wax+px_*0.15, way+py_*0.15, sill_h);  ai_gt = c4(wax+px_*0.15, way+py_*0.15, head_h)
                    ao_g  = c4(wax-px_*0.15, way-py_*0.15, sill_h);  ao_gt = c4(wax-px_*0.15, way-py_*0.15, head_h)
                    bi_g  = c4(wbx+px_*0.15, wby+py_*0.15, sill_h);  bi_gt = c4(wbx+px_*0.15, wby+py_*0.15, head_h)
                    bo_g  = c4(wbx-px_*0.15, wby-py_*0.15, sill_h);  bo_gt = c4(wbx-px_*0.15, wby-py_*0.15, head_h)
                    for q in [
                        (ai_g, bi_g, bi_gt, ai_gt, wp,  _COL_GLASS),
                        (bo_g, ao_g, ao_gt, bo_gt, -wp, _COL_GLASS),
                        (ai_gt, bi_gt, bo_gt, ao_gt, _UP, _COL_GLASS),
                        (ao_g, bo_g, bi_g, ai_g, _DOWN, _COL_GLASS),
                    ]:
                        v, n, c_, i, base = _quad(*q[:4], q[4], q[5], base)
                        results.append((v, n, c_, i))
            else:
                # No window cuts — standard 5-face wall box
                for q in [
                    (ai,  bi,  bi_t, ai_t,  wp,   col),
                    (bo,  ao,  ao_t, bo_t, -wp,   col),
                    (ao,  ai,  ai_t, ao_t, -wd,   col),
                    (bi,  bo,  bo_t, bi_t,  wd,   col),
                    (ai_t,bi_t,bo_t, ao_t,  _UP,  top_col),
                    (ao,  bo,  bi,   ai,   _DOWN, col),
                ]:
                    v, n, c_, i, base = _quad(*q[:4], q[4], q[5], base)
                    results.append((v, n, c_, i))

        return results, base

    # ── Room slab (floor or ceiling) ──────────────────────────────────────────

    def _slab(self, room: Room, to_world, elevation, base, color, up: bool):
        pts = room.polygon_pts
        if len(pts) < 3:
            return [], base
        clean = [pts[0]]
        for p in pts[1:]:
            if math.hypot(p[0] - clean[-1][0], p[1] - clean[-1][1]) > 0.5:
                clean.append(p)
        if len(clean) < 3:
            return [], base

        n_vec = np.array([0., 1. if up else -1., 0.], np.float32)
        world_pts = [to_world(p[0], p[1], elevation) for p in clean]
        tri_indices = _triangulate_polygon_2d(clean)

        result = []
        for i0, i1, i2 in tri_indices:
            pA = world_pts[i0]
            pB = world_pts[i1]
            pC = world_pts[i2]
            tri = [pA, pB, pC] if up else [pA, pC, pB]
            result.append((tri, [n_vec] * 3, [color] * 3,
                           [np.array([base, base + 1, base + 2], np.int32)]))
            base += 3
        return result, base

    # ── Ground base slab ──────────────────────────────────────────────────────

    def _ground_slab(self, building: Building, to_world, px_per_m: float, base: int):
        """Thin dark slab from y = -0.06 m to y = 0 covering the whole footprint."""
        # Use wall bounding box to determine slab extent
        all_pts = []
        for w in building.walls:
            all_pts += [w.start, w.end]
        if not all_pts:
            return [], base

        xs = [p[0] for p in all_pts]
        ys = [p[1] for p in all_pts]
        pad = 10.0  # px padding
        x_min, x_max = min(xs) - pad, max(xs) + pad
        y_min, y_max = min(ys) - pad, max(ys) + pad

        slab_y  = -0.07   # bottom of slab (m)
        top_y   = -0.01   # top of slab — just below room floors (at 0.002) to prevent z-fight

        bl = to_world(x_min, y_min, slab_y); bl_t = to_world(x_min, y_min, top_y)
        br = to_world(x_max, y_min, slab_y); br_t = to_world(x_max, y_min, top_y)
        tr = to_world(x_max, y_max, slab_y); tr_t = to_world(x_max, y_max, top_y)
        tl = to_world(x_min, y_max, slab_y); tl_t = to_world(x_min, y_max, top_y)

        results = []
        # Top face (visible)
        v, n, c_, i, base = _quad(bl_t, br_t, tr_t, tl_t, _UP, _BASE_COL, base)
        results.append((v, n, c_, i))
        # Bottom face
        v, n, c_, i, base = _quad(tl, tr, br, bl, _DOWN, _BASE_COL, base)
        results.append((v, n, c_, i))
        # Four sides
        fwd = _norm(np.array([0., 0., -1.], np.float32))
        bck = _norm(np.array([0., 0.,  1.], np.float32))
        rgt = _norm(np.array([1., 0.,  0.], np.float32))
        lft = _norm(np.array([-1., 0., 0.], np.float32))
        for q in [
            (bl,  br,  br_t, bl_t,  fwd, _BASE_COL),
            (tr,  tl,  tl_t, tr_t,  bck, _BASE_COL),
            (br,  tr,  tr_t, br_t,  rgt, _BASE_COL),
            (tl,  bl,  bl_t, tl_t,  lft, _BASE_COL),
        ]:
            v, n, c_, i, base = _quad(*q[:4], q[4], q[5], base)
            results.append((v, n, c_, i))

        return results, base

    # ── Utilities ─────────────────────────────────────────────────────────────

    def _resolve_scale(self, building: Building) -> float:
        try:
            s = (building.scale or "").strip()
            if ":" in s:
                d = float(s.split(":")[1])
                if d > 0:
                    return d
        except (ValueError, IndexError):
            pass
        return self._px_per_m

    def _pixel_centroid(self, building: Building) -> Tuple[float, float]:
        if building.rooms:
            xs = [r.centroid[0] for r in building.rooms]
            ys = [r.centroid[1] for r in building.rooms]
            return float(np.mean(xs)), float(np.mean(ys))
        xs, ys = [], []
        for w in building.walls:
            xs += [w.start[0], w.end[0]]; ys += [w.start[1], w.end[1]]
        if not xs:
            return building.image_size[0]/2, building.image_size[1]/2
        return float(np.mean(xs)), float(np.mean(ys))

    def _collect_excl(self, x1, y1, L, dx, dy, segs, perp_tol: float):
        """Collect t-interval exclusions on a wall segment from nearby feature segments."""
        excl = []
        for dcx, dcy, hw in segs:
            t = ((dcx - x1) * dx + (dcy - y1) * dy) / L
            if not -0.1 < t < 1.1:
                continue
            if abs(-(dcy - y1) * dx + (dcx - x1) * dy) > perp_tol:
                continue
            excl.append((max(0., t - hw / L), min(1., t + hw / L)))
        return excl

    def _merge_excl(self, excl):
        """Merge overlapping exclusion intervals."""
        if not excl:
            return []
        excl = sorted(excl)
        merged = [list(excl[0])]
        for a, b in excl[1:]:
            if a <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        return [(a, b) for a, b in merged]

    def _subtract_excl(self, spans, excl):
        """Subtract exclusion intervals from a list of spans."""
        result = []
        for span_a, span_b in spans:
            cur = span_a
            for ea, eb in excl:
                if ea > cur:
                    result.append((cur, min(span_b, ea)))
                cur = max(cur, eb)
                if cur >= span_b:
                    break
            if cur < span_b:
                result.append((cur, span_b))
        return [(a, b) for a, b in result if b - a > 1e-4]


ROOM_MATERIAL_PALETTE = {
    "bedroom":    np.array([0.78, 0.58, 0.38], dtype=np.float32),  # Warm Timber / Oak Hardwood
    "master":     np.array([0.75, 0.52, 0.34], dtype=np.float32),  # Dark Walnut
    "living":     np.array([0.84, 0.68, 0.48], dtype=np.float32),  # Light Natural Oak
    "hall":       np.array([0.82, 0.65, 0.45], dtype=np.float32),  # Parquet Oak
    "kitchen":    np.array([0.82, 0.83, 0.85], dtype=np.float32),  # Light Marble Ceramic Tile
    "bathroom":   np.array([0.90, 0.92, 0.95], dtype=np.float32),  # Polished Porcelain White Tile
    "toilet":     np.array([0.88, 0.90, 0.92], dtype=np.float32),  # White Mosaic Tile
    "balcony":    np.array([0.65, 0.60, 0.55], dtype=np.float32),  # Outdoor Stone Pavers
    "garage":     np.array([0.55, 0.56, 0.58], dtype=np.float32),  # Sealed Polished Concrete
    "office":     np.array([0.72, 0.54, 0.36], dtype=np.float32),  # Teak Hardwood
    "corridor":   np.array([0.80, 0.78, 0.72], dtype=np.float32),  # Cream Stone
}


def _room_color(room: Room) -> np.ndarray:
    """Return realistic material color for floor slab based on room label."""
    lbl = (room.label or "").lower().strip()
    for key, color in ROOM_MATERIAL_PALETTE.items():
        if key in lbl:
            return color

    b, g, r = room.color_bgr
    rf = 0.20 + (r / 255) * 0.65
    gf = 0.20 + (g / 255) * 0.65
    bf = 0.20 + (b / 255) * 0.65
    return np.array([rf, gf, bf], np.float32)
