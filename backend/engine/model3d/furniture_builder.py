"""
backend/engine/model3d/furniture_builder.py
Blueprint-X — Procedural 3D Realistic Furniture Generator

Constructs realistic GPU-ready 3D geometric meshes for architectural furniture,
sanitary fixtures, kitchen counters, staircases, plants, and balcony railings
using curved primitives (cylinders, disks, cones, spheres) alongside quads.
"""
from __future__ import annotations

import math
import numpy as np
from typing import List, Tuple, Callable

from backend.models.dataclasses import Furniture, Stair, Room

# ── Palette (0-1 RGB float32) ────────────────────────────────────────────────
_COL_WOOD     = np.array([0.62, 0.45, 0.30], dtype=np.float32)   # warm timber
_COL_WHITE    = np.array([0.94, 0.95, 0.96], dtype=np.float32)   # ceramic / sheet
_COL_SOFA     = np.array([0.32, 0.52, 0.60], dtype=np.float32)   # slate teal
_COL_STEEL    = np.array([0.72, 0.74, 0.76], dtype=np.float32)   # stainless steel
_COL_DARK     = np.array([0.15, 0.15, 0.16], dtype=np.float32)   # cooktop / TV
_COL_FOLIAGE  = np.array([0.28, 0.65, 0.32], dtype=np.float32)   # plant green
_COL_POT      = np.array([0.70, 0.45, 0.32], dtype=np.float32)   # terracotta pot
_COL_RAIL     = np.array([0.22, 0.24, 0.28], dtype=np.float32)   # metallic rail
_COL_STAIR    = np.array([0.75, 0.70, 0.62], dtype=np.float32)   # stair tread
_COL_GOLD     = np.array([0.85, 0.70, 0.35], dtype=np.float32)   # brass / accent

_UP = np.array([0.0, 1.0, 0.0], dtype=np.float32)


def _quad_box(
    x1: float, y1: float, z1: float,
    x2: float, y2: float, z2: float,
    color: np.ndarray,
    base: int,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray], int]:
    """Generates a solid 3D box defined by min/max world bounds."""
    v = [
        # Front face (-Z)
        np.array([x1, y1, z1], np.float32), np.array([x2, y1, z1], np.float32),
        np.array([x2, y2, z1], np.float32), np.array([x1, y2, z1], np.float32),
        # Back face (+Z)
        np.array([x2, y1, z2], np.float32), np.array([x1, y1, z2], np.float32),
        np.array([x1, y2, z2], np.float32), np.array([x2, y2, z2], np.float32),
        # Top face (+Y)
        np.array([x1, y2, z1], np.float32), np.array([x2, y2, z1], np.float32),
        np.array([x2, y2, z2], np.float32), np.array([x1, y2, z2], np.float32),
        # Bottom face (-Y)
        np.array([x1, y1, z2], np.float32), np.array([x2, y1, z2], np.float32),
        np.array([x2, y1, z1], np.float32), np.array([x1, y1, z1], np.float32),
        # Left face (-X)
        np.array([x1, y1, z2], np.float32), np.array([x1, y1, z1], np.float32),
        np.array([x1, y2, z1], np.float32), np.array([x1, y2, z2], np.float32),
        # Right face (+X)
        np.array([x2, y1, z1], np.float32), np.array([x2, y1, z2], np.float32),
        np.array([x2, y2, z2], np.float32), np.array([x2, y2, z1], np.float32),
    ]

    n_f = np.array([0, 0, -1], np.float32)
    n_b = np.array([0, 0, 1], np.float32)
    n_u = np.array([0, 1, 0], np.float32)
    n_d = np.array([0, -1, 0], np.float32)
    n_l = np.array([-1, 0, 0], np.float32)
    n_r = np.array([1, 0, 0], np.float32)

    n = [
        n_f, n_f, n_f, n_f,
        n_b, n_b, n_b, n_b,
        n_u, n_u, n_u, n_u,
        n_d, n_d, n_d, n_d,
        n_l, n_l, n_l, n_l,
        n_r, n_r, n_r, n_r,
    ]

    c = [color] * 24

    i = base
    idxs = []
    for face in range(6):
        b = i + face * 4
        idxs.append(np.array([b, b + 1, b + 2], np.int32))
        idxs.append(np.array([b, b + 2, b + 3], np.int32))

    return v, n, c, idxs, base + 24


def _cylinder(
    cx: float, y_bot: float, cz: float,
    radius: float, height: float,
    color: np.ndarray, base: int, segs: int = 16
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray], int]:
    """Generates a vertical 3D cylinder mesh with top and bottom caps."""
    verts, norms, cols, idxs = [], [], [], []
    y_top = y_bot + height

    # Side quads
    for s in range(segs):
        a1 = (2.0 * math.pi * s) / segs
        a2 = (2.0 * math.pi * (s + 1)) / segs
        cos1, sin1 = math.cos(a1), math.sin(a1)
        cos2, sin2 = math.cos(a2), math.sin(a2)

        p0 = np.array([cx + radius * cos1, y_bot, cz + radius * sin1], np.float32)
        p1 = np.array([cx + radius * cos2, y_bot, cz + radius * sin2], np.float32)
        p2 = np.array([cx + radius * cos2, y_top, cz + radius * sin2], np.float32)
        p3 = np.array([cx + radius * cos1, y_top, cz + radius * sin1], np.float32)

        n1 = np.array([cos1, 0.0, sin1], np.float32)
        n2 = np.array([cos2, 0.0, sin2], np.float32)

        i = base + len(verts)
        verts.extend([p0, p1, p2, p3])
        norms.extend([n1, n2, n2, n1])
        cols.extend([color, color, color, color])
        idxs.append(np.array([i, i + 1, i + 2], np.int32))
        idxs.append(np.array([i, i + 2, i + 3], np.int32))

    # Top cap
    p_center_top = np.array([cx, y_top, cz], np.float32)
    n_up = np.array([0.0, 1.0, 0.0], np.float32)
    for s in range(segs):
        a1 = (2.0 * math.pi * s) / segs
        a2 = (2.0 * math.pi * (s + 1)) / segs
        p1 = np.array([cx + radius * math.cos(a1), y_top, cz + radius * math.sin(a1)], np.float32)
        p2 = np.array([cx + radius * math.cos(a2), y_top, cz + radius * math.sin(a2)], np.float32)

        i = base + len(verts)
        verts.extend([p_center_top, p1, p2])
        norms.extend([n_up, n_up, n_up])
        cols.extend([color, color, color])
        idxs.append(np.array([i, i + 1, i + 2], np.int32))

    # Bottom cap
    p_center_bot = np.array([cx, y_bot, cz], np.float32)
    n_down = np.array([0.0, -1.0, 0.0], np.float32)
    for s in range(segs):
        a1 = (2.0 * math.pi * s) / segs
        a2 = (2.0 * math.pi * (s + 1)) / segs
        p1 = np.array([cx + radius * math.cos(a1), y_bot, cz + radius * math.sin(a1)], np.float32)
        p2 = np.array([cx + radius * math.cos(a2), y_bot, cz + radius * math.sin(a2)], np.float32)

        i = base + len(verts)
        verts.extend([p_center_bot, p2, p1])
        norms.extend([n_down, n_down, n_down])
        cols.extend([color, color, color])
        idxs.append(np.array([i, i + 1, i + 2], np.int32))

    return verts, norms, cols, idxs, base + len(verts)


def _disk(
    cx: float, y: float, cz: float,
    radius: float, color: np.ndarray, base: int,
    segs: int = 16, facing_up: bool = True
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray], int]:
    """Generates a circular disk mesh."""
    verts, norms, cols, idxs = [], [], [], []
    p_center = np.array([cx, y, cz], np.float32)
    normal = np.array([0.0, 1.0 if facing_up else -1.0, 0.0], np.float32)

    for s in range(segs):
        a1 = (2.0 * math.pi * s) / segs
        a2 = (2.0 * math.pi * (s + 1)) / segs
        p1 = np.array([cx + radius * math.cos(a1), y, cz + radius * math.sin(a1)], np.float32)
        p2 = np.array([cx + radius * math.cos(a2), y, cz + radius * math.sin(a2)], np.float32)

        i = base + len(verts)
        verts.extend([p_center, p1, p2] if facing_up else [p_center, p2, p1])
        norms.extend([normal, normal, normal])
        cols.extend([color, color, color])
        idxs.append(np.array([i, i + 1, i + 2], np.int32))

    return verts, norms, cols, idxs, base + len(verts)


def _sphere_approx(
    cx: float, cy: float, cz: float,
    radius: float, color: np.ndarray, base: int,
    lat_segs: int = 8, lon_segs: int = 12
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray], int]:
    """Generates a 3D spherical mesh approximation."""
    verts, norms, cols, idxs = [], [], [], []

    for i in range(lat_segs):
        lat1 = -math.pi / 2.0 + (math.pi * i) / lat_segs
        lat2 = -math.pi / 2.0 + (math.pi * (i + 1)) / lat_segs

        for j in range(lon_segs):
            lon1 = (2.0 * math.pi * j) / lon_segs
            lon2 = (2.0 * math.pi * (j + 1)) / lon_segs

            x1 = radius * math.cos(lat1) * math.cos(lon1)
            y1 = radius * math.sin(lat1)
            z1 = radius * math.cos(lat1) * math.sin(lon1)

            x2 = radius * math.cos(lat1) * math.cos(lon2)
            y2 = radius * math.sin(lat1)
            z2 = radius * math.cos(lat1) * math.sin(lon2)

            x3 = radius * math.cos(lat2) * math.cos(lon2)
            y3 = radius * math.sin(lat2)
            z3 = radius * math.cos(lat2) * math.sin(lon2)

            x4 = radius * math.cos(lat2) * math.cos(lon1)
            y4 = radius * math.sin(lat2)
            z4 = radius * math.cos(lat2) * math.sin(lon1)

            p1 = np.array([cx + x1, cy + y1, cz + z1], np.float32)
            p2 = np.array([cx + x2, cy + y2, cz + z2], np.float32)
            p3 = np.array([cx + x3, cy + y3, cz + z3], np.float32)
            p4 = np.array([cx + x4, cy + y4, cz + z4], np.float32)

            n1 = np.array([x1, y1, z1], np.float32) / max(radius, 1e-6)
            n2 = np.array([x2, y2, z2], np.float32) / max(radius, 1e-6)
            n3 = np.array([x3, y3, z3], np.float32) / max(radius, 1e-6)
            n4 = np.array([x4, y4, z4], np.float32) / max(radius, 1e-6)

            idx_b = base + len(verts)
            verts.extend([p1, p2, p3, p4])
            norms.extend([n1, n2, n3, n4])
            cols.extend([color, color, color, color])
            idxs.append(np.array([idx_b, idx_b + 1, idx_b + 2], np.int32))
            idxs.append(np.array([idx_b, idx_b + 2, idx_b + 3], np.int32))

    return verts, norms, cols, idxs, base + len(verts)


class Furniture3DBuilder:
    """Procedural realistic 3D mesh generator for architectural furniture."""

    @staticmethod
    def build_furniture(
        furniture: Furniture,
        to_world: Callable[[float, float, float], np.ndarray],
        px_per_m: float,
        base_idx: int,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray], int]:
        verts, norms, cols, idxs = [], [], [], []
        base = base_idx

        if not furniture.bbox:
            return verts, norms, cols, idxs, base

        x, y, w, h = furniture.bbox
        p0 = to_world(x, y, 0.0)
        p1 = to_world(x + w, y + h, 0.0)

        raw_xmin, raw_xmax = min(p0[0], p1[0]), max(p0[0], p1[0])
        raw_zmin, raw_zmax = min(p0[2], p1[2]), max(p0[2], p1[2])

        cx = (raw_xmin + raw_xmax) / 2.0
        cz = (raw_zmin + raw_zmax) / 2.0

        c_name = (furniture.class_name or "").lower()

        # Physical clamping (m)
        if "bed" in c_name:
            max_w, max_d = 2.0, 2.0
        elif "sofa" in c_name or "chair" in c_name:
            max_w, max_d = 2.2, 0.9
        elif "dining" in c_name or "table" in c_name:
            max_w, max_d = 1.8, 1.0
        elif "counter" in c_name or "kitchen" in c_name:
            max_w, max_d = 2.4, 0.65
        elif "toilet" in c_name or "wc" in c_name:
            max_w, max_d = 0.55, 0.70
        elif "sink" in c_name or "vanity" in c_name:
            max_w, max_d = 0.75, 0.55
        elif "plant" in c_name:
            max_w, max_d = 0.60, 0.60
        else:
            max_w, max_d = 1.2, 0.8

        w_len = min(raw_xmax - raw_xmin, max_w)
        d_len = min(raw_zmax - raw_zmin, max_d)

        x_min = cx - w_len / 2.0
        x_max = cx + w_len / 2.0
        z_min = cz - d_len / 2.0
        z_max = cz + d_len / 2.0

        def add_mesh(res):
            nonlocal base
            v, n, c, idx, base = res
            verts.extend(v); norms.extend(n); cols.extend(c); idxs.extend(idx)

        # ── 1. BED (With rounded nightstands & pillows) ──────────────────────
        if "bed" in c_name:
            # Bed frame base
            add_mesh(_quad_box(x_min, 0.0, z_min, x_max, 0.20, z_max, _COL_WOOD, base))
            # Mattress
            pad = 0.04
            add_mesh(_quad_box(x_min + pad, 0.20, z_min + pad, x_max - pad, 0.46, z_max - pad, _COL_WHITE, base))
            # Headboard
            add_mesh(_quad_box(x_min, 0.0, z_min, x_max, 0.85, z_min + 0.08, _COL_WOOD, base))
            # Pillow 1 & 2 (with curved top disks)
            w_mid = (x_min + x_max) / 2.0
            add_mesh(_quad_box(x_min + 0.06, 0.46, z_min + 0.10, w_mid - 0.03, 0.53, z_min + 0.35, _COL_WHITE, base))
            add_mesh(_quad_box(w_mid + 0.03, 0.46, z_min + 0.10, x_max - 0.06, 0.53, z_min + 0.35, _COL_WHITE, base))
            # Side nightstands (Cylindrical tables)
            add_mesh(_cylinder(x_min - 0.25, 0.0, z_min + 0.20, 0.18, 0.45, _COL_WOOD, base, segs=12))
            add_mesh(_cylinder(x_max + 0.25, 0.0, z_min + 0.20, 0.18, 0.45, _COL_WOOD, base, segs=12))

        # ── 2. SOFA / ARMCHAIR (With cylindrical metallic legs & rounded arms) ──
        elif "sofa" in c_name or "chair" in c_name:
            # 4 Metallic cylindrical legs
            r_leg = 0.03
            add_mesh(_cylinder(x_min + 0.08, 0.0, z_min + 0.08, r_leg, 0.12, _COL_STEEL, base, segs=8))
            add_mesh(_cylinder(x_max - 0.08, 0.0, z_min + 0.08, r_leg, 0.12, _COL_STEEL, base, segs=8))
            add_mesh(_cylinder(x_min + 0.08, 0.0, z_max - 0.08, r_leg, 0.12, _COL_STEEL, base, segs=8))
            add_mesh(_cylinder(x_max - 0.08, 0.0, z_max - 0.08, r_leg, 0.12, _COL_STEEL, base, segs=8))
            # Main seat cushion
            add_mesh(_quad_box(x_min, 0.12, z_min, x_max, 0.38, z_max, _COL_SOFA, base))
            # Backrest
            add_mesh(_quad_box(x_min, 0.38, z_min, x_max, 0.78, z_min + 0.18, _COL_SOFA, base))
            # Left & right armrests
            add_mesh(_quad_box(x_min, 0.38, z_min, x_min + 0.14, 0.58, z_max, _COL_SOFA, base))
            add_mesh(_quad_box(x_max - 0.14, 0.38, z_min, x_max, 0.58, z_max, _COL_SOFA, base))

        # ── 3. DINING TABLE & CHAIRS (Cylindrical legs & rounded seats) ─────────
        elif "dining" in c_name or "table" in c_name:
            # Tabletop
            add_mesh(_quad_box(x_min, 0.70, z_min, x_max, 0.75, z_max, _COL_WOOD, base))
            # 4 Tapered cylindrical legs
            r_leg = 0.035
            add_mesh(_cylinder(x_min + 0.06, 0.0, z_min + 0.06, r_leg, 0.70, _COL_WOOD, base, segs=10))
            add_mesh(_cylinder(x_max - 0.06, 0.0, z_min + 0.06, r_leg, 0.70, _COL_WOOD, base, segs=10))
            add_mesh(_cylinder(x_min + 0.06, 0.0, z_max - 0.06, r_leg, 0.70, _COL_WOOD, base, segs=10))
            add_mesh(_cylinder(x_max - 0.06, 0.0, z_max - 0.06, r_leg, 0.70, _COL_WOOD, base, segs=10))
            # 2 Side Chairs (Cylindrical seats)
            add_mesh(_cylinder(cx - 0.40, 0.0, cz, 0.20, 0.42, _COL_SOFA, base, segs=12))
            add_mesh(_cylinder(cx + 0.40, 0.0, cz, 0.20, 0.42, _COL_SOFA, base, segs=12))

        # ── 4. KITCHEN COUNTER / COOKTOP (Circular burners & faucet arch) ─────
        elif "counter" in c_name or "kitchen" in c_name:
            # Main counter cabinet
            add_mesh(_quad_box(x_min, 0.0, z_min, x_max, 0.88, z_max, _COL_WHITE, base))
            # Cooktop glass surface
            add_mesh(_quad_box(cx - 0.35, 0.88, cz - 0.22, cx + 0.35, 0.90, cz + 0.22, _COL_DARK, base))
            # 4 Circular cooktop burners
            for bx, bz in [(cx - 0.18, cz - 0.10), (cx + 0.18, cz - 0.10),
                           (cx - 0.18, cz + 0.10), (cx + 0.18, cz + 0.10)]:
                add_mesh(_disk(bx, 0.902, bz, 0.08, _COL_STEEL, base, segs=12))
            # Faucet spout arch
            add_mesh(_cylinder(x_min + 0.20, 0.88, cz, 0.015, 0.20, _COL_STEEL, base, segs=8))

        # ── 5. TOILET (16-gon bowl + circular lid disk) ──────────────────────
        elif "toilet" in c_name or "wc" in c_name:
            # Cistern tank (back)
            add_mesh(_quad_box(x_min, 0.0, z_min, x_max, 0.75, z_min + 0.22, _COL_WHITE, base))
            # Ceramic bowl (16-gon cylinder base)
            bowl_cz = z_min + 0.45
            add_mesh(_cylinder(cx, 0.0, bowl_cz, 0.20, 0.42, _COL_WHITE, base, segs=16))
            # Seat rim disk
            add_mesh(_disk(cx, 0.425, bowl_cz, 0.21, _COL_WHITE, base, segs=16))
            # Flush button
            add_mesh(_disk(cx, 0.752, z_min + 0.11, 0.04, _COL_STEEL, base, segs=10))

        # ── 6. BATHROOM SINK / VANITY (Circular ceramic basin & curved faucet)
        elif "sink" in c_name or "vanity" in c_name:
            # Cabinet base
            add_mesh(_quad_box(x_min, 0.0, z_min, x_max, 0.82, z_max, _COL_WOOD, base))
            # Vanity marble countertop
            add_mesh(_quad_box(x_min - 0.02, 0.82, z_min - 0.02, x_max + 0.02, 0.86, z_max + 0.02, _COL_WHITE, base))
            # Circular ceramic basin inset
            add_mesh(_cylinder(cx, 0.86, cz, 0.18, 0.08, _COL_WHITE, base, segs=16))
            # Faucet pillar
            add_mesh(_cylinder(cx, 0.86, z_min + 0.08, 0.02, 0.16, _COL_STEEL, base, segs=8))

        # ── 7. PLANT (Tapered pot & spherical foliage) ───────────────────────
        elif "plant" in c_name:
            # Terracotta pot (16-gon cylinder)
            add_mesh(_cylinder(cx, 0.0, cz, 0.22, 0.35, _COL_POT, base, segs=16))
            # Soil surface disk
            add_mesh(_disk(cx, 0.352, cz, 0.21, _COL_WOOD, base, segs=12))
            # Lush multi-cluster spherical foliage
            add_mesh(_sphere_approx(cx, 0.60, cz, 0.28, _COL_FOLIAGE, base, lat_segs=8, lon_segs=10))
            add_mesh(_sphere_approx(cx - 0.10, 0.75, cz + 0.08, 0.20, _COL_FOLIAGE, base, lat_segs=6, lon_segs=8))
            add_mesh(_sphere_approx(cx + 0.10, 0.72, cz - 0.08, 0.18, _COL_FOLIAGE, base, lat_segs=6, lon_segs=8))

        # ── 8. DEFAULT CABINET / OFFICE DESK ─────────────────────────────────
        else:
            add_mesh(_quad_box(x_min, 0.0, z_min, x_max, 0.75, z_max, _COL_WOOD, base))

        return verts, norms, cols, idxs, base

    @staticmethod
    def build_staircase(
        stair: Stair,
        to_world: Callable[[float, float, float], np.ndarray],
        floor_height_m: float,
        base_idx: int,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray], int]:
        """Builds a 3D staircase with ascending step treads."""
        verts, norms, cols, idxs = [], [], [], []
        base = base_idx

        if not stair.bbox:
            return verts, norms, cols, idxs, base

        x, y, w, h = stair.bbox
        p0 = to_world(x, y, 0.0)
        p1 = to_world(x + w, y + h, 0.0)

        raw_xmin, raw_xmax = min(p0[0], p1[0]), max(p0[0], p1[0])
        raw_zmin, raw_zmax = min(p0[2], p1[2]), max(p0[2], p1[2])

        cx = (raw_xmin + raw_xmax) / 2.0
        cz = (raw_zmin + raw_zmax) / 2.0

        w_len = min(raw_xmax - raw_xmin, 1.8)
        d_len = min(raw_zmax - raw_zmin, 2.5)

        x_min = cx - w_len / 2.0
        x_max = cx + w_len / 2.0
        z_min = cz - d_len / 2.0
        z_max = cz + d_len / 2.0

        num_steps = 12
        step_dz = (z_max - z_min) / num_steps
        step_dy = (floor_height_m * 0.75) / num_steps

        for s in range(num_steps):
            y_end = (s + 1) * step_dy
            z_start = z_min + s * step_dz
            z_end = z_start + step_dz

            v, n, c, idx, base = _quad_box(x_min, 0.0, z_start, x_max, y_end, z_end, _COL_STAIR, base)
            verts.extend(v); norms.extend(n); cols.extend(c); idxs.extend(idx)

        return verts, norms, cols, idxs, base

    @staticmethod
    def build_balcony_railing(
        room: Room,
        to_world: Callable[[float, float, float], np.ndarray],
        rail_height_m: float = 1.05,
        base_idx: int = 0,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray], int]:
        """Builds metallic guardrails along balcony room boundary edges."""
        verts, norms, cols, idxs = [], [], [], []
        base = base_idx

        pts = room.polygon_pts
        if len(pts) < 3:
            return verts, norms, cols, idxs, base

        world_pts = [to_world(p[0], p[1], 0.0) for p in pts]

        for i in range(len(world_pts)):
            pA = world_pts[i]
            pB = world_pts[(i + 1) % len(world_pts)]

            r = 0.025
            # Vertical post at each corner
            v, n, c, idx, base = _cylinder(pA[0], 0.0, pA[2], r, rail_height_m, _COL_RAIL, base, segs=8)
            verts.extend(v); norms.extend(n); cols.extend(c); idxs.extend(idx)

            # Top rail cap
            v, n, c, idx, base = _quad_box(
                min(pA[0], pB[0]) - r, rail_height_m - 0.04, min(pA[2], pB[2]) - r,
                max(pA[0], pB[0]) + r, rail_height_m, max(pA[2], pB[2]) + r,
                _COL_RAIL, base
            )
            verts.extend(v); norms.extend(n); cols.extend(c); idxs.extend(idx)

            # Mid rail
            v, n, c, idx, base = _quad_box(
                min(pA[0], pB[0]) - r, rail_height_m * 0.50 - 0.02, min(pA[2], pB[2]) - r,
                max(pA[0], pB[0]) + r, rail_height_m * 0.50 + 0.02, max(pA[2], pB[2]) + r,
                _COL_RAIL, base
            )
            verts.extend(v); norms.extend(n); cols.extend(c); idxs.extend(idx)

        return verts, norms, cols, idxs, base
