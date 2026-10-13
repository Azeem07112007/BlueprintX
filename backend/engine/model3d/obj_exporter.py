"""
backend/engine/model3d/obj_exporter.py
Blueprint-X — Milestone 4

Exports Building3D to Wavefront OBJ + MTL files.
Zero external dependencies — pure Python + numpy stdlib only.

Output
------
<base>.obj  — geometry (vertices, normals, faces)
<base>.mtl  — material colours

OBJ coordinate system uses right-handed Y-up (same as Building3D).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from backend.models.dataclasses import Building3D

logger = logging.getLogger(__name__)


class ObjExporter:
    """Exports a Building3D mesh to Wavefront OBJ / MTL."""

    def export(
        self,
        model: Building3D,
        output_path: str,
        mtl_name: Optional[str] = None,
    ) -> str:
        """
        Write <output_path>.obj and <output_path>.mtl.

        Parameters
        ----------
        model : Building3D
            The mesh to export.
        output_path : str
            Destination path (with or without .obj extension).
        mtl_name : str, optional
            Override the MTL filename; defaults to <stem>.mtl.

        Returns
        -------
        str
            Absolute path to the written OBJ file.
        """
        obj_path = Path(output_path).with_suffix(".obj")
        mtl_path = obj_path.with_suffix(".mtl")
        mtl_ref  = mtl_name or mtl_path.name

        if model.vertex_count == 0:
            logger.warning("[ObjExporter] Building3D has no geometry — skipping.")
            return str(obj_path)

        V = model.vertices   # (N, 3)
        N = model.normals    # (N, 3)
        C = model.colors     # (N, 3) 0-1 RGB
        I = model.indices    # (M, 3) int32

        # ── Write MTL ─────────────────────────────────────────────────────────
        # We assign a unique material per triangle based on its colour.
        # Round colours to 2 decimals to group similar colours.
        unique_colors: dict[str, np.ndarray] = {}
        tri_materials: list[str] = []
        for tri in I:
            avg_col = C[tri].mean(axis=0)
            key = f"m_{int(avg_col[0]*100):03d}_{int(avg_col[1]*100):03d}_{int(avg_col[2]*100):03d}"
            unique_colors[key] = avg_col
            tri_materials.append(key)

        mtl_lines = ["# Blueprint-X Material Library\n"]
        for mat_name, col in unique_colors.items():
            r, g, b = float(col[0]), float(col[1]), float(col[2])
            mtl_lines += [
                f"newmtl {mat_name}\n",
                f"Ka {r:.4f} {g:.4f} {b:.4f}\n",
                f"Kd {r:.4f} {g:.4f} {b:.4f}\n",
                f"Ks 0.1 0.1 0.1\n",
                f"Ns 32.0\n",
                f"d 1.0\n",
                f"illum 2\n\n",
            ]
        mtl_path.write_text("".join(mtl_lines), encoding="utf-8")

        # ── Write OBJ ─────────────────────────────────────────────────────────
        obj_lines = [
            f"# Blueprint-X 3D Export\n",
            f"# Vertices: {len(V)}   Triangles: {len(I)}\n",
            f"# Floor height: {model.floor_height_m:.2f} m\n",
            f"# Scale: {model.px_per_m:.1f} px/m\n\n",
            f"mtllib {mtl_ref}\n\n",
        ]

        # Vertices
        for (x, y, z) in V:
            obj_lines.append(f"v {x:.6f} {y:.6f} {z:.6f}\n")
        obj_lines.append("\n")

        # Normals
        for (nx, ny, nz) in N:
            obj_lines.append(f"vn {nx:.6f} {ny:.6f} {nz:.6f}\n")
        obj_lines.append("\n")

        # Faces (grouped by material)
        obj_lines.append("o Building\n")
        current_mat = None
        for tri_i, (i0, i1, i2) in enumerate(I):
            mat = tri_materials[tri_i]
            if mat != current_mat:
                obj_lines.append(f"usemtl {mat}\n")
                current_mat = mat
            # OBJ uses 1-based indexing; format: v//vn
            a, b, c = int(i0) + 1, int(i1) + 1, int(i2) + 1
            obj_lines.append(f"f {a}//{a} {b}//{b} {c}//{c}\n")

        obj_path.write_text("".join(obj_lines), encoding="utf-8")

        size_kb = obj_path.stat().st_size / 1024
        logger.info(
            f"[ObjExporter] Written {obj_path.name} "
            f"({len(V)} verts, {len(I)} tris, {size_kb:.1f} KB)"
        )
        return str(obj_path)
