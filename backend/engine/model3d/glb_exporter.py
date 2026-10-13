"""
backend/engine/model3d/glb_exporter.py
Blueprint-X — Milestone 4

Exports Building3D to binary glTF 2.0 (.glb) format using trimesh.
Falls back gracefully if trimesh is not installed.

GLB files can be opened in:
  - Windows 3D Viewer, Blender, Unity, Unreal
  - Web: https://gltf-viewer.donmccurdy.com
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np

from backend.models.dataclasses import Building3D

logger = logging.getLogger(__name__)


class GlbExporter:
    """
    Exports a Building3D mesh to .glb (binary glTF 2.0).
    Requires: pip install trimesh
    """

    def export(self, model: Building3D, output_path: str) -> Optional[str]:
        """
        Write <output_path> as a .glb file.

        Returns the output path on success, or None if trimesh is unavailable
        or the model is empty.
        """
        try:
            import trimesh
        except ImportError:
            logger.info(
                "[GlbExporter] trimesh not installed — skipping GLB export. "
                "Install with: pip install trimesh"
            )
            return None

        if model.vertex_count == 0:
            logger.warning("[GlbExporter] Building3D has no geometry — skipping.")
            return None

        glb_path = Path(output_path).with_suffix(".glb")

        V = model.vertices   # (N, 3) float32
        N = model.normals    # (N, 3) float32
        C = model.colors     # (N, 3) float32  (0-1 RGB)
        I = model.indices    # (M, 3) int32

        # trimesh expects uint8 vertex_colors as (N, 4) RGBA
        rgba = np.ones((len(C), 4), np.uint8)
        rgba[:, :3] = (C * 255).clip(0, 255).astype(np.uint8)

        mesh = trimesh.Trimesh(
            vertices       = V.astype(np.float64),
            faces          = I.astype(np.int64),
            vertex_normals = N.astype(np.float64),
            vertex_colors  = rgba,
            process        = False,
        )

        # Embed in a scene so glTF metadata is preserved
        scene = trimesh.Scene(geometry={"building": mesh})
        scene.export(str(glb_path))

        size_kb = glb_path.stat().st_size / 1024
        logger.info(
            f"[GlbExporter] Written {glb_path.name} "
            f"({model.vertex_count} verts, {model.triangle_count} tris, {size_kb:.1f} KB)"
        )
        return str(glb_path)
