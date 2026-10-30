"""
ui/viewer3d.py
Blueprint-X — Milestone 4

Real-time 3D OpenGL viewer for Building3D meshes.
Built on PySide6 QOpenGLWidget + PyOpenGL.

Controls
────────
  Left drag    → Orbit  (yaw / pitch)
  Right drag   → Pan
  Scroll wheel → Zoom
  Double-click → Reset view
  W            → Toggle wireframe
  G            → Toggle grid
  R            → Reset view
"""
from __future__ import annotations

import ctypes
import logging
import math
from typing import Optional, Tuple

import numpy as np

from PySide6.QtCore import Qt, QPoint, QSize, QRectF
from PySide6.QtGui import (
    QMatrix4x4, QVector3D, QSurfaceFormat, QFont,
    QPainter, QColor, QPen, QBrush, QLinearGradient,
)
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel

from backend.models.dataclasses import Building3D

logger = logging.getLogger(__name__)

# ── OpenGL import (graceful fallback) ─────────────────────────────────────────
try:
    import OpenGL.GL as gl
    _GL_OK = True
except ImportError:
    _GL_OK = False
    logger.warning("[Viewer3D] PyOpenGL not installed — 3D view disabled.")


# ── GLSL Sources ──────────────────────────────────────────────────────────────

_VERT = """
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec3 aColor;

uniform mat4 uMVP;
uniform mat4 uMV;

out vec3 vNormal;
out vec3 vColor;
out vec3 vFragPos;

void main() {
    vNormal     = mat3(uMV) * aNormal;
    vColor      = aColor;
    vFragPos    = aPos;
    gl_Position = uMVP * vec4(aPos, 1.0);
}
"""

_FRAG = """
#version 330 core
in  vec3 vNormal;
in  vec3 vColor;
in  vec3 vFragPos;
out vec4 FragColor;

uniform vec3  uLightDir;
uniform vec3  uViewPos;
uniform float uAmbient;
uniform int   uWireframe;

void main() {
    if (uWireframe == 1) {
        FragColor = vec4(min(vColor + 0.35, vec3(1.0)), 1.0);
        return;
    }
    vec3  n        = normalize(vNormal);
    vec3  l        = normalize(uLightDir);
    vec3  v        = normalize(uViewPos - vFragPos);
    vec3  r        = reflect(-l, n);

    float diff     = max(dot(n, l), 0.0);
    float back     = max(dot(-n, l), 0.0) * 0.25;
    float spec     = pow(max(dot(v, r), 0.0), 32.0) * 0.35;

    vec3  ambient  = uAmbient * vColor;
    vec3  diffuse  = (diff + back) * vColor;
    vec3  specular = spec * vec3(1.0, 1.0, 1.0);

    vec3  col      = ambient + (1.0 - uAmbient) * diffuse + specular;
    FragColor      = vec4(clamp(col, 0.0, 1.0), 1.0);
}
"""

_GRID_VERT = """
#version 330 core
layout(location = 0) in vec3 aPos;
uniform mat4 uMVP;
void main() { gl_Position = uMVP * vec4(aPos, 1.0); }
"""

_GRID_FRAG = """
#version 330 core
out vec4 FragColor;
uniform vec4 uColor;
void main() { FragColor = uColor; }
"""


# ── Utilities ─────────────────────────────────────────────────────────────────

def _mat4(m: QMatrix4x4) -> np.ndarray:
    """Extract a QMatrix4x4 into a (16,) float32 array (column-major for GL)."""
    return np.array(m.data(), dtype=np.float32)


def _compile_shader(src: str, kind: int) -> int:
    s = gl.glCreateShader(kind)
    gl.glShaderSource(s, src)
    gl.glCompileShader(s)
    if not gl.glGetShaderiv(s, gl.GL_COMPILE_STATUS):
        raise RuntimeError(f"Shader compile error:\n{gl.glGetShaderInfoLog(s).decode()}")
    return s


def _link_program(vert_src: str, frag_src: str) -> int:
    vs = _compile_shader(vert_src, gl.GL_VERTEX_SHADER)
    fs = _compile_shader(frag_src, gl.GL_FRAGMENT_SHADER)
    prog = gl.glCreateProgram()
    gl.glAttachShader(prog, vs)
    gl.glAttachShader(prog, fs)
    gl.glLinkProgram(prog)
    if not gl.glGetProgramiv(prog, gl.GL_LINK_STATUS):
        raise RuntimeError(f"Program link error:\n{gl.glGetProgramInfoLog(prog).decode()}")
    gl.glDeleteShader(vs)
    gl.glDeleteShader(fs)
    return prog


def _build_grid(size: float = 24.0, spacing: float = 1.0, y: float = -0.01) -> np.ndarray:
    """XZ-plane grid lines at height y."""
    rows, verts = int(size / spacing) + 1, []
    half = size / 2
    for i in range(rows):
        t = -half + i * spacing
        verts += [[t, y, -half], [t, y,  half]]
        verts += [[-half, y, t], [ half, y, t]]
    return np.array(verts, dtype=np.float32)


# ── Main Widget ───────────────────────────────────────────────────────────────

class Viewer3D(QOpenGLWidget):
    """
    Interactive OpenGL 3-D viewer for Building3D meshes.
    Falls back to a styled error label when PyOpenGL is unavailable.
    Includes a Qt-painted HUD overlay for stats and control hints.
    """

    def __init__(self, parent: QWidget = None):
        # Set OpenGL 3.3 Core profile BEFORE calling super()
        fmt = QSurfaceFormat()
        fmt.setVersion(3, 3)
        fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
        fmt.setDepthBufferSize(24)
        fmt.setSamples(4)
        QSurfaceFormat.setDefaultFormat(fmt)

        super().__init__(parent)

        # Model state
        self._model: Optional[Building3D] = None
        self._pending:  Optional[Building3D] = None

        # Camera
        self._yaw      = 45.0     # 45-degree corner view
        self._pitch    = -22.0    # slightly above eye level — walls clearly visible
        self._distance = 20.0
        self._pan      = QVector3D(0.0, 0.0, 0.0)
        self._center   = (0.0, 0.0, 0.0)
        self._last_pos = QPoint()
        self._camera_initialized = False
        self._pending_reset = False

        # Rendering flags
        self._wireframe = False
        self._show_grid = True
        self._show_hud  = True      # HUD toggle

        # GL object handles (created in initializeGL)
        self._prog      = 0
        self._grid_prog = 0
        self._vao = self._vbo = self._ebo = None
        self._grid_vao = self._grid_vbo = None
        self._n_indices   = 0
        self._n_grid_verts = 0
        self._gl_ready    = False

        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMinimumSize(400, 300)

    # ── Public API ────────────────────────────────────────────────────────────

    def load_model(self, model: Building3D, reset_camera: bool = False) -> None:
        """Queue a Building3D for upload on the next paint cycle."""
        self._pending = model
        self._pending_reset = reset_camera
        self.update()

    def set_wireframe(self, on: bool) -> None:
        self._wireframe = on
        self.update()

    def set_grid(self, on: bool) -> None:
        self._show_grid = on
        self.update()

    def reset_view(self) -> None:
        self._yaw = 45.0; self._pitch = -22.0
        self._pan = QVector3D(0, 0, 0)
        if self._model:
            bmin = np.array(self._model.bounds_min)
            bmax = np.array(self._model.bounds_max)
            diag = float(np.linalg.norm(bmax - bmin))
            self._distance = max(5.0, diag * 1.5)
        self.update()

    # ── OpenGL lifecycle ──────────────────────────────────────────────────────

    def initializeGL(self) -> None:
        if not _GL_OK:
            return
        try:
            gl.glClearColor(0.08, 0.09, 0.13, 1.0)
            gl.glEnable(gl.GL_DEPTH_TEST)
            gl.glEnable(gl.GL_MULTISAMPLE)

            self._prog      = _link_program(_VERT, _FRAG)
            self._grid_prog = _link_program(_GRID_VERT, _GRID_FRAG)

            # Grid geometry
            grid_v = _build_grid(24.0, 1.0)
            self._n_grid_verts = len(grid_v)
            self._grid_vao = gl.glGenVertexArrays(1)
            self._grid_vbo = gl.glGenBuffers(1)
            gl.glBindVertexArray(self._grid_vao)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._grid_vbo)
            gl.glBufferData(gl.GL_ARRAY_BUFFER, grid_v.nbytes, grid_v, gl.GL_STATIC_DRAW)
            gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 12, ctypes.c_void_p(0))
            gl.glEnableVertexAttribArray(0)
            gl.glBindVertexArray(0)

            self._gl_ready = True
        except Exception as e:
            logger.error(f"[Viewer3D] initializeGL failed: {e}")

    def resizeGL(self, w: int, h: int) -> None:
        if _GL_OK:
            gl.glViewport(0, 0, w, h)

    def paintGL(self) -> None:
        if not _GL_OK or not self._gl_ready:
            return

        # Upload pending model on GL thread
        if self._pending is not None:
            self._upload(self._pending)
            self._pending = None

        # Re-assert OpenGL state every frame to protect against QPainter state mutation
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glDepthFunc(gl.GL_LEQUAL)
        gl.glDepthMask(gl.GL_TRUE)
        gl.glEnable(gl.GL_POLYGON_OFFSET_FILL)
        gl.glPolygonOffset(1.0, 1.0)
        gl.glDisable(gl.GL_CULL_FACE)   # Render both front & back wall quads cleanly at all angles

        gl.glClearColor(0.08, 0.09, 0.13, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        view = self._view_matrix()
        proj = self._proj_matrix()
        mvp  = proj * view

        # Grid
        if self._show_grid and self._n_grid_verts > 0:
            self._draw_grid(mvp)

        # Model
        if self._n_indices > 0 and self._vao is not None:
            self._draw_model(view, mvp)

        # HUD via QPainter on top of GL frame
        if self._show_hud:
            p = QPainter()
            if p.begin(self):
                try:
                    p.setRenderHint(QPainter.RenderHint.Antialiasing)
                    self._paint_hud_with_painter(p)
                finally:
                    p.end()

    # ── HUD overlay ───────────────────────────────────────────────────────────

    def _paint_hud_with_painter(self, p: QPainter) -> None:
        """Draw informational HUD using an already-active QPainter."""
        W, H = self.width(), self.height()

        # ── Top-left stats panel ─────────────────────────────────────────────────────
        if self._model:
            stats = [
                ("Vertices",  f"{self._model.vertex_count:,}"),
                ("Triangles", f"{self._model.triangle_count:,}"),
                ("Floor Ht",  f"{self._model.floor_height_m:.1f} m"),
                ("Scale",     f"{self._model.px_per_m:.0f} px/m"),
                ("Yaw",       f"{self._yaw:.0f}°"),
                ("Pitch",     f"{self._pitch:.0f}°"),
            ]
            panel_w, panel_h = 180, len(stats) * 18 + 14
            panel_x, panel_y = 12, 12

            p.setBrush(QBrush(QColor(8, 10, 22, 200)))
            p.setPen(QPen(QColor(60, 70, 120, 180), 1))
            p.drawRoundedRect(QRectF(panel_x, panel_y, panel_w, panel_h), 8, 8)

            grad = QLinearGradient(panel_x, panel_y, panel_x + panel_w, panel_y)
            grad.setColorAt(0, QColor(99, 102, 241, 160))
            grad.setColorAt(1, QColor(34, 211, 238, 100))
            p.setBrush(QBrush(grad))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawRoundedRect(QRectF(panel_x, panel_y, panel_w, 3), 1, 1)

            font = QFont("Consolas", 9)
            p.setFont(font)
            for k, (label, value) in enumerate(stats):
                y = panel_y + 16 + k * 18
                p.setPen(QColor(100, 110, 160))
                p.drawText(panel_x + 10, y, label)
                p.setPen(QColor(200, 210, 255))
                p.drawText(panel_x + 95, y, value)

        # ── Bottom hint bar (thin, semi-transparent) ──────────────────────────────
        hints = "Drag:orbit  ·  R-drag:pan  ·  Scroll:zoom  ·  W:wire  ·  G:grid  ·  R:reset  ·  H:HUD"
        hint_h = 22
        p.setBrush(QBrush(QColor(5, 7, 18, 170)))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawRect(0, H - hint_h, W, hint_h)
        p.setPen(QColor(70, 80, 130))
        font_small = QFont("Segoe UI", 8)
        p.setFont(font_small)
        p.drawText(10, H - 6, hints)

    # ── GL draw calls ─────────────────────────────────────────────────────────

    def _draw_model(self, view: QMatrix4x4, mvp: QMatrix4x4) -> None:
        gl.glUseProgram(self._prog)

        def _u(name: str): return gl.glGetUniformLocation(self._prog, name)

        gl.glUniformMatrix4fv(_u("uMVP"), 1, gl.GL_FALSE, _mat4(mvp))
        gl.glUniformMatrix4fv(_u("uMV"),  1, gl.GL_FALSE, _mat4(view))
        gl.glUniform3f(_u("uLightDir"), 0.55, 1.0, 0.75)
        gl.glUniform3f(_u("uViewPos"), float(self._pan.x()), float(self._pan.y() + 5.0), float(self._distance))
        gl.glUniform1f(_u("uAmbient"),  0.32)
        gl.glUniform1i(_u("uWireframe"), 1 if self._wireframe else 0)

        if self._wireframe:
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)

        gl.glBindVertexArray(self._vao)
        gl.glDrawElements(gl.GL_TRIANGLES, self._n_indices, gl.GL_UNSIGNED_INT, None)
        gl.glBindVertexArray(0)

        if self._wireframe:
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)

    def _draw_grid(self, mvp: QMatrix4x4) -> None:
        gl.glUseProgram(self._grid_prog)
        gl.glUniformMatrix4fv(
            gl.glGetUniformLocation(self._grid_prog, "uMVP"),
            1, gl.GL_FALSE, _mat4(mvp)
        )
        gl.glUniform4f(
            gl.glGetUniformLocation(self._grid_prog, "uColor"),
            0.18, 0.20, 0.28, 1.0
        )
        gl.glBindVertexArray(self._grid_vao)
        gl.glDrawArrays(gl.GL_LINES, 0, self._n_grid_verts)
        gl.glBindVertexArray(0)

    # ── Model upload ──────────────────────────────────────────────────────────

    def _upload(self, model: Building3D) -> None:
        self._model = model

        # Compute centre and auto-distance
        bmin = np.array(model.bounds_min, dtype=np.float32)
        bmax = np.array(model.bounds_max, dtype=np.float32)
        cx, cy, cz = model.center
        self._center   = (cx, cy, cz)
        diag           = float(np.linalg.norm(bmax - bmin))
        self._distance = max(5.0, diag * 1.6)
        self._pitch    = -22.0   # isometric side view — walls visible
        self._yaw      = 45.0
        self._pan      = QVector3D(0, -model.floor_height_m * 0.25, 0)  # shift camera down slightly

        # Interleaved VBO data: (N, 9) → flatten
        data_v  = model.interleaved().flatten()     # float32
        data_i  = model.flat_indices().astype(np.uint32)  # uint32 for GL_UNSIGNED_INT

        STRIDE = 9 * 4  # 9 floats × 4 bytes

        if self._vao is None:
            self._vao = gl.glGenVertexArrays(1)
            self._vbo = gl.glGenBuffers(1)
            self._ebo = gl.glGenBuffers(1)

        gl.glBindVertexArray(self._vao)

        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, data_v.nbytes, data_v, gl.GL_DYNAMIC_DRAW)

        # aPos   (location 0): offset=0
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, STRIDE, ctypes.c_void_p(0))
        gl.glEnableVertexAttribArray(0)
        # aNormal (location 1): offset=12
        gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, STRIDE, ctypes.c_void_p(12))
        gl.glEnableVertexAttribArray(1)
        # aColor  (location 2): offset=24
        gl.glVertexAttribPointer(2, 3, gl.GL_FLOAT, gl.GL_FALSE, STRIDE, ctypes.c_void_p(24))
        gl.glEnableVertexAttribArray(2)

        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self._ebo)
        gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, data_i.nbytes, data_i, gl.GL_DYNAMIC_DRAW)

        gl.glBindVertexArray(0)

        self._n_indices = len(data_i)
        logger.info(f"[Viewer3D] Uploaded {model.vertex_count} verts, {model.triangle_count} tris.")
        self.update()

    # ── Camera matrices ───────────────────────────────────────────────────────

    def _view_matrix(self) -> QMatrix4x4:
        m = QMatrix4x4()
        m.translate(self._pan.x(), self._pan.y(), -self._distance)
        m.rotate(self._pitch, 1.0, 0.0, 0.0)
        m.rotate(self._yaw,   0.0, 1.0, 0.0)
        cx, cy, cz = self._center
        m.translate(-cx, -cy, -cz)
        return m

    def _proj_matrix(self) -> QMatrix4x4:
        m = QMatrix4x4()
        w, h = max(1, self.width()), max(1, self.height())
        m.perspective(45.0, w / h, 0.05, 500.0)
        return m

    # ── Mouse / keyboard controls ─────────────────────────────────────────────

    def mousePressEvent(self, event) -> None:
        self._last_pos = event.position().toPoint()

    def mouseMoveEvent(self, event) -> None:
        pos  = event.position().toPoint()
        dx   = pos.x() - self._last_pos.x()
        dy   = pos.y() - self._last_pos.y()

        if event.buttons() & Qt.MouseButton.LeftButton:
            # Orbit
            self._yaw   = (self._yaw   + dx * 0.45) % 360.0
            self._pitch = max(-89.0, min(89.0, self._pitch + dy * 0.45))

        elif event.buttons() & (Qt.MouseButton.RightButton | Qt.MouseButton.MiddleButton):
            # Pan (in view-space XY plane)
            pan_speed = self._distance * 0.0025
            self._pan += QVector3D(-dx * pan_speed, dy * pan_speed, 0.0)

        self._last_pos = pos
        self.update()

    def wheelEvent(self, event) -> None:
        delta = event.angleDelta().y()
        factor = 1.0 - delta / 1200.0
        self._distance = max(0.5, min(300.0, self._distance * factor))
        self.update()

    def mouseDoubleClickEvent(self, event) -> None:
        self.reset_view()

    def keyPressEvent(self, event) -> None:
        key = event.key()
        if key == Qt.Key.Key_W:
            self._wireframe = not self._wireframe
            self.update()
        elif key == Qt.Key.Key_G:
            self._show_grid = not self._show_grid
            self.update()
        elif key == Qt.Key.Key_R:
            self.reset_view()
        elif key == Qt.Key.Key_H:
            self._show_hud = not self._show_hud
            self.update()


class Viewer3DPlaceholder(QWidget):
    """Shown when OpenGL is unavailable or no model is loaded yet."""

    def __init__(self, message: str = "Run detection to generate 3D model"):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background: #0f1117;")

        icon = QLabel("🏗")
        icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon.setStyleSheet("font-size: 56px; margin-bottom: 16px;")

        msg = QLabel(message)
        msg.setAlignment(Qt.AlignmentFlag.AlignCenter)
        msg.setStyleSheet("color: #8892b0; font-size: 15px;")

        hint = QLabel("Left drag: orbit  ·  Right drag: pan  ·  Scroll: zoom  ·  W: wireframe  ·  H: HUD")
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hint.setStyleSheet("color: #4a5270; font-size: 12px; margin-top: 10px;")

        layout.addWidget(icon)
        layout.addWidget(msg)
        layout.addWidget(hint)
