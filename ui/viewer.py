"""
ui/viewer.py
Blueprint-X — Redesign

Full-screen zoomable and pannable image viewer.
Built on QGraphicsView / QGraphicsScene.
"""

import numpy as np
import cv2
from PySide6.QtCore import Qt, QPointF, Signal
from PySide6.QtGui import QImage, QPixmap, QWheelEvent, QMouseEvent, QPainter
from PySide6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QWidget, QVBoxLayout, QLabel

from ui.widgets import C_BG, C_TEXT_DIM

def _bgr_to_pixmap(bgr: np.ndarray) -> QPixmap:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    qi = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qi)

class ZoomableImageViewer(QGraphicsView):
    """
    A QGraphicsView subclass that supports zooming (mouse wheel) and panning (drag).
    """
    room_clicked = Signal(float, float)

    def __init__(self, parent: QWidget = None):
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)
        
        # Setup view properties
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        self.setStyleSheet(f"background: {C_BG}; border: none;")
        
        self._zoom_factor = 1.15
        self._is_empty = True
        self._mouse_pressed = False
        
    def set_image(self, bgr: np.ndarray):
        """Set the image and reset zoom to fit if this is the first load."""
        pixmap = _bgr_to_pixmap(bgr)
        self.pixmap_item.setPixmap(pixmap)
        self.scene.setSceneRect(self.pixmap_item.boundingRect())
        
        if self._is_empty:
            self.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
            self._is_empty = False

    def clear(self):
        self.pixmap_item.setPixmap(QPixmap())
        self._is_empty = True

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self._mouse_pressed = True
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if self._mouse_pressed and event.button() == Qt.MouseButton.LeftButton:
            # Emit scene coordinates on click
            pos = self.mapToScene(event.position().toPoint())
            self.room_clicked.emit(float(pos.x()), float(pos.y()))
        self._mouse_pressed = False
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event: QWheelEvent):
        """Zoom in/out with mouse wheel."""
        if self._is_empty:
            return

        zoom_in = event.angleDelta().y() > 0
        factor = self._zoom_factor if zoom_in else 1.0 / self._zoom_factor
        
        # Map viewport pos to scene pos
        target_scene_pos = self.mapToScene(event.position().toPoint())
        
        self.scale(factor, factor)
        
        # Center on the cursor
        current_scene_pos = self.mapToScene(event.position().toPoint())
        delta = target_scene_pos - current_scene_pos
        self.translate(-delta.x(), -delta.y())
        
    def mouseDoubleClickEvent(self, event: QMouseEvent):
        """Reset zoom to fit on double click."""
        if not self._is_empty:
            self.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

class PlaceholderViewer(QWidget):
    """Empty state viewer."""
    def __init__(self, message: str):
        super().__init__()
        layout = QVBoxLayout(self)
        lbl = QLabel(message)
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setStyleSheet(f"color: {C_TEXT_DIM}; font-size: 14px;")
        layout.addWidget(lbl)
        self.setStyleSheet(f"background: {C_BG};")
