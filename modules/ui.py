"""
modules/ui.py
Blueprint-X Prototype - Milestone 1

PySide6 main window with:
  - Dark-themed, three-panel layout  (original | processed | reconstruction)
  - Left sidebar with controls, statistics, and status messages
  - Progress indicator during processing
  - Export button wired to json_exporter
  - Worker thread so OpenCV processing never freezes the UI
"""

import sys
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from PySide6.QtCore import (
    Qt, QThread, Signal, QSize, QTimer, QPropertyAnimation,
    QEasingCurve, Property, QObject,
)
from PySide6.QtGui import (
    QImage, QPixmap, QFont, QColor, QPalette, QIcon,
    QFontDatabase, QLinearGradient, QPainter, QBrush,
)
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QFrame, QScrollArea,
    QProgressBar, QSizePolicy, QSplitter, QGraphicsDropShadowEffect,
    QStatusBar, QSpacerItem, QSlider, QTabWidget
)

from modules.image_loader import load_floor_plan, FloorPlanImage, ImageLoadError
from modules.structural_pipeline import run_pipeline, BlueprintAnalysis
from modules.overlay_renderer import (
    render_overlay, render_wall_extraction_debug,
    render_room_segmentation_debug, render_topology_graph_debug
)
from modules.json_exporter import (
    build_export_payload, export_to_json, suggest_output_path, ExportError,
)


# ---------------------------------------------------------------------------
# Colour tokens
# ---------------------------------------------------------------------------
C_BG          = "#0f1117"
C_SURFACE     = "#1a1d27"
C_SURFACE2    = "#22263a"
C_BORDER      = "#2e3350"
C_ACCENT      = "#5865f2"
C_ACCENT2     = "#7c83ff"
C_TEXT        = "#e8eaf6"
C_TEXT_DIM    = "#8892b0"
C_SUCCESS     = "#43b581"
C_WARNING     = "#faa61a"
C_ERROR       = "#f04747"
C_OUTER_WALL  = "#dccc00"
C_INNER_WALL  = "#ff8c00"


# ---------------------------------------------------------------------------
# Worker thread
# ---------------------------------------------------------------------------

class AnalysisWorker(QThread):
    """
    Runs the heavy OpenCV pipeline off the main thread.
    Emits progress steps (0-100) and the final results.
    """
    progress   = Signal(int, str)                  # (percent, message)
    finished   = Signal(object)    # (BlueprintAnalysis)
    error      = Signal(str)

    def __init__(self, file_path: str):
        super().__init__()
        self.file_path = file_path

    def run(self):
        try:
            print(f"[Worker] Loading image from {self.file_path}")
            self.progress.emit(10, "Loading image…")
            fp = load_floor_plan(self.file_path)

            print("[Worker] Running pipeline...")
            self.progress.emit(40, "Running multi-layer structural pipeline…")
            analysis = run_pipeline(fp)

            print("[Worker] Emitting finished signal...")
            self.progress.emit(100, "Analysis complete.")
            self.finished.emit(analysis)
            print("[Worker] Run completed successfully.")

        except ImageLoadError as e:
            print(f"[Worker] ImageLoadError: {e}")
            self.error.emit(str(e))
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(f"Unexpected error: {e}")


# ---------------------------------------------------------------------------
# Helpers: numpy → QPixmap
# ---------------------------------------------------------------------------

def bgr_to_pixmap(bgr: np.ndarray) -> QPixmap:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    qi = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qi)


def gray_to_pixmap(gray: np.ndarray) -> QPixmap:
    h, w = gray.shape
    qi = QImage(gray.data, w, h, w, QImage.Format.Format_Grayscale8)
    return QPixmap.fromImage(qi)


# ---------------------------------------------------------------------------
# Reusable UI widgets
# ---------------------------------------------------------------------------

class PanelHeader(QLabel):
    """Small label used as the title bar inside each image panel."""
    def __init__(self, text: str, icon: str = ""):
        super().__init__(f"{icon}  {text}" if icon else text)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet(f"""
            QLabel {{
                background: {C_SURFACE2};
                color: {C_TEXT_DIM};
                font-size: 11px;
                font-weight: 600;
                letter-spacing: 1.5px;
                text-transform: uppercase;
                padding: 8px 0;
                border-bottom: 1px solid {C_BORDER};
            }}
        """)


class ImagePanel(QFrame):
    """
    A framed panel that holds a centred, aspect-ratio-preserving image label.
    """
    def __init__(self, title: str, subtitle: str = "", icon: str = ""):
        super().__init__()
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setStyleSheet(f"""
            QFrame {{
                background: {C_SURFACE};
                border: 1px solid {C_BORDER};
                border-radius: 10px;
            }}
        """)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.header = PanelHeader(title, icon)
        layout.addWidget(self.header)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.image_label.setMinimumSize(200, 150)

        # Placeholder text
        self._placeholder = subtitle or "No image loaded"
        self._show_placeholder()
        layout.addWidget(self.image_label, 1)

    def _show_placeholder(self):
        self.image_label.setText(self._placeholder)
        self.image_label.setStyleSheet(f"""
            QLabel {{
                color: {C_TEXT_DIM};
                font-size: 13px;
                background: transparent;
            }}
        """)

    def set_image(self, bgr: np.ndarray):
        pixmap = bgr_to_pixmap(bgr)
        self._set_pixmap(pixmap)

    def set_pixmap(self, pixmap: QPixmap):
        self._set_pixmap(pixmap)

    def _set_pixmap(self, pixmap: QPixmap):
        self.image_label.setStyleSheet("background: transparent;")
        scaled = pixmap.scaled(
            self.image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.image_label.setPixmap(scaled)
        self._raw_pixmap = pixmap

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "_raw_pixmap"):
            scaled = self._raw_pixmap.scaled(
                self.image_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.image_label.setPixmap(scaled)

    def clear(self):
        self.image_label.setPixmap(QPixmap())
        self._show_placeholder()
        if hasattr(self, "_raw_pixmap"):
            del self._raw_pixmap


class SidebarButton(QPushButton):
    """Styled sidebar action button."""
    def __init__(self, text: str, primary: bool = False, icon_char: str = ""):
        label = f"{icon_char}  {text}" if icon_char else text
        super().__init__(label)
        bg      = C_ACCENT  if primary else C_SURFACE2
        bg_hov  = C_ACCENT2 if primary else C_BORDER
        self.setStyleSheet(f"""
            QPushButton {{
                background: {bg};
                color: {C_TEXT};
                border: 1px solid {C_BORDER};
                border-radius: 8px;
                padding: 10px 14px;
                font-size: 13px;
                font-weight: 600;
                text-align: left;
            }}
            QPushButton:hover {{
                background: {bg_hov};
                border-color: {C_ACCENT};
            }}
            QPushButton:pressed {{
                background: {C_ACCENT};
            }}
            QPushButton:disabled {{
                color: {C_TEXT_DIM};
                background: {C_SURFACE};
                border-color: {C_SURFACE2};
            }}
        """)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMinimumHeight(44)


class StatRow(QWidget):
    """Key-value stat display row."""
    def __init__(self, key: str, value: str = "—"):
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 4)

        self.key_lbl = QLabel(key)
        self.key_lbl.setStyleSheet(f"color: {C_TEXT_DIM}; font-size: 12px;")

        self.val_lbl = QLabel(value)
        self.val_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.val_lbl.setStyleSheet(f"color: {C_TEXT}; font-size: 12px; font-weight: 600;")

        layout.addWidget(self.key_lbl)
        layout.addStretch()
        layout.addWidget(self.val_lbl)

    def update_value(self, value: str):
        self.val_lbl.setText(value)


class Divider(QFrame):
    def __init__(self):
        super().__init__()
        self.setFrameShape(QFrame.Shape.HLine)
        self.setStyleSheet(f"color: {C_BORDER}; margin: 6px 0;")


class StatusBadge(QLabel):
    """
    Coloured pill-shaped status indicator.
    States: idle | loading | success | error
    """
    _STYLES = {
        "idle":    (C_TEXT_DIM, C_SURFACE2),
        "loading": (C_WARNING,  "#2a2200"),
        "success": (C_SUCCESS,  "#0d2112"),
        "error":   (C_ERROR,    "#1f0000"),
    }

    def __init__(self):
        super().__init__("● Ready")
        self.set_state("idle")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def set_state(self, state: str, message: str = ""):
        fg, bg = self._STYLES.get(state, self._STYLES["idle"])
        icons = {"idle": "●", "loading": "◌", "success": "✓", "error": "✕"}
        icon = icons.get(state, "●")
        label_map = {"idle": "Ready", "loading": "Processing…", "success": "Done", "error": "Error"}
        text = message if message else label_map.get(state, state.capitalize())
        self.setText(f"{icon}  {text}")
        self.setStyleSheet(f"""
            QLabel {{
                color: {fg};
                background: {bg};
                border: 1px solid {fg};
                border-radius: 10px;
                padding: 4px 12px;
                font-size: 11px;
                font-weight: 700;
            }}
        """)


# ---------------------------------------------------------------------------
# Main Window
# ---------------------------------------------------------------------------

class BlueprintXWindow(QMainWindow):

    # ---- constructor -------------------------------------------------------

    def __init__(self):
        super().__init__()
        self._analysis: Optional[BlueprintAnalysis] = None
        self._worker: Optional[AnalysisWorker] = None

        self.setWindowTitle("Blueprint-X Prototype  —  Milestone 1")
        self.setMinimumSize(1200, 720)
        self.resize(1400, 820)
        self._apply_global_style()
        self._build_ui()

    # ---- styling -----------------------------------------------------------

    def _apply_global_style(self):
        self.setStyleSheet(f"""
            QMainWindow, QWidget {{
                background: {C_BG};
                color: {C_TEXT};
                font-family: 'Segoe UI', 'Inter', 'Helvetica Neue', sans-serif;
            }}
            QScrollArea {{ border: none; background: transparent; }}
            QScrollBar:vertical {{
                background: {C_SURFACE};
                width: 6px;
                margin: 0;
                border-radius: 3px;
            }}
            QScrollBar::handle:vertical {{
                background: {C_BORDER};
                border-radius: 3px;
                min-height: 20px;
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
            QProgressBar {{
                background: {C_SURFACE2};
                border: 1px solid {C_BORDER};
                border-radius: 6px;
                height: 8px;
                text-align: center;
            }}
            QProgressBar::chunk {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {C_ACCENT}, stop:1 {C_ACCENT2});
                border-radius: 6px;
            }}
            QStatusBar {{
                background: {C_SURFACE};
                color: {C_TEXT_DIM};
                font-size: 11px;
                border-top: 1px solid {C_BORDER};
                padding: 2px 8px;
            }}
        """)

    # ---- UI construction ---------------------------------------------------

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Sidebar
        sidebar = self._build_sidebar()
        root.addWidget(sidebar)

        # Panels area
        panels_widget = self._build_panels()
        root.addWidget(panels_widget, 1)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Welcome to Blueprint-X  —  Upload a floor plan to begin.")

    # ---- sidebar -----------------------------------------------------------

    def _build_sidebar(self) -> QWidget:
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFixedWidth(280)  # slightly wider to accommodate scrollbar
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setStyleSheet(f"""
            QScrollArea {{
                background: {C_SURFACE};
                border: none;
                border-right: 1px solid {C_BORDER};
            }}
            QScrollArea > QWidget > QWidget {{
                background: {C_SURFACE};
            }}
        """)

        sidebar = QFrame()
        sidebar.setStyleSheet(f"""
            QFrame {{
                background: transparent;
                border: none;
            }}
        """)

        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(16, 20, 16, 16)
        layout.setSpacing(10)

        # ---- Logo / title ----
        logo_lbl = QLabel("Blueprint-X")
        logo_lbl.setStyleSheet(f"""
            color: {C_ACCENT2};
            font-size: 22px;
            font-weight: 800;
            letter-spacing: -0.5px;
        """)
        sub_lbl = QLabel("Prototype  ·  Milestone 1")
        sub_lbl.setStyleSheet(f"color: {C_TEXT_DIM}; font-size: 10px; letter-spacing: 1px;")
        layout.addWidget(logo_lbl)
        layout.addWidget(sub_lbl)
        layout.addWidget(Divider())

        # ---- Actions ----
        section_lbl = QLabel("ACTIONS")
        section_lbl.setStyleSheet(f"color: {C_TEXT_DIM}; font-size: 10px; letter-spacing: 2px; font-weight: 700;")
        layout.addWidget(section_lbl)

        self.btn_upload = SidebarButton("Upload Floor Plan", primary=True, icon_char="📂")
        self.btn_upload.clicked.connect(self._on_upload)
        layout.addWidget(self.btn_upload)

        self.btn_analyse = SidebarButton("Analyse Blueprint", icon_char="🔍")
        self.btn_analyse.setEnabled(False)
        self.btn_analyse.clicked.connect(self._on_analyse)
        layout.addWidget(self.btn_analyse)

        self.btn_export = SidebarButton("Export Building Data", icon_char="💾")
        self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(self._on_export)
        layout.addWidget(self.btn_export)

        self.btn_clear = SidebarButton("Clear", icon_char="✕")
        self.btn_clear.setEnabled(False)
        self.btn_clear.clicked.connect(self._on_clear)
        layout.addWidget(self.btn_clear)

        layout.addWidget(Divider())

        # ---- Progress ----
        prog_lbl = QLabel("PROGRESS")
        prog_lbl.setStyleSheet(f"color: {C_TEXT_DIM}; font-size: 10px; letter-spacing: 2px; font-weight: 700;")
        layout.addWidget(prog_lbl)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(8)
        layout.addWidget(self.progress_bar)

        self.status_badge = StatusBadge()
        layout.addWidget(self.status_badge)

        self.progress_msg = QLabel("")
        self.progress_msg.setStyleSheet(f"color: {C_TEXT_DIM}; font-size: 11px;")
        self.progress_msg.setWordWrap(True)
        layout.addWidget(self.progress_msg)

        layout.addWidget(Divider())

        # ---- Overlay Controls ----
        overlay_lbl = QLabel("OVERLAY")
        overlay_lbl.setStyleSheet(f"color: {C_TEXT_DIM}; font-size: 10px; letter-spacing: 2px; font-weight: 700;")
        layout.addWidget(overlay_lbl)

        self.slider_alpha = QSlider(Qt.Orientation.Horizontal)
        self.slider_alpha.setRange(0, 100)
        self.slider_alpha.setValue(70)
        self.slider_alpha.setEnabled(False)
        self.slider_alpha.valueChanged.connect(self._on_alpha_changed)
        layout.addWidget(self.slider_alpha)
        
        layout.addWidget(Divider())

        # ---- Statistics ----
        stats_lbl = QLabel("STATISTICS")
        stats_lbl.setStyleSheet(f"color: {C_TEXT_DIM}; font-size: 10px; letter-spacing: 2px; font-weight: 700;")
        layout.addWidget(stats_lbl)

        self.stat_rooms     = StatRow("Total Rooms")
        self.stat_doors     = StatRow("Total Doors")
        self.stat_windows   = StatRow("Total Windows")
        self.stat_stairs    = StatRow("Total Staircases")
        self.stat_furniture = StatRow("Total Furniture")
        self.stat_area      = StatRow("Est. Floor Area")
        self.stat_conf      = StatRow("Confidence")

        for w in [self.stat_rooms, self.stat_doors, self.stat_windows,
                  self.stat_stairs, self.stat_furniture, self.stat_area, self.stat_conf]:
            layout.addWidget(w)

        layout.addStretch()
        layout.addWidget(Divider())

        # ---- Legend ----
        legend_lbl = QLabel("WALL LEGEND")
        legend_lbl.setStyleSheet(f"color: {C_TEXT_DIM}; font-size: 10px; letter-spacing: 2px; font-weight: 700;")
        layout.addWidget(legend_lbl)

        def legend_row(color, label):
            row = QWidget()
            h = QHBoxLayout(row)
            h.setContentsMargins(0, 2, 0, 2)
            dot = QLabel("━━")
            dot.setStyleSheet(f"color: {color}; font-size: 14px; font-weight: 900;")
            txt = QLabel(label)
            txt.setStyleSheet(f"color: {C_TEXT_DIM}; font-size: 11px;")
            h.addWidget(dot)
            h.addWidget(txt)
            h.addStretch()
            return row

        layout.addWidget(legend_row(C_OUTER_WALL,  "Outer Wall"))
        layout.addWidget(legend_row(C_INNER_WALL,  "Internal Wall"))
        layout.addWidget(legend_row("#00FF00",     "Door"))
        layout.addWidget(legend_row("#00FFFF",     "Window"))
        layout.addWidget(legend_row("#B432B4",     "Furniture"))

        # ---- Footer version ----
        ver = QLabel("v0.1.0  ·  © 2024 Blueprint-X")
        ver.setStyleSheet(f"color: {C_TEXT_DIM}; font-size: 9px; margin-top: 6px;")
        ver.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(ver)

        scroll_area.setWidget(sidebar)
        return scroll_area

    # ---- panels area -------------------------------------------------------

    def _build_panels(self) -> QWidget:
        container = QWidget()
        container.setStyleSheet(f"background: {C_BG};")
        layout = QVBoxLayout(container)
        layout.setContentsMargins(16, 16, 16, 8)
        layout.setSpacing(8)

        # Top bar
        top_bar = QHBoxLayout()
        title = QLabel("Floor Plan Analysis")
        title.setStyleSheet(f"""
            color: {C_TEXT};
            font-size: 18px;
            font-weight: 700;
        """)
        hint = QLabel("Upload a PNG / JPG floor plan to get started")
        hint.setStyleSheet(f"color: {C_TEXT_DIM}; font-size: 12px;")
        top_bar.addWidget(title)
        top_bar.addStretch()
        top_bar.addWidget(hint)
        layout.addLayout(top_bar)

        # Three panels in a splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(6)
        splitter.setStyleSheet(f"""
            QSplitter::handle {{
                background: {C_BORDER};
                border-radius: 3px;
            }}
        """)

        self.panel_original    = ImagePanel("Original Blueprint",   "Upload a floor plan image", "📄")
        self.panel_processed   = ImagePanel("Detected Walls",        "Walls will appear here after analysis", "🔍")
        
        # Tabs for right panel
        self.right_tabs = QTabWidget()
        self.right_tabs.setStyleSheet(f"""
            QTabWidget::pane {{ border: 1px solid {C_BORDER}; border-radius: 4px; }}
            QTabBar::tab {{ background: {C_SURFACE2}; color: {C_TEXT_DIM}; padding: 8px 16px; border: 1px solid {C_BORDER}; border-bottom: none; border-top-left-radius: 4px; border-top-right-radius: 4px; }}
            QTabBar::tab:selected {{ background: {C_SURFACE}; color: {C_TEXT}; font-weight: bold; border-bottom: 1px solid {C_SURFACE}; }}
        """)
        
        self.panel_reconstruct = ImagePanel("2D Reconstruction",     "Simplified layout will appear here", "🗺")
        self.panel_overlay = ImagePanel("Overlay Comparison", "Adjust transparency in sidebar", "👁")
        self.panel_debug_walls = ImagePanel("Structural Walls", "Wall extraction debug view", "🧱")
        self.panel_debug_rooms = ImagePanel("Room Segmentation", "Room extraction debug view", "🟩")
        self.panel_debug_graph = ImagePanel("Topology Graph", "Adjacency graph debug view", "🕸")
        
        self.right_tabs.addTab(self.panel_reconstruct, "Reconstruction")
        self.right_tabs.addTab(self.panel_overlay, "Overlay")
        self.right_tabs.addTab(self.panel_debug_walls, "Debug Walls")
        self.right_tabs.addTab(self.panel_debug_rooms, "Debug Rooms")
        self.right_tabs.addTab(self.panel_debug_graph, "Topology")

        splitter.addWidget(self.panel_original)
        splitter.addWidget(self.panel_processed)
        splitter.addWidget(self.right_tabs)
        splitter.setSizes([400, 400, 400])

        layout.addWidget(splitter, 1)

        return container

    # ---- slots / event handlers --------------------------------------------

    def _on_upload(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Floor Plan Image",
            "",
            "Images (*.png *.jpg *.jpeg);;All Files (*)",
        )
        if not path:
            return

        # Show thumbnail immediately
        try:
            fp = load_floor_plan(path)
        except ImageLoadError as e:
            self._set_status("error", str(e))
            return

        self._uploaded_path = path
        self._analysis = None
        self.panel_original.set_image(fp.original_bgr)
        self.panel_processed.clear()
        self.panel_reconstruct.clear()
        self.panel_overlay.clear()
        self._reset_stats()
        self.slider_alpha.setEnabled(False)

        self.btn_analyse.setEnabled(True)
        self.btn_export.setEnabled(False)
        self.btn_clear.setEnabled(True)

        self.progress_bar.setValue(0)
        self._set_status("idle", f"Loaded: {fp.file_path.name}")
        self.status_bar.showMessage(f"Loaded  {path}")

    def _on_analyse(self):
        if self.panel_original.image_label.pixmap() is None:
            return

        self.btn_upload.setEnabled(False)
        self.btn_analyse.setEnabled(False)
        self.btn_export.setEnabled(False)
        self.progress_bar.setValue(0)
        self._set_status("loading")

        # We need the path. Load_floor_plan has it in a roundabout way, or we can store it.
        # It's better to store fp and pass that to the worker to avoid reloading.
        # But for now, we just grab the path from the status bar or a stored variable.
        # Easiest is to store the path during _on_upload. Let's assume we still have _floor_plan.
        # I'll re-add _floor_plan just for the path.
        if not hasattr(self, '_uploaded_path'):
            return
            
        self._worker = AnalysisWorker(self._uploaded_path)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_analysis_done)
        self._worker.error.connect(self._on_analysis_error)
        self._worker.start()

    def _on_progress(self, pct: int, msg: str):
        self.progress_bar.setValue(pct)
        self.progress_msg.setText(msg)
        self.status_bar.showMessage(msg)

    def _on_analysis_done(self, analysis: BlueprintAnalysis):
        self._analysis = analysis

        # Update panels
        self.panel_original.set_image(analysis.floor_plan.original_bgr)
        self.panel_processed.set_image(analysis.walls.annotated_bgr)
        self.panel_reconstruct.set_image(analysis.rooms.reconstruction_bgr)
        
        # Render Debug Views
        self.panel_debug_walls.set_image(render_wall_extraction_debug(analysis))
        self.panel_debug_rooms.set_image(render_room_segmentation_debug(analysis))
        self.panel_debug_graph.set_image(render_topology_graph_debug(analysis))
        
        # Initial overlay render
        self._on_alpha_changed(self.slider_alpha.value())
        self.slider_alpha.setEnabled(True)

        # Update stats
        self.stat_rooms.update_value(str(analysis.rooms.estimated_room_count))
        self.stat_doors.update_value(str(analysis.doors_windows.total_doors))
        self.stat_windows.update_value(str(analysis.doors_windows.total_windows))
        self.stat_stairs.update_value(str(analysis.doors_windows.total_stairs))
        self.stat_furniture.update_value(str(len(analysis.furniture)))
        self.stat_area.update_value(f"{analysis.estimated_floor_area_px2:,.0f} px²")
        self.stat_conf.update_value(f"{analysis.reconstruction_confidence:.0%}")

        self.btn_upload.setEnabled(True)
        self.btn_analyse.setEnabled(True)
        self.btn_export.setEnabled(True)
        self.btn_clear.setEnabled(True)
        self.progress_bar.setValue(100)
        self._set_status("success", "Analysis complete")
        self.status_bar.showMessage(
            f"Analysis complete  ·  {analysis.rooms.estimated_room_count} rooms  "
            f"·  {len(analysis.furniture)} furniture objects"
        )
        
    def _on_alpha_changed(self, value: int):
        if not self._analysis:
            return
        alpha = value / 100.0
        overlay_bgr = render_overlay(self._analysis.floor_plan.original_bgr, self._analysis, alpha)
        self.panel_overlay.set_image(overlay_bgr)

    def _on_analysis_error(self, message: str):
        self.btn_upload.setEnabled(True)
        self.btn_analyse.setEnabled(True)
        self.progress_bar.setValue(0)
        self._set_status("error", message)
        self.status_bar.showMessage(f"Error: {message}")

    def _on_export(self):
        if not self._analysis:
            return

        default_path = suggest_output_path(self._analysis.floor_plan)
        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Building Data",
            default_path,
            "JSON Files (*.json);;All Files (*)",
        )
        if not save_path:
            return

        try:
            payload = build_export_payload(self._analysis)
            out = export_to_json(payload, save_path)
            self._set_status("success", "Exported successfully")
            self.status_bar.showMessage(f"Exported → {out}")
        except ExportError as e:
            self._set_status("error", str(e))
            self.status_bar.showMessage(f"Export failed: {e}")

    def _on_clear(self):
        self._analysis = None

        self.panel_original.clear()
        self.panel_processed.clear()
        self.panel_reconstruct.clear()
        self.panel_overlay.clear()
        self._reset_stats()
        self.slider_alpha.setEnabled(False)

        self.btn_analyse.setEnabled(False)
        self.btn_export.setEnabled(False)
        self.btn_clear.setEnabled(False)
        self.progress_bar.setValue(0)
        self.progress_msg.setText("")
        self._set_status("idle")
        self.status_bar.showMessage("Cleared  —  Upload a floor plan to begin.")

    # ---- helpers -----------------------------------------------------------

    def _set_status(self, state: str, message: str = ""):
        self.status_badge.set_state(state, message)

    def _reset_stats(self):
        for stat in [self.stat_rooms, self.stat_doors, self.stat_windows,
                     self.stat_stairs, self.stat_furniture, self.stat_area, self.stat_conf]:
            stat.update_value("—")
