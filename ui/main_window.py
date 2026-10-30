"""
ui/main_window.py
Blueprint-X — Milestone 4

Main application window.
  - Left sidebar: controls, stats, 3D settings
  - Top tab bar: 8 views (Original → 3D View)
  - Centre: ZoomableImageViewer (2D) or Viewer3D (3D tab)
"""
from __future__ import annotations

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QClipboard, QColor
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QScrollArea, QFrame, QLabel, QProgressBar,
    QSlider, QTabWidget, QFileDialog, QSplitter, QStackedWidget,
    QCheckBox, QApplication, QPushButton, QColorDialog,
)

from pipeline import BlueprintPipeline, PipelineResult
from ui.widgets import (
    SidebarButton, StatRow, Divider, StatusBadge, SectionLabel,
    C_BG, C_SURFACE, C_SURFACE2, C_BORDER, C_TEXT, C_TEXT_DIM,
    C_ACCENT, C_ACCENT2, C_GOLD,
)
from ui.viewer import ZoomableImageViewer, PlaceholderViewer
from ui.loading_overlay import LoadingOverlay
from ui.room_editor import RoomEditorPanel

# ── Worker Thread ─────────────────────────────────────────────────────────────

class PipelineWorker(QThread):
    progress = Signal(int, str)
    finished = Signal(object)
    error    = Signal(str)

    def __init__(self, file_path: str):
        super().__init__()
        self.file_path = file_path

    def run(self):
        try:
            pipeline = BlueprintPipeline()

            def cb(step, total, msg):
                self.progress.emit(int(step / total * 100), msg)

            result = pipeline.run(self.file_path, progress_cb=cb)
            self.finished.emit(result)
        except Exception as e:
            import traceback; traceback.print_exc()
            self.error.emit(str(e))


# ── Main Window ───────────────────────────────────────────────────────────────

class BlueprintXWindow(QMainWindow):

    # Tab indices
    TAB_ORIGINAL  = 0
    TAB_WALLS     = 1
    TAB_ROOMS     = 2
    TAB_OVERLAY   = 3
    TAB_TOPOLOGY  = 4
    TAB_SEMANTICS = 5
    TAB_DEBUG     = 6
    TAB_3D        = 7

    def __init__(self):
        super().__init__()
        self._result: PipelineResult | None = None
        self._worker: PipelineWorker | None = None
        self._viewer3d = None   # lazily created to avoid GL init before widget shown
        self._selected_room_id: int | None = None

        # 3D Customization state
        self._include_furniture: bool = True
        self._custom_wall_color_rgb: tuple[float, float, float] | None = None
        self._custom_wall_color_hex: str | None = None
        self._preset_buttons: list[QPushButton] = []

        self.setWindowTitle("Blueprint-X  —  M4")
        self.setMinimumSize(1200, 720)
        self.resize(1440, 860)

        self._apply_global_style()
        self._build_ui()

        # Full-window overlay (must be created AFTER _build_ui so self is sized)
        self._overlay = LoadingOverlay(self)
        self._overlay.hide()

    # ── Styling ───────────────────────────────────────────────────────────────

    def _apply_global_style(self):
        self.setStyleSheet(f"""
            QMainWindow, QWidget {{
                background: {C_BG};
                color: {C_TEXT};
                font-family: 'Segoe UI', 'Inter', 'Helvetica Neue', sans-serif;
            }}
            QScrollArea {{ border: none; background: transparent; }}
            QScrollBar:vertical {{
                background: transparent; width: 5px; margin: 0;
            }}
            QScrollBar::handle:vertical {{
                background: {C_BORDER}; border-radius: 2px; min-height: 24px;
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
            QProgressBar {{
                background: {C_SURFACE}; border: 1px solid {C_BORDER};
                border-radius: 6px; height: 6px; text-align: center;
            }}
            QProgressBar::chunk {{
                background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                    stop:0 {C_ACCENT}, stop:1 {C_ACCENT2});
                border-radius: 6px;
            }}
            QCheckBox {{ color: {C_TEXT_DIM}; font-size: 12px; spacing: 7px; }}
            QCheckBox::indicator {{
                width: 16px; height: 16px; border-radius: 4px;
                border: 1px solid {C_BORDER}; background: {C_SURFACE};
            }}
            QCheckBox::indicator:checked {{
                background: qlineargradient(x1:0,y1:0,x2:1,y2:1,
                    stop:0 {C_ACCENT}, stop:1 {C_ACCENT2});
                border-color: {C_ACCENT};
            }}
            QSlider::groove:horizontal {{
                background: {C_SURFACE}; border: 1px solid {C_BORDER};
                height: 4px; border-radius: 2px;
            }}
            QSlider::handle:horizontal {{
                background: {C_ACCENT2}; border: 2px solid {C_ACCENT};
                width: 14px; height: 14px; margin: -6px 0; border-radius: 7px;
            }}
            QSlider::sub-page:horizontal {{
                background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                    stop:0 {C_ACCENT}, stop:1 {C_ACCENT2});
                border-radius: 2px;
            }}
        """)

    # ── Layout ────────────────────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        left  = self._build_left_panel()
        mid   = self._build_main_area()
        right = self._build_right_panel()

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(mid)
        splitter.addWidget(right)
        splitter.setSizes([260, 1000, 320])
        splitter.setCollapsible(0, False)
        splitter.setCollapsible(2, False)
        splitter.setStyleSheet(f"""
            QSplitter::handle {{ background: {C_BORDER}; width: 2px; }}
        """)
        root.addWidget(splitter)

    def _build_main_area(self) -> QWidget:
        area   = QWidget()
        layout = QVBoxLayout(area)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # ── Tab bar ───────────────────────────────────────────────────────
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet(f"""
            QTabWidget::pane {{
                border: none;
                border-top: 1px solid {C_BORDER};
            }}
            QTabBar::tab {{
                background: {C_BG};
                color: {C_TEXT_DIM};
                padding: 12px 20px;
                border: none;
                border-right: 1px solid {C_BORDER};
                font-size: 13px;
                font-weight: 600;
            }}
            QTabBar::tab:selected {{
                background: {C_SURFACE};
                color: {C_TEXT};
                border-bottom: 2px solid {C_ACCENT};
            }}
            QTabBar::tab:hover {{ background: {C_SURFACE}; color: {C_TEXT}; }}
        """)
        for label in ["🖼  Original", "🧱  Walls", "🟦  Rooms", "🌈  Overlay",
                      "🔗  Topology", "🏷  Semantics", "🔬  Debug", "🏗  3D View"]:
            self.tabs.addTab(QWidget(), label)
        self.tabs.currentChanged.connect(self._on_tab_changed)
        layout.addWidget(self.tabs)

        # ── Debug sub-tabs ────────────────────────────────────────────────
        self.debug_tabs = QTabWidget()
        self.debug_tabs.setStyleSheet(f"""
            QTabWidget::pane {{ border: none; border-top: 1px solid {C_BORDER}; }}
            QTabBar::tab {{
                background: {C_BG}; color: {C_TEXT_DIM};
                padding: 8px 12px; border: none;
                border-right: 1px solid {C_BORDER}; font-size: 11px;
            }}
            QTabBar::tab:selected {{ color: {C_ACCENT2}; border-bottom: 1px solid {C_ACCENT2}; }}
        """)
        self.debug_keys = [
            "01_grayscale", "02_deskewed", "05_binary", "08_cleaned_binary",
            "wd_01_skeleton", "wd_02_hough", "wd_03_walls",
            "ai_01_detections", "ocr_01_mask",
        ]
        for k in self.debug_keys:
            self.debug_tabs.addTab(QWidget(), k)
        self.debug_tabs.currentChanged.connect(self._on_debug_tab_changed)
        self.debug_tabs.setVisible(False)
        layout.addWidget(self.debug_tabs)

        # ── Viewer stack (2D image viewer vs 3D viewer) ───────────────────
        self.view_stack = QStackedWidget()

        # Page 0 — 2D viewers
        page_2d = QWidget()
        p2d_lay = QVBoxLayout(page_2d)
        p2d_lay.setContentsMargins(0, 0, 0, 0)
        self.viewer      = ZoomableImageViewer()
        self.placeholder = PlaceholderViewer("Upload a floor plan to begin.")
        p2d_lay.addWidget(self.viewer)
        p2d_lay.addWidget(self.placeholder)
        self.viewer.setVisible(False)
        self.view_stack.addWidget(page_2d)      # index 0

        # Page 1 — 3D viewer (lazy import)
        self._page_3d = QWidget()
        p3d_lay = QVBoxLayout(self._page_3d)
        p3d_lay.setContentsMargins(0, 0, 0, 0)
        self.view_stack.addWidget(self._page_3d)  # index 1

        layout.addWidget(self.view_stack, 1)
        return area

    def _build_left_panel(self) -> QWidget:
        """Left panel: logo, actions, progress, overlay, 3D controls, stats."""
        container = QFrame()
        container.setMinimumWidth(250)
        container.setMaximumWidth(320)
        container.setStyleSheet(f"""
            QFrame {{
                background: {C_SURFACE};
                border-right: 1px solid {C_BORDER};
            }}
        """)

        container_lay = QVBoxLayout(container)
        container_lay.setContentsMargins(0, 0, 0, 0)
        container_lay.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        panel = QWidget()
        panel.setStyleSheet(f"background: {C_SURFACE};")
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(14, 16, 14, 14)
        lay.setSpacing(8)

        # Logo
        logo = QLabel("Blueprint-X")
        logo.setStyleSheet(
            f"color: {C_ACCENT2}; font-size: 22px; font-weight: 900;"
            " letter-spacing: -0.8px; background: transparent; border: none;"
        )
        lay.addWidget(logo)
        sub = QLabel("AI Floor Plan → 3D")
        sub.setStyleSheet(
            f"color: {C_TEXT_DIM}; font-size: 10px; letter-spacing: 1px;"
            " background: transparent; border: none;"
        )
        lay.addWidget(sub)
        lay.addWidget(Divider())

        # ── Actions ───────────────────────────────────────────────────────
        lay.addWidget(SectionLabel("ACTIONS"))

        self.btn_upload = SidebarButton("Upload Blueprint", primary=True, icon_char="📂")
        self.btn_upload.clicked.connect(self._on_upload)
        lay.addWidget(self.btn_upload)

        self.btn_analyse = SidebarButton("Run Deep Analysis", icon_char="🔍")
        self.btn_analyse.setEnabled(False)
        self.btn_analyse.clicked.connect(self._on_analyse)
        lay.addWidget(self.btn_analyse)

        # Export row (3 compact buttons side by side)
        exp_row = QHBoxLayout()
        exp_row.setSpacing(6)
        self.btn_export = SidebarButton("JSON", compact=True, icon_char="💾")
        self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(self._on_export_json)
        exp_row.addWidget(self.btn_export)
        self.btn_export_obj = SidebarButton("OBJ", compact=True, icon_char="📦")
        self.btn_export_obj.setEnabled(False)
        self.btn_export_obj.clicked.connect(self._on_export_obj)
        exp_row.addWidget(self.btn_export_obj)
        self.btn_export_glb = SidebarButton("GLB", compact=True, icon_char="🌐")
        self.btn_export_glb.setEnabled(False)
        self.btn_export_glb.clicked.connect(self._on_export_glb)
        exp_row.addWidget(self.btn_export_glb)
        lay.addLayout(exp_row)

        self.btn_copy_report = SidebarButton("Copy Report", icon_char="📋")
        self.btn_copy_report.setEnabled(False)
        self.btn_copy_report.clicked.connect(self._on_copy_report)
        lay.addWidget(self.btn_copy_report)

        self.btn_launch_unity = SidebarButton("Launch Unity 3D", icon_char="🎮")
        self.btn_launch_unity.setEnabled(False)
        self.btn_launch_unity.clicked.connect(self._on_launch_unity)
        lay.addWidget(self.btn_launch_unity)

        lay.addWidget(Divider())

        # ── Progress ──────────────────────────────────────────────────────
        lay.addWidget(SectionLabel("STATUS"))
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        lay.addWidget(self.progress_bar)

        self.status_badge = StatusBadge()
        lay.addWidget(self.status_badge)

        self.progress_msg = QLabel("")
        self.progress_msg.setStyleSheet(
            f"color: {C_TEXT_DIM}; font-size: 10px; background: transparent; border: none;"
        )
        self.progress_msg.setWordWrap(True)
        lay.addWidget(self.progress_msg)
        lay.addWidget(Divider())

        # ── Overlay opacity ───────────────────────────────────────────────
        lay.addWidget(SectionLabel("OVERLAY OPACITY"))
        self.slider_alpha = QSlider(Qt.Orientation.Horizontal)
        self.slider_alpha.setRange(0, 100)
        self.slider_alpha.setValue(65)
        self.slider_alpha.setEnabled(False)
        self.slider_alpha.valueChanged.connect(self._on_alpha_changed)
        lay.addWidget(self.slider_alpha)
        lay.addWidget(Divider())

        # ── 3D Controls ───────────────────────────────────────────────────
        lay.addWidget(SectionLabel("3D VIEW"))

        chk_row = QHBoxLayout()
        chk_row.setSpacing(10)
        self.chk_wireframe = QCheckBox("Wire")
        self.chk_wireframe.setChecked(False)
        self.chk_wireframe.toggled.connect(self._on_wireframe_toggled)
        chk_row.addWidget(self.chk_wireframe)
        self.chk_grid = QCheckBox("Grid")
        self.chk_grid.setChecked(True)
        self.chk_grid.toggled.connect(self._on_grid_toggled)
        chk_row.addWidget(self.chk_grid)
        self.chk_ceiling = QCheckBox("Ceiling")
        self.chk_ceiling.setChecked(False)
        self.chk_ceiling.toggled.connect(self._on_ceiling_toggled)
        chk_row.addWidget(self.chk_ceiling)
        lay.addLayout(chk_row)

        self.btn_reset_cam = SidebarButton("Reset Camera", icon_char="🎯")
        self.btn_reset_cam.setMinimumHeight(36)
        self.btn_reset_cam.clicked.connect(self._on_reset_camera)
        lay.addWidget(self.btn_reset_cam)

        lbl_h = QLabel("Floor Height: 2.8 m")
        lbl_h.setStyleSheet(
            f"color: {C_TEXT_DIM}; font-size: 11px; background: transparent; border: none;"
        )
        lay.addWidget(lbl_h)
        self._lbl_floor_h = lbl_h

        self.slider_floor_h = QSlider(Qt.Orientation.Horizontal)
        self.slider_floor_h.setRange(20, 60)
        self.slider_floor_h.setValue(28)
        self.slider_floor_h.setEnabled(False)
        self.slider_floor_h.valueChanged.connect(self._on_floor_height_changed)
        lay.addWidget(self.slider_floor_h)

        # Furniture placement toggle button
        self.btn_toggle_furniture = QPushButton("🛋️  Furniture: Placed")
        self.btn_toggle_furniture.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_toggle_furniture.setToolTip("Toggle placement of 3D furniture models inside rooms")
        self.btn_toggle_furniture.setEnabled(False)
        self.btn_toggle_furniture.clicked.connect(self._on_toggle_furniture)
        self._update_furniture_btn_style()
        lay.addWidget(self.btn_toggle_furniture)

        lay.addWidget(Divider())

        # ── Wall Customization ────────────────────────────────────────────
        lay.addWidget(SectionLabel("WALL CUSTOMIZATION"))

        lbl_w = QLabel("Wall Width: 0.30 m (30 cm)")
        lbl_w.setStyleSheet(
            f"color: {C_TEXT_DIM}; font-size: 11px; background: transparent; border: none;"
        )
        lay.addWidget(lbl_w)
        self._lbl_wall_w = lbl_w

        self.slider_wall_w = QSlider(Qt.Orientation.Horizontal)
        self.slider_wall_w.setRange(10, 80)
        self.slider_wall_w.setValue(30)
        self.slider_wall_w.setEnabled(False)
        self.slider_wall_w.setToolTip("Adjust thickness/width of walls in metres (10 cm to 80 cm)")
        self.slider_wall_w.valueChanged.connect(self._on_wall_width_changed)
        lay.addWidget(self.slider_wall_w)

        lbl_col = QLabel("Wall Paint Color:")
        lbl_col.setStyleSheet(
            f"color: {C_TEXT_DIM}; font-size: 11px; background: transparent; border: none;"
        )
        lay.addWidget(lbl_col)

        # Swatch + Name + Custom Button
        col_row = QHBoxLayout()
        col_row.setSpacing(8)

        self.color_swatch = QLabel()
        self.color_swatch.setFixedSize(22, 22)
        self.color_swatch.setStyleSheet(
            f"background: #DFC752; border: 1px solid {C_BORDER}; border-radius: 4px;"
        )
        col_row.addWidget(self.color_swatch)

        self.lbl_wall_color = QLabel("Default Gold")
        self.lbl_wall_color.setStyleSheet(
            f"color: {C_TEXT}; font-size: 11px; font-weight: 600; background: transparent; border: none;"
        )
        col_row.addWidget(self.lbl_wall_color)
        col_row.addStretch()

        self.btn_pick_color = SidebarButton("Pick Color...", compact=True, icon_char="🎨")
        self.btn_pick_color.setEnabled(False)
        self.btn_pick_color.setToolTip("Open color picker dialog to choose a custom paint color")
        self.btn_pick_color.clicked.connect(self._on_pick_wall_color)
        col_row.addWidget(self.btn_pick_color)
        lay.addLayout(col_row)

        # Presets row
        preset_row = QHBoxLayout()
        preset_row.setSpacing(5)

        self._preset_buttons.clear()
        presets = [
            ("Default", "#DFC752", (0.88, 0.78, 0.32)),
            ("White",   "#F5F5F0", (0.96, 0.96, 0.94)),
            ("Greige",  "#D5CDC2", (0.84, 0.80, 0.76)),
            ("Slate",   "#546274", (0.33, 0.38, 0.45)),
            ("Sage",    "#7E937A", (0.49, 0.58, 0.48)),
            ("Terra",   "#C4664F", (0.77, 0.40, 0.31)),
            ("Navy",    "#23324A", (0.14, 0.20, 0.29)),
        ]

        for name, hex_val, (pr, pg, pb) in presets:
            chip = QPushButton()
            chip.setFixedSize(22, 22)
            chip.setToolTip(f"{name} ({hex_val})")
            chip.setCursor(Qt.CursorShape.PointingHandCursor)
            chip.setStyleSheet(f"""
                QPushButton {{
                    background: {hex_val};
                    border: 1px solid {C_BORDER};
                    border-radius: 4px;
                }}
                QPushButton:hover {{
                    border: 2px solid {C_ACCENT2};
                }}
                QPushButton:disabled {{
                    opacity: 0.35;
                    border-color: {C_BORDER};
                }}
            """)
            chip.clicked.connect(lambda _, r=pr, g=pg, b=pb, hx=hex_val, nm=name: self._set_wall_color(r, g, b, hx, nm))
            chip.setEnabled(False)
            self._preset_buttons.append(chip)
            preset_row.addWidget(chip)

        self.btn_reset_color = SidebarButton("↺", compact=True)
        self.btn_reset_color.setToolTip("Reset to Default Architectural Colors")
        self.btn_reset_color.setFixedWidth(28)
        self.btn_reset_color.setEnabled(False)
        self.btn_reset_color.clicked.connect(self._on_reset_wall_color)
        preset_row.addWidget(self.btn_reset_color)
        lay.addLayout(preset_row)

        lay.addWidget(Divider())

        # ── Statistics ────────────────────────────────────────────────────
        lay.addWidget(SectionLabel("STATISTICS"))
        self.stat_rooms   = StatRow("Rooms")
        self.stat_walls   = StatRow("Walls")
        self.stat_doors   = StatRow("Doors")
        self.stat_windows = StatRow("Windows")
        self.stat_text    = StatRow("Text Blocks")
        self.stat_verts   = StatRow("3D Vertices")
        self.stat_tris    = StatRow("3D Triangles")
        for s in [self.stat_rooms, self.stat_walls, self.stat_doors,
                  self.stat_windows, self.stat_text,
                  self.stat_verts, self.stat_tris]:
            lay.addWidget(s)

        lay.addStretch()

        scroll.setWidget(panel)
        container_lay.addWidget(scroll)
        return container

    def _build_right_panel(self) -> QWidget:
        """Right panel: full-height Room Label Editor."""
        panel = QFrame()
        panel.setMinimumWidth(260)
        panel.setMaximumWidth(400)
        panel.setStyleSheet(f"""
            QFrame {{
                background: {C_SURFACE};
                border-left: 1px solid {C_BORDER};
            }}
        """)
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(14, 16, 14, 14)
        lay.setSpacing(10)

        # Header
        hdr = QLabel("🏷  Room Label Editor")
        hdr.setStyleSheet(
            f"color: {C_TEXT}; font-size: 14px; font-weight: 800;"
            " background: transparent; border: none;"
        )
        lay.addWidget(hdr)

        sub = QLabel("Click any room in the floor plan or in the list below to edit its label. Changes apply to the 3D model.")
        sub.setWordWrap(True)
        sub.setStyleSheet(
            f"color: {C_TEXT_DIM}; font-size: 10px; background: transparent; border: none;"
        )
        lay.addWidget(sub)
        lay.addWidget(Divider())

        # Room editor (takes all remaining space)
        self.room_editor = RoomEditorPanel()
        self.room_editor.label_changed.connect(self._on_room_label_changed)
        self.room_editor.rebuild_requested.connect(self._on_rebuild_3d)
        self.room_editor.room_selected.connect(self._on_room_selected_from_editor)
        # Remove the internal max-height cap from the scroll area so it fills the panel
        for child in self.room_editor.findChildren(QScrollArea):
            child.setMaximumHeight(16777215)  # Qt's QWIDGETSIZE_MAX
        lay.addWidget(self.room_editor, 1)   # stretch = 1 → takes all space

        return panel


    # ── Lazy 3D viewer ────────────────────────────────────────────────────────

    def _get_viewer3d(self):
        """Create the Viewer3D on first use (avoids premature GL context init)."""
        if self._viewer3d is None:
            try:
                from ui.viewer3d import Viewer3D, Viewer3DPlaceholder
                self._viewer3d = Viewer3D(self._page_3d)
                lay = self._page_3d.layout()
                if lay:
                    lay.addWidget(self._viewer3d)
                else:
                    from PySide6.QtWidgets import QVBoxLayout as _VBL
                    _l = _VBL(self._page_3d)
                    _l.setContentsMargins(0, 0, 0, 0)
                    _l.addWidget(self._viewer3d)
            except Exception as e:
                import traceback; traceback.print_exc()
        return self._viewer3d

    # ── Actions ───────────────────────────────────────────────────────────────

    def _on_upload(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Floor Plan", "", "Images (*.png *.jpg *.jpeg)"
        )
        if not path:
            return
        self._uploaded_path = path
        self._result        = None

        v3d = self._get_viewer3d()
        if v3d is not None:
            v3d._camera_initialized = False

        import cv2
        bgr = cv2.imread(path)
        if bgr is not None:
            self.viewer.set_image(bgr)
            self.viewer.setVisible(True)
            self.placeholder.setVisible(False)

        self.btn_analyse.setEnabled(True)
        self.btn_export.setEnabled(False)
        self.btn_export_obj.setEnabled(False)
        self.btn_export_glb.setEnabled(False)
        self.slider_alpha.setEnabled(False)
        self.slider_floor_h.setEnabled(False)
        self.btn_toggle_furniture.setEnabled(False)
        self.slider_wall_w.setEnabled(False)
        self.btn_pick_color.setEnabled(False)
        self.btn_reset_color.setEnabled(False)
        for chip in self._preset_buttons:
            chip.setEnabled(False)
        self.progress_bar.setValue(0)
        self.progress_msg.setText("")
        self.status_badge.set_state("idle", f"Loaded {path.split('/')[-1]}")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._overlay and self._overlay.isVisible():
            self._overlay.resize_to_parent()

    def _on_analyse(self):
        self.btn_upload.setEnabled(False)
        self.btn_analyse.setEnabled(False)
        self.status_badge.set_state("loading")
        self.progress_bar.setValue(0)

        # Show animated loading overlay
        if self._overlay:
            self._overlay.show_overlay(
                "Deep Analysis Running...",
                "Initializing pipeline"
            )

        self._worker = PipelineWorker(self._uploaded_path)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_analysis_done)
        self._worker.error.connect(self._on_analysis_error)
        self._worker.start()

    def _on_progress(self, pct: int, msg: str):
        self.progress_bar.setValue(pct)
        self.progress_msg.setText(msg)
        if self._overlay:
            self._overlay.update_step(msg)
            self._overlay.set_progress(pct / 100.0)
            stage_map = {
                10: "Preprocessing Image",
                20: "Preprocessing Image",
                30: "AI Object Detection",
                40: "OCR Text Extraction",
                50: "Wall Detection",
                60: "Geometry Graph",
                70: "Topology Extraction",
                80: "Room Generation",
                90: "Semantic Fusion",
                100: "3D Model Generation",
            }
            closest = min(stage_map, key=lambda k: abs(k - pct))
            self._overlay.update_stage(stage_map[closest])

    def _on_analysis_done(self, result: PipelineResult):
        # Hide loading overlay
        if self._overlay:
            self._overlay.hide_overlay()

        self._result = result
        b = result.building

        self.stat_rooms.update_value(str(len(b.rooms)))
        self.stat_walls.update_value(str(len(b.walls)))
        self.stat_doors.update_value(str(len(b.doors)))
        self.stat_windows.update_value(str(len(b.windows)))
        self.stat_text.update_value(str(len(b.text_blocks)))

        if b.model_3d:
            self.stat_verts.update_value(f"{b.model_3d.vertex_count:,}")
            self.stat_tris.update_value(f"{b.model_3d.triangle_count:,}")
            # Pre-load into 3D viewer
            v3d = self._get_viewer3d()
            if v3d and b.model_3d:
                v3d.load_model(b.model_3d, reset_camera=True)

        # Populate room editor
        self.room_editor.load_rooms(b.rooms)
        # Connect floor-plan clicks to room hit-test
        self.viewer.room_clicked.connect(self._on_viewer_room_clicked)

        self.btn_upload.setEnabled(True)
        self.btn_analyse.setEnabled(True)
        self.btn_export.setEnabled(True)
        self.btn_export_obj.setEnabled(bool(b.model_3d))
        self.btn_export_glb.setEnabled(bool(b.model_3d))
        self.btn_copy_report.setEnabled(True)
        self.btn_launch_unity.setEnabled(bool(b.model_3d))
        self.slider_alpha.setEnabled(True)
        self.slider_floor_h.setEnabled(True)
        self.btn_toggle_furniture.setEnabled(True)
        self.slider_wall_w.setEnabled(True)
        self.btn_pick_color.setEnabled(True)
        self.btn_reset_color.setEnabled(True)
        for chip in self._preset_buttons:
            chip.setEnabled(True)
        self.progress_bar.setValue(100)
        self.status_badge.set_state("success", "Analysis Complete")

        # Apply any active customized 3D / wall settings if not default
        if not self._include_furniture or self._custom_wall_color_rgb is not None or self.slider_wall_w.value() != 30 or self.chk_ceiling.isChecked():
            self._rebuild_3d()

        # Room stat — show count + total area if available
        total_area = sum(r.area_m2 for r in b.rooms if r.area_m2 > 0)
        rooms_txt = str(len(b.rooms))
        if total_area > 0:
            rooms_txt += f"  ·  {total_area:.0f} m²"
        self.stat_rooms.update_value(rooms_txt)

        self._update_viewer()

    def _on_analysis_error(self, msg: str):
        if self._overlay:
            self._overlay.hide_overlay()
        self.btn_upload.setEnabled(True)
        self.btn_analyse.setEnabled(True)
        self.progress_bar.setValue(0)
        self.status_badge.set_state("error", "Failed")
        self.progress_msg.setText(msg[:120])

    # ── Exports ───────────────────────────────────────────────────────────────

    def _on_copy_report(self):
        """Copy a text summary of the analysis to the clipboard."""
        if not self._result:
            return
        b = self._result.building
        lines = [
            "Blueprint-X — Analysis Report",
            "=" * 40,
            f"Source:    {b.source_path.split('/')[-1].split(chr(92))[-1]}",
            f"Image:     {b.image_size[0]} × {b.image_size[1]} px",
            f"Scale:     {b.scale or 'unknown'}",
            "",
            f"Walls:     {len(b.walls)}  (outer: {sum(1 for w in b.walls if w.wall_type=='outer')},"
            f" internal: {sum(1 for w in b.walls if w.wall_type=='internal')})",
            f"Rooms:     {len(b.rooms)}",
            f"Doors:     {len(b.doors)}",
            f"Windows:   {len(b.windows)}",
            f"Text blks: {len(b.text_blocks)}",
        ]
        total_area = sum(r.area_m2 for r in b.rooms if r.area_m2 > 0)
        if total_area > 0:
            lines.append(f"Total area: {total_area:.1f} m\u00b2")
        lines += ["", "Rooms:", "-" * 30]
        for r in b.rooms:
            area = f"  {r.area_m2:.1f} m\u00b2" if r.area_m2 > 0 else ""
            adj = f"  adj→{r.adjacent_room_ids}" if r.adjacent_room_ids else ""
            lines.append(f"  R{r.room_id:02d}  {r.label or 'Unknown'}{area}{adj}")
        if b.model_3d:
            lines += ["", "3D Model:", "-" * 30,
                      f"  Vertices:  {b.model_3d.vertex_count:,}",
                      f"  Triangles: {b.model_3d.triangle_count:,}",
                      f"  Floor Ht:  {b.model_3d.floor_height_m:.1f} m"]
        report = "\n".join(lines)
        QApplication.clipboard().setText(report)
        self.status_badge.set_state("success", "Report Copied")
        self.progress_msg.setText("Text report copied to clipboard.")

    def _on_export_json(self):
        if not self._result:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export JSON", "building_export.json", "JSON (*.json)"
        )
        if not path:
            return
        try:
            from backend.engine.export.exporter import Exporter
            Exporter().export(self._result.building, path)
            self.status_badge.set_state("success", "JSON Saved")
            self.progress_msg.setText(f"Saved: {path}")
        except Exception as e:
            self.status_badge.set_state("error", "Export Failed")
            self.progress_msg.setText(str(e))

    def _on_export_obj(self):
        if not (self._result and self._result.building.model_3d):
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export OBJ", "building_3d.obj", "Wavefront OBJ (*.obj)"
        )
        if not path:
            return
        try:
            from backend.engine.model3d.obj_exporter import ObjExporter
            ObjExporter().export(self._result.building.model_3d, path)
            self.status_badge.set_state("success", "OBJ Saved")
            self.progress_msg.setText(f"Saved: {path}")
        except Exception as e:
            self.status_badge.set_state("error", "OBJ Export Failed")
            self.progress_msg.setText(str(e))

    def _on_export_glb(self):
        if not (self._result and self._result.building.model_3d):
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export GLB", "building_3d.glb", "glTF Binary (*.glb)"
        )
        if not path:
            return
        try:
            from backend.engine.model3d.glb_exporter import GlbExporter
            out = GlbExporter().export(self._result.building.model_3d, path)
            if out:
                self.status_badge.set_state("success", "GLB Saved")
                self.progress_msg.setText(f"Saved: {out}")
            else:
                self.status_badge.set_state("error", "GLB: trimesh not installed")
                self.progress_msg.setText("pip install trimesh")
        except Exception as e:
            self.status_badge.set_state("error", "GLB Export Failed")
            self.progress_msg.setText(str(e))

    def _on_launch_unity(self):
        if not self._result:
            return

        import os, subprocess, shutil
        from pathlib import Path
        from backend.engine.export.exporter import Exporter
        from backend.engine.model3d.obj_exporter import ObjExporter
        from backend.engine.model3d.glb_exporter import GlbExporter

        try:
            # 1. Delete all previous exported files before creating new ones
            root_json = Path("building_3d.json").resolve()
            root_obj  = Path("building_3d.obj").resolve()
            root_mtl  = Path("building_3d.mtl").resolve()
            root_glb  = Path("building_3d.glb").resolve()

            for old_file in [root_json, root_obj, root_mtl, root_glb]:
                if old_file.exists():
                    try:
                        old_file.unlink()
                    except Exception:
                        pass

            unity_streaming_dirs = [
                Path(r"D:\ME GA\GAmes\to be named\BlueprintX_Unity\Assets\StreamingAssets"),
                Path(r"D:\ME GA\GAmes\to be named\BlueprintX_Unity\Build\BlueprintX_3D_Data\StreamingAssets"),
                Path(r"D:\BlueprintX\unity\build\BlueprintX_3D_Data\StreamingAssets"),
                Path(r"unity\BlueprintX_Unity\Assets\StreamingAssets").resolve()
            ]

            for s_dir in unity_streaming_dirs:
                if s_dir.exists():
                    for f_name in ["building_3d.json", "building_3d.obj", "building_3d.mtl", "building_3d.glb"]:
                        old_s_file = s_dir / f_name
                        if old_s_file.exists():
                            try:
                                old_s_file.unlink()
                            except Exception:
                                pass

            # Export new JSON v2.1.0, OBJ, MTL, and GLB
            Exporter().export(self._result.building, str(root_json))
            if self._result.building.model_3d:
                ObjExporter().export(self._result.building.model_3d, str(root_obj))
                GlbExporter().export(self._result.building.model_3d, str(root_glb))

            # 2. Copy fresh files to all Unity project StreamingAssets locations
            for s_dir in unity_streaming_dirs:
                try:
                    if s_dir.parent.exists():
                        s_dir.mkdir(parents=True, exist_ok=True)
                        shutil.copy(root_json, s_dir / "building_3d.json")
                        if root_obj.exists():
                            shutil.copy(root_obj, s_dir / "building_3d.obj")
                        if root_mtl.exists():
                            shutil.copy(root_mtl, s_dir / "building_3d.mtl")
                        if root_glb.exists():
                            shutil.copy(root_glb, s_dir / "building_3d.glb")
                except Exception as ex:
                    print(f"StreamingAssets copy warning for {s_dir}: {ex}")

            # 3. Check for Unity Standalone Executable
            candidate_exes = [
                Path(os.environ.get("UNITY_EXE_PATH", "")),
                Path(r"D:\BlueprintX\unity\build\BlueprintX_3D.exe"),
                Path(r"D:\ME GA\GAmes\to be named\BlueprintX_Unity\Build\BlueprintX_3D.exe"),
                Path(r"unity\build\BlueprintX_3D.exe").resolve(),
                Path("BlueprintX_3D.exe").resolve(),
            ]

            launched_exe = None
            for cand in candidate_exes:
                if str(cand) and cand.exists() and cand.is_file():
                    launched_exe = cand
                    break

            if launched_exe:
                # Copy JSON directly into exe directory as well as StreamingAssets
                try:
                    shutil.copy(root_json, launched_exe.parent / "building_3d.json")
                    s_assets = launched_exe.parent / "BlueprintX_3D_Data" / "StreamingAssets"
                    s_assets.mkdir(parents=True, exist_ok=True)
                    shutil.copy(root_json, s_assets / "building_3d.json")
                except Exception:
                    pass

                subprocess.Popen([str(launched_exe)], cwd=str(launched_exe.parent))
                self.status_badge.set_state("success", "Unity App Launched")
                self.progress_msg.setText(f"Running standalone 3D player: {launched_exe.name}")
            else:
                # Try opening Unity Editor directly with project path if Unity.exe exists
                unity_editor_exe = Path(r"C:\Program Files\Unity\Hub\Editor\6000.5.3f1-x86_64\Editor\Unity.exe")
                if not unity_editor_exe.exists():
                    unity_editor_exe = Path(r"C:\Program Files\Unity\Hub\Editor\6000.5.3f1\Editor\Unity.exe")
                unity_proj_path = Path(r"D:\ME GA\GAmes\to be named\BlueprintX_Unity")
                if not unity_proj_path.exists():
                    unity_proj_path = Path(r"d:\BlueprintX\unity\BlueprintX_Unity").resolve()

                if unity_editor_exe.exists() and unity_proj_path.exists():
                    try:
                        subprocess.Popen([str(unity_editor_exe), "-projectPath", str(unity_proj_path)])
                        self.status_badge.set_state("success", "Unity Editor Opened")
                        self.progress_msg.setText("Opening Unity Editor... Press Play (▶) or click BlueprintX -> Build Standalone Player.")
                    except Exception:
                        self.status_badge.set_state("success", "Unity Data Fresh")
                        self.progress_msg.setText("Data exported! In Unity, click BlueprintX -> Build Standalone Player to build the EXE.")
                else:
                    self.status_badge.set_state("success", "Unity Data Fresh")
                    self.progress_msg.setText("Data exported! In Unity, click BlueprintX -> Build Standalone Player to build the EXE.")
        except Exception as e:
            self.status_badge.set_state("error", "Unity Launch Error")
            self.progress_msg.setText(str(e))

    # ── 3D controls ───────────────────────────────────────────────────────────

    def _on_room_label_changed(self, room_id: int, label: str):
        """Live label change — refresh the current 2D view if on Rooms/Semantics tab."""
        idx = self.tabs.currentIndex()
        if idx in (self.TAB_ROOMS, self.TAB_SEMANTICS, self.TAB_OVERLAY):
            self._update_viewer()

    def _on_viewer_room_clicked(self, scene_x: float, scene_y: float):
        """Hit-test which room contains the clicked pixel and select it in the editor."""
        if not self._result:
            return
        # Only active on the Rooms / Semantics tabs
        idx = self.tabs.currentIndex()
        if idx not in (self.TAB_ROOMS, self.TAB_SEMANTICS, self.TAB_OVERLAY):
            return
        px, py = int(scene_x), int(scene_y)
        best_room = None
        best_area = float("inf")
        for room in self._result.building.rooms:
            rx, ry, rw, rh = room.bounding_box
            if rx <= px <= rx + rw and ry <= py <= ry + rh:
                if room.area_px < best_area:
                    best_area = room.area_px
                    best_room = room
        if best_room is not None:
            self._selected_room_id = best_room.room_id
            self.room_editor.select_room_by_id(best_room.room_id)
            self._update_viewer()

    def _on_room_selected_from_editor(self, room_id: int):
        """Room row clicked in editor — highlight it in the 2D view."""
        self._selected_room_id = room_id
        idx = self.tabs.currentIndex()
        if idx in (self.TAB_ROOMS, self.TAB_SEMANTICS, self.TAB_OVERLAY):
            self._update_viewer()

    def _update_furniture_btn_style(self):
        if self._include_furniture:
            self.btn_toggle_furniture.setText("🛋️  Furniture: Placed")
            self.btn_toggle_furniture.setStyleSheet(f"""
                QPushButton {{
                    background: qlineargradient(
                        x1:0, y1:0, x2:1, y2:0,
                        stop:0 #4f46e5, stop:1 #0891b2);
                    color: #ffffff;
                    border: 1px solid {C_ACCENT};
                    border-radius: 10px;
                    padding: 8px 12px;
                    font-size: 12px;
                    font-weight: 700;
                    text-align: center;
                }}
                QPushButton:hover {{
                    background: qlineargradient(
                        x1:0, y1:0, x2:1, y2:0,
                        stop:0 #6366f1, stop:1 #06b6d4);
                }}
                QPushButton:disabled {{
                    background: #141829;
                    color: {C_TEXT_DIM};
                    border: 1px solid {C_BORDER};
                }}
            """)
        else:
            self.btn_toggle_furniture.setText("🛋️  Furniture: OFF")
            self.btn_toggle_furniture.setStyleSheet(f"""
                QPushButton {{
                    background: {C_SURFACE2};
                    color: {C_TEXT_DIM};
                    border: 1px dashed {C_BORDER};
                    border-radius: 10px;
                    padding: 8px 12px;
                    font-size: 12px;
                    font-weight: 600;
                    text-align: center;
                }}
                QPushButton:hover {{
                    background: #1e2340;
                    color: {C_TEXT};
                    border-color: {C_ACCENT};
                }}
                QPushButton:disabled {{
                    background: #141829;
                    color: {C_TEXT_DIM};
                    border: 1px solid {C_BORDER};
                }}
            """)

    def _on_toggle_furniture(self):
        self._include_furniture = not self._include_furniture
        self._update_furniture_btn_style()
        self._rebuild_3d()
        state_txt = "placed" if self._include_furniture else "hidden"
        self.status_badge.set_state("success", f"Furniture {state_txt.title()}")
        self.progress_msg.setText(f"3D Model updated: furniture is now {state_txt}.")

    def _on_wall_width_changed(self, val: int):
        w_m = val / 100.0
        self._lbl_wall_w.setText(f"Wall Width: {w_m:.2f} m ({val} cm)")
        self._rebuild_3d()

    def _on_pick_wall_color(self):
        initial = QColor(self._custom_wall_color_hex or "#DFC752")
        color = QColorDialog.getColor(initial, self, "Select Custom Wall Paint Color")
        if color.isValid():
            self._set_wall_color(color.redF(), color.greenF(), color.blueF(), color.name().upper())

    def _set_wall_color(self, r: float, g: float, b: float, hex_code: str, name: str = ""):
        self._custom_wall_color_rgb = (r, g, b)
        self._custom_wall_color_hex = hex_code
        display_name = name if name else hex_code.upper()
        self.lbl_wall_color.setText(display_name)
        self.color_swatch.setStyleSheet(
            f"background: {hex_code}; border: 1px solid {C_BORDER}; border-radius: 4px;"
        )
        self._rebuild_3d()
        self.status_badge.set_state("success", "Wall Color Applied")
        self.progress_msg.setText(f"Wall paint color set to {display_name}.")

    def _on_reset_wall_color(self):
        self._custom_wall_color_rgb = None
        self._custom_wall_color_hex = None
        self.lbl_wall_color.setText("Default Gold")
        self.color_swatch.setStyleSheet(
            f"background: #DFC752; border: 1px solid {C_BORDER}; border-radius: 4px;"
        )
        self._rebuild_3d()
        self.status_badge.set_state("success", "Wall Color Reset")
        self.progress_msg.setText("Wall paint color restored to default architectural palette.")

    def _rebuild_3d(self, redecorate: bool = False):
        """Unified rebuild of 3D model and 2D views applying all current settings:
        floor height, ceiling, furniture placement, custom wall color, and wall width."""
        if not (self._result and self._result.building.walls):
            return

        bld = self._result.building

        if redecorate:
            from backend.engine.semantics.procedural_house_decorator import ProceduralHouseDecorator
            bld.furniture = ProceduralHouseDecorator.decorate_building(bld)

        h = self.slider_floor_h.value() / 10.0
        ceil = self.chk_ceiling.isChecked()
        wall_w_m = self.slider_wall_w.value() / 100.0

        from backend.engine.model3d.builder import Model3DBuilder
        builder = Model3DBuilder(
            floor_height_m=h,
            show_ceiling=ceil,
            include_furniture=self._include_furniture,
            wall_color=self._custom_wall_color_rgb,
            wall_width_m=wall_w_m,
        )
        bld.model_3d = builder.run(bld)

        if bld.model_3d:
            self.stat_verts.update_value(f"{bld.model_3d.vertex_count:,}")
            self.stat_tris.update_value(f"{bld.model_3d.triangle_count:,}")
            v3d = self._get_viewer3d()
            if v3d:
                v3d.load_model(bld.model_3d)

        # Update wall thickness in data structures proportionally for exports
        px_per_m = builder._resolve_scale(bld)
        new_thickness_px = wall_w_m * px_per_m
        for w in bld.walls:
            if w.wall_type == "outer":
                w.thickness_px = new_thickness_px
            else:
                w.thickness_px = max(4.0, new_thickness_px * 0.67)

        # Also update 2D Renderer
        wall_color_bgr = None
        if self._custom_wall_color_rgb is not None:
            r, g, b = self._custom_wall_color_rgb
            wall_color_bgr = (int(b * 255), int(g * 255), int(r * 255))

        wall_thickness_scale = wall_w_m / 0.30
        from backend.engine.visualization.renderer import Renderer
        self._result.renderer = Renderer(
            self._result.original_bgr,
            bld,
            wall_color_bgr=wall_color_bgr,
            wall_thickness_scale=wall_thickness_scale,
            include_furniture=self._include_furniture,
        )
        self._update_viewer()

    def _on_rebuild_3d(self):
        """Sync labels from editor and rebuild 3D."""
        if not self._result:
            return
        self.room_editor.sync_all_labels()
        self._rebuild_3d(redecorate=True)
        self.status_badge.set_state("success", "3D Rebuilt")
        self.progress_msg.setText("Labels applied. 3D model regenerated with corrected room types.")

    def _on_wireframe_toggled(self, on: bool):
        if self._viewer3d:
            self._viewer3d.set_wireframe(on)

    def _on_grid_toggled(self, on: bool):
        if self._viewer3d:
            self._viewer3d.set_grid(on)

    def _on_ceiling_toggled(self, on: bool):
        """Rebuild the 3D model with ceiling on or off."""
        self._rebuild_3d()

    def _on_reset_camera(self):
        if self._viewer3d:
            self._viewer3d.reset_view()

    def _on_floor_height_changed(self, val: int):
        h = val / 10.0
        self._lbl_floor_h.setText(f"Floor Height: {h:.1f} m")
        self._rebuild_3d()

    # ── View switching ────────────────────────────────────────────────────────

    def _on_tab_changed(self, index: int):
        is_3d    = index == self.TAB_3D
        is_debug = index == self.TAB_DEBUG

        self.debug_tabs.setVisible(is_debug)
        self.view_stack.setCurrentIndex(1 if is_3d else 0)

        if is_3d:
            # Ensure viewer3d is created and model loaded
            v3d = self._get_viewer3d()
            if v3d and self._result and self._result.building.model_3d:
                v3d.load_model(self._result.building.model_3d)
        else:
            self._update_viewer()
            if self._result:
                self.viewer.fitInView(
                    self.viewer.scene.sceneRect(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                )

    def _on_debug_tab_changed(self, _index: int):
        self._update_viewer()

    def _on_alpha_changed(self, _value: int):
        if self.tabs.currentIndex() in (self.TAB_OVERLAY, self.TAB_SEMANTICS):
            self._update_viewer()

    def _update_viewer(self):
        if not self._result:
            return

        idx = self.tabs.currentIndex()
        if idx == self.TAB_3D:
            return   # handled by 3D viewer directly

        r   = self._result.renderer
        img = None

        if idx == self.TAB_ORIGINAL:
            img = self._result.original_bgr.copy()
        elif idx == self.TAB_WALLS:
            img = r.render_walls()
        elif idx == self.TAB_ROOMS:
            img = r.render_rooms(self._result.debug_images.get("room_01_segmentation"))
        elif idx == self.TAB_OVERLAY:
            img = r.render_overlay(self.slider_alpha.value() / 100.0)
        elif idx == self.TAB_TOPOLOGY:
            img = r.render_topology()
        elif idx == self.TAB_SEMANTICS:
            img = r.render_semantics()
        elif idx == self.TAB_DEBUG:
            d = self.debug_tabs.currentIndex()
            if 0 <= d < len(self.debug_keys):
                img = self._result.debug_images.get(self.debug_keys[d])

        if img is not None:
            self.viewer.set_image(img)
            self.viewer.fitInView(
                self.viewer.scene.sceneRect(),
                Qt.AspectRatioMode.KeepAspectRatio,
            )
