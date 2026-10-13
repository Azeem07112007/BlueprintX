"""
ui/room_editor.py
Blueprint-X — Room Label Editor

Interactive panel that lets the user correct any room label detected
by the AI pipeline. Features:
  • Scrollable list of all detected rooms, each with:
      - Room colour swatch matching the floor plan
      - Area badge (m²)
      - Label dropdown (common room types) + free-text override field
  • "Selected" highlight when a room row is active
  • "Apply & Rebuild 3D" button triggers 3D model regeneration
  • Signals: label_changed(room_id, new_label), rebuild_requested()
"""
from __future__ import annotations

from typing import List, Optional

from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtGui import QColor, QPainter, QBrush
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QScrollArea,
    QFrame, QComboBox, QLineEdit, QPushButton, QSizePolicy,
)

from backend.models.dataclasses import Room
from ui.widgets import (
    C_BG, C_SURFACE, C_SURFACE2, C_BORDER, C_TEXT, C_TEXT_DIM,
    C_ACCENT, C_ACCENT2,
)

# ── Common room label options ─────────────────────────────────────────────────

ROOM_LABELS = [
    "Bedroom", "Master Bedroom", "Bedroom 1", "Bedroom 2", "Bedroom 3",
    "Living Room", "Living Space", "Lounge", "Family Room",
    "Kitchen", "Dining Room", "Open Plan",
    "Bathroom", "Master Bathroom", "En Suite", "WC / Toilet",
    "Study", "Office", "Work Room",
    "Hallway", "Corridor", "Entrance",
    "Balcony", "Terrace", "Patio",
    "Garage", "Storage", "Utility Room", "Laundry",
    "Staircase", "Unknown",
]


# ── Colour swatch ─────────────────────────────────────────────────────────────

class _ColourSwatch(QWidget):
    """Small rounded rectangle filled with the room's detection colour."""

    def __init__(self, bgr: tuple, parent=None):
        super().__init__(parent)
        b, g, r = bgr
        self._color = QColor(r, g, b)
        self.setFixedSize(QSize(14, 14))

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(self._color))
        p.drawRoundedRect(self.rect(), 3, 3)


# ── Single room row ───────────────────────────────────────────────────────────

class RoomRow(QFrame):
    """
    One row per room. Contains:
      colour swatch · R{id} · area badge · label combo · custom text entry
    """
    label_changed = Signal(int, str)   # room_id, new_label

    def __init__(self, room: Room, parent=None):
        super().__init__(parent)
        self._room = room
        self._selected = False

        self.setObjectName("RoomRow")
        self._apply_style(selected=False)
        self.setMinimumHeight(60)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 8, 10, 8)
        outer.setSpacing(6)

        # ── Top line: swatch · id · area ─────────────────────────────────────
        top = QHBoxLayout()
        top.setSpacing(6)

        swatch = _ColourSwatch(room.color_bgr)
        top.addWidget(swatch)

        id_lbl = QLabel(f"R{room.room_id:02d}")
        id_lbl.setStyleSheet(
            f"color: {C_ACCENT2}; font-size: 11px; font-weight: 800;"
            " background: transparent; border: none;"
        )
        top.addWidget(id_lbl)

        area_txt = f"{room.area_m2:.1f} m\u00b2" if room.area_m2 > 0 else f"{room.area_px:.0f} px\u00b2"
        area_lbl = QLabel(area_txt)
        area_lbl.setStyleSheet(
            f"color: {C_TEXT_DIM}; font-size: 10px;"
            " background: transparent; border: none;"
        )
        top.addWidget(area_lbl)
        top.addStretch()
        outer.addLayout(top)

        # ── Bottom line: dropdown · free-text ────────────────────────────────
        bot = QHBoxLayout()
        bot.setSpacing(6)

        self._combo = QComboBox()
        self._combo.setEditable(False)
        self._combo.addItems(ROOM_LABELS)
        # Pre-select matching label
        current = room.label or "Unknown"
        idx = self._combo.findText(current, Qt.MatchFlag.MatchFixedString)
        if idx >= 0:
            self._combo.setCurrentIndex(idx)
        else:
            self._combo.setCurrentIndex(self._combo.count() - 1)  # "Unknown"
        self._combo.setStyleSheet(self._combo_style())
        self._combo.currentTextChanged.connect(self._on_combo_changed)
        bot.addWidget(self._combo, 2)

        self._edit = QLineEdit()
        self._edit.setPlaceholderText("Override…")
        self._edit.setText(current if idx < 0 else "")
        self._edit.setStyleSheet(self._edit_style())
        self._edit.textChanged.connect(self._on_edit_changed)
        bot.addWidget(self._edit, 1)

        outer.addLayout(bot)

        self.setCursor(Qt.CursorShape.PointingHandCursor)

    # ── Style helpers ─────────────────────────────────────────────────────────

    def _apply_style(self, selected: bool):
        border = C_ACCENT if selected else C_BORDER
        bg = C_SURFACE2 if selected else C_SURFACE
        self.setStyleSheet(f"""
            QFrame#RoomRow {{
                background: {bg};
                border: 1px solid {border};
                border-radius: 10px;
            }}
        """)

    def _combo_style(self):
        return f"""
            QComboBox {{
                background: {C_SURFACE2};
                color: {C_TEXT};
                border: 1px solid {C_BORDER};
                border-radius: 6px;
                padding: 4px 8px;
                font-size: 11px;
                font-weight: 600;
            }}
            QComboBox::drop-down {{ border: none; width: 18px; }}
            QComboBox::down-arrow {{ image: none; }}
            QComboBox QAbstractItemView {{
                background: {C_SURFACE2};
                color: {C_TEXT};
                selection-background-color: {C_ACCENT};
                border: 1px solid {C_BORDER};
            }}
        """

    def _edit_style(self):
        return f"""
            QLineEdit {{
                background: {C_SURFACE2};
                color: {C_TEXT};
                border: 1px solid {C_BORDER};
                border-radius: 6px;
                padding: 4px 8px;
                font-size: 11px;
            }}
            QLineEdit:focus {{ border-color: {C_ACCENT2}; }}
        """

    # ── Public API ────────────────────────────────────────────────────────────

    def set_selected(self, selected: bool):
        self._selected = selected
        self._apply_style(selected)

    @property
    def room_id(self) -> int:
        return self._room.room_id

    def current_label(self) -> str:
        override = self._edit.text().strip()
        return override if override else self._combo.currentText()

    def sync_to_room(self):
        """Write the current label back to the Room object."""
        self._room.label = self.current_label()

    # ── Slots ─────────────────────────────────────────────────────────────────

    def _on_combo_changed(self, text: str):
        if not self._edit.text().strip():
            self._room.label = text
            self.label_changed.emit(self._room.room_id, text)

    def _on_edit_changed(self, text: str):
        lbl = text.strip() if text.strip() else self._combo.currentText()
        self._room.label = lbl
        self.label_changed.emit(self._room.room_id, lbl)

    # ── Mouse ─────────────────────────────────────────────────────────────────

    def mousePressEvent(self, event):
        """Row click selects the room (parent panel handles deselecting others)."""
        if event.button() == Qt.MouseButton.LeftButton:
            self._select_self()
        super().mousePressEvent(event)

    def _select_self(self):
        parent = self.parent()
        if parent and hasattr(parent, "_select_row"):
            parent._select_row(self)


# ── Main Editor Panel ─────────────────────────────────────────────────────────

class RoomEditorPanel(QWidget):
    """
    Panel embedded in the sidebar.
    Lists all rooms, lets the user edit labels, and emits rebuild_requested.
    """
    rebuild_requested = Signal()
    room_selected     = Signal(int)    # room_id
    label_changed     = Signal(int, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._rows: List[RoomRow] = []
        self._selected_row: Optional[RoomRow] = None

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(6)

        # ── Scroll area containing room rows ─────────────────────────────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        self._list_widget = QWidget()
        self._list_layout = QVBoxLayout(self._list_widget)
        self._list_layout.setContentsMargins(0, 0, 4, 0)
        self._list_layout.setSpacing(6)
        self._list_layout.addStretch()

        scroll.setWidget(self._list_widget)
        scroll.setMinimumHeight(120)
        scroll.setMaximumHeight(380)
        root.addWidget(scroll)

        # ── Rebuild button ────────────────────────────────────────────────────
        self._btn_rebuild = QPushButton("\u2726  Apply Labels & Rebuild 3D")
        self._btn_rebuild.setEnabled(False)
        self._btn_rebuild.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_rebuild.setMinimumHeight(42)
        self._btn_rebuild.setStyleSheet(f"""
            QPushButton {{
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #059669, stop:1 {C_ACCENT2});
                color: #ffffff;
                border: none;
                border-radius: 10px;
                padding: 10px 14px;
                font-size: 12px;
                font-weight: 800;
                text-align: left;
            }}
            QPushButton:hover {{
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #10b981, stop:1 #38ddf2);
            }}
            QPushButton:pressed {{
                background: #059669;
            }}
            QPushButton:disabled {{
                background: #1a1d2e;
                color: {C_TEXT_DIM};
                border: 1px solid {C_BORDER};
            }}
        """)
        self._btn_rebuild.clicked.connect(self.rebuild_requested.emit)
        root.addWidget(self._btn_rebuild)

        # hint label
        self._hint = QLabel("Run analysis to see rooms here.")
        self._hint.setWordWrap(True)
        self._hint.setStyleSheet(
            f"color: {C_TEXT_DIM}; font-size: 10px; background: transparent;"
        )
        root.addWidget(self._hint)

    # ── Public API ────────────────────────────────────────────────────────────

    def load_rooms(self, rooms: List[Room]):
        """Populate the list from a fresh analysis result."""
        self._rows.clear()
        self._selected_row = None

        # Clear existing room rows (keep final stretch)
        while self._list_layout.count() > 1:
            item = self._list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        for room in sorted(rooms, key=lambda r: r.room_id):
            row = RoomRow(room, parent=self._list_widget)
            row.label_changed.connect(self.label_changed.emit)
            self._list_layout.insertWidget(self._list_layout.count() - 1, row)
            self._rows.append(row)

        self._btn_rebuild.setEnabled(bool(rooms))
        n = len(rooms)
        self._hint.setText(
            f"{n} room{'s' if n != 1 else ''} detected. "
            "Click any row or click the floor-plan to select & rename."
        )

    def select_room_by_id(self, room_id: int):
        """Highlight the row matching room_id (called from main window)."""
        for row in self._rows:
            if row.room_id == room_id:
                self._select_row(row)
                return

    def sync_all_labels(self):
        """Write all current UI labels back into the Room objects."""
        for row in self._rows:
            row.sync_to_room()

    # ── Internal ─────────────────────────────────────────────────────────────

    def _select_row(self, row: RoomRow):
        if self._selected_row and self._selected_row is not row:
            self._selected_row.set_selected(False)
        row.set_selected(True)
        self._selected_row = row
        self.room_selected.emit(row.room_id)
