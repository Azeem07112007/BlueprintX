"""
ui/widgets.py
Blueprint-X — Premium UI Components

Deep-navy / electric-indigo / cyan design system.
All components use QSS gradients, rounded corners, and hover micro-animations.
"""
from __future__ import annotations

from PySide6.QtCore import Qt, QPropertyAnimation, QEasingCurve, QSize
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QPushButton, QLabel, QWidget, QHBoxLayout, QVBoxLayout, QFrame, QGraphicsOpacityEffect
)

# ── Colour tokens ─────────────────────────────────────────────────────────────
C_BG          = "#070914"      # deepest navy
C_SURFACE     = "#0e1120"      # card surface
C_SURFACE2    = "#141829"      # raised surface
C_BORDER      = "#1e2740"      # subtle border
C_ACCENT      = "#6366f1"      # electric indigo
C_ACCENT2     = "#22d3ee"      # vivid cyan
C_GOLD        = "#fbbf24"      # amber/gold
C_TEXT        = "#e2e8f0"      # warm white
C_TEXT_DIM    = "#64748b"      # slate grey
C_SUCCESS     = "#10b981"      # emerald
C_WARNING     = "#f59e0b"      # amber
C_ERROR       = "#ef4444"      # red
C_OUTER_WALL  = "#fbbf24"      # gold (wall colour alias)
C_INNER_WALL  = "#818cf8"      # soft indigo (wall colour alias)


# ── Sidebar Action Button ─────────────────────────────────────────────────────

class SidebarButton(QPushButton):
    """
    Premium sidebar action button.
    primary=True  → indigo→cyan gradient
    primary=False → dark surface with indigo glow on hover
    """

    def __init__(self, text: str, primary: bool = False, icon_char: str = "", compact: bool = False):
        label = f"{icon_char}  {text}" if icon_char else text
        super().__init__(label)

        if compact:
            self.setStyleSheet(f"""
                QPushButton {{
                    background: qlineargradient(
                        x1:0, y1:0, x2:0, y2:1,
                        stop:0 {C_SURFACE2}, stop:1 {C_SURFACE});
                    color: {C_TEXT};
                    border: 1px solid {C_BORDER};
                    border-radius: 8px;
                    padding: 6px 4px;
                    font-size: 11px;
                    font-weight: 600;
                    text-align: center;
                }}
                QPushButton:hover {{
                    background: qlineargradient(
                        x1:0, y1:0, x2:0, y2:1,
                        stop:0 #1e2340, stop:1 #161928);
                    border-color: {C_ACCENT};
                    color: #ffffff;
                }}
                QPushButton:pressed {{
                    background: {C_SURFACE};
                    border-color: {C_ACCENT2};
                }}
                QPushButton:disabled {{
                    color: {C_TEXT_DIM};
                    background: {C_SURFACE};
                    border-color: {C_BORDER};
                }}
            """)
        elif primary:
            self.setStyleSheet(f"""
                QPushButton {{
                    background: qlineargradient(
                        x1:0, y1:0, x2:1, y2:0,
                        stop:0 {C_ACCENT}, stop:1 {C_ACCENT2});
                    color: #ffffff;
                    border: none;
                    border-radius: 12px;
                    padding: 11px 16px;
                    font-size: 13px;
                    font-weight: 700;
                    text-align: left;
                    letter-spacing: 0.2px;
                }}
                QPushButton:hover {{
                    background: qlineargradient(
                        x1:0, y1:0, x2:1, y2:0,
                        stop:0 #7577f5, stop:1 #38ddf2);
                }}
                QPushButton:pressed {{
                    background: qlineargradient(
                        x1:0, y1:0, x2:1, y2:0,
                        stop:0 #4f52d6, stop:1 #17b5cc);
                }}
                QPushButton:disabled {{
                    background: #1a1d2e;
                    color: {C_TEXT_DIM};
                    border: 1px solid {C_BORDER};
                }}
            """)
        else:
            self.setStyleSheet(f"""
                QPushButton {{
                    background: qlineargradient(
                        x1:0, y1:0, x2:0, y2:1,
                        stop:0 {C_SURFACE2}, stop:1 {C_SURFACE});
                    color: {C_TEXT};
                    border: 1px solid {C_BORDER};
                    border-radius: 12px;
                    padding: 10px 16px;
                    font-size: 13px;
                    font-weight: 600;
                    text-align: left;
                }}
                QPushButton:hover {{
                    background: qlineargradient(
                        x1:0, y1:0, x2:0, y2:1,
                        stop:0 #1e2340, stop:1 #161928);
                    border-color: {C_ACCENT};
                    color: #ffffff;
                }}
                QPushButton:pressed {{
                    background: {C_SURFACE};
                    border-color: {C_ACCENT2};
                }}
                QPushButton:disabled {{
                    color: {C_TEXT_DIM};
                    background: {C_SURFACE};
                    border-color: {C_BORDER};
                }}
            """)

        self.setCursor(Qt.CursorShape.PointingHandCursor)
        if compact:
            self.setMinimumHeight(34)
        else:
            self.setMinimumHeight(46)


# ── Stat Row ──────────────────────────────────────────────────────────────────

class StatRow(QWidget):
    """Key → value display row with subtle accent on the value."""

    def __init__(self, key: str, value: str = "—"):
        super().__init__()
        self.setStyleSheet(f"""
            QWidget {{
                background: {C_SURFACE};
                border: 1px solid {C_BORDER};
                border-radius: 8px;
            }}
        """)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 6, 10, 6)

        self._key = QLabel(key)
        self._key.setStyleSheet(
            f"color: {C_TEXT_DIM}; font-size: 11px; font-weight: 500;"
            " background: transparent; border: none;"
        )

        self._val = QLabel(value)
        self._val.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._val.setStyleSheet(
            f"color: {C_ACCENT2}; font-size: 12px; font-weight: 700;"
            " background: transparent; border: none;"
        )

        lay.addWidget(self._key)
        lay.addStretch()
        lay.addWidget(self._val)

    def update_value(self, value: str):
        self._val.setText(value)


# ── Divider ───────────────────────────────────────────────────────────────────

class Divider(QFrame):
    def __init__(self):
        super().__init__()
        self.setFixedHeight(1)
        self.setStyleSheet(f"background: {C_BORDER}; border: none; margin: 8px 0;")


# ── Status Badge ──────────────────────────────────────────────────────────────

class StatusBadge(QLabel):
    """
    Pill-shaped status badge.
    States: idle | loading | success | error
    """
    _PALETTE = {
        "idle":    ("#64748b", "#0e1120", "◌", "Ready"),
        "loading": ("#f59e0b", "#1a1200", "⟳", "Processing…"),
        "success": ("#10b981", "#001a10", "✓", "Done"),
        "error":   ("#ef4444", "#1a0000", "✕", "Error"),
    }

    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.set_state("idle")

    def set_state(self, state: str, message: str = ""):
        fg, bg, icon, default_msg = self._PALETTE.get(state, self._PALETTE["idle"])
        text = message or default_msg
        self.setText(f"{icon}  {text}")
        self.setStyleSheet(f"""
            QLabel {{
                color: {fg};
                background: {bg};
                border: 1px solid {fg}40;
                border-radius: 12px;
                padding: 5px 14px;
                font-size: 11px;
                font-weight: 700;
                letter-spacing: 0.3px;
            }}
        """)


# ── Section Label ─────────────────────────────────────────────────────────────

class SectionLabel(QLabel):
    """Uppercase section header with subtle left accent bar."""

    def __init__(self, text: str):
        super().__init__(text)
        self.setStyleSheet(f"""
            QLabel {{
                color: {C_TEXT_DIM};
                font-size: 10px;
                font-weight: 800;
                letter-spacing: 2.5px;
                padding: 6px 0 4px 10px;
                border-left: 2px solid {C_ACCENT};
                background: transparent;
            }}
        """)
        self.setContentsMargins(0, 4, 0, 4)
