"""
ui/loading_overlay.py
Blueprint-X — Full-window animated loading screen.

Covers the ENTIRE application window (sidebar + main area).
Parent must be the QMainWindow itself.
"""
from __future__ import annotations

import math

from PySide6.QtCore import Qt, QTimer, QRectF, QPointF, QRect
from PySide6.QtGui import QPainter, QColor, QPen, QLinearGradient, QFont, QBrush, QRadialGradient
from PySide6.QtWidgets import QWidget, QLabel


class _Spinner(QWidget):
    """Three-arc animated spinner."""
    def __init__(self, size=100, parent=None):
        super().__init__(parent)
        self.setFixedSize(size, size)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self._a1, self._a2, self._a3 = 0.0, 120.0, 240.0
        t = QTimer(self)
        t.timeout.connect(self._tick)
        t.start(14)

    def _tick(self):
        self._a1 = (self._a1 + 5.0)  % 360
        self._a2 = (self._a2 - 3.5)  % 360
        self._a3 = (self._a3 + 2.0)  % 360
        self.update()

    def paintEvent(self, _ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w = self.width()
        s = 7

        # Outer ring — indigo
        pen = QPen(QColor("#6366f1"), 5); pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        p.setPen(pen)
        p.drawArc(QRectF(s, s, w-2*s, w-2*s), int(self._a1*16), 260*16)

        # Middle ring — cyan
        s2 = s + 13
        pen2 = QPen(QColor("#22d3ee"), 4); pen2.setCapStyle(Qt.PenCapStyle.RoundCap)
        p.setPen(pen2)
        p.drawArc(QRectF(s2, s2, w-2*s2, w-2*s2), int(self._a2*16), 200*16)

        # Inner ring — gold
        s3 = s + 24
        pen3 = QPen(QColor("#fbbf24"), 3); pen3.setCapStyle(Qt.PenCapStyle.RoundCap)
        p.setPen(pen3)
        p.drawArc(QRectF(s3, s3, w-2*s3, w-2*s3), int(self._a3*16), 140*16)


class LoadingOverlay(QWidget):
    """
    Full-window translucent loading overlay.

    Must be parented to the QMainWindow, NOT to a child widget.
    Call resize_to_parent() whenever the parent resizes.
    """

    def __init__(self, main_window: QWidget):
        super().__init__(main_window)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, False)
        self._mw = main_window
        self._setup_ui()
        self.hide()

    def _setup_ui(self):
        # Spinner
        self._spinner = _Spinner(110, self)

        # Stage label
        self._lbl_stage = QLabel("Analyzing Blueprint...", self)
        self._lbl_stage.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._lbl_stage.setStyleSheet(
            "color: #e2e8f0; font-size: 22px; font-weight: 800;"
            " letter-spacing: -0.5px; background: transparent;"
        )
        self._lbl_stage.adjustSize()

        # Step label
        self._lbl_step = QLabel("Initializing...", self)
        self._lbl_step.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._lbl_step.setStyleSheet(
            "color: #6b7280; font-size: 13px; background: transparent;"
        )
        self._lbl_step.adjustSize()

        # Progress fill bar (manual widget, no QSS needed)
        self._prog_bg = QWidget(self)
        self._prog_bg.setFixedHeight(6)
        self._prog_bg.setStyleSheet(
            "background: #1e2740; border-radius: 3px;"
        )
        self._prog_fill = QWidget(self._prog_bg)
        self._prog_fill.setFixedHeight(6)
        self._prog_fill.setStyleSheet(
            "background: qlineargradient(x1:0,y1:0,x2:1,y2:0,"
            "stop:0 #6366f1, stop:1 #22d3ee); border-radius: 3px;"
        )
        self._prog_fill.setFixedWidth(0)

        # Hint
        self._lbl_hint = QLabel("This may take a moment — high-quality analysis in progress", self)
        self._lbl_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._lbl_hint.setStyleSheet("color: #374151; font-size: 11px; background: transparent;")
        self._lbl_hint.adjustSize()

    def paintEvent(self, _ev):
        """Draw dark translucent backdrop + glass card."""
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Full-window dark overlay
        p.fillRect(self.rect(), QColor(5, 7, 18, 230))

        # Glass card
        cw, ch = 460, 360
        cx = (self.width()  - cw) // 2
        cy = (self.height() - ch) // 2
        card = QRectF(cx, cy, cw, ch)

        p.setBrush(QColor(12, 16, 36, 245))
        p.setPen(QPen(QColor(45, 55, 100, 160), 1))
        p.drawRoundedRect(card, 24, 24)

        # Subtle gradient accent at top of card
        grad = QLinearGradient(cx, cy, cx+cw, cy)
        grad.setColorAt(0, QColor(99, 102, 241, 60))
        grad.setColorAt(0.5, QColor(34, 211, 238, 40))
        grad.setColorAt(1, QColor(99, 102, 241, 60))
        p.setBrush(QBrush(grad))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawRoundedRect(QRectF(cx, cy, cw, 4), 2, 2)

        self._card_rect = QRect(cx, cy, cw, ch)
        p.end()
        self._layout_children()

    def _layout_children(self):
        if not hasattr(self, '_card_rect'):
            return
        r = self._card_rect
        cx, cy, cw = r.x(), r.y(), r.width()

        # Spinner — top 1/3 of card
        sw = self._spinner.width()
        self._spinner.move(cx + (cw - sw)//2, cy + 30)

        # Stage label
        self._lbl_stage.setFixedWidth(cw - 40)
        self._lbl_stage.adjustSize()
        self._lbl_stage.setFixedWidth(cw - 40)
        self._lbl_stage.move(cx + 20, cy + 160)

        # Step label
        self._lbl_step.setFixedWidth(cw - 60)
        self._lbl_step.adjustSize()
        self._lbl_step.setFixedWidth(cw - 60)
        self._lbl_step.move(cx + 30, cy + 200)

        # Progress bar
        bar_w = cw - 80
        self._prog_bg.setFixedWidth(bar_w)
        self._prog_bg.move(cx + 40, cy + 250)

        # Hint
        self._lbl_hint.setFixedWidth(cw - 40)
        self._lbl_hint.move(cx + 20, cy + 300)

    def show_overlay(self, stage: str = "Analyzing Blueprint...", step: str = "Initializing..."):
        self.resize_to_parent()
        self._lbl_stage.setText(stage)
        self._lbl_step.setText(step)
        self._prog_fill.setFixedWidth(0)
        self.raise_()
        self.show()
        self.update()

    def hide_overlay(self):
        self.hide()

    def update_stage(self, stage: str):
        self._lbl_stage.setText(stage)
        self.update()

    def update_step(self, step: str):
        self._lbl_step.setText(step)
        self.update()

    def set_progress(self, fraction: float):
        w = max(6, int(self._prog_bg.width() * max(0., min(1., fraction))))
        self._prog_fill.setFixedWidth(w)

    def resize_to_parent(self):
        if self._mw:
            self.setGeometry(self._mw.rect())
