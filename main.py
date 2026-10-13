"""
main.py
Blueprint-X — Milestone 2

Application entry point. Bootstraps Qt, applies high-DPI settings,
and launches the main window.

Usage:
    python main.py
    (or) run.bat     # on Windows

Requires: venv313  (Python 3.13 + OpenCV + PySide6 + NetworkX + Shapely)
"""

import sys
import os
import logging

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s  %(name)s  %(message)s",
)

# Must be set BEFORE creating QApplication on some platforms
os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")
os.environ.setdefault("QT_SCALE_FACTOR_ROUNDING_POLICY", "PassThrough")

# IMPORTANT: Import cv2 and numpy BEFORE PySide6 to avoid DLL load conflicts on Windows
import numpy  # noqa: E402
import cv2    # noqa: E402


def main():
    from PySide6.QtWidgets import QApplication
    from PySide6.QtGui import QIcon

    from ui.main_window import BlueprintXWindow

    app = QApplication(sys.argv)
    app.setApplicationName("Blueprint-X")
    app.setApplicationVersion("2.0.0")
    app.setOrganizationName("Blueprint-X")

    # High-DPI scaling is enabled by default in Qt6
    window = BlueprintXWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
