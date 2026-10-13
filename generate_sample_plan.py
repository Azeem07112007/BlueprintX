"""
generate_sample_plan.py
Blueprint-X Prototype - Milestone 1

Generates a synthetic residential floor plan PNG so you can test
the application without needing a real blueprint image.

Usage:
    python generate_sample_plan.py
Produces:
    sample_floorplan.png  (in the current directory)
"""

import cv2
import numpy as np
from pathlib import Path


def draw_thick_line(img, p1, p2, color=(30, 30, 30), thickness=8):
    cv2.line(img, p1, p2, color, thickness, cv2.LINE_AA)


def draw_dashed_line(img, p1, p2, color=(80, 80, 80), thickness=2, dash=12, gap=8):
    """Draw a dashed line between two points."""
    x1, y1 = p1
    x2, y2 = p2
    length = np.hypot(x2 - x1, y2 - y1)
    if length == 0:
        return
    dx = (x2 - x1) / length
    dy = (y2 - y1) / length
    pos = 0.0
    drawing = True
    while pos < length:
        seg_len = dash if drawing else gap
        end_pos = min(pos + seg_len, length)
        if drawing:
            sx = int(x1 + dx * pos)
            sy = int(y1 + dy * pos)
            ex = int(x1 + dx * end_pos)
            ey = int(y1 + dy * end_pos)
            cv2.line(img, (sx, sy), (ex, ey), color, thickness, cv2.LINE_AA)
        pos = end_pos
        drawing = not drawing


def generate_floor_plan(
    output_path: str = "sample_floorplan.png",
    width: int = 900,
    height: int = 700,
):
    # White background
    img = np.full((height, width, 3), 248, dtype=np.uint8)

    W = 8    # outer wall thickness
    IW = 5   # inner wall thickness

    # ------------------------------------------------------------------ #
    #  Building outline  (outer walls)
    #  Top-left: (60, 60)   Bottom-right: (840, 640)
    # ------------------------------------------------------------------ #
    bx1, by1, bx2, by2 = 60, 60, 840, 640

    # Outer perimeter
    pts = [(bx1, by1), (bx2, by1), (bx2, by2), (bx1, by2)]
    for i in range(len(pts)):
        draw_thick_line(img, pts[i], pts[(i + 1) % len(pts)], thickness=W)

    # ------------------------------------------------------------------ #
    #  Internal horizontal walls
    # ------------------------------------------------------------------ #
    # Floor separator (mid-height)
    draw_thick_line(img, (bx1, 340), (bx2, 340), thickness=IW)

    # Upper-floor horizontal partition
    draw_thick_line(img, (bx1, 200), (480, 200), thickness=IW)

    # Lower-floor horizontal partition
    draw_thick_line(img, (bx1, 490), (560, 490), thickness=IW)

    # ------------------------------------------------------------------ #
    #  Internal vertical walls
    # ------------------------------------------------------------------ #
    # Left cluster vertical
    draw_thick_line(img, (300, by1), (300, 340), thickness=IW)
    draw_thick_line(img, (480, by1), (480, 200), thickness=IW)

    # Right cluster vertical
    draw_thick_line(img, (560, 340), (560, by2), thickness=IW)
    draw_thick_line(img, (700, 340), (700, by2), thickness=IW)

    # Centre divider
    draw_thick_line(img, (300, 340), (300, by2), thickness=IW)

    # ------------------------------------------------------------------ #
    #  Door openings (small gaps – white rectangles to break walls)
    # ------------------------------------------------------------------ #
    door_color = (248, 248, 248)

    # Doors in horizontal walls
    cv2.rectangle(img, (155, 337), (215, 343), door_color, -1)   # room 1 door
    cv2.rectangle(img, (355, 337), (415, 343), door_color, -1)   # room 3 door
    cv2.rectangle(img, (620, 337), (680, 343), door_color, -1)   # right corridor door
    cv2.rectangle(img, (155, 487), (215, 493), door_color, -1)   # lower room door
    cv2.rectangle(img, (630, 487), (680, 493), door_color, -1)   # lower right door

    # Doors in vertical walls
    cv2.rectangle(img, (297, 130), (303, 185), door_color, -1)
    cv2.rectangle(img, (297, 420), (303, 470), door_color, -1)
    cv2.rectangle(img, (557, 430), (563, 480), door_color, -1)
    cv2.rectangle(img, (697, 420), (703, 470), door_color, -1)

    # ------------------------------------------------------------------ #
    #  Window markers (dashed lines on outer wall)
    # ------------------------------------------------------------------ #
    draw_dashed_line(img, (130, by1), (260, by1), color=(120, 120, 120), thickness=3)
    draw_dashed_line(img, (380, by1), (470, by1), color=(120, 120, 120), thickness=3)
    draw_dashed_line(img, (600, by1), (780, by1), color=(120, 120, 120), thickness=3)
    draw_dashed_line(img, (bx2, 120), (bx2, 240), color=(120, 120, 120), thickness=3)
    draw_dashed_line(img, (bx2, 400), (bx2, 580), color=(120, 120, 120), thickness=3)
    draw_dashed_line(img, (130, by2), (270, by2), color=(120, 120, 120), thickness=3)
    draw_dashed_line(img, (420, by2), (540, by2), color=(120, 120, 120), thickness=3)

    # ------------------------------------------------------------------ #
    #  Room labels
    # ------------------------------------------------------------------ #
    font = cv2.FONT_HERSHEY_SIMPLEX
    label_color = (100, 100, 100)

    labels = [
        ("Bedroom 1",     (85,  100)),
        ("Bedroom 2",     (310, 100)),
        ("Living Room",   (500, 140)),
        ("Kitchen",       (340, 260)),
        ("Bathroom",      (85,  260)),
        ("Master Bed",    (85,  390)),
        ("Study",         (310, 390)),
        ("Dining",        (580, 410)),
        ("Garage",        (720, 410)),
        ("Storage",       (620, 560)),
    ]

    for text, (tx, ty) in labels:
        # Shadow
        cv2.putText(img, text, (tx + 1, ty + 1), font, 0.42,
                    (210, 210, 210), 1, cv2.LINE_AA)
        # Label
        cv2.putText(img, text, (tx, ty), font, 0.42,
                    label_color, 1, cv2.LINE_AA)

    # ------------------------------------------------------------------ #
    #  Title block
    # ------------------------------------------------------------------ #
    cv2.putText(img, "SAMPLE RESIDENTIAL FLOOR PLAN",
                (220, 680), font, 0.55, (60, 60, 60), 1, cv2.LINE_AA)
    cv2.putText(img, "Scale: 1:100  |  Blueprint-X Prototype  |  Milestone 1",
                (210, 695), font, 0.38, (150, 150, 150), 1, cv2.LINE_AA)

    # Border around title
    cv2.rectangle(img, (0, 0), (width - 1, height - 1), (180, 180, 180), 2)

    cv2.imwrite(output_path, img)
    print(f"[OK] Sample floor plan saved -> {Path(output_path).resolve()}")
    return output_path


if __name__ == "__main__":
    generate_floor_plan("sample_floorplan.png")
