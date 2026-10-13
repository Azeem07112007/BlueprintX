# Blueprint-X — Milestone 2

> **Blueprint-to-3D Building Generation System**
> Milestone 2: Hybrid CV + AI Detection + Geometry Topology Pipeline

---

## Overview

Blueprint-X is a modular desktop application that processes architectural floor plan images
using a multi-stage computer vision and AI pipeline, then presents a seven-panel analysis view.

| Tab | Content |
|-----|---------|
| **Original** | Raw uploaded floor plan |
| **Walls** | Detected wall vectors (outer = cyan, internal = blue) + door gap markers |
| **Rooms** | Colour-coded room segmentation with labels |
| **Overlay** | Blueprint + translucent room fills + walls + labels |
| **Topology** | Abstract room adjacency graph |
| **Semantics** | Full semantic annotation: labels, doors, windows, adjacency + legend |
| **Debug** | Step-by-step intermediate pipeline images |

---

## Project Structure

```
BlueprintX/
├── main.py                          # Entry point (Qt6 + logging bootstrap)
├── pipeline.py                      # 9-stage orchestration pipeline
├── requirements.txt                 # Full dependency list
├── run.bat                          # Windows launcher (uses venv313)
├── smoke_test.py                    # Headless end-to-end sanity check
├── generate_sample_plan.py          # Generates a synthetic test floor plan
│
├── backend/
│   ├── models/
│   │   └── dataclasses.py           # Canonical data models (Wall, Room, Door, …)
│   └── engine/
│       ├── preprocessing/
│       │   └── preprocessor.py      # Deskew → CLAHE → adaptive threshold (dual pipeline)
│       ├── detection/
│       │   ├── wall_detector.py     # Skeletonize → Hough → merge → snap → thickness
│       │   ├── ai_detector.py       # YOLO11 segmentation (graceful geometric fallback)
│       │   └── ocr_engine.py        # EasyOCR text extraction + scale detection
│       ├── geometry/
│       │   └── geometry_engine.py   # Planar wall graph (DCEL-compatible)
│       ├── topology/
│       │   └── topology_engine.py   # Planar face extraction → room adjacency graph
│       ├── rooms/
│       │   └── room_detector.py     # Room polygons + door/window association
│       ├── semantics/
│       │   └── semantic_engine.py   # OCR label fusion + heuristic fallback labels
│       ├── visualization/
│       │   └── renderer.py          # All 6 UI view renderers
│       └── export/
│           └── exporter.py          # JSON export (schema v2.0)
│
├── ui/
│   ├── main_window.py               # Main Qt window + pipeline worker thread
│   ├── viewer.py                    # ZoomableImageViewer (scroll/pan/zoom)
│   └── widgets.py                   # Sidebar components + colour tokens
│
├── tests/
│   ├── test_pipeline.py             # Integration tests (full pipeline + export)
│   ├── test_geometry.py             # GeometryEngine unit tests
│   └── test_wall_detector.py        # WallDetector + Preprocessor unit tests
│
└── training/
    ├── blueprint_yolo.yaml          # YOLO11 training config
    ├── train.py                     # Training script
    └── weights/                     # Place best.pt here after training
```

---

## Quick Start & Running Guide

For a detailed step-by-step setup and demonstration guide, see [HOW_TO_RUN.md](file:///d:/BlueprintX/HOW_TO_RUN.md).

### 1. Installation & Running

```bat
:: Create & activate virtual environment (Windows)
python -m venv venv
venv\Scripts\activate

:: Install requirements
pip install -r requirements.txt

:: Launch application
run.bat
:: (or run manually: python main.py)
```

### 2. Generate a sample floor plan (optional)

```bat
python generate_sample_plan.py
```

### 3. Run the headless smoke test

```bat
python smoke_test.py
```

### 4. Run the test suite

```bat
python -m pytest tests/ -v
```

---

## Detection Pipeline (9 stages)

```
Image Load
    └─► 1. Preprocessing
    │       Grayscale → Deskew → CLAHE → Resize
    │       Auto-detect style: "simple" (line drawing) | "rendered" (colour plan)
    │       Dual threshold pipeline → structural binary mask
    │
    └─► 2. AI Detection (YOLO11)
    │       Doors / Windows / Furniture / Stairs / Columns
    │       Graceful fallback to geometric-only mode if no weights found
    │
    └─► 3. OCR Text Extraction (EasyOCR)
    │       Room labels / Dimension values / Scale notation
    │
    └─► 4. Wall Detection (OpenCV)
    │       Skeletonize → Hough LinesP → Merge collinear → Snap endpoints
    │       Detect door gaps → Estimate wall thickness
    │
    └─► 5. Geometry Graph
    │       Group segments by line → find intersections → build DCEL
    │
    └─► 6. Topology Engine
    │       Planar face extraction (leftmost-turn traversal)
    │       Identify exterior face → build room adjacency graph
    │
    └─► 7. Room Detection
    │       Room polygons from topology faces
    │       Associate doors/windows to rooms
    │       Re-classify walls as outer / internal
    │
    └─► 8. Semantic Engine
            Fuse OCR room labels → heuristic size-based fallback
```

---

## Exported JSON Schema (v2.0.0)

```jsonc
{
  "schema_version": "2.0.0",
  "milestone": 2,
  "metadata": {
    "source": "floorplan.png",
    "image_size": { "width": 900, "height": 700 },
    "scale": "1:100",
    "exported_at_utc": "2026-08-24T13:00:00+00:00"
  },
  "statistics": {
    "total_walls": 19, "outer_walls": 4, "internal_walls": 15,
    "total_rooms": 10, "total_doors": 8, "total_windows": 0
  },
  "walls":  [ { "id": "…", "start": [x, y], "end": [x, y], "wall_type": "outer",
                "length_px": 300, "thickness_px": 8.2, "angle_deg": 0, "confidence": 1.0 } ],
  "rooms":  [ { "room_id": 1, "label": "Living Space", "centroid": [180, 200],
                "area_px": 24500, "polygon_points": [[…]], "adjacent_room_ids": [2, 3] } ],
  "topology": { "nodes": [ … ], "edges": [ … ] },
  "floors":   [],     // Milestone 3
  "elevations": [],   // Milestone 3
  "model_3d": null    // Milestone 4
}
```

---

## Milestone Roadmap

| Milestone | Feature | Status |
|-----------|---------|--------|
| **1 ✅** | Wall/room detection, JSON export, 2D view | Done |
| **2 ✅** | YOLO door/window AI, OCR, topology graph, multi-view UI | **Done** |
| **3** | Multi-storey / elevation support | Planned |
| **4** | Unity 3D model generation from JSON | Planned |

---

## Tech Stack

- **Python 3.13** (`venv313`)
- **PySide6** — Qt6 desktop UI with zoomable/pannable viewer
- **OpenCV 5** — image processing, skeletonization, Hough transform
- **NumPy / SciPy** — numerical operations
- **NetworkX** — graph algorithms
- **Shapely** — polygon geometry
- **Ultralytics YOLO11** — AI object detection (optional: needs `best.pt`)
- **EasyOCR** — text extraction (optional)

---

## Tips for Best Results

- Use **clean, high-contrast** floor plan scans (dark walls on white/light background)
- Images at **600–1200 px** width work best; pipeline auto-resizes larger images
- The preprocessor **auto-detects** blueprint style (line drawing vs rendered colour plan)
- Without `training/weights/best.pt`, the AI stage runs in geometric-only mode — walls and rooms still work perfectly
