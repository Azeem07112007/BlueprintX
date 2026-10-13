# Blueprint-X — Setup & Execution Guide

This guide provides simple step-by-step instructions to set up, run, and demonstrate the **Blueprint-X** floor-plan-to-3D pipeline on any PC.

---

## 🚀 Quick Start (Windows)

### Prerequisites
- **Python 3.10, 3.11, 3.12, or 3.13** installed.
- *Important during Python installation:* Make sure to check the box **"Add python.exe to PATH"**.

---

### Step 1: Open Terminal / Command Prompt in Project Folder
1. Open the project folder (`BlueprintX`).
2. Click on the address bar at the top of File Explorer, type `cmd`, and press **Enter**.

---

### Step 2: Create Environment & Install Dependencies

Run the following commands in the terminal window:

```cmd
:: 1. Create a virtual environment
python -m venv venv

:: 2. Activate the virtual environment
venv\Scripts\activate

:: 3. Install required packages
pip install -r requirements.txt
```

*(Note: First-time dependency installation may take 1–2 minutes as PySide6, OpenCV, and PyTorch packages download).*

---

### Step 3: Run the Application

You can launch the application in two ways:

#### Option A: Double-Click Launcher (Easiest)
Simply double-click the `run.bat` file in the project folder!

#### Option B: Terminal Command
```cmd
python main.py
```

---

## 🍏 Quick Start (macOS / Linux)

```bash
# 1. Create virtual environment
python3 -m venv venv

# 2. Activate environment
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch app
python3 main.py
```

---

## 🎯 How to Demonstrate Project Progress ("Showcase Guide")

Follow this step-by-step flow to give a smooth demonstration of the project:

### 1. Load a Sample Floor Plan
- Once the app opens, click **"Load Blueprint"** on the left panel.
- Select `sample_floorplan.png` (located directly in the project folder).

### 2. Run the AI & Computer Vision Pipeline
- Click the **"Process Blueprint"** or **"Run Full Pipeline"** button on the left sidebar.
- The status bar will process the 9-stage pipeline (Deskewing → Wall Detection → OCR → Room Segmentation → 3D Model Building).

### 3. Tour the View Panels (Showcase Tabs)
Switch through the top tabs to demonstrate each stage of the analysis:

1. 🖼️ **Original**: Shows the raw uploaded floor plan image.
2. 🧱 **Walls**: Displays outer walls (cyan), internal partition walls (blue), and detected door gap markers.
3. 🚪 **Rooms**: Shows extracted room polygons with color-coded fills and computed floor areas.
4. 🎨 **Overlay**: Superimposes wall outlines and translucent room fills onto the original floor plan.
5. 🕸️ **Topology**: Displays the architectural room-adjacency node graph.
6. 🏷️ **Semantics**: Displays extracted text/room labels, associated doors/windows, and semantic legend.
7. 🧊 **3D Model**: Shows the generated interactive 3D building with walls, doors, windows, and room spaces. You can **click & drag to rotate**, **scroll to zoom**, and **right-click to pan**!

### 4. Demonstrate Export Capabilities
- Click **"Export JSON"** to export structured architectural geometry data (`schema v2.0`).
- Click **"Export 3D Model"** to save standard `.obj` or `.glb` files ready for Unity, Blender, or 3D web viewers.

---

## 🧪 Optional: Run Automated Sanity & Unit Tests

To verify that all core modules (Geometry, Room Segmentation, Wall Detection, 3D Mesh Generation) are working cleanly on your system:

```cmd
python -m pytest tests/ -v
```

All **103 unit & integration tests** should pass.

---

## ❓ Troubleshooting

- **Error: `'python' is not recognized as an internal or external command`**
  - Fix: Python is not added to system PATH. Re-run the Python installer and check **"Add python.exe to PATH"**, or specify the full path to `python.exe`.
- **Display / Graphics Warning in 3D Tab:**
  - The 3D viewer uses OpenGL. Ensure your graphics drivers are up to date.
