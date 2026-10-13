"""
pipeline.py
Blueprint-X — Milestone 4

Main orchestration pipeline. Connects UI to the modular engines.
10-stage pipeline from raw image to 3D model.
"""
from __future__ import annotations

import cv2
import time
from dataclasses import dataclass
from typing import Callable, Optional

from backend.models.dataclasses import Building
from backend.engine.preprocessing.preprocessor import Preprocessor
from backend.engine.detection.wall_detector import WallDetector
from backend.engine.detection.ai_detector import AIDetector
from backend.engine.detection.ocr_engine import OCREngine
from backend.engine.geometry.geometry_engine import GeometryEngine
from backend.engine.topology.topology_engine import TopologyEngine
from backend.engine.rooms.room_detector import RoomDetector
from backend.engine.semantics.semantic_engine import SemanticEngine
from backend.engine.detection.heuristic_furniture_detector import HeuristicFurnitureDetector
from backend.engine.detection.llm_vision_detector import LLMVisionDetector
from backend.engine.detection.window_synthesizer import WindowSynthesizer
from backend.engine.semantics.procedural_house_decorator import ProceduralHouseDecorator
from backend.engine.model3d.builder import Model3DBuilder
from backend.engine.visualization.renderer import Renderer

_TOTAL_STEPS = 10


@dataclass
class PipelineResult:
    """Carries the final canonical model and debug images to the UI."""
    image_path:   str
    original_bgr: cv2.typing.MatLike
    building:     Building
    debug_images: dict          # dict[str, MatLike]
    renderer:     Renderer


class BlueprintPipeline:
    def __init__(self):
        self.preprocessor    = Preprocessor()
        self.wall_detector   = WallDetector()
        self.ai_detector     = AIDetector()
        self.ocr_engine      = OCREngine()
        self.geometry_engine = GeometryEngine()
        self.topology_engine = TopologyEngine()
        self.room_detector   = RoomDetector()
        self.semantic_engine = SemanticEngine()
        self.furniture_detector = HeuristicFurnitureDetector()
        self.llm_vision      = LLMVisionDetector()
        self.model3d_builder = Model3DBuilder()

    def run(
        self,
        image_path: str,
        progress_cb: Optional[Callable[[int, int, str], None]] = None,
    ) -> PipelineResult:

        def _p(step: int, msg: str):
            if progress_cb:
                progress_cb(step, _TOTAL_STEPS, msg)

        # ── 0. Load image ──────────────────────────────────────────────────
        _p(1, "Loading image...")
        bgr = cv2.imread(image_path)
        if bgr is None:
            raise ValueError(f"Failed to load image: {image_path}")
        h, w = bgr.shape[:2]

        building = Building(source_path=image_path, image_size=(w, h))
        debug    = {}

        # ── 1. Preprocessing ───────────────────────────────────────────────
        _p(2, "Preprocessing (deskew, CLAHE, threshold)...")
        prep = self.preprocessor.run(bgr)
        debug.update(prep.debug)
        if prep.cleaned_binary is None:
            raise RuntimeError("Preprocessing failed to produce a binary mask.")

        # ── 2. AI Detection ────────────────────────────────────────────────
        _p(3, "AI Detection (doors, windows, furniture)...")
        ai = self.ai_detector.run(bgr)
        building.doors.extend(ai.doors)
        building.windows.extend(ai.windows)
        building.furniture.extend(ai.furniture)
        building.columns.extend(ai.columns)
        building.stairs.extend(ai.stairs)
        debug.update(ai.debug)

        # ── 3. OCR ─────────────────────────────────────────────────────────
        _p(4, "OCR text extraction...")
        ocr = self.ocr_engine.run(bgr)
        building.text_blocks = ocr.text_blocks
        building.scale       = ocr.scale
        debug.update(ocr.debug)

        # ── 4. Wall Detection ──────────────────────────────────────────────
        _p(5, "Wall detection (Hough + skeletonize)...")
        wd = self.wall_detector.run(
            prep.cleaned_binary, bgr,
            blueprint_style=prep.blueprint_style,
        )
        building.walls = wd.walls
        building.doors.extend(wd.door_gaps)
        debug.update(wd.debug)

        # ── 5. Geometry Graph ──────────────────────────────────────────────
        _p(6, "Geometry graph construction...")
        graph = self.geometry_engine.run(building.walls, building.doors, (h, w))

        # ── 6. Topology ────────────────────────────────────────────────────
        _p(7, "Topology extraction (planar faces)...")
        topo = self.topology_engine.run(graph, building.doors)
        building.topology_nodes = topo.nodes
        building.topology_edges = topo.edges

        # ── 7. Room Detection ──────────────────────────────────────────────
        _p(8, "Room polygon generation...")
        # Use the dedicated room-segmentation mask (preserves thin internal walls).
        # Fall back to cleaned_binary if room_seg_mask is unavailable.
        _room_mask = prep.room_seg_mask if prep.room_seg_mask is not None else prep.cleaned_binary
        rooms = self.room_detector.run(
            graph=graph, topology=topo,
            walls=building.walls, doors=building.doors,
            windows=building.windows, image_size=(h, w),
            wall_mask=_room_mask,
        )
        building.rooms = rooms.rooms
        debug["room_01_segmentation"] = rooms.segmentation

        # ── 8. Semantic Fusion ─────────────────────────────────────────────────
        _p(9, "Semantic label fusion...")
        # Resolve scale for real-world area computation
        _px_per_m: float = 100.0
        try:
            s = (building.scale or "").strip()
            if ":" in s:
                d = float(s.split(":")[1])
                if d > 0:
                    _px_per_m = d
        except (ValueError, IndexError):
            pass
        self.semantic_engine.run(building.rooms, building.text_blocks, px_per_m=_px_per_m)

        # Synthesize exterior windows for rooms facing outside
        building.windows = WindowSynthesizer.synthesize_windows(building, px_per_m=_px_per_m)

        # Run CV heuristic furniture extraction and merge with AI detections
        h_furniture, h_stairs = self.furniture_detector.run(
            bgr=bgr,
            wall_mask=prep.cleaned_binary,
            rooms=building.rooms,
            text_blocks=building.text_blocks,
            doors=building.doors,
            windows=building.windows,
        )
        building.furniture = HeuristicFurnitureDetector.merge_detections(building.furniture, h_furniture)
        building.stairs.extend(h_stairs)

        # Procedurally furnish/decorate house rooms based on semantic labels
        building.furniture = ProceduralHouseDecorator.decorate_building(building)

        # ── 9. 3D Model Generation ─────────────────────────────────────────
        _p(10, "3D model generation...")
        building.model_3d = self.model3d_builder.run(building)

        renderer = Renderer(bgr, building)

        return PipelineResult(
            image_path   = image_path,
            original_bgr = bgr,
            building     = building,
            debug_images = debug,
            renderer     = renderer,
        )
