"""
modules/structural_pipeline.py
Blueprint-X Prototype - Milestone 1.5

Orchestrates the multi-layer architectural understanding pipeline.
"""

from dataclasses import dataclass
from typing import Any, List
import numpy as np

from modules.image_loader import FloorPlanImage
from modules.wall_detector import detect_walls, WallDetectionResult
from modules.room_detector import detect_rooms, RoomDetectionResult
from modules.door_window_detector import detect_doors_and_windows, DoorWindowResult
from modules.furniture_detector import detect_furniture, FurnitureItem

@dataclass
class BlueprintAnalysis:
    floor_plan: FloorPlanImage
    walls: WallDetectionResult
    rooms: RoomDetectionResult
    doors_windows: DoorWindowResult
    furniture: List[FurnitureItem]
    
    @property
    def estimated_floor_area_px2(self) -> float:
        return sum(r.area_px for r in self.rooms.rooms)
        
    @property
    def reconstruction_confidence(self) -> float:
        """
        Confidence heuristic based on detected elements vs image properties.
        """
        walls_detected = self.walls.total_wall_segments
        rooms_detected = self.rooms.estimated_room_count
        wall_total_length = sum(w.length() for w in self.walls.walls)
        
        h, w = self.floor_plan.height, self.floor_plan.width
        image_perimeter = 2 * (h + w)
        
        # Calculate structural noise ratio
        if self.walls.binary_mask is not None and self.walls.structural_mask is not None:
            full_px = np.count_nonzero(self.walls.binary_mask)
            struct_px = np.count_nonzero(self.walls.structural_mask)
            noise_ratio = (full_px - struct_px) / max(1, full_px)
        else:
            noise_ratio = 0.5
            
        conf = (
            0.3 * (1.0 if walls_detected > 4 else 0.0) +
            0.3 * (1.0 if rooms_detected > 0 else 0.0) +
            0.2 * min(1.0, wall_total_length / max(1, image_perimeter)) +
            0.2 * (1.0 - noise_ratio)
        )
        return max(0.0, min(1.0, conf))

def run_pipeline(floor_plan: FloorPlanImage) -> BlueprintAnalysis:
    """
    Executes the full multi-layer pipeline.
    """
    # 1. Structural Layer: Walls (now internally filters thickness)
    wall_result = detect_walls(floor_plan.original_bgr)
    
    # 2. Structural Layer: Rooms (now uses cleaned structural mask)
    room_result = detect_rooms(
        wall_result.structural_mask if wall_result.structural_mask is not None else wall_result.binary_mask, 
        floor_plan.original_bgr
    )
    
    # 3. Structural Layer: Doors, Windows, Stairs (Topological)
    if wall_result.structural_mask is not None:
        dw_result = detect_doors_and_windows(
            wall_result.structural_mask, 
            floor_plan.original_bgr,
            room_result.adjacency_graph.edges
        )
    else:
        dw_result = DoorWindowResult([], [], [])
    
    # 4. Object Layer: Furniture
    if wall_result.binary_mask is not None and wall_result.structural_mask is not None:
        furniture = detect_furniture(wall_result.binary_mask, wall_result.structural_mask)
    else:
        furniture = []
        
    return BlueprintAnalysis(
        floor_plan=floor_plan,
        walls=wall_result,
        rooms=room_result,
        doors_windows=dw_result,
        furniture=furniture
    )
