"""
modules/overlay_renderer.py
Blueprint-X Prototype - Milestone 1.5

Renders a multi-layer overlay comparison view.
"""

import cv2
import numpy as np
from typing import Any

def render_overlay(original_bgr: np.ndarray, analysis: Any, alpha: float) -> np.ndarray:
    """
    Overlay reconstructed elements on top of the original blueprint.
    
    Args:
        original_bgr: The original floor plan image
        analysis: BlueprintAnalysis object containing all detected elements
        alpha: Transparency factor (0.0 to 1.0). 0=original only, 1=reconstruction only
    """
    overlay = original_bgr.copy()
    
    # Render furniture (purple, semi-transparent fill)
    furniture_overlay = overlay.copy()
    for f in analysis.furniture:
        x, y, w, h = f.bounding_box
        cv2.rectangle(furniture_overlay, (x, y), (x+w, y+h), (180, 50, 180), -1)
        # Add label
        cv2.putText(furniture_overlay, f.label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 50, 180), 1)
    
    # Render rooms (faint tinted fill)
    for room in analysis.rooms.rooms:
        color = (200, 200, 200) # Simple gray tint for overlay
        cv2.drawContours(furniture_overlay, [room.polygon], -1, color, cv2.FILLED)
        
    # Apply alpha blend for fills
    cv2.addWeighted(furniture_overlay, alpha * 0.5, overlay, 1 - (alpha * 0.5), 0, overlay)

    # Render walls (gold / orange stroke)
    for wall in analysis.walls.walls:
        color = (0, 215, 255) if wall.wall_type == "outer" else (0, 140, 255)
        thickness = 3 if wall.wall_type == "outer" else 2
        cv2.line(overlay, wall.start, wall.end, color, thickness)
        
    # Render doors (green arc/circle)
    for door in analysis.doors_windows.doors:
        cv2.circle(overlay, door.center, door.radius, (0, 255, 0), 2)
        
    # Render windows (cyan bar)
    for window in analysis.doors_windows.windows:
        cv2.line(overlay, window.start, window.end, (255, 255, 0), window.thickness_px)
        
    # Render stairs (red box)
    for stair in analysis.doors_windows.stairs:
        x, y, w, h = stair.bounding_box
        cv2.rectangle(overlay, (x, y), (x+w, y+h), (0, 0, 255), 2)

    # Final blend with original based on alpha
    # If alpha is 1.0, we see the full overlay. If 0.0, we see original.
    final_render = cv2.addWeighted(overlay, alpha, original_bgr, 1 - alpha, 0)
    
    return final_render

def render_wall_extraction_debug(analysis: Any) -> np.ndarray:
    """Debug view 1: Shows raw structural mask with Hough lines overlaid."""
    if analysis.walls.structural_mask is None:
        return analysis.floor_plan.original_bgr.copy()
        
    # Convert mask to 3-channel for colored lines
    debug_bgr = cv2.cvtColor(analysis.walls.structural_mask, cv2.COLOR_GRAY2BGR)
    
    # Draw detected lines on top
    for wall in analysis.walls.walls:
        color = (0, 0, 255) if wall.wall_type == "outer" else (0, 255, 0)
        cv2.line(debug_bgr, wall.start, wall.end, color, 2)
        
    return debug_bgr

def render_room_segmentation_debug(analysis: Any) -> np.ndarray:
    """Debug view 2: Shows each room as a solid distinct color to verify segmentation boundaries."""
    h, w = analysis.floor_plan.height, analysis.floor_plan.width
    debug_bgr = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Palette for rooms
    palette = [
        (70, 130, 180), (34, 139, 34), (218, 112, 214), (210, 105, 30),
        (72, 209, 204), (255, 165, 0), (147, 112, 219), (220, 20, 60),
        (0, 206, 209), (154, 205, 50)
    ]
    
    for i, room in enumerate(analysis.rooms.rooms):
        color = palette[i % len(palette)]
        cv2.drawContours(debug_bgr, [room.polygon], -1, color, cv2.FILLED)
        
        # Room ID label at centroid
        cx, cy = room.centroid
        cv2.putText(debug_bgr, f"R{room.room_id}", (cx-10, cy), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    
    return debug_bgr

def render_topology_graph_debug(analysis: Any) -> np.ndarray:
    """Debug view 3: Shows nodes (room centroids) and edges (adjacencies) representing the graph."""
    debug_bgr = analysis.floor_plan.original_bgr.copy()
    
    # Dim the background to make the graph pop
    debug_bgr = cv2.addWeighted(debug_bgr, 0.3, np.zeros_like(debug_bgr), 0.7, 0)
    
    room_centers = {f"room_{r.room_id}": r.centroid for r in analysis.rooms.rooms}
    
    # Draw edges
    for edge in analysis.rooms.adjacency_graph.edges:
        src = edge["source"]
        tgt = edge["target"]
        if src in room_centers and tgt in room_centers:
            pt1 = room_centers[src]
            pt2 = room_centers[tgt]
            cv2.line(debug_bgr, pt1, pt2, (0, 255, 255), 2)
            
            # If gap center is known, draw it
            if "center" in edge:
                cx, cy = edge["center"]
                cv2.circle(debug_bgr, (cx, cy), 6, (0, 0, 255), -1)
                
    # Draw nodes
    for r in analysis.rooms.rooms:
        cv2.circle(debug_bgr, r.centroid, 8, (255, 0, 0), -1)
        cv2.putText(debug_bgr, f"R{r.room_id}", (r.centroid[0]+10, r.centroid[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
    return debug_bgr
