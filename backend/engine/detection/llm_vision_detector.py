"""
backend/engine/detection/llm_vision_detector.py
Blueprint-X — LLM Vision Blueprint Parser

Connects Multimodal Vision AI models (Gemini Vision API / OpenAI GPT-4o / Ollama Vision)
directly to Blueprint-X.

The Vision AI inspects the blueprint image, detects structural walls, doors, windows,
rooms, and furniture, and returns a canonical JSON structure that is parsed directly
into Blueprint-X 3D building objects.
"""
from __future__ import annotations

import os
import json
import base64
import logging
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np

from backend.models.dataclasses import (
    Building, Wall, Door, Window, Furniture, Stair, Column, Room, TextBlock
)

logger = logging.getLogger(__name__)


PROMPT_SYSTEM = """You are an expert architectural CAD AI. Analyze the provided floor plan image carefully.
Identify all architectural and structural elements.

Return ONLY a valid JSON object matching this schema:
{
  "scale": "1:100",
  "walls": [
    {"start": [x1, y1], "end": [x2, y2], "wall_type": "outer", "thickness_px": 12.0},
    {"start": [x1, y1], "end": [x2, y2], "wall_type": "internal", "thickness_px": 8.0}
  ],
  "windows": [
    {"center": [cx, cy], "width_px": w, "height_px": h}
  ],
  "doors": [
    {"center": [cx, cy], "width_px": w, "height_px": h, "angle_deg": 0.0}
  ],
  "rooms": [
    {"room_id": 1, "label": "Living Room", "polygon_points": [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]}
  ],
  "furniture": [
    {"class": "sofa", "sub_type": "l_sofa", "center": [cx, cy], "bbox": [x, y, w, h]},
    {"class": "dining_table", "sub_type": "dining_set", "center": [cx, cy], "bbox": [x, y, w, h]},
    {"class": "toilet", "sub_type": "flush_toilet", "center": [cx, cy], "bbox": [x, y, w, h]},
    {"class": "kitchen_counter", "sub_type": "countertop", "center": [cx, cy], "bbox": [x, y, w, h]}
  ],
  "stairs": [
    {"center": [cx, cy], "bbox": [x, y, w, h]}
  ]
}

Ensure coordinates [x, y] are in pixel units relative to image width and height.
Do not output markdown code blocks or explanations, only valid JSON.
"""


class LLMVisionDetector:
    """
    Vision AI engine supporting Gemini API, OpenAI GPT-4o, or local Vision endpoints.
    """

    def __init__(self, api_key: Optional[str] = None, provider: str = "gemini"):
        self.provider = provider.lower()
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("OPENAI_API_KEY")

    def is_available(self) -> bool:
        return bool(self.api_key)

    def run(self, bgr: np.ndarray) -> Optional[Building]:
        """
        Sends floor plan image to Vision LLM and returns parsed Building dataclass.
        """
        if not self.is_available():
            logger.info("[LLMVision] No API key found — skipping Vision LLM inference.")
            return None

        h, w = bgr.shape[:2]
        # Encode image as JPEG base64
        _, buffer = cv2.imencode('.jpg', bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        img_b64 = base64.b64encode(buffer).decode('utf-8')

        try:
            if self.provider == "gemini" or "GEMINI_API_KEY" in os.environ:
                json_str = self._call_gemini(img_b64)
            else:
                json_str = self._call_openai(img_b64)

            if not json_str:
                return None

            data = self._clean_json(json_str)
            return self._parse_building(data, (w, h))

        except Exception as e:
            logger.error(f"[LLMVision] Inference error: {e}")
            return None

    def _call_gemini(self, img_b64: str) -> Optional[str]:
        """Calls Google Gemini Vision API."""
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            
            image_part = {
                "mime_type": "image/jpeg",
                "data": img_b64
            }
            response = model.generate_content([PROMPT_SYSTEM, image_part])
            return response.text
        except Exception as e:
            logger.warning(f"[LLMVision] Gemini API call failed: {e}")
            return None

    def _call_openai(self, img_b64: str) -> Optional[str]:
        """Calls OpenAI GPT-4o Vision API."""
        try:
            import urllib.request
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            payload = {
                "model": "gpt-4o",
                "messages": [
                    {"role": "system", "content": PROMPT_SYSTEM},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
                        ]
                    }
                ],
                "max_tokens": 4096,
                "response_format": {"type": "json_object"}
            }
            req = urllib.request.Request("https://api.openai.com/v1/chat/completions",
                                         data=json.dumps(payload).encode('utf-8'),
                                         headers=headers)
            with urllib.request.urlopen(req) as resp:
                res = json.loads(resp.read().decode('utf-8'))
                return res["choices"][0]["message"]["content"]
        except Exception as e:
            logger.warning(f"[LLMVision] OpenAI API call failed: {e}")
            return None

    @staticmethod
    def _clean_json(raw: str) -> Dict[str, Any]:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            cleaned = "\n".join(lines[1:-1]) if len(lines) > 2 else cleaned
        return json.loads(cleaned)

    def _parse_building(self, data: Dict[str, Any], img_size: Tuple[int, int]) -> Building:
        w, h = img_size
        b = Building(image_size=(w, h), scale=data.get("scale", "1:100"))

        # Walls
        for w_data in data.get("walls", []):
            st = tuple(w_data.get("start", [0, 0]))
            en = tuple(w_data.get("end", [0, 0]))
            w_type = w_data.get("wall_type", "internal")
            t_px = float(w_data.get("thickness_px", 10.0))
            b.walls.append(Wall(start=st, end=en, wall_type=w_type, thickness_px=t_px))

        # Windows
        for win in data.get("windows", []):
            cx, cy = tuple(win.get("center", [0, 0]))
            ww = float(win.get("width_px", 30.0))
            wh = float(win.get("height_px", 10.0))
            b.windows.append(Window(
                center=(cx, cy), width_px=ww, height_px=wh,
                bbox=(int(cx - ww/2), int(cy - wh/2), int(ww), int(wh))
            ))

        # Doors
        for d in data.get("doors", []):
            cx, cy = tuple(d.get("center", [0, 0]))
            dw = float(d.get("width_px", 35.0))
            dh = float(d.get("height_px", 10.0))
            b.doors.append(Door(
                center=(cx, cy), width_px=dw, height_px=dh,
                bbox=(int(cx - dw/2), int(cy - dh/2), int(dw), int(dh)),
                source="ai"
            ))

        # Rooms
        for r in data.get("rooms", []):
            pts = [tuple(p) for p in r.get("polygon_points", [])]
            lbl = r.get("label", "Room")
            rid = int(r.get("room_id", 1))
            if len(pts) >= 3:
                cx = float(np.mean([p[0] for p in pts]))
                cy = float(np.mean([p[1] for p in pts]))
                b.rooms.append(Room(
                    room_id=rid, label=lbl, polygon_pts=pts, centroid=(cx, cy)
                ))

        # Furniture
        for f in data.get("furniture", []):
            cls_name = f.get("class", "furniture")
            sub = f.get("sub_type", "")
            bx = f.get("bbox")
            cx, cy = tuple(f.get("center", [0, 0]))
            b.furniture.append(Furniture(
                class_name=cls_name, sub_type=sub, center=(cx, cy),
                bbox=tuple(bx) if bx else (int(cx-15), int(cy-15), 30, 30)
            ))

        # Stairs
        for s in data.get("stairs", []):
            bx = s.get("bbox")
            cx, cy = tuple(s.get("center", [0, 0]))
            b.stairs.append(Stair(
                center=(cx, cy),
                bbox=tuple(bx) if bx else (int(cx-20), int(cy-20), 40, 40)
            ))

        logger.info(f"[LLMVision] Parsed {len(b.walls)} walls, {len(b.rooms)} rooms, {len(b.furniture)} furniture items via Vision AI.")
        return b
