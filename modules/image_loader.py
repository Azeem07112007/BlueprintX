"""
modules/image_loader.py
Blueprint-X Prototype - Milestone 1

Handles loading, validation, and basic preprocessing of floor plan images.
Designed to be extended in future milestones with additional format support.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, Any


SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg"}


class ImageLoadError(Exception):
    """Raised when an image cannot be loaded or validated."""
    pass


class FloorPlanImage:
    """
    Encapsulates a loaded floor plan image with its metadata.
    Future milestones can extend this with scale/DPI information.
    """

    def __init__(
        self,
        original_bgr: np.ndarray,
        file_path: Path,
    ):
        self.original_bgr: np.ndarray = original_bgr
        self.file_path: Path = file_path
        self.height: int = original_bgr.shape[0]
        self.width: int = original_bgr.shape[1]
        self.channels: int = original_bgr.shape[2] if original_bgr.ndim == 3 else 1

    @property
    def original_rgb(self) -> np.ndarray:
        """Return a copy in RGB order (suitable for Qt display)."""
        return cv2.cvtColor(self.original_bgr, cv2.COLOR_BGR2RGB)

    @property
    def grayscale(self) -> np.ndarray:
        """Return grayscale version for processing."""
        return cv2.cvtColor(self.original_bgr, cv2.COLOR_BGR2GRAY)

    @property
    def dimensions_text(self) -> str:
        return f"{self.width} × {self.height} px"

    def metadata(self) -> Dict[str, Any]:
        return {
            "file_name": self.file_path.name,
            "file_path": str(self.file_path),
            "width_px": self.width,
            "height_px": self.height,
            "channels": self.channels,
        }


def load_floor_plan(file_path: str) -> FloorPlanImage:
    """
    Load and validate a floor plan image from disk.

    Args:
        file_path: Absolute or relative path to the image file.

    Returns:
        FloorPlanImage instance with the loaded data.

    Raises:
        ImageLoadError: If the file cannot be read or is unsupported.
    """
    path = Path(file_path)

    if not path.exists():
        raise ImageLoadError(f"File not found: {file_path}")

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ImageLoadError(
            f"Unsupported format '{path.suffix}'. "
            f"Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    # Read with flag IMREAD_COLOR to always get 3-channel BGR
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)

    if image is None:
        raise ImageLoadError(
            f"OpenCV could not decode the image: {file_path}"
        )

    if image.shape[0] < 50 or image.shape[1] < 50:
        raise ImageLoadError(
            "Image is too small to be a valid floor plan (minimum 50×50 px)."
        )

    return FloorPlanImage(original_bgr=image, file_path=path)


def preprocess_for_detection(
    floor_plan: FloorPlanImage,
    max_dimension: int = 1200,
) -> Tuple[np.ndarray, float]:
    """
    Resize and normalise the image for wall/room detection.
    Keeps aspect ratio; returns the scale factor so coordinates can be
    mapped back to the original resolution.

    Args:
        floor_plan: A loaded FloorPlanImage.
        max_dimension: Maximum width or height of the output image.

    Returns:
        (preprocessed_bgr, scale_factor)
    """
    h, w = floor_plan.height, floor_plan.width
    scale = min(max_dimension / w, max_dimension / h, 1.0)

    if scale < 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(
            floor_plan.original_bgr,
            (new_w, new_h),
            interpolation=cv2.INTER_AREA,
        )
    else:
        resized = floor_plan.original_bgr.copy()
        scale = 1.0

    return resized, scale
