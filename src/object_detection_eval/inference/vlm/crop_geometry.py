"""Crop-window geometry shared by the two-pass crop pipelines.

Pure functions (numpy only, no model, no API client), so the crop wrappers can
share them without importing each other's backends.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from object_detection_eval.schemas.detection import BoundingBox, Detection


def crop_window(
    image: npt.NDArray[np.uint8],
    bbox: BoundingBox,
    w: int,
    h: int,
    pad: float,
    min_crop_frac: float,
) -> tuple[npt.NDArray[np.uint8], int, int, int, int]:
    """Return ``(crop, ox, oy, cw, ch)``: a padded window centred on ``bbox``.

    Padded to ``pad`` x the box size, floored to ``min_crop_frac`` of the shorter
    side, clamped to the frame. Pure geometry (no API/torch), so it is unit-
    testable without constructing an inferencer.
    """
    cx = (bbox.x + bbox.w / 2) * w
    cy = (bbox.y + bbox.h / 2) * h
    floor = min_crop_frac * min(w, h)
    half_w = max(bbox.w * w * pad, floor) / 2
    half_h = max(bbox.h * h * pad, floor) / 2

    x1 = max(0, round(cx - half_w))
    y1 = max(0, round(cy - half_h))
    x2 = min(w, round(cx + half_w))
    y2 = min(h, round(cy + half_h))
    return image[y1:y2, x1:x2], x1, y1, x2 - x1, y2 - y1


def remap_crop_detection(
    crop_det: Detection, class_id: int, ox: int, oy: int, cw: int, ch: int, w: int, h: int
) -> Detection:
    """Map a crop-space detection (normalised to the crop) back to the frame."""
    b = crop_det.bbox
    fx = min(max((ox + b.x * cw) / w, 0.0), 1.0)
    fy = min(max((oy + b.y * ch) / h, 0.0), 1.0)
    fw = min((b.w * cw) / w, 1.0 - fx)
    fh = min((b.h * ch) / h, 1.0 - fy)
    return Detection(
        bbox=BoundingBox(x=fx, y=fy, w=fw, h=fh),
        confidence=crop_det.confidence,
        class_id=class_id,
    )
