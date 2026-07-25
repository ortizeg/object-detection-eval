"""Bounding-box helpers used by the evaluation harness.

Disposition (resolves 02-RESEARCH Open Question 2): only the two helpers the
harness actually consumes are ported, and both are pure scalar Python — no
torch enters the core import graph (CORE-08). The source module's torch-typed
converters are intentionally NOT ported:

- ``box_iou_1_to_n`` — redundant with the per-class NMS IoU that lives inside
  ``inference.postprocess`` (ported there in Plan 02-06); no other consumer.
- ``cxcywh_to_xyxy`` / ``xyxy_to_cxcywh`` — torch-only tensor converters with no
  Phase-2 consumer; the CORE-06 letterbox de-transform is new numpy code in
  ``inference.preprocess``.
"""

from __future__ import annotations


def pad_and_clamp_bbox(
    bbox_x: float,
    bbox_y: float,
    bbox_w: float,
    bbox_h: float,
    img_w: int,
    img_h: int,
    padding_ratio: float,
) -> tuple[int, int, int, int]:
    """Pad a bounding box by a ratio and clamp to image bounds.

    Expands the box by ``padding_ratio * dimension`` on each side, then clamps
    to ``[0, img_w) x [0, img_h)``.

    Args:
        bbox_x: Top-left x of the original bbox (COCO format).
        bbox_y: Top-left y of the original bbox (COCO format).
        bbox_w: Width of the original bbox.
        bbox_h: Height of the original bbox.
        img_w: Image width in pixels.
        img_h: Image height in pixels.
        padding_ratio: Fraction of bbox dimension to add as padding.

    Returns:
        Tuple of ``(x1, y1, x2, y2)`` as clamped integer pixel coordinates.
    """
    pad_w = bbox_w * padding_ratio
    pad_h = bbox_h * padding_ratio

    x1 = max(0, int(bbox_x - pad_w))
    y1 = max(0, int(bbox_y - pad_h))
    x2 = min(img_w, int(bbox_x + bbox_w + pad_w))
    y2 = min(img_h, int(bbox_y + bbox_h + pad_h))

    return x1, y1, x2, y2


def pixel_xyxy_to_normalized_xywh(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    """Convert pixel xyxy coordinates to normalised xywh (top-left origin).

    Args:
        x1: Left edge in pixels.
        y1: Top edge in pixels.
        x2: Right edge in pixels.
        y2: Bottom edge in pixels.
        image_width: Image width in pixels.
        image_height: Image height in pixels.

    Returns:
        Tuple of ``(x, y, w, h)`` normalised to ``[0, 1]``.
    """
    x = x1 / image_width
    y = y1 / image_height
    w = (x2 - x1) / image_width
    h = (y2 - y1) / image_height
    return x, y, w, h
