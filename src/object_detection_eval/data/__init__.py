"""Torch-free data tier: COCO ground truth loading, taxonomy resolution, images."""

from __future__ import annotations

from object_detection_eval.data.coco_gt import load_coco_gt
from object_detection_eval.data.image import ImageLoader
from object_detection_eval.data.taxonomy import (
    dedupe_merged_class_detections,
    identity_taxonomy_from_coco,
    remap_detections,
    resolve_taxonomy,
)

__all__ = [
    "ImageLoader",
    "dedupe_merged_class_detections",
    "identity_taxonomy_from_coco",
    "load_coco_gt",
    "remap_detections",
    "resolve_taxonomy",
]
