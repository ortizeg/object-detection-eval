"""Typed schemas for detections, annotations, and taxonomies."""

from __future__ import annotations

from object_detection_eval.schemas.annotation import AnnotationInfo, DetectionAnnotation
from object_detection_eval.schemas.detection import BoundingBox, Detection

__all__ = [
    "AnnotationInfo",
    "BoundingBox",
    "Detection",
    "DetectionAnnotation",
]
