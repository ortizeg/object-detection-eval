"""Typed schemas for detections, annotations, and taxonomies."""

from __future__ import annotations

from object_detection_eval.schemas.annotation import AnnotationInfo, DetectionAnnotation
from object_detection_eval.schemas.detection import BoundingBox, Detection
from object_detection_eval.schemas.taxonomy import TaxonomySpec, load_taxonomy_spec

__all__ = [
    "AnnotationInfo",
    "BoundingBox",
    "Detection",
    "DetectionAnnotation",
    "TaxonomySpec",
    "load_taxonomy_spec",
]
