"""Scoring core: mAP, F1 threshold sweep, and PR-curve computation."""

from __future__ import annotations

from object_detection_eval.metrics.detection_map import (
    compute_metrics,
    detections_to_sv,
)

__all__ = [
    "compute_metrics",
    "detections_to_sv",
]
