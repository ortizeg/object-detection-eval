"""Reproducible evaluation harness for object detection networks on small datasets.

Public API assembled across the torch-free core tiers (CORE-07/CORE-08):
data loading and taxonomy remapping, scoring (mAP/F1/PR-curve/bootstrap),
and the 7 ONNX detectors behind `BaseInferencer`.
"""

from __future__ import annotations

from object_detection_eval.data.coco_gt import load_coco_gt
from object_detection_eval.data.taxonomy import (
    dedupe_merged_class_detections,
    remap_detections,
    resolve_taxonomy,
)
from object_detection_eval.inference.base import BaseInferencer
from object_detection_eval.inference.detectors import (
    DamoDetector,
    DeimDetector,
    RFDETRDetector,
    RTDETRv2Detector,
    RTMDetDetector,
    YOLO26Detector,
    YOLOXDetector,
)
from object_detection_eval.metrics.bootstrap import build_report, run_bootstrap
from object_detection_eval.metrics.curves import compute_pr_curve
from object_detection_eval.metrics.detection_map import compute_metrics
from object_detection_eval.metrics.prf1 import compute_prf1_at_threshold, find_best_threshold

__version__ = "0.1.0"

__all__ = [
    "BaseInferencer",
    "DamoDetector",
    "DeimDetector",
    "RFDETRDetector",
    "RTDETRv2Detector",
    "RTMDetDetector",
    "YOLO26Detector",
    "YOLOXDetector",
    "__version__",
    "build_report",
    "compute_metrics",
    "compute_pr_curve",
    "compute_prf1_at_threshold",
    "dedupe_merged_class_detections",
    "find_best_threshold",
    "load_coco_gt",
    "remap_detections",
    "resolve_taxonomy",
    "run_bootstrap",
]
